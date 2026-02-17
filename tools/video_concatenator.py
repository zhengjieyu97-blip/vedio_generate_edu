import os
import re
import shutil
import subprocess
import tempfile
import uuid
import json
from typing import Any, List
from collections.abc import Generator

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

class VideoConcatenator(Tool):
    """
    视频片段拼接器
    职责：接收多个视频片段（文件列表），合并成一个长视频。
    """

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        # 1. 获取输入参数
        # 注意：在 Dify 迭代器中，如果是 Array[File]，这里拿到的可能是列表对象
        # tool_parameters["video_files"] 可能是:
        # A. list of File objects (dify interneal) -> 需要特殊处理? 
        #    目前 Dify Tool 接口接收 File 类型时，通常传进来的是文件对象描述
        # B. 这里的场景比较特殊：manim_renderer 在迭代器里跑，返回的是 JSON Message 和 Blob Message。
        #    迭代器的最终输出如果是 Array[File]，传到这里就是 File 列表。
        #    如果是 Array[Object] (包含 file_path)，则需要解析。
        
        # 为了兼容性，我们假设输入可能是 "手动传入的文件列表" 或者 "基于 JSON 的路径列表"
        # 实际上 Dify 的 File 传递机制，Tool 拿到的通常是 file_identifier。
        # 但这里的 trick 是：manim_renderer 是本地运行的，它返回了 absolute file path。
        # 我们可以利用这个 path 做简单的本地合并 (因为是在同一个 worker 节点上运行)。
        
        videos_input = tool_parameters.get("video_files", [])
        
        if not videos_input:
            yield self.create_text_message("❌ 错误：未提供 video_files 列表")
            return

        # 2. 解析文件路径列表
        video_paths = []
        
        # 尝试解析输入 (兼容 Array[Object] 格式)
        if isinstance(videos_input, str):
            try:
                videos_input = json.loads(videos_input)
            except:
                pass
        
        # 处理迭代节点的输出格式：{"output": [...]}
        if isinstance(videos_input, dict) and "output" in videos_input:
            videos_input = videos_input["output"]

        # [New] Normalize input list: Parse JSON strings if present
        if isinstance(videos_input, list):
            normalized_list = []
            for item in videos_input:
                if isinstance(item, str) and item.strip().startswith("{"):
                    try:
                        normalized_list.append(json.loads(item))
                    except:
                        normalized_list.append(item)
                else:
                    normalized_list.append(item)
            videos_input = normalized_list
                
        if isinstance(videos_input, list):
            for item in videos_input:
                # 跳过错误项（status="error"）
                if isinstance(item, dict) and item.get("status") == "error":
                    continue
                    
                # 情况 A: item 是对象，包含 file_path (manim_renderer 的 json 输出)
                if isinstance(item, dict) and "file_path" in item:
                    video_paths.append(item["file_path"])
                # 情况 A2: item 是嵌套结构 {"result": {"file_path": "..."}} (结果归一化节点的输出)
                elif isinstance(item, dict) and "result" in item:
                    result_obj = item["result"]
                    if isinstance(result_obj, dict) and "file_path" in result_obj:
                        video_paths.append(result_obj["file_path"])
                # 情况 B: item 直接是路径字符串
                elif isinstance(item, str) and (item.endswith(".mp4") or os.path.exists(item)):
                    video_paths.append(item)
                # 情况 C: item 是 dify 的 file 结构 (复杂，暂时不处理，优先支持本地路径)
        
        # 过滤无效路径
        valid_paths = [p for p in video_paths if os.path.exists(p)]
        
        if not valid_paths:
            # 提供更详细的错误信息
            error_msg = f"❌ 找不到有效的视频文件。\n"
            error_msg += f"接收到的输入类型: {type(videos_input).__name__}\n"
            error_msg += f"解析出的路径数量: {len(video_paths)}\n"
            if video_paths:
                error_msg += f"无效路径示例: {video_paths[:3]}\n"
            error_msg += f"原始输入预览: {str(videos_input)[:300]}..."
            yield self.create_text_message(error_msg)
            return

        yield self.create_text_message(f"🔗 准备拼接 {len(valid_paths)} 个视频片段...")
        
        # 按 segment_id 排序（优先使用 segment_id，否则按文件修改时间）
        # 需要从原始输入中提取 segment_id 信息
        try:
            # 创建一个路径到 segment_id 的映射
            path_to_segment_id = {}
            if isinstance(videos_input, list):
                for item in videos_input:
                    if not isinstance(item, dict):
                        continue

                    # 情况 1：扁平结构，file_path 和 segment_id 在同一层
                    if "file_path" in item:
                        file_path = item["file_path"]
                        segment_id = item.get("segment_id")
                        if segment_id is not None:
                            path_to_segment_id[file_path] = int(segment_id)

                    # 情况 2：嵌套结构 {"result": {"file_path": "...", "segment_id": 1}}
                    elif "result" in item and isinstance(item["result"], dict):
                        result_obj = item["result"]
                        file_path = result_obj.get("file_path")
                        segment_id = result_obj.get("segment_id")
                        if file_path and segment_id is not None:
                            path_to_segment_id[file_path] = int(segment_id)
            
            # 排序函数：优先使用 segment_id，否则使用文件修改时间
            def get_sort_key(path):
                if path in path_to_segment_id:
                    return (0, path_to_segment_id[path])  # 有 segment_id 的排在前面
                else:
                    # 尝试从路径中提取 segment_id
                    match = re.search(r'segment[_\s]*(\d+)', path, re.IGNORECASE)
                    if match:
                        return (0, int(match.group(1)))
                    # 否则按文件修改时间排序（使用时间戳，确保唯一性）
                    return (1, os.path.getmtime(path))
            
            valid_paths.sort(key=get_sort_key)
        except Exception as e:
            # 如果排序失败，按文件修改时间排序
            try:
                valid_paths.sort(key=lambda x: os.path.getmtime(x))
            except:
                pass  # 如果还是失败，保持原顺序

        # 3. 创建工作目录
        task_id = str(uuid.uuid4())[:8]
        work_dir = tempfile.mkdtemp(prefix=f"manim_concat_{task_id}_")
        
        # 4. 生成 ffmpeg concat file
        list_file_path = os.path.join(work_dir, "concat_list.txt")
        output_filename = f"final_movie_{task_id}.mp4"
        output_path = os.path.join(work_dir, output_filename)
        
        with open(list_file_path, "w", encoding="utf-8") as f:
            for v_path in valid_paths:
                # FFmpeg concat file 格式: file '/path/to/file'
                safe_path = v_path.replace("\\", "/")
                f.write(f"file '{safe_path}'\n")
        
        # 5. 调用 FFmpeg
        cmd = [
            "ffmpeg", "-f", "concat", "-safe", "0",
            "-i", list_file_path, "-c", "copy", "-y", output_path
        ]
        
        try:
            # yield self.create_text_message(f"执行命令: {' '.join(cmd)}")
            subprocess.run(
                cmd, cwd=work_dir, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, 
                check=True, timeout=300
            )
            
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                
                # 读取最终文件
                with open(output_path, "rb") as f:
                    final_data = f.read()
                    
                # 1. 输出 JSON 格式的路径信息 (Step 5813)
                result_json = {
                    "status": "success",
                    "file_path": os.path.abspath(output_path),
                    "file_size_mb": round(file_size, 2)
                }
                yield self.create_text_message(json.dumps(result_json, ensure_ascii=False))

                # 2. 输出二进制文件 (Blob)
                yield self.create_blob_message(
                    blob=final_data,
                    meta={"mime_type": "video/mp4", "filename": output_filename}
                )
            else:
                yield self.create_text_message(json.dumps({"status": "error", "message": "FFmpeg 执行完成但未找到输出文件"}, ensure_ascii=False))
                
        except Exception as e:
            yield self.create_text_message(json.dumps({"status": "error", "message": f"拼接过程发生异常: {str(e)}"}, ensure_ascii=False))

