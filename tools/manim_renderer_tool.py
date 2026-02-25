from typing import Any, Dict
import os
import sys
import uuid
import json
import subprocess
import tempfile
import traceback
from collections.abc import Generator
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

class ManimRendererTool(Tool):
    """
    [Dify Plugin] Manim 视频渲染器
    职责：
    1. 接收 Manim Python 代码 (或 {"code":..., "segment_id":...} JSON 字符串)。
    2. 在临时目录或 output 目录生成 .py 脚本。
    3. 调用 manim 命令行进行渲染。
    4. 返回生成的视频文件绝对路径。
    """

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """
        执行渲染工具
        """
        # 1. 提取输入参数
        # 根据 yaml 定义，参数可能叫 'code' 或 'manim_code'
        # 这里假设 yaml 会传递 'code'，或者用户直接传了字符串
        input_data = tool_parameters.get("code", "")
        quality_str = tool_parameters.get("quality", "low")
        
        if not input_data:
            yield self.create_text_message(json.dumps({"status": "error", "message": "No Input Code provided."}, ensure_ascii=False))
            return

        # 2. 解析输入 (支持纯代码 或 JSON+ID 模式)
        manim_code = ""
        segment_id = None
        
        try:
            # 尝试作为 JSON 解析 (兼容 Step 5576 的架构建议)
            data_obj = json.loads(input_data)
            if isinstance(data_obj, dict):
                manim_code = data_obj.get("code", "")
                segment_id = data_obj.get("segment_id")
                # 如果 input_data 本身就是个 dict 而不是 str (Dify 特性)
            else:
                manim_code = input_data
        except:
            # 解析失败，说明不仅是 JSON，就是纯代码字符串
            manim_code = input_data

        # 再次确认 manim_code（如果 JSON 里没 code 字段，可能传错了）
        if not manim_code or not isinstance(manim_code, str):
             # 最后的 fallback：如果 input_data 本身就是代码
             if isinstance(input_data, str) and len(input_data) > 10:
                 manim_code = input_data
             else:
                yield self.create_text_message(json.dumps({"status": "error", "message": "Could not extract valid Manim code from input."}, ensure_ascii=False))
                return

        # 3. 确定文件名和目录
        # 使用插件目录下的 output 文件夹，方便用户查看 (Step 5597/5604)
        # current_file_path = os.path.abspath(__file__)
        # plugin_dir = os.path.dirname(os.path.dirname(current_file_path)) # 回退两级: tools -> vedio_generate_edu
        
        # 为了更稳健，直接使用当前工作目录或相对路径
        # 在 Dify 容器中，通常有固定工作目录。本地开发则为项目根目录。
        output_base_dir = os.path.join(os.getcwd(), "output")
        if not os.path.exists(output_base_dir):
            os.makedirs(output_base_dir, exist_ok=True)
            
        # 生成唯一文件名
        if segment_id is not None:
             file_basename = f"segment_{segment_id}"
        else:
             file_basename = f"segment_{uuid.uuid4().hex[:8]}"
        
        script_path = os.path.join(output_base_dir, f"{file_basename}.py")
        
        # 4. 写入 Python 脚本
        # [FIX] 在脚本开头注入 sys.path，确保 manim_smart_components 等本地模块可被找到
        # 这比依赖 PYTHONPATH 环境变量更可靠，因为路径直接写入了脚本本身
        tool_dir_for_prepend = os.path.dirname(os.path.abspath(__file__))
        plugin_root_for_prepend = os.path.dirname(tool_dir_for_prepend)
        path_prepend = (
            "import sys as _sys\n"
            f"_sys.path.insert(0, {repr(plugin_root_for_prepend)})\n"
            "\n"
            "# [WinFix] Monkey-patch shutil.move with retry logic to handle WinError 32 (file locked by FFmpeg)\n"
            "import shutil as _shutil, time as _time\n"
            "_orig_shutil_move = _shutil.move\n"
            "def _safe_shutil_move(src, dst, copy_function=_shutil.copy2):\n"
            "    for _retry in range(10):\n"
            "        try:\n"
            "            return _orig_shutil_move(src, dst, copy_function=copy_function)\n"
            "        except PermissionError:\n"
            "            _time.sleep(0.5)\n"
            "    return _orig_shutil_move(src, dst, copy_function=copy_function)\n"
            "_shutil.move = _safe_shutil_move\n"
            "\n"
        )
        try:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(path_prepend + manim_code)
        except Exception as e:
            yield self.create_text_message(json.dumps({"status": "error", "message": f"Error writing script file: {e}"}, ensure_ascii=False))
            return

        print(f"📜 Script saved to: {script_path}")
        print(f"🎬 Starting Manim rendering ({quality_str}) for {file_basename}...")

        # 5. 构建 Manim 命令
        # Mapping quality to flags and folder names
        quality_map = {
            "low": {"flag": "-ql", "folder": "480p15"},
            "medium": {"flag": "-qm", "folder": "720p30"},
            "high": {"flag": "-qh", "folder": "1080p60"},
            "4k": {"flag": "-qk", "folder": "2160p60"}
        }
        
        q_config = quality_map.get(quality_str, quality_map["low"])
        quality_flag = q_config["flag"]
        
        # 此处我们需要从代码中解析 Scene 类名，或者让 Manim 自动渲染第一个 Scene
        # 通常 Default 行为是渲染定义的 Scene。如果不指定 SceneName，Manim 可能提示选择。
        # 我们的生成器生成的类通常叫 GeneratedScene
        scene_name = "GeneratedScene" 
        
        media_output_name = file_basename # segment_1
        
        # [FIX] Use isolated media dir for each segment to prevent Windows file lock conflicts
        # during concurrent rendering and SVG generation.
        isolated_media_dir = os.path.join(output_base_dir, "media", f"seg_{segment_id or uuid.uuid4().hex[:8]}")
        if not os.path.exists(isolated_media_dir):
            os.makedirs(isolated_media_dir, exist_ok=True)

        # 5. 构建 Manim 命令
        # Mapping quality to flags and folder names
        quality_map = {
            "low": {"flag": "-ql", "folder": "480p15"},
            "medium": {"flag": "-qm", "folder": "720p30"},
            "high": {"flag": "-qh", "folder": "1080p60"},
            "4k": {"flag": "-qk", "folder": "2160p60"}
        }
        
        q_config = quality_map.get(quality_str, quality_map["low"])
        quality_flag = q_config["flag"]
        
        # 此处我们需要从代码中解析 Scene 类名，或者让 Manim 自动渲染第一个 Scene
        # 通常 Default 行为是渲染定义的 Scene。如果不指定 SceneName，Manim 可能提示选择。
        # 我们的生成器生成的类通常叫 GeneratedScene
        scene_name = "GeneratedScene" 
        
        media_output_name = file_basename # segment_1
        
        # [FIX] Use 'python -m manim' to ensure current environment settings are respected
        cmd = [
            sys.executable, "-m", "manim",
            quality_flag,       # Quality flag
            "--media_dir", isolated_media_dir, # [FIX] Use isolated path
            script_path,        # 脚本路径
            scene_name,         # Scene 类名
            "-o", media_output_name, # 指定输出视频文件名
            "--flush_cache"     # 避免缓存导致的问题
        ]
        
        # 6. 执行命令
        env = os.environ.copy()
        # 确保能找到项目根目录的依赖 (如 manim_smart_components)
        tool_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(tool_dir)

        # [FIX] 注入 pkg_resources 兼容垫片，解决 manim_voiceover 在 Python 3.12+ 下的
        # ModuleNotFoundError: No module named 'pkg_resources' 问题。
        # 做法：在临时目录写一个 pkg_resources.py 文件，并把该目录放到 PYTHONPATH 最前面，
        # 子进程 Python 启动时会优先找到这个文件作为 pkg_resources 模块。
        _shim_dir = tempfile.mkdtemp(prefix="dify_pkg_shim_")
        _shim_content = (
            "import importlib.metadata as _m\n"
            "class _Dist:\n"
            "    def __init__(self, n):\n"
            "        try: self.version = _m.version(n)\n"
            "        except: self.version = '0.0.0'\n"
            "require = lambda *a, **kw: None\n"
            "get_distribution = lambda n: _Dist(n)\n"
            "WorkingSet = type('WorkingSet', (), {})\n"
            "DistributionNotFound = Exception\n"
            "VersionConflict = Exception\n"
        )
        with open(os.path.join(_shim_dir, "pkg_resources.py"), "w") as _f:
            _f.write(_shim_content)

        env["PYTHONPATH"] = f"{_shim_dir}{os.pathsep}{project_root}{os.pathsep}{env.get('PYTHONPATH', '')}"

        # [FIX] Force UTF-8 for Manim processes on Windows
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        # [FIX] Inject User's local LaTeX (MikTeX) path into environment PATH
        # This is needed so that Manim subprocess can find 'xelatex.exe' for rendering math text.
        latex_bin_dir = r"C:\Users\小余\AppData\Local\Programs\MiKTeX\miktex\bin\x64"
        env["PATH"] = f"{latex_bin_dir}{os.pathsep}{env.get('PATH', '')}"


        try:
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                env=env,
                cwd=output_base_dir # 在 output 目录下运行
            )

            if process.returncode != 0:
                # [DEBUG] Save full logs to file for inspection
                log_path = os.path.join(output_base_dir, f"{file_basename}_render.log")
                with open(log_path, "w", encoding="utf-8") as log_f:
                    log_f.write(f"--- COMMAND ---\n{' '.join(cmd)}\n\n")
                    log_f.write(f"--- STDERR ---\n{process.stderr}\n\n")
                    log_f.write(f"--- STDOUT ---\n{process.stdout}\n")
                
                # Extract last 10 lines of stderr for concise error reporting
                stderr_last_lines = "\n".join(process.stderr.splitlines()[-10:])
                error_msg = f"Manim Execution Failed.\nLog saved to: {log_path}\nLast Error:\n{stderr_last_lines}"
                print(error_msg)
                yield self.create_text_message(json.dumps({"status": "error", "message": error_msg}, ensure_ascii=False))
                return
            
        except Exception as e:
            yield self.create_text_message(json.dumps({"status": "error", "message": f"Error executing manim command: {e}"}, ensure_ascii=False))
            return

        # 7. 定位输出视频
        # [FIX] Adaptive search path for isolated media
        quality_folder = q_config["folder"]
        
        video_relative_path = os.path.join(
            "videos", file_basename, quality_folder, f"{media_output_name}.mp4"
        )
        video_full_path = os.path.join(isolated_media_dir, video_relative_path)
        
        # 规范化路径分隔符
        video_full_path = os.path.abspath(video_full_path)

        if os.path.exists(video_full_path):
            print(f"✅ Render Success! Video at: {video_full_path}")
            # 返回 JSON 结果，包含路径和状态
            result_json = {
                "status": "success",
                "file_path": video_full_path,
                "segment_id": segment_id
            }
            yield self.create_text_message(json.dumps(result_json, ensure_ascii=False))
        else:
            # 尝试搜索一下，万一目录结构不对
            warning_msg = f"⚠️ Render finished but video not found at expected path: {video_full_path}"
            print(warning_msg)
            print(f"Manim Output: {process.stdout}")
            yield self.create_text_message(json.dumps({"status": "warning", "message": warning_msg, "manim_stdout": process.stdout}, ensure_ascii=False))
