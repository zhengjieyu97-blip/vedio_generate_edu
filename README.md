# Video Generate Edu (Dify Plugin)

**Version:** 1.0.0
**Author:** zhengjieyu97-blip
**Type:** Dify Tool Plugin

## 📖 简介 (Introduction)

`vedio_generate_edu` 是一个专为 Dify 平台设计的教育视频自动生成插件。它基于 **Manim (Mathematical Animation Engine)** 引擎，能够将文本脚本转化为高质量的数学/物理教学动画视频。

### 核心功能 (Core Features)
1.  **原子化同步 (Atomic Synchronization)**：
    *   实现语音(TTS)、字幕(Subtitle)、动画(Animation)的毫秒级精确同步。
2.  **智能布局 (Smart Layout)**：
    *   基于槽位（Slot）算法，自动管理画面元素位置，防止重叠。
3.  **Edu 专用组件库**:
    *   内置 `SmartTriangle`, `SmartVectorSystem`, `SmartStatSystem` 等高阶组件，通过简单参数即可生成复杂的几何/统计图表。
4.  **多格式输出**:
    *   支持返回视频文件路径 (JSON) 及直接下载视频文件 (Blob)。

---

## ⚠️ 环境依赖 (Prerequisites) - 非常重要！

本插件依赖本地 Python 环境执行 Manim 渲染。为了确保插件正常运行，**部署 Dify 插件的服务器/容器必须预先安装以下系统级依赖**：

### 1. LaTeX 环境 (必须)
Manim 使用 LaTeX 渲染数学公式。
*   **Windows**: 请安装 [MiKTeX](https://miktex.org/download) 或 TeX Live。
*   **Linux**: 安装 TeX Live (`sudo apt-get install texlive-full`)。
*   **验证**: 在终端输入 `latex --version` 确保能输出版本信息。

### 2. FFmpeg (必须)
用于视频合成与音频处理。
*   **Windows/Linux/macOS**: 安装 FFmpeg 并确保 `ffmpeg` 命令在系统 `PATH` 中。

### 3. Python 依赖
插件运行需要安装以下 Python 库（已包含在 `requirements.txt`）：
*   `manim`
*   `manim-voiceover`
*   `dify_plugin`

> **注意**: 如果您在 Docker 中运行 Dify，请确保使用包含上述依赖的自定义镜像，或者挂载本地环境。

---

## 🚀 安装与使用 (Installation & Usage)

### 1. 安装插件

**方法 ：从 GitHub 安装**
1.  进入 Dify -> **插件 (Plugins)** -> **安装插件**。
2.  选择 **从 GitHub 安装**。
3.  输入仓库地址：`https://github.com/YOUR_GITHUB_USERNAME/vedio_generate_edu` (请替换为实际地址)。

### 2. 工具介绍
本插件包含以下工具：

#### `segment_code_generator` (代码生成器)
*   **功能**: 将自然语言或结构化数据转换为 Manim Python 脚本。
*   **输入**: 视频片段描述 (JSON/Text)。
*   **输出**: Python 代码字符串。

#### `manim_renderer_tool` (渲染器)
*   **功能**: 执行 Manim 脚本，生成视频片段。
*   **输入**: Manim Python 代码。
*   **输出**: 视频文件路径 (JSON) + 视频文件流 (Blob)。

#### `video_concatenator` (视频拼接)
*   **功能**: 将多个视频片段合并为一个长视频。
*   **输入**: 视频文件路径列表。
*   **输出**: 合并后的视频文件。

---

## 🛠️ 常见问题 (Troubleshooting)

### 1. LaTeX 编译错误 / 找不到 latex 命令
*   **现象**: 报错 `RuntimeError: Latex failed to compile` 或 `FileNotFoundError: latex`。
*   **原因**: 服务器未安装 LaTeX 或 `PATH` 环境变量未配置。
*   **解决**: 
    1.  确认已安装 MiKTeX/TeX Live。
    2.  确保 `latex` 命令所在的 `bin` 目录已添加到系统环境变量 `PATH` 中。
    3.  重启 Dify 服务以加载新的环境变量。

### 2. 中文乱码 / UnicodeDecodeError
*   **现象**: Windows 下报错 `'gbk' codec can't decode...`。
*   **原因**: 控制台编码不匹配。
*   **解决**: 插件已内置 `os.environ["PYTHONUTF8"] = "1"` 修复此问题。如果仍有报错，请检查系统区域设置。

### 3. 字幕与语音不同步
*   **原因**: TTS 生成速度波动或网络延迟。
*   **解决**: 插件使用 `tracker.duration` 动态对齐动画时长，通常能自动修正。如果误差过大，请检查网络连接（如使用 OpenAI TTS）。

---

## 📄 开源协议 (License)
MIT License
