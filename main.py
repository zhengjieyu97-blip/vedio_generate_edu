from dify_plugin import Plugin, DifyPluginEnv

# Manim 渲染非常耗时，特别是涉及 LaTeX 编译和高清渲染时
# 将最大超时时间延长到 1 小时 (3600秒) 以防止被强制杀死
# 如果仍然超时，请考虑简化视频内容或使用更低的渲染质量
plugin = Plugin(DifyPluginEnv(MAX_REQUEST_TIMEOUT=3600))

if __name__ == '__main__':
    plugin.run()
