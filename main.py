import sys

# Python 3.12+ 兼容性修复：确保 pkg_resources 可用（manim_voiceover 依赖它）
# 此行必须在其他任何 import 之前执行
try:
    import pkg_resources  # noqa: F401
    # 验证关键方法存在
    pkg_resources.get_distribution
except (ImportError, AttributeError):
    import types
    import importlib.metadata as _meta

    class _Distribution:
        def __init__(self, name):
            try:
                self.version = _meta.version(name)
            except Exception:
                self.version = "0.0.0"

    _shim = types.ModuleType("pkg_resources")
    _shim.require = lambda *a, **kw: None
    _shim.get_distribution = lambda name: _Distribution(name)
    _shim.WorkingSet = type("WorkingSet", (), {})
    _shim.DistributionNotFound = Exception
    _shim.VersionConflict = Exception
    sys.modules["pkg_resources"] = _shim


from dify_plugin import Plugin, DifyPluginEnv

# Manim 渲染非常耗时，特别是涉及 LaTeX 编译和高清渲染时
# 将最大超时时间延长到 1 小时 (3600秒) 以防止被强制杀死
# 如果仍然超时，请考虑简化视频内容或使用更低的渲染质量
plugin = Plugin(DifyPluginEnv(MAX_REQUEST_TIMEOUT=3600))

if __name__ == '__main__':
    plugin.run()

