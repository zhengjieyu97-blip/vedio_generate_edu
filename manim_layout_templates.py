from manim import *
import numpy as np
from manim_smart_components import TopoGraph, GeoNode, GeoNodeType, SmartLayoutUtils

class ColorPalettes:
    ocean = ["#0077be", "#0096c7", "#48cae4", "#ade8f4", "#90e0ef"]
    forest = ["#2d6a4f", "#40916c", "#52b788", "#74c69d", "#95d5b2"]


class LayoutZones:
    """
    布局安全区域常量 (Layout Safe Zones)
    
    Manim 默认帧尺寸: 14.22 x 8.0 (config.frame_width x config.frame_height)
    
    布局分区示意图:
    ┌──────────────────────────────────────────┐  y = +4.0 (TOP)
    │  TITLE_ZONE (标题区)                     │
    │  高度: 1.0                               │
    ├──────────────────────────────────────────┤  y = +3.0 (TITLE_BOTTOM)
    │                                          │
    │  CONTENT_ZONE (主内容区)                 │
    │  包含: 几何图形、板书、公式              │
    │  高度: 约 5.0                            │
    │                                          │
    ├──────────────────────────────────────────┤  y = -2.5 (CONTENT_BOTTOM)
    │  CAPTION_ZONE (字幕安全区)               │
    │  高度: 1.5                               │
    │  用途: 口语化讲解词 (固定在底部)         │
    └──────────────────────────────────────────┘  y = -4.0 (BOTTOM)
    """
    
    # 纵向边界 (Y 坐标)
    FRAME_TOP = 4.0
    FRAME_BOTTOM = -4.0
    
    # 标题区
    TITLE_TOP_BUFF = 0.5                          # 标题距顶部边距
    TITLE_ZONE_HEIGHT = 1.0                       # 标题区高度
    TITLE_BOTTOM = FRAME_TOP - TITLE_ZONE_HEIGHT  # 标题区底边 = 3.0
    
    # 字幕安全区 (最重要!)
    CAPTION_ZONE_HEIGHT = 1.3                     # 字幕区高度
    CAPTION_TOP = FRAME_BOTTOM + CAPTION_ZONE_HEIGHT  # 字幕区顶边 = -2.7
    CAPTION_CENTER_Y = FRAME_BOTTOM + CAPTION_ZONE_HEIGHT / 2  # 字幕中心 = -3.35
    CAPTION_BOTTOM_BUFF = 0.4                     # 字幕距底部边距
    
    # 主内容区 (Content Zone)
    CONTENT_TOP = TITLE_BOTTOM - 0.3              # 内容区顶边 = 2.7
    CONTENT_BOTTOM = CAPTION_TOP + 0.2            # 内容区底边 = -2.5
    CONTENT_HEIGHT = CONTENT_TOP - CONTENT_BOTTOM # 可用高度 = 5.2
    CONTENT_CENTER_Y = (CONTENT_TOP + CONTENT_BOTTOM) / 2  # 内容中心 = 0.1
    
    # 横向边界 (X 坐标)
    FRAME_LEFT = -7.0
    FRAME_RIGHT = 7.0
    CONTENT_MAX_WIDTH = 13.0                      # 内容最大宽度
    
    # 分栏布局参数
    LEFT_PANEL_CENTER_X = -3.5                    # 左侧面板中心
    RIGHT_PANEL_CENTER_X = 3.5                    # 右侧面板中心
    PANEL_MAX_WIDTH = 6.0                         # 单侧面板最大宽度
    
    @classmethod
    def clamp_to_content_zone(cls, mobject, max_w=None, max_h=None, skip_shift_up=False):
        """
        将 Mobject 限制在内容安全区内。
        :param mobject: 目标对象
        :param max_w: 自定义最大宽度（可选）
        :param max_h: 自定义最大高度（可选）
        :param skip_shift_up: 是否跳过自动上移（用于 LayoutManager 管理的地方防止抖动）
        """
        m_w = max_w if max_w is not None else cls.CONTENT_MAX_WIDTH
        m_h = max_h if max_h is not None else cls.CONTENT_HEIGHT
        
        # 1. 尺寸约束
        scale_factor = 1.0
        if mobject.width > m_w:
            s = m_w / mobject.width
            mobject.scale(s)
            scale_factor *= s
        if mobject.height > m_h:
            s = m_h / mobject.height
            mobject.scale(s)
            scale_factor *= s
        
        # 2. 纵向边界修正
        if mobject.get_top()[1] > cls.CONTENT_TOP:
            mobject.shift(DOWN * (mobject.get_top()[1] - cls.CONTENT_TOP + 0.1))
        
        if not skip_shift_up:
            if mobject.get_bottom()[1] < cls.CONTENT_BOTTOM:
                mobject.shift(UP * (cls.CONTENT_BOTTOM - mobject.get_bottom()[1] + 0.1))
            
        # 3. 横向边界修正
        if mobject.get_left()[0] < cls.FRAME_LEFT + 0.3:
            mobject.shift(RIGHT * (cls.FRAME_LEFT + 0.3 - mobject.get_left()[0]))
        if mobject.get_right()[0] > cls.FRAME_RIGHT - 0.3:
            mobject.shift(LEFT * (mobject.get_right()[0] - cls.FRAME_RIGHT + 0.3))
        
        return mobject, scale_factor


class LayoutUtils:
    @staticmethod
    def bullet_list(items, width=10, buff=0.5, font_size=32):
        """Standardized arrow-bullet list"""
        group = VGroup()
        for item in items:
            if isinstance(item, str):
                line = MathTex(r"\text{" + item + r"}", font_size=font_size)
            else:
                line = item
                
            dot = MathTex("\\Rightarrow", color=GOLD_C, font_size=font_size).next_to(line, LEFT, buff=0.2)
            row = VGroup(dot, line)
            group.add(row)
            
        group.arrange(DOWN, buff=buff, aligned_edge=LEFT)
        if group.width > width:
            group.scale_to_fit_width(width)
        return group

# =============================================================================
# Layout Manager (Added from Design Doc)
# =============================================================================
class LayoutManager:
    def __init__(self, layout_type: str):
        self.layout_type = layout_type
        self.slots = self._init_slots()
        self.placed_mobs = {"visual": [], "board": []}
        
    def _init_slots(self) -> dict:
        Z = LayoutZones
        if self.layout_type == "split_screen":
            return {
                "visual": {"center_x": Z.LEFT_PANEL_CENTER_X, "top_y": Z.CONTENT_TOP - 0.3,
                           "max_w": Z.PANEL_MAX_WIDTH, "max_h": Z.CONTENT_HEIGHT * 0.85},
                "board":  {"center_x": Z.RIGHT_PANEL_CENTER_X, "top_y": Z.CONTENT_TOP - 0.3,
                           "max_w": Z.PANEL_MAX_WIDTH, "max_h": Z.CONTENT_HEIGHT * 0.85}
            }
        elif self.layout_type == "standard_vertical":
            return {
                "visual": {"center_x": 0, "top_y": Z.CONTENT_TOP - 0.3,
                           "max_w": Z.CONTENT_MAX_WIDTH, "max_h": Z.CONTENT_HEIGHT * 0.5},
                "board":  {"center_x": 0, "top_y": Z.CONTENT_CENTER_Y - 0.5,
                           "max_w": Z.CONTENT_MAX_WIDTH, "max_h": Z.CONTENT_HEIGHT * 0.4}
            }
        elif self.layout_type == "sandwich":
            return {
                "visual": {"center_x": 0, "top_y": Z.CONTENT_TOP - 0.3,
                           "max_w": Z.CONTENT_MAX_WIDTH, "max_h": Z.CONTENT_HEIGHT * 0.6},
                "board":  {"center_x": 0, "top_y": Z.CONTENT_BOTTOM + 1.2,
                           "max_w": Z.CONTENT_MAX_WIDTH, "max_h": 0.8, "stack_up": True}
            }
        elif self.layout_type == "grid":
            return {
                "visual": {"center_x": 0, "top_y": Z.CONTENT_CENTER_Y,
                           "max_w": Z.CONTENT_MAX_WIDTH, "max_h": Z.CONTENT_HEIGHT * 0.9,
                           "use_grid": True}
            }
        else:  # fallback
            return {
                "visual": {"center_x": 0, "top_y": Z.CONTENT_TOP - 0.3,
                           "max_w": Z.CONTENT_MAX_WIDTH, "max_h": Z.CONTENT_HEIGHT * 0.6},
                "board":  {"center_x": 0, "top_y": Z.CONTENT_BOTTOM + 1.5,
                           "max_w": Z.CONTENT_MAX_WIDTH, "max_h": Z.CONTENT_HEIGHT * 0.3}
            }

    def place(self, mob, slot_name: str) -> Mobject:
        if not mob: return mob
        slot = self.slots.get(slot_name)
        if slot_name not in self.placed_mobs:
             self.placed_mobs[slot_name] = []
        placed = self.placed_mobs[slot_name]
        
        # [Architecture Fix] Capture PRE-LAYOUT state of ALL relevant objects
        mobs_to_track = placed + [mob] if slot_name == "board" else [mob]
        states_before = {}
        for m in mobs_to_track:
            # We use width and center to detect ANY scaling/shifting from any source
            states_before[id(m)] = {
                "center": m.get_center().copy(),
                "width": m.width
            }

        # --- Layout Logic ---
        if slot_name == "board":
            # [Reflow Logic for Board]
            placed.append(mob)
            stack = VGroup(*placed)
            stack.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
            
            # Check if exceeds height
            if stack.height > slot["max_h"]:
                s_h = (slot["max_h"] / stack.height) * 0.95
                stack.scale(s_h)
                stack.arrange(DOWN, buff=0.3 * s_h, aligned_edge=LEFT)
            
            # Check width
            if stack.width > slot["max_w"]:
                s_w = slot["max_w"] / stack.width
                stack.scale(s_w)
            
            # Position the group
            stack.move_to([slot["center_x"], slot["top_y"] - stack.height/2, 0])
            
            # Final boundary guard
            LayoutZones.clamp_to_content_zone(stack, max_w=slot["max_w"], max_h=slot["max_h"], skip_shift_up=True)
            
        else:
            # [Legacy Logic for Visual/Others]
            if len(placed) == 0:
                target_pos = np.array([slot["center_x"], slot["top_y"] - mob.height/2, 0])
                mob.move_to(target_pos)
            else:
                mob.next_to(placed[-1], DOWN, buff=0.3)
            
            LayoutZones.clamp_to_content_zone(mob, max_w=slot["max_w"], max_h=slot["max_h"])
            placed.append(mob)

        # [Architecture Fix] Compare POST-LAYOUT state and Broadcast transformations
        for m in mobs_to_track:
            if hasattr(m, "mark_layout_shift") or hasattr(m, "mark_layout_scale"):
                old = states_before.get(id(m))
                if old:
                    # 1. Scale Propagation (derived from width change)
                    new_w = m.width
                    s_factor = new_w / old["width"] if old["width"] > 1e-6 else 1.0
                    if abs(s_factor - 1.0) > 1e-6:
                        m.mark_layout_scale(s_factor)
                    
                    # 2. Shift Propagation (derived from center change)
                    new_center = m.get_center()
                    shift_vec = new_center - old["center"]
                    if np.linalg.norm(shift_vec) > 1e-6:
                        m.mark_layout_shift(shift_vec)
            
        return mob
    
    def _clamp(self, mob):
        """调用全局安全区约束"""
        return LayoutZones.clamp_to_content_zone(mob)

    def compress_slot(self, slot_name: str):
        slot = self.slots.get(slot_name)
        if slot is None: return

        mobs = self.placed_mobs.get(slot_name, [])
        if len(mobs) < 2:
            return
        
        # 只要超过1个物体，就对整个插槽进行统一的比例压缩与重新对齐
        group = VGroup(*mobs)
        if group.height > slot["max_h"]:
            scale_factor = (slot["max_h"] / group.height) * 0.95
            group.scale(scale_factor)
            
            # 重新定位：确保整体顶部依然对齐插槽顶部
            group.move_to([slot["center_x"], slot["top_y"] - group.height/2, 0])
        
        # 最后对整个组进行一次硬性边界钳制
        LayoutZones.clamp_to_content_zone(group, max_w=slot["max_w"], max_h=slot["max_h"])


class SmartEduScene(Scene, TopoGraph):
    """
    [Master Component: Smart Edu Scene]
    Combining Layout templates with TopoGraph capabilities.
    All layout positions are anchored via GeoNodes.
    """
    def __init__(self, *args, **kwargs):
        Scene.__init__(self, *args, **kwargs)
        TopoGraph.__init__(self, **kwargs)
        Tex.set_default(tex_template=TexTemplateLibrary.ctex)
        self.camera.background_color = "#1a1a2e"

    # [Visual Fix] We no longer override add_node here.
    # The base class TopoGraph.add_node already has the logic:
    # 1. Skip self.add() if self is a Scene (to avoid ghosting).
    # 2. Add updaters (needed for animations like damping/following).
    # By removing this override, we restore the missing updaters.

    def add_smart(self, *mobjects):
        """Shortcut to add smart components and their internal nodes to the scene graph."""
        for mob in mobjects:
            if mob not in self.mobjects:
                self.add(mob)
            if hasattr(mob, "nodes"):
                for node in mob.nodes:
                    if node not in self.nodes:
                         self.add_node(node)
        return self

    def layout_setup(self, title, subtitle=None, color_theme=None):
        """Standard setup for a scene"""
        # Title - Master Class format with $$\text{...}$$
        t = Tex(r"$$\text{" + title + r"}$$", font_size=48, color=GOLD_C).to_edge(UP, buff=0.5)
        self.add(t)
        
        # Subtitle
        if subtitle:
            s = Tex(r"$$\text{" + subtitle + r"}$$", font_size=32, color=BLUE_B).next_to(t, DOWN, buff=0.2)
            self.add(s)
            
        return t

    # =========================================================================
    # 模式 1: 标准上下流 (Standard Vertical Flow)
    # 适用: 定义、定理、单个公式、列表
    # 【更新】使用 LayoutZones 确保不侵入字幕区
    # =========================================================================
    def layout_standard_vertical(self, title_text, content_mobs, subtitle_text=None):
        """
        构建标准上下布局
        :param title_text: 标题文本 (str)
        :param content_mobs: 内容对象列表 (list of Mobjects) 或 单个 VGroup
        :param subtitle_text: 副标题文本 (str, optional)
        :return: (title_group, content_group) 元组，用于后续动画
        """
        # 1. 标题锚定在标题区
        title = Tex(r"$$\text{" + title_text + r"}$$", font_size=48, color=GOLD_C)
        title.move_to([0, LayoutZones.FRAME_TOP - LayoutZones.TITLE_TOP_BUFF - 0.3, 0])
        
        header_group = VGroup(title)
        if subtitle_text:
            subtitle = Tex(r"$$\text{" + subtitle_text + r"}$$", font_size=32, color=BLUE_B)
            subtitle.next_to(title, DOWN, buff=0.2)
            header_group.add(subtitle)
            
        # 2. 内容打包
        if isinstance(content_mobs, VGroup):
            content = content_mobs
        else:
            content = VGroup(*content_mobs).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
            
        # 3. 定位在内容区中心
        content.move_to([0, LayoutZones.CONTENT_CENTER_Y, 0])
        
        # 4. 安全边界检查 - 使用集中化方法
        LayoutZones.clamp_to_content_zone(content, max_h=LayoutZones.CONTENT_HEIGHT * 0.85)
                
        return header_group, content

    # =========================================================================
    # 模式 2: 左右分栏布局 (Split Screen Flow)
    # 适用: 对比、左图右公式、联动展示
    # 【更新】使用 LayoutZones 确保不侵入字幕区
    # =========================================================================
    def layout_split_screen(self, title_text, left_mobs, right_mobs):
        """
        构建左右分栏布局
        :param title_text: 标题
        :param left_mobs: 左侧内容 (Mobject/VGroup) - 通常是几何图形
        :param right_mobs: 右侧内容 (Mobject/VGroup) - 通常是公式/板书
        :return: (title, left_group, right_group)
        """
        # 1. 标题
        title = Tex(r"$$\text{" + title_text + r"}$$", font_size=48, color=GOLD_C)
        title.move_to([0, LayoutZones.FRAME_TOP - LayoutZones.TITLE_TOP_BUFF - 0.3, 0])
        
        # 2. 左右容器化
        left_group = left_mobs if isinstance(left_mobs, VGroup) else VGroup(left_mobs)
        right_group = right_mobs if isinstance(right_mobs, VGroup) else VGroup(right_mobs)
        
        # 3. 使用 LayoutZones 定位
        left_group.move_to([LayoutZones.LEFT_PANEL_CENTER_X, LayoutZones.CONTENT_CENTER_Y, 0])
        right_group.move_to([LayoutZones.RIGHT_PANEL_CENTER_X, LayoutZones.CONTENT_CENTER_Y, 0])
        
        # 4. 安全边界检查 - 使用集中化方法
        for group in [left_group, right_group]:
            LayoutZones.clamp_to_content_zone(group, max_w=LayoutZones.PANEL_MAX_WIDTH, max_h=LayoutZones.CONTENT_HEIGHT * 0.85)

        return title, left_group, right_group

    # =========================================================================
    # 模式 3: 三段式/汉堡包布局 (Sandwich Layout)
    # 适用: 复杂作图、大图展示、定理证明
    # 【注意】此布局的 bottom_text 是板书说明，不是口语化字幕
    # 口语化字幕应使用 show_caption() 方法单独显示在最底部
    # =========================================================================
    def layout_sandwich(self, title_text, main_visual, bottom_text):
        """
        构建三段式布局
        :param title_text: 顶部标题
        :param main_visual: 中间视觉主体
        :param bottom_text: 底部板书说明 (非口语化字幕)
        :return: (title, main_visual, bottom_desc)
        """
        # 1. Top Bun (Title)
        title = Tex(r"$$\text{" + title_text + r"}$$", font_size=48, color=GOLD_C)
        title.move_to([0, LayoutZones.FRAME_TOP - LayoutZones.TITLE_TOP_BUFF - 0.3, 0])
        
        # 2. Bottom Bun (板书说明 - 在内容区底部，不在字幕区)
        desc = Tex(r"$$\text{" + bottom_text + r"}$$", font_size=32, color=WHITE)
        desc.move_to([0, LayoutZones.CONTENT_BOTTOM + 0.4, 0])  # 在内容区底部
        
        # 3. Main Visual - 在标题和底部说明之间
        main_visual.move_to([0, LayoutZones.CONTENT_CENTER_Y + 0.3, 0])
        
        # 4. 计算可用高度 (标题底部到底部说明顶部)
        available_h = (LayoutZones.CONTENT_TOP - 0.2) - (desc.get_top()[1] + 0.2)
        
        # 如果主体高度超过可用空间，进行缩放
        if main_visual.height > available_h:
            main_visual.scale_to_fit_height(available_h * 0.9)
        
        # 宽度检查
        if main_visual.width > LayoutZones.CONTENT_MAX_WIDTH:
            main_visual.scale_to_fit_width(LayoutZones.CONTENT_MAX_WIDTH)
            
        # 居中在可用区域
        center_y = ((LayoutZones.CONTENT_TOP - 0.2) + (desc.get_top()[1] + 0.2)) / 2
        main_visual.move_to([0, center_y, 0])
        
        return title, main_visual, desc

    # =========================================================================
    # 模式 4: 列表/阵列流 (Grid/Matrix Flow)
    # 适用: 总结、表格、公式组
    # 【更新】使用 LayoutZones 确保不侵入字幕区
    # =========================================================================
    def layout_grid(self, items, cols=2, title_text=None):
        """
        构建网格布局
        :param items: 元素列表
        :param cols: 列数
        :param title_text: 可选标题
        :return: (title, grid_group)
        """
        grid = VGroup(*items).arrange_in_grid(cols=cols, buff=(1.0, 0.5))
        
        title = None
        if title_text:
            title = Tex(r"$$\text{" + title_text + r"}$$", font_size=48, color=GOLD_C)
            title.move_to([0, LayoutZones.FRAME_TOP - LayoutZones.TITLE_TOP_BUFF - 0.3, 0])
        
        # 定位在内容区中心
        grid.move_to([0, LayoutZones.CONTENT_CENTER_Y, 0])
        
        # 尺寸与边界安全检查 - 使用集中化方法
        LayoutZones.clamp_to_content_zone(grid, max_h=LayoutZones.CONTENT_HEIGHT * 0.85)
            
        return title, grid

    # =========================================================================
    # 模式 5: 3D HUD 模式 (3D HUD Flow)
    # 注意: 需要在 ThreeDScene 中使用
    # =========================================================================
    def layout_3d_hud(self, scene_ref, title_text):
        """
        构建 3D 场景的 HUD 标题
        :param scene_ref: 当前场景实例 (self)
        :param title_text: 标题
        :return: title Mobject
        """
        title = Tex(r"$$\text{" + title_text + r"}$$", font_size=48, color=GOLD_C)
        title.move_to([0, LayoutZones.FRAME_TOP - LayoutZones.TITLE_TOP_BUFF - 0.3, 0])
        
        # 关键: 固定在相机帧上
        scene_ref.add_fixed_in_frame_mobjects(title)
        return title
    
    # =========================================================================
    # 字幕系统 (Caption System)
    # 用于显示口语化讲解词，固定在屏幕底部字幕安全区
    # =========================================================================
    
    def show_caption(self, text, font_size=32, color=WHITE):
        """
        在字幕安全区显示口语化讲解词
        :param text: 口语化文字
        :return: caption Mobject (Wrapped in a GeoNode if in SmartEduScene)
        """
        caption = Text(
            text, 
            font_size=font_size, 
            color=color,
            font="Microsoft YaHei"
        )
        
        if caption.width > LayoutZones.CONTENT_MAX_WIDTH - 1:
            caption.scale_to_fit_width(LayoutZones.CONTENT_MAX_WIDTH - 1)
        
        caption.move_to([0, LayoutZones.CAPTION_CENTER_Y, 0])
        
        # Add to graph as a label/UI node
        cap_node = GeoNode(caption, type=GeoNodeType.LABEL)
        self.add_node(cap_node)
        self.add(caption) # [Restored] Subtitles must appear instantly through update_caption
        
        return caption
    
    def update_caption(self, old_caption, new_text, font_size=32, color=WHITE):
        """
        更新字幕内容（用于原子化同步）
        字幕直接切换，无淡入淡出动画
        
        :param old_caption: 当前字幕 Mobject (可以是 None)
        :param new_text: 新的口语化文字
        :return: new_caption (新字幕对象)
        """
        # 直接移除旧字幕
        if old_caption is not None:
            self.remove(old_caption)
        
        # 创建并直接显示新字幕
        new_caption = self.show_caption(new_text, font_size, color)
        
        return new_caption


class EduLayoutScene(SmartEduScene):
    """Legacy compatibility alias for SmartEduScene"""
    pass


# =============================================================================
# 示例用法 (Reference Implementation & Test)
# =============================================================================
class DemoTemplateGallery(EduLayoutScene):
    def construct(self):
        super().construct() # 设置背景
        
        self.show_vertical_demo()
        self.show_split_demo()
        self.show_sandwich_demo()
        self.show_grid_demo()
        
    def show_vertical_demo(self):
        # 1. 准备内容 - 使用大师级 Tex 格式
        lines = [
            Tex(r"$$\text{第一行：定义明确}$$", font_size=32),
            Tex(r"$$\text{第二行：逻辑清晰}$$", font_size=32),
            Tex(r"$$E = mc^2$$", font_size=48)
        ]
        
        # 2. 调用模版
        header, content = self.layout_standard_vertical("模式一：标准上下流", lines, "The Standard Vertical Flow")
        
        # 3. 动画
        self.play(Write(header))
        self.play(FadeIn(content, shift=UP))
        self.wait(1)
        
        # 4. 清理 (FadeOut Group)
        self.play(FadeOut(header), FadeOut(content))
        
    def show_split_demo(self):
        # 1. 准备内容 - 使用大师级 Tex 格式
        left_text = Tex(r"$$\text{左侧：图形说明}$$" + "\n\n" + r"$$\text{这是一个圆。}$$", font_size=32)
        right_visual = Circle(radius=1.5, color=TEAL_C).set_fill(TEAL_C, 0.5)
        
        # 2. 调用模版
        title, left, right = self.layout_split_screen("模式二：左右分栏", left_text, right_visual)
        
        # 3. 动画
        self.play(Write(title))
        self.play(Write(left))
        self.play(DrawBorderThenFill(right))
        self.wait(1)
        
        # 4. 清理
        self.play(FadeOut(title), FadeOut(left), FadeOut(right))
        
    def show_sandwich_demo(self):
        # 1. 准备内容
        main_geo = VGroup(
            Square(side_length=3, color=BLUE),
            Circle(radius=1.5, color=RED)
        )
        desc = "底部说明：正方形内切圆"
        
        # 2. 调用模版
        title, main, bottom = self.layout_sandwich("模式三：三段式布局", main_geo, desc)
        
        # 3. 动画
        self.play(FadeIn(title, shift=DOWN))
        self.play(Create(main))
        self.play(Write(bottom))
        self.wait(1)
        
        # 4. 清理
        self.play(FadeOut(title), FadeOut(main), FadeOut(bottom))

    def show_grid_demo(self):
        # 1. 准备内容
        items = [Square(side_length=1, color=c).set_fill(c, 0.5) for c in [RED, GREEN, BLUE, YELLOW]]
        
        # 2. 调用模版
        title, grid = self.layout_grid(items, cols=2, title_text="模式四：网格布局")
        
        # 3. 动画
        self.play(Write(title))
        self.play(LaggedStart(*[FadeIn(x, scale=0.5) for x in grid], lag_ratio=0.1))
        self.wait(1)
        
        # 4. 清理
        self.play(FadeOut(title), FadeOut(grid))

class Demo3DTemplateGallery(ThreeDScene, EduLayoutScene):
    def construct(self):
        # 注意：ThreeDScene 没有 camera.background_color 属性，或者设置方式不同
        # 这里我们手动设置
        self.camera.background_color = "#1a1a2e"
        
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        
        self.show_3d_hud_demo()
        
    def show_3d_hud_demo(self):
        # 1. 3D 世界的内容
        axes = ThreeDAxes()
        sphere = Surface(
            lambda u, v: np.array([
                1.5 * np.cos(u) * np.cos(v),
                1.5 * np.cos(u) * np.sin(v),
                1.5 * np.sin(u)
            ]), v_range=[0, TAU], u_range=[-PI/2, PI/2],
            checkerboard_colors=[RED_D, RED_E], resolution=(16, 32)
        )
        
        # 2. 调用模版 (HUD Title)
        # 注意：混合继承时，方法都在 self 上
        # 但 layout_3d_hud 需要 scene_ref (即 self) 来 add_fixed_in_frame_mobjects
        title = self.layout_3d_hud(self, "模式五：3D HUD 模式 (Fixed Overlay)")
        
        # 3. 动画
        self.play(Write(title))
        self.play(Create(axes), Create(sphere))
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(3)
        self.stop_ambient_camera_rotation()
        
        # 4. 清理
        self.play(FadeOut(title), FadeOut(axes), FadeOut(sphere))
