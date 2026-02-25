from manim import *
import numpy as np
import re

# =====================================================================
# 📐 1. The Math Kernel (Pure Math, Stateless)
#    Implements the core geometric algorithms.
# =====================================================================

class SmartLayoutUtils:
    """
    [Math Kernel]
    Provides pure geometric calculation functions.
    Stateless.
    """
    @staticmethod
    def safe_tex(text, color=WHITE):
        """
        [LaTeX Safety Wrapper] 
        Ensures that:
        1. Unicode math symbols are converted to LaTeX equivalents.
        2. Chinese characters are wrapped in \\text{}.
        3. The entire expression is wrapped in $$...$$ for consistent math/text rendering.
        4. Returns a Tex mobject.
        """
        if not text: return Tex("")
        
        processed = str(text).strip()
        
        # [FIX] 0. Unicode Math Symbol Normalization
        # LLM may output Unicode superscripts/subscripts that XeLaTeX doesn't handle well
        unicode_replacements = {
            '²': '^2', '³': '^3', '⁴': '^4', '⁵': '^5',
            '₀': '_0', '₁': '_1', '₂': '_2', '₃': '_3',
            '×': '\\times ', '÷': '\\div ', '±': '\\pm ',
            '≤': '\\le ', '≥': '\\ge ', '≠': '\\ne ',
            '→': '\\to ', '←': '\\leftarrow ', '∞': '\\infty ',
            'π': '\\pi ', 'θ': '\\theta ', 'α': '\\alpha ', 'β': '\\beta ',
            '√': '\\sqrt', 'Δ': '\\Delta ', '∑': '\\sum ', '∫': '\\int ',
            '✓': '\\checkmark ', '✔': '\\checkmark ',
        }
        for uni, latex in unicode_replacements.items():
            processed = processed.replace(uni, latex)
        
        # 1. Detection of CJK (Chinese, Japanese, Korean)
        has_cjk = bool(re.search(r'[\u4e00-\u9fff]', processed))
        
        # 2. Add \\text{} if CJK is present but not already wrapped or in $$
        if has_cjk and "\\text{" not in processed:
            # Check if it's purely Chinese or mixed
            if not processed.startswith("$$"):
                # [Improved] Only wrap CJK chars in \text{} to allow math symbols to work
                processed = re.sub(r'([\u4e00-\u9fff]+)', r'\\text{\1}', processed)
        
        # 3. Final $$ wrapping for consistent math mode
        final_text = processed if processed.startswith("$$") else f"$${processed}$$"
        
        # [Critical Fix] Import-Time Safety for Non-Render Contexts
        # If Manim is imported in a bare environment (like Dify plugin analysis), 
        # the 'media/Tex' directory may not exist. We catch this to prevent crash.
        try:
            actual_color = SmartLayoutUtils.parse_color(color)
            return Tex(final_text, color=actual_color)
        except (FileNotFoundError, IndexError):
            # [IndexError Fix] Manim 0.18.x can crash with IndexError in _break_up_by_substrings
            # if the TeX compiler returns an empty/unexpected result for certain chars.
            # We return an empty VMobject to prevent the whole scene from crashing.
            return VMobject()

    @staticmethod
    def normalize(vec):
        norm = np.linalg.norm(vec)
        if norm < 1e-6: return RIGHT
        return vec / norm

    @staticmethod
    def get_centroid(points):
        return np.mean(np.array(points), axis=0)

    @staticmethod
    def get_bisector(prev_pt, curr_pt, next_pt):
        v1 = SmartLayoutUtils.normalize(prev_pt - curr_pt)
        v2 = SmartLayoutUtils.normalize(next_pt - curr_pt)
        bisector = v1 + v2
        if np.linalg.norm(bisector) < 1e-3:
            bisector = np.array([-v2[1], v2[0], 0])
        return SmartLayoutUtils.normalize(bisector)

    @staticmethod
    def resolve_label_dir(curr, bisector, centroid):
        test = curr + bisector * 0.1
        if np.linalg.norm(test - centroid) < np.linalg.norm(curr - centroid):
            return -bisector
        return bisector

    @staticmethod
    def get_line_intersection(p1, p2, p3, p4):
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        x3, y3 = p3[0], p3[1]
        x4, y4 = p4[0], p4[1]
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6: return None
        px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
        return np.array([px, py, 0])

    @staticmethod
    def project_point_to_line(point, line_start, line_end):
        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-6: return line_start
        line_unit = line_vec / line_len
        proj_len = np.dot(point_vec, line_unit)
        return line_start + line_unit * proj_len

    @staticmethod
    def get_angle_properties(v1, v2):
        """
        返回角度属性：是否直角，是否钝角
        """
        u1 = SmartLayoutUtils.normalize(v1)
        u2 = SmartLayoutUtils.normalize(v2)
        dot = np.dot(u1, u2)
        
        is_right = abs(dot) < 0.05
        is_obtuse = dot < 0
        return is_right, is_obtuse

    @staticmethod
    def get_direction_vec(direction):
        """
        [Helper] Converts direction strings (e.g. "UP") to Manim vectors.
        Safety: Defaults to UP if unknown.
        """
        if isinstance(direction, np.ndarray):
            return direction
        if not isinstance(direction, str):
            return UP
            
        d_map = {
            "UP": UP, "DOWN": DOWN, "LEFT": LEFT, "RIGHT": RIGHT,
            "UL": UL, "UR": UR, "DL": DL, "DR": DR,
            "IN": IN, "OUT": OUT
        }
        return d_map.get(direction.upper(), UP)

    @staticmethod
    def resolve_id(target, caller_frame=None):
        """
        [LLM Hallucination Guard] Resolves string references like '$$obj_ref$$' or 'obj' 
        into actual objects from the caller's environment.
        """
        if not isinstance(target, str):
            return target
            
        clean = re.sub(r'[\$]+', '', target).replace('_ref', '').strip()
        if not clean:
            return target
            
        import inspect
        # Walk up the stack to find the variable in user's script
        # Start from the caller's frame
        frame = caller_frame or inspect.currentframe().f_back
        
        # Limit depth to avoid performance issues (e.g., 10 frames)
        for _ in range(10):
            if not frame: break
            # 1. Check frame locals
            if clean in frame.f_locals:
                return frame.f_locals[clean]
            # 2. Check frame globals
            if clean in frame.f_globals:
                return frame.f_globals[clean]
            # Move up
            frame = frame.f_back

        # 3. Last stand: check system globals
        if clean in globals():
            return globals()[clean]

        return target

    # --- 🧠 Calculus & Curve Analysis Pack (Added for K-12 Coverage) ---

    @staticmethod
    def get_numerical_derivative(func, x, dx=1e-4):
        """
        [Calculus] 计算函数在 x 处的导数 (斜率).
        Uses central difference method for higher accuracy.
        """
        dy = func(x + dx) - func(x - dx)
        return dy / (2 * dx)

    @staticmethod
    def get_tangent_vector(func, x, dx=1e-4):
        """
        [Calculus] 获取切线方向向量 (Normalized).
        """
        slope = SmartLayoutUtils.get_numerical_derivative(func, x, dx)
        # Vector (1, slope)
        v = np.array([1, slope, 0])
        return SmartLayoutUtils.normalize(v)

    @staticmethod
    def project_point_to_curve(point, func, x_range, steps=20):
        """
        [Curve Analysis] 计算点到函数曲线的最近点 (Projection).
        Uses brute-force scan followed by local optimization (Gradient Descent).
        Must be efficient for 60fps.
        """
        # 1. Coarse Scan
        xs = np.linspace(x_range[0], x_range[1], steps)
        min_dist = float('inf')
        best_x = xs[0]
        
        px, py = point[0], point[1]
        
        for x_val in xs:
            y_val = func(x_val)
            dist = (x_val - px)**2 + (y_val - py)**2
            if dist < min_dist:
                min_dist = dist
                best_x = x_val
                
        # 2. Fine Tune (Newton's Method / Gradient Descent Lite)
        # Minimize D(t) = (t-px)^2 + (f(t)-py)^2
        # current t = best_x
        t = best_x
        lr = 0.1
        for _ in range(5):
            ft = func(t)
            f_prime = SmartLayoutUtils.get_numerical_derivative(func, t)
            
            # Derivative of Distance Squared
            dD = 2*(t - px) + 2*(ft - py)*f_prime
            
            # Geometry aware step
            t = t - lr * dD
            # Clamp to range
            t = max(x_range[0], min(x_range[1], t))
            
        return np.array([t, func(t), 0]), t

    # --- 📐 Circle & Geometry Pack ---

    @staticmethod
    def get_tangent_points_to_circle(external_point, circle_center, radius):
        """
        [Geometry] 过圆外一点作切线，求两个切点坐标.
        """
        # Shift to origin relative
        P = external_point - circle_center
        dist = np.linalg.norm(P)
        if dist < radius: return [] # Inside circle
        
        # Angle of vector P
        alpha = np.arctan2(P[1], P[0])
        # Angle offset for tangent: beta = acos(r/d)
        beta = np.arccos(radius / dist)
        
        t1 = alpha + beta
        t2 = alpha - beta
        
        p1 = circle_center + np.array([np.cos(t1), np.sin(t1), 0]) * radius
        p2 = circle_center + np.array([np.cos(t2), np.sin(t2), 0]) * radius
        return [p1, p2]

    # --- ⚖️ Physics Vector Pack ---

    @staticmethod
    def decompose_vector(target_vec, axis_vec):
        """
        [Physics] 向量正交分解.
        返回 (parallel_component, perpendicular_component)
        """
        axis_u = SmartLayoutUtils.normalize(axis_vec)
        parallel_mag = np.dot(target_vec, axis_u)
        vec_par = axis_u * parallel_mag
        vec_perp = target_vec - vec_par
        return vec_par, vec_perp

    # --- 📐 Middle School Geometry Pack (Added for Full Coverage) ---

    @staticmethod
    def get_segment_point(p1, p2, ratio=0.5):
        """
        [Middle School] 定比分点公式.
        ratio=0.5 -> Midpoint
        ratio=0.33 -> Trisection
        """
        return p1 + (p2 - p1) * ratio

    @staticmethod
    def rotate_point(point, center, angle):
        """
        [Middle School] 旋转变换 (Rotation).
        P' = R * (P - C) + C
        """
        # 2D Rotation matrix logic adapted for 3D points
        v = point - center
        c, s = np.cos(angle), np.sin(angle)
        # x' = x*c - y*s
        # y' = x*s + y*c
        new_x = v[0]*c - v[1]*s
        new_y = v[0]*s + v[1]*c
        return center + np.array([new_x, new_y, 0])

    @staticmethod
    def rotate_vector(vector, angle):
        """
        [Utility] 旋转向量 (Vector Rotation).
        """
        c, s = np.cos(angle), np.sin(angle)
        new_x = vector[0]*c - vector[1]*s
        new_y = vector[0]*s + vector[1]*c
        return np.array([new_x, new_y, 0])

    @staticmethod
    def parse_color(color):
        """
        [Utility] Robust color parser for LLM-generated strings.
        Maps common names to Manim constants or HEX values.
        """
        if not isinstance(color, str):
            return color # Already a Color object or something else
            
        color_upper = color.upper()
        
        # If already hex format, use it directly
        if color.startswith("#"):
            return color
            
        # Standard Manim colors (guaranteed via globals or library)
        standard_colors = {
            "WHITE": WHITE, "BLACK": BLACK, "RED": RED, 
            "GREEN": GREEN, "BLUE": BLUE, "YELLOW": YELLOW,
            "GOLD": GOLD, "PINK": PINK, "ORANGE": ORANGE, "PURPLE": PURPLE,
            "TEAL": TEAL, "GREY": GREY, "GRAY": GRAY
        }
        
        # Extended colors that may not exist in user's Manim version - use hex values
        hex_color_map = {
            "CYAN": "#00FFFF",
            "MAGENTA": "#FF00FF",
            "MAROON": "#800000",
            "NAVY": "#000080",
            "OLIVE": "#808000",
            "LIME": "#00FF00",
            "AQUA": "#00FFFF",
            "SILVER": "#C0C0C0",
            "BROWN": "#A52A2A",
            "VIOLET": "#EE82EE"
        }
        
        # 1. Try standard
        if color_upper in standard_colors:
            return standard_colors[color_upper]
        # 2. Try hex map
        if color_upper in hex_color_map:
            return hex_color_map[color_upper]
            
        # 3. Last ditch: Let Manim try, or default to WHITE
        return color # Return original string, hope Manim gets it or it's a hex
    
    @staticmethod
    def get_angle_degrees(v1, v2):
        """
        [Middle School] 计算两向量夹角 (Degrees).
        """
        u1 = SmartLayoutUtils.normalize(v1)
        u2 = SmartLayoutUtils.normalize(v2)
        dot = np.dot(u1, u2)
        # Clip to avoid numerical error outside [-1, 1]
        dot = np.clip(dot, -1.0, 1.0)
        angle_rad = np.arccos(dot)
        return angle_rad * (180 / np.pi)


# =====================================================================
# 2. The Universal Topo-Graph Engine
#    Strict implementation of Design Doc Chapter 3.
# =====================================================================

class GeoNodeType:
    VERTEX = "vertex"
    POINT = "vertex" # Alias for backward compatibility
    EDGE = "edge" 
    LABEL = "label"
    VIRTUAL = "virtual"
    SURFACE = "surface"


class GeoNode:
    """
    [The Atom]
    Represents a node in the dependency graph.
    It can wrap a Manim Mobject, or be a virtual point.
    """
    def __init__(self, value, type=GeoNodeType.VERTEX):
        # [Fix] Force float datatype for numpy arrays to prevent UFuncTypeError
        # when adding float vectors (like layout shifts) to integer coordinate arrays.
        if isinstance(value, np.ndarray):
            self.value = value.astype(np.float64)
        else:
            self.value = value
            
        self.type = type
        self.name = None # Optional unique ID within parent TopoGraph
        self.constraints = [] # List of (solver_func, dependencies)
        self.dependents = []
        self._live_data_lambda = None
        self.damping_factor = 0.2 # [Visual Optim] 0.0=Rigid, 1.0=Instant(Rigid), 0.1=Very Slow
        self.logical_pos = None # [NEW] Stores the user-intended coordinate (before snapping)
        
    def add_constraint(self, solver_func, parents):
        """
        solver_func(target_node, parent_nodes) -> None (in-place) or NewValue
        """
        self.constraints.append((solver_func, parents))
        for p in parents:
            if self not in p.dependents:
                p.dependents.append(self)

    def get_live_data(self):
        """Fetch current world coordinate/state"""
        if isinstance(self.value, Mobject):
            return self.value.get_center()
        # [Precision Fix] Support scalar types and existing numpy arrays
        if isinstance(self.value, (np.ndarray, float, int)):
            return self.value
        # If it's a virtual node tracking something else (functional), logic handled externally
        return self.value # Return raw value as ultimate fallback

    def set_value(self, new_val, local_offset=None):
        """
        Set the node's value, updating Mobject if applicable.
        [Architecture Upgrade] Added local_offset for Layout Locking.
        If local_offset is provided, new_val is treated as LOCAL coord.
        Target World Pos = new_val + local_offset
        """
        # [Bug Fix] Automatic Type Conversion: List -> Numpy
        if isinstance(new_val, (list, tuple)):
            target_val = np.array(new_val, dtype=np.float64)
        else:
            target_val = new_val
        
        # 1. Apply Offset if provided (Logic -> World mapping)
        # [Strict Fix] ONLY apply layout_offset to spatial nodes (VERTEX/POINT).
        # Virtual parameters should NOT be shifted by layout.
        if local_offset is not None and self.type in [GeoNodeType.VERTEX, GeoNodeType.POINT]:
             if isinstance(target_val, np.ndarray) and target_val.size == 3:
                 target_val = target_val + local_offset

        if isinstance(self.value, Mobject):
            if isinstance(target_val, np.ndarray) and target_val.shape == (3,):
                self.value.move_to(target_val)
            # Handle other Mobject updates if needed (e.g., color, scale)
        elif isinstance(self.value, np.ndarray):
            self.value[:] = target_val # Update in-place for numpy array
        else:
            self.value = target_val # For other types, direct assignment
            
    # Alias for symmetry with get_live_data
    set_live_data = set_value

    def compute_and_update(self, dt=0):
        """
        Re-evaluates constraints.
        [Updated] Now supports damping for smoother visual following.
        """
        # 1. Base Strategy: If I am live-bound (e.g. tracking a Dot), update my value from source
        if self._live_data_lambda:
            current_val = self._live_data_lambda()
            # Live data source is usually the "Authority", so we sync instantly?
            # Or do we dampen even the source tracking? Usually source is User/Animation driven.
            self.set_value(current_val) # For now, direct sync for live data
            return # Live data is authoritative, constraints don't apply

        # 2. Constraint Strategy
        for solver, parents in self.constraints:
            # Solve returns either None (in-place modification of self.value) or a new coordinate
            result = solver(self, parents)
            
            if result is not None:
                # [Visual Optim] Damping Logic
                # Only applying damping if result is numeric (coords) or compatible Mobject func?
                # Usually result is a coordinate for POINTS/VERTICES.
                
                # Check if we should dampen
                if self.damping_factor > 0 and self.damping_factor < 1.0 and isinstance(result, (np.ndarray, float, int)):
                     # Interpolate
                     current = self.get_live_data()
                     if current is not None:
                         # Linear Interpolation: curr + (targ - curr) * alpha
                         # Note: Manim's interpolate is handy but we can use numpy direct
                         new_val = current + (result - current) * self.damping_factor
                         self.set_value(new_val)
                     else:
                         self.set_value(result) # First frame
                else:
                    self.set_value(result)
                # Assuming only one constraint is active for position for simplicity
                # If multiple constraints, need a more complex resolution strategy (e.g., averaging, priority)
                break # Apply first constraint that returns a value and exit

class Constraint:
    """
    [The Rule]
    Standard constraint solvers library.
    """
    @staticmethod
    def label_repulsion(target_node, parent_nodes, buff=0.35):
        """
        Parents: [prev_vert, curr_vert, next_vert, centroid]
        """
        # Resolve parent data
        # Note: In a real graph, we might cache this, but here we fetch live
        prev, curr, next_pt, cent = [p.get_live_data() for p in parent_nodes]
        
        bisector = SmartLayoutUtils.get_bisector(prev, curr, next_pt)
        direction = SmartLayoutUtils.resolve_label_dir(curr, bisector, cent)
        
        return curr + direction * buff

    @staticmethod
    def connect_points(target_node, parent_nodes):
        def extract_coord(node):
            if node is None: return None
            data = node.get_live_data()
            if hasattr(data, 'get_center'): data = data.get_center()
            if data is None: return None
            # Handle list/tuple centers from manim
            d = np.array(data, dtype=float).flatten()
            if d.shape[0] < 3: d = np.pad(d, (0, 3 - d.shape[0]))
            return d

        if not parent_nodes: return target_node.value

        # Logic for endpoints
        if len(parent_nodes) == 2:
            p1 = extract_coord(parent_nodes[0])
            p2 = extract_coord(parent_nodes[1])
        elif len(parent_nodes) == 3:
            # Endpoint 1: p0, Endpoint 2: avg(p1, p2)
            p1 = extract_coord(parent_nodes[0])
            coords_p2 = [extract_coord(n) for n in parent_nodes[1:]]
            coords_p2 = [c for c in coords_p2 if c is not None]
            p2 = np.mean(coords_p2, axis=0) if coords_p2 else None
        elif len(parent_nodes) == 4:
            # avg(p0, p1) -> avg(p2, p3)
            coords_p1 = [extract_coord(n) for n in parent_nodes[:2]]
            coords_p1 = [c for c in coords_p1 if c is not None]
            p1 = np.mean(coords_p1, axis=0) if coords_p1 else None
            coords_p2 = [extract_coord(n) for n in parent_nodes[2:]]
            coords_p2 = [c for c in coords_p2 if c is not None]
            p2 = np.mean(coords_p2, axis=0) if coords_p2 else None
        else:
            p1 = extract_coord(parent_nodes[0])
            p2 = extract_coord(parent_nodes[1])

        # [Safety] Final validation
        if p1 is None or p2 is None:
            return target_node.value
        
        # [Safety] Avoid zero-length line (causes np.cross ValueError)
        if np.allclose(p1, p2):
            p2 = p1 + np.array([1e-4, 0, 0])
        
        if isinstance(target_node.value, Mobject):
            target_node.value.put_start_and_end_on(p1, p2)
        return target_node.value


class TopoGraph(VGroup):
    """
    [The Engine]
    Manages the update loop for the entire graph.
    """
    def __init__(self):
        super().__init__()
        self.nodes = [] # List of GeoNodes
        # [Architecture Upgrade] Local Coordinate Locking
        # Records the cumulative shift and scale applied by LayoutManager.
        self.layout_offset = np.array([0., 0., 0.]) 
        self.layout_scale = 1.0
        self.parent = None 
        self.node_map = {} # [Named Nodes] Map of string ID -> GeoNode

    def add(self, *mobjects):
        """
        [Infrastructure] Parent Tracking
        Sets a .parent attribute to children so TopoGraph can detect hierarchy.
        """
        for m in mobjects:
            if isinstance(m, Mobject):
                m.parent = self
        return super().add(*mobjects)

    def move_center(self, to, **kwargs):
        """
        [Compatibility] Alias for move_to, used by generated code.
        """
        self.move_to(np.array(to), **kwargs)
        return self

    def mark_layout_shift(self, vector):
        """
        Called by LayoutManager to register a world-space shift.
        Propagate this shift only to coordinate-based nodes (VERTEX).
        """
        self.layout_offset += vector
        for node in self.nodes:
            if node.type in [GeoNodeType.VERTEX, GeoNodeType.POINT]:
                if isinstance(node.value, np.ndarray) and node.value.shape == (3,):
                    node.value += vector

    def mark_layout_scale(self, factor):
        """
        Called by LayoutManager to register a scaling event.
        Propagate to all nodes containing 3D parameters to maintain model scale.
        """
        self.layout_scale *= factor
        for node in self.nodes:
            if isinstance(node.value, np.ndarray) and node.value.shape == (3,):
                node.value *= factor


    def add_node(self, node, name=None):
        self.nodes.append(node)
        if name:
            self.node_map[name] = node
        if isinstance(node.value, Mobject):
            # [Fix] Only add to VGroup if not a Scene and doesn't have a parent yet.
            # This prevents duplicate addition or adding to Scene when LM handles it.
            if not isinstance(self, Scene) and getattr(node.value, "parent", None) is None:
                self.add(node.value)
            
            # Standard updater
            def updater(m):
                for solver, parents in node.constraints:
                    res = solver(node, parents)
                    if res is not None:
                        # [Robust] Handle numpy vs vector results
                        if isinstance(res, np.ndarray) and res.shape == (3,):
                             m.move_to(res)
            node.value.add_updater(updater)

    # =====================================================================
    # [Infrastructure Upgrade] Universal Component Growth
    # =====================================================================
    def add_point(self, coords=ORIGIN, label_text="P", color=WHITE, radius=0.08):
        """
        Universal method to add a point growing on this component.
        Ensures local coordinate system alignment (parent=self).
        """
        pts_local = np.array(coords, dtype=np.float64)
        pt = _SmartPoint(coords=pts_local + self.layout_offset, label_text=label_text, color=color, radius=radius)
        if not isinstance(self, Scene):
            self.add(pt)
            
        # [Precision] Record the intended logical position for later line matching
        pt.center_node.logical_pos = pts_local
        
        # [Precision Fix] Register node name so add_angle and other semantic APIs can find it
        # Extract clean identifier from label_text (e.g. "T" from "$T$")
        name = str(label_text).replace("$", "").replace("\\", "").replace("{", "").replace("}", "").strip()
        
        # Integrate Point's Node into our Graph for global control
        self.add_node(pt.center_node, name=name if name else None)
        return pt

    def _resolve_node_or_coord(self, val):
        """
        [Precision Engine] Resolves input into a Node or absolute coord.
        Supports name lookup and proximity matching to handle snapped points.
        """
        # 1. Lookup by Name (String)
        if isinstance(val, str):
            node = self.node_map.get(val)
            if node: return node
            # Clean name fallback
            clean = val.replace("$", "").replace("\\", "").strip()
            node = self.node_map.get(clean)
            if node: return node

        # 2. Lookup by Proximity (Logical coordinate matching)
        if isinstance(val, (list, tuple, np.ndarray)):
            target_arr = np.array(val, dtype=np.float64)
            for node in self.nodes:
                if node.type in [GeoNodeType.VERTEX, GeoNodeType.POINT]:
                    # Priority Match: Use intended logical position
                    if node.logical_pos is not None:
                        if np.allclose(node.logical_pos, target_arr, atol=1e-4):
                            return node
                    # Secondary Match: World pos proximity (fallback)
                    else:
                        node_local = node.get_live_data() - self.layout_offset
                        if np.allclose(node_local, target_arr, atol=1e-3):
                            return node
        
        return val # Fallback (it's a raw coordinate)

    def add_line(self, start_node=ORIGIN, end_node=RIGHT, color=BLUE, stroke_width=4):
        """
        Universal method to add a line connecting two points or coords.
        Auto-resolves coordinates to existing nodes if they overlap.
        """
        # [Precision] Resolve inputs to existing nodes if they match by name or location
        s = self._resolve_node_or_coord(start_node)
        e = self._resolve_node_or_coord(end_node)
        
        # [Precision] Parse color string safely
        actual_color = SmartLayoutUtils.parse_color(color)
        
        # Apply layout offset ONLY to raw coordinates, Nodes handle it via transform
        s_input = s + self.layout_offset if isinstance(s, (list, tuple, np.ndarray)) else s
        e_input = e + self.layout_offset if isinstance(e, (list, tuple, np.ndarray)) else e

        line = _SmartLine(start_node=s_input, end_node=e_input, color=actual_color, stroke_width=stroke_width)
        if not isinstance(self, Scene):
            self.add(line)
        # Integrate Line's Edge Node into our Graph
        self.add_node(line.line_node)
        return line

    def set_node_local_data(self, node_or_index, data):
        """
        [Architecture Upgrade] Proxy method for LLM.
        Sets node data treating inputs as LOCAL logical coordinates.
        Automatically adds self.layout_offset (now spatial-aware).
        """
        target_node = None
        # 1. Look up by Index
        if isinstance(node_or_index, int):
            if 0 <= node_or_index < len(self.nodes):
                target_node = self.nodes[node_or_index]
        # 2. Look up by Name (Named Node Access)
        elif isinstance(node_or_index, str):
            target_node = self.node_map.get(node_or_index)
        # 3. Direct Node Reference
        elif isinstance(node_or_index, GeoNode):
            target_node = node_or_index
            
        if target_node:
            target_node.set_live_data(data, local_offset=self.layout_offset)
        else:
            # Silently fail or log for LLM debugging?
            pass

# =====================================================================
# 🧱 3. Atomic Primitives (The "Lego Bricks")
#    These are the CONCRETE implementations using the Abstract Engine.
# =====================================================================

class _SmartPoint(TopoGraph):
    """
    [Internal Atom: Point]
    A point that can carry a label. Use TopoGraph.add_point() instead.
    """
    def __init__(self, coords=ORIGIN, label_text=None, color=WHITE, radius=0.08, parent=None):
        super().__init__()
        
        real_coords = coords
        # [Precision] Parse color string safely
        actual_color = SmartLayoutUtils.parse_color(color)
        
        # 1. Physical Body (Manim Dot)
        self.dot = Dot(point=real_coords, color=actual_color, radius=radius)
        self.add(self.dot)
        
        # 2. Graph Node (The Brain)
        self.center_node = GeoNode(value=self.dot, type=GeoNodeType.VERTEX)
        self.add_node(self.center_node)
        
        # 3. Label System (Optional)
        if label_text:
            self.add_label(label_text, color)

    def add_label(self, text="P", color=WHITE, direction=UR, buff=0.35):
        # [Critical Fix] Use safe_tex to handle CJK and math mode consistently
        lbl = SmartLayoutUtils.safe_tex(text, color=color)
        l_node = GeoNode(value=lbl, type=GeoNodeType.LABEL)
        
        # Use Kernel for Robustness
        def smart_solver(target, parents):
            center = parents[0].get_live_data()
            # Normalize direction using Kernel
            dir_vec = SmartLayoutUtils.normalize(np.array(direction))
            return center + dir_vec * buff
            
        l_node.add_constraint(smart_solver, [self.center_node])
        self.add_node(l_node)
        
        # [Fix] Manually trigger solver once to position label BEFORE animation
        # This prevents the "jump" effect when updater kicks in later
        result = smart_solver(l_node, [self.center_node])
        if result is not None:
            lbl.move_to(result)
            
        return lbl # Return the label for play()


class _SmartLine(TopoGraph):
    """
    [Internal Atom: Line]
    A line that connects two points. Use TopoGraph.add_line() instead.
    """
    def __init__(self, start_node=ORIGIN, end_node=RIGHT, color=BLUE, stroke_width=4, parent=None):
        super().__init__()
        
        def map_coord(node):
            """[Compatibility Stub] Identity mapping, reserved for future coordinate transforms."""
            return node
        
        # 1. Unpack Inputs (Allow raw coords or objects with nodes)
        s_node = map_coord(start_node)
        if isinstance(s_node, (list, tuple, np.ndarray)):
            self.p1_dot = Dot(point=s_node, radius=0, fill_opacity=0)
            self.p1_node = GeoNode(value=self.p1_dot, type=GeoNodeType.VERTEX)
            self.add_node(self.p1_node)
        elif hasattr(s_node, "center_node"):
            self.p1_node = s_node.center_node
        elif hasattr(s_node, "get_live_data"): # It's a GeoNode directly
            self.p1_node = s_node
        else:
            raise ValueError(f"Start must be coord or a component with center_node, got {type(s_node)}")

        e_node = map_coord(end_node)
        if isinstance(e_node, (list, tuple, np.ndarray)):
            self.p2_dot = Dot(point=e_node, radius=0, fill_opacity=0)
            self.p2_node = GeoNode(value=self.p2_dot, type=GeoNodeType.VERTEX)
            self.add_node(self.p2_node)
        elif hasattr(e_node, "center_node"):
            self.p2_node = e_node.center_node
        elif hasattr(e_node, "get_live_data"): # It's a GeoNode directly
            self.p2_node = e_node
        else:
             raise ValueError(f"End must be coord or a component with center_node, got {type(e_node)}")

        # 2. Physical Body (Line)
        # Initialize at current positions
        start_pos = self.p1_node.get_live_data()
        end_pos = self.p2_node.get_live_data()
        
        # [Precision] Parse color string safely
        actual_color = SmartLayoutUtils.parse_color(color)
        
        self.line = Line(start_pos, end_pos, color=actual_color, stroke_width=stroke_width)
        self.add(self.line)
        
        # 3. Graph Node for Line
        self.line_node = GeoNode(value=self.line, type=GeoNodeType.EDGE)
        
        # [Fix] Add aliases to be compatible with other components
        self.p1 = self.p1_node
        self.p2 = self.p2_node
        
        # Constraint: "Connects P1 and P2"
        # Since p1_node and p2_node might belong to OTHER SmartGraphs
        def connection_solver(target, parents):
            return Constraint.connect_points(target, parents)
            
        self.line_node.add_constraint(connection_solver, [self.p1_node, self.p2_node])
        self.add_node(self.line_node)

    def add_length_label(self, text="d", buff=0.2, color=WHITE):
        lbl = SmartLayoutUtils.safe_tex(text, color=color)
        lbl.scale(0.6)  # Match MathTex font_size=24 roughly
        # [Fix] Removed set_opacity(0) - Write() doesn't restore opacity
        l_node = GeoNode(value=lbl, type=GeoNodeType.LABEL)
        
        def mid_perp_solver(target, parents):
            p1 = parents[0].get_live_data()
            p2 = parents[1].get_live_data()
            mid = (p1 + p2) / 2
            v = p2 - p1
            # Perpendicular logic from Kernel
            perp = np.array([-v[1], v[0], 0])
            perp = SmartLayoutUtils.normalize(perp)
            if perp[1] < 0: perp = -perp # Upward bias
            return mid + perp * buff
            
        l_node.add_constraint(mid_perp_solver, [self.p1_node, self.p2_node])
        self.add_node(l_node)
        self.add(lbl) # Add to visual graph
        return lbl


# =====================================================================
# 🏭 4. Smart Geometry Factory (The Standard Library)
# =====================================================================

class SmartPolygon(TopoGraph):
    """
    [Component: Universal Polygon]
    Composed of _SmartPoints and _SmartLines.
    Dynamic topology.
    """
    def __init__(self, *args, color=BLUE, stroke_width=4):
        super().__init__()
        self.vertices = [] # List of internal points
        self.edges = []    # List of internal lines
        
        all_pts = args
        
        # 1. Create Vertices
        for i, p in enumerate(all_pts):
            if hasattr(p, "center_node"):
                pt = p
                # [FIX]: Parent Stealing Prevention
                # Do NOT self.add(pt) here. 
                # If pt belongs to another object (like a Triangle), adding it here would steal it.
                # We only assume ownership of the NODE graph.
            else:
                pt = _SmartPoint(p, color=color, radius=0.01) # Small joint
                self.add(pt) # We created it, we own it.
            
            self.vertices.append(pt)
            self.add_node(pt.center_node, name=f"v{i}")
            
        # 2. Create Edges (Loop)
        self.edge_group = VGroup() # [Atomic Sync] 聚合所有边
        n = len(self.vertices)
        for i in range(n):
            curr = self.vertices[i]
            next_pt = self.vertices[(i+1)%n]
            
            edge = _SmartLine(curr, next_pt, color=color, stroke_width=stroke_width)
            self.edges.append(edge)
            # Add line to edge_group but DO NOT add the edge wrapper to self
            # This avoids parent-child conflict.
            self.edge_group.add(edge.line) 
            
        self.add(self.edge_group)

    def move_vertex(self, index_or_name, to):
        """
        [Semantic API] Moves a vertex to a new position.
        Automatically updates connected edges and labels.
        """
        # If it's a raw number, convert to our named node ID 'v0', 'v1', etc.
        target = index_or_name
        if isinstance(index_or_name, int):
            target = f"v{index_or_name}"
            
        self.set_node_local_data(target, to)
        # Return self for chaining? 
        return self

    def add_vertex_labels(self, labels=["A", "B", "C", "D", "E", "F"], buff=0.35, color=WHITE):
        """
        Smart labeling using Repulsion (Bisector) logic.
        """
        poly_centroid = GeoNode(None)
        # Dynamic centroid calculation
        poly_centroid.get_live_data = lambda: SmartLayoutUtils.get_centroid([v.center_node.get_live_data() for v in self.vertices])
        self.nodes.append(poly_centroid) # Virtual

        n = len(self.vertices)
        new_labels = VGroup()
        for i, text in enumerate(labels):
            if i >= n: break
            
            target_pt = self.vertices[i]
            target_pt.label_text = text # [New] Record the label text for semantic side resolution
            prev_pt = self.vertices[(i-1)%n]
            # [FIX]: Naming conflict resolution (was 'next')
            post_pt = self.vertices[(i+1)%n]
            
            # Use safe_tex for consistent CJK handling
            lbl = SmartLayoutUtils.safe_tex(text, color=color)
            # [Fix] Removed set_opacity(0) - Write() doesn't restore opacity
            new_labels.add(lbl)
            
            l_node = GeoNode(lbl, type=GeoNodeType.LABEL)
            
            parents = [
                prev_pt.center_node, 
                target_pt.center_node, 
                post_pt.center_node, 
                poly_centroid
            ]
            
            def poly_label_solver(target, rents):
                return Constraint.label_repulsion(target, rents, buff)
                
            l_node.add_constraint(poly_label_solver, parents)
            self.add_node(l_node)
            self.add(lbl) # Add to visual graph
            
            # [Fix] Manually trigger solver once to position label BEFORE animation
            result = poly_label_solver(l_node, parents)
            if result is not None:
                lbl.move_to(result)

        return new_labels

    def add_side_label(self, side_index=0, text="a", buff=0.3, color=WHITE):
        """
        [Master Feature] 给指定边添加长度/名称标签。
        side_index: 支持整数索引 (0, 1...) 或语义索引如 'e0', 'v0v1'。
        """
        idx = side_index
        # [Resolution] Resolve semantic side index
        if isinstance(side_index, str):
            if side_index.startswith('e'):
                try: idx = int(side_index[1:])
                except: pass
            elif 'v' in side_index:
                # Handle 'v0v1' format
                pts = side_index.split('v')
                if len(pts) >= 2:
                    try:
                        v_idx = int(pts[1])
                        idx = v_idx
                    except: pass
            elif len(side_index) >= 2 and side_index.isalpha():
                # [Semantic Fix] Handle 'AB', 'BC' etc.
                # Logic: Find vertex index for each char, then find common edge
                L1, L2 = side_index[0], side_index[1]
                v1_idx, v2_idx = -1, -1
                for i, v in enumerate(self.vertices):
                    v_label = getattr(v, 'label_text', '').strip('$') # Support both 'A' and '$$A$$'
                    if v_label == L1: v1_idx = i
                    if v_label == L2: v2_idx = i
                
                if v1_idx != -1 and v2_idx != -1:
                    # In a simple polygon, edge i connects v_i and v_{i+1}
                    n = len(self.vertices)
                    if (v1_idx + 1) % n == v2_idx: idx = v1_idx
                    elif (v2_idx + 1) % n == v1_idx: idx = v2_idx
        
        # [Safety] Ensure idx is int to prevent TypeError: '<=' not supported between 'int' and 'str'
        try:
            if not isinstance(idx, int):
                idx = int(idx)
        except:
            idx = -1 # Invalidated
        
        if 0 <= idx < len(self.edges):
            edge = self.edges[idx]
            
            # 1. Create the label (Write animation support)
            lbl = SmartLayoutUtils.safe_tex(text, color=color)
            # [Fix] Removed set_opacity(0) - Write() doesn't restore opacity
            l_node = GeoNode(lbl, type=GeoNodeType.LABEL)
            
            # 2. Centroid tracking
            poly_centroid = GeoNode(None)
            poly_centroid.get_live_data = lambda: SmartLayoutUtils.get_centroid([v.center_node.get_live_data() for v in self.vertices])
            self.nodes.append(poly_centroid) # Virtual
            
            # 3. Solver: Centrifugal repulsion
            def side_label_solver(target, parents):
                p1, p2, cent = [p.get_live_data() for p in parents]
                mid = (p1 + p2) / 2
                v = p2 - p1
                # Standard perpendicular
                perp = np.array([-v[1], v[0], 0])
                perp = SmartLayoutUtils.normalize(perp)
                
                # Check direction vs centroid
                pos1 = mid + perp * buff
                pos2 = mid - perp * buff
                dist1 = np.linalg.norm(pos1 - cent)
                dist2 = np.linalg.norm(pos2 - cent)
                
                final_dir = perp if dist1 > dist2 else -perp
                return mid + final_dir * buff
            
            l_node.add_constraint(side_label_solver, [edge.p1_node, edge.p2_node, poly_centroid])
            self.add_node(l_node)
            self.add(lbl)
            
            # [Fix] Manually trigger solver once to position label BEFORE animation
            result = side_label_solver(l_node, [edge.p1_node, edge.p2_node, poly_centroid])
            if result is not None:
                lbl.move_to(result)
            
            return lbl
        # [Safety Fix] Index out of range: return empty VMobject instead of None
        # This prevents Write(None) crashes when LLM uses invalid edge indices
        print(f"[Warning] add_side_label: side_index '{side_index}' (resolved to {idx}) is out of range (0-{len(self.edges)-1}). Returning empty object.")
        return VMobject()

    def add_angle(self, index=0, label=None, radius=0.5, color=YELLOW, other_angle=False):
        """
        [Incremental API] Adds an angle mark at the specified vertex index.
        index: 支持整数索引 (0, 1...) 或语义索引如 'v0', 'A'。
        """
        idx = index
        if isinstance(index, str):
            # Resolve via topograph nodes
            node = self._resolve_node_or_coord(index)
            for i, v in enumerate(self.vertices):
                if v.center_node == node:
                    idx = i
                    break
                    
        n = len(self.vertices)
        if not (isinstance(idx, int) and 0 <= idx < n):
            return None
            
        # 1. Identify O, A, B
        # O = current vertex
        # A = next vertex (Counter-Clockwise usually start ray?)
        # B = prev vertex 
        # Standard Manim Polygon vertices are usually CCW.
        # Angle(L1, L2) draws from L1 to L2.
        # To get the interior angle, we usually want (Right, Up) -> CCW.
        # Let's try: O=curr, A=next, B=prev.
        
        curr_pt = self.vertices[idx]
        next_pt = self.vertices[(idx + 1) % n]
        prev_pt = self.vertices[(idx - 1 + n) % n]
        
        # 2. Create SmartAngle
        # We use the internal vertices directly
        angle_obj = SmartAngle(
            O=curr_pt, 
            A=next_pt, 
            B=prev_pt, 
            radius=radius, 
            color=color, 
            other_angle=other_angle
        )
        
        # 3. Add Label if requested
        if label:
            angle_obj.add_label(text=label, radius_buff=1.4, color=color)
            
        # 4. Register
        self.add(angle_obj)
        # Note: We don't need to add_node(angle_obj) because SmartAngle is a TopoGraph 
        # and manages its own nodes. But we might want to track it?
        # For now, just visual addition and return is enough for "Incremental" usage.
        
        return angle_obj

    def add_label(self, text="S", buff=0.1, color=WHITE):
        """
        Adds a label to the polygon's centroid.
        """
        lbl = SmartLayoutUtils.safe_tex(text, color=color)
        l_node = GeoNode(lbl, type=GeoNodeType.LABEL)
        
        # Centroid node
        poly_centroid = GeoNode(None)
        poly_centroid.get_live_data = lambda: SmartLayoutUtils.get_centroid([v.center_node.get_live_data() for v in self.vertices])
        self.nodes.append(poly_centroid) 
        
        def center_solver(target, rents):
            c = rents[0].get_live_data()
            return c  # Centered at centroid
            
        l_node.add_constraint(center_solver, [poly_centroid])
        self.add_node(l_node)
        self.add(lbl)
        
        # [Fix] Immediate update
        res = center_solver(l_node, [poly_centroid])
        if res is not None: lbl.move_to(res)
        
        return lbl


class SmartTriangle(SmartPolygon):
    """
    [Component: SmartTriangle]
    Inherits generic logic, adds Triangle-specific features.
    """
    def __init__(self, A=None, B=None, C=None, color=BLUE, stroke_width=4):
        if A is None: A = [-1, -1, 0]
        if B is None: B = [1, -1, 0]
        if C is None: C = [0, 1, 0]
        super().__init__(A, B, C, color=color, stroke_width=stroke_width)
        # Register semantic aliases
        self.node_map["A"] = self.vertices[0].center_node
        self.node_map["B"] = self.vertices[1].center_node
        self.node_map["C"] = self.vertices[2].center_node

    def add_angle_mark(self, index=0, label="\\alpha", radius=0.5, color=YELLOW):
        """
        Adds a semantic angle mark (arc or right-angle).
        index: 支持语义标识 'A', 'B', 'C' 或 'v0', 'v1', 'v2'。
        """
        idx = index
        if isinstance(index, str):
            node = self._resolve_node_or_coord(index)
            for i, v in enumerate(self.vertices):
                if v.center_node == node:
                    idx = i
                    break
                    
        # 1. Identify nodes
        n = 3
        if not (isinstance(idx, int) and 0 <= idx < n):
             return None
             
        curr = self.vertices[idx]
        prev = self.vertices[(idx-1)%n]
        # [FIX]: Avoid 'next' builtin shadowing
        post = self.vertices[(idx+1)%n]
        
        # 2. Create Mobject Container
        mark_group = VGroup()
        self.add(mark_group)
        
        # 3. Create Virtual Node for Mark (to hook updater)
        mark_node = GeoNode(mark_group, type=GeoNodeType.VIRTUAL)
        
        parents = [prev.center_node, curr.center_node, post.center_node]
        
        def angle_solver(target, rents):
            # Fetch live
            p_prev, p_curr, p_post = [p.get_live_data() for p in rents]
            v1 = p_prev - p_curr
            v2 = p_post - p_curr
            
            # Kernel Algo
            is_right, _ = SmartLayoutUtils.get_angle_properties(v1, v2)
            
            m = target.value
            # [Fix] VGroup does not have clear_points. Use submobjects list.
            # m.clear_points() 
            m.submobjects = [] # Clear children
            
            # Dynamic Radius based on side length (avoid oversized arcs)
            safe_r = min(radius, np.linalg.norm(v1)*0.3, np.linalg.norm(v2)*0.3)
            
            if is_right:
                # Draw square
                perp1 = SmartLayoutUtils.normalize(v1) * safe_r
                perp2 = SmartLayoutUtils.normalize(v2) * safe_r
                corners = [
                    p_curr + perp1,
                    p_curr + perp1 + perp2,
                    p_curr + perp2
                ]
                elbow = VMobject().set_points_as_corners(corners)
                elbow.set_stroke(color, 2)  # [Fix] Removed set_opacity(0)
                m.add(elbow)
            else:
                # Draw Arc
                # Angles in Manim are tricky. Absolute angle atan2.
                start_angle = np.arctan2(v2[1], v2[0])
                end_angle = np.arctan2(v1[1], v1[0])
                
                # Normalize logic to always be the interior angle
                diff = (end_angle - start_angle) % TAU
                if diff > PI: 
                    # Swap order to draw the smaller arc
                    start_angle, end_angle = end_angle, start_angle
                    diff = TAU - diff
                
                arc = Arc(radius=safe_r, start_angle=start_angle, angle=diff, arc_center=p_curr, color=color)
                m.add(arc)
                
                if label:
                    # Place label at bisector
                    mid_angle = start_angle + diff/2
                    lbl_pos = p_curr + np.array([np.cos(mid_angle), np.sin(mid_angle), 0]) * (safe_r + 0.25)
                    # [Fix] Strip $$ wrapper from label before MathTex (which already provides math mode)
                    clean_label = label.strip('$').strip()
                    if not clean_label: clean_label = label  # safety fallback
                    tex = MathTex(clean_label, font_size=24, color=color).move_to(lbl_pos)
                    m.add(tex)
            
            return None # Updater handled internal change
            
        mark_node.add_constraint(angle_solver, parents)
        self.add_node(mark_node)
        
        # [Fix] Manually trigger solver once to populate content BEFORE animation
        # Otherwise Create() animates an empty VGroup
        angle_solver(mark_node, parents)
        
        return mark_group # Return the mark for play()


class SmartCircle(TopoGraph):
    """
    [Component: SmartCircle]
    Defined by Center (Node) and Radius (Value).
    """
    def __init__(self, center_node=ORIGIN, radius=2.0, color=WHITE, show_center=True):
        super().__init__()
        
        # 1. Center
        if isinstance(center_node, GeoNode):
            self.center_node = center_node
        elif hasattr(center_node, "center_node"):
            self.center_node = center_node.center_node
        else:
            self.center_node = GeoNode(np.array(center_node, dtype=np.float64), type=GeoNodeType.VERTEX)
        
        self.add_node(self.center_node, name="center")
        
        if show_center:
            # Create a dot and bind the node to it
            self.center_dot = Dot(self.center_node.get_live_data(), color=color, radius=0.04)
            self.add(self.center_dot)
            self.center_node.value = self.center_dot
            
        # 2. Radius
        self.radius_node = GeoNode(float(radius), type=GeoNodeType.VIRTUAL)
        self.add_node(self.radius_node, name="radius")
        
        # 3. The Circle Mobject
        self.circle = Circle(radius=radius, color=color)
        self.circle.move_to(self.center_node.get_live_data())
        
        # Bind position and radius to circle mobject
        c_node = GeoNode(self.circle, type=GeoNodeType.EDGE)
        
        def circle_pos_solver(target, rents):
            c = rents[0].get_live_data()
            r = rents[1].get_live_data()
            target.value.move_to(c)
            # [FIX] Multiply by layout_scale so LayoutManager's scaling is preserved.
            # Without this, clamp_to_content_zone's scaling is undone every animation frame.
            target.value.set_width(2 * r * self.layout_scale)
            return None
        
        c_node.add_constraint(circle_pos_solver, [self.center_node, self.radius_node])
        self.add_node(c_node)

    def get_visual_radius(self):
        """Returns the actual visual radius in world space, read from Circle geometry."""
        # [FIX] Read from self.circle.width/2 instead of radius_node.value * layout_scale.
        # This is always accurate even when the constraint system redraws the circle.
        return self.circle.width / 2

    def set_radius(self, radius):
        """
        [Semantic API] Updates the circle radius.
        """
        self.set_node_local_data("radius", radius)
        return self

    def move_center(self, to):
        """
        [Semantic API] Moves the center of the circle.
        """
        self.set_node_local_data("center", to)
        return self

    def add_label(self, text="C", direction=UR, buff=0.3, color=WHITE):
        """
        Adds a label to the circle, position relative to circumference.
        """
        lbl = SmartLayoutUtils.safe_tex(text, color=color)
        self.add(lbl)
        label_node = GeoNode(lbl, type=GeoNodeType.LABEL)
        
        def label_solver(target, rents):
            # [FIX] Convert direction to array to allow multiplication with float radius/scale
            pos = self.get_visual_center() + np.array(direction) * (self.get_visual_radius() + buff)
            target.value.move_to(pos)
            return pos  # [Bug #3 Fix] Return position for initial placement
            
        label_node.add_constraint(label_solver, [self.center_node])
        self.add_node(label_node)
        
        # [Fix] Manually trigger solver once to position label BEFORE animation
        result = label_solver(label_node, [self.center_node])
        if result is not None:
            lbl.move_to(result)
            
        # [Fix] Return the label object instead of self (Circle) to avoid redundant Create() animation
        return lbl

    def add_center_label(self, text="O", direction=UR, buff=0.35, color=WHITE):
        """
        Adds a label to the circle's center point.
        Delegate to the center point's add_label.
        """
        # The original _SmartPoint had an add_label. Now center_node is a GeoNode.
        # We need to create a label directly for the center_node.
        lbl = SmartLayoutUtils.safe_tex(text, color=color)
        self.add(lbl)
        label_node = GeoNode(lbl, type=GeoNodeType.LABEL)

        def center_label_solver(target, rents):
            center_pos = rents[0].get_live_data()
            # If center_node.value is a Dot, get its center. Otherwise, it's a point.
            if hasattr(center_pos, 'get_center'):
                center_pos = center_pos.get_center()
            return center_pos + np.array(direction) * buff

        label_node.add_constraint(center_label_solver, [self.center_node])
        self.add_node(label_node)
        center_label_solver(label_node, [self.center_node]) # Initial update
        return lbl

    def get_visual_center(self):
        """Returns the actual visual center in world space."""
        return self.circle.get_center()

    def add_radius_line(self, angle=0, color=YELLOW, show_label=True, label_text="r"):
        """
        Draw a radius line from center to circumference at a given angle.
        
        Args:
            angle: Angle in radians (default 0, pointing right)
            color: Line color
            show_label: Whether to show a label on the radius
            label_text: Text for the label (default "r")
        """
        line_group = VGroup()
        self.add(line_group)
        r_node = GeoNode(line_group, type=GeoNodeType.VIRTUAL)
        
        parents = [self.center_node, self.radius_node]
        
        def radius_solver(target, rents):
            c = self.get_visual_center() # Use visual center
            r = self.get_visual_radius() # Use visual radius
            
            # Calculate endpoint on circumference
            # [Visual Fix] Adjust to account for the thickness of the dot at the end
            dot_radius = 0.05
            effective_r = r - dot_radius
            
            # end_point for the DOT (center of the dot)
            dot_center = c + effective_r * np.array([np.cos(angle), np.sin(angle), 0])
            # end_point for the LINE (exactly on circumference)
            actual_end = c + r * np.array([np.cos(angle), np.sin(angle), 0])
            
            m = target.value
            m.submobjects = []  # Clear
            
            # 1. Draw the radius line (to the actual circumference)
            line = Line(c, actual_end, color=color)
            m.add(line)
            
            # 2. Draw the dot (tangent to circumference inner edge)
            dot = Dot(dot_center, color=color, radius=dot_radius)
            m.add(dot)
            
            # Add label if requested
            if show_label and label_text:
                mid = (c + actual_end) / 2
                # Offset perpendicular to the line
                perp = np.array([-np.sin(angle), np.cos(angle), 0]) * 0.25
                lbl = SmartLayoutUtils.safe_tex(label_text, color=color)
                lbl.move_to(mid + perp)
                lbl.scale(0.7)
                m.add(lbl)
        
        r_node.add_constraint(radius_solver, parents)
        self.add_node(r_node)
        return line_group

    def add_angle(self, index, radius=0.4, color=YELLOW, label="\\alpha", other_angle=False):
        """
        [Incremental API] Adds an angle mark at a specific point on the circle.
        Automatically finds lines connected to or passing through this point to define the angle.
        """
        # 1. Resolve Target Node
        name = str(index).replace("$", "").replace("\\", "").replace("{", "").replace("}", "").strip()
        target_node = self.node_map.get(name)
        
        if not target_node:
            # Proximity fallback: find vertex closest to o_pos
            return VGroup()
            
        o_pos = target_node.get_live_data()
        
        # 2. Collect candidate ray directions from all edges
        # [Bug #4 Fix] Unified line segment discovery:
        #   - _SmartLine objects (from add_line)
        #   - Line objects inside VGroups (from add_radius_line, add_tangent_line)
        #   - Constraint parents of EDGE nodes (fallback)
        candidate_dirs = []
        
        def _collect_from_endpoints(p1, p2):
            """Helper: given two endpoints, check if o_pos lies on/near the segment and collect ray directions."""
            v_line = p2 - p1
            l_sq = np.dot(v_line, v_line)
            if l_sq < 1e-6: return
            t = np.dot(o_pos - p1, v_line) / l_sq
            proj_pt = p1 + t * v_line
            if np.linalg.norm(proj_pt - o_pos) < 0.1:
                if np.linalg.norm(o_pos - p1) < 0.05:
                    candidate_dirs.append(SmartLayoutUtils.normalize(p2 - p1))
                elif np.linalg.norm(o_pos - p2) < 0.05:
                    candidate_dirs.append(SmartLayoutUtils.normalize(p1 - p2))
                elif -0.01 < t < 1.01:
                    candidate_dirs.append(SmartLayoutUtils.normalize(p1 - o_pos))
                    candidate_dirs.append(SmartLayoutUtils.normalize(p2 - o_pos))
        
        for node in self.nodes:
            # Strategy A: _SmartLine edge nodes (from add_line)
            if node.type == GeoNodeType.EDGE and isinstance(node.value, _SmartLine):
                line = node.value
                p1 = line.start_node.get_live_data() if hasattr(line.start_node, "get_live_data") else line.start_node
                p2 = line.end_node.get_live_data() if hasattr(line.end_node, "get_live_data") else line.end_node
                _collect_from_endpoints(p1, p2)
            # Strategy B: EDGE nodes with constraint parents (from add_radius_line etc.)
            elif node.type in [GeoNodeType.EDGE, GeoNodeType.VIRTUAL] and node.constraints:
                for solver, parents in node.constraints:
                    if len(parents) >= 2:
                        try:
                            p1 = parents[0].get_live_data()
                            p2 = parents[1].get_live_data()
                            if hasattr(p1, 'get_center'): p1 = p1.get_center()
                            if hasattr(p2, 'get_center'): p2 = p2.get_center()
                            if isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray) and p1.shape == (3,) and p2.shape == (3,):
                                _collect_from_endpoints(p1, p2)
                        except: pass
        
        # Strategy C: Scan child _SmartLine sub-components
        for sub in self.submobjects:
            if isinstance(sub, _SmartLine):
                p1 = sub.p1_node.get_live_data()
                p2 = sub.p2_node.get_live_data()
                if hasattr(p1, 'get_center'): p1 = p1.get_center()
                if hasattr(p2, 'get_center'): p2 = p2.get_center()
                _collect_from_endpoints(p1, p2)

        # 3. Filter unique directions (ignore opposite vectors for now)
        if len(candidate_dirs) < 2:
            return VGroup()
            
        # Prioritize Picking Ray Pairs
        # Typically we want Radius (O-Center) and Tangent segment
        ray1 = candidate_dirs[0]
        ray2 = None
        
        # Look for a ray that is NOT collinear with ray1
        for d in candidate_dirs[1:]:
            dot = np.dot(ray1, d)
            if abs(dot) < 0.95: # Not same or opposite
                ray2 = d
                break
        
        if ray2 is None:
            # If all rays are collinear, we can't form an angle
            return VGroup()
            
        # 4. Final creation
        actual_color = SmartLayoutUtils.parse_color(color)
        angle_obj = SmartAngle(
            O=target_node,
            A=o_pos + ray1,
            B=o_pos + ray2,
            radius=radius,
            color=actual_color,
            other_angle=other_angle
        )
        
        if label:
            angle_obj.add_label(text=label, color=actual_color)
            
        self.add(angle_obj)
        return angle_obj

    def add_tangent_at_point(self, point=RIGHT*2, length=4, color=YELLOW, show_right_angle=True, right_angle_size=0.2):
        """
        [过圆上一点做切线] 在圆周上指定点处画一条切线。
        切线垂直于该点处的半径，只画一条线。

        Args:
            point: 圆上的点。支持三种输入：
                   - 字符串名称（如 "T"，通过 add_point 注册的标识符）
                   - 坐标数组（如 [2, 0, 0]，局部坐标）
                   - 角度浮点值（如 PI/4，从圆心出发的弧度角）
            length: 切线总长度（以切点为中心向两侧各延伸 length/2）
            color: 切线颜色
            show_right_angle: 是否在切点处绘制直角符号（默认 True）
            right_angle_size: 直角符号大小（默认 0.2）
        """
        line_group = VGroup()
        self.add(line_group)
        t_node = GeoNode(line_group, type=GeoNodeType.VIRTUAL)

        # --- 解析 point 输入 ---
        resolved = self._resolve_node_or_coord(point)

        if isinstance(resolved, GeoNode):
            # 通过名称或坐标匹配到了已注册的节点
            point_node = resolved
        elif isinstance(resolved, (int, float)):
            # 输入是角度（弧度），计算圆上对应坐标
            angle_val = float(resolved)
            r = self.radius_node.get_live_data()
            c = self.center_node.get_live_data()
            if hasattr(c, 'get_center'):
                c = c.get_center()
            pt_world = np.array(c) + np.array([np.cos(angle_val), np.sin(angle_val), 0]) * r
            point_node = GeoNode(pt_world, type=GeoNodeType.VIRTUAL)
            self.nodes.append(point_node)
        else:
            # 输入是坐标数组（局部坐标），转为世界坐标
            pt_world = np.array(resolved, dtype=np.float64) + self.layout_offset
            point_node = GeoNode(pt_world, type=GeoNodeType.VIRTUAL)
            self.nodes.append(point_node)

        parents = [self.center_node, self.radius_node, point_node]

        def tangent_at_solver(target, rents):
            c = rents[0].get_live_data()  # Center (world)
            r_val = rents[1].get_live_data()  # Radius
            tp = rents[2].get_live_data()  # Tangent point (world)

            if hasattr(c, 'get_center'):
                c = c.get_center()
            if hasattr(tp, 'get_center'):
                tp = tp.get_center()

            if c is None or tp is None:
                return

            c = np.array(c, dtype=np.float64)
            tp = np.array(tp, dtype=np.float64)

            # 半径方向：从圆心指向切点
            radius_dir = tp - c
            radius_len = np.linalg.norm(radius_dir)
            if radius_len < 1e-8:
                return
            radius_dir = radius_dir / radius_len

            # 将切点"钉"在圆周上（防止坐标偏差）
            tp_snapped = c + radius_dir * r_val

            # 切线方向：垂直于半径
            tangent_dir = np.array([-radius_dir[1], radius_dir[0], 0])

            # 画切线：以切点为中心向两边延伸
            half = length / 2.0
            p1 = tp_snapped - tangent_dir * half
            p2 = tp_snapped + tangent_dir * half

            m = target.value
            m.submobjects = []

            actual_color = SmartLayoutUtils.parse_color(color)
            l = Line(p1, p2, color=actual_color)
            dot = Dot(tp_snapped, color=actual_color, radius=0.06)
            m.add(l, dot)

            # 直角符号
            if show_right_angle:
                s = right_angle_size
                corner1 = tp_snapped - radius_dir * s
                corner2 = tp_snapped - radius_dir * s + tangent_dir * s
                corner3 = tp_snapped + tangent_dir * s

                angle_mark = VGroup(
                    Line(corner1, corner2, color=actual_color, stroke_width=2),
                    Line(corner2, corner3, color=actual_color, stroke_width=2)
                )
                m.add(angle_mark)

        t_node.add_constraint(tangent_at_solver, parents)
        self.add_node(t_node)
        return line_group

    def add_tangent_line(self, external_point=RIGHT*3, length=4, color=YELLOW, show_right_angle=True, right_angle_size=0.2):
        """
        Dynamic Tangent Line from an external point.
        Uses Kernel's get_tangent_points_to_circle.
        
        Args:
            external_point: Point outside the circle
            length: Length of the tangent line
            color: Line color
            show_right_angle: Whether to show a right angle mark at the tangent point (default True)
            right_angle_size: Size of the right angle mark (default 0.2)
        """
        # Create Line Container
        line_group = VGroup()
        self.add(line_group)
        t_node = GeoNode(line_group, type=GeoNodeType.VIRTUAL)
        
        # Ensure external point is a node for tracking
        if hasattr(external_point, "center_node"):
            ext_node = external_point.center_node
        else:
            # [Fix] 将局部坐标转为世界坐标，与 add_point 保持一致
            ext_node = GeoNode(np.array(external_point, dtype=np.float64) + self.layout_offset, type=GeoNodeType.VIRTUAL)
            self.nodes.append(ext_node)
        
        parents = [self.center_node, self.radius_node, ext_node]
        
        def tangent_solver(target, rents):
            c = rents[0].get_live_data() # Center
            r_val = rents[1].get_live_data() # Radius value
            p = rents[2].get_live_data() # External point
            
            # Call Kernel
            tan_pts = SmartLayoutUtils.get_tangent_points_to_circle(p, c, r_val)
            
            m = target.value
            m.submobjects = [] # Clear
            
            if not tan_pts: return # Point inside circle
            
            # Draw lines to both tangent points
            for tp in tan_pts:
                # Extension logic
                # Vector from P to TangentPoint
                v = SmartLayoutUtils.normalize(tp - p)
                end = p + v * length
                l = Line(p, end, color=color)
                dot = Dot(tp, color=color, radius=0.06) # Mark touch point
                m.add(l, dot)
                
                # Add right angle mark at tangent point
                if show_right_angle:
                    # The tangent is perpendicular to the radius at the tangent point
                    # Radius direction: from center to tangent point
                    radius_dir = SmartLayoutUtils.normalize(tp - c)
                    # Tangent direction: perpendicular to radius
                    tangent_dir = np.array([-radius_dir[1], radius_dir[0], 0])
                    
                    # Create the right angle mark (small square)
                    s = right_angle_size
                    corner1 = tp - radius_dir * s
                    corner2 = tp - radius_dir * s + tangent_dir * s
                    corner3 = tp + tangent_dir * s
                    
                    angle_mark = VGroup(
                        Line(corner1, corner2, color=color, stroke_width=2),
                        Line(corner2, corner3, color=color, stroke_width=2)
                    )
                    m.add(angle_mark)
                
        t_node.add_constraint(tangent_solver, parents)
        self.add_node(t_node)
        return line_group

    def add_side_label(self, side_index=0, text="a", buff=0.3, color=WHITE):
        """
        [Master Feature] 给通过 add_radius_line() / add_line() 添加的线段添加文本标签。
        API 签名与 SmartPolygon.add_side_label 保持一致。
        
        参数：
            side_index: 整数索引 (0, 1...) 或语义索引 ('e0', 'e1'...)。
                        按 add_radius_line / add_line 调用的先后顺序排列。
            text:       LaTeX 标签内容（如 "r", "$$\\pi r$$"）。
            buff:       标签到线段中点的偏移距离。
            color:      标签颜色。
        """
        # --- Step 1: 解析 side_index ---
        idx = side_index
        if isinstance(side_index, str):
            if side_index.startswith('e'):
                try: idx = int(side_index[1:])
                except ValueError: idx = 0
            else:
                try: idx = int(side_index)
                except ValueError: idx = 0

        # --- Step 2: 从 self.nodes 中收集所有 EDGE 类型节点 ---
        edge_nodes = [n for n in self.nodes if n.type == GeoNodeType.EDGE]

        if not (isinstance(idx, int) and 0 <= idx < len(edge_nodes)):
            print(f"[Warning] SmartCircle.add_side_label: side_index '{side_index}' (resolved to {idx}) is out of range (0-{len(edge_nodes)-1}). Returning empty object.")
            return VMobject()

        edge_node = edge_nodes[idx]
        
        # --- Step 3: 从边的约束中提取两个端点节点 ---
        s_node, e_node = None, None
        for solver, parents in edge_node.constraints:
            if len(parents) >= 2:
                s_node, e_node = parents[0], parents[1]
                break
        
        if s_node is None or e_node is None:
            print(f"[Warning] SmartCircle.add_side_label: Could not extract endpoints from edge {idx}. Returning empty object.")
            return VMobject()

        # --- Step 4: 创建标签 Mobject ---
        actual_color = SmartLayoutUtils.parse_color(color)
        lbl = SmartLayoutUtils.safe_tex(text, color=actual_color)
        l_node = GeoNode(lbl, type=GeoNodeType.LABEL)
        
        # --- Step 5: 偏置参考点（圆心） ---
        circle_centroid = GeoNode(None)
        circle_centroid.get_live_data = lambda: self.get_visual_center()
        self.nodes.append(circle_centroid)
        
        # --- Step 6: 求解器 - 外侧反冲逻辑 (Outward Repulsion) ---
        def side_label_solver(target, parents):
            try:
                p1, p2, cent = [p.get_live_data() for p in parents]
                p1_pos = p1.get_center() if hasattr(p1, 'get_center') else p1
                p2_pos = p2.get_center() if hasattr(p2, 'get_center') else p2
                cent_pos = cent.get_center() if hasattr(cent, 'get_center') else cent
            except:
                return target.value.get_center()
                
            mid = (p1_pos + p2_pos) / 2
            v = p2_pos - p1_pos
            if np.linalg.norm(v) < 1e-4: return mid
            
            perp = np.array([-v[1], v[0], 0])
            perp = SmartLayoutUtils.normalize(perp)
            
            pos1 = mid + perp * buff
            pos2 = mid - perp * buff
            dist1 = np.linalg.norm(pos1 - cent_pos)
            dist2 = np.linalg.norm(pos2 - cent_pos)
            
            final_dir = perp if dist1 > dist2 else -perp
            return mid + final_dir * buff
        
        l_node.add_constraint(side_label_solver, [s_node, e_node, circle_centroid])
        self.add_node(l_node)
        self.add(lbl)
        
        # 初始位置对齐
        init_pos = side_label_solver(l_node, [s_node, e_node, circle_centroid])
        if init_pos is not None: lbl.move_to(init_pos)
        
        return lbl


class SmartFnPlot(TopoGraph):
    """
    [Component: Universal Function Plot]
    Wraps Manim's FunctionGraph/Axes for calculus & algebra.
    Features: Auto-Derivative, Tangent Lines, Area under curve.
    """
    def __init__(self, function_code="x**2", x_range=None, axes_config=None, color=BLUE):
        super().__init__()
        
        # Default x_range
        if x_range is None:
            x_range = [-3, 3]
        
        # 1. Parse Function (Safe eval for known math functions)
        # In a real production environment, use a safer parser.
        # Here we support basic numpy syntax: np.sin, np.cos, x, etc.
        # [FIX] Escape backslashes for eval to avoid SyntaxWarning for LaTeX symbols like \pi
        safe_code = function_code.replace("\\", "\\\\")
        self.func_lambda = eval(f"lambda x: {safe_code}", {"np": np, "sys": None, "os": None})
        self.x_range = x_range
        
        # 2. Create Axes (Default standard with AUTO y_range)
        if axes_config is None:
            # [FIX] Auto-calculate y_range with sensible step sizes
            # Increase sampling points to 200 to ensure we don't miss vertices
            x_samples = np.linspace(x_range[0], x_range[1], 200)
            y_samples = [self.func_lambda(x) for x in x_samples]
            y_min_raw = min(y_samples)
            y_max_raw = max(y_samples)
            
            y_span_raw = max(y_max_raw - y_min_raw, 0.1)
            
            # Heuristic for step size
            if y_span_raw <= 5: y_step = 1
            elif y_span_raw <= 15: y_step = 2
            elif y_span_raw <= 40: y_step = 5
            elif y_span_raw <= 150: y_step = 10
            elif y_span_raw <= 400: y_step = 50
            else: y_step = 100
            
            # [FIX] Align range to steps and add AT LEAST 1 step of padding on each side
            # This ensures the curve's peaks and valleys are well within the visible axis ticks
            y_min = (np.floor(y_min_raw / y_step) - 1) * y_step
            y_max = (np.ceil(y_max_raw / y_step) + 1) * y_step
            
            # Similarly for X if span is large
            x_span_raw = x_range[1] - x_range[0]
            x_step = 1
            if x_span_raw > 15: x_step = 5
            if x_span_raw > 40: x_step = 10

            # [ROOT CAUSE FIX] Use FIXED screen dimensions for consistent visual appearance
            # The previous dynamic scaling caused different-sized coordinate systems across plots.
            # Solution: ALWAYS use the same x_length and y_length. Let Manim handle the internal scaling.
            # The aspect ratio will naturally adjust based on x_range and y_range.
            x_length = 5.4  # Fixed width
            y_length = 4.0  # Fixed height - this ensures all plots have the same size coordinate system

            # [FIX] Enhanced axis config to show ticks and prevent overlap
            axis_common_config = {
                "include_tip": True,
                "include_numbers": True,
                "font_size": 20,  # Smaller font to avoid overlap
                "decimal_number_config": {"num_decimal_places": 0} # Keep it clean for school math
            }
            
            axes_config = {
                "x_length": x_length, 
                "y_length": y_length, 
                "x_range": [x_range[0], x_range[1], x_step], 
                "y_range": [y_min, y_max, y_step],
                "axis_config": axis_common_config
            }
            
        self.axes = Axes(**axes_config)
        self.add(self.axes)
        self.add_node(GeoNode(self.axes, type=GeoNodeType.VIRTUAL)) # Anchor
        
        # 3. Create Curve
        self.curve = self.axes.plot(self.func_lambda, x_range=x_range, color=color)
        self.add(self.curve)
        
        # 4. Curve Node (for point projection)
        self.curve_node = GeoNode(self.curve, type=GeoNodeType.EDGE)
        
        # Solver: No dynamic update needed for static function unless params change
        # But we need to expose the lambda for others to project onto
        self.curve_node.get_live_data = lambda: self.func_lambda # Special: returns function!
        
        self.add_node(self.curve_node)

    def add_label(self, text="f(x)", x_val=0, direction=UP, buff=0.3, color=WHITE):
        """Label the function at a specific x"""
        # Convert string direction to Manim constant
        if isinstance(direction, str):
            direction_map = {'UP': UP, 'DOWN': DOWN, 'LEFT': LEFT, 'RIGHT': RIGHT, 
                             'UR': UR, 'UL': UL, 'DR': DR, 'DL': DL}
            direction = direction_map.get(direction.upper(), UP)
        
        # Calculate pos
        y_val = self.func_lambda(x_val)
        pos = self.axes.c2p(x_val, y_val)
        
        # Use safe_tex for consistent CJK handling
        lbl = SmartLayoutUtils.safe_tex(text, color=color)
        l_node = GeoNode(lbl, type=GeoNodeType.LABEL)
        
        # [FIX] Convert direction to array to allow multiplication with float radius/scale
        final_pos = pos + np.array(direction) * buff
        lbl.move_to(final_pos)
        self.add(lbl)
        self.add_node(l_node)
        return lbl
    def add_point(self, coords=[0, 0, 0], label_text="P", color=WHITE, radius=0.08):
        """
        [Override] Adds a point using logical coordinates (x, y) on the function graph.
        Converts logical coords to screen coords via axes.c2p() before placing the dot.
        """
        # Ensure 3D coords for API consistency
        if len(coords) == 2: coords = list(coords) + [0]
        
        # [Bug Fix] Convert logical coordinates (x, y) to Manim world coordinates
        # Previously coords were passed directly, causing points to appear at wrong screen positions
        world_pos = np.array(self.axes.c2p(coords[0], coords[1]), dtype=np.float64)
        
        pt = _SmartPoint(coords=world_pos, label_text=label_text, color=color, radius=radius)
        if not isinstance(self, Scene):
            self.add(pt)
        
        # Record logical position for semantic APIs (e.g. add_line between points)
        pt.center_node.logical_pos = np.array(coords[:2], dtype=np.float64)
        
        # Register node name for semantic lookup
        name = str(label_text).replace("$", "").replace("\\", "").replace("{", "").replace("}", "").strip()
        self.add_node(pt.center_node, name=name if name else None)
        return pt

    def add_line(self, start_node=[0, 0, 0], end_node=[1, 1, 0], color=BLUE, stroke_width=4):
        """
        [Override] Adds a line using logical coordinates (x1, y1) -> (x2, y2).
        Converts logical coords to screen coords via axes.c2p().
        """
        # [Bug Fix] Convert logical coordinates to Manim world coordinates
        def _to_world(node_or_coord):
            if isinstance(node_or_coord, (list, tuple, np.ndarray)):
                c = list(node_or_coord)
                if len(c) == 2: c = c + [0]
                return np.array(self.axes.c2p(c[0], c[1]), dtype=np.float64)
            return node_or_coord  # Already a component/node, pass through
        
        world_start = _to_world(start_node)
        world_end = _to_world(end_node)
        
        line = _SmartLine(start_node=world_start, end_node=world_end, color=color, stroke_width=stroke_width)
        if not isinstance(self, Scene):
             self.add(line)
        self.add_node(line.line_node)
        return line

    def add_side_label(self, side_index=0, text="a", buff=0.3, color=WHITE):
        """
        [SmartFnPlot Override] 给通过 add_line() 添加的辅助线段添加文本标签。
        side_index: 支持整数索引 (0, 1...) 或语义索引 'e0', 'e1'...
                    索引按 add_line() 调用的先后顺序排列。
        text:       LaTeX 标签内容（如 "x=h"）。
        buff:       标签到线段中点的偏移距离。
        color:      标签颜色。
        """
        # --- Step 1: 解析 side_index ---
        idx = side_index
        if isinstance(side_index, str):
            if side_index.startswith('e'):
                try: idx = int(side_index[1:])
                except ValueError: idx = 0
            else:
                try: idx = int(side_index)
                except ValueError: idx = 0

        # --- Step 2: 从 self.nodes 中收集所有 EDGE 类型节点 ---
        edge_nodes = [n for n in self.nodes if n.type == GeoNodeType.EDGE]

        if not (0 <= idx < len(edge_nodes)):
            print(f"[Warning] SmartFnPlot.add_side_label: index {idx} out of range (0-{len(edge_nodes)-1}). Returning empty object.")
            return VMobject()  # [Safety Fix] Prevent Write(None) crashes

        edge_node = edge_nodes[idx]

        # --- Step 3: 获取端点节点 ---
        # _SmartLine 的 line_node 通过 add_constraint(solver, [p1_node, p2_node]) 注册了端点
        if not edge_node.constraints:
            return None
        _, parents = edge_node.constraints[0]
        if len(parents) < 2:
            return None
        p1_node, p2_node = parents[0], parents[1]

        # --- Step 4: 创建 LaTeX 标签 ---
        lbl = SmartLayoutUtils.safe_tex(text, color=color)
        l_node = GeoNode(lbl, type=GeoNodeType.LABEL)

        # --- Step 5: 位置 Solver —— 垂直偏移到线段中点外侧 ---
        # 对于函数图像，我们用"远离曲线"的方向来决定标签偏移。
        # 具体策略：计算线段中点处函数值，如果中点在曲线同侧则反向。
        func_lambda = self.func_lambda
        axes = self.axes

        def side_label_solver(target, parents):
            p1 = parents[0].get_live_data()
            p2 = parents[1].get_live_data()
            mid = (p1 + p2) / 2.0

            # 线段方向向量
            v = p2 - p1
            length_v = np.linalg.norm(v)
            if length_v < 1e-9:
                return mid  # 退化情况

            # 垂直于线段的方向（在屏幕空间）
            perp = np.array([-v[1], v[0], 0.0])
            perp = SmartLayoutUtils.normalize(perp)

            # 智能方向选择：远离函数曲线
            # 将线段中点反映射回逻辑坐标，查询函数值
            try:
                mid_logical = axes.p2c(mid)  # screen -> logical
                x_mid = mid_logical[0]
                y_curve = func_lambda(x_mid)
                curve_screen = np.array(axes.c2p(x_mid, y_curve), dtype=np.float64)

                # 选择远离曲线的方向
                pos1 = mid + perp * buff
                pos2 = mid - perp * buff
                d1 = np.linalg.norm(pos1 - curve_screen)
                d2 = np.linalg.norm(pos2 - curve_screen)
                final_dir = perp if d1 > d2 else -perp
            except Exception:
                # 回退：默认向右上方偏移
                final_dir = perp if perp[1] >= 0 else -perp

            return mid + final_dir * buff

        l_node.add_constraint(side_label_solver, [p1_node, p2_node])
        self.add_node(l_node)
        self.add(lbl)

        # 立即触发一次 solver，确保标签在动画前就已就位
        result = side_label_solver(l_node, [p1_node, p2_node])
        if result is not None:
            lbl.move_to(result)

        return lbl

    def create_tangent_line(self, x_val, length=3, color=YELLOW):
        """
        Creates a tangent line at x_val.
        [FIX] Uses Math-to-Screen projection to ensure accuracy regardless of axis aspect ratio.
        """
        t_group = VGroup()
        self.add(t_group)
        t_node = GeoNode(t_group, type=GeoNodeType.VIRTUAL)
        
        # 1. Calculate anchor point in screen space
        y_val = self.func_lambda(x_val)
        pt_axes = self.axes.c2p(x_val, y_val)
        
        # 2. Calculate slope in math space
        slope = SmartLayoutUtils.get_numerical_derivative(self.func_lambda, x_val)
        
        # 3. Project a tiny segment to determine screen angle
        dt = 0.01
        p1_math = np.array([x_val - dt, y_val - dt * slope, 0])
        p2_math = np.array([x_val + dt, y_val + dt * slope, 0])
        
        p1_screen = self.axes.c2p(p1_math[0], p1_math[1])
        p2_screen = self.axes.c2p(p2_math[0], p2_math[1])
        
        # Direction in screen space
        screen_vec = p2_screen - p1_screen
        screen_unit = screen_vec / np.linalg.norm(screen_vec)
        
        # 4. Draw line using screen-space unit vector
        l = Line(pt_axes - screen_unit * length/2, pt_axes + screen_unit * length/2, color=color)
        t_group.add(l)
        
        # Add a subtle dot at the interaction point
        # [Rule 3.2] Dots for Path B operations
        dot = Dot(pt_axes, color=color, radius=0.06)
        t_group.add(dot)
        
        self.add_node(t_node)
        return t_group


class SmartSector(TopoGraph):
    """
    [Component: Smart Sector]
    Wraps Manim's Sector.
    Used for: Circle partitioning, Pie charts, Area proofs.
    """
    def __init__(self, radius=1.5, start_angle=0, angle=PI/3, color=BLUE, fill_opacity=0.5):
        super().__init__()
        
        # 1. State
        self.radius_val = radius
        self.start_angle_val = start_angle
        self.angle_val = angle
        
        # 2. Visual Object
        self.sector = Sector(
            outer_radius=radius,
            inner_radius=0, # Can be modified for Annulus sector later
            start_angle=start_angle,
            angle=angle,
            color=color,
            fill_opacity=fill_opacity
        )
        self.add(self.sector)
        
        # 3. Nodes
        # [Visual Fix] Use a Dot anchor so the center follows LayoutManager shifts.
        self.center_anchor = Dot(radius=0, fill_opacity=0, stroke_opacity=0)
        self.add(self.center_anchor)
        self.center_node = GeoNode(self.center_anchor, type=GeoNodeType.POINT)
        
        # [Visual Fix] Sync origin immediately using safe shift
        self.sector.shift(self.center_node.get_live_data() - self.sector.get_arc_center())
        self.sector_node = GeoNode(self.sector, type=GeoNodeType.SURFACE)
        
        # Updater to link sector pos to center_node
        def sector_pos_updater(s, parents):
            center = parents[0].get_live_data()
            s.value.shift(center - s.value.get_arc_center())
            return None

        self.sector_node.add_constraint(sector_pos_updater, [self.center_node])
        self.add_node(self.center_node)
        self.add_node(self.sector_node)
        
    def add_angle(self, label="\\theta", radius=0.6, color=YELLOW):
        """
        [Incremental API] Adds a central angle mark.
        """
        # 1. Create Invisible Anchor Points for Rays
        # We need them to ensure the angle tracks the sector's rotation/layout
        
        # A: Start Angle Ray Point
        pt_a = _SmartPoint(ORIGIN, label_text="", radius=0)
        pt_a.dot.set_opacity(0)
        self.add(pt_a) # Visual child
        # Note: We do NOT add to self.nodes to avoid triple-updater registration
        # (Once by _SmartPoint, Once by SmartAngle, Once by us)
        # _SmartPoint already handles its own update logic internally.
        
        # B: End Angle Ray Point
        pt_b = _SmartPoint(ORIGIN, label_text="", radius=0)
        pt_b.dot.set_opacity(0)
        self.add(pt_b)

        # 2. Constraints (Bind positions to center + angle)
        def ray_solver(target, parents, angle_offset=0):
            c = parents[0].get_live_data()
            r = self.get_visual_radius()
            ang = self.start_angle_val + angle_offset
            vec = np.array([np.cos(ang), np.sin(ang), 0])
            return c + vec * r

        # Force immediate calculation so SmartAngle gets correct initial positions
        # (Otherwise it might init at ORIGIN and jump)
        pt_a.center_node.add_constraint(lambda t, p: ray_solver(t, p, 0), [self.center_node])
        pt_b.center_node.add_constraint(lambda t, p: ray_solver(t, p, self.angle_val), [self.center_node])
        
        # [Critical Fix] Manually set initial positions before creating SmartAngle
        # to prevent parallel lines error when both points are at ORIGIN
        center_pos = self.center_node.get_live_data()
        radius = self.get_visual_radius()
        
        # Calculate initial positions
        start_ang = self.start_angle_val
        end_ang = self.start_angle_val + self.angle_val
        
        pt_a_pos = center_pos + radius * np.array([np.cos(start_ang), np.sin(start_ang), 0])
        pt_b_pos = center_pos + radius * np.array([np.cos(end_ang), np.sin(end_ang), 0])
        
        # Set initial positions explicitly
        pt_a.center_node.set_value(pt_a_pos)
        pt_b.center_node.set_value(pt_b_pos)
        
        # 3. Create SmartAngle
        sa = SmartAngle(
            O=self.center_node, 
            A=pt_a, 
            B=pt_b, 
            radius=radius, 
            color=color
        )
        
        if label:
            sa.add_label(text=label, radius_buff=1.5, color=color)
            
        self.add(sa)
        return sa

    def get_visual_radius(self):
        """Returns the actual visual radius accounting for scale."""
        return self.radius_val * self.layout_scale
        
    def add_label(self, text="S", buff=0.4, color=WHITE):
        """Add label inside the sector (centroid approximation)"""
        lbl = SmartLayoutUtils.safe_tex(text, color=color)
        l_node = GeoNode(lbl, type=GeoNodeType.LABEL)
        
        def centroid_solver(target, rents):
            # Simple polar centroid
            c_pos = self.center_node.get_live_data()
            mid_angle = self.start_angle_val + self.angle_val / 2
            # Use visual radius
            r_pos = self.get_visual_radius() * buff 
            
            vec = np.array([np.cos(mid_angle), np.sin(mid_angle), 0])
            return c_pos + vec * r_pos
            
        l_node.add_constraint(centroid_solver, [self.center_node])
        self.add_node(l_node)
        return lbl

    def add_arc_label(self, text="L", buff=0.3, color=WHITE):
        """Add label outside the arc"""
        # Use safe_tex for consistent CJK handling
        lbl = SmartLayoutUtils.safe_tex(text, color=color)
        l_node = GeoNode(lbl, type=GeoNodeType.LABEL)
        
        def arc_solver(target, parents):
            c_pos = self.center_node.get_live_data()
            mid_angle = self.start_angle_val + self.angle_val / 2
            r_pos = self.get_visual_radius() + buff
            vec = np.array([np.cos(mid_angle), np.sin(mid_angle), 0])
            return c_pos + vec * r_pos
            
        l_node.add_constraint(arc_solver, [self.center_node])
        self.add_node(l_node)
        return lbl


class SmartAngle(TopoGraph):
    """
    [Component: Smart Angle]
    Wraps Manim's Angle.
    Defined by 3 points: O (Vertex), A (Start Ray), B (End Ray).
    """
    def __init__(self, O=ORIGIN, A=RIGHT, B=UP, radius=0.5, color=YELLOW, other_angle=False):
        super().__init__()
        self.radius = radius
        self.color = color
        self.other_angle = other_angle
        
        # 1. Nodes (Inputs) - Allow passing in objects with center_node or coords
        def _resolve_node(P):
            if isinstance(P, GeoNode): # [Fix] Direct GeoNode passed
                return P
            if hasattr(P, "center_node"):
                return P.center_node
            elif hasattr(P, "p_node"): # Compatibility check
                return P.p_node
            else:
                return GeoNode(np.array(P), type=GeoNodeType.POINT)

        self.O_node = _resolve_node(O)
        self.A_node = _resolve_node(A)
        self.B_node = _resolve_node(B)
        
        if not hasattr(O, 'center_node'): self.add_node(self.O_node)
        if not hasattr(A, 'center_node'): self.add_node(self.A_node)
        if not hasattr(B, 'center_node'): self.add_node(self.B_node)

        # 2. Visual (Angle Arc)
        self.arc = Angle(Line(ORIGIN, RIGHT), Line(ORIGIN, UP), radius=radius, color=color, other_angle=other_angle)
        # [Visual Fix] Sync origin immediately to prevent "Split" appearance 
        self.arc.move_to(self.O_node.get_live_data())
        self.add(self.arc)
        
        self.arc_node = GeoNode(self.arc, type=GeoNodeType.EDGE)
        
        # 3. Solver
        def angle_updater(target, parents):
            o = parents[0].get_live_data()
            a = parents[1].get_live_data()
            b = parents[2].get_live_data()
            
            # [Safety Fix] Check for parallel lines before creating Angle
            # If points are too close or lines are parallel, skip update to avoid ValueError
            vec_a = a - o
            vec_b = b - o
            
            # Check if vectors are too small (points too close to origin)
            if np.linalg.norm(vec_a) < 1e-6 or np.linalg.norm(vec_b) < 1e-6:
                return None  # Skip update if points are at origin
            
            # Check if lines are parallel (cross product is zero)
            cross_prod = np.cross(vec_a[:2], vec_b[:2])  # Use 2D cross product
            if abs(cross_prod) < 1e-6:
                return None  # Skip update if lines are parallel
            
            # Reconstruct Angle logic dynamically
            # We can't just move points, Angle needs Lines to define it. 
            # But creating new Lines every frame is expensive.
            # Efficient way: Update lines inside the existing Angle object? 
            # Manim's Angle stores 'lines'. 
            
            # [Lens Fix] Normalize to Logical Space before regeneration
            # If the layout has been scaled, the visual radius we want must be de-scaled
            # to survive the upcoming .scale(self.layout_scale)
            l1 = Line(o, a)
            l2 = Line(o, b)
            
            try:
                new_angle = Angle(l1, l2, radius=self.radius, other_angle=self.other_angle, color=self.color)
            except ValueError as e:
                # If Angle creation fails (e.g., parallel lines), skip update
                if "parallel" in str(e).lower() or "intersection" in str(e).lower():
                    return None
                raise  # Re-raise other errors
            
            # [Lens Fix] Apply the layout-aware scale lens
            if self.layout_scale != 1.0:
                 new_angle.scale(self.layout_scale)
            
            # Transfer path points
            target.value.points = new_angle.points
            return None
            
        self.arc_node.add_constraint(angle_updater, [self.O_node, self.A_node, self.B_node])
        self.add_node(self.arc_node)
        
        # [Visual Fix] Force immediate update to prevent "popping" from logic 1.0 
        # to actual parameters during the first frame of animation.
        angle_updater(self.arc_node, [self.O_node, self.A_node, self.B_node])
        
    def add_label(self, text="\\theta", radius_buff=1.2, color=YELLOW):
        """Add angle value label"""
        # Use safe_tex for consistent CJK handling
        lbl = SmartLayoutUtils.safe_tex(text, color=color)
        l_node = GeoNode(lbl, type=GeoNodeType.LABEL)
        
        def label_pos_solver(target, parents):
            # Midpoint of angle
            o = self.O_node.get_live_data()
            a = self.A_node.get_live_data()
            b = self.B_node.get_live_data()
            
            # Vector OA, OB
            va = a - o
            vb = b - o
            angle_a = np.arctan2(va[1], va[0])
            angle_b = np.arctan2(vb[1], vb[0])
            
            # [Bug #5 Fix] Proper circular mean to handle ±π branch cut
            # Simple average fails when angles straddle the ±π boundary
            diff = (angle_b - angle_a) % (2 * np.pi)
            if diff > np.pi:
                diff -= 2 * np.pi
            mid_angle = angle_a + diff / 2
            
            # Position
            r = self.arc.radius + radius_buff # Relative to arc radius?
            vec = np.array([np.cos(mid_angle), np.sin(mid_angle), 0])
            
            # Correction if angle passes through branch cut?
            # Manim Angle handles this internally, we might need to query it.
            # Simplified for now.
            return o + vec * r
            
        l_node.add_constraint(label_pos_solver, [self.O_node, self.A_node, self.B_node])
        self.add_node(l_node)
        
        # [Fix] Immediate update
        label_pos_solver(l_node, [self.O_node, self.A_node, self.B_node])
        
        return lbl


class SmartWindmill(TopoGraph):
   
    def __init__(self, target: 'SmartTriangle' = None, color=YELLOW):
        super().__init__()
        self.squares = []  # [Safety] Always initialize before early return
        
        if target is None:
            target = SmartTriangle()
        
        # [LLM Hallucination Fix] Resolve string references
        target = SmartLayoutUtils.resolve_id(target)
        
        # 1. Resolve Vertices
        self.vertices = []
        
        # [Fix] Support dict coordinate input directly (as seen in LLM hallucinated code)
        if isinstance(target, dict):
            # Case A: {'A': [...], 'B': [...], 'C': [...]}
            if all(k in target for k in ['A', 'B', 'C']):
                target = SmartTriangle(A=target['A'], B=target['B'], C=target['C'], color=color)
            else:
                # Case B: Just a generic dict of points?
                points = [v for k, v in target.items() if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 2][:3]
                if len(points) == 3:
                    target = SmartTriangle(A=points[0], B=points[1], C=points[2], color=color)

        if isinstance(target, SmartTriangle) or isinstance(target, SmartPolygon):
            self.vertices = target.vertices
            self.core_triangle = target if isinstance(target, SmartTriangle) else None
        elif isinstance(target, (list, tuple)):
            for p in target:
                if hasattr(p, "center_node"):
                    self.vertices.append(p)
                else:
                    pt = _SmartPoint(p, label_text="", radius=0)
                    self.add_node(pt.center_node)
                    self.vertices.append(pt)
        
        # Store core triangle for delegation if possible
        if not hasattr(self, "core_triangle"):
            self.core_triangle = None
        
        if len(self.vertices) < 3: return
        for i in range(3):
            # Each square is a SmartPolygon with 4 vertices
            # Two vertices are shared with the triangle edge
            # Two are new and dependent
            # Side i: from vertex i to vertex (i+1)%3
            v1 = self.vertices[i]
            v2 = self.vertices[(i+1)%3]
            
            # v3, v4 are virtual dot helpers for the square corners
            # We use _SmartPoint to create them as nodes in the graph
            sp3 = _SmartPoint(ORIGIN, label_text="", radius=0)
            sp4 = _SmartPoint(ORIGIN, label_text="", radius=0)
            self.add(sp3, sp4)
            
            v3_node = sp3.center_node
            v4_node = sp4.center_node
            
            # Windmill logic: Outward normal is Rotate -90 deg of p2-p1 (assuming CCW triangle)
            def v3_solver(target, parents):
                p1 = parents[0].get_live_data()
                p2 = parents[1].get_live_data()
                v = p2 - p1
                n = np.array([v[1], -v[0], 0]) 
                return p2 + n
                
            def v4_solver(target, parents):
                p1 = parents[0].get_live_data()
                p2 = parents[1].get_live_data()
                v = p2 - p1
                n = np.array([v[1], -v[0], 0])
                return p1 + n

            v3_node.add_constraint(v3_solver, [v1.center_node, v2.center_node])
            v4_node.add_constraint(v4_solver, [v1.center_node, v2.center_node])
            
            self.add_node(v3_node)
            self.add_node(v4_node)
            
            sq = SmartPolygon(v2, sp3, sp4, v1, color=color)
            self.add(sq)
            self.squares.append(sq)

    def add_area_labels(self, labels=["a^2", "b^2", "c^2"]):
        """Add labels to the center of squares"""
        res_group = VGroup()
        for i, sq in enumerate(self.squares):
            if i < len(labels):
                lbl = SmartLayoutUtils.safe_tex(labels[i])
                self.add(lbl)
                res_group.add(lbl)
                l_node = GeoNode(lbl, type=GeoNodeType.LABEL)
                
                # Gather all 4 vertices of the square
                parents = [v.center_node for v in sq.vertices]
                
                def center_solver(target, rents):
                    pts = [p.get_live_data() for p in rents]
                    return np.mean(pts, axis=0)
                    
                l_node.add_constraint(center_solver, parents)
                self.add_node(l_node)
                
                # [Fix] Immediate update
                res = center_solver(l_node, parents)
                if res is not None: lbl.move_to(res)

        return res_group

    def add_side_label(self, side_index=0, text="a", buff=0.3, color=WHITE):
        """[Delegation] Forward to core triangle"""
        if self.core_triangle:
            return self.core_triangle.add_side_label(side_index=side_index, text=text, buff=buff, color=color)
        return VMobject()

    def add_angle(self, index=0, label=None, radius=0.5, color=YELLOW, other_angle=False):
        """[Delegation] Forward to core triangle"""
        if self.core_triangle:
            # SmartTriangle uses add_angle_mark instead of add_angle for specialized behavior
            if hasattr(self.core_triangle, 'add_angle_mark'):
                return self.core_triangle.add_angle_mark(index=index, label=label, radius=radius, color=color)
            return self.core_triangle.add_angle(index=index, label=label, radius=radius, color=color, other_angle=other_angle)
        return VMobject()

    def add_point(self, coords, label_text="<Identifier>", color=WHITE):
        """[Infrastructure] Standard add_point for Windmill system"""
        dot = Dot(coords, color=color)
        self.add(dot)
        lbl = SmartLayoutUtils.safe_tex(label_text, color=color)
        self.add(lbl)
        lbl.next_to(dot, UR, buff=0.1)
        p_node = GeoNode(dot, type=GeoNodeType.POINT)
        self.add_node(p_node)
        return VGroup(dot, lbl)

    def add_line(self, start_node, end_node, color=WHITE):
        """[Infrastructure] Standard add_line for Windmill system"""
        # Convert list coords to static nodes if needed
        if isinstance(start_node, (list, tuple, np.ndarray)):
            p1 = start_node
        else:
            p1 = start_node.get_live_data()
            if hasattr(p1, 'get_center'): p1 = p1.get_center()
            
        if isinstance(end_node, (list, tuple, np.ndarray)):
            p2 = end_node
        else:
            p2 = end_node.get_live_data()
            if hasattr(p2, 'get_center'): p2 = p2.get_center()
            
        line = Line(p1, p2, color=color)
        self.add(line)
        l_node = GeoNode(line, type=GeoNodeType.EDGE)
        self.add_node(l_node)
        return line


class SmartAnnulus(TopoGraph):
    """
    [Recipe: Smart Annulus]
    Composite of two concentric circles or a filled Annulus.
    Driven by: center and two radius definitions (can be numbers or points).
    """
    def __init__(self, center=ORIGIN, r_in=1.0, r_out=2.0, color=BLUE_A):
        super().__init__()
        self.color = color
        
        # 1. Setup Center Node
        if hasattr(center, "center_node"):
            self.center_node = center.center_node
        else:
            # Create an invisible anchor point (radius=0, no label)
            self.center_pt = _SmartPoint(center, label_text="", radius=0)
            # [Structural Fix] Must add to VGroup so it follows LayoutManager shifts,
            # even if it's invisible.
            self.add(self.center_pt)
            self.add_node(self.center_pt.center_node)
            self.center_node = self.center_pt.center_node

        # 2. Setup Radius Dependencies
        # We store them so they can be accessed by labels
        self.r_in_ref = r_in
        self.r_out_ref = r_out
        
        # 3. Visual: Manim Annulus
        self.annulus = Annulus(inner_radius=1, outer_radius=2, color=color)
        self.add(self.annulus)
        
        # [Visual Fix] Sync origin immediately
        self.annulus.move_to(self.center_node.get_live_data())
        
        self.body_node = GeoNode(self.annulus, type=GeoNodeType.SURFACE)
        # 4. Connect dependencies
        parents = [self.center_node]
        if hasattr(r_in, "center_node"): parents.append(r_in.center_node)
        if hasattr(r_out, "center_node"): parents.append(r_out.center_node)

        def annulus_solver(target, rents):
            c = rents[0].get_live_data()
            
            # Determine r_in
            if hasattr(self.r_in_ref, "center_node"):
                p_in = self.r_in_ref.center_node.get_live_data()
                ri = np.linalg.norm(p_in - c)
            else:
                ri = self.r_in_ref
                
            # Determine r_out
            if hasattr(self.r_out_ref, "center_node"):
                p_out = self.r_out_ref.center_node.get_live_data()
                ro_visual = np.linalg.norm(p_out - c)
                # [Lens Fix] De-scale visual distance to get the logical radius
                ro = ro_visual / self.layout_scale
            else:
                ro = self.r_out_ref
            
            # [Lens Fix] Same for ri
            if hasattr(self.r_in_ref, "center_node"):
                p_in = self.r_in_ref.center_node.get_live_data()
                ri_visual = np.linalg.norm(p_in - c)
                ri = ri_visual / self.layout_scale
            else:
                ri = self.r_in_ref
            
            # Update live properties for labels to read (Logical values)
            self.r_in_val = ri
            self.r_out_val = ro
            
            # [Lens Fix] Recreate at LOGICAL scale 1.0
            new_ann = Annulus(inner_radius=ri, outer_radius=ro, color=self.color)
            
            # [Lens Fix] Apply the layout-aware scale lens
            if self.layout_scale != 1.0:
                new_ann.scale(self.layout_scale)
            
            # [CRITICAL FIX] Move the NEW object to visual center 'c' BEFORE become()
            new_ann.move_to(c)
                
            target.value.become(new_ann)
            return None

        self.body_node.add_constraint(annulus_solver, parents)
        self.add_node(self.body_node)

        # [Visual Fix] Force immediate update to prevent "popping" from logic 1.0 
        # to actual parameters during the first frame of animation.
        annulus_solver(self.body_node, parents)

    def get_visual_radii(self):
        """Returns (visual_inner, visual_outer) accounting for scale."""
        # Manim Annulus doesn't have simple width/2 for both.
        # However, it scales uniformly.
        scale = self.annulus.width / (2 * self.r_out_val) if self.r_out_val != 0 else 1.0
        return self.r_in_val * scale, self.r_out_val * scale

    def add_label(self, text="Annulus", direction=UP, buff=0.2, color=WHITE):
        """Label in the middle of the band"""
        lbl = SmartLayoutUtils.safe_tex(text, color=color)
        self.add(lbl)
        l_node = GeoNode(lbl, type=GeoNodeType.LABEL)
        
        def band_solver(target, rents):
            c = self.center_node.get_live_data()
            # Use visual radii
            ri_vis, ro_vis = self.get_visual_radii()
            r_mid = (ri_vis + ro_vis) / 2 + buff
            v = SmartLayoutUtils.normalize(np.array(direction))
            return c + v * r_mid
            
        l_node.add_constraint(band_solver, [self.center_node])
        self.add_node(l_node)
        
        # [Fix] Immediate update
        band_solver(l_node, [self.center_node])
        
        return lbl


class SmartInscribed(TopoGraph):
    
    def __init__(self, target: 'SmartPolygon' = None, shape="Circle", color=WHITE):
        super().__init__()
        if target is None:
            target = SmartTriangle([-2,-1,0], [2,-1,0], [0,2,0])
        
        # [LLM Hallucination Fix] Resolve string references
        target = SmartLayoutUtils.resolve_id(target)
        
        # 1. Resolve Vertices
        self.vertices = []
        if isinstance(target, SmartPolygon):
            self.vertices = target.vertices
        elif isinstance(target, (list, tuple)):
            for p in target:
                if hasattr(p, "center_node"):
                    self.vertices.append(p)
                else:
                    pt = _SmartPoint(p, label_text="", radius=0)
                    self.add_node(pt.center_node)
                    self.vertices.append(pt)
        
        # [Safety] Expose center_node early to prevent AttributeError in add_center_label
        self.center_node = GeoNode(None, type=GeoNodeType.VIRTUAL)

        if not self.vertices: return

        # 2. Visual: Circle
        self.circle = Circle(color=color)
        self.add(self.circle)
        self.shape_node = GeoNode(self.circle, type=GeoNodeType.EDGE)
        
        # [Safety] Expose center_node for add_center_label support
        self.center_node = GeoNode(None, type=GeoNodeType.VIRTUAL)
        self.add_node(self.center_node)
        
        parents = [v.center_node for v in self.vertices]
        
        # 3. Solver: Intelligent mode selection
        def universal_inscribed_solver(target, rents):
            pts = [p.get_live_data() for p in rents]
            n = len(pts)
            
            if n == 3:
                # --- Specific Incenter (Triangle) ---
                A, B, C = pts
                a = np.linalg.norm(B - C)
                b = np.linalg.norm(A - C)
                c_len = np.linalg.norm(A - B)
                perimeter = a + b + c_len
                if perimeter < 1e-6:
                    target.value.set_opacity(0)
                    return
                # Formula
                I = (a*A + b*B + c_len*C) / perimeter
                s = perimeter / 2
                area = np.sqrt(max(0, s * (s-a) * (s-b) * (s-c_len)))
                r = area / s if s > 0 else 0
                center, radius = I, r
            else:
                # --- Fallback: Centroid + Min Dist to edges ---
                center = np.mean(pts, axis=0)
                min_dist = float('inf')
                for i in range(n):
                    p1, p2 = pts[i], pts[(i+1)%n]
                    proj = SmartLayoutUtils.project_point_to_line(center, p1, p2)
                    d = np.linalg.norm(proj - center)
                    if d < min_dist: min_dist = d
                radius = min_dist
            
            # Update Circle
            target.value.set_opacity(1)
            target.value.move_to(center)
            target.value.set_width(2 * radius)
            
            # Update center_node value (for add_center_label)
            self.center_node.value = center
            return None

        self.shape_node.add_constraint(universal_inscribed_solver, parents)
        self.add_node(self.shape_node)

    def add_center_label(self, text="I", direction=UR, buff=0.35, color=WHITE):
        """
        Adds a label to the inscribed circle's center (Incenter).
        """
        lbl = SmartLayoutUtils.safe_tex(text, color=color)
        self.add(lbl)
        label_node = GeoNode(lbl, type=GeoNodeType.LABEL)

        def center_label_solver(target, rents):
            center_pos = rents[0].get_live_data()
            if hasattr(center_pos, 'get_center'):
                center_pos = center_pos.get_center()
            return center_pos + np.array(direction) * buff

        label_node.add_constraint(center_label_solver, [self.center_node])
        self.add_node(label_node)
        return lbl

    def add_radius_line(self, angle=30*DEGREES, color=WHITE, label=None):
        """
        Adds a radius line to the inscribed circle.
        """
        # Create a radius line object
        line = Line(ORIGIN, RIGHT, color=color)
        self.add(line)
        line_node = GeoNode(line, type=GeoNodeType.EDGE)
        
        # Virtual point on circumference
        p_node = GeoNode(None, type=GeoNodeType.VIRTUAL)
        self.add_node(p_node)

        def radius_solver(target, rents):
            # rent[0] is center, rent[1] is the circle itself (to get radius)
            c = rents[0].get_live_data()
            if hasattr(c, 'get_center'): c = c.get_center()
            if c is None: return ORIGIN
            
            circle_mob = rents[1].get_live_data()
            # [Safety] radius extraction
            r = 1.0
            if hasattr(circle_mob, 'width'):
                r = circle_mob.width / 2
            elif hasattr(circle_mob, 'get_width'):
                r = circle_mob.get_width() / 2
            
            p_pos = c + np.array([np.cos(angle), np.sin(angle), 0]) * r
            target.value = p_pos
            return p_pos

        p_node.add_constraint(radius_solver, [self.center_node, self.shape_node])
        
        # Use robust connect_points logic for the line
        line_node.add_constraint(Constraint.connect_points, [self.center_node, p_node])
        self.add_node(line_node)
        
        if label:
            # Re-implement label with repulsion for better positioning
            lbl = SmartLayoutUtils.safe_tex(label, color=color)
            self.add(lbl)
            label_node = GeoNode(lbl, type=GeoNodeType.LABEL)
            
            def radius_label_solver(target, rents):
                # Put label at midpoint with small offset
                c_pos = rents[0].get_live_data()
                p_pos = rents[1].get_live_data()
                if hasattr(c_pos, 'get_center'): c_pos = c_pos.get_center()
                if hasattr(p_pos, 'get_center'): p_pos = p_pos.get_center()
                if c_pos is None or p_pos is None: return ORIGIN
                
                mid = (c_pos + p_pos) / 2
                perp = np.array([-np.sin(angle), np.cos(angle), 0]) * 0.25
                return mid + perp
                
            label_node.add_constraint(radius_label_solver, [self.center_node, p_node])
            self.add_node(label_node)
            
        return line

    def add_side_label(self, side_index=0, text="a", buff=0.3, color=WHITE):
        """
        [Compatibility] Same API as SmartCircle.add_side_label.
        """
        # --- Step 1: 解析 side_index ---
        idx = side_index
        if isinstance(side_index, str):
            if side_index.startswith('e'):
                try: idx = int(side_index[1:])
                except ValueError: idx = 0
            else:
                try: idx = int(side_index)
                except ValueError: idx = 0

        # --- Step 2: 收集 EDGE 节点 ---
        edge_nodes = [n for n in self.nodes if n.type == GeoNodeType.EDGE]

        if not (isinstance(idx, int) and 0 <= idx < len(edge_nodes)):
            print(f"[Warning] SmartInscribed.add_side_label: side_index '{side_index}' out of range. Returning empty.")
            return VMobject()

        edge_node = edge_nodes[idx]
        
        # --- Step 3: 提取端点 ---
        s_node, e_node = None, None
        for solver, parents in edge_node.constraints:
            if len(parents) >= 2:
                s_node, e_node = parents[0], parents[1]
                break
        
        if s_node is None or e_node is None:
            return VMobject()

        # --- Step 4/5/6: Label and Repulsion ---
        actual_color = SmartLayoutUtils.parse_color(color)
        lbl = SmartLayoutUtils.safe_tex(text, color=actual_color)
        l_node = GeoNode(lbl, type=GeoNodeType.LABEL)
        
        circle_centroid = self.center_node # Already calculated
        
        def side_label_solver(target, parents):
            try:
                p1, p2, cent = [p.get_live_data() for p in parents]
                p1 = p1.get_center() if hasattr(p1, 'get_center') else p1
                p2 = p2.get_center() if hasattr(p2, 'get_center') else p2
                cent = cent.get_center() if hasattr(cent, 'get_center') else cent
            except:
                return target.value.get_center()
                
            mid = (p1 + p2) / 2
            v = p2 - p1
            if np.linalg.norm(v) < 1e-4: return mid
            
            perp = np.array([-v[1], v[0], 0])
            perp = SmartLayoutUtils.normalize(perp)
            
            pos1 = mid + perp * buff
            pos2 = mid - perp * buff
            dist1 = np.linalg.norm(pos1 - cent)
            dist2 = np.linalg.norm(pos2 - cent)
            
            final_dir = perp if dist1 > dist2 else -perp
            return mid + final_dir * buff
        
        l_node.add_constraint(side_label_solver, [s_node, e_node, circle_centroid])
        self.add_node(l_node)
        self.add(lbl)
        return lbl


class SmartTangentSystem(TopoGraph):
    """
    [Recipe: Smart Tangent System]
    A hybrid component that handles two distinct tangent scenarios:
    1. Tangent AT a specific point on the circle (1 line).
    2. Tangents FROM an external point (2 lines + T1/T2 + Chord).
    """
    def __init__(self, circle: 'SmartCircle' = None, at_point=None, from_point=None, length=6, color=YELLOW, show_chord=False):
        super().__init__()
        self.length = length
        self.color = color
        if circle is None:
            circle = SmartCircle(radius=2)
        
        # [Fix] Resolve string ID
        self.circle = SmartLayoutUtils.resolve_id(circle)
        self.show_chord_flag = show_chord
        self.mode = "AT_POINT" if at_point is not None else "FROM_POINT"
        
        self.tangent_group = VGroup()
        self.add(self.tangent_group)
        self.system_node = GeoNode(self.tangent_group, type=GeoNodeType.VIRTUAL)
        
        # --- MODE 1: Tangent AT a specific point ---
        if self.mode == "AT_POINT":
            if hasattr(at_point, "center_node"):
                self.ref_node = at_point.center_node
            else:
                self.ref_pt = _SmartPoint(at_point, label_text="", radius=0) # Invisible helper
                self.add_node(self.ref_pt.center_node)
                self.ref_node = self.ref_pt.center_node
                
            parents = [self.circle.center_node, self.ref_node]
            
            def single_tangent_solver(target, rents):
                c = rents[0].get_live_data()
                p = rents[1].get_live_data()
                
                # Setup
                tg = target.value
                tg.submobjects = []
                
                # Vector C -> P
                v = p - c
                dist = np.linalg.norm(v)
                if dist < 1e-3: return # Point at center, undefined tangent
                
                # Tangent direction (rotate 90 deg)
                # In 3D: cross product with Z-axis
                v_norm = v / dist
                tan_dir = np.array([-v_norm[1], v_norm[0], 0])
                
                # Line
                start = p - tan_dir * (self.length / 2)
                end = p + tan_dir * (self.length / 2)
                
                l = Line(start, end, color=self.color)
                tg.add(l)
                return None
                
            self.system_node.add_constraint(single_tangent_solver, parents)
            self.add_node(self.system_node)
            
            # [Fix] Immediate update
            single_tangent_solver(self.system_node, parents)

            
            # [Fix] Immediate update
            single_tangent_solver(self.system_node, parents)

            
            # For consistent API, expose get_tangent_points returning just the one point?
            self.tangent_points = [self.ref_node]

        # --- MODE 2: Tangents FROM an external point ---
        else:
            # Handle External Point
            if hasattr(from_point, "center_node"):
                self.ext_node = from_point.center_node
            elif from_point is not None:
                self.ext_pt = _SmartPoint(from_point, radius=0.06, color=color)
                self.add(self.ext_pt)
                self.ext_node = self.ext_pt.center_node
                self.add_node(self.ext_node)
            else:
                # Fallback or Error? 
                # Let's assume some default if neither provided, but better to require one.
                # Just create dummy external at Right
                self.ext_pt = _SmartPoint(RIGHT*4, radius=0.06, color=color)
                self.add(self.ext_pt)
                self.ext_node = self.ext_pt.center_node
                self.add_node(self.ext_node)

            # Virtual Nodes for T1 and T2
            self.t1_dot = Dot(radius=0.04, color=color)
            self.t2_dot = Dot(radius=0.04, color=color)
            
            self.t1_node = GeoNode(self.t1_dot, type=GeoNodeType.VERTEX)
            self.t2_node = GeoNode(self.t2_dot, type=GeoNodeType.VERTEX)
            
            # Solver
            parents = [self.circle.center_node, self.circle.radius_node, self.ext_node]
            
            def dual_tangent_solver(target, rents):
                c = rents[0].get_live_data()
                r = rents[1].get_live_data()
                p = rents[2].get_live_data()
                
                tg = target.value
                tg.submobjects = []
                
                tan_pts = SmartLayoutUtils.get_tangent_points_to_circle(p, c, r)
                if not tan_pts: 
                    target.value.set_opacity(0)
                    return 
                
                target.value.set_opacity(1)
                t1, t2 = tan_pts[0], tan_pts[1]
                
                # Update Nodes
                self.t1_node.set_value(t1) 
                self.t2_node.set_value(t2)
                
                # Visual Lines
                l1 = Line(p, t1, color=color)
                l2 = Line(p, t2, color=color)
                tg.add(l1, l2, self.t1_dot, self.t2_dot)
                
                if self.show_chord_flag:
                    chord = Line(t1, t2, color=color, stroke_opacity=0.5)
                    tg.add(chord)
                    
                return None

            self.system_node.add_constraint(dual_tangent_solver, parents)
            self.add_node(self.system_node)
            
            # [Fix] Immediate update
            dual_tangent_solver(self.system_node, parents)

            
            # [Fix] Immediate update
            dual_tangent_solver(self.system_node, parents)

            self.add_node(self.t1_node)
            self.add_node(self.t2_node)
            self.tangent_points = [self.t1_node, self.t2_node]
            
    def get_tangent_points(self):
        """Returns list of tangent point nodes."""
        return self.tangent_points


class SmartRegularPolygon(TopoGraph):
    """
    [Recipe: Flexible Regular Polygon]
    Unlike a fixed template, this is driven by TWO control points:
    1. center: The geometric center.
    2. first_vertex: Defines the radius and orientation.
    The component handles the "Regularity" constraint for the other N-1 vertices.
    """
    def __init__(self, n_sides=6, center=ORIGIN, first_vertex=RIGHT, color=BLUE):
        super().__init__()
        self.n = n_sides
        
        # 1. Setup Center
        if hasattr(center, "center_node"):
            self.center_node = center.center_node
        else:
            self.center_pt = _SmartPoint(center, radius=0.04, color=GRAY)
            self.add(self.center_pt)
            self.center_node = self.center_pt.center_node
            self.add_node(self.center_node)
            
        # 2. Setup First Vertex (The Primary Control)
        if hasattr(first_vertex, "center_node"):
            self.v1 = first_vertex
        else:
            self.v1 = _SmartPoint(first_vertex, color=color)
            self.add(self.v1)
        
        self.vertices = [self.v1]
        self.add_node(self.v1.center_node)
        
        # 3. Create N-1 Dependent Vertices (The geometric slaves)
        for i in range(1, n_sides):
            v_i = _SmartPoint(ORIGIN, radius=0.01, color=color)
            # Hide secondary dots by default to maintain "teacher randomness"
            v_i.dot.set_opacity(0) 
            self.add(v_i)
            self.vertices.append(v_i)
            self.add_node(v_i.center_node)
            
            # Rotation angle: i * (360/n)
            angle = i * (TAU / n_sides)
            
            # We use a closure here safely since angle is local to the loop
            def reg_vertex_solver(target, rents, ang=angle):
                c = rents[0].get_live_data()
                v1_pos = rents[1].get_live_data()
                # Use Math Kernel: Rotate v1 around center
                return SmartLayoutUtils.rotate_point(v1_pos, c, ang)
                
            v_i.center_node.add_constraint(reg_vertex_solver, [self.center_node, self.v1.center_node])

        # 4. Build the Polygon visual using the standard SmartPolygon
        # It handles the Edge lines and constraints automatically.
        self.poly_visual = SmartPolygon(*self.vertices, color=color)
        self.add(self.poly_visual)

    def add_circumcircle(self, color=WHITE, stroke_opacity=0.3):
        """Adds a helper circle passing through all vertices."""
        c_circle = Circle(color=color, stroke_opacity=stroke_opacity)
        self.add(c_circle)
        cc_node = GeoNode(c_circle, type=GeoNodeType.VIRTUAL)
        
        def circum_solver(target, rents):
            c = rents[0].get_live_data()
            v1_pos = rents[1].get_live_data()
            r = np.linalg.norm(v1_pos - c)
            target.value.move_to(c)
            target.value.set_width(2 * r)
            return None
            
        cc_node.add_constraint(circum_solver, [self.center_node, self.v1.center_node])
        self.add_node(cc_node)
        return c_circle

    def add_vertex_labels(self, labels=["A", "B", "C"], buff=0.4, color=WHITE):
        """
        [增量接口] 为各个顶点添加标签。
        逻辑与 SmartPolygon 保持一致，使用斥力算法自动避开多边形内部。
        """
        n = len(self.vertices)
        new_labels = VGroup()
        for i, text in enumerate(labels):
            if i >= n: break
            
            target_pt = self.vertices[i]
            target_pt.label_text = text 
            prev_pt = self.vertices[(i-1)%n]
            post_pt = self.vertices[(i+1)%n]
            
            lbl = SmartLayoutUtils.safe_tex(text, color=color)
            new_labels.add(lbl)
            
            l_node = GeoNode(lbl, type=GeoNodeType.LABEL)
            
            parents = [
                prev_pt.center_node, 
                target_pt.center_node, 
                post_pt.center_node, 
                self.center_node
            ]
            
            def poly_label_solver(target, rents):
                return Constraint.label_repulsion(target, rents, buff)
                
            l_node.add_constraint(poly_label_solver, parents)
            self.add_node(l_node)
            self.add(lbl)
            
            # [Fix] Immediate update
            result = poly_label_solver(l_node, parents)
            if result is not None:
                lbl.move_to(result)

        return new_labels

    def add_apothem(self, side_index=0, color=YELLOW, label="h"):
        """Adds a line from center to the midpoint of the specified side."""
        # --- Step 1: 解析 side_index ---
        idx = side_index
        if isinstance(side_index, str):
            if side_index.startswith('e'):
                try: idx = int(side_index[1:])
                except ValueError: idx = 0
            else:
                try: idx = int(side_index)
                except ValueError: idx = 0
        
        if not (isinstance(idx, int) and 0 <= idx < self.n):
            print(f"[Warning] SmartRegularPolygon.add_apothem: side_index '{side_index}' (resolved to {idx}) is out of range. Defaulting to 0.")
            idx = 0

        line = Line(ORIGIN, RIGHT, color=color)
        self.add(line)
        a_node = GeoNode(line, type=GeoNodeType.VIRTUAL)
        
        v_curr = self.vertices[idx].center_node
        v_next = self.vertices[(idx+1)%self.n].center_node
        
        a_node.add_constraint(Constraint.connect_points, [self.center_node, v_curr, v_next])
        self.add_node(a_node)
        
        if label:
            # Add dynamic label following the apothem midpoint
            res_group = VGroup(line)
            lbl = MathTex(label.strip('$').strip() or label, color=color, font_size=24)
            self.add(lbl)
            res_group.add(lbl)
            lbl_node = GeoNode(lbl, type=GeoNodeType.LABEL)
            def label_solver(target, rents):
                c = rents[0].get_live_data()
                p1 = rents[1].get_live_data()
                p2 = rents[2].get_live_data()
                mid = (p1 + p2) / 2
                vec = mid - c
                norm_vec = vec / (np.linalg.norm(vec) + 1e-6)
                target.value.move_to(mid - norm_vec * 0.25)
                return None
            lbl_node.add_constraint(label_solver, [self.center_node, v_curr, v_next])
            self.add_node(lbl_node)
            return res_group

        return line

    def add_center_label(self, text="O", direction=UR, buff=0.35, color=WHITE):
        """
        [增量接口] 为多边形中心添加标签。
        逻辑与 SmartCircle 保持一致。
        """
        lbl = SmartLayoutUtils.safe_tex(text, color=color)
        self.add(lbl)
        label_node = GeoNode(lbl, type=GeoNodeType.LABEL)

        def center_label_solver(target, rents):
            center_pos = rents[0].get_live_data()
            if hasattr(center_pos, 'get_center'):
                center_pos = center_pos.get_center()
            return center_pos + np.array(direction) * buff

        label_node.add_constraint(center_label_solver, [self.center_node])
        self.add_node(label_node)
        
        # [Fix] Immediate update
        res = center_label_solver(label_node, [self.center_node])
        if res is not None: lbl.move_to(res)
        
        return lbl

    def add_radius_line(self, angle=0, color=YELLOW, show_label=True, label_text="R"):
        """
        [增量接口] 添加外接圆半径线。
        Mimics SmartCircle.add_radius_line.
        Drawing from Center to a point at (R, angle).
        """
        line = Line(ORIGIN, RIGHT, color=color)
        self.add(line)
        r_node = GeoNode(line, type=GeoNodeType.EDGE)
        
        # Solver: Place line from Center to Center + R*angle_vec
        def radius_solver(target, rents):
            c = rents[0].get_live_data()
            if hasattr(c, 'get_center'): c = c.get_center()
            
            # Get R from first vertex distance
            v0 = rents[1].get_live_data()
            if hasattr(v0, 'get_center'): v0 = v0.get_center()
            
            # Distance from center to vertex is Circumradius R
            R = np.linalg.norm(v0 - c)
            if R < 1e-3: R = 1.0
            
            vec = np.array([np.cos(angle), np.sin(angle), 0])
            end = c + vec * R
            target.value.put_start_and_end_on(c, end)
            return None

        # Parents: [Center, Vertex[0]]
        # We assume vertex 0 exists (SmartRegularPolygon always has vertices)
        v0_node = self.vertices[0].center_node
        r_node.add_constraint(radius_solver, [self.center_node, v0_node])
        self.add_node(r_node)
        
        # Manually trigger once to set initial position
        # radius_solver(r_node, [self.center_node, v0_node])
        
        # Label
        if show_label and label_text:
            lbl = SmartLayoutUtils.safe_tex(label_text, color=color)
            self.add(lbl)
            l_node = GeoNode(lbl, type=GeoNodeType.LABEL)
            
            def r_label_solver(target, rents):
                l_obj = rents[0].value
                # r_node value is the line mobject
                line_obj = rents[1].value 
                s, e = line_obj.get_start(), line_obj.get_end()
                mid = (s + e) / 2
                
                # Direction of line
                direction = SmartLayoutUtils.normalize(e - s)
                # Perpendicular offset (rotate 90 deg around Z)
                perp = np.array([-direction[1], direction[0], 0])
                
                target.value.move_to(mid + perp * 0.25)
                return None
            
            l_node.add_constraint(r_label_solver, [l_node, r_node])
            self.add_node(l_node)

        return line

    def add_line(self, start_node, end_node, color=WHITE):
        """[Infrastructure] Standard add_line for RegularPolygon system"""
        # Convert list coords to static nodes if needed
        if isinstance(start_node, (list, tuple, np.ndarray)):
            p1 = start_node
        else:
            p1 = start_node.get_live_data()
            if hasattr(p1, 'get_center'): p1 = p1.get_center()
            
        if isinstance(end_node, (list, tuple, np.ndarray)):
            p2 = end_node
        else:
            p2 = end_node.get_live_data()
            if hasattr(p2, 'get_center'): p2 = p2.get_center()
            
        line_obj = _SmartLine(p1, p2, color=color)
        self.add(line_obj)
        
        # Track for add_side_label
        if not hasattr(self, "extra_lines"): self.extra_lines = []
        self.extra_lines.append(line_obj)
        
        return line_obj

    def add_side_label(self, side_index=0, text="a", buff=0.3, color=WHITE):
        """
        [增量接口] 为多边形边或额外线段添加标签。
        side_index: 
            - int/str: 0..n-1 -> Polygon Edge (Delegated to poly_visual)
            - str: 'e0' -> extra_line[0]
        """
        # Case 1: Extra Lines (e0, e1...) if handled by self.extra_lines
        if isinstance(side_index, str) and side_index.startswith('e'):
            try:
                idx = int(side_index[1:])
                if hasattr(self, "extra_lines") and 0 <= idx < len(self.extra_lines):
                    line = self.extra_lines[idx]
                    lbl = SmartLayoutUtils.safe_tex(text, color=color)
                    self.add(lbl)
                    
                    # Simple static placement for now (since extra lines are static usually)
                    s, e = line.get_start(), line.get_end()
                    mid = (s+e)/2
                    vec = e-s
                    perp = np.array([-vec[1], vec[0], 0])
                    perp = SmartLayoutUtils.normalize(perp)
                    lbl.move_to(mid + perp * buff)
                    return lbl
            except:
                pass
                
        # Case 2: Polygon Sides (Delegated to poly_visual or self.edges)
        # poly_visual is a SmartPolygon which fully supports add_side_label
        if hasattr(self, 'poly_visual'):
            return self.poly_visual.add_side_label(side_index, text, buff, color)
            
        return VMobject()
        
    def add_point(self, coords, label_text="<Identifier>", color=WHITE):
        """Standard add_point."""
        pt = _SmartPoint(coords, color=color)
        pt.label_text = label_text
        self.add(pt)
        if hasattr(self, 'add_node'): 
            self.add_node(pt.center_node) # If TopoGraph supports add_node
        return pt


class SmartTransversal(TopoGraph):
    """
    [Recipe: Smart Transversal System]
    Manages two lines and a transversal.
    Auto-calculates intersection points O1, O2.
    Supports easy angle labeling (Angles 1-8).
    """
    def __init__(self, line1=None, line2=None, transversal=None, color=WHITE):
        super().__init__()
        if line1 is None:
            line1 = _SmartLine(LEFT*3+UP, RIGHT*3+UP)
        if line2 is None:
            line2 = _SmartLine(LEFT*3+DOWN, RIGHT*3+DOWN)
        if transversal is None:
            transversal = _SmartLine(LEFT*2+DOWN*2, RIGHT*2+UP*2)
        
        def resolve_nodes(item):
            if hasattr(item, "line_node"):
                return item.p1.center_node, item.p2.center_node
            nodes = []
            for p in item:
                if hasattr(p, "center_node"): nodes.append(p.center_node)
                else:
                    sp = _SmartPoint(p, label_text="", radius=0)
                    self.add_node(sp.center_node)
                    nodes.append(sp.center_node)
            return nodes[0], nodes[1]

        self.l1_a, self.l1_b = resolve_nodes(line1)
        self.l2_a, self.l2_b = resolve_nodes(line2)
        self.ta, self.tb = resolve_nodes(transversal)

        # Intersection Dots
        self.o1_dot = Dot(radius=0.05, color=color)
        self.o2_dot = Dot(radius=0.05, color=color)
        self.add(self.o1_dot, self.o2_dot)
        
        self.o1_node = GeoNode(self.o1_dot, type=GeoNodeType.VERTEX)
        self.o2_node = GeoNode(self.o2_dot, type=GeoNodeType.VERTEX)
        
        def inter_solver(target, rents):
            p1, p2, p3, p4 = [r.get_live_data() for r in rents]
            res = SmartLayoutUtils.get_line_intersection(p1, p2, p3, p4)
            return res if res is not None else ORIGIN

        self.o1_node.add_constraint(inter_solver, [self.l1_a, self.l1_b, self.ta, self.tb])
        self.o2_node.add_constraint(inter_solver, [self.l2_a, self.l2_b, self.ta, self.tb])
        
        self.add_node(self.o1_node)
        self.add_node(self.o2_node)

    def add_angle(self, index=1, radius=0.4, color=YELLOW, label="\\alpha"):
        """
        Adds one of the 8 standard angles.
        O1 (Line 1): 1=UR, 2=UL, 3=LL, 4=LR
        O2 (Line 2): 5=UR, 6=UL, 7=LL, 8=LR
        """
        is_o1 = index <= 4
        v_node = self.o1_node if is_o1 else self.o2_node
        l_a = self.l1_a if is_o1 else self.l2_a
        l_b = self.l1_b if is_o1 else self.l2_b

        # Cleaner approach: Directly use parents in SmartAngle
        # We'll create a single GeoNode for the entire angle mark
        mark_group = VGroup()
        self.add(mark_group)
        mark_node = GeoNode(mark_group, type=GeoNodeType.VIRTUAL)
        
        parents = [v_node, l_a, l_b, self.ta, self.tb]
        
        def angle_mark_solver(target, rents, idx=index):
            o, la, lb, ta, tb = [r.get_live_data() for r in rents]
            dirs = [
                SmartLayoutUtils.normalize(la - o),
                SmartLayoutUtils.normalize(lb - o),
                SmartLayoutUtils.normalize(ta - o),
                SmartLayoutUtils.normalize(tb - o)
            ]
            # Sort CCW
            dirs.sort(key=lambda d: np.arctan2(d[1], d[0]))
            
            # Standard sector numbering (1-4)
            # We might want to "align" 1 with Upper Right.
            # Find the ray closest to (1,1)
            # For simplicity, we just use the sorted order.
            
            s_idx = (idx - 1) % 4
            d1 = dirs[s_idx]
            d2 = dirs[(s_idx + 1) % 4]
            
            a1 = np.arctan2(d1[1], d1[0])
            a2 = np.arctan2(d2[1], d2[0])
            diff = (a2 - a1) % TAU
            
            m = target.value
            m.submobjects = []
            arc = Arc(radius=radius, start_angle=a1, angle=diff, arc_center=o, color=color)
            m.add(arc)
            
            if label:
                lbl_tex = MathTex(label.strip('$').strip() or label, font_size=24, color=color)
                mid_a = a1 + diff/2
                lbl_pos = o + np.array([np.cos(mid_a), np.sin(mid_a), 0]) * (radius + 0.25)
                lbl_tex.move_to(lbl_pos)
                m.add(lbl_tex)
            return None

        mark_node.add_constraint(angle_mark_solver, parents)
        self.add_node(mark_node)
        return mark_group

    def get_intersections(self):
        """Returns the two intersection nodes."""
        return self.o1_node, self.o2_node


class SmartPolygonTransform(TopoGraph):
    """
    [Recipe: Smart Polygon Transform]
    A 'Master Component' for geometric rearrangement (Area Proofs).
    Manages multiple SmartPolygon pieces and provides a state-based system
    to move/rotate them collectively or individually.
    """
    def __init__(self, pieces: list = None, color=BLUE):
        super().__init__()
        if pieces is None:
            pieces = [SmartPolygon(_SmartPoint(ORIGIN), _SmartPoint(RIGHT), _SmartPoint(UP))]
        self.pieces = []
        
        # 1. Process Pieces
        for item in pieces:
            if isinstance(item, SmartPolygon):
                p = item
            else:
                # Assume list of points
                p = SmartPolygon(*item, color=color)
            
            self.pieces.append(p)
            self.add(p)
            # Register vertices to the main graph for syncing
            for v in p.vertices:
                self.add_node(v.center_node)

        # 2. State Management
        # We use GeoNodes to act as 'Transform Controllers' for each piece
        self.controllers = []
        for i, p in enumerate(self.pieces):
            # Each piece gets a controller node (storing [center_x, center_y, rotation_angle])
            c_node = GeoNode(np.array([0.0, 0.0, 0.0]), type=GeoNodeType.VIRTUAL)
            self.add_node(c_node)
            self.controllers.append(c_node)
            
            # Record original relative positions of vertices
            # We assume current position is 'home' (offset 0, rot 0)
            p._orig_pts = [v.center_node.get_live_data() for v in p.vertices]
            p._pivot = np.mean(p._orig_pts, axis=0)
            p._rel_pts = [pt - p._pivot for pt in p._orig_pts]

            # Bind vertices to the controller
            for v_idx, v in enumerate(p.vertices):
                def piece_vertex_solver(target, rents, p_idx=i, v_idx=v_idx):
                    ctrl_data = rents[0].get_live_data() # [dx, dy, angle]
                    offset = ctrl_data[:2]
                    angle = ctrl_data[2]
                    
                    p_obj = self.pieces[p_idx]
                    rel_pt = p_obj._rel_pts[v_idx]
                    
                    # Apply rotation around original pivot
                    rot_mat = np.array([
                        [np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle),  np.cos(angle), 0],
                        [0, 0, 1]
                    ])
                    rotated_rel = np.dot(rot_mat, rel_pt)
                    
                    # Final position = Original Pivot + Global Offset + Rotated Relative
                    # [Fix] Respect Layout Offset (if component was moved by LayoutManager)
                    layout_shift = self.layout_offset if hasattr(self, "layout_offset") else np.array([0., 0., 0.])
                    return p_obj._pivot + offset + rotated_rel + layout_shift

                v.center_node.add_constraint(piece_vertex_solver, [c_node])

    def set_piece_state(self, index, offset=ORIGIN, angle=0):
        """
        Instantly sets the position and rotation of a piece.
        offset: np.array([x, y, 0])
        angle: rotation in radians
        """
        if index < len(self.controllers):
            data = np.array([offset[0], offset[1], angle])
            self.controllers[index].set_live_data(data)
        return self

    def add_labels(self, labels=["P_1", "P_2", "P_3", "P_4"], buff=0.1):
        """
        Adds text labels to the centers of all pieces.
        labels: List of strings.
        """
        res_group = VGroup()
        for i, p in enumerate(self.pieces):
            if i < len(labels):
                clean_lbl = labels[i].strip('$').strip() or labels[i]
                lbl = MathTex(clean_lbl)
                self.add(lbl)
                res_group.add(lbl)
                l_node = GeoNode(lbl, type=GeoNodeType.LABEL)
                
                # Label follows the vertices of the piece
                parents = [v.center_node for v in p.vertices]
                def label_center_solver(target, rents):
                    pts = [pt.get_live_data() for pt in rents]
                    return np.mean(pts, axis=0)
                
                l_node.add_constraint(label_center_solver, parents)
                self.add_node(l_node)
        return res_group


class SmartSimilarSystem(TopoGraph):
    """
    [Recipe: Smart Similar System]
    Manages a base polygon and a similar copy.
    Automates similarity ratio tracking and labeling.
    """
    def __init__(self, base_poly: 'SmartPolygon' = None, scale_factor=0.5, offset=RIGHT*4, color=YELLOW):
        super().__init__()
        if base_poly is None:
            base_poly = SmartTriangle(_SmartPoint(ORIGIN), _SmartPoint(RIGHT), _SmartPoint(UP))
        
        # 1. Setup Base
        if hasattr(base_poly, "body_node"):
            self.base = base_poly
        elif isinstance(base_poly, (list, tuple)):
            # Create a polygon from vertex list
            self.base = SmartPolygon(*base_poly, color=color)
        else:
            # Fallback for unexpected types or empty inputs
            self.base = SmartTriangle(color=color)
        self.add(self.base)
        for v in self.base.vertices: self.add_node(v.center_node)

        # 2. Create Similar Copy
        # The copy's vertices are constrained to: Base_Pivot + Offset + (Base_Vertex - Base_Pivot) * Scale
        self.copy_vertices = []
        
        # Scale/Offset Controller (dx, dy, s)
        self.ctrl = GeoNode(np.array([offset[0], offset[1], scale_factor]), type=GeoNodeType.VIRTUAL)
        self.add_node(self.ctrl)

        # Base Centroid (Virtual Node) - [Fix] Define once before loop
        self.base_centroid = GeoNode(None)
        self.base_centroid.get_live_data = lambda: SmartLayoutUtils.get_centroid([v.center_node.get_live_data() for v in self.base.vertices])
        self.add_node(self.base_centroid)

        for i, v_base in enumerate(self.base.vertices):
            v_copy = _SmartPoint(ORIGIN, label_text="", radius=0.01)
            self.add(v_copy)
            self.copy_vertices.append(v_copy)
            self.add_node(v_copy.center_node)

            def similar_solver(target, rents, idx=i):
                # rents: [base_centroid, base_v, ctrl]
                c_base = rents[0].get_live_data()
                p_base = rents[1].get_live_data()
                ctrl_data = rents[2].get_live_data()
                
                off = ctrl_data[:2]
                s = ctrl_data[2]
                
                return c_base + off + (p_base - c_base) * s
            
            v_copy.center_node.add_constraint(similar_solver, [self.base_centroid, v_base.center_node, self.ctrl])

        self.copy_poly = SmartPolygon(*self.copy_vertices, color=color)
        self.add(self.copy_poly)

    def add_ratio_label(self, label_text="k", buff=0.5):
        """Displays the scale factor as a label between the two polygons."""
        lbl = MathTex(f"{label_text} = ", "1.0")
        self.add(lbl)
        l_node = GeoNode(lbl, type=GeoNodeType.LABEL)
        
        def ratio_solver(target, rents):
            c1 = rents[0].get_live_data()
            ctrl_data = rents[1].get_live_data()
            s = ctrl_data[2]
            off = ctrl_data[:2]
            
            # [Fix] Handle MathTex VGroup properly
            new_val_tex = MathTex(f"{s:.2f}", color=YELLOW)
            # If new_val_tex is VGroup, take first submobject; otherwise use directly
            if isinstance(new_val_tex, VGroup) and len(new_val_tex) > 0:
                new_val_mob = new_val_tex[0]
            else:
                new_val_mob = new_val_tex
            
            # Replace second submobject
            if len(target.value) > 1:
                target.value[1].become(new_val_mob)
            
            return c1 + off/2 + UP*buff
            
        l_node.add_constraint(ratio_solver, [self.base_centroid, self.ctrl])
        self.add_node(l_node)
        return lbl


class SmartSectorReform(TopoGraph):
    """
    [Recipe: Smart Sector Reform]
    Replaces legacy 'SectorGroup'. 
    Handles circle-to-parallelogram transformation for Area Proofs.
    """
    def __init__(self, radius=2, n_slices=8, mode="parallelogram", color=BLUE):
        super().__init__()
        self.r = radius
        self.n = n_slices
        self.mode = mode.lower()
        self.layout_scale = 1.0  # [Fix] Initialize layout_scale for visual_radius calculations
        self.color = color # Added for consistency with other classes
        
        self.slices = []
        self.sector_nodes = [] 
        angle = TAU / n_slices
        
        # [Architecture Fix] Use a Dot anchor so the center follows LayoutManager shifts.
        # This prevents the "Teleportation" bug where array nodes stay at world origin.
        self.center_anchor = Dot(radius=0, fill_opacity=0, stroke_opacity=0)
        self.add(self.center_anchor)
        self.center_node = GeoNode(self.center_anchor, type=GeoNodeType.POINT)
        self.add_node(self.center_node)
        
        # State: 0 = Circle, 1 = Rearranged
        self.state_node = GeoNode(0.0, type=GeoNodeType.VIRTUAL)
        self.add_node(self.state_node)
        
        # [Fix] To avoid rotation accumulation, we track the last applied state
        self.last_applied_states = [0.0] * n_slices
        # [Visual Fix] Tracking "rendered" state for per-slice damping
        self.rendered_states = [0.0] * n_slices

        for i in range(n_slices):
            s_color = self.color if i % 2 == 0 else GREEN
            sector = Sector(outer_radius=radius, inner_radius=0, angle=angle, start_angle=i*angle, color=s_color, fill_opacity=0.5)
            self.add(sector)
            self.slices.append(sector)
            
        # [Final Fix] Single robust updater for the whole component transformation
        def component_updater(mob):
            state = mob.state_node.get_live_data()
            center = mob.center_node.get_live_data()
            
            # Manual damping
            curr_rendered = getattr(mob, "_last_rendered_state", 0.0)
            render_state = curr_rendered + (state - curr_rendered) * 0.3
            mob._last_rendered_state = render_state
            
            # Tracking for rotation sync
            last_rots = getattr(mob, "_last_rots", [j * (TAU/mob.n) for j in range(mob.n)])
            
            angle = TAU / mob.n
            # [FIX] Use layout_scale-aware radius so reform targets match the scaled circle
            r_visual = mob.r * getattr(mob, 'layout_scale', 1.0)
            slice_w = r_visual * angle
            
            for i, slice_obj in enumerate(mob.slices):
                if mob.mode == "parallelogram":
                    row_x = (i - mob.n/2) * (slice_w * 0.5)
                    if i % 2 == 0:
                        t_pos = np.array([row_x, r_visual/2, 0]) + center
                        t_rot = -angle/2 - (i * angle) + PI
                    else:
                        t_pos = np.array([row_x, -r_visual/2, 0]) + center
                        t_rot = -angle/2 - (i * angle)
                elif mob.mode == "unroll":
                    row_x = (i - mob.n/2) * slice_w
                    t_pos = np.array([row_x, -r_visual/2, 0]) + center
                    t_rot = -angle/2 - (i * angle)
                else:
                    t_pos = center
                    t_rot = 0
                
                goal_pos = center * (1 - render_state) + t_pos * render_state
                slice_obj.shift(goal_pos - slice_obj.get_arc_center())
                
                # Rotation sync
                goal_rot = (i * angle) * (1 - render_state) + t_rot * render_state
                delta_rot = goal_rot - last_rots[i]
                slice_obj.rotate(delta_rot, about_point=goal_pos)
                last_rots[i] = goal_rot
            
            mob._last_rots = last_rots

        self.add_updater(component_updater)

    def reform(self, val=1.0):
        """Sets the reform state goal."""
        self.state_node.set_value(val)
        return self

    def get_visual_center(self):
        """Returns the actual visual center in world space."""
        return self.center_node.get_live_data()

    def get_visual_radius(self):
        """Returns the visual radius, accounting for layout_scale."""
        # Use getattr with default 1.0 to be safe, though layout_scale should exist
        return self.r * getattr(self, "layout_scale", 1.0)

    def add_label(self, text, direction=DOWN, buff=0.2, color=WHITE):
        """Adds a label to the reformed shape"""
        lbl = SmartLayoutUtils.safe_tex(text, color=color)
        self.add(lbl)
        label_node = GeoNode(lbl, type=GeoNodeType.LABEL)
        def label_solver(target, rents):
            # [FIX] Convert direction to array to allow multiplication with float radius/scale
            # And use get_visual_radius() to account for layout_scale automatically
            target.value.move_to(self.get_visual_center() + np.array(direction) * (self.get_visual_radius() + buff))
        label_node.add_constraint(label_solver, [self.center_node])
        self.add_node(label_node)
        
        # [Fix] Immediate update
        label_solver(label_node, [self.center_node])
        
        return lbl

    def add_radius_line(self, angle=0, color=YELLOW, show_label=True, label_text="r"):
        """
        Adds a radius line. For SmartSectorReform, we attach it to the center node.
        Returns the created _SmartLine object.
        """
        # We reuse the SmartCircle logic but adapted for this sector group
        radius_line = _SmartLine(ORIGIN, RIGHT * self.r, color=color)
        self.add(radius_line)
        
        # Track for add_side_label
        if not hasattr(self, "extra_lines"): self.extra_lines = []
        self.extra_lines.append(radius_line)
        
        def radius_solver(target, rents):
            # Start at center, end at arc boundary
            start = self.get_visual_center()
            end = start + SmartLayoutUtils.rotate_vector(RIGHT * self.r * self.layout_scale, angle)
            # [Safety]
            if hasattr(start, 'get_center'): start = start.get_center()
            if hasattr(end, 'get_center'): end = end.get_center()
            if start is None or end is None: return None
            
            target.value.put_start_and_end_on(start, end)
        
        radius_line.line_node.add_constraint(radius_solver, [self.center_node])
        self.add_node(radius_line.line_node)
        
        # [Fix] Immediate update
        radius_solver(radius_line.line_node, [self.center_node])
        
        if show_label:
            radius_line.add_length_label(label_text, color=color)
            
        return radius_line

    def add_point(self, coords, label_text="<Identifier>", color=WHITE):
        """[Infrastructure] Standard add_point for SectorReform system"""
        dot = Dot(coords, color=color)
        self.add(dot)
        lbl = SmartLayoutUtils.safe_tex(label_text, color=color)
        self.add(lbl)
        lbl.next_to(dot, UR, buff=0.1)
        p_node = GeoNode(dot, type=GeoNodeType.POINT)
        self.add_node(p_node)
        return VGroup(dot, lbl)

    def add_line(self, start_node, end_node, color=WHITE):
        """[Infrastructure] Standard add_line for SectorReform system"""
        # Convert list coords to static nodes if needed
        if isinstance(start_node, (list, tuple, np.ndarray)):
            p1 = start_node
        else:
            p1 = start_node.get_live_data()
            if hasattr(p1, 'get_center'): p1 = p1.get_center()
            
        if isinstance(end_node, (list, tuple, np.ndarray)):
            p2 = end_node
        else:
            p2 = end_node.get_live_data()
            if hasattr(p2, 'get_center'): p2 = p2.get_center()
            
        line_obj = _SmartLine(p1, p2, color=color)
        self.add(line_obj)
        
        # Track for add_side_label
        if not hasattr(self, "extra_lines"): self.extra_lines = []
        self.extra_lines.append(line_obj)
        
        return line_obj

    def add_side_label(self, side_index=0, text="a", buff=0.3, color=WHITE):
        """Adds a label to a line added via add_line or add_radius_line."""
        # 解析索引
        idx = side_index
        if isinstance(side_index, str):
            if side_index.startswith('e'):
                try: idx = int(side_index[1:])
                except ValueError: idx = 0
            else:
                try: idx = int(side_index)
                except ValueError: idx = 0

        extra_lines = getattr(self, "extra_lines", [])
        if not (isinstance(idx, int) and 0 <= idx < len(extra_lines)):
            print(f"[Warning] SmartSectorReform.add_side_label: index {side_index} out of range.")
            return VMobject()
            
        target_line = extra_lines[idx]
        return target_line.add_length_label(text, buff=buff, color=color)


class SmartConicSection(TopoGraph):
    """
    [Recipe: Smart Conic Section]
    Universal Ellipse/Hyperbola component.
    Driven by: focal distance c, semi-major axis a.
    Replaces legacy 'Ellipse' and 'Hyperbola'.
    """
    def __init__(self, c=2, a=3, type="Ellipse", color=WHITE):
        super().__init__()
        self.color = color
        self.type = type.lower()
        # [Config] 控制双曲线展开范围（t 的取值区间）
        self._hyperbola_t_max = 2.0
        # [Line Tracking] 跟踪所有通过 add_line 添加的线段，用于 add_side_label
        self.lines = []
        
        # 1. Center - Anchor for all calculations
        self.center_node = GeoNode(np.array(ORIGIN, dtype=np.float64), type=GeoNodeType.VERTEX)
        self.add_node(self.center_node, name="center")

        # 2. Controls - Virtual parameters (Pure logic, no shift)
        self.ctrl = GeoNode(np.array([c, a, 0.0], dtype=np.float64), type=GeoNodeType.VIRTUAL)
        self.add_node(self.ctrl, name="params")
        
        # 3. Focus Points
        self.f1_dot = Dot(color=color, radius=0.04)
        self.f2_dot = Dot(color=color, radius=0.04)
        self.add(self.f1_dot, self.f2_dot)
        
        self.f1_node = GeoNode(self.f1_dot, type=GeoNodeType.VERTEX)
        self.f2_node = GeoNode(self.f2_dot, type=GeoNodeType.VERTEX)
        # [Named Nodes] Register focus points
        self.add_node(self.f1_node, name="f1")
        self.add_node(self.f2_node, name="f2")
        
        def focus_solver(target, rents, sign=1):
            ctrl_data = rents[0].get_live_data()
            center_pos = rents[1].get_live_data()
            c_val = ctrl_data[0]
            # [Fix] Position must be relative to the anchor node
            return center_pos + np.array([sign * c_val, 0, 0])
            
        self.f1_node.add_constraint(lambda t, r: focus_solver(t, r, -1), [self.ctrl, self.center_node])
        self.f2_node.add_constraint(lambda t, r: focus_solver(t, r, 1), [self.ctrl, self.center_node])

        # 4. Curve（初始曲线）
        if self.type == "ellipse":
            # 椭圆：x^2/a^2 + y^2/b^2 = 1，b^2 = a^2 - c^2
            b = np.sqrt(max(1e-6, a**2 - c**2))
            self.curve = Ellipse(width=2*a, height=2*b, color=color)
        else:
            # 双曲线：x^2/a^2 - y^2/b^2 = 1，c^2 = a^2 + b^2
            b = np.sqrt(max(1e-6, c**2 - a**2))
            t_max = self._hyperbola_t_max

            def _right_branch(t):
                return np.array([a * np.cosh(t), b * np.sinh(t), 0])

            def _left_branch(t):
                return np.array([-a * np.cosh(t), b * np.sinh(t), 0])

            right = ParametricFunction(_right_branch, t_range=[-t_max, t_max], color=color)
            left = ParametricFunction(_left_branch, t_range=[-t_max, t_max], color=color)
            self.curve = VGroup(left, right)
        
        self.add(self.curve)
        self.curve_node = GeoNode(self.curve, type=GeoNodeType.EDGE)
        
        def curve_solver(target, rents):
            ctrl_data = rents[0].get_live_data()
            center_pos = rents[1].get_live_data()
            c_val, a_val = ctrl_data[0], ctrl_data[1]
            
            if self.type == "ellipse":
                # 椭圆：x^2/a^2 + y^2/b^2 = 1
                b_val = np.sqrt(max(1e-6, a_val**2 - c_val**2))
                new_curve = Ellipse(width=2*a_val, height=2*b_val, color=self.color)
            else:
                # 双曲线：x^2/a^2 - y^2/b^2 = 1
                b_val = np.sqrt(max(1e-6, c_val**2 - a_val**2))
                t_max = self._hyperbola_t_max

                def _right_branch(t):
                    return np.array([a_val * np.cosh(t), b_val * np.sinh(t), 0])

                def _left_branch(t):
                    return np.array([-a_val * np.cosh(t), b_val * np.sinh(t), 0])

                right = ParametricFunction(_right_branch, t_range=[-t_max, t_max], color=self.color)
                left = ParametricFunction(_left_branch, t_range=[-t_max, t_max], color=self.color)
                new_curve = VGroup(left, right)
                
            # [Fix] Align to anchor：曲线始终以 center_node 为中心
            new_curve.move_to(center_pos)
            target.value.become(new_curve)
            return None

        self.curve_node.add_constraint(curve_solver, [self.ctrl, self.center_node])
        self.add_node(self.curve_node)

    def set_params(self, c=None, a=None):
        """
        [Semantic API] Updates the focal distance c and semi-major axis a.
        """
        current_ctrl = self.ctrl.value
        new_c = c if c is not None else current_ctrl[0]
        new_a = a if a is not None else current_ctrl[1]
        
        self.set_node_local_data("params", [new_c, new_a, 0.0])
        return self

    def move_center(self, to):
        """
        [Semantic API] Moves the center of the conic section.
        """
        self.set_node_local_data("center", to)
        return self

    def add_focus_labels(self, labels=["F_1", "F_2"], color=WHITE):
        """Adds custom LaTeX labels to focus points."""
        new_labels = VGroup()
        for i, text in enumerate(labels):
            if i >= 2: break # Conic sections only have 2 foci
            dot = self.f1_dot if i == 0 else self.f2_dot
            # [Precision] Parse color string safely
            actual_color = SmartLayoutUtils.parse_color(color)
            lbl = SafeTex(text, color=actual_color).scale(0.8)
            self.add(lbl)
            new_labels.add(lbl)
            l_node = GeoNode(lbl, type=GeoNodeType.LABEL)
            f_node = self.f1_node if i == 0 else self.f2_node
            
            def label_solver(target, rents):
                p = rents[0].get_live_data()
                return p + DOWN * 0.3
            
            l_node.add_constraint(label_solver, [f_node])
            self.add_node(l_node)
        return new_labels

    def add_label(self, text="S", direction=UP, buff=0.5, color=WHITE):
        """Adds a centered LaTeX label to the conic section."""
        # [Precision] Parse color string safely
        actual_color = SmartLayoutUtils.parse_color(color)
        lbl = SafeTex(text, color=actual_color)
        l_node = GeoNode(lbl, type=GeoNodeType.LABEL)
        
        def label_solver(target, rents):
            c = rents[0].get_live_data()
            dir_vec = SmartLayoutUtils.get_direction_vec(direction) if isinstance(direction, str) else direction
            return c + dir_vec * buff
            
        l_node.add_constraint(label_solver, [self.center_node])
        self.add_node(l_node)
        self.add(lbl)
        
        # Immediate update
        res = label_solver(l_node, [self.center_node])
        if res is not None: lbl.move_to(res)
        return lbl

    def project_point_to_conic(self, point):
        """
        [Math Solver] Projects a 3D point onto the nearest location on the conic curve.
        For Ellipse: Uses a refined angular search.
        For Hyperbola: Uses parametric closest search.
        """
        # 1. Get current params
        c, a, _ = self.ctrl.get_live_data()
        center = self.center_node.get_live_data()
        
        # Localize point
        local_p = point - center
        
        if self.type == "ellipse":
            b = np.sqrt(max(1e-6, a**2 - c**2))
            
            # Parametric: x = a*cos(t), y = b*sin(t)
            # We want to find t that minimizes dist.
            # Angle method: atan2 for angle from center
            # Note: This is an approximation for non-circles, but accurate enough for visual snapping
            # Refined for ellipse geometry: t = atan2(a*y, b*x) helps map visual angle to parameter t
            t = np.arctan2(a * local_p[1], b * local_p[0])
            
            proj_pos = np.array([a * np.cos(t), b * np.sin(t), 0])
            return center + proj_pos
            
        else:
            # Hyperbola
            b = np.sqrt(max(1e-6, c**2 - a**2))
            
            # Determine branch
            sign = 1 if local_p[0] >= 0 else -1
            
            # Parametric: x = +/- a cosh t, y = b sinh t
            # y = b sinh t => t = asinh(y/b)
            t = np.arcsinh(local_p[1] / b)
            proj_x = a * np.cosh(t) * sign
            proj_y = b * np.sinh(t)
            
            return center + np.array([proj_x, proj_y, 0])

    def get_visual_center(self):
        """
        返回当前圆锥曲线在世界坐标中的可视中心。
        与其他组件的 get_visual_center 接口保持一致，供布局与标签使用。
        """
        return self.center_node.get_live_data()

    def add_angle(self, index, radius=0.4, color=YELLOW, label="\\alpha", other_angle=False):
        """
        [高效标注] 在系统的交角处自动生成角标。
        For Conic Section, 'index' usually refers to a point label (e.g., 'P').
        This method tries to intelligently guess which angle to mark based on context.
        Case 1: Reflection property (Angle between Focal Radius and Tangent/Normal).
        """
        # 1. Find the target node by name (registered in add_point)
        target_node = self.node_map.get(str(index))
        
        # Fallback: exact label matching
        if target_node is None:
            for node in self.nodes:
                if hasattr(node, "value") and isinstance(node.value, Mobject) and getattr(node.value, "tex_string", "") == index:
                     # This is a label node. We need the Vertex node it belongs to.
                     # Heuristic: Find a vertex node closest to this label? 
                     # Better: Just error out or look for "P" manually.
                     pass
        
        if target_node is None:
            # Fallback for "P" or "Q" if standard naming was used but not registered
            # We assume the user meant the last added vertex point if index matches
            pass

        if target_node is None:
            return VGroup() # Safer return to avoid Manim crash

        # 2. Setup Angle
        # Scene: Reflection Property.
        # We need: F1, F2, P (target_node).
        # We want to measure angle(F1, P, Tangent) vs angle(F2, P, Tangent)?
        # Or incident vs reflected relative to Normal? 
        # Standard: Incident = Angle(F1, P, Normal). Reflected = Angle(F2, P, Normal).
        
        angle_grp = VGroup()
        self.add(angle_grp)
        
        a_node = GeoNode(angle_grp, type=GeoNodeType.LABEL)
        
        def reflection_angle_solver(target, rents):
            # rents: [P_node, F1_node, F2_node, Ctrl, Center]
            p = rents[0].get_live_data()
            f1 = rents[1].get_live_data()
            f2 = rents[2].get_live_data()
            c_vals = rents[3].get_live_data() # [c, a, 0]
            center = rents[4].get_live_data()
            a, c_foc = c_vals[1], c_vals[0]
            b = np.sqrt(max(1e-6, a**2 - c_foc**2))
            
            # 1. Calculate Normal Vector at P
            # Gradient of x^2/a^2 + y^2/b^2 = 1 is (2x/a^2, 2y/b^2)
            # Normal vector n = (x/a^2, y/b^2)
            local_p = p - center
            nx = local_p[0] / (a**2 + 1e-6)
            ny = local_p[1] / (b**2 + 1e-6)
            normal = SmartLayoutUtils.normalize(np.array([nx, ny, 0]))
            
            # Tangent is perpendicular to normal
            tangent = np.array([-normal[1], normal[0], 0])
            
            # 2. Directions
            v_pf1 = SmartLayoutUtils.normalize(f1 - p)
            v_pf2 = SmartLayoutUtils.normalize(f2 - p)
            
            # 3. Determine which angle to draw
            # "other_angle=False" -> Incident (F1-P-Normal)
            # "other_angle=True"  -> Reflected (F2-P-Normal)
            # Actually, usually incidence is relative to Normal.
            # Let's draw angle from v_pf1 to Normal, and v_pf2 to Normal.
            
            if not other_angle:
                # Incident: P->F1 vs Normal
                # We want the angle between the ray FROM F1 (which is -v_pf1) and Normal?
                # Visual: Line P-F1 and Normal.
                v_start = v_pf1
                v_end = normal
            else:
                # Reflected: P->F2 vs Normal
                v_start = normal
                v_end = v_pf2
                
            # Create Arc
            # Calculate angles
            angle_start = np.arctan2(v_start[1], v_start[0])
            angle_end = np.arctan2(v_end[1], v_end[0])
            
            # Normalize delta angle
            delta = angle_end - angle_start
            while delta <= -PI: delta += TAU
            while delta > PI: delta -= TAU
            
            # Correct for "inner" angle vs reflex
            # We want the acute angle typically
            
            # Update Visual
            target.value.submobjects = []
            
            # [Fix] Use simple Sector/Arc based on start/end
            # Manim Arc: start_angle, angle
            
            # Visual Tweak: ensure we draw the "inside" angle relative to the tangent plane
            # If dot(v_start, v_end) < 0? 
            
            new_arc = Arc(radius=radius, start_angle=angle_start, angle=delta, color=color)
            target.value.add(new_arc)
            
            # Label
            if label:
                lbl = SmartLayoutUtils.safe_tex(label, color=color).scale(0.6)
                # Position: along bisector
                mid_angle = angle_start + delta/2
                lbl_pos = p + np.array([np.cos(mid_angle), np.sin(mid_angle), 0]) * (radius + 0.2)
                lbl.move_to(lbl_pos)
                target.value.add(lbl)
            
            return None

        # Bind
        parents = [target_node, self.f1_node, self.f2_node, self.ctrl, self.center_node]
        a_node.add_constraint(reflection_angle_solver, parents)
        self.add_node(a_node)
        
        # Init
        reflection_angle_solver(a_node, parents)
        return angle_grp

    def add_point(self, coords=[0, 0, 0], label_text="P", color=WHITE, constrain_to_curve=True):
        # [Precision Fix] Convert potential list/array to world points early
        raw_pos = self.get_visual_center() + np.array(coords)
        
        if constrain_to_curve:
            pos = self.project_point_to_conic(raw_pos)
        else:
            pos = raw_pos
            
        dot = Dot(pos, color=color, radius=0.08)
        self.add(dot)
        
        p_node = GeoNode(dot, type=GeoNodeType.VERTEX)
        
        # [Precision] Record the intended logical position for later line matching
        # This handles the 'Snap' case: the node knows its original coordinate.
        p_node.logical_pos = np.array(coords, dtype=np.float64)

        # [Crucial Restoration] Register node name so add_angle can find it
        name = label_text if label_text else f"pt_{len(self.nodes)}"
        clean_name = name.replace("$", "").replace("\\", "").strip() if name else None
        self.add_node(p_node, name=clean_name)
        
        if label_text:
            lbl = SafeTex(label_text, color=color)
            self.add(lbl)
            
            l_node = GeoNode(lbl, type=GeoNodeType.LABEL)
            def label_solver(target, rents):
                return rents[0].get_live_data() + UR * 0.15
            
            l_node.add_constraint(label_solver, [p_node])
            # Initial position
            lbl.move_to(label_solver(l_node, [p_node]))
            self.add_node(l_node)
            
            # [Requested Preservation] Return VGroup for animation
            return VGroup(dot, lbl)
            
        return dot

    def add_line(self, start_node=ORIGIN, end_node=RIGHT, color=BLUE, stroke_width=4):
        """
        [Override] 使用相对于圆锥曲线中心的坐标添加线段。
        原始坐标通过 get_visual_center() 偏移，与 add_point 保持坐标系一致。
        已注册的节点（如通过 add_point 创建的 P 点）会自动匹配，直接使用其世界坐标。
        """
        actual_color = SmartLayoutUtils.parse_color(color)
        center = self.get_visual_center()
        
        def _resolve(val):
            # 1. 尝试匹配已有节点（通过 name 或 logical_pos）
            node = self._resolve_node_or_coord(val)
            if not isinstance(node, (list, tuple, np.ndarray)):
                return node  # 是 GeoNode，直接使用其世界坐标
            # 2. 原始坐标：加上 visual_center 偏移（与 add_point 一致）
            return center + np.array(val, dtype=np.float64)
        
        s_input = _resolve(start_node)
        e_input = _resolve(end_node)
        
        line = _SmartLine(start_node=s_input, end_node=e_input, color=actual_color, stroke_width=stroke_width)
        if not isinstance(self, Scene):
            self.add(line)
        self.add_node(line.line_node)
        
        # 追踪线段，供 add_side_label 使用
        self.lines.append(line)
        return line

    def add_side_label(self, side_index=0, text="a", buff=0.3, color=WHITE):
        """
        [Master Feature] 给指定线段添加长度/名称标签。
        支持通过 side_index (字符串如 'e0', 'e1' 或整数索引) 来引用通过 add_line 添加的线段。
        """
        # 1. Resolve side_index to line object
        target_line = None
        
        if isinstance(side_index, str):
            # Handle string indices like 'e0', 'e1'
            if side_index.startswith('e'):
                try:
                    idx = int(side_index[1:])
                    if 0 <= idx < len(self.lines):
                        target_line = self.lines[idx]
                except ValueError:
                    pass
        elif isinstance(side_index, int):
            # Handle integer indices
            if 0 <= side_index < len(self.lines):
                target_line = self.lines[side_index]
        
        if target_line is None:
            # [Safety] Never return None for a method used in Play(Write(...))
            return VGroup()
        
        # 2. Color conversion: Use universal parser
        actual_color = SmartLayoutUtils.parse_color(color)
        
        # 3. Create the label
        lbl = SmartLayoutUtils.safe_tex(text, color=actual_color)
        l_node = GeoNode(lbl, type=GeoNodeType.LABEL)
        
        # 3. Centroid tracking (use center of conic section as reference)
        conic_centroid = GeoNode(None)
        conic_centroid.get_live_data = lambda: self.get_visual_center()
        self.nodes.append(conic_centroid)  # Virtual
        
        # 4. Solver: Centrifugal repulsion (similar to SmartPolygon)
        def side_label_solver(target, parents):
            try:
                p1, p2, cent = [p.get_live_data() for p in parents]
            except Exception:
                return target.value.get_center() # Safety
                
            mid = (p1 + p2) / 2
            v = p2 - p1
            # Standard perpendicular
            perp = np.array([-v[1], v[0], 0])
            perp = SmartLayoutUtils.normalize(perp)
            
            # Check direction vs centroid (outward bias)
            pos1 = mid + perp * buff
            pos2 = mid - perp * buff
            dist1 = np.linalg.norm(pos1 - cent)
            dist2 = np.linalg.norm(pos2 - cent)
            
            final_dir = perp if dist1 > dist2 else -perp
            return mid + final_dir * buff
        
        # Use the line's p1_node and p2_node
        l_node.add_constraint(side_label_solver, [target_line.p1_node, target_line.p2_node, conic_centroid])
        self.add_node(l_node)
        self.add(lbl)
        
        # [Fix] Manually trigger solver once to position label BEFORE animation
        result = side_label_solver(l_node, [target_line.p1_node, target_line.p2_node, conic_centroid])
        if result is not None:
            lbl.move_to(result)
        
        return lbl


class SmartLimitBox(TopoGraph):
    """
    [Recipe: Smart Limit Box]
    Visualizes a limit region (epsilon band).
    Replaces legacy 'LimitBand'.
    """
    def __init__(self, center_y=0, epsilon=0.5, width=8, color=YELLOW):
        super().__init__()
        self.width_val = width # [Fix] Save for closure
        
        # 1. State
        self.state = GeoNode(np.array([center_y, epsilon]), type=GeoNodeType.VIRTUAL)
        self.add_node(self.state)
        
        # 2. Visual Band
        self.rect = Rectangle(width=width, height=2*epsilon, color=color, fill_opacity=0.2, stroke_width=0)
        self.add(self.rect)
        
        # 3. Boundary Lines
        y_top = center_y + epsilon
        y_bot = center_y - epsilon
        self.line_top = Line(LEFT*width/2 + UP*y_top, RIGHT*width/2 + UP*y_top, color=color, stroke_width=2)
        self.line_bot = Line(LEFT*width/2 + UP*y_bot, RIGHT*width/2 + UP*y_bot, color=color, stroke_width=2)
        self.add(self.line_top, self.line_bot)
        
        v_node = GeoNode(self.rect, type=GeoNodeType.SURFACE)
        top_node = GeoNode(self.line_top, type=GeoNodeType.EDGE)
        bot_node = GeoNode(self.line_bot, type=GeoNodeType.EDGE)
        
        def limit_solver(target, rents, mode="rect"):
            cy, eps = rents[0].get_live_data()
            if mode == "rect":
                target.value.stretch_to_fit_height(max(1e-6, 2 * eps))
                target.value.move_to(np.array([0, cy, 0]))
            elif mode == "top":
                target.value.move_to(np.array([0, cy + eps, 0]))
            else:
                target.value.move_to(np.array([0, cy - eps, 0]))
            return None

        v_node.add_constraint(lambda t, r: limit_solver(t, r, "rect"), [self.state])
        top_node.add_constraint(lambda t, r: limit_solver(t, r, "top"), [self.state])
        bot_node.add_constraint(lambda t, r: limit_solver(t, r, "bot"), [self.state])
        
        self.add_node(v_node)
        self.add_node(top_node)
        self.add_node(bot_node)

    def add_epsilon_labels(self, label_text="\\epsilon", color=YELLOW):
        """Adds labels to the top and bottom boundaries."""
        res_group = VGroup()
        for mode in ["top", "bot"]:
            # Display L + eps / L - eps
            text = f"L + {label_text}" if mode == "top" else f"L - {label_text}"
            lbl = SafeTex(text, color=color).scale(0.7)
            self.add(lbl)
            res_group.add(lbl)
            
            l_node = GeoNode(lbl, type=GeoNodeType.LABEL)
            parents = [self.state]
            
            def label_solver(target, rents, m=mode):
                cy, eps = rents[0].get_live_data()
                y = cy + eps if m == "top" else cy - eps
                # Position to the right of the width
                return np.array([self.width_val/2 + 0.8, y, 0])

            l_node.add_constraint(label_solver, parents)
            self.add_node(l_node)
            
            # [Fix] Immediate update
            lbl.move_to(label_solver(l_node, parents))
            
        return res_group


class SmartRigidLink(TopoGraph):
    """
    [Recipe: Smart Rigid Link]
    Universal component for mechanical systems (Levers, Ladders, Pulleys).
    Maintains distance constraints between points.
    """
    def __init__(self, p1=ORIGIN, p2=RIGHT, length=2.0, color=WHITE):
        super().__init__()
        
        # 1. Resolve Points
        def resolve_pt(p):
            if hasattr(p, "center_node"): return p
            sp = _SmartPoint(p)
            self.add(sp)
            self.add_node(sp.center_node)
            return sp
            
        self.p1 = resolve_pt(p1)
        self.p2 = resolve_pt(p2)
        
        # 2. Determine Length
        self.len_val = length if length is not None else np.linalg.norm(self.p1.center_node.get_live_data() - self.p2.center_node.get_live_data())
        
        # 3. Visual Line
        start_pos = self.p1.center_node.get_live_data()
        end_pos = self.p2.center_node.get_live_data()
        self.line = Line(start_pos, end_pos, color=color)
        self.add(self.line)
        self.line_node = GeoNode(self.line, type=GeoNodeType.EDGE)
        
        self.line_node.add_constraint(Constraint.connect_points, [self.p1.center_node, self.p2.center_node])
        self.add_node(self.line_node)

    def add_pivot(self, point=ORIGIN, color=WHITE):
        """Adds a visual pivot point (grounded)."""
        pivot = Dot(point, color=color, radius=0.08)
        self.add(pivot)
        return pivot


class SmartSolid3D(TopoGraph):
    """
    [Recipe: Smart Solid 3D]
    Universal 3D shape wrapper.
    Replaces Cone3D, Cylinder, Sphere.
    """
    def __init__(self, type="Sphere", radius=1.0, height=2.0, color=BLUE):
        super().__init__()
        self.type = type.lower()
        self.radius = radius # Store radius
        self.height = height # Store height
        self.color = color # Store color
        
        # 1. State: [radius, height]
        self.state = GeoNode(np.array([radius, height]), type=GeoNodeType.VIRTUAL)
        self.add_node(self.state)
        
        # 2. Visual
        if self.type == "sphere":
            self.solid = Sphere(radius=radius, color=color)
        elif self.type == "cylinder":
            self.solid = Cylinder(radius=radius, height=height, color=color)
        elif self.type == "cone":
            self.solid = Cone(base_radius=radius, height=height, color=color)
            
        self.add(self.solid)
        self.solid_node = GeoNode(self.solid, type=GeoNodeType.SURFACE)
        
        def solid_solver(target, rents):
            r, h = rents[0].get_live_data()
            # Manim 3D objects are harder to 'become' vs 2D. 
            # We scale/stretch or recreate. Scaling is safer for simple solids.
            # But for radius vs height, we need specific stretch.
            
            # Recreate for accurate geometry
            if self.type == "sphere":
                new_s = Sphere(radius=r, color=self.color) # Use self.color
            elif self.type == "cylinder":
                new_s = Cylinder(radius=r, height=h, color=self.color) # Use self.color
            else: # cone
                new_s = Cone(base_radius=r, height=h, color=self.color) # Use self.color
                
            target.value.become(new_s)
            return None

        self.solid_node.add_constraint(solid_solver, [self.state])
        self.add_node(self.solid_node)

    def add_dimension_labels(self, r_label="r", h_label="h"):
        """Adds 3D-aware labels for dimensions."""
        # This would require 3D coordinate mapping.
        # For now, we expose the state nodes.
        return self.state


class SmartUnitCircle(TopoGraph):
    """
    [Recipe: Smart Unit Circle]
    Master component for Trigonometry.
    Tracks angle, sin, cos, and tan projections.
    """
    def __init__(self, radius=2.5, initial_angle=PI/4):
        super().__init__()
        self.radius = radius
        
        # 1. Background Layers (Static UI)
        self.circle = Circle(radius=radius, color=GRAY, stroke_opacity=0.3)
        self.axes = Axes(x_range=[-1.2, 1.2], y_range=[-1.2, 1.2], x_length=2.4*radius, y_length=2.4*radius)
        self.add(self.axes, self.circle)
        
        # 2. Actors (Dynamic Elements)
        cos_init = radius * np.cos(initial_angle)
        sin_init = radius * np.sin(initial_angle)
        p_init = np.array([cos_init, sin_init, 0])
        
        self.dot = Dot(p_init, color=YELLOW)
        self.cos_line = Line(ORIGIN, [cos_init, 0, 0], color=RED, stroke_width=6)
        self.sin_line = Line(ORIGIN, [0, sin_init, 0], color=BLUE, stroke_width=6)
        self.hypotenuse = Line(ORIGIN, p_init, color=WHITE, stroke_width=2)
        self.v_helper = DashedLine([cos_init, 0, 0], p_init, color=GRAY, stroke_opacity=0.5)
        self.h_helper = DashedLine([0, sin_init, 0], p_init, color=GRAY, stroke_opacity=0.5)
        
        # 3. Component State (Logical Nodes)
        # Note: We store INITIAL logic pos. 
        # Crucial: p_node here is just a DATA container.
        self.p_node = GeoNode(np.array([radius*np.cos(initial_angle), radius*np.sin(initial_angle), 0]), type=GeoNodeType.VERTEX)
        self.dot_node = GeoNode(self.dot, type=GeoNodeType.VERTEX)
        
        # Use add_node to handle Mobject registration and Topo tracking
        self.add_node(self.p_node)
        self.add_node(self.dot_node)
        # Add visual actors to the VGroup (Only once!)
        self.add(self.cos_line, self.sin_line, self.hypotenuse, self.v_helper, self.h_helper)
        
        # 4. Master Unified Updater (Absolute Physics Lock)
        def unified_updater(m):
            # A. Get Absolute World Anchor
            center = self.circle.get_center()
            # [FIX] Read actual visual radius from Circle geometry, respecting layout_scale
            r = self.circle.width / 2
            
            # B. Get World Intention
            target_world = self.p_node.get_live_data()
            vec = target_world - center
            
            # C. Safety Clamp (The Antidote for "Origin Snap")
            dist = np.linalg.norm(vec)
            if dist < 0.01:
                # If we've lost direction, force a default direction
                direction = RIGHT
            else:
                direction = vec / dist
                
            # D. Force Snap to Perimeter
            physics_dot_pos = center + direction * r
            self.dot.move_to(physics_dot_pos)
            
            # E. Redraw Projections based on the clamped Dot
            v_current = physics_dot_pos - center
            dx, dy = v_current[0], v_current[1]
            
            # [Safety] Helper for robust coordinate update
            def safe_put(mob, p1, p2):
                if hasattr(p1, 'get_center'): p1 = p1.get_center()
                if hasattr(p2, 'get_center'): p2 = p2.get_center()
                if p1 is None or p2 is None: return
                if np.allclose(p1, p2): p2 = p1 + [1e-4, 0, 0]
                mob.put_start_and_end_on(p1, p2)

            safe_put(self.cos_line, center, center + np.array([dx, 0, 0]))
            safe_put(self.sin_line, center, center + np.array([0, dy, 0]))
            safe_put(self.hypotenuse, center, physics_dot_pos)
            safe_put(self.v_helper, center + np.array([dx, 0, 0]), physics_dot_pos)
            safe_put(self.h_helper, center + np.array([0, dy, 0]), physics_dot_pos)
            
        self.add_updater(unified_updater)

    def add_trig_labels(self, labels=["\\sin", "\\cos", "\\tan"]):
        """Adaptive labels that track projection lines."""
        group = VGroup()
        for i, text in enumerate(labels):
            lbl = MathTex(text, font_size=24)
            def lbl_updater(l, idx=i):
                center = self.circle.get_center()
                dot_pos = self.dot.get_center()
                v = dot_pos - center
                if idx == 0: # cos (x)
                    l.move_to(center + np.array([v[0]/2, -0.4, 0]))
                    l.set_color(RED)
                else: # sin (y)
                    l.move_to(center + np.array([-0.6, v[1]/2, 0]))
                    l.set_color(BLUE)
            lbl.add_updater(lbl_updater)
            group.add(lbl)
        self.add(group)
        return self


class SmartVectorSystem(TopoGraph):
    """
    [Recipe: Smart Vector System]
    Handles vector operations: Addition, Decomposition.
    """
    def __init__(self, v1_end=None, v2_end=None, origin=None):
        super().__init__()
        
        # [Fix] Default values & Safe Copy to prevent modifying global constants (ORIGIN)
        if origin is None: origin = ORIGIN
        if v1_end is None: v1_end = RIGHT*2
        if v2_end is None: v2_end = UP*2
        
        # CRITICAL: Always copy inputs to new numpy arrays
        # This breaks reference to Manim's global ORIGIN/RIGHT/UP constants
        safe_origin = np.array(origin, dtype=float)
        safe_v1 = np.array(v1_end, dtype=float)
        safe_v2 = np.array(v2_end, dtype=float)

        # [Robustness] Prevent Arrow creation failure on zero-length vectors
        # This happens when provided v2 is [0,0,0] (e.g. in single vector demos)
        if np.linalg.norm(safe_v1 - safe_origin) < 1e-6:
            safe_v1 = safe_origin + np.array([1e-3, 0, 0])
        if np.linalg.norm(safe_v2 - safe_origin) < 1e-6:
            safe_v2 = safe_origin + np.array([0, 1e-3, 0])
        
        # 1. Base Nodes
        self.o_node = GeoNode(safe_origin, type=GeoNodeType.VERTEX)
        self.v1_node = GeoNode(safe_v1, type=GeoNodeType.VERTEX)
        self.v2_node = GeoNode(safe_v2, type=GeoNodeType.VERTEX)
        self.add_node(self.o_node)
        self.add_node(self.v1_node)
        self.add_node(self.v2_node)

        # 2. Visual Vectors
        # 2. Visual Vectors
        self.vec1 = Arrow(safe_origin, safe_v1, buff=0, color=BLUE)
        self.vec2 = Arrow(safe_origin, safe_v2, buff=0, color=RED)
        self.add(self.vec1, self.vec2)
        
        v1_edge = GeoNode(self.vec1, type=GeoNodeType.EDGE)
        v2_edge = GeoNode(self.vec2, type=GeoNodeType.EDGE)
        
        v1_edge.add_constraint(Constraint.connect_points, [self.o_node, self.v1_node])
        v2_edge.add_constraint(Constraint.connect_points, [self.o_node, self.v2_node])
        self.add_node(v1_edge)
        self.add_node(v2_edge)

    def add_sum_vector(self, color=RED, label="\\vec{s}", rule="parallelogram"):
        """Adds the sum vector (v1 + v2)."""
        sum_vec = Arrow(ORIGIN, RIGHT, buff=0, color=color)
        self.add(sum_vec)
        sum_node = GeoNode(sum_vec, type=GeoNodeType.EDGE)
        
        def sum_solver(target, rents):
            o = rents[0].get_live_data()
            v1_e = rents[1].get_live_data()
            v2_e = rents[2].get_live_data()
            if hasattr(o, 'get_center'): o = o.get_center()
            if hasattr(v1_e, 'get_center'): v1_e = v1_e.get_center()
            if hasattr(v2_e, 'get_center'): v2_e = v2_e.get_center()
            if o is None or v1_e is None or v2_e is None: return None
            
            # Vector sum: (v1-o) + (v2-o) + o = v1 + v2 - o
            try:
                end_pos = v1_e + v2_e - o
                if np.allclose(o, end_pos): end_pos = o + [1e-4, 0, 0]
                target.value.put_start_and_end_on(o, end_pos)
            except: pass
            return None

        sum_node.add_constraint(sum_solver, [self.o_node, self.v1_node, self.v2_node])
        self.add_node(sum_node)
        
        if rule == "parallelogram":
            # Add dashed helper lines
            o_pos = self.o_node.get_live_data()
            v1_pos = self.v1_node.get_live_data()
            v2_pos = self.v2_node.get_live_data()
            sum_pos = v1_pos + v2_pos - o_pos
            
            h1 = DashedLine(v1_pos, sum_pos, color=GRAY)
            h2 = DashedLine(v2_pos, sum_pos, color=GRAY)
            self.add(h1, h2)
            h1_node = GeoNode(h1, type=GeoNodeType.EDGE)
            h2_node = GeoNode(h2, type=GeoNodeType.EDGE)
            
            def helper_solver(target, rents, mode=1):
                o, v1, v2 = [r.get_live_data() for r in rents]
                if mode == 1: target.value.put_start_and_end_on(v1, v1 + v2 - o)
                else: target.value.put_start_and_end_on(v2, v1 + v2 - o)
                return None
            
            h1_node.add_constraint(lambda t, r: helper_solver(t, r, 1), [self.o_node, self.v1_node, self.v2_node])
            h2_node.add_constraint(lambda t, r: helper_solver(t, r, 2), [self.o_node, self.v1_node, self.v2_node])
            self.add_node(h1_node)
            self.add_node(h2_node)
            
        return sum_vec

    def add_label(self, text, x_val=None, direction=UP, buff=0.2, color=WHITE, font_size=24):
        """Add a label to the vector system, optionally aligned to a specific X coordinate."""
        actual_color = SmartLayoutUtils.parse_color(color)
        
        # Handle string direction (e.g. 'UP' -> UP)
        if isinstance(direction, str):
            d_map = {'UP': UP, 'DOWN': DOWN, 'LEFT': LEFT, 'RIGHT': RIGHT}
            d_vec = d_map.get(direction.upper(), UP)
        else:
            d_vec = np.array(direction)

        lbl = Tex(text, font_size=font_size, color=actual_color)
        
        # Position logic
        if x_val is not None:
             # Use the system's bounding box to determine Y level
             top_y = self.get_top()[1]
             bottom_y = self.get_bottom()[1]
             
             target_y = top_y if d_vec[1] >= 0 else bottom_y
             
             # Create position at [x_val, target_y, 0]
             pos = np.array([float(x_val), target_y, 0])
             
             # Shift label so its edge touches the target position + buff
             if d_vec[1] >= 0: # Above
                 lbl.move_to(pos + UP * (lbl.height/2 + buff))
             else: # Below
                 lbl.move_to(pos + DOWN * (lbl.height/2 + buff))
        else:
             # Default: Next to the entire bounding box
             lbl.next_to(self, np.array(d_vec), buff=buff)
             
        self.add(lbl)
        return lbl

    def add_angle(self, index='O', radius=0.5, color=YELLOW, label=None):
        """
        Adds an angle mark between the two vectors at the origin.
        index: Expected 'O', but we default to the angle between v1 and v2 at origin.
        """
        actual_color = SmartLayoutUtils.parse_color(color)

        # 1. Create Mobject Container
        mark_group = VGroup()
        self.add(mark_group)
        
        # 2. Create Virtual Node for Mark
        mark_node = GeoNode(mark_group, type=GeoNodeType.VIRTUAL)
        
        # Parents: v1 -> O -> v2
        parents = [self.v1_node, self.o_node, self.v2_node]
        
        def angle_solver(target, rents):
            p_v1, p_o, p_v2 = [p.get_live_data() for p in rents]
            
            # Vectors from O
            v1 = p_v1 - p_o
            v2 = p_v2 - p_o
            
            # Robustness: Check for zero length
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                return None
            
            # Kernel Algo
            is_right, _ = SmartLayoutUtils.get_angle_properties(v1, v2)
            
            m = target.value
            m.submobjects = [] # Clear children
            
            # Dynamic Radius
            safe_r = min(radius, n1*0.4, n2*0.4)
            
            if is_right:
                # Draw square
                perp1 = SmartLayoutUtils.normalize(v1) * safe_r
                perp2 = SmartLayoutUtils.normalize(v2) * safe_r
                corners = [
                    p_o + perp1,
                    p_o + perp1 + perp2,
                    p_o + perp2
                ]
                elbow = VMobject().set_points_as_corners(corners)
                elbow.set_stroke(actual_color, 2)
                m.add(elbow)
            else:
                # Draw Arc
                angle_v1 = np.arctan2(v1[1], v1[0])
                angle_v2 = np.arctan2(v2[1], v2[0])
                
                # Normalize angle diff to be interior angle <= PI
                diff = (angle_v2 - angle_v1) % TAU
                start = angle_v1
                if diff > PI:
                    diff = TAU - diff
                    start = angle_v2
                
                arc = Arc(radius=safe_r, start_angle=start, angle=diff, arc_center=p_o, color=actual_color)
                m.add(arc)
                
                if label:
                    # Place label at bisector
                    mid_angle = start + diff/2
                    # Position: slightly further out
                    lbl_pos = p_o + np.array([np.cos(mid_angle), np.sin(mid_angle), 0]) * (safe_r + 0.25)
                    
                    clean_label = str(label).strip('$').strip()
                    if not clean_label: clean_label = label
                    
                    tex = MathTex(clean_label, font_size=24, color=actual_color).move_to(lbl_pos)
                    m.add(tex)
            
            return None

        mark_node.add_constraint(angle_solver, parents)
        self.add_node(mark_node)
        
        # Trigger once
        angle_solver(mark_node, parents)
        
        return mark_group


class SmartStatSystem(TopoGraph):
    """
    [Recipe: Smart Statistics System]
    ── 柱状图 / 条形图组件，内置 Axes 坐标轴映射 ──
    利用 Axes.c2p() 把原始数值自动归一化到指定的 width × height
    视觉区域内，确保任何量级的数据都不会溢出屏幕。

    Constructor:
        SmartStatSystem(data=[3,5,2,8,4], width=6, height=4, color=BLUE)

    Methods (Path B):
        .add_mean_line(color=YELLOW, label="\\mu")
        .add_point(coords=[x, y, 0], label_text="P", color=WHITE)
        .add_line(start_node=[x1,y1,0], end_node=[x2,y2,0], color=BLUE)
        .set_data(new_data)
    """
    def __init__(self, data=None, width=6, height=4, color=BLUE):
        super().__init__()
        self.lines = []  # Initialize lines list
        if data is None:
            data = [3, 5, 2, 8, 4]
        self.data_arr = np.array(data, dtype=float)
        self._width = width
        self._height = height
        n = len(data)

        # ── 1. Y 轴范围：自动计算"好看"的刻度步长 ──
        max_val = float(max(data)) if max(data) > 0 else 1.0
        raw_step = max_val / 5
        mag = 10 ** np.floor(np.log10(max(raw_step, 1e-9)))
        r = raw_step / mag
        nice = 1 if r <= 1 else (2 if r <= 2 else (5 if r <= 5 else 10))
        y_step = float(nice * mag)
        y_max = float(np.ceil(max_val * 1.15 / y_step) * y_step)
        if y_max <= 0:
            y_max = y_step

        y_ticks = []
        v = y_step
        while v <= y_max + 0.001:
            y_ticks.append(round(v, 6))
            v += y_step
        if not y_ticks:
            y_ticks = [y_step]

        # ── 2. 坐标轴 ──
        self.axes = Axes(
            x_range=[0, n, 1],
            y_range=[0, y_max, y_step],
            x_length=width,
            y_length=height,
            axis_config={"color": GREY_B, "include_tip": False, "stroke_width": 2},
            x_axis_config={"numbers_to_include": []},
            y_axis_config={
                "numbers_to_include": y_ticks,
                "font_size": 20,
                "decimal_number_config": {"num_decimal_places": 0},
            },
        )
        self.add(self.axes)

        # ── 3. 数据状态节点 ──
        self.data_node = GeoNode(self.data_arr, type=GeoNodeType.VIRTUAL)
        self.add_node(self.data_node)

        # ── 4. 柱状条 ──
        self.bars = VGroup()
        palette = [PURPLE_B, TEAL_C, GOLD_C, PINK, GREEN_C,
                   ORANGE, RED_C, BLUE_C, MAROON_C, YELLOW_C]
        
        # [Robustness] Safe comparison for ManimColor vs str
        parsed_color = SmartLayoutUtils.parse_color(color)
        try:
            use_palette = (parsed_color == BLUE)
        except TypeError:
            use_palette = False
            
        bar_color_base = parsed_color

        self._value_labels = VGroup()
        self._idx_labels = VGroup()

        for i in range(n):
            val = float(data[i])
            xc = i + 0.5  # 柱子中心 (数据坐标)

            # 柱子左右边界的世界 x 坐标
            xl = self.axes.c2p(xc - 0.35, 0)[0]
            xr = self.axes.c2p(xc + 0.35, 0)[0]
            yb = self.axes.c2p(0, 0)[1]
            yt = self.axes.c2p(0, val)[1]

            bw = xr - xl
            bh = max(0.01, yt - yb)
            bc = palette[i % len(palette)] if use_palette else bar_color_base

            bar = Rectangle(
                width=bw, height=bh,
                fill_color=bc, fill_opacity=0.8,
                stroke_color=WHITE, stroke_width=2,
            )
            bar.move_to(np.array([(xl + xr) / 2.0, yb + bh / 2.0, 0]))
            self.bars.add(bar)

            # 数值标签
            fmt = str(int(val)) if val == int(val) else f"{val:.1f}"
            vlbl = Tex(f"$${fmt}$$", font_size=20, color=WHITE)
            vlbl.next_to(bar, UP, buff=0.08)
            self._value_labels.add(vlbl)

            # X 轴索引标签
            ilbl = Tex(f"$${i + 1}$$", font_size=18, color=GREY_B)
            ilbl.move_to(self.axes.c2p(xc, 0) + DOWN * 0.25)
            self._idx_labels.add(ilbl)

            # 约束节点
            b_node = GeoNode(bar, type=GeoNodeType.SURFACE)
            self.add_node(b_node)

        self.add(self.bars)
        self.add(self._value_labels)
        self.add(self._idx_labels)

        center_pos = np.array(self.axes.c2p(n / 2.0, y_max / 2.0), dtype=np.float64)
        self.center_node = GeoNode(center_pos, type=GeoNodeType.VIRTUAL)
        self.add_node(self.center_node, name="center")

    def add_label(self, text, direction=UP, buff=0.2, color=WHITE, font_size=24):
        """Add a general label to the chart (e.g., title or summary)."""
        actual_color = SmartLayoutUtils.parse_color(color)
        
        # Handle string direction (e.g. 'UP' -> UP)
        d_vec = direction
        if isinstance(direction, str):
            d_map = {'UP': UP, 'DOWN': DOWN, 'LEFT': LEFT, 'RIGHT': RIGHT}
            d_vec = d_map.get(direction.upper(), UP)

        lbl = Tex(text, font_size=font_size, color=actual_color)
        
        # Position relative to the bars (visual content) or axes
        target = self.bars if len(self.bars) > 0 else self.axes
        lbl.next_to(target, d_vec, buff=buff)
        
        self.add(lbl)
        return lbl

    # ───────────────── add_mean_line ───────────────── #

        self.add(lbl)
        return lbl

    def add_side_label(self, side_index="e0", text="L", buff=0.2, color=WHITE, font_size=24):
        """给指定的线段（统计线、趋势线）添加标签。"""
        # 1. Resolve index
        idx = -1
        if isinstance(side_index, int):
            idx = side_index
        elif isinstance(side_index, str) and side_index.startswith('e'):
             try: idx = int(side_index[1:])
             except: pass
        
        if idx < 0 or idx >= len(self.lines):
            return VMobject()
            
        line_obj = self.lines[idx]
        actual_color = SmartLayoutUtils.parse_color(color)
        
        # 2. Extract start/end points
        start_pt, end_pt = None, None
        
        if hasattr(line_obj, 'p1_node') and hasattr(line_obj, 'p2_node'):
             # _SmartLine
             start_pt = line_obj.p1_node.get_live_data()
             end_pt = line_obj.p2_node.get_live_data()
        elif hasattr(line_obj, 'line_node'):
             # Wrapper around DashedLine or Line
             mobj = line_obj.line_node.value
             if hasattr(mobj, 'get_start') and hasattr(mobj, 'get_end'):
                 start_pt = mobj.get_start()
                 end_pt = mobj.get_end()
        
        if start_pt is None or end_pt is None:
            return VMobject()
            
        # 3. Calculate position (Middle + Perpendicular)
        mid = (start_pt + end_pt) / 2
        v = end_pt - start_pt
        if np.linalg.norm(v) < 1e-6:
            direction = UP
        else:
            # Perpendicular vector [-dy, dx]
            perp = np.array([-v[1], v[0], 0])
            direction = SmartLayoutUtils.normalize(perp)
            # Heuristic: Bias generally right/up unless specified
            # For horizontal lines (v[1]=0), perp is vertical.
            # Default to UP if purely horizontal
            if abs(v[1]) < 0.01: 
                direction = UP if direction[1] > 0 else -direction
            # For vertical lines (v[0]=0), perp is horizontal.
            elif abs(v[0]) < 0.01:
                direction = RIGHT if direction[0] > 0 else -direction
            else:
                 # General bias
                 pass

        lbl = Tex(text, font_size=font_size, color=actual_color)
        lbl.move_to(mid + direction * buff)
        self.add(lbl)
        return lbl

    # ───────────────── add_mean_line ───────────────── #
    def add_mean_line(self, color=YELLOW, label="\\mu"):
        """在平均值高度画一条虚线。"""
        actual_color = SmartLayoutUtils.parse_color(color)
        n = len(self.data_arr)
        m_val = float(np.mean(self.data_arr))

        lp = np.array(self.axes.c2p(-0.3, m_val))
        rp = np.array(self.axes.c2p(n + 0.3, m_val))

        m_line = DashedLine(lp, rp, color=actual_color,
                            stroke_width=3, dash_length=0.15)
        self.add(m_line)

        if label:
            clean = str(label).replace("$$", "").strip()
            lbl = Tex(f"$${clean}$$", font_size=22, color=actual_color)
            lbl.next_to(m_line, RIGHT, buff=0.15)
            self.add(lbl)

        m_node = GeoNode(m_line, type=GeoNodeType.EDGE)
        self.add_node(m_node)
        self.lines.append(type('obj', (object,), {'line_node': m_node})())
        return m_line

    # ───────────────── add_point (override) ───────────────── #
    def add_point(self, coords=[0, 0, 0], label_text="P",
                  color=WHITE, radius=0.08):
        """在数据坐标系中标注一个点（通过 axes.c2p 映射到视觉位置）。"""
        actual_color = SmartLayoutUtils.parse_color(color)
        c = list(coords)
        if len(c) == 2:
            c = c + [0]
        world_pos = np.array(self.axes.c2p(c[0], c[1]), dtype=np.float64)

        pt = _SmartPoint(coords=world_pos, label_text=label_text,
                         color=actual_color, radius=radius)
        pt.center_node.logical_pos = np.array(c[:2], dtype=np.float64)

        if not isinstance(self, Scene):
            self.add(pt)
        name = (label_text.replace("$", "").replace("\\", "").strip()
                if label_text else None)
        self.add_node(pt.center_node, name=name)
        return pt

    # ───────────────── add_line (override) ───────────────── #
    def add_line(self, start_node=ORIGIN, end_node=RIGHT,
                 color=BLUE, stroke_width=4):
        """在数据坐标系中连线（通过 axes.c2p 映射到视觉位置）。"""
        actual_color = SmartLayoutUtils.parse_color(color)

        def _to_world(val):
            if isinstance(val, (list, tuple, np.ndarray)):
                c = list(val)
                if len(c) == 2:
                    c = c + [0]
                return np.array(self.axes.c2p(c[0], c[1]), dtype=np.float64)
            return val  # GeoNode / _SmartPoint 直接传递

        s = _to_world(start_node)
        e = _to_world(end_node)

        line = _SmartLine(start_node=s, end_node=e,
                          color=actual_color, stroke_width=stroke_width)
        if not isinstance(self, Scene):
            self.add(line)
        self.add_node(line.line_node)
        self.lines.append(line)
        return line

    # ───────────────── set_data ───────────────── #
    def set_data(self, new_data):
        """动态更新数据。"""
        self.data_node.set_live_data(np.array(new_data, dtype=float))
        return self


# =====================================================================
# 🛡️ 5. Safety Patches (Monkey Patches & Overrides)
# =====================================================================

class SafeTex(MathTex):
    """
    [Drop-in Replacement for MathTex/Tex]
    Automatically sanitizes input for CJK and Unicode compliance.
    Used to intercept direct Tex() calls from LLM to prevent LaTeX errors.
    """
    def __init__(self, *tex_strings, **kwargs):
        # 1. Flatten input
        text = " ".join([str(t) for t in tex_strings])
        
        # 2. Sanitize using SmartLayoutUtils logic (Inline version)
        if text:
            processed = str(text).strip()
            
            # [CRITICAL FIX] LaTeX Redundancy Protection
            # Remove manual math delimiters added by LLM (e.g. $$, \[) 
            # to prevent clashing with Manim's internal align* environment.
            if processed.startswith("$$") and processed.endswith("$$"):
                 processed = processed[2:-2].strip()
            if processed.startswith("$") and processed.endswith("$"):
                 processed = processed[1:-1].strip()
            if processed.startswith("\\[") and processed.endswith("\\]"):
                 processed = processed[2:-2].strip()
            if processed.startswith("\\(") and processed.endswith("\\)"):
                 processed = processed[2:-2].strip()
                 
            # 2.1 Unicode Replacement
            unicode_replacements = {
                '²': '^2', '³': '^3', '⁴': '^4', '⁵': '^5',
                '₀': '_0', '₁': '_1', '₂': '_2', '₃': '_3',
                '×': r'\times ', '÷': r'\div ', '±': r'\pm ',
                '≤': r'\le ', '≥': r'\ge ', '≠': r'\ne ',
                '→': r'\to ', '←': r'\leftarrow ', '∞': r'\infty ',
                'π': r'\pi ', 'θ': r'\theta ', 'α': r'\alpha ', 'β': r'\beta ',
                '√': r'\sqrt', 'Δ': r'\Delta ', '∑': r'\sum ', '∫': r'\int ',
            }
            for uni, latex in unicode_replacements.items():
                processed = processed.replace(uni, latex)
            
            # 2.2 CJK Wrapping (Smart)
            # 算法专家视角：大模型生成的 JSON 中常常带有 `\\text{}` (转义的反斜杠)，
            # 到达这里时可能表现为原始字符串 `\text{}` 或者 `\\text{}`。
            # 避免对已被 `\text{...}` 或 `\\text{...}` 包裹的中文进行重复包裹。
            if re.search(r'[\u4e00-\u9fff]', processed):
                # 如果已经存在文本包裹标记（考虑转义字面量的情况），则不要再强行全局替换
                if r"\text{" not in processed and r"\\text{" not in processed:
                    # 将连续的中文字符块安全地包裹起来
                    processed = re.sub(r'([\u4e00-\u9fff]+)', r'\\text{\1}', processed)
            
            # 2.3 Math Mode Wrapping (Implicit/Template based)
            # We don't force $$ here anymore because Tex class uses templates
            # like align* which are already math environments.
            final_text = processed
        else:
            final_text = ""
            
        # [Precision] Intercept and parse color from kwargs
        if "color" in kwargs:
            kwargs["color"] = SmartLayoutUtils.parse_color(kwargs["color"])
            
        super().__init__(final_text, **kwargs)

