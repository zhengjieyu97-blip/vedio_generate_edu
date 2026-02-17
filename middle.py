import json
import re
import os
import inspect
import ast
import difflib
from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

# Import Component & Layout Libraries
import manim_smart_components as MC_Geo
import manim_layout_templates as MC_Layout

# =====================================================================
# Middleware Architecture: The "Director" of Manim Generation
# Core Responsibilities (The 9 Rules):
# 1. Full Field Parsing: Handle all JSON fields from LLM output.
# 2. Scene/Segment Management: Clear screen on scene_id change; reset on segment change.
# 3. Component Instantiation: Combine params and visual_action with the component library.
# 4. Captions & Board Sync: Fixed captions at bottom; board content in designated slot.
# 5. Atomic Synchronization: Use voiceover to sync speech, visual, and board actions.
# 6. Non-Overlapping Layout: Ensure Board, Captions, and Visual zones never overlap.
# 7. Title Persistence: segment_title stays fixed across scenes.
# 8. Logic Guard (Iron Laws): Enforce is_incremental, parent, and target_id rules.
# 9. No Premature Scaling: Scale ONLY if content exceeds its slot boundary.
# =====================================================================

class MiddlewareParser:
    """
    [Role: Parser]
    Responsibility: Parse JSON and validate structure.
    """
    def __init__(self, json_path):
        self.json_path = json_path
        self.data = None

    def load_and_validate(self):
        """Loads JSON and performs basic schema validation."""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # 1. Markdown Stripping
            if content.startswith("```"):
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1:
                    content = content[start:end+1]
            
            # 2. Conservative Character Cleaner
            # [A] Clean Illegal Control Characters (0-31), except \n, \r, \t
            content = "".join(c if ord(c) >= 32 or c in "\n\r\t" else " " for c in content)

            def aggressive_repair(raw_content):
                # [B] "Comma Injector" - Handle missing commas between major blocks
                # [Refined] Only inject if followed by a property key (quote) to avoid LaTeX }{ misidentification
                raw_content = re.sub(r'}\s*({(?=\s*"))', r'}, \1', raw_content)
                raw_content = re.sub(r']\s*({(?=\s*"))', r'], \1', raw_content)
                raw_content = re.sub(r'}\s*(\[(?=\s*"))', r'}, \1', raw_content)
                
                # [C] "Property Comma Injector" - Handle missing commas between "key": "val" "key2": "val2"
                # Refined to be more specific to JSON structure
                raw_content = re.sub(r'(":\s*(?:[\d.]+|true|false|null|"[^"]*"))\s+(?=")', r'\1,', raw_content)
                
                # [E] Structural Cleanups
                raw_content = re.sub(r'(["\w]+":\s*[\d\w".]+),\s*{\s*\1', r'\1', raw_content)
                raw_content = re.sub(r',\s*([}\]])', r'\1', raw_content)
                return raw_content

            # [F] Python Expression Stringifier - Wrap bare math expressions in quotes
            python_expr_pattern = r'(:\s*)(\b(?:PI|np\.(?:pi|sin|cos|tan|sqrt|exp|log|abs)|TAU|DEGREES|math\.\w+)(?:[/*+-]\s*(?:\d+(?:\.\d+)?|PI|TAU|np\.\w+))*)'
            content = re.sub(python_expr_pattern, r'\1"\2"', content)
            
            # [G] Negative Number Edge Case
            content = re.sub(r'(:\s*)(-(?:PI|TAU|np\.\w+)(?:[/*+-]\s*(?:\d+(?:\.\d+)?|PI|TAU|np\.\w+))*)', r'\1"\2"', content)
            
            # [H] Unquoted Manim/Python Constant Fixer
            # LLM sometimes outputs JSON with bare Python constants like: "direction": UP
            # These MUST be quoted to be valid JSON: "direction": "UP"
            # Pattern matches: ": <CONSTANT>" where CONSTANT is a known Python/Manim name
            MANIM_CONSTANTS = [
                'UP', 'DOWN', 'LEFT', 'RIGHT', 'ORIGIN', 'UL', 'UR', 'DL', 'DR',
                'IN', 'OUT', 'DEGREES', 'RADIANS', 'TAU',
                'RED', 'GREEN', 'BLUE', 'YELLOW', 'CYAN', 'MAGENTA', 'WHITE', 'BLACK',
                'ORANGE', 'PURPLE', 'PINK', 'TEAL', 'GOLD', 'GREY', 'GRAY',
                'True', 'False', 'None'  # Python keywords that might slip through
            ]
            for const in MANIM_CONSTANTS:
                # Match ": CONST" or ": CONST," or ": CONST}" (not already quoted)
                pattern = rf'(:\s*)({const})(\s*[,\}}\]])'
                content = re.sub(pattern, rf'\1"{const}"\3', content)
            
            # 3. JSON Loading Attempt
            try:
                self.data = json.loads(content, strict=False)
            except json.JSONDecodeError as first_err:
                # [Fallback] If strict/semi-strict load fails, try aggressive repair
                try:
                    repaired_content = aggressive_repair(content)
                    self.data = json.loads(repaired_content, strict=False)
                    print(f"[Parser] Aggressive repair succeeded after initial failure.")
                except:
                    # [Second Fallback] Fix broken quotes in speech content
                    try:
                        def fix_nested_quotes(m):
                            k = m.group(1)
                            v = m.group(2)
                            v_fixed = v.replace('"', "'")
                            return f'{k}: "{v_fixed}"'
                        repaired_content = re.sub(r'("speech_content"|"text"):\s*"(.*)"', fix_nested_quotes, content)
                        self.data = json.loads(repaired_content, strict=False)
                    except:
                        # [Final Attempt] Crop to outermost braces
                        start_match = re.search(r'[\[{]', content)
                        end_match = re.search(r'[\]}](?=[^\]}]*$)', content)
                        if start_match and end_match:
                            content = content[start_match.start():end_match.end()]
                            self.data = json.loads(content, strict=False)
                        else:
                            raise first_err
            
            # 3. Schema Adaptation
            if "segments" not in self.data:
                if "content" in self.data and "segments" in self.data["content"]:
                    self.data = self.data["content"]
                else:
                    raise ValueError("JSON missing 'segments' field.")
                
            print(f"Successfully loaded and normalized plan from {self.json_path}")
            return self.data
        except Exception as e:
            print(f"Error loading/parsing JSON from {self.json_path}: {e}")
            # [Emergency fallback] If standard json fails, try a manual regex fix
            try:
                print("Attempting emergency structural repair...")
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                repaired = re.sub(r'("\w+":\s*\d+),\s*\{\s*"\w+":\s*\d+,', r'\1,', content)
                self.data = json.loads(repaired, strict=False)
                return self.data
            except:
                return None


class ManimGenerator:
    """
    [The Instantiator & Sync Engine]
    Translates the JSON plan into a STANDALONE Manim script.
    Implements the 9 Core Responsibilities from the design document.
    """
    
    # =========================================================================
    # [核心配置] 组件库对齐 - 基于 manim_smart_components.py 的 TopoGraph 继承树
    # =========================================================================
    
    # [地基工厂白名单] 所有可作为场景宿主的 TopoGraph 子类
    # 这些组件被创建时，会成为当前场景的"地基"，后续的 SmartPoint/SmartLine 会自动绑定到它
    FACTORY_WHITELIST = [
        # 基础几何
        "SmartPolygon", "SmartTriangle", "SmartCircle",
        # 函数与曲线
        "SmartFnPlot", "SmartSector", "SmartAngle", "SmartAnnulus",
        # 高级几何结构
        "SmartWindmill", "SmartInscribed", "SmartTangentSystem", "SmartRegularPolygon",
        "SmartTransversal", "SmartPolygonTransform", "SmartSimilarSystem",
        # 特殊几何
        "SmartSectorReform", "SmartConicSection", "SmartLimitBox", "SmartRigidLink",
        # 3D与向量
        "SmartSolid3D", "SmartUnitCircle", "SmartVectorSystem", "SmartStatSystem"
    ]
    
    # [注意] ATOMIC_OPERATIONS 已弃用
    # 新架构下，操作通过 "id.method" 模式调用（如 fn_1.add_label, fn_1.create_tangent_line）
    # 中间层通过检测 target_component 中的点号(.)来识别操作模式

    def __init__(self, plan, target_segment_id=None):
        self.plan = plan
        self.segments = plan.get("segments", [])
        self.target_segment_id = target_segment_id
        self.code_lines = []
        self.defined_ids = set()
        self.indent = "    "
        
        # [Rule 8] Host management
        self.scene_host_id = None  # The current "ground" for all sub-components
        self.current_factory_id = None
        self.active_slot_objects = {"visual": None, "board": None} # Track for fade transitions
        self.id_to_comp = {} # Track ID to Component Name mapping
        
        # [Rule 2.1] Bulletproof Colors
        self.SAFE_COLOR_MAP = {
            "WHITE": "#FFFFFF", "BLACK": "#000000", "RED": "#FC6255", 
            "GREEN": "#83C167", "BLUE": "#58C4DD", "YELLOW": "#FFFF00", 
            "CYAN": "#00FFFF", "MAGENTA": "#FF00FF", "GREY": "#888888", 
            "ORANGE": "#FF862F", "PURPLE": "#94424F", "TEAL": "#5CD0B3", 
            "GOLD": "#F1C40F", "PINK": "#FF86DC", "NAVY": "#000080"
        }

    # =========================================================================
    # [Rule 2.1] Data Sanitization & Repair
    # =========================================================================
    def _sanitize_params(self, value, key=None):
        """
        Recursively sanitize params:
        1. Auto-wrap Chinese text in \\text{}
        2. Force colors to Hex
        3. Convert numeric strings to floats
        4. Normalize Unicode math symbols to LaTeX
        """
        if isinstance(value, dict):
            return {k: self._sanitize_params(v, k) for k, v in value.items()}
        elif isinstance(value, list):
            # Special: Coordinate repair [ "1", "2" ] -> [ 1.0, 2.0 ]
            if key in ["coords", "coordinates", "start_node", "end_node"]:
                new_list = []
                for x in value:
                    try:
                        new_list.append(float(x) if isinstance(x, (str, int, float)) else x)
                    except:
                        new_list.append(x)
                return [self._sanitize_params(x) for x in new_list]
            return [self._sanitize_params(v) for v in value]
        elif isinstance(value, str):
            # 0. [FIX] Unicode Math Symbol Normalization
            # LLM may output Unicode superscripts/subscripts that XeLaTeX doesn't handle well
            unicode_replacements = {
                '²': '^2', '³': '^3', '⁴': '^4', '⁵': '^5',
                '₀': '_0', '₁': '_1', '₂': '_2', '₃': '_3',
                '×': '\\times ', '÷': '\\div ', '±': '\\pm ',
                '≤': '\\le ', '≥': '\\ge ', '≠': '\\ne ',
                '→': '\\to ', '←': '\\leftarrow ', '∞': '\\infty ',
                'π': '\\pi ', 'θ': '\\theta ', 'α': '\\alpha ', 'β': '\\beta ',
                '√': '\\sqrt', 'Δ': '\\Delta ', '∑': '\\sum ', '∫': '\\int ',
            }
            for uni, latex in unicode_replacements.items():
                value = value.replace(uni, latex)
            
            # 1. Color Repair
            if key in ["color", "text_color", "stroke_color", "fill_color"]:
                if value.startswith("#"):
                     return value
                v_upper = value.upper()
                return self.SAFE_COLOR_MAP.get(v_upper, "#FFFFFF")
            
            # 2. Chinese LaTeX Repair
            if re.search(r'[\u4e00-\u9fff]', value):
                if "\\text{" not in value and "$$" not in value:
                    return f"\\text{{{value}}}"
                
            # 3. Numeric Repair (if key suggests number)
            num_keys = ["radius", "buff", "font_size", "stroke_width", "opacity", "side_index"]
            if key in num_keys:
                try:
                    return float(value)
                except:
                    pass
            
            # 4. Direction Repair (if key is 'direction')
            if key == "direction":
                direction_map = {"UP": "UP", "DOWN": "DOWN", "LEFT": "LEFT", "RIGHT": "RIGHT",
                                 "UR": "UR", "UL": "UL", "DR": "DR", "DL": "DL"}
                v_upper = value.upper()
                if v_upper in direction_map:
                    return v_upper  # Keep as canonical string, component will convert
        return value


    def _get_valid_params(self, target_entity, data):
        """
        [STRICT AUDIT] Filters 'data' to keep only keys present in target_entity signature.
        Ignores 'kwargs' to allow for strict validation even if the component has a broad signature.
        """
        if not target_entity:
            return data
            
        try:
            sig = inspect.signature(target_entity)
            valid_keys = list(sig.parameters.keys())
            
            # [Iron Law] We prioritize explicit parameters. 
            # We ignore 'kwargs' and 'args' to force alignment with documented API.
            # However, we must allow 'points_or_coords' for SmartPolygon specifically 
            # if the signature contains a variadic *args (which usually represents points).
            has_variadic = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
            
            final_data = {}
            for k, v in data.items():
                if k in valid_keys:
                    if k not in ["self", "args", "kwargs"]:
                        final_data[k] = v
                elif has_variadic and k == "points_or_coords":
                    # Special Case: Universal components using *args
                    final_data[k] = v
                else:
                    # Attempt fuzzy match for hallucinated parameters
                    fuzzy_k = self._get_fuzzy_match(k, [vk for vk in valid_keys if vk not in ["self", "args", "kwargs"]])
                    if fuzzy_k:
                        final_data[fuzzy_k] = v
                        print(f"[Audit] Healed invalid parameter '{k}' -> '{fuzzy_k}' for {target_entity}")
                    else:
                        print(f"[Audit] Dropping invalid parameter '{k}' for {target_entity}")
                    
            return final_data
        except Exception as e:
            print(f"[Warning] Signature inspection failed for {target_entity}: {e}")
            return data

    def _get_fuzzy_match(self, key, valid_keys):
        """Finds a close match for a key in valid_keys."""
        if not valid_keys: return None
        matches = difflib.get_close_matches(key, valid_keys, n=1, cutoff=0.6)
        return matches[0] if matches else None

    def _parse_call_args(self, code_str):
        """
        Parses a method call string like 'obj.method(a=1, b=2)' and returns a dict of args.
        Returns empty dict if parsing fails or not a call.
        """
        try:
            # Handle cases like "Indicate(fn_1.add_label(...))" by extracting the inner call
            inner_call = re.search(r'\w+\.\w+\(.*\)', code_str)
            if not inner_call: return {}
            
            tree = ast.parse(inner_call.group(0))
            if not isinstance(tree.body[0], ast.Expr) or not isinstance(tree.body[0].value, ast.Call):
                return {}
            
            call = tree.body[0].value
            args_dict = {}
            
            # Note: We prioritize keywords. Positional args are harder to map without full context,
            # but for our simple Smart Components, LLM usually uses keyword args or predictable positions.
            for kw in call.keywords:
                # Convert AST node back to a Python value (approximate)
                try:
                    val = ast.literal_eval(kw.value)
                except:
                    # If it's a complex expression (like 'UP' or 'np.pi'), keep it as a string for now
                    val = ast.unparse(kw.value) if hasattr(ast, 'unparse') else "EXPR" 
                args_dict[kw.arg] = val
            
            return args_dict
        except:
            return {}

    def _resolve_val_str(self, v):
        """
        Recursive Resolver for converting Python values to representative strings.
        Handles math constants, unquoted IDs, and complex structures.
        """
        if isinstance(v, str):
            # Check if it's a defined variable ID
            if v in self.defined_ids:
                return v
            # Check if it's a Python math expression or Manim constant that should NOT be quoted
            # Examples: PI, np.pi, UP, RIGHT, PI/2, ORIGIN, etc.
            if re.match(r'^-?(?:PI|TAU|np\.(?:pi|sin|cos|tan|sqrt|exp|log|abs)|math\.\w+|UP|DOWN|LEFT|RIGHT|UR|UL|DR|DL|ORIGIN)(?:[/*+-]\s*(?:\d+(?:\.\d+)?|PI|TAU|np\.\w+|UP|DOWN|LEFT|RIGHT|ORIGIN))*$', v):
                return v
            return repr(v)
        elif isinstance(v, (int, float, bool)) or v is None:
            return repr(v)
        elif isinstance(v, list):
            return "[" + ", ".join(self._resolve_val_str(x) for x in v) + "]"
        elif isinstance(v, dict):
            parts = [f"{repr(mk)}: {self._resolve_val_str(mv)}" for mk, mv in v.items()]
            return "{" + ", ".join(parts) + "}"
        return repr(v)

    def _reconstruct_param_str(self, target_entity, data):
        """
        [DEEP HEALING] Signature-driven parameter reconstruction.
        Handles:
        1. Correct ordering of positional arguments.
        2. Mapping of variadic inputs (e.g. 'points_or_coords' -> *args).
        3. Exclusion of invalid/extra parameters.
        """
        if not target_entity:
            # Fallback to simple KV
            return ", ".join(f"{k}={self._resolve_val_str(v)}" for k, v in data.items())

        try:
            # Deep Copy data to avoid side effects
            work_data = data.copy()
            sig = inspect.signature(target_entity)
            params = sig.parameters
            
            pos_args = []
            kw_args = []
            used_keys = set()
            
            # Step 1: Identify and fill Positional-Only / Positional-or-Keyword arguments
            # We prioritize certain known keys to be positional for cleaner Manim code
            priority_pos = ["target", "node_or_index", "side_index", "index", "A", "B", "C", "center"]
            
            for name, param in params.items():
                if name == "self": continue
                
                # Check if we have this parameter in our data
                val = None
                if name in work_data:
                    val = work_data.pop(name)
                    used_keys.add(name)
                
                if val is not None:
                    # Decide if it should be positional or keyword
                    # rule: if it's in priority_pos AND it's one of the first few args, or it's POSITIONAL_ONLY
                    if param.kind == inspect.Parameter.POSITIONAL_ONLY or (name in priority_pos and len(kw_args) == 0):
                        pos_args.append(self._resolve_val_str(val))
                    else:
                        kw_args.append(f"{name}={self._resolve_val_str(val)}")
                
                # Handle Variadic Positional (*args)
                if param.kind == inspect.Parameter.VAR_POSITIONAL:
                    # Special case for SmartPolygon points
                    if "points_or_coords" in work_data:
                        pts = work_data.pop("points_or_coords")
                        if isinstance(pts, list):
                            for p in pts:
                                pos_args.append(self._resolve_val_str(p))
                        else:
                            pos_args.append(self._resolve_val_str(pts))
                        used_keys.add("points_or_coords")

            # Step 2: Add remaining keys as keywords (the "leaking" protection)
            # Actually, _get_valid_params should have filtered these out, 
            # but we iterate through data to ensure we catch everything intended.
            for k, v in work_data.items():
                kw_args.append(f"{k}={self._resolve_val_str(v)}")
                    
            return ", ".join(pos_args + kw_args)
            
        except Exception as e:
            print(f"[Warning] Param reconstruction failed for {target_entity}: {e}")
            return ", ".join(f"{k}={self._resolve_val_str(v)}" for k, v in data.items())

    # =========================================================================
    # [Rule 8] Logic Guard: The Iron Laws
    # =========================================================================
    def _logic_guard(self, atom):
        """
        Enforces the 3 Iron Laws before processing any atom:
        1. Factory Reset: is_incremental=False, parent=None, becomes host.
        2. Atomic Incremental: is_incremental=True, parent forced to host.
        3. Default: All others get is_incremental=False.
        """
        params = atom.get("params", {})
        comp = params.get("target_component", "")
        data = params.get("data", {})
        
        # [Rule 8] Host Logic Guard
        # Law 1: Factory Reset & Host Update
        if comp in self.FACTORY_WHITELIST:
            # If it's a creation of a new object, force is_incremental=False
            if params.get("id") not in self.defined_ids:
                params["is_incremental"] = False
            
            self.current_factory_id = params.get("id")
            self.scene_host_id = params.get("id")
            
        # Law 1.5: Operation Mode Security
        elif "." in comp:
            parent_id = comp.split(".")[0]
            # Auto-correction: Dot in component means IT MUST be incremental
            params["is_incremental"] = True
            # ID Fix: Operation atoms should not have their own unique ID if they are sub-methods
            # Just let them be part of the sync block
            if params.get("id") not in self.defined_ids:
                # We skip recording this as a 'new' object in defined_ids to avoid pollution
                pass
            
        # Law 2: Default for all other components
        # (Operation mode is already handled above via dot detection)
        else:
            if "." not in comp:
                 params["is_incremental"] = False


        # [Final Cleaning] Reserved keywords filter
        # Ensure that control fields don't leak into the constructor 'data' dictionary
        reserved_keys = ["id", "target_id", "slot", "is_incremental", "target_component"]
        for rk in reserved_keys:
            data.pop(rk, None)

    # =========================================================================
    # Core Generation Logic
    # =========================================================================
    def generate_full_script(self):
        """Generates the final Python code."""
        # 1. Boilerplate
        self.code_lines.extend([
            "from manim import *",
            "from manim_voiceover import VoiceoverScene",
            "from manim_voiceover.services.gtts import GTTSService",
            "import manim_smart_components as MC_Geo",
            "import manim_layout_templates as MC_Layout",
            "import numpy as np",
            "import os",
            "",
            "# --- Clash Proxy Configuration (for gTTS to access Google) ---",
            "",
            "# --- LaTeX CJK Configuration ---",
            "os.environ['PATH'] += os.pathsep + r'C:\\Users\\小余\\AppData\\Local\\Programs\\MiKTeX\\miktex\\bin\\x64'",
            "chinese_template = TexTemplate(tex_compiler='xelatex', output_format='.xdv', preamble=r'\\usepackage{xeCJK}')",
            "",
            "class GeneratedScene(MC_Layout.EduLayoutScene, VoiceoverScene):",
            f"{self.indent}def construct(self):",
            f"{self.indent}{self.indent}# 1. Global Config",
            f"{self.indent}{self.indent}Tex.set_default(tex_template=TexTemplateLibrary.ctex)",
            f"{self.indent}{self.indent}config.tex_template = TexTemplateLibrary.ctex",
            f"{self.indent}{self.indent}self.set_speech_service(GTTSService(lang='zh-cn'))",
            f"{self.indent}{self.indent}self.defined_ids = set()",
            f"{self.indent}{self.indent}self.persistent_title = None",
            f"{self.indent}{self.indent}self.current_caption = None",
            f"{self.indent}{self.indent}self.lm = None",
            f"{self.indent}{self.indent}self.scene_host_id = None",
            f"{self.indent}{self.indent}self.last_scene_id = None",
            ""
        ])

        # 2. Iterate Segments
        # [Feature] Segment Filtering
        active_segments = self.segments
        if self.target_segment_id is not None:
            active_segments = [s for s in self.segments if str(s.get("segment_id")) == str(self.target_segment_id)]
            if not active_segments:
                print(f"[Warning] Segment ID {self.target_segment_id} not found in plan.")

        for seg in active_segments:
            self._gen_segment(seg)

        return "\n".join(self.code_lines)

    # =========================================================================
    # [Rule 2.2] Segment Management
    # =========================================================================
    def _gen_segment(self, segment):
        title_text = segment.get("segment_title", "Untitled")
        
        # [Rule 3.3] Segment Transition: FadeOut everything, reset title.
        self.code_lines.extend([
            f"{self.indent}{self.indent}# ==================================================",
            f"{self.indent}{self.indent}# --- Segment: {title_text} ---",
            f"{self.indent}{self.indent}# ==================================================",
            f"{self.indent}{self.indent}# [Rule 3.3] Segment Transition: Full clear with fade.",
            f"{self.indent}{self.indent}if self.persistent_title:",
            f"{self.indent}{self.indent}{self.indent}self.play(FadeOut(self.persistent_title), run_time=0.3)",
            f"{self.indent}{self.indent}self.clear()",
            f"{self.indent}{self.indent}self.persistent_title = Tex(r'$$\\\\text{{{title_text}}}$$', font_size=48, color=GOLD)",
            f"{self.indent}{self.indent}self.persistent_title.to_edge(UP, buff=0.5)",
            f"{self.indent}{self.indent}self.play(FadeIn(self.persistent_title), run_time=0.5)",
            f"{self.indent}{self.indent}self.lm = None",
            f"{self.indent}{self.indent}self.last_scene_id = None",
            ""
        ])
        
        # Iterate Scenes
        scenes = segment.get("scenes", [])
        for scene in scenes:
            self._gen_scene(scene)

    # =========================================================================
    # [Rule 2.2] Scene Management
    # =========================================================================
    def _gen_scene(self, scene):
        layout_type = scene.get("layout_type", "split_screen")
        scene_id = scene.get("scene_id", "default_scene")
        atoms = scene.get("atoms", [])
        
        # [Rule: Scene Capacity Redline] - Auto-split if atoms > 8
        CHUNK_SIZE = 8
        atom_chunks = [atoms[i:i + CHUNK_SIZE] for i in range(0, len(atoms), CHUNK_SIZE)]
        
        for idx, chunk in enumerate(atom_chunks):
            sub_id = f"{scene_id}_p{idx+1}" if len(atom_chunks) > 1 else scene_id
            
            # Reset host for new scene/part
            self.scene_host_id = None
            self.current_factory_id = None

            self.code_lines.append(f"{self.indent}{self.indent}# --- Scene Part: {sub_id} ---")
            
            # [Rules 3.2 & 3.4] Transition logic
            if idx == 0:
                # First chunk of a scene: standard cross-scene fade
                self.code_lines.extend([
                    f"{self.indent}{self.indent}if self.last_scene_id is not None and self.last_scene_id != '{sub_id}':",
                    f"{self.indent}{self.indent}{self.indent}objs_to_fade = [m for m in self.mobjects if m != self.persistent_title and m != self.current_caption]",
                    f"{self.indent}{self.indent}{self.indent}if objs_to_fade:",
                    f"{self.indent}{self.indent}{self.indent}{self.indent}self.play(*[FadeOut(o) for o in objs_to_fade], run_time=0.3)",
                    f"{self.indent}{self.indent}{self.indent}if self.lm: self.lm.placed_mobs = {{'visual': [], 'board': []}}",
                ])
            else:
                # Subsequent chunks: KEEP visual, CLEAR board (Visual Continuity)
                self.code_lines.extend([
                    f"{self.indent}{self.indent}# [Rule 3.4] Auto-Page: Clear board, keep visual objects.",
                    f"{self.indent}{self.indent}if self.lm and self.lm.placed_mobs['board']:",
                    f"{self.indent}{self.indent}{self.indent}self.play(*[FadeOut(o) for o in self.lm.placed_mobs['board']], run_time=0.3)",
                    f"{self.indent}{self.indent}{self.indent}self.lm.placed_mobs['board'] = []",
                ])

            self.code_lines.extend([
                f"{self.indent}{self.indent}self.last_scene_id = '{sub_id}'",
                f"{self.indent}{self.indent}if self.lm is None or self.lm.layout_type != '{layout_type}':",
                f"{self.indent}{self.indent}{self.indent}self.lm = MC_Layout.LayoutManager('{layout_type}')",
                ""
            ])

            for atom in chunk:
                self._gen_atom(atom)

    # =========================================================================
    # [Rule 3, 4, 5, 7, 8] Atom Generation
    # =========================================================================
    def _gen_atom(self, atom):
        # Apply Logic Guard first!
        self._logic_guard(atom)

        params = atom.get("params", {})
        obj_id = params.get("id")
        comp = params.get("target_component", "")
        slot = params.get("slot", "visual")
        is_incremental = params.get("is_incremental", False)

        if not obj_id or not comp:
            # [Fix] Allow commentary atoms (pure speech/board actions)
            v_segs = atom.get("voice_segments", [])
            for v in v_segs:
                self._gen_v_seg(v, slot)
            return

        # Instantiation
        # [Rule 8] Skip instantiation if it's an operation on an existing ID (dot in comp)
        is_operation = "." in comp

        if not is_operation and obj_id not in self.defined_ids and comp:
            cls_name = comp.split(".")[-1]

            # 1. Audit and Clean Data
            data = self._sanitize_params(params.get("data", {}))

            # 2. Signature-based Filtering
            cls_obj = getattr(MC_Geo, cls_name, None)
            if cls_obj:
                data = self._get_valid_params(cls_obj, data)

            # [Note] Auto-parent injection (Rule 2.8) is DISABLED for this JSON protocol.
            # The new protocol expects parent to be explicitly specified in JSON data.
            # Components without explicit parent are treated as independent objects.

            # 3. Generate param string using Unified Reconstruction
            param_str = self._reconstruct_param_str(cls_obj, data)

            # [Rule 3.1] Fade Transition for non-incremental components
            if not is_incremental and cls_name in self.FACTORY_WHITELIST:
                self.code_lines.extend([
                    f"{self.indent}{self.indent}# [Rule 3.1] Fade Transition: Clear slot before new factory component.",
                    f"{self.indent}{self.indent}slot_content = self.lm.placed_mobs.get('{slot}', [])",
                    f"{self.indent}{self.indent}if slot_content:",
                    f"{self.indent}{self.indent}{self.indent}self.play(*[FadeOut(o) for o in slot_content], run_time=0.3)",
                    f"{self.indent}{self.indent}{self.indent}self.lm.placed_mobs['{slot}'] = []",
                ])

            self.code_lines.append(f"{self.indent}{self.indent}{obj_id} = MC_Geo.{cls_name}({param_str})")

            # [Rule 8] Update host if factory
            if cls_name in self.FACTORY_WHITELIST:
                self.code_lines.append(f"{self.indent}{self.indent}self.scene_host_id = '{obj_id}'")

            # [Rule 6] Automated Layout - All components use layout manager
            self.code_lines.append(f"{self.indent}{self.indent}self.lm.place({obj_id}, '{slot}')")
            self.code_lines.append("")
            # 记录 ID 与组件类名的映射，后续可以用于分析
            self.id_to_comp[obj_id] = cls_name

            # [Path A Deep Healing] Sanitize visual_action for factory components
            # If LLM attempts to re-instantiate in visual_action, force it to just 'Create(id)'
            if cls_name in self.FACTORY_WHITELIST:
                v_segs = atom.get("voice_segments", [])
                for v_entry in v_segs:
                    vis = v_entry.get("visual_action", "").strip()
                    # Pattern: Create(MC_Geo.SmartTriangle(...)) -> Create(tri_1)
                    if any(kw in vis for kw in ["MC_Geo", cls_name, "("]) and obj_id not in vis:
                         # Reconstruct based on standard action type
                         if "Write" in vis:
                             v_entry["visual_action"] = f"Write({obj_id})"
                         else:
                             v_entry["visual_action"] = f"Create({obj_id})"
                         print(f"[Deep Healing Path A] Sanitized redundant constructor in visual_action for {obj_id}")
                    elif vis == "" or vis == obj_id:
                         # Ensure some action exists
                         v_entry["visual_action"] = f"Create({obj_id})"

        # [Path B Deep Healing] Reinforce visual_action from data to prevent hallucinations
        if is_operation:
            host_id, method_name = comp.split(".", 1)
            host_cls_name = self.id_to_comp.get(host_id)
            if host_cls_name:
                host_cls = getattr(MC_Geo, host_cls_name, None)
                method_obj = getattr(host_cls, method_name, None) if host_cls else None
                
                if method_obj:
                    # Source of Truth from Plan
                    plan_data = self._sanitize_params(params.get("data", {}))
                    
                    v_segs = atom.get("voice_segments", [])
                    for v_entry in v_segs:
                        old_vis = v_entry.get("visual_action", "").strip()
                        if method_name not in old_vis: continue
                        
                        # Step 1: Parse existing visual_action and check consistency with plan_data
                        vis_args = self._parse_call_args(old_vis)
                        
                        needs_healing = False
                        merged_args = vis_args.copy()
                        
                        for k, v in plan_data.items():
                            if k not in merged_args or merged_args[k] != v:
                                # Inconsistency detected! Use plan_data value
                                merged_args[k] = v
                                needs_healing = True
                        
                        # Step 2: Library Signature Alignment (Clean-up keys and removal of extras)
                        valid_args = self._get_valid_params(method_obj, merged_args)
                        if len(valid_args) != len(merged_args):
                            needs_healing = True

                        if needs_healing:
                            # Step 3: Reconstruction using Unified Reconstruction
                            healed_params = self._reconstruct_param_str(method_obj, valid_args)
                            
                            # Preserve the outer wrapper (like Indicate(...) or Create(...)) if it exists
                            wrapper_match = re.match(r'^(\w+)\(.*\)$', old_vis)
                            if wrapper_match and wrapper_match.group(1) not in [host_id, method_name]:
                                wrapper = wrapper_match.group(1)
                                healed_action = f"{wrapper}({host_id}.{method_name}({healed_params}))"
                            else:
                                healed_action = f"{host_id}.{method_name}({healed_params})"
                            
                            v_entry["visual_action"] = healed_action
                            print(f"[Deep Healing] Re-aligned {host_id}.{method_name} in visual_action.")

        # Sync Block
        v_segs = atom.get("voice_segments", [])
        for v in v_segs:
            self._gen_v_seg(v, slot)

        # [Important] Only after the sync block (Create/Write) of THIS atom is done,
        # we record the ID in defined_ids. This ensures:
        # 1. The first Create() isn't intercepted by the Indicate() logic.
        # 2. Subsequent atoms can correctly lookup this ID.
        if not is_operation and obj_id not in self.defined_ids:
            self.code_lines.extend([
                f"{self.indent}{self.indent}self.defined_ids.add('{obj_id}')",
                f"{self.indent}{self.indent}globals()['{obj_id}'] = {obj_id}",
            ])
            self.defined_ids.add(obj_id)

    # =========================================================================
    # [Rule 2.3, 2.5] Voice Segment Sync
    # =========================================================================
    def _gen_v_seg(self, v_seg, default_slot):
        speech = v_seg.get("speech_content", "").replace("'", "\\'")
        vis_code = v_seg.get("visual_action", "").strip()
        brd_code = v_seg.get("board_action", "").strip()

        # [Rule 8] Redundant Creation Interceptor
        # If LLM says "Create(obj)" but obj is already defined, change to "Indicate" or wait
        pattern_create = re.compile(r"(Create|Write)\((.*?)\)")
        match = pattern_create.search(vis_code)
        if match:
            target_id = match.group(2)
            if target_id in self.defined_ids:
                # Replace with a harmless 'Indicate' (Emphasis) instead of a crashing 'Create'
                vis_code = vis_code.replace(f"{match.group(1)}({target_id})", f"Indicate({target_id})")

        def wrap_layout(code, slot):
            if not code:
                return None
            # Skip layout for point/line math objects or explicit parent
            is_math_obj = ("SmartPoint" in code or "SmartLine" in code) and "MC_Geo" in code
            if is_math_obj or "parent=" in code or "c2p(" in code:
                return code
            # Wrap Tex/MathTex/MC_Geo creation with place
            # [Fix] Atoms are now explicit in _gen_atom, only wrap non-atomic creators here
            pattern = re.compile(r"((?:MC_Geo\.(?!SmartPoint|SmartLine|SmartTranslate|SmartRotate|SmartScale)|MathTex|Tex)\((?:[^()]|\([^()]*\))*\))")
            if pattern.search(code):
                return pattern.sub(rf"self.lm.place(\1, '{slot}')", code)
            return code

        def wrap_raw_strings(code):
            # 1. LaTeX Protection: Wrap substrings that look like latex with backslashes in r'...'
            def r_replacer(match):
                prefix = match.group(1)
                q = match.group(2)
                content = match.group(3)
                # [Fix] Normalize excessive backslashes created by LLM hallucination
                # e.g. \\\\text -> \text inside r''
                content = re.sub(r"\\{2,}", r"\\", content)
                
                # [Fix] CJK Sanitization: Wrap Chinese in \text{} if not already wrapped
                if re.search(r'[\u4e00-\u9fff]', content):
                    if "\\text{" not in content and "$$" not in content:
                        content = f"\\text{{{content}}}"

                if "\\" in content:
                    # If it already has an 'r' prefix, don't double it
                    final_prefix = prefix if prefix else "r"
                    return f"{final_prefix}{q}{content}{q}"
                return f"{prefix}{q}{content}{q}"
            
            # Regex to find quoted strings with optional prefix: ([r]?'...' or [r]?"...")
            processed = re.sub(r"([rfb]?)(['\"])(.*?)\2", r_replacer, code)
            return processed

        vis_final = wrap_layout(wrap_raw_strings(vis_code), 'visual')
        brd_final = wrap_layout(wrap_raw_strings(brd_code), 'board')

        self.code_lines.append(f"{self.indent}{self.indent}with self.voiceover(text=r'{speech}') as tracker:")
        
        # [Rule 2.5] Caption Logic - Direct switch (no fade animation)
        self.code_lines.extend([
            f"{self.indent}{self.indent}{self.indent}self.current_caption = self.update_caption(self.current_caption, r'{speech}')",
        ])

        # [Rule 2.3] Animation Logic - All synced to voiceover duration
        # [Fix] Auto-wrap Mobjects that are not wrapped in an Animation
        def ensure_animation(a):
            if not a: return None
            # Common animation names in Manim
            anim_keywords = ["Write", "FadeIn", "Create", "FadeOut", "Transform", "ReplacementTransform", "Indicate", "Circumscribe", "Flash", "FocusOn", "Rotate", "Scale", "Shift", "MoveTo", "Become"]
            if any(a.strip().startswith(kw+"(") for kw in anim_keywords):
                return a
            # If it's a dot call like obj.animate.become(...), it's also an animation
            if ".animate." in a:
                return a
            
            # [Smooth] Smart Animation Selection
            # Text/Labels -> Write
            if any(kw in a for kw in ["Text", "Tex", "label", "caption"]):
                return f"Write({a})"
            # Smart Components (non-incremental creation) -> Create
            if "MC_Geo" in a and "." not in a:
                return f"Create({a})"
            # Operations/Marks -> Create or Write
            if "mark" in a or "elbow" in a:
                return f"Create({a})"
            
            # General default
            return f"FadeIn({a})"

        anims = [ensure_animation(a) for a in [vis_final, brd_final] if a]
        # [Simplified] Caption directly switches, no fade animations
        anims_str = "[" + ", ".join(anims) + "]" if anims else "[]"
        self.code_lines.extend([
            f"{self.indent}{self.indent}{self.indent}self.play(*{anims_str}, run_time=tracker.duration)"
        ])
        self.code_lines.append("")
        self.code_lines.append("")


if __name__ == "__main__":
    import sys
    
    target_file = "docement/1.md"
    target_segment = None
    
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    if len(sys.argv) > 2:
        target_segment = sys.argv[2]
        
    if not os.path.exists(target_file):
        print(f"[Error] File not found: {target_file}")
        sys.exit(1)
        
    parser = MiddlewareParser(target_file)
    plan = parser.load_and_validate()
    
    if plan:
        full_code = ManimGenerator(plan, target_segment_id=target_segment).generate_full_script()
        with open("generated_manim.py", "w", encoding="utf-8") as f:
            f.write(full_code)
        
        msg = f"\n[Success] Code generated for {'all segments' if not target_segment else f'segment {target_segment}'}."
        print(msg)
        print(f"Run: manim -pql generated_manim.py GeneratedScene")
