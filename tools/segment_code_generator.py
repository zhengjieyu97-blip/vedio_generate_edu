
import json
import re
import os
import inspect
import ast
import difflib
from typing import Any, Dict, List, Optional, Union
import traceback
from collections.abc import Generator

# 导入 Manim 相关库（仅用于签名检查和常量引用，不进行渲染）
from manim import *
import manim_smart_components as MC_Geo
import manim_layout_templates as MC_Layout

try:
    from dify_plugin import Tool
    from dify_plugin.entities.tool import ToolInvokeMessage
except ImportError:
    # 本地调试 Mock
    class Tool:
        def create_text_message(self, text):
            return {"type": "text", "text": text}
    ToolInvokeMessage = dict

class SegmentCodeGenerator(Tool):
    """
    [Dify Plugin] Manim 代码生成器
    职责：
    1. 接收单个 Segment 的 JSON 数据。
    2. 使用 MiddlewareParser 进行清洗和校验。
    3. 使用 ManimGenerator 生成独立的 Manim Python 脚本。
    4. 返回纯 Python 代码字符串。
    """

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """
        Dify 工具入口
        参数:
        - segment_json: 单个片段的 JSON 对象或字符串
        """
        try:
            # 1. 获取并解析输入
            segment_input = tool_parameters.get("segment_json", "")
            if not segment_input:
                yield self.create_text_message("# Error: segment_json is required")
                return

            # 兼容 JSON 字符串或直接的字典对象
            if isinstance(segment_input, str):
                try:
                    # 尝试使用 MiddlewareParser 的清洗逻辑（如果是字符串）
                    parser = MiddlewareParser(segment_input)
                    segment_data = parser.load_and_validate()
                except Exception as e:
                    # 如果 Parser 失败，尝试直接 loads
                    segment_data = json.loads(segment_input)
            else:
                segment_data = segment_input

            # 2. 构造虚拟的 Plan 对象 (Generator 期望的格式)
            # Generator 期望一个包含 "segments" 列表的 plan，或者我们改造 Generator 构造函数
            # 为了最小化改动，我们构造一个包装对象
            if "segments" not in segment_data:
                # 假设输入就是单个 segment
                plan = {"segments": [segment_data]}
                target_id = segment_data.get("segment_id")
            else:
                # 输入是一个完整的 plan
                plan = segment_data
                # 默认只处理第一个，或者需要指定 ID？
                # 根据需求 "接收单个迭代节点单个片段"，我们假设 plan['segments'] 里只有一个，或者我们只取第一个
                target_id = plan["segments"][0].get("segment_id")

            # 3. 生成代码
            generator = ManimGenerator(plan, target_segment_id=target_id)
            code = generator.generate_full_script()

            # 4. Return as JSON String (Step 5668)
            result = {
                "code": code,
                "segment_id": target_id
            }
            yield self.create_text_message(json.dumps(result, ensure_ascii=False))

        except Exception as e:
            error_msg = f"# Code Generation Error:\n\"\"\"\n{traceback.format_exc()}\n\"\"\""
            yield self.create_text_message(error_msg)


# =====================================================================
# Middleware Architecture: The "Director" of Manim Generation
# (Migrated from middle.py)
# =====================================================================

class MiddlewareParser:
    """
    [Role: Parser]
    Responsibility: Parse JSON and validate structure.
    """
    def __init__(self, content_str=None, json_path=None):
        self.json_path = json_path
        self.content_str = content_str
        self.data = None

    def load_and_validate(self):
        """Loads JSON and performs basic schema validation."""
        content = ""
        try:
            if self.json_path:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
            elif self.content_str:
                content = self.content_str.strip()
            else:
                return None
            
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
                    # print(f"[Parser] Aggressive repair succeeded after initial failure.") # Logger removed for Dify
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
                # For single segment tool use, we might accept a single segment dict, but Generator expects structure
                # We will handle structure adaptation in the Tool._invoke or Generator init
                
            return self.data
        except Exception as e:
            # print(f"Error loading/parsing JSON: {e}")
            # [Emergency fallback] If standard json fails, try a manual regex fix
            try:
                # print("Attempting emergency structural repair...")
                # with open(self.json_path, 'r', encoding='utf-8') as f: # Not using file here
                #    content = f.read()
                if not content: return None
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

            # [LaTeX Auto-Repair] Fix common LaTeX syntax errors from LLM
            value = ManimGenerator._fix_latex_string(value)
            
            # 1. Color Repair
            if key in ["color", "text_color", "stroke_color", "fill_color"]:
                if value.startswith("#"):
                     return value
                v_upper = value.upper()
                # Direct match first
                if v_upper in self.SAFE_COLOR_MAP:
                    return self.SAFE_COLOR_MAP[v_upper]
                # Smart fallback: Try to match color variants (e.g., BLUE_A -> BLUE, RED_B -> RED)
                # Pattern: COLOR_LETTER -> COLOR (strip suffix like _A, _B, _C, etc.)
                base_color = re.sub(r'_[A-Z]$', '', v_upper)
                if base_color in self.SAFE_COLOR_MAP:
                    return self.SAFE_COLOR_MAP[base_color]
                # Final fallback: default to WHITE (safer than random color)
                return self.SAFE_COLOR_MAP.get("WHITE", "#FFFFFF")
            
            # 2. LaTeX Math & Chinese Repair
            if re.search(r'[\u4e00-\u9fff]', value):
                if "\\text{" not in value and "$$" not in value:
                    # [Fix] Mixed Math/Chinese: If it looks like math (has subscript/superscript), wrap in $$
                    if any(c in value for c in "_^=\\"):
                         return f"$${value}$$"
                    return f"\\text{{{value}}}"
            
            # [Add] Automatic Math Wrapping for non-Chinese strings
            elif any(c in value for c in "^_\\#") or (key == "text" and any(c in value for c in "+-*/=")):
                if "$$" not in value:
                    # Detect common math patterns that LLM might forget to wrap
                    return f"$${value}$$"
                
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

    # =========================================================================
    # [LaTeX Auto-Repair] Fix common LaTeX syntax errors generated by LLM
    # =========================================================================
    @staticmethod
    def _fix_latex_string(s: str) -> str:
        """
        [LaTeX Auto-Repair] 修复 LLM 生成的 LaTeX 字符串中常见的语法错误。
        修复项目：
        1. \\; 在数学模式中非法 → \\, (数学模式内合法的细间距)
        2. \\; / \\: / \\! 在 \\text{} 内部 → 替换为空格
        3. 括号不匹配（多余 } 或 {}）→ 补全或裁切
        4. 双层 $$$$ 嵌套 → 合并为单层 $$
        5. \\end{align*} 前多余的 \\\\\ → 移除
        """
        if not s or not isinstance(s, str):
            return s

        # 1. 修复 \text{} 内部的间距命令：\; \: \! 在文本模式中不合法
        def _fix_text_env(m):
            inner = m.group(1)
            inner = inner.replace('\\;', ' ').replace('\\:', ' ').replace('\\!', '')
            return '\\text{' + inner + '}'
        s = re.sub(r'\\text\{([^}]*)\}', _fix_text_env, s)

        # 2. 修复数学模式中的 \; → \,（\text{} 内部已处理，剩余均可替换）
        s = s.replace('\\;', '\\,')

        # 3. 修复双 $$ 嵌套：$$$$...$$$$  →  $$...$$
        s = re.sub(r'\$\$\s*\$\$', '$$', s)
        s = re.sub(r'\$\$\s*\$\$', '$$', s)  # second pass

        # 4. 修复括号不匹配
        # 注意：只对 LaTeX 内容（含 $ 或 \ ）应用，避免影响 Python 代码字符串
        if '$' in s or '\\' in s:
            open_count = s.count('{')
            close_count = s.count('}')
            diff = open_count - close_count
            if diff > 0:
                # 多 { → 补充 }
                s = s + '}' * diff
            elif diff < 0:
                # 多 } → 从末尾倒序运算移除多余的 }
                for _ in range(-diff):
                    last_brace = s.rfind('}')
                    if last_brace != -1:
                        s = s[:last_brace] + s[last_brace + 1:]

        # 5. 移除 \end{align*} / \end{equation} 前多余的 \\\r
        s = re.sub(r'\\\\(\s*\\end\{(?:align|align\*|equation|equation\*|gather|gather\*)\})', r'\1', s)

        return s


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
                        # print(f"[Audit] Healed invalid parameter '{k}' -> '{fuzzy_k}' for {target_entity}")
                    else:
                        # print(f"[Audit] Dropping invalid parameter '{k}' for {target_entity}")
                        pass
                    
            return final_data
        except Exception as e:
            # print(f"[Warning] Signature inspection failed for {target_entity}: {e}")
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
            
            # [FIX] Escape backslashes for ast.parse to avoid SyntaxWarning for LaTeX symbols like \pi or \cdot
            safe_str = inner_call.group(0).replace("\\", "\\\\")
            tree = ast.parse(safe_str)
            if not isinstance(tree.body[0], ast.Expr) or not isinstance(tree.body[0].value, ast.Call):
                return {}
            
            call = tree.body[0].value
            args_dict = {}
            
            # Note: We prioritize keywords. Positional args are harder to map without full context,
            # but for our simple Smart Components, LLM usually uses keyword args or predictable positions.
            for kw in call.keywords:
                # Convert AST node back to a Python value (approximate)
                try:
                    # literal_eval is safer, but we still guard it
                    val = ast.literal_eval(kw.value)
                except:
                    # If it's a complex expression (like 'UP' or 'np.pi'), keep it as a string for now
                    val = ast.unparse(kw.value) if hasattr(ast, 'unparse') else "EXPR" 
                args_dict[kw.arg] = val
            
            return args_dict
        except:
            return {}

    def _resolve_val_str(self, v, key=None):
        """
        Recursive Resolver for converting Python values to representative strings.
        Handles math constants, unquoted IDs, and complex structures.
        """
        if isinstance(v, str):
            # Check if it's a defined variable ID
            if v in self.defined_ids:
                return v
            # [Color Fix] Convert color names to hex values if key indicates color
            if key in ["color", "text_color", "stroke_color", "fill_color"]:
                if v.startswith("#"):
                    return repr(v)  # Already hex, just quote it
                v_upper = v.upper()
                if v_upper in self.SAFE_COLOR_MAP:
                    return repr(self.SAFE_COLOR_MAP[v_upper])  # Convert to hex and quote
                # Try base color (e.g., BLUE_A -> BLUE)
                base_color = re.sub(r'_[A-Z]$', '', v_upper)
                if base_color in self.SAFE_COLOR_MAP:
                    return repr(self.SAFE_COLOR_MAP[base_color])
            # Check if it's a Python math expression or Manim constant that should NOT be quoted
            # Examples: PI, np.pi, UP, RIGHT, PI/2, ORIGIN, etc.
            if re.match(r'^-?(?:PI|TAU|np\.(?:pi|sin|cos|tan|sqrt|exp|log|abs)|math\.\w+|UP|DOWN|LEFT|RIGHT|UR|UL|DR|DL|ORIGIN)(?:[/*+-]\s*(?:\d+(?:\.\d+)?|PI|TAU|np\.\w+|UP|DOWN|LEFT|RIGHT|ORIGIN))*$', v):
                return v
            return repr(v)
        elif isinstance(v, (int, float, bool)) or v is None:
            return repr(v)
        elif isinstance(v, list):
            return "[" + ", ".join(self._resolve_val_str(x, key) for x in v) + "]"
        elif isinstance(v, dict):
            parts = [f"{repr(mk)}: {self._resolve_val_str(mv, mk)}" for mk, mv in v.items()]
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
            return ", ".join(f"{k}={self._resolve_val_str(v, k)}" for k, v in data.items())

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
                        pos_args.append(self._resolve_val_str(val, name))
                    else:
                        kw_args.append(f"{name}={self._resolve_val_str(val, name)}")
                
                # Handle Variadic Positional (*args)
                if param.kind == inspect.Parameter.VAR_POSITIONAL:
                    # Special case for SmartPolygon points
                    if "points_or_coords" in work_data:
                        pts = work_data.pop("points_or_coords")
                        if isinstance(pts, list):
                            for p in pts:
                                pos_args.append(self._resolve_val_str(p, "points_or_coords"))
                        else:
                            pos_args.append(self._resolve_val_str(pts, "points_or_coords"))
                        used_keys.add("points_or_coords")

            # Step 2: Add remaining keys as keywords (the "leaking" protection)
            # Actually, _get_valid_params should have filtered these out, 
            # but we iterate through data to ensure we catch everything intended.
            for k, v in work_data.items():
                kw_args.append(f"{k}={self._resolve_val_str(v, k)}")
                    
            return ", ".join(pos_args + kw_args)
            
        except Exception as e:
            # print(f"[Warning] Param reconstruction failed for {target_entity}: {e}")
            return ", ".join(f"{k}={self._resolve_val_str(v, k)}" for k, v in data.items())

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
            "# -*- coding: utf-8 -*-",
            "from manim import *",
            "from manim_voiceover import VoiceoverScene",
            "from manim_voiceover.services.gtts import GTTSService",
            "import manim_smart_components as MC_Geo",
            "import manim_layout_templates as MC_Layout",
            "import numpy as np",
            "import os",
            "from pathlib import Path",
            "# [Safety Patch] Intercept direct Tex() calls from LLM",
            "Tex = MC_Geo.SafeTex",
            "MathTex = MC_Geo.SafeTex",
            "",
            "# --- LaTeX CJK Configuration ---",
            "chinese_template = TexTemplate(tex_compiler='xelatex', output_format='.xdv', preamble=r'\\usepackage{xeCJK}')",
            "",
            "class GeneratedScene(MC_Layout.EduLayoutScene, VoiceoverScene):",
            f"{self.indent}def construct(self):",
            f"{self.indent}{self.indent}# 1. Global Config",
            f"{self.indent}{self.indent}# Default to Ctex for better Chinese support and fewer Win file lock issues",
            f"{self.indent}{self.indent}Tex.set_default(tex_template=TexTemplateLibrary.ctex)",
            f"{self.indent}{self.indent}config.tex_template = TexTemplateLibrary.ctex",
            f"{self.indent}{self.indent}config.no_latex_cleanup = True  # Avoid WinError 32 on Windows",
            f"{self.indent}{self.indent}# [FIX] Ensure voiceover_cache is a Path object to support '/' operator inside manim-voiceover",
            f"{self.indent}{self.indent}voiceover_cache = Path('media') / 'voiceovers' / 'seg_{self.target_segment_id or 0}'",
            f"{self.indent}{self.indent}self.set_speech_service(GTTSService(lang='zh-cn', cache_dir=voiceover_cache))",
            f"{self.indent}{self.indent}self.defined_ids = set()",
            f"{self.indent}{self.indent}self.persistent_title = None",
            f"{self.indent}{self.indent}self.current_caption = None",
            f"{self.indent}{self.indent}self.lm = None",
            f"{self.indent}{self.indent}self.scene_host_id = None",
            f"{self.indent}{self.indent}self.last_scene_id = None",
            "",
            f"{self.indent}{self.indent}# [Global Safety Net] Prevent Write(None) / FadeIn(None) crashes",
            f"{self.indent}{self.indent}# If any component method returns None (e.g. index out of range), _safe() converts it to an empty VMobject.",
            f"{self.indent}{self.indent}def _safe(mob):",
            f"{self.indent}{self.indent}{self.indent}return mob if mob is not None else VMobject()",
            "",
            f"{self.indent}{self.indent}# [Robustness] Fallback for Missing Components",
            f"{self.indent}{self.indent}class SafeFallback(VMobject):",
            f"{self.indent}{self.indent}{self.indent}def __init__(self, *args, **kwargs):",
            f"{self.indent}{self.indent}{self.indent}{self.indent}# [FIX] Pre-inject internal Manim arrays via object.__setattr__ BEFORE super().__init__()",
            f"{self.indent}{self.indent}{self.indent}{self.indent}# so that Manim's set_color() -> update_rgbas_array() does NOT trigger __getattr__",
            f"{self.indent}{self.indent}{self.indent}{self.indent}# which would return a lambda, causing len(lambda) -> TypeError.",
            f"{self.indent}{self.indent}{self.indent}{self.indent}import numpy as _np",
            f"{self.indent}{self.indent}{self.indent}{self.indent}object.__setattr__(self, 'fill_rgbas', _np.zeros((1, 4)))",
            f"{self.indent}{self.indent}{self.indent}{self.indent}object.__setattr__(self, 'stroke_rgbas', _np.zeros((1, 4)))",
            f"{self.indent}{self.indent}{self.indent}{self.indent}super().__init__()",
            f"{self.indent}{self.indent}{self.indent}def __getattr__(self, name):",
            f"{self.indent}{self.indent}{self.indent}{self.indent}# [FIX] Do NOT intercept dunder attributes — Python internals need them to raise AttributeError",
            f"{self.indent}{self.indent}{self.indent}{self.indent}if name.startswith('__') and name.endswith('__'):",
            f"{self.indent}{self.indent}{self.indent}{self.indent}{self.indent}raise AttributeError(name)",
            f"{self.indent}{self.indent}{self.indent}{self.indent}# Swallow all other method calls to prevent crash, return self for chaining",
            f"{self.indent}{self.indent}{self.indent}{self.indent}return lambda *args, **kwargs: self",
            ""
        ])

        # 2. Iterate Segments
        # [Feature] Segment Filtering
        active_segments = self.segments
        if self.target_segment_id is not None:
            active_segments = [s for s in self.segments if str(s.get("segment_id")) == str(self.target_segment_id)]
            if not active_segments:
                # print(f"[Warning] Segment ID {self.target_segment_id} not found in plan.")
                pass

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
            f"{self.indent}{self.indent}self.persistent_title = Tex(r\"\"\"$$\\text{{{title_text}}}$$ \"\"\", font_size=48, color=GOLD)",
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

            # 2. Signature-based Filtering & Correction
            cls_obj = getattr(MC_Geo, cls_name, None)
            
            # [Feature] Class Name Auto-Correction (SmartLine -> _SmartLine)
            if cls_obj is None:
                # Try finding internal class with underscore prefix
                internal_name = f"_{cls_name}"
                internal_cls = getattr(MC_Geo, internal_name, None)
                if internal_cls:
                    # Correction successful
                    cls_name = internal_name
                    cls_obj = internal_cls
                    # print(f"[Generator] Auto-corrected class name: {comp} -> {cls_name}")
                
            if cls_obj:
                data = self._get_valid_params(cls_obj, data)
                # 3. Generate param string using Unified Reconstruction
                param_str = self._reconstruct_param_str(cls_obj, data)
            else:
                # [Fallback] If correction failed, use SafeFallback to prevent crash
                # print(f"[Generator Warning] Component {comp} not found. Using SafeFallback.")
                cls_name = "SafeFallback" # Use the injected local class
                param_str = "" # No params needed for fallback

            # [Note] Auto-parent injection (Rule 2.8) is DISABLED for this JSON protocol.
            # The new protocol expects parent to be explicitly specified in JSON data.
            # Components without explicit parent are treated as independent objects.

            # [Rule 3.1] Fade Transition for non-incremental components
            if not is_incremental and cls_name in self.FACTORY_WHITELIST:
                self.code_lines.extend([
                    f"{self.indent}{self.indent}# [Rule 3.1] Fade Transition: Clear slot before new factory component.",
                    f"{self.indent}{self.indent}slot_content = self.lm.placed_mobs.get('{slot}', [])",
                    f"{self.indent}{self.indent}if slot_content:",
                    f"{self.indent}{self.indent}{self.indent}self.play(*[FadeOut(o) for o in slot_content], run_time=0.3)",
                    f"{self.indent}{self.indent}{self.indent}self.lm.placed_mobs['{slot}'] = []",
                ])

            # Instantiation Line
            if cls_name == "SafeFallback":
                 # Local class, no MC_Geo prefix
                 self.code_lines.append(f"{self.indent}{self.indent}{obj_id} = SafeFallback()")
            elif cls_name.startswith("_"):
                 # Internal class needs explicit access
                 self.code_lines.append(f"{self.indent}{self.indent}{obj_id} = MC_Geo.{cls_name}({param_str})")
            else:
                 # Standard public class
                 self.code_lines.append(f"{self.indent}{self.indent}{obj_id} = MC_Geo.{cls_name}({param_str})")

            # [Rule 8] Update host if factory
            # Note: _SmartLine or SafeFallback unlikely to be factory, but check original name just in case
            original_cls = comp.split(".")[-1]
            if original_cls in self.FACTORY_WHITELIST or cls_name in self.FACTORY_WHITELIST:
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
                         # print(f"[Deep Healing Path A] Sanitized redundant constructor in visual_action for {obj_id}")
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
                        # [Color Fix] Sanitize color values in vis_args (from visual_action string)
                        for k, v in vis_args.items():
                            if k in ["color", "text_color", "stroke_color", "fill_color"] and isinstance(v, str):
                                if not v.startswith("#"):
                                    v_upper = v.upper()
                                    if v_upper in self.SAFE_COLOR_MAP:
                                        if vis_args[k] != self.SAFE_COLOR_MAP[v_upper]:
                                            vis_args[k] = self.SAFE_COLOR_MAP[v_upper]
                                            needs_healing = True # [CRITICAL] 只要转换了，就必须触发重写覆盖原始代码
                                    else:
                                        base_color = re.sub(r'_[A-Z]$', '', v_upper)
                                        if base_color in self.SAFE_COLOR_MAP:
                                            if vis_args[k] != self.SAFE_COLOR_MAP[base_color]:
                                                vis_args[k] = self.SAFE_COLOR_MAP[base_color]
                                                needs_healing = True # [CRITICAL]
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
                            # print(f"[Deep Healing] Re-aligned {host_id}.{method_name} in visual_action.")

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
                content = re.sub(r"\\{2,}", r"\\", content)
                
                # [Fix] CJK Sanitization: Wrap Chinese in \text{} if not already wrapped
                if re.search(r'[\u4e00-\u9fff]', content):
                    if "\\text{" not in content and "$$" not in content:
                        content = f"\\text{{{content}}}"

                # [LaTeX Auto-Repair] 最后防线：对最终输出内容做 LaTeX 语法修复
                content = ManimGenerator._fix_latex_string(content)

                # [Decision] Use Raw Strings & Triple Quotes carefully
                needs_triple = "'" in content or '"' in content or "\n" in content
                
                # Use a delimiter that is NOT in the content (for single-layer raw strings)
                delim = "'" if '"' in content else '"'
                
                if needs_triple:
                    # Bulletproof for f'(x) or strings containing quotes
                    return f'{prefix}r"""{content}"""'
                else:
                    # Always wrap in r"..." or r'...' to be safe and consistent, 
                    # even if no special characters like backslashes are present.
                    # This prevents returning match.group(0) which might have triggered 
                    # the "missing quotes" bug if original r? was poorly handled.
                    return f'{prefix}r{delim}{content}{delim}'

            # [HYPER BULLETPROOF FIX] Support r? prefix and Internal Quote Tolerance.
            # Logic: Match until we find the quote type FOLLOWING a comma, closed paren, or space.
            pattern = re.compile(r'(\bTex\(|\bMathTex\(|\blabel=|\btext=|\blabel_text=|\bside_index=)r?([\'"])((?:(?!\2(?=[),\s]|$)).|\\.)*)\2')
            code = re.sub(pattern, r_replacer, code)
            return code

        # [CRITICAL SYNC] Animation Wrapper (Ported from middle.py)
        # Guarantees that self.play() always receives an Animation, not a Mobject
        def ensure_animation(a):
            if not a: return None
            a_stripped = a.strip()
            
            # [LLM Hallucination Guard] Detect bare placeholder names from LLM.
            # If the string has NO dot, NO parenthesis, and NO known animation keyword,
            # it's likely a hallucinated variable name (e.g., "board_action_only", "none", "visual_placeholder").
            # These would cause NameError at runtime, so we discard them.
            if '.' not in a_stripped and '(' not in a_stripped:
                # It's a bare identifier. Check if it's a known defined variable (from self.defined_ids context).
                # Since we can't check at generation time, we simply reject any bare name
                # that doesn't look like a standard Manim constant (e.g., UP, DOWN, LEFT, RIGHT).
                known_constants = {"UP", "DOWN", "LEFT", "RIGHT", "ORIGIN", "IN", "OUT", "UL", "UR", "DL", "DR"}
                if a_stripped not in known_constants:
                    return None
            # Common animation names in Manim
            anim_keywords = ["Write", "FadeIn", "Create", "FadeOut", "Transform", "ReplacementTransform", "Indicate", "Circumscribe", "Flash", "FocusOn", "Rotate", "Scale", "Shift", "MoveTo", "Become"]
            if any(a.strip().startswith(kw+"(") for kw in anim_keywords):
                return a
            # If it's a dot call like obj.animate.become(...), it's also an animation
            if ".animate." in a:
                return a
            
            # [Promotion] Animation Promotion for Semantic API
            # Form: id.method(...) 
            # Pattern matches <id>.<method>(<any>)
            # Includes move_, set_, shift, rotate, scale
            promo_match = re.match(r"^([\w_]+)\.((?:move_|set_|shift|rotate|scale|apply_)\w+)\((.*)\)$", a.strip())
            if promo_match:
                obj_id = promo_match.group(1)
                method = promo_match.group(2)
                args = promo_match.group(3)
                # Exclusion: set_node_local_data is too low-level for reliable .animate in some contexts
                if "set_node_local_data" not in method:
                    return f"{obj_id}.animate.{method}({args})"

            # [Smooth] Smart Animation Selection
            # Text/Labels -> Write (with _safe() wrapping for None protection)
            if any(kw in a for kw in ["Text", "Tex", "label", "caption"]):
                return f"Write(_safe({a}))"
            # Smart Components (non-incremental creation) -> Create
            if "MC_Geo" in a and "." not in a:
                return f"Create(_safe({a}))"
            # Operations/Marks -> Create or Write
            if "mark" in a or "elbow" in a:
                return f"Create(_safe({a}))"
            
            # General default -> Wrap Mobject method results in FadeIn with _safe()
            return f"FadeIn(_safe({a}))"

        def fix_python_syntax(code):
            """
            [Deep Cleaning] Automatically fixes JS-style syntax and LLM hallucinations in generated code strings.
            - true -> True
            - false -> False
            - null -> None
            - '$$var_name$$' -> var_name (strip LaTeX-wrapped variable references)
            - var_ref -> var (strip _ref suffix from variable names)
            """
            if not code: return code
            # Use word boundaries (\b) to avoid partial matches (e.g. "status_false" -> "status_False")
            code = re.sub(r'\btrue\b', 'True', code)
            code = re.sub(r'\bfalse\b', 'False', code)
            code = re.sub(r'\bnull\b', 'None', code)
            
            # [LLM Hallucination Fix] Strip $$...$$ LaTeX wrapping from variable references.
            # But ONLY if it's likely a Python variable (lowercase_number) to avoid mangling LaTeX labels.
            def syntax_replacer(match):
                full = match.group(0)
                var = match.group(1)
                # Pattern for internal IDs: lowercase name followed by _number (e.g., triangle_1, c_2)
                # OR if it's explicitly in our defined_ids set.
                # [Fix] Only replace if it is DEFINED, to avoid catching latex labels like $$d_1$$
                if var in self.defined_ids:
                    return var
                # If it doesn't look like a variable, preserve original (it might be a LaTeX label like $$T_1$$)
                return full

            code = re.sub(r"""['\"]?\$\$(\w+?)(?:_ref)?\$\$['\"]?""", syntax_replacer, code)
            
            return code

        # Pre-process codes - [Order Fix] Syntax fix MUST come before raw string wrapping
        if vis_code:
            vis_code = wrap_raw_strings(wrap_layout(fix_python_syntax(vis_code), default_slot))
        if brd_code:
            brd_code = wrap_raw_strings(wrap_layout(fix_python_syntax(brd_code), "board"))

        # [Feature] Auto-Translation for Wait()
        if vis_code and "Wait(" in vis_code:
            try:
                 sec = float(re.search(r"Wait\((.*?)\)", vis_code).group(1))
                 vis_code = f"self.wait({sec})"
            except:
                 pass

        self.code_lines.append("")
        self.code_lines.append(f"{self.indent}{self.indent}# Sync Block")
        
        # [Rule 5] Atomic Synchronization Strategy 
        # [Optimized] All objects are now passed through ensure_animation()
        has_speech = bool(speech)
        has_visual = bool(vis_code) and "wait" not in vis_code.lower()
        has_board = bool(brd_code)

        if has_speech:
            self.code_lines.append(f"{self.indent}{self.indent}with self.voiceover(text=r\"\"\"{speech}\"\"\") as tracker:")
            # [Rule 2.5] Caption Logic - Direct switch (no fade animation)
            self.code_lines.extend([
                f"{self.indent}{self.indent}{self.indent}self.current_caption = self.update_caption(self.current_caption, r\"\"\"{speech}\"\"\")",
            ])
            
            # [Fix] Handle "Instant State Change" methods (non-animations)
            # These should be executed immediately, not wrapped in FadeIn/Play
            if vis_code and any(k in vis_code for k in [".reform(", ".set_node_local_data(", ".set_value("]):
                 self.code_lines.append(f"{self.indent}{self.indent}{self.indent}{vis_code}")
                 vis_code = None # Prevent adding to play()

            anims = [ensure_animation(a) for a in [vis_code, brd_code] if a]
            if anims:
                 self.code_lines.append(f"{self.indent}{self.indent}{self.indent}self.play({', '.join(anims)}, run_time=tracker.duration)")
            else:
                 # [Visual Fix] Ensure the segment doesn't "freeze" or "skip" by waiting for the audio duration.
                 self.code_lines.append(f"{self.indent}{self.indent}{self.indent}self.wait(tracker.duration)")
        else:
            # Fallback: Fixed duration if no speech
            fallback_duration = 2.0
            anims = [ensure_animation(a) for a in [vis_code, brd_code] if a]
            if anims:
                 self.code_lines.append(f"{self.indent}{self.indent}self.play({', '.join(anims)}, run_time={fallback_duration})")
            elif vis_code and "wait" in vis_code.lower():
                 self.code_lines.append(f"{self.indent}{self.indent}{vis_code}")
            else:
                 self.code_lines.append(f"{self.indent}{self.indent}self.wait(1)")
