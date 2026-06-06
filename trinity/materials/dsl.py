"""Material DSL: Pythonic surface shader authoring with WGSL codegen.

This module provides the core abstractions for writing material surface shaders
in Python that compile to WGSL. The key components are:

- MaterialMeta: Metaclass that extracts surface() method source and compiles to WGSL
- SurfaceContext: Input proxy providing texture sampling and shader inputs
- SurfaceOutput: Output proxy for PBR material parameters
- Material: Base class for user-defined materials
- surface: Decorator marking the shader entry point

Example usage::

    class GoldMaterial(Material, metaclass=MaterialMeta):
        albedo_texture = Texture2D(default="white", srgb=True)

        @surface
        def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
            out.base_color = ctx.sample(self.albedo_texture, ctx.uv())
            out.metallic = 0.9
            out.roughness = 0.3

The MaterialMeta metaclass will:
1. Extract the surface() method source via inspect
2. Parse it to a Python AST
3. Walk the AST to translate Python expressions to WGSL
4. Store the compiled WGSL on the class as _wgsl_source
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from trinity.materials.textures import Texture2D, TextureCube


# =============================================================================
# WGSL TYPE MARKERS (local copies for standalone operation)
# =============================================================================


class Vec2:
    """WGSL vec2<f32> proxy for material DSL."""
    _wgsl_name = "vec2<f32>"

    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"Vec2({self.x}, {self.y})"


class Vec3:
    """WGSL vec3<f32> proxy for material DSL."""
    _wgsl_name = "vec3<f32>"

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self) -> str:
        return f"Vec3({self.x}, {self.y}, {self.z})"


class Vec4:
    """WGSL vec4<f32> proxy for material DSL."""
    _wgsl_name = "vec4<f32>"

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 1.0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def __repr__(self) -> str:
        return f"Vec4({self.x}, {self.y}, {self.z}, {self.w})"


# =============================================================================
# SURFACE CONTEXT: Shader input proxy
# =============================================================================


@dataclass
class SurfaceContext:
    """Proxy for shader input values available in material surface shaders.

    Provides access to vertex attributes, texture sampling, and built-in
    functions. All methods generate corresponding WGSL code when invoked
    during AST translation.

    Attributes:
        position: World-space position (vec3<f32>)
        normal: World-space normal (vec3<f32>)
        tangent: World-space tangent (vec3<f32>)
        uv: Primary UV coordinates (vec2<f32>)
        vertex_color: Vertex color (vec4<f32>)
        time: Current time in seconds (f32)
    """
    position: Vec3 = field(default_factory=Vec3)
    normal: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 1.0))
    tangent: Vec3 = field(default_factory=lambda: Vec3(1.0, 0.0, 0.0))
    uv: Vec2 = field(default_factory=Vec2)
    vertex_color: Vec4 = field(default_factory=lambda: Vec4(1.0, 1.0, 1.0, 1.0))
    time: float = 0.0

    # Additional inputs
    view_direction: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 1.0))
    screen_uv: Vec2 = field(default_factory=Vec2)

    def sample(self, texture: Any, uv: Tuple[float, float] | Vec2) -> Vec4:
        """Sample a 2D texture at the given UV coordinates.

        Maps to WGSL: textureSample(texture, sampler, uv)

        Args:
            texture: Texture2D descriptor bound to the material
            uv: UV coordinates as tuple or Vec2

        Returns:
            Sampled color as Vec4 (RGBA)
        """
        # Runtime stub - actual sampling happens in WGSL
        return Vec4(1.0, 1.0, 1.0, 1.0)

    def sample_cube(self, texture: Any, direction: Tuple[float, float, float] | Vec3) -> Vec4:
        """Sample a cubemap texture in the given direction.

        Maps to WGSL: textureSample(texture, sampler, direction)

        Args:
            texture: TextureCube descriptor bound to the material
            direction: Sampling direction as tuple or Vec3

        Returns:
            Sampled color as Vec4 (RGBA)
        """
        # Runtime stub - actual sampling happens in WGSL
        return Vec4(1.0, 1.0, 1.0, 1.0)

    def sample_level(self, texture: Any, uv: Tuple[float, float] | Vec2, level: float) -> Vec4:
        """Sample a 2D texture at a specific mip level.

        Maps to WGSL: textureSampleLevel(texture, sampler, uv, level)

        Args:
            texture: Texture2D descriptor bound to the material
            uv: UV coordinates as tuple or Vec2
            level: Mip level to sample (0 = base level)

        Returns:
            Sampled color as Vec4 (RGBA)
        """
        return Vec4(1.0, 1.0, 1.0, 1.0)

    def noise(self, pos: Tuple[float, ...] | Vec2 | Vec3, scale: float = 1.0) -> float:
        """Generate procedural noise at the given position.

        Maps to WGSL: noise(pos * scale)

        Args:
            pos: Position (2D or 3D) to sample noise
            scale: Noise frequency multiplier

        Returns:
            Noise value in range [-1, 1]
        """
        return 0.0

    def world_position(self) -> Vec3:
        """Get the world-space position of the current fragment.

        Maps to WGSL: in.world_position
        """
        return self.position

    def world_normal(self) -> Vec3:
        """Get the world-space normal of the current fragment.

        Maps to WGSL: in.world_normal
        """
        return self.normal

    def world_tangent(self) -> Vec3:
        """Get the world-space tangent of the current fragment.

        Maps to WGSL: in.world_tangent
        """
        return self.tangent

    def get_uv(self) -> Vec2:
        """Get the primary UV coordinates.

        Maps to WGSL: in.uv
        """
        return self.uv

    def get_vertex_color(self) -> Vec4:
        """Get the vertex color.

        Maps to WGSL: in.vertex_color
        """
        return self.vertex_color

    def get_time(self) -> float:
        """Get the current time in seconds (elapsed since animation start).

        Maps to WGSL: uniforms.elapsed_seconds

        Use this for time-based material animations like UV scrolling,
        color pulsing, or emission flickering.
        """
        return self.time

    def get_delta_time(self) -> float:
        """Get the delta time since last frame in seconds.

        Maps to WGSL: uniforms.delta_time

        Use this for frame-rate independent animations.
        """
        return 0.016  # Default ~60 FPS

    def get_frame_count(self) -> int:
        """Get the current frame count.

        Maps to WGSL: uniforms.frame_count
        """
        return 0

    def get_view_direction(self) -> Vec3:
        """Get the view direction (fragment to camera).

        Maps to WGSL: normalize(uniforms.camera_position - in.world_position)
        """
        return self.view_direction


# =============================================================================
# SURFACE OUTPUT: Shader output proxy
# =============================================================================


@dataclass
class SurfaceOutput:
    """Proxy for shader output values in material surface shaders.

    Represents the PBR material parameters that the surface shader produces.
    All fields map directly to the PBROutput WGSL struct.

    Attributes:
        base_color: Albedo color (vec3<f32>), default white
        metallic: Metallic factor (f32), 0.0 = dielectric, 1.0 = metal
        roughness: Roughness factor (f32), 0.0 = smooth, 1.0 = rough
        normal: Tangent-space normal (vec3<f32>), default up
        emissive: Emissive color (vec3<f32>), default black
        ao: Ambient occlusion (f32), 0.0 = fully occluded, 1.0 = none
        alpha: Opacity (f32), for translucent materials
        specular: Specular intensity (f32), for dielectrics
        subsurface: Subsurface scattering factor (f32)
        subsurface_color: Subsurface scattering color (vec3<f32>)
        clearcoat: Clear coat intensity (f32)
        clearcoat_roughness: Clear coat roughness (f32)
        anisotropy: Anisotropy strength (f32)
        anisotropy_direction: Anisotropy direction in tangent space (vec2<f32>)
        sheen: Sheen intensity (f32)
        sheen_color: Sheen tint color (vec3<f32>)
        transmission: Transmission factor (f32) for thin-walled transparency
        ior: Index of refraction (f32) for transmission
    """
    # Core PBR parameters
    base_color: Vec3 = field(default_factory=lambda: Vec3(1.0, 1.0, 1.0))
    metallic: float = 0.0
    roughness: float = 0.5
    normal: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 1.0))
    emissive: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 0.0))
    ao: float = 1.0
    alpha: float = 1.0

    # Extended PBR parameters
    specular: float = 0.5
    subsurface: float = 0.0
    subsurface_color: Vec3 = field(default_factory=lambda: Vec3(1.0, 1.0, 1.0))
    clearcoat: float = 0.0
    clearcoat_roughness: float = 0.03
    anisotropy: float = 0.0
    anisotropy_direction: Vec2 = field(default_factory=lambda: Vec2(1.0, 0.0))
    sheen: float = 0.0
    sheen_color: Vec3 = field(default_factory=lambda: Vec3(1.0, 1.0, 1.0))
    transmission: float = 0.0
    ior: float = 1.5

    # Legacy aliases (deprecated, for backward compatibility)
    @property
    def albedo(self) -> Tuple[float, float, float]:
        """Deprecated: Use base_color instead."""
        return (self.base_color.x, self.base_color.y, self.base_color.z)

    @albedo.setter
    def albedo(self, value: Tuple[float, float, float] | Vec3) -> None:
        """Deprecated: Use base_color instead."""
        if isinstance(value, Vec3):
            self.base_color = value
        else:
            self.base_color = Vec3(value[0], value[1], value[2])

    @property
    def emission(self) -> Tuple[float, float, float]:
        """Deprecated: Use emissive instead."""
        return (self.emissive.x, self.emissive.y, self.emissive.z)

    @emission.setter
    def emission(self, value: Tuple[float, float, float] | Vec3) -> None:
        """Deprecated: Use emissive instead."""
        if isinstance(value, Vec3):
            self.emissive = value
        else:
            self.emissive = Vec3(value[0], value[1], value[2])


# =============================================================================
# SURFACE DECORATOR
# =============================================================================


def surface(func: Callable) -> Callable:
    """Decorator: marks a method as the material surface shader entry point.

    The decorated method will be extracted and compiled to WGSL when the
    class is created. The method signature must be:

        def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:

    Example::

        class MyMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(1.0, 0.5, 0.2)
                out.roughness = 0.4
    """
    func._is_surface = True
    return func


# =============================================================================
# PYTHON TO WGSL AST TRANSLATOR
# =============================================================================


class WGSLTranslationError(Exception):
    """Raised when Python AST cannot be translated to WGSL."""
    pass


class PythonToWGSLTranslator(ast.NodeVisitor):
    """Translates Python AST to WGSL shader code.

    Supports the following AST node types:
    - Expr: Expression statement
    - Assign: Assignment (a = b)
    - AnnAssign: Annotated assignment (a: T = b)
    - Call: Function/method call
    - BinOp: Binary operation (+, -, *, /, etc.)
    - If: Conditional expression
    - Attribute: Attribute access (a.b)
    - Name: Variable name
    - Constant: Literal value
    - Subscript: Index access (a[i])
    - UnaryOp: Unary operation (-, not)
    - Compare: Comparison (<, >, ==, etc.)
    - BoolOp: Boolean operation (and, or)
    - Return: Return statement
    - Tuple: Tuple construction (for vec types)
    """

    # Python operator to WGSL operator mapping
    BINOP_MAP = {
        ast.Add: "+",
        ast.Sub: "-",
        ast.Mult: "*",
        ast.Div: "/",
        ast.Mod: "%",
        ast.Pow: "pow",  # Special case: becomes pow(a, b)
        ast.BitAnd: "&",
        ast.BitOr: "|",
        ast.BitXor: "^",
        ast.LShift: "<<",
        ast.RShift: ">>",
    }

    UNARYOP_MAP = {
        ast.USub: "-",
        ast.Not: "!",
        ast.Invert: "~",
        ast.UAdd: "+",
    }

    CMPOP_MAP = {
        ast.Eq: "==",
        ast.NotEq: "!=",
        ast.Lt: "<",
        ast.LtE: "<=",
        ast.Gt: ">",
        ast.GtE: ">=",
    }

    BOOLOP_MAP = {
        ast.And: "&&",
        ast.Or: "||",
    }

    # Built-in function mapping (Python name -> WGSL name)
    # Core WGSL built-ins
    BUILTIN_MAP = {
        "abs": "abs",
        "min": "min",
        "max": "max",
        "pow": "pow",
        "sqrt": "sqrt",
        "sin": "sin",
        "cos": "cos",
        "tan": "tan",
        "asin": "asin",
        "acos": "acos",
        "atan": "atan",
        "atan2": "atan2",
        "exp": "exp",
        "exp2": "exp2",
        "log": "log",
        "log2": "log2",
        "floor": "floor",
        "ceil": "ceil",
        "round": "round",
        "fract": "fract",
        "trunc": "trunc",
        "sign": "sign",
        "clamp": "clamp",
        "saturate": "saturate",
        "mix": "mix",
        "lerp": "mix",  # Alias
        "step": "step",
        "smoothstep": "smoothstep",
        "length": "length",
        "distance": "distance",
        "dot": "dot",
        "cross": "cross",
        "normalize": "normalize",
        "reflect": "reflect",
        "refract": "refract",
        "faceforward": "faceforward",
        # Custom builtins (require WGSL helper functions from builtins.py)
        # Noise functions
        "value_noise": "value_noise",
        "perlin_noise": "perlin_noise",
        "simplex_noise": "simplex_noise",
        "worley_noise": "worley_noise",
        "fbm": "fbm",
        "turbulence": "turbulence",
        # Color conversion
        "rgb_to_hsv": "rgb_to_hsv",
        "hsv_to_rgb": "hsv_to_rgb",
        "srgb_to_linear": "srgb_to_linear",
        "linear_to_srgb": "linear_to_srgb",
        # Tonemapping
        "tonemap_reinhard": "tonemap_reinhard",
        "tonemap_aces": "tonemap_aces",
        "tonemap_uncharted2": "tonemap_uncharted2",
        "tonemap_agx": "tonemap_agx",
        # Math utilities
        "remap": "remap",
        "inverse_lerp": "inverse_lerp",
        "smooth_min": "smooth_min",
        "smooth_max": "smooth_max",
        "smootherstep": "smootherstep",
    }

    # Custom builtins that require WGSL helper functions
    CUSTOM_BUILTINS = {
        "value_noise", "perlin_noise", "simplex_noise", "worley_noise",
        "fbm", "turbulence",
        "rgb_to_hsv", "hsv_to_rgb", "srgb_to_linear", "linear_to_srgb",
        "tonemap_reinhard", "tonemap_aces", "tonemap_uncharted2", "tonemap_agx",
        "remap", "inverse_lerp", "smooth_min", "smooth_max", "smootherstep",
    }

    # Context method mapping (SurfaceContext.method -> WGSL)
    CONTEXT_METHOD_MAP = {
        "sample": "textureSample",
        "sample_cube": "textureSample",
        "sample_level": "textureSampleLevel",
        "world_position": "in.world_position",
        "world_normal": "in.world_normal",
        "world_tangent": "in.world_tangent",
        "get_uv": "in.uv",
        "uv": "in.uv",
        "get_vertex_color": "in.vertex_color",
        "vertex_color": "in.vertex_color",
        # Time accessors (T-MAT-5.5)
        "get_time": "time_uniforms.elapsed_seconds",
        "time": "time_uniforms.elapsed_seconds",
        "get_delta_time": "time_uniforms.delta_time",
        "delta_time": "time_uniforms.delta_time",
        "get_frame_count": "time_uniforms.frame_count",
        "frame_count": "time_uniforms.frame_count",
        # View direction
        "get_view_direction": "normalize(uniforms.camera_position - in.world_position)",
        "view_direction": "normalize(uniforms.camera_position - in.world_position)",
    }

    # Output field mapping (SurfaceOutput.field -> PBROutput.field)
    OUTPUT_FIELD_MAP = {
        "base_color": "out.base_color",
        "albedo": "out.base_color",  # Alias
        "metallic": "out.metallic",
        "roughness": "out.roughness",
        "normal": "out.normal",
        "emissive": "out.emissive",
        "emission": "out.emissive",  # Alias
        "ao": "out.ambient_occlusion",
        "alpha": "out.alpha",
        "specular": "out.specular",
        "subsurface": "out.subsurface",
        "subsurface_color": "out.subsurface_color",
        "clearcoat": "out.clearcoat",
        "clearcoat_roughness": "out.clearcoat_roughness",
        "anisotropy": "out.anisotropy",
        "anisotropy_direction": "out.anisotropy_direction",
        "sheen": "out.sheen",
        "sheen_color": "out.sheen_color",
        "transmission": "out.transmission",
        "ior": "out.ior",
    }

    def __init__(self, parent_wgsl: Optional[str] = None, parent_name: Optional[str] = None):
        """Initialize the WGSL translator.

        Args:
            parent_wgsl: Optional WGSL source from parent material (for inheritance)
            parent_name: Optional name of parent material class
        """
        self.indent_level = 0
        self.lines: list[str] = []
        self.used_builtins: set[str] = set()  # Track custom builtins used
        self.parent_wgsl = parent_wgsl
        self.parent_name = parent_name
        self.has_super_call = False  # Track if super() was called

    def indent(self) -> str:
        """Return current indentation string."""
        return "    " * self.indent_level

    def emit(self, line: str) -> None:
        """Emit a line of WGSL code with current indentation."""
        self.lines.append(f"{self.indent()}{line}")

    def translate(self, tree: ast.AST) -> str:
        """Translate an AST tree to WGSL code."""
        self.lines = []
        self.indent_level = 0
        self.used_builtins = set()
        self.has_super_call = False
        self.visit(tree)
        return "\n".join(self.lines)

    def _is_super_call(self, node: ast.Call) -> bool:
        """Detect super().surface(ctx, out) pattern for inheritance.

        Matches patterns like:
        - super().surface(ctx, out)
        - super().surface(ctx)

        Args:
            node: AST Call node to check

        Returns:
            True if this is a super().surface() call
        """
        if not isinstance(node.func, ast.Attribute):
            return False

        # Check if calling .surface method
        if node.func.attr != "surface":
            return False

        # Check if the object is a super() call
        if not isinstance(node.func.value, ast.Call):
            return False

        # Check if it's super()
        if isinstance(node.func.value.func, ast.Name):
            if node.func.value.func.id == "super":
                return True

        return False

    def _emit_parent_surface_call(self) -> str:
        """Generate WGSL code to call parent surface logic.

        When super().surface(ctx, out) is called in Python, we inline
        the parent's surface shader body to simulate inheritance.

        Returns:
            WGSL code that represents the parent surface call
        """
        if self.parent_wgsl is None:
            raise WGSLTranslationError(
                "super().surface() called but no parent material found. "
                "Ensure the base class has a @surface method."
            )

        self.has_super_call = True
        # Return a comment indicating parent call - actual inlining done at class level
        parent_ref = self.parent_name or "parent"
        return f"/* super().surface() -> {parent_ref} */"

    def get_used_builtins(self) -> set[str]:
        """Get the set of custom builtins used in the last translation."""
        return self.used_builtins.copy()

    def visit_Module(self, node: ast.Module) -> None:
        """Visit module node (top-level container)."""
        for stmt in node.body:
            self.visit(stmt)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition (surface shader entry)."""
        # Skip the function signature - we extract just the body
        for stmt in node.body:
            self.visit(stmt)

    def visit_Expr(self, node: ast.Expr) -> None:
        """Visit expression statement."""
        expr = self.visit(node.value)
        if expr:
            self.emit(f"{expr};")

    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit assignment statement."""
        value = self.visit(node.value)
        for target in node.targets:
            target_str = self.visit(target)
            self.emit(f"{target_str} = {value};")

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Visit annotated assignment (type-hinted variable)."""
        if node.value is None:
            return  # Declaration without value
        target = self.visit(node.target)
        value = self.visit(node.value)
        # In WGSL, we use let for local variables
        self.emit(f"let {target} = {value};")

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Visit augmented assignment (+=, -=, etc.)."""
        target = self.visit(node.target)
        value = self.visit(node.value)
        op = self.BINOP_MAP.get(type(node.op), "+")
        self.emit(f"{target} = {target} {op} {value};")

    def visit_If(self, node: ast.If) -> str:
        """Visit if statement."""
        test = self.visit(node.test)
        self.emit(f"if ({test}) {{")
        self.indent_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent_level -= 1

        if node.orelse:
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                # elif chain
                self.emit("} else ")
                self.visit(node.orelse[0])
            else:
                self.emit("} else {")
                self.indent_level += 1
                for stmt in node.orelse:
                    self.visit(stmt)
                self.indent_level -= 1
                self.emit("}")
        else:
            self.emit("}")
        return ""

    def visit_Return(self, node: ast.Return) -> None:
        """Visit return statement."""
        if node.value:
            value = self.visit(node.value)
            self.emit(f"return {value};")
        else:
            self.emit("return;")

    def visit_Pass(self, node: ast.Pass) -> None:
        """Visit pass statement (no-op in WGSL)."""
        pass  # WGSL doesn't need explicit pass

    def visit_BinOp(self, node: ast.BinOp) -> str:
        """Visit binary operation."""
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)

        if op_type == ast.Pow:
            return f"pow({left}, {right})"

        op = self.BINOP_MAP.get(op_type, "+")
        return f"({left} {op} {right})"

    def visit_UnaryOp(self, node: ast.UnaryOp) -> str:
        """Visit unary operation."""
        operand = self.visit(node.operand)
        op = self.UNARYOP_MAP.get(type(node.op), "-")
        return f"({op}{operand})"

    def visit_Compare(self, node: ast.Compare) -> str:
        """Visit comparison expression."""
        left = self.visit(node.left)
        parts = [left]

        for op, comparator in zip(node.ops, node.comparators):
            op_str = self.CMPOP_MAP.get(type(op), "==")
            comp = self.visit(comparator)
            parts.append(f"{op_str} {comp}")

        return " ".join(parts)

    def visit_BoolOp(self, node: ast.BoolOp) -> str:
        """Visit boolean operation (and, or)."""
        op = self.BOOLOP_MAP.get(type(node.op), "&&")
        values = [self.visit(v) for v in node.values]
        return f"({f' {op} '.join(values)})"

    def visit_IfExp(self, node: ast.IfExp) -> str:
        """Visit ternary expression (a if cond else b)."""
        test = self.visit(node.test)
        body = self.visit(node.body)
        orelse = self.visit(node.orelse)
        return f"select({orelse}, {body}, {test})"

    def visit_Call(self, node: ast.Call) -> str:
        """Visit function/method call."""
        # Check for super().surface() call FIRST, before visiting args
        if self._is_super_call(node):
            return self._emit_parent_surface_call()

        args = [self.visit(arg) for arg in node.args]

        if isinstance(node.func, ast.Attribute):
            # Method call: obj.method(args)
            obj = self.visit(node.func.value)
            method = node.func.attr

            # Handle context methods
            if obj == "ctx" and method in self.CONTEXT_METHOD_MAP:
                wgsl_method = self.CONTEXT_METHOD_MAP[method]
                if method in ("sample", "sample_cube", "sample_level"):
                    # Texture sampling needs special handling
                    return f"{wgsl_method}({', '.join(args)})"
                elif callable(wgsl_method) or "(" in wgsl_method:
                    return wgsl_method
                else:
                    return wgsl_method

            # Handle vector constructors
            if obj in ("Vec2", "Vec3", "Vec4"):
                wgsl_type = f"vec{obj[-1]}<f32>"
                return f"{wgsl_type}({', '.join(args)})"

            # Generic method call
            return f"{obj}.{method}({', '.join(args)})"

        elif isinstance(node.func, ast.Name):
            func_name = node.func.id

            # Handle built-in functions
            if func_name in self.BUILTIN_MAP:
                wgsl_func = self.BUILTIN_MAP[func_name]
                # Track custom builtins that need WGSL helper functions
                if func_name in self.CUSTOM_BUILTINS:
                    self.used_builtins.add(func_name)
                return f"{wgsl_func}({', '.join(args)})"

            # Handle vector constructors
            if func_name in ("Vec2", "Vec3", "Vec4"):
                wgsl_type = f"vec{func_name[-1]}<f32>"
                return f"{wgsl_type}({', '.join(args)})"

            # Generic function call
            return f"{func_name}({', '.join(args)})"

        # Fallback
        func = self.visit(node.func)
        return f"{func}({', '.join(args)})"

    def visit_Attribute(self, node: ast.Attribute) -> str:
        """Visit attribute access."""
        obj = self.visit(node.value)
        attr = node.attr

        # Handle output field mapping
        if obj == "out" and attr in self.OUTPUT_FIELD_MAP:
            return self.OUTPUT_FIELD_MAP[attr]

        # Handle context property access
        if obj == "ctx":
            if attr in self.CONTEXT_METHOD_MAP:
                wgsl = self.CONTEXT_METHOD_MAP[attr]
                if "(" not in wgsl:
                    return wgsl
            # Direct property access
            return f"in.{attr}"

        # Handle self.texture references
        if obj == "self":
            return f"material.{attr}"

        return f"{obj}.{attr}"

    def visit_Name(self, node: ast.Name) -> str:
        """Visit variable name."""
        name = node.id

        # Remap special names
        if name == "True":
            return "true"
        elif name == "False":
            return "false"
        elif name == "None":
            return "0.0"  # WGSL has no null

        return name

    def visit_Constant(self, node: ast.Constant) -> str:
        """Visit literal constant."""
        value = node.value

        if isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, (int, float)):
            # Ensure floats have decimal point
            s = str(value)
            if isinstance(value, float) and "." not in s and "e" not in s.lower():
                s += ".0"
            return s
        elif isinstance(value, str):
            # Strings are used for texture paths in Python but not in WGSL
            return f'"{value}"'
        elif value is None:
            return "0.0"

        return str(value)

    def visit_Subscript(self, node: ast.Subscript) -> str:
        """Visit subscript (index) access."""
        obj = self.visit(node.value)

        if isinstance(node.slice, ast.Constant):
            idx = self.visit(node.slice)
        else:
            idx = self.visit(node.slice)

        return f"{obj}[{idx}]"

    def visit_Tuple(self, node: ast.Tuple) -> str:
        """Visit tuple (treated as vec constructor in WGSL)."""
        elements = [self.visit(e) for e in node.elts]
        n = len(elements)

        if n == 2:
            return f"vec2<f32>({', '.join(elements)})"
        elif n == 3:
            return f"vec3<f32>({', '.join(elements)})"
        elif n == 4:
            return f"vec4<f32>({', '.join(elements)})"

        # Fallback for other sizes
        return f"({', '.join(elements)})"

    def visit_List(self, node: ast.List) -> str:
        """Visit list (same as tuple for vec construction)."""
        return self.visit_Tuple(ast.Tuple(elts=node.elts, ctx=node.ctx))

    def generic_visit(self, node: ast.AST) -> str:
        """Fallback for unsupported node types."""
        raise WGSLTranslationError(
            f"Unsupported Python construct: {type(node).__name__}. "
            f"Material DSL only supports a subset of Python for shader authoring."
        )


# =============================================================================
# MATERIAL METACLASS
# =============================================================================


class MaterialMeta(type):
    """Metaclass for material classes that compiles surface() to WGSL.

    When a class is created with this metaclass, it:
    1. Finds the surface() method (decorated with @surface or named 'surface')
    2. Extracts its source code via inspect
    3. Parses the source to a Python AST
    4. Walks the AST to generate WGSL shader code
    5. Stores the result on the class as _wgsl_source

    The class also gets:
    - _surface_method: The original Python method
    - _compilation_error: Any error during compilation (or None)
    - _texture_bindings: Dict of texture descriptors
    - _inherited_textures: Dict of textures inherited from parent classes
    - _parent_material: Reference to first parent with surface method (or None)
    - _has_super_call: True if surface() calls super().surface()

    Inheritance Support (T-MAT-5.2):
    - Child classes inherit texture slots from parent classes
    - Child can override surface method while calling super().surface()
    - super().surface(ctx, out) inlines parent WGSL into child shader
    - MRO is respected for multiple inheritance

    Usage::

        class MyMaterial(Material, metaclass=MaterialMeta):
            albedo = Texture2D(default="white", srgb=True)

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                color = ctx.sample(self.albedo, ctx.uv)
                out.base_color = color.xyz
                out.roughness = 0.5

        # Inheritance example:
        class ChildMaterial(MyMaterial, metaclass=MaterialMeta):
            specular_map = Texture2D(default="white")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                super().surface(ctx, out)  # Apply parent shader
                out.roughness *= ctx.sample(self.specular_map, ctx.uv).r

        # Access compiled WGSL:
        print(ChildMaterial._wgsl_source)
    """

    # Registry of all material classes
    _registry: dict[str, type] = {}

    def __new__(mcs, name: str, bases: tuple, namespace: dict, **kwargs):
        """Create the material class and compile its surface shader."""
        cls = super().__new__(mcs, name, bases, namespace)

        # Skip compilation for the base Material class
        if name == "Material" and not bases:
            return cls

        # Initialize class attributes
        cls._wgsl_source = ""
        cls._surface_method = None
        cls._compilation_error = None
        cls._texture_bindings = {}
        cls._inherited_textures = {}
        cls._used_builtins: set[str] = set()
        cls._parent_material = None
        cls._has_super_call = False

        # =====================================================================
        # INHERITANCE: Collect texture slots from parent classes (MRO order)
        # =====================================================================
        inherited_textures = {}
        parent_material = None
        parent_wgsl = None
        parent_name = None

        for base in bases:
            # Collect inherited texture bindings
            if hasattr(base, "_texture_bindings"):
                for tex_name, tex_desc in base._texture_bindings.items():
                    if tex_name not in inherited_textures:
                        inherited_textures[tex_name] = tex_desc

            # Also collect from _inherited_textures for multi-level inheritance
            if hasattr(base, "_inherited_textures"):
                for tex_name, tex_desc in base._inherited_textures.items():
                    if tex_name not in inherited_textures:
                        inherited_textures[tex_name] = tex_desc

            # Find first parent with a surface method (for super() support)
            if parent_material is None and hasattr(base, "_surface_method") and base._surface_method:
                parent_material = base
                parent_wgsl = getattr(base, "_wgsl_source", None)
                parent_name = base.__name__

        cls._inherited_textures = inherited_textures
        cls._parent_material = parent_material

        # =====================================================================
        # Find surface method in this class or inherit from parent
        # =====================================================================
        surface_method = None
        owns_surface = False  # True if this class defines its own surface()

        for attr_name, attr_value in namespace.items():
            if callable(attr_value):
                # Check for @surface decorator
                if getattr(attr_value, "_is_surface", False):
                    surface_method = attr_value
                    owns_surface = True
                    break
                # Or named 'surface'
                elif attr_name == "surface" and attr_name != "__init__":
                    surface_method = attr_value
                    owns_surface = True
                    break

        # Inherit surface from parent if not overridden
        if surface_method is None and parent_material is not None:
            surface_method = parent_material._surface_method
            owns_surface = False

        if surface_method is not None:
            cls._surface_method = surface_method

            # =====================================================================
            # Extract texture bindings (own + inherited)
            # =====================================================================
            # Start with inherited textures
            cls._texture_bindings = dict(inherited_textures)

            # Add/override with own texture bindings
            for attr_name, attr_value in namespace.items():
                if hasattr(attr_value, "_is_texture_descriptor"):
                    cls._texture_bindings[attr_name] = attr_value

            # =====================================================================
            # Compile to WGSL (with inheritance support)
            # =====================================================================
            try:
                if owns_surface:
                    # This class has its own surface - compile with parent context
                    cls._wgsl_source, cls._used_builtins, cls._has_super_call = mcs._compile_surface(
                        cls, surface_method, parent_wgsl=parent_wgsl, parent_name=parent_name
                    )
                else:
                    # Inherited surface - copy parent WGSL
                    cls._wgsl_source = parent_wgsl or ""
                    cls._used_builtins = getattr(parent_material, "_used_builtins", set()).copy()
                    cls._has_super_call = getattr(parent_material, "_has_super_call", False)
            except Exception as e:
                cls._compilation_error = e
                cls._wgsl_source = f"// Compilation error: {e}"

        # Register the material class
        mcs._registry[name] = cls

        return cls

    @staticmethod
    def _compile_surface(
        cls: type,
        method: Callable,
        parent_wgsl: Optional[str] = None,
        parent_name: Optional[str] = None,
    ) -> tuple[str, set[str], bool]:
        """Compile a surface() method to WGSL shader code.

        Args:
            cls: The material class being created
            method: The surface() method to compile
            parent_wgsl: Optional WGSL source from parent material
            parent_name: Optional name of parent material class

        Returns:
            Tuple of (WGSL shader code, set of used builtin function names, has_super_call)
        """
        # Get source code
        try:
            source = inspect.getsource(method)
        except OSError as e:
            raise WGSLTranslationError(f"Cannot retrieve source for {method}: {e}")

        # Dedent and parse
        source = textwrap.dedent(source)
        tree = ast.parse(source)

        # Translate to WGSL with parent context for inheritance
        translator = PythonToWGSLTranslator(parent_wgsl=parent_wgsl, parent_name=parent_name)
        wgsl_body = translator.translate(tree)
        used_builtins = translator.get_used_builtins()
        has_super_call = translator.has_super_call

        # If super() was called, inline parent WGSL
        if has_super_call and parent_wgsl:
            # Replace the super() comment with actual parent code
            # The marker appears as "/* super().surface() -> ParentName */;" (with semicolon)
            super_marker = f"/* super().surface() -> {parent_name} */"

            # Indent parent code to match current context
            parent_lines = parent_wgsl.strip().split("\n")
            # Get parent body lines (skip empty lines)
            indented_parent = "\n".join(f"    {line}" for line in parent_lines if line.strip())

            # Replace marker with or without indentation/semicolon variations
            # Pattern 1: "    marker;" (indented with semicolon)
            if f"    {super_marker};" in wgsl_body:
                wgsl_body = wgsl_body.replace(
                    f"    {super_marker};",
                    f"    // BEGIN parent material ({parent_name})\n{indented_parent}\n    // END parent material"
                )
            # Pattern 2: "marker;" (no indent, with semicolon)
            elif f"{super_marker};" in wgsl_body:
                wgsl_body = wgsl_body.replace(
                    f"{super_marker};",
                    f"// BEGIN parent material ({parent_name})\n{indented_parent}\n// END parent material"
                )
            # Pattern 3: just the marker (no semicolon)
            elif super_marker in wgsl_body:
                wgsl_body = wgsl_body.replace(
                    super_marker,
                    f"// BEGIN parent material ({parent_name})\n{indented_parent}\n// END parent material"
                )

            # Merge parent builtins
            parent_builtins = getattr(cls._parent_material, "_used_builtins", set())
            used_builtins = used_builtins.union(parent_builtins)

        return wgsl_body, used_builtins, has_super_call

    def __init_subclass__(cls, **kwargs):
        """Hook called when a subclass is created."""
        super().__init_subclass__(**kwargs)

    @classmethod
    def get_material(mcs, name: str) -> Optional[type]:
        """Get a registered material class by name."""
        return mcs._registry.get(name)

    @classmethod
    def all_materials(mcs) -> list[type]:
        """Get all registered material classes."""
        return list(mcs._registry.values())


# =============================================================================
# MATERIAL BASE CLASS
# =============================================================================


class Material:
    """Base class for user-defined materials. Override surface().

    Materials define how surfaces appear when rendered. The surface() method
    is compiled to WGSL shader code that runs on the GPU.

    Example::

        class BrickMaterial(Material, metaclass=MaterialMeta):
            albedo = Texture2D(path="textures/brick_albedo.png", srgb=True)
            normal = Texture2D(path="textures/brick_normal.png")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                uv = ctx.uv
                out.base_color = ctx.sample(self.albedo, uv).xyz
                out.normal = ctx.sample(self.normal, uv).xyz * 2.0 - 1.0
                out.roughness = 0.8
                out.metallic = 0.0

    Attributes:
        _wgsl_source: Compiled WGSL shader code
        _surface_method: The original Python surface() method
        _compilation_error: Any error during compilation, or None
        _texture_bindings: Dict mapping texture names to descriptors
    """

    def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
        """Override this method to define the material's appearance.

        Args:
            ctx: Input context with vertex attributes and sampling functions
            out: Output structure to write PBR parameters
        """
        pass

    @classmethod
    def get_wgsl(cls) -> str:
        """Get the compiled WGSL shader code.

        Returns:
            WGSL shader code string, or empty string if not compiled.

        Raises:
            RuntimeError: If compilation failed.
        """
        if hasattr(cls, "_compilation_error") and cls._compilation_error:
            raise RuntimeError(f"Material compilation failed: {cls._compilation_error}")
        return getattr(cls, "_wgsl_source", "")

    @classmethod
    def has_texture(cls, name: str) -> bool:
        """Check if the material has a texture binding with the given name."""
        return name in getattr(cls, "_texture_bindings", {})

    @classmethod
    def get_textures(cls) -> dict[str, Any]:
        """Get all texture bindings for this material."""
        return getattr(cls, "_texture_bindings", {}).copy()

    @classmethod
    def get_inherited_textures(cls) -> dict[str, Any]:
        """Get texture bindings inherited from parent classes.

        Returns:
            Dict mapping texture names to descriptors inherited from parents.
            Does not include textures defined directly on this class.
        """
        return getattr(cls, "_inherited_textures", {}).copy()

    @classmethod
    def get_own_textures(cls) -> dict[str, Any]:
        """Get texture bindings defined directly on this class.

        Returns:
            Dict mapping texture names to descriptors defined on this class,
            excluding inherited textures.
        """
        all_textures = getattr(cls, "_texture_bindings", {})
        inherited = getattr(cls, "_inherited_textures", {})
        return {k: v for k, v in all_textures.items() if k not in inherited}

    @classmethod
    def get_parent_material(cls) -> Optional[type]:
        """Get the parent material class (first base with surface method).

        Returns:
            Parent material class, or None if no parent has a surface method.
        """
        return getattr(cls, "_parent_material", None)

    @classmethod
    def has_super_call(cls) -> bool:
        """Check if this material's surface() calls super().surface().

        Returns:
            True if super().surface() is called in the surface method.
        """
        return getattr(cls, "_has_super_call", False)

    @classmethod
    def get_inheritance_chain(cls) -> list[type]:
        """Get the chain of material classes in inheritance order.

        Returns:
            List of material classes from this class to the root parent,
            following the MRO but only including classes with surface methods.
        """
        chain = [cls]
        current = cls
        while True:
            parent = getattr(current, "_parent_material", None)
            if parent is None:
                break
            chain.append(parent)
            current = parent
        return chain
