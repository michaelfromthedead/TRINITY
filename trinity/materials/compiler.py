"""AST to WGSL compiler for Material DSL."""
import textwrap
import ast
import inspect
from typing import TYPE_CHECKING, Any, Optional, Dict, Tuple
from itertools import product
import re

if TYPE_CHECKING:
    from trinity.materials.variants import VariantConfig


class MaterialCompiler:
    """Compiles Material subclasses to WGSL shader code.

    Supports:
    - Basic material compilation via compile()
    - Variant compilation with quality tiers via compile_with_variants()
    - Compile all 75 variants via compile_all_variants()
    - Type mapping from Python types to WGSL types

    Args:
        include_pbr_template: If True, include PBR shader template wrapper.
        variant_config: Optional variant configuration for shader generation.
    """

    TYPE_MAP = {float: "f32", int: "i32", bool: "bool", str: "str"}

    def __init__(
        self,
        include_pbr_template: bool = True,
        variant_config: Optional["VariantConfig"] = None,
    ):
        """Initialize the compiler.

        Args:
            include_pbr_template: Whether to wrap output in PBR template.
            variant_config: Optional variant configuration for const declarations.
        """
        self.include_pbr_template = include_pbr_template
        self.variant_config = variant_config

    def compile(self, material_class: type) -> str:
        """Compile a Material subclass to WGSL surface function body.

        Args:
            material_class: Material subclass with a surface() method.

        Returns:
            WGSL shader code string.
        """
        parts = []

        # If variant_config is set, emit const declarations first
        if self.variant_config is not None:
            const_decls = self.variant_config.generate_const_declarations()
            parts.append(const_decls)
            parts.append("")

        # Add default blend consts if no variant config (for default blend handling)
        if self.variant_config is None and self.include_pbr_template:
            parts.append(self._get_default_blend_consts())
            parts.append("")

        # Add PBR structs if requested
        if self.include_pbr_template:
            parts.append(self._get_pbr_structs())
            parts.append("")
            parts.append(self._get_blend_mode_function())
            parts.append("")

        # Emit texture bindings if material has them
        if hasattr(material_class, '_texture_bindings'):
            texture_bindings = material_class._texture_bindings
            for name, desc in texture_bindings.items():
                if hasattr(desc, 'generate_wgsl_binding'):
                    parts.append(desc.generate_wgsl_binding(name))

        # Inject builtin helper functions if used by material
        if hasattr(material_class, '_used_builtins'):
            used_builtins = material_class._used_builtins
            builtin_helpers = self._get_builtin_helpers(used_builtins)
            if builtin_helpers:
                parts.append(builtin_helpers)
                parts.append("")

        # Try to compile the surface method
        if not hasattr(material_class, 'surface') or material_class.surface is None:
            raise ValueError(
                f"Material class {material_class.__name__} has no surface method"
            )

        try:
            source = textwrap.dedent(inspect.getsource(material_class.surface))
            tree = ast.parse(source)
            # Walk AST, translate Python expressions to WGSL
            body = self._walk(tree)
            parts.append(body)
        except (OSError, TypeError):
            # Fallback for dynamically created classes
            parts.append("// WGSL surface body placeholder")

        return "\n".join(parts) if parts else "// WGSL surface body placeholder"

    def _get_builtin_helpers(self, used_builtins: set) -> str:
        """Get WGSL implementations of used custom builtins.

        Args:
            used_builtins: Set of builtin function names used in the material.

        Returns:
            WGSL code string with helper function implementations.
        """
        helpers = []

        # Noise builtins require hash functions
        noise_builtins = {'perlin_noise', 'simplex_noise', 'value_noise', 'worley_noise', 'fbm', 'turbulence'}
        if used_builtins & noise_builtins:
            helpers.append(self._get_noise_helpers())

        # Color builtins
        color_builtins = {'rgb_to_hsv', 'hsv_to_rgb', 'srgb_to_linear', 'linear_to_srgb',
                          'tonemap_reinhard', 'tonemap_aces', 'tonemap_uncharted2', 'tonemap_agx'}
        if used_builtins & color_builtins:
            helpers.append(self._get_color_helpers(used_builtins & color_builtins))

        # Math utility builtins
        math_builtins = {'remap', 'inverse_lerp', 'smooth_min', 'smooth_max', 'smootherstep'}
        if used_builtins & math_builtins:
            helpers.append(self._get_math_helpers(used_builtins & math_builtins))

        return "\n\n".join(helpers)

    def _get_noise_helpers(self) -> str:
        """Get WGSL noise helper functions."""
        return """// Noise helper functions
fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

fn perlin_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}"""

    def _get_color_helpers(self, used: set) -> str:
        """Get WGSL color helper functions."""
        helpers = ["// Color helper functions"]

        if 'rgb_to_hsv' in used:
            helpers.append("""fn rgb_to_hsv(c: vec3<f32>) -> vec3<f32> {
    let K = vec4<f32>(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    let p = mix(vec4<f32>(c.bg, K.wz), vec4<f32>(c.gb, K.xy), step(c.b, c.g));
    let q = mix(vec4<f32>(p.xyw, c.r), vec4<f32>(c.r, p.yzx), step(p.x, c.r));
    let d = q.x - min(q.w, q.y);
    let e = 1.0e-10;
    return vec3<f32>(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}""")

        if 'hsv_to_rgb' in used:
            helpers.append("""fn hsv_to_rgb(c: vec3<f32>) -> vec3<f32> {
    let K = vec4<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, vec3<f32>(0.0), vec3<f32>(1.0)), c.y);
}""")

        return "\n\n".join(helpers)

    def _get_math_helpers(self, used: set) -> str:
        """Get WGSL math utility functions."""
        helpers = ["// Math utility functions"]

        if 'remap' in used:
            helpers.append("""fn remap(value: f32, in_min: f32, in_max: f32, out_min: f32, out_max: f32) -> f32 {
    return out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min);
}""")

        if 'inverse_lerp' in used:
            helpers.append("""fn inverse_lerp(a: f32, b: f32, value: f32) -> f32 {
    return (value - a) / (b - a);
}""")

        if 'smooth_min' in used:
            helpers.append("""fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    let h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}""")

        if 'smooth_max' in used:
            helpers.append("""fn smooth_max(a: f32, b: f32, k: f32) -> f32 {
    return -smooth_min(-a, -b, k);
}""")

        if 'smootherstep' in used:
            helpers.append("""fn smootherstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}""")

        return "\n\n".join(helpers)

    def _get_pbr_structs(self) -> str:
        """Get PBR shader struct definitions."""
        return """struct PBRInput {
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    normal: vec3<f32>,
    emissive: vec3<f32>,
    ao: f32,
}

struct PBRParams {
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    normal: vec3<f32>,
    emissive: vec3<f32>,
    ao: f32,
}

struct SurfaceOutput {
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    normal: vec3<f32>,
    emissive: vec3<f32>,
    ao: f32,
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
}

struct FragmentInput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> FragmentInput {
    var out: FragmentInput;
    out.position = vec4<f32>(input.position, 1.0);
    out.uv = input.uv;
    out.normal = input.normal;
    out.tangent = input.tangent;
    return out;
}"""

    def _get_default_blend_consts(self) -> str:
        """Get default blend mode constants when no variant config is set."""
        return """// Default blend mode constants
const ALPHA_TEST_ENABLED: bool = false;
const ALPHA_BLEND_ENABLED: bool = false;
const BLEND_OPAQUE: bool = true;
const BLEND_MASKED: bool = false;
const BLEND_TRANSLUCENT: bool = false;
const BLEND_ADDITIVE: bool = false;
const BLEND_MODULATE: bool = false;"""

    def _get_blend_mode_function(self) -> str:
        """Get the apply_blend_mode function for alpha handling."""
        return """// Blend mode handling
fn apply_blend_mode(color: vec4<f32>) -> vec4<f32> {
    // Handle alpha test (MASKED blend mode)
    if (ALPHA_TEST_ENABLED) {
        if (color.a < 0.5) {
            discard;
        }
    }

    // Handle opaque blend mode (force alpha = 1.0)
    if (BLEND_OPAQUE) {
        return vec4<f32>(color.rgb, 1.0);
    }

    // For translucent, additive, modulate - preserve alpha
    return color;
}"""

    def compile_with_variants(
        self,
        material_class: type,
        config: "VariantConfig",
    ) -> str:
        """Compile a Material with variant configuration.

        This method compiles the material with quality tier-specific optimizations
        and domain-specific shader code. Temporarily sets variant_config for the
        duration of compilation.

        Args:
            material_class: Material subclass to compile.
            config: Variant configuration specifying domain, blend mode, and quality.

        Returns:
            WGSL shader code optimized for the specified configuration.
        """
        # Save current config and temporarily set the new one
        old_config = self.variant_config
        self.variant_config = config

        try:
            # Compile with variant config applied
            wgsl = self.compile(material_class)
        finally:
            # Restore original config
            self.variant_config = old_config

        return wgsl

    def compile_all_variants(
        self,
        material_class: type,
    ) -> Dict[int, str]:
        """Compile all 75 material variants (5 domains x 5 blends x 3 qualities).

        Args:
            material_class: Material subclass to compile.

        Returns:
            Dictionary mapping variant key (int) to WGSL shader code.
        """
        from trinity.materials.variants import (
            VariantConfig,
            MaterialDomain,
            BlendMode,
            QualityTier,
        )

        all_variants: Dict[int, str] = {}

        for domain, blend, quality in product(
            MaterialDomain, BlendMode, QualityTier
        ):
            config = VariantConfig(domain=domain, blend=blend, quality=quality)
            wgsl = self.compile_with_variants(material_class, config)
            all_variants[config.get_variant_key()] = wgsl

        return all_variants

    def _walk(self, node: ast.AST) -> str:
        """Walk AST and generate WGSL code.

        Uses PythonToWGSLTranslator to convert Python surface() method
        to WGSL code wrapped in fs_main function.

        Args:
            node: Python AST node (Module containing function definition)

        Returns:
            WGSL code string with fs_main wrapper (or just body if no template)
        """
        from trinity.materials.dsl import PythonToWGSLTranslator

        translator = PythonToWGSLTranslator()
        lines = []

        if self.include_pbr_template:
            # Add fs_main function wrapper with @fragment
            lines.append("@fragment")
            lines.append("fn fs_main(input: FragmentInput) -> SurfaceOutput {")
            lines.append("    var out: SurfaceOutput;")

        # Find the function body and translate it
        for child in ast.walk(node):
            if isinstance(child, ast.FunctionDef) and child.name == 'surface':
                for stmt in child.body:
                    wgsl_line = translator.translate(stmt)
                    if wgsl_line and not wgsl_line.startswith("//"):
                        if self.include_pbr_template:
                            lines.append(f"    {wgsl_line}")
                        else:
                            lines.append(wgsl_line)
                break

        if self.include_pbr_template:
            # Apply blend mode to output
            lines.append("    let final_color = vec4<f32>(out.base_color, 1.0);")
            lines.append("    let blended = apply_blend_mode(final_color);")
            lines.append("    return out;")
            lines.append("}")

        return "\n".join(lines)

    def validate_wgsl(self, wgsl: str) -> Tuple[bool, Optional[str]]:
        """Validate WGSL shader code.

        Performs basic syntax validation of WGSL code. This is a simplified
        validator that checks for common structural issues.

        Args:
            wgsl: WGSL shader code string to validate.

        Returns:
            Tuple of (is_valid: bool, error_message: Optional[str]).
            If valid, returns (True, None).
            If invalid, returns (False, "error description").
        """
        if not wgsl or not wgsl.strip():
            return (False, "Empty WGSL code")

        # Check for balanced braces
        brace_count = 0
        for char in wgsl:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
            if brace_count < 0:
                return (False, "Unbalanced braces: extra '}'")

        if brace_count != 0:
            return (False, f"Unbalanced braces: {brace_count} unclosed '{{' ")

        # Check for balanced parentheses
        paren_count = 0
        for char in wgsl:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            if paren_count < 0:
                return (False, "Unbalanced parentheses: extra ')'")

        if paren_count != 0:
            return (False, f"Unbalanced parentheses: {paren_count} unclosed '('")

        # Check for const declarations without values
        const_pattern = re.compile(r'const\s+\w+\s*:\s*\w+\s*;')
        invalid_consts = const_pattern.findall(wgsl)
        if invalid_consts:
            # Const declarations must have = value
            for const in invalid_consts:
                if '=' not in const:
                    return (False, f"Const declaration missing value: {const}")

        # Check for basic WGSL keywords present
        required_patterns = []

        # If we have a function definition, it should have proper syntax
        fn_pattern = re.compile(r'\bfn\s+\w+\s*\(')
        if fn_pattern.search(wgsl):
            # Has function definition, validate it has a body
            if '{' not in wgsl:
                return (False, "Function definition missing body")

        # Check for invalid WGSL syntax patterns
        invalid_patterns = [
            (r';\s*;', "Double semicolon"),
            (r'\(\s*\)', "Empty parameter list in expression"),  # Removed - valid in WGSL
        ]

        for pattern, error in invalid_patterns:
            if pattern == r'\(\s*\)':
                # Skip this check - empty parens are valid in some contexts
                continue
            if re.search(pattern, wgsl):
                return (False, error)

        return (True, None)
