"""
WGSL code generation for materials (T-DEMO-2.6).

Generates WGSL code for PBR materials:
  - Material struct with albedo, roughness, metallic, emission, ambient_occlusion
  - scene_material(material_id: u32) -> Material lookup function
  - Support for solid colors, procedural patterns, and height-based palettes

The generated code integrates with the SDF scene to provide material
properties for shading based on the material ID returned by scene_sdf().

Usage:
    >>> from engine.rendering.demoscene.ast_nodes import MaterialNode, Vec3Node, FloatNode
    >>> from engine.rendering.demoscene.material_codegen import MaterialCodegen
    >>> mat = MaterialNode(
    ...     material_id=0,
    ...     albedo=Vec3Node(0.8, 0.2, 0.2),
    ...     roughness=FloatNode(0.5),
    ...     metallic=FloatNode(0.0),
    ... )
    >>> gen = MaterialCodegen()
    >>> wgsl = gen.generate([mat])
"""

from __future__ import annotations

from typing import Optional, Sequence

from .ast_nodes import FloatNode, MaterialNode, Vec3Node


# =============================================================================
# MATERIAL STRUCT DEFINITION
# =============================================================================

MATERIAL_STRUCT = """\
/// PBR Material properties for surface shading.
///   albedo           -- base color RGB (diffuse reflection color)
///   roughness        -- surface roughness [0=smooth, 1=rough]
///   metallic         -- metalness factor [0=dielectric, 1=metal]
///   emission         -- emissive color RGB (self-illumination)
///   ambient_occlusion -- AO factor [0=occluded, 1=fully lit]
struct Material {
    albedo: vec3<f32>,
    roughness: f32,
    metallic: f32,
    emission: vec3<f32>,
    ambient_occlusion: f32,
}
"""

DEFAULT_MATERIAL = MaterialNode(
    material_id=0,
    albedo=Vec3Node(0.8, 0.2, 0.2),
    roughness=FloatNode(0.5),
    metallic=FloatNode(0.0),
    emission=Vec3Node(0.0, 0.0, 0.0),
    ambient_occlusion=FloatNode(1.0),
)


# =============================================================================
# PROCEDURAL PATTERN FUNCTIONS
# =============================================================================

PROCEDURAL_PATTERNS = """\
/// Checker pattern: returns 0.0 or 1.0 based on position.
fn pattern_checker(p: vec3<f32>, scale: f32) -> f32 {
    let q = floor(p * scale);
    return f32((i32(q.x) + i32(q.y) + i32(q.z)) & 1);
}

/// Stripe pattern along axis (0=x, 1=y, 2=z).
fn pattern_stripe(p: vec3<f32>, scale: f32, axis: u32) -> f32 {
    var coord: f32;
    switch axis {
        case 0u { coord = p.x; }
        case 1u { coord = p.y; }
        case 2u { coord = p.z; }
        default { coord = p.x; }
    }
    return step(0.5, fract(coord * scale));
}

/// Gradient based on height (y-coordinate).
fn pattern_height_gradient(p: vec3<f32>, min_y: f32, max_y: f32) -> f32 {
    return clamp((p.y - min_y) / (max_y - min_y), 0.0, 1.0);
}

/// Radial gradient from origin.
fn pattern_radial(p: vec3<f32>, max_radius: f32) -> f32 {
    return clamp(length(p) / max_radius, 0.0, 1.0);
}

/// Blend between two materials based on pattern value.
fn material_blend(a: Material, b: Material, t: f32) -> Material {
    return Material(
        mix(a.albedo, b.albedo, t),
        mix(a.roughness, b.roughness, t),
        mix(a.metallic, b.metallic, t),
        mix(a.emission, b.emission, t),
        mix(a.ambient_occlusion, b.ambient_occlusion, t),
    );
}
"""

HEIGHT_PALETTE = """\
/// Height-based color palette lookup.
/// Returns a material with albedo interpolated through a palette based on height.
fn height_palette(p: vec3<f32>, min_y: f32, max_y: f32) -> Material {
    let t = clamp((p.y - min_y) / (max_y - min_y), 0.0, 1.0);

    // 5-color palette: deep -> low -> mid -> high -> peak
    let c0 = vec3<f32>(0.1, 0.2, 0.4);  // Deep blue
    let c1 = vec3<f32>(0.2, 0.5, 0.2);  // Green
    let c2 = vec3<f32>(0.6, 0.5, 0.3);  // Brown
    let c3 = vec3<f32>(0.7, 0.7, 0.7);  // Gray
    let c4 = vec3<f32>(1.0, 1.0, 1.0);  // White

    var albedo: vec3<f32>;
    var roughness: f32;

    if (t < 0.25) {
        let s = t / 0.25;
        albedo = mix(c0, c1, s);
        roughness = mix(0.3, 0.6, s);
    } else if (t < 0.5) {
        let s = (t - 0.25) / 0.25;
        albedo = mix(c1, c2, s);
        roughness = mix(0.6, 0.7, s);
    } else if (t < 0.75) {
        let s = (t - 0.5) / 0.25;
        albedo = mix(c2, c3, s);
        roughness = mix(0.7, 0.5, s);
    } else {
        let s = (t - 0.75) / 0.25;
        albedo = mix(c3, c4, s);
        roughness = mix(0.5, 0.2, s);
    }

    return Material(
        albedo,
        roughness,
        0.0,  // metallic
        vec3<f32>(0.0),  // emission
        1.0,  // ambient_occlusion
    );
}
"""


# =============================================================================
# MATERIAL CODEGEN CLASS
# =============================================================================

class MaterialCodegen:
    """Generates WGSL code for PBR materials.

    This class handles:
      - Material struct definition
      - scene_material() function with switch/case for each material
      - Optional procedural pattern functions
      - Height-based palette support

    Usage::

        gen = MaterialCodegen()
        wgsl = gen.generate(materials, include_patterns=True)
    """

    def __init__(self) -> None:
        self._emitted_functions: set[str] = set()

    def generate(
        self,
        materials: Sequence[MaterialNode],
        *,
        include_struct: bool = True,
        include_patterns: bool = False,
        include_height_palette: bool = False,
        name: str = "",
    ) -> str:
        """Generate WGSL code for materials.

        Args:
            materials: Sequence of MaterialNode objects.
            include_struct: Whether to include the Material struct definition.
            include_patterns: Whether to include procedural pattern functions.
            include_height_palette: Whether to include height-based palette.
            name: Optional name for the generated module.

        Returns:
            WGSL source code as a string.
        """
        self._emitted_functions.clear()
        lines: list[str] = []

        lines.append("// SPDX-License-Identifier: MIT")
        lines.append("//")
        lines.append("// Auto-generated by MaterialCodegen (T-DEMO-2.6).")
        if name:
            lines.append(f"// Module: {name}")
        lines.append("//")
        lines.append("")

        if include_struct:
            lines.append(MATERIAL_STRUCT)

        if include_patterns:
            lines.append(PROCEDURAL_PATTERNS)

        if include_height_palette:
            lines.append(HEIGHT_PALETTE)

        lines.append(self._generate_scene_material(materials))

        return "\n".join(lines)

    def _generate_scene_material(
        self,
        materials: Sequence[MaterialNode],
    ) -> str:
        """Generate the scene_material() function.

        Args:
            materials: Sequence of MaterialNode objects.

        Returns:
            WGSL function definition as a string.
        """
        lines: list[str] = []

        lines.append("/// Returns the material properties for a given material ID.")
        lines.append("///   material_id -- the material identifier from scene_sdf()")
        lines.append("///   returns     -- Material struct with PBR properties")
        lines.append("fn scene_material(material_id: u32) -> Material {")

        if not materials:
            # No materials provided: return default
            lines.append("    // Default fallback material")
            lines.append(self._format_material_return(DEFAULT_MATERIAL, indent=4))
        elif len(materials) == 1:
            # Single material: no switch needed
            mat = materials[0]
            lines.append(f"    // Material {mat.material_id}")
            lines.append(self._format_material_return(mat, indent=4))
        else:
            # Multiple materials: use switch/case
            lines.append("    switch material_id {")
            for mat in materials:
                lines.append(f"        case {mat.material_id}u: {{")
                lines.append(self._format_material_return(mat, indent=12))
                lines.append("        }")
            lines.append("        default: {")
            lines.append(self._format_material_return(DEFAULT_MATERIAL, indent=12))
            lines.append("        }")
            lines.append("    }")

        lines.append("}")

        return "\n".join(lines)

    def _format_material_return(
        self,
        mat: MaterialNode,
        indent: int = 4,
    ) -> str:
        """Format a return statement for a material.

        Args:
            mat: MaterialNode to format.
            indent: Number of spaces to indent.

        Returns:
            Formatted return statement.
        """
        pad = " " * indent
        albedo = _fmt_vec3(mat.albedo)
        emission = _fmt_vec3(mat.emission)
        roughness = _fmt_float(mat.roughness.value)
        metallic = _fmt_float(mat.metallic.value)
        ao = _fmt_float(mat.ambient_occlusion.value)

        return (
            f"{pad}return Material(\n"
            f"{pad}    {albedo},\n"
            f"{pad}    {roughness},\n"
            f"{pad}    {metallic},\n"
            f"{pad}    {emission},\n"
            f"{pad}    {ao},\n"
            f"{pad});"
        )

    def generate_inline_material(
        self,
        mat: MaterialNode,
    ) -> str:
        """Generate an inline Material constructor call.

        Args:
            mat: MaterialNode to format.

        Returns:
            Inline Material(...) expression.
        """
        albedo = _fmt_vec3(mat.albedo)
        emission = _fmt_vec3(mat.emission)
        roughness = _fmt_float(mat.roughness.value)
        metallic = _fmt_float(mat.metallic.value)
        ao = _fmt_float(mat.ambient_occlusion.value)

        return f"Material({albedo}, {roughness}, {metallic}, {emission}, {ao})"


# =============================================================================
# FORMAT HELPERS
# =============================================================================


def _fmt_float(val: float) -> str:
    """Format a float for WGSL output."""
    if val == int(val) and not (val == 0.0 and str(val).startswith("-")):
        return f"{int(val)}.0"
    return f"{val}"


def _fmt_vec3(v: Vec3Node) -> str:
    """Format a Vec3Node for WGSL output."""
    return (
        f"vec3<f32>("
        f"{_fmt_float(v.x)}, {_fmt_float(v.y)}, {_fmt_float(v.z)}"
        f")"
    )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def generate_material_wgsl(
    materials: Sequence[MaterialNode],
    *,
    include_struct: bool = True,
    include_patterns: bool = False,
    include_height_palette: bool = False,
    name: str = "",
) -> str:
    """Generate WGSL code for materials.

    Args:
        materials: Sequence of MaterialNode objects.
        include_struct: Whether to include the Material struct definition.
        include_patterns: Whether to include procedural pattern functions.
        include_height_palette: Whether to include height-based palette.
        name: Optional name for the generated module.

    Returns:
        WGSL source code as a string.

    Example:
        >>> mat = MaterialNode(
        ...     material_id=0,
        ...     albedo=Vec3Node(0.8, 0.2, 0.2),
        ...     roughness=FloatNode(0.5),
        ... )
        >>> wgsl = generate_material_wgsl([mat])
        >>> "struct Material" in wgsl
        True
    """
    gen = MaterialCodegen()
    return gen.generate(
        materials,
        include_struct=include_struct,
        include_patterns=include_patterns,
        include_height_palette=include_height_palette,
        name=name,
    )
