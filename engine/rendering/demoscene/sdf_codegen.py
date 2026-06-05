"""
WGSL Code Generator for SDF Primitives and Domain Operations (T-DEMO-2.3, T-DEMO-2.5).

This module implements a visitor-pattern WGSL code generator that produces GPU shader
code for signed distance field (SDF) primitives and domain transformation operations.

Features:
- T-DEMO-2.3: WGSL code generation for all 12 primitive types
- T-DEMO-2.5: WGSL code generation for 6 domain operations
- Visitor pattern for extensible node handling
- Proper transformation wrapping (translate, rotate, scale)
- Correct domain operation chaining

Primitives (T-DEMO-2.3):
    Sphere, Box, Torus, Cylinder, Cone, Plane, Capsule, Ellipsoid,
    BoxFrame, RoundedBox, Octahedron, Pyramid

Domain Operations (T-DEMO-2.5):
    Repeat, Mirror, KIFS, Twist, Bend, Stretch

Reference: Inigo Quilez -- Distance Functions
    https://iquilezles.org/articles/distfunctions/

Usage:
    >>> from engine.rendering.demoscene.sdf_codegen import WGSLCodegen
    >>> from engine.rendering.demoscene.ast_nodes import SphereNode, PositionNode, FloatNode
    >>> codegen = WGSLCodegen()
    >>> node = SphereNode(PositionNode(), FloatNode(1.0))
    >>> wgsl = codegen.generate_primitive(node)
    >>> print(wgsl)
    sdf_sphere(p - center, 1.0)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union, TYPE_CHECKING

from .ast_nodes import (
    Axis,
    BendNode,
    BoxFrameNode,
    BoxNode,
    CapsuleNode,
    ConeNode,
    CylinderNode,
    DomainOpNode,
    EllipsoidNode,
    ExprNode,
    FloatNode,
    KifsNode,
    MirrorNode,
    OctahedronNode,
    PlaneNode,
    PositionNode,
    PyramidNode,
    RepeatNode,
    RoundedBoxNode,
    SceneGraph,
    SdfPrimitiveNode,
    SphereNode,
    StretchNode,
    TorusNode,
    TwistNode,
    Vec3Node,
)

# =============================================================================
# WGSL PRIMITIVE FUNCTION TEMPLATES
# =============================================================================

SDF_SPHERE_WGSL = """\
fn sdf_sphere(p: vec3<f32>, r: f32) -> f32 {
    return length(p) - r;
}"""

SDF_BOX_WGSL = """\
fn sdf_box(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}"""

SDF_TORUS_WGSL = """\
fn sdf_torus(p: vec3<f32>, r: vec2<f32>) -> f32 {
    let q = vec2<f32>(length(p.xz) - r.x, p.y);
    return length(q) - r.y;
}"""

SDF_CYLINDER_WGSL = """\
fn sdf_cylinder(p: vec3<f32>, h: vec2<f32>) -> f32 {
    let d = abs(vec2<f32>(length(p.xz), p.y)) - h;
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
}"""

SDF_CONE_WGSL = """\
fn sdf_cone(p: vec3<f32>, c: vec2<f32>, h: f32) -> f32 {
    let q = h * vec2<f32>(c.x / c.y, -1.0);
    let w = vec2<f32>(length(p.xz), p.y);
    let a = w - q * clamp(dot(w, q) / dot(q, q), 0.0, 1.0);
    let b = w - q * vec2<f32>(clamp(w.x / q.x, 0.0, 1.0), 1.0);
    let k = sign(q.y);
    let d = min(dot(a, a), dot(b, b));
    let s = max(k * (w.x * q.y - w.y * q.x), k * (w.y - h));
    return sqrt(d) * sign(s);
}"""

SDF_PLANE_WGSL = """\
fn sdf_plane(p: vec3<f32>, n: vec4<f32>) -> f32 {
    return dot(p, n.xyz) + n.w;
}"""

SDF_CAPSULE_WGSL = """\
fn sdf_capsule(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}"""

SDF_ELLIPSOID_WGSL = """\
fn sdf_ellipsoid(p: vec3<f32>, r: vec3<f32>) -> f32 {
    let k0 = length(p / r);
    let k1 = length(p / (r * r));
    return k0 * (k0 - 1.0) / k1;
}"""

SDF_BOX_FRAME_WGSL = """\
fn sdf_box_frame(p: vec3<f32>, b: vec3<f32>, e: f32) -> f32 {
    let q = abs(p) - b;
    let w = abs(q + e) - e;
    return min(min(
        length(max(vec3<f32>(q.x, w.y, w.z), vec3<f32>(0.0))) + min(max(q.x, max(w.y, w.z)), 0.0),
        length(max(vec3<f32>(w.x, q.y, w.z), vec3<f32>(0.0))) + min(max(w.x, max(q.y, w.z)), 0.0)
    ),
        length(max(vec3<f32>(w.x, w.y, q.z), vec3<f32>(0.0))) + min(max(w.x, max(w.y, q.z)), 0.0)
    );
}"""

SDF_ROUNDED_BOX_WGSL = """\
fn sdf_rounded_box(p: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}"""

SDF_OCTAHEDRON_WGSL = """\
fn sdf_octahedron(p: vec3<f32>, s: f32) -> f32 {
    let q = abs(p);
    let m = q.x + q.y + q.z - s;
    var k: vec3<f32>;
    if (3.0 * q.x < m) {
        k = q.xyz;
    } else if (3.0 * q.y < m) {
        k = q.yzx;
    } else if (3.0 * q.z < m) {
        k = q.zxy;
    } else {
        return m * 0.57735027;
    }
    let o = clamp(0.5 * (k.z - k.y + s), 0.0, s);
    return length(vec3<f32>(k.x, k.y - s + o, k.z - o));
}"""

SDF_PYRAMID_WGSL = """\
fn sdf_pyramid(p: vec3<f32>, h: f32) -> f32 {
    let m2 = h * h + 0.25;
    var q = vec3<f32>(abs(p.x), p.y, abs(p.z));
    if (q.z > q.x) {
        q = vec3<f32>(q.z, q.y, q.x);
    }
    q = vec3<f32>(q.x - 0.5, q.y, q.z - 0.5);
    let a = vec3<f32>(q.z, h * q.y - 0.5 * q.x, h * q.x + 0.5 * q.y);
    let s = max(-a.x, 0.0);
    let t = clamp((a.y - 0.5 * a.z) / (m2 + 0.25), 0.0, 1.0);
    let k1 = vec2<f32>(s, h * s - q.y);
    let k2 = vec2<f32>(t * 0.5, h * t) - vec2<f32>(q.x, q.y);
    let d1 = dot(k1, k1);
    let d2 = dot(k2, k2);
    let d = sqrt(min(d1, d2));
    let inside = max(a.y, -q.y - h);
    return select(d, -d, inside < 0.0);
}"""

# =============================================================================
# DOMAIN OPERATION WGSL TEMPLATES
# =============================================================================

DOMAIN_REPEAT_WGSL = """\
fn domain_repeat(p: vec3<f32>, cell_size: vec3<f32>) -> vec3<f32> {
    let c = max(cell_size, vec3<f32>(1e-8));
    return p - c * round(p / c);
}"""

DOMAIN_MIRROR_X_WGSL = """\
fn domain_mirror_x(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(abs(p.x), p.y, p.z);
}"""

DOMAIN_MIRROR_Y_WGSL = """\
fn domain_mirror_y(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(p.x, abs(p.y), p.z);
}"""

DOMAIN_MIRROR_Z_WGSL = """\
fn domain_mirror_z(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(p.x, p.y, abs(p.z));
}"""

DOMAIN_KIFS_WGSL = """\
fn domain_fold_kifs(p: vec3<f32>, iterations: i32) -> vec4<f32> {
    var q = p;
    var scale = 1.0;
    let fold_scale = 2.0;
    let offset = vec3<f32>(1.0, 1.0, 1.0);

    for (var i = 0; i < iterations; i = i + 1) {
        q = abs(q);
        q = q * fold_scale - offset * (fold_scale - 1.0);
        scale = scale * fold_scale;
    }

    return vec4<f32>(q, scale);
}"""

DOMAIN_TWIST_X_WGSL = """\
fn domain_twist_x(p: vec3<f32>, amount: f32) -> vec3<f32> {
    let angle = amount * p.x;
    let c = cos(angle);
    let s = sin(angle);
    return vec3<f32>(p.x, c * p.y - s * p.z, s * p.y + c * p.z);
}"""

DOMAIN_TWIST_Y_WGSL = """\
fn domain_twist_y(p: vec3<f32>, amount: f32) -> vec3<f32> {
    let angle = amount * p.y;
    let c = cos(angle);
    let s = sin(angle);
    return vec3<f32>(c * p.x - s * p.z, p.y, s * p.x + c * p.z);
}"""

DOMAIN_TWIST_Z_WGSL = """\
fn domain_twist_z(p: vec3<f32>, amount: f32) -> vec3<f32> {
    let angle = amount * p.z;
    let c = cos(angle);
    let s = sin(angle);
    return vec3<f32>(c * p.x - s * p.y, s * p.x + c * p.y, p.z);
}"""

DOMAIN_BEND_X_WGSL = """\
fn domain_bend_x(p: vec3<f32>, radius: f32) -> vec3<f32> {
    let r = max(abs(radius), 1e-8);
    if (abs(radius) < 1e-8) { return p; }
    let theta = p.x / r;
    let c = cos(theta);
    let s = sin(theta);
    return vec3<f32>((r + p.y) * s, (r + p.y) * c - r, p.z);
}"""

DOMAIN_BEND_Y_WGSL = """\
fn domain_bend_y(p: vec3<f32>, radius: f32) -> vec3<f32> {
    let r = max(abs(radius), 1e-8);
    if (abs(radius) < 1e-8) { return p; }
    let theta = p.y / r;
    let c = cos(theta);
    let s = sin(theta);
    return vec3<f32>((r + p.x) * s, (r + p.x) * c - r, p.z);
}"""

DOMAIN_BEND_Z_WGSL = """\
fn domain_bend_z(p: vec3<f32>, radius: f32) -> vec3<f32> {
    let r = max(abs(radius), 1e-8);
    if (abs(radius) < 1e-8) { return p; }
    let theta = p.z / r;
    let c = cos(theta);
    let s = sin(theta);
    return vec3<f32>((r + p.x) * c - r, p.y, (r + p.x) * s);
}"""

DOMAIN_STRETCH_X_WGSL = """\
fn domain_stretch_x(p: vec3<f32>, scale: f32) -> vec3<f32> {
    let s = max(abs(scale), 1e-8);
    return vec3<f32>(p.x * s, p.y, p.z);
}"""

DOMAIN_STRETCH_Y_WGSL = """\
fn domain_stretch_y(p: vec3<f32>, scale: f32) -> vec3<f32> {
    let s = max(abs(scale), 1e-8);
    return vec3<f32>(p.x, p.y * s, p.z);
}"""

DOMAIN_STRETCH_Z_WGSL = """\
fn domain_stretch_z(p: vec3<f32>, scale: f32) -> vec3<f32> {
    let s = max(abs(scale), 1e-8);
    return vec3<f32>(p.x, p.y, p.z * s);
}"""


# =============================================================================
# PRIMITIVE FUNCTION REGISTRY
# =============================================================================

PRIMITIVE_WGSL_FUNCTIONS = {
    SphereNode: ("sdf_sphere", SDF_SPHERE_WGSL),
    BoxNode: ("sdf_box", SDF_BOX_WGSL),
    TorusNode: ("sdf_torus", SDF_TORUS_WGSL),
    CylinderNode: ("sdf_cylinder", SDF_CYLINDER_WGSL),
    ConeNode: ("sdf_cone", SDF_CONE_WGSL),
    PlaneNode: ("sdf_plane", SDF_PLANE_WGSL),
    CapsuleNode: ("sdf_capsule", SDF_CAPSULE_WGSL),
    EllipsoidNode: ("sdf_ellipsoid", SDF_ELLIPSOID_WGSL),
    BoxFrameNode: ("sdf_box_frame", SDF_BOX_FRAME_WGSL),
    RoundedBoxNode: ("sdf_rounded_box", SDF_ROUNDED_BOX_WGSL),
    OctahedronNode: ("sdf_octahedron", SDF_OCTAHEDRON_WGSL),
    PyramidNode: ("sdf_pyramid", SDF_PYRAMID_WGSL),
}


# =============================================================================
# TRANSFORMATION CONTEXT
# =============================================================================

@dataclass
class TransformContext:
    """Tracks accumulated transformations for a primitive."""
    translate: Optional[Vec3Node] = None
    rotate: Optional[Vec3Node] = None  # Euler angles (radians)
    scale: Optional[Vec3Node] = None


# =============================================================================
# WGSL CODE GENERATOR (VISITOR PATTERN)
# =============================================================================

class WGSLCodegen:
    """
    WGSL code generator using the visitor pattern.

    Generates WGSL shader code for SDF primitives and domain operations
    by dispatching to type-specific visitor methods.

    Attributes:
        _emitted_functions: Set of already emitted function names to avoid duplication.
        _function_definitions: List of emitted function definition strings.
        _indent: Current indentation level for code formatting.

    Example:
        >>> codegen = WGSLCodegen()
        >>> sphere = SphereNode(PositionNode(), FloatNode(1.5))
        >>> call = codegen.generate_primitive(sphere)
        >>> print(call)
        sdf_sphere(p, 1.5)
    """

    def __init__(self) -> None:
        self._emitted_functions: set[str] = set()
        self._function_definitions: list[str] = []
        self._indent: int = 0

    def reset(self) -> None:
        """Reset internal state for a fresh generation pass."""
        self._emitted_functions.clear()
        self._function_definitions.clear()
        self._indent = 0

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    def generate_primitive(
        self,
        node: SdfPrimitiveNode,
        pos_var: str = "p",
        center_var: Optional[str] = None,
    ) -> str:
        """
        Generate a WGSL function call for an SDF primitive node.

        Args:
            node: The SDF primitive AST node.
            pos_var: Name of the position variable (default "p").
            center_var: Optional center offset variable (e.g., "center").
                        If provided, generates `sdf_*(p - center, ...)`.

        Returns:
            WGSL function call string, e.g. "sdf_sphere(p - center, 1.5)".

        Raises:
            ValueError: If node type is not a recognized primitive.
        """
        visitor_method = f"_visit_{type(node).__name__}"
        if hasattr(self, visitor_method):
            return getattr(self, visitor_method)(node, pos_var, center_var)
        raise ValueError(f"Unsupported primitive node type: {type(node).__name__}")

    def generate_domain_op(
        self,
        node: DomainOpNode,
        pos_var: str = "p",
    ) -> str:
        """
        Generate a WGSL function call for a domain operation node.

        Args:
            node: The domain operation AST node.
            pos_var: Name of the position variable to transform.

        Returns:
            WGSL function call string wrapping the child SDF, e.g.
            "domain_repeat(p, vec3<f32>(2.0, 2.0, 2.0))".

        Raises:
            ValueError: If node type is not a recognized domain operation.
        """
        visitor_method = f"_visit_{type(node).__name__}"
        if hasattr(self, visitor_method):
            return getattr(self, visitor_method)(node, pos_var)
        raise ValueError(f"Unsupported domain operation type: {type(node).__name__}")

    def get_function_definition(self, node_type: type) -> str:
        """
        Get the WGSL function definition for a given node type.

        Args:
            node_type: The AST node class (e.g., SphereNode).

        Returns:
            The complete WGSL function definition string.

        Raises:
            ValueError: If node type has no registered function definition.
        """
        if node_type in PRIMITIVE_WGSL_FUNCTIONS:
            return PRIMITIVE_WGSL_FUNCTIONS[node_type][1]
        raise ValueError(f"No WGSL function definition for: {node_type.__name__}")

    def emit_function(self, node_type: type) -> bool:
        """
        Emit the function definition for a node type if not already emitted.

        Args:
            node_type: The AST node class.

        Returns:
            True if the function was emitted, False if already present.
        """
        if node_type not in PRIMITIVE_WGSL_FUNCTIONS:
            return False

        fn_name, fn_def = PRIMITIVE_WGSL_FUNCTIONS[node_type]
        if fn_name in self._emitted_functions:
            return False

        self._emitted_functions.add(fn_name)
        self._function_definitions.append(fn_def)
        return True

    def get_emitted_functions(self) -> str:
        """Get all emitted function definitions as a single string."""
        return "\n\n".join(self._function_definitions)

    # -------------------------------------------------------------------------
    # PRIMITIVE VISITORS (T-DEMO-2.3)
    # -------------------------------------------------------------------------

    def _visit_SphereNode(
        self,
        node: SphereNode,
        pos_var: str,
        center_var: Optional[str],
    ) -> str:
        """Generate: sdf_sphere(p - center, radius)"""
        self.emit_function(SphereNode)
        pos = self._offset_position(pos_var, center_var)
        radius = _fmt_float(node.radius.value)
        return f"sdf_sphere({pos}, {radius})"

    def _visit_BoxNode(
        self,
        node: BoxNode,
        pos_var: str,
        center_var: Optional[str],
    ) -> str:
        """Generate: sdf_box(p - center, half_extents)"""
        self.emit_function(BoxNode)
        pos = self._offset_position(pos_var, center_var)
        size = _fmt_vec3(node.size)
        return f"sdf_box({pos}, {size})"

    def _visit_TorusNode(
        self,
        node: TorusNode,
        pos_var: str,
        center_var: Optional[str],
    ) -> str:
        """Generate: sdf_torus(p - center, vec2(major_r, minor_r))"""
        self.emit_function(TorusNode)
        pos = self._offset_position(pos_var, center_var)
        major = _fmt_float(node.major_radius.value)
        minor = _fmt_float(node.minor_radius.value)
        return f"sdf_torus({pos}, vec2<f32>({major}, {minor}))"

    def _visit_CylinderNode(
        self,
        node: CylinderNode,
        pos_var: str,
        center_var: Optional[str],
    ) -> str:
        """Generate: sdf_cylinder(p - center, half_height)"""
        self.emit_function(CylinderNode)
        pos = self._offset_position(pos_var, center_var)
        radius = _fmt_float(node.radius.value)
        half_height = _fmt_float(node.height.value / 2.0)
        return f"sdf_cylinder({pos}, vec2<f32>({radius}, {half_height}))"

    def _visit_ConeNode(
        self,
        node: ConeNode,
        pos_var: str,
        center_var: Optional[str],
    ) -> str:
        """Generate: sdf_cone(p - center, slope)"""
        self.emit_function(ConeNode)
        pos = self._offset_position(pos_var, center_var)
        height = _fmt_float(node.height.value)
        # Compute sin/cos of half-angle from radii
        r_top = node.radius_top.value
        r_bottom = node.radius_bottom.value
        # Approximate slope angle using the larger radius
        import math
        r_max = max(abs(r_top), abs(r_bottom), 0.001)
        h = max(abs(node.height.value), 0.001)
        angle = math.atan2(r_max, h)
        sin_a = _fmt_float(math.sin(angle))
        cos_a = _fmt_float(math.cos(angle))
        return f"sdf_cone({pos}, vec2<f32>({sin_a}, {cos_a}), {height})"

    def _visit_PlaneNode(
        self,
        node: PlaneNode,
        pos_var: str,
        center_var: Optional[str],
    ) -> str:
        """Generate: sdf_plane(p, normal)"""
        self.emit_function(PlaneNode)
        pos = self._offset_position(pos_var, center_var)
        nx = _fmt_float(node.normal.x)
        ny = _fmt_float(node.normal.y)
        nz = _fmt_float(node.normal.z)
        d = _fmt_float(node.distance.value)
        return f"sdf_plane({pos}, vec4<f32>({nx}, {ny}, {nz}, {d}))"

    def _visit_CapsuleNode(
        self,
        node: CapsuleNode,
        pos_var: str,
        center_var: Optional[str],
    ) -> str:
        """Generate: sdf_capsule(p, a, b, radius)"""
        self.emit_function(CapsuleNode)
        pos = self._offset_position(pos_var, center_var)
        a = _fmt_vec3(node.endpoint_a)
        b = _fmt_vec3(node.endpoint_b)
        radius = _fmt_float(node.radius.value)
        return f"sdf_capsule({pos}, {a}, {b}, {radius})"

    def _visit_EllipsoidNode(
        self,
        node: EllipsoidNode,
        pos_var: str,
        center_var: Optional[str],
    ) -> str:
        """Generate: sdf_ellipsoid(p - center, radii)"""
        self.emit_function(EllipsoidNode)
        pos = self._offset_position(pos_var, center_var)
        radii = _fmt_vec3(node.radii)
        return f"sdf_ellipsoid({pos}, {radii})"

    def _visit_BoxFrameNode(
        self,
        node: BoxFrameNode,
        pos_var: str,
        center_var: Optional[str],
    ) -> str:
        """Generate: sdf_box_frame(p - center, size, edge_thickness)"""
        self.emit_function(BoxFrameNode)
        pos = self._offset_position(pos_var, center_var)
        size = _fmt_vec3(node.size)
        thickness = _fmt_float(node.edge_thickness.value)
        return f"sdf_box_frame({pos}, {size}, {thickness})"

    def _visit_RoundedBoxNode(
        self,
        node: RoundedBoxNode,
        pos_var: str,
        center_var: Optional[str],
    ) -> str:
        """Generate: sdf_rounded_box(p - center, size, corner_radius)"""
        self.emit_function(RoundedBoxNode)
        pos = self._offset_position(pos_var, center_var)
        size = _fmt_vec3(node.size)
        corner_radius = _fmt_float(node.corner_radius.value)
        return f"sdf_rounded_box({pos}, {size}, {corner_radius})"

    def _visit_OctahedronNode(
        self,
        node: OctahedronNode,
        pos_var: str,
        center_var: Optional[str],
    ) -> str:
        """Generate: sdf_octahedron(p - center, size)"""
        self.emit_function(OctahedronNode)
        pos = self._offset_position(pos_var, center_var)
        size = _fmt_float(node.size.value)
        return f"sdf_octahedron({pos}, {size})"

    def _visit_PyramidNode(
        self,
        node: PyramidNode,
        pos_var: str,
        center_var: Optional[str],
    ) -> str:
        """Generate: sdf_pyramid(p - center, height)"""
        self.emit_function(PyramidNode)
        pos = self._offset_position(pos_var, center_var)
        height = _fmt_float(node.height.value)
        return f"sdf_pyramid({pos}, {height})"

    # -------------------------------------------------------------------------
    # DOMAIN OPERATION VISITORS (T-DEMO-2.5)
    # -------------------------------------------------------------------------

    def _visit_RepeatNode(
        self,
        node: RepeatNode,
        pos_var: str,
    ) -> str:
        """Generate: domain_repeat(p, cell_size)"""
        cell_size = _fmt_vec3(node.cell_size)
        return f"domain_repeat({pos_var}, {cell_size})"

    def _visit_MirrorNode(
        self,
        node: MirrorNode,
        pos_var: str,
    ) -> str:
        """Generate: domain_mirror_{axis}(p)"""
        axis = node.axis.value.lower()
        return f"domain_mirror_{axis}({pos_var})"

    def _visit_KifsNode(
        self,
        node: KifsNode,
        pos_var: str,
    ) -> str:
        """Generate: domain_fold_kifs(p, iterations)"""
        iterations = int(node.folds.value)
        return f"domain_fold_kifs({pos_var}, {iterations})"

    def _visit_TwistNode(
        self,
        node: TwistNode,
        pos_var: str,
    ) -> str:
        """Generate: domain_twist_y(p, amount) - defaults to Y axis."""
        amount = _fmt_float(node.rate.value)
        # Default twist around Y axis (most common for vertical structures)
        return f"domain_twist_y({pos_var}, {amount})"

    def _visit_BendNode(
        self,
        node: BendNode,
        pos_var: str,
    ) -> str:
        """Generate: domain_bend_z(p, radius) - defaults to Z axis."""
        radius = _fmt_float(node.radius.value)
        # Default bend along Z axis
        return f"domain_bend_z({pos_var}, {radius})"

    def _visit_StretchNode(
        self,
        node: StretchNode,
        pos_var: str,
    ) -> str:
        """Generate: domain_stretch_{axis}(p, scale)"""
        axis = node.axis.value.lower()
        scale = _fmt_float(node.stretch.value)
        return f"domain_stretch_{axis}({pos_var}, {scale})"

    # -------------------------------------------------------------------------
    # HELPER METHODS
    # -------------------------------------------------------------------------

    def _offset_position(
        self,
        pos_var: str,
        center_var: Optional[str],
    ) -> str:
        """Generate position expression with optional center offset."""
        if center_var:
            return f"{pos_var} - {center_var}"
        return pos_var

    def generate_domain_chain(
        self,
        pipeline: tuple[DomainOpNode, ...],
        input_var: str = "p",
        output_var: str = "p_transformed",
    ) -> str:
        """
        Generate WGSL code for a chain of domain operations.

        Args:
            pipeline: Tuple of domain operation nodes to chain.
            input_var: Name of the input position variable.
            output_var: Name for the final transformed position.

        Returns:
            WGSL code block with nested domain transformations.

        Example:
            >>> codegen = WGSLCodegen()
            >>> pipeline = (
            ...     RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),
            ...     MirrorNode(PositionNode(), Axis.X),
            ... )
            >>> print(codegen.generate_domain_chain(pipeline))
            let p_transformed = domain_mirror_x(domain_repeat(p, vec3<f32>(2.0, 2.0, 2.0)));
        """
        if not pipeline:
            return f"let {output_var} = {input_var};"

        # Build nested expression from innermost to outermost
        expr = input_var
        for op in reversed(pipeline):
            expr = self.generate_domain_op(op, expr)

        return f"let {output_var} = {expr};"

    def generate_primitive_with_transforms(
        self,
        node: SdfPrimitiveNode,
        context: TransformContext,
        pos_var: str = "p",
    ) -> str:
        """
        Generate primitive call with transformations applied.

        Args:
            node: The SDF primitive node.
            context: Transformation context (translate, rotate, scale).
            pos_var: Position variable name.

        Returns:
            WGSL expression with transformations wrapped around the primitive.
        """
        transformed_pos = pos_var

        # Apply transformations in order: scale, rotate, translate
        if context.scale is not None:
            # Inverse scale for SDF (scale up position = scale down SDF)
            sx = _fmt_float(1.0 / context.scale.x if context.scale.x != 0 else 1.0)
            sy = _fmt_float(1.0 / context.scale.y if context.scale.y != 0 else 1.0)
            sz = _fmt_float(1.0 / context.scale.z if context.scale.z != 0 else 1.0)
            transformed_pos = f"({transformed_pos} * vec3<f32>({sx}, {sy}, {sz}))"

        if context.rotate is not None:
            # TODO: Full rotation matrix application
            # For now, just document that rotation is complex
            pass

        center_var = None
        if context.translate is not None:
            # Generate center subtraction
            center_var = f"vec3<f32>({_fmt_float(context.translate.x)}, {_fmt_float(context.translate.y)}, {_fmt_float(context.translate.z)})"

        return self.generate_primitive(node, transformed_pos, center_var)


# =============================================================================
# DOMAIN OPERATION FUNCTION REGISTRY
# =============================================================================

DOMAIN_OP_WGSL_FUNCTIONS = {
    "repeat": DOMAIN_REPEAT_WGSL,
    "mirror_x": DOMAIN_MIRROR_X_WGSL,
    "mirror_y": DOMAIN_MIRROR_Y_WGSL,
    "mirror_z": DOMAIN_MIRROR_Z_WGSL,
    "kifs": DOMAIN_KIFS_WGSL,
    "twist_x": DOMAIN_TWIST_X_WGSL,
    "twist_y": DOMAIN_TWIST_Y_WGSL,
    "twist_z": DOMAIN_TWIST_Z_WGSL,
    "bend_x": DOMAIN_BEND_X_WGSL,
    "bend_y": DOMAIN_BEND_Y_WGSL,
    "bend_z": DOMAIN_BEND_Z_WGSL,
    "stretch_x": DOMAIN_STRETCH_X_WGSL,
    "stretch_y": DOMAIN_STRETCH_Y_WGSL,
    "stretch_z": DOMAIN_STRETCH_Z_WGSL,
}


def get_all_primitive_wgsl() -> str:
    """Get all SDF primitive function definitions as a single WGSL string."""
    functions = [fn_def for _, fn_def in PRIMITIVE_WGSL_FUNCTIONS.values()]
    return "\n\n".join(functions)


def get_all_domain_op_wgsl() -> str:
    """Get all domain operation function definitions as a single WGSL string."""
    return "\n\n".join(DOMAIN_OP_WGSL_FUNCTIONS.values())


# =============================================================================
# FORMAT HELPERS
# =============================================================================

def _fmt_float(val: float) -> str:
    """Format a float value for WGSL output."""
    if val == int(val) and not (val == 0.0 and str(val).startswith("-")):
        return f"{int(val)}.0"
    return f"{val}"


def _fmt_vec3(v: Vec3Node) -> str:
    """Format a Vec3Node as a WGSL vec3 constructor."""
    return f"vec3<f32>({_fmt_float(v.x)}, {_fmt_float(v.y)}, {_fmt_float(v.z)})"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_primitive_wgsl(node: SdfPrimitiveNode, pos_var: str = "p") -> str:
    """
    Generate WGSL for a single primitive node.

    Args:
        node: The SDF primitive AST node.
        pos_var: Position variable name.

    Returns:
        WGSL function call string.

    Example:
        >>> sphere = SphereNode(PositionNode(), FloatNode(1.5))
        >>> print(generate_primitive_wgsl(sphere))
        sdf_sphere(p, 1.5)
    """
    codegen = WGSLCodegen()
    return codegen.generate_primitive(node, pos_var)


def generate_domain_op_wgsl(node: DomainOpNode, pos_var: str = "p") -> str:
    """
    Generate WGSL for a single domain operation node.

    Args:
        node: The domain operation AST node.
        pos_var: Position variable name.

    Returns:
        WGSL function call string.

    Example:
        >>> repeat = RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0))
        >>> print(generate_domain_op_wgsl(repeat))
        domain_repeat(p, vec3<f32>(2.0, 2.0, 2.0))
    """
    codegen = WGSLCodegen()
    return codegen.generate_domain_op(node, pos_var)


def generate_scene_sdf(graph: SceneGraph, name: str = "scene") -> str:
    """
    Generate a complete WGSL scene SDF function.

    Args:
        graph: The scene graph AST.
        name: Name for the generated function.

    Returns:
        Complete WGSL source code with function definitions and scene entry point.
    """
    codegen = WGSLCodegen()
    lines: list[str] = []

    # Header
    lines.append("// SPDX-License-Identifier: MIT")
    lines.append("// Auto-generated by WGSLCodegen (T-DEMO-2.3, T-DEMO-2.5)")
    lines.append(f"// Scene: {name}")
    lines.append("")

    # Emit primitive function definitions
    for prim in graph.primitives:
        codegen.emit_function(type(prim))

    lines.append(codegen.get_emitted_functions())
    lines.append("")

    # Generate scene function
    fn_name = f"sd_scene_{name.replace(' ', '_').replace('-', '_')}"
    lines.append(f"fn {fn_name}(p: vec3<f32>) -> f32 {{")

    # Apply domain operations
    if graph.pipeline:
        domain_chain = codegen.generate_domain_chain(graph.pipeline, "p", "p_d")
        lines.append(f"    {domain_chain}")
        pos_var = "p_d"
    else:
        pos_var = "p"

    # Evaluate primitives
    if len(graph.primitives) == 1:
        call = codegen.generate_primitive(graph.primitives[0], pos_var)
        lines.append(f"    return {call};")
    else:
        for i, prim in enumerate(graph.primitives):
            call = codegen.generate_primitive(prim, pos_var)
            lines.append(f"    let d{i} = {call};")

        # Union of all primitives
        parts = [f"d{i}" for i in range(len(graph.primitives))]
        if len(parts) == 2:
            lines.append(f"    return min({parts[0]}, {parts[1]});")
        else:
            expr = parts[0]
            for p in parts[1:]:
                expr = f"min({expr}, {p})"
            lines.append(f"    return {expr};")

    lines.append("}")

    return "\n".join(lines)
