"""
SDF Error Reporting for Invalid Scenes (T-DEMO-2.14).

Provides a comprehensive error reporting system for SDF scene validation:
  - Exception hierarchy for different error categories
  - SDFValidator for scene graph validation
  - Clear, actionable error messages with node paths
  - Integration point for CachedSDFCompiler

Usage:
    >>> from engine.rendering.demoscene.sdf_errors import SDFValidator, SDFValidationError
    >>> validator = SDFValidator()
    >>> errors = validator.validate(scene)
    >>> if errors:
    ...     for error in errors:
    ...         print(f"{error.path}: {error.message}")

    >>> # Or raise on first error
    >>> validator.validate_strict(scene)  # Raises SDFValidationError
"""

from __future__ import annotations

import math
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

from .sdf_ast import (
    Axis,
    BendNode,
    BoxFrameNode,
    BoxNode,
    CameraNode,
    CapsuleNode,
    CombinatorNode,
    ConeNode,
    CylinderNode,
    DisplacedNode,
    DomainOpNode,
    EllipsoidNode,
    IntersectionNode,
    KIFSNode,
    LightNode,
    MaterialNode,
    MirrorNode,
    OctahedronNode,
    PlaneNode,
    PrimitiveNode,
    PyramidNode,
    RenderSettingsNode,
    RepeatNode,
    RoundedBoxNode,
    SceneNode,
    SDFNode,
    SmoothIntersectionNode,
    SmoothSubtractionNode,
    SmoothUnionNode,
    SphereNode,
    StretchNode,
    SubtractionNode,
    TorusNode,
    TwistNode,
    UnionNode,
    Vec3,
)


# =============================================================================
# EXCEPTION HIERARCHY
# =============================================================================

class SDFError(Exception):
    """Base exception for all SDF-related errors.

    Attributes:
        message: Human-readable error description
        path: Path to the node in the scene graph (e.g., "/scene/root/union[0]")
        node: The SDFNode that caused the error (if available)
        suggestion: Optional suggestion for fixing the error
    """

    def __init__(
        self,
        message: str,
        path: str = "",
        node: Optional[SDFNode] = None,
        suggestion: Optional[str] = None,
    ) -> None:
        self.message = message
        self.path = path
        self.node = node
        self.suggestion = suggestion
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the full error message with path and suggestion."""
        parts = []
        if self.path:
            parts.append(f"{self.path}: ")
        parts.append(self.message)
        if self.suggestion:
            parts.append(f" (suggestion: {self.suggestion})")
        return "".join(parts)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.message!r}, path={self.path!r})"


class SDFValidationError(SDFError):
    """Error raised when scene validation fails.

    Covers general validation issues like missing required fields,
    structural problems, or constraint violations.
    """
    pass


class SDFCompilationError(SDFError):
    """Error raised during SDF compilation to WGSL.

    Indicates that the scene graph cannot be compiled to valid shader code,
    typically due to unsupported node types or code generation failures.
    """
    pass


class SDFTypeError(SDFError):
    """Error raised when a parameter has an incorrect type.

    Common cases:
      - String passed where float expected
      - Integer passed where Vec3 expected
      - Wrong node type used as child
    """
    pass


class SDFRecursionError(SDFError):
    """Error raised when infinite recursion is detected in the scene graph.

    This occurs when nodes form a cycle, either directly (node references itself)
    or indirectly (A -> B -> C -> A).
    """

    def __init__(
        self,
        message: str,
        path: str = "",
        node: Optional[SDFNode] = None,
        cycle: Optional[List[int]] = None,
        suggestion: Optional[str] = None,
    ) -> None:
        self.cycle = cycle or []
        super().__init__(message, path, node, suggestion)

    def _format_message(self) -> str:
        base = super()._format_message()
        if self.cycle:
            cycle_str = " -> ".join(str(nid) for nid in self.cycle)
            return f"{base} [cycle: {cycle_str}]"
        return base


class SDFImpossibleSDFError(SDFError):
    """Error raised when SDF parameters create an impossible or degenerate shape.

    Examples:
      - Sphere with negative radius
      - Box with zero or negative dimensions
      - Torus with minor radius >= major radius
      - Cylinder with zero height
    """
    pass


# =============================================================================
# VALIDATION SEVERITY
# =============================================================================

class Severity(Enum):
    """Severity level for validation issues."""
    ERROR = auto()    # Must be fixed, will cause compilation failure
    WARNING = auto()  # Should be fixed, may cause unexpected behavior
    INFO = auto()     # Informational, style or optimization suggestion


# =============================================================================
# VALIDATION ISSUE
# =============================================================================

@dataclass(frozen=True)
class ValidationIssue:
    """Represents a single validation issue found in the scene graph.

    Attributes:
        severity: How serious the issue is
        error: The SDFError instance with details
        node_id: Unique ID of the problematic node
        node_type: Type name of the problematic node
    """
    severity: Severity
    error: SDFError
    node_id: int
    node_type: str

    @property
    def message(self) -> str:
        return self.error.message

    @property
    def path(self) -> str:
        return self.error.path

    @property
    def suggestion(self) -> Optional[str]:
        return self.error.suggestion

    def __str__(self) -> str:
        prefix = f"[{self.severity.name}]"
        return f"{prefix} {self.error._format_message()}"


# =============================================================================
# VALIDATION CONTEXT
# =============================================================================

@dataclass
class ValidationContext:
    """Context passed through validation to track path and visited nodes."""
    path_parts: List[str] = field(default_factory=list)
    visited_ids: Set[int] = field(default_factory=set)
    issues: List[ValidationIssue] = field(default_factory=list)
    parent_node: Optional[SDFNode] = None
    depth: int = 0
    max_depth: int = 1000  # Prevent stack overflow

    def push(self, name: str) -> "ValidationContext":
        """Create a child context with an additional path component."""
        return ValidationContext(
            path_parts=self.path_parts + [name],
            visited_ids=self.visited_ids.copy(),
            issues=self.issues,  # Shared reference
            parent_node=None,
            depth=self.depth + 1,
            max_depth=self.max_depth,
        )

    @property
    def path(self) -> str:
        """Get the current path as a string."""
        if not self.path_parts:
            return "/"
        return "/" + "/".join(self.path_parts)

    def add_error(
        self,
        error_class: Type[SDFError],
        message: str,
        node: SDFNode,
        suggestion: Optional[str] = None,
        severity: Severity = Severity.ERROR,
    ) -> None:
        """Add a validation issue to the context."""
        error = error_class(
            message=message,
            path=self.path,
            node=node,
            suggestion=suggestion,
        )
        issue = ValidationIssue(
            severity=severity,
            error=error,
            node_id=node._node_id,
            node_type=type(node).__name__,
        )
        self.issues.append(issue)


# =============================================================================
# VALIDATION RULES
# =============================================================================

def _validate_positive_float(
    ctx: ValidationContext,
    node: SDFNode,
    field_name: str,
    value: float,
    allow_zero: bool = False,
) -> None:
    """Validate that a float value is positive."""
    if allow_zero and value == 0.0:
        return
    if value <= 0.0:
        ctx.add_error(
            SDFImpossibleSDFError,
            f"{field_name} must be positive, got {value}",
            node,
            f"set {field_name} to a value > 0",
        )


def _validate_non_negative_float(
    ctx: ValidationContext,
    node: SDFNode,
    field_name: str,
    value: float,
) -> None:
    """Validate that a float value is non-negative."""
    if value < 0.0:
        ctx.add_error(
            SDFImpossibleSDFError,
            f"{field_name} must be non-negative, got {value}",
            node,
            f"set {field_name} to a value >= 0",
        )


def _validate_range(
    ctx: ValidationContext,
    node: SDFNode,
    field_name: str,
    value: float,
    min_val: float,
    max_val: float,
) -> None:
    """Validate that a float value is within a range."""
    if value < min_val or value > max_val:
        ctx.add_error(
            SDFValidationError,
            f"{field_name} must be in range [{min_val}, {max_val}], got {value}",
            node,
            f"set {field_name} to a value between {min_val} and {max_val}",
        )


def _validate_vec3_positive(
    ctx: ValidationContext,
    node: SDFNode,
    field_name: str,
    vec: Vec3,
    allow_zero: bool = False,
) -> None:
    """Validate that all Vec3 components are positive."""
    components = [("x", vec.x), ("y", vec.y), ("z", vec.z)]
    for comp_name, comp_value in components:
        if allow_zero and comp_value == 0.0:
            continue
        if comp_value <= 0.0:
            ctx.add_error(
                SDFImpossibleSDFError,
                f"{field_name}.{comp_name} must be positive, got {comp_value}",
                node,
                f"set {field_name}.{comp_name} to a value > 0",
            )


def _validate_vec3_non_zero(
    ctx: ValidationContext,
    node: SDFNode,
    field_name: str,
    vec: Vec3,
) -> None:
    """Validate that a Vec3 is not the zero vector."""
    if vec.x == 0.0 and vec.y == 0.0 and vec.z == 0.0:
        ctx.add_error(
            SDFImpossibleSDFError,
            f"{field_name} must not be zero vector",
            node,
            f"set at least one component of {field_name} to non-zero",
        )


def _validate_type(
    ctx: ValidationContext,
    node: SDFNode,
    field_name: str,
    value: Any,
    expected_type: Union[type, Tuple[type, ...]],
    type_name: str,
) -> bool:
    """Validate that a value has the expected type. Returns True if valid."""
    if not isinstance(value, expected_type):
        ctx.add_error(
            SDFTypeError,
            f"{field_name} must be {type_name}, got {type(value).__name__}",
            node,
            f"provide a {type_name} value for {field_name}",
        )
        return False
    return True


def _validate_color(
    ctx: ValidationContext,
    node: SDFNode,
    field_name: str,
    color: Vec3,
) -> None:
    """Validate that a color has valid component values."""
    components = [("r", color.x), ("g", color.y), ("b", color.z)]
    for comp_name, comp_value in components:
        if comp_value < 0.0:
            ctx.add_error(
                SDFValidationError,
                f"{field_name}.{comp_name} must be non-negative, got {comp_value}",
                node,
                f"set {field_name}.{comp_name} to a value >= 0",
                severity=Severity.WARNING,
            )
        elif comp_value > 1.0:
            ctx.add_error(
                SDFValidationError,
                f"{field_name}.{comp_name} exceeds 1.0 (HDR value: {comp_value})",
                node,
                severity=Severity.INFO,
            )


# =============================================================================
# SDF VALIDATOR
# =============================================================================

class SDFValidator:
    """Validates SDF scene graphs for common issues.

    The validator traverses the scene graph and checks for:
      - Infinite recursion (cycles in the graph)
      - Type errors (wrong parameter types)
      - Impossible SDFs (negative radius, zero dimensions, etc.)
      - Missing required fields
      - Invalid domain operations
      - Material errors

    Usage:
        >>> validator = SDFValidator()
        >>> errors = validator.validate(scene)
        >>> for error in errors:
        ...     print(error)

        >>> # Strict mode raises on first error
        >>> validator.validate_strict(scene)
    """

    def __init__(
        self,
        max_depth: int = 1000,
        strict_types: bool = True,
        warn_on_degenerate: bool = True,
    ) -> None:
        """Initialize the validator.

        Args:
            max_depth: Maximum recursion depth before raising error
            strict_types: Whether to enforce strict type checking
            warn_on_degenerate: Whether to warn about degenerate but valid SDFs
        """
        self.max_depth = max_depth
        self.strict_types = strict_types
        self.warn_on_degenerate = warn_on_degenerate

    def validate(self, scene: SDFNode) -> List[SDFError]:
        """Validate a scene graph and return all errors found.

        Args:
            scene: The root node of the scene graph

        Returns:
            List of SDFError instances for all issues found
        """
        ctx = ValidationContext(max_depth=self.max_depth)
        self._validate_node(scene, ctx)
        return [issue.error for issue in ctx.issues if issue.severity == Severity.ERROR]

    def validate_all(self, scene: SDFNode) -> List[ValidationIssue]:
        """Validate a scene graph and return all issues (including warnings).

        Args:
            scene: The root node of the scene graph

        Returns:
            List of ValidationIssue instances for all issues found
        """
        ctx = ValidationContext(max_depth=self.max_depth)
        self._validate_node(scene, ctx)
        return ctx.issues

    def validate_strict(self, scene: SDFNode) -> None:
        """Validate a scene graph and raise on the first error found.

        Args:
            scene: The root node of the scene graph

        Raises:
            SDFValidationError: If any validation error is found
        """
        errors = self.validate(scene)
        if errors:
            raise errors[0]

    def is_valid(self, scene: SDFNode) -> bool:
        """Check if a scene graph is valid without returning error details.

        Args:
            scene: The root node of the scene graph

        Returns:
            True if the scene is valid, False otherwise
        """
        return len(self.validate(scene)) == 0

    def _validate_node(self, node: SDFNode, ctx: ValidationContext) -> None:
        """Recursively validate a node and its children."""
        # Check recursion depth
        if ctx.depth > ctx.max_depth:
            ctx.add_error(
                SDFRecursionError,
                f"maximum recursion depth ({ctx.max_depth}) exceeded",
                node,
                "reduce scene graph depth or check for cycles",
            )
            return

        # Check for cycles (self-reference)
        node_id = node._node_id
        if node_id in ctx.visited_ids:
            ctx.add_error(
                SDFRecursionError,
                "cycle detected: node references itself or an ancestor",
                node,
                "remove circular reference in scene graph",
            )
            return

        ctx.visited_ids.add(node_id)

        # Dispatch to specific validator
        if isinstance(node, SceneNode):
            self._validate_scene(node, ctx)
        elif isinstance(node, PrimitiveNode):
            self._validate_primitive(node, ctx)
        elif isinstance(node, CombinatorNode):
            self._validate_combinator(node, ctx)
        elif isinstance(node, DomainOpNode):
            self._validate_domain_op(node, ctx)
        elif isinstance(node, DisplacedNode):
            self._validate_displaced(node, ctx)
        elif isinstance(node, CameraNode):
            self._validate_camera(node, ctx)
        elif isinstance(node, LightNode):
            self._validate_light(node, ctx)
        elif isinstance(node, MaterialNode):
            self._validate_material(node, ctx)
        elif isinstance(node, RenderSettingsNode):
            self._validate_render_settings(node, ctx)

        ctx.visited_ids.discard(node_id)

    # -------------------------------------------------------------------------
    # Scene Validation
    # -------------------------------------------------------------------------

    def _validate_scene(self, scene: SceneNode, ctx: ValidationContext) -> None:
        """Validate a SceneNode."""
        scene_ctx = ctx.push(f"scene[{scene.name or 'unnamed'}]")

        # Validate root
        if scene.root is None:
            ctx.add_error(
                SDFValidationError,
                "scene must have a root node",
                scene,
                "provide a root SDF node",
            )
        else:
            root_ctx = scene_ctx.push("root")
            self._validate_node(scene.root, root_ctx)

        # Validate camera
        if scene.camera is not None:
            camera_ctx = scene_ctx.push("camera")
            self._validate_node(scene.camera, camera_ctx)

        # Validate lights
        for i, light in enumerate(scene.lights):
            light_ctx = scene_ctx.push(f"lights[{i}]")
            self._validate_node(light, light_ctx)

        # Validate materials
        for i, material in enumerate(scene.materials):
            material_ctx = scene_ctx.push(f"materials[{i}]")
            self._validate_node(material, material_ctx)

        # Validate render settings
        if scene.render_settings is not None:
            settings_ctx = scene_ctx.push("render_settings")
            self._validate_node(scene.render_settings, settings_ctx)

    # -------------------------------------------------------------------------
    # Primitive Validation
    # -------------------------------------------------------------------------

    def _validate_primitive(self, prim: PrimitiveNode, ctx: ValidationContext) -> None:
        """Validate a primitive node."""
        prim_name = type(prim).__name__
        prim_ctx = ctx.push(prim_name)

        # Validate position type
        if not _validate_type(prim_ctx, prim, "position", prim.position, Vec3, "Vec3"):
            return

        # Dispatch to specific primitive validator
        if isinstance(prim, SphereNode):
            self._validate_sphere(prim, prim_ctx)
        elif isinstance(prim, BoxNode):
            self._validate_box(prim, prim_ctx)
        elif isinstance(prim, TorusNode):
            self._validate_torus(prim, prim_ctx)
        elif isinstance(prim, CylinderNode):
            self._validate_cylinder(prim, prim_ctx)
        elif isinstance(prim, ConeNode):
            self._validate_cone(prim, prim_ctx)
        elif isinstance(prim, PlaneNode):
            self._validate_plane(prim, prim_ctx)
        elif isinstance(prim, CapsuleNode):
            self._validate_capsule(prim, prim_ctx)
        elif isinstance(prim, EllipsoidNode):
            self._validate_ellipsoid(prim, prim_ctx)
        elif isinstance(prim, BoxFrameNode):
            self._validate_box_frame(prim, prim_ctx)
        elif isinstance(prim, RoundedBoxNode):
            self._validate_rounded_box(prim, prim_ctx)
        elif isinstance(prim, OctahedronNode):
            self._validate_octahedron(prim, prim_ctx)
        elif isinstance(prim, PyramidNode):
            self._validate_pyramid(prim, prim_ctx)

    def _validate_sphere(self, sphere: SphereNode, ctx: ValidationContext) -> None:
        """Validate a SphereNode."""
        if not _validate_type(ctx, sphere, "radius", sphere.radius, (int, float), "float"):
            return
        _validate_positive_float(ctx, sphere, "radius", sphere.radius)

    def _validate_box(self, box: BoxNode, ctx: ValidationContext) -> None:
        """Validate a BoxNode."""
        if not _validate_type(ctx, box, "half_extents", box.half_extents, Vec3, "Vec3"):
            return
        _validate_vec3_positive(ctx, box, "half_extents", box.half_extents)

    def _validate_torus(self, torus: TorusNode, ctx: ValidationContext) -> None:
        """Validate a TorusNode."""
        if not _validate_type(ctx, torus, "major_radius", torus.major_radius, (int, float), "float"):
            return
        if not _validate_type(ctx, torus, "minor_radius", torus.minor_radius, (int, float), "float"):
            return

        _validate_positive_float(ctx, torus, "major_radius", torus.major_radius)
        _validate_positive_float(ctx, torus, "minor_radius", torus.minor_radius)

        # Minor radius should be less than major radius
        if torus.minor_radius >= torus.major_radius:
            ctx.add_error(
                SDFImpossibleSDFError,
                f"minor_radius ({torus.minor_radius}) must be less than major_radius ({torus.major_radius})",
                torus,
                "reduce minor_radius or increase major_radius",
                severity=Severity.WARNING,
            )

    def _validate_cylinder(self, cylinder: CylinderNode, ctx: ValidationContext) -> None:
        """Validate a CylinderNode."""
        if not _validate_type(ctx, cylinder, "radius", cylinder.radius, (int, float), "float"):
            return
        if not _validate_type(ctx, cylinder, "height", cylinder.height, (int, float), "float"):
            return

        _validate_positive_float(ctx, cylinder, "radius", cylinder.radius)
        _validate_positive_float(ctx, cylinder, "height", cylinder.height)

    def _validate_cone(self, cone: ConeNode, ctx: ValidationContext) -> None:
        """Validate a ConeNode."""
        if not _validate_type(ctx, cone, "angle", cone.angle, (int, float), "float"):
            return
        if not _validate_type(ctx, cone, "height", cone.height, (int, float), "float"):
            return

        _validate_positive_float(ctx, cone, "height", cone.height)

        # Angle should be in valid range (0, pi/2)
        if cone.angle <= 0.0 or cone.angle >= math.pi / 2:
            ctx.add_error(
                SDFImpossibleSDFError,
                f"angle must be in range (0, pi/2), got {cone.angle}",
                cone,
                "set angle to a value between 0 and 1.57 (exclusive)",
            )

    def _validate_plane(self, plane: PlaneNode, ctx: ValidationContext) -> None:
        """Validate a PlaneNode."""
        if not _validate_type(ctx, plane, "normal", plane.normal, Vec3, "Vec3"):
            return
        if not _validate_type(ctx, plane, "distance", plane.distance, (int, float), "float"):
            return

        _validate_vec3_non_zero(ctx, plane, "normal", plane.normal)

    def _validate_capsule(self, capsule: CapsuleNode, ctx: ValidationContext) -> None:
        """Validate a CapsuleNode."""
        if not _validate_type(ctx, capsule, "endpoint_a", capsule.endpoint_a, Vec3, "Vec3"):
            return
        if not _validate_type(ctx, capsule, "endpoint_b", capsule.endpoint_b, Vec3, "Vec3"):
            return
        if not _validate_type(ctx, capsule, "radius", capsule.radius, (int, float), "float"):
            return

        _validate_positive_float(ctx, capsule, "radius", capsule.radius)

        # Endpoints should not be identical
        diff = capsule.endpoint_b - capsule.endpoint_a
        if diff.length() < 1e-10:
            ctx.add_error(
                SDFImpossibleSDFError,
                "endpoint_a and endpoint_b must be different points",
                capsule,
                "set endpoint_a and endpoint_b to different positions",
                severity=Severity.WARNING,
            )

    def _validate_ellipsoid(self, ellipsoid: EllipsoidNode, ctx: ValidationContext) -> None:
        """Validate an EllipsoidNode."""
        if not _validate_type(ctx, ellipsoid, "radii", ellipsoid.radii, Vec3, "Vec3"):
            return
        _validate_vec3_positive(ctx, ellipsoid, "radii", ellipsoid.radii)

    def _validate_box_frame(self, box_frame: BoxFrameNode, ctx: ValidationContext) -> None:
        """Validate a BoxFrameNode."""
        if not _validate_type(ctx, box_frame, "half_extents", box_frame.half_extents, Vec3, "Vec3"):
            return
        if not _validate_type(ctx, box_frame, "edge_thickness", box_frame.edge_thickness, (int, float), "float"):
            return

        _validate_vec3_positive(ctx, box_frame, "half_extents", box_frame.half_extents)
        _validate_positive_float(ctx, box_frame, "edge_thickness", box_frame.edge_thickness)

    def _validate_rounded_box(self, rounded_box: RoundedBoxNode, ctx: ValidationContext) -> None:
        """Validate a RoundedBoxNode."""
        if not _validate_type(ctx, rounded_box, "half_extents", rounded_box.half_extents, Vec3, "Vec3"):
            return
        if not _validate_type(ctx, rounded_box, "corner_radius", rounded_box.corner_radius, (int, float), "float"):
            return

        _validate_vec3_positive(ctx, rounded_box, "half_extents", rounded_box.half_extents)
        _validate_non_negative_float(ctx, rounded_box, "corner_radius", rounded_box.corner_radius)

        # Corner radius should not exceed smallest half extent
        min_extent = min(
            rounded_box.half_extents.x,
            rounded_box.half_extents.y,
            rounded_box.half_extents.z,
        )
        if rounded_box.corner_radius > min_extent:
            ctx.add_error(
                SDFImpossibleSDFError,
                f"corner_radius ({rounded_box.corner_radius}) exceeds smallest half_extent ({min_extent})",
                rounded_box,
                f"reduce corner_radius to at most {min_extent}",
                severity=Severity.WARNING,
            )

    def _validate_octahedron(self, octahedron: OctahedronNode, ctx: ValidationContext) -> None:
        """Validate an OctahedronNode."""
        if not _validate_type(ctx, octahedron, "size", octahedron.size, (int, float), "float"):
            return
        _validate_positive_float(ctx, octahedron, "size", octahedron.size)

    def _validate_pyramid(self, pyramid: PyramidNode, ctx: ValidationContext) -> None:
        """Validate a PyramidNode."""
        if not _validate_type(ctx, pyramid, "height", pyramid.height, (int, float), "float"):
            return
        _validate_positive_float(ctx, pyramid, "height", pyramid.height)

    # -------------------------------------------------------------------------
    # Combinator Validation
    # -------------------------------------------------------------------------

    def _validate_combinator(self, comb: CombinatorNode, ctx: ValidationContext) -> None:
        """Validate a combinator node."""
        comb_name = type(comb).__name__
        comb_ctx = ctx.push(comb_name)

        # Validate children exist
        if comb.left is None:
            comb_ctx.add_error(
                SDFValidationError,
                "left operand is required",
                comb,
                "provide a left SDF node",
            )
        else:
            left_ctx = comb_ctx.push("left")
            self._validate_node(comb.left, left_ctx)

        if comb.right is None:
            comb_ctx.add_error(
                SDFValidationError,
                "right operand is required",
                comb,
                "provide a right SDF node",
            )
        else:
            right_ctx = comb_ctx.push("right")
            self._validate_node(comb.right, right_ctx)

        # Validate smooth parameters
        if isinstance(comb, (SmoothUnionNode, SmoothIntersectionNode, SmoothSubtractionNode)):
            if not _validate_type(comb_ctx, comb, "k", comb.k, (int, float), "float"):
                return
            if comb.k < 0.0:
                comb_ctx.add_error(
                    SDFValidationError,
                    f"smoothness parameter k must be non-negative, got {comb.k}",
                    comb,
                    "set k to a value >= 0",
                )
            elif comb.k == 0.0:
                comb_ctx.add_error(
                    SDFValidationError,
                    "smoothness parameter k=0 is equivalent to non-smooth operation",
                    comb,
                    "use the non-smooth variant or set k > 0",
                    severity=Severity.INFO,
                )

    # -------------------------------------------------------------------------
    # Domain Operation Validation
    # -------------------------------------------------------------------------

    def _validate_domain_op(self, op: DomainOpNode, ctx: ValidationContext) -> None:
        """Validate a domain operation node."""
        op_name = type(op).__name__
        op_ctx = ctx.push(op_name)

        # Validate child
        if op.child is None:
            op_ctx.add_error(
                SDFValidationError,
                "child operand is required",
                op,
                "provide a child SDF node",
            )
        else:
            child_ctx = op_ctx.push("child")
            self._validate_node(op.child, child_ctx)

        # Dispatch to specific domain op validator
        if isinstance(op, RepeatNode):
            self._validate_repeat(op, op_ctx)
        elif isinstance(op, MirrorNode):
            self._validate_mirror(op, op_ctx)
        elif isinstance(op, KIFSNode):
            self._validate_kifs(op, op_ctx)
        elif isinstance(op, TwistNode):
            self._validate_twist(op, op_ctx)
        elif isinstance(op, BendNode):
            self._validate_bend(op, op_ctx)
        elif isinstance(op, StretchNode):
            self._validate_stretch(op, op_ctx)

    def _validate_repeat(self, repeat: RepeatNode, ctx: ValidationContext) -> None:
        """Validate a RepeatNode."""
        if not _validate_type(ctx, repeat, "cell_size", repeat.cell_size, Vec3, "Vec3"):
            return

        # Check for zero cell size
        if repeat.cell_size.x == 0.0 or repeat.cell_size.y == 0.0 or repeat.cell_size.z == 0.0:
            zero_axes = []
            if repeat.cell_size.x == 0.0:
                zero_axes.append("x")
            if repeat.cell_size.y == 0.0:
                zero_axes.append("y")
            if repeat.cell_size.z == 0.0:
                zero_axes.append("z")
            ctx.add_error(
                SDFImpossibleSDFError,
                f"cell_size must be non-zero on all axes, got zero on: {', '.join(zero_axes)}",
                repeat,
                "set all cell_size components to non-zero values",
            )

        # Check for negative cell size
        if repeat.cell_size.x < 0.0 or repeat.cell_size.y < 0.0 or repeat.cell_size.z < 0.0:
            ctx.add_error(
                SDFValidationError,
                "cell_size components should be positive",
                repeat,
                "use absolute values for cell_size",
                severity=Severity.WARNING,
            )

    def _validate_mirror(self, mirror: MirrorNode, ctx: ValidationContext) -> None:
        """Validate a MirrorNode."""
        if not _validate_type(ctx, mirror, "axis", mirror.axis, Axis, "Axis"):
            return

    def _validate_kifs(self, kifs: KIFSNode, ctx: ValidationContext) -> None:
        """Validate a KIFSNode."""
        if not _validate_type(ctx, kifs, "iterations", kifs.iterations, int, "int"):
            return
        if not _validate_type(ctx, kifs, "scale", kifs.scale, (int, float), "float"):
            return
        if not _validate_type(ctx, kifs, "offset", kifs.offset, Vec3, "Vec3"):
            return

        if kifs.iterations < 1:
            ctx.add_error(
                SDFValidationError,
                f"iterations must be at least 1, got {kifs.iterations}",
                kifs,
                "set iterations to a value >= 1",
            )

        if kifs.scale <= 0.0:
            ctx.add_error(
                SDFImpossibleSDFError,
                f"scale must be positive, got {kifs.scale}",
                kifs,
                "set scale to a value > 0",
            )
        elif kifs.scale == 1.0:
            ctx.add_error(
                SDFValidationError,
                "scale=1.0 creates no scaling effect",
                kifs,
                "use a different scale value for visible effect",
                severity=Severity.INFO,
            )

    def _validate_twist(self, twist: TwistNode, ctx: ValidationContext) -> None:
        """Validate a TwistNode."""
        if not _validate_type(ctx, twist, "axis", twist.axis, Axis, "Axis"):
            return
        if not _validate_type(ctx, twist, "rate", twist.rate, (int, float), "float"):
            return

        if twist.rate == 0.0:
            ctx.add_error(
                SDFValidationError,
                "twist rate=0 has no effect",
                twist,
                "remove the twist node or use a non-zero rate",
                severity=Severity.INFO,
            )

    def _validate_bend(self, bend: BendNode, ctx: ValidationContext) -> None:
        """Validate a BendNode."""
        if not _validate_type(ctx, bend, "axis", bend.axis, Axis, "Axis"):
            return
        if not _validate_type(ctx, bend, "radius", bend.radius, (int, float), "float"):
            return

        if bend.radius == 0.0:
            ctx.add_error(
                SDFImpossibleSDFError,
                "bend radius must be non-zero",
                bend,
                "set radius to a non-zero value",
            )

    def _validate_stretch(self, stretch: StretchNode, ctx: ValidationContext) -> None:
        """Validate a StretchNode."""
        if not _validate_type(ctx, stretch, "axis", stretch.axis, Axis, "Axis"):
            return
        if not _validate_type(ctx, stretch, "scale", stretch.scale, (int, float), "float"):
            return

        if stretch.scale <= 0.0:
            ctx.add_error(
                SDFImpossibleSDFError,
                f"scale must be positive, got {stretch.scale}",
                stretch,
                "set scale to a value > 0",
            )
        elif stretch.scale == 1.0:
            ctx.add_error(
                SDFValidationError,
                "scale=1.0 has no stretching effect",
                stretch,
                "remove the stretch node or use a different scale",
                severity=Severity.INFO,
            )

    # -------------------------------------------------------------------------
    # Displaced Node Validation
    # -------------------------------------------------------------------------

    def _validate_displaced(self, displaced: DisplacedNode, ctx: ValidationContext) -> None:
        """Validate a DisplacedNode."""
        disp_ctx = ctx.push("DisplacedNode")

        # Validate child
        if displaced.child is None:
            disp_ctx.add_error(
                SDFValidationError,
                "child operand is required",
                displaced,
                "provide a child SDF node",
            )
        else:
            child_ctx = disp_ctx.push("child")
            self._validate_node(displaced.child, child_ctx)

        # Validate amplitude
        if not _validate_type(disp_ctx, displaced, "amplitude", displaced.amplitude, (int, float), "float"):
            return

        # Validate frequency
        if not _validate_type(disp_ctx, displaced, "frequency", displaced.frequency, (int, float), "float"):
            return

        if displaced.frequency <= 0.0:
            disp_ctx.add_error(
                SDFValidationError,
                f"frequency must be positive, got {displaced.frequency}",
                displaced,
                "set frequency to a value > 0",
            )

        if displaced.amplitude == 0.0:
            disp_ctx.add_error(
                SDFValidationError,
                "amplitude=0 has no displacement effect",
                displaced,
                "remove the displaced node or use non-zero amplitude",
                severity=Severity.INFO,
            )

    # -------------------------------------------------------------------------
    # Camera Validation
    # -------------------------------------------------------------------------

    def _validate_camera(self, camera: CameraNode, ctx: ValidationContext) -> None:
        """Validate a CameraNode."""
        # Validate FOV
        if not _validate_type(ctx, camera, "fov", camera.fov, (int, float), "float"):
            return

        if camera.fov <= 0.0 or camera.fov >= 180.0:
            ctx.add_error(
                SDFValidationError,
                f"fov must be in range (0, 180), got {camera.fov}",
                camera,
                "set fov to a value between 0 and 180 degrees (exclusive)",
            )

        # Validate aspect ratio
        if not _validate_type(ctx, camera, "aspect_ratio", camera.aspect_ratio, (int, float), "float"):
            return

        if camera.aspect_ratio <= 0.0:
            ctx.add_error(
                SDFValidationError,
                f"aspect_ratio must be positive, got {camera.aspect_ratio}",
                camera,
                "set aspect_ratio to a value > 0",
            )

        # Validate vectors
        if not _validate_type(ctx, camera, "origin", camera.origin, Vec3, "Vec3"):
            return
        if not _validate_type(ctx, camera, "look_at", camera.look_at, Vec3, "Vec3"):
            return
        if not _validate_type(ctx, camera, "up", camera.up, Vec3, "Vec3"):
            return

        # Up vector should not be zero
        _validate_vec3_non_zero(ctx, camera, "up", camera.up)

        # Origin and look_at should be different
        diff = camera.look_at - camera.origin
        if diff.length() < 1e-10:
            ctx.add_error(
                SDFValidationError,
                "origin and look_at must be different positions",
                camera,
                "set origin and look_at to different positions",
            )

        # Aperture validation
        if not _validate_type(ctx, camera, "aperture", camera.aperture, (int, float), "float"):
            return
        _validate_non_negative_float(ctx, camera, "aperture", camera.aperture)

        # Focal distance validation
        if not _validate_type(ctx, camera, "focal_distance", camera.focal_distance, (int, float), "float"):
            return
        _validate_positive_float(ctx, camera, "focal_distance", camera.focal_distance)

    # -------------------------------------------------------------------------
    # Light Validation
    # -------------------------------------------------------------------------

    def _validate_light(self, light: LightNode, ctx: ValidationContext) -> None:
        """Validate a LightNode."""
        # Validate position
        if not _validate_type(ctx, light, "position", light.position, Vec3, "Vec3"):
            return

        # Validate color
        if not _validate_type(ctx, light, "color", light.color, Vec3, "Vec3"):
            return
        _validate_color(ctx, light, "color", light.color)

        # Validate intensity
        if not _validate_type(ctx, light, "intensity", light.intensity, (int, float), "float"):
            return
        _validate_non_negative_float(ctx, light, "intensity", light.intensity)

    # -------------------------------------------------------------------------
    # Material Validation
    # -------------------------------------------------------------------------

    def _validate_material(self, material: MaterialNode, ctx: ValidationContext) -> None:
        """Validate a MaterialNode."""
        # Validate color
        if not _validate_type(ctx, material, "color", material.color, Vec3, "Vec3"):
            return
        _validate_color(ctx, material, "color", material.color)

        # Validate metallic
        if not _validate_type(ctx, material, "metallic", material.metallic, (int, float), "float"):
            return
        _validate_range(ctx, material, "metallic", material.metallic, 0.0, 1.0)

        # Validate roughness
        if not _validate_type(ctx, material, "roughness", material.roughness, (int, float), "float"):
            return
        _validate_range(ctx, material, "roughness", material.roughness, 0.0, 1.0)

        # Validate emission
        if not _validate_type(ctx, material, "emission", material.emission, Vec3, "Vec3"):
            return
        # Emission components should be non-negative
        for comp_name, comp_value in [("x", material.emission.x), ("y", material.emission.y), ("z", material.emission.z)]:
            if comp_value < 0.0:
                ctx.add_error(
                    SDFValidationError,
                    f"emission.{comp_name} must be non-negative, got {comp_value}",
                    material,
                    f"set emission.{comp_name} to a value >= 0",
                )

        # Validate material_id
        if not _validate_type(ctx, material, "material_id", material.material_id, int, "int"):
            return
        if material.material_id < 0:
            ctx.add_error(
                SDFValidationError,
                f"material_id must be non-negative, got {material.material_id}",
                material,
                "set material_id to a value >= 0",
            )

    # -------------------------------------------------------------------------
    # Render Settings Validation
    # -------------------------------------------------------------------------

    def _validate_render_settings(self, settings: RenderSettingsNode, ctx: ValidationContext) -> None:
        """Validate a RenderSettingsNode."""
        # Validate dimensions
        if not _validate_type(ctx, settings, "width", settings.width, int, "int"):
            return
        if not _validate_type(ctx, settings, "height", settings.height, int, "int"):
            return

        if settings.width <= 0:
            ctx.add_error(
                SDFValidationError,
                f"width must be positive, got {settings.width}",
                settings,
                "set width to a value > 0",
            )

        if settings.height <= 0:
            ctx.add_error(
                SDFValidationError,
                f"height must be positive, got {settings.height}",
                settings,
                "set height to a value > 0",
            )

        # Validate max_steps
        if not _validate_type(ctx, settings, "max_steps", settings.max_steps, int, "int"):
            return
        if settings.max_steps <= 0:
            ctx.add_error(
                SDFValidationError,
                f"max_steps must be positive, got {settings.max_steps}",
                settings,
                "set max_steps to a value > 0",
            )

        # Validate max_distance
        if not _validate_type(ctx, settings, "max_distance", settings.max_distance, (int, float), "float"):
            return
        _validate_positive_float(ctx, settings, "max_distance", settings.max_distance)

        # Validate epsilon
        if not _validate_type(ctx, settings, "epsilon", settings.epsilon, (int, float), "float"):
            return
        _validate_positive_float(ctx, settings, "epsilon", settings.epsilon)

        if settings.epsilon > 0.1:
            ctx.add_error(
                SDFValidationError,
                f"epsilon ({settings.epsilon}) is unusually large, may cause artifacts",
                settings,
                "typical epsilon values are 0.0001 to 0.01",
                severity=Severity.WARNING,
            )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def validate_scene(scene: SDFNode) -> List[SDFError]:
    """Validate a scene graph and return all errors.

    This is a convenience function that creates a default SDFValidator
    and validates the scene.

    Args:
        scene: The root node of the scene graph

    Returns:
        List of SDFError instances for all issues found

    Example:
        >>> from engine.rendering.demoscene.sdf_errors import validate_scene
        >>> errors = validate_scene(my_scene)
        >>> if errors:
        ...     for error in errors:
        ...         print(f"Error: {error}")
    """
    validator = SDFValidator()
    return validator.validate(scene)


def validate_scene_strict(scene: SDFNode) -> None:
    """Validate a scene graph and raise on the first error.

    Args:
        scene: The root node of the scene graph

    Raises:
        SDFValidationError: If any validation error is found

    Example:
        >>> from engine.rendering.demoscene.sdf_errors import validate_scene_strict
        >>> validate_scene_strict(my_scene)  # Raises if invalid
    """
    validator = SDFValidator()
    validator.validate_strict(scene)


def is_scene_valid(scene: SDFNode) -> bool:
    """Check if a scene graph is valid.

    Args:
        scene: The root node of the scene graph

    Returns:
        True if the scene is valid, False otherwise

    Example:
        >>> from engine.rendering.demoscene.sdf_errors import is_scene_valid
        >>> if is_scene_valid(my_scene):
        ...     print("Scene is valid!")
    """
    validator = SDFValidator()
    return validator.is_valid(scene)


def get_validation_report(scene: SDFNode) -> str:
    """Generate a human-readable validation report for a scene.

    Args:
        scene: The root node of the scene graph

    Returns:
        A formatted string containing all validation issues

    Example:
        >>> from engine.rendering.demoscene.sdf_errors import get_validation_report
        >>> print(get_validation_report(my_scene))
    """
    validator = SDFValidator()
    issues = validator.validate_all(scene)

    if not issues:
        return "Scene validation passed: no issues found."

    lines = [f"Scene validation found {len(issues)} issue(s):"]
    lines.append("")

    # Group by severity
    errors = [i for i in issues if i.severity == Severity.ERROR]
    warnings = [i for i in issues if i.severity == Severity.WARNING]
    infos = [i for i in issues if i.severity == Severity.INFO]

    if errors:
        lines.append(f"ERRORS ({len(errors)}):")
        for issue in errors:
            lines.append(f"  - {issue}")
        lines.append("")

    if warnings:
        lines.append(f"WARNINGS ({len(warnings)}):")
        for issue in warnings:
            lines.append(f"  - {issue}")
        lines.append("")

    if infos:
        lines.append(f"INFO ({len(infos)}):")
        for issue in infos:
            lines.append(f"  - {issue}")

    return "\n".join(lines)


# =============================================================================
# COMPILER INTEGRATION
# =============================================================================

class ValidatingCompiler:
    """Wrapper that adds validation to any SDF compiler.

    Usage:
        >>> from engine.rendering.demoscene.sdf_errors import ValidatingCompiler
        >>> base_compiler = MySDFCompiler()
        >>> compiler = ValidatingCompiler(base_compiler)
        >>> wgsl = compiler.compile(scene)  # Validates first
    """

    def __init__(
        self,
        compiler: Any,
        validate_before_compile: bool = True,
        strict: bool = True,
    ) -> None:
        """Initialize the validating compiler wrapper.

        Args:
            compiler: The underlying compiler to wrap
            validate_before_compile: Whether to validate before compiling
            strict: Whether to raise on first error (True) or collect all errors (False)
        """
        self._compiler = compiler
        self._validate_before_compile = validate_before_compile
        self._strict = strict
        self._validator = SDFValidator()
        self._last_errors: List[SDFError] = []

    @property
    def validate_before_compile(self) -> bool:
        """Whether validation is enabled before compilation."""
        return self._validate_before_compile

    @validate_before_compile.setter
    def validate_before_compile(self, value: bool) -> None:
        self._validate_before_compile = value

    @property
    def last_errors(self) -> List[SDFError]:
        """Errors from the last compilation attempt."""
        return self._last_errors

    def compile(self, scene: SDFNode, *args: Any, **kwargs: Any) -> Any:
        """Compile a scene, optionally validating first.

        Args:
            scene: The scene to compile
            *args: Additional arguments for the underlying compiler
            **kwargs: Additional keyword arguments for the underlying compiler

        Returns:
            The compilation result from the underlying compiler

        Raises:
            SDFValidationError: If validation is enabled and errors are found
        """
        self._last_errors = []

        if self._validate_before_compile:
            errors = self._validator.validate(scene)
            self._last_errors = errors

            if errors:
                if self._strict:
                    raise errors[0]
                else:
                    # Collect all errors into a single exception
                    messages = [str(e) for e in errors]
                    combined_message = f"Scene validation failed with {len(errors)} error(s):\n" + "\n".join(messages)
                    raise SDFValidationError(combined_message)

        return self._compiler.compile(scene, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying compiler."""
        return getattr(self._compiler, name)
