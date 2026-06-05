"""
TRINITY SDF DSL AST Builder (T-DEMO-2.1 and T-DEMO-2.2)

This module provides a Python DSL for Signed Distance Field scene authoring.
Each AST node follows the Trinity metaclass pattern with:
- Mirror: Introspection for field access and type information
- Tracker: Dirty tracking for cache invalidation

The AST faithfully represents the scene graph structure and can be:
- Traversed for code generation (WGSL)
- Serialized for caching
- Validated for correctness
- Introspected for debugging

Reference:
- Rust primitives: crates/renderer-backend/src/sdf_primitives.rs
- Rust combinators: crates/renderer-backend/src/sdf_combinators.rs
- Rust domain ops: crates/renderer-backend/src/sdf_domain_ops.rs
"""

from __future__ import annotations

import threading
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    FrozenSet,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)


# =============================================================================
# Constants
# =============================================================================

__all__ = [
    # Base
    "SDFNode",
    "SDFNodeMeta",
    # Primitives (T-DEMO-2.1)
    "PrimitiveNode",
    "SphereNode",
    "BoxNode",
    "TorusNode",
    "CylinderNode",
    "ConeNode",
    "PlaneNode",
    "CapsuleNode",
    "EllipsoidNode",
    "BoxFrameNode",
    "RoundedBoxNode",
    "OctahedronNode",
    "PyramidNode",
    # Combinators
    "CombinatorNode",
    "UnionNode",
    "IntersectionNode",
    "SubtractionNode",
    "SmoothUnionNode",
    "SmoothIntersectionNode",
    "SmoothSubtractionNode",
    "DisplacedNode",
    # Domain Operations
    "DomainOpNode",
    "RepeatNode",
    "MirrorNode",
    "KIFSNode",
    "TwistNode",
    "BendNode",
    "StretchNode",
    # Scene
    "MaterialNode",
    "SceneNode",
    "CameraNode",
    "LightNode",
    "RenderSettingsNode",
    # Helpers
    "Vec3",
    "Axis",
    "build_ast",
    # Tracking
    "Mirror",
    "Tracker",
]


# =============================================================================
# Axis Enumeration
# =============================================================================

class Axis(Enum):
    """Axis for domain operations."""
    X = "x"
    Y = "y"
    Z = "z"

    def to_index(self) -> int:
        """Convert axis to array index."""
        return {"x": 0, "y": 1, "z": 2}[self.value]

    def to_wgsl(self) -> str:
        """Convert to WGSL component accessor."""
        return self.value


# =============================================================================
# Vec3 Helper (Immutable 3D Vector)
# =============================================================================

@dataclass(frozen=True, slots=True)
class Vec3:
    """Immutable 3D vector for SDF parameters."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @classmethod
    def from_tuple(cls, t: Tuple[float, float, float]) -> "Vec3":
        """Create Vec3 from tuple."""
        return cls(float(t[0]), float(t[1]), float(t[2]))

    @classmethod
    def from_scalar(cls, s: float) -> "Vec3":
        """Create Vec3 with all components equal."""
        return cls(s, s, s)

    @classmethod
    def unit_x(cls) -> "Vec3":
        return cls(1.0, 0.0, 0.0)

    @classmethod
    def unit_y(cls) -> "Vec3":
        return cls(0.0, 1.0, 0.0)

    @classmethod
    def unit_z(cls) -> "Vec3":
        return cls(0.0, 0.0, 1.0)

    def as_tuple(self) -> Tuple[float, float, float]:
        """Return as tuple."""
        return (self.x, self.y, self.z)

    def length(self) -> float:
        """Compute vector length."""
        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5

    def normalized(self) -> "Vec3":
        """Return normalized vector."""
        length = self.length()
        if length < 1e-10:
            return Vec3(0.0, 0.0, 0.0)
        return Vec3(self.x / length, self.y / length, self.z / length)

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __neg__(self) -> "Vec3":
        return Vec3(-self.x, -self.y, -self.z)

    def to_wgsl(self) -> str:
        """Generate WGSL vec3 literal."""
        return f"vec3<f32>({self.x}, {self.y}, {self.z})"


# =============================================================================
# Mirror - Introspection System (T-DEMO-2.2)
# =============================================================================

class Mirror:
    """
    Mirror provides introspection for SDF nodes.

    Following the Trinity pattern, Mirror allows:
    - Field enumeration
    - Type inspection
    - Value access
    - Metadata retrieval
    """

    __slots__ = ("_node",)

    def __init__(self, node: "SDFNode") -> None:
        self._node = node

    @property
    def node_type(self) -> str:
        """Return the node type name."""
        return type(self._node).__name__

    @property
    def node_id(self) -> int:
        """Return the unique node ID."""
        return self._node._node_id

    @property
    def fields(self) -> Dict[str, Any]:
        """Return all field names and values."""
        result = {}
        for name in self._node._field_names:
            result[name] = getattr(self._node, name, None)
        return result

    @property
    def field_types(self) -> Dict[str, type]:
        """Return field names and their types."""
        return dict(self._node._field_types)

    @property
    def children(self) -> Tuple["SDFNode", ...]:
        """Return child nodes."""
        return self._node.children()

    @property
    def is_dirty(self) -> bool:
        """Check if node has been modified."""
        return bool(self._node._dirty_fields)

    @property
    def dirty_fields(self) -> FrozenSet[str]:
        """Return set of modified field names."""
        return frozenset(self._node._dirty_fields)

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return node metadata."""
        return {
            "node_type": self.node_type,
            "node_id": self.node_id,
            "is_dirty": self.is_dirty,
            "child_count": len(self.children),
            "field_count": len(self._node._field_names),
        }

    def get_field(self, name: str) -> Any:
        """Get field value by name."""
        if name not in self._node._field_names:
            raise AttributeError(f"Unknown field: {name}")
        return getattr(self._node, name)

    def get_field_type(self, name: str) -> type:
        """Get field type by name."""
        if name not in self._node._field_types:
            raise AttributeError(f"Unknown field: {name}")
        return self._node._field_types[name]

    def walk(self) -> Generator[Tuple["SDFNode", int], None, None]:
        """Walk the node tree depth-first, yielding (node, depth) pairs."""
        yield from self._node.walk()

    def __repr__(self) -> str:
        return f"<Mirror for {self.node_type}#{self.node_id}>"


# =============================================================================
# Tracker - Dirty Tracking System (T-DEMO-2.2)
# =============================================================================

class Tracker:
    """
    Tracker provides dirty tracking for SDF nodes.

    Following the Trinity pattern, Tracker enables:
    - Change detection
    - Cache invalidation
    - Versioning
    """

    __slots__ = ("_node",)

    def __init__(self, node: "SDFNode") -> None:
        self._node = node

    @property
    def is_dirty(self) -> bool:
        """Check if node or any child is dirty."""
        if self._node._dirty_fields:
            return True
        for child in self._node.children():
            if child.tracker.is_dirty:
                return True
        return False

    @property
    def dirty_fields(self) -> FrozenSet[str]:
        """Return set of dirty field names on this node."""
        return frozenset(self._node._dirty_fields)

    @property
    def version(self) -> int:
        """Return node version (increments on each change)."""
        return self._node._version

    def mark_dirty(self, field_name: str) -> None:
        """Mark a field as dirty."""
        self._node._dirty_fields.add(field_name)
        self._node._version += 1

    def mark_all_dirty(self) -> None:
        """Mark all fields as dirty."""
        self._node._dirty_fields.update(self._node._field_names)
        self._node._version += 1

    def clear(self) -> None:
        """Clear dirty flags on this node."""
        self._node._dirty_fields.clear()

    def clear_recursive(self) -> None:
        """Clear dirty flags on this node and all children."""
        self.clear()
        for child in self._node.children():
            child.tracker.clear_recursive()

    def get_dirty_tree(self) -> List[Tuple["SDFNode", FrozenSet[str]]]:
        """Return list of (node, dirty_fields) for all dirty nodes in tree."""
        result = []
        if self._node._dirty_fields:
            result.append((self._node, frozenset(self._node._dirty_fields)))
        for child in self._node.children():
            result.extend(child.tracker.get_dirty_tree())
        return result

    def __repr__(self) -> str:
        dirty_count = len(self._node._dirty_fields)
        return f"<Tracker dirty={dirty_count} version={self.version}>"


# =============================================================================
# SDFNodeMeta - Metaclass for Trinity Pattern (T-DEMO-2.2)
# =============================================================================

class SDFNodeMeta(type):
    """
    Metaclass for SDF nodes implementing Trinity patterns.

    Provides:
    - Unique node ID generation
    - Field registration
    - Type registration
    - Dirty tracking setup
    """

    _registry: ClassVar[Dict[int, Type["SDFNode"]]] = {}
    _name_to_id: ClassVar[Dict[str, int]] = {}
    _next_id: ClassVar[int] = 1
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(
        mcs,
        name: str,
        bases: Tuple[type, ...],
        namespace: Dict[str, Any],
        **kwargs: Any,
    ) -> "SDFNodeMeta":
        """Create a new SDF node type."""
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Skip base classes
        if name in ("SDFNode", "PrimitiveNode", "CombinatorNode", "DomainOpNode"):
            cls._node_type_id = 0
            cls._field_names = frozenset()
            cls._field_types = {}
            return cls

        with mcs._lock:
            # Generate unique type ID
            qualified_name = f"{cls.__module__}.{name}"
            if qualified_name in mcs._name_to_id:
                # Idempotent registration
                return mcs._registry[mcs._name_to_id[qualified_name]]

            cls._node_type_id = mcs._next_id
            mcs._next_id += 1

            # Extract field information from __init__ annotations
            field_names = set()
            field_types = {}

            # Collect from annotations
            annotations = getattr(cls, "__annotations__", {})
            for field_name, field_type in annotations.items():
                if not field_name.startswith("_"):
                    field_names.add(field_name)
                    field_types[field_name] = field_type

            cls._field_names = frozenset(field_names)
            cls._field_types = field_types

            # Register
            mcs._registry[cls._node_type_id] = cls
            mcs._name_to_id[qualified_name] = cls._node_type_id

        return cls

    def __repr__(cls) -> str:
        return f"<SDFNode '{cls.__name__}'>"

    @classmethod
    def get_all_node_types(mcs) -> Dict[str, Type["SDFNode"]]:
        """Return all registered node types."""
        with mcs._lock:
            return {
                name: mcs._registry[id_]
                for name, id_ in mcs._name_to_id.items()
            }


# =============================================================================
# SDFNode - Base Class for All Nodes
# =============================================================================

class SDFNode(metaclass=SDFNodeMeta):
    """
    Base class for all SDF AST nodes.

    Every node has:
    - Unique instance ID
    - Dirty tracking
    - Mirror introspection
    - Tree traversal
    """

    _instance_counter: ClassVar[int] = 0
    _counter_lock: ClassVar[threading.Lock] = threading.Lock()

    # Set by metaclass
    _node_type_id: ClassVar[int]
    _field_names: ClassVar[FrozenSet[str]]
    _field_types: ClassVar[Dict[str, type]]

    __slots__ = ("_node_id", "_dirty_fields", "_version", "_mirror", "_tracker")

    def __init__(self) -> None:
        """Initialize node with unique ID and tracking."""
        with SDFNode._counter_lock:
            SDFNode._instance_counter += 1
            self._node_id = SDFNode._instance_counter

        self._dirty_fields: set = set()
        self._version: int = 0
        self._mirror: Optional[Mirror] = None
        self._tracker: Optional[Tracker] = None

    @property
    def mirror(self) -> Mirror:
        """Get Mirror instance for introspection."""
        if self._mirror is None:
            self._mirror = Mirror(self)
        return self._mirror

    @property
    def tracker(self) -> Tracker:
        """Get Tracker instance for dirty tracking."""
        if self._tracker is None:
            self._tracker = Tracker(self)
        return self._tracker

    def children(self) -> Tuple["SDFNode", ...]:
        """Return child nodes. Override in subclasses."""
        return ()

    def walk(self, depth: int = 0) -> Generator[Tuple["SDFNode", int], None, None]:
        """Walk tree depth-first, yielding (node, depth) pairs."""
        yield self, depth
        for child in self.children():
            yield from child.walk(depth + 1)

    def label(self) -> str:
        """Return a short label for debugging."""
        return type(self).__name__

    def pretty(self, indent: int = 0) -> str:
        """Return pretty-printed tree representation."""
        lines = ["  " * indent + self.label()]
        for child in self.children():
            lines.append(child.pretty(indent + 1))
        return "\n".join(lines)

    def clone(self) -> "SDFNode":
        """Create a deep copy of the node tree."""
        raise NotImplementedError("Subclasses must implement clone()")

    def __repr__(self) -> str:
        return f"<{self.label()} id={self._node_id}>"


# =============================================================================
# PrimitiveNode - Base Class for SDF Primitives (T-DEMO-2.1)
# =============================================================================

class PrimitiveNode(SDFNode):
    """
    Base class for SDF primitive shapes.

    Primitives are leaf nodes that define basic geometric shapes
    at specific positions.
    """

    __slots__ = ("position",)

    position: Vec3

    def __init__(self, position: Optional[Vec3] = None) -> None:
        super().__init__()
        self.position = position or Vec3()

    @property
    def wgsl_function(self) -> str:
        """Return the WGSL function name for this primitive."""
        raise NotImplementedError


# =============================================================================
# Primitive Node Implementations (T-DEMO-2.1)
# =============================================================================

class SphereNode(PrimitiveNode):
    """Sphere SDF primitive."""

    __slots__ = ("radius",)

    radius: float

    def __init__(
        self,
        radius: float = 1.0,
        position: Optional[Vec3] = None,
    ) -> None:
        super().__init__(position)
        self.radius = radius
        self.tracker.mark_dirty("radius")
        self.tracker.mark_dirty("position")

    @property
    def wgsl_function(self) -> str:
        return "sdf_sphere"

    def label(self) -> str:
        return f"Sphere(r={self.radius})"

    def clone(self) -> "SphereNode":
        return SphereNode(self.radius, self.position)


class BoxNode(PrimitiveNode):
    """Axis-aligned box SDF primitive."""

    __slots__ = ("half_extents",)

    half_extents: Vec3

    def __init__(
        self,
        half_extents: Optional[Vec3] = None,
        position: Optional[Vec3] = None,
    ) -> None:
        super().__init__(position)
        self.half_extents = half_extents or Vec3(1.0, 1.0, 1.0)
        self.tracker.mark_dirty("half_extents")

    @property
    def wgsl_function(self) -> str:
        return "sdf_box"

    def label(self) -> str:
        return f"Box(size={self.half_extents.as_tuple()})"

    def clone(self) -> "BoxNode":
        return BoxNode(self.half_extents, self.position)


class TorusNode(PrimitiveNode):
    """Torus SDF primitive (donut shape)."""

    __slots__ = ("major_radius", "minor_radius")

    major_radius: float
    minor_radius: float

    def __init__(
        self,
        major_radius: float = 1.0,
        minor_radius: float = 0.25,
        position: Optional[Vec3] = None,
    ) -> None:
        super().__init__(position)
        self.major_radius = major_radius
        self.minor_radius = minor_radius
        self.tracker.mark_dirty("major_radius")
        self.tracker.mark_dirty("minor_radius")

    @property
    def wgsl_function(self) -> str:
        return "sdf_torus"

    def label(self) -> str:
        return f"Torus(R={self.major_radius}, r={self.minor_radius})"

    def clone(self) -> "TorusNode":
        return TorusNode(self.major_radius, self.minor_radius, self.position)


class CylinderNode(PrimitiveNode):
    """Capped cylinder SDF primitive."""

    __slots__ = ("radius", "height")

    radius: float
    height: float

    def __init__(
        self,
        radius: float = 0.5,
        height: float = 1.0,
        position: Optional[Vec3] = None,
    ) -> None:
        super().__init__(position)
        self.radius = radius
        self.height = height
        self.tracker.mark_dirty("radius")
        self.tracker.mark_dirty("height")

    @property
    def wgsl_function(self) -> str:
        return "sdf_cylinder"

    def label(self) -> str:
        return f"Cylinder(r={self.radius}, h={self.height})"

    def clone(self) -> "CylinderNode":
        return CylinderNode(self.radius, self.height, self.position)


class ConeNode(PrimitiveNode):
    """Capped cone SDF primitive."""

    __slots__ = ("angle", "height")

    angle: float  # Half-angle at apex in radians
    height: float

    def __init__(
        self,
        angle: float = 0.7854,  # 45 degrees
        height: float = 1.0,
        position: Optional[Vec3] = None,
    ) -> None:
        super().__init__(position)
        self.angle = angle
        self.height = height
        self.tracker.mark_dirty("angle")
        self.tracker.mark_dirty("height")

    @property
    def wgsl_function(self) -> str:
        return "sdf_cone"

    def label(self) -> str:
        import math
        degrees = math.degrees(self.angle)
        return f"Cone(angle={degrees:.1f}deg, h={self.height})"

    def clone(self) -> "ConeNode":
        return ConeNode(self.angle, self.height, self.position)


class PlaneNode(PrimitiveNode):
    """Infinite plane SDF primitive."""

    __slots__ = ("normal", "distance")

    normal: Vec3
    distance: float

    def __init__(
        self,
        normal: Optional[Vec3] = None,
        distance: float = 0.0,
        position: Optional[Vec3] = None,
    ) -> None:
        super().__init__(position)
        self.normal = (normal or Vec3(0.0, 1.0, 0.0)).normalized()
        self.distance = distance
        self.tracker.mark_dirty("normal")
        self.tracker.mark_dirty("distance")

    @property
    def wgsl_function(self) -> str:
        return "sdf_plane"

    def label(self) -> str:
        return f"Plane(n={self.normal.as_tuple()}, d={self.distance})"

    def clone(self) -> "PlaneNode":
        return PlaneNode(self.normal, self.distance, self.position)


class CapsuleNode(PrimitiveNode):
    """Capsule (line segment with radius) SDF primitive."""

    __slots__ = ("endpoint_a", "endpoint_b", "radius")

    endpoint_a: Vec3
    endpoint_b: Vec3
    radius: float

    def __init__(
        self,
        endpoint_a: Optional[Vec3] = None,
        endpoint_b: Optional[Vec3] = None,
        radius: float = 0.25,
        position: Optional[Vec3] = None,
    ) -> None:
        super().__init__(position)
        self.endpoint_a = endpoint_a or Vec3(0.0, -0.5, 0.0)
        self.endpoint_b = endpoint_b or Vec3(0.0, 0.5, 0.0)
        self.radius = radius
        self.tracker.mark_dirty("endpoint_a")
        self.tracker.mark_dirty("endpoint_b")
        self.tracker.mark_dirty("radius")

    @property
    def wgsl_function(self) -> str:
        return "sdf_capsule"

    def label(self) -> str:
        return f"Capsule(r={self.radius})"

    def clone(self) -> "CapsuleNode":
        return CapsuleNode(self.endpoint_a, self.endpoint_b, self.radius, self.position)


class EllipsoidNode(PrimitiveNode):
    """Ellipsoid SDF primitive."""

    __slots__ = ("radii",)

    radii: Vec3

    def __init__(
        self,
        radii: Optional[Vec3] = None,
        position: Optional[Vec3] = None,
    ) -> None:
        super().__init__(position)
        self.radii = radii or Vec3(1.0, 1.5, 1.0)
        self.tracker.mark_dirty("radii")

    @property
    def wgsl_function(self) -> str:
        return "sdf_ellipsoid"

    def label(self) -> str:
        return f"Ellipsoid(r={self.radii.as_tuple()})"

    def clone(self) -> "EllipsoidNode":
        return EllipsoidNode(self.radii, self.position)


class BoxFrameNode(PrimitiveNode):
    """Hollow box frame (edges only) SDF primitive."""

    __slots__ = ("half_extents", "edge_thickness")

    half_extents: Vec3
    edge_thickness: float

    def __init__(
        self,
        half_extents: Optional[Vec3] = None,
        edge_thickness: float = 0.05,
        position: Optional[Vec3] = None,
    ) -> None:
        super().__init__(position)
        self.half_extents = half_extents or Vec3(1.0, 1.0, 1.0)
        self.edge_thickness = edge_thickness
        self.tracker.mark_dirty("half_extents")
        self.tracker.mark_dirty("edge_thickness")

    @property
    def wgsl_function(self) -> str:
        return "sdf_box_frame"

    def label(self) -> str:
        return f"BoxFrame(e={self.edge_thickness})"

    def clone(self) -> "BoxFrameNode":
        return BoxFrameNode(self.half_extents, self.edge_thickness, self.position)


class RoundedBoxNode(PrimitiveNode):
    """Box with rounded corners SDF primitive."""

    __slots__ = ("half_extents", "corner_radius")

    half_extents: Vec3
    corner_radius: float

    def __init__(
        self,
        half_extents: Optional[Vec3] = None,
        corner_radius: float = 0.1,
        position: Optional[Vec3] = None,
    ) -> None:
        super().__init__(position)
        self.half_extents = half_extents or Vec3(1.0, 1.0, 1.0)
        self.corner_radius = corner_radius
        self.tracker.mark_dirty("half_extents")
        self.tracker.mark_dirty("corner_radius")

    @property
    def wgsl_function(self) -> str:
        return "sdf_rounded_box"

    def label(self) -> str:
        return f"RoundedBox(r={self.corner_radius})"

    def clone(self) -> "RoundedBoxNode":
        return RoundedBoxNode(self.half_extents, self.corner_radius, self.position)


class OctahedronNode(PrimitiveNode):
    """Regular octahedron SDF primitive."""

    __slots__ = ("size",)

    size: float

    def __init__(
        self,
        size: float = 1.0,
        position: Optional[Vec3] = None,
    ) -> None:
        super().__init__(position)
        self.size = size
        self.tracker.mark_dirty("size")

    @property
    def wgsl_function(self) -> str:
        return "sdf_octahedron"

    def label(self) -> str:
        return f"Octahedron(s={self.size})"

    def clone(self) -> "OctahedronNode":
        return OctahedronNode(self.size, self.position)


class PyramidNode(PrimitiveNode):
    """Square pyramid SDF primitive."""

    __slots__ = ("height",)

    height: float

    def __init__(
        self,
        height: float = 1.0,
        position: Optional[Vec3] = None,
    ) -> None:
        super().__init__(position)
        self.height = height
        self.tracker.mark_dirty("height")

    @property
    def wgsl_function(self) -> str:
        return "sdf_pyramid"

    def label(self) -> str:
        return f"Pyramid(h={self.height})"

    def clone(self) -> "PyramidNode":
        return PyramidNode(self.height, self.position)


# =============================================================================
# CombinatorNode - Base Class for Boolean Operations
# =============================================================================

class CombinatorNode(SDFNode):
    """
    Base class for SDF combinator operations.

    Combinators combine two SDFs using boolean operations
    (union, intersection, subtraction).
    """

    __slots__ = ("left", "right")

    left: SDFNode
    right: SDFNode

    def __init__(self, left: SDFNode, right: SDFNode) -> None:
        super().__init__()
        self.left = left
        self.right = right
        self.tracker.mark_dirty("left")
        self.tracker.mark_dirty("right")

    def children(self) -> Tuple[SDFNode, ...]:
        return (self.left, self.right)

    @property
    def wgsl_function(self) -> str:
        """Return the WGSL function name for this combinator."""
        raise NotImplementedError


class UnionNode(CombinatorNode):
    """Union of two SDFs (CSG OR)."""

    @property
    def wgsl_function(self) -> str:
        return "sdf_union"

    def label(self) -> str:
        return "Union"

    def clone(self) -> "UnionNode":
        return UnionNode(self.left.clone(), self.right.clone())


class IntersectionNode(CombinatorNode):
    """Intersection of two SDFs (CSG AND)."""

    @property
    def wgsl_function(self) -> str:
        return "sdf_intersection"

    def label(self) -> str:
        return "Intersection"

    def clone(self) -> "IntersectionNode":
        return IntersectionNode(self.left.clone(), self.right.clone())


class SubtractionNode(CombinatorNode):
    """Subtraction of one SDF from another (CSG DIFF)."""

    @property
    def wgsl_function(self) -> str:
        return "sdf_subtraction"

    def label(self) -> str:
        return "Subtraction"

    def clone(self) -> "SubtractionNode":
        return SubtractionNode(self.left.clone(), self.right.clone())


class SmoothUnionNode(CombinatorNode):
    """Smooth union with blending factor."""

    __slots__ = ("k",)

    k: float

    def __init__(self, left: SDFNode, right: SDFNode, k: float = 0.1) -> None:
        super().__init__(left, right)
        self.k = k
        self.tracker.mark_dirty("k")

    @property
    def wgsl_function(self) -> str:
        return "sdf_smooth_union"

    def label(self) -> str:
        return f"SmoothUnion(k={self.k})"

    def clone(self) -> "SmoothUnionNode":
        return SmoothUnionNode(self.left.clone(), self.right.clone(), self.k)


class SmoothIntersectionNode(CombinatorNode):
    """Smooth intersection with blending factor."""

    __slots__ = ("k",)

    k: float

    def __init__(self, left: SDFNode, right: SDFNode, k: float = 0.1) -> None:
        super().__init__(left, right)
        self.k = k
        self.tracker.mark_dirty("k")

    @property
    def wgsl_function(self) -> str:
        return "sdf_smooth_intersection"

    def label(self) -> str:
        return f"SmoothIntersection(k={self.k})"

    def clone(self) -> "SmoothIntersectionNode":
        return SmoothIntersectionNode(self.left.clone(), self.right.clone(), self.k)


class SmoothSubtractionNode(CombinatorNode):
    """Smooth subtraction with blending factor."""

    __slots__ = ("k",)

    k: float

    def __init__(self, left: SDFNode, right: SDFNode, k: float = 0.1) -> None:
        super().__init__(left, right)
        self.k = k
        self.tracker.mark_dirty("k")

    @property
    def wgsl_function(self) -> str:
        return "sdf_smooth_subtraction"

    def label(self) -> str:
        return f"SmoothSubtraction(k={self.k})"

    def clone(self) -> "SmoothSubtractionNode":
        return SmoothSubtractionNode(self.left.clone(), self.right.clone(), self.k)


class DisplacedNode(SDFNode):
    """Applies noise displacement to an SDF."""

    __slots__ = ("child", "amplitude", "frequency")

    child: SDFNode
    amplitude: float
    frequency: float

    def __init__(
        self,
        child: SDFNode,
        amplitude: float = 0.1,
        frequency: float = 1.0,
    ) -> None:
        super().__init__()
        self.child = child
        self.amplitude = amplitude
        self.frequency = frequency
        self.tracker.mark_dirty("child")
        self.tracker.mark_dirty("amplitude")
        self.tracker.mark_dirty("frequency")

    def children(self) -> Tuple[SDFNode, ...]:
        return (self.child,)

    @property
    def wgsl_function(self) -> str:
        return "sdf_displaced"

    def label(self) -> str:
        return f"Displaced(amp={self.amplitude}, freq={self.frequency})"

    def clone(self) -> "DisplacedNode":
        return DisplacedNode(self.child.clone(), self.amplitude, self.frequency)


# =============================================================================
# DomainOpNode - Base Class for Domain Operations
# =============================================================================

class DomainOpNode(SDFNode):
    """
    Base class for domain operations.

    Domain operations transform the coordinate space before
    evaluating the child SDF.
    """

    __slots__ = ("child",)

    child: SDFNode

    def __init__(self, child: SDFNode) -> None:
        super().__init__()
        self.child = child
        self.tracker.mark_dirty("child")

    def children(self) -> Tuple[SDFNode, ...]:
        return (self.child,)

    @property
    def wgsl_function(self) -> str:
        """Return the WGSL function name for this operation."""
        raise NotImplementedError


class RepeatNode(DomainOpNode):
    """Infinite repetition of space."""

    __slots__ = ("cell_size",)

    cell_size: Vec3

    def __init__(
        self,
        child: SDFNode,
        cell_size: Optional[Vec3] = None,
    ) -> None:
        super().__init__(child)
        self.cell_size = cell_size or Vec3(2.0, 2.0, 2.0)
        self.tracker.mark_dirty("cell_size")

    @property
    def wgsl_function(self) -> str:
        return "domain_repeat"

    def label(self) -> str:
        return f"Repeat(cell={self.cell_size.as_tuple()})"

    def clone(self) -> "RepeatNode":
        return RepeatNode(self.child.clone(), self.cell_size)


class MirrorNode(DomainOpNode):
    """Mirror space across an axis plane."""

    __slots__ = ("axis",)

    axis: Axis

    def __init__(self, child: SDFNode, axis: Axis = Axis.X) -> None:
        super().__init__(child)
        self.axis = axis
        self.tracker.mark_dirty("axis")

    @property
    def wgsl_function(self) -> str:
        return f"domain_mirror_{self.axis.value}"

    def label(self) -> str:
        return f"Mirror({self.axis.value.upper()})"

    def clone(self) -> "MirrorNode":
        return MirrorNode(self.child.clone(), self.axis)


class KIFSNode(DomainOpNode):
    """Kaleidoscopic Iterated Function System fold."""

    __slots__ = ("iterations", "scale", "offset")

    iterations: int
    scale: float
    offset: Vec3

    def __init__(
        self,
        child: SDFNode,
        iterations: int = 6,
        scale: float = 2.0,
        offset: Optional[Vec3] = None,
    ) -> None:
        super().__init__(child)
        self.iterations = iterations
        self.scale = scale
        self.offset = offset or Vec3(1.0, 1.0, 1.0)
        self.tracker.mark_dirty("iterations")
        self.tracker.mark_dirty("scale")
        self.tracker.mark_dirty("offset")

    @property
    def wgsl_function(self) -> str:
        return "domain_fold_kifs"

    def label(self) -> str:
        return f"KIFS(iter={self.iterations}, scale={self.scale})"

    def clone(self) -> "KIFSNode":
        return KIFSNode(self.child.clone(), self.iterations, self.scale, self.offset)


class TwistNode(DomainOpNode):
    """Twist space around an axis."""

    __slots__ = ("axis", "rate")

    axis: Axis
    rate: float

    def __init__(
        self,
        child: SDFNode,
        axis: Axis = Axis.Y,
        rate: float = 0.5,
    ) -> None:
        super().__init__(child)
        self.axis = axis
        self.rate = rate
        self.tracker.mark_dirty("axis")
        self.tracker.mark_dirty("rate")

    @property
    def wgsl_function(self) -> str:
        return f"domain_twist_{self.axis.value}"

    def label(self) -> str:
        return f"Twist({self.axis.value.upper()}, rate={self.rate})"

    def clone(self) -> "TwistNode":
        return TwistNode(self.child.clone(), self.axis, self.rate)


class BendNode(DomainOpNode):
    """Bend space along a circular arc."""

    __slots__ = ("axis", "radius")

    axis: Axis
    radius: float

    def __init__(
        self,
        child: SDFNode,
        axis: Axis = Axis.Z,
        radius: float = 10.0,
    ) -> None:
        super().__init__(child)
        self.axis = axis
        self.radius = radius
        self.tracker.mark_dirty("axis")
        self.tracker.mark_dirty("radius")

    @property
    def wgsl_function(self) -> str:
        return f"domain_bend_{self.axis.value}"

    def label(self) -> str:
        return f"Bend({self.axis.value.upper()}, r={self.radius})"

    def clone(self) -> "BendNode":
        return BendNode(self.child.clone(), self.axis, self.radius)


class StretchNode(DomainOpNode):
    """Stretch space along an axis."""

    __slots__ = ("axis", "scale")

    axis: Axis
    scale: float

    def __init__(
        self,
        child: SDFNode,
        axis: Axis = Axis.X,
        scale: float = 2.0,
    ) -> None:
        super().__init__(child)
        self.axis = axis
        self.scale = scale
        self.tracker.mark_dirty("axis")
        self.tracker.mark_dirty("scale")

    @property
    def wgsl_function(self) -> str:
        return f"domain_stretch_{self.axis.value}"

    def label(self) -> str:
        return f"Stretch({self.axis.value.upper()}, s={self.scale})"

    def clone(self) -> "StretchNode":
        return StretchNode(self.child.clone(), self.axis, self.scale)


# =============================================================================
# MaterialNode - Material Properties
# =============================================================================

class MaterialNode(SDFNode):
    """Material properties for shading."""

    __slots__ = ("color", "metallic", "roughness", "emission", "material_id")

    color: Vec3
    metallic: float
    roughness: float
    emission: Vec3
    material_id: int

    def __init__(
        self,
        color: Optional[Vec3] = None,
        metallic: float = 0.0,
        roughness: float = 0.5,
        emission: Optional[Vec3] = None,
        material_id: int = 0,
    ) -> None:
        super().__init__()
        self.color = color or Vec3(0.8, 0.8, 0.8)
        self.metallic = metallic
        self.roughness = roughness
        self.emission = emission or Vec3(0.0, 0.0, 0.0)
        self.material_id = material_id
        self.tracker.mark_dirty("color")
        self.tracker.mark_dirty("metallic")
        self.tracker.mark_dirty("roughness")
        self.tracker.mark_dirty("emission")
        self.tracker.mark_dirty("material_id")

    def label(self) -> str:
        return f"Material(id={self.material_id})"

    def clone(self) -> "MaterialNode":
        return MaterialNode(
            self.color, self.metallic, self.roughness, self.emission, self.material_id
        )


# =============================================================================
# CameraNode - Camera Parameters
# =============================================================================

class CameraNode(SDFNode):
    """Camera configuration for ray generation."""

    __slots__ = (
        "origin", "look_at", "up", "fov", "aspect_ratio",
        "aperture", "focal_distance",
    )

    origin: Vec3
    look_at: Vec3
    up: Vec3
    fov: float
    aspect_ratio: float
    aperture: float
    focal_distance: float

    def __init__(
        self,
        origin: Optional[Vec3] = None,
        look_at: Optional[Vec3] = None,
        up: Optional[Vec3] = None,
        fov: float = 60.0,
        aspect_ratio: float = 16.0 / 9.0,
        aperture: float = 0.0,
        focal_distance: float = 10.0,
    ) -> None:
        super().__init__()
        self.origin = origin or Vec3(0.0, 0.0, 5.0)
        self.look_at = look_at or Vec3(0.0, 0.0, 0.0)
        self.up = up or Vec3(0.0, 1.0, 0.0)
        self.fov = fov
        self.aspect_ratio = aspect_ratio
        self.aperture = aperture
        self.focal_distance = focal_distance
        for field in ("origin", "look_at", "up", "fov", "aspect_ratio",
                      "aperture", "focal_distance"):
            self.tracker.mark_dirty(field)

    def label(self) -> str:
        return f"Camera(fov={self.fov})"

    def clone(self) -> "CameraNode":
        return CameraNode(
            self.origin, self.look_at, self.up, self.fov,
            self.aspect_ratio, self.aperture, self.focal_distance,
        )


# =============================================================================
# LightNode - Light Source
# =============================================================================

class LightNode(SDFNode):
    """Light source for shading."""

    __slots__ = ("position", "color", "intensity")

    position: Vec3
    color: Vec3
    intensity: float

    def __init__(
        self,
        position: Optional[Vec3] = None,
        color: Optional[Vec3] = None,
        intensity: float = 1.0,
    ) -> None:
        super().__init__()
        self.position = position or Vec3(5.0, 5.0, 5.0)
        self.color = color or Vec3(1.0, 1.0, 1.0)
        self.intensity = intensity
        self.tracker.mark_dirty("position")
        self.tracker.mark_dirty("color")
        self.tracker.mark_dirty("intensity")

    def label(self) -> str:
        return f"Light(i={self.intensity})"

    def clone(self) -> "LightNode":
        return LightNode(self.position, self.color, self.intensity)


# =============================================================================
# RenderSettingsNode - Render Configuration
# =============================================================================

class RenderSettingsNode(SDFNode):
    """Render configuration for the compute shader."""

    __slots__ = (
        "width", "height", "max_steps", "max_distance",
        "epsilon", "workgroup_size",
    )

    width: int
    height: int
    max_steps: int
    max_distance: float
    epsilon: float
    workgroup_size: Tuple[int, int]

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        max_steps: int = 256,
        max_distance: float = 100.0,
        epsilon: float = 0.001,
        workgroup_size: Tuple[int, int] = (8, 8),
    ) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.max_distance = max_distance
        self.epsilon = epsilon
        self.workgroup_size = workgroup_size
        for field in ("width", "height", "max_steps", "max_distance",
                      "epsilon", "workgroup_size"):
            self.tracker.mark_dirty(field)

    def label(self) -> str:
        return f"RenderSettings({self.width}x{self.height})"

    def clone(self) -> "RenderSettingsNode":
        return RenderSettingsNode(
            self.width, self.height, self.max_steps, self.max_distance,
            self.epsilon, self.workgroup_size,
        )


# =============================================================================
# SceneNode - Root Scene Container
# =============================================================================

class SceneNode(SDFNode):
    """
    Root scene node containing the complete SDF scene.

    Holds the scene graph, camera, lights, materials, and render settings.
    """

    __slots__ = ("root", "camera", "lights", "materials", "render_settings", "name")

    root: SDFNode
    camera: CameraNode
    lights: Tuple[LightNode, ...]
    materials: Tuple[MaterialNode, ...]
    render_settings: RenderSettingsNode
    name: str

    def __init__(
        self,
        root: SDFNode,
        camera: Optional[CameraNode] = None,
        lights: Optional[Sequence[LightNode]] = None,
        materials: Optional[Sequence[MaterialNode]] = None,
        render_settings: Optional[RenderSettingsNode] = None,
        name: str = "",
    ) -> None:
        super().__init__()
        self.root = root
        self.camera = camera or CameraNode()
        self.lights = tuple(lights) if lights else (LightNode(),)
        self.materials = tuple(materials) if materials else ()
        self.render_settings = render_settings or RenderSettingsNode()
        self.name = name
        for field in ("root", "camera", "lights", "materials", "render_settings", "name"):
            self.tracker.mark_dirty(field)

    def children(self) -> Tuple[SDFNode, ...]:
        return (self.root, self.camera, self.render_settings) + self.lights + self.materials

    def label(self) -> str:
        return f"Scene({self.name or 'unnamed'})"

    def clone(self) -> "SceneNode":
        return SceneNode(
            self.root.clone(),
            self.camera.clone(),
            tuple(light.clone() for light in self.lights),
            tuple(mat.clone() for mat in self.materials),
            self.render_settings.clone(),
            self.name,
        )


# =============================================================================
# build_ast - DSL Object to AST Conversion
# =============================================================================

# Type dispatch table for DSL objects
_PRIMITIVE_MAP: Dict[str, Type[PrimitiveNode]] = {
    "sphere": SphereNode,
    "box": BoxNode,
    "torus": TorusNode,
    "cylinder": CylinderNode,
    "cone": ConeNode,
    "plane": PlaneNode,
    "capsule": CapsuleNode,
    "ellipsoid": EllipsoidNode,
    "box_frame": BoxFrameNode,
    "rounded_box": RoundedBoxNode,
    "octahedron": OctahedronNode,
    "pyramid": PyramidNode,
}

_COMBINATOR_MAP: Dict[str, Type[CombinatorNode]] = {
    "union": UnionNode,
    "intersection": IntersectionNode,
    "subtraction": SubtractionNode,
    "smooth_union": SmoothUnionNode,
    "smooth_intersection": SmoothIntersectionNode,
    "smooth_subtraction": SmoothSubtractionNode,
}

_DOMAIN_OP_MAP: Dict[str, Type[DomainOpNode]] = {
    "repeat": RepeatNode,
    "mirror": MirrorNode,
    "kifs": KIFSNode,
    "twist": TwistNode,
    "bend": BendNode,
    "stretch": StretchNode,
}


def _to_vec3(value: Any) -> Vec3:
    """Convert various types to Vec3."""
    if isinstance(value, Vec3):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return Vec3(float(value[0]), float(value[1]), float(value[2]))
    if isinstance(value, (int, float)):
        return Vec3.from_scalar(float(value))
    raise TypeError(f"Cannot convert {type(value).__name__} to Vec3")


def _to_axis(value: Any) -> Axis:
    """Convert various types to Axis."""
    if isinstance(value, Axis):
        return value
    if isinstance(value, str):
        return Axis(value.lower())
    raise TypeError(f"Cannot convert {type(value).__name__} to Axis")


def build_ast(obj: Any) -> SDFNode:
    """
    Recursively build an AST from a Python DSL object.

    Accepts:
    - dict with "type" key
    - SDFNode instances (returned as-is)
    - Objects with _node_type attribute

    Example:
        scene = build_ast({
            "type": "scene",
            "root": {
                "type": "union",
                "left": {"type": "sphere", "radius": 1.0},
                "right": {"type": "box", "half_extents": [1, 1, 1]},
            },
        })
    """
    if isinstance(obj, SDFNode):
        return obj

    if isinstance(obj, dict):
        return _build_from_dict(obj)

    # Check for DSL object with _node_type
    node_type = getattr(obj, "_node_type", None)
    if node_type:
        return _build_from_dsl_object(obj, node_type)

    raise TypeError(f"Cannot build AST from {type(obj).__name__}")


def _build_from_dict(d: Dict[str, Any]) -> SDFNode:
    """Build AST node from dictionary representation."""
    node_type = d.get("type", "").lower()

    # Check primitives
    if node_type in _PRIMITIVE_MAP:
        return _build_primitive(node_type, d)

    # Check combinators
    if node_type in _COMBINATOR_MAP:
        return _build_combinator(node_type, d)

    # Check domain ops
    if node_type in _DOMAIN_OP_MAP:
        return _build_domain_op(node_type, d)

    # Check special nodes
    if node_type == "displaced":
        child = build_ast(d["child"])
        return DisplacedNode(
            child,
            amplitude=d.get("amplitude", 0.1),
            frequency=d.get("frequency", 1.0),
        )

    if node_type == "material":
        return MaterialNode(
            color=_to_vec3(d.get("color", [0.8, 0.8, 0.8])),
            metallic=d.get("metallic", 0.0),
            roughness=d.get("roughness", 0.5),
            emission=_to_vec3(d.get("emission", [0, 0, 0])),
            material_id=d.get("material_id", 0),
        )

    if node_type == "camera":
        return CameraNode(
            origin=_to_vec3(d.get("origin", [0, 0, 5])),
            look_at=_to_vec3(d.get("look_at", [0, 0, 0])),
            up=_to_vec3(d.get("up", [0, 1, 0])),
            fov=d.get("fov", 60.0),
            aspect_ratio=d.get("aspect_ratio", 16.0 / 9.0),
            aperture=d.get("aperture", 0.0),
            focal_distance=d.get("focal_distance", 10.0),
        )

    if node_type == "light":
        return LightNode(
            position=_to_vec3(d.get("position", [5, 5, 5])),
            color=_to_vec3(d.get("color", [1, 1, 1])),
            intensity=d.get("intensity", 1.0),
        )

    if node_type == "render_settings":
        return RenderSettingsNode(
            width=d.get("width", 1920),
            height=d.get("height", 1080),
            max_steps=d.get("max_steps", 256),
            max_distance=d.get("max_distance", 100.0),
            epsilon=d.get("epsilon", 0.001),
            workgroup_size=tuple(d.get("workgroup_size", [8, 8])),
        )

    if node_type == "scene":
        root = build_ast(d["root"])
        camera = build_ast(d.get("camera", {"type": "camera"})) if "camera" in d else None
        lights = [build_ast(l) for l in d.get("lights", [])] if "lights" in d else None
        materials = [build_ast(m) for m in d.get("materials", [])] if "materials" in d else None
        settings = (
            build_ast(d.get("render_settings", {"type": "render_settings"}))
            if "render_settings" in d else None
        )
        return SceneNode(
            root,
            camera,
            lights,
            materials,
            settings,
            name=d.get("name", ""),
        )

    raise ValueError(f"Unknown node type: {node_type}")


def _build_primitive(node_type: str, d: Dict[str, Any]) -> PrimitiveNode:
    """Build a primitive node from dictionary."""
    position = _to_vec3(d.get("position", [0, 0, 0])) if "position" in d else None

    if node_type == "sphere":
        return SphereNode(d.get("radius", 1.0), position)
    if node_type == "box":
        half_extents = _to_vec3(d.get("half_extents", [1, 1, 1]))
        return BoxNode(half_extents, position)
    if node_type == "torus":
        return TorusNode(
            d.get("major_radius", 1.0),
            d.get("minor_radius", 0.25),
            position,
        )
    if node_type == "cylinder":
        return CylinderNode(d.get("radius", 0.5), d.get("height", 1.0), position)
    if node_type == "cone":
        return ConeNode(d.get("angle", 0.7854), d.get("height", 1.0), position)
    if node_type == "plane":
        normal = _to_vec3(d.get("normal", [0, 1, 0]))
        return PlaneNode(normal, d.get("distance", 0.0), position)
    if node_type == "capsule":
        return CapsuleNode(
            _to_vec3(d.get("endpoint_a", [0, -0.5, 0])),
            _to_vec3(d.get("endpoint_b", [0, 0.5, 0])),
            d.get("radius", 0.25),
            position,
        )
    if node_type == "ellipsoid":
        radii = _to_vec3(d.get("radii", [1, 1.5, 1]))
        return EllipsoidNode(radii, position)
    if node_type == "box_frame":
        half_extents = _to_vec3(d.get("half_extents", [1, 1, 1]))
        return BoxFrameNode(half_extents, d.get("edge_thickness", 0.05), position)
    if node_type == "rounded_box":
        half_extents = _to_vec3(d.get("half_extents", [1, 1, 1]))
        return RoundedBoxNode(half_extents, d.get("corner_radius", 0.1), position)
    if node_type == "octahedron":
        return OctahedronNode(d.get("size", 1.0), position)
    if node_type == "pyramid":
        return PyramidNode(d.get("height", 1.0), position)

    raise ValueError(f"Unknown primitive: {node_type}")


def _build_combinator(node_type: str, d: Dict[str, Any]) -> CombinatorNode:
    """Build a combinator node from dictionary."""
    left = build_ast(d["left"])
    right = build_ast(d["right"])

    if node_type == "union":
        return UnionNode(left, right)
    if node_type == "intersection":
        return IntersectionNode(left, right)
    if node_type == "subtraction":
        return SubtractionNode(left, right)
    if node_type == "smooth_union":
        return SmoothUnionNode(left, right, d.get("k", 0.1))
    if node_type == "smooth_intersection":
        return SmoothIntersectionNode(left, right, d.get("k", 0.1))
    if node_type == "smooth_subtraction":
        return SmoothSubtractionNode(left, right, d.get("k", 0.1))

    raise ValueError(f"Unknown combinator: {node_type}")


def _build_domain_op(node_type: str, d: Dict[str, Any]) -> DomainOpNode:
    """Build a domain operation node from dictionary."""
    child = build_ast(d["child"])

    if node_type == "repeat":
        cell_size = _to_vec3(d.get("cell_size", [2, 2, 2]))
        return RepeatNode(child, cell_size)
    if node_type == "mirror":
        axis = _to_axis(d.get("axis", "x"))
        return MirrorNode(child, axis)
    if node_type == "kifs":
        offset = _to_vec3(d.get("offset", [1, 1, 1])) if "offset" in d else None
        return KIFSNode(
            child,
            d.get("iterations", 6),
            d.get("scale", 2.0),
            offset,
        )
    if node_type == "twist":
        axis = _to_axis(d.get("axis", "y"))
        return TwistNode(child, axis, d.get("rate", 0.5))
    if node_type == "bend":
        axis = _to_axis(d.get("axis", "z"))
        return BendNode(child, axis, d.get("radius", 10.0))
    if node_type == "stretch":
        axis = _to_axis(d.get("axis", "x"))
        return StretchNode(child, axis, d.get("scale", 2.0))

    raise ValueError(f"Unknown domain op: {node_type}")


def _build_from_dsl_object(obj: Any, node_type: str) -> SDFNode:
    """Build AST node from DSL object with _node_type attribute."""
    # Convert DSL object to dictionary
    d = {"type": node_type}
    for attr in dir(obj):
        if not attr.startswith("_"):
            value = getattr(obj, attr)
            if not callable(value):
                d[attr] = value
    return _build_from_dict(d)
