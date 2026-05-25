from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generator, Sequence, Tuple

@dataclass(frozen=True)
class ExprNode:
    def walk(self, depth=0):
        yield self, depth
        for c in self.children():
            yield from c.walk(depth + 1)
    def children(self) -> Sequence[ExprNode]:
        return []
    def pretty(self, indent=0):
        return "  " * indent + self.label()
    def label(self):
        return type(self).__name__

@dataclass(frozen=True)
class FloatNode(ExprNode):
    value: float
    def label(self):
        return f"Float({self.value})"

@dataclass(frozen=True)
class Vec3Node(ExprNode):
    x: float; y: float; z: float
    @classmethod
    def from_tuple(cls, t):
        return cls(*t)
    def as_tuple(self):
        return (float(self.x), float(self.y), float(self.z))
    def label(self):
        return f"Vec3({float(self.x)}, {float(self.y)}, {float(self.z)})"

@dataclass(frozen=True)
class PositionNode(ExprNode):
    def label(self):
        return "Position(p)"

class Axis(Enum):
    X = "x"; Y = "y"; Z = "z"

@dataclass(frozen=True)
class DomainOpNode(ExprNode):
    input: ExprNode
    def children(self):
        return (self.input,)

@dataclass(frozen=True)
class RepeatNode(DomainOpNode):
    cell_size: Vec3Node
    def label(self):
        return f"Repeat(cell_size={self.cell_size.as_tuple()})"

@dataclass(frozen=True)
class CellIdNode(DomainOpNode):
    cell_size: Vec3Node
    def label(self):
        return f"CellId(cell_size={self.cell_size.as_tuple()})"

@dataclass(frozen=True)
class MirrorNode(DomainOpNode):
    axis: Axis
    def label(self):
        return f"Mirror(axis={self.axis.value.upper()})"

@dataclass(frozen=True)
class KifsNode(DomainOpNode):
    folds: FloatNode
    def label(self):
        return f"Kifs(folds={self.folds.value})"

@dataclass(frozen=True)
class TwistNode(DomainOpNode):
    rate: FloatNode
    def label(self):
        return f"Twist(rate={self.rate.value})"

@dataclass(frozen=True)
class BendNode(DomainOpNode):
    radius: FloatNode
    def label(self):
        return f"Bend(radius={self.radius.value})"

@dataclass(frozen=True)
class StretchNode(DomainOpNode):
    stretch: FloatNode; axis: Axis
    def label(self):
        return f"Stretch(axis={self.axis.value.upper()}, factor={self.stretch.value})"

class Kind(Enum):
    REPEAT = "repeat"; KIFS = "kifs"; STRETCH = "stretch"; TWIST = "twist"

@dataclass(frozen=True)
class CompensationNode(ExprNode):
    kind: Kind; param: float
    def label(self):
        return f"Compensation({self.kind.value}, {self.param})"

@dataclass(frozen=True)
class SdfPrimitiveNode(ExprNode):
    position: ExprNode
    def children(self):
        return (self.position,)

@dataclass(frozen=True)
class SphereNode(SdfPrimitiveNode):
    radius: FloatNode
    def label(self):
        return f"Sphere(r={self.radius.value})"

@dataclass(frozen=True)
class BoxNode(SdfPrimitiveNode):
    size: Vec3Node
    def label(self):
        return f"Box(size={self.size.as_tuple()})"

@dataclass(frozen=True)
class TorusNode(SdfPrimitiveNode):
    major_radius: FloatNode; minor_radius: FloatNode
    def label(self):
        return f"Torus(major={self.major_radius.value}, minor={self.minor_radius.value})"

@dataclass(frozen=True)
class CylinderNode(SdfPrimitiveNode):
    height: FloatNode; radius: FloatNode
    def label(self):
        return f"Cylinder(h={self.height.value}, r={self.radius.value})"

@dataclass(frozen=True)
class ConeNode(SdfPrimitiveNode):
    height: FloatNode; radius_top: FloatNode; radius_bottom: FloatNode
    def label(self):
        return f"Cone(h={self.height.value}, r1={self.radius_top.value}, r2={self.radius_bottom.value})"

@dataclass(frozen=True)
class PlaneNode(SdfPrimitiveNode):
    normal: Vec3Node; distance: FloatNode
    def label(self):
        return f"Plane(n={self.normal.as_tuple()}, d={self.distance.value})"

@dataclass(frozen=True)
class CapsuleNode(SdfPrimitiveNode):
    endpoint_a: Vec3Node; endpoint_b: Vec3Node; radius: FloatNode
    def children(self):
        return (self.position, self.endpoint_a, self.endpoint_b)
    def label(self):
        return f"Capsule(a={self.endpoint_a.as_tuple()}, b={self.endpoint_b.as_tuple()}, r={self.radius.value})"

@dataclass(frozen=True)
class CombineNode(ExprNode):
    kind: str; left: ExprNode; right: ExprNode
    def children(self):
        return self.left, self.right
    def label(self):
        return f"Combine({self.kind})"

@dataclass(frozen=True)
class UnionNode(CombineNode):
    def __init__(self, left, right):
        object.__setattr__(self, "kind", "union")
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
    def label(self):
        return "Union(...)"

@dataclass(frozen=True)
class IntersectionNode(CombineNode):
    def __init__(self, left, right):
        object.__setattr__(self, "kind", "intersection")
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
    def label(self):
        return "Intersection(...)"

@dataclass(frozen=True)
class SubtractionNode(CombineNode):
    def __init__(self, left, right):
        object.__setattr__(self, "kind", "subtraction")
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
    def label(self):
        return "Subtraction(...)"

@dataclass(frozen=True)
class SceneGraph(ExprNode):
    primitives: tuple[SdfPrimitiveNode, ...]
    pipeline: tuple[DomainOpNode, ...] = ()
    name: str = ""
    def children(self):
        return (*self.pipeline, *self.primitives)
    def label(self):
        np = f" {self.name!r}" if self.name else ""
        return f"SceneGraph{np} ({len(self.pipeline)} pipeline, {len(self.primitives)} primitives)"
    def deep_label(self):
        lines = [f"SceneGraph: {self.name or '(unnamed)'}"]
        if self.pipeline:
            lines.append("  Pipeline:")
            for op in self.pipeline:
                lines.append(f"    {op.label()}")
        lines.append("  Primitives:")
        for p in self.primitives:
            lines.append(f"    {p.label()}")
        return "\n".join(lines)


@dataclass(frozen=True)
class CameraNode(ExprNode):
    """Camera parameters for ray generation."""
    origin: Vec3Node
    look_at: Vec3Node
    up: Vec3Node
    fov: FloatNode
    aspect_ratio: FloatNode
    aperture: FloatNode = FloatNode(0.0)
    focal_distance: FloatNode = FloatNode(10.0)
    def label(self):
        return f"Camera(fov={self.fov.value}, ar={self.aspect_ratio.value})"

@dataclass(frozen=True)
class LightNode(ExprNode):
    """A directional or point light source."""
    position: Vec3Node
    color: Vec3Node
    intensity: FloatNode
    def label(self):
        return f"Light(pos={self.position.as_tuple()}, color={self.color.as_tuple()}, i={self.intensity.value})"

@dataclass(frozen=True)
class RenderSettingsNode(ExprNode):
    """Render configuration for the compute shader."""
    width: int = 1920
    height: int = 1080
    max_steps: int = 256
    max_distance: float = 100.0
    workgroup_size_x: int = 8
    workgroup_size_y: int = 8
    def label(self):
        return f"RenderSettings({self.width}x{self.height}, steps={self.max_steps}, dist={self.max_distance})"


SDF_PRIMITIVE_TYPE_MAP = {
    SphereNode: "sdSphere",
    BoxNode: "sdBox",
    TorusNode: "sdTorus",
    CylinderNode: "sdCylinder",
    ConeNode: "sdCone",
    PlaneNode: "sdPlane",
}

DOMAIN_OP_TYPE_MAP = {
    RepeatNode: "domain_repeat",
    CellIdNode: "domain_cell_id",
    MirrorNode: "domain_mirror",
    KifsNode: "domain_kifs",
    TwistNode: "domain_twist",
    BendNode: "domain_bend",
    StretchNode: "domain_stretch",
}
