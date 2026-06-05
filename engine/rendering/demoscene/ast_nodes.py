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
        return f"Capsule(r={self.radius.value})"

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

class LightType(Enum):
    """Light type enumeration for lighting calculations."""
    POINT = "point"
    DIRECTIONAL = "directional"
    AREA = "area"
    SPOT = "spot"


@dataclass(frozen=True)
class LightNode(ExprNode):
    """A directional or point light source."""
    position: Vec3Node
    color: Vec3Node
    intensity: FloatNode
    light_type: LightType = LightType.POINT
    direction: Vec3Node = None
    radius: FloatNode = None
    def __post_init__(self):
        # Set defaults for optional fields using object.__setattr__ for frozen dataclass
        if self.direction is None:
            object.__setattr__(self, 'direction', Vec3Node(0.0, -1.0, 0.0))
        if self.radius is None:
            object.__setattr__(self, 'radius', FloatNode(10.0))
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
    epsilon: float = 0.001
    def label(self):
        return f"RenderSettings({self.width}x{self.height}, steps={self.max_steps}, dist={self.max_distance})"


# =============================================================================
# Additional SDF Primitive Nodes
# =============================================================================

@dataclass(frozen=True)
class EllipsoidNode(SdfPrimitiveNode):
    """Ellipsoid SDF primitive with radii along each axis."""
    radii: Vec3Node
    def label(self):
        return f"Ellipsoid(radii={self.radii.as_tuple()})"


@dataclass(frozen=True)
class BoxFrameNode(SdfPrimitiveNode):
    """Hollow box frame (edges only) SDF primitive."""
    size: Vec3Node
    edge_thickness: FloatNode
    def label(self):
        return f"BoxFrame(size={self.size.as_tuple()}, edge={self.edge_thickness.value})"


@dataclass(frozen=True)
class RoundedBoxNode(SdfPrimitiveNode):
    """Box with rounded corners SDF primitive."""
    size: Vec3Node
    corner_radius: FloatNode
    def label(self):
        return f"RoundedBox(size={self.size.as_tuple()}, r={self.corner_radius.value})"


@dataclass(frozen=True)
class OctahedronNode(SdfPrimitiveNode):
    """Regular octahedron SDF primitive."""
    size: FloatNode
    def label(self):
        return f"Octahedron(size={self.size.value})"


@dataclass(frozen=True)
class PyramidNode(SdfPrimitiveNode):
    """Square pyramid SDF primitive."""
    height: FloatNode
    base_size: FloatNode = None
    def __post_init__(self):
        if self.base_size is None:
            object.__setattr__(self, 'base_size', FloatNode(1.0))
    def label(self):
        return f"Pyramid(h={self.height.value}, base={self.base_size.value})"


# =============================================================================
# Material Node
# =============================================================================

@dataclass(frozen=True)
class MaterialNode(ExprNode):
    """PBR material properties for shading."""
    albedo: Vec3Node = None
    roughness: FloatNode = None
    metallic: FloatNode = None
    ambient_occlusion: FloatNode = None
    emission: Vec3Node = None
    material_id: int = 0
    def __post_init__(self):
        if self.albedo is None:
            object.__setattr__(self, 'albedo', Vec3Node(0.8, 0.8, 0.8))
        if self.roughness is None:
            object.__setattr__(self, 'roughness', FloatNode(0.5))
        if self.metallic is None:
            object.__setattr__(self, 'metallic', FloatNode(0.0))
        if self.ambient_occlusion is None:
            object.__setattr__(self, 'ambient_occlusion', FloatNode(1.0))
        if self.emission is None:
            object.__setattr__(self, 'emission', Vec3Node(0.0, 0.0, 0.0))
    def label(self):
        return f"Material(id={self.material_id}, roughness={self.roughness.value}, metallic={self.metallic.value})"


# =============================================================================
# Full Scene Node
# =============================================================================

@dataclass(frozen=True)
class FullSceneNode(ExprNode):
    """Complete scene with geometry, camera, lights, and materials."""
    scene_graph: SceneGraph
    name: str = ""
    camera: CameraNode = None
    lights: Tuple[LightNode, ...] = ()
    materials: Tuple[MaterialNode, ...] = ()
    render_settings: RenderSettingsNode = None
    settings: RenderSettingsNode = None  # Alias for render_settings
    def __post_init__(self):
        # Handle 'settings' alias for 'render_settings'
        if self.settings is not None and self.render_settings is None:
            object.__setattr__(self, 'render_settings', self.settings)
        if self.camera is None:
            object.__setattr__(self, 'camera', CameraNode(
                origin=Vec3Node(0.0, 0.0, 5.0),
                look_at=Vec3Node(0.0, 0.0, 0.0),
                up=Vec3Node(0.0, 1.0, 0.0),
                fov=FloatNode(60.0),
                aspect_ratio=FloatNode(16.0/9.0),
            ))
        if self.render_settings is None:
            object.__setattr__(self, 'render_settings', RenderSettingsNode())
        # Keep settings in sync with render_settings for backwards compatibility
        if self.settings is None:
            object.__setattr__(self, 'settings', self.render_settings)
    def children(self):
        result = [self.scene_graph]
        if self.camera:
            result.append(self.camera)
        result.extend(self.lights)
        result.extend(self.materials)
        if self.render_settings:
            result.append(self.render_settings)
        return tuple(result)
    def label(self):
        name = f" {self.name!r}" if self.name else ""
        return f"FullScene{name}({len(self.lights)} lights, {len(self.materials)} materials)"


SDF_PRIMITIVE_TYPE_MAP = {
    SphereNode: "sdSphere",
    BoxNode: "sdBox",
    TorusNode: "sdTorus",
    CylinderNode: "sdCylinder",
    ConeNode: "sdCone",
    PlaneNode: "sdPlane",
    CapsuleNode: "sdCapsule",
    EllipsoidNode: "sdEllipsoid",
    BoxFrameNode: "sdBoxFrame",
    RoundedBoxNode: "sdRoundedBox",
    OctahedronNode: "sdOctahedron",
    PyramidNode: "sdPyramid",
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
