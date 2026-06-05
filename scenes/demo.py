"""
Demo scene definition using the TRINITY SDF DSL.

This file defines a demonstration scene that is compiled to WGSL
at build time via the build.rs -> compile_demo.py pipeline.

The scene showcases:
  - Multiple SDF primitives (sphere, box, torus)
  - Domain operations (mirror, twist)
  - PBR materials
  - Point and directional lights
  - Custom camera setup

Usage:
    This file is imported by scripts/compile_demo.py and the
    `SCENE` variable is compiled to WGSL.
"""

from engine.rendering.demoscene.ast_nodes import (
    # Primitives
    SphereNode, BoxNode, TorusNode, CylinderNode,
    # Nodes
    PositionNode, FloatNode, Vec3Node,
    # Domain ops
    MirrorNode, TwistNode, RepeatNode, Axis,
    # Scene structure
    SceneGraph, MaterialNode, LightNode, CameraNode,
    RenderSettingsNode, FullSceneNode, LightType,
)


# =============================================================================
# Scene Definition
# =============================================================================

# Position node (shared reference)
p = PositionNode()

# Define primitives
sphere = SphereNode(
    position=p,
    radius=FloatNode(1.0),
)

box = BoxNode(
    position=p,
    size=Vec3Node(0.5, 0.8, 0.5),
)

torus = TorusNode(
    position=p,
    major_radius=FloatNode(1.5),
    minor_radius=FloatNode(0.3),
)

# Domain operations pipeline
pipeline = (
    MirrorNode(input=p, axis=Axis.X),
    TwistNode(input=p, rate=FloatNode(0.5)),
)

# Build scene graph
scene_graph = SceneGraph(
    primitives=(sphere, box, torus),
    pipeline=pipeline,
    name="demo",
)

# Materials
materials = (
    MaterialNode(
        material_id=0,
        albedo=Vec3Node(0.8, 0.2, 0.2),  # Red sphere
        roughness=FloatNode(0.3),
        metallic=FloatNode(0.0),
    ),
    MaterialNode(
        material_id=1,
        albedo=Vec3Node(0.2, 0.8, 0.2),  # Green box
        roughness=FloatNode(0.5),
        metallic=FloatNode(0.1),
    ),
    MaterialNode(
        material_id=2,
        albedo=Vec3Node(0.2, 0.2, 0.8),  # Blue torus
        roughness=FloatNode(0.2),
        metallic=FloatNode(0.8),
    ),
)

# Camera
camera = CameraNode(
    origin=Vec3Node(0.0, 2.0, 6.0),
    look_at=Vec3Node(0.0, 0.0, 0.0),
    up=Vec3Node(0.0, 1.0, 0.0),
    fov=FloatNode(60.0),
    aspect_ratio=FloatNode(16.0 / 9.0),
)

# Lights
lights = (
    LightNode(
        position=Vec3Node(5.0, 5.0, 5.0),
        color=Vec3Node(1.0, 0.98, 0.95),
        intensity=FloatNode(2.0),
        light_type=LightType.POINT,
    ),
    LightNode(
        position=Vec3Node(0.0, 10.0, 0.0),
        direction=Vec3Node(0.0, -1.0, 0.0),
        color=Vec3Node(0.5, 0.6, 0.8),
        intensity=FloatNode(1.0),
        light_type=LightType.DIRECTIONAL,
    ),
)

# Render settings
settings = RenderSettingsNode(
    width=1920,
    height=1080,
    max_steps=256,
    max_distance=100.0,
    epsilon=0.0001,
    workgroup_size_x=8,
    workgroup_size_y=8,
)

# Full scene export (this is what compile_demo.py looks for)
SCENE = FullSceneNode(
    scene_graph=scene_graph,
    materials=materials,
    camera=camera,
    lights=lights,
    settings=settings,
    name="demo",
)
