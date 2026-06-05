# PHASE 2 ARCH: Python SDF DSL Compiler

## Overview

Phase 2 implements a Python DSL compiler that translates a high-level scene description (primitives, domain operations, materials) into compilable WGSL source code. The pipeline is:

```
Python Scene Description
    |
    v
AstBuilder (ast_builder.py)          T-DEMO-2.1, 2.2
    - dict marker format
    - lambda disassembly
    - DSL object introspection
    |
    v
SceneGraph AST (ast_nodes.py)        T-DEMO-2.1
    - ExprNode hierarchy
    - DomainOpNode chain
    - SdfPrimitiveNode list
    - MaterialNode list
    |
    v
WgslCodeGen (wgsl_codegen.py)        T-DEMO-2.3, 2.5, 2.6
    - SDF primitive function emission
    - Domain op pipeline expression
    - Compensation function emission
    - Material struct + scene_material()
    - Scene entry point sd_scene__{name}()
    |
    v
WGSL Source Code --> embedded in Rust binary (Phase 5 target)
```

## Module: ast_nodes.py

Defines the AST node hierarchy for SDF scenes:

```
ExprNode (base)
    |
    +-- FloatNode(value)
    +-- Vec3Node(x, y, z)
    +-- PositionNode()              -- the 'p' parameter
    +-- CompensationNode(kind, param)
    |
    +-- DomainOpNode(input)         -- abstract domain operation
    |       +-- RepeatNode(cell_size)
    |       +-- CellIdNode(cell_size)
    |       +-- MirrorNode(axis)
    |       +-- KifsNode(folds)
    |       +-- TwistNode(rate)
    |       +-- BendNode(radius)
    |       +-- StretchNode(stretch, axis)
    |
    +-- SdfPrimitiveNode(position, material_id)  -- abstract primitive
    |       +-- SphereNode(radius)
    |       +-- BoxNode(size)
    |       +-- TorusNode(major_radius, minor_radius)
    |       +-- CylinderNode(height, radius)
    |       +-- ConeNode(height, radius_top, radius_bottom)
    |       +-- PlaneNode(normal, distance)
    |       +-- CapsuleNode(endpoint_a, endpoint_b, radius)
    |
    +-- CombineNode(kind, left, right) -- abstract combinator
    |       +-- UnionNode
    |       +-- IntersectionNode
    |       +-- SubtractionNode
    |
    +-- MaterialNode(material_id, albedo, roughness, metallic, emissive, ambient_occlusion)
    +-- SceneGraph(primitives, pipeline, materials, name)
```

Missing AST nodes: EllipsoidNode, BoxFrameNode, RoundedBoxNode, OctahedronNode, PyramidNode, Smooth*Node, DisplacementNode, CameraNode, LightNode, RenderSettingsNode.

## Module: ast_builder.py

Three input formats supported:

### 1. Dict marker format
```python
scene = {
    "type": "sphere", "radius": 1.5,
    "material": {"material_id": 0, "albedo": (0.9, 0.1, 0.1), ...}
}
```
Dispatch table `_MARKER_DISPATCH` supports: sphere, box, torus, cylinder, cone, plane, capsule, repeat, cell_id, mirror, kifs, twist, bend, stretch, material.

### 2. Lambda composition format
```python
def my_scene(p):
    return sdSphere(sdTorus(domain_twist(p, rate=2.0), major=3.0, minor=1.0), r=1.0)
graph = walk_composition(my_scene)
```
Uses AST disassembly via `inspect.getsource()` + `ast.parse()` to extract the call chain. Supports `_COMPOSITE_DISPATCH` which merges domain ops and primitive dispatches.

### 3. DSL object format
```python
class MySphere:
    _node_type = "sphere"
    radius = 1.0
```
Introspects `_node_type` attribute and walks `__dict__` for parameters.

### Pipeline/Scene building
The `_build_scene()` method assembles a `SceneGraph` from dicts with "pipeline", "primitives", "materials" arrays. The `walk_composition()` function auto-separates domain ops from primitives based on node type.

## Module: wgsl_codegen.py

### SDF Function Emission (T-DEMO-2.3)
Generates WGSL `fn sdSphere`, `fn sdBox`, etc. from the `_PRIMITIVE_TEMPLATES` dict. Each primitive type maps to a `(fn_name, wgsl_source_string)` pair. The codegen deduplicates (only emits one copy of each function even if used multiple times).

Template strings for 6 primitives: sphere, box, torus, cylinder, cone, plane, capsule.
**Missing**: ellipsoid, box_frame, rounded_box, octahedron, pyramid.

### Domain Operation Emission (T-DEMO-2.5)
Builds a pipeline expression chain by reversing the pipeline and wrapping each op:
```wgsl
let p_d = domain_mirror_x(domain_twist(domain_repeat(p, vec3<f32>(2,2,2)), 1.0));
```
Appends compensation functions for KIFS and Stretch (only if those ops appear in pipeline).
Compensation expression multiplies all compensation factors:
```wgsl
let comp = domain_kifs_compensation(6.0) * domain_stretch_compensation(2.0);
```

### Material Emission (T-DEMO-2.6)
Generates:
```wgsl
struct Material { albedo: vec3<f32>, roughness: f32, metallic: f32, emissive: f32, ambient_occlusion: f32 };
fn scene_material(id: i32) -> Material { switch id {
    case 0: { return Material(vec3<f32>(0.8, 0.2, 0.2), 0.5, 0.0, 0.0, 1.0); }
    ...
    default: { return Material(vec3<f32>(0.8, 0.2, 0.2), 0.5, 0.0, 0.0, 1.0); }
}}
```

### Scene Entry Point (T-DEMO-2.7 -- partial)
Generates `sd_scene__{name}(p: vec3<f32>) -> vec2<f32>` that:
1. Applies domain transformations to position
2. Evaluates each primitive as `vec2<f32>(distance, material_id)`
3. For multiple primitives, chains `select()` calls to propagate nearest material_id
4. Applies distance compensation division
5. Returns `vec2<f32>(result.x / comp, result.y)`

**Missing**: Camera/light/render settings codegen. The scene entry point only handles SDF evaluation, not the full ray marching pipeline.

## What's Missing in Phase 2

| Task | Description | Status |
|------|-------------|--------|
| T-DEMO-2.3 | Primitive codegen | Partial -- missing ellipsoid, box_frame, rounded_box from template map |
| T-DEMO-2.4 | Combinator codegen | NOT IMPLEMENTED -- no Union/Intersection/Subtraction/Smooth* WGSL generation |
| T-DEMO-2.7 | Scene codegen | Partial -- sd_scene() generated but no camera/light/render settings |
| T-DEMO-2.8 | Constant folding | NOT IMPLEMENTED |
| T-DEMO-2.9 | Dead code elimination | NOT IMPLEMENTED |
| T-DEMO-2.10 | CSE | NOT IMPLEMENTED |
| T-DEMO-2.11 | Domain repetition flattening | NOT IMPLEMENTED |
| T-DEMO-2.12 | Material merging | NOT IMPLEMENTED |
| T-DEMO-2.13 | Cached compilation | NOT IMPLEMENTED |
| T-DEMO-2.14 | Error reporting | NOT IMPLEMENTED |
