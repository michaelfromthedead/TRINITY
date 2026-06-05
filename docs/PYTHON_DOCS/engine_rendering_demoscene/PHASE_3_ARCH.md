# PHASE 3 ARCHITECTURE: WGSL Code Generation

## Overview

Phase 3 implements the WGSL code emitter that transforms the AST into valid GPU shader source code.

## Architectural Decisions

### AD-3.1: WgslCodeGen Walker Pattern

**Decision**: WgslCodeGen is a full AST walker that visits nodes and emits WGSL strings.

**Rationale**:
- Separates emission logic from representation
- Different output targets possible (future: GLSL, HLSL)
- Visitor pattern enables clean dispatch per node type

### AD-3.2: SDF Function Templates

**Decision**: Use string templates for SDF primitive functions following Inigo Quilez conventions.

**Templates**:
- `sdSphere(p, r)`: `length(p) - r`
- `sdBox(p, b)`: `length(max(abs(p) - b, 0.0))`
- `sdTorus(p, t)`: Correct 2D distance formula
- `sdCylinder(p, h, r)`: Capped cylinder
- `sdCone(p, c)`: Angle-based cone
- `sdPlane(p, n, h)`: Plane with normal and offset
- `sdCapsule(p, a, b, r)`: Line segment with radius

**Rationale**:
- Inigo Quilez formulas are well-tested and widely used
- Templates ensure consistent formatting
- Easy to update or optimize individual primitives

### AD-3.3: Domain Operation Chain Emission

**Decision**: Domain operations are emitted as function chains transforming position.

**Pipeline Flow**:
```wgsl
var p = world_pos;
p = domain_repeat(p, cell_size);
p = domain_twist(p, twist_amount);
// ... final SDF evaluation
```

**Rationale**:
- Position transformation before SDF evaluation
- Clear order of operations
- Each domain op is a pure function

### AD-3.4: Distance Compensation for Non-Isometric Operations

**Decision**: Emit compensation factor functions for KIFS and Stretch operations.

**Affected Operations**:
- KifsNode: Kaleidoscopic folding changes distance metric
- StretchNode: Non-uniform scaling distorts distances

**Compensation Implementation**:
```wgsl
fn domain_kifs_compensation(folds: f32) -> f32 {
    // Compute worst-case distance scaling factor
    let angle = 6.283185307179586 / max(abs(folds), 1.0);
    let half_angle = angle * 0.5;
    let per_fold = cos(half_angle);
    // Accumulate per-fold compression
    var comp: f32 = 1.0;
    for (var i = 0u; i < u32(safe_folds); i++) {
        comp *= per_fold;
    }
    return max(comp, 1e-8);
}
```

**Rationale**:
- Sphere tracing assumes Lipschitz-1 distance fields
- Non-isometric ops violate this, causing artifacts
- Compensation factor scales step size appropriately

### AD-3.5: PBR Material System

**Decision**: Emit Material struct and scene_material() switch function.

**Structure**:
```wgsl
struct Material {
    albedo: vec3<f32>,
    roughness: f32,
    metallic: f32,
    emissive: vec3<f32>,
    ambient_occlusion: f32,
}

fn scene_material(id: u32) -> Material {
    switch id {
        case 0u: return Material(...);
        case 1u: return Material(...);
        default: return Material(...);  // Default material
    }
}
```

**Rationale**:
- PBR aligns with modern rendering standards
- Switch-based lookup matches material IDs from scene
- Default material prevents undefined behavior

### AD-3.6: Scene Entry Point Generation

**Decision**: Emit `sd_scene(p: vec3<f32>) -> vec2<f32>` as the main scene function.

**Return Value**:
- x: Distance to nearest surface
- y: Material ID (as float, cast to u32 for lookup)

**Implementation**:
```wgsl
fn sd_scene(p: vec3<f32>) -> vec2<f32> {
    var pos = p;
    // Apply domain pipeline
    pos = domain_op1(pos, ...);
    pos = domain_op2(pos, ...);
    // Evaluate primitives
    var d = sdPrimitive1(pos, ...);
    var mat_id = 0.0;
    // CSG combinations
    ...
    return vec2<f32>(d * compensation, mat_id);
}
```

**Rationale**:
- Standard entry point for sphere tracing loop
- vec2 allows carrying material info alongside distance
- Compensation applied to final distance

### AD-3.7: Pipeline Expression Builder

**Decision**: Build domain transformation as composable expression pipeline.

**Rationale**:
- Each domain op can be inlined or function call
- Order matches SceneGraph.pipeline tuple
- Enables optimization (e.g., combining adjacent transforms)

## Dependencies

- Phase 1 AST nodes (all node types)
- Phase 2 builder (for test scene construction)

## Interfaces

### Input
- SceneGraph root node from Phase 1

### Output
- Complete WGSL source string
- Can be compiled by wgpu/dawn shader compiler
