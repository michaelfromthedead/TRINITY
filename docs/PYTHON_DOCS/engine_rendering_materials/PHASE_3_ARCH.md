# PHASE 3 ARCHITECTURE: Material Graph and Code Generation

## Overview

Phase 3 covers the node-based material graph system and its GLSL code generation pipeline. This is the core authoring system for procedural and artist-driven materials.

## Components

### MaterialNode (Abstract Base)
- Input/output ports with typed connections
- `generate_code(input_vars, output_var) -> str` method
- Connection validation

### Math Nodes
- Add, Subtract, Multiply, Divide
- Lerp (linear interpolation)
- Clamp, Saturate
- Power, Sqrt, Abs
- Min, Max, Frac, Floor, Ceil

### Texture Nodes
- TextureSampleNode: Sample texture with UV, outputs rgba/rgb/r/g/b/a
- UVNode: Provides UV coordinates

### Utility Nodes
- OneMinus: 1.0 - input
- ComponentMask: Extract xyz/rgb components
- AppendNode: Combine scalars into vectors

### OutputNode
- PBR terminal node
- Inputs: base_color, metallic, roughness, normal, emissive, ao, opacity
- Connects graph to shader output

### MaterialGraph
- DAG container for nodes and connections
- Validation: type checking, cycle detection
- Topological sort for code generation order

### GraphCompiler
- Traverses graph in topological order
- Generates uniform declarations
- Generates sampler declarations
- Emits node code in dependency order

## Architecture Decisions

### AD-7: GLSL as Primary Target
**Decision**: GraphCompiler outputs GLSL syntax.
**Rationale**: Wide tooling support, readable output, cross-compile to SPIR-V.
**Consequences**: WGSL support requires additional code path or transpilation.

### AD-8: Topological Sort for Code Order
**Decision**: Generate code in topological order of node dependencies.
**Rationale**: Ensures variables are defined before use.
**Consequences**: Cycles are illegal; validation must detect them.

### AD-9: Per-Node Code Generation
**Decision**: Each node implements `generate_code` independently.
**Rationale**: Encapsulation, testability, easy extension.
**Consequences**: Code generation is distributed, not centralized.

### AD-10: Type-Safe Connections
**Decision**: Port connections are type-checked at graph build time.
**Rationale**: Catch errors early, prevent invalid shader generation.
**Consequences**: Dynamic type coercion requires explicit conversion nodes.

## Code Generation Flow

```
MaterialGraph
    |
    v
[Validation]
    |  - Type checking
    |  - Cycle detection
    v
[Topological Sort]
    |
    v
[GraphCompiler.compile()]
    |
    +--> Uniform declarations (u_paramName)
    +--> Sampler declarations (tex_textureName)
    +--> Node code in order
    +--> Output assignment
    |
    v
GLSL source string
```

## Generated GLSL Structure

```glsl
// Uniforms
uniform float u_roughnessScale;
uniform vec3 u_tintColor;

// Samplers
uniform sampler2D tex_baseColor;
uniform sampler2D tex_normal;

// Main shader code
void materialMain() {
    // Node outputs in topological order
    vec4 node_1_rgba = texture(tex_baseColor, v_uv);
    vec3 node_1_rgb = node_1_rgba.rgb;
    vec3 node_2 = node_1_rgb * u_tintColor;
    // ... more nodes
    
    // Output assignment
    out_baseColor = node_N;
    out_roughness = u_roughnessScale;
}
```

## Integration Points

- `engine/rendering/materials/material_functions.py` - Embeddable function library
- `engine/rendering/materials/shader_compiler.py` - GLSL to bytecode
- `engine/rendering/frame_graph` - Pass-specific shader variants

## Validation Strategy

1. Unit test each node type's code generation
2. Test graph validation (cycles, type mismatches)
3. Compile generated GLSL with glslang
4. Compare output against reference shaders
