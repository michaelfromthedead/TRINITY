# CLARIFICATION: engine/rendering/materials

## Philosophical Framing

The materials system embodies a **data-driven, node-graph approach** to shader authoring. Rather than hand-writing monolithic shaders, artists and technical artists compose materials from reusable nodes and functions. The system generates optimized shader code from these graphs, enabling:

1. **Non-destructive iteration** - Change any node, regenerate shader
2. **Permutation management** - Feature toggles without shader explosion
3. **Code reuse** - Material functions compose across materials
4. **Pipeline abstraction** - Same graph, multiple backends (GLSL, WGSL, HLSL)

## Design Rationale

### Why Node Graphs?

Modern PBR workflows require expressing complex surface properties:
- Base color with texture variation
- Roughness with procedural noise
- Normal maps with detail blending
- Emission with animated patterns

Monolithic shaders become unmaintainable. Node graphs decompose complexity into testable units. Each node has a single responsibility: sample a texture, blend two values, apply a function.

### Why GLSL as Intermediate?

The codebase chose GLSL as the primary output format because:
1. Wide tooling support (glslang, spirv-cross)
2. Readable debugging output
3. Cross-compile to SPIR-V, then to any target

The `ShaderLanguage` enum includes WGSL for future WebGPU support, but the GraphCompiler currently targets GLSL.

### Why Material Functions as Library?

The 20+ material functions (Fresnel, normal blending, parallax, noise) are:
- **Embeddable** - Injected into generated shaders as needed
- **Dependency-tracked** - Only include what the graph uses
- **Testable** - Each function has known inputs/outputs

This avoids shader include hell while maintaining reusability.

### Why Dirty Flags?

GPU resources (uniform buffers, textures) are expensive to re-upload. The `DirtyFlags` system tracks:
- `PARAMETERS` - Scalar uniforms changed
- `TEXTURES` - Texture bindings changed
- `SHADER` - Graph recompiled

The renderer queries dirty state and only updates what changed.

### Why Weak References?

`MaterialInstance` holds weak references to its parent `MaterialTemplate`. This enables:
- Templates can be hot-reloaded without dangling instances
- Garbage collection of unused instances
- Safe template invalidation propagation

## What "Partial" Means for ShaderCompiler

The infrastructure is complete:
- Permutation conflict detection
- PSO cache with LRU eviction
- Hot-reload file watching
- Reflection data structures

The placeholder is `_compile_internal()`, which returns a SHA-256 hash instead of bytecode. Integration points are clear; implementation requires native toolchain bindings.

## GLSL vs WGSL Reality

The investigation confirms:
- `GraphCompiler` outputs GLSL syntax (`uniform`, `texture()`, `vec3`)
- The enum includes `ShaderLanguage.WGSL` but no code path uses it
- WGSL support requires a parallel code generator or transpilation step

This is not a bug; it's an intentional deferral. GLSL-to-SPIR-V-to-WGSL is a viable path once the core pipeline is validated.

## Advanced Models Strategy

The advanced shading models (SSS, clearcoat, anisotropy, sheen, iridescence, transmission) are:
- **Dataclass-based** - Parameters are structured, validated
- **Preset-driven** - Skin, wax, jade, milk for SSS; fabric types for sheen
- **Modular** - Can be combined with base PBR

These models add complexity to the lighting pass. Integration requires:
1. Additional uniform blocks in the shading pass
2. Potentially separate passes (SSS blur, transmission)
3. Performance profiling per platform
