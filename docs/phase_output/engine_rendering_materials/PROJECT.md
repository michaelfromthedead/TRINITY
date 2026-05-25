# PROJECT: engine/rendering/materials

## Scope

The materials system provides a complete PBR material pipeline for the TRINITY engine, including:
- Core material templates, instances, functions, and layers
- PBR metallic-roughness and specular-glossiness workflows
- Node-based material graph system with GLSL code generation
- Shader compilation infrastructure with permutation management
- Advanced shading models (SSS, clearcoat, anisotropy, sheen, iridescence, transmission)

## Current State

**Status: REAL IMPLEMENTATION**

| Component | Lines | Status |
|-----------|-------|--------|
| `__init__.py` | 277 | Complete |
| `material_system.py` | 904 | Complete |
| `shader_compiler.py` | 902 | Partial (placeholder bytecode) |
| `material_graph.py` | 1281 | Complete |
| `material_functions.py` | 1091 | Complete |
| `pbr_model.py` | 698 | Complete |
| `advanced_models.py` | 636 | Complete |
| `constants.py` | ~100 | Complete |

**Total: ~5,889 lines of production code**

## Goals

1. Complete the shader compiler integration (glslang/dxc/naga)
2. Add WGSL code generation pathway (currently GLSL only)
3. Integrate material system with Rust renderer backend
4. Validate material graph outputs against reference shaders
5. Connect PBR materials to GPU texture table and draw system

## Constraints

- Shader compiler has placeholder `_compile_internal()` returning hash instead of bytecode
- Material graph generates GLSL, not WGSL (despite enum including WGSL)
- Advanced models (SSS, transmission) may need compute shader support
- Hot-reload system depends on file polling, not filesystem events

## Acceptance Criteria

1. ShaderCompiler produces valid SPIR-V bytecode from GLSL sources
2. MaterialGraph can optionally emit WGSL for WebGPU targets
3. PBRMaterial dirty flags correctly trigger GPU re-upload
4. Material instances properly inherit from templates with override semantics
5. All 20+ material functions produce valid, tested GLSL output
6. Advanced shading models integrate with forward/deferred rendering paths

## Dependencies

- `engine/rendering/gpu_driven` - Draw indirect, texture table
- `engine/rendering/frame_graph` - Render pass scheduling
- `crates/renderer-backend` - Rust GPU abstraction
- External: glslang, spirv-cross, or naga for shader compilation

## Risk Areas

1. Shader compilation requires native toolchain integration
2. GLSL-to-WGSL conversion may have semantic differences
3. Advanced models (SSS, transmission) are compute-intensive
4. Material graph topological sort assumes acyclic; cycles cause failure
