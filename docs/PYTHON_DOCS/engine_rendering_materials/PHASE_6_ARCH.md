# PHASE 6 ARCHITECTURE: Shader Compiler Infrastructure

## Overview

Phase 6 focuses on the shader compilation pipeline, including permutation management, PSO caching, and hot-reload. This phase addresses the placeholder `_compile_internal()` method.

## Components

### ShaderSource
- Load from file or string
- Language auto-detection (GLSL, HLSL, WGSL by extension/content)
- Preprocessor macro support
- Include resolution

### ShaderStage
- Enum: vertex, fragment, compute, geometry, tessellation, mesh, raytracing
- Determines compilation target
- Affects available built-ins

### ShaderLanguage
- Enum: HLSL, GLSL, Metal, SPIRV, WGSL
- Source language identification
- Target language for cross-compilation

### ShaderPermutation
- Feature combination definition
- Conflict detection (mutually exclusive features)
- Bitfield encoding for fast comparison

### PermutationKey
- Frozenset-based variant selection
- Hashable for cache lookup
- Sorted canonical form

### CompiledShader
- Bytecode storage (SPIR-V)
- Reflection data (uniforms, samplers, vertex inputs)
- Compilation timing metrics

### PSOCache
- LRU cache for compiled shaders
- Hit/miss statistics
- Memory budget management

### HotReloadWatcher
- File change polling
- Shader invalidation on source change
- Callback notification system

## Architecture Decisions

### AD-18: SPIR-V as IR
**Decision**: Compile to SPIR-V, then cross-compile to target.
**Rationale**: Industry standard IR, wide tooling support.
**Consequences**: Requires glslang or dxc for initial compilation.

### AD-19: LRU Cache Strategy
**Decision**: Use LRU eviction for PSO cache.
**Rationale**: Predictable memory usage, retains frequently used shaders.
**Consequences**: Cold shaders may be recompiled.

### AD-20: Polling for Hot-Reload
**Decision**: Use file polling, not filesystem events.
**Rationale**: Cross-platform compatibility, simpler implementation.
**Consequences**: Small latency on changes, CPU overhead.

### AD-21: Permutation Conflict Detection
**Decision**: Define mutual exclusion rules for features.
**Rationale**: Prevents invalid shader combinations.
**Consequences**: Feature matrix must be explicitly defined.

## Compilation Pipeline

```
ShaderSource (GLSL/HLSL)
    |
    v
[Preprocessor]
    |  - Macro expansion
    |  - Include resolution
    v
[Validation]
    |  - Syntax check
    |  - Type check
    v
[Compilation] (glslang/dxc)
    |
    v
SPIR-V bytecode
    |
    v
[Reflection]
    |  - Extract uniforms
    |  - Extract samplers
    |  - Extract vertex inputs
    v
CompiledShader
    |
    v
[PSO Cache]
```

## Placeholder Location

```python
# shader_compiler.py:869-881
def _compile_internal(self, source: ShaderSource, optimize: bool) -> bytes:
    """Internal compilation implementation.
    This is a placeholder that should be overridden or extended
    with actual compiler integration (glslang, dxc, etc.).
    """
    # Placeholder: return hash of source as "bytecode"
    code = source.get_preprocessed_code()
    return hashlib.sha256(code.encode()).digest()
```

## Integration Options

| Compiler | Source | Target | Notes |
|----------|--------|--------|-------|
| glslang | GLSL | SPIR-V | Vulkan-focused, official Khronos |
| dxc | HLSL | SPIR-V/DXIL | DirectX shader compiler |
| naga | WGSL/SPIR-V | Multiple | Rust-native, WebGPU focused |
| spirv-cross | SPIR-V | GLSL/HLSL/MSL | Cross-compilation |

## Validation Strategy

1. Test ShaderSource loading and preprocessing
2. Test permutation generation and conflict detection
3. Test PSO cache hit/miss behavior
4. Integration test with real compiler (glslang)
