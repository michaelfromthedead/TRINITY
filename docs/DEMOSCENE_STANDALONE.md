# TRINITY Demoscene: Standalone Requirements

This document specifies the standalone requirements for the TRINITY demoscene renderer, ensuring it can run without external dependencies on any compatible system.

## Overview

The TRINITY demoscene module provides two rendering modes:

1. **Standard Mode**: Full-featured demoscene renderer with hot-reloading, Python DSL integration, and extensive shader libraries.

2. **4K Mode**: Extreme minimization for demoscene competitions targeting sub-4KB binary size.

## Standalone Requirements

### No Asset Files Required

The demoscene renderer embeds all assets at compile time:

| Asset Type | Standard Mode | 4K Mode |
|------------|---------------|---------|
| Shaders | `include_str!` | Inline literal |
| Textures | None (procedural) | None |
| Models | None (SDF) | None |
| Audio | None | None |
| Config | Constants | Constants |

**Verification:**
```bash
./scripts/verify_standalone.sh
# Check: "No texture file loading" - PASS
# Check: "No model file loading" - PASS
# Check: "Shader embedded inline" - PASS
```

### No Python Runtime

The core rendering functionality requires no Python:

| Component | Python Dependency |
|-----------|-------------------|
| `demoscene/minimal.rs` | None |
| `demoscene/bootstrap.rs` | None |
| `demoscene_render.rs` | None |
| `demoscene/mod.rs` | Optional (DSL compiler) |

The Python DSL compiler (`scripts/compile_demo.py`) is only used for:
- Scene definition in `scenes/demo.py`
- Build-time shader generation
- Development tooling

**Runtime Independence:**
```rust
// This works without Python
let renderer = MinimalRenderer::new(&device, 800, 600);
renderer.render(&device, &queue, 0.0);
```

**Build-time Optional:**
```toml
[features]
# Only enable if Python DSL is needed
build-compiled-shaders = []
```

### No Network Dependencies

The demoscene renderer has zero network requirements:

| Operation | Network Required |
|-----------|------------------|
| Initialization | No |
| Shader loading | No (embedded) |
| Rendering | No |
| GPU backend | No |

**Verification:**
```bash
# Check Cargo.toml for network crates
grep -E "reqwest|hyper|tokio.*net" crates/renderer-backend/Cargo.toml
# Should return empty
```

### GPU Backend Requirements

The demoscene renderer uses wgpu, which supports multiple backends:

| Platform | Primary Backend | Fallback |
|----------|-----------------|----------|
| Linux | Vulkan | OpenGL |
| macOS | Metal | N/A |
| Windows | DirectX 12 | DirectX 11, Vulkan |
| Web | WebGPU | WebGL (limited) |
| Android | Vulkan | OpenGL ES |
| iOS | Metal | N/A |

**Minimum Requirements:**
- Vulkan 1.0 OR Metal 2.0 OR DirectX 12
- GPU with compute shader support
- 128MB VRAM (for 1080p)
- 512MB VRAM (for 4K)

**Fallback Chain:**
```rust
// wgpu automatically selects the best available backend
let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
    backends: wgpu::Backends::all(), // Vulkan | Metal | DX12 | GL
    ..Default::default()
});
```

## 4K Mode Specifics

The 4K mode (`demoscene/minimal.rs`) is designed for extreme size optimization:

### Size Budget

| Component | Uncompressed | LZ4 (~60%) | Target |
|-----------|--------------|------------|--------|
| Shader | ~1.5KB | ~600B | < 1KB |
| Rust code | ~5KB | ~2KB | < 2KB |
| Total | ~6.5KB | ~2.6KB | < 4KB |

**Note:** True 4KB binary requires custom GPU backend (OpenGL 1.1 + assembly). The wgpu-based implementation is ~50KB due to wgpu runtime overhead.

### Optimization Techniques

1. **Single-letter variables in shader:**
   ```wgsl
   struct U{time:f32,rx:f32,ry:f32,_p:f32}
   fn sd(p:vec3<f32>)->f32{...}
   ```

2. **No whitespace/comments in shader:**
   ```wgsl
   fn n(p:vec3<f32>)->vec3<f32>{let e=vec2(.001,0.);...}
   ```

3. **Minimal uniform struct (16 bytes):**
   ```rust
   #[repr(C)]
   pub struct MinimalUniforms {
       pub time: f32,    // 4 bytes
       pub rx: f32,      // 4 bytes
       pub ry: f32,      // 4 bytes
       pub _p: f32,      // 4 bytes padding
   }
   ```

4. **Inline everything:**
   ```rust
   #[inline(always)]
   pub fn render(&mut self, ...) { ... }
   ```

## Verification Script

Run the standalone verification:

```bash
# Full verification
./scripts/verify_standalone.sh --verbose

# Skip GPU checks (for CI environments)
./scripts/verify_standalone.sh --skip-gpu
```

### Checks Performed

1. **Asset Embedding**
   - Shader inline string present
   - No texture loading code
   - No model loading code
   - No config file loading
   - No audio loading

2. **Python Independence**
   - No PyO3 imports
   - No Python subprocess calls
   - PyO3 is optional feature
   - build.rs has no Python deps

3. **Network Independence**
   - No HTTP client imports
   - No socket usage
   - No URL parsing
   - No network deps in Cargo.toml

4. **GPU Backend**
   - wgpu dependency present
   - Platform-specific backend available
   - OpenGL fallback available

5. **Binary Self-Containment**
   - Shader is compact
   - Uniforms are minimal
   - Has embedded tests

## Testing on Clean System

To verify standalone operation:

```bash
# 1. Build the demo binary
cargo build --release -p renderer-backend

# 2. Copy to a clean directory
mkdir /tmp/demo-test
cp target/release/librenderer_backend.so /tmp/demo-test/

# 3. Run tests (no dev tools needed)
cd /tmp/demo-test
LD_LIBRARY_PATH=. ./librenderer_backend.so test

# 4. Verify no missing dependencies
ldd ./librenderer_backend.so | grep "not found"
# Should return empty
```

## Dependencies Summary

### Required (Always)
| Dependency | Purpose | Size Impact |
|------------|---------|-------------|
| wgpu | GPU abstraction | ~50KB |
| bytemuck | Pod/Zeroable | ~1KB |
| pollster | Async block | ~1KB |

### Optional (Feature-gated)
| Dependency | Feature | Purpose |
|------------|---------|---------|
| pyo3 | `pyo3` | Python bindings |
| naga | dev-only | Shader validation |

### Build-time Only
| Dependency | Purpose |
|------------|---------|
| Python 3.13 | DSL compiler (optional) |

## Compatibility Matrix

| Platform | Rust | GPU | Status |
|----------|------|-----|--------|
| Linux x86_64 | 1.75+ | Vulkan/GL | Tested |
| Linux ARM64 | 1.75+ | Vulkan/GL | Tested |
| macOS x86_64 | 1.75+ | Metal | Tested |
| macOS ARM64 | 1.75+ | Metal | Tested |
| Windows x64 | 1.75+ | DX12/Vulkan | Tested |
| WASM32 | 1.75+ | WebGPU | Experimental |

## Troubleshooting

### No GPU Adapter Found

```rust
// Error: BootstrapError::NoAdapter
```

**Solutions:**
1. Install GPU drivers (Vulkan, Metal, or DX12)
2. Check GPU is not disabled in BIOS
3. Use software renderer: `LIBGL_ALWAYS_SOFTWARE=1`

### Shader Compilation Failed

```rust
// Error: ShaderModuleDescriptor validation failed
```

**Solutions:**
1. Update wgpu to latest version
2. Check shader syntax with naga validation
3. Verify workgroup size is supported

### Out of Memory

```rust
// Error: wgpu::DeviceError::OutOfMemory
```

**Solutions:**
1. Reduce render resolution
2. Use lower-end limits: `wgpu::Limits::downlevel_defaults()`
3. Free GPU memory from other applications
