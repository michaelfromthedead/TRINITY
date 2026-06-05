# PHASE_1_ARCH.md — Inline Ray Queries for Shadow Rays

> **Phase**: 1 of 3
> **Status**: Foundation [x], Implementation [-]
> **Tasks**: 15 (3 [~], 12 [-])
> **Gaps Covered**: S10-G2, S10-G3, S10-G6, S10-G8
> **Platform Gate**: None (wgpu `ray_query` + `acceleration_structure` available now)

---

## 1. Architecture Overview

Phase 1 implements inline ray queries for shadow rays using wgpu's `ray_query` and `acceleration_structure` features. This is the "entry level" ray tracing -- no separate raygen/hit/miss shaders needed; all ray traversal logic is inline in a compute shader.

```
Python Layer                      Rust/wgpu Layer
┌─────────────────┐              ┌──────────────────────┐
│  RTCapability    │──detect──▶  │  wgpu::Device        │
│  Detection       │              │  .has_feature(...)   │
└────────┬────────┘              └──────────────────────┘
         │
         ▼
┌─────────────────┐              ┌──────────────────────┐
│  BLASPool        │──build──▶   │  BLAS Builder        │
│  (ref count)     │              │  (wgpu::BLAS)        │
└────────┬────────┘              └──────────────────────┘
         │
         ▼
┌─────────────────┐              ┌──────────────────────┐
│  Instance Buffer │──upload──▶  │  TLAS Builder        │
│  Manager         │              │  (wgpu::TLAS)        │
└────────┬────────┘              └──────────────────────┘
         │
         ▼
┌─────────────────┐              ┌──────────────────────┐
│  RTShadows       │──dispatch──▶│  Ray Query Compute   │
│  (Python)        │              │  Shader (WGSL)       │
└────────┬────────┘              └──────────────────────┘
         │
         ▼
┌─────────────────┐              ┌──────────────────────┐
│  Denoiser        │──denoise──▶ │  A Trous WGSL        │
│  Dispatch        │              │  Spatial Denoiser    │
└────────┬────────┘              └──────────────────────┘
         │
         ▼
┌─────────────────┐
│  Fallback Chain  │
│  RT -> CSM+PCSS  │
└─────────────────┘
```

## 2. Component Details

### 2.1 RTCapability Detection [~] (T-RT-P1.4)

**Current state**: `FeatureSupport.ray_tracing` bool exists in `device.py`. Simple discrete vs integrated check.

**Required expansion**: 3-level `RTCapability` enum:
```python
class RTCapability(Enum):
    NONE = 0             # No RT support
    RAY_QUERY_ONLY = 1   # Inline ray queries only (Phase 1)
    FULL = 2             # Full RT pipeline (Phase 2+)
```

**Files to create**: `engine/platform/rhi/capability.py` or extend `device.py`.

### 2.2 BLAS/TLAS Management [~] (T-RT-P1.1)

**Current state**: `BLASDesc`, `TLASDesc`, `AccelerationStructure` ABC, `NullAccelerationStructure`, `BuildFlags` exist.

**Required additions**:
- `BLASManager`: `build_static()`, `build_dynamic()`, `refit()`, `compact()` methods.
- `TLASManager`: `build_frame()` with per-frame rebuild support.
- `BLASPool`: Mesh-asset-ID-keyed reference counting, batch build/compact queues.

### 2.3 Rust BLAS Backend [-] (T-RT-P1.2)

**Required implementation**:
```rust
// Pseudocode for BLAS creation
fn create_blas(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    vertex_buffer: &wgpu::Buffer,
    desc: &BLASDesc,
) -> Result<BlasHandle, RtError> {
    let blas_desc = wgpu::BlasDescriptor {
        vertex: ...,
        index: ...,
        flags: ...,
        ..Default::default()
    };
    let build_sizes = device.get_blas_build_sizes(&blas_desc);
    let scratch = device.create_buffer(&wgpu::BufferDescriptor {
        size: build_sizes.build_scrub_size,
        usage: wgpu::BufferUsages::RAY_TRACING_ACCELERATION_STRUCTURE_SCRATCH,
        ..Default::default()
    });
    let blas = device.create_blas(&blas_desc);
    // Build, compaction query, compacted copy
}
```

### 2.4 Rust TLAS Backend [-] (T-RT-P1.3)

**Required implementation**:
```rust
fn create_tlas(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    instances: &[wgpu::RayTracingInstance],
    flags: wgpu::BlasFlags,
) -> Result<TlasHandle, RtError> {
    let instance_buffer = device.create_buffer_init(...);
    let tlas_desc = wgpu::TlasDescriptor { ... };
    let tlas = device.create_tlas(&tlas_desc);
    // Build acceleration structure from instances
}
```

### 2.5 RT Shadow Ray Query Shader [-] (T-RT-P1.5)

**Required WGSL skeleton**:
```wgsl
@group(0) @binding(0) var<storage, read> gbuffer_depth: texture_2d<f32>;
@group(0) @binding(1) var<storage, read> gbuffer_normal: texture_2d<f32>;
@group(0) @binding(2) var<uniform> camera: CameraUniform;
@group(0) @binding(3) var<storage> tlas: acceleration_structure;
@group(1) @binding(0) var<storage, read_write> shadow_mask: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let depth = textureLoad(gbuffer_depth, id.xy, 0).r;
    let normal = textureLoad(gbuffer_normal, id.xy, 0).xyz * 2.0 - 1.0;
    let world_pos = reconstruct_world_pos(id.xy, depth, camera);
    
    var ray_query: RayQuery;
    ray_query.initialize(world_pos, light_dir, 0.001, 1000.0, tlas);
    
    while (ray_query.proceed()) {
        switch ray_query.get_intersection_type() {
            case RayQueryIntersectionType::TRIANGLE: {
                // Check alpha if needed
                ray_query.terminate();
            }
        }
    }
    
    let occluded = ray_query.get_intersection_type() == RayQueryIntersectionType::TRIANGLE;
    textureStore(shadow_mask, id.xy, vec4(f32(!occluded), 0, 0, 0));
}
```

### 2.6 A Trous Spatial Denoiser [-] (T-RT-P1.9)

**Required WGSL skeleton**: 3-4 iterative passes with step sizes 1, 2, 4, 8. Edge-stopping weights from depth, normal, luminance differences. Separable horizontal + vertical passes.

## 3. File Map

| Task | New Files Required | Existing Files to Modify |
|------|-------------------|-------------------------|
| P1.1 | `engine/platform/rhi/blas_manager.py` | `engine/platform/rhi/raytracing.py` |
| P1.2 | `crates/renderer-backend/src/rt/blas.rs` | - |
| P1.3 | `crates/renderer-backend/src/rt/tlas.rs` | - |
| P1.4 | `engine/platform/rhi/capability.py` | `engine/platform/rhi/device.py` |
| P1.5 | `shaders/rt_shadow.comp.wgsl` | - |
| P1.6 | (inline in P1.5 shader) | - |
| P1.7 | `engine/rendering/rt/shadows.py` | `engine/rendering/__init__.py` |
| P1.8 | `engine/rendering/lighting/fallback_shadows.py` | `engine/rendering/lighting/` |
| P1.9 | `shaders/denoiser_spatial.comp.wgsl` | - |
| P1.10 | `engine/rendering/rt/denoiser.py` | `engine/rendering/__init__.py` |
| P1.11 | `engine/platform/rhi/blas_pool.py` | `engine/platform/rhi/raytracing.py` |
| P1.12 | `engine/rendering/rt/instance_buffer.py` | - |
| P1.13 | `engine/rendering/rt/ray_budget.py` | - |
| P1.14 | - | Engine `mesh_asset.rs` integration |
| P1.15 | - | `engine/rendering/rt/` integration |

## 4. Data Flow (Frame)

```
Frame Start
  │
  ├─ Scene Culling
  │
  ├─ Instance Buffer Update (P1.12)
  │   └─ Gather transforms from scene graph
  │   └─ Build RayTracingInstance array
  │   └─ Upload to ping-pong buffer
  │
  ├─ BLAS Update Queue (P1.15)
  │   ├─ Refit rigid dynamic BLASes
  │   └─ Rebuild skinned mesh BLASes
  │
  ├─ TLAS Build (P1.3)
  │   └─ Build from instance buffer
  │
  ├─ G-Buffer Pass
  │   └─ Depth + normal + albedo
  │
  ├─ RT Shadow Pass (P1.7)
  │   ├─ Bind G-Buffer + TLAS
  │   ├─ Dispatch ray query compute shader
  │   └─ Output shadow factor texture
  │
  ├─ Spatial Denoiser (P1.9, P1.10)
  │   └─ 3-4 A Trous iterations
  │
  └─ Composite / Lighting
```

## 5. Effort Breakdown

| Task | Effort | Type | Priority |
|------|--------|------|----------|
| P1.2 Rust BLAS | 8 | Backend | High (blocks TLAS) |
| P1.3 Rust TLAS | 5 | Backend | High (blocks shadows) |
| P1.5 Shadow shader | 8 | Shader | High |
| P1.1 Manager classes | 5 | Python | High |
| P1.11 BLASPool | 5 | Python | High |
| P1.12 Instance buffer | 5 | Python | High |
| P1.14 Static BLAS on load | 5 | Integration | High |
| P1.7 Shadow dispatch | 5 | Python | High |
| P1.9 Denoiser shader | 8 | Shader | Medium |
| P1.10 Denoiser dispatch | 3 | Python | Medium |
| P1.4 Capability enum | 3 | Python | Medium |
| P1.8 Fallback chain | 8 | Python | Medium |
| P1.13 Ray budget | 3 | Python | Medium |
| P1.6 Any-hit alpha | 5 | Shader | Medium |
| P1.15 Dynamic refit | 5 | Python | Medium |
