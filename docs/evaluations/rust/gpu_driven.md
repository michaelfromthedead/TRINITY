# GPU-Driven Module Evaluation

**Module:** renderer-backend::gpu_driven
**Location:** `/crates/renderer-backend/src/gpu_driven/`
**Lines:** ~5,000
**Quality Grade:** A

---

## Purpose

Bindless GPU resource tables for modern GPU-driven rendering. Enables indirect draw calls and material lookups without rebinding.

---

## File Inventory

| File | Lines | Purpose | Quality |
|------|-------|---------|---------|
| mod.rs | ~100 | Module re-exports | A |
| buffers.rs | 777 | Triple-buffered staging | A |
| mesh_table.rs | ~800 | Bindless mesh registry | A |
| material_table.rs | ~800 | Bindless material slots | A |
| texture_table.rs | ~600 | Texture array management | A |
| mesh_table.wgsl | ~200 | WGSL companion | A |
| material_table.wgsl | ~200 | WGSL companion | A |

---

## BufferRegistry (buffers.rs)

### Triple-Buffered Staging

```rust
pub struct BufferRegistry {
    slots: Vec<BufferSlot>,
    free_list: VecDeque<usize>,
    // Triple-buffering for upload/GPU/readback
}

pub struct BufferSlot {
    pub state: SlotState,
    pub buffer: Option<wgpu::Buffer>,
    pub size: usize,
    pub generation: u32,
}

pub enum SlotState {
    Free,
    Reserved,
    Uploading,
    GpuOwned,
    Readback,
}
```

### State Machine

```
Free → Reserved → Uploading → GpuOwned → Readback → Free
         ↑                                    ↓
         └────────────────────────────────────┘
```

**Features:**
- Slot reuse via free list
- Generation tracking for stale handle detection
- State transition validation
- 13 unit tests covering all transitions

---

## MeshTable (mesh_table.rs)

### Bindless Mesh Registry

```rust
pub struct MeshTable {
    entries: Vec<MeshEntry>,
    free_list: VecDeque<u32>,
    gpu_buffer: Option<wgpu::Buffer>,
}

pub struct MeshEntry {
    pub vertex_offset: u32,
    pub vertex_count: u32,
    pub index_offset: u32,
    pub index_count: u32,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub flags: u32,
}
```

### WGSL Companion

```wgsl
struct MeshTableEntry {
    vertex_offset: u32,
    vertex_count: u32,
    index_offset: u32,
    index_count: u32,
    bounds_min: vec3<f32>,
    bounds_max: vec3<f32>,
    flags: u32,
}

@group(0) @binding(0)
var<storage, read> mesh_table: array<MeshTableEntry>;
```

---

## MaterialTable (material_table.rs)

### Bindless Material Slots

```rust
pub struct MaterialTable {
    entries: Vec<MaterialEntry>,
    free_list: VecDeque<u32>,
    gpu_buffer: Option<wgpu::Buffer>,
}

pub struct MaterialEntry {
    pub albedo_texture: u32,
    pub normal_texture: u32,
    pub metallic_roughness_texture: u32,
    pub occlusion_texture: u32,
    pub emissive_texture: u32,
    pub base_color: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub emissive_factor: [f32; 3],
    pub alpha_cutoff: f32,
    pub flags: u32,
}
```

### WGSL Companion

```wgsl
struct MaterialTableEntry {
    albedo_texture: u32,
    normal_texture: u32,
    metallic_roughness_texture: u32,
    // ... all fields match Rust struct
}

fn get_material(index: u32) -> MaterialTableEntry {
    return material_table[index];
}
```

---

## TextureTable (texture_table.rs)

### Texture Array Management

```rust
pub struct TextureTable {
    textures: Vec<TextureEntry>,
    free_list: VecDeque<u32>,
}

pub struct TextureEntry {
    pub texture: Option<wgpu::TextureView>,
    pub sampler: Option<wgpu::Sampler>,
    pub format: wgpu::TextureFormat,
    pub dimensions: [u32; 2],
}
```

---

## Test Coverage

| Test File | Coverage |
|-----------|----------|
| blackbox_buffer_registry.rs | Slot FSM, acquire/release |
| blackbox_mesh_table.rs | Entry allocation, GPU upload |
| blackbox_material_table.rs | PBR fields, flags |
| blackbox_texture_table.rs | Format handling |
| whitebox_material_table.rs | Internal invariants |

**Status:** Tests comprehensive, **cannot compile** due to missing exports.

---

## Blocking Issues

### 1. Not exported from lib.rs

```rust
// Need: pub mod gpu_driven;
```

### 2. No texture array binding

`binding_array<texture_2d<f32>>` is defined in spec but not implemented.

### 3. No GPU upload path

Tables manage slots but don't wire to wgpu buffer creation.

---

## Recommendations

1. **Export from lib.rs** - Immediate
2. **Wire buffer creation** - Connect to wgpu::Device
3. **Add bindless texture arrays** - WebGPU limits permitting

---

## Python Counterpart

| Rust | Python | Status |
|------|--------|--------|
| BufferRegistry | engine/core/memory/ring.py | Different design |
| MeshTable | engine/rendering/gpu_driven/mesh_table.py | Parallel |
| MaterialTable | engine/rendering/gpu_driven/material_table.py | Parallel |
| TextureTable | engine/rendering/gpu_driven/texture_table.py | Parallel |

---
