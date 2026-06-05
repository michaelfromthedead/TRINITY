# Phase 1 Architecture: glTF Mesh Pipeline

## Overview

Phase 1 covers the end-to-end pipeline from glTF/glB source files to populated GPU bindless tables. The pipeline has three conceptual stages: Parse -> Convert -> Upload.

## Current Architecture (as implemented)

```
Source File (.gltf / .glb)
    |
    v
AssetLoader::load_gltf(path)              [asset_loader.rs:147]
    |  sends LoadRequest via mpsc channel
    v
AssetLoader::process_load(path)            [asset_loader.rs:162]
    |  validates extension, returns Ok(Vec::new())  *** PLACEHOLDER ***
    v
Result<Vec<MeshData>, AssetError>          [asset_loader.rs:26-36]
    |  MeshData { vertex_data, index_data, vertex_count, index_count }
    v
(No code to consume MeshData and populate GPU tables)
```

## Target Architecture (as described by TODO)

```
Source File (.gltf / .glb)
    |
    v
[1] glTF 2.0 Parser (T-AS-1.1)
    |  JSON manifest, buffer resolution, accessor slicing, node hierarchy
    v
Raw glTF Data (accessors, buffer views, nodes, skins)
    |
    v
[2] Vertex Format Conversion (T-AS-1.2)
    |  Interleaved/Split/Compressed, quantization, axis conversion
    v
[3] Index Buffer Optimization (T-AS-1.3)
    |  16/32/8-bit selection, cache optimization, stripification
    v
Engine Mesh Data
    |
    +---> [4] Meshlet Generation (T-AS-1.4)
    |         Cluster partitioning, bounds, adjacency
    |
    +---> [5] Skeleton Extraction (T-AS-1.5)
    |         Joint hierarchy, inverse bind, animation
    |
    v
[6] GPU Upload (T-AS-1.6)
    |  MeshTable::add(), MaterialTable::add(), BufferRegistry staging
    v
[7] LOD Generation (T-AS-1.7)
    |  Simplification, LOD chain, screen-space error
    v
GPU Tables (bindless storage buffers)
```

## Key Components

### Component 1: AssetLoader (existing, needs parser integration)

**File**: `crates/renderer-backend/src/asset_loader.rs` (367 lines)

**What exists**:
- `AssetLoader` struct with `tx: Sender<LoadRequest>` and `handle: Option<JoinHandle<()>>`
- `new()` spawns a named thread ("asset-loader") with a `for request in rx` event loop
- `load_gltf(path)` sends request via channel, blocks on `rx.recv()` for response
- `Drop` closes channel and joins thread (with `std::mem::replace` trick to avoid deadlock)
- `MeshData` struct: `vertex_data: Vec<u8>`, `index_data: Vec<u8>`, `vertex_count: u32`, `index_count: u32`
- `AssetError` enum: `NotFound`, `ParseError`, `UnsupportedFormat`

**What's needed**:
- Replace `process_load()` body with actual glTF parsing using the `gltf` crate
- Add `serde_json` dependency for manifest parsing
- Handle buffer views: resolve base64 data URIs and external .bin files
- Extract accessors: positions, normals, tangents, UVs, colors, joints, weights
- Flatten node hierarchy into per-primitive MeshData
- Progressive loading: skeleton/bounds first pass, then vertices, then skinning data

**Design notes**:
- The threading architecture is correct and well-tested (20+ tests)
- The `process_load` function runs on the worker thread, so blocking IO is acceptable
- Test coverage includes: empty path, unsupported extensions, no extension, .gltf/.glb paths with directories, case sensitivity, multiple concurrent calls, creation/load/error cycles

### Component 2: GPU Bindless Tables (existing, complete)

**MeshTable**: `crates/renderer-backend/src/gpu_driven/mesh_table.rs` (1804 lines)
- `MeshTableEntry`: `#[repr(C)]` -- 6 x u32 = 24 bytes
  - `vertex_buffer_offset`, `vertex_buffer_size`, `index_buffer_offset`, `index_buffer_size`
  - `bounds_min_x`, `bounds_max_x`, packed in `packed_bounds_0/1/2/3` (5:6:5 per component)
  - `material_index`, `flags`, `visibility_bits`
- `MeshTable`: `Vec<MeshTableEntry>`, hole-preserving, live_count, free_list
- `add(entry)` -> `Option<u32>` -- auto-grows, reuses freed slots
- `remove(index)` -> zeroes entry, returns index to free list
- `stage_and_submit(registry)` -> full table serialization to BufferRegistry staging
- `as_bytes()` -> zero-copy slice for GPU upload

**MaterialTable**: `crates/renderer-backend/src/gpu_driven/material_table.rs` (1303 lines)
- `MaterialTableEntry`: `#[repr(C, align(16))]` -- 80 bytes
  - PBR parameters: base_color (u32 packed 10:10:10:2), emissive (u32 packed), metallic/roughness/occlusion (f32), normal_scale (f32)
  - 4x texture_id (u32): albedo, normal, metallic_roughness, emissive
  - flags (u32): bit 31 = dirty, bits 0-15 = texture usage masks
  - alpha_cutoff (f32)
- Dirty-bit tracking: `stage_dirty_range()` only uploads modified entries
- Dirty range tracked as `(first_dirty, last_dirty)` for efficient upload

**TextureTable**: `crates/renderer-backend/src/gpu_driven/texture_table.rs` (271 lines)
- `TextureTableEntry`: `#[repr(C)]` -- 6 x u32 = 24 bytes
  - `width`, `height`, `mip_levels`, `format`, `layer_count`, `flags`
- Free-list allocation, MAX 4096 textures
- No WGSL binding (unlike mesh/material tables)

**Note**: MeshTable entries have NO vertex_format field. The vertex data is assumed to be in a pre-negotiated interleaved format. This is fine for the current interleaved-only approach but would need extension for split or compressed layouts.

### Component 3: BufferRegistry (existing, complete)

**File**: `crates/renderer-backend/src/gpu_driven/buffers.rs`

- Multiple staging slots for triple-buffering
- `acquire_staging()` -> `AcquireResult::Acquired { slot_index }` or `NoSlotAvailable`
- `slot_mut(slot_index)` -> mutable ref to staging buffer
- `submit_staging(slot_index, byte_size)` -> `SubmitResult::Submitted` or `InvalidSlot`
- Each slot supports `resize(new_cap)` for dynamic sizing

### Component 4: glTF Parser (needed, T-AS-1.1)

**Where to build**: `crates/renderer-backend/src/asset_loader.rs` (replace process_load)

**Key implementation details**:
- Add `gltf` crate dependency (MIT/Apache-2.0, mature Khronos-ratified parser)
- Parse order:
  1. Load JSON manifest (`gltf::Gltf::from_path()` or `serde_json::from_slice()`)
  2. Collect buffer views, resolve data URIs / load .bin files
  3. For each mesh -> primitive -> accessor:
     a. Read POSITION accessor -> flatten to f32 byte stream
     b. Read NORMAL, TANGENT, TEXCOORD_0, COLOR_0 as available
     c. Read JOINTS_0, WEIGHTS_0 for skinned meshes
     d. Read indices (u8/u16/u32) -> flatten to u32 byte stream
  4. Pack into MeshData { vertex_data, index_data, vertex_count, index_count }
  5. Return Vec<MeshData> (one per primitive)

**Progressive loading**:
- First pass: skeleton (skinned mesh joint lists), bounds (from accessor min/max)
- Second pass: vertex data (streaming from .bin file)
- Third pass: skinning data (JOINTS_0, WEIGHTS_0) -- optional, may not exist

### Component 5: Vertex Format Conversion (needed, T-AS-1.2)

**Where to build**: New module, e.g., `crates/renderer-backend/src/vertex_conversion.rs`

**Design**:
- `VertexLayout` enum: Interleaved, Split, Compressed
- `VertexAttribute` enum: Position(f32x3), Normal(f32x3 or SNORM), Tangent, TexCoord(f32x2 or f16x2), Color, Joints(u16x4), Weights(f32x4)
- `QuantizationScheme`: Float32, Float16, UNorm8, SNorm8, UNorm10_10_10_2, Octahedral16
- Conversion function: `convert_gltf_to_engine(gltf_data, layout, settings) -> EngineMesh`
- `@import_settings` parser: scale, axis_conversion (Y-up vs Z-up), merge_meshes

### Component 6: GPU Upload Pipeline (partially done, T-AS-1.6)

**What's done**:
- All GPU tables support `stage_and_submit(BufferRegistry)` for upload
- no changes needed to the GPU tables

**What's needed**:
- New module: `crates/renderer-backend/src/upload_pipeline.rs`
- `UploadPipeline::ingest_mesh(mesh_data, &mut MeshTable, &mut MaterialTable, &mut BufferRegistry) -> u32`
- This is the bridge: takes parsed MeshData, creates MeshTableEntry + MaterialTableEntry, stages to GPU

## Data Flow (target)

```
load_gltf("model.gltf")
    |  mpsc send -> worker thread
    v
process_load(path)                                 [T-AS-1.1]
    |  gltf crate: parse JSON -> buffers -> accessors -> primitives
    v
Vec<RawPrimitive> { positions, normals, uvs, indices, joints, weights }
    |
    v
vertex_conversion(primitives, settings)            [T-AS-1.2]
    |  interleave, quantize, axis-convert
    v
index_optimization(converted_meshes)               [T-AS-1.3]
    |  select index size, cache-optimize, optionally stripify
    v
Vec<MeshData> { vertex_data, index_data, vcount, icount }
    |
    v
upload_pipeline::ingest_mesh(mesh_data, tables)    [T-AS-1.6]
    |  MeshTable::add(), MaterialTable::add(), stage_and_submit()
    v
GPU tables populated
```

## Dependencies

| Component | Depends On | Crate/Tool |
|-----------|------------|------------|
| T-AS-1.1 Parser | S15 math.rs | gltf, serde_json, base64 |
| T-AS-1.2 Conversion | T-AS-1.1, S14 RHI | (none beyond std) |
| T-AS-1.3 Index Opt | T-AS-1.2 | meshopt |
| T-AS-1.4 Meshlets | T-AS-1.2, GAPSET_11 | (custom) |
| T-AS-1.5 Skeleton | T-AS-1.1 | gltf |
| T-AS-1.6 Upload | T-AS-1.1, GPU tables | BufferRegistry |
| T-AS-1.7 LOD | T-AS-1.1 | meshopt |

## Test Strategy

| Component | Tests |
|-----------|-------|
| AssetLoader (existing) | 20+ tests: creation, load, error paths, multiple calls, Drop |
| glTF Parser | Load Khronos sample models (Basic, PBR, Skin), verify vertex/index counts, verify bounds |
| Vertex Conversion | Round-trip: convert known data, verify byte-exact output for each vertex format |
| Index Optimization | Verify idempotency, verify ACMR improvement > baseline |
| Meshlet Generation | Verify cluster sizes in [N-2, N+2] range, verify bounds contain all vertices |
| GPU Upload | Verify MeshTable entries match source data after stage_and_submit |
| Integration | Load glTF -> process -> upload -> verify GPU buffer content byte-exact |
