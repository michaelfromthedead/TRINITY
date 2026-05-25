# GAPSET_12_ASSETS -- Project Overview

## Purpose

Build the complete asset pipeline for the TRINITY engine: import, process, cook, stream, and render 3D assets (meshes, textures, audio) from source files to GPU memory.

## Architecture Layers

```
Source Files (.gltf, .glb, .png, .wav, etc.)
        |
   [Python Tooling Layer]  -- engine/tooling/assettools/
   Import, validate, process, reference-track, content-browse
        |
   [Python Runtime Layer]  -- engine/resource/
   Asset handles, dependency graphs, build pipelines, streaming, memory mgmt
        |
   [Rust Bridge]           -- (MISSING -- no Python-to-Rust bridge exists)
        |
   [Rust GPU Layer]        -- crates/renderer-backend/src/
   AssetLoader (threading), MeshTable, MaterialTable, TextureTable, BufferRegistry
        |
   [GPU]                   -- WGSL shader bindings
   mesh_table.wgsl, material_table.wgsl
```

## Current State

### Complete (production-quality)
- **MeshTable** (`gpu_driven/mesh_table.rs`): 1804 lines. Bindless GPU table with `#[repr(C)]` 24-byte entries, hole-preserving add/remove/update, `live_count`/`free_count` tracking, `stage_and_submit()` via BufferRegistry, 60+ tests, WGSL binding.
- **MaterialTable** (`gpu_driven/material_table.rs`): 1303 lines. PBR material table with `#[repr(C, align(16))]` 80-byte entries, dirty-bit tracking (bit 31), dirty-range staging, 60+ tests, WGSL binding.
- **TextureTable** (`gpu_driven/texture_table.rs`): 271 lines. Bindless texture table with free-list, 4096 max, BufferRegistry staging. Missing WGSL binding.
- **BufferRegistry** (`gpu_driven/buffers.rs`): Triple-buffered staging, acquire/submit protocol, slot management.

### Partial (architecture done, implementation placeholder)
- **AssetLoader** (`asset_loader.rs`): Worker thread, mpsc channel, MeshData struct, AssetError enum, 20+ tests. `process_load()` is `Ok(Vec::new())`.

### Scaffolding Only (no real processing)
- **Python Runtime** (~2000 lines): AssetHandle (generational), AssetManager (slot/path/ref management), DependencyGraph (DAG), mesh/texture/audio/skeleton/animation/material/scene types, build pipeline (import/cook/process/package), streaming system (priority queue, LOD/mip managers, priority calculator), memory management (budget/eviction/residency), virtualization (texture/geometry/shadow maps).
- **Python Tooling** (~4500 lines): ImportPipeline (format settings, presets, batch), AssetProcessor (compression pipeline, thread pool), AssetValidator (rule-based validation), ReferenceManager (graph, cycle detection, redirect), ContentBrowser, Search, Metadata, Collections, ThumbnailGenerator.

## Key Design Decisions

### GPU Tables: Bindless via BufferRegistry
All GPU-visible asset tables (MeshTable, MaterialTable, TextureTable) use the same pattern:
1. CPU-side `Vec<Entry>` with hole-preserving mutation
2. `as_bytes()` for zero-copy staging serialization
3. `BufferRegistry.stage_and_submit()` for triple-buffered GPU upload
4. WGSL `array<Entry>` matches `#[repr(C)]` Rust layout

This means the GPU pipeline architecture is correct -- only the data ingestion path is missing.

### Two-Tier Asset System
- **Rust tier**: GPU-facing. Minimal, high-performance. Assumes pre-processed data ready for upload.
- **Python tier**: Editor-facing. Rich tooling, validation, batch processing, streaming orchestration.
- **Problem**: No bridge exists between tiers. Python cannot feed the Rust GPU tables.

### Generational Handles
- Python `AssetHandle`: 24-bit index + 8-bit generation, packed into one int. States: REQUESTED -> QUEUED -> LOADING -> LOADED -> READY -> FAILED -> UNLOADING -> UNLOADED.
- Rust GPU tables: raw `u32` index. No generation counter. No handle validation.
- These need unification for a coherent end-to-end asset system.

## Build Pipeline Stages (Python scaffolding)
1. **Import** -> Read source format, extract raw data (all stub)
2. **Cook** -> Platform-specific compression/format conversion (all stub)
3. **Process** -> Quality/format transforms, LOD generation (all stub)
4. **Package** -> PAK/ZIP archive generation (file-copy stub)

## Streaming System (Python scaffolding)
- Priority-queue based stream manager
- Separate MeshStreamManager (LOD levels) and TextureStreamManager (mip levels)
- PriorityCalculator with distance/screen-size/frequency weights
- MAX_CONCURRENT_STREAMS=8
- Not wired to any actual I/O or GPU residency

## Memory Management (Python scaffolding)
- BudgetManager: per-category (texture/mesh/audio/other) budgets
- EvictionManager: LRU/LFU/Size/Priority policies
- ResidencyManager: combines budget + eviction
- Not wired to GPU memory

## Virtualization (Python scaffolding)
- VirtualTextureSystem: page table + physical pool + LRU eviction + feedback mechanism
- VirtualGeometrySystem: nanite-style cluster LOD
- VirtualShadowMaps: shadow atlas allocation

## Immediate Next Steps

1. **Replace `process_load()` placeholder** with actual glTF parsing using the `gltf` crate. This is the smallest-scope item that unlocks everything.
2. **Wire AssetLoader -> MeshTable/MaterialTable**: after parsing, populate the bindless GPU tables.
3. **Add `texture_table.wgsl`**: match the Rust TextureTable layout for WGSL shader access.
4. **Define Rust-side audio and streaming** to match what the Python scaffolding describes.
5. **Build the Python-to-Rust bridge** so the editor tooling can feed real data to the GPU layer.
