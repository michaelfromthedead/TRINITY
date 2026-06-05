# GAPSET_12_ASSETS -- Clarification Requests

## Critical: What is the intended relationship between Python and Rust?

The codebase has two completely separate asset systems:
- **Python**: `engine/resource/` and `engine/tooling/assettools/` -- full scaffolding for asset lifecycle, build pipeline, streaming, memory management, editor tooling (~7000+ lines)
- **Rust**: `crates/renderer-backend/src/` -- production-quality GPU tables (MeshTable, MaterialTable, TextureTable) with BufferRegistry staging, plus AssetLoader thread architecture (~3500 lines)

**There is no bridge between them.** Python cannot call into Rust. Rust has no Python bindings. The GPU tables have no API for Python to feed them data.

**Question**: Is this intentional separation (Python = editor tooling, Rust = runtime, connected via IPC/socket/file), or is one of these supposed to be replaced? The TODO items all assume Rust implementation, but the Python scaffolding suggests a Python-centric editor workflow.

## Scope: 40 tasks or 7?

The TODO header claims 40 tasks across 6 phases, but only Phase 1 has task definitions (7 tasks), and even those are incomplete (T-AS-1.4 is truncated, T-AS-1.5/1.6/1.7 are named but have no ACs).

**Question**: Should the remaining 33 tasks be written out as concrete specifications, or should the scope be adjusted to match only what's actually defined? Writing 33 new task definitions would be a substantial documentation effort.

## glTF Parser: Rust or Python?

The TODO says "Worker-thread offloadable" and "Morton order" which imply a Rust implementation (the `AssetLoader` already has a worker thread). But `engine/tooling/assettools/import_pipeline.py` has glTF settings and presets.

**Question**: Should the glTF parser be in Rust (as the AssetLoader suggests), Python (as the tooling suggests), or both (Python for editor import, Rust for runtime load)?

## TextureTable WGSL: Missing shader binding

`mesh_table.rs` has `mesh_table.wgsl`. `material_table.rs` has `material_table.wgsl`. But `texture_table.rs` has no corresponding `texture_table.wgsl`.

**Question**: Was this an oversight, or is the texture table accessed differently on the GPU (e.g., via Vulkan sampler arrays rather than a storage buffer)?

## MeshTable Entry Format: No vertex format field

`MeshTableEntry` (24 bytes, six u32 fields) describes mesh layout but has NO vertex format field. The `vertex_buffer_offset` and `vertex_buffer_size` fields assume the vertex data is already in the correct format.

**Question**: Should `MeshTableEntry` include a `vertex_format` field (packed bitfield or enum), or is vertex format handled outside the table (e.g., via a separate pipeline layout)?

## LOD: Python MeshAsset structure vs. MeshTableEntry

Python `MeshAsset` has `lod_chain: list[SubMesh]`. Rust `MeshTableEntry` has no LOD fields. `MeshStreamManager` tracks LOD state independently.

**Question**: Should `MeshTableEntry` store per-mesh LOD count and offsets (to enable GPU LOD selection), or should LOD remain a CPU-side concern (Python streaming manager decides, CPU updates visible flag)?

## Validation: Python framework, Rust enforcement

`engine/tooling/assettools/asset_validation.py` (886 lines) has a complete rule-based validation framework in Python. But the Rust tables accept any data without validation.

**Question**: Should validation happen in Python (before import), Rust (at GPU upload time), or both? Python validation is natural for editor workflows. Rust validation would catch runtime issues.

## Memory Management: Python controls, Rust executes?

The Python scaffolding has BudgetManager, EvictionManager, ResidencyManager. The Rust side has no memory management beyond BufferRegistry's staging slots.

**Question**: Is memory management intended to be Python-orchestrated (deciding what to load/unload) with Rust executing (upload/evict GPU resources)? If so, this needs an explicit IPC protocol.

## Streaming: Python logic, Rust I/O?

Same split question: Python `StreamManager` does priority-based scheduling. Rust has no streaming support.

**Question**: Should streaming I/O happen in Rust (via async read from disk) with Python providing scheduling decisions? Or is this a pure Rust streaming system that the Python scaffolding anticipated but never connected?

## Meshlet Support: Flat-only MeshTable

T-AS-1.4 asks for meshlet partitioning, but `MeshTableEntry` has per-entry bounds (not per-meshlet). The meshlet system would need either:
- **New table**: `MeshletTable` with per-meshlet entries (bounds, vertex/triangle offsets)
- **Extended MeshTable**: `MeshTableEntry` grows to include meshlet range info

**Question**: Which approach is preferred? The TODO mentions "Morton order or k-means" partitioning, but the table architecture determines the implementation significantly.

## T-AS-1.2 "Interleaved, Split, Compressed" -- which layout?

The GPU tables assume a single interleaved vertex buffer per mesh. "Split" and "Compressed" layouts would change the MeshTable contract.

**Question**: Should the vertex format conversion produce the interleaved format the GPU tables expect, or should the tables support multiple layouts (potentially with per-attribute buffer references)?
