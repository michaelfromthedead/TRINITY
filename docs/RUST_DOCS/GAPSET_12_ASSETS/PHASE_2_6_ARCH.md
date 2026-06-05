# Phases 2-6 Architecture: Texture, Audio, Build, Streaming, Editor

## Overview

Phases 2-6 have no TODO task definitions (the PHASE_N_TODO.md only defines Phase 1 tasks). However, the codebase contains extensive Python scaffolding for all these phases. This document describes the architecture as implemented (scaffolding) and the gaps that need to be filled.

## Layer Map

```
[Python Editor Tooling]        engine/tooling/assettools/   ~4500 lines
    |  import, process, validate, reference-track, browse
    v
[Python Runtime / Build]       engine/resource/             ~2500 lines
    |  asset lifecycle, build pipeline, streaming, memory
    |
    |  ** NO BRIDGE EXISTS **
    v
[Rust GPU Tables]              crates/renderer-backend/     ~3500 lines
    |  MeshTable, MaterialTable, TextureTable, AssetLoader, BufferRegistry
    v
[WGSL Shader Bindings]
```

---

## Phase 2: Texture Import Pipeline

### Existing Code

**Python types**: `engine/resource/types/texture_asset.py`
- `TextureAsset` class with `format: TextureFormat`, `width`, `height`, `mip_count`, `is_sRGB`, `has_mips`
- `TextureFormat` enum: R8, RG8, RGB8, RGBA8, SRGB8_ALPHA8, BC1-7, ASTC_4x4-12x12, ETC2_RGB8, ETC2_RGBA8
- GPU upload params: `generate_mips: bool`, `preferred_format: TextureFormat`

**Editor import**: `engine/tooling/assettools/import_pipeline.py` (843 lines)
- `TextureImportSettings` class: format override, mip generation, max resolution
- Presets: `normal_map`, `ui_texture`, `hdr_environment`
- `_import_texture()` implementation: copies source file, returns path -- **NO DECODING**

**Editor processing**: `engine/tooling/assettools/asset_processor.py` (837 lines)
- `CompressionPreset` enum: BC1, BC3, BC4, BC5, BC7, ASTC_4x4, ETC2_RGB8
- `_compress_texture()` implementation: copies source file -- **NO COMPRESSION**

**Runtime table**: `crates/renderer-backend/src/gpu_driven/texture_table.rs` (271 lines)
- `TextureTableEntry`: width, height, mip_levels, format, layer_count, flags
- Free-list allocation, MAX 4096, BufferRegistry staging
- **Missing**: `texture_table.wgsl` (no WGSL binding, unlike mesh/material tables)

**Editor validation**: `engine/tooling/assettools/asset_validation.py` (886 lines)
- `TextureValidationRule`: max_dimensions, power_of_2, max_file_size, valid_format
- Context-data based, no actual file parsing

### Architecture

**Target flow**:
```
Source (.png, .jpg, .tga, .hdr, .exr, .ktx2)
    |
    v
[Import] decode pixels, extract metadata     (image crate)
    |  validate dimensions, format
    v
[Process] resize, generate mips, compress    (bcn crate, oodle)
    |  BC1/BC3/BC4/BC5/BC7/ASTC/ETC2
    v
[Upload] TextureTable::add() -> stage -> submit
    |  GPU: array<TextureTableEntry> via storage buffer
    v
[WGSL] texture_table.wgsl  *** MISSING ***
```

**Gaps**:
1. No image decoding (no `image` crate in Rust, no PIL in Python tooling)
2. No texture compression (no `bcn` or `etc2comp` integration)
3. No mipmap generation
4. No `texture_table.wgsl` WGSL binding
5. No bridge from Python processing output to Rust TextureTable
6. No streaming from disk to GPU (TextureStreamManager schedules but doesn't load)

---

## Phase 3: Audio Import Pipeline

### Existing Code

**Python types**: `engine/resource/types/audio_asset.py`
- `AudioAsset` class with `format: AudioFormat`, `channels`, `sample_rate`, `bit_depth`, `duration`
- `AudioFormat` enum: WAV, Ogg, MP3, FLAC
- `total_samples: int`, `loop_start: int`, `loop_end: int`

**Editor import**: `engine/tooling/assettools/import_pipeline.py`
- `AudioImportSettings` class: format conversion, sample rate, channel count
- Presets: `voice_dialogue`, `sound_effect`, `music`
- `_import_audio()` implementation: copies source file -- **NO DECODING**

### Architecture

**Target flow**:
```
Source (.wav, .ogg, .mp3, .flac, .aiff)
    |
    v
[Import] decode audio, extract PCM samples     (Symphonia or similar)
    |  validate format, sample rate
    v
[Process] resample, normalize, compress        (opus crate)
    |  Vorbis/Opus for runtime
    v
[Stream] chunk-based streaming                 (AudioStreamManager)
    |  priority-based, seek support
    v
[Playback] audio engine output                 (cpal or similar)
```

**Gaps**:
1. No audio decoding anywhere
2. No Rust audio system (no audio engine, no audio buffer management)
3. No streaming audio from disk
4. No audio-specific GPU tables or buffers
5. No Python-to-Rust bridge for audio asset references

---

## Phase 4: Build Pipeline

### Existing Code (all Python)

**Import stage**: `engine/resource/build/import_pipeline.py`
- `Importer` abstract base class: `import_asset(source_path, dest_path) -> ImportResult`
- `ImporterRegistry`: `register(format, importer)`, `find_importer(source_path)`
- `ImportPipeline`: `run_pipeline(assets, settings)` -- orchestrates batch imports
- No concrete importers registered

**Cook stage**: `engine/resource/build/cook_pipeline.py`
- `Cooker` abstract base class: `cook(source_path, target_platform) -> CookResult`
- `PlatformTarget` enum: WINDOWS, LINUX, MACOS, ANDROID, IOS, WEB
- `CompressionType` enum: NONE, LZ4, ZSTD, DEFLATE
- `CookManager`: registry for per-format-per-platform cookers
- No concrete cookers registered

**Process stage**: `engine/resource/build/process_pipeline.py`
- `ProcessStage` abstract base class: `process(context) -> ProcessResult`
- `ProcessContext`: quality setting (DRAFT, LOW, MEDIUM, HIGH, CINEMATIC), format, platform
- Stage chaining: `ProcessingPipeline` runs stages in order
- No concrete stages implemented

**Package stage**: `engine/resource/build/package_pipeline.py`
- `PackageBuilder`: creates `PackageManifest` with entries (source_path, archive_offset, packed_size, original_size, crc32)
- `ManifestEntry`: CRC32 checksum, compression type, chunk offsets
- `PackageReader`: reads entries from archive by name, decompresses on read
- Implementation is file-copy only -- no real PAK/ZIP archive format
- CRC32 is real (zlib-compatible)

**Distributed build**: `engine/resource/build/distributed_build.py`
- `DistributedBuildCoordinator`: worker registration, job creation/assignment, progress tracking
- `BuildJob`: asset list, build settings, target platform, assigned worker
- Round-robin assignment, timeout tracking, result collection

**Dependency tracking**: `engine/resource/build/dependency_tracker.py`
- `BuildDependencyTracker`: forward/reverse dependency maps, content hash tracking
- DFS cycle detection, topological sort
- Dirty-bit marking when hash changes

### Architecture

**Target flow**:
```
Source Assets
    |
    v
[IMPORT] -> read source format -> extract raw data
    |  handles: glTF, FBX, OBJ, PNG, JPG, WAV, OGG
    |  output: engine-intermediate format (.tasset)
    v
[COOK] -> platform-specific compression/conversion
    |  textures: BC1-7 (D3D), ASTC (mobile), ETC2 (Android)
    |  meshes: vertex format optimization
    |  audio: Vorbis/Opus compression
    v
[PROCESS] -> quality/format transforms
    |  LOD generation, normal map conversion, atlas packing
    |  quality levels: DRAFT -> LOW -> MEDIUM -> HIGH -> CINEMATIC
    v
[PACKAGE] -> PAK/ZIP archive generation
    |  CRC32 verification, optional encryption
    |  chunk-based for streaming
    v
Runtime-ready assets (.pak/.zip archive)
```

**Gaps**:
1. No concrete importers, cookers, or process stages implemented
2. No intermediate asset format defined (.tasset or similar)
3. No PAK/ZIP archive format implemented (file-copy only)
4. No integration with Rust GPU tables
5. Distributed build coordinator is architecturally sound but has no useful jobs to dispatch

---

## Phase 5: Streaming System

### Existing Code (all Python)

**Stream manager**: `engine/resource/streaming/stream_manager.py`
- `StreamManager`: singleton managing concurrent streams
- `MAX_CONCURRENT_STREAMS = 8`
- Priority queue (heap implementation): dequeue returns highest priority item
- `StreamPriority`: CRITICAL=0, HIGH=1, MEDIUM=2, LOW=3, BACKGROUND=4
- `StreamType`: TEXTURE_MIP, MESH_LOD, AUDIO_CHUNK, WORLD_CHUNK
- `request(asset_id, priority, stream_type)` -> enqueue
- `cancel(asset_id)` -> remove from queue
- `update()` -> process next batch

**Mesh streaming**: `engine/resource/streaming/mesh_stream_manager.py`
- `MeshStreamManager`: tracks per-mesh current vs. target LOD
- `request_lod(mesh_id, target_lod, priority)` -> enqueue
- `update()` -> process LOD transitions
- LOD level tracking

**Texture streaming**: `engine/resource/streaming/texture_stream_manager.py`
- `TextureStreamManager`: tracks per-texture current vs. target mip
- `request_mip(texture_id, target_mip, priority)` -> enqueue
- Priority calculation from screen size and distance

**Priority calculator**: `engine/resource/streaming/priority_calculator.py`
- `PriorityCalculator`: computes priority from multiple factors
- `screen_coverage_weight`: larger on-screen -> higher priority
- `distance_weight`: closer -> higher priority
- `frequency_of_use_weight`: accessed more often -> higher priority
- `latency_sensitive_weight`: interactive assets -> higher priority
- `calculate(asset, camera_state) -> float`

**Configuration**: `engine/resource/streaming/constants.py`
- `StreamingConfig`: enable flags per stream type, concurrency limits, timeouts
- Mesh config: max_requests_per_frame, visibility_timeout
- Texture config: max_mip_requests, mip_budget_mb
- Audio config: max_concurrent_streams, chunk_size, preload_duration

**Mesh LOD config**: `engine/resource/types/mesh_asset.py`
- `MeshAsset.lod_chain: list[SubMesh]` -- multiple LOD levels per mesh

### Architecture

**Target flow**:
```
Camera/View State
    |
    v
PriorityCalculator        (scheduling decisions)
    |  screen coverage, distance, frequency, latency
    v
StreamManager             (request queue)
    |  priority heap, MAX_CONCURRENT_STREAMS=8
    v
MeshStreamManager | TextureStreamManager   (LOD/mip tracking)
    |  current vs. target, transition management
    v
Disk I/O layer            *** MISSING ***
    |  async file reads, decompression
    v
GPU Upload                (BufferRegistry staging)
    |  MeshTable::update(), TextureTable::update()
    v
GPU memory
```

**Gaps**:
1. No disk I/O layer -- stream manager has nothing to load
2. No Rust integration -- streaming logic is Python-only, GPU tables are Rust-only
3. No GPU residency management -- Python ResidencyManager has no way to track GPU memory
4. No async I/O -- all streaming code is synchronous Python
5. No LOD selection in MeshTable -- MeshTableEntry has no LOD fields
6. PriorityCalculator is architecturally well-designed but never called

---

## Phase 6: Editor Integration

### Existing Code (all Python)

**Import pipeline (editor)**: `engine/tooling/assettools/import_pipeline.py` (843 lines)
- `ImportSettings` base with specialized types: `FBXImportSettings`, `OBJImportSettings`, `glTFImportSettings`, `TextureImportSettings`, `AudioImportSettings`
- `ImportPreset`: game_ready_mesh, static_prop, normal_map, ui_texture, voice_dialogue, sound_effect, music -- each with preset settings
- `ImportBatch`: batch import from directory with progress tracking via worker threads
- `ContentStore`: import deduplication via content hashing, provenance tracking (original path, timestamps, source hash)
- `_import_mesh()`: copies file (placeholder)
- `_import_texture()`: copies file (placeholder)
- `_import_audio()`: copies file (placeholder)

**Asset processor**: `engine/tooling/assettools/asset_processor.py` (837 lines)
- `BatchProcessor`: thread pool processing, progress reporting, dependency ordering
- `CompressionPreset`: BC1, BC3, BC4, BC5, BC7, ASTC_4x4, ASTC_6x6, ASTC_8x8, ASTC_10x10, ASTC_12x12, ETC2_RGB8, ETC2_RGBA8 -- all defined, none implemented
- `ProcessingPipeline`: formal chain of `ProcessingStage` objects
- `_compress_mesh()`: copies file (placeholder)
- `_compress_texture()`: copies file (placeholder)
- `_convert_audio()`: copies file (placeholder)

**Asset validation**: `engine/tooling/assettools/asset_validation.py` (886 lines)
- `ValidationRule` abstract base: `validate(asset_data, context) -> list[ValidationIssue]`
- `ValidationIssue`: severity (ERROR, WARNING, INFO), rule name, message, path, suggested fix
- `TextureValidationRule`: checks max_dimensions, power_of_2, max_file_size, valid_format -- from context data
- `MeshValidationRule`: checks max_vertices (65535), max_triangles (131071), requires_normals, requires_uvs, bone_limit -- from context data
- `MaterialValidationRule`: checks shader_compatibility, texture_references -- from context data
- `NamingConventionRule`: snake_case/PascalCase presets, prefix patterns, forbidden characters
- `AssetValidator`: batch validation with per-asset rule sets, error summary

**Reference management**: `engine/tooling/assettools/reference_manager.py` (827 lines)
- `ReferenceGraph`: forward edges (`agent -> dependency list`) and reverse edges (`resource -> dependents list`)
- `resolve_dependencies(asset_id) -> list[asset_id]`: breadth-first transitive resolution
- `has_transitive_dependency(from_id, to_id) -> bool`: DFS with visited set
- `detect_cycles() -> list[cycle]`: DFS-based cycle detection
- `BrokenReferenceDetector`: finds references to non-existent assets, with ContentStore hash fallback for moved/renamed files
- `ReferenceRedirect`: stores replacements, auto-applies on access, batch apply/revert
- `repair_broken_references()`: path similarity scoring, rename-based fix suggestions
- `default_json_scanner()`: regex-based JSON key scanning for `"$ref": "path"` patterns
- `default_python_scanner()`: Python import/from pattern scanning

**Content browser**: `engine/tooling/assettools/content_browser.py`
- `ContentBrowser`: directory tree scanning, file type filtering, search integration
- Thumbnail display integration with `ThumbnailGenerator`

**Other tooling**:
- `collections.py`: asset collection management
- `metadata.py`: `AssetMetadata` with `to_dict()`/`from_dict()` serialization
- `search.py`: `AssetSearchEngine` with fuzzy matching, tag-based filtering, facet counts
- `thumbnail_generator.py`: `ThumbnailGenerator` with PIL-based resize, format conversion, caching

### Architecture

**Target flow**:
```
User in Editor
    |
    +---> Import Pipeline
    |       format settings, presets, batch import
    |       -> validates -> imports -> processes -> packages
    |
    +---> Asset Processor
    |       batch compression, optimization, conversion
    |       -> thread pool, progress, dependency ordering
    |
    +---> Reference Manager
    |       dependency graph, broken ref detection, redirects
    |       -> cycle detection, auto-fix suggestions
    |
    +---> Content Browser / Search / Metadata
    |       asset discovery, filtering, thumbnails
    |
    v
Asset Database (JSON metadata + processed assets)
    |
    v
Runtime ingest                                             *** MISSING ***
    |  Python tools produce processed assets
    |  -> need bridge to Rust GPU tables
    v
GPU-ready data
```

**Gaps**:
1. All processing is placeholder (copy file)
2. No asset database backend (metadata stored in-memory only)
3. No bridge from Python tooling to Rust runtime
4. Validation rules work on context data only -- no actual file parsing
5. Thumbnail generator mirrors the import pipeline -- processes files that were already "imported" (copied)
6. Reference manager works on JSON/Python refs -- needs asset database integration

---

## Cross-Cutting Architecture Issues (Phases 2-6)

### 1. Python-Rust Bridge Required

All Python scaffolding produces no data that Rust can consume. The Rust GPU tables have no Python API. The gap is absolute. Solutions:
- **FFI bridge**: `pyo3` or `cbindgen` to expose Rust functions to Python
- **IPC**: Python writes processed asset files, Rust reads them (file-based protocol)
- **Unification**: move all asset processing to Rust, use Python only for editor UI

### 2. No Asset Database

Metadata, collections, search, and reference tracking all operate on in-memory data structures. There is no persistent asset database (no SQLite, no custom format). All tooling state is ephemeral.

### 3. No Real File Format Support

Every file format mentioned across all tooling (BC1-7, ASTC, ETC2, ZSTD, LZ4, WAV, Ogg, Opus, KTX2) is referenced by enum value but never actually called. Adding any single format requires adding the corresponding library dependency and replacing a copy-file stub.

### 4. Streaming Architecture is Decoupled from Reality

The streaming system assumes it can load LOD/mip chunks on demand, but:
- No chunks exist (no package format)
- No disk I/O layer exists
- No GPU residency tracking exists
- No LOD/mip generation exists
