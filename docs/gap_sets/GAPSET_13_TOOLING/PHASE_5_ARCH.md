# PHASE_5_ARCH.md — Advanced Editor Architecture

## Overview

Phase 5 builds the advanced editor panels: Material Editor (DSL compiler, node graph, preview), Asset Browser (filesystem tree, thumbnails, search), and Profiler Overlay (frame timing, pass breakdown). All panels have existing Python implementations that need Rust/egui panel wiring.

## Current State

| Task | Status | What Exists | What's Missing |
|------|--------|-------------|----------------|
| T-TL-5.1 | [~] PARTIAL | Python material_compiler.py, material_nodes.py | Python-to-WGSL, naga validation |
| T-TL-5.2 | [~] PARTIAL | Python material_graph.py, material_parameters.py | Rust/egui material editor panel |
| T-TL-5.3 | [~] PARTIAL | Python content_browser.py, assets.rs commands | egui Asset Browser panel |
| T-TL-5.4 | [~] PARTIAL | Python thumbnail_generator.py | Rust/wgpu GPU thumbnailing |
| T-TL-5.5 | [~] PARTIAL | Python profiler_overlay.py | Rust/egui profiler overlay panel |
| T-TL-5.6 | [~] PARTIAL | Python material_library.py, material_instances.py | Rust material library integration |
| T-TL-5.7 | [~] PARTIAL | Python material_graph.py, node_factory.py | Rust/egui node graph widget |

## Architecture: Material DSL Compiler (Gap)

```
┌────────────────────────────────────────────────────┐
│  Material DSL Compiler Pipeline                      │
│                                                     │
│  Python DSL input:                                  │
│  ┌────────────────────────────────────────────┐     │
│  │ material PBRGlass                           │     │
│  │   property baseColor = color(1.0, 1.0, 1.0)│     │
│  │   property roughness = 0.1                  │     │
│  │   property metallic = 0.0                   │     │
│  │                                              │     │
│  │   fragment {                                │     │
│  │     let albedo = baseColor.rgb;             │     │
│  │     let f0 = mix(0.04, albedo, metallic);   │     │
│  │     let N = normalize(world_normal);        │     │
│  │     let V = normalize(camera_position -     │     │
│  │                world_position);             │     │
│  │     ...                                      │     │
│  │   }                                          │     │
│  └────────────────────────────────────────────┘     │
│                                                     │
│  Compiler stages:                                   │
│  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────┐ │
│  │ Python  │→ │ AST      │→ │ WGSL    │→ │ naga │ │
│  │ Parser  │  │ Builder  │  │ Codegen │  │ Valid│ │
│  └─────────┘  └──────────┘  └─────────┘  └──────┘ │
│                                                     │
│  Features:                                          │
│  ├── Python asteval or custom parser                │
│  ├── AST optimization (constant folding)            │
│  ├── WGSL code generation with template patterns    │
│  └── naga::Validator for compile-time error checks  │
└────────────────────────────────────────────────────┘
```

## Architecture: Material Editor Panel (Gap)

```
┌──────────────────────────────────────────────────────┐
│  Material Editor Panel (egui)                         │
│                                                       │
│  ┌─────────────────────┬──────────────────────────┐   │
│  │  Code Editor        │  Preview Viewport         │   │
│  │                     │                           │   │
│  │  ┌───────────────┐  │  ┌────────────────────┐  │   │
│  │  │ material PBR  │  │  │                    │  │   │
│  │  │   property .. │  │  │   Preview Sphere   │  │   │
│  │  │   fragment {  │  │  │                    │  │   │
│  │  │     ...       │  │  │                    │  │   │
│  │  │   }           │  │  └────────────────────┘  │   │
│  │  └───────────────┘  │                           │   │
│  │                     │  Parameters:               │   │
│  │  [Compile] [Save]   │  baseColor: [■]           │   │
│  │                     │  roughness: [════●══] 0.1 │   │
│  │  Errors:            │  metallic:  [●══════] 0.0 │   │
│  │  ✓ Compilation OK   │                           │   │
│  └─────────────────────┴──────────────────────────┘   │
│                                                       │
│  Features:                                            │
│  ├── Syntax-highlighted text editor                   │
│  ├── Compilation error display (inline markers)       │
│  ├── Parameter sliders live-update preview            │
│  └── Preview sphere in dedicated viewport             │
└──────────────────────────────────────────────────────┘
```

## Architecture: Asset Browser Panel (Gap)

```
┌──────────────────────────────────────────────────────┐
│  Asset Browser Panel (egui)                           │
│                                                       │
│  ┌──────────────┬──────────────────────────────┐     │
│  │  Tree View   │  Thumbnail Grid               │     │
│  │              │                               │     │
│  │  ▼ project   │  ┌────┐ ┌────┐ ┌────┐ ┌────┐ │     │
│  │    ▼ meshes  │  │    │ │    │ │    │ │    │ │     │
│  │      hero    │  │ M1 │ │ M2 │ │ T1 │ │ T2 │ │     │
│  │      enemy   │  └────┘ └────┘ └────┘ └────┘ │     │
│  │    ▼ textures│  ┌────┐ ┌────┐ ┌────┐ ┌────┐ │     │
│  │      diffuse │  │    │ │    │ │    │ │    │ │     │
│  │      normal  │  │ M3 │ │ T3 │ │ T4 │ │ S1 │ │     │
│  │    ▼ materials│  └────┘ └────┘ └────┘ └────┘ │     │
│  │      pbr_glass│                               │     │
│  │      metal    │  [List view] [Grid view]      │     │
│  │    ▼ shaders  │  Filter: [all         ▼]     │     │
│  │      pbr.wgsl │  Search: [🔍          ]      │     │
│  └──────────────┴──────────────────────────────┘     │
│                                                       │
│  Data sources:                                        │
│  ├── Filesystem via Tauri FS plugin                   │
│  ├── assets.rs commands (import_asset, get_asset_url) │
│  └── Thumbnail cache (T-TL-5.4)                       │
│                                                       │
│  Features:                                            │
│  ├── Filesystem tree with expand/collapse              │
│  ├── File type filters (models, textures, materials)   │
│  ├── Thumbnail grid view                              │
│  ├── Search/filter with real-time matching             │
│  └── Drag-drop import (Tauri dialog)                  │
└──────────────────────────────────────────────────────┘
```

## Architecture: Thumbnail Cache (Gap)

```
┌──────────────────────────────────────────────────────┐
│  ThumbnailCache                                       │
│                                                       │
│  Generation pipeline:                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐ │
│  │ Asset    │→ │ GPU      │→ │ Mipmap   │→ │ Cache │ │
│  │ Import   │  │ Render   │  │ Downscale│  │ Store │ │
│  │ Trigger  │  │ (128x128)│  │          │  │ (LRU) │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────┘ │
│                                                       │
│  struct ThumbnailCache {                             │
│      entries: LruCache<AssetId, Thumbnail>,          │
│      pending: Vec<AssetId>,  // queued for generation│
│      max_resolution: (u32, u32),  // 128x128 default │
│  }                                                    │
│                                                       │
│  Features:                                            │
│  ├── Automatic generation on import                   │
│  ├── GPU mipmap-based downscaling                     │
│  ├── LRU eviction with configurable max entries       │
│  └── Fallback icon for unsupported types              │
└──────────────────────────────────────────────────────┘
```

## Architecture: Profiler Overlay Panel (Gap)

```
┌──────────────────────────────────────────────────────┐
│  Profiler Overlay Panel (egui)                        │
│                                                       │
│  ┌──────────────────────────────────────────────┐     │
│  │ Frame Timing Sparkline                        │     │
│  │ ──▄▄▆███▆▄▄▆▇███▇▆▄▄▆█▆▄──                  │     │
│  │  16ms budget ──────▄▄▄▄▄▄▄▄▄▄────             │     │
│  │                        ▲ spike 22ms            │     │
│  ├──────────────────────────────────────────────┤     │
│  │ Per-Pass Breakdown    GPU       CPU           │     │
│  │ ShadowPass            ████ 4.1ms   ██ 0.2ms  │     │
│  │ OpaquePass            ███████ 7.2ms ██ 0.3ms │     │
│  │ TransparentPass       ██ 2.1ms   █ 0.1ms    │     │
│  │ PostProcess           █ 1.0ms    █ 0.1ms    │     │
│  ├──────────────────────────────────────────────┤     │
│  │ Budget: 12.4ms / 16.0ms ████████████░░░░░    │     │
│  │ Memory: 2.4GB GPU, 1.2GB CPU                 │     │
│  │ Draw calls: 1,234  Triangles: 2.5M           │     │
│  └──────────────────────────────────────────────┘     │
│                                                       │
│  Data sources:                                        │
│  ├── FrameProfiler::get_latest_frame()                │
│  ├── GpuMemoryTracker::current_stats()                │
│  └── Editor::entity_count()                           │
│                                                       │
│  Features:                                            │
│  ├── Real-time frame timing sparkline                 │
│  ├── Per-pass GPU/CPU breakdown                       │
│  ├── Budget bar with color coding                     │
│  ├── Memory usage display                             │
│  └── Tooltip on hover for detailed info               │
└──────────────────────────────────────────────────────┘
```

## Dependency Chain

```
Phase 2 (Core Panels)
  │  Provides: panel infrastructure, EditorCamera
  ▼
T-TL-5.1 Material DSL Compiler
  │
  ├──► T-TL-5.2 Material Editor Panel ──► Phase 3 T-TL-3.5 (preview vp)
  ├──► T-TL-5.7 Material Node Graph
  └──► T-TL-5.6 Material Library ──► T-TL-5.3 Asset Browser
                                      │
                                      └──► T-TL-5.4 Thumbnail Cache
Phase 4 (Debug Integration)
  │
  └──► T-TL-5.5 Profiler Overlay Panel
```

## Implementation Order

1. T-TL-5.3: Asset Browser panel (filesystem tree + file type filters)
2. T-TL-5.4: Thumbnail cache (GPU generation + LRU store)
3. T-TL-5.1: Material DSL compiler (Python parser -> AST -> WGSL)
4. T-TL-5.2: Material Editor panel (text editor + preview + params)
5. T-TL-5.6: Material library (save/load from asset library)
6. T-TL-5.7: Material node graph visualization (read-only initially)
7. T-TL-5.5: Profiler overlay panel (sparkline + pass breakdown)

## Success Criteria

- Material DSL compiles to WGSL with naga validation
- Material Editor renders preview sphere with live parameter updates
- Asset Browser lists files with thumbnails and search
- Thumbnails generate on import with LRU cache eviction
- Profiler Overlay shows real-time frame timing and pass breakdown
- Materials can be saved to and loaded from the asset library
- Node graph visualizes material as editable directed graph
