# PHASE_3_ARCH.md — GPU Foundation Architecture

## Overview

Phase 3 establishes the GPU rendering infrastructure for the editor viewport: wgpu-egui integration for the 3D viewport, multi-viewport layout (4-up), gizmo rendering, GPU picking, scene rendering, GPU query timestamps, and 14 debug visualization passes. This phase is blocked on external gap sets S1-S9 (GPU Pipeline) and S10+ (Frame Graph).

## Current State

| Task | Status | What Exists | What's Missing |
|------|--------|-------------|----------------|
| T-TL-3.1 | [-] NOT STARTED | Nothing | wgpu-egui integration, surface, swapchain |
| T-TL-3.2 | [~] PARTIAL | Python viewport.py | Rust 4-up layout |
| T-TL-3.3 | [~] PARTIAL | Python gizmos.py | Rust egui gizmo widget |
| T-TL-3.4 | [-] NOT STARTED | Nothing | GPU picking pipeline |
| T-TL-3.5 | [-] NOT STARTED | gpu_driven module exists | Viewport scene rendering |
| T-TL-3.6 | [-] NOT STARTED | Python gpu_profiler.py | wgpu QuerySet management |
| T-TL-3.7 | [~] PARTIAL | Python debug_overlays.py | 14 wgpu debug passes |
| T-TL-3.8 | [~] PARTIAL | Python debug_overlays.py API | Rust DebugOverlayController |
| T-TL-3.9 | [-] NOT STARTED | Nothing | Shadow map visualization |

## Architecture: wgpu-egui Viewport (Gap)

This is the core infrastructure that all viewport features depend on.

```
┌──────────────────────────────────────────────────────────┐
│                Tauri Window (wgpu Surface)                 │
│                                                            │
│  ┌──────────────────────────────────────────────────┐     │
│  │  egui Frame                                      │     │
│  │  ┌──────────────┐ ┌──────────────┐              │     │
│  │  │  Panel Sidebar │ │  Viewport    │              │     │
│  │  │  (Inspector)   │ │  (egui widget)             │     │
│  │  │               │ │              │              │     │
│  │  │  ▸ Transform  │ │  ┌────────┐ │              │     │
│  │  │  ▸ Mesh       │ │  │ scene  │ │              │     │
│  │  │               │ │  │ render │ │              │     │
│  │  └──────────────┘ │  └────────┘ │              │     │
│  │                   │  [F] [T] [R] │              │     │
│  └───────────────────┴──────────────┘              │     │
│                                                            │
│  wgpu Surface (shared context)                             │
│  ├── egui paints UI on top                                 │
│  └── Viewport renders scene below                          │
└──────────────────────────────────────────────────────────┘
```

**wgpu-egui integration flow:**

```rust
// Pseudocode for viewport setup
struct EditorRenderer {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    egui_renderer: egui_wgpu::Renderer,
    egui_state: egui::Context,
}

impl EditorRenderer {
    fn new(window: &Window, instance: &wgpu::Instance) -> Self { ... }

    fn render(&mut self, scene_view: &SceneView) {
        let frame = self.surface.get_current_texture().unwrap();
        let view = frame.texture.create_view(&Default::default());

        // 1. Render scene to viewport texture
        scene_view.render(&self.device, &mut self.queue, &view);

        // 2. Render egui overlay on top
        let egui_output = self.egui_state.run(/* ... */);
        let paint_jobs = self.egui_state.tessellate(egui_output.shapes);
        self.egui_renderer.update_buffers(/* ... */);
        self.egui_renderer.render(&mut encoder, &view, /* ... */);

        frame.present();
    }
}
```

## Architecture: 4-Up Viewport Layout (Gap)

```
┌──────────────────────────────────────────────┐
│  4-Up Viewport Layout                        │
│                                              │
│  ┌─────────────────┬─────────────────┐       │
│  │  Perspective    │  Top (Y)        │       │
│  │  (3D orbit)     │  (orthographic) │       │
│  │                 │                 │       │
│  │  [selected ent] │  grid + axes    │       │
│  ├─────────────────┼─────────────────┤       │
│  │  Front (Z)      │  Side (X)       │       │
│  │  (orthographic) │  (orthographic) │       │
│  │                 │                 │       │
│  │  grid + axes    │  grid + axes    │       │
│  └─────────────────┴─────────────────┘       │
│                                              │
│  Each viewport has:                          │
│  ├── Dedicated EditorCamera                  │
│  ├── Grid overlay                            │
│  ├── Axis indicator                          │
│  └── Selection highlight                     │
└──────────────────────────────────────────────┘
```

## Architecture: Gizmo Rendering (Gap)

Gizmos are 2D projected handles overlaid on the 3D viewport.

```
┌──────────────────────────────────────────┐
│  Gizmo Renderer                          │
│                                          │
│  Translate gizmo:                        │
│    ┌───┐  Y (green)                     │
│    │   │                                 │
│    └───┼─── X (red)                     │
│         \                                │
│          Z (blue)                        │
│                                          │
│  Modes:                                  │
│  ├── Translate (arrow cones)             │
│  ├── Rotate (torus arcs)                 │
│  └── Scale (cube ends)                   │
│                                          │
│  Interaction:                            │
│  ├── Mouse hover → highlight axis        │
│  ├── Mouse drag → constrain to axis      │
│  └── Keyboard shortcut cycling           │
│     (W=translate, E=rotate, R=scale)     │
│                                          │
│  Screen-space projected from 3D:         │
│  ├── Project entity position to screen   │
│  ├── Render handles in screen space      │
│  └── Ensure constant screen-size         │
└──────────────────────────────────────────┘
```

## Architecture: GPU Picking (Gap)

```
┌──────────────────────────────────────────────┐
│  GPU Picking Pipeline                         │
│                                               │
│  Per-frame:                                   │
│  ┌──────────────────────────────────────┐     │
│  │ 1. Render all entities to offscreen  │     │
│  │    texture with unique color IDs     │     │
│  │    (entity_id encoded as RGBA)       │     │
│  ├──────────────────────────────────────┤     │
│  │ 2. On mouse click, readback single   │     │
│  │    pixel at cursor position          │     │
│  ├──────────────────────────────────────┤     │
│  │ 3. Decode RGBA → entity_id          │     │
│  ├──────────────────────────────────────┤     │
│  │ 4. Set selected_entity in EditorState│     │
│  └──────────────────────────────────────┘     │
│                                               │
│  Optimizations:                               │
│  ├── Render to 1x1 texture for readback       │
│  ├── Deferred readback (non-blocking)         │
│  └── Cache pick texture across frames         │
└──────────────────────────────────────────────┘
```

## Architecture: GPUQueryManager (Gap)

```
┌──────────────────────────────────────────────────┐
│  GPUQueryManager (wgpu QuerySet pool)             │
│                                                   │
│  struct GPUQueryManager {                         │
│      pool: Vec<wgpu::QuerySet>,     // TIMESTAMP  │
│      free: Vec<usize>,              // available  │
│      pending_readback: VecDeque<PendingQuery>,    │
│      max_pool_size: usize,          // 32 default │
│  }                                                │
│                                                   │
│  Methods:                                         │
│  ├── begin_pass(name) -> QueryId                  │
│  ├── end_pass(id)                                 │
│  ├── write_timestamp(label)                       │
│  └── collect_results() -> Vec<PassTiming>         │
│                                                   │
│  PassTiming {                                     │
│      name: String,                                │
│      gpu_start_ns: u64,                           │
│      gpu_end_ns: u64,                             │
│      gpu_duration_ns: u64,                        │
│  }                                                │
│                                                   │
│  FrameProfile (300-frame ring buffer):            │
│  ├── frame_index: u64                             │
│  ├── frame_type: FrameType (normal|shadow|... )   │
│  ├── passes: Vec<PassTiming>                      │
│  ├── cpu_duration_ns: u64                         │
│  └── gpu_duration_ns: u64                         │
└──────────────────────────────────────────────────┘
```

## Architecture: Debug Visualization Passes (14 Passes)

```
┌──────────────────────────────────────────────┐
│  DebugVisualizationManager                    │
│                                               │
│  GBuffer Views (7):                          │
│  ├── 1. Depth buffer                         │
│  ├── 2. Normals                              │
│  ├── 3. Albedo                               │
│  ├── 4. Metallic                             │
│  ├── 5. Roughness                            │
│  ├── 6. Ambient Occlusion                    │
│  └── 7. Velocity                             │
│                                               │
│  Lighting Views (3):                         │
│  ├── 8. Direct light contribution            │
│  ├── 9. Indirect light (GI)                  │
│  └── 10. Shadow maps                         │
│                                               │
│  Post-FX Views (3):                          │
│  ├── 11. Bloom                               │
│  ├── 12. SSAO                                │
│  └── 13. SSR                                 │
│                                               │
│  Overlay (1):                                │
│  └── 14. Wireframe                           │
│                                               │
│  API:                                         │
│  ├── set_visualization(id) -> set active view │
│  ├── add_visualization(id) -> composite view  │
│  ├── remove_visualization(id)                 │
│  ├── clear() -> return to normal rendering    │
│  ├── set_heatmap_scale(scale)                 │
│  └── set_opacity(opacity)                     │
│                                               │
│  Hotkeys:                                     │
│  ├── F2: toggle debug visualization           │
│  ├── F3: cycle through visualizations         │
│  ├── F4: toggle HUD overlay                   │
│  └── F5: toggle waterfall timeline            │
└──────────────────────────────────────────────┘
```

## Dependency Chain

```
Phase 1 (EguiUIContext)
  │
  ▼
S1-S9 GPU Pipeline (external)
  │  Provides: wgpu device, queue, surface, shaders
  ▼
T-TL-3.1 wgpu-egui integration
  │
  ├──► T-TL-3.2 4-up layout
  │     ├──► T-TL-3.3 Gizmo rendering
  │     ├──► T-TL-3.4 GPU picking
  │     └──► T-TL-3.5 Scene rendering ──► S10+ Frame Graph
  │
  └──► T-TL-3.6 GPUQueryManager ──► T-TL-3.7 Debug passes
                                        │
                                        └──► T-TL-3.8 DebugOverlayController
                                              └──► T-TL-3.9 Shadow viz
```

## Implementation Order (after S1-S9 available)

1. T-TL-3.1: wgpu-egui surface creation, resize handling, context sharing
2. T-TL-3.2: 4-up viewport layout with orthographic/perspective cameras
3. T-TL-3.6: GPUQueryManager with QuerySet pool and deferred readback
4. T-TL-3.4: GPU picking (render entity IDs, readback, resolve)
5. T-TL-3.3: Gizmo rendering (translate/rotate/scale handles)
6. T-TL-3.5: Scene rendering in viewport (submit existing draw calls)
7. T-TL-3.7: Debug visualization passes (14 passes, incremental)
8. T-TL-3.8: DebugOverlayController API
9. T-TL-3.9: Shadow map visualization

## Success Criteria

- 3D viewport renders scene with correct perspective
- 4-up layout shows Perspective + Top + Front + Side views
- Orbit/pan/zoom works in perspective viewport
- GPU picking selects entity under mouse cursor
- Gizmo handles render and allow axis-constrained manipulation
- GPUQueryManager returns valid GPU timing data
- Debug visualization passes toggle between GBuffer/lighting/post-fx views
- All four viewport hotkeys (F2-F5) function correctly
