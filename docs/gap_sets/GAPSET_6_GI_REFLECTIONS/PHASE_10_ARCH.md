# Phase 10: GI Visualization -- Architecture

## Overview

Phase 10 implements debug visualization overlays for all GI and reflection techniques: probe grid visualization, voxel wireframe, SSGI heatmap, and technique mask display.

## Tasks

| ID | Status | Description |
|----|--------|-------------|
| T-GIR-P10.1 | [-] | Implement GI debug visualization overlay |

## Current State: NOT BUILT

No GI visualization code exists anywhere in the codebase.

## Required Architecture

### GI Debug Visualization (T-GIR-P10.1)

Design for a comprehensive debug overlay system:

```
Visualization Modes (selectable via debug menu):

1. DDGI Probe Grid Overlay:
   - Render probe positions as 3D spheres/points
   - Color probes by: SH dominant direction, probe activity, update frequency
   - Show probe grid cell boundaries as wireframe
   - Display probe visibility weights per pixel

2. SSGI Heatmap:
   - Overlay SSGI contribution as false-color heatmap
   - Blue = low indirect, Red = high indirect
   - Show ray march steps as line overlay
   - Display hit confidence per pixel

3. SSR Visualizations:
   - Overlay SSR trace results with highlighted ray paths
   - Show screen-space hit points vs. miss points (green = hit, red = miss)
   - Display roughness-driven mip selection overlay
   - Show temporal reprojection confidence

4. RT Reflection Overlay:
   - Show ray count per pixel
   - Display fallback technique selection (color-coded by technique)
   - Denoiser variance heatmap
   - TLAS instance bounding boxes

5. Voxel GI:
   - Render voxel grid wireframe
   - Color voxels by: occupancy, albedo, normal direction
   - Show cone trace paths as animated lines

6. Probe Visualization:
   - Display reflection probe bounds as wireframe AABBs
   - Show probe blend regions with gradient
   - Render cubemap face selections

7. Technique Mask:
   - Per-pixel color coding of active GI technique:
     * Red = DDGI only
     * Green = SSGI only
     * Blue = SSR only
     * Yellow = Voxel GI
     * Cyan = RT Reflections
     * White = Full combination
     * Black = No GI
```

### Implementation Approach

```
Pass structure:
1. Debug state uniform: selected visualization mode, probe index, intensity scale
2. Separate compute pass or fragment post-process pass
3. Render overlay on top of final lit scene
4. Optional: render to separate debug buffer for UI integration

Controls:
- Keyboard shortcuts (F5-F12 for different modes)
- Inspector panel for probe/technique details
- Mouse picking for probe/voxel inspection
```

## Dependencies

- T-GIR-P10.1: T-GIR-P2.1 (probe grid debug), T-GIR-P3.1 (SSGI debug), T-GIR-P8.5 (fallback mask)

## Files to Create

| File | Purpose |
|------|---------|
| `crates/renderer-backend/src/gi_visualization.rs` | Visualization state, modes, controls |
| `crates/renderer-backend/shaders/gi_debug_overlay.wgsl` | Debug overlay rendering |

## Acceptance Criteria (All Failing)

| Criterion | Status |
|-----------|--------|
| Probe grid overlay shows probe positions and states | Failing -- not built |
| SSGI heatmap shows indirect contribution | Failing -- not built |
| Technique mask shows active GI technique per pixel | Failing -- not built |
| Debug overlay <1ms overhead | Failing -- not built |
| Visualizations support debug menu integration | Failing -- not built |
