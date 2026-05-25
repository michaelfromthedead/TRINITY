# PHASE 8 ARCH: Algorithmic Research and Optimization

## Status: NOT IMPLEMENTATED

Phase 8 has zero implementation in the main source tree. No files exist for any of the 8 tasks.

## Design Intent (from PHASE_N_TODO.md)

Phase 8 covers ongoing research topics for SDF ray marching quality and performance:

| Task | Topic | Description |
|------|-------|-------------|
| T-DEMO-8.1 | Analytic gradients | Per-primitive gradient functions with winner-ID tracking, replacing central differences |
| T-DEMO-8.2 | DSL optimization | Pattern matching, CSE, automatic LOD for distant rays in compiled WGSL |
| T-DEMO-8.3 | Fractal SDF bounding | Distance estimation ratio bailout, step count limits for Mandelbulb/KIFS |
| T-DEMO-8.4 | Importance-driven SDF | Adaptive step count based on gradient magnitude |
| T-DEMO-8.5 | TAA for ray marching | World-space reprojection, no motion vectors needed |
| T-DEMO-8.6 | Automatic LOD | Reduced iterations + simplified approximations with distance |
| T-DEMO-8.7 | Bidirectional ray marching | SSS/translucency through thin geometry |
| T-DEMO-8.8 | DSL recompilation | Incremental compilation strategies for WGSL |

## Prerequisites

All of Phases 1-6 must exist for research targets to be meaningful.
