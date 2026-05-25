# PHASE 5 ARCH: 4K/64K Size-Constrained Mode

## Status: NOT IMPLEMENTED

Phase 5 has zero implementation in the main source tree. No files exist for any of the 8 tasks.

## Design Intent (from PHASE_N_TODO.md)

Phase 5 creates standalone executables size-constrained for demoscene competitions:

### 64K mode (T-DEMO-5.1-5.6)
1. Minimal wgpu-rs bootstrap (< 100 lines)
2. Window/presentation layer
3. WGSL shader embedded as string literal in Rust binary
4. Render loop (update, dispatch, present, poll)
5. Build-time DSL compilation pipeline
6. Binary size optimization (strip, LTO, UPX)

### 4K mode (T-DEMO-5.7)
Extreme minimization -- if achievable with WGSL + wgpu-rs runtime overhead.

### Verification (T-DEMO-5.8)
No external dependencies at runtime (no asset files, no Python, no network).

## Prerequisites
Phase 2 (build-time DSL compilation) and Phase 3 (ray marching pipeline).
