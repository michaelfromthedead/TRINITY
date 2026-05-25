# GAPSET 7: Post-Processing Stack -- Clarification Document

**Date**: 2026-05-22
**Context**: RDC analysis of GAPSET_7_POST_PROCESS comparing the planned 70 tasks against actual source files on disk.

---

## Critical Ambiguities Requiring Resolution

### 1. Rust vs Python -- Which Path Is Authoritative?

**Issue**: The codebase has two independent post-process implementations:
- Rust `post_process.rs` -- builds frame graph IR pass nodes (pass planning)
- Python `engine/rendering/postprocess/` -- full OOP hierarchy (effect execution)

**These paths are not connected.** The Rust chain builder (`create_post_process_chain`) produces tonemap->bloom->TAA passes. The Python stack (`PostProcessStackExecutor`) produces 9+ effects in priority order. Neither calls the other.

**Question**: Should the post-process pipeline be:
- (a) Rust-only: effect dispatch logic in Rust frame graph passes, Python removed
- (b) Python-only: Python drives all effect logic, Rust frame graph is just a thin dispatch layer
- (c) Hybrid: Python manages effect config/ordering, Rust executes the passes
- (d) Python reference, Rust production: Python provides CPU reference implementations that Rust shaders should match

**Recommendation**: Option (d) aligns with the existing architecture best -- Python CPU reference implementations verify correctness, Rust/WGSL shaders provide GPU performance.

---

### 2. Missing Effects -- Implementation or Removal?

**Issue**: Four effects listed in the plan have zero code anywhere:
- Vignette (T-PP-1.6, T-PP-1.7)
- Chromatic Aberration (T-PP-2.4, T-PP-2.4a, T-PP-2.4b, T-PP-2.5a)
- Lens Flare (T-PP-2.3, T-PP-2.3a, T-PP-2.3b)
- Film Grain (T-PP-3.5, T-PP-3.5a)

No Python module, no Rust function, no WGSL shader, no settings reference, no post-process code reference.

**Question**: Should these 4 effects be:
- (a) Implemented from scratch (12 tasks, ~40% of missing work)
- (b) Removed from scope (adjust plan and quality presets)
- (c) Deferred to a GAPSET 8 (future post-process additions)

**Recommendation**: Option (c) -- defer to a follow-up gap set. The 4 missing effects are relatively simple pixel shaders that do not block the core pipeline.

---

### 3. WGSL Shader Location and Naming Convention

**Issue**: Zero WGSL shaders exist for any post-process effect. The plan calls for shaders but does not specify:
- Where shaders should live (new `shaders/postprocess/` directory?)
- Naming convention (e.g., `tonemap.wgsl`, `bloom_downsample.wgsl`)
- Uniform buffer binding convention
- Workgroup size convention (16x16, 8x8, or dynamic)
- How shaders map to Rust pass dispatch (`DispatchSource::Direct` with hardcoded sizes)

**Question**: What is the shader convention to adopt?

**Recommendation**: 
- `shaders/postprocess/` directory
- `<effect>_<stage>.wgsl` naming (e.g., `tonemap_aces.wgsl`, `bloom_downsample.wgsl`)
- 16x16 workgroups standard, 8x8 for blur passes
- Push constants for effect parameters where possible
- Storage buffers for read/write textures

---

### 4. TSR vs Traditional Upscaling

**Issue**: The plan calls for TSR (Temporal Super Resolution) with Lanczos upsampling (T-PP-6.3 through T-PP-6.4). TSR does not exist in any form -- no module, no reference, no plan for implementation beyond one paragraph.

Meanwhile, the codebase already has:
- `SpatialUpscaler` ABC (bilinear, FSR1, CAS)
- `TemporalUpscaler` ABC (FSR2, DLSS, XeSS)
- Auto-detection fallback chain

**Question**: Should TSR be:
- (a) Implemented as a custom temporal upscaler with Lanczos kernels (as planned)
- (b) Removed in favor of the existing upscaler ABC hierarchy (FSR2 provides similar quality)
- (c) Implemented only as a fallback when DLSS/XeSS/FSR2 are unavailable

**Recommendation**: Option (c) -- TSR makes sense as a last-resort fallback when no vendor SDK is available. The existing upscaler ABC hierarchy already provides this pattern.

---

### 5. Integration Test Framework

**Issue**: The plan specifies 6 integration test phases (T-PP-1.7, T-PP-2.5, T-PP-3.6, T-PP-4.4, T-PP-5.3, T-PP-6.6). Zero integration tests exist. The Rust unit tests (22 tests) only verify pass wiring, not pixel output.

**Question**: What test framework and methodology?
- (a) Snapshot-based: render reference images, compare pixel-by-pixel
- (b) Property-based: verify invariants (e.g., occlusion 0-1, tonemap preserves hue)
- (c) Comparative: run Python CPU reference against shader output, check delta < threshold
- (d) All of the above, tiered by phase

**Recommendation**: Option (d) -- each effect should have:
- Unit tests for parameter ranges and edge cases
- CPU reference test via Python reference implementation
- GPU pixel-output test against CPU reference (once shaders exist)
- Per-quality-preset budget validation

---

## Minor Clarifications

### 6. Quality Preset Effect List Mismatch

Some effects in quality presets are not actually implemented or the preset references effects that don't exist:
- Low preset includes DOF but DOF blur methods return None
- Medium preset includes Motion Blur but apply returns input unchanged
- Ultra preset references SMAA which is a stub returning None
- Lens dirt texture references in bloom settings but no lens flare module

### 7. Constants Bug: HISTOGRAM_BINS_MAX

`constants.py` defines `HISTOGRAM_BINS_MAX = 256` but Ultra quality preset specifies `bin_count=512`. This will silently cap the histogram to 256 bins, reducing auto-exposure precision on Ultra settings. The constant or the preset must change.

### 8. Rust Resource Handle Collision Risk

The Rust `create_post_process_chain()` uses hardcoded `ResourceHandle(0xFF00)`, `ResourceHandle(0xFF01)`, `ResourceHandle(0xFF02)` for transient resources. If the application uses handles in the 0xFF00-0xFF02 range, there will be silent handle collisions. These should use a dynamic allocation strategy or a higher reserved range.

### 9. No Performance Budget Infrastructure

The plan specifies per-effect performance budgets (e.g., bloom ~0.32ms, DOF ~0.35-1.05ms). No infrastructure exists to measure, track, or validate these budgets. No GPU timestamp queries, no CPU profiling hooks, no CI gates for regressions.
