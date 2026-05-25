# CLARIFICATION.md — GAPSET_9_RAY_TRACING

> **Purpose**: Corrections and clarifications found during RDC (2026-05-22).
> **Methodology**: Compared PHASE_N_TODO.md claims against actual source files on disk.

---

## Critical Corrections

### C1: T-RT-P1.1 is PARTIALLY DONE, not "not started"

**TODO claim**: "Define Python ABI stubs for BLAS/TLAS management" -- unchecked.

**Reality**: `BLASDesc` (`raytracing.py:20-28`) and `TLASDesc` (`raytracing.py:30-36`) dataclasses exist with all specified fields. The `AccelerationStructure` ABC (`raytracing.py:38-63`) and `NullAccelerationStructure` (`raytracing.py:65-98`) exist. `BLASManager`, `TLASManager`, `BLASPool` do NOT exist.

**Correction**: Task status [~] partial. 3 of 8 acceptance criteria met (BLASDesc, TLASDesc, BuildFlags).

### C2: T-RT-P1.4 is PARTIALLY DONE, not "not started"

**TODO claim**: "Implement RTCapability Detection" -- unchecked.

**Reality**: `FeatureSupport.ray_tracing` exists as a `bool` in `device.py`. However, it is a simple boolean (True for DISCRETE, False for INTEGRATED), not the 3-level `RTCapability` enum (NONE/RAY_QUERY_ONLY/FULL) described in the spec. No `get_rt_capability()` function exists.

**Correction**: Task status [~] partial. Basic feature detection exists but is not a proper capability enum.

### C3: T-RT-P1.7 is PARTIALLY DONE, not "not started"

**TODO claim**: "Implement Python shadow ray dispatch" -- unchecked.

**Reality**: The `RayTracingPass` Python class (`pass_node.py:581-696`) exists with dispatch dimensions, TLAS binding, SBT binding, output, and recursion depth -- the full frame graph pass scaffolding. `RTShadows` class does NOT exist.

**Correction**: Task status [~] partial. The pass node exists but the actual shadow dispatch logic is not implemented.

### C4: T-RT-P2.14 is PARTIALLY DONE, not "not started"

**TODO claim**: "Implement S1 frame graph integration for RT passes" -- unchecked.

**Reality**: The `RayTracingPass` node is fully defined in the Python frame graph (`pass_node.py:581-696`), the Rust IR has `PassType::RayTracing` with proper serialization (`frame_graph/mod.rs:94`), and the Serde JSON deserialization handles `"RayTracing"` pass types (`frame_graph/mod.rs:1863`). However, no actual RT effects (shadows, reflections, GI) are wired into the S1 frame graph pipeline.

**Correction**: Task status [~] partial. The frame graph IR scaffolding is complete; the S1 pipeline integration of RT effects is not.

## Minor Corrections

### M1: No WGSL shader files exist

The TODO document implies shaders for T-RT-P1.5, T-RT-P1.6, T-RT-P1.9, T-RT-P2.3, T-RT-P2.4, T-RT-P2.7, T-RT-P2.8, T-RT-P3.1, T-RT-P3.3. A recursive search for `.wgsl`, `.frag`, `.vert`, `.comp` files found zero files anywhere in the repository. The shader compilation pipeline appears to be entirely runtime-compiled via `naga` or not yet connected. All shader tasks are [-] not started.

### M2: Rust RT backend is entirely absent

The TODO document's Phase 1 tasks T-RT-P1.2 and T-RT-P1.3 describe `wgpu::RayTracingAccelerationStructure` usage. A search of all Rust source files (excluding `frame_graph/mod.rs` and `python.rs` which only have pass type IR) found zero references to `AccelerationStructure`, `RayTracing`, `BLAS`, `TLAS`, or `ray_query` in the crates directory. No RT backend crate/module exists.

### M3: T-RT-P2.10 clarification

`BindlessManager` exists at `engine/rendering/gpu_driven/bindless.py`, but it is a general-purpose bindless resource manager. The RT-specific bindless material table with `instance_custom_index` linkage to material data is not implemented.

## Semantic Clarifications

### S1: Phase 1 vs Phase 2 boundary

The TODO correctly identifies that Phase 1 (inline ray queries) is implementable now because wgpu's `ray_query` and `acceleration_structure` features are available in current stable wgpu. Phase 2 (ray tracing pipeline) is correctly gated on `ray_tracing_pipeline` stability. However, the Rust RT backend needed for Phase 1 (BLAS/TLAS build) uses `wgpu::RayTracingAccelerationStructure` which is part of the `acceleration_structure` feature, so it is NOT gated. This boundary is correctly drawn.

### S2: Test coverage reality

The TODO plans 46 unit tests, 10 integration tests, 9 visual tests, and 13 performance tests across all three phases. On disk, only 3 Python tests and 1 Rust test exist, all testing pass node creation. No RT effect tests, no BLAS/TLAS tests, no shader tests, and no fallback chain tests exist.

### S3: No shader compilation pipeline

The TODO assumes WGSL shader files. However, there is no evidence of a shader compilation pipeline (no shader include system, no `naga`-based compilation entry point, no SPIR-V cross-compilation). The RT shader implementation needs to either add WGSL files with a build-time compilation step or implement runtime `naga` compilation. This foundational piece is not addressed in the TODO plan.
