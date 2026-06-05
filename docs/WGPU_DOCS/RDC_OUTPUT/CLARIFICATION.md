# CLARIFICATION - Philosophical Framing

**Purpose:** Meta-context and design philosophy for TRINITY's wgpu implementation
**Document Type:** Non-task guidance for architects and developers

---

## Core Principle

> **TRINITY implements the complete wgpu API surface. Phase gates are scheduling constraints, not architectural boundaries. The architecture is designed for the complete picture from day one.**

This principle means:

1. **No shortcuts** - Even if Phase 1 only implements device initialization, the code structure anticipates ray tracing, bindless, and mesh shaders.

2. **No refactoring later** - Module boundaries, trait abstractions, and data structures are designed for the final system.

3. **Capability tiers, not feature flags** - The system degrades gracefully via capability tiers (Minimal/Standard/Advanced/Full), not scattered `#[cfg]` blocks.

---

## Why wgpu?

TRINITY chose wgpu for GPU abstraction because:

1. **Rust-native** - Memory safety, concurrency, and performance without C++ baggage.

2. **Cross-platform** - Single API targets Vulkan, Metal, DX12, WebGPU, OpenGL.

3. **WebGPU alignment** - The web is a first-class target, not an afterthought.

4. **Active development** - wgpu evolves rapidly; ray tracing support is imminent.

5. **No vendor lock-in** - Unlike DirectX-only or Metal-only engines.

---

## What is a "Phase"?

Phases are **scheduling constraints** that answer: "What order should we build this?"

Phases are **not**:
- Separate projects
- Optional features
- Independent modules

All phases produce code that lives in a single, cohesive renderer. Phase 7 code calls Phase 1 code; Phase 3 code uses Phase 2 resources.

**Phase duration estimates assume:**
- Single developer
- Full-time focus
- Existing familiarity with GPU programming
- Access to all target hardware

---

## Capability Tiers Explained

| Tier | What it means |
|------|---------------|
| **Minimal** | WebGL2-class hardware. Forward rendering only. No compute. |
| **Standard** | Desktop GL 4.5 / DX11. Deferred rendering, compute, 8K textures. |
| **Advanced** | DX12 / Vulkan 1.2. Bindless, multi-draw indirect, GPU culling. |
| **Full** | RTX/RDNA2+. Ray tracing pipeline, ray query, full feature set. |

**Tier detection happens once** at startup. The renderer selects a render path and never re-checks.

**Graceful degradation means:**
- Full tier unavailable -> use Advanced tier render path
- Advanced unavailable -> use Standard tier
- Standard unavailable -> use Minimal tier

**Graceful degradation does NOT mean:**
- Runtime feature checks scattered through code
- Fallback implementations in hot paths
- Different code paths for same visual output

---

## The Frame Graph: Why?

The frame graph is TRINITY's answer to "how do we manage rendering complexity?"

**Without a frame graph:**
- Manual barrier management
- Explicit resource lifetime tracking
- Ad-hoc pass ordering
- Memory leaks from forgotten resources

**With a frame graph:**
- Declare resources, declare passes, execute
- Barriers computed automatically
- Transient resources pooled automatically
- Pass ordering from dependency analysis
- Resource aliasing for memory efficiency

The frame graph is **not optional**. All rendering goes through it.

---

## Python Bridge: Why?

The Python bridge answers: "How do we iterate quickly?"

**Use Python for:**
- Rapid prototyping of effects
- Scene setup and testing
- Tooling and automation
- Non-performance-critical rendering

**Use Rust for:**
- Hot paths
- Per-frame operations
- GPU resource management
- Production rendering

The Python API mirrors the Rust API but with ergonomic Python patterns (builders, dataclasses, exceptions).

---

## What This Document Set Does NOT Include

1. **Asset pipeline** - Mesh/texture loading is out of scope (handled by `engine/`).

2. **Scene graph** - Object hierarchy is out of scope (handled by Python frontend).

3. **Physics** - Collision/dynamics is out of scope.

4. **Audio** - Sound is out of scope.

5. **Networking** - Multiplayer is out of scope.

This document set covers the **renderer backend** only: the code in `crates/renderer-backend/`.

---

## Relationship to GAPSET

The GAPSET system tracks implementation status at a finer granularity:

- **GAPSET_1**: Frame Graph (corresponds to Phase 7.5)
- **GAPSET_6**: GI/Reflections (corresponds to Phase 5.9)
- **GAPSET_9**: Ray Tracing (corresponds to Phase 5)
- **GAPSET_12**: RHI (corresponds to Phases 1-4)

Task IDs in this document (T-WGPU-P*) are **not** GAPSET task IDs. They are internal to the wgpu implementation plan and will be mapped to GAPSET tasks during execution.

---

## Experimental vs Stable

| Feature | wgpu Status | TRINITY Approach |
|---------|-------------|------------------|
| Core rendering | Stable | Ship it |
| Compute | Stable | Ship it |
| Ray Query | Stable | Ship it |
| RT Pipeline | Experimental | Feature-gated, prepare code |
| Mesh Shaders | Not in wgpu | Prepare abstraction, wait |
| OMM/DMM/SER | Not in wgpu | Document for future |

**Experimental features** are behind `#[cfg(feature = "experimental")]`. They compile but are not enabled by default.

**Future features** (mesh shaders) have abstraction layers ready but no implementation.

---

## Error Handling Philosophy

1. **Validation at boundaries** - Check inputs at API boundaries, not deep in code.

2. **Result for recoverable** - Device lost, out of memory -> return Result.

3. **Panic for bugs** - Invalid state machine transition -> panic (debug builds).

4. **Log for warnings** - Suboptimal usage -> log warning, continue.

5. **Python exceptions** - All Rust errors become Python exceptions.

---

## Performance Philosophy

1. **Measure first** - No optimization without profiling data.

2. **GPU is the bottleneck** - CPU overhead is acceptable if GPU is saturated.

3. **Batch everything** - Draw calls, buffer uploads, command submissions.

4. **Cache aggressively** - Pipelines, layouts, samplers, shaders.

5. **Pool resources** - Buffers, textures, transient allocations.

---

## Testing Philosophy

1. **Unit tests for logic** - AABB intersection, hash functions, state machines.

2. **Integration tests for APIs** - Create resource, use resource, destroy resource.

3. **Visual tests for correctness** - Reference images, comparison thresholds.

4. **Fuzzing for robustness** - Random descriptor permutations (future).

5. **Platform matrix in CI** - Vulkan, Metal, DX12, WebGPU.

---

## Glossary

| Term | Definition |
|------|------------|
| **RHI** | Render Hardware Interface (wgpu abstraction) |
| **PSO** | Pipeline State Object |
| **BLAS** | Bottom-Level Acceleration Structure |
| **TLAS** | Top-Level Acceleration Structure |
| **SBT** | Shader Binding Table |
| **HiZ** | Hierarchical Z-buffer |
| **LOD** | Level of Detail |
| **MSAA** | Multisample Anti-Aliasing |
| **MRT** | Multiple Render Targets |
| **GI** | Global Illumination |
| **AO** | Ambient Occlusion |

---

## Final Note

This document set represents approximately 1000 hours of implementation work. The estimates are aggressive but achievable with focused effort.

The goal is not perfection on the first pass. The goal is a complete, working system that can be iterated upon.

**Ship early. Ship often. Ship correctly.**

---

*End of CLARIFICATION.md*
