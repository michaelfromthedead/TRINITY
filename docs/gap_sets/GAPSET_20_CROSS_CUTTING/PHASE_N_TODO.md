# GAPSET 20: Cross-Cutting Concerns -- Task List

> **Status**: Planning Document
> **Gap Set**: GAPSET_20_CROSS_CUTTING
> **RDC_WORKFLOW**: CLUSTER_CONSOLIDATOR -> SDLC_WORKFLOW
> **Task ID Format**: T-CC-{PHASE}.{N}
> **Total Tasks**: 46
> **Covers**: Quality tiers, mobile fallback, hot-reload, determinism, Holy Grails tracking, risk mitigation

---

## Phase 0: Foundation Tasks (T-CC-0.x)

### Quality Tier System

- [ ] **T-CC-0.1**: Define QualityTier enum (Low, Medium, High, Ultra) in trinity_core config
  - Acceptance: Enum defined with tier ordering and numeric scoring
  - Dependencies: None (Phase 0 prerequisite)
  - Effort: Small (hours)

- [ ] **T-CC-0.2**: Implement capability scoring from wgpu adapter information
  - Acceptance: Adapter query produces capability score 0.0-1.0
  - Dependencies: T-CC-0.1
  - Effort: Small (1 day)

- [ ] **T-CC-0.3**: Build QualityManager with tier selection, per-subsystem overrides, dynamic tier adjustment
  - Acceptance: Manager selects correct tier based on hardware, frame budget monitoring works
  - Dependencies: T-CC-0.1, T-CC-0.2
  - Effort: Medium (3 days)

- [ ] **T-CC-0.4**: Define QualityCapabilities trait for all subsystems
  - Acceptance: Trait defined with tier_features, tier_budget, tier_resolution, fallback_chain
  - Dependencies: T-CC-0.3
  - Effort: Small (1 day)

- [ ] **T-CC-0.5**: Create tier feature configuration for S3 (Materials), S4 (Lighting), S5 (Shadows), S6 (GI), S8 (Post-Processing), S11 (Atmosphere)
  - Acceptance: Each subsystem declares feature set per tier
  - Dependencies: T-CC-0.4
  - Effort: Medium (3 days)

- [ ] **T-CC-0.6**: Create tier feature configuration for S7 (Reflections), S9 (Particles), S10 (RT), S12 (Water/Terrain), S13 (Demoscene)
  - Acceptance: Each subsystem declares feature set per tier and fallback chain
  - Dependencies: T-CC-0.4
  - Effort: Medium (3 days)

- [ ] **T-CC-0.7**: Implement shader variant pruning based on quality tier
  - Acceptance: Only variants needed for selected tier are compiled (390 total vs. 550+)
  - Dependencies: T-CC-0.5, T-CC-0.6
  - Effort: Large (1 week)

- [ ] **T-CC-0.8**: Implement dynamic tier adjustment (frame budget monitoring -> auto tier change)
  - Acceptance: When frame budget exceeds 120% for 10+ consecutive frames, tier drops automatically
  - Dependencies: T-CC-0.3
  - Effort: Medium (3 days)

### Mobile/Metal/GLES Fallback

- [ ] **T-CC-0.9**: Implement GLES 3.1 capability detection and feature workarounds
  - Acceptance: Compute-less rendering path works; workarounds documented
  - Dependencies: T-CC-0.1, T-CC-0.2
  - Effort: Medium (3 days)

- [ ] **T-CC-0.10**: Build Forward+ renderer for Low tier (no deferred path)
  - Acceptance: Depth pre-pass, light culling in vertex shader, forward shading, tone map
  - Dependencies: S1 Frame Graph + S14 RHI (from GAPSET_1, GAPSET_14)
  - Effort: Large (1 week)

- [ ] **T-CC-0.11**: Implement Low-tier memory budget (max 256MB GPU memory)
  - Acceptance: Max texture 1024x1024, max RT 1280x720, ETC2/ASTC textures, reduced draw calls
  - Dependencies: T-CC-0.10
  - Effort: Medium (3 days)

- [ ] **T-CC-0.12**: Create Metal-specific rendering path optimizations (TBDR, argument buffers)
  - Acceptance: Metal backend with TBDR-aware rendering, unified memory optimization
  - Dependencies: S14 RHI
  - Effort: Medium (3 days)

- [ ] **T-CC-0.13**: Implement fallback selection logic (startup capability check -> tier assignment)
  - Acceptance: Engine selects correct tier + fallback chain on startup via wgpu adapter query
  - Dependencies: T-CC-0.3, T-CC-0.9
  - Effort: Small (1 day)

### Determinism Foundation

- [ ] **T-CC-0.14**: Implement Fixed16 Q8.8 and Fixed32 Q16.16 math types
  - Acceptance: Types defined with arithmetic operators, conversions to/from float
  - Dependencies: S15 math.rs (from GAPSET_15)
  - Effort: Medium (3 days)

- [ ] **T-CC-0.15**: Implement PCG-based deterministic RNG
  - Acceptance: PCG RNG with seed chaining; same seed produces same sequence
  - Dependencies: T-CC-0.14
  - Effort: Small (1 day)

- [ ] **T-CC-0.16**: Implement deterministic command buffer for component writes
  - Acceptance: All component writes through ordered buffer; replay produces identical state
  - Dependencies: S15 ComponentStore (from GAPSET_15)
  - Effort: Medium (3 days)

- [ ] **T-CC-0.17**: Build 13-phase tick scheduler skeleton
  - Acceptance: Scheduler executes phases in fixed order with fixed timestep
  - Dependencies: T-CC-0.15, T-CC-0.16
  - Effort: Medium (3 days)

- [ ] **T-CC-0.18**: Implement tick checksumming for replay verification (optional)
  - Acceptance: Each tick produces a checksum of deterministic state; replay verification works
  - Dependencies: T-CC-0.17
  - Effort: Small (1 day)

### Risk Mitigation (Phase 0)

- [ ] **T-CC-0.19**: Build JSON-based bridge channel protocol prototype
  - Acceptance: Type/Data/Command channels functional with JSON wire format
  - Dependencies: S1 Bridge Channel (from GAPSET_1)
  - Effort: Medium (3 days)

- [ ] **T-CC-0.20**: Implement manual Rust frame graph (no Python compiler yet)
  - Acceptance: Frame graph defined in Rust directly, bypassing Python DSL
  - Dependencies: S1 Frame Graph (from GAPSET_1)
  - Effort: Large (1 week)

- [ ] **T-CC-0.21**: Document determinism precision requirements per subsystem
  - Acceptance: Document lists which subsystems need fixed-point and at what precision
  - Dependencies: T-CC-0.14
  - Effort: Small (1 day)

---

## Phase 1: MVP Rendering Tasks (T-CC-1.x)

### Quality Tier Integration in Rendering

- [ ] **T-CC-1.1**: Wire quality tier into S4 Lighting (light count per tier, resolution per tier)
  - Acceptance: Low tier: 8 lights forward; Medium: 64 lights clustered; High: 256; Ultra: unlimited
  - Dependencies: T-CC-0.5, Phase 1 rendering subsystems
  - Effort: Medium (3 days)

- [ ] **T-CC-1.2**: Wire quality tier into S8 Post-Processing (feature set per tier)
  - Acceptance: Low: tonemap + bloom; Medium: +DOF + TAA; High: full; Ultra: +upscaling
  - Dependencies: T-CC-0.5
  - Effort: Medium (2 days)

- [ ] **T-CC-1.3**: Wire quality tier into S3 Materials (variant count per tier)
  - Acceptance: Low: 1 variant; Medium: 3; High: 10; Ultra: all including advanced shading
  - Dependencies: T-CC-0.7
  - Effort: Medium (3 days)

- [ ] **T-CC-1.4**: Wire quality tier into S5 Shadows (shadow resolution, filtering method per tier)
  - Acceptance: Low: PCF 512; Medium: PCF 1024; High: VSM 2048; Ultra: RT shadows
  - Dependencies: T-CC-0.5
  - Effort: Medium (2 days)

### Hot-Reload Level 1

- [ ] **T-CC-1.5**: Implement file watcher for config/data files
  - Acceptance: notify-based file watcher detects changes, triggers reload callback
  - Dependencies: None
  - Effort: Small (1 day)

- [ ] **T-CC-1.6**: Implement config reload callback registration
  - Acceptance: Systems register reload handlers; config changes apply without restart
  - Dependencies: T-CC-1.5
  - Effort: Small (1 day)

### G3 Data-Driven Patterns

- [ ] **T-CC-1.7**: Implement `@data_driven` decorator with schema generation
  - Acceptance: Decorated classes auto-generate JSON schema from type annotations
  - Dependencies: Trinity Pattern (ComponentMeta)
  - Effort: Medium (3 days)

- [ ] **T-CC-1.8**: Implement DataBoundDescriptor for runtime data binding
  - Acceptance: Fields bound to external data sources; changes reflect on instances
  - Dependencies: T-CC-1.7
  - Effort: Medium (3 days)

### G8 Error Propagation

- [ ] **T-CC-1.9**: Implement Result types for all Rust bridge functions
  - Acceptance: All functions return Result<T, Error>; no silent failures
  - Dependencies: S15 Core Systems
  - Effort: Medium (3 days)

- [ ] **T-CC-1.10**: Build error aggregation system with error panel
  - Acceptance: All errors flow to central aggregator; categorized by severity; visible in debug tools
  - Dependencies: T-CC-1.9
  - Effort: Medium (2 days)

---

## Phase 2: Advanced Rendering Tasks (T-CC-2.x)

### Determinism Extension

- [ ] **T-CC-2.1**: Apply Fixed32 to particle system (S9) initial conditions
  - Acceptance: Particle spawn positions/velocities use Fixed32; deterministic simulation
  - Dependencies: T-CC-0.14, S9 Particles
  - Effort: Medium (3 days)

- [ ] **T-CC-2.2**: Apply Fixed32 to water simulation (S12) Gerstner parameters
  - Acceptance: Gerstner wave parameters use Fixed32; deterministic across frames
  - Dependencies: T-CC-0.14, S12 Water/Terrain
  - Effort: Medium (2 days)

- [ ] **T-CC-2.3**: Apply Fixed32 to animation (S14) skeleton blending
  - Acceptance: Skeleton bone positions use Fixed32; bit-identical animation playback
  - Dependencies: T-CC-0.14, S14 Animation
  - Effort: Medium (2 days)

### G4 Unified Serialization

- [ ] **T-CC-2.4**: Implement Serializable trait with schema versioning
  - Acceptance: Trait with serialize/deserialize/schema methods; version detection on load
  - Dependencies: T-CC-1.7 (G3 Data-Driven)
  - Effort: Large (1 week)

- [ ] **T-CC-2.5**: Implement BinaryWriter/Reader + JSONWriter/Reader + DiffWriter/Reader
  - Acceptance: Multiple serialization formats; same interface for all
  - Dependencies: T-CC-2.4
  - Effort: Large (1 week)

- [ ] **T-CC-2.6**: Implement reference handling (EntityID resolution on deserialize)
  - Acceptance: Entity references serialized as IDs; resolved on deserialization; cycle-safe
  - Dependencies: T-CC-2.4, S15 ECS
  - Effort: Medium (3 days)

- [ ] **T-CC-2.7**: Implement partial serialization with SerializationContext
  - Acceptance: Serialization scope controlled by context (root entity, depth, reference follow)
  - Dependencies: T-CC-2.4
  - Effort: Medium (3 days)

- [ ] **T-CC-2.8**: Implement diff-based serialization for undo/network delta
  - Acceptance: DiffWriter produces delta between two snapshots; DiffReader applies delta
  - Dependencies: T-CC-2.5
  - Effort: Medium (3 days)

---

## Phase 3: Tooling Tasks (T-CC-3.x)

### Hot-Reload Level 2-5

- [ ] **T-CC-3.1**: Implement asset hot-reload (Level 2) with handle indirection
  - Acceptance: Textures, meshes, audio reload via handle table; existing handles remain valid
  - Dependencies: S16 Asset Pipeline
  - Effort: Large (1 week)

- [ ] **T-CC-3.2**: Implement shader hot-reload with dependency cascade
  - Acceptance: Shader change recompiles -> updates PSO cache -> all affected materials re-bind
  - Dependencies: T-CC-3.1, S3 Materials
  - Effort: Large (1 week)

- [ ] **T-CC-3.3**: Implement Python script hot-reload (Level 3) with state preservation
  - Acceptance: Script change serializes state, swaps module, deserializes, resumes
  - Dependencies: S18 Editor
  - Effort: Large (1 week)

- [ ] **T-CC-3.4**: Implement Rust native code hot-reload (Level 4) with function table patching
  - Acceptance: DLL/SO hot-swap with function pointer indirection; state migration on reload
  - Dependencies: T-CC-3.3
  - Effort: X-Large (2 weeks)

- [ ] **T-CC-3.5**: Implement schema migration for structural changes (Level 5)
  - Acceptance: ComponentMeta detects schema change, generates migration, applies, validates
  - Dependencies: ComponentMeta, T-CC-2.4 (G4 Serialization)
  - Effort: Large (1 week)

### G5 Debug UI

- [ ] **T-CC-3.6**: Integrate Dear ImGui (egui) into S18 Editor
  - Acceptance: Debug UI framework available; auto-inspector generates UI from TrinityMirror
  - Dependencies: S18 Editor
  - Effort: Medium (3 days)

- [ ] **T-CC-3.7**: Implement `@debuggable` decorator with auto-inspector
  - Acceptance: Decorated components get auto-generated debug UI (sliders, inputs, dropdowns)
  - Dependencies: T-CC-3.6, TrinityMirror
  - Effort: Medium (3 days)

### G6 Frame-Perfect Profiling

- [ ] **T-CC-3.8**: Implement GPU timestamp instrumentation via wgpu query API
  - Acceptance: GPU timestamps for all render passes; results streamed to event capture
  - Dependencies: S17 Debug/Profiling
  - Effort: Medium (3 days)

- [ ] **T-CC-3.9**: Implement event stream with Chrome Tracing format output
  - Acceptance: Profiling events flow to ring buffer; Chrome Tracing viewer compatible
  - Dependencies: T-CC-3.8
  - Effort: Medium (3 days)

- [ ] **T-CC-3.10**: Implement frame budget system with automatic quality adjustment
  - Acceptance: Budget violation triggers auto quality tier reduction
  - Dependencies: T-CC-0.8, T-CC-3.8
  - Effort: Medium (3 days)

### G7 Asset Pipeline

- [ ] **T-CC-3.11**: Implement BLAKE3 content-addressed asset hashing
  - Acceptance: All assets content-addressed via BLAKE3 hash; automatic deduplication
  - Dependencies: S16 Asset Pipeline
  - Effort: Medium (3 days)

- [ ] **T-CC-3.12**: Implement asset dependency graph with rebuild cascade
  - Acceptance: Change to source asset triggers rebuild of all dependent assets
  - Dependencies: T-CC-3.11
  - Effort: Large (1 week)

- [ ] **T-CC-3.13**: Implement distributed asset cache (shared across team)
  - Acceptance: Built assets cached; team members share cache; CI populates
  - Dependencies: T-CC-3.12
  - Effort: Large (1 week)

- [ ] **T-CC-3.14**: Implement platform-specific asset variants
  - Acceptance: Same source produces correct format per platform (BC7, ASTC, ETC2)
  - Dependencies: T-CC-3.11
  - Effort: Medium (3 days)

---

## Phase 4: Premium Tasks (T-CC-4.x)

### G9 Time-Travel Debugging

- [ ] **T-CC-4.1**: Implement snapshot-based time-travel (requires G1 Determinism + G4 Serialization)
  - Acceptance: Step backward through simulation; restore snapshot + replay to target tick
  - Dependencies: T-CC-0.17, T-CC-2.5
  - Effort: Large (1 week)

- [ ] **T-CC-4.2**: Implement conditional breakpoints and value watches in time-travel
  - Acceptance: "Stop when condition becomes true"; binary search for value change time
  - Dependencies: T-CC-4.1
  - Effort: Medium (3 days)

- [ ] **T-CC-4.3**: Implement time-travel UI (scrub bar, step buttons, state diff)
  - Acceptance: Visual timeline with seek, step forward/backward, diff view
  - Dependencies: T-CC-4.1, T-CC-3.6 (G5 Debug UI)
  - Effort: Medium (3 days)

### G10 Live Collaboration

- [ ] **T-CC-4.4**: Implement CRDT/OT merge for scene edits
  - Acceptance: Multiple editors make concurrent changes; CRDT provides automatic conflict resolution
  - Dependencies: G4 Serialization (T-CC-2.5)
  - Effort: X-Large (2-3 weeks)

- [ ] **T-CC-4.5**: Implement soft locking and presence system
  - Acceptance: Entity selection shows "soft lock"; presence shows other editors' cursors/selection
  - Dependencies: T-CC-4.4
  - Effort: Large (1 week)

- [ ] **T-CC-4.6**: Implement collaboration server with operation log
  - Acceptance: Server maintains authoritative world state; operation log enables undo/history
  - Dependencies: T-CC-4.4
  - Effort: Large (1 week)

---

## Risk Mitigation Tasks (Cross-Phase)

- [ ] **T-CC-R1**: Monitor bridge protocol for stalls during Phase 0 implementation
  - Acceptance: JSON prototype operational before Phase 1 rendering begins
  - Effort: Ongoing

- [ ] **T-CC-R2**: Track shader variant count and prune aggressively per tier
  - Acceptance: Variant count stays under 400 via tier-based pruning
  - Effort: Ongoing (referenced in T-CC-0.7)

- [ ] **T-CC-R3**: Ensure editor is NOT started until Phase 3 (core systems must be stable)
  - Acceptance: Editor work only begins after Phase 1 rendering is complete
  - Effort: Scheduling constraint

- [ ] **T-CC-R4**: Parallelize independent subsystems where possible (Phase 0 tasks vs. Phase 1 shaders)
  - Acceptance: WGSL shader authoring proceeds in parallel with Rust backend
  - Effort: Workflow optimization

---

## Task Summary

| Phase | Task Count | Focus |
|-------|-----------|-------|
| Phase 0 | 21 | Quality tiers, mobile fallback, determinism foundation, risk mitigation |
| Phase 1 | 10 | Tier integration in rendering, Level 1 hot-reload, G3 Data-Driven, G8 Error |
| Phase 2 | 8 | Determinism extension, G4 Unified Serialization |
| Phase 3 | 14 | Hot-reload Levels 2-5, G5 Debug UI, G6 Profiling, G7 Asset Pipeline |
| Phase 4 | 6 | G9 Time-Travel, G10 Live Collaboration |
| Cross-phase | 4 | Risk monitoring |
| **TOTAL** | **46+** | |
