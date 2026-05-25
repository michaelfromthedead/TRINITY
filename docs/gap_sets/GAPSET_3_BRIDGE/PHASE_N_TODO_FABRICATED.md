# GAPSET_3_BRIDGE -- Tasks

TASK_ID format: `T-BRG-{PHASE}.{N}`

---

## Phase 0: Crate Scaffolding (Rust-only, no Python changes)

### T-BRG-0.1: Create renderer-backend crate skeleton
- [x] Create `crates/renderer-backend/Cargo.toml` with wgpu, bytemuck, crossbeam, parking_lot, slotmap dependencies
- [x] Create `crates/renderer-backend/src/lib.rs` as crate root with re-exports
- [x] Create `crates/renderer-backend/src/type_registry.rs` with skeleton structs (TypeRegistry, ComponentTypeInfo, FieldLayout, FieldType)
- [x] Create `crates/renderer-backend/src/bridge.rs` with placeholder PyO3 functions (_omega module placeholder)
- [x] Update `Cargo.toml` to include renderer-backend as workspace member
- [x] Verify `cargo build` succeeds and `omega-ude --info` still works
- **Acceptance:** `cargo build` succeeds. No new runtime behavior. Existing tests pass.
- **Dependencies:** None
- **Effort:** Small (2-3 hours)

### T-BRG-0.2: Add rusqlite/math/fixed-point dependencies to workspace
- [x] Add `crates/math/` for GPU math library (FUTURE: Phase 3 will fill this)
- [x] Add `crates/fixed-point/` for Fixed16/Fixed32 types
- [x] Update workspace-level Cargo.toml
- [x] Verify all workspace crates compile independently
- **Acceptance:** All workspace crates compile. No runtime regressions.
- **Dependencies:** T-BRG-0.1
- **Effort:** Small (1 hour)

---

## Phase 1: Type Channel Protocol

### T-BRG-1.1: Implement full TypeRegistry in Rust
- [x] Complete `ComponentTypeInfo` with all fields, flags, and layout info
- [x] Implement `TypeRegistry` with `RwLock<HashMap<u32, ComponentTypeInfo>>`
- [x] Implement `ArchetypeId` derivation from component ID combinations
- [x] Implement archetype deduplication (same component IDs = same archetype)
- [x] Add debug `type_list()` PyO3 function to inspect registry from Python
- [x] Write Rust unit tests for type registration, deduplication, archetype derivation
- **Acceptance:** Types can be registered and queried. Archetype IDs are deterministic given the same component set.
- **Dependencies:** T-BRG-0.1
- **Effort:** Medium (4-6 hours)

### T-BRG-1.2: Implement _build_rust_layout() in ComponentMeta
- [x] Add `TYPE_MAP` dictionary (float→f32, int→i32, bool→u8, str→string, Fixed16, Fixed32)
- [x] Implement type resolution through Annotated[] wrappers
- [x] Compute field offsets sequentially based on type_size
- [x] Handle variable-width types (string → offset = -1)
- [x] Add fallback: if `_omega` not available, skip step silently
- **Acceptance:** Any Component subclass with standard field types produces a valid Rust layout. Pure Python mode unaffected.
- **Dependencies:** T-BRG-1.1
- **Effort:** Medium (3-4 hours)

### T-BRG-1.3: Wire type_register() into ComponentMeta.__new__()
- [x] Add step 6b after Foundation registration in `ComponentMeta.__new__()`
- [x] Call `_omega.type_register()` with `_build_rust_layout()` output
- [x] Record `Op.REGISTER` step in `_metaclass_steps` for traceability
- [x] Handle `ImportError` (no _omega) and `RuntimeError` (component store not init'd) with silent fallback
- [x] Test: define a Component in Python, verify type appears in Rust registry via debug `type_list()`
- **Acceptance:** Every Component subclass definition triggers a `type_register` call. Pure Python mode unaffected. All 990 existing tests pass.
- **Dependencies:** T-BRG-1.2
- **Effort:** Small (2-3 hours)

### T-BRG-1.4: Add type_register() to bridge.rs
- [x] Implement `type_register()` PyO3 function signature
- [x] Deserialize `field_layouts` from Python list of dicts to Rust `Vec<FieldLayout>`
- [x] Store `ComponentTypeInfo` in global `TypeRegistry`
- [x] Return PyResult<()>
- [x] Verify `type_list()` debug function shows correct data
- **Acceptance:** Python `type_register` calls populate TypeRegistry correctly. Error on invalid type codes.
- **Dependencies:** T-BRG-1.1, T-BRG-1.3
- **Effort:** Medium (3-4 hours)

---

## Phase 2: Data Channel -- Component Store

### T-BRG-2.1: Implement ComponentStore in Rust
- [x] Implement `Archetype` struct with SoA columns (`Vec<Vec<u8>>`)
- [x] Implement `ComponentStore::spawn()` -- find/create archetype, allocate row, write initial data
- [x] Implement `ComponentStore::despawn()` -- mark row as free, update entity_index
- [x] Implement `ComponentStore::read_field()` -- index by archetype+row+offset, return slice
- [x] Implement `ComponentStore::write_field()` -- index by archetype+row+offset, copy bytes
- [x] Implement `ComponentStore::query()` -- iterate archetypes matching component set, collect entity IDs
- [x] Implement `ComponentStore::column_slice()` -- return &[u8] for entire component column in archetype
- [x] Write Rust unit tests: spawn/read/write/despawn cycle, archetype routing, query correctness
- **Acceptance:** Full ECS storage with correct field read/write, entity lifecycle, query, SoA layout.
- **Dependencies:** T-BRG-1.1
- **Effort:** Large (12-16 hours)

### T-BRG-2.2: Create RustStorageDescriptor in Python
- [x] Create `/home/user/dev/USER/PROJECTS_VOID/TRINITY/trinity/descriptors/rust_storage.py`
- [x] Implement `RustStorageDescriptor` as innermost descriptor
- [x] `_get_stored()` calls `_omega.component_read(entity_id, component_id, field_offset, field_type)`
- [x] `_set_stored()` calls `_omega.component_write(entity_id, component_id, field_offset, value)`
- [x] Handle RuntimeError gracefully with Python-side fallback
- **Acceptance:** Component field access routes through Rust. Fallback preserves existing behavior.
- **Dependencies:** T-BRG-2.1
- **Effort:** Medium (3-4 hours)

### T-BRG-2.3: Wire RustStorageDescriptor into ComponentMeta
- [x] Modify `ComponentMeta._install_descriptors()` to check for Rust backend availability
- [x] Add `_storage_descriptor_for()` class method that picks RustStorageDescriptor when available
- [x] Ensure fallback: if `_omega` module missing or component store not initialized, use StorageDescriptor
- [x] Test with both modes (Rust available, Rust unavailable)
- **Acceptance:** Component fields use Rust storage when available, Python dict storage otherwise. All existing tests pass in both modes.
- **Dependencies:** T-BRG-2.2
- **Effort:** Small (2 hours)

### T-BRG-2.4: Add Data channel functions to bridge.rs
- [x] Implement `component_read()` PyO3 function
- [x] Implement `component_write()` PyO3 function
- [x] Implement `world_spawn()` PyO3 function
- [x] Implement `world_despawn()` PyO3 function
- [x] Implement `world_query()` PyO3 function
- [x] Wrap in Arc<RwLock<ComponentStore>> for thread safety
- **Acceptance:** All 5 Data channel functions work from Python. Correct read/write through Rust storage.
- **Dependencies:** T-BRG-2.1
- **Effort:** Medium (5-7 hours)

### T-BRG-2.5: Create Python World and Entity classes
- [x] Create `/home/user/dev/USER/PROJECTS_VOID/TRINITY/trinity/world.py`
- [x] Implement `World.__init__()` -- calls `_omega.world_create()`
- [x] Implement `World.spawn(*components)` -- serializes components, calls `world_spawn()`, returns Entity
- [x] Implement `World.despawn(entity)` -- calls `world_despawn()`
- [x] Implement `World.query(*component_types)` -- calls `world_query()`, returns Entity list
- [x] Implement `Entity.__getattr__()` -- routes component access through world
- [x] Write Python integration test: spawn → read → write → read → despawn → query
- **Acceptance:** Full entity lifecycle in Python backed by Rust storage. Correct field access through descriptors.
- **Dependencies:** T-BRG-2.4
- **Effort:** Medium (4-5 hours)

---

## Phase 3: Data Channel -- GPU Math Library

### T-BRG-3.1: Implement Vec2, Vec3, Vec4 (math.rs ~300 lines)
- [x] `Vec2` (f32, f32) -- 8 bytes, bytemuck::Pod + Zeroable
- [x] `Vec3` (f32, f32, f32) -- 12 bytes, bytemuck::Pod + Zeroable
- [x] `Vec4` (f32, f32, f32, f32) -- 16 bytes, bytemuck::Pod + Zeroable
- [x] Implement: length, normalize, dot, cross (Vec3), lerp, reflect, refract, min, max, clamp, distance, project, reject, homogenize
- [x] Write Rust tests for each operation with known inputs/outputs
- **Acceptance:** All vector operations produce correct results. bytemuck casting works. No nalgebra dependency.
- **Dependencies:** T-BRG-0.2
- **Effort:** Medium (4-5 hours)

### T-BRG-3.2: Implement Mat4 (math.rs ~300 lines)
- [x] Mat4 column-major `[[f32; 4]; 4]` -- 64 bytes, bytemuck::Pod + Zeroable
- [x] Implement: identity, zero, transpose, determinant, inverse, multiply, multiply_vec3, multiply_vec4
- [x] Implement: translate, rotate_x/y/z, scale, perspective, orthographic, look_at, from_transform
- [x] Verify against known matrices (identity, perspective, look_at)
- [x] Write Rust tests: inverse(identity) == identity, perspective * inverse is affine, look_at produces correct axes
- **Acceptance:** Full 4x4 matrix library. Deterministic. Inverse is accurate within 1e-6. Bytemuck-castable for GPU upload.
- **Dependencies:** T-BRG-3.1
- **Effort:** Medium (5-6 hours)

### T-BRG-3.3: Implement Quat and Transform (math.rs ~300 lines)
- [x] Quat (f32, f32, f32, f32) -- 16 bytes, bytemuck::Pod + Zeroable
- [x] Quat operations: identity, from_axis_angle, from_euler, multiply, normalize, conjugate, inverse, slerp, to_rotation_matrix, rotate_vec3
- [x] Transform (Vec3 scale, Quat rotation, Vec3 translation) -- 80 bytes, bytemuck::Pod + Zeroable
- [x] Transform operations: identity, from_parts, to_mat4, decompose, inverse, lerp
- [x] Write Rust tests: quat slerp correctness, transform to_mat4 and back, transform lerp
- **Acceptance:** Full quaternion and transform math. Deterministic. GPU-uploadable.
- **Dependencies:** T-BRG-3.1, T-BRG-3.2
- **Effort:** Medium (5-6 hours)

### T-BRG-3.4: Implement AABB, Frustum, Ray (math.rs ~300 lines)
- [x] AABB (Vec3 min, Vec3 max) -- 24 bytes
- [x] AABB operations: new, from_center_halfext, union, intersection, contains_point, contains_aabb, intersects_sphere, intersects_frustum, transform, grow
- [x] Frustum (6 planes: left, right, top, bottom, near, far) -- 96 bytes
- [x] Frustum operations: from_view_proj, from_matrix, contains_aabb, intersects_aabb, test_sphere, test_point
- [x] Ray (Vec3 origin, Vec3 direction) -- 24 bytes
- [x] Ray operations: new, at, transform, intersects_aabb, intersects_triangle, intersects_plane, intersects_sphere
- [x] Write Rust tests: AABB union/intersection, frustum culling against known test case, ray-AABB intersection
- **Acceptance:** Full spatial math library. Deterministic. bytemuck-compatible. Ready for GPU culling in Phase 4.
- **Dependencies:** T-BRG-3.1, T-BRG-3.2
- **Effort:** Medium (5-6 hours)

---

## Phase 4: Command Channel -- Triangle in wgpu

### T-BRG-4.1: Implement wgpu Renderer skeleton
- [x] Create `crates/renderer-backend/src/renderer.rs`
- [x] wgpu Instance, Adapter, Device, Queue creation
- [x] Surface creation and configuration
- [x] Render loop with command channel drain
- [x] Basic triangle vertex/fragment shaders (inline WGSL)
- [x] Vertex buffer with triangle data
- [x] Render pipeline creation
- [x] Verify: `omega-ude --ui` opens window with rendered triangle
- **Acceptance:** Window opens, triangle renders. Clean shutdown via command channel.
- **Dependencies:** T-BRG-0.1
- **Effort:** Large (10-14 hours)

### T-BRG-4.2: Implement command channel in bridge.rs
- [x] Add crossbeam SPSC channel (Sender to Python, Receiver in Renderer)
- [x] Implement `renderer_resize(w, h)` -- sends `Command::Resize` via channel
- [x] Implement `renderer_screenshot(path)` -- sends `Command::Screenshot` via channel
- [x] Implement `renderer_recompile_materials(ids)` -- sends `Command::RecompilePipelines` via channel
- [x] Implement `renderer_shutdown()` -- sends `Command::Shutdown` via channel
- [x] Register all 4 in `_omega` PyO3 module
- **Acceptance:** Python can trigger window resize, screenshot, pipeline recompile, and clean shutdown.
- **Dependencies:** T-BRG-4.1
- **Effort:** Medium (3-4 hours)

### T-BRG-4.3: Implement winit window management
- [x] Create `crates/renderer-backend/src/window.rs`
- [x] Event loop with resize handling
- [x] Input event forwarding for egui (Phase 11)
- [x] Window close triggers Shutdown command
- [x] Verify window management works across resize/move/close
- **Acceptance:** Window creates, resizes, and closes cleanly. No GPU leaks on shutdown.
- **Dependencies:** T-BRG-4.1
- **Effort:** Medium (3-4 hours)

### T-BRG-4.4: Implement MappedRingBuffer for transform upload
- [x] Create `crates/renderer-backend/src/upload.rs`
- [x] 2-3 slot ring buffer, each slot 64MB
- [x] Persistently mapped as wgpu MAP_WRITE
- [x] memcpy-based upload (no staging buffer)
- [x] Slot advance on frame boundary
- [x] Test with triangle vertex data (phase 4), later with transform arrays (phase 5)
- **Acceptance:** Ring buffer allocates, maps, advances correctly. Data visible to GPU shaders.
- **Dependencies:** T-BRG-4.1
- **Effort:** Medium (4-5 hours)

---

## Phase 5: Data Channel -- Scene Rendering

### T-BRG-5.1: Implement MeshRegistry
- [x] Create `crates/renderer-backend/src/mesh_registry.rs`
- [x] Load meshes: Vertex buffer + Index buffer per mesh
- [x] Vertex format: position (Vec3), normal (Vec3), texcoord (Vec2), tangent (Vec4)
- [x] Mesh handle (AssetId → (vertex_buffer, index_buffer, index_count))
- [x] GPU buffer creation from CPU data
- **Acceptance:** Meshes can be loaded and rendered. Vertex format matches WGSL shader expectations.
- **Dependencies:** T-BRG-4.1, T-BRG-3.1
- **Effort:** Medium (4-5 hours)

### T-BRG-5.2: Implement glTF mesh loading
- [x] Create `crates/renderer-backend/src/asset_loader.rs`
- [x] Background thread for mesh/texture loading
- [x] glTF parsing (via `gltf` crate) with mesh extraction
- [x] Texture loading (via `image` crate) with PNG/JPG/KTX support
- [x] Asset upload channel (lock-free SPSC to render thread)
- [x] Handle: mesh loading, texture loading, shader compilation requests
- **Acceptance:** glTF files load in background. Meshes/textures arrive on render thread. Render thread processes uploads.
- **Dependencies:** T-BRG-5.1
- **Effort:** Medium (5-6 hours)

### T-BRG-5.3: Wire component store data to renderer
- [x] In render loop: acquire read lock on component store
- [x] For each archetype with MeshRenderer + Transform components:
  - memcpy transform column to MappedRingBuffer
  - collect mesh handles for draw calls
- [x] Generate indirect draw commands per archetype
- [x] Render: for each archetype, set vertex buffers, draw indexed
- **Acceptance:** Python-spawned entities with Transform + MeshRenderer appear in viewport. 1000 cubes at 60fps.
- **Dependencies:** T-BRG-5.1, T-BRG-2.5
- **Effort:** Large (10-12 hours)

---

## Phase 6: Data Channel -- PBR + Lights

### T-BRG-6.1: Implement PipelineTable and shader cache
- [x] Create `crates/renderer-backend/src/pipeline.rs`
- [x] PipelineTable: HashMap<u32, CachedPipeline>
- [x] ShaderCache: content-addressed WGSL cache (SHA-256 key), disk-persisted
- [x] Pipeline compilation from WGSL source
- [x] Bind group layout creation per material
- **Acceptance:** Pipelines compile from WGSL and cache. Same source returns cached pipeline.
- **Dependencies:** T-BRG-4.1
- **Effort:** Medium (5-6 hours)

### T-BRG-6.2: Implement PBR shaders
- [x] Write `shaders/pbr.vert.wgsl` -- transform from component store (instanced)
- [x] Write `shaders/pbr.frag.wgsl` -- Cook-Torrance BRDF, GGX, Smith-GGX, Schlick
- [x] Write `shaders/shadow.vert.wgsl` -- shadow map rendering
- [x] Write `shaders/shadow.frag.wgsl` -- depth-only (or PCF variant)
- [x] Verify: PBRBRDF matches reference (Python-side reference in lighting module)
- **Acceptance:** Meshes render with PBR shading. Directional light produces correct specular highlights.
- **Dependencies:** T-BRG-6.1
- **Effort:** Large (10-14 hours)

### T-BRG-6.3: Implement forward+ light culling (compute shader)
- [x] Froxel grid creation (64x48x32 per architecture spec)
- [x] Light culling compute shader (cluster lights per froxel)
- [x] Light buffer upload (directional + point + spot light data)
- [x] Bind light data in PBR fragment shader
- **Acceptance:** Forward+ light culling works. 100 point lights correctly culled, only visible lights evaluated per pixel.
- **Dependencies:** T-BRG-6.2
- **Effort:** Large (8-10 hours)

### T-BRG-6.4: Implement shadow maps for directional light
- [x] CSM with PSSM: cascade splits, shadow map atlas
- [x] Shadow map rendering pass
- [x] PCF filtering in PBR fragment shader (4x4 or 8x8)
- [x] Optional: PCSS for softer shadows
- **Acceptance:** Directional light casts shadows with correct cascades. Shadow acne and peter panning mitigated.
- **Dependencies:** T-BRG-6.2
- **Effort:** Medium (6-8 hours)

---

## Phase 7: Command Channel -- Frame Graph

### T-BRG-7.1: Implement FrameGraphExecutor in Rust
- [x] Create `crates/renderer-backend/src/frame_graph.rs`
- [x] CompiledPass: Graphics, Compute, Copy variants
- [x] Pass compilation: validate DAG, assign resources, insert barriers
- [x] Per-frame execution: iterate passes in topological order
- [x] Resource aliasing: transient resource lifetime analysis
- [x] Barrier insertion: automatic wgpu barrier derivation
- **Acceptance:** Frame graph compiles and executes. Barrier insertion correct. Transient resources alias correctly.
- **Dependencies:** T-BRG-4.1
- **Effort:** Large (12-16 hours)

### T-BRG-7.2: Connect Python FrameGraph declarations to Rust executor
- [x] Python-side frame graph declaration → compiled representation
- [x] Serialize compiled frame graph for Rust consumption
- [x] Rust executor reads pass structure
- [x] Runtime pass reordering from Python (add/remove passes)
- **Acceptance:** Change frame graph from Python at runtime. Viewport updates without restart.
- **Dependencies:** T-BRG-7.1, T-BRG-4.2
- **Effort:** Medium (5-7 hours)

---

## Phase 8: Command Channel -- Material DSL + Hot Reload

### T-BRG-8.1: Create Material DSL Python module
- [x] Create `/home/user/dev/USER/PROJECTS_VOID/TRINITY/trinity/materials/dsl.py`
- [x] Material base class, SurfaceContext, SurfaceOutput
- [x] `sample()`, `noise()`, `texture()` stub functions
- [x] MaterialMeta metaclass: validates surface(), walks AST, translates to WGSL
- [x] PBR template WGSL assembly (PBR body + surface body)
- **Acceptance:** Python-defined material class produces WGSL source that compiles via naga.
- **Dependencies:** T-BRG-6.2
- **Effort:** Large (10-14 hours)

### T-BRG-8.2: Create Material DSL → WGSL compiler
- [x] Create `/home/user/dev/USER/PROJECTS_VOID/TRINITY/trinity/materials/compiler.py`
- [x] Python AST node walker
- [x] AST→WGSL translation table (Python → WGSL built-in mapping)
- [x] Type mapping: Python float → f32, Vec3 → vec3<f32>, etc.
- [x] Function call interception for sample(), noise(), etc.
- [x] WGSL text assembly from compiled surface body + PBR template
- **Acceptance:** Python AST translates to valid WGSL. Complex material compiles without error.
- **Dependencies:** T-BRG-8.1
- **Effort:** Large (10-12 hours)

### T-BRG-8.3: Implement dependency graph for hot-reload
- [x] Extend PipelineTable with DepGraph
- [x] include_to_materials mapping
- [x] material_to_includes tracking
- [x] BFS invalidation when include changes
- [x] Atomic pipeline swap (no frame drop)
- **Acceptance:** Material hot-reload: change REPL, viewport updates within 2 frames. Dependency cascade works correctly.
- **Dependencies:** T-BRG-6.1, T-BRG-4.2
- **Effort:** Medium (4-5 hours)

### T-BRG-8.4: Wire material_register() into bridge
- [x] Implement `material_register()` PyO3 function
- [x] Store material metadata in PipelineTable
- [x] Compile material WGSL to wgpu pipeline
- [x] Cache pipeline handle for render loop
- **Acceptance:** Materials defined in Python compile to working wgpu pipelines. Hot-reload works.
- **Dependencies:** T-BRG-8.3, T-BRG-8.2
- **Effort:** Medium (4-5 hours)

---

## Phase 9: Full Features (Post-Processing, Particles, GI)

### T-BRG-9.1: Implement post-process stack
- [x] Tonemapping (ACES) compute shader
- [x] Bloom (Gaussian pyramid) compute shader
- [x] TAA compute shader
- [x] Post-process pipeline orchestrator (ordered pass chain)
- [x] Each effect as a FrameGraph pass
- **Acceptance:** All 3 effects work. Tonemapping produces ACES-standard output. Bloom is correct. TAA reduces aliasing.
- **Dependencies:** T-BRG-7.1
- **Effort:** Large (12-16 hours)

### T-BRG-9.2: Implement GPU particles
- [x] Particle spawn compute pass
- [x] Particle update compute pass (velocity, color, size, noise, collision)
- [x] Particle render pass (GPU vertex shader draws quads from particle data)
- [x] Particle compact pass (remove dead particles)
- [x] VFX graph runtime (node-based effect execution)
- **Acceptance:** GPU particles render at 60fps with 100,000 particles. Emitter lifecycle works (spawn → update → dead → compact).
- **Dependencies:** T-BRG-5.3
- **Effort:** Large (12-16 hours)

### T-BRG-9.3: Implement DDGI probes
- [x] SH probe encoding/decoding library
- [x] Probe placement (uniform grid, adaptive based on geometry)
- [x] Probe ray tracing (ray queries via wgpu RT)
- [x] Probe blend (irradiance interpolation)
- [x] Temporal accumulation and denoising
- **Acceptance:** DDGI provides indirect diffuse lighting. Probes update over time. Correct bounce lighting on diffuse surfaces.
- **Dependencies:** T-BRG-6.3
- **Effort:** Large (12-16 hours)

---

## Phase 10: GPU Memory Management

### T-BRG-10.1: Implement GpuMemoryManager
- [x] Create `crates/renderer-backend/src/memory.rs`
- [x] FrameAllocator: bump-pointer, per-frame reset
- [x] PoolAllocator: fixed-size blocks (64KB, 256KB, 1MB, 4MB)
- [x] StackAllocator: LIFO for nested staging allocations
- [x] Budget enforcement (configurable GPU memory cap)
- [x] Used bytes tracking (AtomicU64)
- **Acceptance:** All three allocators work correctly. Budget enforcement prevents OOM. Usage tracking reports correct values.
- **Dependencies:** T-BRG-4.1
- **Effort:** Medium (5-6 hours)

### T-BRG-10.2: Implement streaming resource pool
- [x] LRU eviction for textures/meshes exceeding budget
- [x] Page-in/page-out based on distance and visibility
- [x] Mipmap streaming (load base mip, stream higher mips)
- [x] Ensure minimum-quality fallback while streaming
- **Acceptance:** Resources page in/out correctly. Budget stays within limit. Visual quality degrades gracefully during streaming.
- **Dependencies:** T-BRG-10.1
- **Effort:** Large (8-10 hours)

---

## Phase 11: Editor Integration

### T-BRG-11.1: Implement egui-wgpu integration
- [x] Create `crates/renderer-backend/src/egui_integration.rs`
- [x] egui-wgpu adapter for renderer
- [x] egui-winit integration for input
- [x] Editor panels: Viewport (wgpu surface embedded in egui), Inspector, Hierarchy
- [x] Panel data source: read component store through Data channel
- **Acceptance:** Editor opens with three panels. Viewport shows rendered scene. Inspector shows entity components. Hierarchy lists entities.
- **Dependencies:** T-BRG-5.3
- **Effort:** Large (10-14 hours)

### T-BRG-11.2: Connect REPL to live runtime
- [x] IPython REPL integration: evaluate Python expressions against live component store
- [x] Entity creation/destruction from REPL
- [x] Component field mutation from REPL (visible in viewport next frame)
- **Acceptance:** REPL can spawn entities, modify components, and see results in viewport.
- **Dependencies:** T-BRG-11.1
- **Effort:** Medium (5-6 hours)

---

## Summary

| Phase | Tasks | Total Estimated Effort | Dependencies |
|-------|-------|----------------------|--------------|
| 0: Scaffolding | 2 | 3-4 hours | None |
| 1: Type Channel | 4 | 12-17 hours | Phase 0 |
| 2: Component Store | 5 | 26-34 hours | Phase 1 |
| 3: GPU Math Library | 4 | 19-23 hours | Phase 0 |
| 4: Triangle | 4 | 20-27 hours | Phase 0, 3 |
| 5: Scene Rendering | 3 | 19-23 hours | Phase 2, 3, 4 |
| 6: PBR + Lights | 4 | 29-38 hours | Phase 4, 5 |
| 7: Frame Graph | 2 | 17-23 hours | Phase 4 |
| 8: Material DSL | 4 | 28-35 hours | Phase 6, 7 |
| 9: Full Features | 3 | 36-48 hours | Phase 6, 7 |
| 10: GPU Memory | 2 | 13-16 hours | Phase 4 |
| 11: Editor | 2 | 15-20 hours | Phase 5 |
| **ALL** | **39 tasks** | **~237-308 hours** | |

**Critical path:** Phase 0 → Phase 3 → Phase 4 → Phase 5 → Phase 6 → Phase 8 → Phase 9

**Phases that can run in parallel once Phase 0 completes:**
- Phase 1 (Type Channel) || Phase 3 (GPU Math Library)

**Phases independent of each other:**
- Phase 7 (Frame Graph) can begin after Phase 4 (doesn't need Phases 5-6)
- Phase 10 (GPU Memory) can begin after Phase 4
- Phase 11 (Editor) can begin after Phase 5
