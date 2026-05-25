# Game Engine Integration — Implementation TODO

> Tracks all work needed to connect Trinity (definition-time) + Foundation (runtime) 
> to the actual engine systems specified in `/DIAGRAMS/`.
>
> **Prerequisites complete:** Trinity Pattern (Phases 1-10), Foundation (24 modules), FlowForge backend.
> **This document tracks:** Building the engine runtime that Trinity declares and Foundation observes.

## Status Key

- [ ] Not started
- [~] In progress  
- [x] Complete

---

## Table of Contents

1. [Engine Bootstrap & Game Loop](#1-engine-bootstrap--game-loop)
2. [Platform Layer](#2-platform-layer)
3. [Core Systems](#3-core-systems)
4. [Resource Layer](#4-resource-layer)
5. [Rendering Layer](#5-rendering-layer)
6. [Simulation Layer](#6-simulation-layer)
7. [Animation Layer](#7-animation-layer)
8. [Audio Layer](#8-audio-layer)
9. [Gameplay Layer](#9-gameplay-layer)
10. [UI Layer](#10-ui-layer)
11. [Networking Layer](#11-networking-layer)
12. [World Layer](#12-world-layer)
13. [Tooling Layer](#13-tooling-layer)
14. [Debug Layer](#14-debug-layer)
15. [XR Layer](#15-xr-layer)
16. [Cross-Cutting Integration](#16-cross-cutting-integration)
17. [Deterministic Simulation](#17-deterministic-simulation)
18. [Testing & Validation](#18-testing--validation)

---

## 1. Engine Bootstrap & Game Loop

> **Ref:** `DIAGRAMS/FULL_ARCHITECTURE.md` §Frame Structure (L2737-2777)
> **Integration:** GAME_ENGINE_INTEGRATION.md §4 (How Engine Layers Connect Through Foundation)

The engine needs a main loop, system scheduler, and frame structure before any layer can run.

### 1.1 Frame Structure
- [ ] Implement `Engine` class using `EngineMeta` metaclass
- [ ] Implement fixed-timestep game loop (fixed update + variable render)
- [ ] Implement frame phases: Input → Simulation → Animation → Rendering → Audio → Cleanup
- [ ] Wire phase ordering to `SystemMeta` phase assignments (`@system(phase="gameplay")`)
- [ ] Integrate Foundation EventLog — record frame boundaries as Events
- [ ] Integrate Foundation Tracker — flush dirty flags per frame

### 1.2 System Scheduler
- [ ] Implement topological sort of Systems based on `SystemMeta._dependencies`
- [ ] Implement parallel system execution within phases (non-conflicting queries)
- [ ] Implement `@exclusive` systems that require sole access
- [ ] Wire `@phase` decorator to scheduler phase assignment
- [ ] Wire `@parallel` decorator to mark parallelizable systems
- [ ] Integrate Foundation Registry — discover all registered Systems at startup

### 1.3 World Management
- [ ] Implement World as the central entity container (bridge ShellLang World to engine World)
- [ ] Implement entity lifecycle: create → attach components → update → destroy
- [ ] Implement component storage (archetype-based SoA as specified in ARCHITECTURE_CORE.md)
- [ ] Wire `ComponentMeta._registry` to archetype storage initialization
- [ ] Wire Foundation Bridge — TrinityWorldAdapter syncs engine World ↔ ShellLang World

### 1.4 Session Integration
- [ ] Implement Session save/load using Foundation Serializer
- [ ] Implement crash recovery from Session snapshots
- [ ] Wire Foundation ContentStore for session storage with structural sharing
- [ ] Wire Foundation DeltaSync for incremental session saves

---

## 2. Platform Layer

> **Ref:** `DIAGRAMS/ARCHITECTURE_PLATFORM.md`
> **Ref:** `DIAGRAMS/FULL_ARCHITECTURE.md` §Platform Layer (L136-178)
> **Trinity decorators:** `@native`, `@ffi`, `@target`, `@unsafe` (compilation.py), `@platform_specifics`

### 2.1 OS Abstraction
- [ ] Implement Window management (create, resize, fullscreen, multi-monitor)
- [ ] Implement Input polling (keyboard, mouse, gamepad raw events)
- [ ] Implement File I/O abstraction (async file reads, path resolution)
- [ ] Implement Threading primitives (threads, mutexes, atomics, fibers)
- [ ] Register platform Resources via `ResourceMeta` → Foundation Registry
- [ ] Wire `@native` decorator to mark platform-specific implementations

### 2.2 Render Hardware Interface (RHI)
- [ ] Define RHI abstraction protocol using `ProtocolMeta`
- [ ] Implement Vulkan backend (or initial backend choice)
- [ ] Implement GPU resource management (buffers, textures, pipelines, shaders)
- [ ] Implement command buffer recording and submission
- [ ] Implement swap chain management
- [ ] Register GPU device as Resource in Foundation Registry
- [ ] Wire `@gpu_buffer`, `@gpu_kernel` decorators to RHI resources

### 2.3 Platform Services
- [ ] Implement timer/clock (high-resolution, monotonic)
- [ ] Implement dynamic library loading (for plugins/mods)
- [ ] Wire `@target` decorator for platform-conditional compilation

---

## 3. Core Systems

> **Ref:** `DIAGRAMS/ARCHITECTURE_CORE.md`
> **Ref:** `DIAGRAMS/FULL_ARCHITECTURE.md` §Core Systems (L178-278)
> **Trinity decorators:** Memory (memory.py), Scheduling (scheduling.py), ECS (ecs_core.py)

### 3.1 Memory System
- [ ] Implement 6 allocator types: Linear, Stack, Pool, Ring, Slab, TLSF
- [ ] Wire `@pooled` decorator → Pool allocator for component storage
- [ ] Wire `@packed` decorator → SoA/packed memory layout
- [ ] Wire `@aligned` decorator → alignment requirements
- [ ] Wire `@arena` decorator → arena allocator scope
- [ ] Wire `@budget` decorator → memory budget tracking
- [ ] Wire `@allocator` decorator → custom allocator assignment
- [ ] Integrate Foundation Mirror — expose allocator stats for inspection
- [ ] Integrate Foundation Tracker — track allocation/deallocation events

### 3.2 Math Library
- [ ] Implement Vec2, Vec3, Vec4 with SIMD (or use existing library)
- [ ] Implement Mat3, Mat4, Quat
- [ ] Implement AABB, OBB, Sphere, Plane, Ray, Frustum
- [ ] Implement curves: Bezier, Catmull-Rom, Hermite
- [ ] Implement fixed-point types for deterministic simulation
- [ ] Register math types with Foundation Registry for Mirror/Inspector support

### 3.3 Task System
- [ ] Implement job graph with work-stealing scheduler
- [ ] Implement fiber-based coroutines (or async equivalent)
- [ ] Wire `@parallel` decorator → task graph parallelism
- [ ] Wire `@exclusive` decorator → exclusive access requirements
- [ ] Wire SystemMeta phase/dependency info → task graph construction
- [ ] Integrate Foundation EventLog — profile task execution timing

### 3.4 ECS Runtime
- [ ] Implement archetype-based component storage (SoA)
- [ ] Implement entity ID generation (generational indices)
- [ ] Implement query system (`@query` decorator → runtime query execution)
- [ ] Implement component add/remove with archetype migration
- [ ] Wire ALL ComponentMeta-registered types to archetype storage
- [ ] Wire `@relation` decorator → entity relationship tracking
- [ ] Wire `@derived` decorator → computed component generation
- [ ] Wire Foundation Query → ECS query execution backend
- [ ] Wire Foundation Tracker → dirty flag integration with query cache invalidation

---

## 4. Resource Layer

> **Ref:** `DIAGRAMS/ARCHITECTURE_RESOURCE.md`
> **Ref:** `DIAGRAMS/FULL_ARCHITECTURE.md` §Resource Layer (L278-352)
> **Trinity decorators:** `@serializable`, `@asset` (ecs_core.py, data_flow.py), lod_streaming.py

### 4.1 Asset Pipeline
- [ ] Implement Asset handle system (typed references with ref-counting)
- [ ] Implement async asset loading with priority queues
- [ ] Implement asset hot-reload (file watcher → reload → notify)
- [ ] Wire `AssetMeta`-registered types to asset loader dispatch
- [ ] Wire `@serializable` descriptor chain → asset serialization format
- [ ] Wire Foundation ContentStore → content-addressable asset storage
- [ ] Integrate Foundation EventLog — log asset load/unload events

### 4.2 Asset Types
- [ ] Implement Texture asset (formats, mip generation, compression)
- [ ] Implement Mesh asset (vertex formats, index buffers, LOD levels)
- [ ] Implement Material asset (shader parameters, texture references)
- [ ] Implement Shader asset (compilation, reflection, variants)
- [ ] Implement Animation asset (clips, curves, events)
- [ ] Implement Audio asset (streaming, compression formats)
- [ ] Implement Prefab asset (entity templates)
- [ ] Register all asset types via AssetMeta → Foundation Registry

### 4.3 Virtualization
- [ ] Implement virtual texturing (tile-based streaming, feedback buffer)
- [ ] Implement virtual geometry (Nanite-style, cluster-based LOD)
- [ ] Wire `@lod` decorator → LOD level definitions
- [ ] Wire `@streaming` decorator → streaming priority/distance rules

---

## 5. Rendering Layer

> **Ref:** `DIAGRAMS/ARCHITECTURE_RENDERING.md`
> **Ref:** `DIAGRAMS/FULL_ARCHITECTURE.md` §Rendering Layer (L352-477)
> **Trinity decorators:** rendering.py (24 decorators), gpu.py, particles_vfx.py

### 5.1 Frame Graph
- [ ] Implement frame graph (render pass declaration, resource aliasing)
- [ ] Implement automatic resource barrier insertion
- [ ] Implement render pass dependency analysis and scheduling
- [ ] Wire rendering Systems (`@system(phase="render")`) as frame graph nodes

### 5.2 GPU-Driven Rendering
- [ ] Implement indirect draw call generation
- [ ] Implement GPU culling (frustum, occlusion, distance)
- [ ] Implement instance batching and merging
- [ ] Wire `@render_layer` decorator → render layer assignment
- [ ] Wire `@shadow_caster` decorator → shadow pass inclusion

### 5.3 Materials & Shading
- [ ] Implement PBR material model (metallic-roughness)
- [ ] Implement material instance system (parameter overrides)
- [ ] Implement shader variant compilation
- [ ] Wire `@material_domain` decorator → material type classification
- [ ] Wire `@material_blend` decorator → blend mode selection
- [ ] Wire ValidatedDescriptor → material parameter validation

### 5.4 Lighting & GI
- [ ] Implement direct lighting (point, spot, directional, area)
- [ ] Implement shadow mapping (cascaded, point, spot)
- [ ] Implement global illumination (Lumen-style or probe-based)
- [ ] Wire `@gi_contributor` decorator → GI participation flags
- [ ] Wire `@reflection_probe` decorator → reflection probe placement

### 5.5 Post-Processing
- [ ] Implement post-process stack (bloom, tone mapping, DOF, motion blur, AO)
- [ ] Implement temporal anti-aliasing (TAA/TSR)
- [ ] Wire post-process Systems to frame graph

### 5.6 Particles & VFX
- [ ] Implement GPU particle system
- [ ] Implement VFX graph (node-based effect authoring)
- [ ] Wire `@particles_vfx` decorators → particle system configuration

---

## 6. Simulation Layer

> **Ref:** `DIAGRAMS/ARCHITECTURE_SIMULATION.md`
> **Ref:** `DIAGRAMS/FULL_ARCHITECTURE.md` §Simulation Layer (L477-547)
> **Trinity decorators:** physics_sim.py, destruction.py

### 6.1 Physics Core
- [ ] Implement (or integrate) rigid body dynamics
- [ ] Implement broadphase collision (BVH or sweep-and-prune)
- [ ] Implement narrowphase collision (GJK/EPA, SAT)
- [ ] Implement constraint solver (sequential impulse or PGS)
- [ ] Wire `@simulation_domain` decorator → physics world assignment
- [ ] Wire `@substep` decorator → substep count configuration
- [ ] Wire `@solver` decorator → solver iteration count
- [ ] Wire `@sleep_threshold` decorator → sleep parameters
- [ ] Wire `@ccd` decorator → continuous collision detection

### 6.2 Physics Components
- [ ] Implement RigidBody component (mass, inertia, velocity, forces)
- [ ] Implement Collider components (box, sphere, capsule, mesh, convex hull)
- [ ] Implement physics materials (friction, restitution, density)
- [ ] Wire `@physics_material` decorator → material properties
- [ ] Register physics components via ComponentMeta → Foundation Registry
- [ ] Wire TrackedDescriptor → Tracker for physics state changes

### 6.3 Constraints
- [ ] Implement joint types (hinge, ball, prismatic, fixed, distance, spring)
- [ ] Wire `@joint` decorator → joint configuration
- [ ] Implement motors and limits on joints

### 6.4 Advanced Simulation
- [ ] Implement destruction system (fracture, debris, damage propagation)
- [ ] Wire `@destructible`, `@fracture`, `@damage_type`, `@damage_resistance` decorators
- [ ] Implement buoyancy (`@buoyancy` decorator)
- [ ] Implement wind system (`@wind` decorator)
- [ ] Integrate Foundation EventLog — log collision events, destruction events

---

## 7. Animation Layer

> **Ref:** `DIAGRAMS/ARCHITECTURE_ANIMATION.md`
> **Ref:** `DIAGRAMS/FULL_ARCHITECTURE.md` §Animation Layer (L547-703)
> **Trinity decorators:** animation.py, ik_procedural.py

### 7.1 Skeletal Animation
- [ ] Implement skeleton/bone hierarchy
- [ ] Implement animation clip playback (sampling, looping, events)
- [ ] Implement animation blending (linear, additive)
- [ ] Wire `@tween` decorator → tween animation support
- [ ] Wire `@blend_tree` decorator → blend tree configuration

### 7.2 Animation Graph
- [ ] Implement state machine (states, transitions, conditions)
- [ ] Implement blend trees (1D, 2D, additive)
- [ ] Wire StateMeta → animation state registration
- [ ] Integrate Foundation Tracker — track animation state changes

### 7.3 IK & Procedural
- [ ] Implement IK solvers (two-bone, FABRIK, CCD)
- [ ] Implement procedural animation (look-at, foot placement, ragdoll blend)
- [ ] Wire `@ik_procedural` decorators → IK target configuration

### 7.4 Motion Matching
- [ ] Implement motion matching database (pose search, trajectory matching)
- [ ] Implement motion matching runtime (query, blend, transition)
- [ ] Wire motion data as Assets via AssetMeta

### 7.5 Facial & Skinning
- [ ] Implement blend shape / morph target system
- [ ] Implement skinning (LBS, dual quaternion)
- [ ] Implement facial animation (FACS, visemes)

---

## 8. Audio Layer

> **Ref:** `DIAGRAMS/ARCHITECTURE_AUDIO.md`
> **Ref:** `DIAGRAMS/FULL_ARCHITECTURE.md` §Audio Layer (L703-802)
> **Trinity decorators:** audio.py, audio_extended.py

### 8.1 Audio Core
- [ ] Implement (or integrate) audio backend (platform audio APIs)
- [ ] Implement audio source/listener system
- [ ] Implement audio clip playback (one-shot, looping, streaming)
- [ ] Wire `@sound` decorator → audio source configuration
- [ ] Wire `@spatial_audio` decorator → 3D spatialization
- [ ] Register audio sources/listeners as Components via ComponentMeta

### 8.2 Mix Bus
- [ ] Implement mix bus hierarchy (master → submixes → voices)
- [ ] Implement volume, pitch, low-pass, high-pass per bus
- [ ] Wire `@audio_bus` decorator → bus assignment and routing
- [ ] Register mix buses as Resources via ResourceMeta

### 8.3 Spatial & Acoustic
- [ ] Implement HRTF-based 3D audio
- [ ] Implement reverb zones and acoustic simulation
- [ ] Implement occlusion/obstruction
- [ ] Implement distance attenuation curves

### 8.4 Adaptive Music
- [ ] Implement adaptive music system (layers, transitions, stingers)
- [ ] Implement music state machine
- [ ] Wire StateMeta → music state registration

### 8.5 DSP Effects
- [ ] Implement DSP effect chain (reverb, delay, EQ, compressor, chorus)
- [ ] Implement real-time parameter modulation

---

## 9. Gameplay Layer

> **Ref:** `DIAGRAMS/ARCHITECTURE_GAMEPLAY.md`
> **Ref:** `DIAGRAMS/FULL_ARCHITECTURE.md` §Gameplay Layer (L802-917)
> **Trinity decorators:** gameplay.py, game_ai.py, state_machine.py, crafting.py, input.py

### 9.1 Entity & Object Model
- [ ] Implement Actor class (entity + transform + lifecycle)
- [ ] Implement entity prefab instantiation from `@prefab` definitions
- [ ] Implement entity lifecycle hooks (spawn, begin play, tick, end play, destroy)
- [ ] Wire `@lifecycle` decorators → lifecycle event registration
- [ ] Integrate Foundation EventLog — log entity lifecycle events

### 9.2 Behavior Trees
- [ ] Implement BT runtime (selector, sequence, parallel, decorator nodes)
- [ ] Implement blackboard (shared AI state)
- [ ] Wire `@behavior_tree` decorator → BT definition
- [ ] Register BT node types via Foundation Registry

### 9.3 AI Systems
- [ ] Implement GOAP planner (goals, actions, world state)
- [ ] Implement utility AI (scoring, curves, considerations)
- [ ] Implement perception system (sight, hearing, awareness)
- [ ] Wire `@game_ai` decorators → AI agent configuration
- [ ] Integrate Foundation EventLog — log AI decisions for debugging

### 9.4 Navigation
- [ ] Implement NavMesh generation and pathfinding (A*, string pulling)
- [ ] Implement navigation agent (steering, avoidance, crowds)
- [ ] Wire `@navmesh` decorator → navigation mesh configuration

### 9.5 Input System
- [ ] Implement input action mapping (action → key bindings)
- [ ] Implement input contexts (gameplay, UI, menu)
- [ ] Implement input buffering and combo detection
- [ ] Wire `@input` decorators → input action definitions
- [ ] Register input mappings as Resources via ResourceMeta

### 9.6 Ability System
- [ ] Implement Gameplay Ability System (abilities, effects, attributes)
- [ ] Implement cooldowns, costs, targeting
- [ ] Wire `@ability`, `@buff`, `@gameplay_tag` decorators
- [ ] Wire TrackedDescriptor → attribute change tracking

### 9.7 Quest & Narrative
- [ ] Implement quest system (objectives, conditions, rewards)
- [ ] Implement dialogue system
- [ ] Wire `@narrative` and `@cinematics` decorators
- [ ] Integrate Foundation EventLog — log quest progression

### 9.8 Economy & Crafting
- [ ] Implement inventory system
- [ ] Implement crafting system (`@recipe`, `@ingredient`, `@crafting_station`)
- [ ] Implement economy (currencies, transactions, trading)
- [ ] Wire `@economy`, `@crafting` decorators
- [ ] Wire `@transactions` decorator → atomic inventory operations

---

## 10. UI Layer

> **Ref:** `DIAGRAMS/ARCHITECTURE_UI.md`
> **Ref:** `DIAGRAMS/FULL_ARCHITECTURE.md` §UI Layer (L917-1171)
> **Trinity decorators:** ui.py, accessibility.py

### 10.1 UI Framework
- [ ] Implement retained-mode UI widget tree
- [ ] Implement flex layout engine
- [ ] Implement UI coordinate system (screen-space, anchors, margins)
- [ ] Implement UI input handling (focus, hover, click, drag)
- [ ] Register UI widgets as Components via ComponentMeta

### 10.2 Widget System
- [ ] Implement base widgets (Text, Image, Button, Slider, Toggle, TextInput, ScrollView)
- [ ] Implement game widgets (HealthBar, Minimap, Inventory, Tooltip, DamageNumbers)
- [ ] Implement widget states (normal, hovered, pressed, disabled, focused)
- [ ] Wire `@ui` decorators → widget configuration

### 10.3 Data Binding
- [ ] Implement data binding (UI ↔ game state)
- [ ] Implement list virtualization (virtual scroll for large lists)
- [ ] Wire TrackedDescriptor.on_change → UI re-render on data change
- [ ] Wire Foundation Tracker → UI dirty flag system

### 10.4 Styling
- [ ] Implement style system (themes, cascading properties)
- [ ] Implement transitions and animations
- [ ] Wire `@accessibility` decorators → screen reader support, high contrast

### 10.5 Screen Management
- [ ] Implement screen/page stack (push, pop, transition)
- [ ] Implement modal system
- [ ] Wire StateMeta → screen state machine

---

## 11. Networking Layer

> **Ref:** `DIAGRAMS/ARCHITECTURE_NETWORKING.md`
> **Ref:** `DIAGRAMS/FULL_ARCHITECTURE.md` §Networking Layer (L1171-1438)
> **Trinity decorators:** network_extended.py, rpc.py, data_flow.py

### 11.1 Transport
- [ ] Implement UDP transport with reliability layer
- [ ] Implement channel system (reliable-ordered, reliable-unordered, unreliable)
- [ ] Implement connection management (handshake, heartbeat, timeout, reconnect)
- [ ] Wire ProtocolMeta-registered protocols → wire format

### 11.2 Replication
- [ ] Implement property replication (server → client)
- [ ] Wire NetworkedDescriptor dirty flags → replication prioritization
- [ ] Implement relevancy / interest management
- [ ] Wire `@networked` decorator → replication configuration (authority, interpolate, priority)
- [ ] Wire Foundation Tracker.all_dirty() → collect dirty networked fields per frame
- [ ] Wire SerializableDescriptor → network serialization format

### 11.3 RPCs
- [ ] Implement RPC system (client→server, server→client, multicast)
- [ ] Wire `@rpc` decorators → RPC registration and dispatch
- [ ] Implement RPC validation and rate limiting

### 11.4 Prediction & Reconciliation
- [ ] Implement client-side prediction (predict locally, reconcile on server correction)
- [ ] Implement server reconciliation (replay inputs on correction)
- [ ] Implement entity interpolation (buffered, for non-predicted entities)
- [ ] Wire Foundation EventLog → operation history for rollback
- [ ] Wire Foundation Tracker.undo() → state rollback for reconciliation
- [ ] Wire `@reconciliation` decorator → reconciliation strategy

### 11.5 Lag Compensation
- [ ] Implement lag compensation (server rewinds world state to client's view time)
- [ ] Wire Foundation snapshots → historical state storage
- [ ] Wire Foundation DeltaSync → efficient snapshot diffing

### 11.6 Anti-Cheat & Security
- [ ] Implement server authority validation
- [ ] Implement input validation and rate limiting
- [ ] Wire Foundation Capabilities → permission checking for network commands
- [ ] Wire `@security` decorators → authority rules

### 11.7 Matchmaking & Voice
- [ ] Implement matchmaking (lobby, queue, skill-based)
- [ ] Implement voice chat (or integrate middleware)
- [ ] Wire `@social` decorators → social system integration

---

## 12. World Layer

> **Ref:** `DIAGRAMS/ARCHITECTURE_WORLD.md`
> **Ref:** `DIAGRAMS/FULL_ARCHITECTURE.md` §World Layer (L1438-1743)
> **Trinity decorators:** world_building.py, lod_streaming.py, spatial.py

### 12.1 World Partition
- [ ] Implement world partition grid (cells, streaming volumes)
- [ ] Implement cell states (unloaded, loading, loaded, activated, deactivating)
- [ ] Implement data layers (static geometry, gameplay, foliage, audio)
- [ ] Implement streaming logic (distance-based, priority, budget)
- [ ] Wire StateMeta → cell state machine
- [ ] Wire ResourceMeta → streaming budget as global Resource

### 12.2 HLOD System
- [ ] Implement Hierarchical LOD (HLOD) generation
- [ ] Implement HLOD proxy mesh generation
- [ ] Wire `@lod` decorator → LOD distance/quality settings

### 12.3 Terrain
- [ ] Implement heightmap terrain (representation, sculpting, rendering)
- [ ] Implement terrain materials (splat maps, layer blending)
- [ ] Implement terrain LOD (clipmap or quadtree)
- [ ] Wire terrain as component with TrackedDescriptor for live editing

### 12.4 Foliage
- [ ] Implement foliage instancing (procedural scatter, GPU instanced draw)
- [ ] Implement foliage interaction (bend on touch, destruction)
- [ ] Wire `@foliage_type` decorator → foliage definition

### 12.5 Procedural Placement (PCG)
- [ ] Implement PCG framework (rules, distributions, constraints)
- [ ] Wire `@procedural` decorators → PCG rule definitions
- [ ] Integrate Foundation Provenance → track PCG derivation

### 12.6 Environment
- [ ] Implement sky system (atmosphere, sun, moon, stars)
- [ ] Implement time-of-day cycle
- [ ] Implement weather system (rain, snow, fog, wind)
- [ ] Wire `@water` decorator → water body configuration
- [ ] Wire `@trigger_volume` decorator → volume trigger regions
- [ ] Implement environment volumes (fog, post-process, audio)

### 12.7 World Queries
- [ ] Implement spatial queries (overlap, raycast, sweep)
- [ ] Wire `@spatial` decorators → spatial indexing configuration
- [ ] Wire Foundation Query → spatial query backend

---

## 13. Tooling Layer

> **Ref:** `DIAGRAMS/ARCHITECTURE_TOOLING.md`
> **Ref:** `DIAGRAMS/FULL_ARCHITECTURE.md` §Tooling Layer (L1743-2015)
> **Existing:** FlowForge backend (AST parsing, codegen, trinity_adapter)

### 13.1 Editor Framework
- [ ] Implement editor application shell (window, menus, panels, docking)
- [ ] Implement viewport rendering (3D scene view, camera controls)
- [ ] Implement transform gizmos (translate, rotate, scale)
- [ ] Implement selection system (click, box select, multi-select)
- [ ] Wire Foundation Inspector → property panel for selected objects
- [ ] Wire Foundation Shell → editor console
- [ ] Wire Foundation Mirror → object property display

### 13.2 Level Editor
- [ ] Implement entity placement, duplication, deletion in viewport
- [ ] Implement snapping (grid, surface, vertex, edge)
- [ ] Implement undo/redo via Foundation Tracker.undo()/redo()
- [ ] Wire Foundation EventLog → editor action history

### 13.3 Asset Tools
- [ ] Implement content browser (asset listing, search, thumbnails)
- [ ] Implement asset import pipeline (mesh, texture, audio, animation)
- [ ] Wire AssetMeta registry → content browser type filtering

### 13.4 FlowForge Integration (Visual Scripting)
- [ ] Connect FlowForge backend to editor (embed visual scripting panel)
- [ ] Wire FlowForge trinity_adapter → live decorator introspection in editor
- [ ] Implement FlowForge node execution (run visual scripts at runtime)
- [ ] Wire Foundation Capabilities → FlowForge sandbox permissions

### 13.5 Material Editor
- [ ] Implement node-based material editor
- [ ] Implement material preview (sphere, plane, mesh)
- [ ] Wire material parameters to ValidatedDescriptor for live editing

### 13.6 Animation Tools
- [ ] Implement timeline/sequencer
- [ ] Implement animation curve editor
- [ ] Implement animation preview

### 13.7 Build & Cook
- [ ] Implement build pipeline (source → cooked → packaged)
- [ ] Implement asset cooking (platform-specific optimization)
- [ ] Implement incremental builds
- [ ] Wire `@build_deploy` decorators → build configuration

### 13.8 Version Control
- [ ] Implement VCS integration (lock/unlock, diff, merge)
- [ ] Wire Foundation ContentStore → content-addressable asset versioning

---

## 14. Debug Layer

> **Ref:** `DIAGRAMS/ARCHITECTURE_DEBUG.md`
> **Ref:** `DIAGRAMS/FULL_ARCHITECTURE.md` §Debug Layer (L2015-2255)
> **Trinity decorators:** debug_safety.py, debug_cheat.py, debug_extended.py
> **Existing:** Foundation Inspector, Shell, EventLog

### 14.1 Logging
- [ ] Implement logging system (categories, levels, output targets)
- [ ] Implement log filtering (per-category, per-level)
- [ ] Implement log output to file, console, network
- [ ] Wire Foundation EventLog → structured log entries

### 14.2 Console System
- [ ] Implement in-game console (CVars, commands)
- [ ] Implement CVar system (typed variables, change callbacks)
- [ ] Wire Foundation Shell → console command execution
- [ ] Wire Foundation Tracker.on_change → CVar change notifications

### 14.3 Visual Debugging
- [ ] Implement debug draw (lines, boxes, spheres, text, arrows)
- [ ] Implement debug overlays (wireframe, collision, navmesh, audio)
- [ ] Wire `@debug_cheat` decorators → cheat command registration

### 14.4 Profiling
- [ ] Implement CPU profiler (hierarchical timer, flame graph)
- [ ] Implement GPU profiler (timestamp queries, pipeline stats)
- [ ] Implement memory profiler (allocation tracking, leak detection)
- [ ] Implement network profiler (bandwidth, latency, packet loss)
- [ ] Wire Foundation EventLog → profiling event capture
- [ ] Wire `@debug_safety` decorators → read/write tracking

### 14.5 Crash Handling
- [ ] Implement crash reporter (minidump, callstack, state capture)
- [ ] Implement assertion system (verify, check, ensure)
- [ ] Wire Foundation Session → crash state capture

### 14.6 Replay & Recording
- [ ] Implement input replay (record inputs → deterministic replay)
- [ ] Implement state recording (periodic snapshots + deltas)
- [ ] Wire Foundation EventLog → replay event source
- [ ] Wire Foundation DeltaSync → efficient replay storage
- [ ] Wire `@replay` decorators → replay system configuration

### 14.7 Testing Framework
- [ ] Implement in-engine test runner (unit, functional, integration, stress)
- [ ] Implement automated testing (bots, scenarios)
- [ ] Wire Foundation Shell → test execution from console

---

## 15. XR Layer

> **Ref:** `DIAGRAMS/ARCHITECTURE_XR.md`
> **Ref:** `DIAGRAMS/FULL_ARCHITECTURE.md` §XR Layer (L2255-2643)

### 15.1 XR Runtime
- [ ] Implement OpenXR integration (session, reference spaces)
- [ ] Implement HMD tracking (head pose, display specs)
- [ ] Register XR session as Resource via ResourceMeta

### 15.2 Controller & Hand Tracking
- [ ] Implement controller input (buttons, triggers, thumbsticks, haptics)
- [ ] Implement hand tracking (25 joints per hand, gestures)
- [ ] Wire input system → XR input sources

### 15.3 XR Rendering
- [ ] Implement stereo rendering (multi-view or instanced stereo)
- [ ] Implement foveated rendering (fixed and eye-tracked)
- [ ] Implement reprojection (ASW/ATW)
- [ ] Implement XR performance targets (90fps VR, 72fps standalone)

### 15.4 Eye Tracking
- [ ] Implement eye tracking data (gaze direction, fixation, pupil)
- [ ] Implement foveated rendering from eye tracking

### 15.5 Spatial Understanding
- [ ] Implement plane detection, mesh scanning
- [ ] Implement spatial anchors (persistent, shared)
- [ ] Implement passthrough (AR overlay)

### 15.6 XR Interaction
- [ ] Implement grab/touch interaction (direct manipulation)
- [ ] Implement ray interaction (pointer, teleport)
- [ ] Implement locomotion (teleport, smooth, snap turn)
- [ ] Implement comfort options (vignette, snap turn, comfort mode)

### 15.7 XR Avatars
- [ ] Implement body IK from HMD + controllers
- [ ] Implement face tracking for avatar expressions

---

## 16. Cross-Cutting Integration

> These items span multiple layers and ensure Trinity+Foundation work end-to-end.

### 16.1 Decorator → Runtime Binding
- [ ] For EVERY decorator in trinity/decorators/ (58 files), verify it has a runtime handler
- [ ] Implement runtime dispatch: decorator Step → engine system call
- [ ] Document which decorators are definition-only vs runtime-active

### 16.2 Foundation ↔ Engine Sync
- [ ] Implement per-frame Foundation flush: Tracker.flush_dirty() → collect all changes
- [ ] Implement per-frame EventLog tick advancement
- [ ] Implement query cache invalidation on archetype changes
- [ ] Implement Foundation Mirror update when components are added/removed

### 16.3 Descriptor Chain → Engine Pipeline
- [ ] Verify all 31 descriptors have correct Foundation integration
- [ ] Test descriptor chains of depth 3+ (e.g., Networked → Tracked → Validated → Storage)
- [ ] Verify descriptor_steps() accuracy for all descriptors
- [ ] Test Annotated syntax for all descriptor combinations

### 16.4 FlowForge → Engine
- [ ] Wire FlowForge node execution to engine systems (not just AST parsing)
- [ ] Implement live FlowForge → Foundation → engine state modification
- [ ] Test FlowForge round-trip: visual edit → codegen → runtime behavior

### 16.5 ShellLang → Engine
- [ ] Wire ShellLang World.query() to actual ECS query execution
- [ ] Wire ShellLang MUTATE to actual component modification through descriptors
- [ ] Wire AIInterface commands to engine operations
- [ ] Test Shell live debugging with running game

### 16.6 Mod Support
- [ ] Implement mod loading (discover, validate, load)
- [ ] Wire Foundation Capabilities → mod sandboxing
- [ ] Wire `@modding` decorators → mod-accessible API surface
- [ ] Wire Foundation SecureShell → mod scripting environment

---

## 17. Deterministic Simulation

> **Ref:** `TRINITY_LATEST.md` Part V
> **Ref:** `DIAGRAMS/FULL_ARCHITECTURE.md` §Frame Structure

### 17.1 Deterministic Core
- [ ] Implement simulation boundary (deterministic kernel vs presentation)
- [ ] Implement fixed-point math types (no floats in simulation)
- [ ] Implement deterministic RNG (seeded, reproducible)
- [ ] Implement command-based input (ordered, timestamped)

### 17.2 Snapshot & Rollback
- [ ] Implement hierarchical checksums (per-entity, per-archetype, global)
- [ ] Implement snapshot system (Foundation ContentStore → efficient snapshots)
- [ ] Implement rollback (Foundation Tracker.undo() → revert to snapshot)
- [ ] Implement desync detection (compare checksums across clients)

### 17.3 Replay
- [ ] Implement frame-perfect replay from EventLog
- [ ] Implement replay scrubbing (jump to any frame)
- [ ] Implement replay comparison (divergence detection)

### 17.4 Network Determinism
- [ ] Implement lockstep networking (wait for all inputs)
- [ ] Implement rollback networking (predict → rollback on mismatch)
- [ ] Wire Foundation DeltaSync → efficient state correction

---

## 18. Testing & Validation

### 18.1 Unit Tests
- [ ] Tests for every engine system in isolation
- [ ] Tests for every descriptor → Foundation integration
- [ ] Tests for every metaclass → Foundation registration
- [ ] Tests for every decorator → runtime handler

### 18.2 Integration Tests
- [ ] Test full frame loop (input → simulate → render → present)
- [ ] Test entity lifecycle (create → add components → update → destroy)
- [ ] Test Foundation observation (Tracker, EventLog, Mirror) with live engine
- [ ] Test FlowForge → Foundation → Trinity → engine round-trip
- [ ] Test ShellLang commands against live engine state

### 18.3 Performance Tests
- [ ] Benchmark ECS query performance (10K, 100K, 1M entities)
- [ ] Benchmark descriptor chain overhead (1, 3, 5 deep)
- [ ] Benchmark Foundation Tracker overhead per frame
- [ ] Benchmark serialization/deserialization throughput
- [ ] Benchmark network replication bandwidth

### 18.4 Determinism Tests
- [ ] Verify simulation produces identical output from identical input
- [ ] Verify snapshot → restore → continue produces same result
- [ ] Verify replay matches original execution bit-for-bit
- [ ] Verify cross-platform determinism (if applicable)

---

## Priority Order

The recommended implementation order (each phase enables the next):

### Phase A: Bootstrap (Sections 1, 3.1-3.4)
Engine loop, system scheduler, memory, math, ECS runtime.
**Enables:** Everything else.

### Phase B: Platform + Rendering (Sections 2, 5)
Window, RHI, frame graph, basic rendering.
**Enables:** Visual output, editor viewport.

### Phase C: Simulation + Animation (Sections 6, 7)
Physics, collision, animation playback.
**Enables:** Interactive gameplay.

### Phase D: Gameplay + Input (Sections 9.1-9.6)
Entities, AI, abilities, input.
**Enables:** Playable prototype.

### Phase E: Audio + UI (Sections 8, 10)
Audio playback, UI widgets.
**Enables:** Complete game experience.

### Phase F: Networking (Section 11)
Replication, prediction, netcode.
**Enables:** Multiplayer.

### Phase G: World + Resources (Sections 4, 12)
Asset pipeline, streaming, terrain, world partition.
**Enables:** Large worlds.

### Phase H: Tooling (Section 13)
Editor, content browser, visual scripting runtime.
**Enables:** Content creation.

### Phase I: Debug + Determinism (Sections 14, 17)
Profiling, replay, crash handling, deterministic simulation.
**Enables:** Production quality.

### Phase J: XR (Section 15)
VR/AR support.
**Enables:** Immersive experiences.

### Phase K: Cross-Cutting + Testing (Sections 16, 18)
Full integration verification, performance benchmarks.
**Enables:** Ship-ready confidence.

---

## Document References

| Document | Location | Covers |
|----------|----------|--------|
| Full Architecture | `DIAGRAMS/FULL_ARCHITECTURE.md` | All 14 layers unified |
| Platform | `DIAGRAMS/ARCHITECTURE_PLATFORM.md` | OS, RHI, Window |
| Core | `DIAGRAMS/ARCHITECTURE_CORE.md` | Memory, Math, Tasks, ECS |
| Resource | `DIAGRAMS/ARCHITECTURE_RESOURCE.md` | Assets, Streaming, Virtualization |
| Rendering | `DIAGRAMS/ARCHITECTURE_RENDERING.md` | GPU, Materials, Lighting, GI |
| Simulation | `DIAGRAMS/ARCHITECTURE_SIMULATION.md` | Physics, Collision, Destruction |
| Animation | `DIAGRAMS/ARCHITECTURE_ANIMATION.md` | Skeletal, IK, Motion Matching |
| Audio | `DIAGRAMS/ARCHITECTURE_AUDIO.md` | Spatial, Mixing, DSP, Music |
| Gameplay | `DIAGRAMS/ARCHITECTURE_GAMEPLAY.md` | Entities, AI, Abilities, Quests |
| UI | `DIAGRAMS/ARCHITECTURE_UI.md` | Widgets, Layout, Binding |
| Networking | `DIAGRAMS/ARCHITECTURE_NETWORKING.md` | Replication, Prediction, Netcode |
| World | `DIAGRAMS/ARCHITECTURE_WORLD.md` | Levels, Terrain, Streaming |
| Tooling | `DIAGRAMS/ARCHITECTURE_TOOLING.md` | Editor, Build Pipeline |
| Debug | `DIAGRAMS/ARCHITECTURE_DEBUG.md` | Logging, Profiling, Crash |
| XR | `DIAGRAMS/ARCHITECTURE_XR.md` | VR/AR/MR |
| Trinity Spec | `docs/TRINITY_LATEST.md` | Metaclasses, Descriptors, Decorators |
| Integration | `docs/GAME_ENGINE_INTEGRATION.md` | Trinity ↔ Foundation ↔ Engine |
