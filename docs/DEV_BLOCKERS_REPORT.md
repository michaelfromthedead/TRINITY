# DEV Implementation Blockers Report

**Generated:** 2026-06-03
**Total Blocked Tests:** 311 (7 Python + 304 Rust)

---

## Executive Summary

311 tests are written (TDD-style) but cannot run because the classes/modules they import don't exist yet. These are **not QA issues** — they are **DEV implementation tasks**.

| Category | Blocked Tests | Primary Blocker |
|----------|---------------|-----------------|
| Python TDD | 7 | Missing class implementations |
| Rust frame_graph | 79 | Internal modules not exported |
| Rust gpu_driven | 55 | Internal modules not exported |
| Rust resources | 47 | Internal modules not exported |
| Rust render_pipeline | 18 | Internal modules not exported |
| Rust profiling | 10 | Internal modules not exported |
| Rust debug | 188 | debug_utils not exported |
| Rust demoscene | 5 | Internal modules not exported |

---

## Python TDD Blockers (7 Tests)

### 1. test_crowd_system_t_an_9_9.py

**Missing Classes in `engine/animation/systems/crowd_system.py`:**

```python
# Enums needed:
SteeringMode          # RVO steering behavior modes
CullingMode           # Frustum culling strategies  
AnimationBakeMode     # How animation data is baked to textures

# Frustum Culling:
Plane                 # Frustum plane representation
Frustum               # View frustum for culling

# RVO/ORCA Steering (Reciprocal Velocity Obstacles):
VelocityObstacle      # Velocity obstacle representation
ORCALine              # Optimal Reciprocal Collision Avoidance line
RVOConfig             # RVO algorithm configuration
RVOSteering           # RVO steering calculator

# GPU Instance Buffer:
CrowdInstanceData     # Per-instance GPU data
CrowdInstanceBuffer   # GPU buffer for instances
```

**Effort Estimate:** 2-3 days (RVO algorithm is complex)

---

### 2. test_ik_system_t_an_9_4.py

**Missing Classes in `engine/animation/systems/ik_system.py`:**

```python
IKSolverType          # FABRIK, CCD, Analytical, etc.
IKHintType            # Pole vector, rotation hints
IKGoal                # Target position/rotation for IK
IKChainBone           # Bone in IK chain
IKSolveResult         # Result of IK solve
IKComponent           # ECS component for IK
IKSystemStats         # Performance statistics
```

**Also needs:** `engine/animation/ik/fullbody.py` with `SkeletonMapping`, `BodyPart`

**Effort Estimate:** 1-2 days

---

### 3. test_skinning_system_t_an_9_6.py

**Missing Classes in `engine/animation/systems/skinning_system.py`:**

```python
# Enums:
SkinningMethod        # Linear, dual-quaternion, etc.
SkinningBackend       # CPU, GPU compute, GPU vertex
LODInfluenceLevel     # Bone influence LOD levels

# Data Structures:
BoneInfluence         # (bone_index, weight) pair
VertexSkinData        # Per-vertex skinning data
SkinningData          # Full mesh skinning data
MeshData              # Mesh geometry reference
GPUDispatchConfig     # Compute dispatch parameters
SkinningDispatch      # Single skinning dispatch
SkinningBatch         # Batched skinning operations
```

**Effort Estimate:** 1-2 days

---

### 4. test_motion_matching_system_t_an_9_7.py

**Missing Classes in `engine/animation/systems/motion_matching_system.py`:**

```python
FallbackReason        # Why motion matching fell back
MotionMatchingMode    # Modes (full, trajectory-only, etc.)
MotionMatchingConfig  # Configuration
MotionMatchingStatistics  # Performance stats
MotionMatchingComponent   # ECS component
MotionMatchingInput   # Input state
TrajectoryState       # Future trajectory prediction
MotionInput           # Legacy compat
MotionFeature         # Feature vector
```

**Also needs:** `engine/animation/motionmatching/database.py` with motion database classes

**Effort Estimate:** 2-3 days (motion matching is complex)

---

### 5. test_animation_graph_system_t_an_9_3.py

**Missing Classes in `engine/animation/systems/animation_graph_system.py`:**

```python
AnimationGraphComponent   # ECS component
BoneTransformSoA          # SoA bone transforms
DirtyFlags                # What needs update
AnimationDirtyState       # Dirty tracking
StateMachineOutput        # State machine result
ClipSampler               # Animation clip sampling
BlendTreeEvaluator        # Blend tree evaluation
```

**Effort Estimate:** 1-2 days

---

## Rust Pending Test Blockers (304 Tests)

### Category 1: frame_graph Internal Modules (79 tests)

**Current State:**
- `frame_graph/mod.rs` exports types directly (5000+ lines)
- Internal submodules exist but aren't exported:
  - `aliasing.rs` - AliasPolicy, AliasAnalyzer
  - `barriers.rs` - BarrierBatch, BarrierResolver
  - `async_compute.rs` - AsyncPartitioner
  - `scheduling.rs` - PassScheduler
  - `execution.rs` - FrameGraphExecutor

**Tests Need:**
```rust
// These exist but aren't exported:
BarrierOptimizer        // In barriers.rs, not public
FrameGraphCompiler      // Doesn't exist
BarrierResolveContext   // Doesn't exist
PassValidator           // Doesn't exist
PassRegistry            // Doesn't exist
CompilerProfile         // Doesn't exist
JsonExporter            // Doesn't exist
```

**Fix Options:**
1. **Export existing modules:** Add `pub mod aliasing;` etc. to mod.rs
2. **Create missing types:** Some types referenced by tests don't exist at all

**Effort Estimate:** 1 day to export existing, 3-5 days to implement missing

---

### Category 2: gpu_driven Internal Modules (55 tests)

**Tests Need:**
```rust
// From gpu_driven module:
BindlessBufferRegistry
BindlessTextureRegistry
IndirectDrawBuffer
CullingDispatcher
OcclusionCuller
```

**Current State:** `gpu_driven/mod.rs` exports some types but not all submodules

**Effort Estimate:** 1 day to export, 2-3 days if implementation needed

---

### Category 3: resources Internal Modules (47 tests)

**Tests Need:**
```rust
// From resources module:
BufferPool
TexturePool
BindGroupCache
BindGroupLayoutCache
ResourceTracker
```

**Current State:** Module exists at `resources.rs` but may not export everything

**Effort Estimate:** 1-2 days

---

### Category 4: render_pipeline Internal Modules (18 tests)

**Tests Need:**
```rust
// From render_pipeline module:
DepthStencilState
BlendState
RasterState
PipelineLayoutBuilder
```

**Current State:** Module exists, some types already exported

**Effort Estimate:** 0.5-1 day

---

### Category 5: debug_utils (188 tests blocked!)

**This is the LARGEST blocker!**

**Tests Need:**
```rust
// Debug utilities not exported:
debug_utils module
debug_marker macro
debug_marker_if macro
debug_marker_timed macro
DebugScope
ProfileScope
```

**Current State:** Debug macros/utilities may not exist or aren't exported

**Effort Estimate:** 1-2 days to implement debug infrastructure

---

### Category 6: profiling Internal Modules (10 tests)

**Tests Need:**
```rust
// Profiling infrastructure:
BottleneckAnalyzer
GpuTimestampQuery
FrameProfiler
```

**Effort Estimate:** 1 day

---

### Category 7: demoscene Internal Modules (5 tests)

**Tests Need:**
```rust
// From demoscene module:
demoscene::minimal
demoscene::multipass
demoscene::post_integration
```

**Current State:** `demoscene/mod.rs` doesn't export these submodules

**Effort Estimate:** 0.5 day

---

## Recommended Priority Order

### Phase 1: Quick Wins (1-2 days)
Export existing modules that just need `pub mod`:
1. `render_pipeline` submodules
2. `demoscene` submodules
3. `profiling` submodules
4. **debug_utils** (if it exists, just export it)

**Potential Unblock:** ~220 tests

### Phase 2: Frame Graph Exports (2-3 days)
1. Export `frame_graph` internal modules
2. Add missing convenience types

**Potential Unblock:** ~80 tests

### Phase 3: GPU Infrastructure (2-3 days)
1. `gpu_driven` exports
2. `resources` exports

**Potential Unblock:** ~100 tests

### Phase 4: Python Animation Systems (5-7 days)
1. RVO/ORCA steering (most complex)
2. Motion matching database
3. IK system types
4. Skinning system
5. Animation graph evaluator

**Potential Unblock:** 7 tests

---

## Summary

| Phase | Days | Tests Unblocked |
|-------|------|-----------------|
| Quick Wins | 1-2 | ~220 |
| Frame Graph | 2-3 | ~80 |
| GPU Infrastructure | 2-3 | ~100 |
| Python Animation | 5-7 | 7 |
| **Total** | **10-15 days** | **311** |

The bulk of blocked tests (220+) can be unblocked by simply **exporting existing modules**. The Python tests require actual implementation work.
