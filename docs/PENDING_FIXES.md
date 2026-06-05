# Pending Fixes — Test Unblock Status

**Date:** 2026-06-03  
**Session:** Module export unblock effort

---

## Summary

Exported existing Rust modules to unblock **847 tests**. However, **296 Rust tests** and **7 Python tests** still require DEV implementation work — not just exports.

---

## What Was Fixed (847 Tests Unblocked)

### Module Exports Added to lib.rs
```rust
pub mod debug;
pub mod debug_utils;
pub mod resource_state;
```

### Module Exports Added to frame_graph/mod.rs
```rust
pub mod barriers;
pub mod swap;
```

### Module Exports Added to gpu_driven/mod.rs
```rust
pub mod compact;
pub mod frustum;
pub mod hiz_occlusion;
pub mod hiz_pyramid;
pub mod hzb;
pub mod lod;
pub mod stream_compact;
pub mod visibility_flags;
// + existing: buffers, indirect_draw, material_table, mesh_table, sort, texture_table
```

### Module Exports Added to demoscene/mod.rs
```rust
pub mod bootstrap;
pub mod depth_barriers;
pub mod hybrid_depth;
```

### Dependencies Added to Cargo.toml
```toml
bitflags = "2"
arc-swap = "1"
```

### Test Files Enabled (moved from tests_pending/ to tests/)
| File | Tests |
|------|-------|
| blackbox_resource_state.rs | 80 |
| blackbox_debug_utils.rs | 60 |
| blackbox_debug_markers.rs | 131 |
| blackbox_barrier_batching.rs | 71 |
| blackbox_layout_transitions.rs | 63 |
| blackbox_frame_acquisition.rs | 186 |
| blackbox_sampler.rs | 83 |
| blackbox_visualization.rs | 125 |
| **Total** | **847** |

---

## RUST: 296 Pending Tests — Detailed Breakdown

### Module Dependency Analysis

| Module | Tests Blocked | Status |
|--------|---------------|--------|
| frame_graph | 91 | Needs internal types/fields |
| gpu_driven | 61 | Needs submodule exports |
| resources | 44 | Needs submodule exports |
| render_pipeline | 21 | Needs internal types |
| presentation | 20 | Needs internal types |
| device | 13 | Needs methods |
| profiling | 12 | Needs submodules |
| backend | 10 | Needs submodules |
| shaders | 9 | Needs module |
| debug | 7 | Partially done |
| compute_library | 7 | Module doesn't exist |
| demoscene | 5 | Needs more submodules |
| query_pool | 4 | Module doesn't exist |
| frame_sync | 4 | Module doesn't exist |
| buffer_mapping | 3 | Module doesn't exist |

---

### Blocker 1: frame_graph Missing Types (91 tests)

**File:** `crates/renderer-backend/src/frame_graph/mod.rs`

Tests need these types that don't exist or aren't exported:

```rust
// MISSING from CompiledFrameGraph struct:
pub struct CompiledFrameGraph {
    pub scheduled_passes: Vec<ScheduledPass>,      // ADD THIS
    pub interference_graph: InterferenceGraph,     // ADD THIS
    pub compilation_time_us: u64,                  // ADD THIS
    pub eliminated_pass_names: Vec<String>,        // ADD THIS
}

// MISSING types that tests import:
pub struct ScheduledPass { /* pass scheduling info */ }
pub struct BarrierTuple { /* barrier pair */ }
pub struct CompilerProfile { /* compilation stats */ }
pub fn is_pass_live(pass: PassIndex) -> bool { /* check if pass was culled */ }

// MISSING submodule exports:
pub mod graph;           // graph traversal utilities
pub mod passes;          // pass types and utilities  
pub mod resources;       // resource lifetime tracking
pub mod external;        // external resource handles
pub mod scheduling;      // pass scheduling algorithm
pub mod async_compute;   // async compute partitioning
pub mod wgpu_barriers;   // wgpu barrier generation
```

**Implementation Notes:**
- `ScheduledPass` should contain: pass index, barriers before/after, resource states
- `InterferenceGraph` tracks which resources are live simultaneously
- `is_pass_live()` checks the dead-pass elimination result

---

### Blocker 2: gpu_driven Missing Exports (61 tests)

**File:** `crates/renderer-backend/src/gpu_driven/mod.rs`

Tests need these submodules exported:

```rust
// NEEDS pub mod:
pub mod meshlet;            // Meshlet data structures
pub mod meshlet_generator;  // Meshlet generation
pub mod lod_buffer;         // LOD buffer management
pub mod scene_data;         // Scene data buffers
pub mod object_data;        // Per-object GPU data
pub mod buffer_registry;    // Buffer tracking
pub mod build_indirect;     // Indirect draw building

// NEEDS re-export at mod.rs level:
pub use indirect_draw::{DrawIndirectArgs, DrawIndexedIndirectArgs, DispatchIndirectArgs};
pub struct CountBuffer { /* atomic counter buffer */ }
pub struct FrustumBuffer { /* frustum planes */ }
pub struct SceneDataBuffers { /* scene-wide GPU data */ }
```

**Implementation Notes:**
- `CountBuffer` wraps an atomic counter buffer for GPU counting
- `FrustumBuffer` holds 6 frustum planes for GPU culling
- These are mostly struct definitions + buffer management

---

### Blocker 3: resources Missing Submodules (44 tests)

**File:** `crates/renderer-backend/src/resources.rs` or `resources/mod.rs`

Tests need these submodules:

```rust
// NEEDS to exist as submodules:
pub mod buffer;              // Buffer creation/management
pub mod texture;             // Texture creation/management
pub mod sampler;             // Sampler state
pub mod sampler_cache;       // Sampler deduplication
pub mod bind_group_cache;    // BindGroup caching
pub mod bind_group_layout_cache;  // BindGroupLayout caching
pub mod buffer_pool;         // Buffer pooling/reuse
pub mod deferred_destroyer;  // Deferred resource destruction
pub mod bindless_buffers;    // Bindless buffer arrays
pub mod bindless_textures;   // Bindless texture arrays
pub mod texture_uploads;     // Async texture uploads
pub mod texture_formats;     // Format utilities
pub mod mip_generator;       // Mipmap generation
pub mod index_allocator;     // Index allocation for bindless

// Key types needed:
pub struct BindGroupLayoutCache { /* layout deduplication */ }
pub struct TrinitySamplerDescriptor { /* sampler config */ }
```

---

### Blocker 4: Missing Modules (must be created from scratch)

#### compute_library (7 tests)
**Create:** `crates/renderer-backend/src/compute_library.rs`
```rust
pub struct ComputeLibrary {
    pipelines: HashMap<String, ComputePipeline>,
}

impl ComputeLibrary {
    pub fn register(&mut self, name: &str, shader: &str) -> Result<(), Error>;
    pub fn get(&self, name: &str) -> Option<&ComputePipeline>;
    pub fn dispatch(&self, name: &str, encoder: &mut CommandEncoder, workgroups: [u32; 3]);
}
```

#### compute_pass (tests need this)
**Create:** `crates/renderer-backend/src/compute_pass.rs`
```rust
pub struct ComputePass<'a> {
    pass: wgpu::ComputePass<'a>,
}

impl<'a> ComputePass<'a> {
    pub fn set_pipeline(&mut self, pipeline: &ComputePipeline);
    pub fn set_bind_group(&mut self, index: u32, bind_group: &BindGroup);
    pub fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32);
    pub fn dispatch_workgroups_indirect(&mut self, buffer: &Buffer, offset: u64);
}
```

#### buffer_mapping (3 tests)
**Create:** `crates/renderer-backend/src/buffer_mapping.rs`
```rust
pub struct MappedBuffer<'a> {
    slice: wgpu::BufferSlice<'a>,
    view: wgpu::BufferView<'a>,
}

pub async fn map_buffer_async(buffer: &Buffer, mode: MapMode) -> Result<MappedBuffer, Error>;
pub fn map_buffer_blocking(buffer: &Buffer, mode: MapMode) -> Result<MappedBuffer, Error>;
```

#### frame_sync (4 tests)
**Create:** `crates/renderer-backend/src/frame_sync.rs`
```rust
pub struct FrameSynchronizer {
    frame_index: u64,
    fences: [Option<wgpu::SubmissionIndex>; 3],
}

impl FrameSynchronizer {
    pub fn begin_frame(&mut self) -> FrameContext;
    pub fn end_frame(&mut self, submission: wgpu::SubmissionIndex);
    pub fn wait_for_frame(&self, frame: u64);
}
```

#### query_pool (4 tests)
**Create:** `crates/renderer-backend/src/query_pool.rs`
```rust
pub struct TimestampQueryPool {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
}

impl TimestampQueryPool {
    pub fn new(device: &Device, count: u32) -> Self;
    pub fn write_timestamp(&self, encoder: &mut CommandEncoder, index: u32);
    pub fn resolve(&self, encoder: &mut CommandEncoder);
    pub fn read_results(&self) -> Vec<u64>;
}
```

---

### Blocker 5: Missing Methods

#### RhiDevice::try_new_headless() (13 lib tests)
**File:** `crates/renderer-backend/src/rhi_device.rs`
```rust
impl RhiDevice {
    /// Create a headless device for testing (no surface required)
    pub fn try_new_headless() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor::default(),
            None,
        )).ok()?;
        Some(Self::new(device, queue))
    }
}
```

#### InterferenceGraph::all_handles()
**File:** `crates/renderer-backend/src/frame_graph/mod.rs` (or aliasing.rs)
```rust
impl InterferenceGraph {
    pub fn all_handles(&self) -> impl Iterator<Item = ResourceHandle> + '_ {
        self.nodes.keys().copied()
    }
}
```

---

## PYTHON: 7 TDD Tests — Detailed Implementation Specs

### 1. test_crowd_system_t_an_9_9.py

**File:** `engine/animation/systems/crowd_system.py`

```python
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

class SteeringMode(Enum):
    """RVO steering behavior modes"""
    DISABLED = auto()
    RVO = auto()           # Reciprocal Velocity Obstacles
    ORCA = auto()          # Optimal Reciprocal Collision Avoidance
    HYBRID = auto()        # Combined approach

class CullingMode(Enum):
    """Frustum culling strategies"""
    NONE = auto()
    FRUSTUM = auto()       # View frustum culling
    OCCLUSION = auto()     # Occlusion culling
    DISTANCE = auto()      # Distance-based culling

class AnimationBakeMode(Enum):
    """How animation data is baked to textures"""
    NONE = auto()
    VERTEX = auto()        # Vertex animation texture
    BONE = auto()          # Bone matrix texture

@dataclass
class Plane:
    """Frustum plane representation (ax + by + cz + d = 0)"""
    normal: 'Vec3'
    distance: float
    
    def signed_distance(self, point: 'Vec3') -> float:
        return self.normal.dot(point) + self.distance

@dataclass
class Frustum:
    """View frustum for culling (6 planes)"""
    planes: List[Plane] = field(default_factory=list)
    
    def contains_point(self, point: 'Vec3') -> bool:
        return all(p.signed_distance(point) >= 0 for p in self.planes)
    
    def contains_sphere(self, center: 'Vec3', radius: float) -> bool:
        return all(p.signed_distance(center) >= -radius for p in self.planes)

@dataclass
class VelocityObstacle:
    """Velocity obstacle for collision avoidance"""
    apex: 'Vec3'           # Apex of the VO cone
    left_leg: 'Vec3'       # Left boundary direction
    right_leg: 'Vec3'      # Right boundary direction
    
    def contains_velocity(self, velocity: 'Vec3') -> bool:
        """Check if velocity is inside the obstacle cone"""
        rel = velocity - self.apex
        # Cross product signs determine which side of each leg
        return (self.left_leg.cross(rel).y >= 0 and 
                rel.cross(self.right_leg).y >= 0)

@dataclass  
class ORCALine:
    """ORCA half-plane constraint"""
    point: 'Vec3'          # Point on the line
    direction: 'Vec3'      # Line direction (unit vector)
    
    def signed_distance(self, velocity: 'Vec3') -> float:
        """Distance from velocity to the constraint line"""
        return (velocity - self.point).dot(
            Vec3(-self.direction.z, 0, self.direction.x)
        )

@dataclass
class RVOConfig:
    """RVO algorithm configuration"""
    neighbor_distance: float = 15.0    # How far to look for neighbors
    max_neighbors: int = 10            # Max neighbors to consider
    time_horizon: float = 10.0         # Planning time horizon
    time_horizon_obstacle: float = 5.0 # Time horizon for static obstacles
    radius: float = 0.5                # Agent radius
    max_speed: float = 2.0             # Maximum agent speed

class RVOSteering:
    """RVO steering calculator"""
    def __init__(self, config: RVOConfig):
        self.config = config
        self._orca_lines: List[ORCALine] = []
    
    def compute_velocity(
        self, 
        current_pos: 'Vec3',
        current_vel: 'Vec3', 
        preferred_vel: 'Vec3',
        neighbors: List['CrowdAgent']
    ) -> 'Vec3':
        """Compute collision-free velocity using ORCA"""
        self._orca_lines.clear()
        # Build ORCA lines from neighbors
        for neighbor in neighbors:
            self._add_orca_line(current_pos, current_vel, neighbor)
        # Linear program to find best velocity
        return self._solve_linear_program(preferred_vel)
    
    def _add_orca_line(self, pos, vel, neighbor) -> None:
        # Implementation of ORCA line computation
        pass
    
    def _solve_linear_program(self, preferred: 'Vec3') -> 'Vec3':
        # 2D linear programming solver
        pass

@dataclass
class CrowdInstanceData:
    """Per-instance GPU data for crowd rendering"""
    transform: 'Mat4'              # World transform
    animation_offset: float        # Offset into animation texture
    animation_speed: float = 1.0   # Playback speed multiplier
    lod_level: int = 0             # Current LOD level
    flags: int = 0                 # Visibility/state flags

class CrowdInstanceBuffer:
    """GPU buffer for crowd instances"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.instances: List[CrowdInstanceData] = []
        self._dirty = True
    
    def add(self, instance: CrowdInstanceData) -> int:
        """Add instance, return index"""
        idx = len(self.instances)
        self.instances.append(instance)
        self._dirty = True
        return idx
    
    def update(self, index: int, instance: CrowdInstanceData) -> None:
        self.instances[index] = instance
        self._dirty = True
    
    def upload(self, device: 'GPUDevice') -> None:
        """Upload to GPU buffer"""
        if self._dirty:
            # Pack instances into GPU buffer
            self._dirty = False
```

**Effort:** 2-3 days (RVO/ORCA algorithm is mathematically complex)

---

### 2. test_ik_system_t_an_9_4.py

**File:** `engine/animation/systems/ik_system.py`

```python
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

class IKSolverType(Enum):
    """IK solver algorithm types"""
    FABRIK = auto()        # Forward And Backward Reaching IK
    CCD = auto()           # Cyclic Coordinate Descent
    ANALYTICAL = auto()    # Closed-form (2-bone only)
    JACOBIAN = auto()      # Jacobian transpose/pseudoinverse

class IKHintType(Enum):
    """Types of IK hints/constraints"""
    NONE = auto()
    POLE_VECTOR = auto()   # Pole target for elbow/knee direction
    ROTATION = auto()      # Preferred rotation
    TWIST = auto()         # Twist limit

@dataclass
class IKGoal:
    """Target for IK solver"""
    position: 'Vec3'
    rotation: Optional['Quat'] = None
    weight: float = 1.0
    hint_type: IKHintType = IKHintType.NONE
    hint_value: Optional['Vec3'] = None

@dataclass
class IKChainBone:
    """Single bone in an IK chain"""
    index: int                     # Bone index in skeleton
    length: float                  # Bone length
    min_angle: float = -180.0      # Joint limit min (degrees)
    max_angle: float = 180.0       # Joint limit max (degrees)
    axis: 'Vec3' = field(default_factory=lambda: Vec3(0, 0, 1))  # Rotation axis

@dataclass
class IKSolveResult:
    """Result of IK solve"""
    solved: bool                   # Did solver converge?
    iterations: int                # Iterations used
    error: float                   # Final position error
    rotations: List['Quat']        # Solved bone rotations

@dataclass
class IKComponent:
    """ECS component for IK"""
    chain: List[IKChainBone]
    goal: Optional[IKGoal] = None
    solver_type: IKSolverType = IKSolverType.FABRIK
    max_iterations: int = 10
    tolerance: float = 0.001
    enabled: bool = True

@dataclass
class IKSystemStats:
    """Performance statistics for IK system"""
    chains_solved: int = 0
    total_iterations: int = 0
    average_error: float = 0.0
    solve_time_ms: float = 0.0
```

**Also needs:** `engine/animation/ik/fullbody.py`
```python
from enum import Enum, auto

class BodyPart(Enum):
    SPINE = auto()
    LEFT_ARM = auto()
    RIGHT_ARM = auto()
    LEFT_LEG = auto()
    RIGHT_LEG = auto()
    HEAD = auto()

@dataclass
class SkeletonMapping:
    """Maps body parts to bone indices"""
    mappings: Dict[BodyPart, List[int]] = field(default_factory=dict)
    
    def get_chain(self, part: BodyPart) -> List[int]:
        return self.mappings.get(part, [])
```

**Effort:** 1-2 days

---

### 3. test_skinning_system_t_an_9_6.py

**File:** `engine/animation/systems/skinning_system.py`

```python
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

class SkinningMethod(Enum):
    """Skinning algorithm"""
    LINEAR = auto()           # Linear blend skinning (LBS)
    DUAL_QUATERNION = auto()  # Dual quaternion skinning (DQS)
    BLEND = auto()            # Hybrid LBS+DQS

class SkinningBackend(Enum):
    """Where skinning computation runs"""
    CPU = auto()
    GPU_COMPUTE = auto()      # Compute shader
    GPU_VERTEX = auto()       # Vertex shader

class LODInfluenceLevel(Enum):
    """Bone influence count per LOD"""
    LOD0 = 4  # 4 bones per vertex
    LOD1 = 2  # 2 bones per vertex
    LOD2 = 1  # 1 bone per vertex

@dataclass
class BoneInfluence:
    """Single bone influence on a vertex"""
    bone_index: int
    weight: float

@dataclass
class VertexSkinData:
    """Per-vertex skinning data"""
    influences: List[BoneInfluence] = field(default_factory=list)
    
    def normalize(self) -> None:
        """Normalize weights to sum to 1.0"""
        total = sum(i.weight for i in self.influences)
        if total > 0:
            for i in self.influences:
                i.weight /= total

@dataclass
class SkinningData:
    """Full mesh skinning data"""
    vertices: List[VertexSkinData]
    bone_names: List[str]
    bind_poses: List['Mat4']  # Inverse bind pose matrices

@dataclass
class MeshData:
    """Mesh geometry reference"""
    vertex_count: int
    index_count: int
    vertex_buffer: 'GPUBufferHandle'
    index_buffer: 'GPUBufferHandle'

@dataclass
class GPUDispatchConfig:
    """Compute dispatch parameters"""
    workgroup_size: Tuple[int, int, int] = (64, 1, 1)
    workgroup_count: Tuple[int, int, int] = (1, 1, 1)

@dataclass
class SkinningDispatch:
    """Single skinning dispatch"""
    mesh: MeshData
    bone_matrices: 'GPUBufferHandle'
    output_buffer: 'GPUBufferHandle'
    vertex_count: int

@dataclass
class SkinningBatch:
    """Batched skinning operations"""
    dispatches: List[SkinningDispatch] = field(default_factory=list)
    
    def add(self, dispatch: SkinningDispatch) -> None:
        self.dispatches.append(dispatch)
    
    def execute(self, encoder: 'CommandEncoder') -> None:
        for d in self.dispatches:
            # Dispatch compute shader
            pass

@dataclass
class SkinningStats:
    """Performance statistics"""
    vertices_skinned: int = 0
    dispatches: int = 0
    time_ms: float = 0.0

@dataclass
class LODComponent:
    """LOD tracking for skinning"""
    current_lod: int = 0
    influence_level: LODInfluenceLevel = LODInfluenceLevel.LOD0

@dataclass
class GPUBufferHandle:
    """Handle to GPU buffer"""
    id: int
    size: int

@dataclass
class GPUCapabilities:
    """GPU feature support"""
    compute_shaders: bool = True
    max_workgroup_size: int = 256

@dataclass
class SkinnedMeshComponent:
    """ECS component for skinned mesh"""
    mesh: MeshData
    skinning_data: SkinningData
    method: SkinningMethod = SkinningMethod.LINEAR
    backend: SkinningBackend = SkinningBackend.GPU_COMPUTE

class GPUSkinningDispatcher:
    """Dispatches GPU skinning work"""
    def __init__(self, device: 'GPUDevice'):
        self.device = device
        self.pipeline = None  # Compute pipeline
    
    def dispatch(self, batch: SkinningBatch) -> None:
        pass

class CPUSkinningFallback:
    """CPU fallback when GPU unavailable"""
    def skin_mesh(self, mesh: MeshData, bones: List['Mat4']) -> None:
        pass
```

**Effort:** 1-2 days

---

### 4. test_motion_matching_system_t_an_9_7.py

**File:** `engine/animation/systems/motion_matching_system.py`

```python
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

class FallbackReason(Enum):
    """Why motion matching fell back to blend tree"""
    NONE = auto()
    NO_DATABASE = auto()
    SEARCH_FAILED = auto()
    QUALITY_TOO_LOW = auto()
    TRANSITION_LOCKED = auto()

class MotionMatchingMode(Enum):
    """Motion matching operating modes"""
    FULL = auto()              # Full motion matching
    TRAJECTORY_ONLY = auto()   # Only match trajectory
    POSE_ONLY = auto()         # Only match pose
    DISABLED = auto()          # Use fallback blend tree

@dataclass
class MotionMatchingConfig:
    """Motion matching configuration"""
    search_interval: float = 0.1    # Seconds between searches
    quality_threshold: float = 0.8  # Min match quality
    blend_time: float = 0.2         # Transition blend duration
    trajectory_weight: float = 1.0  # Trajectory feature weight
    pose_weight: float = 1.0        # Pose feature weight
    velocity_weight: float = 0.5    # Velocity feature weight

@dataclass
class MotionMatchingStatistics:
    """Runtime statistics"""
    searches_per_second: float = 0.0
    average_quality: float = 0.0
    fallback_ratio: float = 0.0
    database_size: int = 0

@dataclass
class TrajectoryState:
    """Future trajectory prediction"""
    positions: List['Vec3'] = field(default_factory=list)  # Future positions
    directions: List['Vec3'] = field(default_factory=list)  # Future facing dirs
    times: List[float] = field(default_factory=list)        # Sample times

@dataclass
class MotionMatchingInput:
    """Input state for motion matching"""
    desired_velocity: 'Vec3'
    desired_facing: 'Vec3'
    trajectory: TrajectoryState
    
@dataclass
class MotionInput:
    """Legacy compatibility alias"""
    velocity: 'Vec3'
    direction: 'Vec3'

@dataclass
class MotionFeature:
    """Feature vector for matching"""
    values: List[float] = field(default_factory=list)
    
    def distance(self, other: 'MotionFeature') -> float:
        """Compute weighted distance"""
        return sum((a - b) ** 2 for a, b in zip(self.values, other.values)) ** 0.5

@dataclass
class MotionMatchingComponent:
    """ECS component for motion matching"""
    config: MotionMatchingConfig = field(default_factory=MotionMatchingConfig)
    mode: MotionMatchingMode = MotionMatchingMode.FULL
    current_clip_index: int = -1
    current_time: float = 0.0
    last_search_time: float = 0.0
    fallback_reason: FallbackReason = FallbackReason.NONE
```

**Also needs:** `engine/animation/motionmatching/database.py`, `features.py`, `search.py`, `transition.py`

**Effort:** 2-3 days (motion matching is algorithmically complex)

---

### 5. test_animation_graph_system_t_an_9_3.py

**File:** `engine/animation/systems/animation_graph_system.py`

```python
from enum import Flag, auto
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

class DirtyFlags(Flag):
    """What needs update in animation system"""
    NONE = 0
    PARAMETERS = auto()     # Graph parameters changed
    POSE = auto()           # Pose needs recalc
    BLEND_WEIGHTS = auto()  # Blend weights changed
    STATE = auto()          # State machine state changed
    ALL = PARAMETERS | POSE | BLEND_WEIGHTS | STATE

@dataclass
class AnimationDirtyState:
    """Tracks what needs updating"""
    flags: DirtyFlags = DirtyFlags.NONE
    
    def mark(self, flag: DirtyFlags) -> None:
        self.flags |= flag
    
    def clear(self, flag: DirtyFlags) -> None:
        self.flags &= ~flag
    
    def is_dirty(self, flag: DirtyFlags) -> bool:
        return bool(self.flags & flag)

@dataclass
class BoneTransformSoA:
    """Structure-of-Arrays bone transforms for SIMD"""
    positions_x: List[float] = field(default_factory=list)
    positions_y: List[float] = field(default_factory=list)
    positions_z: List[float] = field(default_factory=list)
    rotations_x: List[float] = field(default_factory=list)
    rotations_y: List[float] = field(default_factory=list)
    rotations_z: List[float] = field(default_factory=list)
    rotations_w: List[float] = field(default_factory=list)
    scales_x: List[float] = field(default_factory=list)
    scales_y: List[float] = field(default_factory=list)
    scales_z: List[float] = field(default_factory=list)
    
    @classmethod
    def from_transforms(cls, transforms: List['Transform']) -> 'BoneTransformSoA':
        result = cls()
        for t in transforms:
            result.positions_x.append(t.position.x)
            result.positions_y.append(t.position.y)
            result.positions_z.append(t.position.z)
            result.rotations_x.append(t.rotation.x)
            result.rotations_y.append(t.rotation.y)
            result.rotations_z.append(t.rotation.z)
            result.rotations_w.append(t.rotation.w)
            result.scales_x.append(t.scale.x)
            result.scales_y.append(t.scale.y)
            result.scales_z.append(t.scale.z)
        return result

@dataclass
class StateMachineOutput:
    """Output from state machine evaluation"""
    current_state: str
    next_state: Optional[str] = None
    transition_progress: float = 0.0
    clips_to_sample: List[Tuple[str, float]] = field(default_factory=list)

class ClipSampler:
    """Samples animation clips at given time"""
    def __init__(self, clip: 'AnimationClip'):
        self.clip = clip
    
    def sample(self, time: float) -> 'Pose':
        """Sample pose at time"""
        pass
    
    def sample_bone(self, bone_index: int, time: float) -> 'Transform':
        """Sample single bone"""
        pass

class BlendTreeEvaluator:
    """Evaluates blend trees"""
    def __init__(self):
        self._cache: Dict[int, 'Pose'] = {}
    
    def evaluate(self, tree: 'BlendTree', params: Dict[str, float]) -> 'Pose':
        """Evaluate blend tree with parameters"""
        pass
    
    def evaluate_1d(self, tree: 'BlendTree1D', blend_param: float) -> 'Pose':
        """Evaluate 1D blend tree"""
        pass
    
    def evaluate_2d(self, tree: 'BlendTree2D', x: float, y: float) -> 'Pose':
        """Evaluate 2D blend tree"""
        pass

@dataclass
class AnimationGraphComponent:
    """ECS component for animation graph"""
    graph: 'AnimationGraph'
    parameters: Dict[str, float] = field(default_factory=dict)
    dirty_state: AnimationDirtyState = field(default_factory=AnimationDirtyState)
    current_pose: Optional['Pose'] = None
    bone_transforms: Optional[BoneTransformSoA] = None
```

**Effort:** 1-2 days

---

### 6. test_animation_pipeline.py (integration)

**File:** `engine/animation/systems/procedural_system.py`

```python
from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod

class ProceduralModifier(ABC):
    """Base class for procedural animation modifiers"""
    
    @abstractmethod
    def apply(self, pose: 'Pose', dt: float) -> 'Pose':
        """Apply modifier to pose"""
        pass
    
    @property
    @abstractmethod
    def priority(self) -> int:
        """Execution priority (lower = earlier)"""
        pass

@dataclass
class BreathingModifier(ProceduralModifier):
    """Adds breathing motion to spine/chest"""
    spine_bones: List[int] = field(default_factory=list)
    amplitude: float = 0.02
    frequency: float = 0.25
    _phase: float = 0.0
    
    def apply(self, pose: 'Pose', dt: float) -> 'Pose':
        self._phase += dt * self.frequency * 2 * math.pi
        offset = math.sin(self._phase) * self.amplitude
        # Apply to spine bones
        return pose
    
    @property
    def priority(self) -> int:
        return 100

@dataclass
class SpringBoneModifier(ProceduralModifier):
    """Physics-based secondary motion (hair, cloth, etc.)"""
    bone_indices: List[int] = field(default_factory=list)
    stiffness: float = 100.0
    damping: float = 5.0
    gravity: float = -9.8
    
    def apply(self, pose: 'Pose', dt: float) -> 'Pose':
        # Spring physics simulation
        return pose
    
    @property
    def priority(self) -> int:
        return 200
```

**Effort:** 0.5 days

---

### 7. test_behavior_config.py

**File:** `engine/animation/config.py`

```python
from dataclasses import dataclass, field
from typing import Any, Dict, TypeVar, Generic

T = TypeVar('T')

class MutableConfig(Generic[T]):
    """Runtime-mutable configuration wrapper"""
    
    def __init__(self, initial: T):
        self._value = initial
        self._listeners: List[Callable[[T], None]] = []
    
    @property
    def value(self) -> T:
        return self._value
    
    @value.setter
    def value(self, new_value: T) -> None:
        self._value = new_value
        for listener in self._listeners:
            listener(new_value)
    
    def subscribe(self, callback: Callable[[T], None]) -> None:
        self._listeners.append(callback)
    
    def unsubscribe(self, callback: Callable[[T], None]) -> None:
        self._listeners.remove(callback)
```

**Effort:** 0.5 days

---

## Priority Execution Order

### Week 1: Quick Wins
1. `MutableConfig` in config.py (0.5 days) → unblocks test_behavior_config.py
2. `ProceduralModifier` in procedural_system.py (0.5 days) → unblocks test_animation_pipeline.py
3. `RhiDevice::try_new_headless()` (0.5 days) → unblocks 13 lib tests

### Week 1-2: Core Animation
4. IK system types (1-2 days) → unblocks test_ik_system
5. Skinning system types (1-2 days) → unblocks test_skinning_system
6. Animation graph system types (1-2 days) → unblocks test_animation_graph_system

### Week 2-3: Complex Systems
7. Motion matching (2-3 days) → unblocks test_motion_matching_system
8. Crowd/RVO system (2-3 days) → unblocks test_crowd_system

### Week 3-4: Rust Infrastructure
9. CompiledFrameGraph fields (1-2 days) → unblocks ~50 Rust tests
10. gpu_driven exports (1 day) → unblocks ~30 Rust tests
11. resources submodules (2 days) → unblocks ~40 Rust tests
12. Missing Rust modules (5+ days) → unblocks ~100 Rust tests

---

## Current Test Counts

| Category | Passing | Blocked |
|----------|---------|---------|
| Rust integration tests | 3,254 | 296 |
| Rust lib tests | 6,784 | ~13 |
| Python tests | 74,450 | 7 files |
| WGSL shaders | 80 | 0 |

---

## Files Modified This Session

- `crates/renderer-backend/src/lib.rs` — added debug, debug_utils, resource_state exports
- `crates/renderer-backend/src/frame_graph/mod.rs` — added barriers, swap exports
- `crates/renderer-backend/src/gpu_driven/mod.rs` — added 8 submodule exports
- `crates/renderer-backend/src/demoscene/mod.rs` — added 3 submodule exports
- `crates/renderer-backend/Cargo.toml` — added bitflags, arc-swap dependencies
- 8 test files moved from `tests_pending/` to `tests/`
