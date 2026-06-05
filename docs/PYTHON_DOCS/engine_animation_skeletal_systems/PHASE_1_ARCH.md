# PHASE 1 ARCHITECTURE: Core Skeletal Animation

**Generated:** 2026-05-23
**Subsystem:** engine/animation/skeletal

---

## Phase Overview

Phase 1 establishes the foundational data structures and algorithms for skeletal animation. This phase covers the `engine/animation/skeletal` directory and provides the core abstractions that all animation systems build upon.

**Total Lines:** ~7,398
**Classification:** REAL IMPLEMENTATION (100%)

---

## 1. Scope

### 1.1 Included

- Skeleton definition and bone hierarchy
- Pose representation and space conversion
- Animation clip data structures and playback
- Pose blending (override, additive, multiply)
- Linear Blend Skinning (LBS)
- Dual Quaternion Skinning (DQS)
- Animation compression
- Skeleton retargeting
- Root motion extraction

### 1.2 Dependencies

| Dependency | Purpose |
|------------|---------|
| `engine.core.math` | Vec3, Quat, Mat4, Transform |
| `engine.animation.skeletal.constants` | Configuration constants |

---

## 2. Module Architecture

```
skeletal/
├── skeleton.py      [700 lines]  - Bone hierarchy definition
├── pose.py          [761 lines]  - Per-skeleton transforms
├── clip.py          [1102 lines] - Animation keyframe data
├── clip_player.py   [853 lines]  - Playback control
├── blending.py      [923 lines]  - Multi-layer pose blending
├── skinning.py      [797 lines]  - LBS and DQS implementation
├── compression.py   [984 lines]  - Animation compression
├── retargeting.py   [778 lines]  - Cross-skeleton transfer
├── root_motion.py   [~580 lines] - Root motion extraction
├── constants.py     [~50 lines]  - Configuration constants
└── __init__.py      [255 lines]  - Public exports
```

---

## 3. Data Structure Architecture

### 3.1 Skeleton Module (`skeleton.py`)

```
Bone
├── index: int
├── name: str
├── parent_index: int (-1 for root)
├── local_bind_pose: Transform
└── inverse_bind_pose: Mat4

Skeleton
├── _bones: List[Bone]
├── _name_to_index: Dict[str, int]
├── _children: Dict[int, List[int]]
└── methods:
    ├── get_bone(index/name)
    ├── get_chain(start, end)
    ├── traverse_depth_first()
    ├── traverse_breadth_first()
    ├── compute_world_transforms(local_transforms)
    └── compute_skinning_matrices(pose)
```

**Key Algorithm:** Bone Chain Path Finding (lines 276-355)
- Finds path between two bones through common ancestor
- O(depth) complexity via parent traversal

### 3.2 Pose Module (`pose.py`)

```
BoneTransform
├── translation: Vec3
├── rotation: Quat
├── scale: Vec3
└── methods:
    ├── lerp(other, t) -> BoneTransform
    ├── identity() -> BoneTransform
    └── to_matrix() -> Mat4

Pose
├── skeleton: Skeleton
├── local_transforms: List[BoneTransform]
├── model_transforms: Optional[List[BoneTransform]]
└── methods:
    ├── local_to_model()
    ├── get_bone_transform(index)
    ├── set_bone_transform(index, transform)
    ├── blend(other, weight, mode)
    └── clone()

PoseBuffer
├── capacity: int
├── buffer: List[Pose]
└── methods:
    └── store/retrieve poses for temporal blending
```

**Key Algorithm:** Local-to-Model Space Conversion
- Multiply each bone's local transform by parent's model transform
- Must process bones in hierarchy order (parents first)

### 3.3 Clip Module (`clip.py`)

```
Keyframe
├── time: float
├── value: Any (float, Vec3, Quat)
├── in_tangent: Optional[Any]
└── out_tangent: Optional[Any]

AnimationCurve
├── keyframes: List[Keyframe]
├── interpolation: InterpolationType (STEP, LINEAR, CUBIC)
└── methods:
    ├── sample(time) -> value
    ├── binary_search(time) -> keyframe_index
    └── _interpolate_cubic(prev, next, t)

BoneTrack
├── bone_index: int
├── position_curve: AnimationCurve
├── rotation_curve: AnimationCurve
└── scale_curve: AnimationCurve

AnimationClip
├── name: str
├── duration: float
├── tracks: Dict[int, BoneTrack]
├── events: List[AnimationEvent]
├── loop: bool
└── root_motion_data: Optional[RootMotionData]
```

**Key Algorithm:** Hermite Cubic Interpolation (lines 312-355)
```
h00 = 2t^3 - 3t^2 + 1
h10 = t^3 - 2t^2 + t
h01 = -2t^3 + 3t^2
h11 = t^3 - t^2
result = h00*a + h10*dt*out_tan + h01*b + h11*dt*in_tan
```

### 3.4 Blending Module (`blending.py`)

```
BlendMode (enum)
├── OVERRIDE
├── ADDITIVE
└── MULTIPLY

BoneMask
├── weights: Dict[int, float]
└── factory methods:
    ├── full(skeleton)
    ├── upper_body(skeleton)
    ├── lower_body(skeleton)
    └── chain(skeleton, start, end)

LayeredBlender
├── layers: List[BlendLayer]
└── methods:
    ├── add_layer(pose, weight, mode, mask)
    └── evaluate() -> Pose

PoseCache
├── capacity: int
├── cache: Dict[str, Pose]
├── lru_order: List[str]
└── methods:
    ├── get(key)
    ├── put(key, pose)
    └── evict()
```

### 3.5 Skinning Module (`skinning.py`)

```
VertexWeight
├── bone_indices: Tuple[int, int, int, int]
└── weights: Tuple[float, float, float, float]

SkinningData
├── vertex_weights: List[VertexWeight]
└── bind_pose_matrices: List[Mat4]

DualQuaternion
├── real: Quat (rotation)
├── dual: Quat (translation encoded)
└── methods:
    ├── from_transform(rot, trans)
    ├── transform_point(point)
    ├── transform_normal(normal)
    └── normalized()

SkinningMethod (enum)
├── LBS (Linear Blend Skinning)
└── DQS (Dual Quaternion Skinning)
```

**Key Algorithm:** Dual Quaternion Skinning (lines 151-260)
1. Build DQ from rotation + translation: `d = 0.5 * t * r`
2. Handle antipodality: flip sign if `dot(dq[i].real, dq[max].real) < 0`
3. Blend: weighted sum of dual quaternions
4. Normalize: `dq / |dq.real|`
5. Transform: `p' = r * p * r^-1 + 2 * (r.w * d.xyz - d.w * r.xyz + cross(r.xyz, d.xyz))`

### 3.6 Compression Module (`compression.py`)

```
CompressionSettings
├── position_error_threshold: float
├── rotation_error_threshold: float
├── scale_error_threshold: float
└── adaptive_bitrate: bool

CompressedClip
├── header: CompressionHeader
├── quantized_tracks: List[QuantizedTrack]
└── methods:
    └── decompress() -> AnimationClip

QuantizedTrack
├── bone_index: int
├── bit_depth: int
├── range_min: Vec3/Quat
├── range_max: Vec3/Quat
└── data: bytes
```

**Key Algorithms:**
- Ramer-Douglas-Peucker Curve Simplification (lines 624-681)
- ACL-Style Variable Bitrate Selection (lines 684-747)

---

## 4. Integration Points

### 4.1 Inputs

| Source | Data | Usage |
|--------|------|-------|
| Asset system | AnimationClip files | Loaded and decompressed |
| Gameplay | Playback commands | Play, stop, blend requests |

### 4.2 Outputs

| Target | Data | Format |
|--------|------|--------|
| Skinning systems | Pose | BoneTransform array |
| GPU | Skinning matrices | Mat4 array (4x4 floats each) |

---

## 5. Critical Invariants

1. **Quaternion normalization**: All quaternions must be normalized after interpolation
2. **Hierarchy order**: Bones must be processed parent-before-child for model space
3. **Antipodality**: Quaternion blending must check hemisphere and flip if needed
4. **Weight normalization**: Vertex weights must sum to 1.0
5. **4-bone limit**: Each vertex influenced by maximum 4 bones
