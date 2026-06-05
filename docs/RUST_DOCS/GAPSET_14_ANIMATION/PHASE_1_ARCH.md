# Phase 1: Skeleton Data Structures & Foundation -- Architecture

## Status: 3 [x] 3 [~] 2 [-]

## Module: `engine/animation/skeletal/`

### Files
| File | Lines | Purpose |
|------|-------|---------|
| skeleton.py | 701 | Bone hierarchy, Skeleton, humanoid factory |
| pose.py | 762 | Pose, BoneTransform, PoseBuffer |
| clip.py | 1102 | AnimationClip, Keyframe, AnimationCurve, BoneTrack |
| constants.py | 49 | Shared constants for skeletal subsystem |
| __init__.py | 254 | Public API exports |

### Architecture

**Skeleton** (`skeleton.py`):
- `Bone` dataclass: name, parent_index, local_transform, rest_transform
- `Skeleton`: bone array, name-to-index dict, world transform propagation via pre-order traversal
- `create_humanoid_skeleton()`: standardized 28-bone humanoid skeleton
- `@animation_data` decorator: marks classes with animation metadata
- Validation methods: circular reference detection, root index integrity

**Pose** (`pose.py`):
- `BoneTransform`: Vec3 position, Quat rotation, Vec3 scale
- `Pose`: SoA storage (lists of Vec3/Quat/Vec3 per bone), PoseSpace enum (local/model)
- `PoseBuffer`: ring buffer of poses (capacity-based, push/sample operations)
- `lerp_poses()`: per-bone lerp/slerp blending
- `additive_blend()`: base + delta * weight
- `compute_additive_pose()`: ref - base pose difference
- `blend_multiple_poses()`: weighted sum of N poses

**AnimationClip** (`clip.py`):
- `Keyframe`: time, value, interpolation, in/out tangents
- `AnimationCurve`: keyframes per channel (step/linear/cubic Hermite)
- `BoneTrack`: position/rotation/scale curves per bone
- `AnimationClip`: duration, frame_rate, bone_tracks, event_tracks, curve_tracks
- `AnimationEvent`: time, name, parameters
- `create_simple_clip()`: helper for quick clip creation

### Missing
- T-AN-1.4: Rust backend (crates/animation/ does not exist)
- T-AN-1.6: Foundation EventMeta integration
- T-AN-1.8: Tests

### Partial
- T-AN-1.1: Missing Foundation AssetMeta serialization (.skel extension)
- T-AN-1.5: Custom @animation_data exists, Foundation @asset not wired
- T-AN-1.7: Config values exist, Foundation @resource not wired

### Key Design Decisions
- SoA storage for poses enables SIMD-friendly layout for future Rust port
- Bone name-to-index mapping is O(1) dict lookup
- AnimationCurve supports Hermite interpolation for smooth motion
- Ring buffer PoseBuffer avoids allocation during gameplay
