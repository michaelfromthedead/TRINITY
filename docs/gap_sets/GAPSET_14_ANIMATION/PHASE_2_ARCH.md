# Phase 2: Animation Playback & Blending -- Architecture

## Status: 5 [x] 0 [~] 2 [-]

## Module: `engine/animation/skeletal/`

### Files
| File | Lines | Purpose |
|------|-------|---------|
| clip_player.py | 853 | ClipPlayer, ClipQueue, CrossfadePlayer |
| blending.py | 923 | BlendMode, BoneMask, LayeredBlender |
| root_motion.py | 621 | Root motion extraction and application |
| retargeting.py | 779 | Skeleton retargeting pipeline |
| compression.py | 984 | Animation data compression |

### Architecture

**ClipPlayer** (`clip_player.py`):
- `ClipPlayer`: play/pause/stop/resume, speed (-1 to 1), looping (once/loop/ping-pong), seek
- Event firing (footstep, attack windows) based on AnimationEvent tracks
- Curve sampling for float animation channels
- `ClipQueue`: sequential clip playback with automatic transitions
- `CrossfadePlayer`: synchronized fade between clips with configurable curves

**Blending** (`blending.py`):
- `BlendMode`: OVERRIDE, ADDITIVE, MULTIPLY
- `BoneMask`: per-bone blend weights with preset creation (upper_body, lower_body, etc.)
- `blend_poses()`: linear lerp between two poses
- `_blend_override`: full override with bone mask support
- `_blend_additive`: base + additive * weight
- `_blend_multiply`: multiply bone transforms
- `blend_multiple_poses()`: N-way weighted blend
- `compute_additive_pose()`: reference - base
- `apply_additive_pose()`: base + additive
- `LayeredBlender`: stack of layers with per-layer bone masks and weights
- `PoseCache`: LRU cache with configurable capacity

**RootMotion** (`root_motion.py`):
- `RootMotionMode`: IN_PLACE, EXTRACT_XZ, EXTRACT_XYZ, EXTRACT_ROTATION, EXTRACT_ALL
- `RootMotionData`: per-frame deltas, total delta, accumulate/seek
- `extract_root_motion()`: delta computation between frames
- `apply_root_motion()`: transform-space delta application
- `RootMotionAccumulator`: continuous accumulation across frames/loops
- `RootMotionConfig`: scale, rotation_scale, ground clamping
- `RootMotionBlender`: weighted blend of multiple accumulators

**Retargeting** (`retargeting.py`):
- `BoneMappingStrategy`: NAME, POSITION, HEURISTIC
- `BoneMapping`: name/code/key mapping with chain info
- `RetargetMap`: source-to-target bone mapping with scale factors
- `SkeletonInfo`: bone lengths, chain data for retargeting
- `RetargetPipeline`: full pipeline with source/target skeletons + config
- `retarget_pose()`: scale + rotate each bone from source to target space
- `preserve_foot_contact()`: IK-based foot position maintenance

**Compression** (`compression.py`):
- `CompressionMethod`: KEY_REDUCTION, QUANTIZATION, UNIFORM_SAMPLING, VARIABLE_BITRATE, ACL, CUSTOM
- `CompressionSettings`: per-track error thresholds, bit depths
- `CompressedTrack`/`CompressedClip`: compressed storage formats
- `compress_clip()`: applies selected compression method
- `decompress_clip()`: runtime decompression
- `compute_compression_error()`: error metrics (max, mean, RMS)

### Missing
- T-AN-2.6: Rust backend (crates/animation/)
- T-AN-2.7: Tests

### Key Design Decisions
- Root motion blends use quaternion antipodality handling for correct blending
- Compression supports multiple quality presets (LOW=8bit, MEDIUM=16bit, HIGH=32bit)
- Retargeting uses chain length normalization for proportion correction
- Pose blending distinguishes local-space vs. model-space for correctness
