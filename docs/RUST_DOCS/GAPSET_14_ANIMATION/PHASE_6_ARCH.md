# Phase 6: Motion Matching -- Architecture

## Status: 5 [x] 0 [~] 1 [-]

## Module: `engine/animation/motionmatching/`

### Files
| File | Lines | Purpose |
|------|-------|---------|
| database.py | 1111 | Motion database, build pipeline, quantization |
| features.py | 963 | Feature extraction and normalization |
| search.py | 1073 | Search algorithms (KD-tree, LSH, brute force) |
| transition.py | 961 | Inertialization-based blending |
| context.py | 987 | Runtime controller and state management |
| annotation.py | 915 | Automated clip tagging and contact detection |
| config.py | 244 | All configuration parameters |
| __init__.py | 198 | Public API |

### Architecture

**Database** (`database.py`):
- `MotionDatabase`: frame entries, normalization stats, ANN index
- `DatabaseEntry`: pose features, trajectory features, foot contacts, tags, clip ref
- `ClipMetadata`: clip name, frame range, annotation tags
- `NormalizationStats`: mean/std for feature normalization
- `QuantizationLevel`: NONE, INT16, INT8 storage modes
- `build_database()`: extract features from clip collection
- `merge_databases()`: combine multiple databases
- `motion_matching()`: one-shot match function

**Features** (`features.py`):
- `FeatureSet`: pose (bone positions/velocities relative to root, normalized by height)
- `FeatureExtractor`: compute features from pose + trajectory
- `FeatureNormalizer`: zero-mean unit-variance normalization
- `FeatureType`: enumeration of all feature dimensions
- `FeatureWeights`: per-type and per-bone weight configuration
- `TrajectoryPoint`: predicted future root position at T+0.2/0.5/1.0s
- `FootContact`: contact state, position, velocity detection

**Search** (`search.py`):
- `SearchMethod`: BRUTE_FORCE, KD_TREE, LSH
- `MotionSearch`: weighted cost = pose_cost * w_pose + trajectory * w_traj + velocity * w_vel + transition_continuity
- `compute_cost()`: per-frame cost with pruning
- `compute_cost_vectorized()`: numpy-accelerated batch cost
- `KDTree`: scipy-based ANN with leaf_size config
- `LSHIndex`: locality-sensitive hashing for approximate search
- `SearchConfig`: cost weights, thresholds, stickiness, min_clip_time

**Transition** (`transition.py`):
- `InertializationBlender`: derivative-continuous spring-based blend
- `InertializationOffset`: position + velocity matching state
- `BlendMode`: CROSSFADE, INERTIALIZATION, NONE
- `FootSlidingCorrector`: contact-aware foot position maintenance
- Quaternion operations: slerp, multiply, inverse helpers

**Context** (`context.py`):
- `MotionMatchingController`: full runtime controller
- `MotionContext`: parameter set including stylization (aggressive, cautious, injured)
- `ControllerConfig`: search interval, min clip time, budget
- `DesiredTrajectory`: input direction prediction
- `IdleDetector`: velocity threshold-based idle detection
- `TrajectoryBuilder`: future trajectory construction from input

**Annotation** (`annotation.py`):
- `AnnotatedClip`: clips with auto-detected tags
- `MotionTag`/`TagType`: LOCOMOTION, TURN, CONTACT, ACTION, TERRAIN
- `auto_detect_contacts()`: velocity + height threshold detection
- `auto_detect_locomotion_tags()`: speed-based walk/run/sprint
- `auto_detect_turn_tags()`: angular velocity threshold
- `merge_overlapping_tags()`: tag combination/merging

### Missing
- T-AN-6.6: Tests

### Key Design Decisions
- Three search methods with different performance profiles (O(n), O(log n), O(1))
- Feature extraction follows standard MM recipe (pose + trajectory + contacts)
- Inertialization uses critically damped spring model for derivative continuity
- Foot sliding corrector prevents penetration during transitions
- Database supports quantization for memory-efficient storage (INT16/INT8)
- Trajectory features at 0.2/0.5/1.0s follow GDC 2016 recommendations
