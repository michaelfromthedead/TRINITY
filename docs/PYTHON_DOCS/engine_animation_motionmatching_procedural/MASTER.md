# MASTER: Animation Motion Matching and Procedural Systems

**RDC Consolidated Document**
**Generated:** 2026-05-23
**Total Lines Analyzed:** ~11,195

---

## 1. Classification

**Status:** REAL (Both subsystems)
**Confidence:** HIGH

Both `engine/animation/motionmatching` and `engine/animation/procedural` contain production-quality, fully implemented animation systems. These are not stubs - they implement sophisticated algorithms including KD-tree search acceleration, Verlet integration physics, inertialization blending, and procedural gait generation.

---

## 2. Motion Matching System (~6,451 lines)

### 2.1 File Structure

| File | Lines | Key Algorithms | Status |
|------|-------|----------------|--------|
| `__init__.py` | 198 | Complete public API with detailed docstrings | REAL |
| `database.py` | 1,111 | Feature storage, quantization (INT8/INT16/FLOAT16), normalization, serialization with MMDB format | REAL |
| `search.py` | 1,073 | KD-tree, LSH (locality-sensitive hashing), brute-force vectorized search | REAL |
| `context.py` | 987 | Motion matching controller, trajectory prediction, state machine | REAL |
| `features.py` | 963 | Feature extraction (bone positions/velocities, trajectory, foot contacts), normalization | REAL |
| `transition.py` | 961 | Inertialization blending, spring-based decay, SLERP, foot sliding correction | REAL |
| `annotation.py` | 915 | Auto-detection of contacts/locomotion/turns, tag management | REAL |
| `config.py` | 243 | Centralized configuration with dataclasses | REAL |

### 2.2 Database System

**Core Classes:**
- `MotionDatabase`: Stores clips, features, metadata with tag indices
- `DatabaseEntry`: Per-frame entries with features, tags, cost modifiers
- `ClipMetadata`: Clip-level metadata (frame count, rate, looping, root motion)
- `NormalizationStats`: Feature normalization (mean/std/min/max)
- `QuantizationLevel`: Memory-efficient storage (FLOAT16, INT16, INT8)

**Capabilities:**
- Binary serialization with gzip compression (MMDB format)
- Database merging for combining multiple sources
- Tag indices for fast filtering (Set[int] per tag)
- Clip-to-entry range mapping for efficient lookup
- Quantized storage reducing memory 2-4x

### 2.3 Feature Extraction Pipeline

**FeatureExtractor Components:**
- Bone positions relative to root (local space)
- Bone velocities (computed if not available)
- Trajectory positions at configurable future times (default: 0.2s, 0.4s, 0.6s)
- Trajectory facing as 2D direction vectors
- Foot contact states from height/velocity
- Standard bones: hips, spine, chest, limbs, feet, hands

**Normalization:**
- z-score normalization
- min-max normalization
- Configurable feature weights

### 2.4 Search System

**Search Methods:**
| Method | Complexity | Description |
|--------|------------|-------------|
| BRUTE_FORCE | O(n) | Linear scan, guaranteed optimal |
| KD_TREE | O(log n) | Custom implementation with weighted distance |
| LSH | O(1) approximate | Locality-sensitive hashing for large databases |

**KD-Tree Implementation (search.py:269-456):**
- Full recursive implementation with split dimension cycling
- Weighted distance support
- Filter functions for tag-based filtering
- Early termination optimization

**LSH Implementation (search.py:463-588):**
- Multiple hash tables for increased recall
- Random projection vectors
- Bucket-based candidate retrieval

**SearchConfig Parameters:**
- Tags filtering
- Clip exclusion
- Frame distance constraints
- Per-feature cost breakdown

### 2.5 Transition System

**Inertialization Blending (transition.py:412-595):**
- Spring-based offset decay (critical damped)
- Per-bone position and rotation offsets
- Verlet-style update: `new_offset = (offset + velocity * dt) * decay_factor`

**Blend Modes:**
- LINEAR
- INERTIALIZATION
- CROSSFADE

**Additional Features:**
- Foot sliding correction with contact-based foot locking
- Velocity matching at transition points
- quaternion_slerp/multiply/inverse utilities

### 2.6 Runtime Controller

**MotionMatchingController States:**
- IDLE
- MOVING
- TRANSITIONING
- STOPPED

**Components:**
- `MotionContext`: Runtime state (clip, frame, pose, trajectory)
- `DesiredTrajectory`: Future positions/facings from player input
- `IdleDetector`: Hysteresis-based idle state detection
- `TrajectoryBuilder`: Gamepad, keyboard, and velocity input support
- Cost improvement threshold for intelligent transition triggering

### 2.7 Annotation System

**Automatic Detection:**
- `auto_detect_contacts`: Height + velocity based detection
- `auto_detect_locomotion_tags`: Speed-based walk/run/sprint
- `auto_detect_turn_tags`: Angular velocity based turn detection
- Contact smoothing to remove noise

**Data Structures:**
- `AnnotatedClip`: Wrapper adding tags/contacts to clips
- `MotionTag`: Named frame ranges with TagType classification
- `ContactAnnotation`: Per-frame foot contact arrays

### 2.8 Configuration System

9 dataclass configs in `config.py`:
- `FeatureWeightConfig` - Feature importance weights
- `SearchParameterConfig` - KD-tree/LSH parameters
- `TransitionParameterConfig` - Inertialization spring constants
- `ContactDetectionConfig` - Auto-detection thresholds
- `IdleDetectionConfig` - Stationary detection
- `ControllerTimingConfig` - Search intervals
- `DatabaseConfig` - Quantization scales
- `LocomotionSpeedConfig` - Tag speed thresholds
- `TurnDetectionConfig` - Angular velocity thresholds

---

## 3. Procedural Animation System (~4,744 lines)

### 3.1 File Structure

| File | Lines | Key Algorithms | Status |
|------|-------|----------------|--------|
| `__init__.py` | 129 | Comprehensive exports | REAL |
| `spring_bone.py` | 652-653 | Verlet integration, distance constraints, collision (sphere/capsule) | REAL |
| `ragdoll.py` | 808-809 | Physics body/joint creation, kinematic-dynamic transitions, partial ragdoll | REAL |
| `secondary_motion.py` | 719-720 | Delayed motion, oscillation, Perlin noise, impulse response | REAL |
| `locomotion.py` | 675-676 | Procedural gait generation, foot trajectory arcs, body dynamics | REAL |
| `lookat.py` | 646-647 | Head/neck/eye IK, saccade generation, angle limits | REAL |
| `twist.py` | 496-497 | Twist extraction/distribution (swing-twist decomposition) | REAL |
| `breathing.py` | 476-477 | Breathing cycle phases, exertion levels, spine/chest animation | REAL |
| `config.py` | 273 | Centralized constants, physics presets | REAL |

### 3.2 Spring Bone System

**Verlet Integration (spring_bone.py:298-384):**
```
x_new = 2*x - x_old + a*dt^2
```
- Spring force: `F = -k*x - c*v`
- Timestep clamping for numerical stability
- Collision detection/response

**Distance Constraints (spring_bone.py:472-515):**
- Position-based constraint solver
- Configurable iteration count
- Soft constraint with 0.5 lerp factor

**Collision Primitives:**
- Sphere collision
- Capsule collision

**Wind Forces:**
- Configurable strength and turbulence

### 3.3 Ragdoll System

**Components:**
- `RagdollBody`: Physics body representation
- `RagdollJoint`: Joint with limits and motors
- `JointLimits`: Twist/swing angle constraints
- `JointMotor`: Motorized joints for active ragdoll

**States:**
- DYNAMIC
- KINEMATIC
- BLENDING

**Capabilities:**
- Full ragdoll
- Partial ragdoll with `active_bodies` set
- Kinematic-dynamic transitions
- Physics world protocol for external engine integration
- `blend_weight` with slerp/lerp interpolation

### 3.4 Procedural Locomotion

**Gait Generation (locomotion.py:129-224):**
- Foot trajectory: stance phase (ground slide) + swing phase (parabolic arc)
- Body dynamics: vertical bob, lateral sway, hip rotation, spine twist
- Speed-adaptive cycle duration

**Supported Gaits:**
- Walk
- Run
- Trot (quadruped)
- Gallop (quadruped)

**Configurations:**
- Biped support
- Quadruped support

### 3.5 Look-At Controller

**Features:**
- Head rotation tracking
- Neck rotation tracking
- Eye rotation tracking
- Angle limits per joint

**Saccade Generation (lookat.py:276-347):**
- Random interval between saccades (0.1-3.0s)
- High-speed eye movement (500 deg/s)
- Smooth transition to target offset

**Interest Point System:**
- Priority-based target selection

### 3.6 Breathing Controller

**Exertion Levels:** 5 levels with different breath rates

**Animation Targets:**
- Spine bones
- Chest bones
- Configurable phase offsets

**Cycle Phases:**
- Inhale
- Exhale
- Pause

### 3.7 Twist Bone Distribution

**Swing-Twist Decomposition (twist.py:186-207):**
- Projects rotation axis onto twist axis
- Extracts twist angle via dot product
- Reconstructs twist quaternion

**Distribution Modes:**
- Single twist bone
- Multi-bone distribution with weights

### 3.8 Secondary Motion System

**Motion Types:**
| Type | Description |
|------|-------------|
| DelayedMotion | Time-buffered motion lag |
| OscillatingMotion | Sine-wave oscillation |
| NoiseMotion | Perlin noise FBM |
| ImpulseResponse | Damped spring response |
| MotionComposer | Stackable effects |

**Perlin Noise Implementation (secondary_motion.py:140-208):**
- Permutation table (256 entries, shuffled)
- Fade function: `t^3 * (t * (t * 6 - 15) + 10)`
- Fractal Brownian Motion (FBM) with octaves

**Impulse Response (secondary_motion.py:537-658):**
- Acceleration detection via finite differences
- Damped spring response to sudden movements
- Threshold-based triggering

### 3.9 Configuration System

10 frozen dataclasses in `config.py`:
- `SpringPhysicsConfig` - Stiffness, damping, mass limits
- `WindForceConfig` - Wind strength, turbulence
- `LookAtConfig` - Angle limits, speeds
- `SaccadeConfig` - Eye movement intervals
- `TwistConfig` - Twist axes
- `RagdollConfig` - Body masses, dimensions
- `LocomotionConfig` - Gait parameters
- `BreathingConfig` - Breath rates per exertion level
- `SecondaryMotionConfig` - Effect parameters

---

## 4. Cross-Module Integration

### 4.1 Shared Patterns

1. **Protocol-based interfaces**: Both modules use `typing.Protocol` for `Pose`, `Skeleton` rather than concrete classes
2. **Centralized configuration**: Both have dedicated `config.py` with frozen dataclasses
3. **Quaternion utilities**: Both implement SLERP, multiply, axis-angle conversion
4. **Vector utilities**: Both implement add, sub, scale, length, normalize

### 4.2 Integration Points

- `transition.py` imports from `database.py` (DatabaseEntry)
- `context.py` integrates `database`, `features`, `search`, `transition`
- Procedural modules are standalone with protocol-based pose interfaces

### 4.3 Dependencies

**External:**
- `numpy` - Vector/matrix operations

**Internal:**
- `engine.animation.motionmatching.config`
- `engine.animation.procedural.config`

---

## 5. Quality Indicators

### 5.1 Evidence of Real Implementation

1. **Algorithm depth** - KD-tree, LSH, Verlet integration, inertialization are non-trivial
2. **Edge case handling** - Zero-length vector checks, timestep clamping, angle clamping
3. **Numerical stability** - Epsilon values, float64 for cost computation, clamping
4. **Documentation** - Comprehensive docstrings with formulas and usage examples
5. **Configuration** - Tunable magic numbers in dedicated config files
6. **Validation** - `__post_init__` checks on dataclasses
7. **Serialization** - Binary MMDB format with gzip compression
8. **Testing-friendly** - Protocol-based interfaces enable mock injection

### 5.2 No Stub Indicators

- No `raise NotImplementedError`
- No `pass` in method bodies
- No TODO/FIXME indicating missing implementation
- All declared methods have full implementations

---

## 6. Key Algorithms Summary

### Motion Matching
| Algorithm | Location | Description |
|-----------|----------|-------------|
| KD-Tree Nearest Neighbor | search.py:269-456 | Recursive with weighted distance |
| Locality-Sensitive Hashing | search.py:463-588 | Multiple tables, random projection |
| Inertialization Blending | transition.py:412-595 | Spring-based offset decay |
| Feature Extraction | features.py:259-558 | Bone position/velocity, trajectory |
| Foot Sliding Correction | transition.py:812-910 | Contact-based foot locking |

### Procedural Animation
| Algorithm | Location | Description |
|-----------|----------|-------------|
| Verlet Integration | spring_bone.py:298-384 | x_new = 2x - x_old + a*dt^2 |
| Distance Constraints | spring_bone.py:472-515 | Position-based solver |
| Swing-Twist Decomposition | twist.py:186-207 | Axis projection |
| Procedural Gait | locomotion.py:129-224 | Stance/swing phases |
| Saccade Generation | lookat.py:276-347 | Random interval eye movement |
| Perlin Noise (1D FBM) | secondary_motion.py:140-208 | Fade + octave summation |
| Impulse Response | secondary_motion.py:537-658 | Damped spring on acceleration |

---

## 7. Industry References

Both modules represent production-quality animation systems implementing industry-standard techniques:

- **Motion Matching**: Per Ubisoft's "Motion Matching and The Road to Next-Gen Animation" (GDC 2016)
- **Inertialization**: Per GDC 2018 technique for artifact-free transitions
- **Procedural Animation**: Per GDC talks on secondary motion and physics-based character animation
