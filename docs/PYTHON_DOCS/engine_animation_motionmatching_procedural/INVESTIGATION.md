# Archaeological Investigation: engine/animation/motionmatching + engine/animation/procedural

**Date**: 2026-05-22
**Investigator**: Research Agent (Opus 4.5)
**Total Lines Analyzed**: ~11,195 lines

---

## Executive Summary

**Classification**: REAL (Both subdirectories)

Both `engine/animation/motionmatching` and `engine/animation/procedural` contain production-quality, fully implemented animation systems. These are not stubs - they implement sophisticated algorithms including KD-tree search acceleration, Verlet integration physics, inertialization blending, and procedural gait generation. The code demonstrates deep domain expertise in game animation techniques.

---

## engine/animation/motionmatching (~6,451 lines)

### Classification: REAL

### Evidence of Real Implementation

| File | Lines | Key Algorithms | Status |
|------|-------|----------------|--------|
| database.py | 1,111 | Feature storage, quantization (INT8/INT16/FLOAT16), normalization, serialization with MMDB format | REAL |
| search.py | 1,073 | KD-tree, LSH (locality-sensitive hashing), brute-force vectorized search | REAL |
| context.py | 987 | Motion matching controller, trajectory prediction, state machine | REAL |
| features.py | 963 | Feature extraction (bone positions/velocities, trajectory, foot contacts), normalization | REAL |
| transition.py | 961 | Inertialization blending, spring-based decay, SLERP, foot sliding correction | REAL |
| annotation.py | 915 | Auto-detection of contacts/locomotion/turns, tag management | REAL |

### Key Algorithms Found

1. **KD-Tree Nearest Neighbor Search** (search.py:269-456)
   - Full recursive implementation with split dimension cycling
   - Weighted distance support
   - Filter functions for tag-based filtering
   - Early termination optimization

2. **Locality-Sensitive Hashing** (search.py:463-588)
   - Multiple hash tables for increased recall
   - Random projection vectors
   - Bucket-based candidate retrieval

3. **Inertialization Blending** (transition.py:412-595)
   - Spring-based offset decay (critical damped)
   - Per-bone position and rotation offsets
   - Verlet-style update: `new_offset = (offset + velocity * dt) * decay_factor`

4. **Feature Extraction Pipeline** (features.py:259-558)
   - Bone position/velocity in local space
   - Trajectory prediction with quaternion-to-facing conversion
   - Foot contact detection via velocity threshold

5. **Foot Sliding Correction** (transition.py:812-910)
   - Contact-based foot locking
   - Position constraint when foot is grounded

### Configuration System (config.py)

Centralized configuration with 9 dataclass configs:
- `FeatureWeightConfig` - Feature importance weights
- `SearchParameterConfig` - KD-tree/LSH parameters
- `TransitionParameterConfig` - Inertialization spring constants
- `ContactDetectionConfig` - Auto-detection thresholds
- `IdleDetectionConfig` - Stationary detection
- `ControllerTimingConfig` - Search intervals
- `DatabaseConfig` - Quantization scales
- `LocomotionSpeedConfig` - Tag speed thresholds
- `TurnDetectionConfig` - Angular velocity thresholds

### Dependencies

- `numpy` - Vector/matrix operations
- Internal imports: `engine.animation.motionmatching.config`

---

## engine/animation/procedural (~4,744 lines)

### Classification: REAL

### Evidence of Real Implementation

| File | Lines | Key Algorithms | Status |
|------|-------|----------------|--------|
| ragdoll.py | 808 | Physics body/joint creation, kinematic-dynamic transitions, partial ragdoll | REAL |
| secondary_motion.py | 719 | Delayed motion, oscillation, Perlin noise, impulse response | REAL |
| locomotion.py | 675 | Procedural gait generation, foot trajectory arcs, body dynamics | REAL |
| spring_bone.py | 652 | Verlet integration, distance constraints, collision (sphere/capsule) | REAL |
| lookat.py | 646 | Head/neck/eye IK, saccade generation, angle limits | REAL |
| twist.py | 496 | Twist extraction/distribution (swing-twist decomposition) | REAL |
| breathing.py | 476 | Breathing cycle phases, exertion levels, spine/chest animation | REAL |

### Key Algorithms Found

1. **Verlet Integration** (spring_bone.py:298-384)
   ```
   x_new = 2*x - x_old + a*dt^2
   ```
   - Spring force: `F = -k*x - c*v`
   - Timestep clamping for numerical stability
   - Collision detection/response

2. **Distance Constraints** (spring_bone.py:472-515)
   - Position-based constraint solver
   - Configurable iteration count
   - Soft constraint with 0.5 lerp factor

3. **Swing-Twist Decomposition** (twist.py:186-207)
   - Projects rotation axis onto twist axis
   - Extracts twist angle via dot product
   - Reconstructs twist quaternion

4. **Procedural Gait Generation** (locomotion.py:129-224)
   - Foot trajectory: stance phase (ground slide) + swing phase (parabolic arc)
   - Body dynamics: vertical bob, lateral sway, hip rotation, spine twist
   - Speed-adaptive cycle duration

5. **Saccade Generation** (lookat.py:276-347)
   - Random interval between saccades (0.1-3.0s)
   - High-speed eye movement (500 deg/s)
   - Smooth transition to target offset

6. **Perlin Noise (1D)** (secondary_motion.py:140-208)
   - Permutation table (256 entries, shuffled)
   - Fade function: `t^3 * (t * (t * 6 - 15) + 10)`
   - Fractal Brownian Motion (FBM) with octaves

7. **Impulse Response** (secondary_motion.py:537-658)
   - Acceleration detection via finite differences
   - Damped spring response to sudden movements
   - Threshold-based triggering

### Configuration System (config.py)

Centralized configuration with 10 frozen dataclasses:
- `SpringPhysicsConfig` - Stiffness, damping, mass limits
- `WindForceConfig` - Wind strength, turbulence
- `LookAtConfig` - Angle limits, speeds
- `SaccadeConfig` - Eye movement intervals
- `TwistConfig` - Twist axes
- `RagdollConfig` - Body masses, dimensions
- `LocomotionConfig` - Gait parameters
- `BreathingConfig` - Breath rates per exertion level
- `SecondaryMotionConfig` - Effect parameters

### Dependencies

- Internal imports: `engine.animation.procedural.config`
- Type hints use `Protocol` for duck-typed interfaces

---

## Cross-Module Analysis

### Shared Patterns

1. **Protocol-based interfaces** - Both modules use `typing.Protocol` for `Pose`, `Skeleton` rather than concrete classes
2. **Centralized configuration** - Both have dedicated `config.py` with frozen dataclasses
3. **Quaternion utilities** - Both implement SLERP, multiply, axis-angle conversion
4. **Vector utilities** - Both implement add, sub, scale, length, normalize

### Integration Points

- `transition.py` imports from `database.py` (DatabaseEntry)
- `context.py` integrates `database`, `features`, `search`, `transition`
- Procedural modules are standalone with protocol-based pose interfaces

---

## Quality Indicators

### Positive Indicators (REAL)

1. **Algorithm depth** - KD-tree, LSH, Verlet integration, inertialization are non-trivial
2. **Edge case handling** - Zero-length vector checks, timestep clamping, angle clamping
3. **Numerical stability** - Epsilon values, float64 for cost computation, clamping
4. **Documentation** - Comprehensive docstrings with formulas and usage examples
5. **Configuration** - Tunable magic numbers in dedicated config files
6. **Validation** - `__post_init__` checks on dataclasses
7. **Serialization** - Binary MMDB format with gzip compression
8. **Testing-friendly** - Protocol-based interfaces enable mock injection

### No Stub Indicators

- No `raise NotImplementedError`
- No `pass` in method bodies
- No TODO/FIXME indicating missing implementation
- All declared methods have full implementations

---

## Summary

| Subdirectory | Lines | Classification | Confidence |
|--------------|-------|----------------|------------|
| motionmatching | 6,451 | REAL | HIGH |
| procedural | 4,744 | REAL | HIGH |

Both modules represent production-quality animation systems implementing industry-standard techniques (motion matching per Ubisoft's "Motion Matching and The Road to Next-Gen Animation", procedural animation per GDC talks on secondary motion). The implementation depth, numerical stability considerations, and configuration architecture indicate these are working systems, not scaffolding.
