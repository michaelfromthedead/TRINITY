# GAPSET_14_ANIMATION -- Project Overview

## Animation System Architecture

The Trinity animation system is a comprehensive character animation pipeline implemented primarily in Python with planned Rust/WGSL acceleration layers. It spans ~39,827 lines across 60 files in 9 submodules.

---

## Layer Architecture

### Layer 0: Math Foundation (omega/src/)
- Vec3, Quat, Mat4, Transform, RigidTransform
- Omega math library used by all animation modules

### Layer 1: Skeletal Animation (engine/animation/skeletal/)
- **Skeleton**: Bone hierarchy, world transforms, inverse bind pose, validation
- **Pose**: SoA transforms, lerp/slerp, additive blending, buffering
- **Clip**: Keyframe curves, tracks, events, interpolation (step/linear/cubic)
- **ClipPlayer**: Playback control, looping, speed, queue, crossfade
- **Blending**: Override/additive/multiply blends, bone masks, layered blending
- **RootMotion**: Extraction (5 modes), accumulation, application, blending
- **Retargeting**: Bone mapping, chain normalization, foot contact preservation
- **Compression**: 6 methods (keyframe reduction, quantization, uniform sampling, variable bitrate)
- **Skinning**: LBS, DQS, hybrid, GPU data preparation, caching

### Layer 2: Animation Graph (engine/animation/graph/)
- **Graph DAG**: Nodes, connections, parameters, dirty-flag evaluation
- **StateMachine**: States, transitions, conditions, decoration API
- **BlendTree**: 1D/2D (Delaunay)/Direct parametric blending
- **BlendNode**: 8 node types (clip, blend, additive, layer, mirror, timescale, cache, select)
- **LayerStack**: Layer ordering, bone masks, blend modes
- **SyncGroup**: Marker-based animation synchronization, leader-follower

### Layer 3: IK Solvers (engine/animation/ik/)
- **TwoBoneIK**: Analytical O(1), swivel control, constraints
- **FABRIK**: Forward/backward reaching, multi-chain, joint constraints
- **CCD**: Iterative cyclic descent, rotation limits, damping
- **JacobianIK**: DLS/SVD pseudoinverse, multi-effector, task prioritization
- **FullBodyIK**: Multi-effector with balance (CoM), posture preservation, priority layering
- **IKGoal**: 7 goal types, decorator API, blending, priority
- **FootPlacement**: Terrain adaptation, ankle roll, toe alignment, stance detection

### Layer 4: Motion Matching (engine/animation/motionmatching/)
- **Database**: Built from clips, ANN index, quantization, serialization
- **Features**: Pose, trajectory, foot contact extraction with normalization
- **Search**: KD-tree, LSH, brute force with configurable cost function
- **Transition**: Inertialization (derivative-continuous blend), foot sliding correction
- **Context**: Controller, trajectory builder, idle detection, styling parameters

### Layer 5: Facial Animation (engine/animation/facial/)
- **BlendShapes**: ARKit-compatible (52+ shapes), correctives, remapping
- **FACS**: 52 Action Units, expressions, asymmetry
- **LipSync**: Phoneme/viseme mapping, coarticulation, event system
- **EyeAnimation**: Saccades, drift, tremor, blinking, pupil dilation
- **FaceRig**: Priority layers, emotion states, animation blending
- **FaceCapture**: Keyframe-based playback, retargeting

### Layer 6: Procedural Animation (engine/animation/procedural/)
- **SpringBone**: Damped spring physics, collision, wind
- **LookAt**: Head/eye tracking with soft cone limits, velocity prediction
- **Twist**: Twist distribution across bone chains
- **Ragdoll**: Blend-in/out, partial activation, active ragdoll
- **Locomotion**: Procedural gait generation
- **Breathing**: Natural breath cycles with exertion levels
- **SecondaryMotion**: Delay, oscillation, noise, impulse response effects

### Layer 7: Crowd Systems (engine/animation/crowds/)
- **AnimationTexture**: Bone transform baking to RGBA textures
- **CrowdRenderer**: GPU instancing, per-instance data, frustum culling
- **CrowdLOD**: 3 LOD levels, distance-based selection, transition smoothing
- **CrowdBehavior**: Agent steering (idle, walking, fleeing, formation), simulator

### Layer 8: ECS Systems (engine/animation/systems/)
- **AnimationGraphSystem**: State machine evaluation, pose sampling
- **IKSystem**: Solver dispatch, priority ordering
- **ProceduralSystem**: Spring/lookat/sway/breathing orchestration
- **SkinningSystem**: LBS/DQS/GPU dispatch
- **MotionMatchingSystem**: Trajectory computation, search, transition
- **FacialSystem**: Blend shape/FACS/lip sync/eye animation
- **CrowdSystem**: Agent steering, texture baking, LOD, culling

## Evaluation Pipeline
1. Simulation: State machine evaluates transitions, outputs blend weights
2. Sampling: Clips sampled at current time for active states
3. Blending: Poses blended according to state machine weights
4. Graph: Blend trees, layers, sync groups applied
5. IK: Goals resolved in priority order
6. Procedural: Spring bones, look-at, ragdoll applied
7. Facial: Blend shapes, FACS, lip sync, eye tracking
8. Skinning: LBS/DQS computed, GPU data prepared
9. Output: Transforms consumed by renderer

## Planned Missing Layers
- **Rust Backend**: crates/animation/ -- SIMD-accelerated math and parallel processing
- **WGSL Shaders**: shaders/skinning/, shaders/crowd/ -- GPU compute skinning
- **Cinematics**: engine/animation/cinematics/ -- cutscene playback and camera tracks
- **Foundation Integration**: AssetMeta, EventMeta, ResourceMeta, Tracker, Session decorators
- **Tests**: Unit and integration tests for all layers
