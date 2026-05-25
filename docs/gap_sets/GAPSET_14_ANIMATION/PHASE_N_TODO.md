# GAPSET_14_ANIMATION -- Phase Tasks

> **TASK_ID Format**: T-AN-{PHASE}.{N}
> **Total Tasks**: 68
> **Estimated Effort**: 12-16 weeks (single developer)
> **Implementation Status**: 0% complete, 68 tasks pending

---

## Phase 1: Skeleton Data Structures & Foundation (8 tasks)

**Dependencies**: S15 Core Systems (math.rs, component_store.rs), S16 Asset Pipeline
**Files**: 3 Python + 1 Rust

### T-AN-1.1 -- Implement Skeleton asset
- **Description**: Create `engine/animation/skeletal/skeleton.py` with Skeleton class
- **Acceptance Criteria**:
  - Bone structure: name, parent_index, local_transform, inverse_bind_transform
  - Bone array with fast traversal (pre-computed world transforms)
  - Bone name-to-index mapping
  - Serialization via AssetMeta with `.skel` extension
  - Validation: no circular parent references, root at index 0
- **Dependencies**: None (asset pipeline assumed)
- **Effort**: 2 days

### T-AN-1.2 -- Implement Pose representation
- **Description**: Create `engine/animation/skeletal/pose.py` with Pose classes
- **Acceptance Criteria**:
  - BindPose, ReferencePose, CurrentPose, AdditivePose types
  - SoA storage: Vec3 positions, Quat rotations, Vec3 scales per bone
  - Pose-to-pose weight blend (lerp for pos/scale, slerp for rot)
  - TransientDescriptor: bone transforms NOT saved to session
- **Dependencies**: T-AN-1.1
- **Effort**: 1 day

### T-AN-1.3 -- Implement AnimationClip asset
- **Description**: Create `engine/animation/skeletal/clip.py` with AnimationClip class
- **Acceptance Criteria**:
  - Per-bone transform tracks: arrays of (time, value, interpolation_mode, tangents)
  - Channels: position, rotation, scale per bone
  - Keyframe interpolation: step, linear, cubic (Hermite/Catmull-Rom)
  - Duration, frame_rate, looping mode
  - Event tracks: named notifies at specific times
  - Curve tracks: continuous float channels
  - Serialization via AssetMeta with `.anim` extension
- **Dependencies**: T-AN-1.1
- **Effort**: 3 days

### T-AN-1.4 -- Create Rust skeleton backend
- **Description**: Implement `crates/animation/skeleton.rs` and `crates/animation/pose.rs`
- **Acceptance Criteria**:
  - Bone struct with parent index (i32), local/model/world transforms
  - Pose as flat arrays for SIMD access
  - ComputeBoneChain: local-to-world transform propagation
  - Inverse bind matrix computation
  - SIMD pose blending (lerp/slerp on SoA data)
- **Dependencies**: math.rs (Vec3, Quat, Mat4)
- **Effort**: 3 days

### T-AN-1.5 -- Wire Foundation: AssetMeta for animation assets
- **Description**: Register Skeleton, AnimationClip, MotionDatabase assets with Foundation Registry
- **Acceptance Criteria**:
  - `@asset(extensions=[".skel"])` on Skeleton class
  - `@asset(extensions=[".anim", ".fbx", ".glb"])` on AnimationClip class
  - `@asset(extensions=[".mmdb"])` on MotionDatabase class
  - Registry query: `registry.subclasses(Asset)` filtered by animation extensions
- **Dependencies**: T-AN-1.1, T-AN-1.3
- **Effort**: 1 day

### T-AN-1.6 -- Wire Foundation: EventMeta for animation events
- **Description**: Create AnimationNotify, StateTransition, RagdollActivated events via EventMeta
- **Acceptance Criteria**:
  - `@event` AnimationNotify: entity_id, clip_name, notify_name, time
  - `@event` StateTransition: entity_id, from_state, to_state
  - `@event` RagdollActivated: entity_id, blend_time, impact_force
  - Events fire into Foundation EventLog with causal chains
- **Dependencies**: T-AN-1.3
- **Effort**: 1 day

### T-AN-1.7 -- Wire Foundation: AnimationConfig resource
- **Description**: Create AnimationConfig global resource via ResourceMeta
- **Acceptance Criteria**:
  - `@resource` AnimationConfig: global_speed, max_active_ik_chains, motion_matching_budget_ms, lod_distances
  - Config accessible from all animation systems
  - Hot-reloadable via Foundation Inspector
- **Dependencies**: None
- **Effort**: 1 day

### T-AN-1.8 -- Write skeleton unit tests
- **Description**: Test skeleton hierarchy, pose blending, clip sampling
- **Acceptance Criteria**:
  - Skeleton parent-child chain evaluation
  - Bind/inverse bind matrix correctness
  - Pose lerp/slerp precision
  - Clip keyframe interpolation accuracy
  - Event notification timing
- **Dependencies**: T-AN-1.1, T-AN-1.2, T-AN-1.3, T-AN-1.4
- **Effort**: 2 days

---

## Phase 2: Animation Playback & Blending (7 tasks)

**Dependencies**: Phase 1 complete
**Files**: 5 Python + 2 Rust

### T-AN-2.1 -- Implement clip player
- **Description**: Create `engine/animation/skeletal/clip_player.py` with ClipPlayer class
- **Acceptance Criteria**:
  - Play/pause/stop/resume controls
  - Play rate (forward, reverse, time scaling)
  - Loop modes: once, loop, ping-pong
  - Seek to arbitrary time
  - Event firing during playback (footstep, attack window)
  - Curve sampling (float channels)
- **Dependencies**: T-AN-1.3
- **Effort**: 3 days

### T-AN-2.2 -- Implement pose blending
- **Description**: Create `engine/animation/skeletal/blending.py` with blending functions
- **Acceptance Criteria**:
  - Linear pose blend (lerp): positions + scales
  - Spherical pose blend (slerp/nlerp): rotations
  - Additive pose blend: base + delta * weight
  - Inertialization: momentum-preserving transition (position + velocity continuity)
  - Crossfade with configurable curve (linear, ease-in, ease-out, step)
  - Per-bone blend weights for masking
- **Dependencies**: T-AN-1.2
- **Effort**: 4 days

### T-AN-2.3 -- Implement root motion
- **Description**: Create `engine/animation/skeletal/root_motion.py` for root bone extraction
- **Acceptance Criteria**:
  - Extract root bone delta from animation clip
  - Three modes: animation-driven, physics-driven, blended
  - Root motion accumulation across frames
  - Separation of horizontal/vertical/rotation components
- **Dependencies**: T-AN-2.1
- **Effort**: 2 days

### T-AN-2.4 -- Implement skeleton retargeting
- **Description**: Create `engine/animation/skeletal/retargeting.py`
- **Acceptance Criteria**:
  - Bone name mapping (source skeleton -> target skeleton)
  - Proportion correction: chain length normalization
  - IK retargeting: maintain foot/hand contact positions
  - Root motion scaling for different character sizes
  - Retargeting database for common skeleton types (human, quadruped)
- **Dependencies**: T-AN-1.1, T-AN-2.1
- **Effort**: 3 days

### T-AN-2.5 -- Implement clip compression
- **Description**: Create `engine/animation/skeletal/compression.py`
- **Acceptance Criteria**:
  - Keyframe reduction: remove keys within configurable tolerance (2-5x)
  - Quantization: 16-bit fixed-point for translations, 16-bit quaternion for rotations (2-4x)
  - Uniform sampling: convert to fixed-rate keyframes (1.5-2x)
  - Variable bitrate: per-track precision based on bone importance (3-6x)
  - Decompression: runtime unpacking with minimal overhead
  - Format selection: Raw (debug), ACL (production), custom codec
- **Dependencies**: T-AN-1.3
- **Effort**: 4 days

### T-AN-2.6 -- Create Rust playback + blending backend
- **Description**: Implement `crates/animation/playback.rs` and `crates/animation/blending.rs`
- **Acceptance Criteria**:
  - SIMD clip sampling (fetch 2 keyframes, interpolate all bone channels)
  - SIMD pose blending (parallel bone blend across SoA arrays)
  - Inertialization math (critically damped spring, derivative matching)
  - Event detection (binary search through event track)
- **Dependencies**: T-AN-1.4, T-AN-2.1, T-AN-2.2
- **Effort**: 3 days

### T-AN-2.7 -- Write playback + blending tests
- **Description**: Unit tests for all playback and blending features
- **Acceptance Criteria**:
  - Clip playback correctness (rate, looping, direction)
  - Event timing accuracy
  - Blend correctness (lerp, slerp, additive, inertialization)
  - Root motion extraction precision
  - Retargeting fidelity
  - Compression/decompression round-trip tolerance
- **Dependencies**: T-AN-2.1 through T-AN-2.6
- **Effort**: 2 days

---

## Phase 3: Skinning Compute Shaders (5 tasks)

**Dependencies**: Phase 1 (skeleton), mesh pipeline from rendering
**Files**: 1 Python + 1 Rust + 3 WGSL

### T-AN-3.1 -- Implement skinning orchestrator
- **Description**: Create `engine/animation/skeletal/skinning.py` with skinning pipeline
- **Acceptance Criteria**:
  - LBS implementation: weighted sum of bone matrices
  - DQS implementation: dual quaternion blend
  - Hybrid: LBS default, DQS for configured problem joints
  - Bone influence packing (4 influences -> 2 -> 1)
  - Corrective support: blend shapes, PSD, delta mush
  - Three target backends: GPU compute, GPU vertex, CPU SIMD
- **Dependencies**: T-AN-1.1
- **Effort**: 4 days

### T-AN-3.2 -- Write WGSL LBS compute shader
- **Description**: Create `shaders/skinning/skinning_lbs.comp.wgsl`
- **Acceptance Criteria**:
  - Thread per vertex (groups of 64)
  - Read bone matrices from SSBO (storage buffer)
  - Read vertex bind data from SSBO (influences, weights, bind position/normal)
  - Write final position + normal to output buffer
  - Support up to 4 bone influences per vertex
  - Async compute compatible (no UAV barriers within group)
- **Dependencies**: T-AN-3.1
- **Effort**: 3 days

### T-AN-3.3 -- Write WGSL DQS compute shader
- **Description**: Create `shaders/skinning/skinning_dqs.comp.wgsl`
- **Acceptance Criteria**:
  - Dual quaternion blending of bone transforms
  - Volume-preserving skinning (no candy-wrapper)
  - Same thread-per-vertex dispatch as LBS
  - Conditional: only process vertices flagged for DQS
  - Antipodality handling (sign correction for dual quaternion blend)
- **Dependencies**: T-AN-3.1
- **Effort**: 3 days

### T-AN-3.4 -- Write WGSL vertex shader skinning (fallback)
- **Description**: Create `shaders/skinning/skinning_vert.wgsl`
- **Acceptance Criteria**:
  - Vertex shader-based skinning (simpler path)
  - LBS only (no DQS in vertex shader)
  - Read bone matrices from uniform buffer or SSBO
  - Works on platforms without compute shader support
- **Dependencies**: T-AN-3.1
- **Effort**: 2 days

### T-AN-3.5 -- Write skinning unit + shader tests
- **Description**: Tests for skinning correctness
- **Acceptance Criteria**:
  - LBS vs DQS regression tests (known vertex outputs)
  - Hybrid switch correctness
  - Corrective blend shape application
  - Bone influence LOD switching (4 -> 2 -> 1)
  - GPU vs CPU skinning determinism (within float tolerance)
- **Dependencies**: T-AN-3.1 through T-AN-3.4
- **Effort**: 3 days

---

## Phase 4: IK Solver Library (7 tasks)

**Dependencies**: Phase 1 (skeleton), Phase 2 (pose)
**Files**: 6 Python + 1 Rust

### T-AN-4.1 -- Implement Two-Bone IK
- **Description**: Create `engine/animation/ik/two_bone.py` with analytical solver
- **Acceptance Criteria**:
  - Law of cosines angle calculation
  - Hip/shoulder swivel direction hint (elbow/knee orientation)
  - Constraint: maximum/minimum joint angles
  - O(1) performance, no iteration
  - Singularity handling (fully extended)
- **Dependencies**: T-AN-1.1, T-AN-1.2
- **Effort**: 2 days

### T-AN-4.2 -- Implement FABRIK solver
- **Description**: Create `engine/animation/ik/fabrik.py` with FABRIK algorithm
- **Acceptance Criteria**:
  - Forward pass: effector-to-root chain adjustment
  - Backward pass: root-to-effector chain adjustment
  - Iteration until convergence (or max iterations)
  - Configurable iteration count, tolerance
  - Handles chains of any length
  - Joint constraint enforcement per bone
- **Dependencies**: T-AN-1.1, T-AN-1.2
- **Effort**: 3 days

### T-AN-4.3 -- Implement CCD solver
- **Description**: Create `engine/animation/ik/ccd.py` with CCD algorithm
- **Acceptance Criteria**:
  - Iterative joint rotation to minimize effector-target distance
  - Configurable iteration count
  - Constraint handling (angle limits per joint)
  - Convergence detection
- **Dependencies**: T-AN-1.1, T-AN-1.2
- **Effort**: 2 days

### T-AN-4.4 -- Implement Jacobian solver
- **Description**: Create `engine/animation/ik/jacobian.py` with Jacobian-based IK
- **Acceptance Criteria**:
  - Jacobian matrix construction (3xN for position, 6xN for position+rotation)
  - Damped Least Squares (DLS) pseudo-inverse
  - Singular Value Decomposition for numerical stability
  - Multiple end effector support
  - Task prioritization (null-space projection)
- **Dependencies**: T-AN-1.1, T-AN-1.2 (requires Rust math library for matrix ops)
- **Effort**: 5 days

### T-AN-4.5 -- Implement Full-Body IK
- **Description**: Create `engine/animation/ik/fullbody.py` with multi-effector FB IK
- **Acceptance Criteria**:
  - Multiple end effectors: feet, hands, head, hips
  - Balance: center of mass projection within support polygon
  - Posture preservation: maintain natural spine curve
  - Joint limit enforcement across all chains
  - Priority layering: balance > foot placement > hand reach
  - Integration with all 4 lower-level IK solvers as components
- **Dependencies**: T-AN-4.1, T-AN-4.2, T-AN-4.3, T-AN-4.4
- **Effort**: 6 days

### T-AN-4.6 -- Implement IK goal system + foot placement
- **Description**: Create `engine/animation/ik/ik_goal.py` and `engine/animation/ik/foot_placement.py`
- **Acceptance Criteria**:
  - `@ik_goal` decorator: priority, blend_speed
  - IKGoal component: target_position, target_normal, weight
  - Foot IK: terrain height sampling, ankle roll, toe alignment
  - Goal priority resolution (higher priority solved first)
  - Smooth blend speed (prevents IK popping)
  - Temporal coherence (maintain contact during stance phase)
- **Dependencies**: T-AN-4.1, T-AN-4.2
- **Effort**: 3 days

### T-AN-4.7 -- Write IK solver tests
- **Description**: Unit tests for all 5 IK methods
- **Acceptance Criteria**:
  - Two-bone end effector reaches target within tolerance
  - FABRIK converges for N-bone chains
  - CCD resolves to target with constraints
  - Jacobian handles multiple effectors
  - FB IK maintains balance while reaching
  - Foot placement follows terrain
  - Performance benchmarks per solver
- **Dependencies**: T-AN-4.1 through T-AN-4.6
- **Effort**: 3 days

---

## Phase 5: Animation Graph Runtime (8 tasks)

**Dependencies**: Phase 2 (playback + blending)
**Files**: 6 Python + 1 Rust

### T-AN-5.1 -- Implement animation graph container
- **Description**: Create `engine/animation/graph/animation_graph.py`
- **Acceptance Criteria**:
  - Graph structure: layers, entry points, parameter mapping
  - Layer ordering and blending (base, upper body, additive, override)
  - Parameter set (float, int, bool, Vec3) driving evaluation
  - Dirty-flag evaluation: only re-evaluate when parameters change
  - Per-entity graph instances
  - Graph serialization via SerializableDescriptor
- **Dependencies**: T-AN-2.2
- **Effort**: 3 days

### T-AN-5.2 -- Implement state machine
- **Description**: Create `engine/animation/graph/state_machine.py`
- **Acceptance Criteria**:
  - States with associated clip/blend tree
  - Transitions: condition, blend_time, blend_curve, sync_mode
  - Condition evaluation: boolean expression on animation parameters
  - Transition queuing (don't interrupt current transition)
  - Wildcard transitions (from any state)
  - `@on_enter` / `@on_exit` hooks for state callbacks
  - Wire to StateMeta for state registration
- **Dependencies**: T-AN-5.1
- **Effort**: 4 days

### T-AN-5.3 -- Implement blend trees (1D, 2D, additive)
- **Description**: Create `engine/animation/graph/blend_tree.py`
- **Acceptance Criteria**:
  - 1D blend tree: single parameter, N clips, linear interpolation between nearest
  - 2D blend tree: two parameters, triangulation-based interpolation
  - 2D Directional blend space: direction + speed, radial interpolation
  - Additive blend space: base + overlay, per-bone mask
  - Blend parameter validation (range, type)
  - Wire `@blend_tree` decorator -> blend tree configuration
- **Dependencies**: T-AN-5.1, T-AN-2.2
- **Effort**: 4 days

### T-AN-5.4 -- Implement blend node types
- **Description**: Create `engine/animation/graph/blend_node.py`
- **Acceptance Criteria**:
  - ClipNode: plays a single animation clip
  - BlendNode: 1D/2D blend between children
  - AdditiveNode: additive blend of child onto base
  - OverrideNode: full-body override with blend weight
  - LayerNode: masked layer with bone mask
  - Node graph execution: recursive child evaluation
- **Dependencies**: T-AN-5.1
- **Effort**: 3 days

### T-AN-5.5 -- Implement animation layers
- **Description**: Create `engine/animation/graph/layer.py`
- **Acceptance Criteria**:
  - Layer with bone mask (which bones this layer affects)
  - Blend mode: override, additive, masked-additive
  - Layer weight (master influence)
  - Layer ordering: base first, additive last
  - Source blend (blend with previous layer result)
- **Dependencies**: T-AN-5.1, T-AN-2.2
- **Effort**: 2 days

### T-AN-5.6 -- Implement sync groups
- **Description**: Create `engine/animation/graph/sync.py`
- **Acceptance Criteria**:
  - Sync markers: labeled time positions in clips
  - Sync groups: clips synchronized to match markers
  - Leader-follower: leader plays at natural rate, followers match
  - Time warp: stretch/compress follower to match leader markers
- **Dependencies**: T-AN-5.2, T-AN-2.1
- **Effort**: 2 days

### T-AN-5.7 -- Create animation state machine system
- **Description**: Create `engine/animation/systems/animation_state_system.py` (simulation-side)
- **Acceptance Criteria**:
  - `@system(phase="animation")` annotation
  - Evaluates state machine state only (deterministic, simulation tick)
  - Outputs: current_state, blend_weights, sync_progress
  - Reads from Foundation Tracker dirty flags
  - Emits StateTransition events via EventLog
  - Does NOT perform playback or bone transforms (presentation phase)
- **Dependencies**: T-AN-5.2, T-AN-1.6
- **Effort**: 2 days

### T-AN-5.8 -- Write animation graph tests
- **Description**: Unit tests for state machine, blend trees, layers, sync
- **Acceptance Criteria**:
  - State machine transition correctness
  - Blend tree interpolation accuracy (1D, 2D)
  - Layer composition correctness
  - Sync group timing
  - Dirty-flag re-evaluation optimization
- **Dependencies**: T-AN-5.1 through T-AN-5.7
- **Effort**: 3 days

---

## Phase 6: Motion Matching (6 tasks)

**Dependencies**: Phase 2 (playback + blending), Phase 5 (fallback to state machine)
**Files**: 5 Python + 1 Rust

### T-AN-6.1 -- Implement motion matching database
- **Description**: Create `engine/animation/motionmatching/database.py`
- **Acceptance Criteria**:
  - Database of pre-processed animation frames
  - Per-frame features: pose, trajectory, foot contacts, tags
  - ANN index (KD-tree or VP-tree) for runtime search
  - Database build pipeline from animation clips
  - Serialization via AssetMeta with `.mmdb` extension
- **Dependencies**: T-AN-1.3, T-AN-1.5
- **Effort**: 4 days

### T-AN-6.2 -- Implement feature extraction
- **Description**: Create `engine/animation/motionmatching/features.py`
- **Acceptance Criteria**:
  - Pose features: joint positions/velocities relative to root (normalized by height)
  - Trajectory features: future root position/facing at T+0.2, T+0.5, T+1.0s
  - Foot features: contact state, position, velocity
  - Feature normalization (zero mean, unit variance)
  - Tagging: locomotion style, terrain type, action type
- **Dependencies**: T-AN-6.1
- **Effort**: 3 days

### T-AN-6.3 -- Implement runtime search
- **Description**: Create `engine/animation/motionmatching/search.py`
- **Acceptance Criteria**:
  - Cost function: weighted sum of pose + trajectory + velocity + transition cost
  - ANN nearest neighbor search (KD-tree)
  - Distance-based pruning (skip far poses)
  - Performance: search completes within configurable budget_ms
  - Configurable: cost weights, cost threshold, min clip time, stickiness
- **Dependencies**: T-AN-6.1, T-AN-6.2
- **Effort**: 4 days

### T-AN-6.4 -- Implement inertialization transitions
- **Description**: Create `engine/animation/motionmatching/transition.py`
- **Acceptance Criteria**:
  - Inertialization: derivative-continuous blend from current to matched pose
  - Handles position + velocity matching
  - Configurable blend duration
  - Fallback to crossfade when inertialization not suitable
  - Prevents foot sliding during transition
- **Dependencies**: T-AN-2.2, T-AN-6.3
- **Effort**: 3 days

### T-AN-6.5 -- Implement context system
- **Description**: Create `engine/animation/motionmatching/context.py`
- **Acceptance Criteria**:
  - Styling parameters: aggressive, cautious, injured (bias tagged clips)
  - Terrain adaptation: foot height to terrain
  - Query modifiers: ground-only, weapon-drawn, crouching
  - Style interpolation: blend between multiple cost profiles
- **Dependencies**: T-AN-6.3
- **Effort**: 2 days

### T-AN-6.6 -- Write motion matching tests
- **Description**: Unit tests for database, search, transitions
- **Acceptance Criteria**:
  - Database build correctness (feature extraction matches reference)
  - Search returns lowest-cost pose
  - Inertialization maintains continuity (no pose pop)
  - Context modifiers change search results correctly
  - Performance: budget_ms enforcement
- **Dependencies**: T-AN-6.1 through T-AN-6.5
- **Effort**: 3 days

---

## Phase 7: Facial & Procedural Animation (9 tasks)

**Dependencies**: Phase 2 (blending), Phase 4 (IK for gaze)
**Files**: 8 Python

### T-AN-7.1 -- Implement blend shape system
- **Description**: Create `engine/animation/facial/blend_shapes.py`
- **Acceptance Criteria**:
  - Morph target storage (vertex deltas per shape)
  - Blend weight application (linear combination of deltas)
  - Shape masking (per-region blendshape application)
  - Pre/post skinning application modes
  - 50-200 shapes per character support
- **Dependencies**: T-AN-3.1 (skinning pipeline)
- **Effort**: 3 days

### T-AN-7.2 -- Implement FACS action units
- **Description**: Create `engine/animation/facial/facs.py`
- **Acceptance Criteria**:
  - 52 FACS Action Units defined (AU1-AU46)
  - Each AU maps to blend shape(s) or bone transformation
  - AU intensity (0-1 continuous scale)
  - AU asymmetry support (left/right independent)
  - FACS -> blend shape weight conversion
- **Dependencies**: T-AN-7.1
- **Effort**: 3 days

### T-AN-7.3 -- Implement lip sync
- **Description**: Create `engine/animation/facial/lip_sync.py`
- **Acceptance Criteria**:
  - Phoneme-to-viseme mapping table
  - Audio analysis input or phoneme track input
  - Coarticulation smoothing (viseme blending)
  - Timing: phoneme duration and transition shaping
  - Performance: real-time lip sync
- **Dependencies**: T-AN-7.1
- **Effort**: 3 days

### T-AN-7.4 -- Implement eye animation
- **Description**: Create `engine/animation/facial/eye_animation.py`
- **Acceptance Criteria**:
  - Gaze IK: look-at target with Donders' law soft constraint
  - Saccades: random micro-movements (200-600ms interval)
  - Drift: slow continuous movement during fixation
  - Tremor: high-frequency micro-oscillations
  - Blinking: random interval (mean 4s), reactive, 100-400ms duration
  - Pupil dilation: response to light level
- **Dependencies**: T-AN-4.3 (CCD IK for gaze)
- **Effort**: 3 days

### T-AN-7.5 -- Implement spring/jiggle bones
- **Description**: Create `engine/animation/procedural/spring_bone.py`
- **Acceptance Criteria**:
  - Damped spring physics per bone
  - Parameters: stiffness, damping, gravity_scale, mass, wind influence
  - Collision: simple sphere/capsule collision with body
  - Used for: hair, cloth, accessories, tail
- **Dependencies**: T-AN-1.1
- **Effort**: 2 days

### T-AN-7.6 -- Implement look-at/aim controller
- **Description**: Create `engine/animation/procedural/lookat.py`
- **Acceptance Criteria**:
  - Look-at: rotate bone toward target position
  - Aim: lead target with velocity prediction
  - Soft cone limit (break target outside cone)
  - Per-bone weight distribution along chain
  - Used for: head tracking, eye gaze, weapon aiming
- **Dependencies**: T-AN-1.1
- **Effort**: 2 days

### T-AN-7.7 -- Implement twist distribution
- **Description**: Create `engine/animation/procedural/twist.py`
- **Acceptance Criteria**:
  - Spread twist rotation across bone chain
  - Configurable per-bone twist weight
  - Used for: forearm twist, spine twist
- **Dependencies**: T-AN-1.1
- **Effort**: 1 day

### T-AN-7.8 -- Implement ragdoll blending
- **Description**: Create `engine/animation/procedural/ragdoll.py`
- **Acceptance Criteria**:
  - Blend-in: animation to ragdoll over configurable time
  - Blend-out: ragdoll to animation (get-up animation)
  - Per-bone ragdoll activation (partial ragdoll)
  - Active ragdoll: muscle-driven recovery
  - Joint limit enforcement
  - Wire `@ragdoll` decorator
- **Dependencies**: T-AN-2.2 (blending), Physics (ragdoll bodies)
- **Effort**: 4 days

### T-AN-7.9 -- Write facial + procedural tests
- **Description**: Unit tests for facial and procedural animation
- **Acceptance Criteria**:
  - Blend shape application correctness
  - FACS AU combination accuracy
  - Lip sync phoneme timing
  - Eye saccade/ blink distribution
  - Spring bone physics stability
  - Look-at target accuracy
  - Ragdoll blend smoothness
- **Dependencies**: T-AN-7.1 through T-AN-7.8
- **Effort**: 3 days

---

## Phase 8: Crowd System (6 tasks)

**Dependencies**: Phase 2 (playback), Phase 3 (skinning)
**Files**: 3 Python + 2 WGSL + 1 Rust

### T-AN-8.1 -- Implement animation textures
- **Description**: Create `engine/animation/crowds/animation_texture.py`
- **Acceptance Criteria**:
  - Bake bone transforms to RGBA texture
  - Texture layout: bone_index x clip_frame
  - Update on animation change (not every frame)
  - Mipmap generation for texture filtering
  - Support for multiple animation clips per character
- **Dependencies**: T-AN-2.1
- **Effort**: 3 days

### T-AN-8.2 -- Implement instanced crowd renderer
- **Description**: Create `engine/animation/crowds/crowd_renderer.py`
- **Acceptance Criteria**:
  - GPU instancing with per-instance data buffer (position, animation_id, phase, lod)
  - Single draw call per crowd mesh
  - Per-instance animation time offset
  - LOD selection per instance
  - Frustum culling per instance
- **Dependencies**: T-AN-8.1
- **Effort**: 4 days

### T-AN-8.3 -- Write crowd skinning vertex shader
- **Description**: Create `shaders/crowd/crowd_skinning.vert.wgsl`
- **Acceptance Criteria**:
  - Skin vertex from animation texture
  - Sample bone transforms via texture read
  - Support per-instance animation_id and phase
  - Support multiple bone influence levels (4/2/1)
- **Dependencies**: T-AN-8.1
- **Effort**: 3 days

### T-AN-8.4 -- Write impostor shader
- **Description**: Create `shaders/crowd/impostor.frag.wgsl`
- **Acceptance Criteria**:
  - Billboard rendering for far LOD
  - Pre-rendered impostor textures from multiple angles
  - Smooth impostor-to-skeleton transition
  - Alpha fade at transition boundary
- **Dependencies**: T-AN-8.2
- **Effort**: 2 days

### T-AN-8.5 -- Implement crowd LOD system
- **Description**: Create `engine/animation/crowds/crowd_lod.py`
- **Acceptance Criteria**:
  - Three LOD levels: full skeleton, simplified (4-bone), impostor
  - Distance-based LOD selection
  - LOD transition smoothing (dither/fade)
  - Graph LOD: full -> state machine only -> baked
  - AI LOD: full AI -> rules -> flow field
  - Budget-aware LOD (lower quality when over budget)
- **Dependencies**: T-AN-8.2
- **Effort**: 3 days

### T-AN-8.6 -- Write crowd system tests
- **Description**: Unit + shader tests for crowds
- **Acceptance Criteria**:
  - Animation texture baking correctness
  - Instanced rendering: N instances produce N unique poses
  - LOD switching distance thresholds
  - Impostor rendering correctness
  - Performance: frame time with 1000+ agents
- **Dependencies**: T-AN-8.1 through T-AN-8.5
- **Effort**: 3 days

---

## Phase 9: Cinematics & Full Engine Integration (12 tasks)

**Dependencies**: All prior phases completed
**Files**: 2 Python + integration across all systems

### T-AN-9.1 -- Implement cutscene playback
- **Description**: Create `engine/animation/cinematics/cutscene.py`
- **Acceptance Criteria**:
  - Timeline: sequential animation events, camera cuts, dialogue triggers
  - Skippable (default) or forced cutscenes
  - Gameplay pause option
  - State save before cutscene, restore after
  - Wire `@cutscene` decorator
- **Dependencies**: All animation systems
- **Effort**: 3 days

### T-AN-9.2 -- Implement camera tracks
- **Description**: Create `engine/animation/cinematics/camera_track.py`
- **Acceptance Criteria**:
  - Camera animation: position, rotation, FOV over time
  - Spline interpolation (Catmull-Rom, cubic Bezier)
  - Blend in/out from gameplay camera
  - Look-at targets
  - Wire `@camera_track` decorator
- **Dependencies**: T-AN-9.1
- **Effort**: 3 days

### T-AN-9.3 -- Create animation graph system (presentation phase)
- **Description**: Create `engine/animation/systems/animation_graph_system.py`
- **Acceptance Criteria**:
  - `@system(phase="animation")` annotation
  - Reads state machine state from simulation (deterministic)
  - Performs actual playback: sample clips, blend, evaluate trees
  - Dirty-flag driven: re-evaluates only when parameters change
  - Outputs bone transforms to next system in pipeline
  - Task-parallel per entity evaluation
- **Dependencies**: T-AN-5.1 through T-AN-5.8
- **Effort**: 3 days

### T-AN-9.4 -- Create IK system
- **Description**: Create `engine/animation/systems/ik_system.py`
- **Acceptance Criteria**:
  - `@system(phase="animation")` annotation
  - Runs AFTER animation graph (evaluates on base pose)
  - Applies IK goals in priority order
  - Dispatches to appropriate solver per chain
  - Outputs adjusted pose to procedural/skinning system
- **Dependencies**: T-AN-4.1 through T-AN-4.7
- **Effort**: 2 days

### T-AN-9.5 -- Create procedural system
- **Description**: Create `engine/animation/systems/procedural_system.py`
- **Acceptance Criteria**:
  - `@system(phase="animation")` annotation
  - Runs AFTER IK system
  - Applies spring bones, look-at, twist, ragdoll blend
  - Per-bone procedural effect ordering
- **Dependencies**: T-AN-7.5 through T-AN-7.8
- **Effort**: 2 days

### T-AN-9.6 -- Create skinning system
- **Description**: Create `engine/animation/systems/skinning_system.py`
- **Acceptance Criteria**:
  - `@system(phase="animation")` annotation
  - Dispatches GPU skinning compute shaders
  - LOD-based influence reduction
  - CPU fallback path
  - Async compute overlap where possible
- **Dependencies**: T-AN-3.1 through T-AN-3.5
- **Effort**: 2 days

### T-AN-9.7 -- Create motion matching system
- **Description**: Create `engine/animation/systems/motion_matching_system.py`
- **Acceptance Criteria**:
  - `@system(phase="animation")` annotation
  - Replaces standard state machine for MM characters
  - Each frame: compute trajectory, search database, transition
  - Budget enforcement (motion_matching_budget_ms)
  - Fallback to state machine when budget exceeded
- **Dependencies**: T-AN-6.1 through T-AN-6.6
- **Effort**: 3 days

### T-AN-9.8 -- Create facial system
- **Description**: Create `engine/animation/systems/facial_system.py`
- **Acceptance Criteria**:
  - `@system(phase="animation")` annotation
  - Applies blend shapes, FACS, lip sync, eye animation
  - Layer: facial animation overrides facial bones
  - Integration with audio system for lip sync
- **Dependencies**: T-AN-7.1 through T-AN-7.4
- **Effort**: 2 days

### T-AN-9.9 -- Create crowd system
- **Description**: Create `engine/animation/systems/crowd_system.py`
- **Acceptance Criteria**:
  - `@system(phase="animation")` annotation
  - Agent steering update (RVO/ORCA)
  - Animation texture baking
  - LOD selection per agent
  - Frustum culling per agent
- **Dependencies**: T-AN-8.1 through T-AN-8.6
- **Effort**: 3 days

### T-AN-9.10 -- Wire Foundation: Tracker integration
- **Description**: Connect Foundation Tracker to animation systems
- **Acceptance Criteria**:
  - `TrackedDescriptor` on animation parameters (speed, direction, state)
  - `tracker.all_dirty()` in animation graph system
  - Only re-evaluate graph when parameters change
  - `tracker.on_change(AnimationState, callback)` for type-level subscriptions
  - Track current_state changes for state machine validation
- **Dependencies**: T-AN-9.3, Foundation Tracker
- **Effort**: 2 days

### T-AN-9.11 -- Wire Foundation: Session persistence
- **Description**: Connect Foundation Serializer/Session to animation state
- **Acceptance Criteria**:
  - Animation state machine state persists across sessions
  - Blend parameter values persist
  - IK enabled/disabled persists
  - Bone transforms do NOT persist (TransientDescriptor)
  - Active clip playback does NOT persist
- **Dependencies**: T-AN-9.3, Foundation Serializer, Session
- **Effort**: 2 days

### T-AN-9.12 -- Full animation system integration tests
- **Description**: End-to-end tests for complete animation pipeline
- **Acceptance Criteria**:
  - Full pipeline: skeleton -> playback -> IK -> procedural -> skinning -> output
  - Motion matching -> inertialization -> skinning
  - State machine -> blend tree -> facial -> skinning
  - Crowd system -> animation textures -> instanced rendering
  - Deterministic replay of animation state transitions
  - Frame time within budget across all LOD levels
- **Dependencies**: T-AN-9.1 through T-AN-9.11
- **Effort**: 5 days

---

## Summary

| Phase | Tasks | Files | Key Dependencies | Effort |
|-------|-------|-------|------------------|--------|
| 1: Skeleton & Foundation | 8 | 4 Python + 1 Rust | S15 (math, ECS), S16 (assets) | 2 weeks |
| 2: Playback & Blending | 7 | 5 Python + 2 Rust | Phase 1 | 2.5 weeks |
| 3: Skinning Compute | 5 | 1 Python + 1 Rust + 3 WGSL | Phase 1, Mesh pipeline | 2 weeks |
| 4: IK Solver Library | 7 | 6 Python + 1 Rust | Phase 1, Phase 2 | 3 weeks |
| 5: Animation Graph | 8 | 6 Python + 1 Rust | Phase 2 | 3 weeks |
| 6: Motion Matching | 6 | 5 Python + 1 Rust | Phase 2, Phase 5 | 2.5 weeks |
| 7: Facial & Procedural | 9 | 8 Python | Phase 2, Phase 4 | 3 weeks |
| 8: Crowd System | 6 | 3 Python + 2 WGSL + 1 Rust | Phase 2, Phase 3 | 2.5 weeks |
| 9: Cinematics & Integration | 12 | 2 Python + integration | All prior phases | 3.5 weeks |
| **TOTAL** | **68** | **40 Python + 7 Rust + 5 WGSL** | | **~24 weeks** |
