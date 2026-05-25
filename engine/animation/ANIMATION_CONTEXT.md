# Animation Layer — Implementation Context

> Everything needed to implement `engine/animation/`. No other document required.
>
> **Architecture spec:** `DIAGRAMS/ARCHITECTURE_ANIMATION.md` (1264 lines)
> **Integration spec:** `docs/GAME_ENGINE_INTEGRATION.md` §4.8
> **Trinity spec:** `docs/TRINITY_LATEST.md`
> **TODO checklist:** `docs/GAME_ENGINE_INTEGRATION_TODO.md` §7

---

## 1. Architecture Summary

The animation layer sits between Simulation and Rendering in the frame pipeline. It handles:
- Skeletal animation (bone hierarchies, clip playback, blending)
- Animation graphs (state machines, blend trees, layers)
- Inverse kinematics (FABRIK, CCD, Jacobian, full-body)
- Procedural animation (jiggle, spring, look-at, aim, twist bones)
- Motion matching (database search, trajectory/pose cost, inertialization)
- Ragdoll physics (blend-in, joint limits, recovery)
- Skinning (LBS, dual quaternion, GPU compute)
- Facial animation (blend shapes, FACS, lip sync)
- Crowds (GPU instancing, animation textures, LOD)
- Cinematics (cutscenes, camera tracks)

### Evaluation Pipeline
```
Sample Clips -> Blend (state machine / blend tree) -> IK -> Post-Process -> Skinning -> Output
```

### Frame Phase
Animation executes in phase order: Input -> Simulation -> **Animation** -> Rendering -> Audio -> Cleanup

### Determinism Rule
- **Animation State** (state machine, current state, sync points) -> DETERMINISTIC (lives in simulation tick)
- **Animation Playback** (bone transforms, blending, visual output) -> NON-DETERMINISTIC (presentation phase)
- State machines output to both sides: simulation gets damage windows/abilities/movement speed; presentation gets animation state/audio triggers/VFX

### Threading
- Animation evaluation is parallel (task-based per entity)
- SIMD for bone transform computation
- GPU compute for skinning and crowd animation

### Budget / LOD
- Full (60Hz): full graph evaluation, all IK, all procedural
- Half (30Hz): simplified graph, reduced IK iterations
- Quarter (15Hz): direct state machine, no IK
- Adaptive: distance-based, visibility culling, importance

---

## 2. Trinity Decorators for Animation

### 2.1 Core Animation Decorators (Tier 17: ANIMATION)

From `trinity/decorators/animation.py`:

#### @tween
```python
@tween(property: str, duration: float, easing: str = "linear")
```
Configures tween animation on a property.
- `property` (str, required): Property name to animate
- `duration` (float, required, >0): Duration in seconds
- `easing` ("linear" | "ease_in" | "ease_out" | "ease_in_out" | "bounce"): Easing function
- **Steps**: TAG(tween=True), TAG(tween_property), TAG(tween_duration), TAG(tween_easing), REGISTER(animation)
- **Use**: Simple property interpolation (UI elements, camera, gameplay values)

#### @blend_tree
```python
@blend_tree(parameter: str, clips: list)
```
Configures a blend tree with a driving parameter and clip list.
- `parameter` (str, required): Blend parameter name (e.g., "speed", "direction")
- `clips` (list, required, non-empty): List of animation clip names
- **Steps**: TAG(blend_tree=True), TAG(blend_parameter), TAG(blend_clips), REGISTER(animation)
- **Use**: 1D/2D blend spaces (walk/run by speed, directional locomotion)

### 2.2 IK & Procedural Decorators (Tier 44: IK_PROCEDURAL)

From `trinity/decorators/ik_procedural.py`:

#### @ik_chain
```python
@ik_chain(solver: str = "fabrik", iterations: int = 10)
```
Configures an IK chain with solver and iteration count.
- `solver` ("fabrik" | "ccd" | "jacobian" | "fullbody"): Solver algorithm
- `iterations` (int, >0): Max solver iterations
- **Steps**: TAG(ik_chain), TAG(ik_solver), TAG(ik_iterations), REGISTER(ik_procedural)
- **Use**: Arm/leg IK, reach targets, foot placement

#### @ik_goal
```python
@ik_goal(priority: int = 0, blend_speed: float = 10.0)
```
Defines an IK target with priority and blend speed.
- `priority` (int): Higher = more important when multiple goals conflict
- `blend_speed` (float, >0): Speed to blend toward target
- **Steps**: TAG(ik_goal), TAG(ik_goal_priority), TAG(ik_goal_blend_speed), REGISTER(ik_procedural)
- **Use**: Foot IK targets, hand grip points, look-at targets

#### @procedural_bone
```python
@procedural_bone(type: str)
```
Marks a bone for procedural animation.
- `type` ("jiggle" | "spring" | "lookat" | "aim" | "twist"): Procedural behavior
- **Steps**: TAG(procedural_bone), TAG(procedural_bone_type), REGISTER(ik_procedural)
- **Use**: Hair/cloth jiggle, weapon aim, spine twist distribution

#### @motion_matching
```python
@motion_matching(database: str, trajectory_weight: float = 1.0, pose_weight: float = 1.0)
```
Configures motion matching with a database and cost weights.
- `database` (str, required): Motion database asset reference
- `trajectory_weight` (float, >0): Weight for trajectory matching cost
- `pose_weight` (float, >0): Weight for pose matching cost
- **Steps**: TAG(motion_matching), TAG(motion_database), TAG(motion_trajectory_weight), TAG(motion_pose_weight), REGISTER(ik_procedural)
- **Use**: High-quality locomotion, responsive movement without hand-authored transitions

#### @ragdoll
```python
@ragdoll(blend_time: float = 0.2, joint_limits: bool = True)
```
Configures ragdoll physics blend.
- `blend_time` (float, >=0): Time to blend from animation to ragdoll
- `joint_limits` (bool): Whether to enforce anatomical joint constraints
- **Steps**: TAG(ragdoll), TAG(ragdoll_blend_time), TAG(ragdoll_joint_limits), REGISTER(ik_procedural)
- **Use**: Death ragdoll, hit reactions, physics-driven stumble/recovery

### 2.3 State Machine Decorators (Tier: STATE_MACHINE)

From `trinity/decorators/state_machine.py`:

#### @state_machine
```python
@state_machine(initial: str, states: set, transitions: dict = None)
```
Defines a state machine with states and valid transitions.
- `initial` (str, required, must be in `states`): Starting state
- `states` (set, required, non-empty): Valid state names
- `transitions` (dict, optional): {source_state: [valid_targets]}
- **Steps**: TAG(state_machine), TAG values, REGISTER(state_machine)
- **Validation**: Initial must be in states; all transition sources/targets must be valid states
- **Use**: Animation state machines (idle -> walk -> run -> jump)

#### @on_enter / @on_exit
```python
@on_enter(state: str)
@on_exit(state: str)
```
Hooks for state entry/exit callbacks.
- `state` (str, required): State name to hook
- **Use**: Trigger sound on entering "attack" state, reset variables on exiting "jump"

### 2.4 Cinematic Decorators (Tier 35: CINEMATICS)

From `trinity/decorators/cinematics.py`:

#### @cutscene
```python
@cutscene(id: str, skippable: bool = True, pause_gameplay: bool = True)
```
Defines a cutscene sequence.
- `id` (str, required): Cutscene identifier
- `skippable` (bool): Whether player can skip
- `pause_gameplay` (bool): Whether to pause game systems during playback

#### @camera_track
```python
@camera_track(blend_in: float = 0.5, blend_out: float = 0.5)
```
Camera animation track with blend transitions.
- `blend_in` / `blend_out` (float, >=0): Blend times in seconds

### 2.5 Replay Decorators (Tier: REPLAY)

From `trinity/decorators/replay.py`:

#### @recorded
```python
@recorded(frequency: str = "fixed_tick")
```
Marks a component for animation recording. Frequency: "every_frame" | "fixed_tick" | "on_change".

#### @replay_authority
```python
@replay_authority(source: str = "recording")
```
Replay behavior source: "recording" | "simulation" | "hybrid".

#### @keyframe
```python
@keyframe(interval: float = 1.0)
```
Snapshot interval for replay seeking (seconds, >0).

### 2.6 Related Stacks

From `trinity/decorators/builtin_stacks/ai.py`:

#### @complete_ai (includes @state_machine)
```python
@complete_ai(
    behavior_tree_id: str,
    sense: str = "sight", sense_range: float = 50, sense_fov: float = 120,
    states: set = None, initial_state: str = "idle"
)
```
Includes `@state_machine(initial=initial_state, states=states)` -- the AI state machine often drives animation states. Default states: {"idle", "alert", "combat"}.

**No dedicated animation composite stack exists yet.** Consider creating one:
```python
# Potential: @animated_character stack
# Composes: @component + @tracked + @blend_tree + @ik_chain + @ragdoll + @serializable
```

---

## 3. Metaclasses Relevant to Animation

### ComponentMeta
Animated entities have animation components:
```python
@component
@tracked
@blend_tree(parameter="speed", clips=["idle", "walk", "run"])
@ik_chain(solver="fabrik", iterations=10)
class AnimatedCharacter(Component):
    current_state: Annotated[str, Tracked()] = "idle"
    blend_weight: Annotated[float, Tracked(), Range(0.0, 1.0)] = 0.0
    speed: Annotated[float, Tracked(), Range(0.0, 10.0)] = 0.0
    ik_enabled: Annotated[bool, Tracked()] = True
```

### SystemMeta
Animation systems run in the "animation" phase:
```python
@system(phase="animation")
@traced
class AnimationGraphSystem(System):
    def update(self, dt: float):
        # Evaluate state machines, blend trees, sample clips
        pass

@system(phase="animation")
class IKSystem(System):
    def update(self, dt: float):
        # Solve IK chains after graph evaluation
        pass
```

### StateMeta
Animation state machines use StateMeta:
```python
@state
class LocomotionState(State):
    _valid_transitions = {
        "idle": ["walk", "run", "jump", "fall"],
        "walk": ["idle", "run", "jump"],
        "run": ["idle", "walk", "jump", "slide"],
        "jump": ["fall", "land"],
        "fall": ["land"],
        "land": ["idle", "walk", "run"],
        "slide": ["idle", "run"],
    }
```

### AssetMeta
Animation data is loaded as Assets:
```python
@asset(extensions=[".anim", ".fbx", ".glb"])
class AnimationClip(Asset):
    duration: float = 0.0
    frame_rate: float = 30.0
    bone_count: int = 0
    loop: bool = False
    root_motion: bool = False
    compressed: bool = True

@asset(extensions=[".mmdb"])
class MotionDatabase(Asset):
    clip_count: int = 0
    feature_dimension: int = 0
    total_frames: int = 0

@asset(extensions=[".skel"])
class Skeleton(Asset):
    bone_count: int = 0
    has_retarget_data: bool = False
```

### EventMeta
Animation events fire during playback:
```python
@event
class AnimationNotify(Event):
    entity_id: int
    clip_name: str
    notify_name: str  # e.g., "footstep_left", "attack_window_open"
    time: float

@event
class StateTransition(Event):
    entity_id: int
    from_state: str
    to_state: str

@event
class RagdollActivated(Event):
    entity_id: int
    blend_time: float
```

### ResourceMeta
Global animation configuration:
```python
@resource
class AnimationConfig(Resource):
    global_speed: float = 1.0
    max_active_ik_chains: int = 32
    motion_matching_budget_ms: float = 2.0
    lod_distances: list = [10.0, 25.0, 50.0, 100.0]
```

---

## 4. Descriptors Relevant to Animation

### TrackedDescriptor
Animation parameters that drive blending and state:
```python
@component
@tracked
class AnimationState(Component):
    current_state: Annotated[str, Tracked()] = "idle"
    blend_weight: Annotated[float, Tracked(), Range(0.0, 1.0)] = 0.0
    speed: Annotated[float, Tracked()] = 0.0
    direction: Annotated[float, Tracked(), Range(-180.0, 180.0)] = 0.0
```
- `TrackedDescriptor.post_set()` -> `tracker.mark_dirty(obj, field, old, new)`
- Animation graph system reads dirty state parameters each frame
- Only re-evaluates graph when parameters actually change

### ValidatedDescriptor / RangeDescriptor
Animation parameters have strict ranges:
- Blend weights: 0.0-1.0
- Speed: >=0
- Direction: -180 to 180
- IK iterations: >0
- Blend time: >=0

### InterpolatedDescriptor (Phase 7)
Smoothly interpolate animation parameters between frames:
- Blend weights transition smoothly rather than snapping
- Camera blend_in/blend_out use interpolation
- IK goal positions interpolate to targets

### TransientDescriptor
Runtime animation state NOT saved:
- Current bone transforms (recalculated from state)
- Active IK solver state
- Motion matching search results
- Ragdoll physics body handles
- Decode cursors for streaming clips
```python
@component
@blend_tree(parameter="speed", clips=["idle", "walk", "run"])
class AnimatedCharacter(Component):
    speed: float = 0.0
    _bone_transforms: Annotated[list, Transient()] = None  # Not saved
    _active_clip_time: Annotated[float, Transient()] = 0.0  # Not saved
    _mm_last_pose_id: Annotated[int, Transient()] = -1     # Not saved
```

### SerializableDescriptor
State machine current state, blend parameters, and config save/load:
- Current animation state persists in Session
- Blend tree parameter values persist
- IK enabled/disabled state persists

### NetworkedDescriptor
In multiplayer, animation state replicates:
```python
@component
@tracked
@networked(authority="server")
class NetworkedAnimation(Component):
    state: Annotated[str, Tracked(), Networked()] = "idle"
    speed: Annotated[float, Tracked(), Networked(interpolate=True)] = 0.0
    # Bone transforms are NOT replicated -- each client evaluates locally from state
```

### ProfiledDescriptor
For animation performance monitoring:
- Track graph evaluation time per entity
- Track IK solve time
- Track motion matching search time
- Identify expensive animation graphs

---

## 5. Foundation Integration Points

### 5.1 Registry
- All animation Components registered via ComponentMeta -> Foundation Registry
- All animation Assets (clips, skeletons, motion databases) via AssetMeta
- All animation Events (notify, state transition, ragdoll) via EventMeta
- All animation States (locomotion states) via StateMeta
- Animation config Resource via ResourceMeta
- **Query at startup**: `registry.subclasses(Asset)` filtered by animation extensions

### 5.2 Tracker
- Animation systems read `tracker.all_dirty()` to find entities with changed animation parameters
- Only re-evaluate animation graph when state/speed/direction actually change
- `tracker.on_change(AnimationState, callback)` for type-level subscriptions
- Track current_state changes for state machine transition validation

### 5.3 EventLog
- Record state transitions with causal chains (what caused the state change?)
- Record animation notifies (footstep events, attack windows)
- `@traced` on animation system update methods for profiling
- Used by: replay system (re-trigger animation events), debugging ("why did character enter ragdoll?")

### 5.4 Mirror
- `mirror(animated_character)` returns field info for Inspector display
- Schema hash for animation asset versioning
- Used by: animation editor, runtime bone visualization, blend weight display

### 5.5 Bridge / ShellLang
- `world.query(has=AnimatedCharacter, where=lambda a: a.current_state == "attack")` -- find attacking entities
- `entity.animation.speed = 5.0` -- live animation parameter tweaking
- `entity.animation.current_state = "idle"` -- force state change from Shell

### 5.6 Session
- Current animation state persists
- Blend tree parameter values persist
- IK enabled/disabled persists
- Bone transforms do NOT persist (recalculated)
- Active clip playback positions do NOT persist

---

## 6. Architecture Spec Details

### 6.1 Skeletal Systems
**Skeleton**: Array of bones with parent indices, bind poses, names. Each bone has local/model/world transforms.
**Poses**: Bind pose (T-pose), reference pose (default), current pose (evaluated), delta pose (additive).
**Retargeting**: Map animations between different skeletons via bone name mapping + proportion correction.

### 6.2 Animation Data
**Clips**: Per-bone tracks with keyframes (time, value, interpolation mode, tangents).
**Channels**: Transform (pos/rot/scale), Float (blend shapes, material params), Attribute (events, curves).
**Compression**: Keyframe reduction (2-5x), quantization (2-4x), uniform sampling (1.5-2x), variable bitrate (3-6x).
**Formats**: Raw (debug), ACL (production, best ratio), Oodle Animation (platform-specific), Custom codec.

### 6.3 Skinning
**LBS** (Linear Blend Skinning): Fast, standard. Artifacts at extreme joint angles (candy wrapper).
**DQS** (Dual Quaternion): Volume-preserving, no candy wrapper. Higher cost.
**Hybrid**: LBS for most, DQS for problem joints.
**Correctives**: Blend shapes, RBF interpolation, pose space deformation, delta mush.
**Compute**: GPU vertex shader (most common), GPU compute shader (crowd), CPU SIMD (fallback).
**LOD**: 4 bone influences -> 2 -> 1 -> skip skinning.

### 6.4 Animation Playback
**Control**: play rate, current time, loop mode (once/loop/ping-pong), direction (forward/reverse).
**Pose sampling**: Linear interpolation for position, slerp/nlerp for rotation.
**Events**: Notifies (named events at specific times), sync markers (for blending sync), curves (continuous float data).
**Root Motion**: Extract root bone movement and apply to entity transform. Modes: animation-driven, physics-driven, blended.

### 6.5 Animation Graphs
**State Machines**: States with associated clips/blend trees. Transitions have: condition, blend time, curve, sync mode.
**Blend Trees**: 1D (speed -> idle/walk/run), 2D (speed + direction -> omnidirectional locomotion), Additive (base + overlay).
**Layers**: Full body, upper body override, additive. Each layer has bone mask defining affected bones.
**Blending modes**: Lerp, slerp, additive, inertialization (momentum-based, no blend time, instant transitions).

### 6.6 Inverse Kinematics
**Two-Bone IK**: Analytical (law of cosines). Fast. For arms/legs.
**FABRIK**: Iterative, forward-and-backward reaching. Natural-looking, handles chains of any length.
**CCD**: Cyclic Coordinate Descent. Simple, handles constraints well.
**Jacobian**: Matrix-based. Flexible but expensive. For complex chains.
**Full-Body IK**: Multiple effectors, joint limits, balance, posture. For full character adjustment.
**Runtime uses**: Foot placement (align feet to terrain), hand placement (grab ledges), weapon IK (aim direction), environmental (duck under obstacles).

### 6.7 Procedural Animation
**Jiggle/Spring bones**: Secondary motion for hair, cloth, accessories. Damped spring physics.
**Look-at/Aim**: Rotate bones to face target. For head tracking, weapon aiming.
**Twist distribution**: Spread twist rotation across chain (forearm twist).
**Procedural locomotion**: Walk cycles generated from parameters. For multi-legged creatures, terrain adaptation.
**Physics-driven**: Blend between animation and ragdoll. Hit reactions, stumble, recovery.

### 6.8 Motion Matching
**Database**: Pre-processed animation clips with extracted features per frame:
  - Pose features: joint positions/velocities relative to root
  - Trajectory: future root position/facing at T+0.2, T+0.5, T+1.0s
  - Foot features: contact state, position, velocity
  - Tags: locomotion style, terrain type, action
**Runtime search**: Cost function = weighted sum of (pose cost + trajectory cost + velocity cost + transition cost). Find lowest cost frame.
**Transition**: Inertialization (blend current momentum into new animation). Configurable: cost threshold, min time in current clip, stickiness.

### 6.9 Facial Animation
**Blend shapes**: Morph targets (50-200 shapes). FACS: 52 Action Units mapping to facial muscles.
**Lip sync**: Phoneme -> viseme mapping. Audio analysis or neural-network driven.
**Eye animation**: Gaze IK (look at target), micro-movements (saccades, drift, tremor), blinking (random intervals + reactive), pupil dilation.

### 6.10 Crowds
**Rendering**: GPU instancing with animation textures (bone transforms baked to texture). Impostors for far LOD.
**Simulation**: Agent steering, RVO/ORCA collision avoidance, flow fields, formations.
**LOD**: Full skeleton -> simplified -> impostor. Full graph -> state machine -> baked. Full AI -> rules -> flow.

### 6.11 Evaluation and Parallel Processing
- **Evaluation**: Sample -> Blend -> IK -> Post-Process -> Output
- **Parallel**: Task-based per entity, SIMD bone math, GPU compute skinning
- **Budget**: Priority queue, distance-based LOD, visibility culling, adaptive update rate

---

## 7. TODO Checklist

From `GAME_ENGINE_INTEGRATION_TODO.md` section 7:

### 7.1 Skeletal Animation
- [ ] Implement skeleton/bone hierarchy
- [ ] Implement animation clip playback (sampling, looping, events)
- [ ] Implement animation blending (linear, additive)
- [ ] Wire `@tween` decorator -> tween animation support
- [ ] Wire `@blend_tree` decorator -> blend tree configuration

### 7.2 Animation Graph
- [ ] Implement state machine (states, transitions, conditions)
- [ ] Implement blend trees (1D, 2D, additive)
- [ ] Wire StateMeta -> animation state registration
- [ ] Integrate Foundation Tracker -- track animation state changes

### 7.3 IK & Procedural
- [ ] Implement IK solvers (two-bone, FABRIK, CCD)
- [ ] Implement procedural animation (look-at, foot placement, ragdoll blend)
- [ ] Wire `@ik_procedural` decorators -> IK target configuration

### 7.4 Motion Matching
- [ ] Implement motion matching database (pose search, trajectory matching)
- [ ] Implement motion matching runtime (query, blend, transition)
- [ ] Wire motion data as Assets via AssetMeta

### 7.5 Facial & Skinning
- [ ] Implement blend shape / morph target system
- [ ] Implement skinning (LBS, dual quaternion)
- [ ] Implement facial animation (FACS, visemes)

---

## 8. Directory Structure

```
engine/animation/
├── __init__.py
├── ANIMATION_CONTEXT.md           <- This file
├── skeletal/
│   ├── __init__.py
│   ├── skeleton.py                # Skeleton asset, bone hierarchy
│   ├── pose.py                    # Pose representation (bind, current, delta)
│   ├── clip.py                    # AnimationClip asset, sampling
│   ├── clip_player.py             # Playback control (time, rate, loop, events)
│   ├── blending.py                # Pose blending (lerp, slerp, additive, inertialization)
│   ├── root_motion.py             # Root motion extraction and application
│   ├── retargeting.py             # Skeleton retargeting (bone mapping, proportion)
│   ├── compression.py             # Clip compression (keyframe reduction, quantization)
│   └── skinning.py                # LBS, DQS, hybrid, correctives, GPU compute
├── graph/
│   ├── __init__.py
│   ├── animation_graph.py         # Graph container (layers, state machines, blend trees)
│   ├── state_machine.py           # State machine (states, transitions, conditions)
│   ├── blend_tree.py              # Blend trees (1D, 2D, additive)
│   ├── blend_node.py              # Blend node types (lerp, additive, override)
│   ├── layer.py                   # Animation layers with bone masks
│   └── sync.py                    # Sync markers, sync groups, leader-follower
├── ik/
│   ├── __init__.py
│   ├── two_bone.py                # Analytical two-bone IK
│   ├── fabrik.py                  # FABRIK solver
│   ├── ccd.py                     # CCD solver
│   ├── jacobian.py                # Jacobian solver
│   ├── fullbody.py                # Full-body IK (multi-effector, balance)
│   ├── ik_goal.py                 # IK target component
│   └── foot_placement.py          # Foot IK (terrain adaptation)
├── motionmatching/
│   ├── __init__.py
│   ├── database.py                # Motion database asset (features, search index)
│   ├── features.py                # Feature extraction (pose, trajectory, foot)
│   ├── search.py                  # Runtime search (cost function, KD-tree)
│   ├── transition.py              # Inertialization transitions
│   └── context.py                 # Contextual matching, style matching
├── facial/
│   ├── __init__.py
│   ├── blend_shapes.py            # Morph target system
│   ├── facs.py                    # FACS action units
│   ├── lip_sync.py                # Phoneme-to-viseme, audio analysis
│   └── eye_animation.py           # Gaze IK, saccades, blinking
├── procedural/
│   ├── __init__.py
│   ├── spring_bone.py             # Jiggle/spring secondary motion
│   ├── lookat.py                  # Look-at/aim bone controller
│   ├── twist.py                   # Twist distribution
│   ├── ragdoll.py                 # Ragdoll blend, recovery
│   └── locomotion.py              # Procedural walk cycles
├── crowds/
│   ├── __init__.py
│   ├── animation_texture.py       # Bake bone transforms to GPU texture
│   ├── crowd_renderer.py          # GPU instanced crowd rendering
│   └── crowd_lod.py               # Crowd LOD (skeleton -> impostor)
└── systems/
    ├── __init__.py
    ├── animation_graph_system.py   # Main graph eval (@system phase="animation")
    ├── ik_system.py                # IK solve after graph
    ├── procedural_system.py        # Procedural bones (jiggle, spring, aim)
    ├── skinning_system.py          # Skinning compute (GPU/CPU)
    ├── motion_matching_system.py   # Motion matching search + transition
    ├── facial_system.py            # Facial animation update
    └── crowd_system.py             # Crowd animation update
```

---

## 9. Canonical Usage Examples

### Animated Character with Blend Tree + IK
```python
@component
@tracked
@blend_tree(parameter="speed", clips=["idle", "walk", "run", "sprint"])
@ik_chain(solver="fabrik", iterations=10)
@ragdoll(blend_time=0.3, joint_limits=True)
class CharacterAnimation(Component):
    speed: Annotated[float, Tracked(), Range(0.0, 15.0)] = 0.0
    direction: Annotated[float, Tracked(), Range(-180.0, 180.0)] = 0.0
    current_state: Annotated[str, Tracked()] = "idle"
    ik_enabled: Annotated[bool, Tracked()] = True
    ragdoll_active: Annotated[bool, Tracked()] = False
    _bone_transforms: Annotated[list, Transient()] = None
    _active_clip_time: Annotated[float, Transient()] = 0.0
```

### Locomotion State Machine
```python
@state
class LocomotionState(State):
    _valid_transitions = {
        "idle": ["walk", "run", "jump", "fall"],
        "walk": ["idle", "run", "jump"],
        "run": ["idle", "walk", "jump", "slide"],
        "jump": ["fall"],
        "fall": ["land"],
        "land": ["idle", "walk", "run"],
        "slide": ["idle", "run"],
    }
```

### Motion Matching Character
```python
@component
@tracked
@motion_matching(database="locomotion_db", trajectory_weight=1.0, pose_weight=0.8)
class MotionMatchedCharacter(Component):
    desired_velocity: Annotated[tuple, Tracked()] = (0.0, 0.0, 0.0)
    desired_facing: Annotated[float, Tracked()] = 0.0
    _current_pose_id: Annotated[int, Transient()] = -1
    _search_cost: Annotated[float, Transient()] = 0.0
```

### IK Foot Placement
```python
@component
@ik_goal(priority=1, blend_speed=15.0)
class FootIKTarget(Component):
    target_position: Annotated[tuple, Tracked()] = (0.0, 0.0, 0.0)
    target_normal: Annotated[tuple, Tracked()] = (0.0, 1.0, 0.0)
    weight: Annotated[float, Tracked(), Range(0.0, 1.0)] = 1.0
```

### Procedural Bones
```python
@component
@procedural_bone(type="spring")
class HairBone(Component):
    stiffness: Annotated[float, Tracked(), Range(0.0, 100.0)] = 50.0
    damping: Annotated[float, Tracked(), Range(0.0, 1.0)] = 0.3
    gravity_scale: float = 1.0
```

### Animation Assets
```python
@asset(extensions=[".anim", ".fbx", ".glb"])
class AnimationClip(Asset):
    duration: float = 0.0
    frame_rate: float = 30.0
    bone_count: int = 0
    loop: bool = False
    root_motion: bool = False
    events: list = None  # AnimationNotify timestamps

@asset(extensions=[".mmdb"])
class MotionDatabase(Asset):
    clip_count: int = 0
    feature_dimension: int = 0
    total_frames: int = 0

@asset(extensions=[".skel"])
class Skeleton(Asset):
    bone_count: int = 0
    has_retarget_data: bool = False
```

### Animation Graph System
```python
@system(phase="animation")
@traced
class AnimationGraphSystem(System):
    def update(self, dt: float):
        for entity in self.query(CharacterAnimation):
            anim = entity.character_animation
            dirty = tracker.dirty_fields(anim)
            
            if dirty:
                # Re-evaluate state machine transitions
                if 'speed' in dirty or 'current_state' in dirty:
                    new_state = self._evaluate_transitions(anim)
                    if new_state != anim.current_state:
                        self.emit(StateTransition(
                            entity_id=entity.id,
                            from_state=anim.current_state,
                            to_state=new_state
                        ))
                        anim.current_state = new_state
                
                # Evaluate blend tree with current parameters
                pose = self._evaluate_blend_tree(anim)
                anim._bone_transforms = pose
```

### Animation Events
```python
@event
class AnimationNotify(Event):
    entity_id: int
    clip_name: str
    notify_name: str
    time: float

@event
class StateTransition(Event):
    entity_id: int
    from_state: str
    to_state: str

@event
class RagdollActivated(Event):
    entity_id: int
    impact_force: float
```

### Cutscene
```python
@cutscene(id="boss_intro", skippable=True, pause_gameplay=True)
@camera_track(blend_in=1.0, blend_out=0.5)
class BossIntroCutscene:
    duration: float = 15.0
```

---

## 10. Key Integration Patterns

### Pattern: Dirty-Flag Driven Graph Evaluation
Animation graph only re-evaluates when parameters change:
```
Game Thread: entity.animation.speed = 5.0
  -> TrackedDescriptor.post_set()
  -> tracker.mark_dirty(entity.animation, "speed", 0.0, 5.0)

Animation Phase: AnimationGraphSystem.update()
  -> for entity in tracker.all_dirty() where has(CharacterAnimation):
  ->   re-evaluate blend tree with new speed
  ->   output new bone transforms
```

### Pattern: State Machine -> Animation + Simulation
State machines live in the deterministic simulation boundary but drive both sides:
```
Simulation (deterministic):
  State "attack" -> enable damage hitbox, consume stamina
  
Presentation (non-deterministic):
  State "attack" -> play attack animation, trigger VFX, play sound
```
Sync points are defined by simulation tick, NOT animation time.

### Pattern: IK After Graph
IK runs AFTER the animation graph produces a base pose:
```
1. AnimationGraphSystem: evaluate state machine -> blend tree -> base pose
2. IKSystem: adjust base pose for foot placement, hand grips, look-at
3. ProceduralSystem: add secondary motion (jiggle, spring)
4. SkinningSystem: compute final vertex positions
```

### Pattern: Motion Matching as Alternative to State Machine
Motion matching replaces hand-authored state machines with database search:
- NO state machine needed -- just set desired_velocity and desired_facing
- Runtime searches database for best matching pose
- Inertialization handles all transitions automatically
- Higher quality, lower authoring cost, more memory

### Pattern: Foundation EventLog for Animation Replay
```
EventLog records:
  tick=100: StateTransition(entity=42, from="idle", to="attack")
  tick=100: AnimationNotify(entity=42, clip="sword_slash", notify="damage_start")
  tick=115: AnimationNotify(entity=42, clip="sword_slash", notify="damage_end")
  tick=120: StateTransition(entity=42, from="attack", to="idle")

Replay: re-trigger transitions and notifies at exact ticks
```

---

## 11. Decorator Quick Reference

| Decorator | Tier | File | Registry | Key Params |
|-----------|------|------|----------|------------|
| @tween | 17 | animation.py | animation | property, duration, easing |
| @blend_tree | 17 | animation.py | animation | parameter, clips |
| @ik_chain | 44 | ik_procedural.py | ik_procedural | solver, iterations |
| @ik_goal | 44 | ik_procedural.py | ik_procedural | priority, blend_speed |
| @procedural_bone | 44 | ik_procedural.py | ik_procedural | type |
| @motion_matching | 44 | ik_procedural.py | ik_procedural | database, trajectory_weight, pose_weight |
| @ragdoll | 44 | ik_procedural.py | ik_procedural | blend_time, joint_limits |
| @state_machine | -- | state_machine.py | state_machine | initial, states, transitions |
| @on_enter | -- | state_machine.py | state_machine | state |
| @on_exit | -- | state_machine.py | state_machine | state |
| @cutscene | 35 | cinematics.py | cinematics | id, skippable, pause_gameplay |
| @camera_track | 35 | cinematics.py | cinematics | blend_in, blend_out |
| @recorded | -- | replay.py | replay | frequency |
| @replay_authority | -- | replay.py | replay | source |
| @keyframe | -- | replay.py | replay | interval |

## 12. Descriptor Quick Reference

| Descriptor | Use in Animation | Foundation System |
|-----------|-----------------|------------------|
| TrackedDescriptor | Speed, direction, state, blend weight -> dirty flags | Tracker |
| ValidatedDescriptor | Blend weight 0-1, speed >=0, iterations >0 | (internal) |
| RangeDescriptor | Numeric parameter clamping | (internal) |
| InterpolatedDescriptor | Smooth blend weight transitions, camera blends | Tracker |
| ObservableDescriptor | Editor/inspector callbacks on param change | Tracker |
| NetworkedDescriptor | Replicate animation state for multiplayer | Tracker |
| SerializableDescriptor | State machine state, config save/load | Mirror |
| TransientDescriptor | Bone transforms, clip cursors, IK state (NOT saved) | Mirror |
| ProfiledDescriptor | Graph eval timing, IK solve timing | EventLog |
