# PHASE 2 ARCHITECTURE: Animation Systems Integration

**Generated:** 2026-05-23
**Subsystem:** engine/animation/systems

---

## Phase Overview

Phase 2 builds upon Phase 1 core skeletal animation by providing ECS systems that process animation state per-frame. This phase covers the `engine/animation/systems` directory and integrates skeletal animation with gameplay, physics, and rendering.

**Total Lines:** ~3,225
**Classification:** REAL IMPLEMENTATION (100%)
**Dependency:** Phase 1 (Core Skeletal Animation)

---

## 1. Scope

### 1.1 Included

- ECS skinning system
- Inverse Kinematics solvers (Two-Bone, FABRIK, CCD)
- Procedural animation controllers (Spring, LookAt, Sway, Breathing)
- Facial animation with expressions and lip sync
- Motion matching database and search
- Animation graph state machines
- Crowd animation integration

### 1.2 Dependencies

| Dependency | Purpose |
|------------|---------|
| Phase 1 (skeletal/) | Skeleton, Pose, Clip, Skinning |
| `engine.core.math` | Vec3, Quat, Mat4, Transform |
| `engine.core.ecs` | Entity, World, Component, System |
| `engine.animation.config` | Configuration dataclasses |
| `engine.animation.crowds.*` | Crowd rendering integration |

---

## 2. Module Architecture

```
systems/
├── skinning_system.py        [388 lines]  - ECS skinning pipeline
├── ik_system.py              [503 lines]  - IK solvers
├── procedural_system.py      [518 lines]  - Secondary motion
├── facial_system.py          [495 lines]  - Face animation
├── motion_matching_system.py [483 lines]  - Motion matching
├── animation_graph_system.py [409 lines]  - State machines
└── crowd_system.py           [343 lines]  - Crowd integration
```

---

## 3. System Architectures

### 3.1 Skinning System (`skinning_system.py`)

**Purpose:** Convert poses to GPU-ready skinning matrices for all animated entities.

```
SkinningComponent
├── skeleton: Skeleton
├── current_pose: Pose
├── skinning_method: SkinningMethod
└── skinning_matrices: List[Mat4]

SkinningSystem
├── query: entities with SkinningComponent
└── process(world, dt):
    for entity in query:
        pose = entity.current_pose
        if method == LBS:
            matrices = compute_lbs_matrices(skeleton, pose)
        else:
            matrices = compute_dqs_matrices(skeleton, pose)
        entity.skinning_matrices = matrices
        compute_bounding_box(entity)
```

**Output:** Skinning matrices ready for GPU upload.

### 3.2 IK System (`ik_system.py`)

**Purpose:** Adjust poses to reach procedural targets (foot placement, look-at, hand IK).

```
IKSolverType (enum)
├── TWO_BONE
├── FABRIK
└── CCD

IKTarget
├── target_position: Vec3
├── pole_vector: Optional[Vec3]
├── weight: float
└── chain: List[int]  # bone indices

IKComponent
├── solver_type: IKSolverType
├── targets: List[IKTarget]
└── max_iterations: int

IKSystem
├── query: entities with IKComponent
└── process(world, dt):
    for entity in query:
        pose = entity.current_pose
        for target in entity.targets:
            if solver == TWO_BONE:
                solve_two_bone(pose, target)
            elif solver == FABRIK:
                solve_fabrik(pose, target, max_iter)
            else:
                solve_ccd(pose, target, max_iter)
```

**Key Algorithms:**

**Two-Bone IK (lines 206-314):**
```
# Law of cosines for elbow angle
a = upper_length
b = lower_length
c = target_distance (clamped to a+b)
angle = acos((a² + b² - c²) / (2ab))

# Pole vector for plane orientation
plane_normal = cross(target_dir, pole_dir)
```

**FABRIK (lines 316-378):**
```
for iteration in range(max_iterations):
    # Forward pass: end to root
    positions[-1] = target
    for i in range(len-2, -1, -1):
        dir = normalize(positions[i] - positions[i+1])
        positions[i] = positions[i+1] + dir * bone_lengths[i]
    
    # Backward pass: root to end
    positions[0] = root_position
    for i in range(1, len):
        dir = normalize(positions[i] - positions[i-1])
        positions[i] = positions[i-1] + dir * bone_lengths[i-1]
    
    if distance(positions[-1], target) < tolerance:
        break
```

**CCD (lines 380-445):**
```
for iteration in range(max_iterations):
    for bone in reversed(chain[:-1]):
        to_effector = effector_pos - bone_world_pos
        to_target = target - bone_world_pos
        axis = cross(to_effector, to_target)
        angle = acos(clamp(dot(normalize(to_effector), normalize(to_target)), -1, 1))
        bone.rotation = rotate_around_axis(axis, angle) * bone.rotation
```

### 3.3 Procedural System (`procedural_system.py`)

**Purpose:** Add secondary motion that responds to physics/gameplay.

```
ProceduralController (base)
└── update(dt, pose, entity) -> None

SpringController
├── stiffness: float
├── damping: float
├── target_bone: int
├── anchor_bone: int
├── current_velocity: Vec3
└── update(dt, pose, entity):
    # Hooke's law
    displacement = current - target
    force = -stiffness * displacement - damping * velocity
    velocity += force * dt
    position += velocity * dt
    # Stretch limiting
    if length(displacement) > max_stretch:
        position = clamp_to_max_stretch()
        velocity *= stretch_damping

LookAtController
├── eye_bone: int
├── target_entity: Optional[Entity]
├── angle_limits: (min_yaw, max_yaw, min_pitch, max_pitch)
└── update(dt, pose, entity):
    direction = normalize(target_pos - eye_pos)
    yaw = atan2(direction.x, direction.z)
    pitch = asin(direction.y)
    # Clamp to limits
    yaw = clamp(yaw, min_yaw, max_yaw)
    pitch = clamp(pitch, min_pitch, max_pitch)
    pose[eye_bone].rotation = quat_from_yaw_pitch(yaw, pitch)

SwayController
├── bones: List[int]
├── frequency: float
├── amplitude: float
├── noise_seed: int

BreathingController
├── chest_bone: int
├── belly_bone: int
├── breath_rate: float  # breaths per minute
├── inhale_ratio: float
```

### 3.4 Facial System (`facial_system.py`)

**Purpose:** Drive facial expressions, lip sync, and eye movement.

```
EmotionType (enum)
├── HAPPY
├── SAD
├── ANGRY
├── SURPRISED
└── FEARFUL

FacialComponent
├── blend_shapes: Dict[str, float]
├── current_emotion: Optional[EmotionType]
├── emotion_intensity: float
├── phoneme_queue: List[Phoneme]
├── eye_target: Optional[Vec3]
├── blink_timer: float
└── saccade_timer: float

FacialSystem
└── process(world, dt):
    # Emotion expressions
    apply_emotion_blend_shapes(entity, emotion, intensity)
    
    # Lip sync
    current_phoneme = phoneme_queue.front()
    target_viseme = phoneme_to_viseme(current_phoneme)
    crossfade_to_viseme(entity, target_viseme, dt)
    rotate_jaw_bone(entity, phoneme_category)
    
    # Eye tracking
    look_direction = normalize(eye_target - eye_position)
    apply_eye_rotation(entity, look_direction)
    
    # Blinking
    if blink_timer <= 0:
        trigger_blink(entity)
        blink_timer = random_blink_interval()
    
    # Saccades (small eye movements)
    if saccade_timer <= 0:
        apply_micro_saccade(entity)
        saccade_timer = random_saccade_interval()
```

**Key Algorithm:** Phoneme Transition Blending (lines 338-377)
```
# Smooth crossfade between viseme shapes
for shape_name, target_weight in target_viseme.items():
    current_weight = entity.blend_shapes[shape_name]
    new_weight = lerp(current_weight, target_weight, blend_speed * dt)
    entity.blend_shapes[shape_name] = new_weight

# Jaw bone rotation based on phoneme category
if phoneme.is_open:
    jaw_angle = -15 degrees * audio_intensity
else:
    jaw_angle = -5 degrees * audio_intensity
```

### 3.5 Motion Matching System (`motion_matching_system.py`)

**Purpose:** Select animation clips based on gameplay state using feature matching.

```
MotionFeature
├── trajectory: List[Vec3]  # future positions
├── velocity: Vec3
├── facing_direction: Vec3
└── foot_positions: (Vec3, Vec3)

MotionDatabase
├── clips: List[AnimationClip]
├── features: List[MotionFeature]  # one per frame of all clips
├── feature_weights: Dict[str, float]
└── methods:
    ├── build_from_clips(clips)
    ├── search(query_feature) -> (clip_index, frame_index)
    └── compute_distance(a, b) -> float

MotionMatchingComponent
├── database: MotionDatabase
├── current_clip: int
├── current_frame: int
├── query_feature: MotionFeature
└── continuation_threshold: float

MotionMatchingSystem
└── process(world, dt):
    for entity in query:
        # Build query from gameplay state
        query = MotionFeature(
            trajectory=predict_trajectory(entity, lookahead=1.0),
            velocity=entity.velocity,
            facing=entity.facing,
            feet=entity.foot_positions
        )
        
        # Search database
        best_match = database.search(query)
        
        # Hysteresis: only switch if improvement > threshold
        current_cost = database.compute_distance(query, current_feature)
        best_cost = database.compute_distance(query, best_match)
        if best_cost < current_cost - continuation_threshold:
            transition_to(best_match)
```

**Key Algorithm:** Motion Matching Search (lines 316-338)
```
def search(query: MotionFeature) -> Match:
    best_cost = infinity
    best_match = None
    
    for i, feature in enumerate(features):
        cost = 0
        # Weighted trajectory distance
        for j, (q, f) in enumerate(zip(query.trajectory, feature.trajectory)):
            cost += weights['trajectory'] * distance(q, f)
        # Weighted velocity distance
        cost += weights['velocity'] * distance(query.velocity, feature.velocity)
        # Weighted facing distance
        cost += weights['facing'] * angle_between(query.facing, feature.facing)
        
        if cost < best_cost:
            best_cost = cost
            best_match = Match(clip=i // frames_per_clip, frame=i % frames_per_clip)
    
    return best_match
```

### 3.6 Animation Graph System (`animation_graph_system.py`)

**Purpose:** Execute state machine logic for animation control.

```
GraphParameter
├── name: str
├── type: Type (float, int, bool, trigger)
└── value: Any

StateTransition
├── from_state: str
├── to_state: str
├── conditions: List[Condition]
├── exit_time: Optional[float]
├── blend_duration: float
└── can_interrupt: bool

AnimationState
├── name: str
├── clip: AnimationClip
├── transitions: List[StateTransition]
├── speed_multiplier: float
└── loop: bool

AnimationGraph
├── states: Dict[str, AnimationState]
├── parameters: Dict[str, GraphParameter]
├── entry_state: str
└── any_state_transitions: List[StateTransition]

AnimationGraphComponent
├── graph: AnimationGraph
├── current_state: str
├── current_time: float
├── transition: Optional[ActiveTransition]
└── parameter_values: Dict[str, Any]

AnimationGraphSystem
└── process(world, dt):
    for entity in query:
        # Check any-state transitions first
        for trans in graph.any_state_transitions:
            if evaluate_conditions(trans, params):
                start_transition(entity, trans)
                break
        
        # Check current state transitions
        state = graph.states[current_state]
        for trans in state.transitions:
            if evaluate_conditions(trans, params):
                start_transition(entity, trans)
                break
        
        # Update transition blend if active
        if entity.transition:
            t = entity.transition.elapsed / entity.transition.duration
            entity.pose = blend(from_pose, to_pose, t)
            if t >= 1.0:
                complete_transition(entity)
        
        # Sample current clip
        entity.current_time += dt * state.speed_multiplier
        entity.pose = sample_clip(state.clip, entity.current_time)
```

### 3.7 Crowd System (`crowd_system.py`)

**Purpose:** Batch animation processing for large numbers of entities.

```
CrowdInstance
├── entity: Entity
├── animation_clip: int
├── animation_time: float
├── lod_level: int
└── formation_offset: Vec3

CrowdComponent
├── instances: List[CrowdInstance]
├── texture_atlas: AnimationTextureAtlas
├── lod_distances: List[float]
└── formation_type: FormationType (CIRCLE, GRID, RANDOM)

CrowdSystem
└── process(world, dt):
    for crowd in query:
        # Update LOD based on camera distance
        for instance in crowd.instances:
            dist = distance(instance.position, camera.position)
            instance.lod_level = compute_lod(dist, crowd.lod_distances)
        
        # Sync agent positions
        for instance in crowd.instances:
            agent = simulation.get_agent(instance.entity)
            instance.position = agent.position
            instance.velocity = agent.velocity
        
        # Batch update animation times
        for instance in crowd.instances:
            clip = atlas.clips[instance.animation_clip]
            instance.animation_time += dt
            if instance.animation_time > clip.duration:
                if clip.loop:
                    instance.animation_time %= clip.duration
                else:
                    trigger_event('animation_complete', instance)
        
        # Prepare GPU data
        upload_instance_data(crowd.instances, atlas)
```

---

## 4. System Execution Order

Systems must execute in dependency order:

```
1. MotionMatchingSystem    - Select clips based on gameplay
2. AnimationGraphSystem    - Execute state machine
3. [ClipPlayer sampling]   - Sample animation clips (from Phase 1)
4. IKSystem                - Adjust for procedural targets
5. ProceduralSystem        - Add secondary motion
6. FacialSystem            - Update face
7. SkinningSystem          - Convert to GPU matrices
8. CrowdSystem             - Batch processing for crowds
```

---

## 5. Integration Points

### 5.1 Inputs

| Source | Data | System |
|--------|------|--------|
| Gameplay | Player input | MotionMatchingSystem |
| AI | Behavior state | AnimationGraphSystem |
| Physics | Contact points | IKSystem |
| Audio | Phonemes | FacialSystem |
| Crowds | Agent state | CrowdSystem |

### 5.2 Outputs

| Target | Data | System |
|--------|------|--------|
| Renderer | Skinning matrices | SkinningSystem |
| Renderer | Crowd instances | CrowdSystem |
| Gameplay | Animation events | AnimationGraphSystem |
