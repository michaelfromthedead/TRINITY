# Investigation: engine/animation/systems

## Summary
The animation systems directory contains a comprehensive set of ECS-integrated animation systems with real implementations. These include animation state machines with transitions, multiple IK solvers (Two-Bone, FABRIK, CCD), procedural animation controllers (spring, look-at, sway, breathing), motion matching with database search, facial animation with lip sync/emotions/eye tracking, and crowd simulation with LOD/culling. All systems follow consistent ECS patterns with World and Entity integration, typed components, and proper update loops.

## Files
| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `__init__.py` | 87 | COMPLETE | Exports all systems and components |
| `animation_graph_system.py` | 410 | COMPLETE | State machine, transitions, pose blending |
| `ik_system.py` | 504 | COMPLETE | 3 IK solvers with full math |
| `procedural_system.py` | 519 | COMPLETE | 4 controller types with physics |
| `motion_matching_system.py` | 484 | COMPLETE | Database search, feature extraction |
| `facial_system.py` | 496 | COMPLETE | Lip sync, emotions, eye tracking |
| `skinning_system.py` | 389 | COMPLETE | LBS and dual quaternion skinning |
| `crowd_system.py` | 344 | COMPLETE | LOD, culling, formations |

**Total: ~3,233 lines of implementation**

## System Components

### ECS Integration
- All systems take `World` and `entity_components: list[tuple[Entity, Component]]`
- Components are dataclasses with proper defaults
- Systems follow consistent `update(world, dt, entity_components)` pattern

### Update Loops
- `AnimationGraphSystem.update()` - state machine evaluation, transitions
- `IKSystem.update()` - returns modified pose data
- `ProceduralSystem.update()` - chains controllers, merges modifications
- `MotionMatchingSystem.update()` - database search, pose blending
- `FacialSystem.update()` - emotion/lip-sync/eye updates
- `SkinningSystem.update()` - computes skinning matrices, transforms vertices
- `CrowdSystem.update()` - fixed-rate simulation, LOD, culling

### Component Types
- `AnimationGraphComponent` - graph instance, output pose, parameter bindings
- `IKComponent` - goals list, blend to animation
- `ProceduralComponent` - controllers list
- `MotionMatchingComponent` - controller, input provider, output pose
- `FacialComponent` - face rig, emotion, lip sync, eye state
- `SkinnedMeshComponent` - mesh data, skinning data, method
- `CrowdComponent` - simulator, renderer, LOD

## Implementation

- Real ECS integration? **YES** - All systems use World/Entity, take entity_components list
- Real update loops? **YES** - Full dt-based update with proper state management
- Real component sync? **YES** - Components output poses/matrices, chain between systems

### Key Implementation Details

**Animation Graph System:**
- State machine with states, transitions, conditions
- Transition blending with configurable duration
- Parameter system (float, int, bool, trigger)
- Pose evaluation with animation provider callback

**IK System:**
- Two-Bone solver with law of cosines for limb IK
- FABRIK forward/backward reaching algorithm
- CCD cyclic coordinate descent
- World transform computation from hierarchy
- Blend weight application

**Procedural System:**
- Spring dynamics with stiffness/damping/gravity
- Look-at with angle limits and smoothing
- Sway oscillation with noise
- Breathing with inhale/exhale curves

**Motion Matching System:**
- Feature-based database search
- Trajectory/velocity/direction features
- Cost computation with weights
- Pose blending during transitions
- Database builder from animation clips

**Facial System:**
- Emotion expressions with blend shapes
- Phoneme-based lip sync with transitions
- Eye tracking with saccades
- Blinking with timer/duration
- Audio processing for phoneme detection

**Skinning System:**
- Linear blend skinning (CPU)
- Dual quaternion skinning (reduces artifacts)
- GPU buffer preparation
- Bind pose * world matrix computation

**Crowd System:**
- Fixed-rate simulation updates
- Agent-to-instance synchronization
- LOD distance thresholds
- Distance culling
- Formation spawning (circle, grid, random)

## Verdict
**REAL IMPLEMENTATION**

This is a comprehensive, production-quality animation systems layer. All 7 systems have complete implementations with proper algorithms, math, and ECS integration. The code demonstrates knowledge of standard game animation techniques (FABRIK, dual quaternion skinning, motion matching, phoneme lip sync).

## Evidence

**Real IK Math (Two-Bone using law of cosines):**
```python
# ik_system.py:244-247
cos_angle = (upper_length**2 + lower_length**2 - target_dist**2) / (2 * upper_length * lower_length)
cos_angle = max(-1.0, min(1.0, cos_angle))
joint_angle = math.acos(cos_angle)
```

**Real FABRIK Algorithm:**
```python
# ik_system.py:341-358
# Forward pass
positions[0] = target
for i in range(len(positions) - 1):
    direction = (positions[i+1] - positions[i]).normalized()
    positions[i+1] = positions[i] + direction * lengths[i]

# Backward pass
positions[-1] = root
for i in range(len(positions) - 2, -1, -1):
    direction = (positions[i] - positions[i+1]).normalized()
    positions[i] = positions[i+1] + direction * lengths[i]
```

**Real Spring Physics:**
```python
# procedural_system.py:106-121
spring_force = displacement * self.stiffness
damping_force = velocity * (-self.damping * self.stiffness)
gravity_force = self.gravity * self.mass
total_force = spring_force + damping_force + gravity_force
acceleration = total_force / self.mass
velocity = velocity + acceleration * dt
new_pos = current_pos + velocity * dt
```

**Real Dual Quaternion Skinning:**
```python
# skinning_system.py:324-339
def _mat4_to_dual_quat(self, mat: Mat4) -> tuple[Quat, Quat]:
    transform = Transform.from_matrix(mat)
    rot = transform.rotation.normalized()
    t = transform.translation
    dual = Quat(
        0.5 * (t.x * rot.w + t.y * rot.z - t.z * rot.y),
        0.5 * (-t.x * rot.z + t.y * rot.w + t.z * rot.x),
        0.5 * (t.x * rot.y - t.y * rot.x + t.z * rot.w),
        -0.5 * (t.x * rot.x + t.y * rot.y + t.z * rot.z),
    )
```

**Real Motion Matching Search:**
```python
# motion_matching_system.py:316-338
def _search_database(self, query: list[float], controller: MotionMatchingController) -> MotionMatchResult:
    for i, frame in enumerate(database.frames):
        cost = self._compute_cost(query, frame.feature_vector, feature_weights, database.features)
        if cost < best_result.cost:
            best_result.frame_index = i
            best_result.cost = cost
```
