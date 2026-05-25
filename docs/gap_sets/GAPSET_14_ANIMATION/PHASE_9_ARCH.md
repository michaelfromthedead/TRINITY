# Phase 9: Cinematics & Full Engine Integration -- Architecture

## Status: 7 [x] 2 [~] 3 [-]

## Modules: `engine/animation/systems/`, *(missing)* `engine/animation/cinematics/`

### Files
| File | Lines | Purpose |
|------|-------|---------|
| systems/animation_graph_system.py | 409 | Animation graph ECS system |
| systems/ik_system.py | 503 | IK solver ECS system |
| systems/procedural_system.py | 518 | Procedural animation ECS system |
| systems/skinning_system.py | 388 | Skinning ECS system |
| systems/motion_matching_system.py | 483 | Motion matching ECS system |
| systems/facial_system.py | 495 | Facial animation ECS system |
| systems/crowd_system.py | 343 | Crowd animation ECS system |
| *(missing)* cinematics/cutscene.py | 0 | **Does not exist** |
| *(missing)* cinematics/camera_track.py | 0 | **Does not exist** |
| *(missing)* tests/ | 0 | **Does not exist** |

### ECS System Architecture

All 7 systems follow a consistent pattern:
- `*Component` dataclass: per-entity state (graph ref, pose buffer, IK goals, etc.)
- `*System` class: `update(world, dt, entity_components)` method
- Integration via `engine.core.ecs.World` and `Entity`

**Pipeline Order** (defined in `config.py`):
1. AnimationGraphSystem (priority 100): state machine + pose sampling
2. MotionMatchingSystem (priority 150): trajectory computation + search + transition
3. IKSystem (priority 200): goal resolution + solver dispatch
4. ProceduralSystem (priority 300): spring/lookat/sway/breathing
5. FacialSystem (priority 300): blend shapes/FACS/lip sync/eye (parallel to procedural)
6. SkinningSystem (priority 400): LBS/DQS/buffer prep
7. CrowdSystem (priority 500): agent steering + LOD + texture baking

**System Details:**

**AnimationGraphSystem** (`animation_graph_system.py`):
- `AnimationGraphComponent`: graph instance, output pose, parameter bindings
- Evaluates state machine transitions (exit time + conditions)
- Blends poses during transitions
- Samples animations via provider function
- Parameter binding from gameplay properties

**IKSystem** (`ik_system.py`):
- `IKComponent`: IK goals, enabled solvers, chain definitions
- Goal resolution in priority order (balance > foot > hand)
- Dispatches to appropriate solver (two_bone/fabrik/ccd/jacobian/fullbody)
- Outputs adjusted pose to next pipeline stage

**ProceduralSystem** (`procedural_system.py`):
- `ProceduralComponent`: controller instances with ordering
- Spring, look-at, sway, breathing controllers
- Per-bone effect ordering (spring before look-at, etc.)

**SkinningSystem** (`skinning_system.py`):
- `SkinnedMeshComponent`: mesh, skinning data, method, output buffers
- Computes skinning matrices: world * inverse_bind
- CPU path: LBS or DQS vertex transformation
- GPU path: prepare flattened buffer for SSBO upload

**MotionMatchingSystem** (`motion_matching_system.py`):
- `MotionMatchingComponent`: database ref, controller, budget tracking
- Per-frame: trajectory computation -> search -> transition -> pose
- Budget enforcement (motion_matching_budget_ms)
- Fallback to state machine when budget exceeded

**FacialSystem** (`facial_system.py`):
- `FacialComponent`: expression weights, lip sync audio, eye targets
- Orchard FACS -> blend shape weight mapping
- Lip sync -> viseme weight mapping
- Eye saccade/blink/gaze computation

**CrowdSystem** (`crowd_system.py`):
- `CrowdComponent`: agent references, LOD level, animation texture
- Agent steering update (RVO/ORCA-style)
- Animation texture baking on clip change
- LOD selection + frustum culling per agent

### Missing
- T-AN-9.1: `cinematics/cutscene.py` -- timeline, skip, state save/restore
- T-AN-9.2: `cinematics/camera_track.py` -- spline interpolation, look-at targets
- T-AN-9.12: Integration tests

### Partial
- T-AN-9.10: Tracker integration -- manual param bindings exist, Foundation Tracker not wired
- T-AN-9.11: Session persistence -- dataclasses serialization-ready, Foundation Session not wired

### Key Design Decisions
- Systems use consistent component/system pattern for ECS integration
- Pipeline priority numbers reserve room for insertions (gaps between 100, 150, 200, 300, 400, 500)
- Procedural and facial run at same priority (different bone sets, task-parallel)
- AnimationGraphSystem merges simulation + presentation roles (TODO had them split)
- Parameter binding bridges gameplay to animation via string-keyed dictionary
- Skinning system supports 3 methods (LBS/DQS/GPU) with matching component API
