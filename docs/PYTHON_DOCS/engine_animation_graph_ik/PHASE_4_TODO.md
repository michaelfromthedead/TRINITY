# PHASE 4 TODO: Full Body IK and Integration

**Phase**: 4 of 4
**Focus**: Full Body IK, Foot Placement, Synchronization, Integration

---

## Tasks

### T-FB-4.1: COMCalculator

**Priority**: P0 (Critical)
**Estimate**: 2 hours
**Dependencies**: Phase 3 complete

**Acceptance Criteria:**
- [x] COMCalculator class
- [x] Per-bone mass configuration
- [x] Weighted average position calculation
- [x] Update on pose change
- [x] Support for partial skeleton COM

**SDLC Status:** GREEN_LIGHT (2026-06-03)
- Tests: 121 pass (70 whitebox + 51 blackbox)
- Implementation: fullbody.py:39-175
- Methods: set_bone_mass, calculate, calculate_partial, calculate_from_transforms

---

### T-FB-4.2: Support Polygon

**Priority**: P0 (Critical)
**Estimate**: 2 hours
**Dependencies**: T-FB-4.1

**Acceptance Criteria:**
- [x] Build support polygon from foot positions
- [x] Point-in-polygon test (ray casting)
- [x] Handle horizontal edges
- [x] Project to ground plane (XZ)

**SDLC Status:** GREEN_LIGHT (2026-06-03)
- Tests: 94 pass (44 whitebox + 50 blackbox)
- Implementation: fullbody.py:201-275
- Methods: from_foot_positions, contains_point, project_to_ground

---

### T-FB-4.3: Closest Point on Polygon

**Priority**: P1 (High)
**Estimate**: 2 hours
**Dependencies**: T-FB-4.2

**Acceptance Criteria:**
- [x] Find closest point on polygon boundary
- [x] closest_point_on_segment helper
- [x] Edge iteration
- [x] Used for COM correction

**SDLC Status:** GREEN_LIGHT (2026-06-03)
- Tests: 102 pass (57 whitebox + 45 blackbox)
- Implementation: fullbody.py:284-375
- Methods: closest_point_on_segment, closest_point_on_boundary, correction_vector

---

### T-FB-4.4: Balance Controller

**Priority**: P0 (Critical)
**Estimate**: 3 hours
**Dependencies**: T-FB-4.2, T-FB-4.3

**Acceptance Criteria:**
- [x] BalanceController class
- [x] Check if COM is in support polygon
- [x] Calculate correction vector if outside
- [x] Apply pelvis/spine adjustment
- [x] Configurable correction strength

**SDLC Status:** GREEN_LIGHT (2026-06-03)
- Tests: 99 pass (61 whitebox + 38 blackbox)
- Implementation: fullbody.py:379-479
- Methods: is_balanced, get_correction, apply_correction, set_correction_strength, update_support_polygon

---

### T-FB-4.5: IKChain Definition

**Priority**: P0 (Critical)
**Estimate**: 2 hours
**Dependencies**: Phase 3 solvers

**Acceptance Criteria:**
- [x] IKChain dataclass
- [x] Bone references (root, joints, effector)
- [x] Associated solver (TwoBone, FABRIK, etc.)
- [x] Chain weight/priority
- [x] Enable/disable per chain

**SDLC Status:** GREEN_LIGHT (2026-06-03)
- Tests: 195 pass (105 whitebox + 90 blackbox)
- Implementation: fullbody.py:502-600
- Classes: IKSolverType enum, IKChain dataclass
- Methods: bone_count, all_bones, set_weight, set_enabled, arm_chain, leg_chain

---

### T-FB-4.6: FullBodyIK Core

**Priority**: P0 (Critical)
**Estimate**: 4 hours
**Dependencies**: T-FB-4.4, T-FB-4.5

**Acceptance Criteria:**
- [x] FullBodyIK class
- [x] Multiple chain registration
- [x] Goal distribution to chains
- [x] Solve order (legs -> spine -> arms -> head)
- [x] Balance integration
- [x] FullBodyResult dataclass

**SDLC Status:** GREEN_LIGHT (2026-06-03)
- Tests: 222 pass (121 whitebox + 101 blackbox)
- Implementation: fullbody.py:657-943
- Classes: FullBodyIKGoal, FullBodyIKResult, FullBodyIK
- Methods: solve, _adjust_pelvis_height, _solve_spine, _solve_goal, _maintain_balance

---

### T-FB-4.7: LookAtSolver

**Priority**: P1 (High)
**Estimate**: 3 hours
**Dependencies**: T-FB-4.6

**Acceptance Criteria:**
- [x] LookAtSolver class
- [x] Head bone tracking
- [x] Spine bone distribution
- [x] Distribution weights configuration
- [x] Smooth rotation accumulation
- [x] Handle targets behind character

**SDLC Status:** GREEN_LIGHT (2026-06-03)
- Tests: 154 pass (83 whitebox + 71 blackbox)
- Implementation: fullbody.py:1191-1320
- Methods: solve, _rotation_between, _scale_rotation, _quat_to_axis_angle

---

### T-FB-4.8: Pelvis Height Adjustment

**Priority**: P0 (Critical)
**Estimate**: 2 hours
**Dependencies**: Phase 3 TwoBoneIK

**Acceptance Criteria:**
- [x] Calculate required pelvis drop
- [x] Safety margin for leg reach
- [x] Max pelvis drop limit
- [x] Smooth adjustment over time

**SDLC Status:** GREEN_LIGHT (2026-06-03)
- Tests: 168 pass (94 whitebox + 74 blackbox)
- Implementation: fullbody.py:1191-1431
- Classes: PelvisAdjustmentConfig, PelvisHeightAdjuster
- Methods: calculate_required_drop, adjust, reset, set_config, get_target_offset

---

### T-FB-4.9: Raycast Interface

**Priority**: P0 (Critical)
**Estimate**: 1 hour
**Dependencies**: None

**Acceptance Criteria:**
- [x] RaycastHit dataclass
- [x] RaycastCallback type definition
- [x] hit, position, normal, distance fields
- [x] Clear documentation for physics integration

**SDLC Status:** GREEN_LIGHT (2026-06-03)
- Tests: 130 pass (74 whitebox + 56 blackbox)
- Implementation: foot_placement.py:34-72
- Classes: RaycastHit dataclass, RaycastCallback type alias
- Methods: miss() static factory

---

### T-FB-4.10: FootPlacement Core

**Priority**: P0 (Critical)
**Estimate**: 4 hours
**Dependencies**: T-FB-4.8, T-FB-4.9

**Acceptance Criteria:**
- [x] FootPlacement class
- [x] Left/right leg IK
- [x] Raycast for foot targets
- [x] Pelvis adjustment integration
- [x] FootPlacementResult dataclass
- [x] dt parameter for smoothing

**SDLC Status:** GREEN_LIGHT (2026-06-03)
- Tests: 180 pass (103 whitebox + 77 blackbox)
- Implementation: foot_placement.py:120-550
- Classes: FootPlacement, FootPlacementResult, FootData, FootState
- Methods: solve, set_raycast_callback, _raycast_foot, _calculate_pelvis_offset

---

### T-FB-4.11: Foot Alignment

**Priority**: P1 (High)
**Estimate**: 2 hours
**Dependencies**: T-FB-4.10

**Acceptance Criteria:**
- [x] Align foot to terrain normal
- [x] Toe bone adjustment
- [x] Ankle rotation limits (via weighted blending)
- [x] Smooth alignment transitions

**SDLC Status:** GREEN_LIGHT (2026-06-03)
- Tests: 142 pass (77 whitebox + 65 blackbox)
- Implementation: foot_placement.py:487-552
- Methods: _align_foot_to_terrain, _rotation_between_vectors, _scale_rotation

---

### T-FB-4.12: FootPlacementAnimated

**Priority**: P2 (Medium)
**Estimate**: 2 hours
**Dependencies**: T-FB-4.10

**Acceptance Criteria:**
- [x] FootPlacementAnimated class (wraps FootPlacement)
- [x] Animation curve integration
- [x] Curves for lift, plant, etc.
- [x] Blend with IK result

**SDLC Status:** GREEN_LIGHT (2026-06-03)
- Tests: 136 pass (81 whitebox + 55 blackbox)
- Implementation: foot_placement.py:555-622
- Methods: set_height_curves, update, solve

---

### T-FB-4.13: MultiLegFootPlacement

**Priority**: P2 (Medium)
**Estimate**: 3 hours
**Dependencies**: T-FB-4.10

**Acceptance Criteria:**
- [x] MultiLegFootPlacement class
- [x] LegConfig dataclass (uses FootData)
- [x] N-leg support (spiders, centaurs)
- [x] Body adjustment for multiple legs
- [x] Per-leg ray offsets (via FootData)
- [x] MultiLegResult dataclass (returns List[Transform])

**SDLC Status:** GREEN_LIGHT (2026-06-03)
- Tests: 154 pass (90 whitebox + 64 blackbox)
- Implementation: foot_placement.py:624-720
- Methods: solve, _calculate_multi_pelvis_offset

---

### T-FB-4.14: SyncMarker and MarkerTrack

**Priority**: P1 (High)
**Estimate**: 2 hours
**Dependencies**: Phase 2 complete

**Acceptance Criteria:**
- [x] SyncMarker dataclass
- [x] MarkerTrack class (SyncMarkerTrack)
- [x] get_nearest_marker(time, name=None)
- [x] Marker lookup by name
- [x] Handle wrap-around

**SDLC Status:** GREEN_LIGHT (2026-06-03)
- Tests: 179 pass (95 whitebox + 84 blackbox)
- Implementation: sync.py:46-115
- Classes: SyncMarker, SyncMarkerTrack
- Methods: add_marker, get_markers_by_name, get_nearest_marker, get_markers_in_range

---

### T-FB-4.15: SyncGroup Core

**Priority**: P1 (High)
**Estimate**: 3 hours
**Dependencies**: T-FB-4.14

**Acceptance Criteria:**
- [x] SyncGroup class
- [x] SyncEntry dataclass
- [x] SyncMode enum (5 modes)
- [x] update(dt) method
- [x] Leader selection
- [x] Entry registration

**SDLC Status:** GREEN_LIGHT (2026-06-03)
- Tests: 227 pass (130 whitebox + 97 blackbox)
- Implementation: sync.py:123-280
- Classes: SyncMode, SyncEntry, SyncGroup
- Methods: add_entry, remove_entry, set_leader, get_leader, update

---

### T-FB-4.16: Normalized Sync Mode

**Priority**: P1 (High)
**Estimate**: 2 hours
**Dependencies**: T-FB-4.15

**Acceptance Criteria:**
- [x] Match normalized time [0,1]
- [x] Handle different animation lengths
- [x] Smooth time adjustment
- [x] Handle loop wrap

**SDLC Status:** GREEN_LIGHT (2026-06-03)
- Tests: Covered by T-FB-4.15 SyncGroup tests (227 tests)
- Implementation: sync.py:258-277 (_sync_normalized)
- Tested via SyncMode.NORMALIZED in SyncGroup.update()

---

### T-FB-4.17: Phase Sync Mode

**Priority**: P1 (High)
**Estimate**: 3 hours
**Dependencies**: T-FB-4.15, T-FB-4.14

**Acceptance Criteria:**
- [x] Sync via markers
- [x] Find corresponding markers in follower
- [x] Calculate offset from leader marker
- [x] Apply offset to follower
- [x] Handle missing markers gracefully

**SDLC Status:** GREEN_LIGHT (2026-06-03)
- Tests: Covered by T-FB-4.15 SyncGroup tests (227 tests)
- Implementation: sync.py:279-315 (_sync_phase)
- Tested via SyncMode.PHASE in SyncGroup.update()

---

### T-FB-4.18: EventSynchronizer

**Priority**: P2 (Medium)
**Estimate**: 2 hours
**Dependencies**: T-FB-4.17

**Acceptance Criteria:**
- [x] EventSynchronizer class
- [x] Cross-animation event coordination
- [x] sync_event(event_name) method (queue_event + process_events)
- [x] Event timing alignment

**SDLC Status:** GREEN_LIGHT (2026-06-03)
- Tests: 151 pass (81 whitebox + 70 blackbox)
- Implementation: sync.py:583-650
- Classes: SyncEvent, EventSynchronizer
- Methods: register_handler, queue_event, process_events

---

### T-FB-4.19: IKLayer

**Priority**: P0 (Critical)
**Estimate**: 3 hours
**Dependencies**: T-FB-4.6, Phase 2 LayerStack

**Acceptance Criteria:**
- [x] IKLayer as standalone layer class
- [x] IK solver reference (FullBodyIK, FootPlacement, TwoBoneIK, FABRIKChain)
- [x] Goal update from context (IKGoalContext)
- [x] Apply IK to pose
- [x] Blend mode support (OVERRIDE, ADDITIVE, BLEND)

**SDLC Status:** GREEN_LIGHT (2026-06-03)
- Tests: 210 pass (151 whitebox + 59 blackbox)
- Implementation: ik_layer.py
- Classes: IKBlendMode, IKGoalContext, IKLayer, IKLayerStack, IKLayerResult

---

### T-FB-4.20: Graph+IK Integration

**Priority**: P0 (Critical)
**Estimate**: 3 hours
**Dependencies**: T-FB-4.19

**Acceptance Criteria:**
- [x] AnimationGraph + IKLayer composition
- [x] Goal sources (context, components)
- [x] Solve order management
- [x] Result combination

**SDLC Status:** GREEN_LIGHT (2026-06-03)
- Tests: 216 pass (127 whitebox + 89 blackbox)
- Implementation: graph_integration.py
- Classes: IKSolveOrder, IKGoalSource, AnimationIKController, AnimationIKResult
- Goal Sources: ComponentGoalSource, CallbackGoalSource, StaticGoalSource

---

### T-FB-4.21: ECS Components

**Priority**: P1 (High)
**Estimate**: 2 hours
**Dependencies**: T-FB-4.20

**Acceptance Criteria:**
- [x] FullBodyIKController component
- [x] AnimationGraphController component
- [x] Look-at target field (LookAtTarget component)
- [x] Foot placement reference (FootPlacementController)
- [x] Trinity @component decorators

**SDLC Status:** GREEN_LIGHT (2026-06-03)
- Tests: 230 pass (168 whitebox + 62 blackbox)
- Implementation: ecs_components.py
- Components: FullBodyIKController, AnimationGraphController, LookAtTarget, FootPlacementController, IKTargetComponent

---

### T-FB-4.22: ECS Systems

**Priority**: P1 (High)
**Estimate**: 3 hours
**Dependencies**: T-FB-4.21

**Acceptance Criteria:**
- [x] AnimationGraphSystem (phase: animation)
- [x] FullBodyIKSystem (phase: animation_late)
- [x] Update order: graph -> foot -> full body
- [x] Result composition
- [x] Trinity @system decorators

**SDLC Status:** GREEN_LIGHT (2026-06-03)
- Tests: 187 pass (122 whitebox + 65 blackbox)
- Implementation: ecs_systems.py
- Systems: AnimationGraphIKSystem, FootPlacementSystem, FullBodyIKSystem, LookAtSystem, AnimationIKCompositeSystem
- Helpers: register_animation_ik_systems(), register_composite_system()

---

## 🏆 PHASE 4 COMPLETE

All 22 tasks at GREEN_LIGHT. Total: ~3097 tests.

---

## Task Summary

| Task ID | Description | Priority | Est. Hours | Dependencies |
|---------|-------------|----------|------------|--------------|
| T-FB-4.1 | COMCalculator | P0 | 2 | Phase 3 |
| T-FB-4.2 | Support Polygon | P0 | 2 | T-FB-4.1 |
| T-FB-4.3 | Closest Point | P1 | 2 | T-FB-4.2 |
| T-FB-4.4 | Balance Controller | P0 | 3 | T-FB-4.2, T-FB-4.3 |
| T-FB-4.5 | IKChain Definition | P0 | 2 | Phase 3 |
| T-FB-4.6 | FullBodyIK Core | P0 | 4 | T-FB-4.4, T-FB-4.5 |
| T-FB-4.7 | LookAtSolver | P1 | 3 | T-FB-4.6 |
| T-FB-4.8 | Pelvis Adjustment | P0 | 2 | Phase 3 |
| T-FB-4.9 | Raycast Interface | P0 | 1 | None |
| T-FB-4.10 | FootPlacement Core | P0 | 4 | T-FB-4.8, T-FB-4.9 |
| T-FB-4.11 | Foot Alignment | P1 | 2 | T-FB-4.10 |
| T-FB-4.12 | FootPlacementAnimated | P2 | 2 | T-FB-4.10 |
| T-FB-4.13 | MultiLegFootPlacement | P2 | 3 | T-FB-4.10 |
| T-FB-4.14 | SyncMarker/Track | P1 | 2 | Phase 2 |
| T-FB-4.15 | SyncGroup Core | P1 | 3 | T-FB-4.14 |
| T-FB-4.16 | Normalized Sync | P1 | 2 | T-FB-4.15 |
| T-FB-4.17 | Phase Sync | P1 | 3 | T-FB-4.15, T-FB-4.14 |
| T-FB-4.18 | EventSynchronizer | P2 | 2 | T-FB-4.17 |
| T-FB-4.19 | IKLayer | P0 | 3 | T-FB-4.6, Phase 2 |
| T-FB-4.20 | Graph+IK Integration | P0 | 3 | T-FB-4.19 |
| T-FB-4.21 | ECS Components | P1 | 2 | T-FB-4.20 |
| T-FB-4.22 | ECS Systems | P1 | 3 | T-FB-4.21 |

**Total Estimate**: 56 hours

---

## Verification Checklist

After Phase 4 completion:

- [ ] COM calculation correct
- [ ] Balance detection works
- [ ] Full body IK solves
- [ ] LookAt distributes across spine
- [ ] Foot placement adapts to terrain
- [ ] Pelvis adjusts correctly
- [ ] Multi-leg placement works
- [ ] All sync modes work
- [ ] IK layers apply correctly
- [ ] ECS integration works
- [ ] All tests pass

---

## Phase Summary

| Phase | Focus | Est. Hours |
|-------|-------|------------|
| 1 | Core Animation Graph | 21 |
| 2 | State Machines & Blend Trees | 46 |
| 3 | IK Solvers | 53 |
| 4 | Full Body IK & Integration | 56 |
| **Total** | | **176 hours** |
