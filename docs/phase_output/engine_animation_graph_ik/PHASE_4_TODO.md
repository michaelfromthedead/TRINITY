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
- [ ] COMCalculator class
- [ ] Per-bone mass configuration
- [ ] Weighted average position calculation
- [ ] Update on pose change
- [ ] Support for partial skeleton COM

---

### T-FB-4.2: Support Polygon

**Priority**: P0 (Critical)
**Estimate**: 2 hours
**Dependencies**: T-FB-4.1

**Acceptance Criteria:**
- [ ] Build support polygon from foot positions
- [ ] Point-in-polygon test (ray casting)
- [ ] Handle horizontal edges
- [ ] Project to ground plane (XZ)

---

### T-FB-4.3: Closest Point on Polygon

**Priority**: P1 (High)
**Estimate**: 2 hours
**Dependencies**: T-FB-4.2

**Acceptance Criteria:**
- [ ] Find closest point on polygon boundary
- [ ] closest_point_on_segment helper
- [ ] Edge iteration
- [ ] Used for COM correction

---

### T-FB-4.4: Balance Controller

**Priority**: P0 (Critical)
**Estimate**: 3 hours
**Dependencies**: T-FB-4.2, T-FB-4.3

**Acceptance Criteria:**
- [ ] BalanceController class
- [ ] Check if COM is in support polygon
- [ ] Calculate correction vector if outside
- [ ] Apply pelvis/spine adjustment
- [ ] Configurable correction strength

---

### T-FB-4.5: IKChain Definition

**Priority**: P0 (Critical)
**Estimate**: 2 hours
**Dependencies**: Phase 3 solvers

**Acceptance Criteria:**
- [ ] IKChain dataclass
- [ ] Bone references (root, joints, effector)
- [ ] Associated solver (TwoBone, FABRIK, etc.)
- [ ] Chain weight/priority
- [ ] Enable/disable per chain

---

### T-FB-4.6: FullBodyIK Core

**Priority**: P0 (Critical)
**Estimate**: 4 hours
**Dependencies**: T-FB-4.4, T-FB-4.5

**Acceptance Criteria:**
- [ ] FullBodyIK class
- [ ] Multiple chain registration
- [ ] Goal distribution to chains
- [ ] Solve order (legs -> spine -> arms -> head)
- [ ] Balance integration
- [ ] FullBodyResult dataclass

---

### T-FB-4.7: LookAtSolver

**Priority**: P1 (High)
**Estimate**: 3 hours
**Dependencies**: T-FB-4.6

**Acceptance Criteria:**
- [ ] LookAtSolver class
- [ ] Head bone tracking
- [ ] Spine bone distribution
- [ ] Distribution weights configuration
- [ ] Smooth rotation accumulation
- [ ] Handle targets behind character

---

### T-FB-4.8: Pelvis Height Adjustment

**Priority**: P0 (Critical)
**Estimate**: 2 hours
**Dependencies**: Phase 3 TwoBoneIK

**Acceptance Criteria:**
- [ ] Calculate required pelvis drop
- [ ] Safety margin for leg reach
- [ ] Max pelvis drop limit
- [ ] Smooth adjustment over time

---

### T-FB-4.9: Raycast Interface

**Priority**: P0 (Critical)
**Estimate**: 1 hour
**Dependencies**: None

**Acceptance Criteria:**
- [ ] RaycastHit dataclass
- [ ] RaycastCallback type definition
- [ ] hit, position, normal, distance fields
- [ ] Clear documentation for physics integration

---

### T-FB-4.10: FootPlacement Core

**Priority**: P0 (Critical)
**Estimate**: 4 hours
**Dependencies**: T-FB-4.8, T-FB-4.9

**Acceptance Criteria:**
- [ ] FootPlacement class
- [ ] Left/right leg IK
- [ ] Raycast for foot targets
- [ ] Pelvis adjustment integration
- [ ] FootPlacementResult dataclass
- [ ] dt parameter for smoothing

---

### T-FB-4.11: Foot Alignment

**Priority**: P1 (High)
**Estimate**: 2 hours
**Dependencies**: T-FB-4.10

**Acceptance Criteria:**
- [ ] Align foot to terrain normal
- [ ] Toe bone adjustment
- [ ] Ankle rotation limits
- [ ] Smooth alignment transitions

---

### T-FB-4.12: FootPlacementAnimated

**Priority**: P2 (Medium)
**Estimate**: 2 hours
**Dependencies**: T-FB-4.10

**Acceptance Criteria:**
- [ ] FootPlacementAnimated class (extends FootPlacement)
- [ ] Animation curve integration
- [ ] Curves for lift, plant, etc.
- [ ] Blend with IK result

---

### T-FB-4.13: MultiLegFootPlacement

**Priority**: P2 (Medium)
**Estimate**: 3 hours
**Dependencies**: T-FB-4.10

**Acceptance Criteria:**
- [ ] MultiLegFootPlacement class
- [ ] LegConfig dataclass
- [ ] N-leg support (spiders, centaurs)
- [ ] Body adjustment for multiple legs
- [ ] Per-leg ray offsets
- [ ] MultiLegResult dataclass

---

### T-FB-4.14: SyncMarker and MarkerTrack

**Priority**: P1 (High)
**Estimate**: 2 hours
**Dependencies**: Phase 2 complete

**Acceptance Criteria:**
- [ ] SyncMarker dataclass
- [ ] MarkerTrack class
- [ ] get_nearest_marker(time, name=None)
- [ ] Marker lookup by name
- [ ] Handle wrap-around

---

### T-FB-4.15: SyncGroup Core

**Priority**: P1 (High)
**Estimate**: 3 hours
**Dependencies**: T-FB-4.14

**Acceptance Criteria:**
- [ ] SyncGroup class
- [ ] SyncEntry dataclass
- [ ] SyncMode enum (5 modes)
- [ ] update(dt) method
- [ ] Leader selection
- [ ] Entry registration

---

### T-FB-4.16: Normalized Sync Mode

**Priority**: P1 (High)
**Estimate**: 2 hours
**Dependencies**: T-FB-4.15

**Acceptance Criteria:**
- [ ] Match normalized time [0,1]
- [ ] Handle different animation lengths
- [ ] Smooth time adjustment
- [ ] Handle loop wrap

---

### T-FB-4.17: Phase Sync Mode

**Priority**: P1 (High)
**Estimate**: 3 hours
**Dependencies**: T-FB-4.15, T-FB-4.14

**Acceptance Criteria:**
- [ ] Sync via markers
- [ ] Find corresponding markers in follower
- [ ] Calculate offset from leader marker
- [ ] Apply offset to follower
- [ ] Handle missing markers gracefully

---

### T-FB-4.18: EventSynchronizer

**Priority**: P2 (Medium)
**Estimate**: 2 hours
**Dependencies**: T-FB-4.17

**Acceptance Criteria:**
- [ ] EventSynchronizer class
- [ ] Cross-animation event coordination
- [ ] sync_event(event_name) method
- [ ] Event timing alignment

---

### T-FB-4.19: IKLayer

**Priority**: P0 (Critical)
**Estimate**: 3 hours
**Dependencies**: T-FB-4.6, Phase 2 LayerStack

**Acceptance Criteria:**
- [ ] IKLayer extends AnimationLayer
- [ ] IK solver reference
- [ ] Goal update from context
- [ ] Apply IK to pose
- [ ] Blend mode support

---

### T-FB-4.20: Graph+IK Integration

**Priority**: P0 (Critical)
**Estimate**: 3 hours
**Dependencies**: T-FB-4.19

**Acceptance Criteria:**
- [ ] AnimationGraph + IKLayer composition
- [ ] Goal sources (context, components)
- [ ] Solve order management
- [ ] Result combination

---

### T-FB-4.21: ECS Components

**Priority**: P1 (High)
**Estimate**: 2 hours
**Dependencies**: T-FB-4.20

**Acceptance Criteria:**
- [ ] FullBodyIKController component
- [ ] AnimationGraphController component
- [ ] Look-at target field
- [ ] Foot placement reference
- [ ] Trinity @component decorators

---

### T-FB-4.22: ECS Systems

**Priority**: P1 (High)
**Estimate**: 3 hours
**Dependencies**: T-FB-4.21

**Acceptance Criteria:**
- [ ] AnimationGraphSystem (phase: animation)
- [ ] FullBodyIKSystem (phase: animation_late)
- [ ] Update order: graph -> foot -> full body
- [ ] Result composition
- [ ] Trinity @system decorators

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
