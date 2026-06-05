# Phase 4: IK Solver Library -- Architecture

## Status: 6 [x] 0 [~] 1 [-]

## Module: `engine/animation/ik/`

### Files
| File | Lines | Purpose |
|------|-------|---------|
| two_bone.py | 493 | Analytical two-bone IK with swivel control |
| fabrik.py | 615 | FABRIK multi-chain solver |
| ccd.py | 690 | Cyclic Coordinate Descent solver |
| jacobian.py | 691 | Jacobian-based IK (3 methods) |
| fullbody.py | 767 | Multi-effector full body IK |
| ik_goal.py | 568 | Goal types, blending, decorator API |
| foot_placement.py | 736 | Terrain-adaptive foot IK |
| config.py | 155 | Solver tolerances, limits, damping |
| __init__.py | 216 | Public API, re-exports all solvers |

### Architecture

**Solver Hierarchy:**
1. **TwoBoneIK** (analytical, O(1)): Law of cosines, swivel plane control, joint angle limits, singularity at full extension
2. **FABRIK** (iterative): Forward/backward pass, any chain length, joint constraints, multi-chain variant
3. **CCD** (iterative): Rotation per joint toward target, rotation limits, weighted, damped variant
4. **JacobianIK** (numerical): Jacobian transpose, DLS, SVD pseudoinverse, multi-target, null-space projection
5. **FullBodyIK** (composite): Multi-effector (feet/hands/head/hips), CoM balance, posture preservation, priority layering

**IK Goal System** (`ik_goal.py`):
- `IKGoalType`: POSITION, ROTATION, LOOK_AT, POSITION_ROTATION, POLE_VECTOR, CHAIN, CENTER_OF_MASS
- `IKGoal`: component with target, weight, priority, blend_speed
- `IKGoalBlender`: smooth interpolation between goal states
- `@ik_goal`: decorator marking pose-affecting goals
- `@ik_chain`: decorator marking bone chains for IK

**Foot Placement** (`foot_placement.py`):
- `FootState`: SUSPENDED, STANCE, LOCKED, SLIDING, RECOVERING
- `FootData`: current/previous positions, state, blend
- `FootPlacement`: terrain height sampling, ankle roll, toe alignment
- `FootPlacementAnimated`: pre-baked foot animation curves
- `MultiLegFootPlacement`: coordinated multi-foot adjustment with pelvis compensation

**Configuration** (`ik/config.py`):
- Per-solver tolerances and iteration limits
- Damping factors (DLS=0.5, CCD=1.0)
- Soft IK parameters (ratio=0.9, blend=0.5)
- Joint limits (angles, cone constraints)
- Look-at weights (head=0.6, neck=0.3, spine=0.1)

### Missing
- T-AN-4.7: Tests for all 5 IK methods

### Key Design Decisions
- TwoBoneIK is analytical O(1) -- no iteration, ideal for arms/legs
- FullBodyIK uses priority layering: balance > foot placement > hand reach
- Jacobian implements all 3 methods (transpose/DLS/SVD) with fallback
- Foot placement uses state machine (5 states) with smooth transitions
- All solvers use common IKGoal component for uniform API
- Config module centralizes 40+ tunable parameters
