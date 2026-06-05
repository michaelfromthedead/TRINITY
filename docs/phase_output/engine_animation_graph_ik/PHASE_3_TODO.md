# PHASE 3 TODO: IK Solvers

**Phase**: 3 of 4
**Focus**: Inverse Kinematics Algorithms

---

## Tasks

### T-IK-3.1: IK Goal Base Classes

**Priority**: P0 (Critical)
**Estimate**: 2 hours
**Dependencies**: None

**Acceptance Criteria:**
- [ ] IKGoal abstract base class
- [ ] PositionGoal dataclass
- [ ] RotationGoal dataclass
- [ ] LookAtGoal dataclass
- [ ] PositionRotationGoal dataclass
- [ ] PoleVectorGoal dataclass
- [ ] COMGoal dataclass
- [ ] IKGoalBlender for weighted blending

---

### T-IK-3.2: Joint Constraints Base

**Priority**: P0 (Critical)
**Estimate**: 3 hours
**Dependencies**: None

**Acceptance Criteria:**
- [ ] JointConstraint abstract base class
- [ ] apply(direction, parent_rotation) signature
- [ ] BallSocketConstraint with cone angle
- [ ] HingeConstraint with axis and angle limits
- [ ] TwistConstraint with twist limits
- [ ] Numerical stability checks

---

### T-IK-3.3: Joint Limits for CCD

**Priority**: P1 (High)
**Estimate**: 2 hours
**Dependencies**: T-IK-3.2

**Acceptance Criteria:**
- [ ] JointLimit abstract base class
- [ ] clamp(rotation) signature
- [ ] EulerLimit with min/max per axis
- [ ] EulerOrder enum (6 orders)
- [ ] SwingTwistLimit decomposition
- [ ] Proper Euler extraction and reconstruction

---

### T-IK-3.4: Two-Bone IK Core

**Priority**: P0 (Critical)
**Estimate**: 3 hours
**Dependencies**: T-IK-3.1

**Acceptance Criteria:**
- [ ] TwoBoneIK class
- [ ] Law of cosines angle calculation
- [ ] `cos_mid = (a^2 + b^2 - c^2) / (2ab)`
- [ ] Numerical stability clamping
- [ ] solve(root, mid, end, target, pole) method
- [ ] TwoBoneIKResult dataclass

---

### T-IK-3.5: Two-Bone Soft IK ✅ COMPLETE (2026-06-04)

**Priority**: P1 (High)
**Estimate**: 2 hours
**Dependencies**: T-IK-3.4
**Tests Added**: 20 new tests

**Acceptance Criteria:**
- [x] SoftIKConfig dataclass
- [x] Exponential falloff implementation
- [x] `d_soft = d_start + (d_max - d_start) * (1 - e^(-k * overshoot))`
- [x] Smooth blending between hard and soft limits
- [x] Handle unreachable targets gracefully

**Implementation Notes:**
- Added `SoftIkConfig` struct with `enabled`, `soft_start_ratio`, `falloff_rate` fields
- Integrated into `TwoBoneIkParams` with builder methods
- Modified `solve_two_bone_ik_world()` to use soft IK for unreachable targets
- Added dependency: `glam = { version = "0.29", features = ["serde"] }`

---

### T-IK-3.6: Two-Bone Pole Vector ✅ COMPLETE (2026-06-04)

**Priority**: P1 (High)
**Estimate**: 2 hours
**Dependencies**: T-IK-3.4
**Tests Added**: 7 new twist tests

**Acceptance Criteria:**
- [x] Pole vector positioning (pre-existing)
- [x] Plane projection for elbow/knee direction (pre-existing)
- [x] Twist control (NEW: `twist_angle` field, `with_twist()` builder)
- [x] Handle pole vector at singular positions (fallback to perpendicular)

**Implementation Notes:**
- Added `twist_angle` field to `TwoBoneIkParams`
- Added `with_twist()` and `with_twist_degrees()` builder methods
- Twist rotates the IK plane around root-to-target axis
- Improved singular pole vector handling with graceful fallback

---

### T-IK-3.7: FABRIK Chain Core

**Priority**: P0 (Critical)
**Estimate**: 4 hours
**Dependencies**: T-IK-3.2

**Acceptance Criteria:**
- [ ] FABRIKChain class
- [ ] Bone length storage
- [ ] Forward pass (end to root)
- [ ] Backward pass (root to end)
- [ ] Convergence check
- [ ] Max iterations limit
- [ ] FABRIKResult dataclass

---

### T-IK-3.8: FABRIK Constraint Integration

**Priority**: P1 (High)
**Estimate**: 2 hours
**Dependencies**: T-IK-3.7, T-IK-3.2

**Acceptance Criteria:**
- [ ] Per-joint constraint application
- [ ] Constraint application in both passes
- [ ] Ball-socket constraint integration
- [ ] Hinge constraint integration
- [ ] Constraint affects direction preservation

---

### T-IK-3.9: FABRIK Multi-Chain

**Priority**: P1 (High)
**Estimate**: 3 hours
**Dependencies**: T-IK-3.7

**Acceptance Criteria:**
- [ ] FABRIKMultiChain class
- [ ] Shared joint handling
- [ ] Average position at shared joints
- [ ] Multiple target support
- [ ] Chain priority/weighting
- [ ] MultiChainResult dataclass

---

### T-IK-3.10: CCD Solver Core

**Priority**: P0 (Critical)
**Estimate**: 4 hours
**Dependencies**: T-IK-3.3

**Acceptance Criteria:**
- [ ] CCDSolver class
- [ ] Per-joint rotation calculation
- [ ] to-end / to-target vector computation
- [ ] Rotation axis from cross product
- [ ] Angle from acos(dot)
- [ ] Damping factor application
- [ ] CCDResult dataclass

---

### T-IK-3.11: CCD Joint Limit Integration

**Priority**: P1 (High)
**Estimate**: 2 hours
**Dependencies**: T-IK-3.10, T-IK-3.3

**Acceptance Criteria:**
- [ ] JointLimit.clamp() after each rotation
- [ ] Hinge constraint enforcement
- [ ] Euler limit enforcement
- [ ] Proper rotation order handling

---

### T-IK-3.12: CCD Weighted Variant

**Priority**: P1 (High)
**Estimate**: 2 hours
**Dependencies**: T-IK-3.10

**Acceptance Criteria:**
- [ ] CCDSolverWithWeights class
- [ ] Per-joint weight storage
- [ ] Weight affects rotation amount
- [ ] Zero weight = locked joint
- [ ] High weight = more influence

---

### T-IK-3.13: CCD Custom Constraints

**Priority**: P2 (Medium)
**Estimate**: 2 hours
**Dependencies**: T-IK-3.10

**Acceptance Criteria:**
- [ ] ConstrainedCCDSolver class
- [ ] Custom constraint function support
- [ ] Constraint function signature: (rotation, joint_idx) -> rotation
- [ ] Multiple constraint functions per joint

---

### T-IK-3.14: Matrix Class

**Priority**: P0 (Critical)
**Estimate**: 3 hours
**Dependencies**: None

**Acceptance Criteria:**
- [ ] Matrix class for arbitrary dimensions
- [ ] Matrix multiplication
- [ ] Matrix transpose
- [ ] Gauss-Jordan inversion
- [ ] Identity matrix creation
- [ ] Column/row access
- [ ] Numerical stability handling

---

### T-IK-3.15: Jacobian Computation

**Priority**: P0 (Critical)
**Estimate**: 3 hours
**Dependencies**: T-IK-3.14

**Acceptance Criteria:**
- [ ] compute_jacobian() method
- [ ] `J_col = axis x (end_effector - joint_pos)`
- [ ] Multi-DOF joints support
- [ ] Rotation axis per joint
- [ ] Jacobian matrix construction

---

### T-IK-3.16: Jacobian Transpose Method

**Priority**: P1 (High)
**Estimate**: 2 hours
**Dependencies**: T-IK-3.15

**Acceptance Criteria:**
- [ ] `dq = alpha * J^T * e`
- [ ] Step size calculation
- [ ] Convergence handling
- [ ] Fast but may oscillate

---

### T-IK-3.17: Jacobian Pseudoinverse Method

**Priority**: P1 (High)
**Estimate**: 2 hours
**Dependencies**: T-IK-3.15

**Acceptance Criteria:**
- [ ] `dq = J^+ * e` where J^+ is pseudoinverse
- [ ] Pseudoinverse computation
- [ ] Handle singular/near-singular matrices
- [ ] Most accurate when invertible

---

### T-IK-3.18: Jacobian DLS Method

**Priority**: P0 (Critical)
**Estimate**: 3 hours
**Dependencies**: T-IK-3.15, T-IK-3.14

**Acceptance Criteria:**
- [ ] `dq = J^T * (J * J^T + lambda^2 * I)^-1 * e`
- [ ] Damping factor parameter
- [ ] Gauss-Jordan matrix inversion
- [ ] Avoid singularities with damping
- [ ] Most robust method

---

### T-IK-3.19: Jacobian SDLS Method

**Priority**: P2 (Medium)
**Estimate**: 2 hours
**Dependencies**: T-IK-3.18

**Acceptance Criteria:**
- [ ] Selectively Damped Least Squares
- [ ] Per-joint damping factors
- [ ] Heterogeneous chain support
- [ ] Better behavior at joint limits

---

### T-IK-3.20: Multi-Target Jacobian

**Priority**: P1 (High)
**Estimate**: 3 hours
**Dependencies**: T-IK-3.18

**Acceptance Criteria:**
- [ ] MultiTargetJacobianIK class
- [ ] Multiple effectors
- [ ] Per-effector weights
- [ ] Stacked Jacobian construction
- [ ] Weighted error vector

---

### T-IK-3.21: IK Configuration Module

**Priority**: P1 (High)
**Estimate**: 1 hour
**Dependencies**: None

**Acceptance Criteria:**
- [ ] config.py with all constants
- [ ] DEFAULT_TOLERANCE
- [ ] DEFAULT_MAX_ITERATIONS
- [ ] CCD_DAMPING
- [ ] JACOBIAN_DAMPING
- [ ] SOFT_IK_FALLOFF
- [ ] EPSILON values
- [ ] Documentation

---

## Task Summary

| Task ID | Description | Priority | Est. Hours | Dependencies |
|---------|-------------|----------|------------|--------------|
| T-IK-3.1 | IK Goal Classes | P0 | 2 | None |
| T-IK-3.2 | Joint Constraints | P0 | 3 | None |
| T-IK-3.3 | Joint Limits CCD | P1 | 2 | T-IK-3.2 |
| T-IK-3.4 | Two-Bone Core | P0 | 3 | T-IK-3.1 |
| T-IK-3.5 | Two-Bone Soft IK | P1 | 2 | T-IK-3.4 |
| T-IK-3.6 | Two-Bone Pole Vector | P1 | 2 | T-IK-3.4 |
| T-IK-3.7 | FABRIK Core | P0 | 4 | T-IK-3.2 |
| T-IK-3.8 | FABRIK Constraints | P1 | 2 | T-IK-3.7, T-IK-3.2 |
| T-IK-3.9 | FABRIK Multi-Chain | P1 | 3 | T-IK-3.7 |
| T-IK-3.10 | CCD Core | P0 | 4 | T-IK-3.3 |
| T-IK-3.11 | CCD Limits | P1 | 2 | T-IK-3.10, T-IK-3.3 |
| T-IK-3.12 | CCD Weighted | P1 | 2 | T-IK-3.10 |
| T-IK-3.13 | CCD Custom | P2 | 2 | T-IK-3.10 |
| T-IK-3.14 | Matrix Class | P0 | 3 | None |
| T-IK-3.15 | Jacobian Compute | P0 | 3 | T-IK-3.14 |
| T-IK-3.16 | Jacobian Transpose | P1 | 2 | T-IK-3.15 |
| T-IK-3.17 | Jacobian Pseudo | P1 | 2 | T-IK-3.15 |
| T-IK-3.18 | Jacobian DLS | P0 | 3 | T-IK-3.15, T-IK-3.14 |
| T-IK-3.19 | Jacobian SDLS | P2 | 2 | T-IK-3.18 |
| T-IK-3.20 | Multi-Target | P1 | 3 | T-IK-3.18 |
| T-IK-3.21 | IK Config | P1 | 1 | None |

**Total Estimate**: 53 hours

---

## Verification Checklist

After Phase 3 completion:

- [ ] Two-Bone IK solves arm configurations
- [ ] Soft IK handles unreachable targets
- [ ] FABRIK converges for chains
- [ ] Multi-chain FABRIK works
- [ ] CCD converges
- [ ] CCD respects joint limits
- [ ] Jacobian DLS solves without singularities
- [ ] Multi-target Jacobian works
- [ ] All constraints enforce correctly
- [ ] All tests pass
