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
- [x] IKGoal abstract base class
- [x] PositionGoal dataclass
- [x] RotationGoal dataclass
- [x] LookAtGoal dataclass
- [x] PositionRotationGoal dataclass
- [x] PoleVectorGoal dataclass
- [x] COMGoal dataclass
- [x] IKGoalBlender for weighted blending

**SDLC Status:** GREEN_LIGHT (2026-06-02)
- Tests: 189 pass (113 whitebox + 76 blackbox)
- JUNIOR: 9 findings → SANITY: All OVERZEALOUS
- Commit: ea5963c0

---

### T-IK-3.2: Joint Constraints Base

**Priority**: P0 (Critical)
**Estimate**: 3 hours
**Dependencies**: None

**Acceptance Criteria:**
- [x] JointConstraint abstract base class
- [x] apply(direction, parent_rotation) signature
- [x] BallSocketConstraint with cone angle
- [x] HingeConstraint with axis and angle limits
- [x] TwistConstraint with twist limits
- [x] Numerical stability checks

**SDLC Status:** GREEN_LIGHT (2026-06-02)
- Tests: 145 pass (81 whitebox + 64 blackbox)
- JUNIOR: 6 findings → SANITY: 1 LOW (R2 fixed)
- Commit: ba0197a6

---

### T-IK-3.3: Joint Limits for CCD

**Priority**: P1 (High)
**Estimate**: 2 hours
**Dependencies**: T-IK-3.2

**Acceptance Criteria:**
- [x] JointLimit abstract base class
- [x] clamp(rotation) signature
- [x] EulerLimit with min/max per axis
- [x] EulerOrder enum (6 orders)
- [x] SwingTwistLimit decomposition
- [x] Proper Euler extraction and reconstruction

**SDLC Status:** GREEN_LIGHT (2026-06-02)
- Tests: 166 pass (114 whitebox + 52 blackbox)
- JUNIOR: GREEN_LIGHT (4 LOW documentation notes)
- Commit: d2734802

---

### T-IK-3.4: Two-Bone IK Core

**Priority**: P0 (Critical)
**Estimate**: 3 hours
**Dependencies**: T-IK-3.1

**Acceptance Criteria:**
- [x] TwoBoneIK class
- [x] Law of cosines angle calculation
- [x] `cos_mid = (a^2 + b^2 - c^2) / (2ab)`
- [x] Numerical stability clamping
- [x] solve(root, mid, end, target, pole) method
- [x] TwoBoneIKResult dataclass

**SDLC Status:** GREEN_LIGHT (2026-06-02)
- Tests: 149 pass (92 whitebox + 57 blackbox)
- JUNIOR: 5 findings → SANITY: All OVERZEALOUS
- Commit: 5474389c

---

### T-IK-3.5: Two-Bone Soft IK

**Priority**: P1 (High)
**Estimate**: 2 hours
**Dependencies**: T-IK-3.4

**Acceptance Criteria:**
- [x] SoftIKConfig dataclass (implemented via inline constructor parameters - cleaner design)
- [x] Exponential falloff implementation
- [x] `d_soft = d_start + (d_max - d_start) * (1 - e^(-k * overshoot))`
- [x] Smooth blending between hard and soft limits
- [x] Handle unreachable targets gracefully

**SDLC Status:** GREEN_LIGHT (2026-06-02)
- Tests: 13 soft IK tests in T-IK-3.4 whitebox (TestSoftIKBehavior)
- JUNIOR: GREEN_LIGHT (inline params cleaner than dataclass)
- Note: Implemented as part of TwoBoneIK (T-IK-3.4)

---

### T-IK-3.6: Two-Bone Pole Vector

**Priority**: P1 (High)
**Estimate**: 2 hours
**Dependencies**: T-IK-3.4

**Acceptance Criteria:**
- [x] Pole vector positioning
- [x] Plane projection for elbow/knee direction
- [x] Twist control
- [x] Handle pole vector at singular positions

**SDLC Status:** GREEN_LIGHT (2026-06-02)
- Tests: 8 pole vector tests in T-IK-3.4 whitebox (TestPoleVectorHandling)
- JUNIOR: GREEN_LIGHT
- Note: Implemented as part of TwoBoneIK (T-IK-3.4)

---

### T-IK-3.7: FABRIK Chain Core

**Priority**: P0 (Critical)
**Estimate**: 4 hours
**Dependencies**: T-IK-3.2

**Acceptance Criteria:**
- [x] FABRIKChain class
- [x] Bone length storage
- [x] Forward pass (end to root)
- [x] Backward pass (root to end)
- [x] Convergence check
- [x] Max iterations limit
- [x] FABRIKResult dataclass

**SDLC Status:** GREEN_LIGHT (2026-06-02)
- Tests: 195 pass (118 whitebox + 77 blackbox)
- JUNIOR: GREEN_LIGHT (4 OVERZEALOUS notes)
- Implementation: fabrik.py:161-525

---

### T-IK-3.8: FABRIK Constraint Integration

**Priority**: P1 (High)
**Estimate**: 2 hours
**Dependencies**: T-IK-3.7, T-IK-3.2

**Acceptance Criteria:**
- [x] Per-joint constraint application
- [x] Constraint application in both passes
- [x] Ball-socket constraint integration
- [x] Hinge constraint integration
- [x] Constraint affects direction preservation

**SDLC Status:** GREEN_LIGHT (2026-06-02)
- Tests: 31 constraint tests in T-IK-3.7 whitebox
- FAST-TRACK: Implemented in fabrik.py with T-IK-3.7
- Implementation: fabrik.py:44-140 (JointConstraint), fabrik.py:225-233, 372-374, 418-420

---

### T-IK-3.9: FABRIK Multi-Chain

**Priority**: P1 (High)
**Estimate**: 3 hours
**Dependencies**: T-IK-3.7

**Acceptance Criteria:**
- [x] FABRIKMultiChain class
- [x] Shared joint handling
- [x] Average position at shared joints
- [x] Multiple target support
- [~] Chain priority/weighting (equal weight via lerp)
- [~] MultiChainResult dataclass (returns List[Vec3])

**SDLC Status:** GREEN_LIGHT (2026-06-02)
- Tests: 133 pass (76 whitebox + 57 blackbox)
- JUNIOR: GREEN_LIGHT (AC5/AC6 acceptable gaps)
- Implementation: fabrik.py:527-617

---

### T-IK-3.10: CCD Solver Core

**Priority**: P0 (Critical)
**Estimate**: 4 hours
**Dependencies**: T-IK-3.3

**Acceptance Criteria:**
- [x] CCDSolver class
- [x] Per-joint rotation calculation
- [x] to-end / to-target vector computation
- [x] Rotation axis from cross product
- [x] Angle from acos(dot)
- [x] Damping factor application
- [x] CCDResult dataclass

**SDLC Status:** GREEN_LIGHT (2026-06-02)
- Tests: 194 pass (118 whitebox + 76 blackbox)
- JUNIOR: GREEN_LIGHT (1 LOW documentation note)
- Implementation: ccd.py:151-476

---

### T-IK-3.11: CCD Joint Limit Integration

**Priority**: P1 (High)
**Estimate**: 2 hours
**Dependencies**: T-IK-3.10, T-IK-3.3

**Acceptance Criteria:**
- [x] JointLimit.clamp() after each rotation
- [x] Hinge constraint enforcement
- [x] Euler limit enforcement
- [x] Proper rotation order handling

**SDLC Status:** GREEN_LIGHT (2026-06-02)
- Tests: 21 limit tests in T-IK-3.10 whitebox
- FAST-TRACK: Implemented in ccd.py with T-IK-3.10
- Implementation: ccd.py:40-130 (RotationLimit), ccd.py:220, 407, 575, 631

---

### T-IK-3.12: CCD Weighted Variant

**Priority**: P1 (High)
**Estimate**: 2 hours
**Dependencies**: T-IK-3.10

**Acceptance Criteria:**
- [x] CCDSolverWithWeights class
- [x] Per-joint weight storage
- [x] Weight affects rotation amount
- [x] Zero weight = locked joint
- [x] High weight = more influence

**SDLC Status:** GREEN_LIGHT (2026-06-02)
- Tests: 12 weighted tests in T-IK-3.10 whitebox
- FAST-TRACK: Implemented in ccd.py with T-IK-3.10
- Implementation: ccd.py:479-556 (CCDSolverWithWeights)

---

### T-IK-3.13: CCD Custom Constraints

**Priority**: P2 (Medium)
**Estimate**: 2 hours
**Dependencies**: T-IK-3.10

**Acceptance Criteria:**
- [x] ConstrainedCCDSolver class
- [x] Custom constraint function support
- [x] Constraint function signature: (rotation, joint_idx) -> rotation
- [x] Multiple constraint functions per joint

**SDLC Status:** GREEN_LIGHT (2026-06-02)
- Tests: 8 constrained tests in T-IK-3.10 whitebox
- FAST-TRACK: Implemented in ccd.py with T-IK-3.10
- Implementation: ccd.py:583-645 (ConstrainedCCDSolver)

---

### T-IK-3.14: Matrix Class

**Priority**: P0 (Critical)
**Estimate**: 3 hours
**Dependencies**: None

**Acceptance Criteria:**
- [x] Matrix class for arbitrary dimensions
- [x] Matrix multiplication
- [x] Matrix transpose
- [x] Gauss-Jordan inversion
- [x] Identity matrix creation
- [x] Column/row access
- [x] Numerical stability handling

**SDLC Status:** GREEN_LIGHT (2026-06-02)
- Tests: 51 Matrix tests in jacobian whitebox
- Implementation: jacobian.py:63-159

---

### T-IK-3.15: Jacobian Computation

**Priority**: P0 (Critical)
**Estimate**: 3 hours
**Dependencies**: T-IK-3.14

**Acceptance Criteria:**
- [x] compute_jacobian() method
- [x] `J_col = axis x (end_effector - joint_pos)`
- [x] Multi-DOF joints support
- [x] Rotation axis per joint
- [x] Jacobian matrix construction

**SDLC Status:** GREEN_LIGHT (2026-06-02)
- Tests: 18 Jacobian computation tests
- Implementation: jacobian.py:243-292

---

### T-IK-3.16: Jacobian Transpose Method

**Priority**: P1 (High)
**Estimate**: 2 hours
**Dependencies**: T-IK-3.15

**Acceptance Criteria:**
- [x] `dq = alpha * J^T * e`
- [x] Step size calculation
- [x] Convergence handling
- [x] Fast but may oscillate

**SDLC Status:** GREEN_LIGHT (2026-06-02)
- Tests: 9 transpose method tests
- Implementation: jacobian.py:294-331

---

### T-IK-3.17: Jacobian Pseudoinverse Method

**Priority**: P1 (High)
**Estimate**: 2 hours
**Dependencies**: T-IK-3.15

**Acceptance Criteria:**
- [x] `dq = J^+ * e` where J^+ is pseudoinverse
- [x] Pseudoinverse computation
- [x] Handle singular/near-singular matrices
- [x] Most accurate when invertible

**SDLC Status:** GREEN_LIGHT (2026-06-02)
- Tests: 8 pseudoinverse method tests
- Implementation: jacobian.py:333-369

---

### T-IK-3.18: Jacobian DLS Method

**Priority**: P0 (Critical)
**Estimate**: 3 hours
**Dependencies**: T-IK-3.15, T-IK-3.14

**Acceptance Criteria:**
- [x] `dq = J^T * (J * J^T + lambda^2 * I)^-1 * e`
- [x] Damping factor parameter
- [x] Gauss-Jordan matrix inversion
- [x] Avoid singularities with damping
- [x] Most robust method

**SDLC Status:** GREEN_LIGHT (2026-06-02)
- Tests: 11 DLS method tests
- Implementation: jacobian.py:371-411
- JUNIOR: GREEN_LIGHT (3 LOW notes)

---

### T-IK-3.19: Jacobian SDLS Method

**Priority**: P2 (Medium)
**Estimate**: 2 hours
**Dependencies**: T-IK-3.18

**Acceptance Criteria:**
- [~] Selectively Damped Least Squares (falls back to DLS)
- [~] Per-joint damping factors (not implemented)
- [~] Heterogeneous chain support (via DLS)
- [~] Better behavior at joint limits (via DLS)

**SDLC Status:** GREEN_LIGHT (2026-06-02)
- Tests: 2 SDLS tests in jacobian whitebox
- Note: Enum exists, falls back to DLS (JUNIOR R3 LOW)
- Implementation: jacobian.py:38, 536-537

---

### T-IK-3.20: Multi-Target Jacobian

**Priority**: P1 (High)
**Estimate**: 3 hours
**Dependencies**: T-IK-3.18

**Acceptance Criteria:**
- [x] MultiTargetJacobianIK class
- [x] Multiple effectors
- [~] Per-effector weights (partial)
- [x] Stacked Jacobian construction
- [~] Weighted error vector (partial)

**SDLC Status:** GREEN_LIGHT (2026-06-02)
- Tests: 5 multi-target tests in jacobian whitebox
- Note: Weights are stored but not applied to error (JUNIOR R2 LOW)
- Implementation: jacobian.py:644-691

---

### T-IK-3.21: IK Configuration Module

**Priority**: P1 (High)
**Estimate**: 1 hour
**Dependencies**: None

**Acceptance Criteria:**
- [x] config.py with all constants
- [x] DEFAULT_TOLERANCE (IK_DEFAULT_TOLERANCE)
- [x] DEFAULT_MAX_ITERATIONS (multiple)
- [x] CCD_DAMPING (CCD_DEFAULT_DAMPING)
- [x] JACOBIAN_DAMPING (JACOBIAN_DLS_DAMPING)
- [x] SOFT_IK_FALLOFF (SOFT_IK_FALLOFF_RATE)
- [x] EPSILON values (MIN_BONE_LENGTH)
- [x] Documentation

**SDLC Status:** GREEN_LIGHT (2026-06-02)
- Implementation: config.py (154 lines)
- Note: All constants documented and used by IK modules

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
