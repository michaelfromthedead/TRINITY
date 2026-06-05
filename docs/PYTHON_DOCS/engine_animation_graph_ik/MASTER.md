# MASTER: engine_animation_graph_ik

**Consolidated Knowledge Document**
**Last Updated**: 2026-05-23
**Source Documents**: 3

---

## 1. Subsystem Overview

### 1.1 Animation Graph Subsystem (engine/animation/graph)
- **Lines**: ~5,057-5,500
- **Files**: 8 Python modules
- **Classification**: REAL IMPLEMENTATION
- **Comparable To**: Unity Animator, Unreal AnimGraph

### 1.2 IK Subsystem (engine/animation/ik)
- **Lines**: ~4,776-4,930
- **Files**: 8-9 Python modules
- **Classification**: REAL IMPLEMENTATION
- **Comparable To**: Production-quality IK system with all standard algorithms

---

## 2. Animation Graph Components

### 2.1 Core Architecture

| Component | Description | Lines |
|-----------|-------------|-------|
| AnimationGraph | DAG container with nodes, connections, parameters, subgraphs | 1039-1040 |
| AnimationNode | Base class with metaclass auto-registration (GraphNodeMeta) | Part of animation_graph.py |
| GraphContext | Evaluation context with parameters, skeleton, dt, sync groups | Part of animation_graph.py |

### 2.2 State Machine System

| File | Lines | Status |
|------|-------|--------|
| state_machine.py | 828-829 | Complete |

**Features:**
- Full FSM with AnimationState, StateTransition, TransitionCondition
- 8 comparison operators for conditions
- 6 blend curves: linear, ease-in, ease-out, ease-in-out, smooth-step, smoother-step
- Smoother-step formula: `t^3 * (t * (t * 6 - 15) + 10)`
- Any-state transitions
- Priority system
- Interrupt handling
- Exit time support
- Fixed/percentage duration modes

**Sync Modes:**
- None
- Normalized
- Proportional
- Marker-based

### 2.3 Blend Tree System

| Component | Lines | Features |
|-----------|-------|----------|
| BlendTree1D | Part of blend_tree.py (849) | 1D parameter blending with gradient bands |
| BlendTree2D | Part of blend_tree.py (849) | 4 modes: Cartesian, Polar, Freeform Directional, Freeform Cartesian (Delaunay) |
| BlendTreeDirect | Part of blend_tree.py (849) | Explicit weight-controlled blending |

**2D Blending Algorithms:**
- Delaunay triangulation (Bowyer-Watson algorithm)
- Barycentric interpolation
- Inverse distance weighting
- Polar interpolation

### 2.4 Blend Nodes

| Node Type | Purpose |
|-----------|---------|
| ClipNode | Single animation clip playback |
| BlendNode | Generic blending |
| AdditiveNode | Additive animation layering |
| LayerNode | Layer-based composition |
| MirrorNode | Animation mirroring |
| TimeScaleNode | Playback speed control |
| PoseCacheNode | Pose caching for performance |
| SelectNode | Conditional node selection |

**Additional Features:**
- Keyframe sampling
- Loop modes

### 2.5 Layer System

| File | Lines | Status |
|------|-------|--------|
| layer.py | 551-552 | Complete |

**Features:**
- LayerStack for multi-layer animation
- 3 blend modes
- Fluent builder pattern (LayerStackBuilder)
- BoneMaskPresets: upper body, lower body, arms, legs, gradient masks

### 2.6 Synchronization System

| File | Lines | Status |
|------|-------|--------|
| sync.py | 671-672 | Complete |

**Sync Modes:**
- None
- Normalized
- Phase
- Leader-follower
- Weighted

**Features:**
- SyncGroup for animation synchronization
- Sync markers for locomotion
- EventSynchronizer for cross-animation event coordination

---

## 3. IK Subsystem Components

### 3.1 Two-Bone IK

| File | Lines | Status |
|------|-------|--------|
| two_bone.py | 493-494 | Complete |

**Algorithm:**
```
cos_mid = (a^2 + b^2 - c^2) / (2ab)
mid_angle = pi - acos(cos_mid)
```

**Features:**
- Analytical solution using law of cosines
- Soft IK with exponential falloff
- Formula: `d_soft = d_start + (d_max - d_start) * (1 - e^(-k * overshoot))`
- Pole vector control
- Bend angle constraints

### 3.2 FABRIK (Forward And Backward Reaching IK)

| File | Lines | Status |
|------|-------|--------|
| fabrik.py | 615-616 | Complete |

**Algorithm:**
- Forward pass: End effector to target, work backward maintaining lengths
- Backward pass: Root to original position, work forward
- Joint constraint application at each step

**Solver Variants:**
- FABRIKChain: Single chain solver
- FABRIKMultiChain: Multi-chain with shared joints

**Joint Constraints:**
- Ball-socket (cone constraint)
- Hinge joints
- Twist limits

### 3.3 CCD (Cyclic Coordinate Descent)

| File | Lines | Status |
|------|-------|--------|
| ccd.py | 690-691 | Complete |

**Algorithm:**
- Compute to-end and to-target vectors
- Rotation axis from cross product
- Angle from acos(dot) with damping

**Solver Variants:**
- CCDSolver: Basic CCD
- CCDSolverWithWeights: Per-joint weights
- ConstrainedCCDSolver: Custom constraint functions

**Features:**
- 3 rotation orders
- Hinge/Euler constraint types
- Weighted joints
- Custom constraint functions

### 3.4 Jacobian IK

| File | Lines | Status |
|------|-------|--------|
| jacobian.py | 691-692 | Complete |

**Methods:**
- Transpose
- Pseudoinverse
- DLS (Damped Least Squares): `dq = J^T * (J * J^T + lambda^2 * I)^-1 * e`
- SDLS (Selectively Damped Least Squares)

**Features:**
- Custom Matrix class
- Gauss-Jordan matrix inversion
- Multi-effector support
- Jacobian computation: `J_col = axis cross (end_effector - joint_pos)`

**Solver Variants:**
- JacobianIK: Single target
- MultiTargetJacobianIK: Weighted multi-effector

### 3.5 Full Body IK

| File | Lines | Status |
|------|-------|--------|
| fullbody.py | 767-768 | Complete |

**Features:**
- Multi-chain solving
- Pelvis height adjustment
- COM (Center of Mass) balance maintenance
- Point-in-polygon test (ray casting algorithm)
- Closest point on polygon edge for balance correction
- LookAtSolver with distributed rotation

### 3.6 Foot Placement

| File | Lines | Status |
|------|-------|--------|
| foot_placement.py | 736-737 | Complete |

**Solver Variants:**
- FootPlacement: Terrain-adaptive bipedal foot IK
- FootPlacementAnimated: With animation curve support
- MultiLegFootPlacement: N-legged characters (spiders, centaurs)

**Features:**
- Raycasting callbacks
- Pelvis offset calculation
- Terrain slope alignment
- Toe alignment
- Multi-leg support

### 3.7 IK Goals

| File | Lines | Status |
|------|-------|--------|
| ik_goal.py | 568-569 | Complete |

**Goal Types (6):**
- Position
- Rotation
- Look-at
- Position + Rotation
- Pole vector
- COM (Center of Mass)

**Features:**
- Goal blending utility
- Decorator support (@ik_goal, @ik_chain)

---

## 4. Key Algorithms

### 4.1 Quaternion SLERP (animation_graph.py:91-128)
- Proper shortest-path handling via dot product sign check
- Threshold-based linear interpolation fallback
- Sin theta safety check

### 4.2 Delaunay Triangulation (blend_tree.py:551-626)
- Bowyer-Watson algorithm implementation
- Circumcircle test with matrix determinant method
- Super-triangle for boundary handling

### 4.3 Barycentric Coordinates (blend_tree.py:291-322)
- Proper dot product computation
- Clamping for numerical stability

### 4.4 DAG Cycle Detection (animation_graph.py:924-947)
- Three-color (WHITE/GRAY/BLACK) DFS algorithm
- Back-edge detection for cycle identification

### 4.5 Point-in-Polygon Balance Check (fullbody.py:569-599)
- Ray casting algorithm for support polygon
- Handles horizontal edges

---

## 5. Dependencies

### 5.1 External (Core Engine)
- `engine.core.math.vec.Vec3`
- `engine.core.math.quat.Quat`
- `engine.core.math.transform.Transform`
- `engine.core.constants.MATH_EPSILON`

### 5.2 Internal Cross-Module (Graph)
- `graph/blend_node.py` imports from `graph/animation_graph.py`
- `graph/sync.py` imports from `graph/blend_node.py`
- `graph/layer.py` imports from `graph/animation_graph.py`

### 5.3 Internal Cross-Module (IK)
- `ik/fullbody.py` imports from `ik/fabrik.py`, `ik/two_bone.py`, `ik/ik_goal.py`
- `ik/foot_placement.py` imports from `ik/two_bone.py`

---

## 6. Configuration

### 6.1 Graph Configuration (config.py: 100 lines)
- Centralized tuning constants
- `quaternion.SLERP_DOT_THRESHOLD` for SLERP fallback

### 6.2 IK Configuration (config.py: 155 lines)
- All tuning constants centralized
- Numerical stability thresholds

---

## 7. Architectural Features

### 7.1 Metaclass System
- GraphNodeMeta: Auto-registration of animation node types

### 7.2 Builder Patterns
- StateMachineBuilder: Fluent state machine construction
- LayerStackBuilder: Fluent layer stack construction

### 7.3 Decorator DSL
- @state_machine: Declarative state machine definitions
- @blend_tree: Declarative blend tree definitions
- @ik_goal: IK goal specification
- @ik_chain: IK chain definition

### 7.4 Data Classes
- Pose, Transform: Full 3D transforms with quaternion support
- Skeleton, Bone, BoneMask: Skeletal hierarchy and masking
- FABRIKResult, TwoBoneIKResult: Solver result containers

---

## 8. Quality Indicators

### 8.1 REAL Implementation Evidence
| Indicator | Graph | IK |
|-----------|-------|-----|
| Complete algorithmic implementations | Yes | Yes |
| Numerical stability handling (epsilon checks) | Yes | Yes |
| Multiple algorithmic variants | 6 blend modes, 4 sync modes | 4 Jacobian methods, 3 CCD orders |
| Cross-module imports/dependencies | Yes | Yes |
| Configuration externalization | Yes | Yes |
| Comprehensive docstrings | Yes | Yes |
| Result dataclasses with multiple fields | Yes | Yes |
| Builder patterns | Yes | N/A |
| Decorators for DSL | Yes | Yes |
| Edge case handling | Yes | Yes |

### 8.2 STUB Indicators (All Absent)
- No `raise NotImplementedError`
- No `pass` without implementation
- No placeholder comments like `# TODO`
- No empty method bodies
- No hardcoded dummy values

---

## 9. Total Line Counts

| Subsystem | Lines |
|-----------|-------|
| Animation Graph | ~5,057-5,500 |
| IK | ~4,776-4,930 |
| **Combined** | **~9,833-10,430** |

---

## 10. Verdict

**REAL, PRODUCTION-QUALITY IMPLEMENTATIONS** with:
1. Complete mathematical algorithms
2. Proper software engineering patterns
3. Numerical robustness
4. Coherent architecture

Both subsystems are suitable for production use. No stubs detected.
