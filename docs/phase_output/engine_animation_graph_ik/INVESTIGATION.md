# Archaeological Investigation: engine/animation/graph + engine/animation/ik

**Date**: 2026-05-22
**Investigator**: Research Agent
**Classification**: REAL implementations (both subdirectories)

---

## Executive Summary

Both `engine/animation/graph` (~5,057 lines) and `engine/animation/ik` (~4,776 lines) contain **production-quality REAL implementations** with complete algorithmic logic, proper error handling, and coherent cross-module integration. These are not stubs.

---

## engine/animation/graph Classification: REAL

### File Analysis

| File | Lines | Classification | Evidence |
|------|-------|----------------|----------|
| `animation_graph.py` | 1039 | REAL | Complete DAG-based animation graph with metaclass auto-registration, pose/transform math with SLERP, skeleton hierarchy, bone masks, typed parameters, subgraph support, cycle detection |
| `blend_tree.py` | 848 | REAL | Full 1D/2D/Direct blend trees with Delaunay triangulation for 2D Cartesian blending, inverse distance weighting, polar interpolation, gradient bands |
| `state_machine.py` | 828 | REAL | Complete FSM with states, transitions, conditions (8 comparison operators), blend curves (6 easing functions), any-state transitions, interruption handling |
| `blend_node.py` | 775 | REAL | 10 node types: ClipNode, BlendNode, AdditiveNode, LayerNode, MirrorNode, TimeScaleNode, PoseCacheNode, SelectNode, with keyframe sampling and loop modes |
| `sync.py` | 671 | REAL | Animation synchronization with 4 sync modes (none/normalized/phase/leader-follower), sync markers for locomotion, event synchronization |
| `layer.py` | 551 | REAL | Layer stack system with 3 blend modes, bone mask presets (upper/lower body, arms, legs, gradient), fluent builder pattern |

### Key Algorithms Identified

1. **Quaternion SLERP** (`animation_graph.py:91-128`)
   - Proper shortest-path handling via dot product sign check
   - Threshold-based linear interpolation fallback
   - Sin theta safety check

2. **Delaunay Triangulation** (`blend_tree.py:551-626`)
   - Bowyer-Watson algorithm implementation
   - Circumcircle test with matrix determinant method
   - Super-triangle for boundary handling

3. **Barycentric Coordinates** (`blend_tree.py:291-322`)
   - Proper dot product computation
   - Clamping for numerical stability

4. **DAG Cycle Detection** (`animation_graph.py:924-947`)
   - Three-color (WHITE/GRAY/BLACK) DFS algorithm
   - Back-edge detection for cycle identification

5. **Blend Curve Evaluation** (`state_machine.py:59-77`)
   - 6 easing curves: linear, ease-in/out/in-out, smooth-step, smoother-step
   - Smoother-step: `t^3 * (t * (t * 6 - 15) + 10)`

### Integration Evidence

- Imports from `engine.animation.graph.config` for configurable parameters
- Consistent use of dataclasses with factory defaults
- Comprehensive `__all__` exports in each module
- Metaclass-based node registration (`GraphNodeMeta`)
- Cross-references between modules (e.g., `blend_tree.py` imports from `animation_graph.py`)

---

## engine/animation/ik Classification: REAL

### File Analysis

| File | Lines | Classification | Evidence |
|------|-------|----------------|----------|
| `fullbody.py` | 767 | REAL | Complete full-body IK with multi-chain solving, pelvis height adjustment, COM balance maintenance, point-in-polygon test, LookAtSolver with distributed rotation |
| `foot_placement.py` | 736 | REAL | Foot IK for terrain adaptation with raycasting callbacks, pelvis offset calculation, terrain slope alignment, multi-leg support (spiders, etc.) |
| `jacobian.py` | 691 | REAL | Jacobian-based IK with 4 methods (transpose, pseudoinverse, DLS, SDLS), custom Matrix class with Gauss-Jordan inversion, multi-effector support |
| `ccd.py` | 690 | REAL | Complete CCD solver with 3 rotation orders, hinge/Euler constraint types, weighted joints, custom constraint functions |
| `fabrik.py` | 615 | REAL | FABRIK forward/backward passes, joint constraints (hinge/ball-socket/twist), multi-chain solver for connected chains |
| `ik_goal.py` | 568 | REAL | 6 goal types (position, rotation, look-at, pos+rot, pole vector, COM), goal blending utility, decorator support |
| `two_bone.py` | 493 | REAL | Analytical two-bone IK using law of cosines, soft IK with exponential falloff, pole vector control, bend angle constraints |

### Key Algorithms Identified

1. **Law of Cosines Two-Bone IK** (`two_bone.py:233-246`)
   ```
   cos_mid = (a^2 + b^2 - c^2) / (2ab)
   mid_angle = pi - acos(cos_mid)
   ```

2. **Soft IK Exponential Falloff** (`two_bone.py:118-156`)
   - `d_soft = d_start + (d_max - d_start) * (1 - e^(-k * overshoot))`
   - Smooth blending between hard and soft limits

3. **Jacobian Computation** (`jacobian.py:247-292`)
   - `J_col = axis cross (end_effector - joint_pos)` for rotation joints
   - Multi-DOF support per joint
   - Multiple end-effector handling

4. **Damped Least Squares** (`jacobian.py:377-411`)
   - `dq = J^T * (J * J^T + lambda^2 * I)^-1 * e`
   - Gauss-Jordan matrix inversion

5. **FABRIK Forward/Backward Passes** (`fabrik.py:333-424`)
   - Forward: End effector to target, work backward maintaining lengths
   - Backward: Root to original position, work forward
   - Joint constraint application at each step

6. **CCD Joint Rotation** (`ccd.py:345-413`)
   - Compute to-end and to-target vectors
   - Rotation axis from cross product
   - Angle from acos(dot) with damping

7. **Point-in-Polygon (Balance Check)** (`fullbody.py:569-599`)
   - Ray casting algorithm for support polygon
   - Handles horizontal edges

8. **Closest Point on Polygon Edge** (`fullbody.py:601-641`)
   - Edge projection with clamping
   - Used for COM balance correction

### Integration Evidence

- All modules import from `engine.core.math.vec`, `engine.core.math.quat`, `engine.core.math.transform`
- Uses `engine.core.constants.MATH_EPSILON` for numerical stability
- Consistent configuration from `engine.animation.ik.config`
- Cross-references: `fullbody.py` uses `TwoBoneIK` and `FABRIKChain`
- `foot_placement.py` instantiates `TwoBoneIK` solvers

---

## Evidence Summary

### REAL Indicators Present

| Indicator | graph | ik |
|-----------|-------|-----|
| Complete algorithmic implementations | Yes | Yes |
| Numerical stability handling (epsilon checks) | Yes | Yes |
| Multiple algorithmic variants | 6 blend modes, 4 sync modes | 4 Jacobian methods, 3 CCD orders |
| Cross-module imports/dependencies | Yes | Yes |
| Configuration externalization | Yes (`config.py`) | Yes (`config.py`) |
| Comprehensive docstrings | Yes | Yes |
| Result dataclasses with multiple fields | `Pose`, `Transform`, etc. | `FABRIKResult`, `TwoBoneIKResult`, etc. |
| Builder patterns | `StateMachineBuilder`, `LayerStackBuilder` | N/A |
| Decorators for DSL | `@blend_tree`, `@state_machine` | `@ik_goal`, `@ik_chain` |
| Edge case handling | Empty inputs, cycles, degenerate chains | Zero-length vectors, unreachable targets |

### STUB Indicators Absent

- No `raise NotImplementedError`
- No `pass` without implementation
- No placeholder comments like `# TODO`
- No empty method bodies
- No hardcoded dummy values

---

## Dependencies Identified

### External (core engine)
- `engine.core.math.vec.Vec3`
- `engine.core.math.quat.Quat`
- `engine.core.math.transform.Transform`
- `engine.core.constants.MATH_EPSILON`

### Internal Cross-Module
- `graph/blend_node.py` imports from `graph/animation_graph.py`
- `graph/sync.py` imports from `graph/blend_node.py`
- `graph/layer.py` imports from `graph/animation_graph.py`
- `ik/fullbody.py` imports from `ik/fabrik.py`, `ik/two_bone.py`, `ik/ik_goal.py`
- `ik/foot_placement.py` imports from `ik/two_bone.py`

---

## Conclusion

Both `engine/animation/graph` and `engine/animation/ik` are **REAL, production-quality implementations** with:

1. **Complete mathematical algorithms** (SLERP, Delaunay, Law of Cosines, Jacobian methods, FABRIK, CCD)
2. **Proper software engineering** (dataclasses, builders, decorators, comprehensive exports)
3. **Numerical robustness** (epsilon comparisons, clamping, singularity handling)
4. **Coherent architecture** (shared config, consistent patterns, cross-module integration)

These ~9,833 lines represent substantial animation system functionality suitable for production use. No stubs detected.
