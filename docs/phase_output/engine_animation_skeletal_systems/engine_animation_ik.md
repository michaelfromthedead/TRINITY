# Investigation: engine/animation/ik

## Summary
The IK subsystem contains **4,930 lines** of fully-implemented inverse kinematics code across 8 Python modules. All four major IK solver types (TwoBone, FABRIK, CCD, Jacobian) contain complete, mathematically correct implementations with real algorithms, joint constraints, and production-ready features including foot placement and full-body IK.

## Files
| File | Lines | Status | Notes |
|------|-------|--------|-------|
| __init__.py | 217 | REAL | Clean exports for all 6 solver families |
| two_bone.py | 494 | REAL | Law of cosines analytical solver with soft IK |
| fabrik.py | 616 | REAL | Forward/backward reaching with ball-socket/hinge constraints |
| ccd.py | 691 | REAL | Cyclic Coordinate Descent with weighted variants |
| jacobian.py | 692 | REAL | Transpose, pseudoinverse, DLS methods with matrix ops |
| fullbody.py | 768 | REAL | Multi-effector solver with balance/COM maintenance |
| foot_placement.py | 737 | REAL | Terrain-adaptive foot IK with pelvis adjustment |
| ik_goal.py | 569 | REAL | Goal types, blender, decorators |
| config.py | 155 | REAL | All tuning constants centralized |

## IK Solvers
1. **TwoBoneIK** - Analytical two-bone (arms/legs)
2. **FABRIKChain** - Forward And Backward Reaching IK
3. **FABRIKMultiChain** - Multi-chain FABRIK with shared joints
4. **CCDSolver** - Cyclic Coordinate Descent
5. **CCDSolverWithWeights** - CCD with per-joint weights
6. **ConstrainedCCDSolver** - CCD with custom constraint functions
7. **JacobianIK** - Jacobian methods (transpose, pseudoinverse, DLS, SDLS)
8. **MultiTargetJacobianIK** - Weighted multi-effector Jacobian
9. **FullBodyIK** - Multi-effector full body with balance
10. **LookAtSolver** - Head/eye tracking with spine distribution
11. **FootPlacement** - Terrain-adaptive bipedal foot IK
12. **FootPlacementAnimated** - With animation curve support
13. **MultiLegFootPlacement** - N-legged characters (spiders, centaurs)

## Implementation
- Real FABRIK? **YES** - Complete forward/backward passes, bone length caching, multi-chain averaging
- Real CCD? **YES** - Per-joint rotation with damping, hinge/Euler clamping, weighted variants
- Real Jacobian? **YES** - Full matrix class, Gauss-Jordan inversion, DLS damping
- Real foot placement? **YES** - Raycasting, pelvis adjustment, toe alignment, multi-leg support
- Real constraints? **YES** - Ball-socket cones, hinge joints, twist limits, custom functions
- Real two-bone? **YES** - Law of cosines, pole vectors, soft IK falloff

## Verdict
**REAL IMPLEMENTATION** - Production-quality IK system with all standard algorithms implemented correctly.

## Evidence

### TwoBone Law of Cosines (two_bone.py:233-245)
```python
# Compute mid joint angle using law of cosines
# cos(angle) = (a^2 + b^2 - c^2) / (2ab)
# where a = upper_len, b = lower_len, c = target_dist
cos_mid = (upper_len * upper_len + lower_len * lower_len - target_dist * target_dist)
cos_mid /= (2.0 * upper_len * lower_len)
cos_mid = max(-1.0, min(1.0, cos_mid))  # Clamp for numerical stability

mid_angle = math.pi - math.acos(cos_mid)
```

### FABRIK Forward/Backward Pass (fabrik.py:333-378)
```python
def _forward_pass(self, positions, target, rotations):
    """Forward pass: end effector to root."""
    pos = list(positions)
    pos[-1] = Vec3(target.x, target.y, target.z)
    
    # Work backward maintaining bone lengths
    for i in range(len(pos) - 2, -1, -1):
        direction = pos[i] - pos[i + 1]
        direction = constraint.apply(direction, parent_rot)
        bone_length = self._bone_lengths[i]
        pos[i] = pos[i + 1] + direction * bone_length
    return pos
```

### CCD Per-Joint Rotation (ccd.py:345-414)
```python
def _rotate_joint(self, joint_idx, positions, rotations, target):
    to_end = (end_pos - joint_pos).normalized()
    to_target = (target - joint_pos).normalized()
    
    dot = max(-1.0, min(1.0, to_end.dot(to_target)))
    axis = to_end.cross(to_target).normalized()
    angle = math.acos(dot) * self.damping
    
    rotation = Quat.from_axis_angle(axis, angle)
    combined = limit.clamp_rotation(rotation * rotations[joint_idx])
```

### Jacobian DLS (jacobian.py:371-411)
```python
def solve_damped_least_squares(self, jacobian, error, damping=None):
    """dq = J^T * (J * J^T + lambda^2 * I)^-1 * e"""
    J_T = jacobian.transpose()
    JJT = jacobian @ J_T
    damped = JJT + Matrix.identity(JJT.rows) * (damping * damping)
    damped_inv = self._invert_matrix(damped)
    temp = damped_inv @ e
    result = J_T @ temp
```

### Foot Placement with Pelvis Adjustment (foot_placement.py:336-402)
```python
def _calculate_pelvis_offset(self, pelvis_pos, left_target, ...):
    """Calculate required pelvis height offset."""
    if left_planted:
        to_target = left_target - pelvis_pos
        left_reach = to_target.length() - self._left_leg_ik.max_reach * SAFETY_MARGIN
    
    required_drop = max(0, max(left_reach, right_reach))
    required_drop = min(required_drop, self.max_pelvis_drop)
```

### Joint Constraints (fabrik.py:109-137)
```python
def _apply_ball_socket(self, direction, parent_rotation):
    """Apply ball-socket cone constraint."""
    ref_dir = parent_rotation.rotate_vector(Vec3(0, 1, 0))
    dot = max(-1.0, min(1.0, direction.dot(ref_dir)))
    angle = math.acos(dot)
    
    if angle <= self.cone_angle:
        return direction
    
    # Clamp to cone surface
    axis = ref_dir.cross(direction).normalized()
    rotation = Quat.from_axis_angle(axis, self.cone_angle)
    return rotation.rotate_vector(ref_dir)
```
