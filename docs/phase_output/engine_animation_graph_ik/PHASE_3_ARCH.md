# PHASE 3 ARCHITECTURE: IK Solvers

**Phase**: 3 of 4
**Focus**: Inverse Kinematics Algorithms
**Subsystem**: engine/animation/ik (core solvers)

---

## 1. Phase Scope

Implement all IK solver algorithms:
- Analytical: Two-Bone IK
- Iterative: FABRIK, CCD, Jacobian
- Joint constraints: Ball-socket, hinge, twist
- IK goal system

---

## 2. Module Architecture

### 2.1 two_bone.py (~494 lines)

**Purpose**: Analytical two-bone IK for arms and legs

```
TwoBoneIK
    |
    +-- upper_length: float
    +-- lower_length: float
    +-- max_reach: float
    +-- soft_ik: SoftIKConfig
    |
    +-- solve(root, mid, end, target, pole) -> TwoBoneIKResult
```

**Law of Cosines:**
```python
def solve_angles(upper_len, lower_len, target_dist):
    """
    Using law of cosines: c^2 = a^2 + b^2 - 2ab*cos(C)
    Solve for mid joint angle.
    """
    cos_mid = (upper_len**2 + lower_len**2 - target_dist**2)
    cos_mid /= (2.0 * upper_len * lower_len)
    cos_mid = clamp(cos_mid, -1.0, 1.0)  # Numerical stability
    
    mid_angle = math.pi - math.acos(cos_mid)
    return mid_angle
```

**Soft IK:**
```python
@dataclass
class SoftIKConfig:
    soft_distance: float  # Distance at which soft IK starts
    max_distance: float   # Maximum reach
    falloff: float        # Exponential falloff rate

def apply_soft_ik(target_dist, config):
    """
    Exponential falloff for unreachable targets.
    d_soft = d_start + (d_max - d_start) * (1 - e^(-k * overshoot))
    """
    if target_dist <= config.soft_distance:
        return target_dist
    
    overshoot = target_dist - config.soft_distance
    softened = config.soft_distance + \
        (config.max_distance - config.soft_distance) * \
        (1 - math.exp(-config.falloff * overshoot))
    
    return min(softened, config.max_distance)
```

### 2.2 fabrik.py (~616 lines)

**Purpose**: FABRIK (Forward And Backward Reaching IK)

```
FABRIKChain
    |
    +-- joint_positions: List[Vec3]
    +-- bone_lengths: List[float]
    +-- constraints: List[JointConstraint]
    +-- tolerance: float
    +-- max_iterations: int
    |
    +-- solve(target) -> FABRIKResult
    
FABRIKMultiChain
    |
    +-- chains: List[FABRIKChain]
    +-- shared_joints: Dict[int, List[ChainIndex]]
    |
    +-- solve(targets: Dict[int, Vec3]) -> MultiChainResult
```

**Algorithm:**
```python
def solve(self, target: Vec3) -> FABRIKResult:
    positions = list(self.joint_positions)
    root = positions[0]
    
    for iteration in range(self.max_iterations):
        # Check convergence
        if (positions[-1] - target).length() < self.tolerance:
            break
        
        # Forward pass: end -> root
        positions = self._forward_pass(positions, target)
        
        # Backward pass: root -> end
        positions = self._backward_pass(positions, root)
    
    return FABRIKResult(positions, converged, iterations)

def _forward_pass(self, positions, target):
    pos = list(positions)
    pos[-1] = target
    
    for i in range(len(pos) - 2, -1, -1):
        direction = (pos[i] - pos[i + 1]).normalized()
        direction = self.constraints[i].apply(direction)
        pos[i] = pos[i + 1] + direction * self.bone_lengths[i]
    
    return pos

def _backward_pass(self, positions, root):
    pos = list(positions)
    pos[0] = root
    
    for i in range(len(pos) - 1):
        direction = (pos[i + 1] - pos[i]).normalized()
        direction = self.constraints[i].apply(direction)
        pos[i + 1] = pos[i] + direction * self.bone_lengths[i]
    
    return pos
```

### 2.3 ccd.py (~691 lines)

**Purpose**: Cyclic Coordinate Descent IK

```
CCDSolver
    |
    +-- joint_rotations: List[Quat]
    +-- joint_positions: List[Vec3]
    +-- limits: List[JointLimit]
    +-- damping: float
    |
    +-- solve(target) -> CCDResult
    
CCDSolverWithWeights
    |
    +-- weights: List[float]  # Per-joint influence
    
ConstrainedCCDSolver
    |
    +-- constraint_functions: List[Callable]
```

**Algorithm:**
```python
def solve(self, target: Vec3) -> CCDResult:
    for iteration in range(self.max_iterations):
        # Check convergence
        end_effector = self._get_end_effector()
        if (end_effector - target).length() < self.tolerance:
            break
        
        # Iterate joints from end to root
        for joint_idx in range(len(self.joints) - 2, -1, -1):
            self._rotate_joint(joint_idx, target)
    
    return CCDResult(self.joint_rotations, converged, iterations)

def _rotate_joint(self, joint_idx, target):
    joint_pos = self.joint_positions[joint_idx]
    end_pos = self._get_end_effector()
    
    # Vectors from joint to end and target
    to_end = (end_pos - joint_pos).normalized()
    to_target = (target - joint_pos).normalized()
    
    # Rotation to align
    dot = clamp(to_end.dot(to_target), -1.0, 1.0)
    axis = to_end.cross(to_target).normalized()
    angle = math.acos(dot) * self.damping
    
    # Apply rotation with limits
    rotation = Quat.from_axis_angle(axis, angle)
    combined = rotation * self.joint_rotations[joint_idx]
    self.joint_rotations[joint_idx] = self.limits[joint_idx].clamp(combined)
```

### 2.4 jacobian.py (~692 lines)

**Purpose**: Jacobian-based IK with multiple methods

```
JacobianIK
    |
    +-- joint_rotations: List[Quat]
    +-- joint_axes: List[Vec3]  # Rotation axes per DOF
    +-- method: JacobianMethod
    +-- damping: float
    |
    +-- solve(target) -> JacobianResult
    
MultiTargetJacobianIK
    |
    +-- effectors: List[EffectorConfig]
    +-- target_weights: List[float]
```

**Methods:**
```python
class JacobianMethod(Enum):
    TRANSPOSE = "transpose"
    PSEUDOINVERSE = "pseudoinverse"
    DLS = "damped_least_squares"
    SDLS = "selectively_damped_least_squares"
```

**Jacobian Computation:**
```python
def compute_jacobian(self, end_effector_pos):
    """
    J_col = axis x (end_effector - joint_pos) for rotation joints
    """
    jacobian = Matrix(3, len(self.joint_axes))
    
    for i, (joint_pos, axis) in enumerate(zip(self.joint_positions, self.joint_axes)):
        r = end_effector_pos - joint_pos
        jacobian.set_column(i, axis.cross(r))
    
    return jacobian
```

**DLS Solution:**
```python
def solve_dls(self, jacobian, error, damping):
    """
    dq = J^T * (J * J^T + lambda^2 * I)^-1 * e
    """
    J_T = jacobian.transpose()
    JJT = jacobian @ J_T
    
    # Add damping to diagonal
    damped = JJT + Matrix.identity(JJT.rows) * (damping ** 2)
    
    # Invert (Gauss-Jordan)
    damped_inv = self._invert_matrix(damped)
    
    # Compute joint deltas
    temp = damped_inv @ error
    return J_T @ temp
```

### 2.5 ik_goal.py (~569 lines)

**Purpose**: IK goal definitions and blending

```
IKGoal (abstract)
    |
    +-- PositionGoal
    +-- RotationGoal
    +-- LookAtGoal
    +-- PositionRotationGoal
    +-- PoleVectorGoal
    +-- COMGoal

IKGoalBlender
    |
    +-- goals: List[Tuple[IKGoal, float]]  # goal, weight
    |
    +-- blend() -> IKGoal
```

**Goal Types:**
```python
@dataclass
class PositionGoal(IKGoal):
    target: Vec3
    weight: float = 1.0
    
@dataclass
class RotationGoal(IKGoal):
    target: Quat
    weight: float = 1.0
    
@dataclass
class LookAtGoal(IKGoal):
    target: Vec3
    up_vector: Vec3 = Vec3(0, 1, 0)
    
@dataclass
class PoleVectorGoal(IKGoal):
    pole_position: Vec3
    twist: float = 0.0
    
@dataclass
class COMGoal(IKGoal):
    target_com: Vec3
    support_polygon: List[Vec3]
```

---

## 3. Joint Constraints

### 3.1 Constraint Types

```python
class JointConstraint:
    def apply(self, direction: Vec3, parent_rotation: Quat) -> Vec3:
        raise NotImplementedError

class BallSocketConstraint(JointConstraint):
    cone_angle: float  # Max deviation from parent axis
    
    def apply(self, direction, parent_rotation):
        ref_dir = parent_rotation.rotate_vector(Vec3(0, 1, 0))
        dot = clamp(direction.dot(ref_dir), -1, 1)
        angle = math.acos(dot)
        
        if angle <= self.cone_angle:
            return direction
        
        # Clamp to cone surface
        axis = ref_dir.cross(direction).normalized()
        rotation = Quat.from_axis_angle(axis, self.cone_angle)
        return rotation.rotate_vector(ref_dir)

class HingeConstraint(JointConstraint):
    axis: Vec3
    min_angle: float
    max_angle: float

class TwistConstraint(JointConstraint):
    min_twist: float
    max_twist: float
```

### 3.2 Joint Limits for CCD

```python
class JointLimit:
    def clamp(self, rotation: Quat) -> Quat:
        raise NotImplementedError

class EulerLimit(JointLimit):
    min_euler: Vec3
    max_euler: Vec3
    order: EulerOrder  # XYZ, YZX, ZXY, etc.
    
class SwingTwistLimit(JointLimit):
    max_swing: float
    min_twist: float
    max_twist: float
```

---

## 4. Configuration

### 4.1 config.py (~155 lines)

```python
class IKConfig:
    # Convergence
    DEFAULT_TOLERANCE = 0.001
    DEFAULT_MAX_ITERATIONS = 10
    
    # Damping
    CCD_DAMPING = 0.8
    JACOBIAN_DAMPING = 0.5
    
    # Soft IK
    SOFT_IK_FALLOFF = 5.0
    
    # Numerical stability
    EPSILON = 1e-6
    MIN_BONE_LENGTH = 0.001
```

---

## 5. Dependencies

### 5.1 External

| Dependency | Usage |
|------------|-------|
| Vec3 | Position vectors |
| Quat | Rotation quaternions |
| math.acos, math.sin, math.exp | IK calculations |

### 5.2 Internal (Phase 1)

| Dependency | From |
|------------|------|
| Transform | animation_graph.py (optional) |
