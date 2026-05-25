"""
Jacobian Computation for Constraint Solver.

This module provides utilities for computing constraint Jacobians,
effective masses, and applying impulses to rigid bodies.

A Jacobian maps body velocities to constraint velocity:
    cdot = J * v

Where v = [v_a, w_a, v_b, w_b]^T contains linear and angular velocities
of the two constrained bodies.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from .constraint_solver import RigidBody


@dataclass
class Vec3:
    """Simple 3D vector for physics calculations."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Vec3":
        return self.__mul__(scalar)

    def __neg__(self) -> "Vec3":
        return Vec3(-self.x, -self.y, -self.z)

    def __truediv__(self, scalar: float) -> "Vec3":
        if abs(scalar) < 1e-10:
            return Vec3(0.0, 0.0, 0.0)
        inv = 1.0 / scalar
        return Vec3(self.x * inv, self.y * inv, self.z * inv)

    def dot(self, other: "Vec3") -> float:
        """Compute dot product."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        """Compute cross product."""
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def length(self) -> float:
        """Compute vector length."""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def length_squared(self) -> float:
        """Compute squared length."""
        return self.x * self.x + self.y * self.y + self.z * self.z

    def normalized(self) -> "Vec3":
        """Return normalized vector."""
        length = self.length()
        if length < 1e-10:
            return Vec3(0, 0, 0)
        return self / length

    def is_zero(self, tolerance: float = 1e-10) -> bool:
        """Check if vector is approximately zero."""
        return self.length_squared() < tolerance * tolerance

    @staticmethod
    def zero() -> "Vec3":
        return Vec3(0, 0, 0)

    @staticmethod
    def unit_x() -> "Vec3":
        return Vec3(1, 0, 0)

    @staticmethod
    def unit_y() -> "Vec3":
        return Vec3(0, 1, 0)

    @staticmethod
    def unit_z() -> "Vec3":
        return Vec3(0, 0, 1)

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @classmethod
    def from_tuple(cls, t: Tuple[float, float, float]) -> "Vec3":
        return cls(t[0], t[1], t[2])


@dataclass
class Mat3:
    """3x3 matrix for inertia tensors and rotations."""
    # Row-major storage
    m00: float = 1.0
    m01: float = 0.0
    m02: float = 0.0
    m10: float = 0.0
    m11: float = 1.0
    m12: float = 0.0
    m20: float = 0.0
    m21: float = 0.0
    m22: float = 1.0

    def __mul__(self, other) -> "Vec3":
        if isinstance(other, Vec3):
            return Vec3(
                self.m00 * other.x + self.m01 * other.y + self.m02 * other.z,
                self.m10 * other.x + self.m11 * other.y + self.m12 * other.z,
                self.m20 * other.x + self.m21 * other.y + self.m22 * other.z
            )
        raise TypeError(f"Cannot multiply Mat3 with {type(other)}")

    def mat_mul(self, other: "Mat3") -> "Mat3":
        """Matrix-matrix multiplication."""
        return Mat3(
            self.m00 * other.m00 + self.m01 * other.m10 + self.m02 * other.m20,
            self.m00 * other.m01 + self.m01 * other.m11 + self.m02 * other.m21,
            self.m00 * other.m02 + self.m01 * other.m12 + self.m02 * other.m22,
            self.m10 * other.m00 + self.m11 * other.m10 + self.m12 * other.m20,
            self.m10 * other.m01 + self.m11 * other.m11 + self.m12 * other.m21,
            self.m10 * other.m02 + self.m11 * other.m12 + self.m12 * other.m22,
            self.m20 * other.m00 + self.m21 * other.m10 + self.m22 * other.m20,
            self.m20 * other.m01 + self.m21 * other.m11 + self.m22 * other.m21,
            self.m20 * other.m02 + self.m21 * other.m12 + self.m22 * other.m22,
        )

    def transpose(self) -> "Mat3":
        """Return transposed matrix."""
        return Mat3(
            self.m00, self.m10, self.m20,
            self.m01, self.m11, self.m21,
            self.m02, self.m12, self.m22
        )

    def determinant(self) -> float:
        """Compute matrix determinant."""
        return (
            self.m00 * (self.m11 * self.m22 - self.m12 * self.m21)
            - self.m01 * (self.m10 * self.m22 - self.m12 * self.m20)
            + self.m02 * (self.m10 * self.m21 - self.m11 * self.m20)
        )

    def inverse(self) -> "Mat3":
        """Compute matrix inverse."""
        det = self.determinant()
        if abs(det) < 1e-10:
            return Mat3.identity()

        inv_det = 1.0 / det
        return Mat3(
            (self.m11 * self.m22 - self.m12 * self.m21) * inv_det,
            (self.m02 * self.m21 - self.m01 * self.m22) * inv_det,
            (self.m01 * self.m12 - self.m02 * self.m11) * inv_det,
            (self.m12 * self.m20 - self.m10 * self.m22) * inv_det,
            (self.m00 * self.m22 - self.m02 * self.m20) * inv_det,
            (self.m02 * self.m10 - self.m00 * self.m12) * inv_det,
            (self.m10 * self.m21 - self.m11 * self.m20) * inv_det,
            (self.m01 * self.m20 - self.m00 * self.m21) * inv_det,
            (self.m00 * self.m11 - self.m01 * self.m10) * inv_det,
        )

    def scale(self, s: float) -> "Mat3":
        """Return scaled matrix."""
        return Mat3(
            self.m00 * s, self.m01 * s, self.m02 * s,
            self.m10 * s, self.m11 * s, self.m12 * s,
            self.m20 * s, self.m21 * s, self.m22 * s
        )

    @staticmethod
    def identity() -> "Mat3":
        return Mat3()

    @staticmethod
    def zero() -> "Mat3":
        return Mat3(0, 0, 0, 0, 0, 0, 0, 0, 0)

    @staticmethod
    def from_diagonal(x: float, y: float, z: float) -> "Mat3":
        """Create diagonal matrix from values."""
        return Mat3(x, 0, 0, 0, y, 0, 0, 0, z)

    @staticmethod
    def skew_symmetric(v: Vec3) -> "Mat3":
        """Create skew-symmetric matrix from vector."""
        return Mat3(
            0, -v.z, v.y,
            v.z, 0, -v.x,
            -v.y, v.x, 0
        )


@dataclass
class Quaternion:
    """Quaternion for rotation representation."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        """Quaternion multiplication."""
        return Quaternion(
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        )

    def conjugate(self) -> "Quaternion":
        """Return conjugate quaternion."""
        return Quaternion(-self.x, -self.y, -self.z, self.w)

    def rotate_vector(self, v: Vec3) -> Vec3:
        """Rotate a vector by this quaternion."""
        q_vec = Vec3(self.x, self.y, self.z)
        t = q_vec.cross(v) * 2.0
        return v + t * self.w + q_vec.cross(t)

    def inverse_rotate_vector(self, v: Vec3) -> Vec3:
        """Inverse rotate a vector by this quaternion."""
        return self.conjugate().rotate_vector(v)

    def to_matrix(self) -> Mat3:
        """Convert to rotation matrix."""
        xx = self.x * self.x
        yy = self.y * self.y
        zz = self.z * self.z
        xy = self.x * self.y
        xz = self.x * self.z
        yz = self.y * self.z
        wx = self.w * self.x
        wy = self.w * self.y
        wz = self.w * self.z

        return Mat3(
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)
        )

    def normalized(self) -> "Quaternion":
        """Return normalized quaternion."""
        length = math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w)
        if length < 1e-10:
            return Quaternion()
        inv_length = 1.0 / length
        return Quaternion(
            self.x * inv_length,
            self.y * inv_length,
            self.z * inv_length,
            self.w * inv_length
        )

    @staticmethod
    def identity() -> "Quaternion":
        return Quaternion(0, 0, 0, 1)

    @staticmethod
    def from_axis_angle(axis: Vec3, angle: float) -> "Quaternion":
        """Create quaternion from axis and angle (radians)."""
        # Handle zero angle or zero-length axis
        axis_length_sq = axis.length_squared()
        if axis_length_sq < 1e-10 or abs(angle) < 1e-10:
            return Quaternion.identity()

        half_angle = angle * 0.5
        s = math.sin(half_angle)
        inv_length = 1.0 / math.sqrt(axis_length_sq)
        return Quaternion(
            axis.x * inv_length * s,
            axis.y * inv_length * s,
            axis.z * inv_length * s,
            math.cos(half_angle)
        )


@dataclass
class Jacobian:
    """
    Constraint Jacobian representing how body velocities affect constraint.

    For a constraint between body A and body B:
    cdot = linear_a . v_a + angular_a . w_a + linear_b . v_b + angular_b . w_b

    Attributes:
        linear_a: Linear velocity coefficient for body A.
        angular_a: Angular velocity coefficient for body A.
        linear_b: Linear velocity coefficient for body B.
        angular_b: Angular velocity coefficient for body B.
    """
    linear_a: Vec3 = field(default_factory=Vec3.zero)
    angular_a: Vec3 = field(default_factory=Vec3.zero)
    linear_b: Vec3 = field(default_factory=Vec3.zero)
    angular_b: Vec3 = field(default_factory=Vec3.zero)

    def compute_velocity(
        self,
        vel_a: Vec3,
        ang_vel_a: Vec3,
        vel_b: Vec3,
        ang_vel_b: Vec3
    ) -> float:
        """
        Compute constraint velocity: cdot = J * v

        Args:
            vel_a: Linear velocity of body A.
            ang_vel_a: Angular velocity of body A.
            vel_b: Linear velocity of body B.
            ang_vel_b: Angular velocity of body B.

        Returns:
            Constraint velocity (scalar).
        """
        return (
            self.linear_a.dot(vel_a) +
            self.angular_a.dot(ang_vel_a) +
            self.linear_b.dot(vel_b) +
            self.angular_b.dot(ang_vel_b)
        )

    def negate(self) -> "Jacobian":
        """Return negated Jacobian."""
        return Jacobian(
            -self.linear_a,
            -self.angular_a,
            -self.linear_b,
            -self.angular_b
        )

    def scale(self, s: float) -> "Jacobian":
        """Return scaled Jacobian."""
        return Jacobian(
            self.linear_a * s,
            self.angular_a * s,
            self.linear_b * s,
            self.angular_b * s
        )


@dataclass
class JacobianPair:
    """
    Pair of Jacobians for 2D constraints (e.g., friction).

    Used for constraints that need two perpendicular directions,
    such as friction constraints.
    """
    j1: Jacobian
    j2: Jacobian


def compute_jacobian(
    constraint_type: str,
    anchor_a: Vec3,
    anchor_b: Vec3,
    axis: Optional[Vec3] = None,
    **kwargs
) -> Jacobian:
    """
    Compute Jacobian for various constraint types.

    Args:
        constraint_type: Type of constraint ("point", "axis", "angular", etc.).
        anchor_a: Anchor point in body A's local space (transformed to world).
        anchor_b: Anchor point in body B's local space (transformed to world).
        axis: Constraint axis in world space (for axis constraints).
        **kwargs: Additional parameters for specific constraint types.

    Returns:
        Jacobian for the constraint.
    """
    if constraint_type == "point" or constraint_type == "ball":
        # Point-to-point constraint (distance)
        # Direction from anchor_a to anchor_b
        direction = anchor_b - anchor_a
        length = direction.length()
        if length < 1e-10:
            direction = Vec3.unit_x()
        else:
            direction = direction / length

        # r_a and r_b are vectors from body centers to anchors (in world space)
        r_a = kwargs.get("r_a", Vec3.zero())
        r_b = kwargs.get("r_b", Vec3.zero())

        return Jacobian(
            linear_a=-direction,
            angular_a=-r_a.cross(direction),
            linear_b=direction,
            angular_b=r_b.cross(direction)
        )

    elif constraint_type == "axis" or constraint_type == "hinge":
        # Axis constraint - maintain axis alignment
        if axis is None:
            axis = Vec3.unit_x()

        r_a = kwargs.get("r_a", Vec3.zero())
        r_b = kwargs.get("r_b", Vec3.zero())

        return Jacobian(
            linear_a=-axis,
            angular_a=-r_a.cross(axis),
            linear_b=axis,
            angular_b=r_b.cross(axis)
        )

    elif constraint_type == "angular":
        # Pure angular constraint
        if axis is None:
            axis = Vec3.unit_x()

        return Jacobian(
            linear_a=Vec3.zero(),
            angular_a=-axis,
            linear_b=Vec3.zero(),
            angular_b=axis
        )

    elif constraint_type == "limit":
        # Limit constraint (one-sided)
        if axis is None:
            axis = Vec3.unit_x()

        r_a = kwargs.get("r_a", Vec3.zero())
        r_b = kwargs.get("r_b", Vec3.zero())

        return Jacobian(
            linear_a=-axis,
            angular_a=-r_a.cross(axis),
            linear_b=axis,
            angular_b=r_b.cross(axis)
        )

    elif constraint_type == "contact":
        # Contact constraint - normal direction
        normal = axis if axis is not None else Vec3.unit_y()
        r_a = kwargs.get("r_a", Vec3.zero())
        r_b = kwargs.get("r_b", Vec3.zero())

        return Jacobian(
            linear_a=-normal,
            angular_a=-r_a.cross(normal),
            linear_b=normal,
            angular_b=r_b.cross(normal)
        )

    elif constraint_type == "friction":
        # Friction constraint - tangent direction
        tangent = axis if axis is not None else Vec3.unit_x()
        r_a = kwargs.get("r_a", Vec3.zero())
        r_b = kwargs.get("r_b", Vec3.zero())

        return Jacobian(
            linear_a=-tangent,
            angular_a=-r_a.cross(tangent),
            linear_b=tangent,
            angular_b=r_b.cross(tangent)
        )

    elif constraint_type == "distance":
        # Fixed distance constraint
        direction = anchor_b - anchor_a
        length = direction.length()
        if length < 1e-10:
            direction = Vec3.unit_x()
        else:
            direction = direction / length

        r_a = kwargs.get("r_a", Vec3.zero())
        r_b = kwargs.get("r_b", Vec3.zero())

        return Jacobian(
            linear_a=-direction,
            angular_a=-r_a.cross(direction),
            linear_b=direction,
            angular_b=r_b.cross(direction)
        )

    else:
        raise ValueError(f"Unknown constraint type: {constraint_type}")


def compute_effective_mass(
    jacobian: Jacobian,
    inv_mass_a: float,
    inv_inertia_a: Mat3,
    inv_mass_b: float,
    inv_inertia_b: Mat3
) -> float:
    """
    Compute effective mass for a constraint.

    The effective mass is: K = J * M^-1 * J^T

    Where M^-1 is the inverse mass matrix and J is the Jacobian.

    Args:
        jacobian: Constraint Jacobian.
        inv_mass_a: Inverse mass of body A.
        inv_inertia_a: Inverse inertia tensor of body A (world space).
        inv_mass_b: Inverse mass of body B.
        inv_inertia_b: Inverse inertia tensor of body B (world space).

    Returns:
        Effective mass (scalar). Returns inf if both bodies are static.
    """
    # K = J_la . (1/m_a) . J_la + J_aa . I_a^-1 . J_aa
    #   + J_lb . (1/m_b) . J_lb + J_ab . I_b^-1 . J_ab

    k = 0.0

    # Body A contribution
    if inv_mass_a > 0:
        k += inv_mass_a * jacobian.linear_a.dot(jacobian.linear_a)

    ang_a_contrib = inv_inertia_a * jacobian.angular_a
    k += jacobian.angular_a.dot(ang_a_contrib)

    # Body B contribution
    if inv_mass_b > 0:
        k += inv_mass_b * jacobian.linear_b.dot(jacobian.linear_b)

    ang_b_contrib = inv_inertia_b * jacobian.angular_b
    k += jacobian.angular_b.dot(ang_b_contrib)

    # Return inverse (effective mass)
    # When k is too small, both bodies are effectively static - return 0 to skip constraint
    if k < 1e-10:
        return 0.0

    return 1.0 / k


def compute_effective_mass_matrix(
    jacobians: List[Jacobian],
    inv_mass_a: float,
    inv_inertia_a: Mat3,
    inv_mass_b: float,
    inv_inertia_b: Mat3
) -> List[List[float]]:
    """
    Compute effective mass matrix for multiple constraints.

    Used for block solving of multiple coupled constraints.

    Args:
        jacobians: List of constraint Jacobians.
        inv_mass_a: Inverse mass of body A.
        inv_inertia_a: Inverse inertia tensor of body A.
        inv_mass_b: Inverse mass of body B.
        inv_inertia_b: Inverse inertia tensor of body B.

    Returns:
        NxN effective mass matrix where N = len(jacobians).
    """
    n = len(jacobians)
    k_matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            ji = jacobians[i]
            jj = jacobians[j]

            k = 0.0

            # Linear contribution
            if inv_mass_a > 0:
                k += inv_mass_a * ji.linear_a.dot(jj.linear_a)
            if inv_mass_b > 0:
                k += inv_mass_b * ji.linear_b.dot(jj.linear_b)

            # Angular contribution
            k += ji.angular_a.dot(inv_inertia_a * jj.angular_a)
            k += ji.angular_b.dot(inv_inertia_b * jj.angular_b)

            k_matrix[i][j] = k

    return k_matrix


def apply_impulse(
    jacobian: Jacobian,
    impulse: float,
    inv_mass_a: float,
    inv_inertia_a: Mat3,
    inv_mass_b: float,
    inv_inertia_b: Mat3
) -> Tuple[Vec3, Vec3, Vec3, Vec3]:
    """
    Apply constraint impulse to bodies.

    The impulse changes velocities: delta_v = M^-1 * J^T * lambda

    Args:
        jacobian: Constraint Jacobian.
        impulse: Impulse magnitude (lambda).
        inv_mass_a: Inverse mass of body A.
        inv_inertia_a: Inverse inertia tensor of body A.
        inv_mass_b: Inverse mass of body B.
        inv_inertia_b: Inverse inertia tensor of body B.

    Returns:
        Tuple of (delta_vel_a, delta_ang_vel_a, delta_vel_b, delta_ang_vel_b).
    """
    # delta_v_a = (1/m_a) * J_la^T * lambda
    delta_vel_a = jacobian.linear_a * (inv_mass_a * impulse)

    # delta_w_a = I_a^-1 * J_aa^T * lambda
    delta_ang_vel_a = inv_inertia_a * (jacobian.angular_a * impulse)

    # delta_v_b = (1/m_b) * J_lb^T * lambda
    delta_vel_b = jacobian.linear_b * (inv_mass_b * impulse)

    # delta_w_b = I_b^-1 * J_ab^T * lambda
    delta_ang_vel_b = inv_inertia_b * (jacobian.angular_b * impulse)

    return delta_vel_a, delta_ang_vel_a, delta_vel_b, delta_ang_vel_b


def compute_relative_velocity(
    jacobian: Jacobian,
    vel_a: Vec3,
    ang_vel_a: Vec3,
    vel_b: Vec3,
    ang_vel_b: Vec3
) -> float:
    """
    Compute relative velocity along constraint direction.

    This is the constraint velocity: cdot = J * v

    Args:
        jacobian: Constraint Jacobian.
        vel_a: Linear velocity of body A.
        ang_vel_a: Angular velocity of body A.
        vel_b: Linear velocity of body B.
        ang_vel_b: Angular velocity of body B.

    Returns:
        Relative velocity scalar.
    """
    return jacobian.compute_velocity(vel_a, ang_vel_a, vel_b, ang_vel_b)


def compute_position_error(
    anchor_a_world: Vec3,
    anchor_b_world: Vec3,
    axis: Optional[Vec3] = None
) -> float:
    """
    Compute position error for constraint.

    Args:
        anchor_a_world: Anchor A position in world space.
        anchor_b_world: Anchor B position in world space.
        axis: Optional axis to project error onto.

    Returns:
        Position error (scalar).
    """
    error = anchor_b_world - anchor_a_world

    if axis is not None:
        return error.dot(axis)
    else:
        return error.length()


def compute_angular_error(
    orientation_a: Quaternion,
    orientation_b: Quaternion,
    axis: Optional[Vec3] = None
) -> float:
    """
    Compute angular error between two orientations.

    Args:
        orientation_a: Orientation of body A.
        orientation_b: Orientation of body B.
        axis: Optional axis to measure rotation around.

    Returns:
        Angular error in radians.
    """
    # Relative rotation: q_rel = q_b * q_a^-1
    q_rel = orientation_b * orientation_a.conjugate()

    # Extract angle from quaternion
    # angle = 2 * atan2(|q.xyz|, q.w)
    q_vec = Vec3(q_rel.x, q_rel.y, q_rel.z)
    sin_half = q_vec.length()

    if sin_half < 1e-10:
        return 0.0

    angle = 2.0 * math.atan2(sin_half, q_rel.w)

    # Normalize to [-pi, pi]
    if angle > math.pi:
        angle -= 2.0 * math.pi
    elif angle < -math.pi:
        angle += 2.0 * math.pi

    if axis is not None:
        # Project onto axis
        rotation_axis = q_vec.normalized()
        return angle * rotation_axis.dot(axis)

    return angle


def compute_friction_basis(normal: Vec3) -> Tuple[Vec3, Vec3]:
    """
    Compute tangent basis for friction from contact normal.

    Args:
        normal: Contact normal (normalized).

    Returns:
        Tuple of two tangent vectors perpendicular to normal.
    """
    # Find a vector not parallel to normal
    if abs(normal.x) < 0.9:
        arbitrary = Vec3.unit_x()
    else:
        arbitrary = Vec3.unit_y()

    # Compute tangents
    tangent1 = normal.cross(arbitrary).normalized()
    tangent2 = normal.cross(tangent1).normalized()

    return tangent1, tangent2


def clamp_impulse(
    impulse: float,
    accumulated: float,
    lower: float,
    upper: float
) -> Tuple[float, float]:
    """
    Clamp impulse with accumulator pattern.

    Used for inequality constraints where impulse must be in [lower, upper].

    Args:
        impulse: New impulse to apply.
        accumulated: Previously accumulated impulse.
        lower: Lower bound.
        upper: Upper bound.

    Returns:
        Tuple of (clamped_impulse, new_accumulated).
    """
    new_accumulated = max(lower, min(upper, accumulated + impulse))
    clamped_impulse = new_accumulated - accumulated
    return clamped_impulse, new_accumulated


def solve_2x2(a: List[List[float]], b: List[float]) -> List[float]:
    """
    Solve 2x2 linear system Ax = b.

    Args:
        a: 2x2 coefficient matrix.
        b: Right-hand side vector.

    Returns:
        Solution vector x.
    """
    det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    if abs(det) < 1e-10:
        return [0.0, 0.0]

    inv_det = 1.0 / det
    return [
        (a[1][1] * b[0] - a[0][1] * b[1]) * inv_det,
        (a[0][0] * b[1] - a[1][0] * b[0]) * inv_det
    ]


def solve_3x3(a: List[List[float]], b: List[float]) -> List[float]:
    """
    Solve 3x3 linear system Ax = b using Cramer's rule.

    Args:
        a: 3x3 coefficient matrix.
        b: Right-hand side vector.

    Returns:
        Solution vector x.
    """
    det = (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )

    if abs(det) < 1e-10:
        return [0.0, 0.0, 0.0]

    inv_det = 1.0 / det

    x0 = (
        b[0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (b[1] * a[2][2] - a[1][2] * b[2])
        + a[0][2] * (b[1] * a[2][1] - a[1][1] * b[2])
    ) * inv_det

    x1 = (
        a[0][0] * (b[1] * a[2][2] - a[1][2] * b[2])
        - b[0] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * b[2] - b[1] * a[2][0])
    ) * inv_det

    x2 = (
        a[0][0] * (a[1][1] * b[2] - b[1] * a[2][1])
        - a[0][1] * (a[1][0] * b[2] - b[1] * a[2][0])
        + b[0] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    ) * inv_det

    return [x0, x1, x2]
