"""Deterministic skeleton blending using Fixed32 arithmetic.

This module provides bit-identical animation blending across platforms
by using Q16.16 fixed-point arithmetic for all position, rotation,
and blending operations.

Key types:
- Fixed32Vec3: Position vector using Fixed32 components
- Fixed32Quat: Quaternion using Fixed32 components
- Fixed32BoneTransform: Complete bone transform with Fixed32 precision
- Fixed32Pose: Deterministic pose representation
- DeterministicBoneBlend: Blending engine with Fixed32 weights

All operations guarantee identical results on any platform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, List, Optional, Tuple

from trinity.types import Fixed32
from trinity.constants import FIXED32_SCALE, FIXED32_SHIFT

if TYPE_CHECKING:
    from engine.animation.skeletal.skeleton import Skeleton


# =============================================================================
# Constants
# =============================================================================

# Fixed32 versions of common values
FIXED32_ZERO = Fixed32(0)
FIXED32_ONE = Fixed32(1)
FIXED32_HALF = Fixed32(0.5)
FIXED32_TWO = Fixed32(2)
FIXED32_NEG_ONE = Fixed32(-1)

# Thresholds for numerical stability (in Fixed32)
FIXED32_EPSILON = Fixed32.from_raw(1)  # Smallest representable positive
FIXED32_SLERP_THRESHOLD = Fixed32(0.9995)  # Dot product threshold for nlerp fallback
FIXED32_SCALE_EPSILON = Fixed32(0.0001)  # Minimum scale for division

# Pre-computed trig table size for deterministic sin/cos
TRIG_TABLE_SIZE = 4096
TRIG_TABLE_MASK = TRIG_TABLE_SIZE - 1


# =============================================================================
# Deterministic Trigonometry (Fixed32)
# =============================================================================

def _build_sin_table() -> List[Fixed32]:
    """Build pre-computed sine table for deterministic trig."""
    import math
    table = []
    for i in range(TRIG_TABLE_SIZE):
        angle = (i / TRIG_TABLE_SIZE) * 2.0 * math.pi
        table.append(Fixed32(math.sin(angle)))
    return table

def _build_cos_table() -> List[Fixed32]:
    """Build pre-computed cosine table for deterministic trig."""
    import math
    table = []
    for i in range(TRIG_TABLE_SIZE):
        angle = (i / TRIG_TABLE_SIZE) * 2.0 * math.pi
        table.append(Fixed32(math.cos(angle)))
    return table

# Lazy initialization of trig tables
_SIN_TABLE: Optional[List[Fixed32]] = None
_COS_TABLE: Optional[List[Fixed32]] = None

def _get_sin_table() -> List[Fixed32]:
    global _SIN_TABLE
    if _SIN_TABLE is None:
        _SIN_TABLE = _build_sin_table()
    return _SIN_TABLE

def _get_cos_table() -> List[Fixed32]:
    global _COS_TABLE
    if _COS_TABLE is None:
        _COS_TABLE = _build_cos_table()
    return _COS_TABLE


def fixed32_sin(angle: Fixed32) -> Fixed32:
    """Deterministic sine using lookup table.

    Args:
        angle: Angle in radians (Fixed32).

    Returns:
        Sine value as Fixed32.
    """
    import math
    # Normalize angle to [0, 2*pi)
    two_pi = Fixed32(2.0 * math.pi)
    while angle < FIXED32_ZERO:
        angle = angle + two_pi
    while angle >= two_pi:
        angle = angle - two_pi

    # Convert to table index
    table = _get_sin_table()
    fraction = angle / two_pi
    index = int(fraction.as_float * TRIG_TABLE_SIZE) & TRIG_TABLE_MASK
    return table[index]


def fixed32_cos(angle: Fixed32) -> Fixed32:
    """Deterministic cosine using lookup table.

    Args:
        angle: Angle in radians (Fixed32).

    Returns:
        Cosine value as Fixed32.
    """
    import math
    # Normalize angle to [0, 2*pi)
    two_pi = Fixed32(2.0 * math.pi)
    while angle < FIXED32_ZERO:
        angle = angle + two_pi
    while angle >= two_pi:
        angle = angle - two_pi

    # Convert to table index
    table = _get_cos_table()
    fraction = angle / two_pi
    index = int(fraction.as_float * TRIG_TABLE_SIZE) & TRIG_TABLE_MASK
    return table[index]


def fixed32_sqrt(value: Fixed32) -> Fixed32:
    """Deterministic square root using Newton-Raphson.

    Args:
        value: Non-negative Fixed32 value.

    Returns:
        Square root as Fixed32.
    """
    if value <= FIXED32_ZERO:
        return FIXED32_ZERO

    # For fixed-point sqrt, we need a good initial guess
    # Convert to float for initial approximation, then refine
    import math
    float_val = value.as_float
    if float_val <= 0:
        return FIXED32_ZERO

    # Initial guess from float sqrt
    initial_float = math.sqrt(float_val)
    x = Fixed32(initial_float)

    # Newton-Raphson iterations for precision refinement
    # x_new = (x + value/x) / 2
    for _ in range(3):
        if x <= FIXED32_ZERO:
            break
        x = (x + value / x) * FIXED32_HALF

    return x


def fixed32_acos(value: Fixed32) -> Fixed32:
    """Deterministic arc cosine approximation.

    Uses polynomial approximation for determinism.

    Args:
        value: Value in [-1, 1] range.

    Returns:
        Angle in radians [0, pi].
    """
    import math

    # Clamp to valid range
    if value >= FIXED32_ONE:
        return FIXED32_ZERO
    if value <= FIXED32_NEG_ONE:
        return Fixed32(math.pi)

    # Use polynomial approximation
    # acos(x) ~= pi/2 - asin(x)
    # asin(x) ~= x + x^3/6 + 3*x^5/40 + ...
    x = value
    x2 = x * x
    x3 = x2 * x
    x5 = x3 * x2

    # Coefficients as Fixed32
    c1 = Fixed32(1.0 / 6.0)
    c2 = Fixed32(3.0 / 40.0)

    asin_approx = x + x3 * c1 + x5 * c2
    pi_half = Fixed32(math.pi / 2.0)

    return pi_half - asin_approx


# =============================================================================
# Fixed32 Vector Types
# =============================================================================

@dataclass(slots=True)
class Fixed32Vec3:
    """3D vector using Fixed32 components for deterministic math.

    All operations use integer arithmetic through Fixed32, ensuring
    identical results across all platforms.
    """

    x: Fixed32 = field(default_factory=lambda: FIXED32_ZERO)
    y: Fixed32 = field(default_factory=lambda: FIXED32_ZERO)
    z: Fixed32 = field(default_factory=lambda: FIXED32_ZERO)

    @staticmethod
    def zero() -> Fixed32Vec3:
        """Create zero vector."""
        return Fixed32Vec3(FIXED32_ZERO, FIXED32_ZERO, FIXED32_ZERO)

    @staticmethod
    def one() -> Fixed32Vec3:
        """Create unit vector (1, 1, 1)."""
        return Fixed32Vec3(FIXED32_ONE, FIXED32_ONE, FIXED32_ONE)

    @staticmethod
    def from_floats(x: float, y: float, z: float) -> Fixed32Vec3:
        """Create from float values."""
        return Fixed32Vec3(Fixed32(x), Fixed32(y), Fixed32(z))

    def __add__(self, other: Fixed32Vec3) -> Fixed32Vec3:
        return Fixed32Vec3(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )

    def __sub__(self, other: Fixed32Vec3) -> Fixed32Vec3:
        return Fixed32Vec3(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
        )

    def __mul__(self, scalar: Fixed32) -> Fixed32Vec3:
        return Fixed32Vec3(
            self.x * scalar,
            self.y * scalar,
            self.z * scalar,
        )

    def __neg__(self) -> Fixed32Vec3:
        return Fixed32Vec3(-self.x, -self.y, -self.z)

    def dot(self, other: Fixed32Vec3) -> Fixed32:
        """Dot product."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Fixed32Vec3) -> Fixed32Vec3:
        """Cross product."""
        return Fixed32Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length_squared(self) -> Fixed32:
        """Squared length (avoids sqrt)."""
        return self.dot(self)

    def length(self) -> Fixed32:
        """Vector length."""
        return fixed32_sqrt(self.length_squared())

    def normalized(self) -> Fixed32Vec3:
        """Unit vector in same direction."""
        ln = self.length()
        if ln <= FIXED32_EPSILON:
            return Fixed32Vec3.zero()
        return Fixed32Vec3(
            self.x / ln,
            self.y / ln,
            self.z / ln,
        )

    def lerp(self, other: Fixed32Vec3, t: Fixed32) -> Fixed32Vec3:
        """Linear interpolation."""
        one_minus_t = FIXED32_ONE - t
        return Fixed32Vec3(
            self.x * one_minus_t + other.x * t,
            self.y * one_minus_t + other.y * t,
            self.z * one_minus_t + other.z * t,
        )

    def as_tuple(self) -> Tuple[float, float, float]:
        """Convert to float tuple for display/debugging."""
        return (self.x.as_float, self.y.as_float, self.z.as_float)

    def raw_tuple(self) -> Tuple[int, int, int]:
        """Get raw integer representation for comparison."""
        return (self.x.raw, self.y.raw, self.z.raw)

    def copy(self) -> Fixed32Vec3:
        """Create a copy."""
        return Fixed32Vec3(
            Fixed32.from_raw(self.x.raw),
            Fixed32.from_raw(self.y.raw),
            Fixed32.from_raw(self.z.raw),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Fixed32Vec3):
            return NotImplemented
        return (
            self.x.raw == other.x.raw and
            self.y.raw == other.y.raw and
            self.z.raw == other.z.raw
        )

    def __hash__(self) -> int:
        return hash((self.x.raw, self.y.raw, self.z.raw))

    def __repr__(self) -> str:
        return f"Fixed32Vec3({self.x}, {self.y}, {self.z})"


@dataclass(slots=True)
class Fixed32Quat:
    """Quaternion using Fixed32 components for deterministic rotations.

    Represents rotation as (x, y, z, w) where w is the scalar part.
    All operations use fixed-point arithmetic.
    """

    x: Fixed32 = field(default_factory=lambda: FIXED32_ZERO)
    y: Fixed32 = field(default_factory=lambda: FIXED32_ZERO)
    z: Fixed32 = field(default_factory=lambda: FIXED32_ZERO)
    w: Fixed32 = field(default_factory=lambda: FIXED32_ONE)

    @staticmethod
    def identity() -> Fixed32Quat:
        """Create identity quaternion (no rotation)."""
        return Fixed32Quat(FIXED32_ZERO, FIXED32_ZERO, FIXED32_ZERO, FIXED32_ONE)

    @staticmethod
    def from_floats(x: float, y: float, z: float, w: float) -> Fixed32Quat:
        """Create from float values."""
        return Fixed32Quat(Fixed32(x), Fixed32(y), Fixed32(z), Fixed32(w))

    @staticmethod
    def from_axis_angle(axis: Fixed32Vec3, angle: Fixed32) -> Fixed32Quat:
        """Create from axis and angle (radians)."""
        half_angle = angle * FIXED32_HALF
        s = fixed32_sin(half_angle)
        c = fixed32_cos(half_angle)
        axis_n = axis.normalized()
        return Fixed32Quat(
            axis_n.x * s,
            axis_n.y * s,
            axis_n.z * s,
            c,
        )

    def __mul__(self, other: Fixed32Quat) -> Fixed32Quat:
        """Quaternion multiplication (composition of rotations)."""
        return Fixed32Quat(
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
        )

    def conjugate(self) -> Fixed32Quat:
        """Conjugate (inverse for unit quaternion)."""
        return Fixed32Quat(-self.x, -self.y, -self.z, self.w)

    def dot(self, other: Fixed32Quat) -> Fixed32:
        """Dot product."""
        return (
            self.x * other.x +
            self.y * other.y +
            self.z * other.z +
            self.w * other.w
        )

    def length_squared(self) -> Fixed32:
        """Squared length."""
        return self.dot(self)

    def length(self) -> Fixed32:
        """Length (should be 1 for unit quaternion)."""
        return fixed32_sqrt(self.length_squared())

    def normalized(self) -> Fixed32Quat:
        """Normalize to unit quaternion."""
        ln = self.length()
        if ln <= FIXED32_EPSILON:
            return Fixed32Quat.identity()
        return Fixed32Quat(
            self.x / ln,
            self.y / ln,
            self.z / ln,
            self.w / ln,
        )

    def inverse(self) -> Fixed32Quat:
        """Inverse quaternion."""
        ls = self.length_squared()
        if ls <= FIXED32_EPSILON:
            return Fixed32Quat.identity()
        return Fixed32Quat(
            -self.x / ls,
            -self.y / ls,
            -self.z / ls,
            self.w / ls,
        )

    def nlerp(self, other: Fixed32Quat, t: Fixed32) -> Fixed32Quat:
        """Normalized linear interpolation (faster, good for small angles)."""
        one_minus_t = FIXED32_ONE - t

        # Handle antipodal quaternions
        d = self.dot(other)
        o = other
        if d < FIXED32_ZERO:
            o = Fixed32Quat(-other.x, -other.y, -other.z, -other.w)

        result = Fixed32Quat(
            self.x * one_minus_t + o.x * t,
            self.y * one_minus_t + o.y * t,
            self.z * one_minus_t + o.z * t,
            self.w * one_minus_t + o.w * t,
        )
        return result.normalized()

    def slerp(self, other: Fixed32Quat, t: Fixed32) -> Fixed32Quat:
        """Spherical linear interpolation (constant angular velocity)."""
        d = self.dot(other)

        # Handle antipodal quaternions (shortest path)
        o = other
        if d < FIXED32_ZERO:
            o = Fixed32Quat(-other.x, -other.y, -other.z, -other.w)
            d = -d

        # Fall back to nlerp for very close quaternions
        if d > FIXED32_SLERP_THRESHOLD:
            return self.nlerp(o, t)

        # Clamp dot product
        if d > FIXED32_ONE:
            d = FIXED32_ONE

        # Compute interpolation
        theta = fixed32_acos(d)
        sin_theta = fixed32_sin(theta)

        if sin_theta <= FIXED32_EPSILON:
            return self.nlerp(o, t)

        one_minus_t = FIXED32_ONE - t
        s0 = fixed32_sin(theta * one_minus_t) / sin_theta
        s1 = fixed32_sin(theta * t) / sin_theta

        return Fixed32Quat(
            self.x * s0 + o.x * s1,
            self.y * s0 + o.y * s1,
            self.z * s0 + o.z * s1,
            self.w * s0 + o.w * s1,
        )

    def rotate_vector(self, v: Fixed32Vec3) -> Fixed32Vec3:
        """Rotate a vector by this quaternion."""
        qv = Fixed32Vec3(self.x, self.y, self.z)
        uv = qv.cross(v)
        uuv = qv.cross(uv)
        return v + (uv * self.w + uuv) * FIXED32_TWO

    def raw_tuple(self) -> Tuple[int, int, int, int]:
        """Get raw integer representation for comparison."""
        return (self.x.raw, self.y.raw, self.z.raw, self.w.raw)

    def copy(self) -> Fixed32Quat:
        """Create a copy."""
        return Fixed32Quat(
            Fixed32.from_raw(self.x.raw),
            Fixed32.from_raw(self.y.raw),
            Fixed32.from_raw(self.z.raw),
            Fixed32.from_raw(self.w.raw),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Fixed32Quat):
            return NotImplemented
        return (
            self.x.raw == other.x.raw and
            self.y.raw == other.y.raw and
            self.z.raw == other.z.raw and
            self.w.raw == other.w.raw
        )

    def __hash__(self) -> int:
        return hash((self.x.raw, self.y.raw, self.z.raw, self.w.raw))

    def __repr__(self) -> str:
        return f"Fixed32Quat({self.x}, {self.y}, {self.z}, {self.w})"


# =============================================================================
# Fixed32 Bone Transform
# =============================================================================

@dataclass(slots=True)
class Fixed32BoneTransform:
    """Complete bone transform using Fixed32 arithmetic.

    Includes translation, rotation, and scale for a single bone.
    All components use fixed-point for deterministic results.
    """

    translation: Fixed32Vec3 = field(default_factory=Fixed32Vec3.zero)
    rotation: Fixed32Quat = field(default_factory=Fixed32Quat.identity)
    scale: Fixed32Vec3 = field(default_factory=Fixed32Vec3.one)

    @staticmethod
    def identity() -> Fixed32BoneTransform:
        """Create identity transform."""
        return Fixed32BoneTransform(
            translation=Fixed32Vec3.zero(),
            rotation=Fixed32Quat.identity(),
            scale=Fixed32Vec3.one(),
        )

    @staticmethod
    def from_floats(
        tx: float, ty: float, tz: float,
        rx: float, ry: float, rz: float, rw: float,
        sx: float, sy: float, sz: float,
    ) -> Fixed32BoneTransform:
        """Create from float values."""
        return Fixed32BoneTransform(
            translation=Fixed32Vec3.from_floats(tx, ty, tz),
            rotation=Fixed32Quat.from_floats(rx, ry, rz, rw),
            scale=Fixed32Vec3.from_floats(sx, sy, sz),
        )

    def lerp(self, other: Fixed32BoneTransform, t: Fixed32) -> Fixed32BoneTransform:
        """Interpolate to another transform.

        Uses linear interpolation for translation and scale,
        spherical interpolation for rotation.
        """
        return Fixed32BoneTransform(
            translation=self.translation.lerp(other.translation, t),
            rotation=self.rotation.slerp(other.rotation, t),
            scale=self.scale.lerp(other.scale, t),
        )

    def copy(self) -> Fixed32BoneTransform:
        """Create a deep copy."""
        return Fixed32BoneTransform(
            translation=self.translation.copy(),
            rotation=self.rotation.copy(),
            scale=self.scale.copy(),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Fixed32BoneTransform):
            return NotImplemented
        return (
            self.translation == other.translation and
            self.rotation == other.rotation and
            self.scale == other.scale
        )

    def __repr__(self) -> str:
        return (
            f"Fixed32BoneTransform(t={self.translation}, "
            f"r={self.rotation}, s={self.scale})"
        )


# =============================================================================
# Fixed32 Pose
# =============================================================================

class Fixed32PoseSpace(Enum):
    """Coordinate space for pose transforms."""
    LOCAL = auto()  # Relative to parent bone
    MODEL = auto()  # Relative to skeleton root


@dataclass
class Fixed32Pose:
    """Complete pose using Fixed32 transforms for all bones.

    Represents the transforms of all bones in a skeleton at a single
    point in time, using deterministic fixed-point arithmetic.
    """

    skeleton: Skeleton
    space: Fixed32PoseSpace = Fixed32PoseSpace.LOCAL
    _transforms: List[Fixed32BoneTransform] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize transforms if not provided."""
        if not self._transforms:
            self._transforms = [
                Fixed32BoneTransform.identity()
                for _ in range(self.skeleton.bone_count)
            ]
        elif len(self._transforms) != self.skeleton.bone_count:
            raise ValueError(
                f"Transform count ({len(self._transforms)}) must match "
                f"bone count ({self.skeleton.bone_count})"
            )

    @property
    def bone_count(self) -> int:
        """Get number of bones."""
        return len(self._transforms)

    def get_bone_transform(self, index: int) -> Fixed32BoneTransform:
        """Get transform for a bone by index."""
        if index < 0 or index >= len(self._transforms):
            raise IndexError(f"Bone index {index} out of range")
        return self._transforms[index].copy()

    def set_bone_transform(self, index: int, transform: Fixed32BoneTransform) -> None:
        """Set transform for a bone by index."""
        if index < 0 or index >= len(self._transforms):
            raise IndexError(f"Bone index {index} out of range")
        self._transforms[index] = transform.copy()

    def get_bone_transform_by_name(self, name: str) -> Optional[Fixed32BoneTransform]:
        """Get transform by bone name."""
        bone = self.skeleton.get_bone_by_name(name)
        if bone is None:
            return None
        return self._transforms[bone.index].copy()

    def copy(self) -> Fixed32Pose:
        """Create a deep copy."""
        return Fixed32Pose(
            skeleton=self.skeleton,
            space=self.space,
            _transforms=[t.copy() for t in self._transforms],
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Fixed32Pose):
            return NotImplemented
        if self.skeleton is not other.skeleton:
            return False
        if self.space != other.space:
            return False
        return all(
            a == b for a, b in zip(self._transforms, other._transforms)
        )

    def __repr__(self) -> str:
        return (
            f"Fixed32Pose(skeleton='{self.skeleton.name}', "
            f"bones={len(self._transforms)}, space={self.space.name})"
        )


# =============================================================================
# Deterministic Bone Blending
# =============================================================================

class DeterministicBlendMode(Enum):
    """Blending mode for deterministic operations."""
    OVERRIDE = auto()  # Replace base with blended value
    ADDITIVE = auto()  # Add delta on top of base
    MULTIPLY = auto()  # Multiply base by factor


@dataclass(slots=True)
class Fixed32BoneMask:
    """Mask for selective bone blending using Fixed32 weights."""

    _weights: dict[int, Fixed32] = field(default_factory=dict)
    default_weight: Fixed32 = field(default_factory=lambda: FIXED32_ZERO)

    def get_weight(self, bone_index: int) -> Fixed32:
        """Get weight for a bone."""
        return self._weights.get(bone_index, self.default_weight)

    def set_weight(self, bone_index: int, weight: Fixed32) -> None:
        """Set weight for a bone (clamped to [0, 1])."""
        if weight < FIXED32_ZERO:
            weight = FIXED32_ZERO
        elif weight > FIXED32_ONE:
            weight = FIXED32_ONE
        self._weights[bone_index] = weight

    def include_bone(self, bone_index: int) -> None:
        """Include bone with full weight."""
        self._weights[bone_index] = FIXED32_ONE

    def exclude_bone(self, bone_index: int) -> None:
        """Exclude bone (weight = 0)."""
        self._weights[bone_index] = FIXED32_ZERO

    def include_all(self, skeleton: Skeleton) -> None:
        """Include all bones with full weight."""
        for i in range(skeleton.bone_count):
            self._weights[i] = FIXED32_ONE

    @staticmethod
    def full_body(skeleton: Skeleton) -> Fixed32BoneMask:
        """Create mask with all bones at full weight."""
        mask = Fixed32BoneMask(default_weight=FIXED32_ONE)
        for i in range(skeleton.bone_count):
            mask._weights[i] = FIXED32_ONE
        return mask

    def copy(self) -> Fixed32BoneMask:
        """Create a copy."""
        return Fixed32BoneMask(
            _weights=dict(self._weights),
            default_weight=Fixed32.from_raw(self.default_weight.raw),
        )


class DeterministicBoneBlend:
    """Deterministic blending engine using Fixed32 arithmetic.

    All operations produce bit-identical results across platforms.
    Supports override, additive, and multiplicative blending.
    """

    def __init__(self, skeleton: Skeleton) -> None:
        """Initialize blender for a skeleton.

        Args:
            skeleton: The skeleton to blend poses for.
        """
        self._skeleton = skeleton

    @property
    def skeleton(self) -> Skeleton:
        """Get the skeleton."""
        return self._skeleton

    def blend_poses(
        self,
        pose_a: Fixed32Pose,
        pose_b: Fixed32Pose,
        alpha: Fixed32,
        mode: DeterministicBlendMode = DeterministicBlendMode.OVERRIDE,
        mask: Optional[Fixed32BoneMask] = None,
    ) -> Fixed32Pose:
        """Blend two poses together.

        Args:
            pose_a: First pose (base).
            pose_b: Second pose (target).
            alpha: Blend factor [0, 1] as Fixed32.
            mode: Blending mode.
            mask: Optional bone mask for selective blending.

        Returns:
            Blended pose.

        Raises:
            ValueError: If poses have different skeletons.
        """
        if pose_a.skeleton is not pose_b.skeleton:
            raise ValueError("Cannot blend poses with different skeletons")
        if pose_a.skeleton is not self._skeleton:
            raise ValueError("Poses must use this blender's skeleton")

        # Clamp alpha
        if alpha < FIXED32_ZERO:
            alpha = FIXED32_ZERO
        elif alpha > FIXED32_ONE:
            alpha = FIXED32_ONE

        # Fast paths
        if alpha == FIXED32_ZERO:
            return pose_a.copy()
        if alpha == FIXED32_ONE and mode == DeterministicBlendMode.OVERRIDE and mask is None:
            return pose_b.copy()

        result_transforms = []

        for bone_idx in range(pose_a.bone_count):
            transform_a = pose_a._transforms[bone_idx]
            transform_b = pose_b._transforms[bone_idx]

            # Apply mask
            effective_alpha = alpha
            if mask is not None:
                mask_weight = mask.get_weight(bone_idx)
                effective_alpha = alpha * mask_weight

            if effective_alpha <= FIXED32_ZERO:
                result_transforms.append(transform_a.copy())
                continue

            if mode == DeterministicBlendMode.OVERRIDE:
                blended = self._blend_override(transform_a, transform_b, effective_alpha)
            elif mode == DeterministicBlendMode.ADDITIVE:
                blended = self._blend_additive(transform_a, transform_b, effective_alpha)
            else:  # MULTIPLY
                blended = self._blend_multiply(transform_a, transform_b, effective_alpha)

            result_transforms.append(blended)

        return Fixed32Pose(
            skeleton=self._skeleton,
            space=pose_a.space,
            _transforms=result_transforms,
        )

    def _blend_override(
        self,
        a: Fixed32BoneTransform,
        b: Fixed32BoneTransform,
        alpha: Fixed32,
    ) -> Fixed32BoneTransform:
        """Standard lerp/slerp blending."""
        return a.lerp(b, alpha)

    def _blend_additive(
        self,
        base: Fixed32BoneTransform,
        additive: Fixed32BoneTransform,
        weight: Fixed32,
    ) -> Fixed32BoneTransform:
        """Additive blending (base + weighted delta)."""
        # Translation: base + additive * weight
        new_translation = Fixed32Vec3(
            base.translation.x + additive.translation.x * weight,
            base.translation.y + additive.translation.y * weight,
            base.translation.z + additive.translation.z * weight,
        )

        # Rotation: base * slerp(identity, additive, weight)
        identity = Fixed32Quat.identity()
        weighted_rotation = identity.slerp(additive.rotation, weight)
        new_rotation = base.rotation * weighted_rotation

        # Scale: base * (1 + (additive - 1) * weight)
        new_scale = Fixed32Vec3(
            base.scale.x * (FIXED32_ONE + (additive.scale.x - FIXED32_ONE) * weight),
            base.scale.y * (FIXED32_ONE + (additive.scale.y - FIXED32_ONE) * weight),
            base.scale.z * (FIXED32_ONE + (additive.scale.z - FIXED32_ONE) * weight),
        )

        return Fixed32BoneTransform(
            translation=new_translation,
            rotation=new_rotation,
            scale=new_scale,
        )

    def _blend_multiply(
        self,
        base: Fixed32BoneTransform,
        factor: Fixed32BoneTransform,
        weight: Fixed32,
    ) -> Fixed32BoneTransform:
        """Multiplicative blending."""
        # Translation: lerp to factor position
        new_translation = base.translation.lerp(factor.translation, weight)

        # Rotation: lerp between base and base*factor
        combined = base.rotation * factor.rotation
        new_rotation = base.rotation.slerp(combined, weight)

        # Scale: multiply
        new_scale = Fixed32Vec3(
            base.scale.x * (FIXED32_ONE + (factor.scale.x - FIXED32_ONE) * weight),
            base.scale.y * (FIXED32_ONE + (factor.scale.y - FIXED32_ONE) * weight),
            base.scale.z * (FIXED32_ONE + (factor.scale.z - FIXED32_ONE) * weight),
        )

        return Fixed32BoneTransform(
            translation=new_translation,
            rotation=new_rotation,
            scale=new_scale,
        )

    def blend_multiple_poses(
        self,
        poses: List[Fixed32Pose],
        weights: List[Fixed32],
        normalize: bool = True,
    ) -> Fixed32Pose:
        """Blend multiple poses with Fixed32 weights.

        Args:
            poses: List of poses to blend.
            weights: List of blend weights.
            normalize: Whether to normalize weights to sum to 1.

        Returns:
            Blended pose.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not poses:
            raise ValueError("Cannot blend empty pose list")
        if len(poses) != len(weights):
            raise ValueError("Pose count must match weight count")

        # Verify all poses use same skeleton
        for pose in poses:
            if pose.skeleton is not self._skeleton:
                raise ValueError("All poses must use this blender's skeleton")

        # Normalize weights
        if normalize:
            total = FIXED32_ZERO
            for w in weights:
                if w > FIXED32_ZERO:
                    total = total + w

            if total <= FIXED32_EPSILON:
                return poses[0].copy()

            weights = [w / total if w > FIXED32_ZERO else FIXED32_ZERO for w in weights]

        # Blend each bone
        result_transforms = []

        for bone_idx in range(self._skeleton.bone_count):
            blended = self._blend_multiple_transforms(
                [p._transforms[bone_idx] for p in poses],
                weights,
            )
            result_transforms.append(blended)

        return Fixed32Pose(
            skeleton=self._skeleton,
            space=poses[0].space,
            _transforms=result_transforms,
        )

    def _blend_multiple_transforms(
        self,
        transforms: List[Fixed32BoneTransform],
        weights: List[Fixed32],
    ) -> Fixed32BoneTransform:
        """Blend multiple transforms with weights."""
        # Weighted average of translations and scales
        total_translation = Fixed32Vec3.zero()
        total_scale = Fixed32Vec3.zero()

        for t, w in zip(transforms, weights):
            if w > FIXED32_EPSILON:
                total_translation = Fixed32Vec3(
                    total_translation.x + t.translation.x * w,
                    total_translation.y + t.translation.y * w,
                    total_translation.z + t.translation.z * w,
                )
                total_scale = Fixed32Vec3(
                    total_scale.x + t.scale.x * w,
                    total_scale.y + t.scale.y * w,
                    total_scale.z + t.scale.z * w,
                )

        # Iterative slerp for rotations
        blended_rotation = transforms[0].rotation
        accumulated_weight = weights[0]

        for i in range(1, len(transforms)):
            if weights[i] > FIXED32_EPSILON:
                accumulated_weight = accumulated_weight + weights[i]
                if accumulated_weight > FIXED32_EPSILON:
                    slerp_factor = weights[i] / accumulated_weight
                    blended_rotation = blended_rotation.slerp(
                        transforms[i].rotation, slerp_factor
                    )

        return Fixed32BoneTransform(
            translation=total_translation,
            rotation=blended_rotation,
            scale=total_scale,
        )

    def compute_additive_pose(
        self,
        reference: Fixed32Pose,
        target: Fixed32Pose,
    ) -> Fixed32Pose:
        """Compute additive delta between poses.

        Args:
            reference: Base reference pose.
            target: Target pose.

        Returns:
            Additive pose (delta).
        """
        if reference.skeleton is not target.skeleton:
            raise ValueError("Poses must have same skeleton")

        additive_transforms = []

        for bone_idx in range(reference.bone_count):
            ref = reference._transforms[bone_idx]
            tgt = target._transforms[bone_idx]

            # Delta translation
            delta_translation = Fixed32Vec3(
                tgt.translation.x - ref.translation.x,
                tgt.translation.y - ref.translation.y,
                tgt.translation.z - ref.translation.z,
            )

            # Delta rotation: inverse(ref) * target
            delta_rotation = ref.rotation.inverse() * tgt.rotation

            # Delta scale as ratio
            delta_scale = Fixed32Vec3(
                tgt.scale.x / ref.scale.x if ref.scale.x > FIXED32_SCALE_EPSILON else FIXED32_ONE,
                tgt.scale.y / ref.scale.y if ref.scale.y > FIXED32_SCALE_EPSILON else FIXED32_ONE,
                tgt.scale.z / ref.scale.z if ref.scale.z > FIXED32_SCALE_EPSILON else FIXED32_ONE,
            )

            additive_transforms.append(Fixed32BoneTransform(
                translation=delta_translation,
                rotation=delta_rotation,
                scale=delta_scale,
            ))

        return Fixed32Pose(
            skeleton=reference.skeleton,
            space=reference.space,
            _transforms=additive_transforms,
        )

    def apply_additive_pose(
        self,
        base: Fixed32Pose,
        additive: Fixed32Pose,
        weight: Fixed32 = None,
        mask: Optional[Fixed32BoneMask] = None,
    ) -> Fixed32Pose:
        """Apply additive pose on top of base.

        Args:
            base: Base pose.
            additive: Additive delta pose.
            weight: How much additive to apply [0, 1].
            mask: Optional bone mask.

        Returns:
            Resulting pose.
        """
        if weight is None:
            weight = FIXED32_ONE

        return self.blend_poses(
            base, additive, weight,
            mode=DeterministicBlendMode.ADDITIVE,
            mask=mask,
        )

    def lerp_poses(
        self,
        pose_a: Fixed32Pose,
        pose_b: Fixed32Pose,
        alpha: Fixed32,
    ) -> Fixed32Pose:
        """Simple linear interpolation between poses.

        Convenience method for basic pose blending.

        Args:
            pose_a: First pose (alpha=0).
            pose_b: Second pose (alpha=1).
            alpha: Interpolation factor [0, 1].

        Returns:
            Interpolated pose.
        """
        return self.blend_poses(
            pose_a, pose_b, alpha,
            mode=DeterministicBlendMode.OVERRIDE,
        )


# =============================================================================
# Animation Time/Progress Using Fixed32
# =============================================================================

@dataclass(slots=True)
class Fixed32AnimationTime:
    """Animation time tracking using Fixed32 for determinism.

    Tracks current time, duration, playback speed, and loop state
    using fixed-point arithmetic for reproducible results.
    """

    current_time: Fixed32 = field(default_factory=lambda: FIXED32_ZERO)
    duration: Fixed32 = field(default_factory=lambda: FIXED32_ONE)
    speed: Fixed32 = field(default_factory=lambda: FIXED32_ONE)
    looping: bool = True
    _playing: bool = False

    @property
    def progress(self) -> Fixed32:
        """Get normalized progress [0, 1]."""
        if self.duration <= FIXED32_ZERO:
            return FIXED32_ZERO
        prog = self.current_time / self.duration
        if prog < FIXED32_ZERO:
            return FIXED32_ZERO
        if prog > FIXED32_ONE:
            return FIXED32_ONE
        return prog

    @property
    def is_playing(self) -> bool:
        """Check if animation is playing."""
        return self._playing

    @property
    def is_finished(self) -> bool:
        """Check if non-looping animation has finished."""
        if self.looping:
            return False
        return self.current_time >= self.duration

    def play(self) -> None:
        """Start playing."""
        self._playing = True

    def pause(self) -> None:
        """Pause playback."""
        self._playing = False

    def stop(self) -> None:
        """Stop and reset to start."""
        self._playing = False
        self.current_time = FIXED32_ZERO

    def seek(self, time: Fixed32) -> None:
        """Seek to a specific time."""
        if time < FIXED32_ZERO:
            time = FIXED32_ZERO
        if time > self.duration and not self.looping:
            time = self.duration
        self.current_time = time

    def seek_normalized(self, progress: Fixed32) -> None:
        """Seek to a normalized position [0, 1]."""
        self.current_time = self.duration * progress

    def advance(self, delta_time: Fixed32) -> None:
        """Advance time by delta.

        Args:
            delta_time: Time to advance (Fixed32).
        """
        if not self._playing:
            return

        self.current_time = self.current_time + delta_time * self.speed

        if self.looping:
            # Wrap around
            while self.current_time >= self.duration:
                self.current_time = self.current_time - self.duration
            while self.current_time < FIXED32_ZERO:
                self.current_time = self.current_time + self.duration
        else:
            # Clamp
            if self.current_time >= self.duration:
                self.current_time = self.duration
                self._playing = False
            elif self.current_time < FIXED32_ZERO:
                self.current_time = FIXED32_ZERO

    def copy(self) -> Fixed32AnimationTime:
        """Create a copy."""
        return Fixed32AnimationTime(
            current_time=Fixed32.from_raw(self.current_time.raw),
            duration=Fixed32.from_raw(self.duration.raw),
            speed=Fixed32.from_raw(self.speed.raw),
            looping=self.looping,
            _playing=self._playing,
        )


# =============================================================================
# Conversion Utilities
# =============================================================================

def convert_pose_to_fixed32(
    pose,  # engine.animation.skeletal.pose.Pose
    skeleton: Skeleton,
) -> Fixed32Pose:
    """Convert a floating-point Pose to Fixed32Pose.

    Args:
        pose: Source Pose with float transforms.
        skeleton: Skeleton for the pose.

    Returns:
        Equivalent Fixed32Pose.
    """
    from engine.animation.skeletal.pose import Pose, PoseSpace

    transforms = []
    for i in range(pose.bone_count):
        bt = pose._transforms[i]
        transforms.append(Fixed32BoneTransform(
            translation=Fixed32Vec3.from_floats(
                bt.translation.x, bt.translation.y, bt.translation.z
            ),
            rotation=Fixed32Quat.from_floats(
                bt.rotation.x, bt.rotation.y, bt.rotation.z, bt.rotation.w
            ),
            scale=Fixed32Vec3.from_floats(
                bt.scale.x, bt.scale.y, bt.scale.z
            ),
        ))

    space = (
        Fixed32PoseSpace.LOCAL
        if pose.space == PoseSpace.LOCAL
        else Fixed32PoseSpace.MODEL
    )

    return Fixed32Pose(
        skeleton=skeleton,
        space=space,
        _transforms=transforms,
    )


def convert_fixed32_to_pose(
    fixed_pose: Fixed32Pose,
):  # -> engine.animation.skeletal.pose.Pose
    """Convert a Fixed32Pose back to floating-point Pose.

    Args:
        fixed_pose: Source Fixed32Pose.

    Returns:
        Equivalent Pose with float transforms.
    """
    from engine.animation.skeletal.pose import Pose, BoneTransform, PoseSpace
    from engine.core.math import Vec3, Quat

    transforms = []
    for ft in fixed_pose._transforms:
        transforms.append(BoneTransform(
            translation=Vec3(
                ft.translation.x.as_float,
                ft.translation.y.as_float,
                ft.translation.z.as_float,
            ),
            rotation=Quat(
                ft.rotation.x.as_float,
                ft.rotation.y.as_float,
                ft.rotation.z.as_float,
                ft.rotation.w.as_float,
            ),
            scale=Vec3(
                ft.scale.x.as_float,
                ft.scale.y.as_float,
                ft.scale.z.as_float,
            ),
        ))

    space = (
        PoseSpace.LOCAL
        if fixed_pose.space == Fixed32PoseSpace.LOCAL
        else PoseSpace.MODEL
    )

    return Pose(
        skeleton=fixed_pose.skeleton,
        space=space,
        bone_transforms=transforms,
    )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Constants
    "FIXED32_ZERO",
    "FIXED32_ONE",
    "FIXED32_HALF",
    "FIXED32_EPSILON",
    # Math functions
    "fixed32_sin",
    "fixed32_cos",
    "fixed32_sqrt",
    "fixed32_acos",
    # Vector types
    "Fixed32Vec3",
    "Fixed32Quat",
    # Transform types
    "Fixed32BoneTransform",
    "Fixed32Pose",
    "Fixed32PoseSpace",
    # Blending
    "DeterministicBlendMode",
    "Fixed32BoneMask",
    "DeterministicBoneBlend",
    # Animation time
    "Fixed32AnimationTime",
    # Conversion
    "convert_pose_to_fixed32",
    "convert_fixed32_to_pose",
]
