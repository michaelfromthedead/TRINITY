"""Transform and Pose data structures for animation graph and IK systems.

This module provides dictionary-based (bone-name keyed) Transform and Pose
classes that integrate with the animation graph system. These complement the
index-based classes in animation_graph.py by providing name-based lookups
suitable for IK solvers and cross-skeleton operations.

Architecture
------------
Transform
    position: Vec3 (x, y, z)
    rotation: Quaternion (x, y, z, w)
    scale: Vec3 (x, y, z)
    blend(other, t)   -- SLERP for rotation, lerp for position/scale
    compose(other)    -- hierarchical transform composition

Pose
    bone_transforms: Dict[str, Transform]
    blend(other, t)   -- per-bone blending with missing bone handling
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

from .config import get_config


# =============================================================================
# NUMERICAL STABILITY CONSTANTS
# =============================================================================

# Epsilon for floating point comparisons
EPSILON: float = 1e-6

# Minimum quaternion length for normalization (below this = zero quaternion)
QUAT_NORMALIZE_EPSILON: float = 1e-6


# =============================================================================
# TYPE ALIASES
# =============================================================================

Vec3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]  # (x, y, z, w)


# =============================================================================
# TRANSFORM
# =============================================================================


@dataclass
class Transform:
    """A 3D transform with position, rotation (quaternion), and scale.

    This is the dictionary-based version of Transform designed for use with
    the bone-name-keyed Pose class. Provides SLERP interpolation for rotation
    and proper hierarchical composition.

    Attributes
    ----------
    position : Vec3
        Translation component (x, y, z). Defaults to origin.
    rotation : Quaternion
        Rotation as a quaternion (x, y, z, w). Defaults to identity.
    scale : Vec3
        Scale component (x, y, z). Defaults to uniform scale of 1.

    Examples
    --------
    >>> t1 = Transform(position=(0, 0, 0), rotation=(0, 0, 0, 1))
    >>> t2 = Transform(position=(1, 2, 3), rotation=(0, 0.707, 0, 0.707))
    >>> blended = t1.blend(t2, 0.5)
    """

    position: Vec3 = (0.0, 0.0, 0.0)
    rotation: Quaternion = (0.0, 0.0, 0.0, 1.0)  # (x, y, z, w) identity
    scale: Vec3 = (1.0, 1.0, 1.0)

    # -------------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------------

    @classmethod
    def identity(cls) -> Transform:
        """Create an identity transform (no translation, no rotation, unit scale)."""
        return cls()

    @classmethod
    def from_position(cls, x: float, y: float, z: float) -> Transform:
        """Create a transform with only a position component."""
        return cls(position=(x, y, z))

    @classmethod
    def from_rotation(cls, x: float, y: float, z: float, w: float) -> Transform:
        """Create a transform with only a rotation component."""
        return cls(rotation=(x, y, z, w))

    @classmethod
    def from_scale(cls, x: float, y: float, z: float) -> Transform:
        """Create a transform with only a scale component."""
        return cls(scale=(x, y, z))

    @classmethod
    def from_uniform_scale(cls, s: float) -> Transform:
        """Create a transform with uniform scale."""
        return cls(scale=(s, s, s))

    # -------------------------------------------------------------------------
    # Copy
    # -------------------------------------------------------------------------

    def copy(self) -> Transform:
        """Create a deep copy of this transform."""
        return Transform(
            position=self.position,
            rotation=self.rotation,
            scale=self.scale,
        )

    # -------------------------------------------------------------------------
    # Blending (SLERP for rotation)
    # -------------------------------------------------------------------------

    def blend(self, other: Transform, t: float) -> Transform:
        """Blend between two transforms using SLERP for rotation.

        Parameters
        ----------
        other : Transform
            The target transform to blend towards.
        t : float
            Blend factor. Clamped to [0, 1] for numerical stability.
            0 = self, 1 = other.

        Returns
        -------
        Transform
            The blended transform.

        Notes
        -----
        - Position and scale use linear interpolation
        - Rotation uses spherical linear interpolation (SLERP)
        - SLERP handles negative dot product (takes shorter path)
        - Falls back to linear interpolation when quaternions are nearly parallel
        """
        # Clamp t for numerical stability
        t = max(0.0, min(1.0, t))

        # Early exit for boundary cases
        if t <= EPSILON:
            return self.copy()
        if t >= 1.0 - EPSILON:
            return other.copy()

        # Linear interpolation for position
        pos = _lerp_vec3(self.position, other.position, t)

        # Linear interpolation for scale
        scl = _lerp_vec3(self.scale, other.scale, t)

        # SLERP for rotation
        rot = _slerp(self.rotation, other.rotation, t)

        return Transform(position=pos, rotation=rot, scale=scl)

    def lerp(self, other: Transform, t: float) -> Transform:
        """Alias for blend() for API compatibility."""
        return self.blend(other, t)

    # -------------------------------------------------------------------------
    # Hierarchical composition
    # -------------------------------------------------------------------------

    def compose(self, other: Transform) -> Transform:
        """Compose this transform with another (hierarchical transform).

        Treats *self* as the parent transform and *other* as the child.
        The result is equivalent to placing the child in the parent's
        local coordinate space.

        Parameters
        ----------
        other : Transform
            The child transform to compose.

        Returns
        -------
        Transform
            A single transform equivalent to the parent-child hierarchy.

        Notes
        -----
        Composition order:
            pos_out = parent.pos + rotate(parent.rot, parent.scale * child.pos)
            rot_out = parent.rot * child.rot
            scale_out = parent.scale * child.scale
        """
        # Scale child position by parent scale
        scaled_child_pos = (
            other.position[0] * self.scale[0],
            other.position[1] * self.scale[1],
            other.position[2] * self.scale[2],
        )

        # Rotate scaled child position by parent rotation
        rotated_child_pos = _rotate_vector(scaled_child_pos, self.rotation)

        # Final position
        pos_out = (
            self.position[0] + rotated_child_pos[0],
            self.position[1] + rotated_child_pos[1],
            self.position[2] + rotated_child_pos[2],
        )

        # Compose rotations (quaternion multiplication)
        rot_out = _multiply_quaternion(self.rotation, other.rotation)

        # Compose scales (component-wise multiplication)
        scale_out = (
            self.scale[0] * other.scale[0],
            self.scale[1] * other.scale[1],
            self.scale[2] * other.scale[2],
        )

        return Transform(position=pos_out, rotation=rot_out, scale=scale_out)

    # -------------------------------------------------------------------------
    # Additive operations
    # -------------------------------------------------------------------------

    def __add__(self, other: Transform) -> Transform:
        """Additive blend (for additive animations).

        Position: component-wise addition
        Rotation: quaternion multiplication
        Scale: component-wise multiplication
        """
        return Transform(
            position=(
                self.position[0] + other.position[0],
                self.position[1] + other.position[1],
                self.position[2] + other.position[2],
            ),
            rotation=_multiply_quaternion(self.rotation, other.rotation),
            scale=(
                self.scale[0] * other.scale[0],
                self.scale[1] * other.scale[1],
                self.scale[2] * other.scale[2],
            ),
        )

    # -------------------------------------------------------------------------
    # Validation / Normalization
    # -------------------------------------------------------------------------

    def normalized(self) -> Transform:
        """Return a copy with the rotation quaternion normalized."""
        return Transform(
            position=self.position,
            rotation=_normalize_quaternion(self.rotation),
            scale=self.scale,
        )

    def is_valid(self) -> bool:
        """Check if this transform has valid components (no NaN/Inf)."""
        for v in self.position + self.rotation + self.scale:
            if math.isnan(v) or math.isinf(v):
                return False
        return True

    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Transform(pos={self.position}, "
            f"rot={self.rotation}, "
            f"scale={self.scale})"
        )


# =============================================================================
# POSE
# =============================================================================


@dataclass
class Pose:
    """A collection of bone transforms representing an animation pose.

    This is a dictionary-based pose keyed by bone name, suitable for IK
    solvers and cross-skeleton operations where bone names are the primary
    identifiers.

    Attributes
    ----------
    bone_transforms : Dict[str, Transform]
        Mapping from bone name to its local-space transform.

    Examples
    --------
    >>> pose1 = Pose(bone_transforms={
    ...     "hip": Transform.identity(),
    ...     "spine": Transform.from_position(0, 1, 0),
    ... })
    >>> pose2 = Pose(bone_transforms={
    ...     "hip": Transform.from_position(0, 0.5, 0),
    ...     "spine": Transform.from_position(0, 1.5, 0),
    ... })
    >>> blended = pose1.blend(pose2, 0.5)
    """

    bone_transforms: Dict[str, Transform] = field(default_factory=dict)

    # -------------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------------

    @classmethod
    def empty(cls) -> Pose:
        """Create an empty pose with no bone transforms."""
        return cls()

    @classmethod
    def identity(cls, bone_names: Optional[list[str]] = None) -> Pose:
        """Create an identity pose for the given bones.

        Parameters
        ----------
        bone_names : list[str] or None
            Names of bones to include. If None, creates an empty pose.

        Returns
        -------
        Pose
            A pose with identity transforms for all specified bones.
        """
        if bone_names is None:
            return cls()
        return cls(
            bone_transforms={name: Transform.identity() for name in bone_names}
        )

    # -------------------------------------------------------------------------
    # Copy
    # -------------------------------------------------------------------------

    def copy(self) -> Pose:
        """Create a deep copy of this pose."""
        return Pose(
            bone_transforms={
                name: xform.copy() for name, xform in self.bone_transforms.items()
            }
        )

    # -------------------------------------------------------------------------
    # Bone accessors
    # -------------------------------------------------------------------------

    def get_transform(self, bone_name: str) -> Optional[Transform]:
        """Get the transform for a bone, or None if not present."""
        return self.bone_transforms.get(bone_name)

    def get_transform_or_identity(self, bone_name: str) -> Transform:
        """Get the transform for a bone, or identity if not present."""
        return self.bone_transforms.get(bone_name, Transform.identity())

    def set_transform(self, bone_name: str, transform: Transform) -> None:
        """Set the transform for a bone."""
        self.bone_transforms[bone_name] = transform

    def has_bone(self, bone_name: str) -> bool:
        """Check if this pose has a transform for the given bone."""
        return bone_name in self.bone_transforms

    def bone_count(self) -> int:
        """Return the number of bones in this pose."""
        return len(self.bone_transforms)

    def bone_names(self) -> list[str]:
        """Return the list of bone names in this pose."""
        return list(self.bone_transforms.keys())

    # -------------------------------------------------------------------------
    # Blending
    # -------------------------------------------------------------------------

    def blend(self, other: Pose, t: float) -> Pose:
        """Blend between two poses, handling missing bones gracefully.

        Parameters
        ----------
        other : Pose
            The target pose to blend towards.
        t : float
            Blend factor. Clamped to [0, 1] for numerical stability.
            0 = self, 1 = other.

        Returns
        -------
        Pose
            The blended pose.

        Notes
        -----
        Missing bone handling:
        - If a bone exists in only one pose, that transform is used directly
        - If a bone exists in both poses, transforms are blended
        - This allows partial poses to be blended without data loss
        """
        # Clamp t for numerical stability
        t = max(0.0, min(1.0, t))

        # Early exit for boundary cases
        if t <= EPSILON:
            return self.copy()
        if t >= 1.0 - EPSILON:
            return other.copy()

        # Gather all bone names from both poses
        all_bones = set(self.bone_transforms.keys()) | set(other.bone_transforms.keys())

        result_transforms: Dict[str, Transform] = {}

        for bone_name in all_bones:
            self_xform = self.bone_transforms.get(bone_name)
            other_xform = other.bone_transforms.get(bone_name)

            if self_xform is not None and other_xform is not None:
                # Both poses have this bone - blend them
                result_transforms[bone_name] = self_xform.blend(other_xform, t)
            elif self_xform is not None:
                # Only self has this bone - use it directly
                result_transforms[bone_name] = self_xform.copy()
            else:
                # Only other has this bone - use it directly
                assert other_xform is not None
                result_transforms[bone_name] = other_xform.copy()

        return Pose(bone_transforms=result_transforms)

    def lerp(self, other: Pose, t: float) -> Pose:
        """Alias for blend() for API compatibility."""
        return self.blend(other, t)

    # -------------------------------------------------------------------------
    # Additive blending
    # -------------------------------------------------------------------------

    def additive_blend(self, additive: Pose, weight: float = 1.0) -> Pose:
        """Apply an additive pose on top of this base pose.

        Parameters
        ----------
        additive : Pose
            The additive pose to apply.
        weight : float
            Weight of the additive contribution. Clamped to [0, 1].

        Returns
        -------
        Pose
            The result of applying the additive pose.
        """
        weight = max(0.0, min(1.0, weight))

        if weight <= EPSILON:
            return self.copy()

        result_transforms: Dict[str, Transform] = {}

        # Copy all bones from self
        for name, xform in self.bone_transforms.items():
            result_transforms[name] = xform.copy()

        # Apply additive contributions
        for name, add_xform in additive.bone_transforms.items():
            if name in result_transforms:
                base = result_transforms[name]

                # Scale additive contribution by weight
                if weight < 1.0 - EPSILON:
                    add_xform = Transform.identity().blend(add_xform, weight)

                result_transforms[name] = base + add_xform
            else:
                # Bone only in additive - apply to identity
                if weight < 1.0 - EPSILON:
                    add_xform = Transform.identity().blend(add_xform, weight)
                result_transforms[name] = Transform.identity() + add_xform

        return Pose(bone_transforms=result_transforms)

    # -------------------------------------------------------------------------
    # Filtering / Masking
    # -------------------------------------------------------------------------

    def filter_bones(self, bone_names: list[str]) -> Pose:
        """Create a new pose containing only the specified bones.

        Parameters
        ----------
        bone_names : list[str]
            Names of bones to include.

        Returns
        -------
        Pose
            A pose with only the specified bones (those that exist).
        """
        return Pose(
            bone_transforms={
                name: self.bone_transforms[name].copy()
                for name in bone_names
                if name in self.bone_transforms
            }
        )

    def exclude_bones(self, bone_names: list[str]) -> Pose:
        """Create a new pose excluding the specified bones.

        Parameters
        ----------
        bone_names : list[str]
            Names of bones to exclude.

        Returns
        -------
        Pose
            A pose without the specified bones.
        """
        exclude_set = set(bone_names)
        return Pose(
            bone_transforms={
                name: xform.copy()
                for name, xform in self.bone_transforms.items()
                if name not in exclude_set
            }
        )

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def is_valid(self) -> bool:
        """Check if all transforms in this pose are valid (no NaN/Inf)."""
        return all(xform.is_valid() for xform in self.bone_transforms.values())

    def normalized(self) -> Pose:
        """Return a copy with all rotation quaternions normalized."""
        return Pose(
            bone_transforms={
                name: xform.normalized()
                for name, xform in self.bone_transforms.items()
            }
        )

    # -------------------------------------------------------------------------
    # Merge / Combine
    # -------------------------------------------------------------------------

    def merge(self, other: Pose, overwrite: bool = True) -> Pose:
        """Merge another pose into this one.

        Parameters
        ----------
        other : Pose
            The pose to merge.
        overwrite : bool
            If True, other's bones overwrite self's. If False, self's
            bones take precedence.

        Returns
        -------
        Pose
            A new pose containing bones from both poses.
        """
        result_transforms: Dict[str, Transform] = {}

        if overwrite:
            # Self first, then other overwrites
            for name, xform in self.bone_transforms.items():
                result_transforms[name] = xform.copy()
            for name, xform in other.bone_transforms.items():
                result_transforms[name] = xform.copy()
        else:
            # Other first, then self overwrites
            for name, xform in other.bone_transforms.items():
                result_transforms[name] = xform.copy()
            for name, xform in self.bone_transforms.items():
                result_transforms[name] = xform.copy()

        return Pose(bone_transforms=result_transforms)

    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Pose(bones={list(self.bone_transforms.keys())})"

    def __len__(self) -> int:
        return len(self.bone_transforms)

    def __contains__(self, bone_name: str) -> bool:
        return bone_name in self.bone_transforms

    def __iter__(self):
        return iter(self.bone_transforms.items())


# =============================================================================
# QUATERNION UTILITIES
# =============================================================================


def _slerp(q1: Quaternion, q2: Quaternion, t: float) -> Quaternion:
    """Spherical linear interpolation for quaternions.

    Parameters
    ----------
    q1 : Quaternion
        Start quaternion (x, y, z, w).
    q2 : Quaternion
        End quaternion (x, y, z, w).
    t : float
        Interpolation factor [0, 1].

    Returns
    -------
    Quaternion
        The interpolated quaternion.

    Notes
    -----
    - Handles negative dot product (flips q2 to take shorter path)
    - Falls back to linear interpolation when quaternions are nearly parallel
      (dot > SLERP_DOT_THRESHOLD) to avoid numerical instability
    """
    config = get_config()

    # Compute dot product
    dot = q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]

    # If dot is negative, negate q2 to take the shorter path
    if dot < 0.0:
        q2 = (-q2[0], -q2[1], -q2[2], -q2[3])
        dot = -dot

    # If quaternions are very close, use linear interpolation
    # This avoids numerical issues when sin(theta) approaches 0
    if dot > config.quaternion.SLERP_DOT_THRESHOLD:
        result = (
            q1[0] + (q2[0] - q1[0]) * t,
            q1[1] + (q2[1] - q1[1]) * t,
            q1[2] + (q2[2] - q1[2]) * t,
            q1[3] + (q2[3] - q1[3]) * t,
        )
        return _normalize_quaternion(result)

    # Standard SLERP
    # Clamp dot to valid acos range [-1, 1]
    dot = max(-1.0, min(1.0, dot))
    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    # Check for near-zero sin_theta_0 (shouldn't happen due to threshold above)
    if sin_theta_0 < config.quaternion.SLERP_MIN_SIN_THETA:
        return q1

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return (
        s0 * q1[0] + s1 * q2[0],
        s0 * q1[1] + s1 * q2[1],
        s0 * q1[2] + s1 * q2[2],
        s0 * q1[3] + s1 * q2[3],
    )


def _multiply_quaternion(q1: Quaternion, q2: Quaternion) -> Quaternion:
    """Multiply two quaternions (Hamilton product).

    Parameters
    ----------
    q1 : Quaternion
        First quaternion (x, y, z, w).
    q2 : Quaternion
        Second quaternion (x, y, z, w).

    Returns
    -------
    Quaternion
        The product q1 * q2.
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _normalize_quaternion(q: Quaternion) -> Quaternion:
    """Normalize a quaternion to unit length.

    Parameters
    ----------
    q : Quaternion
        The quaternion to normalize (x, y, z, w).

    Returns
    -------
    Quaternion
        The normalized quaternion. Returns identity (0, 0, 0, 1) if
        the input quaternion has near-zero length.
    """
    length_sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]

    if length_sq < QUAT_NORMALIZE_EPSILON:
        # Zero-length quaternion - return identity
        return (0.0, 0.0, 0.0, 1.0)

    length = math.sqrt(length_sq)
    return (q[0] / length, q[1] / length, q[2] / length, q[3] / length)


def _rotate_vector(v: Vec3, q: Quaternion) -> Vec3:
    """Rotate a 3D vector by a quaternion.

    Uses the formula: v' = q * v * q^(-1)
    Optimized to avoid explicit quaternion inverse.

    Parameters
    ----------
    v : Vec3
        The vector to rotate (x, y, z).
    q : Quaternion
        The rotation quaternion (x, y, z, w).

    Returns
    -------
    Vec3
        The rotated vector.
    """
    qx, qy, qz, qw = q
    vx, vy, vz = v

    # Compute cross products
    # uv = q_vec x v
    uv_x = qy * vz - qz * vy
    uv_y = qz * vx - qx * vz
    uv_z = qx * vy - qy * vx

    # uuv = q_vec x uv
    uuv_x = qy * uv_z - qz * uv_y
    uuv_y = qz * uv_x - qx * uv_z
    uuv_z = qx * uv_y - qy * uv_x

    # Result: v + 2 * (q.w * uv + uuv)
    return (
        vx + 2.0 * (qw * uv_x + uuv_x),
        vy + 2.0 * (qw * uv_y + uuv_y),
        vz + 2.0 * (qw * uv_z + uuv_z),
    )


def _lerp_vec3(v1: Vec3, v2: Vec3, t: float) -> Vec3:
    """Linear interpolation between two 3D vectors.

    Parameters
    ----------
    v1 : Vec3
        Start vector.
    v2 : Vec3
        End vector.
    t : float
        Interpolation factor [0, 1].

    Returns
    -------
    Vec3
        The interpolated vector.
    """
    return (
        v1[0] + (v2[0] - v1[0]) * t,
        v1[1] + (v2[1] - v1[1]) * t,
        v1[2] + (v2[2] - v1[2]) * t,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "EPSILON",
    "QUAT_NORMALIZE_EPSILON",
    # Type aliases
    "Vec3",
    "Quaternion",
    # Classes
    "Transform",
    "Pose",
    # Utilities (for advanced use)
    "_slerp",
    "_multiply_quaternion",
    "_normalize_quaternion",
    "_rotate_vector",
    "_lerp_vec3",
]
