"""Tests for deterministic skeleton blending using Fixed32 arithmetic.

Comprehensive test suite verifying:
- Fixed32Vec3 operations
- Fixed32Quat operations (including slerp/nlerp)
- Fixed32BoneTransform blending
- Fixed32Pose construction and manipulation
- DeterministicBoneBlend operations
- Fixed32AnimationTime tracking
- Determinism (bit-identical results across runs)
- Conversion between float and Fixed32 poses

Test count: 50+ tests covering all deterministic blending functionality.
"""

import pytest
import math
from typing import List

from trinity.types import Fixed32

from engine.animation.deterministic_blend import (
    # Constants
    FIXED32_ZERO,
    FIXED32_ONE,
    FIXED32_HALF,
    FIXED32_EPSILON,
    FIXED32_TWO,
    FIXED32_NEG_ONE,
    # Math functions
    fixed32_sin,
    fixed32_cos,
    fixed32_sqrt,
    fixed32_acos,
    # Vector types
    Fixed32Vec3,
    Fixed32Quat,
    # Transform types
    Fixed32BoneTransform,
    Fixed32Pose,
    Fixed32PoseSpace,
    # Blending
    DeterministicBlendMode,
    Fixed32BoneMask,
    DeterministicBoneBlend,
    # Animation time
    Fixed32AnimationTime,
    # Conversion
    convert_pose_to_fixed32,
    convert_fixed32_to_pose,
)

from engine.animation.skeletal.skeleton import Skeleton, Bone, create_humanoid_skeleton
from engine.animation.skeletal.pose import Pose, BoneTransform, PoseSpace
from engine.core.math import Vec3, Quat, Transform


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_skeleton() -> Skeleton:
    """Create a simple 3-bone skeleton for testing."""
    skeleton = Skeleton(name="test_skeleton")
    skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))
    skeleton.add_bone(Bone(index=1, name="spine", parent_index=0))
    skeleton.add_bone(Bone(index=2, name="head", parent_index=1))
    skeleton._rebuild_caches()
    return skeleton


@pytest.fixture
def humanoid_skeleton() -> Skeleton:
    """Create a standard humanoid skeleton."""
    return create_humanoid_skeleton()


@pytest.fixture
def identity_pose(simple_skeleton: Skeleton) -> Fixed32Pose:
    """Create an identity pose for the simple skeleton."""
    return Fixed32Pose(skeleton=simple_skeleton)


@pytest.fixture
def translated_pose(simple_skeleton: Skeleton) -> Fixed32Pose:
    """Create a pose with translated bones."""
    pose = Fixed32Pose(skeleton=simple_skeleton)
    pose.set_bone_transform(0, Fixed32BoneTransform(
        translation=Fixed32Vec3.from_floats(1.0, 0.0, 0.0),
    ))
    pose.set_bone_transform(1, Fixed32BoneTransform(
        translation=Fixed32Vec3.from_floats(0.0, 2.0, 0.0),
    ))
    pose.set_bone_transform(2, Fixed32BoneTransform(
        translation=Fixed32Vec3.from_floats(0.0, 0.0, 3.0),
    ))
    return pose


@pytest.fixture
def blender(simple_skeleton: Skeleton) -> DeterministicBoneBlend:
    """Create a blender for the simple skeleton."""
    return DeterministicBoneBlend(simple_skeleton)


# =============================================================================
# Fixed32Vec3 Tests
# =============================================================================

class TestFixed32Vec3:
    """Tests for Fixed32Vec3 operations."""

    def test_zero_vector(self):
        """Test zero vector creation."""
        v = Fixed32Vec3.zero()
        assert v.x == FIXED32_ZERO
        assert v.y == FIXED32_ZERO
        assert v.z == FIXED32_ZERO

    def test_one_vector(self):
        """Test unit vector creation."""
        v = Fixed32Vec3.one()
        assert v.x == FIXED32_ONE
        assert v.y == FIXED32_ONE
        assert v.z == FIXED32_ONE

    def test_from_floats(self):
        """Test creation from float values."""
        v = Fixed32Vec3.from_floats(1.5, 2.5, 3.5)
        assert abs(v.x.as_float - 1.5) < 0.001
        assert abs(v.y.as_float - 2.5) < 0.001
        assert abs(v.z.as_float - 3.5) < 0.001

    def test_addition(self):
        """Test vector addition."""
        a = Fixed32Vec3.from_floats(1.0, 2.0, 3.0)
        b = Fixed32Vec3.from_floats(0.5, 0.5, 0.5)
        c = a + b
        assert abs(c.x.as_float - 1.5) < 0.001
        assert abs(c.y.as_float - 2.5) < 0.001
        assert abs(c.z.as_float - 3.5) < 0.001

    def test_subtraction(self):
        """Test vector subtraction."""
        a = Fixed32Vec3.from_floats(2.0, 3.0, 4.0)
        b = Fixed32Vec3.from_floats(1.0, 1.0, 1.0)
        c = a - b
        assert abs(c.x.as_float - 1.0) < 0.001
        assert abs(c.y.as_float - 2.0) < 0.001
        assert abs(c.z.as_float - 3.0) < 0.001

    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        v = Fixed32Vec3.from_floats(1.0, 2.0, 3.0)
        scaled = v * Fixed32(2.0)
        assert abs(scaled.x.as_float - 2.0) < 0.001
        assert abs(scaled.y.as_float - 4.0) < 0.001
        assert abs(scaled.z.as_float - 6.0) < 0.001

    def test_negation(self):
        """Test vector negation."""
        v = Fixed32Vec3.from_floats(1.0, -2.0, 3.0)
        neg = -v
        assert abs(neg.x.as_float - (-1.0)) < 0.001
        assert abs(neg.y.as_float - 2.0) < 0.001
        assert abs(neg.z.as_float - (-3.0)) < 0.001

    def test_dot_product(self):
        """Test dot product."""
        a = Fixed32Vec3.from_floats(1.0, 0.0, 0.0)
        b = Fixed32Vec3.from_floats(0.0, 1.0, 0.0)
        assert a.dot(b) == FIXED32_ZERO

        c = Fixed32Vec3.from_floats(1.0, 2.0, 3.0)
        d = Fixed32Vec3.from_floats(1.0, 2.0, 3.0)
        assert abs(c.dot(d).as_float - 14.0) < 0.01

    def test_cross_product(self):
        """Test cross product."""
        x = Fixed32Vec3.from_floats(1.0, 0.0, 0.0)
        y = Fixed32Vec3.from_floats(0.0, 1.0, 0.0)
        z = x.cross(y)
        assert abs(z.x.as_float - 0.0) < 0.001
        assert abs(z.y.as_float - 0.0) < 0.001
        assert abs(z.z.as_float - 1.0) < 0.001

    def test_length(self):
        """Test vector length."""
        v = Fixed32Vec3.from_floats(3.0, 4.0, 0.0)
        assert abs(v.length().as_float - 5.0) < 0.01

    def test_normalized(self):
        """Test vector normalization."""
        v = Fixed32Vec3.from_floats(3.0, 4.0, 0.0)
        n = v.normalized()
        assert abs(n.length().as_float - 1.0) < 0.01
        assert abs(n.x.as_float - 0.6) < 0.01
        assert abs(n.y.as_float - 0.8) < 0.01

    def test_lerp(self):
        """Test linear interpolation."""
        a = Fixed32Vec3.from_floats(0.0, 0.0, 0.0)
        b = Fixed32Vec3.from_floats(10.0, 10.0, 10.0)

        mid = a.lerp(b, FIXED32_HALF)
        assert abs(mid.x.as_float - 5.0) < 0.01
        assert abs(mid.y.as_float - 5.0) < 0.01
        assert abs(mid.z.as_float - 5.0) < 0.01

    def test_equality(self):
        """Test vector equality (bit-exact)."""
        a = Fixed32Vec3.from_floats(1.0, 2.0, 3.0)
        b = Fixed32Vec3.from_floats(1.0, 2.0, 3.0)
        c = Fixed32Vec3.from_floats(1.0, 2.0, 3.1)

        assert a == b
        assert a != c

    def test_copy(self):
        """Test vector copying."""
        a = Fixed32Vec3.from_floats(1.0, 2.0, 3.0)
        b = a.copy()
        assert a == b
        assert a is not b

    def test_raw_tuple(self):
        """Test raw integer representation."""
        a = Fixed32Vec3.from_floats(1.0, 2.0, 3.0)
        b = Fixed32Vec3.from_floats(1.0, 2.0, 3.0)
        assert a.raw_tuple() == b.raw_tuple()


# =============================================================================
# Fixed32Quat Tests
# =============================================================================

class TestFixed32Quat:
    """Tests for Fixed32Quat operations."""

    def test_identity_quaternion(self):
        """Test identity quaternion."""
        q = Fixed32Quat.identity()
        assert q.x == FIXED32_ZERO
        assert q.y == FIXED32_ZERO
        assert q.z == FIXED32_ZERO
        assert q.w == FIXED32_ONE

    def test_from_floats(self):
        """Test creation from float values."""
        q = Fixed32Quat.from_floats(0.0, 0.0, 0.0, 1.0)
        assert q == Fixed32Quat.identity()

    def test_multiplication(self):
        """Test quaternion multiplication."""
        identity = Fixed32Quat.identity()
        q = Fixed32Quat.from_floats(0.0, 0.707, 0.0, 0.707)

        # Identity * q = q
        result = identity * q
        assert abs(result.y.as_float - 0.707) < 0.01
        assert abs(result.w.as_float - 0.707) < 0.01

    def test_conjugate(self):
        """Test quaternion conjugate."""
        q = Fixed32Quat.from_floats(0.1, 0.2, 0.3, 0.9)
        conj = q.conjugate()
        assert conj.x == -q.x
        assert conj.y == -q.y
        assert conj.z == -q.z
        assert conj.w == q.w

    def test_dot_product(self):
        """Test quaternion dot product."""
        a = Fixed32Quat.identity()
        b = Fixed32Quat.identity()
        assert abs(a.dot(b).as_float - 1.0) < 0.01

    def test_length(self):
        """Test quaternion length."""
        q = Fixed32Quat.identity()
        assert abs(q.length().as_float - 1.0) < 0.01

    def test_normalized(self):
        """Test quaternion normalization."""
        q = Fixed32Quat.from_floats(0.0, 0.0, 0.0, 2.0)
        n = q.normalized()
        assert abs(n.length().as_float - 1.0) < 0.01

    def test_inverse(self):
        """Test quaternion inverse."""
        q = Fixed32Quat.identity()
        inv = q.inverse()
        product = q * inv
        assert abs(product.w.as_float - 1.0) < 0.01
        assert abs(product.x.as_float) < 0.01
        assert abs(product.y.as_float) < 0.01
        assert abs(product.z.as_float) < 0.01

    def test_nlerp(self):
        """Test normalized linear interpolation."""
        a = Fixed32Quat.identity()
        b = Fixed32Quat.from_floats(0.0, 0.707, 0.0, 0.707)

        mid = a.nlerp(b, FIXED32_HALF)
        assert abs(mid.length().as_float - 1.0) < 0.01

    def test_slerp(self):
        """Test spherical linear interpolation."""
        a = Fixed32Quat.identity()
        b = Fixed32Quat.from_floats(0.0, 0.707, 0.0, 0.707)

        # At t=0, should be a
        result_0 = a.slerp(b, FIXED32_ZERO)
        assert result_0 == a

        # At t=1, should be b
        result_1 = a.slerp(b, FIXED32_ONE)
        assert abs(result_1.y.as_float - b.y.as_float) < 0.01
        assert abs(result_1.w.as_float - b.w.as_float) < 0.01

    def test_slerp_antipodal(self):
        """Test slerp with antipodal quaternions."""
        a = Fixed32Quat.from_floats(0.0, 0.0, 0.0, 1.0)
        b = Fixed32Quat.from_floats(0.0, 0.0, 0.0, -1.0)  # Antipodal

        # Should handle shortest path
        mid = a.slerp(b, FIXED32_HALF)
        assert mid.length() > FIXED32_ZERO

    def test_rotate_vector(self):
        """Test vector rotation by quaternion."""
        # 90 degree rotation around Y axis
        angle = Fixed32(math.pi / 2)
        axis = Fixed32Vec3.from_floats(0.0, 1.0, 0.0)
        q = Fixed32Quat.from_axis_angle(axis, angle)

        v = Fixed32Vec3.from_floats(1.0, 0.0, 0.0)
        rotated = q.rotate_vector(v)

        # X should become approximately Z
        assert abs(rotated.x.as_float) < 0.1
        assert abs(rotated.z.as_float - (-1.0)) < 0.1

    def test_equality(self):
        """Test quaternion equality."""
        a = Fixed32Quat.identity()
        b = Fixed32Quat.identity()
        assert a == b

    def test_copy(self):
        """Test quaternion copying."""
        a = Fixed32Quat.from_floats(0.1, 0.2, 0.3, 0.9)
        b = a.copy()
        assert a == b
        assert a is not b


# =============================================================================
# Fixed32BoneTransform Tests
# =============================================================================

class TestFixed32BoneTransform:
    """Tests for Fixed32BoneTransform operations."""

    def test_identity_transform(self):
        """Test identity transform creation."""
        t = Fixed32BoneTransform.identity()
        assert t.translation == Fixed32Vec3.zero()
        assert t.rotation == Fixed32Quat.identity()
        assert t.scale == Fixed32Vec3.one()

    def test_from_floats(self):
        """Test creation from float values."""
        t = Fixed32BoneTransform.from_floats(
            1.0, 2.0, 3.0,  # translation
            0.0, 0.0, 0.0, 1.0,  # rotation
            1.0, 1.0, 1.0,  # scale
        )
        assert abs(t.translation.x.as_float - 1.0) < 0.001
        assert abs(t.translation.y.as_float - 2.0) < 0.001
        assert abs(t.translation.z.as_float - 3.0) < 0.001

    def test_lerp(self):
        """Test transform interpolation."""
        a = Fixed32BoneTransform.identity()
        b = Fixed32BoneTransform.from_floats(
            10.0, 10.0, 10.0,
            0.0, 0.0, 0.0, 1.0,
            2.0, 2.0, 2.0,
        )

        mid = a.lerp(b, FIXED32_HALF)
        assert abs(mid.translation.x.as_float - 5.0) < 0.01
        assert abs(mid.scale.x.as_float - 1.5) < 0.01

    def test_copy(self):
        """Test transform copying."""
        a = Fixed32BoneTransform.from_floats(
            1.0, 2.0, 3.0,
            0.0, 0.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
        )
        b = a.copy()
        assert a == b
        assert a is not b
        assert a.translation is not b.translation

    def test_equality(self):
        """Test transform equality."""
        a = Fixed32BoneTransform.identity()
        b = Fixed32BoneTransform.identity()
        c = Fixed32BoneTransform.from_floats(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0)
        assert a == b
        assert a != c


# =============================================================================
# Fixed32Pose Tests
# =============================================================================

class TestFixed32Pose:
    """Tests for Fixed32Pose operations."""

    def test_pose_creation(self, simple_skeleton: Skeleton):
        """Test pose creation with skeleton."""
        pose = Fixed32Pose(skeleton=simple_skeleton)
        assert pose.bone_count == 3
        assert pose.space == Fixed32PoseSpace.LOCAL

    def test_get_bone_transform(self, identity_pose: Fixed32Pose):
        """Test getting bone transform by index."""
        t = identity_pose.get_bone_transform(0)
        assert t == Fixed32BoneTransform.identity()

    def test_set_bone_transform(self, identity_pose: Fixed32Pose):
        """Test setting bone transform."""
        new_transform = Fixed32BoneTransform.from_floats(
            1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0
        )
        identity_pose.set_bone_transform(1, new_transform)
        t = identity_pose.get_bone_transform(1)
        assert t == new_transform

    def test_get_bone_transform_by_name(self, simple_skeleton: Skeleton):
        """Test getting transform by bone name."""
        pose = Fixed32Pose(skeleton=simple_skeleton)
        pose.set_bone_transform(1, Fixed32BoneTransform.from_floats(
            5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0
        ))

        t = pose.get_bone_transform_by_name("spine")
        assert t is not None
        assert abs(t.translation.x.as_float - 5.0) < 0.01

        none_t = pose.get_bone_transform_by_name("nonexistent")
        assert none_t is None

    def test_pose_copy(self, translated_pose: Fixed32Pose):
        """Test pose copying."""
        copy = translated_pose.copy()
        assert translated_pose == copy
        assert translated_pose is not copy
        assert translated_pose._transforms[0] is not copy._transforms[0]

    def test_pose_equality(self, simple_skeleton: Skeleton):
        """Test pose equality."""
        a = Fixed32Pose(skeleton=simple_skeleton)
        b = Fixed32Pose(skeleton=simple_skeleton)
        assert a == b

        a.set_bone_transform(0, Fixed32BoneTransform.from_floats(
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0
        ))
        assert a != b

    def test_invalid_transform_count(self, simple_skeleton: Skeleton):
        """Test that wrong transform count raises error."""
        with pytest.raises(ValueError):
            Fixed32Pose(
                skeleton=simple_skeleton,
                _transforms=[Fixed32BoneTransform.identity()],  # Wrong count
            )

    def test_invalid_bone_index(self, identity_pose: Fixed32Pose):
        """Test that invalid index raises error."""
        with pytest.raises(IndexError):
            identity_pose.get_bone_transform(100)

        with pytest.raises(IndexError):
            identity_pose.set_bone_transform(-1, Fixed32BoneTransform.identity())


# =============================================================================
# DeterministicBoneBlend Tests
# =============================================================================

class TestDeterministicBoneBlend:
    """Tests for DeterministicBoneBlend operations."""

    def test_blender_creation(self, simple_skeleton: Skeleton):
        """Test blender creation."""
        blender = DeterministicBoneBlend(simple_skeleton)
        assert blender.skeleton is simple_skeleton

    def test_blend_poses_alpha_zero(
        self,
        blender: DeterministicBoneBlend,
        identity_pose: Fixed32Pose,
        translated_pose: Fixed32Pose,
    ):
        """Test that alpha=0 returns first pose."""
        result = blender.blend_poses(
            identity_pose, translated_pose, FIXED32_ZERO
        )
        assert result == identity_pose

    def test_blend_poses_alpha_one(
        self,
        blender: DeterministicBoneBlend,
        identity_pose: Fixed32Pose,
        translated_pose: Fixed32Pose,
    ):
        """Test that alpha=1 returns second pose."""
        result = blender.blend_poses(
            identity_pose, translated_pose, FIXED32_ONE
        )
        assert result == translated_pose

    def test_blend_poses_half(
        self,
        blender: DeterministicBoneBlend,
        identity_pose: Fixed32Pose,
        translated_pose: Fixed32Pose,
    ):
        """Test blending at 50%."""
        result = blender.blend_poses(
            identity_pose, translated_pose, FIXED32_HALF
        )

        # Root bone: (0,0,0) -> (1,0,0), halfway = (0.5, 0, 0)
        root = result.get_bone_transform(0)
        assert abs(root.translation.x.as_float - 0.5) < 0.01

    def test_blend_with_mask(
        self,
        blender: DeterministicBoneBlend,
        identity_pose: Fixed32Pose,
        translated_pose: Fixed32Pose,
    ):
        """Test blending with bone mask."""
        mask = Fixed32BoneMask()
        mask.include_bone(0)  # Only blend root
        mask.exclude_bone(1)
        mask.exclude_bone(2)

        result = blender.blend_poses(
            identity_pose, translated_pose, FIXED32_ONE, mask=mask
        )

        # Root should be blended
        root = result.get_bone_transform(0)
        assert abs(root.translation.x.as_float - 1.0) < 0.01

        # Spine should be unchanged (masked out)
        spine = result.get_bone_transform(1)
        assert spine.translation.y == FIXED32_ZERO

    def test_blend_additive_mode(
        self,
        blender: DeterministicBoneBlend,
        identity_pose: Fixed32Pose,
        translated_pose: Fixed32Pose,
    ):
        """Test additive blending mode."""
        result = blender.blend_poses(
            identity_pose, translated_pose, FIXED32_ONE,
            mode=DeterministicBlendMode.ADDITIVE,
        )

        # Additive: base + additive * weight
        root = result.get_bone_transform(0)
        # identity (0,0,0) + translated (1,0,0) * 1 = (1,0,0)
        assert abs(root.translation.x.as_float - 1.0) < 0.01

    def test_blend_multiply_mode(
        self,
        blender: DeterministicBoneBlend,
        simple_skeleton: Skeleton,
    ):
        """Test multiplicative blending mode."""
        pose_a = Fixed32Pose(skeleton=simple_skeleton)
        pose_a.set_bone_transform(0, Fixed32BoneTransform.from_floats(
            2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0
        ))

        pose_b = Fixed32Pose(skeleton=simple_skeleton)
        pose_b.set_bone_transform(0, Fixed32BoneTransform.from_floats(
            4.0, 4.0, 4.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0
        ))

        result = blender.blend_poses(
            pose_a, pose_b, FIXED32_ONE,
            mode=DeterministicBlendMode.MULTIPLY,
        )

        root = result.get_bone_transform(0)
        # Translation lerps to factor
        assert abs(root.translation.x.as_float - 4.0) < 0.01

    def test_blend_multiple_poses(
        self,
        blender: DeterministicBoneBlend,
        simple_skeleton: Skeleton,
    ):
        """Test blending multiple poses."""
        poses = []
        for i in range(3):
            pose = Fixed32Pose(skeleton=simple_skeleton)
            pose.set_bone_transform(0, Fixed32BoneTransform.from_floats(
                float(i), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0
            ))
            poses.append(pose)

        # Equal weights
        weights = [Fixed32(1.0), Fixed32(1.0), Fixed32(1.0)]
        result = blender.blend_multiple_poses(poses, weights)

        # Average: (0 + 1 + 2) / 3 = 1.0
        root = result.get_bone_transform(0)
        assert abs(root.translation.x.as_float - 1.0) < 0.01

    def test_compute_additive_pose(
        self,
        blender: DeterministicBoneBlend,
        identity_pose: Fixed32Pose,
        translated_pose: Fixed32Pose,
    ):
        """Test computing additive delta."""
        additive = blender.compute_additive_pose(identity_pose, translated_pose)

        # Delta from identity to translated
        root_delta = additive.get_bone_transform(0)
        assert abs(root_delta.translation.x.as_float - 1.0) < 0.01

    def test_apply_additive_pose(
        self,
        blender: DeterministicBoneBlend,
        identity_pose: Fixed32Pose,
        translated_pose: Fixed32Pose,
    ):
        """Test applying additive pose."""
        additive = blender.compute_additive_pose(identity_pose, translated_pose)

        # Apply to another identity pose
        base = identity_pose.copy()
        result = blender.apply_additive_pose(base, additive)

        # Should equal translated
        assert result.get_bone_transform(0).translation.x.as_float > 0.9

    def test_lerp_poses(
        self,
        blender: DeterministicBoneBlend,
        identity_pose: Fixed32Pose,
        translated_pose: Fixed32Pose,
    ):
        """Test simple lerp convenience method."""
        result = blender.lerp_poses(identity_pose, translated_pose, FIXED32_HALF)
        root = result.get_bone_transform(0)
        assert abs(root.translation.x.as_float - 0.5) < 0.01

    def test_blend_different_skeleton_raises(
        self,
        blender: DeterministicBoneBlend,
        identity_pose: Fixed32Pose,
    ):
        """Test that blending with different skeleton raises error."""
        other_skeleton = Skeleton(name="other")
        other_skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))
        other_pose = Fixed32Pose(skeleton=other_skeleton)

        with pytest.raises(ValueError):
            blender.blend_poses(identity_pose, other_pose, FIXED32_HALF)


# =============================================================================
# Fixed32BoneMask Tests
# =============================================================================

class TestFixed32BoneMask:
    """Tests for Fixed32BoneMask operations."""

    def test_default_weight(self):
        """Test default weight is zero."""
        mask = Fixed32BoneMask()
        assert mask.get_weight(0) == FIXED32_ZERO

    def test_set_weight(self):
        """Test setting bone weight."""
        mask = Fixed32BoneMask()
        mask.set_weight(5, Fixed32(0.75))
        assert abs(mask.get_weight(5).as_float - 0.75) < 0.01

    def test_include_bone(self):
        """Test including bone at full weight."""
        mask = Fixed32BoneMask()
        mask.include_bone(3)
        assert mask.get_weight(3) == FIXED32_ONE

    def test_exclude_bone(self):
        """Test excluding bone."""
        mask = Fixed32BoneMask()
        mask.include_bone(3)
        mask.exclude_bone(3)
        assert mask.get_weight(3) == FIXED32_ZERO

    def test_full_body_mask(self, simple_skeleton: Skeleton):
        """Test full body mask creation."""
        mask = Fixed32BoneMask.full_body(simple_skeleton)
        for i in range(simple_skeleton.bone_count):
            assert mask.get_weight(i) == FIXED32_ONE

    def test_weight_clamping(self):
        """Test that weights are clamped to [0, 1]."""
        mask = Fixed32BoneMask()
        mask.set_weight(0, Fixed32(2.0))  # Should clamp to 1
        mask.set_weight(1, Fixed32(-0.5))  # Should clamp to 0

        assert mask.get_weight(0) == FIXED32_ONE
        assert mask.get_weight(1) == FIXED32_ZERO

    def test_copy(self):
        """Test mask copying."""
        mask = Fixed32BoneMask()
        mask.include_bone(0)
        mask.set_weight(1, FIXED32_HALF)

        copy = mask.copy()
        assert copy.get_weight(0) == FIXED32_ONE
        assert copy.get_weight(1) == FIXED32_HALF


# =============================================================================
# Fixed32AnimationTime Tests
# =============================================================================

class TestFixed32AnimationTime:
    """Tests for Fixed32AnimationTime operations."""

    def test_default_values(self):
        """Test default animation time values."""
        anim = Fixed32AnimationTime()
        assert anim.current_time == FIXED32_ZERO
        assert anim.duration == FIXED32_ONE
        assert anim.speed == FIXED32_ONE
        assert anim.looping is True
        assert anim.is_playing is False

    def test_progress_calculation(self):
        """Test progress calculation."""
        anim = Fixed32AnimationTime(
            current_time=Fixed32(0.5),
            duration=Fixed32(1.0),
        )
        assert abs(anim.progress.as_float - 0.5) < 0.01

    def test_play_pause_stop(self):
        """Test play, pause, stop controls."""
        anim = Fixed32AnimationTime()

        anim.play()
        assert anim.is_playing

        anim.pause()
        assert not anim.is_playing

        anim.play()
        anim.stop()
        assert not anim.is_playing
        assert anim.current_time == FIXED32_ZERO

    def test_seek(self):
        """Test seeking to specific time."""
        anim = Fixed32AnimationTime(duration=Fixed32(2.0))
        anim.seek(Fixed32(1.0))
        assert anim.current_time == Fixed32(1.0)

    def test_seek_normalized(self):
        """Test seeking to normalized position."""
        anim = Fixed32AnimationTime(duration=Fixed32(10.0))
        anim.seek_normalized(FIXED32_HALF)
        assert abs(anim.current_time.as_float - 5.0) < 0.01

    def test_advance_looping(self):
        """Test time advancement with looping."""
        anim = Fixed32AnimationTime(
            duration=Fixed32(1.0),
            looping=True,
        )
        anim.play()

        anim.advance(Fixed32(0.5))
        assert abs(anim.current_time.as_float - 0.5) < 0.01

        anim.advance(Fixed32(0.7))
        # Should wrap: 0.5 + 0.7 = 1.2 -> 0.2
        assert abs(anim.current_time.as_float - 0.2) < 0.01

    def test_advance_non_looping(self):
        """Test time advancement without looping."""
        anim = Fixed32AnimationTime(
            duration=Fixed32(1.0),
            looping=False,
        )
        anim.play()

        anim.advance(Fixed32(0.5))
        anim.advance(Fixed32(0.7))

        # Should clamp at duration
        assert anim.current_time == anim.duration
        assert not anim.is_playing
        assert anim.is_finished

    def test_speed_multiplier(self):
        """Test speed multiplier."""
        anim = Fixed32AnimationTime(
            duration=Fixed32(1.0),
            speed=FIXED32_TWO,
        )
        anim.play()

        anim.advance(Fixed32(0.25))
        # 0.25 * 2 = 0.5
        assert abs(anim.current_time.as_float - 0.5) < 0.01

    def test_copy(self):
        """Test animation time copying."""
        anim = Fixed32AnimationTime(
            current_time=Fixed32(0.5),
            duration=Fixed32(2.0),
        )
        anim.play()

        copy = anim.copy()
        assert copy.current_time == anim.current_time
        assert copy.duration == anim.duration
        assert copy.is_playing == anim.is_playing


# =============================================================================
# Determinism Tests
# =============================================================================

class TestDeterminism:
    """Tests verifying bit-identical results across runs."""

    def test_vec3_operations_deterministic(self):
        """Test that Vec3 operations produce identical results."""
        results = []

        for _ in range(5):
            a = Fixed32Vec3.from_floats(1.5, 2.5, 3.5)
            b = Fixed32Vec3.from_floats(0.5, 0.5, 0.5)
            c = a + b
            d = c * FIXED32_TWO
            e = d.normalized()
            results.append(e.raw_tuple())

        # All results should be identical
        assert all(r == results[0] for r in results)

    def test_quat_slerp_deterministic(self):
        """Test that quaternion slerp is deterministic."""
        results = []

        for _ in range(5):
            a = Fixed32Quat.identity()
            b = Fixed32Quat.from_floats(0.0, 0.707, 0.0, 0.707)
            mid = a.slerp(b, FIXED32_HALF)
            results.append(mid.raw_tuple())

        assert all(r == results[0] for r in results)

    def test_blend_deterministic(
        self,
        blender: DeterministicBoneBlend,
        identity_pose: Fixed32Pose,
        translated_pose: Fixed32Pose,
    ):
        """Test that pose blending is deterministic."""
        results = []

        for _ in range(5):
            result = blender.blend_poses(
                identity_pose, translated_pose,
                Fixed32(0.333333),
            )
            raw = result._transforms[0].translation.raw_tuple()
            results.append(raw)

        assert all(r == results[0] for r in results)

    def test_trig_deterministic(self):
        """Test that trig functions are deterministic."""
        sin_results = []
        cos_results = []

        for _ in range(5):
            angle = Fixed32(1.234)
            sin_results.append(fixed32_sin(angle).raw)
            cos_results.append(fixed32_cos(angle).raw)

        assert all(r == sin_results[0] for r in sin_results)
        assert all(r == cos_results[0] for r in cos_results)

    def test_sqrt_deterministic(self):
        """Test that sqrt is deterministic."""
        results = []

        for _ in range(5):
            val = Fixed32(7.5)
            results.append(fixed32_sqrt(val).raw)

        assert all(r == results[0] for r in results)


# =============================================================================
# Conversion Tests
# =============================================================================

class TestConversion:
    """Tests for pose conversion between float and Fixed32."""

    def test_float_to_fixed32_conversion(self, simple_skeleton: Skeleton):
        """Test converting float Pose to Fixed32Pose."""
        # Create float pose
        float_pose = Pose(skeleton=simple_skeleton, space=PoseSpace.LOCAL)
        float_pose.set_bone_transform(0, BoneTransform(
            translation=Vec3(1.5, 2.5, 3.5),
            rotation=Quat(0.0, 0.0, 0.0, 1.0),
            scale=Vec3(1.0, 1.0, 1.0),
        ))

        # Convert to Fixed32
        fixed_pose = convert_pose_to_fixed32(float_pose, simple_skeleton)

        # Verify
        t = fixed_pose.get_bone_transform(0)
        assert abs(t.translation.x.as_float - 1.5) < 0.001
        assert abs(t.translation.y.as_float - 2.5) < 0.001
        assert abs(t.translation.z.as_float - 3.5) < 0.001

    def test_fixed32_to_float_conversion(self, simple_skeleton: Skeleton):
        """Test converting Fixed32Pose back to float Pose."""
        # Create Fixed32 pose
        fixed_pose = Fixed32Pose(skeleton=simple_skeleton)
        fixed_pose.set_bone_transform(0, Fixed32BoneTransform.from_floats(
            1.5, 2.5, 3.5, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0
        ))

        # Convert to float
        float_pose = convert_fixed32_to_pose(fixed_pose)

        # Verify
        t = float_pose.get_bone_transform(0)
        assert abs(t.translation.x - 1.5) < 0.001
        assert abs(t.translation.y - 2.5) < 0.001
        assert abs(t.translation.z - 3.5) < 0.001

    def test_round_trip_conversion(self, simple_skeleton: Skeleton):
        """Test round-trip conversion preserves values."""
        # Create original
        original = Fixed32Pose(skeleton=simple_skeleton)
        original.set_bone_transform(1, Fixed32BoneTransform.from_floats(
            10.0, 20.0, 30.0, 0.0, 0.707, 0.0, 0.707, 2.0, 2.0, 2.0
        ))

        # Round trip
        float_pose = convert_fixed32_to_pose(original)
        back = convert_pose_to_fixed32(float_pose, simple_skeleton)

        # Compare (allowing for float precision loss)
        t_orig = original.get_bone_transform(1)
        t_back = back.get_bone_transform(1)

        assert abs(t_orig.translation.x.as_float - t_back.translation.x.as_float) < 0.01
        assert abs(t_orig.rotation.y.as_float - t_back.rotation.y.as_float) < 0.01

    def test_space_preservation(self, simple_skeleton: Skeleton):
        """Test that coordinate space is preserved in conversion."""
        fixed_pose = Fixed32Pose(
            skeleton=simple_skeleton,
            space=Fixed32PoseSpace.MODEL,
        )

        float_pose = convert_fixed32_to_pose(fixed_pose)
        assert float_pose.space == PoseSpace.MODEL


# =============================================================================
# Math Function Tests
# =============================================================================

class TestMathFunctions:
    """Tests for Fixed32 math functions."""

    def test_sin_cos_identity(self):
        """Test sin^2 + cos^2 = 1."""
        angle = Fixed32(0.5)
        s = fixed32_sin(angle)
        c = fixed32_cos(angle)
        result = s * s + c * c
        assert abs(result.as_float - 1.0) < 0.05

    def test_sqrt_accuracy(self):
        """Test sqrt accuracy."""
        # Test perfect squares
        assert abs(fixed32_sqrt(Fixed32(4.0)).as_float - 2.0) < 0.01
        assert abs(fixed32_sqrt(Fixed32(9.0)).as_float - 3.0) < 0.01
        assert abs(fixed32_sqrt(Fixed32(16.0)).as_float - 4.0) < 0.01

        # Test arbitrary values
        assert abs(fixed32_sqrt(Fixed32(2.0)).as_float - 1.414) < 0.02

    def test_sqrt_zero(self):
        """Test sqrt of zero."""
        assert fixed32_sqrt(FIXED32_ZERO) == FIXED32_ZERO

    def test_sqrt_negative(self):
        """Test sqrt of negative returns zero."""
        result = fixed32_sqrt(Fixed32(-1.0))
        assert result == FIXED32_ZERO

    def test_acos_bounds(self):
        """Test acos at bounds."""
        assert abs(fixed32_acos(FIXED32_ONE).as_float - 0.0) < 0.01
        assert abs(fixed32_acos(FIXED32_NEG_ONE).as_float - math.pi) < 0.1
        assert abs(fixed32_acos(FIXED32_ZERO).as_float - (math.pi / 2)) < 0.1


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests with full animation scenarios."""

    def test_full_blend_workflow(self, humanoid_skeleton: Skeleton):
        """Test complete blending workflow with humanoid skeleton."""
        blender = DeterministicBoneBlend(humanoid_skeleton)

        # Create idle pose
        idle = Fixed32Pose(skeleton=humanoid_skeleton)

        # Create walk pose with arm swing
        walk = Fixed32Pose(skeleton=humanoid_skeleton)
        arm_idx = humanoid_skeleton.get_bone_index("upperarm_l")
        if arm_idx >= 0:
            walk.set_bone_transform(arm_idx, Fixed32BoneTransform.from_floats(
                0.0, 0.0, 0.5, 0.1, 0.0, 0.0, 0.995, 1.0, 1.0, 1.0
            ))

        # Blend at 60%
        result = blender.blend_poses(idle, walk, Fixed32(0.6))

        # Verify blend happened
        if arm_idx >= 0:
            arm = result.get_bone_transform(arm_idx)
            assert arm.translation.z > FIXED32_ZERO

    def test_layered_animation_blending(self, simple_skeleton: Skeleton):
        """Test layered animation blending."""
        blender = DeterministicBoneBlend(simple_skeleton)

        # Base locomotion
        base = Fixed32Pose(skeleton=simple_skeleton)
        base.set_bone_transform(0, Fixed32BoneTransform.from_floats(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0
        ))

        # Upper body additive (for aiming)
        additive = Fixed32Pose(skeleton=simple_skeleton)
        additive.set_bone_transform(1, Fixed32BoneTransform.from_floats(
            0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.995, 1.0, 1.0, 1.0
        ))

        # Mask for upper body only
        mask = Fixed32BoneMask()
        mask.include_bone(1)  # spine
        mask.include_bone(2)  # head

        # Apply additive layer
        result = blender.blend_poses(
            base, additive, FIXED32_ONE,
            mode=DeterministicBlendMode.ADDITIVE,
            mask=mask,
        )

        # Verify spine was affected
        spine = result.get_bone_transform(1)
        assert spine.translation.z > FIXED32_ZERO

        # Verify root was not affected
        root = result.get_bone_transform(0)
        assert root.translation == Fixed32Vec3.zero()

    def test_animation_time_with_blending(self, simple_skeleton: Skeleton):
        """Test animation time driving blend factor."""
        blender = DeterministicBoneBlend(simple_skeleton)

        pose_a = Fixed32Pose(skeleton=simple_skeleton)
        pose_b = Fixed32Pose(skeleton=simple_skeleton)
        pose_b.set_bone_transform(0, Fixed32BoneTransform.from_floats(
            10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0
        ))

        anim_time = Fixed32AnimationTime(duration=FIXED32_ONE)
        anim_time.play()

        # Simulate frames
        for _ in range(10):
            anim_time.advance(Fixed32(0.1))
            progress = anim_time.progress
            result = blender.blend_poses(pose_a, pose_b, progress)

            # Translation should increase with progress
            root = result.get_bone_transform(0)
            expected = 10.0 * progress.as_float
            assert abs(root.translation.x.as_float - expected) < 0.1
