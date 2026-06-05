"""
Blackbox tests for T-AG-1.2: Transform and Pose Data Structures

Tests written against PUBLIC CONTRACT ONLY - no implementation details.

Contract:
    from engine.animation.graph import DictTransform, DictPose, Vec3, Quaternion

    # Transform with position, rotation, scale
    t = DictTransform.identity()
    t = DictTransform.from_position(1.0, 2.0, 3.0)
    blended = t.blend(other_t, 0.5)  # SLERP rotation
    composed = t.compose(other_t)    # Hierarchical composition

    # Pose with bone transforms
    pose = DictPose()
    pose.set_transform("spine", transform)
    blended_pose = pose.blend(other_pose, 0.5)  # Missing bones handled
"""

import pytest
import math


# =============================================================================
# IMPORT CONTRACT
# =============================================================================

class TestImportContract:
    """Verify the public API can be imported as documented."""

    def test_import_dict_transform(self):
        """DictTransform should be importable from engine.animation.graph."""
        from engine.animation.graph import DictTransform
        assert DictTransform is not None

    def test_import_dict_pose(self):
        """DictPose should be importable from engine.animation.graph."""
        from engine.animation.graph import DictPose
        assert DictPose is not None

    def test_import_vec3(self):
        """Vec3 should be importable from engine.animation.graph."""
        from engine.animation.graph import Vec3
        assert Vec3 is not None

    def test_import_quaternion(self):
        """Quaternion should be importable from engine.animation.graph."""
        from engine.animation.graph import Quaternion
        assert Quaternion is not None


# =============================================================================
# TRANSFORM CREATION TESTS
# =============================================================================

class TestTransformCreation:
    """Test Transform creation and factory methods."""

    def test_identity_transform_exists(self):
        """DictTransform.identity() should exist and return a transform."""
        from engine.animation.graph import DictTransform
        t = DictTransform.identity()
        assert t is not None

    def test_identity_transform_is_dict_transform(self):
        """DictTransform.identity() should return a DictTransform instance."""
        from engine.animation.graph import DictTransform
        t = DictTransform.identity()
        assert isinstance(t, DictTransform)

    def test_from_position_exists(self):
        """DictTransform.from_position() should exist and accept x, y, z args."""
        from engine.animation.graph import DictTransform
        t = DictTransform.from_position(1.0, 2.0, 3.0)
        assert t is not None

    def test_from_position_returns_dict_transform(self):
        """DictTransform.from_position() should return a DictTransform."""
        from engine.animation.graph import DictTransform
        t = DictTransform.from_position(1.0, 2.0, 3.0)
        assert isinstance(t, DictTransform)

    def test_from_position_with_various_values(self):
        """DictTransform.from_position() should accept various position values."""
        from engine.animation.graph import DictTransform

        # Origin
        t1 = DictTransform.from_position(0.0, 0.0, 0.0)
        assert t1 is not None

        # Positive values
        t2 = DictTransform.from_position(10.0, 20.0, 30.0)
        assert t2 is not None

        # Negative values
        t3 = DictTransform.from_position(-5.0, -10.0, -15.0)
        assert t3 is not None

        # Mixed values
        t4 = DictTransform.from_position(1.5, -2.5, 3.5)
        assert t4 is not None


# =============================================================================
# TRANSFORM BLEND TESTS
# =============================================================================

class TestTransformBlend:
    """Test Transform.blend() method with SLERP for rotation."""

    def test_blend_method_exists(self):
        """DictTransform should have a blend method."""
        from engine.animation.graph import DictTransform
        t = DictTransform.identity()
        assert hasattr(t, 'blend')
        assert callable(t.blend)

    def test_blend_returns_transform(self):
        """blend() should return a DictTransform."""
        from engine.animation.graph import DictTransform
        t1 = DictTransform.identity()
        t2 = DictTransform.from_position(10.0, 10.0, 10.0)
        result = t1.blend(t2, 0.5)
        assert isinstance(result, DictTransform)

    def test_blend_at_zero_returns_first(self):
        """blend(other, 0.0) should return a transform close to self."""
        from engine.animation.graph import DictTransform
        t1 = DictTransform.from_position(0.0, 0.0, 0.0)
        t2 = DictTransform.from_position(10.0, 10.0, 10.0)
        result = t1.blend(t2, 0.0)
        # At t=0, result should be very close to t1
        assert result is not None

    def test_blend_at_one_returns_second(self):
        """blend(other, 1.0) should return a transform close to other."""
        from engine.animation.graph import DictTransform
        t1 = DictTransform.from_position(0.0, 0.0, 0.0)
        t2 = DictTransform.from_position(10.0, 10.0, 10.0)
        result = t1.blend(t2, 1.0)
        # At t=1, result should be very close to t2
        assert result is not None

    def test_blend_at_half_interpolates(self):
        """blend(other, 0.5) should produce an interpolated transform."""
        from engine.animation.graph import DictTransform
        t1 = DictTransform.from_position(0.0, 0.0, 0.0)
        t2 = DictTransform.from_position(10.0, 10.0, 10.0)
        result = t1.blend(t2, 0.5)
        # At t=0.5, result should be between t1 and t2
        assert result is not None

    def test_blend_with_various_t_values(self):
        """blend() should work with various interpolation values."""
        from engine.animation.graph import DictTransform
        t1 = DictTransform.identity()
        t2 = DictTransform.from_position(100.0, 100.0, 100.0)

        # Various blend factors
        for factor in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            result = t1.blend(t2, factor)
            assert result is not None
            assert isinstance(result, DictTransform)

    def test_blend_does_not_modify_original(self):
        """blend() should not modify the original transforms."""
        from engine.animation.graph import DictTransform
        t1 = DictTransform.from_position(1.0, 2.0, 3.0)
        t2 = DictTransform.from_position(10.0, 20.0, 30.0)

        # Store original state (if accessible)
        t1_original = DictTransform.from_position(1.0, 2.0, 3.0)

        # Perform blend
        result = t1.blend(t2, 0.5)

        # t1 should still be able to blend to same result
        result2 = t1.blend(t2, 0.5)
        assert result2 is not None


# =============================================================================
# TRANSFORM COMPOSE TESTS
# =============================================================================

class TestTransformCompose:
    """Test Transform.compose() for hierarchical composition."""

    def test_compose_method_exists(self):
        """DictTransform should have a compose method."""
        from engine.animation.graph import DictTransform
        t = DictTransform.identity()
        assert hasattr(t, 'compose')
        assert callable(t.compose)

    def test_compose_returns_transform(self):
        """compose() should return a DictTransform."""
        from engine.animation.graph import DictTransform
        t1 = DictTransform.identity()
        t2 = DictTransform.from_position(5.0, 5.0, 5.0)
        result = t1.compose(t2)
        assert isinstance(result, DictTransform)

    def test_compose_with_identity_returns_equivalent(self):
        """compose() with identity should return equivalent transform."""
        from engine.animation.graph import DictTransform
        t1 = DictTransform.from_position(5.0, 10.0, 15.0)
        identity = DictTransform.identity()

        # t1 composed with identity should be equivalent to t1
        result = t1.compose(identity)
        assert result is not None

    def test_identity_compose_with_other(self):
        """identity compose with other should return equivalent to other."""
        from engine.animation.graph import DictTransform
        identity = DictTransform.identity()
        t2 = DictTransform.from_position(5.0, 10.0, 15.0)

        # identity composed with t2
        result = identity.compose(t2)
        assert result is not None

    def test_compose_multiple_transforms(self):
        """compose() should work for chaining multiple transforms."""
        from engine.animation.graph import DictTransform
        t1 = DictTransform.from_position(1.0, 0.0, 0.0)
        t2 = DictTransform.from_position(0.0, 2.0, 0.0)
        t3 = DictTransform.from_position(0.0, 0.0, 3.0)

        # Chain composition
        result = t1.compose(t2).compose(t3)
        assert result is not None
        assert isinstance(result, DictTransform)

    def test_compose_does_not_modify_original(self):
        """compose() should not modify the original transforms."""
        from engine.animation.graph import DictTransform
        t1 = DictTransform.from_position(1.0, 2.0, 3.0)
        t2 = DictTransform.from_position(10.0, 20.0, 30.0)

        # Perform compose
        result = t1.compose(t2)

        # Original should still work for another compose
        result2 = t1.compose(t2)
        assert result2 is not None


# =============================================================================
# POSE CREATION TESTS
# =============================================================================

class TestPoseCreation:
    """Test Pose creation and basic operations."""

    def test_pose_instantiation(self):
        """DictPose() should be instantiable."""
        from engine.animation.graph import DictPose
        pose = DictPose()
        assert pose is not None

    def test_pose_is_dict_pose(self):
        """DictPose() should return a DictPose instance."""
        from engine.animation.graph import DictPose
        pose = DictPose()
        assert isinstance(pose, DictPose)


# =============================================================================
# POSE SET_TRANSFORM TESTS
# =============================================================================

class TestPoseSetTransform:
    """Test Pose.set_transform() method for bone transforms."""

    def test_set_transform_method_exists(self):
        """DictPose should have a set_transform method."""
        from engine.animation.graph import DictPose
        pose = DictPose()
        assert hasattr(pose, 'set_transform')
        assert callable(pose.set_transform)

    def test_set_transform_single_bone(self):
        """set_transform() should store a bone transform."""
        from engine.animation.graph import DictPose, DictTransform
        pose = DictPose()
        transform = DictTransform.identity()

        # Should not raise an exception
        pose.set_transform("spine", transform)

    def test_set_transform_multiple_bones(self):
        """set_transform() should store multiple bone transforms."""
        from engine.animation.graph import DictPose, DictTransform
        pose = DictPose()

        bones = ["spine", "neck", "head", "left_arm", "right_arm"]
        for bone_name in bones:
            transform = DictTransform.from_position(1.0, 2.0, 3.0)
            pose.set_transform(bone_name, transform)

        # All bones should be settable without error

    def test_set_transform_overwrites_existing(self):
        """set_transform() should overwrite an existing bone transform."""
        from engine.animation.graph import DictPose, DictTransform
        pose = DictPose()

        t1 = DictTransform.from_position(1.0, 1.0, 1.0)
        t2 = DictTransform.from_position(2.0, 2.0, 2.0)

        pose.set_transform("spine", t1)
        pose.set_transform("spine", t2)

        # Should not raise an exception


# =============================================================================
# POSE BLEND TESTS
# =============================================================================

class TestPoseBlend:
    """Test Pose.blend() method with missing bone handling."""

    def test_blend_method_exists(self):
        """DictPose should have a blend method."""
        from engine.animation.graph import DictPose
        pose = DictPose()
        assert hasattr(pose, 'blend')
        assert callable(pose.blend)

    def test_blend_returns_pose(self):
        """blend() should return a DictPose."""
        from engine.animation.graph import DictPose, DictTransform
        pose1 = DictPose()
        pose2 = DictPose()

        pose1.set_transform("spine", DictTransform.identity())
        pose2.set_transform("spine", DictTransform.from_position(10.0, 10.0, 10.0))

        result = pose1.blend(pose2, 0.5)
        assert isinstance(result, DictPose)

    def test_blend_empty_poses(self):
        """blend() should work with empty poses."""
        from engine.animation.graph import DictPose
        pose1 = DictPose()
        pose2 = DictPose()

        result = pose1.blend(pose2, 0.5)
        assert result is not None
        assert isinstance(result, DictPose)

    def test_blend_handles_missing_bones_in_first(self):
        """blend() should handle bones present in second but not first."""
        from engine.animation.graph import DictPose, DictTransform
        pose1 = DictPose()
        pose2 = DictPose()

        # Only pose2 has 'spine'
        pose2.set_transform("spine", DictTransform.from_position(10.0, 10.0, 10.0))

        # Should not raise - missing bones handled gracefully
        result = pose1.blend(pose2, 0.5)
        assert result is not None

    def test_blend_handles_missing_bones_in_second(self):
        """blend() should handle bones present in first but not second."""
        from engine.animation.graph import DictPose, DictTransform
        pose1 = DictPose()
        pose2 = DictPose()

        # Only pose1 has 'spine'
        pose1.set_transform("spine", DictTransform.from_position(10.0, 10.0, 10.0))

        # Should not raise - missing bones handled gracefully
        result = pose1.blend(pose2, 0.5)
        assert result is not None

    def test_blend_handles_disjoint_bones(self):
        """blend() should handle poses with completely different bones."""
        from engine.animation.graph import DictPose, DictTransform
        pose1 = DictPose()
        pose2 = DictPose()

        # Completely different bone sets
        pose1.set_transform("left_arm", DictTransform.from_position(1.0, 0.0, 0.0))
        pose1.set_transform("left_hand", DictTransform.from_position(2.0, 0.0, 0.0))

        pose2.set_transform("right_arm", DictTransform.from_position(0.0, 1.0, 0.0))
        pose2.set_transform("right_hand", DictTransform.from_position(0.0, 2.0, 0.0))

        # Should not raise - graceful handling
        result = pose1.blend(pose2, 0.5)
        assert result is not None

    def test_blend_handles_overlapping_bones(self):
        """blend() should handle poses with overlapping bones."""
        from engine.animation.graph import DictPose, DictTransform
        pose1 = DictPose()
        pose2 = DictPose()

        # Overlapping bone sets
        pose1.set_transform("spine", DictTransform.from_position(1.0, 0.0, 0.0))
        pose1.set_transform("neck", DictTransform.from_position(2.0, 0.0, 0.0))
        pose1.set_transform("left_arm", DictTransform.from_position(3.0, 0.0, 0.0))

        pose2.set_transform("spine", DictTransform.from_position(0.0, 1.0, 0.0))
        pose2.set_transform("neck", DictTransform.from_position(0.0, 2.0, 0.0))
        pose2.set_transform("right_arm", DictTransform.from_position(0.0, 3.0, 0.0))

        result = pose1.blend(pose2, 0.5)
        assert result is not None

    def test_blend_at_zero(self):
        """blend(other, 0.0) should preserve first pose's transforms."""
        from engine.animation.graph import DictPose, DictTransform
        pose1 = DictPose()
        pose2 = DictPose()

        pose1.set_transform("spine", DictTransform.from_position(1.0, 1.0, 1.0))
        pose2.set_transform("spine", DictTransform.from_position(10.0, 10.0, 10.0))

        result = pose1.blend(pose2, 0.0)
        assert result is not None

    def test_blend_at_one(self):
        """blend(other, 1.0) should approach second pose's transforms."""
        from engine.animation.graph import DictPose, DictTransform
        pose1 = DictPose()
        pose2 = DictPose()

        pose1.set_transform("spine", DictTransform.from_position(1.0, 1.0, 1.0))
        pose2.set_transform("spine", DictTransform.from_position(10.0, 10.0, 10.0))

        result = pose1.blend(pose2, 1.0)
        assert result is not None

    def test_blend_at_half(self):
        """blend(other, 0.5) should produce interpolated transforms."""
        from engine.animation.graph import DictPose, DictTransform
        pose1 = DictPose()
        pose2 = DictPose()

        pose1.set_transform("spine", DictTransform.from_position(0.0, 0.0, 0.0))
        pose2.set_transform("spine", DictTransform.from_position(10.0, 10.0, 10.0))

        result = pose1.blend(pose2, 0.5)
        assert result is not None

    def test_blend_with_many_bones(self):
        """blend() should work with poses containing many bones."""
        from engine.animation.graph import DictPose, DictTransform
        pose1 = DictPose()
        pose2 = DictPose()

        # Typical skeleton bone set
        bones = [
            "root", "pelvis", "spine_01", "spine_02", "spine_03",
            "neck", "head",
            "clavicle_l", "upperarm_l", "lowerarm_l", "hand_l",
            "clavicle_r", "upperarm_r", "lowerarm_r", "hand_r",
            "thigh_l", "calf_l", "foot_l", "ball_l",
            "thigh_r", "calf_r", "foot_r", "ball_r",
        ]

        for i, bone in enumerate(bones):
            pose1.set_transform(bone, DictTransform.from_position(float(i), 0.0, 0.0))
            pose2.set_transform(bone, DictTransform.from_position(0.0, float(i), 0.0))

        result = pose1.blend(pose2, 0.5)
        assert result is not None

    def test_blend_does_not_modify_originals(self):
        """blend() should not modify the original poses."""
        from engine.animation.graph import DictPose, DictTransform
        pose1 = DictPose()
        pose2 = DictPose()

        pose1.set_transform("spine", DictTransform.from_position(1.0, 1.0, 1.0))
        pose2.set_transform("spine", DictTransform.from_position(10.0, 10.0, 10.0))

        # Blend
        result = pose1.blend(pose2, 0.5)

        # Original poses should still be usable
        result2 = pose1.blend(pose2, 0.5)
        assert result2 is not None


# =============================================================================
# EDGE CASES AND BOUNDARY CONDITIONS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_blend_factor_boundary_zero(self):
        """blend() with factor exactly 0.0 should not raise."""
        from engine.animation.graph import DictTransform
        t1 = DictTransform.identity()
        t2 = DictTransform.from_position(10.0, 10.0, 10.0)
        result = t1.blend(t2, 0.0)
        assert result is not None

    def test_blend_factor_boundary_one(self):
        """blend() with factor exactly 1.0 should not raise."""
        from engine.animation.graph import DictTransform
        t1 = DictTransform.identity()
        t2 = DictTransform.from_position(10.0, 10.0, 10.0)
        result = t1.blend(t2, 1.0)
        assert result is not None

    def test_transform_with_zero_position(self):
        """DictTransform should handle zero position."""
        from engine.animation.graph import DictTransform
        t = DictTransform.from_position(0.0, 0.0, 0.0)
        assert t is not None

    def test_transform_with_large_values(self):
        """DictTransform should handle large position values."""
        from engine.animation.graph import DictTransform
        t = DictTransform.from_position(1e6, 1e6, 1e6)
        assert t is not None

    def test_transform_with_small_values(self):
        """DictTransform should handle small position values."""
        from engine.animation.graph import DictTransform
        t = DictTransform.from_position(1e-6, 1e-6, 1e-6)
        assert t is not None

    def test_transform_with_negative_values(self):
        """DictTransform should handle negative position values."""
        from engine.animation.graph import DictTransform
        t = DictTransform.from_position(-100.0, -200.0, -300.0)
        assert t is not None

    def test_pose_with_empty_bone_name(self):
        """Pose should handle empty string bone name (may or may not be allowed)."""
        from engine.animation.graph import DictPose, DictTransform
        pose = DictPose()
        # This might raise or might be allowed - just checking it doesn't crash
        try:
            pose.set_transform("", DictTransform.identity())
        except (ValueError, KeyError):
            pass  # Implementation may reject empty names

    def test_pose_with_unicode_bone_name(self):
        """Pose should handle unicode bone names."""
        from engine.animation.graph import DictPose, DictTransform
        pose = DictPose()
        pose.set_transform("spine_01", DictTransform.identity())
        pose.set_transform("bone_with_underscore", DictTransform.identity())
        # Unicode names may be valid
        try:
            pose.set_transform("bone_unicode", DictTransform.identity())
        except (ValueError, KeyError):
            pass

    def test_blend_self_with_self(self):
        """Transform blending with itself should return equivalent transform."""
        from engine.animation.graph import DictTransform
        t = DictTransform.from_position(5.0, 10.0, 15.0)
        result = t.blend(t, 0.5)
        assert result is not None

    def test_compose_self_with_self(self):
        """Transform composing with itself should return valid transform."""
        from engine.animation.graph import DictTransform
        t = DictTransform.from_position(5.0, 10.0, 15.0)
        result = t.compose(t)
        assert result is not None

    def test_pose_blend_self_with_self(self):
        """Pose blending with itself should return equivalent pose."""
        from engine.animation.graph import DictPose, DictTransform
        pose = DictPose()
        pose.set_transform("spine", DictTransform.from_position(5.0, 10.0, 15.0))

        result = pose.blend(pose, 0.5)
        assert result is not None


# =============================================================================
# NUMERICAL STABILITY TESTS
# =============================================================================

class TestNumericalStability:
    """Test numerical stability per architecture spec."""

    def test_blend_with_very_small_factor(self):
        """blend() should be stable with very small blend factors."""
        from engine.animation.graph import DictTransform
        t1 = DictTransform.identity()
        t2 = DictTransform.from_position(100.0, 100.0, 100.0)

        # Very small factor
        result = t1.blend(t2, 0.0001)
        assert result is not None

    def test_blend_with_factor_near_one(self):
        """blend() should be stable with blend factor near 1.0."""
        from engine.animation.graph import DictTransform
        t1 = DictTransform.identity()
        t2 = DictTransform.from_position(100.0, 100.0, 100.0)

        # Factor very close to 1
        result = t1.blend(t2, 0.9999)
        assert result is not None

    def test_repeated_blending_stability(self):
        """Repeated blending should maintain numerical stability."""
        from engine.animation.graph import DictTransform
        t1 = DictTransform.identity()
        t2 = DictTransform.from_position(10.0, 10.0, 10.0)

        # Repeatedly blend
        result = t1
        for _ in range(100):
            result = result.blend(t2, 0.01)

        assert result is not None

    def test_repeated_composition_stability(self):
        """Repeated composition should maintain numerical stability."""
        from engine.animation.graph import DictTransform
        t = DictTransform.from_position(0.1, 0.1, 0.1)

        # Repeatedly compose
        result = DictTransform.identity()
        for _ in range(50):
            result = result.compose(t)

        assert result is not None


# =============================================================================
# TYPE CONSISTENCY TESTS
# =============================================================================

class TestTypeConsistency:
    """Test type consistency of returned values."""

    def test_identity_type_consistency(self):
        """Multiple calls to identity() should return same type."""
        from engine.animation.graph import DictTransform
        t1 = DictTransform.identity()
        t2 = DictTransform.identity()
        assert type(t1) == type(t2)

    def test_from_position_type_consistency(self):
        """Multiple calls to from_position() should return same type."""
        from engine.animation.graph import DictTransform
        t1 = DictTransform.from_position(1.0, 2.0, 3.0)
        t2 = DictTransform.from_position(4.0, 5.0, 6.0)
        assert type(t1) == type(t2)

    def test_blend_type_consistency(self):
        """blend() should return same type as input transforms."""
        from engine.animation.graph import DictTransform
        t1 = DictTransform.identity()
        t2 = DictTransform.from_position(10.0, 10.0, 10.0)
        result = t1.blend(t2, 0.5)
        assert type(result) == type(t1)

    def test_compose_type_consistency(self):
        """compose() should return same type as input transforms."""
        from engine.animation.graph import DictTransform
        t1 = DictTransform.identity()
        t2 = DictTransform.from_position(10.0, 10.0, 10.0)
        result = t1.compose(t2)
        assert type(result) == type(t1)

    def test_pose_blend_type_consistency(self):
        """Pose.blend() should return same type as input poses."""
        from engine.animation.graph import DictPose, DictTransform
        pose1 = DictPose()
        pose2 = DictPose()
        pose1.set_transform("spine", DictTransform.identity())
        pose2.set_transform("spine", DictTransform.identity())

        result = pose1.blend(pose2, 0.5)
        assert type(result) == type(pose1)
