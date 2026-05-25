"""
Tests for Tier 44: IK_PROCEDURAL decorators.
"""

import pytest

from trinity.decorators.ik_procedural import (
    VALID_BONE_TYPES,
    VALID_IK_SOLVERS,
    ik_chain,
    ik_goal,
    motion_matching,
    procedural_bone,
    ragdoll,
)
from trinity.decorators.registry import Tier, registry


class TestIKChain:
    """Test @ik_chain decorator."""

    def test_basic_application(self):
        """Test basic decorator application with default params."""

        @ik_chain()
        class IKArm:
            pass

        assert hasattr(IKArm, "_ik_chain")
        assert IKArm._ik_chain is True
        assert IKArm._ik_solver == "fabrik"
        assert IKArm._ik_iterations == 10

    def test_custom_solver(self):
        """Test custom solver parameter."""

        @ik_chain(solver="ccd", iterations=5)
        class IKLeg:
            pass

        assert IKLeg._ik_solver == "ccd"
        assert IKLeg._ik_iterations == 5

    def test_all_solvers(self):
        """Test all valid solver types."""
        for solver in VALID_IK_SOLVERS:

            @ik_chain(solver=solver)
            class IKTest:
                pass

            assert IKTest._ik_solver == solver

    def test_invalid_solver(self):
        """Test invalid solver raises ValueError."""
        with pytest.raises(ValueError, match="Invalid solver"):

            @ik_chain(solver="invalid")
            class IKBad:
                pass

    def test_invalid_iterations(self):
        """Test invalid iterations raises ValueError."""
        with pytest.raises(ValueError, match="iterations must be > 0"):

            @ik_chain(iterations=0)
            class IKBad:
                pass

        with pytest.raises(ValueError, match="iterations must be > 0"):

            @ik_chain(iterations=-5)
            class IKBad2:
                pass

    def test_tags(self):
        """Test that tags are applied."""

        @ik_chain(solver="jacobian", iterations=15)
        class IKSpine:
            pass

        assert hasattr(IKSpine, "_tags")
        assert IKSpine._tags["ik_chain"] is True
        assert IKSpine._tags["ik_solver"] == "jacobian"
        assert IKSpine._tags["ik_iterations"] == 15

    def test_registry_registration(self):
        """Test decorator is registered in registry."""
        spec = registry.get("ik_chain")
        assert spec is not None
        assert spec.name == "ik_chain"
        assert spec.tier == Tier.IK_PROCEDURAL

    def test_applied_decorators_tracking(self):
        """Test decorator application is tracked."""

        @ik_chain()
        class IKTest:
            pass

        assert hasattr(IKTest, "_applied_decorators")
        assert "ik_chain" in IKTest._applied_decorators


class TestIKGoal:
    """Test @ik_goal decorator."""

    def test_basic_application(self):
        """Test basic decorator application."""

        @ik_goal()
        class IKTarget:
            pass

        assert hasattr(IKTarget, "_ik_goal")
        assert IKTarget._ik_goal is True
        assert IKTarget._ik_goal_priority == 0
        assert IKTarget._ik_goal_blend_speed == 10.0

    def test_custom_params(self):
        """Test custom parameters."""

        @ik_goal(priority=5, blend_speed=20.0)
        class IKHighPriority:
            pass

        assert IKHighPriority._ik_goal_priority == 5
        assert IKHighPriority._ik_goal_blend_speed == 20.0

    def test_invalid_blend_speed(self):
        """Test invalid blend_speed raises ValueError."""
        with pytest.raises(ValueError, match="blend_speed must be > 0"):

            @ik_goal(blend_speed=0)
            class IKBad:
                pass

        with pytest.raises(ValueError, match="blend_speed must be > 0"):

            @ik_goal(blend_speed=-1.0)
            class IKBad2:
                pass

    def test_tags(self):
        """Test that tags are applied."""

        @ik_goal(priority=3, blend_speed=15.0)
        class IKTest:
            pass

        assert IKTest._tags["ik_goal"] is True
        assert IKTest._tags["ik_goal_priority"] == 3
        assert IKTest._tags["ik_goal_blend_speed"] == 15.0

    def test_registry_registration(self):
        """Test decorator is registered in registry."""
        spec = registry.get("ik_goal")
        assert spec is not None
        assert spec.tier == Tier.IK_PROCEDURAL


class TestProceduralBone:
    """Test @procedural_bone decorator."""

    def test_basic_application(self):
        """Test basic decorator application."""

        @procedural_bone(type="jiggle")
        class BoneJiggle:
            pass

        assert hasattr(BoneJiggle, "_procedural_bone")
        assert BoneJiggle._procedural_bone is True
        assert BoneJiggle._procedural_bone_type == "jiggle"

    def test_all_bone_types(self):
        """Test all valid bone types."""
        for bone_type in VALID_BONE_TYPES:

            @procedural_bone(type=bone_type)
            class BoneTest:
                pass

            assert BoneTest._procedural_bone_type == bone_type

    def test_invalid_bone_type(self):
        """Test invalid bone type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid type"):

            @procedural_bone(type="invalid")
            class BoneBad:
                pass

    def test_tags(self):
        """Test that tags are applied."""

        @procedural_bone(type="spring")
        class BoneTest:
            pass

        assert BoneTest._tags["procedural_bone"] is True
        assert BoneTest._tags["procedural_bone_type"] == "spring"

    def test_registry_registration(self):
        """Test decorator is registered in registry."""
        spec = registry.get("procedural_bone")
        assert spec is not None
        assert spec.tier == Tier.IK_PROCEDURAL


class TestMotionMatching:
    """Test @motion_matching decorator."""

    def test_basic_application(self):
        """Test basic decorator application."""

        @motion_matching(database="locomotion")
        class MotionController:
            pass

        assert hasattr(MotionController, "_motion_matching")
        assert MotionController._motion_matching is True
        assert MotionController._motion_database == "locomotion"
        assert MotionController._motion_trajectory_weight == 1.0
        assert MotionController._motion_pose_weight == 1.0

    def test_custom_weights(self):
        """Test custom weight parameters."""

        @motion_matching(
            database="combat",
            trajectory_weight=2.0,
            pose_weight=0.5,
        )
        class CombatController:
            pass

        assert CombatController._motion_database == "combat"
        assert CombatController._motion_trajectory_weight == 2.0
        assert CombatController._motion_pose_weight == 0.5

    def test_empty_database_validation(self):
        """Test empty database raises ValueError."""
        with pytest.raises(ValueError, match="database must be a non-empty string"):

            @motion_matching(database="")
            class MotionBad:
                pass

    def test_invalid_trajectory_weight(self):
        """Test invalid trajectory_weight raises ValueError."""
        with pytest.raises(ValueError, match="trajectory_weight must be > 0"):

            @motion_matching(database="db", trajectory_weight=0)
            class MotionBad:
                pass

    def test_invalid_pose_weight(self):
        """Test invalid pose_weight raises ValueError."""
        with pytest.raises(ValueError, match="pose_weight must be > 0"):

            @motion_matching(database="db", pose_weight=-1.0)
            class MotionBad:
                pass

    def test_tags(self):
        """Test that tags are applied."""

        @motion_matching(database="test", trajectory_weight=1.5, pose_weight=0.8)
        class MotionTest:
            pass

        assert MotionTest._tags["motion_matching"] is True
        assert MotionTest._tags["motion_database"] == "test"
        assert MotionTest._tags["motion_trajectory_weight"] == 1.5
        assert MotionTest._tags["motion_pose_weight"] == 0.8

    def test_registry_registration(self):
        """Test decorator is registered in registry."""
        spec = registry.get("motion_matching")
        assert spec is not None
        assert spec.tier == Tier.IK_PROCEDURAL


class TestRagdoll:
    """Test @ragdoll decorator."""

    def test_basic_application(self):
        """Test basic decorator application."""

        @ragdoll()
        class RagdollCharacter:
            pass

        assert hasattr(RagdollCharacter, "_ragdoll")
        assert RagdollCharacter._ragdoll is True
        assert RagdollCharacter._ragdoll_blend_time == 0.2
        assert RagdollCharacter._ragdoll_joint_limits is True

    def test_custom_params(self):
        """Test custom parameters."""

        @ragdoll(blend_time=0.5, joint_limits=False)
        class CustomRagdoll:
            pass

        assert CustomRagdoll._ragdoll_blend_time == 0.5
        assert CustomRagdoll._ragdoll_joint_limits is False

    def test_zero_blend_time(self):
        """Test zero blend time is allowed."""

        @ragdoll(blend_time=0.0)
        class InstantRagdoll:
            pass

        assert InstantRagdoll._ragdoll_blend_time == 0.0

    def test_invalid_blend_time(self):
        """Test negative blend_time raises ValueError."""
        with pytest.raises(ValueError, match="blend_time must be >= 0"):

            @ragdoll(blend_time=-0.1)
            class RagdollBad:
                pass

    def test_tags(self):
        """Test that tags are applied."""

        @ragdoll(blend_time=0.3, joint_limits=True)
        class RagdollTest:
            pass

        assert RagdollTest._tags["ragdoll"] is True
        assert RagdollTest._tags["ragdoll_blend_time"] == 0.3
        assert RagdollTest._tags["ragdoll_joint_limits"] is True

    def test_registry_registration(self):
        """Test decorator is registered in registry."""
        spec = registry.get("ragdoll")
        assert spec is not None
        assert spec.tier == Tier.IK_PROCEDURAL


class TestComposition:
    """Test decorator composition."""

    def test_multiple_decorators(self):
        """Test applying multiple IK decorators together."""

        @ragdoll(blend_time=0.3)
        @ik_chain(solver="fabrik", iterations=8)
        @ik_goal(priority=1, blend_speed=15.0)
        class ComplexIKSystem:
            pass

        # All decorators applied
        assert ComplexIKSystem._ik_chain is True
        assert ComplexIKSystem._ik_goal is True
        assert ComplexIKSystem._ragdoll is True

        # All parameters preserved
        assert ComplexIKSystem._ik_solver == "fabrik"
        assert ComplexIKSystem._ik_iterations == 8
        assert ComplexIKSystem._ik_goal_priority == 1
        assert ComplexIKSystem._ik_goal_blend_speed == 15.0
        assert ComplexIKSystem._ragdoll_blend_time == 0.3

    def test_procedural_bone_with_motion_matching(self):
        """Test combining procedural bone with motion matching."""

        @motion_matching(database="animation_db", trajectory_weight=1.5)
        @procedural_bone(type="lookat")
        class AnimatedBone:
            pass

        assert AnimatedBone._procedural_bone is True
        assert AnimatedBone._motion_matching is True
        assert AnimatedBone._procedural_bone_type == "lookat"
        assert AnimatedBone._motion_database == "animation_db"


class TestRegistryIntegration:
    """Test registry integration for all decorators."""

    def test_all_decorators_registered(self):
        """Test all decorators are registered in tier 44."""
        tier_specs = registry.by_tier(Tier.IK_PROCEDURAL)
        decorator_names = {spec.name for spec in tier_specs}

        expected = {
            "ik_chain",
            "ik_goal",
            "procedural_bone",
            "motion_matching",
            "ragdoll",
        }

        assert expected.issubset(decorator_names)

    def test_decorator_metadata(self):
        """Test decorator metadata is correct."""
        for name in ["ik_chain", "ik_goal", "procedural_bone", "motion_matching", "ragdoll"]:
            spec = registry.get(name)
            assert spec is not None
            assert spec.tier == Tier.IK_PROCEDURAL
            assert spec.foundation is False
            assert "class" in spec.target_types
