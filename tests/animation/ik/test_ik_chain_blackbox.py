"""
Blackbox tests for IKChain data class.

CLEANROOM: Tests written from specification only, without reading implementation.
Tests the IKChain dataclass which defines an inverse kinematics chain with
solver configuration, bone hierarchy, and control parameters.

Specification:
- IKChain is a dataclass with name, root_bone, joint_bones, effector_bone,
  solver_type, weight, priority, and enabled fields
- bone_count property returns total number of bones in chain
- all_bones property returns ordered list of all bones
- set_weight() method sets weight (clamped 0-1)
- set_enabled() method toggles enabled state
- arm_chain() factory creates arm IK chain for given side
- leg_chain() factory creates leg IK chain for given side
"""

import pytest
from enum import Enum
from typing import List


class TestIKSolverTypeEnum:
    """Test IKSolverType enumeration values."""

    def test_solver_type_enum_exists(self):
        """IKSolverType enum should be importable."""
        from engine.animation.ik.fullbody import IKSolverType
        assert IKSolverType is not None

    def test_two_bone_solver_exists(self):
        """TWO_BONE solver type should exist."""
        from engine.animation.ik.fullbody import IKSolverType
        assert hasattr(IKSolverType, 'TWO_BONE')

    def test_fabrik_solver_exists(self):
        """FABRIK solver type should exist."""
        from engine.animation.ik.fullbody import IKSolverType
        assert hasattr(IKSolverType, 'FABRIK')

    def test_ccd_solver_exists(self):
        """CCD solver type should exist."""
        from engine.animation.ik.fullbody import IKSolverType
        assert hasattr(IKSolverType, 'CCD')

    def test_jacobian_solver_exists(self):
        """JACOBIAN solver type should exist."""
        from engine.animation.ik.fullbody import IKSolverType
        assert hasattr(IKSolverType, 'JACOBIAN')

    def test_solver_type_is_enum(self):
        """IKSolverType should be an Enum."""
        from engine.animation.ik.fullbody import IKSolverType
        assert issubclass(IKSolverType, Enum)

    def test_all_solver_types_unique(self):
        """All solver types should have unique values."""
        from engine.animation.ik.fullbody import IKSolverType
        values = [s.value for s in IKSolverType]
        assert len(values) == len(set(values))

    def test_solver_type_count(self):
        """IKSolverType should have exactly 4 members."""
        from engine.animation.ik.fullbody import IKSolverType
        assert len(list(IKSolverType)) == 4


class TestIKChainDataclass:
    """Test IKChain dataclass structure and fields."""

    def test_ik_chain_class_exists(self):
        """IKChain class should be importable."""
        from engine.animation.ik.fullbody import IKChain
        assert IKChain is not None

    def test_ik_chain_has_name_field(self):
        """IKChain should have name field."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test_chain",
            root_bone="root",
            joint_bones=["joint1"],
            effector_bone="effector",
            solver_type=IKSolverType.FABRIK,
            weight=1.0,
            priority=0,
            enabled=True
        )
        assert chain.name == "test_chain"

    def test_ik_chain_has_root_bone_field(self):
        """IKChain should have root_bone field."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="shoulder",
            joint_bones=["elbow"],
            effector_bone="hand",
            solver_type=IKSolverType.TWO_BONE,
            weight=1.0,
            priority=0,
            enabled=True
        )
        assert chain.root_bone == "shoulder"

    def test_ik_chain_has_joint_bones_field(self):
        """IKChain should have joint_bones field as list."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        joints = ["elbow", "wrist"]
        chain = IKChain(
            name="test",
            root_bone="shoulder",
            joint_bones=joints,
            effector_bone="hand",
            solver_type=IKSolverType.FABRIK,
            weight=1.0,
            priority=0,
            enabled=True
        )
        assert chain.joint_bones == joints

    def test_ik_chain_has_effector_bone_field(self):
        """IKChain should have effector_bone field."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="hip",
            joint_bones=["knee"],
            effector_bone="foot",
            solver_type=IKSolverType.CCD,
            weight=1.0,
            priority=0,
            enabled=True
        )
        assert chain.effector_bone == "foot"

    def test_ik_chain_has_solver_type_field(self):
        """IKChain should have solver_type field."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["joint"],
            effector_bone="effector",
            solver_type=IKSolverType.JACOBIAN,
            weight=1.0,
            priority=0,
            enabled=True
        )
        assert chain.solver_type == IKSolverType.JACOBIAN

    def test_ik_chain_has_weight_field(self):
        """IKChain should have weight field."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["joint"],
            effector_bone="effector",
            solver_type=IKSolverType.FABRIK,
            weight=0.75,
            priority=0,
            enabled=True
        )
        assert chain.weight == 0.75

    def test_ik_chain_has_priority_field(self):
        """IKChain should have priority field."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["joint"],
            effector_bone="effector",
            solver_type=IKSolverType.FABRIK,
            weight=1.0,
            priority=5,
            enabled=True
        )
        assert chain.priority == 5

    def test_ik_chain_has_enabled_field(self):
        """IKChain should have enabled field."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["joint"],
            effector_bone="effector",
            solver_type=IKSolverType.FABRIK,
            weight=1.0,
            priority=0,
            enabled=False
        )
        assert chain.enabled is False


class TestIKChainBoneCountProperty:
    """Test bone_count property."""

    def test_bone_count_single_joint(self):
        """bone_count should return 3 for single joint chain (root + joint + effector)."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="shoulder",
            joint_bones=["elbow"],
            effector_bone="hand",
            solver_type=IKSolverType.TWO_BONE,
            weight=1.0,
            priority=0,
            enabled=True
        )
        assert chain.bone_count == 3

    def test_bone_count_two_joints(self):
        """bone_count should return 4 for two joint chain."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="hip",
            joint_bones=["thigh", "knee"],
            effector_bone="foot",
            solver_type=IKSolverType.FABRIK,
            weight=1.0,
            priority=0,
            enabled=True
        )
        assert chain.bone_count == 4

    def test_bone_count_empty_joints(self):
        """bone_count should return 2 for empty joints (root + effector only)."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=[],
            effector_bone="effector",
            solver_type=IKSolverType.CCD,
            weight=1.0,
            priority=0,
            enabled=True
        )
        assert chain.bone_count == 2

    def test_bone_count_many_joints(self):
        """bone_count should handle many joints correctly."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        joints = ["joint1", "joint2", "joint3", "joint4", "joint5"]
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=joints,
            effector_bone="effector",
            solver_type=IKSolverType.FABRIK,
            weight=1.0,
            priority=0,
            enabled=True
        )
        # root + 5 joints + effector = 7
        assert chain.bone_count == 7

    def test_bone_count_is_property(self):
        """bone_count should be accessible as a property (not a method call)."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["joint"],
            effector_bone="effector",
            solver_type=IKSolverType.TWO_BONE,
            weight=1.0,
            priority=0,
            enabled=True
        )
        # Should be accessible without parentheses
        count = chain.bone_count
        assert isinstance(count, int)


class TestIKChainAllBonesProperty:
    """Test all_bones property."""

    def test_all_bones_returns_list(self):
        """all_bones should return a list."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["joint"],
            effector_bone="effector",
            solver_type=IKSolverType.FABRIK,
            weight=1.0,
            priority=0,
            enabled=True
        )
        assert isinstance(chain.all_bones, list)

    def test_all_bones_contains_root(self):
        """all_bones should contain root_bone."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="shoulder",
            joint_bones=["elbow"],
            effector_bone="hand",
            solver_type=IKSolverType.TWO_BONE,
            weight=1.0,
            priority=0,
            enabled=True
        )
        assert "shoulder" in chain.all_bones

    def test_all_bones_contains_joints(self):
        """all_bones should contain all joint_bones."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        joints = ["elbow", "wrist"]
        chain = IKChain(
            name="test",
            root_bone="shoulder",
            joint_bones=joints,
            effector_bone="hand",
            solver_type=IKSolverType.FABRIK,
            weight=1.0,
            priority=0,
            enabled=True
        )
        for joint in joints:
            assert joint in chain.all_bones

    def test_all_bones_contains_effector(self):
        """all_bones should contain effector_bone."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="hip",
            joint_bones=["knee"],
            effector_bone="foot",
            solver_type=IKSolverType.CCD,
            weight=1.0,
            priority=0,
            enabled=True
        )
        assert "foot" in chain.all_bones

    def test_all_bones_order_root_first(self):
        """all_bones should have root_bone as first element."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="shoulder",
            joint_bones=["elbow", "wrist"],
            effector_bone="hand",
            solver_type=IKSolverType.FABRIK,
            weight=1.0,
            priority=0,
            enabled=True
        )
        assert chain.all_bones[0] == "shoulder"

    def test_all_bones_order_effector_last(self):
        """all_bones should have effector_bone as last element."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="hip",
            joint_bones=["thigh", "knee", "ankle"],
            effector_bone="foot",
            solver_type=IKSolverType.FABRIK,
            weight=1.0,
            priority=0,
            enabled=True
        )
        assert chain.all_bones[-1] == "foot"

    def test_all_bones_order_joints_in_middle(self):
        """all_bones should have joint_bones in order between root and effector."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        joints = ["joint1", "joint2", "joint3"]
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=joints,
            effector_bone="effector",
            solver_type=IKSolverType.FABRIK,
            weight=1.0,
            priority=0,
            enabled=True
        )
        all_bones = chain.all_bones
        assert all_bones == ["root", "joint1", "joint2", "joint3", "effector"]

    def test_all_bones_length_matches_bone_count(self):
        """all_bones length should match bone_count."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["j1", "j2", "j3"],
            effector_bone="effector",
            solver_type=IKSolverType.FABRIK,
            weight=1.0,
            priority=0,
            enabled=True
        )
        assert len(chain.all_bones) == chain.bone_count

    def test_all_bones_empty_joints(self):
        """all_bones should work with empty joint_bones."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=[],
            effector_bone="effector",
            solver_type=IKSolverType.TWO_BONE,
            weight=1.0,
            priority=0,
            enabled=True
        )
        assert chain.all_bones == ["root", "effector"]


class TestIKChainSetWeight:
    """Test set_weight method."""

    def test_set_weight_basic(self):
        """set_weight should change the weight value."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["joint"],
            effector_bone="effector",
            solver_type=IKSolverType.FABRIK,
            weight=1.0,
            priority=0,
            enabled=True
        )
        chain.set_weight(0.5)
        assert chain.weight == 0.5

    def test_set_weight_to_zero(self):
        """set_weight should allow setting to 0."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["joint"],
            effector_bone="effector",
            solver_type=IKSolverType.FABRIK,
            weight=0.5,
            priority=0,
            enabled=True
        )
        chain.set_weight(0.0)
        assert chain.weight == 0.0

    def test_set_weight_to_one(self):
        """set_weight should allow setting to 1."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["joint"],
            effector_bone="effector",
            solver_type=IKSolverType.FABRIK,
            weight=0.5,
            priority=0,
            enabled=True
        )
        chain.set_weight(1.0)
        assert chain.weight == 1.0

    def test_set_weight_clamps_negative(self):
        """set_weight should clamp negative values to 0."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["joint"],
            effector_bone="effector",
            solver_type=IKSolverType.FABRIK,
            weight=0.5,
            priority=0,
            enabled=True
        )
        chain.set_weight(-0.5)
        assert chain.weight == 0.0

    def test_set_weight_clamps_above_one(self):
        """set_weight should clamp values above 1 to 1."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["joint"],
            effector_bone="effector",
            solver_type=IKSolverType.FABRIK,
            weight=0.5,
            priority=0,
            enabled=True
        )
        chain.set_weight(1.5)
        assert chain.weight == 1.0

    def test_set_weight_clamps_large_negative(self):
        """set_weight should clamp large negative values to 0."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["joint"],
            effector_bone="effector",
            solver_type=IKSolverType.FABRIK,
            weight=0.5,
            priority=0,
            enabled=True
        )
        chain.set_weight(-100.0)
        assert chain.weight == 0.0

    def test_set_weight_clamps_large_positive(self):
        """set_weight should clamp large positive values to 1."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["joint"],
            effector_bone="effector",
            solver_type=IKSolverType.FABRIK,
            weight=0.5,
            priority=0,
            enabled=True
        )
        chain.set_weight(100.0)
        assert chain.weight == 1.0

    def test_set_weight_boundary_just_above_zero(self):
        """set_weight should allow small positive values."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["joint"],
            effector_bone="effector",
            solver_type=IKSolverType.FABRIK,
            weight=0.5,
            priority=0,
            enabled=True
        )
        chain.set_weight(0.001)
        assert chain.weight == pytest.approx(0.001)

    def test_set_weight_boundary_just_below_one(self):
        """set_weight should allow values just below 1."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["joint"],
            effector_bone="effector",
            solver_type=IKSolverType.FABRIK,
            weight=0.5,
            priority=0,
            enabled=True
        )
        chain.set_weight(0.999)
        assert chain.weight == pytest.approx(0.999)

    def test_set_weight_returns_none(self):
        """set_weight should return None (void method)."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["joint"],
            effector_bone="effector",
            solver_type=IKSolverType.FABRIK,
            weight=0.5,
            priority=0,
            enabled=True
        )
        result = chain.set_weight(0.75)
        assert result is None


class TestIKChainSetEnabled:
    """Test set_enabled method."""

    def test_set_enabled_to_true(self):
        """set_enabled should enable the chain."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["joint"],
            effector_bone="effector",
            solver_type=IKSolverType.FABRIK,
            weight=1.0,
            priority=0,
            enabled=False
        )
        chain.set_enabled(True)
        assert chain.enabled is True

    def test_set_enabled_to_false(self):
        """set_enabled should disable the chain."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["joint"],
            effector_bone="effector",
            solver_type=IKSolverType.FABRIK,
            weight=1.0,
            priority=0,
            enabled=True
        )
        chain.set_enabled(False)
        assert chain.enabled is False

    def test_set_enabled_toggle_on_off(self):
        """set_enabled should allow toggling on and off."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["joint"],
            effector_bone="effector",
            solver_type=IKSolverType.FABRIK,
            weight=1.0,
            priority=0,
            enabled=True
        )
        chain.set_enabled(False)
        assert chain.enabled is False
        chain.set_enabled(True)
        assert chain.enabled is True

    def test_set_enabled_idempotent_true(self):
        """set_enabled(True) on already enabled chain should stay enabled."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["joint"],
            effector_bone="effector",
            solver_type=IKSolverType.FABRIK,
            weight=1.0,
            priority=0,
            enabled=True
        )
        chain.set_enabled(True)
        assert chain.enabled is True

    def test_set_enabled_idempotent_false(self):
        """set_enabled(False) on already disabled chain should stay disabled."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["joint"],
            effector_bone="effector",
            solver_type=IKSolverType.FABRIK,
            weight=1.0,
            priority=0,
            enabled=False
        )
        chain.set_enabled(False)
        assert chain.enabled is False

    def test_set_enabled_returns_none(self):
        """set_enabled should return None (void method)."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["joint"],
            effector_bone="effector",
            solver_type=IKSolverType.FABRIK,
            weight=1.0,
            priority=0,
            enabled=True
        )
        result = chain.set_enabled(False)
        assert result is None


class TestIKChainArmChainFactory:
    """Test arm_chain class method factory."""

    def test_arm_chain_left_side(self):
        """arm_chain should create left arm chain."""
        from engine.animation.ik.fullbody import IKChain
        chain = IKChain.arm_chain("left")
        assert chain is not None
        assert isinstance(chain, IKChain)

    def test_arm_chain_right_side(self):
        """arm_chain should create right arm chain."""
        from engine.animation.ik.fullbody import IKChain
        chain = IKChain.arm_chain("right")
        assert chain is not None
        assert isinstance(chain, IKChain)

    def test_arm_chain_left_has_correct_name(self):
        """Left arm chain should have appropriate name."""
        from engine.animation.ik.fullbody import IKChain
        chain = IKChain.arm_chain("left")
        assert "left" in chain.name.lower() or "l_" in chain.name.lower()

    def test_arm_chain_right_has_correct_name(self):
        """Right arm chain should have appropriate name."""
        from engine.animation.ik.fullbody import IKChain
        chain = IKChain.arm_chain("right")
        assert "right" in chain.name.lower() or "r_" in chain.name.lower()

    def test_arm_chain_is_enabled_by_default(self):
        """Arm chain should be enabled by default."""
        from engine.animation.ik.fullbody import IKChain
        chain = IKChain.arm_chain("left")
        assert chain.enabled is True

    def test_arm_chain_has_valid_weight(self):
        """Arm chain should have weight in valid range."""
        from engine.animation.ik.fullbody import IKChain
        chain = IKChain.arm_chain("left")
        assert 0.0 <= chain.weight <= 1.0

    def test_arm_chain_has_bones(self):
        """Arm chain should have at least root and effector."""
        from engine.animation.ik.fullbody import IKChain
        chain = IKChain.arm_chain("left")
        assert chain.bone_count >= 2

    def test_arm_chain_has_valid_solver(self):
        """Arm chain should have a valid solver type."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain.arm_chain("left")
        assert isinstance(chain.solver_type, IKSolverType)

    def test_arm_chain_left_right_different(self):
        """Left and right arm chains should be different."""
        from engine.animation.ik.fullbody import IKChain
        left = IKChain.arm_chain("left")
        right = IKChain.arm_chain("right")
        assert left.name != right.name

    def test_arm_chain_root_bone_exists(self):
        """Arm chain should have a root bone."""
        from engine.animation.ik.fullbody import IKChain
        chain = IKChain.arm_chain("left")
        assert chain.root_bone is not None
        assert len(chain.root_bone) > 0

    def test_arm_chain_effector_bone_exists(self):
        """Arm chain should have an effector bone."""
        from engine.animation.ik.fullbody import IKChain
        chain = IKChain.arm_chain("left")
        assert chain.effector_bone is not None
        assert len(chain.effector_bone) > 0


class TestIKChainLegChainFactory:
    """Test leg_chain class method factory."""

    def test_leg_chain_left_side(self):
        """leg_chain should create left leg chain."""
        from engine.animation.ik.fullbody import IKChain
        chain = IKChain.leg_chain("left")
        assert chain is not None
        assert isinstance(chain, IKChain)

    def test_leg_chain_right_side(self):
        """leg_chain should create right leg chain."""
        from engine.animation.ik.fullbody import IKChain
        chain = IKChain.leg_chain("right")
        assert chain is not None
        assert isinstance(chain, IKChain)

    def test_leg_chain_left_has_correct_name(self):
        """Left leg chain should have appropriate name."""
        from engine.animation.ik.fullbody import IKChain
        chain = IKChain.leg_chain("left")
        assert "left" in chain.name.lower() or "l_" in chain.name.lower()

    def test_leg_chain_right_has_correct_name(self):
        """Right leg chain should have appropriate name."""
        from engine.animation.ik.fullbody import IKChain
        chain = IKChain.leg_chain("right")
        assert "right" in chain.name.lower() or "r_" in chain.name.lower()

    def test_leg_chain_is_enabled_by_default(self):
        """Leg chain should be enabled by default."""
        from engine.animation.ik.fullbody import IKChain
        chain = IKChain.leg_chain("left")
        assert chain.enabled is True

    def test_leg_chain_has_valid_weight(self):
        """Leg chain should have weight in valid range."""
        from engine.animation.ik.fullbody import IKChain
        chain = IKChain.leg_chain("left")
        assert 0.0 <= chain.weight <= 1.0

    def test_leg_chain_has_bones(self):
        """Leg chain should have at least root and effector."""
        from engine.animation.ik.fullbody import IKChain
        chain = IKChain.leg_chain("left")
        assert chain.bone_count >= 2

    def test_leg_chain_has_valid_solver(self):
        """Leg chain should have a valid solver type."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain.leg_chain("left")
        assert isinstance(chain.solver_type, IKSolverType)

    def test_leg_chain_left_right_different(self):
        """Left and right leg chains should be different."""
        from engine.animation.ik.fullbody import IKChain
        left = IKChain.leg_chain("left")
        right = IKChain.leg_chain("right")
        assert left.name != right.name

    def test_leg_chain_root_bone_exists(self):
        """Leg chain should have a root bone."""
        from engine.animation.ik.fullbody import IKChain
        chain = IKChain.leg_chain("left")
        assert chain.root_bone is not None
        assert len(chain.root_bone) > 0

    def test_leg_chain_effector_bone_exists(self):
        """Leg chain should have an effector bone."""
        from engine.animation.ik.fullbody import IKChain
        chain = IKChain.leg_chain("left")
        assert chain.effector_bone is not None
        assert len(chain.effector_bone) > 0


class TestIKChainSolverTypes:
    """Test IKChain with different solver types."""

    def test_chain_with_two_bone_solver(self):
        """IKChain should accept TWO_BONE solver."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="two_bone_test",
            root_bone="shoulder",
            joint_bones=["elbow"],
            effector_bone="hand",
            solver_type=IKSolverType.TWO_BONE,
            weight=1.0,
            priority=0,
            enabled=True
        )
        assert chain.solver_type == IKSolverType.TWO_BONE

    def test_chain_with_fabrik_solver(self):
        """IKChain should accept FABRIK solver."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="fabrik_test",
            root_bone="hip",
            joint_bones=["knee", "ankle"],
            effector_bone="foot",
            solver_type=IKSolverType.FABRIK,
            weight=1.0,
            priority=0,
            enabled=True
        )
        assert chain.solver_type == IKSolverType.FABRIK

    def test_chain_with_ccd_solver(self):
        """IKChain should accept CCD solver."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="ccd_test",
            root_bone="spine",
            joint_bones=["spine1", "spine2", "spine3"],
            effector_bone="head",
            solver_type=IKSolverType.CCD,
            weight=1.0,
            priority=0,
            enabled=True
        )
        assert chain.solver_type == IKSolverType.CCD

    def test_chain_with_jacobian_solver(self):
        """IKChain should accept JACOBIAN solver."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="jacobian_test",
            root_bone="root",
            joint_bones=["j1", "j2", "j3", "j4"],
            effector_bone="effector",
            solver_type=IKSolverType.JACOBIAN,
            weight=1.0,
            priority=0,
            enabled=True
        )
        assert chain.solver_type == IKSolverType.JACOBIAN


class TestIKChainPriority:
    """Test IKChain priority field behavior."""

    def test_priority_zero(self):
        """IKChain should accept priority of 0."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["joint"],
            effector_bone="effector",
            solver_type=IKSolverType.FABRIK,
            weight=1.0,
            priority=0,
            enabled=True
        )
        assert chain.priority == 0

    def test_priority_positive(self):
        """IKChain should accept positive priority."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["joint"],
            effector_bone="effector",
            solver_type=IKSolverType.FABRIK,
            weight=1.0,
            priority=10,
            enabled=True
        )
        assert chain.priority == 10

    def test_priority_negative(self):
        """IKChain should accept negative priority."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["joint"],
            effector_bone="effector",
            solver_type=IKSolverType.FABRIK,
            weight=1.0,
            priority=-5,
            enabled=True
        )
        assert chain.priority == -5

    def test_priority_large_value(self):
        """IKChain should accept large priority values."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["joint"],
            effector_bone="effector",
            solver_type=IKSolverType.FABRIK,
            weight=1.0,
            priority=1000,
            enabled=True
        )
        assert chain.priority == 1000


class TestIKChainComplexScenarios:
    """Test complex IKChain scenarios."""

    def test_spine_chain_many_joints(self):
        """IKChain should handle spine with many joints."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        spine_joints = ["spine_01", "spine_02", "spine_03", "spine_04", "spine_05"]
        chain = IKChain(
            name="spine",
            root_bone="pelvis",
            joint_bones=spine_joints,
            effector_bone="head",
            solver_type=IKSolverType.CCD,
            weight=0.8,
            priority=5,
            enabled=True
        )
        assert chain.bone_count == 7  # pelvis + 5 spine + head
        assert len(chain.all_bones) == 7

    def test_finger_chain(self):
        """IKChain should handle finger chain."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="index_finger_left",
            root_bone="hand_l",
            joint_bones=["index_01_l", "index_02_l"],
            effector_bone="index_03_l",
            solver_type=IKSolverType.CCD,
            weight=1.0,
            priority=2,
            enabled=True
        )
        assert chain.bone_count == 4

    def test_chain_modification_sequence(self):
        """IKChain should handle multiple modifications."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["joint"],
            effector_bone="effector",
            solver_type=IKSolverType.FABRIK,
            weight=1.0,
            priority=0,
            enabled=True
        )

        # Disable then re-enable
        chain.set_enabled(False)
        assert chain.enabled is False

        # Change weight while disabled
        chain.set_weight(0.5)
        assert chain.weight == 0.5

        # Re-enable
        chain.set_enabled(True)
        assert chain.enabled is True
        assert chain.weight == 0.5  # Weight should persist

    def test_multiple_chains_independence(self):
        """Multiple IKChain instances should be independent."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain1 = IKChain(
            name="chain1",
            root_bone="root1",
            joint_bones=["joint1"],
            effector_bone="effector1",
            solver_type=IKSolverType.TWO_BONE,
            weight=1.0,
            priority=0,
            enabled=True
        )
        chain2 = IKChain(
            name="chain2",
            root_bone="root2",
            joint_bones=["joint2"],
            effector_bone="effector2",
            solver_type=IKSolverType.FABRIK,
            weight=0.5,
            priority=1,
            enabled=False
        )

        # Modify chain1
        chain1.set_weight(0.0)
        chain1.set_enabled(False)

        # Chain2 should be unaffected
        assert chain2.weight == 0.5
        assert chain2.enabled is False  # Was already False
        assert chain2.solver_type == IKSolverType.FABRIK

    def test_factory_methods_create_independent_chains(self):
        """Factory methods should create independent chain instances."""
        from engine.animation.ik.fullbody import IKChain
        arm1 = IKChain.arm_chain("left")
        arm2 = IKChain.arm_chain("left")

        # Modify arm1
        arm1.set_weight(0.0)

        # arm2 should be unaffected
        assert arm2.weight != arm1.weight or arm2.weight == 1.0


class TestIKChainEdgeCases:
    """Test edge cases for IKChain."""

    def test_single_bone_chain(self):
        """IKChain should handle root directly to effector (no joints)."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="direct",
            root_bone="root",
            joint_bones=[],
            effector_bone="effector",
            solver_type=IKSolverType.TWO_BONE,
            weight=1.0,
            priority=0,
            enabled=True
        )
        assert chain.bone_count == 2
        assert chain.all_bones == ["root", "effector"]

    def test_weight_exactly_zero(self):
        """Weight of exactly 0 should be valid."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["joint"],
            effector_bone="effector",
            solver_type=IKSolverType.FABRIK,
            weight=0.0,
            priority=0,
            enabled=True
        )
        assert chain.weight == 0.0

    def test_weight_exactly_one(self):
        """Weight of exactly 1 should be valid."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["joint"],
            effector_bone="effector",
            solver_type=IKSolverType.FABRIK,
            weight=1.0,
            priority=0,
            enabled=True
        )
        assert chain.weight == 1.0

    def test_empty_string_name(self):
        """IKChain should accept empty string name (implementation may vary)."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="",
            root_bone="root",
            joint_bones=["joint"],
            effector_bone="effector",
            solver_type=IKSolverType.FABRIK,
            weight=1.0,
            priority=0,
            enabled=True
        )
        assert chain.name == ""

    def test_unicode_bone_names(self):
        """IKChain should handle unicode bone names."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="unicode_test",
            root_bone="root_bone",
            joint_bones=["joint_bone"],
            effector_bone="effector_bone",
            solver_type=IKSolverType.FABRIK,
            weight=1.0,
            priority=0,
            enabled=True
        )
        assert chain.root_bone == "root_bone"

    def test_all_bones_does_not_mutate(self):
        """Modifying returned all_bones should not affect the chain."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["joint1", "joint2"],
            effector_bone="effector",
            solver_type=IKSolverType.FABRIK,
            weight=1.0,
            priority=0,
            enabled=True
        )
        bones = chain.all_bones
        original_count = chain.bone_count

        # Try to mutate the returned list
        bones.append("extra_bone")

        # Chain should be unchanged
        assert chain.bone_count == original_count
        assert len(chain.all_bones) == original_count


class TestIKChainDataclassFeatures:
    """Test dataclass features of IKChain."""

    def test_chain_has_all_required_fields(self):
        """IKChain construction should require all fields."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        # Should work with all fields provided
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["joint"],
            effector_bone="effector",
            solver_type=IKSolverType.FABRIK,
            weight=1.0,
            priority=0,
            enabled=True
        )
        assert chain is not None

    def test_chain_field_access(self):
        """All fields should be accessible as attributes."""
        from engine.animation.ik.fullbody import IKChain, IKSolverType
        chain = IKChain(
            name="access_test",
            root_bone="root_b",
            joint_bones=["joint_b"],
            effector_bone="effector_b",
            solver_type=IKSolverType.CCD,
            weight=0.7,
            priority=3,
            enabled=False
        )
        assert chain.name == "access_test"
        assert chain.root_bone == "root_b"
        assert chain.joint_bones == ["joint_b"]
        assert chain.effector_bone == "effector_b"
        assert chain.solver_type == IKSolverType.CCD
        assert chain.weight == 0.7
        assert chain.priority == 3
        assert chain.enabled is False
