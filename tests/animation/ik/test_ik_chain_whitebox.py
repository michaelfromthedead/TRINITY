"""Whitebox tests for IKChain and IKSolverType classes.

Tests internal implementation details of the IKSolverType enum and
IKChain dataclass, including enum values, dataclass fields, property
implementations, method internals, factory methods, and __post_init__
weight clamping behavior.
"""

from __future__ import annotations

import pytest
from enum import auto
from dataclasses import fields, field

from engine.animation.ik.fullbody import IKSolverType, IKChain


# =============================================================================
# IKSolverType Enum Tests
# =============================================================================


class TestIKSolverTypeEnumValues:
    """Tests for IKSolverType enum value definitions."""

    def test_two_bone_exists(self) -> None:
        """TWO_BONE value exists in enum."""
        assert hasattr(IKSolverType, 'TWO_BONE')
        assert IKSolverType.TWO_BONE.name == 'TWO_BONE'

    def test_fabrik_exists(self) -> None:
        """FABRIK value exists in enum."""
        assert hasattr(IKSolverType, 'FABRIK')
        assert IKSolverType.FABRIK.name == 'FABRIK'

    def test_ccd_exists(self) -> None:
        """CCD value exists in enum."""
        assert hasattr(IKSolverType, 'CCD')
        assert IKSolverType.CCD.name == 'CCD'

    def test_jacobian_exists(self) -> None:
        """JACOBIAN value exists in enum."""
        assert hasattr(IKSolverType, 'JACOBIAN')
        assert IKSolverType.JACOBIAN.name == 'JACOBIAN'

    def test_enum_has_exactly_four_members(self) -> None:
        """IKSolverType has exactly 4 members."""
        members = list(IKSolverType)
        assert len(members) == 4

    def test_enum_member_order(self) -> None:
        """Enum members are in defined order (TWO_BONE, FABRIK, CCD, JACOBIAN)."""
        members = list(IKSolverType)
        assert members[0] == IKSolverType.TWO_BONE
        assert members[1] == IKSolverType.FABRIK
        assert members[2] == IKSolverType.CCD
        assert members[3] == IKSolverType.JACOBIAN


class TestIKSolverTypeEnumBehavior:
    """Tests for IKSolverType enum behavior and properties."""

    def test_values_are_unique(self) -> None:
        """Each enum member has a unique value."""
        values = [member.value for member in IKSolverType]
        assert len(values) == len(set(values))

    def test_enum_members_are_comparable(self) -> None:
        """Enum members can be compared for equality."""
        assert IKSolverType.TWO_BONE == IKSolverType.TWO_BONE
        assert IKSolverType.TWO_BONE != IKSolverType.FABRIK
        assert IKSolverType.FABRIK != IKSolverType.CCD
        assert IKSolverType.CCD != IKSolverType.JACOBIAN

    def test_enum_members_are_hashable(self) -> None:
        """Enum members can be used as dictionary keys."""
        solver_map = {
            IKSolverType.TWO_BONE: "simple",
            IKSolverType.FABRIK: "iterative",
            IKSolverType.CCD: "iterative",
            IKSolverType.JACOBIAN: "analytical",
        }
        assert solver_map[IKSolverType.TWO_BONE] == "simple"
        assert solver_map[IKSolverType.JACOBIAN] == "analytical"

    def test_enum_identity(self) -> None:
        """Enum members maintain identity when accessed multiple times."""
        assert IKSolverType.TWO_BONE is IKSolverType.TWO_BONE
        assert IKSolverType['FABRIK'] is IKSolverType.FABRIK

    def test_enum_can_be_accessed_by_name(self) -> None:
        """Enum members can be accessed by string name."""
        assert IKSolverType['TWO_BONE'] == IKSolverType.TWO_BONE
        assert IKSolverType['FABRIK'] == IKSolverType.FABRIK
        assert IKSolverType['CCD'] == IKSolverType.CCD
        assert IKSolverType['JACOBIAN'] == IKSolverType.JACOBIAN

    def test_invalid_name_raises_key_error(self) -> None:
        """Accessing invalid enum name raises KeyError."""
        with pytest.raises(KeyError):
            _ = IKSolverType['INVALID']


# =============================================================================
# IKChain Dataclass Field Tests
# =============================================================================


class TestIKChainFields:
    """Tests for IKChain dataclass field definitions."""

    def test_has_name_field(self) -> None:
        """IKChain has name field of type str."""
        field_names = [f.name for f in fields(IKChain)]
        assert 'name' in field_names

    def test_has_root_bone_field(self) -> None:
        """IKChain has root_bone field of type str."""
        field_names = [f.name for f in fields(IKChain)]
        assert 'root_bone' in field_names

    def test_has_joint_bones_field(self) -> None:
        """IKChain has joint_bones field of type List[str]."""
        field_names = [f.name for f in fields(IKChain)]
        assert 'joint_bones' in field_names

    def test_has_effector_bone_field(self) -> None:
        """IKChain has effector_bone field of type str."""
        field_names = [f.name for f in fields(IKChain)]
        assert 'effector_bone' in field_names

    def test_has_solver_type_field(self) -> None:
        """IKChain has solver_type field of type IKSolverType."""
        field_names = [f.name for f in fields(IKChain)]
        assert 'solver_type' in field_names

    def test_has_weight_field(self) -> None:
        """IKChain has weight field of type float."""
        field_names = [f.name for f in fields(IKChain)]
        assert 'weight' in field_names

    def test_has_priority_field(self) -> None:
        """IKChain has priority field of type int."""
        field_names = [f.name for f in fields(IKChain)]
        assert 'priority' in field_names

    def test_has_enabled_field(self) -> None:
        """IKChain has enabled field of type bool."""
        field_names = [f.name for f in fields(IKChain)]
        assert 'enabled' in field_names

    def test_total_field_count(self) -> None:
        """IKChain has exactly 8 fields."""
        assert len(fields(IKChain)) == 8


class TestIKChainDefaults:
    """Tests for IKChain default field values."""

    def test_joint_bones_default_empty_list(self) -> None:
        """joint_bones defaults to empty list."""
        chain = IKChain(name="test", root_bone="root")
        assert chain.joint_bones == []
        assert isinstance(chain.joint_bones, list)

    def test_effector_bone_default_empty_string(self) -> None:
        """effector_bone defaults to empty string."""
        chain = IKChain(name="test", root_bone="root")
        assert chain.effector_bone == ""

    def test_solver_type_default_two_bone(self) -> None:
        """solver_type defaults to TWO_BONE."""
        chain = IKChain(name="test", root_bone="root")
        assert chain.solver_type == IKSolverType.TWO_BONE

    def test_weight_default_one(self) -> None:
        """weight defaults to 1.0."""
        chain = IKChain(name="test", root_bone="root")
        assert chain.weight == 1.0

    def test_priority_default_zero(self) -> None:
        """priority defaults to 0."""
        chain = IKChain(name="test", root_bone="root")
        assert chain.priority == 0

    def test_enabled_default_true(self) -> None:
        """enabled defaults to True."""
        chain = IKChain(name="test", root_bone="root")
        assert chain.enabled is True

    def test_default_factory_creates_unique_lists(self) -> None:
        """Each instance gets its own joint_bones list via default_factory."""
        chain1 = IKChain(name="test1", root_bone="root1")
        chain2 = IKChain(name="test2", root_bone="root2")
        chain1.joint_bones.append("bone")
        assert chain2.joint_bones == []
        assert chain1.joint_bones is not chain2.joint_bones


# =============================================================================
# IKChain Initialization Tests
# =============================================================================


class TestIKChainInit:
    """Tests for IKChain initialization behavior."""

    def test_minimal_init_with_required_fields(self) -> None:
        """IKChain can be created with only required fields."""
        chain = IKChain(name="test_chain", root_bone="shoulder")
        assert chain.name == "test_chain"
        assert chain.root_bone == "shoulder"

    def test_full_init_with_all_fields(self) -> None:
        """IKChain can be created with all fields specified."""
        chain = IKChain(
            name="arm",
            root_bone="shoulder",
            joint_bones=["upper_arm", "lower_arm"],
            effector_bone="hand",
            solver_type=IKSolverType.FABRIK,
            weight=0.75,
            priority=5,
            enabled=False,
        )
        assert chain.name == "arm"
        assert chain.root_bone == "shoulder"
        assert chain.joint_bones == ["upper_arm", "lower_arm"]
        assert chain.effector_bone == "hand"
        assert chain.solver_type == IKSolverType.FABRIK
        assert chain.weight == 0.75
        assert chain.priority == 5
        assert chain.enabled is False

    def test_init_preserves_joint_bones_order(self) -> None:
        """joint_bones list maintains order of input."""
        bones = ["a", "b", "c", "d"]
        chain = IKChain(name="test", root_bone="root", joint_bones=bones)
        assert chain.joint_bones == ["a", "b", "c", "d"]
        assert chain.joint_bones[0] == "a"
        assert chain.joint_bones[3] == "d"


# =============================================================================
# IKChain __post_init__ Weight Clamping Tests
# =============================================================================


class TestIKChainPostInitWeightClamping:
    """Tests for __post_init__ weight clamping behavior."""

    def test_weight_at_zero_unchanged(self) -> None:
        """Weight of exactly 0.0 remains 0.0."""
        chain = IKChain(name="test", root_bone="root", weight=0.0)
        assert chain.weight == 0.0

    def test_weight_at_one_unchanged(self) -> None:
        """Weight of exactly 1.0 remains 1.0."""
        chain = IKChain(name="test", root_bone="root", weight=1.0)
        assert chain.weight == 1.0

    def test_weight_negative_clamped_to_zero(self) -> None:
        """Negative weight is clamped to 0.0."""
        chain = IKChain(name="test", root_bone="root", weight=-0.5)
        assert chain.weight == 0.0

    def test_weight_large_negative_clamped_to_zero(self) -> None:
        """Large negative weight is clamped to 0.0."""
        chain = IKChain(name="test", root_bone="root", weight=-100.0)
        assert chain.weight == 0.0

    def test_weight_above_one_clamped_to_one(self) -> None:
        """Weight above 1.0 is clamped to 1.0."""
        chain = IKChain(name="test", root_bone="root", weight=1.5)
        assert chain.weight == 1.0

    def test_weight_large_positive_clamped_to_one(self) -> None:
        """Large positive weight is clamped to 1.0."""
        chain = IKChain(name="test", root_bone="root", weight=999.0)
        assert chain.weight == 1.0

    def test_weight_in_valid_range_unchanged(self) -> None:
        """Weight within [0, 1] range is not modified."""
        for weight in [0.1, 0.25, 0.5, 0.75, 0.99]:
            chain = IKChain(name="test", root_bone="root", weight=weight)
            assert chain.weight == weight

    def test_weight_clamping_uses_min_max_chain(self) -> None:
        """Verify clamping formula: max(0.0, min(1.0, weight))."""
        # Verify by checking boundary behavior
        chain_neg = IKChain(name="test", root_bone="root", weight=-0.001)
        assert chain_neg.weight == 0.0

        chain_pos = IKChain(name="test", root_bone="root", weight=1.001)
        assert chain_pos.weight == 1.0


# =============================================================================
# IKChain bone_count Property Tests
# =============================================================================


class TestIKChainBoneCountProperty:
    """Tests for bone_count property implementation."""

    def test_bone_count_no_joints(self) -> None:
        """bone_count with no joint bones returns 2 (root + effector)."""
        chain = IKChain(name="test", root_bone="root", effector_bone="end")
        assert chain.bone_count == 2

    def test_bone_count_one_joint(self) -> None:
        """bone_count with 1 joint bone returns 3."""
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["mid"],
            effector_bone="end",
        )
        assert chain.bone_count == 3

    def test_bone_count_two_joints(self) -> None:
        """bone_count with 2 joint bones returns 4."""
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["mid1", "mid2"],
            effector_bone="end",
        )
        assert chain.bone_count == 4

    def test_bone_count_many_joints(self) -> None:
        """bone_count with many joint bones returns correct count."""
        joints = [f"joint_{i}" for i in range(10)]
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=joints,
            effector_bone="end",
        )
        assert chain.bone_count == 12  # 2 + 10

    def test_bone_count_formula_is_two_plus_joint_count(self) -> None:
        """Verify bone_count = 2 + len(joint_bones)."""
        for joint_count in range(20):
            joints = [f"j{i}" for i in range(joint_count)]
            chain = IKChain(name="test", root_bone="root", joint_bones=joints)
            assert chain.bone_count == 2 + joint_count

    def test_bone_count_is_readonly_property(self) -> None:
        """bone_count cannot be set directly."""
        chain = IKChain(name="test", root_bone="root")
        with pytest.raises(AttributeError):
            chain.bone_count = 5


# =============================================================================
# IKChain all_bones Property Tests
# =============================================================================


class TestIKChainAllBonesProperty:
    """Tests for all_bones property implementation."""

    def test_all_bones_no_joints(self) -> None:
        """all_bones with no joints returns [root, effector]."""
        chain = IKChain(name="test", root_bone="shoulder", effector_bone="hand")
        assert chain.all_bones == ["shoulder", "hand"]

    def test_all_bones_with_joints(self) -> None:
        """all_bones includes root, joints, and effector in order."""
        chain = IKChain(
            name="test",
            root_bone="shoulder",
            joint_bones=["upper_arm", "lower_arm"],
            effector_bone="hand",
        )
        assert chain.all_bones == ["shoulder", "upper_arm", "lower_arm", "hand"]

    def test_all_bones_order_root_first(self) -> None:
        """all_bones starts with root_bone."""
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["a", "b"],
            effector_bone="end",
        )
        assert chain.all_bones[0] == "root"

    def test_all_bones_order_effector_last(self) -> None:
        """all_bones ends with effector_bone."""
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["a", "b"],
            effector_bone="end",
        )
        assert chain.all_bones[-1] == "end"

    def test_all_bones_order_joints_in_middle(self) -> None:
        """all_bones has joints between root and effector."""
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["j1", "j2", "j3"],
            effector_bone="end",
        )
        bones = chain.all_bones
        assert bones[1:4] == ["j1", "j2", "j3"]

    def test_all_bones_returns_new_list(self) -> None:
        """all_bones returns a new list each time (concatenation)."""
        chain = IKChain(
            name="test",
            root_bone="root",
            joint_bones=["mid"],
            effector_bone="end",
        )
        bones1 = chain.all_bones
        bones2 = chain.all_bones
        assert bones1 == bones2
        assert bones1 is not bones2

    def test_all_bones_length_matches_bone_count(self) -> None:
        """len(all_bones) equals bone_count."""
        for joint_count in [0, 1, 2, 5, 10]:
            joints = [f"j{i}" for i in range(joint_count)]
            chain = IKChain(name="test", root_bone="root", joint_bones=joints, effector_bone="end")
            assert len(chain.all_bones) == chain.bone_count

    def test_all_bones_is_readonly_property(self) -> None:
        """all_bones cannot be set directly."""
        chain = IKChain(name="test", root_bone="root")
        with pytest.raises(AttributeError):
            chain.all_bones = ["a", "b", "c"]


# =============================================================================
# IKChain set_weight Method Tests
# =============================================================================


class TestIKChainSetWeight:
    """Tests for set_weight method implementation."""

    def test_set_weight_valid_value(self) -> None:
        """set_weight with valid value updates weight."""
        chain = IKChain(name="test", root_bone="root", weight=1.0)
        chain.set_weight(0.5)
        assert chain.weight == 0.5

    def test_set_weight_zero(self) -> None:
        """set_weight to 0.0 sets weight to zero."""
        chain = IKChain(name="test", root_bone="root", weight=1.0)
        chain.set_weight(0.0)
        assert chain.weight == 0.0

    def test_set_weight_one(self) -> None:
        """set_weight to 1.0 sets weight to one."""
        chain = IKChain(name="test", root_bone="root", weight=0.0)
        chain.set_weight(1.0)
        assert chain.weight == 1.0

    def test_set_weight_negative_clamped_to_zero(self) -> None:
        """set_weight with negative value clamps to 0.0."""
        chain = IKChain(name="test", root_bone="root", weight=0.5)
        chain.set_weight(-0.5)
        assert chain.weight == 0.0

    def test_set_weight_large_negative_clamped_to_zero(self) -> None:
        """set_weight with large negative value clamps to 0.0."""
        chain = IKChain(name="test", root_bone="root", weight=0.5)
        chain.set_weight(-1000.0)
        assert chain.weight == 0.0

    def test_set_weight_above_one_clamped_to_one(self) -> None:
        """set_weight with value above 1 clamps to 1.0."""
        chain = IKChain(name="test", root_bone="root", weight=0.5)
        chain.set_weight(1.5)
        assert chain.weight == 1.0

    def test_set_weight_large_positive_clamped_to_one(self) -> None:
        """set_weight with large positive value clamps to 1.0."""
        chain = IKChain(name="test", root_bone="root", weight=0.5)
        chain.set_weight(999.0)
        assert chain.weight == 1.0

    def test_set_weight_returns_none(self) -> None:
        """set_weight returns None."""
        chain = IKChain(name="test", root_bone="root")
        result = chain.set_weight(0.5)
        assert result is None

    def test_set_weight_multiple_calls(self) -> None:
        """set_weight can be called multiple times."""
        chain = IKChain(name="test", root_bone="root", weight=0.0)
        chain.set_weight(0.3)
        assert chain.weight == 0.3
        chain.set_weight(0.7)
        assert chain.weight == 0.7
        chain.set_weight(0.0)
        assert chain.weight == 0.0


# =============================================================================
# IKChain set_enabled Method Tests
# =============================================================================


class TestIKChainSetEnabled:
    """Tests for set_enabled method implementation."""

    def test_set_enabled_true(self) -> None:
        """set_enabled(True) enables the chain."""
        chain = IKChain(name="test", root_bone="root", enabled=False)
        chain.set_enabled(True)
        assert chain.enabled is True

    def test_set_enabled_false(self) -> None:
        """set_enabled(False) disables the chain."""
        chain = IKChain(name="test", root_bone="root", enabled=True)
        chain.set_enabled(False)
        assert chain.enabled is False

    def test_set_enabled_idempotent_true(self) -> None:
        """set_enabled(True) on already enabled chain stays enabled."""
        chain = IKChain(name="test", root_bone="root", enabled=True)
        chain.set_enabled(True)
        assert chain.enabled is True

    def test_set_enabled_idempotent_false(self) -> None:
        """set_enabled(False) on already disabled chain stays disabled."""
        chain = IKChain(name="test", root_bone="root", enabled=False)
        chain.set_enabled(False)
        assert chain.enabled is False

    def test_set_enabled_returns_none(self) -> None:
        """set_enabled returns None."""
        chain = IKChain(name="test", root_bone="root")
        result = chain.set_enabled(False)
        assert result is None

    def test_set_enabled_toggle(self) -> None:
        """set_enabled can toggle enabled state multiple times."""
        chain = IKChain(name="test", root_bone="root", enabled=True)
        chain.set_enabled(False)
        assert chain.enabled is False
        chain.set_enabled(True)
        assert chain.enabled is True
        chain.set_enabled(False)
        assert chain.enabled is False


# =============================================================================
# IKChain arm_chain Factory Method Tests
# =============================================================================


class TestIKChainArmChainFactory:
    """Tests for arm_chain classmethod implementation."""

    def test_arm_chain_returns_ik_chain(self) -> None:
        """arm_chain returns an IKChain instance."""
        chain = IKChain.arm_chain()
        assert isinstance(chain, IKChain)

    def test_arm_chain_left_default(self) -> None:
        """arm_chain defaults to left side."""
        chain = IKChain.arm_chain()
        assert chain.name == "left_arm"
        assert chain.root_bone == "l_shoulder"
        assert chain.effector_bone == "l_hand"

    def test_arm_chain_left_explicit(self) -> None:
        """arm_chain with side='left' creates left arm."""
        chain = IKChain.arm_chain(side="left")
        assert chain.name == "left_arm"
        assert chain.root_bone == "l_shoulder"
        assert chain.joint_bones == ["l_upper_arm", "l_lower_arm"]
        assert chain.effector_bone == "l_hand"

    def test_arm_chain_right(self) -> None:
        """arm_chain with side='right' creates right arm."""
        chain = IKChain.arm_chain(side="right")
        assert chain.name == "right_arm"
        assert chain.root_bone == "r_shoulder"
        assert chain.joint_bones == ["r_upper_arm", "r_lower_arm"]
        assert chain.effector_bone == "r_hand"

    def test_arm_chain_solver_type_two_bone(self) -> None:
        """arm_chain uses TWO_BONE solver."""
        chain = IKChain.arm_chain()
        assert chain.solver_type == IKSolverType.TWO_BONE

    def test_arm_chain_priority_10(self) -> None:
        """arm_chain has priority of 10."""
        chain = IKChain.arm_chain()
        assert chain.priority == 10

    def test_arm_chain_weight_default_1(self) -> None:
        """arm_chain has default weight of 1.0."""
        chain = IKChain.arm_chain()
        assert chain.weight == 1.0

    def test_arm_chain_enabled_default_true(self) -> None:
        """arm_chain is enabled by default."""
        chain = IKChain.arm_chain()
        assert chain.enabled is True

    def test_arm_chain_bone_count_is_4(self) -> None:
        """arm_chain has 4 bones (shoulder, upper_arm, lower_arm, hand)."""
        chain = IKChain.arm_chain()
        assert chain.bone_count == 4

    def test_arm_chain_prefix_determination(self) -> None:
        """arm_chain uses 'l_' prefix for left and 'r_' prefix for right."""
        left = IKChain.arm_chain(side="left")
        right = IKChain.arm_chain(side="right")

        # All left bones start with l_
        for bone in left.all_bones:
            assert bone.startswith("l_")

        # All right bones start with r_
        for bone in right.all_bones:
            assert bone.startswith("r_")


# =============================================================================
# IKChain leg_chain Factory Method Tests
# =============================================================================


class TestIKChainLegChainFactory:
    """Tests for leg_chain classmethod implementation."""

    def test_leg_chain_returns_ik_chain(self) -> None:
        """leg_chain returns an IKChain instance."""
        chain = IKChain.leg_chain()
        assert isinstance(chain, IKChain)

    def test_leg_chain_left_default(self) -> None:
        """leg_chain defaults to left side."""
        chain = IKChain.leg_chain()
        assert chain.name == "left_leg"
        assert chain.root_bone == "l_thigh"
        assert chain.effector_bone == "l_foot"

    def test_leg_chain_left_explicit(self) -> None:
        """leg_chain with side='left' creates left leg."""
        chain = IKChain.leg_chain(side="left")
        assert chain.name == "left_leg"
        assert chain.root_bone == "l_thigh"
        assert chain.joint_bones == ["l_shin"]
        assert chain.effector_bone == "l_foot"

    def test_leg_chain_right(self) -> None:
        """leg_chain with side='right' creates right leg."""
        chain = IKChain.leg_chain(side="right")
        assert chain.name == "right_leg"
        assert chain.root_bone == "r_thigh"
        assert chain.joint_bones == ["r_shin"]
        assert chain.effector_bone == "r_foot"

    def test_leg_chain_solver_type_two_bone(self) -> None:
        """leg_chain uses TWO_BONE solver."""
        chain = IKChain.leg_chain()
        assert chain.solver_type == IKSolverType.TWO_BONE

    def test_leg_chain_priority_20(self) -> None:
        """leg_chain has priority of 20 (higher than arms)."""
        chain = IKChain.leg_chain()
        assert chain.priority == 20

    def test_leg_chain_priority_higher_than_arm(self) -> None:
        """leg_chain priority is higher than arm_chain priority."""
        leg = IKChain.leg_chain()
        arm = IKChain.arm_chain()
        assert leg.priority > arm.priority

    def test_leg_chain_weight_default_1(self) -> None:
        """leg_chain has default weight of 1.0."""
        chain = IKChain.leg_chain()
        assert chain.weight == 1.0

    def test_leg_chain_enabled_default_true(self) -> None:
        """leg_chain is enabled by default."""
        chain = IKChain.leg_chain()
        assert chain.enabled is True

    def test_leg_chain_bone_count_is_3(self) -> None:
        """leg_chain has 3 bones (thigh, shin, foot)."""
        chain = IKChain.leg_chain()
        assert chain.bone_count == 3

    def test_leg_chain_has_one_joint_bone(self) -> None:
        """leg_chain has exactly one joint bone (shin)."""
        chain = IKChain.leg_chain()
        assert len(chain.joint_bones) == 1

    def test_leg_chain_prefix_determination(self) -> None:
        """leg_chain uses 'l_' prefix for left and 'r_' prefix for right."""
        left = IKChain.leg_chain(side="left")
        right = IKChain.leg_chain(side="right")

        # All left bones start with l_
        for bone in left.all_bones:
            assert bone.startswith("l_")

        # All right bones start with r_
        for bone in right.all_bones:
            assert bone.startswith("r_")


# =============================================================================
# IKChain Factory Method Edge Cases
# =============================================================================


class TestIKChainFactoryEdgeCases:
    """Edge case tests for factory methods."""

    def test_arm_chain_unknown_side_uses_right_prefix(self) -> None:
        """arm_chain with unknown side uses r_ prefix (not 'left')."""
        chain = IKChain.arm_chain(side="center")
        # When side != "left", prefix is "r_"
        assert chain.root_bone == "r_shoulder"
        assert chain.name == "center_arm"

    def test_leg_chain_unknown_side_uses_right_prefix(self) -> None:
        """leg_chain with unknown side uses r_ prefix (not 'left')."""
        chain = IKChain.leg_chain(side="middle")
        # When side != "left", prefix is "r_"
        assert chain.root_bone == "r_thigh"
        assert chain.name == "middle_leg"

    def test_factory_methods_create_independent_instances(self) -> None:
        """Factory methods create independent chain instances."""
        chain1 = IKChain.arm_chain(side="left")
        chain2 = IKChain.arm_chain(side="left")

        chain1.set_weight(0.5)
        assert chain2.weight == 1.0  # Unchanged

        chain1.joint_bones.append("extra")
        assert len(chain2.joint_bones) == 2  # Unchanged


# =============================================================================
# IKChain Mutability Tests
# =============================================================================


class TestIKChainMutability:
    """Tests for IKChain mutability behavior."""

    def test_name_is_mutable(self) -> None:
        """name field can be modified after creation."""
        chain = IKChain(name="old", root_bone="root")
        chain.name = "new"
        assert chain.name == "new"

    def test_root_bone_is_mutable(self) -> None:
        """root_bone field can be modified after creation."""
        chain = IKChain(name="test", root_bone="old")
        chain.root_bone = "new"
        assert chain.root_bone == "new"

    def test_joint_bones_list_is_mutable(self) -> None:
        """joint_bones list can be modified after creation."""
        chain = IKChain(name="test", root_bone="root", joint_bones=["a"])
        chain.joint_bones.append("b")
        assert chain.joint_bones == ["a", "b"]

    def test_effector_bone_is_mutable(self) -> None:
        """effector_bone field can be modified after creation."""
        chain = IKChain(name="test", root_bone="root", effector_bone="old")
        chain.effector_bone = "new"
        assert chain.effector_bone == "new"

    def test_solver_type_is_mutable(self) -> None:
        """solver_type field can be modified after creation."""
        chain = IKChain(name="test", root_bone="root")
        chain.solver_type = IKSolverType.FABRIK
        assert chain.solver_type == IKSolverType.FABRIK

    def test_weight_direct_assignment_no_clamping(self) -> None:
        """Direct weight assignment bypasses clamping (only __post_init__ and set_weight clamp)."""
        chain = IKChain(name="test", root_bone="root")
        chain.weight = 2.0  # Direct assignment
        assert chain.weight == 2.0  # Not clamped by direct assignment

    def test_priority_is_mutable(self) -> None:
        """priority field can be modified after creation."""
        chain = IKChain(name="test", root_bone="root")
        chain.priority = 100
        assert chain.priority == 100

    def test_enabled_direct_assignment(self) -> None:
        """enabled field can be modified via direct assignment."""
        chain = IKChain(name="test", root_bone="root")
        chain.enabled = False
        assert chain.enabled is False


# =============================================================================
# IKChain Comparison and Hashing Tests
# =============================================================================


class TestIKChainComparison:
    """Tests for IKChain comparison behavior (dataclass default)."""

    def test_equality_same_values(self) -> None:
        """Chains with same values are equal."""
        chain1 = IKChain(name="test", root_bone="root", effector_bone="end")
        chain2 = IKChain(name="test", root_bone="root", effector_bone="end")
        assert chain1 == chain2

    def test_equality_different_name(self) -> None:
        """Chains with different names are not equal."""
        chain1 = IKChain(name="test1", root_bone="root")
        chain2 = IKChain(name="test2", root_bone="root")
        assert chain1 != chain2

    def test_equality_different_weight(self) -> None:
        """Chains with different weights are not equal."""
        chain1 = IKChain(name="test", root_bone="root", weight=0.5)
        chain2 = IKChain(name="test", root_bone="root", weight=0.7)
        assert chain1 != chain2

    def test_equality_different_solver_type(self) -> None:
        """Chains with different solver types are not equal."""
        chain1 = IKChain(name="test", root_bone="root", solver_type=IKSolverType.TWO_BONE)
        chain2 = IKChain(name="test", root_bone="root", solver_type=IKSolverType.FABRIK)
        assert chain1 != chain2
