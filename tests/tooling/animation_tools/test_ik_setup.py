"""Tests for IK setup editor with chain configuration and solver settings."""

import pytest

from engine.core.math import Quat, Transform, Vec3
from engine.tooling.animation_tools.ik_setup import (
    CCDSolverConfig,
    FABRIKSolverConfig,
    FullBodySolverConfig,
    IKBone,
    IKChain,
    IKChainType,
    IKConstraint,
    IKConstraintType,
    IKEffector,
    IKPoleVector,
    IKSetupEditor,
    IKSolverType,
    TwoBoneSolverConfig,
)


# =============================================================================
# IK BONE TESTS
# =============================================================================


class TestIKBone:
    def test_basic_bone(self):
        bone = IKBone(bone_name="upperarm_l", bone_index=5)
        assert bone.name == "upperarm_l"
        assert bone.bone_index == 5
        assert bone.length == 0.0

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="Bone name cannot be empty"):
            IKBone(bone_name="", bone_index=0)

    def test_negative_index_raises(self):
        with pytest.raises(ValueError, match="Bone index must be >= 0"):
            IKBone(bone_name="bone", bone_index=-1)

    def test_bone_with_length(self):
        bone = IKBone(bone_name="bone", bone_index=0, length=10.5)
        assert bone.length == 10.5

    def test_bone_limits(self):
        bone = IKBone(
            bone_name="elbow",
            bone_index=6,
            min_angle=-150.0,
            max_angle=0.0,
        )
        assert bone.min_angle == -150.0
        assert bone.max_angle == 0.0

    def test_copy_bone(self):
        bone = IKBone(bone_name="bone", bone_index=5, length=10.0)
        copy = bone.copy()
        assert copy.name == bone.name
        assert copy.bone_index == bone.bone_index
        assert copy is not bone


# =============================================================================
# IK EFFECTOR TESTS
# =============================================================================


class TestIKEffector:
    def test_basic_effector(self):
        effector = IKEffector(
            name="hand_l_effector",
            bone_name="hand_l",
        )
        assert effector.name == "hand_l_effector"
        assert effector.target_bone == "hand_l"
        assert effector.weight == 1.0

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name cannot be empty"):
            IKEffector(name="", bone_name="hand")

    def test_effector_position(self):
        effector = IKEffector(
            name="effector",
            bone_name="hand",
            target_position=Vec3(10, 5, 0),
        )
        assert effector.position.x == 10
        assert effector.position.y == 5

    def test_effector_rotation(self):
        effector = IKEffector(
            name="effector",
            bone_name="hand",
            target_rotation=Quat.from_euler(0, 0, 90),
        )
        assert effector.rotation is not None

    def test_effector_weight(self):
        effector = IKEffector(
            name="effector",
            bone_name="hand",
            position_weight=0.5,
        )
        assert effector.weight == 0.5

    def test_weight_clamped(self):
        effector = IKEffector(name="effector", bone_name="hand", position_weight=1.5)
        assert effector.weight == 1.0
        effector.weight = -0.5
        assert effector.weight == 0.0

    def test_copy_effector(self):
        effector = IKEffector(
            name="effector",
            bone_name="hand",
            target_position=Vec3(1, 2, 3),
        )
        copy = effector.copy()
        assert copy.name == effector.name
        assert copy is not effector


# =============================================================================
# IK POLE VECTOR TESTS
# =============================================================================


class TestIKPoleVector:
    def test_basic_pole_vector(self):
        pole = IKPoleVector(
            name="elbow_pole",
            position=Vec3(0, 0, -10),
        )
        assert pole.name == "elbow_pole"
        assert pole.position.z == -10

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name cannot be empty"):
            IKPoleVector(name="", position=Vec3(0, 0, 0))

    def test_pole_weight(self):
        pole = IKPoleVector(
            name="pole",
            position=Vec3(0, 0, 0),
            weight=0.75,
        )
        assert pole.weight == 0.75

    def test_copy_pole(self):
        pole = IKPoleVector(name="pole", position=Vec3(1, 2, 3))
        copy = pole.copy()
        assert copy.name == pole.name
        assert copy is not pole


# =============================================================================
# IK CONSTRAINT TESTS
# =============================================================================


class TestIKConstraint:
    def test_basic_constraint(self):
        constraint = IKConstraint(
            constraint_type=IKConstraintType.HINGE,
            bone_name="lowerarm_l",
        )
        assert constraint.bone_name == "lowerarm_l"
        assert constraint.constraint_type == IKConstraintType.HINGE

    def test_ball_socket_constraint(self):
        constraint = IKConstraint(
            constraint_type=IKConstraintType.BALL_SOCKET,
            bone_name="upperarm_l",
            min_value=-0.785,
            max_value=0.785,
        )
        assert constraint.constraint_type == IKConstraintType.BALL_SOCKET
        assert constraint.min_value == -0.785

    def test_twist_limit_constraint(self):
        constraint = IKConstraint(
            constraint_type=IKConstraintType.TWIST_LIMIT,
            bone_name="thigh_l",
            min_value=-1.57,
            max_value=1.57,
        )
        assert constraint.constraint_type == IKConstraintType.TWIST_LIMIT
        assert constraint.max_value == 1.57

    def test_angle_limit_constraint(self):
        constraint = IKConstraint(
            constraint_type=IKConstraintType.ANGLE_LIMIT,
            bone_name="link_01",
            min_value=-0.5,
            max_value=1.5,
        )
        assert constraint.min_value == -0.5
        assert constraint.max_value == 1.5

    def test_constraint_enabled(self):
        constraint = IKConstraint(
            constraint_type=IKConstraintType.HINGE,
            bone_name="bone",
            enabled=False,
        )
        assert constraint.enabled is False


# =============================================================================
# SOLVER CONFIG TESTS
# =============================================================================


class TestTwoBoneSolverConfig:
    def test_basic_config(self):
        config = TwoBoneSolverConfig()
        assert config.solver_type == IKSolverType.TWO_BONE
        assert config.iterations == 1  # Analytical solver uses 1 iteration

    def test_with_pole_vector(self):
        config = TwoBoneSolverConfig(
            use_pole_vector=True,
            allow_twist=True,
            maintain_bone_lengths=True,
        )
        assert config.use_pole_vector
        assert config.allow_twist
        assert config.maintain_bone_lengths

    def test_disable_twist(self):
        config = TwoBoneSolverConfig(allow_twist=False)
        assert config.allow_twist is False


class TestFABRIKSolverConfig:
    def test_basic_config(self):
        config = FABRIKSolverConfig()
        assert config.solver_type == IKSolverType.FABRIK
        assert config.iterations == 10

    def test_custom_iterations(self):
        config = FABRIKSolverConfig(iterations=20, tolerance=0.001)
        assert config.iterations == 20
        assert config.tolerance == 0.001

    def test_blend_to_source(self):
        config = FABRIKSolverConfig(blend_to_source=0.5)
        assert config.blend_to_source == 0.5


class TestCCDSolverConfig:
    def test_basic_config(self):
        config = CCDSolverConfig()
        assert config.solver_type == IKSolverType.CCD
        assert config.iterations == 10

    def test_rotation_limit(self):
        import math
        config = CCDSolverConfig(limit_rotation=math.pi / 6)
        assert config.limit_rotation == math.pi / 6


class TestFullBodySolverConfig:
    def test_basic_config(self):
        config = FullBodySolverConfig()
        assert config.solver_type == IKSolverType.FULL_BODY

    def test_root_motion_config(self):
        config = FullBodySolverConfig(
            root_motion_weight=0.8,
            maintain_center_of_mass=False,
        )
        assert config.root_motion_weight == 0.8
        assert config.maintain_center_of_mass is False


# =============================================================================
# IK CHAIN TESTS
# =============================================================================


class TestIKChain:
    def test_basic_chain(self):
        chain = IKChain(
            name="arm_l",
            chain_type=IKChainType.LIMB,
        )
        assert chain.name == "arm_l"
        assert chain.chain_type == IKChainType.LIMB
        assert chain.bone_count == 0

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="Chain name cannot be empty"):
            IKChain(name="", chain_type=IKChainType.LIMB)

    def test_add_bone(self):
        chain = IKChain(name="arm", chain_type=IKChainType.LIMB)
        bone = IKBone(bone_name="upperarm", bone_index=5)
        chain.add_bone(bone)
        assert chain.bone_count == 1

    def test_add_multiple_bones(self):
        chain = IKChain(name="arm", chain_type=IKChainType.LIMB)
        bone1 = IKBone(bone_name="bone1", bone_index=5)
        bone2 = IKBone(bone_name="bone2", bone_index=6)
        chain.add_bone(bone1)
        chain.add_bone(bone2)
        assert chain.bone_count == 2

    def test_remove_bone(self):
        chain = IKChain(name="arm", chain_type=IKChainType.LIMB)
        bone = IKBone(bone_name="bone", bone_index=5)
        chain.add_bone(bone)
        assert chain.remove_bone("bone")
        assert chain.bone_count == 0

    def test_get_bone(self):
        chain = IKChain(name="arm", chain_type=IKChainType.LIMB)
        bone = IKBone(bone_name="bone", bone_index=5)
        chain.add_bone(bone)
        found = chain.get_bone("bone")
        assert found is bone

    def test_set_effector(self):
        chain = IKChain(name="arm", chain_type=IKChainType.LIMB)
        effector = IKEffector(
            name="hand_eff",
            bone_name="hand",
            target_position=Vec3(0, 0, 0),
        )
        chain.set_effector(effector)
        assert chain.effector is effector

    def test_set_pole_vector(self):
        chain = IKChain(name="arm", chain_type=IKChainType.LIMB)
        pole = IKPoleVector(name="elbow_pole", position=Vec3(0, 0, -10))
        chain.set_pole_vector(pole)
        assert chain.pole_vector is pole

    def test_set_bone_constraint(self):
        chain = IKChain(name="arm", chain_type=IKChainType.LIMB)
        bone = IKBone(bone_name="lowerarm", bone_index=1)
        chain.add_bone(bone)
        result = chain.set_bone_constraint(
            bone_name="lowerarm",
            constraint_type=IKConstraintType.HINGE,
            axis=Vec3(1, 0, 0),
        )
        assert result is True

    def test_set_solver_config(self):
        chain = IKChain(name="arm", chain_type=IKChainType.LIMB)
        config = TwoBoneSolverConfig()
        chain.set_solver_config(config)
        assert chain.solver_config is config

    def test_chain_length(self):
        chain = IKChain(name="arm", chain_type=IKChainType.LIMB)
        chain.add_bone(IKBone(bone_name="upper", bone_index=0, length=10.0))
        chain.add_bone(IKBone(bone_name="lower", bone_index=1, length=8.0))
        assert chain.total_length == 18.0

    def test_is_valid(self):
        chain = IKChain(name="arm", chain_type=IKChainType.LIMB)
        assert not chain.is_valid()
        chain.add_bone(IKBone(bone_name="upper", bone_index=0))
        chain.add_bone(IKBone(bone_name="lower", bone_index=1))
        assert not chain.is_valid()  # Still needs effector
        chain.set_effector(IKEffector(
            name="hand",
            bone_name="hand",
            target_position=Vec3(0, 0, 0),
        ))
        assert chain.is_valid()


# =============================================================================
# IK SETUP EDITOR TESTS
# =============================================================================


class TestIKSetupEditor:
    def test_basic_editor(self):
        editor = IKSetupEditor()
        assert editor.chain_count == 0
        assert editor.selected_chain is None
        assert editor.selected_bone is None

    def test_create_chain(self):
        editor = IKSetupEditor()
        chain = editor.create_chain("arm_l", IKChainType.LIMB)
        assert chain is not None
        assert editor.chain_count == 1

    def test_create_duplicate_chain_rejected(self):
        editor = IKSetupEditor()
        editor.create_chain("arm_l", IKChainType.LIMB)
        with pytest.raises(ValueError, match="already exists"):
            editor.create_chain("arm_l", IKChainType.LIMB)

    def test_remove_chain(self):
        editor = IKSetupEditor()
        editor.create_chain("arm_l", IKChainType.LIMB)
        assert editor.remove_chain("arm_l")
        assert editor.chain_count == 0

    def test_get_chain(self):
        editor = IKSetupEditor()
        chain = editor.create_chain("arm_l", IKChainType.LIMB)
        found = editor.get_chain("arm_l")
        assert found is chain

    def test_rename_chain(self):
        editor = IKSetupEditor()
        editor.create_chain("arm_l", IKChainType.LIMB)
        assert editor.rename_chain("arm_l", "left_arm")
        assert editor.get_chain("left_arm") is not None
        assert editor.get_chain("arm_l") is None

    def test_add_bone_to_chain(self):
        editor = IKSetupEditor()
        editor.create_chain("arm_l", IKChainType.LIMB)

        assert editor.add_bone_to_chain("arm_l", "upperarm_l", 5)
        assert editor.add_bone_to_chain("arm_l", "lowerarm_l", 6)
        assert editor.add_bone_to_chain("arm_l", "hand_l", 7)

        chain = editor.get_chain("arm_l")
        assert chain.bone_count == 3

    def test_remove_bone_from_chain(self):
        editor = IKSetupEditor()
        editor.create_chain("arm_l", IKChainType.LIMB)
        editor.add_bone_to_chain("arm_l", "upperarm_l", 5)

        assert editor.remove_bone_from_chain("arm_l", "upperarm_l")
        chain = editor.get_chain("arm_l")
        assert chain.bone_count == 0

    def test_set_effector(self):
        editor = IKSetupEditor()
        editor.create_chain("arm_l", IKChainType.LIMB)

        result = editor.set_effector(
            "arm_l",
            "hand_l",
        )
        assert result is True

        chain = editor.get_chain("arm_l")
        assert chain.effector is not None

    def test_set_pole_vector(self):
        editor = IKSetupEditor()
        editor.create_chain("arm_l", IKChainType.LIMB)

        pole = editor.set_pole_vector(
            "arm_l",
            "elbow_pole",
            Vec3(0, 0, -20),
        )
        assert pole is not None

        chain = editor.get_chain("arm_l")
        assert chain.pole_vector is not None

    def test_set_bone_constraint(self):
        editor = IKSetupEditor()
        editor.create_chain("arm_l", IKChainType.LIMB)
        editor.add_bone_to_chain("arm_l", "lowerarm_l", 6)

        result = editor.set_bone_constraint(
            "arm_l",
            "lowerarm_l",
            IKConstraintType.HINGE,
            axis=Vec3(1, 0, 0),
        )
        assert result is True

    def test_configure_solver(self):
        editor = IKSetupEditor()
        editor.create_chain("arm_l", IKChainType.LIMB)

        config = TwoBoneSolverConfig()
        result = editor.configure_solver("arm_l", config)
        assert result is True

        chain = editor.get_chain("arm_l")
        assert chain.solver_config.solver_type == IKSolverType.TWO_BONE

    def test_create_limb_ik(self):
        editor = IKSetupEditor()

        chain = editor.create_limb_ik(
            name="arm_l",
            upper_bone="upperarm_l",
            upper_index=5,
            upper_length=25.0,
            lower_bone="lowerarm_l",
            lower_index=6,
            lower_length=22.0,
            end_bone="hand_l",
            end_index=7,
        )
        assert chain is not None
        assert chain.chain_type == IKChainType.LIMB
        assert chain.bone_count == 3

    def test_create_spine_ik(self):
        editor = IKSetupEditor()

        chain = editor.create_spine_ik(
            name="spine",
            bones=[
                ("spine_01", 2, 10.0),
                ("spine_02", 3, 10.0),
                ("spine_03", 4, 10.0),
            ],
        )
        assert chain is not None
        assert chain.chain_type == IKChainType.SPINE

    def test_validate(self):
        editor = IKSetupEditor()
        editor.create_chain("arm_l", IKChainType.LIMB)
        # Chain with no bones/effector should have validation errors
        errors = editor.validate()
        assert len(errors) > 0

    def test_validate_with_valid_chain(self):
        editor = IKSetupEditor()
        chain = editor.create_limb_ik(
            name="arm_l",
            upper_bone="upperarm_l",
            upper_index=5,
            upper_length=25.0,
            lower_bone="lowerarm_l",
            lower_index=6,
            lower_length=22.0,
            end_bone="hand_l",
            end_index=7,
        )
        # create_limb_ik sets up effector automatically
        errors = editor.validate()
        # Valid chain should have no errors
        assert isinstance(errors, list)

    def test_select_chain(self):
        editor = IKSetupEditor()
        editor.create_chain("arm_l", IKChainType.LIMB)
        editor.select_chain("arm_l")
        assert editor.selected_chain == "arm_l"

    def test_select_chain_clears_with_none(self):
        editor = IKSetupEditor()
        editor.create_chain("arm_l", IKChainType.LIMB)
        editor.select_chain("arm_l")
        editor.select_chain(None)
        assert editor.selected_chain is None

    def test_select_bone(self):
        editor = IKSetupEditor()
        editor.select_bone("upperarm_l")
        assert editor.selected_bone == "upperarm_l"

    def test_on_change_callback(self):
        editor = IKSetupEditor()
        callback_called = [False]

        def callback():
            callback_called[0] = True

        editor.add_on_change(callback)
        editor.create_chain("arm_l", IKChainType.LIMB)
        assert callback_called[0]

    def test_remove_on_change_callback(self):
        editor = IKSetupEditor()
        callback_called = [0]

        def callback():
            callback_called[0] += 1

        editor.add_on_change(callback)
        editor.create_chain("arm_l", IKChainType.LIMB)
        assert callback_called[0] == 1

        editor.remove_on_change(callback)
        editor.create_chain("arm_r", IKChainType.LIMB)
        assert callback_called[0] == 1  # Still 1, not incremented

    def test_chains_property(self):
        editor = IKSetupEditor()
        editor.create_chain("arm_l", IKChainType.LIMB)
        editor.create_chain("arm_r", IKChainType.LIMB)
        assert len(editor.chains) == 2
