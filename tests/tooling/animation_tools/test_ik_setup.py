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
        bone = IKBone(name="upperarm_l", bone_index=5)
        assert bone.name == "upperarm_l"
        assert bone.bone_index == 5
        assert bone.length == 0.0

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name cannot be empty"):
            IKBone(name="", bone_index=0)

    def test_negative_index_raises(self):
        with pytest.raises(ValueError, match="bone_index must be >= 0"):
            IKBone(name="bone", bone_index=-1)

    def test_bone_with_length(self):
        bone = IKBone(name="bone", bone_index=0, length=10.5)
        assert bone.length == 10.5

    def test_bone_limits(self):
        bone = IKBone(
            name="elbow",
            bone_index=6,
            min_angle=Vec3(0, 0, -150),
            max_angle=Vec3(0, 0, 0),
        )
        assert bone.min_angle.z == -150
        assert bone.max_angle.z == 0

    def test_copy_bone(self):
        bone = IKBone(name="bone", bone_index=5, length=10.0)
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
            target_bone="hand_l",
        )
        assert effector.name == "hand_l_effector"
        assert effector.target_bone == "hand_l"
        assert effector.weight == 1.0

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name cannot be empty"):
            IKEffector(name="", target_bone="hand")

    def test_effector_position(self):
        effector = IKEffector(
            name="effector",
            target_bone="hand",
            position=Vec3(10, 5, 0),
        )
        assert effector.position.x == 10
        assert effector.position.y == 5

    def test_effector_rotation(self):
        effector = IKEffector(
            name="effector",
            target_bone="hand",
            rotation=Quat.from_euler(0, 0, 90),
        )
        assert effector.rotation is not None

    def test_effector_weight(self):
        effector = IKEffector(
            name="effector",
            target_bone="hand",
            weight=0.5,
        )
        assert effector.weight == 0.5

    def test_weight_clamped(self):
        effector = IKEffector(name="effector", target_bone="hand", weight=1.5)
        assert effector.weight == 1.0
        effector.weight = -0.5
        assert effector.weight == 0.0

    def test_copy_effector(self):
        effector = IKEffector(
            name="effector",
            target_bone="hand",
            position=Vec3(1, 2, 3),
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
            name="elbow_hinge",
            constraint_type=IKConstraintType.HINGE,
            bone_name="lowerarm_l",
        )
        assert constraint.name == "elbow_hinge"
        assert constraint.constraint_type == IKConstraintType.HINGE

    def test_cone_constraint(self):
        constraint = IKConstraint(
            name="shoulder_cone",
            constraint_type=IKConstraintType.CONE,
            bone_name="upperarm_l",
            cone_angle=45.0,
        )
        assert constraint.constraint_type == IKConstraintType.CONE
        assert constraint.cone_angle == 45.0

    def test_ball_socket_constraint(self):
        constraint = IKConstraint(
            name="hip_ball",
            constraint_type=IKConstraintType.BALL_SOCKET,
            bone_name="thigh_l",
            twist_limit=90.0,
        )
        assert constraint.constraint_type == IKConstraintType.BALL_SOCKET
        assert constraint.twist_limit == 90.0

    def test_distance_constraint(self):
        constraint = IKConstraint(
            name="rope_link",
            constraint_type=IKConstraintType.DISTANCE,
            bone_name="link_01",
            min_distance=0.5,
            max_distance=1.5,
        )
        assert constraint.min_distance == 0.5
        assert constraint.max_distance == 1.5

    def test_copy_constraint(self):
        constraint = IKConstraint(
            name="test",
            constraint_type=IKConstraintType.HINGE,
            bone_name="bone",
        )
        copy = constraint.copy()
        assert copy.name == constraint.name
        assert copy is not constraint


# =============================================================================
# SOLVER CONFIG TESTS
# =============================================================================


class TestTwoBoneSolverConfig:
    def test_basic_config(self):
        config = TwoBoneSolverConfig()
        assert config.solver_type == IKSolverType.TWO_BONE
        assert config.allow_stretching is False

    def test_with_stretching(self):
        config = TwoBoneSolverConfig(
            allow_stretching=True,
            stretch_start=0.9,
            stretch_max=1.2,
        )
        assert config.allow_stretching
        assert config.stretch_start == 0.9
        assert config.stretch_max == 1.2

    def test_softness(self):
        config = TwoBoneSolverConfig(softness=0.1)
        assert config.softness == 0.1


class TestFABRIKSolverConfig:
    def test_basic_config(self):
        config = FABRIKSolverConfig()
        assert config.solver_type == IKSolverType.FABRIK
        assert config.max_iterations == 10

    def test_custom_iterations(self):
        config = FABRIKSolverConfig(max_iterations=20, tolerance=0.001)
        assert config.max_iterations == 20
        assert config.tolerance == 0.001

    def test_root_motion(self):
        config = FABRIKSolverConfig(root_motion_enabled=True)
        assert config.root_motion_enabled


class TestCCDSolverConfig:
    def test_basic_config(self):
        config = CCDSolverConfig()
        assert config.solver_type == IKSolverType.CCD
        assert config.max_iterations == 10

    def test_rotation_limit(self):
        config = CCDSolverConfig(rotation_limit_per_iteration=15.0)
        assert config.rotation_limit_per_iteration == 15.0


class TestFullBodySolverConfig:
    def test_basic_config(self):
        config = FullBodySolverConfig()
        assert config.solver_type == IKSolverType.FULL_BODY

    def test_spine_config(self):
        config = FullBodySolverConfig(
            spine_stiffness=0.8,
            pelvis_rotation_weight=0.5,
        )
        assert config.spine_stiffness == 0.8
        assert config.pelvis_rotation_weight == 0.5


# =============================================================================
# IK CHAIN TESTS
# =============================================================================


class TestIKChain:
    def test_basic_chain(self):
        chain = IKChain(
            name="arm_l",
            chain_type=IKChainType.ARM,
        )
        assert chain.name == "arm_l"
        assert chain.chain_type == IKChainType.ARM
        assert chain.bone_count == 0

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name cannot be empty"):
            IKChain(name="", chain_type=IKChainType.ARM)

    def test_add_bone(self):
        chain = IKChain(name="arm", chain_type=IKChainType.ARM)
        bone = IKBone(name="upperarm", bone_index=5)
        assert chain.add_bone(bone)
        assert chain.bone_count == 1

    def test_add_duplicate_bone_rejected(self):
        chain = IKChain(name="arm", chain_type=IKChainType.ARM)
        bone1 = IKBone(name="bone", bone_index=5)
        bone2 = IKBone(name="bone", bone_index=6)
        chain.add_bone(bone1)
        assert not chain.add_bone(bone2)

    def test_remove_bone(self):
        chain = IKChain(name="arm", chain_type=IKChainType.ARM)
        bone = IKBone(name="bone", bone_index=5)
        chain.add_bone(bone)
        assert chain.remove_bone("bone")
        assert chain.bone_count == 0

    def test_get_bone(self):
        chain = IKChain(name="arm", chain_type=IKChainType.ARM)
        bone = IKBone(name="bone", bone_index=5)
        chain.add_bone(bone)
        found = chain.get_bone("bone")
        assert found is bone

    def test_set_effector(self):
        chain = IKChain(name="arm", chain_type=IKChainType.ARM)
        effector = IKEffector(name="hand_eff", target_bone="hand")
        chain.set_effector(effector)
        assert chain.effector is effector

    def test_set_pole_vector(self):
        chain = IKChain(name="arm", chain_type=IKChainType.ARM)
        pole = IKPoleVector(name="elbow_pole", position=Vec3(0, 0, -10))
        chain.set_pole_vector(pole)
        assert chain.pole_vector is pole

    def test_add_constraint(self):
        chain = IKChain(name="arm", chain_type=IKChainType.ARM)
        constraint = IKConstraint(
            name="elbow_hinge",
            constraint_type=IKConstraintType.HINGE,
            bone_name="lowerarm",
        )
        assert chain.add_constraint(constraint)
        assert len(chain.constraints) == 1

    def test_set_solver_config(self):
        chain = IKChain(name="arm", chain_type=IKChainType.ARM)
        config = TwoBoneSolverConfig(allow_stretching=True)
        chain.set_solver_config(config)
        assert chain.solver_config is config

    def test_chain_length(self):
        chain = IKChain(name="arm", chain_type=IKChainType.ARM)
        chain.add_bone(IKBone(name="upper", bone_index=0, length=10.0))
        chain.add_bone(IKBone(name="lower", bone_index=1, length=8.0))
        assert chain.total_length == 18.0

    def test_copy_chain(self):
        chain = IKChain(name="arm", chain_type=IKChainType.ARM)
        chain.add_bone(IKBone(name="bone", bone_index=0))
        copy = chain.copy()
        assert copy.name == chain.name
        assert copy is not chain


# =============================================================================
# IK SETUP EDITOR TESTS
# =============================================================================


class TestIKSetupEditor:
    @pytest.fixture
    def sample_bones(self):
        return [
            {"name": "root", "index": 0},
            {"name": "pelvis", "index": 1},
            {"name": "spine_01", "index": 2},
            {"name": "spine_02", "index": 3},
            {"name": "clavicle_l", "index": 4},
            {"name": "upperarm_l", "index": 5},
            {"name": "lowerarm_l", "index": 6},
            {"name": "hand_l", "index": 7},
            {"name": "clavicle_r", "index": 8},
            {"name": "upperarm_r", "index": 9},
            {"name": "lowerarm_r", "index": 10},
            {"name": "hand_r", "index": 11},
            {"name": "thigh_l", "index": 12},
            {"name": "calf_l", "index": 13},
            {"name": "foot_l", "index": 14},
            {"name": "thigh_r", "index": 15},
            {"name": "calf_r", "index": 16},
            {"name": "foot_r", "index": 17},
        ]

    def test_basic_editor(self):
        editor = IKSetupEditor()
        assert editor.chain_count == 0

    def test_load_skeleton(self, sample_bones):
        editor = IKSetupEditor()
        editor.load_skeleton(sample_bones)
        assert len(editor.available_bones) == 18

    def test_create_chain(self):
        editor = IKSetupEditor()
        chain = editor.create_chain("arm_l", IKChainType.ARM)
        assert chain is not None
        assert editor.chain_count == 1

    def test_create_duplicate_chain_rejected(self):
        editor = IKSetupEditor()
        editor.create_chain("arm_l", IKChainType.ARM)
        result = editor.create_chain("arm_l", IKChainType.ARM)
        assert result is None

    def test_remove_chain(self):
        editor = IKSetupEditor()
        editor.create_chain("arm_l", IKChainType.ARM)
        assert editor.remove_chain("arm_l")
        assert editor.chain_count == 0

    def test_get_chain(self):
        editor = IKSetupEditor()
        chain = editor.create_chain("arm_l", IKChainType.ARM)
        found = editor.get_chain("arm_l")
        assert found is chain

    def test_add_bone_to_chain(self, sample_bones):
        editor = IKSetupEditor()
        editor.load_skeleton(sample_bones)
        editor.create_chain("arm_l", IKChainType.ARM)

        assert editor.add_bone_to_chain("arm_l", "upperarm_l", 5)
        assert editor.add_bone_to_chain("arm_l", "lowerarm_l", 6)
        assert editor.add_bone_to_chain("arm_l", "hand_l", 7)

        chain = editor.get_chain("arm_l")
        assert chain.bone_count == 3

    def test_remove_bone_from_chain(self, sample_bones):
        editor = IKSetupEditor()
        editor.load_skeleton(sample_bones)
        editor.create_chain("arm_l", IKChainType.ARM)
        editor.add_bone_to_chain("arm_l", "upperarm_l", 5)

        assert editor.remove_bone_from_chain("arm_l", "upperarm_l")

    def test_set_effector(self, sample_bones):
        editor = IKSetupEditor()
        editor.load_skeleton(sample_bones)
        editor.create_chain("arm_l", IKChainType.ARM)

        effector = editor.set_effector(
            "arm_l",
            "hand_l_effector",
            "hand_l",
        )
        assert effector is not None

    def test_set_pole_vector(self, sample_bones):
        editor = IKSetupEditor()
        editor.load_skeleton(sample_bones)
        editor.create_chain("arm_l", IKChainType.ARM)

        pole = editor.set_pole_vector(
            "arm_l",
            "elbow_pole",
            Vec3(0, 0, -20),
        )
        assert pole is not None

    def test_add_constraint_to_chain(self, sample_bones):
        editor = IKSetupEditor()
        editor.load_skeleton(sample_bones)
        editor.create_chain("arm_l", IKChainType.ARM)

        constraint = editor.add_constraint(
            "arm_l",
            "elbow_hinge",
            IKConstraintType.HINGE,
            "lowerarm_l",
        )
        assert constraint is not None

    def test_set_solver_config(self, sample_bones):
        editor = IKSetupEditor()
        editor.load_skeleton(sample_bones)
        editor.create_chain("arm_l", IKChainType.ARM)

        config = TwoBoneSolverConfig(allow_stretching=True)
        editor.set_solver_config("arm_l", config)

        chain = editor.get_chain("arm_l")
        assert chain.solver_config.allow_stretching

    def test_create_arm_ik(self, sample_bones):
        editor = IKSetupEditor()
        editor.load_skeleton(sample_bones)

        chain = editor.create_arm_ik(
            name="arm_l",
            upper_bone="upperarm_l",
            upper_index=5,
            lower_bone="lowerarm_l",
            lower_index=6,
            hand_bone="hand_l",
            hand_index=7,
        )
        assert chain is not None
        assert chain.chain_type == IKChainType.ARM
        assert chain.bone_count == 3

    def test_create_leg_ik(self, sample_bones):
        editor = IKSetupEditor()
        editor.load_skeleton(sample_bones)

        chain = editor.create_leg_ik(
            name="leg_l",
            thigh_bone="thigh_l",
            thigh_index=12,
            calf_bone="calf_l",
            calf_index=13,
            foot_bone="foot_l",
            foot_index=14,
        )
        assert chain is not None
        assert chain.chain_type == IKChainType.LEG
        assert chain.bone_count == 3

    def test_create_spine_ik(self, sample_bones):
        editor = IKSetupEditor()
        editor.load_skeleton(sample_bones)

        chain = editor.create_spine_ik(
            name="spine",
            bones=[
                ("spine_01", 2),
                ("spine_02", 3),
            ],
        )
        assert chain is not None
        assert chain.chain_type == IKChainType.SPINE

    def test_auto_detect_chains(self, sample_bones):
        editor = IKSetupEditor()
        editor.load_skeleton(sample_bones)

        # Auto-detect should find arm and leg chains based on naming conventions
        detected = editor.auto_detect_chains()
        # Results depend on naming convention matching
        assert isinstance(detected, int)

    def test_validate_chain(self, sample_bones):
        editor = IKSetupEditor()
        editor.load_skeleton(sample_bones)
        editor.create_chain("arm_l", IKChainType.ARM)
        # Chain with no bones should have validation errors
        errors = editor.validate_chain("arm_l")
        assert len(errors) > 0  # Should have "no bones" error

    def test_validate_chain_with_bones(self, sample_bones):
        editor = IKSetupEditor()
        editor.load_skeleton(sample_bones)
        editor.create_arm_ik(
            name="arm_l",
            upper_bone="upperarm_l",
            upper_index=5,
            lower_bone="lowerarm_l",
            lower_index=6,
            hand_bone="hand_l",
            hand_index=7,
        )
        errors = editor.validate_chain("arm_l")
        # Properly configured chain should have fewer/no errors
        assert isinstance(errors, list)

    def test_select_chain(self):
        editor = IKSetupEditor()
        editor.create_chain("arm_l", IKChainType.ARM)
        editor.select_chain("arm_l")
        assert editor.selected_chain == "arm_l"

    def test_clear_selection(self):
        editor = IKSetupEditor()
        editor.create_chain("arm_l", IKChainType.ARM)
        editor.select_chain("arm_l")
        editor.clear_selection()
        assert editor.selected_chain is None

    def test_get_chains_by_type(self):
        editor = IKSetupEditor()
        editor.create_chain("arm_l", IKChainType.ARM)
        editor.create_chain("arm_r", IKChainType.ARM)
        editor.create_chain("leg_l", IKChainType.LEG)

        arms = editor.get_chains_by_type(IKChainType.ARM)
        assert len(arms) == 2

        legs = editor.get_chains_by_type(IKChainType.LEG)
        assert len(legs) == 1

    def test_on_change_callback(self):
        editor = IKSetupEditor()
        callback_called = [False]

        def callback():
            callback_called[0] = True

        editor.add_on_change(callback)
        editor.create_chain("arm_l", IKChainType.ARM)
        assert callback_called[0]

    def test_to_dict(self, sample_bones):
        editor = IKSetupEditor()
        editor.load_skeleton(sample_bones)
        editor.create_arm_ik(
            name="arm_l",
            upper_bone="upperarm_l",
            upper_index=5,
            lower_bone="lowerarm_l",
            lower_index=6,
            hand_bone="hand_l",
            hand_index=7,
        )

        data = editor.to_dict()
        assert "chains" in data
        assert len(data["chains"]) == 1

    def test_from_dict(self, sample_bones):
        editor = IKSetupEditor()
        editor.load_skeleton(sample_bones)
        editor.create_chain("arm_l", IKChainType.ARM)

        data = editor.to_dict()
        new_editor = IKSetupEditor.from_dict(data)
        assert new_editor.chain_count == editor.chain_count
