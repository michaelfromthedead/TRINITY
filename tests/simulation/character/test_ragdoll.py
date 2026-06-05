"""
Whitebox tests for engine/simulation/character/ragdoll.py

Tests Ragdoll, RagdollSetup, RagdollState, and ragdoll physics.
"""

import pytest
from engine.simulation.character.ragdoll import (
    BodyPartType,
    Ragdoll,
    RagdollBodyDef,
    RagdollBodyState,
    RagdollJointDef,
    RagdollPhysicsInterface,
    RagdollPose,
    RagdollSetup,
    RagdollState,
    SkeletonInterface,
    create_default_humanoid_setup,
)
from engine.simulation.character.character_controller import (
    Quaternion,
    Transform,
    Vector3,
)


class TestRagdollState:
    """Tests for RagdollState enum."""

    def test_inactive_value(self):
        """INACTIVE should have expected value."""
        assert RagdollState.INACTIVE.value == "inactive"

    def test_activating_value(self):
        """ACTIVATING should have expected value."""
        assert RagdollState.ACTIVATING.value == "activating"

    def test_active_value(self):
        """ACTIVE should have expected value."""
        assert RagdollState.ACTIVE.value == "active"

    def test_settling_value(self):
        """SETTLING should have expected value."""
        assert RagdollState.SETTLING.value == "settling"

    def test_recovering_value(self):
        """RECOVERING should have expected value."""
        assert RagdollState.RECOVERING.value == "recovering"


class TestBodyPartType:
    """Tests for BodyPartType enum."""

    def test_pelvis_value(self):
        """PELVIS should have expected value."""
        assert BodyPartType.PELVIS.value == "pelvis"

    def test_head_value(self):
        """HEAD should have expected value."""
        assert BodyPartType.HEAD.value == "head"

    def test_all_limbs_present(self):
        """All major limb parts should be defined."""
        parts = [p.value for p in BodyPartType]
        assert "upper_arm_l" in parts
        assert "upper_arm_r" in parts
        assert "upper_leg_l" in parts
        assert "upper_leg_r" in parts
        assert "hand_l" in parts
        assert "hand_r" in parts
        assert "foot_l" in parts
        assert "foot_r" in parts


class TestRagdollBodyDef:
    """Tests for RagdollBodyDef dataclass."""

    def test_default_construction(self):
        """Default RagdollBodyDef should have reasonable defaults."""
        body = RagdollBodyDef(
            part_type=BodyPartType.HEAD,
            bone_name="Head",
        )
        assert body.part_type == BodyPartType.HEAD
        assert body.bone_name == "Head"
        assert body.mass == 1.0
        assert body.shape_type == "capsule"

    def test_custom_construction(self):
        """RagdollBodyDef should accept custom values."""
        body = RagdollBodyDef(
            part_type=BodyPartType.HEAD,
            bone_name="Head",
            mass=5.0,
            shape_type="sphere",
            dimensions=(0.1,),
        )
        assert body.mass == 5.0
        assert body.shape_type == "sphere"
        assert body.dimensions == (0.1,)


class TestRagdollJointDef:
    """Tests for RagdollJointDef dataclass."""

    def test_default_construction(self):
        """Default RagdollJointDef should have reasonable defaults."""
        joint = RagdollJointDef(
            parent_part=BodyPartType.NECK,
            child_part=BodyPartType.HEAD,
        )
        assert joint.parent_part == BodyPartType.NECK
        assert joint.child_part == BodyPartType.HEAD
        assert joint.joint_type == "cone"
        assert joint.swing_limit == 45.0
        assert joint.twist_limit == 30.0

    def test_custom_construction(self):
        """RagdollJointDef should accept custom values."""
        joint = RagdollJointDef(
            parent_part=BodyPartType.UPPER_ARM_L,
            child_part=BodyPartType.LOWER_ARM_L,
            joint_type="hinge",
            swing_limit=140.0,
            twist_limit=0.0,
        )
        assert joint.joint_type == "hinge"
        assert joint.swing_limit == 140.0
        assert joint.twist_limit == 0.0


class TestRagdollSetup:
    """Tests for RagdollSetup dataclass."""

    def test_default_construction(self):
        """Default RagdollSetup should have empty collections."""
        setup = RagdollSetup()
        assert len(setup.bodies_per_bone) == 0
        assert len(setup.joints) == 0
        assert setup.total_mass == 70.0
        assert setup.self_collision is False

    def test_custom_construction(self):
        """RagdollSetup should accept custom values."""
        setup = RagdollSetup(
            total_mass=80.0,
            self_collision=True,
        )
        assert setup.total_mass == 80.0
        assert setup.self_collision is True


class TestCreateDefaultHumanoidSetup:
    """Tests for create_default_humanoid_setup function."""

    def test_creates_setup(self):
        """create_default_humanoid_setup should create valid setup."""
        setup = create_default_humanoid_setup()
        assert setup is not None
        assert isinstance(setup, RagdollSetup)

    def test_has_all_body_parts(self):
        """Setup should have bodies for all major parts."""
        setup = create_default_humanoid_setup()
        bone_names = list(setup.bodies_per_bone.keys())
        assert len(bone_names) >= 20  # At least 20 body parts

    def test_has_joints(self):
        """Setup should have joints defined."""
        setup = create_default_humanoid_setup()
        assert len(setup.joints) >= 15  # At least 15 joints

    def test_total_mass_calculated(self):
        """Total mass should be sum of all body masses."""
        setup = create_default_humanoid_setup()
        calculated_mass = sum(
            body.mass for body in setup.bodies_per_bone.values()
        )
        assert setup.total_mass == pytest.approx(calculated_mass)

    def test_pelvis_defined(self):
        """Pelvis should be defined."""
        setup = create_default_humanoid_setup()
        pelvis = next(
            (b for b in setup.bodies_per_bone.values()
             if b.part_type == BodyPartType.PELVIS),
            None
        )
        assert pelvis is not None

    def test_head_defined(self):
        """Head should be defined."""
        setup = create_default_humanoid_setup()
        head = next(
            (b for b in setup.bodies_per_bone.values()
             if b.part_type == BodyPartType.HEAD),
            None
        )
        assert head is not None


class TestRagdollBodyState:
    """Tests for RagdollBodyState dataclass."""

    def test_default_construction(self):
        """Default RagdollBodyState should have reasonable defaults."""
        state = RagdollBodyState()
        assert state.body_id == 0
        assert state.is_kinematic is False

    def test_custom_construction(self):
        """RagdollBodyState should accept custom values."""
        state = RagdollBodyState(
            body_id=42,
            position=Vector3(1.0, 2.0, 3.0),
            velocity=Vector3(0.0, -1.0, 0.0),
            is_kinematic=True,
        )
        assert state.body_id == 42
        assert state.position.y == 2.0
        assert state.is_kinematic is True


class TestRagdollPose:
    """Tests for RagdollPose dataclass."""

    def test_default_construction(self):
        """Default RagdollPose should have empty body states."""
        pose = RagdollPose()
        assert len(pose.body_states) == 0

    def test_custom_construction(self):
        """RagdollPose should store body states."""
        pose = RagdollPose()
        pose.body_states[BodyPartType.HEAD] = RagdollBodyState(
            body_id=1,
            position=Vector3(0.0, 1.8, 0.0),
        )
        assert BodyPartType.HEAD in pose.body_states


class MockRagdollPhysics(RagdollPhysicsInterface):
    """Mock physics interface for ragdoll testing."""

    def __init__(self):
        self.bodies: dict[int, dict] = {}
        self.joints: list[int] = []
        self.next_id = 1

    def create_body(self, position, rotation, shape_type, dimensions,
                    mass, collision_group, collision_mask):
        body_id = self.next_id
        self.next_id += 1
        self.bodies[body_id] = {
            "position": position,
            "rotation": rotation,
            "velocity": Vector3.zero(),
            "angular_velocity": Vector3.zero(),
            "kinematic": False,
        }
        return body_id

    def destroy_body(self, body_id):
        if body_id in self.bodies:
            del self.bodies[body_id]

    def create_joint(self, parent_body, child_body, joint_type,
                     anchor_parent, anchor_child, limits):
        joint_id = self.next_id
        self.next_id += 1
        self.joints.append(joint_id)
        return joint_id

    def destroy_joint(self, joint_id):
        if joint_id in self.joints:
            self.joints.remove(joint_id)

    def get_body_transform(self, body_id):
        body = self.bodies.get(body_id)
        if body:
            return body["position"], body["rotation"]
        return Vector3.zero(), Quaternion.identity()

    def get_body_velocity(self, body_id):
        body = self.bodies.get(body_id)
        if body:
            return body["velocity"], body["angular_velocity"]
        return Vector3.zero(), Vector3.zero()

    def set_body_transform(self, body_id, position, rotation):
        if body_id in self.bodies:
            self.bodies[body_id]["position"] = position
            self.bodies[body_id]["rotation"] = rotation

    def set_body_velocity(self, body_id, linear, angular):
        if body_id in self.bodies:
            self.bodies[body_id]["velocity"] = linear
            self.bodies[body_id]["angular_velocity"] = angular

    def set_body_kinematic(self, body_id, kinematic):
        if body_id in self.bodies:
            self.bodies[body_id]["kinematic"] = kinematic

    def apply_impulse(self, body_id, impulse, point=None):
        if body_id in self.bodies:
            self.bodies[body_id]["velocity"] = (
                self.bodies[body_id]["velocity"] + impulse
            )


class MockSkeleton(SkeletonInterface):
    """Mock skeleton interface for testing."""

    def __init__(self):
        self.bones: dict[str, Transform] = {}

    def get_bone_names(self):
        return list(self.bones.keys())

    def get_bone_transform(self, bone_name):
        return self.bones.get(bone_name, Transform())

    def get_bone_parent(self, bone_name):
        return None

    def set_bone_transform(self, bone_name, transform):
        self.bones[bone_name] = transform


class TestRagdoll:
    """Tests for Ragdoll class."""

    def test_construction(self):
        """Ragdoll should be constructible."""
        physics = MockRagdollPhysics()
        skeleton = MockSkeleton()
        ragdoll = Ragdoll(physics, skeleton)
        assert ragdoll is not None
        assert ragdoll.state == RagdollState.INACTIVE

    def test_construction_with_custom_setup(self):
        """Ragdoll should accept custom setup."""
        physics = MockRagdollPhysics()
        skeleton = MockSkeleton()
        setup = RagdollSetup()
        ragdoll = Ragdoll(physics, skeleton, setup)
        assert ragdoll is not None

    def test_state_property(self):
        """state property should return current state."""
        physics = MockRagdollPhysics()
        skeleton = MockSkeleton()
        ragdoll = Ragdoll(physics, skeleton)
        assert ragdoll.state == RagdollState.INACTIVE

    def test_is_active_false_initially(self):
        """is_active should return False initially."""
        physics = MockRagdollPhysics()
        skeleton = MockSkeleton()
        ragdoll = Ragdoll(physics, skeleton)
        assert ragdoll.is_active is False

    def test_is_settled_false_initially(self):
        """is_settled should return False initially."""
        physics = MockRagdollPhysics()
        skeleton = MockSkeleton()
        ragdoll = Ragdoll(physics, skeleton)
        assert ragdoll.is_settled is False

    def test_blend_progress_zero_initially(self):
        """blend_progress should be 0 initially."""
        physics = MockRagdollPhysics()
        skeleton = MockSkeleton()
        ragdoll = Ragdoll(physics, skeleton)
        assert ragdoll.blend_progress == 0.0

    def test_setup_from_skeleton(self):
        """setup_from_skeleton should create bodies."""
        physics = MockRagdollPhysics()
        skeleton = MockSkeleton()
        setup = create_default_humanoid_setup()
        # Add bone transforms
        for bone_name in setup.bodies_per_bone.keys():
            skeleton.bones[bone_name] = Transform()
        ragdoll = Ragdoll(physics, skeleton, setup)
        result = ragdoll.setup_from_skeleton()
        assert result is True
        assert len(physics.bodies) > 0

    def test_setup_from_skeleton_creates_joints(self):
        """setup_from_skeleton should create joints."""
        physics = MockRagdollPhysics()
        skeleton = MockSkeleton()
        setup = create_default_humanoid_setup()
        for bone_name in setup.bodies_per_bone.keys():
            skeleton.bones[bone_name] = Transform()
        ragdoll = Ragdoll(physics, skeleton, setup)
        ragdoll.setup_from_skeleton()
        assert len(physics.joints) > 0

    def test_destroy(self):
        """destroy should remove all bodies and joints."""
        physics = MockRagdollPhysics()
        skeleton = MockSkeleton()
        setup = create_default_humanoid_setup()
        for bone_name in setup.bodies_per_bone.keys():
            skeleton.bones[bone_name] = Transform()
        ragdoll = Ragdoll(physics, skeleton, setup)
        ragdoll.setup_from_skeleton()
        ragdoll.destroy()
        assert len(physics.bodies) == 0
        assert len(physics.joints) == 0
        assert ragdoll.state == RagdollState.INACTIVE

    def test_activate(self):
        """activate should change state to ACTIVATING."""
        physics = MockRagdollPhysics()
        skeleton = MockSkeleton()
        setup = create_default_humanoid_setup()
        for bone_name in setup.bodies_per_bone.keys():
            skeleton.bones[bone_name] = Transform()
        ragdoll = Ragdoll(physics, skeleton, setup)
        ragdoll.activate(time=0.0)
        assert ragdoll.state == RagdollState.ACTIVATING
        assert ragdoll.is_active is True

    def test_activate_with_velocity(self):
        """activate should apply initial velocity to bodies."""
        physics = MockRagdollPhysics()
        skeleton = MockSkeleton()
        setup = create_default_humanoid_setup()
        for bone_name in setup.bodies_per_bone.keys():
            skeleton.bones[bone_name] = Transform()
        ragdoll = Ragdoll(physics, skeleton, setup)
        ragdoll.activate(initial_velocity=Vector3(0.0, -5.0, 0.0), time=0.0)
        # Check that bodies have velocity
        for body in physics.bodies.values():
            assert body["velocity"].y == -5.0

    def test_activate_with_impulse(self):
        """activate should apply impulse at point."""
        physics = MockRagdollPhysics()
        skeleton = MockSkeleton()
        setup = create_default_humanoid_setup()
        for bone_name in setup.bodies_per_bone.keys():
            skeleton.bones[bone_name] = Transform()
        ragdoll = Ragdoll(physics, skeleton, setup)
        ragdoll.activate(
            impulse_point=Vector3(0.0, 1.0, 0.0),
            impulse=Vector3(10.0, 0.0, 0.0),
            time=0.0,
        )
        # Some body should have received impulse
        has_impulse = any(
            body["velocity"].magnitude() > 0
            for body in physics.bodies.values()
        )
        assert has_impulse

    def test_activate_callback(self):
        """activate should trigger callback."""
        physics = MockRagdollPhysics()
        skeleton = MockSkeleton()
        setup = create_default_humanoid_setup()
        for bone_name in setup.bodies_per_bone.keys():
            skeleton.bones[bone_name] = Transform()
        ragdoll = Ragdoll(physics, skeleton, setup)
        activated = []
        ragdoll.set_activate_callback(lambda: activated.append(True))
        ragdoll.activate(time=0.0)
        assert len(activated) == 1

    def test_deactivate(self):
        """deactivate should change state to RECOVERING."""
        physics = MockRagdollPhysics()
        skeleton = MockSkeleton()
        setup = create_default_humanoid_setup()
        for bone_name in setup.bodies_per_bone.keys():
            skeleton.bones[bone_name] = Transform()
        ragdoll = Ragdoll(physics, skeleton, setup)
        ragdoll.activate(time=0.0)
        ragdoll.deactivate()
        assert ragdoll.state == RagdollState.RECOVERING

    def test_deactivate_callback(self):
        """deactivate should trigger callback."""
        physics = MockRagdollPhysics()
        skeleton = MockSkeleton()
        setup = create_default_humanoid_setup()
        for bone_name in setup.bodies_per_bone.keys():
            skeleton.bones[bone_name] = Transform()
        ragdoll = Ragdoll(physics, skeleton, setup)
        ragdoll.activate(time=0.0)
        deactivated = []
        ragdoll.set_deactivate_callback(lambda: deactivated.append(True))
        ragdoll.deactivate()
        assert len(deactivated) == 1

    def test_deactivate_when_not_active(self):
        """deactivate when not active should do nothing."""
        physics = MockRagdollPhysics()
        skeleton = MockSkeleton()
        ragdoll = Ragdoll(physics, skeleton)
        ragdoll.deactivate()  # Should not error
        assert ragdoll.state == RagdollState.INACTIVE

    def test_update_inactive(self):
        """update when inactive should do nothing."""
        physics = MockRagdollPhysics()
        skeleton = MockSkeleton()
        ragdoll = Ragdoll(physics, skeleton)
        ragdoll.update(0.1, 0.1)  # Should not error
        assert ragdoll.state == RagdollState.INACTIVE

    def test_update_activating_progress(self):
        """update during activating should increase blend progress."""
        physics = MockRagdollPhysics()
        skeleton = MockSkeleton()
        setup = create_default_humanoid_setup()
        for bone_name in setup.bodies_per_bone.keys():
            skeleton.bones[bone_name] = Transform()
        ragdoll = Ragdoll(physics, skeleton, setup)
        ragdoll.activate(time=0.0)
        ragdoll.update(0.1, 0.1)
        assert ragdoll.blend_progress > 0.0

    def test_get_pose(self):
        """get_pose should return current ragdoll pose."""
        physics = MockRagdollPhysics()
        skeleton = MockSkeleton()
        setup = create_default_humanoid_setup()
        for bone_name in setup.bodies_per_bone.keys():
            skeleton.bones[bone_name] = Transform()
        ragdoll = Ragdoll(physics, skeleton, setup)
        ragdoll.setup_from_skeleton()
        pose = ragdoll.get_pose()
        assert isinstance(pose, RagdollPose)
        assert len(pose.body_states) > 0

    def test_get_bone_transform_unknown_bone(self):
        """get_bone_transform for unknown bone should return None."""
        physics = MockRagdollPhysics()
        skeleton = MockSkeleton()
        ragdoll = Ragdoll(physics, skeleton)
        result = ragdoll.get_bone_transform("UnknownBone")
        assert result is None

    def test_get_center_of_mass(self):
        """get_center_of_mass should return weighted average position."""
        physics = MockRagdollPhysics()
        skeleton = MockSkeleton()
        setup = create_default_humanoid_setup()
        for bone_name in setup.bodies_per_bone.keys():
            skeleton.bones[bone_name] = Transform()
        ragdoll = Ragdoll(physics, skeleton, setup)
        ragdoll.setup_from_skeleton()
        com = ragdoll.get_center_of_mass()
        assert isinstance(com, Vector3)

    def test_get_average_velocity(self):
        """get_average_velocity should return weighted average."""
        physics = MockRagdollPhysics()
        skeleton = MockSkeleton()
        setup = create_default_humanoid_setup()
        for bone_name in setup.bodies_per_bone.keys():
            skeleton.bones[bone_name] = Transform()
        ragdoll = Ragdoll(physics, skeleton, setup)
        ragdoll.setup_from_skeleton()
        # Set some velocities
        for body in physics.bodies.values():
            body["velocity"] = Vector3(0.0, -1.0, 0.0)
        avg_vel = ragdoll.get_average_velocity()
        assert avg_vel.y == pytest.approx(-1.0)

    def test_write_to_skeleton(self):
        """write_to_skeleton should update skeleton transforms."""
        physics = MockRagdollPhysics()
        skeleton = MockSkeleton()
        setup = create_default_humanoid_setup()
        for bone_name in setup.bodies_per_bone.keys():
            skeleton.bones[bone_name] = Transform()
        ragdoll = Ragdoll(physics, skeleton, setup)
        ragdoll.setup_from_skeleton()
        # Move a body
        first_body_id = list(physics.bodies.keys())[0]
        physics.bodies[first_body_id]["position"] = Vector3(10.0, 20.0, 30.0)
        ragdoll.write_to_skeleton()
        # Some bone should have updated position
        has_updated = any(
            bone.position.x > 0 or bone.position.y > 0 or bone.position.z > 0
            for bone in skeleton.bones.values()
        )
        assert has_updated

    def test_get_debug_info(self):
        """get_debug_info should return debug dictionary."""
        physics = MockRagdollPhysics()
        skeleton = MockSkeleton()
        ragdoll = Ragdoll(physics, skeleton)
        info = ragdoll.get_debug_info()
        assert "state" in info
        assert "body_count" in info
        assert "joint_count" in info
        assert "blend_progress" in info


class TestSkeletonInterface:
    """Tests for SkeletonInterface base class."""

    def test_get_bone_names_default(self):
        """Default get_bone_names should return empty list."""
        skeleton = SkeletonInterface()
        assert skeleton.get_bone_names() == []

    def test_get_bone_transform_default(self):
        """Default get_bone_transform should return default Transform."""
        skeleton = SkeletonInterface()
        transform = skeleton.get_bone_transform("any")
        assert isinstance(transform, Transform)

    def test_get_bone_parent_default(self):
        """Default get_bone_parent should return None."""
        skeleton = SkeletonInterface()
        assert skeleton.get_bone_parent("any") is None


class TestRagdollPhysicsInterface:
    """Tests for RagdollPhysicsInterface base class."""

    def test_create_body_default(self):
        """Default create_body should return 0."""
        physics = RagdollPhysicsInterface()
        result = physics.create_body(
            Vector3.zero(), Quaternion.identity(),
            "capsule", (0.1, 0.2), 1.0, 0, 0
        )
        assert result == 0

    def test_get_body_transform_default(self):
        """Default get_body_transform should return identity."""
        physics = RagdollPhysicsInterface()
        pos, rot = physics.get_body_transform(1)
        assert pos.x == 0.0
        assert rot.w == 1.0

    def test_get_body_velocity_default(self):
        """Default get_body_velocity should return zero."""
        physics = RagdollPhysicsInterface()
        linear, angular = physics.get_body_velocity(1)
        assert linear.x == 0.0
        assert angular.x == 0.0
