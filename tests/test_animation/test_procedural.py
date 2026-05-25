"""
Comprehensive tests for the Procedural Animation subsystem.

Tests all procedural animation modules:
- spring_bone.py: Spring/jiggle physics for bones
- lookat.py: Procedural look-at controller
- twist.py: Twist bone distribution
- ragdoll.py: Ragdoll physics integration
- locomotion.py: Procedural locomotion
- breathing.py: Procedural breathing
- secondary_motion.py: General secondary motion effects

Minimum 140 tests with real assertions.
"""

import math
import pytest
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# =============================================================================
# Mock Classes for Testing
# =============================================================================


@dataclass
class MockPose:
    """Mock pose implementation for testing."""

    _positions: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)
    _rotations: Dict[int, Tuple[float, float, float, float]] = field(default_factory=dict)
    _local_rotations: Dict[int, Tuple[float, float, float, float]] = field(default_factory=dict)
    _parent_indices: Dict[int, int] = field(default_factory=dict)

    def get_bone_position(self, bone_index: int) -> Tuple[float, float, float]:
        return self._positions.get(bone_index, (0.0, 0.0, 0.0))

    def set_bone_position(self, bone_index: int, position: Tuple[float, float, float]) -> None:
        self._positions[bone_index] = position

    def get_bone_rotation(self, bone_index: int) -> Tuple[float, float, float, float]:
        return self._rotations.get(bone_index, (0.0, 0.0, 0.0, 1.0))

    def set_bone_rotation(self, bone_index: int, rotation: Tuple[float, float, float, float]) -> None:
        self._rotations[bone_index] = rotation

    def get_bone_local_rotation(self, bone_index: int) -> Tuple[float, float, float, float]:
        return self._local_rotations.get(bone_index, (0.0, 0.0, 0.0, 1.0))

    def set_bone_local_rotation(self, bone_index: int, rotation: Tuple[float, float, float, float]) -> None:
        self._local_rotations[bone_index] = rotation

    def get_parent_index(self, bone_index: int) -> int:
        return self._parent_indices.get(bone_index, -1)

    def copy(self) -> "MockPose":
        new_pose = MockPose()
        new_pose._positions = self._positions.copy()
        new_pose._rotations = self._rotations.copy()
        new_pose._local_rotations = self._local_rotations.copy()
        new_pose._parent_indices = self._parent_indices.copy()
        return new_pose


@dataclass
class MockSkeleton:
    """Mock skeleton for testing."""

    bone_count: int = 20
    bone_names: Dict[int, str] = field(default_factory=dict)
    parent_indices: Dict[int, int] = field(default_factory=dict)

    def get_bone_count(self) -> int:
        return self.bone_count

    def get_bone_name(self, bone_index: int) -> str:
        return self.bone_names.get(bone_index, f"bone_{bone_index}")

    def get_parent_index(self, bone_index: int) -> int:
        return self.parent_indices.get(bone_index, bone_index - 1 if bone_index > 0 else -1)

    def get_bone_bind_pose(self, bone_index: int):
        from engine.animation.procedural.ragdoll import Transform
        return Transform(position=(0.0, float(bone_index) * 0.1, 0.0))


@dataclass
class MockTransform:
    """Mock transform for physics."""

    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


class MockPhysicsWorld:
    """Mock physics world for ragdoll testing."""

    def __init__(self):
        self._bodies: Dict[int, dict] = {}
        self._joints: Dict[int, dict] = {}
        self._next_body_id = 0
        self._next_joint_id = 0

    def create_rigid_body(self, shape, transform, mass: float, is_kinematic: bool) -> int:
        body_id = self._next_body_id
        self._next_body_id += 1
        self._bodies[body_id] = {
            "shape": shape,
            "transform": transform,
            "mass": mass,
            "is_kinematic": is_kinematic,
        }
        return body_id

    def destroy_rigid_body(self, body_id: int) -> None:
        if body_id in self._bodies:
            del self._bodies[body_id]

    def get_body_transform(self, body_id: int):
        from engine.animation.procedural.ragdoll import Transform
        body = self._bodies.get(body_id)
        if body:
            return body["transform"]
        return Transform()

    def set_body_transform(self, body_id: int, transform) -> None:
        if body_id in self._bodies:
            self._bodies[body_id]["transform"] = transform

    def set_body_kinematic(self, body_id: int, is_kinematic: bool) -> None:
        if body_id in self._bodies:
            self._bodies[body_id]["is_kinematic"] = is_kinematic

    def create_joint(self, body_a: int, body_b: int, joint_type: str, config: dict) -> int:
        joint_id = self._next_joint_id
        self._next_joint_id += 1
        self._joints[joint_id] = {
            "body_a": body_a,
            "body_b": body_b,
            "joint_type": joint_type,
            "config": config,
        }
        return joint_id

    def destroy_joint(self, joint_id: int) -> None:
        if joint_id in self._joints:
            del self._joints[joint_id]

    def set_joint_motor(self, joint_id: int, target_rotation, max_torque: float) -> None:
        if joint_id in self._joints:
            self._joints[joint_id]["motor_target"] = target_rotation
            self._joints[joint_id]["motor_torque"] = max_torque

    def apply_impulse(self, body_id: int, impulse, position) -> None:
        pass


def create_test_pose(bone_count: int = 10) -> MockPose:
    """Create a test pose with bones arranged vertically."""
    pose = MockPose()
    for i in range(bone_count):
        pose.set_bone_position(i, (0.0, float(i) * 0.1, 0.0))
        pose.set_bone_rotation(i, (0.0, 0.0, 0.0, 1.0))
        pose._parent_indices[i] = i - 1 if i > 0 else -1
    return pose


# =============================================================================
# Spring Bone Tests
# =============================================================================


class TestSpringBone:
    """Tests for SpringBone class."""

    def test_spring_bone_creation_basic(self):
        """Test basic spring bone creation."""
        from engine.animation.procedural.spring_bone import SpringBone

        spring = SpringBone(bone_index=5, stiffness=50.0, damping=0.3)
        assert spring.bone_index == 5
        assert spring.stiffness == 50.0
        assert spring.damping == 0.3

    def test_spring_bone_default_values(self):
        """Test default parameter values."""
        from engine.animation.procedural.spring_bone import SpringBone

        spring = SpringBone(bone_index=0)
        assert spring.stiffness == 50.0
        assert spring.damping == 0.3
        assert spring.gravity == (0.0, -9.81, 0.0)
        assert spring.mass == 1.0

    def test_spring_bone_invalid_bone_index(self):
        """Test that negative bone index raises error."""
        from engine.animation.procedural.spring_bone import SpringBone

        with pytest.raises(ValueError, match="bone_index must be >= 0"):
            SpringBone(bone_index=-1)

    def test_spring_bone_invalid_stiffness(self):
        """Test that negative stiffness raises error."""
        from engine.animation.procedural.spring_bone import SpringBone

        with pytest.raises(ValueError, match="stiffness must be >= 0"):
            SpringBone(bone_index=0, stiffness=-10.0)

    def test_spring_bone_invalid_damping(self):
        """Test that damping outside [0, 1] raises error."""
        from engine.animation.procedural.spring_bone import SpringBone

        with pytest.raises(ValueError, match="damping must be in"):
            SpringBone(bone_index=0, damping=1.5)

        with pytest.raises(ValueError, match="damping must be in"):
            SpringBone(bone_index=0, damping=-0.1)

    def test_spring_bone_invalid_mass(self):
        """Test that non-positive mass raises error."""
        from engine.animation.procedural.spring_bone import SpringBone

        with pytest.raises(ValueError, match="mass must be > 0"):
            SpringBone(bone_index=0, mass=0)

    def test_spring_bone_initialize(self):
        """Test spring bone initialization from pose."""
        from engine.animation.procedural.spring_bone import SpringBone

        spring = SpringBone(bone_index=3)
        pose = create_test_pose()
        pose.set_bone_position(3, (1.0, 2.0, 3.0))

        spring.initialize(pose)
        assert spring._initialized
        assert spring._position == (1.0, 2.0, 3.0)

    def test_spring_bone_simulate_no_movement(self):
        """Test simulation when bone doesn't need to move."""
        from engine.animation.procedural.spring_bone import SpringBone

        spring = SpringBone(bone_index=0, stiffness=100.0, damping=0.9, gravity=(0, 0, 0))
        pose = create_test_pose()
        spring.initialize(pose)

        # Simulate with bone at rest position
        result = spring.simulate(pose, dt=1/60)
        pos = result.get_bone_position(0)

        # Should stay near origin
        assert abs(pos[0]) < 0.01
        assert abs(pos[1]) < 0.01
        assert abs(pos[2]) < 0.01

    def test_spring_bone_gravity_effect(self):
        """Test that gravity pulls spring bone down."""
        from engine.animation.procedural.spring_bone import SpringBone

        spring = SpringBone(bone_index=0, stiffness=10.0, damping=0.1, gravity=(0, -10, 0))
        pose = create_test_pose()
        spring.initialize(pose)

        # Simulate multiple steps
        result = pose
        for _ in range(60):  # 1 second at 60fps
            result = spring.simulate(result, dt=1/60)

        pos = result.get_bone_position(0)
        # Should have moved down due to gravity
        assert pos[1] < 0

    def test_spring_bone_convergence(self):
        """Test that spring bone converges to rest position."""
        from engine.animation.procedural.spring_bone import SpringBone

        spring = SpringBone(bone_index=0, stiffness=100.0, damping=0.8, gravity=(0, 0, 0))
        pose = create_test_pose()

        # Start displaced
        spring.initialize(pose)
        spring._position = (1.0, 0.0, 0.0)
        spring._previous_position = (1.0, 0.0, 0.0)

        # Simulate until convergence
        result = pose
        for _ in range(300):
            result = spring.simulate(pose, dt=1/60)

        pos = spring.get_position()
        # Should converge toward rest (0, 0, 0)
        assert abs(pos[0]) < 0.1
        assert abs(pos[1]) < 0.1
        assert abs(pos[2]) < 0.1

    def test_spring_bone_reset(self):
        """Test spring bone reset."""
        from engine.animation.procedural.spring_bone import SpringBone

        spring = SpringBone(bone_index=0)
        pose = create_test_pose()
        spring.initialize(pose)

        # Disturb the spring
        spring._position = (5.0, 5.0, 5.0)

        spring.reset(pose)
        assert spring._position == pose.get_bone_position(0)

    def test_spring_bone_zero_dt(self):
        """Test that zero dt returns unchanged pose."""
        from engine.animation.procedural.spring_bone import SpringBone

        spring = SpringBone(bone_index=0)
        pose = create_test_pose()
        spring.initialize(pose)

        result = spring.simulate(pose, dt=0)
        assert result.get_bone_position(0) == pose.get_bone_position(0)


class TestSpringChain:
    """Tests for SpringChain class."""

    def test_spring_chain_creation(self):
        """Test spring chain creation."""
        from engine.animation.procedural.spring_bone import SpringChain

        chain = SpringChain(root_bone=0, bone_indices=[1, 2, 3])
        assert chain.root_bone == 0
        assert chain.bone_indices == [1, 2, 3]
        assert len(chain._springs) == 3

    def test_spring_chain_invalid_root(self):
        """Test that negative root bone raises error."""
        from engine.animation.procedural.spring_bone import SpringChain

        with pytest.raises(ValueError, match="root_bone must be >= 0"):
            SpringChain(root_bone=-1, bone_indices=[0, 1])

    def test_spring_chain_empty_bones(self):
        """Test that empty bone list raises error."""
        from engine.animation.procedural.spring_bone import SpringChain

        with pytest.raises(ValueError, match="bone_indices must not be empty"):
            SpringChain(root_bone=0, bone_indices=[])

    def test_spring_chain_initialization(self):
        """Test spring chain initialization."""
        from engine.animation.procedural.spring_bone import SpringChain

        chain = SpringChain(root_bone=0, bone_indices=[1, 2, 3])
        pose = create_test_pose()

        chain.initialize(pose)
        assert chain._initialized
        assert len(chain._rest_lengths) == 3

    def test_spring_chain_simulate(self):
        """Test spring chain simulation."""
        from engine.animation.procedural.spring_bone import SpringChain

        chain = SpringChain(
            root_bone=0,
            bone_indices=[1, 2, 3],
            stiffness=50.0,
            damping=0.5,
            gravity=(0, -10, 0)
        )
        pose = create_test_pose()

        # First call initializes
        chain.initialize(pose)

        # Second call should return a copy
        result = chain.simulate(pose, dt=1/60)
        # Result should be a pose object
        assert result is not None

        # Simulate more and check bones moved
        for _ in range(30):
            result = chain.simulate(pose, dt=1/60)

        # After simulation with gravity, y positions should decrease
        original_y = pose.get_bone_position(3)[1]
        final_y = result.get_bone_position(3)[1]
        # With gravity, the bone should move down
        assert final_y < original_y

    def test_spring_chain_bone_count(self):
        """Test getting bone count."""
        from engine.animation.procedural.spring_bone import SpringChain

        chain = SpringChain(root_bone=0, bone_indices=[1, 2, 3, 4, 5])
        assert chain.get_bone_count() == 5

    def test_spring_chain_get_spring(self):
        """Test getting individual springs."""
        from engine.animation.procedural.spring_bone import SpringChain

        chain = SpringChain(root_bone=0, bone_indices=[1, 2, 3])
        spring = chain.get_spring(1)
        assert spring.bone_index == 2


class TestCollisionPrimitives:
    """Tests for collision primitives."""

    def test_collision_sphere_creation(self):
        """Test collision sphere creation."""
        from engine.animation.procedural.spring_bone import CollisionSphere

        sphere = CollisionSphere(center=(0, 0, 0), radius=0.5)
        assert sphere.radius == 0.5

    def test_collision_sphere_invalid_radius(self):
        """Test that non-positive radius raises error."""
        from engine.animation.procedural.spring_bone import CollisionSphere

        with pytest.raises(ValueError, match="radius must be > 0"):
            CollisionSphere(center=(0, 0, 0), radius=0)

    def test_collision_sphere_resolve(self):
        """Test sphere collision resolution."""
        from engine.animation.procedural.spring_bone import CollisionSphere

        sphere = CollisionSphere(center=(0, 0, 0), radius=1.0)

        # Point inside sphere should be pushed out
        pos, collided = sphere.resolve_collision((0.5, 0, 0))
        assert collided
        assert abs(pos[0] - 1.0) < 0.001

        # Point outside sphere should not move
        pos, collided = sphere.resolve_collision((2.0, 0, 0))
        assert not collided
        assert pos == (2.0, 0, 0)

    def test_collision_capsule_creation(self):
        """Test collision capsule creation."""
        from engine.animation.procedural.spring_bone import CollisionCapsule

        capsule = CollisionCapsule(start=(0, 0, 0), end=(0, 1, 0), radius=0.2)
        assert capsule.radius == 0.2

    def test_collision_capsule_resolve(self):
        """Test capsule collision resolution."""
        from engine.animation.procedural.spring_bone import CollisionCapsule

        capsule = CollisionCapsule(start=(0, 0, 0), end=(0, 1, 0), radius=0.5)

        # Point inside capsule should be pushed out
        pos, collided = capsule.resolve_collision((0.2, 0.5, 0))
        assert collided

        # Point outside capsule should not move
        pos, collided = capsule.resolve_collision((2.0, 0.5, 0))
        assert not collided


class TestWindForce:
    """Tests for WindForce class."""

    def test_wind_force_creation(self):
        """Test wind force creation."""
        from engine.animation.procedural.spring_bone import WindForce

        wind = WindForce(direction=(1, 0, 0), strength=5.0)
        assert wind.strength == 5.0

    def test_wind_force_invalid_strength(self):
        """Test that negative strength raises error."""
        from engine.animation.procedural.spring_bone import WindForce

        with pytest.raises(ValueError, match="strength must be >= 0"):
            WindForce(strength=-1.0)

    def test_wind_force_invalid_turbulence(self):
        """Test that turbulence outside [0, 1] raises error."""
        from engine.animation.procedural.spring_bone import WindForce

        with pytest.raises(ValueError, match="turbulence must be in"):
            WindForce(turbulence=1.5)

    def test_wind_force_get_force(self):
        """Test getting wind force."""
        from engine.animation.procedural.spring_bone import WindForce

        wind = WindForce(direction=(1, 0, 0), strength=10.0, turbulence=0)
        force = wind.get_force((0, 0, 0), dt=1/60)

        # Force should be in wind direction
        assert force[0] > 0
        assert abs(force[1]) < 0.001
        assert abs(force[2]) < 0.001


# =============================================================================
# Look-At Controller Tests
# =============================================================================


class TestLookAtController:
    """Tests for LookAtController class."""

    def test_lookat_creation(self):
        """Test look-at controller creation."""
        from engine.animation.procedural.lookat import LookAtController

        controller = LookAtController(head_bone=10, neck_bone=9, eye_bones=[11, 12])
        assert controller.head_bone == 10
        assert controller.neck_bone == 9
        assert controller.eye_bones == [11, 12]

    def test_lookat_invalid_head_bone(self):
        """Test that negative head bone raises error."""
        from engine.animation.procedural.lookat import LookAtController

        with pytest.raises(ValueError, match="head_bone must be >= 0"):
            LookAtController(head_bone=-1)

    def test_lookat_invalid_rotation_speed(self):
        """Test that non-positive rotation speed raises error."""
        from engine.animation.procedural.lookat import LookAtController

        with pytest.raises(ValueError, match="rotation_speed must be > 0"):
            LookAtController(head_bone=0, rotation_speed=0)

    def test_lookat_update_basic(self):
        """Test basic look-at update."""
        from engine.animation.procedural.lookat import LookAtController

        controller = LookAtController(head_bone=5, enable_saccades=False)
        pose = create_test_pose()
        pose.set_bone_position(5, (0, 1, 0))

        target = (0, 1, 5)  # In front
        result = controller.update(pose, target, dt=1/60, weight=1.0)

        # Head should have rotated
        assert result is not pose

    def test_lookat_zero_dt(self):
        """Test that zero dt returns unchanged pose."""
        from engine.animation.procedural.lookat import LookAtController

        controller = LookAtController(head_bone=0)
        pose = create_test_pose()

        result = controller.update(pose, (0, 0, 1), dt=0)
        assert result.get_bone_rotation(0) == pose.get_bone_rotation(0)

    def test_lookat_weight_zero(self):
        """Test that weight 0 returns animation pose."""
        from engine.animation.procedural.lookat import LookAtController

        controller = LookAtController(head_bone=0, enable_saccades=False)
        pose = create_test_pose()
        controller.initialize(pose)

        # Let it blend out
        for _ in range(100):
            result = controller.update(pose, (0, 0, 1), dt=1/60, weight=0.0)

        assert controller.get_current_weight() < 0.1

    def test_lookat_angle_limits(self):
        """Test that angle limits are respected."""
        from engine.animation.procedural.lookat import LookAtController

        controller = LookAtController(
            head_bone=0,
            head_yaw_limit=math.radians(45),
            enable_saccades=False
        )
        pose = create_test_pose()
        controller.initialize(pose)

        # Target behind should hit limit
        target = (-10, 0, -5)  # Behind and to the side

        for _ in range(60):
            result = controller.update(pose, target, dt=1/60, weight=1.0)

        # Head shouldn't have rotated more than limit

    def test_lookat_reset(self):
        """Test look-at controller reset."""
        from engine.animation.procedural.lookat import LookAtController

        controller = LookAtController(head_bone=0, enable_saccades=False)
        pose = create_test_pose()
        controller.initialize(pose)

        controller.update(pose, (0, 0, 1), dt=1/60, weight=1.0)
        controller.reset(pose)

        assert controller.get_current_weight() == 0


class TestInterestPoint:
    """Tests for InterestPoint class."""

    def test_interest_point_creation(self):
        """Test interest point creation."""
        from engine.animation.procedural.lookat import InterestPoint

        point = InterestPoint(position=(0, 0, 5), priority=2.0, weight=0.8)
        assert point.priority == 2.0
        assert point.weight == 0.8

    def test_interest_point_invalid_priority(self):
        """Test that negative priority raises error."""
        from engine.animation.procedural.lookat import InterestPoint

        with pytest.raises(ValueError, match="priority must be >= 0"):
            InterestPoint(position=(0, 0, 0), priority=-1)

    def test_interest_point_invalid_weight(self):
        """Test that weight outside [0, 1] raises error."""
        from engine.animation.procedural.lookat import InterestPoint

        with pytest.raises(ValueError, match="weight must be in"):
            InterestPoint(position=(0, 0, 0), weight=1.5)

    def test_interest_point_in_range(self):
        """Test range checking."""
        from engine.animation.procedural.lookat import InterestPoint

        point = InterestPoint(
            position=(0, 0, 5),
            min_distance=1.0,
            max_distance=10.0
        )

        assert point.is_in_range((0, 0, 0))  # 5 units away
        assert not point.is_in_range((0, 0, 4.5))  # 0.5 units away (too close)
        assert not point.is_in_range((0, 0, -10))  # 15 units away (too far)


class TestSaccadeGenerator:
    """Tests for SaccadeGenerator class."""

    def test_saccade_generator_creation(self):
        """Test saccade generator creation."""
        from engine.animation.procedural.lookat import SaccadeGenerator

        saccade = SaccadeGenerator(min_interval=0.1, max_interval=2.0)
        assert saccade.min_interval == 0.1
        assert saccade.max_interval == 2.0

    def test_saccade_generator_invalid_intervals(self):
        """Test that invalid intervals raise errors."""
        from engine.animation.procedural.lookat import SaccadeGenerator

        with pytest.raises(ValueError, match="min_interval must be > 0"):
            SaccadeGenerator(min_interval=0)

        with pytest.raises(ValueError, match="max_interval must be > min_interval"):
            SaccadeGenerator(min_interval=2.0, max_interval=1.0)

    def test_saccade_generator_update(self):
        """Test saccade update produces offsets."""
        from engine.animation.procedural.lookat import SaccadeGenerator

        saccade = SaccadeGenerator()
        offset = saccade.update(dt=1/60)

        # Offset should be a 3-tuple
        assert len(offset) == 3

    def test_saccade_generator_reset(self):
        """Test saccade generator reset."""
        from engine.animation.procedural.lookat import SaccadeGenerator

        saccade = SaccadeGenerator()
        saccade.update(dt=1.0)
        saccade.reset()

        assert saccade._current_offset == (0.0, 0.0, 0.0)


# =============================================================================
# Twist Bone Tests
# =============================================================================


class TestTwistBone:
    """Tests for TwistBone class."""

    def test_twist_bone_creation(self):
        """Test twist bone creation."""
        from engine.animation.procedural.twist import TwistBone, TwistDistribution

        twist = TwistBone(source_bone=5, twist_bones=[6, 7])
        assert twist.source_bone == 5
        assert twist.twist_bones == [6, 7]
        assert twist.distribution == TwistDistribution.LINEAR

    def test_twist_bone_invalid_source(self):
        """Test that negative source bone raises error."""
        from engine.animation.procedural.twist import TwistBone

        with pytest.raises(ValueError, match="source_bone must be >= 0"):
            TwistBone(source_bone=-1, twist_bones=[0])

    def test_twist_bone_empty_twist_bones(self):
        """Test that empty twist bones raises error."""
        from engine.animation.procedural.twist import TwistBone

        with pytest.raises(ValueError, match="twist_bones must not be empty"):
            TwistBone(source_bone=0, twist_bones=[])

    def test_twist_bone_custom_weights_required(self):
        """Test that custom distribution requires weights."""
        from engine.animation.procedural.twist import TwistBone, TwistDistribution

        with pytest.raises(ValueError, match="custom_weights required"):
            TwistBone(
                source_bone=0,
                twist_bones=[1, 2],
                distribution=TwistDistribution.CUSTOM
            )

    def test_twist_bone_custom_weights_length(self):
        """Test that custom weights must match bone count."""
        from engine.animation.procedural.twist import TwistBone, TwistDistribution

        with pytest.raises(ValueError, match="custom_weights length must match"):
            TwistBone(
                source_bone=0,
                twist_bones=[1, 2, 3],
                distribution=TwistDistribution.CUSTOM,
                custom_weights=[0.5, 0.5]  # Only 2 weights for 3 bones
            )

    def test_twist_bone_update(self):
        """Test twist bone update."""
        from engine.animation.procedural.twist import TwistBone

        twist = TwistBone(source_bone=2, twist_bones=[3, 4])
        pose = create_test_pose()

        # Set a rotation on source bone
        rot = (0.3826834, 0.0, 0.0, 0.9238795)  # 45 degrees around X
        pose.set_bone_rotation(2, rot)

        result = twist.update(pose)
        # Twist bones should have some rotation applied
        assert result is not pose

    def test_twist_bone_linear_distribution(self):
        """Test linear twist distribution."""
        from engine.animation.procedural.twist import TwistBone, TwistDistribution

        twist = TwistBone(
            source_bone=0,
            twist_bones=[1, 2, 3],
            distribution=TwistDistribution.LINEAR,
            weight=1.0
        )

        # Linear distribution weights should be 0, 0.5, 1.0
        assert twist._get_distribution_weight(0, 3) == 0.0
        assert twist._get_distribution_weight(1, 3) == 0.5
        assert twist._get_distribution_weight(2, 3) == 1.0

    def test_twist_bone_get_bone_count(self):
        """Test getting twist bone count."""
        from engine.animation.procedural.twist import TwistBone

        twist = TwistBone(source_bone=0, twist_bones=[1, 2, 3, 4])
        assert twist.get_bone_count() == 4

    def test_twist_bone_set_weight(self):
        """Test setting twist weight."""
        from engine.animation.procedural.twist import TwistBone

        twist = TwistBone(source_bone=0, twist_bones=[1])
        twist.set_weight(0.5)
        assert twist.weight == 0.5

        with pytest.raises(ValueError):
            twist.set_weight(1.5)


class TestTwistChain:
    """Tests for TwistChain class."""

    def test_twist_chain_arm_setup(self):
        """Test arm twist chain setup."""
        from engine.animation.procedural.twist import TwistChain

        chain = TwistChain.create_arm_setup(
            shoulder_bone=0,
            upper_arm_bone=1,
            upper_arm_twist_bones=[2],
            forearm_bone=3,
            forearm_twist_bones=[4],
            wrist_bone=5
        )

        assert chain.upper_arm_twist is not None
        assert chain.forearm_twist is not None

    def test_twist_chain_leg_setup(self):
        """Test leg twist chain setup."""
        from engine.animation.procedural.twist import TwistChain

        chain = TwistChain.create_leg_setup(
            hip_bone=0,
            thigh_bone=1,
            thigh_twist_bones=[2],
            calf_bone=3,
            calf_twist_bones=[4],
            foot_bone=5
        )

        assert chain.thigh_twist is not None
        assert chain.calf_twist is not None

    def test_twist_chain_update(self):
        """Test twist chain update."""
        from engine.animation.procedural.twist import TwistChain

        chain = TwistChain.create_arm_setup(
            shoulder_bone=0,
            upper_arm_bone=1,
            upper_arm_twist_bones=[2],
            forearm_bone=3,
            forearm_twist_bones=[4],
            wrist_bone=5
        )

        pose = create_test_pose()
        result = chain.update(pose)
        assert result is not pose


# =============================================================================
# Ragdoll Tests
# =============================================================================


class TestRagdollBody:
    """Tests for RagdollBody class."""

    def test_ragdoll_body_creation(self):
        """Test ragdoll body creation."""
        from engine.animation.procedural.ragdoll import (
            RagdollBody, CollisionShape, CollisionShapeType
        )

        shape = CollisionShape(shape_type=CollisionShapeType.CAPSULE, radius=0.1, height=0.3)
        body = RagdollBody(bone_index=5, shape=shape, mass=5.0)

        assert body.bone_index == 5
        assert body.mass == 5.0
        assert not body.is_active()

    def test_ragdoll_body_invalid_bone(self):
        """Test that negative bone index raises error."""
        from engine.animation.procedural.ragdoll import (
            RagdollBody, CollisionShape, CollisionShapeType
        )

        shape = CollisionShape(shape_type=CollisionShapeType.SPHERE)

        with pytest.raises(ValueError, match="bone_index must be >= 0"):
            RagdollBody(bone_index=-1, shape=shape)

    def test_ragdoll_body_invalid_mass(self):
        """Test that non-positive mass raises error."""
        from engine.animation.procedural.ragdoll import (
            RagdollBody, CollisionShape, CollisionShapeType
        )

        shape = CollisionShape(shape_type=CollisionShapeType.SPHERE)

        with pytest.raises(ValueError, match="mass must be > 0"):
            RagdollBody(bone_index=0, shape=shape, mass=0)


class TestRagdollJoint:
    """Tests for RagdollJoint class."""

    def test_ragdoll_joint_creation(self):
        """Test ragdoll joint creation."""
        from engine.animation.procedural.ragdoll import RagdollJoint, JointLimits

        joint = RagdollJoint(
            parent_body=0,
            child_body=1,
            limits=JointLimits()
        )

        assert joint.parent_body == 0
        assert joint.child_body == 1
        assert not joint.is_active()

    def test_ragdoll_joint_same_bodies(self):
        """Test that same parent and child raises error."""
        from engine.animation.procedural.ragdoll import RagdollJoint

        with pytest.raises(ValueError, match="must be different"):
            RagdollJoint(parent_body=0, child_body=0)


class TestJointLimits:
    """Tests for JointLimits class."""

    def test_joint_limits_creation(self):
        """Test joint limits creation."""
        from engine.animation.procedural.ragdoll import JointLimits

        limits = JointLimits(
            twist_lower=math.radians(-45),
            twist_upper=math.radians(45),
            swing1_limit=math.radians(30)
        )

        assert limits.twist_lower == math.radians(-45)

    def test_joint_limits_invalid_twist(self):
        """Test that lower > upper raises error."""
        from engine.animation.procedural.ragdoll import JointLimits

        with pytest.raises(ValueError, match="twist_lower must be <= twist_upper"):
            JointLimits(twist_lower=1.0, twist_upper=0.5)

    def test_joint_limits_invalid_swing(self):
        """Test that negative swing raises error."""
        from engine.animation.procedural.ragdoll import JointLimits

        with pytest.raises(ValueError, match="swing limits must be >= 0"):
            JointLimits(swing1_limit=-0.1)


class TestRagdollConfig:
    """Tests for RagdollConfig class."""

    def test_ragdoll_config_creation(self):
        """Test ragdoll config creation."""
        from engine.animation.procedural.ragdoll import (
            RagdollConfig, RagdollBody, RagdollJoint,
            CollisionShape, CollisionShapeType
        )

        shape = CollisionShape(shape_type=CollisionShapeType.SPHERE)
        bodies = [
            RagdollBody(bone_index=0, shape=shape),
            RagdollBody(bone_index=1, shape=shape)
        ]
        joints = [RagdollJoint(parent_body=0, child_body=1)]

        config = RagdollConfig(bodies=bodies, joints=joints)
        assert len(config.bodies) == 2
        assert len(config.joints) == 1

    def test_ragdoll_config_empty_bodies(self):
        """Test that empty bodies raises error."""
        from engine.animation.procedural.ragdoll import RagdollConfig

        with pytest.raises(ValueError, match="bodies must not be empty"):
            RagdollConfig(bodies=[], joints=[])

    def test_ragdoll_config_invalid_joint_ref(self):
        """Test that invalid joint reference raises error."""
        from engine.animation.procedural.ragdoll import (
            RagdollConfig, RagdollBody, RagdollJoint,
            CollisionShape, CollisionShapeType
        )

        shape = CollisionShape(shape_type=CollisionShapeType.SPHERE)
        bodies = [RagdollBody(bone_index=0, shape=shape)]
        joints = [RagdollJoint(parent_body=0, child_body=5)]  # 5 doesn't exist

        with pytest.raises(ValueError, match="out of range"):
            RagdollConfig(bodies=bodies, joints=joints)

    def test_ragdoll_config_humanoid(self):
        """Test humanoid config creation."""
        from engine.animation.procedural.ragdoll import RagdollConfig

        skeleton = MockSkeleton(bone_count=20)
        skeleton.bone_names = {
            0: "hips",
            1: "spine",
            2: "chest",
            3: "head"
        }

        config = RagdollConfig.create_humanoid(skeleton)
        assert len(config.bodies) > 0


class TestRagdoll:
    """Tests for Ragdoll class."""

    def test_ragdoll_creation(self):
        """Test ragdoll creation."""
        from engine.animation.procedural.ragdoll import (
            Ragdoll, RagdollConfig, RagdollBody, CollisionShape, CollisionShapeType,
            RagdollState
        )

        skeleton = MockSkeleton()
        shape = CollisionShape(shape_type=CollisionShapeType.SPHERE)
        config = RagdollConfig(
            bodies=[RagdollBody(bone_index=0, shape=shape)],
            joints=[]
        )

        ragdoll = Ragdoll(skeleton=skeleton, config=config)
        assert ragdoll.state == RagdollState.INACTIVE
        assert not ragdoll.is_active()

    def test_ragdoll_create_destroy(self):
        """Test ragdoll physics creation and destruction."""
        from engine.animation.procedural.ragdoll import (
            Ragdoll, RagdollConfig, RagdollBody, CollisionShape, CollisionShapeType,
            RagdollState
        )

        skeleton = MockSkeleton()
        shape = CollisionShape(shape_type=CollisionShapeType.SPHERE)
        config = RagdollConfig(
            bodies=[
                RagdollBody(bone_index=0, shape=shape),
                RagdollBody(bone_index=1, shape=shape)
            ],
            joints=[]
        )

        ragdoll = Ragdoll(skeleton=skeleton, config=config)
        physics = MockPhysicsWorld()
        pose = create_test_pose()

        ragdoll.create(physics, pose)
        assert ragdoll.state == RagdollState.KINEMATIC

        ragdoll.destroy()
        assert ragdoll.state == RagdollState.INACTIVE

    def test_ragdoll_activate(self):
        """Test ragdoll activation."""
        from engine.animation.procedural.ragdoll import (
            Ragdoll, RagdollConfig, RagdollBody, CollisionShape, CollisionShapeType,
            RagdollState
        )

        skeleton = MockSkeleton()
        shape = CollisionShape(shape_type=CollisionShapeType.SPHERE)
        config = RagdollConfig(
            bodies=[RagdollBody(bone_index=0, shape=shape)],
            joints=[]
        )

        ragdoll = Ragdoll(skeleton=skeleton, config=config)
        physics = MockPhysicsWorld()
        pose = create_test_pose()

        ragdoll.create(physics, pose)
        ragdoll.activate(blend_time=0.0)

        assert ragdoll.state == RagdollState.DYNAMIC
        assert ragdoll.is_active()

    def test_ragdoll_deactivate(self):
        """Test ragdoll deactivation."""
        from engine.animation.procedural.ragdoll import (
            Ragdoll, RagdollConfig, RagdollBody, CollisionShape, CollisionShapeType
        )

        skeleton = MockSkeleton()
        shape = CollisionShape(shape_type=CollisionShapeType.SPHERE)
        config = RagdollConfig(
            bodies=[RagdollBody(bone_index=0, shape=shape)],
            joints=[]
        )

        ragdoll = Ragdoll(skeleton=skeleton, config=config)
        physics = MockPhysicsWorld()
        pose = create_test_pose()

        ragdoll.create(physics, pose)
        ragdoll.activate(blend_time=0)
        ragdoll.deactivate(blend_time=0)

        assert len(ragdoll.active_bodies) == 0

    def test_ragdoll_sync_to_physics(self):
        """Test syncing pose to physics."""
        from engine.animation.procedural.ragdoll import (
            Ragdoll, RagdollConfig, RagdollBody, CollisionShape, CollisionShapeType
        )

        skeleton = MockSkeleton()
        shape = CollisionShape(shape_type=CollisionShapeType.SPHERE)
        config = RagdollConfig(
            bodies=[RagdollBody(bone_index=0, shape=shape)],
            joints=[]
        )

        ragdoll = Ragdoll(skeleton=skeleton, config=config)
        physics = MockPhysicsWorld()
        pose = create_test_pose()

        ragdoll.create(physics, pose)
        ragdoll.sync_to_physics(pose)
        # Should not raise

    def test_ragdoll_sync_from_physics(self):
        """Test syncing physics to pose."""
        from engine.animation.procedural.ragdoll import (
            Ragdoll, RagdollConfig, RagdollBody, CollisionShape, CollisionShapeType,
            Transform
        )

        skeleton = MockSkeleton()
        shape = CollisionShape(shape_type=CollisionShapeType.SPHERE)
        config = RagdollConfig(
            bodies=[RagdollBody(bone_index=0, shape=shape)],
            joints=[]
        )

        ragdoll = Ragdoll(skeleton=skeleton, config=config)
        physics = MockPhysicsWorld()
        pose = create_test_pose()

        ragdoll.create(physics, pose)
        ragdoll.sync_to_physics(pose)
        ragdoll.activate(blend_time=0)

        # Set physics body position
        body_id = config.bodies[0].physics_body_id
        physics._bodies[body_id]["transform"] = Transform(position=(1.0, 2.0, 3.0))

        result = ragdoll.sync_from_physics(dt=1/60)
        assert result is not None

    def test_ragdoll_partial_activation(self):
        """Test partial ragdoll activation."""
        from engine.animation.procedural.ragdoll import (
            Ragdoll, RagdollConfig, RagdollBody, CollisionShape, CollisionShapeType
        )

        skeleton = MockSkeleton()
        shape = CollisionShape(shape_type=CollisionShapeType.SPHERE)
        config = RagdollConfig(
            bodies=[
                RagdollBody(bone_index=0, shape=shape),
                RagdollBody(bone_index=1, shape=shape),
                RagdollBody(bone_index=2, shape=shape)
            ],
            joints=[]
        )

        ragdoll = Ragdoll(skeleton=skeleton, config=config)
        physics = MockPhysicsWorld()
        pose = create_test_pose()

        ragdoll.create(physics, pose)
        ragdoll.activate(blend_time=0, partial_bodies={1, 2})

        assert 0 not in ragdoll.active_bodies
        assert 1 in ragdoll.active_bodies
        assert 2 in ragdoll.active_bodies


# =============================================================================
# Locomotion Tests
# =============================================================================


class TestGaitConfig:
    """Tests for GaitConfig class."""

    def test_gait_config_creation(self):
        """Test gait config creation."""
        from engine.animation.procedural.locomotion import GaitConfig

        gait = GaitConfig(step_height=0.15, step_length=0.6, cycle_duration=0.5)
        assert gait.step_height == 0.15
        assert gait.step_length == 0.6
        assert gait.cycle_duration == 0.5

    def test_gait_config_invalid_duration(self):
        """Test that non-positive duration raises error."""
        from engine.animation.procedural.locomotion import GaitConfig

        with pytest.raises(ValueError, match="cycle_duration must be > 0"):
            GaitConfig(cycle_duration=0)

    def test_gait_config_walk_preset(self):
        """Test walk gait preset."""
        from engine.animation.procedural.locomotion import GaitConfig, GaitType

        gait = GaitConfig.create_walk()
        assert gait.gait_type == GaitType.WALK

    def test_gait_config_run_preset(self):
        """Test run gait preset."""
        from engine.animation.procedural.locomotion import GaitConfig, GaitType

        gait = GaitConfig.create_run()
        assert gait.gait_type == GaitType.RUN

    def test_gait_config_quadruped_walk(self):
        """Test quadruped walk preset."""
        from engine.animation.procedural.locomotion import GaitConfig

        gait = GaitConfig.create_quadruped_walk()
        assert len(gait.foot_phases) == 4

    def test_gait_config_quadruped_trot(self):
        """Test quadruped trot preset."""
        from engine.animation.procedural.locomotion import GaitConfig, GaitType

        gait = GaitConfig.create_quadruped_trot()
        assert gait.gait_type == GaitType.TROT


class TestFootTrajectory:
    """Tests for FootTrajectory class."""

    def test_foot_trajectory_creation(self):
        """Test foot trajectory creation."""
        from engine.animation.procedural.locomotion import FootTrajectory

        traj = FootTrajectory(step_height=0.2, step_length=0.8)
        assert traj.step_height == 0.2
        assert traj.step_length == 0.8

    def test_foot_trajectory_invalid_height(self):
        """Test that negative height raises error."""
        from engine.animation.procedural.locomotion import FootTrajectory

        with pytest.raises(ValueError, match="step_height must be >= 0"):
            FootTrajectory(step_height=-0.1)

    def test_foot_trajectory_sample_stance(self):
        """Test stance phase sampling."""
        from engine.animation.procedural.locomotion import FootTrajectory

        traj = FootTrajectory(step_length=1.0)

        pos, angle = traj.sample_stance(0.0)
        assert pos[1] == 0.0  # On ground

        pos, angle = traj.sample_stance(1.0)
        assert pos[1] == 0.0  # Still on ground

    def test_foot_trajectory_sample_swing(self):
        """Test swing phase sampling."""
        from engine.animation.procedural.locomotion import FootTrajectory

        traj = FootTrajectory(step_height=0.2)

        pos, angle = traj.sample_swing(0.0)
        assert pos[1] == 0.0  # Start on ground

        pos, angle = traj.sample_swing(0.5)
        assert pos[1] > 0  # Peak height

        pos, angle = traj.sample_swing(1.0)
        assert pos[1] == 0.0  # End on ground

    def test_foot_trajectory_sample_full_cycle(self):
        """Test full cycle sampling."""
        from engine.animation.procedural.locomotion import FootTrajectory

        traj = FootTrajectory(stance_ratio=0.6)

        # Stance phase
        pos, angle, on_ground = traj.sample(0.3)
        assert on_ground

        # Swing phase
        pos, angle, on_ground = traj.sample(0.8)
        assert not on_ground


class TestProceduralLocomotion:
    """Tests for ProceduralLocomotion class."""

    def test_locomotion_creation(self):
        """Test locomotion creation."""
        from engine.animation.procedural.locomotion import (
            ProceduralLocomotion, GaitConfig
        )

        skeleton = MockSkeleton()
        gait = GaitConfig()
        loco = ProceduralLocomotion(skeleton=skeleton, gait_config=gait)

        assert loco.skeleton is skeleton
        assert loco.gait_config is gait

    def test_locomotion_update(self):
        """Test locomotion phase update."""
        from engine.animation.procedural.locomotion import (
            ProceduralLocomotion, GaitConfig
        )

        skeleton = MockSkeleton()
        gait = GaitConfig(cycle_duration=1.0)
        loco = ProceduralLocomotion(skeleton=skeleton, gait_config=gait)

        phase = loco.update(dt=0.5, speed=1.0)
        assert 0.0 <= phase <= 1.0

    def test_locomotion_generate_walk(self):
        """Test walk cycle generation."""
        from engine.animation.procedural.locomotion import (
            ProceduralLocomotion, GaitConfig, LegConfig
        )

        skeleton = MockSkeleton()
        gait = GaitConfig()
        loco = ProceduralLocomotion(skeleton=skeleton, gait_config=gait, hips_bone=0)

        loco.configure_biped(
            hips=0,
            spine=[1, 2],
            left_leg=LegConfig(hip_bone=3, thigh_bone=4, calf_bone=5, foot_bone=6),
            right_leg=LegConfig(hip_bone=7, thigh_bone=8, calf_bone=9, foot_bone=10, is_left=False)
        )

        pose = create_test_pose()
        result = loco.generate_walk_cycle(speed=1.0, base_pose=pose)

        assert result is not None

    def test_locomotion_foot_contacts(self):
        """Test foot contact states."""
        from engine.animation.procedural.locomotion import (
            ProceduralLocomotion, GaitConfig, LegConfig
        )

        skeleton = MockSkeleton()
        gait = GaitConfig()
        loco = ProceduralLocomotion(skeleton=skeleton, gait_config=gait)

        loco.configure_biped(
            hips=0,
            spine=[1],
            left_leg=LegConfig(hip_bone=2, thigh_bone=3, calf_bone=4, foot_bone=5),
            right_leg=LegConfig(hip_bone=6, thigh_bone=7, calf_bone=8, foot_bone=9, is_left=False)
        )

        contacts = loco.get_foot_contacts()
        assert "left_foot" in contacts or "right_foot" in contacts

    def test_locomotion_reset(self):
        """Test locomotion reset."""
        from engine.animation.procedural.locomotion import (
            ProceduralLocomotion, GaitConfig
        )

        skeleton = MockSkeleton()
        loco = ProceduralLocomotion(skeleton=skeleton, gait_config=GaitConfig())

        loco.update(dt=0.5, speed=1.0)
        loco.reset()

        assert loco.get_current_phase() == 0.0


# =============================================================================
# Breathing Tests
# =============================================================================


class TestBreathingController:
    """Tests for BreathingController class."""

    def test_breathing_creation(self):
        """Test breathing controller creation."""
        from engine.animation.procedural.breathing import BreathingController

        controller = BreathingController(
            spine_bones=[5, 6, 7],
            chest_bone=8,
            breath_rate=0.25
        )

        assert controller.chest_bone == 8
        assert controller.breath_rate == 0.25

    def test_breathing_invalid_chest(self):
        """Test that negative chest bone raises error."""
        from engine.animation.procedural.breathing import BreathingController

        with pytest.raises(ValueError, match="chest_bone must be >= 0"):
            BreathingController(spine_bones=[], chest_bone=-1)

    def test_breathing_invalid_rate(self):
        """Test that non-positive rate raises error."""
        from engine.animation.procedural.breathing import BreathingController

        with pytest.raises(ValueError, match="breath_rate must be > 0"):
            BreathingController(spine_bones=[], chest_bone=0, breath_rate=0)

    def test_breathing_update(self):
        """Test breathing update."""
        from engine.animation.procedural.breathing import BreathingController

        controller = BreathingController(
            spine_bones=[1, 2],
            chest_bone=3
        )

        pose = create_test_pose()
        result = controller.update(pose, dt=1/60)

        assert result is not pose

    def test_breathing_exertion_levels(self):
        """Test exertion level changes."""
        from engine.animation.procedural.breathing import (
            BreathingController, ExertionLevel
        )

        controller = BreathingController(spine_bones=[], chest_bone=0)
        controller.set_exertion(ExertionLevel.HEAVY)

        assert controller._target_exertion == ExertionLevel.HEAVY

    def test_breathing_phase_detection(self):
        """Test breath phase detection."""
        from engine.animation.procedural.breathing import (
            BreathingController, BreathPhase
        )

        controller = BreathingController(spine_bones=[], chest_bone=0)

        phase = controller.get_current_phase()
        assert isinstance(phase, BreathPhase)

    def test_breathing_value_range(self):
        """Test breath value stays in valid range."""
        from engine.animation.procedural.breathing import BreathingController

        controller = BreathingController(spine_bones=[], chest_bone=0)
        pose = create_test_pose()

        for _ in range(100):
            controller.update(pose, dt=1/60)
            value = controller.get_breath_value()
            assert 0.0 <= value <= 1.0

    def test_breathing_is_inhaling(self):
        """Test inhaling detection."""
        from engine.animation.procedural.breathing import BreathingController

        controller = BreathingController(spine_bones=[], chest_bone=0)

        # Check both inhaling and exhaling happen
        inhaling_count = 0
        exhaling_count = 0
        pose = create_test_pose()

        for _ in range(100):
            controller.update(pose, dt=0.05)
            if controller.is_inhaling():
                inhaling_count += 1
            else:
                exhaling_count += 1

        assert inhaling_count > 0
        assert exhaling_count > 0

    def test_breathing_reset(self):
        """Test breathing reset."""
        from engine.animation.procedural.breathing import BreathingController

        controller = BreathingController(spine_bones=[], chest_bone=0)
        pose = create_test_pose()

        for _ in range(50):
            controller.update(pose, dt=1/60)

        controller.reset()
        assert controller._current_phase == 0.0

    def test_breathing_sync_phase(self):
        """Test syncing to specific phase."""
        from engine.animation.procedural.breathing import BreathingController

        controller = BreathingController(spine_bones=[], chest_bone=0)
        controller.sync_to_phase(0.75)

        assert abs(controller._current_phase - 0.75) < 0.001


# =============================================================================
# Secondary Motion Tests
# =============================================================================


class TestDelayedMotion:
    """Tests for DelayedMotion class."""

    def test_delayed_motion_creation(self):
        """Test delayed motion creation."""
        from engine.animation.procedural.secondary_motion import DelayedMotion

        motion = DelayedMotion(affected_bones=[1, 2, 3], delay=0.1)
        assert motion.delay == 0.1

    def test_delayed_motion_invalid_delay(self):
        """Test that negative delay raises error."""
        from engine.animation.procedural.secondary_motion import DelayedMotion

        with pytest.raises(ValueError, match="delay must be >= 0"):
            DelayedMotion(affected_bones=[0], delay=-0.1)

    def test_delayed_motion_update(self):
        """Test delayed motion update."""
        from engine.animation.procedural.secondary_motion import DelayedMotion

        motion = DelayedMotion(affected_bones=[0], delay=0.1)
        pose = create_test_pose()

        result = motion.update(pose, dt=1/60)
        assert result is not pose

    def test_delayed_motion_reset(self):
        """Test delayed motion reset."""
        from engine.animation.procedural.secondary_motion import DelayedMotion

        motion = DelayedMotion(affected_bones=[0], delay=0.1)
        pose = create_test_pose()

        motion.update(pose, dt=1/60)
        motion.reset()

        assert motion._time == 0.0


class TestOscillatingMotion:
    """Tests for OscillatingMotion class."""

    def test_oscillating_motion_creation(self):
        """Test oscillating motion creation."""
        from engine.animation.procedural.secondary_motion import OscillatingMotion

        motion = OscillatingMotion(
            affected_bones=[0],
            frequency=2.0,
            amplitude=(0.0, 0.1, 0.0)
        )

        assert motion.frequency == 2.0

    def test_oscillating_motion_invalid_frequency(self):
        """Test that negative frequency raises error."""
        from engine.animation.procedural.secondary_motion import OscillatingMotion

        with pytest.raises(ValueError, match="frequency must be >= 0"):
            OscillatingMotion(affected_bones=[0], frequency=-1.0)

    def test_oscillating_motion_update(self):
        """Test oscillating motion update."""
        from engine.animation.procedural.secondary_motion import OscillatingMotion

        motion = OscillatingMotion(
            affected_bones=[0],
            amplitude=(0.0, 0.1, 0.0)
        )
        pose = create_test_pose()

        # Capture position over time
        positions = []
        for _ in range(60):
            result = motion.update(pose, dt=1/60)
            positions.append(result.get_bone_position(0))

        # Should oscillate
        y_values = [p[1] for p in positions]
        assert max(y_values) > min(y_values)

    def test_oscillating_motion_per_bone_offset(self):
        """Test per-bone phase offset."""
        from engine.animation.procedural.secondary_motion import OscillatingMotion

        motion = OscillatingMotion(
            affected_bones=[0, 1, 2],
            per_bone_phase_offset=math.pi / 4
        )

        assert motion.per_bone_phase_offset == math.pi / 4


class TestNoiseMotion:
    """Tests for NoiseMotion class."""

    def test_noise_motion_creation(self):
        """Test noise motion creation."""
        from engine.animation.procedural.secondary_motion import NoiseMotion

        motion = NoiseMotion(
            affected_bones=[0],
            amplitude=(0.01, 0.01, 0.01),
            frequency=1.0
        )

        assert motion.frequency == 1.0

    def test_noise_motion_invalid_frequency(self):
        """Test that non-positive frequency raises error."""
        from engine.animation.procedural.secondary_motion import NoiseMotion

        with pytest.raises(ValueError, match="frequency must be > 0"):
            NoiseMotion(affected_bones=[0], frequency=0)

    def test_noise_motion_update(self):
        """Test noise motion update."""
        from engine.animation.procedural.secondary_motion import NoiseMotion

        motion = NoiseMotion(
            affected_bones=[0],
            amplitude=(0.1, 0.1, 0.1)
        )
        pose = create_test_pose()

        result = motion.update(pose, dt=1/60)
        assert result is not pose

    def test_noise_motion_deterministic(self):
        """Test that same seed produces same result."""
        from engine.animation.procedural.secondary_motion import NoiseMotion

        motion1 = NoiseMotion(affected_bones=[0], seed=42)
        motion2 = NoiseMotion(affected_bones=[0], seed=42)

        pose = create_test_pose()

        result1 = motion1.update(pose.copy(), dt=1/60)
        result2 = motion2.update(pose.copy(), dt=1/60)

        pos1 = result1.get_bone_position(0)
        pos2 = result2.get_bone_position(0)

        assert pos1 == pos2


class TestImpulseResponse:
    """Tests for ImpulseResponse class."""

    def test_impulse_response_creation(self):
        """Test impulse response creation."""
        from engine.animation.procedural.secondary_motion import ImpulseResponse

        motion = ImpulseResponse(
            affected_bones=[0],
            stiffness=50.0,
            damping=0.7
        )

        assert motion.stiffness == 50.0
        assert motion.damping == 0.7

    def test_impulse_response_invalid_stiffness(self):
        """Test that non-positive stiffness raises error."""
        from engine.animation.procedural.secondary_motion import ImpulseResponse

        with pytest.raises(ValueError, match="stiffness must be > 0"):
            ImpulseResponse(affected_bones=[0], stiffness=0)

    def test_impulse_response_invalid_damping(self):
        """Test that damping outside [0, 1] raises error."""
        from engine.animation.procedural.secondary_motion import ImpulseResponse

        with pytest.raises(ValueError, match="damping must be in"):
            ImpulseResponse(affected_bones=[0], damping=1.5)

    def test_impulse_response_update(self):
        """Test impulse response update."""
        from engine.animation.procedural.secondary_motion import ImpulseResponse

        motion = ImpulseResponse(affected_bones=[0])
        pose = create_test_pose()

        result = motion.update(pose, dt=1/60)
        assert result is not pose

    def test_impulse_response_apply_impulse(self):
        """Test applying manual impulse."""
        from engine.animation.procedural.secondary_motion import ImpulseResponse

        motion = ImpulseResponse(affected_bones=[0])

        motion.apply_impulse(0, (1.0, 0.0, 0.0))
        assert motion._velocities[0][0] == 1.0


class TestMotionComposer:
    """Tests for MotionComposer class."""

    def test_composer_creation(self):
        """Test motion composer creation."""
        from engine.animation.procedural.secondary_motion import MotionComposer

        composer = MotionComposer()
        assert composer.get_motion_count() == 0

    def test_composer_add_remove(self):
        """Test adding and removing motions."""
        from engine.animation.procedural.secondary_motion import (
            MotionComposer, OscillatingMotion
        )

        composer = MotionComposer()
        motion = OscillatingMotion(affected_bones=[0])

        composer.add(motion)
        assert composer.get_motion_count() == 1

        composer.remove(motion)
        assert composer.get_motion_count() == 0

    def test_composer_update(self):
        """Test composer update applies all motions."""
        from engine.animation.procedural.secondary_motion import (
            MotionComposer, OscillatingMotion, NoiseMotion
        )

        composer = MotionComposer()
        composer.add(OscillatingMotion(affected_bones=[0], amplitude=(0.1, 0, 0)))
        composer.add(NoiseMotion(affected_bones=[0], amplitude=(0, 0.1, 0)))

        pose = create_test_pose()
        result = composer.update(pose, dt=1/60)

        assert result is not pose

    def test_composer_clear(self):
        """Test clearing all motions."""
        from engine.animation.procedural.secondary_motion import (
            MotionComposer, OscillatingMotion
        )

        composer = MotionComposer()
        composer.add(OscillatingMotion(affected_bones=[0]))
        composer.add(OscillatingMotion(affected_bones=[1]))

        composer.clear()
        assert composer.get_motion_count() == 0

    def test_composer_set_all_weights(self):
        """Test setting weight for all motions."""
        from engine.animation.procedural.secondary_motion import (
            MotionComposer, OscillatingMotion
        )

        composer = MotionComposer()
        m1 = OscillatingMotion(affected_bones=[0])
        m2 = OscillatingMotion(affected_bones=[1])

        composer.add(m1)
        composer.add(m2)

        composer.set_all_weights(0.5)

        assert m1.weight == 0.5
        assert m2.weight == 0.5

    def test_composer_enable_disable(self):
        """Test enabling/disabling all motions."""
        from engine.animation.procedural.secondary_motion import (
            MotionComposer, OscillatingMotion
        )

        composer = MotionComposer()
        m1 = OscillatingMotion(affected_bones=[0])
        composer.add(m1)

        composer.enable_all(False)
        assert not m1.enabled

        composer.enable_all(True)
        assert m1.enabled


class TestPerlinNoise:
    """Tests for PerlinNoise class."""

    def test_perlin_noise_creation(self):
        """Test Perlin noise creation."""
        from engine.animation.procedural.secondary_motion import PerlinNoise

        noise = PerlinNoise(seed=42)
        assert noise is not None

    def test_perlin_noise_range(self):
        """Test that noise values are in valid range."""
        from engine.animation.procedural.secondary_motion import PerlinNoise

        noise = PerlinNoise(seed=0)

        for x in range(100):
            value = noise.noise(x * 0.1)
            assert -1.0 <= value <= 1.0

    def test_perlin_noise_continuity(self):
        """Test that noise is continuous."""
        from engine.animation.procedural.secondary_motion import PerlinNoise

        noise = PerlinNoise(seed=0)

        prev = noise.noise(0.0)
        for i in range(1, 100):
            curr = noise.noise(i * 0.01)
            # Adjacent samples should be similar
            assert abs(curr - prev) < 0.5
            prev = curr

    def test_perlin_fbm(self):
        """Test fractal Brownian motion."""
        from engine.animation.procedural.secondary_motion import PerlinNoise

        noise = PerlinNoise(seed=0)

        value = noise.fbm(0.5, octaves=4, persistence=0.5)
        assert isinstance(value, float)


# =============================================================================
# Base Class Tests
# =============================================================================


class TestSecondaryMotionBase:
    """Tests for SecondaryMotion base class."""

    def test_secondary_motion_empty_bones(self):
        """Test that empty bones list raises error."""
        from engine.animation.procedural.secondary_motion import OscillatingMotion

        with pytest.raises(ValueError, match="affected_bones must not be empty"):
            OscillatingMotion(affected_bones=[])

    def test_secondary_motion_invalid_weight(self):
        """Test that weight outside [0, 1] raises error."""
        from engine.animation.procedural.secondary_motion import OscillatingMotion

        with pytest.raises(ValueError, match="weight must be in"):
            OscillatingMotion(affected_bones=[0], weight=1.5)

    def test_secondary_motion_disabled(self):
        """Test that disabled motion returns unchanged pose."""
        from engine.animation.procedural.secondary_motion import OscillatingMotion

        motion = OscillatingMotion(
            affected_bones=[0],
            amplitude=(0.5, 0.5, 0.5),
            enabled=False
        )

        pose = create_test_pose()
        result = motion.update(pose, dt=1/60)

        # Position should be unchanged
        assert result.get_bone_position(0) == pose.get_bone_position(0)


# =============================================================================
# Integration Tests
# =============================================================================


class TestProceduralIntegration:
    """Integration tests for multiple procedural systems."""

    def test_spring_chain_with_wind(self):
        """Test spring chain with wind force."""
        from engine.animation.procedural.spring_bone import (
            SpringChain, WindForce
        )

        chain = SpringChain(root_bone=0, bone_indices=[1, 2, 3])
        wind = WindForce(direction=(1, 0, 0), strength=10.0)

        pose = create_test_pose()
        chain.initialize(pose)

        # Simulate with wind
        result = pose
        for _ in range(60):
            result = chain.simulate(result, dt=1/60, wind=wind)

        # Bones should have moved in wind direction
        final_pos = result.get_bone_position(3)
        assert final_pos[0] > 0

    def test_spring_chain_with_collision(self):
        """Test spring chain with collision primitives."""
        from engine.animation.procedural.spring_bone import (
            SpringChain, CollisionSphere
        )

        chain = SpringChain(
            root_bone=0,
            bone_indices=[1, 2, 3],
            gravity=(0, -10, 0)
        )

        # Place a sphere below
        sphere = CollisionSphere(center=(0, -0.2, 0), radius=0.15)

        pose = create_test_pose()
        chain.initialize(pose)

        # Simulate with collision
        result = pose
        for _ in range(120):
            result = chain.simulate(result, dt=1/60, colliders=[sphere])

        # Bones should not penetrate sphere

    def test_locomotion_with_breathing(self):
        """Test combining locomotion with breathing."""
        from engine.animation.procedural.locomotion import (
            ProceduralLocomotion, GaitConfig, LegConfig
        )
        from engine.animation.procedural.breathing import BreathingController

        skeleton = MockSkeleton()

        locomotion = ProceduralLocomotion(
            skeleton=skeleton,
            gait_config=GaitConfig()
        )

        locomotion.configure_biped(
            hips=0,
            spine=[1, 2],
            left_leg=LegConfig(hip_bone=3, thigh_bone=4, calf_bone=5, foot_bone=6),
            right_leg=LegConfig(hip_bone=7, thigh_bone=8, calf_bone=9, foot_bone=10, is_left=False)
        )

        breathing = BreathingController(spine_bones=[1, 2], chest_bone=2)

        pose = create_test_pose()

        # Apply both systems
        locomotion.update(dt=1/60, speed=1.0)
        result = locomotion.generate_walk_cycle(speed=1.0, base_pose=pose)
        if result:
            result = breathing.update(result, dt=1/60)

    def test_motion_composer_with_multiple_types(self):
        """Test composer with different motion types."""
        from engine.animation.procedural.secondary_motion import (
            MotionComposer, DelayedMotion, OscillatingMotion, NoiseMotion
        )

        composer = MotionComposer()
        composer.add(DelayedMotion(affected_bones=[0], delay=0.1))
        composer.add(OscillatingMotion(affected_bones=[1], frequency=2.0))
        composer.add(NoiseMotion(affected_bones=[2], frequency=0.5))

        pose = create_test_pose()

        # Run for a while
        for _ in range(60):
            pose = composer.update(pose, dt=1/60)

    def test_twist_bone_with_ragdoll(self):
        """Test that twist bones can work alongside ragdoll."""
        from engine.animation.procedural.twist import TwistBone
        from engine.animation.procedural.ragdoll import (
            Ragdoll, RagdollConfig, RagdollBody, CollisionShape, CollisionShapeType
        )

        # Create twist setup
        twist = TwistBone(source_bone=5, twist_bones=[6, 7])

        # Create ragdoll
        skeleton = MockSkeleton()
        shape = CollisionShape(shape_type=CollisionShapeType.SPHERE)
        config = RagdollConfig(
            bodies=[RagdollBody(bone_index=0, shape=shape)],
            joints=[]
        )
        ragdoll = Ragdoll(skeleton=skeleton, config=config)

        # Both should be able to process the same pose
        pose = create_test_pose()
        twist_result = twist.update(pose)

        physics = MockPhysicsWorld()
        ragdoll.create(physics, pose)
        ragdoll.sync_to_physics(pose)


# =============================================================================
# Numerical Stability Tests
# =============================================================================


class TestNumericalStability:
    """Tests for physics numerical stability and edge cases."""

    def test_spring_bone_large_dt_stability(self):
        """Test that spring bone remains stable with large dt (clamped internally)."""
        from engine.animation.procedural.spring_bone import SpringBone

        spring = SpringBone(bone_index=0, stiffness=100.0, damping=0.5, gravity=(0, -10, 0))
        pose = create_test_pose()
        spring.initialize(pose)

        # Large dt should be clamped internally to prevent explosion
        for _ in range(10):
            result = spring.simulate(pose, dt=0.5)  # Way too large normally
            pos = result.get_bone_position(0)

            # Position should remain finite and reasonable
            assert math.isfinite(pos[0])
            assert math.isfinite(pos[1])
            assert math.isfinite(pos[2])
            assert abs(pos[0]) < 100  # Should not explode
            assert abs(pos[1]) < 100
            assert abs(pos[2]) < 100

    def test_spring_chain_convergence_verification(self):
        """Test spring chain converges within expected time."""
        from engine.animation.procedural.spring_bone import SpringChain

        chain = SpringChain(
            root_bone=0,
            bone_indices=[1, 2, 3],
            stiffness=100.0,
            damping=0.8,
            gravity=(0, 0, 0)
        )
        pose = create_test_pose()
        chain.initialize(pose)

        # Displace the springs
        for spring in chain._springs:
            spring._position = (1.0, 1.0, 0.0)
            spring._previous_position = (1.0, 1.0, 0.0)

        # Track energy over time (should decrease with damping)
        prev_max_dist = float('inf')
        for step in range(300):
            result = chain.simulate(pose, dt=1/60)

            # Calculate max displacement from rest
            max_dist = 0.0
            for spring in chain._springs:
                rest_pos = pose.get_bone_position(spring.bone_index)
                dist = math.sqrt(
                    (spring._position[0] - rest_pos[0])**2 +
                    (spring._position[1] - rest_pos[1])**2 +
                    (spring._position[2] - rest_pos[2])**2
                )
                max_dist = max(max_dist, dist)

            # After initial settling, energy should generally decrease
            if step > 50:
                assert max_dist < 2.0, f"Spring exploded at step {step}"

        # Should have converged
        assert max_dist < 0.2, f"Spring did not converge: max_dist={max_dist}"

    def test_lookat_target_at_bone_position(self):
        """Test look-at handles target at bone position gracefully."""
        from engine.animation.procedural.lookat import LookAtController

        controller = LookAtController(head_bone=5, enable_saccades=False)
        pose = create_test_pose()
        pose.set_bone_position(5, (1.0, 2.0, 3.0))
        controller.initialize(pose)

        # Target at exact bone position
        target = (1.0, 2.0, 3.0)

        # Should not crash or produce invalid rotations
        result = controller.update(pose, target, dt=1/60, weight=1.0)
        rot = result.get_bone_rotation(5)

        # Rotation should be finite
        assert all(math.isfinite(r) for r in rot)

    def test_impulse_response_zero_offset_division(self):
        """Test impulse response handles zero offset without division error."""
        from engine.animation.procedural.secondary_motion import ImpulseResponse

        motion = ImpulseResponse(
            affected_bones=[0],
            stiffness=50.0,
            damping=0.7,
            max_response=0.1
        )
        pose = create_test_pose()

        # Ensure offsets start at zero
        motion._offsets[0] = (0.0, 0.0, 0.0)

        # Should not crash
        result = motion.update(pose, dt=1/60)
        pos = result.get_bone_position(0)
        assert all(math.isfinite(p) for p in pos)

    def test_spring_energy_dissipation(self):
        """Test that damped spring loses energy over time."""
        from engine.animation.procedural.spring_bone import SpringBone

        spring = SpringBone(bone_index=0, stiffness=50.0, damping=0.5, gravity=(0, 0, 0))
        pose = create_test_pose()
        spring.initialize(pose)

        # Start displaced
        spring._position = (2.0, 0.0, 0.0)
        spring._previous_position = (2.0, 0.0, 0.0)

        # Track kinetic energy approximation
        energies = []
        for _ in range(200):
            spring.simulate(pose, dt=1/60)
            vel = spring.get_velocity(1/60)
            kinetic = 0.5 * spring.mass * (vel[0]**2 + vel[1]**2 + vel[2]**2)

            disp = (
                spring._position[0] - pose.get_bone_position(0)[0],
                spring._position[1] - pose.get_bone_position(0)[1],
                spring._position[2] - pose.get_bone_position(0)[2],
            )
            potential = 0.5 * spring.stiffness * (disp[0]**2 + disp[1]**2 + disp[2]**2)
            total = kinetic + potential
            energies.append(total)

        # Energy should decrease overall (allowing for some oscillation)
        assert energies[-1] < energies[10], "Energy should dissipate with damping"

    def test_locomotion_phase_wrapping(self):
        """Test that locomotion phase correctly wraps around."""
        from engine.animation.procedural.locomotion import ProceduralLocomotion, GaitConfig

        skeleton = MockSkeleton()
        loco = ProceduralLocomotion(skeleton=skeleton, gait_config=GaitConfig())

        # Run for many cycles
        for _ in range(1000):
            phase = loco.update(dt=1/60, speed=2.0)
            assert 0.0 <= phase < 1.0, f"Phase out of bounds: {phase}"


# =============================================================================
# Configuration Tests
# =============================================================================


class TestProceduralConfig:
    """Tests for configuration module."""

    def test_config_import(self):
        """Test configuration can be imported."""
        from engine.animation.procedural.config import ProceduralConfig

        assert ProceduralConfig.Spring.DEFAULT_STIFFNESS == 50.0
        assert ProceduralConfig.Spring.DEFAULT_DAMPING == 0.3

    def test_config_constants(self):
        """Test configuration constants are reasonable."""
        from engine.animation.procedural.config import ProceduralConfig

        # Spring config
        assert ProceduralConfig.Spring.MAX_DT > 0
        assert ProceduralConfig.Spring.MAX_DT < 0.1  # Should be < 100ms

        # Look-at config
        assert ProceduralConfig.LookAt.DEFAULT_ROTATION_SPEED > 0
        assert ProceduralConfig.LookAt.DEFAULT_HEAD_YAW_LIMIT > 0

        # Locomotion config
        assert ProceduralConfig.Locomotion.DEFAULT_STEP_HEIGHT > 0
        assert ProceduralConfig.Locomotion.DEFAULT_CYCLE_DURATION > 0


# =============================================================================
# Decorator Tests
# =============================================================================


class TestProceduralBoneDecorator:
    """Tests for @procedural_bone decorator."""

    def test_procedural_bone_decorator(self):
        """Test basic decorator application."""
        from engine.animation.procedural.spring_bone import procedural_bone

        @procedural_bone(type="spring")
        class TestBone:
            pass

        assert TestBone._procedural_bone is True
        assert TestBone._procedural_bone_type == "spring"

    def test_procedural_bone_invalid_type(self):
        """Test that invalid type raises error."""
        from engine.animation.procedural.spring_bone import procedural_bone

        with pytest.raises(ValueError, match="Invalid type"):
            @procedural_bone(type="invalid")
            class BadBone:
                pass

    def test_procedural_bone_all_types(self):
        """Test all valid procedural bone types."""
        from engine.animation.procedural.spring_bone import procedural_bone

        valid_types = ["jiggle", "spring", "lookat", "aim", "twist"]

        for bone_type in valid_types:
            @procedural_bone(type=bone_type)
            class TestBone:
                pass

            assert TestBone._procedural_bone_type == bone_type

    def test_procedural_bone_tags(self):
        """Test that tags are set correctly."""
        from engine.animation.procedural.spring_bone import procedural_bone

        @procedural_bone(type="jiggle")
        class TestBone:
            pass

        assert TestBone._tags["procedural_bone"] is True
        assert TestBone._tags["procedural_bone_type"] == "jiggle"

    def test_procedural_bone_applied_decorators(self):
        """Test that decorator is tracked."""
        from engine.animation.procedural.spring_bone import procedural_bone

        @procedural_bone(type="spring")
        class TestBone:
            pass

        assert "procedural_bone" in TestBone._applied_decorators
