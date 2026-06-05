"""
Whitebox tests for JointComponent.

Tests cover:
- Joint creation and configuration
- Different joint types (fixed, hinge, ball socket, slider, spring)
- Joint limits and drives
- Breaking behavior
- Serialization/deserialization
"""

import math
import pytest

from engine.simulation.character.character_controller import Quaternion, Vector3
from engine.simulation.components.joint_component import (
    JointComponent,
    JointDrive,
    JointLimits,
    JointMotion,
    JointType,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fixed_joint() -> JointComponent:
    """Create a fixed joint."""
    return JointComponent(entity_id=1, joint_type=JointType.FIXED)


@pytest.fixture
def hinge_joint() -> JointComponent:
    """Create a hinge joint."""
    joint = JointComponent(entity_id=2, joint_type=JointType.HINGE)
    joint.configure_hinge(
        axis=Vector3(0.0, 1.0, 0.0),
        lower_angle=-math.pi / 2,
        upper_angle=math.pi / 2,
    )
    return joint


@pytest.fixture
def ball_socket_joint() -> JointComponent:
    """Create a ball socket joint."""
    joint = JointComponent(entity_id=3)
    joint.configure_ball_socket(swing_limit=1.0, twist_limit=0.5)
    return joint


@pytest.fixture
def slider_joint() -> JointComponent:
    """Create a slider joint."""
    joint = JointComponent(entity_id=4)
    joint.configure_slider(
        axis=Vector3(1.0, 0.0, 0.0),
        lower_distance=0.0,
        upper_distance=2.0,
    )
    return joint


@pytest.fixture
def spring_joint() -> JointComponent:
    """Create a spring joint."""
    joint = JointComponent(entity_id=5)
    joint.configure_spring(stiffness=100.0, damping=10.0, rest_length=1.0)
    return joint


# =============================================================================
# JointType Tests
# =============================================================================


class TestJointType:
    """Tests for JointType enum."""

    def test_all_types(self):
        """Test all joint types exist."""
        assert JointType.FIXED.value == "fixed"
        assert JointType.HINGE.value == "hinge"
        assert JointType.BALL_SOCKET.value == "ball"
        assert JointType.SLIDER.value == "slider"
        assert JointType.CONE_TWIST.value == "cone_twist"
        assert JointType.SPRING.value == "spring"
        assert JointType.DISTANCE.value == "distance"
        assert JointType.D6.value == "d6"


class TestJointMotion:
    """Tests for JointMotion enum."""

    def test_all_motions(self):
        """Test all motion types."""
        assert JointMotion.LOCKED.value == "locked"
        assert JointMotion.LIMITED.value == "limited"
        assert JointMotion.FREE.value == "free"


# =============================================================================
# JointLimits Tests
# =============================================================================


class TestJointLimits:
    """Tests for JointLimits dataclass."""

    def test_default_values(self):
        """Test default limit values."""
        limits = JointLimits()

        assert limits.lower == 0.0
        assert limits.upper == 0.0
        assert limits.stiffness == 0.0
        assert limits.damping == 0.0
        assert limits.restitution == 0.0
        assert limits.contact_distance == 0.01

    def test_custom_values(self):
        """Test custom limit values."""
        limits = JointLimits(
            lower=-1.5,
            upper=1.5,
            stiffness=100.0,
            damping=10.0,
            restitution=0.5,
            contact_distance=0.05,
        )

        assert limits.lower == -1.5
        assert limits.upper == 1.5
        assert limits.stiffness == 100.0


# =============================================================================
# JointDrive Tests
# =============================================================================


class TestJointDrive:
    """Tests for JointDrive dataclass."""

    def test_default_values(self):
        """Test default drive values."""
        drive = JointDrive()

        assert drive.stiffness == 0.0
        assert drive.damping == 0.0
        assert drive.max_force == float("inf")
        assert drive.target == 0.0
        assert drive.target_velocity == 0.0
        assert drive.mode == "position"

    def test_custom_values(self):
        """Test custom drive values."""
        drive = JointDrive(
            stiffness=500.0,
            damping=50.0,
            max_force=1000.0,
            target=0.5,
            target_velocity=1.0,
            mode="velocity",
        )

        assert drive.stiffness == 500.0
        assert drive.max_force == 1000.0
        assert drive.mode == "velocity"


# =============================================================================
# JointComponent Creation Tests
# =============================================================================


class TestJointCreation:
    """Tests for joint creation."""

    def test_create_fixed_joint(self, fixed_joint):
        """Test creating fixed joint."""
        assert fixed_joint.entity_id == 1
        assert fixed_joint.joint_type == JointType.FIXED
        assert fixed_joint.joint_id is None

    def test_create_with_default_type(self):
        """Test default joint type is fixed."""
        joint = JointComponent(entity_id=1)
        assert joint.joint_type == JointType.FIXED

    def test_initial_state(self, fixed_joint):
        """Test initial joint state."""
        assert fixed_joint.connected_body_a is None
        assert fixed_joint.connected_body_b is None
        assert fixed_joint.is_broken is False
        assert fixed_joint.enabled is True


# =============================================================================
# Connection Tests
# =============================================================================


class TestJointConnection:
    """Tests for connecting bodies to joints."""

    def test_connect_two_bodies(self, fixed_joint):
        """Test connecting two bodies."""
        fixed_joint.connect(body_a=10, body_b=20)

        assert fixed_joint.connected_body_a == 10
        assert fixed_joint.connected_body_b == 20

    def test_connect_to_world(self, fixed_joint):
        """Test connecting to world (body_b = None)."""
        fixed_joint.connect(body_a=10)

        assert fixed_joint.connected_body_a == 10
        assert fixed_joint.connected_body_b is None

    def test_connect_with_anchors(self, fixed_joint):
        """Test connecting with custom anchors."""
        fixed_joint.connect(
            body_a=10,
            body_b=20,
            anchor_a=Vector3(1.0, 0.0, 0.0),
            anchor_b=Vector3(-1.0, 0.0, 0.0),
        )

        assert fixed_joint.anchor_a.x == 1.0
        assert fixed_joint.anchor_b.x == -1.0

    def test_disconnect(self, fixed_joint):
        """Test disconnecting joint."""
        fixed_joint.connect(body_a=10, body_b=20)
        fixed_joint.disconnect()

        assert fixed_joint.connected_body_a is None
        assert fixed_joint.connected_body_b is None

    def test_anchor_properties(self, fixed_joint):
        """Test anchor getter/setter."""
        fixed_joint.anchor_a = Vector3(2.0, 3.0, 4.0)
        fixed_joint.anchor_b = Vector3(5.0, 6.0, 7.0)

        assert fixed_joint.anchor_a.x == 2.0
        assert fixed_joint.anchor_b.z == 7.0

    def test_axis_property(self, fixed_joint):
        """Test axis getter/setter normalizes."""
        fixed_joint.axis = Vector3(2.0, 0.0, 0.0)

        # Should be normalized
        assert fixed_joint.axis.x == 1.0
        assert fixed_joint.axis.y == 0.0


# =============================================================================
# Hinge Joint Tests
# =============================================================================


class TestHingeJoint:
    """Tests for hinge joint configuration."""

    def test_configure_hinge(self, hinge_joint):
        """Test hinge configuration."""
        assert hinge_joint.joint_type == JointType.HINGE
        assert hinge_joint.axis.y == 1.0

    def test_hinge_limits(self, hinge_joint):
        """Test hinge has angular limits."""
        limits = hinge_joint._limits.get("angular_x")
        assert limits is not None
        assert limits.lower == -math.pi / 2
        assert limits.upper == math.pi / 2

    def test_hinge_without_limits(self):
        """Test hinge without limits."""
        joint = JointComponent(entity_id=1)
        joint.configure_hinge(
            axis=Vector3(0.0, 1.0, 0.0),
            use_limits=False,
        )

        assert joint._angular_motion_x == JointMotion.FREE


# =============================================================================
# Ball Socket Joint Tests
# =============================================================================


class TestBallSocketJoint:
    """Tests for ball socket joint configuration."""

    def test_configure_ball_socket(self, ball_socket_joint):
        """Test ball socket configuration."""
        assert ball_socket_joint.joint_type == JointType.BALL_SOCKET

    def test_ball_socket_limits(self, ball_socket_joint):
        """Test ball socket has angular limits on all axes."""
        assert ball_socket_joint._angular_motion_x == JointMotion.LIMITED
        assert ball_socket_joint._angular_motion_y == JointMotion.LIMITED
        assert ball_socket_joint._angular_motion_z == JointMotion.LIMITED

    def test_twist_limit(self, ball_socket_joint):
        """Test twist limit (X axis)."""
        limits = ball_socket_joint._limits.get("angular_x")
        assert limits is not None
        assert limits.lower == -0.5
        assert limits.upper == 0.5

    def test_swing_limits(self, ball_socket_joint):
        """Test swing limits (Y and Z axes)."""
        limits_y = ball_socket_joint._limits.get("angular_y")
        limits_z = ball_socket_joint._limits.get("angular_z")

        assert limits_y.lower == -1.0
        assert limits_y.upper == 1.0
        assert limits_z.lower == -1.0


# =============================================================================
# Slider Joint Tests
# =============================================================================


class TestSliderJoint:
    """Tests for slider joint configuration."""

    def test_configure_slider(self, slider_joint):
        """Test slider configuration."""
        assert slider_joint.joint_type == JointType.SLIDER
        assert slider_joint.axis.x == 1.0

    def test_slider_linear_limits(self, slider_joint):
        """Test slider has linear limits."""
        assert slider_joint._motion_x == JointMotion.LIMITED
        limits = slider_joint._limits.get("linear_x")
        assert limits is not None
        assert limits.lower == 0.0
        assert limits.upper == 2.0


# =============================================================================
# Spring Joint Tests
# =============================================================================


class TestSpringJoint:
    """Tests for spring joint configuration."""

    def test_configure_spring(self, spring_joint):
        """Test spring configuration."""
        assert spring_joint.joint_type == JointType.SPRING

    def test_spring_drive(self, spring_joint):
        """Test spring has drive configuration."""
        drive = spring_joint._drives.get("x")
        assert drive is not None
        assert drive.stiffness == 100.0
        assert drive.damping == 10.0
        assert drive.target == 1.0
        assert drive.mode == "position"

    def test_spring_motion_free(self, spring_joint):
        """Test spring allows free linear motion."""
        assert spring_joint._motion_x == JointMotion.FREE


# =============================================================================
# Limits Configuration Tests
# =============================================================================


class TestLimitsConfiguration:
    """Tests for setting joint limits."""

    def test_set_linear_limit(self, fixed_joint):
        """Test setting linear limit."""
        limits = JointLimits(lower=-1.0, upper=1.0)
        fixed_joint.set_linear_limit("x", limits)

        stored = fixed_joint._limits["linear_x"]
        assert stored.lower == -1.0
        assert stored.upper == 1.0

    def test_set_angular_limit(self, fixed_joint):
        """Test setting angular limit."""
        limits = JointLimits(lower=-math.pi, upper=math.pi)
        fixed_joint.set_angular_limit("y", limits)

        stored = fixed_joint._limits["angular_y"]
        assert stored.lower == -math.pi

    def test_set_motion(self, fixed_joint):
        """Test setting motion freedom."""
        fixed_joint.set_motion(
            linear_x=JointMotion.FREE,
            linear_y=JointMotion.LIMITED,
            angular_z=JointMotion.FREE,
        )

        assert fixed_joint._motion_x == JointMotion.FREE
        assert fixed_joint._motion_y == JointMotion.LIMITED
        assert fixed_joint._angular_motion_z == JointMotion.FREE


# =============================================================================
# Drive Configuration Tests
# =============================================================================


class TestDriveConfiguration:
    """Tests for setting joint drives."""

    def test_set_drive(self, fixed_joint):
        """Test setting drive."""
        drive = JointDrive(stiffness=200.0, damping=20.0, target=1.5)
        fixed_joint.set_drive("x", drive)

        stored = fixed_joint._drives["x"]
        assert stored.stiffness == 200.0
        assert stored.target == 1.5

    def test_set_target_position(self, spring_joint):
        """Test setting target position."""
        spring_joint.set_target_position("x", 2.5)

        assert spring_joint._drives["x"].target == 2.5

    def test_set_target_velocity(self, spring_joint):
        """Test setting target velocity."""
        spring_joint.set_target_velocity("x", 5.0)

        assert spring_joint._drives["x"].target_velocity == 5.0

    def test_set_target_position_nonexistent_drive(self, fixed_joint):
        """Test setting target on nonexistent drive does nothing."""
        fixed_joint.set_target_position("x", 1.0)
        # Should not raise, just do nothing

    def test_set_target_rotation(self, fixed_joint):
        """Test setting target rotation."""
        rotation = Quaternion(0.0, 0.707, 0.0, 0.707)
        fixed_joint.set_target_rotation(rotation)
        # Currently a stub, just verify no error


# =============================================================================
# Breaking Tests
# =============================================================================


class TestJointBreaking:
    """Tests for joint breaking behavior."""

    def test_set_break_thresholds(self, fixed_joint):
        """Test setting break thresholds."""
        fixed_joint.set_break_thresholds(force=1000.0, torque=500.0)

        assert fixed_joint._break_force == 1000.0
        assert fixed_joint._break_torque == 500.0

    def test_default_thresholds_infinite(self, fixed_joint):
        """Test default break thresholds are infinite."""
        assert fixed_joint._break_force == float("inf")
        assert fixed_joint._break_torque == float("inf")

    def test_check_break_below_threshold(self, fixed_joint):
        """Test joint doesn't break below threshold."""
        fixed_joint.set_break_thresholds(force=1000.0, torque=500.0)

        result = fixed_joint.check_break(force=500.0, torque=200.0)

        assert result is False
        assert fixed_joint.is_broken is False
        assert fixed_joint.enabled is True

    def test_check_break_force_exceeded(self, fixed_joint):
        """Test joint breaks when force exceeded."""
        fixed_joint.set_break_thresholds(force=1000.0, torque=500.0)

        result = fixed_joint.check_break(force=1500.0, torque=0.0)

        assert result is True
        assert fixed_joint.is_broken is True
        assert fixed_joint.enabled is False

    def test_check_break_torque_exceeded(self, fixed_joint):
        """Test joint breaks when torque exceeded."""
        fixed_joint.set_break_thresholds(force=1000.0, torque=500.0)

        result = fixed_joint.check_break(force=0.0, torque=600.0)

        assert result is True
        assert fixed_joint.is_broken is True

    def test_break_callback(self, fixed_joint):
        """Test break callback is called."""
        callback_called = []
        fixed_joint.set_break_callback(lambda: callback_called.append(True))
        fixed_joint.set_break_thresholds(force=100.0)

        fixed_joint.check_break(force=200.0, torque=0.0)

        assert len(callback_called) == 1

    def test_already_broken(self, fixed_joint):
        """Test already broken joint stays broken."""
        fixed_joint.set_break_thresholds(force=100.0)
        fixed_joint.check_break(force=200.0, torque=0.0)

        # Check again
        result = fixed_joint.check_break(force=50.0, torque=0.0)

        assert result is True  # Still broken

    def test_repair(self, fixed_joint):
        """Test repairing a broken joint."""
        fixed_joint.set_break_thresholds(force=100.0)
        fixed_joint.check_break(force=200.0, torque=0.0)
        assert fixed_joint.is_broken is True

        fixed_joint.repair()

        assert fixed_joint.is_broken is False
        assert fixed_joint.enabled is True


# =============================================================================
# Query Tests
# =============================================================================


class TestJointQueries:
    """Tests for joint query methods."""

    def test_get_current_force(self, fixed_joint):
        """Test get current force (stub)."""
        force = fixed_joint.get_current_force()
        assert force.magnitude() == 0.0

    def test_get_current_torque(self, fixed_joint):
        """Test get current torque (stub)."""
        torque = fixed_joint.get_current_torque()
        assert torque.magnitude() == 0.0

    def test_get_position(self, slider_joint):
        """Test get joint position (stub)."""
        pos = slider_joint.get_position()
        assert pos == 0.0

    def test_get_velocity(self, slider_joint):
        """Test get joint velocity (stub)."""
        vel = slider_joint.get_velocity()
        assert vel == 0.0

    def test_get_angle(self, hinge_joint):
        """Test get joint angle (stub)."""
        angle = hinge_joint.get_angle()
        assert angle == 0.0


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestJointLifecycle:
    """Tests for joint lifecycle management."""

    def test_initialize(self, fixed_joint):
        """Test initialization with physics ID."""
        assert fixed_joint.joint_id is None

        fixed_joint.initialize(joint_id=42)

        assert fixed_joint.joint_id == 42

    def test_cleanup(self, fixed_joint):
        """Test cleanup."""
        fixed_joint.initialize(joint_id=42)
        fixed_joint.cleanup()

        assert fixed_joint.joint_id is None

    def test_enabled_property(self, fixed_joint):
        """Test enabled property."""
        assert fixed_joint.enabled is True

        fixed_joint.enabled = False
        assert fixed_joint.enabled is False


# =============================================================================
# Serialization Tests
# =============================================================================


class TestJointSerialization:
    """Tests for state serialization/deserialization."""

    def test_get_state(self, hinge_joint):
        """Test getting serializable state."""
        hinge_joint.connect(body_a=10, body_b=20)
        hinge_joint.anchor_a = Vector3(1.0, 0.0, 0.0)
        hinge_joint.set_break_thresholds(force=500.0, torque=250.0)

        state = hinge_joint.get_state()

        assert state["entity_id"] == 2
        assert state["joint_type"] == "hinge"
        assert state["connected_body_a"] == 10
        assert state["connected_body_b"] == 20
        assert state["anchor_a"] == (1.0, 0.0, 0.0)
        assert state["axis"] == (0.0, 1.0, 0.0)
        assert state["break_force"] == 500.0
        assert state["break_torque"] == 250.0
        assert state["is_broken"] is False
        assert state["enabled"] is True

    def test_load_state(self, fixed_joint):
        """Test loading from serialized state."""
        state = {
            "joint_type": "slider",
            "connected_body_a": 100,
            "connected_body_b": 200,
            "anchor_a": (1.0, 2.0, 3.0),
            "anchor_b": (4.0, 5.0, 6.0),
            "axis": (0.0, 0.0, 1.0),
            "break_force": 1000.0,
            "break_torque": 500.0,
            "is_broken": True,
            "enabled": False,
        }

        fixed_joint.load_state(state)

        assert fixed_joint.joint_type == JointType.SLIDER
        assert fixed_joint.connected_body_a == 100
        assert fixed_joint.connected_body_b == 200
        assert fixed_joint.anchor_a.x == 1.0
        assert fixed_joint.axis.z == 1.0
        assert fixed_joint._break_force == 1000.0
        assert fixed_joint.is_broken is True
        assert fixed_joint.enabled is False

    def test_load_state_partial(self, fixed_joint):
        """Test loading partial state uses defaults."""
        state = {
            "connected_body_a": 50,
        }

        fixed_joint.load_state(state)

        assert fixed_joint.connected_body_a == 50
        # Defaults for missing fields
        assert fixed_joint.connected_body_b is None
        assert fixed_joint._break_force == float("inf")

    def test_roundtrip_serialization(self, ball_socket_joint):
        """Test state survives serialization roundtrip."""
        ball_socket_joint.connect(body_a=1, body_b=2)
        ball_socket_joint.set_break_thresholds(force=800.0)

        state = ball_socket_joint.get_state()

        new_joint = JointComponent(entity_id=99)
        new_joint.load_state(state)

        assert new_joint.joint_type == JointType.BALL_SOCKET
        assert new_joint.connected_body_a == 1
        assert new_joint._break_force == 800.0


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_break_threshold(self):
        """Test zero break threshold breaks immediately."""
        joint = JointComponent(entity_id=1)
        joint.set_break_thresholds(force=0.0)

        result = joint.check_break(force=0.001, torque=0.0)

        assert result is True

    def test_negative_break_threshold(self):
        """Test negative break threshold."""
        joint = JointComponent(entity_id=1)
        joint.set_break_thresholds(force=-100.0)

        # Any positive force should exceed negative threshold
        result = joint.check_break(force=0.001, torque=0.0)
        assert result is True

    def test_very_large_limits(self):
        """Test very large limit values."""
        joint = JointComponent(entity_id=1)
        limits = JointLimits(lower=-1e10, upper=1e10)
        joint.set_linear_limit("x", limits)

        assert joint._limits["linear_x"].upper == 1e10

    def test_zero_drive_stiffness(self):
        """Test zero stiffness drive (disabled)."""
        joint = JointComponent(entity_id=1)
        drive = JointDrive(stiffness=0.0, damping=10.0)
        joint.set_drive("x", drive)

        assert joint._drives["x"].stiffness == 0.0

    def test_multiple_joints_same_bodies(self):
        """Test multiple joints can connect same bodies."""
        joint1 = JointComponent(entity_id=1)
        joint2 = JointComponent(entity_id=2)

        joint1.connect(body_a=10, body_b=20)
        joint2.connect(body_a=10, body_b=20)

        assert joint1.connected_body_a == joint2.connected_body_a
        assert joint1.entity_id != joint2.entity_id
