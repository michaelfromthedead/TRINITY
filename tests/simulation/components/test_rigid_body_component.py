"""
Whitebox tests for RigidBodyComponent.

Tests cover:
- Component lifecycle (creation, initialization, cleanup)
- Physics state management (position, rotation, velocity)
- Force and impulse application
- Sleep management
- Collision callbacks
- Serialization/deserialization
"""

import pytest
from dataclasses import dataclass

from engine.simulation.character.character_controller import (
    Quaternion,
    Transform,
    Vector3,
)
from engine.simulation.components.rigid_body_component import (
    ActivationState,
    CollisionEvent,
    RigidBodyComponent,
    RigidBodyConfig,
    RigidBodyType,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_config() -> RigidBodyConfig:
    """Default rigid body configuration."""
    return RigidBodyConfig()


@pytest.fixture
def dynamic_component() -> RigidBodyComponent:
    """Dynamic rigid body component."""
    config = RigidBodyConfig(
        mass=10.0,
        friction=0.5,
        restitution=0.3,
        body_type=RigidBodyType.DYNAMIC,
    )
    return RigidBodyComponent(entity_id=1, config=config)


@pytest.fixture
def kinematic_component() -> RigidBodyComponent:
    """Kinematic rigid body component."""
    config = RigidBodyConfig(
        mass=0.0,
        body_type=RigidBodyType.KINEMATIC,
    )
    return RigidBodyComponent(entity_id=2, config=config)


@pytest.fixture
def static_component() -> RigidBodyComponent:
    """Static rigid body component."""
    config = RigidBodyConfig(
        mass=0.0,
        body_type=RigidBodyType.STATIC,
    )
    return RigidBodyComponent(entity_id=3, config=config)


# =============================================================================
# RigidBodyConfig Tests
# =============================================================================


class TestRigidBodyConfig:
    """Tests for RigidBodyConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RigidBodyConfig()

        assert config.mass == 1.0
        assert config.friction == 0.5
        assert config.restitution == 0.0
        assert config.linear_damping == 0.0
        assert config.angular_damping == 0.05
        assert config.gravity_scale == 1.0
        assert config.body_type == RigidBodyType.DYNAMIC
        assert config.continuous_collision is False
        assert config.collision_group == 1
        assert config.collision_mask == 0xFFFF

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RigidBodyConfig(
            mass=50.0,
            friction=0.8,
            restitution=0.5,
            linear_damping=0.1,
            angular_damping=0.2,
            gravity_scale=0.5,
            body_type=RigidBodyType.KINEMATIC,
            continuous_collision=True,
            collision_group=2,
            collision_mask=0x00FF,
        )

        assert config.mass == 50.0
        assert config.friction == 0.8
        assert config.restitution == 0.5
        assert config.linear_damping == 0.1
        assert config.angular_damping == 0.2
        assert config.gravity_scale == 0.5
        assert config.body_type == RigidBodyType.KINEMATIC
        assert config.continuous_collision is True
        assert config.collision_group == 2
        assert config.collision_mask == 0x00FF


class TestRigidBodyType:
    """Tests for RigidBodyType enum."""

    def test_enum_values(self):
        """Test enum string values."""
        assert RigidBodyType.STATIC.value == "static"
        assert RigidBodyType.KINEMATIC.value == "kinematic"
        assert RigidBodyType.DYNAMIC.value == "dynamic"

    def test_enum_from_string(self):
        """Test creating enum from string."""
        assert RigidBodyType("static") == RigidBodyType.STATIC
        assert RigidBodyType("kinematic") == RigidBodyType.KINEMATIC
        assert RigidBodyType("dynamic") == RigidBodyType.DYNAMIC


class TestActivationState:
    """Tests for ActivationState enum."""

    def test_enum_values(self):
        """Test all activation states exist."""
        assert ActivationState.ACTIVE.value == "active"
        assert ActivationState.SLEEPING.value == "sleeping"
        assert ActivationState.WANTS_DEACTIVATION.value == "wants_deactivation"
        assert ActivationState.DISABLE_DEACTIVATION.value == "disable_deactivation"
        assert ActivationState.DISABLE_SIMULATION.value == "disable_simulation"


class TestCollisionEvent:
    """Tests for CollisionEvent dataclass."""

    def test_default_values(self):
        """Test default collision event values."""
        event = CollisionEvent()

        assert event.other_entity == 0
        assert event.contact_point.x == 0.0
        assert event.contact_normal.y == 1.0  # up vector
        assert event.impulse == 0.0

    def test_custom_values(self):
        """Test custom collision event values."""
        event = CollisionEvent(
            other_entity=42,
            contact_point=Vector3(1.0, 2.0, 3.0),
            contact_normal=Vector3(0.0, 0.0, 1.0),
            impulse=100.0,
            relative_velocity=Vector3(5.0, 0.0, 0.0),
        )

        assert event.other_entity == 42
        assert event.contact_point.x == 1.0
        assert event.contact_normal.z == 1.0
        assert event.impulse == 100.0
        assert event.relative_velocity.x == 5.0


# =============================================================================
# RigidBodyComponent Creation Tests
# =============================================================================


class TestRigidBodyComponentCreation:
    """Tests for component creation and initialization."""

    def test_create_with_default_config(self):
        """Test creating component with default config."""
        component = RigidBodyComponent(entity_id=1)

        assert component.entity_id == 1
        assert component.body_id is None
        assert component.is_dynamic is True
        assert component.mass == 1.0

    def test_create_with_custom_config(self, dynamic_component):
        """Test creating component with custom config."""
        assert dynamic_component.entity_id == 1
        assert dynamic_component.mass == 10.0
        assert dynamic_component.is_dynamic is True

    def test_initialization_with_body_id(self, dynamic_component):
        """Test initialization with physics body ID."""
        assert dynamic_component.body_id is None

        dynamic_component.initialize(body_id=100)

        assert dynamic_component.body_id == 100

    def test_cleanup_resets_state(self, dynamic_component):
        """Test cleanup resets component state."""
        dynamic_component.initialize(body_id=100)
        dynamic_component.position = Vector3(1.0, 2.0, 3.0)

        dynamic_component.cleanup()

        assert dynamic_component.body_id is None


# =============================================================================
# Body Type Tests
# =============================================================================


class TestBodyTypes:
    """Tests for different body types."""

    def test_dynamic_body_properties(self, dynamic_component):
        """Test dynamic body type properties."""
        assert dynamic_component.is_dynamic is True
        assert dynamic_component.is_kinematic is False
        assert dynamic_component.is_static is False

    def test_kinematic_body_properties(self, kinematic_component):
        """Test kinematic body type properties."""
        assert kinematic_component.is_dynamic is False
        assert kinematic_component.is_kinematic is True
        assert kinematic_component.is_static is False

    def test_static_body_properties(self, static_component):
        """Test static body type properties."""
        assert static_component.is_dynamic is False
        assert static_component.is_kinematic is False
        assert static_component.is_static is True

    def test_set_kinematic(self, dynamic_component):
        """Test setting kinematic mode."""
        assert dynamic_component.is_kinematic is False

        dynamic_component.set_kinematic(True)

        assert dynamic_component.is_kinematic is True
        assert dynamic_component.body_type == RigidBodyType.KINEMATIC

    def test_unset_kinematic(self, kinematic_component):
        """Test unsetting kinematic mode."""
        kinematic_component.set_kinematic(False)

        assert kinematic_component.is_kinematic is False
        assert kinematic_component.is_dynamic is True


# =============================================================================
# Transform Tests
# =============================================================================


class TestTransformManagement:
    """Tests for transform management."""

    def test_initial_position(self, dynamic_component):
        """Test initial position is zero."""
        assert dynamic_component.position.x == 0.0
        assert dynamic_component.position.y == 0.0
        assert dynamic_component.position.z == 0.0

    def test_set_position(self, dynamic_component):
        """Test setting position."""
        dynamic_component.position = Vector3(10.0, 20.0, 30.0)

        assert dynamic_component.position.x == 10.0
        assert dynamic_component.position.y == 20.0
        assert dynamic_component.position.z == 30.0

    def test_initial_rotation(self, dynamic_component):
        """Test initial rotation is identity."""
        rot = dynamic_component.rotation
        assert rot.x == 0.0
        assert rot.y == 0.0
        assert rot.z == 0.0
        assert rot.w == 1.0

    def test_set_rotation(self, dynamic_component):
        """Test setting rotation."""
        q = Quaternion(0.0, 0.707, 0.0, 0.707)
        dynamic_component.rotation = q

        assert dynamic_component.rotation.y == 0.707

    def test_get_transform(self, dynamic_component):
        """Test getting full transform."""
        dynamic_component.position = Vector3(1.0, 2.0, 3.0)

        transform = dynamic_component.get_transform()

        assert transform.position.x == 1.0
        assert transform.position.y == 2.0
        assert transform.position.z == 3.0

    def test_set_transform(self, dynamic_component):
        """Test setting full transform."""
        transform = Transform(
            position=Vector3(5.0, 6.0, 7.0),
            rotation=Quaternion(0.0, 0.5, 0.0, 0.866),
        )

        dynamic_component.set_transform(transform)

        assert dynamic_component.position.x == 5.0
        assert dynamic_component.rotation.y == 0.5

    def test_teleport(self, dynamic_component):
        """Test teleport resets velocities."""
        dynamic_component.linear_velocity = Vector3(10.0, 0.0, 0.0)
        dynamic_component.angular_velocity = Vector3(0.0, 5.0, 0.0)

        dynamic_component.teleport(Vector3(100.0, 0.0, 0.0))

        assert dynamic_component.position.x == 100.0
        assert dynamic_component.linear_velocity.x == 0.0
        assert dynamic_component.angular_velocity.y == 0.0

    def test_teleport_with_rotation(self, dynamic_component):
        """Test teleport with rotation."""
        rotation = Quaternion(0.0, 1.0, 0.0, 0.0)

        dynamic_component.teleport(Vector3(0.0, 10.0, 0.0), rotation)

        assert dynamic_component.position.y == 10.0
        assert dynamic_component.rotation.y == 1.0


# =============================================================================
# Velocity Tests
# =============================================================================


class TestVelocityManagement:
    """Tests for velocity management."""

    def test_initial_velocities(self, dynamic_component):
        """Test initial velocities are zero."""
        assert dynamic_component.linear_velocity.magnitude() == 0.0
        assert dynamic_component.angular_velocity.magnitude() == 0.0

    def test_set_linear_velocity(self, dynamic_component):
        """Test setting linear velocity."""
        dynamic_component.linear_velocity = Vector3(10.0, 5.0, 0.0)

        assert dynamic_component.linear_velocity.x == 10.0
        assert dynamic_component.linear_velocity.y == 5.0

    def test_set_angular_velocity(self, dynamic_component):
        """Test setting angular velocity."""
        dynamic_component.angular_velocity = Vector3(0.0, 3.14, 0.0)

        assert dynamic_component.angular_velocity.y == 3.14

    def test_get_velocity_at_point(self, dynamic_component):
        """Test velocity at a specific point (includes angular contribution)."""
        dynamic_component.position = Vector3(0.0, 0.0, 0.0)
        dynamic_component.linear_velocity = Vector3(10.0, 0.0, 0.0)
        dynamic_component.angular_velocity = Vector3(0.0, 1.0, 0.0)  # Rotating around Y

        # Point offset in X direction
        point = Vector3(0.0, 0.0, 1.0)
        velocity = dynamic_component.get_velocity_at_point(point)

        # Linear + angular contribution
        # Angular Y cross (0, 0, 1) = (-1, 0, 0) (simplified)
        assert velocity.x != 0.0


# =============================================================================
# Force Application Tests
# =============================================================================


class TestForceApplication:
    """Tests for force and impulse application."""

    def test_add_impulse(self, dynamic_component):
        """Test adding impulse changes velocity instantly."""
        impulse = Vector3(100.0, 0.0, 0.0)

        dynamic_component.add_force(impulse, mode="impulse")

        # Impulse / mass = velocity change
        expected_vel = 100.0 / 10.0  # mass = 10.0
        assert abs(dynamic_component.linear_velocity.x - expected_vel) < 0.001

    def test_add_acceleration(self, dynamic_component):
        """Test adding acceleration."""
        accel = Vector3(5.0, 0.0, 0.0)

        dynamic_component.add_force(accel, mode="acceleration")

        assert dynamic_component.linear_velocity.x == 5.0

    def test_force_on_static_body_ignored(self, static_component):
        """Test that forces are ignored on static bodies."""
        static_component.add_force(Vector3(1000.0, 0.0, 0.0), mode="impulse")

        assert static_component.linear_velocity.x == 0.0

    def test_force_on_kinematic_body_ignored(self, kinematic_component):
        """Test that forces are ignored on kinematic bodies."""
        kinematic_component.add_force(Vector3(1000.0, 0.0, 0.0), mode="impulse")

        assert kinematic_component.linear_velocity.x == 0.0

    def test_add_force_at_position(self, dynamic_component):
        """Test adding force at a position creates torque."""
        dynamic_component.position = Vector3(0.0, 0.0, 0.0)

        # Apply force offset from center
        force = Vector3(0.0, 100.0, 0.0)
        position = Vector3(1.0, 0.0, 0.0)

        dynamic_component.add_force_at_position(force, position, mode="impulse")

        # Should create angular velocity
        assert dynamic_component.angular_velocity.magnitude() > 0.0

    def test_add_torque(self, dynamic_component):
        """Test adding torque."""
        torque = Vector3(0.0, 10.0, 0.0)

        dynamic_component.add_torque(torque, mode="impulse")

        assert dynamic_component.angular_velocity.y == 10.0

    def test_add_explosive_force(self, dynamic_component):
        """Test explosive force application."""
        dynamic_component.position = Vector3(0.0, 0.0, 5.0)
        explosion_pos = Vector3(0.0, 0.0, 0.0)

        dynamic_component.add_explosive_force(
            force=1000.0,
            explosion_position=explosion_pos,
            radius=10.0,
            upward_modifier=0.0,
        )

        # Body should be pushed away from explosion
        assert dynamic_component.linear_velocity.z > 0.0

    def test_explosive_force_outside_radius(self, dynamic_component):
        """Test explosive force has no effect outside radius."""
        dynamic_component.position = Vector3(0.0, 0.0, 15.0)
        explosion_pos = Vector3(0.0, 0.0, 0.0)

        dynamic_component.add_explosive_force(
            force=1000.0,
            explosion_position=explosion_pos,
            radius=10.0,
        )

        assert dynamic_component.linear_velocity.magnitude() == 0.0

    def test_explosive_force_falloff(self, dynamic_component):
        """Test explosive force falloff with distance."""
        # Close to explosion
        close = RigidBodyComponent(
            entity_id=10,
            config=RigidBodyConfig(mass=10.0),
        )
        close.position = Vector3(0.0, 0.0, 2.0)

        # Far from explosion
        far = RigidBodyComponent(
            entity_id=11,
            config=RigidBodyConfig(mass=10.0),
        )
        far.position = Vector3(0.0, 0.0, 8.0)

        explosion_pos = Vector3(0.0, 0.0, 0.0)

        close.add_explosive_force(1000.0, explosion_pos, 10.0)
        far.add_explosive_force(1000.0, explosion_pos, 10.0)

        # Close should have higher velocity
        assert close.linear_velocity.z > far.linear_velocity.z


# =============================================================================
# Sleep Management Tests
# =============================================================================


class TestSleepManagement:
    """Tests for sleep/activation state management."""

    def test_initial_state_active(self, dynamic_component):
        """Test initial activation state is active."""
        assert dynamic_component.is_sleeping is False

    def test_put_to_sleep(self, dynamic_component):
        """Test putting body to sleep."""
        dynamic_component.linear_velocity = Vector3(1.0, 0.0, 0.0)

        dynamic_component.put_to_sleep()

        assert dynamic_component.is_sleeping is True
        assert dynamic_component.linear_velocity.x == 0.0
        assert dynamic_component.angular_velocity.magnitude() == 0.0

    def test_wake_up(self, dynamic_component):
        """Test waking up a sleeping body."""
        dynamic_component.put_to_sleep()
        assert dynamic_component.is_sleeping is True

        dynamic_component.wake_up()

        assert dynamic_component.is_sleeping is False

    def test_disable_sleep(self, dynamic_component):
        """Test disabling sleep."""
        dynamic_component.set_sleep_allowed(False)

        dynamic_component.put_to_sleep()

        # Should not go to sleep when disabled
        assert dynamic_component.is_sleeping is False

    def test_re_enable_sleep(self, dynamic_component):
        """Test re-enabling sleep."""
        dynamic_component.set_sleep_allowed(False)
        dynamic_component.set_sleep_allowed(True)

        dynamic_component.put_to_sleep()

        assert dynamic_component.is_sleeping is True


# =============================================================================
# Configuration Tests
# =============================================================================


class TestConfigurationMethods:
    """Tests for configuration modification methods."""

    def test_set_mass(self, dynamic_component):
        """Test setting mass."""
        dynamic_component.set_mass(50.0)

        assert dynamic_component.mass == 50.0

    def test_set_mass_clamped_to_zero(self, dynamic_component):
        """Test mass is clamped to minimum zero."""
        dynamic_component.set_mass(-10.0)

        assert dynamic_component.mass == 0.0

    def test_set_gravity_scale(self, dynamic_component):
        """Test setting gravity scale."""
        dynamic_component.set_gravity_scale(0.5)

        assert dynamic_component.config.gravity_scale == 0.5

    def test_set_friction(self, dynamic_component):
        """Test setting friction."""
        dynamic_component.set_friction(0.8)

        assert dynamic_component.config.friction == 0.8

    def test_set_friction_clamped(self, dynamic_component):
        """Test friction is clamped to minimum zero."""
        dynamic_component.set_friction(-0.5)

        assert dynamic_component.config.friction == 0.0

    def test_set_restitution(self, dynamic_component):
        """Test setting restitution."""
        dynamic_component.set_restitution(0.7)

        assert dynamic_component.config.restitution == 0.7

    def test_set_restitution_clamped(self, dynamic_component):
        """Test restitution is clamped to 0-1 range."""
        dynamic_component.set_restitution(1.5)
        assert dynamic_component.config.restitution == 1.0

        dynamic_component.set_restitution(-0.5)
        assert dynamic_component.config.restitution == 0.0


# =============================================================================
# Collision Callback Tests
# =============================================================================


class TestCollisionCallbacks:
    """Tests for collision callback handling."""

    def test_set_collision_callbacks(self, dynamic_component):
        """Test setting collision callbacks."""
        enter_called = []
        stay_called = []
        exit_called = []

        dynamic_component.set_collision_callbacks(
            on_enter=lambda e: enter_called.append(e),
            on_stay=lambda e: stay_called.append(e),
            on_exit=lambda e: exit_called.append(e),
        )

        # Trigger enter
        event = CollisionEvent(other_entity=99)
        dynamic_component.handle_collision(event)

        assert len(enter_called) == 1
        assert enter_called[0].other_entity == 99

    def test_collision_stay(self, dynamic_component):
        """Test collision stay callback."""
        stay_called = []
        dynamic_component.set_collision_callbacks(on_stay=lambda e: stay_called.append(e))

        # First collision - enter
        event1 = CollisionEvent(other_entity=99)
        dynamic_component.handle_collision(event1)

        # Second collision - stay
        event2 = CollisionEvent(other_entity=99)
        dynamic_component.handle_collision(event2)

        assert len(stay_called) == 1

    def test_collision_exit(self, dynamic_component):
        """Test collision exit callback."""
        exit_called = []
        dynamic_component.set_collision_callbacks(on_exit=lambda e: exit_called.append(e))

        # Enter collision
        event = CollisionEvent(other_entity=99)
        dynamic_component.handle_collision(event)

        # Exit collision
        dynamic_component.handle_collision_end(99)

        assert len(exit_called) == 1
        assert exit_called[0] == 99

    def test_collision_end_not_colliding(self, dynamic_component):
        """Test collision end for non-colliding entity does nothing."""
        exit_called = []
        dynamic_component.set_collision_callbacks(on_exit=lambda e: exit_called.append(e))

        dynamic_component.handle_collision_end(999)

        assert len(exit_called) == 0


# =============================================================================
# Serialization Tests
# =============================================================================


class TestSerialization:
    """Tests for state serialization/deserialization."""

    def test_get_state(self, dynamic_component):
        """Test getting serializable state."""
        dynamic_component.position = Vector3(1.0, 2.0, 3.0)
        dynamic_component.linear_velocity = Vector3(5.0, 0.0, 0.0)

        state = dynamic_component.get_state()

        assert state["entity_id"] == 1
        assert state["position"] == (1.0, 2.0, 3.0)
        assert state["linear_velocity"] == (5.0, 0.0, 0.0)
        assert state["config"]["mass"] == 10.0
        assert state["config"]["body_type"] == "dynamic"

    def test_load_state(self, dynamic_component):
        """Test loading from serialized state."""
        state = {
            "position": (10.0, 20.0, 30.0),
            "rotation": (0.0, 0.707, 0.0, 0.707),
            "linear_velocity": (5.0, 0.0, 0.0),
            "angular_velocity": (0.0, 1.0, 0.0),
            "activation_state": "sleeping",
            "config": {
                "mass": 25.0,
                "friction": 0.8,
                "restitution": 0.5,
                "body_type": "kinematic",
            },
        }

        dynamic_component.load_state(state)

        assert dynamic_component.position.x == 10.0
        assert dynamic_component.position.y == 20.0
        assert dynamic_component.rotation.y == 0.707
        assert dynamic_component.linear_velocity.x == 5.0
        assert dynamic_component.config.mass == 25.0
        assert dynamic_component.is_sleeping is True

    def test_load_state_partial(self, dynamic_component):
        """Test loading partial state uses defaults."""
        state = {
            "position": (5.0, 0.0, 0.0),
        }

        dynamic_component.load_state(state)

        assert dynamic_component.position.x == 5.0
        # Default values for missing fields
        assert dynamic_component.linear_velocity.magnitude() == 0.0

    def test_roundtrip_serialization(self, dynamic_component):
        """Test state survives serialization roundtrip."""
        dynamic_component.position = Vector3(100.0, 50.0, 25.0)
        dynamic_component.linear_velocity = Vector3(10.0, 5.0, 2.0)
        # Note: put_to_sleep() zeroes velocity, so test without sleeping

        state = dynamic_component.get_state()

        new_component = RigidBodyComponent(entity_id=999)
        new_component.load_state(state)

        assert new_component.position.x == 100.0
        assert state["linear_velocity"] == (10.0, 5.0, 2.0)
        assert new_component.linear_velocity.x == 10.0

    def test_roundtrip_serialization_sleeping(self, dynamic_component):
        """Test sleeping state survives serialization roundtrip."""
        dynamic_component.position = Vector3(100.0, 50.0, 25.0)
        dynamic_component.put_to_sleep()

        state = dynamic_component.get_state()

        new_component = RigidBodyComponent(entity_id=999)
        new_component.load_state(state)

        assert new_component.position.x == 100.0
        assert new_component.is_sleeping is True
        # Velocity is zeroed by put_to_sleep
        assert new_component.linear_velocity.x == 0.0


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_mass_body(self):
        """Test body with zero mass."""
        config = RigidBodyConfig(mass=0.0)
        component = RigidBodyComponent(entity_id=1, config=config)

        # Should still be dynamic but mass=0
        assert component.mass == 0.0

    def test_negative_mass_normalized(self):
        """Test negative mass in config is handled."""
        component = RigidBodyComponent(entity_id=1)
        component.set_mass(-100.0)

        assert component.mass >= 0.0

    def test_extreme_velocity(self, dynamic_component):
        """Test handling extreme velocities."""
        extreme_vel = Vector3(1000000.0, 1000000.0, 1000000.0)
        dynamic_component.linear_velocity = extreme_vel

        assert dynamic_component.linear_velocity.x == 1000000.0

    def test_many_collisions_tracked(self, dynamic_component):
        """Test tracking many simultaneous collisions."""
        for i in range(100):
            event = CollisionEvent(other_entity=i)
            dynamic_component.handle_collision(event)

        # All should be tracked
        # Access private member for test verification
        assert len(dynamic_component._colliding_entities) == 100

    def test_cleanup_clears_collisions(self, dynamic_component):
        """Test cleanup clears collision tracking."""
        for i in range(10):
            event = CollisionEvent(other_entity=i)
            dynamic_component.handle_collision(event)

        dynamic_component.cleanup()

        assert len(dynamic_component._colliding_entities) == 0
