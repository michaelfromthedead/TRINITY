"""Comprehensive tests for the axis mapping system.

Tests cover axis binding types, axis state management, axis value calculation,
smoothing, callbacks, 2D vector mapping, and the input_axis decorator.
"""

import pytest
import math
from time import time, sleep
from unittest.mock import Mock, MagicMock

from engine.gameplay.input.axis_mapper import (
    AxisBindingType,
    AxisBinding,
    AxisEvent,
    AxisDefinition,
    AxisState,
    AxisMapper,
    Vector2Binding,
    Vector2Definition,
    Vector2Event,
    Vector2Mapper,
    input_axis,
)
from engine.gameplay.input.constants import (
    DEFAULT_AXIS_SENSITIVITY,
    DEFAULT_DEAD_ZONE,
)


# =============================================================================
# Axis Binding Type Tests
# =============================================================================

class TestAxisBindingType:
    """Tests for AxisBindingType enum."""

    def test_binding_types_exist(self):
        """All binding types exist."""
        assert AxisBindingType.DIGITAL
        assert AxisBindingType.ANALOG
        assert AxisBindingType.COMPOSITE


# =============================================================================
# Axis Binding Tests
# =============================================================================

class TestAxisBinding:
    """Tests for AxisBinding dataclass."""

    def test_digital_binding(self):
        """Digital binding creation."""
        binding = AxisBinding(
            binding_type=AxisBindingType.DIGITAL,
            positive_keys=["d", "right"],
            negative_keys=["a", "left"]
        )
        assert binding.binding_type == AxisBindingType.DIGITAL
        assert "d" in binding.positive_keys
        assert "a" in binding.negative_keys

    def test_analog_binding(self):
        """Analog binding creation."""
        binding = AxisBinding(
            binding_type=AxisBindingType.ANALOG,
            analog_key="left_stick_x"
        )
        assert binding.binding_type == AxisBindingType.ANALOG
        assert binding.analog_key == "left_stick_x"

    def test_composite_binding(self):
        """Composite binding creation."""
        binding = AxisBinding(
            binding_type=AxisBindingType.COMPOSITE,
            positive_keys=["d"],
            negative_keys=["a"],
            analog_key="left_stick_x"
        )
        assert binding.binding_type == AxisBindingType.COMPOSITE

    def test_binding_defaults(self):
        """AxisBinding has sensible defaults."""
        binding = AxisBinding()
        assert binding.binding_type == AxisBindingType.DIGITAL
        assert binding.positive_keys == []
        assert binding.negative_keys == []
        assert binding.analog_key == ""
        assert binding.scale == 1.0
        assert binding.dead_zone == DEFAULT_DEAD_ZONE
        assert binding.invert is False

    def test_binding_scale(self):
        """Binding scale is stored."""
        binding = AxisBinding(scale=2.0)
        assert binding.scale == 2.0

    def test_binding_invert(self):
        """Binding invert is stored."""
        binding = AxisBinding(invert=True)
        assert binding.invert is True


# =============================================================================
# Axis Definition Tests
# =============================================================================

class TestAxisDefinition:
    """Tests for AxisDefinition dataclass."""

    def test_axis_creation(self):
        """AxisDefinition can be created."""
        axis = AxisDefinition(
            name="horizontal",
            bindings=[AxisBinding(
                positive_keys=["d"],
                negative_keys=["a"]
            )],
            sensitivity=1.5,
            dead_zone=0.15
        )
        assert axis.name == "horizontal"
        assert len(axis.bindings) == 1
        assert axis.sensitivity == 1.5

    def test_axis_defaults(self):
        """AxisDefinition has sensible defaults."""
        axis = AxisDefinition(name="test")
        assert axis.bindings == []
        assert axis.sensitivity == DEFAULT_AXIS_SENSITIVITY
        assert axis.dead_zone == DEFAULT_DEAD_ZONE
        assert axis.smoothing == 0.0
        assert axis.snap_to_zero is True
        assert axis.clamp is True
        assert axis.description == ""


class TestAxisEvent:
    """Tests for AxisEvent dataclass."""

    def test_event_creation(self):
        """AxisEvent can be created."""
        event = AxisEvent(
            axis_name="horizontal",
            value=0.75,
            raw_value=0.8,
            delta=0.1,
            timestamp=time()
        )
        assert event.axis_name == "horizontal"
        assert event.value == 0.75
        assert event.raw_value == 0.8
        assert event.delta == 0.1


class TestAxisState:
    """Tests for AxisState dataclass."""

    def test_state_creation(self):
        """AxisState can be created."""
        state = AxisState()
        assert state.value == 0.0
        assert state.raw_value == 0.0
        assert state.target_value == 0.0
        assert state.previous_value == 0.0


# =============================================================================
# Axis Mapper Basic Tests
# =============================================================================

class TestAxisMapperBasic:
    """Basic tests for AxisMapper class."""

    @pytest.fixture
    def mapper(self):
        """Create an axis mapper."""
        return AxisMapper()

    @pytest.fixture
    def horizontal_axis(self):
        """Create a horizontal axis definition."""
        return AxisDefinition(
            name="horizontal",
            bindings=[AxisBinding(
                binding_type=AxisBindingType.DIGITAL,
                positive_keys=["d", "right"],
                negative_keys=["a", "left"]
            )]
        )

    def test_register_axis(self, mapper, horizontal_axis):
        """register_axis adds axis to mapper."""
        result = mapper.register_axis(horizontal_axis)
        assert result is True
        assert mapper.get_axis("horizontal") is not None

    def test_register_duplicate_fails(self, mapper, horizontal_axis):
        """Registering duplicate axis fails."""
        mapper.register_axis(horizontal_axis)
        result = mapper.register_axis(horizontal_axis)
        assert result is False

    def test_unregister_axis(self, mapper, horizontal_axis):
        """unregister_axis removes axis."""
        mapper.register_axis(horizontal_axis)
        result = mapper.unregister_axis("horizontal")
        assert result is True
        assert mapper.get_axis("horizontal") is None

    def test_unregister_nonexistent(self, mapper):
        """Unregistering nonexistent axis fails."""
        result = mapper.unregister_axis("nonexistent")
        assert result is False

    def test_get_axis(self, mapper, horizontal_axis):
        """get_axis returns axis definition."""
        mapper.register_axis(horizontal_axis)
        axis = mapper.get_axis("horizontal")
        assert axis is horizontal_axis

    def test_get_axis_nonexistent(self, mapper):
        """get_axis returns None for nonexistent."""
        assert mapper.get_axis("nonexistent") is None

    def test_enabled_property(self, mapper):
        """enabled property controls processing."""
        assert mapper.enabled is True
        mapper.enabled = False
        assert mapper.enabled is False


# =============================================================================
# Axis Value Calculation Tests
# =============================================================================

class TestAxisValueCalculation:
    """Tests for axis value calculation."""

    @pytest.fixture
    def mapper(self):
        """Create an axis mapper with digital axis."""
        m = AxisMapper()
        m.register_axis(AxisDefinition(
            name="horizontal",
            bindings=[AxisBinding(
                binding_type=AxisBindingType.DIGITAL,
                positive_keys=["d"],
                negative_keys=["a"]
            )],
            dead_zone=0.0  # Disable dead zone for testing
        ))
        return m

    def test_initial_value_is_zero(self, mapper):
        """Axis starts at zero."""
        assert mapper.get_axis_value("horizontal") == 0.0

    def test_positive_key_gives_positive(self, mapper):
        """Positive key gives positive value."""
        mapper.set_input_state("d", True, 1.0)
        mapper.update(0.016)
        assert mapper.get_axis_value("horizontal") > 0

    def test_negative_key_gives_negative(self, mapper):
        """Negative key gives negative value."""
        mapper.set_input_state("a", True, 1.0)
        mapper.update(0.016)
        assert mapper.get_axis_value("horizontal") < 0

    def test_both_keys_cancel(self, mapper):
        """Both keys pressed cancel out."""
        mapper.set_input_state("d", True, 1.0)
        mapper.set_input_state("a", True, 1.0)
        mapper.update(0.016)
        assert mapper.get_axis_value("horizontal") == 0.0

    def test_release_returns_to_zero(self, mapper):
        """Releasing key returns to zero."""
        mapper.set_input_state("d", True, 1.0)
        mapper.update(0.016)
        mapper.set_input_state("d", False, 0.0)
        mapper.update(0.016)
        assert mapper.get_axis_value("horizontal") == 0.0

    def test_get_raw_axis_value(self, mapper):
        """get_raw_axis_value returns unprocessed value."""
        mapper.set_input_state("d", True, 1.0)
        mapper.update(0.016)
        assert mapper.get_raw_axis_value("horizontal") == 1.0


class TestAxisAnalogBinding:
    """Tests for analog axis bindings."""

    @pytest.fixture
    def mapper(self):
        """Create an axis mapper with analog axis."""
        m = AxisMapper()
        m.register_axis(AxisDefinition(
            name="stick_x",
            bindings=[AxisBinding(
                binding_type=AxisBindingType.ANALOG,
                analog_key="left_x",
                dead_zone=0.15
            )],
            dead_zone=0.0  # Axis-level dead zone off
        ))
        return m

    def test_analog_value_passthrough(self, mapper):
        """Analog values pass through."""
        mapper.set_input_state("left_x", True, 0.5)
        mapper.update(0.016)
        # After binding dead zone applied
        assert mapper.get_axis_value("stick_x") > 0

    def test_analog_negative_value(self, mapper):
        """Analog negative values work."""
        mapper.set_input_state("left_x", True, -0.75)
        mapper.update(0.016)
        assert mapper.get_axis_value("stick_x") < 0

    def test_analog_dead_zone(self, mapper):
        """Analog binding respects dead zone."""
        mapper.set_input_state("left_x", True, 0.1)  # Within dead zone
        mapper.update(0.016)
        assert mapper.get_axis_value("stick_x") == 0.0


class TestAxisCompositeBinding:
    """Tests for composite axis bindings."""

    @pytest.fixture
    def mapper(self):
        """Create an axis mapper with composite axis."""
        m = AxisMapper()
        m.register_axis(AxisDefinition(
            name="move_x",
            bindings=[AxisBinding(
                binding_type=AxisBindingType.COMPOSITE,
                positive_keys=["d"],
                negative_keys=["a"],
                analog_key="left_x"
            )],
            dead_zone=0.0
        ))
        return m

    def test_digital_input_works(self, mapper):
        """Digital input works in composite."""
        mapper.set_input_state("d", True, 1.0)
        mapper.update(0.016)
        assert mapper.get_axis_value("move_x") > 0

    def test_analog_input_works(self, mapper):
        """Analog input works in composite."""
        mapper.set_input_state("left_x", True, 0.75)
        mapper.update(0.016)
        assert mapper.get_axis_value("move_x") > 0

    def test_analog_overrides_digital_when_larger(self, mapper):
        """Analog overrides digital when larger magnitude."""
        mapper.set_input_state("d", True, 1.0)  # Digital = 1.0
        mapper.set_input_state("left_x", True, 0.5)  # Analog = 0.5
        mapper.update(0.016)
        # Should use digital (larger)
        value = mapper.get_axis_value("move_x")
        assert value == pytest.approx(1.0, rel=0.01)


# =============================================================================
# Axis Binding Management Tests
# =============================================================================

class TestAxisBindingManagement:
    """Tests for managing axis bindings."""

    @pytest.fixture
    def mapper(self):
        """Create an axis mapper."""
        m = AxisMapper()
        m.register_axis(AxisDefinition(name="test"))
        return m

    def test_add_binding(self, mapper):
        """add_binding adds binding to axis."""
        binding = AxisBinding(positive_keys=["w"], negative_keys=["s"])
        result = mapper.add_binding("test", binding)
        assert result is True

        axis = mapper.get_axis("test")
        assert len(axis.bindings) == 1

    def test_add_binding_nonexistent_axis(self, mapper):
        """add_binding to nonexistent axis fails."""
        binding = AxisBinding()
        result = mapper.add_binding("nonexistent", binding)
        assert result is False

    def test_remove_binding(self, mapper):
        """remove_binding removes binding."""
        binding = AxisBinding(positive_keys=["w"])
        mapper.add_binding("test", binding)

        result = mapper.remove_binding("test", 0)
        assert result is True

        axis = mapper.get_axis("test")
        assert len(axis.bindings) == 0

    def test_remove_binding_invalid_index(self, mapper):
        """remove_binding with invalid index fails."""
        result = mapper.remove_binding("test", 0)
        assert result is False

    def test_remove_binding_nonexistent_axis(self, mapper):
        """remove_binding from nonexistent axis fails."""
        result = mapper.remove_binding("nonexistent", 0)
        assert result is False


# =============================================================================
# Axis Sensitivity and Processing Tests
# =============================================================================

class TestAxisSensitivity:
    """Tests for axis sensitivity and processing."""

    @pytest.fixture
    def mapper(self):
        """Create an axis mapper."""
        m = AxisMapper()
        m.register_axis(AxisDefinition(
            name="test",
            bindings=[AxisBinding(positive_keys=["d"], negative_keys=["a"])],
            sensitivity=2.0,
            dead_zone=0.0
        ))
        return m

    def test_sensitivity_multiplies_value(self, mapper):
        """Sensitivity multiplies axis value."""
        mapper.set_input_state("d", True, 0.5)
        mapper.update(0.016)

        # 0.5 * 2.0 = 1.0 (clamped)
        assert mapper.get_axis_value("test") == 1.0

    def test_sensitivity_clamped(self, mapper):
        """Sensitivity result is clamped."""
        mapper.set_input_state("d", True, 1.0)
        mapper.update(0.016)

        # 1.0 * 2.0 = 2.0, clamped to 1.0
        assert mapper.get_axis_value("test") == 1.0


class TestAxisDeadZone:
    """Tests for axis-level dead zone."""

    @pytest.fixture
    def mapper(self):
        """Create an axis mapper with dead zone."""
        m = AxisMapper()
        m.register_axis(AxisDefinition(
            name="test",
            bindings=[AxisBinding(
                binding_type=AxisBindingType.ANALOG,
                analog_key="stick"
            )],
            dead_zone=0.2
        ))
        return m

    def test_within_dead_zone_is_zero(self, mapper):
        """Values within dead zone return zero."""
        mapper.set_input_state("stick", True, 0.15)
        mapper.update(0.016)
        assert mapper.get_axis_value("test") == 0.0

    def test_outside_dead_zone_nonzero(self, mapper):
        """Values outside dead zone return non-zero."""
        mapper.set_input_state("stick", True, 0.5)
        mapper.update(0.016)
        assert mapper.get_axis_value("test") > 0


class TestAxisSmoothing:
    """Tests for axis smoothing."""

    @pytest.fixture
    def mapper(self):
        """Create an axis mapper with smoothing."""
        m = AxisMapper()
        m.register_axis(AxisDefinition(
            name="test",
            bindings=[AxisBinding(
                binding_type=AxisBindingType.ANALOG,
                analog_key="stick"
            )],
            smoothing=0.5,
            dead_zone=0.0
        ))
        return m

    def test_smoothing_reduces_sudden_changes(self, mapper):
        """Smoothing reduces sudden value changes."""
        # Start at 0
        mapper.update(0.016)

        # Jump to 1
        mapper.set_input_state("stick", True, 1.0)
        mapper.update(0.016)

        # Value should be less than 1 due to smoothing
        assert mapper.get_axis_value("test") < 1.0

    def test_smoothing_converges(self, mapper):
        """Smoothing eventually converges to target."""
        mapper.set_input_state("stick", True, 1.0)

        # Multiple updates
        for _ in range(50):
            mapper.update(0.016)

        # Should be close to 1.0
        assert mapper.get_axis_value("test") > 0.9


class TestAxisSnapToZero:
    """Tests for snap-to-zero behavior."""

    @pytest.fixture
    def mapper_snap(self):
        """Create an axis mapper with snap_to_zero."""
        m = AxisMapper()
        m.register_axis(AxisDefinition(
            name="test",
            bindings=[AxisBinding(
                binding_type=AxisBindingType.ANALOG,
                analog_key="stick"
            )],
            snap_to_zero=True,
            dead_zone=0.0
        ))
        return m

    @pytest.fixture
    def mapper_no_snap(self):
        """Create an axis mapper without snap_to_zero."""
        m = AxisMapper()
        m.register_axis(AxisDefinition(
            name="test",
            bindings=[AxisBinding(
                binding_type=AxisBindingType.ANALOG,
                analog_key="stick"
            )],
            snap_to_zero=False,
            dead_zone=0.0
        ))
        return m

    def test_snap_to_zero_on_cross(self, mapper_snap):
        """Value snaps to zero when crossing."""
        # Start positive
        mapper_snap.set_input_state("stick", True, 0.5)
        mapper_snap.update(0.016)

        # Go negative
        mapper_snap.set_input_state("stick", True, -0.5)
        mapper_snap.update(0.016)

        # Should snap to 0 when crossing
        assert mapper_snap.get_axis_value("test") == 0.0

    def test_no_snap_allows_cross(self, mapper_no_snap):
        """Without snap, value can cross zero."""
        # Start positive
        mapper_no_snap.set_input_state("stick", True, 0.5)
        mapper_no_snap.update(0.016)

        # Go negative
        mapper_no_snap.set_input_state("stick", True, -0.5)
        mapper_no_snap.update(0.016)

        # Should be negative
        assert mapper_no_snap.get_axis_value("test") < 0


# =============================================================================
# Axis Callback Tests
# =============================================================================

class TestAxisCallbacks:
    """Tests for axis callbacks."""

    @pytest.fixture
    def mapper(self):
        """Create an axis mapper."""
        m = AxisMapper()
        m.register_axis(AxisDefinition(
            name="test",
            bindings=[AxisBinding(positive_keys=["d"])],
            dead_zone=0.0
        ))
        return m

    def test_bind_callback(self, mapper):
        """bind_callback binds to axis."""
        callback = Mock()
        result = mapper.bind_callback("test", callback)
        assert result is True

    def test_bind_callback_nonexistent(self, mapper):
        """bind_callback to nonexistent axis fails."""
        callback = Mock()
        result = mapper.bind_callback("nonexistent", callback)
        assert result is False

    def test_callback_invoked_on_change(self, mapper):
        """Callback is invoked when axis value changes."""
        callback = Mock()
        mapper.bind_callback("test", callback)

        mapper.set_input_state("d", True, 1.0)
        mapper.update(0.016)

        callback.assert_called_once()
        event = callback.call_args[0][0]
        assert event.axis_name == "test"

    def test_callback_not_invoked_no_change(self, mapper):
        """Callback not invoked when value unchanged."""
        callback = Mock()
        mapper.bind_callback("test", callback)

        # No input change
        mapper.update(0.016)
        mapper.update(0.016)

        callback.assert_not_called()

    def test_unbind_callback(self, mapper):
        """unbind_callback removes callback."""
        callback = Mock()
        mapper.bind_callback("test", callback)
        result = mapper.unbind_callback("test", callback)
        assert result is True

        mapper.set_input_state("d", True, 1.0)
        mapper.update(0.016)

        callback.assert_not_called()

    def test_callback_exception_handled(self, mapper):
        """Callback exception doesn't break mapper."""
        def bad_callback(event):
            raise ValueError("Test error")

        mapper.bind_callback("test", bad_callback)
        mapper.set_input_state("d", True, 1.0)

        # Should not raise
        mapper.update(0.016)


# =============================================================================
# Axis Event Tests
# =============================================================================

class TestAxisEvents:
    """Tests for axis events."""

    @pytest.fixture
    def mapper(self):
        """Create an axis mapper."""
        m = AxisMapper()
        m.register_axis(AxisDefinition(
            name="test",
            bindings=[AxisBinding(positive_keys=["d"])],
            dead_zone=0.0
        ))
        return m

    def test_update_returns_events(self, mapper):
        """update returns list of events."""
        mapper.set_input_state("d", True, 1.0)
        events = mapper.update(0.016)

        assert len(events) == 1
        assert events[0].axis_name == "test"

    def test_event_contains_delta(self, mapper):
        """Event contains value delta."""
        mapper.set_input_state("d", True, 1.0)
        events = mapper.update(0.016)

        assert events[0].delta > 0

    def test_no_events_when_disabled(self, mapper):
        """No events when mapper disabled."""
        mapper.enabled = False
        mapper.set_input_state("d", True, 1.0)
        events = mapper.update(0.016)

        assert events == []


# =============================================================================
# Axis Reset Tests
# =============================================================================

class TestAxisReset:
    """Tests for axis reset."""

    @pytest.fixture
    def mapper(self):
        """Create an axis mapper."""
        m = AxisMapper()
        m.register_axis(AxisDefinition(
            name="test",
            bindings=[AxisBinding(positive_keys=["d"])],
            smoothing=0.5,
            dead_zone=0.0
        ))
        return m

    def test_reset_clears_value(self, mapper):
        """reset clears axis values."""
        mapper.set_input_state("d", True, 1.0)
        mapper.update(0.016)

        mapper.reset()

        assert mapper.get_axis_value("test") == 0.0

    def test_reset_clears_input_states(self, mapper):
        """reset clears input states."""
        mapper.set_input_state("d", True, 1.0)
        mapper.reset()
        mapper.update(0.016)

        assert mapper.get_axis_value("test") == 0.0


# =============================================================================
# Vector2 Mapper Tests
# =============================================================================

class TestVector2Binding:
    """Tests for Vector2Binding dataclass."""

    def test_binding_with_axis_references(self):
        """Binding can reference other axes."""
        binding = Vector2Binding(
            x_axis="horizontal",
            y_axis="vertical"
        )
        assert binding.x_axis == "horizontal"
        assert binding.y_axis == "vertical"

    def test_binding_with_direct_keys(self):
        """Binding can have direct key mappings."""
        binding = Vector2Binding(
            up_keys=["w"],
            down_keys=["s"],
            left_keys=["a"],
            right_keys=["d"]
        )
        assert "w" in binding.up_keys
        assert "d" in binding.right_keys

    def test_binding_with_analog(self):
        """Binding can have analog inputs."""
        binding = Vector2Binding(
            analog_x="left_x",
            analog_y="left_y"
        )
        assert binding.analog_x == "left_x"
        assert binding.analog_y == "left_y"


class TestVector2Definition:
    """Tests for Vector2Definition dataclass."""

    def test_definition_creation(self):
        """Vector2Definition can be created."""
        vector = Vector2Definition(
            name="movement",
            bindings=[Vector2Binding(
                up_keys=["w"],
                down_keys=["s"],
                left_keys=["a"],
                right_keys=["d"]
            )],
            normalize=True
        )
        assert vector.name == "movement"
        assert vector.normalize is True

    def test_definition_defaults(self):
        """Vector2Definition has sensible defaults."""
        vector = Vector2Definition(name="test")
        assert vector.bindings == []
        assert vector.normalize is True
        assert vector.dead_zone == DEFAULT_DEAD_ZONE
        assert vector.sensitivity == DEFAULT_AXIS_SENSITIVITY


class TestVector2Event:
    """Tests for Vector2Event dataclass."""

    def test_event_creation(self):
        """Vector2Event can be created."""
        event = Vector2Event(
            axis_name="movement",
            x=0.5,
            y=0.8,
            magnitude=0.94,
            angle=1.0,
            timestamp=time()
        )
        assert event.axis_name == "movement"
        assert event.x == 0.5
        assert event.y == 0.8


class TestVector2Mapper:
    """Tests for Vector2Mapper class."""

    @pytest.fixture
    def mapper(self):
        """Create a vector mapper."""
        return Vector2Mapper()

    @pytest.fixture
    def movement_vector(self):
        """Create a movement vector definition."""
        return Vector2Definition(
            name="movement",
            bindings=[Vector2Binding(
                up_keys=["w"],
                down_keys=["s"],
                left_keys=["a"],
                right_keys=["d"]
            )],
            normalize=True,
            dead_zone=0.0
        )

    def test_register_vector(self, mapper, movement_vector):
        """register_vector adds vector to mapper."""
        result = mapper.register_vector(movement_vector)
        assert result is True

    def test_register_duplicate_fails(self, mapper, movement_vector):
        """Registering duplicate vector fails."""
        mapper.register_vector(movement_vector)
        result = mapper.register_vector(movement_vector)
        assert result is False

    def test_unregister_vector(self, mapper, movement_vector):
        """unregister_vector removes vector."""
        mapper.register_vector(movement_vector)
        result = mapper.unregister_vector("movement")
        assert result is True

    def test_get_vector_initial_zero(self, mapper, movement_vector):
        """Vector starts at (0, 0)."""
        mapper.register_vector(movement_vector)
        x, y = mapper.get_vector("movement")
        assert x == 0.0
        assert y == 0.0

    def test_get_vector_nonexistent(self, mapper):
        """get_vector returns (0, 0) for nonexistent."""
        x, y = mapper.get_vector("nonexistent")
        assert x == 0.0
        assert y == 0.0

    def test_update_with_input(self, mapper, movement_vector):
        """Update processes input."""
        mapper.register_vector(movement_vector)

        input_states = {"d": (True, 1.0)}
        events = mapper.update(0.016, input_states)

        x, y = mapper.get_vector("movement")
        assert x > 0  # Moving right

    def test_diagonal_movement(self, mapper, movement_vector):
        """Diagonal movement works."""
        mapper.register_vector(movement_vector)

        input_states = {
            "d": (True, 1.0),
            "w": (True, 1.0)
        }
        mapper.update(0.016, input_states)

        x, y = mapper.get_vector("movement")
        assert x > 0
        assert y > 0

    def test_normalized_diagonal(self, mapper, movement_vector):
        """Normalized diagonal doesn't exceed 1."""
        mapper.register_vector(movement_vector)

        input_states = {
            "d": (True, 1.0),
            "w": (True, 1.0)
        }
        mapper.update(0.016, input_states)

        x, y = mapper.get_vector("movement")
        magnitude = math.sqrt(x*x + y*y)
        assert magnitude <= 1.0 + 0.01

    def test_callback_binding(self, mapper, movement_vector):
        """Can bind callback to vector."""
        mapper.register_vector(movement_vector)
        callback = Mock()

        result = mapper.bind_callback("movement", callback)
        assert result is True

        input_states = {"d": (True, 1.0)}
        mapper.update(0.016, input_states)

        callback.assert_called_once()

    def test_reset_clears_values(self, mapper, movement_vector):
        """reset clears vector values."""
        mapper.register_vector(movement_vector)

        input_states = {"d": (True, 1.0)}
        mapper.update(0.016, input_states)

        mapper.reset()

        x, y = mapper.get_vector("movement")
        assert x == 0.0
        assert y == 0.0


class TestVector2WithAxisMapper:
    """Tests for Vector2Mapper with AxisMapper integration."""

    def test_vector_uses_axis_mapper(self):
        """Vector can use AxisMapper for axis values."""
        axis_mapper = AxisMapper()
        axis_mapper.register_axis(AxisDefinition(
            name="horizontal",
            bindings=[AxisBinding(positive_keys=["d"], negative_keys=["a"])],
            dead_zone=0.0
        ))
        axis_mapper.register_axis(AxisDefinition(
            name="vertical",
            bindings=[AxisBinding(positive_keys=["w"], negative_keys=["s"])],
            dead_zone=0.0
        ))

        vector_mapper = Vector2Mapper(axis_mapper)
        vector_mapper.register_vector(Vector2Definition(
            name="movement",
            bindings=[Vector2Binding(
                x_axis="horizontal",
                y_axis="vertical"
            )],
            dead_zone=0.0
        ))

        # Update axis mapper
        axis_mapper.set_input_state("d", True, 1.0)
        axis_mapper.update(0.016)

        # Update vector mapper
        vector_mapper.update(0.016, {})

        x, y = vector_mapper.get_vector("movement")
        assert x > 0


# =============================================================================
# input_axis Decorator Tests
# =============================================================================

class TestInputAxisDecorator:
    """Tests for input_axis decorator."""

    def test_decorator_marks_function(self):
        """Decorator marks function as input axis."""
        @input_axis(name="test", positive=["d"], negative=["a"])
        def handler(event):
            pass

        assert handler._input_axis is True
        assert handler._axis_name == "test"

    def test_decorator_stores_bindings(self):
        """Decorator stores axis bindings."""
        @input_axis(name="horizontal", positive=["d", "right"], negative=["a", "left"])
        def handler(event):
            pass

        assert "d" in handler._axis_positive
        assert "a" in handler._axis_negative

    def test_decorator_stores_sensitivity(self):
        """Decorator stores sensitivity."""
        @input_axis(name="test", positive=["d"], negative=["a"], sensitivity=2.0)
        def handler(event):
            pass

        assert handler._axis_sensitivity == 2.0

    def test_decorator_requires_name(self):
        """Decorator requires name parameter."""
        with pytest.raises(ValueError):
            @input_axis(name="", positive=["d"], negative=["a"])
            def handler(event):
                pass

    def test_decorator_requires_positive(self):
        """Decorator requires positive parameter."""
        with pytest.raises(ValueError):
            @input_axis(name="test", positive=[], negative=["a"])
            def handler(event):
                pass

    def test_decorator_requires_negative(self):
        """Decorator requires negative parameter."""
        with pytest.raises(ValueError):
            @input_axis(name="test", positive=["d"], negative=[])
            def handler(event):
                pass

    def test_decorated_function_still_callable(self):
        """Decorated function still works normally."""
        result = []

        @input_axis(name="test", positive=["d"], negative=["a"])
        def handler(event):
            result.append(event)

        handler("test_event")
        assert result == ["test_event"]

    def test_decorator_adds_metadata(self):
        """Decorator adds various metadata."""
        @input_axis(name="test", positive=["d"], negative=["a"])
        def handler(event):
            pass

        assert hasattr(handler, '_tags')
        assert handler._tags['input_axis'] is True
        assert 'input' in handler._registries


# =============================================================================
# Integration Tests
# =============================================================================

class TestAxisIntegration:
    """Integration tests for axis system."""

    def test_character_movement_simulation(self):
        """Simulate character movement with axes."""
        mapper = AxisMapper()

        # Register movement axes
        mapper.register_axis(AxisDefinition(
            name="horizontal",
            bindings=[
                AxisBinding(
                    binding_type=AxisBindingType.COMPOSITE,
                    positive_keys=["d", "right"],
                    negative_keys=["a", "left"],
                    analog_key="left_x"
                )
            ],
            dead_zone=0.15,
            smoothing=0.2
        ))
        mapper.register_axis(AxisDefinition(
            name="vertical",
            bindings=[
                AxisBinding(
                    binding_type=AxisBindingType.COMPOSITE,
                    positive_keys=["w", "up"],
                    negative_keys=["s", "down"],
                    analog_key="left_y"
                )
            ],
            dead_zone=0.15,
            smoothing=0.2
        ))

        # Simulate keyboard input
        mapper.set_input_state("d", True, 1.0)
        mapper.set_input_state("w", True, 1.0)
        mapper.update(0.016)

        h = mapper.get_axis_value("horizontal")
        v = mapper.get_axis_value("vertical")

        assert h > 0
        assert v > 0

        # Switch to gamepad
        mapper.set_input_state("d", False, 0.0)
        mapper.set_input_state("w", False, 0.0)
        mapper.set_input_state("left_x", True, 0.5)
        mapper.set_input_state("left_y", True, -0.3)

        # Multiple updates for smoothing
        for _ in range(10):
            mapper.update(0.016)

        h = mapper.get_axis_value("horizontal")
        v = mapper.get_axis_value("vertical")

        assert h > 0
        assert v < 0

    def test_camera_control_simulation(self):
        """Simulate camera control with look axes."""
        mapper = AxisMapper()

        mapper.register_axis(AxisDefinition(
            name="look_x",
            bindings=[AxisBinding(
                binding_type=AxisBindingType.ANALOG,
                analog_key="right_x"
            )],
            sensitivity=2.0,
            dead_zone=0.1
        ))
        mapper.register_axis(AxisDefinition(
            name="look_y",
            bindings=[AxisBinding(
                binding_type=AxisBindingType.ANALOG,
                analog_key="right_y",
                invert=True  # Invert Y for camera
            )],
            sensitivity=2.0,
            dead_zone=0.1
        ))

        # Simulate stick input
        mapper.set_input_state("right_x", True, 0.5)
        mapper.set_input_state("right_y", True, 0.3)
        mapper.update(0.016)

        look_x = mapper.get_axis_value("look_x")
        look_y = mapper.get_axis_value("look_y")

        # X should be positive
        assert look_x > 0
        # Y should be negative (inverted)
        assert look_y < 0
