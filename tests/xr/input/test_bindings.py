"""
Tests for XR input action binding system.

Tests the @xr_action and @xr_axis decorators, action registry,
input profiles, and action value queries.
"""

import pytest

from engine.xr.input.bindings import (
    xr_action,
    xr_axis,
    XRActionType,
    XRInputSource,
    XRActionBinding,
    XRInputProfile,
    XRActionRegistry,
    get_xr_action_registry,
    bind_action,
    unbind_action,
    create_profile,
    get_action_value,
)


# =============================================================================
# @xr_action DECORATOR TESTS
# =============================================================================


class TestXRActionDecorator:
    """Test @xr_action decorator."""

    def test_basic_application(self):
        """Test basic decorator application."""
        @xr_action(name="grab", bindings=["xr_left_grip", "xr_right_grip"])
        def on_grab():
            pass

        assert on_grab._xr_action is True
        assert on_grab._xr_action_name == "grab"

    def test_action_type(self):
        """Test action type specification."""
        @xr_action(
            name="trigger",
            action_type=XRActionType.FLOAT,
            bindings=["xr_left_trigger"],
        )
        def on_trigger():
            pass

        assert on_trigger._xr_action_type == XRActionType.FLOAT

    def test_bindings_stored(self):
        """Test bindings are stored."""
        @xr_action(
            name="fire",
            bindings=["xr_left_trigger", "xr_right_trigger"],
        )
        def on_fire():
            pass

        assert on_fire._xr_action_bindings == ["xr_left_trigger", "xr_right_trigger"]

    def test_threshold(self):
        """Test threshold parameter."""
        @xr_action(
            name="activate",
            bindings=["xr_left_trigger"],
            threshold=0.8,
        )
        def on_activate():
            pass

        assert on_activate._xr_action_threshold == 0.8

    def test_tags(self):
        """Test decorator tags."""
        @xr_action(name="test", bindings=["xr_left_primary"])
        def f():
            pass

        assert f._tags["xr_action"] is True
        assert f._tags["xr_action_name"] == "test"

    def test_registered_in_registry(self):
        """Test action is registered in global registry."""
        @xr_action(name="registered_action", bindings=["xr_left_grip"])
        def on_registered():
            pass

        registry = get_xr_action_registry()
        action = registry.get_action("registered_action")

        assert action is not None
        assert action.action_name == "registered_action"

    # --- Validation ---

    def test_missing_name(self):
        """Test missing name raises error."""
        with pytest.raises(ValueError, match="'name' parameter is required"):
            @xr_action(bindings=["xr_left_grip"])
            def f():
                pass

    def test_empty_name(self):
        """Test empty name raises error."""
        with pytest.raises(ValueError, match="'name' parameter is required"):
            @xr_action(name="", bindings=["xr_left_grip"])
            def f():
                pass

    def test_missing_bindings(self):
        """Test missing bindings raises error."""
        with pytest.raises(ValueError, match="'bindings' parameter is required"):
            @xr_action(name="test")
            def f():
                pass

    def test_empty_bindings(self):
        """Test empty bindings raises error."""
        with pytest.raises(ValueError, match="'bindings' parameter is required"):
            @xr_action(name="test", bindings=[])
            def f():
                pass


# =============================================================================
# @xr_axis DECORATOR TESTS
# =============================================================================


class TestXRAxisDecorator:
    """Test @xr_axis decorator."""

    def test_basic_application(self):
        """Test basic decorator application."""
        @xr_axis(
            name="move_forward",
            positive=["xr_left_thumbstick_up"],
            negative=["xr_left_thumbstick_down"],
        )
        def on_move():
            pass

        assert on_move._xr_axis is True
        assert on_move._xr_axis_name == "move_forward"

    def test_positive_bindings(self):
        """Test positive bindings stored."""
        @xr_axis(
            name="test",
            positive=["xr_left_thumbstick_up", "xr_right_thumbstick_up"],
            negative=["xr_left_thumbstick_down"],
        )
        def f():
            pass

        assert f._xr_axis_positive == ["xr_left_thumbstick_up", "xr_right_thumbstick_up"]

    def test_negative_bindings(self):
        """Test negative bindings stored."""
        @xr_axis(
            name="test",
            positive=["xr_left_thumbstick_up"],
            negative=["xr_left_thumbstick_down", "xr_right_thumbstick_down"],
        )
        def f():
            pass

        assert f._xr_axis_negative == ["xr_left_thumbstick_down", "xr_right_thumbstick_down"]

    def test_deadzone(self):
        """Test deadzone parameter."""
        @xr_axis(
            name="test",
            positive=["xr_left_thumbstick_up"],
            negative=["xr_left_thumbstick_down"],
            deadzone=0.2,
        )
        def f():
            pass

        assert f._xr_axis_deadzone == 0.2

    def test_tags(self):
        """Test decorator tags."""
        @xr_axis(
            name="look_x",
            positive=["xr_right_thumbstick_right"],
            negative=["xr_right_thumbstick_left"],
        )
        def f():
            pass

        assert f._tags["xr_axis"] is True
        assert f._tags["xr_axis_name"] == "look_x"

    # --- Validation ---

    def test_missing_name(self):
        """Test missing name raises error."""
        with pytest.raises(ValueError, match="'name' parameter is required"):
            @xr_axis(
                positive=["xr_left_thumbstick_up"],
                negative=["xr_left_thumbstick_down"],
            )
            def f():
                pass

    def test_missing_positive(self):
        """Test missing positive raises error."""
        with pytest.raises(ValueError, match="'positive' parameter is required"):
            @xr_axis(
                name="test",
                negative=["xr_left_thumbstick_down"],
            )
            def f():
                pass

    def test_missing_negative(self):
        """Test missing negative raises error."""
        with pytest.raises(ValueError, match="'negative' parameter is required"):
            @xr_axis(
                name="test",
                positive=["xr_left_thumbstick_up"],
            )
            def f():
                pass


# =============================================================================
# ACTION REGISTRY TESTS
# =============================================================================


class TestXRActionRegistry:
    """Test XRActionRegistry."""

    def test_register_action(self):
        """Test registering an action."""
        registry = XRActionRegistry()
        registry.register_action(
            name="test_action",
            action_type=XRActionType.BOOLEAN,
            default_bindings=["xr_left_primary"],
        )

        action = registry.get_action("test_action")
        assert action is not None
        assert action.action_type == XRActionType.BOOLEAN

    def test_list_actions(self):
        """Test listing actions."""
        registry = XRActionRegistry()
        registry.register_action("action1", XRActionType.BOOLEAN, ["xr_left_primary"])
        registry.register_action("action2", XRActionType.FLOAT, ["xr_left_trigger"])

        actions = registry.list_actions()
        assert "action1" in actions
        assert "action2" in actions

    def test_add_handler(self):
        """Test adding action handler."""
        registry = XRActionRegistry()
        registry.register_action("handled_action", XRActionType.BOOLEAN, ["xr_left_primary"])

        def handler():
            pass

        registry.add_handler("handled_action", handler)
        handlers = registry.get_handlers("handled_action")

        assert handler in handlers

    def test_remove_handler(self):
        """Test removing action handler."""
        registry = XRActionRegistry()
        registry.register_action("test", XRActionType.BOOLEAN, ["xr_left_primary"])

        def handler():
            pass

        registry.add_handler("test", handler)
        registry.remove_handler("test", handler)

        handlers = registry.get_handlers("test")
        assert handler not in handlers


# =============================================================================
# INPUT PROFILE TESTS
# =============================================================================


class TestXRInputProfile:
    """Test XRInputProfile."""

    def test_create_profile(self):
        """Test creating an input profile."""
        profile = XRInputProfile(
            name="oculus_touch",
            vendor="Meta",
            controller_type="motion_controller",
        )

        assert profile.name == "oculus_touch"
        assert profile.vendor == "Meta"

    def test_add_binding(self):
        """Test adding binding to profile."""
        profile = XRInputProfile(name="test_profile")
        binding = XRActionBinding(
            action_name="grab",
            action_type=XRActionType.BOOLEAN,
            sources=["xr_left_grip"],
        )

        profile.add_binding(binding)

        assert profile.get_binding("grab") is binding

    def test_register_profile(self):
        """Test registering profile with registry."""
        registry = XRActionRegistry()
        profile = XRInputProfile(name="custom_profile")

        registry.register_profile(profile)

        assert "custom_profile" in registry.list_profiles()

    def test_set_active_profile(self):
        """Test setting active profile."""
        registry = XRActionRegistry()
        profile = XRInputProfile(name="active_profile")
        registry.register_profile(profile)

        result = registry.set_active_profile("active_profile")

        assert result is True

    def test_set_invalid_profile(self):
        """Test setting non-existent profile."""
        registry = XRActionRegistry()

        result = registry.set_active_profile("nonexistent")

        assert result is False


# =============================================================================
# BINDING RETRIEVAL TESTS
# =============================================================================


class TestBindingRetrieval:
    """Test action binding retrieval."""

    def test_get_binding_from_default(self):
        """Test getting binding from default actions."""
        registry = XRActionRegistry()
        registry.register_action(
            name="default_action",
            action_type=XRActionType.BOOLEAN,
            default_bindings=["xr_left_primary"],
            threshold=0.5,
        )

        binding = registry.get_binding_for_action("default_action")

        assert binding is not None
        assert binding.sources == ["xr_left_primary"]

    def test_get_binding_from_active_profile(self):
        """Test getting binding from active profile."""
        registry = XRActionRegistry()

        # Register default
        registry.register_action(
            name="overridden_action",
            action_type=XRActionType.BOOLEAN,
            default_bindings=["default_binding"],
        )

        # Create profile with override
        profile = XRInputProfile(name="override_profile")
        profile.add_binding(XRActionBinding(
            action_name="overridden_action",
            action_type=XRActionType.BOOLEAN,
            sources=["profile_binding"],
        ))

        registry.register_profile(profile)
        registry.set_active_profile("override_profile")

        binding = registry.get_binding_for_action("overridden_action")

        assert binding.sources == ["profile_binding"]


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_bind_action(self):
        """Test bind_action function."""
        # Register action first
        registry = get_xr_action_registry()
        registry.register_action("bind_test", XRActionType.BOOLEAN, ["xr_left_primary"])

        def handler():
            pass

        bind_action("bind_test", handler)

        handlers = registry.get_handlers("bind_test")
        assert handler in handlers

    def test_unbind_action(self):
        """Test unbind_action function."""
        registry = get_xr_action_registry()
        registry.register_action("unbind_test", XRActionType.BOOLEAN, ["xr_left_primary"])

        def handler():
            pass

        bind_action("unbind_test", handler)
        unbind_action("unbind_test", handler)

        handlers = registry.get_handlers("unbind_test")
        assert handler not in handlers

    def test_create_profile_function(self):
        """Test create_profile function."""
        profile = create_profile(
            name="func_profile",
            vendor="Test",
            controller_type="test_controller",
        )

        registry = get_xr_action_registry()
        assert "func_profile" in registry.list_profiles()


# =============================================================================
# GET ACTION VALUE TESTS
# =============================================================================


class TestGetActionValue:
    """Test get_action_value function."""

    def test_boolean_action_true(self):
        """Test boolean action returns True when pressed."""
        registry = get_xr_action_registry()
        registry.register_action(
            "bool_test",
            XRActionType.BOOLEAN,
            ["test_button"],
            threshold=0.5,
        )

        state = {"test_button": True}
        value = get_action_value("bool_test", state)

        assert value is True

    def test_boolean_action_false(self):
        """Test boolean action returns False when not pressed."""
        registry = get_xr_action_registry()
        registry.register_action(
            "bool_test_false",
            XRActionType.BOOLEAN,
            ["test_button2"],
        )

        state = {"test_button2": False}
        value = get_action_value("bool_test_false", state)

        assert value is False

    def test_boolean_from_analog(self):
        """Test boolean action from analog value with threshold."""
        registry = get_xr_action_registry()
        registry.register_action(
            "bool_analog",
            XRActionType.BOOLEAN,
            ["analog_source"],
            threshold=0.5,
        )

        state = {"analog_source": 0.7}
        value = get_action_value("bool_analog", state)

        assert value is True

        state = {"analog_source": 0.3}
        value = get_action_value("bool_analog", state)

        assert value is False

    def test_float_action(self):
        """Test float action returns analog value."""
        registry = get_xr_action_registry()
        registry.register_action(
            "float_test",
            XRActionType.FLOAT,
            ["float_source"],
        )

        state = {"float_source": 0.75}
        value = get_action_value("float_test", state)

        assert value == 0.75

    def test_vector2_action(self):
        """Test vector2 action returns tuple."""
        registry = get_xr_action_registry()
        registry.register_action(
            "vec2_test",
            XRActionType.VECTOR2,
            ["thumbstick"],
        )

        state = {"thumbstick": (0.5, -0.3)}
        value = get_action_value("vec2_test", state)

        assert value == (0.5, -0.3)

    def test_unknown_action(self):
        """Test unknown action returns None."""
        value = get_action_value("nonexistent_action", {})
        assert value is None


# =============================================================================
# DECORATOR STACKING TESTS
# =============================================================================


class TestDecoratorStacking:
    """Test decorator stacking."""

    def test_action_and_axis_stack(self):
        """Test @xr_action and @xr_axis can stack."""
        @xr_axis(
            name="combined_axis",
            positive=["xr_left_thumbstick_up"],
            negative=["xr_left_thumbstick_down"],
        )
        @xr_action(
            name="combined_action",
            bindings=["xr_left_primary"],
        )
        def combined_handler():
            pass

        assert combined_handler._xr_action is True
        assert combined_handler._xr_axis is True
        assert "xr_action" in combined_handler._applied_decorators
        assert "xr_axis" in combined_handler._applied_decorators


# =============================================================================
# XR INPUT SOURCE TESTS
# =============================================================================


class TestXRInputSource:
    """Test XRInputSource enum."""

    def test_button_sources(self):
        """Test button input sources."""
        assert XRInputSource.LEFT_TRIGGER.value == "xr_left_trigger"
        assert XRInputSource.RIGHT_GRIP.value == "xr_right_grip"
        assert XRInputSource.LEFT_PRIMARY.value == "xr_left_primary"

    def test_axis_sources(self):
        """Test axis input sources."""
        assert XRInputSource.LEFT_THUMBSTICK.value == "xr_left_thumbstick"
        assert XRInputSource.RIGHT_THUMBSTICK_X.value == "xr_right_thumbstick_x"

    def test_pose_sources(self):
        """Test pose input sources."""
        assert XRInputSource.LEFT_GRIP_POSE.value == "xr_left_grip_pose"
        assert XRInputSource.HEAD_POSE.value == "xr_head_pose"

    def test_touch_sources(self):
        """Test touch input sources."""
        assert XRInputSource.LEFT_TRIGGER_TOUCH.value == "xr_left_trigger_touch"
        assert XRInputSource.RIGHT_THUMBREST_TOUCH.value == "xr_right_thumbrest_touch"

    def test_haptic_sources(self):
        """Test haptic output sources."""
        assert XRInputSource.LEFT_HAPTIC.value == "xr_left_haptic"
        assert XRInputSource.RIGHT_HAPTIC.value == "xr_right_haptic"
