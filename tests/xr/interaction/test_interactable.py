"""
Tests for XR Interactable component (interactable.py).

Tests the base XR interactable component and decorator:
    XRInteractable, @xr_interactable
"""

import pytest

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.xr.interaction.interactable import (
    InteractionState,
    InteractionType,
    InteractorType,
    InteractionEvent,
    InteractionHit,
    XRInteractable,
    InteractableManager,
    xr_interactable,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def create_event():
    """Factory fixture for creating test events."""
    def _create_event(
        interaction_type: InteractionType = InteractionType.HOVER,
        interactor_id: int = 1,
        timestamp: float = 0.0
    ) -> InteractionEvent:
        return InteractionEvent(
            interactor_type=InteractorType.RAY,
            interactor_id=interactor_id,
            interaction_type=interaction_type,
            position=Vec3(0, 0, 0),
            rotation=Quat.identity(),
            timestamp=timestamp
        )
    return _create_event


class ConcreteInteractable(XRInteractable):
    """Concrete implementation for testing."""
    pass


# =============================================================================
# @xr_interactable Decorator Tests
# =============================================================================


class TestXRInteractableDecorator:
    def test_basic_application(self):
        @xr_interactable()
        class TestObject(XRInteractable):
            pass

        assert TestObject._xr_interactable is True

    def test_interaction_layers(self):
        @xr_interactable(interaction_layers=["ui", "world"])
        class TestObject(XRInteractable):
            pass

        assert TestObject._interaction_layers == ["ui", "world"]

    def test_default_layers(self):
        @xr_interactable()
        class TestObject(XRInteractable):
            pass

        assert TestObject._interaction_layers == ["default"]

    def test_priority(self):
        @xr_interactable(priority=10)
        class TestObject(XRInteractable):
            pass

        assert TestObject._interaction_priority == 10

    def test_enabled_default(self):
        @xr_interactable(enabled=False)
        class TestObject(XRInteractable):
            pass

        assert TestObject._interaction_enabled_default is False

    def test_applied_decorators_tracked(self):
        @xr_interactable()
        class TestObject(XRInteractable):
            pass

        assert 'xr_interactable' in TestObject._applied_decorators

    def test_tags_stored(self):
        @xr_interactable(interaction_layers=["special"], priority=5)
        class TestObject(XRInteractable):
            pass

        assert TestObject._class_tags['xr_interactable'] is True
        assert TestObject._class_tags['interaction_layers'] == ["special"]
        assert TestObject._class_tags['interaction_priority'] == 5


# =============================================================================
# XRInteractable Basic Tests
# =============================================================================


class TestXRInteractableBasic:
    def test_initialization(self):
        obj = ConcreteInteractable(entity_id=42)
        assert obj.entity_id == 42
        assert obj.state == InteractionState.IDLE
        assert not obj.is_hovered
        assert not obj.is_selected
        assert not obj.is_grabbed

    def test_default_layers(self):
        obj = ConcreteInteractable()
        assert obj.interaction_layers == ["default"]

    def test_custom_layers(self):
        obj = ConcreteInteractable(interaction_layers=["ui", "world"])
        assert "ui" in obj.interaction_layers
        assert "world" in obj.interaction_layers

    def test_is_in_layer(self):
        obj = ConcreteInteractable(interaction_layers=["ui", "world"])
        assert obj.is_in_layer("ui")
        assert obj.is_in_layer("world")
        assert not obj.is_in_layer("enemy")

    def test_priority(self):
        obj = ConcreteInteractable(priority=5)
        assert obj.priority == 5

    def test_enabled_by_default(self):
        obj = ConcreteInteractable()
        assert obj.enabled is True

    def test_disable(self):
        obj = ConcreteInteractable()
        obj.enabled = False
        assert obj.enabled is False


# =============================================================================
# Hover Interaction Tests
# =============================================================================


class TestHoverInteraction:
    def test_hover_enter(self, create_event):
        obj = ConcreteInteractable()
        event = create_event(InteractionType.HOVER, interactor_id=1)

        obj.on_hover_enter(1, event)

        assert obj.is_hovered
        assert obj.state == InteractionState.HOVERED
        assert 1 in obj.get_hovering_interactors()

    def test_hover_exit(self, create_event):
        obj = ConcreteInteractable()
        enter_event = create_event(InteractionType.HOVER, interactor_id=1)
        exit_event = create_event(InteractionType.HOVER, interactor_id=1)

        obj.on_hover_enter(1, enter_event)
        obj.on_hover_exit(1, exit_event)

        assert not obj.is_hovered
        assert obj.state == InteractionState.IDLE
        assert 1 not in obj.get_hovering_interactors()

    def test_multiple_hovers(self, create_event):
        obj = ConcreteInteractable()

        obj.on_hover_enter(1, create_event(interactor_id=1))
        obj.on_hover_enter(2, create_event(interactor_id=2))

        assert obj.is_hovered
        assert len(obj.get_hovering_interactors()) == 2

        obj.on_hover_exit(1, create_event(interactor_id=1))

        assert obj.is_hovered  # Still hovered by interactor 2
        assert len(obj.get_hovering_interactors()) == 1

    def test_hover_duration(self, create_event):
        obj = ConcreteInteractable()
        event = create_event(timestamp=1.0)

        obj.on_hover_enter(1, event)

        duration = obj.get_hover_duration(2.5)
        assert duration == pytest.approx(1.5, rel=0.01)

    def test_hover_duration_not_hovering(self):
        obj = ConcreteInteractable()
        assert obj.get_hover_duration(1.0) == 0.0

    def test_hover_disabled(self, create_event):
        obj = ConcreteInteractable(enabled=False)
        event = create_event()

        obj.on_hover_enter(1, event)

        assert not obj.is_hovered


# =============================================================================
# Select Interaction Tests
# =============================================================================


class TestSelectInteraction:
    def test_select_enter(self, create_event):
        obj = ConcreteInteractable()
        event = create_event(InteractionType.SELECT, interactor_id=1)

        obj.on_select_enter(1, event)

        assert obj.is_selected
        assert obj.state == InteractionState.SELECTED

    def test_select_exit(self, create_event):
        obj = ConcreteInteractable()

        obj.on_select_enter(1, create_event(InteractionType.SELECT))
        obj.on_select_exit(1, create_event(InteractionType.SELECT))

        assert not obj.is_selected
        assert obj.state == InteractionState.IDLE

    def test_select_duration(self, create_event):
        obj = ConcreteInteractable()
        event = create_event(InteractionType.SELECT, timestamp=0.0)

        obj.on_select_enter(1, event)

        duration = obj.get_select_duration(1.0)
        assert duration == pytest.approx(1.0, rel=0.01)

    def test_select_disabled(self, create_event):
        obj = ConcreteInteractable(enabled=False)

        obj.on_select_enter(1, create_event(InteractionType.SELECT))

        assert not obj.is_selected


# =============================================================================
# Grab Interaction Tests
# =============================================================================


class TestGrabInteraction:
    def test_grab_enter(self, create_event):
        obj = ConcreteInteractable()
        event = create_event(InteractionType.GRAB)

        result = obj.on_grab_enter(1, event)

        assert result is True
        assert obj.is_grabbed
        assert obj.state == InteractionState.GRABBED
        assert obj.get_grabbing_interactor() == 1

    def test_grab_exit(self, create_event):
        obj = ConcreteInteractable()

        obj.on_grab_enter(1, create_event(InteractionType.GRAB))
        obj.on_grab_exit(1, create_event(InteractionType.GRAB))

        assert not obj.is_grabbed
        assert obj.get_grabbing_interactor() is None

    def test_grab_only_one(self, create_event):
        obj = ConcreteInteractable()

        obj.on_grab_enter(1, create_event(InteractionType.GRAB))
        result = obj.on_grab_enter(2, create_event(InteractionType.GRAB))

        assert result is False
        assert obj.get_grabbing_interactor() == 1

    def test_grab_wrong_interactor_exit(self, create_event):
        obj = ConcreteInteractable()

        obj.on_grab_enter(1, create_event(InteractionType.GRAB))
        obj.on_grab_exit(2, create_event(InteractionType.GRAB))  # Wrong interactor

        assert obj.is_grabbed  # Still grabbed by interactor 1

    def test_grab_disabled(self, create_event):
        obj = ConcreteInteractable(enabled=False)

        result = obj.on_grab_enter(1, create_event(InteractionType.GRAB))

        assert result is False
        assert not obj.is_grabbed


# =============================================================================
# State Priority Tests
# =============================================================================


class TestStatePriority:
    def test_grab_overrides_select(self, create_event):
        obj = ConcreteInteractable()

        obj.on_select_enter(1, create_event(InteractionType.SELECT))
        obj.on_grab_enter(2, create_event(InteractionType.GRAB))

        assert obj.state == InteractionState.GRABBED

    def test_select_overrides_hover(self, create_event):
        obj = ConcreteInteractable()

        obj.on_hover_enter(1, create_event(InteractionType.HOVER))
        obj.on_select_enter(2, create_event(InteractionType.SELECT))

        assert obj.state == InteractionState.SELECTED

    def test_state_degrades_on_release(self, create_event):
        obj = ConcreteInteractable()

        obj.on_hover_enter(1, create_event(InteractionType.HOVER))
        obj.on_select_enter(1, create_event(InteractionType.SELECT))
        obj.on_grab_enter(1, create_event(InteractionType.GRAB))

        assert obj.state == InteractionState.GRABBED

        obj.on_grab_exit(1, create_event(InteractionType.GRAB))
        assert obj.state == InteractionState.SELECTED

        obj.on_select_exit(1, create_event(InteractionType.SELECT))
        assert obj.state == InteractionState.HOVERED

        obj.on_hover_exit(1, create_event(InteractionType.HOVER))
        assert obj.state == InteractionState.IDLE


# =============================================================================
# Callback Tests
# =============================================================================


class TestCallbacks:
    def test_add_callback(self, create_event):
        obj = ConcreteInteractable()
        received_events = []

        def callback(event):
            received_events.append(event)

        obj.add_callback(InteractionType.HOVER, callback)
        obj.on_hover_enter(1, create_event(InteractionType.HOVER))

        assert len(received_events) == 1

    def test_remove_callback(self, create_event):
        obj = ConcreteInteractable()
        received_events = []

        def callback(event):
            received_events.append(event)

        obj.add_callback(InteractionType.HOVER, callback)
        obj.remove_callback(InteractionType.HOVER, callback)
        obj.on_hover_enter(1, create_event(InteractionType.HOVER))

        assert len(received_events) == 0

    def test_multiple_callbacks(self, create_event):
        obj = ConcreteInteractable()
        counter = [0]

        def callback1(event):
            counter[0] += 1

        def callback2(event):
            counter[0] += 10

        obj.add_callback(InteractionType.HOVER, callback1)
        obj.add_callback(InteractionType.HOVER, callback2)
        obj.on_hover_enter(1, create_event(InteractionType.HOVER))

        assert counter[0] == 11

    def test_callback_error_doesnt_break(self, create_event):
        obj = ConcreteInteractable()
        received_events = []

        def bad_callback(event):
            raise RuntimeError("Oops")

        def good_callback(event):
            received_events.append(event)

        obj.add_callback(InteractionType.HOVER, bad_callback)
        obj.add_callback(InteractionType.HOVER, good_callback)

        # Should not raise
        obj.on_hover_enter(1, create_event(InteractionType.HOVER))

        # Good callback still called
        assert len(received_events) == 1


# =============================================================================
# InteractableManager Tests
# =============================================================================


class TestInteractableManager:
    def test_register(self):
        manager = InteractableManager()
        obj = ConcreteInteractable()

        interactable_id = manager.register(obj)

        assert manager.get(interactable_id) is obj

    def test_unregister(self):
        manager = InteractableManager()
        obj = ConcreteInteractable()

        interactable_id = manager.register(obj)
        manager.unregister(interactable_id)

        assert manager.get(interactable_id) is None

    def test_get_by_layer(self):
        manager = InteractableManager()
        obj1 = ConcreteInteractable(interaction_layers=["ui"])
        obj2 = ConcreteInteractable(interaction_layers=["world"])
        obj3 = ConcreteInteractable(interaction_layers=["ui", "world"])

        manager.register(obj1)
        manager.register(obj2)
        manager.register(obj3)

        ui_objects = manager.get_by_layer("ui")
        assert obj1 in ui_objects
        assert obj2 not in ui_objects
        assert obj3 in ui_objects

    def test_get_sorted_by_priority(self, create_event):
        manager = InteractableManager()
        obj_low = ConcreteInteractable(priority=1)
        obj_high = ConcreteInteractable(priority=10)

        manager.register(obj_low)
        manager.register(obj_high)

        hits = [
            InteractionHit(obj_low, Vec3.zero(), Vec3.up(), 1.0),
            InteractionHit(obj_high, Vec3.zero(), Vec3.up(), 1.0)
        ]

        sorted_hits = manager.get_sorted_by_priority(hits)

        assert sorted_hits[0].interactable is obj_high
        assert sorted_hits[1].interactable is obj_low

    def test_filter_by_layer_mask(self):
        manager = InteractableManager()
        obj_ui = ConcreteInteractable(interaction_layers=["ui"])
        obj_world = ConcreteInteractable(interaction_layers=["world"])

        manager.register(obj_ui)
        manager.register(obj_world)

        hits = [
            InteractionHit(obj_ui, Vec3.zero(), Vec3.up(), 1.0),
            InteractionHit(obj_world, Vec3.zero(), Vec3.up(), 1.0)
        ]

        sorted_hits = manager.get_sorted_by_priority(hits, layer_mask=["ui"])

        assert len(sorted_hits) == 1
        assert sorted_hits[0].interactable is obj_ui


# =============================================================================
# Activation Tests
# =============================================================================


class TestActivation:
    def test_activate(self, create_event):
        obj = ConcreteInteractable()
        received_events = []

        def callback(event):
            received_events.append(event)

        obj.add_callback(InteractionType.ACTIVATE, callback)
        obj.on_activate(create_event(InteractionType.ACTIVATE))

        assert len(received_events) == 1

    def test_activate_disabled(self, create_event):
        obj = ConcreteInteractable(enabled=False)
        received_events = []

        def callback(event):
            received_events.append(event)

        obj.add_callback(InteractionType.ACTIVATE, callback)
        obj.on_activate(create_event(InteractionType.ACTIVATE))

        assert len(received_events) == 0


# =============================================================================
# Disable Clears State Tests
# =============================================================================


class TestDisableClearsState:
    def test_disable_clears_hover(self, create_event):
        obj = ConcreteInteractable()
        obj.on_hover_enter(1, create_event(InteractionType.HOVER))

        obj.enabled = False

        assert not obj.is_hovered
        assert obj.state == InteractionState.IDLE

    def test_disable_clears_select(self, create_event):
        obj = ConcreteInteractable()
        obj.on_select_enter(1, create_event(InteractionType.SELECT))

        obj.enabled = False

        assert not obj.is_selected

    def test_disable_clears_grab(self, create_event):
        obj = ConcreteInteractable()
        obj.on_grab_enter(1, create_event(InteractionType.GRAB))

        obj.enabled = False

        assert not obj.is_grabbed
        assert obj.get_grabbing_interactor() is None
