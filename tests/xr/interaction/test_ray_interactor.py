"""
Tests for Ray Interactor component (ray_interactor.py).

Tests the ray-based XR interactor for laser pointer interactions.
"""

import pytest

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.xr.interaction.interactable import (
    InteractionEvent,
    InteractionType,
    InteractorType,
    InteractionHit,
    XRInteractable,
    InteractableManager,
)
from engine.xr.interaction.grabbable import XRGrabbable, GrabType
from engine.xr.interaction.ray_interactor import (
    RayVisualMode,
    RayHitIndicator,
    RayConfig,
    RayState,
    RayCastResult,
    RayInteractor,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class ConcreteInteractable(XRInteractable):
    """Concrete implementation for testing."""
    pass


@pytest.fixture
def ray_interactor():
    """Create a basic ray interactor for testing."""
    return RayInteractor(interactor_id=1)


@pytest.fixture
def interactable_manager():
    """Create an interactable manager with test objects."""
    manager = InteractableManager()
    return manager


@pytest.fixture
def mock_raycast():
    """Create a mock raycast callback that returns hits."""
    def _create_callback(hits: list[RayCastResult]):
        def callback(origin: Vec3, direction: Vec3, max_distance: float):
            return hits
        return callback
    return _create_callback


# =============================================================================
# RayConfig Tests
# =============================================================================


class TestRayConfig:
    def test_default_values(self):
        config = RayConfig()

        assert config.max_distance == 10.0
        assert config.ray_width == 0.005
        assert config.visual_mode == RayVisualMode.LINE
        assert config.hit_indicator == RayHitIndicator.RETICLE
        assert config.select_threshold == 0.5
        assert config.grab_threshold == 0.7  # From XR_CONFIG.interaction.GRAB_ACTIVATION_THRESHOLD

    def test_custom_values(self):
        config = RayConfig(
            max_distance=20.0,
            visual_mode=RayVisualMode.CURVED,
            select_threshold=0.3
        )

        assert config.max_distance == 20.0
        assert config.visual_mode == RayVisualMode.CURVED
        assert config.select_threshold == 0.3


# =============================================================================
# RayInteractor Basic Tests
# =============================================================================


class TestRayInteractorBasic:
    def test_initialization(self):
        interactor = RayInteractor(interactor_id=42)

        assert interactor.interactor_id == 42
        assert interactor.is_active is True
        assert not interactor.is_hovering
        assert not interactor.is_selecting
        assert not interactor.is_grabbing

    def test_custom_config(self):
        config = RayConfig(max_distance=5.0)
        interactor = RayInteractor(interactor_id=1, config=config)

        assert interactor.config.max_distance == 5.0

    def test_deactivate(self):
        interactor = RayInteractor(interactor_id=1)
        interactor.is_active = False

        assert interactor.is_active is False


# =============================================================================
# Hover Tests
# =============================================================================


class TestRayHover:
    def test_hover_on_update(self, ray_interactor, mock_raycast):
        obj = ConcreteInteractable()
        hit = RayCastResult(
            hit=True,
            point=Vec3(0, 0, -1),
            normal=Vec3(0, 0, 1),
            distance=1.0,
            interactable=obj
        )

        ray_interactor.set_raycast_callback(mock_raycast([hit]))
        ray_interactor.update(Vec3.zero(), Vec3(0, 0, -1), 0.0, 0.0)

        assert ray_interactor.is_hovering
        assert ray_interactor.state.hovered_interactable is obj

    def test_hover_exit_on_move_away(self, ray_interactor, mock_raycast):
        obj = ConcreteInteractable()

        # First hover on object
        hit = RayCastResult(
            hit=True,
            point=Vec3(0, 0, -1),
            distance=1.0,
            interactable=obj
        )
        ray_interactor.set_raycast_callback(mock_raycast([hit]))
        ray_interactor.update(Vec3.zero(), Vec3(0, 0, -1), 0.0, 0.0)

        # Then move ray away (no hits)
        ray_interactor.set_raycast_callback(mock_raycast([]))
        ray_interactor.update(Vec3.zero(), Vec3(0, 1, 0), 0.0, 0.1)

        assert not ray_interactor.is_hovering
        assert ray_interactor.state.hovered_interactable is None

    def test_hover_priority(self, ray_interactor, mock_raycast):
        obj_low = ConcreteInteractable(priority=1)
        obj_high = ConcreteInteractable(priority=10)

        hits = [
            RayCastResult(hit=True, point=Vec3(0, 0, -1), distance=1.0, interactable=obj_low),
            RayCastResult(hit=True, point=Vec3(0, 0, -0.5), distance=0.5, interactable=obj_high),
        ]

        ray_interactor.set_raycast_callback(mock_raycast(hits))
        ray_interactor.update(Vec3.zero(), Vec3(0, 0, -1), 0.0, 0.0)

        # Higher priority should be hovered despite distance
        assert ray_interactor.state.hovered_interactable is obj_high


# =============================================================================
# Select Tests
# =============================================================================


class TestRaySelect:
    def test_select_on_trigger(self, ray_interactor, mock_raycast):
        obj = ConcreteInteractable()
        hit = RayCastResult(
            hit=True,
            point=Vec3(0, 0, -1),
            distance=1.0,
            interactable=obj
        )

        ray_interactor.set_raycast_callback(mock_raycast([hit]))

        # Update with trigger below threshold
        ray_interactor.update(Vec3.zero(), Vec3(0, 0, -1), 0.3, 0.0)
        assert not ray_interactor.is_selecting

        # Update with trigger above threshold
        ray_interactor.update(Vec3.zero(), Vec3(0, 0, -1), 0.6, 0.1)
        assert ray_interactor.is_selecting
        assert obj.is_selected

    def test_deselect_on_trigger_release(self, ray_interactor, mock_raycast):
        obj = ConcreteInteractable()
        hit = RayCastResult(
            hit=True,
            point=Vec3(0, 0, -1),
            distance=1.0,
            interactable=obj
        )

        ray_interactor.set_raycast_callback(mock_raycast([hit]))

        # Select
        ray_interactor.update(Vec3.zero(), Vec3(0, 0, -1), 0.6, 0.0)
        assert ray_interactor.is_selecting

        # Release trigger
        ray_interactor.update(Vec3.zero(), Vec3(0, 0, -1), 0.2, 0.1)
        assert not ray_interactor.is_selecting


# =============================================================================
# Grab Tests
# =============================================================================


class TestRayGrab:
    def test_grab_on_high_trigger(self, ray_interactor, mock_raycast):
        obj = XRGrabbable()
        hit = RayCastResult(
            hit=True,
            point=Vec3(0, 0, -1),
            distance=1.0,
            interactable=obj
        )

        ray_interactor.set_raycast_callback(mock_raycast([hit]))

        # Hover first
        ray_interactor.update(Vec3.zero(), Vec3(0, 0, -1), 0.0, 0.0)

        # Grab with high trigger
        ray_interactor.update(Vec3.zero(), Vec3(0, 0, -1), 0.9, 0.1)

        assert ray_interactor.is_grabbing
        assert obj.is_grabbed

    def test_release_grab(self, ray_interactor, mock_raycast):
        obj = XRGrabbable()
        hit = RayCastResult(
            hit=True,
            point=Vec3(0, 0, -1),
            distance=1.0,
            interactable=obj
        )

        ray_interactor.set_raycast_callback(mock_raycast([hit]))

        # Grab
        ray_interactor.update(Vec3.zero(), Vec3(0, 0, -1), 0.9, 0.0)
        assert ray_interactor.is_grabbing

        # Release
        ray_interactor.update(Vec3.zero(), Vec3(0, 0, -1), 0.3, 0.1)
        assert not ray_interactor.is_grabbing
        assert not obj.is_grabbed

    def test_cannot_grab_non_grabbable(self, ray_interactor, mock_raycast):
        obj = ConcreteInteractable()  # Not grabbable
        hit = RayCastResult(
            hit=True,
            point=Vec3(0, 0, -1),
            distance=1.0,
            interactable=obj
        )

        ray_interactor.set_raycast_callback(mock_raycast([hit]))
        ray_interactor.update(Vec3.zero(), Vec3(0, 0, -1), 0.9, 0.0)

        assert not ray_interactor.is_grabbing


# =============================================================================
# Layer Mask Tests
# =============================================================================


class TestRayLayerMask:
    def test_filter_by_layer(self, ray_interactor, mock_raycast):
        obj_ui = ConcreteInteractable(interaction_layers=["ui"])
        obj_world = ConcreteInteractable(interaction_layers=["world"])

        hits = [
            RayCastResult(hit=True, point=Vec3(0, 0, -1), distance=1.0, interactable=obj_ui),
            RayCastResult(hit=True, point=Vec3(0, 0, -2), distance=2.0, interactable=obj_world),
        ]

        ray_interactor.set_layer_mask(["world"])
        ray_interactor.set_raycast_callback(mock_raycast(hits))
        ray_interactor.update(Vec3.zero(), Vec3(0, 0, -1), 0.0, 0.0)

        # Should only hover world object
        assert ray_interactor.state.hovered_interactable is obj_world


# =============================================================================
# Callback Tests
# =============================================================================


class TestRayCallbacks:
    def test_hover_callback(self, ray_interactor, mock_raycast):
        events = []
        ray_interactor.add_hover_callback(lambda e: events.append(e))

        obj = ConcreteInteractable()
        hit = RayCastResult(
            hit=True,
            point=Vec3(0, 0, -1),
            distance=1.0,
            interactable=obj
        )

        ray_interactor.set_raycast_callback(mock_raycast([hit]))
        ray_interactor.update(Vec3.zero(), Vec3(0, 0, -1), 0.0, 0.0)

        assert len(events) == 1
        assert events[0].interaction_type == InteractionType.HOVER

    def test_select_callback(self, ray_interactor, mock_raycast):
        events = []
        ray_interactor.add_select_callback(lambda e: events.append(e))

        obj = ConcreteInteractable()
        hit = RayCastResult(
            hit=True,
            point=Vec3(0, 0, -1),
            distance=1.0,
            interactable=obj
        )

        ray_interactor.set_raycast_callback(mock_raycast([hit]))

        # Hover then select
        ray_interactor.update(Vec3.zero(), Vec3(0, 0, -1), 0.0, 0.0)
        ray_interactor.update(Vec3.zero(), Vec3(0, 0, -1), 0.6, 0.1)

        assert len(events) == 1
        assert events[0].interaction_type == InteractionType.SELECT

    def test_grab_callback(self, ray_interactor, mock_raycast):
        events = []
        ray_interactor.add_grab_callback(lambda e: events.append(e))

        obj = XRGrabbable()
        hit = RayCastResult(
            hit=True,
            point=Vec3(0, 0, -1),
            distance=1.0,
            interactable=obj
        )

        ray_interactor.set_raycast_callback(mock_raycast([hit]))
        ray_interactor.update(Vec3.zero(), Vec3(0, 0, -1), 0.9, 0.0)

        assert len(events) == 1
        assert events[0].interaction_type == InteractionType.GRAB


# =============================================================================
# Ray Visual Tests
# =============================================================================


class TestRayVisual:
    def test_get_ray_points_line(self, ray_interactor, mock_raycast):
        ray_interactor.update(Vec3.zero(), Vec3(0, 0, -1), 0.0, 0.0)

        points = ray_interactor.get_ray_points()

        assert len(points) == 2
        assert points[0] == Vec3.zero()

    def test_get_ray_points_hidden(self):
        config = RayConfig(visual_mode=RayVisualMode.HIDDEN)
        interactor = RayInteractor(interactor_id=1, config=config)
        interactor.update(Vec3.zero(), Vec3(0, 0, -1), 0.0, 0.0)

        points = interactor.get_ray_points()

        assert len(points) == 0

    def test_get_ray_points_curved(self, mock_raycast):
        config = RayConfig(visual_mode=RayVisualMode.CURVED, curve_points=10)
        interactor = RayInteractor(interactor_id=1, config=config)
        interactor.update(Vec3.zero(), Vec3(0, 0, -1), 0.0, 0.0)

        points = interactor.get_ray_points()

        assert len(points) == 10


# =============================================================================
# Hit Point Tests
# =============================================================================


class TestRayHitPoint:
    def test_current_hit_point_with_hit(self, ray_interactor, mock_raycast):
        obj = ConcreteInteractable()
        hit = RayCastResult(
            hit=True,
            point=Vec3(1, 2, 3),
            distance=1.0,
            interactable=obj
        )

        ray_interactor.set_raycast_callback(mock_raycast([hit]))
        ray_interactor.update(Vec3.zero(), Vec3(0, 0, -1), 0.0, 0.0)

        assert ray_interactor.current_hit_point == Vec3(1, 2, 3)

    def test_current_hit_point_no_hit(self, ray_interactor, mock_raycast):
        ray_interactor.set_raycast_callback(mock_raycast([]))
        ray_interactor.update(Vec3.zero(), Vec3(0, 0, -1), 0.0, 0.0)

        assert ray_interactor.current_hit_point is None


# =============================================================================
# Deactivation Tests
# =============================================================================


class TestRayDeactivation:
    def test_deactivation_releases_all(self, ray_interactor, mock_raycast):
        obj = XRGrabbable()
        hit = RayCastResult(
            hit=True,
            point=Vec3(0, 0, -1),
            distance=1.0,
            interactable=obj
        )

        ray_interactor.set_raycast_callback(mock_raycast([hit]))

        # Grab object
        ray_interactor.update(Vec3.zero(), Vec3(0, 0, -1), 0.9, 0.0)
        assert ray_interactor.is_grabbing

        # Deactivate
        ray_interactor.is_active = False

        assert not ray_interactor.is_grabbing
        assert not ray_interactor.is_hovering


# =============================================================================
# Activation Tests
# =============================================================================


class TestRayActivation:
    def test_activate_target(self, ray_interactor, mock_raycast):
        obj = ConcreteInteractable()
        activate_events = []

        def callback(event):
            activate_events.append(event)

        obj.add_callback(InteractionType.ACTIVATE, callback)

        ray_interactor.activate(obj)

        assert len(activate_events) == 1
