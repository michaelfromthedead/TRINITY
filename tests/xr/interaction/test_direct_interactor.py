"""
Tests for Direct Interactor component (direct_interactor.py).

Tests the direct/poke XR interactor for touch-based interactions.
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
)
from engine.xr.interaction.grabbable import XRGrabbable
from engine.xr.interaction.direct_interactor import (
    PokeMode,
    GrabDetection,
    DirectConfig,
    ContactPoint,
    DirectState,
    DirectInteractor,
    MultiPointDirectInteractor,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class ConcreteInteractable(XRInteractable):
    """Concrete implementation for testing."""
    pass


@pytest.fixture
def direct_interactor():
    """Create a basic direct interactor for testing."""
    return DirectInteractor(interactor_id=1)


@pytest.fixture
def mock_collision():
    """Create a mock collision callback."""
    def _create_callback(hits: list[InteractionHit]):
        def callback(position: Vec3, radius: float):
            return hits
        return callback
    return _create_callback


# =============================================================================
# DirectConfig Tests
# =============================================================================


class TestDirectConfig:
    def test_default_values(self):
        config = DirectConfig()

        assert config.poke_mode == PokeMode.FINGERTIP
        assert config.grab_detection == GrabDetection.PINCH
        assert config.interaction_radius == 0.02
        assert config.poke_depth_threshold == 0.01
        assert config.grab_threshold == 0.7
        assert config.hover_distance == 0.05

    def test_custom_values(self):
        config = DirectConfig(
            poke_mode=PokeMode.SPHERE,
            grab_detection=GrabDetection.GRIP,
            interaction_radius=0.05
        )

        assert config.poke_mode == PokeMode.SPHERE
        assert config.grab_detection == GrabDetection.GRIP
        assert config.interaction_radius == 0.05


# =============================================================================
# DirectInteractor Basic Tests
# =============================================================================


class TestDirectInteractorBasic:
    def test_initialization(self):
        interactor = DirectInteractor(interactor_id=42)

        assert interactor.interactor_id == 42
        assert interactor.is_active is True
        assert not interactor.is_hovering
        assert not interactor.is_poking
        assert not interactor.is_grabbing

    def test_custom_config(self):
        config = DirectConfig(hover_distance=0.1)
        interactor = DirectInteractor(interactor_id=1, config=config)

        assert interactor.config.hover_distance == 0.1

    def test_deactivate(self):
        interactor = DirectInteractor(interactor_id=1)
        interactor.is_active = False

        assert interactor.is_active is False


# =============================================================================
# Hover Tests
# =============================================================================


class TestDirectHover:
    def test_hover_on_approach(self, direct_interactor, mock_collision):
        obj = ConcreteInteractable()
        hit = InteractionHit(
            interactable=obj,
            hit_point=Vec3(0, 0, 0),
            hit_normal=Vec3(0, 1, 0),
            distance=0.03  # Within hover distance
        )

        direct_interactor.set_collision_callback(mock_collision([hit]))
        direct_interactor.update(Vec3.zero(), Quat.identity(), 0.0, 0.0, 0.0)

        assert direct_interactor.is_hovering
        assert direct_interactor.state.hovered_interactable is obj

    def test_no_hover_when_too_far(self, direct_interactor, mock_collision):
        obj = ConcreteInteractable()
        hit = InteractionHit(
            interactable=obj,
            hit_point=Vec3(0, 0, 0),
            hit_normal=Vec3(0, 1, 0),
            distance=0.1  # Beyond hover distance (default 0.05)
        )

        direct_interactor.set_collision_callback(mock_collision([hit]))
        direct_interactor.update(Vec3.zero(), Quat.identity(), 0.0, 0.0, 0.0)

        assert not direct_interactor.is_hovering


# =============================================================================
# Poke Tests
# =============================================================================


class TestDirectPoke:
    def test_poke_on_penetration(self, direct_interactor, mock_collision):
        obj = ConcreteInteractable()
        hit = InteractionHit(
            interactable=obj,
            hit_point=Vec3(0, 0, 0),
            hit_normal=Vec3(0, 1, 0),
            distance=-0.02  # Negative = penetrating
        )

        direct_interactor.set_collision_callback(mock_collision([hit]))
        direct_interactor.update(Vec3.zero(), Quat.identity(), 0.0, 0.0, 0.0)

        assert direct_interactor.is_poking
        assert direct_interactor.state.poked_interactable is obj

    def test_poke_ends_on_exit(self, direct_interactor, mock_collision):
        obj = ConcreteInteractable()

        # Start poking
        hit_poke = InteractionHit(
            interactable=obj,
            hit_point=Vec3(0, 0, 0),
            hit_normal=Vec3(0, 1, 0),
            distance=-0.02
        )
        direct_interactor.set_collision_callback(mock_collision([hit_poke]))
        direct_interactor.update(Vec3.zero(), Quat.identity(), 0.0, 0.0, 0.0)
        assert direct_interactor.is_poking

        # Exit poke (no penetration)
        hit_hover = InteractionHit(
            interactable=obj,
            hit_point=Vec3(0, 0, 0),
            hit_normal=Vec3(0, 1, 0),
            distance=0.03
        )
        direct_interactor.set_collision_callback(mock_collision([hit_hover]))
        direct_interactor.update(Vec3.zero(), Quat.identity(), 0.0, 0.0, 0.1)

        assert not direct_interactor.is_poking

    def test_poke_progress(self, direct_interactor, mock_collision):
        obj = ConcreteInteractable()
        hit = InteractionHit(
            interactable=obj,
            hit_point=Vec3(0, 0, 0),
            hit_normal=Vec3(0, 1, 0),
            distance=-0.015  # Penetrating
        )

        direct_interactor.set_collision_callback(mock_collision([hit]))
        direct_interactor.update(Vec3.zero(), Quat.identity(), 0.0, 0.0, 0.0)

        progress = direct_interactor.get_poke_progress()
        assert 0.0 < progress < 1.0


# =============================================================================
# Grab Tests
# =============================================================================


class TestDirectGrab:
    def test_grab_with_pinch(self, direct_interactor, mock_collision):
        obj = XRGrabbable()
        hit = InteractionHit(
            interactable=obj,
            hit_point=Vec3(0, 0, 0),
            hit_normal=Vec3(0, 1, 0),
            distance=0.01
        )

        direct_interactor.set_collision_callback(mock_collision([hit]))

        # Update with high pinch value
        direct_interactor.update(Vec3.zero(), Quat.identity(), 0.8, 0.0, 0.0)

        assert direct_interactor.is_grabbing
        assert obj.is_grabbed

    def test_grab_with_grip(self, mock_collision):
        config = DirectConfig(grab_detection=GrabDetection.GRIP)
        interactor = DirectInteractor(interactor_id=1, config=config)

        obj = XRGrabbable()
        hit = InteractionHit(
            interactable=obj,
            hit_point=Vec3(0, 0, 0),
            hit_normal=Vec3(0, 1, 0),
            distance=0.01
        )

        interactor.set_collision_callback(mock_collision([hit]))

        # Update with high grip value (not pinch)
        interactor.update(Vec3.zero(), Quat.identity(), 0.0, 0.8, 0.0)

        assert interactor.is_grabbing

    def test_release_grab(self, mock_collision):
        # Use non-sticky grab config for this test
        config = DirectConfig(sticky_grab=False)
        interactor = DirectInteractor(interactor_id=1, config=config)

        obj = XRGrabbable()
        hit = InteractionHit(
            interactable=obj,
            hit_point=Vec3(0, 0, 0),
            hit_normal=Vec3(0, 1, 0),
            distance=0.01
        )

        interactor.set_collision_callback(mock_collision([hit]))

        # Grab
        interactor.update(Vec3.zero(), Quat.identity(), 0.8, 0.0, 0.0)
        assert interactor.is_grabbing

        # Release
        interactor.update(Vec3.zero(), Quat.identity(), 0.1, 0.0, 0.1)
        assert not interactor.is_grabbing

    def test_cannot_grab_non_grabbable(self, direct_interactor, mock_collision):
        obj = ConcreteInteractable()  # Not grabbable
        hit = InteractionHit(
            interactable=obj,
            hit_point=Vec3(0, 0, 0),
            hit_normal=Vec3(0, 1, 0),
            distance=0.01
        )

        direct_interactor.set_collision_callback(mock_collision([hit]))
        direct_interactor.update(Vec3.zero(), Quat.identity(), 0.8, 0.0, 0.0)

        assert not direct_interactor.is_grabbing


# =============================================================================
# Velocity Tests
# =============================================================================


class TestDirectVelocity:
    def test_velocity_calculated(self, direct_interactor, mock_collision):
        direct_interactor.set_collision_callback(mock_collision([]))

        # First update at t=1.0
        direct_interactor.update(Vec3(0, 0, 0), Quat.identity(), 0.0, 0.0, 1.0)

        # Second update at t=1.1 with position change
        direct_interactor.update(Vec3(1, 0, 0), Quat.identity(), 0.0, 0.0, 1.1)

        # Velocity should be approximately 10 m/s in X (1m / 0.1s)
        assert direct_interactor.state.velocity.x == pytest.approx(10.0, rel=0.1)


# =============================================================================
# Callback Tests
# =============================================================================


class TestDirectCallbacks:
    def test_hover_callback(self, direct_interactor, mock_collision):
        events = []
        direct_interactor.add_hover_callback(lambda e: events.append(e))

        obj = ConcreteInteractable()
        hit = InteractionHit(
            interactable=obj,
            hit_point=Vec3(0, 0, 0),
            hit_normal=Vec3(0, 1, 0),
            distance=0.03
        )

        direct_interactor.set_collision_callback(mock_collision([hit]))
        direct_interactor.update(Vec3.zero(), Quat.identity(), 0.0, 0.0, 0.0)

        assert len(events) == 1
        assert events[0].interactor_type == InteractorType.DIRECT

    def test_poke_callback(self, direct_interactor, mock_collision):
        events = []
        direct_interactor.add_poke_callback(lambda e: events.append(e))

        obj = ConcreteInteractable()
        hit = InteractionHit(
            interactable=obj,
            hit_point=Vec3(0, 0, 0),
            hit_normal=Vec3(0, 1, 0),
            distance=-0.02
        )

        direct_interactor.set_collision_callback(mock_collision([hit]))
        direct_interactor.update(Vec3.zero(), Quat.identity(), 0.0, 0.0, 0.0)

        assert len(events) == 1

    def test_grab_callback(self, direct_interactor, mock_collision):
        events = []
        direct_interactor.add_grab_callback(lambda e: events.append(e))

        obj = XRGrabbable()
        hit = InteractionHit(
            interactable=obj,
            hit_point=Vec3(0, 0, 0),
            hit_normal=Vec3(0, 1, 0),
            distance=0.01
        )

        direct_interactor.set_collision_callback(mock_collision([hit]))
        direct_interactor.update(Vec3.zero(), Quat.identity(), 0.8, 0.0, 0.0)

        assert len(events) == 1


# =============================================================================
# Layer Mask Tests
# =============================================================================


class TestDirectLayerMask:
    def test_filter_by_layer(self, direct_interactor, mock_collision):
        obj_ui = ConcreteInteractable(interaction_layers=["ui"])
        obj_world = ConcreteInteractable(interaction_layers=["world"])

        hits = [
            InteractionHit(interactable=obj_ui, hit_point=Vec3.zero(),
                          hit_normal=Vec3.up(), distance=0.03),
            InteractionHit(interactable=obj_world, hit_point=Vec3.zero(),
                          hit_normal=Vec3.up(), distance=0.04),
        ]

        direct_interactor.set_layer_mask(["world"])
        direct_interactor.set_collision_callback(mock_collision(hits))
        direct_interactor.update(Vec3.zero(), Quat.identity(), 0.0, 0.0, 0.0)

        # Should only hover world object
        assert direct_interactor.state.hovered_interactable is obj_world


# =============================================================================
# Force Release Tests
# =============================================================================


class TestDirectForceRelease:
    def test_force_release(self, direct_interactor, mock_collision):
        obj = XRGrabbable()
        hit = InteractionHit(
            interactable=obj,
            hit_point=Vec3(0, 0, 0),
            hit_normal=Vec3(0, 1, 0),
            distance=0.01
        )

        direct_interactor.set_collision_callback(mock_collision([hit]))

        # Grab
        direct_interactor.update(Vec3.zero(), Quat.identity(), 0.8, 0.0, 0.0)
        assert direct_interactor.is_grabbing

        # Force release
        direct_interactor.force_release()
        assert not direct_interactor.is_grabbing


# =============================================================================
# Deactivation Tests
# =============================================================================


class TestDirectDeactivation:
    def test_deactivation_releases_all(self, direct_interactor, mock_collision):
        obj = XRGrabbable()
        hit = InteractionHit(
            interactable=obj,
            hit_point=Vec3(0, 0, 0),
            hit_normal=Vec3(0, 1, 0),
            distance=0.01
        )

        direct_interactor.set_collision_callback(mock_collision([hit]))

        # Grab
        direct_interactor.update(Vec3.zero(), Quat.identity(), 0.8, 0.0, 0.0)
        assert direct_interactor.is_grabbing

        # Deactivate
        direct_interactor.is_active = False

        assert not direct_interactor.is_grabbing
        assert not direct_interactor.is_hovering


# =============================================================================
# MultiPointDirectInteractor Tests
# =============================================================================


class TestMultiPointDirectInteractor:
    def test_initialization(self):
        interactor = MultiPointDirectInteractor(interactor_id=1, max_contacts=5)

        assert len(interactor.get_active_contacts()) == 0

    def test_add_contact(self):
        interactor = MultiPointDirectInteractor(interactor_id=1)

        interactor.update_contact(0, Vec3.zero(), Quat.identity(), 0.0, 0.0, 0.0)

        contacts = interactor.get_active_contacts()
        assert len(contacts) == 1

    def test_multiple_contacts(self):
        interactor = MultiPointDirectInteractor(interactor_id=1)

        interactor.update_contact(0, Vec3(0, 0, 0), Quat.identity(), 0.0, 0.0, 0.0)
        interactor.update_contact(1, Vec3(1, 0, 0), Quat.identity(), 0.0, 0.0, 0.0)
        interactor.update_contact(2, Vec3(2, 0, 0), Quat.identity(), 0.0, 0.0, 0.0)

        contacts = interactor.get_active_contacts()
        assert len(contacts) == 3

    def test_remove_contact(self):
        interactor = MultiPointDirectInteractor(interactor_id=1)

        interactor.update_contact(0, Vec3.zero(), Quat.identity(), 0.0, 0.0, 0.0)
        interactor.update_contact(1, Vec3.zero(), Quat.identity(), 0.0, 0.0, 0.0)

        interactor.remove_contact(0)

        contacts = interactor.get_active_contacts()
        assert len(contacts) == 1

    def test_release_all(self):
        interactor = MultiPointDirectInteractor(interactor_id=1)

        interactor.update_contact(0, Vec3.zero(), Quat.identity(), 0.0, 0.0, 0.0)
        interactor.update_contact(1, Vec3.zero(), Quat.identity(), 0.0, 0.0, 0.0)

        interactor.release_all()

        contacts = interactor.get_active_contacts()
        assert len(contacts) == 0

    def test_max_contacts_respected(self):
        interactor = MultiPointDirectInteractor(interactor_id=1, max_contacts=2)

        interactor.update_contact(0, Vec3.zero(), Quat.identity(), 0.0, 0.0, 0.0)
        interactor.update_contact(1, Vec3.zero(), Quat.identity(), 0.0, 0.0, 0.0)
        interactor.update_contact(5, Vec3.zero(), Quat.identity(), 0.0, 0.0, 0.0)  # Beyond max

        contacts = interactor.get_active_contacts()
        assert len(contacts) == 2
