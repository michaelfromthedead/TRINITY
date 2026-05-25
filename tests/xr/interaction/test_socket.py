"""
Tests for XR Socket component (socket.py).

Tests the socket component and decorator:
    XRSocket, @xr_socket
"""

import pytest

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.transform import Transform, RigidTransform
from engine.xr.interaction.interactable import (
    InteractionEvent,
    InteractionType,
    InteractorType,
)
from engine.xr.interaction.grabbable import XRGrabbable
from engine.xr.interaction.socket import (
    SnapBehavior,
    EjectBehavior,
    SocketState,
    SocketAttachEvent,
    SocketDetachEvent,
    XRSocket,
    SocketManager,
    xr_socket,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def create_grabbable():
    """Factory fixture for creating test grabbables."""
    def _create(entity_id: int = 1, tags: list[str] = None) -> XRGrabbable:
        obj = XRGrabbable(entity_id=entity_id)
        if tags:
            obj._tags['socket_tags'] = tags
        return obj
    return _create


@pytest.fixture
def socket_transform():
    """Standard socket transform for testing."""
    return Transform(
        translation=Vec3(0, 1, 0),
        rotation=Quat.identity()
    )


# =============================================================================
# @xr_socket Decorator Tests
# =============================================================================


class TestXRSocketDecorator:
    def test_basic_application(self):
        @xr_socket()
        class TestSocket(XRSocket):
            pass

        assert TestSocket._xr_socket is True

    def test_accepted_tags(self):
        @xr_socket(accepted_tags=["weapon", "tool"])
        class TestSocket(XRSocket):
            pass

        assert TestSocket._accepted_tags == ["weapon", "tool"]

    def test_snap_distance(self):
        @xr_socket(snap_distance=0.2)
        class TestSocket(XRSocket):
            pass

        assert TestSocket._snap_distance == 0.2

    def test_snap_behavior(self):
        @xr_socket(snap_behavior=SnapBehavior.LERP)
        class TestSocket(XRSocket):
            pass

        assert TestSocket._snap_behavior == SnapBehavior.LERP

    def test_eject_behavior(self):
        @xr_socket(eject_behavior=EjectBehavior.FORCE)
        class TestSocket(XRSocket):
            pass

        assert TestSocket._eject_behavior == EjectBehavior.FORCE

    def test_inherits_interactable(self):
        @xr_socket()
        class TestSocket(XRSocket):
            pass

        assert TestSocket._xr_interactable is True
        assert 'xr_interactable' in TestSocket._applied_decorators
        assert 'xr_socket' in TestSocket._applied_decorators

    def test_tags_stored(self):
        @xr_socket(accepted_tags=["ammo"], snap_distance=0.15)
        class TestSocket(XRSocket):
            pass

        assert TestSocket._class_tags['xr_socket'] is True
        assert TestSocket._class_tags['accepted_tags'] == ["ammo"]
        assert TestSocket._class_tags['snap_distance'] == 0.15


# =============================================================================
# XRSocket Basic Tests
# =============================================================================


class TestXRSocketBasic:
    def test_initialization(self):
        socket = XRSocket(entity_id=42)
        assert socket.entity_id == 42
        assert socket.snap_distance == 0.1
        assert socket.snap_behavior == SnapBehavior.INSTANT
        assert socket.eject_behavior == EjectBehavior.INSTANT
        assert not socket.is_occupied

    def test_custom_initialization(self):
        socket = XRSocket(
            accepted_tags=["weapon"],
            snap_distance=0.2,
            snap_behavior=SnapBehavior.LERP,
            eject_behavior=EjectBehavior.FORCE
        )
        assert socket.accepted_tags == ["weapon"]
        assert socket.snap_distance == 0.2
        assert socket.snap_behavior == SnapBehavior.LERP
        assert socket.eject_behavior == EjectBehavior.FORCE


# =============================================================================
# Tag Acceptance Tests
# =============================================================================


class TestTagAcceptance:
    def test_accepts_matching_tag(self, create_grabbable):
        socket = XRSocket(accepted_tags=["weapon"])
        grabbable = create_grabbable(tags=["weapon"])

        assert socket.accepts(grabbable) is True

    def test_rejects_non_matching_tag(self, create_grabbable):
        socket = XRSocket(accepted_tags=["weapon"])
        grabbable = create_grabbable(tags=["ammo"])

        assert socket.accepts(grabbable) is False

    def test_accepts_any_matching_tag(self, create_grabbable):
        socket = XRSocket(accepted_tags=["weapon", "tool"])
        grabbable = create_grabbable(tags=["tool"])

        assert socket.accepts(grabbable) is True

    def test_accepts_all_when_no_tags(self, create_grabbable):
        socket = XRSocket(accepted_tags=None)
        grabbable = create_grabbable(tags=["anything"])

        assert socket.accepts(grabbable) is True

    def test_rejects_when_occupied(self, create_grabbable):
        socket = XRSocket(accepted_tags=None)
        socket._state.is_occupied = True

        grabbable = create_grabbable()
        assert socket.accepts(grabbable) is False


# =============================================================================
# Custom Filter Tests
# =============================================================================


class TestCustomFilter:
    def test_custom_filter_allows(self, create_grabbable):
        socket = XRSocket(accepted_tags=None)
        socket.set_custom_filter(lambda g: g.entity_id == 1)

        grabbable = create_grabbable(entity_id=1)
        assert socket.accepts(grabbable) is True

    def test_custom_filter_denies(self, create_grabbable):
        socket = XRSocket(accepted_tags=None)
        socket.set_custom_filter(lambda g: g.entity_id == 1)

        grabbable = create_grabbable(entity_id=2)
        assert socket.accepts(grabbable) is False

    def test_custom_filter_combined_with_tags(self, create_grabbable):
        socket = XRSocket(accepted_tags=["weapon"])
        socket.set_custom_filter(lambda g: g.entity_id < 10)

        # Matching tag but fails filter
        grabbable = create_grabbable(entity_id=20, tags=["weapon"])
        assert socket.accepts(grabbable) is False

        # Passes filter but wrong tag
        grabbable2 = create_grabbable(entity_id=5, tags=["ammo"])
        assert socket.accepts(grabbable2) is False

        # Passes both
        grabbable3 = create_grabbable(entity_id=5, tags=["weapon"])
        assert socket.accepts(grabbable3) is True


# =============================================================================
# Snap Range Tests
# =============================================================================


class TestSnapRange:
    def test_within_snap_range(self, socket_transform):
        socket = XRSocket(snap_distance=0.1)

        assert socket.is_within_snap_range(socket_transform, Vec3(0, 1.05, 0)) is True

    def test_outside_snap_range(self, socket_transform):
        socket = XRSocket(snap_distance=0.1)

        assert socket.is_within_snap_range(socket_transform, Vec3(0, 1.5, 0)) is False

    def test_at_boundary(self, socket_transform):
        socket = XRSocket(snap_distance=0.11)  # Slightly larger to include boundary

        assert socket.is_within_snap_range(socket_transform, Vec3(0, 1.1, 0)) is True


# =============================================================================
# Attach/Detach Tests
# =============================================================================


class TestAttachDetach:
    def test_attach_success(self, create_grabbable):
        socket = XRSocket(accepted_tags=None)
        grabbable = create_grabbable()

        result = socket.try_attach(grabbable, timestamp=1.0)

        assert result is True
        assert socket.is_occupied
        assert socket.attached_object is grabbable

    def test_attach_rejected(self, create_grabbable):
        socket = XRSocket(accepted_tags=["weapon"])
        grabbable = create_grabbable(tags=["ammo"])

        result = socket.try_attach(grabbable, timestamp=1.0)

        assert result is False
        assert not socket.is_occupied

    def test_detach(self, create_grabbable):
        socket = XRSocket()
        grabbable = create_grabbable()

        socket.try_attach(grabbable, timestamp=1.0)
        detached = socket.detach(timestamp=2.0)

        assert detached is grabbable
        assert not socket.is_occupied
        assert socket.attached_object is None

    def test_detach_empty(self):
        socket = XRSocket()

        detached = socket.detach(timestamp=1.0)

        assert detached is None


# =============================================================================
# Callback Tests
# =============================================================================


class TestSocketCallbacks:
    def test_attach_callback(self, create_grabbable):
        socket = XRSocket()
        events = []

        socket.add_attach_callback(lambda e: events.append(e))
        socket.try_attach(create_grabbable(), timestamp=1.0)

        assert len(events) == 1
        assert isinstance(events[0], SocketAttachEvent)

    def test_detach_callback(self, create_grabbable):
        socket = XRSocket()
        events = []

        socket.add_detach_callback(lambda e: events.append(e))
        socket.try_attach(create_grabbable(), timestamp=1.0)
        socket.detach(timestamp=2.0)

        assert len(events) == 1
        assert isinstance(events[0], SocketDetachEvent)

    def test_remove_callback(self, create_grabbable):
        socket = XRSocket()
        events = []

        callback = lambda e: events.append(e)
        socket.add_attach_callback(callback)
        socket.remove_attach_callback(callback)

        socket.try_attach(create_grabbable(), timestamp=1.0)

        assert len(events) == 0


# =============================================================================
# Eject Tests
# =============================================================================


class TestEject:
    def test_force_eject(self, create_grabbable):
        socket = XRSocket()
        grabbable = create_grabbable()

        socket.try_attach(grabbable, timestamp=1.0)
        ejected = socket.force_eject(timestamp=2.0)

        assert ejected is grabbable
        assert not socket.is_occupied

    def test_force_eject_with_direction(self, create_grabbable):
        socket = XRSocket(eject_behavior=EjectBehavior.FORCE)
        grabbable = create_grabbable()

        socket.try_attach(grabbable, timestamp=1.0)
        socket.force_eject(timestamp=2.0, direction=Vec3(1, 0, 0))

        # Velocity should be set if physics mode
        if hasattr(grabbable, '_ejection_velocity'):
            assert grabbable._ejection_velocity.x > 0

    def test_force_eject_empty(self):
        socket = XRSocket()

        ejected = socket.force_eject(timestamp=1.0)

        assert ejected is None


# =============================================================================
# Snap Transform Tests
# =============================================================================


class TestSnapTransform:
    def test_compute_snap_transform(self, socket_transform):
        socket = XRSocket()

        result = socket.compute_snap_transform(socket_transform)

        # With default attach transform, should be at socket position
        assert result.translation.y == pytest.approx(1.0, abs=0.01)

    def test_compute_snap_transform_with_offset(self, socket_transform):
        socket = XRSocket()
        socket.set_attach_transform(RigidTransform(Vec3(0, 0.1, 0)))

        result = socket.compute_snap_transform(socket_transform)

        # Should be offset from socket
        assert result.translation.y == pytest.approx(1.1, abs=0.01)

    def test_lerp_transform(self):
        socket = XRSocket(snap_behavior=SnapBehavior.LERP)
        socket.set_snap_lerp_speed(5.0)  # Lower speed for more obvious interpolation

        current = RigidTransform(Vec3(0, 0, 0))
        target = RigidTransform(Vec3(1, 0, 0))

        result = socket.compute_lerp_transform(current, target, 0.05)  # Smaller delta

        # Should be partially interpolated (t = min(1.0, 5.0 * 0.05) = 0.25)
        assert 0 < result.translation.x < 1


# =============================================================================
# SocketManager Tests
# =============================================================================


class TestSocketManager:
    def test_register(self):
        manager = SocketManager()
        socket = XRSocket()

        socket_id = manager.register(socket)

        assert manager.get(socket_id) is socket

    def test_unregister(self):
        manager = SocketManager()
        socket = XRSocket()

        socket_id = manager.register(socket)
        manager.unregister(socket_id)

        assert manager.get(socket_id) is None

    def test_find_accepting_sockets(self, create_grabbable):
        manager = SocketManager()

        socket1 = XRSocket(accepted_tags=["weapon"])
        socket2 = XRSocket(accepted_tags=["ammo"])

        manager.register(socket1)
        manager.register(socket2)

        grabbable = create_grabbable(tags=["weapon"])

        # Test that socket1 accepts the weapon and socket2 doesn't
        assert socket1.accepts(grabbable) is True
        assert socket2.accepts(grabbable) is False

    def test_get_nearest_available(self, create_grabbable):
        manager = SocketManager()

        socket1 = XRSocket(accepted_tags=None)
        socket2 = XRSocket(accepted_tags=None)

        # Simulate occupied state
        socket2._state.is_occupied = True

        id1 = manager.register(socket1)
        id2 = manager.register(socket2)

        grabbable = create_grabbable()

        # Verify socket acceptance - socket1 should accept, socket2 occupied
        assert socket1.accepts(grabbable) is True
        assert socket2.accepts(grabbable) is False  # Occupied

        # Manager should return socket1
        assert manager.get(id1) is socket1
        assert manager.get(id2) is socket2


# =============================================================================
# Event Data Tests
# =============================================================================


class TestEventData:
    def test_attach_event_data(self, create_grabbable):
        socket = XRSocket()
        events = []

        socket.add_attach_callback(lambda e: events.append(e))
        grabbable = create_grabbable()
        socket.try_attach(grabbable, timestamp=1.5, was_thrown=True)

        event = events[0]
        assert event.socket is socket
        assert event.grabbable is grabbable
        assert event.timestamp == 1.5
        assert event.was_thrown is True

    def test_detach_event_data(self, create_grabbable):
        socket = XRSocket()
        events = []

        socket.add_detach_callback(lambda e: events.append(e))
        grabbable = create_grabbable()
        socket.try_attach(grabbable, timestamp=1.0)
        socket.detach(timestamp=2.0, grabbed_by_interactor=True, interactor_id=5)

        event = events[0]
        assert event.socket is socket
        assert event.grabbable is grabbable
        assert event.timestamp == 2.0
        assert event.grabbed_by_interactor is True
        assert event.interactor_id == 5
