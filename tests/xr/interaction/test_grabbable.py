"""
Tests for XR Grabbable component (grabbable.py).

Tests the grabbable component and decorator:
    XRGrabbable, @xr_grabbable
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
from engine.xr.interaction.grabbable import (
    GrabType,
    AttachmentMode,
    HandPoseMode,
    GrabAttachPoint,
    ThrowData,
    GrabState,
    XRGrabbable,
    xr_grabbable,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def create_event():
    """Factory fixture for creating test events."""
    def _create_event(
        interaction_type: InteractionType = InteractionType.GRAB,
        interactor_id: int = 1,
        timestamp: float = 0.0,
        position: Vec3 = None
    ) -> InteractionEvent:
        return InteractionEvent(
            interactor_type=InteractorType.DIRECT,
            interactor_id=interactor_id,
            interaction_type=interaction_type,
            position=position or Vec3(0, 0, 0),
            rotation=Quat.identity(),
            timestamp=timestamp
        )
    return _create_event


@pytest.fixture
def interactor_transform():
    """Standard interactor transform for testing."""
    return RigidTransform(
        translation=Vec3(0, 1, 0),
        rotation=Quat.identity()
    )


@pytest.fixture
def object_transform():
    """Standard object transform for testing."""
    return Transform(
        translation=Vec3(0, 0.5, 0),
        rotation=Quat.identity()
    )


# =============================================================================
# @xr_grabbable Decorator Tests
# =============================================================================


class TestXRGrabbableDecorator:
    def test_basic_application(self):
        @xr_grabbable()
        class TestObject(XRGrabbable):
            pass

        assert TestObject._xr_grabbable is True

    def test_grab_type(self):
        @xr_grabbable(grab_type=GrabType.RAY)
        class TestObject(XRGrabbable):
            pass

        assert TestObject._grab_type == GrabType.RAY

    def test_attachment_mode(self):
        @xr_grabbable(attachment_mode=AttachmentMode.PHYSICS)
        class TestObject(XRGrabbable):
            pass

        assert TestObject._attachment_mode == AttachmentMode.PHYSICS

    def test_throwable(self):
        @xr_grabbable(throwable=False)
        class TestObject(XRGrabbable):
            pass

        assert TestObject._throwable is False

    def test_two_handed(self):
        @xr_grabbable(two_handed=True)
        class TestObject(XRGrabbable):
            pass

        assert TestObject._two_handed is True

    def test_hand_pose_mode(self):
        @xr_grabbable(hand_pose_mode=HandPoseMode.PRESET)
        class TestObject(XRGrabbable):
            pass

        assert TestObject._hand_pose_mode == HandPoseMode.PRESET

    def test_inherits_interactable(self):
        @xr_grabbable()
        class TestObject(XRGrabbable):
            pass

        assert TestObject._xr_interactable is True
        assert 'xr_interactable' in TestObject._applied_decorators
        assert 'xr_grabbable' in TestObject._applied_decorators

    def test_tags_stored(self):
        @xr_grabbable(throwable=True, two_handed=True)
        class TestObject(XRGrabbable):
            pass

        assert TestObject._class_tags['xr_grabbable'] is True
        assert TestObject._class_tags['throwable'] is True
        assert TestObject._class_tags['two_handed'] is True


# =============================================================================
# XRGrabbable Basic Tests
# =============================================================================


class TestXRGrabbableBasic:
    def test_initialization(self):
        obj = XRGrabbable(entity_id=42)
        assert obj.entity_id == 42
        assert obj.grab_type == GrabType.DIRECT
        assert obj.attachment_mode == AttachmentMode.FIXED
        assert obj.throwable is True
        assert obj.supports_two_handed is False

    def test_custom_initialization(self):
        obj = XRGrabbable(
            grab_type=GrabType.RAY,
            attachment_mode=AttachmentMode.PHYSICS,
            throwable=False,
            two_handed=True
        )
        assert obj.grab_type == GrabType.RAY
        assert obj.attachment_mode == AttachmentMode.PHYSICS
        assert obj.throwable is False
        assert obj.supports_two_handed is True

    def test_default_attach_point(self):
        obj = XRGrabbable()
        points = obj.get_attach_points()
        assert len(points) == 1

    def test_add_attach_point(self):
        obj = XRGrabbable()
        point = GrabAttachPoint(
            local_position=Vec3(0.1, 0, 0),
            hand_pose_id="pistol_grip"
        )
        obj.add_attach_point(point)

        points = obj.get_attach_points()
        assert len(points) == 2


# =============================================================================
# Grab Tests
# =============================================================================


class TestGrab:
    def test_can_be_grabbed(self):
        obj = XRGrabbable()
        assert obj.can_be_grabbed(1, GrabType.DIRECT) is True

    def test_cannot_grab_disabled(self):
        obj = XRGrabbable()
        obj.enabled = False
        assert obj.can_be_grabbed(1, GrabType.DIRECT) is False

    def test_cannot_grab_already_grabbed(self):
        obj = XRGrabbable()
        obj._grab_interactor = 1  # Simulate grabbed state

        assert obj.can_be_grabbed(2, GrabType.DIRECT) is False

    def test_two_handed_allows_second_grab(self):
        obj = XRGrabbable(two_handed=True)
        obj._grab_interactor = 1

        assert obj.can_be_grabbed(2, GrabType.DIRECT) is True

    def test_try_grab_success(
        self, create_event, interactor_transform, object_transform
    ):
        obj = XRGrabbable()
        event = create_event()

        result = obj.try_grab(
            1, GrabType.DIRECT, interactor_transform, object_transform, event
        )

        assert result is True
        assert obj.is_grabbed
        assert obj.grab_state is not None
        assert obj.grab_state.interactor_id == 1

    def test_try_grab_failure(self, create_event, interactor_transform, object_transform):
        obj = XRGrabbable()
        obj.enabled = False
        event = create_event()

        result = obj.try_grab(
            1, GrabType.DIRECT, interactor_transform, object_transform, event
        )

        assert result is False
        assert not obj.is_grabbed

    def test_grab_filter(self, create_event, interactor_transform, object_transform):
        obj = XRGrabbable()

        # Only allow interactor 1
        obj.set_grab_filter(lambda interactor_id, grab_type: interactor_id == 1)

        assert obj.can_be_grabbed(1, GrabType.DIRECT) is True
        assert obj.can_be_grabbed(2, GrabType.DIRECT) is False


# =============================================================================
# Release and Throw Tests
# =============================================================================


class TestReleaseAndThrow:
    def test_release(
        self, create_event, interactor_transform, object_transform
    ):
        obj = XRGrabbable()
        obj.try_grab(1, GrabType.DIRECT, interactor_transform, object_transform, create_event())

        result = obj.release(1, create_event())

        assert not obj.is_grabbed
        assert obj.grab_state is None

    def test_release_wrong_interactor(
        self, create_event, interactor_transform, object_transform
    ):
        obj = XRGrabbable()
        obj.try_grab(1, GrabType.DIRECT, interactor_transform, object_transform, create_event())

        result = obj.release(2, create_event())  # Wrong interactor

        assert result is None
        assert obj.is_grabbed

    def test_release_returns_throw_data(
        self, create_event, interactor_transform, object_transform
    ):
        obj = XRGrabbable(throwable=True)
        obj.try_grab(1, GrabType.DIRECT, interactor_transform, object_transform, create_event())

        throw_data = obj.release(
            1,
            create_event(),
            current_velocity=Vec3(1, 0, 0),
            current_angular_velocity=Vec3(0, 1, 0)
        )

        assert throw_data is not None
        assert isinstance(throw_data, ThrowData)
        assert throw_data.linear_velocity.x == pytest.approx(1.0)

    def test_non_throwable_returns_throw_data_with_zero_velocity(
        self, create_event, interactor_transform, object_transform
    ):
        obj = XRGrabbable(throwable=False)
        obj.try_grab(1, GrabType.DIRECT, interactor_transform, object_transform, create_event())

        throw_data = obj.release(1, create_event())

        # Non-throwable still returns data but with no velocity transfer
        assert throw_data is None or throw_data.linear_velocity.length() == pytest.approx(0.0, abs=0.01)


# =============================================================================
# Two-Handed Grab Tests
# =============================================================================


class TestTwoHandedGrab:
    def test_two_handed_grab(
        self, create_event, interactor_transform, object_transform
    ):
        obj = XRGrabbable(two_handed=True)

        # First hand grabs
        obj.try_grab(1, GrabType.DIRECT, interactor_transform, object_transform, create_event())

        # Second hand grabs
        second_transform = RigidTransform(Vec3(0.2, 1, 0), Quat.identity())
        result = obj.try_grab(
            2, GrabType.DIRECT, second_transform, object_transform, create_event(interactor_id=2)
        )

        assert result is True
        assert obj.is_two_hand_grabbed

    def test_release_secondary_hand(
        self, create_event, interactor_transform, object_transform
    ):
        obj = XRGrabbable(two_handed=True)

        obj.try_grab(1, GrabType.DIRECT, interactor_transform, object_transform, create_event())
        second_transform = RigidTransform(Vec3(0.2, 1, 0), Quat.identity())
        obj.try_grab(2, GrabType.DIRECT, second_transform, object_transform, create_event(interactor_id=2))

        # Release secondary
        obj.release(2, create_event(interactor_id=2))

        assert obj.is_grabbed  # Still grabbed by primary
        assert not obj.is_two_hand_grabbed


# =============================================================================
# Velocity Tracking Tests
# =============================================================================


class TestVelocityTracking:
    def test_track_velocity(self):
        obj = XRGrabbable()

        obj.track_velocity(0.0, Vec3(0, 0, 0), Quat.identity())
        obj.track_velocity(0.016, Vec3(0.1, 0, 0), Quat.identity())
        obj.track_velocity(0.032, Vec3(0.2, 0, 0), Quat.identity())

        # Velocity tracker should have samples
        assert len(obj._velocity_tracker) == 3

    def test_velocity_history_limited(self):
        obj = XRGrabbable()

        # Add samples over a long period
        for i in range(100):
            t = i * 0.002  # 2ms apart
            obj.track_velocity(t, Vec3(i * 0.01, 0, 0), Quat.identity())

        # Should only keep last 100ms worth (but with some tolerance)
        assert len(obj._velocity_tracker) <= 55


# =============================================================================
# Attach Transform Tests
# =============================================================================


class TestAttachTransform:
    def test_compute_attached_transform_single_hand(
        self, create_event, interactor_transform, object_transform
    ):
        obj = XRGrabbable()
        obj.try_grab(1, GrabType.DIRECT, interactor_transform, object_transform, create_event())

        new_interactor_transform = RigidTransform(Vec3(1, 2, 3), Quat.identity())
        result = obj.compute_attached_transform(new_interactor_transform)

        assert result.translation is not None
        assert result.rotation is not None

    def test_compute_two_handed_transform(
        self, create_event, interactor_transform, object_transform
    ):
        obj = XRGrabbable(two_handed=True)
        obj.try_grab(1, GrabType.DIRECT, interactor_transform, object_transform, create_event())

        second_transform = RigidTransform(Vec3(0.2, 1, 0), Quat.identity())
        obj.try_grab(2, GrabType.DIRECT, second_transform, object_transform, create_event(interactor_id=2))

        primary = RigidTransform(Vec3(0, 1, 0), Quat.identity())
        secondary = RigidTransform(Vec3(0.3, 1, 0), Quat.identity())

        result = obj.compute_attached_transform(primary, secondary)

        # Position should be midpoint
        expected_midpoint = Vec3(0.15, 1, 0)
        assert result.translation.x == pytest.approx(expected_midpoint.x, abs=0.01)


# =============================================================================
# Callback Tests
# =============================================================================


class TestGrabbableCallbacks:
    def test_grab_callback(
        self, create_event, interactor_transform, object_transform
    ):
        obj = XRGrabbable()
        grab_events = []

        obj.add_grab_callback(lambda state: grab_events.append(state))
        obj.try_grab(1, GrabType.DIRECT, interactor_transform, object_transform, create_event())

        assert len(grab_events) == 1
        assert grab_events[0].interactor_id == 1

    def test_throw_callback(
        self, create_event, interactor_transform, object_transform
    ):
        obj = XRGrabbable(throwable=True)
        throw_events = []

        obj.add_throw_callback(lambda data: throw_events.append(data))
        obj.try_grab(1, GrabType.DIRECT, interactor_transform, object_transform, create_event())
        obj.release(1, create_event(), current_velocity=Vec3(1, 0, 0))

        assert len(throw_events) == 1


# =============================================================================
# Throw Scale Tests
# =============================================================================


class TestThrowScale:
    def test_velocity_scale(
        self, create_event, interactor_transform, object_transform
    ):
        obj = XRGrabbable(throwable=True)
        obj.set_throw_scales(velocity_scale=2.0)

        obj.try_grab(1, GrabType.DIRECT, interactor_transform, object_transform, create_event())
        throw_data = obj.release(1, create_event(), current_velocity=Vec3(1, 0, 0))

        assert throw_data.linear_velocity.x == pytest.approx(2.0)

    def test_angular_scale(
        self, create_event, interactor_transform, object_transform
    ):
        obj = XRGrabbable(throwable=True)
        obj.set_throw_scales(angular_scale=0.5)

        obj.try_grab(1, GrabType.DIRECT, interactor_transform, object_transform, create_event())
        throw_data = obj.release(
            1, create_event(), current_angular_velocity=Vec3(2, 0, 0)
        )

        assert throw_data.angular_velocity.x == pytest.approx(1.0)


# =============================================================================
# Attach Point Selection Tests
# =============================================================================


class TestAttachPointSelection:
    def test_nearest_attach_point(self):
        obj = XRGrabbable()

        # Add multiple attach points
        point1 = GrabAttachPoint(local_position=Vec3(0.1, 0, 0))
        point2 = GrabAttachPoint(local_position=Vec3(-0.1, 0, 0))

        obj.add_attach_point(point1)
        obj.add_attach_point(point2)

        # Grab from right side
        transform = Transform(translation=Vec3(0, 0, 0))
        nearest = obj.get_nearest_attach_point(Vec3(0.2, 0, 0), transform)

        assert nearest.local_position.x == pytest.approx(0.1, abs=0.01)
