"""
Tests for image tracking module.

Tests the AR image tracking system including:
    - Image reference database
    - Image target creation and tracking
    - Tracking states and callbacks
    - Pose updates
    - ImageTracker operations
"""

import pytest

from engine.core.math.vec import Vec2, Vec3
from engine.core.math.quat import Quat
from engine.xr.spatial.image_tracking import (
    ImageReference,
    ImageTarget,
    ImageTargetPose,
    ImageTracker,
    ImageTrackerConfig,
    ImageTrackingState,
    TrackingMode,
    ar_trackable,
)


class TestARTrackableDecorator:
    """Tests for the @ar_trackable decorator."""

    def test_decorator_image_type(self):
        @ar_trackable(trackable_type="image", reference_id="marker-001")
        class TestTarget:
            pass

        assert TestTarget._ar_trackable is True
        assert TestTarget._trackable_type == "image"
        assert TestTarget._reference_id == "marker-001"

    def test_decorator_object_type(self):
        @ar_trackable(trackable_type="object", reference_id="model-001")
        class TestTarget:
            pass

        assert TestTarget._trackable_type == "object"

    def test_decorator_invalid_type(self):
        with pytest.raises(ValueError, match="Invalid trackable_type"):
            @ar_trackable(trackable_type="invalid")
            class BadTarget:
                pass

    def test_decorator_tags(self):
        @ar_trackable(trackable_type="image", reference_id="test-ref")
        class TestTarget:
            pass

        assert TestTarget._tags["ar_trackable"] is True
        assert TestTarget._tags["trackable_type"] == "image"
        assert TestTarget._tags["reference_id"] == "test-ref"

    def test_decorator_applied_decorators(self):
        @ar_trackable(trackable_type="image")
        class TestTarget:
            pass

        assert "ar_trackable" in TestTarget._applied_decorators


class TestImageReference:
    """Tests for ImageReference class."""

    def test_default_reference(self):
        ref = ImageReference()
        assert ref.name == ""
        assert ref.physical_width == 0.1
        assert ref.is_enabled is True

    def test_custom_reference(self):
        ref = ImageReference(
            name="Company Logo",
            physical_width=0.15,
            image_path="/assets/logo.png",
            metadata={"category": "brand"},
        )

        assert ref.name == "Company Logo"
        assert ref.physical_width == 0.15
        assert ref.image_path == "/assets/logo.png"
        assert ref.metadata["category"] == "brand"


class TestImageTargetPose:
    """Tests for ImageTargetPose class."""

    def test_default_pose(self):
        pose = ImageTargetPose()
        assert pose.position == Vec3.zero()
        assert pose.orientation == Quat.identity()

    def test_transform_point(self):
        pose = ImageTargetPose(
            position=Vec3(5, 0, 5),
            orientation=Quat.identity(),
        )

        local = Vec3(1, 0, 0)
        world = pose.transform_point(local)

        assert abs(world.x - 6.0) < 0.01
        assert abs(world.y - 0.0) < 0.01
        assert abs(world.z - 5.0) < 0.01


class TestImageTarget:
    """Tests for ImageTarget class."""

    def test_create_target(self):
        ref = ImageReference(name="Test", physical_width=0.1)
        target = ImageTarget(reference=ref)

        assert target.name == "Test"
        assert target.tracking_state == ImageTrackingState.NONE
        assert target.is_tracked is False
        assert target.is_active is False

    def test_activate_target(self):
        ref = ImageReference(name="Test", physical_width=0.1)
        target = ImageTarget(reference=ref)

        result = target.activate()

        assert result is True
        assert target.is_active is True
        assert target.tracking_state == ImageTrackingState.DETECTING

    def test_double_activate(self):
        ref = ImageReference(name="Test", physical_width=0.1)
        target = ImageTarget(reference=ref)
        target.activate()

        result = target.activate()
        assert result is False

    def test_deactivate_target(self):
        ref = ImageReference(name="Test", physical_width=0.1)
        target = ImageTarget(reference=ref)
        target.activate()

        result = target.deactivate()

        assert result is True
        assert target.is_active is False
        assert target.tracking_state == ImageTrackingState.NONE

    def test_update_pose(self):
        ref = ImageReference(name="Test", physical_width=0.1)
        target = ImageTarget(reference=ref)
        target.activate()

        target.update_pose(
            position=Vec3(1, 2, 3),
            orientation=Quat.identity(),
            confidence=0.95,
            tracked_size=Vec2(0.1, 0.07),
            timestamp=1.0,
        )

        assert target.is_tracked is True
        assert target.position == Vec3(1, 2, 3)
        assert target.confidence == 0.95
        assert target.tracked_size == Vec2(0.1, 0.07)

    def test_first_detection(self):
        ref = ImageReference(name="Test", physical_width=0.1)
        target = ImageTarget(reference=ref)
        target.activate()

        callback_results = []

        def on_detected(t):
            callback_results.append("detected")

        target.add_callback("detected", on_detected)
        target.update_pose(
            Vec3.zero(), Quat.identity(), 1.0, Vec2(0.1, 0.1), 1.0
        )

        assert "detected" in callback_results

    def test_update_tracking_state(self):
        ref = ImageReference(name="Test", physical_width=0.1)
        target = ImageTarget(reference=ref)
        target.activate()
        target.update_pose(Vec3.zero(), Quat.identity(), 1.0, Vec2(0.1, 0.1), 1.0)

        target.update_tracking_state(ImageTrackingState.LIMITED, confidence=0.5)

        assert target.tracking_state == ImageTrackingState.LIMITED
        assert target.confidence == 0.5

    def test_is_visible(self):
        ref = ImageReference(name="Test", physical_width=0.1)
        target = ImageTarget(reference=ref)
        target.activate()
        target.update_pose(Vec3.zero(), Quat.identity(), 1.0, Vec2(0.1, 0.1), 1.0)

        assert target.is_visible is True

        target.update_tracking_state(ImageTrackingState.EXTENDED)
        assert target.is_visible is False

    def test_get_corner_positions(self):
        ref = ImageReference(name="Test", physical_width=0.2)
        target = ImageTarget(reference=ref)
        target.activate()
        target.update_pose(
            Vec3(0, 0, 0),
            Quat.identity(),
            1.0,
            Vec2(0.2, 0.1),
            1.0,
        )

        corners = target.get_corner_positions()

        assert len(corners) == 4
        # Check corners are at expected positions (roughly)
        assert any(abs(c.x - (-0.1)) < 0.01 for c in corners)
        assert any(abs(c.x - 0.1) < 0.01 for c in corners)

    def test_local_to_world(self):
        ref = ImageReference(name="Test", physical_width=0.2)
        target = ImageTarget(reference=ref)
        target.activate()
        target.update_pose(
            Vec3(10, 0, 10),
            Quat.identity(),
            1.0,
            Vec2(0.2, 0.1),
            1.0,
        )

        world_point = target.local_to_world(Vec3(1, 0, 0))

        assert abs(world_point.x - 11) < 0.01
        assert abs(world_point.z - 10) < 0.01

    def test_world_to_local(self):
        ref = ImageReference(name="Test", physical_width=0.2)
        target = ImageTarget(reference=ref)
        target.activate()
        target.update_pose(
            Vec3(10, 0, 10),
            Quat.identity(),
            1.0,
            Vec2(0.2, 0.1),
            1.0,
        )

        local_point = target.world_to_local(Vec3(11, 0, 10))

        assert abs(local_point.x - 1) < 0.01
        assert abs(local_point.z - 0) < 0.01

    def test_callbacks(self):
        ref = ImageReference(name="Test", physical_width=0.1)
        target = ImageTarget(reference=ref)
        target.activate()

        results = []

        def on_lost(t):
            results.append("lost")

        target.add_callback("lost", on_lost)
        target.update_pose(Vec3.zero(), Quat.identity(), 1.0, Vec2(0.1, 0.1), 1.0)
        target.update_tracking_state(ImageTrackingState.LOST)

        assert "lost" in results


class TestImageTracker:
    """Tests for ImageTracker class."""

    def test_default_config(self):
        tracker = ImageTracker()
        assert tracker.config.max_tracked_images == 4
        assert tracker.config.tracking_mode == TrackingMode.CONTINUOUS
        assert tracker.is_running is False

    def test_custom_config(self):
        config = ImageTrackerConfig(
            max_tracked_images=8,
            tracking_mode=TrackingMode.ADAPTIVE,
        )
        tracker = ImageTracker(config)

        assert tracker.config.max_tracked_images == 8

    def test_add_reference_image(self):
        tracker = ImageTracker()

        ref = tracker.add_reference_image(
            name="Logo",
            physical_width=0.15,
            image_path="/assets/logo.png",
        )

        assert ref is not None
        assert tracker.database_size == 1
        assert tracker.get_reference_image(ref.reference_id) is ref

    def test_remove_reference_image(self):
        tracker = ImageTracker()
        ref = tracker.add_reference_image("Test", 0.1)

        result = tracker.remove_reference_image(ref.reference_id)

        assert result is True
        assert tracker.database_size == 0

    def test_set_reference_enabled(self):
        tracker = ImageTracker()
        ref = tracker.add_reference_image("Test", 0.1)

        tracker.set_reference_enabled(ref.reference_id, False)

        assert ref.is_enabled is False

    def test_create_target(self):
        tracker = ImageTracker()
        ref = tracker.add_reference_image("Logo", 0.15)

        target = tracker.create_target(ref.reference_id)

        assert target is not None
        assert target.reference_id == ref.reference_id
        assert len(tracker.get_all_targets()) == 1

    def test_create_target_disabled_reference(self):
        tracker = ImageTracker()
        ref = tracker.add_reference_image("Logo", 0.15)
        ref.is_enabled = False

        target = tracker.create_target(ref.reference_id)

        assert target is None

    def test_create_target_invalid_reference(self):
        tracker = ImageTracker()

        target = tracker.create_target("nonexistent-ref")

        assert target is None

    def test_remove_target(self):
        tracker = ImageTracker()
        ref = tracker.add_reference_image("Logo", 0.15)
        target = tracker.create_target(ref.reference_id)

        result = tracker.remove_target(target.target_id)

        assert result is True
        assert len(tracker.get_all_targets()) == 0

    def test_start_stop(self):
        tracker = ImageTracker()
        ref = tracker.add_reference_image("Test", 0.1)
        target = tracker.create_target(ref.reference_id)

        tracker.start()

        assert tracker.is_running is True
        assert target.is_active is True

        tracker.stop()

        assert tracker.is_running is False
        assert target.is_active is False

    def test_double_start(self):
        tracker = ImageTracker()
        tracker.start()

        result = tracker.start()
        assert result is False

    def test_get_tracked_targets(self):
        tracker = ImageTracker()
        ref = tracker.add_reference_image("Test", 0.1)
        target = tracker.create_target(ref.reference_id)
        tracker.start()

        # Initially not tracked
        assert len(tracker.get_tracked_targets()) == 0

        # Simulate detection
        target.update_pose(
            Vec3.zero(), Quat.identity(), 1.0, Vec2(0.1, 0.1), 1.0
        )

        assert len(tracker.get_tracked_targets()) == 1

    def test_get_visible_targets(self):
        tracker = ImageTracker()
        ref = tracker.add_reference_image("Test", 0.1)
        target = tracker.create_target(ref.reference_id)
        tracker.start()
        target.update_pose(
            Vec3.zero(), Quat.identity(), 1.0, Vec2(0.1, 0.1), 1.0
        )

        visible = tracker.get_visible_targets()
        assert len(visible) == 1

        target.update_tracking_state(ImageTrackingState.EXTENDED)
        visible = tracker.get_visible_targets()
        assert len(visible) == 0

    def test_find_target_by_reference(self):
        tracker = ImageTracker()
        ref = tracker.add_reference_image("Test", 0.1)
        target = tracker.create_target(ref.reference_id)

        found = tracker.find_target_by_reference(ref.reference_id)

        assert found is target

    def test_update_tracking_timeout(self):
        tracker = ImageTracker(ImageTrackerConfig(tracking_timeout=2.0))
        ref = tracker.add_reference_image("Test", 0.1)
        target = tracker.create_target(ref.reference_id)
        tracker.start()
        target.update_pose(
            Vec3.zero(), Quat.identity(), 1.0, Vec2(0.1, 0.1), 0.0
        )
        target.update_tracking_state(ImageTrackingState.EXTENDED)

        # Update past timeout
        tracker.update(timestamp=5.0)

        assert target.tracking_state == ImageTrackingState.LOST

    def test_clear(self):
        tracker = ImageTracker()
        tracker.add_reference_image("Test1", 0.1)
        tracker.add_reference_image("Test2", 0.2)
        tracker.start()

        tracker.clear()

        assert tracker.database_size == 0
        assert len(tracker.get_all_targets()) == 0
        assert tracker.is_running is False

    def test_callbacks(self):
        tracker = ImageTracker()
        results = []

        def on_started():
            results.append("started")

        def on_stopped():
            results.append("stopped")

        tracker.add_callback("tracking_started", on_started)
        tracker.add_callback("tracking_stopped", on_stopped)

        tracker.start()
        tracker.stop()

        assert "started" in results
        assert "stopped" in results

    def test_tracked_count(self):
        tracker = ImageTracker()
        ref1 = tracker.add_reference_image("Test1", 0.1)
        ref2 = tracker.add_reference_image("Test2", 0.1)
        t1 = tracker.create_target(ref1.reference_id)
        t2 = tracker.create_target(ref2.reference_id)
        tracker.start()

        assert tracker.tracked_count == 0

        t1.update_pose(Vec3.zero(), Quat.identity(), 1.0, Vec2(0.1, 0.1), 1.0)
        assert tracker.tracked_count == 1

        t2.update_pose(Vec3.zero(), Quat.identity(), 1.0, Vec2(0.1, 0.1), 1.0)
        assert tracker.tracked_count == 2
