"""
Tests for spatial anchors module.

Tests the spatial anchor system including:
    - Anchor creation and lifecycle
    - Local, persistent, and cloud anchors
    - Anchor tracking states
    - Pose updates and callbacks
    - AnchorManager operations
"""

import pytest

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.xr.spatial.anchor import (
    AnchorManager,
    AnchorPersistenceState,
    AnchorTrackingState,
    AnchorType,
    SpatialAnchor,
    spatial_anchor,
)


class TestSpatialAnchorDecorator:
    """Tests for the @spatial_anchor decorator."""

    def test_decorator_sets_anchor_type(self):
        @spatial_anchor(anchor_type="local")
        class TestAnchor:
            pass

        assert TestAnchor._spatial_anchor is True
        assert TestAnchor._anchor_type == "local"

    def test_decorator_sets_persistent(self):
        @spatial_anchor(anchor_type="persistent", persistent=True)
        class TestAnchor:
            pass

        assert TestAnchor._anchor_persistent is True

    def test_decorator_sets_cloud_id(self):
        @spatial_anchor(anchor_type="cloud", cloud_id="test-cloud-123")
        class TestAnchor:
            pass

        assert TestAnchor._anchor_cloud_id == "test-cloud-123"

    def test_decorator_invalid_type(self):
        with pytest.raises(ValueError, match="Invalid anchor_type"):
            @spatial_anchor(anchor_type="invalid")
            class BadAnchor:
                pass

    def test_decorator_tags(self):
        @spatial_anchor(anchor_type="persistent", persistent=True)
        class TestAnchor:
            pass

        assert TestAnchor._tags["spatial_anchor"] is True
        assert TestAnchor._tags["anchor_type"] == "persistent"
        assert TestAnchor._tags["anchor_persistent"] is True

    def test_decorator_applied_decorators(self):
        @spatial_anchor(anchor_type="local")
        class TestAnchor:
            pass

        assert "spatial_anchor" in TestAnchor._applied_decorators


class TestSpatialAnchor:
    """Tests for SpatialAnchor class."""

    def test_create_local_anchor(self):
        anchor = SpatialAnchor(anchor_type=AnchorType.LOCAL)
        assert anchor.anchor_type == AnchorType.LOCAL
        assert not anchor.is_persistent
        assert not anchor.is_cloud_anchor

    def test_create_persistent_anchor(self):
        anchor = SpatialAnchor(anchor_type=AnchorType.PERSISTENT)
        assert anchor.anchor_type == AnchorType.PERSISTENT
        assert anchor.is_persistent

    def test_create_cloud_anchor(self):
        anchor = SpatialAnchor(anchor_type=AnchorType.CLOUD)
        assert anchor.anchor_type == AnchorType.CLOUD
        assert anchor.is_cloud_anchor

    def test_anchor_initial_state(self):
        anchor = SpatialAnchor()
        assert anchor.tracking_state == AnchorTrackingState.UNKNOWN
        assert anchor.confidence == 0.0
        assert not anchor.is_active
        assert not anchor.is_tracking

    def test_anchor_with_initial_pose(self):
        pos = Vec3(1.0, 2.0, 3.0)
        rot = Quat.from_euler(0.1, 0.2, 0.3)
        anchor = SpatialAnchor(position=pos, orientation=rot)

        assert anchor.position == pos
        assert anchor.orientation == rot

    def test_anchor_create(self):
        anchor = SpatialAnchor()
        result = anchor.create(timestamp=1.0)

        assert result is True
        assert anchor.is_active
        assert anchor.is_tracking
        assert anchor.tracking_state == AnchorTrackingState.TRACKING

    def test_anchor_double_create(self):
        anchor = SpatialAnchor()
        anchor.create()
        result = anchor.create()
        assert result is False

    def test_anchor_destroy(self):
        anchor = SpatialAnchor()
        anchor.create()
        result = anchor.destroy()

        assert result is True
        assert not anchor.is_active
        assert not anchor.is_tracking

    def test_anchor_destroy_inactive(self):
        anchor = SpatialAnchor()
        result = anchor.destroy()
        assert result is False

    def test_anchor_update_pose(self):
        anchor = SpatialAnchor()
        anchor.create()

        new_pos = Vec3(5.0, 6.0, 7.0)
        new_rot = Quat.identity()
        anchor.update_pose(new_pos, new_rot, confidence=0.9, timestamp=2.0)

        assert anchor.position == new_pos
        assert anchor.orientation == new_rot
        assert anchor.confidence == 0.9

    def test_anchor_update_tracking_state(self):
        anchor = SpatialAnchor()
        anchor.create()

        anchor.update_tracking_state(AnchorTrackingState.LIMITED, confidence=0.5)
        assert anchor.tracking_state == AnchorTrackingState.LIMITED
        assert anchor.confidence == 0.5

    def test_anchor_attach_entity(self):
        anchor = SpatialAnchor()
        anchor.attach_entity(100)
        anchor.attach_entity(200)

        entities = anchor.get_attached_entities()
        assert 100 in entities
        assert 200 in entities
        assert len(entities) == 2

    def test_anchor_detach_entity(self):
        anchor = SpatialAnchor()
        anchor.attach_entity(100)
        anchor.attach_entity(200)
        anchor.detach_entity(100)

        entities = anchor.get_attached_entities()
        assert 100 not in entities
        assert 200 in entities

    def test_anchor_callbacks(self):
        anchor = SpatialAnchor()
        callback_results = []

        def on_pose_updated(a):
            callback_results.append("pose_updated")

        anchor.add_callback("pose_updated", on_pose_updated)
        anchor.create()
        anchor.update_pose(Vec3(1, 2, 3), Quat.identity(), 1.0, 1.0)

        assert "pose_updated" in callback_results

    def test_anchor_remove_callback(self):
        anchor = SpatialAnchor()
        callback_results = []

        def on_tracking(a):
            callback_results.append("tracking")

        anchor.add_callback("tracking_changed", on_tracking)
        anchor.remove_callback("tracking_changed", on_tracking)
        anchor.create()
        anchor.update_tracking_state(AnchorTrackingState.LOST)

        assert "tracking" not in callback_results


class TestCloudAnchor:
    """Tests for cloud anchor functionality."""

    def test_save_to_cloud(self):
        anchor = SpatialAnchor(anchor_type=AnchorType.CLOUD)
        anchor.create()
        result = anchor.save_to_cloud(expires_in_days=30)

        assert result is True
        assert anchor.persistence_state == AnchorPersistenceState.PENDING_SAVE

    def test_save_to_cloud_non_cloud_anchor(self):
        anchor = SpatialAnchor(anchor_type=AnchorType.LOCAL)
        anchor.create()
        result = anchor.save_to_cloud()
        assert result is False

    def test_resolve_cloud_anchor(self):
        anchor = SpatialAnchor(anchor_type=AnchorType.CLOUD)
        result = anchor.resolve_cloud_anchor("cloud-id-123")

        assert result is True
        assert anchor.persistence_state == AnchorPersistenceState.PENDING_LOAD

    def test_on_cloud_save_complete_success(self):
        anchor = SpatialAnchor(anchor_type=AnchorType.CLOUD)
        anchor.create()
        anchor.save_to_cloud()
        anchor.on_cloud_save_complete("saved-id-456", success=True)

        assert anchor.cloud_anchor_id == "saved-id-456"
        assert anchor.persistence_state == AnchorPersistenceState.SAVED

    def test_on_cloud_save_complete_failure(self):
        anchor = SpatialAnchor(anchor_type=AnchorType.CLOUD)
        anchor.create()
        anchor.save_to_cloud()
        anchor.on_cloud_save_complete("", success=False)

        assert anchor.persistence_state == AnchorPersistenceState.SAVE_FAILED

    def test_on_cloud_resolve_complete_success(self):
        anchor = SpatialAnchor(anchor_type=AnchorType.CLOUD)
        anchor.resolve_cloud_anchor("cloud-id-123")
        anchor.on_cloud_resolve_complete(success=True)

        assert anchor.persistence_state == AnchorPersistenceState.LOADED
        assert anchor.tracking_state == AnchorTrackingState.TRACKING


class TestAnchorManager:
    """Tests for AnchorManager class."""

    def test_create_anchor(self):
        manager = AnchorManager()
        anchor = manager.create_anchor(
            position=Vec3(1, 2, 3),
            orientation=Quat.identity(),
            anchor_type=AnchorType.LOCAL,
        )

        assert anchor is not None
        assert anchor.is_active
        assert anchor.position == Vec3(1, 2, 3)

    def test_destroy_anchor(self):
        manager = AnchorManager()
        anchor = manager.create_anchor(Vec3.zero(), Quat.identity())

        result = manager.destroy_anchor(anchor.anchor_id)
        assert result is True
        assert manager.get_anchor(anchor.anchor_id) is None

    def test_destroy_nonexistent_anchor(self):
        manager = AnchorManager()
        result = manager.destroy_anchor("nonexistent-id")
        assert result is False

    def test_get_anchor(self):
        manager = AnchorManager()
        anchor = manager.create_anchor(Vec3.zero(), Quat.identity())

        retrieved = manager.get_anchor(anchor.anchor_id)
        assert retrieved is anchor

    def test_get_all_anchors(self):
        manager = AnchorManager()
        manager.create_anchor(Vec3.zero(), Quat.identity())
        manager.create_anchor(Vec3(1, 0, 0), Quat.identity())
        manager.create_anchor(Vec3(0, 1, 0), Quat.identity())

        all_anchors = manager.get_all_anchors()
        assert len(all_anchors) == 3

    def test_get_anchors_by_type(self):
        manager = AnchorManager()
        manager.create_anchor(Vec3.zero(), Quat.identity(), AnchorType.LOCAL)
        manager.create_anchor(Vec3(1, 0, 0), Quat.identity(), AnchorType.PERSISTENT)
        manager.create_anchor(Vec3(0, 1, 0), Quat.identity(), AnchorType.LOCAL)

        local_anchors = manager.get_anchors_by_type(AnchorType.LOCAL)
        assert len(local_anchors) == 2

    def test_get_tracking_anchors(self):
        manager = AnchorManager()
        a1 = manager.create_anchor(Vec3.zero(), Quat.identity())
        a2 = manager.create_anchor(Vec3(1, 0, 0), Quat.identity())
        a2.update_tracking_state(AnchorTrackingState.LOST)

        tracking = manager.get_tracking_anchors()
        assert len(tracking) == 1
        assert a1 in tracking

    def test_get_anchors_near(self):
        manager = AnchorManager()
        manager.create_anchor(Vec3.zero(), Quat.identity())
        manager.create_anchor(Vec3(1, 0, 0), Quat.identity())
        manager.create_anchor(Vec3(10, 0, 0), Quat.identity())

        nearby = manager.get_anchors_near(Vec3.zero(), max_distance=5.0)
        assert len(nearby) == 2

    def test_save_persistent_anchors(self):
        manager = AnchorManager()
        manager.create_anchor(Vec3.zero(), Quat.identity(), AnchorType.PERSISTENT)
        manager.create_anchor(Vec3(1, 0, 0), Quat.identity(), AnchorType.LOCAL)

        count = manager.save_persistent_anchors()
        assert count == 1

    def test_enable_cloud_anchors(self):
        manager = AnchorManager()
        manager.enable_cloud_anchors(True)

        anchor = manager.resolve_cloud_anchor("cloud-id-test")
        assert anchor is not None

    def test_resolve_cloud_anchor_disabled(self):
        manager = AnchorManager()
        manager.enable_cloud_anchors(False)

        anchor = manager.resolve_cloud_anchor("cloud-id-test")
        assert anchor is None

    def test_clear_all(self):
        manager = AnchorManager()
        manager.create_anchor(Vec3.zero(), Quat.identity())
        manager.create_anchor(Vec3(1, 0, 0), Quat.identity())

        manager.clear_all()
        assert len(manager.get_all_anchors()) == 0

    def test_update_decays_confidence(self):
        manager = AnchorManager()
        anchor = manager.create_anchor(Vec3.zero(), Quat.identity())
        anchor.update_tracking_state(AnchorTrackingState.LIMITED, confidence=0.5)

        # Simulate time passing
        manager.update(delta_time=10.0)

        # After decay, should be LOST
        assert anchor.tracking_state == AnchorTrackingState.LOST
