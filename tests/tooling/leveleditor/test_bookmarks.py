"""
Tests for the bookmarks module.

Tests save/load camera positions.
"""

import pytest
import sys

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.tooling.leveleditor.bookmarks import (
    CameraBookmark,
    BookmarkManager,
    BookmarkCategory,
    CameraSettings,
    TransitionSettings,
    TransitionType,
    BookmarkIcon,
)
from engine.tooling.leveleditor.placement import Vector3, Quaternion
from foundation.tracker import tracker


@pytest.fixture(autouse=True)
def reset_tracker():
    """Reset tracker state before each test."""
    tracker._dirty.clear()
    tracker._cb_global.clear()
    tracker._cb_type.clear()
    tracker._cb_obj.clear()
    tracker._undo.clear()
    tracker._redo.clear()
    tracker._txn = None
    yield


class TestCameraBookmark:
    """Tests for CameraBookmark class."""

    def test_creation(self):
        """Bookmark should initialize with name and position."""
        bookmark = CameraBookmark("View1", Vector3(10, 5, 20))
        assert bookmark.name == "View1"
        assert bookmark.position.x == 10

    def test_unique_id(self):
        """Each bookmark should have unique ID."""
        bm1 = CameraBookmark("A")
        bm2 = CameraBookmark("B")
        assert bm1.id != bm2.id

    def test_rotation(self):
        """Should store rotation."""
        rot = Quaternion.from_axis_angle(Vector3(0, 1, 0), 1.57)
        bookmark = CameraBookmark("Test", rotation=rot)
        assert bookmark.rotation.y == rot.y

    def test_camera_settings(self):
        """Should store camera settings."""
        bookmark = CameraBookmark("Test")
        assert bookmark.camera_settings.fov == 60.0
        assert bookmark.camera_settings.orthographic is False

    def test_shortcut_key(self):
        """Should store shortcut key."""
        bookmark = CameraBookmark("Test")
        bookmark.shortcut_key = "F1"
        assert bookmark.shortcut_key == "F1"

    def test_description(self):
        """Should store description."""
        bookmark = CameraBookmark("Test")
        bookmark.description = "Main gameplay view"
        assert bookmark.description == "Main gameplay view"

    def test_tags(self):
        """Should manage tags."""
        bookmark = CameraBookmark("Test")
        bookmark.add_tag("important")
        bookmark.add_tag("gameplay")

        assert "important" in bookmark.tags
        assert "gameplay" in bookmark.tags

        bookmark.remove_tag("important")
        assert "important" not in bookmark.tags

    def test_icon(self):
        """Should store icon."""
        bookmark = CameraBookmark("Test")
        bookmark.icon = BookmarkIcon.STAR
        assert bookmark.icon == BookmarkIcon.STAR

    def test_update_from_camera(self):
        """Should update from camera state."""
        bookmark = CameraBookmark("Test")
        new_pos = Vector3(100, 50, 200)
        new_rot = Quaternion.from_axis_angle(Vector3(1, 0, 0), 0.5)

        bookmark.update_from_camera(new_pos, new_rot)

        assert bookmark.position.x == 100
        assert bookmark.rotation.x == new_rot.x

    def test_to_dict(self):
        """Should serialize to dictionary."""
        bookmark = CameraBookmark("Test", Vector3(10, 20, 30))
        data = bookmark.to_dict()

        assert data["name"] == "Test"
        assert data["position"]["x"] == 10
        assert "rotation" in data
        assert "camera_settings" in data

    def test_from_dict(self):
        """Should deserialize from dictionary."""
        data = {
            "id": "test-id",
            "name": "Restored",
            "position": {"x": 5, "y": 10, "z": 15},
            "rotation": {"x": 0, "y": 0, "z": 0, "w": 1},
            "camera_settings": {"fov": 90.0},
            "icon": "STAR",
            "shortcut_key": "F5",
        }

        bookmark = CameraBookmark.from_dict(data)

        assert bookmark.id == "test-id"
        assert bookmark.name == "Restored"
        assert bookmark.position.x == 5
        assert bookmark.camera_settings.fov == 90.0
        assert bookmark.icon == BookmarkIcon.STAR
        assert bookmark.shortcut_key == "F5"

    def test_timestamps(self):
        """Should track creation and modification times."""
        bookmark = CameraBookmark("Test")
        created = bookmark.created_at

        bookmark.name = "Updated"

        assert bookmark.created_at == created
        assert bookmark.modified_at >= created


class TestBookmarkCategory:
    """Tests for BookmarkCategory class."""

    def test_creation(self):
        """Category should initialize with name."""
        category = BookmarkCategory("Gameplay Views")
        assert category.name == "Gameplay Views"
        assert category.icon == BookmarkIcon.BOOKMARK

    def test_unique_id(self):
        """Each category should have unique ID."""
        cat1 = BookmarkCategory("A")
        cat2 = BookmarkCategory("B")
        assert cat1.id != cat2.id

    def test_icon(self):
        """Should store icon."""
        category = BookmarkCategory("Test", BookmarkIcon.STAR)
        assert category.icon == BookmarkIcon.STAR

    def test_color(self):
        """Should store custom color."""
        category = BookmarkCategory("Test")
        category.set_color(1.0, 0.5, 0.0)
        assert category.color == (1.0, 0.5, 0.0)

    def test_expanded(self):
        """Should track expanded state."""
        category = BookmarkCategory("Test")
        assert category.expanded is True

        category.expanded = False
        assert category.expanded is False


class TestBookmarkManager:
    """Tests for BookmarkManager class."""

    def test_creation(self):
        """Manager should initialize with default category."""
        manager = BookmarkManager()
        assert manager.category_count == 1
        assert manager.bookmark_count == 0

    def test_create_category(self):
        """Should create new category."""
        manager = BookmarkManager()
        category = manager.create_category("Custom")

        assert category.name == "Custom"
        assert manager.get_category(category.id) is category

    def test_delete_category(self):
        """Should delete category."""
        manager = BookmarkManager()
        category = manager.create_category("ToDelete")

        result = manager.delete_category(category.id)

        assert result is True
        assert manager.get_category(category.id) is None

    def test_delete_moves_bookmarks(self):
        """Deleting category should move bookmarks."""
        manager = BookmarkManager()
        category = manager.create_category("ToDelete")
        bookmark = manager.create_bookmark(
            "Test",
            Vector3(0, 0, 0),
            Quaternion.identity(),
            category.id
        )

        manager.delete_category(category.id)

        # Bookmark should be moved to default category
        categories = manager.get_all_categories()
        assert bookmark.category_id == categories[0].id

    def test_create_bookmark(self):
        """Should create bookmark."""
        manager = BookmarkManager()
        bookmark = manager.create_bookmark(
            "TestView",
            Vector3(10, 20, 30),
            Quaternion.identity()
        )

        assert bookmark.name == "TestView"
        assert manager.get_bookmark(bookmark.id) is bookmark

    def test_create_bookmark_with_category(self):
        """Should create bookmark in category."""
        manager = BookmarkManager()
        category = manager.create_category("Custom")
        bookmark = manager.create_bookmark(
            "Test",
            Vector3(0, 0, 0),
            Quaternion.identity(),
            category.id
        )

        assert bookmark.category_id == category.id

    def test_delete_bookmark(self):
        """Should delete bookmark."""
        manager = BookmarkManager()
        bookmark = manager.create_bookmark(
            "Test",
            Vector3(0, 0, 0),
            Quaternion.identity()
        )

        result = manager.delete_bookmark(bookmark.id)

        assert result is True
        assert manager.get_bookmark(bookmark.id) is None

    def test_get_bookmark_by_name(self):
        """Should get bookmark by name."""
        manager = BookmarkManager()
        bookmark = manager.create_bookmark(
            "MyView",
            Vector3(0, 0, 0),
            Quaternion.identity()
        )

        found = manager.get_bookmark_by_name("MyView")

        assert found is bookmark

    def test_get_bookmark_by_shortcut(self):
        """Should get bookmark by shortcut."""
        manager = BookmarkManager()
        bookmark = manager.create_bookmark(
            "Test",
            Vector3(0, 0, 0),
            Quaternion.identity()
        )
        bookmark.shortcut_key = "F1"

        found = manager.get_bookmark_by_shortcut("F1")

        assert found is bookmark

    def test_get_bookmarks_in_category(self):
        """Should get bookmarks in category."""
        manager = BookmarkManager()
        category = manager.create_category("Custom")

        manager.create_bookmark("BM1", Vector3(0, 0, 0), Quaternion.identity(), category.id)
        manager.create_bookmark("BM2", Vector3(0, 0, 0), Quaternion.identity(), category.id)
        manager.create_bookmark("BM3", Vector3(0, 0, 0), Quaternion.identity())  # Default

        bookmarks = manager.get_bookmarks_in_category(category.id)

        assert len(bookmarks) == 2

    def test_find_bookmarks_by_tag(self):
        """Should find bookmarks by tag."""
        manager = BookmarkManager()
        bm1 = manager.create_bookmark("BM1", Vector3(0, 0, 0), Quaternion.identity())
        bm1.add_tag("important")
        bm2 = manager.create_bookmark("BM2", Vector3(0, 0, 0), Quaternion.identity())
        bm2.add_tag("important")
        manager.create_bookmark("BM3", Vector3(0, 0, 0), Quaternion.identity())

        found = manager.find_bookmarks_by_tag("important")

        assert len(found) == 2

    def test_search_bookmarks(self):
        """Should search by name and description."""
        manager = BookmarkManager()
        bm1 = manager.create_bookmark("GameplayView", Vector3(0, 0, 0), Quaternion.identity())
        bm2 = manager.create_bookmark("Other", Vector3(0, 0, 0), Quaternion.identity())
        bm2.description = "Used for gameplay testing"

        found = manager.search_bookmarks("gameplay")

        assert len(found) == 2

    def test_navigate_to(self):
        """Should navigate to bookmark."""
        manager = BookmarkManager()
        bookmark = manager.create_bookmark(
            "Test",
            Vector3(100, 50, 200),
            Quaternion.identity()
        )

        result = manager.navigate_to(bookmark.id)

        assert result is not None
        pos, rot, settings, transition = result
        assert pos.x == 100

    def test_navigate_instant(self):
        """Should navigate instantly when specified."""
        manager = BookmarkManager()
        bookmark = manager.create_bookmark(
            "Test",
            Vector3(0, 0, 0),
            Quaternion.identity()
        )

        result = manager.navigate_to(bookmark.id, instant=True)

        pos, rot, settings, transition = result
        assert transition.transition_type == TransitionType.INSTANT
        assert transition.duration == 0

    def test_navigate_back(self):
        """Should navigate back in history."""
        manager = BookmarkManager()
        bm1 = manager.create_bookmark("BM1", Vector3(0, 0, 0), Quaternion.identity())
        bm2 = manager.create_bookmark("BM2", Vector3(10, 0, 0), Quaternion.identity())

        manager.navigate_to(bm1.id)
        manager.navigate_to(bm2.id)

        result = manager.navigate_back()

        assert result is not None
        pos, _, _, _ = result
        assert pos.x == 0

    def test_navigate_forward(self):
        """Should navigate forward in history."""
        manager = BookmarkManager()
        bm1 = manager.create_bookmark("BM1", Vector3(0, 0, 0), Quaternion.identity())
        bm2 = manager.create_bookmark("BM2", Vector3(10, 0, 0), Quaternion.identity())

        manager.navigate_to(bm1.id)
        manager.navigate_to(bm2.id)
        manager.navigate_back()

        result = manager.navigate_forward()

        assert result is not None
        pos, _, _, _ = result
        assert pos.x == 10

    def test_can_go_back(self):
        """Should report if can go back."""
        manager = BookmarkManager()
        bm1 = manager.create_bookmark("BM1", Vector3(0, 0, 0), Quaternion.identity())

        assert manager.can_go_back() is False

        manager.navigate_to(bm1.id)
        manager.create_bookmark("BM2", Vector3(0, 0, 0), Quaternion.identity())

        # Still can't go back with only one item
        assert manager.can_go_back() is False

    def test_can_go_forward(self):
        """Should report if can go forward."""
        manager = BookmarkManager()
        bm1 = manager.create_bookmark("BM1", Vector3(0, 0, 0), Quaternion.identity())
        bm2 = manager.create_bookmark("BM2", Vector3(0, 0, 0), Quaternion.identity())

        manager.navigate_to(bm1.id)
        manager.navigate_to(bm2.id)

        assert manager.can_go_forward() is False

        manager.navigate_back()
        assert manager.can_go_forward() is True

    def test_export_bookmarks(self):
        """Should export all bookmarks."""
        manager = BookmarkManager()
        manager.create_bookmark("BM1", Vector3(10, 20, 30), Quaternion.identity())

        data = manager.export_bookmarks()

        assert "version" in data
        assert "categories" in data
        assert "bookmarks" in data
        assert len(data["bookmarks"]) == 1

    def test_import_bookmarks_merge(self):
        """Should import and merge bookmarks."""
        manager = BookmarkManager()
        manager.create_bookmark("Existing", Vector3(0, 0, 0), Quaternion.identity())

        data = {
            "version": 1,
            "categories": [
                {"id": "cat-1", "name": "Imported", "icon": "STAR"}
            ],
            "bookmarks": [
                {
                    "id": "bm-1",
                    "name": "Imported",
                    "position": {"x": 100, "y": 0, "z": 0},
                    "rotation": {"x": 0, "y": 0, "z": 0, "w": 1},
                }
            ]
        }

        cat_count, bm_count = manager.import_bookmarks(data, merge=True)

        assert cat_count == 1
        assert bm_count == 1
        assert manager.bookmark_count == 2

    def test_import_bookmarks_replace(self):
        """Should import and replace bookmarks."""
        manager = BookmarkManager()
        manager.create_bookmark("Existing", Vector3(0, 0, 0), Quaternion.identity())

        data = {
            "version": 1,
            "categories": [],
            "bookmarks": [
                {
                    "id": "bm-1",
                    "name": "New",
                    "position": {"x": 100, "y": 0, "z": 0},
                    "rotation": {"x": 0, "y": 0, "z": 0, "w": 1},
                }
            ]
        }

        cat_count, bm_count = manager.import_bookmarks(data, merge=False)

        assert bm_count == 1
        assert manager.bookmark_count == 1

    def test_transition_settings(self):
        """Should use transition settings."""
        manager = BookmarkManager()
        manager.transition_settings = TransitionSettings(
            transition_type=TransitionType.LINEAR,
            duration=1.0
        )

        assert manager.transition_settings.transition_type == TransitionType.LINEAR

    def test_callbacks(self):
        """Should trigger callbacks."""
        manager = BookmarkManager()
        events = []

        manager.on("on_bookmark_create", lambda b: events.append("create"))
        manager.on("on_bookmark_delete", lambda b: events.append("delete"))
        manager.on("on_navigate", lambda b: events.append("navigate"))

        bookmark = manager.create_bookmark("Test", Vector3(0, 0, 0), Quaternion.identity())
        manager.navigate_to(bookmark.id)
        manager.delete_bookmark(bookmark.id)

        assert "create" in events
        assert "navigate" in events
        assert "delete" in events

    def test_get_statistics(self):
        """Should return statistics."""
        manager = BookmarkManager()
        bm = manager.create_bookmark("Test", Vector3(0, 0, 0), Quaternion.identity())
        bm.shortcut_key = "F1"

        stats = manager.get_statistics()

        assert stats["total_bookmarks"] == 1
        assert stats["bookmarks_with_shortcuts"] == 1
