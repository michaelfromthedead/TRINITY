"""
Comprehensive tests for ContentBrowser functionality.

Tests navigation, filtering, selection, favorites, history, and drag-drop.
"""

import pytest
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.tooling.assettools.content_browser import (
    ContentBrowser,
    BrowserItem,
    BrowserFilter,
    BrowserFavorites,
    BrowserHistory,
    DragDropPayload,
    AssetType,
    SortOrder,
    _get_asset_type,
)


@pytest.fixture
def temp_asset_dir():
    """Create a temporary directory with test assets."""
    path = Path(tempfile.mkdtemp())

    # Create directory structure
    (path / "textures").mkdir()
    (path / "models").mkdir()
    (path / "audio").mkdir()
    (path / "models" / "characters").mkdir()

    # Create test files
    (path / "textures" / "hero_diffuse.png").write_text("png data")
    (path / "textures" / "hero_normal.png").write_text("png data")
    (path / "textures" / "environment.jpg").write_text("jpg data")
    (path / "textures" / "sky.hdr").write_text("hdr data")
    (path / "models" / "hero.fbx").write_text("fbx data")
    (path / "models" / "weapon.obj").write_text("obj data")
    (path / "models" / "characters" / "npc.glb").write_text("glb data")
    (path / "audio" / "music.ogg").write_text("ogg data")
    (path / "audio" / "effect.wav").write_text("wav data")
    (path / "readme.txt").write_text("readme")
    (path / ".hidden_file").write_text("hidden")

    yield path
    shutil.rmtree(path)


class TestAssetType:
    """Test asset type detection."""

    def test_mesh_types(self, temp_asset_dir):
        """Mesh extensions should return MESH type."""
        assert _get_asset_type(Path("model.fbx")) == AssetType.MESH
        assert _get_asset_type(Path("model.obj")) == AssetType.MESH
        assert _get_asset_type(Path("model.gltf")) == AssetType.MESH
        assert _get_asset_type(Path("model.glb")) == AssetType.MESH

    def test_texture_types(self):
        """Texture extensions should return TEXTURE type."""
        assert _get_asset_type(Path("tex.png")) == AssetType.TEXTURE
        assert _get_asset_type(Path("tex.jpg")) == AssetType.TEXTURE
        assert _get_asset_type(Path("tex.tga")) == AssetType.TEXTURE
        assert _get_asset_type(Path("tex.dds")) == AssetType.TEXTURE

    def test_audio_types(self):
        """Audio extensions should return AUDIO type."""
        assert _get_asset_type(Path("sound.wav")) == AssetType.AUDIO
        assert _get_asset_type(Path("sound.ogg")) == AssetType.AUDIO
        assert _get_asset_type(Path("sound.mp3")) == AssetType.AUDIO

    def test_folder_type(self, temp_asset_dir):
        """Directories should return FOLDER type."""
        assert _get_asset_type(temp_asset_dir / "textures") == AssetType.FOLDER

    def test_unknown_type(self):
        """Unknown extensions should return UNKNOWN type."""
        assert _get_asset_type(Path("file.xyz")) == AssetType.UNKNOWN


class TestBrowserItem:
    """Test BrowserItem dataclass."""

    def test_browser_item_creation(self):
        """BrowserItem should store all attributes."""
        item = BrowserItem(
            path=Path("/test/file.png"),
            name="file.png",
            asset_type=AssetType.TEXTURE,
            size_bytes=1024,
            modified_time=1000.0,
        )

        assert item.path == Path("/test/file.png")
        assert item.name == "file.png"
        assert item.asset_type == AssetType.TEXTURE
        assert item.size_bytes == 1024

    def test_extension_property(self):
        """extension property should return extension without dot."""
        item = BrowserItem(path=Path("/test/file.PNG"), name="file.PNG", asset_type=AssetType.TEXTURE)
        assert item.extension == "png"

    def test_modified_datetime(self):
        """modified_datetime should convert timestamp."""
        item = BrowserItem(
            path=Path("/test/file.png"),
            name="file.png",
            asset_type=AssetType.TEXTURE,
            modified_time=1000000.0,
        )
        assert isinstance(item.modified_datetime, datetime)

    def test_item_equality(self):
        """Items with same path should be equal."""
        item1 = BrowserItem(path=Path("/test/file.png"), name="file.png", asset_type=AssetType.TEXTURE)
        item2 = BrowserItem(path=Path("/test/file.png"), name="file.png", asset_type=AssetType.TEXTURE)
        assert item1 == item2

    def test_item_hashable(self):
        """Items should be hashable for use in sets."""
        item1 = BrowserItem(path=Path("/test/a.png"), name="a.png", asset_type=AssetType.TEXTURE)
        item2 = BrowserItem(path=Path("/test/b.png"), name="b.png", asset_type=AssetType.TEXTURE)
        items = {item1, item2}
        assert len(items) == 2


class TestBrowserFilter:
    """Test BrowserFilter functionality."""

    def test_empty_filter_matches_all(self):
        """Empty filter should match all items."""
        filter = BrowserFilter()
        item = BrowserItem(
            path=Path("/test/file.png"),
            name="file.png",
            asset_type=AssetType.TEXTURE,
        )
        assert filter.matches(item)

    def test_asset_type_filter(self):
        """Asset type filter should only match specified types."""
        filter = BrowserFilter(asset_types={AssetType.TEXTURE})

        texture = BrowserItem(path=Path("t.png"), name="t.png", asset_type=AssetType.TEXTURE)
        mesh = BrowserItem(path=Path("m.fbx"), name="m.fbx", asset_type=AssetType.MESH)

        assert filter.matches(texture)
        assert not filter.matches(mesh)

    def test_extension_filter(self):
        """Extension filter should only match specified extensions."""
        filter = BrowserFilter(extensions={"png", "jpg"})

        png = BrowserItem(path=Path("t.png"), name="t.png", asset_type=AssetType.TEXTURE)
        tga = BrowserItem(path=Path("t.tga"), name="t.tga", asset_type=AssetType.TEXTURE)

        assert filter.matches(png)
        assert not filter.matches(tga)

    def test_size_filter_min(self):
        """Min size filter should exclude small files."""
        filter = BrowserFilter(min_size=1000)

        small = BrowserItem(path=Path("s.png"), name="s.png", asset_type=AssetType.TEXTURE, size_bytes=500)
        large = BrowserItem(path=Path("l.png"), name="l.png", asset_type=AssetType.TEXTURE, size_bytes=2000)

        assert not filter.matches(small)
        assert filter.matches(large)

    def test_size_filter_max(self):
        """Max size filter should exclude large files."""
        filter = BrowserFilter(max_size=1000)

        small = BrowserItem(path=Path("s.png"), name="s.png", asset_type=AssetType.TEXTURE, size_bytes=500)
        large = BrowserItem(path=Path("l.png"), name="l.png", asset_type=AssetType.TEXTURE, size_bytes=2000)

        assert filter.matches(small)
        assert not filter.matches(large)

    def test_hidden_file_filter(self):
        """Hidden files should be filtered by default."""
        filter = BrowserFilter(include_hidden=False)

        normal = BrowserItem(path=Path("file.png"), name="file.png", asset_type=AssetType.TEXTURE)
        hidden = BrowserItem(path=Path(".hidden"), name=".hidden", asset_type=AssetType.UNKNOWN)

        assert filter.matches(normal)
        assert not filter.matches(hidden)

    def test_name_pattern_filter(self):
        """Name pattern filter should match glob patterns."""
        filter = BrowserFilter(name_pattern="hero*")

        hero = BrowserItem(path=Path("hero_diffuse.png"), name="hero_diffuse.png", asset_type=AssetType.TEXTURE)
        other = BrowserItem(path=Path("enemy.png"), name="enemy.png", asset_type=AssetType.TEXTURE)

        assert filter.matches(hero)
        assert not filter.matches(other)

    def test_date_filter(self):
        """Date filters should filter by modification time."""
        filter = BrowserFilter(modified_after=1000.0, modified_before=2000.0)

        old = BrowserItem(path=Path("o.png"), name="o.png", asset_type=AssetType.TEXTURE, modified_time=500.0)
        mid = BrowserItem(path=Path("m.png"), name="m.png", asset_type=AssetType.TEXTURE, modified_time=1500.0)
        new = BrowserItem(path=Path("n.png"), name="n.png", asset_type=AssetType.TEXTURE, modified_time=2500.0)

        assert not filter.matches(old)
        assert filter.matches(mid)
        assert not filter.matches(new)

    def test_filter_reset(self):
        """reset() should clear all filter settings."""
        filter = BrowserFilter(
            asset_types={AssetType.TEXTURE},
            extensions={"png"},
            min_size=100,
        )

        filter.reset()

        assert len(filter.asset_types) == 0
        assert len(filter.extensions) == 0
        assert filter.min_size is None


class TestBrowserFavorites:
    """Test BrowserFavorites functionality."""

    def test_add_favorite(self):
        """add() should add path to favorites."""
        favorites = BrowserFavorites()
        favorites.add(Path("/test/file.png"))

        assert favorites.is_favorite(Path("/test/file.png"))

    def test_remove_favorite(self):
        """remove() should remove path from favorites."""
        favorites = BrowserFavorites()
        favorites.add(Path("/test/file.png"))
        favorites.remove(Path("/test/file.png"))

        assert not favorites.is_favorite(Path("/test/file.png"))

    def test_toggle_favorite(self):
        """toggle() should toggle favorite status."""
        favorites = BrowserFavorites()

        result1 = favorites.toggle(Path("/test/file.png"))
        assert result1 is True
        assert favorites.is_favorite(Path("/test/file.png"))

        result2 = favorites.toggle(Path("/test/file.png"))
        assert result2 is False
        assert not favorites.is_favorite(Path("/test/file.png"))

    def test_clear_favorites(self):
        """clear() should remove all favorites."""
        favorites = BrowserFavorites()
        favorites.add(Path("/test/a.png"))
        favorites.add(Path("/test/b.png"))

        favorites.clear()

        assert len(favorites.items) == 0

    def test_change_listener(self):
        """Change listeners should be notified."""
        favorites = BrowserFavorites()
        changes = []

        favorites.on_change(lambda path, added: changes.append((path, added)))
        favorites.add(Path("/test/file.png"))
        favorites.remove(Path("/test/file.png"))

        assert len(changes) == 2
        assert changes[0][1] is True
        assert changes[1][1] is False


class TestBrowserHistory:
    """Test BrowserHistory functionality."""

    def test_push_and_current(self):
        """push() should add path and update current."""
        history = BrowserHistory()
        history.push(Path("/dir1"))
        history.push(Path("/dir2"))

        assert history.current() == Path("/dir2")

    def test_back(self):
        """back() should navigate backwards."""
        history = BrowserHistory()
        history.push(Path("/dir1"))
        history.push(Path("/dir2"))

        result = history.back()

        assert result == Path("/dir1")
        assert history.current() == Path("/dir1")

    def test_forward(self):
        """forward() should navigate forwards."""
        history = BrowserHistory()
        history.push(Path("/dir1"))
        history.push(Path("/dir2"))
        history.back()

        result = history.forward()

        assert result == Path("/dir2")
        assert history.current() == Path("/dir2")

    def test_can_go_back(self):
        """can_go_back() should return correct status."""
        history = BrowserHistory()
        assert not history.can_go_back()

        history.push(Path("/dir1"))
        assert not history.can_go_back()

        history.push(Path("/dir2"))
        assert history.can_go_back()

    def test_can_go_forward(self):
        """can_go_forward() should return correct status."""
        history = BrowserHistory()
        history.push(Path("/dir1"))
        history.push(Path("/dir2"))

        assert not history.can_go_forward()

        history.back()
        assert history.can_go_forward()

    def test_push_clears_forward(self):
        """push() should clear forward history."""
        history = BrowserHistory()
        history.push(Path("/dir1"))
        history.push(Path("/dir2"))
        history.back()
        history.push(Path("/dir3"))

        assert not history.can_go_forward()
        assert history.current() == Path("/dir3")

    def test_max_history_size(self):
        """History should respect max_size limit."""
        history = BrowserHistory(max_size=3)

        for i in range(10):
            history.push(Path(f"/dir{i}"))

        # Should only keep last 3
        assert history.current() == Path("/dir9")
        assert len(history._history) == 3

    def test_duplicate_push_ignored(self):
        """Pushing current path should be ignored."""
        history = BrowserHistory()
        history.push(Path("/dir1"))
        history.push(Path("/dir1"))

        assert len(history._history) == 1


class TestDragDropPayload:
    """Test DragDropPayload functionality."""

    def test_payload_creation(self):
        """Payload should store items and metadata."""
        item = BrowserItem(path=Path("/test/file.png"), name="file.png", asset_type=AssetType.TEXTURE)
        payload = DragDropPayload(items=[item], source="browser", operation="copy")

        assert len(payload.items) == 1
        assert payload.source == "browser"
        assert payload.operation == "copy"

    def test_paths_property(self):
        """paths property should return all item paths."""
        items = [
            BrowserItem(path=Path("/a.png"), name="a.png", asset_type=AssetType.TEXTURE),
            BrowserItem(path=Path("/b.png"), name="b.png", asset_type=AssetType.TEXTURE),
        ]
        payload = DragDropPayload(items=items)

        assert len(payload.paths) == 2
        assert Path("/a.png") in payload.paths

    def test_is_single(self):
        """is_single should return True for single item."""
        item = BrowserItem(path=Path("/file.png"), name="file.png", asset_type=AssetType.TEXTURE)

        single = DragDropPayload(items=[item])
        multi = DragDropPayload(items=[item, item])

        assert single.is_single
        assert not multi.is_single

    def test_first(self):
        """first property should return first item or None."""
        item = BrowserItem(path=Path("/file.png"), name="file.png", asset_type=AssetType.TEXTURE)

        with_items = DragDropPayload(items=[item])
        empty = DragDropPayload()

        assert with_items.first == item
        assert empty.first is None


class TestContentBrowser:
    """Test ContentBrowser main class."""

    def test_browser_creation(self, temp_asset_dir):
        """Browser should initialize with root path."""
        browser = ContentBrowser(temp_asset_dir)

        assert browser.root_path == temp_asset_dir
        assert browser.current_path == temp_asset_dir

    def test_get_items(self, temp_asset_dir):
        """get_items() should return items in current directory."""
        browser = ContentBrowser(temp_asset_dir)
        items = browser.get_items()

        # Should include directories and files
        names = [item.name for item in items]
        assert "textures" in names
        assert "models" in names
        assert "readme.txt" in names

    def test_navigate_to(self, temp_asset_dir):
        """navigate_to() should change directory."""
        browser = ContentBrowser(temp_asset_dir)

        success = browser.navigate_to(temp_asset_dir / "textures")

        assert success
        assert browser.current_path == temp_asset_dir / "textures"

    def test_navigate_to_invalid(self, temp_asset_dir):
        """navigate_to() should fail for invalid paths."""
        browser = ContentBrowser(temp_asset_dir)

        # Non-existent path
        assert not browser.navigate_to(temp_asset_dir / "nonexistent")

        # File instead of directory
        assert not browser.navigate_to(temp_asset_dir / "readme.txt")

    def test_navigate_up(self, temp_asset_dir):
        """navigate_up() should go to parent."""
        browser = ContentBrowser(temp_asset_dir)
        browser.navigate_to(temp_asset_dir / "textures")

        success = browser.navigate_up()

        assert success
        assert browser.current_path == temp_asset_dir

    def test_navigate_up_at_root(self, temp_asset_dir):
        """navigate_up() should fail at root."""
        browser = ContentBrowser(temp_asset_dir)

        success = browser.navigate_up()

        assert not success
        assert browser.current_path == temp_asset_dir

    def test_navigate_back_forward(self, temp_asset_dir):
        """navigate_back/forward should work with history."""
        browser = ContentBrowser(temp_asset_dir)
        browser.navigate_to(temp_asset_dir / "textures")
        browser.navigate_to(temp_asset_dir / "models")

        browser.navigate_back()
        assert browser.current_path == temp_asset_dir / "textures"

        browser.navigate_forward()
        assert browser.current_path == temp_asset_dir / "models"

    def test_get_breadcrumbs(self, temp_asset_dir):
        """get_breadcrumbs() should return path trail."""
        browser = ContentBrowser(temp_asset_dir)
        browser.navigate_to(temp_asset_dir / "models" / "characters")

        crumbs = browser.get_breadcrumbs()

        assert len(crumbs) >= 2
        # Last crumb should be current directory
        assert crumbs[-1][1] == temp_asset_dir / "models" / "characters"

    def test_selection(self, temp_asset_dir):
        """Selection operations should work correctly."""
        browser = ContentBrowser(temp_asset_dir)
        items = browser.get_items()

        # Select first item
        browser.select(items[0].path)
        assert browser.is_selected(items[0].path)

        # Add to selection
        browser.add_to_selection(items[1].path)
        assert len(browser.get_selection()) == 2

        # Remove from selection
        browser.remove_from_selection(items[0].path)
        assert len(browser.get_selection()) == 1

        # Clear selection
        browser.clear_selection()
        assert len(browser.get_selection()) == 0

    def test_toggle_selection(self, temp_asset_dir):
        """toggle_selection() should toggle item."""
        browser = ContentBrowser(temp_asset_dir)
        items = browser.get_items()

        result1 = browser.toggle_selection(items[0].path)
        assert result1 is True

        result2 = browser.toggle_selection(items[0].path)
        assert result2 is False

    def test_sort_order(self, temp_asset_dir):
        """set_sort_order() should change sorting."""
        browser = ContentBrowser(temp_asset_dir / "textures")

        browser.set_sort_order(SortOrder.NAME_ASC)
        items_asc = browser.get_items()

        browser.set_sort_order(SortOrder.NAME_DESC)
        items_desc = browser.get_items()

        # Compare non-directory items
        files_asc = [i for i in items_asc if not i.is_directory]
        files_desc = [i for i in items_desc if not i.is_directory]

        if files_asc and files_desc:
            assert files_asc[0].name != files_desc[0].name or len(files_asc) == 1

    def test_filter_application(self, temp_asset_dir):
        """set_filter() should filter items."""
        browser = ContentBrowser(temp_asset_dir / "textures")

        # Filter to PNG only
        filter = BrowserFilter(extensions={"png"})
        browser.set_filter(filter)

        items = browser.get_items()
        for item in items:
            if not item.is_directory:
                assert item.extension == "png"

    def test_create_drag_payload(self, temp_asset_dir):
        """create_drag_payload() should create payload from selection."""
        browser = ContentBrowser(temp_asset_dir)
        items = browser.get_items()

        browser.select([items[0].path, items[1].path])
        payload = browser.create_drag_payload()

        assert len(payload.items) == 2
        assert payload.source == "content_browser"

    def test_refresh(self, temp_asset_dir):
        """refresh() should update items."""
        browser = ContentBrowser(temp_asset_dir)

        # Get initial items
        items1 = browser.get_items()

        # Create new file
        (temp_asset_dir / "new_file.txt").write_text("new")

        browser.refresh()
        items2 = browser.get_items()

        assert len(items2) > len(items1)

    def test_change_listener(self, temp_asset_dir):
        """Change listeners should be notified."""
        browser = ContentBrowser(temp_asset_dir)
        changes = []

        browser.on_change(lambda: changes.append("changed"))
        browser.navigate_to(temp_asset_dir / "textures")

        assert len(changes) > 0

    def test_hidden_files_filtering(self, temp_asset_dir):
        """Hidden files should be filtered by default."""
        browser = ContentBrowser(temp_asset_dir)
        items = browser.get_items()

        names = [item.name for item in items]
        assert ".hidden_file" not in names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
