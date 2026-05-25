"""
Comprehensive tests for AssetMeta - Metaclass for asset handle types.

Tests cover:
- Asset ID assignment
- Extension mapping (get_for_extension)
- Extension conflict detection (raises TypeError)
- get_for_path
- _asset_type_code generation
- Hot-reload flag
- get_supported_extensions
- get_hot_reloadable
- Registry clearing
"""
import pytest

from trinity.metaclasses import AssetMeta
from trinity.constants import ASSET_TYPE_CODE_LENGTH


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear registry before and after each test."""
    AssetMeta.clear_registry()
    yield
    AssetMeta.clear_registry()


def test_asset_id_assignment():
    """Test that asset IDs are assigned sequentially."""

    class Asset1(metaclass=AssetMeta):
        _asset_extensions = (".png",)

    class Asset2(metaclass=AssetMeta):
        _asset_extensions = (".jpg",)

    class Asset3(metaclass=AssetMeta):
        _asset_extensions = (".wav",)

    assert Asset1._asset_id == 1
    assert Asset2._asset_id == 2
    assert Asset3._asset_id == 3


def test_asset_qualified_name():
    """Test that asset qualified name includes module."""

    class TestAsset(metaclass=AssetMeta):
        _asset_extensions = (".test",)

    assert "." in TestAsset._asset_name
    assert TestAsset._asset_name.endswith(".TestAsset")


def test_asset_type_code_generation():
    """Test that _asset_type_code is generated from class name."""

    class TextureAsset(metaclass=AssetMeta):
        _asset_extensions = (".png",)

    # Should be first ASSET_TYPE_CODE_LENGTH chars, uppercase
    assert TextureAsset._asset_type_code == "TEXTUREA"
    assert len(TextureAsset._asset_type_code) == ASSET_TYPE_CODE_LENGTH


def test_extensions_required():
    """Test that _asset_extensions is required."""

    with pytest.raises(TypeError, match="must define _asset_extensions"):

        class NoExtensions(metaclass=AssetMeta):
            pass


def test_extensions_normalization():
    """Test that extensions are normalized (lowercase, leading dot)."""

    class TestAsset(metaclass=AssetMeta):
        _asset_extensions = ("PNG", "jpg", ".gif")

    # All should be lowercase with leading dot
    assert ".png" in TestAsset._asset_extensions
    assert ".jpg" in TestAsset._asset_extensions
    assert ".gif" in TestAsset._asset_extensions


def test_extension_mapping():
    """Test that extensions are mapped to asset types."""

    class ImageAsset(metaclass=AssetMeta):
        _asset_extensions = (".png", ".jpg")

    assert AssetMeta.get_for_extension(".png") is ImageAsset
    assert AssetMeta.get_for_extension(".jpg") is ImageAsset


def test_extension_conflict_detection():
    """Test that duplicate extension registration raises TypeError."""

    class Asset1(metaclass=AssetMeta):
        _asset_extensions = (".png",)

    with pytest.raises(TypeError, match="already registered"):

        class Asset2(metaclass=AssetMeta):
            _asset_extensions = (".png",)


def test_get_for_extension_basic():
    """Test get_for_extension retrieves correct asset type."""

    class TextAsset(metaclass=AssetMeta):
        _asset_extensions = (".txt",)

    retrieved = AssetMeta.get_for_extension(".txt")
    assert retrieved is TextAsset


def test_get_for_extension_case_insensitive():
    """Test get_for_extension is case-insensitive."""

    class ImageAsset(metaclass=AssetMeta):
        _asset_extensions = (".png",)

    assert AssetMeta.get_for_extension(".PNG") is ImageAsset
    assert AssetMeta.get_for_extension(".Png") is ImageAsset


def test_get_for_extension_without_dot():
    """Test get_for_extension works without leading dot."""

    class AudioAsset(metaclass=AssetMeta):
        _asset_extensions = (".wav",)

    assert AssetMeta.get_for_extension("wav") is AudioAsset


def test_get_for_extension_not_found():
    """Test get_for_extension returns None for unknown extension."""

    assert AssetMeta.get_for_extension(".unknown") is None


def test_get_for_path_basic():
    """Test get_for_path extracts extension and finds asset type."""

    class ImageAsset(metaclass=AssetMeta):
        _asset_extensions = (".png",)

    retrieved = AssetMeta.get_for_path("path/to/image.png")
    assert retrieved is ImageAsset


def test_get_for_path_case_insensitive():
    """Test get_for_path is case-insensitive."""

    class ImageAsset(metaclass=AssetMeta):
        _asset_extensions = (".png",)

    assert AssetMeta.get_for_path("image.PNG") is ImageAsset


def test_get_for_path_multiple_dots():
    """Test get_for_path handles multiple dots in filename."""

    class ArchiveAsset(metaclass=AssetMeta):
        _asset_extensions = (".gz",)

    # Should use the last extension
    retrieved = AssetMeta.get_for_path("file.tar.gz")
    assert retrieved is ArchiveAsset


def test_get_for_path_no_extension():
    """Test get_for_path returns None for files without extension."""

    assert AssetMeta.get_for_path("noextension") is None


def test_hot_reload_default():
    """Test that _asset_hot_reload defaults to False."""

    class TestAsset(metaclass=AssetMeta):
        _asset_extensions = (".test",)

    assert TestAsset._asset_hot_reload is False


def test_hot_reload_custom():
    """Test that _asset_hot_reload can be set."""

    class HotAsset(metaclass=AssetMeta):
        _asset_extensions = (".hot",)
        _asset_hot_reload = True

    assert HotAsset._asset_hot_reload is True


def test_get_hot_reloadable():
    """Test get_hot_reloadable returns only hot-reloadable assets."""

    class HotAsset(metaclass=AssetMeta):
        _asset_extensions = (".hot",)
        _asset_hot_reload = True

    class ColdAsset(metaclass=AssetMeta):
        _asset_extensions = (".cold",)
        _asset_hot_reload = False

    hot_assets = AssetMeta.get_hot_reloadable()

    assert HotAsset in hot_assets
    assert ColdAsset not in hot_assets


def test_get_supported_extensions():
    """Test get_supported_extensions returns all registered extensions."""

    class Asset1(metaclass=AssetMeta):
        _asset_extensions = (".png", ".jpg")

    class Asset2(metaclass=AssetMeta):
        _asset_extensions = (".wav",)

    extensions = AssetMeta.get_supported_extensions()

    assert ".png" in extensions
    assert ".jpg" in extensions
    assert ".wav" in extensions
    assert len(extensions) == 3


def test_priority_default():
    """Test that _asset_priority defaults to 0."""

    class TestAsset(metaclass=AssetMeta):
        _asset_extensions = (".test",)

    assert TestAsset._asset_priority == 0


def test_priority_custom():
    """Test that _asset_priority can be set."""

    class HighPriorityAsset(metaclass=AssetMeta):
        _asset_extensions = (".high",)
        _asset_priority = 10

    assert HighPriorityAsset._asset_priority == 10


def test_dependencies_default():
    """Test that _asset_dependencies defaults to empty tuple."""

    class TestAsset(metaclass=AssetMeta):
        _asset_extensions = (".test",)

    assert TestAsset._asset_dependencies == ()


def test_dependencies_custom():
    """Test that _asset_dependencies can be set."""

    class BaseAsset(metaclass=AssetMeta):
        _asset_extensions = (".base",)

    class DependentAsset(metaclass=AssetMeta):
        _asset_extensions = (".dep",)
        _asset_dependencies = (BaseAsset,)

    assert BaseAsset in DependentAsset._asset_dependencies


def test_loader_default():
    """Test that _asset_loader defaults to None."""

    class TestAsset(metaclass=AssetMeta):
        _asset_extensions = (".test",)

    assert TestAsset._asset_loader is None


def test_loader_custom():
    """Test that _asset_loader can be set."""

    class CustomLoader:
        pass

    class TestAsset(metaclass=AssetMeta):
        _asset_extensions = (".test",)
        _asset_loader = CustomLoader

    assert TestAsset._asset_loader is CustomLoader


def test_get_loader():
    """Test get_loader retrieves the loader class."""

    class CustomLoader:
        pass

    class TestAsset(metaclass=AssetMeta):
        _asset_extensions = (".test",)
        _asset_loader = CustomLoader

    loader = AssetMeta.get_loader(TestAsset)
    assert loader is CustomLoader


def test_cache_policy_default():
    """Test that _asset_cache_policy defaults to CachePolicy."""

    from trinity.types import CachePolicy

    class TestAsset(metaclass=AssetMeta):
        _asset_extensions = (".test",)

    # Should be an instance of CachePolicy
    assert isinstance(TestAsset._asset_cache_policy, CachePolicy)


def test_get_by_id():
    """Test retrieving asset by ID."""

    class TestAsset(metaclass=AssetMeta):
        _asset_extensions = (".test",)

    retrieved = AssetMeta.get_by_id(TestAsset._asset_id)
    assert retrieved is TestAsset


def test_get_by_name():
    """Test retrieving asset by qualified name."""

    class TestAsset(metaclass=AssetMeta):
        _asset_extensions = (".test",)

    retrieved = AssetMeta.get_by_name(TestAsset._asset_name)
    assert retrieved is TestAsset


def test_all_assets():
    """Test all_assets returns all registered assets."""

    class Asset1(metaclass=AssetMeta):
        _asset_extensions = (".a1",)

    class Asset2(metaclass=AssetMeta):
        _asset_extensions = (".a2",)

    all_assets = AssetMeta.all_assets()

    assert len(all_assets) == 2
    assert Asset1 in all_assets
    assert Asset2 in all_assets


def test_clear_registry():
    """Test that clear_registry removes all assets."""

    class Asset1(metaclass=AssetMeta):
        _asset_extensions = (".a1",)

    class Asset2(metaclass=AssetMeta):
        _asset_extensions = (".a2",)

    assert len(AssetMeta.all_assets()) == 2

    AssetMeta.clear_registry()

    assert len(AssetMeta.all_assets()) == 0


def test_clear_registry_clears_extension_map():
    """Test that clear_registry clears the extension map."""

    class TestAsset(metaclass=AssetMeta):
        _asset_extensions = (".test",)

    assert AssetMeta.get_for_extension(".test") is TestAsset

    AssetMeta.clear_registry()

    assert AssetMeta.get_for_extension(".test") is None


def test_clear_registry_resets_id():
    """Test that clear_registry resets ID counter."""

    class Asset1(metaclass=AssetMeta):
        _asset_extensions = (".a1",)

    assert Asset1._asset_id == 1

    AssetMeta.clear_registry()

    class Asset2(metaclass=AssetMeta):
        _asset_extensions = (".a2",)

    assert Asset2._asset_id == 1


def test_base_asset_class_skipped():
    """Test that base Asset class is not registered."""

    class Asset(metaclass=AssetMeta):
        _asset_extensions = (".base",)

    # Base "Asset" should be skipped
    assert len(AssetMeta.all_assets()) == 0


def test_multiple_extensions_same_asset():
    """Test that one asset can handle multiple extensions."""

    class ImageAsset(metaclass=AssetMeta):
        _asset_extensions = (".png", ".jpg", ".jpeg", ".bmp")

    assert AssetMeta.get_for_extension(".png") is ImageAsset
    assert AssetMeta.get_for_extension(".jpg") is ImageAsset
    assert AssetMeta.get_for_extension(".jpeg") is ImageAsset
    assert AssetMeta.get_for_extension(".bmp") is ImageAsset


def test_extension_normalization_tuple():
    """Test that extensions are returned as tuple."""

    class TestAsset(metaclass=AssetMeta):
        _asset_extensions = (".png", ".jpg")

    assert isinstance(TestAsset._asset_extensions, tuple)


def test_type_code_truncation():
    """Test that long class names are truncated to ASSET_TYPE_CODE_LENGTH."""

    class VeryLongAssetTypeName(metaclass=AssetMeta):
        _asset_extensions = (".long",)

    # Should be truncated to 8 chars
    assert len(VeryLongAssetTypeName._asset_type_code) == ASSET_TYPE_CODE_LENGTH
    assert VeryLongAssetTypeName._asset_type_code == "VERYLONG"


# =============================================================================
# ASYNC LOADING QUEUE EDGE CASES
# =============================================================================


def test_queue_load_with_none_path():
    """Test queue_load rejects None path."""

    class TestAsset(metaclass=AssetMeta):
        _asset_extensions = (".test",)

    with pytest.raises(ValueError, match="path cannot be None or empty"):
        AssetMeta.queue_load(TestAsset, None)


def test_queue_load_with_empty_path():
    """Test queue_load rejects empty path."""

    class TestAsset(metaclass=AssetMeta):
        _asset_extensions = (".test",)

    with pytest.raises(ValueError, match="path cannot be None or empty"):
        AssetMeta.queue_load(TestAsset, "")


def test_process_queue_empty():
    """Test process_queue on empty queue returns 0."""
    processed = AssetMeta.process_queue()
    assert processed == 0


def test_process_queue_respects_max_items():
    """Test process_queue respects max_items limit."""

    class TestAsset(metaclass=AssetMeta):
        _asset_extensions = (".test",)

    # Queue 20 items
    for i in range(20):
        AssetMeta.queue_load(TestAsset, f"file{i}.test")

    # Process max 5
    processed = AssetMeta.process_queue(max_items=5)
    assert processed == 5

    status = AssetMeta.get_queue_status()
    assert status["pending"] == 15


def test_queue_callback_exception_handling():
    """Test that callback exceptions don't crash process_queue."""

    class TestAsset(metaclass=AssetMeta):
        _asset_extensions = (".test",)

    def bad_callback(asset_cls, path):
        raise RuntimeError("Callback failed")

    AssetMeta.queue_load(TestAsset, "test.test", callback=bad_callback)

    # Should not raise
    processed = AssetMeta.process_queue()
    assert processed == 1


def test_queue_priority_ordering():
    """Test that higher priority items are processed first."""

    class TestAsset(metaclass=AssetMeta):
        _asset_extensions = (".test",)

    processed_order = []

    def track_callback(asset_cls, path):
        processed_order.append(path)

    # Queue with different priorities
    AssetMeta.queue_load(TestAsset, "low.test", priority=1, callback=track_callback)
    AssetMeta.queue_load(TestAsset, "high.test", priority=10, callback=track_callback)
    AssetMeta.queue_load(TestAsset, "medium.test", priority=5, callback=track_callback)

    AssetMeta.process_queue(max_items=10)

    # Should be processed high to low priority
    assert processed_order == ["high.test", "medium.test", "low.test"]


# =============================================================================
# HOT-RELOAD WATCHER EDGE CASES
# =============================================================================


def test_watch_non_existent_file():
    """Test watching a non-existent file doesn't crash."""
    import tempfile
    import os

    class TestAsset(metaclass=AssetMeta):
        _asset_extensions = (".test",)

    non_existent = os.path.join(tempfile.gettempdir(), "does_not_exist.test")

    # Should not raise
    AssetMeta.watch(TestAsset, non_existent)


def test_check_changes_on_deleted_file():
    """Test check_changes handles deleted files gracefully."""
    import tempfile
    import os

    class TestAsset(metaclass=AssetMeta):
        _asset_extensions = (".test",)

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".test", delete=False) as f:
        temp_path = f.name
        f.write("content")

    try:
        # Watch it
        AssetMeta.watch(TestAsset, temp_path)

        # Delete it
        os.unlink(temp_path)

        # Check for changes - should not crash and should unwatch
        changes = AssetMeta.check_changes()
        assert len(changes) == 0

        # Verify it was unwatched
        changes = AssetMeta.check_changes()
        assert len(changes) == 0
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_unwatch_non_watched_file():
    """Test unwatch on non-watched file is a no-op."""
    AssetMeta.unwatch("/some/random/path.test")  # Should not crash


# =============================================================================
# DEPENDENCY LOADING EDGE CASES
# =============================================================================


def test_get_load_order_no_dependencies():
    """Test get_load_order with no dependencies."""

    class SimpleAsset(metaclass=AssetMeta):
        _asset_extensions = (".simple",)

    order = AssetMeta.get_load_order(SimpleAsset)
    assert order == [SimpleAsset]


def test_get_load_order_circular_dependency():
    """Test get_load_order detects circular dependencies."""

    class AssetA(metaclass=AssetMeta):
        _asset_extensions = (".a",)

    class AssetB(metaclass=AssetMeta):
        _asset_extensions = (".b",)

    # Create circular dependency via monkey-patching
    AssetA._asset_dependencies = (AssetB,)
    AssetB._asset_dependencies = (AssetA,)

    with pytest.raises(ValueError, match="Circular dependency detected"):
        AssetMeta.get_load_order(AssetA)


def test_get_load_order_self_dependency():
    """Test get_load_order detects self-dependency."""

    class SelfDepAsset(metaclass=AssetMeta):
        _asset_extensions = (".self",)

    # Create self-dependency via monkey-patching
    SelfDepAsset._asset_dependencies = (SelfDepAsset,)

    with pytest.raises(ValueError, match="Circular dependency detected"):
        AssetMeta.get_load_order(SelfDepAsset)


def test_get_load_order_complex_graph():
    """Test get_load_order with complex dependency graph."""

    class BaseAsset(metaclass=AssetMeta):
        _asset_extensions = (".base",)

    class MiddleAsset(metaclass=AssetMeta):
        _asset_extensions = (".middle",)
        _asset_dependencies = (BaseAsset,)

    class TopAsset(metaclass=AssetMeta):
        _asset_extensions = (".top",)
        _asset_dependencies = (MiddleAsset, BaseAsset)

    order = AssetMeta.get_load_order(TopAsset)

    # BaseAsset must come before MiddleAsset
    assert order.index(BaseAsset) < order.index(MiddleAsset)
    # MiddleAsset must come before TopAsset
    assert order.index(MiddleAsset) < order.index(TopAsset)
    # All should be present
    assert set(order) == {BaseAsset, MiddleAsset, TopAsset}


def test_get_load_order_diamond_dependency():
    """Test get_load_order with diamond-shaped dependency graph."""

    class BaseAsset(metaclass=AssetMeta):
        _asset_extensions = (".base",)

    class LeftAsset(metaclass=AssetMeta):
        _asset_extensions = (".left",)
        _asset_dependencies = (BaseAsset,)

    class RightAsset(metaclass=AssetMeta):
        _asset_extensions = (".right",)
        _asset_dependencies = (BaseAsset,)

    class TopAsset(metaclass=AssetMeta):
        _asset_extensions = (".top",)
        _asset_dependencies = (LeftAsset, RightAsset)

    order = AssetMeta.get_load_order(TopAsset)

    # BaseAsset must come first
    assert order[0] == BaseAsset
    # TopAsset must come last
    assert order[-1] == TopAsset
    # No duplicates
    assert len(order) == len(set(order))
