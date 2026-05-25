"""Tests for AssetManager."""
import os
import tempfile

import pytest

from engine.resource.asset.asset_handle import AssetHandle, AssetState
from engine.resource.asset.asset_loader import AssetLoader, LoadResult, SyncLoader
from engine.resource.asset.asset_manager import AssetManager


class FakeLoader(AssetLoader):
    """Loader that returns the path as data."""
    __slots__ = ("_fail_paths",)

    def __init__(self, fail_paths: set[str] | None = None) -> None:
        self._fail_paths: set[str] = fail_paths or set()

    def load(self, path: str, asset_type: type) -> LoadResult:
        if path in self._fail_paths:
            return LoadResult.fail(f"Cannot load {path}")
        return LoadResult.ok(f"data:{path}")

    def unload(self, data: object) -> None:
        pass


class TestAssetManager:
    def test_load_returns_valid_handle(self) -> None:
        mgr = AssetManager(FakeLoader())
        h = mgr.load("test.png", bytes)
        assert h.is_valid()

    def test_state_queued_before_update(self) -> None:
        mgr = AssetManager(FakeLoader())
        h = mgr.load("a.png")
        assert mgr.get_state(h) == AssetState.QUEUED

    def test_state_ready_after_update(self) -> None:
        mgr = AssetManager(FakeLoader())
        h = mgr.load("a.png")
        mgr.update()
        assert mgr.get_state(h) == AssetState.READY
        assert mgr.is_loaded(h)

    def test_get_returns_data(self) -> None:
        mgr = AssetManager(FakeLoader())
        h = mgr.load("hero.obj")
        mgr.update()
        assert mgr.get(h) == "data:hero.obj"

    def test_get_returns_none_before_load(self) -> None:
        mgr = AssetManager(FakeLoader())
        h = mgr.load("hero.obj")
        assert mgr.get(h) is None

    def test_unload_clears_data(self) -> None:
        mgr = AssetManager(FakeLoader())
        h = mgr.load("x.wav")
        mgr.update()
        mgr.unload(h)
        assert mgr.get(h) is None
        assert mgr.get_state(h) == AssetState.UNLOADED

    def test_ref_counting_keeps_alive(self) -> None:
        mgr = AssetManager(FakeLoader())
        h1 = mgr.load("shared.png")
        h2 = mgr.load("shared.png")  # same path -> ref count 2
        mgr.update()
        mgr.unload(h1)
        # Still alive because h2 holds a ref
        assert mgr.get_state(h2) == AssetState.READY

    def test_ref_counting_unloads_at_zero(self) -> None:
        mgr = AssetManager(FakeLoader())
        h1 = mgr.load("shared.png")
        h2 = mgr.load("shared.png")
        mgr.update()
        mgr.unload(h1)
        mgr.unload(h2)
        assert mgr.get_state(h1) == AssetState.UNLOADED

    def test_failed_load(self) -> None:
        mgr = AssetManager(FakeLoader(fail_paths={"bad.dat"}))
        h = mgr.load("bad.dat")
        mgr.update()
        assert mgr.get_state(h) == AssetState.FAILED
        assert mgr.get(h) is None

    def test_slot_reuse_after_unload(self) -> None:
        mgr = AssetManager(FakeLoader())
        h1 = mgr.load("a.png")
        mgr.update()
        idx1 = h1.index
        mgr.unload(h1)
        h2 = mgr.load("b.png")
        assert h2.index == idx1  # reused slot
        assert h2.generation == h1.generation + 1

    def test_null_handle_get_returns_none(self) -> None:
        mgr = AssetManager(FakeLoader())
        assert mgr.get(AssetHandle.null()) is None

    def test_stale_handle_after_unload_reload(self) -> None:
        mgr = AssetManager(FakeLoader())
        h1 = mgr.load("a.png")
        mgr.update()
        mgr.unload(h1)
        h2 = mgr.load("b.png")
        mgr.update()
        # h1 is stale (old generation)
        assert mgr.get(h1) is None
        assert mgr.get(h2) == "data:b.png"

    def test_sync_loader_with_real_file(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"hello")
            path = f.name
        try:
            mgr = AssetManager(SyncLoader())
            h = mgr.load(path, bytes)
            mgr.update()
            assert mgr.get(h) == b"hello"
        finally:
            os.unlink(path)
