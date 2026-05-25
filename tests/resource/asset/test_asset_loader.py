"""Tests for AssetLoader, SyncLoader, AsyncLoader."""
import os
import tempfile

import pytest

from engine.resource.asset.asset_loader import (
    AsyncLoader,
    LoadRequest,
    LoadResult,
    SyncLoader,
)
from engine.resource.constants import DEFAULT_LOAD_PRIORITY


class TestLoadResult:
    def test_ok_result(self) -> None:
        r = LoadResult.ok("data")
        assert r.success is True
        assert r.data == "data"
        assert r.error is None

    def test_fail_result(self) -> None:
        r = LoadResult.fail("oops")
        assert r.success is False
        assert r.error == "oops"
        assert r.data is None


class TestLoadRequest:
    def test_defaults(self) -> None:
        req = LoadRequest(path="x.png", asset_type=bytes)
        assert req.priority == DEFAULT_LOAD_PRIORITY
        assert req.callback is None


class TestSyncLoader:
    def test_load_existing_file(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"\x01\x02\x03")
            path = f.name
        try:
            loader = SyncLoader()
            result = loader.load(path, bytes)
            assert result.success
            assert result.data == b"\x01\x02\x03"
        finally:
            os.unlink(path)

    def test_load_missing_file(self) -> None:
        loader = SyncLoader()
        result = loader.load("/nonexistent/file.xyz", bytes)
        assert not result.success
        assert result.error is not None

    def test_unload_is_noop(self) -> None:
        loader = SyncLoader()
        assert loader.unload(b"data") is None


class TestAsyncLoader:
    def test_async_load(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"async")
            path = f.name
        try:
            loader = AsyncLoader(max_workers=1)
            req = LoadRequest(path=path, asset_type=bytes)
            future = loader.load_async(req)
            result = future.result(timeout=5)
            assert result.success
            assert result.data == b"async"
            loader.shutdown()
        finally:
            os.unlink(path)

    def test_async_load_with_callback(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"cb")
            path = f.name
        results: list[LoadResult] = []
        try:
            loader = AsyncLoader(max_workers=1)
            req = LoadRequest(path=path, asset_type=bytes, callback=results.append)
            future = loader.load_async(req)
            future.result(timeout=5)
            loader.shutdown()
            assert len(results) == 1
            assert results[0].success
        finally:
            os.unlink(path)

    def test_async_load_failure(self) -> None:
        loader = AsyncLoader(max_workers=1)
        req = LoadRequest(path="/no/such/file.bin", asset_type=bytes)
        future = loader.load_async(req)
        result = future.result(timeout=5)
        assert not result.success
        loader.shutdown()
