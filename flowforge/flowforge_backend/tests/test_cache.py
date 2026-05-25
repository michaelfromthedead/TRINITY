"""Tests for ASTCache."""

from __future__ import annotations

import os
import tempfile
import time

import pytest

from ..ast_parser.cache import ASTCache
from ..ast_parser.graph_types import NodeGraph


@pytest.fixture
def cache() -> ASTCache:
    return ASTCache()


@pytest.fixture
def tmp_py_file(tmp_path):
    """Create a temporary Python file and return its path."""
    p = tmp_path / "sample.py"
    p.write_text("x = 1\n")
    return str(p)


def _make_graph() -> NodeGraph:
    return NodeGraph(nodes=[], edges=[], metadata={"test": True})


class TestASTCache:
    def test_miss_returns_none(self, cache: ASTCache, tmp_py_file: str):
        assert cache.get(tmp_py_file) is None

    def test_hit_after_put(self, cache: ASTCache, tmp_py_file: str):
        graph = _make_graph()
        cache.put(tmp_py_file, graph)
        result = cache.get(tmp_py_file)
        assert result is not None
        assert result.metadata == {"test": True}

    def test_invalidate_on_mtime_change(self, cache: ASTCache, tmp_py_file: str):
        graph = _make_graph()
        cache.put(tmp_py_file, graph)
        assert cache.get(tmp_py_file) is not None

        # Ensure mtime changes (some filesystems have 1s resolution)
        time.sleep(0.05)
        with open(tmp_py_file, "w") as f:
            f.write("x = 2\n")
        # Force a different mtime by touching with future time
        new_mtime = os.path.getmtime(tmp_py_file) + 1
        os.utime(tmp_py_file, (new_mtime, new_mtime))

        assert cache.get(tmp_py_file) is None

    def test_invalidate_method(self, cache: ASTCache, tmp_py_file: str):
        cache.put(tmp_py_file, _make_graph())
        cache.invalidate(tmp_py_file)
        assert cache.get(tmp_py_file) is None

    def test_clear(self, cache: ASTCache, tmp_py_file: str):
        cache.put(tmp_py_file, _make_graph())
        cache.clear()
        assert cache.get(tmp_py_file) is None

    def test_missing_file_returns_none(self, cache: ASTCache):
        cache._entries["/nonexistent/file.py"] = (0.0, _make_graph())
        assert cache.get("/nonexistent/file.py") is None

    def test_put_nonexistent_file_ignored(self, cache: ASTCache):
        cache.put("/nonexistent/file.py", _make_graph())
        assert len(cache._entries) == 0
