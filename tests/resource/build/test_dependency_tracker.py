"""Tests for the build dependency tracker."""
from engine.resource.build.dependency_tracker import BuildDependencyTracker, FileRecord


class TestFileRecord:
    def test_slots(self) -> None:
        r = FileRecord(path="a.py", mtime=1.0, content_hash="abc")
        assert not hasattr(r, "__dict__")

    def test_defaults(self) -> None:
        r = FileRecord(path="a.py", mtime=1.0, content_hash="abc")
        assert r.dependencies == set()


class TestBuildDependencyTracker:
    def test_record_and_retrieve(self) -> None:
        t = BuildDependencyTracker()
        rec = t.record_file("a.py", 1.0, b"content")
        assert rec.path == "a.py"
        assert t.get_record("a.py") is rec

    def test_not_dirty_when_unchanged(self) -> None:
        t = BuildDependencyTracker()
        t.record_file("a.py", 1.0, b"hello")
        assert t.is_dirty("a.py", 1.0, b"hello") is False

    def test_dirty_when_content_changes(self) -> None:
        t = BuildDependencyTracker()
        t.record_file("a.py", 1.0, b"old")
        assert t.is_dirty("a.py", 2.0, b"new") is True

    def test_dirty_when_unknown_file(self) -> None:
        t = BuildDependencyTracker()
        assert t.is_dirty("unknown.py", 1.0, b"data") is True

    def test_get_dirty_files(self) -> None:
        t = BuildDependencyTracker()
        t.record_file("a.py", 1.0, b"a")
        t.record_file("b.py", 1.0, b"b")
        dirty = t.get_dirty_files({
            "a.py": (1.0, b"a"),    # clean
            "b.py": (2.0, b"b2"),   # dirty
            "c.py": (1.0, b"c"),    # new
        })
        assert "b.py" in dirty
        assert "c.py" in dirty
        assert "a.py" not in dirty

    def test_build_order_topological(self) -> None:
        t = BuildDependencyTracker()
        t.record_file("base.py", 1.0, b"b", dependencies=set())
        t.record_file("mid.py", 1.0, b"m", dependencies={"base.py"})
        t.record_file("top.py", 1.0, b"t", dependencies={"mid.py"})
        order = t.get_build_order(["base.py", "mid.py", "top.py"])
        assert order.index("base.py") < order.index("mid.py")
        assert order.index("mid.py") < order.index("top.py")

    def test_clear(self) -> None:
        t = BuildDependencyTracker()
        t.record_file("a.py", 1.0, b"a")
        t.clear()
        assert t.get_record("a.py") is None

    def test_mtime_same_content_same_not_dirty(self) -> None:
        t = BuildDependencyTracker()
        t.record_file("x.py", 5.0, b"data")
        # same mtime => skip hash check => not dirty
        assert t.is_dirty("x.py", 5.0, b"different_but_same_mtime") is False
