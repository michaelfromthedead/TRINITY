"""
Tests for MigrationRegistry.

Verifies:
- Direct migration registration and execution
- Multi-step migration path finding (BFS)
- Error handling for missing paths
"""
import pytest
import sys
sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from foundation.migrations import (
    MigrationRegistry,
    register_migration,
    migrate,
    has_migration_path,
    get_migration_path,
    clear_migrations,
)


@pytest.fixture
def registry():
    """Fresh migration registry for each test."""
    return MigrationRegistry()


@pytest.fixture(autouse=True)
def clear_global_registry():
    """Clear global registry between tests."""
    clear_migrations()
    yield
    clear_migrations()


class TestMigrationRegistryBasics:
    """Test basic registration and lookup."""

    def test_register_migration(self, registry):
        """Registering a migration stores it."""
        def migrate_fn(data):
            return data

        registry.register("hash_a", "hash_b", migrate_fn)
        assert registry.has_migration("hash_a", "hash_b")

    def test_has_migration_false_for_unregistered(self, registry):
        """has_migration returns False for unregistered migrations."""
        assert not registry.has_migration("unknown", "other")

    def test_has_path_for_direct_migration(self, registry):
        """has_path returns True for direct migrations."""
        registry.register("a", "b", lambda d: d)
        assert registry.has_path("a", "b")

    def test_has_path_for_same_hash(self, registry):
        """has_path returns True when from == to (no migration needed)."""
        assert registry.has_path("same", "same")

    def test_has_path_false_for_no_path(self, registry):
        """has_path returns False when no path exists."""
        registry.register("a", "b", lambda d: d)
        assert not registry.has_path("b", "a")  # No reverse registered
        assert not registry.has_path("c", "d")  # Completely unknown


class TestMigrationExecution:
    """Test migration execution."""

    def test_direct_migration(self, registry):
        """Direct migration applies the function."""
        def add_field(data):
            data["new_field"] = "added"
            return data

        registry.register("v1", "v2", add_field)
        result = registry.migrate({"existing": 1}, "v1", "v2")

        assert result["existing"] == 1
        assert result["new_field"] == "added"

    def test_same_hash_returns_unchanged(self, registry):
        """Migrating to same hash returns data unchanged."""
        data = {"value": 42}
        result = registry.migrate(data, "hash", "hash")
        assert result is data

    def test_missing_path_raises_error(self, registry):
        """Migrating with no path raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            registry.migrate({}, "unknown_a", "unknown_b")
        assert "No migration path" in str(exc_info.value)

    def test_migration_transforms_data(self, registry):
        """Migration function receives and transforms data."""
        def rename_field(data):
            data["name"] = data.pop("old_name")
            return data

        registry.register("old", "new", rename_field)
        result = registry.migrate({"old_name": "test"}, "old", "new")

        assert "name" in result
        assert result["name"] == "test"
        assert "old_name" not in result


class TestMultiStepMigration:
    """Test multi-step migration path finding."""

    def test_two_step_migration(self, registry):
        """Two-step migration applies both functions in order."""
        registry.register("v1", "v2", lambda d: {**d, "step1": True})
        registry.register("v2", "v3", lambda d: {**d, "step2": True})

        result = registry.migrate({"original": 1}, "v1", "v3")

        assert result["original"] == 1
        assert result["step1"] is True
        assert result["step2"] is True

    def test_three_step_migration(self, registry):
        """Three-step migration works correctly."""
        registry.register("a", "b", lambda d: {**d, "a_to_b": True})
        registry.register("b", "c", lambda d: {**d, "b_to_c": True})
        registry.register("c", "d", lambda d: {**d, "c_to_d": True})

        result = registry.migrate({}, "a", "d")

        assert result["a_to_b"] is True
        assert result["b_to_c"] is True
        assert result["c_to_d"] is True

    def test_finds_shortest_path(self, registry):
        """BFS finds the shortest migration path."""
        # Direct path: v1 -> v3 (1 step)
        registry.register("v1", "v3", lambda d: {**d, "direct": True})
        # Indirect path: v1 -> v2 -> v3 (2 steps)
        registry.register("v1", "v2", lambda d: {**d, "step1": True})
        registry.register("v2", "v3", lambda d: {**d, "step2": True})

        result = registry.migrate({}, "v1", "v3")

        # Should use direct path (1 step)
        assert result.get("direct") is True
        assert "step1" not in result
        assert "step2" not in result

    def test_path_through_branch(self, registry):
        """Path finding works with branching graphs."""
        # Branch 1: a -> b -> d (dead end for c)
        registry.register("a", "b", lambda d: d)
        registry.register("b", "d", lambda d: d)
        # Branch 2: a -> c (target)
        registry.register("a", "c", lambda d: {**d, "correct": True})

        result = registry.migrate({}, "a", "c")
        assert result["correct"] is True


class TestGetPath:
    """Test path retrieval."""

    def test_get_path_returns_steps(self, registry):
        """get_path returns list of (from, to) tuples."""
        registry.register("a", "b", lambda d: d)
        registry.register("b", "c", lambda d: d)

        path = registry.get_path("a", "c")

        assert path == [("a", "b"), ("b", "c")]

    def test_get_path_for_direct(self, registry):
        """get_path returns single step for direct migration."""
        registry.register("x", "y", lambda d: d)

        path = registry.get_path("x", "y")
        assert path == [("x", "y")]

    def test_get_path_for_same_hash(self, registry):
        """get_path returns empty list for same hash."""
        path = registry.get_path("same", "same")
        assert path == []

    def test_get_path_returns_none_for_no_path(self, registry):
        """get_path returns None when no path exists."""
        path = registry.get_path("unknown", "other")
        assert path is None


class TestClear:
    """Test clearing the registry."""

    def test_clear_removes_all_migrations(self, registry):
        """clear() removes all registered migrations."""
        registry.register("a", "b", lambda d: d)
        registry.register("b", "c", lambda d: d)

        registry.clear()

        assert not registry.has_migration("a", "b")
        assert not registry.has_path("a", "c")


class TestGlobalFunctions:
    """Test module-level convenience functions."""

    def test_register_migration_function(self):
        """register_migration uses global registry."""
        register_migration("g1", "g2", lambda d: {**d, "global": True})

        assert has_migration_path("g1", "g2")

    def test_migrate_function(self):
        """migrate uses global registry."""
        register_migration("ga", "gb", lambda d: {**d, "migrated": True})

        result = migrate({"original": 1}, "ga", "gb")
        assert result["migrated"] is True

    def test_has_migration_path_function(self):
        """has_migration_path uses global registry."""
        register_migration("p1", "p2", lambda d: d)

        assert has_migration_path("p1", "p2")
        assert not has_migration_path("p2", "p1")

    def test_get_migration_path_function(self):
        """get_migration_path uses global registry."""
        register_migration("x", "y", lambda d: d)
        register_migration("y", "z", lambda d: d)

        path = get_migration_path("x", "z")
        assert path == [("x", "y"), ("y", "z")]


class TestEdgeCases:
    """Test edge cases and complex scenarios."""

    def test_cyclic_graph_does_not_loop(self, registry):
        """BFS handles cycles without infinite loops."""
        registry.register("a", "b", lambda d: d)
        registry.register("b", "c", lambda d: d)
        registry.register("c", "a", lambda d: d)  # Cycle back

        # Should find path without looping
        path = registry.get_path("a", "c")
        assert path == [("a", "b"), ("b", "c")]

    def test_multiple_paths_finds_one(self, registry):
        """Multiple valid paths still returns a path."""
        registry.register("start", "mid1", lambda d: d)
        registry.register("start", "mid2", lambda d: d)
        registry.register("mid1", "end", lambda d: d)
        registry.register("mid2", "end", lambda d: d)

        path = registry.get_path("start", "end")
        assert path is not None
        assert len(path) == 2

    def test_overwrite_migration(self, registry):
        """Registering same pair twice overwrites."""
        registry.register("a", "b", lambda d: {**d, "first": True})
        registry.register("a", "b", lambda d: {**d, "second": True})

        result = registry.migrate({}, "a", "b")
        assert result.get("second") is True
        assert result.get("first") is None

    def test_migration_with_nested_data(self, registry):
        """Migration handles nested data structures."""
        def flatten(data):
            result = dict(data)
            nested = result.pop("nested", {})
            result.update({f"flat_{k}": v for k, v in nested.items()})
            return result

        registry.register("nested", "flat", flatten)

        result = registry.migrate(
            {"top": 1, "nested": {"a": 2, "b": 3}},
            "nested",
            "flat"
        )

        assert result["top"] == 1
        assert result["flat_a"] == 2
        assert result["flat_b"] == 3
        assert "nested" not in result
