"""
Comprehensive unit tests for the Delta Sync module.
Tests computing and applying minimal change patches between dict states.
"""
import pytest
import sys
sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from foundation.delta_sync import DeltaPatch, DeltaSync


class TestDeltaPatch:
    """Test DeltaPatch dataclass."""

    def test_empty_patch(self):
        """New patch should be empty."""
        patch = DeltaPatch()
        assert patch.is_empty()
        assert len(patch) == 0

    def test_patch_with_changes(self):
        """Patch with changes should not be empty."""
        patch = DeltaPatch(changes=[("a", 1)])
        assert not patch.is_empty()
        assert len(patch) == 1

    def test_patch_with_removes(self):
        """Patch with removes should not be empty."""
        patch = DeltaPatch(removes=["a"])
        assert not patch.is_empty()
        assert len(patch) == 1

    def test_patch_length(self):
        """Patch length should sum changes and removes."""
        patch = DeltaPatch(
            changes=[("a", 1), ("b", 2)],
            removes=["c", "d", "e"]
        )
        assert len(patch) == 5


class TestComputeDeltaNoChange:
    """Test compute_delta with identical dicts."""

    def test_compute_delta_no_change(self):
        """Identical dicts should produce empty delta."""
        sync = DeltaSync()
        old = {"a": 1, "b": 2}
        new = {"a": 1, "b": 2}
        delta = sync.compute_delta(old, new)
        assert delta.is_empty()
        assert delta.changes == []
        assert delta.removes == []

    def test_compute_delta_empty_dicts(self):
        """Two empty dicts should produce empty delta."""
        sync = DeltaSync()
        delta = sync.compute_delta({}, {})
        assert delta.is_empty()

    def test_compute_delta_nested_no_change(self):
        """Identical nested dicts should produce empty delta."""
        sync = DeltaSync()
        old = {"a": {"b": {"c": 1}}}
        new = {"a": {"b": {"c": 1}}}
        delta = sync.compute_delta(old, new)
        assert delta.is_empty()


class TestComputeDeltaValueChange:
    """Test compute_delta with value changes."""

    def test_compute_delta_value_change(self):
        """Detect simple value change."""
        sync = DeltaSync()
        old = {"health": 100}
        new = {"health": 80}
        delta = sync.compute_delta(old, new)
        assert delta.changes == [("health", 80)]
        assert delta.removes == []

    def test_compute_delta_multiple_value_changes(self):
        """Detect multiple value changes."""
        sync = DeltaSync()
        old = {"a": 1, "b": 2, "c": 3}
        new = {"a": 10, "b": 2, "c": 30}
        delta = sync.compute_delta(old, new)
        # Changes should include a and c (b is unchanged)
        changes_dict = dict(delta.changes)
        assert changes_dict["a"] == 10
        assert changes_dict["c"] == 30
        assert "b" not in changes_dict
        assert len(delta.changes) == 2


class TestComputeDeltaNestedChange:
    """Test compute_delta with nested value changes."""

    def test_compute_delta_nested_change(self):
        """Detect nested value change."""
        sync = DeltaSync()
        old = {"a": {"b": {"c": 1}}}
        new = {"a": {"b": {"c": 2}}}
        delta = sync.compute_delta(old, new)
        assert delta.changes == [("a.b.c", 2)]
        assert delta.removes == []

    def test_compute_delta_nested_multiple_changes(self):
        """Detect multiple nested changes."""
        sync = DeltaSync()
        old = {"pos": {"x": 0, "y": 0}}
        new = {"pos": {"x": 5, "y": 10}}
        delta = sync.compute_delta(old, new)
        changes_dict = dict(delta.changes)
        assert changes_dict["pos.x"] == 5
        assert changes_dict["pos.y"] == 10

    def test_compute_delta_deep_nested(self):
        """Detect deeply nested change."""
        sync = DeltaSync()
        old = {"l1": {"l2": {"l3": {"l4": 0}}}}
        new = {"l1": {"l2": {"l3": {"l4": 99}}}}
        delta = sync.compute_delta(old, new)
        assert delta.changes == [("l1.l2.l3.l4", 99)]


class TestComputeDeltaAddition:
    """Test compute_delta with field additions."""

    def test_compute_delta_addition(self):
        """Detect new field added."""
        sync = DeltaSync()
        old = {"a": 1}
        new = {"a": 1, "b": 2}
        delta = sync.compute_delta(old, new)
        assert delta.changes == [("b", 2)]
        assert delta.removes == []

    def test_compute_delta_nested_addition(self):
        """Detect new nested field added."""
        sync = DeltaSync()
        old = {"data": {"x": 1}}
        new = {"data": {"x": 1, "y": 2}}
        delta = sync.compute_delta(old, new)
        assert delta.changes == [("data.y", 2)]

    def test_compute_delta_new_nested_dict(self):
        """Detect entirely new nested dict added."""
        sync = DeltaSync()
        old = {"a": 1}
        new = {"a": 1, "b": {"c": 2}}
        delta = sync.compute_delta(old, new)
        assert delta.changes == [("b", {"c": 2})]


class TestComputeDeltaRemoval:
    """Test compute_delta with field removals."""

    def test_compute_delta_removal(self):
        """Detect field removed."""
        sync = DeltaSync()
        old = {"a": 1, "b": 2}
        new = {"a": 1}
        delta = sync.compute_delta(old, new)
        assert delta.changes == []
        assert delta.removes == ["b"]

    def test_compute_delta_nested_removal(self):
        """Detect nested field removed."""
        sync = DeltaSync()
        old = {"data": {"x": 1, "y": 2}}
        new = {"data": {"x": 1}}
        delta = sync.compute_delta(old, new)
        assert delta.removes == ["data.y"]

    def test_compute_delta_multiple_removals(self):
        """Detect multiple fields removed."""
        sync = DeltaSync()
        old = {"a": 1, "b": 2, "c": 3}
        new = {"a": 1}
        delta = sync.compute_delta(old, new)
        assert set(delta.removes) == {"b", "c"}


class TestComputeDeltaMultiple:
    """Test compute_delta with multiple types of changes."""

    def test_compute_delta_multiple(self):
        """Multiple changes at once."""
        sync = DeltaSync()
        old = {"a": 1, "b": {"c": 2}, "d": 4}
        new = {"a": 10, "b": {"c": 3}, "e": 5}  # a changed, b.c changed, d removed, e added
        delta = sync.compute_delta(old, new)

        changes_dict = dict(delta.changes)
        assert changes_dict["a"] == 10
        assert changes_dict["b.c"] == 3
        assert changes_dict["e"] == 5
        assert delta.removes == ["d"]

    def test_compute_delta_complex_scenario(self):
        """Complex scenario with deep nesting and multiple changes."""
        sync = DeltaSync()
        old = {
            "player": {
                "health": 100,
                "position": {"x": 0, "y": 0, "z": 0},
                "inventory": {"gold": 50}
            },
            "enemies": {"count": 5}
        }
        new = {
            "player": {
                "health": 80,
                "position": {"x": 10, "y": 0},  # z removed
                "inventory": {"gold": 50, "items": 3}  # items added
            }
            # enemies removed entirely
        }
        delta = sync.compute_delta(old, new)

        changes_dict = dict(delta.changes)
        assert changes_dict["player.health"] == 80
        assert changes_dict["player.position.x"] == 10
        assert changes_dict["player.inventory.items"] == 3

        assert "player.position.z" in delta.removes
        assert "enemies" in delta.removes


class TestApplyDeltaChange:
    """Test apply_delta with value changes."""

    def test_apply_delta_change(self):
        """Apply value change."""
        sync = DeltaSync()
        target = {"a": 1, "b": 2}
        delta = DeltaPatch(changes=[("a", 10)])
        result = sync.apply_delta(target, delta)
        assert result["a"] == 10
        assert result["b"] == 2
        assert result is target  # Modified in place


class TestApplyDeltaNested:
    """Test apply_delta with nested changes."""

    def test_apply_delta_nested(self):
        """Apply nested change."""
        sync = DeltaSync()
        target = {"a": {"b": {"c": 1}}}
        delta = DeltaPatch(changes=[("a.b.c", 99)])
        result = sync.apply_delta(target, delta)
        assert result["a"]["b"]["c"] == 99

    def test_apply_delta_multiple_nested(self):
        """Apply multiple nested changes."""
        sync = DeltaSync()
        target = {"pos": {"x": 0, "y": 0}}
        delta = DeltaPatch(changes=[("pos.x", 5), ("pos.y", 10)])
        result = sync.apply_delta(target, delta)
        assert result["pos"]["x"] == 5
        assert result["pos"]["y"] == 10


class TestApplyDeltaAdd:
    """Test apply_delta with additions."""

    def test_apply_delta_add(self):
        """Apply addition creates new field."""
        sync = DeltaSync()
        target = {"a": 1}
        delta = DeltaPatch(changes=[("b", 2)])
        result = sync.apply_delta(target, delta)
        assert result["b"] == 2

    def test_apply_delta_add_creates_intermediate(self):
        """Apply addition creates intermediate dicts."""
        sync = DeltaSync()
        target = {}
        delta = DeltaPatch(changes=[("a.b.c", 99)])
        result = sync.apply_delta(target, delta)
        assert result["a"]["b"]["c"] == 99

    def test_apply_delta_add_nested(self):
        """Apply addition to existing nested structure."""
        sync = DeltaSync()
        target = {"data": {"x": 1}}
        delta = DeltaPatch(changes=[("data.y", 2)])
        result = sync.apply_delta(target, delta)
        assert result["data"]["y"] == 2
        assert result["data"]["x"] == 1  # Preserved


class TestApplyDeltaRemove:
    """Test apply_delta with removals."""

    def test_apply_delta_remove(self):
        """Apply removal deletes field."""
        sync = DeltaSync()
        target = {"a": 1, "b": 2}
        delta = DeltaPatch(removes=["b"])
        result = sync.apply_delta(target, delta)
        assert "b" not in result
        assert result["a"] == 1

    def test_apply_delta_remove_nested(self):
        """Apply removal deletes nested field."""
        sync = DeltaSync()
        target = {"data": {"x": 1, "y": 2}}
        delta = DeltaPatch(removes=["data.y"])
        result = sync.apply_delta(target, delta)
        assert "y" not in result["data"]
        assert result["data"]["x"] == 1

    def test_apply_delta_remove_nonexistent(self):
        """Removing nonexistent path should not raise."""
        sync = DeltaSync()
        target = {"a": 1}
        delta = DeltaPatch(removes=["nonexistent.path"])
        result = sync.apply_delta(target, delta)
        assert result == {"a": 1}


class TestApplyDeltaEmpty:
    """Test apply_delta with empty delta."""

    def test_apply_delta_empty(self):
        """Empty delta should not change target."""
        sync = DeltaSync()
        target = {"a": 1, "b": {"c": 2}}
        original = {"a": 1, "b": {"c": 2}}
        delta = DeltaPatch()
        result = sync.apply_delta(target, delta)
        assert result == original


class TestRoundtrip:
    """Test compute then apply produces same result."""

    def test_roundtrip(self):
        """Compute then apply produces same result as new."""
        sync = DeltaSync()
        old = {"health": 100, "position": {"x": 0, "y": 0}}
        new = {"health": 80, "position": {"x": 5, "y": 0}, "status": "alive"}

        delta = sync.compute_delta(old, new)

        # Apply to a copy of old
        target = {"health": 100, "position": {"x": 0, "y": 0}}
        result = sync.apply_delta(target, delta)

        assert result == new

    def test_roundtrip_with_removals(self):
        """Roundtrip with removals produces same result."""
        sync = DeltaSync()
        old = {"a": 1, "b": 2, "c": {"d": 3, "e": 4}}
        new = {"a": 10, "c": {"d": 30}}  # b removed, c.e removed

        delta = sync.compute_delta(old, new)
        target = {"a": 1, "b": 2, "c": {"d": 3, "e": 4}}
        result = sync.apply_delta(target, delta)

        assert result == new

    def test_roundtrip_complex(self):
        """Complex roundtrip with all types of changes."""
        sync = DeltaSync()
        old = {
            "player": {
                "health": 100,
                "mana": 50,
                "position": {"x": 0, "y": 0}
            },
            "world": {
                "time": 0,
                "weather": "sunny"
            }
        }
        new = {
            "player": {
                "health": 75,
                "position": {"x": 10, "y": 5},
                "buffs": ["speed"]
            },
            "world": {
                "time": 100
            }
        }

        delta = sync.compute_delta(old, new)

        # Apply to fresh copy
        import copy
        target = copy.deepcopy(old)
        result = sync.apply_delta(target, delta)

        assert result == new

    def test_roundtrip_empty_to_populated(self):
        """Roundtrip from empty dict to populated."""
        sync = DeltaSync()
        old = {}
        new = {"a": 1, "b": {"c": 2}}

        delta = sync.compute_delta(old, new)
        target = {}
        result = sync.apply_delta(target, delta)

        assert result == new

    def test_roundtrip_populated_to_empty(self):
        """Roundtrip from populated dict to empty."""
        sync = DeltaSync()
        old = {"a": 1, "b": {"c": 2}}
        new = {}

        delta = sync.compute_delta(old, new)
        target = {"a": 1, "b": {"c": 2}}
        result = sync.apply_delta(target, delta)

        assert result == new


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_list_value_change(self):
        """Lists are compared by equality, not recursively."""
        sync = DeltaSync()
        old = {"items": [1, 2, 3]}
        new = {"items": [1, 2, 4]}
        delta = sync.compute_delta(old, new)
        # List should be replaced entirely
        assert delta.changes == [("items", [1, 2, 4])]

    def test_none_value(self):
        """None values should be handled correctly."""
        sync = DeltaSync()
        old = {"a": 1}
        new = {"a": None}
        delta = sync.compute_delta(old, new)
        assert delta.changes == [("a", None)]

    def test_none_to_value(self):
        """Changing from None to value should work."""
        sync = DeltaSync()
        old = {"a": None}
        new = {"a": 1}
        delta = sync.compute_delta(old, new)
        assert delta.changes == [("a", 1)]

    def test_type_change_dict_to_value(self):
        """Changing from dict to primitive should replace entirely."""
        sync = DeltaSync()
        old = {"a": {"b": 1}}
        new = {"a": 42}
        delta = sync.compute_delta(old, new)
        assert delta.changes == [("a", 42)]

    def test_type_change_value_to_dict(self):
        """Changing from primitive to dict should replace entirely."""
        sync = DeltaSync()
        old = {"a": 42}
        new = {"a": {"b": 1}}
        delta = sync.compute_delta(old, new)
        assert delta.changes == [("a", {"b": 1})]

    def test_special_characters_in_keys(self):
        """Keys with underscores and numbers should work."""
        sync = DeltaSync()
        old = {"player_1": {"score_total": 100}}
        new = {"player_1": {"score_total": 150}}
        delta = sync.compute_delta(old, new)
        assert delta.changes == [("player_1.score_total", 150)]

    def test_apply_removes_before_changes(self):
        """Removes should be applied before changes."""
        sync = DeltaSync()
        target = {"a": {"old": 1}}
        # Remove old, then add new in same nested dict
        delta = DeltaPatch(
            removes=["a.old"],
            changes=[("a.new", 2)]
        )
        result = sync.apply_delta(target, delta)
        assert "old" not in result["a"]
        assert result["a"]["new"] == 2
