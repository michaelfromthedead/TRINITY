"""
Tests for tracking descriptors: TrackedDescriptor, VersionedDescriptor, DiffDescriptor.

Verifies:
- Dirty tracking with set-based and bitmask approaches
- Version incrementing on set
- Diff detection with shallow, deep, and custom strategies
- Edge cases: None, empty values, wrong types
- Metadata correctness with actual value verification
"""
import pytest
import sys
sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from trinity.descriptors.tracking import (
    TrackedDescriptor,
    VersionedDescriptor,
    DiffDescriptor,
    is_dirty,
    get_dirty_fields,
    clear_dirty,
    clear_dirty_field,
)
from trinity.descriptors.storage import StorageDescriptor


class TestTrackedDescriptor:
    """Test TrackedDescriptor marks fields as dirty when values change."""

    def test_set_based_tracking(self):
        """Setting a value should add field to _dirty_fields set."""
        class Foo:
            value = TrackedDescriptor(field_type=int, use_bitmask=False)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = 10
        assert 'value' in f._dirty_fields
        assert is_dirty(f, 'value')

    def test_bitmask_tracking(self):
        """Setting a value should set bit in _dirty_mask."""
        class Foo:
            value = TrackedDescriptor(field_type=int, use_bitmask=True, field_offset=3)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = 10
        # Bit 3 should be set: 1 << 3 = 8
        assert f._dirty_mask & (1 << 3) != 0

    def test_no_dirty_if_value_unchanged(self):
        """Setting same value should not mark field dirty."""
        class Foo:
            value = TrackedDescriptor(field_type=int, use_bitmask=False)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = 10
        clear_dirty(f)
        f.value = 10  # Same value
        assert not is_dirty(f, 'value')

    def test_get_dirty_fields(self):
        """get_dirty_fields should return all dirty field names."""
        class Foo:
            a = TrackedDescriptor(field_type=int, use_bitmask=False)
            b = TrackedDescriptor(field_type=int, use_bitmask=False)
        Foo.a.__set_name__(Foo, 'a')
        Foo.b.__set_name__(Foo, 'b')
        f = Foo()
        f.a = 1
        f.b = 2
        dirty = get_dirty_fields(f)
        assert 'a' in dirty
        assert 'b' in dirty

    def test_clear_dirty(self):
        """clear_dirty should reset all dirty flags."""
        class Foo:
            value = TrackedDescriptor(field_type=int, use_bitmask=False)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = 10
        assert is_dirty(f, 'value')
        clear_dirty(f)
        assert not is_dirty(f, 'value')

    def test_clear_dirty_field(self):
        """clear_dirty_field should clear specific field."""
        class Foo:
            a = TrackedDescriptor(field_type=int, use_bitmask=False)
            b = TrackedDescriptor(field_type=int, use_bitmask=False)
        Foo.a.__set_name__(Foo, 'a')
        Foo.b.__set_name__(Foo, 'b')
        f = Foo()
        f.a = 1
        f.b = 2
        clear_dirty_field(f, 'a')
        assert not is_dirty(f, 'a')
        assert is_dirty(f, 'b')

    def test_metadata_values(self):
        """Metadata should contain actual field_offset and use_bitmask values."""
        class Foo:
            value = TrackedDescriptor(field_type=int, use_bitmask=True, field_offset=5)
        Foo.value.__set_name__(Foo, 'value')
        meta = Foo.value.get_metadata()
        assert meta["descriptor_id"] == "tracked"
        assert meta["field_offset"] == 5
        assert meta["use_bitmask"] is True

    def test_with_inner_descriptor(self):
        """TrackedDescriptor should work with inner storage descriptor."""
        class Foo:
            value = TrackedDescriptor(
                field_type=int,
                inner=StorageDescriptor(field_type=int),
                use_bitmask=False
            )
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = 42
        assert f.value == 42
        assert is_dirty(f, 'value')

    def test_none_value_tracking(self):
        """Setting None should be tracked if different from previous."""
        class Foo:
            value = TrackedDescriptor(field_type=object, use_bitmask=False)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = "something"
        clear_dirty(f)
        f.value = None
        assert is_dirty(f, 'value')

    def test_empty_value_tracking(self):
        """Empty values (empty list, empty string) should be tracked."""
        class Foo:
            data = TrackedDescriptor(field_type=list, use_bitmask=False)
        Foo.data.__set_name__(Foo, 'data')
        f = Foo()
        f.data = [1, 2, 3]
        clear_dirty(f)
        f.data = []  # Empty list
        assert is_dirty(f, 'data')


class TestVersionedDescriptor:
    """Test VersionedDescriptor tracks a version number that increments on each set."""

    def test_version_increments(self):
        """Setting value 3 times should result in version 3."""
        class Foo:
            value = VersionedDescriptor(field_type=int)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = 1
        f.value = 2
        f.value = 3
        version = Foo.value.get_version(f)
        assert version == 3

    def test_initial_version_zero(self):
        """Before any set, version should be 0."""
        class Foo:
            value = VersionedDescriptor(field_type=int)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        version = Foo.value.get_version(f)
        assert version == 0

    def test_metadata_values(self):
        """Metadata should contain actual descriptor_id and version_attr."""
        class Foo:
            value = VersionedDescriptor(field_type=int)
        Foo.value.__set_name__(Foo, 'value')
        meta = Foo.value.get_metadata()
        assert meta["descriptor_id"] == "versioned"
        assert meta["version_attr"] == "_version_value"

    def test_independent_instances(self):
        """Each instance should track its own version independently."""
        class Foo:
            value = VersionedDescriptor(field_type=int)
        Foo.value.__set_name__(Foo, 'value')
        a = Foo()
        b = Foo()
        a.value = 10
        a.value = 20
        b.value = 99
        assert Foo.value.get_version(a) == 2
        assert Foo.value.get_version(b) == 1

    def test_same_value_still_increments(self):
        """Setting the same value should still increment the version."""
        class Foo:
            value = VersionedDescriptor(field_type=int)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = 5
        f.value = 5
        assert Foo.value.get_version(f) == 2

    def test_none_value_increments(self):
        """Setting None should increment version."""
        class Foo:
            value = VersionedDescriptor(field_type=object)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = None
        f.value = None
        assert Foo.value.get_version(f) == 2

    def test_with_inner_descriptor(self):
        """VersionedDescriptor should work with inner storage."""
        class Foo:
            value = VersionedDescriptor(
                field_type=int,
                inner=StorageDescriptor(field_type=int)
            )
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = 10
        f.value = 20
        assert f.value == 20
        assert Foo.value.get_version(f) == 2


class TestDiffDescriptor:
    """Test DiffDescriptor tracks previous values and detects changes."""

    def test_stores_previous(self):
        """After setting twice, previous value should be the first value."""
        class Foo:
            value = DiffDescriptor(field_type=int)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = 10
        f.value = 20
        assert Foo.value.get_previous(f) == 10

    def test_has_changed_shallow(self):
        """Shallow diff should detect value changes by equality."""
        class Foo:
            value = DiffDescriptor(field_type=int, strategy="shallow")
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = 10
        f.value = 20
        assert Foo.value.has_changed(f) is True

    def test_has_changed_shallow_same_value(self):
        """Shallow diff with same value should return False."""
        class Foo:
            value = DiffDescriptor(field_type=int, strategy="shallow")
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = 10
        f.value = 10
        assert Foo.value.has_changed(f) is False

    def test_has_changed_deep(self):
        """Deep diff should detect nested changes in mutable objects."""
        class Foo:
            data = DiffDescriptor(field_type=dict, strategy="deep")
        Foo.data.__set_name__(Foo, 'data')
        f = Foo()
        f.data = {"a": 1}
        f.data = {"a": 1}  # Same content
        assert Foo.data.has_changed(f) is False
        f.data = {"a": 2}
        assert Foo.data.has_changed(f) is True

    def test_custom_differ(self):
        """Custom differ function should be used for comparison."""
        def length_differ(old, new):
            """Consider changed only if lengths differ."""
            if old is None:
                return new is not None
            if new is None:
                return True
            return len(old) != len(new)

        class Foo:
            items = DiffDescriptor(field_type=list, strategy="custom", custom_differ=length_differ)
        Foo.items.__set_name__(Foo, 'items')
        f = Foo()
        f.items = [1, 2, 3]
        f.items = [4, 5, 6]  # Same length
        assert Foo.items.has_changed(f) is False
        f.items = [1, 2]  # Different length
        assert Foo.items.has_changed(f) is True

    def test_invalid_strategy_raises(self):
        """An invalid strategy should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid diff strategy"):
            DiffDescriptor(field_type=int, strategy="nonexistent")

    def test_custom_without_differ_raises(self):
        """Strategy 'custom' without a differ function should raise ValueError."""
        with pytest.raises(ValueError, match="custom_differ is required"):
            DiffDescriptor(field_type=int, strategy="custom")

    def test_no_previous_before_second_set(self):
        """Before a second set, previous should be None."""
        class Foo:
            value = DiffDescriptor(field_type=int)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = 42
        assert Foo.value.get_previous(f) is None

    def test_metadata_values(self):
        """Metadata should contain actual strategy and custom_differ info."""
        class Foo:
            value = DiffDescriptor(field_type=int, strategy="shallow")
        Foo.value.__set_name__(Foo, 'value')
        meta = Foo.value.get_metadata()
        assert meta["descriptor_id"] == "diff"
        assert meta["strategy"] == "shallow"
        assert meta["has_custom_differ"] is False

    def test_metadata_custom_differ(self):
        """Metadata should indicate when custom differ is present."""
        def custom(old, new):
            return old != new
        class Foo:
            value = DiffDescriptor(field_type=int, strategy="custom", custom_differ=custom)
        Foo.value.__set_name__(Foo, 'value')
        meta = Foo.value.get_metadata()
        assert meta["has_custom_differ"] is True

    def test_none_to_value_change(self):
        """Changing from None to a value should be detected."""
        class Foo:
            value = DiffDescriptor(field_type=object, strategy="deep")
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = None
        f.value = "something"
        assert Foo.value.has_changed(f) is True

    def test_value_to_none_change(self):
        """Changing from a value to None should be detected."""
        class Foo:
            value = DiffDescriptor(field_type=object, strategy="deep")
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = "something"
        f.value = None
        assert Foo.value.has_changed(f) is True

    def test_structural_strategy_fallback(self):
        """Structural strategy should fall back to shallow comparison."""
        class Foo:
            value = DiffDescriptor(field_type=int, strategy="structural")
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = 10
        f.value = 20
        assert Foo.value.has_changed(f) is True

    def test_empty_list_change(self):
        """Changing to empty list should be detected."""
        class Foo:
            data = DiffDescriptor(field_type=list, strategy="shallow")
        Foo.data.__set_name__(Foo, 'data')
        f = Foo()
        f.data = [1, 2, 3]
        f.data = []
        assert Foo.data.has_changed(f) is True

    def test_empty_string_change(self):
        """Changing to empty string should be detected."""
        class Foo:
            text = DiffDescriptor(field_type=str, strategy="shallow")
        Foo.text.__set_name__(Foo, 'text')
        f = Foo()
        f.text = "hello"
        f.text = ""
        assert Foo.text.has_changed(f) is True
