"""Tests for ImmutableDescriptor and IndexedDescriptor."""

from __future__ import annotations

import pytest

from trinity.descriptors.immutable import ImmutableDescriptor
from trinity.descriptors.indexing import IndexedDescriptor, find_by_index


# ---------------------------------------------------------------------------
# Test fixtures using descriptors
# ---------------------------------------------------------------------------

class FrozenEntity:
    name = ImmutableDescriptor(field_type=str, freeze_after_init=True)

    def __init__(self, name: str):
        self.name = name


class OptionalFreezeEntity:
    label = ImmutableDescriptor(field_type=str, freeze_after_init=False)

    def __init__(self, label: str):
        self.label = label


class IndexedEntity:
    color = IndexedDescriptor(field_type=str)

    def __init__(self, color: str):
        self.color = color


class UniqueIndexedEntity:
    code = IndexedDescriptor(field_type=str, unique=True)

    def __init__(self, code: str):
        self.code = code


# ---------------------------------------------------------------------------
# ImmutableDescriptor tests
# ---------------------------------------------------------------------------

class TestImmutableDescriptor:
    def test_freeze_after_first_set(self):
        e = FrozenEntity("Alice")
        assert e.name == "Alice"
        with pytest.raises(AttributeError, match="Cannot set immutable field 'name'"):
            e.name = "Bob"

    def test_value_preserved_after_freeze(self):
        e = FrozenEntity("Alice")
        with pytest.raises(AttributeError):
            e.name = "Bob"
        assert e.name == "Alice"

    def test_no_freeze_when_disabled(self):
        e = OptionalFreezeEntity("first")
        assert e.label == "first"
        e.label = "second"
        assert e.label == "second"
        e.label = "third"
        assert e.label == "third"

    def test_excludes(self):
        desc = ImmutableDescriptor()
        assert "tracked" in desc.excludes
        assert "observable" in desc.excludes
        assert "networked" in desc.excludes

    def test_descriptor_steps(self):
        desc = ImmutableDescriptor()
        desc.__set_name__(type("X", (), {}), "f")
        steps = desc.descriptor_steps
        assert len(steps) == 1
        assert steps[0].args["set"] == "deny_after_init"

    def test_get_metadata(self):
        desc = ImmutableDescriptor(freeze_after_init=True)
        desc.__set_name__(type("X", (), {}), "f")
        meta = desc.get_metadata()
        assert meta["freeze_after_init"] is True


# ---------------------------------------------------------------------------
# IndexedDescriptor tests
# ---------------------------------------------------------------------------

class TestIndexedDescriptor:
    def test_indexing_on_set(self):
        # Reset class-level index
        if hasattr(IndexedEntity, "_index_color"):
            delattr(IndexedEntity, "_index_color")
        IndexedEntity.color = IndexedDescriptor(field_type=str)
        IndexedEntity.color.__set_name__(IndexedEntity, "color")

        a = IndexedEntity("red")
        b = IndexedEntity("red")
        c = IndexedEntity("blue")

        result = find_by_index(IndexedEntity, "color", "red")
        assert id(a) in result
        assert id(b) in result
        assert id(c) not in result

    def test_pre_set_removes_old_index(self):
        if hasattr(IndexedEntity, "_index_color"):
            delattr(IndexedEntity, "_index_color")
        IndexedEntity.color = IndexedDescriptor(field_type=str)
        IndexedEntity.color.__set_name__(IndexedEntity, "color")

        a = IndexedEntity("red")
        assert id(a) in find_by_index(IndexedEntity, "color", "red")

        a.color = "blue"
        assert id(a) not in find_by_index(IndexedEntity, "color", "red")
        assert id(a) in find_by_index(IndexedEntity, "color", "blue")

    def test_unique_enforcement(self):
        if hasattr(UniqueIndexedEntity, "_index_code"):
            delattr(UniqueIndexedEntity, "_index_code")
        UniqueIndexedEntity.code = IndexedDescriptor(field_type=str, unique=True)
        UniqueIndexedEntity.code.__set_name__(UniqueIndexedEntity, "code")

        a = UniqueIndexedEntity("ABC")
        with pytest.raises(ValueError, match="Unique constraint violated"):
            UniqueIndexedEntity("ABC")

    def test_find_by(self):
        if hasattr(IndexedEntity, "_index_color"):
            delattr(IndexedEntity, "_index_color")
        IndexedEntity.color = IndexedDescriptor(field_type=str)
        IndexedEntity.color.__set_name__(IndexedEntity, "color")

        a = IndexedEntity("green")
        result = IndexedDescriptor.find_by(IndexedEntity, "color", "green")
        assert id(a) in result
        assert find_by_index(IndexedEntity, "color", "nonexistent") == set()

    def test_descriptor_steps(self):
        desc = IndexedDescriptor()
        desc.__set_name__(type("X", (), {}), "f")
        steps = desc.descriptor_steps
        assert len(steps) == 2

    def test_get_metadata(self):
        desc = IndexedDescriptor(unique=True)
        desc.__set_name__(type("X", (), {}), "f")
        meta = desc.get_metadata()
        assert meta["unique"] is True
