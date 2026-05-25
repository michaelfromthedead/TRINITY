"""
Tests for schema hash detection and comparison.
"""
import pytest
from dataclasses import dataclass
from typing import Optional

from engine.tooling.hotreload.schema_hash import (
    SchemaHasher,
    SchemaComparison,
    SchemaChange,
    SchemaChangeType,
)


class TestSchemaChangeType:
    """Tests for SchemaChangeType enum."""

    def test_non_breaking_types(self):
        """Test non-breaking change types exist."""
        assert hasattr(SchemaChangeType, "FIELD_ADDED_WITH_DEFAULT")
        assert hasattr(SchemaChangeType, "METADATA_CHANGED")
        assert hasattr(SchemaChangeType, "METHOD_ADDED")
        assert hasattr(SchemaChangeType, "METHOD_REMOVED")
        assert hasattr(SchemaChangeType, "DEFAULT_VALUE_CHANGED")

    def test_breaking_types(self):
        """Test breaking change types exist."""
        assert hasattr(SchemaChangeType, "FIELD_REMOVED")
        assert hasattr(SchemaChangeType, "FIELD_ADDED_WITHOUT_DEFAULT")
        assert hasattr(SchemaChangeType, "FIELD_TYPE_CHANGED")
        assert hasattr(SchemaChangeType, "CLASS_RENAMED")


class TestSchemaChange:
    """Tests for SchemaChange."""

    def test_change_creation(self):
        """Test creating a schema change."""
        change = SchemaChange(
            change_type=SchemaChangeType.FIELD_REMOVED,
            field_name="x",
            description="Field 'x' was removed",
            old_value=int,
            new_value=None,
        )

        assert change.change_type == SchemaChangeType.FIELD_REMOVED
        assert change.field_name == "x"
        assert "removed" in change.description

    def test_is_breaking(self):
        """Test breaking change detection."""
        breaking = SchemaChange(
            change_type=SchemaChangeType.FIELD_REMOVED,
            field_name="x",
            description="Removed",
        )
        assert breaking.is_breaking is True

        non_breaking = SchemaChange(
            change_type=SchemaChangeType.FIELD_ADDED_WITH_DEFAULT,
            field_name="y",
            description="Added",
        )
        assert non_breaking.is_breaking is False

    def test_severity_levels(self):
        """Test severity level categorization."""
        breaking = SchemaChange(
            change_type=SchemaChangeType.FIELD_REMOVED,
            field_name="x",
            description="",
        )
        assert breaking.severity == "breaking"

        warning = SchemaChange(
            change_type=SchemaChangeType.FIELD_TYPE_WIDENED,
            field_name="x",
            description="",
        )
        assert warning.severity == "warning"

        info = SchemaChange(
            change_type=SchemaChangeType.METADATA_CHANGED,
            field_name="x",
            description="",
        )
        assert info.severity == "info"


class TestSchemaComparison:
    """Tests for SchemaComparison."""

    def test_comparison_identical(self):
        """Test comparison with identical schemas."""
        comparison = SchemaComparison(
            old_hash="abc123",
            new_hash="abc123",
            old_class_name="TestClass",
            new_class_name="TestClass",
            changes=[],
        )

        assert comparison.is_identical is True
        assert comparison.is_compatible is True
        assert len(comparison.breaking_changes) == 0

    def test_comparison_compatible(self):
        """Test comparison with compatible changes."""
        comparison = SchemaComparison(
            old_hash="abc123",
            new_hash="def456",
            old_class_name="TestClass",
            new_class_name="TestClass",
            changes=[
                SchemaChange(
                    change_type=SchemaChangeType.FIELD_ADDED_WITH_DEFAULT,
                    field_name="new_field",
                    description="Added new field",
                ),
            ],
        )

        assert comparison.is_identical is False
        assert comparison.is_compatible is True

    def test_comparison_incompatible(self):
        """Test comparison with breaking changes."""
        comparison = SchemaComparison(
            old_hash="abc123",
            new_hash="def456",
            old_class_name="TestClass",
            new_class_name="TestClass",
            changes=[
                SchemaChange(
                    change_type=SchemaChangeType.FIELD_REMOVED,
                    field_name="old_field",
                    description="Removed field",
                ),
            ],
        )

        assert comparison.is_compatible is False
        assert len(comparison.breaking_changes) == 1

    def test_summary(self):
        """Test comparison summary."""
        identical = SchemaComparison(
            old_hash="a", new_hash="a",
            old_class_name="T", new_class_name="T",
            changes=[],
        )
        assert "identical" in identical.summary().lower()

        incompatible = SchemaComparison(
            old_hash="a", new_hash="b",
            old_class_name="T", new_class_name="T",
            changes=[
                SchemaChange(
                    change_type=SchemaChangeType.FIELD_REMOVED,
                    field_name="x",
                    description="",
                ),
            ],
        )
        assert "incompatible" in incompatible.summary().lower()


class TestSchemaHasher:
    """Tests for SchemaHasher."""

    def setup_method(self):
        """Create fresh hasher for each test."""
        self.hasher = SchemaHasher()

    def test_compute_hash(self):
        """Test computing schema hash."""
        @dataclass
        class TestClass:
            x: int = 10
            y: str = "test"

        hash_value = self.hasher.compute_hash(TestClass)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 16  # Schema hash length

    def test_hash_consistency(self):
        """Test that same class produces same hash."""
        @dataclass
        class TestClass:
            x: int = 10

        hash1 = self.hasher.compute_hash(TestClass)
        hash2 = self.hasher.compute_hash(TestClass)

        assert hash1 == hash2

    def test_hash_differs_for_different_classes(self):
        """Test that different classes produce different hashes."""
        @dataclass
        class ClassA:
            x: int = 10

        @dataclass
        class ClassB:
            y: str = "test"

        hash_a = self.hasher.compute_hash(ClassA)
        hash_b = self.hasher.compute_hash(ClassB)

        assert hash_a != hash_b

    def test_get_schema_info(self):
        """Test getting detailed schema info."""
        @dataclass
        class TestClass:
            x: int = 10
            y: str = "test"

            def method(self):
                pass

        info = self.hasher.get_schema_info(TestClass)

        assert info["class_name"] == "TestClass"
        assert "x" in info["fields"]
        assert "y" in info["fields"]
        assert "method" in info["methods"]

    def test_compare_identical_schemas(self):
        """Test comparing identical schemas."""
        @dataclass
        class TestClass:
            x: int = 10

        comparison = self.hasher.compare_schemas(TestClass, TestClass)

        assert comparison.is_identical is True
        assert len(comparison.changes) == 0

    def test_compare_field_added(self):
        """Test detecting added field."""
        @dataclass
        class OldClass:
            x: int = 10

        @dataclass
        class NewClass:
            x: int = 10
            y: str = "new"

        comparison = self.hasher.compare_schemas(OldClass, NewClass)

        assert not comparison.is_identical
        assert any(
            c.change_type == SchemaChangeType.FIELD_ADDED_WITH_DEFAULT
            for c in comparison.changes
        )

    def test_compare_field_removed(self):
        """Test detecting removed field."""
        @dataclass
        class OldClass:
            x: int = 10
            y: str = "old"

        @dataclass
        class NewClass:
            x: int = 10

        comparison = self.hasher.compare_schemas(OldClass, NewClass)

        assert any(
            c.change_type == SchemaChangeType.FIELD_REMOVED
            for c in comparison.changes
        )
        assert not comparison.is_compatible

    def test_compare_field_type_changed(self):
        """Test detecting field type change."""
        @dataclass
        class OldClass:
            x: int = 10

        @dataclass
        class NewClass:
            x: str = "10"

        comparison = self.hasher.compare_schemas(OldClass, NewClass)

        type_changes = [
            c for c in comparison.changes
            if c.change_type in {
                SchemaChangeType.FIELD_TYPE_CHANGED,
                SchemaChangeType.FIELD_TYPE_WIDENED,
                SchemaChangeType.FIELD_TYPE_NARROWED,
            }
        ]
        assert len(type_changes) > 0

    def test_compare_default_changed(self):
        """Test detecting default value change."""
        @dataclass
        class OldClass:
            x: int = 10

        @dataclass
        class NewClass:
            x: int = 20

        comparison = self.hasher.compare_schemas(OldClass, NewClass)

        assert any(
            c.change_type == SchemaChangeType.DEFAULT_VALUE_CHANGED
            for c in comparison.changes
        )

    def test_compare_class_renamed(self):
        """Test detecting class rename."""
        @dataclass
        class OldName:
            x: int = 10

        @dataclass
        class NewName:
            x: int = 10

        comparison = self.hasher.compare_schemas(OldName, NewName)

        assert any(
            c.change_type == SchemaChangeType.CLASS_RENAMED
            for c in comparison.changes
        )

    def test_requires_migration(self):
        """Test migration requirement detection."""
        @dataclass
        class OldClass:
            x: int = 10
            y: str = "test"

        @dataclass
        class NewClass:
            x: int = 10
            # y removed

        assert self.hasher.requires_migration(OldClass, NewClass) is True

        # Same class should not require migration
        assert self.hasher.requires_migration(OldClass, OldClass) is False

    def test_generate_migration_hints(self):
        """Test migration hint generation."""
        @dataclass
        class OldClass:
            x: int = 10
            y: str = "test"

        @dataclass
        class NewClass:
            x: int = 10
            # y removed

        comparison = self.hasher.compare_schemas(OldClass, NewClass)
        hints = self.hasher.generate_migration_hints(comparison)

        assert len(hints) > 0
        # Should have hint for removed field


class TestTypeWidening:
    """Tests for type widening/narrowing detection."""

    def setup_method(self):
        self.hasher = SchemaHasher()

    def test_int_to_float_widening(self):
        """Test int to float is widening."""
        @dataclass
        class OldClass:
            x: int = 10

        @dataclass
        class NewClass:
            x: float = 10.0

        comparison = self.hasher.compare_schemas(OldClass, NewClass)

        type_changes = [
            c for c in comparison.changes
            if "type" in c.change_type.name.lower()
        ]
        # int -> float should be widening
        assert any(
            c.change_type == SchemaChangeType.FIELD_TYPE_WIDENED
            for c in type_changes
        )

    def test_float_to_int_narrowing(self):
        """Test float to int is narrowing."""
        @dataclass
        class OldClass:
            x: float = 10.0

        @dataclass
        class NewClass:
            x: int = 10

        comparison = self.hasher.compare_schemas(OldClass, NewClass)

        type_changes = [
            c for c in comparison.changes
            if "type" in c.change_type.name.lower()
        ]
        # float -> int should be narrowing
        assert any(
            c.change_type == SchemaChangeType.FIELD_TYPE_NARROWED
            for c in type_changes
        )
