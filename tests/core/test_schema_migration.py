"""Tests for schema migration system (T-CC-3.5).

Tests cover:
- SchemaMigration base class and concrete implementations
- MigrationRegistry for storing and chaining migrations
- Schema change detection via introspection
- Migration scaffold generation
- Migration validation and history tracking
"""
import copy
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pytest

from engine.core.schema_migration import (
    ChangeType,
    CompositeMigration,
    ExtractedSchema,
    FieldAddedMigration,
    FieldInfo,
    FieldRemovedMigration,
    FieldRenamedMigration,
    FieldTypeChangedMigration,
    MigrationBuilder,
    MigrationHistoryEntry,
    MigrationRegistry,
    MigrationResult,
    MigrationValidationResult,
    SchemaChange,
    SchemaChangeSet,
    SchemaMigration,
    apply_migration_safely,
    create_migration_chain,
    detect_schema_changes,
    generate_migration,
    generate_migration_from_classes,
)
from engine.core.serialization import SchemaVersion, serializable


# =============================================================================
# Test Fixtures and Helper Classes
# =============================================================================

@serializable(version="1.0.0")
@dataclass
class PersonV1:
    """Version 1 of Person schema."""
    name: str
    age: int


@serializable(version="1.1.0")
@dataclass
class PersonV1_1:
    """Version 1.1 with added email field."""
    name: str
    age: int
    email: str = ""


@serializable(version="2.0.0")
@dataclass
class PersonV2:
    """Version 2 with renamed field and new structure."""
    full_name: str  # Renamed from 'name'
    age: int
    email: str
    active: bool = True


@dataclass
class SimpleData:
    value: int
    label: str


@dataclass
class ComplexData:
    items: List[int]
    mapping: Dict[str, str]
    optional: Optional[str] = None


class TestPersonMigration(SchemaMigration):
    """Migration from PersonV1 to PersonV1_1."""
    from_version = SchemaVersion(1, 0, 0)
    to_version = SchemaVersion(1, 1, 0)
    description = "Add email field"

    def migrate(self, old_data: Dict[str, Any], old_version: SchemaVersion) -> Dict[str, Any]:
        result = copy.deepcopy(old_data)
        if "email" not in result:
            result["email"] = ""
        return result

    def rollback(self, new_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        result = copy.deepcopy(new_data)
        result.pop("email", None)
        return result

    def get_test_cases(self) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        # Only include test cases where rollback can restore original state
        # (i.e., original data doesn't have 'email' field that would be removed)
        return [
            ({"name": "John", "age": 30}, {"name": "John", "age": 30, "email": ""}),
        ]


class TestPersonMigration2(SchemaMigration):
    """Migration from PersonV1_1 to PersonV2."""
    from_version = SchemaVersion(1, 1, 0)
    to_version = SchemaVersion(2, 0, 0)
    description = "Rename name to full_name, add active"

    def migrate(self, old_data: Dict[str, Any], old_version: SchemaVersion) -> Dict[str, Any]:
        result = copy.deepcopy(old_data)
        if "name" in result:
            result["full_name"] = result.pop("name")
        if "active" not in result:
            result["active"] = True
        return result

    def rollback(self, new_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        result = copy.deepcopy(new_data)
        if "full_name" in result:
            result["name"] = result.pop("full_name")
        result.pop("active", None)
        return result

    def get_test_cases(self) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        return [
            (
                {"name": "John", "age": 30, "email": "john@test.com"},
                {"full_name": "John", "age": 30, "email": "john@test.com", "active": True},
            ),
        ]


# =============================================================================
# Tests for SchemaVersion
# =============================================================================

class TestSchemaChangeType:
    """Tests for ChangeType enum."""

    def test_all_types_exist(self):
        assert ChangeType.FIELD_ADDED is not None
        assert ChangeType.FIELD_REMOVED is not None
        assert ChangeType.FIELD_RENAMED is not None
        assert ChangeType.FIELD_TYPE_CHANGED is not None
        assert ChangeType.FIELD_DEFAULT_CHANGED is not None
        assert ChangeType.FIELD_OPTIONAL_CHANGED is not None
        assert ChangeType.NESTED_SCHEMA_CHANGED is not None


# =============================================================================
# Tests for SchemaChange
# =============================================================================

class TestSchemaChange:
    """Tests for SchemaChange dataclass."""

    def test_field_added(self):
        change = SchemaChange(
            change_type=ChangeType.FIELD_ADDED,
            field_name="email",
            new_type="str",
            new_value="",
        )
        assert "email" in str(change)
        assert "added" in str(change).lower()

    def test_field_removed(self):
        change = SchemaChange(
            change_type=ChangeType.FIELD_REMOVED,
            field_name="old_field",
            old_type="int",
        )
        assert "old_field" in str(change)
        assert "removed" in str(change).lower()

    def test_field_renamed(self):
        change = SchemaChange(
            change_type=ChangeType.FIELD_RENAMED,
            field_name="name",
            old_value="name",
            new_value="full_name",
        )
        assert "renamed" in str(change).lower()
        assert "name" in str(change)
        assert "full_name" in str(change)

    def test_field_type_changed(self):
        change = SchemaChange(
            change_type=ChangeType.FIELD_TYPE_CHANGED,
            field_name="count",
            old_type="str",
            new_type="int",
        )
        assert "count" in str(change)
        assert "str" in str(change)
        assert "int" in str(change)

    def test_field_default_changed(self):
        change = SchemaChange(
            change_type=ChangeType.FIELD_DEFAULT_CHANGED,
            field_name="value",
            old_value=0,
            new_value=10,
        )
        assert "value" in str(change)
        assert "default" in str(change).lower()

    def test_frozen(self):
        change = SchemaChange(
            change_type=ChangeType.FIELD_ADDED,
            field_name="test",
        )
        # Frozen dataclass should not allow modification
        with pytest.raises(Exception):
            change.field_name = "other"


# =============================================================================
# Tests for SchemaChangeSet
# =============================================================================

class TestSchemaChangeSet:
    """Tests for SchemaChangeSet."""

    def test_creation(self):
        change_set = SchemaChangeSet(
            old_version=SchemaVersion(1, 0, 0),
            new_version=SchemaVersion(1, 1, 0),
        )
        assert change_set.is_empty()
        assert not change_set.breaking
        assert change_set.auto_migratable

    def test_add_change(self):
        change_set = SchemaChangeSet(
            old_version=SchemaVersion(1, 0, 0),
            new_version=SchemaVersion(1, 1, 0),
        )
        change_set.add_change(SchemaChange(
            change_type=ChangeType.FIELD_ADDED,
            field_name="email",
        ))
        assert not change_set.is_empty()
        assert len(change_set.changes) == 1

    def test_breaking_change_detection(self):
        change_set = SchemaChangeSet(
            old_version=SchemaVersion(1, 0, 0),
            new_version=SchemaVersion(2, 0, 0),
        )
        # Field removed is breaking
        change_set.add_change(SchemaChange(
            change_type=ChangeType.FIELD_REMOVED,
            field_name="important",
        ))
        assert change_set.breaking

    def test_type_change_is_breaking(self):
        change_set = SchemaChangeSet(
            old_version=SchemaVersion(1, 0, 0),
            new_version=SchemaVersion(2, 0, 0),
        )
        change_set.add_change(SchemaChange(
            change_type=ChangeType.FIELD_TYPE_CHANGED,
            field_name="count",
            old_type="str",
            new_type="int",
        ))
        assert change_set.breaking

    def test_get_changes_by_type(self):
        change_set = SchemaChangeSet(
            old_version=SchemaVersion(1, 0, 0),
            new_version=SchemaVersion(2, 0, 0),
        )
        change_set.add_change(SchemaChange(change_type=ChangeType.FIELD_ADDED, field_name="a"))
        change_set.add_change(SchemaChange(change_type=ChangeType.FIELD_ADDED, field_name="b"))
        change_set.add_change(SchemaChange(change_type=ChangeType.FIELD_REMOVED, field_name="c"))

        added = change_set.get_changes_by_type(ChangeType.FIELD_ADDED)
        assert len(added) == 2

        removed = change_set.get_changes_by_type(ChangeType.FIELD_REMOVED)
        assert len(removed) == 1

    def test_summary(self):
        change_set = SchemaChangeSet(
            old_version=SchemaVersion(1, 0, 0),
            new_version=SchemaVersion(1, 1, 0),
        )
        change_set.add_change(SchemaChange(
            change_type=ChangeType.FIELD_ADDED,
            field_name="email",
            new_type="str",
        ))
        summary = change_set.summary()
        assert "email" in summary
        assert "1.0.0" in summary
        assert "1.1.0" in summary


# =============================================================================
# Tests for MigrationResult
# =============================================================================

class TestMigrationResult:
    """Tests for MigrationResult."""

    def test_success(self):
        result = MigrationResult.succeeded({"value": 42})
        assert result.success
        assert result.data["value"] == 42
        assert len(result.errors) == 0

    def test_failure(self):
        result = MigrationResult.failed("Migration failed")
        assert not result.success
        assert "Migration failed" in result.errors[0]

    def test_failure_with_data(self):
        result = MigrationResult.failed("Error", {"partial": True})
        assert not result.success
        assert result.data["partial"]

    def test_add_warning(self):
        result = MigrationResult.succeeded({})
        result.add_warning("Data loss possible")
        assert "Data loss possible" in result.warnings


# =============================================================================
# Tests for MigrationValidationResult
# =============================================================================

class TestMigrationValidationResult:
    """Tests for MigrationValidationResult."""

    def test_initial_state(self):
        result = MigrationValidationResult()
        assert result.valid
        assert bool(result)

    def test_add_error(self):
        result = MigrationValidationResult()
        result.add_error("Invalid input")
        assert not result.valid
        assert not bool(result)
        assert "Invalid input" in result.errors

    def test_add_warning(self):
        result = MigrationValidationResult()
        result.add_warning("Possible data loss")
        assert result.valid  # Warnings don't invalidate
        assert "Possible data loss" in result.warnings

    def test_add_test_result_pass(self):
        result = MigrationValidationResult()
        result.add_test_result("test_1", True, "Passed")
        assert result.valid

    def test_add_test_result_fail(self):
        result = MigrationValidationResult()
        result.add_test_result("test_1", False, "Expected X got Y")
        assert not result.valid


# =============================================================================
# Tests for SchemaMigration Base Class
# =============================================================================

class TestSchemaMigration:
    """Tests for SchemaMigration base class."""

    def test_abstract_migrate(self):
        with pytest.raises(TypeError):
            SchemaMigration()

    def test_concrete_implementation(self):
        migration = TestPersonMigration()
        assert migration.from_version == SchemaVersion(1, 0, 0)
        assert migration.to_version == SchemaVersion(1, 1, 0)

    def test_migrate(self):
        migration = TestPersonMigration()
        result = migration.migrate(
            {"name": "John", "age": 30},
            SchemaVersion(1, 0, 0),
        )
        assert result["email"] == ""

    def test_rollback(self):
        migration = TestPersonMigration()
        result = migration.rollback(
            {"name": "John", "age": 30, "email": "john@test.com"},
        )
        assert "email" not in result

    def test_supports_rollback(self):
        migration = TestPersonMigration()
        assert migration.supports_rollback()

    def test_validate_input(self):
        migration = TestPersonMigration()
        result = migration.validate_input({"name": "John", "age": 30})
        assert result.valid

    def test_validate_input_invalid(self):
        migration = TestPersonMigration()
        result = migration.validate_input("not a dict")
        assert not result.valid

    def test_get_test_cases(self):
        migration = TestPersonMigration()
        cases = migration.get_test_cases()
        assert len(cases) >= 1


# =============================================================================
# Tests for FieldAddedMigration
# =============================================================================

class TestFieldAddedMigration:
    """Tests for FieldAddedMigration."""

    def test_add_field(self):
        migration = FieldAddedMigration(
            field_name="email",
            default_value="",
            from_ver=SchemaVersion(1, 0, 0),
            to_ver=SchemaVersion(1, 1, 0),
        )
        result = migration.migrate({"name": "John"}, SchemaVersion(1, 0, 0))
        assert result["email"] == ""

    def test_preserves_existing(self):
        migration = FieldAddedMigration(
            field_name="email",
            default_value="default@test.com",
            from_ver=SchemaVersion(1, 0, 0),
            to_ver=SchemaVersion(1, 1, 0),
        )
        result = migration.migrate(
            {"name": "John", "email": "john@test.com"},
            SchemaVersion(1, 0, 0),
        )
        assert result["email"] == "john@test.com"

    def test_rollback(self):
        migration = FieldAddedMigration(
            field_name="email",
            default_value="",
            from_ver=SchemaVersion(1, 0, 0),
            to_ver=SchemaVersion(1, 1, 0),
        )
        result = migration.rollback({"name": "John", "email": "john@test.com"})
        assert "email" not in result

    def test_complex_default(self):
        migration = FieldAddedMigration(
            field_name="tags",
            default_value=["default"],
            from_ver=SchemaVersion(1, 0, 0),
            to_ver=SchemaVersion(1, 1, 0),
        )
        result = migration.migrate({}, SchemaVersion(1, 0, 0))
        assert result["tags"] == ["default"]


# =============================================================================
# Tests for FieldRemovedMigration
# =============================================================================

class TestFieldRemovedMigration:
    """Tests for FieldRemovedMigration."""

    def test_remove_field(self):
        migration = FieldRemovedMigration(
            field_name="old_field",
            from_ver=SchemaVersion(1, 0, 0),
            to_ver=SchemaVersion(2, 0, 0),
        )
        result = migration.migrate(
            {"name": "John", "old_field": "value"},
            SchemaVersion(1, 0, 0),
        )
        assert "old_field" not in result
        assert result["name"] == "John"

    def test_preserve_in_metadata(self):
        migration = FieldRemovedMigration(
            field_name="old_field",
            from_ver=SchemaVersion(1, 0, 0),
            to_ver=SchemaVersion(2, 0, 0),
            preserve_in_metadata=True,
        )
        result = migration.migrate(
            {"name": "John", "old_field": "important"},
            SchemaVersion(1, 0, 0),
        )
        assert "old_field" not in result
        assert result["__removed_fields__"]["old_field"] == "important"

    def test_rollback_from_metadata(self):
        migration = FieldRemovedMigration(
            field_name="old_field",
            from_ver=SchemaVersion(1, 0, 0),
            to_ver=SchemaVersion(2, 0, 0),
            preserve_in_metadata=True,
        )
        result = migration.rollback({
            "name": "John",
            "__removed_fields__": {"old_field": "value"},
        })
        assert result["old_field"] == "value"

    def test_missing_field(self):
        migration = FieldRemovedMigration(
            field_name="nonexistent",
            from_ver=SchemaVersion(1, 0, 0),
            to_ver=SchemaVersion(2, 0, 0),
        )
        result = migration.migrate({"name": "John"}, SchemaVersion(1, 0, 0))
        assert result == {"name": "John"}


# =============================================================================
# Tests for FieldRenamedMigration
# =============================================================================

class TestFieldRenamedMigration:
    """Tests for FieldRenamedMigration."""

    def test_rename_field(self):
        migration = FieldRenamedMigration(
            old_name="name",
            new_name="full_name",
            from_ver=SchemaVersion(1, 0, 0),
            to_ver=SchemaVersion(2, 0, 0),
        )
        result = migration.migrate({"name": "John Doe"}, SchemaVersion(1, 0, 0))
        assert "name" not in result
        assert result["full_name"] == "John Doe"

    def test_rollback(self):
        migration = FieldRenamedMigration(
            old_name="name",
            new_name="full_name",
            from_ver=SchemaVersion(1, 0, 0),
            to_ver=SchemaVersion(2, 0, 0),
        )
        result = migration.rollback({"full_name": "John Doe"})
        assert "full_name" not in result
        assert result["name"] == "John Doe"

    def test_missing_field(self):
        migration = FieldRenamedMigration(
            old_name="missing",
            new_name="new_name",
            from_ver=SchemaVersion(1, 0, 0),
            to_ver=SchemaVersion(2, 0, 0),
        )
        result = migration.migrate({"other": "value"}, SchemaVersion(1, 0, 0))
        assert "new_name" not in result


# =============================================================================
# Tests for FieldTypeChangedMigration
# =============================================================================

class TestFieldTypeChangedMigration:
    """Tests for FieldTypeChangedMigration."""

    def test_type_conversion(self):
        migration = FieldTypeChangedMigration(
            field_name="count",
            converter=int,
            from_ver=SchemaVersion(1, 0, 0),
            to_ver=SchemaVersion(2, 0, 0),
        )
        result = migration.migrate({"count": "42"}, SchemaVersion(1, 0, 0))
        assert result["count"] == 42
        assert isinstance(result["count"], int)

    def test_rollback_with_reverse(self):
        migration = FieldTypeChangedMigration(
            field_name="count",
            converter=int,
            from_ver=SchemaVersion(1, 0, 0),
            to_ver=SchemaVersion(2, 0, 0),
            reverse_converter=str,
        )
        result = migration.rollback({"count": 42})
        assert result["count"] == "42"
        assert isinstance(result["count"], str)

    def test_rollback_without_reverse(self):
        migration = FieldTypeChangedMigration(
            field_name="count",
            converter=int,
            from_ver=SchemaVersion(1, 0, 0),
            to_ver=SchemaVersion(2, 0, 0),
        )
        result = migration.rollback({"count": 42})
        assert result is None

    def test_complex_conversion(self):
        def list_to_set(value):
            return list(set(value))

        migration = FieldTypeChangedMigration(
            field_name="items",
            converter=list_to_set,
            from_ver=SchemaVersion(1, 0, 0),
            to_ver=SchemaVersion(2, 0, 0),
        )
        result = migration.migrate({"items": [1, 2, 2, 3]}, SchemaVersion(1, 0, 0))
        assert sorted(result["items"]) == [1, 2, 3]


# =============================================================================
# Tests for CompositeMigration
# =============================================================================

class TestCompositeMigration:
    """Tests for CompositeMigration."""

    def test_multiple_operations(self):
        migration = CompositeMigration(
            migrations=[
                FieldAddedMigration("email", "", SchemaVersion(1, 0, 0), SchemaVersion(1, 1, 0)),
                FieldRenamedMigration("name", "full_name", SchemaVersion(1, 0, 0), SchemaVersion(1, 1, 0)),
            ],
            from_ver=SchemaVersion(1, 0, 0),
            to_ver=SchemaVersion(2, 0, 0),
        )
        result = migration.migrate({"name": "John"}, SchemaVersion(1, 0, 0))
        assert result["email"] == ""
        assert result["full_name"] == "John"
        assert "name" not in result

    def test_rollback_composite(self):
        migration = CompositeMigration(
            migrations=[
                FieldAddedMigration("email", "", SchemaVersion(1, 0, 0), SchemaVersion(1, 1, 0)),
                FieldRenamedMigration("name", "full_name", SchemaVersion(1, 0, 0), SchemaVersion(1, 1, 0)),
            ],
            from_ver=SchemaVersion(1, 0, 0),
            to_ver=SchemaVersion(2, 0, 0),
        )
        result = migration.rollback({"full_name": "John", "email": ""})
        assert "email" not in result
        assert result["name"] == "John"

    def test_rollback_fails_if_any_fails(self):
        migration = CompositeMigration(
            migrations=[
                FieldAddedMigration("email", "", SchemaVersion(1, 0, 0), SchemaVersion(1, 1, 0)),
                FieldTypeChangedMigration("count", int, SchemaVersion(1, 0, 0), SchemaVersion(1, 1, 0)),  # No reverse
            ],
            from_ver=SchemaVersion(1, 0, 0),
            to_ver=SchemaVersion(2, 0, 0),
        )
        result = migration.rollback({"email": "", "count": 42})
        assert result is None


# =============================================================================
# Tests for MigrationRegistry
# =============================================================================

class TestMigrationRegistry:
    """Tests for MigrationRegistry."""

    def setup_method(self):
        MigrationRegistry.get_instance().clear()
        MigrationRegistry.get_instance().clear_history()

    def test_singleton(self):
        r1 = MigrationRegistry.get_instance()
        r2 = MigrationRegistry.get_instance()
        assert r1 is r2

    def test_register_and_get(self):
        registry = MigrationRegistry.get_instance()
        migration = TestPersonMigration()
        registry.register(migration)

        result = registry.get_migration(
            SchemaVersion(1, 0, 0),
            SchemaVersion(1, 1, 0),
        )
        assert result is migration

    def test_unregister(self):
        registry = MigrationRegistry.get_instance()
        migration = TestPersonMigration()
        registry.register(migration)
        registry.unregister(SchemaVersion(1, 0, 0), SchemaVersion(1, 1, 0))

        result = registry.get_migration(
            SchemaVersion(1, 0, 0),
            SchemaVersion(1, 1, 0),
        )
        assert result is None

    def test_find_migration_path_direct(self):
        registry = MigrationRegistry.get_instance()
        registry.register(TestPersonMigration())

        path = registry.find_migration_path(
            SchemaVersion(1, 0, 0),
            SchemaVersion(1, 1, 0),
        )
        assert path is not None
        assert len(path) == 1

    def test_find_migration_path_chain(self):
        registry = MigrationRegistry.get_instance()
        registry.register(TestPersonMigration())
        registry.register(TestPersonMigration2())

        path = registry.find_migration_path(
            SchemaVersion(1, 0, 0),
            SchemaVersion(2, 0, 0),
        )
        assert path is not None
        assert len(path) == 2

    def test_find_migration_path_none(self):
        registry = MigrationRegistry.get_instance()
        registry.register(TestPersonMigration())

        path = registry.find_migration_path(
            SchemaVersion(1, 0, 0),
            SchemaVersion(3, 0, 0),
        )
        assert path is None

    def test_find_migration_path_same_version(self):
        registry = MigrationRegistry.get_instance()
        path = registry.find_migration_path(
            SchemaVersion(1, 0, 0),
            SchemaVersion(1, 0, 0),
        )
        assert path == []

    def test_migrate_direct(self):
        registry = MigrationRegistry.get_instance()
        registry.register(TestPersonMigration())

        result = registry.migrate(
            {"name": "John", "age": 30},
            SchemaVersion(1, 0, 0),
            SchemaVersion(1, 1, 0),
        )
        assert result.success
        assert result.data["email"] == ""

    def test_migrate_chain(self):
        registry = MigrationRegistry.get_instance()
        registry.register(TestPersonMigration())
        registry.register(TestPersonMigration2())

        result = registry.migrate(
            {"name": "John", "age": 30},
            SchemaVersion(1, 0, 0),
            SchemaVersion(2, 0, 0),
        )
        assert result.success
        assert result.data["full_name"] == "John"
        assert result.data["active"]

    def test_migrate_no_path(self):
        registry = MigrationRegistry.get_instance()
        result = registry.migrate(
            {"name": "John"},
            SchemaVersion(1, 0, 0),
            SchemaVersion(5, 0, 0),
        )
        assert not result.success

    def test_validate_migration(self):
        registry = MigrationRegistry.get_instance()
        migration = TestPersonMigration()

        result = registry.validate_migration(migration)
        assert result.valid

    def test_validate_migration_with_test_data(self):
        registry = MigrationRegistry.get_instance()
        migration = TestPersonMigration()

        result = registry.validate_migration(
            migration,
            test_data=[
                ({"name": "Test", "age": 20}, {"name": "Test", "age": 20, "email": ""}),
            ],
        )
        assert result.valid

    def test_history_recording(self):
        registry = MigrationRegistry.get_instance()
        registry.register(TestPersonMigration())

        registry.migrate(
            {"name": "John", "age": 30},
            SchemaVersion(1, 0, 0),
            SchemaVersion(1, 1, 0),
        )

        history = registry.get_history()
        assert len(history) == 1
        assert history[0].success

    def test_history_limit(self):
        registry = MigrationRegistry.get_instance()
        registry.register(TestPersonMigration())

        for _ in range(5):
            registry.migrate(
                {"name": "John", "age": 30},
                SchemaVersion(1, 0, 0),
                SchemaVersion(1, 1, 0),
            )

        history = registry.get_history(limit=2)
        assert len(history) == 2

    def test_list_migrations(self):
        registry = MigrationRegistry.get_instance()
        registry.register(TestPersonMigration())
        registry.register(TestPersonMigration2())

        migrations = registry.list_migrations()
        assert len(migrations) == 2


# =============================================================================
# Tests for MigrationHistoryEntry
# =============================================================================

class TestMigrationHistoryEntry:
    """Tests for MigrationHistoryEntry."""

    def test_to_dict(self):
        entry = MigrationHistoryEntry(
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            from_version=SchemaVersion(1, 0, 0),
            to_version=SchemaVersion(1, 1, 0),
            migration_name="TestMigration",
            success=True,
            duration_ms=42.5,
        )
        d = entry.to_dict()
        assert d["migration_name"] == "TestMigration"
        assert d["success"]
        assert d["duration_ms"] == 42.5

    def test_from_dict(self):
        d = {
            "timestamp": "2024-01-15T10:30:00",
            "from_version": "1.0.0",
            "to_version": "1.1.0",
            "migration_name": "TestMigration",
            "success": True,
            "duration_ms": 42.5,
        }
        entry = MigrationHistoryEntry.from_dict(d)
        assert entry.migration_name == "TestMigration"
        assert entry.success


# =============================================================================
# Tests for FieldInfo
# =============================================================================

class TestFieldInfo:
    """Tests for FieldInfo."""

    def test_equality(self):
        f1 = FieldInfo(name="value", type_name="int")
        f2 = FieldInfo(name="value", type_name="int")
        f3 = FieldInfo(name="value", type_name="str")

        assert f1 == f2
        assert f1 != f3

    def test_optional_flag(self):
        f = FieldInfo(name="value", type_name="Optional[str]", is_optional=True)
        assert f.is_optional


# =============================================================================
# Tests for ExtractedSchema
# =============================================================================

class TestExtractedSchema:
    """Tests for ExtractedSchema."""

    def test_from_dataclass(self):
        schema = ExtractedSchema.from_class(PersonV1)
        assert schema.class_name == "PersonV1"
        assert "name" in schema.fields
        assert "age" in schema.fields
        assert schema.fields["name"].type_name == "str"

    def test_from_dict(self):
        data = {
            "__schema__": {"name": "Test", "version": "1.0.0"},
            "value": 42,
            "label": "test",
        }
        schema = ExtractedSchema.from_dict(data)
        assert "value" in schema.fields
        assert schema.fields["value"].type_name == "int"

    def test_from_dict_infers_types(self):
        data = {
            "items": [1, 2, 3],
            "mapping": {"a": "b"},
            "flag": True,
        }
        schema = ExtractedSchema.from_dict(data)
        assert "List" in schema.fields["items"].type_name
        assert "Dict" in schema.fields["mapping"].type_name
        assert schema.fields["flag"].type_name == "bool"


# =============================================================================
# Tests for detect_schema_changes
# =============================================================================

class TestDetectSchemaChanges:
    """Tests for detect_schema_changes function."""

    def test_no_changes(self):
        changes = detect_schema_changes(PersonV1, PersonV1)
        assert changes.is_empty()

    def test_field_added(self):
        changes = detect_schema_changes(PersonV1, PersonV1_1)
        added = changes.get_changes_by_type(ChangeType.FIELD_ADDED)
        assert len(added) == 1
        assert added[0].field_name == "email"

    def test_field_removed(self):
        @dataclass
        class OldSchema:
            name: str
            old_field: int

        @dataclass
        class NewSchema:
            name: str

        changes = detect_schema_changes(OldSchema, NewSchema)
        removed = changes.get_changes_by_type(ChangeType.FIELD_REMOVED)
        assert len(removed) == 1
        assert removed[0].field_name == "old_field"

    def test_field_renamed_detection(self):
        @serializable(version="1.0.0")
        @dataclass
        class OldSchema:
            username: str

        @serializable(version="1.1.0")
        @dataclass
        class NewSchema:
            user_name: str

        changes = detect_schema_changes(OldSchema, NewSchema, detect_renames=True)
        renamed = changes.get_changes_by_type(ChangeType.FIELD_RENAMED)
        assert len(renamed) == 1

    def test_field_type_changed(self):
        @dataclass
        class OldSchema:
            count: str

        @dataclass
        class NewSchema:
            count: int

        changes = detect_schema_changes(OldSchema, NewSchema)
        type_changed = changes.get_changes_by_type(ChangeType.FIELD_TYPE_CHANGED)
        assert len(type_changed) == 1
        assert type_changed[0].old_type == "str"
        assert type_changed[0].new_type == "int"

    def test_optional_changed(self):
        @dataclass
        class OldSchema:
            value: str

        @dataclass
        class NewSchema:
            value: Optional[str]

        changes = detect_schema_changes(OldSchema, NewSchema)
        optional_changed = changes.get_changes_by_type(ChangeType.FIELD_OPTIONAL_CHANGED)
        assert len(optional_changed) == 1

    def test_multiple_changes(self):
        @dataclass
        class OldSchema:
            name: str
            old_field: int

        @dataclass
        class NewSchema:
            name: str
            new_field: str
            another: bool = False

        changes = detect_schema_changes(OldSchema, NewSchema)
        assert not changes.is_empty()
        assert len(changes.changes) >= 2


# =============================================================================
# Tests for generate_migration
# =============================================================================

class TestGenerateMigration:
    """Tests for generate_migration function."""

    def test_generate_for_added_field(self):
        change_set = SchemaChangeSet(
            old_version=SchemaVersion(1, 0, 0),
            new_version=SchemaVersion(1, 1, 0),
        )
        change_set.add_change(SchemaChange(
            change_type=ChangeType.FIELD_ADDED,
            field_name="email",
            new_type="str",
            new_value="",
        ))

        code = generate_migration(change_set, "AddEmailMigration")
        assert "class AddEmailMigration" in code
        assert "email" in code
        assert "def migrate" in code

    def test_generate_for_removed_field(self):
        change_set = SchemaChangeSet(
            old_version=SchemaVersion(1, 0, 0),
            new_version=SchemaVersion(2, 0, 0),
        )
        change_set.add_change(SchemaChange(
            change_type=ChangeType.FIELD_REMOVED,
            field_name="old_field",
        ))

        code = generate_migration(change_set)
        assert "pop" in code
        assert "old_field" in code

    def test_generate_for_renamed_field(self):
        change_set = SchemaChangeSet(
            old_version=SchemaVersion(1, 0, 0),
            new_version=SchemaVersion(2, 0, 0),
        )
        change_set.add_change(SchemaChange(
            change_type=ChangeType.FIELD_RENAMED,
            field_name="name",
            old_value="name",
            new_value="full_name",
        ))

        code = generate_migration(change_set)
        assert "name" in code
        assert "full_name" in code

    def test_generate_includes_rollback(self):
        change_set = SchemaChangeSet(
            old_version=SchemaVersion(1, 0, 0),
            new_version=SchemaVersion(1, 1, 0),
        )
        change_set.add_change(SchemaChange(
            change_type=ChangeType.FIELD_ADDED,
            field_name="email",
        ))

        code = generate_migration(change_set)
        assert "def rollback" in code


class TestGenerateMigrationFromClasses:
    """Tests for generate_migration_from_classes function."""

    def test_generate_from_classes(self):
        code = generate_migration_from_classes(
            PersonV1,
            PersonV1_1,
            "PersonMigration",
        )
        assert "class PersonMigration" in code
        assert "email" in code


# =============================================================================
# Tests for MigrationBuilder
# =============================================================================

class TestMigrationBuilder:
    """Tests for MigrationBuilder."""

    def test_add_field(self):
        migration = (
            MigrationBuilder(SchemaVersion(1, 0, 0), SchemaVersion(1, 1, 0))
            .add_field("email", "")
            .build()
        )
        result = migration.migrate({"name": "John"}, SchemaVersion(1, 0, 0))
        assert result["email"] == ""

    def test_remove_field(self):
        migration = (
            MigrationBuilder(SchemaVersion(1, 0, 0), SchemaVersion(2, 0, 0))
            .remove_field("old_field")
            .build()
        )
        result = migration.migrate({"name": "John", "old_field": "x"}, SchemaVersion(1, 0, 0))
        assert "old_field" not in result

    def test_rename_field(self):
        migration = (
            MigrationBuilder(SchemaVersion(1, 0, 0), SchemaVersion(2, 0, 0))
            .rename_field("name", "full_name")
            .build()
        )
        result = migration.migrate({"name": "John"}, SchemaVersion(1, 0, 0))
        assert result["full_name"] == "John"

    def test_change_field_type(self):
        migration = (
            MigrationBuilder(SchemaVersion(1, 0, 0), SchemaVersion(2, 0, 0))
            .change_field_type("count", int, str)
            .build()
        )
        result = migration.migrate({"count": "42"}, SchemaVersion(1, 0, 0))
        assert result["count"] == 42

    def test_chained_operations(self):
        migration = (
            MigrationBuilder(SchemaVersion(1, 0, 0), SchemaVersion(2, 0, 0))
            .add_field("email", "")
            .rename_field("name", "full_name")
            .remove_field("temp")
            .build()
        )
        result = migration.migrate(
            {"name": "John", "temp": "x"},
            SchemaVersion(1, 0, 0),
        )
        assert result["email"] == ""
        assert result["full_name"] == "John"
        assert "temp" not in result

    def test_single_operation_returns_simple_migration(self):
        migration = (
            MigrationBuilder(SchemaVersion(1, 0, 0), SchemaVersion(1, 1, 0))
            .add_field("email", "")
            .build()
        )
        assert isinstance(migration, FieldAddedMigration)

    def test_multiple_operations_returns_composite(self):
        migration = (
            MigrationBuilder(SchemaVersion(1, 0, 0), SchemaVersion(2, 0, 0))
            .add_field("a", "")
            .add_field("b", "")
            .build()
        )
        assert isinstance(migration, CompositeMigration)


# =============================================================================
# Tests for apply_migration_safely
# =============================================================================

class TestApplyMigrationSafely:
    """Tests for apply_migration_safely function."""

    def test_successful_migration(self):
        migration = FieldAddedMigration(
            "email", "", SchemaVersion(1, 0, 0), SchemaVersion(1, 1, 0),
        )
        result = apply_migration_safely({"name": "John"}, migration)
        assert result.success
        assert result.data["email"] == ""

    def test_input_validation_failure(self):
        class StrictMigration(SchemaMigration):
            from_version = SchemaVersion(1, 0, 0)
            to_version = SchemaVersion(1, 1, 0)

            def migrate(self, data, version):
                return data

            def validate_input(self, data):
                result = MigrationValidationResult()
                if "required" not in data:
                    result.add_error("Missing required field")
                return result

        migration = StrictMigration()
        result = apply_migration_safely({"name": "John"}, migration)
        assert not result.success

    def test_migration_exception(self):
        class FailingMigration(SchemaMigration):
            from_version = SchemaVersion(1, 0, 0)
            to_version = SchemaVersion(1, 1, 0)

            def migrate(self, data, version):
                raise RuntimeError("Migration failed")

        migration = FailingMigration()
        result = apply_migration_safely({"name": "John"}, migration)
        assert not result.success


# =============================================================================
# Tests for create_migration_chain
# =============================================================================

class TestCreateMigrationChain:
    """Tests for create_migration_chain function."""

    def test_empty_list(self):
        result = create_migration_chain([])
        assert result is None

    def test_single_migration(self):
        migration = TestPersonMigration()
        result = create_migration_chain([migration])
        assert result is not None

    def test_valid_chain(self):
        result = create_migration_chain([
            TestPersonMigration(),
            TestPersonMigration2(),
        ])
        assert result is not None
        assert result.from_version == SchemaVersion(1, 0, 0)
        assert result.to_version == SchemaVersion(2, 0, 0)

    def test_invalid_chain(self):
        # Migrations don't connect
        m1 = FieldAddedMigration("a", "", SchemaVersion(1, 0, 0), SchemaVersion(1, 1, 0))
        m2 = FieldAddedMigration("b", "", SchemaVersion(2, 0, 0), SchemaVersion(2, 1, 0))

        result = create_migration_chain([m1, m2])
        assert result is None


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the migration system."""

    def setup_method(self):
        MigrationRegistry.get_instance().clear()
        MigrationRegistry.get_instance().clear_history()

    def test_full_migration_workflow(self):
        """Test complete migration from v1 to v2."""
        registry = MigrationRegistry.get_instance()
        registry.register(TestPersonMigration())
        registry.register(TestPersonMigration2())

        # Original v1 data
        v1_data = {"name": "John Doe", "age": 30}

        # Migrate to v2
        result = registry.migrate(
            v1_data,
            SchemaVersion(1, 0, 0),
            SchemaVersion(2, 0, 0),
        )

        assert result.success
        assert result.data["full_name"] == "John Doe"
        assert result.data["age"] == 30
        assert result.data["email"] == ""
        assert result.data["active"]

    def test_migration_with_validation(self):
        """Test migration with validation."""
        registry = MigrationRegistry.get_instance()
        migration = TestPersonMigration()

        # Validate first
        validation = registry.validate_migration(migration)
        assert validation.valid

        # Then apply
        registry.register(migration)
        result = registry.migrate(
            {"name": "John", "age": 30},
            SchemaVersion(1, 0, 0),
            SchemaVersion(1, 1, 0),
        )
        assert result.success

    def test_detect_and_generate_migration(self):
        """Test detecting changes and generating migration code."""
        # Detect changes
        changes = detect_schema_changes(PersonV1, PersonV1_1)

        # Generate migration
        code = generate_migration(changes, "AutoGeneratedMigration")

        # Verify generated code contains expected elements
        assert "email" in code
        assert "def migrate" in code

    def test_builder_pattern_workflow(self):
        """Test using builder pattern for migrations."""
        migration = (
            MigrationBuilder(SchemaVersion(1, 0, 0), SchemaVersion(2, 0, 0))
            .add_field("email", "")
            .rename_field("name", "full_name")
            .add_field("active", True)
            .build()
        )

        result = migration.migrate(
            {"name": "John", "age": 30},
            SchemaVersion(1, 0, 0),
        )

        assert result["full_name"] == "John"
        assert result["email"] == ""
        assert result["active"]

    def test_roundtrip_migration(self):
        """Test migrating forward and back."""
        migration = FieldRenamedMigration(
            "name", "full_name",
            SchemaVersion(1, 0, 0),
            SchemaVersion(2, 0, 0),
        )

        original = {"name": "John", "age": 30}
        migrated = migration.migrate(original, SchemaVersion(1, 0, 0))
        restored = migration.rollback(migrated)

        assert restored == original


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_empty_data(self):
        migration = FieldAddedMigration(
            "email", "default",
            SchemaVersion(1, 0, 0), SchemaVersion(1, 1, 0),
        )
        result = migration.migrate({}, SchemaVersion(1, 0, 0))
        assert result["email"] == "default"

    def test_nested_data_preserved(self):
        migration = FieldAddedMigration(
            "metadata", {},
            SchemaVersion(1, 0, 0), SchemaVersion(1, 1, 0),
        )
        original = {"data": {"nested": {"deep": [1, 2, 3]}}}
        result = migration.migrate(original, SchemaVersion(1, 0, 0))
        assert result["data"]["nested"]["deep"] == [1, 2, 3]

    def test_unicode_field_names(self):
        migration = FieldRenamedMigration(
            "nombre",
            "name",
            SchemaVersion(1, 0, 0),
            SchemaVersion(2, 0, 0),
        )
        result = migration.migrate({"nombre": "Juan"}, SchemaVersion(1, 0, 0))
        assert result["name"] == "Juan"

    def test_special_characters_in_values(self):
        migration = FieldAddedMigration(
            "pattern", r"\d+\.\d+",
            SchemaVersion(1, 0, 0), SchemaVersion(1, 1, 0),
        )
        result = migration.migrate({}, SchemaVersion(1, 0, 0))
        assert result["pattern"] == r"\d+\.\d+"

    def test_large_data_migration(self):
        migration = FieldAddedMigration(
            "index", 0,
            SchemaVersion(1, 0, 0), SchemaVersion(1, 1, 0),
        )
        large_data = {"items": list(range(10000))}
        result = migration.migrate(large_data, SchemaVersion(1, 0, 0))
        assert len(result["items"]) == 10000
        assert result["index"] == 0

    def test_null_values_handled(self):
        migration = FieldTypeChangedMigration(
            "value",
            lambda x: x if x is not None else 0,
            SchemaVersion(1, 0, 0),
            SchemaVersion(2, 0, 0),
        )
        result = migration.migrate({"value": None}, SchemaVersion(1, 0, 0))
        assert result["value"] == 0

    def test_concurrent_registry_access(self):
        """Test thread safety of registry."""
        import threading

        registry = MigrationRegistry.get_instance()
        errors = []

        def register_migration(i):
            try:
                m = FieldAddedMigration(
                    f"field_{i}", i,
                    SchemaVersion(1, 0, i), SchemaVersion(1, 0, i + 1),
                )
                registry.register(m)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_migration, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
