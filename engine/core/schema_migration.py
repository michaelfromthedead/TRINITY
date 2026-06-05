"""T-CC-3.5: Schema migration for structural changes (Level 5 hot-reload).

Provides a robust schema migration system that:
- Detects schema changes via introspection
- Generates migration scaffolds for common changes
- Chains migrations for version-to-version upgrades
- Validates migrations before applying to real data
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, is_dataclass, MISSING
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from .serialization import SchemaInfo, SchemaVersion, SerializationError


T = TypeVar('T')


class ChangeType(Enum):
    """Types of schema changes that can occur."""
    FIELD_ADDED = auto()
    FIELD_REMOVED = auto()
    FIELD_RENAMED = auto()
    FIELD_TYPE_CHANGED = auto()
    FIELD_DEFAULT_CHANGED = auto()
    FIELD_OPTIONAL_CHANGED = auto()
    NESTED_SCHEMA_CHANGED = auto()


@dataclass(frozen=True)
class SchemaChange:
    """Represents a single change between schema versions."""
    change_type: ChangeType
    field_name: str
    old_value: Any = None
    new_value: Any = None
    old_type: Optional[str] = None
    new_type: Optional[str] = None
    suggested_migration: Optional[str] = None

    def __str__(self) -> str:
        if self.change_type == ChangeType.FIELD_ADDED:
            return f"Field '{self.field_name}' added (type: {self.new_type}, default: {self.new_value})"
        elif self.change_type == ChangeType.FIELD_REMOVED:
            return f"Field '{self.field_name}' removed (was type: {self.old_type})"
        elif self.change_type == ChangeType.FIELD_RENAMED:
            return f"Field renamed: '{self.old_value}' -> '{self.new_value}'"
        elif self.change_type == ChangeType.FIELD_TYPE_CHANGED:
            return f"Field '{self.field_name}' type changed: {self.old_type} -> {self.new_type}"
        elif self.change_type == ChangeType.FIELD_DEFAULT_CHANGED:
            return f"Field '{self.field_name}' default changed: {self.old_value} -> {self.new_value}"
        elif self.change_type == ChangeType.FIELD_OPTIONAL_CHANGED:
            return f"Field '{self.field_name}' optional status changed: {self.old_value} -> {self.new_value}"
        elif self.change_type == ChangeType.NESTED_SCHEMA_CHANGED:
            return f"Field '{self.field_name}' has nested schema changes"
        return f"Unknown change on '{self.field_name}'"


@dataclass
class SchemaChangeSet:
    """A collection of changes between two schema versions."""
    old_version: SchemaVersion
    new_version: SchemaVersion
    changes: List[SchemaChange] = field(default_factory=list)
    breaking: bool = False
    auto_migratable: bool = True

    def add_change(self, change: SchemaChange) -> None:
        """Add a change to the set."""
        self.changes.append(change)
        # Determine if this is a breaking change
        if change.change_type in (ChangeType.FIELD_REMOVED, ChangeType.FIELD_TYPE_CHANGED):
            self.breaking = True
        # Type changes without conversion function are not auto-migratable
        if change.change_type == ChangeType.FIELD_TYPE_CHANGED and not change.suggested_migration:
            self.auto_migratable = False

    def is_empty(self) -> bool:
        """Check if there are no changes."""
        return len(self.changes) == 0

    def get_changes_by_type(self, change_type: ChangeType) -> List[SchemaChange]:
        """Get all changes of a specific type."""
        return [c for c in self.changes if c.change_type == change_type]

    def summary(self) -> str:
        """Get a human-readable summary of changes."""
        if self.is_empty():
            return "No schema changes detected"

        lines = [f"Schema changes from {self.old_version} to {self.new_version}:"]
        for change in self.changes:
            lines.append(f"  - {change}")

        if self.breaking:
            lines.append("WARNING: Contains breaking changes")
        if not self.auto_migratable:
            lines.append("WARNING: Manual migration required")

        return "\n".join(lines)


@dataclass
class MigrationResult:
    """Result of a migration operation."""
    success: bool
    data: Dict[str, Any]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    fields_added: List[str] = field(default_factory=list)
    fields_removed: List[str] = field(default_factory=list)
    fields_converted: List[str] = field(default_factory=list)

    @classmethod
    def succeeded(cls, data: Dict[str, Any]) -> "MigrationResult":
        """Create a successful result."""
        return cls(success=True, data=data)

    @classmethod
    def failed(cls, error: str, data: Optional[Dict[str, Any]] = None) -> "MigrationResult":
        """Create a failed result."""
        return cls(success=False, data=data or {}, errors=[error])

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)


class MigrationValidationResult:
    """Result of migration validation."""

    def __init__(self):
        self.valid = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.test_results: List[Tuple[str, bool, str]] = []

    def add_error(self, error: str) -> None:
        """Add a validation error."""
        self.valid = False
        self.errors.append(error)

    def add_warning(self, warning: str) -> None:
        """Add a validation warning."""
        self.warnings.append(warning)

    def add_test_result(self, test_name: str, passed: bool, message: str = "") -> None:
        """Add a test result."""
        self.test_results.append((test_name, passed, message))
        if not passed:
            self.valid = False

    def __bool__(self) -> bool:
        return self.valid


class SchemaMigration(ABC):
    """Base class for schema migrations.

    Subclass this to implement version-to-version migrations.
    """

    # Class-level version information
    from_version: ClassVar[SchemaVersion]
    to_version: ClassVar[SchemaVersion]
    description: ClassVar[str] = ""

    @abstractmethod
    def migrate(self, old_data: Dict[str, Any], old_version: SchemaVersion) -> Dict[str, Any]:
        """Migrate data from old version to new version.

        Args:
            old_data: The data in the old schema format
            old_version: The version of the old data

        Returns:
            The data in the new schema format
        """
        pass

    def validate_input(self, data: Dict[str, Any]) -> MigrationValidationResult:
        """Validate input data before migration.

        Override this to add custom input validation.
        """
        result = MigrationValidationResult()
        if not isinstance(data, dict):
            result.add_error("Input data must be a dictionary")
        return result

    def validate_output(self, data: Dict[str, Any]) -> MigrationValidationResult:
        """Validate output data after migration.

        Override this to add custom output validation.
        """
        result = MigrationValidationResult()
        if not isinstance(data, dict):
            result.add_error("Output data must be a dictionary")
        return result

    def rollback(self, new_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Rollback a migration (optional).

        Override this to support bidirectional migrations.
        Returns None if rollback is not supported.
        """
        return None

    def supports_rollback(self) -> bool:
        """Check if this migration supports rollback."""
        return self.rollback.__func__ is not SchemaMigration.rollback

    def get_test_cases(self) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Get test cases for validation.

        Returns list of (input, expected_output) tuples.
        Override to provide test data.
        """
        return []


class FieldAddedMigration(SchemaMigration):
    """Migration for adding a new field with a default value."""

    def __init__(
        self,
        field_name: str,
        default_value: Any,
        from_ver: SchemaVersion,
        to_ver: SchemaVersion,
    ):
        self.field_name = field_name
        self.default_value = default_value
        self._from_version = from_ver
        self._to_version = to_ver

    @property
    def from_version(self) -> SchemaVersion:
        return self._from_version

    @property
    def to_version(self) -> SchemaVersion:
        return self._to_version

    def migrate(self, old_data: Dict[str, Any], old_version: SchemaVersion) -> Dict[str, Any]:
        result = copy.deepcopy(old_data)
        if self.field_name not in result:
            result[self.field_name] = copy.deepcopy(self.default_value)
        return result

    def rollback(self, new_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        result = copy.deepcopy(new_data)
        result.pop(self.field_name, None)
        return result


class FieldRemovedMigration(SchemaMigration):
    """Migration for removing a field."""

    def __init__(
        self,
        field_name: str,
        from_ver: SchemaVersion,
        to_ver: SchemaVersion,
        preserve_in_metadata: bool = False,
    ):
        self.field_name = field_name
        self._from_version = from_ver
        self._to_version = to_ver
        self.preserve_in_metadata = preserve_in_metadata

    @property
    def from_version(self) -> SchemaVersion:
        return self._from_version

    @property
    def to_version(self) -> SchemaVersion:
        return self._to_version

    def migrate(self, old_data: Dict[str, Any], old_version: SchemaVersion) -> Dict[str, Any]:
        result = copy.deepcopy(old_data)
        if self.field_name in result:
            old_value = result.pop(self.field_name)
            if self.preserve_in_metadata:
                if "__removed_fields__" not in result:
                    result["__removed_fields__"] = {}
                result["__removed_fields__"][self.field_name] = old_value
        return result

    def rollback(self, new_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        result = copy.deepcopy(new_data)
        # Try to restore from metadata
        removed = result.pop("__removed_fields__", {})
        if self.field_name in removed:
            result[self.field_name] = removed[self.field_name]
        return result


class FieldRenamedMigration(SchemaMigration):
    """Migration for renaming a field."""

    def __init__(
        self,
        old_name: str,
        new_name: str,
        from_ver: SchemaVersion,
        to_ver: SchemaVersion,
    ):
        self.old_name = old_name
        self.new_name = new_name
        self._from_version = from_ver
        self._to_version = to_ver

    @property
    def from_version(self) -> SchemaVersion:
        return self._from_version

    @property
    def to_version(self) -> SchemaVersion:
        return self._to_version

    def migrate(self, old_data: Dict[str, Any], old_version: SchemaVersion) -> Dict[str, Any]:
        result = copy.deepcopy(old_data)
        if self.old_name in result:
            result[self.new_name] = result.pop(self.old_name)
        return result

    def rollback(self, new_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        result = copy.deepcopy(new_data)
        if self.new_name in result:
            result[self.old_name] = result.pop(self.new_name)
        return result


class FieldTypeChangedMigration(SchemaMigration):
    """Migration for changing a field's type."""

    def __init__(
        self,
        field_name: str,
        converter: Callable[[Any], Any],
        from_ver: SchemaVersion,
        to_ver: SchemaVersion,
        reverse_converter: Optional[Callable[[Any], Any]] = None,
    ):
        self.field_name = field_name
        self.converter = converter
        self.reverse_converter = reverse_converter
        self._from_version = from_ver
        self._to_version = to_ver

    @property
    def from_version(self) -> SchemaVersion:
        return self._from_version

    @property
    def to_version(self) -> SchemaVersion:
        return self._to_version

    def migrate(self, old_data: Dict[str, Any], old_version: SchemaVersion) -> Dict[str, Any]:
        result = copy.deepcopy(old_data)
        if self.field_name in result:
            result[self.field_name] = self.converter(result[self.field_name])
        return result

    def rollback(self, new_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self.reverse_converter is None:
            return None
        result = copy.deepcopy(new_data)
        if self.field_name in result:
            result[self.field_name] = self.reverse_converter(result[self.field_name])
        return result


class CompositeMigration(SchemaMigration):
    """Migration that combines multiple migrations."""

    def __init__(
        self,
        migrations: List[SchemaMigration],
        from_ver: SchemaVersion,
        to_ver: SchemaVersion,
    ):
        self.migrations = migrations
        self._from_version = from_ver
        self._to_version = to_ver

    @property
    def from_version(self) -> SchemaVersion:
        return self._from_version

    @property
    def to_version(self) -> SchemaVersion:
        return self._to_version

    def migrate(self, old_data: Dict[str, Any], old_version: SchemaVersion) -> Dict[str, Any]:
        result = copy.deepcopy(old_data)
        for migration in self.migrations:
            result = migration.migrate(result, old_version)
        return result

    def rollback(self, new_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        result = copy.deepcopy(new_data)
        # Apply rollbacks in reverse order
        for migration in reversed(self.migrations):
            rolled = migration.rollback(result)
            if rolled is None:
                return None
            result = rolled
        return result


@dataclass
class MigrationHistoryEntry:
    """An entry in the migration history."""
    timestamp: datetime
    from_version: SchemaVersion
    to_version: SchemaVersion
    migration_name: str
    success: bool
    duration_ms: float
    errors: List[str] = field(default_factory=list)
    data_hash_before: str = ""
    data_hash_after: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "from_version": str(self.from_version),
            "to_version": str(self.to_version),
            "migration_name": self.migration_name,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "errors": self.errors,
            "data_hash_before": self.data_hash_before,
            "data_hash_after": self.data_hash_after,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MigrationHistoryEntry":
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            from_version=SchemaVersion.from_string(data["from_version"]),
            to_version=SchemaVersion.from_string(data["to_version"]),
            migration_name=data["migration_name"],
            success=data["success"],
            duration_ms=data["duration_ms"],
            errors=data.get("errors", []),
            data_hash_before=data.get("data_hash_before", ""),
            data_hash_after=data.get("data_hash_after", ""),
        )


class MigrationRegistry:
    """Registry for schema migrations.

    Stores migrations and chains them for multi-version upgrades.
    """

    _instance: Optional["MigrationRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "MigrationRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._migrations: Dict[Tuple[str, str], SchemaMigration] = {}
                cls._instance._history: List[MigrationHistoryEntry] = []
                cls._instance._schema_to_migrations: Dict[str, List[SchemaMigration]] = {}
            return cls._instance

    @classmethod
    def get_instance(cls) -> "MigrationRegistry":
        """Get the singleton instance."""
        return cls()

    def register(
        self,
        migration: SchemaMigration,
        schema_name: Optional[str] = None,
    ) -> None:
        """Register a migration."""
        key = (str(migration.from_version), str(migration.to_version))
        self._migrations[key] = migration

        if schema_name:
            if schema_name not in self._schema_to_migrations:
                self._schema_to_migrations[schema_name] = []
            self._schema_to_migrations[schema_name].append(migration)

    def unregister(self, from_version: SchemaVersion, to_version: SchemaVersion) -> None:
        """Unregister a migration."""
        key = (str(from_version), str(to_version))
        self._migrations.pop(key, None)

    def get_migration(
        self,
        from_version: SchemaVersion,
        to_version: SchemaVersion,
    ) -> Optional[SchemaMigration]:
        """Get a direct migration between two versions."""
        key = (str(from_version), str(to_version))
        return self._migrations.get(key)

    def find_migration_path(
        self,
        from_version: SchemaVersion,
        to_version: SchemaVersion,
    ) -> Optional[List[SchemaMigration]]:
        """Find a chain of migrations from one version to another.

        Uses BFS to find the shortest path.
        """
        if from_version == to_version:
            return []

        # Build graph of version transitions
        from_str = str(from_version)
        to_str = str(to_version)

        # BFS
        queue: List[Tuple[str, List[SchemaMigration]]] = [(from_str, [])]
        visited: Set[str] = {from_str}

        while queue:
            current, path = queue.pop(0)

            # Find all migrations from current version
            for (fv, tv), migration in self._migrations.items():
                if fv == current and tv not in visited:
                    new_path = path + [migration]

                    if tv == to_str:
                        return new_path

                    visited.add(tv)
                    queue.append((tv, new_path))

        return None

    def migrate(
        self,
        data: Dict[str, Any],
        from_version: SchemaVersion,
        to_version: SchemaVersion,
        validate: bool = True,
        record_history: bool = True,
    ) -> MigrationResult:
        """Migrate data from one version to another.

        Chains migrations if no direct path exists.
        """
        import time
        start_time = time.time()

        # Find migration path
        path = self.find_migration_path(from_version, to_version)
        if path is None:
            return MigrationResult.failed(
                f"No migration path from {from_version} to {to_version}"
            )

        if not path:
            # Same version, no migration needed
            return MigrationResult.succeeded(copy.deepcopy(data))

        # Compute hash before
        data_hash_before = self._compute_hash(data)

        # Apply migrations in sequence
        result = copy.deepcopy(data)
        current_version = from_version

        for migration in path:
            # Validate input
            if validate:
                validation = migration.validate_input(result)
                if not validation:
                    return MigrationResult.failed(
                        f"Input validation failed for {migration.__class__.__name__}: "
                        f"{validation.errors}"
                    )

            # Apply migration
            try:
                result = migration.migrate(result, current_version)
            except Exception as e:
                return MigrationResult.failed(
                    f"Migration {migration.__class__.__name__} failed: {e}"
                )

            # Validate output
            if validate:
                validation = migration.validate_output(result)
                if not validation:
                    return MigrationResult.failed(
                        f"Output validation failed for {migration.__class__.__name__}: "
                        f"{validation.errors}"
                    )

            current_version = migration.to_version

        # Compute hash after
        data_hash_after = self._compute_hash(result)

        duration_ms = (time.time() - start_time) * 1000

        # Record history
        if record_history:
            entry = MigrationHistoryEntry(
                timestamp=datetime.now(),
                from_version=from_version,
                to_version=to_version,
                migration_name=" -> ".join(m.__class__.__name__ for m in path),
                success=True,
                duration_ms=duration_ms,
                data_hash_before=data_hash_before,
                data_hash_after=data_hash_after,
            )
            self._history.append(entry)

        return MigrationResult.succeeded(result)

    def validate_migration(
        self,
        migration: SchemaMigration,
        test_data: Optional[List[Tuple[Dict[str, Any], Dict[str, Any]]]] = None,
    ) -> MigrationValidationResult:
        """Validate a migration before applying to real data.

        Args:
            migration: The migration to validate
            test_data: Optional list of (input, expected_output) test cases
        """
        result = MigrationValidationResult()

        # Check version consistency
        if migration.from_version >= migration.to_version:
            result.add_error(
                f"from_version ({migration.from_version}) must be less than "
                f"to_version ({migration.to_version})"
            )

        # Get test cases from migration or use provided ones
        test_cases = test_data or migration.get_test_cases()

        if not test_cases:
            result.add_warning("No test cases provided for validation")
        else:
            for i, (input_data, expected_output) in enumerate(test_cases):
                test_name = f"test_case_{i}"
                try:
                    actual_output = migration.migrate(input_data, migration.from_version)

                    if actual_output == expected_output:
                        result.add_test_result(test_name, True, "Passed")
                    else:
                        result.add_test_result(
                            test_name,
                            False,
                            f"Output mismatch: expected {expected_output}, got {actual_output}",
                        )
                except Exception as e:
                    result.add_test_result(test_name, False, f"Exception: {e}")

        # Test rollback if supported
        if migration.supports_rollback() and test_cases:
            for i, (input_data, expected_output) in enumerate(test_cases):
                test_name = f"rollback_test_{i}"
                try:
                    migrated = migration.migrate(input_data, migration.from_version)
                    rolled_back = migration.rollback(migrated)

                    if rolled_back == input_data:
                        result.add_test_result(test_name, True, "Rollback passed")
                    else:
                        result.add_test_result(
                            test_name,
                            False,
                            f"Rollback mismatch: expected {input_data}, got {rolled_back}",
                        )
                except Exception as e:
                    result.add_test_result(test_name, False, f"Rollback exception: {e}")

        return result

    def get_history(
        self,
        schema_name: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[MigrationHistoryEntry]:
        """Get migration history."""
        history = self._history
        if limit:
            history = history[-limit:]
        return list(history)

    def clear_history(self) -> None:
        """Clear migration history."""
        self._history.clear()

    def list_migrations(
        self,
        schema_name: Optional[str] = None,
    ) -> List[Tuple[SchemaVersion, SchemaVersion, str]]:
        """List all registered migrations."""
        if schema_name and schema_name in self._schema_to_migrations:
            return [
                (m.from_version, m.to_version, m.__class__.__name__)
                for m in self._schema_to_migrations[schema_name]
            ]

        return [
            (m.from_version, m.to_version, m.__class__.__name__)
            for m in self._migrations.values()
        ]

    def clear(self) -> None:
        """Clear all registrations."""
        self._migrations.clear()
        self._schema_to_migrations.clear()

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute a hash of the data for tracking."""
        try:
            serialized = json.dumps(data, sort_keys=True, default=str)
            return hashlib.md5(serialized.encode()).hexdigest()[:8]
        except Exception:
            return ""


@dataclass
class FieldInfo:
    """Information about a schema field."""
    name: str
    type_name: str
    default_value: Any = None
    has_default: bool = False
    is_optional: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FieldInfo):
            return False
        return (
            self.name == other.name
            and self.type_name == other.type_name
            and self.is_optional == other.is_optional
        )


@dataclass
class ExtractedSchema:
    """Schema extracted from a class for comparison."""
    class_name: str
    version: SchemaVersion
    fields: Dict[str, FieldInfo]
    hash: str

    @classmethod
    def from_class(cls, klass: type) -> "ExtractedSchema":
        """Extract schema information from a class."""
        # Get version if available
        version = getattr(klass, "__schema_version__", SchemaVersion(1, 0, 0))

        # Extract fields
        field_infos: Dict[str, FieldInfo] = {}

        if is_dataclass(klass):
            try:
                hints = get_type_hints(klass)
            except Exception:
                hints = {}

            for f in fields(klass):
                type_hint = hints.get(f.name)
                type_name = _type_to_str(type_hint) if type_hint else "Any"

                has_default = f.default is not MISSING or f.default_factory is not MISSING
                default_value = f.default if f.default is not MISSING else None

                # Check if optional
                is_optional = False
                if type_hint:
                    origin = get_origin(type_hint)
                    args = get_args(type_hint)
                    is_optional = origin is Union and type(None) in args

                field_infos[f.name] = FieldInfo(
                    name=f.name,
                    type_name=type_name,
                    default_value=default_value,
                    has_default=has_default,
                    is_optional=is_optional,
                    metadata=dict(f.metadata),
                )
        else:
            # Handle non-dataclass via annotations
            try:
                hints = get_type_hints(klass)
            except Exception:
                hints = getattr(klass, "__annotations__", {})

            for name, type_hint in hints.items():
                if name.startswith("_"):
                    continue

                type_name = _type_to_str(type_hint)
                has_default = hasattr(klass, name)
                default_value = getattr(klass, name, None) if has_default else None

                origin = get_origin(type_hint)
                args = get_args(type_hint)
                is_optional = origin is Union and type(None) in args

                field_infos[name] = FieldInfo(
                    name=name,
                    type_name=type_name,
                    default_value=default_value,
                    has_default=has_default,
                    is_optional=is_optional,
                )

        # Compute hash
        hash_data = f"{klass.__name__}:{version}:" + ",".join(
            f"{k}:{v.type_name}" for k, v in sorted(field_infos.items())
        )
        schema_hash = hashlib.md5(hash_data.encode()).hexdigest()[:8]

        return cls(
            class_name=klass.__name__,
            version=version,
            fields=field_infos,
            hash=schema_hash,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractedSchema":
        """Extract schema from a dictionary (e.g., stored schema info)."""
        schema_info = data.get("__schema__", {})

        fields_dict: Dict[str, FieldInfo] = {}
        for key, value in data.items():
            if key == "__schema__":
                continue

            type_name = _infer_type_from_value(value)
            fields_dict[key] = FieldInfo(
                name=key,
                type_name=type_name,
                default_value=None,
                has_default=False,
                is_optional=value is None,
            )

        version = SchemaVersion(1, 0, 0)
        if "version" in schema_info:
            version = SchemaVersion.from_string(schema_info["version"])

        class_name = schema_info.get("name", "Unknown")

        # Compute hash
        hash_data = f"{class_name}:{version}:" + ",".join(
            f"{k}:{v.type_name}" for k, v in sorted(fields_dict.items())
        )
        schema_hash = hashlib.md5(hash_data.encode()).hexdigest()[:8]

        return cls(
            class_name=class_name,
            version=version,
            fields=fields_dict,
            hash=schema_hash,
        )


def _type_to_str(type_hint: Any) -> str:
    """Convert a type hint to a string representation."""
    if type_hint is None:
        return "None"

    origin = get_origin(type_hint)
    args = get_args(type_hint)

    if origin is None:
        if hasattr(type_hint, "__name__"):
            return type_hint.__name__
        return str(type_hint)

    origin_name = getattr(origin, "__name__", str(origin))

    if args:
        args_str = ", ".join(_type_to_str(a) for a in args)
        return f"{origin_name}[{args_str}]"

    return origin_name


def _infer_type_from_value(value: Any) -> str:
    """Infer type name from a value."""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        if value:
            inner = _infer_type_from_value(value[0])
            return f"List[{inner}]"
        return "List[Any]"
    if isinstance(value, dict):
        if value:
            k_type = _infer_type_from_value(next(iter(value.keys())))
            v_type = _infer_type_from_value(next(iter(value.values())))
            return f"Dict[{k_type}, {v_type}]"
        return "Dict[str, Any]"
    if isinstance(value, set):
        if value:
            inner = _infer_type_from_value(next(iter(value)))
            return f"Set[{inner}]"
        return "Set[Any]"
    if isinstance(value, tuple):
        if value:
            return f"Tuple[{', '.join(_infer_type_from_value(v) for v in value)}]"
        return "Tuple[()]"

    return type(value).__name__


def detect_schema_changes(
    old_schema: Union[type, ExtractedSchema],
    new_schema: Union[type, ExtractedSchema],
    detect_renames: bool = True,
    rename_threshold: float = 0.8,
) -> SchemaChangeSet:
    """Detect structural differences between two schemas.

    Args:
        old_schema: The old schema (class or ExtractedSchema)
        new_schema: The new schema (class or ExtractedSchema)
        detect_renames: Whether to try to detect field renames
        rename_threshold: Similarity threshold for rename detection (0-1)

    Returns:
        SchemaChangeSet with all detected changes
    """
    # Convert to ExtractedSchema if needed
    if isinstance(old_schema, type):
        old_schema = ExtractedSchema.from_class(old_schema)
    if isinstance(new_schema, type):
        new_schema = ExtractedSchema.from_class(new_schema)

    change_set = SchemaChangeSet(
        old_version=old_schema.version,
        new_version=new_schema.version,
    )

    old_fields = old_schema.fields
    new_fields = new_schema.fields

    old_names = set(old_fields.keys())
    new_names = set(new_fields.keys())

    # Fields that exist in both
    common = old_names & new_names
    # Fields only in old (potentially removed or renamed)
    removed = old_names - new_names
    # Fields only in new (potentially added or renamed)
    added = new_names - old_names

    # Try to detect renames
    renames: Dict[str, str] = {}  # old_name -> new_name

    if detect_renames and removed and added:
        for old_name in list(removed):
            old_field = old_fields[old_name]

            for new_name in list(added):
                new_field = new_fields[new_name]

                # Check if types match
                if old_field.type_name == new_field.type_name:
                    # Calculate name similarity
                    similarity = _string_similarity(old_name, new_name)
                    if similarity >= rename_threshold:
                        renames[old_name] = new_name
                        removed.discard(old_name)
                        added.discard(new_name)
                        break

    # Record renames
    for old_name, new_name in renames.items():
        change_set.add_change(SchemaChange(
            change_type=ChangeType.FIELD_RENAMED,
            field_name=old_name,
            old_value=old_name,
            new_value=new_name,
            suggested_migration=f"data['{new_name}'] = data.pop('{old_name}')",
        ))

    # Record removed fields
    for name in removed:
        old_field = old_fields[name]
        change_set.add_change(SchemaChange(
            change_type=ChangeType.FIELD_REMOVED,
            field_name=name,
            old_type=old_field.type_name,
            suggested_migration=f"data.pop('{name}', None)",
        ))

    # Record added fields
    for name in added:
        new_field = new_fields[name]
        default_val = new_field.default_value if new_field.has_default else None
        change_set.add_change(SchemaChange(
            change_type=ChangeType.FIELD_ADDED,
            field_name=name,
            new_type=new_field.type_name,
            new_value=default_val,
            suggested_migration=f"data['{name}'] = {repr(default_val)}",
        ))

    # Check common fields for type/default/optional changes
    for name in common:
        old_field = old_fields[name]
        new_field = new_fields[name]

        # Type change
        if old_field.type_name != new_field.type_name:
            change_set.add_change(SchemaChange(
                change_type=ChangeType.FIELD_TYPE_CHANGED,
                field_name=name,
                old_type=old_field.type_name,
                new_type=new_field.type_name,
            ))

        # Default change
        if old_field.has_default != new_field.has_default or (
            old_field.has_default
            and new_field.has_default
            and old_field.default_value != new_field.default_value
        ):
            change_set.add_change(SchemaChange(
                change_type=ChangeType.FIELD_DEFAULT_CHANGED,
                field_name=name,
                old_value=old_field.default_value if old_field.has_default else "<no default>",
                new_value=new_field.default_value if new_field.has_default else "<no default>",
            ))

        # Optional change
        if old_field.is_optional != new_field.is_optional:
            change_set.add_change(SchemaChange(
                change_type=ChangeType.FIELD_OPTIONAL_CHANGED,
                field_name=name,
                old_value=old_field.is_optional,
                new_value=new_field.is_optional,
            ))

    return change_set


def _string_similarity(s1: str, s2: str) -> float:
    """Calculate similarity between two strings using Levenshtein ratio."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    # Simple Levenshtein distance
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    max_len = max(m, n)
    return 1.0 - (dp[m][n] / max_len)


def generate_migration(
    change_set: SchemaChangeSet,
    class_name: str = "GeneratedMigration",
) -> str:
    """Generate migration scaffold code for a change set.

    Args:
        change_set: The detected changes
        class_name: Name for the generated migration class

    Returns:
        Python code string for the migration class
    """
    lines = [
        f"class {class_name}(SchemaMigration):",
        f'    """Auto-generated migration from {change_set.old_version} to {change_set.new_version}."""',
        f"",
        f"    from_version = SchemaVersion.from_string('{change_set.old_version}')",
        f"    to_version = SchemaVersion.from_string('{change_set.new_version}')",
        f"",
        f"    def migrate(self, old_data: Dict[str, Any], old_version: SchemaVersion) -> Dict[str, Any]:",
        f"        result = copy.deepcopy(old_data)",
        f"",
    ]

    # Generate migration code for each change
    for change in change_set.changes:
        if change.change_type == ChangeType.FIELD_ADDED:
            default = repr(change.new_value)
            lines.append(f"        # Field added: {change.field_name}")
            lines.append(f"        if '{change.field_name}' not in result:")
            lines.append(f"            result['{change.field_name}'] = {default}")
            lines.append("")

        elif change.change_type == ChangeType.FIELD_REMOVED:
            lines.append(f"        # Field removed: {change.field_name}")
            lines.append(f"        result.pop('{change.field_name}', None)")
            lines.append("")

        elif change.change_type == ChangeType.FIELD_RENAMED:
            old_name = change.old_value
            new_name = change.new_value
            lines.append(f"        # Field renamed: {old_name} -> {new_name}")
            lines.append(f"        if '{old_name}' in result:")
            lines.append(f"            result['{new_name}'] = result.pop('{old_name}')")
            lines.append("")

        elif change.change_type == ChangeType.FIELD_TYPE_CHANGED:
            lines.append(f"        # Field type changed: {change.field_name}")
            lines.append(f"        # Old type: {change.old_type}, New type: {change.new_type}")
            lines.append(f"        if '{change.field_name}' in result:")
            lines.append(f"            # TODO: Implement type conversion")
            lines.append(f"            result['{change.field_name}'] = convert_{change.field_name}(result['{change.field_name}'])")
            lines.append("")

    lines.append("        return result")
    lines.append("")

    # Generate rollback if all changes are reversible
    if change_set.auto_migratable:
        lines.extend([
            "    def rollback(self, new_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:",
            "        result = copy.deepcopy(new_data)",
            "",
        ])

        # Reverse order for rollback
        for change in reversed(change_set.changes):
            if change.change_type == ChangeType.FIELD_ADDED:
                lines.append(f"        # Rollback field added: {change.field_name}")
                lines.append(f"        result.pop('{change.field_name}', None)")
                lines.append("")

            elif change.change_type == ChangeType.FIELD_REMOVED:
                lines.append(f"        # Rollback field removed: {change.field_name}")
                lines.append(f"        # WARNING: Cannot restore removed field data")
                lines.append("")

            elif change.change_type == ChangeType.FIELD_RENAMED:
                old_name = change.old_value
                new_name = change.new_value
                lines.append(f"        # Rollback field renamed: {new_name} -> {old_name}")
                lines.append(f"        if '{new_name}' in result:")
                lines.append(f"            result['{old_name}'] = result.pop('{new_name}')")
                lines.append("")

        lines.append("        return result")
        lines.append("")

    # Generate test cases template
    lines.extend([
        "    def get_test_cases(self) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:",
        "        return [",
        "            # TODO: Add test cases",
        "            # (input_data, expected_output),",
        "        ]",
    ])

    return "\n".join(lines)


def generate_migration_from_classes(
    old_class: type,
    new_class: type,
    class_name: str = "GeneratedMigration",
    detect_renames: bool = True,
) -> str:
    """Generate migration code from two class definitions.

    Args:
        old_class: The old schema class
        new_class: The new schema class
        class_name: Name for the generated migration class
        detect_renames: Whether to detect field renames

    Returns:
        Python code string for the migration class
    """
    changes = detect_schema_changes(old_class, new_class, detect_renames=detect_renames)
    return generate_migration(changes, class_name)


class MigrationBuilder:
    """Builder pattern for creating migrations programmatically."""

    def __init__(
        self,
        from_version: SchemaVersion,
        to_version: SchemaVersion,
    ):
        self.from_version = from_version
        self.to_version = to_version
        self._operations: List[SchemaMigration] = []

    def add_field(self, field_name: str, default_value: Any) -> "MigrationBuilder":
        """Add a field addition operation."""
        self._operations.append(FieldAddedMigration(
            field_name=field_name,
            default_value=default_value,
            from_ver=self.from_version,
            to_ver=self.to_version,
        ))
        return self

    def remove_field(
        self,
        field_name: str,
        preserve_in_metadata: bool = False,
    ) -> "MigrationBuilder":
        """Add a field removal operation."""
        self._operations.append(FieldRemovedMigration(
            field_name=field_name,
            from_ver=self.from_version,
            to_ver=self.to_version,
            preserve_in_metadata=preserve_in_metadata,
        ))
        return self

    def rename_field(self, old_name: str, new_name: str) -> "MigrationBuilder":
        """Add a field rename operation."""
        self._operations.append(FieldRenamedMigration(
            old_name=old_name,
            new_name=new_name,
            from_ver=self.from_version,
            to_ver=self.to_version,
        ))
        return self

    def change_field_type(
        self,
        field_name: str,
        converter: Callable[[Any], Any],
        reverse_converter: Optional[Callable[[Any], Any]] = None,
    ) -> "MigrationBuilder":
        """Add a field type change operation."""
        self._operations.append(FieldTypeChangedMigration(
            field_name=field_name,
            converter=converter,
            from_ver=self.from_version,
            to_ver=self.to_version,
            reverse_converter=reverse_converter,
        ))
        return self

    def build(self) -> SchemaMigration:
        """Build the composite migration."""
        if len(self._operations) == 1:
            return self._operations[0]

        return CompositeMigration(
            migrations=self._operations,
            from_ver=self.from_version,
            to_ver=self.to_version,
        )


def apply_migration_safely(
    data: Dict[str, Any],
    migration: SchemaMigration,
    rollback_on_failure: bool = True,
) -> MigrationResult:
    """Apply a migration with automatic rollback on failure.

    Args:
        data: The data to migrate
        migration: The migration to apply
        rollback_on_failure: Whether to attempt rollback if validation fails

    Returns:
        MigrationResult with the migrated data or error information
    """
    # Validate input
    input_validation = migration.validate_input(data)
    if not input_validation:
        return MigrationResult.failed(
            f"Input validation failed: {input_validation.errors}",
            data,
        )

    # Apply migration
    try:
        result_data = migration.migrate(data, migration.from_version)
    except Exception as e:
        return MigrationResult.failed(f"Migration failed: {e}", data)

    # Validate output
    output_validation = migration.validate_output(result_data)
    if not output_validation:
        if rollback_on_failure and migration.supports_rollback():
            rolled_back = migration.rollback(result_data)
            if rolled_back is not None:
                return MigrationResult.failed(
                    f"Output validation failed, rolled back: {output_validation.errors}",
                    rolled_back,
                )
        return MigrationResult.failed(
            f"Output validation failed: {output_validation.errors}",
            result_data,
        )

    return MigrationResult.succeeded(result_data)


def create_migration_chain(
    migrations: List[SchemaMigration],
) -> Optional[CompositeMigration]:
    """Create a chained migration from a list of migrations.

    Returns None if the migrations don't form a valid chain.
    """
    if not migrations:
        return None

    # Sort by version
    sorted_migrations = sorted(migrations, key=lambda m: m.from_version)

    # Verify chain
    for i in range(len(sorted_migrations) - 1):
        current = sorted_migrations[i]
        next_migration = sorted_migrations[i + 1]
        if current.to_version != next_migration.from_version:
            return None

    return CompositeMigration(
        migrations=sorted_migrations,
        from_ver=sorted_migrations[0].from_version,
        to_ver=sorted_migrations[-1].to_version,
    )
