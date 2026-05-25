"""
Schema Hash - Detect and analyze schema changes for safe hot-reloading.

Provides detailed schema comparison beyond simple hash matching,
identifying specific changes and whether they are breaking.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Type

from foundation import mirror, schema_hash as foundation_schema_hash, FieldInfo


class SchemaChangeType(Enum):
    """Types of schema changes."""

    # Non-breaking changes
    FIELD_ADDED_WITH_DEFAULT = auto()
    METADATA_CHANGED = auto()
    METHOD_ADDED = auto()
    METHOD_REMOVED = auto()
    METHOD_SIGNATURE_CHANGED = auto()
    DEFAULT_VALUE_CHANGED = auto()

    # Potentially breaking changes
    FIELD_TYPE_WIDENED = auto()  # e.g., int -> float

    # Breaking changes
    FIELD_REMOVED = auto()
    FIELD_ADDED_WITHOUT_DEFAULT = auto()
    FIELD_TYPE_CHANGED = auto()
    FIELD_TYPE_NARROWED = auto()  # e.g., float -> int
    CLASS_RENAMED = auto()


@dataclass(frozen=True)
class SchemaChange:
    """Describes a single change between schema versions."""

    change_type: SchemaChangeType
    field_name: Optional[str]
    description: str
    old_value: Any = None
    new_value: Any = None

    @property
    def is_breaking(self) -> bool:
        """Check if this change is breaking."""
        return self.change_type in {
            SchemaChangeType.FIELD_REMOVED,
            SchemaChangeType.FIELD_ADDED_WITHOUT_DEFAULT,
            SchemaChangeType.FIELD_TYPE_CHANGED,
            SchemaChangeType.FIELD_TYPE_NARROWED,
            SchemaChangeType.CLASS_RENAMED,
        }

    @property
    def severity(self) -> str:
        """Get severity level: 'breaking', 'warning', or 'info'."""
        if self.is_breaking:
            return "breaking"
        elif self.change_type in {
            SchemaChangeType.FIELD_TYPE_WIDENED,
            SchemaChangeType.DEFAULT_VALUE_CHANGED,
        }:
            return "warning"
        return "info"


@dataclass
class SchemaComparison:
    """Result of comparing two schema versions."""

    old_hash: str
    new_hash: str
    old_class_name: str
    new_class_name: str
    changes: List[SchemaChange] = field(default_factory=list)

    @property
    def is_compatible(self) -> bool:
        """Check if the schemas are compatible (no breaking changes)."""
        return not any(c.is_breaking for c in self.changes)

    @property
    def is_identical(self) -> bool:
        """Check if schemas are identical."""
        return self.old_hash == self.new_hash

    @property
    def breaking_changes(self) -> List[SchemaChange]:
        """Get only breaking changes."""
        return [c for c in self.changes if c.is_breaking]

    @property
    def non_breaking_changes(self) -> List[SchemaChange]:
        """Get only non-breaking changes."""
        return [c for c in self.changes if not c.is_breaking]

    def summary(self) -> str:
        """Get a human-readable summary."""
        if self.is_identical:
            return "Schemas are identical"

        breaking = len(self.breaking_changes)
        non_breaking = len(self.non_breaking_changes)

        if breaking > 0:
            return f"INCOMPATIBLE: {breaking} breaking change(s), {non_breaking} non-breaking"
        return f"Compatible: {non_breaking} non-breaking change(s)"


class SchemaHasher:
    """
    Advanced schema hashing and comparison.

    Features:
    - Detailed schema difference analysis
    - Type compatibility checking
    - Migration requirement detection
    - Versioned schema tracking
    """

    # Type compatibility map (old_type -> set of compatible new_types)
    TYPE_WIDENING: Dict[type, Set[type]] = {
        int: {float, complex},
        float: {complex},
        str: set(),
        bool: {int},
        list: set(),
        tuple: {list},
        dict: set(),
    }

    def __init__(self):
        """Initialize the schema hasher."""
        self._cache: Dict[str, Dict[str, Any]] = {}  # class_name -> schema_info

    def compute_hash(self, cls: Type) -> str:
        """
        Compute schema hash for a class.

        Uses Foundation's schema_hash for consistency.

        Args:
            cls: Class to hash.

        Returns:
            16-character hex hash string.
        """
        return foundation_schema_hash(cls)

    def get_schema_info(self, cls: Type) -> Dict[str, Any]:
        """
        Get detailed schema information for a class.

        Args:
            cls: Class to analyze.

        Returns:
            Dictionary with schema details.
        """
        m = mirror(cls)

        fields = {}
        for name, info in m.fields.items():
            fields[name] = {
                "type": info.type.__name__ if info.type else "Any",
                "type_obj": info.type,
                "has_default": info.has_default,
                "default": info.default,
                "metadata": dict(info.metadata),
            }

        methods = {}
        for name, info in m.methods.items():
            methods[name] = {
                "is_property": info.is_property,
                "signature": str(info.signature),
            }

        return {
            "class_name": cls.__name__,
            "module_name": cls.__module__,
            "full_name": f"{cls.__module__}.{cls.__name__}",
            "hash": self.compute_hash(cls),
            "fields": fields,
            "methods": methods,
        }

    def compare_schemas(
        self,
        old_cls: Type,
        new_cls: Type,
    ) -> SchemaComparison:
        """
        Compare two class schemas and identify changes.

        Args:
            old_cls: Previous class version.
            new_cls: New class version.

        Returns:
            SchemaComparison with detailed change analysis.
        """
        old_info = self.get_schema_info(old_cls)
        new_info = self.get_schema_info(new_cls)

        changes: List[SchemaChange] = []

        # Check class rename
        if old_info["class_name"] != new_info["class_name"]:
            changes.append(SchemaChange(
                change_type=SchemaChangeType.CLASS_RENAMED,
                field_name=None,
                description=f"Class renamed from {old_info['class_name']} to {new_info['class_name']}",
                old_value=old_info["class_name"],
                new_value=new_info["class_name"],
            ))

        # Compare fields
        old_fields = set(old_info["fields"].keys())
        new_fields = set(new_info["fields"].keys())

        # Removed fields
        for name in old_fields - new_fields:
            changes.append(SchemaChange(
                change_type=SchemaChangeType.FIELD_REMOVED,
                field_name=name,
                description=f"Field '{name}' was removed",
                old_value=old_info["fields"][name],
            ))

        # Added fields
        for name in new_fields - old_fields:
            field_info = new_info["fields"][name]
            if field_info["has_default"]:
                changes.append(SchemaChange(
                    change_type=SchemaChangeType.FIELD_ADDED_WITH_DEFAULT,
                    field_name=name,
                    description=f"Field '{name}' added with default value",
                    new_value=field_info,
                ))
            else:
                changes.append(SchemaChange(
                    change_type=SchemaChangeType.FIELD_ADDED_WITHOUT_DEFAULT,
                    field_name=name,
                    description=f"Field '{name}' added without default value",
                    new_value=field_info,
                ))

        # Changed fields
        for name in old_fields & new_fields:
            old_field = old_info["fields"][name]
            new_field = new_info["fields"][name]

            # Type changes
            if old_field["type"] != new_field["type"]:
                change_type = self._classify_type_change(
                    old_field["type_obj"],
                    new_field["type_obj"],
                )
                changes.append(SchemaChange(
                    change_type=change_type,
                    field_name=name,
                    description=f"Field '{name}' type changed from {old_field['type']} to {new_field['type']}",
                    old_value=old_field["type"],
                    new_value=new_field["type"],
                ))

            # Default value changes
            if old_field["default"] != new_field["default"]:
                changes.append(SchemaChange(
                    change_type=SchemaChangeType.DEFAULT_VALUE_CHANGED,
                    field_name=name,
                    description=f"Field '{name}' default changed",
                    old_value=old_field["default"],
                    new_value=new_field["default"],
                ))

            # Metadata changes
            if old_field["metadata"] != new_field["metadata"]:
                changes.append(SchemaChange(
                    change_type=SchemaChangeType.METADATA_CHANGED,
                    field_name=name,
                    description=f"Field '{name}' metadata changed",
                    old_value=old_field["metadata"],
                    new_value=new_field["metadata"],
                ))

        # Compare methods (non-breaking)
        old_methods = set(old_info["methods"].keys())
        new_methods = set(new_info["methods"].keys())

        for name in old_methods - new_methods:
            changes.append(SchemaChange(
                change_type=SchemaChangeType.METHOD_REMOVED,
                field_name=name,
                description=f"Method '{name}' was removed",
            ))

        for name in new_methods - old_methods:
            changes.append(SchemaChange(
                change_type=SchemaChangeType.METHOD_ADDED,
                field_name=name,
                description=f"Method '{name}' was added",
            ))

        for name in old_methods & new_methods:
            old_sig = old_info["methods"][name]["signature"]
            new_sig = new_info["methods"][name]["signature"]
            if old_sig != new_sig:
                changes.append(SchemaChange(
                    change_type=SchemaChangeType.METHOD_SIGNATURE_CHANGED,
                    field_name=name,
                    description=f"Method '{name}' signature changed",
                    old_value=old_sig,
                    new_value=new_sig,
                ))

        return SchemaComparison(
            old_hash=old_info["hash"],
            new_hash=new_info["hash"],
            old_class_name=old_info["full_name"],
            new_class_name=new_info["full_name"],
            changes=changes,
        )

    def _classify_type_change(
        self,
        old_type: Optional[type],
        new_type: Optional[type],
    ) -> SchemaChangeType:
        """Classify a type change as widening, narrowing, or incompatible."""
        if old_type is None or new_type is None:
            return SchemaChangeType.FIELD_TYPE_CHANGED

        # Check if widening (safe)
        if old_type in self.TYPE_WIDENING:
            if new_type in self.TYPE_WIDENING[old_type]:
                return SchemaChangeType.FIELD_TYPE_WIDENED

        # Check if narrowing (breaking)
        if new_type in self.TYPE_WIDENING:
            if old_type in self.TYPE_WIDENING[new_type]:
                return SchemaChangeType.FIELD_TYPE_NARROWED

        return SchemaChangeType.FIELD_TYPE_CHANGED

    def requires_migration(
        self,
        old_cls: Type,
        new_cls: Type,
    ) -> bool:
        """
        Check if a migration is required between versions.

        Args:
            old_cls: Previous class version.
            new_cls: New class version.

        Returns:
            True if migration is required.
        """
        comparison = self.compare_schemas(old_cls, new_cls)
        return not comparison.is_compatible

    def generate_migration_hints(
        self,
        comparison: SchemaComparison,
    ) -> List[str]:
        """
        Generate hints for writing a migration function.

        Args:
            comparison: Schema comparison result.

        Returns:
            List of migration hint strings.
        """
        hints = []

        for change in comparison.breaking_changes:
            if change.change_type == SchemaChangeType.FIELD_REMOVED:
                hints.append(
                    f"Handle removed field '{change.field_name}' - "
                    f"old data may have this field"
                )
            elif change.change_type == SchemaChangeType.FIELD_ADDED_WITHOUT_DEFAULT:
                hints.append(
                    f"Provide default for new field '{change.field_name}' - "
                    f"old data won't have this field"
                )
            elif change.change_type == SchemaChangeType.FIELD_TYPE_CHANGED:
                hints.append(
                    f"Convert field '{change.field_name}' from "
                    f"{change.old_value} to {change.new_value}"
                )
            elif change.change_type == SchemaChangeType.FIELD_TYPE_NARROWED:
                hints.append(
                    f"Safely narrow field '{change.field_name}' from "
                    f"{change.old_value} to {change.new_value} (may lose precision)"
                )

        return hints


__all__ = [
    "SchemaChangeType",
    "SchemaChange",
    "SchemaComparison",
    "SchemaHasher",
]
