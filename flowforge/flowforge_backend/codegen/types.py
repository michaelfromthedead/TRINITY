"""Type definitions for code generation in FlowForge.

This module defines dataclasses used for code generation and validation,
including validation results, errors, and generation output structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Severity(str, Enum):
    """Severity level for validation messages."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationError:
    """Represents a single validation error or warning.

    Attributes:
        line: Line number where the error occurred (1-indexed).
        column: Column number where the error occurred (0-indexed).
        message: Human-readable error message.
        severity: Severity level (error, warning, info).
        code: Optional error code for programmatic handling.
        end_line: Optional end line for multi-line errors.
        end_column: Optional end column.
    """
    line: int
    column: int
    message: str
    severity: Severity = Severity.ERROR
    code: Optional[str] = None
    end_line: Optional[int] = None
    end_column: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result: dict[str, Any] = {
            "line": self.line,
            "column": self.column,
            "message": self.message,
            "severity": self.severity.value,
        }
        if self.code is not None:
            result["code"] = self.code
        if self.end_line is not None:
            result["end_line"] = self.end_line
        if self.end_column is not None:
            result["end_column"] = self.end_column
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidationError":
        """Create from dictionary."""
        return cls(
            line=data["line"],
            column=data["column"],
            message=data["message"],
            severity=Severity(data.get("severity", "error")),
            code=data.get("code"),
            end_line=data.get("end_line"),
            end_column=data.get("end_column"),
        )

    @classmethod
    def syntax_error(
        cls,
        message: str,
        line: int,
        column: int = 0,
        end_line: Optional[int] = None,
        end_column: Optional[int] = None,
    ) -> "ValidationError":
        """Create a syntax error."""
        return cls(
            line=line,
            column=column,
            message=message,
            severity=Severity.ERROR,
            code="E001",
            end_line=end_line,
            end_column=end_column,
        )

    @classmethod
    def undefined_name(cls, name: str, line: int, column: int = 0) -> "ValidationError":
        """Create an undefined name error."""
        return cls(
            line=line,
            column=column,
            message=f"Undefined name '{name}'",
            severity=Severity.ERROR,
            code="E002",
        )

    @classmethod
    def unused_import(cls, name: str, line: int, column: int = 0) -> "ValidationError":
        """Create an unused import warning."""
        return cls(
            line=line,
            column=column,
            message=f"Unused import '{name}'",
            severity=Severity.WARNING,
            code="W001",
        )

    @classmethod
    def shadowed_builtin(cls, name: str, line: int, column: int = 0) -> "ValidationError":
        """Create a shadowed builtin warning."""
        return cls(
            line=line,
            column=column,
            message=f"Variable '{name}' shadows a Python builtin",
            severity=Severity.WARNING,
            code="W002",
        )


@dataclass
class ValidationResult:
    """Result of validating Python source code.

    Attributes:
        success: True if the code is valid (no errors, warnings are ok).
        errors: List of validation errors (severity=error).
        warnings: List of validation warnings (severity=warning or info).
        source_hash: Optional hash of the validated source for caching.
    """
    success: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)
    source_hash: Optional[str] = None

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0

    @property
    def all_issues(self) -> list[ValidationError]:
        """Get all errors and warnings combined."""
        return self.errors + self.warnings

    @property
    def error_count(self) -> int:
        """Get the number of errors."""
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Get the number of warnings."""
        return len(self.warnings)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result: dict[str, Any] = {
            "success": self.success,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
        }
        if self.source_hash is not None:
            result["source_hash"] = self.source_hash
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidationResult":
        """Create from dictionary."""
        return cls(
            success=data["success"],
            errors=[ValidationError.from_dict(e) for e in data.get("errors", [])],
            warnings=[ValidationError.from_dict(w) for w in data.get("warnings", [])],
            source_hash=data.get("source_hash"),
        )

    @classmethod
    def valid(cls, source_hash: Optional[str] = None) -> "ValidationResult":
        """Create a successful validation result."""
        return cls(success=True, source_hash=source_hash)

    @classmethod
    def invalid(
        cls,
        errors: list[ValidationError],
        warnings: Optional[list[ValidationError]] = None,
        source_hash: Optional[str] = None,
    ) -> "ValidationResult":
        """Create a failed validation result."""
        return cls(
            success=False,
            errors=errors,
            warnings=warnings or [],
            source_hash=source_hash,
        )


@dataclass
class ImportInfo:
    """Information about an import statement.

    Attributes:
        module: The module being imported (e.g., "trinity.ecs").
        names: Names imported from the module (for from imports).
        alias: Alias if using `import X as Y`.
        is_from_import: True if this is a `from X import Y` statement.
        line: Line number of the import.
    """
    module: str
    names: list[str] = field(default_factory=list)
    alias: Optional[str] = None
    is_from_import: bool = False
    line: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result: dict[str, Any] = {
            "module": self.module,
            "is_from_import": self.is_from_import,
            "line": self.line,
        }
        if self.names:
            result["names"] = self.names
        if self.alias is not None:
            result["alias"] = self.alias
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImportInfo":
        """Create from dictionary."""
        return cls(
            module=data["module"],
            names=data.get("names", []),
            alias=data.get("alias"),
            is_from_import=data.get("is_from_import", False),
            line=data.get("line", 0),
        )


@dataclass
class GenerationResult:
    """Result of generating Python code from a graph.

    Attributes:
        source: The generated Python source code.
        validation: Validation result for the generated code.
        imports: List of imports used in the generated code.
        node_count: Number of nodes processed.
        metadata: Additional metadata about the generation.
    """
    source: str
    validation: ValidationResult
    imports: list[ImportInfo] = field(default_factory=list)
    node_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if generation was successful (valid code)."""
        return self.validation.success

    @property
    def has_errors(self) -> bool:
        """Check if there are validation errors."""
        return self.validation.has_errors

    @property
    def has_warnings(self) -> bool:
        """Check if there are validation warnings."""
        return self.validation.has_warnings

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result: dict[str, Any] = {
            "source": self.source,
            "validation": self.validation.to_dict(),
            "imports": [i.to_dict() for i in self.imports],
            "node_count": self.node_count,
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationResult":
        """Create from dictionary."""
        return cls(
            source=data["source"],
            validation=ValidationResult.from_dict(data["validation"]),
            imports=[ImportInfo.from_dict(i) for i in data.get("imports", [])],
            node_count=data.get("node_count", 0),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def empty(cls) -> "GenerationResult":
        """Create an empty generation result."""
        return cls(
            source="",
            validation=ValidationResult.valid(),
            node_count=0,
        )

    @classmethod
    def error(cls, message: str, line: int = 1, column: int = 0) -> "GenerationResult":
        """Create a generation result with an error."""
        return cls(
            source="",
            validation=ValidationResult.invalid([
                ValidationError.syntax_error(message, line, column)
            ]),
            node_count=0,
        )
