"""
Structured logging support for the game engine.

Provides dataclasses and utilities for creating structured log
entries that can be easily parsed and analyzed.

Example:
    >>> from engine.debug.logging.structured import StructuredLog, LogField
    >>>
    >>> log = StructuredLog(
    ...     message="Player action",
    ...     category="LogPlayer",
    ...     level="INFO",
    ...     fields={
    ...         "player_id": 123,
    ...         "action": "jump",
    ...         "position": {"x": 10.5, "y": 20.0},
    ...     }
    ... )
    >>> print(log.to_json())
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, TypeVar, Generic
from enum import Enum


@dataclass
class LogField:
    """
    A typed field for structured logging.

    Provides metadata about a log field including name, type,
    and optional description.

    Attributes:
        name: Field name
        value: Field value
        field_type: Type name for documentation
        description: Optional description
    """
    name: str
    value: Any
    field_type: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        """Infer type if not provided."""
        if not self.field_type:
            self.field_type = type(self.value).__name__

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {"name": self.name, "value": self.value}
        if self.field_type:
            result["type"] = self.field_type
        if self.description:
            result["description"] = self.description
        return result


@dataclass
class StructuredLog:
    """
    A structured log entry with typed fields.

    Designed for easy serialization to JSON and parsing by
    log analysis tools.

    Attributes:
        timestamp: When the log was created
        level: Log severity level
        category: Engine subsystem category
        message: The log message
        fields: Additional structured data
        logger_name: Name of the logger
        source_file: Optional source file
        source_line: Optional source line
        trace_id: Optional distributed trace ID
        span_id: Optional span ID for tracing

    Example:
        >>> log = StructuredLog(
        ...     message="Request completed",
        ...     level="INFO",
        ...     category="LogNetwork",
        ...     fields={
        ...         "request_id": "abc123",
        ...         "latency_ms": 45.2,
        ...         "status_code": 200,
        ...     },
        ...     trace_id="trace-123",
        ... )
    """
    message: str
    level: str
    category: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    fields: dict[str, Any] = field(default_factory=dict)
    logger_name: str = ""
    source_file: str | None = None
    source_line: int | None = None
    trace_id: str | None = None
    span_id: str | None = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the log
        """
        result = {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "category": self.category,
            "message": self.message,
        }

        if self.logger_name:
            result["logger"] = self.logger_name

        if self.fields:
            result["fields"] = self.fields

        if self.source_file:
            result["source"] = {
                "file": self.source_file,
            }
            if self.source_line:
                result["source"]["line"] = self.source_line

        if self.trace_id:
            result["trace_id"] = self.trace_id

        if self.span_id:
            result["span_id"] = self.span_id

        if self.tags:
            result["tags"] = self.tags

        return result

    def to_json(self, pretty: bool = False) -> str:
        """
        Convert to JSON string.

        Args:
            pretty: Whether to format with indentation

        Returns:
            JSON string representation
        """
        indent = 2 if pretty else None
        return json.dumps(self.to_dict(), default=str, indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StructuredLog:
        """
        Create from dictionary.

        Args:
            data: Dictionary with log data

        Returns:
            StructuredLog instance
        """
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now(timezone.utc)

        source = data.get("source", {})

        return cls(
            message=data["message"],
            level=data["level"],
            category=data["category"],
            timestamp=timestamp,
            fields=data.get("fields", {}),
            logger_name=data.get("logger", ""),
            source_file=source.get("file"),
            source_line=source.get("line"),
            trace_id=data.get("trace_id"),
            span_id=data.get("span_id"),
            tags=data.get("tags", []),
        )

    @classmethod
    def from_json(cls, json_str: str) -> StructuredLog:
        """
        Create from JSON string.

        Args:
            json_str: JSON string to parse

        Returns:
            StructuredLog instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def with_field(self, name: str, value: Any) -> StructuredLog:
        """
        Create a copy with an additional field.

        Args:
            name: Field name
            value: Field value

        Returns:
            New StructuredLog with the field added
        """
        new_fields = {**self.fields, name: value}
        return StructuredLog(
            message=self.message,
            level=self.level,
            category=self.category,
            timestamp=self.timestamp,
            fields=new_fields,
            logger_name=self.logger_name,
            source_file=self.source_file,
            source_line=self.source_line,
            trace_id=self.trace_id,
            span_id=self.span_id,
            tags=self.tags.copy(),
        )

    def with_tag(self, tag: str) -> StructuredLog:
        """
        Create a copy with an additional tag.

        Args:
            tag: Tag to add

        Returns:
            New StructuredLog with the tag added
        """
        new_tags = self.tags.copy()
        if tag not in new_tags:
            new_tags.append(tag)

        return StructuredLog(
            message=self.message,
            level=self.level,
            category=self.category,
            timestamp=self.timestamp,
            fields=self.fields.copy(),
            logger_name=self.logger_name,
            source_file=self.source_file,
            source_line=self.source_line,
            trace_id=self.trace_id,
            span_id=self.span_id,
            tags=new_tags,
        )


@dataclass
class LogSchema:
    """
    Schema definition for structured logs.

    Defines expected fields and their types for validation
    and documentation purposes.

    Attributes:
        name: Schema name
        fields: Dictionary of field name to expected type
        required_fields: Set of required field names
        description: Optional schema description
    """
    name: str
    fields: dict[str, type]
    required_fields: set[str] = field(default_factory=set)
    description: str = ""

    def validate(self, log: StructuredLog) -> list[str]:
        """
        Validate a log entry against this schema.

        Args:
            log: The log entry to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check required fields
        for field_name in self.required_fields:
            if field_name not in log.fields:
                errors.append(f"Missing required field: {field_name}")

        # Check field types
        for field_name, expected_type in self.fields.items():
            if field_name in log.fields:
                value = log.fields[field_name]
                if not isinstance(value, expected_type):
                    errors.append(
                        f"Field {field_name} has wrong type: "
                        f"expected {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )

        return errors

    def is_valid(self, log: StructuredLog) -> bool:
        """
        Check if a log entry is valid against this schema.

        Args:
            log: The log entry to validate

        Returns:
            True if valid, False otherwise
        """
        return len(self.validate(log)) == 0


class StructuredLogBuilder:
    """
    Builder for creating structured log entries.

    Provides a fluent interface for constructing complex log entries.

    Example:
        >>> log = (StructuredLogBuilder()
        ...     .message("Player action")
        ...     .level("INFO")
        ...     .category("LogPlayer")
        ...     .field("player_id", 123)
        ...     .field("action", "jump")
        ...     .tag("gameplay")
        ...     .build())
    """

    def __init__(self) -> None:
        """Initialize the builder with default values."""
        self._message: str = ""
        self._level: str = "INFO"
        self._category: str = "LogEngine"
        self._timestamp: datetime | None = None
        self._fields: dict[str, Any] = {}
        self._logger_name: str = ""
        self._source_file: str | None = None
        self._source_line: int | None = None
        self._trace_id: str | None = None
        self._span_id: str | None = None
        self._tags: list[str] = []

    def message(self, message: str) -> StructuredLogBuilder:
        """Set the log message."""
        self._message = message
        return self

    def level(self, level: str) -> StructuredLogBuilder:
        """Set the log level."""
        self._level = level
        return self

    def category(self, category: str) -> StructuredLogBuilder:
        """Set the log category."""
        self._category = category
        return self

    def timestamp(self, timestamp: datetime) -> StructuredLogBuilder:
        """Set the timestamp."""
        self._timestamp = timestamp
        return self

    def field(self, name: str, value: Any) -> StructuredLogBuilder:
        """Add a field."""
        self._fields[name] = value
        return self

    def fields(self, **kwargs: Any) -> StructuredLogBuilder:
        """Add multiple fields."""
        self._fields.update(kwargs)
        return self

    def logger(self, name: str) -> StructuredLogBuilder:
        """Set the logger name."""
        self._logger_name = name
        return self

    def source(self, file: str, line: int | None = None) -> StructuredLogBuilder:
        """Set the source location."""
        self._source_file = file
        self._source_line = line
        return self

    def trace(self, trace_id: str, span_id: str | None = None) -> StructuredLogBuilder:
        """Set tracing IDs."""
        self._trace_id = trace_id
        self._span_id = span_id
        return self

    def tag(self, tag: str) -> StructuredLogBuilder:
        """Add a tag."""
        if tag not in self._tags:
            self._tags.append(tag)
        return self

    def tags(self, *tags: str) -> StructuredLogBuilder:
        """Add multiple tags."""
        for tag in tags:
            self.tag(tag)
        return self

    def build(self) -> StructuredLog:
        """Build the structured log entry."""
        return StructuredLog(
            message=self._message,
            level=self._level,
            category=self._category,
            timestamp=self._timestamp or datetime.now(timezone.utc),
            fields=self._fields,
            logger_name=self._logger_name,
            source_file=self._source_file,
            source_line=self._source_line,
            trace_id=self._trace_id,
            span_id=self._span_id,
            tags=self._tags,
        )


class LogContext:
    """
    Context manager for adding fields to all logs within a scope.

    Useful for adding request-specific or transaction-specific
    data to all logs within a block.

    Example:
        >>> with LogContext(request_id="abc123", user_id=456) as ctx:
        ...     logger.info("Processing request")  # Includes request_id, user_id
    """

    _current: LogContext | None = None

    def __init__(self, **fields: Any) -> None:
        """
        Initialize context with fields.

        Args:
            **fields: Fields to add to all logs in this context
        """
        self._fields = fields
        self._parent: LogContext | None = None

    def __enter__(self) -> LogContext:
        """Enter the context."""
        self._parent = LogContext._current
        LogContext._current = self
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit the context."""
        LogContext._current = self._parent

    @property
    def fields(self) -> dict[str, Any]:
        """Get all fields including parent context."""
        if self._parent:
            return {**self._parent.fields, **self._fields}
        return self._fields.copy()

    @classmethod
    def current(cls) -> LogContext | None:
        """Get the current context."""
        return cls._current

    @classmethod
    def current_fields(cls) -> dict[str, Any]:
        """Get fields from current context (empty if none)."""
        if cls._current:
            return cls._current.fields
        return {}


def parse_log_line(line: str) -> StructuredLog | None:
    """
    Parse a JSON log line into a StructuredLog.

    Args:
        line: JSON string to parse

    Returns:
        StructuredLog if parsing succeeds, None otherwise
    """
    try:
        return StructuredLog.from_json(line.strip())
    except (json.JSONDecodeError, KeyError):
        return None


def parse_log_file(path: str) -> list[StructuredLog]:
    """
    Parse a file containing JSON log lines.

    Args:
        path: Path to the log file

    Returns:
        List of parsed StructuredLog entries
    """
    logs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            log = parse_log_line(line)
            if log:
                logs.append(log)
    return logs
