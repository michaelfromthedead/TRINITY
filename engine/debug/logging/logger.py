"""
Core logging module for the game engine.

Provides a flexible, high-performance logging system with:
- Multiple log levels (VERBOSE through FATAL)
- Category-based filtering for engine subsystems
- Structured logging with JSON output
- Multiple output sinks support
- Thread-safe operation

Example:
    >>> from engine.debug.logging.logger import Logger, LogLevel, LogCategory
    >>> logger = Logger("GameEngine")
    >>> logger.info("Engine initialized", LogCategory.LogEngine)
    >>> logger.set_level(LogCategory.LogRendering, LogLevel.DEBUG)
    >>> logger.debug("Frame rendered", LogCategory.LogRendering, frame=60)
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any, Callable

from engine.core.constants import (
    LOG_LEVEL_VERBOSE,
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARNING,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_FATAL,
)

if TYPE_CHECKING:
    from engine.debug.logging.sinks import LogSink
    from engine.debug.logging.filters import LogFilter


class LogLevel(IntEnum):
    """
    Log severity levels ordered from most to least verbose.

    Lower numeric values are more verbose. Filtering works by setting
    a minimum level - all messages at or above that level are logged.

    Attributes:
        VERBOSE: Extremely detailed tracing information
        DEBUG: Debugging information for development
        INFO: General informational messages
        WARNING: Warning conditions that might need attention
        ERROR: Error conditions that affect operation
        FATAL: Critical errors that may cause termination
    """
    VERBOSE = LOG_LEVEL_VERBOSE
    DEBUG = LOG_LEVEL_DEBUG
    INFO = LOG_LEVEL_INFO
    WARNING = LOG_LEVEL_WARNING
    ERROR = LOG_LEVEL_ERROR
    FATAL = LOG_LEVEL_FATAL

    def __str__(self) -> str:
        return self.name


class LogCategory(IntEnum):
    """
    Log categories for different engine subsystems.

    Each category can have its own log level filter, allowing
    fine-grained control over which messages are logged.

    Attributes:
        LogEngine: Core engine operations
        LogRendering: Graphics and rendering
        LogPhysics: Physics simulation
        LogAI: AI and behavior systems
        LogNetwork: Networking and multiplayer
        LogAudio: Audio and sound
        LogAnimation: Animation systems
        LogInput: Input handling
        LogGameplay: Gameplay mechanics
        LogPlayer: Player-specific events
        LogUI: User interface
    """
    LogEngine = auto()
    LogRendering = auto()
    LogPhysics = auto()
    LogAI = auto()
    LogNetwork = auto()
    LogAudio = auto()
    LogAnimation = auto()
    LogInput = auto()
    LogGameplay = auto()
    LogPlayer = auto()
    LogUI = auto()

    def __str__(self) -> str:
        return self.name


@dataclass
class LogEntry:
    """
    Represents a single log entry with all metadata.

    Attributes:
        timestamp: When the log was created (UTC)
        level: Severity level of the log
        category: Engine subsystem category
        message: The log message
        logger_name: Name of the logger that created this entry
        fields: Additional structured data fields
        source_file: Optional source file location
        source_line: Optional source line number
    """
    timestamp: datetime
    level: LogLevel
    category: LogCategory
    message: str
    logger_name: str
    fields: dict[str, Any] = field(default_factory=dict)
    source_file: str | None = None
    source_line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert log entry to dictionary for serialization."""
        result = {
            "timestamp": self.timestamp.isoformat(),
            "level": str(self.level),
            "category": str(self.category),
            "message": self.message,
            "logger": self.logger_name,
        }
        if self.fields:
            result["fields"] = self.fields
        if self.source_file:
            result["source"] = {
                "file": self.source_file,
                "line": self.source_line,
            }
        return result

    def to_json(self) -> str:
        """Convert log entry to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class Logger:
    """
    Main logger class for the game engine.

    Provides methods for logging at different levels with category-based
    filtering. Supports multiple output sinks and filters.

    Attributes:
        name: Logger name for identification

    Example:
        >>> logger = Logger("Physics")
        >>> logger.set_level(LogCategory.LogPhysics, LogLevel.DEBUG)
        >>> logger.debug("Collision detected", LogCategory.LogPhysics,
        ...              object_a="player", object_b="wall")
    """

    # Default log level for all categories
    _default_level: LogLevel = LogLevel.INFO

    # Class-level storage for shared configuration
    _global_sinks: list[LogSink] = []
    _global_filters: list[LogFilter] = []
    _global_lock = threading.Lock()

    def __init__(self, name: str) -> None:
        """
        Initialize a new logger.

        Args:
            name: Identifier for this logger instance
        """
        self.name = name
        self._level_overrides: dict[LogCategory, LogLevel] = {}
        self._sinks: list[LogSink] = []
        self._filters: list[LogFilter] = []
        self._lock = threading.RLock()
        self._enabled = True
        self._callbacks: list[Callable[[LogEntry], None]] = []

    @property
    def enabled(self) -> bool:
        """Check if logger is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable the logger."""
        self._enabled = value

    def set_level(self, category: LogCategory, level: LogLevel) -> None:
        """
        Set the minimum log level for a specific category.

        Messages below this level will be filtered out for the category.

        Args:
            category: The log category to configure
            level: Minimum level to log for this category
        """
        with self._lock:
            self._level_overrides[category] = level

    def get_level(self, category: LogCategory) -> LogLevel:
        """
        Get the current log level for a category.

        Args:
            category: The log category to query

        Returns:
            The minimum log level for the category
        """
        with self._lock:
            return self._level_overrides.get(category, self._default_level)

    def reset_level(self, category: LogCategory) -> None:
        """
        Reset a category to use the default log level.

        Args:
            category: The category to reset
        """
        with self._lock:
            self._level_overrides.pop(category, None)

    def add_sink(self, sink: LogSink) -> None:
        """
        Add an output sink for log messages.

        Args:
            sink: The sink to add
        """
        with self._lock:
            if sink not in self._sinks:
                self._sinks.append(sink)

    def remove_sink(self, sink: LogSink) -> None:
        """
        Remove an output sink.

        Args:
            sink: The sink to remove
        """
        with self._lock:
            if sink in self._sinks:
                self._sinks.remove(sink)

    def add_filter(self, log_filter: LogFilter) -> None:
        """
        Add a filter for log messages.

        Args:
            log_filter: The filter to add
        """
        with self._lock:
            if log_filter not in self._filters:
                self._filters.append(log_filter)

    def remove_filter(self, log_filter: LogFilter) -> None:
        """
        Remove a filter.

        Args:
            log_filter: The filter to remove
        """
        with self._lock:
            if log_filter in self._filters:
                self._filters.remove(log_filter)

    def add_callback(self, callback: Callable[[LogEntry], None]) -> None:
        """
        Add a callback to be invoked for each log entry.

        Args:
            callback: Function to call with each LogEntry
        """
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[LogEntry], None]) -> None:
        """
        Remove a callback.

        Args:
            callback: The callback to remove
        """
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def _should_log(self, level: LogLevel, category: LogCategory) -> bool:
        """Check if a message at the given level/category should be logged."""
        if not self._enabled:
            return False
        min_level = self.get_level(category)
        return level >= min_level

    def _apply_filters(self, entry: LogEntry) -> bool:
        """Apply all filters to determine if entry should be logged."""
        # Check local filters
        for f in self._filters:
            if not f.should_log(entry):
                return False
        # Check global filters
        with self._global_lock:
            for f in self._global_filters:
                if not f.should_log(entry):
                    return False
        return True

    def _write_to_sinks(self, entry: LogEntry) -> None:
        """Write the entry to all configured sinks."""
        # Write to local sinks
        for sink in self._sinks:
            try:
                sink.write(entry)
            except Exception:
                pass  # Don't let sink errors crash the application

        # Write to global sinks
        with self._global_lock:
            for sink in self._global_sinks:
                try:
                    sink.write(entry)
                except Exception:
                    pass

    def _invoke_callbacks(self, entry: LogEntry) -> None:
        """Invoke all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(entry)
            except Exception:
                pass  # Don't let callback errors crash the application

    def _log(
        self,
        level: LogLevel,
        message: str,
        category: LogCategory,
        source_file: str | None = None,
        source_line: int | None = None,
        **fields: Any,
    ) -> None:
        """
        Internal method to create and dispatch a log entry.

        Args:
            level: Log severity level
            message: The log message
            category: Engine subsystem category
            source_file: Optional source file
            source_line: Optional source line
            **fields: Additional structured data fields
        """
        if not self._should_log(level, category):
            return

        entry = LogEntry(
            timestamp=datetime.now(timezone.utc),
            level=level,
            category=category,
            message=message,
            logger_name=self.name,
            fields=fields,
            source_file=source_file,
            source_line=source_line,
        )

        with self._lock:
            if not self._apply_filters(entry):
                return
            self._write_to_sinks(entry)
            self._invoke_callbacks(entry)

    def verbose(
        self,
        message: str,
        category: LogCategory = LogCategory.LogEngine,
        **fields: Any,
    ) -> None:
        """
        Log a verbose message.

        Use for extremely detailed tracing information.

        Args:
            message: The log message
            category: Engine subsystem category
            **fields: Additional structured data
        """
        self._log(LogLevel.VERBOSE, message, category, **fields)

    def debug(
        self,
        message: str,
        category: LogCategory = LogCategory.LogEngine,
        **fields: Any,
    ) -> None:
        """
        Log a debug message.

        Use for debugging information during development.

        Args:
            message: The log message
            category: Engine subsystem category
            **fields: Additional structured data
        """
        self._log(LogLevel.DEBUG, message, category, **fields)

    def info(
        self,
        message: str,
        category: LogCategory = LogCategory.LogEngine,
        **fields: Any,
    ) -> None:
        """
        Log an informational message.

        Use for general informational messages.

        Args:
            message: The log message
            category: Engine subsystem category
            **fields: Additional structured data
        """
        self._log(LogLevel.INFO, message, category, **fields)

    def warning(
        self,
        message: str,
        category: LogCategory = LogCategory.LogEngine,
        **fields: Any,
    ) -> None:
        """
        Log a warning message.

        Use for warning conditions that might need attention.

        Args:
            message: The log message
            category: Engine subsystem category
            **fields: Additional structured data
        """
        self._log(LogLevel.WARNING, message, category, **fields)

    def error(
        self,
        message: str,
        category: LogCategory = LogCategory.LogEngine,
        **fields: Any,
    ) -> None:
        """
        Log an error message.

        Use for error conditions that affect operation.

        Args:
            message: The log message
            category: Engine subsystem category
            **fields: Additional structured data
        """
        self._log(LogLevel.ERROR, message, category, **fields)

    def fatal(
        self,
        message: str,
        category: LogCategory = LogCategory.LogEngine,
        **fields: Any,
    ) -> None:
        """
        Log a fatal error message.

        Use for critical errors that may cause termination.

        Args:
            message: The log message
            category: Engine subsystem category
            **fields: Additional structured data
        """
        self._log(LogLevel.FATAL, message, category, **fields)

    def structured(
        self,
        message: str,
        category: LogCategory = LogCategory.LogEngine,
        level: LogLevel = LogLevel.INFO,
        **fields: Any,
    ) -> None:
        """
        Log a structured message with typed fields.

        This is the preferred method for logging data that will be
        parsed and analyzed. Fields are serialized to JSON.

        Args:
            message: The log message
            category: Engine subsystem category
            level: Log severity level
            **fields: Typed data fields to include

        Example:
            >>> logger.structured(
            ...     "Player action",
            ...     LogCategory.LogPlayer,
            ...     level=LogLevel.INFO,
            ...     player_id=123,
            ...     action="jump",
            ...     position={"x": 10.5, "y": 20.0},
            ... )
        """
        self._log(level, message, category, **fields)

    def log_exception(
        self,
        exception: BaseException,
        message: str | None = None,
        category: LogCategory = LogCategory.LogEngine,
    ) -> None:
        """
        Log an exception with traceback information.

        Args:
            exception: The exception to log
            message: Optional additional message
            category: Engine subsystem category
        """
        import traceback

        exc_info = {
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "traceback": traceback.format_exc(),
        }

        log_message = message or f"Exception: {exception}"
        self._log(LogLevel.ERROR, log_message, category, **exc_info)

    @classmethod
    def add_global_sink(cls, sink: LogSink) -> None:
        """
        Add a sink that receives messages from all loggers.

        Args:
            sink: The sink to add globally
        """
        with cls._global_lock:
            if sink not in cls._global_sinks:
                cls._global_sinks.append(sink)

    @classmethod
    def remove_global_sink(cls, sink: LogSink) -> None:
        """
        Remove a global sink.

        Args:
            sink: The sink to remove
        """
        with cls._global_lock:
            if sink in cls._global_sinks:
                cls._global_sinks.remove(sink)

    @classmethod
    def add_global_filter(cls, log_filter: LogFilter) -> None:
        """
        Add a filter that applies to all loggers.

        Args:
            log_filter: The filter to add globally
        """
        with cls._global_lock:
            if log_filter not in cls._global_filters:
                cls._global_filters.append(log_filter)

    @classmethod
    def remove_global_filter(cls, log_filter: LogFilter) -> None:
        """
        Remove a global filter.

        Args:
            log_filter: The filter to remove
        """
        with cls._global_lock:
            if log_filter in cls._global_filters:
                cls._global_filters.remove(log_filter)

    @classmethod
    def set_default_level(cls, level: LogLevel) -> None:
        """
        Set the default log level for all categories.

        Args:
            level: The default minimum log level
        """
        cls._default_level = level

    @classmethod
    def clear_global_sinks(cls) -> None:
        """Remove all global sinks."""
        with cls._global_lock:
            cls._global_sinks.clear()

    @classmethod
    def clear_global_filters(cls) -> None:
        """Remove all global filters."""
        with cls._global_lock:
            cls._global_filters.clear()


def get_logger(name: str) -> Logger:
    """
    Get or create a logger with the given name.

    This is the recommended way to obtain loggers.

    Args:
        name: The logger name

    Returns:
        A Logger instance
    """
    return Logger(name)
