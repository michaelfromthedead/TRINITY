"""Centralized logging system with levels, categories, and thread-safe operation.

Provides the core logging infrastructure for the game engine.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum, auto
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .log_targets import LogTarget
    from .log_filter import LogFilter
    from .log_format import LogFormatter


class LogLevel(IntEnum):
    """Log severity levels in order of increasing severity."""
    TRACE = 0  # Finest-grained information
    DEBUG = 1  # Debug information
    INFO = 2  # General information
    WARNING = 3  # Warning conditions
    ERROR = 4  # Error conditions
    FATAL = 5  # Fatal errors

    @property
    def name_short(self) -> str:
        """Get short level name for display."""
        return {
            LogLevel.TRACE: "TRC",
            LogLevel.DEBUG: "DBG",
            LogLevel.INFO: "INF",
            LogLevel.WARNING: "WRN",
            LogLevel.ERROR: "ERR",
            LogLevel.FATAL: "FTL",
        }[self]


class LogCategory(IntEnum):
    """Pre-defined log categories for filtering."""
    ENGINE = auto()
    GAME = auto()
    RENDER = auto()
    PHYSICS = auto()
    AI = auto()
    NETWORK = auto()
    AUDIO = auto()
    EDITOR = auto()
    INPUT = auto()
    RESOURCE = auto()
    SCRIPT = auto()
    UI = auto()
    MEMORY = auto()
    PROFILE = auto()
    CUSTOM = auto()  # User-defined category

    @classmethod
    def from_string(cls, name: str) -> LogCategory:
        """Get category from string name.

        Args:
            name: Category name (case-insensitive)

        Returns:
            Matching category or CUSTOM if not found
        """
        name_upper = name.upper()
        for cat in cls:
            if cat.name == name_upper:
                return cat
        return cls.CUSTOM


@dataclass(slots=True)
class LogMessage:
    """A single log message with all metadata."""
    level: LogLevel
    category: LogCategory
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    thread_id: int = field(default_factory=threading.get_ident)
    thread_name: str = field(default_factory=lambda: threading.current_thread().name)
    file: Optional[str] = None
    line: Optional[int] = None
    function: Optional[str] = None
    exception: Optional[BaseException] = None
    context: dict = field(default_factory=dict)
    category_name: Optional[str] = None  # Custom category name if CUSTOM

    @property
    def elapsed_ms(self) -> float:
        """Get milliseconds since logging started (approximate)."""
        return (self.timestamp.timestamp() - LogSystem._start_time) * 1000

    def format_source(self) -> str:
        """Format source location string.

        Returns:
            Formatted source location
        """
        parts = []
        if self.file:
            parts.append(self.file)
        if self.line:
            parts.append(str(self.line))
        if self.function:
            parts.append(self.function)
        return ":".join(parts) if parts else ""


@dataclass
class LogConfig:
    """Configuration for the logging system."""
    min_level: LogLevel = LogLevel.INFO
    enabled_categories: Optional[set[LogCategory]] = None  # None = all
    disabled_categories: set[LogCategory] = field(default_factory=set)
    include_source: bool = True
    async_logging: bool = True
    buffer_size: int = 1024
    flush_interval: float = 0.1  # seconds
    max_message_length: int = 10000


class LogSystem:
    """Central logging system with thread-safe operation.

    Provides:
    - Multiple output targets
    - Level and category filtering
    - Optional async logging with buffer
    - Callbacks for log events
    """
    __slots__ = (
        '_config', '_targets', '_filters', '_formatter',
        '_lock', '_callbacks', '_enabled', '_buffer',
        '_buffer_lock', '_flush_thread', '_running'
    )

    _instance: Optional[LogSystem] = None
    _instance_lock = threading.Lock()
    _start_time: float = time.time()

    def __init__(self, config: Optional[LogConfig] = None):
        """Initialize the log system.

        Args:
            config: Logging configuration
        """
        self._config = config or LogConfig()
        self._targets: list[LogTarget] = []
        self._filters: list[LogFilter] = []
        self._formatter: Optional[LogFormatter] = None
        self._lock = threading.RLock()
        self._callbacks: list[Callable[[LogMessage], None]] = []
        self._enabled = True
        self._buffer: list[LogMessage] = []
        self._buffer_lock = threading.Lock()
        self._flush_thread: Optional[threading.Thread] = None
        self._running = False

        if self._config.async_logging:
            self._start_flush_thread()

    @classmethod
    def get_instance(cls) -> LogSystem:
        """Get the singleton instance.

        Returns:
            The global LogSystem instance
        """
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._instance_lock:
            if cls._instance:
                cls._instance.shutdown()
            cls._instance = None
            cls._start_time = time.time()

    @property
    def config(self) -> LogConfig:
        """Get current configuration."""
        return self._config

    @property
    def enabled(self) -> bool:
        """Check if logging is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable logging."""
        self._enabled = value

    def add_target(self, target: 'LogTarget') -> None:
        """Add an output target.

        Args:
            target: Target to add
        """
        with self._lock:
            if target not in self._targets:
                self._targets.append(target)

    def remove_target(self, target: 'LogTarget') -> None:
        """Remove an output target.

        Args:
            target: Target to remove
        """
        with self._lock:
            try:
                self._targets.remove(target)
            except ValueError:
                pass

    def add_filter(self, log_filter: 'LogFilter') -> None:
        """Add a log filter.

        Args:
            log_filter: Filter to add
        """
        with self._lock:
            if log_filter not in self._filters:
                self._filters.append(log_filter)

    def remove_filter(self, log_filter: 'LogFilter') -> None:
        """Remove a log filter.

        Args:
            log_filter: Filter to remove
        """
        with self._lock:
            try:
                self._filters.remove(log_filter)
            except ValueError:
                pass

    def set_formatter(self, formatter: 'LogFormatter') -> None:
        """Set the log formatter.

        Args:
            formatter: Formatter to use
        """
        self._formatter = formatter

    def add_callback(self, callback: Callable[[LogMessage], None]) -> None:
        """Add a log callback.

        Args:
            callback: Function to call for each log message
        """
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[LogMessage], None]) -> None:
        """Remove a log callback.

        Args:
            callback: Callback to remove
        """
        with self._lock:
            try:
                self._callbacks.remove(callback)
            except ValueError:
                pass

    def log(
        self,
        level: LogLevel,
        category: LogCategory,
        message: str,
        exception: Optional[BaseException] = None,
        context: Optional[dict] = None,
        **kwargs: Any
    ) -> None:
        """Log a message.

        Args:
            level: Log level
            category: Log category
            message: Message text
            exception: Optional exception
            context: Additional context data
            **kwargs: Additional fields (file, line, function)
        """
        if not self._enabled:
            return

        # Quick level check
        if level < self._config.min_level:
            return

        # Quick category check
        if self._config.enabled_categories is not None:
            if category not in self._config.enabled_categories:
                return

        if category in self._config.disabled_categories:
            return

        # Truncate message if needed
        if len(message) > self._config.max_message_length:
            message = message[:self._config.max_message_length] + "..."

        # Create message
        log_msg = LogMessage(
            level=level,
            category=category,
            message=message,
            exception=exception,
            context=context or {},
            file=kwargs.get('file'),
            line=kwargs.get('line'),
            function=kwargs.get('function'),
            category_name=kwargs.get('category_name')
        )

        # Apply filters
        with self._lock:
            for f in self._filters:
                from .log_filter import FilterAction
                action = f.filter(log_msg)
                if action == FilterAction.DROP:
                    return
                elif action == FilterAction.MODIFY:
                    # Filter may have modified the message
                    pass

        # Write to targets
        if self._config.async_logging:
            with self._buffer_lock:
                self._buffer.append(log_msg)
                if len(self._buffer) >= self._config.buffer_size:
                    self._flush_buffer()
        else:
            self._write_message(log_msg)

        # Notify callbacks
        with self._lock:
            for callback in self._callbacks:
                try:
                    callback(log_msg)
                except Exception:
                    pass

    def _write_message(self, message: LogMessage) -> None:
        """Write message to all targets.

        Args:
            message: Message to write
        """
        with self._lock:
            for target in self._targets:
                try:
                    target.write(message, self._formatter)
                except Exception:
                    pass  # Don't let target errors break logging

    def _flush_buffer(self) -> None:
        """Flush buffered messages to targets."""
        with self._buffer_lock:
            messages = self._buffer.copy()
            self._buffer.clear()

        for msg in messages:
            self._write_message(msg)

    def _start_flush_thread(self) -> None:
        """Start the background flush thread."""
        self._running = True
        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            name="LogFlushThread",
            daemon=True
        )
        self._flush_thread.start()

    def _flush_loop(self) -> None:
        """Background thread for periodic flushing."""
        while self._running:
            time.sleep(self._config.flush_interval)
            self._flush_buffer()

    def flush(self) -> None:
        """Flush all pending messages immediately."""
        self._flush_buffer()

    def shutdown(self) -> None:
        """Shutdown the logging system."""
        self._running = False
        if self._flush_thread:
            self._flush_thread.join(timeout=1.0)

        self.flush()

        with self._lock:
            for target in self._targets:
                try:
                    target.close()
                except Exception:
                    pass
            self._targets.clear()

    def set_level(self, level: LogLevel) -> None:
        """Set minimum log level.

        Args:
            level: New minimum level
        """
        self._config.min_level = level

    def enable_category(self, category: LogCategory) -> None:
        """Enable a category.

        Args:
            category: Category to enable
        """
        if self._config.enabled_categories is not None:
            self._config.enabled_categories.add(category)
        self._config.disabled_categories.discard(category)

    def disable_category(self, category: LogCategory) -> None:
        """Disable a category.

        Args:
            category: Category to disable
        """
        self._config.disabled_categories.add(category)
        if self._config.enabled_categories is not None:
            self._config.enabled_categories.discard(category)

    # Convenience methods

    def trace(
        self,
        message: str,
        category: LogCategory = LogCategory.ENGINE,
        **kwargs
    ) -> None:
        """Log trace message."""
        self.log(LogLevel.TRACE, category, message, **kwargs)

    def debug(
        self,
        message: str,
        category: LogCategory = LogCategory.ENGINE,
        **kwargs
    ) -> None:
        """Log debug message."""
        self.log(LogLevel.DEBUG, category, message, **kwargs)

    def info(
        self,
        message: str,
        category: LogCategory = LogCategory.ENGINE,
        **kwargs
    ) -> None:
        """Log info message."""
        self.log(LogLevel.INFO, category, message, **kwargs)

    def warning(
        self,
        message: str,
        category: LogCategory = LogCategory.ENGINE,
        **kwargs
    ) -> None:
        """Log warning message."""
        self.log(LogLevel.WARNING, category, message, **kwargs)

    def error(
        self,
        message: str,
        category: LogCategory = LogCategory.ENGINE,
        exception: Optional[BaseException] = None,
        **kwargs
    ) -> None:
        """Log error message."""
        self.log(LogLevel.ERROR, category, message, exception=exception, **kwargs)

    def fatal(
        self,
        message: str,
        category: LogCategory = LogCategory.ENGINE,
        exception: Optional[BaseException] = None,
        **kwargs
    ) -> None:
        """Log fatal message."""
        self.log(LogLevel.FATAL, category, message, exception=exception, **kwargs)


# Module-level convenience functions
def get_logger() -> LogSystem:
    """Get the global log system instance."""
    return LogSystem.get_instance()


def trace(message: str, category: LogCategory = LogCategory.ENGINE, **kwargs) -> None:
    """Log trace message to global logger."""
    get_logger().trace(message, category, **kwargs)


def debug(message: str, category: LogCategory = LogCategory.ENGINE, **kwargs) -> None:
    """Log debug message to global logger."""
    get_logger().debug(message, category, **kwargs)


def info(message: str, category: LogCategory = LogCategory.ENGINE, **kwargs) -> None:
    """Log info message to global logger."""
    get_logger().info(message, category, **kwargs)


def warning(message: str, category: LogCategory = LogCategory.ENGINE, **kwargs) -> None:
    """Log warning message to global logger."""
    get_logger().warning(message, category, **kwargs)


def error(
    message: str,
    category: LogCategory = LogCategory.ENGINE,
    exception: Optional[BaseException] = None,
    **kwargs
) -> None:
    """Log error message to global logger."""
    get_logger().error(message, category, exception, **kwargs)


def fatal(
    message: str,
    category: LogCategory = LogCategory.ENGINE,
    exception: Optional[BaseException] = None,
    **kwargs
) -> None:
    """Log fatal message to global logger."""
    get_logger().fatal(message, category, exception, **kwargs)
