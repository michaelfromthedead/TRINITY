"""Logging subsystem for the AI Game Engine.

This module provides a comprehensive logging system with:
- Multiple log levels (Trace, Debug, Info, Warning, Error, Fatal)
- Multiple output targets (file, console, network, memory)
- Category-based filtering
- Structured logging for analytics
- Thread-safe operation with lock-free ring buffer
"""

from .log_system import (
    LogSystem,
    LogLevel,
    LogCategory,
    LogMessage,
    LogConfig,
)
from .log_targets import (
    LogTarget,
    ConsoleTarget,
    FileTarget,
    NetworkTarget,
    RingBufferTarget,
    CompositeTarget,
)
from .log_filter import (
    LogFilter,
    LevelFilter,
    CategoryFilter,
    PatternFilter,
    CompositeFilter,
    FilterAction,
)
from .log_format import (
    LogFormatter,
    DefaultFormatter,
    JsonFormatter,
    CompactFormatter,
    DetailedFormatter,
    ColorFormatter,
)
from .structured_log import (
    StructuredLogger,
    LogContext,
    Span,
    SpanContext,
)

__all__ = [
    # Core
    "LogSystem",
    "LogLevel",
    "LogCategory",
    "LogMessage",
    "LogConfig",
    # Targets
    "LogTarget",
    "ConsoleTarget",
    "FileTarget",
    "NetworkTarget",
    "RingBufferTarget",
    "CompositeTarget",
    # Filters
    "LogFilter",
    "LevelFilter",
    "CategoryFilter",
    "PatternFilter",
    "CompositeFilter",
    "FilterAction",
    # Formatting
    "LogFormatter",
    "DefaultFormatter",
    "JsonFormatter",
    "CompactFormatter",
    "DetailedFormatter",
    "ColorFormatter",
    # Structured
    "StructuredLogger",
    "LogContext",
    "Span",
    "SpanContext",
]
