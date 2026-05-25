"""
Debug Logging System for the AI Game Engine.

A flexible, high-performance logging system with:
- Multiple log levels (VERBOSE through FATAL)
- Category-based filtering for engine subsystems
- Structured logging with JSON output
- Multiple output sinks (console, file, network)
- File rotation by size or time
- Thread-safe operation

Quick Start:
    >>> from engine.debug.logging import Logger, LogLevel, LogCategory
    >>> from engine.debug.logging import ConsoleSink, FileSink
    >>>
    >>> # Create a logger
    >>> logger = Logger("MyGame")
    >>>
    >>> # Add output destinations
    >>> logger.add_sink(ConsoleSink(use_colors=True))
    >>> logger.add_sink(FileSink("game.log", max_size=10*1024*1024))
    >>>
    >>> # Log messages
    >>> logger.info("Game started", LogCategory.LogEngine)
    >>> logger.debug("Frame rendered", LogCategory.LogRendering, fps=60)
    >>> logger.error("Connection failed", LogCategory.LogNetwork, error="timeout")

Structured Logging:
    >>> logger.structured(
    ...     "Player action",
    ...     LogCategory.LogPlayer,
    ...     level=LogLevel.INFO,
    ...     player_id=123,
    ...     action="jump",
    ...     position={"x": 10.5, "y": 20.0},
    ... )

Filtering:
    >>> from engine.debug.logging import LevelFilter, CategoryFilter
    >>>
    >>> # Only show warnings and above
    >>> logger.add_filter(LevelFilter(LogLevel.WARNING))
    >>>
    >>> # Exclude input logging
    >>> logger.add_filter(CategoryFilter(exclude=[LogCategory.LogInput]))

File Rotation:
    >>> from engine.debug.logging import RotatingFileHandler, TimedRotatingFileHandler
    >>>
    >>> # Rotate by size (10 MB)
    >>> handler = RotatingFileHandler("game.log", max_bytes=10*1024*1024)
    >>>
    >>> # Rotate daily at midnight
    >>> handler = TimedRotatingFileHandler("game.log", when="midnight")
"""

# Core logger
from engine.debug.logging.logger import (
    Logger,
    LogLevel,
    LogCategory,
    LogEntry,
    get_logger,
)

# Output sinks
from engine.debug.logging.sinks import (
    LogSink,
    ConsoleSink,
    FileSink,
    NetworkSink,
    BufferedSink,
    MultiplexSink,
)

# Filters
from engine.debug.logging.filters import (
    LogFilter,
    LevelFilter,
    CategoryFilter,
    KeywordFilter,
    CompositeFilter,
    NegateFilter,
    RateLimitFilter,
    SamplingFilter,
    CallbackFilter,
    FieldFilter,
)

# Structured logging
from engine.debug.logging.structured import (
    StructuredLog,
    StructuredLogBuilder,
    LogField,
    LogSchema,
    LogContext,
    parse_log_line,
    parse_log_file,
)

# File rotation
from engine.debug.logging.rotation import (
    RotatingFileHandler,
    TimedRotatingFileHandler,
    CompressedFileReader,
    LogArchiver,
)

__all__ = [
    # Logger
    "Logger",
    "LogLevel",
    "LogCategory",
    "LogEntry",
    "get_logger",
    # Sinks
    "LogSink",
    "ConsoleSink",
    "FileSink",
    "NetworkSink",
    "BufferedSink",
    "MultiplexSink",
    # Filters
    "LogFilter",
    "LevelFilter",
    "CategoryFilter",
    "KeywordFilter",
    "CompositeFilter",
    "NegateFilter",
    "RateLimitFilter",
    "SamplingFilter",
    "CallbackFilter",
    "FieldFilter",
    # Structured
    "StructuredLog",
    "StructuredLogBuilder",
    "LogField",
    "LogSchema",
    "LogContext",
    "parse_log_line",
    "parse_log_file",
    # Rotation
    "RotatingFileHandler",
    "TimedRotatingFileHandler",
    "CompressedFileReader",
    "LogArchiver",
]
