# Investigation Report: engine/debug/logging/

**Date:** 2026-05-22  
**Classification:** REAL (Fully Implemented)  
**Total Lines:** 3,182  

## Summary

The `engine/debug/logging/` module is a **fully implemented**, production-ready logging system with comprehensive functionality including multiple log levels, category-based filtering, multiple output sinks, file rotation, and structured JSON logging with distributed tracing support.

## Classification: REAL

**Evidence of full implementation:**
- All classes contain complete working logic with no placeholder/stub code
- Thread-safe implementations using `threading.Lock` and `threading.RLock`
- Real file I/O operations with error handling
- Network socket implementation for remote logging
- Gzip compression support for rotated files
- Integration with `engine.core.constants` for configuration defaults

## Architecture Overview

```
engine/debug/logging/
    __init__.py          (147 lines) - Module exports and documentation
    logger.py            (635 lines) - Core Logger class and LogEntry
    sinks.py             (681 lines) - Output destinations (Console, File, Network)
    rotation.py          (624 lines) - File rotation handlers
    filters.py           (575 lines) - Log filtering system
    structured.py        (520 lines) - Structured logging with JSON support
```

## Logging Levels

Defined in `logger.py` and backed by constants in `engine.core.constants`:

| Level | Value | Use Case |
|-------|-------|----------|
| VERBOSE | 0 | Extremely detailed tracing information |
| DEBUG | 10 | Debugging information for development |
| INFO | 20 | General informational messages (default) |
| WARNING | 30 | Warning conditions that might need attention |
| ERROR | 40 | Error conditions that affect operation |
| FATAL | 50 | Critical errors that may cause termination |

## Log Categories

Engine subsystem categories for fine-grained filtering:

| Category | Purpose |
|----------|---------|
| LogEngine | Core engine operations |
| LogRendering | Graphics and rendering |
| LogPhysics | Physics simulation |
| LogAI | AI and behavior systems |
| LogNetwork | Networking and multiplayer |
| LogAudio | Audio and sound |
| LogAnimation | Animation systems |
| LogInput | Input handling |
| LogGameplay | Gameplay mechanics |
| LogPlayer | Player-specific events |
| LogUI | User interface |

## Output Sinks

### ConsoleSink
- ANSI color support with automatic TTY detection
- Windows 10+ ANSI escape sequence support
- Configurable timestamp format
- Thread-safe output

**Color mapping:**
- VERBOSE: Dim gray (`\033[90m`)
- DEBUG: Cyan (`\033[36m`)
- INFO: White (`\033[37m`)
- WARNING: Yellow (`\033[33m`)
- ERROR: Red (`\033[31m`)
- FATAL: Bright red, bold (`\033[1;91m`)

### FileSink
- Size-based rotation (default: 10 MB)
- Configurable backup count (default: 5)
- Optional gzip compression for rotated files
- JSON or pipe-delimited format
- Auto-creates parent directories

### NetworkSink
- TCP or UDP transport
- Batched sending with configurable batch size (default: 100)
- Automatic reconnection with delay (default: 5s)
- Background sender thread (daemon)
- JSON-formatted log entries

### BufferedSink
- Wrapper for any sink with buffering
- Size-based flush (default: 100 entries)
- Time-based flush (default: 1.0 seconds)
- Background flush timer thread

### MultiplexSink
- Fan-out to multiple sinks
- Error isolation between sinks

## File Rotation

### RotatingFileHandler
- Size-based rotation with configurable threshold
- Numbered backup files (.1, .2, .3, etc.)
- Optional gzip compression
- Backup count limit with automatic cleanup

### TimedRotatingFileHandler
- Time-based rotation intervals:
  - `s`: Seconds (testing)
  - `m`: Minutes
  - `h`: Hours
  - `d`: Days
  - `w`: Weeks
  - `midnight`: Daily at midnight
- Timestamped backup naming (YYYYMMDD format)
- UTC or local time support
- Automatic old backup cleanup

### LogArchiver
- Bulk archival of old logs
- Compression during archival
- Configurable age thresholds (default: 30 days archive, 90 days cleanup)
- Statistics gathering

### CompressedFileReader
- Read both plain and gzipped log files
- Iterator interface for line-by-line reading

## Filters

### LevelFilter
- Minimum severity level threshold
- Standard comparison operators

### CategoryFilter
- Include-only or exclude-only modes (mutually exclusive)
- Dynamic category addition

### KeywordFilter
- String keyword matching (include/exclude)
- Regex pattern support
- Case-sensitive or insensitive
- Optional field searching

### CompositeFilter
- AND/OR combination of multiple filters
- Operator overloading (`&`, `|`)
- Empty filter list returns True

### NegateFilter
- Invert any filter result
- Operator overloading (`~`)

### RateLimitFilter
- Sliding window rate limiting
- Per-category or per-logger limits
- Thread-safe timestamp tracking

### SamplingFilter
- Probabilistic sampling (0.0-1.0)
- Optional seed for reproducibility

### CallbackFilter
- Custom function-based filtering
- Maximum flexibility

### FieldFilter
- Filter on structured log field values
- Lambda predicates
- Optional field requirement

## Structured Logging

### StructuredLog
- Dataclass with full serialization support
- JSON import/export
- Distributed tracing fields (trace_id, span_id)
- Tags support
- Immutable field/tag addition (returns new instance)

### StructuredLogBuilder
- Fluent API for constructing logs
- Method chaining
- Default values

### LogSchema
- Schema definition with type constraints
- Validation with error collection
- Required field enforcement

### LogContext
- Context manager for scoped fields
- Nested context support
- Class-level current context tracking

### Parsing Utilities
- `parse_log_line()`: Single JSON line parsing
- `parse_log_file()`: Full file parsing

## Thread Safety

All components use appropriate synchronization:
- `Logger`: `threading.RLock` for re-entrant access
- Sinks: `threading.Lock` for write operations
- Global sinks/filters: Class-level `threading.Lock`
- Filters: `threading.Lock` where state is mutable

## Configuration Defaults

From `engine/core/constants.py`:

| Constant | Value |
|----------|-------|
| LOG_FILE_MAX_SIZE | 10 MB |
| LOG_FILE_MAX_BACKUPS | 5 |
| LOG_FILE_ENCODING | utf-8 |
| LOG_NETWORK_TIMEOUT | 5.0s |
| LOG_NETWORK_BATCH_SIZE | 100 |
| LOG_NETWORK_FLUSH_INTERVAL | 1.0s |
| LOG_NETWORK_RECONNECT_DELAY | 5.0s |
| LOG_BUFFER_SIZE | 100 |
| LOG_BUFFER_FLUSH_INTERVAL | 1.0s |
| LOG_ROTATION_DAILY_BACKUPS | 7 |
| LOG_ARCHIVER_DEFAULT_DAYS | 30 |
| LOG_CLEANUP_DEFAULT_DAYS | 90 |

## Usage Example

```python
from engine.debug.logging import (
    Logger, LogLevel, LogCategory,
    ConsoleSink, FileSink, RotatingFileHandler,
    LevelFilter, CategoryFilter, StructuredLogBuilder
)

# Create logger
logger = Logger("GameEngine")

# Add sinks
logger.add_sink(ConsoleSink(use_colors=True))
logger.add_sink(FileSink("game.log", max_size=10*1024*1024))

# Add filters
logger.add_filter(LevelFilter(LogLevel.INFO))
logger.add_filter(CategoryFilter(exclude=[LogCategory.LogInput]))

# Log messages
logger.info("Engine started", LogCategory.LogEngine)
logger.debug("Frame rendered", LogCategory.LogRendering, fps=60, frame=1234)
logger.error("Connection lost", LogCategory.LogNetwork, error="timeout")

# Structured logging
log = (StructuredLogBuilder()
    .message("Player action")
    .level("INFO")
    .category("LogPlayer")
    .field("player_id", 123)
    .field("action", "jump")
    .trace("trace-abc", "span-123")
    .build())
```

## Quality Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| Implementation Completeness | High | All features fully implemented |
| Thread Safety | High | Comprehensive locking throughout |
| Error Handling | Medium | Exceptions silently caught in sinks/callbacks |
| Documentation | High | Docstrings and examples throughout |
| Type Hints | High | Full type annotations |
| Test Coverage | Unknown | No test files in this module |

## Dependencies

- `engine.core.constants`: Configuration defaults
- Standard library: `json`, `threading`, `queue`, `socket`, `gzip`, `shutil`, `re`, `time`, `datetime`, `pathlib`, `dataclasses`, `enum`, `abc`

## Recommendations

1. Consider adding unit tests for rotation edge cases
2. The silent exception catching in sinks could mask issues; consider optional error logging
3. NetworkSink could benefit from SSL/TLS support
4. Consider adding log entry deduplication support
