# Engine Debug Crash System Investigation

**Classification: REAL**  
**Files Analyzed:** 5 files (2,472 lines total)  
**Date:** 2026-05-22

---

## Summary

The crash handling system is a **fully implemented, production-ready** module providing comprehensive crash diagnostics. All four submodules contain real, working code with proper error handling, thread safety, and security considerations. The only stubs are platform-specific native operations (native minidumps, GPU info, screenshot capture) that require OS-specific APIs.

---

## Module Classification

| File | Lines | Classification | Notes |
|------|-------|----------------|-------|
| `__init__.py` | 118 | REAL | Clean public API export |
| `handler.py` | 497 | REAL | Full signal/exception handling |
| `assertions.py` | 479 | REAL | UE4-style assertion macros |
| `minidump.py` | 624 | REAL (with platform stubs) | Python dumps real; native stubs |
| `reporter.py` | 754 | REAL (with stubs) | Full reporting; screenshot stub |

---

## Architecture Overview

```
CrashHandler (installs signal handlers, exception hook)
     |
     v
CrashContext (captured crash state)
     |
     v
CrashReporter (creates CrashReport)
     |
     +---> Minidump (generates dump files)
     +---> SystemInfoSnapshot (captures system state)
     |
     v
Local Save / Remote Upload
```

---

## 1. Crash Handler (`handler.py`)

### Key Components

**CrashContext** (dataclass)
- `exception`: The exception object if crash was exception-based
- `stack_trace`: Formatted traceback string
- `recent_logs`: Circular buffer of recent log messages
- `timestamp`: When crash occurred
- `thread_id`, `thread_name`: Thread identification
- `signal_number`, `signal_name`: For signal-based crashes
- `additional_data`: Extensible metadata dictionary

**RecentLogHandler** (logging.Handler)
- Thread-safe deque with configurable max entries (default 100)
- Captures formatted log records for post-mortem analysis
- Auto-attached to root logger on handler installation

**CrashHandler**
- Signal handlers: SIGINT, SIGTERM, SIGABRT, SIGSEGV, SIGBUS, SIGFPE, SIGILL
- Exception hook: Replaces sys.excepthook
- Thread-safe callback registration
- Idempotent install/uninstall
- Re-entrant protection (`_handling_crash` flag)

### Implementation Quality

- Proper original handler restoration on uninstall
- Exception-safe callback invocation
- atexit handler for cleanup
- Global singleton pattern with `get_global_handler()` / `install_global_handler()`

---

## 2. Assertions (`assertions.py`)

### Assertion Macros (UE4-style naming)

| Macro | Fatal | Debug-only | Description |
|-------|-------|------------|-------------|
| `check(cond)` | Yes | No | Basic fatal assertion |
| `checkf(cond, msg, *args)` | Yes | No | Fatal with format string |
| `verify(value)` | Yes | No | Returns value if truthy, crashes if falsy |
| `ensure(cond)` | No | No | Non-fatal, logs once per location |
| `ensureAlways(cond)` | No | No | Non-fatal, logs every time |
| `checkSlow(cond)` | Yes | Yes | Expensive checks stripped in release |
| `checkSlowf(cond, msg, *args)` | Yes | Yes | Expensive checks with message |

### Configuration

**AssertionResponse** (Enum)
- `BREAK`: Trigger debugger breakpoint via `breakpoint()`
- `LOG`: Log error and continue
- `CRASH`: Terminate via `sys.exit(1)`
- `CONTINUE`: Silent continue (not recommended)

**AssertionConfig**
- `response`: Global behavior setting
- `is_debug_build`: Controls checkSlow behavior (auto-detected from env vars)
- `log_callback`: Custom logging hook
- `crash_callback`: Pre-crash cleanup hook
- `logged_assertions`: Set tracking logged ensure() locations

**AssertionContext** (context manager)
- Temporarily change assertion behavior within a scope
- Useful for tests

### Implementation Details

- Location tracking via `sys._getframe()` for deduplication
- Environment variable auto-detection: `GAME_ENGINE_DEBUG`, `GAME_ENGINE_RELEASE`
- Stack trace printing on crash
- Custom log/crash callbacks for integration

---

## 3. Minidump Generation (`minidump.py`)

### Dump Levels

| Level | Contents |
|-------|----------|
| `MINI` | Stack traces, thread info, platform info, Python info |
| `MEDIUM` | + Loaded modules, memory regions, sanitized environment |
| `FULL` | + Complete memory dump (not implemented in Python) |

### Data Structures

**ThreadInfo**: thread_id, name, is_daemon, is_alive, stack_trace  
**ModuleInfo**: name, path, version (from sys.modules)  
**MemoryRegion**: start_address, size, protection, type_name  
**MinidumpData**: Complete dump container

### Security Features

- **Path validation**: Null byte injection check, directory traversal protection
- **Allowed directories**: `/tmp`, `/var/log`, `/var/crash`, `$HOME`, Windows AppData/temp
- **Environment sanitization**: 30+ secret patterns excluded (API_KEY, TOKEN, PASSWORD, etc.)
- **Secret detection heuristics**: Long hex strings, base64-encoded values redacted
- **Exact name exclusions**: HOME, USER, LOGNAME, MAIL, HOSTNAME

### Platform Support

| Feature | Linux | Windows | macOS |
|---------|-------|---------|-------|
| Memory regions | `/proc/self/maps` parsing | Stub | Stub |
| Thread stack traces | Real via `sys._current_frames()` | Real | Real |
| Module info | Real via `sys.modules` | Real | Real |
| Native minidump | Stub (ptrace) | Stub (MiniDumpWriteDump) | Stub (MachExceptionHandler) |

### Output Format

- JSON format for Python-level dumps
- Platform-specific binary would require native extensions

---

## 4. Crash Reporter (`reporter.py`)

### Key Components

**SystemInfoSnapshot**
- OS, version, architecture
- CPU info, core count
- Memory total/available (Linux: `/proc/meminfo`, others: stub)
- GPU info: stub
- Display info: stub

**CrashReport**
- Unique UUID report_id
- CrashContext reference
- SystemInfoSnapshot
- Game/build metadata
- Optional screenshot path
- Optional minidump path
- Custom data dictionary
- SHA256 fingerprint for deduplication (based on exception type + top 10 stack lines)

**CrashReporter**
- Configurable: game_version, build_id, build_type
- Platform-specific default reports directory
- Custom data providers (callables returning dict)
- Upload callbacks for notification

### Features

**Local Storage**
- Atomic write via temp file + `os.replace()`
- Path sanitization (alphanumeric report_id only)
- JSON format

**Remote Upload**
- Async implementation (`async def upload()`)
- Synchronous wrapper for non-async contexts
- Exponential backoff with max 3 retries
- Payload size limits (10 MB max, 500 stack trace lines, 100 recent logs)
- HTTP/HTTPS endpoint validation
- Timeout handling (30s default)
- Currently stub (simulates success after 0.1s)

**Report Management**
- `get_pending_reports()`: List unsent reports
- `load_report()`: Deserialize from disk
- `cleanup_old_reports()`: Delete reports older than N days (default 30)

### Security

- Report ID sanitization prevents path injection
- Null byte checks
- Size limits prevent DoS

---

## Integration Points

### Global Singleton Pattern

```python
# Handler
from engine.debug.crash import install_global_handler
handler = install_global_handler()
handler.on_crash(my_callback)

# Reporter
from engine.debug.crash import configure_global_reporter
reporter = configure_global_reporter(
    game_version="1.2.3",
    build_id="abc123",
    build_type="release"
)

# Assertions
from engine.debug.crash import check, ensure, set_assertion_response, AssertionResponse
set_assertion_response(AssertionResponse.LOG)  # For development
```

### Crash Flow

1. Signal/exception triggers CrashHandler
2. CrashContext captured with stack trace, logs, thread info
3. Registered callbacks invoked (error-tolerant)
4. CrashReporter creates CrashReport with system snapshot
5. Minidump generated (Python-level)
6. Report saved locally
7. Async upload attempted with retries

---

## Stubs Requiring Implementation

| Component | Status | Implementation Notes |
|-----------|--------|---------------------|
| Native minidump (Windows) | Stub | Requires ctypes/MiniDumpWriteDump |
| Native minidump (Linux) | Stub | Requires ptrace or /proc coredump |
| Native minidump (macOS) | Stub | Requires MachExceptionHandler |
| GPU info | Stub | Platform-specific (lspci, WMI, IOKit) |
| Memory info (Windows) | Stub | GlobalMemoryStatusEx |
| Memory info (macOS) | Stub | sysctl or vm_stat |
| Screenshot capture | Stub | Requires rendering system integration |
| Remote upload | Stub | aiohttp implementation needed |

---

## Code Quality Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| Error handling | Excellent | Try/except throughout, graceful degradation |
| Thread safety | Excellent | Proper locking, re-entrancy protection |
| Security | Excellent | Path validation, secret redaction, size limits |
| Documentation | Excellent | Comprehensive docstrings, usage examples |
| Testability | Good | Context managers for assertions, callbacks |
| Type hints | Good | TypeVar usage, Optional annotations |
| Constants | Excellent | Named constants at module level |

---

## Recommendations

1. **Native minidump integration**: Consider calling out to platform tools (gcore on Linux, procdump on Windows) as interim solution
2. **Upload implementation**: Replace asyncio.sleep stub with actual aiohttp POST
3. **GPU detection**: Use subprocess to call lspci/wmic/system_profiler
4. **Screenshot capture**: Hook into rendering system's swapchain present
5. **Telemetry consent**: Add user consent mechanism before upload

---

## Public API Summary

### Assertions
- `check()`, `checkf()`, `verify()`, `ensure()`, `ensureAlways()`, `checkSlow()`, `checkSlowf()`
- `set_assertion_response()`, `get_assertion_response()`
- `set_debug_build()`, `is_debug_build()`
- `set_log_callback()`, `set_crash_callback()`
- `reset_logged_assertions()`
- `AssertionResponse`, `AssertionContext`

### Handler
- `CrashContext`, `CrashCallback`, `CrashHandler`, `RecentLogHandler`
- `get_global_handler()`, `install_global_handler()`

### Minidump
- `MinidumpLevel`, `ThreadInfo`, `ModuleInfo`, `MemoryRegion`, `MinidumpData`, `Minidump`
- `generate_crash_dump()`, `get_current_stack_trace()`

### Reporter
- `SystemInfoSnapshot`, `CrashReport`, `CrashReporter`
- `get_global_reporter()`, `configure_global_reporter()`
