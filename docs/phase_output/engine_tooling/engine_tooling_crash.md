# Engine Tooling Crash Investigation

**Path:** `engine/tooling/crash/`
**Total Lines:** 2,757
**Classification:** REAL (fully implemented with working logic)

## Summary

The crash reporting system is a **REAL, production-quality implementation** providing comprehensive crash reporting infrastructure for the AI Game Engine. All five modules contain complete, functional implementations with proper data structures, error handling, and integration points.

## Files Analyzed

| File | Lines | Classification | Purpose |
|------|-------|----------------|---------|
| `crash_reporter.py` | 564 | REAL | Core crash capture and reporting |
| `assertions.py` | 497 | REAL | Design-by-contract assertions |
| `crash_analytics.py` | 597 | REAL | Crash grouping and pattern detection |
| `crash_upload.py` | 492 | REAL | Server upload with retry/compression |
| `symbol_server.py` | 492 | REAL | Debug symbol resolution |
| `__init__.py` | 115 | REAL | Module exports |

## Module Details

### 1. crash_reporter.py (564 lines) - REAL

**Purpose:** Core crash reporting with stack traces and system information capture.

**Key Classes:**
- `CrashSeverity` (Enum): INFO, WARNING, ERROR, CRITICAL, FATAL levels
- `SystemInfo`: Captures OS, CPU, memory, Python version, process/thread IDs
- `StackFrame`: Individual stack frame with filename, line, function, locals
- `ExceptionInfo`: Full exception info with chained exception support
- `CrashContext`: User/session ID, build info, game state, user actions, logs
- `CrashReport`: Complete crash report with fingerprinting for deduplication
- `CrashReporter`: Main reporter with hooks, filters, auto-upload

**Key Features:**
- Exception capture with full traceback extraction (line 144-151)
- Chained exception support (`__cause__`, `__context__`) (line 159-163)
- Fingerprint generation for crash grouping using MD5 hash (line 244-254)
- JSON serialization and file persistence (line 272-287)
- Global exception handler installation via `sys.excepthook` (line 479-487)
- Configurable hooks and filters for custom processing
- Auto-upload capability to remote server

**Implementation Quality:** Production-ready with proper error handling.

### 2. assertions.py (497 lines) - REAL

**Purpose:** Design-by-contract style assertions with decorators.

**Key Classes:**
- `InvariantError`, `PreconditionError`, `PostconditionError`: Typed exceptions
- `AssertionConfig`: Global configuration for enabling/disabling checks
- `ContractMixin`: Mixin class for adding contract support to classes

**Key Decorators:**
- `@invariant(condition, message)`: Class invariant checked after __init__ and methods
- `@precondition(condition, message, parameter)`: Function precondition checks
- `@postcondition(condition, result_check)`: Function result validation
- `@with_contracts(invariant_func)`: Full contract support decorator

**Key Functions:**
- `check(condition, message)`: Runtime check with configurable exception
- `ensure(condition, message)`: Postcondition-style check
- `require(condition, message)`: Precondition-style check
- `assert_type(value, expected_type)`: Type checking assertion
- `assert_not_none(value, message)`: Null check assertion
- `assert_in_range(value, min, max)`: Range validation

**Implementation Quality:** Complete DBC implementation with proper decorator mechanics.

### 3. crash_analytics.py (597 lines) - REAL

**Purpose:** Crash grouping, pattern detection, and trend analysis.

**Key Classes:**
- `CrashGroup`: Group of similar crashes by fingerprint with metadata
- `CrashPattern`: Identified patterns (stack_trace, exception, context, temporal)
- `CrashTrend`: Time-series trend data with direction indicators
- `CrashAnalytics`: Main analytics engine

**Key Features:**
- **Crash Grouping:** Groups by fingerprint with version/platform tracking (line 172-212)
- **Message Pattern Extraction:** Normalizes error messages with placeholders (line 214-233)
  - Replaces hex addresses with `<addr>`
  - Replaces numbers with `<num>`
  - Replaces quoted strings with `<str>`
  - Replaces file paths with `<path>`
- **Pattern Detection:**
  - Stack patterns: Functions appearing in 3+ crashes (line 397-422)
  - Version patterns: Versions with >30% crash share (line 425-455)
  - Time patterns: Hours with 2x average crashes (line 458-488)
- **Trend Analysis:** Hourly/daily/weekly trends with direction (line 268-323)

**Implementation Quality:** Sophisticated analytics with real algorithms.

### 4. crash_upload.py (492 lines) - REAL

**Purpose:** Upload crash reports to collection servers.

**Key Classes:**
- `UploadStatus` (Enum): PENDING, IN_PROGRESS, SUCCESS, FAILED, RETRY
- `UploadResult`: Upload outcome with server ID, duration, retry count
- `UploadConfig`: Server URL, API key, compression, timeouts, rate limits
- `CrashUploader`: Synchronous uploader with retry logic
- `AsyncCrashUploader`: Async version with concurrent uploads

**Key Features:**
- **Gzip Compression:** Optional payload compression (line 140-144)
- **Rate Limiting:** Configurable requests per second (line 127-138)
- **Retry Logic:** Exponential backoff for 429/503 errors (line 240-243)
- **Batch Uploads:** Queue and flush pattern (line 351-367)
- **Minidump Support:** Binary file upload with octet-stream (line 270-349)
- **Async Support:** `asyncio` integration with semaphore concurrency (line 400-425)
- **SSL Configuration:** Optional certificate verification (line 176-181)
- **Environment Config:** `CRASH_SERVER_URL`, `CRASH_API_KEY`, `CRASH_PROJECT_ID`

**Implementation Quality:** Production-ready with proper HTTP handling.

### 5. symbol_server.py (492 lines) - REAL

**Purpose:** Debug symbol management and stack symbolication.

**Key Classes:**
- `SymbolInfo`: Symbol with address, name, module, source location, offset
- `ModuleInfo`: Loaded module with base address, build ID, debug info
- `SymbolCache`: LRU cache with TTL for symbol lookups
- `SymbolServer`: Main server for symbol resolution

**Key Features:**
- **Symbol File Formats:**
  - JSON format with `symbols` array (line 233-245)
  - Text format: `address name [filename:line]` (line 247-287)
  - Auto-detection based on extension (line 216-228)
- **Address Resolution:**
  - Exact match lookup (line 341-342)
  - Nearest lower address with offset calculation (line 344-365)
- **Symbol Caching:** 
  - Configurable max size and TTL (line 112-113)
  - LRU eviction of oldest 10% (line 141-151)
- **Stack Symbolication:** Batch resolution of stack frames (line 380-413)
- **Module Registration:** Track loaded modules for resolution (line 193-195)

**Implementation Quality:** Complete symbolication system with caching.

## Architecture

```
engine/tooling/crash/
    __init__.py           # Exports all public APIs
    crash_reporter.py     # Exception capture -> CrashReport
    assertions.py         # DBC decorators -> InvariantError/etc
    crash_analytics.py    # CrashReport -> CrashGroup/Pattern/Trend
    crash_upload.py       # CrashReport -> Server
    symbol_server.py      # Address -> SymbolInfo
```

**Data Flow:**
1. Exception occurs or manual `report_crash()` called
2. `CrashReporter` captures exception, system info, context
3. Creates `CrashReport` with fingerprint for deduplication
4. Optionally uploads via `CrashUploader` with retry/compression
5. `CrashAnalytics` groups by fingerprint, detects patterns
6. `SymbolServer` resolves addresses for native crashes

## Key Design Patterns

1. **Singleton Pattern:** Global instances with lazy initialization
   - `_reporter`, `_analytics`, `_uploader`, `_server` globals
   - `get_*()` accessors create on first use

2. **Decorator Pattern:** Contract decorators wrap functions/classes
   - `@invariant`, `@precondition`, `@postcondition`
   - Preserve function signatures via `functools.wraps`

3. **Observer Pattern:** Hook system for crash events
   - `add_hook(callback)` for custom processing
   - `add_callback()` for upload completion

4. **Template Method:** Common HTTP handling in uploader
   - `_build_request()`, `_send_request()` factored out
   - Subclasses add async behavior

5. **Cache-Aside:** Symbol caching with explicit get/put
   - Check cache -> miss -> load -> cache -> return

## Dependencies

**Standard Library Only:**
- `hashlib`, `json`, `traceback`, `uuid`, `time`
- `urllib.request`, `urllib.parse`, `urllib.error`
- `asyncio`, `gzip`, `ssl`, `struct`
- `dataclasses`, `enum`, `typing`, `functools`, `inspect`

**Optional:**
- `psutil`: Memory information in SystemInfo (graceful fallback)

## Integration Points

1. **Global Exception Handler:** `install_exception_handler()` hooks `sys.excepthook`
2. **Context Updates:** `set_context()`, `log_action()`, `log_message()` for breadcrumbs
3. **Filter System:** `add_filter()` for selective reporting (privacy, noise)
4. **Hook System:** `add_hook()` for integration with alerting, logging
5. **Environment Variables:** Server config via `CRASH_*` env vars

## Testing Considerations

- All classes are well-structured for unit testing
- Global state can be reset via re-initialization
- `disable_assertions()` for performance-critical paths
- Mock HTTP for upload testing
- Symbol files are simple text/JSON for test fixtures

## Classification Rationale

**REAL Implementation Evidence:**
1. Complete data structures with all fields populated
2. Working algorithms (fingerprinting, pattern detection, trend analysis)
3. Proper error handling throughout
4. HTTP client implementation with retry logic
5. File I/O for symbol loading and crash persistence
6. Async support with proper asyncio patterns
7. Configuration via environment variables
8. Integration hooks for extensibility

**No Stub Indicators:**
- No `pass` or `raise NotImplementedError`
- No placeholder comments like "TODO: implement"
- All methods have meaningful implementations
- Return values match declared types
