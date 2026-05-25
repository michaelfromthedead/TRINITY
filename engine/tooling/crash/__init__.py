"""
Crash reporting subsystem for the AI Game Engine.

Provides comprehensive crash reporting infrastructure including:
- Crash reporter with stack traces and minidumps
- Custom assertions with @invariant decorator
- Crash upload to server
- Symbol server for debug symbols
- Crash analytics and grouping
"""

from .crash_reporter import (
    CrashReporter,
    CrashReport,
    CrashSeverity,
    CrashContext,
    ExceptionInfo,
    SystemInfo,
    initialize_crash_reporter,
    report_crash,
    capture_exception,
)

from .assertions import (
    invariant,
    precondition,
    postcondition,
    check,
    ensure,
    require,
    InvariantError,
    PreconditionError,
    PostconditionError,
    AssertionConfig,
    enable_assertions,
    disable_assertions,
)

from .crash_upload import (
    CrashUploader,
    UploadResult,
    UploadConfig,
    upload_crash_report,
    upload_minidump,
    upload_async,
)

from .symbol_server import (
    SymbolServer,
    SymbolInfo,
    SymbolCache,
    resolve_symbol,
    lookup_address,
    symbolicate_stack,
)

from .crash_analytics import (
    CrashAnalytics,
    CrashGroup,
    CrashPattern,
    CrashTrend,
    analyze_crash,
    group_crashes,
    get_crash_statistics,
    detect_patterns,
)

__all__ = [
    # Crash reporter
    "CrashReporter",
    "CrashReport",
    "CrashSeverity",
    "CrashContext",
    "ExceptionInfo",
    "SystemInfo",
    "initialize_crash_reporter",
    "report_crash",
    "capture_exception",
    # Assertions
    "invariant",
    "precondition",
    "postcondition",
    "check",
    "ensure",
    "require",
    "InvariantError",
    "PreconditionError",
    "PostconditionError",
    "AssertionConfig",
    "enable_assertions",
    "disable_assertions",
    # Crash upload
    "CrashUploader",
    "UploadResult",
    "UploadConfig",
    "upload_crash_report",
    "upload_minidump",
    "upload_async",
    # Symbol server
    "SymbolServer",
    "SymbolInfo",
    "SymbolCache",
    "resolve_symbol",
    "lookup_address",
    "symbolicate_stack",
    # Crash analytics
    "CrashAnalytics",
    "CrashGroup",
    "CrashPattern",
    "CrashTrend",
    "analyze_crash",
    "group_crashes",
    "get_crash_statistics",
    "detect_patterns",
]
