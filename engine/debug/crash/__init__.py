"""
Crash Handling System for the game engine.

This module provides comprehensive crash handling capabilities:
- Runtime assertions with configurable behavior
- Signal and exception handling
- Minidump generation for crash diagnostics
- Crash reporting and telemetry

Usage:
    >>> from engine.debug.crash import check, checkf, ensure, CrashHandler
    >>>
    >>> # Install crash handler
    >>> handler = CrashHandler()
    >>> handler.install()
    >>> handler.on_crash(my_callback)
    >>>
    >>> # Use assertions
    >>> check(player is not None)
    >>> checkf(health > 0, "Health must be positive, got %d", health)
    >>>
    >>> # Non-fatal assertions
    >>> if not ensure(data is not None):
    ...     return default_value
"""

# Assertions
from .assertions import (
    AssertionContext,
    AssertionResponse,
    check,
    checkf,
    checkSlow,
    checkSlowf,
    ensure,
    ensureAlways,
    get_assertion_response,
    is_debug_build,
    reset_logged_assertions,
    set_assertion_response,
    set_crash_callback,
    set_debug_build,
    set_log_callback,
    verify,
)

# Crash Handler
from .handler import (
    CrashCallback,
    CrashContext,
    CrashHandler,
    RecentLogHandler,
    get_global_handler,
    install_global_handler,
)

# Minidump
from .minidump import (
    MemoryRegion,
    Minidump,
    MinidumpData,
    MinidumpLevel,
    ModuleInfo,
    ThreadInfo,
    generate_crash_dump,
    get_current_stack_trace,
)

# Reporter
from .reporter import (
    CrashReport,
    CrashReporter,
    SystemInfoSnapshot,
    configure_global_reporter,
    get_global_reporter,
)

__all__ = [
    # Assertions
    'AssertionResponse',
    'AssertionContext',
    'set_assertion_response',
    'get_assertion_response',
    'set_debug_build',
    'is_debug_build',
    'set_log_callback',
    'set_crash_callback',
    'reset_logged_assertions',
    'check',
    'checkf',
    'verify',
    'ensure',
    'ensureAlways',
    'checkSlow',
    'checkSlowf',
    # Crash Handler
    'CrashContext',
    'CrashCallback',
    'CrashHandler',
    'RecentLogHandler',
    'get_global_handler',
    'install_global_handler',
    # Minidump
    'MinidumpLevel',
    'ThreadInfo',
    'ModuleInfo',
    'MemoryRegion',
    'MinidumpData',
    'Minidump',
    'generate_crash_dump',
    'get_current_stack_trace',
    # Reporter
    'SystemInfoSnapshot',
    'CrashReport',
    'CrashReporter',
    'get_global_reporter',
    'configure_global_reporter',
]
