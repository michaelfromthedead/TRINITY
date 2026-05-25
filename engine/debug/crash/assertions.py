"""
Runtime assertion system for the game engine.

Provides various assertion macros with configurable behavior including:
- check/checkf: Fatal assertions that halt execution
- verify: Fatal assertion that returns the value
- ensure/ensureAlways: Non-fatal assertions that log and return bool
- checkSlow: Debug-only assertions

The behavior of assertions can be configured globally via set_assertion_response().
"""

import logging
import os
import sys
import traceback
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, TypeVar

T = TypeVar('T')

# Module-level logger
_logger = logging.getLogger(__name__)


class AssertionResponse(Enum):
    """
    Configurable response behavior for failed assertions.

    BREAK: Trigger debugger breakpoint (if available)
    LOG: Log the failure and continue
    CRASH: Terminate the application
    CONTINUE: Silently continue (not recommended)
    """
    BREAK = auto()
    LOG = auto()
    CRASH = auto()
    CONTINUE = auto()


@dataclass
class AssertionConfig:
    """
    Global configuration for assertion behavior.

    Attributes:
        response: Default response to assertion failures
        is_debug_build: Whether this is a debug build (enables slow checks)
        log_callback: Optional custom logging callback
        crash_callback: Optional callback invoked before crashing
        logged_assertions: Set of assertion locations that have already logged (for ensure)
    """
    response: AssertionResponse = AssertionResponse.CRASH
    is_debug_build: bool = True
    log_callback: Optional[Callable[[str, str, int, str], None]] = None
    crash_callback: Optional[Callable[[str, str, int], None]] = None
    logged_assertions: set = field(default_factory=set)

    def __post_init__(self):
        # Auto-detect debug build from environment
        if os.environ.get('GAME_ENGINE_DEBUG', '').lower() in ('1', 'true', 'yes'):
            self.is_debug_build = True
        elif os.environ.get('GAME_ENGINE_RELEASE', '').lower() in ('1', 'true', 'yes'):
            self.is_debug_build = False


# Global configuration instance
_config = AssertionConfig()


def set_assertion_response(response: AssertionResponse) -> None:
    """
    Set the global assertion response behavior.

    Args:
        response: The AssertionResponse to use for failed assertions

    Example:
        >>> set_assertion_response(AssertionResponse.LOG)
        >>> check(False)  # Will log instead of crashing
    """
    global _config
    _config.response = response


def get_assertion_response() -> AssertionResponse:
    """
    Get the current assertion response behavior.

    Returns:
        The current AssertionResponse setting
    """
    return _config.response


def set_debug_build(is_debug: bool) -> None:
    """
    Set whether this is a debug build.

    Debug builds enable checkSlow assertions.

    Args:
        is_debug: True for debug build, False for release
    """
    global _config
    _config.is_debug_build = is_debug


def is_debug_build() -> bool:
    """
    Check if this is a debug build.

    Returns:
        True if debug build, False otherwise
    """
    return _config.is_debug_build


def set_log_callback(callback: Optional[Callable[[str, str, int, str], None]]) -> None:
    """
    Set a custom logging callback for assertion failures.

    Args:
        callback: Function(filename, function, line, message) or None to use default
    """
    global _config
    _config.log_callback = callback


def set_crash_callback(callback: Optional[Callable[[str, str, int], None]]) -> None:
    """
    Set a callback invoked just before crashing on assertion failure.

    Useful for cleanup or crash reporting.

    Args:
        callback: Function(filename, function, line) or None
    """
    global _config
    _config.crash_callback = callback


def reset_logged_assertions() -> None:
    """
    Clear the set of logged assertion locations.

    This allows ensure() to log the same assertion again.
    Primarily useful for testing.
    """
    global _config
    _config.logged_assertions.clear()


def _get_caller_info(stack_depth: int = 2) -> tuple[str, str, int]:
    """
    Get information about the calling code.

    Args:
        stack_depth: How many frames up to look

    Returns:
        Tuple of (filename, function_name, line_number)
    """
    frame = sys._getframe(stack_depth)
    return (
        frame.f_code.co_filename,
        frame.f_code.co_name,
        frame.f_lineno
    )


def _format_assertion_message(
    condition_str: str,
    filename: str,
    function: str,
    line: int,
    message: Optional[str] = None
) -> str:
    """
    Format an assertion failure message.

    Args:
        condition_str: String representation of the failed condition
        filename: Source file name
        function: Function name
        line: Line number
        message: Optional additional message

    Returns:
        Formatted assertion message
    """
    base = f"Assertion failed: {condition_str}\n  at {filename}:{line} in {function}()"
    if message:
        base += f"\n  Message: {message}"
    return base


def _handle_assertion_failure(
    condition_str: str,
    message: Optional[str] = None,
    fatal: bool = True,
    stack_depth: int = 3
) -> None:
    """
    Handle an assertion failure according to current configuration.

    Args:
        condition_str: String representation of the failed condition
        message: Optional additional message
        fatal: If True and response is CRASH, terminate
        stack_depth: Stack frames to skip when getting caller info
    """
    filename, function, line = _get_caller_info(stack_depth)
    full_message = _format_assertion_message(condition_str, filename, function, line, message)

    # Custom log callback
    if _config.log_callback:
        _config.log_callback(filename, function, line, full_message)
    else:
        _logger.error(full_message)

    response = _config.response

    if response == AssertionResponse.BREAK:
        # Try to trigger debugger breakpoint
        try:
            # Python 3.7+ has breakpoint()
            breakpoint()
        except Exception:
            # Fallback: log that we couldn't break
            _logger.warning("Could not trigger debugger breakpoint")

    elif response == AssertionResponse.CRASH and fatal:
        # Invoke crash callback if set
        if _config.crash_callback:
            try:
                _config.crash_callback(filename, function, line)
            except Exception as e:
                _logger.error(f"Crash callback failed: {e}")

        # Print stack trace
        _logger.critical("Stack trace:\n" + "".join(traceback.format_stack()[:-2]))

        # Terminate
        sys.exit(1)

    elif response == AssertionResponse.LOG:
        # Already logged above
        pass

    elif response == AssertionResponse.CONTINUE:
        # Silently continue
        pass


def check(condition: bool) -> None:
    """
    Fatal assertion that halts execution if condition is False.

    This is the most basic assertion. Use when the condition must be true
    for the program to continue correctly.

    Args:
        condition: The condition to check

    Raises:
        SystemExit: If condition is False and response is CRASH

    Example:
        >>> check(player is not None)
        >>> check(health > 0)
    """
    if not condition:
        _handle_assertion_failure("condition", fatal=True)


def checkf(condition: bool, message: str, *args: Any) -> None:
    """
    Fatal assertion with a formatted message.

    Like check(), but allows a custom message with format arguments.

    Args:
        condition: The condition to check
        message: Format string for the error message
        *args: Arguments for the format string

    Raises:
        SystemExit: If condition is False and response is CRASH

    Example:
        >>> checkf(index >= 0, "Index must be non-negative, got %d", index)
        >>> checkf(player.health > 0, "Player %s is dead", player.name)
    """
    if not condition:
        formatted_message = message % args if args else message
        _handle_assertion_failure("condition", message=formatted_message, fatal=True)


def verify(condition: T) -> T:
    """
    Fatal assertion that returns the checked value.

    Useful when you want to both check and use a value in one expression.
    The condition is evaluated for truthiness.

    Args:
        condition: The value to verify and return

    Returns:
        The condition value if truthy

    Raises:
        SystemExit: If condition is falsy and response is CRASH

    Example:
        >>> player = verify(get_player())  # Crashes if None
        >>> item = verify(inventory.pop())  # Crashes if empty
    """
    if not condition:
        _handle_assertion_failure("verify(condition)", fatal=True)
    return condition


def ensure(condition: bool) -> bool:
    """
    Non-fatal assertion that logs once and returns the result.

    Unlike check(), this does not halt execution. It logs the failure
    only the first time it occurs at each location (based on file:line).

    Args:
        condition: The condition to check

    Returns:
        True if condition is truthy, False otherwise

    Example:
        >>> if not ensure(data is not None):
        ...     return default_value
    """
    if not condition:
        filename, function, line = _get_caller_info(2)
        location_key = f"{filename}:{line}"

        if location_key not in _config.logged_assertions:
            _config.logged_assertions.add(location_key)
            _handle_assertion_failure(
                "ensure(condition)",
                message="(logged once)",
                fatal=False,
                stack_depth=3
            )

    return bool(condition)


def ensureAlways(condition: bool) -> bool:
    """
    Non-fatal assertion that logs every time and returns the result.

    Unlike ensure(), this logs every time the condition fails,
    not just the first time.

    Args:
        condition: The condition to check

    Returns:
        True if condition is truthy, False otherwise

    Example:
        >>> if not ensureAlways(connection.is_alive()):
        ...     reconnect()
    """
    if not condition:
        _handle_assertion_failure(
            "ensureAlways(condition)",
            fatal=False,
            stack_depth=3
        )

    return bool(condition)


def checkSlow(condition: bool) -> None:
    """
    Debug-only assertion that is compiled out in release builds.

    Use for expensive checks that should not run in production.
    In release builds, this function does nothing.

    Args:
        condition: The condition to check (only evaluated in debug builds)

    Raises:
        SystemExit: If condition is False in debug build and response is CRASH

    Example:
        >>> checkSlow(validate_entire_scene_graph())  # Expensive check
        >>> checkSlow(verify_memory_integrity())  # Debug-only
    """
    if _config.is_debug_build and not condition:
        _handle_assertion_failure("checkSlow(condition)", fatal=True)


def checkSlowf(condition: bool, message: str, *args: Any) -> None:
    """
    Debug-only assertion with a formatted message.

    Combination of checkSlow and checkf. Only active in debug builds.

    Args:
        condition: The condition to check
        message: Format string for the error message
        *args: Arguments for the format string

    Raises:
        SystemExit: If condition is False in debug build and response is CRASH
    """
    if _config.is_debug_build and not condition:
        formatted_message = message % args if args else message
        _handle_assertion_failure("checkSlow(condition)", message=formatted_message, fatal=True)


class AssertionContext:
    """
    Context manager for temporarily changing assertion behavior.

    Useful for tests or controlled sections of code.

    Example:
        >>> with AssertionContext(AssertionResponse.LOG):
        ...     # Assertions will log instead of crashing
        ...     check(False)  # Logs but doesn't crash
        >>> # Original behavior restored
    """

    def __init__(self, response: AssertionResponse):
        """
        Initialize the context with a new assertion response.

        Args:
            response: The AssertionResponse to use within the context
        """
        self._new_response = response
        self._old_response: Optional[AssertionResponse] = None

    def __enter__(self) -> 'AssertionContext':
        """Enter the context, saving and replacing the response."""
        self._old_response = get_assertion_response()
        set_assertion_response(self._new_response)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context, restoring the original response."""
        if self._old_response is not None:
            set_assertion_response(self._old_response)


# Export public API
__all__ = [
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
]
