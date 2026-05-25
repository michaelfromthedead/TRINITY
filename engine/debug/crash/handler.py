"""
Crash handler for the game engine.

Provides centralized crash handling with:
- Signal handling for SIGSEGV, SIGABRT, etc.
- Crash context capture (exception, stack trace, recent logs)
- Callback registration for crash notifications
- Thread-safe crash state management
"""

import atexit
import logging
import signal
import sys
import threading
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Deque, List, Optional, Set

# Module-level logger
_logger = logging.getLogger(__name__)


# Configuration constants
DEFAULT_MAX_RECENT_LOGS = 100
MAX_STACK_TRACE_DEPTH = 50
MAX_CALLBACK_EXECUTION_TIME_SECONDS = 5.0


@dataclass
class CrashContext:
    """
    Context information captured when a crash occurs.

    Attributes:
        exception: The exception that caused the crash (if any)
        stack_trace: Formatted stack trace string
        recent_logs: Recent log messages leading up to the crash
        timestamp: When the crash occurred
        thread_id: Thread where the crash occurred
        thread_name: Name of the thread where the crash occurred
        signal_number: Signal number if crash was caused by a signal
        signal_name: Human-readable signal name
        additional_data: Dictionary for any extra crash data
    """
    exception: Optional[BaseException] = None
    stack_trace: str = ""
    recent_logs: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    thread_id: int = 0
    thread_name: str = ""
    signal_number: Optional[int] = None
    signal_name: Optional[str] = None
    additional_data: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.thread_id == 0:
            self.thread_id = threading.current_thread().ident or 0
        if not self.thread_name:
            self.thread_name = threading.current_thread().name


# Type alias for crash callbacks
CrashCallback = Callable[[CrashContext], None]


class RecentLogHandler(logging.Handler):
    """
    Logging handler that keeps a buffer of recent log messages.

    Used by CrashHandler to capture logs leading up to a crash.
    """

    def __init__(self, max_entries: int = DEFAULT_MAX_RECENT_LOGS):
        """
        Initialize the handler with a maximum buffer size.

        Args:
            max_entries: Maximum number of log entries to retain
        """
        super().__init__()
        self._max_entries = max_entries
        self._buffer: Deque[str] = deque(maxlen=max_entries)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        """
        Store a log record in the buffer.

        Args:
            record: The log record to store
        """
        try:
            msg = self.format(record)
            with self._lock:
                self._buffer.append(msg)
        except Exception:
            self.handleError(record)

    def get_recent_logs(self) -> List[str]:
        """
        Get a copy of the recent log messages.

        Returns:
            List of recent log messages
        """
        with self._lock:
            return list(self._buffer)

    def clear(self) -> None:
        """Clear the log buffer."""
        with self._lock:
            self._buffer.clear()


class CrashHandler:
    """
    Centralized crash handler for the game engine.

    Handles signal-based crashes, uncaught exceptions, and provides
    crash context capture and callback notification.

    Usage:
        >>> handler = CrashHandler()
        >>> handler.install()
        >>> handler.on_crash(my_crash_callback)
        >>> # ... later, on crash ...
        >>> handler.uninstall()
    """

    # Signals to handle (platform-specific availability)
    HANDLED_SIGNALS = [
        signal.SIGINT,   # Interrupt (Ctrl+C)
        signal.SIGTERM,  # Termination request
        signal.SIGABRT,  # Abort
    ]

    # Add Unix-specific signals if available
    if hasattr(signal, 'SIGSEGV'):
        HANDLED_SIGNALS.append(signal.SIGSEGV)  # Segmentation fault
    if hasattr(signal, 'SIGBUS'):
        HANDLED_SIGNALS.append(signal.SIGBUS)   # Bus error
    if hasattr(signal, 'SIGFPE'):
        HANDLED_SIGNALS.append(signal.SIGFPE)   # Floating point exception
    if hasattr(signal, 'SIGILL'):
        HANDLED_SIGNALS.append(signal.SIGILL)   # Illegal instruction

    def __init__(self, max_recent_logs: int = DEFAULT_MAX_RECENT_LOGS):
        """
        Initialize the crash handler.

        Args:
            max_recent_logs: Maximum number of recent log messages to retain
        """
        self._callbacks: List[CrashCallback] = []
        self._original_handlers: dict[int, Any] = {}
        self._original_excepthook: Optional[Callable] = None
        self._installed = False
        self._handling_crash = False
        self._lock = threading.Lock()

        # Set up log buffer
        self._log_handler = RecentLogHandler(max_recent_logs)
        self._log_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )

        # Thread-local storage for crash context
        self._crash_context: Optional[CrashContext] = None

    def install(self) -> None:
        """
        Install the crash handler.

        Registers signal handlers and exception hook.
        Safe to call multiple times (idempotent).
        """
        with self._lock:
            if self._installed:
                _logger.debug("CrashHandler already installed")
                return

            # Install signal handlers
            for sig in self.HANDLED_SIGNALS:
                try:
                    self._original_handlers[sig] = signal.signal(sig, self._signal_handler)
                    _logger.debug(f"Installed handler for signal {sig}")
                except (OSError, ValueError) as e:
                    # Some signals can't be caught (e.g., SIGKILL, SIGSTOP)
                    # or may not be available on this platform
                    _logger.debug(f"Could not install handler for signal {sig}: {e}")

            # Install exception hook
            self._original_excepthook = sys.excepthook
            sys.excepthook = self._exception_handler

            # Add log handler to root logger
            logging.root.addHandler(self._log_handler)

            # Register atexit handler for cleanup
            atexit.register(self._atexit_handler)

            self._installed = True
            _logger.info("CrashHandler installed")

    def uninstall(self) -> None:
        """
        Uninstall the crash handler.

        Restores original signal handlers and exception hook.
        Safe to call multiple times (idempotent).
        """
        with self._lock:
            if not self._installed:
                _logger.debug("CrashHandler not installed, nothing to uninstall")
                return

            # Restore signal handlers
            for sig, handler in self._original_handlers.items():
                try:
                    signal.signal(sig, handler)
                    _logger.debug(f"Restored handler for signal {sig}")
                except (OSError, ValueError) as e:
                    _logger.debug(f"Could not restore handler for signal {sig}: {e}")
            self._original_handlers.clear()

            # Restore exception hook
            if self._original_excepthook:
                sys.excepthook = self._original_excepthook
                self._original_excepthook = None

            # Remove log handler
            logging.root.removeHandler(self._log_handler)

            # Unregister atexit handler
            try:
                atexit.unregister(self._atexit_handler)
            except Exception:
                pass

            self._installed = False
            _logger.info("CrashHandler uninstalled")

    def on_crash(self, callback: CrashCallback) -> None:
        """
        Register a callback to be called when a crash occurs.

        Callbacks receive a CrashContext with crash information.
        Multiple callbacks can be registered; they are called in order.

        Args:
            callback: Function that takes a CrashContext

        Example:
            >>> def my_callback(ctx: CrashContext):
            ...     print(f"Crash at {ctx.timestamp}: {ctx.exception}")
            >>> handler.on_crash(my_callback)
        """
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)
                _logger.debug(f"Registered crash callback: {callback}")

    def remove_callback(self, callback: CrashCallback) -> bool:
        """
        Remove a previously registered crash callback.

        Args:
            callback: The callback to remove

        Returns:
            True if the callback was found and removed, False otherwise
        """
        with self._lock:
            try:
                self._callbacks.remove(callback)
                _logger.debug(f"Removed crash callback: {callback}")
                return True
            except ValueError:
                return False

    def capture_state(self,
                      exception: Optional[BaseException] = None,
                      signal_number: Optional[int] = None) -> CrashContext:
        """
        Capture current state as a CrashContext.

        Can be called at any time to get a snapshot of the current state.

        Args:
            exception: The exception that triggered the capture (if any)
            signal_number: Signal number if crash was signal-based

        Returns:
            CrashContext with current state information
        """
        # Get stack trace
        if exception:
            stack_trace = "".join(traceback.format_exception(
                type(exception), exception, exception.__traceback__
            ))
        else:
            stack_trace = "".join(traceback.format_stack())

        # Get signal name if applicable
        signal_name = None
        if signal_number is not None:
            try:
                signal_name = signal.Signals(signal_number).name
            except (ValueError, AttributeError):
                signal_name = f"UNKNOWN({signal_number})"

        context = CrashContext(
            exception=exception,
            stack_trace=stack_trace,
            recent_logs=self._log_handler.get_recent_logs(),
            timestamp=datetime.now(),
            thread_id=threading.current_thread().ident or 0,
            thread_name=threading.current_thread().name,
            signal_number=signal_number,
            signal_name=signal_name,
        )

        return context

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """
        Internal signal handler.

        Called when a registered signal is received.
        """
        # Prevent recursive crash handling
        if self._handling_crash:
            return

        self._handling_crash = True

        try:
            _logger.critical(f"Received signal {signum}")

            # Capture crash context
            context = self.capture_state(signal_number=signum)
            self._crash_context = context

            # Notify callbacks
            self._notify_callbacks(context)

        except Exception as e:
            _logger.error(f"Error in signal handler: {e}")

        finally:
            self._handling_crash = False

            # Re-raise certain signals for default handling
            if signum in (signal.SIGINT, signal.SIGTERM):
                # For graceful shutdown signals, allow default behavior
                original = self._original_handlers.get(signum, signal.SIG_DFL)
                if callable(original):
                    original(signum, frame)
                else:
                    sys.exit(128 + signum)

    def _exception_handler(self,
                           exc_type: type,
                           exc_value: BaseException,
                           exc_tb: Any) -> None:
        """
        Internal exception hook handler.

        Called for uncaught exceptions.
        """
        # Prevent recursive crash handling
        if self._handling_crash:
            if self._original_excepthook:
                self._original_excepthook(exc_type, exc_value, exc_tb)
            return

        self._handling_crash = True

        try:
            _logger.critical(f"Uncaught exception: {exc_type.__name__}: {exc_value}")

            # Capture crash context
            context = self.capture_state(exception=exc_value)
            self._crash_context = context

            # Notify callbacks
            self._notify_callbacks(context)

        except Exception as e:
            _logger.error(f"Error in exception handler: {e}")

        finally:
            self._handling_crash = False

            # Call original exception hook
            if self._original_excepthook:
                self._original_excepthook(exc_type, exc_value, exc_tb)

    def _atexit_handler(self) -> None:
        """
        Internal atexit handler.

        Called when the program is exiting normally.
        """
        # Only capture if we haven't already handled a crash
        if not self._handling_crash and self._crash_context is None:
            _logger.debug("Normal program exit")

    def _notify_callbacks(self, context: CrashContext) -> None:
        """
        Notify all registered callbacks of a crash.

        Callbacks are called in registration order. Exceptions in
        callbacks are caught and logged but don't prevent other
        callbacks from being called.

        Args:
            context: The crash context to pass to callbacks
        """
        with self._lock:
            callbacks = list(self._callbacks)

        for callback in callbacks:
            try:
                callback(context)
            except Exception as e:
                _logger.error(f"Error in crash callback {callback}: {e}")

    def get_last_crash_context(self) -> Optional[CrashContext]:
        """
        Get the last captured crash context.

        Returns:
            The last CrashContext, or None if no crash has occurred
        """
        return self._crash_context

    def is_installed(self) -> bool:
        """
        Check if the crash handler is currently installed.

        Returns:
            True if installed, False otherwise
        """
        return self._installed

    def clear_logs(self) -> None:
        """Clear the recent log buffer."""
        self._log_handler.clear()


# Global crash handler instance
_global_handler: Optional[CrashHandler] = None


def get_global_handler() -> CrashHandler:
    """
    Get the global CrashHandler instance.

    Creates one if it doesn't exist.

    Returns:
        The global CrashHandler instance
    """
    global _global_handler
    if _global_handler is None:
        _global_handler = CrashHandler()
    return _global_handler


def install_global_handler() -> CrashHandler:
    """
    Install the global crash handler.

    Convenience function that gets or creates the global handler
    and installs it.

    Returns:
        The installed CrashHandler instance
    """
    handler = get_global_handler()
    handler.install()
    return handler


# Export public API
__all__ = [
    'CrashContext',
    'CrashCallback',
    'CrashHandler',
    'RecentLogHandler',
    'get_global_handler',
    'install_global_handler',
]
