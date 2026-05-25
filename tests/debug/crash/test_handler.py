"""
Tests for the crash handler.

Tests cover:
- CrashContext dataclass
- CrashHandler installation/uninstallation
- Callback registration and notification
- State capture
- RecentLogHandler buffer
- Configuration constants
"""

import logging
import signal
import sys
import threading
import time
from datetime import datetime

import pytest

from engine.debug.crash.handler import (
    CrashCallback,
    CrashContext,
    CrashHandler,
    RecentLogHandler,
    get_global_handler,
    install_global_handler,
    DEFAULT_MAX_RECENT_LOGS,
    MAX_STACK_TRACE_DEPTH,
)


@pytest.fixture
def handler():
    """Create a fresh CrashHandler for each test."""
    h = CrashHandler()
    yield h
    # Ensure cleanup
    h.uninstall()


@pytest.fixture
def installed_handler(handler):
    """Create an installed CrashHandler."""
    handler.install()
    yield handler
    handler.uninstall()


class TestCrashContext:
    """Tests for CrashContext dataclass."""

    def test_default_values(self):
        """CrashContext should have sensible defaults."""
        ctx = CrashContext()

        assert ctx.exception is None
        assert ctx.stack_trace == ""
        assert ctx.recent_logs == []
        assert isinstance(ctx.timestamp, datetime)
        assert ctx.thread_id != 0
        assert ctx.thread_name != ""
        assert ctx.signal_number is None
        assert ctx.signal_name is None
        assert ctx.additional_data == {}

    def test_with_exception(self):
        """CrashContext should store exception information."""
        try:
            raise ValueError("test error")
        except ValueError as e:
            ctx = CrashContext(exception=e)

        assert ctx.exception is not None
        assert isinstance(ctx.exception, ValueError)
        assert str(ctx.exception) == "test error"

    def test_with_stack_trace(self):
        """CrashContext should store stack trace."""
        ctx = CrashContext(stack_trace="line 1\nline 2\nline 3")
        assert "line 1" in ctx.stack_trace
        assert "line 2" in ctx.stack_trace

    def test_with_recent_logs(self):
        """CrashContext should store recent logs."""
        logs = ["Log 1", "Log 2", "Log 3"]
        ctx = CrashContext(recent_logs=logs)
        assert ctx.recent_logs == logs

    def test_with_signal(self):
        """CrashContext should store signal information."""
        ctx = CrashContext(signal_number=signal.SIGTERM, signal_name="SIGTERM")
        assert ctx.signal_number == signal.SIGTERM
        assert ctx.signal_name == "SIGTERM"

    def test_thread_info_populated(self):
        """CrashContext should populate thread info on creation."""
        ctx = CrashContext()

        assert ctx.thread_id == threading.current_thread().ident
        assert ctx.thread_name == threading.current_thread().name

    def test_additional_data(self):
        """CrashContext should support additional custom data."""
        ctx = CrashContext(additional_data={"key": "value", "count": 42})
        assert ctx.additional_data["key"] == "value"
        assert ctx.additional_data["count"] == 42


class TestCrashHandler:
    """Tests for CrashHandler class."""

    def test_initial_state(self, handler):
        """Handler should start uninstalled."""
        assert not handler.is_installed()
        assert handler.get_last_crash_context() is None

    def test_install_uninstall(self, handler):
        """Handler should install and uninstall correctly."""
        handler.install()
        assert handler.is_installed()

        handler.uninstall()
        assert not handler.is_installed()

    def test_install_idempotent(self, handler):
        """Multiple installs should be safe."""
        handler.install()
        handler.install()  # Should not raise
        assert handler.is_installed()

        handler.uninstall()

    def test_uninstall_idempotent(self, handler):
        """Multiple uninstalls should be safe."""
        handler.install()
        handler.uninstall()
        handler.uninstall()  # Should not raise
        assert not handler.is_installed()

    def test_callback_registration(self, handler):
        """Callbacks should be registered and removed."""
        callback_calls = []

        def callback(ctx):
            callback_calls.append(ctx)

        handler.on_crash(callback)
        assert handler.remove_callback(callback) is True
        assert handler.remove_callback(callback) is False  # Already removed

    def test_callback_not_duplicated(self, handler):
        """Same callback should not be registered twice."""
        callback_calls = []

        def callback(ctx):
            callback_calls.append(ctx)

        handler.on_crash(callback)
        handler.on_crash(callback)  # Should not add again

        # Verify by removing once (should be removed completely)
        assert handler.remove_callback(callback) is True
        assert handler.remove_callback(callback) is False

    def test_capture_state_basic(self, installed_handler):
        """capture_state should return valid CrashContext."""
        ctx = installed_handler.capture_state()

        assert isinstance(ctx, CrashContext)
        assert ctx.stack_trace != ""
        assert ctx.thread_id != 0
        assert isinstance(ctx.timestamp, datetime)

    def test_capture_state_with_exception(self, installed_handler):
        """capture_state should include exception info."""
        try:
            raise RuntimeError("test error")
        except RuntimeError as e:
            ctx = installed_handler.capture_state(exception=e)

        assert ctx.exception is not None
        assert isinstance(ctx.exception, RuntimeError)
        assert "test error" in ctx.stack_trace
        assert "RuntimeError" in ctx.stack_trace

    def test_capture_state_with_signal(self, installed_handler):
        """capture_state should include signal info."""
        ctx = installed_handler.capture_state(signal_number=signal.SIGTERM)

        assert ctx.signal_number == signal.SIGTERM
        assert ctx.signal_name == "SIGTERM"

    def test_capture_state_includes_logs(self, installed_handler):
        """capture_state should include recent logs."""
        logger = logging.getLogger("test_crash_handler")
        # Ensure logger level is set to capture INFO messages
        original_level = logger.level
        logger.setLevel(logging.DEBUG)

        try:
            logger.info("Test log message 1")
            logger.warning("Test log message 2")

            ctx = installed_handler.capture_state()

            # Should have captured logs
            assert len(ctx.recent_logs) >= 2
            assert any("Test log message 1" in log for log in ctx.recent_logs)
            assert any("Test log message 2" in log for log in ctx.recent_logs)
        finally:
            logger.setLevel(original_level)

    def test_clear_logs(self, installed_handler):
        """clear_logs should empty the log buffer."""
        logger = logging.getLogger("test_crash_handler_clear")
        original_level = logger.level
        logger.setLevel(logging.DEBUG)

        try:
            logger.warning("Test message")  # Use warning to ensure it's captured

            ctx_before = installed_handler.capture_state()
            assert len(ctx_before.recent_logs) > 0, "Expected at least one log message"

            installed_handler.clear_logs()

            ctx_after = installed_handler.capture_state()
            assert len(ctx_after.recent_logs) == 0
        finally:
            logger.setLevel(original_level)


class TestRecentLogHandler:
    """Tests for RecentLogHandler class."""

    def test_buffer_capacity(self):
        """Handler should respect max_entries limit."""
        handler = RecentLogHandler(max_entries=5)
        logger = logging.getLogger("test_buffer")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        for i in range(10):
            logger.info(f"Message {i}")

        logs = handler.get_recent_logs()
        assert len(logs) == 5
        # Should have newest messages
        assert any("Message 9" in log for log in logs)
        assert not any("Message 0" in log for log in logs)

        logger.removeHandler(handler)

    def test_clear(self):
        """clear() should empty the buffer."""
        handler = RecentLogHandler(max_entries=10)
        logger = logging.getLogger("test_clear")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        logger.info("Test message")
        assert len(handler.get_recent_logs()) > 0

        handler.clear()
        assert len(handler.get_recent_logs()) == 0

        logger.removeHandler(handler)

    def test_thread_safety(self):
        """Handler should be thread-safe."""
        handler = RecentLogHandler(max_entries=100)
        logger = logging.getLogger("test_thread_safety")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        errors = []

        def log_messages(thread_id):
            try:
                for i in range(50):
                    logger.info(f"Thread {thread_id} message {i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=log_messages, args=(i,))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(handler.get_recent_logs()) == 100  # Capped at max

        logger.removeHandler(handler)


class TestGlobalHandler:
    """Tests for global handler functions."""

    def test_get_global_handler(self):
        """get_global_handler should return same instance."""
        h1 = get_global_handler()
        h2 = get_global_handler()
        assert h1 is h2

    def test_install_global_handler(self):
        """install_global_handler should install and return handler."""
        handler = install_global_handler()
        assert handler.is_installed()
        handler.uninstall()


class TestExceptionHook:
    """Tests for exception hook handling."""

    def test_uncaught_exception_captured(self, installed_handler):
        """Uncaught exceptions should be captured."""
        callback_contexts = []

        def callback(ctx):
            callback_contexts.append(ctx)

        installed_handler.on_crash(callback)

        # Manually trigger exception hook
        try:
            raise ValueError("uncaught test error")
        except ValueError:
            exc_info = sys.exc_info()
            sys.excepthook(exc_info[0], exc_info[1], exc_info[2])

        assert len(callback_contexts) == 1
        assert callback_contexts[0].exception is not None
        assert "uncaught test error" in str(callback_contexts[0].exception)


class TestCallbackExceptionHandling:
    """Tests for handling exceptions in callbacks."""

    def test_callback_exception_does_not_break_others(self, installed_handler):
        """Exception in one callback should not prevent others."""
        callback1_called = [False]
        callback3_called = [False]

        def callback1(ctx):
            callback1_called[0] = True

        def callback2(ctx):
            raise RuntimeError("Callback error")

        def callback3(ctx):
            callback3_called[0] = True

        installed_handler.on_crash(callback1)
        installed_handler.on_crash(callback2)
        installed_handler.on_crash(callback3)

        # Trigger by capturing state and notifying
        ctx = installed_handler.capture_state()
        installed_handler._notify_callbacks(ctx)

        assert callback1_called[0] is True
        assert callback3_called[0] is True


class TestConfigurationConstants:
    """Tests for configuration constants."""

    def test_default_max_recent_logs_reasonable(self):
        """DEFAULT_MAX_RECENT_LOGS should be reasonable."""
        assert DEFAULT_MAX_RECENT_LOGS > 0
        assert DEFAULT_MAX_RECENT_LOGS <= 1000

    def test_max_stack_trace_depth_reasonable(self):
        """MAX_STACK_TRACE_DEPTH should be reasonable."""
        assert MAX_STACK_TRACE_DEPTH > 0
        assert MAX_STACK_TRACE_DEPTH <= 200

    def test_handler_uses_default_log_limit(self):
        """CrashHandler should use DEFAULT_MAX_RECENT_LOGS."""
        handler = CrashHandler()
        assert handler._log_handler._max_entries == DEFAULT_MAX_RECENT_LOGS
        handler.uninstall()

    def test_recent_log_handler_uses_default(self):
        """RecentLogHandler should use provided max_entries."""
        handler = RecentLogHandler()
        assert handler._max_entries == DEFAULT_MAX_RECENT_LOGS

        custom_handler = RecentLogHandler(max_entries=50)
        assert custom_handler._max_entries == 50
