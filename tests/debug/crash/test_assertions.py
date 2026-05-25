"""
Tests for the assertion system.

Tests cover:
- check/checkf behavior with different responses
- verify returning values
- ensure/ensureAlways logging behavior
- checkSlow debug-only behavior
- AssertionContext context manager
"""

import logging
import pytest

from engine.debug.crash.assertions import (
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


@pytest.fixture(autouse=True)
def reset_assertion_state():
    """Reset assertion configuration before each test."""
    # Store original state
    original_response = get_assertion_response()
    original_debug = is_debug_build()

    # Reset to defaults
    set_assertion_response(AssertionResponse.LOG)  # Use LOG for tests
    set_debug_build(True)
    reset_logged_assertions()
    set_log_callback(None)
    set_crash_callback(None)

    yield

    # Restore original state
    set_assertion_response(original_response)
    set_debug_build(original_debug)
    reset_logged_assertions()
    set_log_callback(None)
    set_crash_callback(None)


class TestCheck:
    """Tests for check() assertion."""

    def test_check_passes_on_true(self):
        """check() should do nothing when condition is True."""
        # Should not raise or log
        check(True)
        check(1)
        check("non-empty")
        check([1, 2, 3])

    def test_check_with_log_response(self):
        """check() should log when condition is False and response is LOG."""
        set_assertion_response(AssertionResponse.LOG)
        # Should log but not crash
        check(False)  # Just verifying it doesn't raise

    def test_check_with_continue_response(self):
        """check() should silently continue when response is CONTINUE."""
        set_assertion_response(AssertionResponse.CONTINUE)
        check(False)  # Should do nothing

    def test_check_with_crash_response(self):
        """check() should exit when condition is False and response is CRASH."""
        set_assertion_response(AssertionResponse.CRASH)
        with pytest.raises(SystemExit):
            check(False)

    def test_check_custom_log_callback(self, caplog):
        """check() should use custom log callback when set."""
        logged_messages = []

        def custom_logger(filename, function, line, message):
            logged_messages.append((filename, function, line, message))

        set_log_callback(custom_logger)
        set_assertion_response(AssertionResponse.LOG)

        check(False)

        assert len(logged_messages) == 1
        assert "Assertion failed" in logged_messages[0][3]


class TestCheckf:
    """Tests for checkf() formatted assertion."""

    def test_checkf_passes_on_true(self):
        """checkf() should do nothing when condition is True."""
        checkf(True, "This should not appear")
        checkf(1, "Value is %d", 1)

    def test_checkf_formats_message(self, caplog):
        """checkf() should format the message with arguments."""
        logged_messages = []

        def custom_logger(filename, function, line, message):
            logged_messages.append(message)

        set_log_callback(custom_logger)
        set_assertion_response(AssertionResponse.LOG)

        checkf(False, "Value is %d, expected %d", 10, 20)

        assert len(logged_messages) == 1
        assert "Value is 10, expected 20" in logged_messages[0]

    def test_checkf_with_no_args(self):
        """checkf() should work without format arguments."""
        set_assertion_response(AssertionResponse.LOG)
        checkf(False, "Simple message")  # Should not raise

    def test_checkf_with_crash_response(self):
        """checkf() should exit when condition is False and response is CRASH."""
        set_assertion_response(AssertionResponse.CRASH)
        with pytest.raises(SystemExit):
            checkf(False, "Expected positive value, got %d", -5)


class TestVerify:
    """Tests for verify() assertion that returns value."""

    def test_verify_returns_truthy_value(self):
        """verify() should return the value when truthy."""
        assert verify(42) == 42
        assert verify("hello") == "hello"
        assert verify([1, 2, 3]) == [1, 2, 3]
        assert verify({'key': 'value'}) == {'key': 'value'}

    def test_verify_returns_zero(self):
        """verify(0) should fail since 0 is falsy."""
        set_assertion_response(AssertionResponse.LOG)
        # 0 is falsy, so verify should trigger assertion
        # But since we're using LOG, it should still return
        result = verify(0)
        # Note: verify returns the value even if assertion fails with LOG
        # This is because it only crashes with CRASH response
        assert result == 0

    def test_verify_fails_on_none(self):
        """verify() should fail on None."""
        set_assertion_response(AssertionResponse.CRASH)
        with pytest.raises(SystemExit):
            verify(None)

    def test_verify_fails_on_false(self):
        """verify() should fail on False."""
        set_assertion_response(AssertionResponse.CRASH)
        with pytest.raises(SystemExit):
            verify(False)

    def test_verify_fails_on_empty_list(self):
        """verify() should fail on empty list."""
        set_assertion_response(AssertionResponse.CRASH)
        with pytest.raises(SystemExit):
            verify([])


class TestEnsure:
    """Tests for ensure() non-fatal assertion."""

    def test_ensure_returns_true_on_success(self):
        """ensure() should return True when condition is True."""
        assert ensure(True) is True
        assert ensure(1) is True
        assert ensure("non-empty") is True

    def test_ensure_returns_false_on_failure(self):
        """ensure() should return False when condition is False."""
        assert ensure(False) is False
        assert ensure(0) is False
        assert ensure("") is False
        assert ensure(None) is False

    def test_ensure_logs_once(self):
        """ensure() should only log the first failure at each location."""
        logged_count = [0]

        def custom_logger(filename, function, line, message):
            logged_count[0] += 1

        set_log_callback(custom_logger)

        # Multiple failures at same location
        for _ in range(5):
            ensure(False)

        # Should only log once
        assert logged_count[0] == 1

    def test_ensure_logs_different_locations(self):
        """ensure() should log for each unique location."""
        logged_count = [0]

        def custom_logger(filename, function, line, message):
            logged_count[0] += 1

        set_log_callback(custom_logger)
        reset_logged_assertions()

        # Different lines = different locations
        ensure(False)  # Line A
        ensure(False)  # Line B

        # Each location logged once
        assert logged_count[0] == 2

    def test_ensure_never_crashes(self):
        """ensure() should never crash even with CRASH response."""
        set_assertion_response(AssertionResponse.CRASH)
        # ensure is non-fatal, should not exit
        result = ensure(False)
        assert result is False


class TestEnsureAlways:
    """Tests for ensureAlways() non-fatal assertion."""

    def test_ensureAlways_returns_true_on_success(self):
        """ensureAlways() should return True when condition is True."""
        assert ensureAlways(True) is True

    def test_ensureAlways_returns_false_on_failure(self):
        """ensureAlways() should return False when condition is False."""
        assert ensureAlways(False) is False

    def test_ensureAlways_logs_every_time(self):
        """ensureAlways() should log every failure."""
        logged_count = [0]

        def custom_logger(filename, function, line, message):
            logged_count[0] += 1

        set_log_callback(custom_logger)

        # Multiple failures at same location
        for _ in range(5):
            ensureAlways(False)

        # Should log every time
        assert logged_count[0] == 5

    def test_ensureAlways_never_crashes(self):
        """ensureAlways() should never crash even with CRASH response."""
        set_assertion_response(AssertionResponse.CRASH)
        result = ensureAlways(False)
        assert result is False


class TestCheckSlow:
    """Tests for checkSlow() debug-only assertion."""

    def test_checkSlow_runs_in_debug(self):
        """checkSlow() should run in debug builds."""
        set_debug_build(True)
        set_assertion_response(AssertionResponse.CRASH)

        # Should trigger assertion
        with pytest.raises(SystemExit):
            checkSlow(False)

    def test_checkSlow_skipped_in_release(self):
        """checkSlow() should be skipped in release builds."""
        set_debug_build(False)
        set_assertion_response(AssertionResponse.CRASH)

        # Should not trigger assertion
        checkSlow(False)  # No crash

    def test_checkSlow_passes_on_true(self):
        """checkSlow() should pass when condition is True."""
        set_debug_build(True)
        checkSlow(True)  # Should not raise

    def test_checkSlowf_formats_message(self):
        """checkSlowf() should format message in debug builds."""
        set_debug_build(True)

        logged_messages = []

        def custom_logger(filename, function, line, message):
            logged_messages.append(message)

        set_log_callback(custom_logger)
        set_assertion_response(AssertionResponse.LOG)

        checkSlowf(False, "Value is %d", 42)

        assert len(logged_messages) == 1
        assert "Value is 42" in logged_messages[0]


class TestAssertionContext:
    """Tests for AssertionContext context manager."""

    def test_context_changes_response(self):
        """AssertionContext should change response within context."""
        set_assertion_response(AssertionResponse.CRASH)

        with AssertionContext(AssertionResponse.LOG):
            assert get_assertion_response() == AssertionResponse.LOG
            check(False)  # Should not crash

        assert get_assertion_response() == AssertionResponse.CRASH

    def test_context_restores_on_exit(self):
        """AssertionContext should restore original response on exit."""
        set_assertion_response(AssertionResponse.CONTINUE)

        with AssertionContext(AssertionResponse.LOG):
            pass

        assert get_assertion_response() == AssertionResponse.CONTINUE

    def test_context_restores_on_exception(self):
        """AssertionContext should restore response even after exception."""
        set_assertion_response(AssertionResponse.CONTINUE)

        try:
            with AssertionContext(AssertionResponse.LOG):
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert get_assertion_response() == AssertionResponse.CONTINUE

    def test_nested_contexts(self):
        """Nested AssertionContext should work correctly."""
        set_assertion_response(AssertionResponse.CRASH)

        with AssertionContext(AssertionResponse.LOG):
            assert get_assertion_response() == AssertionResponse.LOG

            with AssertionContext(AssertionResponse.CONTINUE):
                assert get_assertion_response() == AssertionResponse.CONTINUE

            assert get_assertion_response() == AssertionResponse.LOG

        assert get_assertion_response() == AssertionResponse.CRASH


class TestCrashCallback:
    """Tests for crash callback functionality."""

    def test_crash_callback_called(self):
        """Crash callback should be called before crashing."""
        callback_calls = []

        def crash_callback(filename, function, line):
            callback_calls.append((filename, function, line))

        set_crash_callback(crash_callback)
        set_assertion_response(AssertionResponse.CRASH)

        with pytest.raises(SystemExit):
            check(False)

        assert len(callback_calls) == 1

    def test_crash_callback_receives_location(self):
        """Crash callback should receive correct location info."""
        location = [None]

        def crash_callback(filename, function, line):
            location[0] = (filename, function, line)

        set_crash_callback(crash_callback)
        set_assertion_response(AssertionResponse.CRASH)

        with pytest.raises(SystemExit):
            check(False)

        assert location[0] is not None
        assert __file__ in location[0][0]  # Filename contains test file
        assert isinstance(location[0][2], int)  # Line is integer


class TestAssertionResponseEnum:
    """Tests for AssertionResponse enum values."""

    def test_all_response_types_exist(self):
        """All expected response types should exist."""
        assert AssertionResponse.BREAK is not None
        assert AssertionResponse.LOG is not None
        assert AssertionResponse.CRASH is not None
        assert AssertionResponse.CONTINUE is not None

    def test_set_and_get_response(self):
        """Setting and getting response should work."""
        for response in AssertionResponse:
            set_assertion_response(response)
            assert get_assertion_response() == response


class TestDebugBuildToggle:
    """Tests for debug build toggle."""

    def test_set_and_get_debug_build(self):
        """Setting and getting debug build should work."""
        set_debug_build(True)
        assert is_debug_build() is True

        set_debug_build(False)
        assert is_debug_build() is False
