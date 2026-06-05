"""Tests for error aggregation system with error panel (T-CC-1.10)."""
import threading
import time

import pytest

from engine.core.error_aggregation import (
    ErrorAggregator,
    ErrorEntry,
    ErrorFilter,
    ErrorPanel,
    ErrorSeverity,
    ErrorSource,
    ErrorStats,
    get_global_aggregator,
    report_critical,
    report_error,
    report_warning,
    set_global_aggregator,
)
from engine.core.result import Error, ErrorKind


class TestErrorSeverity:
    """Tests for ErrorSeverity enum."""

    def test_all_severities_exist(self):
        severities = [
            ErrorSeverity.DEBUG,
            ErrorSeverity.INFO,
            ErrorSeverity.WARNING,
            ErrorSeverity.ERROR,
            ErrorSeverity.CRITICAL,
            ErrorSeverity.FATAL,
        ]
        assert len(severities) == 6

    def test_severity_ordering(self):
        assert ErrorSeverity.DEBUG.value < ErrorSeverity.INFO.value
        assert ErrorSeverity.INFO.value < ErrorSeverity.WARNING.value
        assert ErrorSeverity.WARNING.value < ErrorSeverity.ERROR.value
        assert ErrorSeverity.ERROR.value < ErrorSeverity.CRITICAL.value
        assert ErrorSeverity.CRITICAL.value < ErrorSeverity.FATAL.value


class TestErrorSource:
    """Tests for ErrorSource enum."""

    def test_all_sources_exist(self):
        sources = [
            ErrorSource.UNKNOWN,
            ErrorSource.RENDERING,
            ErrorSource.PHYSICS,
            ErrorSource.AUDIO,
            ErrorSource.NETWORK,
            ErrorSource.SCRIPTING,
            ErrorSource.ASSET,
            ErrorSource.GPU,
            ErrorSource.FFI,
            ErrorSource.MEMORY,
            ErrorSource.IO,
            ErrorSource.CONFIG,
            ErrorSource.VALIDATION,
        ]
        assert len(sources) == 13


class TestErrorEntry:
    """Tests for ErrorEntry dataclass."""

    def test_basic_entry(self):
        entry = ErrorEntry(
            id=1,
            message="Test error",
            severity=ErrorSeverity.ERROR,
            source=ErrorSource.RENDERING,
        )
        assert entry.id == 1
        assert entry.message == "Test error"
        assert entry.count == 1

    def test_is_critical(self):
        critical = ErrorEntry(
            id=1, message="", severity=ErrorSeverity.CRITICAL,
            source=ErrorSource.UNKNOWN
        )
        fatal = ErrorEntry(
            id=2, message="", severity=ErrorSeverity.FATAL,
            source=ErrorSource.UNKNOWN
        )
        error = ErrorEntry(
            id=3, message="", severity=ErrorSeverity.ERROR,
            source=ErrorSource.UNKNOWN
        )

        assert critical.is_critical
        assert fatal.is_critical
        assert not error.is_critical

    def test_age_seconds(self):
        entry = ErrorEntry(
            id=1, message="", severity=ErrorSeverity.ERROR,
            source=ErrorSource.UNKNOWN,
            timestamp=time.time() - 5.0
        )
        assert 4.9 < entry.age_seconds < 6.0

    def test_matches(self):
        entry1 = ErrorEntry(
            id=1, message="Error A", severity=ErrorSeverity.ERROR,
            source=ErrorSource.GPU
        )
        entry2 = ErrorEntry(
            id=2, message="Error A", severity=ErrorSeverity.ERROR,
            source=ErrorSource.GPU
        )
        entry3 = ErrorEntry(
            id=3, message="Error B", severity=ErrorSeverity.ERROR,
            source=ErrorSource.GPU
        )

        assert entry1.matches(entry2)
        assert not entry1.matches(entry3)


class TestErrorStats:
    """Tests for ErrorStats dataclass."""

    def test_empty_stats(self):
        stats = ErrorStats()
        assert stats.total_count == 0
        assert stats.first_error_time is None

    def test_update(self):
        stats = ErrorStats()
        entry = ErrorEntry(
            id=1, message="Test", severity=ErrorSeverity.ERROR,
            source=ErrorSource.GPU, kind=ErrorKind.GPU
        )
        stats.update(entry)

        assert stats.total_count == 1
        assert stats.by_severity[ErrorSeverity.ERROR] == 1
        assert stats.by_source[ErrorSource.GPU] == 1
        assert stats.by_kind[ErrorKind.GPU] == 1
        assert stats.first_error_time is not None


class TestErrorFilter:
    """Tests for ErrorFilter."""

    def test_min_severity(self):
        filter = ErrorFilter(min_severity=ErrorSeverity.WARNING)
        debug = ErrorEntry(id=1, message="", severity=ErrorSeverity.DEBUG, source=ErrorSource.UNKNOWN)
        warning = ErrorEntry(id=2, message="", severity=ErrorSeverity.WARNING, source=ErrorSource.UNKNOWN)
        error = ErrorEntry(id=3, message="", severity=ErrorSeverity.ERROR, source=ErrorSource.UNKNOWN)

        assert not filter.matches(debug)
        assert filter.matches(warning)
        assert filter.matches(error)

    def test_max_severity(self):
        filter = ErrorFilter(max_severity=ErrorSeverity.WARNING)
        warning = ErrorEntry(id=1, message="", severity=ErrorSeverity.WARNING, source=ErrorSource.UNKNOWN)
        error = ErrorEntry(id=2, message="", severity=ErrorSeverity.ERROR, source=ErrorSource.UNKNOWN)

        assert filter.matches(warning)
        assert not filter.matches(error)

    def test_sources(self):
        filter = ErrorFilter(sources={ErrorSource.GPU, ErrorSource.RENDERING})
        gpu = ErrorEntry(id=1, message="", severity=ErrorSeverity.ERROR, source=ErrorSource.GPU)
        audio = ErrorEntry(id=2, message="", severity=ErrorSeverity.ERROR, source=ErrorSource.AUDIO)

        assert filter.matches(gpu)
        assert not filter.matches(audio)

    def test_kinds(self):
        filter = ErrorFilter(kinds={ErrorKind.GPU, ErrorKind.TIMEOUT})
        gpu = ErrorEntry(id=1, message="", severity=ErrorSeverity.ERROR, source=ErrorSource.UNKNOWN, kind=ErrorKind.GPU)
        io = ErrorEntry(id=2, message="", severity=ErrorSeverity.ERROR, source=ErrorSource.UNKNOWN, kind=ErrorKind.IO)

        assert filter.matches(gpu)
        assert not filter.matches(io)

    def test_message_pattern(self):
        filter = ErrorFilter(message_pattern="failed")
        match = ErrorEntry(id=1, message="Operation failed", severity=ErrorSeverity.ERROR, source=ErrorSource.UNKNOWN)
        nomatch = ErrorEntry(id=2, message="Success", severity=ErrorSeverity.ERROR, source=ErrorSource.UNKNOWN)

        assert filter.matches(match)
        assert not filter.matches(nomatch)

    def test_max_age(self):
        filter = ErrorFilter(max_age_seconds=10.0)
        recent = ErrorEntry(
            id=1, message="", severity=ErrorSeverity.ERROR,
            source=ErrorSource.UNKNOWN, timestamp=time.time()
        )
        old = ErrorEntry(
            id=2, message="", severity=ErrorSeverity.ERROR,
            source=ErrorSource.UNKNOWN, timestamp=time.time() - 20
        )

        assert filter.matches(recent)
        assert not filter.matches(old)


class TestErrorAggregator:
    """Tests for ErrorAggregator."""

    def test_add_error(self):
        agg = ErrorAggregator()
        entry = agg.add("Test error", ErrorSeverity.ERROR, ErrorSource.GPU)

        assert entry is not None
        assert entry.id == 1
        assert agg.entry_count == 1

    def test_add_multiple(self):
        agg = ErrorAggregator()
        agg.add("Error 1")
        agg.add("Error 2")
        agg.add("Error 3")

        assert agg.entry_count == 3

    def test_deduplication(self):
        agg = ErrorAggregator(deduplicate=True, dedupe_window_seconds=60)
        agg.add("Same error")
        agg.add("Same error")
        agg.add("Same error")

        assert agg.entry_count == 1
        entry = agg.get_recent(1)[0]
        assert entry.count == 3

    def test_no_deduplication(self):
        agg = ErrorAggregator(deduplicate=False)
        agg.add("Same error")
        agg.add("Same error")

        assert agg.entry_count == 2

    def test_max_entries(self):
        agg = ErrorAggregator(max_entries=5)
        for i in range(10):
            agg.add(f"Error {i}")

        assert agg.entry_count == 5

    def test_add_from_error(self):
        agg = ErrorAggregator()
        error = Error("FFI failed", ErrorKind.FFI, context={"code": 1})
        entry = agg.add_from_error(error, ErrorSeverity.ERROR, ErrorSource.FFI)

        assert entry is not None
        assert entry.message == "FFI failed"
        assert entry.kind == ErrorKind.FFI
        assert entry.error is error

    def test_suppress(self):
        agg = ErrorAggregator()
        agg.suppress("Suppressed error")

        entry = agg.add("Suppressed error")
        assert entry is None
        assert agg.entry_count == 0

    def test_suppress_with_source(self):
        agg = ErrorAggregator()
        agg.suppress("Error", ErrorSource.GPU)

        # Different source should work
        entry = agg.add("Error", source=ErrorSource.AUDIO)
        assert entry is not None

        # Same source should be suppressed
        entry = agg.add("Error", source=ErrorSource.GPU)
        assert entry is None

    def test_unsuppress(self):
        agg = ErrorAggregator()
        agg.suppress("Error")
        agg.unsuppress("Error")

        entry = agg.add("Error")
        assert entry is not None

    def test_get_entries_with_filter(self):
        agg = ErrorAggregator()
        agg.add("Error 1", ErrorSeverity.ERROR)
        agg.add("Warning 1", ErrorSeverity.WARNING)
        agg.add("Error 2", ErrorSeverity.ERROR)

        filter = ErrorFilter(min_severity=ErrorSeverity.ERROR)
        entries = agg.get_entries(filter=filter)

        assert len(entries) == 2

    def test_get_entries_with_limit_offset(self):
        agg = ErrorAggregator()
        for i in range(10):
            agg.add(f"Error {i}")

        entries = agg.get_entries(limit=3, offset=2)
        assert len(entries) == 3

    def test_get_recent(self):
        agg = ErrorAggregator()
        for i in range(10):
            agg.add(f"Error {i}")

        recent = agg.get_recent(3)
        assert len(recent) == 3
        assert recent[-1].message == "Error 9"

    def test_get_by_severity(self):
        agg = ErrorAggregator()
        agg.add("Error", ErrorSeverity.ERROR)
        agg.add("Warning", ErrorSeverity.WARNING)
        agg.add("Error 2", ErrorSeverity.ERROR)

        errors = agg.get_by_severity(ErrorSeverity.ERROR)
        assert len(errors) == 2

    def test_get_critical(self):
        agg = ErrorAggregator()
        agg.add("Error", ErrorSeverity.ERROR)
        agg.add("Critical", ErrorSeverity.CRITICAL)
        agg.add("Fatal", ErrorSeverity.FATAL)

        critical = agg.get_critical()
        assert len(critical) == 2

    def test_handler_registration(self):
        agg = ErrorAggregator()
        received = []

        def handler(entry):
            received.append(entry)

        agg.register_handler(handler)
        agg.add("Test")

        assert len(received) == 1
        assert received[0].message == "Test"

    def test_handler_unregistration(self):
        agg = ErrorAggregator()
        received = []

        def handler(entry):
            received.append(entry)

        agg.register_handler(handler)
        agg.unregister_handler(handler)
        agg.add("Test")

        assert len(received) == 0

    def test_clear(self):
        agg = ErrorAggregator()
        agg.add("Error 1")
        agg.add("Error 2")
        agg.clear()

        assert agg.entry_count == 0

    def test_clear_old(self):
        agg = ErrorAggregator()
        # Add entries with old timestamps
        entry1 = ErrorEntry(
            id=1, message="Old", severity=ErrorSeverity.ERROR,
            source=ErrorSource.UNKNOWN, timestamp=time.time() - 100
        )
        entry2 = ErrorEntry(
            id=2, message="New", severity=ErrorSeverity.ERROR,
            source=ErrorSource.UNKNOWN, timestamp=time.time()
        )
        agg._entries.extend([entry1, entry2])

        cleared = agg.clear_old(50)
        assert cleared == 1
        assert agg.entry_count == 1

    def test_stats_updated(self):
        agg = ErrorAggregator()
        agg.add("Error", ErrorSeverity.ERROR, ErrorSource.GPU)

        stats = agg.stats
        assert stats.total_count == 1
        assert stats.by_severity[ErrorSeverity.ERROR] == 1
        assert stats.by_source[ErrorSource.GPU] == 1


class TestErrorPanel:
    """Tests for ErrorPanel."""

    def test_creation(self):
        agg = ErrorAggregator()
        panel = ErrorPanel(agg)

        assert panel.visible
        assert panel.get_visible_entries() == []

    def test_visibility(self):
        agg = ErrorAggregator()
        panel = ErrorPanel(agg)

        panel.visible = False
        assert not panel.visible
        assert panel.get_visible_entries() == []

    def test_visible_entries(self):
        agg = ErrorAggregator()
        panel = ErrorPanel(agg)

        agg.add("Error 1")
        agg.add("Error 2")

        entries = panel.get_visible_entries()
        assert len(entries) == 2

    def test_filter(self):
        agg = ErrorAggregator()
        panel = ErrorPanel(agg)

        agg.add("Error", ErrorSeverity.ERROR)
        agg.add("Warning", ErrorSeverity.WARNING)

        panel.filter = ErrorFilter(min_severity=ErrorSeverity.ERROR)
        entries = panel.get_visible_entries()

        assert len(entries) == 1

    def test_selection(self):
        agg = ErrorAggregator()
        panel = ErrorPanel(agg)

        entry = agg.add("Test")
        panel.select(entry.id)

        assert panel.selected_entry is entry

    def test_expand_collapse(self):
        agg = ErrorAggregator()
        panel = ErrorPanel(agg)

        entry = agg.add("Test")

        assert not panel.is_expanded(entry.id)

        panel.expand(entry.id)
        assert panel.is_expanded(entry.id)

        panel.collapse(entry.id)
        assert not panel.is_expanded(entry.id)

    def test_toggle_expand(self):
        agg = ErrorAggregator()
        panel = ErrorPanel(agg)

        entry = agg.add("Test")

        panel.toggle_expand(entry.id)
        assert panel.is_expanded(entry.id)

        panel.toggle_expand(entry.id)
        assert not panel.is_expanded(entry.id)

    def test_acknowledge(self):
        agg = ErrorAggregator()
        panel = ErrorPanel(agg)

        entry = agg.add("Test")
        panel.acknowledge(entry.id)

        assert entry.id not in [e.id for e in panel.get_visible_entries()]

    def test_acknowledge_all(self):
        agg = ErrorAggregator()
        panel = ErrorPanel(agg)

        agg.add("Error 1")
        agg.add("Error 2")

        panel.acknowledge_all()
        assert len(panel.get_visible_entries()) == 0

    def test_unacknowledge(self):
        agg = ErrorAggregator()
        panel = ErrorPanel(agg)

        entry = agg.add("Test")
        panel.acknowledge(entry.id)
        panel.unacknowledge(entry.id)

        assert len(panel.get_visible_entries()) == 1

    def test_auto_expand_critical(self):
        agg = ErrorAggregator()
        panel = ErrorPanel(agg, auto_expand_critical=True)

        entry = agg.add("Critical", ErrorSeverity.CRITICAL)

        assert panel.is_expanded(entry.id)

    def test_listener(self):
        agg = ErrorAggregator()
        panel = ErrorPanel(agg)
        called = []

        def listener():
            called.append(True)

        panel.add_listener(listener)
        agg.add("Test")

        assert len(called) >= 1

    def test_remove_listener(self):
        agg = ErrorAggregator()
        panel = ErrorPanel(agg)
        called = []

        def listener():
            called.append(True)

        panel.add_listener(listener)
        panel.remove_listener(listener)
        agg.add("Test")

        assert len(called) == 0

    def test_get_summary(self):
        agg = ErrorAggregator()
        panel = ErrorPanel(agg)

        agg.add("Error", ErrorSeverity.ERROR)
        agg.add("Critical", ErrorSeverity.CRITICAL)

        summary = panel.get_summary()
        assert summary['entry_count'] == 2
        assert summary['critical_count'] == 1
        assert summary['visible']

    def test_format_entry(self):
        agg = ErrorAggregator()
        panel = ErrorPanel(agg)

        entry = agg.add("Test error", ErrorSeverity.ERROR, ErrorSource.GPU)

        formatted = panel.format_entry(entry)
        assert "[ERROR]" in formatted
        assert "Test error" in formatted
        assert "GPU" in formatted

    def test_format_entry_with_context(self):
        agg = ErrorAggregator()
        panel = ErrorPanel(agg)

        entry = agg.add("Test", context={"key": "value"})

        formatted = panel.format_entry(entry, include_context=True)
        assert "key" in formatted


class TestGlobalFunctions:
    """Tests for global convenience functions."""

    def setup_method(self):
        set_global_aggregator(ErrorAggregator())

    def test_get_global_aggregator(self):
        agg = get_global_aggregator()
        assert agg is not None

    def test_report_error(self):
        entry = report_error("Test error", ErrorSeverity.ERROR)
        assert entry is not None
        assert entry.severity == ErrorSeverity.ERROR

    def test_report_warning(self):
        entry = report_warning("Test warning")
        assert entry is not None
        assert entry.severity == ErrorSeverity.WARNING

    def test_report_critical(self):
        entry = report_critical("Test critical")
        assert entry is not None
        assert entry.severity == ErrorSeverity.CRITICAL

    def test_report_with_context(self):
        entry = report_error("Test", context={"key": "value"})
        assert entry.context["key"] == "value"


class TestConcurrency:
    """Thread safety tests."""

    def test_concurrent_add(self):
        agg = ErrorAggregator()
        errors = []

        def add_errors():
            try:
                for i in range(100):
                    agg.add(f"Error {threading.current_thread().name}_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_errors) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Some may be deduplicated, so count may vary

    def test_concurrent_read(self):
        agg = ErrorAggregator()
        for i in range(50):
            agg.add(f"Error {i}")

        errors = []

        def read_entries():
            try:
                for _ in range(100):
                    agg.get_entries()
                    agg.get_recent(10)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_entries) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestEdgeCases:
    """Edge case tests."""

    def test_handler_exception_doesnt_crash(self):
        agg = ErrorAggregator()

        def bad_handler(entry):
            raise ValueError("Handler error")

        agg.register_handler(bad_handler)

        # Should not raise
        entry = agg.add("Test")
        assert entry is not None

    def test_rate_limiting(self):
        agg = ErrorAggregator()
        agg._rate_limit_max = 5
        agg._rate_limit_window = 1.0

        # Should add first 5
        for i in range(10):
            entry = agg.add("Same message")

        # Some should be rate limited
        assert agg.entry_count < 10

    def test_empty_message(self):
        agg = ErrorAggregator()
        entry = agg.add("")
        assert entry is not None
        assert entry.message == ""

    def test_none_context(self):
        agg = ErrorAggregator()
        entry = agg.add("Test", context=None)
        assert entry.context is None

    def test_selected_entry_not_found(self):
        agg = ErrorAggregator()
        panel = ErrorPanel(agg)

        panel.select(9999)  # Non-existent ID
        assert panel.selected_entry is None
