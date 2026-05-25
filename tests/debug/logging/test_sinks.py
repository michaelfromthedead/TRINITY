"""
Comprehensive tests for log sinks.

Tests cover:
- ConsoleSink output and color formatting
- FileSink writing and rotation
- NetworkSink connection and batching
- BufferedSink batching behavior
- MultiplexSink distribution
"""

import pytest
import sys
import io
import json
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, "/home/user/dev/AI_GAME_ENGINE")

from engine.debug.logging.logger import LogLevel, LogCategory, LogEntry
from engine.debug.logging.sinks import (
    LogSink,
    ConsoleSink,
    FileSink,
    NetworkSink,
    BufferedSink,
    MultiplexSink,
)


def make_entry(
    message: str = "Test message",
    level: LogLevel = LogLevel.INFO,
    category: LogCategory = LogCategory.LogEngine,
    logger_name: str = "Test",
    fields: dict = None,
) -> LogEntry:
    """Helper to create test log entries."""
    return LogEntry(
        timestamp=datetime.now(timezone.utc),
        level=level,
        category=category,
        message=message,
        logger_name=logger_name,
        fields=fields or {},
    )


class TestLogSinkBase:
    """Tests for LogSink abstract base class."""

    def test_abstract_write_method(self):
        """Verify LogSink requires write() implementation."""
        with pytest.raises(TypeError):
            LogSink()

    def test_subclass_implementation(self):
        """Verify subclasses can implement write()."""
        class TestSink(LogSink):
            def __init__(self):
                self.entries = []

            def write(self, entry: LogEntry) -> None:
                self.entries.append(entry)

        sink = TestSink()
        entry = make_entry()
        sink.write(entry)

        assert len(sink.entries) == 1
        assert sink.entries[0] is entry


class TestConsoleSink:
    """Tests for ConsoleSink."""

    def test_basic_output(self):
        """Verify ConsoleSink writes to stream."""
        output = io.StringIO()
        sink = ConsoleSink(use_colors=False, stream=output)

        entry = make_entry("Hello World")
        sink.write(entry)

        content = output.getvalue()
        assert "Hello World" in content
        assert "INFO" in content

    def test_level_included_in_output(self):
        """Verify log level is included in output."""
        output = io.StringIO()
        sink = ConsoleSink(use_colors=False, stream=output)

        sink.write(make_entry(level=LogLevel.WARNING))
        sink.write(make_entry(level=LogLevel.ERROR))

        content = output.getvalue()
        assert "WARNING" in content
        assert "ERROR" in content

    def test_category_included_in_output(self):
        """Verify category is included when enabled."""
        output = io.StringIO()
        sink = ConsoleSink(
            use_colors=False,
            stream=output,
            include_category=True,
        )

        sink.write(make_entry(category=LogCategory.LogNetwork))

        content = output.getvalue()
        assert "LogNetwork" in content

    def test_category_excluded_when_disabled(self):
        """Verify category is excluded when disabled."""
        output = io.StringIO()
        sink = ConsoleSink(
            use_colors=False,
            stream=output,
            include_category=False,
        )

        sink.write(make_entry(category=LogCategory.LogNetwork))

        content = output.getvalue()
        assert "LogNetwork" not in content

    def test_timestamp_included(self):
        """Verify timestamp is included when enabled."""
        output = io.StringIO()
        sink = ConsoleSink(
            use_colors=False,
            stream=output,
            include_timestamp=True,
        )

        sink.write(make_entry())

        content = output.getvalue()
        # Should have date-like content
        assert "[20" in content  # Year starting with 20xx

    def test_fields_included(self):
        """Verify fields are included in output."""
        output = io.StringIO()
        sink = ConsoleSink(use_colors=False, stream=output)

        sink.write(make_entry(fields={"player_id": 123, "action": "jump"}))

        content = output.getvalue()
        assert "player_id=123" in content
        assert "action=jump" in content

    def test_colors_with_different_levels(self):
        """Verify different colors for different levels."""
        output = io.StringIO()
        sink = ConsoleSink(use_colors=True, stream=output)

        # Colors are tested by checking ANSI codes appear
        for level in LogLevel:
            sink.write(make_entry(level=level))

        content = output.getvalue()
        # Check reset code appears (common to all colored output)
        assert "\033[0m" in content or not sink.use_colors

    def test_flush(self):
        """Verify flush() works correctly."""
        output = io.StringIO()
        sink = ConsoleSink(use_colors=False, stream=output)

        sink.write(make_entry())
        sink.flush()
        # Should not raise


class TestFileSink:
    """Tests for FileSink."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    def test_basic_file_write(self, temp_dir):
        """Verify FileSink writes to file."""
        log_path = temp_dir / "test.log"
        sink = FileSink(log_path)

        sink.write(make_entry("Test message"))
        sink.close()

        content = log_path.read_text()
        assert "Test message" in content

    def test_file_append_mode(self, temp_dir):
        """Verify multiple writes append correctly."""
        log_path = temp_dir / "test.log"
        sink = FileSink(log_path)

        sink.write(make_entry("First"))
        sink.write(make_entry("Second"))
        sink.close()

        content = log_path.read_text()
        assert "First" in content
        assert "Second" in content

    def test_json_format(self, temp_dir):
        """Verify JSON format option works."""
        log_path = temp_dir / "test.log"
        sink = FileSink(log_path, json_format=True)

        sink.write(make_entry("Test", fields={"key": "value"}))
        sink.close()

        content = log_path.read_text().strip()
        parsed = json.loads(content)
        assert parsed["message"] == "Test"
        assert parsed["fields"]["key"] == "value"

    def test_file_rotation_by_size(self, temp_dir):
        """Verify rotation occurs when max_size is reached."""
        log_path = temp_dir / "test.log"
        # Small max_size for testing
        sink = FileSink(log_path, max_size=100, max_files=3)

        # Write enough to trigger rotation
        for i in range(20):
            sink.write(make_entry(f"Message {i} with some padding text"))

        sink.close()

        # Should have rotated files
        files = list(temp_dir.glob("test.log*"))
        assert len(files) > 1

    def test_rotation_with_compression(self, temp_dir):
        """Verify rotated files are compressed when enabled."""
        log_path = temp_dir / "test.log"
        sink = FileSink(
            log_path,
            max_size=100,
            max_files=3,
            compress_rotated=True,
        )

        # Write enough to trigger rotation
        for i in range(20):
            sink.write(make_entry(f"Message {i} with some padding text"))

        sink.close()

        # Should have .gz files
        gz_files = list(temp_dir.glob("*.gz"))
        assert len(gz_files) >= 1

    def test_max_files_limit(self, temp_dir):
        """Verify only max_files backups are kept."""
        log_path = temp_dir / "test.log"
        max_files = 2
        sink = FileSink(log_path, max_size=50, max_files=max_files)

        # Write a lot to trigger multiple rotations
        for i in range(50):
            sink.write(make_entry(f"Message {i}"))

        sink.close()

        # Count backup files (excluding current)
        all_files = list(temp_dir.glob("test.log*"))
        # Should be at most max_files + 1 (current file)
        assert len(all_files) <= max_files + 1

    def test_creates_parent_directory(self, temp_dir):
        """Verify parent directories are created."""
        log_path = temp_dir / "subdir" / "deep" / "test.log"
        sink = FileSink(log_path)

        sink.write(make_entry("Test"))
        sink.close()

        assert log_path.exists()

    def test_flush(self, temp_dir):
        """Verify flush() writes buffer to disk."""
        log_path = temp_dir / "test.log"
        sink = FileSink(log_path)

        sink.write(make_entry("Test"))
        sink.flush()

        content = log_path.read_text()
        assert "Test" in content

        sink.close()


class TestNetworkSink:
    """Tests for NetworkSink."""

    def test_invalid_protocol_raises(self):
        """Verify invalid protocol raises ValueError."""
        with pytest.raises(ValueError):
            NetworkSink("localhost", 5514, protocol="invalid")

    def test_tcp_initialization(self):
        """Verify TCP sink can be initialized."""
        sink = NetworkSink("localhost", 5514, protocol="tcp")
        assert sink.protocol == "tcp"
        sink.close()

    def test_udp_initialization(self):
        """Verify UDP sink can be initialized."""
        sink = NetworkSink("localhost", 5514, protocol="udp")
        assert sink.protocol == "udp"
        sink.close()

    def test_entry_queued(self):
        """Verify entries are queued for sending."""
        sink = NetworkSink("localhost", 5514, protocol="udp")

        sink.write(make_entry("Test"))

        # Entry should be in queue
        assert not sink._queue.empty()
        sink.close()


class TestBufferedSink:
    """Tests for BufferedSink."""

    def test_buffering(self):
        """Verify entries are buffered before writing."""
        inner_sink = MagicMock(spec=LogSink)
        sink = BufferedSink(inner_sink, buffer_size=5, flush_interval=100)

        # Write fewer than buffer_size
        for i in range(3):
            sink.write(make_entry(f"Message {i}"))

        # Should not have written yet
        inner_sink.write.assert_not_called()

        sink.close()

    def test_buffer_flush_on_size(self):
        """Verify buffer flushes when size reached."""
        inner_sink = MagicMock(spec=LogSink)
        sink = BufferedSink(inner_sink, buffer_size=3, flush_interval=100)

        # Write exactly buffer_size
        for i in range(3):
            sink.write(make_entry(f"Message {i}"))

        # Should have flushed
        assert inner_sink.write.call_count == 3

        sink.close()

    def test_manual_flush(self):
        """Verify manual flush() works."""
        inner_sink = MagicMock(spec=LogSink)
        sink = BufferedSink(inner_sink, buffer_size=100, flush_interval=100)

        sink.write(make_entry("Test"))
        inner_sink.write.assert_not_called()

        sink.flush()
        inner_sink.write.assert_called_once()

        sink.close()


class TestMultiplexSink:
    """Tests for MultiplexSink."""

    def test_writes_to_all_sinks(self):
        """Verify writes go to all underlying sinks."""
        sink1 = MagicMock(spec=LogSink)
        sink2 = MagicMock(spec=LogSink)
        sink3 = MagicMock(spec=LogSink)

        multiplex = MultiplexSink([sink1, sink2, sink3])
        multiplex.write(make_entry("Test"))

        sink1.write.assert_called_once()
        sink2.write.assert_called_once()
        sink3.write.assert_called_once()

    def test_add_sink(self):
        """Verify sinks can be added dynamically."""
        sink1 = MagicMock(spec=LogSink)
        sink2 = MagicMock(spec=LogSink)

        multiplex = MultiplexSink([sink1])
        multiplex.write(make_entry("First"))

        multiplex.add_sink(sink2)
        multiplex.write(make_entry("Second"))

        assert sink1.write.call_count == 2
        assert sink2.write.call_count == 1

    def test_remove_sink(self):
        """Verify sinks can be removed."""
        sink1 = MagicMock(spec=LogSink)
        sink2 = MagicMock(spec=LogSink)

        multiplex = MultiplexSink([sink1, sink2])
        multiplex.write(make_entry("First"))

        multiplex.remove_sink(sink1)
        multiplex.write(make_entry("Second"))

        assert sink1.write.call_count == 1
        assert sink2.write.call_count == 2

    def test_sink_error_doesnt_affect_others(self):
        """Verify one sink's error doesn't affect others."""
        sink1 = MagicMock(spec=LogSink)
        sink1.write.side_effect = RuntimeError("Sink error")
        sink2 = MagicMock(spec=LogSink)

        multiplex = MultiplexSink([sink1, sink2])
        multiplex.write(make_entry("Test"))

        # sink2 should still be called despite sink1's error
        sink2.write.assert_called_once()

    def test_flush_all(self):
        """Verify flush() flushes all sinks."""
        sink1 = MagicMock(spec=LogSink)
        sink2 = MagicMock(spec=LogSink)

        multiplex = MultiplexSink([sink1, sink2])
        multiplex.flush()

        sink1.flush.assert_called_once()
        sink2.flush.assert_called_once()

    def test_close_all(self):
        """Verify close() closes all sinks."""
        sink1 = MagicMock(spec=LogSink)
        sink2 = MagicMock(spec=LogSink)

        multiplex = MultiplexSink([sink1, sink2])
        multiplex.close()

        sink1.close.assert_called_once()
        sink2.close.assert_called_once()


class TestSinkThreadSafety:
    """Tests for sink thread safety."""

    def test_console_sink_thread_safety(self):
        """Verify ConsoleSink handles concurrent writes."""
        output = io.StringIO()
        sink = ConsoleSink(use_colors=False, stream=output)

        def write_messages(prefix):
            for i in range(50):
                sink.write(make_entry(f"{prefix}-{i}"))

        threads = [
            threading.Thread(target=write_messages, args=(f"thread-{i}",))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without errors
        lines = output.getvalue().strip().split("\n")
        assert len(lines) == 250

    def test_file_sink_thread_safety(self):
        """Verify FileSink handles concurrent writes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "test.log"
            sink = FileSink(log_path)

            def write_messages(prefix):
                for i in range(50):
                    sink.write(make_entry(f"{prefix}-{i}"))

            threads = [
                threading.Thread(target=write_messages, args=(f"thread-{i}",))
                for i in range(5)
            ]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            sink.close()

            # Should complete without errors
            lines = log_path.read_text().strip().split("\n")
            assert len(lines) == 250
