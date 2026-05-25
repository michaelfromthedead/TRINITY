"""Tests for log output targets.

Tests all log targets: console, file, network, ring buffer.
"""

import pytest
import tempfile
import socket
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

from engine.tooling.logging.log_system import LogMessage, LogLevel, LogCategory
from engine.tooling.logging.log_targets import (
    LogTarget,
    ConsoleTarget,
    FileTarget,
    NetworkTarget,
    RingBufferTarget,
    CompositeTarget,
)
from engine.tooling.logging.log_format import DefaultFormatter


@pytest.fixture
def sample_message():
    """Create a sample log message."""
    return LogMessage(
        level=LogLevel.INFO,
        category=LogCategory.ENGINE,
        message="Test message",
        context={"key": "value"}
    )


class TestLogTarget:
    """Tests for base LogTarget."""

    def test_enabled_property(self):
        target = RingBufferTarget()  # Concrete implementation
        assert target.enabled is True

        target.enabled = False
        assert target.enabled is False

    def test_name_property(self):
        target = RingBufferTarget(name="test_buffer")
        assert target.name == "test_buffer"


class TestConsoleTarget:
    """Tests for ConsoleTarget."""

    def test_creation(self):
        target = ConsoleTarget()
        assert target.enabled is True

    def test_write_to_stdout(self, sample_message):
        target = ConsoleTarget(use_stderr_for_errors=False, use_colors=False)

        with patch('sys.stdout') as mock_stdout:
            mock_stdout.write = MagicMock()
            mock_stdout.flush = MagicMock()
            target.write(sample_message)
            mock_stdout.write.assert_called()

    def test_write_error_to_stderr(self, sample_message):
        sample_message.level = LogLevel.ERROR
        target = ConsoleTarget(use_stderr_for_errors=True, use_colors=False)

        with patch('sys.stderr') as mock_stderr:
            mock_stderr.write = MagicMock()
            mock_stderr.flush = MagicMock()
            target.write(sample_message)
            mock_stderr.write.assert_called()

    def test_disabled_target(self, sample_message):
        target = ConsoleTarget()
        target.enabled = False

        with patch('sys.stdout') as mock_stdout:
            mock_stdout.write = MagicMock()
            target.write(sample_message)
            mock_stdout.write.assert_not_called()


class TestFileTarget:
    """Tests for FileTarget."""

    def test_basic_write(self, sample_message):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.log"
            target = FileTarget(path)

            target.write(sample_message)
            target.close()

            content = path.read_text()
            assert "Test message" in content

    def test_append_mode(self, sample_message):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.log"

            # Write first message
            target1 = FileTarget(path, mode="a")
            target1.write(sample_message)
            target1.close()

            # Write second message
            target2 = FileTarget(path, mode="a")
            sample_message.message = "Second message"
            target2.write(sample_message)
            target2.close()

            content = path.read_text()
            assert "Test message" in content
            assert "Second message" in content

    def test_overwrite_mode(self, sample_message):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.log"
            path.write_text("Existing content\n")

            target = FileTarget(path, mode="w")
            target.write(sample_message)
            target.close()

            content = path.read_text()
            assert "Existing content" not in content
            assert "Test message" in content

    def test_rotation(self, sample_message):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.log"
            target = FileTarget(path, max_size=100, max_files=3)

            # Write enough messages to trigger rotation
            for i in range(50):
                sample_message.message = f"Message {i}" + "x" * 50
                target.write(sample_message)

            target.close()

            # Check rotated files exist
            assert path.exists()
            rotated = path.with_suffix(".1.log")
            assert rotated.exists()

    def test_creates_directory(self, sample_message):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "nested" / "test.log"
            target = FileTarget(path)

            target.write(sample_message)
            target.close()

            assert path.exists()

    def test_disabled_target(self, sample_message):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.log"
            target = FileTarget(path)
            target.enabled = False

            target.write(sample_message)
            target.close()

            # File should exist but be empty (opened but not written to)
            content = path.read_text()
            assert "Test message" not in content


class TestNetworkTarget:
    """Tests for NetworkTarget."""

    def test_udp_creation(self):
        target = NetworkTarget("localhost", 9999, protocol="udp")
        assert target._connected is True
        target.close()

    def test_udp_write(self, sample_message):
        # Create UDP server
        server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server.bind(("localhost", 0))
        port = server.getsockname()[1]
        server.settimeout(1.0)

        target = NetworkTarget("localhost", port, protocol="udp", use_json=True)

        try:
            target.write(sample_message)
            data, _ = server.recvfrom(4096)
            parsed = json.loads(data.decode('utf-8'))

            assert parsed["message"] == "Test message"
            assert parsed["level"] == "INFO"
        finally:
            target.close()
            server.close()

    def test_disabled_target(self, sample_message):
        target = NetworkTarget("localhost", 9999, protocol="udp")
        target.enabled = False

        # Should not raise even with no server
        target.write(sample_message)
        target.close()

    def test_json_format(self, sample_message):
        server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server.bind(("localhost", 0))
        port = server.getsockname()[1]
        server.settimeout(1.0)

        target = NetworkTarget("localhost", port, protocol="udp", use_json=True)

        try:
            target.write(sample_message)
            data, _ = server.recvfrom(4096)
            parsed = json.loads(data.decode('utf-8'))

            assert "level" in parsed
            assert "category" in parsed
            assert "message" in parsed
            assert "timestamp" in parsed
        finally:
            target.close()
            server.close()


class TestRingBufferTarget:
    """Tests for RingBufferTarget."""

    def test_basic_creation(self):
        target = RingBufferTarget(capacity=100)
        assert target.capacity == 100
        assert target.count == 0

    def test_write(self, sample_message):
        target = RingBufferTarget()
        target.write(sample_message)
        assert target.count == 1

    def test_capacity_limit(self):
        target = RingBufferTarget(capacity=5)

        for i in range(10):
            msg = LogMessage(
                level=LogLevel.INFO,
                category=LogCategory.ENGINE,
                message=f"Message {i}"
            )
            target.write(msg)

        assert target.count == 5
        entries = target.get_entries()
        assert entries[0].message.message == "Message 5"

    def test_get_entries(self, sample_message):
        target = RingBufferTarget()

        for i in range(5):
            sample_message.message = f"Message {i}"
            target.write(sample_message)

        entries = target.get_entries()
        assert len(entries) == 5

    def test_get_entries_count(self, sample_message):
        target = RingBufferTarget()

        for i in range(10):
            sample_message.message = f"Message {i}"
            target.write(sample_message)

        entries = target.get_entries(count=3)
        assert len(entries) == 3
        assert entries[-1].message.message == "Message 9"

    def test_get_entries_level_filter(self):
        target = RingBufferTarget()

        msg1 = LogMessage(
            level=LogLevel.INFO,
            category=LogCategory.ENGINE,
            message="Info"
        )
        target.write(msg1)

        msg2 = LogMessage(
            level=LogLevel.WARNING,
            category=LogCategory.ENGINE,
            message="Warning"
        )
        target.write(msg2)

        msg3 = LogMessage(
            level=LogLevel.ERROR,
            category=LogCategory.ENGINE,
            message="Error"
        )
        target.write(msg3)

        entries = target.get_entries(level=LogLevel.WARNING)
        assert len(entries) == 2

    def test_get_messages(self, sample_message):
        target = RingBufferTarget()
        target.write(sample_message)

        messages = target.get_messages()
        assert len(messages) == 1
        assert messages[0].message == "Test message"

    def test_get_formatted(self, sample_message):
        target = RingBufferTarget()
        target.write(sample_message, DefaultFormatter())

        formatted = target.get_formatted()
        assert len(formatted) == 1
        assert "Test message" in formatted[0]

    def test_search(self):
        target = RingBufferTarget()

        msg1 = LogMessage(
            level=LogLevel.INFO,
            category=LogCategory.ENGINE,
            message="Error in physics"
        )
        target.write(msg1)

        msg2 = LogMessage(
            level=LogLevel.INFO,
            category=LogCategory.ENGINE,
            message="Rendering complete"
        )
        target.write(msg2)

        msg3 = LogMessage(
            level=LogLevel.INFO,
            category=LogCategory.ENGINE,
            message="Another error"
        )
        target.write(msg3)

        results = target.search("error")
        assert len(results) == 2

    def test_clear(self, sample_message):
        target = RingBufferTarget()
        target.write(sample_message)
        target.write(sample_message)

        target.clear()
        assert target.count == 0

    def test_iteration(self):
        target = RingBufferTarget()

        for i in range(3):
            msg = LogMessage(
                level=LogLevel.INFO,
                category=LogCategory.ENGINE,
                message=f"Message {i}"
            )
            target.write(msg)

        messages = [e.message.message for e in target]
        assert messages == ["Message 0", "Message 1", "Message 2"]

    def test_index_tracking(self, sample_message):
        target = RingBufferTarget()

        for i in range(3):
            target.write(sample_message)

        entries = target.get_entries()
        assert entries[0].index == 0
        assert entries[1].index == 1
        assert entries[2].index == 2


class TestCompositeTarget:
    """Tests for CompositeTarget."""

    def test_basic_creation(self):
        target = CompositeTarget()
        assert len(target.targets) == 0

    def test_with_children(self, sample_message):
        child1 = RingBufferTarget()
        child2 = RingBufferTarget()
        composite = CompositeTarget([child1, child2])

        composite.write(sample_message)

        assert child1.count == 1
        assert child2.count == 1

    def test_add_target(self, sample_message):
        composite = CompositeTarget()
        child = RingBufferTarget()

        composite.add_target(child)
        composite.write(sample_message)

        assert child.count == 1

    def test_remove_target(self, sample_message):
        child = RingBufferTarget()
        composite = CompositeTarget([child])

        composite.remove_target(child)
        composite.write(sample_message)

        assert child.count == 0

    def test_disabled_composite(self, sample_message):
        child = RingBufferTarget()
        composite = CompositeTarget([child])
        composite.enabled = False

        composite.write(sample_message)
        assert child.count == 0

    def test_child_exception_handled(self, sample_message):
        child1 = RingBufferTarget()

        class BadTarget(LogTarget):
            def write(self, message, formatter=None):
                raise Exception("Target error")

        bad_child = BadTarget()
        composite = CompositeTarget([bad_child, child1])

        # Should not raise
        composite.write(sample_message)

        # Good child should still receive message
        assert child1.count == 1

    def test_close(self):
        child1 = RingBufferTarget()
        child2 = RingBufferTarget()
        composite = CompositeTarget([child1, child2])

        composite.close()  # Should not raise
