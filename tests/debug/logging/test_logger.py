"""
Comprehensive tests for the Logger class and related components.

Tests cover:
- LogLevel enum ordering and string representation
- LogCategory enum values
- Logger instance creation and configuration
- Log level filtering per category
- Structured logging with fields
- Callbacks and sinks integration
- Thread safety
"""

import pytest
import sys
import threading
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

sys.path.insert(0, "/home/user/dev/AI_GAME_ENGINE")

from engine.debug.logging.logger import (
    Logger,
    LogLevel,
    LogCategory,
    LogEntry,
    get_logger,
)


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_level_ordering(self):
        """Verify log levels are ordered from verbose to fatal."""
        assert LogLevel.VERBOSE < LogLevel.DEBUG
        assert LogLevel.DEBUG < LogLevel.INFO
        assert LogLevel.INFO < LogLevel.WARNING
        assert LogLevel.WARNING < LogLevel.ERROR
        assert LogLevel.ERROR < LogLevel.FATAL

    def test_level_values(self):
        """Verify level numeric values are spaced correctly."""
        assert LogLevel.VERBOSE.value == 0
        assert LogLevel.DEBUG.value == 10
        assert LogLevel.INFO.value == 20
        assert LogLevel.WARNING.value == 30
        assert LogLevel.ERROR.value == 40
        assert LogLevel.FATAL.value == 50

    def test_level_string_representation(self):
        """Verify string conversion of log levels."""
        assert str(LogLevel.VERBOSE) == "VERBOSE"
        assert str(LogLevel.DEBUG) == "DEBUG"
        assert str(LogLevel.INFO) == "INFO"
        assert str(LogLevel.WARNING) == "WARNING"
        assert str(LogLevel.ERROR) == "ERROR"
        assert str(LogLevel.FATAL) == "FATAL"

    def test_level_comparison(self):
        """Verify levels can be compared."""
        assert LogLevel.WARNING >= LogLevel.INFO
        assert LogLevel.ERROR > LogLevel.WARNING
        assert LogLevel.DEBUG <= LogLevel.DEBUG


class TestLogCategory:
    """Tests for LogCategory enum."""

    def test_all_categories_exist(self):
        """Verify all expected categories are defined."""
        expected = [
            "LogEngine",
            "LogRendering",
            "LogPhysics",
            "LogAI",
            "LogNetwork",
            "LogAudio",
            "LogAnimation",
            "LogInput",
            "LogGameplay",
            "LogPlayer",
            "LogUI",
        ]
        for name in expected:
            assert hasattr(LogCategory, name)

    def test_category_string_representation(self):
        """Verify string conversion of categories."""
        assert str(LogCategory.LogEngine) == "LogEngine"
        assert str(LogCategory.LogRendering) == "LogRendering"

    def test_categories_are_unique(self):
        """Verify all categories have unique values."""
        values = [c.value for c in LogCategory]
        assert len(values) == len(set(values))


class TestLogEntry:
    """Tests for LogEntry dataclass."""

    def test_entry_creation(self):
        """Verify LogEntry can be created with all fields."""
        now = datetime.now(timezone.utc)
        entry = LogEntry(
            timestamp=now,
            level=LogLevel.INFO,
            category=LogCategory.LogEngine,
            message="Test message",
            logger_name="TestLogger",
            fields={"key": "value"},
            source_file="test.py",
            source_line=42,
        )

        assert entry.timestamp == now
        assert entry.level == LogLevel.INFO
        assert entry.category == LogCategory.LogEngine
        assert entry.message == "Test message"
        assert entry.logger_name == "TestLogger"
        assert entry.fields == {"key": "value"}
        assert entry.source_file == "test.py"
        assert entry.source_line == 42

    def test_entry_to_dict(self):
        """Verify LogEntry converts to dictionary correctly."""
        now = datetime.now(timezone.utc)
        entry = LogEntry(
            timestamp=now,
            level=LogLevel.WARNING,
            category=LogCategory.LogNetwork,
            message="Connection timeout",
            logger_name="Network",
            fields={"host": "example.com", "port": 8080},
        )

        d = entry.to_dict()
        assert d["timestamp"] == now.isoformat()
        assert d["level"] == "WARNING"
        assert d["category"] == "LogNetwork"
        assert d["message"] == "Connection timeout"
        assert d["logger"] == "Network"
        assert d["fields"]["host"] == "example.com"

    def test_entry_to_json(self):
        """Verify LogEntry converts to JSON correctly."""
        import json

        now = datetime.now(timezone.utc)
        entry = LogEntry(
            timestamp=now,
            level=LogLevel.ERROR,
            category=LogCategory.LogEngine,
            message="Fatal error",
            logger_name="Engine",
        )

        json_str = entry.to_json()
        parsed = json.loads(json_str)
        assert parsed["level"] == "ERROR"
        assert parsed["message"] == "Fatal error"

    def test_entry_with_source_info(self):
        """Verify source info is included in to_dict when present."""
        entry = LogEntry(
            timestamp=datetime.now(timezone.utc),
            level=LogLevel.DEBUG,
            category=LogCategory.LogEngine,
            message="Debug info",
            logger_name="Test",
            source_file="module.py",
            source_line=100,
        )

        d = entry.to_dict()
        assert "source" in d
        assert d["source"]["file"] == "module.py"
        assert d["source"]["line"] == 100


class TestLoggerBasic:
    """Basic tests for Logger class."""

    def test_logger_creation(self):
        """Verify logger can be created with a name."""
        logger = Logger("TestLogger")
        assert logger.name == "TestLogger"
        assert logger.enabled is True

    def test_get_logger_function(self):
        """Verify get_logger returns a Logger instance."""
        logger = get_logger("GameEngine")
        assert isinstance(logger, Logger)
        assert logger.name == "GameEngine"

    def test_logger_enable_disable(self):
        """Verify logger can be enabled and disabled."""
        logger = Logger("Test")
        assert logger.enabled is True

        logger.enabled = False
        assert logger.enabled is False

        logger.enabled = True
        assert logger.enabled is True


class TestLoggerLevels:
    """Tests for Logger level filtering."""

    @pytest.fixture
    def logger(self):
        """Create a test logger with captured entries."""
        logger = Logger("TestLogger")
        logger.entries = []

        def capture(entry):
            logger.entries.append(entry)

        logger.add_callback(capture)
        return logger

    def test_default_level(self, logger):
        """Verify default level is INFO."""
        assert logger.get_level(LogCategory.LogEngine) == LogLevel.INFO

    def test_set_level(self, logger):
        """Verify level can be set per category."""
        logger.set_level(LogCategory.LogRendering, LogLevel.DEBUG)
        assert logger.get_level(LogCategory.LogRendering) == LogLevel.DEBUG

        # Other categories unchanged
        assert logger.get_level(LogCategory.LogEngine) == LogLevel.INFO

    def test_reset_level(self, logger):
        """Verify level can be reset to default."""
        logger.set_level(LogCategory.LogPhysics, LogLevel.ERROR)
        logger.reset_level(LogCategory.LogPhysics)
        assert logger.get_level(LogCategory.LogPhysics) == LogLevel.INFO

    def test_level_filtering(self, logger):
        """Verify messages below level are filtered."""
        logger.set_level(LogCategory.LogEngine, LogLevel.WARNING)

        logger.debug("Debug message", LogCategory.LogEngine)
        logger.info("Info message", LogCategory.LogEngine)
        logger.warning("Warning message", LogCategory.LogEngine)
        logger.error("Error message", LogCategory.LogEngine)

        # Only WARNING and above should be logged
        assert len(logger.entries) == 2
        assert logger.entries[0].level == LogLevel.WARNING
        assert logger.entries[1].level == LogLevel.ERROR

    def test_disabled_logger_logs_nothing(self, logger):
        """Verify disabled logger doesn't log anything."""
        logger.enabled = False

        logger.info("Info message", LogCategory.LogEngine)
        logger.error("Error message", LogCategory.LogEngine)
        logger.fatal("Fatal message", LogCategory.LogEngine)

        assert len(logger.entries) == 0


class TestLoggerMethods:
    """Tests for Logger logging methods."""

    @pytest.fixture
    def logger(self):
        """Create a test logger with captured entries."""
        logger = Logger("TestLogger")
        logger.entries = []

        def capture(entry):
            logger.entries.append(entry)

        logger.add_callback(capture)
        # Set to VERBOSE to capture all messages
        Logger.set_default_level(LogLevel.VERBOSE)
        return logger

    @pytest.fixture(autouse=True)
    def reset_default_level(self):
        """Reset default level after each test."""
        yield
        Logger.set_default_level(LogLevel.INFO)

    def test_verbose_method(self, logger):
        """Verify verbose() logs at VERBOSE level."""
        logger.verbose("Verbose message", LogCategory.LogEngine)
        assert len(logger.entries) == 1
        assert logger.entries[0].level == LogLevel.VERBOSE

    def test_debug_method(self, logger):
        """Verify debug() logs at DEBUG level."""
        logger.debug("Debug message", LogCategory.LogEngine)
        assert len(logger.entries) == 1
        assert logger.entries[0].level == LogLevel.DEBUG

    def test_info_method(self, logger):
        """Verify info() logs at INFO level."""
        logger.info("Info message", LogCategory.LogEngine)
        assert len(logger.entries) == 1
        assert logger.entries[0].level == LogLevel.INFO

    def test_warning_method(self, logger):
        """Verify warning() logs at WARNING level."""
        logger.warning("Warning message", LogCategory.LogEngine)
        assert len(logger.entries) == 1
        assert logger.entries[0].level == LogLevel.WARNING

    def test_error_method(self, logger):
        """Verify error() logs at ERROR level."""
        logger.error("Error message", LogCategory.LogEngine)
        assert len(logger.entries) == 1
        assert logger.entries[0].level == LogLevel.ERROR

    def test_fatal_method(self, logger):
        """Verify fatal() logs at FATAL level."""
        logger.fatal("Fatal message", LogCategory.LogEngine)
        assert len(logger.entries) == 1
        assert logger.entries[0].level == LogLevel.FATAL

    def test_default_category(self, logger):
        """Verify default category is LogEngine."""
        logger.info("Test message")
        assert logger.entries[0].category == LogCategory.LogEngine


class TestStructuredLogging:
    """Tests for structured logging functionality."""

    @pytest.fixture
    def logger(self):
        """Create a test logger with captured entries."""
        logger = Logger("TestLogger")
        logger.entries = []

        def capture(entry):
            logger.entries.append(entry)

        logger.add_callback(capture)
        return logger

    def test_log_with_fields(self, logger):
        """Verify logging with additional fields."""
        logger.info(
            "Player action",
            LogCategory.LogPlayer,
            player_id=123,
            action="jump",
        )

        entry = logger.entries[0]
        assert entry.fields["player_id"] == 123
        assert entry.fields["action"] == "jump"

    def test_structured_method(self, logger):
        """Verify structured() method works correctly."""
        logger.structured(
            "Request completed",
            LogCategory.LogNetwork,
            level=LogLevel.DEBUG,
            request_id="abc123",
            latency_ms=45.2,
            status_code=200,
        )

        # Set level to allow DEBUG
        logger.set_level(LogCategory.LogNetwork, LogLevel.DEBUG)
        logger.structured(
            "Request completed",
            LogCategory.LogNetwork,
            level=LogLevel.DEBUG,
            request_id="abc123",
        )

        # Check last entry (after level change)
        entry = logger.entries[-1]
        assert entry.level == LogLevel.DEBUG
        assert entry.fields["request_id"] == "abc123"

    def test_log_exception(self, logger):
        """Verify log_exception captures exception info."""
        try:
            raise ValueError("Test error")
        except ValueError as e:
            logger.log_exception(e, "Operation failed", LogCategory.LogEngine)

        entry = logger.entries[0]
        assert entry.level == LogLevel.ERROR
        assert "exception_type" in entry.fields
        assert entry.fields["exception_type"] == "ValueError"
        assert "traceback" in entry.fields


class TestLoggerCallbacks:
    """Tests for Logger callback functionality."""

    def test_add_callback(self):
        """Verify callbacks can be added and invoked."""
        logger = Logger("Test")
        received = []

        def callback(entry):
            received.append(entry)

        logger.add_callback(callback)
        logger.info("Test message")

        assert len(received) == 1
        assert received[0].message == "Test message"

    def test_remove_callback(self):
        """Verify callbacks can be removed."""
        logger = Logger("Test")
        received = []

        def callback(entry):
            received.append(entry)

        logger.add_callback(callback)
        logger.info("First")

        logger.remove_callback(callback)
        logger.info("Second")

        assert len(received) == 1
        assert received[0].message == "First"

    def test_callback_exception_doesnt_crash(self):
        """Verify callback exceptions don't crash logging."""
        logger = Logger("Test")

        def bad_callback(entry):
            raise RuntimeError("Callback error")

        logger.add_callback(bad_callback)

        # Should not raise
        logger.info("Test message")


class TestLoggerSinks:
    """Tests for Logger sink functionality."""

    def test_add_sink(self):
        """Verify sinks can be added."""
        from engine.debug.logging.sinks import LogSink

        logger = Logger("Test")
        sink = MagicMock(spec=LogSink)

        logger.add_sink(sink)
        logger.info("Test message")

        sink.write.assert_called_once()

    def test_remove_sink(self):
        """Verify sinks can be removed."""
        from engine.debug.logging.sinks import LogSink

        logger = Logger("Test")
        sink = MagicMock(spec=LogSink)

        logger.add_sink(sink)
        logger.info("First")

        logger.remove_sink(sink)
        logger.info("Second")

        assert sink.write.call_count == 1


class TestLoggerGlobal:
    """Tests for Logger global configuration."""

    @pytest.fixture(autouse=True)
    def clear_global_state(self):
        """Clear global sinks and filters after each test."""
        yield
        Logger.clear_global_sinks()
        Logger.clear_global_filters()
        Logger.set_default_level(LogLevel.INFO)

    def test_add_global_sink(self):
        """Verify global sinks receive messages from all loggers."""
        from engine.debug.logging.sinks import LogSink

        sink = MagicMock(spec=LogSink)
        Logger.add_global_sink(sink)

        logger1 = Logger("Logger1")
        logger2 = Logger("Logger2")

        logger1.info("Message 1")
        logger2.info("Message 2")

        assert sink.write.call_count == 2

    def test_remove_global_sink(self):
        """Verify global sinks can be removed."""
        from engine.debug.logging.sinks import LogSink

        sink = MagicMock(spec=LogSink)
        Logger.add_global_sink(sink)

        logger = Logger("Test")
        logger.info("First")

        Logger.remove_global_sink(sink)
        logger.info("Second")

        assert sink.write.call_count == 1

    def test_set_default_level(self):
        """Verify default level can be changed globally."""
        Logger.set_default_level(LogLevel.WARNING)

        logger = Logger("Test")
        assert logger.get_level(LogCategory.LogEngine) == LogLevel.WARNING


class TestLoggerThreadSafety:
    """Tests for Logger thread safety."""

    def test_concurrent_logging(self):
        """Verify concurrent logging is thread-safe."""
        logger = Logger("Test")
        received = []
        lock = threading.Lock()

        def callback(entry):
            with lock:
                received.append(entry)

        logger.add_callback(callback)

        def log_messages(prefix):
            for i in range(100):
                logger.info(f"{prefix}-{i}", LogCategory.LogEngine)

        threads = [
            threading.Thread(target=log_messages, args=(f"thread-{i}",))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(received) == 500

    def test_concurrent_level_changes(self):
        """Verify concurrent level changes are thread-safe."""
        logger = Logger("Test")
        errors = []
        lock = threading.Lock()

        def change_levels():
            try:
                for _ in range(100):
                    logger.set_level(LogCategory.LogEngine, LogLevel.DEBUG)
                    level = logger.get_level(LogCategory.LogEngine)
                    # Level should always be a valid LogLevel
                    if level not in (LogLevel.DEBUG, LogLevel.WARNING, LogLevel.INFO):
                        with lock:
                            errors.append(f"Invalid level: {level}")
                    logger.set_level(LogCategory.LogEngine, LogLevel.WARNING)
            except Exception as e:
                with lock:
                    errors.append(str(e))

        threads = [
            threading.Thread(target=change_levels)
            for _ in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify no errors occurred during concurrent access
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        # Final level should be one of the set values
        final_level = logger.get_level(LogCategory.LogEngine)
        assert final_level in (LogLevel.DEBUG, LogLevel.WARNING, LogLevel.INFO)
