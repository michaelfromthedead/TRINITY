"""Tests for the centralized logging system.

Tests all log levels, categories, and core functionality.
"""

import pytest
import threading
import time

from engine.tooling.logging.log_system import (
    LogSystem,
    LogLevel,
    LogCategory,
    LogMessage,
    LogConfig,
)
from engine.tooling.logging.log_targets import RingBufferTarget


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_level_ordering(self):
        assert LogLevel.TRACE < LogLevel.DEBUG
        assert LogLevel.DEBUG < LogLevel.INFO
        assert LogLevel.INFO < LogLevel.WARNING
        assert LogLevel.WARNING < LogLevel.ERROR
        assert LogLevel.ERROR < LogLevel.FATAL

    def test_short_names(self):
        assert LogLevel.TRACE.name_short == "TRC"
        assert LogLevel.DEBUG.name_short == "DBG"
        assert LogLevel.INFO.name_short == "INF"
        assert LogLevel.WARNING.name_short == "WRN"
        assert LogLevel.ERROR.name_short == "ERR"
        assert LogLevel.FATAL.name_short == "FTL"


class TestLogCategory:
    """Tests for LogCategory enum."""

    def test_from_string(self):
        assert LogCategory.from_string("ENGINE") == LogCategory.ENGINE
        assert LogCategory.from_string("engine") == LogCategory.ENGINE
        assert LogCategory.from_string("RENDER") == LogCategory.RENDER
        assert LogCategory.from_string("unknown") == LogCategory.CUSTOM

    def test_all_categories(self):
        # Ensure all expected categories exist
        expected = [
            "ENGINE", "GAME", "RENDER", "PHYSICS", "AI",
            "NETWORK", "AUDIO", "EDITOR", "INPUT", "RESOURCE",
            "SCRIPT", "UI", "MEMORY", "PROFILE", "CUSTOM"
        ]
        for name in expected:
            assert hasattr(LogCategory, name)


class TestLogMessage:
    """Tests for LogMessage."""

    def test_basic_creation(self):
        msg = LogMessage(
            level=LogLevel.INFO,
            category=LogCategory.ENGINE,
            message="Test message"
        )
        assert msg.level == LogLevel.INFO
        assert msg.category == LogCategory.ENGINE
        assert msg.message == "Test message"
        assert msg.timestamp is not None
        assert msg.thread_id is not None

    def test_format_source_full(self):
        msg = LogMessage(
            level=LogLevel.DEBUG,
            category=LogCategory.ENGINE,
            message="Test",
            file="test.py",
            line=42,
            function="test_func"
        )
        source = msg.format_source()
        assert "test.py" in source
        assert "42" in source
        assert "test_func" in source

    def test_format_source_partial(self):
        msg = LogMessage(
            level=LogLevel.DEBUG,
            category=LogCategory.ENGINE,
            message="Test",
            file="test.py"
        )
        source = msg.format_source()
        assert source == "test.py"

    def test_format_source_empty(self):
        msg = LogMessage(
            level=LogLevel.DEBUG,
            category=LogCategory.ENGINE,
            message="Test"
        )
        source = msg.format_source()
        assert source == ""


class TestLogConfig:
    """Tests for LogConfig."""

    def test_defaults(self):
        config = LogConfig()
        assert config.min_level == LogLevel.INFO
        assert config.enabled_categories is None  # All enabled
        assert config.async_logging is True

    def test_custom_config(self):
        config = LogConfig(
            min_level=LogLevel.DEBUG,
            disabled_categories={LogCategory.AUDIO},
            async_logging=False
        )
        assert config.min_level == LogLevel.DEBUG
        assert LogCategory.AUDIO in config.disabled_categories


class TestLogSystem:
    """Tests for LogSystem core functionality."""

    @pytest.fixture(autouse=True)
    def reset_system(self):
        """Reset singleton before and after each test."""
        LogSystem.reset_instance()
        yield
        LogSystem.reset_instance()

    @pytest.fixture
    def log_system(self):
        """Create a log system with ring buffer target."""
        config = LogConfig(async_logging=False)  # Sync for easier testing
        system = LogSystem(config)
        target = RingBufferTarget(capacity=100)
        system.add_target(target)
        return system, target

    def test_singleton(self):
        instance1 = LogSystem.get_instance()
        instance2 = LogSystem.get_instance()
        assert instance1 is instance2

    def test_basic_logging(self, log_system):
        system, target = log_system
        system.info("Test message")

        entries = target.get_entries()
        assert len(entries) == 1
        assert entries[0].message.message == "Test message"
        assert entries[0].message.level == LogLevel.INFO

    def test_all_levels(self, log_system):
        system, target = log_system
        system.config.min_level = LogLevel.TRACE

        system.trace("Trace message")
        system.debug("Debug message")
        system.info("Info message")
        system.warning("Warning message")
        system.error("Error message")
        system.fatal("Fatal message")

        entries = target.get_entries()
        assert len(entries) == 6

    def test_level_filtering(self, log_system):
        system, target = log_system
        system.set_level(LogLevel.WARNING)

        system.debug("Debug")
        system.info("Info")
        system.warning("Warning")
        system.error("Error")

        entries = target.get_entries()
        assert len(entries) == 2
        assert all(e.message.level >= LogLevel.WARNING for e in entries)

    def test_category_logging(self, log_system):
        system, target = log_system

        system.info("Engine message", LogCategory.ENGINE)
        system.info("Render message", LogCategory.RENDER)
        system.info("Audio message", LogCategory.AUDIO)

        entries = target.get_entries()
        assert len(entries) == 3
        categories = [e.message.category for e in entries]
        assert LogCategory.ENGINE in categories
        assert LogCategory.RENDER in categories
        assert LogCategory.AUDIO in categories

    def test_disable_category(self, log_system):
        system, target = log_system

        system.disable_category(LogCategory.AUDIO)

        system.info("Engine message", LogCategory.ENGINE)
        system.info("Audio message", LogCategory.AUDIO)

        entries = target.get_entries()
        assert len(entries) == 1
        assert entries[0].message.category == LogCategory.ENGINE

    def test_enable_category(self, log_system):
        system, target = log_system
        system.config.enabled_categories = {LogCategory.ENGINE}

        system.info("Engine message", LogCategory.ENGINE)
        system.info("Render message", LogCategory.RENDER)

        entries = target.get_entries()
        assert len(entries) == 1
        assert entries[0].message.category == LogCategory.ENGINE

        system.enable_category(LogCategory.RENDER)
        system.info("Render message 2", LogCategory.RENDER)

        entries = target.get_entries()
        assert len(entries) == 2

    def test_context_data(self, log_system):
        system, target = log_system

        system.info(
            "Test message",
            context={"user_id": "123", "action": "login"}
        )

        entries = target.get_entries()
        assert entries[0].message.context["user_id"] == "123"
        assert entries[0].message.context["action"] == "login"

    def test_exception_logging(self, log_system):
        system, target = log_system

        try:
            raise ValueError("Test error")
        except ValueError as e:
            system.error("Error occurred", exception=e)

        entries = target.get_entries()
        assert entries[0].message.exception is not None
        assert isinstance(entries[0].message.exception, ValueError)

    def test_message_truncation(self):
        config = LogConfig(max_message_length=20, async_logging=False)
        system = LogSystem(config)
        target = RingBufferTarget()
        system.add_target(target)

        system.info("This is a very long message that should be truncated")

        entries = target.get_entries()
        assert len(entries[0].message.message) <= 23  # 20 + "..."

    def test_enabled_toggle(self, log_system):
        system, target = log_system

        system.enabled = False
        system.info("Should not appear")

        entries = target.get_entries()
        assert len(entries) == 0

        system.enabled = True
        system.info("Should appear")

        entries = target.get_entries()
        assert len(entries) == 1


class TestLogSystemTargets:
    """Tests for log system target management."""

    @pytest.fixture(autouse=True)
    def reset_system(self):
        LogSystem.reset_instance()
        yield
        LogSystem.reset_instance()

    def test_add_target(self):
        config = LogConfig(async_logging=False)
        system = LogSystem(config)
        target = RingBufferTarget()

        system.add_target(target)
        system.info("Test")

        assert target.count == 1

    def test_remove_target(self):
        config = LogConfig(async_logging=False)
        system = LogSystem(config)
        target = RingBufferTarget()

        system.add_target(target)
        system.remove_target(target)
        system.info("Test")

        assert target.count == 0

    def test_multiple_targets(self):
        config = LogConfig(async_logging=False)
        system = LogSystem(config)
        target1 = RingBufferTarget()
        target2 = RingBufferTarget()

        system.add_target(target1)
        system.add_target(target2)
        system.info("Test")

        assert target1.count == 1
        assert target2.count == 1


class TestLogSystemCallbacks:
    """Tests for log system callbacks."""

    @pytest.fixture(autouse=True)
    def reset_system(self):
        LogSystem.reset_instance()
        yield
        LogSystem.reset_instance()

    def test_callback(self):
        config = LogConfig(async_logging=False)
        system = LogSystem(config)
        received = []

        def callback(msg: LogMessage):
            received.append(msg)

        system.add_callback(callback)
        system.info("Test message")

        assert len(received) == 1
        assert received[0].message == "Test message"

    def test_remove_callback(self):
        config = LogConfig(async_logging=False)
        system = LogSystem(config)
        received = []

        def callback(msg: LogMessage):
            received.append(msg)

        system.add_callback(callback)
        system.remove_callback(callback)
        system.info("Test message")

        assert len(received) == 0

    def test_callback_exception_handled(self):
        config = LogConfig(async_logging=False)
        system = LogSystem(config)
        target = RingBufferTarget()
        system.add_target(target)

        def bad_callback(msg: LogMessage):
            raise Exception("Callback error")

        system.add_callback(bad_callback)
        system.info("Test")  # Should not raise

        assert target.count == 1


class TestLogSystemAsync:
    """Tests for async logging."""

    @pytest.fixture(autouse=True)
    def reset_system(self):
        LogSystem.reset_instance()
        yield
        LogSystem.reset_instance()

    def test_async_logging(self):
        config = LogConfig(async_logging=True, flush_interval=0.01)
        system = LogSystem(config)
        target = RingBufferTarget()
        system.add_target(target)

        system.info("Test message")

        # Message should be buffered
        time.sleep(0.05)  # Wait for flush

        assert target.count == 1

    def test_flush(self):
        config = LogConfig(async_logging=True, flush_interval=10.0)  # Long interval
        system = LogSystem(config)
        target = RingBufferTarget()
        system.add_target(target)

        system.info("Test message")
        assert target.count == 0  # Buffered

        system.flush()
        assert target.count == 1

    def test_shutdown(self):
        config = LogConfig(async_logging=True)
        system = LogSystem(config)
        target = RingBufferTarget()
        system.add_target(target)

        system.info("Test message")
        system.shutdown()

        assert target.count == 1  # Flushed on shutdown


class TestLogSystemThreadSafety:
    """Tests for thread-safe logging."""

    @pytest.fixture(autouse=True)
    def reset_system(self):
        LogSystem.reset_instance()
        yield
        LogSystem.reset_instance()

    def test_concurrent_logging(self):
        config = LogConfig(async_logging=False)
        system = LogSystem(config)
        target = RingBufferTarget(capacity=10000)
        system.add_target(target)

        errors = []

        def log_messages(thread_id):
            try:
                for i in range(100):
                    system.info(f"Thread {thread_id} message {i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=log_messages, args=(i,))
                  for i in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert target.count == 1000


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture(autouse=True)
    def reset_system(self):
        LogSystem.reset_instance()
        yield
        LogSystem.reset_instance()

    def test_module_functions(self):
        from engine.tooling.logging.log_system import (
            get_logger, trace, debug, info, warning, error, fatal
        )

        config = LogConfig(min_level=LogLevel.TRACE, async_logging=False)
        system = LogSystem.get_instance()
        system._config = config  # Override config
        target = RingBufferTarget()
        system.add_target(target)

        trace("Trace")
        debug("Debug")
        info("Info")
        warning("Warning")
        error("Error")
        fatal("Fatal")

        assert target.count == 6
