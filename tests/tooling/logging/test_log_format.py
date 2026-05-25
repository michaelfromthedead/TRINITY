"""Tests for log formatters.

Tests all formatter types: default, compact, detailed, JSON, color.
"""

import pytest
import json
from datetime import datetime

from engine.tooling.logging.log_system import LogMessage, LogLevel, LogCategory
from engine.tooling.logging.log_format import (
    FormatConfig,
    LogFormatter,
    DefaultFormatter,
    CompactFormatter,
    DetailedFormatter,
    JsonFormatter,
    ColorFormatter,
    TemplateFormatter,
    SyslogFormatter,
)


@pytest.fixture
def sample_message():
    """Create a sample log message."""
    return LogMessage(
        level=LogLevel.INFO,
        category=LogCategory.ENGINE,
        message="Test message",
        file="test.py",
        line=42,
        function="test_func",
        context={"key": "value"}
    )


@pytest.fixture
def error_message():
    """Create an error message with exception."""
    try:
        raise ValueError("Test error")
    except ValueError as e:
        return LogMessage(
            level=LogLevel.ERROR,
            category=LogCategory.ENGINE,
            message="Error occurred",
            exception=e
        )


class TestFormatConfig:
    """Tests for FormatConfig."""

    def test_defaults(self):
        config = FormatConfig()
        assert config.include_timestamp is True
        assert config.include_level is True
        assert config.include_category is True
        assert config.include_thread is False
        assert config.include_source is False
        assert config.include_context is False

    def test_custom_config(self):
        config = FormatConfig(
            include_thread=True,
            include_source=True,
            max_message_width=80
        )
        assert config.include_thread is True
        assert config.include_source is True
        assert config.max_message_width == 80


class TestDefaultFormatter:
    """Tests for DefaultFormatter."""

    def test_basic_format(self, sample_message):
        formatter = DefaultFormatter()
        result = formatter.format(sample_message)

        assert "Test message" in result
        assert "INFO" in result
        assert "ENGINE" in result

    def test_includes_timestamp(self, sample_message):
        config = FormatConfig(include_timestamp=True)
        formatter = DefaultFormatter(config)
        result = formatter.format(sample_message)

        # Timestamp format: YYYY-MM-DD HH:MM:SS
        assert "-" in result  # Date separator

    def test_excludes_timestamp(self, sample_message):
        config = FormatConfig(include_timestamp=False)
        formatter = DefaultFormatter(config)
        result = formatter.format(sample_message)

        # Still has content
        assert "Test message" in result

    def test_includes_thread(self, sample_message):
        config = FormatConfig(include_thread=True)
        formatter = DefaultFormatter(config)
        result = formatter.format(sample_message)

        assert sample_message.thread_name in result

    def test_includes_source(self, sample_message):
        config = FormatConfig(include_source=True)
        formatter = DefaultFormatter(config)
        result = formatter.format(sample_message)

        assert "test.py" in result
        assert "42" in result

    def test_includes_context(self, sample_message):
        config = FormatConfig(include_context=True)
        formatter = DefaultFormatter(config)
        result = formatter.format(sample_message)

        assert "key=value" in result

    def test_exception_format(self, error_message):
        formatter = DefaultFormatter()
        result = formatter.format(error_message)

        assert "ValueError" in result
        assert "Test error" in result

    def test_max_message_width(self, sample_message):
        config = FormatConfig(max_message_width=30)
        formatter = DefaultFormatter(config)
        sample_message.message = "This is a very long message that should be truncated"

        result = formatter.format(sample_message)
        assert len(result) <= 33  # 30 + "..."


class TestCompactFormatter:
    """Tests for CompactFormatter."""

    def test_basic_format(self, sample_message):
        formatter = CompactFormatter()
        result = formatter.format(sample_message)

        assert "Test message" in result
        assert "INF" in result  # Short level
        assert "ENGI" in result  # First 4 chars of category

    def test_timestamp_format(self, sample_message):
        formatter = CompactFormatter()
        result = formatter.format(sample_message)

        # Should have HH:MM:SS.mmm format
        parts = result.split()
        assert ":" in parts[0]  # Time separator

    def test_max_width(self, sample_message):
        config = FormatConfig(max_message_width=40)
        formatter = CompactFormatter(config)
        sample_message.message = "A" * 100

        result = formatter.format(sample_message)
        assert len(result) <= 43


class TestDetailedFormatter:
    """Tests for DetailedFormatter."""

    def test_basic_format(self, sample_message):
        formatter = DetailedFormatter()
        result = formatter.format(sample_message)

        assert "Timestamp:" in result
        assert "Level:" in result
        assert "Category:" in result
        assert "Thread:" in result
        assert "Message:" in result

    def test_includes_source(self, sample_message):
        formatter = DetailedFormatter()
        result = formatter.format(sample_message)

        assert "Source:" in result
        assert "test.py" in result

    def test_includes_context(self, sample_message):
        formatter = DetailedFormatter()
        result = formatter.format(sample_message)

        assert "Context:" in result
        assert "key: value" in result

    def test_includes_exception(self, error_message):
        formatter = DetailedFormatter()
        result = formatter.format(error_message)

        assert "Exception:" in result
        assert "ValueError" in result

    def test_separator_lines(self, sample_message):
        formatter = DetailedFormatter()
        result = formatter.format(sample_message)

        assert "=" * 60 in result


class TestJsonFormatter:
    """Tests for JsonFormatter."""

    def test_basic_format(self, sample_message):
        formatter = JsonFormatter()
        result = formatter.format(sample_message)

        data = json.loads(result)
        assert data["level"] == "INFO"
        assert data["category"] == "ENGINE"
        assert data["message"] == "Test message"

    def test_includes_all_fields(self, sample_message):
        formatter = JsonFormatter(include_all=True)
        result = formatter.format(sample_message)

        data = json.loads(result)
        assert "thread_id" in data
        assert "thread_name" in data
        assert "file" in data
        assert "line" in data

    def test_excludes_optional_fields(self, sample_message):
        formatter = JsonFormatter(include_all=False)
        result = formatter.format(sample_message)

        data = json.loads(result)
        assert "thread_id" not in data

    def test_includes_context(self, sample_message):
        formatter = JsonFormatter()
        result = formatter.format(sample_message)

        data = json.loads(result)
        assert data["context"]["key"] == "value"

    def test_includes_exception(self, error_message):
        formatter = JsonFormatter()
        result = formatter.format(error_message)

        data = json.loads(result)
        assert "exception" in data
        assert data["exception"]["type"] == "ValueError"
        assert "traceback" in data["exception"]

    def test_pretty_format(self, sample_message):
        formatter = JsonFormatter(pretty=True)
        result = formatter.format(sample_message)

        # Pretty format should have newlines and indentation
        assert "\n" in result
        assert "  " in result

    def test_timestamp_iso_format(self, sample_message):
        formatter = JsonFormatter()
        result = formatter.format(sample_message)

        data = json.loads(result)
        # Should be parseable as ISO format
        datetime.fromisoformat(data["timestamp"])


class TestColorFormatter:
    """Tests for ColorFormatter."""

    def test_basic_format(self, sample_message):
        formatter = ColorFormatter()
        result = formatter.format(sample_message)

        assert "Test message" in result
        # Should contain ANSI codes
        assert "\033[" in result

    def test_level_colors(self, sample_message):
        formatter = ColorFormatter()

        for level in LogLevel:
            sample_message.level = level
            result = formatter.format(sample_message)
            assert "\033[" in result  # Has color codes

    def test_reset_code(self, sample_message):
        formatter = ColorFormatter()
        result = formatter.format(sample_message)

        assert "\033[0m" in result  # Reset code present

    def test_includes_context(self, sample_message):
        config = FormatConfig(include_context=True)
        formatter = ColorFormatter(config)
        result = formatter.format(sample_message)

        assert "key=value" in result


class TestTemplateFormatter:
    """Tests for TemplateFormatter."""

    def test_default_template(self, sample_message):
        formatter = TemplateFormatter()
        result = formatter.format(sample_message)

        assert "Test message" in result
        assert "INFO" in result
        assert "ENGINE" in result

    def test_custom_template(self, sample_message):
        formatter = TemplateFormatter(
            template="[{level_short}] {message}"
        )
        result = formatter.format(sample_message)

        assert result == "[INF] Test message"

    def test_all_placeholders(self, sample_message):
        formatter = TemplateFormatter(
            template="{level} {category} {thread_name} {file}:{line} {function} - {message}"
        )
        result = formatter.format(sample_message)

        assert "INFO" in result
        assert "ENGINE" in result
        assert "test.py" in result
        assert "42" in result
        assert "test_func" in result

    def test_elapsed_ms(self, sample_message):
        formatter = TemplateFormatter(
            template="[{elapsed_ms}ms] {message}"
        )
        result = formatter.format(sample_message)

        assert "ms]" in result
        assert "Test message" in result

    def test_exception_appended(self, error_message):
        formatter = TemplateFormatter(template="{message}")
        result = formatter.format(error_message)

        assert "ValueError" in result

    def test_template_property(self):
        formatter = TemplateFormatter()
        original = formatter.template

        formatter.template = "NEW: {message}"
        assert formatter.template == "NEW: {message}"


class TestSyslogFormatter:
    """Tests for SyslogFormatter."""

    def test_basic_format(self, sample_message):
        formatter = SyslogFormatter(app_name="test_app", hostname="testhost")
        result = formatter.format(sample_message)

        assert "test_app" in result
        assert "testhost" in result
        assert "ENGINE" in result
        assert "Test message" in result

    def test_priority_calculation(self, sample_message):
        formatter = SyslogFormatter()
        result = formatter.format(sample_message)

        # Should start with priority in angle brackets
        assert result.startswith("<")
        assert ">" in result

    def test_structured_data(self, sample_message):
        formatter = SyslogFormatter()
        result = formatter.format(sample_message)

        # Context should be in structured data
        assert '[context key="value"]' in result

    def test_no_context(self):
        msg = LogMessage(
            level=LogLevel.INFO,
            category=LogCategory.ENGINE,
            message="No context"
        )
        formatter = SyslogFormatter()
        result = formatter.format(msg)

        # Nil value for no structured data
        parts = result.split()
        # Should have "-" for empty SD
        assert "-" in parts

    def test_severity_mapping(self):
        formatter = SyslogFormatter()

        # Check different levels produce different priorities
        priorities = set()
        for level in LogLevel:
            msg = LogMessage(
                level=level,
                category=LogCategory.ENGINE,
                message="Test"
            )
            result = formatter.format(msg)
            # Extract priority
            priority = int(result[1:result.index(">")])
            priorities.add(priority)

        # Should have different priorities for different severities
        # (though DEBUG and TRACE map to same severity)
        assert len(priorities) >= 4
