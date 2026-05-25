"""Customizable log formatting with timestamps, categories, and colors.

Provides various formatters for different output needs.
"""

from __future__ import annotations

import json
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .log_system import LogMessage, LogLevel


@dataclass
class FormatConfig:
    """Configuration for log formatting."""
    timestamp_format: str = "%Y-%m-%d %H:%M:%S.%f"
    include_timestamp: bool = True
    include_level: bool = True
    include_category: bool = True
    include_thread: bool = False
    include_source: bool = False
    include_context: bool = False
    max_message_width: Optional[int] = None
    level_width: int = 7  # For alignment


class LogFormatter(ABC):
    """Base class for log formatters."""
    __slots__ = ('_config',)

    def __init__(self, config: Optional[FormatConfig] = None):
        """Initialize formatter.

        Args:
            config: Format configuration
        """
        self._config = config or FormatConfig()

    @property
    def config(self) -> FormatConfig:
        """Get format configuration."""
        return self._config

    @abstractmethod
    def format(self, message: 'LogMessage') -> str:
        """Format a log message.

        Args:
            message: Message to format

        Returns:
            Formatted string
        """
        pass

    def _format_timestamp(self, timestamp: datetime) -> str:
        """Format timestamp.

        Args:
            timestamp: Datetime to format

        Returns:
            Formatted timestamp string
        """
        return timestamp.strftime(self._config.timestamp_format)

    def _format_exception(self, exception: BaseException) -> str:
        """Format exception with traceback.

        Args:
            exception: Exception to format

        Returns:
            Formatted exception string
        """
        lines = traceback.format_exception(
            type(exception), exception, exception.__traceback__
        )
        return "".join(lines)


class DefaultFormatter(LogFormatter):
    """Default log formatter with configurable components."""
    __slots__ = ()

    def format(self, message: 'LogMessage') -> str:
        """Format message with default style."""
        parts = []

        if self._config.include_timestamp:
            parts.append(self._format_timestamp(message.timestamp))

        if self._config.include_level:
            level_str = message.level.name.ljust(self._config.level_width)
            parts.append(f"[{level_str}]")

        if self._config.include_category:
            parts.append(f"[{message.category.name}]")

        if self._config.include_thread:
            parts.append(f"[{message.thread_name}]")

        if self._config.include_source and message.file:
            source = message.format_source()
            parts.append(f"[{source}]")

        parts.append(message.message)

        result = " ".join(parts)

        if self._config.include_context and message.context:
            context_str = " ".join(f"{k}={v}" for k, v in message.context.items())
            result += f" | {context_str}"

        if message.exception:
            result += "\n" + self._format_exception(message.exception)

        if self._config.max_message_width and len(result) > self._config.max_message_width:
            result = result[:self._config.max_message_width - 3] + "..."

        return result


class CompactFormatter(LogFormatter):
    """Compact single-line formatter for high-volume logging."""
    __slots__ = ()

    def format(self, message: 'LogMessage') -> str:
        """Format message in compact style."""
        timestamp = message.timestamp.strftime("%H:%M:%S.%f")[:-3]
        level = message.level.name_short
        category = message.category.name[:4]

        text = f"{timestamp} {level} {category}: {message.message}"

        if self._config.max_message_width and len(text) > self._config.max_message_width:
            text = text[:self._config.max_message_width - 3] + "..."

        return text


class DetailedFormatter(LogFormatter):
    """Detailed multi-line formatter for debugging."""
    __slots__ = ()

    def format(self, message: 'LogMessage') -> str:
        """Format message with full details."""
        lines = [
            "=" * 60,
            f"Timestamp: {self._format_timestamp(message.timestamp)}",
            f"Level:     {message.level.name}",
            f"Category:  {message.category.name}",
            f"Thread:    {message.thread_name} ({message.thread_id})",
        ]

        if message.file:
            lines.append(f"Source:    {message.format_source()}")

        lines.append(f"Message:   {message.message}")

        if message.context:
            lines.append("Context:")
            for key, value in message.context.items():
                lines.append(f"  {key}: {value}")

        if message.exception:
            lines.append("Exception:")
            lines.append(self._format_exception(message.exception))

        lines.append("=" * 60)

        return "\n".join(lines)


class JsonFormatter(LogFormatter):
    """JSON formatter for structured logging."""
    __slots__ = ('_pretty', '_include_all')

    def __init__(
        self,
        config: Optional[FormatConfig] = None,
        pretty: bool = False,
        include_all: bool = True
    ):
        """Initialize JSON formatter.

        Args:
            config: Format configuration
            pretty: Pretty-print JSON
            include_all: Include all fields
        """
        super().__init__(config)
        self._pretty = pretty
        self._include_all = include_all

    def format(self, message: 'LogMessage') -> str:
        """Format message as JSON."""
        data = {
            "level": message.level.name,
            "category": message.category.name,
            "message": message.message,
            "timestamp": message.timestamp.isoformat(),
        }

        if self._include_all:
            data.update({
                "thread_id": message.thread_id,
                "thread_name": message.thread_name,
            })

            if message.file:
                data["file"] = message.file
            if message.line:
                data["line"] = message.line
            if message.function:
                data["function"] = message.function

        if message.context:
            data["context"] = message.context

        if message.exception:
            data["exception"] = {
                "type": type(message.exception).__name__,
                "message": str(message.exception),
                "traceback": self._format_exception(message.exception)
            }

        if self._pretty:
            return json.dumps(data, indent=2, default=str)
        else:
            return json.dumps(data, default=str)


class ColorFormatter(LogFormatter):
    """ANSI color formatter for terminal output."""
    __slots__ = ('_colors', '_reset')

    # ANSI color codes
    COLORS = {
        'TRACE': '\033[90m',  # Gray
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',  # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'FATAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    def __init__(self, config: Optional[FormatConfig] = None):
        """Initialize color formatter.

        Args:
            config: Format configuration
        """
        super().__init__(config)
        self._colors = self.COLORS
        self._reset = self.RESET

    def format(self, message: 'LogMessage') -> str:
        """Format message with colors."""
        color = self._colors.get(message.level.name, '')
        parts = []

        if self._config.include_timestamp:
            timestamp = self._format_timestamp(message.timestamp)
            parts.append(f"{self.DIM}{timestamp}{self._reset}")

        if self._config.include_level:
            level_str = message.level.name.ljust(self._config.level_width)
            parts.append(f"{color}{self.BOLD}[{level_str}]{self._reset}")

        if self._config.include_category:
            parts.append(f"{self.DIM}[{message.category.name}]{self._reset}")

        if self._config.include_thread:
            parts.append(f"{self.DIM}[{message.thread_name}]{self._reset}")

        if self._config.include_source and message.file:
            source = message.format_source()
            parts.append(f"{self.DIM}[{source}]{self._reset}")

        parts.append(f"{color}{message.message}{self._reset}")

        result = " ".join(parts)

        if self._config.include_context and message.context:
            context_str = " ".join(f"{k}={v}" for k, v in message.context.items())
            result += f" {self.DIM}| {context_str}{self._reset}"

        if message.exception:
            result += f"\n{self.COLORS['ERROR']}{self._format_exception(message.exception)}{self._reset}"

        return result


class TemplateFormatter(LogFormatter):
    """Template-based formatter for custom formats.

    Supports placeholders: {timestamp}, {level}, {category}, {message},
    {thread_id}, {thread_name}, {file}, {line}, {function}, {context}
    """
    __slots__ = ('_template',)

    DEFAULT_TEMPLATE = "{timestamp} [{level}] [{category}] {message}"

    def __init__(
        self,
        template: str = DEFAULT_TEMPLATE,
        config: Optional[FormatConfig] = None
    ):
        """Initialize template formatter.

        Args:
            template: Format template string
            config: Format configuration
        """
        super().__init__(config)
        self._template = template

    @property
    def template(self) -> str:
        """Get the template string."""
        return self._template

    @template.setter
    def template(self, value: str) -> None:
        """Set the template string."""
        self._template = value

    def format(self, message: 'LogMessage') -> str:
        """Format message using template."""
        context_str = ""
        if message.context:
            context_str = " ".join(f"{k}={v}" for k, v in message.context.items())

        result = self._template.format(
            timestamp=self._format_timestamp(message.timestamp),
            level=message.level.name,
            level_short=message.level.name_short,
            category=message.category.name,
            message=message.message,
            thread_id=message.thread_id,
            thread_name=message.thread_name,
            file=message.file or "",
            line=message.line or "",
            function=message.function or "",
            source=message.format_source(),
            context=context_str,
            elapsed_ms=f"{message.elapsed_ms:.2f}",
        )

        if message.exception:
            result += "\n" + self._format_exception(message.exception)

        return result


class SyslogFormatter(LogFormatter):
    """Syslog-compatible formatter (RFC 5424)."""
    __slots__ = ('_app_name', '_hostname')

    SEVERITY_MAP = {
        'TRACE': 7,  # Debug
        'DEBUG': 7,  # Debug
        'INFO': 6,  # Informational
        'WARNING': 4,  # Warning
        'ERROR': 3,  # Error
        'FATAL': 2,  # Critical
    }

    def __init__(
        self,
        app_name: str = "game_engine",
        hostname: str = "localhost",
        config: Optional[FormatConfig] = None
    ):
        """Initialize syslog formatter.

        Args:
            app_name: Application name
            hostname: Host name
            config: Format configuration
        """
        super().__init__(config)
        self._app_name = app_name
        self._hostname = hostname

    def format(self, message: 'LogMessage') -> str:
        """Format message in syslog format."""
        # Calculate priority (facility * 8 + severity)
        # Using LOCAL0 (16) as facility
        severity = self.SEVERITY_MAP.get(message.level.name, 6)
        priority = 16 * 8 + severity

        timestamp = message.timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
        if not timestamp.endswith(('+', '-')):
            timestamp += "Z"

        # Structured data
        sd = ""
        if message.context:
            params = " ".join(
                f'{k}="{v}"' for k, v in message.context.items()
            )
            sd = f'[context {params}]'
        else:
            sd = "-"

        return (
            f"<{priority}>1 {timestamp} {self._hostname} {self._app_name} "
            f"{message.thread_id} {message.category.name} {sd} {message.message}"
        )
