"""Output targets for the logging system.

Provides various destinations for log messages including console, file,
network, and memory ring buffer.
"""

from __future__ import annotations

import json
import queue
import socket
import struct
import sys
import threading
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Iterator, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .log_system import LogMessage, LogLevel
    from .log_format import LogFormatter


class LogTarget(ABC):
    """Base class for log output targets."""
    __slots__ = ('_name', '_enabled')

    def __init__(self, name: str = ""):
        """Initialize the target.

        Args:
            name: Target identifier
        """
        self._name = name or self.__class__.__name__
        self._enabled = True

    @property
    def name(self) -> str:
        """Get target name."""
        return self._name

    @property
    def enabled(self) -> bool:
        """Check if target is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable target."""
        self._enabled = value

    @abstractmethod
    def write(
        self,
        message: 'LogMessage',
        formatter: Optional['LogFormatter'] = None
    ) -> None:
        """Write a log message.

        Args:
            message: Message to write
            formatter: Optional formatter to use
        """
        pass

    def close(self) -> None:
        """Close the target and release resources."""
        pass


class ConsoleTarget(LogTarget):
    """Output target that writes to console (stdout/stderr).

    Supports color output and level-based stream selection.
    """
    __slots__ = ('_use_stderr_for_errors', '_use_colors', '_lock')

    def __init__(
        self,
        name: str = "console",
        use_stderr_for_errors: bool = True,
        use_colors: bool = True
    ):
        """Initialize console target.

        Args:
            name: Target identifier
            use_stderr_for_errors: Write errors to stderr
            use_colors: Enable ANSI colors
        """
        super().__init__(name)
        self._use_stderr_for_errors = use_stderr_for_errors
        self._use_colors = use_colors
        self._lock = threading.Lock()

    def write(
        self,
        message: 'LogMessage',
        formatter: Optional['LogFormatter'] = None
    ) -> None:
        """Write message to console."""
        if not self._enabled:
            return

        from .log_system import LogLevel
        from .log_format import DefaultFormatter, ColorFormatter

        # Choose formatter
        if formatter:
            text = formatter.format(message)
        elif self._use_colors:
            text = ColorFormatter().format(message)
        else:
            text = DefaultFormatter().format(message)

        # Choose stream
        if self._use_stderr_for_errors and message.level >= LogLevel.ERROR:
            stream = sys.stderr
        else:
            stream = sys.stdout

        with self._lock:
            try:
                stream.write(text + "\n")
                stream.flush()
            except Exception:
                pass


class FileTarget(LogTarget):
    """Output target that writes to a file.

    Supports rotation, append mode, and binary format.
    """
    __slots__ = (
        '_path', '_mode', '_max_size', '_max_files',
        '_file', '_current_size', '_lock', '_binary'
    )

    def __init__(
        self,
        path: Path,
        name: str = "",
        mode: str = "a",
        max_size: int = 10 * 1024 * 1024,  # 10 MB
        max_files: int = 5,
        binary: bool = False
    ):
        """Initialize file target.

        Args:
            path: Log file path
            name: Target identifier
            mode: File mode ('a' for append, 'w' for overwrite)
            max_size: Maximum file size before rotation
            max_files: Maximum number of rotated files to keep
            binary: Write in binary mode
        """
        super().__init__(name or str(path))
        self._path = Path(path)
        self._mode = mode
        self._max_size = max_size
        self._max_files = max_files
        self._binary = binary
        self._file = None
        self._current_size = 0
        self._lock = threading.Lock()

        self._open_file()

    def _open_file(self) -> None:
        """Open the log file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)

        file_mode = self._mode + ('b' if self._binary else '')
        self._file = open(self._path, file_mode)

        if self._path.exists():
            self._current_size = self._path.stat().st_size

    def _rotate(self) -> None:
        """Rotate log files."""
        if self._file:
            self._file.close()

        # Rotate existing files
        for i in range(self._max_files - 1, 0, -1):
            old_path = self._path.with_suffix(f".{i}{self._path.suffix}")
            new_path = self._path.with_suffix(f".{i + 1}{self._path.suffix}")

            if old_path.exists():
                if new_path.exists():
                    new_path.unlink()
                old_path.rename(new_path)

        # Rename current file
        if self._path.exists():
            rotated = self._path.with_suffix(f".1{self._path.suffix}")
            if rotated.exists():
                rotated.unlink()
            self._path.rename(rotated)

        # Remove excess files
        for i in range(self._max_files + 1, self._max_files + 10):
            excess = self._path.with_suffix(f".{i}{self._path.suffix}")
            if excess.exists():
                excess.unlink()

        # Open new file
        self._open_file()

    def write(
        self,
        message: 'LogMessage',
        formatter: Optional['LogFormatter'] = None
    ) -> None:
        """Write message to file."""
        if not self._enabled or not self._file:
            return

        from .log_format import DefaultFormatter

        if formatter:
            text = formatter.format(message)
        else:
            text = DefaultFormatter().format(message)

        with self._lock:
            try:
                if self._binary:
                    data = text.encode('utf-8') + b'\n'
                    self._file.write(data)
                    self._current_size += len(data)
                else:
                    line = text + "\n"
                    self._file.write(line)
                    self._current_size += len(line)

                self._file.flush()

                # Check for rotation
                if self._current_size >= self._max_size:
                    self._rotate()

            except Exception:
                pass

    def close(self) -> None:
        """Close the file."""
        with self._lock:
            if self._file:
                try:
                    self._file.close()
                except Exception:
                    pass
                self._file = None


class NetworkTarget(LogTarget):
    """Output target that sends logs over network.

    Supports UDP and TCP protocols with optional JSON formatting.
    """
    __slots__ = (
        '_host', '_port', '_protocol', '_socket',
        '_lock', '_use_json', '_connected'
    )

    def __init__(
        self,
        host: str,
        port: int,
        name: str = "",
        protocol: str = "udp",
        use_json: bool = True
    ):
        """Initialize network target.

        Args:
            host: Remote host
            port: Remote port
            name: Target identifier
            protocol: "udp" or "tcp"
            use_json: Send as JSON instead of formatted text
        """
        super().__init__(name or f"{host}:{port}")
        self._host = host
        self._port = port
        self._protocol = protocol.lower()
        self._use_json = use_json
        self._socket = None
        self._lock = threading.Lock()
        self._connected = False

        self._connect()

    def _connect(self) -> None:
        """Establish network connection."""
        try:
            if self._protocol == "udp":
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self._connected = True
            elif self._protocol == "tcp":
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._socket.connect((self._host, self._port))
                self._connected = True
        except socket.error:
            self._connected = False

    def write(
        self,
        message: 'LogMessage',
        formatter: Optional['LogFormatter'] = None
    ) -> None:
        """Send message over network."""
        if not self._enabled or not self._connected:
            return

        with self._lock:
            try:
                if self._use_json:
                    data = json.dumps({
                        "level": message.level.name,
                        "category": message.category.name,
                        "message": message.message,
                        "timestamp": message.timestamp.isoformat(),
                        "thread_id": message.thread_id,
                        "file": message.file,
                        "line": message.line,
                        "function": message.function,
                        "context": message.context
                    })
                else:
                    from .log_format import DefaultFormatter
                    if formatter:
                        data = formatter.format(message)
                    else:
                        data = DefaultFormatter().format(message)

                encoded = data.encode('utf-8')

                if self._protocol == "udp":
                    self._socket.sendto(encoded, (self._host, self._port))
                else:
                    # TCP: send length prefix + data
                    length = struct.pack('>I', len(encoded))
                    self._socket.sendall(length + encoded)

            except socket.error:
                self._connected = False

    def close(self) -> None:
        """Close network connection."""
        with self._lock:
            if self._socket:
                try:
                    self._socket.close()
                except Exception:
                    pass
                self._socket = None
            self._connected = False


@dataclass(slots=True)
class RingBufferEntry:
    """Entry in the ring buffer."""
    message: 'LogMessage'
    formatted: str
    index: int


class RingBufferTarget(LogTarget):
    """Lock-free ring buffer for in-memory log storage.

    Provides fast, thread-safe storage with fixed capacity.
    Useful for capturing recent logs for crash reports.
    """
    __slots__ = ('_buffer', '_capacity', '_lock', '_index_counter')

    def __init__(self, capacity: int = 1000, name: str = "ringbuffer"):
        """Initialize ring buffer target.

        Args:
            capacity: Maximum entries to store
            name: Target identifier
        """
        super().__init__(name)
        self._capacity = capacity
        self._buffer: Deque[RingBufferEntry] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._index_counter = 0

    @property
    def capacity(self) -> int:
        """Get buffer capacity."""
        return self._capacity

    @property
    def count(self) -> int:
        """Get current entry count."""
        return len(self._buffer)

    def write(
        self,
        message: 'LogMessage',
        formatter: Optional['LogFormatter'] = None
    ) -> None:
        """Store message in ring buffer."""
        if not self._enabled:
            return

        from .log_format import DefaultFormatter

        if formatter:
            formatted = formatter.format(message)
        else:
            formatted = DefaultFormatter().format(message)

        with self._lock:
            entry = RingBufferEntry(
                message=message,
                formatted=formatted,
                index=self._index_counter
            )
            self._buffer.append(entry)
            self._index_counter += 1

    def get_entries(
        self,
        count: Optional[int] = None,
        level: Optional['LogLevel'] = None
    ) -> list[RingBufferEntry]:
        """Get entries from buffer.

        Args:
            count: Maximum entries (None = all)
            level: Filter by minimum level

        Returns:
            List of entries (oldest first)
        """
        with self._lock:
            entries = list(self._buffer)

        if level is not None:
            entries = [e for e in entries if e.message.level >= level]

        if count is not None:
            entries = entries[-count:]

        return entries

    def get_messages(self, count: Optional[int] = None) -> list['LogMessage']:
        """Get log messages from buffer.

        Args:
            count: Maximum messages

        Returns:
            List of messages
        """
        entries = self.get_entries(count)
        return [e.message for e in entries]

    def get_formatted(self, count: Optional[int] = None) -> list[str]:
        """Get formatted strings from buffer.

        Args:
            count: Maximum entries

        Returns:
            List of formatted strings
        """
        entries = self.get_entries(count)
        return [e.formatted for e in entries]

    def search(self, pattern: str) -> list[RingBufferEntry]:
        """Search buffer for matching entries.

        Args:
            pattern: Search pattern (case-insensitive substring)

        Returns:
            Matching entries
        """
        pattern_lower = pattern.lower()
        with self._lock:
            return [
                e for e in self._buffer
                if pattern_lower in e.message.message.lower()
            ]

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()

    def __iter__(self) -> Iterator[RingBufferEntry]:
        """Iterate over entries."""
        with self._lock:
            return iter(list(self._buffer))


class CompositeTarget(LogTarget):
    """Target that writes to multiple child targets.

    Useful for sending logs to multiple destinations.
    """
    __slots__ = ('_targets',)

    def __init__(self, targets: list[LogTarget] = None, name: str = "composite"):
        """Initialize composite target.

        Args:
            targets: Child targets
            name: Target identifier
        """
        super().__init__(name)
        self._targets = list(targets) if targets else []

    @property
    def targets(self) -> list[LogTarget]:
        """Get child targets."""
        return self._targets

    def add_target(self, target: LogTarget) -> None:
        """Add a child target.

        Args:
            target: Target to add
        """
        if target not in self._targets:
            self._targets.append(target)

    def remove_target(self, target: LogTarget) -> None:
        """Remove a child target.

        Args:
            target: Target to remove
        """
        try:
            self._targets.remove(target)
        except ValueError:
            pass

    def write(
        self,
        message: 'LogMessage',
        formatter: Optional['LogFormatter'] = None
    ) -> None:
        """Write to all child targets."""
        if not self._enabled:
            return

        for target in self._targets:
            try:
                target.write(message, formatter)
            except Exception:
                pass

    def close(self) -> None:
        """Close all child targets."""
        for target in self._targets:
            try:
                target.close()
            except Exception:
                pass
