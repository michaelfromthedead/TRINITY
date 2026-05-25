"""
Log output sinks for the game engine logging system.

Provides different destinations for log messages:
- ConsoleSink: Output to stdout with ANSI color support
- FileSink: Output to files with rotation support
- NetworkSink: Send logs to a remote endpoint

Example:
    >>> from engine.debug.logging.sinks import ConsoleSink, FileSink
    >>> from engine.debug.logging.logger import Logger
    >>>
    >>> logger = Logger("Game")
    >>> logger.add_sink(ConsoleSink())
    >>> logger.add_sink(FileSink("/var/log/game.log"))
"""

from __future__ import annotations

import gzip
import json
import queue
import shutil
import socket
import sys
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

from engine.core.constants import (
    LOG_FILE_MAX_SIZE,
    LOG_FILE_MAX_BACKUPS,
    LOG_FILE_ENCODING,
    LOG_NETWORK_TIMEOUT,
    LOG_NETWORK_BATCH_SIZE,
    LOG_NETWORK_FLUSH_INTERVAL,
    LOG_NETWORK_RECONNECT_DELAY,
    LOG_BUFFER_SIZE,
    LOG_BUFFER_FLUSH_INTERVAL,
)

if TYPE_CHECKING:
    from engine.debug.logging.logger import LogEntry, LogLevel


class LogSink(ABC):
    """
    Abstract base class for log output destinations.

    All sinks must implement the write() method to handle
    log entries in their specific way.
    """

    @abstractmethod
    def write(self, entry: LogEntry) -> None:
        """
        Write a log entry to the sink.

        Args:
            entry: The log entry to write
        """
        pass

    def flush(self) -> None:
        """
        Flush any buffered output.

        Override this if the sink buffers output.
        """
        pass

    def close(self) -> None:
        """
        Close the sink and release resources.

        Override this if the sink holds resources that need cleanup.
        """
        pass


class ConsoleSink(LogSink):
    """
    Log sink that outputs to the console with ANSI color support.

    Colors are automatically applied based on log level:
    - VERBOSE: Dim gray
    - DEBUG: Cyan
    - INFO: White
    - WARNING: Yellow
    - ERROR: Red
    - FATAL: Bright red with bold

    Attributes:
        use_colors: Whether to use ANSI color codes
        stream: Output stream (defaults to sys.stdout)

    Example:
        >>> sink = ConsoleSink(use_colors=True)
        >>> logger.add_sink(sink)
    """

    # ANSI color codes
    COLORS = {
        "VERBOSE": "\033[90m",      # Dim gray
        "DEBUG": "\033[36m",        # Cyan
        "INFO": "\033[37m",         # White
        "WARNING": "\033[33m",      # Yellow
        "ERROR": "\033[31m",        # Red
        "FATAL": "\033[1;91m",      # Bright red, bold
    }
    RESET = "\033[0m"

    def __init__(
        self,
        use_colors: bool = True,
        stream: TextIO | None = None,
        include_timestamp: bool = True,
        include_category: bool = True,
        timestamp_format: str = "%Y-%m-%d %H:%M:%S.%f",
    ) -> None:
        """
        Initialize the console sink.

        Args:
            use_colors: Whether to use ANSI color codes
            stream: Output stream (defaults to sys.stdout)
            include_timestamp: Whether to include timestamps
            include_category: Whether to include category names
            timestamp_format: Format string for timestamps
        """
        self.use_colors = use_colors and self._supports_color()
        self.stream = stream or sys.stdout
        self.include_timestamp = include_timestamp
        self.include_category = include_category
        self.timestamp_format = timestamp_format
        self._lock = threading.Lock()

    def _supports_color(self) -> bool:
        """Check if the terminal supports ANSI colors."""
        # Check if stdout is a TTY
        if not hasattr(sys.stdout, "isatty"):
            return False
        if not sys.stdout.isatty():
            return False
        # Check for Windows without ANSI support
        if sys.platform == "win32":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                # Enable ANSI escape sequences on Windows 10+
                kernel32.SetConsoleMode(
                    kernel32.GetStdHandle(-11), 7
                )
                return True
            except Exception:
                return False
        return True

    def _format_entry(self, entry: LogEntry) -> str:
        """Format a log entry for console output."""
        parts = []

        if self.include_timestamp:
            timestamp_str = entry.timestamp.strftime(self.timestamp_format)
            # Truncate microseconds to 3 digits
            if ".%f" in self.timestamp_format:
                timestamp_str = timestamp_str[:-3]
            parts.append(f"[{timestamp_str}]")

        level_str = f"[{entry.level.name:7s}]"
        parts.append(level_str)

        if self.include_category:
            parts.append(f"[{entry.category.name}]")

        parts.append(f"[{entry.logger_name}]")
        parts.append(entry.message)

        if entry.fields:
            fields_str = " ".join(f"{k}={v}" for k, v in entry.fields.items())
            parts.append(f"{{ {fields_str} }}")

        return " ".join(parts)

    def write(self, entry: LogEntry) -> None:
        """Write a log entry to the console."""
        with self._lock:
            formatted = self._format_entry(entry)

            if self.use_colors:
                color = self.COLORS.get(entry.level.name, "")
                line = f"{color}{formatted}{self.RESET}"
            else:
                line = formatted

            print(line, file=self.stream)

    def flush(self) -> None:
        """Flush the output stream."""
        with self._lock:
            self.stream.flush()


class FileSink(LogSink):
    """
    Log sink that writes to files with optional rotation.

    Supports rotation by size or age, with optional compression
    of rotated files.

    Attributes:
        path: Path to the log file
        max_size: Maximum file size before rotation (bytes)
        max_files: Maximum number of rotated files to keep
        compress_rotated: Whether to gzip rotated files

    Example:
        >>> sink = FileSink(
        ...     "/var/log/game.log",
        ...     max_size=10 * 1024 * 1024,  # 10 MB
        ...     max_files=5,
        ...     compress_rotated=True,
        ... )
    """

    def __init__(
        self,
        path: str | Path,
        max_size: int | None = None,
        max_files: int = LOG_FILE_MAX_BACKUPS,
        compress_rotated: bool = False,
        mode: str = "a",
        encoding: str = LOG_FILE_ENCODING,
        json_format: bool = False,
    ) -> None:
        """
        Initialize the file sink.

        Args:
            path: Path to the log file
            max_size: Maximum file size before rotation (None = no rotation)
            max_files: Maximum number of rotated files to keep
            compress_rotated: Whether to compress rotated files with gzip
            mode: File open mode ('a' for append, 'w' for overwrite)
            encoding: File encoding
            json_format: Whether to write entries as JSON lines
        """
        self.path = Path(path)
        self.max_size = max_size
        self.max_files = max_files
        self.compress_rotated = compress_rotated
        self.mode = mode
        self.encoding = encoding
        self.json_format = json_format

        self._lock = threading.Lock()
        self._file: TextIO | None = None
        self._current_size = 0

        # Ensure directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Open the file
        self._open_file()

    def _open_file(self) -> None:
        """Open the log file."""
        self._file = open(
            self.path,
            mode=self.mode,
            encoding=self.encoding,
        )
        # Get current file size - file exists after open() in append mode
        try:
            self._current_size = self.path.stat().st_size
        except OSError:
            self._current_size = 0

    def _format_entry(self, entry: LogEntry) -> str:
        """Format a log entry for file output."""
        if self.json_format:
            return entry.to_json()

        parts = [
            entry.timestamp.isoformat(),
            entry.level.name,
            entry.category.name,
            entry.logger_name,
            entry.message,
        ]

        line = " | ".join(parts)

        if entry.fields:
            fields_json = json.dumps(entry.fields, default=str)
            line = f"{line} | {fields_json}"

        return line

    def _should_rotate(self) -> bool:
        """Check if the file should be rotated."""
        if self.max_size is None:
            return False
        return self._current_size >= self.max_size

    def _rotate(self) -> None:
        """Rotate the log file."""
        if self._file:
            self._file.close()
            self._file = None

        # Build list of rotated file names
        rotated_files = []
        for i in range(self.max_files, 0, -1):
            suffix = f".{i}"
            if self.compress_rotated and i > 0:
                suffix += ".gz"
            rotated = self.path.with_suffix(self.path.suffix + suffix)
            rotated_files.append((i, rotated))

        # Delete oldest if it exists
        if self.max_files > 0:
            oldest_suffix = f".{self.max_files}"
            if self.compress_rotated:
                oldest_suffix += ".gz"
            oldest = self.path.with_suffix(self.path.suffix + oldest_suffix)
            if oldest.exists():
                oldest.unlink()

        # Rename existing rotated files (n -> n+1)
        for i, rotated in rotated_files[1:]:
            prev_suffix = f".{i-1}" if i > 1 else ""
            if self.compress_rotated and i > 1:
                prev_suffix += ".gz"
            prev = self.path.with_suffix(self.path.suffix + prev_suffix) if i > 1 else self.path

            if prev.exists() and prev != self.path:
                if rotated.exists():
                    rotated.unlink()
                prev.rename(rotated)

        # Rotate current file
        if self.path.exists():
            new_name = self.path.with_suffix(self.path.suffix + ".1")
            if self.compress_rotated:
                # Compress the file
                with open(self.path, "rb") as f_in:
                    with gzip.open(str(new_name) + ".gz", "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                self.path.unlink()
            else:
                self.path.rename(new_name)

        # Open new file
        self._open_file()

    def write(self, entry: LogEntry) -> None:
        """Write a log entry to the file."""
        with self._lock:
            if self._file is None:
                self._open_file()

            if self._should_rotate():
                self._rotate()

            line = self._format_entry(entry) + "\n"
            self._file.write(line)
            self._current_size += len(line.encode(self.encoding))

    def flush(self) -> None:
        """Flush the file buffer."""
        with self._lock:
            if self._file:
                self._file.flush()

    def close(self) -> None:
        """Close the file."""
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None


class NetworkSink(LogSink):
    """
    Log sink that sends logs to a remote endpoint.

    Uses TCP or UDP sockets to send JSON-formatted log entries
    to a remote log collector.

    Attributes:
        host: Remote host address
        port: Remote port number
        protocol: 'tcp' or 'udp'

    Example:
        >>> sink = NetworkSink("logs.example.com", 5514, protocol="tcp")
        >>> logger.add_sink(sink)
    """

    def __init__(
        self,
        host: str,
        port: int,
        protocol: str = "tcp",
        timeout: float = LOG_NETWORK_TIMEOUT,
        batch_size: int = LOG_NETWORK_BATCH_SIZE,
        flush_interval: float = LOG_NETWORK_FLUSH_INTERVAL,
        reconnect_delay: float = LOG_NETWORK_RECONNECT_DELAY,
    ) -> None:
        """
        Initialize the network sink.

        Args:
            host: Remote host address
            port: Remote port number
            protocol: Transport protocol ('tcp' or 'udp')
            timeout: Socket timeout in seconds
            batch_size: Number of entries to batch before sending
            flush_interval: Maximum time between flushes
            reconnect_delay: Delay before reconnection attempts
        """
        self.host = host
        self.port = port
        self.protocol = protocol.lower()
        self.timeout = timeout
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.reconnect_delay = reconnect_delay

        if self.protocol not in ("tcp", "udp"):
            raise ValueError(f"Invalid protocol: {protocol}")

        self._lock = threading.Lock()
        self._queue: queue.Queue[LogEntry] = queue.Queue()
        self._socket: socket.socket | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_flush = time.time()
        self._connected = False

        # Start background sender thread
        self._start_sender()

    def _start_sender(self) -> None:
        """Start the background sender thread."""
        self._running = True
        self._thread = threading.Thread(target=self._sender_loop, daemon=True)
        self._thread.start()

    def _connect(self) -> bool:
        """Establish connection to the remote endpoint."""
        try:
            if self.protocol == "tcp":
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._socket.settimeout(self.timeout)
                self._socket.connect((self.host, self.port))
            else:
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self._socket.settimeout(self.timeout)

            self._connected = True
            return True
        except Exception:
            self._connected = False
            return False

    def _disconnect(self) -> None:
        """Close the socket connection."""
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
        self._connected = False

    def _send_batch(self, entries: list[LogEntry]) -> bool:
        """Send a batch of entries to the remote endpoint."""
        if not entries:
            return True

        if not self._connected:
            if not self._connect():
                return False

        try:
            for entry in entries:
                data = entry.to_json().encode("utf-8") + b"\n"
                if self.protocol == "tcp":
                    self._socket.sendall(data)
                else:
                    self._socket.sendto(data, (self.host, self.port))
            return True
        except Exception:
            self._disconnect()
            return False

    def _sender_loop(self) -> None:
        """Background thread that sends batched log entries."""
        batch: list[LogEntry] = []

        while self._running:
            try:
                # Try to get an entry from the queue
                try:
                    entry = self._queue.get(timeout=0.1)
                    batch.append(entry)
                except queue.Empty:
                    pass

                # Check if we should flush
                should_flush = (
                    len(batch) >= self.batch_size or
                    (batch and time.time() - self._last_flush >= self.flush_interval)
                )

                if should_flush:
                    if not self._send_batch(batch):
                        # Failed to send, retry after delay
                        time.sleep(self.reconnect_delay)
                        continue
                    batch = []
                    self._last_flush = time.time()

            except Exception:
                time.sleep(self.reconnect_delay)

    def write(self, entry: LogEntry) -> None:
        """Queue a log entry for sending."""
        self._queue.put(entry)

    def flush(self) -> None:
        """Force flush of queued entries."""
        # Wait for queue to drain
        deadline = time.time() + self.timeout
        while not self._queue.empty() and time.time() < deadline:
            time.sleep(0.01)

    def close(self) -> None:
        """Stop the sender thread and close the connection."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=self.timeout)
        self._disconnect()


class BufferedSink(LogSink):
    """
    Wrapper sink that buffers log entries before writing.

    Useful for reducing I/O operations by batching writes.

    Attributes:
        sink: The underlying sink to write to
        buffer_size: Number of entries to buffer
        flush_interval: Maximum time between flushes
    """

    def __init__(
        self,
        sink: LogSink,
        buffer_size: int = LOG_BUFFER_SIZE,
        flush_interval: float = LOG_BUFFER_FLUSH_INTERVAL,
    ) -> None:
        """
        Initialize the buffered sink.

        Args:
            sink: The underlying sink to write to
            buffer_size: Number of entries to buffer before flushing
            flush_interval: Maximum seconds between flushes
        """
        self.sink = sink
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval

        self._buffer: list[LogEntry] = []
        self._lock = threading.Lock()
        self._last_flush = time.time()

        # Start flush timer thread
        self._running = True
        self._thread = threading.Thread(target=self._flush_timer, daemon=True)
        self._thread.start()

    def _flush_timer(self) -> None:
        """Background thread that flushes periodically."""
        while self._running:
            time.sleep(0.1)
            if time.time() - self._last_flush >= self.flush_interval:
                self.flush()

    def write(self, entry: LogEntry) -> None:
        """Buffer a log entry."""
        with self._lock:
            self._buffer.append(entry)
            if len(self._buffer) >= self.buffer_size:
                self._flush_internal()

    def _flush_internal(self) -> None:
        """Internal flush without acquiring lock."""
        for entry in self._buffer:
            self.sink.write(entry)
        self._buffer.clear()
        self.sink.flush()
        self._last_flush = time.time()

    def flush(self) -> None:
        """Flush buffered entries to the underlying sink."""
        with self._lock:
            self._flush_internal()

    def close(self) -> None:
        """Stop the flush timer and close the underlying sink."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self.flush()
        self.sink.close()


class MultiplexSink(LogSink):
    """
    Sink that writes to multiple underlying sinks.

    Useful for sending logs to multiple destinations with
    different configurations.
    """

    def __init__(self, sinks: list[LogSink] | None = None) -> None:
        """
        Initialize the multiplex sink.

        Args:
            sinks: List of sinks to write to
        """
        self._sinks = sinks or []
        self._lock = threading.Lock()

    def add_sink(self, sink: LogSink) -> None:
        """Add a sink to the multiplex."""
        with self._lock:
            if sink not in self._sinks:
                self._sinks.append(sink)

    def remove_sink(self, sink: LogSink) -> None:
        """Remove a sink from the multiplex."""
        with self._lock:
            if sink in self._sinks:
                self._sinks.remove(sink)

    def write(self, entry: LogEntry) -> None:
        """Write to all underlying sinks."""
        with self._lock:
            for sink in self._sinks:
                try:
                    sink.write(entry)
                except Exception:
                    pass  # Don't let one sink failure affect others

    def flush(self) -> None:
        """Flush all underlying sinks."""
        with self._lock:
            for sink in self._sinks:
                try:
                    sink.flush()
                except Exception:
                    pass

    def close(self) -> None:
        """Close all underlying sinks."""
        with self._lock:
            for sink in self._sinks:
                try:
                    sink.close()
                except Exception:
                    pass
