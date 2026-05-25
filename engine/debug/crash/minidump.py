"""
Minidump generation for the game engine.

Provides platform-specific minidump/core dump generation with
configurable levels of detail.

Note: Full minidump generation requires platform-specific native code.
This module provides Python-level implementations and stubs for
native integration.
"""

import json
import logging
import os
import platform
import re
import sys
import threading
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional

# Module-level logger
_logger = logging.getLogger(__name__)


# Configuration constants
MAX_STACK_TRACE_LINES = 100
MAX_MODULES_TO_CAPTURE = 500
MAX_MEMORY_REGIONS = 1000
MAX_DUMP_FILE_SIZE_MB = 50
FINGERPRINT_STACK_LINES = 10


class MinidumpLevel(Enum):
    """
    Level of detail for minidump generation.

    MINI: Minimal dump - stack trace and basic thread info
    MEDIUM: Medium dump - adds loaded modules, memory regions
    FULL: Full dump - complete memory dump (large file size)
    """
    MINI = auto()
    MEDIUM = auto()
    FULL = auto()


@dataclass
class ThreadInfo:
    """
    Information about a single thread.

    Attributes:
        thread_id: Thread identifier
        name: Thread name
        is_daemon: Whether the thread is a daemon thread
        is_alive: Whether the thread is currently alive
        stack_trace: Stack trace for the thread
    """
    thread_id: int
    name: str
    is_daemon: bool
    is_alive: bool
    stack_trace: str = ""


@dataclass
class ModuleInfo:
    """
    Information about a loaded module.

    Attributes:
        name: Module name
        path: Module file path (if available)
        version: Module version (if available)
    """
    name: str
    path: Optional[str] = None
    version: Optional[str] = None


@dataclass
class MemoryRegion:
    """
    Information about a memory region.

    Attributes:
        start_address: Starting address of the region
        size: Size of the region in bytes
        protection: Memory protection flags
        type_name: Type of memory region
    """
    start_address: int
    size: int
    protection: str
    type_name: str


@dataclass
class MinidumpData:
    """
    Complete minidump data structure.

    Attributes:
        level: Dump level used
        timestamp: When the dump was created
        platform_info: Platform information
        python_info: Python interpreter information
        threads: List of thread information
        modules: List of loaded modules (MEDIUM+)
        memory_regions: List of memory regions (MEDIUM+)
        exception_info: Exception information if crash-triggered
        environment: Environment variables (sanitized)
        command_line: Command line arguments
    """
    level: MinidumpLevel
    timestamp: datetime = field(default_factory=datetime.now)
    platform_info: Dict[str, str] = field(default_factory=dict)
    python_info: Dict[str, str] = field(default_factory=dict)
    threads: List[ThreadInfo] = field(default_factory=list)
    modules: List[ModuleInfo] = field(default_factory=list)
    memory_regions: List[MemoryRegion] = field(default_factory=list)
    exception_info: Optional[Dict[str, Any]] = None
    environment: Dict[str, str] = field(default_factory=dict)
    command_line: List[str] = field(default_factory=list)


class Minidump:
    """
    Minidump generator for crash diagnostics.

    Generates platform-appropriate dump files with varying levels
    of detail for crash analysis.

    Usage:
        >>> dump = Minidump()
        >>> path = dump.generate(MinidumpLevel.MEDIUM, "/tmp/crash.dmp")
        >>> trace = dump.get_stack_trace()
    """

    # Environment variable patterns to exclude from dumps (security)
    # These patterns are matched as substrings (case-insensitive)
    EXCLUDED_ENV_PATTERNS = frozenset({
        'API_KEY', 'APIKEY', 'SECRET', 'PASSWORD', 'PASSWD', 'TOKEN',
        'CREDENTIAL', 'CRED', 'PRIVATE_KEY', 'PRIVATEKEY', 'AWS_SECRET',
        'AUTH', 'SESSION', 'COOKIE', 'JWT', 'BEARER', 'ACCESS_KEY',
        'ACCESSKEY', 'CLIENT_SECRET', 'CLIENT_ID', 'ENCRYPTION_KEY',
        'SIGNING_KEY', 'DATABASE_URL', 'DB_PASS', 'MONGO_URI', 'REDIS_URL',
        'SSH_KEY', 'GPG_KEY', 'PGP_KEY', 'CERT', 'CERTIFICATE',
    })

    # Exact environment variable names to always exclude
    EXCLUDED_ENV_EXACT = frozenset({
        'HOME', 'USER', 'LOGNAME', 'MAIL', 'HOSTNAME',
    })

    def __init__(self, exception: Optional[BaseException] = None):
        """
        Initialize the minidump generator.

        Args:
            exception: Optional exception that triggered the dump
        """
        self._exception = exception
        self._data: Optional[MinidumpData] = None

    def generate(self, level: MinidumpLevel, path: str) -> str:
        """
        Generate a minidump file.

        Args:
            level: Level of detail for the dump
            path: File path to write the dump to

        Returns:
            The path to the generated dump file

        Raises:
            OSError: If the file cannot be written
            PermissionError: If write permission is denied
            ValueError: If the path is invalid or contains traversal attempts
        """
        # Validate and sanitize path
        safe_path = self._validate_path(path)

        _logger.info(f"Generating {level.name} minidump to {safe_path}")

        try:
            # Collect dump data
            self._data = self._collect_data(level)

            # Ensure parent directory exists
            Path(safe_path).parent.mkdir(parents=True, exist_ok=True)

            # Write dump file
            self._write_dump(safe_path)

            _logger.info(f"Minidump written to {safe_path}")
            return safe_path

        except Exception as e:
            _logger.error(f"Failed to generate minidump: {e}")
            raise

    @staticmethod
    def _validate_path(path: str) -> str:
        """
        Validate and sanitize a file path.

        Args:
            path: The path to validate

        Returns:
            The validated absolute path

        Raises:
            ValueError: If the path is invalid or contains traversal attempts
        """
        if not path:
            raise ValueError("Path cannot be empty")

        # Convert to absolute path and resolve any .. or symlinks
        try:
            resolved = Path(path).resolve()
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Invalid path: {e}")

        # Check for null bytes (injection attempt)
        if '\x00' in str(resolved):
            raise ValueError("Path contains null bytes")

        # Ensure path doesn't escape to unexpected locations
        # Allow common crash dump directories
        allowed_prefixes = [
            '/tmp',
            '/var/log',
            '/var/crash',
            str(Path.home()),
        ]

        # On Windows, also allow AppData and temp
        if platform.system() == 'Windows':
            allowed_prefixes.extend([
                os.environ.get('LOCALAPPDATA', ''),
                os.environ.get('APPDATA', ''),
                os.environ.get('TEMP', ''),
            ])

        path_str = str(resolved)
        is_allowed = any(
            path_str.startswith(prefix) for prefix in allowed_prefixes if prefix
        )

        if not is_allowed:
            _logger.warning(f"Path {path_str} is outside allowed directories")
            # Still allow but log warning - strict mode could raise here

        return str(resolved)

    def get_stack_trace(self, thread_id: Optional[int] = None) -> str:
        """
        Get formatted stack trace.

        Args:
            thread_id: Optional specific thread ID. If None, returns
                      stack trace for all threads.

        Returns:
            Formatted stack trace string
        """
        if thread_id is not None:
            # Get stack trace for specific thread
            frame = sys._current_frames().get(thread_id)
            if frame:
                return "".join(traceback.format_stack(frame))
            return f"No stack trace available for thread {thread_id}"

        # Get stack trace for all threads
        traces = []
        for tid, frame in sys._current_frames().items():
            thread_name = self._get_thread_name(tid)
            traces.append(f"\n--- Thread {tid} ({thread_name}) ---\n")
            traces.append("".join(traceback.format_stack(frame)))

        return "".join(traces)

    def _collect_data(self, level: MinidumpLevel) -> MinidumpData:
        """
        Collect dump data at the specified level.

        Args:
            level: Level of detail to collect

        Returns:
            MinidumpData with collected information
        """
        data = MinidumpData(level=level)

        # Always collect basic info (MINI level)
        data.platform_info = self._collect_platform_info()
        data.python_info = self._collect_python_info()
        data.threads = self._collect_thread_info()
        data.command_line = sys.argv.copy()

        if self._exception:
            data.exception_info = self._collect_exception_info()

        # MEDIUM and FULL levels add more info
        if level in (MinidumpLevel.MEDIUM, MinidumpLevel.FULL):
            data.modules = self._collect_module_info()
            data.environment = self._collect_environment()
            data.memory_regions = self._collect_memory_regions()

        return data

    def _collect_platform_info(self) -> Dict[str, str]:
        """Collect platform information (excludes hostname for security)."""
        return {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            # Note: hostname intentionally excluded to prevent leaking internal infrastructure
        }

    def _collect_python_info(self) -> Dict[str, str]:
        """Collect Python interpreter information."""
        return {
            'version': platform.python_version(),
            'implementation': platform.python_implementation(),
            'compiler': platform.python_compiler(),
            'executable': sys.executable,
            'prefix': sys.prefix,
        }

    def _collect_thread_info(self) -> List[ThreadInfo]:
        """Collect information about all threads."""
        threads = []
        frames = sys._current_frames()

        for thread in threading.enumerate():
            tid = thread.ident or 0
            frame = frames.get(tid)
            stack_trace = "".join(traceback.format_stack(frame)) if frame else ""

            threads.append(ThreadInfo(
                thread_id=tid,
                name=thread.name,
                is_daemon=thread.daemon,
                is_alive=thread.is_alive(),
                stack_trace=stack_trace,
            ))

        return threads

    def _collect_module_info(self) -> List[ModuleInfo]:
        """Collect information about loaded modules (limited to prevent huge dumps)."""
        modules = []
        count = 0

        for name, module in sys.modules.items():
            if module is None:
                continue

            if count >= MAX_MODULES_TO_CAPTURE:
                _logger.debug(f"Module capture limit reached ({MAX_MODULES_TO_CAPTURE})")
                break

            path = getattr(module, '__file__', None)
            version = getattr(module, '__version__', None)

            modules.append(ModuleInfo(
                name=name,
                path=path,
                version=version,
            ))
            count += 1

        return modules

    def _collect_environment(self) -> Dict[str, str]:
        """
        Collect sanitized environment variables.

        Excludes variables that might contain secrets or sensitive information.
        """
        env = {}
        for key, value in os.environ.items():
            key_upper = key.upper()

            # Check exact matches first
            if key_upper in self.EXCLUDED_ENV_EXACT:
                env[key] = "[REDACTED]"
                continue

            # Check if key contains any sensitive pattern
            if any(pattern in key_upper for pattern in self.EXCLUDED_ENV_PATTERNS):
                env[key] = "[REDACTED]"
                continue

            # Redact values that look like secrets (e.g., long hex strings, base64)
            if self._looks_like_secret(value):
                env[key] = "[REDACTED]"
            else:
                env[key] = value
        return env

    @staticmethod
    def _looks_like_secret(value: str) -> bool:
        """
        Check if a value looks like a secret (heuristic).

        Args:
            value: The value to check

        Returns:
            True if the value appears to be a secret
        """
        if not value or len(value) < 20:
            return False

        # Check for long hex strings (API keys, tokens)
        if re.match(r'^[0-9a-fA-F]{32,}$', value):
            return True

        # Check for base64-encoded data (JWT tokens, etc.)
        if re.match(r'^[A-Za-z0-9+/=]{40,}$', value) and '=' in value[-3:]:
            return True

        return False

    def _collect_memory_regions(self) -> List[MemoryRegion]:
        """
        Collect memory region information.

        Note: This is a stub. Full implementation requires native code
        and is platform-specific.
        """
        regions = []

        # Try to read from /proc/self/maps on Linux
        if platform.system() == 'Linux':
            try:
                with open('/proc/self/maps', 'r') as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 5:
                            addr_range = parts[0].split('-')
                            if len(addr_range) == 2:
                                try:
                                    start = int(addr_range[0], 16)
                                    end = int(addr_range[1], 16)
                                    regions.append(MemoryRegion(
                                        start_address=start,
                                        size=end - start,
                                        protection=parts[1],
                                        type_name=parts[-1] if len(parts) > 5 else "anonymous",
                                    ))
                                except ValueError:
                                    continue
            except Exception as e:
                _logger.debug(f"Could not read memory maps: {e}")

        return regions

    def _collect_exception_info(self) -> Dict[str, Any]:
        """Collect exception information."""
        if not self._exception:
            return {}

        return {
            'type': type(self._exception).__name__,
            'message': str(self._exception),
            'traceback': "".join(traceback.format_exception(
                type(self._exception),
                self._exception,
                self._exception.__traceback__
            )),
        }

    def _get_thread_name(self, thread_id: int) -> str:
        """Get the name of a thread by its ID."""
        for thread in threading.enumerate():
            if thread.ident == thread_id:
                return thread.name
        return "unknown"

    def _write_dump(self, path: str) -> None:
        """
        Write dump data to a file.

        Uses JSON format for Python-level dumps.
        Native minidumps would use platform-specific binary formats.

        Args:
            path: Path to write the dump to
        """
        if not self._data:
            raise RuntimeError("No dump data collected")

        # Convert to serializable format
        dump_dict = self._to_serializable(self._data)

        # Write as JSON
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(dump_dict, f, indent=2, default=str)

    def _to_serializable(self, obj: Any) -> Any:
        """
        Convert an object to a JSON-serializable format.

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable representation
        """
        if isinstance(obj, MinidumpData):
            return {
                'level': obj.level.name,
                'timestamp': obj.timestamp.isoformat(),
                'platform_info': obj.platform_info,
                'python_info': obj.python_info,
                'threads': [self._to_serializable(t) for t in obj.threads],
                'modules': [self._to_serializable(m) for m in obj.modules],
                'memory_regions': [self._to_serializable(r) for r in obj.memory_regions],
                'exception_info': obj.exception_info,
                'environment': obj.environment,
                'command_line': obj.command_line,
            }
        elif isinstance(obj, (ThreadInfo, ModuleInfo, MemoryRegion)):
            return asdict(obj)
        elif isinstance(obj, Enum):
            return obj.name
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj

    @staticmethod
    def generate_native_minidump(path: str, level: MinidumpLevel) -> bool:
        """
        Generate a native minidump (platform-specific stub).

        On Windows, this would use MiniDumpWriteDump.
        On Linux/macOS, this would generate a core dump.

        Args:
            path: Path to write the minidump
            level: Level of detail

        Returns:
            True if successful, False otherwise
        """
        system = platform.system()

        if system == 'Windows':
            # Would call MiniDumpWriteDump via ctypes or native extension
            _logger.warning("Native minidump not implemented for Windows")
            return False

        elif system == 'Linux':
            # Would use ptrace or /proc/<pid>/coredump_filter
            _logger.warning("Native minidump not implemented for Linux")
            return False

        elif system == 'Darwin':
            # Would use MachExceptionHandler
            _logger.warning("Native minidump not implemented for macOS")
            return False

        else:
            _logger.warning(f"Native minidump not supported on {system}")
            return False


def generate_crash_dump(
    path: str,
    level: MinidumpLevel = MinidumpLevel.MEDIUM,
    exception: Optional[BaseException] = None
) -> str:
    """
    Convenience function to generate a crash dump.

    Args:
        path: Path to write the dump
        level: Level of detail (default: MEDIUM)
        exception: Optional exception that triggered the dump

    Returns:
        Path to the generated dump file
    """
    dump = Minidump(exception=exception)
    return dump.generate(level, path)


def get_current_stack_trace() -> str:
    """
    Get the current stack trace for all threads.

    Convenience function for quick diagnostics.

    Returns:
        Formatted stack trace string
    """
    dump = Minidump()
    return dump.get_stack_trace()


# Export public API
__all__ = [
    'MinidumpLevel',
    'ThreadInfo',
    'ModuleInfo',
    'MemoryRegion',
    'MinidumpData',
    'Minidump',
    'generate_crash_dump',
    'get_current_stack_trace',
]
