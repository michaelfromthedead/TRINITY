"""
File system abstraction with Result pattern, async I/O, and memory-mapped files.
"""
import asyncio
import logging
import mmap
import os
import pathlib
from dataclasses import dataclass
from enum import Enum, Flag, auto
from typing import Generic, TypeVar, Optional, Union, Callable

from ..constants import DEFAULT_FD_START

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass(slots=True)
class Result(Generic[T]):
    """Result type representing success or error."""
    value: Optional[T] = None
    error: Optional[str] = None

    @property
    def is_ok(self) -> bool:
        return self.error is None

    @property
    def is_err(self) -> bool:
        return self.error is not None

    def unwrap(self) -> T:
        if self.is_err:
            raise ValueError(f"Called unwrap on error: {self.error}")
        return self.value

    def unwrap_or(self, default: T) -> T:
        return self.value if self.is_ok else default

    @classmethod
    def ok(cls, value: T) -> 'Result[T]':
        return cls(value=value)

    @classmethod
    def err(cls, error: str) -> 'Result[T]':
        return cls(error=error)


class FileMode(Enum):
    """File open modes."""
    READ = 'r'
    WRITE = 'w'
    APPEND = 'a'
    READ_WRITE = 'r+'
    READ_BINARY = 'rb'
    WRITE_BINARY = 'wb'
    READ_WRITE_BINARY = 'rb+'


class MapMode(Enum):
    """Memory map modes."""
    READ = mmap.ACCESS_READ
    WRITE = mmap.ACCESS_WRITE
    COPY = mmap.ACCESS_COPY


class MappedFile:
    """RAII wrapper for memory-mapped file — closes both mmap and file handle."""
    __slots__ = ('_mmap', '_file')

    def __init__(self, mm: mmap.mmap, file_obj):
        self._mmap = mm
        self._file = file_obj

    def read(self, n: int = -1) -> bytes:
        return self._mmap.read(n)

    def write(self, data: bytes):
        self._mmap.write(data)

    def seek(self, pos: int, whence: int = 0):
        self._mmap.seek(pos, whence)

    def close(self):
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        if self._file:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()

    def __len__(self):
        return len(self._mmap) if self._mmap else 0

    def __getitem__(self, key):
        return self._mmap[key]

    def __setitem__(self, key, value):
        self._mmap[key] = value


@dataclass(slots=True)
class FileHandle:
    """Handle to an open file."""
    fd: int
    path: str
    mode: FileMode
    file_obj: object = None


class FileSystem:
    """Platform-independent file system operations."""

    def __init__(self):
        self._handles: dict[int, FileHandle] = {}
        self._next_fd = DEFAULT_FD_START
        self._watchers: list[tuple[str, Callable]] = []

    def safe_validate_path(self, base: str, target: str) -> Result[str]:
        """
        Validate path to prevent directory traversal attacks.

        Args:
            base: Base directory that should contain the target
            target: Target path to validate

        Returns:
            Result with normalized absolute path or error
        """
        try:
            base_path = pathlib.Path(base).resolve()
            target_path = pathlib.Path(target).resolve()

            # Check if target is within base directory
            try:
                target_path.relative_to(base_path)
            except ValueError:
                return Result.err(f"Path traversal detected: {target} escapes {base}")

            return Result.ok(str(target_path))
        except Exception as e:
            return Result.err(f"Path validation failed: {e}")

    def normalize_path(self, path: str) -> str:
        """Normalize path to use platform separators."""
        return os.path.normpath(path)

    def exists(self, path: str) -> bool:
        """Check if file or directory exists."""
        return os.path.exists(path)

    def file_size(self, path: str) -> Result[int]:
        """Get file size in bytes."""
        try:
            size = os.path.getsize(path)
            return Result.ok(size)
        except Exception as e:
            return Result.err(f"Failed to get file size: {e}")

    def open(self, path: str, mode: FileMode) -> Result[FileHandle]:
        """
        Open a file and return a handle.

        Args:
            path: Path to file
            mode: Open mode

        Returns:
            Result with FileHandle or error
        """
        try:
            file_obj = open(path, mode.value)
            fd = self._next_fd
            self._next_fd += 1

            handle = FileHandle(
                fd=fd,
                path=path,
                mode=mode,
                file_obj=file_obj
            )
            self._handles[fd] = handle

            return Result.ok(handle)
        except Exception as e:
            return Result.err(f"Failed to open file: {e}")

    def read_sync(self, handle: FileHandle, size: int = -1) -> Result[Union[str, bytes]]:
        """
        Synchronously read from file.

        Args:
            handle: File handle
            size: Number of bytes/chars to read (-1 for all)

        Returns:
            Result with data or error
        """
        try:
            if handle.fd not in self._handles:
                return Result.err("Invalid file handle")

            data = handle.file_obj.read(size)
            return Result.ok(data)
        except Exception as e:
            return Result.err(f"Read failed: {e}")

    def write_sync(self, handle: FileHandle, data: Union[str, bytes]) -> Result[int]:
        """
        Synchronously write to file.

        Args:
            handle: File handle
            data: Data to write

        Returns:
            Result with bytes written or error
        """
        try:
            if handle.fd not in self._handles:
                return Result.err("Invalid file handle")

            written = handle.file_obj.write(data)
            handle.file_obj.flush()
            return Result.ok(written)
        except Exception as e:
            return Result.err(f"Write failed: {e}")

    def close(self, handle: FileHandle) -> Result[None]:
        """
        Close file handle.

        Args:
            handle: File handle to close

        Returns:
            Result with None or error
        """
        try:
            if handle.fd not in self._handles:
                return Result.err("Invalid file handle")

            handle.file_obj.close()
            del self._handles[handle.fd]
            return Result.ok(None)
        except Exception as e:
            return Result.err(f"Close failed: {e}")

    async def read_async(self, path: str, mode: FileMode = FileMode.READ_BINARY) -> Result[Union[str, bytes]]:
        """
        Asynchronously read entire file.

        Args:
            path: Path to file
            mode: Open mode

        Returns:
            Result with file contents or error
        """
        try:
            loop = asyncio.get_event_loop()

            def _read():
                with open(path, mode.value) as f:
                    return f.read()

            data = await loop.run_in_executor(None, _read)
            return Result.ok(data)
        except Exception as e:
            return Result.err(f"Async read failed: {e}")

    async def write_async(self, path: str, data: Union[str, bytes], mode: FileMode = FileMode.WRITE_BINARY) -> Result[int]:
        """
        Asynchronously write to file.

        Args:
            path: Path to file
            data: Data to write
            mode: Open mode

        Returns:
            Result with bytes written or error
        """
        try:
            loop = asyncio.get_event_loop()

            def _write():
                with open(path, mode.value) as f:
                    return f.write(data)

            written = await loop.run_in_executor(None, _write)
            return Result.ok(written)
        except Exception as e:
            return Result.err(f"Async write failed: {e}")

    def mmap_read(self, path: str, offset: int = 0, size: Optional[int] = None) -> Result[MappedFile]:
        """
        Memory-map file for reading.

        Args:
            path: Path to file
            offset: Offset in file
            size: Number of bytes to map (None for entire file)

        Returns:
            Result with MappedFile object or error
        """
        try:
            file_obj = open(path, 'rb')
            if size is None:
                size = 0  # Map entire file

            mm = mmap.mmap(file_obj.fileno(), size, access=mmap.ACCESS_READ, offset=offset)
            mapped_file = MappedFile(mm, file_obj)
            return Result.ok(mapped_file)
        except Exception as e:
            return Result.err(f"Memory map read failed: {e}")

    def mmap_write(self, path: str, size: int, offset: int = 0) -> Result[MappedFile]:
        """
        Memory-map file for writing.

        Args:
            path: Path to file
            size: Size of mapping
            offset: Offset in file

        Returns:
            Result with MappedFile object or error
        """
        try:
            file_obj = open(path, 'r+b')
            mm = mmap.mmap(file_obj.fileno(), size, access=mmap.ACCESS_WRITE, offset=offset)
            mapped_file = MappedFile(mm, file_obj)
            return Result.ok(mapped_file)
        except Exception as e:
            return Result.err(f"Memory map write failed: {e}")

    def watch(self, path: str, callback: Callable[[str], None]) -> Result[None]:
        """
        Watch a file or directory for changes.

        Args:
            path: Path to watch
            callback: Function to call on change

        Returns:
            Result with None or error
        """
        try:
            if not os.path.exists(path):
                return Result.err(f"Path does not exist: {path}")

            self._watchers.append((path, callback))
            return Result.ok(None)
        except Exception as e:
            return Result.err(f"Watch setup failed: {e}")

    def unwatch(self, path: str) -> Result[None]:
        """
        Stop watching a path.

        Args:
            path: Path to stop watching

        Returns:
            Result with None or error
        """
        try:
            self._watchers = [(p, c) for p, c in self._watchers if p != path]
            return Result.ok(None)
        except Exception as e:
            return Result.err(f"Unwatch failed: {e}")
