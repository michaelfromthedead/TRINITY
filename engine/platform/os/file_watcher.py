"""
File system watching using polling (os.stat based).
"""
import logging
import os
import time
import threading
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Optional, Dict
from pathlib import Path

from ..constants import DEFAULT_POLL_INTERVAL_SECONDS, THREAD_JOIN_TIMEOUT_MULTIPLIER

logger = logging.getLogger(__name__)


class FileEvent(Enum):
    """File system event types."""
    CREATED = auto()
    MODIFIED = auto()
    DELETED = auto()


@dataclass(slots=True)
class FileEventData:
    """File event information."""
    event: FileEvent
    path: str
    timestamp: float


@dataclass(slots=True)
class WatchedFile:
    """Information about a watched file."""
    path: str
    mtime: float
    size: int
    exists: bool


class FileWatcher:
    """
    File system watcher using polling.
    Monitors files/directories for changes and invokes callbacks.
    """

    def __init__(self, poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS):
        """
        Initialize file watcher.

        Args:
            poll_interval: Polling interval in seconds
        """
        self._poll_interval = poll_interval
        self._watched: Dict[str, WatchedFile] = {}
        self._callbacks: Dict[str, list[Callable[[FileEventData], None]]] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def watch_file(self, path: str, callback: Callable[[FileEventData], None]) -> bool:
        """
        Watch a single file for changes.

        Args:
            path: Path to file
            callback: Function to call on change

        Returns:
            True if watch was set up successfully
        """
        try:
            abs_path = os.path.abspath(path)

            with self._lock:
                # Initialize file state
                if os.path.exists(abs_path):
                    stat = os.stat(abs_path)
                    self._watched[abs_path] = WatchedFile(
                        path=abs_path,
                        mtime=stat.st_mtime,
                        size=stat.st_size,
                        exists=True
                    )
                else:
                    self._watched[abs_path] = WatchedFile(
                        path=abs_path,
                        mtime=0,
                        size=0,
                        exists=False
                    )

                # Add callback
                if abs_path not in self._callbacks:
                    self._callbacks[abs_path] = []
                self._callbacks[abs_path].append(callback)

            # Start polling thread if not running
            if not self._running:
                self.start()

            return True
        except Exception as e:
            logger.exception("Failed to watch file")
            return False

    def watch_directory(self, path: str, callback: Callable[[FileEventData], None],
                       recursive: bool = False) -> bool:
        """
        Watch a directory for changes.

        Args:
            path: Path to directory
            callback: Function to call on change
            recursive: Watch subdirectories recursively

        Returns:
            True if watch was set up successfully
        """
        try:
            abs_path = os.path.abspath(path)

            if not os.path.isdir(abs_path):
                return False

            # Watch all files in directory
            paths_to_watch = []

            if recursive:
                for root, _, files in os.walk(abs_path):
                    for filename in files:
                        file_path = os.path.join(root, filename)
                        paths_to_watch.append(file_path)
            else:
                for item in os.listdir(abs_path):
                    item_path = os.path.join(abs_path, item)
                    if os.path.isfile(item_path):
                        paths_to_watch.append(item_path)

            # Watch each file
            for file_path in paths_to_watch:
                self.watch_file(file_path, callback)

            return True
        except Exception as e:
            logger.exception("Failed to watch file")
            return False

    def unwatch(self, path: str) -> bool:
        """
        Stop watching a path.

        Args:
            path: Path to stop watching

        Returns:
            True if path was being watched
        """
        abs_path = os.path.abspath(path)

        with self._lock:
            if abs_path in self._watched:
                del self._watched[abs_path]
                if abs_path in self._callbacks:
                    del self._callbacks[abs_path]
                return True

        return False

    def start(self):
        """Start the file watcher polling thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the file watcher polling thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=self._poll_interval * THREAD_JOIN_TIMEOUT_MULTIPLIER)
            self._thread = None

    def _poll_loop(self):
        """Main polling loop."""
        while self._running:
            self._check_changes()
            time.sleep(self._poll_interval)

    def _check_changes(self):
        """Check for file changes and invoke callbacks."""
        with self._lock:
            paths_to_check = list(self._watched.keys())

        for path in paths_to_check:
            try:
                with self._lock:
                    if path not in self._watched:
                        continue

                    watched = self._watched[path]
                    callbacks = self._callbacks.get(path, [])

                current_exists = os.path.exists(path)

                # Check for deletion
                if watched.exists and not current_exists:
                    event = FileEventData(
                        event=FileEvent.DELETED,
                        path=path,
                        timestamp=time.time()
                    )
                    for callback in callbacks:
                        try:
                            callback(event)
                        except Exception as e:
                            logger.exception("File watcher callback error")

                    with self._lock:
                        self._watched[path].exists = False

                # Check for creation
                elif not watched.exists and current_exists:
                    stat = os.stat(path)
                    event = FileEventData(
                        event=FileEvent.CREATED,
                        path=path,
                        timestamp=time.time()
                    )
                    for callback in callbacks:
                        try:
                            callback(event)
                        except Exception as e:
                            logger.exception("File watcher callback error")

                    with self._lock:
                        self._watched[path].exists = True
                        self._watched[path].mtime = stat.st_mtime
                        self._watched[path].size = stat.st_size

                # Check for modification
                elif current_exists:
                    stat = os.stat(path)
                    if stat.st_mtime > watched.mtime or stat.st_size != watched.size:
                        event = FileEventData(
                            event=FileEvent.MODIFIED,
                            path=path,
                            timestamp=time.time()
                        )
                        for callback in callbacks:
                            try:
                                callback(event)
                            except Exception:
                                pass

                        with self._lock:
                            self._watched[path].mtime = stat.st_mtime
                            self._watched[path].size = stat.st_size

            except Exception as e:
                logger.exception("Error checking file changes")

    def is_watching(self, path: str) -> bool:
        """Check if a path is being watched."""
        abs_path = os.path.abspath(path)
        with self._lock:
            return abs_path in self._watched

    def watched_paths(self) -> list[str]:
        """Get list of all watched paths."""
        with self._lock:
            return list(self._watched.keys())
