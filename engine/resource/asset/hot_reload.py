"""File watcher using mtime polling for hot reload."""
from __future__ import annotations

import os
import threading
import time
from typing import Callable

from engine.resource.constants import HOT_RELOAD_POLL_INTERVAL, THREAD_JOIN_TIMEOUT_MULTIPLIER

__all__ = ["HotReloadWatcher"]


class HotReloadWatcher:
    """Monitors files for changes using mtime polling."""
    __slots__ = ("_watches", "_running", "_thread", "_interval", "_lock")

    def __init__(self, interval: float = HOT_RELOAD_POLL_INTERVAL) -> None:
        self._watches: dict[str, tuple[float, Callable[[str], None]]] = {}
        self._running: bool = False
        self._thread: threading.Thread | None = None
        self._interval: float = interval
        self._lock: threading.Lock = threading.Lock()

    def register(self, path: str, callback: Callable[[str], None]) -> None:
        mtime = os.path.getmtime(path) if os.path.exists(path) else 0.0
        with self._lock:
            self._watches[path] = (mtime, callback)

    def unregister(self, path: str) -> None:
        with self._lock:
            self._watches.pop(path, None)

    def poll(self) -> list[str]:
        """Check all watched paths; fire callbacks for changed files. Returns changed paths."""
        changed: list[str] = []
        with self._lock:
            snapshot = list(self._watches.items())
        for path, (old_mtime, callback) in snapshot:
            try:
                current_mtime = os.path.getmtime(path)
            except OSError:
                continue
            if current_mtime != old_mtime:
                with self._lock:
                    if path in self._watches:
                        self._watches[path] = (current_mtime, callback)
                callback(path)
                changed.append(path)
        return changed

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self._interval * THREAD_JOIN_TIMEOUT_MULTIPLIER)
            self._thread = None

    def _poll_loop(self) -> None:
        while self._running:
            self.poll()
            time.sleep(self._interval)
