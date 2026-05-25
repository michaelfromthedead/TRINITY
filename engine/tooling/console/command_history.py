"""Command history management with search and persistence.

Provides command history tracking, search functionality, and cross-session
persistence for the developer console.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional


@dataclass
class HistoryEntry:
    """A single command history entry."""
    command: str
    timestamp: datetime
    success: bool = True
    result_summary: str = ""
    session_id: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "command": self.command,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "result_summary": self.result_summary,
            "session_id": self.session_id
        }

    @classmethod
    def from_dict(cls, data: dict) -> HistoryEntry:
        """Create from dictionary.

        Args:
            data: Dictionary data

        Returns:
            HistoryEntry instance
        """
        return cls(
            command=data["command"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            success=data.get("success", True),
            result_summary=data.get("result_summary", ""),
            session_id=data.get("session_id")
        )


class CommandHistory:
    """Manages command history with search and persistence.

    Provides:
    - Circular buffer for memory efficiency
    - Search functionality (prefix, substring, regex)
    - Cross-session persistence
    - Duplicate filtering
    """
    __slots__ = (
        '_entries', '_capacity', '_lock', '_file_path',
        '_current_index', '_session_id', '_ignore_duplicates'
    )

    DEFAULT_CAPACITY = 1000

    def __init__(
        self,
        capacity: int = DEFAULT_CAPACITY,
        file_path: Optional[Path] = None,
        session_id: Optional[str] = None,
        ignore_duplicates: bool = True
    ):
        """Initialize command history.

        Args:
            capacity: Maximum entries to store
            file_path: Path for persistence
            session_id: Current session identifier
            ignore_duplicates: Skip consecutive duplicate commands
        """
        self._entries: list[HistoryEntry] = []
        self._capacity = max(1, capacity)
        self._lock = threading.RLock()
        self._file_path = file_path
        self._current_index = -1  # For navigation
        self._session_id = session_id
        self._ignore_duplicates = ignore_duplicates

    @property
    def capacity(self) -> int:
        """Get the maximum capacity."""
        return self._capacity

    @property
    def count(self) -> int:
        """Get current number of entries."""
        with self._lock:
            return len(self._entries)

    @property
    def session_id(self) -> Optional[str]:
        """Get the current session ID."""
        return self._session_id

    @session_id.setter
    def session_id(self, value: Optional[str]) -> None:
        """Set the current session ID."""
        self._session_id = value

    def add(
        self,
        command: str,
        success: bool = True,
        result_summary: str = ""
    ) -> HistoryEntry:
        """Add a command to history.

        Args:
            command: The command string
            success: Whether command succeeded
            result_summary: Brief result description

        Returns:
            The created history entry
        """
        command = command.strip()
        if not command:
            raise ValueError("Command cannot be empty")

        entry = HistoryEntry(
            command=command,
            timestamp=datetime.now(),
            success=success,
            result_summary=result_summary,
            session_id=self._session_id
        )

        with self._lock:
            # Check for duplicate
            if self._ignore_duplicates and self._entries:
                if self._entries[-1].command == command:
                    # Update existing entry instead
                    self._entries[-1] = entry
                    self._reset_navigation()
                    return entry

            self._entries.append(entry)

            # Enforce capacity
            if len(self._entries) > self._capacity:
                self._entries.pop(0)

            self._reset_navigation()

        return entry

    def get(self, index: int) -> Optional[HistoryEntry]:
        """Get entry by index.

        Args:
            index: Index (negative for reverse indexing)

        Returns:
            Entry if found, None otherwise
        """
        with self._lock:
            try:
                return self._entries[index]
            except IndexError:
                return None

    def get_recent(self, count: int = 10) -> list[HistoryEntry]:
        """Get most recent entries.

        Args:
            count: Number of entries to return

        Returns:
            List of recent entries (newest last)
        """
        with self._lock:
            return self._entries[-count:]

    def previous(self) -> Optional[str]:
        """Navigate to previous command.

        Returns:
            Previous command, or None if at start
        """
        with self._lock:
            if not self._entries:
                return None

            if self._current_index == -1:
                self._current_index = len(self._entries) - 1
            elif self._current_index > 0:
                self._current_index -= 1

            return self._entries[self._current_index].command

    def next(self) -> Optional[str]:
        """Navigate to next command.

        Returns:
            Next command, or None if at end
        """
        with self._lock:
            if not self._entries or self._current_index == -1:
                return None

            if self._current_index < len(self._entries) - 1:
                self._current_index += 1
                return self._entries[self._current_index].command

            # At end, reset navigation
            self._current_index = -1
            return None

    def _reset_navigation(self) -> None:
        """Reset navigation index."""
        self._current_index = -1

    def search(
        self,
        query: str,
        max_results: int = 20,
        match_type: str = "prefix"
    ) -> list[HistoryEntry]:
        """Search command history.

        Args:
            query: Search query
            max_results: Maximum results to return
            match_type: "prefix", "substring", or "regex"

        Returns:
            Matching entries (newest first)
        """
        if not query:
            return self.get_recent(max_results)

        with self._lock:
            matches = []
            query_lower = query.lower()

            if match_type == "regex":
                import re
                try:
                    pattern = re.compile(query, re.IGNORECASE)
                except re.error:
                    return []

                for entry in reversed(self._entries):
                    if pattern.search(entry.command):
                        matches.append(entry)
                        if len(matches) >= max_results:
                            break

            elif match_type == "prefix":
                for entry in reversed(self._entries):
                    if entry.command.lower().startswith(query_lower):
                        matches.append(entry)
                        if len(matches) >= max_results:
                            break

            else:  # substring
                for entry in reversed(self._entries):
                    if query_lower in entry.command.lower():
                        matches.append(entry)
                        if len(matches) >= max_results:
                            break

            return matches

    def reverse_search(self, query: str) -> Iterator[HistoryEntry]:
        """Incrementally search backwards through history.

        Args:
            query: Search query (substring match)

        Yields:
            Matching entries from newest to oldest
        """
        query_lower = query.lower()

        with self._lock:
            for entry in reversed(self._entries):
                if query_lower in entry.command.lower():
                    yield entry

    def get_unique_commands(self, max_count: int = 100) -> list[str]:
        """Get unique commands from history.

        Args:
            max_count: Maximum commands to return

        Returns:
            List of unique command strings (most recent first)
        """
        seen = set()
        unique = []

        with self._lock:
            for entry in reversed(self._entries):
                if entry.command not in seen:
                    seen.add(entry.command)
                    unique.append(entry.command)
                    if len(unique) >= max_count:
                        break

        return unique

    def get_by_session(self, session_id: str) -> list[HistoryEntry]:
        """Get entries for a specific session.

        Args:
            session_id: Session identifier

        Returns:
            Entries from that session
        """
        with self._lock:
            return [e for e in self._entries if e.session_id == session_id]

    def get_failed_commands(self, max_count: int = 20) -> list[HistoryEntry]:
        """Get recently failed commands.

        Args:
            max_count: Maximum entries

        Returns:
            Failed command entries
        """
        with self._lock:
            failed = []
            for entry in reversed(self._entries):
                if not entry.success:
                    failed.append(entry)
                    if len(failed) >= max_count:
                        break
            return failed

    def clear(self) -> None:
        """Clear all history."""
        with self._lock:
            self._entries.clear()
            self._reset_navigation()

    def clear_session(self, session_id: str) -> int:
        """Clear history for a specific session.

        Args:
            session_id: Session identifier

        Returns:
            Number of entries removed
        """
        with self._lock:
            original_count = len(self._entries)
            self._entries = [
                e for e in self._entries
                if e.session_id != session_id
            ]
            self._reset_navigation()
            return original_count - len(self._entries)

    def save(self, path: Optional[Path] = None) -> None:
        """Save history to file.

        Args:
            path: Override file path
        """
        path = path or self._file_path
        if path is None:
            return

        with self._lock:
            data = {
                "version": 1,
                "entries": [e.to_dict() for e in self._entries]
            }

            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)

    def load(self, path: Optional[Path] = None) -> int:
        """Load history from file.

        Args:
            path: Override file path

        Returns:
            Number of entries loaded
        """
        path = path or self._file_path
        if path is None or not path.exists():
            return 0

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            entries = [
                HistoryEntry.from_dict(e)
                for e in data.get("entries", [])
            ]

            with self._lock:
                self._entries = entries[-self._capacity:]
                self._reset_navigation()

            return len(self._entries)

        except (json.JSONDecodeError, KeyError, ValueError):
            return 0

    def merge(self, other: CommandHistory) -> int:
        """Merge entries from another history.

        Args:
            other: History to merge from

        Returns:
            Number of entries added
        """
        with self._lock:
            original_count = len(self._entries)

            # Get all entries sorted by timestamp
            all_entries = sorted(
                self._entries + other._entries,
                key=lambda e: e.timestamp
            )

            # Remove duplicates (same command and timestamp)
            seen = set()
            unique = []
            for entry in all_entries:
                key = (entry.command, entry.timestamp)
                if key not in seen:
                    seen.add(key)
                    unique.append(entry)

            # Apply capacity limit
            self._entries = unique[-self._capacity:]
            self._reset_navigation()

            return len(self._entries) - original_count

    def __len__(self) -> int:
        """Get number of entries."""
        return self.count

    def __iter__(self) -> Iterator[HistoryEntry]:
        """Iterate over entries (oldest first)."""
        with self._lock:
            return iter(self._entries.copy())

    def __contains__(self, command: str) -> bool:
        """Check if command exists in history."""
        with self._lock:
            return any(e.command == command for e in self._entries)
