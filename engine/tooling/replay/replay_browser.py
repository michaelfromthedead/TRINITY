"""
Replay Browser - Browse, search, and filter replays.

Provides functionality to browse replay files, search by various
criteria, and filter results for easy replay management.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from .replay_file import ReplayFile, ReplayMetadata


class ReplaySortOrder(Enum):
    """Sort orders for replay browsing."""
    DATE_NEWEST = auto()
    DATE_OLDEST = auto()
    DURATION_LONGEST = auto()
    DURATION_SHORTEST = auto()
    NAME_ASC = auto()
    NAME_DESC = auto()
    SIZE_LARGEST = auto()
    SIZE_SMALLEST = auto()
    SCORE_HIGHEST = auto()
    SCORE_LOWEST = auto()


@dataclass
class ReplayFilter:
    """Filter criteria for replay browsing."""
    # Text search
    search_text: Optional[str] = None
    search_in_player: bool = True
    search_in_map: bool = True
    search_in_tags: bool = True

    # Date filters
    recorded_after: Optional[datetime] = None
    recorded_before: Optional[datetime] = None

    # Duration filters
    min_duration: Optional[float] = None
    max_duration: Optional[float] = None

    # Game filters
    game_name: Optional[str] = None
    game_version: Optional[str] = None
    map_name: Optional[str] = None
    game_mode: Optional[str] = None

    # Player filters
    player_name: Optional[str] = None
    player_id: Optional[str] = None

    # Result filters
    result: Optional[str] = None  # "win", "loss", "draw"
    min_score: Optional[int] = None
    max_score: Optional[int] = None

    # Tag filters
    required_tags: list[str] = field(default_factory=list)
    excluded_tags: list[str] = field(default_factory=list)

    # Size filters
    min_size: Optional[int] = None
    max_size: Optional[int] = None

    # Custom filter function
    custom_filter: Optional[Callable[['ReplayEntry'], bool]] = None

    def matches(self, entry: 'ReplayEntry') -> bool:
        """Check if a replay entry matches this filter.

        Args:
            entry: Replay entry to check

        Returns:
            True if entry matches all filter criteria
        """
        metadata = entry.metadata

        # Text search
        if self.search_text:
            search_lower = self.search_text.lower()
            found = False

            if self.search_in_player and metadata.player_name:
                if search_lower in metadata.player_name.lower():
                    found = True

            if self.search_in_map and metadata.map_name:
                if search_lower in metadata.map_name.lower():
                    found = True

            if self.search_in_tags:
                for tag in metadata.tags:
                    if search_lower in tag.lower():
                        found = True
                        break

            if not found:
                return False

        # Date filters
        if self.recorded_after and metadata.recorded_at < self.recorded_after:
            return False
        if self.recorded_before and metadata.recorded_at > self.recorded_before:
            return False

        # Duration filters
        if self.min_duration is not None and metadata.duration < self.min_duration:
            return False
        if self.max_duration is not None and metadata.duration > self.max_duration:
            return False

        # Game filters
        if self.game_name and metadata.game_name != self.game_name:
            return False
        if self.game_version and metadata.game_version != self.game_version:
            return False
        if self.map_name and metadata.map_name != self.map_name:
            return False
        if self.game_mode and metadata.game_mode != self.game_mode:
            return False

        # Player filters
        if self.player_name and metadata.player_name != self.player_name:
            return False
        if self.player_id and metadata.player_id != self.player_id:
            return False

        # Result filters
        if self.result and metadata.result != self.result:
            return False
        if self.min_score is not None and (metadata.score is None or metadata.score < self.min_score):
            return False
        if self.max_score is not None and (metadata.score is None or metadata.score > self.max_score):
            return False

        # Tag filters
        if self.required_tags:
            if not all(tag in metadata.tags for tag in self.required_tags):
                return False
        if self.excluded_tags:
            if any(tag in metadata.tags for tag in self.excluded_tags):
                return False

        # Size filters
        if self.min_size is not None and entry.file_size < self.min_size:
            return False
        if self.max_size is not None and entry.file_size > self.max_size:
            return False

        # Custom filter
        if self.custom_filter and not self.custom_filter(entry):
            return False

        return True


@dataclass
class ReplayEntry:
    """Entry representing a replay file."""
    path: Path
    metadata: ReplayMetadata
    file_size: int
    modified_time: datetime

    @property
    def filename(self) -> str:
        """Get filename without path."""
        return self.path.name

    @property
    def duration_formatted(self) -> str:
        """Get formatted duration string."""
        duration = self.metadata.duration
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)

        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    @property
    def size_formatted(self) -> str:
        """Get formatted file size string."""
        size = self.file_size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    @property
    def date_formatted(self) -> str:
        """Get formatted date string."""
        return self.metadata.recorded_at.strftime("%Y-%m-%d %H:%M")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'path': str(self.path),
            'filename': self.filename,
            'metadata': self.metadata.to_dict(),
            'file_size': self.file_size,
            'modified_time': self.modified_time.isoformat(),
            'duration_formatted': self.duration_formatted,
            'size_formatted': self.size_formatted,
            'date_formatted': self.date_formatted,
        }


@dataclass
class ReplaySearchResult:
    """Result of a replay search operation."""
    entries: list[ReplayEntry]
    total_count: int
    filter_used: Optional[ReplayFilter]
    sort_order: ReplaySortOrder
    search_time: float
    page: int = 1
    page_size: int = 50

    @property
    def total_pages(self) -> int:
        """Get total number of pages."""
        return (self.total_count + self.page_size - 1) // self.page_size

    @property
    def has_next_page(self) -> bool:
        """Check if there's a next page."""
        return self.page < self.total_pages

    @property
    def has_previous_page(self) -> bool:
        """Check if there's a previous page."""
        return self.page > 1


class ReplayBrowser:
    """Browse, search, and filter replay files.

    Provides functionality to scan replay directories, search by
    various criteria, and manage replay collections.
    """
    __slots__ = (
        '_root_paths', '_cache', '_cache_time', '_extensions',
        '_metadata_cache_enabled', '_on_scan_progress'
    )

    def __init__(
        self,
        root_paths: Optional[list[str | Path]] = None,
        extensions: Optional[list[str]] = None
    ):
        """Initialize replay browser.

        Args:
            root_paths: Root directories to search for replays
            extensions: File extensions to include (default: ['.replay', '.rpy'])
        """
        self._root_paths: list[Path] = []
        if root_paths:
            for path in root_paths:
                self._root_paths.append(Path(path))

        self._extensions = extensions or ['.replay', '.rpy', '.rep']
        self._cache: dict[Path, ReplayEntry] = {}
        self._cache_time: Optional[datetime] = None
        self._metadata_cache_enabled = True
        self._on_scan_progress: Optional[Callable[[int, int], None]] = None

    @property
    def root_paths(self) -> list[Path]:
        """Get root search paths."""
        return self._root_paths.copy()

    @property
    def cache_size(self) -> int:
        """Get number of cached entries."""
        return len(self._cache)

    def add_root_path(self, path: str | Path) -> None:
        """Add a root path to search.

        Args:
            path: Directory path
        """
        path = Path(path)
        if path not in self._root_paths:
            self._root_paths.append(path)
            self._cache_time = None  # Invalidate cache

    def remove_root_path(self, path: str | Path) -> bool:
        """Remove a root path.

        Args:
            path: Directory path to remove

        Returns:
            True if path was removed
        """
        path = Path(path)
        if path in self._root_paths:
            self._root_paths.remove(path)
            self._cache_time = None
            return True
        return False

    def set_progress_callback(
        self,
        callback: Optional[Callable[[int, int], None]]
    ) -> None:
        """Set callback for scan progress.

        Args:
            callback: Function taking (current, total) arguments
        """
        self._on_scan_progress = callback

    def scan(self, force_refresh: bool = False) -> int:
        """Scan root paths for replay files.

        Args:
            force_refresh: Force re-scan even if cache is valid

        Returns:
            Number of replay files found
        """
        if not force_refresh and self._cache_time is not None:
            return len(self._cache)

        self._cache.clear()
        files_found = []

        # Find all replay files
        for root_path in self._root_paths:
            if not root_path.exists():
                continue

            for ext in self._extensions:
                files_found.extend(root_path.rglob(f"*{ext}"))

        total = len(files_found)

        # Load metadata for each file
        for i, file_path in enumerate(files_found):
            try:
                entry = self._load_entry(file_path)
                if entry:
                    self._cache[file_path] = entry
            except Exception:
                # Skip files that can't be loaded
                pass

            if self._on_scan_progress:
                self._on_scan_progress(i + 1, total)

        self._cache_time = datetime.now()
        return len(self._cache)

    def search(
        self,
        filter: Optional[ReplayFilter] = None,
        sort: ReplaySortOrder = ReplaySortOrder.DATE_NEWEST,
        page: int = 1,
        page_size: int = 50
    ) -> ReplaySearchResult:
        """Search for replays matching criteria.

        Args:
            filter: Filter criteria
            sort: Sort order
            page: Page number (1-indexed)
            page_size: Results per page

        Returns:
            Search result with matching entries
        """
        import time
        start_time = time.perf_counter()

        # Ensure cache is populated
        if self._cache_time is None:
            self.scan()

        # Filter entries
        entries = list(self._cache.values())
        if filter:
            entries = [e for e in entries if filter.matches(e)]

        # Sort entries
        entries = self._sort_entries(entries, sort)

        total_count = len(entries)

        # Paginate
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        entries = entries[start_idx:end_idx]

        search_time = time.perf_counter() - start_time

        return ReplaySearchResult(
            entries=entries,
            total_count=total_count,
            filter_used=filter,
            sort_order=sort,
            search_time=search_time,
            page=page,
            page_size=page_size
        )

    def get_entry(self, path: str | Path) -> Optional[ReplayEntry]:
        """Get a specific replay entry.

        Args:
            path: Replay file path

        Returns:
            Replay entry, or None if not found
        """
        path = Path(path)
        if path in self._cache:
            return self._cache[path]
        return self._load_entry(path)

    def get_all_entries(self) -> list[ReplayEntry]:
        """Get all cached entries.

        Returns:
            List of all replay entries
        """
        if self._cache_time is None:
            self.scan()
        return list(self._cache.values())

    def iter_entries(self) -> Iterator[ReplayEntry]:
        """Iterate over all entries.

        Yields:
            Replay entries
        """
        if self._cache_time is None:
            self.scan()
        yield from self._cache.values()

    def get_unique_values(self, field: str) -> list[Any]:
        """Get unique values for a metadata field.

        Args:
            field: Metadata field name

        Returns:
            List of unique values
        """
        values = set()
        for entry in self._cache.values():
            value = getattr(entry.metadata, field, None)
            if value is not None:
                if isinstance(value, list):
                    values.update(value)
                else:
                    values.add(value)
        return sorted(values, key=str)

    def get_maps(self) -> list[str]:
        """Get list of unique map names."""
        return self.get_unique_values('map_name')

    def get_players(self) -> list[str]:
        """Get list of unique player names."""
        return self.get_unique_values('player_name')

    def get_game_modes(self) -> list[str]:
        """Get list of unique game modes."""
        return self.get_unique_values('game_mode')

    def get_tags(self) -> list[str]:
        """Get list of all tags used."""
        return self.get_unique_values('tags')

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the replay collection.

        Returns:
            Dictionary of statistics
        """
        if self._cache_time is None:
            self.scan()

        total_duration = sum(e.metadata.duration for e in self._cache.values())
        total_size = sum(e.file_size for e in self._cache.values())
        total_frames = sum(e.metadata.total_frames for e in self._cache.values())

        wins = sum(1 for e in self._cache.values() if e.metadata.result == 'win')
        losses = sum(1 for e in self._cache.values() if e.metadata.result == 'loss')
        draws = sum(1 for e in self._cache.values() if e.metadata.result == 'draw')

        durations = [e.metadata.duration for e in self._cache.values() if e.metadata.duration > 0]

        return {
            'total_replays': len(self._cache),
            'total_duration': total_duration,
            'total_size': total_size,
            'total_frames': total_frames,
            'average_duration': total_duration / len(self._cache) if self._cache else 0,
            'longest_duration': max(durations) if durations else 0,
            'shortest_duration': min(durations) if durations else 0,
            'unique_maps': len(self.get_maps()),
            'unique_players': len(self.get_players()),
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': wins / (wins + losses) if (wins + losses) > 0 else 0,
        }

    def find_by_date_range(
        self,
        start: datetime,
        end: datetime
    ) -> list[ReplayEntry]:
        """Find replays recorded within date range.

        Args:
            start: Start date
            end: End date

        Returns:
            List of matching entries
        """
        filter = ReplayFilter(recorded_after=start, recorded_before=end)
        return self.search(filter).entries

    def find_recent(self, days: int = 7) -> list[ReplayEntry]:
        """Find recently recorded replays.

        Args:
            days: Number of days to look back

        Returns:
            List of recent entries
        """
        cutoff = datetime.now() - timedelta(days=days)
        filter = ReplayFilter(recorded_after=cutoff)
        return self.search(filter, sort=ReplaySortOrder.DATE_NEWEST).entries

    def find_by_player(self, player: str) -> list[ReplayEntry]:
        """Find replays by player.

        Args:
            player: Player name or ID

        Returns:
            List of matching entries
        """
        # Search in both name and ID
        results = []
        for entry in self._cache.values():
            if (entry.metadata.player_name == player or
                    entry.metadata.player_id == player):
                results.append(entry)
        return results

    def delete_replay(self, path: str | Path) -> bool:
        """Delete a replay file.

        Args:
            path: Replay file path

        Returns:
            True if file was deleted
        """
        path = Path(path)
        try:
            if path.exists():
                path.unlink()
            self._cache.pop(path, None)
            return True
        except Exception:
            return False

    def clear_cache(self) -> None:
        """Clear the metadata cache."""
        self._cache.clear()
        self._cache_time = None

    def refresh_entry(self, path: str | Path) -> Optional[ReplayEntry]:
        """Refresh a single entry from disk.

        Args:
            path: Replay file path

        Returns:
            Updated entry, or None if load failed
        """
        path = Path(path)
        entry = self._load_entry(path)
        if entry:
            self._cache[path] = entry
        else:
            self._cache.pop(path, None)
        return entry

    def _load_entry(self, path: Path) -> Optional[ReplayEntry]:
        """Load a single replay entry."""
        if not path.exists():
            return None

        try:
            replay = ReplayFile(path)
            metadata = replay.load_metadata_only(path)

            stat = path.stat()

            return ReplayEntry(
                path=path,
                metadata=metadata,
                file_size=stat.st_size,
                modified_time=datetime.fromtimestamp(stat.st_mtime)
            )
        except Exception:
            return None

    def _sort_entries(
        self,
        entries: list[ReplayEntry],
        sort: ReplaySortOrder
    ) -> list[ReplayEntry]:
        """Sort entries by specified order."""
        if sort == ReplaySortOrder.DATE_NEWEST:
            return sorted(entries, key=lambda e: e.metadata.recorded_at, reverse=True)
        elif sort == ReplaySortOrder.DATE_OLDEST:
            return sorted(entries, key=lambda e: e.metadata.recorded_at)
        elif sort == ReplaySortOrder.DURATION_LONGEST:
            return sorted(entries, key=lambda e: e.metadata.duration, reverse=True)
        elif sort == ReplaySortOrder.DURATION_SHORTEST:
            return sorted(entries, key=lambda e: e.metadata.duration)
        elif sort == ReplaySortOrder.NAME_ASC:
            return sorted(entries, key=lambda e: e.filename.lower())
        elif sort == ReplaySortOrder.NAME_DESC:
            return sorted(entries, key=lambda e: e.filename.lower(), reverse=True)
        elif sort == ReplaySortOrder.SIZE_LARGEST:
            return sorted(entries, key=lambda e: e.file_size, reverse=True)
        elif sort == ReplaySortOrder.SIZE_SMALLEST:
            return sorted(entries, key=lambda e: e.file_size)
        elif sort == ReplaySortOrder.SCORE_HIGHEST:
            return sorted(entries, key=lambda e: e.metadata.score or 0, reverse=True)
        elif sort == ReplaySortOrder.SCORE_LOWEST:
            return sorted(entries, key=lambda e: e.metadata.score or 0)
        return entries
