"""
Log filtering system for the game engine.

Provides filter classes for controlling which log messages
are processed:
- LevelFilter: Filter by minimum severity level
- CategoryFilter: Include/exclude specific categories
- KeywordFilter: Filter by message content
- CompositeFilter: Combine multiple filters

Example:
    >>> from engine.debug.logging.filters import LevelFilter, CategoryFilter
    >>> from engine.debug.logging.logger import Logger, LogLevel, LogCategory
    >>>
    >>> logger = Logger("Game")
    >>> logger.add_filter(LevelFilter(LogLevel.WARNING))
    >>> logger.add_filter(CategoryFilter(exclude=[LogCategory.LogInput]))
"""

from __future__ import annotations

import re
import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Pattern, Set

if TYPE_CHECKING:
    from engine.debug.logging.logger import LogCategory, LogEntry, LogLevel


class LogFilter(ABC):
    """
    Abstract base class for log filters.

    Filters determine whether a log entry should be processed.
    They can be combined to create complex filtering logic.
    """

    @abstractmethod
    def should_log(self, entry: LogEntry) -> bool:
        """
        Determine if a log entry should be processed.

        Args:
            entry: The log entry to evaluate

        Returns:
            True if the entry should be logged, False otherwise
        """
        pass

    def __and__(self, other: LogFilter) -> CompositeFilter:
        """Combine filters with AND logic."""
        return CompositeFilter([self, other], mode="and")

    def __or__(self, other: LogFilter) -> CompositeFilter:
        """Combine filters with OR logic."""
        return CompositeFilter([self, other], mode="or")

    def __invert__(self) -> NegateFilter:
        """Invert the filter (NOT logic)."""
        return NegateFilter(self)


class LevelFilter(LogFilter):
    """
    Filter log entries by minimum severity level.

    Only entries at or above the specified level will pass.

    Attributes:
        min_level: Minimum log level to allow

    Example:
        >>> filter = LevelFilter(LogLevel.WARNING)
        >>> # Only WARNING, ERROR, and FATAL will pass
    """

    def __init__(self, min_level: LogLevel) -> None:
        """
        Initialize the level filter.

        Args:
            min_level: Minimum log level to allow
        """
        from engine.debug.logging.logger import LogLevel
        self.min_level = min_level

    def should_log(self, entry: LogEntry) -> bool:
        """Check if entry meets minimum level."""
        return entry.level >= self.min_level

    def __repr__(self) -> str:
        return f"LevelFilter(min_level={self.min_level.name})"


class CategoryFilter(LogFilter):
    """
    Filter log entries by category inclusion/exclusion.

    Can be configured to either include only specific categories
    or exclude specific categories from logging.

    Attributes:
        include: Set of categories to include (if set)
        exclude: Set of categories to exclude (if set)

    Example:
        >>> # Only log rendering and physics
        >>> filter = CategoryFilter(
        ...     include=[LogCategory.LogRendering, LogCategory.LogPhysics]
        ... )
        >>>
        >>> # Log everything except input
        >>> filter = CategoryFilter(
        ...     exclude=[LogCategory.LogInput]
        ... )
    """

    def __init__(
        self,
        include: list[LogCategory] | None = None,
        exclude: list[LogCategory] | None = None,
    ) -> None:
        """
        Initialize the category filter.

        Args:
            include: Categories to include (mutually exclusive with exclude)
            exclude: Categories to exclude (mutually exclusive with include)

        Raises:
            ValueError: If both include and exclude are specified
        """
        if include is not None and exclude is not None:
            raise ValueError("Cannot specify both include and exclude")

        self.include: Set[LogCategory] | None = set(include) if include else None
        self.exclude: Set[LogCategory] | None = set(exclude) if exclude else None

    def should_log(self, entry: LogEntry) -> bool:
        """Check if entry's category passes the filter."""
        if self.include is not None:
            return entry.category in self.include
        if self.exclude is not None:
            return entry.category not in self.exclude
        return True

    def add_include(self, category: LogCategory) -> None:
        """Add a category to the include set."""
        if self.exclude is not None:
            raise ValueError("Cannot add include when exclude is set")
        if self.include is None:
            self.include = set()
        self.include.add(category)

    def add_exclude(self, category: LogCategory) -> None:
        """Add a category to the exclude set."""
        if self.include is not None:
            raise ValueError("Cannot add exclude when include is set")
        if self.exclude is None:
            self.exclude = set()
        self.exclude.add(category)

    def __repr__(self) -> str:
        if self.include:
            cats = ", ".join(c.name for c in self.include)
            return f"CategoryFilter(include=[{cats}])"
        if self.exclude:
            cats = ", ".join(c.name for c in self.exclude)
            return f"CategoryFilter(exclude=[{cats}])"
        return "CategoryFilter()"


class KeywordFilter(LogFilter):
    """
    Filter log entries by message content.

    Can be configured with include/exclude patterns using
    string matching or regular expressions.

    Attributes:
        include_patterns: Patterns that must match for entry to pass
        exclude_patterns: Patterns that must NOT match for entry to pass

    Example:
        >>> # Only log messages containing "error" or "exception"
        >>> filter = KeywordFilter(include=["error", "exception"])
        >>>
        >>> # Exclude debug spam
        >>> filter = KeywordFilter(exclude=["heartbeat", "keepalive"])
        >>>
        >>> # Use regex patterns
        >>> filter = KeywordFilter(
        ...     include_regex=[r"player_\\d+"],
        ...     case_sensitive=False
        ... )
    """

    def __init__(
        self,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        include_regex: list[str] | None = None,
        exclude_regex: list[str] | None = None,
        case_sensitive: bool = False,
        search_fields: bool = False,
    ) -> None:
        """
        Initialize the keyword filter.

        Args:
            include: Keywords that must appear in message
            exclude: Keywords that must NOT appear in message
            include_regex: Regex patterns that must match
            exclude_regex: Regex patterns that must NOT match
            case_sensitive: Whether matching is case-sensitive
            search_fields: Whether to also search in entry fields
        """
        self.include_keywords = include or []
        self.exclude_keywords = exclude or []
        self.case_sensitive = case_sensitive
        self.search_fields = search_fields

        # Compile regex patterns
        flags = 0 if case_sensitive else re.IGNORECASE
        self.include_patterns: list[Pattern[str]] = [
            re.compile(p, flags) for p in (include_regex or [])
        ]
        self.exclude_patterns: list[Pattern[str]] = [
            re.compile(p, flags) for p in (exclude_regex or [])
        ]

        # Convert keywords for case-insensitive matching
        if not case_sensitive:
            self.include_keywords = [k.lower() for k in self.include_keywords]
            self.exclude_keywords = [k.lower() for k in self.exclude_keywords]

    def _get_searchable_text(self, entry: LogEntry) -> str:
        """Get the text to search in."""
        text = entry.message
        if self.search_fields and entry.fields:
            import json
            text = f"{text} {json.dumps(entry.fields)}"
        return text

    def should_log(self, entry: LogEntry) -> bool:
        """Check if entry's message passes the filter."""
        text = self._get_searchable_text(entry)
        search_text = text if self.case_sensitive else text.lower()

        # Check include keywords
        if self.include_keywords:
            if not any(kw in search_text for kw in self.include_keywords):
                return False

        # Check exclude keywords
        if self.exclude_keywords:
            if any(kw in search_text for kw in self.exclude_keywords):
                return False

        # Check include patterns
        if self.include_patterns:
            if not any(p.search(text) for p in self.include_patterns):
                return False

        # Check exclude patterns
        if self.exclude_patterns:
            if any(p.search(text) for p in self.exclude_patterns):
                return False

        return True

    def __repr__(self) -> str:
        parts = []
        if self.include_keywords:
            parts.append(f"include={self.include_keywords}")
        if self.exclude_keywords:
            parts.append(f"exclude={self.exclude_keywords}")
        if self.include_patterns:
            parts.append(f"include_regex={len(self.include_patterns)} patterns")
        if self.exclude_patterns:
            parts.append(f"exclude_regex={len(self.exclude_patterns)} patterns")
        return f"KeywordFilter({', '.join(parts)})"


class CompositeFilter(LogFilter):
    """
    Combine multiple filters with AND or OR logic.

    Attributes:
        filters: List of filters to combine
        mode: 'and' (all must pass) or 'or' (any must pass)

    Example:
        >>> # Both level and category must pass
        >>> filter = CompositeFilter([
        ...     LevelFilter(LogLevel.WARNING),
        ...     CategoryFilter(include=[LogCategory.LogNetwork]),
        ... ], mode="and")
        >>>
        >>> # Using operators
        >>> filter = LevelFilter(LogLevel.WARNING) & CategoryFilter(include=[...])
    """

    def __init__(
        self,
        filters: list[LogFilter],
        mode: str = "and",
    ) -> None:
        """
        Initialize the composite filter.

        Args:
            filters: List of filters to combine
            mode: 'and' or 'or'

        Raises:
            ValueError: If mode is not 'and' or 'or'
        """
        if mode not in ("and", "or"):
            raise ValueError(f"Invalid mode: {mode}")

        self.filters = filters
        self.mode = mode

    def should_log(self, entry: LogEntry) -> bool:
        """Evaluate all filters according to mode."""
        if not self.filters:
            return True

        if self.mode == "and":
            return all(f.should_log(entry) for f in self.filters)
        else:  # mode == "or"
            return any(f.should_log(entry) for f in self.filters)

    def add_filter(self, filter: LogFilter) -> None:
        """Add a filter to the composite."""
        self.filters.append(filter)

    def __repr__(self) -> str:
        return f"CompositeFilter(mode={self.mode}, count={len(self.filters)})"


class NegateFilter(LogFilter):
    """
    Invert the result of another filter.

    Attributes:
        filter: The filter to invert

    Example:
        >>> # Log everything EXCEPT errors
        >>> filter = ~LevelFilter(LogLevel.ERROR)
    """

    def __init__(self, filter: LogFilter) -> None:
        """
        Initialize the negate filter.

        Args:
            filter: The filter to invert
        """
        self.filter = filter

    def should_log(self, entry: LogEntry) -> bool:
        """Return the inverse of the wrapped filter."""
        return not self.filter.should_log(entry)

    def __repr__(self) -> str:
        return f"NegateFilter({self.filter!r})"


class RateLimitFilter(LogFilter):
    """
    Limit the rate of log messages.

    Useful for preventing log flooding from high-frequency events.

    Attributes:
        max_entries: Maximum entries per time window
        window_seconds: Time window in seconds

    Example:
        >>> # Max 100 messages per second
        >>> filter = RateLimitFilter(max_entries=100, window_seconds=1.0)
    """

    def __init__(
        self,
        max_entries: int,
        window_seconds: float = 1.0,
        per_category: bool = False,
        per_logger: bool = False,
    ) -> None:
        """
        Initialize the rate limit filter.

        Args:
            max_entries: Maximum entries per window
            window_seconds: Time window in seconds
            per_category: Apply limit per category
            per_logger: Apply limit per logger name
        """
        self.max_entries = max_entries
        self.window_seconds = window_seconds
        self.per_category = per_category
        self.per_logger = per_logger

        self._lock = threading.Lock()
        self._counts: dict[str, list[float]] = {}

    def _get_key(self, entry: LogEntry) -> str:
        """Get the rate limit key for an entry."""
        parts = []
        if self.per_category:
            parts.append(str(entry.category.name))
        if self.per_logger:
            parts.append(entry.logger_name)
        return ":".join(parts) if parts else "_global_"

    def should_log(self, entry: LogEntry) -> bool:
        """Check if entry is within rate limit."""
        import time

        now = time.time()
        cutoff = now - self.window_seconds
        key = self._get_key(entry)

        with self._lock:
            # Get or create timestamp list
            if key not in self._counts:
                self._counts[key] = []

            timestamps = self._counts[key]

            # Remove expired timestamps
            timestamps[:] = [t for t in timestamps if t > cutoff]

            # Check rate limit
            if len(timestamps) >= self.max_entries:
                return False

            # Record this entry
            timestamps.append(now)
            return True

    def __repr__(self) -> str:
        return (
            f"RateLimitFilter(max_entries={self.max_entries}, "
            f"window_seconds={self.window_seconds})"
        )


class SamplingFilter(LogFilter):
    """
    Sample log messages at a configurable rate.

    Useful for reducing log volume while maintaining visibility.

    Attributes:
        sample_rate: Fraction of messages to allow (0.0 to 1.0)

    Example:
        >>> # Log only 10% of messages
        >>> filter = SamplingFilter(sample_rate=0.1)
    """

    def __init__(
        self,
        sample_rate: float,
        seed: int | None = None,
    ) -> None:
        """
        Initialize the sampling filter.

        Args:
            sample_rate: Fraction of messages to allow (0.0 to 1.0)
            seed: Random seed for reproducibility

        Raises:
            ValueError: If sample_rate is not in [0.0, 1.0]
        """
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError(f"sample_rate must be in [0.0, 1.0], got {sample_rate}")

        self.sample_rate = sample_rate

        import random
        self._rng = random.Random(seed)
        self._lock = threading.Lock()

    def should_log(self, entry: LogEntry) -> bool:
        """Determine if entry passes sampling."""
        with self._lock:
            return self._rng.random() < self.sample_rate

    def __repr__(self) -> str:
        return f"SamplingFilter(sample_rate={self.sample_rate})"


class CallbackFilter(LogFilter):
    """
    Filter using a custom callback function.

    Provides maximum flexibility for custom filtering logic.

    Attributes:
        callback: Function that determines if entry should be logged

    Example:
        >>> # Only log entries from specific players
        >>> def player_filter(entry):
        ...     player_id = entry.fields.get("player_id")
        ...     return player_id in watched_players
        >>> filter = CallbackFilter(player_filter)
    """

    def __init__(self, callback: Callable[[LogEntry], bool]) -> None:
        """
        Initialize the callback filter.

        Args:
            callback: Function that takes LogEntry and returns bool
        """
        self.callback = callback

    def should_log(self, entry: LogEntry) -> bool:
        """Evaluate the callback for the entry."""
        return self.callback(entry)

    def __repr__(self) -> str:
        return f"CallbackFilter(callback={self.callback.__name__})"


class FieldFilter(LogFilter):
    """
    Filter based on structured log fields.

    Allows filtering on specific field values in structured logs.

    Example:
        >>> # Only log entries with specific field values
        >>> filter = FieldFilter("player_id", lambda v: v == 123)
        >>> filter = FieldFilter("latency_ms", lambda v: v > 100)
    """

    def __init__(
        self,
        field_name: str,
        predicate: Callable[[object], bool],
        require_field: bool = True,
    ) -> None:
        """
        Initialize the field filter.

        Args:
            field_name: Name of the field to check
            predicate: Function to evaluate the field value
            require_field: If True, entries without the field are rejected
        """
        self.field_name = field_name
        self.predicate = predicate
        self.require_field = require_field

    def should_log(self, entry: LogEntry) -> bool:
        """Check if field value passes predicate."""
        if self.field_name not in entry.fields:
            return not self.require_field

        value = entry.fields[self.field_name]
        return self.predicate(value)

    def __repr__(self) -> str:
        return f"FieldFilter(field_name={self.field_name!r})"
