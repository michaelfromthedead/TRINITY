"""Log filtering with level, category, and pattern matching.

Provides flexible filtering to control which log messages are processed.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .log_system import LogMessage, LogLevel, LogCategory


class FilterAction(Enum):
    """Action to take after filtering."""
    PASS = auto()  # Allow message through
    DROP = auto()  # Block message
    MODIFY = auto()  # Message was modified


class LogFilter(ABC):
    """Base class for log filters."""
    __slots__ = ('_name', '_enabled')

    def __init__(self, name: str = ""):
        """Initialize the filter.

        Args:
            name: Filter identifier
        """
        self._name = name or self.__class__.__name__
        self._enabled = True

    @property
    def name(self) -> str:
        """Get filter name."""
        return self._name

    @property
    def enabled(self) -> bool:
        """Check if filter is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable filter."""
        self._enabled = value

    @abstractmethod
    def filter(self, message: 'LogMessage') -> FilterAction:
        """Filter a log message.

        Args:
            message: Message to filter

        Returns:
            Action to take
        """
        pass


class LevelFilter(LogFilter):
    """Filter by log level.

    Filters messages below the minimum level or above the maximum level.
    """
    __slots__ = ('_min_level', '_max_level')

    def __init__(
        self,
        min_level: Optional['LogLevel'] = None,
        max_level: Optional['LogLevel'] = None,
        name: str = ""
    ):
        """Initialize level filter.

        Args:
            min_level: Minimum level to pass (inclusive)
            max_level: Maximum level to pass (inclusive)
            name: Filter identifier
        """
        super().__init__(name or "LevelFilter")
        self._min_level = min_level
        self._max_level = max_level

    @property
    def min_level(self) -> Optional['LogLevel']:
        """Get minimum level."""
        return self._min_level

    @min_level.setter
    def min_level(self, value: Optional['LogLevel']) -> None:
        """Set minimum level."""
        self._min_level = value

    @property
    def max_level(self) -> Optional['LogLevel']:
        """Get maximum level."""
        return self._max_level

    @max_level.setter
    def max_level(self, value: Optional['LogLevel']) -> None:
        """Set maximum level."""
        self._max_level = value

    def filter(self, message: 'LogMessage') -> FilterAction:
        """Filter by level."""
        if not self._enabled:
            return FilterAction.PASS

        if self._min_level is not None and message.level < self._min_level:
            return FilterAction.DROP

        if self._max_level is not None and message.level > self._max_level:
            return FilterAction.DROP

        return FilterAction.PASS


class CategoryFilter(LogFilter):
    """Filter by log category.

    Can include or exclude specific categories.
    """
    __slots__ = ('_include', '_exclude', '_mode')

    def __init__(
        self,
        include: Optional[set['LogCategory']] = None,
        exclude: Optional[set['LogCategory']] = None,
        name: str = ""
    ):
        """Initialize category filter.

        Args:
            include: Categories to include (None = all)
            exclude: Categories to exclude
            name: Filter identifier
        """
        super().__init__(name or "CategoryFilter")
        self._include = set(include) if include else None
        self._exclude = set(exclude) if exclude else set()

    @property
    def include_categories(self) -> Optional[set['LogCategory']]:
        """Get included categories."""
        return self._include

    @property
    def exclude_categories(self) -> set['LogCategory']:
        """Get excluded categories."""
        return self._exclude

    def include(self, category: 'LogCategory') -> None:
        """Add category to include list.

        Args:
            category: Category to include
        """
        if self._include is None:
            self._include = set()
        self._include.add(category)
        self._exclude.discard(category)

    def exclude(self, category: 'LogCategory') -> None:
        """Add category to exclude list.

        Args:
            category: Category to exclude
        """
        self._exclude.add(category)
        if self._include:
            self._include.discard(category)

    def filter(self, message: 'LogMessage') -> FilterAction:
        """Filter by category."""
        if not self._enabled:
            return FilterAction.PASS

        if message.category in self._exclude:
            return FilterAction.DROP

        if self._include is not None and message.category not in self._include:
            return FilterAction.DROP

        return FilterAction.PASS


class PatternFilter(LogFilter):
    """Filter by regex pattern matching on message text.

    Can match against message, file, function, or all fields.
    """
    __slots__ = ('_pattern', '_compiled', '_match_field', '_invert')

    def __init__(
        self,
        pattern: str,
        match_field: str = "message",
        invert: bool = False,
        ignore_case: bool = True,
        name: str = ""
    ):
        """Initialize pattern filter.

        Args:
            pattern: Regex pattern
            match_field: Field to match ("message", "file", "function", "all")
            invert: If True, drop matching messages
            ignore_case: Case-insensitive matching
            name: Filter identifier
        """
        super().__init__(name or f"PatternFilter({pattern})")
        self._pattern = pattern
        self._match_field = match_field
        self._invert = invert

        flags = re.IGNORECASE if ignore_case else 0
        self._compiled = re.compile(pattern, flags)

    @property
    def pattern(self) -> str:
        """Get the pattern string."""
        return self._pattern

    @property
    def match_field(self) -> str:
        """Get the field being matched."""
        return self._match_field

    @property
    def invert(self) -> bool:
        """Check if filter is inverted."""
        return self._invert

    def filter(self, message: 'LogMessage') -> FilterAction:
        """Filter by pattern."""
        if not self._enabled:
            return FilterAction.PASS

        # Get text to match
        if self._match_field == "message":
            text = message.message
        elif self._match_field == "file":
            text = message.file or ""
        elif self._match_field == "function":
            text = message.function or ""
        elif self._match_field == "all":
            text = f"{message.message} {message.file or ''} {message.function or ''}"
        else:
            text = message.message

        matches = bool(self._compiled.search(text))

        if self._invert:
            return FilterAction.DROP if matches else FilterAction.PASS
        else:
            return FilterAction.PASS if matches else FilterAction.DROP


class RateLimitFilter(LogFilter):
    """Filter that rate-limits messages.

    Prevents log flooding by limiting messages per time window.
    """
    __slots__ = ('_max_count', '_window_seconds', '_counts', '_window_start')

    def __init__(
        self,
        max_count: int = 100,
        window_seconds: float = 1.0,
        name: str = ""
    ):
        """Initialize rate limit filter.

        Args:
            max_count: Maximum messages per window
            window_seconds: Time window in seconds
            name: Filter identifier
        """
        super().__init__(name or "RateLimitFilter")
        self._max_count = max_count
        self._window_seconds = window_seconds
        self._counts: dict[str, int] = {}  # Key -> count
        self._window_start: dict[str, float] = {}  # Key -> window start time

    @property
    def max_count(self) -> int:
        """Get maximum count."""
        return self._max_count

    @property
    def window_seconds(self) -> float:
        """Get window duration."""
        return self._window_seconds

    def filter(self, message: 'LogMessage') -> FilterAction:
        """Apply rate limiting."""
        if not self._enabled:
            return FilterAction.PASS

        import time

        # Use category + level as key
        key = f"{message.category.name}:{message.level.name}"
        now = time.time()

        # Reset window if expired
        if key in self._window_start:
            if now - self._window_start[key] > self._window_seconds:
                self._counts[key] = 0
                self._window_start[key] = now
        else:
            self._window_start[key] = now
            self._counts[key] = 0

        # Check limit
        self._counts[key] = self._counts.get(key, 0) + 1

        if self._counts[key] > self._max_count:
            return FilterAction.DROP

        return FilterAction.PASS


class SamplingFilter(LogFilter):
    """Filter that samples messages.

    Allows only a percentage of messages through.
    """
    __slots__ = ('_sample_rate', '_counter')

    def __init__(self, sample_rate: float = 0.1, name: str = ""):
        """Initialize sampling filter.

        Args:
            sample_rate: Fraction of messages to keep (0.0 to 1.0)
            name: Filter identifier
        """
        super().__init__(name or "SamplingFilter")
        self._sample_rate = max(0.0, min(1.0, sample_rate))
        self._counter = 0

    @property
    def sample_rate(self) -> float:
        """Get sample rate."""
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: float) -> None:
        """Set sample rate."""
        self._sample_rate = max(0.0, min(1.0, value))

    def filter(self, message: 'LogMessage') -> FilterAction:
        """Apply sampling."""
        if not self._enabled:
            return FilterAction.PASS

        if self._sample_rate >= 1.0:
            return FilterAction.PASS

        if self._sample_rate <= 0.0:
            return FilterAction.DROP

        # Use deterministic sampling based on counter
        self._counter += 1
        if (self._counter % int(1.0 / self._sample_rate)) == 0:
            return FilterAction.PASS

        return FilterAction.DROP


class DeduplicationFilter(LogFilter):
    """Filter that removes duplicate messages.

    Uses a sliding window to detect and suppress repeated messages.
    """
    __slots__ = ('_window_size', '_recent_messages')

    def __init__(self, window_size: int = 100, name: str = ""):
        """Initialize deduplication filter.

        Args:
            window_size: Number of recent messages to track
            name: Filter identifier
        """
        super().__init__(name or "DeduplicationFilter")
        self._window_size = window_size
        self._recent_messages: list[str] = []

    @property
    def window_size(self) -> int:
        """Get window size."""
        return self._window_size

    def filter(self, message: 'LogMessage') -> FilterAction:
        """Apply deduplication."""
        if not self._enabled:
            return FilterAction.PASS

        # Create message key
        key = f"{message.level}:{message.category}:{message.message}"

        if key in self._recent_messages:
            return FilterAction.DROP

        # Add to recent
        self._recent_messages.append(key)
        if len(self._recent_messages) > self._window_size:
            self._recent_messages.pop(0)

        return FilterAction.PASS


class CompositeFilter(LogFilter):
    """Filter that combines multiple filters.

    Supports AND and OR combination modes.
    """
    __slots__ = ('_filters', '_mode')

    def __init__(
        self,
        filters: Optional[list[LogFilter]] = None,
        mode: str = "and",
        name: str = ""
    ):
        """Initialize composite filter.

        Args:
            filters: Child filters
            mode: "and" (all must pass) or "or" (any must pass)
            name: Filter identifier
        """
        super().__init__(name or "CompositeFilter")
        self._filters = list(filters) if filters else []
        self._mode = mode.lower()

    @property
    def filters(self) -> list[LogFilter]:
        """Get child filters."""
        return self._filters

    @property
    def mode(self) -> str:
        """Get combination mode."""
        return self._mode

    def add_filter(self, log_filter: LogFilter) -> None:
        """Add a child filter.

        Args:
            log_filter: Filter to add
        """
        if log_filter not in self._filters:
            self._filters.append(log_filter)

    def remove_filter(self, log_filter: LogFilter) -> None:
        """Remove a child filter.

        Args:
            log_filter: Filter to remove
        """
        try:
            self._filters.remove(log_filter)
        except ValueError:
            pass

    def filter(self, message: 'LogMessage') -> FilterAction:
        """Apply all filters."""
        if not self._enabled:
            return FilterAction.PASS

        if not self._filters:
            return FilterAction.PASS

        modified = False

        if self._mode == "or":
            # Any filter must pass
            for f in self._filters:
                result = f.filter(message)
                if result == FilterAction.PASS:
                    return FilterAction.PASS
                elif result == FilterAction.MODIFY:
                    modified = True

            return FilterAction.MODIFY if modified else FilterAction.DROP

        else:  # "and" mode
            # All filters must pass
            for f in self._filters:
                result = f.filter(message)
                if result == FilterAction.DROP:
                    return FilterAction.DROP
                elif result == FilterAction.MODIFY:
                    modified = True

            return FilterAction.MODIFY if modified else FilterAction.PASS


class CallbackFilter(LogFilter):
    """Filter that calls a custom function.

    Allows arbitrary filtering logic.
    """
    __slots__ = ('_callback',)

    def __init__(
        self,
        callback: callable,
        name: str = ""
    ):
        """Initialize callback filter.

        Args:
            callback: Function that takes LogMessage and returns FilterAction
            name: Filter identifier
        """
        super().__init__(name or "CallbackFilter")
        self._callback = callback

    def filter(self, message: 'LogMessage') -> FilterAction:
        """Apply callback."""
        if not self._enabled:
            return FilterAction.PASS

        try:
            result = self._callback(message)
            if isinstance(result, FilterAction):
                return result
            elif isinstance(result, bool):
                return FilterAction.PASS if result else FilterAction.DROP
            else:
                return FilterAction.PASS
        except Exception:
            return FilterAction.PASS
