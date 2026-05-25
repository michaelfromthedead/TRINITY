"""Statistics system for game engine performance monitoring.

Provides counters, timers, graphs, and bar charts for visualizing
and tracking game metrics.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

from engine.debug.profiling import config as profiling_config


class StatType(Enum):
    """Types of statistics."""

    COUNTER = auto()  # Simple incrementing counter
    TIMER = auto()    # Time measurements
    GRAPH = auto()    # Time-series data
    BAR = auto()      # Categorical data


@dataclass
class StatValue:
    """A single stat sample."""

    value: float
    timestamp: float = field(default_factory=time.time)


class Stat(ABC):
    """Base class for all statistics."""

    def __init__(self, name: str, stat_type: StatType) -> None:
        """Initialize the stat.

        Args:
            name: Name of the statistic.
            stat_type: Type of statistic.
        """
        self._name = name
        self._type = stat_type
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def stat_type(self) -> StatType:
        return self._type

    @abstractmethod
    def get_value(self) -> Union[float, Dict[str, float], List[float]]:
        """Get the current value of the stat."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the stat to initial state."""
        pass


class CounterStat(Stat):
    """Counter statistic that can be incremented or decremented."""

    def __init__(self, name: str, initial_value: float = 0.0) -> None:
        super().__init__(name, StatType.COUNTER)
        self._value = initial_value
        self._initial = initial_value
        self._total_increments = 0
        self._total_decrements = 0

    def increment(self, amount: float = 1.0) -> float:
        """Increment the counter.

        Args:
            amount: Amount to add.

        Returns:
            New value.
        """
        with self._lock:
            self._value += amount
            if amount >= 0:
                self._total_increments += int(amount)
            else:
                self._total_decrements += int(-amount)
            return self._value

    def decrement(self, amount: float = 1.0) -> float:
        """Decrement the counter.

        Args:
            amount: Amount to subtract.

        Returns:
            New value.
        """
        return self.increment(-amount)

    def set(self, value: float) -> None:
        """Set the counter to a specific value.

        Args:
            value: New value.
        """
        with self._lock:
            self._value = value

    def get_value(self) -> float:
        with self._lock:
            return self._value

    def reset(self) -> None:
        with self._lock:
            self._value = self._initial
            self._total_increments = 0
            self._total_decrements = 0


class TimerStat(Stat):
    """Timer statistic for measuring execution time."""

    def __init__(
        self,
        name: str,
        history_size: Optional[int] = None,
        warn_threshold_ms: Optional[float] = None
    ) -> None:
        super().__init__(name, StatType.TIMER)
        effective_size = (
            history_size if history_size is not None
            else profiling_config.stats_timer_history_size.value
        )
        self._history: Deque[float] = deque(maxlen=effective_size)
        self._warn_threshold = warn_threshold_ms
        self._min_ms: Optional[float] = None
        self._max_ms: Optional[float] = None
        self._total_ms: float = 0.0
        self._count: int = 0

    def record(self, duration_ms: float) -> None:
        """Record a timing measurement.

        Args:
            duration_ms: Duration in milliseconds.
        """
        with self._lock:
            self._history.append(duration_ms)
            self._total_ms += duration_ms
            self._count += 1

            if self._min_ms is None or duration_ms < self._min_ms:
                self._min_ms = duration_ms
            if self._max_ms is None or duration_ms > self._max_ms:
                self._max_ms = duration_ms

    def get_value(self) -> float:
        """Get the average timing."""
        with self._lock:
            if not self._history:
                return 0.0
            return sum(self._history) / len(self._history)

    @property
    def average_ms(self) -> float:
        return self.get_value()

    @property
    def min_ms(self) -> float:
        return self._min_ms or 0.0

    @property
    def max_ms(self) -> float:
        return self._max_ms or 0.0

    @property
    def last_ms(self) -> float:
        with self._lock:
            return self._history[-1] if self._history else 0.0

    @property
    def total_ms(self) -> float:
        return self._total_ms

    @property
    def count(self) -> int:
        return self._count

    def get_history(self) -> List[float]:
        with self._lock:
            return list(self._history)

    def is_over_threshold(self) -> bool:
        """Check if the last measurement exceeded the warning threshold."""
        if self._warn_threshold is None:
            return False
        with self._lock:
            if not self._history:
                return False
            return self._history[-1] > self._warn_threshold

    def reset(self) -> None:
        with self._lock:
            self._history.clear()
            self._min_ms = None
            self._max_ms = None
            self._total_ms = 0.0
            self._count = 0


class GraphStat(Stat):
    """Graph statistic for time-series data visualization."""

    def __init__(self, name: str, history_size: Optional[int] = None) -> None:
        super().__init__(name, StatType.GRAPH)
        effective_size = (
            history_size if history_size is not None
            else profiling_config.stats_graph_history_size.value
        )
        self._history_size = effective_size
        self._values: Deque[StatValue] = deque(maxlen=effective_size)
        self._min: Optional[float] = None
        self._max: Optional[float] = None

    def record(self, value: float) -> None:
        """Record a data point.

        Args:
            value: Value to record.
        """
        with self._lock:
            self._values.append(StatValue(value=value))

            if self._min is None or value < self._min:
                self._min = value
            if self._max is None or value > self._max:
                self._max = value

    def get_value(self) -> List[float]:
        """Get all recorded values."""
        with self._lock:
            return [v.value for v in self._values]

    def get_values_with_timestamps(self) -> List[Tuple[float, float]]:
        """Get values with their timestamps."""
        with self._lock:
            return [(v.timestamp, v.value) for v in self._values]

    @property
    def current(self) -> float:
        with self._lock:
            return self._values[-1].value if self._values else 0.0

    @property
    def average(self) -> float:
        with self._lock:
            if not self._values:
                return 0.0
            return sum(v.value for v in self._values) / len(self._values)

    @property
    def min(self) -> float:
        return self._min or 0.0

    @property
    def max(self) -> float:
        return self._max or 0.0

    def reset(self) -> None:
        with self._lock:
            self._values.clear()
            self._min = None
            self._max = None


class BarStat(Stat):
    """Bar chart statistic for categorical data."""

    def __init__(self, name: str) -> None:
        super().__init__(name, StatType.BAR)
        self._categories: Dict[str, float] = {}

    def set_category(self, category: str, value: float) -> None:
        """Set a category value.

        Args:
            category: Category name.
            value: Category value.
        """
        with self._lock:
            self._categories[category] = value

    def increment_category(self, category: str, amount: float = 1.0) -> None:
        """Increment a category value.

        Args:
            category: Category name.
            amount: Amount to add.
        """
        with self._lock:
            current = self._categories.get(category, 0.0)
            self._categories[category] = current + amount

    def get_value(self) -> Dict[str, float]:
        """Get all category values."""
        with self._lock:
            return dict(self._categories)

    def get_category(self, category: str) -> float:
        """Get a specific category value."""
        with self._lock:
            return self._categories.get(category, 0.0)

    @property
    def total(self) -> float:
        with self._lock:
            return sum(self._categories.values())

    def reset(self) -> None:
        with self._lock:
            self._categories.clear()


class Stats:
    """Central statistics manager for the game engine.

    Provides methods to create and access various types of statistics.

    Example:
        stats = Stats()

        # Counter
        stats.counter("entities_spawned", 1)

        # Timer
        stats.timer("frame_time", 16.7)

        # Graph
        stats.graph("fps", 60.0)

        # Get stats
        fps_stat = stats.get_stat("fps")
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stats: Dict[str, Stat] = {}
        self._groups: Dict[str, List[str]] = {}

        # Initialize built-in stat groups
        self._init_builtin_groups()

    def _init_builtin_groups(self) -> None:
        """Initialize built-in stat groups."""
        graph_size = profiling_config.stats_graph_history_size.value

        # FPS group
        self._create_graph("fps.current", history_size=graph_size)
        self._create_timer("fps.frame_time")
        self._create_counter("fps.frame_count")
        self._groups["fps"] = ["fps.current", "fps.frame_time", "fps.frame_count"]

        # Memory group
        self._create_graph("memory.used_mb", history_size=graph_size)
        self._create_graph("memory.allocated_mb", history_size=graph_size)
        self._create_bar("memory.by_category")
        self._groups["memory"] = [
            "memory.used_mb", "memory.allocated_mb", "memory.by_category"
        ]

        # GPU group
        self._create_timer("gpu.frame_time")
        self._create_bar("gpu.pass_times")
        self._create_graph("gpu.utilization", history_size=graph_size)
        self._groups["gpu"] = [
            "gpu.frame_time", "gpu.pass_times", "gpu.utilization"
        ]

        # Unit (entity) group
        self._create_counter("unit.total")
        self._create_counter("unit.active")
        self._create_bar("unit.by_type")
        self._create_timer("unit.update_time")
        self._groups["unit"] = [
            "unit.total", "unit.active", "unit.by_type", "unit.update_time"
        ]

    def _create_graph(
        self,
        name: str,
        history_size: Optional[int] = None
    ) -> GraphStat:
        """Create a graph stat if it doesn't exist."""
        if name not in self._stats:
            self._stats[name] = GraphStat(name, history_size)
        return self._stats[name]  # type: ignore

    def _create_timer(
        self,
        name: str,
        history_size: Optional[int] = None,
        warn_threshold_ms: Optional[float] = None
    ) -> TimerStat:
        """Create a timer stat if it doesn't exist."""
        if name not in self._stats:
            self._stats[name] = TimerStat(name, history_size, warn_threshold_ms)
        return self._stats[name]  # type: ignore

    def _create_counter(
        self,
        name: str,
        initial_value: float = 0.0
    ) -> CounterStat:
        """Create a counter stat if it doesn't exist."""
        if name not in self._stats:
            self._stats[name] = CounterStat(name, initial_value)
        return self._stats[name]  # type: ignore

    def _create_bar(self, name: str) -> BarStat:
        """Create a bar stat if it doesn't exist."""
        if name not in self._stats:
            self._stats[name] = BarStat(name)
        return self._stats[name]  # type: ignore

    def counter(self, name: str, value: float = 1.0) -> float:
        """Increment a counter stat.

        Args:
            name: Name of the counter.
            value: Value to add (negative to decrement).

        Returns:
            New counter value.
        """
        with self._lock:
            stat = self._create_counter(name)
            return stat.increment(value)

    def timer(self, name: str, ms: float) -> None:
        """Record a timing measurement.

        Args:
            name: Name of the timer.
            ms: Duration in milliseconds.
        """
        with self._lock:
            stat = self._create_timer(name)
            stat.record(ms)

    def graph(
        self,
        name: str,
        value: float,
        history_size: Optional[int] = None
    ) -> None:
        """Record a value for a graph stat.

        Args:
            name: Name of the graph.
            value: Value to record.
            history_size: Size of history buffer.
        """
        with self._lock:
            stat = self._create_graph(name, history_size)
            stat.record(value)

    def bar(self, name: str, category: str, value: float) -> None:
        """Set a category value for a bar stat.

        Args:
            name: Name of the bar stat.
            category: Category name.
            value: Category value.
        """
        with self._lock:
            stat = self._create_bar(name)
            stat.set_category(category, value)

    def bar_increment(
        self,
        name: str,
        category: str,
        amount: float = 1.0
    ) -> None:
        """Increment a category value for a bar stat.

        Args:
            name: Name of the bar stat.
            category: Category name.
            amount: Amount to add.
        """
        with self._lock:
            stat = self._create_bar(name)
            stat.increment_category(category, amount)

    def get_stat(self, name: str) -> Optional[Stat]:
        """Get a stat by name.

        Args:
            name: Name of the stat.

        Returns:
            The stat or None if not found.
        """
        with self._lock:
            return self._stats.get(name)

    def get_value(self, name: str) -> Optional[Any]:
        """Get the value of a stat.

        Args:
            name: Name of the stat.

        Returns:
            The stat's value or None if not found.
        """
        with self._lock:
            stat = self._stats.get(name)
            if stat is None:
                return None
            return stat.get_value()

    def get_group(self, group_name: str) -> Dict[str, Stat]:
        """Get all stats in a group.

        Args:
            group_name: Name of the group.

        Returns:
            Dictionary of stat name to stat object.
        """
        stat_names = self._groups.get(group_name, [])
        return {
            name: self._stats[name]
            for name in stat_names
            if name in self._stats
        }

    def get_group_values(self, group_name: str) -> Dict[str, Any]:
        """Get all stat values in a group.

        Args:
            group_name: Name of the group.

        Returns:
            Dictionary of stat name to value.
        """
        return {
            name: stat.get_value()
            for name, stat in self.get_group(group_name).items()
        }

    def define_group(self, group_name: str, stat_names: List[str]) -> None:
        """Define a custom stat group.

        Args:
            group_name: Name for the group.
            stat_names: List of stat names to include.
        """
        with self._lock:
            self._groups[group_name] = stat_names

    def list_stats(self) -> List[str]:
        """List all stat names."""
        return list(self._stats.keys())

    def list_groups(self) -> List[str]:
        """List all group names."""
        return list(self._groups.keys())

    def reset(self, name: Optional[str] = None) -> None:
        """Reset stats.

        Args:
            name: Name of stat to reset, or None to reset all.
        """
        with self._lock:
            if name is not None:
                if name in self._stats:
                    self._stats[name].reset()
            else:
                for stat in self._stats.values():
                    stat.reset()

    def reset_group(self, group_name: str) -> None:
        """Reset all stats in a group.

        Args:
            group_name: Name of the group to reset.
        """
        with self._lock:
            stat_names = self._groups.get(group_name, [])
            for name in stat_names:
                if name in self._stats:
                    self._stats[name].reset()

    def format_group(self, group_name: str) -> str:
        """Format a stat group as a human-readable string.

        Args:
            group_name: Name of the group.

        Returns:
            Formatted string.
        """
        stats = self.get_group(group_name)
        if not stats:
            return f"No stats in group '{group_name}'"

        lines = [f"{group_name} Stats:", "-" * (len(group_name) + 7)]

        for name, stat in stats.items():
            short_name = name.split(".")[-1]
            value = stat.get_value()

            if isinstance(stat, CounterStat):
                lines.append(f"  {short_name}: {value:.0f}")
            elif isinstance(stat, TimerStat):
                lines.append(
                    f"  {short_name}: {value:.2f}ms "
                    f"(min: {stat.min_ms:.2f}, max: {stat.max_ms:.2f})"
                )
            elif isinstance(stat, GraphStat):
                lines.append(
                    f"  {short_name}: {stat.current:.2f} "
                    f"(avg: {stat.average:.2f})"
                )
            elif isinstance(stat, BarStat):
                if value:
                    parts = [f"{k}: {v:.1f}" for k, v in value.items()]
                    lines.append(f"  {short_name}: {{{', '.join(parts)}}}")
                else:
                    lines.append(f"  {short_name}: {{}}")

        return "\n".join(lines)


# Global default stats instance
_default_stats = Stats()


def get_default_stats() -> Stats:
    """Get the global default stats instance."""
    return _default_stats


def set_default_stats(stats: Stats) -> None:
    """Set the global default stats instance."""
    global _default_stats
    _default_stats = stats
