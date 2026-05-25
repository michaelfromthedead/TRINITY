"""Tests for statistics system."""

import time

import pytest

from engine.debug.profiling.stats import (
    BarStat,
    CounterStat,
    GraphStat,
    Stat,
    Stats,
    StatType,
    TimerStat,
    get_default_stats,
    set_default_stats,
)


class TestStatType:
    """Tests for StatType enum."""

    def test_all_types_exist(self) -> None:
        """Test all expected types exist."""
        expected = ["COUNTER", "TIMER", "GRAPH", "BAR"]
        for type_name in expected:
            assert hasattr(StatType, type_name)


class TestCounterStat:
    """Tests for CounterStat class."""

    def test_initial_value(self) -> None:
        """Test counter with initial value."""
        counter = CounterStat("test", initial_value=10.0)
        assert counter.get_value() == 10.0

    def test_increment(self) -> None:
        """Test incrementing counter."""
        counter = CounterStat("test")

        result = counter.increment(5.0)
        assert result == 5.0
        assert counter.get_value() == 5.0

        result = counter.increment(3.0)
        assert result == 8.0

    def test_decrement(self) -> None:
        """Test decrementing counter."""
        counter = CounterStat("test", initial_value=10.0)

        result = counter.decrement(3.0)
        assert result == 7.0

    def test_set(self) -> None:
        """Test setting counter value."""
        counter = CounterStat("test")
        counter.set(42.0)
        assert counter.get_value() == 42.0

    def test_reset(self) -> None:
        """Test resetting counter."""
        counter = CounterStat("test", initial_value=5.0)
        counter.increment(10.0)
        assert counter.get_value() == 15.0

        counter.reset()
        assert counter.get_value() == 5.0

    def test_stat_type(self) -> None:
        """Test counter has correct stat type."""
        counter = CounterStat("test")
        assert counter.stat_type == StatType.COUNTER


class TestTimerStat:
    """Tests for TimerStat class."""

    def test_record_timing(self) -> None:
        """Test recording a timing measurement."""
        timer = TimerStat("test")
        timer.record(16.67)
        assert timer.get_value() == 16.67
        assert timer.last_ms == 16.67

    def test_average_timing(self) -> None:
        """Test average calculation."""
        timer = TimerStat("test")
        timer.record(10.0)
        timer.record(20.0)
        timer.record(30.0)

        assert timer.average_ms == 20.0

    def test_min_max(self) -> None:
        """Test min/max tracking."""
        timer = TimerStat("test")
        timer.record(10.0)
        timer.record(50.0)
        timer.record(30.0)

        assert timer.min_ms == 10.0
        assert timer.max_ms == 50.0

    def test_history(self) -> None:
        """Test timing history."""
        timer = TimerStat("test", history_size=3)
        timer.record(10.0)
        timer.record(20.0)
        timer.record(30.0)
        timer.record(40.0)  # Should push out 10.0

        history = timer.get_history()
        assert len(history) == 3
        assert history == [20.0, 30.0, 40.0]

    def test_count(self) -> None:
        """Test sample count."""
        timer = TimerStat("test")
        timer.record(10.0)
        timer.record(20.0)

        assert timer.count == 2

    def test_threshold_check(self) -> None:
        """Test warning threshold check."""
        timer = TimerStat("test", warn_threshold_ms=16.67)

        timer.record(10.0)
        assert timer.is_over_threshold() is False

        timer.record(20.0)
        assert timer.is_over_threshold() is True

    def test_reset(self) -> None:
        """Test resetting timer."""
        timer = TimerStat("test")
        timer.record(10.0)
        timer.record(20.0)

        timer.reset()

        assert timer.get_value() == 0.0
        assert timer.count == 0
        assert timer.min_ms == 0.0
        assert timer.max_ms == 0.0

    def test_stat_type(self) -> None:
        """Test timer has correct stat type."""
        timer = TimerStat("test")
        assert timer.stat_type == StatType.TIMER


class TestGraphStat:
    """Tests for GraphStat class."""

    def test_record_value(self) -> None:
        """Test recording a value."""
        graph = GraphStat("test")
        graph.record(60.0)
        assert graph.current == 60.0

    def test_get_values(self) -> None:
        """Test getting all values."""
        graph = GraphStat("test")
        graph.record(60.0)
        graph.record(59.0)
        graph.record(61.0)

        values = graph.get_value()
        assert values == [60.0, 59.0, 61.0]

    def test_history_size(self) -> None:
        """Test history size limiting."""
        graph = GraphStat("test", history_size=3)
        graph.record(1.0)
        graph.record(2.0)
        graph.record(3.0)
        graph.record(4.0)

        values = graph.get_value()
        assert values == [2.0, 3.0, 4.0]

    def test_min_max(self) -> None:
        """Test min/max tracking."""
        graph = GraphStat("test")
        graph.record(10.0)
        graph.record(50.0)
        graph.record(30.0)

        assert graph.min == 10.0
        assert graph.max == 50.0

    def test_average(self) -> None:
        """Test average calculation."""
        graph = GraphStat("test")
        graph.record(10.0)
        graph.record(20.0)
        graph.record(30.0)

        assert graph.average == 20.0

    def test_values_with_timestamps(self) -> None:
        """Test getting values with timestamps."""
        graph = GraphStat("test")
        before = time.time()
        graph.record(60.0)
        after = time.time()

        values = graph.get_values_with_timestamps()
        assert len(values) == 1
        timestamp, value = values[0]
        assert before <= timestamp <= after
        assert value == 60.0

    def test_reset(self) -> None:
        """Test resetting graph."""
        graph = GraphStat("test")
        graph.record(60.0)

        graph.reset()

        assert graph.get_value() == []
        assert graph.min == 0.0
        assert graph.max == 0.0

    def test_stat_type(self) -> None:
        """Test graph has correct stat type."""
        graph = GraphStat("test")
        assert graph.stat_type == StatType.GRAPH


class TestBarStat:
    """Tests for BarStat class."""

    def test_set_category(self) -> None:
        """Test setting a category value."""
        bar = BarStat("test")
        bar.set_category("rendering", 50.0)

        assert bar.get_category("rendering") == 50.0

    def test_increment_category(self) -> None:
        """Test incrementing a category."""
        bar = BarStat("test")
        bar.increment_category("count")
        bar.increment_category("count")
        bar.increment_category("count", 5.0)

        assert bar.get_category("count") == 7.0

    def test_get_all_values(self) -> None:
        """Test getting all category values."""
        bar = BarStat("test")
        bar.set_category("a", 10.0)
        bar.set_category("b", 20.0)
        bar.set_category("c", 30.0)

        values = bar.get_value()
        assert values == {"a": 10.0, "b": 20.0, "c": 30.0}

    def test_total(self) -> None:
        """Test total calculation."""
        bar = BarStat("test")
        bar.set_category("a", 10.0)
        bar.set_category("b", 20.0)

        assert bar.total == 30.0

    def test_nonexistent_category(self) -> None:
        """Test getting non-existent category returns 0."""
        bar = BarStat("test")
        assert bar.get_category("does_not_exist") == 0.0

    def test_reset(self) -> None:
        """Test resetting bar."""
        bar = BarStat("test")
        bar.set_category("a", 10.0)

        bar.reset()

        assert bar.get_value() == {}

    def test_stat_type(self) -> None:
        """Test bar has correct stat type."""
        bar = BarStat("test")
        assert bar.stat_type == StatType.BAR


class TestStats:
    """Tests for Stats manager class."""

    def test_counter_shorthand(self) -> None:
        """Test counter shorthand method."""
        stats = Stats()

        result = stats.counter("test_counter", 5.0)
        assert result == 5.0

        result = stats.counter("test_counter", 3.0)
        assert result == 8.0

    def test_timer_shorthand(self) -> None:
        """Test timer shorthand method."""
        stats = Stats()
        stats.timer("test_timer", 16.67)

        stat = stats.get_stat("test_timer")
        assert stat is not None
        assert isinstance(stat, TimerStat)
        assert stat.last_ms == 16.67

    def test_graph_shorthand(self) -> None:
        """Test graph shorthand method."""
        stats = Stats()
        stats.graph("test_graph", 60.0)

        stat = stats.get_stat("test_graph")
        assert stat is not None
        assert isinstance(stat, GraphStat)
        assert stat.current == 60.0

    def test_bar_shorthand(self) -> None:
        """Test bar shorthand method."""
        stats = Stats()
        stats.bar("test_bar", "category1", 10.0)

        stat = stats.get_stat("test_bar")
        assert stat is not None
        assert isinstance(stat, BarStat)
        assert stat.get_category("category1") == 10.0

    def test_bar_increment_shorthand(self) -> None:
        """Test bar_increment shorthand method."""
        stats = Stats()
        stats.bar_increment("test_bar", "count")
        stats.bar_increment("test_bar", "count")

        stat = stats.get_stat("test_bar")
        assert stat is not None
        assert stat.get_category("count") == 2.0

    def test_get_value(self) -> None:
        """Test getting stat value directly."""
        stats = Stats()
        stats.counter("my_counter", 42.0)

        value = stats.get_value("my_counter")
        assert value == 42.0

    def test_get_value_nonexistent(self) -> None:
        """Test getting non-existent stat returns None."""
        stats = Stats()
        assert stats.get_value("does_not_exist") is None

    def test_builtin_fps_group(self) -> None:
        """Test built-in fps stat group."""
        stats = Stats()

        fps_stats = stats.get_group("fps")
        assert "fps.current" in fps_stats
        assert "fps.frame_time" in fps_stats
        assert "fps.frame_count" in fps_stats

    def test_builtin_memory_group(self) -> None:
        """Test built-in memory stat group."""
        stats = Stats()

        memory_stats = stats.get_group("memory")
        assert "memory.used_mb" in memory_stats
        assert "memory.allocated_mb" in memory_stats
        assert "memory.by_category" in memory_stats

    def test_builtin_gpu_group(self) -> None:
        """Test built-in gpu stat group."""
        stats = Stats()

        gpu_stats = stats.get_group("gpu")
        assert "gpu.frame_time" in gpu_stats
        assert "gpu.pass_times" in gpu_stats
        assert "gpu.utilization" in gpu_stats

    def test_builtin_unit_group(self) -> None:
        """Test built-in unit stat group."""
        stats = Stats()

        unit_stats = stats.get_group("unit")
        assert "unit.total" in unit_stats
        assert "unit.active" in unit_stats
        assert "unit.by_type" in unit_stats
        assert "unit.update_time" in unit_stats

    def test_define_custom_group(self) -> None:
        """Test defining a custom stat group."""
        stats = Stats()
        stats.counter("custom.a")
        stats.counter("custom.b")

        stats.define_group("custom", ["custom.a", "custom.b"])

        custom_stats = stats.get_group("custom")
        assert "custom.a" in custom_stats
        assert "custom.b" in custom_stats

    def test_get_group_values(self) -> None:
        """Test getting all values in a group."""
        stats = Stats()
        stats.counter("custom.a", 10.0)
        stats.counter("custom.b", 20.0)
        stats.define_group("custom", ["custom.a", "custom.b"])

        values = stats.get_group_values("custom")
        assert values["custom.a"] == 10.0
        assert values["custom.b"] == 20.0

    def test_list_stats(self) -> None:
        """Test listing all stat names."""
        stats = Stats()
        stats.counter("stat1")
        stats.timer("stat2", 0)
        stats.graph("stat3", 0)

        names = stats.list_stats()
        assert "stat1" in names
        assert "stat2" in names
        assert "stat3" in names

    def test_list_groups(self) -> None:
        """Test listing all group names."""
        stats = Stats()
        groups = stats.list_groups()

        assert "fps" in groups
        assert "memory" in groups
        assert "gpu" in groups
        assert "unit" in groups

    def test_reset_single(self) -> None:
        """Test resetting a single stat."""
        stats = Stats()
        stats.counter("test", 10.0)

        stats.reset("test")

        assert stats.get_value("test") == 0.0

    def test_reset_all(self) -> None:
        """Test resetting all stats."""
        stats = Stats()
        stats.counter("a", 10.0)
        stats.counter("b", 20.0)

        stats.reset()

        assert stats.get_value("a") == 0.0
        assert stats.get_value("b") == 0.0

    def test_reset_group(self) -> None:
        """Test resetting a stat group."""
        stats = Stats()
        stats.counter("custom.a", 10.0)
        stats.counter("custom.b", 20.0)
        stats.counter("other", 30.0)
        stats.define_group("custom", ["custom.a", "custom.b"])

        stats.reset_group("custom")

        assert stats.get_value("custom.a") == 0.0
        assert stats.get_value("custom.b") == 0.0
        assert stats.get_value("other") == 30.0  # Unchanged

    def test_format_group(self) -> None:
        """Test formatting a group as string."""
        stats = Stats()
        stats.graph("fps.current", 60.0)
        stats.timer("fps.frame_time", 16.67)
        stats.counter("fps.frame_count", 1000)

        output = stats.format_group("fps")

        assert "fps Stats:" in output
        assert "current" in output
        assert "frame_time" in output
        assert "frame_count" in output

    def test_format_empty_group(self) -> None:
        """Test formatting non-existent group."""
        stats = Stats()
        output = stats.format_group("does_not_exist")
        assert "No stats" in output


class TestDefaultStats:
    """Tests for default stats instance."""

    def test_get_default_stats(self) -> None:
        """Test getting default stats instance."""
        stats = get_default_stats()
        assert stats is not None
        assert isinstance(stats, Stats)

    def test_set_default_stats(self) -> None:
        """Test setting default stats instance."""
        original = get_default_stats()
        new_stats = Stats()

        set_default_stats(new_stats)
        assert get_default_stats() is new_stats

        # Restore original
        set_default_stats(original)
