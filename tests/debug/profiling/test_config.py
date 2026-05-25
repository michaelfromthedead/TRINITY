"""Tests for profiling configuration."""

import pytest

from engine.debug.profiling.config import (
    cpu_warn_threshold_ms,
    gpu_average_frames,
    gpu_frame_history_size,
    memory_freed_history_max,
    memory_freed_history_trim,
    memory_large_allocation_bytes,
    memory_leak_min_age_seconds,
    memory_medium_allocation_bytes,
    memory_stack_trace_depth,
    network_bandwidth_history_seconds,
    network_packet_history_size,
    network_packet_timeout_multiplier,
    network_rtt_sample_size,
    network_stats_window_seconds,
    stats_graph_history_size,
    stats_timer_history_size,
)
from engine.debug.console.cvar import CVarFlags


class TestGPUConfig:
    """Tests for GPU profiler configuration CVars."""

    def test_gpu_frame_history_size_default(self) -> None:
        """Test default value for GPU frame history size."""
        assert gpu_frame_history_size.default == 120
        assert gpu_frame_history_size.value == 120

    def test_gpu_average_frames_default(self) -> None:
        """Test default value for GPU average frames."""
        assert gpu_average_frames.default == 60
        assert gpu_average_frames.value == 60

    def test_gpu_cvars_have_config_flag(self) -> None:
        """Test GPU CVars have CONFIG flag for persistence."""
        assert CVarFlags.CONFIG in gpu_frame_history_size.flags
        assert CVarFlags.CONFIG in gpu_average_frames.flags


class TestMemoryConfig:
    """Tests for memory profiler configuration CVars."""

    def test_memory_stack_trace_depth_default(self) -> None:
        """Test default value for stack trace depth."""
        assert memory_stack_trace_depth.default == 10

    def test_memory_leak_min_age_seconds_default(self) -> None:
        """Test default value for leak detection min age."""
        assert memory_leak_min_age_seconds.default == 60.0

    def test_memory_freed_history_max_default(self) -> None:
        """Test default value for freed history max."""
        assert memory_freed_history_max.default == 10000

    def test_memory_freed_history_trim_default(self) -> None:
        """Test default value for freed history trim."""
        assert memory_freed_history_trim.default == 5000

    def test_memory_large_allocation_bytes_default(self) -> None:
        """Test default value for large allocation threshold."""
        assert memory_large_allocation_bytes.default == 1024 * 1024  # 1 MB

    def test_memory_medium_allocation_bytes_default(self) -> None:
        """Test default value for medium allocation threshold."""
        assert memory_medium_allocation_bytes.default == 1024 * 100  # 100 KB

    def test_memory_cvars_have_config_flag(self) -> None:
        """Test memory CVars have CONFIG flag for persistence."""
        assert CVarFlags.CONFIG in memory_stack_trace_depth.flags
        assert CVarFlags.CONFIG in memory_leak_min_age_seconds.flags
        assert CVarFlags.CONFIG in memory_freed_history_max.flags


class TestNetworkConfig:
    """Tests for network profiler configuration CVars."""

    def test_network_stats_window_seconds_default(self) -> None:
        """Test default value for stats window."""
        assert network_stats_window_seconds.default == 1.0

    def test_network_packet_history_size_default(self) -> None:
        """Test default value for packet history size."""
        assert network_packet_history_size.default == 1000

    def test_network_rtt_sample_size_default(self) -> None:
        """Test default value for RTT sample size."""
        assert network_rtt_sample_size.default == 100

    def test_network_packet_timeout_multiplier_default(self) -> None:
        """Test default value for packet timeout multiplier."""
        assert network_packet_timeout_multiplier.default == 5.0

    def test_network_bandwidth_history_seconds_default(self) -> None:
        """Test default value for bandwidth history duration."""
        assert network_bandwidth_history_seconds.default == 10.0

    def test_network_cvars_have_config_flag(self) -> None:
        """Test network CVars have CONFIG flag for persistence."""
        assert CVarFlags.CONFIG in network_stats_window_seconds.flags
        assert CVarFlags.CONFIG in network_packet_history_size.flags


class TestStatsConfig:
    """Tests for stats configuration CVars."""

    def test_stats_timer_history_size_default(self) -> None:
        """Test default value for timer history size."""
        assert stats_timer_history_size.default == 100

    def test_stats_graph_history_size_default(self) -> None:
        """Test default value for graph history size."""
        assert stats_graph_history_size.default == 120

    def test_stats_cvars_have_config_flag(self) -> None:
        """Test stats CVars have CONFIG flag for persistence."""
        assert CVarFlags.CONFIG in stats_timer_history_size.flags
        assert CVarFlags.CONFIG in stats_graph_history_size.flags


class TestCPUConfig:
    """Tests for CPU profiler configuration CVars."""

    def test_cpu_warn_threshold_ms_default(self) -> None:
        """Test default value for CPU warning threshold."""
        assert cpu_warn_threshold_ms.default == 16.67

    def test_cpu_cvars_have_config_flag(self) -> None:
        """Test CPU CVars have CONFIG flag for persistence."""
        assert CVarFlags.CONFIG in cpu_warn_threshold_ms.flags
