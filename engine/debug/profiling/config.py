"""Configuration for the profiling system.

Provides centralized configuration for all profilers using the CVar system.
This allows runtime tuning of profiling parameters through the console.
"""

from __future__ import annotations

from engine.debug.console.cvar import CVar, CVarFlags


# =============================================================================
# GPU Profiler Configuration
# =============================================================================

gpu_frame_history_size = CVar[int](
    name="profiler.gpu.FrameHistorySize",
    default=120,
    flags=CVarFlags.CONFIG,
    description="Number of GPU frames to keep in history for analysis"
)

gpu_average_frames = CVar[int](
    name="profiler.gpu.AverageFrames",
    default=60,
    flags=CVarFlags.CONFIG,
    description="Number of frames to average for GPU timing statistics"
)


# =============================================================================
# Memory Profiler Configuration
# =============================================================================

memory_stack_trace_depth = CVar[int](
    name="profiler.memory.StackTraceDepth",
    default=10,
    flags=CVarFlags.CONFIG,
    description="Maximum depth of stack traces captured for allocations"
)

memory_leak_min_age_seconds = CVar[float](
    name="profiler.memory.LeakMinAgeSeconds",
    default=60.0,
    flags=CVarFlags.CONFIG,
    description="Minimum age in seconds for an allocation to be considered a potential leak"
)

memory_freed_history_max = CVar[int](
    name="profiler.memory.FreedHistoryMax",
    default=10000,
    flags=CVarFlags.CONFIG,
    description="Maximum number of freed allocations to keep in history"
)

memory_freed_history_trim = CVar[int](
    name="profiler.memory.FreedHistoryTrim",
    default=5000,
    flags=CVarFlags.CONFIG,
    description="Number of freed allocations to keep when trimming history"
)

memory_large_allocation_bytes = CVar[int](
    name="profiler.memory.LargeAllocationBytes",
    default=1024 * 1024,  # 1 MB
    flags=CVarFlags.CONFIG,
    description="Size threshold in bytes for a 'large' allocation in leak detection"
)

memory_medium_allocation_bytes = CVar[int](
    name="profiler.memory.MediumAllocationBytes",
    default=1024 * 100,  # 100 KB
    flags=CVarFlags.CONFIG,
    description="Size threshold in bytes for a 'medium' allocation in leak detection"
)


# =============================================================================
# Network Profiler Configuration
# =============================================================================

network_stats_window_seconds = CVar[float](
    name="profiler.network.StatsWindowSeconds",
    default=1.0,
    flags=CVarFlags.CONFIG,
    description="Time window in seconds for calculating network statistics"
)

network_packet_history_size = CVar[int](
    name="profiler.network.PacketHistorySize",
    default=1000,
    flags=CVarFlags.CONFIG,
    description="Number of packets to keep in history"
)

network_rtt_sample_size = CVar[int](
    name="profiler.network.RttSampleSize",
    default=100,
    flags=CVarFlags.CONFIG,
    description="Number of RTT samples to keep for averaging"
)

network_packet_timeout_multiplier = CVar[float](
    name="profiler.network.PacketTimeoutMultiplier",
    default=5.0,
    flags=CVarFlags.CONFIG,
    description="Multiplier for window_seconds to determine packet loss timeout"
)

network_bandwidth_history_seconds = CVar[float](
    name="profiler.network.BandwidthHistorySeconds",
    default=10.0,
    flags=CVarFlags.CONFIG,
    description="Default duration in seconds for bandwidth history queries"
)


# =============================================================================
# Stats Configuration
# =============================================================================

stats_timer_history_size = CVar[int](
    name="profiler.stats.TimerHistorySize",
    default=100,
    flags=CVarFlags.CONFIG,
    description="Default history size for timer statistics"
)

stats_graph_history_size = CVar[int](
    name="profiler.stats.GraphHistorySize",
    default=120,
    flags=CVarFlags.CONFIG,
    description="Default history size for graph statistics"
)


# =============================================================================
# CPU Profiler Configuration
# =============================================================================

cpu_warn_threshold_ms = CVar[float](
    name="profiler.cpu.WarnThresholdMs",
    default=16.67,
    flags=CVarFlags.CONFIG,
    description="Default warning threshold in milliseconds for CPU profiling"
)
