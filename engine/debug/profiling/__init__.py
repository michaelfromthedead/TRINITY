"""Profiling system for game engine performance analysis.

This module provides comprehensive profiling tools for:
- CPU timing and execution profiling
- GPU render pass timing
- Memory allocation tracking and leak detection
- Network bandwidth and latency monitoring
- General statistics collection and visualization
"""

# Import config first to ensure CVars are registered
from engine.debug.profiling import config as profiling_config

from engine.debug.profiling.cpu import (
    CPUProfiler,
    FlatProfileEntry,
    ProfileSample,
    get_default_profiler,
    profile,
    profile_scope,
    set_default_profiler,
)
from engine.debug.profiling.gpu import (
    GPUFrameTiming,
    GPUPassTiming,
    GPUPassType,
    GPUProfiler,
    get_default_gpu_profiler,
    set_default_gpu_profiler,
)
from engine.debug.profiling.memory import (
    AllocationRecord,
    LeakCandidate,
    MemoryDiff,
    MemoryProfiler,
    MemorySnapshot,
    MemoryTag,
    get_default_memory_profiler,
    set_default_memory_profiler,
)
from engine.debug.profiling.network import (
    ConnectionStats,
    NetworkProfiler,
    NetworkStats,
    PacketDirection,
    PacketRecord,
    PacketType,
    get_default_network_profiler,
    set_default_network_profiler,
)
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

__all__ = [
    # Config
    "profiling_config",
    # CPU Profiler
    "ProfileSample",
    "FlatProfileEntry",
    "CPUProfiler",
    "get_default_profiler",
    "set_default_profiler",
    "profile",
    "profile_scope",
    # GPU Profiler
    "GPUPassType",
    "GPUPassTiming",
    "GPUFrameTiming",
    "GPUProfiler",
    "get_default_gpu_profiler",
    "set_default_gpu_profiler",
    # Memory Profiler
    "MemoryTag",
    "AllocationRecord",
    "MemorySnapshot",
    "MemoryDiff",
    "LeakCandidate",
    "MemoryProfiler",
    "get_default_memory_profiler",
    "set_default_memory_profiler",
    # Network Profiler
    "PacketType",
    "PacketDirection",
    "PacketRecord",
    "NetworkStats",
    "ConnectionStats",
    "NetworkProfiler",
    "get_default_network_profiler",
    "set_default_network_profiler",
    # Stats
    "StatType",
    "Stat",
    "CounterStat",
    "TimerStat",
    "GraphStat",
    "BarStat",
    "Stats",
    "get_default_stats",
    "set_default_stats",
]
