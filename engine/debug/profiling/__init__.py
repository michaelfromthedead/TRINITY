"""Profiling system for game engine performance analysis.

This module provides comprehensive profiling tools for:
- CPU timing and execution profiling
- GPU render pass timing
- GPU timestamp instrumentation via wgpu query API
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
from engine.debug.profiling.gpu_timestamps import (
    FrameTimestamps,
    GPUTimestampEvent,
    GPUTimestampEventEmitter,
    GPUTimestampProfiler,
    GPUTimestampQuery,
    QueryState,
    RenderPassTimer,
    RingBufferEntry,
    TimestampPair,
    TimestampResult,
    TimestampRingBuffer,
    get_gpu_timestamp_profiler,
    initialize_gpu_timestamps,
    shutdown_gpu_timestamps,
)
from engine.debug.profiling.event_stream import (
    BinaryTraceExporter,
    ChromeTracingExporter,
    EventCategory,
    EventRingBuffer,
    EventScope,
    EventSlot,
    EventStream,
    EventType,
    FrameScope,
    ProfileEvent,
    export_chrome_tracing,
    get_event_stream,
    initialize_event_stream,
    shutdown_event_stream,
)
from engine.debug.profiling.frame_budget import (
    AutoQualityAdjuster,
    BudgetState,
    BudgetViolation,
    BudgetViolationDetector,
    FrameBudget,
    FrameBudgetConfig,
    FrameBudgetManager,
    FrameTiming,
    RecoveryTracker,
    TierTransition,
    TierTransitionDirection,
    get_default_frame_budget_manager,
    reset_default_frame_budget_manager,
    set_default_frame_budget_manager,
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
    # GPU Timestamps (wgpu Query API)
    "GPUTimestampQuery",
    "TimestampRingBuffer",
    "RenderPassTimer",
    "GPUTimestampProfiler",
    "TimestampResult",
    "TimestampPair",
    "FrameTimestamps",
    "QueryState",
    "RingBufferEntry",
    "GPUTimestampEvent",
    "GPUTimestampEventEmitter",
    "get_gpu_timestamp_profiler",
    "initialize_gpu_timestamps",
    "shutdown_gpu_timestamps",
    # Event Stream (Chrome Tracing)
    "EventType",
    "EventScope",
    "EventCategory",
    "ProfileEvent",
    "EventSlot",
    "EventRingBuffer",
    "EventStream",
    "ChromeTracingExporter",
    "BinaryTraceExporter",
    "FrameScope",
    "get_event_stream",
    "initialize_event_stream",
    "shutdown_event_stream",
    "export_chrome_tracing",
    # Frame Budget System (T-CC-3.10)
    "FrameBudget",
    "BudgetViolationDetector",
    "RecoveryTracker",
    "AutoQualityAdjuster",
    "FrameBudgetManager",
    "BudgetState",
    "TierTransitionDirection",
    "FrameTiming",
    "BudgetViolation",
    "TierTransition",
    "FrameBudgetConfig",
    "get_default_frame_budget_manager",
    "set_default_frame_budget_manager",
    "reset_default_frame_budget_manager",
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
