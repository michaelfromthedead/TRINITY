"""
Profiling subsystem for the AI Game Engine tooling layer.

This module provides comprehensive profiling capabilities including:
- CPU profiling with flame graphs, call trees, and hot path detection
- GPU profiling with timing, draw calls, shader stats, and bandwidth tracking
- Memory profiling with allocation tracking, leak detection, and snapshots
- Network profiling with bandwidth, packet inspection, and latency graphs
- Per-frame breakdown with timeline and spike detection
- In-game overlay for real-time stats
- Profiling markers and decorators (@profile, @gpu_profile)
- Export to Chrome trace format, CSV, and custom formats
- Session comparison and diff analysis
"""

from __future__ import annotations

# Core profilers
from engine.tooling.profiling.cpu_profiler import (
    CPUProfiler,
    CPUProfileSample,
    CallTreeNode,
    FlameGraphData,
    HotPath,
    cpu_profiler,
)

from engine.tooling.profiling.gpu_profiler import (
    GPUProfiler,
    GPUProfileSample,
    DrawCallStats,
    ShaderStats,
    GPUMemoryStats,
    RenderPassTiming,
    gpu_profiler,
)

from engine.tooling.profiling.memory_profiler import (
    MemoryProfiler,
    AllocationRecord,
    MemorySnapshot,
    MemoryCategory,
    LeakReport,
    FragmentationStats,
    memory_profiler,
)

from engine.tooling.profiling.network_profiler import (
    NetworkProfiler,
    NetworkStats,
    PacketRecord,
    BandwidthSample,
    LatencyGraph,
    network_profiler,
)

from engine.tooling.profiling.frame_profiler import (
    FrameProfiler,
    FrameData,
    FrameTimeline,
    SpikeDetector,
    frame_profiler,
)

# UI and markers
from engine.tooling.profiling.profiler_overlay import (
    ProfilerOverlay,
    OverlayConfig,
    OverlayPanel,
)

from engine.tooling.profiling.profiler_markers import (
    profile,
    gpu_profile,
    ProfileMarker,
    GPUProfileMarker,
    MarkerScope,
    begin_marker,
    end_marker,
)

# Export and comparison
from engine.tooling.profiling.profiler_export import (
    ProfilerExporter,
    ChromeTraceExporter,
    CSVExporter,
    JSONExporter,
    ExportFormat,
)

from engine.tooling.profiling.profiler_compare import (
    ProfilerComparator,
    SessionDiff,
    ComparisonReport,
    RegressionDetector,
)

__all__ = [
    # CPU Profiler
    "CPUProfiler",
    "CPUProfileSample",
    "CallTreeNode",
    "FlameGraphData",
    "HotPath",
    "cpu_profiler",
    # GPU Profiler
    "GPUProfiler",
    "GPUProfileSample",
    "DrawCallStats",
    "ShaderStats",
    "GPUMemoryStats",
    "RenderPassTiming",
    "gpu_profiler",
    # Memory Profiler
    "MemoryProfiler",
    "AllocationRecord",
    "MemorySnapshot",
    "MemoryCategory",
    "LeakReport",
    "FragmentationStats",
    "memory_profiler",
    # Network Profiler
    "NetworkProfiler",
    "NetworkStats",
    "PacketRecord",
    "BandwidthSample",
    "LatencyGraph",
    "network_profiler",
    # Frame Profiler
    "FrameProfiler",
    "FrameData",
    "FrameTimeline",
    "SpikeDetector",
    "frame_profiler",
    # Overlay
    "ProfilerOverlay",
    "OverlayConfig",
    "OverlayPanel",
    # Markers
    "profile",
    "gpu_profile",
    "ProfileMarker",
    "GPUProfileMarker",
    "MarkerScope",
    "begin_marker",
    "end_marker",
    # Export
    "ProfilerExporter",
    "ChromeTraceExporter",
    "CSVExporter",
    "JSONExporter",
    "ExportFormat",
    # Comparison
    "ProfilerComparator",
    "SessionDiff",
    "ComparisonReport",
    "RegressionDetector",
]
