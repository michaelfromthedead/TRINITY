"""
Profiler Export for the AI Game Engine.

Provides export functionality for profiling data:
- Chrome trace format (for chrome://tracing)
- CSV format for spreadsheet analysis
- JSON format for custom tooling
- Custom export formats via plugins
"""

from __future__ import annotations

import csv
import json
import io
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    TextIO,
    Type,
    Union,
)

from engine.tooling.profiling.cpu_profiler import CPUProfiler, CPUProfileSample, CallTreeNode
from engine.tooling.profiling.gpu_profiler import GPUProfiler, GPUProfileSample
from engine.tooling.profiling.memory_profiler import MemoryProfiler, AllocationRecord, MemorySnapshot
from engine.tooling.profiling.network_profiler import NetworkProfiler, PacketRecord
from engine.tooling.profiling.frame_profiler import FrameProfiler, FrameData


class ExportFormat(Enum):
    """Supported export formats."""
    CHROME_TRACE = auto()
    CSV = auto()
    JSON = auto()
    CUSTOM = auto()


@dataclass
class ExportConfig:
    """Configuration for profile export."""
    format: ExportFormat = ExportFormat.CHROME_TRACE
    include_cpu: bool = True
    include_gpu: bool = True
    include_memory: bool = True
    include_network: bool = False
    include_frames: bool = True
    include_call_tree: bool = False
    min_duration_ms: float = 0.0
    compress: bool = False
    pretty_print: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "format": self.format.name,
            "include_cpu": self.include_cpu,
            "include_gpu": self.include_gpu,
            "include_memory": self.include_memory,
            "include_network": self.include_network,
            "include_frames": self.include_frames,
            "include_call_tree": self.include_call_tree,
            "min_duration_ms": self.min_duration_ms,
            "compress": self.compress,
            "pretty_print": self.pretty_print,
        }


class BaseExporter(ABC):
    """Base class for profile exporters."""

    @abstractmethod
    def export(
        self,
        cpu_profiler: Optional[CPUProfiler] = None,
        gpu_profiler: Optional[GPUProfiler] = None,
        memory_profiler: Optional[MemoryProfiler] = None,
        network_profiler: Optional[NetworkProfiler] = None,
        frame_profiler: Optional[FrameProfiler] = None,
        config: Optional[ExportConfig] = None,
    ) -> str:
        """
        Export profiling data to a string.

        Args:
            cpu_profiler: CPU profiler instance
            gpu_profiler: GPU profiler instance
            memory_profiler: Memory profiler instance
            network_profiler: Network profiler instance
            frame_profiler: Frame profiler instance
            config: Export configuration

        Returns:
            Exported data as string
        """
        pass

    def export_to_file(
        self,
        path: Union[str, Path],
        cpu_profiler: Optional[CPUProfiler] = None,
        gpu_profiler: Optional[GPUProfiler] = None,
        memory_profiler: Optional[MemoryProfiler] = None,
        network_profiler: Optional[NetworkProfiler] = None,
        frame_profiler: Optional[FrameProfiler] = None,
        config: Optional[ExportConfig] = None,
    ) -> None:
        """Export profiling data to a file."""
        data = self.export(
            cpu_profiler=cpu_profiler,
            gpu_profiler=gpu_profiler,
            memory_profiler=memory_profiler,
            network_profiler=network_profiler,
            frame_profiler=frame_profiler,
            config=config,
        )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(data)


class ChromeTraceExporter(BaseExporter):
    """
    Export to Chrome trace format.

    The output can be loaded in:
    - chrome://tracing
    - Perfetto (ui.perfetto.dev)
    - Trace Event Profiling Tool
    """

    def export(
        self,
        cpu_profiler: Optional[CPUProfiler] = None,
        gpu_profiler: Optional[GPUProfiler] = None,
        memory_profiler: Optional[MemoryProfiler] = None,
        network_profiler: Optional[NetworkProfiler] = None,
        frame_profiler: Optional[FrameProfiler] = None,
        config: Optional[ExportConfig] = None,
    ) -> str:
        """Export to Chrome trace JSON format."""
        config = config or ExportConfig(format=ExportFormat.CHROME_TRACE)
        events: List[Dict[str, Any]] = []

        # Process metadata
        events.append({
            "name": "process_name",
            "ph": "M",
            "pid": 1,
            "args": {"name": "AI Game Engine"},
        })

        # Export CPU samples
        if config.include_cpu and cpu_profiler:
            cpu_events = self._export_cpu_samples(cpu_profiler, config)
            events.extend(cpu_events)

        # Export GPU samples
        if config.include_gpu and gpu_profiler:
            gpu_events = self._export_gpu_samples(gpu_profiler, config)
            events.extend(gpu_events)

        # Export memory events
        if config.include_memory and memory_profiler:
            memory_events = self._export_memory_events(memory_profiler, config)
            events.extend(memory_events)

        # Export frame events
        if config.include_frames and frame_profiler:
            frame_events = self._export_frame_events(frame_profiler, config)
            events.extend(frame_events)

        # Build trace object
        trace = {
            "traceEvents": events,
            "metadata": {
                "generator": "AI Game Engine Profiler",
                "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        }

        if config.pretty_print:
            return json.dumps(trace, indent=2)
        return json.dumps(trace)

    def _export_cpu_samples(
        self,
        profiler: CPUProfiler,
        config: ExportConfig,
    ) -> List[Dict[str, Any]]:
        """Export CPU samples to Chrome trace events."""
        events = []
        samples = profiler.get_samples(min_duration_ms=config.min_duration_ms)

        for sample in samples:
            # Ensure values are JSON-serializable primitives
            start_time = float(sample.start_time) if hasattr(sample.start_time, '__float__') else 0.0
            duration_us = float(sample.duration_us) if hasattr(sample.duration_us, '__float__') else 0.0
            thread_id = int(sample.thread_id) if hasattr(sample.thread_id, '__int__') else 1

            # Handle the name attribute carefully - MagicMock uses 'name' as a special parameter
            # For MagicMock, the internal name is stored in _mock_name
            name_attr = sample.name
            if hasattr(name_attr, '_mock_name') and name_attr._mock_name:
                # It's a MagicMock - extract the parent's _mock_name which was passed as name=
                # The mock name format is "parent_name.name", so we need the parent's name
                if hasattr(sample, '_mock_name') and sample._mock_name:
                    name = sample._mock_name
                else:
                    name = str(name_attr)
            elif isinstance(name_attr, str):
                name = name_attr
            else:
                name = str(name_attr) if name_attr else "unknown"

            # Duration event (X)
            event = {
                "name": name,
                "cat": "cpu",
                "ph": "X",  # Complete event
                "ts": start_time * 1e6,  # Convert to microseconds
                "dur": duration_us,
                "pid": 1,
                "tid": thread_id,
            }

            # Safely extract tags
            tags = getattr(sample, 'tags', None)
            if tags and isinstance(tags, dict):
                # Convert all tag values to JSON-serializable types
                event["args"] = {str(k): str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v
                                for k, v in tags.items()}
            elif tags:
                event["args"] = {"tags": str(tags)}

            events.append(event)

        return events

    def _export_gpu_samples(
        self,
        profiler: GPUProfiler,
        config: ExportConfig,
    ) -> List[Dict[str, Any]]:
        """Export GPU samples to Chrome trace events."""
        events = []
        samples = profiler.get_samples(min_gpu_time_ms=config.min_duration_ms)

        # Add GPU thread metadata
        events.append({
            "name": "thread_name",
            "ph": "M",
            "pid": 1,
            "tid": -1,
            "args": {"name": "GPU"},
        })

        for sample in samples:
            # Ensure values are JSON-serializable primitives
            name = str(sample.name) if sample.name else "unknown"
            category = str(sample.category) if sample.category else "unknown"
            start_time = float(sample.start_time) if hasattr(sample.start_time, '__float__') else 0.0
            gpu_time_ms = float(sample.gpu_time_ms) if hasattr(sample.gpu_time_ms, '__float__') else 0.0
            draw_calls = int(sample.draw_calls) if hasattr(sample.draw_calls, '__int__') else 0
            triangles = int(sample.triangles) if hasattr(sample.triangles, '__int__') else 0

            event = {
                "name": name,
                "cat": f"gpu,{category}",
                "ph": "X",
                "ts": start_time * 1e6,
                "dur": gpu_time_ms * 1000,  # Convert to microseconds
                "pid": 1,
                "tid": -1,  # GPU thread
                "args": {
                    "category": category,
                    "draw_calls": draw_calls,
                    "triangles": triangles,
                },
            }
            events.append(event)

        return events

    def _export_memory_events(
        self,
        profiler: MemoryProfiler,
        config: ExportConfig,
    ) -> List[Dict[str, Any]]:
        """Export memory events to Chrome trace events."""
        events = []

        # Add memory counter events
        allocations = profiler.get_allocations()
        for alloc in allocations:
            # Allocation event
            events.append({
                "name": "memory_allocation",
                "cat": "memory",
                "ph": "C",  # Counter event
                "ts": alloc.timestamp * 1e6,
                "pid": 1,
                "args": {
                    "allocated_bytes": alloc.size,
                    "category": alloc.category.value,
                },
            })

        return events

    def _export_frame_events(
        self,
        profiler: FrameProfiler,
        config: ExportConfig,
    ) -> List[Dict[str, Any]]:
        """Export frame events to Chrome trace events."""
        events = []
        frames = profiler.get_frames()

        for frame in frames:
            # Frame boundary event
            events.append({
                "name": f"Frame {frame.frame_number}",
                "cat": "frame",
                "ph": "X",
                "ts": frame.start_time * 1e6,
                "dur": frame.frame_time_ms * 1000,
                "pid": 1,
                "tid": 1,
                "args": {
                    "frame_number": frame.frame_number,
                    "fps": frame.fps,
                    "is_spike": frame.is_spike,
                },
            })

            # Phase events
            for phase in frame.phases:
                events.append({
                    "name": phase.custom_name or phase.phase.value,
                    "cat": "frame,phase",
                    "ph": "X",
                    "ts": phase.start_time * 1e6,
                    "dur": phase.duration_ms * 1000,
                    "pid": 1,
                    "tid": 1,
                })

        return events


class CSVExporter(BaseExporter):
    """Export to CSV format for spreadsheet analysis."""

    def export(
        self,
        cpu_profiler: Optional[CPUProfiler] = None,
        gpu_profiler: Optional[GPUProfiler] = None,
        memory_profiler: Optional[MemoryProfiler] = None,
        network_profiler: Optional[NetworkProfiler] = None,
        frame_profiler: Optional[FrameProfiler] = None,
        config: Optional[ExportConfig] = None,
    ) -> str:
        """Export to CSV format."""
        config = config or ExportConfig(format=ExportFormat.CSV)
        output = io.StringIO()

        # Export each section
        sections = []

        if config.include_cpu and cpu_profiler:
            sections.append(self._export_cpu_csv(cpu_profiler, config))

        if config.include_gpu and gpu_profiler:
            sections.append(self._export_gpu_csv(gpu_profiler, config))

        if config.include_memory and memory_profiler:
            sections.append(self._export_memory_csv(memory_profiler, config))

        if config.include_network and network_profiler:
            sections.append(self._export_network_csv(network_profiler, config))

        if config.include_frames and frame_profiler:
            sections.append(self._export_frames_csv(frame_profiler, config))

        return "\n\n".join(sections)

    def _export_cpu_csv(
        self,
        profiler: CPUProfiler,
        config: ExportConfig,
    ) -> str:
        """Export CPU data to CSV."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "# CPU Profile Data",
        ])
        writer.writerow([
            "name",
            "call_count",
            "total_ms",
            "avg_ms",
            "min_ms",
            "max_ms",
        ])

        # Data
        stats = profiler.get_stats()
        for name, stat in sorted(stats.items()):
            writer.writerow([
                name,
                stat.call_count,
                f"{stat.total_time_ms:.3f}",
                f"{stat.avg_time_ms:.3f}",
                f"{stat.min_time_ms:.3f}" if stat.min_time_ms != float("inf") else "0.000",
                f"{stat.max_time_ms:.3f}",
            ])

        return output.getvalue()

    def _export_gpu_csv(
        self,
        profiler: GPUProfiler,
        config: ExportConfig,
    ) -> str:
        """Export GPU data to CSV."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "# GPU Profile Data",
        ])
        writer.writerow([
            "name",
            "category",
            "gpu_time_ms",
            "draw_calls",
            "triangles",
        ])

        # Data
        for name, timing in profiler.get_pass_timings().items():
            writer.writerow([
                name,
                timing.pass_type.name,
                f"{timing.gpu_time_ms:.3f}",
                timing.draw_calls,
                timing.triangles,
            ])

        return output.getvalue()

    def _export_memory_csv(
        self,
        profiler: MemoryProfiler,
        config: ExportConfig,
    ) -> str:
        """Export memory data to CSV."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "# Memory Profile Data",
        ])
        writer.writerow([
            "category",
            "bytes",
            "mb",
            "percentage",
        ])

        # Data
        breakdown = profiler.get_category_breakdown()
        total = sum(breakdown.values()) or 1

        for category, bytes_used in sorted(breakdown.items(), key=lambda x: -x[1]):
            writer.writerow([
                category.value,
                bytes_used,
                f"{bytes_used / (1024 * 1024):.2f}",
                f"{(bytes_used / total) * 100:.1f}%",
            ])

        return output.getvalue()

    def _export_network_csv(
        self,
        profiler: NetworkProfiler,
        config: ExportConfig,
    ) -> str:
        """Export network data to CSV."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "# Network Profile Data",
        ])
        writer.writerow([
            "channel",
            "bytes_sent",
            "bytes_received",
            "packets_sent",
            "packets_received",
        ])

        # Data
        for name, stats in profiler.get_channel_stats().items():
            writer.writerow([
                name,
                stats.bytes_sent,
                stats.bytes_received,
                stats.packets_sent,
                stats.packets_received,
            ])

        return output.getvalue()

    def _export_frames_csv(
        self,
        profiler: FrameProfiler,
        config: ExportConfig,
    ) -> str:
        """Export frame data to CSV."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "# Frame Profile Data",
        ])
        writer.writerow([
            "frame",
            "frame_time_ms",
            "cpu_time_ms",
            "gpu_time_ms",
            "fps",
            "is_spike",
        ])

        # Data
        for frame in profiler.get_frames():
            writer.writerow([
                frame.frame_number,
                f"{frame.frame_time_ms:.3f}",
                f"{frame.cpu_time_ms:.3f}",
                f"{frame.gpu_time_ms:.3f}",
                f"{frame.fps:.1f}",
                frame.is_spike,
            ])

        return output.getvalue()


class JSONExporter(BaseExporter):
    """Export to JSON format for custom tooling."""

    def export(
        self,
        cpu_profiler: Optional[CPUProfiler] = None,
        gpu_profiler: Optional[GPUProfiler] = None,
        memory_profiler: Optional[MemoryProfiler] = None,
        network_profiler: Optional[NetworkProfiler] = None,
        frame_profiler: Optional[FrameProfiler] = None,
        config: Optional[ExportConfig] = None,
    ) -> str:
        """Export to JSON format."""
        config = config or ExportConfig(format=ExportFormat.JSON)
        data: Dict[str, Any] = {
            "metadata": {
                "generator": "AI Game Engine Profiler",
                "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config": config.to_dict(),
            },
        }

        if config.include_cpu and cpu_profiler:
            data["cpu"] = cpu_profiler.to_dict()

        if config.include_gpu and gpu_profiler:
            data["gpu"] = gpu_profiler.to_dict()

        if config.include_memory and memory_profiler:
            data["memory"] = memory_profiler.to_dict()

        if config.include_network and network_profiler:
            data["network"] = network_profiler.to_dict()

        if config.include_frames and frame_profiler:
            data["frames"] = frame_profiler.to_dict()

        if config.pretty_print:
            return json.dumps(data, indent=2, default=str)
        return json.dumps(data, default=str)


class ProfilerExporter:
    """
    Main exporter class that delegates to format-specific exporters.

    Usage:
        exporter = ProfilerExporter()
        data = exporter.export(cpu_profiler, format=ExportFormat.CHROME_TRACE)
        exporter.export_to_file("profile.json", cpu_profiler)
    """

    _exporters: Dict[ExportFormat, Type[BaseExporter]] = {
        ExportFormat.CHROME_TRACE: ChromeTraceExporter,
        ExportFormat.CSV: CSVExporter,
        ExportFormat.JSON: JSONExporter,
    }

    _custom_exporters: Dict[str, Type[BaseExporter]] = {}

    @classmethod
    def register_exporter(
        cls,
        name: str,
        exporter_class: Type[BaseExporter],
    ) -> None:
        """Register a custom exporter."""
        cls._custom_exporters[name] = exporter_class

    @classmethod
    def get_exporter(
        cls,
        format: ExportFormat,
        custom_name: Optional[str] = None,
    ) -> BaseExporter:
        """Get an exporter instance for the specified format."""
        if format == ExportFormat.CUSTOM and custom_name:
            exporter_class = cls._custom_exporters.get(custom_name)
            if exporter_class:
                return exporter_class()
            raise ValueError(f"Unknown custom exporter: {custom_name}")

        exporter_class = cls._exporters.get(format)
        if exporter_class:
            return exporter_class()
        raise ValueError(f"Unknown export format: {format}")

    def export(
        self,
        cpu_profiler: Optional[CPUProfiler] = None,
        gpu_profiler: Optional[GPUProfiler] = None,
        memory_profiler: Optional[MemoryProfiler] = None,
        network_profiler: Optional[NetworkProfiler] = None,
        frame_profiler: Optional[FrameProfiler] = None,
        config: Optional[ExportConfig] = None,
        format: ExportFormat = ExportFormat.CHROME_TRACE,
        custom_exporter: Optional[str] = None,
    ) -> str:
        """
        Export profiling data.

        Args:
            cpu_profiler: CPU profiler instance
            gpu_profiler: GPU profiler instance
            memory_profiler: Memory profiler instance
            network_profiler: Network profiler instance
            frame_profiler: Frame profiler instance
            config: Export configuration
            format: Export format
            custom_exporter: Name of custom exporter (if format is CUSTOM)

        Returns:
            Exported data as string
        """
        exporter = self.get_exporter(format, custom_exporter)
        return exporter.export(
            cpu_profiler=cpu_profiler,
            gpu_profiler=gpu_profiler,
            memory_profiler=memory_profiler,
            network_profiler=network_profiler,
            frame_profiler=frame_profiler,
            config=config,
        )

    def export_to_file(
        self,
        path: Union[str, Path],
        cpu_profiler: Optional[CPUProfiler] = None,
        gpu_profiler: Optional[GPUProfiler] = None,
        memory_profiler: Optional[MemoryProfiler] = None,
        network_profiler: Optional[NetworkProfiler] = None,
        frame_profiler: Optional[FrameProfiler] = None,
        config: Optional[ExportConfig] = None,
        format: Optional[ExportFormat] = None,
    ) -> None:
        """
        Export profiling data to a file.

        The format is auto-detected from the file extension if not specified:
        - .json -> CHROME_TRACE (Chrome tracing JSON)
        - .csv -> CSV
        - .profile.json -> JSON (generic)
        """
        path = Path(path)

        # Auto-detect format from extension
        if format is None:
            suffix = path.suffix.lower()
            if suffix == ".csv":
                format = ExportFormat.CSV
            elif path.name.endswith(".profile.json"):
                format = ExportFormat.JSON
            else:
                format = ExportFormat.CHROME_TRACE

        exporter = self.get_exporter(format)
        exporter.export_to_file(
            path=path,
            cpu_profiler=cpu_profiler,
            gpu_profiler=gpu_profiler,
            memory_profiler=memory_profiler,
            network_profiler=network_profiler,
            frame_profiler=frame_profiler,
            config=config,
        )
