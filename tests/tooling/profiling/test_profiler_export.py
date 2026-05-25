"""Tests for the profiler export module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from engine.tooling.profiling.profiler_export import (
    ExportFormat,
    ExportConfig,
    BaseExporter,
    ChromeTraceExporter,
    CSVExporter,
    JSONExporter,
    ProfilerExporter,
)
from engine.tooling.profiling.cpu_profiler import CPUProfiler, CPUProfileSample, ProfilerStats
from engine.tooling.profiling.gpu_profiler import GPUProfiler, GPUProfileSample, RenderPassType
from engine.tooling.profiling.memory_profiler import MemoryProfiler, AllocationRecord, MemoryCategory
from engine.tooling.profiling.network_profiler import NetworkProfiler, ChannelStats
from engine.tooling.profiling.frame_profiler import FrameProfiler, FrameData, PhaseTimestamp, FramePhase


class TestExportFormat:
    """Tests for ExportFormat enum."""

    def test_chrome_trace_format(self):
        """Test Chrome trace format exists."""
        assert ExportFormat.CHROME_TRACE is not None
        assert ExportFormat.CHROME_TRACE.name == "CHROME_TRACE"

    def test_csv_format(self):
        """Test CSV format exists."""
        assert ExportFormat.CSV is not None
        assert ExportFormat.CSV.name == "CSV"

    def test_json_format(self):
        """Test JSON format exists."""
        assert ExportFormat.JSON is not None
        assert ExportFormat.JSON.name == "JSON"

    def test_custom_format(self):
        """Test custom format exists."""
        assert ExportFormat.CUSTOM is not None
        assert ExportFormat.CUSTOM.name == "CUSTOM"


class TestExportConfig:
    """Tests for ExportConfig."""

    def test_default_creation(self):
        """Test default configuration."""
        config = ExportConfig()
        assert config.format == ExportFormat.CHROME_TRACE
        assert config.include_cpu is True
        assert config.include_gpu is True
        assert config.include_memory is True
        assert config.include_network is False
        assert config.include_frames is True
        assert config.min_duration_ms == 0.0
        assert config.compress is False
        assert config.pretty_print is True

    def test_custom_creation(self):
        """Test custom configuration."""
        config = ExportConfig(
            format=ExportFormat.CSV,
            include_cpu=False,
            include_gpu=True,
            include_memory=False,
            include_network=True,
            include_frames=False,
            min_duration_ms=1.5,
            compress=True,
            pretty_print=False,
        )
        assert config.format == ExportFormat.CSV
        assert config.include_cpu is False
        assert config.include_network is True
        assert config.min_duration_ms == 1.5

    def test_to_dict(self):
        """Test dictionary conversion."""
        config = ExportConfig(
            format=ExportFormat.JSON,
            include_call_tree=True,
            min_duration_ms=0.5,
        )
        data = config.to_dict()

        assert data["format"] == "JSON"
        assert data["include_call_tree"] is True
        assert data["min_duration_ms"] == 0.5
        assert "include_cpu" in data
        assert "include_gpu" in data


class TestChromeTraceExporter:
    """Tests for ChromeTraceExporter."""

    @pytest.fixture
    def exporter(self):
        """Create a fresh exporter instance."""
        return ChromeTraceExporter()

    @pytest.fixture
    def mock_cpu_profiler(self):
        """Create a mock CPU profiler."""
        profiler = MagicMock(spec=CPUProfiler)
        profiler.get_samples.return_value = [
            MagicMock(
                name="test_function",
                start_time=0.001,
                duration_us=1500.0,
                thread_id=1,
                tags={"category": "gameplay"},
            ),
            MagicMock(
                name="update_loop",
                start_time=0.002,
                duration_us=2000.0,
                thread_id=1,
                tags={},
            ),
        ]
        return profiler

    @pytest.fixture
    def mock_gpu_profiler(self):
        """Create a mock GPU profiler."""
        profiler = MagicMock(spec=GPUProfiler)
        profiler.get_samples.return_value = [
            MagicMock(
                name="shadow_pass",
                category="shadows",
                start_time=0.001,
                gpu_time_ms=2.5,
                draw_calls=100,
                triangles=50000,
            ),
        ]
        return profiler

    @pytest.fixture
    def mock_memory_profiler(self):
        """Create a mock memory profiler."""
        profiler = MagicMock(spec=MemoryProfiler)
        profiler.get_allocations.return_value = [
            MagicMock(
                timestamp=0.001,
                size=1024,
                category=MagicMock(value="textures"),
            ),
        ]
        return profiler

    @pytest.fixture
    def mock_frame_profiler(self):
        """Create a mock frame profiler."""
        profiler = MagicMock(spec=FrameProfiler)
        profiler.get_frames.return_value = [
            MagicMock(
                frame_number=1,
                start_time=0.0,
                frame_time_ms=16.67,
                fps=60.0,
                is_spike=False,
                phases=[
                    MagicMock(
                        phase=MagicMock(value="update"),
                        custom_name=None,
                        start_time=0.0,
                        duration_ms=5.0,
                    ),
                ],
            ),
        ]
        return profiler

    def test_export_empty(self, exporter):
        """Test export with no data."""
        result = exporter.export()
        data = json.loads(result)

        assert "traceEvents" in data
        assert "metadata" in data
        assert data["metadata"]["generator"] == "AI Game Engine Profiler"

    def test_export_with_cpu_profiler(self, exporter, mock_cpu_profiler):
        """Test export with CPU profiler."""
        config = ExportConfig(include_cpu=True, include_gpu=False, include_memory=False, include_frames=False)
        result = exporter.export(cpu_profiler=mock_cpu_profiler, config=config)
        data = json.loads(result)

        # Find CPU events
        cpu_events = [e for e in data["traceEvents"] if e.get("cat") == "cpu"]
        assert len(cpu_events) == 2
        assert cpu_events[0]["name"] == "test_function"
        assert cpu_events[0]["ph"] == "X"  # Complete event

    def test_export_with_gpu_profiler(self, exporter, mock_gpu_profiler):
        """Test export with GPU profiler."""
        config = ExportConfig(include_cpu=False, include_gpu=True, include_memory=False, include_frames=False)
        result = exporter.export(gpu_profiler=mock_gpu_profiler, config=config)
        data = json.loads(result)

        # Find GPU events
        gpu_events = [e for e in data["traceEvents"] if "gpu" in str(e.get("cat", ""))]
        assert len(gpu_events) >= 1

    def test_export_with_memory_profiler(self, exporter, mock_memory_profiler):
        """Test export with memory profiler."""
        config = ExportConfig(include_cpu=False, include_gpu=False, include_memory=True, include_frames=False)
        result = exporter.export(memory_profiler=mock_memory_profiler, config=config)
        data = json.loads(result)

        # Find memory events
        memory_events = [e for e in data["traceEvents"] if e.get("cat") == "memory"]
        assert len(memory_events) >= 1

    def test_export_with_frame_profiler(self, exporter, mock_frame_profiler):
        """Test export with frame profiler."""
        config = ExportConfig(include_cpu=False, include_gpu=False, include_memory=False, include_frames=True)
        result = exporter.export(frame_profiler=mock_frame_profiler, config=config)
        data = json.loads(result)

        # Find frame events
        frame_events = [e for e in data["traceEvents"] if "frame" in str(e.get("cat", ""))]
        assert len(frame_events) >= 1

    def test_export_pretty_print(self, exporter):
        """Test pretty print option."""
        config = ExportConfig(pretty_print=True)
        result = exporter.export(config=config)
        assert "\n" in result
        assert "  " in result  # Indentation

    def test_export_compact(self, exporter):
        """Test compact output."""
        config = ExportConfig(pretty_print=False)
        result = exporter.export(config=config)
        # Compact JSON should be on fewer lines
        assert result.count("\n") < 5

    def test_export_to_file(self, exporter):
        """Test export to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.json"
            exporter.export_to_file(path)

            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert "traceEvents" in data


class TestCSVExporter:
    """Tests for CSVExporter."""

    @pytest.fixture
    def exporter(self):
        """Create a fresh exporter instance."""
        return CSVExporter()

    @pytest.fixture
    def mock_cpu_profiler(self):
        """Create a mock CPU profiler with stats."""
        profiler = MagicMock(spec=CPUProfiler)
        profiler.get_stats.return_value = {
            "test_func": MagicMock(
                call_count=10,
                total_time_ms=50.0,
                avg_time_ms=5.0,
                min_time_ms=2.0,
                max_time_ms=10.0,
            ),
        }
        return profiler

    @pytest.fixture
    def mock_gpu_profiler(self):
        """Create a mock GPU profiler."""
        profiler = MagicMock(spec=GPUProfiler)
        profiler.get_pass_timings.return_value = {
            "main_pass": MagicMock(
                pass_type=MagicMock(name="FORWARD"),
                gpu_time_ms=5.0,
                draw_calls=200,
                triangles=100000,
            ),
        }
        return profiler

    @pytest.fixture
    def mock_memory_profiler(self):
        """Create a mock memory profiler."""
        profiler = MagicMock(spec=MemoryProfiler)
        profiler.get_category_breakdown.return_value = {
            MagicMock(value="textures"): 1024 * 1024,
            MagicMock(value="meshes"): 512 * 1024,
        }
        return profiler

    @pytest.fixture
    def mock_network_profiler(self):
        """Create a mock network profiler."""
        profiler = MagicMock(spec=NetworkProfiler)
        profiler.get_channel_stats.return_value = {
            "reliable": MagicMock(
                bytes_sent=1000,
                bytes_received=2000,
                packets_sent=10,
                packets_received=20,
            ),
        }
        return profiler

    @pytest.fixture
    def mock_frame_profiler(self):
        """Create a mock frame profiler."""
        profiler = MagicMock(spec=FrameProfiler)
        profiler.get_frames.return_value = [
            MagicMock(
                frame_number=1,
                frame_time_ms=16.67,
                cpu_time_ms=10.0,
                gpu_time_ms=8.0,
                fps=60.0,
                is_spike=False,
            ),
        ]
        return profiler

    def test_export_empty(self, exporter):
        """Test export with no data."""
        result = exporter.export()
        assert result == ""

    def test_export_cpu_csv(self, exporter, mock_cpu_profiler):
        """Test CPU data export to CSV."""
        config = ExportConfig(include_cpu=True, include_gpu=False, include_memory=False, include_frames=False)
        result = exporter.export(cpu_profiler=mock_cpu_profiler, config=config)

        assert "# CPU Profile Data" in result
        assert "name" in result
        assert "call_count" in result
        assert "test_func" in result

    def test_export_gpu_csv(self, exporter, mock_gpu_profiler):
        """Test GPU data export to CSV."""
        config = ExportConfig(include_cpu=False, include_gpu=True, include_memory=False, include_frames=False)
        result = exporter.export(gpu_profiler=mock_gpu_profiler, config=config)

        assert "# GPU Profile Data" in result
        assert "main_pass" in result
        assert "draw_calls" in result

    def test_export_memory_csv(self, exporter, mock_memory_profiler):
        """Test memory data export to CSV."""
        config = ExportConfig(include_cpu=False, include_gpu=False, include_memory=True, include_frames=False)
        result = exporter.export(memory_profiler=mock_memory_profiler, config=config)

        assert "# Memory Profile Data" in result
        assert "category" in result
        assert "bytes" in result

    def test_export_network_csv(self, exporter, mock_network_profiler):
        """Test network data export to CSV."""
        config = ExportConfig(include_cpu=False, include_gpu=False, include_memory=False, include_network=True, include_frames=False)
        result = exporter.export(network_profiler=mock_network_profiler, config=config)

        assert "# Network Profile Data" in result
        assert "channel" in result
        assert "reliable" in result

    def test_export_frames_csv(self, exporter, mock_frame_profiler):
        """Test frame data export to CSV."""
        config = ExportConfig(include_cpu=False, include_gpu=False, include_memory=False, include_frames=True)
        result = exporter.export(frame_profiler=mock_frame_profiler, config=config)

        assert "# Frame Profile Data" in result
        assert "frame_time_ms" in result
        assert "fps" in result

    def test_export_multiple_sections(self, exporter, mock_cpu_profiler, mock_frame_profiler):
        """Test exporting multiple sections."""
        config = ExportConfig(include_cpu=True, include_gpu=False, include_memory=False, include_frames=True)
        result = exporter.export(
            cpu_profiler=mock_cpu_profiler,
            frame_profiler=mock_frame_profiler,
            config=config,
        )

        assert "# CPU Profile Data" in result
        assert "# Frame Profile Data" in result


class TestJSONExporter:
    """Tests for JSONExporter."""

    @pytest.fixture
    def exporter(self):
        """Create a fresh exporter instance."""
        return JSONExporter()

    @pytest.fixture
    def mock_cpu_profiler(self):
        """Create a mock CPU profiler."""
        profiler = MagicMock(spec=CPUProfiler)
        profiler.to_dict.return_value = {
            "enabled": True,
            "stats": {"test": {"call_count": 5}},
        }
        return profiler

    @pytest.fixture
    def mock_gpu_profiler(self):
        """Create a mock GPU profiler."""
        profiler = MagicMock(spec=GPUProfiler)
        profiler.to_dict.return_value = {
            "enabled": True,
            "pass_timings": {"shadow": {"gpu_time_ms": 2.0}},
        }
        return profiler

    def test_export_empty(self, exporter):
        """Test export with no data."""
        result = exporter.export()
        data = json.loads(result)

        assert "metadata" in data
        assert data["metadata"]["generator"] == "AI Game Engine Profiler"
        assert "exported_at" in data["metadata"]

    def test_export_with_cpu(self, exporter, mock_cpu_profiler):
        """Test export with CPU profiler."""
        config = ExportConfig(include_cpu=True, include_gpu=False, include_memory=False, include_frames=False)
        result = exporter.export(cpu_profiler=mock_cpu_profiler, config=config)
        data = json.loads(result)

        assert "cpu" in data
        assert data["cpu"]["enabled"] is True

    def test_export_with_gpu(self, exporter, mock_gpu_profiler):
        """Test export with GPU profiler."""
        config = ExportConfig(include_cpu=False, include_gpu=True, include_memory=False, include_frames=False)
        result = exporter.export(gpu_profiler=mock_gpu_profiler, config=config)
        data = json.loads(result)

        assert "gpu" in data
        assert data["gpu"]["enabled"] is True

    def test_config_in_metadata(self, exporter):
        """Test that config is included in metadata."""
        config = ExportConfig(
            format=ExportFormat.JSON,
            min_duration_ms=1.0,
        )
        result = exporter.export(config=config)
        data = json.loads(result)

        assert "config" in data["metadata"]
        assert data["metadata"]["config"]["format"] == "JSON"

    def test_pretty_print(self, exporter):
        """Test pretty print option."""
        config = ExportConfig(pretty_print=True)
        result = exporter.export(config=config)
        assert "\n" in result
        assert "  " in result

    def test_compact(self, exporter):
        """Test compact output."""
        config = ExportConfig(pretty_print=False)
        result = exporter.export(config=config)
        assert result.count("\n") < 2


class TestProfilerExporter:
    """Tests for ProfilerExporter factory class."""

    @pytest.fixture
    def exporter(self):
        """Create a fresh exporter instance."""
        return ProfilerExporter()

    def test_get_chrome_trace_exporter(self):
        """Test getting Chrome trace exporter."""
        exporter = ProfilerExporter.get_exporter(ExportFormat.CHROME_TRACE)
        assert isinstance(exporter, ChromeTraceExporter)

    def test_get_csv_exporter(self):
        """Test getting CSV exporter."""
        exporter = ProfilerExporter.get_exporter(ExportFormat.CSV)
        assert isinstance(exporter, CSVExporter)

    def test_get_json_exporter(self):
        """Test getting JSON exporter."""
        exporter = ProfilerExporter.get_exporter(ExportFormat.JSON)
        assert isinstance(exporter, JSONExporter)

    def test_register_custom_exporter(self):
        """Test registering a custom exporter."""

        class CustomExporter(BaseExporter):
            def export(self, **kwargs):
                return "custom output"

        ProfilerExporter.register_exporter("my_custom", CustomExporter)
        exporter = ProfilerExporter.get_exporter(ExportFormat.CUSTOM, "my_custom")
        assert isinstance(exporter, CustomExporter)

    def test_unknown_custom_exporter(self):
        """Test error on unknown custom exporter."""
        with pytest.raises(ValueError, match="Unknown custom exporter"):
            ProfilerExporter.get_exporter(ExportFormat.CUSTOM, "nonexistent")

    def test_export_chrome_trace(self, exporter):
        """Test export to Chrome trace format."""
        result = exporter.export(format=ExportFormat.CHROME_TRACE)
        data = json.loads(result)
        assert "traceEvents" in data

    def test_export_csv(self, exporter):
        """Test export to CSV format."""
        result = exporter.export(format=ExportFormat.CSV)
        # Empty CSV returns empty string
        assert isinstance(result, str)

    def test_export_json(self, exporter):
        """Test export to JSON format."""
        result = exporter.export(format=ExportFormat.JSON)
        data = json.loads(result)
        assert "metadata" in data

    def test_export_to_file_auto_detect_json(self, exporter):
        """Test auto-detection of JSON format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profile.json"
            exporter.export_to_file(path)

            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            # Chrome trace format is default for .json
            assert "traceEvents" in data

    def test_export_to_file_auto_detect_csv(self, exporter):
        """Test auto-detection of CSV format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profile.csv"
            exporter.export_to_file(path)

            assert path.exists()
            content = path.read_text()
            assert isinstance(content, str)

    def test_export_to_file_auto_detect_profile_json(self, exporter):
        """Test auto-detection of generic JSON format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.profile.json"
            exporter.export_to_file(path)

            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            # Generic JSON format has metadata
            assert "metadata" in data

    def test_export_to_file_explicit_format(self, exporter):
        """Test explicit format specification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "output.txt"
            exporter.export_to_file(path, format=ExportFormat.CSV)

            assert path.exists()

    def test_export_to_file_creates_parent_dirs(self, exporter):
        """Test that parent directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dirs" / "profile.json"
            exporter.export_to_file(path)

            assert path.exists()


class TestExportIntegration:
    """Integration tests for export functionality."""

    def test_full_export_pipeline(self):
        """Test complete export pipeline with all profilers."""
        # Create mock profilers
        cpu_profiler = MagicMock(spec=CPUProfiler)
        cpu_profiler.get_samples.return_value = []
        cpu_profiler.get_stats.return_value = {}
        cpu_profiler.to_dict.return_value = {"enabled": True}

        gpu_profiler = MagicMock(spec=GPUProfiler)
        gpu_profiler.get_samples.return_value = []
        gpu_profiler.get_pass_timings.return_value = {}
        gpu_profiler.to_dict.return_value = {"enabled": True}

        memory_profiler = MagicMock(spec=MemoryProfiler)
        memory_profiler.get_allocations.return_value = []
        memory_profiler.get_category_breakdown.return_value = {}
        memory_profiler.to_dict.return_value = {"enabled": True}

        frame_profiler = MagicMock(spec=FrameProfiler)
        frame_profiler.get_frames.return_value = []
        frame_profiler.to_dict.return_value = {"enabled": True}

        # Export to all formats
        exporter = ProfilerExporter()

        # Chrome trace
        chrome_result = exporter.export(
            cpu_profiler=cpu_profiler,
            gpu_profiler=gpu_profiler,
            format=ExportFormat.CHROME_TRACE,
        )
        assert "traceEvents" in chrome_result

        # JSON
        json_result = exporter.export(
            cpu_profiler=cpu_profiler,
            memory_profiler=memory_profiler,
            frame_profiler=frame_profiler,
            format=ExportFormat.JSON,
        )
        data = json.loads(json_result)
        assert "cpu" in data
        assert "memory" in data
        assert "frames" in data

    def test_config_filtering(self):
        """Test that config filtering works correctly."""
        cpu_profiler = MagicMock(spec=CPUProfiler)
        cpu_profiler.to_dict.return_value = {"enabled": True}

        gpu_profiler = MagicMock(spec=GPUProfiler)
        gpu_profiler.to_dict.return_value = {"enabled": True}

        config = ExportConfig(
            include_cpu=True,
            include_gpu=False,
        )

        exporter = JSONExporter()
        result = exporter.export(
            cpu_profiler=cpu_profiler,
            gpu_profiler=gpu_profiler,
            config=config,
        )
        data = json.loads(result)

        assert "cpu" in data
        assert "gpu" not in data
