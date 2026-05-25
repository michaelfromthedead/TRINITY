"""
Comprehensive tests for AssetProcessor functionality.

Tests batch processing, compression, format conversion, and pipelines.
"""

import pytest
import sys
import tempfile
import shutil
import time
from pathlib import Path

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.tooling.assettools.asset_processor import (
    AssetProcessor,
    BatchProcessor,
    ProcessingTask,
    ProcessingResult,
    ProcessingPipeline,
    ProcessingStatus,
    CompressionFormat,
    AudioFormat,
    CompressionSettings,
    FormatConversionSettings,
    AudioConversionSettings,
    MeshOptimizationSettings,
)


@pytest.fixture
def temp_processor_dir():
    """Create a temporary directory for processor tests."""
    path = Path(tempfile.mkdtemp())

    (path / "source").mkdir()
    (path / "output").mkdir()

    # Create test files
    (path / "source" / "texture.png").write_bytes(b"png data")
    (path / "source" / "texture2.png").write_bytes(b"png data 2")
    (path / "source" / "model.fbx").write_bytes(b"fbx data")
    (path / "source" / "audio.wav").write_bytes(b"wav data")
    (path / "source" / "audio2.ogg").write_bytes(b"ogg data")

    yield path
    shutil.rmtree(path)


class TestProcessingStatus:
    """Test ProcessingStatus enum."""

    def test_status_values(self):
        """All status values should be defined."""
        assert ProcessingStatus.PENDING
        assert ProcessingStatus.QUEUED
        assert ProcessingStatus.IN_PROGRESS
        assert ProcessingStatus.COMPLETED
        assert ProcessingStatus.FAILED
        assert ProcessingStatus.CANCELLED


class TestCompressionSettings:
    """Test CompressionSettings dataclass."""

    def test_default_settings(self):
        """Default settings should be sensible."""
        settings = CompressionSettings()

        assert settings.format == CompressionFormat.BC7
        assert settings.quality == 0.8
        assert settings.generate_mipmaps is True

    def test_custom_settings(self):
        """Custom settings should be stored."""
        settings = CompressionSettings(
            format=CompressionFormat.BC3,
            quality=0.5,
            max_size=2048,
        )

        assert settings.format == CompressionFormat.BC3
        assert settings.quality == 0.5
        assert settings.max_size == 2048


class TestFormatConversionSettings:
    """Test FormatConversionSettings dataclass."""

    def test_default_settings(self):
        """Default settings should be sensible."""
        settings = FormatConversionSettings()

        assert settings.preserve_metadata is True
        assert settings.overwrite_existing is False

    def test_custom_settings(self):
        """Custom settings should be stored."""
        settings = FormatConversionSettings(
            output_format="dds",
            delete_original=True,
        )

        assert settings.output_format == "dds"
        assert settings.delete_original is True


class TestAudioConversionSettings:
    """Test AudioConversionSettings dataclass."""

    def test_default_settings(self):
        """Default settings should be sensible."""
        settings = AudioConversionSettings()

        assert settings.format == AudioFormat.OGG
        assert settings.sample_rate == 44100
        assert settings.channels == 2

    def test_custom_settings(self):
        """Custom settings should be stored."""
        settings = AudioConversionSettings(
            format=AudioFormat.WAV,
            channels=1,
            normalize=True,
        )

        assert settings.format == AudioFormat.WAV
        assert settings.channels == 1
        assert settings.normalize is True


class TestMeshOptimizationSettings:
    """Test MeshOptimizationSettings dataclass."""

    def test_default_settings(self):
        """Default settings should be sensible."""
        settings = MeshOptimizationSettings()

        assert settings.optimize_vertex_cache is True
        assert settings.remove_degenerate is True

    def test_custom_settings(self):
        """Custom settings should be stored."""
        settings = MeshOptimizationSettings(
            simplify=True,
            target_triangles=10000,
        )

        assert settings.simplify is True
        assert settings.target_triangles == 10000


class TestProcessingTask:
    """Test ProcessingTask dataclass."""

    def test_task_creation(self):
        """Task should store all attributes."""
        task = ProcessingTask(
            id="task_1",
            source_path=Path("/source.png"),
            output_path=Path("/output.dds"),
            operation="compress",
        )

        assert task.id == "task_1"
        assert task.source_path == Path("/source.png")
        assert task.operation == "compress"
        assert task.status == ProcessingStatus.PENDING

    def test_duration_ms(self):
        """duration_ms should calculate elapsed time."""
        task = ProcessingTask(
            id="task_1",
            source_path=Path("/s.png"),
            output_path=Path("/o.png"),
            operation="test",
        )

        task.started_at = time.time() - 1.0  # 1 second ago
        task.completed_at = time.time()

        assert task.duration_ms >= 1000


class TestProcessingResult:
    """Test ProcessingResult dataclass."""

    def test_result_creation(self):
        """Result should store task and outcome."""
        task = ProcessingTask(
            id="task_1",
            source_path=Path("/s.png"),
            output_path=Path("/o.png"),
            operation="test",
        )

        result = ProcessingResult(
            task=task,
            success=True,
            output_paths=[Path("/o.png")],
        )

        assert result.success is True
        assert len(result.output_paths) == 1


class TestProcessingPipeline:
    """Test ProcessingPipeline functionality."""

    def test_pipeline_creation(self):
        """Pipeline should initialize empty."""
        pipeline = ProcessingPipeline(name="test")

        assert pipeline.name == "test"
        assert len(pipeline.steps) == 0

    def test_add_step(self):
        """add_step() should add steps."""
        pipeline = ProcessingPipeline()

        pipeline.add_step("compress", {"quality": 0.8})
        pipeline.add_step("convert", {"format": "dds"})

        assert len(pipeline.steps) == 2

    def test_chained_methods(self):
        """Pipeline methods should support chaining."""
        pipeline = (
            ProcessingPipeline()
            .compress(CompressionSettings(quality=0.5))
            .resize(max_size=1024)
        )

        assert len(pipeline.steps) == 2

    def test_audio_pipeline(self):
        """Pipeline should support audio operations."""
        pipeline = (
            ProcessingPipeline()
            .convert_audio(AudioConversionSettings(channels=1))
        )

        assert len(pipeline.steps) == 1

    def test_mesh_pipeline(self):
        """Pipeline should support mesh operations."""
        pipeline = (
            ProcessingPipeline()
            .optimize_mesh(MeshOptimizationSettings(simplify=True))
        )

        assert len(pipeline.steps) == 1


class TestAssetProcessor:
    """Test AssetProcessor main class."""

    def test_processor_creation(self, temp_processor_dir):
        """Processor should initialize correctly."""
        processor = AssetProcessor(output_directory=temp_processor_dir / "output")

        assert processor.output_directory == temp_processor_dir / "output"
        assert processor.max_workers == 4

    def test_process_compress(self, temp_processor_dir):
        """process() should compress textures."""
        processor = AssetProcessor(output_directory=temp_processor_dir / "output")
        source = temp_processor_dir / "source" / "texture.png"

        result = processor.process(
            source,
            operation="compress",
            settings=CompressionSettings(),
        )

        assert result.success
        assert len(result.output_paths) == 1

    def test_process_convert(self, temp_processor_dir):
        """process() should convert formats."""
        processor = AssetProcessor(output_directory=temp_processor_dir / "output")
        source = temp_processor_dir / "source" / "texture.png"

        result = processor.process(
            source,
            operation="convert",
            settings=FormatConversionSettings(output_format="jpg"),
        )

        assert result.success

    def test_process_resize(self, temp_processor_dir):
        """process() should resize textures."""
        processor = AssetProcessor(output_directory=temp_processor_dir / "output")
        source = temp_processor_dir / "source" / "texture.png"

        result = processor.process(
            source,
            operation="resize",
            settings={"max_size": 512},
        )

        assert result.success

    def test_process_optimize_mesh(self, temp_processor_dir):
        """process() should optimize meshes."""
        processor = AssetProcessor(output_directory=temp_processor_dir / "output")
        source = temp_processor_dir / "source" / "model.fbx"

        result = processor.process(
            source,
            operation="optimize_mesh",
            settings=MeshOptimizationSettings(),
        )

        assert result.success

    def test_process_convert_audio(self, temp_processor_dir):
        """process() should convert audio."""
        processor = AssetProcessor(output_directory=temp_processor_dir / "output")
        source = temp_processor_dir / "source" / "audio.wav"

        result = processor.process(
            source,
            operation="convert_audio",
            settings=AudioConversionSettings(format=AudioFormat.OGG),
        )

        assert result.success

    def test_process_copy(self, temp_processor_dir):
        """process() should copy files."""
        processor = AssetProcessor(output_directory=temp_processor_dir / "output")
        source = temp_processor_dir / "source" / "texture.png"

        result = processor.process(source, operation="copy")

        assert result.success
        assert result.output_paths[0].exists()

    def test_process_missing_file(self, temp_processor_dir):
        """process() should fail for missing files."""
        processor = AssetProcessor(output_directory=temp_processor_dir / "output")

        result = processor.process(
            temp_processor_dir / "missing.png",
            operation="copy",
        )

        assert not result.success
        assert result.task.status == ProcessingStatus.FAILED

    def test_process_unknown_operation(self, temp_processor_dir):
        """process() should fail for unknown operations."""
        processor = AssetProcessor(output_directory=temp_processor_dir / "output")
        source = temp_processor_dir / "source" / "texture.png"

        result = processor.process(source, operation="unknown_op")

        assert not result.success

    def test_process_async(self, temp_processor_dir):
        """process_async() should return task ID."""
        processor = AssetProcessor(output_directory=temp_processor_dir / "output")
        source = temp_processor_dir / "source" / "texture.png"

        task_id = processor.process_async(source, operation="copy")

        assert task_id is not None
        assert task_id.startswith("task_")

        # Wait for completion
        result = processor.get_task_result(task_id, timeout=5.0)
        assert result is not None
        assert result.success

        processor.shutdown()

    def test_process_batch_parallel(self, temp_processor_dir):
        """process_batch() should process files in parallel."""
        processor = AssetProcessor(output_directory=temp_processor_dir / "output")

        items = [
            (temp_processor_dir / "source" / "texture.png", "copy", None),
            (temp_processor_dir / "source" / "texture2.png", "copy", None),
            (temp_processor_dir / "source" / "model.fbx", "copy", None),
        ]

        results = processor.process_batch(items, parallel=True)

        assert len(results) == 3
        assert all(r.success for r in results)

        processor.shutdown()

    def test_process_batch_sequential(self, temp_processor_dir):
        """process_batch() should process sequentially when specified."""
        processor = AssetProcessor(output_directory=temp_processor_dir / "output")

        items = [
            (temp_processor_dir / "source" / "texture.png", "copy", None),
            (temp_processor_dir / "source" / "texture2.png", "copy", None),
        ]

        results = processor.process_batch(items, parallel=False)

        assert len(results) == 2
        assert all(r.success for r in results)

    def test_process_pipeline(self, temp_processor_dir):
        """process_pipeline() should run multi-step pipeline."""
        processor = AssetProcessor(output_directory=temp_processor_dir / "output")
        source = temp_processor_dir / "source" / "texture.png"

        pipeline = (
            ProcessingPipeline()
            .add_step("copy", None)
        )

        result = processor.process_pipeline(source, pipeline)

        assert result.success

    def test_get_task_status(self, temp_processor_dir):
        """get_task_status() should return task state."""
        processor = AssetProcessor(output_directory=temp_processor_dir / "output")
        source = temp_processor_dir / "source" / "texture.png"

        task_id = processor.process_async(source, operation="copy")

        # Wait a bit for task to complete
        time.sleep(0.5)

        task = processor.get_task_status(task_id)

        assert task is not None
        assert task.id == task_id

        processor.shutdown()

    def test_cancel_task(self, temp_processor_dir):
        """cancel_task() should request cancellation."""
        processor = AssetProcessor(output_directory=temp_processor_dir / "output")
        source = temp_processor_dir / "source" / "texture.png"

        # Create task but cancel immediately
        task_id = processor.process_async(source, operation="copy")

        success = processor.cancel_task(task_id)

        assert success

        processor.shutdown()

    def test_cancel_all(self, temp_processor_dir):
        """cancel_all() should cancel all pending tasks."""
        processor = AssetProcessor(output_directory=temp_processor_dir / "output")

        count = processor.cancel_all()

        assert count >= 0

        processor.shutdown()

    def test_progress_callback(self, temp_processor_dir):
        """Progress callback should be called."""
        processor = AssetProcessor(output_directory=temp_processor_dir / "output")
        source = temp_processor_dir / "source" / "texture.png"
        progress_updates = []

        processor.on_progress(lambda task: progress_updates.append(task))

        result = processor.process(source, operation="copy")

        assert len(progress_updates) > 0
        assert result.success

    def test_get_stats(self, temp_processor_dir):
        """get_stats() should return statistics."""
        processor = AssetProcessor(output_directory=temp_processor_dir / "output")
        source = temp_processor_dir / "source" / "texture.png"

        processor.process(source, operation="copy")

        stats = processor.get_stats()

        assert "total_tasks" in stats
        assert stats["total_tasks"] >= 1

    def test_shutdown(self, temp_processor_dir):
        """shutdown() should cleanup executor."""
        processor = AssetProcessor(output_directory=temp_processor_dir / "output")
        source = temp_processor_dir / "source" / "texture.png"

        processor.process_async(source, operation="copy")
        processor.shutdown(wait=True)

        # Should not raise
        assert processor._executor is None


class TestBatchProcessor:
    """Test BatchProcessor functionality."""

    def test_batch_processor_creation(self, temp_processor_dir):
        """BatchProcessor should initialize correctly."""
        batch = BatchProcessor()

        assert batch.processor is not None

    def test_process_directory(self, temp_processor_dir):
        """process_directory() should process all files."""
        batch = BatchProcessor()

        results = batch.process_directory(
            temp_processor_dir / "source",
            operation="copy",
            output_directory=temp_processor_dir / "output",
            filter_extensions={"png"},
        )

        # Should process PNG files
        assert len(results) >= 2

        batch.processor.shutdown()

    def test_process_directory_recursive(self, temp_processor_dir):
        """process_directory() should handle recursion."""
        # Create subdirectory
        (temp_processor_dir / "source" / "sub").mkdir()
        (temp_processor_dir / "source" / "sub" / "nested.png").write_bytes(b"data")

        batch = BatchProcessor()

        results = batch.process_directory(
            temp_processor_dir / "source",
            operation="copy",
            recursive=True,
            filter_extensions={"png"},
        )

        assert len(results) >= 3  # 2 PNG + 1 nested

        batch.processor.shutdown()

    def test_generate_report(self, temp_processor_dir):
        """generate_report() should return processing report."""
        batch = BatchProcessor()

        batch.process_directory(
            temp_processor_dir / "source",
            operation="copy",
            filter_extensions={"png"},
        )

        report = batch.generate_report()

        assert "total_processed" in report
        assert "success_count" in report
        assert "by_operation" in report

        batch.processor.shutdown()

    def test_clear_results(self, temp_processor_dir):
        """clear_results() should clear stored results."""
        batch = BatchProcessor()

        batch.process_directory(
            temp_processor_dir / "source",
            operation="copy",
            filter_extensions={"png"},
        )

        batch.clear_results()

        report = batch.generate_report()
        assert report["total_processed"] == 0

        batch.processor.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
