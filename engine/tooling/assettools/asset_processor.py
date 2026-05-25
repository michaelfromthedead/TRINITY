"""
AssetProcessor - Batch processing, compression, and format conversion.

Provides asset processing capabilities:
- Batch processing with progress tracking
- Texture compression and resizing
- Audio format conversion
- Mesh optimization
- Pipeline-based processing
"""

from __future__ import annotations

import hashlib
import os
import queue
import shutil
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Union
from concurrent.futures import ThreadPoolExecutor, Future

from trinity.decorators.dev import editor


class ProcessingStatus(Enum):
    """Status of a processing operation."""

    PENDING = auto()
    QUEUED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class CompressionFormat(Enum):
    """Texture compression formats."""

    NONE = auto()
    BC1 = auto()  # DXT1 - RGB, 1-bit alpha
    BC3 = auto()  # DXT5 - RGBA with interpolated alpha
    BC4 = auto()  # Single channel
    BC5 = auto()  # Two channels (normal maps)
    BC7 = auto()  # High quality RGBA
    ASTC_4x4 = auto()  # ASTC 4x4 block
    ASTC_6x6 = auto()  # ASTC 6x6 block
    ASTC_8x8 = auto()  # ASTC 8x8 block
    ETC2 = auto()  # ETC2 RGB
    ETC2_ALPHA = auto()  # ETC2 RGBA


class AudioFormat(Enum):
    """Audio output formats."""

    WAV = auto()
    OGG = auto()
    MP3 = auto()
    FLAC = auto()
    OPUS = auto()


@dataclass
class CompressionSettings:
    """Settings for texture compression.

    Attributes:
        format: Compression format to use
        quality: Compression quality (0.0-1.0)
        generate_mipmaps: Generate mipmap chain
        max_size: Maximum dimension (will resize if larger)
        resize_filter: Filter for resizing
        srgb: Treat as sRGB color space
        is_normal_map: Treat as normal map
        alpha_mode: Alpha handling (none, straight, premultiplied)
    """

    format: CompressionFormat = CompressionFormat.BC7
    quality: float = 0.8
    generate_mipmaps: bool = True
    max_size: Optional[int] = None
    resize_filter: str = "lanczos"
    srgb: bool = True
    is_normal_map: bool = False
    alpha_mode: str = "straight"


@dataclass
class FormatConversionSettings:
    """Settings for format conversion.

    Attributes:
        output_format: Target format
        preserve_metadata: Keep original metadata
        overwrite_existing: Overwrite if exists
        delete_original: Delete source after conversion
        output_directory: Directory for output (None = same as source)
    """

    output_format: str = ""
    preserve_metadata: bool = True
    overwrite_existing: bool = False
    delete_original: bool = False
    output_directory: Optional[Path] = None


@dataclass
class AudioConversionSettings:
    """Settings for audio conversion.

    Attributes:
        format: Output format
        sample_rate: Target sample rate
        channels: Target channel count
        bit_depth: Target bit depth (for PCM)
        bitrate: Target bitrate (for lossy formats)
        quality: Quality level (0.0-1.0)
        normalize: Normalize audio levels
        trim_silence: Remove silence from ends
    """

    format: AudioFormat = AudioFormat.OGG
    sample_rate: int = 44100
    channels: int = 2
    bit_depth: int = 16
    bitrate: int = 192000
    quality: float = 0.7
    normalize: bool = False
    trim_silence: bool = False


@dataclass
class MeshOptimizationSettings:
    """Settings for mesh optimization.

    Attributes:
        optimize_vertex_cache: Optimize for vertex cache
        optimize_overdraw: Optimize for overdraw
        merge_meshes: Merge meshes by material
        remove_degenerate: Remove degenerate triangles
        weld_vertices: Weld coincident vertices
        weld_threshold: Distance threshold for welding
        simplify: Simplify mesh
        target_triangles: Target triangle count for simplification
        compute_normals: Recompute normals
        compute_tangents: Compute tangent space
    """

    optimize_vertex_cache: bool = True
    optimize_overdraw: bool = True
    merge_meshes: bool = False
    remove_degenerate: bool = True
    weld_vertices: bool = False
    weld_threshold: float = 0.0001
    simplify: bool = False
    target_triangles: Optional[int] = None
    compute_normals: bool = False
    compute_tangents: bool = True


@dataclass
class ProcessingTask:
    """Represents a processing task.

    Attributes:
        id: Unique task identifier
        source_path: Source file path
        output_path: Output file path
        operation: Operation type
        settings: Operation-specific settings
        status: Current status
        progress: Progress 0.0-1.0
        error_message: Error message if failed
        created_at: Creation timestamp
        started_at: Start timestamp
        completed_at: Completion timestamp
        metadata: Additional task data
    """

    id: str
    source_path: Path
    output_path: Path
    operation: str
    settings: Any = None
    status: ProcessingStatus = ProcessingStatus.PENDING
    progress: float = 0.0
    error_message: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Get task duration in milliseconds."""
        if self.started_at is None:
            return 0.0
        end = self.completed_at or time.time()
        return (end - self.started_at) * 1000


@dataclass
class ProcessingResult:
    """Result of a processing operation.

    Attributes:
        task: The completed task
        success: Whether processing succeeded
        output_paths: List of output file paths
        metrics: Processing metrics
    """

    task: ProcessingTask
    success: bool = False
    output_paths: list[Path] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


class ProcessingPipeline:
    """Pipeline for chaining processing operations.

    Allows building complex processing workflows by chaining
    multiple operations together.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.steps: list[tuple[str, Any]] = []

    def add_step(self, operation: str, settings: Any) -> "ProcessingPipeline":
        """Add a processing step."""
        self.steps.append((operation, settings))
        return self

    def compress(self, settings: CompressionSettings) -> "ProcessingPipeline":
        """Add compression step."""
        return self.add_step("compress", settings)

    def convert(self, settings: FormatConversionSettings) -> "ProcessingPipeline":
        """Add format conversion step."""
        return self.add_step("convert", settings)

    def resize(self, max_size: int, filter: str = "lanczos") -> "ProcessingPipeline":
        """Add resize step."""
        return self.add_step("resize", {"max_size": max_size, "filter": filter})

    def optimize_mesh(self, settings: MeshOptimizationSettings) -> "ProcessingPipeline":
        """Add mesh optimization step."""
        return self.add_step("optimize_mesh", settings)

    def convert_audio(self, settings: AudioConversionSettings) -> "ProcessingPipeline":
        """Add audio conversion step."""
        return self.add_step("convert_audio", settings)


class ContentStoreProtocol(Protocol):
    """Protocol for ContentStore integration."""

    def put(self, obj: Any) -> Any:
        """Store object, return content hash."""
        ...

    def has(self, hash: Any) -> bool:
        """Check if hash exists."""
        ...


@editor(category="Assets")
class AssetProcessor:
    """Processes assets with compression, conversion, and optimization.

    Provides:
    - Single and batch processing
    - Progress tracking
    - Cancellation support
    - Pipeline-based processing
    - ContentStore integration for caching

    Attributes:
        output_directory: Default output directory
        content_store: ContentStore for caching
        max_workers: Maximum concurrent workers
        _executor: Thread pool for processing
        _tasks: Active processing tasks
        _progress_callbacks: Progress notification callbacks
    """

    def __init__(
        self,
        output_directory: Optional[Union[str, Path]] = None,
        content_store: Optional[ContentStoreProtocol] = None,
        max_workers: int = 4,
    ) -> None:
        """Initialize the asset processor.

        Args:
            output_directory: Default output directory
            content_store: ContentStore for caching
            max_workers: Maximum concurrent workers
        """
        self.output_directory = Path(output_directory) if output_directory else None
        self.content_store = content_store
        self.max_workers = max_workers

        self._executor: Optional[ThreadPoolExecutor] = None
        self._tasks: dict[str, ProcessingTask] = {}
        self._futures: dict[str, Future] = {}
        self._progress_callbacks: list[Callable[[ProcessingTask], None]] = []
        self._cancelled: set[str] = set()
        self._task_counter = 0
        self._lock = threading.Lock()

    def process(
        self,
        source_path: Union[str, Path],
        operation: str,
        settings: Any = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> ProcessingResult:
        """Process a single asset synchronously.

        Args:
            source_path: Source file path
            operation: Operation to perform
            settings: Operation settings
            output_path: Output path (auto-generated if None)

        Returns:
            ProcessingResult with outcome
        """
        task = self._create_task(
            Path(source_path),
            operation,
            settings,
            Path(output_path) if output_path else None,
        )

        return self._execute_task(task)

    def process_async(
        self,
        source_path: Union[str, Path],
        operation: str,
        settings: Any = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Process a single asset asynchronously.

        Args:
            source_path: Source file path
            operation: Operation to perform
            settings: Operation settings
            output_path: Output path (auto-generated if None)

        Returns:
            Task ID for tracking
        """
        task = self._create_task(
            Path(source_path),
            operation,
            settings,
            Path(output_path) if output_path else None,
        )

        self._start_executor()
        future = self._executor.submit(self._execute_task, task)
        self._futures[task.id] = future

        return task.id

    def process_batch(
        self,
        items: list[tuple[Union[str, Path], str, Any]],
        parallel: bool = True,
    ) -> list[ProcessingResult]:
        """Process multiple assets.

        Args:
            items: List of (source_path, operation, settings) tuples
            parallel: Process in parallel

        Returns:
            List of ProcessingResults
        """
        if parallel:
            self._start_executor()

            tasks = [
                self._create_task(Path(source), op, settings)
                for source, op, settings in items
            ]

            futures = [
                self._executor.submit(self._execute_task, task)
                for task in tasks
            ]

            return [f.result() for f in futures]
        else:
            results = []
            for source, operation, settings in items:
                result = self.process(source, operation, settings)
                results.append(result)
            return results

    def process_pipeline(
        self,
        source_path: Union[str, Path],
        pipeline: ProcessingPipeline,
    ) -> ProcessingResult:
        """Process an asset through a pipeline.

        Args:
            source_path: Source file path
            pipeline: Processing pipeline

        Returns:
            Final ProcessingResult
        """
        current_path = Path(source_path)
        result: Optional[ProcessingResult] = None

        for operation, settings in pipeline.steps:
            result = self.process(current_path, operation, settings)

            if not result.success:
                return result

            if result.output_paths:
                current_path = result.output_paths[0]

        return result or ProcessingResult(
            task=ProcessingTask(
                id=self._generate_task_id(),
                source_path=Path(source_path),
                output_path=current_path,
                operation="pipeline",
            ),
            success=True,
            output_paths=[current_path],
        )

    def get_task_status(self, task_id: str) -> Optional[ProcessingTask]:
        """Get status of a task."""
        return self._tasks.get(task_id)

    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[ProcessingResult]:
        """Wait for and get result of a task.

        Args:
            task_id: Task ID
            timeout: Maximum time to wait

        Returns:
            ProcessingResult or None if not found/timeout
        """
        future = self._futures.get(task_id)
        if future:
            try:
                return future.result(timeout=timeout)
            except Exception:
                return None
        return None

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task.

        Args:
            task_id: Task to cancel

        Returns:
            True if cancellation requested
        """
        if task_id in self._tasks:
            self._cancelled.add(task_id)
            task = self._tasks[task_id]
            if task.status in (ProcessingStatus.PENDING, ProcessingStatus.QUEUED):
                task.status = ProcessingStatus.CANCELLED
            return True
        return False

    def cancel_all(self) -> int:
        """Cancel all pending tasks.

        Returns:
            Number of tasks cancelled
        """
        count = 0
        for task_id, task in self._tasks.items():
            if task.status in (ProcessingStatus.PENDING, ProcessingStatus.QUEUED, ProcessingStatus.IN_PROGRESS):
                self._cancelled.add(task_id)
                count += 1
        return count

    def on_progress(self, callback: Callable[[ProcessingTask], None]) -> None:
        """Register a progress callback."""
        self._progress_callbacks.append(callback)

    def get_stats(self) -> dict[str, Any]:
        """Get processing statistics."""
        by_status = {}
        total_time = 0.0

        for task in self._tasks.values():
            status_name = task.status.name
            by_status[status_name] = by_status.get(status_name, 0) + 1
            if task.completed_at and task.started_at:
                total_time += task.completed_at - task.started_at

        return {
            "total_tasks": len(self._tasks),
            "by_status": by_status,
            "total_processing_time_sec": total_time,
            "active_workers": self._executor._threads.__len__() if self._executor else 0,
        }

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the processor.

        Args:
            wait: Wait for pending tasks to complete
        """
        if self._executor:
            self._executor.shutdown(wait=wait)
            self._executor = None

    def _create_task(
        self,
        source_path: Path,
        operation: str,
        settings: Any,
        output_path: Optional[Path] = None,
    ) -> ProcessingTask:
        """Create a processing task."""
        task_id = self._generate_task_id()

        if output_path is None:
            output_dir = self.output_directory or source_path.parent
            output_path = output_dir / source_path.name

        task = ProcessingTask(
            id=task_id,
            source_path=source_path,
            output_path=output_path,
            operation=operation,
            settings=settings,
        )

        self._tasks[task_id] = task
        return task

    def _execute_task(self, task: ProcessingTask) -> ProcessingResult:
        """Execute a processing task."""
        task.status = ProcessingStatus.IN_PROGRESS
        task.started_at = time.time()
        self._notify_progress(task)

        result = ProcessingResult(task=task)

        try:
            # Check for cancellation
            if task.id in self._cancelled:
                task.status = ProcessingStatus.CANCELLED
                return result

            # Check source exists
            if not task.source_path.exists():
                raise FileNotFoundError(f"Source not found: {task.source_path}")

            # Dispatch to operation handler
            if task.operation == "compress":
                output = self._compress_texture(task)
            elif task.operation == "convert":
                output = self._convert_format(task)
            elif task.operation == "resize":
                output = self._resize_texture(task)
            elif task.operation == "optimize_mesh":
                output = self._optimize_mesh(task)
            elif task.operation == "convert_audio":
                output = self._convert_audio(task)
            elif task.operation == "copy":
                output = self._copy_file(task)
            else:
                raise ValueError(f"Unknown operation: {task.operation}")

            task.progress = 1.0
            task.status = ProcessingStatus.COMPLETED
            task.completed_at = time.time()

            result.success = True
            result.output_paths = [output] if output else []
            result.metrics["duration_ms"] = task.duration_ms

        except Exception as e:
            task.status = ProcessingStatus.FAILED
            task.error_message = str(e)
            task.completed_at = time.time()
            result.success = False

        self._notify_progress(task)
        return result

    def _compress_texture(self, task: ProcessingTask) -> Path:
        """Compress a texture."""
        settings: CompressionSettings = task.settings or CompressionSettings()

        # In a real implementation, this would use image processing libraries
        # For now, just copy the file as a placeholder
        output_path = task.output_path.with_suffix(".dds")

        task.progress = 0.5
        self._notify_progress(task)

        shutil.copy2(task.source_path, output_path)
        return output_path

    def _convert_format(self, task: ProcessingTask) -> Path:
        """Convert file format."""
        settings: FormatConversionSettings = task.settings or FormatConversionSettings()

        output_path = task.output_path
        if settings.output_format:
            output_path = output_path.with_suffix(f".{settings.output_format}")

        if settings.output_directory:
            output_path = settings.output_directory / output_path.name

        task.progress = 0.5
        self._notify_progress(task)

        # Copy file (real implementation would convert)
        shutil.copy2(task.source_path, output_path)

        if settings.delete_original and task.source_path != output_path:
            task.source_path.unlink()

        return output_path

    def _resize_texture(self, task: ProcessingTask) -> Path:
        """Resize a texture."""
        settings = task.settings or {}
        max_size = settings.get("max_size", 2048)

        task.progress = 0.5
        self._notify_progress(task)

        # Copy file (real implementation would resize)
        shutil.copy2(task.source_path, task.output_path)
        return task.output_path

    def _optimize_mesh(self, task: ProcessingTask) -> Path:
        """Optimize a mesh."""
        settings: MeshOptimizationSettings = task.settings or MeshOptimizationSettings()

        task.progress = 0.5
        self._notify_progress(task)

        # Copy file (real implementation would optimize)
        shutil.copy2(task.source_path, task.output_path)
        return task.output_path

    def _convert_audio(self, task: ProcessingTask) -> Path:
        """Convert audio format."""
        settings: AudioConversionSettings = task.settings or AudioConversionSettings()

        ext_map = {
            AudioFormat.WAV: ".wav",
            AudioFormat.OGG: ".ogg",
            AudioFormat.MP3: ".mp3",
            AudioFormat.FLAC: ".flac",
            AudioFormat.OPUS: ".opus",
        }

        output_path = task.output_path.with_suffix(ext_map.get(settings.format, ".ogg"))

        task.progress = 0.5
        self._notify_progress(task)

        # Copy file (real implementation would convert)
        shutil.copy2(task.source_path, output_path)
        return output_path

    def _copy_file(self, task: ProcessingTask) -> Path:
        """Simple file copy."""
        task.progress = 0.5
        self._notify_progress(task)

        task.output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(task.source_path, task.output_path)
        return task.output_path

    def _start_executor(self) -> None:
        """Start the thread pool executor if not running."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def _generate_task_id(self) -> str:
        """Generate a unique task ID."""
        with self._lock:
            self._task_counter += 1
            return f"task_{self._task_counter}_{int(time.time() * 1000)}"

    def _notify_progress(self, task: ProcessingTask) -> None:
        """Notify progress callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(task)
            except Exception:
                pass


@editor(category="Assets")
class BatchProcessor:
    """Batch processor for large-scale asset processing.

    Provides high-level batch processing with:
    - Directory-based processing
    - Filter-based selection
    - Report generation
    """

    def __init__(self, processor: Optional[AssetProcessor] = None) -> None:
        """Initialize the batch processor.

        Args:
            processor: AssetProcessor instance (creates new if None)
        """
        self.processor = processor or AssetProcessor()
        self._results: list[ProcessingResult] = []

    def process_directory(
        self,
        directory: Union[str, Path],
        operation: str,
        settings: Any = None,
        recursive: bool = True,
        filter_extensions: Optional[set[str]] = None,
        output_directory: Optional[Union[str, Path]] = None,
    ) -> list[ProcessingResult]:
        """Process all matching files in a directory.

        Args:
            directory: Source directory
            operation: Operation to perform
            settings: Operation settings
            recursive: Process subdirectories
            filter_extensions: Only process these extensions
            output_directory: Output directory

        Returns:
            List of ProcessingResults
        """
        directory = Path(directory)
        output_dir = Path(output_directory) if output_directory else None

        if recursive:
            files = list(directory.rglob("*"))
        else:
            files = list(directory.iterdir())

        items = []
        for file in files:
            if not file.is_file():
                continue

            ext = file.suffix.lstrip(".").lower()
            if filter_extensions and ext not in filter_extensions:
                continue

            output_path = None
            if output_dir:
                relative = file.relative_to(directory)
                output_path = output_dir / relative

            items.append((file, operation, settings))

        results = self.processor.process_batch(items, parallel=True)
        self._results.extend(results)

        return results

    def generate_report(self) -> dict[str, Any]:
        """Generate a processing report."""
        success_count = sum(1 for r in self._results if r.success)
        failed_count = len(self._results) - success_count

        total_time = sum(r.task.duration_ms for r in self._results)

        by_operation = {}
        for result in self._results:
            op = result.task.operation
            if op not in by_operation:
                by_operation[op] = {"success": 0, "failed": 0}
            if result.success:
                by_operation[op]["success"] += 1
            else:
                by_operation[op]["failed"] += 1

        return {
            "total_processed": len(self._results),
            "success_count": success_count,
            "failed_count": failed_count,
            "total_time_ms": total_time,
            "by_operation": by_operation,
            "failures": [
                {
                    "path": str(r.task.source_path),
                    "operation": r.task.operation,
                    "error": r.task.error_message,
                }
                for r in self._results if not r.success
            ],
        }

    def clear_results(self) -> None:
        """Clear stored results."""
        self._results.clear()


__all__ = [
    "ProcessingStatus",
    "CompressionFormat",
    "AudioFormat",
    "CompressionSettings",
    "FormatConversionSettings",
    "AudioConversionSettings",
    "MeshOptimizationSettings",
    "ProcessingTask",
    "ProcessingResult",
    "ProcessingPipeline",
    "AssetProcessor",
    "BatchProcessor",
]
