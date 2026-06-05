"""
Async cell loading pipeline for world partition streaming.

Provides a multi-stage loading pipeline:
    1. Terrain loading (heightmap, material data)
    2. Height data processing
    3. GPU upload (buffers, textures)

Supports cancellation, progress tracking, and error handling.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import time
import uuid


T = TypeVar("T")


class LoadStage(Enum):
    """Stages in the async loading pipeline."""

    PENDING = auto()       # Queued but not started
    TERRAIN = auto()       # Loading terrain data
    HEIGHT_DATA = auto()   # Processing height data
    GPU_UPLOAD = auto()    # Uploading to GPU
    COMPLETE = auto()      # Successfully completed
    FAILED = auto()        # Failed with error
    CANCELLED = auto()     # Cancelled by request

    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal stage."""
        return self in (LoadStage.COMPLETE, LoadStage.FAILED, LoadStage.CANCELLED)

    @property
    def is_active(self) -> bool:
        """Check if loading is actively in progress."""
        return self in (LoadStage.TERRAIN, LoadStage.HEIGHT_DATA, LoadStage.GPU_UPLOAD)

    @property
    def progress_fraction(self) -> float:
        """Get approximate progress fraction for this stage."""
        fractions = {
            LoadStage.PENDING: 0.0,
            LoadStage.TERRAIN: 0.33,
            LoadStage.HEIGHT_DATA: 0.66,
            LoadStage.GPU_UPLOAD: 0.9,
            LoadStage.COMPLETE: 1.0,
            LoadStage.FAILED: 0.0,
            LoadStage.CANCELLED: 0.0,
        }
        return fractions.get(self, 0.0)


@dataclass
class LoadError:
    """Error information for failed load operations."""

    stage: LoadStage
    message: str
    exception: Optional[Exception] = None
    cell_id: Optional[Tuple[int, int]] = None
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        loc = f"cell {self.cell_id}" if self.cell_id else "unknown cell"
        return f"LoadError at {self.stage.name} for {loc}: {self.message}"


@dataclass
class LoadResult(Generic[T]):
    """Result of a load operation."""

    success: bool
    data: Optional[T] = None
    error: Optional[LoadError] = None
    load_time_ms: float = 0.0
    cell_id: Optional[Tuple[int, int]] = None

    @classmethod
    def ok(
        cls,
        data: T,
        load_time_ms: float = 0.0,
        cell_id: Optional[Tuple[int, int]] = None,
    ) -> "LoadResult[T]":
        """Create a successful result."""
        return cls(success=True, data=data, load_time_ms=load_time_ms, cell_id=cell_id)

    @classmethod
    def fail(
        cls,
        error: LoadError,
        cell_id: Optional[Tuple[int, int]] = None,
    ) -> "LoadResult[T]":
        """Create a failed result."""
        return cls(success=False, error=error, cell_id=cell_id)


@dataclass
class LoadRequest:
    """Request to load a cell."""

    cell_x: int
    cell_y: int
    priority: int = 0
    request_time: float = field(default_factory=time.time)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Loading state
    stage: LoadStage = LoadStage.PENDING
    progress: float = 0.0

    # Timing
    start_time: float = 0.0
    stage_start_time: float = 0.0

    # Cancellation
    cancelled: bool = False

    # Callbacks
    on_complete: Optional[Callable[[LoadResult], None]] = None
    on_progress: Optional[Callable[[float, LoadStage], None]] = None

    @property
    def cell_id(self) -> Tuple[int, int]:
        """Get the cell ID as a tuple."""
        return (self.cell_x, self.cell_y)

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time since request was made."""
        return time.time() - self.request_time

    @property
    def load_time(self) -> float:
        """Get elapsed time since loading started."""
        if self.start_time <= 0:
            return 0.0
        return time.time() - self.start_time

    def update_progress(self, progress: float, stage: LoadStage) -> None:
        """Update the loading progress."""
        self.progress = max(0.0, min(1.0, progress))
        if stage != self.stage:
            self.stage = stage
            self.stage_start_time = time.time()
        if self.on_progress:
            self.on_progress(self.progress, self.stage)

    def cancel(self) -> None:
        """Request cancellation of this load."""
        self.cancelled = True
        self.stage = LoadStage.CANCELLED


# =============================================================================
# LOADER PROTOCOLS AND BASE CLASSES
# =============================================================================


class CellDataLoader(Protocol):
    """Protocol for cell data loaders."""

    async def load(
        self,
        cell_x: int,
        cell_y: int,
        **kwargs: Any,
    ) -> LoadResult[Any]:
        """Load data for a cell."""
        ...

    def can_cancel(self) -> bool:
        """Check if this loader supports cancellation."""
        ...


@dataclass
class TerrainData:
    """Terrain data for a cell."""

    heightmap: bytes = b""
    heightmap_width: int = 0
    heightmap_height: int = 0
    min_height: float = 0.0
    max_height: float = 100.0
    material_ids: bytes = b""
    material_weights: bytes = b""


@dataclass
class HeightData:
    """Processed height data for a cell."""

    heights: List[float] = field(default_factory=list)
    normals: List[Tuple[float, float, float]] = field(default_factory=list)
    width: int = 0
    height: int = 0
    min_height: float = 0.0
    max_height: float = 100.0


@dataclass
class GPUBufferHandle:
    """Handle to a GPU buffer."""

    buffer_id: int = 0
    size_bytes: int = 0
    buffer_type: str = "vertex"
    is_valid: bool = True

    def release(self) -> None:
        """Release the GPU buffer."""
        self.is_valid = False


@dataclass
class GPUCellData:
    """GPU resources for a cell."""

    vertex_buffer: Optional[GPUBufferHandle] = None
    index_buffer: Optional[GPUBufferHandle] = None
    heightmap_texture: Optional[GPUBufferHandle] = None
    normal_texture: Optional[GPUBufferHandle] = None
    material_texture: Optional[GPUBufferHandle] = None

    def release_all(self) -> None:
        """Release all GPU resources."""
        if self.vertex_buffer:
            self.vertex_buffer.release()
        if self.index_buffer:
            self.index_buffer.release()
        if self.heightmap_texture:
            self.heightmap_texture.release()
        if self.normal_texture:
            self.normal_texture.release()
        if self.material_texture:
            self.material_texture.release()


class TerrainLoader:
    """
    Loads terrain data from disk or network.

    First stage in the loading pipeline.
    """

    def __init__(
        self,
        base_path: str = "",
        chunk_size: int = 256,
        cache_enabled: bool = True,
    ) -> None:
        self.base_path = base_path
        self.chunk_size = chunk_size
        self.cache_enabled = cache_enabled
        self._cache: Dict[Tuple[int, int], TerrainData] = {}
        self._loading: Set[Tuple[int, int]] = set()

    async def load(
        self,
        cell_x: int,
        cell_y: int,
        **kwargs: Any,
    ) -> LoadResult[TerrainData]:
        """
        Load terrain data for a cell.

        Args:
            cell_x: Cell X coordinate
            cell_y: Cell Y coordinate
            **kwargs: Additional options

        Returns:
            LoadResult containing TerrainData or error
        """
        cell_id = (cell_x, cell_y)
        start_time = time.time()

        # Check cache
        if self.cache_enabled and cell_id in self._cache:
            elapsed = (time.time() - start_time) * 1000
            return LoadResult.ok(self._cache[cell_id], elapsed, cell_id)

        # Prevent duplicate loads
        if cell_id in self._loading:
            return LoadResult.fail(
                LoadError(
                    stage=LoadStage.TERRAIN,
                    message="Cell is already loading",
                    cell_id=cell_id,
                ),
                cell_id,
            )

        self._loading.add(cell_id)

        try:
            # Simulate async loading (in real implementation, read from disk)
            await asyncio.sleep(0.01)  # Simulated IO delay

            # Generate placeholder terrain data
            terrain = TerrainData(
                heightmap=bytes(self.chunk_size * self.chunk_size * 2),
                heightmap_width=self.chunk_size,
                heightmap_height=self.chunk_size,
                min_height=0.0,
                max_height=100.0,
                material_ids=bytes(self.chunk_size * self.chunk_size),
                material_weights=bytes(self.chunk_size * self.chunk_size * 4),
            )

            # Cache result
            if self.cache_enabled:
                self._cache[cell_id] = terrain

            elapsed = (time.time() - start_time) * 1000
            return LoadResult.ok(terrain, elapsed, cell_id)

        except Exception as e:
            return LoadResult.fail(
                LoadError(
                    stage=LoadStage.TERRAIN,
                    message=str(e),
                    exception=e,
                    cell_id=cell_id,
                ),
                cell_id,
            )
        finally:
            self._loading.discard(cell_id)

    def can_cancel(self) -> bool:
        """Check if this loader supports cancellation."""
        return True

    def clear_cache(self) -> None:
        """Clear the terrain cache."""
        self._cache.clear()

    def evict_from_cache(self, cell_x: int, cell_y: int) -> bool:
        """Evict a specific cell from cache."""
        cell_id = (cell_x, cell_y)
        if cell_id in self._cache:
            del self._cache[cell_id]
            return True
        return False


class HeightDataLoader:
    """
    Processes raw terrain data into height/normal data.

    Second stage in the loading pipeline.
    """

    def __init__(self, generate_normals: bool = True) -> None:
        self.generate_normals = generate_normals

    async def load(
        self,
        cell_x: int,
        cell_y: int,
        terrain_data: TerrainData,
        **kwargs: Any,
    ) -> LoadResult[HeightData]:
        """
        Process terrain data into height data.

        Args:
            cell_x: Cell X coordinate
            cell_y: Cell Y coordinate
            terrain_data: Raw terrain data to process
            **kwargs: Additional options

        Returns:
            LoadResult containing HeightData or error
        """
        cell_id = (cell_x, cell_y)
        start_time = time.time()

        try:
            # Simulate processing delay
            await asyncio.sleep(0.005)

            width = terrain_data.heightmap_width
            height = terrain_data.heightmap_height

            # Generate placeholder height data
            heights = [0.0] * (width * height)
            normals = [(0.0, 1.0, 0.0)] * (width * height) if self.generate_normals else []

            height_data = HeightData(
                heights=heights,
                normals=normals,
                width=width,
                height=height,
                min_height=terrain_data.min_height,
                max_height=terrain_data.max_height,
            )

            elapsed = (time.time() - start_time) * 1000
            return LoadResult.ok(height_data, elapsed, cell_id)

        except Exception as e:
            return LoadResult.fail(
                LoadError(
                    stage=LoadStage.HEIGHT_DATA,
                    message=str(e),
                    exception=e,
                    cell_id=cell_id,
                ),
                cell_id,
            )

    def can_cancel(self) -> bool:
        """Check if this loader supports cancellation."""
        return False  # Processing is usually fast enough to complete


class GPUUploader:
    """
    Uploads cell data to GPU buffers.

    Third stage in the loading pipeline.
    """

    def __init__(
        self,
        max_upload_size_mb: float = 16.0,
        async_upload: bool = True,
    ) -> None:
        self.max_upload_size_mb = max_upload_size_mb
        self.async_upload = async_upload
        self._next_buffer_id = 1
        self._allocated_buffers: Dict[int, GPUBufferHandle] = {}

    async def load(
        self,
        cell_x: int,
        cell_y: int,
        height_data: HeightData,
        **kwargs: Any,
    ) -> LoadResult[GPUCellData]:
        """
        Upload height data to GPU.

        Args:
            cell_x: Cell X coordinate
            cell_y: Cell Y coordinate
            height_data: Processed height data to upload
            **kwargs: Additional options

        Returns:
            LoadResult containing GPUCellData or error
        """
        cell_id = (cell_x, cell_y)
        start_time = time.time()

        try:
            # Simulate GPU upload delay
            if self.async_upload:
                await asyncio.sleep(0.002)

            # Create GPU buffers (simulated)
            vertex_size = height_data.width * height_data.height * 12  # 3 floats
            index_size = (height_data.width - 1) * (height_data.height - 1) * 6 * 4
            texture_size = height_data.width * height_data.height * 4

            vertex_buffer = self._allocate_buffer(vertex_size, "vertex")
            index_buffer = self._allocate_buffer(index_size, "index")
            heightmap_texture = self._allocate_buffer(texture_size, "texture")
            normal_texture = self._allocate_buffer(texture_size * 3, "texture")

            gpu_data = GPUCellData(
                vertex_buffer=vertex_buffer,
                index_buffer=index_buffer,
                heightmap_texture=heightmap_texture,
                normal_texture=normal_texture,
            )

            elapsed = (time.time() - start_time) * 1000
            return LoadResult.ok(gpu_data, elapsed, cell_id)

        except Exception as e:
            return LoadResult.fail(
                LoadError(
                    stage=LoadStage.GPU_UPLOAD,
                    message=str(e),
                    exception=e,
                    cell_id=cell_id,
                ),
                cell_id,
            )

    def _allocate_buffer(self, size: int, buffer_type: str) -> GPUBufferHandle:
        """Allocate a GPU buffer (simulated)."""
        buffer_id = self._next_buffer_id
        self._next_buffer_id += 1

        handle = GPUBufferHandle(
            buffer_id=buffer_id,
            size_bytes=size,
            buffer_type=buffer_type,
            is_valid=True,
        )
        self._allocated_buffers[buffer_id] = handle
        return handle

    def can_cancel(self) -> bool:
        """Check if this loader supports cancellation."""
        return self.async_upload

    def release_buffer(self, buffer_id: int) -> bool:
        """Release a GPU buffer."""
        if buffer_id in self._allocated_buffers:
            self._allocated_buffers[buffer_id].release()
            del self._allocated_buffers[buffer_id]
            return True
        return False

    def get_allocated_memory(self) -> int:
        """Get total allocated GPU memory in bytes."""
        return sum(b.size_bytes for b in self._allocated_buffers.values() if b.is_valid)


# =============================================================================
# ASYNC LOAD PIPELINE
# =============================================================================


@dataclass
class LoadPipelineConfig:
    """Configuration for the async load pipeline."""

    max_concurrent_loads: int = 4
    max_pending_requests: int = 64
    terrain_cache_enabled: bool = True
    gpu_async_upload: bool = True
    timeout_seconds: float = 30.0
    retry_count: int = 2
    retry_delay_ms: float = 100.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_concurrent_loads < 1:
            raise ValueError("max_concurrent_loads must be at least 1")
        if self.max_pending_requests < 1:
            raise ValueError("max_pending_requests must be at least 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.retry_count < 0:
            raise ValueError("retry_count must be non-negative")


@dataclass
class CellLoadData:
    """Combined load data for a cell."""

    terrain: Optional[TerrainData] = None
    height_data: Optional[HeightData] = None
    gpu_data: Optional[GPUCellData] = None
    load_time_ms: float = 0.0
    cell_id: Optional[Tuple[int, int]] = None


class AsyncLoadPipeline:
    """
    Async loading pipeline for world partition streaming.

    Manages a three-stage loading process:
    1. Terrain loading (disk/network)
    2. Height data processing (CPU)
    3. GPU upload

    Supports priority queuing, cancellation, and progress tracking.
    """

    def __init__(
        self,
        config: Optional[LoadPipelineConfig] = None,
        terrain_loader: Optional[TerrainLoader] = None,
        height_loader: Optional[HeightDataLoader] = None,
        gpu_uploader: Optional[GPUUploader] = None,
    ) -> None:
        self.config = config or LoadPipelineConfig()

        # Initialize loaders
        self.terrain_loader = terrain_loader or TerrainLoader(
            cache_enabled=self.config.terrain_cache_enabled
        )
        self.height_loader = height_loader or HeightDataLoader()
        self.gpu_uploader = gpu_uploader or GPUUploader(
            async_upload=self.config.gpu_async_upload
        )

        # Request queues
        self._pending: List[LoadRequest] = []
        self._active: Dict[str, LoadRequest] = {}
        self._completed: Dict[Tuple[int, int], LoadResult[CellLoadData]] = {}

        # Callbacks
        self._on_complete: List[Callable[[LoadResult[CellLoadData]], None]] = []
        self._on_error: List[Callable[[LoadError], None]] = []

        # Stats
        self._total_loads = 0
        self._successful_loads = 0
        self._failed_loads = 0
        self._cancelled_loads = 0

    @property
    def pending_count(self) -> int:
        """Get number of pending requests."""
        return len(self._pending)

    @property
    def active_count(self) -> int:
        """Get number of active loads."""
        return len(self._active)

    @property
    def can_accept_request(self) -> bool:
        """Check if pipeline can accept more requests."""
        return len(self._pending) < self.config.max_pending_requests

    def submit(
        self,
        cell_x: int,
        cell_y: int,
        priority: int = 0,
        on_complete: Optional[Callable[[LoadResult], None]] = None,
        on_progress: Optional[Callable[[float, LoadStage], None]] = None,
    ) -> Optional[LoadRequest]:
        """
        Submit a load request.

        Args:
            cell_x: Cell X coordinate
            cell_y: Cell Y coordinate
            priority: Load priority (higher = more important)
            on_complete: Callback when load completes
            on_progress: Callback for progress updates

        Returns:
            LoadRequest if submitted, None if queue is full
        """
        if not self.can_accept_request:
            return None

        cell_id = (cell_x, cell_y)

        # Check if already loading or completed
        for req in self._pending:
            if req.cell_id == cell_id:
                return req

        for req in self._active.values():
            if req.cell_id == cell_id:
                return req

        request = LoadRequest(
            cell_x=cell_x,
            cell_y=cell_y,
            priority=priority,
            on_complete=on_complete,
            on_progress=on_progress,
        )

        self._pending.append(request)
        self._sort_pending()

        return request

    def cancel(self, cell_x: int, cell_y: int) -> bool:
        """
        Cancel a load request.

        Args:
            cell_x: Cell X coordinate
            cell_y: Cell Y coordinate

        Returns:
            True if request was found and cancelled
        """
        cell_id = (cell_x, cell_y)

        # Check pending queue
        for i, req in enumerate(self._pending):
            if req.cell_id == cell_id:
                req.cancel()
                self._pending.pop(i)
                self._cancelled_loads += 1
                return True

        # Check active loads
        for req_id, req in list(self._active.items()):
            if req.cell_id == cell_id:
                req.cancel()
                return True

        return False

    def cancel_all(self) -> int:
        """
        Cancel all pending and active loads.

        Returns:
            Number of cancelled requests
        """
        count = 0

        for req in self._pending:
            req.cancel()
            count += 1
        self._pending.clear()

        for req in self._active.values():
            req.cancel()
            count += 1

        self._cancelled_loads += count
        return count

    async def process(self) -> int:
        """
        Process the load queue.

        Starts new loads up to the concurrent limit.

        Returns:
            Number of loads started
        """
        started = 0

        while (
            self._pending
            and len(self._active) < self.config.max_concurrent_loads
        ):
            request = self._pending.pop(0)
            if request.cancelled:
                continue

            request.start_time = time.time()
            self._active[request.request_id] = request
            self._total_loads += 1

            # Start async load
            asyncio.create_task(self._execute_load(request))
            started += 1

        return started

    async def _execute_load(self, request: LoadRequest) -> None:
        """Execute a single load request through the pipeline."""
        cell_id = request.cell_id
        total_start = time.time()

        try:
            # Stage 1: Terrain loading
            request.update_progress(0.0, LoadStage.TERRAIN)
            if request.cancelled:
                return

            terrain_result = await self.terrain_loader.load(
                request.cell_x, request.cell_y
            )
            if not terrain_result.success:
                self._handle_failure(request, terrain_result.error)
                return

            # Stage 2: Height data processing
            request.update_progress(0.33, LoadStage.HEIGHT_DATA)
            if request.cancelled:
                return

            height_result = await self.height_loader.load(
                request.cell_x, request.cell_y, terrain_result.data
            )
            if not height_result.success:
                self._handle_failure(request, height_result.error)
                return

            # Stage 3: GPU upload
            request.update_progress(0.66, LoadStage.GPU_UPLOAD)
            if request.cancelled:
                return

            gpu_result = await self.gpu_uploader.load(
                request.cell_x, request.cell_y, height_result.data
            )
            if not gpu_result.success:
                self._handle_failure(request, gpu_result.error)
                return

            # Success
            request.update_progress(1.0, LoadStage.COMPLETE)

            total_time = (time.time() - total_start) * 1000
            combined_data = CellLoadData(
                terrain=terrain_result.data,
                height_data=height_result.data,
                gpu_data=gpu_result.data,
                load_time_ms=total_time,
                cell_id=cell_id,
            )

            result = LoadResult.ok(combined_data, total_time, cell_id)
            self._handle_success(request, result)

        except asyncio.CancelledError:
            request.stage = LoadStage.CANCELLED
            self._cancelled_loads += 1
        except Exception as e:
            error = LoadError(
                stage=request.stage,
                message=str(e),
                exception=e,
                cell_id=cell_id,
            )
            self._handle_failure(request, error)
        finally:
            self._active.pop(request.request_id, None)

    def _handle_success(
        self,
        request: LoadRequest,
        result: LoadResult[CellLoadData],
    ) -> None:
        """Handle successful load completion."""
        self._successful_loads += 1
        self._completed[request.cell_id] = result

        if request.on_complete:
            request.on_complete(result)

        for callback in self._on_complete:
            callback(result)

    def _handle_failure(
        self,
        request: LoadRequest,
        error: Optional[LoadError],
    ) -> None:
        """Handle load failure."""
        request.stage = LoadStage.FAILED
        self._failed_loads += 1

        if error:
            result = LoadResult[CellLoadData].fail(error, request.cell_id)
            self._completed[request.cell_id] = result

            for callback in self._on_error:
                callback(error)

            if request.on_complete:
                request.on_complete(result)

    def _sort_pending(self) -> None:
        """Sort pending requests by priority."""
        self._pending.sort(key=lambda r: r.priority, reverse=True)

    def get_request(self, cell_x: int, cell_y: int) -> Optional[LoadRequest]:
        """Get a load request by cell coordinates."""
        cell_id = (cell_x, cell_y)

        for req in self._pending:
            if req.cell_id == cell_id:
                return req

        for req in self._active.values():
            if req.cell_id == cell_id:
                return req

        return None

    def get_result(
        self,
        cell_x: int,
        cell_y: int,
    ) -> Optional[LoadResult[CellLoadData]]:
        """Get a completed load result."""
        return self._completed.get((cell_x, cell_y))

    def clear_result(self, cell_x: int, cell_y: int) -> bool:
        """Clear a completed result from cache."""
        cell_id = (cell_x, cell_y)
        if cell_id in self._completed:
            del self._completed[cell_id]
            return True
        return False

    def on_complete(
        self,
        callback: Callable[[LoadResult[CellLoadData]], None],
    ) -> None:
        """Register a callback for load completion."""
        self._on_complete.append(callback)

    def on_error(self, callback: Callable[[LoadError], None]) -> None:
        """Register a callback for load errors."""
        self._on_error.append(callback)

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "total_loads": self._total_loads,
            "successful_loads": self._successful_loads,
            "failed_loads": self._failed_loads,
            "cancelled_loads": self._cancelled_loads,
            "pending_count": len(self._pending),
            "active_count": len(self._active),
            "completed_count": len(self._completed),
            "success_rate": (
                self._successful_loads / self._total_loads
                if self._total_loads > 0
                else 0.0
            ),
        }

    def reset_stats(self) -> None:
        """Reset pipeline statistics."""
        self._total_loads = 0
        self._successful_loads = 0
        self._failed_loads = 0
        self._cancelled_loads = 0


__all__ = [
    # Enums
    "LoadStage",
    # Data classes
    "LoadError",
    "LoadResult",
    "LoadRequest",
    "TerrainData",
    "HeightData",
    "GPUBufferHandle",
    "GPUCellData",
    "CellLoadData",
    "LoadPipelineConfig",
    # Loaders
    "TerrainLoader",
    "HeightDataLoader",
    "GPUUploader",
    # Pipeline
    "AsyncLoadPipeline",
]
