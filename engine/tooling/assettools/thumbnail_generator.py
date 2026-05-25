"""
ThumbnailGenerator - Generate thumbnails for assets asynchronously.

Provides thumbnail generation with:
- Async generation with priority queue
- Intelligent caching
- Multiple size presets
- Format-specific thumbnail generation
- Memory-efficient processing
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from queue import PriorityQueue
from typing import Any, Callable, Optional, Protocol, Union

from trinity.decorators.dev import editor


class ThumbnailSize(Enum):
    """Standard thumbnail sizes."""

    TINY = (32, 32)
    SMALL = (64, 64)
    MEDIUM = (128, 128)
    LARGE = (256, 256)
    XLARGE = (512, 512)

    @property
    def dimensions(self) -> tuple[int, int]:
        """Get width and height."""
        return self.value


class ThumbnailStatus(Enum):
    """Status of thumbnail generation."""

    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    CACHED = auto()


class ThumbnailPriority(Enum):
    """Priority levels for thumbnail generation."""

    LOW = 3
    NORMAL = 2
    HIGH = 1
    IMMEDIATE = 0


@dataclass(order=True)
class ThumbnailRequest:
    """Request for thumbnail generation.

    Attributes:
        priority: Generation priority
        asset_path: Path to the asset
        size: Requested thumbnail size
        callback: Callback when complete
        request_time: When request was made
    """

    priority: int = field(compare=True)
    asset_path: Path = field(compare=False)
    size: ThumbnailSize = field(compare=False)
    callback: Optional[Callable[["ThumbnailResult"], None]] = field(compare=False, default=None)
    request_time: float = field(compare=False, default_factory=time.time)


@dataclass
class ThumbnailResult:
    """Result of thumbnail generation.

    Attributes:
        asset_path: Path to the source asset
        thumbnail_path: Path to generated thumbnail
        size: Size of the thumbnail
        status: Generation status
        error_message: Error message if failed
        generation_time_ms: Time to generate
        from_cache: Whether result was from cache
        width: Actual thumbnail width
        height: Actual thumbnail height
    """

    asset_path: Path
    thumbnail_path: Optional[Path] = None
    size: ThumbnailSize = ThumbnailSize.MEDIUM
    status: ThumbnailStatus = ThumbnailStatus.PENDING
    error_message: Optional[str] = None
    generation_time_ms: float = 0.0
    from_cache: bool = False
    width: int = 0
    height: int = 0


class ThumbnailCache:
    """Cache for generated thumbnails.

    Provides:
    - Disk-based caching
    - LRU eviction
    - Content-hash based keys
    - Size-specific caching

    Attributes:
        cache_directory: Directory for cached thumbnails
        max_size_mb: Maximum cache size in MB
        _index: Index of cached thumbnails
        _lock: Thread lock for cache operations
    """

    def __init__(
        self,
        cache_directory: Union[str, Path],
        max_size_mb: float = 500.0,
    ) -> None:
        """Initialize the thumbnail cache.

        Args:
            cache_directory: Directory for cached thumbnails
            max_size_mb: Maximum cache size in MB
        """
        self.cache_directory = Path(cache_directory)
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb

        self._index: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._total_size_bytes = 0

        # Load existing cache index
        self._load_index()

    def get(
        self,
        asset_path: Path,
        size: ThumbnailSize,
        asset_mtime: float,
    ) -> Optional[Path]:
        """Get cached thumbnail if valid.

        Args:
            asset_path: Path to the asset
            size: Requested size
            asset_mtime: Asset modification time

        Returns:
            Path to cached thumbnail or None
        """
        key = self._cache_key(asset_path, size)

        with self._lock:
            entry = self._index.get(key)
            if entry is None:
                return None

            # Check if cache is still valid
            if entry.get("asset_mtime", 0) < asset_mtime:
                # Asset modified, cache invalid
                self._remove_entry(key)
                return None

            cache_path = self.cache_directory / entry["filename"]
            if not cache_path.exists():
                self._remove_entry(key)
                return None

            # Update access time for LRU
            entry["last_access"] = time.time()

            return cache_path

    def put(
        self,
        asset_path: Path,
        size: ThumbnailSize,
        thumbnail_data: bytes,
        asset_mtime: float,
    ) -> Path:
        """Store a thumbnail in cache.

        Args:
            asset_path: Path to the asset
            size: Thumbnail size
            thumbnail_data: Thumbnail image data
            asset_mtime: Asset modification time

        Returns:
            Path to cached thumbnail
        """
        key = self._cache_key(asset_path, size)

        # Generate filename
        filename = f"{key}.png"
        cache_path = self.cache_directory / filename

        # Check cache size before adding
        data_size = len(thumbnail_data)
        self._ensure_space(data_size)

        # Write thumbnail
        cache_path.write_bytes(thumbnail_data)

        with self._lock:
            # Remove old entry if exists
            if key in self._index:
                self._remove_entry(key)

            # Add new entry
            self._index[key] = {
                "filename": filename,
                "asset_path": str(asset_path),
                "asset_mtime": asset_mtime,
                "size": size.name,
                "file_size": data_size,
                "created": time.time(),
                "last_access": time.time(),
            }

            self._total_size_bytes += data_size

        return cache_path

    def invalidate(self, asset_path: Path) -> int:
        """Invalidate all cached thumbnails for an asset.

        Args:
            asset_path: Path to the asset

        Returns:
            Number of thumbnails invalidated
        """
        count = 0
        asset_str = str(asset_path)

        with self._lock:
            keys_to_remove = [
                key for key, entry in self._index.items()
                if entry.get("asset_path") == asset_str
            ]

            for key in keys_to_remove:
                self._remove_entry(key)
                count += 1

        return count

    def clear(self) -> None:
        """Clear the entire cache."""
        with self._lock:
            # Remove all cached files
            for entry in self._index.values():
                cache_path = self.cache_directory / entry["filename"]
                if cache_path.exists():
                    cache_path.unlink()

            self._index.clear()
            self._total_size_bytes = 0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "entries": len(self._index),
                "size_mb": self._total_size_bytes / (1024 * 1024),
                "max_size_mb": self.max_size_mb,
                "fill_ratio": self._total_size_bytes / (self.max_size_mb * 1024 * 1024),
            }

    def _cache_key(self, asset_path: Path, size: ThumbnailSize) -> str:
        """Generate cache key for an asset and size."""
        path_hash = hashlib.md5(str(asset_path).encode()).hexdigest()[:16]
        return f"{path_hash}_{size.name}"

    def _ensure_space(self, needed_bytes: int) -> None:
        """Ensure space in cache by evicting old entries."""
        max_bytes = self.max_size_mb * 1024 * 1024

        with self._lock:
            while self._total_size_bytes + needed_bytes > max_bytes and self._index:
                # Find least recently used entry
                lru_key = min(
                    self._index.keys(),
                    key=lambda k: self._index[k].get("last_access", 0),
                )
                self._remove_entry(lru_key)

    def _remove_entry(self, key: str) -> None:
        """Remove an entry from the cache (must hold lock)."""
        entry = self._index.pop(key, None)
        if entry:
            cache_path = self.cache_directory / entry["filename"]
            if cache_path.exists():
                try:
                    cache_path.unlink()
                except OSError:
                    pass
            self._total_size_bytes -= entry.get("file_size", 0)

    def _load_index(self) -> None:
        """Load cache index from disk."""
        index_path = self.cache_directory / "cache_index.json"
        if index_path.exists():
            import json
            try:
                with open(index_path, "r") as f:
                    self._index = json.load(f)

                # Calculate total size
                self._total_size_bytes = sum(
                    entry.get("file_size", 0)
                    for entry in self._index.values()
                )
            except Exception:
                self._index = {}

    def _save_index(self) -> None:
        """Save cache index to disk."""
        import json
        index_path = self.cache_directory / "cache_index.json"
        with self._lock:
            with open(index_path, "w") as f:
                json.dump(self._index, f)


# Asset type to generator mapping
_SUPPORTED_EXTENSIONS = {
    # Images - direct thumbnail
    "png", "jpg", "jpeg", "bmp", "tga", "tiff", "tif", "exr", "hdr",
    # 3D Models - render preview
    "fbx", "obj", "gltf", "glb", "dae",
    # Audio - waveform
    "wav", "ogg", "mp3", "flac", "aiff",
    # Materials - preview
    "mat", "mtl",
}


@editor(category="Assets")
class ThumbnailGenerator:
    """Generates thumbnails for assets asynchronously.

    Provides:
    - Async generation with priority queue
    - Caching with LRU eviction
    - Format-specific generation
    - Progress tracking
    - Batch generation

    Attributes:
        cache: Thumbnail cache
        max_workers: Maximum concurrent workers
        default_size: Default thumbnail size
        _executor: Thread pool for generation
        _queue: Priority queue for requests
        _results: Recent results
        _progress_callbacks: Progress notification callbacks
    """

    def __init__(
        self,
        cache_directory: Union[str, Path],
        max_workers: int = 4,
        default_size: ThumbnailSize = ThumbnailSize.MEDIUM,
        max_cache_mb: float = 500.0,
    ) -> None:
        """Initialize the thumbnail generator.

        Args:
            cache_directory: Directory for cached thumbnails
            max_workers: Maximum concurrent workers
            default_size: Default thumbnail size
            max_cache_mb: Maximum cache size in MB
        """
        self.cache = ThumbnailCache(cache_directory, max_cache_mb)
        self.max_workers = max_workers
        self.default_size = default_size

        self._executor: Optional[ThreadPoolExecutor] = None
        self._queue: PriorityQueue = PriorityQueue()
        self._results: dict[str, ThumbnailResult] = {}
        self._progress_callbacks: list[Callable[[ThumbnailResult], None]] = []
        self._pending: set[str] = set()
        self._lock = threading.Lock()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

    def request(
        self,
        asset_path: Union[str, Path],
        size: Optional[ThumbnailSize] = None,
        priority: ThumbnailPriority = ThumbnailPriority.NORMAL,
        callback: Optional[Callable[[ThumbnailResult], None]] = None,
    ) -> str:
        """Request thumbnail generation.

        Args:
            asset_path: Path to the asset
            size: Thumbnail size (uses default if None)
            priority: Generation priority
            callback: Callback when complete

        Returns:
            Request ID for tracking
        """
        asset_path = Path(asset_path)
        size = size or self.default_size
        request_id = self._request_id(asset_path, size)

        # Check if already pending
        with self._lock:
            if request_id in self._pending:
                return request_id
            self._pending.add(request_id)

        # Check cache first
        if asset_path.exists():
            cached = self.cache.get(asset_path, size, asset_path.stat().st_mtime)
            if cached:
                result = ThumbnailResult(
                    asset_path=asset_path,
                    thumbnail_path=cached,
                    size=size,
                    status=ThumbnailStatus.CACHED,
                    from_cache=True,
                )
                self._results[request_id] = result

                with self._lock:
                    self._pending.discard(request_id)

                if callback:
                    callback(result)

                return request_id

        # Queue request
        request = ThumbnailRequest(
            priority=priority.value,
            asset_path=asset_path,
            size=size,
            callback=callback,
        )
        self._queue.put(request)

        # Start processing if not running
        self._ensure_running()

        return request_id

    def request_batch(
        self,
        asset_paths: list[Union[str, Path]],
        size: Optional[ThumbnailSize] = None,
        priority: ThumbnailPriority = ThumbnailPriority.NORMAL,
    ) -> list[str]:
        """Request thumbnails for multiple assets.

        Args:
            asset_paths: List of asset paths
            size: Thumbnail size
            priority: Generation priority

        Returns:
            List of request IDs
        """
        return [
            self.request(path, size, priority)
            for path in asset_paths
        ]

    def get_result(self, request_id: str) -> Optional[ThumbnailResult]:
        """Get result for a request.

        Args:
            request_id: Request ID

        Returns:
            ThumbnailResult or None if not ready
        """
        return self._results.get(request_id)

    def get_thumbnail(
        self,
        asset_path: Union[str, Path],
        size: Optional[ThumbnailSize] = None,
    ) -> Optional[Path]:
        """Get thumbnail for an asset if cached.

        Args:
            asset_path: Path to the asset
            size: Thumbnail size

        Returns:
            Path to thumbnail or None
        """
        asset_path = Path(asset_path)
        size = size or self.default_size

        if not asset_path.exists():
            return None

        return self.cache.get(asset_path, size, asset_path.stat().st_mtime)

    def has_thumbnail(self, asset_path: Union[str, Path], size: Optional[ThumbnailSize] = None) -> bool:
        """Check if thumbnail exists for an asset.

        Args:
            asset_path: Path to the asset
            size: Thumbnail size

        Returns:
            True if thumbnail is cached
        """
        return self.get_thumbnail(asset_path, size) is not None

    def invalidate(self, asset_path: Union[str, Path]) -> int:
        """Invalidate thumbnails for an asset.

        Args:
            asset_path: Path to the asset

        Returns:
            Number of thumbnails invalidated
        """
        return self.cache.invalidate(Path(asset_path))

    def on_complete(self, callback: Callable[[ThumbnailResult], None]) -> None:
        """Register a completion callback."""
        self._progress_callbacks.append(callback)

    def get_stats(self) -> dict[str, Any]:
        """Get generator statistics."""
        with self._lock:
            pending_count = len(self._pending)

        cache_stats = self.cache.get_stats()

        return {
            "pending_requests": pending_count,
            "queue_size": self._queue.qsize(),
            "completed": len(self._results),
            "cache": cache_stats,
        }

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the generator.

        Args:
            wait: Wait for pending requests to complete
        """
        self._running = False

        if self._worker_thread and self._worker_thread.is_alive():
            if wait:
                self._worker_thread.join(timeout=5.0)

        if self._executor:
            self._executor.shutdown(wait=wait)
            self._executor = None

        # Save cache index
        self.cache._save_index()

    def _ensure_running(self) -> None:
        """Ensure the worker thread is running."""
        if self._running:
            return

        self._running = True
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()

    def _process_queue(self) -> None:
        """Process the request queue."""
        while self._running:
            try:
                request = self._queue.get(timeout=0.5)
            except Exception:
                continue

            if not self._running:
                break

            # Submit to executor
            self._executor.submit(self._generate_thumbnail, request)

    def _generate_thumbnail(self, request: ThumbnailRequest) -> None:
        """Generate a thumbnail for a request."""
        request_id = self._request_id(request.asset_path, request.size)
        start_time = time.perf_counter()

        result = ThumbnailResult(
            asset_path=request.asset_path,
            size=request.size,
        )

        try:
            if not request.asset_path.exists():
                raise FileNotFoundError(f"Asset not found: {request.asset_path}")

            # Check if supported
            ext = request.asset_path.suffix.lstrip(".").lower()
            if ext not in _SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported format: {ext}")

            # Generate thumbnail data
            thumbnail_data = self._create_thumbnail(
                request.asset_path,
                request.size.dimensions,
            )

            # Store in cache
            mtime = request.asset_path.stat().st_mtime
            cache_path = self.cache.put(
                request.asset_path,
                request.size,
                thumbnail_data,
                mtime,
            )

            result.thumbnail_path = cache_path
            result.status = ThumbnailStatus.COMPLETED
            result.width, result.height = request.size.dimensions

        except Exception as e:
            result.status = ThumbnailStatus.FAILED
            result.error_message = str(e)

        result.generation_time_ms = (time.perf_counter() - start_time) * 1000

        # Store result
        self._results[request_id] = result

        with self._lock:
            self._pending.discard(request_id)

        # Notify callbacks
        if request.callback:
            try:
                request.callback(result)
            except Exception:
                pass

        for callback in self._progress_callbacks:
            try:
                callback(result)
            except Exception:
                pass

    def _create_thumbnail(
        self,
        asset_path: Path,
        dimensions: tuple[int, int],
    ) -> bytes:
        """Create thumbnail data for an asset.

        In a real implementation, this would use image processing
        libraries to generate actual thumbnails. For now, we create
        a placeholder.
        """
        width, height = dimensions
        ext = asset_path.suffix.lstrip(".").lower()

        # Generate a simple placeholder PNG
        # In production, this would use PIL, wand, or similar
        return self._create_placeholder_png(width, height, ext)

    def _create_placeholder_png(
        self,
        width: int,
        height: int,
        asset_type: str,
    ) -> bytes:
        """Create a placeholder PNG thumbnail.

        This is a minimal PNG implementation for testing.
        Real implementation would use proper image libraries.
        """
        import struct
        import zlib

        # PNG signature
        signature = b'\x89PNG\r\n\x1a\n'

        # IHDR chunk
        ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
        ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff
        ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)

        # Generate simple colored rows based on type
        colors = {
            "png": (100, 150, 200),
            "jpg": (200, 150, 100),
            "fbx": (150, 200, 100),
            "obj": (150, 100, 200),
            "wav": (200, 100, 150),
        }
        r, g, b = colors.get(asset_type, (128, 128, 128))

        # Create image data (scanlines with filter byte)
        raw_data = b''
        for y in range(height):
            raw_data += b'\x00'  # Filter type: None
            for x in range(width):
                raw_data += bytes([r, g, b])

        # Compress with zlib
        compressed = zlib.compress(raw_data, 9)

        # IDAT chunk
        idat_crc = zlib.crc32(b'IDAT' + compressed) & 0xffffffff
        idat = struct.pack('>I', len(compressed)) + b'IDAT' + compressed + struct.pack('>I', idat_crc)

        # IEND chunk
        iend_crc = zlib.crc32(b'IEND') & 0xffffffff
        iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)

        return signature + ihdr + idat + iend

    def _request_id(self, asset_path: Path, size: ThumbnailSize) -> str:
        """Generate a request ID."""
        path_hash = hashlib.md5(str(asset_path).encode()).hexdigest()[:16]
        return f"{path_hash}_{size.name}"


__all__ = [
    "ThumbnailSize",
    "ThumbnailStatus",
    "ThumbnailPriority",
    "ThumbnailRequest",
    "ThumbnailResult",
    "ThumbnailCache",
    "ThumbnailGenerator",
]
