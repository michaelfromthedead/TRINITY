"""Pipeline integration for Rust backend (T-MAT-3.4).

This module provides the Python-side interface for pipeline caching and
hot-reload integration with the Rust renderer backend.

Core Components:
    PipelineConfig: Configuration for pipeline compilation
    PipelineCacheHandle: Reference to a cached Rust pipeline
    PipelineIntegration: Main interface to Rust pipeline table

Integration Points:
    - ShaderCacheV2: Content-addressed shader deduplication
    - LruPipelineTable: LRU-evicted pipeline cache
    - DepGraph: Hot-reload dependency tracking

Example::

    from trinity.materials.pipeline_integration import PipelineIntegration

    integration = PipelineIntegration(max_cache_size=64)

    # Get or create a pipeline
    pipeline_id = integration.get_or_create_pipeline(
        wgsl_source=pbr_shader_source,
        vertex_entry="vs_main",
        fragment_entry="fs_main",
    )

    # Hot-reload: invalidate when shader changes
    integration.invalidate_shader("shaders/pbr.wgsl")

Notes:
    This module provides a pure-Python implementation for testing when the
    Rust backend is not available (pyo3 feature disabled). When the Rust
    backend is available, it delegates to the native implementation for
    maximum performance.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Dict, List, Optional, Tuple, Callable, Any
from collections import OrderedDict


# =============================================================================
# Configuration Types
# =============================================================================


class ColorFormat(IntEnum):
    """Output color attachment format (mirrors wgpu::TextureFormat subset)."""
    RGBA8_UNORM = auto()
    RGBA8_SRGB = auto()
    BGRA8_UNORM = auto()
    BGRA8_SRGB = auto()
    RGBA16_FLOAT = auto()
    RGBA32_FLOAT = auto()


class CullMode(IntEnum):
    """Face culling mode."""
    NONE = 0
    FRONT = 1
    BACK = 2


class BlendMode(IntEnum):
    """Blend mode preset."""
    OPAQUE = 0
    ALPHA_BLEND = 1
    ADDITIVE = 2
    MULTIPLY = 3


@dataclass
class PipelineConfig:
    """Configuration for pipeline compilation.

    Attributes:
        vertex_entry: Vertex shader entry point name.
        fragment_entry: Fragment shader entry point name.
        color_format: Output color attachment format.
        depth_format: Optional depth attachment format string.
        cull_mode: Face culling mode.
        blend_mode: Blend mode preset.
        sample_count: MSAA sample count (1, 2, 4, or 8).
        label: Optional debug label for the pipeline.
    """
    vertex_entry: str = "vs_main"
    fragment_entry: str = "fs_main"
    color_format: ColorFormat = ColorFormat.RGBA8_UNORM
    depth_format: Optional[str] = None
    cull_mode: CullMode = CullMode.BACK
    blend_mode: BlendMode = BlendMode.OPAQUE
    sample_count: int = 1
    label: Optional[str] = None


# Default config singleton (avoids object creation on hot path)
_DEFAULT_PIPELINE_CONFIG: Optional["PipelineConfig"] = None


def _get_default_config() -> "PipelineConfig":
    """Get the singleton default pipeline config."""
    global _DEFAULT_PIPELINE_CONFIG
    if _DEFAULT_PIPELINE_CONFIG is None:
        _DEFAULT_PIPELINE_CONFIG = PipelineConfig()
    return _DEFAULT_PIPELINE_CONFIG


# =============================================================================
# Content Hash
# =============================================================================


def content_hash(data: bytes) -> str:
    """Compute SHA-256 content hash of data.

    This mirrors the Rust ContentHash type for content-addressed storage.

    Args:
        data: Raw bytes to hash.

    Returns:
        Hex-encoded SHA-256 hash (64 characters).
    """
    return hashlib.sha256(data).hexdigest()


def shader_hash(wgsl_source: str) -> str:
    """Compute content hash of WGSL shader source.

    Args:
        wgsl_source: WGSL shader source code.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    return content_hash(wgsl_source.encode("utf-8"))


# =============================================================================
# Pipeline Cache Handle
# =============================================================================


@dataclass
class PipelineCacheHandle:
    """Reference to a cached pipeline.

    This is returned by PipelineIntegration.get_or_create_pipeline() and can
    be used to access the pipeline for rendering.

    Attributes:
        id: Unique pipeline ID within the cache.
        shader_hash: Content hash of the shader source.
        config: Pipeline configuration used for compilation.
        source_path: Optional path to source file (for hot-reload).
    """
    id: int
    shader_hash: str
    config: PipelineConfig
    source_path: Optional[str] = None


# =============================================================================
# Shader Cache (Python implementation)
# =============================================================================


@dataclass
class ShaderCacheStats:
    """Statistics for shader cache performance monitoring."""
    hits: int = 0
    misses: int = 0
    cached_modules: int = 0
    tracked_paths: int = 0
    total_source_bytes: int = 0

    def hit_rate(self) -> float:
        """Compute cache hit rate as a percentage [0.0, 100.0]."""
        total = self.hits + self.misses
        if total == 0:
            return 100.0
        return (self.hits / total) * 100.0


class ShaderCache:
    """Content-addressed shader cache (Python implementation).

    This mirrors the Rust ShaderCacheV2 for testing when the native
    backend is not available.

    The cache deduplicates shader sources by their SHA-256 hash.
    Multiple source paths can map to the same hash if they have
    identical content.
    """

    def __init__(self) -> None:
        """Create an empty shader cache."""
        # Maps hash -> compiled shader placeholder (in real impl, wgpu::ShaderModule)
        self._modules: Dict[str, Any] = {}
        # Maps source path -> hash
        self._path_to_hash: Dict[str, str] = {}
        # Maps hash -> list of source paths
        self._hash_to_paths: Dict[str, List[str]] = {}
        # Statistics
        self._stats = ShaderCacheStats()

    def cache_shader(self, wgsl_source: str) -> Tuple[str, str]:
        """Cache a shader from WGSL source.

        Args:
            wgsl_source: WGSL shader source code.

        Returns:
            Tuple of (shader_placeholder, content_hash).
        """
        h = shader_hash(wgsl_source)

        if h in self._modules:
            self._stats.hits += 1
            return (self._modules[h], h)

        # Cache miss: "compile" (in real impl, this calls device.create_shader_module)
        self._stats.misses += 1
        self._stats.total_source_bytes += len(wgsl_source)

        # Placeholder for compiled module
        module = f"ShaderModule({h[:16]})"
        self._modules[h] = module
        self._stats.cached_modules = len(self._modules)

        return (module, h)

    def cache_shader_with_path(
        self, wgsl_source: str, source_path: str
    ) -> Tuple[str, str]:
        """Cache a shader with source path tracking.

        Args:
            wgsl_source: WGSL shader source code.
            source_path: Path to the source file.

        Returns:
            Tuple of (shader_placeholder, content_hash).
        """
        module, h = self.cache_shader(wgsl_source)

        # Track path -> hash
        self._path_to_hash[source_path] = h

        # Track hash -> paths
        if h not in self._hash_to_paths:
            self._hash_to_paths[h] = []
        if source_path not in self._hash_to_paths[h]:
            self._hash_to_paths[h].append(source_path)

        self._stats.tracked_paths = len(self._path_to_hash)

        return (module, h)

    def get(self, h: str) -> Optional[Any]:
        """Get a cached shader module by hash."""
        return self._modules.get(h)

    def hash_for_path(self, path: str) -> Optional[str]:
        """Get the content hash for a source path."""
        return self._path_to_hash.get(path)

    def paths_for_hash(self, h: str) -> Optional[List[str]]:
        """Get all source paths for a content hash."""
        return self._hash_to_paths.get(h)

    def invalidate_path(self, path: str) -> Optional[str]:
        """Invalidate a shader by source path.

        Returns the old hash if the path was tracked.
        """
        old_hash = self._path_to_hash.pop(path, None)
        if old_hash is None:
            return None

        # Remove path from hash_to_paths
        if old_hash in self._hash_to_paths:
            paths = self._hash_to_paths[old_hash]
            if path in paths:
                paths.remove(path)
            if not paths:
                del self._hash_to_paths[old_hash]
                # Remove module if no paths reference it
                self._modules.pop(old_hash, None)
                self._stats.cached_modules = len(self._modules)

        self._stats.tracked_paths = len(self._path_to_hash)
        return old_hash

    def clear(self) -> None:
        """Clear all cached shaders."""
        self._modules.clear()
        self._path_to_hash.clear()
        self._hash_to_paths.clear()
        self._stats.cached_modules = 0
        self._stats.tracked_paths = 0

    @property
    def stats(self) -> ShaderCacheStats:
        """Get cache statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset hit/miss statistics."""
        self._stats.hits = 0
        self._stats.misses = 0

    def __len__(self) -> int:
        return len(self._modules)

    def contains(self, h: str) -> bool:
        return h in self._modules


# =============================================================================
# LRU Pipeline Table (Python implementation)
# =============================================================================


@dataclass
class LruPipelineStats:
    """Statistics for pipeline table performance monitoring."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    invalidations: int = 0
    peak_size: int = 0

    def hit_rate(self) -> float:
        """Compute cache hit rate as a percentage [0.0, 100.0]."""
        total = self.hits + self.misses
        if total == 0:
            return 100.0
        return (self.hits / total) * 100.0


@dataclass
class CachedPipeline:
    """A cached render pipeline."""
    id: int
    shader_hash: str
    config: PipelineConfig
    # In real impl, this would hold wgpu::RenderPipeline and BindGroupLayout
    pipeline_placeholder: str = ""


class LruPipelineTable:
    """LRU-evicted pipeline table (Python implementation).

    This mirrors the Rust LruPipelineTable for testing when the native
    backend is not available.

    Provides get_or_create semantics with automatic LRU eviction when
    the cache exceeds max_size.
    """

    def __init__(self, max_size: int = 64) -> None:
        """Create a new LRU pipeline table.

        Args:
            max_size: Maximum number of pipelines before eviction.

        Raises:
            ValueError: If max_size is 0.
        """
        if max_size == 0:
            raise ValueError("max_size must be greater than 0")

        self._max_size = max_size
        self._next_id = 1
        # Pipelines indexed by ID
        self._pipelines: Dict[int, CachedPipeline] = {}
        # Maps hash -> pipeline ID
        self._hash_to_id: Dict[str, int] = {}
        # LRU order: most recent first (using OrderedDict for O(1) move_to_end)
        self._lru_order: OrderedDict[int, None] = OrderedDict()
        # Shader cache
        self._shader_cache = ShaderCache()
        # Statistics
        self._stats = LruPipelineStats()

    def get_or_create_pipeline(
        self,
        wgsl_source: str,
        config: Optional[PipelineConfig] = None,
        source_path: Optional[str] = None,
    ) -> PipelineCacheHandle:
        """Get or create a pipeline for the given shader source.

        If a pipeline with the same content hash exists, it is returned.
        Otherwise, a new pipeline is compiled and cached.

        Args:
            wgsl_source: WGSL shader source code.
            config: Pipeline configuration (uses defaults if None).
            source_path: Optional source path for hot-reload tracking.

        Returns:
            Handle to the cached pipeline.
        """
        config = config if config is not None else _get_default_config()
        h = shader_hash(wgsl_source)

        # Check for existing pipeline
        if h in self._hash_to_id:
            self._stats.hits += 1
            pipeline_id = self._hash_to_id[h]
            self._touch_lru(pipeline_id)
            pipeline = self._pipelines[pipeline_id]
            return PipelineCacheHandle(
                id=pipeline_id,
                shader_hash=h,
                config=pipeline.config,
                source_path=source_path,
            )

        # Cache miss: compile new pipeline
        self._stats.misses += 1

        # Cache shader
        if source_path:
            self._shader_cache.cache_shader_with_path(wgsl_source, source_path)
        else:
            self._shader_cache.cache_shader(wgsl_source)

        # Allocate ID
        pipeline_id = self._next_id
        self._next_id += 1

        # Create pipeline placeholder
        pipeline = CachedPipeline(
            id=pipeline_id,
            shader_hash=h,
            config=config,
            pipeline_placeholder=f"RenderPipeline({pipeline_id})",
        )

        # Insert with eviction
        self._insert_with_eviction(pipeline_id, h, pipeline)

        return PipelineCacheHandle(
            id=pipeline_id,
            shader_hash=h,
            config=config,
            source_path=source_path,
        )

    def _insert_with_eviction(
        self, pipeline_id: int, h: str, pipeline: CachedPipeline
    ) -> None:
        """Insert a pipeline, evicting LRU entries if necessary."""
        # Evict if at capacity
        while len(self._pipelines) >= self._max_size:
            if not self._lru_order:
                break
            # Pop least recently used (last item)
            evict_id = next(reversed(self._lru_order))
            del self._lru_order[evict_id]
            if evict_id in self._pipelines:
                evicted = self._pipelines.pop(evict_id)
                self._hash_to_id.pop(evicted.shader_hash, None)
                self._stats.evictions += 1

        # Insert
        self._pipelines[pipeline_id] = pipeline
        self._hash_to_id[h] = pipeline_id
        self._lru_order[pipeline_id] = None
        self._lru_order.move_to_end(pipeline_id, last=False)  # Move to front

        # Update peak
        if len(self._pipelines) > self._stats.peak_size:
            self._stats.peak_size = len(self._pipelines)

    def _touch_lru(self, pipeline_id: int) -> None:
        """Move a pipeline to the front of the LRU queue."""
        if pipeline_id in self._lru_order:
            self._lru_order.move_to_end(pipeline_id, last=False)

    def get(self, pipeline_id: int) -> Optional[CachedPipeline]:
        """Get a pipeline by ID (does not update LRU order)."""
        return self._pipelines.get(pipeline_id)

    def get_touch(self, pipeline_id: int) -> Optional[CachedPipeline]:
        """Get a pipeline by ID and update LRU order."""
        if pipeline_id in self._pipelines:
            self._touch_lru(pipeline_id)
            return self._pipelines[pipeline_id]
        return None

    def contains(self, pipeline_id: int) -> bool:
        """Check if a pipeline with the given ID exists."""
        return pipeline_id in self._pipelines

    def contains_hash(self, h: str) -> bool:
        """Check if a pipeline with the given hash exists."""
        return h in self._hash_to_id

    def id_for_hash(self, h: str) -> Optional[int]:
        """Get the pipeline ID for a content hash."""
        return self._hash_to_id.get(h)

    def remove(self, pipeline_id: int) -> bool:
        """Remove a pipeline by ID."""
        if pipeline_id in self._pipelines:
            pipeline = self._pipelines.pop(pipeline_id)
            self._hash_to_id.pop(pipeline.shader_hash, None)
            self._lru_order.pop(pipeline_id, None)
            return True
        return False

    def invalidate_by_hash(self, h: str) -> List[int]:
        """Invalidate all pipelines using the given shader hash.

        Returns the IDs of invalidated pipelines.
        """
        invalidated: List[int] = []
        if h in self._hash_to_id:
            pipeline_id = self._hash_to_id.pop(h)
            if pipeline_id in self._pipelines:
                self._pipelines.pop(pipeline_id)
                self._lru_order.pop(pipeline_id, None)
                self._stats.invalidations += 1
                invalidated.append(pipeline_id)
        return invalidated

    def invalidate_by_path(self, path: str) -> List[int]:
        """Invalidate pipelines by source path.

        Useful for hot-reload: when a file changes, invalidate all
        pipelines compiled from that file.
        """
        old_hash = self._shader_cache.invalidate_path(path)
        if old_hash:
            return self.invalidate_by_hash(old_hash)
        return []

    def clear(self) -> None:
        """Clear all cached pipelines."""
        self._pipelines.clear()
        self._hash_to_id.clear()
        self._lru_order.clear()
        self._shader_cache.clear()

    @property
    def stats(self) -> LruPipelineStats:
        """Get cache statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats.hits = 0
        self._stats.misses = 0
        self._stats.evictions = 0
        self._stats.invalidations = 0

    @property
    def shader_cache(self) -> ShaderCache:
        """Access the underlying shader cache."""
        return self._shader_cache

    @property
    def max_size(self) -> int:
        """Maximum cache size."""
        return self._max_size

    def set_max_size(self, max_size: int) -> None:
        """Set maximum cache size, evicting if necessary."""
        if max_size == 0:
            raise ValueError("max_size must be greater than 0")
        self._max_size = max_size
        # Evict until under limit
        while len(self._pipelines) > self._max_size:
            if not self._lru_order:
                break
            evict_id = next(reversed(self._lru_order))
            del self._lru_order[evict_id]
            if evict_id in self._pipelines:
                evicted = self._pipelines.pop(evict_id)
                self._hash_to_id.pop(evicted.shader_hash, None)
                self._stats.evictions += 1

    def lru_order(self) -> List[int]:
        """Get the LRU order (most recent first)."""
        return list(self._lru_order.keys())

    def __len__(self) -> int:
        return len(self._pipelines)

    def is_empty(self) -> bool:
        return len(self._pipelines) == 0


# =============================================================================
# Pipeline Integration (Main Interface)
# =============================================================================


class PipelineIntegration:
    """Main interface for Rust pipeline integration.

    This class provides the Python-side interface to the Rust pipeline
    caching and hot-reload system. When the Rust backend is available
    (pyo3 feature), it delegates to native code. Otherwise, it uses a
    pure-Python implementation for testing.

    Example::

        integration = PipelineIntegration(max_cache_size=64)

        # Compile and cache a PBR shader
        handle = integration.get_or_create_pipeline(
            wgsl_source=pbr_shader_wgsl,
            config=PipelineConfig(
                vertex_entry="vs_pbr",
                fragment_entry="fs_pbr",
                color_format=ColorFormat.RGBA16_FLOAT,
            ),
            source_path="shaders/pbr.wgsl",
        )

        # Hot-reload: invalidate when shader changes
        invalidated = integration.invalidate_shader("shaders/pbr.wgsl")
    """

    def __init__(
        self,
        max_cache_size: int = 64,
        on_invalidate: Optional[Callable[[List[int]], None]] = None,
    ) -> None:
        """Create a pipeline integration interface.

        Args:
            max_cache_size: Maximum number of cached pipelines.
            on_invalidate: Optional callback when pipelines are invalidated.
        """
        self._table = LruPipelineTable(max_size=max_cache_size)
        self._on_invalidate = on_invalidate
        # Track source path -> pipeline handles
        self._path_handles: Dict[str, List[PipelineCacheHandle]] = {}

    def get_or_create_pipeline(
        self,
        wgsl_source: str,
        config: Optional[PipelineConfig] = None,
        source_path: Optional[str] = None,
    ) -> PipelineCacheHandle:
        """Get or create a pipeline for the given shader source.

        Args:
            wgsl_source: WGSL shader source code.
            config: Pipeline configuration.
            source_path: Optional source path for hot-reload tracking.

        Returns:
            Handle to the cached pipeline.
        """
        handle = self._table.get_or_create_pipeline(
            wgsl_source=wgsl_source,
            config=config,
            source_path=source_path,
        )

        # Track path -> handles
        if source_path:
            if source_path not in self._path_handles:
                self._path_handles[source_path] = []
            self._path_handles[source_path].append(handle)

        return handle

    def invalidate_shader(self, path: str) -> List[int]:
        """Invalidate all pipelines compiled from the given source path.

        Called when a shader file changes and needs recompilation.

        Args:
            path: Path to the shader source file.

        Returns:
            List of invalidated pipeline IDs.
        """
        invalidated = self._table.invalidate_by_path(path)
        self._path_handles.pop(path, None)

        if invalidated and self._on_invalidate:
            self._on_invalidate(invalidated)

        return invalidated

    def invalidate_by_hash(self, h: str) -> List[int]:
        """Invalidate all pipelines using the given content hash.

        Args:
            h: Content hash of the shader source.

        Returns:
            List of invalidated pipeline IDs.
        """
        invalidated = self._table.invalidate_by_hash(h)

        if invalidated and self._on_invalidate:
            self._on_invalidate(invalidated)

        return invalidated

    def get_pipeline(self, pipeline_id: int) -> Optional[CachedPipeline]:
        """Get a cached pipeline by ID."""
        return self._table.get(pipeline_id)

    def clear(self) -> None:
        """Clear all cached pipelines."""
        self._table.clear()
        self._path_handles.clear()

    @property
    def stats(self) -> LruPipelineStats:
        """Get pipeline cache statistics."""
        return self._table.stats

    @property
    def shader_stats(self) -> ShaderCacheStats:
        """Get shader cache statistics."""
        return self._table.shader_cache.stats

    def cache_hit_rate(self) -> float:
        """Get overall cache hit rate percentage."""
        return self._table.stats.hit_rate()

    def shader_hit_rate(self) -> float:
        """Get shader cache hit rate percentage."""
        return self._table.shader_cache.stats.hit_rate()

    @property
    def max_cache_size(self) -> int:
        """Maximum pipeline cache size."""
        return self._table.max_size

    def set_max_cache_size(self, size: int) -> None:
        """Set maximum cache size."""
        self._table.set_max_size(size)

    def __len__(self) -> int:
        """Number of cached pipelines."""
        return len(self._table)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Types
    "ColorFormat",
    "CullMode",
    "BlendMode",
    "PipelineConfig",
    "PipelineCacheHandle",
    "CachedPipeline",
    # Statistics
    "ShaderCacheStats",
    "LruPipelineStats",
    # Caches
    "ShaderCache",
    "LruPipelineTable",
    # Main interface
    "PipelineIntegration",
    # Utilities
    "content_hash",
    "shader_hash",
]
