"""
SDF WGSL Compilation Cache (T-DEMO-2.13).

Implements cached WGSL compilation with dirty tracking invalidation for the
demoscene DSL. Integrates with the Tracker system from sdf_ast.py to
automatically invalidate cached entries when scene nodes are modified.

Features:
    - LRU cache for compiled WGSL keyed by AST structure hash
    - Integration with Tracker for dirty-based invalidation
    - Optimization level included in cache key
    - Hit/miss statistics for monitoring cache effectiveness
    - CachedSDFCompiler wrapper for seamless integration

Usage:
    >>> from engine.rendering.demoscene.sdf_cache import CachedSDFCompiler, WGSLCache
    >>> from engine.rendering.demoscene.sdf_ast import SceneNode, SphereNode
    >>>
    >>> # Create scene
    >>> sphere = SphereNode(radius=1.0)
    >>> scene = SceneNode(root=sphere, name="test")
    >>>
    >>> # Create cached compiler
    >>> compiler = CachedSDFCompiler()
    >>> wgsl = compiler.compile(scene)  # First call: compiles
    >>> wgsl2 = compiler.compile(scene)  # Second call: from cache
    >>>
    >>> # Modify scene - invalidates cache
    >>> sphere.radius = 2.0
    >>> sphere.tracker.mark_dirty("radius")
    >>> wgsl3 = compiler.compile(scene)  # Recompiles
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

from .sdf_ast import (
    SDFNode,
    SceneNode,
    PrimitiveNode,
    CombinatorNode,
    DomainOpNode,
    SphereNode,
    BoxNode,
    TorusNode,
    CylinderNode,
    ConeNode,
    PlaneNode,
    CapsuleNode,
    EllipsoidNode,
    BoxFrameNode,
    RoundedBoxNode,
    OctahedronNode,
    PyramidNode,
    UnionNode,
    IntersectionNode,
    SubtractionNode,
    SmoothUnionNode,
    SmoothIntersectionNode,
    SmoothSubtractionNode,
    DisplacedNode,
    RepeatNode,
    MirrorNode,
    KIFSNode,
    TwistNode,
    BendNode,
    StretchNode,
    MaterialNode,
    CameraNode,
    LightNode,
    RenderSettingsNode,
    Vec3,
    Axis,
    Tracker,
)


__all__ = [
    "WGSLCache",
    "CachedSDFCompiler",
    "CacheStats",
    "CacheEntry",
    "OptimizationLevel",
    "sdf_node_hash",
    "is_cache_valid",
]


# =============================================================================
# OPTIMIZATION LEVELS
# =============================================================================

class OptimizationLevel(IntEnum):
    """Optimization level for WGSL compilation."""
    NONE = 0       # No optimization
    FAST = 1       # Quick optimizations (constant folding, DCE)
    DEFAULT = 2    # Standard optimization passes
    AGGRESSIVE = 3 # All passes, multiple iterations


# =============================================================================
# AST HASHING FOR SDFNode
# =============================================================================

def sdf_node_hash(node: SDFNode, include_version: bool = False) -> int:
    """
    Compute a structural hash for an SDFNode tree.

    This hash captures the complete structure and values of the AST,
    enabling cache key computation for WGSL caching.

    Args:
        node: The SDFNode to hash.
        include_version: If True, include version numbers in hash
                        (makes hash sensitive to any modification).

    Returns:
        Integer hash value.

    Example:
        >>> sphere = SphereNode(radius=1.0)
        >>> h1 = sdf_node_hash(sphere)
        >>> sphere2 = SphereNode(radius=1.0)
        >>> h2 = sdf_node_hash(sphere2)
        >>> h1 == h2  # Same structure and values
        True
    """
    return _hash_node(node, include_version)


def _hash_node(node: SDFNode, include_version: bool = False) -> int:
    """Recursive hash implementation for SDFNode."""

    # Base components: node type and version (optional)
    components: List[Any] = [type(node).__name__]

    if include_version:
        components.append(node.tracker.version)

    # Hash based on specific node type
    if isinstance(node, SceneNode):
        components.extend([
            _hash_node(node.root, include_version),
            _hash_node(node.camera, include_version),
            _hash_node(node.render_settings, include_version),
            tuple(_hash_node(light, include_version) for light in node.lights),
            tuple(_hash_node(mat, include_version) for mat in node.materials),
            node.name,
        ])

    # Primitives
    elif isinstance(node, SphereNode):
        components.extend([node.radius, _hash_vec3(node.position)])
    elif isinstance(node, BoxNode):
        components.extend([_hash_vec3(node.half_extents), _hash_vec3(node.position)])
    elif isinstance(node, TorusNode):
        components.extend([node.major_radius, node.minor_radius, _hash_vec3(node.position)])
    elif isinstance(node, CylinderNode):
        components.extend([node.radius, node.height, _hash_vec3(node.position)])
    elif isinstance(node, ConeNode):
        components.extend([node.angle, node.height, _hash_vec3(node.position)])
    elif isinstance(node, PlaneNode):
        components.extend([_hash_vec3(node.normal), node.distance, _hash_vec3(node.position)])
    elif isinstance(node, CapsuleNode):
        components.extend([
            _hash_vec3(node.endpoint_a),
            _hash_vec3(node.endpoint_b),
            node.radius,
            _hash_vec3(node.position),
        ])
    elif isinstance(node, EllipsoidNode):
        components.extend([_hash_vec3(node.radii), _hash_vec3(node.position)])
    elif isinstance(node, BoxFrameNode):
        components.extend([_hash_vec3(node.half_extents), node.edge_thickness, _hash_vec3(node.position)])
    elif isinstance(node, RoundedBoxNode):
        components.extend([_hash_vec3(node.half_extents), node.corner_radius, _hash_vec3(node.position)])
    elif isinstance(node, OctahedronNode):
        components.extend([node.size, _hash_vec3(node.position)])
    elif isinstance(node, PyramidNode):
        components.extend([node.height, _hash_vec3(node.position)])

    # Combinators
    elif isinstance(node, (UnionNode, IntersectionNode, SubtractionNode)):
        components.extend([
            _hash_node(node.left, include_version),
            _hash_node(node.right, include_version),
        ])
    elif isinstance(node, (SmoothUnionNode, SmoothIntersectionNode, SmoothSubtractionNode)):
        components.extend([
            _hash_node(node.left, include_version),
            _hash_node(node.right, include_version),
            node.k,
        ])
    elif isinstance(node, DisplacedNode):
        components.extend([
            _hash_node(node.child, include_version),
            node.amplitude,
            node.frequency,
        ])

    # Domain Operations
    elif isinstance(node, RepeatNode):
        components.extend([
            _hash_node(node.child, include_version),
            _hash_vec3(node.cell_size),
        ])
    elif isinstance(node, MirrorNode):
        components.extend([
            _hash_node(node.child, include_version),
            node.axis.value,
        ])
    elif isinstance(node, KIFSNode):
        components.extend([
            _hash_node(node.child, include_version),
            node.iterations,
            node.scale,
            _hash_vec3(node.offset),
        ])
    elif isinstance(node, TwistNode):
        components.extend([
            _hash_node(node.child, include_version),
            node.axis.value,
            node.rate,
        ])
    elif isinstance(node, BendNode):
        components.extend([
            _hash_node(node.child, include_version),
            node.axis.value,
            node.radius,
        ])
    elif isinstance(node, StretchNode):
        components.extend([
            _hash_node(node.child, include_version),
            node.axis.value,
            node.scale,
        ])

    # Scene components
    elif isinstance(node, MaterialNode):
        components.extend([
            _hash_vec3(node.color),
            node.metallic,
            node.roughness,
            _hash_vec3(node.emission),
            node.material_id,
        ])
    elif isinstance(node, CameraNode):
        components.extend([
            _hash_vec3(node.origin),
            _hash_vec3(node.look_at),
            _hash_vec3(node.up),
            node.fov,
            node.aspect_ratio,
            node.aperture,
            node.focal_distance,
        ])
    elif isinstance(node, LightNode):
        components.extend([
            _hash_vec3(node.position),
            _hash_vec3(node.color),
            node.intensity,
        ])
    elif isinstance(node, RenderSettingsNode):
        components.extend([
            node.width,
            node.height,
            node.max_steps,
            node.max_distance,
            node.epsilon,
            node.workgroup_size,
        ])

    # Generic primitive fallback
    elif isinstance(node, PrimitiveNode):
        components.append(_hash_vec3(node.position))

    # Generic domain op fallback
    elif isinstance(node, DomainOpNode):
        components.append(_hash_node(node.child, include_version))

    # Generic combinator fallback
    elif isinstance(node, CombinatorNode):
        components.extend([
            _hash_node(node.left, include_version),
            _hash_node(node.right, include_version),
        ])

    return hash(tuple(components))


def _hash_vec3(v: Vec3) -> Tuple[float, float, float]:
    """Convert Vec3 to hashable tuple."""
    return (v.x, v.y, v.z)


def _compute_cache_key(
    scene: SceneNode,
    optimization_level: OptimizationLevel,
) -> str:
    """
    Compute a cache key string for a scene.

    Combines structural hash with optimization level.

    Args:
        scene: The scene to compute key for.
        optimization_level: The optimization level.

    Returns:
        String cache key.
    """
    struct_hash = sdf_node_hash(scene, include_version=False)
    return f"{struct_hash:016x}_{optimization_level.value}"


# =============================================================================
# CACHE STATISTICS
# =============================================================================

@dataclass
class CacheStats:
    """Statistics about cache performance."""
    hits: int = 0
    misses: int = 0
    invalidations: int = 0
    evictions: int = 0
    total_compile_time_ms: float = 0.0
    total_cache_time_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate as percentage."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100.0

    @property
    def total_requests(self) -> int:
        """Total number of cache requests."""
        return self.hits + self.misses

    @property
    def avg_compile_time_ms(self) -> float:
        """Average time per compilation."""
        if self.misses == 0:
            return 0.0
        return self.total_compile_time_ms / self.misses

    @property
    def time_saved_ms(self) -> float:
        """Estimated time saved by cache hits."""
        if self.misses == 0:
            return 0.0
        avg_compile = self.avg_compile_time_ms
        return self.hits * avg_compile - self.total_cache_time_ms

    def reset(self) -> None:
        """Reset all statistics to zero."""
        self.hits = 0
        self.misses = 0
        self.invalidations = 0
        self.evictions = 0
        self.total_compile_time_ms = 0.0
        self.total_cache_time_ms = 0.0

    def __repr__(self) -> str:
        return (
            f"CacheStats(hits={self.hits}, misses={self.misses}, "
            f"hit_rate={self.hit_rate:.1f}%, invalidations={self.invalidations}, "
            f"evictions={self.evictions})"
        )


# =============================================================================
# CACHE ENTRY
# =============================================================================

@dataclass
class CacheEntry:
    """A single cache entry with metadata."""
    wgsl: str
    ast_hash: int
    optimization_level: OptimizationLevel
    created_at: float
    last_accessed: float
    access_count: int = 1
    compile_time_ms: float = 0.0

    def touch(self) -> None:
        """Update access metadata."""
        self.last_accessed = time.time()
        self.access_count += 1


# =============================================================================
# WGSL CACHE
# =============================================================================

class WGSLCache:
    """
    LRU cache for compiled WGSL output.

    Caches compiled WGSL shader code keyed by AST structure hash and
    optimization level. Uses LRU eviction when capacity is exceeded.

    Thread-safe for concurrent access.

    Attributes:
        max_size: Maximum number of entries to cache.
        stats: Cache performance statistics.

    Example:
        >>> cache = WGSLCache(max_size=100)
        >>> scene = SceneNode(root=SphereNode(radius=1.0))
        >>>
        >>> # Check if cached
        >>> wgsl = cache.get(scene)
        >>> if wgsl is None:
        ...     wgsl = compile_scene(scene)
        ...     cache.put(scene, wgsl)
        >>>
        >>> print(cache.stats())
    """

    def __init__(
        self,
        max_size: int = 256,
        optimization_level: OptimizationLevel = OptimizationLevel.DEFAULT,
    ) -> None:
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of entries (default: 256).
            optimization_level: Default optimization level for cache keys.
        """
        self._max_size = max_size
        self._default_opt_level = optimization_level
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats()
        self._lock = threading.RLock()

        # Track which scenes are in the cache (for invalidation)
        self._scene_to_keys: Dict[int, Set[str]] = {}

    @property
    def max_size(self) -> int:
        """Maximum cache size."""
        return self._max_size

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        with self._lock:
            return len(self._cache)

    def get(
        self,
        scene: SceneNode,
        optimization_level: Optional[OptimizationLevel] = None,
    ) -> Optional[str]:
        """
        Get cached WGSL for a scene if valid.

        Returns None if:
        - Entry not in cache
        - Scene or any child is marked dirty

        Args:
            scene: The scene to look up.
            optimization_level: Optimization level (defaults to cache default).

        Returns:
            Cached WGSL string, or None if not found/invalid.

        Example:
            >>> wgsl = cache.get(scene)
            >>> if wgsl is not None:
            ...     print("Cache hit!")
        """
        opt_level = optimization_level if optimization_level is not None else self._default_opt_level
        cache_key = _compute_cache_key(scene, opt_level)

        start = time.perf_counter()

        with self._lock:
            entry = self._cache.get(cache_key)

            if entry is None:
                self._stats.misses += 1
                return None

            # Check if scene is dirty
            if not self.is_cache_valid(scene):
                self._stats.misses += 1
                self._stats.invalidations += 1
                self._remove_entry(cache_key)
                return None

            # Move to end (LRU)
            self._cache.move_to_end(cache_key)
            entry.touch()

            self._stats.hits += 1
            elapsed = (time.perf_counter() - start) * 1000
            self._stats.total_cache_time_ms += elapsed

            return entry.wgsl

    def put(
        self,
        scene: SceneNode,
        wgsl: str,
        optimization_level: Optional[OptimizationLevel] = None,
        compile_time_ms: float = 0.0,
    ) -> None:
        """
        Store compiled WGSL in the cache.

        Args:
            scene: The scene that was compiled.
            wgsl: The compiled WGSL code.
            optimization_level: Optimization level used.
            compile_time_ms: Time taken to compile (for statistics).

        Example:
            >>> start = time.perf_counter()
            >>> wgsl = compile_scene(scene)
            >>> elapsed = (time.perf_counter() - start) * 1000
            >>> cache.put(scene, wgsl, compile_time_ms=elapsed)
        """
        opt_level = optimization_level if optimization_level is not None else self._default_opt_level
        cache_key = _compute_cache_key(scene, opt_level)
        ast_hash = sdf_node_hash(scene)

        now = time.time()
        entry = CacheEntry(
            wgsl=wgsl,
            ast_hash=ast_hash,
            optimization_level=opt_level,
            created_at=now,
            last_accessed=now,
            compile_time_ms=compile_time_ms,
        )

        with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self._max_size:
                self._evict_oldest()

            self._cache[cache_key] = entry

            # Track scene ID for invalidation
            scene_id = id(scene)
            if scene_id not in self._scene_to_keys:
                self._scene_to_keys[scene_id] = set()
            self._scene_to_keys[scene_id].add(cache_key)

            self._stats.total_compile_time_ms += compile_time_ms

    def invalidate(
        self,
        scene: SceneNode,
        optimization_level: Optional[OptimizationLevel] = None,
    ) -> bool:
        """
        Invalidate cached entry for a scene.

        Args:
            scene: The scene to invalidate.
            optimization_level: If specified, only invalidate for this level.
                               If None, invalidate all levels for this scene.

        Returns:
            True if any entry was invalidated.

        Example:
            >>> scene.root.radius = 2.0
            >>> cache.invalidate(scene)
        """
        with self._lock:
            if optimization_level is not None:
                cache_key = _compute_cache_key(scene, optimization_level)
                if cache_key in self._cache:
                    self._remove_entry(cache_key)
                    self._stats.invalidations += 1
                    return True
                return False

            # Invalidate all optimization levels
            scene_id = id(scene)
            keys_to_remove = self._scene_to_keys.get(scene_id, set()).copy()

            invalidated = False
            for key in keys_to_remove:
                if key in self._cache:
                    self._remove_entry(key)
                    self._stats.invalidations += 1
                    invalidated = True

            return invalidated

    def is_cache_valid(self, scene: SceneNode) -> bool:
        """
        Check if a scene's cached version would be valid.

        A cache entry is invalid if the scene or any descendant
        has dirty flags set (modified since last compilation).

        Args:
            scene: The scene to check.

        Returns:
            True if cache would be valid (no dirty flags).

        Example:
            >>> if cache.is_cache_valid(scene):
            ...     wgsl = cache.get(scene)
        """
        return not scene.tracker.is_dirty

    def clear(self) -> None:
        """
        Clear all cached entries.

        Example:
            >>> cache.clear()
            >>> assert cache.size == 0
        """
        with self._lock:
            self._cache.clear()
            self._scene_to_keys.clear()

    def stats(self) -> CacheStats:
        """
        Get cache statistics.

        Returns:
            CacheStats object with hit/miss counts and rates.

        Example:
            >>> stats = cache.stats()
            >>> print(f"Hit rate: {stats.hit_rate:.1f}%")
        """
        with self._lock:
            # Return a copy to avoid threading issues
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                invalidations=self._stats.invalidations,
                evictions=self._stats.evictions,
                total_compile_time_ms=self._stats.total_compile_time_ms,
                total_cache_time_ms=self._stats.total_cache_time_ms,
            )

    def reset_stats(self) -> None:
        """Reset statistics to zero."""
        with self._lock:
            self._stats.reset()

    def _evict_oldest(self) -> None:
        """Evict the oldest (least recently used) entry."""
        if self._cache:
            oldest_key, _ = self._cache.popitem(last=False)
            self._cleanup_scene_key(oldest_key)
            self._stats.evictions += 1

    def _remove_entry(self, cache_key: str) -> None:
        """Remove a specific cache entry."""
        if cache_key in self._cache:
            del self._cache[cache_key]
            self._cleanup_scene_key(cache_key)

    def _cleanup_scene_key(self, cache_key: str) -> None:
        """Remove cache key from scene tracking."""
        for scene_id, keys in list(self._scene_to_keys.items()):
            if cache_key in keys:
                keys.discard(cache_key)
                if not keys:
                    del self._scene_to_keys[scene_id]
                break

    def __len__(self) -> int:
        """Return number of cached entries."""
        return self.size

    def __contains__(self, scene: SceneNode) -> bool:
        """Check if scene is in cache (any optimization level)."""
        with self._lock:
            scene_id = id(scene)
            return scene_id in self._scene_to_keys and bool(self._scene_to_keys[scene_id])

    def __repr__(self) -> str:
        return f"WGSLCache(size={self.size}/{self.max_size}, {self.stats()})"


# =============================================================================
# VALIDITY CHECK HELPER
# =============================================================================

def is_cache_valid(scene: SceneNode) -> bool:
    """
    Check if a scene's cache entry would be valid.

    Convenience function that checks the dirty state of a scene
    and all its children.

    Args:
        scene: The scene to check.

    Returns:
        True if no dirty flags are set (cache would be valid).

    Example:
        >>> scene = SceneNode(root=SphereNode(radius=1.0))
        >>> scene.tracker.clear_recursive()  # Clear initial dirty flags
        >>> assert is_cache_valid(scene) == True
        >>> scene.root.radius = 2.0
        >>> scene.root.tracker.mark_dirty("radius")
        >>> assert is_cache_valid(scene) == False
    """
    return not scene.tracker.is_dirty


# =============================================================================
# STUB COMPILER (for testing/fallback)
# =============================================================================

def _stub_compile(scene: SceneNode, name: str = "") -> str:
    """
    Stub WGSL compiler for testing.

    Generates a simple WGSL representation of the scene.
    In production, this would be replaced by the actual codegen.
    """
    lines = [
        "// WGSL generated by sdf_cache (stub)",
        f"// Scene: {name or scene.name or 'unnamed'}",
        "",
        "fn sd_scene(p: vec3<f32>) -> f32 {",
    ]

    # Generate stub for root SDF
    root_label = scene.root.label()
    lines.append(f"    // Root: {root_label}")
    lines.append("    return 0.0;")
    lines.append("}")

    return "\n".join(lines)


# =============================================================================
# CACHED SDF COMPILER
# =============================================================================

class CachedSDFCompiler:
    """
    WGSL compiler with integrated caching.

    Wraps a WGSL code generator with automatic caching and dirty tracking.
    Clears dirty flags on the scene tree after successful compilation.

    Attributes:
        cache: The underlying WGSLCache instance.
        optimization_level: Default optimization level.

    Example:
        >>> compiler = CachedSDFCompiler()
        >>>
        >>> # First compilation
        >>> wgsl = compiler.compile(scene)
        >>>
        >>> # Second call - from cache (fast)
        >>> wgsl2 = compiler.compile(scene)
        >>>
        >>> # Modify scene
        >>> scene.root.radius = 2.0
        >>> scene.root.tracker.mark_dirty("radius")
        >>>
        >>> # Recompiles (cache miss due to dirty)
        >>> wgsl3 = compiler.compile(scene)
        >>>
        >>> # Check stats
        >>> print(compiler.stats())
    """

    def __init__(
        self,
        cache: Optional[WGSLCache] = None,
        optimization_level: OptimizationLevel = OptimizationLevel.DEFAULT,
        max_cache_size: int = 256,
        compile_func: Optional[Callable[[SceneNode, str], str]] = None,
    ) -> None:
        """
        Initialize the cached compiler.

        Args:
            cache: Existing cache to use, or None to create a new one.
            optimization_level: Default optimization level.
            max_cache_size: Max cache size if creating a new cache.
            compile_func: Custom compile function. If None, uses stub.
                         Signature: (scene: SceneNode, name: str) -> str
        """
        self._cache = cache or WGSLCache(
            max_size=max_cache_size,
            optimization_level=optimization_level,
        )
        self._optimization_level = optimization_level
        self._compile_func = compile_func or _stub_compile

    @property
    def cache(self) -> WGSLCache:
        """Access the underlying cache."""
        return self._cache

    @property
    def optimization_level(self) -> OptimizationLevel:
        """Current optimization level."""
        return self._optimization_level

    @optimization_level.setter
    def optimization_level(self, level: OptimizationLevel) -> None:
        """Set the optimization level."""
        self._optimization_level = level

    def compile(
        self,
        scene: SceneNode,
        name: Optional[str] = None,
        optimization_level: Optional[OptimizationLevel] = None,
        force_recompile: bool = False,
    ) -> str:
        """
        Compile a scene to WGSL, using cache if available.

        Args:
            scene: The scene to compile.
            name: Optional name for the generated shader.
            optimization_level: Override optimization level.
            force_recompile: If True, bypass cache and recompile.

        Returns:
            Compiled WGSL code.

        Example:
            >>> wgsl = compiler.compile(scene)
            >>> wgsl_optimized = compiler.compile(scene, optimization_level=OptimizationLevel.AGGRESSIVE)
        """
        opt_level = optimization_level if optimization_level is not None else self._optimization_level
        scene_name = name or scene.name or ""

        # Try cache first (unless forced recompile)
        if not force_recompile:
            cached = self._cache.get(scene, opt_level)
            if cached is not None:
                return cached

        # Compile
        start = time.perf_counter()
        wgsl = self._compile_func(scene, scene_name)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Store in cache
        self._cache.put(scene, wgsl, opt_level, compile_time_ms=elapsed_ms)

        # Clear dirty flags after successful compilation
        scene.tracker.clear_recursive()

        return wgsl

    def invalidate(
        self,
        scene: SceneNode,
        optimization_level: Optional[OptimizationLevel] = None,
    ) -> bool:
        """
        Invalidate cached entry for a scene.

        Args:
            scene: The scene to invalidate.
            optimization_level: Specific level to invalidate, or all if None.

        Returns:
            True if any entry was invalidated.
        """
        return self._cache.invalidate(scene, optimization_level)

    def is_cache_valid(self, scene: SceneNode) -> bool:
        """
        Check if cache entry for scene is valid.

        Args:
            scene: The scene to check.

        Returns:
            True if cache entry would be valid.
        """
        return self._cache.is_cache_valid(scene)

    def clear_cache(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._cache.stats()

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self._cache.reset_stats()

    def __repr__(self) -> str:
        stats = self.stats()
        return (
            f"CachedSDFCompiler(opt_level={self._optimization_level.name}, "
            f"cache_size={self._cache.size}, hit_rate={stats.hit_rate:.1f}%)"
        )


# =============================================================================
# INTEGRATION WITH REAL CODEGEN (optional)
# =============================================================================

def create_wgsl_compile_func():
    """
    Create a compile function that uses the real WGSL codegen.

    Returns a function that converts SceneNode to SceneGraph and generates WGSL.
    Falls back to stub if codegen not available.

    Returns:
        Compile function: (scene: SceneNode, name: str) -> str
    """
    try:
        from .wgsl_codegen import generate_wgsl
        from .ast_nodes import (
            SceneGraph as AstSceneGraph,
            SphereNode as AstSphereNode,
            BoxNode as AstBoxNode,
            PositionNode,
            FloatNode,
            Vec3Node as AstVec3Node,
        )

        def _convert_and_compile(scene: SceneNode, name: str) -> str:
            """Convert SceneNode to SceneGraph and compile."""
            # Note: This is a simplified conversion
            # A full implementation would convert the entire tree
            primitives = []

            # Convert root to AST primitives (simplified)
            root = scene.root
            if isinstance(root, SphereNode):
                primitives.append(
                    AstSphereNode(PositionNode(), FloatNode(root.radius))
                )
            elif isinstance(root, BoxNode):
                primitives.append(
                    AstBoxNode(
                        PositionNode(),
                        AstVec3Node(
                            root.half_extents.x,
                            root.half_extents.y,
                            root.half_extents.z,
                        )
                    )
                )
            else:
                # Fallback: create a default sphere
                primitives.append(
                    AstSphereNode(PositionNode(), FloatNode(1.0))
                )

            graph = AstSceneGraph(
                primitives=tuple(primitives),
                pipeline=(),
                name=name or scene.name,
            )

            return generate_wgsl(graph, name=name or scene.name)

        return _convert_and_compile

    except ImportError:
        return _stub_compile


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_cached_compiler(
    max_cache_size: int = 256,
    optimization_level: OptimizationLevel = OptimizationLevel.DEFAULT,
    use_real_codegen: bool = False,
) -> CachedSDFCompiler:
    """
    Create a CachedSDFCompiler with standard settings.

    Args:
        max_cache_size: Maximum cache entries.
        optimization_level: Default optimization level.
        use_real_codegen: If True, try to use real WGSL codegen.

    Returns:
        Configured CachedSDFCompiler instance.

    Example:
        >>> compiler = create_cached_compiler(max_cache_size=512)
        >>> wgsl = compiler.compile(scene)
    """
    compile_func = create_wgsl_compile_func() if use_real_codegen else _stub_compile

    return CachedSDFCompiler(
        cache=None,
        optimization_level=optimization_level,
        max_cache_size=max_cache_size,
        compile_func=compile_func,
    )
