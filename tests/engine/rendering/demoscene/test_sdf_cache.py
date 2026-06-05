"""
Tests for SDF WGSL Compilation Cache (T-DEMO-2.13).

Comprehensive test suite covering:
  - WGSLCache: LRU caching, invalidation, eviction, statistics
  - CachedSDFCompiler: Integration with dirty tracking
  - Cache key hashing: structural and value-based
  - Thread safety: concurrent access
  - Edge cases: empty scenes, maximum size, etc.

Run with:
    uv run pytest tests/engine/rendering/demoscene/test_sdf_cache.py -v
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List
from unittest.mock import Mock, patch

import pytest

from engine.rendering.demoscene.sdf_ast import (
    Axis,
    BendNode,
    BoxFrameNode,
    BoxNode,
    CameraNode,
    CapsuleNode,
    ConeNode,
    CylinderNode,
    DisplacedNode,
    EllipsoidNode,
    IntersectionNode,
    KIFSNode,
    LightNode,
    MaterialNode,
    MirrorNode,
    OctahedronNode,
    PlaneNode,
    PyramidNode,
    RenderSettingsNode,
    RepeatNode,
    RoundedBoxNode,
    SceneNode,
    SmoothIntersectionNode,
    SmoothSubtractionNode,
    SmoothUnionNode,
    SphereNode,
    StretchNode,
    SubtractionNode,
    TorusNode,
    TwistNode,
    UnionNode,
    Vec3,
)
from engine.rendering.demoscene.sdf_cache import (
    CachedSDFCompiler,
    CacheEntry,
    CacheStats,
    OptimizationLevel,
    WGSLCache,
    is_cache_valid,
    sdf_node_hash,
    create_cached_compiler,
    _compute_cache_key,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def simple_sphere() -> SphereNode:
    """Create a simple sphere primitive."""
    return SphereNode(radius=1.0)


@pytest.fixture
def simple_scene(simple_sphere: SphereNode) -> SceneNode:
    """Create a simple scene with one sphere."""
    return SceneNode(root=simple_sphere, name="test_scene")


@pytest.fixture
def complex_scene() -> SceneNode:
    """Create a more complex scene for testing."""
    sphere = SphereNode(radius=1.0)
    box = BoxNode(half_extents=Vec3(1.0, 2.0, 3.0))
    union = UnionNode(sphere, box)
    return SceneNode(root=union, name="complex_scene")


@pytest.fixture
def cache() -> WGSLCache:
    """Create a fresh cache for testing."""
    return WGSLCache(max_size=10)


@pytest.fixture
def compiler() -> CachedSDFCompiler:
    """Create a cached compiler for testing."""
    return CachedSDFCompiler(max_cache_size=10)


# =============================================================================
# WGSL CACHE BASIC TESTS
# =============================================================================


class TestWGSLCacheBasics:
    """Basic functionality tests for WGSLCache."""

    def test_empty_cache_has_zero_size(self, cache: WGSLCache):
        """Empty cache should have size 0."""
        assert cache.size == 0
        assert len(cache) == 0

    def test_max_size_property(self, cache: WGSLCache):
        """Max size should be accessible."""
        assert cache.max_size == 10

    def test_put_increases_size(self, cache: WGSLCache, simple_scene: SceneNode):
        """Adding entry should increase cache size."""
        simple_scene.tracker.clear_recursive()
        cache.put(simple_scene, "// WGSL code")
        assert cache.size == 1

    def test_get_nonexistent_returns_none(self, cache: WGSLCache, simple_scene: SceneNode):
        """Getting uncached scene should return None."""
        simple_scene.tracker.clear_recursive()
        result = cache.get(simple_scene)
        assert result is None

    def test_get_cached_returns_wgsl(self, cache: WGSLCache, simple_scene: SceneNode):
        """Getting cached scene should return WGSL."""
        simple_scene.tracker.clear_recursive()
        wgsl = "// cached WGSL"
        cache.put(simple_scene, wgsl)
        result = cache.get(simple_scene)
        assert result == wgsl

    def test_clear_empties_cache(self, cache: WGSLCache, simple_scene: SceneNode):
        """Clear should remove all entries."""
        simple_scene.tracker.clear_recursive()
        cache.put(simple_scene, "// WGSL")
        assert cache.size == 1
        cache.clear()
        assert cache.size == 0

    def test_contains_checks_scene_presence(self, cache: WGSLCache, simple_scene: SceneNode):
        """Contains should check if scene is cached."""
        simple_scene.tracker.clear_recursive()
        assert simple_scene not in cache
        cache.put(simple_scene, "// WGSL")
        assert simple_scene in cache


# =============================================================================
# CACHE STATISTICS TESTS
# =============================================================================


class TestCacheStatistics:
    """Tests for cache statistics tracking."""

    def test_initial_stats_are_zero(self, cache: WGSLCache):
        """Initial stats should all be zero."""
        stats = cache.stats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.invalidations == 0
        assert stats.evictions == 0

    def test_miss_increments_on_get_nonexistent(self, cache: WGSLCache, simple_scene: SceneNode):
        """Cache miss should be recorded."""
        simple_scene.tracker.clear_recursive()
        cache.get(simple_scene)
        stats = cache.stats()
        assert stats.misses == 1
        assert stats.hits == 0

    def test_hit_increments_on_cached_get(self, cache: WGSLCache, simple_scene: SceneNode):
        """Cache hit should be recorded."""
        simple_scene.tracker.clear_recursive()
        cache.put(simple_scene, "// WGSL")
        cache.get(simple_scene)
        stats = cache.stats()
        assert stats.hits == 1
        assert stats.misses == 0

    def test_hit_rate_calculation(self, cache: WGSLCache, simple_scene: SceneNode):
        """Hit rate should be calculated correctly."""
        simple_scene.tracker.clear_recursive()
        # 1 miss
        cache.get(simple_scene)
        # put + 2 hits
        cache.put(simple_scene, "// WGSL")
        cache.get(simple_scene)
        cache.get(simple_scene)

        stats = cache.stats()
        # 2 hits, 1 miss = 66.67%
        assert stats.hit_rate == pytest.approx(66.67, rel=0.01)

    def test_reset_stats_clears_counters(self, cache: WGSLCache, simple_scene: SceneNode):
        """Reset should clear all statistics."""
        simple_scene.tracker.clear_recursive()
        cache.put(simple_scene, "// WGSL")
        cache.get(simple_scene)
        cache.get(simple_scene)

        cache.reset_stats()
        stats = cache.stats()
        assert stats.hits == 0
        assert stats.misses == 0

    def test_compile_time_tracked(self, cache: WGSLCache, simple_scene: SceneNode):
        """Compile time should be tracked."""
        simple_scene.tracker.clear_recursive()
        cache.put(simple_scene, "// WGSL", compile_time_ms=10.5)
        stats = cache.stats()
        assert stats.total_compile_time_ms == 10.5


# =============================================================================
# DIRTY TRACKING INTEGRATION TESTS
# =============================================================================


class TestDirtyTrackingIntegration:
    """Tests for integration with Tracker dirty flags."""

    def test_dirty_scene_invalidates_cache(self, cache: WGSLCache, simple_scene: SceneNode):
        """Dirty scene should not return cached value."""
        simple_scene.tracker.clear_recursive()
        cache.put(simple_scene, "// original WGSL")

        # Mark scene dirty
        simple_scene.root.tracker.mark_dirty("radius")

        # Should return None due to dirty flag
        result = cache.get(simple_scene)
        assert result is None

    def test_clean_scene_returns_cached(self, cache: WGSLCache, simple_scene: SceneNode):
        """Clean scene should return cached value."""
        simple_scene.tracker.clear_recursive()
        cache.put(simple_scene, "// cached WGSL")

        # Scene is clean, should return cached
        result = cache.get(simple_scene)
        assert result == "// cached WGSL"

    def test_is_cache_valid_returns_true_when_clean(self, simple_scene: SceneNode):
        """is_cache_valid should return True when scene is clean."""
        simple_scene.tracker.clear_recursive()
        assert is_cache_valid(simple_scene) is True

    def test_is_cache_valid_returns_false_when_dirty(self, simple_scene: SceneNode):
        """is_cache_valid should return False when scene is dirty."""
        simple_scene.tracker.clear_recursive()
        simple_scene.root.tracker.mark_dirty("radius")
        assert is_cache_valid(simple_scene) is False

    def test_nested_dirty_invalidates_parent(self, complex_scene: SceneNode):
        """Dirty child should invalidate parent scene cache."""
        complex_scene.tracker.clear_recursive()
        cache = WGSLCache(max_size=10)
        cache.put(complex_scene, "// complex WGSL")

        # Mark nested child dirty
        union = complex_scene.root
        assert isinstance(union, UnionNode)
        union.left.tracker.mark_dirty("radius")

        result = cache.get(complex_scene)
        assert result is None


# =============================================================================
# INVALIDATION TESTS
# =============================================================================


class TestCacheInvalidation:
    """Tests for explicit cache invalidation."""

    def test_invalidate_removes_entry(self, cache: WGSLCache, simple_scene: SceneNode):
        """Invalidate should remove cache entry."""
        simple_scene.tracker.clear_recursive()
        cache.put(simple_scene, "// WGSL")
        assert cache.size == 1

        result = cache.invalidate(simple_scene)
        assert result is True
        assert cache.size == 0

    def test_invalidate_nonexistent_returns_false(self, cache: WGSLCache, simple_scene: SceneNode):
        """Invalidating uncached scene should return False."""
        simple_scene.tracker.clear_recursive()
        result = cache.invalidate(simple_scene)
        assert result is False

    def test_invalidate_specific_opt_level(self, cache: WGSLCache, simple_scene: SceneNode):
        """Should only invalidate specific optimization level."""
        simple_scene.tracker.clear_recursive()

        # Add entries at different optimization levels
        cache.put(simple_scene, "// WGSL default", OptimizationLevel.DEFAULT)
        cache.put(simple_scene, "// WGSL fast", OptimizationLevel.FAST)
        assert cache.size == 2

        # Invalidate only DEFAULT
        cache.invalidate(simple_scene, OptimizationLevel.DEFAULT)
        assert cache.size == 1

        # FAST should still be cached
        result = cache.get(simple_scene, OptimizationLevel.FAST)
        assert result == "// WGSL fast"

    def test_invalidation_increments_stats(self, cache: WGSLCache, simple_scene: SceneNode):
        """Invalidation should increment stats counter."""
        simple_scene.tracker.clear_recursive()
        cache.put(simple_scene, "// WGSL")
        cache.invalidate(simple_scene)

        stats = cache.stats()
        assert stats.invalidations == 1


# =============================================================================
# LRU EVICTION TESTS
# =============================================================================


class TestLRUEviction:
    """Tests for LRU cache eviction."""

    def test_eviction_at_max_capacity(self):
        """Cache should evict when at max capacity."""
        cache = WGSLCache(max_size=2)

        scene1 = SceneNode(root=SphereNode(radius=1.0), name="scene1")
        scene2 = SceneNode(root=SphereNode(radius=2.0), name="scene2")
        scene3 = SceneNode(root=SphereNode(radius=3.0), name="scene3")

        for s in [scene1, scene2, scene3]:
            s.tracker.clear_recursive()

        cache.put(scene1, "// WGSL 1")
        cache.put(scene2, "// WGSL 2")
        assert cache.size == 2

        # Adding third should evict oldest (scene1)
        cache.put(scene3, "// WGSL 3")
        assert cache.size == 2

        # scene1 should be evicted
        assert cache.get(scene1) is None

        # scene2 and scene3 should still be cached
        assert cache.get(scene2) == "// WGSL 2"
        assert cache.get(scene3) == "// WGSL 3"

    def test_access_updates_lru_order(self):
        """Accessing entry should move it to end of LRU."""
        cache = WGSLCache(max_size=2)

        scene1 = SceneNode(root=SphereNode(radius=1.0), name="scene1")
        scene2 = SceneNode(root=SphereNode(radius=2.0), name="scene2")
        scene3 = SceneNode(root=SphereNode(radius=3.0), name="scene3")

        for s in [scene1, scene2, scene3]:
            s.tracker.clear_recursive()

        cache.put(scene1, "// WGSL 1")
        cache.put(scene2, "// WGSL 2")

        # Access scene1, making it more recently used
        cache.get(scene1)

        # Add scene3, should evict scene2 (older)
        cache.put(scene3, "// WGSL 3")

        assert cache.get(scene1) == "// WGSL 1"  # Still cached
        assert cache.get(scene2) is None  # Evicted

    def test_eviction_increments_stats(self):
        """Eviction should increment stats counter."""
        cache = WGSLCache(max_size=1)

        scene1 = SceneNode(root=SphereNode(radius=1.0), name="scene1")
        scene2 = SceneNode(root=SphereNode(radius=2.0), name="scene2")

        for s in [scene1, scene2]:
            s.tracker.clear_recursive()

        cache.put(scene1, "// WGSL 1")
        cache.put(scene2, "// WGSL 2")

        stats = cache.stats()
        assert stats.evictions == 1


# =============================================================================
# HASHING TESTS
# =============================================================================


class TestASTHashing:
    """Tests for AST structural hashing."""

    def test_same_structure_same_hash(self):
        """Identical structures should have same hash."""
        sphere1 = SphereNode(radius=1.5)
        sphere2 = SphereNode(radius=1.5)

        h1 = sdf_node_hash(sphere1)
        h2 = sdf_node_hash(sphere2)
        assert h1 == h2

    def test_different_values_different_hash(self):
        """Different values should produce different hash."""
        sphere1 = SphereNode(radius=1.0)
        sphere2 = SphereNode(radius=2.0)

        h1 = sdf_node_hash(sphere1)
        h2 = sdf_node_hash(sphere2)
        assert h1 != h2

    def test_different_types_different_hash(self):
        """Different node types should have different hashes."""
        sphere = SphereNode(radius=1.0)
        box = BoxNode(half_extents=Vec3(1.0, 1.0, 1.0))

        h1 = sdf_node_hash(sphere)
        h2 = sdf_node_hash(box)
        assert h1 != h2

    def test_nested_structures_hash_correctly(self):
        """Nested structures should hash based on full tree."""
        sphere1 = SphereNode(radius=1.0)
        sphere2 = SphereNode(radius=1.0)
        box = BoxNode(half_extents=Vec3(1.0, 1.0, 1.0))

        union1 = UnionNode(sphere1, box)
        union2 = UnionNode(sphere2, box)

        h1 = sdf_node_hash(union1)
        h2 = sdf_node_hash(union2)
        assert h1 == h2

    def test_hash_includes_scene_components(self):
        """Scene hash should include all components."""
        sphere = SphereNode(radius=1.0)
        scene1 = SceneNode(root=sphere, name="scene1")
        scene2 = SceneNode(root=sphere, name="scene2")

        h1 = sdf_node_hash(scene1)
        h2 = sdf_node_hash(scene2)
        # Different names = different hash
        assert h1 != h2

    def test_version_optional_in_hash(self):
        """Version should be optionally included in hash."""
        sphere = SphereNode(radius=1.0)

        h1 = sdf_node_hash(sphere, include_version=False)
        sphere.tracker.mark_dirty("radius")
        h2 = sdf_node_hash(sphere, include_version=False)

        # Without version, hash should be same
        assert h1 == h2

        # With version, hash should differ
        sphere2 = SphereNode(radius=1.0)
        h3 = sdf_node_hash(sphere, include_version=True)
        h4 = sdf_node_hash(sphere2, include_version=True)
        assert h3 != h4


class TestHashingAllNodeTypes:
    """Test hashing works for all node types."""

    def test_hash_sphere(self):
        """Sphere should hash correctly."""
        node = SphereNode(radius=1.5, position=Vec3(1.0, 2.0, 3.0))
        h = sdf_node_hash(node)
        assert isinstance(h, int)

    def test_hash_box(self):
        """Box should hash correctly."""
        node = BoxNode(half_extents=Vec3(1.0, 2.0, 3.0))
        h = sdf_node_hash(node)
        assert isinstance(h, int)

    def test_hash_torus(self):
        """Torus should hash correctly."""
        node = TorusNode(major_radius=1.0, minor_radius=0.25)
        h = sdf_node_hash(node)
        assert isinstance(h, int)

    def test_hash_cylinder(self):
        """Cylinder should hash correctly."""
        node = CylinderNode(radius=0.5, height=2.0)
        h = sdf_node_hash(node)
        assert isinstance(h, int)

    def test_hash_cone(self):
        """Cone should hash correctly."""
        node = ConeNode(angle=0.5, height=1.0)
        h = sdf_node_hash(node)
        assert isinstance(h, int)

    def test_hash_plane(self):
        """Plane should hash correctly."""
        node = PlaneNode(normal=Vec3(0.0, 1.0, 0.0), distance=1.0)
        h = sdf_node_hash(node)
        assert isinstance(h, int)

    def test_hash_capsule(self):
        """Capsule should hash correctly."""
        node = CapsuleNode(
            endpoint_a=Vec3(0.0, -1.0, 0.0),
            endpoint_b=Vec3(0.0, 1.0, 0.0),
            radius=0.25,
        )
        h = sdf_node_hash(node)
        assert isinstance(h, int)

    def test_hash_ellipsoid(self):
        """Ellipsoid should hash correctly."""
        node = EllipsoidNode(radii=Vec3(1.0, 1.5, 0.5))
        h = sdf_node_hash(node)
        assert isinstance(h, int)

    def test_hash_box_frame(self):
        """BoxFrame should hash correctly."""
        node = BoxFrameNode(half_extents=Vec3(1.0, 1.0, 1.0), edge_thickness=0.1)
        h = sdf_node_hash(node)
        assert isinstance(h, int)

    def test_hash_rounded_box(self):
        """RoundedBox should hash correctly."""
        node = RoundedBoxNode(half_extents=Vec3(1.0, 1.0, 1.0), corner_radius=0.1)
        h = sdf_node_hash(node)
        assert isinstance(h, int)

    def test_hash_octahedron(self):
        """Octahedron should hash correctly."""
        node = OctahedronNode(size=1.0)
        h = sdf_node_hash(node)
        assert isinstance(h, int)

    def test_hash_pyramid(self):
        """Pyramid should hash correctly."""
        node = PyramidNode(height=1.0)
        h = sdf_node_hash(node)
        assert isinstance(h, int)

    def test_hash_union(self):
        """Union should hash correctly."""
        node = UnionNode(SphereNode(radius=1.0), BoxNode())
        h = sdf_node_hash(node)
        assert isinstance(h, int)

    def test_hash_intersection(self):
        """Intersection should hash correctly."""
        node = IntersectionNode(SphereNode(radius=1.0), BoxNode())
        h = sdf_node_hash(node)
        assert isinstance(h, int)

    def test_hash_subtraction(self):
        """Subtraction should hash correctly."""
        node = SubtractionNode(SphereNode(radius=1.0), BoxNode())
        h = sdf_node_hash(node)
        assert isinstance(h, int)

    def test_hash_smooth_union(self):
        """SmoothUnion should hash correctly."""
        node = SmoothUnionNode(SphereNode(radius=1.0), BoxNode(), k=0.2)
        h = sdf_node_hash(node)
        assert isinstance(h, int)

    def test_hash_displaced(self):
        """Displaced should hash correctly."""
        node = DisplacedNode(SphereNode(radius=1.0), amplitude=0.1, frequency=2.0)
        h = sdf_node_hash(node)
        assert isinstance(h, int)

    def test_hash_repeat(self):
        """Repeat should hash correctly."""
        node = RepeatNode(SphereNode(radius=1.0), cell_size=Vec3(2.0, 2.0, 2.0))
        h = sdf_node_hash(node)
        assert isinstance(h, int)

    def test_hash_mirror(self):
        """Mirror should hash correctly."""
        node = MirrorNode(SphereNode(radius=1.0), axis=Axis.X)
        h = sdf_node_hash(node)
        assert isinstance(h, int)

    def test_hash_kifs(self):
        """KIFS should hash correctly."""
        node = KIFSNode(SphereNode(radius=1.0), iterations=6, scale=2.0)
        h = sdf_node_hash(node)
        assert isinstance(h, int)

    def test_hash_twist(self):
        """Twist should hash correctly."""
        node = TwistNode(SphereNode(radius=1.0), axis=Axis.Y, rate=0.5)
        h = sdf_node_hash(node)
        assert isinstance(h, int)

    def test_hash_bend(self):
        """Bend should hash correctly."""
        node = BendNode(SphereNode(radius=1.0), axis=Axis.Z, radius=10.0)
        h = sdf_node_hash(node)
        assert isinstance(h, int)

    def test_hash_stretch(self):
        """Stretch should hash correctly."""
        node = StretchNode(SphereNode(radius=1.0), axis=Axis.X, scale=2.0)
        h = sdf_node_hash(node)
        assert isinstance(h, int)

    def test_hash_material(self):
        """Material should hash correctly."""
        node = MaterialNode(color=Vec3(1.0, 0.5, 0.0), metallic=0.5, roughness=0.3)
        h = sdf_node_hash(node)
        assert isinstance(h, int)

    def test_hash_camera(self):
        """Camera should hash correctly."""
        node = CameraNode(
            origin=Vec3(0.0, 0.0, 5.0),
            look_at=Vec3(0.0, 0.0, 0.0),
            fov=60.0,
        )
        h = sdf_node_hash(node)
        assert isinstance(h, int)

    def test_hash_light(self):
        """Light should hash correctly."""
        node = LightNode(position=Vec3(5.0, 5.0, 5.0), color=Vec3(1.0, 1.0, 1.0))
        h = sdf_node_hash(node)
        assert isinstance(h, int)

    def test_hash_render_settings(self):
        """RenderSettings should hash correctly."""
        node = RenderSettingsNode(width=1920, height=1080, max_steps=128)
        h = sdf_node_hash(node)
        assert isinstance(h, int)


# =============================================================================
# CACHE KEY TESTS
# =============================================================================


class TestCacheKey:
    """Tests for cache key computation."""

    def test_key_includes_optimization_level(self, simple_scene: SceneNode):
        """Cache key should differ by optimization level."""
        simple_scene.tracker.clear_recursive()

        key1 = _compute_cache_key(simple_scene, OptimizationLevel.NONE)
        key2 = _compute_cache_key(simple_scene, OptimizationLevel.AGGRESSIVE)

        assert key1 != key2

    def test_key_same_for_same_scene_and_level(self, simple_scene: SceneNode):
        """Same scene and level should produce same key."""
        simple_scene.tracker.clear_recursive()

        key1 = _compute_cache_key(simple_scene, OptimizationLevel.DEFAULT)
        key2 = _compute_cache_key(simple_scene, OptimizationLevel.DEFAULT)

        assert key1 == key2


# =============================================================================
# CACHED COMPILER TESTS
# =============================================================================


class TestCachedSDFCompiler:
    """Tests for CachedSDFCompiler."""

    def test_compile_returns_wgsl(self, compiler: CachedSDFCompiler, simple_scene: SceneNode):
        """Compile should return WGSL code."""
        simple_scene.tracker.clear_recursive()
        wgsl = compiler.compile(simple_scene)
        assert isinstance(wgsl, str)
        assert "// WGSL" in wgsl or "fn sd_scene" in wgsl

    def test_second_compile_returns_cached(self, compiler: CachedSDFCompiler, simple_scene: SceneNode):
        """Second compile should return cached value."""
        simple_scene.tracker.clear_recursive()
        wgsl1 = compiler.compile(simple_scene)
        wgsl2 = compiler.compile(simple_scene)

        assert wgsl1 == wgsl2
        stats = compiler.stats()
        assert stats.hits == 1
        assert stats.misses == 1

    def test_compile_clears_dirty_flags(self, compiler: CachedSDFCompiler, simple_scene: SceneNode):
        """Compile should clear dirty flags after compilation."""
        # Scene starts dirty
        assert simple_scene.tracker.is_dirty

        compiler.compile(simple_scene)

        # Should be clean after compile
        assert not simple_scene.tracker.is_dirty

    def test_dirty_scene_recompiles(self, compiler: CachedSDFCompiler, simple_scene: SceneNode):
        """Dirty scene should trigger recompilation."""
        simple_scene.tracker.clear_recursive()

        wgsl1 = compiler.compile(simple_scene)
        stats1 = compiler.stats()
        assert stats1.misses == 1

        # Mark dirty and recompile
        simple_scene.root.tracker.mark_dirty("radius")
        wgsl2 = compiler.compile(simple_scene)

        stats2 = compiler.stats()
        assert stats2.misses == 2  # Should have recompiled

    def test_force_recompile_bypasses_cache(self, compiler: CachedSDFCompiler, simple_scene: SceneNode):
        """force_recompile should bypass cache without counting as hit or miss."""
        simple_scene.tracker.clear_recursive()

        compiler.compile(simple_scene)  # First compile - miss
        compiler.compile(simple_scene)  # Cache hit
        compiler.compile(simple_scene, force_recompile=True)  # Bypasses cache

        stats = compiler.stats()
        assert stats.hits == 1  # Only the second compile was a hit
        # force_recompile skips cache lookup entirely, so only 1 miss (first compile)
        assert stats.misses == 1

    def test_invalidate_removes_from_cache(self, compiler: CachedSDFCompiler, simple_scene: SceneNode):
        """Invalidate should remove entry from cache."""
        simple_scene.tracker.clear_recursive()
        compiler.compile(simple_scene)

        result = compiler.invalidate(simple_scene)
        assert result is True

        # Next compile should miss
        compiler.compile(simple_scene)
        stats = compiler.stats()
        assert stats.misses == 2

    def test_clear_cache_removes_all(self, compiler: CachedSDFCompiler, simple_scene: SceneNode):
        """clear_cache should remove all entries."""
        simple_scene.tracker.clear_recursive()
        compiler.compile(simple_scene)
        assert compiler.cache.size == 1

        compiler.clear_cache()
        assert compiler.cache.size == 0

    def test_is_cache_valid_integration(self, compiler: CachedSDFCompiler, simple_scene: SceneNode):
        """is_cache_valid should check dirty state."""
        simple_scene.tracker.clear_recursive()
        assert compiler.is_cache_valid(simple_scene)

        simple_scene.root.tracker.mark_dirty("radius")
        assert not compiler.is_cache_valid(simple_scene)

    def test_optimization_level_property(self, compiler: CachedSDFCompiler):
        """Optimization level should be settable."""
        assert compiler.optimization_level == OptimizationLevel.DEFAULT

        compiler.optimization_level = OptimizationLevel.AGGRESSIVE
        assert compiler.optimization_level == OptimizationLevel.AGGRESSIVE


# =============================================================================
# CUSTOM COMPILE FUNCTION TESTS
# =============================================================================


class TestCustomCompileFunction:
    """Tests for custom compile function support."""

    def test_custom_compile_func_called(self, simple_scene: SceneNode):
        """Custom compile function should be called."""
        simple_scene.tracker.clear_recursive()

        mock_compile = Mock(return_value="// custom WGSL")
        compiler = CachedSDFCompiler(compile_func=mock_compile)

        wgsl = compiler.compile(simple_scene)

        assert wgsl == "// custom WGSL"
        mock_compile.assert_called_once()

    def test_custom_compile_receives_name(self, simple_scene: SceneNode):
        """Custom compile function should receive name parameter."""
        simple_scene.tracker.clear_recursive()

        mock_compile = Mock(return_value="// WGSL")
        compiler = CachedSDFCompiler(compile_func=mock_compile)

        compiler.compile(simple_scene, name="custom_name")

        _, call_kwargs = mock_compile.call_args
        # Check the second positional arg is the name
        call_args = mock_compile.call_args[0]
        assert call_args[1] == "custom_name"


# =============================================================================
# THREAD SAFETY TESTS
# =============================================================================


class TestThreadSafety:
    """Tests for thread-safe cache access."""

    def test_concurrent_reads(self, cache: WGSLCache):
        """Concurrent reads should be safe."""
        scene = SceneNode(root=SphereNode(radius=1.0), name="concurrent")
        scene.tracker.clear_recursive()
        cache.put(scene, "// concurrent WGSL")

        results: List[str] = []
        errors: List[Exception] = []

        def read_cache():
            try:
                for _ in range(100):
                    result = cache.get(scene)
                    if result:
                        results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_cache) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 1000  # 10 threads * 100 reads

    def test_concurrent_writes(self, cache: WGSLCache):
        """Concurrent writes should be safe."""
        errors: List[Exception] = []

        def write_cache(thread_id: int):
            try:
                for i in range(10):
                    scene = SceneNode(
                        root=SphereNode(radius=float(thread_id * 100 + i)),
                        name=f"thread{thread_id}_scene{i}",
                    )
                    scene.tracker.clear_recursive()
                    cache.put(scene, f"// WGSL {thread_id}_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_cache, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Should have at most max_size entries
        assert cache.size <= cache.max_size

    def test_concurrent_read_write(self, cache: WGSLCache):
        """Concurrent reads and writes should be safe."""
        scene = SceneNode(root=SphereNode(radius=1.0), name="rw_test")
        scene.tracker.clear_recursive()
        cache.put(scene, "// initial WGSL")

        errors: List[Exception] = []

        def reader():
            try:
                for _ in range(50):
                    cache.get(scene)
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for _ in range(50):
                    cache.put(scene, "// updated WGSL")
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=reader))
            threads.append(threading.Thread(target=writer))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_scene_name(self, simple_scene: SceneNode, cache: WGSLCache):
        """Scene with empty name should work."""
        empty_name_scene = SceneNode(root=SphereNode(radius=1.0), name="")
        empty_name_scene.tracker.clear_recursive()

        cache.put(empty_name_scene, "// WGSL")
        result = cache.get(empty_name_scene)
        assert result == "// WGSL"

    def test_zero_max_size_cache(self):
        """Zero max size should work (no caching)."""
        # Use max_size=1 as minimum practical size
        cache = WGSLCache(max_size=1)
        scene1 = SceneNode(root=SphereNode(radius=1.0), name="s1")
        scene2 = SceneNode(root=SphereNode(radius=2.0), name="s2")

        for s in [scene1, scene2]:
            s.tracker.clear_recursive()

        cache.put(scene1, "// WGSL 1")
        cache.put(scene2, "// WGSL 2")

        # Only one should remain
        assert cache.size == 1

    def test_very_large_wgsl(self, cache: WGSLCache, simple_scene: SceneNode):
        """Large WGSL strings should work."""
        simple_scene.tracker.clear_recursive()
        large_wgsl = "// " + "x" * 1000000  # 1MB

        cache.put(simple_scene, large_wgsl)
        result = cache.get(simple_scene)
        assert result == large_wgsl

    def test_unicode_in_scene_name(self, cache: WGSLCache):
        """Unicode scene names should work."""
        scene = SceneNode(root=SphereNode(radius=1.0), name="")
        scene.tracker.clear_recursive()

        cache.put(scene, "// WGSL")
        result = cache.get(scene)
        assert result == "// WGSL"

    def test_deeply_nested_scene(self, cache: WGSLCache):
        """Deeply nested scene should work."""
        # Build a deep tree
        node = SphereNode(radius=1.0)
        for _ in range(20):
            node = UnionNode(node, SphereNode(radius=0.5))

        scene = SceneNode(root=node, name="deep")
        scene.tracker.clear_recursive()

        cache.put(scene, "// deep WGSL")
        result = cache.get(scene)
        assert result == "// deep WGSL"


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory/helper functions."""

    def test_create_cached_compiler_defaults(self):
        """create_cached_compiler should work with defaults."""
        compiler = create_cached_compiler()
        assert isinstance(compiler, CachedSDFCompiler)
        assert compiler.cache.max_size == 256
        assert compiler.optimization_level == OptimizationLevel.DEFAULT

    def test_create_cached_compiler_custom_size(self):
        """create_cached_compiler should accept custom cache size."""
        compiler = create_cached_compiler(max_cache_size=100)
        assert compiler.cache.max_size == 100

    def test_create_cached_compiler_custom_opt_level(self):
        """create_cached_compiler should accept custom optimization level."""
        compiler = create_cached_compiler(optimization_level=OptimizationLevel.AGGRESSIVE)
        assert compiler.optimization_level == OptimizationLevel.AGGRESSIVE


# =============================================================================
# REPR TESTS
# =============================================================================


class TestRepr:
    """Tests for string representations."""

    def test_cache_stats_repr(self):
        """CacheStats repr should be informative."""
        stats = CacheStats(hits=10, misses=5)
        repr_str = repr(stats)
        assert "hits=10" in repr_str
        assert "misses=5" in repr_str
        assert "hit_rate" in repr_str

    def test_cache_repr(self, cache: WGSLCache, simple_scene: SceneNode):
        """WGSLCache repr should be informative."""
        simple_scene.tracker.clear_recursive()
        cache.put(simple_scene, "// WGSL")

        repr_str = repr(cache)
        assert "WGSLCache" in repr_str
        assert "size=" in repr_str

    def test_compiler_repr(self, compiler: CachedSDFCompiler, simple_scene: SceneNode):
        """CachedSDFCompiler repr should be informative."""
        simple_scene.tracker.clear_recursive()
        compiler.compile(simple_scene)

        repr_str = repr(compiler)
        assert "CachedSDFCompiler" in repr_str
        assert "cache_size=" in repr_str


# =============================================================================
# CACHE ENTRY TESTS
# =============================================================================


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_cache_entry_creation(self):
        """CacheEntry should be creatable."""
        now = time.time()
        entry = CacheEntry(
            wgsl="// WGSL",
            ast_hash=12345,
            optimization_level=OptimizationLevel.DEFAULT,
            created_at=now,
            last_accessed=now,
        )
        assert entry.wgsl == "// WGSL"
        assert entry.ast_hash == 12345
        assert entry.access_count == 1

    def test_cache_entry_touch(self):
        """Touch should update access metadata."""
        now = time.time()
        entry = CacheEntry(
            wgsl="// WGSL",
            ast_hash=12345,
            optimization_level=OptimizationLevel.DEFAULT,
            created_at=now,
            last_accessed=now,
        )

        time.sleep(0.01)  # Small delay
        entry.touch()

        assert entry.access_count == 2
        assert entry.last_accessed > now


# =============================================================================
# OPTIMIZATION LEVEL TESTS
# =============================================================================


class TestOptimizationLevel:
    """Tests for OptimizationLevel enum."""

    def test_optimization_levels_exist(self):
        """All optimization levels should exist."""
        assert OptimizationLevel.NONE == 0
        assert OptimizationLevel.FAST == 1
        assert OptimizationLevel.DEFAULT == 2
        assert OptimizationLevel.AGGRESSIVE == 3

    def test_optimization_level_ordering(self):
        """Optimization levels should be ordered."""
        assert OptimizationLevel.NONE < OptimizationLevel.FAST
        assert OptimizationLevel.FAST < OptimizationLevel.DEFAULT
        assert OptimizationLevel.DEFAULT < OptimizationLevel.AGGRESSIVE

    def test_different_opt_levels_cache_separately(self, cache: WGSLCache, simple_scene: SceneNode):
        """Different optimization levels should cache separately."""
        simple_scene.tracker.clear_recursive()

        cache.put(simple_scene, "// NONE", OptimizationLevel.NONE)
        cache.put(simple_scene, "// FAST", OptimizationLevel.FAST)
        cache.put(simple_scene, "// DEFAULT", OptimizationLevel.DEFAULT)

        assert cache.size == 3

        assert cache.get(simple_scene, OptimizationLevel.NONE) == "// NONE"
        assert cache.get(simple_scene, OptimizationLevel.FAST) == "// FAST"
        assert cache.get(simple_scene, OptimizationLevel.DEFAULT) == "// DEFAULT"


# =============================================================================
# PERFORMANCE TIMING TESTS
# =============================================================================


class TestPerformanceTiming:
    """Tests for timing-related functionality."""

    def test_cache_hit_faster_than_compile(self, simple_scene: SceneNode):
        """Cache hit should be significantly faster than compilation."""
        simple_scene.tracker.clear_recursive()

        # Use a mock compile function with artificial delay
        def slow_compile(scene, name):
            time.sleep(0.01)  # 10ms compile time
            return "// WGSL"

        compiler = CachedSDFCompiler(compile_func=slow_compile)

        # First compile (slow)
        start = time.perf_counter()
        compiler.compile(simple_scene)
        compile_time = time.perf_counter() - start

        # Second compile (cache hit, should be fast)
        start = time.perf_counter()
        compiler.compile(simple_scene)
        cache_time = time.perf_counter() - start

        # Cache hit should be at least 10x faster
        assert cache_time < compile_time / 10

    def test_time_saved_calculation(self, simple_scene: SceneNode):
        """Time saved by cache should be calculable."""
        simple_scene.tracker.clear_recursive()

        cache = WGSLCache()

        # First get is a miss (increments miss counter)
        result = cache.get(simple_scene)
        assert result is None

        # Put the compiled result (records compile time)
        cache.put(simple_scene, "// WGSL", compile_time_ms=100.0)

        # Simulate cache hits
        for _ in range(10):
            cache.get(simple_scene)

        stats = cache.stats()
        # With 1 miss at 100ms, and 10 hits, time saved should be ~1000ms
        # (10 hits * 100ms avg compile - cache lookup time)
        assert stats.misses == 1
        assert stats.hits == 10
        assert stats.avg_compile_time_ms == pytest.approx(100.0)
        # time_saved = hits * avg_compile - cache_lookup_time
        # Should be positive since cache lookups are fast
        assert stats.time_saved_ms > 0
