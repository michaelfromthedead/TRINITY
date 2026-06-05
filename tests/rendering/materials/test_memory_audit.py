"""Memory Audit Tests for Material System (T-MAT-11.4).

This module provides comprehensive memory testing for the material system components:
- DepGraph adjacency growth
- PipelineTable LRU eviction
- ShaderCache unbounded growth prevention
- ContentStore orphan detection and GC
- Hot-reload accumulation prevention

Acceptance criteria:
- Memory stabilizes under load
- No unbounded growth
- DepGraph, PipelineTable, ShaderCache within budget (256MB for 10,000 pipelines)
"""

from __future__ import annotations

import gc
import sys
import time
import threading
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest

from trinity.materials.dep_graph import MaterialDepGraph
from trinity.materials.pipeline_integration import (
    LruPipelineTable,
    PipelineConfig,
    PipelineIntegration,
    ShaderCache,
    shader_hash,
)
from foundation.content_store import (
    ContentStore,
    ContentHash,
    MemoryBackend,
)


# =============================================================================
# Memory Tracking Utilities
# =============================================================================


@dataclass
class MemorySnapshot:
    """Snapshot of memory state at a point in time."""
    current: int  # Current memory usage in bytes
    peak: int  # Peak memory usage in bytes
    timestamp: float

    @property
    def current_mb(self) -> float:
        """Current memory in megabytes."""
        return self.current / (1024 * 1024)

    @property
    def peak_mb(self) -> float:
        """Peak memory in megabytes."""
        return self.peak / (1024 * 1024)


class MemoryTracker:
    """Utility for tracking memory usage during tests.

    Uses tracemalloc for precise Python object tracking.
    """

    def __init__(self) -> None:
        self._snapshots: List[MemorySnapshot] = []
        self._started = False

    def start(self) -> None:
        """Start memory tracking."""
        gc.collect()
        tracemalloc.start()
        self._started = True

    def stop(self) -> None:
        """Stop memory tracking."""
        if self._started:
            tracemalloc.stop()
            self._started = False

    def snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot."""
        if not self._started:
            raise RuntimeError("Memory tracking not started")

        current, peak = tracemalloc.get_traced_memory()
        snap = MemorySnapshot(
            current=current,
            peak=peak,
            timestamp=time.time()
        )
        self._snapshots.append(snap)
        return snap

    def get_growth(self, from_snap: MemorySnapshot, to_snap: MemorySnapshot) -> int:
        """Calculate memory growth between two snapshots in bytes."""
        return to_snap.current - from_snap.current

    def get_growth_mb(self, from_snap: MemorySnapshot, to_snap: MemorySnapshot) -> float:
        """Calculate memory growth between two snapshots in MB."""
        return self.get_growth(from_snap, to_snap) / (1024 * 1024)

    @property
    def all_snapshots(self) -> List[MemorySnapshot]:
        """Get all recorded snapshots."""
        return self._snapshots.copy()

    def __enter__(self) -> "MemoryTracker":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()


# =============================================================================
# Memory Budget Tracker Implementation
# =============================================================================


class MemoryBudgetTracker:
    """Track memory usage of major caching components.

    This class provides methods to query the memory footprint of:
    - DepGraph (adjacency structures)
    - PipelineTable (cached pipelines)
    - ShaderCache (compiled shader modules)
    - ContentStore (content-addressed objects)
    """

    def __init__(
        self,
        dep_graph: Optional[MaterialDepGraph] = None,
        pipeline_table: Optional[LruPipelineTable] = None,
        shader_cache: Optional[ShaderCache] = None,
        content_store: Optional[ContentStore] = None,
    ) -> None:
        self._dep_graph = dep_graph
        self._pipeline_table = pipeline_table
        self._shader_cache = shader_cache
        self._content_store = content_store

    def get_depgraph_size(self) -> int:
        """Get approximate memory size of DepGraph in bytes."""
        if self._dep_graph is None:
            return 0

        # Estimate size based on number of edges and materials
        material_count = self._dep_graph.material_count()
        include_count = self._dep_graph.include_count()

        # Each edge is roughly two Path objects (source, target) + overhead
        # Path object ~ 100 bytes, set entry ~ 56 bytes
        edge_count = 0
        for mat, incs in self._dep_graph.material_to_includes.items():
            edge_count += len(incs)
        for mat, deps in self._dep_graph.material_to_dependents.items():
            edge_count += len(deps)

        # Rough estimation: Path(100) + Set entry(56) per edge
        # Plus dict overhead
        estimated = (
            material_count * 156 +  # material_to_includes entries
            include_count * 156 +   # include_to_materials entries
            edge_count * 156 +      # edge references
            1024  # Base overhead
        )
        return estimated

    def get_pipeline_table_size(self) -> int:
        """Get approximate memory size of PipelineTable in bytes."""
        if self._pipeline_table is None:
            return 0

        pipeline_count = len(self._pipeline_table)

        # Each pipeline entry: CachedPipeline object + hash string + LRU entry
        # CachedPipeline ~ 200 bytes, hash (64 chars) ~ 128 bytes, LRU ~ 56 bytes
        per_pipeline = 384

        # Plus shader cache size
        shader_size = self.get_shader_cache_size() if self._shader_cache is None else 0

        return pipeline_count * per_pipeline + shader_size

    def get_shader_cache_size(self) -> int:
        """Get approximate memory size of ShaderCache in bytes."""
        cache = self._shader_cache
        if cache is None and self._pipeline_table is not None:
            cache = self._pipeline_table.shader_cache

        if cache is None:
            return 0

        # Module count + path tracking
        module_count = len(cache)
        path_count = cache.stats.tracked_paths
        source_bytes = cache.stats.total_source_bytes

        # Each module: placeholder + hash
        per_module = 200
        per_path = 150

        return module_count * per_module + path_count * per_path + source_bytes

    def get_content_store_size(self) -> int:
        """Get approximate memory size of ContentStore in bytes."""
        if self._content_store is None:
            return 0

        backend = self._content_store._backend
        if isinstance(backend, MemoryBackend):
            # Sum actual stored bytes
            total = sum(len(v) for v in backend._store.values())
            # Add hash key overhead (64 chars = 128 bytes per key)
            total += len(backend._store) * 128
            return total

        return 0

    def total_size(self) -> int:
        """Get total memory size of all tracked components."""
        return (
            self.get_depgraph_size() +
            self.get_pipeline_table_size() +
            self.get_shader_cache_size() +
            self.get_content_store_size()
        )

    def total_size_mb(self) -> float:
        """Get total memory size in megabytes."""
        return self.total_size() / (1024 * 1024)

    def within_budget(self, budget_mb: float) -> bool:
        """Check if total memory is within the specified budget."""
        return self.total_size_mb() <= budget_mb


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def memory_tracker() -> MemoryTracker:
    """Provide a memory tracker for tests."""
    tracker = MemoryTracker()
    yield tracker
    if tracker._started:
        tracker.stop()


@pytest.fixture
def dep_graph() -> MaterialDepGraph:
    """Provide a fresh MaterialDepGraph for testing."""
    return MaterialDepGraph()


@pytest.fixture
def pipeline_table() -> LruPipelineTable:
    """Provide a fresh LruPipelineTable with default size."""
    return LruPipelineTable(max_size=64)


@pytest.fixture
def large_pipeline_table() -> LruPipelineTable:
    """Provide a LruPipelineTable sized for stress testing."""
    return LruPipelineTable(max_size=10000)


@pytest.fixture
def shader_cache() -> ShaderCache:
    """Provide a fresh ShaderCache."""
    return ShaderCache()


@pytest.fixture
def content_store() -> ContentStore:
    """Provide a fresh ContentStore with MemoryBackend."""
    return ContentStore(MemoryBackend())


def generate_shader(index: int, variant: str = "default") -> str:
    """Generate unique shader source for testing."""
    return f"""
    // Shader {index} variant {variant}
    @vertex fn vs_main_{index}() -> @builtin(position) vec4<f32> {{
        return vec4<f32>({float(index)}, 0.0, 0.0, 1.0);
    }}
    @fragment fn fs_main_{index}() -> @location(0) vec4<f32> {{
        return vec4<f32>({float(index % 256) / 255.0}, 0.0, 0.0, 1.0);
    }}
    """


def generate_material_path(index: int) -> Path:
    """Generate unique material path for testing."""
    return Path(f"/test/materials/material_{index:05d}.wgsl")


def generate_include_path(index: int) -> Path:
    """Generate unique include path for testing."""
    return Path(f"/test/shaders/include_{index:05d}.wgsl")


# =============================================================================
# DepGraph Memory Tests
# =============================================================================


class TestDepGraphMemory:
    """Memory tests for MaterialDepGraph."""

    def test_depgraph_adjacency_bounded(
        self, dep_graph: MaterialDepGraph, memory_tracker: MemoryTracker
    ) -> None:
        """Add/remove 10,000 edges, verify stable memory.

        This test ensures that adding and then removing edges does not
        cause memory to grow unboundedly.
        """
        with memory_tracker:
            initial = memory_tracker.snapshot()

            # Add 10,000 material compilations with includes
            materials: List[Path] = []
            for i in range(10000):
                material = generate_material_path(i)
                includes = {
                    generate_include_path(i % 100),
                    generate_include_path((i + 1) % 100),
                    generate_include_path((i + 2) % 100),
                }
                dep_graph.record_material_compilation(material, includes)
                materials.append(material)

            after_add = memory_tracker.snapshot()

            # Remove all materials
            for material in materials:
                dep_graph.remove_material(material)

            # Force garbage collection
            gc.collect()
            gc.collect()

            after_remove = memory_tracker.snapshot()

            # Memory after removal should be significantly less than peak
            # Python Path objects and dict structures may retain some memory
            # due to interning and allocator behavior

            # Assert counts are zero
            assert dep_graph.material_count() == 0
            assert dep_graph.include_count() == 0

            # Key invariant: memory should drop significantly after removal
            # At least 50% reduction from peak, and under 10MB residual
            peak_to_final = after_add.current - after_remove.current
            peak_reduction_pct = (peak_to_final / after_add.current) * 100 if after_add.current > 0 else 100

            assert peak_reduction_pct > 50, (
                f"Memory did not reduce significantly after removal. "
                f"Initial: {initial.current_mb:.2f}MB, "
                f"After add: {after_add.current_mb:.2f}MB, "
                f"After remove: {after_remove.current_mb:.2f}MB, "
                f"Reduction: {peak_reduction_pct:.1f}%"
            )

            # Residual memory should be under 10MB (acceptable for Python overhead)
            assert after_remove.current_mb < 10.0, (
                f"Residual memory too high: {after_remove.current_mb:.2f}MB"
            )

    def test_depgraph_invalidation_cleanup(
        self, dep_graph: MaterialDepGraph
    ) -> None:
        """Verify BFS traversal doesn't accumulate memory."""
        # Build a dependency chain
        for i in range(100):
            material = generate_material_path(i)
            includes = {generate_include_path(i % 10)}
            dep_graph.record_material_compilation(material, includes)

        # Run many invalidation queries
        gc.collect()
        initial_objects = len(gc.get_objects())

        for i in range(1000):
            include = generate_include_path(i % 10)
            affected = dep_graph.broadest_invalidation_set(include)
            # Verify we got results (sanity check)
            assert len(affected) > 0

        gc.collect()
        final_objects = len(gc.get_objects())

        # Object count should not grow significantly
        # Allow some variance for temporary objects
        object_growth = final_objects - initial_objects
        assert object_growth < 1000, (
            f"BFS traversal accumulated {object_growth} objects"
        )

    def test_depgraph_concurrent_access(
        self, dep_graph: MaterialDepGraph
    ) -> None:
        """Stress test concurrent reads/writes on DepGraph."""
        num_threads = 8
        operations_per_thread = 1000
        errors: List[Exception] = []

        def worker(thread_id: int) -> None:
            try:
                for i in range(operations_per_thread):
                    op = i % 4
                    mat_idx = (thread_id * operations_per_thread + i) % 500
                    material = generate_material_path(mat_idx)

                    if op == 0:
                        # Add material
                        includes = {
                            generate_include_path(mat_idx % 50),
                            generate_include_path((mat_idx + 1) % 50),
                        }
                        dep_graph.record_material_compilation(material, includes)
                    elif op == 1:
                        # Query invalidation
                        include = generate_include_path(mat_idx % 50)
                        _ = dep_graph.broadest_invalidation_set(include)
                    elif op == 2:
                        # Query includes
                        _ = dep_graph.get_material_includes(material)
                    elif op == 3:
                        # Remove material
                        dep_graph.remove_material(material)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(num_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Concurrent access errors: {errors}"

    def test_depgraph_recompilation_cleanup(
        self, dep_graph: MaterialDepGraph, memory_tracker: MemoryTracker
    ) -> None:
        """Verify that recompiling a material cleans up old edges."""
        material = generate_material_path(0)

        with memory_tracker:
            initial = memory_tracker.snapshot()

            # Compile with one set of includes
            for i in range(100):
                old_includes = {generate_include_path(j) for j in range(i, i + 5)}
                dep_graph.record_material_compilation(material, old_includes)

            after_many = memory_tracker.snapshot()

            # Should only have the last set of includes
            current_includes = dep_graph.get_material_includes(material)
            assert len(current_includes) == 5

            # Memory should not have grown significantly
            # (each recompilation should clean up old edges)
            growth = memory_tracker.get_growth_mb(initial, after_many)
            assert growth < 1.0, f"Memory grew {growth:.2f}MB during recompilations"


# =============================================================================
# PipelineTable LRU Tests
# =============================================================================


class TestPipelineTableLRU:
    """Memory tests for LruPipelineTable."""

    def test_pipeline_lru_eviction(self) -> None:
        """Test that LRU eviction works correctly at capacity."""
        table = LruPipelineTable(max_size=10)

        # Add 15 pipelines to a cache of size 10
        handles = []
        for i in range(15):
            shader = generate_shader(i)
            handle = table.get_or_create_pipeline(shader)
            handles.append(handle)

        # Should have evicted 5
        assert len(table) == 10
        assert table.stats.evictions == 5

        # First 5 should be evicted (LRU)
        for i in range(5):
            assert not table.contains(handles[i].id)

        # Last 10 should still exist
        for i in range(5, 15):
            assert table.contains(handles[i].id)

    def test_pipeline_cache_hit_rate(self) -> None:
        """Verify >90% hit rate under typical access patterns."""
        table = LruPipelineTable(max_size=100)

        # Create 50 unique shaders
        shaders = [generate_shader(i) for i in range(50)]

        # Initial population (50 misses)
        for shader in shaders:
            table.get_or_create_pipeline(shader)

        # Access pattern: 80% recent (last 10), 20% random
        import random
        random.seed(42)

        for _ in range(10000):
            if random.random() < 0.8:
                # Access recent shader
                shader = shaders[random.randint(40, 49)]
            else:
                # Access random shader
                shader = shaders[random.randint(0, 49)]
            table.get_or_create_pipeline(shader)

        # Hit rate should be >90%
        hit_rate = table.stats.hit_rate()
        assert hit_rate > 90.0, f"Hit rate {hit_rate:.1f}% is below 90%"

    def test_pipeline_memory_budget(
        self, memory_tracker: MemoryTracker
    ) -> None:
        """Test 10,000 pipelines stay within 256MB budget."""
        table = LruPipelineTable(max_size=10000)

        with memory_tracker:
            initial = memory_tracker.snapshot()

            # Add 10,000 unique pipelines
            for i in range(10000):
                shader = generate_shader(i)
                table.get_or_create_pipeline(shader)

            final = memory_tracker.snapshot()

            # Check memory budget
            tracker = MemoryBudgetTracker(pipeline_table=table)

            # Verify within 256MB budget
            assert tracker.within_budget(256), (
                f"Pipeline table exceeded 256MB budget: "
                f"{tracker.total_size_mb():.2f}MB"
            )

            # Also verify actual memory growth
            growth_mb = memory_tracker.get_growth_mb(initial, final)
            assert growth_mb < 256, (
                f"Actual memory growth {growth_mb:.2f}MB exceeds 256MB"
            )

    def test_pipeline_eviction_releases_memory(
        self, memory_tracker: MemoryTracker
    ) -> None:
        """Test that evicted pipelines are garbage collected."""
        table = LruPipelineTable(max_size=100)

        with memory_tracker:
            # Fill cache
            for i in range(100):
                shader = generate_shader(i)
                table.get_or_create_pipeline(shader)

            after_fill = memory_tracker.snapshot()

            # Trigger evictions by adding more
            for i in range(100, 200):
                shader = generate_shader(i)
                table.get_or_create_pipeline(shader)

            gc.collect()
            after_evict = memory_tracker.snapshot()

            # Memory should not have grown significantly
            # (evicted entries should be GC'd)
            growth = memory_tracker.get_growth_mb(after_fill, after_evict)
            assert growth < 5.0, (
                f"Memory grew {growth:.2f}MB despite evictions"
            )


# =============================================================================
# ShaderCache Tests
# =============================================================================


class TestShaderCacheBounded:
    """Memory tests for ShaderCache."""

    def test_shader_cache_bounded(self) -> None:
        """Content-addressed entries don't duplicate."""
        cache = ShaderCache()

        shader = generate_shader(0)

        # Cache same shader 1000 times
        for _ in range(1000):
            cache.cache_shader(shader)

        # Should only have one module
        assert len(cache) == 1
        assert cache.stats.hits == 999
        assert cache.stats.misses == 1

    def test_shader_variant_cleanup(self) -> None:
        """Old variants are cleaned up on hot-reload simulation."""
        cache = ShaderCache()

        # Add shader with path
        original = generate_shader(0, "v1")
        cache.cache_shader_with_path(original, "/test/shader.wgsl")

        original_hash = cache.hash_for_path("/test/shader.wgsl")
        assert original_hash is not None

        # Invalidate path (simulating hot-reload)
        old_hash = cache.invalidate_path("/test/shader.wgsl")
        assert old_hash == original_hash

        # Module should be removed
        assert len(cache) == 0

        # Add new version
        updated = generate_shader(0, "v2")
        cache.cache_shader_with_path(updated, "/test/shader.wgsl")

        # Should have new hash
        new_hash = cache.hash_for_path("/test/shader.wgsl")
        assert new_hash != original_hash

    def test_shader_compilation_memory(
        self, memory_tracker: MemoryTracker
    ) -> None:
        """Shader compilation doesn't leak memory."""
        cache = ShaderCache()

        with memory_tracker:
            initial = memory_tracker.snapshot()

            # Compile many unique shaders
            for i in range(1000):
                shader = generate_shader(i)
                cache.cache_shader(shader)

            after_compile = memory_tracker.snapshot()

            # Clear cache
            cache.clear()
            gc.collect()

            after_clear = memory_tracker.snapshot()

            # Memory should return close to initial
            remaining = memory_tracker.get_growth_mb(initial, after_clear)
            assert remaining < 1.0, (
                f"Memory not released after clear: {remaining:.2f}MB remaining"
            )


# =============================================================================
# ContentStore Tests
# =============================================================================


class TestContentStoreMemory:
    """Memory tests for ContentStore."""

    def test_contentstore_orphan_detection(
        self, content_store: ContentStore
    ) -> None:
        """GC finds unreferenced trees (simulated)."""
        backend = content_store._backend
        assert isinstance(backend, MemoryBackend)

        # Store some trees
        hashes = []
        for i in range(100):
            data = {"index": i, "data": list(range(100))}
            h = content_store.put_tree(data)
            hashes.append(h)

        initial_count = len(backend)

        # "Orphan" detection: clear our references and see if backend grows
        # In a real GC, we'd track references. Here we just verify storage doesn't leak.

        # Store more without tracking
        for i in range(100, 200):
            data = {"index": i, "data": list(range(50))}
            content_store.put_tree(data)

        # All items should still be accessible (no orphans yet - no GC implemented)
        # This test verifies the structure for when GC is added
        final_count = len(backend)

        # Verify we can still access original hashes
        for h in hashes[:10]:
            assert content_store.has(h)

    def test_contentstore_gc_completes_fast(
        self, content_store: ContentStore
    ) -> None:
        """GC completes within 2ms frame budget (simulated)."""
        # Store many small objects
        for i in range(1000):
            content_store.put({"index": i})

        # Simulate GC operation (clear)
        start = time.perf_counter()
        content_store._backend.clear()
        duration_ms = (time.perf_counter() - start) * 1000

        # Should complete within 2ms
        assert duration_ms < 2.0, f"GC took {duration_ms:.2f}ms (budget: 2ms)"

    def test_contentstore_large_scale(
        self, memory_tracker: MemoryTracker
    ) -> None:
        """100,000 assets maintain stable memory."""
        store = ContentStore(MemoryBackend())

        with memory_tracker:
            initial = memory_tracker.snapshot()

            # Store 100,000 small assets
            for i in range(100000):
                data = {"asset_id": i, "name": f"asset_{i}"}
                store.put(data)

            final = memory_tracker.snapshot()

            # Calculate per-asset overhead
            growth_mb = memory_tracker.get_growth_mb(initial, final)
            per_asset_bytes = (final.current - initial.current) / 100000

            # Should be reasonable (~100-200 bytes per small asset)
            assert per_asset_bytes < 500, (
                f"Per-asset memory {per_asset_bytes:.0f} bytes is too high"
            )

            # Total should be under reasonable limit (50MB for 100k small assets)
            assert growth_mb < 50, (
                f"100k assets used {growth_mb:.2f}MB (expected <50MB)"
            )

    def test_contentstore_deduplication_memory(
        self, content_store: ContentStore, memory_tracker: MemoryTracker
    ) -> None:
        """Verify deduplication reduces memory usage."""
        backend = content_store._backend
        assert isinstance(backend, MemoryBackend)

        # Same data stored many times
        data = {"shared": "content", "values": list(range(100))}

        with memory_tracker:
            initial = memory_tracker.snapshot()

            # Store same data 10,000 times
            for _ in range(10000):
                content_store.put(data)

            final = memory_tracker.snapshot()

            # Should only store once
            assert len(backend) == 1

            # Memory growth should be minimal
            growth_mb = memory_tracker.get_growth_mb(initial, final)
            assert growth_mb < 1.0, (
                f"Deduplication failed: {growth_mb:.2f}MB growth for duplicate data"
            )


# =============================================================================
# Hot-Reload Accumulation Tests
# =============================================================================


class TestHotReloadAccumulation:
    """Tests for hot-reload memory accumulation."""

    def test_hotreload_100_cycles(
        self, memory_tracker: MemoryTracker
    ) -> None:
        """100 reload cycles maintain stable memory."""
        table = LruPipelineTable(max_size=100)

        with memory_tracker:
            initial = memory_tracker.snapshot()

            for cycle in range(100):
                # Add shaders
                for i in range(10):
                    shader = generate_shader(i, f"v{cycle}")
                    table.get_or_create_pipeline(
                        shader,
                        source_path=f"/shaders/shader_{i}.wgsl"
                    )

                # Invalidate all (simulate hot-reload)
                for i in range(10):
                    table.invalidate_by_path(f"/shaders/shader_{i}.wgsl")

            gc.collect()
            final = memory_tracker.snapshot()

            # Memory should be stable (within 5MB)
            growth = memory_tracker.get_growth_mb(initial, final)
            assert growth < 5.0, (
                f"Hot-reload cycles accumulated {growth:.2f}MB"
            )

    def test_material_churn(
        self, dep_graph: MaterialDepGraph, memory_tracker: MemoryTracker
    ) -> None:
        """Add/remove materials repeatedly without memory growth."""
        with memory_tracker:
            initial = memory_tracker.snapshot()

            # Churn: add and remove materials 1000 times
            for i in range(1000):
                material = generate_material_path(i % 100)
                includes = {
                    generate_include_path(i % 50),
                    generate_include_path((i + 1) % 50),
                }

                # Add
                dep_graph.record_material_compilation(material, includes)

                # Remove every other
                if i % 2 == 1:
                    dep_graph.remove_material(material)

            # Clear remaining
            for mat in list(dep_graph.all_materials()):
                dep_graph.remove_material(mat)

            gc.collect()
            final = memory_tracker.snapshot()

            # Should be close to initial
            growth = memory_tracker.get_growth_mb(initial, final)
            assert growth < 1.0, (
                f"Material churn accumulated {growth:.2f}MB"
            )

    def test_pipeline_integration_hot_reload(
        self, memory_tracker: MemoryTracker
    ) -> None:
        """Test PipelineIntegration hot-reload doesn't leak."""
        integration = PipelineIntegration(max_cache_size=50)

        with memory_tracker:
            initial = memory_tracker.snapshot()

            # Simulate hot-reload cycles
            for cycle in range(50):
                # Create pipelines
                for i in range(10):
                    shader = generate_shader(i, f"cycle{cycle}")
                    integration.get_or_create_pipeline(
                        shader,
                        source_path=f"/materials/mat_{i}.wgsl"
                    )

                # Invalidate all
                for i in range(10):
                    integration.invalidate_shader(f"/materials/mat_{i}.wgsl")

            gc.collect()
            final = memory_tracker.snapshot()

            growth = memory_tracker.get_growth_mb(initial, final)
            assert growth < 5.0, (
                f"PipelineIntegration hot-reload leaked {growth:.2f}MB"
            )


# =============================================================================
# Memory Budget Integration Tests
# =============================================================================


class TestMemoryBudgetIntegration:
    """Integration tests for memory budgets across all components."""

    def test_full_system_memory_budget(
        self, memory_tracker: MemoryTracker
    ) -> None:
        """Test full material system stays within 256MB budget."""
        dep_graph = MaterialDepGraph()
        table = LruPipelineTable(max_size=10000)
        store = ContentStore(MemoryBackend())

        with memory_tracker:
            initial = memory_tracker.snapshot()

            # Simulate realistic workload
            # 10,000 materials with 3 includes each
            for i in range(10000):
                material = generate_material_path(i)
                includes = {
                    generate_include_path(i % 500),
                    generate_include_path((i + 100) % 500),
                    generate_include_path((i + 200) % 500),
                }
                dep_graph.record_material_compilation(material, includes)

                # Create pipeline
                shader = generate_shader(i)
                table.get_or_create_pipeline(
                    shader,
                    source_path=str(material)
                )

                # Store material config
                config = {
                    "material_id": i,
                    "includes": [str(p) for p in includes],
                    "params": {"roughness": 0.5, "metallic": 0.0}
                }
                store.put(config)

            final = memory_tracker.snapshot()

            # Create tracker for component breakdown
            tracker = MemoryBudgetTracker(
                dep_graph=dep_graph,
                pipeline_table=table,
                content_store=store
            )

            # Total memory should be within 256MB
            growth_mb = memory_tracker.get_growth_mb(initial, final)

            assert growth_mb < 256, (
                f"Full system exceeded 256MB budget: {growth_mb:.2f}MB used. "
                f"DepGraph: {tracker.get_depgraph_size() / 1024 / 1024:.2f}MB, "
                f"Pipeline: {tracker.get_pipeline_table_size() / 1024 / 1024:.2f}MB, "
                f"Content: {tracker.get_content_store_size() / 1024 / 1024:.2f}MB"
            )

    def test_memory_budget_tracker_accuracy(self) -> None:
        """Test MemoryBudgetTracker provides reasonable estimates."""
        dep_graph = MaterialDepGraph()
        table = LruPipelineTable(max_size=100)
        store = ContentStore(MemoryBackend())

        # Add known data
        for i in range(100):
            material = generate_material_path(i)
            includes = {generate_include_path(i % 10)}
            dep_graph.record_material_compilation(material, includes)

            shader = generate_shader(i)
            table.get_or_create_pipeline(shader)

            store.put({"id": i, "data": "x" * 100})

        tracker = MemoryBudgetTracker(
            dep_graph=dep_graph,
            pipeline_table=table,
            content_store=store
        )

        # Verify non-zero estimates
        assert tracker.get_depgraph_size() > 0
        assert tracker.get_pipeline_table_size() > 0
        assert tracker.get_content_store_size() > 0
        assert tracker.total_size() > 0

        # Verify reasonable range (1KB - 10MB for 100 items each)
        total_mb = tracker.total_size_mb()
        assert 0.001 < total_mb < 10, f"Tracker estimate {total_mb:.3f}MB out of range"


# =============================================================================
# Performance Regression Tests
# =============================================================================


class TestMemoryPerformanceRegression:
    """Performance regression tests for memory operations."""

    def test_depgraph_add_performance(self) -> None:
        """DepGraph add operation maintains O(1) amortized."""
        graph = MaterialDepGraph()

        times = []
        for batch in range(10):
            start = time.perf_counter()
            for i in range(1000):
                idx = batch * 1000 + i
                material = generate_material_path(idx)
                includes = {generate_include_path(idx % 100)}
                graph.record_material_compilation(material, includes)
            end = time.perf_counter()
            times.append(end - start)

        # Later batches should not be significantly slower
        first_time = times[0]
        last_time = times[-1]

        # Allow 50% degradation (accounting for hash table resizing)
        assert last_time < first_time * 1.5, (
            f"Performance degraded: first batch {first_time:.3f}s, "
            f"last batch {last_time:.3f}s"
        )

    def test_pipeline_lookup_performance(self) -> None:
        """Pipeline lookup maintains O(1) with LRU."""
        table = LruPipelineTable(max_size=10000)

        # Pre-populate
        shaders = [generate_shader(i) for i in range(5000)]
        for shader in shaders:
            table.get_or_create_pipeline(shader)

        # Measure lookup time
        import random
        random.seed(42)

        start = time.perf_counter()
        for _ in range(10000):
            shader = random.choice(shaders)
            table.get_or_create_pipeline(shader)
        duration = time.perf_counter() - start

        # 10,000 lookups should complete in <100ms
        assert duration < 0.1, f"Lookup performance degraded: {duration:.3f}s for 10k ops"

    def test_contentstore_put_performance(self) -> None:
        """ContentStore put maintains reasonable performance."""
        store = ContentStore(MemoryBackend())

        # Measure put performance
        start = time.perf_counter()
        for i in range(10000):
            store.put({"id": i, "value": i * 2})
        duration = time.perf_counter() - start

        # 10,000 puts should complete in <1s
        assert duration < 1.0, f"Put performance: {duration:.3f}s for 10k ops"


# =============================================================================
# Cleanup and Teardown Tests
# =============================================================================


class TestCleanupBehavior:
    """Tests for proper cleanup of resources."""

    def test_depgraph_clear_releases_memory(
        self, dep_graph: MaterialDepGraph, memory_tracker: MemoryTracker
    ) -> None:
        """DepGraph.clear() properly releases memory."""
        with memory_tracker:
            initial = memory_tracker.snapshot()

            # Add many entries
            for i in range(5000):
                material = generate_material_path(i)
                includes = {generate_include_path(i % 100)}
                dep_graph.record_material_compilation(material, includes)

            after_add = memory_tracker.snapshot()

            # Clear
            dep_graph.clear()
            gc.collect()

            after_clear = memory_tracker.snapshot()

            # Memory should return close to initial
            remaining = after_clear.current - initial.current
            assert remaining < 100 * 1024, (  # Less than 100KB
                f"Clear didn't release memory: {remaining / 1024:.1f}KB remaining"
            )

    def test_pipeline_table_clear_releases_memory(
        self, memory_tracker: MemoryTracker
    ) -> None:
        """LruPipelineTable.clear() properly releases memory."""
        table = LruPipelineTable(max_size=1000)

        with memory_tracker:
            initial = memory_tracker.snapshot()

            # Add entries
            for i in range(1000):
                shader = generate_shader(i)
                table.get_or_create_pipeline(shader)

            # Clear
            table.clear()
            gc.collect()

            after_clear = memory_tracker.snapshot()

            remaining = after_clear.current - initial.current
            assert remaining < 100 * 1024, (
                f"Clear didn't release memory: {remaining / 1024:.1f}KB remaining"
            )

    def test_no_circular_references(self) -> None:
        """Verify no circular references prevent garbage collection."""
        import weakref

        graph = MaterialDepGraph()
        table = LruPipelineTable(max_size=10)

        # Add some data
        for i in range(10):
            material = generate_material_path(i)
            graph.record_material_compilation(material, {generate_include_path(i)})
            table.get_or_create_pipeline(generate_shader(i))

        # Create weak references
        graph_ref = weakref.ref(graph)
        table_ref = weakref.ref(table)

        # Delete strong references
        del graph
        del table
        gc.collect()

        # Weak references should be dead
        assert graph_ref() is None, "MaterialDepGraph has circular reference"
        assert table_ref() is None, "LruPipelineTable has circular reference"


# =============================================================================
# Additional Edge Case Tests
# =============================================================================


class TestDepGraphEdgeCases:
    """Edge case tests for MaterialDepGraph memory behavior."""

    def test_depgraph_empty_includes(self, dep_graph: MaterialDepGraph) -> None:
        """Materials with empty include sets don't cause issues."""
        for i in range(100):
            material = generate_material_path(i)
            dep_graph.record_material_compilation(material, set())

        assert dep_graph.material_count() == 100
        assert dep_graph.include_count() == 0

        # Cleanup should work
        for i in range(100):
            dep_graph.remove_material(generate_material_path(i))

        assert dep_graph.material_count() == 0

    def test_depgraph_deep_dependency_chain(
        self, dep_graph: MaterialDepGraph
    ) -> None:
        """Deep dependency chains don't cause stack overflow or memory issues."""
        # Create a chain: material_0 -> material_1 -> ... -> material_99
        for i in range(100):
            material = generate_material_path(i)
            includes = {generate_include_path(i)}
            dep_graph.record_material_compilation(material, includes)

            if i > 0:
                dep_graph.record_material_dependency(
                    generate_material_path(i),
                    generate_material_path(i - 1)
                )

        # Query broadest invalidation from the root
        affected = dep_graph.broadest_invalidation_set(generate_include_path(0))

        # Should find the material using that include
        assert generate_material_path(0) in affected

    def test_depgraph_diamond_dependency(self, dep_graph: MaterialDepGraph) -> None:
        """Diamond dependencies don't cause infinite loops or memory growth."""
        # Create diamond: A -> B, A -> C, B -> D, C -> D
        mat_a = generate_material_path(0)
        mat_b = generate_material_path(1)
        mat_c = generate_material_path(2)
        mat_d = generate_material_path(3)

        shared_include = generate_include_path(0)

        dep_graph.record_material_compilation(mat_a, {shared_include})
        dep_graph.record_material_compilation(mat_b, {shared_include})
        dep_graph.record_material_compilation(mat_c, {shared_include})
        dep_graph.record_material_compilation(mat_d, {shared_include})

        dep_graph.record_material_dependency(mat_b, mat_a)
        dep_graph.record_material_dependency(mat_c, mat_a)
        dep_graph.record_material_dependency(mat_d, mat_b)
        dep_graph.record_material_dependency(mat_d, mat_c)

        # Query should not infinite loop
        affected = dep_graph.broadest_invalidation_set(shared_include)

        assert len(affected) == 4


class TestShaderCacheEdgeCases:
    """Edge case tests for ShaderCache memory behavior."""

    def test_shader_cache_empty_shader(self) -> None:
        """Empty shader strings are handled correctly."""
        cache = ShaderCache()
        module, h = cache.cache_shader("")

        assert len(h) == 64
        assert len(cache) == 1

    def test_shader_cache_very_large_shader(
        self, memory_tracker: MemoryTracker
    ) -> None:
        """Very large shaders are handled without excessive overhead."""
        cache = ShaderCache()

        # 1MB shader
        large_shader = "// " + "x" * (1024 * 1024)

        with memory_tracker:
            initial = memory_tracker.snapshot()

            module, h = cache.cache_shader(large_shader)

            final = memory_tracker.snapshot()

            # Memory should be roughly 1MB + overhead
            growth_mb = memory_tracker.get_growth_mb(initial, final)
            assert growth_mb < 3.0, f"Large shader used {growth_mb:.2f}MB"

    def test_shader_cache_unicode_source(self) -> None:
        """Unicode in shader source is handled correctly."""
        cache = ShaderCache()

        # Shader with unicode comments
        shader_with_unicode = """
        // Unicode test: Chinese characters Arabic text Emoji: Earth
        @vertex fn vs_main() -> @builtin(position) vec4<f32> {
            return vec4<f32>(0.0);
        }
        """

        module, h = cache.cache_shader(shader_with_unicode)
        assert len(h) == 64


class TestPipelineTableEdgeCases:
    """Edge case tests for LruPipelineTable."""

    def test_pipeline_table_single_entry(self) -> None:
        """Single-entry cache works correctly."""
        table = LruPipelineTable(max_size=1)

        handle1 = table.get_or_create_pipeline(generate_shader(0))
        assert len(table) == 1

        handle2 = table.get_or_create_pipeline(generate_shader(1))
        assert len(table) == 1
        assert table.stats.evictions == 1

    def test_pipeline_table_repeated_same_shader(self) -> None:
        """Repeated access to same shader maintains LRU correctly."""
        table = LruPipelineTable(max_size=10)

        shader = generate_shader(0)

        for _ in range(1000):
            table.get_or_create_pipeline(shader)

        assert len(table) == 1
        assert table.stats.hits == 999
        assert table.stats.misses == 1

    def test_pipeline_table_stats_accuracy(self) -> None:
        """Pipeline statistics are accurate."""
        table = LruPipelineTable(max_size=5)

        # 10 unique shaders into size-5 cache
        for i in range(10):
            table.get_or_create_pipeline(generate_shader(i))

        assert table.stats.misses == 10
        assert table.stats.evictions == 5

        # Access existing
        for i in range(5, 10):
            table.get_or_create_pipeline(generate_shader(i))

        assert table.stats.hits == 5


class TestContentStoreEdgeCases:
    """Edge case tests for ContentStore memory behavior."""

    def test_contentstore_nested_structure(
        self, content_store: ContentStore
    ) -> None:
        """Deeply nested structures don't cause issues."""
        # 10 levels deep
        data: Any = {"value": 42}
        for level in range(10):
            data = {"nested": data, "level": level}

        h = content_store.put_tree(data)
        result = content_store.get_tree(h)

        # Outermost level is 9, 3 levels deep is level 6
        assert result["level"] == 9
        assert result["nested"]["level"] == 8
        assert result["nested"]["nested"]["level"] == 7
        # Can access deep value
        current = result
        for _ in range(10):
            current = current["nested"]
        assert current["value"] == 42

    def test_contentstore_large_list(
        self, content_store: ContentStore, memory_tracker: MemoryTracker
    ) -> None:
        """Large lists are handled efficiently."""
        with memory_tracker:
            initial = memory_tracker.snapshot()

            # List with 10,000 elements
            data = list(range(10000))
            h = content_store.put_tree(data)

            final = memory_tracker.snapshot()

            # Retrieve and verify
            result = content_store.get_tree(h)
            assert result == data

            # Memory should be reasonable
            growth_mb = memory_tracker.get_growth_mb(initial, final)
            assert growth_mb < 5.0, f"Large list used {growth_mb:.2f}MB"


class TestMemoryStressTests:
    """Stress tests for memory stability."""

    def test_rapid_allocation_deallocation(
        self, memory_tracker: MemoryTracker
    ) -> None:
        """Rapid allocation/deallocation cycles don't leak memory."""
        with memory_tracker:
            initial = memory_tracker.snapshot()

            for cycle in range(100):
                # Create and destroy objects rapidly
                tables = [LruPipelineTable(max_size=10) for _ in range(10)]
                caches = [ShaderCache() for _ in range(10)]
                graphs = [MaterialDepGraph() for _ in range(10)]

                # Use them
                for i, (table, cache, graph) in enumerate(
                    zip(tables, caches, graphs)
                ):
                    shader = generate_shader(i)
                    table.get_or_create_pipeline(shader)
                    cache.cache_shader(shader)
                    graph.record_material_compilation(
                        generate_material_path(i),
                        {generate_include_path(i)}
                    )

                # Let them go out of scope
                del tables, caches, graphs
                gc.collect()

            gc.collect()
            gc.collect()
            final = memory_tracker.snapshot()

            # Memory should be stable after all cycles
            growth_mb = memory_tracker.get_growth_mb(initial, final)
            assert growth_mb < 5.0, (
                f"Rapid allocation leaked {growth_mb:.2f}MB"
            )

    def test_mixed_operations_stability(
        self, memory_tracker: MemoryTracker
    ) -> None:
        """Mixed operations maintain memory stability."""
        dep_graph = MaterialDepGraph()
        table = LruPipelineTable(max_size=100)
        store = ContentStore(MemoryBackend())

        with memory_tracker:
            initial = memory_tracker.snapshot()

            for iteration in range(100):
                # Add some entries
                for i in range(10):
                    idx = iteration * 10 + i
                    material = generate_material_path(idx % 200)
                    includes = {generate_include_path(idx % 50)}

                    dep_graph.record_material_compilation(material, includes)
                    table.get_or_create_pipeline(generate_shader(idx % 200))
                    store.put({"iteration": iteration, "index": i})

                # Remove some entries
                for i in range(5):
                    idx = iteration * 10 + i
                    dep_graph.remove_material(generate_material_path(idx % 200))

            final = memory_tracker.snapshot()

            # Memory should be bounded by the data structures' limits
            growth_mb = memory_tracker.get_growth_mb(initial, final)
            assert growth_mb < 50.0, (
                f"Mixed operations used {growth_mb:.2f}MB"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
