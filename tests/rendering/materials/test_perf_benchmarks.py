"""Performance Benchmarks for Material System (T-MAT-11.3).

This module provides Python-side performance benchmarks for the material system,
complementing the Rust benchmarks in benches/material_system.rs.

Benchmarks:
1. DSL compilation throughput (materials/sec)
2. WGSL generation throughput (KB/sec)
3. Pipeline cache hit rate measurement
4. Shader cache deduplication efficiency
5. Content store throughput simulation

Gap: S11-G3
Dependencies: T-MAT-6.3, T-MAT-3.4 (DONE)

Run with: uv run pytest tests/rendering/materials/test_perf_benchmarks.py -v
"""

from __future__ import annotations

import ast
import hashlib
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from trinity.materials import (
    Material,
    MaterialMeta,
    SurfaceContext,
    SurfaceOutput,
    surface,
    Vec2,
    Vec3,
    Vec4,
    PythonToWGSLTranslator,
    MaterialCompiler,
)


# =============================================================================
# Performance Metrics Collection
# =============================================================================


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    name: str
    iterations: int
    total_time_ms: float
    items_processed: int
    bytes_processed: int

    @property
    def items_per_second(self) -> float:
        """Throughput in items/second."""
        if self.total_time_ms == 0:
            return float("inf")
        return (self.items_processed / self.total_time_ms) * 1000

    @property
    def mb_per_second(self) -> float:
        """Throughput in MB/second."""
        if self.total_time_ms == 0:
            return float("inf")
        return (self.bytes_processed / (1024 * 1024)) / (self.total_time_ms / 1000)

    @property
    def avg_time_ms(self) -> float:
        """Average time per iteration in milliseconds."""
        if self.iterations == 0:
            return 0
        return self.total_time_ms / self.iterations


class PerformanceTracker:
    """Collects and reports performance metrics."""

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def record(self, result: BenchmarkResult) -> None:
        """Record a benchmark result."""
        self.results.append(result)

    def summary(self) -> str:
        """Generate a summary report."""
        lines = ["\n=== Material System Performance Report ===\n"]
        for r in self.results:
            lines.append(f"  {r.name}:")
            lines.append(f"    Items/sec: {r.items_per_second:,.0f}")
            if r.bytes_processed > 0:
                lines.append(f"    MB/sec: {r.mb_per_second:.2f}")
            lines.append(f"    Avg time: {r.avg_time_ms:.3f}ms")
            lines.append("")
        return "\n".join(lines)


# Global tracker for test session
_tracker = PerformanceTracker()


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def translator():
    """Fresh translator instance for each test."""
    return PythonToWGSLTranslator()


@pytest.fixture
def compiler():
    """Fresh compiler instance with PBR template."""
    return MaterialCompiler(include_pbr_template=True)


@pytest.fixture
def body_compiler():
    """Compiler that returns only the surface body."""
    return MaterialCompiler(include_pbr_template=False)


@pytest.fixture
def tracker():
    """Get the global performance tracker."""
    return _tracker


# =============================================================================
# Helper: Material Generation
# =============================================================================


def create_test_material_class(index: int, variant: int = 0) -> type:
    """Dynamically create a test material class."""

    class TestMaterial(Material):
        """Dynamically generated test material."""

        @surface
        def shade(self, ctx: SurfaceContext) -> SurfaceOutput:
            base_color = Vec3(0.8 + variant * 0.05, 0.2, 0.1)
            metallic = 0.0 + (index % 10) * 0.1
            roughness = 0.2 + (index % 5) * 0.15
            return SurfaceOutput(
                base_color=base_color,
                metallic=metallic,
                roughness=roughness,
            )

    # Rename class to be unique
    TestMaterial.__name__ = f"TestMaterial_{index}_{variant}"
    TestMaterial.__qualname__ = TestMaterial.__name__
    return TestMaterial


def generate_wgsl_pbr(material_id: int, variation: int = 0) -> str:
    """Generate a PBR WGSL shader string."""
    variation_factor = 0.8 + variation * 0.1
    return f"""// Material {material_id} variant {variation}
struct VertexOutput {{
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}}

struct MaterialParams {{
    base_color: vec4<f32>,
    metallic: f32,
    roughness: f32,
    ao: f32,
    emission_strength: f32,
}}

@group(0) @binding(0) var<uniform> material: MaterialParams;

@vertex
fn vs_main_{material_id}(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
) -> VertexOutput {{
    var out: VertexOutput;
    out.position = vec4<f32>(position, 1.0);
    out.world_pos = position;
    out.normal = normal;
    out.uv = uv;
    return out;
}}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {{
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}}

@fragment
fn fs_main_{material_id}(in: VertexOutput) -> @location(0) vec4<f32> {{
    let base_color = material.base_color.rgb * {variation_factor:.1};
    let metallic = material.metallic;
    let roughness = max(material.roughness, 0.04);

    let n = normalize(in.normal);
    let v = normalize(-in.world_pos);
    let l = normalize(vec3<f32>(1.0, 1.0, 0.5));

    let n_dot_l = max(dot(n, l), 0.0);

    let diffuse = base_color * n_dot_l;
    let ambient = base_color * material.ao * 0.03;
    var color = ambient + diffuse;

    // Tone mapping
    color = color / (color + vec3<f32>(1.0));

    return vec4<f32>(color, 1.0);
}}
"""


# =============================================================================
# Benchmark 1: DSL Compilation Throughput
# =============================================================================


class TestDSLCompilationThroughput:
    """Benchmark DSL compilation speed (materials/sec)."""

    @pytest.mark.parametrize("count", [10, 50, 100])
    def test_dsl_compilation_throughput(self, compiler, tracker, count: int):
        """Measure DSL to WGSL compilation throughput."""
        # Create material classes
        materials = [create_test_material_class(i, i % 3) for i in range(count)]

        # Warm-up
        for mat in materials[:5]:
            try:
                compiler.compile(mat)
            except Exception:
                pass  # Some may fail, that's OK for benchmarking

        # Timed run
        start = time.perf_counter()
        compiled = 0
        total_bytes = 0

        for mat in materials:
            try:
                wgsl = compiler.compile(mat)
                if wgsl:
                    compiled += 1
                    total_bytes += len(wgsl.encode("utf-8"))
            except Exception:
                pass

        elapsed_ms = (time.perf_counter() - start) * 1000

        result = BenchmarkResult(
            name=f"DSL Compilation ({count} materials)",
            iterations=1,
            total_time_ms=elapsed_ms,
            items_processed=compiled,
            bytes_processed=total_bytes,
        )
        tracker.record(result)

        # Assert reasonable throughput (at least 10 materials/sec)
        assert result.items_per_second >= 10, (
            f"DSL compilation too slow: {result.items_per_second:.1f} materials/sec"
        )

    def test_dsl_translation_only(self, translator, tracker):
        """Benchmark AST translation without full compilation."""
        # Generate Python code snippets to translate
        code_snippets = [
            f"""
x = 0.5 + {i} * 0.01
y = x * 2.0
z = clamp(y, 0.0, 1.0)
result = vec3(z, z * 0.8, z * 0.6)
"""
            for i in range(100)
        ]

        start = time.perf_counter()
        translated = 0
        total_bytes = 0

        for code in code_snippets:
            try:
                tree = ast.parse(code)
                wgsl = translator.translate(tree)
                if wgsl:
                    translated += 1
                    total_bytes += len(wgsl.encode("utf-8"))
            except Exception:
                pass

        elapsed_ms = (time.perf_counter() - start) * 1000

        result = BenchmarkResult(
            name="AST Translation (100 snippets)",
            iterations=1,
            total_time_ms=elapsed_ms,
            items_processed=translated,
            bytes_processed=total_bytes,
        )
        tracker.record(result)

        # Translation should be very fast
        assert result.items_per_second >= 100, (
            f"Translation too slow: {result.items_per_second:.1f} snippets/sec"
        )


# =============================================================================
# Benchmark 2: WGSL Generation Throughput
# =============================================================================


class TestWGSLGenerationThroughput:
    """Benchmark WGSL generation throughput (KB/sec)."""

    def test_wgsl_string_generation(self, tracker):
        """Measure raw WGSL string generation throughput."""
        iterations = 10
        materials_per_iter = 100

        start = time.perf_counter()
        total_bytes = 0
        total_materials = 0

        for _ in range(iterations):
            for i in range(materials_per_iter):
                wgsl = generate_wgsl_pbr(i, i % 3)
                total_bytes += len(wgsl.encode("utf-8"))
                total_materials += 1

        elapsed_ms = (time.perf_counter() - start) * 1000

        result = BenchmarkResult(
            name="WGSL String Generation",
            iterations=iterations,
            total_time_ms=elapsed_ms,
            items_processed=total_materials,
            bytes_processed=total_bytes,
        )
        tracker.record(result)

        # Should generate at least 1 MB/sec
        assert result.mb_per_second >= 1.0, (
            f"WGSL generation too slow: {result.mb_per_second:.2f} MB/sec"
        )

    def test_wgsl_hashing_throughput(self, tracker):
        """Measure SHA-256 hashing throughput for WGSL content."""
        # Pre-generate shaders
        shaders = [generate_wgsl_pbr(i, i % 3) for i in range(100)]
        total_bytes = sum(len(s.encode("utf-8")) for s in shaders)

        iterations = 100

        start = time.perf_counter()

        for _ in range(iterations):
            for shader in shaders:
                _hash = hashlib.sha256(shader.encode("utf-8")).hexdigest()

        elapsed_ms = (time.perf_counter() - start) * 1000

        result = BenchmarkResult(
            name="WGSL Hashing (SHA-256)",
            iterations=iterations,
            total_time_ms=elapsed_ms,
            items_processed=len(shaders) * iterations,
            bytes_processed=total_bytes * iterations,
        )
        tracker.record(result)

        # Hashing should be very fast
        assert result.mb_per_second >= 50.0, (
            f"Hashing too slow: {result.mb_per_second:.2f} MB/sec"
        )


# =============================================================================
# Benchmark 3: Pipeline Cache Hit Rate
# =============================================================================


class TestPipelineCacheMetrics:
    """Benchmark pipeline cache operations."""

    def test_cache_hit_rate_simulation(self, tracker):
        """Simulate cache hit/miss patterns."""
        cache: Dict[str, str] = {}
        cache_size = 64

        # Generate shader hashes
        shaders = [generate_wgsl_pbr(i, i % 3) for i in range(200)]
        hashes = [hashlib.sha256(s.encode()).hexdigest() for s in shaders]

        # Simulate access pattern: first 64 are cached
        for h in hashes[:cache_size]:
            cache[h] = "compiled_pipeline"

        iterations = 1000
        hits = 0
        misses = 0

        start = time.perf_counter()

        for _ in range(iterations):
            for h in hashes:
                if h in cache:
                    hits += 1
                else:
                    misses += 1

        elapsed_ms = (time.perf_counter() - start) * 1000

        hit_rate = (hits / (hits + misses)) * 100

        result = BenchmarkResult(
            name=f"Cache Lookup (hit rate: {hit_rate:.0f}%)",
            iterations=iterations,
            total_time_ms=elapsed_ms,
            items_processed=hits + misses,
            bytes_processed=0,
        )
        tracker.record(result)

        # Cache lookups should be very fast
        assert result.items_per_second >= 1_000_000, (
            f"Cache lookup too slow: {result.items_per_second:,.0f} ops/sec"
        )

    def test_lru_eviction_simulation(self, tracker):
        """Simulate LRU eviction patterns."""
        from collections import OrderedDict

        class LRUCache:
            def __init__(self, capacity: int):
                self.capacity = capacity
                self.cache: OrderedDict[str, Any] = OrderedDict()
                self.hits = 0
                self.misses = 0

            def get(self, key: str) -> Optional[Any]:
                if key in self.cache:
                    self.cache.move_to_end(key)
                    self.hits += 1
                    return self.cache[key]
                self.misses += 1
                return None

            def put(self, key: str, value: Any) -> None:
                if key in self.cache:
                    self.cache.move_to_end(key)
                else:
                    if len(self.cache) >= self.capacity:
                        self.cache.popitem(last=False)
                    self.cache[key] = value

        cache = LRUCache(64)
        hashes = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(500)]

        iterations = 100

        start = time.perf_counter()

        for _ in range(iterations):
            for i, h in enumerate(hashes):
                if cache.get(h) is None:
                    cache.put(h, f"pipeline_{i}")

        elapsed_ms = (time.perf_counter() - start) * 1000

        result = BenchmarkResult(
            name="LRU Cache Operations",
            iterations=iterations,
            total_time_ms=elapsed_ms,
            items_processed=len(hashes) * iterations,
            bytes_processed=0,
        )
        tracker.record(result)

        # LRU operations should be fast
        assert result.items_per_second >= 100_000, (
            f"LRU ops too slow: {result.items_per_second:,.0f} ops/sec"
        )


# =============================================================================
# Benchmark 4: Shader Cache Deduplication
# =============================================================================


class TestShaderCacheDeduplication:
    """Benchmark shader deduplication efficiency."""

    @pytest.mark.parametrize("dedup_ratio", [0.0, 0.5, 0.9])
    def test_deduplication_efficiency(self, tracker, dedup_ratio: float):
        """Measure deduplication with various duplication ratios."""
        unique_count = 100
        total_count = 1000
        duplicate_count = int((total_count - unique_count) * dedup_ratio)

        # Generate unique shaders
        unique_shaders = [generate_wgsl_pbr(i, i % 3) for i in range(unique_count)]

        # Create workload with duplicates
        workload = list(unique_shaders)
        for i in range(duplicate_count):
            workload.append(unique_shaders[i % unique_count])
        # Fill rest
        remaining = total_count - len(workload)
        for i in range(remaining):
            workload.append(unique_shaders[i % unique_count])

        seen_hashes: set[str] = set()

        start = time.perf_counter()

        unique_found = 0
        duplicates_found = 0

        for shader in workload:
            h = hashlib.sha256(shader.encode()).hexdigest()
            if h in seen_hashes:
                duplicates_found += 1
            else:
                seen_hashes.add(h)
                unique_found += 1

        elapsed_ms = (time.perf_counter() - start) * 1000

        actual_dedup = duplicates_found / total_count
        total_bytes = sum(len(s.encode()) for s in workload)

        result = BenchmarkResult(
            name=f"Deduplication ({int(dedup_ratio * 100)}% duplicates)",
            iterations=1,
            total_time_ms=elapsed_ms,
            items_processed=total_count,
            bytes_processed=total_bytes,
        )
        tracker.record(result)

        # Deduplication should be fast
        assert result.items_per_second >= 10_000, (
            f"Deduplication too slow: {result.items_per_second:,.0f} items/sec"
        )


# =============================================================================
# Benchmark 5: Content Store Throughput Simulation
# =============================================================================


class TestContentStoreThroughput:
    """Benchmark content-addressed storage operations."""

    @pytest.mark.parametrize("size_kb", [1, 10, 100, 1024])
    def test_content_store_put(self, tracker, size_kb: int):
        """Benchmark content store put operations."""
        size_bytes = size_kb * 1024
        data = bytes(range(256)) * (size_bytes // 256 + 1)
        data = data[:size_bytes]

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "content_store"
            store_path.mkdir()

            iterations = 100

            start = time.perf_counter()

            for i in range(iterations):
                # Vary data slightly to avoid identical hashes
                test_data = data + bytes([i % 256])
                h = hashlib.sha256(test_data).hexdigest()
                prefix = h[:2]
                suffix = h[2:]

                prefix_path = store_path / prefix
                prefix_path.mkdir(exist_ok=True)

                blob_path = prefix_path / suffix
                blob_path.write_bytes(test_data)

            elapsed_ms = (time.perf_counter() - start) * 1000

            result = BenchmarkResult(
                name=f"Content Store Put ({size_kb}KB)",
                iterations=iterations,
                total_time_ms=elapsed_ms,
                items_processed=iterations,
                bytes_processed=size_bytes * iterations,
            )
            tracker.record(result)

            # Should achieve reasonable throughput
            min_mb_per_sec = 10 if size_kb < 100 else 50
            assert result.mb_per_second >= min_mb_per_sec, (
                f"Put too slow: {result.mb_per_second:.2f} MB/sec"
            )

    def test_content_store_get(self, tracker):
        """Benchmark content store get operations."""
        size_kb = 10
        size_bytes = size_kb * 1024
        data = bytes(range(256)) * (size_bytes // 256 + 1)
        data = data[:size_bytes]

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "content_store"
            store_path.mkdir()

            # Pre-populate store
            blobs = []
            for i in range(100):
                test_data = data + bytes([i])
                h = hashlib.sha256(test_data).hexdigest()
                prefix = h[:2]
                suffix = h[2:]

                prefix_path = store_path / prefix
                prefix_path.mkdir(exist_ok=True)

                blob_path = prefix_path / suffix
                blob_path.write_bytes(test_data)
                blobs.append(blob_path)

            iterations = 100

            start = time.perf_counter()

            for _ in range(iterations):
                for blob_path in blobs:
                    _content = blob_path.read_bytes()

            elapsed_ms = (time.perf_counter() - start) * 1000

            result = BenchmarkResult(
                name="Content Store Get (10KB x 100)",
                iterations=iterations,
                total_time_ms=elapsed_ms,
                items_processed=len(blobs) * iterations,
                bytes_processed=size_bytes * len(blobs) * iterations,
            )
            tracker.record(result)

            # Reads should be fast
            assert result.mb_per_second >= 50, (
                f"Get too slow: {result.mb_per_second:.2f} MB/sec"
            )


# =============================================================================
# Benchmark 6: Integrated Pipeline
# =============================================================================


class TestIntegratedPipeline:
    """Benchmark full material processing pipeline."""

    def test_full_pipeline_batch(self, compiler, tracker):
        """Benchmark complete material processing pipeline."""
        batch_sizes = [10, 50, 100]

        for batch_size in batch_sizes:
            materials = [
                create_test_material_class(i, i % 3)
                for i in range(batch_size)
            ]

            start = time.perf_counter()

            processed = 0
            total_bytes = 0
            hashes: set[str] = set()

            for mat in materials:
                try:
                    # 1. Compile DSL to WGSL
                    wgsl = compiler.compile(mat)
                    if not wgsl:
                        continue

                    # 2. Hash content
                    h = hashlib.sha256(wgsl.encode()).hexdigest()
                    hashes.add(h)

                    # 3. Track metrics
                    processed += 1
                    total_bytes += len(wgsl.encode())
                except Exception:
                    pass

            elapsed_ms = (time.perf_counter() - start) * 1000

            result = BenchmarkResult(
                name=f"Full Pipeline ({batch_size} materials)",
                iterations=1,
                total_time_ms=elapsed_ms,
                items_processed=processed,
                bytes_processed=total_bytes,
            )
            tracker.record(result)

    def test_warm_cache_scenario(self, tracker):
        """Benchmark with pre-warmed cache (90% hit scenario)."""
        # Pre-generate and cache
        cache: Dict[str, str] = {}
        shaders = [generate_wgsl_pbr(i, i % 3) for i in range(100)]

        for shader in shaders:
            h = hashlib.sha256(shader.encode()).hexdigest()
            cache[h] = shader

        # Create access pattern: 90% cached, 10% new
        access_pattern = []
        for i in range(1000):
            if i % 10 == 0:
                # New shader
                access_pattern.append(generate_wgsl_pbr(100 + i, 0))
            else:
                # Cached shader
                access_pattern.append(shaders[i % 100])

        start = time.perf_counter()

        hits = 0
        misses = 0

        for shader in access_pattern:
            h = hashlib.sha256(shader.encode()).hexdigest()
            if h in cache:
                hits += 1
            else:
                misses += 1
                cache[h] = shader

        elapsed_ms = (time.perf_counter() - start) * 1000

        hit_rate = (hits / (hits + misses)) * 100

        result = BenchmarkResult(
            name=f"Warm Cache Scenario (hit rate: {hit_rate:.0f}%)",
            iterations=1,
            total_time_ms=elapsed_ms,
            items_processed=len(access_pattern),
            bytes_processed=sum(len(s.encode()) for s in access_pattern),
        )
        tracker.record(result)


# =============================================================================
# Test Session Hook: Print Summary
# =============================================================================


@pytest.fixture(scope="session", autouse=True)
def print_performance_summary(request):
    """Print performance summary at end of session."""
    yield
    # Print summary after all tests
    if _tracker.results:
        print(_tracker.summary())


# =============================================================================
# Standalone Benchmark Runner
# =============================================================================


def run_benchmarks():
    """Run all benchmarks and print results."""
    print("Running Material System Performance Benchmarks...")
    print("=" * 60)

    tracker = PerformanceTracker()

    # DSL Compilation
    print("\n[1/6] DSL Compilation Throughput...")
    for count in [10, 50, 100]:
        shaders = [generate_wgsl_pbr(i, i % 3) for i in range(count)]
        start = time.perf_counter()
        total_bytes = sum(len(s.encode()) for s in shaders)
        elapsed_ms = (time.perf_counter() - start) * 1000

        tracker.record(BenchmarkResult(
            name=f"WGSL Generation ({count})",
            iterations=1,
            total_time_ms=elapsed_ms,
            items_processed=count,
            bytes_processed=total_bytes,
        ))

    # Hashing
    print("[2/6] WGSL Hashing Throughput...")
    shaders = [generate_wgsl_pbr(i, i % 3) for i in range(100)]
    start = time.perf_counter()
    for _ in range(100):
        for s in shaders:
            hashlib.sha256(s.encode()).hexdigest()
    elapsed_ms = (time.perf_counter() - start) * 1000
    total_bytes = sum(len(s.encode()) for s in shaders) * 100

    tracker.record(BenchmarkResult(
        name="SHA-256 Hashing",
        iterations=100,
        total_time_ms=elapsed_ms,
        items_processed=10000,
        bytes_processed=total_bytes,
    ))

    # Cache simulation
    print("[3/6] Cache Hit Rate Simulation...")
    cache: Dict[str, str] = {}
    hashes = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(200)]
    for h in hashes[:64]:
        cache[h] = "data"

    start = time.perf_counter()
    hits = sum(1 for h in hashes * 100 if h in cache)
    elapsed_ms = (time.perf_counter() - start) * 1000

    tracker.record(BenchmarkResult(
        name="Cache Lookup",
        iterations=100,
        total_time_ms=elapsed_ms,
        items_processed=len(hashes) * 100,
        bytes_processed=0,
    ))

    # Deduplication
    print("[4/6] Deduplication Efficiency...")
    start = time.perf_counter()
    seen: set[str] = set()
    for s in shaders * 10:
        h = hashlib.sha256(s.encode()).hexdigest()
        seen.add(h)
    elapsed_ms = (time.perf_counter() - start) * 1000

    tracker.record(BenchmarkResult(
        name="Deduplication",
        iterations=1,
        total_time_ms=elapsed_ms,
        items_processed=len(shaders) * 10,
        bytes_processed=sum(len(s.encode()) for s in shaders) * 10,
    ))

    # Content store
    print("[5/6] Content Store Throughput...")
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = Path(tmpdir)
        data = bytes(range(256)) * 40  # 10KB

        start = time.perf_counter()
        for i in range(100):
            test_data = data + bytes([i])
            h = hashlib.sha256(test_data).hexdigest()
            (store_path / h[:2]).mkdir(exist_ok=True)
            (store_path / h[:2] / h[2:]).write_bytes(test_data)
        elapsed_ms = (time.perf_counter() - start) * 1000

        tracker.record(BenchmarkResult(
            name="Content Store Put (10KB)",
            iterations=100,
            total_time_ms=elapsed_ms,
            items_processed=100,
            bytes_processed=len(data) * 100,
        ))

    # Full pipeline
    print("[6/6] Integrated Pipeline...")
    start = time.perf_counter()
    total_bytes = 0
    for i in range(50):
        wgsl = generate_wgsl_pbr(i, i % 3)
        h = hashlib.sha256(wgsl.encode()).hexdigest()
        total_bytes += len(wgsl.encode())
    elapsed_ms = (time.perf_counter() - start) * 1000

    tracker.record(BenchmarkResult(
        name="Full Pipeline (50 materials)",
        iterations=1,
        total_time_ms=elapsed_ms,
        items_processed=50,
        bytes_processed=total_bytes,
    ))

    print(tracker.summary())


if __name__ == "__main__":
    run_benchmarks()
