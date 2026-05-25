"""
Benchmark suite for the engine testing framework.

Provides infrastructure for running performance benchmarks with
statistical analysis and comparison support.

Usage:
    from engine.debug.testing.benchmarks import Benchmark, bench, BenchmarkSuite

    @bench(iterations=1000, warmup=100)
    def bench_vector_add():
        Vec3(1, 2, 3) + Vec3(4, 5, 6)

    # Or use the class-based approach
    benchmark = Benchmark("vector_add", lambda: Vec3(1, 2, 3) + Vec3(4, 5, 6))
    result = benchmark.run(iterations=1000, warmup=100)
    print(f"Average: {result.avg_ms:.4f}ms")
"""

from __future__ import annotations

import functools
import gc
import statistics
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)


__all__ = [
    "BenchmarkResult",
    "Benchmark",
    "BenchmarkSuite",
    "BenchmarkComparison",
    "bench",
    "run_benchmarks",
]


T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

# Constants for magic numbers - benchmark configuration
DEFAULT_BENCHMARK_ITERATIONS = 1000
DEFAULT_BENCHMARK_WARMUP = 100
# Significance threshold for benchmark comparisons (5%)
DEFAULT_SIGNIFICANCE_THRESHOLD = 0.05


@dataclass
class BenchmarkResult:
    """
    Result of a benchmark execution.

    Attributes:
        name: Benchmark name
        iterations: Number of iterations run
        total_ms: Total execution time in milliseconds
        avg_ms: Average time per iteration
        min_ms: Minimum time per iteration
        max_ms: Maximum time per iteration
        std_dev_ms: Standard deviation of iteration times
        median_ms: Median time per iteration
        percentile_95_ms: 95th percentile time
        percentile_99_ms: 99th percentile time
        warmup_iterations: Number of warmup iterations
        gc_collections: Number of GC collections during benchmark
        memory_delta_bytes: Memory change during benchmark (if measured)
    """

    name: str
    iterations: int
    total_ms: float
    avg_ms: float
    min_ms: float
    max_ms: float
    std_dev_ms: float = 0.0
    median_ms: float = 0.0
    percentile_95_ms: float = 0.0
    percentile_99_ms: float = 0.0
    warmup_iterations: int = 0
    gc_collections: int = 0
    memory_delta_bytes: int = 0
    raw_times_ms: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ops_per_second(self) -> float:
        """Calculate operations per second."""
        if self.avg_ms <= 0:
            return float("inf")
        return 1000.0 / self.avg_ms

    @property
    def variance_coefficient(self) -> float:
        """Calculate coefficient of variation (std_dev / avg)."""
        if self.avg_ms <= 0:
            return 0.0
        return self.std_dev_ms / self.avg_ms

    def compare_to(self, baseline: "BenchmarkResult") -> "BenchmarkComparison":
        """
        Compare this result to a baseline.

        Args:
            baseline: The baseline result to compare against

        Returns:
            BenchmarkComparison with relative metrics
        """
        return BenchmarkComparison(
            name=self.name,
            current=self,
            baseline=baseline,
        )

    def __repr__(self) -> str:
        return (
            f"BenchmarkResult({self.name!r}, "
            f"avg={self.avg_ms:.4f}ms, "
            f"min={self.min_ms:.4f}ms, "
            f"max={self.max_ms:.4f}ms, "
            f"n={self.iterations})"
        )

    def format_report(self) -> str:
        """Format a detailed report string."""
        lines = [
            f"Benchmark: {self.name}",
            f"  Iterations: {self.iterations}",
            f"  Warmup: {self.warmup_iterations}",
            f"  Total: {self.total_ms:.4f}ms",
            f"  Average: {self.avg_ms:.6f}ms",
            f"  Min: {self.min_ms:.6f}ms",
            f"  Max: {self.max_ms:.6f}ms",
            f"  Std Dev: {self.std_dev_ms:.6f}ms",
            f"  Median: {self.median_ms:.6f}ms",
            f"  95th %ile: {self.percentile_95_ms:.6f}ms",
            f"  99th %ile: {self.percentile_99_ms:.6f}ms",
            f"  Ops/sec: {self.ops_per_second:,.0f}",
        ]
        if self.gc_collections > 0:
            lines.append(f"  GC Collections: {self.gc_collections}")
        if self.memory_delta_bytes != 0:
            lines.append(f"  Memory Delta: {self.memory_delta_bytes:,} bytes")
        return "\n".join(lines)


@dataclass
class BenchmarkComparison:
    """
    Comparison between two benchmark results.

    Attributes:
        name: Benchmark name
        current: Current benchmark result
        baseline: Baseline benchmark result
    """

    name: str
    current: BenchmarkResult
    baseline: BenchmarkResult

    @property
    def speedup(self) -> float:
        """
        Calculate speedup factor.

        Returns:
            Ratio of baseline to current (>1 means faster)
        """
        if self.current.avg_ms <= 0:
            return float("inf")
        return self.baseline.avg_ms / self.current.avg_ms

    @property
    def percent_change(self) -> float:
        """
        Calculate percent change from baseline.

        Returns:
            Percent change (negative = faster)
        """
        if self.baseline.avg_ms <= 0:
            return 0.0
        return ((self.current.avg_ms - self.baseline.avg_ms) / self.baseline.avg_ms) * 100

    @property
    def is_faster(self) -> bool:
        """Check if current is faster than baseline."""
        return self.current.avg_ms < self.baseline.avg_ms

    @property
    def is_slower(self) -> bool:
        """Check if current is slower than baseline."""
        return self.current.avg_ms > self.baseline.avg_ms

    @property
    def is_significant(self, threshold: float = DEFAULT_SIGNIFICANCE_THRESHOLD) -> bool:
        """
        Check if the difference is statistically significant.

        Uses a simple threshold on coefficient of variation.

        Args:
            threshold: Minimum percent change to consider significant

        Returns:
            True if difference exceeds threshold
        """
        return abs(self.percent_change) > threshold * 100

    def format_report(self) -> str:
        """Format a comparison report string."""
        direction = "faster" if self.is_faster else "slower"
        return (
            f"Comparison: {self.name}\n"
            f"  Current:  {self.current.avg_ms:.6f}ms\n"
            f"  Baseline: {self.baseline.avg_ms:.6f}ms\n"
            f"  Change:   {self.percent_change:+.2f}% ({direction})\n"
            f"  Speedup:  {self.speedup:.2f}x"
        )


class Benchmark:
    """
    A single benchmark that can be run multiple times.

    Provides precise timing with warmup, GC control, and
    statistical analysis of results.

    Example:
        benchmark = Benchmark("my_operation", lambda: expensive_operation())
        result = benchmark.run(iterations=1000, warmup=100)
        print(result.format_report())
    """

    def __init__(
        self,
        name: str,
        func: Callable[[], Any],
        setup: Optional[Callable[[], Any]] = None,
        teardown: Optional[Callable[[Any], None]] = None,
    ) -> None:
        """
        Initialize a benchmark.

        Args:
            name: Benchmark name
            func: Function to benchmark (should take no arguments)
            setup: Optional setup function called before each iteration batch
            teardown: Optional teardown function called after each iteration batch
        """
        self.name = name
        self.func = func
        self.setup = setup
        self.teardown = teardown
        self._last_result: Optional[BenchmarkResult] = None

    def run(
        self,
        iterations: int = DEFAULT_BENCHMARK_ITERATIONS,
        warmup: int = DEFAULT_BENCHMARK_WARMUP,
        gc_disable: bool = True,
        record_all: bool = False,
    ) -> BenchmarkResult:
        """
        Run the benchmark.

        Args:
            iterations: Number of iterations to run
            warmup: Number of warmup iterations (not counted)
            gc_disable: If True, disable GC during measurement
            record_all: If True, record all individual iteration times

        Returns:
            BenchmarkResult with statistics
        """
        func = self.func
        times: List[float] = []

        # Setup
        setup_result = None
        if self.setup:
            setup_result = self.setup()

        # Track GC
        gc_before = gc.get_count()

        # Warmup
        for _ in range(warmup):
            func()

        # Disable GC if requested
        gc_enabled = gc.isenabled()
        if gc_disable and gc_enabled:
            gc.disable()

        try:
            # Run timed iterations
            for _ in range(iterations):
                start = time.perf_counter_ns()
                func()
                end = time.perf_counter_ns()
                times.append((end - start) / 1_000_000)  # Convert to ms

        finally:
            if gc_disable and gc_enabled:
                gc.enable()

        # Track GC collections
        gc_after = gc.get_count()
        gc_collections = sum(gc_after) - sum(gc_before)

        # Teardown
        if self.teardown:
            self.teardown(setup_result)

        # Calculate statistics
        total_ms = sum(times)
        avg_ms = statistics.mean(times)
        min_ms = min(times)
        max_ms = max(times)
        std_dev_ms = statistics.stdev(times) if len(times) > 1 else 0.0
        median_ms = statistics.median(times)

        # Percentiles
        sorted_times = sorted(times)
        p95_idx = int(len(sorted_times) * 0.95)
        p99_idx = int(len(sorted_times) * 0.99)
        percentile_95_ms = sorted_times[p95_idx] if p95_idx < len(sorted_times) else max_ms
        percentile_99_ms = sorted_times[p99_idx] if p99_idx < len(sorted_times) else max_ms

        result = BenchmarkResult(
            name=self.name,
            iterations=iterations,
            total_ms=total_ms,
            avg_ms=avg_ms,
            min_ms=min_ms,
            max_ms=max_ms,
            std_dev_ms=std_dev_ms,
            median_ms=median_ms,
            percentile_95_ms=percentile_95_ms,
            percentile_99_ms=percentile_99_ms,
            warmup_iterations=warmup,
            gc_collections=gc_collections,
            raw_times_ms=times if record_all else [],
        )

        self._last_result = result
        return result

    def get_result(self) -> Optional[BenchmarkResult]:
        """
        Get the result of the last run.

        Returns:
            The last BenchmarkResult or None if never run
        """
        return self._last_result

    def __repr__(self) -> str:
        return f"Benchmark({self.name!r})"


class BenchmarkSuite:
    """
    A collection of benchmarks that can be run together.

    Example:
        suite = BenchmarkSuite("math_operations")

        @suite.benchmark
        def bench_add():
            return 1 + 2

        @suite.benchmark(iterations=10000)
        def bench_multiply():
            return 3 * 4

        results = suite.run()
    """

    # Registry of all suites
    _registry: ClassVar[Dict[str, "BenchmarkSuite"]] = {}

    def __init__(
        self,
        name: str,
        description: str = "",
        default_iterations: int = DEFAULT_BENCHMARK_ITERATIONS,
        default_warmup: int = DEFAULT_BENCHMARK_WARMUP,
    ) -> None:
        """
        Initialize a benchmark suite.

        Args:
            name: Suite name
            description: Human-readable description
            default_iterations: Default iteration count for benchmarks
            default_warmup: Default warmup iteration count
        """
        self.name = name
        self.description = description
        self.default_iterations = default_iterations
        self.default_warmup = default_warmup
        self._benchmarks: Dict[str, Tuple[Benchmark, int, int]] = {}
        self._results: Dict[str, BenchmarkResult] = {}

        BenchmarkSuite._registry[name] = self

    @classmethod
    def get(cls, name: str) -> Optional["BenchmarkSuite"]:
        """Get a suite by name from the registry."""
        return cls._registry.get(name)

    def add(
        self,
        benchmark: Benchmark,
        iterations: Optional[int] = None,
        warmup: Optional[int] = None,
    ) -> None:
        """
        Add a benchmark to the suite.

        Args:
            benchmark: The benchmark to add
            iterations: Custom iteration count (uses default if None)
            warmup: Custom warmup count (uses default if None)
        """
        self._benchmarks[benchmark.name] = (
            benchmark,
            iterations or self.default_iterations,
            warmup or self.default_warmup,
        )

    def benchmark(
        self,
        func: Optional[F] = None,
        *,
        name: Optional[str] = None,
        iterations: Optional[int] = None,
        warmup: Optional[int] = None,
        setup: Optional[Callable[[], Any]] = None,
        teardown: Optional[Callable[[Any], None]] = None,
    ) -> F:
        """
        Decorator to add a benchmark to the suite.

        Can be used with or without arguments:
            @suite.benchmark
            def bench_foo(): ...

            @suite.benchmark(iterations=5000)
            def bench_bar(): ...

        Args:
            func: The function to benchmark
            name: Custom benchmark name (uses function name if None)
            iterations: Custom iteration count
            warmup: Custom warmup count
            setup: Setup function
            teardown: Teardown function

        Returns:
            The decorated function
        """
        def decorator(f: F) -> F:
            bench_name = name or f.__name__
            bench = Benchmark(bench_name, f, setup=setup, teardown=teardown)
            self.add(bench, iterations, warmup)
            return f

        if func is not None:
            return decorator(func)
        return decorator  # type: ignore

    def run(
        self,
        filter_pattern: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, BenchmarkResult]:
        """
        Run all benchmarks in the suite.

        Args:
            filter_pattern: Glob pattern to filter benchmarks
            verbose: If True, print progress

        Returns:
            Dictionary of benchmark name to result
        """
        import fnmatch

        self._results.clear()

        if verbose:
            print(f"Running benchmark suite: {self.name}")
            print("=" * 60)

        for bench_name, (benchmark, iterations, warmup) in self._benchmarks.items():
            if filter_pattern and not fnmatch.fnmatch(bench_name, filter_pattern):
                continue

            if verbose:
                print(f"  {bench_name}...", end=" ", flush=True)

            result = benchmark.run(iterations=iterations, warmup=warmup)
            self._results[bench_name] = result

            if verbose:
                print(f"{result.avg_ms:.6f}ms avg ({result.ops_per_second:,.0f} ops/sec)")

        if verbose:
            print("=" * 60)

        return self._results.copy()

    def get_results(self) -> Dict[str, BenchmarkResult]:
        """Get results from the last run."""
        return self._results.copy()

    def compare_to(
        self,
        baseline_results: Dict[str, BenchmarkResult],
    ) -> Dict[str, BenchmarkComparison]:
        """
        Compare current results to baseline results.

        Args:
            baseline_results: Dictionary of baseline results

        Returns:
            Dictionary of benchmark name to comparison
        """
        comparisons = {}
        for name, current in self._results.items():
            if name in baseline_results:
                comparisons[name] = current.compare_to(baseline_results[name])
        return comparisons

    def __repr__(self) -> str:
        return f"BenchmarkSuite({self.name!r}, {len(self._benchmarks)} benchmarks)"


# Global registry for @bench decorated functions
_global_benchmarks: Dict[str, Tuple[Callable, int, int]] = {}


def bench(
    func: Optional[F] = None,
    *,
    iterations: int = DEFAULT_BENCHMARK_ITERATIONS,
    warmup: int = DEFAULT_BENCHMARK_WARMUP,
    name: Optional[str] = None,
) -> F:
    """
    Decorator to mark a function as a benchmark.

    The decorated function can be run via run_benchmarks().

    Example:
        @bench(iterations=5000, warmup=500)
        def bench_vector_normalize():
            v = Vec3(1, 2, 3)
            return v.normalized()

        # Later, run all benchmarks
        results = run_benchmarks()
    """
    def decorator(f: F) -> F:
        bench_name = name or f.__name__
        _global_benchmarks[bench_name] = (f, iterations, warmup)

        # Store metadata on function
        f._benchmark = True  # type: ignore
        f._benchmark_name = bench_name  # type: ignore
        f._benchmark_iterations = iterations  # type: ignore
        f._benchmark_warmup = warmup  # type: ignore

        return f

    if func is not None:
        return decorator(func)
    return decorator  # type: ignore


def run_benchmarks(
    filter_pattern: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, BenchmarkResult]:
    """
    Run all benchmarks decorated with @bench.

    Args:
        filter_pattern: Glob pattern to filter benchmarks
        verbose: If True, print progress

    Returns:
        Dictionary of benchmark name to result
    """
    import fnmatch

    results = {}

    if verbose:
        print("Running benchmarks")
        print("=" * 60)

    for bench_name, (func, iterations, warmup) in _global_benchmarks.items():
        if filter_pattern and not fnmatch.fnmatch(bench_name, filter_pattern):
            continue

        if verbose:
            print(f"  {bench_name}...", end=" ", flush=True)

        benchmark = Benchmark(bench_name, func)
        result = benchmark.run(iterations=iterations, warmup=warmup)
        results[bench_name] = result

        if verbose:
            print(f"{result.avg_ms:.6f}ms avg ({result.ops_per_second:,.0f} ops/sec)")

    if verbose:
        print("=" * 60)

    return results
