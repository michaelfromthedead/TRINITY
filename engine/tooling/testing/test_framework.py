"""
Test framework with @test and @bench decorators.

Provides decorator-based test definition for the AI Game Engine,
supporting unit tests, benchmarks, and various test configurations.
"""

from __future__ import annotations

import functools
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class BenchmarkResult:
    """Result of a benchmark execution."""

    name: str
    iterations: int
    total_time: float
    min_time: float
    max_time: float
    mean_time: float
    median_time: float
    std_dev: float
    ops_per_second: float
    memory_delta: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.name}: {self.mean_time*1000:.3f}ms mean "
            f"({self.min_time*1000:.3f}-{self.max_time*1000:.3f}ms), "
            f"{self.ops_per_second:.1f} ops/s"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_time": self.total_time,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "mean_time": self.mean_time,
            "median_time": self.median_time,
            "std_dev": self.std_dev,
            "ops_per_second": self.ops_per_second,
            "memory_delta": self.memory_delta,
            "metadata": self.metadata,
        }


def test(
    func: Optional[F] = None,
    *,
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    priority: int = 0,
    timeout: Optional[float] = None,
    expected_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
) -> Union[F, Callable[[F], F]]:
    """
    Decorator to mark a function as a test.

    Args:
        func: The test function (when used without parentheses)
        name: Custom test name
        tags: Tags for filtering tests
        priority: Test priority (higher runs first)
        timeout: Test timeout in seconds
        expected_exceptions: Exceptions that should be raised

    Example:
        @test
        def test_addition():
            assert 1 + 1 == 2

        @test(tags=["slow", "integration"])
        def test_database_connection():
            ...
    """
    def decorator(f: F) -> F:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if expected_exceptions:
                try:
                    result = f(*args, **kwargs)
                    raise AssertionError(
                        f"Expected one of {expected_exceptions} to be raised"
                    )
                except expected_exceptions:
                    return None
            return f(*args, **kwargs)

        wrapper._is_test = True
        wrapper._test_name = name or f.__name__
        wrapper._test_tags = set(tags) if tags else set()
        wrapper._test_priority = priority
        if timeout is not None:
            wrapper._timeout = timeout
        wrapper._expected_exceptions = expected_exceptions

        return wrapper  # type: ignore

    if func is not None:
        return decorator(func)
    return decorator


def bench(
    func: Optional[F] = None,
    *,
    name: Optional[str] = None,
    iterations: int = 1000,
    warmup: int = 100,
    min_time: float = 0.1,
    max_time: float = 10.0,
    track_memory: bool = False,
    compare_to: Optional[str] = None,
) -> Union[F, Callable[[F], F]]:
    """
    Decorator to mark a function as a benchmark.

    Args:
        func: The benchmark function (when used without parentheses)
        name: Custom benchmark name
        iterations: Number of iterations to run
        warmup: Number of warmup iterations
        min_time: Minimum time to run benchmark
        max_time: Maximum time to run benchmark
        track_memory: Whether to track memory usage
        compare_to: Name of another benchmark to compare against

    Example:
        @bench
        def bench_list_append():
            lst = []
            for i in range(1000):
                lst.append(i)

        @bench(iterations=10000, track_memory=True)
        def bench_dict_creation():
            return {i: i*2 for i in range(100)}
    """
    def decorator(f: F) -> F:
        @functools.wraps(f)
        def wrapper(*args, **kwargs) -> BenchmarkResult:
            bench_name = name or f.__name__

            # Warmup
            for _ in range(warmup):
                f(*args, **kwargs)

            # Track memory if requested
            memory_before = 0
            if track_memory:
                import tracemalloc
                tracemalloc.start()
                memory_before = tracemalloc.get_traced_memory()[0]

            # Run benchmark
            times: List[float] = []
            total_start = time.perf_counter()

            iter_count = 0
            while True:
                start = time.perf_counter()
                f(*args, **kwargs)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                iter_count += 1

                total_elapsed = time.perf_counter() - total_start

                # Check termination conditions
                if iter_count >= iterations:
                    break
                if total_elapsed >= max_time:
                    break
                if total_elapsed >= min_time and iter_count >= 10:
                    break

            # Memory tracking
            memory_delta = 0
            if track_memory:
                import tracemalloc
                memory_after = tracemalloc.get_traced_memory()[0]
                memory_delta = memory_after - memory_before
                tracemalloc.stop()

            # Calculate statistics
            total_time = sum(times)
            mean_time = statistics.mean(times)

            result = BenchmarkResult(
                name=bench_name,
                iterations=len(times),
                total_time=total_time,
                min_time=min(times),
                max_time=max(times),
                mean_time=mean_time,
                median_time=statistics.median(times),
                std_dev=statistics.stdev(times) if len(times) > 1 else 0.0,
                ops_per_second=1.0 / mean_time if mean_time > 0 else 0.0,
                memory_delta=memory_delta,
            )

            return result

        wrapper._is_benchmark = True
        wrapper._bench_name = name or f.__name__
        wrapper._bench_iterations = iterations
        wrapper._bench_warmup = warmup
        wrapper._bench_compare_to = compare_to

        return wrapper  # type: ignore

    if func is not None:
        return decorator(func)
    return decorator


def skip(
    func: Optional[F] = None,
    *,
    reason: str = "",
) -> Union[F, Callable[[F], F]]:
    """
    Decorator to skip a test.

    Args:
        func: The test function
        reason: Reason for skipping

    Example:
        @skip(reason="Not implemented yet")
        def test_future_feature():
            ...
    """
    def decorator(f: F) -> F:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            pass  # Test is skipped

        wrapper._skip = True
        wrapper._skip_reason = reason
        # Preserve test markers
        wrapper._is_test = getattr(f, "_is_test", True)

        return wrapper  # type: ignore

    if func is not None:
        return decorator(func)
    return decorator


def skip_if(
    condition: Callable[[], bool],
    reason: str = "",
) -> Callable[[F], F]:
    """
    Decorator to conditionally skip a test.

    Args:
        condition: Function that returns True if test should be skipped
        reason: Reason for skipping

    Example:
        @skip_if(lambda: sys.platform != 'linux', reason="Linux only")
        def test_linux_feature():
            ...
    """
    def decorator(f: F) -> F:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if condition():
                return None
            return f(*args, **kwargs)

        wrapper._skip_if = condition
        wrapper._skip_if_reason = reason
        wrapper._is_test = getattr(f, "_is_test", True)

        return wrapper  # type: ignore

    return decorator


def expected_failure(
    func: Optional[F] = None,
    *,
    reason: str = "",
) -> Union[F, Callable[[F], F]]:
    """
    Decorator to mark a test as expected to fail.

    Args:
        func: The test function
        reason: Reason for expected failure

    Example:
        @expected_failure(reason="Known bug #123")
        def test_buggy_feature():
            assert buggy_function() == expected_value
    """
    def decorator(f: F) -> F:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        wrapper._expected_failure = True
        wrapper._expected_failure_reason = reason
        wrapper._is_test = getattr(f, "_is_test", True)

        return wrapper  # type: ignore

    if func is not None:
        return decorator(func)
    return decorator


def timeout(seconds: float) -> Callable[[F], F]:
    """
    Decorator to set a timeout for a test.

    Args:
        seconds: Timeout in seconds

    Example:
        @timeout(5.0)
        def test_slow_operation():
            ...
    """
    def decorator(f: F) -> F:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        wrapper._timeout = seconds
        wrapper._is_test = getattr(f, "_is_test", True)

        return wrapper  # type: ignore

    return decorator


def parametrize(
    param_names: Union[str, List[str]],
    param_values: List[Any],
    ids: Optional[List[str]] = None,
) -> Callable[[F], List[F]]:
    """
    Decorator to create parameterized tests.

    Args:
        param_names: Names of parameters (comma-separated string or list)
        param_values: List of parameter value tuples
        ids: Optional test IDs for each parameter set

    Example:
        @parametrize("a, b, expected", [
            (1, 2, 3),
            (0, 0, 0),
            (-1, 1, 0),
        ])
        def test_addition(a, b, expected):
            assert a + b == expected
    """
    if isinstance(param_names, str):
        param_names = [p.strip() for p in param_names.split(",")]

    def decorator(f: F) -> List[F]:
        tests = []

        for i, values in enumerate(param_values):
            if not isinstance(values, (tuple, list)):
                values = (values,)

            if ids and i < len(ids):
                test_id = f"[{ids[i]}]"
            else:
                test_id = f"[{i}]"

            @functools.wraps(f)
            def wrapper(*args, _values=values, **kwargs):
                # Bind parameter values
                bound_kwargs = dict(zip(param_names, _values))
                bound_kwargs.update(kwargs)
                return f(*args, **bound_kwargs)

            wrapper.__name__ = f"{f.__name__}{test_id}"
            wrapper._is_test = True
            wrapper._test_name = wrapper.__name__
            wrapper._test_tags = getattr(f, "_test_tags", set())
            wrapper._parametrized = True
            wrapper._param_values = values

            tests.append(wrapper)

        return tests  # type: ignore

    return decorator


class TestCase:
    """
    Base class for test cases with setup/teardown support.

    Example:
        class TestCalculator(TestCase):
            def setup(self):
                self.calc = Calculator()

            def teardown(self):
                self.calc.reset()

            def test_addition(self):
                assert self.calc.add(1, 2) == 3
    """

    @classmethod
    def setup_class(cls) -> None:
        """Called once before any tests in the class."""
        pass

    @classmethod
    def teardown_class(cls) -> None:
        """Called once after all tests in the class."""
        pass

    def setup(self) -> None:
        """Called before each test method."""
        pass

    def teardown(self) -> None:
        """Called after each test method."""
        pass

    def setup_method(self, method: Callable) -> None:
        """Called before each test method with the method reference."""
        self.setup()

    def teardown_method(self, method: Callable) -> None:
        """Called after each test method with the method reference."""
        self.teardown()

    def run_test(self, test_name: str) -> bool:
        """Run a specific test method."""
        method = getattr(self, test_name, None)
        if method is None:
            raise ValueError(f"Test method '{test_name}' not found")

        try:
            self.setup_method(method)
            method()
            return True
        except AssertionError:
            return False
        finally:
            self.teardown_method(method)

    @classmethod
    def get_test_methods(cls) -> List[str]:
        """Get all test method names in this class."""
        return [
            name for name in dir(cls)
            if name.startswith("test_") and callable(getattr(cls, name))
        ]


class BenchmarkSuite:
    """
    Collection of benchmarks with comparison support.

    Example:
        suite = BenchmarkSuite("String Operations")

        @suite.add
        @bench
        def bench_concat():
            s = ""
            for i in range(100):
                s += str(i)

        @suite.add
        @bench
        def bench_join():
            "".join(str(i) for i in range(100))

        results = suite.run()
        suite.print_comparison(results)
    """

    def __init__(self, name: str):
        self.name = name
        self.benchmarks: List[Callable[[], BenchmarkResult]] = []
        self.results: Dict[str, BenchmarkResult] = {}

    def add(self, func: F) -> F:
        """Add a benchmark to the suite."""
        self.benchmarks.append(func)
        return func

    def run(self) -> Dict[str, BenchmarkResult]:
        """Run all benchmarks and return results."""
        self.results.clear()

        for benchmark in self.benchmarks:
            result = benchmark()
            self.results[result.name] = result

        return self.results

    def print_comparison(self, baseline: Optional[str] = None) -> None:
        """Print comparison table of benchmark results."""
        if not self.results:
            print("No results to compare")
            return

        print(f"\n{self.name} Benchmark Results")
        print("=" * 60)

        # Find baseline
        baseline_result = None
        if baseline and baseline in self.results:
            baseline_result = self.results[baseline]
        elif self.results:
            baseline_result = list(self.results.values())[0]

        # Print results
        for name, result in sorted(self.results.items()):
            relative = ""
            if baseline_result and baseline_result.mean_time > 0:
                ratio = result.mean_time / baseline_result.mean_time
                if result != baseline_result:
                    if ratio < 1:
                        relative = f" ({(1-ratio)*100:.1f}% faster)"
                    else:
                        relative = f" ({(ratio-1)*100:.1f}% slower)"

            print(f"{name}: {result.mean_time*1000:.3f}ms{relative}")

        print("=" * 60)
