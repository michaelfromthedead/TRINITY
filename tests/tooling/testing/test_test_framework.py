"""
Tests for test_framework.py - Test decorators and framework.
"""

import pytest

from engine.tooling.testing.test_framework import (
    BenchmarkResult,
    BenchmarkSuite,
    TestCase,
    bench,
    expected_failure,
    parametrize,
    skip,
    skip_if,
    test,
    timeout,
)


class TestTestDecorator:
    """Tests for @test decorator."""

    def test_basic_decorator(self):
        @test
        def test_example():
            pass

        assert test_example._is_test is True
        assert test_example._test_name == "test_example"

    def test_decorator_with_name(self):
        @test(name="CustomName")
        def test_example():
            pass

        assert test_example._test_name == "CustomName"

    def test_decorator_with_tags(self):
        @test(tags=["unit", "fast"])
        def test_example():
            pass

        assert "unit" in test_example._test_tags
        assert "fast" in test_example._test_tags

    def test_decorator_with_priority(self):
        @test(priority=10)
        def test_high_priority():
            pass

        assert test_high_priority._test_priority == 10

    def test_decorator_with_timeout(self):
        @test(timeout=5.0)
        def test_with_timeout():
            pass

        assert test_with_timeout._timeout == 5.0

    def test_decorator_with_expected_exceptions(self):
        @test(expected_exceptions=(ValueError,))
        def test_raises():
            raise ValueError("expected")

        # Should not raise when ValueError is raised
        test_raises()

    def test_decorator_expected_exception_not_raised(self):
        @test(expected_exceptions=(ValueError,))
        def test_no_raise():
            pass

        with pytest.raises(AssertionError):
            test_no_raise()


class TestBenchDecorator:
    """Tests for @bench decorator."""

    def test_basic_bench(self):
        @bench
        def bench_example():
            x = sum(range(100))
            return x

        assert bench_example._is_benchmark is True
        result = bench_example()
        assert isinstance(result, BenchmarkResult)

    def test_bench_with_iterations(self):
        @bench(iterations=100)
        def bench_example():
            pass

        assert bench_example._bench_iterations == 100

    def test_bench_with_warmup(self):
        @bench(warmup=10)
        def bench_example():
            pass

        assert bench_example._bench_warmup == 10

    def test_bench_result_fields(self):
        @bench(iterations=10, warmup=2)
        def bench_example():
            pass

        result = bench_example()

        assert result.name == "bench_example"
        assert result.iterations >= 1
        assert result.total_time > 0
        assert result.min_time >= 0
        assert result.max_time >= result.min_time
        assert result.mean_time > 0
        assert result.ops_per_second > 0

    def test_bench_with_memory_tracking(self):
        @bench(track_memory=True, iterations=10)
        def bench_allocate():
            return [i for i in range(1000)]

        result = bench_allocate()
        # Memory tracking should capture some allocation
        assert isinstance(result.memory_delta, int)


class TestSkipDecorator:
    """Tests for @skip decorator."""

    def test_skip_basic(self):
        @skip
        def test_skipped():
            assert False  # Should not run

        assert test_skipped._skip is True

    def test_skip_with_reason(self):
        @skip(reason="Not implemented")
        def test_skipped():
            pass

        assert test_skipped._skip is True
        assert test_skipped._skip_reason == "Not implemented"


class TestSkipIfDecorator:
    """Tests for @skip_if decorator."""

    def test_skip_if_condition_true(self):
        @skip_if(lambda: True, reason="Condition met")
        def test_skip():
            return "ran"

        result = test_skip()
        assert result is None  # Skipped

    def test_skip_if_condition_false(self):
        @skip_if(lambda: False, reason="Condition not met")
        def test_run():
            return "ran"

        result = test_run()
        assert result == "ran"


class TestExpectedFailureDecorator:
    """Tests for @expected_failure decorator."""

    def test_expected_failure_basic(self):
        @expected_failure
        def test_fails():
            pass

        assert test_fails._expected_failure is True

    def test_expected_failure_with_reason(self):
        @expected_failure(reason="Known bug #123")
        def test_fails():
            pass

        assert test_fails._expected_failure_reason == "Known bug #123"


class TestTimeoutDecorator:
    """Tests for @timeout decorator."""

    def test_timeout_decorator(self):
        @timeout(5.0)
        def test_with_timeout():
            pass

        assert test_with_timeout._timeout == 5.0


class TestParametrizeDecorator:
    """Tests for @parametrize decorator."""

    def test_parametrize_basic(self):
        @parametrize("value", [1, 2, 3])
        def test_values(value):
            return value

        assert len(test_values) == 3

    def test_parametrize_multiple_params(self):
        @parametrize("a, b, expected", [
            (1, 2, 3),
            (0, 0, 0),
            (-1, 1, 0),
        ])
        def test_add(a, b, expected):
            assert a + b == expected

        assert len(test_add) == 3
        # Each should be callable
        for test_func in test_add:
            test_func()

    def test_parametrize_with_ids(self):
        @parametrize("x", [1, 2], ids=["one", "two"])
        def test_values(x):
            pass

        assert test_values[0].__name__ == "test_values[one]"
        assert test_values[1].__name__ == "test_values[two]"


class TestTestCase:
    """Tests for TestCase base class."""

    def test_create_test_case(self):
        class MyTest(TestCase):
            def test_example(self):
                pass

        tc = MyTest()
        assert hasattr(tc, "setup")
        assert hasattr(tc, "teardown")

    def test_setup_teardown_called(self):
        calls = []

        class MyTest(TestCase):
            def setup(self):
                calls.append("setup")

            def teardown(self):
                calls.append("teardown")

            def test_example(self):
                calls.append("test")

        tc = MyTest()
        tc.run_test("test_example")

        assert calls == ["setup", "test", "teardown"]

    def test_get_test_methods(self):
        class MyTest(TestCase):
            def test_one(self):
                pass

            def test_two(self):
                pass

            def helper(self):
                pass

        methods = MyTest.get_test_methods()
        assert "test_one" in methods
        assert "test_two" in methods
        assert "helper" not in methods


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_create_result(self):
        result = BenchmarkResult(
            name="test_bench",
            iterations=100,
            total_time=1.0,
            min_time=0.008,
            max_time=0.012,
            mean_time=0.01,
            median_time=0.01,
            std_dev=0.001,
            ops_per_second=100.0,
        )

        assert result.name == "test_bench"
        assert result.iterations == 100

    def test_str_representation(self):
        result = BenchmarkResult(
            name="test_bench",
            iterations=100,
            total_time=1.0,
            min_time=0.008,
            max_time=0.012,
            mean_time=0.01,
            median_time=0.01,
            std_dev=0.001,
            ops_per_second=100.0,
        )

        str_repr = str(result)
        assert "test_bench" in str_repr
        assert "ms" in str_repr

    def test_to_dict(self):
        result = BenchmarkResult(
            name="test_bench",
            iterations=100,
            total_time=1.0,
            min_time=0.008,
            max_time=0.012,
            mean_time=0.01,
            median_time=0.01,
            std_dev=0.001,
            ops_per_second=100.0,
        )

        data = result.to_dict()
        assert data["name"] == "test_bench"
        assert data["iterations"] == 100


class TestBenchmarkSuite:
    """Tests for BenchmarkSuite class."""

    def test_create_suite(self):
        suite = BenchmarkSuite("String Operations")
        assert suite.name == "String Operations"

    def test_add_benchmark(self):
        suite = BenchmarkSuite("Suite")

        @suite.add
        @bench(iterations=10)
        def bench_example():
            pass

        assert len(suite.benchmarks) == 1

    def test_run_benchmarks(self):
        suite = BenchmarkSuite("Suite")

        @suite.add
        @bench(iterations=10)
        def bench_one():
            pass

        @suite.add
        @bench(iterations=10)
        def bench_two():
            pass

        results = suite.run()
        assert len(results) == 2
        assert "bench_one" in results
        assert "bench_two" in results
