"""
Testing framework for the game engine.

Provides a comprehensive testing infrastructure including:
- Test assertions with detailed failure messages
- Test fixtures for setup/teardown
- Test runner with discovery and execution
- Automation bots for automated gameplay testing
- Benchmark suite for performance testing

Usage:
    from engine.debug.testing import (
        TestRunner,
        TestSuite,
        expect_eq,
        expect_true,
        TestFixture,
        Benchmark,
        bench,
    )

    class MathTests(TestSuite):
        def test_addition(self):
            expect_eq(1 + 1, 2)

        def test_float_comparison(self):
            expect_near(1.0 / 3.0, 0.333, epsilon=0.001)

    runner = TestRunner()
    runner.add_suite(MathTests)
    results = runner.run()

Example with fixtures:
    class DatabaseFixture(TestFixture):
        def setUp(self, context):
            self.db = TestDatabase()

        def tearDown(self, context):
            self.db.close()

    class DatabaseTests(TestSuite):
        fixtures = [DatabaseFixture()]

        def test_insert(self):
            expect_true(self.fixtures[0].db.insert("key", "value"))

Example with benchmarks:
    @bench(iterations=10000, warmup=1000)
    def bench_vector_add():
        Vec3(1, 2, 3) + Vec3(4, 5, 6)

    results = run_benchmarks()
"""

from .assertions import (
    TestFailure,
    expect_eq,
    expect_ne,
    expect_true,
    expect_false,
    expect_near,
    expect_throws,
    expect_contains,
    expect_not_contains,
    expect_is,
    expect_is_not,
    expect_none,
    expect_not_none,
    expect_greater,
    expect_greater_eq,
    expect_less,
    expect_less_eq,
    expect_in_range,
    expect_type,
    expect_instance,
)

from .fixtures import (
    TestFixture,
    SharedFixture,
    CompositeFixture,
    FixtureContext,
    fixture,
    shared_fixture,
)

from .runner import (
    TestResult,
    TestSuite,
    TestRunner,
    ExecutionMode,
    skip,
    skip_if,
    expected_failure,
)

from .automation import (
    AutomationBot,
    TestScenario,
    ScenarioStep,
    StepResult,
    ScenarioResult,
    Action,
    InputSimulator,
    TimeoutError,
    ConditionNotMetError,
)

from .benchmarks import (
    BenchmarkResult,
    Benchmark,
    BenchmarkSuite,
    BenchmarkComparison,
    bench,
    run_benchmarks,
)


__all__ = [
    # Assertions
    "TestFailure",
    "expect_eq",
    "expect_ne",
    "expect_true",
    "expect_false",
    "expect_near",
    "expect_throws",
    "expect_contains",
    "expect_not_contains",
    "expect_is",
    "expect_is_not",
    "expect_none",
    "expect_not_none",
    "expect_greater",
    "expect_greater_eq",
    "expect_less",
    "expect_less_eq",
    "expect_in_range",
    "expect_type",
    "expect_instance",
    # Fixtures
    "TestFixture",
    "SharedFixture",
    "CompositeFixture",
    "FixtureContext",
    "fixture",
    "shared_fixture",
    # Runner
    "TestResult",
    "TestSuite",
    "TestRunner",
    "ExecutionMode",
    "skip",
    "skip_if",
    "expected_failure",
    # Automation
    "AutomationBot",
    "TestScenario",
    "ScenarioStep",
    "StepResult",
    "ScenarioResult",
    "Action",
    "InputSimulator",
    "TimeoutError",
    "ConditionNotMetError",
    # Benchmarks
    "BenchmarkResult",
    "Benchmark",
    "BenchmarkSuite",
    "BenchmarkComparison",
    "bench",
    "run_benchmarks",
]
