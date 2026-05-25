"""
Testing subsystem for the AI Game Engine.

Provides comprehensive testing infrastructure including:
- Test runner with filtering and parallel execution
- Test framework with @test and @bench decorators
- Test fixtures for setup/teardown
- Custom assertions for game testing
- Mocking framework for game objects
- Test reporting in JUnit XML and HTML formats
"""

from .test_runner import (
    TestRunner,
    TestResult,
    TestStatus,
    TestFilter,
    ParallelTestRunner,
    TestSuite,
    run_tests,
    discover_tests,
)

from .test_framework import (
    test,
    bench,
    skip,
    skip_if,
    expected_failure,
    timeout,
    parametrize,
    TestCase,
    BenchmarkResult,
)

from .test_fixtures import (
    Fixture,
    FixtureScope,
    fixture,
    setup,
    teardown,
    before_all,
    after_all,
    FixtureManager,
    GameWorldFixture,
    EntityFixture,
    ResourceFixture,
)

from .test_assertions import (
    assert_vector_equal,
    assert_vector_near,
    assert_quaternion_equal,
    assert_transform_equal,
    assert_entity_has_component,
    assert_entity_count,
    assert_system_executed,
    assert_event_fired,
    assert_no_memory_leaks,
    assert_frame_time,
    GameAssertions,
)

from .test_mocking import (
    Mock,
    MockEntity,
    MockComponent,
    MockSystem,
    MockWorld,
    MockResource,
    patch,
    spy,
    stub,
    MockContext,
    create_mock,
)

from .test_reporting import (
    TestReporter,
    JUnitReporter,
    HTMLReporter,
    ConsoleReporter,
    JSONReporter,
    TestReport,
    generate_report,
)

__all__ = [
    # Test runner
    "TestRunner",
    "TestResult",
    "TestStatus",
    "TestFilter",
    "ParallelTestRunner",
    "TestSuite",
    "run_tests",
    "discover_tests",
    # Test framework
    "test",
    "bench",
    "skip",
    "skip_if",
    "expected_failure",
    "timeout",
    "parametrize",
    "TestCase",
    "BenchmarkResult",
    # Fixtures
    "Fixture",
    "FixtureScope",
    "fixture",
    "setup",
    "teardown",
    "before_all",
    "after_all",
    "FixtureManager",
    "GameWorldFixture",
    "EntityFixture",
    "ResourceFixture",
    # Assertions
    "assert_vector_equal",
    "assert_vector_near",
    "assert_quaternion_equal",
    "assert_transform_equal",
    "assert_entity_has_component",
    "assert_entity_count",
    "assert_system_executed",
    "assert_event_fired",
    "assert_no_memory_leaks",
    "assert_frame_time",
    "GameAssertions",
    # Mocking
    "Mock",
    "MockEntity",
    "MockComponent",
    "MockSystem",
    "MockWorld",
    "MockResource",
    "patch",
    "spy",
    "stub",
    "MockContext",
    "create_mock",
    # Reporting
    "TestReporter",
    "JUnitReporter",
    "HTMLReporter",
    "ConsoleReporter",
    "JSONReporter",
    "TestReport",
    "generate_report",
]
