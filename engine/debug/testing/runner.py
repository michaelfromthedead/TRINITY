"""
Test runner for the engine testing framework.

Provides test discovery, execution, and result collection with support
for multiple execution modes (editor, game, CLI, CI).

Usage:
    from engine.debug.testing import TestRunner, TestSuite, ExecutionMode

    class MyTests(TestSuite):
        def test_feature(self):
            expect_eq(1 + 1, 2)

    runner = TestRunner(mode=ExecutionMode.CLI)
    runner.discover("tests/")
    runner.run("*feature*")
"""

from __future__ import annotations

import fnmatch
import importlib
import importlib.util
import inspect
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterator,
    List,
    Optional,
    Pattern,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from .assertions import TestFailure
from .fixtures import FixtureContext, TestFixture


__all__ = [
    "TestResult",
    "TestSuite",
    "TestRunner",
    "ExecutionMode",
    "skip",
    "skip_if",
    "expected_failure",
]


T = TypeVar("T", bound="TestSuite")

# Constants for magic numbers
DEFAULT_TEST_TIMEOUT_MS = 30000  # 30 seconds default timeout


class ExecutionMode(Enum):
    """
    Test execution mode determining output format and behavior.

    Attributes:
        EDITOR: Running within the game editor with GUI output
        GAME: Running within the game runtime
        CLI: Running from command line with text output
        CI: Running in continuous integration with machine-readable output
    """

    EDITOR = auto()
    GAME = auto()
    CLI = auto()
    CI = auto()


@dataclass
class TestResult:
    """
    Result of a single test execution.

    Attributes:
        name: Full test name (suite.method)
        passed: True if test passed
        failed: True if test failed
        skipped: True if test was skipped
        duration_ms: Execution time in milliseconds
        errors: List of error messages if failed
        output: Captured stdout/stderr
        expected_failure: True if failure was expected
    """

    # Prevent pytest from collecting this class as a test
    __test__ = False

    name: str
    passed: bool = False
    failed: bool = False
    skipped: bool = False
    duration_ms: float = 0.0
    errors: List[str] = field(default_factory=list)
    output: str = ""
    expected_failure: bool = False
    skip_reason: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate result state - does not auto-set failed to allow for mutation."""
        # Note: We don't auto-set failed=True here because the test runner
        # will explicitly set the correct state after running the test.
        # This allows creating a result first and setting state later.
        pass

    @property
    def status(self) -> str:
        """Get the result status as a string."""
        if self.skipped:
            return "SKIP"
        if self.passed:
            return "PASS"
        if self.expected_failure and self.failed:
            return "XFAIL"
        if self.failed:
            return "FAIL"
        # No state set - treat as failed
        return "FAIL"

    def __repr__(self) -> str:
        return f"TestResult({self.name!r}, {self.status}, {self.duration_ms:.2f}ms)"


@dataclass
class SuiteResult:
    """
    Aggregated results for a test suite.

    Attributes:
        suite_name: Name of the test suite
        results: Individual test results
        setup_error: Error from setUpClass if any
        teardown_error: Error from tearDownClass if any
    """

    suite_name: str
    results: List[TestResult] = field(default_factory=list)
    setup_error: Optional[str] = None
    teardown_error: Optional[str] = None

    @property
    def passed(self) -> int:
        """Count of passed tests."""
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        """Count of failed tests."""
        return sum(1 for r in self.results if r.failed and not r.expected_failure)

    @property
    def skipped(self) -> int:
        """Count of skipped tests."""
        return sum(1 for r in self.results if r.skipped)

    @property
    def total(self) -> int:
        """Total number of tests."""
        return len(self.results)

    @property
    def duration_ms(self) -> float:
        """Total execution time in milliseconds."""
        return sum(r.duration_ms for r in self.results)


class _SkipTest(Exception):
    """Internal exception to signal test skip."""

    def __init__(self, reason: str = "") -> None:
        self.reason = reason
        super().__init__(reason)


def skip(reason: str = "") -> Callable[[Callable], Callable]:
    """
    Decorator to skip a test unconditionally.

    Args:
        reason: Explanation for why the test is skipped

    Returns:
        Decorator function

    Example:
        @skip("Not implemented yet")
        def test_future_feature(self):
            pass
    """
    def decorator(func: Callable) -> Callable:
        func._skip = True  # type: ignore
        func._skip_reason = reason  # type: ignore
        return func
    return decorator


def skip_if(
    condition: Union[bool, Callable[[], bool]],
    reason: str = "",
) -> Callable[[Callable], Callable]:
    """
    Decorator to skip a test if a condition is met.

    Args:
        condition: Boolean or callable returning boolean
        reason: Explanation for why the test is skipped

    Returns:
        Decorator function

    Example:
        @skip_if(sys.platform != "linux", "Linux only")
        def test_linux_feature(self):
            pass
    """
    def decorator(func: Callable) -> Callable:
        def check() -> bool:
            if callable(condition):
                return condition()
            return condition

        func._skip_if = check  # type: ignore
        func._skip_reason = reason  # type: ignore
        return func
    return decorator


def expected_failure(reason: str = "") -> Callable[[Callable], Callable]:
    """
    Decorator to mark a test as expected to fail.

    An expected failure that fails counts as a success.
    An expected failure that passes counts as an unexpected success.

    Args:
        reason: Explanation for why the test is expected to fail

    Returns:
        Decorator function

    Example:
        @expected_failure("Known bug #123")
        def test_buggy_feature(self):
            pass
    """
    def decorator(func: Callable) -> Callable:
        func._expected_failure = True  # type: ignore
        func._expected_failure_reason = reason  # type: ignore
        return func
    return decorator


class TestSuite:
    """
    Base class for test suites with automatic test method discovery.

    Test methods are discovered by looking for methods starting with "test_".
    Fixtures can be attached to provide setup/teardown functionality.

    Class Attributes:
        fixtures: List of TestFixture instances to apply

    Example:
        class MathTests(TestSuite):
            fixtures = [CalculatorFixture()]

            def test_addition(self):
                expect_eq(1 + 1, 2)

            def test_multiplication(self):
                expect_eq(2 * 3, 6)
    """

    # Prevent pytest from collecting this class as a test
    __test__ = False

    fixtures: ClassVar[List[TestFixture]] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register subclass and inherit fixtures."""
        super().__init_subclass__(**kwargs)

        # Inherit fixtures from parent classes
        inherited = []
        for base in cls.__mro__[1:]:
            if hasattr(base, "fixtures") and base is not TestSuite:
                inherited.extend(base.fixtures)

        # Combine with class-defined fixtures (avoiding duplicates)
        if not hasattr(cls, "fixtures"):
            cls.fixtures = []

        seen = set(id(f) for f in cls.fixtures)
        for f in inherited:
            if id(f) not in seen:
                cls.fixtures.append(f)
                seen.add(id(f))

    @classmethod
    def get_test_methods(cls) -> List[Tuple[str, Callable]]:
        """
        Discover test methods in this suite.

        Returns:
            List of (method_name, method) tuples
        """
        tests = []
        for name in dir(cls):
            if name.startswith("test_"):
                method = getattr(cls, name)
                if callable(method):
                    tests.append((name, method))

        # Sort by line number for consistent ordering, fallback to name
        def get_line(item: Tuple[str, Callable]) -> Tuple[int, str]:
            try:
                line = inspect.getsourcelines(item[1])[1]
                return (line, item[0])
            except (OSError, TypeError):
                # Fallback to alphabetical for dynamically defined methods
                return (0, item[0])

        tests.sort(key=get_line)
        return tests

    def setUp(self) -> None:
        """
        Called before each test method.

        Override this to provide per-test setup.
        """
        pass

    def tearDown(self) -> None:
        """
        Called after each test method.

        Override this to provide per-test cleanup.
        Always called even if test fails.
        """
        pass

    @classmethod
    def setUpClass(cls) -> None:
        """
        Called once before any tests in the suite.

        Override this to provide one-time setup.
        """
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Called once after all tests in the suite.

        Override this to provide one-time cleanup.
        Always called even if setUpClass fails.
        """
        pass


class TestRunner:
    """
    Test runner that discovers, executes, and reports on tests.

    Supports multiple execution modes with different output formats.

    Example:
        runner = TestRunner(mode=ExecutionMode.CLI, verbose=True)
        runner.discover("tests/")
        runner.add_suite(MyTests)
        results = runner.run(filter_pattern="*math*")

        print(f"Passed: {runner.passed_count}")
        print(f"Failed: {runner.failed_count}")
    """

    # Prevent pytest from collecting this class as a test
    __test__ = False

    def __init__(
        self,
        mode: ExecutionMode = ExecutionMode.CLI,
        verbose: bool = False,
        fail_fast: bool = False,
        capture_output: bool = True,
    ) -> None:
        """
        Initialize the test runner.

        Args:
            mode: Execution mode determining output format
            verbose: If True, show detailed output
            fail_fast: If True, stop on first failure
            capture_output: If True, capture stdout/stderr
        """
        self.mode = mode
        self.verbose = verbose
        self.fail_fast = fail_fast
        self.capture_output = capture_output

        self._suites: List[Type[TestSuite]] = []
        self._results: List[TestResult] = []
        self._suite_results: List[SuiteResult] = []
        self._discovered_modules: Set[str] = set()

    def discover(self, path: Union[str, Path]) -> int:
        """
        Discover test modules in a directory.

        Searches for Python files matching test_*.py or *_test.py
        and imports them, looking for TestSuite subclasses.

        Args:
            path: Directory to search or specific file path

        Returns:
            Number of test suites discovered
        """
        path = Path(path)
        initial_count = len(self._suites)

        if path.is_file():
            self._discover_file(path)
        elif path.is_dir():
            for pattern in ["test_*.py", "*_test.py"]:
                for file_path in path.rglob(pattern):
                    self._discover_file(file_path)

        return len(self._suites) - initial_count

    def _discover_file(self, file_path: Path) -> None:
        """Discover test suites in a single file."""
        module_name = file_path.stem

        # Skip if already discovered
        full_path = str(file_path.resolve())
        if full_path in self._discovered_modules:
            return
        self._discovered_modules.add(full_path)

        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                return

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Find TestSuite subclasses
            for name in dir(module):
                obj = getattr(module, name)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, TestSuite)
                    and obj is not TestSuite
                ):
                    self.add_suite(obj)

        except Exception as e:
            if self.verbose:
                print(f"Error importing {file_path}: {e}")

    def add_suite(self, suite_class: Type[TestSuite]) -> None:
        """
        Add a test suite to be run.

        Args:
            suite_class: TestSuite subclass
        """
        if suite_class not in self._suites:
            self._suites.append(suite_class)

    def run(
        self,
        filter_pattern: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[TestResult]:
        """
        Run all discovered tests.

        Args:
            filter_pattern: Glob pattern to filter test names
            tags: List of tags to filter by (not yet implemented)

        Returns:
            List of TestResult for all executed tests
        """
        self._results.clear()
        self._suite_results.clear()

        for suite_class in self._suites:
            suite_result = self.run_suite(suite_class, filter_pattern)
            self._suite_results.append(suite_result)
            self._results.extend(suite_result.results)

            if self.fail_fast and suite_result.failed > 0:
                break

        self._report_summary()
        return self._results.copy()

    def run_suite(
        self,
        suite_class: Type[TestSuite],
        filter_pattern: Optional[str] = None,
    ) -> SuiteResult:
        """
        Run a single test suite.

        Args:
            suite_class: TestSuite subclass to run
            filter_pattern: Glob pattern to filter test names

        Returns:
            SuiteResult with all test results
        """
        suite_name = suite_class.__name__
        suite_result = SuiteResult(suite_name=suite_name)

        # Get test methods
        test_methods = suite_class.get_test_methods()
        if filter_pattern:
            test_methods = [
                (name, method)
                for name, method in test_methods
                if fnmatch.fnmatch(name, filter_pattern)
                or fnmatch.fnmatch(f"{suite_name}.{name}", filter_pattern)
            ]

        if not test_methods:
            return suite_result

        # Create fixture context
        fixture_context = FixtureContext(suite_name=suite_name)

        # Class-level setup
        try:
            suite_class.setUpClass()
            for fixture in suite_class.fixtures:
                fixture._do_class_setup(fixture_context)
        except Exception as e:
            suite_result.setup_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            self._log(f"  [ERROR] {suite_name}.setUpClass failed: {e}")
            return suite_result

        try:
            # Run each test
            for test_name, test_method in test_methods:
                if self.fail_fast and suite_result.failed > 0:
                    break

                result = self._run_test(suite_class, test_name, test_method)
                suite_result.results.append(result)

        finally:
            # Class-level teardown
            try:
                for fixture in reversed(suite_class.fixtures):
                    fixture._do_class_teardown(fixture_context)
                suite_class.tearDownClass()
            except Exception as e:
                suite_result.teardown_error = f"{type(e).__name__}: {e}"

        return suite_result

    def _run_test(
        self,
        suite_class: Type[TestSuite],
        test_name: str,
        test_method: Callable,
    ) -> TestResult:
        """Run a single test method."""
        full_name = f"{suite_class.__name__}.{test_name}"
        result = TestResult(name=full_name)

        # Check for skip decorators
        if getattr(test_method, "_skip", False):
            result.skipped = True
            result.skip_reason = getattr(test_method, "_skip_reason", "")
            self._log_result(result)
            return result

        skip_if_check = getattr(test_method, "_skip_if", None)
        if skip_if_check and skip_if_check():
            result.skipped = True
            result.skip_reason = getattr(test_method, "_skip_reason", "")
            self._log_result(result)
            return result

        # Check for expected failure
        result.expected_failure = getattr(test_method, "_expected_failure", False)

        # Create instance and fixture context
        instance = suite_class()
        fixture_context = FixtureContext(
            test_name=test_name,
            suite_name=suite_class.__name__,
        )

        start_time = time.perf_counter_ns()

        try:
            # Per-test setup
            instance.setUp()
            for fixture in suite_class.fixtures:
                fixture._do_setup(fixture_context)

            # Run test
            test_method(instance)

            # Test passed (or expected failure unexpectedly passed)
            if result.expected_failure:
                result.failed = True
                result.errors.append("Test unexpectedly passed (expected failure)")
            else:
                result.passed = True

        except _SkipTest as e:
            result.skipped = True
            result.skip_reason = e.reason

        except TestFailure as e:
            if result.expected_failure:
                result.passed = True
            else:
                result.failed = True
                result.errors.append(str(e))

        except Exception as e:
            if result.expected_failure:
                result.passed = True
            else:
                result.failed = True
                result.errors.append(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

        finally:
            # Per-test teardown
            try:
                for fixture in reversed(suite_class.fixtures):
                    fixture._do_teardown(fixture_context)
                instance.tearDown()
            except Exception as e:
                if not result.failed:
                    result.failed = True
                    result.passed = False
                result.errors.append(f"tearDown error: {type(e).__name__}: {e}")

        result.duration_ms = (time.perf_counter_ns() - start_time) / 1_000_000
        self._log_result(result)
        return result

    def get_results(self) -> List[TestResult]:
        """
        Get all test results from the last run.

        Returns:
            List of TestResult
        """
        return self._results.copy()

    def get_suite_results(self) -> List[SuiteResult]:
        """
        Get suite-level results from the last run.

        Returns:
            List of SuiteResult
        """
        return self._suite_results.copy()

    @property
    def passed_count(self) -> int:
        """Count of passed tests."""
        return sum(1 for r in self._results if r.passed)

    @property
    def failed_count(self) -> int:
        """Count of failed tests."""
        return sum(1 for r in self._results if r.failed and not r.expected_failure)

    @property
    def skipped_count(self) -> int:
        """Count of skipped tests."""
        return sum(1 for r in self._results if r.skipped)

    @property
    def total_count(self) -> int:
        """Total number of tests run."""
        return len(self._results)

    @property
    def total_duration_ms(self) -> float:
        """Total execution time in milliseconds."""
        return sum(r.duration_ms for r in self._results)

    def _log(self, message: str) -> None:
        """Log a message based on execution mode."""
        if self.mode == ExecutionMode.CI:
            print(message)
        elif self.mode == ExecutionMode.CLI:
            print(message)
        elif self.mode == ExecutionMode.EDITOR:
            # Would integrate with editor logging
            print(message)
        elif self.mode == ExecutionMode.GAME:
            # Would integrate with game console
            print(message)

    def _log_result(self, result: TestResult) -> None:
        """Log a test result."""
        if not self.verbose and result.passed:
            return

        status_symbols = {
            "PASS": "[PASS]",
            "FAIL": "[FAIL]",
            "SKIP": "[SKIP]",
            "XFAIL": "[XFAIL]",
        }

        symbol = status_symbols.get(result.status, "[????]")
        msg = f"  {symbol} {result.name} ({result.duration_ms:.2f}ms)"

        if result.skip_reason:
            msg += f" - {result.skip_reason}"

        self._log(msg)

        if result.failed and result.errors:
            for error in result.errors:
                for line in error.split("\n"):
                    self._log(f"    {line}")

    def _report_summary(self) -> None:
        """Report test run summary."""
        if self.mode == ExecutionMode.CI:
            # Machine-readable format
            self._log("")
            self._log(f"TOTAL={self.total_count}")
            self._log(f"PASSED={self.passed_count}")
            self._log(f"FAILED={self.failed_count}")
            self._log(f"SKIPPED={self.skipped_count}")
            self._log(f"DURATION_MS={self.total_duration_ms:.2f}")
        else:
            # Human-readable format
            self._log("")
            self._log("=" * 60)
            self._log(
                f"Tests: {self.total_count} | "
                f"Passed: {self.passed_count} | "
                f"Failed: {self.failed_count} | "
                f"Skipped: {self.skipped_count}"
            )
            self._log(f"Duration: {self.total_duration_ms:.2f}ms")
            self._log("=" * 60)
