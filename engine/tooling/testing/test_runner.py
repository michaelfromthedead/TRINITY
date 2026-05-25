"""
Test runner with filtering and parallel execution support.

Provides comprehensive test execution infrastructure for the AI Game Engine,
including test discovery, filtering, parallel execution, and result collection.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import fnmatch
import inspect
import os
import re
import sys
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Pattern,
    Set,
    Tuple,
    Type,
    Union,
)


class TestStatus(Enum):
    """Status of a test execution."""

    PENDING = auto()
    RUNNING = auto()
    PASSED = auto()
    FAILED = auto()
    ERROR = auto()
    SKIPPED = auto()
    EXPECTED_FAILURE = auto()
    UNEXPECTED_SUCCESS = auto()
    TIMEOUT = auto()


@dataclass
class TestResult:
    """Result of a single test execution."""

    name: str
    status: TestStatus
    duration: float = 0.0
    message: str = ""
    traceback: str = ""
    output: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Check if test passed."""
        return self.status in (TestStatus.PASSED, TestStatus.EXPECTED_FAILURE)

    @property
    def failed(self) -> bool:
        """Check if test failed."""
        return self.status in (TestStatus.FAILED, TestStatus.ERROR, TestStatus.TIMEOUT)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "status": self.status.name,
            "duration": self.duration,
            "message": self.message,
            "traceback": self.traceback,
            "output": self.output,
            "metadata": self.metadata,
        }


@dataclass
class TestFilter:
    """Filter for selecting tests to run."""

    patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    exclude_tags: Set[str] = field(default_factory=set)
    regex: Optional[Pattern] = None
    modules: List[str] = field(default_factory=list)

    def matches(self, test_name: str, test_tags: Set[str] = None) -> bool:
        """Check if a test matches this filter."""
        test_tags = test_tags or set()

        # Check exclusions first
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(test_name, pattern):
                return False

        if self.exclude_tags and test_tags & self.exclude_tags:
            return False

        # Check inclusions
        if self.patterns:
            if not any(fnmatch.fnmatch(test_name, p) for p in self.patterns):
                return False

        if self.tags:
            if not (test_tags & self.tags):
                return False

        if self.regex:
            if not self.regex.search(test_name):
                return False

        if self.modules:
            module = test_name.rsplit(".", 1)[0] if "." in test_name else ""
            if not any(module.startswith(m) for m in self.modules):
                return False

        return True

    @classmethod
    def from_string(cls, filter_str: str) -> "TestFilter":
        """Create filter from a string expression."""
        filters = TestFilter()

        for part in filter_str.split(","):
            part = part.strip()
            if not part:
                continue

            if part.startswith("-"):
                filters.exclude_patterns.append(part[1:])
            elif part.startswith("tag:"):
                filters.tags.add(part[4:])
            elif part.startswith("-tag:"):
                filters.exclude_tags.add(part[5:])
            elif part.startswith("re:"):
                filters.regex = re.compile(part[3:])
            elif part.startswith("module:"):
                filters.modules.append(part[7:])
            else:
                filters.patterns.append(part)

        return filters


@dataclass
class TestSuite:
    """Collection of tests to run."""

    name: str
    tests: List[Callable] = field(default_factory=list)
    setup: Optional[Callable] = None
    teardown: Optional[Callable] = None
    fixtures: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_test(self, test_func: Callable) -> None:
        """Add a test to the suite."""
        self.tests.append(test_func)

    def filter_tests(self, test_filter: TestFilter) -> "TestSuite":
        """Return a new suite with filtered tests."""
        filtered = TestSuite(
            name=self.name,
            setup=self.setup,
            teardown=self.teardown,
            fixtures=self.fixtures,
            metadata=self.metadata.copy(),
        )

        for test in self.tests:
            test_name = f"{self.name}.{test.__name__}"
            tags = getattr(test, "_test_tags", set())
            if test_filter.matches(test_name, tags):
                filtered.tests.append(test)

        return filtered


class TestRunner:
    """
    Main test runner for executing tests.

    Features:
    - Test discovery and filtering
    - Setup/teardown management
    - Result collection and reporting
    - Timeout support
    - Verbose output modes
    """

    def __init__(
        self,
        verbose: int = 0,
        fail_fast: bool = False,
        timeout: float = 60.0,
        capture_output: bool = True,
    ):
        self.verbose = verbose
        self.fail_fast = fail_fast
        self.timeout = timeout
        self.capture_output = capture_output
        self.results: List[TestResult] = []
        self._current_output: List[str] = []
        self._hooks: Dict[str, List[Callable]] = {
            "before_test": [],
            "after_test": [],
            "before_suite": [],
            "after_suite": [],
        }

    def add_hook(self, event: str, callback: Callable) -> None:
        """Add a hook for test events."""
        if event in self._hooks:
            self._hooks[event].append(callback)

    def _run_hooks(self, event: str, *args, **kwargs) -> None:
        """Run all hooks for an event."""
        for hook in self._hooks.get(event, []):
            try:
                hook(*args, **kwargs)
            except Exception as e:
                if self.verbose > 0:
                    print(f"Hook error ({event}): {e}")

    def run_test(self, test_func: Callable, suite: Optional[TestSuite] = None) -> TestResult:
        """Run a single test function."""
        test_name = test_func.__name__
        if suite:
            test_name = f"{suite.name}.{test_name}"

        self._run_hooks("before_test", test_func)

        result = TestResult(name=test_name, status=TestStatus.RUNNING)
        self._current_output.clear()

        # Check for skip decorators
        if getattr(test_func, "_skip", False):
            result.status = TestStatus.SKIPPED
            result.message = getattr(test_func, "_skip_reason", "")
            self._run_hooks("after_test", test_func, result)
            return result

        skip_condition = getattr(test_func, "_skip_if", None)
        if skip_condition and skip_condition():
            result.status = TestStatus.SKIPPED
            result.message = getattr(test_func, "_skip_if_reason", "Condition met")
            self._run_hooks("after_test", test_func, result)
            return result

        expected_failure = getattr(test_func, "_expected_failure", False)
        test_timeout = getattr(test_func, "_timeout", self.timeout)

        start_time = time.perf_counter()

        try:
            # Run with timeout
            if test_timeout:
                self._run_with_timeout(test_func, test_timeout)
            else:
                test_func()

            result.duration = time.perf_counter() - start_time

            if expected_failure:
                result.status = TestStatus.UNEXPECTED_SUCCESS
                result.message = "Test was expected to fail but passed"
            else:
                result.status = TestStatus.PASSED

        except TimeoutError:
            result.duration = time.perf_counter() - start_time
            result.status = TestStatus.TIMEOUT
            result.message = f"Test timed out after {test_timeout}s"

        except AssertionError as e:
            result.duration = time.perf_counter() - start_time
            if expected_failure:
                result.status = TestStatus.EXPECTED_FAILURE
            else:
                result.status = TestStatus.FAILED
            result.message = str(e)
            result.traceback = traceback.format_exc()

        except Exception as e:
            result.duration = time.perf_counter() - start_time
            if expected_failure:
                result.status = TestStatus.EXPECTED_FAILURE
            else:
                result.status = TestStatus.ERROR
            result.message = f"{type(e).__name__}: {e}"
            result.traceback = traceback.format_exc()

        result.output = "\n".join(self._current_output)
        self._run_hooks("after_test", test_func, result)

        return result

    def _run_with_timeout(self, func: Callable, timeout: float) -> Any:
        """Run a function with a timeout."""
        import threading

        result_container = {"result": None, "exception": None}

        def target():
            try:
                result_container["result"] = func()
            except Exception as e:
                result_container["exception"] = e

        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            raise TimeoutError(f"Function timed out after {timeout}s")

        if result_container["exception"]:
            raise result_container["exception"]

        return result_container["result"]

    def run_suite(self, suite: TestSuite) -> List[TestResult]:
        """Run all tests in a suite."""
        self._run_hooks("before_suite", suite)
        suite_results = []

        # Run suite setup
        if suite.setup:
            try:
                suite.setup()
            except Exception as e:
                error_result = TestResult(
                    name=f"{suite.name}.setup",
                    status=TestStatus.ERROR,
                    message=f"Suite setup failed: {e}",
                    traceback=traceback.format_exc(),
                )
                suite_results.append(error_result)
                self._run_hooks("after_suite", suite, suite_results)
                return suite_results

        # Run each test
        for test_func in suite.tests:
            result = self.run_test(test_func, suite)
            suite_results.append(result)
            self.results.append(result)

            if self.verbose > 0:
                self._print_result(result)

            if self.fail_fast and result.failed:
                break

        # Run suite teardown
        if suite.teardown:
            try:
                suite.teardown()
            except Exception as e:
                error_result = TestResult(
                    name=f"{suite.name}.teardown",
                    status=TestStatus.ERROR,
                    message=f"Suite teardown failed: {e}",
                )
                suite_results.append(error_result)

        self._run_hooks("after_suite", suite, suite_results)
        return suite_results

    def _print_result(self, result: TestResult) -> None:
        """Print a test result to console."""
        status_chars = {
            TestStatus.PASSED: ".",
            TestStatus.FAILED: "F",
            TestStatus.ERROR: "E",
            TestStatus.SKIPPED: "S",
            TestStatus.TIMEOUT: "T",
            TestStatus.EXPECTED_FAILURE: "x",
            TestStatus.UNEXPECTED_SUCCESS: "u",
        }

        if self.verbose == 1:
            print(status_chars.get(result.status, "?"), end="", flush=True)
        elif self.verbose >= 2:
            status = result.status.name
            print(f"{result.name}: {status} ({result.duration:.3f}s)")
            if result.message and self.verbose >= 3:
                print(f"  {result.message}")

    def run(
        self,
        suites: List[TestSuite],
        test_filter: Optional[TestFilter] = None,
    ) -> List[TestResult]:
        """Run multiple test suites."""
        self.results.clear()

        for suite in suites:
            if test_filter:
                suite = suite.filter_tests(test_filter)

            if suite.tests:
                self.run_suite(suite)

        if self.verbose == 1:
            print()  # Newline after dots

        return self.results

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of test results."""
        summary = {
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r.status == TestStatus.PASSED),
            "failed": sum(1 for r in self.results if r.status == TestStatus.FAILED),
            "errors": sum(1 for r in self.results if r.status == TestStatus.ERROR),
            "skipped": sum(1 for r in self.results if r.status == TestStatus.SKIPPED),
            "timeout": sum(1 for r in self.results if r.status == TestStatus.TIMEOUT),
            "duration": sum(r.duration for r in self.results),
        }
        summary["success"] = summary["failed"] == 0 and summary["errors"] == 0
        return summary


class ParallelTestRunner(TestRunner):
    """
    Test runner with parallel execution support.

    Executes tests concurrently using a thread or process pool.
    """

    def __init__(
        self,
        workers: int = 4,
        use_processes: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.workers = workers
        self.use_processes = use_processes

    def run_suite(self, suite: TestSuite) -> List[TestResult]:
        """Run all tests in a suite in parallel."""
        self._run_hooks("before_suite", suite)

        # Suite setup must be sequential
        if suite.setup:
            try:
                suite.setup()
            except Exception as e:
                error_result = TestResult(
                    name=f"{suite.name}.setup",
                    status=TestStatus.ERROR,
                    message=f"Suite setup failed: {e}",
                    traceback=traceback.format_exc(),
                )
                self._run_hooks("after_suite", suite, [error_result])
                return [error_result]

        # Run tests in parallel
        executor_class = (
            concurrent.futures.ProcessPoolExecutor
            if self.use_processes
            else concurrent.futures.ThreadPoolExecutor
        )

        suite_results = []

        with executor_class(max_workers=self.workers) as executor:
            # Submit all tests
            futures = {
                executor.submit(self._run_test_isolated, test_func, suite.name): test_func
                for test_func in suite.tests
            }

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                suite_results.append(result)
                self.results.append(result)

                if self.verbose > 0:
                    self._print_result(result)

                if self.fail_fast and result.failed:
                    # Cancel remaining tests
                    for f in futures:
                        f.cancel()
                    break

        # Suite teardown
        if suite.teardown:
            try:
                suite.teardown()
            except Exception as e:
                error_result = TestResult(
                    name=f"{suite.name}.teardown",
                    status=TestStatus.ERROR,
                    message=f"Suite teardown failed: {e}",
                )
                suite_results.append(error_result)

        self._run_hooks("after_suite", suite, suite_results)
        return suite_results

    def _run_test_isolated(self, test_func: Callable, suite_name: str) -> TestResult:
        """Run a test in isolation (for parallel execution)."""
        test_name = f"{suite_name}.{test_func.__name__}"
        result = TestResult(name=test_name, status=TestStatus.RUNNING)

        # Check skip conditions
        if getattr(test_func, "_skip", False):
            result.status = TestStatus.SKIPPED
            result.message = getattr(test_func, "_skip_reason", "")
            return result

        skip_condition = getattr(test_func, "_skip_if", None)
        if skip_condition and skip_condition():
            result.status = TestStatus.SKIPPED
            result.message = getattr(test_func, "_skip_if_reason", "Condition met")
            return result

        expected_failure = getattr(test_func, "_expected_failure", False)
        test_timeout = getattr(test_func, "_timeout", self.timeout)

        start_time = time.perf_counter()

        try:
            if test_timeout:
                self._run_with_timeout(test_func, test_timeout)
            else:
                test_func()

            result.duration = time.perf_counter() - start_time

            if expected_failure:
                result.status = TestStatus.UNEXPECTED_SUCCESS
            else:
                result.status = TestStatus.PASSED

        except TimeoutError:
            result.duration = time.perf_counter() - start_time
            result.status = TestStatus.TIMEOUT
            result.message = f"Test timed out after {test_timeout}s"

        except AssertionError as e:
            result.duration = time.perf_counter() - start_time
            result.status = TestStatus.EXPECTED_FAILURE if expected_failure else TestStatus.FAILED
            result.message = str(e)
            result.traceback = traceback.format_exc()

        except Exception as e:
            result.duration = time.perf_counter() - start_time
            result.status = TestStatus.EXPECTED_FAILURE if expected_failure else TestStatus.ERROR
            result.message = f"{type(e).__name__}: {e}"
            result.traceback = traceback.format_exc()

        return result


def discover_tests(
    path: Union[str, Path],
    pattern: str = "test_*.py",
    recursive: bool = True,
) -> List[TestSuite]:
    """
    Discover tests from a directory.

    Args:
        path: Directory to search for tests
        pattern: Glob pattern for test files
        recursive: Whether to search recursively

    Returns:
        List of discovered test suites
    """
    path = Path(path)
    suites = []

    if recursive:
        test_files = list(path.rglob(pattern))
    else:
        test_files = list(path.glob(pattern))

    for test_file in test_files:
        suite = _load_test_file(test_file)
        if suite and suite.tests:
            suites.append(suite)

    return suites


def _load_test_file(file_path: Path) -> Optional[TestSuite]:
    """Load tests from a Python file."""
    import importlib.util

    try:
        spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            suite = TestSuite(name=file_path.stem)

            # Find test functions
            for name in dir(module):
                obj = getattr(module, name)
                if callable(obj) and (
                    name.startswith("test_") or getattr(obj, "_is_test", False)
                ):
                    suite.add_test(obj)

            # Find test classes
            for name in dir(module):
                obj = getattr(module, name)
                if isinstance(obj, type) and name.startswith("Test"):
                    class_suite = _load_test_class(obj, file_path.stem)
                    suite.tests.extend(class_suite.tests)

            return suite

    except Exception as e:
        print(f"Error loading {file_path}: {e}")

    return None


def _load_test_class(test_class: Type, module_name: str) -> TestSuite:
    """Load tests from a test class."""
    suite = TestSuite(name=f"{module_name}.{test_class.__name__}")

    # Get setup/teardown methods
    if hasattr(test_class, "setUp"):
        suite.setup = test_class.setUp
    if hasattr(test_class, "tearDown"):
        suite.teardown = test_class.tearDown

    # Find test methods
    instance = test_class()
    for name in dir(test_class):
        if name.startswith("test_"):
            method = getattr(instance, name)
            if callable(method):
                suite.add_test(method)

    return suite


def run_tests(
    suites: List[TestSuite],
    parallel: bool = False,
    workers: int = 4,
    verbose: int = 1,
    fail_fast: bool = False,
    test_filter: Optional[TestFilter] = None,
) -> Tuple[List[TestResult], bool]:
    """
    Convenience function to run tests.

    Args:
        suites: Test suites to run
        parallel: Whether to run tests in parallel
        workers: Number of parallel workers
        verbose: Verbosity level (0-3)
        fail_fast: Stop on first failure
        test_filter: Filter for selecting tests

    Returns:
        Tuple of (results, success)
    """
    if parallel:
        runner = ParallelTestRunner(workers=workers, verbose=verbose, fail_fast=fail_fast)
    else:
        runner = TestRunner(verbose=verbose, fail_fast=fail_fast)

    results = runner.run(suites, test_filter)
    summary = runner.get_summary()

    return results, summary["success"]
