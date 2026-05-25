"""
Tests for test_runner.py - Test runner, filtering, and parallel execution.
"""

import time

import pytest

from engine.tooling.testing.test_runner import (
    ParallelTestRunner,
    TestFilter,
    TestResult,
    TestRunner,
    TestStatus,
    TestSuite,
    discover_tests,
    run_tests,
)


class TestTestStatus:
    """Tests for TestStatus enum."""

    def test_status_values_exist(self):
        assert TestStatus.PENDING
        assert TestStatus.RUNNING
        assert TestStatus.PASSED
        assert TestStatus.FAILED
        assert TestStatus.ERROR
        assert TestStatus.SKIPPED
        assert TestStatus.TIMEOUT


class TestTestResult:
    """Tests for TestResult dataclass."""

    def test_create_result(self):
        result = TestResult(name="test_example", status=TestStatus.PASSED)
        assert result.name == "test_example"
        assert result.status == TestStatus.PASSED

    def test_passed_property(self):
        passed = TestResult(name="t1", status=TestStatus.PASSED)
        failed = TestResult(name="t2", status=TestStatus.FAILED)
        expected_fail = TestResult(name="t3", status=TestStatus.EXPECTED_FAILURE)

        assert passed.passed is True
        assert failed.passed is False
        assert expected_fail.passed is True

    def test_failed_property(self):
        passed = TestResult(name="t1", status=TestStatus.PASSED)
        failed = TestResult(name="t2", status=TestStatus.FAILED)
        error = TestResult(name="t3", status=TestStatus.ERROR)
        timeout = TestResult(name="t4", status=TestStatus.TIMEOUT)

        assert passed.failed is False
        assert failed.failed is True
        assert error.failed is True
        assert timeout.failed is True

    def test_to_dict(self):
        result = TestResult(
            name="test_example",
            status=TestStatus.PASSED,
            duration=1.5,
            message="Test passed",
        )
        data = result.to_dict()

        assert data["name"] == "test_example"
        assert data["status"] == "PASSED"
        assert data["duration"] == 1.5
        assert data["message"] == "Test passed"


class TestTestFilter:
    """Tests for TestFilter class."""

    def test_empty_filter_matches_all(self):
        f = TestFilter()
        assert f.matches("any_test") is True
        assert f.matches("another_test") is True

    def test_pattern_filter(self):
        f = TestFilter(patterns=["test_foo*"])
        assert f.matches("test_foo_bar") is True
        assert f.matches("test_baz") is False

    def test_exclude_pattern(self):
        f = TestFilter(exclude_patterns=["*slow*"])
        assert f.matches("test_fast") is True
        assert f.matches("test_slow_operation") is False

    def test_tag_filter(self):
        f = TestFilter(tags={"unit"})
        assert f.matches("test_example", {"unit", "fast"}) is True
        assert f.matches("test_example", {"integration"}) is False

    def test_exclude_tag_filter(self):
        f = TestFilter(exclude_tags={"slow"})
        assert f.matches("test_example", {"fast"}) is True
        assert f.matches("test_example", {"slow"}) is False

    def test_from_string(self):
        f = TestFilter.from_string("test_foo*, -test_bar*, tag:unit")
        assert "test_foo*" in f.patterns
        assert "test_bar*" in f.exclude_patterns
        assert "unit" in f.tags

    def test_regex_filter(self):
        import re
        f = TestFilter(regex=re.compile(r"test_\d+"))
        assert f.matches("test_123") is True
        assert f.matches("test_abc") is False

    def test_module_filter(self):
        f = TestFilter(modules=["engine.tooling"])
        assert f.matches("engine.tooling.test_runner.test_foo") is True
        assert f.matches("engine.core.test_foo") is False


class TestTestSuite:
    """Tests for TestSuite class."""

    def test_create_suite(self):
        suite = TestSuite(name="MySuite")
        assert suite.name == "MySuite"
        assert suite.tests == []

    def test_add_test(self):
        suite = TestSuite(name="MySuite")

        def test_example():
            pass

        suite.add_test(test_example)
        assert len(suite.tests) == 1
        assert suite.tests[0] == test_example

    def test_filter_tests(self):
        suite = TestSuite(name="MySuite")

        def test_foo():
            pass

        def test_bar():
            pass

        test_foo._test_tags = {"unit"}
        test_bar._test_tags = {"integration"}

        suite.add_test(test_foo)
        suite.add_test(test_bar)

        f = TestFilter(tags={"unit"})
        filtered = suite.filter_tests(f)

        assert len(filtered.tests) == 1
        assert filtered.tests[0] == test_foo


class TestTestRunner:
    """Tests for TestRunner class."""

    def test_create_runner(self):
        runner = TestRunner(verbose=1, fail_fast=True)
        assert runner.verbose == 1
        assert runner.fail_fast is True

    def test_run_passing_test(self):
        runner = TestRunner()

        def test_pass():
            assert True

        result = runner.run_test(test_pass)
        assert result.status == TestStatus.PASSED

    def test_run_failing_test(self):
        runner = TestRunner()

        def test_fail():
            assert False, "Expected failure"

        result = runner.run_test(test_fail)
        assert result.status == TestStatus.FAILED
        assert "Expected failure" in result.message

    def test_run_error_test(self):
        runner = TestRunner()

        def test_error():
            raise RuntimeError("Something went wrong")

        result = runner.run_test(test_error)
        assert result.status == TestStatus.ERROR
        assert "RuntimeError" in result.message

    def test_run_skipped_test(self):
        runner = TestRunner()

        def test_skip():
            pass

        test_skip._skip = True
        test_skip._skip_reason = "Not implemented"

        result = runner.run_test(test_skip)
        assert result.status == TestStatus.SKIPPED
        assert "Not implemented" in result.message

    def test_run_with_timeout(self):
        runner = TestRunner(timeout=0.1)

        def test_slow():
            time.sleep(1.0)

        result = runner.run_test(test_slow)
        assert result.status == TestStatus.TIMEOUT

    def test_run_suite(self):
        runner = TestRunner()
        suite = TestSuite(name="TestSuite")

        def test_one():
            pass

        def test_two():
            pass

        suite.add_test(test_one)
        suite.add_test(test_two)

        results = runner.run_suite(suite)
        assert len(results) == 2
        assert all(r.status == TestStatus.PASSED for r in results)

    def test_fail_fast(self):
        runner = TestRunner(fail_fast=True)
        suite = TestSuite(name="TestSuite")

        def test_pass():
            pass

        def test_fail():
            assert False

        def test_after():
            pass

        suite.add_test(test_pass)
        suite.add_test(test_fail)
        suite.add_test(test_after)

        results = runner.run_suite(suite)
        assert len(results) == 2  # Third test should not run

    def test_get_summary(self):
        runner = TestRunner()
        suite = TestSuite(name="TestSuite")

        def test_pass():
            pass

        def test_fail():
            assert False

        suite.add_test(test_pass)
        suite.add_test(test_fail)

        runner.run_suite(suite)
        summary = runner.get_summary()

        assert summary["total"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert summary["success"] is False

    def test_add_hook(self):
        runner = TestRunner()
        calls = []

        def on_test_complete(func, result):
            calls.append((func.__name__, result.status))

        runner.add_hook("after_test", on_test_complete)

        def test_example():
            pass

        runner.run_test(test_example)

        assert len(calls) == 1
        assert calls[0][0] == "test_example"
        assert calls[0][1] == TestStatus.PASSED


class TestParallelTestRunner:
    """Tests for ParallelTestRunner class."""

    def test_create_parallel_runner(self):
        runner = ParallelTestRunner(workers=4)
        assert runner.workers == 4

    def test_parallel_execution(self):
        runner = ParallelTestRunner(workers=2)
        suite = TestSuite(name="ParallelSuite")

        results = []

        def test_one():
            results.append(1)

        def test_two():
            results.append(2)

        suite.add_test(test_one)
        suite.add_test(test_two)

        runner.run_suite(suite)

        assert len(runner.results) == 2


class TestRunTestsFunction:
    """Tests for run_tests convenience function."""

    def test_run_tests_basic(self):
        suite = TestSuite(name="Suite")

        def test_example():
            pass

        suite.add_test(test_example)

        results, success = run_tests([suite])
        assert success is True
        assert len(results) == 1

    def test_run_tests_with_filter(self):
        suite = TestSuite(name="Suite")

        def test_include():
            pass

        def test_exclude():
            pass

        suite.add_test(test_include)
        suite.add_test(test_exclude)

        f = TestFilter(patterns=["*include*"])
        results, success = run_tests([suite], test_filter=f)

        assert len(results) == 1
