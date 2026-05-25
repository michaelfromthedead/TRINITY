"""
Tests for the testing framework runner module.

Verifies test discovery, execution, and result collection.
"""

import pytest
import sys
import tempfile
import os
sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.debug.testing.runner import (
    TestResult,
    TestSuite,
    TestRunner,
    ExecutionMode,
    skip,
    skip_if,
    expected_failure,
    SuiteResult,
)
from engine.debug.testing.assertions import expect_eq, TestFailure


class TestTestResult:
    """Tests for TestResult dataclass."""

    def test_default_values(self):
        """TestResult should have sensible defaults."""
        result = TestResult(name="test_foo")
        assert result.name == "test_foo"
        assert result.passed is False
        assert result.failed is False  # No state is set initially
        assert result.skipped is False
        assert result.duration_ms == 0.0
        assert result.errors == []
        # Status should still show FAIL for unset state
        assert result.status == "FAIL"

    def test_status_property(self):
        """TestResult status should reflect state."""
        passed = TestResult(name="test", passed=True)
        assert passed.status == "PASS"

        failed = TestResult(name="test", failed=True)
        assert failed.status == "FAIL"

        skipped = TestResult(name="test", skipped=True)
        assert skipped.status == "SKIP"

    def test_xfail_status(self):
        """Expected failure should show XFAIL status."""
        result = TestResult(name="test", failed=True, expected_failure=True)
        assert result.status == "XFAIL"

    def test_repr(self):
        """TestResult repr should be informative."""
        result = TestResult(name="test_foo", passed=True, duration_ms=1.5)
        assert "test_foo" in repr(result)
        assert "PASS" in repr(result)
        assert "1.5" in repr(result)


class TestTestSuite:
    """Tests for TestSuite base class."""

    def test_discover_test_methods(self):
        """TestSuite should discover methods starting with test_."""
        class MySuite(TestSuite):
            def test_one(self):
                pass

            def test_two(self):
                pass

            def helper(self):
                pass

        methods = MySuite.get_test_methods()
        method_names = [name for name, _ in methods]

        assert "test_one" in method_names
        assert "test_two" in method_names
        assert "helper" not in method_names

    def test_methods_sorted_by_line_number(self):
        """Test methods should be sorted by source line number."""
        class OrderedSuite(TestSuite):
            def test_first(self):
                pass

            def test_second(self):
                pass

            def test_third(self):
                pass

        methods = OrderedSuite.get_test_methods()
        names = [name for name, _ in methods]

        assert names == ["test_first", "test_second", "test_third"]

    def test_setup_teardown_methods(self):
        """TestSuite should have setUp and tearDown methods."""
        class SetupSuite(TestSuite):
            setup_called = False
            teardown_called = False

            def setUp(self):
                SetupSuite.setup_called = True

            def tearDown(self):
                SetupSuite.teardown_called = True

            def test_something(self):
                pass

        runner = TestRunner(verbose=False)
        runner.add_suite(SetupSuite)
        runner.run()

        assert SetupSuite.setup_called
        assert SetupSuite.teardown_called

    def test_class_setup_teardown(self):
        """TestSuite should have setUpClass and tearDownClass methods."""
        class ClassSetupSuite(TestSuite):
            class_setup_called = False
            class_teardown_called = False

            @classmethod
            def setUpClass(cls):
                ClassSetupSuite.class_setup_called = True

            @classmethod
            def tearDownClass(cls):
                ClassSetupSuite.class_teardown_called = True

            def test_one(self):
                pass

            def test_two(self):
                pass

        runner = TestRunner(verbose=False)
        runner.add_suite(ClassSetupSuite)
        runner.run()

        assert ClassSetupSuite.class_setup_called
        assert ClassSetupSuite.class_teardown_called


class TestSkipDecorators:
    """Tests for skip decorators."""

    def test_skip_decorator(self):
        """@skip should skip the test."""
        class SkippedSuite(TestSuite):
            @skip("Not implemented")
            def test_skipped(self):
                raise AssertionError("Should not run")

        runner = TestRunner(verbose=False)
        runner.add_suite(SkippedSuite)
        results = runner.run()

        assert len(results) == 1
        assert results[0].skipped
        assert results[0].skip_reason == "Not implemented"

    def test_skip_if_true(self):
        """@skip_if should skip when condition is True."""
        class ConditionalSuite(TestSuite):
            @skip_if(True, "Condition met")
            def test_skipped(self):
                raise AssertionError("Should not run")

        runner = TestRunner(verbose=False)
        runner.add_suite(ConditionalSuite)
        results = runner.run()

        assert results[0].skipped

    def test_skip_if_false(self):
        """@skip_if should not skip when condition is False."""
        class ConditionalSuite(TestSuite):
            @skip_if(False, "Condition not met")
            def test_not_skipped(self):
                pass

        runner = TestRunner(verbose=False)
        runner.add_suite(ConditionalSuite)
        results = runner.run()

        assert results[0].passed
        assert not results[0].skipped

    def test_skip_if_callable(self):
        """@skip_if should evaluate callable condition."""
        should_skip = True

        class CallableSuite(TestSuite):
            @skip_if(lambda: should_skip, "Lambda returned True")
            def test_maybe_skipped(self):
                raise AssertionError("Should not run")

        runner = TestRunner(verbose=False)
        runner.add_suite(CallableSuite)
        results = runner.run()

        assert results[0].skipped


class TestExpectedFailure:
    """Tests for expected_failure decorator."""

    def test_expected_failure_that_fails(self):
        """@expected_failure test that fails should pass."""
        class XFailSuite(TestSuite):
            @expected_failure("Known bug")
            def test_fails(self):
                raise AssertionError("Expected to fail")

        runner = TestRunner(verbose=False)
        runner.add_suite(XFailSuite)
        results = runner.run()

        assert results[0].passed
        assert results[0].expected_failure

    def test_expected_failure_that_passes(self):
        """@expected_failure test that passes should fail."""
        class XPassSuite(TestSuite):
            @expected_failure("Known bug")
            def test_unexpectedly_passes(self):
                pass  # Oops, it works now

        runner = TestRunner(verbose=False)
        runner.add_suite(XPassSuite)
        results = runner.run()

        assert results[0].failed
        assert results[0].expected_failure


class TestTestRunner:
    """Tests for TestRunner class."""

    def test_add_suite(self):
        """TestRunner should add and track suites."""
        class Suite1(TestSuite):
            def test_one(self):
                pass

        class Suite2(TestSuite):
            def test_two(self):
                pass

        runner = TestRunner(verbose=False)
        runner.add_suite(Suite1)
        runner.add_suite(Suite2)

        results = runner.run()
        assert len(results) == 2

    def test_run_with_filter(self):
        """TestRunner should filter tests by pattern."""
        class FilterSuite(TestSuite):
            def test_math_add(self):
                pass

            def test_math_subtract(self):
                pass

            def test_string_format(self):
                pass

        runner = TestRunner(verbose=False)
        runner.add_suite(FilterSuite)

        results = runner.run(filter_pattern="*math*")
        assert len(results) == 2

        results = runner.run(filter_pattern="*string*")
        assert len(results) == 1

    def test_fail_fast(self):
        """TestRunner with fail_fast should stop on first failure."""
        class FailFastSuite(TestSuite):
            def test_first_passes(self):
                pass

            def test_second_fails(self):
                raise AssertionError("Fail")

            def test_third_skipped(self):
                pass

        runner = TestRunner(verbose=False, fail_fast=True)
        runner.add_suite(FailFastSuite)
        results = runner.run()

        # Should have 2 results: first pass, second fail (third not run)
        assert len(results) == 2
        assert results[0].passed
        assert results[1].failed

    def test_result_counts(self):
        """TestRunner should track result counts."""
        class MixedSuite(TestSuite):
            def test_passes(self):
                pass

            def test_fails(self):
                raise AssertionError("Fail")

            @skip()
            def test_skipped(self):
                pass

        runner = TestRunner(verbose=False)
        runner.add_suite(MixedSuite)
        runner.run()

        assert runner.passed_count == 1
        assert runner.failed_count == 1
        assert runner.skipped_count == 1
        assert runner.total_count == 3

    def test_duration_tracking(self):
        """TestRunner should track execution duration."""
        import time

        class SlowSuite(TestSuite):
            def test_slow(self):
                time.sleep(0.01)  # 10ms

        runner = TestRunner(verbose=False)
        runner.add_suite(SlowSuite)
        results = runner.run()

        assert results[0].duration_ms >= 10

    def test_execution_modes(self):
        """TestRunner should support different execution modes."""
        class ModeSuite(TestSuite):
            def test_simple(self):
                pass

        for mode in ExecutionMode:
            runner = TestRunner(mode=mode, verbose=False)
            runner.add_suite(ModeSuite)
            results = runner.run()
            assert results[0].passed

    def test_suite_result(self):
        """TestRunner should provide suite-level results."""
        class Suite1(TestSuite):
            def test_one(self):
                pass

            def test_two(self):
                raise AssertionError()

        runner = TestRunner(verbose=False)
        runner.add_suite(Suite1)
        runner.run()

        suite_results = runner.get_suite_results()
        assert len(suite_results) == 1
        assert suite_results[0].suite_name == "Suite1"
        assert suite_results[0].passed == 1
        assert suite_results[0].failed == 1

    def test_teardown_always_called(self):
        """tearDown should be called even when test fails."""
        class TeardownSuite(TestSuite):
            teardown_count = 0

            def tearDown(self):
                TeardownSuite.teardown_count += 1

            def test_passes(self):
                pass

            def test_fails(self):
                raise AssertionError()

        runner = TestRunner(verbose=False)
        runner.add_suite(TeardownSuite)
        runner.run()

        assert TeardownSuite.teardown_count == 2


class TestTestDiscovery:
    """Tests for test discovery functionality."""

    def test_discover_from_file(self):
        """TestRunner should discover tests from a file."""
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='_test.py',
            delete=False,
        ) as f:
            f.write("""
import sys
sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')
from engine.debug.testing import TestSuite

class DiscoveredSuite(TestSuite):
    def test_discovered(self):
        pass
""")
            temp_path = f.name

        try:
            runner = TestRunner(verbose=False)
            count = runner.discover(temp_path)

            assert count >= 1
            results = runner.run()
            assert any(r.name == "DiscoveredSuite.test_discovered" for r in results)
        finally:
            os.unlink(temp_path)

    def test_discover_from_directory(self):
        """TestRunner should discover tests from a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_file = os.path.join(temp_dir, "test_example.py")
            with open(test_file, 'w') as f:
                f.write("""
import sys
sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')
from engine.debug.testing import TestSuite

class ExampleSuite(TestSuite):
    def test_example(self):
        pass
""")

            runner = TestRunner(verbose=False)
            count = runner.discover(temp_dir)

            assert count >= 1


class TestErrorHandling:
    """Tests for error handling in test execution."""

    def test_setup_error_captured(self):
        """Errors in setUp should be captured."""
        class SetupErrorSuite(TestSuite):
            def setUp(self):
                raise RuntimeError("Setup failed")

            def test_something(self):
                pass

        runner = TestRunner(verbose=False)
        runner.add_suite(SetupErrorSuite)
        results = runner.run()

        assert results[0].failed
        assert "Setup failed" in str(results[0].errors)

    def test_teardown_error_captured(self):
        """Errors in tearDown should be captured."""
        class TeardownErrorSuite(TestSuite):
            def tearDown(self):
                raise RuntimeError("Teardown failed")

            def test_something(self):
                pass

        runner = TestRunner(verbose=False)
        runner.add_suite(TeardownErrorSuite)
        results = runner.run()

        assert results[0].failed
        assert "tearDown error" in str(results[0].errors)

    def test_class_setup_error_captured(self):
        """Errors in setUpClass should be captured."""
        class ClassSetupErrorSuite(TestSuite):
            @classmethod
            def setUpClass(cls):
                raise RuntimeError("Class setup failed")

            def test_something(self):
                pass

        runner = TestRunner(verbose=False)
        runner.add_suite(ClassSetupErrorSuite)
        runner.run()

        suite_results = runner.get_suite_results()
        assert suite_results[0].setup_error is not None
        assert "Class setup failed" in suite_results[0].setup_error


class TestTestFailureIntegration:
    """Tests for integration with TestFailure assertions."""

    def test_test_failure_reported(self):
        """TestFailure from assertions should be properly reported."""
        class AssertionSuite(TestSuite):
            def test_assertion_fails(self):
                expect_eq(1, 2, "One should equal two")

        runner = TestRunner(verbose=False)
        runner.add_suite(AssertionSuite)
        results = runner.run()

        assert results[0].failed
        assert "One should equal two" in str(results[0].errors)
        assert "expect_eq" in str(results[0].errors)

    def test_assertion_passes(self):
        """Passing assertions should result in passing test."""
        class PassingSuite(TestSuite):
            def test_assertion_passes(self):
                expect_eq(1, 1)
                expect_eq("hello", "hello")

        runner = TestRunner(verbose=False)
        runner.add_suite(PassingSuite)
        results = runner.run()

        assert results[0].passed
