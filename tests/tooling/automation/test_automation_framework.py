"""
Tests for automation_framework.py - Automation framework with decorators.
"""

import time

import pytest

from engine.tooling.automation.automation_framework import (
    AutomationStep,
    AutomationTest,
    AutomationTestResult,
    AutomationTestRunner,
    AutomationTestStatus,
    AutomationTestSuite,
    automation_step,
    automation_test,
    requires,
    timeout,
)


class TestAutomationTestStatus:
    """Tests for AutomationTestStatus enum."""

    def test_status_values_exist(self):
        assert AutomationTestStatus.PENDING
        assert AutomationTestStatus.RUNNING
        assert AutomationTestStatus.PASSED
        assert AutomationTestStatus.FAILED
        assert AutomationTestStatus.ERROR
        assert AutomationTestStatus.SKIPPED
        assert AutomationTestStatus.TIMEOUT


class TestAutomationTestDecorator:
    """Tests for @automation_test decorator."""

    def test_basic_decorator(self):
        @automation_test
        def test_example():
            pass

        assert test_example._is_automation_test is True
        assert test_example._automation_name == "test_example"

    def test_decorator_with_name(self):
        @automation_test(name="CustomTestName")
        def test_example():
            pass

        assert test_example._automation_name == "CustomTestName"

    def test_decorator_with_category(self):
        @automation_test(category="rendering")
        def test_shader():
            pass

        assert test_shader._automation_category == "rendering"

    def test_decorator_with_priority(self):
        @automation_test(priority=100)
        def test_critical():
            pass

        assert test_critical._automation_priority == 100

    def test_decorator_with_timeout(self):
        @automation_test(timeout=60.0)
        def test_long():
            pass

        assert test_long._automation_timeout == 60.0

    def test_decorator_with_retries(self):
        @automation_test(retries=3)
        def test_flaky():
            pass

        assert test_flaky._automation_retries == 3

    def test_decorator_with_tags(self):
        @automation_test(tags=["smoke", "critical"])
        def test_important():
            pass

        assert "smoke" in test_important._automation_tags
        assert "critical" in test_important._automation_tags

    def test_decorator_with_gpu_requirement(self):
        @automation_test(requires_gpu=True)
        def test_render():
            pass

        assert test_render._automation_requires_gpu is True

    def test_decorator_with_network_requirement(self):
        @automation_test(requires_network=True)
        def test_online():
            pass

        assert test_online._automation_requires_network is True


class TestAutomationStepDecorator:
    """Tests for @automation_step decorator."""

    def test_basic_step(self):
        @automation_step("Load Level")
        def load_level():
            pass

        assert load_level._is_automation_step is True
        assert load_level._step_name == "Load Level"


class TestRequiresDecorator:
    """Tests for @requires decorator."""

    def test_requires_dependencies(self):
        @requires("level_loaded", "player_spawned")
        def test_gameplay():
            pass

        assert "level_loaded" in test_gameplay._automation_requires
        assert "player_spawned" in test_gameplay._automation_requires


class TestTimeoutDecorator:
    """Tests for @timeout decorator."""

    def test_timeout_decorator(self):
        @timeout(30.0)
        def test_with_timeout():
            pass

        assert test_with_timeout._automation_timeout == 30.0


class TestAutomationStep:
    """Tests for AutomationStep dataclass."""

    def test_create_step(self):
        step = AutomationStep(name="Initialize")
        assert step.name == "Initialize"
        assert step.status == AutomationTestStatus.PENDING


class TestAutomationTestResult:
    """Tests for AutomationTestResult dataclass."""

    def test_create_result(self):
        result = AutomationTestResult(
            name="test_example",
            status=AutomationTestStatus.PASSED,
        )
        assert result.name == "test_example"
        assert result.passed is True

    def test_passed_property(self):
        passed = AutomationTestResult("t", AutomationTestStatus.PASSED)
        failed = AutomationTestResult("t", AutomationTestStatus.FAILED)

        assert passed.passed is True
        assert failed.passed is False

    def test_failed_property(self):
        passed = AutomationTestResult("t", AutomationTestStatus.PASSED)
        failed = AutomationTestResult("t", AutomationTestStatus.FAILED)
        error = AutomationTestResult("t", AutomationTestStatus.ERROR)
        timeout = AutomationTestResult("t", AutomationTestStatus.TIMEOUT)

        assert passed.failed is False
        assert failed.failed is True
        assert error.failed is True
        assert timeout.failed is True

    def test_to_dict(self):
        result = AutomationTestResult(
            name="test_example",
            status=AutomationTestStatus.PASSED,
            duration=1.5,
        )
        data = result.to_dict()

        assert data["name"] == "test_example"
        assert data["status"] == "PASSED"
        assert data["duration"] == 1.5


class TestAutomationTest:
    """Tests for AutomationTest base class."""

    def test_create_test_class(self):
        class MyTest(AutomationTest):
            @automation_test
            def test_example(self):
                pass

        test = MyTest()
        assert hasattr(test, "setup")
        assert hasattr(test, "teardown")

    def test_add_artifact(self):
        class MyTest(AutomationTest):
            pass

        test = MyTest()
        test.add_artifact("/path/to/file.log")

        assert "/path/to/file.log" in test._artifacts

    def test_log_method(self):
        class MyTest(AutomationTest):
            pass

        test = MyTest()
        test.log("Test message")

        assert any("Test message" in log for log in test._logs)

    def test_set_and_get_context(self):
        class MyTest(AutomationTest):
            pass

        test = MyTest()
        test.set_context("player_id", 123)

        assert test.get_context("player_id") == 123
        assert test.get_context("missing", "default") == "default"

    def test_get_test_methods(self):
        class MyTest(AutomationTest):
            @automation_test
            def test_one(self):
                pass

            @automation_test
            def test_two(self):
                pass

            def helper(self):
                pass

        methods = MyTest.get_test_methods()
        assert "test_one" in methods
        assert "test_two" in methods
        assert "helper" not in methods


class TestAutomationTestSuite:
    """Tests for AutomationTestSuite dataclass."""

    def test_create_suite(self):
        suite = AutomationTestSuite(name="TestSuite")
        assert suite.name == "TestSuite"

    def test_add_test(self):
        suite = AutomationTestSuite(name="TestSuite")

        @automation_test
        def test_example():
            pass

        suite.add_test(test_example)
        assert len(suite.tests) == 1

    def test_add_test_class(self):
        suite = AutomationTestSuite(name="TestSuite")

        class MyTests(AutomationTest):
            @automation_test
            def test_example(self):
                pass

        suite.add_test_class(MyTests)
        assert len(suite.test_classes) == 1

    def test_get_all_tests(self):
        suite = AutomationTestSuite(name="TestSuite")

        @automation_test
        def standalone_test():
            pass

        class MyTests(AutomationTest):
            @automation_test
            def test_from_class(self):
                pass

        suite.add_test(standalone_test)
        suite.add_test_class(MyTests)

        all_tests = suite.get_all_tests()
        assert len(all_tests) == 2


class TestAutomationTestRunner:
    """Tests for AutomationTestRunner class."""

    def test_create_runner(self):
        runner = AutomationTestRunner(timeout=120.0)
        assert runner.timeout == 120.0

    def test_run_passing_test(self):
        runner = AutomationTestRunner()

        @automation_test
        def test_pass():
            pass

        result = runner.run_test(test_pass)
        assert result.status == AutomationTestStatus.PASSED

    def test_run_failing_test(self):
        runner = AutomationTestRunner()

        @automation_test
        def test_fail():
            assert False, "Expected failure"

        result = runner.run_test(test_fail)
        assert result.status == AutomationTestStatus.FAILED

    def test_run_error_test(self):
        runner = AutomationTestRunner()

        @automation_test
        def test_error():
            raise RuntimeError("Unexpected error")

        result = runner.run_test(test_error)
        assert result.status == AutomationTestStatus.ERROR

    def test_run_with_timeout(self):
        runner = AutomationTestRunner()

        @automation_test(timeout=0.1)
        def test_slow():
            time.sleep(1.0)

        result = runner.run_test(test_slow)
        assert result.status == AutomationTestStatus.TIMEOUT

    def test_run_suite(self):
        runner = AutomationTestRunner()
        suite = AutomationTestSuite(name="TestSuite")

        @automation_test
        def test_one():
            pass

        @automation_test
        def test_two():
            pass

        suite.add_test(test_one)
        suite.add_test(test_two)

        results = runner.run_suite(suite)
        assert len(results) == 2
        assert all(r.status == AutomationTestStatus.PASSED for r in results)

    def test_get_summary(self):
        runner = AutomationTestRunner()
        suite = AutomationTestSuite(name="TestSuite")

        @automation_test
        def test_pass():
            pass

        @automation_test
        def test_fail():
            assert False

        suite.add_test(test_pass)
        suite.add_test(test_fail)

        runner.run_suite(suite)
        summary = runner.get_summary()

        assert summary["total"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1

    def test_stop_on_failure(self):
        runner = AutomationTestRunner(stop_on_failure=True)
        suite = AutomationTestSuite(name="TestSuite")

        @automation_test
        def test_fail():
            assert False

        @automation_test
        def test_after():
            pass

        suite.add_test(test_fail)
        suite.add_test(test_after)

        results = runner.run_suite(suite)
        # Should stop after first failure
        assert len(results) == 1
