"""
Automation framework with @automation_test decorator.

Provides a framework for writing and running automation tests,
including setup/teardown, step tracking, and result reporting.
"""

from __future__ import annotations

import functools
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class AutomationTestStatus(Enum):
    """Status of an automation test."""

    PENDING = auto()
    RUNNING = auto()
    PASSED = auto()
    FAILED = auto()
    ERROR = auto()
    SKIPPED = auto()
    TIMEOUT = auto()


@dataclass
class AutomationStep:
    """A single step in an automation test."""

    name: str
    status: AutomationTestStatus = AutomationTestStatus.PENDING
    duration: float = 0.0
    message: str = ""
    screenshot: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AutomationTestResult:
    """Result of an automation test execution."""

    name: str
    status: AutomationTestStatus
    duration: float = 0.0
    message: str = ""
    traceback: str = ""
    steps: List[AutomationStep] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    screenshots: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Check if test passed."""
        return self.status == AutomationTestStatus.PASSED

    @property
    def failed(self) -> bool:
        """Check if test failed."""
        return self.status in (
            AutomationTestStatus.FAILED,
            AutomationTestStatus.ERROR,
            AutomationTestStatus.TIMEOUT,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.name,
            "duration": self.duration,
            "message": self.message,
            "traceback": self.traceback,
            "steps": [
                {
                    "name": s.name,
                    "status": s.status.name,
                    "duration": s.duration,
                    "message": s.message,
                }
                for s in self.steps
            ],
            "artifacts": self.artifacts,
            "screenshots": self.screenshots,
            "metadata": self.metadata,
        }


def automation_test(
    func: Optional[F] = None,
    *,
    name: Optional[str] = None,
    category: str = "default",
    priority: int = 0,
    timeout: float = 300.0,
    retries: int = 0,
    tags: Optional[List[str]] = None,
    requires_gpu: bool = False,
    requires_network: bool = False,
) -> Union[F, Callable[[F], F]]:
    """
    Decorator to mark a function as an automation test.

    Args:
        func: The test function
        name: Custom test name
        category: Test category for grouping
        priority: Test priority (higher runs first)
        timeout: Test timeout in seconds
        retries: Number of retries on failure
        tags: Tags for filtering
        requires_gpu: Whether test requires GPU
        requires_network: Whether test requires network

    Example:
        @automation_test
        def test_level_loading():
            load_level("TestMap")
            assert level_is_loaded()

        @automation_test(category="rendering", requires_gpu=True)
        def test_shader_compilation():
            compile_shaders()
    """
    def decorator(f: F) -> F:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        wrapper._is_automation_test = True
        wrapper._automation_name = name or f.__name__
        wrapper._automation_category = category
        wrapper._automation_priority = priority
        wrapper._automation_timeout = timeout
        wrapper._automation_retries = retries
        wrapper._automation_tags = set(tags) if tags else set()
        wrapper._automation_requires_gpu = requires_gpu
        wrapper._automation_requires_network = requires_network

        return wrapper  # type: ignore

    if func is not None:
        return decorator(func)
    return decorator


def automation_step(name: str) -> Callable[[F], F]:
    """
    Decorator to mark a function as an automation step.

    Steps are tracked and reported as part of the test.

    Example:
        @automation_step("Load Level")
        def load_test_level():
            load_level("TestMap")
    """
    def decorator(f: F) -> F:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Get context from first arg if it's an AutomationTest
            context = args[0] if args and isinstance(args[0], AutomationTest) else None

            step = AutomationStep(name=name)

            if context:
                context._current_steps.append(step)

            step.status = AutomationTestStatus.RUNNING
            start_time = time.perf_counter()

            try:
                result = f(*args, **kwargs)
                step.status = AutomationTestStatus.PASSED
                step.duration = time.perf_counter() - start_time
                return result

            except Exception as e:
                step.status = AutomationTestStatus.FAILED
                step.message = str(e)
                step.duration = time.perf_counter() - start_time
                raise

        wrapper._is_automation_step = True
        wrapper._step_name = name

        return wrapper  # type: ignore

    return decorator


def requires(*dependencies: str) -> Callable[[F], F]:
    """
    Decorator to specify test dependencies.

    Example:
        @requires("level_loaded", "player_spawned")
        def test_player_movement():
            ...
    """
    def decorator(f: F) -> F:
        f._automation_requires = set(dependencies)
        return f

    return decorator


def timeout(seconds: float) -> Callable[[F], F]:
    """
    Decorator to set automation test timeout.

    Example:
        @timeout(60.0)
        def test_long_operation():
            ...
    """
    def decorator(f: F) -> F:
        f._automation_timeout = seconds
        return f

    return decorator


class AutomationTest:
    """
    Base class for automation tests with setup/teardown support.

    Example:
        class LevelLoadTest(AutomationTest):
            def setup(self):
                self.engine.initialize()

            def teardown(self):
                self.engine.shutdown()

            @automation_test
            def test_load_level(self):
                self.engine.load_level("TestMap")
                assert self.engine.is_level_loaded()
    """

    def __init__(self):
        self._current_steps: List[AutomationStep] = []
        self._artifacts: List[str] = []
        self._screenshots: List[str] = []
        self._logs: List[str] = []
        self._context: Dict[str, Any] = {}

    def setup(self) -> None:
        """Called before each test."""
        pass

    def teardown(self) -> None:
        """Called after each test."""
        pass

    @classmethod
    def setup_class(cls) -> None:
        """Called once before all tests in the class."""
        pass

    @classmethod
    def teardown_class(cls) -> None:
        """Called once after all tests in the class."""
        pass

    def add_artifact(self, path: str) -> None:
        """Add an artifact to the test results."""
        self._artifacts.append(path)

    def take_screenshot(self, name: str = "") -> str:
        """Take a screenshot and add to results."""
        # This would be implemented by the engine
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}_{name}.png" if name else f"screenshot_{timestamp}.png"
        self._screenshots.append(filename)
        return filename

    def log(self, message: str) -> None:
        """Add a log message to results."""
        self._logs.append(f"[{time.strftime('%H:%M:%S')}] {message}")

    def set_context(self, key: str, value: Any) -> None:
        """Set a context value for use in subsequent tests."""
        self._context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a context value."""
        return self._context.get(key, default)

    def wait_for_condition(
        self,
        condition: Callable[[], bool],
        timeout: float = 10.0,
        poll_interval: float = 0.1,
        message: str = "Condition not met",
    ) -> bool:
        """Wait for a condition to be true."""
        start = time.perf_counter()
        while time.perf_counter() - start < timeout:
            if condition():
                return True
            time.sleep(poll_interval)
        raise TimeoutError(f"{message} (waited {timeout}s)")

    @classmethod
    def get_test_methods(cls) -> List[str]:
        """Get all test method names."""
        methods = []
        for name in dir(cls):
            method = getattr(cls, name, None)
            if method and getattr(method, "_is_automation_test", False):
                methods.append(name)
        return methods


@dataclass
class AutomationTestSuite:
    """Collection of automation tests."""

    name: str
    tests: List[Callable] = field(default_factory=list)
    test_classes: List[Type[AutomationTest]] = field(default_factory=list)
    setup: Optional[Callable] = None
    teardown: Optional[Callable] = None
    category: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_test(self, test_func: Callable) -> None:
        """Add a test function to the suite."""
        self.tests.append(test_func)

    def add_test_class(self, test_class: Type[AutomationTest]) -> None:
        """Add a test class to the suite."""
        self.test_classes.append(test_class)

    def get_all_tests(self) -> List[Tuple[Optional[AutomationTest], Callable]]:
        """Get all tests as (instance, method) tuples."""
        result = []

        # Add standalone test functions
        for test in self.tests:
            result.append((None, test))

        # Add test methods from classes
        for cls in self.test_classes:
            instance = cls()
            for method_name in cls.get_test_methods():
                method = getattr(instance, method_name)
                result.append((instance, method))

        # Sort by priority
        result.sort(
            key=lambda x: getattr(x[1], "_automation_priority", 0),
            reverse=True,
        )

        return result


class AutomationTestRunner:
    """
    Runner for automation tests.

    Handles test execution, timeout management, retries, and result collection.
    """

    def __init__(
        self,
        timeout: float = 300.0,
        screenshot_on_failure: bool = True,
        stop_on_failure: bool = False,
    ):
        self.timeout = timeout
        self.screenshot_on_failure = screenshot_on_failure
        self.stop_on_failure = stop_on_failure
        self.results: List[AutomationTestResult] = []

    def run_test(
        self,
        test_func: Callable,
        instance: Optional[AutomationTest] = None,
    ) -> AutomationTestResult:
        """Run a single automation test."""
        test_name = getattr(test_func, "_automation_name", test_func.__name__)
        test_timeout = getattr(test_func, "_automation_timeout", self.timeout)
        retries = getattr(test_func, "_automation_retries", 0)

        result = AutomationTestResult(
            name=test_name,
            status=AutomationTestStatus.RUNNING,
        )

        # Setup
        if instance:
            instance._current_steps = []
            instance._artifacts = []
            instance._screenshots = []
            instance._logs = []

            try:
                instance.setup()
            except Exception as e:
                result.status = AutomationTestStatus.ERROR
                result.message = f"Setup failed: {e}"
                result.traceback = traceback.format_exc()
                return result

        # Run test with retries
        last_error = None
        for attempt in range(retries + 1):
            start_time = time.perf_counter()

            try:
                self._run_with_timeout(test_func, test_timeout)
                result.status = AutomationTestStatus.PASSED
                result.duration = time.perf_counter() - start_time
                break

            except TimeoutError:
                result.status = AutomationTestStatus.TIMEOUT
                result.message = f"Test timed out after {test_timeout}s"
                result.duration = time.perf_counter() - start_time
                last_error = result.message

            except AssertionError as e:
                result.status = AutomationTestStatus.FAILED
                result.message = str(e)
                result.traceback = traceback.format_exc()
                result.duration = time.perf_counter() - start_time
                last_error = str(e)

            except Exception as e:
                result.status = AutomationTestStatus.ERROR
                result.message = f"{type(e).__name__}: {e}"
                result.traceback = traceback.format_exc()
                result.duration = time.perf_counter() - start_time
                last_error = str(e)

            # Screenshot on failure
            if self.screenshot_on_failure and instance and result.failed:
                try:
                    instance.take_screenshot(f"failure_attempt_{attempt}")
                except Exception:
                    pass

        # Collect artifacts from instance
        if instance:
            result.steps = instance._current_steps
            result.artifacts = instance._artifacts
            result.screenshots = instance._screenshots
            result.logs = instance._logs

            # Teardown
            try:
                instance.teardown()
            except Exception as e:
                if result.passed:
                    result.status = AutomationTestStatus.ERROR
                    result.message = f"Teardown failed: {e}"

        return result

    def _run_with_timeout(self, func: Callable, timeout: float) -> Any:
        """Run a function with timeout."""
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

    def run_suite(self, suite: AutomationTestSuite) -> List[AutomationTestResult]:
        """Run all tests in a suite."""
        suite_results = []

        # Suite setup
        if suite.setup:
            try:
                suite.setup()
            except Exception as e:
                error_result = AutomationTestResult(
                    name=f"{suite.name}.setup",
                    status=AutomationTestStatus.ERROR,
                    message=f"Suite setup failed: {e}",
                )
                return [error_result]

        # Class setup for test classes
        for cls in suite.test_classes:
            try:
                cls.setup_class()
            except Exception as e:
                error_result = AutomationTestResult(
                    name=f"{cls.__name__}.setup_class",
                    status=AutomationTestStatus.ERROR,
                    message=f"Class setup failed: {e}",
                )
                suite_results.append(error_result)

        # Run tests
        for instance, test_func in suite.get_all_tests():
            result = self.run_test(test_func, instance)
            suite_results.append(result)
            self.results.append(result)

            if self.stop_on_failure and result.failed:
                break

        # Class teardown
        for cls in suite.test_classes:
            try:
                cls.teardown_class()
            except Exception:
                pass

        # Suite teardown
        if suite.teardown:
            try:
                suite.teardown()
            except Exception:
                pass

        return suite_results

    def run(self, suites: List[AutomationTestSuite]) -> List[AutomationTestResult]:
        """Run multiple test suites."""
        self.results.clear()

        for suite in suites:
            self.run_suite(suite)

            if self.stop_on_failure and any(r.failed for r in self.results):
                break

        return self.results

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of test results."""
        return {
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r.status == AutomationTestStatus.PASSED),
            "failed": sum(1 for r in self.results if r.status == AutomationTestStatus.FAILED),
            "errors": sum(1 for r in self.results if r.status == AutomationTestStatus.ERROR),
            "skipped": sum(1 for r in self.results if r.status == AutomationTestStatus.SKIPPED),
            "timeout": sum(1 for r in self.results if r.status == AutomationTestStatus.TIMEOUT),
            "duration": sum(r.duration for r in self.results),
        }
