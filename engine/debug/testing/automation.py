"""
Automated testing support for the engine testing framework.

Provides automation bots and test scenarios for automated gameplay testing,
smoke tests, regression tests, and stress testing.

Usage:
    from engine.debug.testing.automation import AutomationBot, TestScenario

    scenario = TestScenario("login_flow")
    scenario.add_step(Action.input("username", "test@example.com"), expected="username_filled")
    scenario.add_step(Action.click("login_button"), expected="login_screen")
    scenario.add_step(Action.wait(condition=lambda: app.logged_in), expected="logged_in")

    bot = AutomationBot()
    result = bot.run_scenario(scenario)
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)


__all__ = [
    "AutomationBot",
    "TestScenario",
    "ScenarioStep",
    "StepResult",
    "ScenarioResult",
    "Action",
    "InputSimulator",
    "TimeoutError",
    "ConditionNotMetError",
]


T = TypeVar("T")

# Constants for magic numbers
DEFAULT_ACTION_TIMEOUT_MS = 5000.0
DEFAULT_POLL_INTERVAL_MS = 100.0


class TimeoutError(Exception):
    """Raised when a wait operation times out."""

    def __init__(self, message: str, waited_ms: float) -> None:
        self.waited_ms = waited_ms
        super().__init__(f"{message} (waited {waited_ms:.0f}ms)")


class ConditionNotMetError(Exception):
    """Raised when an expected condition is not met."""

    def __init__(self, condition: str, actual: Any = None) -> None:
        self.condition = condition
        self.actual = actual
        super().__init__(f"Condition not met: {condition}" + (f" (got: {actual})" if actual else ""))


class ActionType(Enum):
    """Types of actions that can be performed by the automation bot."""

    INPUT = auto()        # Simulate input (keyboard, mouse, gamepad)
    WAIT = auto()         # Wait for condition or timeout
    EXECUTE = auto()      # Execute a command/function
    VERIFY = auto()       # Verify a condition
    CHECKPOINT = auto()   # Create a state checkpoint
    RESTORE = auto()      # Restore from checkpoint
    LOG = auto()          # Log a message
    SCREENSHOT = auto()   # Capture screenshot


@dataclass
class Action:
    """
    An action to be performed by the automation bot.

    Actions represent atomic operations like clicking buttons,
    entering text, waiting for conditions, etc.
    """

    action_type: ActionType
    target: Optional[str] = None
    value: Any = None
    timeout_ms: float = DEFAULT_ACTION_TIMEOUT_MS
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def input(
        target: str,
        value: Any,
        input_type: str = "text",
    ) -> "Action":
        """
        Create an input action.

        Args:
            target: Target element/component identifier
            value: Value to input
            input_type: Type of input ("text", "key", "mouse", "gamepad")

        Returns:
            Input action
        """
        return Action(
            action_type=ActionType.INPUT,
            target=target,
            value=value,
            metadata={"input_type": input_type},
        )

    @staticmethod
    def click(target: str, button: str = "left") -> "Action":
        """
        Create a mouse click action.

        Args:
            target: Target element/component identifier
            button: Mouse button ("left", "right", "middle")

        Returns:
            Click action
        """
        return Action(
            action_type=ActionType.INPUT,
            target=target,
            value="click",
            metadata={"input_type": "mouse", "button": button},
        )

    @staticmethod
    def key(key: str, modifiers: Optional[List[str]] = None) -> "Action":
        """
        Create a keyboard action.

        Args:
            key: Key to press
            modifiers: Modifier keys (ctrl, shift, alt)

        Returns:
            Key action
        """
        return Action(
            action_type=ActionType.INPUT,
            target=None,
            value=key,
            metadata={"input_type": "key", "modifiers": modifiers or []},
        )

    @staticmethod
    def wait(
        condition: Optional[Callable[[], bool]] = None,
        timeout_ms: float = DEFAULT_ACTION_TIMEOUT_MS,
        poll_interval_ms: float = DEFAULT_POLL_INTERVAL_MS,
    ) -> "Action":
        """
        Create a wait action.

        Args:
            condition: Condition to wait for (None for fixed delay)
            timeout_ms: Maximum wait time
            poll_interval_ms: How often to check condition

        Returns:
            Wait action
        """
        return Action(
            action_type=ActionType.WAIT,
            value=condition,
            timeout_ms=timeout_ms,
            metadata={"poll_interval_ms": poll_interval_ms},
        )

    @staticmethod
    def delay(ms: float) -> "Action":
        """
        Create a fixed delay action.

        Args:
            ms: Delay in milliseconds

        Returns:
            Delay action
        """
        return Action(
            action_type=ActionType.WAIT,
            value=None,
            timeout_ms=ms,
        )

    @staticmethod
    def execute(command: Union[str, Callable[[], Any]]) -> "Action":
        """
        Create an execute action.

        Args:
            command: Command string or callable to execute

        Returns:
            Execute action
        """
        return Action(
            action_type=ActionType.EXECUTE,
            value=command,
        )

    @staticmethod
    def verify(
        condition: Callable[[], bool],
        message: str = "",
    ) -> "Action":
        """
        Create a verify action.

        Args:
            condition: Condition to verify
            message: Error message if verification fails

        Returns:
            Verify action
        """
        return Action(
            action_type=ActionType.VERIFY,
            value=condition,
            metadata={"message": message},
        )

    @staticmethod
    def checkpoint(name: str) -> "Action":
        """
        Create a checkpoint action.

        Args:
            name: Checkpoint name

        Returns:
            Checkpoint action
        """
        return Action(
            action_type=ActionType.CHECKPOINT,
            target=name,
        )

    @staticmethod
    def restore(name: str) -> "Action":
        """
        Create a restore action.

        Args:
            name: Checkpoint name to restore

        Returns:
            Restore action
        """
        return Action(
            action_type=ActionType.RESTORE,
            target=name,
        )

    @staticmethod
    def log(message: str, level: str = "info") -> "Action":
        """
        Create a log action.

        Args:
            message: Message to log
            level: Log level

        Returns:
            Log action
        """
        return Action(
            action_type=ActionType.LOG,
            value=message,
            metadata={"level": level},
        )

    @staticmethod
    def screenshot(name: Optional[str] = None) -> "Action":
        """
        Create a screenshot action.

        Args:
            name: Screenshot filename (auto-generated if None)

        Returns:
            Screenshot action
        """
        return Action(
            action_type=ActionType.SCREENSHOT,
            target=name,
        )


@dataclass
class StepResult:
    """
    Result of executing a single scenario step.

    Attributes:
        step_index: Index of the step in the scenario
        action: The action that was executed
        success: True if step completed successfully
        actual: Actual result/state
        expected: Expected result/state
        duration_ms: Execution time
        error: Error message if failed
    """

    step_index: int
    action: Action
    success: bool = True
    actual: Any = None
    expected: Any = None
    duration_ms: float = 0.0
    error: Optional[str] = None

    @property
    def status(self) -> str:
        """Get the result status as a string."""
        return "PASS" if self.success else "FAIL"


@dataclass
class ScenarioResult:
    """
    Result of executing a complete test scenario.

    Attributes:
        scenario_name: Name of the scenario
        steps: Results for each step
        success: True if all steps passed
        duration_ms: Total execution time
        checkpoints: Captured checkpoints
    """

    scenario_name: str
    steps: List[StepResult] = field(default_factory=list)
    success: bool = True
    duration_ms: float = 0.0
    checkpoints: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed_steps(self) -> int:
        """Count of passed steps."""
        return sum(1 for s in self.steps if s.success)

    @property
    def failed_steps(self) -> int:
        """Count of failed steps."""
        return sum(1 for s in self.steps if not s.success)

    @property
    def total_steps(self) -> int:
        """Total number of steps."""
        return len(self.steps)

    def __repr__(self) -> str:
        status = "PASS" if self.success else "FAIL"
        return (
            f"ScenarioResult({self.scenario_name!r}, {status}, "
            f"{self.passed_steps}/{self.total_steps} steps, {self.duration_ms:.2f}ms)"
        )


@dataclass
class ScenarioStep:
    """
    A single step in a test scenario.

    Attributes:
        action: The action to perform
        expected: Expected state/result after the action
        description: Human-readable description
    """

    action: Action
    expected: Any = None
    description: str = ""


class TestScenario:
    """
    A sequence of steps representing a test scenario.

    Scenarios define automated test flows like:
    - User login flow
    - Combat sequence
    - Menu navigation
    - Save/load cycle

    Example:
        scenario = TestScenario("player_combat")
        scenario.add_step(Action.execute("spawn_enemy"), expected="enemy_spawned")
        scenario.add_step(Action.input("attack", "primary"), expected="damage_dealt")
        scenario.add_step(Action.verify(lambda: enemy.health < 100))
    """

    def __init__(self, name: str, description: str = "") -> None:
        """
        Initialize a test scenario.

        Args:
            name: Scenario name
            description: Human-readable description
        """
        self.name = name
        self.description = description
        self._steps: List[ScenarioStep] = []
        self._variables: Dict[str, Any] = {}

    @property
    def steps(self) -> List[ScenarioStep]:
        """Get all steps in the scenario."""
        return self._steps.copy()

    def add_step(
        self,
        action: Action,
        expected: Any = None,
        description: str = "",
    ) -> "TestScenario":
        """
        Add a step to the scenario.

        Args:
            action: The action to perform
            expected: Expected state/result after the action
            description: Human-readable description

        Returns:
            Self for chaining
        """
        step = ScenarioStep(
            action=action,
            expected=expected,
            description=description or f"Step {len(self._steps) + 1}",
        )
        self._steps.append(step)
        return self

    def set_variable(self, name: str, value: Any) -> "TestScenario":
        """
        Set a scenario variable.

        Args:
            name: Variable name
            value: Variable value

        Returns:
            Self for chaining
        """
        self._variables[name] = value
        return self

    def get_variable(self, name: str, default: Any = None) -> Any:
        """
        Get a scenario variable.

        Args:
            name: Variable name
            default: Default value if not found

        Returns:
            Variable value
        """
        return self._variables.get(name, default)

    def run(self, bot: Optional["AutomationBot"] = None) -> ScenarioResult:
        """
        Run the scenario.

        Args:
            bot: AutomationBot to use (creates one if None)

        Returns:
            ScenarioResult with step results
        """
        if bot is None:
            bot = AutomationBot()
        return bot.run_scenario(self)

    def __repr__(self) -> str:
        return f"TestScenario({self.name!r}, {len(self._steps)} steps)"


class InputSimulator:
    """
    Simulates input for automated testing.

    Override this class to integrate with the actual game input system.
    """

    def __init__(self) -> None:
        """Initialize the input simulator."""
        self._input_log: List[Dict[str, Any]] = []

    def simulate_text(self, target: str, text: str) -> None:
        """
        Simulate text input.

        Args:
            target: Target element
            text: Text to input
        """
        self._input_log.append({
            "type": "text",
            "target": target,
            "value": text,
            "timestamp": time.time(),
        })

    def simulate_key(
        self,
        key: str,
        modifiers: Optional[List[str]] = None,
        press: bool = True,
        release: bool = True,
    ) -> None:
        """
        Simulate a key press.

        Args:
            key: Key to press
            modifiers: Modifier keys
            press: Whether to simulate key press
            release: Whether to simulate key release
        """
        self._input_log.append({
            "type": "key",
            "key": key,
            "modifiers": modifiers or [],
            "press": press,
            "release": release,
            "timestamp": time.time(),
        })

    def simulate_mouse_click(
        self,
        target: str,
        button: str = "left",
        double: bool = False,
    ) -> None:
        """
        Simulate a mouse click.

        Args:
            target: Target element
            button: Mouse button
            double: Double click
        """
        self._input_log.append({
            "type": "mouse_click",
            "target": target,
            "button": button,
            "double": double,
            "timestamp": time.time(),
        })

    def simulate_mouse_move(
        self,
        x: float,
        y: float,
        relative: bool = False,
    ) -> None:
        """
        Simulate mouse movement.

        Args:
            x: X position or delta
            y: Y position or delta
            relative: If True, x/y are relative deltas
        """
        self._input_log.append({
            "type": "mouse_move",
            "x": x,
            "y": y,
            "relative": relative,
            "timestamp": time.time(),
        })

    def simulate_gamepad_button(
        self,
        button: str,
        pressed: bool = True,
    ) -> None:
        """
        Simulate gamepad button.

        Args:
            button: Button name
            pressed: Button state
        """
        self._input_log.append({
            "type": "gamepad_button",
            "button": button,
            "pressed": pressed,
            "timestamp": time.time(),
        })

    def simulate_gamepad_axis(
        self,
        axis: str,
        value: float,
    ) -> None:
        """
        Simulate gamepad axis.

        Args:
            axis: Axis name
            value: Axis value (-1.0 to 1.0)
        """
        self._input_log.append({
            "type": "gamepad_axis",
            "axis": axis,
            "value": max(-1.0, min(1.0, value)),
            "timestamp": time.time(),
        })

    def get_log(self) -> List[Dict[str, Any]]:
        """Get the input log."""
        return self._input_log.copy()

    def clear_log(self) -> None:
        """Clear the input log."""
        self._input_log.clear()


class AutomationBot:
    """
    Automation bot for executing test scenarios.

    The bot executes actions, waits for conditions, and verifies
    expected states. It can be extended to integrate with the
    game's input and state systems.

    Example:
        bot = AutomationBot()

        # Execute a single script
        bot.execute_script([
            Action.execute("load_level level_01"),
            Action.wait(lambda: game.level_loaded),
            Action.execute("spawn_player"),
            Action.verify(lambda: player.is_alive),
        ])

        # Run a scenario
        result = bot.run_scenario(login_scenario)
        assert result.success
    """

    def __init__(
        self,
        input_simulator: Optional[InputSimulator] = None,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the automation bot.

        Args:
            input_simulator: Custom input simulator (creates default if None)
            verbose: If True, log actions as they execute
        """
        self.input_simulator = input_simulator or InputSimulator()
        self.verbose = verbose
        self._checkpoints: Dict[str, Any] = {}
        self._command_handlers: Dict[str, Callable[[str], Any]] = {}
        self._state_getters: Dict[str, Callable[[], Any]] = {}

    def register_command(
        self,
        name: str,
        handler: Callable[[str], Any],
    ) -> None:
        """
        Register a command handler.

        Args:
            name: Command name
            handler: Handler function taking command string
        """
        self._command_handlers[name] = handler

    def register_state_getter(
        self,
        name: str,
        getter: Callable[[], Any],
    ) -> None:
        """
        Register a state getter for verification.

        Args:
            name: State name
            getter: Function returning current state
        """
        self._state_getters[name] = getter

    def execute_script(self, commands: List[Action]) -> List[StepResult]:
        """
        Execute a sequence of actions.

        Args:
            commands: List of actions to execute

        Returns:
            List of step results
        """
        results = []
        for i, action in enumerate(commands):
            result = self._execute_action(i, action)
            results.append(result)
            if not result.success:
                break
        return results

    def run_scenario(self, scenario: TestScenario) -> ScenarioResult:
        """
        Run a complete test scenario.

        Args:
            scenario: The scenario to run

        Returns:
            ScenarioResult with all step results
        """
        result = ScenarioResult(scenario_name=scenario.name)
        start_time = time.perf_counter_ns()

        self._log(f"Running scenario: {scenario.name}")

        for i, step in enumerate(scenario.steps):
            step_result = self._execute_step(i, step)
            result.steps.append(step_result)

            if not step_result.success:
                result.success = False
                break

        result.duration_ms = (time.perf_counter_ns() - start_time) / 1_000_000
        result.checkpoints = self._checkpoints.copy()

        status = "PASS" if result.success else "FAIL"
        self._log(f"Scenario {scenario.name}: {status} ({result.duration_ms:.2f}ms)")

        return result

    def wait_for(
        self,
        condition: Callable[[], bool],
        timeout_ms: float = DEFAULT_ACTION_TIMEOUT_MS,
        poll_interval_ms: float = DEFAULT_POLL_INTERVAL_MS,
        message: str = "Condition not met",
    ) -> float:
        """
        Wait for a condition to become true.

        Args:
            condition: Condition to wait for
            timeout_ms: Maximum wait time
            poll_interval_ms: How often to check condition
            message: Error message if timeout

        Returns:
            Time waited in milliseconds

        Raises:
            TimeoutError: If condition is not met within timeout
        """
        start_time = time.perf_counter_ns()
        timeout_ns = timeout_ms * 1_000_000
        poll_interval_s = poll_interval_ms / 1000.0

        while True:
            if condition():
                return (time.perf_counter_ns() - start_time) / 1_000_000

            elapsed_ns = time.perf_counter_ns() - start_time
            if elapsed_ns >= timeout_ns:
                raise TimeoutError(message, elapsed_ns / 1_000_000)

            time.sleep(poll_interval_s)

    def simulate_input(self, input_data: Dict[str, Any]) -> None:
        """
        Simulate input based on input data dictionary.

        Args:
            input_data: Dictionary describing the input to simulate
                - type: "key", "mouse", "gamepad", "text"
                - Additional fields based on type
        """
        input_type = input_data.get("type", "text")

        if input_type == "text":
            self.input_simulator.simulate_text(
                input_data.get("target", ""),
                input_data.get("value", ""),
            )
        elif input_type == "key":
            self.input_simulator.simulate_key(
                input_data.get("key", ""),
                input_data.get("modifiers"),
                input_data.get("press", True),
                input_data.get("release", True),
            )
        elif input_type == "mouse":
            if input_data.get("action") == "click":
                self.input_simulator.simulate_mouse_click(
                    input_data.get("target", ""),
                    input_data.get("button", "left"),
                    input_data.get("double", False),
                )
            elif input_data.get("action") == "move":
                self.input_simulator.simulate_mouse_move(
                    input_data.get("x", 0),
                    input_data.get("y", 0),
                    input_data.get("relative", False),
                )
        elif input_type == "gamepad":
            if "button" in input_data:
                self.input_simulator.simulate_gamepad_button(
                    input_data["button"],
                    input_data.get("pressed", True),
                )
            elif "axis" in input_data:
                self.input_simulator.simulate_gamepad_axis(
                    input_data["axis"],
                    input_data.get("value", 0.0),
                )

    def _execute_action(self, index: int, action: Action) -> StepResult:
        """Execute a single action."""
        result = StepResult(step_index=index, action=action)
        start_time = time.perf_counter_ns()

        try:
            if action.action_type == ActionType.INPUT:
                self._do_input(action)

            elif action.action_type == ActionType.WAIT:
                self._do_wait(action)

            elif action.action_type == ActionType.EXECUTE:
                result.actual = self._do_execute(action)

            elif action.action_type == ActionType.VERIFY:
                self._do_verify(action)

            elif action.action_type == ActionType.CHECKPOINT:
                self._do_checkpoint(action)

            elif action.action_type == ActionType.RESTORE:
                self._do_restore(action)

            elif action.action_type == ActionType.LOG:
                self._do_log(action)

            elif action.action_type == ActionType.SCREENSHOT:
                self._do_screenshot(action)

            result.success = True

        except Exception as e:
            result.success = False
            result.error = f"{type(e).__name__}: {e}"

        result.duration_ms = (time.perf_counter_ns() - start_time) / 1_000_000
        return result

    def _execute_step(self, index: int, step: ScenarioStep) -> StepResult:
        """Execute a scenario step."""
        result = self._execute_action(index, step.action)
        result.expected = step.expected

        # Verify expected state if provided
        if result.success and step.expected is not None:
            try:
                self._verify_expected(step.expected, result)
            except Exception as e:
                result.success = False
                result.error = str(e)

        return result

    def _do_input(self, action: Action) -> None:
        """Handle input action."""
        input_type = action.metadata.get("input_type", "text")

        if input_type == "text":
            self.input_simulator.simulate_text(action.target or "", action.value)
        elif input_type == "key":
            self.input_simulator.simulate_key(
                action.value,
                action.metadata.get("modifiers"),
            )
        elif input_type == "mouse":
            if action.value == "click":
                self.input_simulator.simulate_mouse_click(
                    action.target or "",
                    action.metadata.get("button", "left"),
                )
            elif action.value == "move":
                self.input_simulator.simulate_mouse_move(
                    action.metadata.get("x", 0),
                    action.metadata.get("y", 0),
                )

        self._log(f"Input: {input_type} -> {action.target or 'global'}")

    def _do_wait(self, action: Action) -> None:
        """Handle wait action."""
        if action.value is None:
            # Fixed delay
            time.sleep(action.timeout_ms / 1000.0)
            self._log(f"Wait: {action.timeout_ms:.0f}ms")
        else:
            # Wait for condition
            poll_interval = action.metadata.get("poll_interval_ms", 100.0)
            waited = self.wait_for(
                action.value,
                action.timeout_ms,
                poll_interval,
            )
            self._log(f"Wait: condition met after {waited:.0f}ms")

    def _do_execute(self, action: Action) -> Any:
        """Handle execute action."""
        if callable(action.value):
            result = action.value()
            self._log(f"Execute: callable -> {result}")
            return result

        # String command
        command = action.value
        parts = command.split(None, 1)
        cmd_name = parts[0] if parts else ""
        cmd_args = parts[1] if len(parts) > 1 else ""

        if cmd_name in self._command_handlers:
            result = self._command_handlers[cmd_name](cmd_args)
            self._log(f"Execute: {cmd_name} -> {result}")
            return result

        self._log(f"Execute: {command} (no handler)")
        return None

    def _do_verify(self, action: Action) -> None:
        """Handle verify action."""
        condition = action.value
        if not callable(condition):
            raise ValueError("Verify action requires callable condition")

        if not condition():
            message = action.metadata.get("message", "Verification failed")
            raise ConditionNotMetError(message)

        self._log("Verify: passed")

    def _do_checkpoint(self, action: Action) -> None:
        """Handle checkpoint action."""
        name = action.target
        if not name:
            raise ValueError("Checkpoint requires name")

        # Store current state (placeholder - would integrate with game state)
        self._checkpoints[name] = {
            "timestamp": time.time(),
            "input_log": self.input_simulator.get_log(),
        }
        self._log(f"Checkpoint: {name}")

    def _do_restore(self, action: Action) -> None:
        """Handle restore action."""
        name = action.target
        if not name:
            raise ValueError("Restore requires checkpoint name")

        if name not in self._checkpoints:
            raise ValueError(f"Checkpoint not found: {name}")

        # Would restore game state here
        self._log(f"Restore: {name}")

    def _do_log(self, action: Action) -> None:
        """Handle log action."""
        level = action.metadata.get("level", "info")
        message = action.value
        print(f"[{level.upper()}] {message}")

    def _do_screenshot(self, action: Action) -> None:
        """Handle screenshot action."""
        name = action.target or f"screenshot_{time.time():.0f}"
        # Would capture screenshot here
        self._log(f"Screenshot: {name}")

    def _verify_expected(self, expected: Any, result: StepResult) -> None:
        """Verify expected state after action."""
        if callable(expected):
            if not expected():
                raise ConditionNotMetError("Expected condition not met")
        elif isinstance(expected, str):
            # Check state getter
            if expected in self._state_getters:
                actual = self._state_getters[expected]()
                result.actual = actual
                if not actual:
                    raise ConditionNotMetError(f"State '{expected}' is falsy", actual)

    def _log(self, message: str) -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(f"[AutomationBot] {message}")
