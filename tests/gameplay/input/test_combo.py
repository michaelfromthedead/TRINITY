"""Comprehensive tests for the combo detection system.

Tests cover combo sequence definition, timing windows, input buffering,
combo cancellation, overlapping combos, priority, and direction + button combos.

Note: Since combo.py doesn't exist in the source, these tests define
the expected interface and behavior for a combo detection system.
"""

import pytest
from typing import List, Optional, Dict, Callable, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from time import time
from unittest.mock import Mock, MagicMock


# =============================================================================
# Combo System Mock Implementation (for testing expected behavior)
# =============================================================================

class ComboInputType(Enum):
    """Types of inputs in a combo."""
    BUTTON = auto()
    DIRECTION = auto()
    HOLD = auto()
    RELEASE = auto()
    SIMULTANEOUS = auto()


class ComboState(Enum):
    """State of a combo."""
    IDLE = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class ComboInput:
    """A single input in a combo sequence."""
    input_type: ComboInputType
    inputs: List[str]  # e.g., ["down", "right", "punch"] for a fireball
    hold_time: float = 0.0  # For HOLD type
    tolerance: float = 0.1  # Timing tolerance


@dataclass
class ComboDefinition:
    """Defines a combo sequence."""
    name: str
    sequence: List[ComboInput]
    window: float = 0.5  # Time window between inputs
    priority: int = 0
    cancelable: bool = True
    buffer_inputs: bool = True
    description: str = ""


@dataclass
class ComboResult:
    """Result of a combo evaluation."""
    combo_name: str
    state: ComboState
    progress: float  # 0.0 to 1.0
    elapsed_time: float
    inputs_matched: int
    total_inputs: int


@dataclass
class BufferedInput:
    """An input stored in the buffer."""
    input_name: str
    timestamp: float
    value: float = 1.0
    consumed: bool = False


class ComboDetector:
    """Detects combo sequences from input."""

    def __init__(
        self,
        buffer_size: int = 32,
        buffer_lifetime: float = 1.0
    ):
        self._combos: Dict[str, ComboDefinition] = {}
        self._states: Dict[str, ComboState] = {}
        self._progress: Dict[str, int] = {}  # Input index in sequence
        self._start_times: Dict[str, float] = {}
        self._last_input_times: Dict[str, float] = {}

        self._input_buffer: List[BufferedInput] = []
        self._buffer_size = buffer_size
        self._buffer_lifetime = buffer_lifetime

        self._callbacks: Dict[str, List[Callable]] = {}
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    def register_combo(self, combo: ComboDefinition) -> bool:
        """Register a combo."""
        if combo.name in self._combos:
            return False

        self._combos[combo.name] = combo
        self._states[combo.name] = ComboState.IDLE
        self._progress[combo.name] = 0
        self._start_times[combo.name] = 0.0
        self._last_input_times[combo.name] = 0.0
        self._callbacks[combo.name] = []
        return True

    def unregister_combo(self, name: str) -> bool:
        """Unregister a combo."""
        if name not in self._combos:
            return False

        del self._combos[name]
        del self._states[name]
        del self._progress[name]
        del self._start_times[name]
        del self._last_input_times[name]
        del self._callbacks[name]
        return True

    def get_combo(self, name: str) -> Optional[ComboDefinition]:
        """Get a combo definition."""
        return self._combos.get(name)

    def get_combo_state(self, name: str) -> ComboState:
        """Get the current state of a combo."""
        return self._states.get(name, ComboState.IDLE)

    def get_combo_progress(self, name: str) -> float:
        """Get combo progress (0.0 to 1.0)."""
        if name not in self._combos:
            return 0.0

        combo = self._combos[name]
        progress = self._progress.get(name, 0)
        return progress / len(combo.sequence) if combo.sequence else 0.0

    def bind_callback(self, combo_name: str, callback: Callable) -> bool:
        """Bind a callback to combo completion."""
        if combo_name not in self._combos:
            return False
        self._callbacks[combo_name].append(callback)
        return True

    def unbind_callback(self, combo_name: str, callback: Callable) -> bool:
        """Unbind a callback."""
        if combo_name not in self._callbacks:
            return False
        try:
            self._callbacks[combo_name].remove(callback)
            return True
        except ValueError:
            return False

    def buffer_input(self, input_name: str, value: float = 1.0) -> None:
        """Add an input to the buffer."""
        buffered = BufferedInput(
            input_name=input_name,
            timestamp=time(),
            value=value
        )
        self._input_buffer.append(buffered)

        # Enforce buffer size
        while len(self._input_buffer) > self._buffer_size:
            self._input_buffer.pop(0)

    def get_buffer(self) -> List[BufferedInput]:
        """Get current input buffer."""
        return self._input_buffer.copy()

    def clear_buffer(self) -> None:
        """Clear the input buffer."""
        self._input_buffer.clear()

    def process_input(self, input_name: str, value: float = 1.0) -> List[ComboResult]:
        """Process an input and check for combo matches."""
        if not self._enabled:
            return []

        current_time = time()
        results = []

        # Buffer the input
        self.buffer_input(input_name, value)

        # Clean old buffer entries
        self._clean_buffer(current_time)

        # Check each combo
        for name, combo in self._combos.items():
            result = self._check_combo(name, combo, input_name, current_time)
            if result:
                results.append(result)

        # Sort by priority (higher first)
        results.sort(key=lambda r: self._combos[r.combo_name].priority, reverse=True)

        return results

    def _clean_buffer(self, current_time: float) -> None:
        """Remove old entries from buffer."""
        self._input_buffer = [
            b for b in self._input_buffer
            if current_time - b.timestamp < self._buffer_lifetime
        ]

    def _check_combo(
        self,
        name: str,
        combo: ComboDefinition,
        input_name: str,
        current_time: float
    ) -> Optional[ComboResult]:
        """Check if input advances or completes a combo."""
        state = self._states[name]
        progress_idx = self._progress[name]

        if not combo.sequence:
            return None

        # Check for timeout
        if state == ComboState.IN_PROGRESS:
            last_input = self._last_input_times[name]
            if current_time - last_input > combo.window:
                # Timed out
                self._reset_combo(name)
                return ComboResult(
                    combo_name=name,
                    state=ComboState.FAILED,
                    progress=0.0,
                    elapsed_time=0.0,
                    inputs_matched=0,
                    total_inputs=len(combo.sequence)
                )

        # Get expected input
        expected = combo.sequence[progress_idx]

        # Check if input matches
        if self._input_matches(input_name, expected):
            # Start or continue combo
            if state == ComboState.IDLE:
                self._states[name] = ComboState.IN_PROGRESS
                self._start_times[name] = current_time

            self._progress[name] = progress_idx + 1
            self._last_input_times[name] = current_time

            # Check if combo completed
            if self._progress[name] >= len(combo.sequence):
                self._states[name] = ComboState.COMPLETED
                elapsed = current_time - self._start_times[name]

                result = ComboResult(
                    combo_name=name,
                    state=ComboState.COMPLETED,
                    progress=1.0,
                    elapsed_time=elapsed,
                    inputs_matched=len(combo.sequence),
                    total_inputs=len(combo.sequence)
                )

                # Invoke callbacks
                for callback in self._callbacks.get(name, []):
                    try:
                        callback(result)
                    except Exception:
                        pass

                # Reset for next attempt
                self._reset_combo(name)

                return result
            else:
                # In progress
                return ComboResult(
                    combo_name=name,
                    state=ComboState.IN_PROGRESS,
                    progress=self.get_combo_progress(name),
                    elapsed_time=current_time - self._start_times[name],
                    inputs_matched=self._progress[name],
                    total_inputs=len(combo.sequence)
                )

        return None

    def _input_matches(self, input_name: str, expected: ComboInput) -> bool:
        """Check if input matches expected combo input."""
        if expected.input_type == ComboInputType.BUTTON:
            return input_name in expected.inputs
        elif expected.input_type == ComboInputType.DIRECTION:
            return input_name in expected.inputs
        elif expected.input_type == ComboInputType.SIMULTANEOUS:
            # Would need to check buffer for all inputs within tolerance
            return input_name in expected.inputs
        return False

    def _reset_combo(self, name: str) -> None:
        """Reset a combo to idle state."""
        self._states[name] = ComboState.IDLE
        self._progress[name] = 0
        self._start_times[name] = 0.0
        self._last_input_times[name] = 0.0

    def cancel_combo(self, name: str) -> bool:
        """Cancel an in-progress combo."""
        if name not in self._combos:
            return False

        combo = self._combos[name]
        if not combo.cancelable:
            return False

        if self._states[name] == ComboState.IN_PROGRESS:
            self._states[name] = ComboState.CANCELLED
            self._reset_combo(name)
            return True

        return False

    def cancel_all_combos(self) -> int:
        """Cancel all in-progress combos."""
        cancelled = 0
        for name in list(self._combos.keys()):
            if self.cancel_combo(name):
                cancelled += 1
        return cancelled

    def update(self, delta_time: float) -> List[ComboResult]:
        """Update combo detector (check for timeouts)."""
        current_time = time()
        results = []

        for name, combo in self._combos.items():
            state = self._states[name]

            if state == ComboState.IN_PROGRESS:
                last_input = self._last_input_times[name]
                if current_time - last_input > combo.window:
                    self._states[name] = ComboState.FAILED
                    results.append(ComboResult(
                        combo_name=name,
                        state=ComboState.FAILED,
                        progress=self.get_combo_progress(name),
                        elapsed_time=current_time - self._start_times[name],
                        inputs_matched=self._progress[name],
                        total_inputs=len(combo.sequence)
                    ))
                    self._reset_combo(name)

        return results

    def reset(self) -> None:
        """Reset all combos and clear buffer."""
        for name in self._combos:
            self._reset_combo(name)
        self.clear_buffer()


# =============================================================================
# Combo Input Type Tests
# =============================================================================

class TestComboInputType:
    """Tests for ComboInputType enum."""

    def test_input_types_exist(self):
        """All input types exist."""
        assert ComboInputType.BUTTON
        assert ComboInputType.DIRECTION
        assert ComboInputType.HOLD
        assert ComboInputType.RELEASE
        assert ComboInputType.SIMULTANEOUS


class TestComboState:
    """Tests for ComboState enum."""

    def test_combo_states_exist(self):
        """All combo states exist."""
        assert ComboState.IDLE
        assert ComboState.IN_PROGRESS
        assert ComboState.COMPLETED
        assert ComboState.FAILED
        assert ComboState.CANCELLED


# =============================================================================
# Combo Input Tests
# =============================================================================

class TestComboInput:
    """Tests for ComboInput dataclass."""

    def test_button_input(self):
        """Button input creation."""
        input_ = ComboInput(
            input_type=ComboInputType.BUTTON,
            inputs=["punch"]
        )
        assert input_.input_type == ComboInputType.BUTTON
        assert "punch" in input_.inputs

    def test_direction_input(self):
        """Direction input creation."""
        input_ = ComboInput(
            input_type=ComboInputType.DIRECTION,
            inputs=["down", "down-right", "right"]
        )
        assert input_.input_type == ComboInputType.DIRECTION

    def test_hold_input(self):
        """Hold input creation."""
        input_ = ComboInput(
            input_type=ComboInputType.HOLD,
            inputs=["punch"],
            hold_time=0.5
        )
        assert input_.hold_time == 0.5

    def test_simultaneous_input(self):
        """Simultaneous input creation."""
        input_ = ComboInput(
            input_type=ComboInputType.SIMULTANEOUS,
            inputs=["punch", "kick"]
        )
        assert len(input_.inputs) == 2

    def test_default_tolerance(self):
        """Default tolerance is set."""
        input_ = ComboInput(
            input_type=ComboInputType.BUTTON,
            inputs=["punch"]
        )
        assert input_.tolerance == 0.1


# =============================================================================
# Combo Definition Tests
# =============================================================================

class TestComboDefinition:
    """Tests for ComboDefinition dataclass."""

    def test_combo_creation(self):
        """ComboDefinition can be created."""
        combo = ComboDefinition(
            name="fireball",
            sequence=[
                ComboInput(ComboInputType.DIRECTION, ["down"]),
                ComboInput(ComboInputType.DIRECTION, ["down-right"]),
                ComboInput(ComboInputType.DIRECTION, ["right"]),
                ComboInput(ComboInputType.BUTTON, ["punch"]),
            ],
            window=0.5
        )
        assert combo.name == "fireball"
        assert len(combo.sequence) == 4

    def test_combo_defaults(self):
        """ComboDefinition has sensible defaults."""
        combo = ComboDefinition(
            name="test",
            sequence=[]
        )
        assert combo.window == 0.5
        assert combo.priority == 0
        assert combo.cancelable is True
        assert combo.buffer_inputs is True

    def test_combo_priority(self):
        """Combo priority is stored."""
        combo = ComboDefinition(
            name="super_move",
            sequence=[],
            priority=100
        )
        assert combo.priority == 100

    def test_combo_cancelable(self):
        """Combo cancelable flag is stored."""
        combo = ComboDefinition(
            name="uncancelable",
            sequence=[],
            cancelable=False
        )
        assert combo.cancelable is False


# =============================================================================
# Combo Detector Basic Tests
# =============================================================================

class TestComboDetectorBasic:
    """Basic tests for ComboDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a combo detector."""
        return ComboDetector()

    @pytest.fixture
    def simple_combo(self):
        """Create a simple 2-input combo."""
        return ComboDefinition(
            name="simple",
            sequence=[
                ComboInput(ComboInputType.BUTTON, ["a"]),
                ComboInput(ComboInputType.BUTTON, ["b"]),
            ],
            window=0.5
        )

    def test_register_combo(self, detector, simple_combo):
        """register_combo adds combo to detector."""
        result = detector.register_combo(simple_combo)
        assert result is True
        assert detector.get_combo("simple") is not None

    def test_register_duplicate_fails(self, detector, simple_combo):
        """Registering duplicate combo fails."""
        detector.register_combo(simple_combo)
        result = detector.register_combo(simple_combo)
        assert result is False

    def test_unregister_combo(self, detector, simple_combo):
        """unregister_combo removes combo."""
        detector.register_combo(simple_combo)
        result = detector.unregister_combo("simple")
        assert result is True
        assert detector.get_combo("simple") is None

    def test_unregister_nonexistent(self, detector):
        """Unregistering nonexistent combo fails."""
        result = detector.unregister_combo("nonexistent")
        assert result is False

    def test_get_combo(self, detector, simple_combo):
        """get_combo returns combo definition."""
        detector.register_combo(simple_combo)
        combo = detector.get_combo("simple")
        assert combo is simple_combo

    def test_get_combo_nonexistent(self, detector):
        """get_combo returns None for nonexistent."""
        assert detector.get_combo("nonexistent") is None

    def test_initial_state_is_idle(self, detector, simple_combo):
        """Combo starts in IDLE state."""
        detector.register_combo(simple_combo)
        assert detector.get_combo_state("simple") == ComboState.IDLE

    def test_initial_progress_is_zero(self, detector, simple_combo):
        """Combo progress starts at 0."""
        detector.register_combo(simple_combo)
        assert detector.get_combo_progress("simple") == 0.0

    def test_enabled_property(self, detector):
        """enabled property can be toggled."""
        assert detector.enabled is True
        detector.enabled = False
        assert detector.enabled is False


# =============================================================================
# Combo Sequence Tests
# =============================================================================

class TestComboSequence:
    """Tests for combo sequence detection."""

    @pytest.fixture
    def detector(self):
        """Create a combo detector with test combo."""
        det = ComboDetector()
        det.register_combo(ComboDefinition(
            name="abc",
            sequence=[
                ComboInput(ComboInputType.BUTTON, ["a"]),
                ComboInput(ComboInputType.BUTTON, ["b"]),
                ComboInput(ComboInputType.BUTTON, ["c"]),
            ],
            window=0.5
        ))
        return det

    def test_first_input_starts_combo(self, detector):
        """First matching input starts combo."""
        results = detector.process_input("a")

        assert len(results) == 1
        assert results[0].state == ComboState.IN_PROGRESS
        assert detector.get_combo_state("abc") == ComboState.IN_PROGRESS

    def test_progress_increases(self, detector):
        """Progress increases with matching inputs."""
        detector.process_input("a")
        results = detector.process_input("b")

        assert results[0].progress == pytest.approx(2/3, rel=0.01)

    def test_combo_completes(self, detector):
        """Combo completes with full sequence."""
        detector.process_input("a")
        detector.process_input("b")
        results = detector.process_input("c")

        assert len(results) == 1
        assert results[0].state == ComboState.COMPLETED
        assert results[0].progress == 1.0

    def test_wrong_input_does_not_match(self, detector):
        """Wrong input doesn't advance combo."""
        detector.process_input("a")
        results = detector.process_input("x")  # Wrong input

        # No result for wrong input (combo stays in progress)
        assert not any(r.combo_name == "abc" for r in results)

    def test_combo_resets_after_completion(self, detector):
        """Combo resets after completion."""
        detector.process_input("a")
        detector.process_input("b")
        detector.process_input("c")

        assert detector.get_combo_state("abc") == ComboState.IDLE
        assert detector.get_combo_progress("abc") == 0.0


# =============================================================================
# Timing Window Tests
# =============================================================================

class TestTimingWindow:
    """Tests for combo timing windows."""

    @pytest.fixture
    def detector(self):
        """Create a combo detector with short window."""
        det = ComboDetector()
        det.register_combo(ComboDefinition(
            name="quick",
            sequence=[
                ComboInput(ComboInputType.BUTTON, ["a"]),
                ComboInput(ComboInputType.BUTTON, ["b"]),
            ],
            window=0.1  # Very short window
        ))
        return det

    def test_inputs_within_window_succeed(self, detector):
        """Inputs within window succeed."""
        detector.process_input("a")
        # Immediate next input
        results = detector.process_input("b")

        assert any(r.state == ComboState.COMPLETED for r in results)

    def test_inputs_outside_window_fail(self, detector):
        """Inputs outside window cause failure."""
        import time

        detector.process_input("a")
        time.sleep(0.15)  # Wait beyond window
        results = detector.update(0.0)  # Trigger timeout check

        assert any(r.state == ComboState.FAILED for r in results)

    def test_window_per_combo(self):
        """Each combo has its own timing window."""
        detector = ComboDetector()

        detector.register_combo(ComboDefinition(
            name="fast",
            sequence=[
                ComboInput(ComboInputType.BUTTON, ["a"]),
                ComboInput(ComboInputType.BUTTON, ["b"]),
            ],
            window=0.1
        ))
        detector.register_combo(ComboDefinition(
            name="slow",
            sequence=[
                ComboInput(ComboInputType.BUTTON, ["a"]),
                ComboInput(ComboInputType.BUTTON, ["b"]),
            ],
            window=1.0
        ))

        # Both have different windows
        fast = detector.get_combo("fast")
        slow = detector.get_combo("slow")
        assert fast.window != slow.window

    def test_elapsed_time_tracked(self, detector):
        """Elapsed time is tracked in results."""
        detector.process_input("a")
        results = detector.process_input("b")

        assert results[0].elapsed_time >= 0.0


# =============================================================================
# Input Buffer Tests
# =============================================================================

class TestInputBuffer:
    """Tests for input buffering."""

    @pytest.fixture
    def detector(self):
        """Create a combo detector."""
        return ComboDetector(buffer_size=10, buffer_lifetime=0.5)

    def test_buffer_input(self, detector):
        """buffer_input adds to buffer."""
        detector.buffer_input("punch")
        buffer = detector.get_buffer()

        assert len(buffer) == 1
        assert buffer[0].input_name == "punch"

    def test_buffer_multiple_inputs(self, detector):
        """Multiple inputs are buffered."""
        detector.buffer_input("punch")
        detector.buffer_input("kick")
        detector.buffer_input("block")

        buffer = detector.get_buffer()
        assert len(buffer) == 3

    def test_buffer_respects_size_limit(self, detector):
        """Buffer doesn't exceed size limit."""
        for i in range(20):
            detector.buffer_input(f"input_{i}")

        buffer = detector.get_buffer()
        assert len(buffer) == 10  # buffer_size

    def test_buffer_removes_oldest_first(self, detector):
        """Oldest inputs are removed first."""
        for i in range(15):
            detector.buffer_input(f"input_{i}")

        buffer = detector.get_buffer()
        # Should have inputs 5-14 (oldest removed)
        names = [b.input_name for b in buffer]
        assert "input_0" not in names
        assert "input_14" in names

    def test_clear_buffer(self, detector):
        """clear_buffer empties the buffer."""
        detector.buffer_input("punch")
        detector.buffer_input("kick")

        detector.clear_buffer()

        assert len(detector.get_buffer()) == 0

    def test_buffer_stores_timestamp(self, detector):
        """Buffer entries have timestamps."""
        import time

        before = time.time()
        detector.buffer_input("punch")
        after = time.time()

        buffer = detector.get_buffer()
        assert before <= buffer[0].timestamp <= after

    def test_buffer_stores_value(self, detector):
        """Buffer entries store input value."""
        detector.buffer_input("trigger", value=0.75)

        buffer = detector.get_buffer()
        assert buffer[0].value == 0.75

    def test_old_buffer_entries_cleaned(self, detector):
        """Old buffer entries are cleaned on process."""
        import time

        detector.buffer_input("old_input")
        time.sleep(0.6)  # Wait beyond lifetime
        detector.process_input("new_input")

        buffer = detector.get_buffer()
        names = [b.input_name for b in buffer]
        assert "old_input" not in names
        assert "new_input" in names


# =============================================================================
# Combo Cancellation Tests
# =============================================================================

class TestComboCancellation:
    """Tests for combo cancellation."""

    @pytest.fixture
    def detector(self):
        """Create a combo detector with combos."""
        det = ComboDetector()
        det.register_combo(ComboDefinition(
            name="cancelable",
            sequence=[
                ComboInput(ComboInputType.BUTTON, ["a"]),
                ComboInput(ComboInputType.BUTTON, ["b"]),
                ComboInput(ComboInputType.BUTTON, ["c"]),
            ],
            cancelable=True
        ))
        det.register_combo(ComboDefinition(
            name="uncancelable",
            sequence=[
                ComboInput(ComboInputType.BUTTON, ["x"]),
                ComboInput(ComboInputType.BUTTON, ["y"]),
            ],
            cancelable=False
        ))
        return det

    def test_cancel_in_progress_combo(self, detector):
        """Can cancel an in-progress combo."""
        detector.process_input("a")
        assert detector.get_combo_state("cancelable") == ComboState.IN_PROGRESS

        result = detector.cancel_combo("cancelable")
        assert result is True
        assert detector.get_combo_state("cancelable") == ComboState.IDLE

    def test_cannot_cancel_uncancelable(self, detector):
        """Cannot cancel uncancelable combo."""
        detector.process_input("x")
        assert detector.get_combo_state("uncancelable") == ComboState.IN_PROGRESS

        result = detector.cancel_combo("uncancelable")
        assert result is False
        assert detector.get_combo_state("uncancelable") == ComboState.IN_PROGRESS

    def test_cancel_idle_combo_fails(self, detector):
        """Cannot cancel idle combo."""
        result = detector.cancel_combo("cancelable")
        assert result is False

    def test_cancel_nonexistent_combo(self, detector):
        """Canceling nonexistent combo fails."""
        result = detector.cancel_combo("nonexistent")
        assert result is False

    def test_cancel_all_combos(self, detector):
        """cancel_all_combos cancels all in-progress."""
        detector.process_input("a")
        detector.process_input("x")

        count = detector.cancel_all_combos()

        assert count == 1  # Only cancelable one
        assert detector.get_combo_state("cancelable") == ComboState.IDLE


# =============================================================================
# Overlapping Combos Tests
# =============================================================================

class TestOverlappingCombos:
    """Tests for overlapping combo sequences."""

    @pytest.fixture
    def detector(self):
        """Create a combo detector with overlapping combos."""
        det = ComboDetector()

        # Combos share "a" as first input
        det.register_combo(ComboDefinition(
            name="short",
            sequence=[
                ComboInput(ComboInputType.BUTTON, ["a"]),
                ComboInput(ComboInputType.BUTTON, ["b"]),
            ],
            priority=0
        ))
        det.register_combo(ComboDefinition(
            name="long",
            sequence=[
                ComboInput(ComboInputType.BUTTON, ["a"]),
                ComboInput(ComboInputType.BUTTON, ["b"]),
                ComboInput(ComboInputType.BUTTON, ["c"]),
            ],
            priority=10
        ))
        return det

    def test_both_combos_start(self, detector):
        """Both overlapping combos start together."""
        results = detector.process_input("a")

        # Both should be in progress
        assert len(results) == 2
        for r in results:
            assert r.state == ComboState.IN_PROGRESS

    def test_shorter_combo_completes_first(self, detector):
        """Shorter combo completes before longer."""
        detector.process_input("a")
        results = detector.process_input("b")

        short_results = [r for r in results if r.combo_name == "short"]
        assert len(short_results) == 1
        assert short_results[0].state == ComboState.COMPLETED

    def test_results_sorted_by_priority(self, detector):
        """Results are sorted by priority."""
        detector.process_input("a")
        results = detector.process_input("b")

        # Higher priority first
        if len(results) >= 2:
            priorities = [detector.get_combo(r.combo_name).priority for r in results]
            assert priorities == sorted(priorities, reverse=True)

    def test_divergent_paths(self):
        """Combos with divergent paths after common start."""
        detector = ComboDetector()

        detector.register_combo(ComboDefinition(
            name="path_b",
            sequence=[
                ComboInput(ComboInputType.BUTTON, ["a"]),
                ComboInput(ComboInputType.BUTTON, ["b"]),
            ]
        ))
        detector.register_combo(ComboDefinition(
            name="path_c",
            sequence=[
                ComboInput(ComboInputType.BUTTON, ["a"]),
                ComboInput(ComboInputType.BUTTON, ["c"]),
            ]
        ))

        # Start both
        detector.process_input("a")

        # Diverge - only path_b continues
        results = detector.process_input("b")

        b_results = [r for r in results if r.combo_name == "path_b"]
        assert len(b_results) == 1
        assert b_results[0].state == ComboState.COMPLETED


# =============================================================================
# Combo Priority Tests
# =============================================================================

class TestComboPriority:
    """Tests for combo priority handling."""

    @pytest.fixture
    def detector(self):
        """Create a combo detector with prioritized combos."""
        det = ComboDetector()

        det.register_combo(ComboDefinition(
            name="low",
            sequence=[
                ComboInput(ComboInputType.BUTTON, ["a"]),
                ComboInput(ComboInputType.BUTTON, ["b"]),
            ],
            priority=0
        ))
        det.register_combo(ComboDefinition(
            name="high",
            sequence=[
                ComboInput(ComboInputType.BUTTON, ["a"]),
                ComboInput(ComboInputType.BUTTON, ["b"]),
            ],
            priority=100
        ))
        return det

    def test_high_priority_listed_first(self, detector):
        """Higher priority combos listed first in results."""
        detector.process_input("a")
        results = detector.process_input("b")

        assert results[0].combo_name == "high"

    def test_same_sequence_both_complete(self, detector):
        """Identical sequences both complete."""
        detector.process_input("a")
        results = detector.process_input("b")

        completed = [r for r in results if r.state == ComboState.COMPLETED]
        assert len(completed) == 2


# =============================================================================
# Failed Combo Tests
# =============================================================================

class TestFailedCombo:
    """Tests for failed combo handling."""

    @pytest.fixture
    def detector(self):
        """Create a combo detector."""
        det = ComboDetector()
        det.register_combo(ComboDefinition(
            name="strict",
            sequence=[
                ComboInput(ComboInputType.BUTTON, ["a"]),
                ComboInput(ComboInputType.BUTTON, ["b"]),
            ],
            window=0.1
        ))
        return det

    def test_timeout_fails_combo(self, detector):
        """Combo fails on timeout."""
        import time

        detector.process_input("a")
        time.sleep(0.15)
        results = detector.update(0.0)

        assert any(r.state == ComboState.FAILED for r in results)

    def test_failed_combo_resets(self, detector):
        """Failed combo resets to idle."""
        import time

        detector.process_input("a")
        time.sleep(0.15)
        detector.update(0.0)

        assert detector.get_combo_state("strict") == ComboState.IDLE

    def test_can_restart_after_failure(self, detector):
        """Can restart combo after failure."""
        import time

        detector.process_input("a")
        time.sleep(0.15)
        detector.update(0.0)

        # Restart
        results = detector.process_input("a")
        assert any(r.state == ComboState.IN_PROGRESS for r in results)


# =============================================================================
# Direction + Button Combo Tests
# =============================================================================

class TestDirectionButtonCombos:
    """Tests for combos with directions and buttons."""

    @pytest.fixture
    def detector(self):
        """Create a combo detector with fighting game moves."""
        det = ComboDetector()

        # Fireball: down, down-right, right + punch
        det.register_combo(ComboDefinition(
            name="fireball",
            sequence=[
                ComboInput(ComboInputType.DIRECTION, ["down"]),
                ComboInput(ComboInputType.DIRECTION, ["down-right"]),
                ComboInput(ComboInputType.DIRECTION, ["right"]),
                ComboInput(ComboInputType.BUTTON, ["punch"]),
            ],
            window=0.5
        ))

        # Dragon punch: right, down, down-right + punch
        det.register_combo(ComboDefinition(
            name="dragon_punch",
            sequence=[
                ComboInput(ComboInputType.DIRECTION, ["right"]),
                ComboInput(ComboInputType.DIRECTION, ["down"]),
                ComboInput(ComboInputType.DIRECTION, ["down-right"]),
                ComboInput(ComboInputType.BUTTON, ["punch"]),
            ],
            window=0.5
        ))

        return det

    def test_fireball_sequence(self, detector):
        """Fireball combo completes with correct sequence."""
        detector.process_input("down")
        detector.process_input("down-right")
        detector.process_input("right")
        results = detector.process_input("punch")

        fb_results = [r for r in results if r.combo_name == "fireball"]
        assert len(fb_results) == 1
        assert fb_results[0].state == ComboState.COMPLETED

    def test_dragon_punch_sequence(self, detector):
        """Dragon punch combo completes with correct sequence."""
        detector.process_input("right")
        detector.process_input("down")
        detector.process_input("down-right")
        results = detector.process_input("punch")

        dp_results = [r for r in results if r.combo_name == "dragon_punch"]
        assert len(dp_results) == 1
        assert dp_results[0].state == ComboState.COMPLETED

    def test_wrong_direction_fails(self, detector):
        """Wrong direction doesn't advance combo."""
        detector.process_input("down")
        detector.process_input("left")  # Wrong!

        # Should not advance fireball
        progress = detector.get_combo_progress("fireball")
        assert progress < 0.5  # Still at first step


# =============================================================================
# Callback Tests
# =============================================================================

class TestComboCallbacks:
    """Tests for combo completion callbacks."""

    @pytest.fixture
    def detector(self):
        """Create a combo detector."""
        det = ComboDetector()
        det.register_combo(ComboDefinition(
            name="test",
            sequence=[
                ComboInput(ComboInputType.BUTTON, ["a"]),
                ComboInput(ComboInputType.BUTTON, ["b"]),
            ]
        ))
        return det

    def test_bind_callback(self, detector):
        """bind_callback binds to combo."""
        callback = Mock()
        result = detector.bind_callback("test", callback)
        assert result is True

    def test_bind_callback_nonexistent(self, detector):
        """bind_callback to nonexistent combo fails."""
        callback = Mock()
        result = detector.bind_callback("nonexistent", callback)
        assert result is False

    def test_callback_invoked_on_completion(self, detector):
        """Callback is invoked when combo completes."""
        callback = Mock()
        detector.bind_callback("test", callback)

        detector.process_input("a")
        detector.process_input("b")

        callback.assert_called_once()
        result = callback.call_args[0][0]
        assert result.state == ComboState.COMPLETED

    def test_unbind_callback(self, detector):
        """unbind_callback removes callback."""
        callback = Mock()
        detector.bind_callback("test", callback)
        result = detector.unbind_callback("test", callback)
        assert result is True

        detector.process_input("a")
        detector.process_input("b")

        callback.assert_not_called()

    def test_callback_exception_handled(self, detector):
        """Callback exception doesn't break detector."""
        def bad_callback(result):
            raise ValueError("Test error")

        detector.bind_callback("test", bad_callback)

        # Should not raise
        detector.process_input("a")
        detector.process_input("b")


# =============================================================================
# Reset Tests
# =============================================================================

class TestComboReset:
    """Tests for combo detector reset."""

    @pytest.fixture
    def detector(self):
        """Create a combo detector with state."""
        det = ComboDetector()
        det.register_combo(ComboDefinition(
            name="test",
            sequence=[
                ComboInput(ComboInputType.BUTTON, ["a"]),
                ComboInput(ComboInputType.BUTTON, ["b"]),
            ]
        ))
        det.process_input("a")
        det.buffer_input("x")
        det.buffer_input("y")
        return det

    def test_reset_clears_combo_state(self, detector):
        """reset clears combo progress."""
        detector.reset()

        assert detector.get_combo_state("test") == ComboState.IDLE
        assert detector.get_combo_progress("test") == 0.0

    def test_reset_clears_buffer(self, detector):
        """reset clears input buffer."""
        detector.reset()

        assert len(detector.get_buffer()) == 0

    def test_can_start_combo_after_reset(self, detector):
        """Can start combo after reset."""
        detector.reset()

        results = detector.process_input("a")
        assert any(r.state == ComboState.IN_PROGRESS for r in results)


# =============================================================================
# Integration Tests
# =============================================================================

class TestComboIntegration:
    """Integration tests for combo system."""

    def test_fighting_game_scenario(self):
        """Simulate a fighting game combo scenario."""
        detector = ComboDetector()

        # Register moves
        detector.register_combo(ComboDefinition(
            name="jab",
            sequence=[ComboInput(ComboInputType.BUTTON, ["light_punch"])],
            priority=0
        ))
        detector.register_combo(ComboDefinition(
            name="fireball",
            sequence=[
                ComboInput(ComboInputType.DIRECTION, ["down"]),
                ComboInput(ComboInputType.DIRECTION, ["down-forward"]),
                ComboInput(ComboInputType.DIRECTION, ["forward"]),
                ComboInput(ComboInputType.BUTTON, ["punch"]),
            ],
            priority=50
        ))
        detector.register_combo(ComboDefinition(
            name="super_fireball",
            sequence=[
                ComboInput(ComboInputType.DIRECTION, ["down"]),
                ComboInput(ComboInputType.DIRECTION, ["down-forward"]),
                ComboInput(ComboInputType.DIRECTION, ["forward"]),
                ComboInput(ComboInputType.DIRECTION, ["down"]),
                ComboInput(ComboInputType.DIRECTION, ["down-forward"]),
                ComboInput(ComboInputType.DIRECTION, ["forward"]),
                ComboInput(ComboInputType.BUTTON, ["punch"]),
            ],
            priority=100
        ))

        # Player does fireball motion
        detector.process_input("down")
        detector.process_input("down-forward")
        detector.process_input("forward")
        results = detector.process_input("punch")

        # Should complete fireball
        completed = [r for r in results if r.state == ComboState.COMPLETED]
        assert any(r.combo_name == "fireball" for r in completed)

    def test_rhythm_game_scenario(self):
        """Simulate a rhythm game combo scenario."""
        detector = ComboDetector()

        # Register beat patterns
        detector.register_combo(ComboDefinition(
            name="basic_beat",
            sequence=[
                ComboInput(ComboInputType.BUTTON, ["hit"]),
                ComboInput(ComboInputType.BUTTON, ["hit"]),
            ],
            window=0.5
        ))
        detector.register_combo(ComboDefinition(
            name="complex_beat",
            sequence=[
                ComboInput(ComboInputType.BUTTON, ["hit"]),
                ComboInput(ComboInputType.BUTTON, ["hit"]),
                ComboInput(ComboInputType.BUTTON, ["hold"]),
                ComboInput(ComboInputType.BUTTON, ["hit"]),
            ],
            window=0.3
        ))

        completed_combos = []
        detector.bind_callback("basic_beat", lambda r: completed_combos.append(r.combo_name))
        detector.bind_callback("complex_beat", lambda r: completed_combos.append(r.combo_name))

        # Player hits the basic beat
        detector.process_input("hit")
        detector.process_input("hit")

        assert "basic_beat" in completed_combos
