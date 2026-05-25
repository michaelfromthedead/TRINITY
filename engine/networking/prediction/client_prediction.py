"""
Client-side prediction system for responsive networked gameplay.

This module implements the client-side prediction pattern where:
1. Client applies inputs immediately for responsiveness
2. Inputs are buffered with sequence numbers
3. Server state is compared against predictions
4. Mispredictions trigger reconciliation

The system maintains smooth gameplay even under high latency conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import math

from engine.networking.config import (
    DEFAULT_INPUT_BUFFER_SIZE,
    DEFAULT_PREDICTION_HISTORY_SIZE,
    DEFAULT_DELTA_TIME,
    DEFAULT_MOVE_SPEED,
    DEFAULT_FRICTION,
    DEFAULT_JUMP_VELOCITY,
    DEFAULT_GRAVITY,
    GROUND_CHECK_TOLERANCE,
    MISPREDICTION_THRESHOLD,
)


# Type aliases for clarity
Vector3 = Tuple[float, float, float]
InputData = Dict[str, Any]


@dataclass
class BufferedInput:
    """A single input entry stored in the input buffer."""

    sequence_num: int
    """Unique sequence number for this input."""

    input_data: InputData
    """The actual input data (keys pressed, mouse delta, etc.)."""

    timestamp: float = 0.0
    """Local timestamp when input was recorded."""

    predicted_state: Optional[PredictionState] = None
    """The predicted state after applying this input (for comparison)."""


class InputBuffer:
    """
    Buffer for storing unconfirmed inputs pending server acknowledgment.

    Inputs are stored with sequence numbers and can be retrieved for
    replay during server reconciliation. The buffer automatically
    discards confirmed inputs when the server acknowledges them.

    Attributes:
        max_size: Maximum number of inputs to buffer (prevents memory bloat).

    Example:
        buffer = InputBuffer(max_size=64)
        buffer.push(1, {"forward": True, "delta_time": 0.016})
        buffer.push(2, {"forward": True, "right": True, "delta_time": 0.016})

        # Server confirms up to sequence 1
        buffer.pop_confirmed(1)

        # Get remaining unconfirmed inputs for replay
        unconfirmed = buffer.get_unconfirmed()
    """

    def __init__(self, max_size: int = DEFAULT_INPUT_BUFFER_SIZE) -> None:
        """
        Initialize the input buffer.

        Args:
            max_size: Maximum number of inputs to store. Oldest inputs
                     are discarded if buffer exceeds this size.
        """
        self._buffer: deque[BufferedInput] = deque(maxlen=max_size)
        self._max_size = max_size
        self._last_confirmed_seq: int = -1

    @property
    def max_size(self) -> int:
        """Maximum buffer size."""
        return self._max_size

    @property
    def size(self) -> int:
        """Current number of buffered inputs."""
        return len(self._buffer)

    @property
    def last_confirmed_sequence(self) -> int:
        """The last sequence number confirmed by the server."""
        return self._last_confirmed_seq

    def push(
        self,
        sequence_num: int,
        input_data: InputData,
        timestamp: float = 0.0,
        predicted_state: Optional[PredictionState] = None,
    ) -> None:
        """
        Add an input to the buffer.

        Args:
            sequence_num: Unique sequence number for this input.
            input_data: Dictionary containing input state.
            timestamp: Optional local timestamp for this input.
            predicted_state: Optional predicted state after this input.
        """
        entry = BufferedInput(
            sequence_num=sequence_num,
            input_data=input_data,
            timestamp=timestamp,
            predicted_state=predicted_state,
        )
        self._buffer.append(entry)

    def pop_confirmed(self, last_confirmed_seq: int) -> List[BufferedInput]:
        """
        Remove all inputs up to and including the confirmed sequence.

        Args:
            last_confirmed_seq: The sequence number that the server has
                               confirmed processing.

        Returns:
            List of removed (confirmed) inputs.
        """
        if last_confirmed_seq <= self._last_confirmed_seq:
            return []

        self._last_confirmed_seq = last_confirmed_seq
        removed: List[BufferedInput] = []

        while self._buffer and self._buffer[0].sequence_num <= last_confirmed_seq:
            removed.append(self._buffer.popleft())

        return removed

    def get_unconfirmed(self) -> List[BufferedInput]:
        """
        Get all unconfirmed inputs for replay.

        Returns:
            List of inputs that haven't been confirmed by the server,
            in chronological order.
        """
        return list(self._buffer)

    def get_input_at_sequence(self, sequence_num: int) -> Optional[BufferedInput]:
        """
        Get a specific input by sequence number.

        Args:
            sequence_num: The sequence number to look up.

        Returns:
            The buffered input if found, None otherwise.
        """
        for entry in self._buffer:
            if entry.sequence_num == sequence_num:
                return entry
        return None

    def get_inputs_after_sequence(self, sequence_num: int) -> List[BufferedInput]:
        """
        Get all inputs after a specific sequence number.

        Args:
            sequence_num: The sequence number to start from (exclusive).

        Returns:
            List of inputs with sequence numbers greater than the given one.
        """
        return [
            entry for entry in self._buffer
            if entry.sequence_num > sequence_num
        ]

    def clear(self) -> None:
        """Clear all buffered inputs."""
        self._buffer.clear()
        self._last_confirmed_seq = -1

    def is_empty(self) -> bool:
        """Check if the buffer is empty."""
        return len(self._buffer) == 0


@dataclass
class PredictionState:
    """
    Represents the predicted state of an entity.

    This state is computed locally based on input predictions and can
    be compared against authoritative server state for reconciliation.

    Attributes:
        predicted_position: The predicted position as (x, y, z).
        predicted_velocity: The predicted velocity as (vx, vy, vz).
        predicted_rotation: Optional predicted rotation as (x, y, z, w) quaternion.
        sequence_num: The input sequence this prediction corresponds to.
        timestamp: When this prediction was made.
        custom_data: Additional state data for game-specific predictions.
    """

    predicted_position: Vector3 = field(default_factory=lambda: (0.0, 0.0, 0.0))
    predicted_velocity: Vector3 = field(default_factory=lambda: (0.0, 0.0, 0.0))
    predicted_rotation: Optional[Tuple[float, float, float, float]] = None
    sequence_num: int = 0
    timestamp: float = 0.0
    custom_data: Dict[str, Any] = field(default_factory=dict)

    def apply_input(
        self,
        input_data: InputData,
        delta_time: float = DEFAULT_DELTA_TIME,
        move_speed: float = DEFAULT_MOVE_SPEED,
        friction: float = DEFAULT_FRICTION,
    ) -> PredictionState:
        """
        Apply input to produce a new predicted state.

        This is a simplified physics simulation. Real implementations
        should match the server's physics exactly.

        Args:
            input_data: Dictionary with movement keys and parameters.
            delta_time: Time step for physics integration.
            move_speed: Movement acceleration factor.
            friction: Velocity damping factor (0-1).

        Returns:
            New PredictionState with updated position and velocity.
        """
        # Extract movement input
        forward = 1.0 if input_data.get("forward", False) else 0.0
        backward = 1.0 if input_data.get("backward", False) else 0.0
        left = 1.0 if input_data.get("left", False) else 0.0
        right = 1.0 if input_data.get("right", False) else 0.0

        # Compute movement direction
        move_x = (right - left) * move_speed * delta_time
        move_z = (forward - backward) * move_speed * delta_time

        # Update velocity with friction
        new_vx = (self.predicted_velocity[0] + move_x) * friction
        new_vy = self.predicted_velocity[1]  # Y velocity (gravity, jumping, etc.)
        new_vz = (self.predicted_velocity[2] + move_z) * friction

        # Handle jumping
        if input_data.get("jump", False) and abs(self.predicted_position[1]) < GROUND_CHECK_TOLERANCE:
            new_vy = input_data.get("jump_velocity", DEFAULT_JUMP_VELOCITY)

        # Apply gravity
        gravity = input_data.get("gravity", DEFAULT_GRAVITY)
        new_vy += gravity * delta_time

        # Update position
        new_x = self.predicted_position[0] + new_vx * delta_time
        new_y = max(0.0, self.predicted_position[1] + new_vy * delta_time)
        new_z = self.predicted_position[2] + new_vz * delta_time

        # Ground check
        if new_y <= 0.0:
            new_y = 0.0
            new_vy = 0.0

        # Get delta_time from input if provided
        dt = input_data.get("delta_time", delta_time)

        return PredictionState(
            predicted_position=(new_x, new_y, new_z),
            predicted_velocity=(new_vx, new_vy, new_vz),
            predicted_rotation=self.predicted_rotation,
            sequence_num=self.sequence_num + 1,
            timestamp=self.timestamp + dt,
            custom_data=dict(self.custom_data),
        )

    def clone(self) -> PredictionState:
        """
        Create a deep copy of this state for rollback.

        Returns:
            New PredictionState with copied values.
        """
        return PredictionState(
            predicted_position=self.predicted_position,
            predicted_velocity=self.predicted_velocity,
            predicted_rotation=self.predicted_rotation,
            sequence_num=self.sequence_num,
            timestamp=self.timestamp,
            custom_data=dict(self.custom_data),
        )

    def distance_to(self, other: PredictionState) -> float:
        """
        Calculate the Euclidean distance to another state.

        Args:
            other: The state to compare against.

        Returns:
            Distance between positions.
        """
        dx = self.predicted_position[0] - other.predicted_position[0]
        dy = self.predicted_position[1] - other.predicted_position[1]
        dz = self.predicted_position[2] - other.predicted_position[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def velocity_difference(self, other: PredictionState) -> float:
        """
        Calculate velocity difference magnitude.

        Args:
            other: The state to compare against.

        Returns:
            Magnitude of velocity difference.
        """
        dvx = self.predicted_velocity[0] - other.predicted_velocity[0]
        dvy = self.predicted_velocity[1] - other.predicted_velocity[1]
        dvz = self.predicted_velocity[2] - other.predicted_velocity[2]
        return math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz)


class ClientPredictor:
    """
    Main client-side prediction system.

    Coordinates input buffering, state prediction, and error detection
    for smooth client-side gameplay with server reconciliation.

    The predictor maintains:
    - Current predicted state
    - Input buffer for replay
    - History of predictions for comparison

    Example:
        predictor = ClientPredictor()

        # Each frame:
        input_data = get_player_input()
        predictor.predict(input_data)  # Immediate local update
        predictor.store_input(current_seq, input_data)
        send_to_server(current_seq, input_data)

        # When server state arrives:
        error = predictor.get_prediction_error(server_state, server_seq)
        if error > threshold:
            # Trigger reconciliation
            pass
    """

    def __init__(
        self,
        initial_state: Optional[PredictionState] = None,
        buffer_size: int = DEFAULT_INPUT_BUFFER_SIZE,
        prediction_history_size: int = DEFAULT_PREDICTION_HISTORY_SIZE,
    ) -> None:
        """
        Initialize the client predictor.

        Args:
            initial_state: Starting state, or creates default if None.
            buffer_size: Maximum inputs to buffer.
            prediction_history_size: Number of prediction states to keep.
        """
        self._current_state = initial_state or PredictionState()
        self._input_buffer = InputBuffer(max_size=buffer_size)
        self._prediction_history: deque[PredictionState] = deque(
            maxlen=prediction_history_size
        )
        self._current_sequence: int = 0
        self._last_server_sequence: int = -1
        self._total_predictions: int = 0
        self._mispredictions: int = 0

    @property
    def current_state(self) -> PredictionState:
        """Get the current predicted state."""
        return self._current_state

    @property
    def input_buffer(self) -> InputBuffer:
        """Access the input buffer directly."""
        return self._input_buffer

    @property
    def current_sequence(self) -> int:
        """Get the current input sequence number."""
        return self._current_sequence

    @property
    def prediction_accuracy(self) -> float:
        """
        Get the prediction accuracy ratio.

        Returns:
            Ratio of correct predictions (0.0 to 1.0).
        """
        if self._total_predictions == 0:
            return 1.0
        return 1.0 - (self._mispredictions / self._total_predictions)

    def predict(
        self,
        input_data: InputData,
        delta_time: float = DEFAULT_DELTA_TIME,
    ) -> PredictionState:
        """
        Apply input and predict the next state.

        This should be called each frame with the player's input.
        The result is applied immediately for responsiveness.

        Args:
            input_data: Current frame's input state.
            delta_time: Frame time for physics.

        Returns:
            The new predicted state after applying input.
        """
        # Store current state in history
        self._prediction_history.append(self._current_state.clone())

        # Apply input to get new predicted state
        input_with_dt = dict(input_data)
        input_with_dt["delta_time"] = delta_time

        self._current_state = self._current_state.apply_input(
            input_with_dt,
            delta_time=delta_time,
        )
        self._current_state.sequence_num = self._current_sequence

        return self._current_state

    def store_input(
        self,
        sequence_num: int,
        input_data: InputData,
        timestamp: float = 0.0,
    ) -> None:
        """
        Store an input in the buffer for potential replay.

        Args:
            sequence_num: Sequence number for this input.
            input_data: The input data to store.
            timestamp: Optional timestamp for this input.
        """
        self._input_buffer.push(
            sequence_num=sequence_num,
            input_data=input_data,
            timestamp=timestamp,
            predicted_state=self._current_state.clone(),
        )
        self._current_sequence = sequence_num + 1

    def get_prediction_error(
        self,
        server_state: PredictionState,
        server_sequence: int,
    ) -> float:
        """
        Calculate the error between predicted and server state.

        Args:
            server_state: The authoritative state from the server.
            server_sequence: The sequence number this state corresponds to.

        Returns:
            Error magnitude (position distance).
        """
        self._total_predictions += 1

        # Find our predicted state at this sequence
        buffered = self._input_buffer.get_input_at_sequence(server_sequence)

        if buffered and buffered.predicted_state:
            predicted = buffered.predicted_state
        else:
            # Fall back to looking in history
            predicted = self._find_state_at_sequence(server_sequence)
            if predicted is None:
                # No prediction to compare - use current
                predicted = self._current_state

        error = predicted.distance_to(server_state)

        if error > MISPREDICTION_THRESHOLD:
            self._mispredictions += 1

        self._last_server_sequence = server_sequence

        return error

    def _find_state_at_sequence(self, sequence: int) -> Optional[PredictionState]:
        """Find a prediction state by sequence number."""
        for state in self._prediction_history:
            if state.sequence_num == sequence:
                return state
        return None

    def acknowledge_inputs(self, last_confirmed_seq: int) -> None:
        """
        Mark inputs as confirmed by the server.

        Args:
            last_confirmed_seq: The last sequence the server processed.
        """
        self._input_buffer.pop_confirmed(last_confirmed_seq)

    def get_unconfirmed_inputs(self) -> List[BufferedInput]:
        """
        Get all inputs not yet confirmed by the server.

        Returns:
            List of unconfirmed inputs for replay.
        """
        return self._input_buffer.get_unconfirmed()

    def set_state(self, state: PredictionState) -> None:
        """
        Set the current state directly (e.g., after reconciliation).

        Args:
            state: The new state to use.
        """
        self._current_state = state.clone()

    def reset(self, initial_state: Optional[PredictionState] = None) -> None:
        """
        Reset the predictor to initial state.

        Args:
            initial_state: Optional new initial state.
        """
        self._current_state = initial_state or PredictionState()
        self._input_buffer.clear()
        self._prediction_history.clear()
        self._current_sequence = 0
        self._last_server_sequence = -1
        self._total_predictions = 0
        self._mispredictions = 0
