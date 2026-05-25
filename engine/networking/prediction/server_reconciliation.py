"""
Server reconciliation system for correcting client mispredictions.

When the authoritative server state diverges from client predictions,
this module handles:
1. Detecting mismatches between predicted and server state
2. Rolling back to the server's authoritative state
3. Replaying buffered inputs to catch up
4. Applying corrections smoothly to hide network artifacts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple
import math

from engine.networking.prediction.client_prediction import (
    BufferedInput,
    InputBuffer,
    PredictionState,
)
from engine.networking.config import (
    DEFAULT_RECONCILIATION_SNAP_THRESHOLD,
    DEFAULT_MATCH_THRESHOLD,
    DEFAULT_MAX_RECONCILE_FRAMES,
    DEFAULT_VELOCITY_WEIGHT,
    DEFAULT_ROTATION_WEIGHT,
    DEFAULT_RECONCILIATION_HISTORY_SIZE,
)


# Type aliases
Vector3 = Tuple[float, float, float]


class ReconciliationResult(Enum):
    """
    Result of comparing predicted state to server state.

    Used to determine what correction action to take.
    """

    MATCH = auto()
    """Prediction matches server state within tolerance."""

    MISMATCH_SMALL = auto()
    """Small error - can be smoothly interpolated."""

    MISMATCH_LARGE = auto()
    """Large error - requires snap or aggressive correction."""

    ERROR = auto()
    """Comparison failed (missing data, invalid state, etc.)."""


@dataclass
class ReconciliationConfig:
    """Configuration for the reconciliation system."""

    snap_threshold: float = DEFAULT_RECONCILIATION_SNAP_THRESHOLD
    """Position error threshold for snapping vs interpolating (units)."""

    match_threshold: float = DEFAULT_MATCH_THRESHOLD
    """Error below this is considered a match (units)."""

    max_reconcile_frames: int = DEFAULT_MAX_RECONCILE_FRAMES
    """Maximum frames of inputs to replay during reconciliation."""

    velocity_weight: float = DEFAULT_VELOCITY_WEIGHT
    """Weight for velocity error in comparison (0-1)."""

    rotation_weight: float = DEFAULT_ROTATION_WEIGHT
    """Weight for rotation error in comparison (0-1)."""


@dataclass
class ReconciliationStats:
    """Statistics about reconciliation operations."""

    total_reconciliations: int = 0
    """Total number of reconciliations performed."""

    snap_corrections: int = 0
    """Number of times we had to snap (large error)."""

    smooth_corrections: int = 0
    """Number of times we could smooth (small error)."""

    matches: int = 0
    """Number of times prediction matched server."""

    total_error: float = 0.0
    """Cumulative error for averaging."""

    max_error: float = 0.0
    """Maximum single error observed."""

    @property
    def average_error(self) -> float:
        """Average error per reconciliation."""
        if self.total_reconciliations == 0:
            return 0.0
        return self.total_error / self.total_reconciliations

    @property
    def snap_ratio(self) -> float:
        """Ratio of snaps to total corrections."""
        total = self.snap_corrections + self.smooth_corrections
        if total == 0:
            return 0.0
        return self.snap_corrections / total


class ServerReconciler:
    """
    Handles reconciliation between predicted and authoritative server state.

    The reconciliation process:
    1. Compare predicted state at sequence N with server state at N
    2. If mismatch detected, rollback to server state
    3. Replay all inputs after sequence N
    4. Apply smoothing to hide the correction

    Example:
        reconciler = ServerReconciler()

        # When server state arrives:
        result = reconciler.compare_states(predicted, server_state)

        if result == ReconciliationResult.MISMATCH_LARGE:
            reconciler.rollback_to_server_state(server_state)
            corrected = reconciler.replay_inputs(input_buffer)
            # Apply corrected state
    """

    def __init__(
        self,
        config: Optional[ReconciliationConfig] = None,
        physics_callback: Optional[Callable[[PredictionState, Dict], PredictionState]] = None,
    ) -> None:
        """
        Initialize the reconciler.

        Args:
            config: Reconciliation configuration.
            physics_callback: Optional custom physics function for replay.
                            Signature: (state, input_data) -> new_state
        """
        self._config = config or ReconciliationConfig()
        self._physics_callback = physics_callback
        self._stats = ReconciliationStats()
        self._last_server_state: Optional[PredictionState] = None
        self._correction_in_progress: bool = False

    @property
    def snap_threshold(self) -> float:
        """Position error threshold for snapping vs interpolating."""
        return self._config.snap_threshold

    @snap_threshold.setter
    def snap_threshold(self, value: float) -> None:
        """Set snap threshold."""
        self._config.snap_threshold = value

    @property
    def max_reconcile_frames(self) -> int:
        """Maximum frames of inputs to replay."""
        return self._config.max_reconcile_frames

    @max_reconcile_frames.setter
    def max_reconcile_frames(self, value: int) -> None:
        """Set max reconcile frames."""
        self._config.max_reconcile_frames = value

    @property
    def stats(self) -> ReconciliationStats:
        """Get reconciliation statistics."""
        return self._stats

    @property
    def is_correcting(self) -> bool:
        """Whether a correction is in progress."""
        return self._correction_in_progress

    def compare_states(
        self,
        predicted: PredictionState,
        authoritative: PredictionState,
    ) -> ReconciliationResult:
        """
        Compare predicted state against authoritative server state.

        Args:
            predicted: The client's predicted state.
            authoritative: The server's authoritative state.

        Returns:
            ReconciliationResult indicating the comparison outcome.
        """
        if predicted is None or authoritative is None:
            return ReconciliationResult.ERROR

        # Calculate position error
        position_error = predicted.distance_to(authoritative)

        # Calculate velocity error
        velocity_error = predicted.velocity_difference(authoritative)

        # Weighted total error
        total_error = (
            position_error +
            velocity_error * self._config.velocity_weight
        )

        # Add rotation error if available
        if predicted.predicted_rotation and authoritative.predicted_rotation:
            rotation_error = self._quaternion_angle_diff(
                predicted.predicted_rotation,
                authoritative.predicted_rotation,
            )
            total_error += rotation_error * self._config.rotation_weight

        # Update stats
        self._stats.total_reconciliations += 1
        self._stats.total_error += position_error
        self._stats.max_error = max(self._stats.max_error, position_error)

        # Classify result
        if position_error <= self._config.match_threshold:
            self._stats.matches += 1
            return ReconciliationResult.MATCH
        elif position_error <= self._config.snap_threshold:
            self._stats.smooth_corrections += 1
            return ReconciliationResult.MISMATCH_SMALL
        else:
            self._stats.snap_corrections += 1
            return ReconciliationResult.MISMATCH_LARGE

    def _quaternion_angle_diff(
        self,
        q1: Tuple[float, float, float, float],
        q2: Tuple[float, float, float, float],
    ) -> float:
        """Calculate angle difference between quaternions in radians."""
        # Compute dot product
        dot = q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]

        # Clamp to valid range for acos
        dot = max(-1.0, min(1.0, abs(dot)))

        # Angle is 2 * acos(|dot|)
        return 2.0 * math.acos(dot)

    def rollback_to_server_state(
        self,
        server_state: PredictionState,
    ) -> PredictionState:
        """
        Rollback to the authoritative server state.

        This should be called when a mismatch is detected before
        replaying inputs.

        Args:
            server_state: The authoritative state to rollback to.

        Returns:
            A copy of the server state to use as the new base.
        """
        self._last_server_state = server_state.clone()
        self._correction_in_progress = True
        return self._last_server_state

    def replay_inputs(
        self,
        input_buffer: InputBuffer,
        from_state: Optional[PredictionState] = None,
    ) -> PredictionState:
        """
        Replay buffered inputs from the server state to catch up.

        Args:
            input_buffer: Buffer containing unconfirmed inputs.
            from_state: Optional starting state (uses last server state if None).

        Returns:
            The corrected state after replaying inputs.
        """
        state = from_state or self._last_server_state
        if state is None:
            raise ValueError("No server state available for replay")

        state = state.clone()
        unconfirmed = input_buffer.get_unconfirmed()

        # Limit replay frames
        inputs_to_replay = unconfirmed[:self._config.max_reconcile_frames]

        for buffered_input in inputs_to_replay:
            if self._physics_callback:
                state = self._physics_callback(state, buffered_input.input_data)
            else:
                state = state.apply_input(buffered_input.input_data)
            state.sequence_num = buffered_input.sequence_num

        self._correction_in_progress = False
        return state

    def get_correction_vector(
        self,
        current: PredictionState,
        target: PredictionState,
    ) -> Vector3:
        """
        Calculate the correction vector from current to target position.

        Args:
            current: Current (incorrect) state.
            target: Target (corrected) state.

        Returns:
            Vector3 correction to apply.
        """
        return (
            target.predicted_position[0] - current.predicted_position[0],
            target.predicted_position[1] - current.predicted_position[1],
            target.predicted_position[2] - current.predicted_position[2],
        )

    def should_snap(self, error: float) -> bool:
        """
        Determine if correction should snap or interpolate.

        Args:
            error: The position error magnitude.

        Returns:
            True if should snap, False if can interpolate.
        """
        return error >= self._config.snap_threshold

    def reset_stats(self) -> None:
        """Reset reconciliation statistics."""
        self._stats = ReconciliationStats()


@dataclass
class ReconciliationFrame:
    """A single frame in the reconciliation history."""

    server_sequence: int
    """The server sequence this frame corresponds to."""

    server_state: PredictionState
    """The authoritative server state."""

    predicted_state: PredictionState
    """Our predicted state at this sequence."""

    error: float
    """The error between predicted and server."""

    result: ReconciliationResult
    """The reconciliation result."""

    replayed_inputs: int = 0
    """Number of inputs that were replayed."""


class ReconciliationHistory:
    """
    Maintains history of reconciliation events for debugging and analysis.

    This is useful for:
    - Debugging network issues
    - Tuning reconciliation parameters
    - Identifying systematic prediction errors
    """

    def __init__(self, max_frames: int = DEFAULT_RECONCILIATION_HISTORY_SIZE) -> None:
        """
        Initialize the history tracker.

        Args:
            max_frames: Maximum history frames to keep.
        """
        self._history: List[ReconciliationFrame] = []
        self._max_frames = max_frames

    def record(
        self,
        server_sequence: int,
        server_state: PredictionState,
        predicted_state: PredictionState,
        error: float,
        result: ReconciliationResult,
        replayed_inputs: int = 0,
    ) -> None:
        """Record a reconciliation event."""
        frame = ReconciliationFrame(
            server_sequence=server_sequence,
            server_state=server_state,
            predicted_state=predicted_state,
            error=error,
            result=result,
            replayed_inputs=replayed_inputs,
        )

        self._history.append(frame)

        # Trim old entries
        if len(self._history) > self._max_frames:
            self._history = self._history[-self._max_frames:]

    def get_recent(self, count: int = 10) -> List[ReconciliationFrame]:
        """Get the most recent reconciliation frames."""
        return self._history[-count:]

    def get_mismatches(self) -> List[ReconciliationFrame]:
        """Get all frames with mismatches."""
        return [
            f for f in self._history
            if f.result in (ReconciliationResult.MISMATCH_SMALL, ReconciliationResult.MISMATCH_LARGE)
        ]

    def clear(self) -> None:
        """Clear all history."""
        self._history.clear()
