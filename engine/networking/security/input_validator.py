"""
Input validation system for network security.

This module validates player inputs to detect cheating attempts such as
speed hacks, teleportation, and impossible movements.

Thread-safety: This module is NOT thread-safe by default. Use external
synchronization when accessing from multiple threads.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
import math
import threading
import time

from engine.networking.security.config import (
    INPUT_VALIDATION,
    VALIDATION_LIMITS,
)


class ValidationResult(Enum):
    """Result of input validation."""
    VALID = auto()
    INVALID_SPEED = auto()
    INVALID_POSITION = auto()
    INVALID_ROTATION = auto()
    INVALID_SEQUENCE = auto()
    INVALID_TELEPORT = auto()
    INVALID_ACTION_RATE = auto()
    INVALID_BOUNDS = auto()


@dataclass
class Vector3:
    """Simple 3D vector for position and movement calculations."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __sub__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __add__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def magnitude(self) -> float:
        """Calculate the magnitude (length) of the vector."""
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def magnitude_2d(self) -> float:
        """Calculate the 2D magnitude (ignoring Y/height)."""
        return math.sqrt(self.x ** 2 + self.z ** 2)


@dataclass
class InputBounds:
    """
    Configuration for input validation bounds.

    All defaults are loaded from security config to avoid magic numbers.

    Attributes:
        max_speed: Maximum allowed speed in units per second
        max_rotation_rate: Maximum rotation rate in degrees per second
        max_action_rate: Maximum actions per second
        world_min: Minimum world coordinates
        world_max: Maximum world coordinates
        max_teleport_distance: Maximum allowed position change per tick
        tolerance_multiplier: Multiplier for tolerances (for lag compensation)
    """
    max_speed: float = INPUT_VALIDATION.MAX_SPEED
    max_rotation_rate: float = INPUT_VALIDATION.MAX_ROTATION_RATE
    max_action_rate: float = INPUT_VALIDATION.MAX_ACTION_RATE
    world_min: Vector3 = field(default_factory=lambda: Vector3(
        INPUT_VALIDATION.WORLD_MIN_X,
        INPUT_VALIDATION.WORLD_MIN_Y,
        INPUT_VALIDATION.WORLD_MIN_Z
    ))
    world_max: Vector3 = field(default_factory=lambda: Vector3(
        INPUT_VALIDATION.WORLD_MAX_X,
        INPUT_VALIDATION.WORLD_MAX_Y,
        INPUT_VALIDATION.WORLD_MAX_Z
    ))
    max_teleport_distance: float = INPUT_VALIDATION.MAX_TELEPORT_DISTANCE
    tolerance_multiplier: float = INPUT_VALIDATION.TOLERANCE_MULTIPLIER


@dataclass
class PlayerState:
    """Tracks player state for validation."""
    player_id: str
    position: Vector3 = field(default_factory=Vector3)
    rotation: float = 0.0  # Degrees
    last_update_time: float = 0.0
    last_action_times: Dict[str, float] = field(default_factory=dict)
    sequence_number: int = 0
    violation_count: int = 0


@dataclass
class ValidationReport:
    """Detailed report of validation result."""
    result: ValidationResult
    player_id: str
    details: str = ""
    expected_value: Optional[float] = None
    actual_value: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


class InputValidator:
    """
    Validates player inputs against bounds and sanity checks.

    This class detects various cheating methods including:
    - Speed hacks (moving faster than allowed)
    - Teleportation (instant position changes)
    - Invalid rotations
    - Sequence manipulation
    - Action rate abuse

    Thread-safety: Uses internal locking for concurrent access.
    """

    def __init__(self, bounds: Optional[InputBounds] = None):
        """
        Initialize the input validator.

        Args:
            bounds: Input validation bounds configuration
        """
        self._bounds = bounds or InputBounds()
        self._player_states: Dict[str, PlayerState] = {}
        self._lock = threading.RLock()

    @property
    def bounds(self) -> InputBounds:
        """Get the current bounds configuration."""
        return self._bounds

    def set_bounds(self, bounds: InputBounds) -> None:
        """Update the bounds configuration."""
        self._bounds = bounds

    def get_player_state(self, player_id: str) -> PlayerState:
        """
        Get or create player state.

        Args:
            player_id: The player's unique identifier

        Returns:
            The player's current state

        Raises:
            ValueError: If player_id is empty or exceeds safe length
        """
        # Input validation on the validator itself
        if not player_id or not isinstance(player_id, str):
            raise ValueError("player_id must be a non-empty string")
        if len(player_id) > 256:
            raise ValueError("player_id exceeds maximum length of 256 characters")

        with self._lock:
            # Prevent memory exhaustion
            if len(self._player_states) >= VALIDATION_LIMITS.MAX_PLAYER_STATE_ENTRIES:
                if player_id not in self._player_states:
                    raise RuntimeError(
                        f"Maximum player state entries ({VALIDATION_LIMITS.MAX_PLAYER_STATE_ENTRIES}) exceeded"
                    )

            if player_id not in self._player_states:
                self._player_states[player_id] = PlayerState(
                    player_id=player_id,
                    last_update_time=time.time()
                )
            return self._player_states[player_id]

    def set_player_position(self, player_id: str, position: Vector3) -> None:
        """
        Set a player's position (for initialization or teleports).

        Args:
            player_id: The player's unique identifier
            position: The new position
        """
        with self._lock:
            state = self.get_player_state(player_id)
            state.position = position
            state.last_update_time = time.time()

    def remove_player(self, player_id: str) -> None:
        """Remove a player from tracking."""
        with self._lock:
            self._player_states.pop(player_id, None)

    def validate_movement(
        self,
        player_id: str,
        new_position: Vector3,
        time_delta: Optional[float] = None
    ) -> ValidationReport:
        """
        Validate a movement input.

        Args:
            player_id: The player's unique identifier
            new_position: The requested new position
            time_delta: Time since last update (auto-calculated if None)

        Returns:
            ValidationReport with the result
        """
        with self._lock:
            state = self.get_player_state(player_id)
            current_time = time.time()

            if time_delta is None:
                time_delta = current_time - state.last_update_time

            # Minimum time delta to prevent division issues
            time_delta = max(time_delta, INPUT_VALIDATION.MIN_TIME_DELTA)

            # Calculate position delta
            position_delta = new_position - state.position
            distance = position_delta.magnitude()
            speed = distance / time_delta

            # Check for teleportation (single large jump)
            if distance > self._bounds.max_teleport_distance * self._bounds.tolerance_multiplier:
                self._increment_violation_count(state)
                return ValidationReport(
                    result=ValidationResult.INVALID_TELEPORT,
                    player_id=player_id,
                    details=f"Position jump of {distance:.2f} exceeds max teleport distance",
                    expected_value=self._bounds.max_teleport_distance,
                    actual_value=distance
                )

            # Check speed
            max_allowed_speed = self._bounds.max_speed * self._bounds.tolerance_multiplier
            if speed > max_allowed_speed:
                self._increment_violation_count(state)
                return ValidationReport(
                    result=ValidationResult.INVALID_SPEED,
                    player_id=player_id,
                    details=f"Speed of {speed:.2f} exceeds maximum {max_allowed_speed:.2f}",
                    expected_value=max_allowed_speed,
                    actual_value=speed
                )

            # Check world bounds
            if not self._is_within_bounds(new_position):
                self._increment_violation_count(state)
                return ValidationReport(
                    result=ValidationResult.INVALID_BOUNDS,
                    player_id=player_id,
                    details=f"Position {new_position} is outside world bounds"
                )

            # Valid movement - update state
            state.position = new_position
            state.last_update_time = current_time

            return ValidationReport(
                result=ValidationResult.VALID,
                player_id=player_id,
                details=f"Movement validated: speed={speed:.2f}, distance={distance:.2f}"
            )

    def _increment_violation_count(self, state: PlayerState) -> None:
        """Safely increment violation count with overflow protection."""
        if state.violation_count < VALIDATION_LIMITS.MAX_VIOLATION_COUNT:
            state.violation_count += 1

    def validate_rotation(
        self,
        player_id: str,
        new_rotation: float,
        time_delta: Optional[float] = None
    ) -> ValidationReport:
        """
        Validate a rotation input.

        Args:
            player_id: The player's unique identifier
            new_rotation: The requested new rotation in degrees
            time_delta: Time since last update (auto-calculated if None)

        Returns:
            ValidationReport with the result
        """
        with self._lock:
            state = self.get_player_state(player_id)
            current_time = time.time()

            if time_delta is None:
                time_delta = current_time - state.last_update_time

            # Minimum time delta
            time_delta = max(time_delta, INPUT_VALIDATION.MIN_TIME_DELTA)

            # Calculate rotation delta (handle wraparound)
            rotation_delta = abs(new_rotation - state.rotation)
            if rotation_delta > 180:
                rotation_delta = 360 - rotation_delta

            rotation_rate = rotation_delta / time_delta

            # Check rotation rate
            max_rotation_rate = self._bounds.max_rotation_rate * self._bounds.tolerance_multiplier
            if rotation_rate > max_rotation_rate:
                self._increment_violation_count(state)
                return ValidationReport(
                    result=ValidationResult.INVALID_ROTATION,
                    player_id=player_id,
                    details=f"Rotation rate of {rotation_rate:.2f} deg/s exceeds maximum",
                    expected_value=max_rotation_rate,
                    actual_value=rotation_rate
                )

            # Valid rotation - update state
            state.rotation = new_rotation
            state.last_update_time = current_time

            return ValidationReport(
                result=ValidationResult.VALID,
                player_id=player_id,
                details=f"Rotation validated: rate={rotation_rate:.2f} deg/s"
            )

    def validate_action(
        self,
        player_id: str,
        action_type: str,
        current_time: Optional[float] = None
    ) -> ValidationReport:
        """
        Validate an action based on rate limiting.

        Args:
            player_id: The player's unique identifier
            action_type: The type of action being performed
            current_time: Current timestamp (auto-calculated if None)

        Returns:
            ValidationReport with the result

        Raises:
            ValueError: If action_type is empty or invalid
        """
        # Validate action_type input
        if not action_type or not isinstance(action_type, str):
            raise ValueError("action_type must be a non-empty string")
        if len(action_type) > 128:
            raise ValueError("action_type exceeds maximum length of 128 characters")

        with self._lock:
            state = self.get_player_state(player_id)

            if current_time is None:
                current_time = time.time()

            last_action_time = state.last_action_times.get(action_type, 0.0)
            time_since_last = current_time - last_action_time

            # Calculate minimum time between actions
            min_action_interval = 1.0 / self._bounds.max_action_rate

            if time_since_last < min_action_interval / self._bounds.tolerance_multiplier:
                self._increment_violation_count(state)
                return ValidationReport(
                    result=ValidationResult.INVALID_ACTION_RATE,
                    player_id=player_id,
                    details=f"Action '{action_type}' performed too quickly ({time_since_last:.3f}s)",
                    expected_value=min_action_interval,
                    actual_value=time_since_last
                )

            # Valid action - update state
            state.last_action_times[action_type] = current_time

            return ValidationReport(
                result=ValidationResult.VALID,
                player_id=player_id,
                details=f"Action '{action_type}' validated"
            )

    def validate_sequence(
        self,
        player_id: str,
        sequence_number: int
    ) -> ValidationReport:
        """
        Validate input sequence number.

        Args:
            player_id: The player's unique identifier
            sequence_number: The sequence number of the input

        Returns:
            ValidationReport with the result

        Raises:
            ValueError: If sequence_number is negative or exceeds max
        """
        # Validate sequence_number to prevent integer overflow
        if not isinstance(sequence_number, int):
            raise ValueError("sequence_number must be an integer")
        if sequence_number < 0:
            raise ValueError("sequence_number must be non-negative")
        if sequence_number > VALIDATION_LIMITS.MAX_SEQUENCE_NUMBER:
            raise ValueError(
                f"sequence_number exceeds maximum ({VALIDATION_LIMITS.MAX_SEQUENCE_NUMBER})"
            )

        with self._lock:
            state = self.get_player_state(player_id)

            # Sequence should be incrementing
            expected_sequence = state.sequence_number + 1

            # Use config constant for sequence window
            sequence_window = INPUT_VALIDATION.SEQUENCE_WINDOW

            if sequence_number <= state.sequence_number - sequence_window:
                # Old/duplicate packet
                return ValidationReport(
                    result=ValidationResult.INVALID_SEQUENCE,
                    player_id=player_id,
                    details=f"Old sequence number {sequence_number} (current: {state.sequence_number})",
                    expected_value=float(expected_sequence),
                    actual_value=float(sequence_number)
                )

            if sequence_number > state.sequence_number + sequence_window:
                # Suspiciously high jump in sequence
                self._increment_violation_count(state)
                return ValidationReport(
                    result=ValidationResult.INVALID_SEQUENCE,
                    player_id=player_id,
                    details=f"Sequence jump from {state.sequence_number} to {sequence_number}",
                    expected_value=float(expected_sequence),
                    actual_value=float(sequence_number)
                )

            # Valid sequence - update state
            state.sequence_number = max(state.sequence_number, sequence_number)

            return ValidationReport(
                result=ValidationResult.VALID,
                player_id=player_id,
                details=f"Sequence {sequence_number} validated"
            )

    def validate_full_input(
        self,
        player_id: str,
        new_position: Vector3,
        new_rotation: float,
        sequence_number: int,
        time_delta: Optional[float] = None
    ) -> List[ValidationReport]:
        """
        Validate a complete input update.

        Args:
            player_id: The player's unique identifier
            new_position: Requested position
            new_rotation: Requested rotation
            sequence_number: Input sequence number
            time_delta: Time since last update

        Returns:
            List of ValidationReports for each check
        """
        reports = []

        # Validate sequence first
        seq_report = self.validate_sequence(player_id, sequence_number)
        reports.append(seq_report)

        # Only continue if sequence is valid
        if seq_report.result != ValidationResult.VALID:
            return reports

        # Validate movement
        move_report = self.validate_movement(player_id, new_position, time_delta)
        reports.append(move_report)

        # Validate rotation
        rot_report = self.validate_rotation(player_id, new_rotation, time_delta)
        reports.append(rot_report)

        return reports

    def _is_within_bounds(self, position: Vector3) -> bool:
        """Check if a position is within world bounds."""
        return (
            self._bounds.world_min.x <= position.x <= self._bounds.world_max.x and
            self._bounds.world_min.y <= position.y <= self._bounds.world_max.y and
            self._bounds.world_min.z <= position.z <= self._bounds.world_max.z
        )

    def get_violation_count(self, player_id: str) -> int:
        """Get the number of violations for a player."""
        with self._lock:
            state = self.get_player_state(player_id)
            return state.violation_count

    def reset_violations(self, player_id: str) -> None:
        """Reset violation count for a player."""
        with self._lock:
            if player_id in self._player_states:
                self._player_states[player_id].violation_count = 0

    def get_all_player_states(self) -> Dict[str, PlayerState]:
        """Get a copy of all player states (thread-safe snapshot)."""
        with self._lock:
            return dict(self._player_states)
