"""
Determinism Checker - Verify deterministic replay by comparing states.

Ensures that replaying recorded inputs produces identical game states,
detecting any non-deterministic behavior or drift.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, Iterator
import math


class DriftSeverity(Enum):
    """Severity levels for state drift."""
    NONE = auto()  # No drift detected
    MINOR = auto()  # Small floating point differences
    MODERATE = auto()  # Noticeable differences
    MAJOR = auto()  # Significant state divergence
    CRITICAL = auto()  # Complete desynchronization


@dataclass
class DriftReport:
    """Report of detected state drift."""
    frame: int
    timestamp: float
    path: str
    expected_value: Any
    actual_value: Any
    severity: DriftSeverity
    difference: Optional[float] = None  # For numeric values
    description: str = ""

    def __str__(self) -> str:
        return (
            f"Frame {self.frame} [{self.severity.name}]: {self.path}\n"
            f"  Expected: {self.expected_value}\n"
            f"  Actual: {self.actual_value}"
        )


@dataclass
class DeterminismResult:
    """Result of determinism verification."""
    is_deterministic: bool
    total_frames_checked: int
    total_drifts: int
    drifts_by_severity: dict[DriftSeverity, int] = field(default_factory=dict)
    drift_reports: list[DriftReport] = field(default_factory=list)
    first_drift_frame: Optional[int] = None
    last_drift_frame: Optional[int] = None
    max_numeric_drift: float = 0.0
    verification_hash: str = ""
    elapsed_time: float = 0.0

    def add_drift(self, report: DriftReport) -> None:
        """Add a drift report."""
        self.drift_reports.append(report)
        self.total_drifts += 1
        self.drifts_by_severity[report.severity] = (
            self.drifts_by_severity.get(report.severity, 0) + 1
        )

        if self.first_drift_frame is None:
            self.first_drift_frame = report.frame
        self.last_drift_frame = report.frame

        if report.difference is not None:
            self.max_numeric_drift = max(self.max_numeric_drift, abs(report.difference))

        # Update determinism status
        if report.severity in (DriftSeverity.MAJOR, DriftSeverity.CRITICAL):
            self.is_deterministic = False

    def get_drifts_at_frame(self, frame: int) -> list[DriftReport]:
        """Get all drifts at a specific frame."""
        return [d for d in self.drift_reports if d.frame == frame]

    def get_drifts_by_path(self, path: str) -> list[DriftReport]:
        """Get all drifts for a specific state path."""
        return [d for d in self.drift_reports if d.path == path]

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Determinism Check: {'PASS' if self.is_deterministic else 'FAIL'}",
            f"Frames Checked: {self.total_frames_checked}",
            f"Total Drifts: {self.total_drifts}",
        ]

        if self.total_drifts > 0:
            lines.append("Drifts by Severity:")
            for severity, count in sorted(self.drifts_by_severity.items(), key=lambda x: x[0].value):
                lines.append(f"  {severity.name}: {count}")
            lines.append(f"First Drift: Frame {self.first_drift_frame}")
            lines.append(f"Last Drift: Frame {self.last_drift_frame}")
            lines.append(f"Max Numeric Drift: {self.max_numeric_drift:.6f}")

        return "\n".join(lines)


@dataclass
class StateComparisonConfig:
    """Configuration for state comparison."""
    # Tolerance settings
    float_tolerance: float = 1e-6  # Absolute tolerance for floats
    float_relative_tolerance: float = 1e-5  # Relative tolerance
    position_tolerance: float = 0.001  # Tolerance for positions
    rotation_tolerance: float = 0.001  # Tolerance for rotations (radians)

    # Severity thresholds
    minor_drift_threshold: float = 0.01
    moderate_drift_threshold: float = 0.1
    major_drift_threshold: float = 1.0

    # Paths to ignore during comparison
    ignored_paths: set[str] = field(default_factory=set)

    # Paths with special tolerance
    path_tolerances: dict[str, float] = field(default_factory=dict)

    # Maximum drifts to collect before stopping
    max_drifts: int = 1000

    # Stop on first critical drift
    stop_on_critical: bool = True

    # Custom comparison functions for specific paths
    custom_comparators: dict[str, Callable[[Any, Any], Optional[DriftReport]]] = field(
        default_factory=dict
    )


class DeterminismChecker:
    """Verifies deterministic replay by comparing states.

    Compares expected states from recording with actual states from
    replay to detect non-deterministic behavior.
    """
    __slots__ = (
        '_config', '_expected_states', '_actual_states',
        '_result', '_is_checking', '_current_frame',
        '_state_hasher', '_comparison_cache'
    )

    def __init__(self, config: Optional[StateComparisonConfig] = None):
        """Initialize determinism checker.

        Args:
            config: Comparison configuration
        """
        self._config = config or StateComparisonConfig()
        self._expected_states: dict[int, dict[str, Any]] = {}
        self._actual_states: dict[int, dict[str, Any]] = {}
        self._result = DeterminismResult(is_deterministic=True, total_frames_checked=0, total_drifts=0)
        self._is_checking = False
        self._current_frame = 0
        self._state_hasher = hashlib.sha256()
        self._comparison_cache: dict[str, bool] = {}

    @property
    def is_checking(self) -> bool:
        """Check if verification is in progress."""
        return self._is_checking

    @property
    def result(self) -> DeterminismResult:
        """Get current verification result."""
        return self._result

    @property
    def is_deterministic(self) -> bool:
        """Check if replay is deterministic so far."""
        return self._result.is_deterministic

    def start(self) -> None:
        """Start determinism checking."""
        self._is_checking = True
        self._result = DeterminismResult(is_deterministic=True, total_frames_checked=0, total_drifts=0)
        self._expected_states.clear()
        self._actual_states.clear()
        self._current_frame = 0
        self._state_hasher = hashlib.sha256()
        self._comparison_cache.clear()

    def stop(self) -> DeterminismResult:
        """Stop checking and return results.

        Returns:
            Verification result
        """
        self._is_checking = False
        self._result.verification_hash = self._state_hasher.hexdigest()
        return self._result

    def add_expected_state(
        self,
        frame: int,
        state: dict[str, Any]
    ) -> None:
        """Add expected state from recording.

        Args:
            frame: Frame number
            state: Expected state dictionary
        """
        self._expected_states[frame] = copy.deepcopy(state)

    def add_actual_state(
        self,
        frame: int,
        state: dict[str, Any]
    ) -> list[DriftReport]:
        """Add actual state from replay and compare.

        Args:
            frame: Frame number
            state: Actual state dictionary

        Returns:
            List of drift reports for this frame
        """
        self._actual_states[frame] = copy.deepcopy(state)

        # Update hash
        state_json = json.dumps(state, sort_keys=True).encode('utf-8')
        self._state_hasher.update(state_json)

        # Compare with expected
        expected = self._expected_states.get(frame)
        if expected is None:
            return []

        return self._compare_states(frame, expected, state)

    def check_frame(
        self,
        frame: int,
        expected: dict[str, Any],
        actual: dict[str, Any],
        timestamp: float = 0.0
    ) -> list[DriftReport]:
        """Check a single frame for determinism.

        Args:
            frame: Frame number
            expected: Expected state
            actual: Actual state
            timestamp: Frame timestamp

        Returns:
            List of drift reports
        """
        self._result.total_frames_checked += 1
        self._current_frame = frame
        return self._compare_states(frame, expected, actual, timestamp)

    def compare_full_replay(
        self,
        expected_states: list[tuple[int, float, dict[str, Any]]],
        actual_states: list[tuple[int, float, dict[str, Any]]]
    ) -> DeterminismResult:
        """Compare full replay state sequences.

        Args:
            expected_states: List of (frame, timestamp, state) tuples
            actual_states: List of (frame, timestamp, state) tuples

        Returns:
            Complete verification result
        """
        import time
        start_time = time.perf_counter()

        self.start()

        # Create lookup by frame
        expected_by_frame = {frame: (ts, state) for frame, ts, state in expected_states}
        actual_by_frame = {frame: (ts, state) for frame, ts, state in actual_states}

        # Check all frames
        all_frames = sorted(set(expected_by_frame.keys()) | set(actual_by_frame.keys()))

        for frame in all_frames:
            if frame in expected_by_frame and frame in actual_by_frame:
                exp_ts, exp_state = expected_by_frame[frame]
                act_ts, act_state = actual_by_frame[frame]
                self.check_frame(frame, exp_state, act_state, exp_ts)

                # Check for early termination
                if (self._config.stop_on_critical and
                        self._result.drifts_by_severity.get(DriftSeverity.CRITICAL, 0) > 0):
                    break

                if self._result.total_drifts >= self._config.max_drifts:
                    break

        self._result.elapsed_time = time.perf_counter() - start_time
        return self.stop()

    def verify_snapshot_chain(
        self,
        snapshots: list['StateSnapshot']
    ) -> DeterminismResult:
        """Verify integrity of snapshot chain.

        Args:
            snapshots: List of state snapshots

        Returns:
            Verification result
        """
        from .state_recorder import StateSnapshot

        self.start()

        for i, snapshot in enumerate(snapshots):
            # Verify checksum
            if snapshot.checksum:
                computed = snapshot.compute_checksum()
                if computed != snapshot.checksum:
                    report = DriftReport(
                        frame=snapshot.frame,
                        timestamp=snapshot.timestamp,
                        path="__checksum__",
                        expected_value=snapshot.checksum,
                        actual_value=computed,
                        severity=DriftSeverity.CRITICAL,
                        description="Snapshot checksum mismatch"
                    )
                    self._result.add_drift(report)

            self._result.total_frames_checked += 1

        return self.stop()

    def compute_state_hash(self, state: dict[str, Any]) -> str:
        """Compute hash of state for comparison.

        Args:
            state: State dictionary

        Returns:
            SHA-256 hash
        """
        state_json = json.dumps(state, sort_keys=True).encode('utf-8')
        return hashlib.sha256(state_json).hexdigest()

    def get_differing_paths(
        self,
        state1: dict[str, Any],
        state2: dict[str, Any]
    ) -> list[str]:
        """Get list of paths that differ between states.

        Args:
            state1: First state
            state2: Second state

        Returns:
            List of differing paths
        """
        diffs = []
        self._find_differences(state1, state2, "", diffs)
        return diffs

    def iter_drifts(self) -> Iterator[DriftReport]:
        """Iterate over all drift reports.

        Yields:
            Drift reports in order
        """
        yield from self._result.drift_reports

    def _compare_states(
        self,
        frame: int,
        expected: dict[str, Any],
        actual: dict[str, Any],
        timestamp: float = 0.0
    ) -> list[DriftReport]:
        """Compare expected and actual states."""
        drifts = []
        self._compare_recursive(frame, timestamp, expected, actual, "", drifts)
        return drifts

    def _compare_recursive(
        self,
        frame: int,
        timestamp: float,
        expected: Any,
        actual: Any,
        path: str,
        drifts: list[DriftReport]
    ) -> None:
        """Recursively compare values."""
        # Check if path is ignored
        if path in self._config.ignored_paths:
            return

        # Check for custom comparator
        if path in self._config.custom_comparators:
            report = self._config.custom_comparators[path](expected, actual)
            if report:
                report.frame = frame
                report.timestamp = timestamp
                report.path = path
                drifts.append(report)
                self._result.add_drift(report)
            return

        # Handle None values
        if expected is None and actual is None:
            return
        if expected is None or actual is None:
            self._add_drift(frame, timestamp, path, expected, actual, drifts)
            return

        # Compare by type
        if isinstance(expected, dict) and isinstance(actual, dict):
            self._compare_dicts(frame, timestamp, expected, actual, path, drifts)
        elif isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
            self._compare_sequences(frame, timestamp, expected, actual, path, drifts)
        elif isinstance(expected, float) or isinstance(actual, float):
            self._compare_floats(frame, timestamp, expected, actual, path, drifts)
        elif expected != actual:
            self._add_drift(frame, timestamp, path, expected, actual, drifts)

    def _compare_dicts(
        self,
        frame: int,
        timestamp: float,
        expected: dict,
        actual: dict,
        path: str,
        drifts: list[DriftReport]
    ) -> None:
        """Compare dictionary values."""
        all_keys = set(expected.keys()) | set(actual.keys())

        for key in all_keys:
            key_path = f"{path}.{key}" if path else key

            if key not in expected:
                self._add_drift(frame, timestamp, key_path, None, actual[key], drifts,
                               description="Key added")
            elif key not in actual:
                self._add_drift(frame, timestamp, key_path, expected[key], None, drifts,
                               description="Key removed")
            else:
                self._compare_recursive(
                    frame, timestamp, expected[key], actual[key], key_path, drifts
                )

    def _compare_sequences(
        self,
        frame: int,
        timestamp: float,
        expected: list | tuple,
        actual: list | tuple,
        path: str,
        drifts: list[DriftReport]
    ) -> None:
        """Compare sequence values."""
        if len(expected) != len(actual):
            self._add_drift(
                frame, timestamp, f"{path}.__len__",
                len(expected), len(actual), drifts,
                description="Sequence length mismatch"
            )

        for i, (exp, act) in enumerate(zip(expected, actual)):
            self._compare_recursive(
                frame, timestamp, exp, act, f"{path}[{i}]", drifts
            )

    def _compare_floats(
        self,
        frame: int,
        timestamp: float,
        expected: float,
        actual: float,
        path: str,
        drifts: list[DriftReport]
    ) -> None:
        """Compare floating point values with tolerance."""
        try:
            exp_float = float(expected)
            act_float = float(actual)
        except (TypeError, ValueError):
            self._add_drift(frame, timestamp, path, expected, actual, drifts)
            return

        # Get tolerance for this path
        tolerance = self._config.path_tolerances.get(path, self._config.float_tolerance)

        # Check absolute difference
        diff = abs(exp_float - act_float)
        if diff <= tolerance:
            return

        # Check relative difference for larger values
        if exp_float != 0:
            rel_diff = diff / abs(exp_float)
            if rel_diff <= self._config.float_relative_tolerance:
                return

        # Determine severity
        severity = self._get_numeric_severity(diff)

        report = DriftReport(
            frame=frame,
            timestamp=timestamp,
            path=path,
            expected_value=exp_float,
            actual_value=act_float,
            severity=severity,
            difference=diff,
            description=f"Numeric drift: {diff:.6f}"
        )
        drifts.append(report)
        self._result.add_drift(report)

    def _add_drift(
        self,
        frame: int,
        timestamp: float,
        path: str,
        expected: Any,
        actual: Any,
        drifts: list[DriftReport],
        description: str = ""
    ) -> None:
        """Add a drift report."""
        # Determine severity
        severity = self._determine_severity(expected, actual)

        report = DriftReport(
            frame=frame,
            timestamp=timestamp,
            path=path,
            expected_value=expected,
            actual_value=actual,
            severity=severity,
            description=description
        )
        drifts.append(report)
        self._result.add_drift(report)

    def _determine_severity(self, expected: Any, actual: Any) -> DriftSeverity:
        """Determine severity of a value difference."""
        if expected is None or actual is None:
            return DriftSeverity.MAJOR

        # Type mismatch is critical
        if type(expected) != type(actual):
            return DriftSeverity.CRITICAL

        # For strings, any difference is major
        if isinstance(expected, str):
            return DriftSeverity.MAJOR

        # For booleans, any difference is moderate
        if isinstance(expected, bool):
            return DriftSeverity.MODERATE

        return DriftSeverity.MODERATE

    def _get_numeric_severity(self, difference: float) -> DriftSeverity:
        """Get severity for numeric drift."""
        if difference < self._config.minor_drift_threshold:
            return DriftSeverity.MINOR
        elif difference < self._config.moderate_drift_threshold:
            return DriftSeverity.MODERATE
        elif difference < self._config.major_drift_threshold:
            return DriftSeverity.MAJOR
        else:
            return DriftSeverity.CRITICAL

    def _find_differences(
        self,
        state1: Any,
        state2: Any,
        path: str,
        diffs: list[str]
    ) -> None:
        """Find all differing paths between states."""
        if isinstance(state1, dict) and isinstance(state2, dict):
            all_keys = set(state1.keys()) | set(state2.keys())
            for key in all_keys:
                key_path = f"{path}.{key}" if path else key
                if key not in state1 or key not in state2:
                    diffs.append(key_path)
                else:
                    self._find_differences(state1[key], state2[key], key_path, diffs)
        elif isinstance(state1, (list, tuple)) and isinstance(state2, (list, tuple)):
            if len(state1) != len(state2):
                diffs.append(path)
            else:
                for i, (v1, v2) in enumerate(zip(state1, state2)):
                    self._find_differences(v1, v2, f"{path}[{i}]", diffs)
        elif state1 != state2:
            diffs.append(path)
