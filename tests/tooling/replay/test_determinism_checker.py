"""
Tests for determinism_checker.py - State comparison and drift detection.
"""

import pytest

from engine.tooling.replay.determinism_checker import (
    DeterminismChecker,
    DeterminismResult,
    DriftReport,
    DriftSeverity,
    StateComparisonConfig,
)


class TestDriftSeverity:
    """Tests for DriftSeverity enum."""

    def test_severity_order(self):
        """Test severity ordering."""
        assert DriftSeverity.NONE.value < DriftSeverity.MINOR.value
        assert DriftSeverity.MINOR.value < DriftSeverity.MODERATE.value
        assert DriftSeverity.MODERATE.value < DriftSeverity.MAJOR.value
        assert DriftSeverity.MAJOR.value < DriftSeverity.CRITICAL.value


class TestDriftReport:
    """Tests for DriftReport dataclass."""

    def test_create_drift_report(self):
        """Test creating a drift report."""
        report = DriftReport(
            frame=100,
            timestamp=1.67,
            path='player.position.x',
            expected_value=10.0,
            actual_value=10.5,
            severity=DriftSeverity.MINOR,
            difference=0.5
        )
        assert report.frame == 100
        assert report.path == 'player.position.x'
        assert report.severity == DriftSeverity.MINOR
        assert report.difference == 0.5

    def test_drift_report_str(self):
        """Test drift report string representation."""
        report = DriftReport(
            frame=50,
            timestamp=0.83,
            path='health',
            expected_value=100,
            actual_value=99,
            severity=DriftSeverity.MODERATE
        )
        string = str(report)

        assert "Frame 50" in string
        assert "MODERATE" in string
        assert "health" in string


class TestDeterminismResult:
    """Tests for DeterminismResult dataclass."""

    def test_create_result(self):
        """Test creating a result."""
        result = DeterminismResult(
            is_deterministic=True,
            total_frames_checked=1000,
            total_drifts=0
        )
        assert result.is_deterministic
        assert result.total_frames_checked == 1000
        assert result.total_drifts == 0

    def test_add_drift(self):
        """Test adding drift to result."""
        result = DeterminismResult(
            is_deterministic=True,
            total_frames_checked=100,
            total_drifts=0
        )

        report = DriftReport(
            frame=50,
            timestamp=0.83,
            path='value',
            expected_value=1,
            actual_value=2,
            severity=DriftSeverity.MINOR
        )
        result.add_drift(report)

        assert result.total_drifts == 1
        assert result.first_drift_frame == 50
        assert result.last_drift_frame == 50
        assert DriftSeverity.MINOR in result.drifts_by_severity

    def test_major_drift_sets_not_deterministic(self):
        """Test that major drift marks result as non-deterministic."""
        result = DeterminismResult(
            is_deterministic=True,
            total_frames_checked=100,
            total_drifts=0
        )

        report = DriftReport(
            frame=25,
            timestamp=0.42,
            path='position',
            expected_value=0.0,
            actual_value=100.0,
            severity=DriftSeverity.MAJOR,
            difference=100.0
        )
        result.add_drift(report)

        assert not result.is_deterministic

    def test_get_drifts_at_frame(self):
        """Test getting drifts at specific frame."""
        result = DeterminismResult(
            is_deterministic=True,
            total_frames_checked=100,
            total_drifts=0
        )

        # Add drifts at different frames
        for frame in [10, 10, 20]:
            result.add_drift(DriftReport(
                frame=frame,
                timestamp=frame * 0.016,
                path='value',
                expected_value=0,
                actual_value=1,
                severity=DriftSeverity.MINOR
            ))

        drifts_at_10 = result.get_drifts_at_frame(10)
        assert len(drifts_at_10) == 2

    def test_get_drifts_by_path(self):
        """Test getting drifts by path."""
        result = DeterminismResult(
            is_deterministic=True,
            total_frames_checked=100,
            total_drifts=0
        )

        result.add_drift(DriftReport(
            frame=1, timestamp=0.0, path='a.b.c',
            expected_value=0, actual_value=1, severity=DriftSeverity.MINOR
        ))
        result.add_drift(DriftReport(
            frame=2, timestamp=0.0, path='x.y.z',
            expected_value=0, actual_value=1, severity=DriftSeverity.MINOR
        ))
        result.add_drift(DriftReport(
            frame=3, timestamp=0.0, path='a.b.c',
            expected_value=0, actual_value=2, severity=DriftSeverity.MINOR
        ))

        abc_drifts = result.get_drifts_by_path('a.b.c')
        assert len(abc_drifts) == 2

    def test_summary(self):
        """Test summary generation."""
        result = DeterminismResult(
            is_deterministic=True,
            total_frames_checked=100,
            total_drifts=0
        )
        summary = result.summary()

        assert "PASS" in summary
        assert "100" in summary


class TestStateComparisonConfig:
    """Tests for StateComparisonConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = StateComparisonConfig()
        assert config.float_tolerance == 1e-6
        assert config.max_drifts == 1000
        assert config.stop_on_critical is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = StateComparisonConfig(
            float_tolerance=0.001,
            position_tolerance=0.01,
            ignored_paths={'debug', 'temp'}
        )
        assert config.float_tolerance == 0.001
        assert 'debug' in config.ignored_paths


class TestDeterminismChecker:
    """Tests for DeterminismChecker class."""

    def test_create_checker(self):
        """Test creating a checker."""
        checker = DeterminismChecker()
        assert not checker.is_checking
        assert checker.is_deterministic

    def test_start_stop(self):
        """Test starting and stopping check."""
        checker = DeterminismChecker()
        checker.start()
        assert checker.is_checking

        result = checker.stop()
        assert not checker.is_checking
        assert isinstance(result, DeterminismResult)

    def test_check_identical_states(self):
        """Test checking identical states."""
        checker = DeterminismChecker()
        checker.start()

        expected = {'player': {'x': 10.0, 'y': 20.0}, 'score': 100}
        actual = {'player': {'x': 10.0, 'y': 20.0}, 'score': 100}

        drifts = checker.check_frame(0, expected, actual)
        assert len(drifts) == 0

        result = checker.stop()
        assert result.is_deterministic

    def test_check_different_values(self):
        """Test checking different values."""
        checker = DeterminismChecker()
        checker.start()

        expected = {'value': 100}
        actual = {'value': 200}

        drifts = checker.check_frame(0, expected, actual)
        assert len(drifts) > 0

        checker.stop()

    def test_float_tolerance(self):
        """Test float comparison with tolerance."""
        config = StateComparisonConfig(float_tolerance=0.01)
        checker = DeterminismChecker(config)
        checker.start()

        expected = {'x': 10.0}
        actual = {'x': 10.001}  # Within tolerance

        drifts = checker.check_frame(0, expected, actual)
        assert len(drifts) == 0

        checker.stop()

    def test_float_beyond_tolerance(self):
        """Test float comparison beyond tolerance."""
        config = StateComparisonConfig(float_tolerance=0.001)
        checker = DeterminismChecker(config)
        checker.start()

        expected = {'x': 10.0}
        actual = {'x': 10.1}  # Beyond tolerance

        drifts = checker.check_frame(0, expected, actual)
        assert len(drifts) > 0

        checker.stop()

    def test_ignored_paths(self):
        """Test ignoring specific paths."""
        config = StateComparisonConfig(ignored_paths={'debug', 'temp.value'})
        checker = DeterminismChecker(config)
        checker.start()

        expected = {'value': 1, 'debug': 'old', 'temp': {'value': 0}}
        actual = {'value': 1, 'debug': 'new', 'temp': {'value': 100}}

        drifts = checker.check_frame(0, expected, actual)
        # Only ignored paths differ, so no drifts
        assert len(drifts) == 0

        checker.stop()

    def test_nested_dict_comparison(self):
        """Test nested dictionary comparison."""
        checker = DeterminismChecker()
        checker.start()

        expected = {
            'player': {
                'position': {'x': 10, 'y': 20},
                'stats': {'health': 100}
            }
        }
        actual = {
            'player': {
                'position': {'x': 10, 'y': 25},  # Different
                'stats': {'health': 100}
            }
        }

        drifts = checker.check_frame(0, expected, actual)
        assert len(drifts) > 0
        assert any('y' in d.path for d in drifts)

        checker.stop()

    def test_list_comparison(self):
        """Test list/sequence comparison."""
        checker = DeterminismChecker()
        checker.start()

        expected = {'items': [1, 2, 3]}
        actual = {'items': [1, 2, 4]}  # Different element

        drifts = checker.check_frame(0, expected, actual)
        assert len(drifts) > 0

        checker.stop()

    def test_list_length_mismatch(self):
        """Test list length mismatch detection."""
        checker = DeterminismChecker()
        checker.start()

        expected = {'items': [1, 2, 3]}
        actual = {'items': [1, 2]}  # Different length

        drifts = checker.check_frame(0, expected, actual)
        assert len(drifts) > 0
        assert any('__len__' in d.path for d in drifts)

        checker.stop()

    def test_missing_key_detection(self):
        """Test missing key detection."""
        checker = DeterminismChecker()
        checker.start()

        expected = {'a': 1, 'b': 2}
        actual = {'a': 1}  # Missing 'b'

        drifts = checker.check_frame(0, expected, actual)
        assert len(drifts) > 0

        checker.stop()

    def test_added_key_detection(self):
        """Test added key detection."""
        checker = DeterminismChecker()
        checker.start()

        expected = {'a': 1}
        actual = {'a': 1, 'b': 2}  # Extra 'b'

        drifts = checker.check_frame(0, expected, actual)
        assert len(drifts) > 0

        checker.stop()

    def test_type_mismatch(self):
        """Test type mismatch detection."""
        checker = DeterminismChecker()
        checker.start()

        expected = {'value': 10}
        actual = {'value': '10'}  # String instead of int

        drifts = checker.check_frame(0, expected, actual)
        assert len(drifts) > 0
        assert any(d.severity == DriftSeverity.CRITICAL for d in drifts)

        checker.stop()

    def test_compare_full_replay(self):
        """Test comparing full replay sequences."""
        checker = DeterminismChecker()

        expected_states = [
            (0, 0.0, {'x': 0}),
            (1, 0.016, {'x': 1}),
            (2, 0.032, {'x': 2}),
        ]
        actual_states = [
            (0, 0.0, {'x': 0}),
            (1, 0.016, {'x': 1}),
            (2, 0.032, {'x': 2}),
        ]

        result = checker.compare_full_replay(expected_states, actual_states)
        assert result.is_deterministic
        assert result.total_frames_checked == 3

    def test_compare_full_replay_with_drift(self):
        """Test comparing full replay with drift."""
        checker = DeterminismChecker()

        expected_states = [
            (0, 0.0, {'x': 0}),
            (1, 0.016, {'x': 1}),
            (2, 0.032, {'x': 2}),
        ]
        actual_states = [
            (0, 0.0, {'x': 0}),
            (1, 0.016, {'x': 100}),  # Drift here
            (2, 0.032, {'x': 2}),
        ]

        result = checker.compare_full_replay(expected_states, actual_states)
        assert result.total_drifts > 0

    def test_compute_state_hash(self):
        """Test state hash computation."""
        checker = DeterminismChecker()

        state1 = {'a': 1, 'b': 2}
        state2 = {'b': 2, 'a': 1}  # Same content, different order
        state3 = {'a': 1, 'b': 3}  # Different content

        hash1 = checker.compute_state_hash(state1)
        hash2 = checker.compute_state_hash(state2)
        hash3 = checker.compute_state_hash(state3)

        assert hash1 == hash2  # Same content = same hash
        assert hash1 != hash3  # Different content = different hash

    def test_get_differing_paths(self):
        """Test getting list of differing paths."""
        checker = DeterminismChecker()

        state1 = {'a': 1, 'b': {'c': 2, 'd': 3}}
        state2 = {'a': 1, 'b': {'c': 5, 'd': 3}}  # 'b.c' differs

        paths = checker.get_differing_paths(state1, state2)
        assert 'b.c' in paths

    def test_iter_drifts(self):
        """Test iterating over drifts."""
        checker = DeterminismChecker()
        checker.start()

        checker.check_frame(0, {'a': 1}, {'a': 2})
        checker.check_frame(1, {'b': 1}, {'b': 2})

        drifts = list(checker.iter_drifts())
        assert len(drifts) >= 2

        checker.stop()

    def test_custom_comparator(self):
        """Test custom comparator function."""
        def custom_compare(expected, actual):
            # Always report as MINOR drift
            if expected != actual:
                return DriftReport(
                    frame=0,
                    timestamp=0.0,
                    path='',
                    expected_value=expected,
                    actual_value=actual,
                    severity=DriftSeverity.MINOR,
                    description="Custom comparison"
                )
            return None

        config = StateComparisonConfig(
            custom_comparators={'special': custom_compare}
        )
        checker = DeterminismChecker(config)
        checker.start()

        drifts = checker.check_frame(
            0,
            {'special': 100},
            {'special': 200}
        )
        assert any("Custom comparison" in d.description for d in drifts)

        checker.stop()

    def test_max_drifts_limit(self):
        """Test maximum drifts limit."""
        config = StateComparisonConfig(max_drifts=5)
        checker = DeterminismChecker(config)

        expected_states = [(i, i * 0.016, {'x': 0}) for i in range(100)]
        actual_states = [(i, i * 0.016, {'x': i}) for i in range(100)]  # All different

        result = checker.compare_full_replay(expected_states, actual_states)
        assert result.total_drifts <= 5

    def test_stop_on_critical(self):
        """Test stopping on critical drift."""
        config = StateComparisonConfig(stop_on_critical=True)
        checker = DeterminismChecker(config)

        # Create states with type mismatch (critical)
        expected_states = [
            (0, 0.0, {'x': 1}),
            (1, 0.016, {'x': 2}),  # Will have critical drift
            (2, 0.032, {'x': 3}),
        ]
        actual_states = [
            (0, 0.0, {'x': 1}),
            (1, 0.016, {'x': 'string'}),  # Type mismatch = critical
            (2, 0.032, {'x': 3}),
        ]

        result = checker.compare_full_replay(expected_states, actual_states)
        # Should stop early after critical drift
        assert result.drifts_by_severity.get(DriftSeverity.CRITICAL, 0) > 0

    def test_verification_hash(self):
        """Test verification hash in result."""
        checker = DeterminismChecker()
        checker.start()

        checker.add_actual_state(0, {'x': 1})
        checker.add_actual_state(1, {'x': 2})

        result = checker.stop()
        assert len(result.verification_hash) == 64  # SHA-256 hex
