"""
Tests for client-side prediction and reconciliation systems.

Tests cover:
- Input buffer operations
- Prediction state management
- Client predictor functionality
- Server reconciliation
- Entity interpolation
- Smoothing methods
"""

import pytest
import math
from engine.networking.prediction.client_prediction import (
    InputBuffer,
    PredictionState,
    ClientPredictor,
    BufferedInput,
)
from engine.networking.prediction.server_reconciliation import (
    ReconciliationResult,
    ServerReconciler,
    ReconciliationConfig,
    ReconciliationHistory,
)
from engine.networking.prediction.entity_interpolation import (
    Snapshot,
    InterpolationBuffer,
    InterpolationMode,
    lerp_position,
    slerp_rotation,
    hermite_interpolate,
    EntityInterpolator,
)
from engine.networking.prediction.smoothing import (
    SmoothingMethod,
    CorrectionSmoother,
    SmoothingConfig,
    smooth_position,
    smooth_rotation,
    exponential_smooth,
    exponential_smooth_vector,
    VisualSmoother,
)


class TestInputBuffer:
    """Tests for InputBuffer class."""

    def test_buffer_creation(self):
        """Test buffer initialization."""
        buffer = InputBuffer(max_size=32)
        assert buffer.max_size == 32
        assert buffer.size == 0
        assert buffer.is_empty()
        assert buffer.last_confirmed_sequence == -1

    def test_push_and_size(self):
        """Test pushing inputs to buffer."""
        buffer = InputBuffer(max_size=10)

        buffer.push(1, {"forward": True})
        assert buffer.size == 1
        assert not buffer.is_empty()

        buffer.push(2, {"forward": True, "right": True})
        assert buffer.size == 2

    def test_pop_confirmed(self):
        """Test removing confirmed inputs."""
        buffer = InputBuffer()

        # Add several inputs
        for i in range(5):
            buffer.push(i, {"seq": i})

        assert buffer.size == 5

        # Confirm up to sequence 2
        removed = buffer.pop_confirmed(2)
        assert len(removed) == 3  # 0, 1, 2
        assert buffer.size == 2  # 3, 4 remain
        assert buffer.last_confirmed_sequence == 2

    def test_get_unconfirmed(self):
        """Test getting unconfirmed inputs."""
        buffer = InputBuffer()

        for i in range(5):
            buffer.push(i, {"seq": i})

        buffer.pop_confirmed(2)
        unconfirmed = buffer.get_unconfirmed()

        assert len(unconfirmed) == 2
        assert unconfirmed[0].sequence_num == 3
        assert unconfirmed[1].sequence_num == 4

    def test_get_input_at_sequence(self):
        """Test looking up specific input."""
        buffer = InputBuffer()

        for i in range(5):
            buffer.push(i, {"value": i * 10})

        entry = buffer.get_input_at_sequence(3)
        assert entry is not None
        assert entry.sequence_num == 3
        assert entry.input_data["value"] == 30

        # Non-existent
        assert buffer.get_input_at_sequence(99) is None

    def test_get_inputs_after_sequence(self):
        """Test getting inputs after a sequence."""
        buffer = InputBuffer()

        for i in range(5):
            buffer.push(i, {"seq": i})

        after = buffer.get_inputs_after_sequence(2)
        assert len(after) == 2
        assert after[0].sequence_num == 3
        assert after[1].sequence_num == 4

    def test_max_size_enforcement(self):
        """Test that buffer respects max size."""
        buffer = InputBuffer(max_size=3)

        for i in range(10):
            buffer.push(i, {"seq": i})

        assert buffer.size == 3
        # Oldest should have been discarded
        assert buffer.get_input_at_sequence(0) is None
        assert buffer.get_input_at_sequence(7) is not None

    def test_clear(self):
        """Test clearing the buffer."""
        buffer = InputBuffer()

        for i in range(5):
            buffer.push(i, {})

        buffer.pop_confirmed(2)
        buffer.clear()

        assert buffer.is_empty()
        assert buffer.last_confirmed_sequence == -1


class TestPredictionState:
    """Tests for PredictionState class."""

    def test_default_state(self):
        """Test default state initialization."""
        state = PredictionState()
        assert state.predicted_position == (0.0, 0.0, 0.0)
        assert state.predicted_velocity == (0.0, 0.0, 0.0)
        assert state.sequence_num == 0

    def test_apply_input_forward(self):
        """Test applying forward movement input."""
        state = PredictionState()

        new_state = state.apply_input(
            {"forward": True},
            delta_time=0.1,
            move_speed=5.0,
        )

        # Should have moved forward (positive Z)
        assert new_state.predicted_position[2] > 0
        assert new_state.sequence_num == 1

    def test_apply_input_strafe(self):
        """Test applying strafe movement."""
        state = PredictionState()

        new_state = state.apply_input(
            {"right": True},
            delta_time=0.1,
            move_speed=5.0,
        )

        # Should have moved right (positive X)
        assert new_state.predicted_position[0] > 0

    def test_clone(self):
        """Test cloning state."""
        state = PredictionState(
            predicted_position=(1.0, 2.0, 3.0),
            predicted_velocity=(0.1, 0.2, 0.3),
            sequence_num=5,
        )

        cloned = state.clone()

        assert cloned.predicted_position == state.predicted_position
        assert cloned.predicted_velocity == state.predicted_velocity
        assert cloned.sequence_num == state.sequence_num
        assert cloned is not state

    def test_distance_to(self):
        """Test distance calculation."""
        state1 = PredictionState(predicted_position=(0.0, 0.0, 0.0))
        state2 = PredictionState(predicted_position=(3.0, 4.0, 0.0))

        distance = state1.distance_to(state2)
        assert abs(distance - 5.0) < 0.001  # 3-4-5 triangle

    def test_velocity_difference(self):
        """Test velocity difference calculation."""
        state1 = PredictionState(predicted_velocity=(1.0, 0.0, 0.0))
        state2 = PredictionState(predicted_velocity=(2.0, 0.0, 0.0))

        diff = state1.velocity_difference(state2)
        assert abs(diff - 1.0) < 0.001


class TestClientPredictor:
    """Tests for ClientPredictor class."""

    def test_predictor_creation(self):
        """Test predictor initialization."""
        predictor = ClientPredictor()
        assert predictor.current_sequence == 0
        assert predictor.prediction_accuracy == 1.0

    def test_predict_and_store(self):
        """Test prediction and input storage."""
        predictor = ClientPredictor()

        # Predict movement
        result = predictor.predict({"forward": True}, delta_time=0.016)
        predictor.store_input(0, {"forward": True})

        assert result.predicted_position[2] > 0
        assert predictor.current_sequence == 1

    def test_get_prediction_error(self):
        """Test error calculation with server state."""
        predictor = ClientPredictor()

        # Make a prediction
        predictor.predict({"forward": True}, delta_time=0.016)
        predictor.store_input(0, {"forward": True})

        # Server state matches
        server_state = predictor.current_state.clone()
        error = predictor.get_prediction_error(server_state, 0)
        assert error < 0.001

        # Server state differs
        server_state.predicted_position = (1.0, 0.0, 0.0)
        error = predictor.get_prediction_error(server_state, 0)
        assert error > 0

    def test_acknowledge_inputs(self):
        """Test acknowledging server-confirmed inputs."""
        predictor = ClientPredictor()

        # Store several inputs
        for i in range(5):
            predictor.predict({"forward": True}, delta_time=0.016)
            predictor.store_input(i, {"forward": True})

        # Acknowledge up to sequence 2
        predictor.acknowledge_inputs(2)
        unconfirmed = predictor.get_unconfirmed_inputs()

        assert len(unconfirmed) == 2  # 3 and 4 remain

    def test_set_state(self):
        """Test setting state directly."""
        predictor = ClientPredictor()

        new_state = PredictionState(
            predicted_position=(10.0, 0.0, 0.0),
            sequence_num=100,
        )

        predictor.set_state(new_state)
        assert predictor.current_state.predicted_position == (10.0, 0.0, 0.0)

    def test_reset(self):
        """Test resetting the predictor."""
        predictor = ClientPredictor()

        # Make some predictions
        for i in range(5):
            predictor.predict({"forward": True})
            predictor.store_input(i, {"forward": True})

        predictor.reset()

        assert predictor.current_sequence == 0
        assert predictor.input_buffer.is_empty()


class TestServerReconciler:
    """Tests for ServerReconciler class."""

    def test_reconciler_creation(self):
        """Test reconciler initialization."""
        reconciler = ServerReconciler()
        assert reconciler.snap_threshold == 0.5
        assert reconciler.max_reconcile_frames == 10

    def test_compare_states_match(self):
        """Test state comparison with matching states."""
        reconciler = ServerReconciler()

        state1 = PredictionState(predicted_position=(1.0, 2.0, 3.0))
        state2 = PredictionState(predicted_position=(1.0, 2.0, 3.0))

        result = reconciler.compare_states(state1, state2)
        assert result == ReconciliationResult.MATCH

    def test_compare_states_small_mismatch(self):
        """Test state comparison with small mismatch."""
        reconciler = ServerReconciler()

        state1 = PredictionState(predicted_position=(1.0, 2.0, 3.0))
        state2 = PredictionState(predicted_position=(1.1, 2.0, 3.0))

        result = reconciler.compare_states(state1, state2)
        assert result == ReconciliationResult.MISMATCH_SMALL

    def test_compare_states_large_mismatch(self):
        """Test state comparison with large mismatch."""
        reconciler = ServerReconciler()

        state1 = PredictionState(predicted_position=(0.0, 0.0, 0.0))
        state2 = PredictionState(predicted_position=(5.0, 0.0, 0.0))

        result = reconciler.compare_states(state1, state2)
        assert result == ReconciliationResult.MISMATCH_LARGE

    def test_rollback_to_server_state(self):
        """Test rollback to server state."""
        reconciler = ServerReconciler()

        server_state = PredictionState(predicted_position=(10.0, 0.0, 0.0))
        rolled_back = reconciler.rollback_to_server_state(server_state)

        assert rolled_back.predicted_position == (10.0, 0.0, 0.0)
        assert reconciler.is_correcting

    def test_replay_inputs(self):
        """Test input replay after rollback."""
        reconciler = ServerReconciler()
        buffer = InputBuffer()

        # Add inputs
        for i in range(3):
            buffer.push(i, {"forward": True})

        # Rollback and replay
        server_state = PredictionState(predicted_position=(0.0, 0.0, 0.0))
        reconciler.rollback_to_server_state(server_state)
        corrected = reconciler.replay_inputs(buffer)

        # Should have applied 3 forward inputs
        assert corrected.predicted_position[2] > 0
        assert not reconciler.is_correcting

    def test_should_snap(self):
        """Test snap threshold checking."""
        reconciler = ServerReconciler()
        reconciler.snap_threshold = 1.0

        assert not reconciler.should_snap(0.5)
        assert reconciler.should_snap(1.0)
        assert reconciler.should_snap(2.0)

    def test_reconciliation_stats(self):
        """Test statistics tracking."""
        reconciler = ServerReconciler()

        state1 = PredictionState(predicted_position=(0.0, 0.0, 0.0))
        state2 = PredictionState(predicted_position=(0.0, 0.0, 0.0))

        reconciler.compare_states(state1, state2)

        assert reconciler.stats.total_reconciliations == 1
        assert reconciler.stats.matches == 1


class TestReconciliationHistory:
    """Tests for ReconciliationHistory class."""

    def test_history_creation(self):
        """Test history initialization."""
        history = ReconciliationHistory(max_frames=50)
        assert len(history.get_recent()) == 0

    def test_record_and_retrieve(self):
        """Test recording and retrieving frames."""
        history = ReconciliationHistory()

        server_state = PredictionState()
        predicted_state = PredictionState()

        history.record(
            server_sequence=1,
            server_state=server_state,
            predicted_state=predicted_state,
            error=0.1,
            result=ReconciliationResult.MISMATCH_SMALL,
        )

        recent = history.get_recent(1)
        assert len(recent) == 1
        assert recent[0].server_sequence == 1
        assert recent[0].error == 0.1

    def test_get_mismatches(self):
        """Test filtering for mismatches."""
        history = ReconciliationHistory()

        server_state = PredictionState()
        predicted_state = PredictionState()

        # Add match
        history.record(1, server_state, predicted_state, 0.0, ReconciliationResult.MATCH)
        # Add mismatch
        history.record(2, server_state, predicted_state, 0.5, ReconciliationResult.MISMATCH_SMALL)

        mismatches = history.get_mismatches()
        assert len(mismatches) == 1
        assert mismatches[0].server_sequence == 2


class TestEntityInterpolation:
    """Tests for entity interpolation system."""

    def test_lerp_position(self):
        """Test linear position interpolation."""
        a = (0.0, 0.0, 0.0)
        b = (10.0, 0.0, 0.0)

        result = lerp_position(a, b, 0.5)
        assert abs(result[0] - 5.0) < 0.001
        assert result[1] == 0.0
        assert result[2] == 0.0

    def test_lerp_position_clamped(self):
        """Test lerp clamping."""
        a = (0.0, 0.0, 0.0)
        b = (10.0, 0.0, 0.0)

        # t > 1 should clamp
        result = lerp_position(a, b, 2.0)
        assert abs(result[0] - 10.0) < 0.001

        # t < 0 should clamp
        result = lerp_position(a, b, -1.0)
        assert abs(result[0] - 0.0) < 0.001

    def test_slerp_rotation(self):
        """Test spherical rotation interpolation."""
        # Identity quaternion
        a = (0.0, 0.0, 0.0, 1.0)
        # 90 degree rotation around Y
        b = (0.0, 0.7071, 0.0, 0.7071)

        result = slerp_rotation(a, b, 0.0)
        assert abs(result[3] - 1.0) < 0.01  # Should be close to identity

        result = slerp_rotation(a, b, 1.0)
        assert abs(result[1] - 0.7071) < 0.01  # Should be close to b

    def test_hermite_interpolate(self):
        """Test hermite position interpolation."""
        p0 = (0.0, 0.0, 0.0)
        p1 = (10.0, 0.0, 0.0)
        v0 = (10.0, 0.0, 0.0)
        v1 = (10.0, 0.0, 0.0)

        result = hermite_interpolate(p0, p1, v0, v1, 0.5, 1.0)
        assert 4.0 < result[0] < 6.0  # Should be roughly in the middle

    def test_hermite_endpoints(self):
        """Test hermite interpolation returns exact endpoints at t=0 and t=1."""
        p0 = (1.0, 2.0, 3.0)
        p1 = (10.0, 20.0, 30.0)
        v0 = (5.0, 5.0, 5.0)
        v1 = (5.0, 5.0, 5.0)

        # At t=0, should return p0
        result_start = hermite_interpolate(p0, p1, v0, v1, 0.0, 1.0)
        assert abs(result_start[0] - 1.0) < 0.001
        assert abs(result_start[1] - 2.0) < 0.001
        assert abs(result_start[2] - 3.0) < 0.001

        # At t=1, should return p1
        result_end = hermite_interpolate(p0, p1, v0, v1, 1.0, 1.0)
        assert abs(result_end[0] - 10.0) < 0.001
        assert abs(result_end[1] - 20.0) < 0.001
        assert abs(result_end[2] - 30.0) < 0.001

    def test_slerp_identical_quaternions(self):
        """Test slerp with identical quaternions doesn't divide by zero."""
        q = (0.0, 0.0, 0.0, 1.0)
        result = slerp_rotation(q, q, 0.5)
        # Should return the same quaternion (normalized)
        assert abs(result[3] - 1.0) < 0.001

    def test_slerp_opposite_quaternions(self):
        """Test slerp with opposite quaternions takes shorter path."""
        q1 = (0.0, 0.0, 0.0, 1.0)
        q2 = (0.0, 0.0, 0.0, -1.0)  # Same rotation, opposite sign
        result = slerp_rotation(q1, q2, 0.5)
        # Result should be valid (normalized)
        length = math.sqrt(sum(x*x for x in result))
        assert abs(length - 1.0) < 0.01

    def test_lerp_all_components(self):
        """Test lerp interpolates all three components correctly."""
        a = (1.0, 2.0, 3.0)
        b = (11.0, 22.0, 33.0)
        result = lerp_position(a, b, 0.5)
        assert abs(result[0] - 6.0) < 0.001
        assert abs(result[1] - 12.0) < 0.001
        assert abs(result[2] - 18.0) < 0.001


class TestInterpolationBuffer:
    """Tests for InterpolationBuffer class."""

    def test_buffer_creation(self):
        """Test buffer initialization."""
        buffer = InterpolationBuffer(buffer_size=3)
        assert buffer.buffer_size == 3
        assert buffer.snapshot_count == 0

    def test_push_snapshot(self):
        """Test adding snapshots."""
        buffer = InterpolationBuffer()

        snapshot = Snapshot(
            position=(1.0, 2.0, 3.0),
            timestamp=1.0,
        )
        buffer.push_snapshot(snapshot)

        assert buffer.snapshot_count == 1
        assert buffer.newest_timestamp == 1.0

    def test_get_interpolated(self):
        """Test getting interpolated state."""
        buffer = InterpolationBuffer()

        # Add two snapshots
        buffer.push_snapshot(Snapshot(position=(0.0, 0.0, 0.0), timestamp=0.0))
        buffer.push_snapshot(Snapshot(position=(10.0, 0.0, 0.0), timestamp=1.0))

        # Interpolate at middle
        result = buffer.get_interpolated(0.5)
        assert result is not None
        assert abs(result.position[0] - 5.0) < 0.001
        assert not result.is_extrapolated

    def test_extrapolation(self):
        """Test extrapolation beyond last snapshot."""
        buffer = InterpolationBuffer(extrapolation_limit=1.0)

        buffer.push_snapshot(Snapshot(
            position=(0.0, 0.0, 0.0),
            velocity=(10.0, 0.0, 0.0),
            timestamp=0.0,
        ))
        buffer.push_snapshot(Snapshot(
            position=(10.0, 0.0, 0.0),
            velocity=(10.0, 0.0, 0.0),
            timestamp=1.0,
        ))

        # Request time beyond last snapshot
        result = buffer.get_interpolated(1.5)
        assert result is not None
        assert result.is_extrapolated
        assert result.position[0] > 10.0

    def test_empty_buffer(self):
        """Test getting interpolated state from empty buffer."""
        buffer = InterpolationBuffer()
        result = buffer.get_interpolated(1.0)
        assert result is None

    def test_single_snapshot(self):
        """Test with single snapshot."""
        buffer = InterpolationBuffer()
        buffer.push_snapshot(Snapshot(position=(5.0, 0.0, 0.0), timestamp=1.0))

        result = buffer.get_interpolated(1.5)
        assert result is not None
        assert result.is_extrapolated

    def test_hermite_mode(self):
        """Test hermite interpolation mode."""
        buffer = InterpolationBuffer(interpolation_mode=InterpolationMode.HERMITE)

        buffer.push_snapshot(Snapshot(
            position=(0.0, 0.0, 0.0),
            velocity=(5.0, 0.0, 0.0),
            timestamp=0.0,
        ))
        buffer.push_snapshot(Snapshot(
            position=(10.0, 0.0, 0.0),
            velocity=(5.0, 0.0, 0.0),
            timestamp=1.0,
        ))

        result = buffer.get_interpolated(0.5, mode=InterpolationMode.HERMITE)
        assert result is not None


class TestEntityInterpolator:
    """Tests for EntityInterpolator class."""

    def test_interpolator_creation(self):
        """Test interpolator initialization."""
        interp = EntityInterpolator(entity_id=1, interpolation_delay=0.1)
        assert interp.entity_id == 1
        assert interp.interpolation_delay == 0.1

    def test_update(self):
        """Test interpolator update."""
        interp = EntityInterpolator(entity_id=1, interpolation_delay=0.0)

        # Add snapshots
        interp.add_snapshot(Snapshot(position=(0.0, 0.0, 0.0), timestamp=0.0))
        interp.add_snapshot(Snapshot(position=(10.0, 0.0, 0.0), timestamp=1.0))

        state = interp.update(server_time=0.5)
        assert state is not None
        assert abs(state.position[0] - 5.0) < 0.001


class TestSmoothing:
    """Tests for smoothing system."""

    def test_smooth_position(self):
        """Test position smoothing."""
        from_pos = (0.0, 0.0, 0.0)
        to_pos = (10.0, 0.0, 0.0)

        result = smooth_position(from_pos, to_pos, 0.5)
        assert abs(result[0] - 5.0) < 0.001

    def test_smooth_rotation(self):
        """Test rotation smoothing."""
        from_rot = (0.0, 0.0, 0.0, 1.0)
        to_rot = (0.0, 0.7071, 0.0, 0.7071)

        result = smooth_rotation(from_rot, to_rot, 0.0)
        assert abs(result[3] - 1.0) < 0.01

    def test_exponential_smooth(self):
        """Test exponential smoothing."""
        current = 0.0
        target = 10.0

        result = exponential_smooth(current, target, factor=10.0, delta_time=0.1)
        assert 0 < result < target  # Should approach but not reach

    def test_exponential_smooth_vector(self):
        """Test exponential vector smoothing."""
        current = (0.0, 0.0, 0.0)
        target = (10.0, 10.0, 10.0)

        result = exponential_smooth_vector(current, target, factor=10.0, delta_time=0.1)
        assert all(0 < r < 10 for r in result)


class TestCorrectionSmoother:
    """Tests for CorrectionSmoother class."""

    def test_smoother_creation(self):
        """Test smoother initialization."""
        smoother = CorrectionSmoother()
        assert smoother.blend_time == 0.1
        assert smoother.snap_threshold == 2.0
        assert not smoother.is_correcting

    def test_apply_correction_snap(self):
        """Test snap correction."""
        smoother = CorrectionSmoother()
        smoother.snap_threshold = 1.0

        current = (0.0, 0.0, 0.0)
        target = (10.0, 0.0, 0.0)  # Large error

        result = smoother.apply_correction(current, target, SmoothingMethod.THRESHOLD)
        assert result == target  # Should snap
        assert not smoother.is_correcting

    def test_apply_correction_interpolate(self):
        """Test interpolated correction."""
        smoother = CorrectionSmoother()
        smoother.snap_threshold = 5.0

        current = (0.0, 0.0, 0.0)
        target = (1.0, 0.0, 0.0)  # Small error

        result = smoother.apply_correction(current, target, SmoothingMethod.THRESHOLD)
        assert result == current  # Starts at current
        assert smoother.is_correcting

    def test_update_interpolation(self):
        """Test update during interpolation."""
        smoother = CorrectionSmoother()
        smoother.snap_threshold = 5.0

        current = (0.0, 0.0, 0.0)
        target = (1.0, 0.0, 0.0)

        smoother.apply_correction(current, target, SmoothingMethod.INTERPOLATE)

        # Update several times
        for _ in range(10):
            pos, _ = smoother.update(delta_time=0.02)

        # Should have moved towards target
        assert pos[0] > 0

    def test_cancel_correction(self):
        """Test canceling correction."""
        smoother = CorrectionSmoother()

        smoother.apply_correction(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            SmoothingMethod.INTERPOLATE,
        )
        assert smoother.is_correcting

        smoother.cancel_correction()
        assert not smoother.is_correcting

    def test_snap_to_target(self):
        """Test immediate snap to target."""
        smoother = CorrectionSmoother()

        smoother.apply_correction(
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            SmoothingMethod.INTERPOLATE,
        )

        smoother.snap_to_target()
        assert smoother.get_position() == (10.0, 0.0, 0.0)


class TestVisualSmoother:
    """Tests for VisualSmoother class."""

    def test_visual_smoother_creation(self):
        """Test visual smoother initialization."""
        smoother = VisualSmoother()
        assert smoother.get_visual_position() == (0.0, 0.0, 0.0)

    def test_set_simulation_state(self):
        """Test setting simulation state."""
        smoother = VisualSmoother()
        smoother.set_simulation_state((10.0, 0.0, 0.0))

        # Visual should still be at origin
        assert smoother.get_visual_position() == (0.0, 0.0, 0.0)

    def test_update_approaches_simulation(self):
        """Test visual approaches simulation over time."""
        smoother = VisualSmoother()
        smoother.set_simulation_state((10.0, 0.0, 0.0))

        # Update several times
        for _ in range(100):
            smoother.update(delta_time=0.016)

        pos = smoother.get_visual_position()
        assert abs(pos[0] - 10.0) < 0.1  # Should be close to simulation

    def test_snap_to_simulation(self):
        """Test snapping visual to simulation."""
        smoother = VisualSmoother()
        smoother.set_simulation_state((10.0, 0.0, 0.0))
        smoother.snap_to_simulation()

        assert smoother.get_visual_position() == (10.0, 0.0, 0.0)


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_input_buffer_operations(self):
        """Test operations on empty buffer."""
        buffer = InputBuffer()

        assert buffer.get_unconfirmed() == []
        assert buffer.get_input_at_sequence(0) is None
        assert buffer.pop_confirmed(10) == []

    def test_interpolation_buffer_out_of_order(self):
        """Test out-of-order snapshot insertion."""
        buffer = InterpolationBuffer()

        buffer.push_snapshot(Snapshot(position=(0.0, 0.0, 0.0), timestamp=2.0))
        buffer.push_snapshot(Snapshot(position=(5.0, 0.0, 0.0), timestamp=1.0))
        buffer.push_snapshot(Snapshot(position=(10.0, 0.0, 0.0), timestamp=3.0))

        assert buffer.oldest_timestamp == 1.0
        assert buffer.newest_timestamp == 3.0

    def test_reconciliation_with_none_states(self):
        """Test reconciler handles None states."""
        reconciler = ServerReconciler()

        result = reconciler.compare_states(None, PredictionState())
        assert result == ReconciliationResult.ERROR

        result = reconciler.compare_states(PredictionState(), None)
        assert result == ReconciliationResult.ERROR

    def test_max_history_exceeded(self):
        """Test behavior when max history is exceeded."""
        buffer = InterpolationBuffer(buffer_size=3)

        for i in range(10):
            buffer.push_snapshot(Snapshot(
                position=(float(i), 0.0, 0.0),
                timestamp=float(i),
            ))

        assert buffer.snapshot_count == 3
        # Oldest should be 7.0
        assert buffer.oldest_timestamp == 7.0

    def test_zero_duration_interpolation(self):
        """Test interpolation with zero duration between snapshots."""
        buffer = InterpolationBuffer()

        buffer.push_snapshot(Snapshot(position=(0.0, 0.0, 0.0), timestamp=1.0))
        buffer.push_snapshot(Snapshot(position=(10.0, 0.0, 0.0), timestamp=1.0))

        result = buffer.get_interpolated(1.0)
        assert result is not None


class TestMathCorrectness:
    """Tests to verify mathematical correctness of interpolation formulas."""

    def test_lerp_linearity(self):
        """Verify lerp produces linear interpolation."""
        a = (0.0, 0.0, 0.0)
        b = (10.0, 10.0, 10.0)

        # Test multiple points along the line
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = lerp_position(a, b, t)
            expected = t * 10.0
            assert abs(result[0] - expected) < 0.001
            assert abs(result[1] - expected) < 0.001
            assert abs(result[2] - expected) < 0.001

    def test_slerp_maintains_unit_length(self):
        """Verify slerp output is always a unit quaternion."""
        # Test with various input quaternions
        test_cases = [
            ((0.0, 0.0, 0.0, 1.0), (0.0, 0.7071, 0.0, 0.7071)),  # Y rotation
            ((0.7071, 0.0, 0.0, 0.7071), (0.0, 0.7071, 0.0, 0.7071)),  # X to Y
            ((0.5, 0.5, 0.5, 0.5), (0.0, 0.0, 0.0, 1.0)),  # Complex to identity
        ]

        for q1, q2 in test_cases:
            for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
                result = slerp_rotation(q1, q2, t)
                length = math.sqrt(sum(x*x for x in result))
                assert abs(length - 1.0) < 0.01, f"Slerp output not unit: {result}, length={length}"

    def test_exponential_smoothing_convergence(self):
        """Verify exponential smoothing converges to target."""
        current = 0.0
        target = 100.0

        # With high factor and many iterations, should converge
        for _ in range(1000):
            current = exponential_smooth(current, target, factor=10.0, delta_time=0.016)

        assert abs(current - target) < 0.001, f"Did not converge: {current}"

    def test_exponential_smoothing_formula(self):
        """Verify exponential smoothing follows correct formula."""
        current = 0.0
        target = 10.0
        factor = 5.0
        dt = 0.1

        result = exponential_smooth(current, target, factor, dt)
        # Formula: current + (target - current) * (1 - e^(-factor * dt))
        expected_blend = 1.0 - math.exp(-factor * dt)
        expected = current + (target - current) * expected_blend

        assert abs(result - expected) < 0.0001

    def test_prediction_physics_determinism(self):
        """Verify prediction produces same result with same inputs."""
        state1 = PredictionState(predicted_position=(1.0, 0.0, 1.0))
        state2 = PredictionState(predicted_position=(1.0, 0.0, 1.0))

        input_data = {"forward": True, "right": True}

        result1 = state1.apply_input(input_data, delta_time=0.016)
        result2 = state2.apply_input(input_data, delta_time=0.016)

        assert result1.predicted_position == result2.predicted_position
        assert result1.predicted_velocity == result2.predicted_velocity

    def test_reconciliation_error_calculation(self):
        """Verify reconciliation correctly calculates position error."""
        reconciler = ServerReconciler()

        # Known distance: 3-4-5 right triangle
        state1 = PredictionState(predicted_position=(0.0, 0.0, 0.0))
        state2 = PredictionState(predicted_position=(3.0, 4.0, 0.0))

        reconciler.compare_states(state1, state2)
        # Error should be 5.0 (hypotenuse of 3-4-5 triangle)
        assert reconciler.stats.total_error > 4.99
        assert reconciler.stats.total_error < 5.01

    def test_correction_smoother_completes(self):
        """Verify correction smoother reaches target position."""
        smoother = CorrectionSmoother()

        current = (0.0, 0.0, 0.0)
        target = (5.0, 5.0, 5.0)

        smoother.apply_correction(current, target, SmoothingMethod.INTERPOLATE)

        # Update until complete
        for _ in range(1000):
            pos, _ = smoother.update(delta_time=0.016)
            if not smoother.is_correcting:
                break

        # Should have reached target
        final_pos = smoother.get_position()
        assert abs(final_pos[0] - 5.0) < 0.01
        assert abs(final_pos[1] - 5.0) < 0.01
        assert abs(final_pos[2] - 5.0) < 0.01


class TestBoundaryConditions:
    """Test boundary conditions and edge cases."""

    def test_prediction_accuracy_no_mispredictions(self):
        """Verify accuracy is 100% with no mispredictions."""
        predictor = ClientPredictor()
        assert predictor.prediction_accuracy == 1.0

        # Make perfect predictions
        for i in range(10):
            predictor.predict({"forward": True}, delta_time=0.016)
            predictor.store_input(i, {"forward": True})
            server_state = predictor.current_state.clone()
            predictor.get_prediction_error(server_state, i)

        # All predictions matched, accuracy should be high
        assert predictor.prediction_accuracy > 0.99

    def test_prediction_accuracy_with_mispredictions(self):
        """Verify accuracy decreases with mispredictions."""
        predictor = ClientPredictor()

        for i in range(10):
            predictor.predict({"forward": True}, delta_time=0.016)
            predictor.store_input(i, {"forward": True})
            # Server state is completely different
            server_state = PredictionState(predicted_position=(100.0, 0.0, 0.0))
            predictor.get_prediction_error(server_state, i)

        # All predictions wrong, accuracy should be 0%
        assert predictor.prediction_accuracy < 0.01

    def test_buffer_boundary_sequence_wrap(self):
        """Test buffer handles large sequence numbers."""
        buffer = InputBuffer(max_size=5)

        # Use very large sequence numbers
        for i in range(1000000, 1000010):
            buffer.push(i, {"seq": i})

        assert buffer.size == 5
        entry = buffer.get_input_at_sequence(1000009)
        assert entry is not None
        assert entry.input_data["seq"] == 1000009

    def test_interpolation_negative_timestamps(self):
        """Test interpolation handles negative timestamps."""
        buffer = InterpolationBuffer()

        buffer.push_snapshot(Snapshot(position=(0.0, 0.0, 0.0), timestamp=-1.0))
        buffer.push_snapshot(Snapshot(position=(10.0, 0.0, 0.0), timestamp=0.0))

        result = buffer.get_interpolated(-0.5)
        assert result is not None
        assert abs(result.position[0] - 5.0) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
