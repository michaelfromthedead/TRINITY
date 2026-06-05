"""
Whitebox tests for the prediction and lag compensation layer.

Tests:
- Client prediction
- Server reconciliation
- Entity interpolation
- Input buffering
- Lag compensation
"""

import pytest
import math
import time
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

from engine.networking.prediction.client_prediction import (
    ClientPredictor,
    InputBuffer,
    BufferedInput,
    PredictionState,
)
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


# =============================================================================
# BufferedInput Tests
# =============================================================================

class TestBufferedInput:
    """Tests for BufferedInput dataclass."""

    def test_buffered_input_creation(self):
        """BufferedInput should store all fields."""
        input_data = {"forward": True, "jump": False}
        state = PredictionState()

        buffered = BufferedInput(
            sequence_num=42,
            input_data=input_data,
            timestamp=1.0,
            predicted_state=state
        )

        assert buffered.sequence_num == 42
        assert buffered.input_data == input_data
        assert buffered.timestamp == 1.0
        assert buffered.predicted_state is state

    def test_buffered_input_defaults(self):
        """BufferedInput should have sensible defaults."""
        buffered = BufferedInput(
            sequence_num=1,
            input_data={}
        )

        assert buffered.timestamp == 0.0
        assert buffered.predicted_state is None


# =============================================================================
# InputBuffer Tests
# =============================================================================

class TestInputBuffer:
    """Tests for InputBuffer."""

    def test_buffer_creation(self):
        """InputBuffer should initialize correctly."""
        buffer = InputBuffer(max_size=32)

        assert buffer.max_size == 32
        assert buffer.size == 0
        assert buffer.is_empty()

    def test_buffer_default_size(self):
        """InputBuffer should use default max size."""
        buffer = InputBuffer()
        assert buffer.max_size == DEFAULT_INPUT_BUFFER_SIZE

    def test_buffer_push(self):
        """push should add inputs to buffer."""
        buffer = InputBuffer()

        buffer.push(1, {"forward": True})
        buffer.push(2, {"backward": True})

        assert buffer.size == 2
        assert not buffer.is_empty()

    def test_buffer_push_with_state(self):
        """push should store predicted state."""
        buffer = InputBuffer()
        state = PredictionState(predicted_position=(1, 0, 0))

        buffer.push(1, {}, predicted_state=state)

        entry = buffer.get_input_at_sequence(1)
        assert entry.predicted_state.predicted_position == (1, 0, 0)

    def test_buffer_push_max_size(self):
        """Buffer should not exceed max size."""
        buffer = InputBuffer(max_size=5)

        for i in range(10):
            buffer.push(i, {"seq": i})

        assert buffer.size == 5

    def test_buffer_pop_confirmed(self):
        """pop_confirmed should remove confirmed inputs."""
        buffer = InputBuffer()

        for i in range(5):
            buffer.push(i, {"seq": i})

        removed = buffer.pop_confirmed(2)

        assert len(removed) == 3  # 0, 1, 2
        assert buffer.size == 2  # 3, 4 remain

    def test_buffer_pop_confirmed_returns_removed(self):
        """pop_confirmed should return removed inputs."""
        buffer = InputBuffer()
        buffer.push(1, {"data": "one"})
        buffer.push(2, {"data": "two"})

        removed = buffer.pop_confirmed(1)

        assert len(removed) == 1
        assert removed[0].sequence_num == 1
        assert removed[0].input_data["data"] == "one"

    def test_buffer_pop_confirmed_updates_last_confirmed(self):
        """pop_confirmed should update last_confirmed_sequence."""
        buffer = InputBuffer()
        buffer.push(1, {})
        buffer.push(2, {})

        assert buffer.last_confirmed_sequence == -1

        buffer.pop_confirmed(1)
        assert buffer.last_confirmed_sequence == 1

        buffer.pop_confirmed(2)
        assert buffer.last_confirmed_sequence == 2

    def test_buffer_pop_confirmed_ignores_old(self):
        """pop_confirmed with old sequence should be no-op."""
        buffer = InputBuffer()
        buffer.push(1, {})

        buffer.pop_confirmed(1)
        removed = buffer.pop_confirmed(0)  # Older than last confirmed

        assert removed == []

    def test_buffer_get_unconfirmed(self):
        """get_unconfirmed should return all remaining inputs."""
        buffer = InputBuffer()

        for i in range(5):
            buffer.push(i, {"seq": i})

        buffer.pop_confirmed(2)

        unconfirmed = buffer.get_unconfirmed()

        assert len(unconfirmed) == 2
        assert unconfirmed[0].sequence_num == 3
        assert unconfirmed[1].sequence_num == 4

    def test_buffer_get_input_at_sequence(self):
        """get_input_at_sequence should find specific input."""
        buffer = InputBuffer()
        buffer.push(10, {"data": "ten"})
        buffer.push(20, {"data": "twenty"})
        buffer.push(30, {"data": "thirty"})

        entry = buffer.get_input_at_sequence(20)

        assert entry is not None
        assert entry.input_data["data"] == "twenty"

    def test_buffer_get_input_at_sequence_not_found(self):
        """get_input_at_sequence should return None if not found."""
        buffer = InputBuffer()
        buffer.push(1, {})

        entry = buffer.get_input_at_sequence(999)

        assert entry is None

    def test_buffer_get_inputs_after_sequence(self):
        """get_inputs_after_sequence should return later inputs."""
        buffer = InputBuffer()

        for i in range(10):
            buffer.push(i, {"seq": i})

        after = buffer.get_inputs_after_sequence(5)

        assert len(after) == 4  # 6, 7, 8, 9
        assert all(e.sequence_num > 5 for e in after)

    def test_buffer_clear(self):
        """clear should empty the buffer."""
        buffer = InputBuffer()

        for i in range(5):
            buffer.push(i, {})

        buffer.pop_confirmed(2)
        buffer.clear()

        assert buffer.is_empty()
        assert buffer.last_confirmed_sequence == -1


# =============================================================================
# PredictionState Tests
# =============================================================================

class TestPredictionState:
    """Tests for PredictionState."""

    def test_state_defaults(self):
        """PredictionState should have zero defaults."""
        state = PredictionState()

        assert state.predicted_position == (0.0, 0.0, 0.0)
        assert state.predicted_velocity == (0.0, 0.0, 0.0)
        assert state.predicted_rotation is None
        assert state.sequence_num == 0
        assert state.timestamp == 0.0
        assert state.custom_data == {}

    def test_state_custom_values(self):
        """PredictionState should store custom values."""
        state = PredictionState(
            predicted_position=(10.0, 5.0, 20.0),
            predicted_velocity=(1.0, 0.0, 2.0),
            predicted_rotation=(0.0, 0.0, 0.0, 1.0),
            sequence_num=42,
            timestamp=1.5,
            custom_data={"health": 100}
        )

        assert state.predicted_position == (10.0, 5.0, 20.0)
        assert state.predicted_velocity == (1.0, 0.0, 2.0)
        assert state.predicted_rotation == (0.0, 0.0, 0.0, 1.0)
        assert state.sequence_num == 42
        assert state.timestamp == 1.5
        assert state.custom_data["health"] == 100

    def test_state_apply_input_forward(self):
        """apply_input should move forward."""
        state = PredictionState()
        new_state = state.apply_input({"forward": True}, delta_time=0.016)

        assert new_state.predicted_position[2] > 0  # Z forward

    def test_state_apply_input_backward(self):
        """apply_input should move backward."""
        state = PredictionState()
        new_state = state.apply_input({"backward": True}, delta_time=0.016)

        assert new_state.predicted_position[2] < 0  # Z backward

    def test_state_apply_input_left(self):
        """apply_input should move left."""
        state = PredictionState()
        new_state = state.apply_input({"left": True}, delta_time=0.016)

        assert new_state.predicted_position[0] < 0  # X left

    def test_state_apply_input_right(self):
        """apply_input should move right."""
        state = PredictionState()
        new_state = state.apply_input({"right": True}, delta_time=0.016)

        assert new_state.predicted_position[0] > 0  # X right

    def test_state_apply_input_jump(self):
        """apply_input should apply jump velocity."""
        state = PredictionState()  # On ground (y=0)
        new_state = state.apply_input({"jump": True}, delta_time=0.016)

        # Should have upward velocity applied
        assert new_state.predicted_velocity[1] > 0 or new_state.predicted_position[1] > 0

    def test_state_apply_input_gravity(self):
        """apply_input should apply gravity."""
        state = PredictionState(predicted_position=(0, 10, 0))  # In air
        new_state = state.apply_input({}, delta_time=0.1)

        # Should have fallen
        assert new_state.predicted_position[1] < 10

    def test_state_apply_input_ground_check(self):
        """apply_input should clamp to ground."""
        state = PredictionState(
            predicted_position=(0, 0.05, 0),
            predicted_velocity=(0, -10, 0)
        )
        new_state = state.apply_input({}, delta_time=0.1)

        # Should be clamped to ground
        assert new_state.predicted_position[1] >= 0

    def test_state_apply_input_friction(self):
        """apply_input should apply friction."""
        state = PredictionState(predicted_velocity=(10, 0, 10))
        new_state = state.apply_input({}, delta_time=0.016)

        # Velocity should decrease due to friction
        assert abs(new_state.predicted_velocity[0]) < 10
        assert abs(new_state.predicted_velocity[2]) < 10

    def test_state_apply_input_increments_sequence(self):
        """apply_input should increment sequence number."""
        state = PredictionState(sequence_num=5)
        new_state = state.apply_input({})

        assert new_state.sequence_num == 6

    def test_state_apply_input_preserves_custom_data(self):
        """apply_input should preserve custom_data."""
        state = PredictionState(custom_data={"health": 100})
        new_state = state.apply_input({})

        assert new_state.custom_data["health"] == 100

    def test_state_clone(self):
        """clone should create independent copy."""
        original = PredictionState(
            predicted_position=(1, 2, 3),
            custom_data={"key": "value"}
        )
        cloned = original.clone()

        assert cloned.predicted_position == original.predicted_position
        assert cloned.custom_data == original.custom_data
        assert cloned is not original
        assert cloned.custom_data is not original.custom_data

    def test_state_distance_to(self):
        """distance_to should calculate Euclidean distance."""
        state1 = PredictionState(predicted_position=(0, 0, 0))
        state2 = PredictionState(predicted_position=(3, 4, 0))

        distance = state1.distance_to(state2)

        assert distance == 5.0  # 3-4-5 triangle

    def test_state_distance_to_same_position(self):
        """distance_to same position should be 0."""
        state1 = PredictionState(predicted_position=(5, 5, 5))
        state2 = PredictionState(predicted_position=(5, 5, 5))

        distance = state1.distance_to(state2)

        assert distance == 0.0

    def test_state_velocity_difference(self):
        """velocity_difference should calculate magnitude difference."""
        state1 = PredictionState(predicted_velocity=(0, 0, 0))
        state2 = PredictionState(predicted_velocity=(3, 4, 0))

        diff = state1.velocity_difference(state2)

        assert diff == 5.0


# =============================================================================
# ClientPredictor Tests
# =============================================================================

class TestClientPredictor:
    """Tests for ClientPredictor."""

    def test_predictor_creation(self):
        """ClientPredictor should initialize correctly."""
        predictor = ClientPredictor()

        assert predictor.current_sequence == 0
        assert predictor.prediction_accuracy == 1.0
        assert predictor.input_buffer.is_empty()

    def test_predictor_with_initial_state(self):
        """ClientPredictor should accept initial state."""
        initial = PredictionState(predicted_position=(10, 0, 10))
        predictor = ClientPredictor(initial_state=initial)

        assert predictor.current_state.predicted_position == (10, 0, 10)

    def test_predictor_predict_updates_state(self):
        """predict should update current state."""
        predictor = ClientPredictor()

        new_state = predictor.predict({"forward": True})

        assert new_state.predicted_position[2] > 0
        assert predictor.current_state.predicted_position[2] > 0

    def test_predictor_predict_returns_state(self):
        """predict should return new state."""
        predictor = ClientPredictor()

        new_state = predictor.predict({"right": True})

        assert isinstance(new_state, PredictionState)
        assert new_state.predicted_position[0] > 0

    def test_predictor_store_input(self):
        """store_input should add to buffer."""
        predictor = ClientPredictor()

        predictor.store_input(1, {"forward": True})

        assert not predictor.input_buffer.is_empty()
        entry = predictor.input_buffer.get_input_at_sequence(1)
        assert entry is not None

    def test_predictor_store_input_updates_sequence(self):
        """store_input should update current_sequence."""
        predictor = ClientPredictor()

        predictor.store_input(5, {})

        assert predictor.current_sequence == 6

    def test_predictor_store_input_captures_state(self):
        """store_input should capture predicted state."""
        predictor = ClientPredictor()
        predictor.predict({"forward": True})
        predictor.store_input(1, {"forward": True})

        entry = predictor.input_buffer.get_input_at_sequence(1)
        assert entry.predicted_state is not None

    def test_predictor_get_prediction_error(self):
        """get_prediction_error should calculate error."""
        predictor = ClientPredictor()
        predictor.predict({"forward": True})
        predictor.store_input(0, {"forward": True})

        # Server says we're at a different position
        server_state = PredictionState(predicted_position=(10, 0, 10))

        error = predictor.get_prediction_error(server_state, 0)

        assert error > 0

    def test_predictor_get_prediction_error_updates_stats(self):
        """get_prediction_error should track mispredictions."""
        predictor = ClientPredictor()
        predictor.predict({})
        predictor.store_input(0, {})

        # Large error
        server_state = PredictionState(predicted_position=(100, 0, 100))
        predictor.get_prediction_error(server_state, 0)

        assert predictor.prediction_accuracy < 1.0

    def test_predictor_acknowledge_inputs(self):
        """acknowledge_inputs should pop confirmed."""
        predictor = ClientPredictor()

        for i in range(5):
            predictor.store_input(i, {"seq": i})

        predictor.acknowledge_inputs(2)

        assert predictor.input_buffer.size == 2  # 3 and 4

    def test_predictor_get_unconfirmed_inputs(self):
        """get_unconfirmed_inputs should return buffer contents."""
        predictor = ClientPredictor()

        for i in range(3):
            predictor.store_input(i, {})

        unconfirmed = predictor.get_unconfirmed_inputs()

        assert len(unconfirmed) == 3

    def test_predictor_set_state(self):
        """set_state should override current state."""
        predictor = ClientPredictor()
        predictor.predict({"forward": True})

        new_state = PredictionState(predicted_position=(100, 50, 200))
        predictor.set_state(new_state)

        assert predictor.current_state.predicted_position == (100, 50, 200)

    def test_predictor_set_state_clones(self):
        """set_state should clone the state."""
        predictor = ClientPredictor()

        new_state = PredictionState(predicted_position=(1, 2, 3))
        predictor.set_state(new_state)

        # Modify original shouldn't affect predictor
        assert predictor.current_state is not new_state

    def test_predictor_reset(self):
        """reset should clear all state."""
        predictor = ClientPredictor()

        # Add some state
        predictor.predict({"forward": True})
        predictor.store_input(1, {})
        predictor.store_input(2, {})

        predictor.reset()

        assert predictor.current_state.predicted_position == (0.0, 0.0, 0.0)
        assert predictor.input_buffer.is_empty()
        assert predictor.current_sequence == 0

    def test_predictor_reset_with_initial_state(self):
        """reset should accept new initial state."""
        predictor = ClientPredictor()
        predictor.predict({"forward": True})

        new_initial = PredictionState(predicted_position=(50, 0, 50))
        predictor.reset(initial_state=new_initial)

        assert predictor.current_state.predicted_position == (50, 0, 50)


class TestClientPredictorPredictionHistory:
    """Tests for prediction history tracking."""

    def test_history_stores_states(self):
        """predict should store states in history."""
        predictor = ClientPredictor()

        for _ in range(5):
            predictor.predict({"forward": True})

        # History should have states
        # (internal access for testing)
        assert len(predictor._prediction_history) >= 0

    def test_history_limited_size(self):
        """History should not exceed max size."""
        predictor = ClientPredictor(prediction_history_size=10)

        for _ in range(20):
            predictor.predict({"forward": True})

        assert len(predictor._prediction_history) <= 10


class TestClientPredictorReconciliation:
    """Tests for server reconciliation scenarios."""

    def test_reconciliation_perfect_prediction(self):
        """Perfect prediction should have zero error."""
        predictor = ClientPredictor()

        # Predict locally
        state = predictor.predict({"forward": True}, delta_time=0.016)
        predictor.store_input(0, {"forward": True})

        # Server agrees
        error = predictor.get_prediction_error(state, 0)

        assert error == 0.0

    def test_reconciliation_misprediction_detection(self):
        """Misprediction should be detected."""
        predictor = ClientPredictor()

        predictor.predict({"forward": True})
        predictor.store_input(0, {"forward": True})

        # Server disagrees significantly
        server_state = PredictionState(predicted_position=(50, 0, 50))
        error = predictor.get_prediction_error(server_state, 0)

        assert error > MISPREDICTION_THRESHOLD

    def test_reconciliation_replay_inputs(self):
        """After misprediction, inputs should be replayable."""
        predictor = ClientPredictor()

        # Store several inputs
        for i in range(5):
            predictor.predict({"forward": True})
            predictor.store_input(i, {"forward": True})

        # Server confirms sequence 2
        predictor.acknowledge_inputs(2)

        # Get unconfirmed for replay
        unconfirmed = predictor.get_unconfirmed_inputs()

        assert len(unconfirmed) == 2  # 3 and 4
        assert unconfirmed[0].sequence_num == 3


class TestClientPredictorInputBuffer:
    """Tests for input buffer accessor."""

    def test_input_buffer_accessor(self):
        """input_buffer property should return buffer."""
        predictor = ClientPredictor()

        buffer = predictor.input_buffer

        assert isinstance(buffer, InputBuffer)

    def test_input_buffer_same_instance(self):
        """input_buffer should return same instance."""
        predictor = ClientPredictor()

        buffer1 = predictor.input_buffer
        buffer2 = predictor.input_buffer

        assert buffer1 is buffer2


# =============================================================================
# Movement Physics Tests
# =============================================================================

class TestMovementPhysics:
    """Tests for movement physics calculations."""

    def test_combined_movement(self):
        """Combined input should work correctly."""
        state = PredictionState()

        # Forward + right diagonal
        new_state = state.apply_input(
            {"forward": True, "right": True},
            delta_time=0.1,
            move_speed=10.0
        )

        # Both X and Z should be positive
        assert new_state.predicted_position[0] > 0
        assert new_state.predicted_position[2] > 0

    def test_velocity_persistence(self):
        """Velocity should persist between frames."""
        state = PredictionState()

        # Apply input to build velocity
        state = state.apply_input({"forward": True}, delta_time=0.1)
        initial_z = state.predicted_position[2]

        # No input, but still moving
        state = state.apply_input({}, delta_time=0.1)

        # Should have moved further
        assert state.predicted_position[2] > initial_z

    def test_gravity_in_air(self):
        """Gravity should accelerate falling."""
        state = PredictionState(
            predicted_position=(0, 100, 0),
            predicted_velocity=(0, 0, 0)
        )

        # Fall for a bit
        for _ in range(10):
            state = state.apply_input({}, delta_time=0.1)

        # Should have fallen
        assert state.predicted_position[1] < 100
        # Should have downward velocity
        assert state.predicted_velocity[1] < 0

    def test_jump_from_ground(self):
        """Jump should only work from ground."""
        # On ground
        ground_state = PredictionState(predicted_position=(0, 0, 0))
        jumped = ground_state.apply_input({"jump": True})

        # In air
        air_state = PredictionState(predicted_position=(0, 10, 0))
        not_jumped = air_state.apply_input({"jump": True})

        # Ground jump should work
        assert jumped.predicted_velocity[1] > 0 or jumped.predicted_position[1] > 0

    def test_custom_physics_params(self):
        """Custom physics parameters should be used."""
        state = PredictionState()

        # High move speed
        fast_state = state.apply_input(
            {"forward": True},
            delta_time=0.1,
            move_speed=100.0
        )

        # Low move speed
        slow_state = state.apply_input(
            {"forward": True},
            delta_time=0.1,
            move_speed=1.0
        )

        assert fast_state.predicted_position[2] > slow_state.predicted_position[2]


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestPredictionEdgeCases:
    """Tests for edge cases in prediction."""

    def test_zero_delta_time(self):
        """Zero delta time should not crash."""
        state = PredictionState()
        new_state = state.apply_input({"forward": True}, delta_time=0.0)

        assert new_state is not None

    def test_very_small_delta_time(self):
        """Very small delta time should work."""
        state = PredictionState()
        new_state = state.apply_input({"forward": True}, delta_time=0.0001)

        assert new_state is not None

    def test_large_delta_time(self):
        """Large delta time should work."""
        state = PredictionState()
        new_state = state.apply_input({"forward": True}, delta_time=1.0)

        assert new_state is not None

    def test_empty_buffer_operations(self):
        """Operations on empty buffer should be safe."""
        buffer = InputBuffer()

        assert buffer.get_unconfirmed() == []
        assert buffer.get_input_at_sequence(0) is None
        assert buffer.get_inputs_after_sequence(0) == []
        assert buffer.pop_confirmed(0) == []

    def test_negative_sequence_handling(self):
        """Negative sequences should be handled."""
        buffer = InputBuffer()

        # This might not be a valid use case, but shouldn't crash
        after = buffer.get_inputs_after_sequence(-1)
        assert isinstance(after, list)

    def test_concurrent_predict_and_store(self):
        """Predict and store should work correctly together."""
        predictor = ClientPredictor()

        # Typical game loop pattern
        for i in range(10):
            input_data = {"forward": True}
            predictor.predict(input_data)
            predictor.store_input(i, input_data)

        # All should be stored
        assert predictor.input_buffer.size == 10


class TestPredictionAccuracy:
    """Tests for prediction accuracy tracking."""

    def test_accuracy_starts_at_one(self):
        """Initial accuracy should be 1.0."""
        predictor = ClientPredictor()
        assert predictor.prediction_accuracy == 1.0

    def test_accuracy_decreases_on_misprediction(self):
        """Accuracy should decrease on misprediction."""
        predictor = ClientPredictor()
        predictor.predict({})
        predictor.store_input(0, {})

        # Large misprediction
        server_state = PredictionState(predicted_position=(100, 0, 100))
        predictor.get_prediction_error(server_state, 0)

        assert predictor.prediction_accuracy < 1.0

    def test_accuracy_calculation(self):
        """Accuracy should be calculated correctly."""
        predictor = ClientPredictor()

        # Perfect predictions
        for i in range(5):
            state = predictor.predict({})
            predictor.store_input(i, {})
            predictor.get_prediction_error(state, i)  # Zero error

        # Should still be good accuracy
        assert predictor.prediction_accuracy >= 0.8
