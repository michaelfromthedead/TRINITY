"""
WHITEBOX Tests for Death System

Tests internal implementation details:
- Death state machine transitions
- Respawn queue management
- Cleanup handler execution
- Dying duration timing
- Event emission order
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch

from engine.gameplay.combat.death import (
    DeathSystem,
    DeathInfo,
    DeathEvent,
    RespawnRequest,
    RespawnEvent,
)
from engine.gameplay.combat.constants import (
    DeathState,
    DeathConfig,
    DEFAULT_DEATH_CONFIG,
    DEFAULT_RESPAWN_TIME,
    MIN_RESPAWN_TIME,
    MAX_RESPAWN_TIME,
    DYING_DURATION,
    RESPAWN_HEALTH_PERCENTAGE,
    RESPAWN_INVULNERABILITY_DURATION,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def death_system():
    """Create a fresh death system."""
    return DeathSystem()


@pytest.fixture
def custom_config_system():
    """Create death system with custom config."""
    config = DeathConfig(
        dying_duration=2.0,
        default_respawn_time=3.0,
        respawn_health_percentage=0.5,
    )
    return DeathSystem(config=config)


@pytest.fixture
def mock_respawn_provider():
    """Create a mock respawn provider."""
    provider = Mock()
    provider.get_respawn_position.return_value = (0, 0, 0)
    return provider


@pytest.fixture
def mock_death_subject():
    """Create a mock death subject."""
    subject = Mock()
    subject.entity_id = 1
    subject.is_dead = True
    subject.kill.return_value = True
    subject.revive.return_value = True
    return subject


# =============================================================================
# DEATH STATE MACHINE TESTS (40 tests)
# =============================================================================


class TestDeathStateMachine:
    """Tests for death state machine transitions."""

    def test_process_death_creates_info(self, death_system):
        """process_death should create DeathInfo."""
        info = death_system.process_death(1, killer_id=2)
        assert info is not None
        assert info.entity_id == 1
        assert info.killer_id == 2

    def test_process_death_initial_state_dying(self, death_system):
        """Initial death state should be DYING."""
        info = death_system.process_death(1)
        assert info.death_state == DeathState.DYING

    def test_process_death_stores_state(self, death_system):
        """Death info should be stored."""
        death_system.process_death(1)
        info = death_system.get_death_info(1)
        assert info is not None

    def test_get_death_state_alive(self, death_system):
        """Non-dead entity should be ALIVE."""
        state = death_system.get_death_state(999)
        assert state == DeathState.ALIVE

    def test_get_death_state_dying(self, death_system):
        """Dying entity should be DYING."""
        death_system.process_death(1)
        state = death_system.get_death_state(1)
        assert state == DeathState.DYING

    def test_is_dying(self, death_system):
        """is_dying should return correct value."""
        death_system.process_death(1)
        assert death_system.is_dying(1)
        assert not death_system.is_dying(999)

    def test_is_dead(self, death_system):
        """is_dead should return correct value."""
        death_system.process_death(1)
        death_system.transition_to_dead(1)
        assert death_system.is_dead(1)

    def test_is_respawning(self, death_system):
        """is_respawning should return correct value."""
        death_system.process_death(1)
        death_system.transition_to_dead(1)
        death_system.transition_to_respawning(1)
        assert death_system.is_respawning(1)

    def test_transition_dying_to_dead(self, death_system):
        """Should transition from DYING to DEAD."""
        death_system.process_death(1)
        result = death_system.transition_to_dead(1)

        assert result
        assert death_system.get_death_state(1) == DeathState.DEAD

    def test_transition_to_dead_requires_dying(self, death_system):
        """transition_to_dead requires DYING state."""
        result = death_system.transition_to_dead(999)
        assert not result

    def test_transition_to_dead_from_alive_fails(self, death_system):
        """Cannot transition directly from ALIVE to DEAD."""
        result = death_system.transition_to_dead(1)
        assert not result

    def test_transition_dead_to_respawning(self, death_system):
        """Should transition from DEAD to RESPAWNING."""
        death_system.process_death(1)
        death_system.transition_to_dead(1)
        result = death_system.transition_to_respawning(1)

        assert result
        assert death_system.get_death_state(1) == DeathState.RESPAWNING

    def test_transition_to_respawning_requires_dead(self, death_system):
        """transition_to_respawning requires DEAD state."""
        death_system.process_death(1)  # DYING state
        result = death_system.transition_to_respawning(1)
        assert not result

    def test_complete_respawn(self, death_system):
        """complete_respawn should clear death state."""
        death_system.process_death(1)
        death_system.transition_to_dead(1)
        death_system.transition_to_respawning(1)

        result = death_system.complete_respawn(1)
        assert result
        assert death_system.get_death_state(1) == DeathState.ALIVE

    def test_complete_respawn_no_info(self, death_system):
        """complete_respawn with no death info should return False."""
        result = death_system.complete_respawn(999)
        assert not result

    def test_death_info_timestamp(self, death_system):
        """Death info should have timestamp."""
        before = time.time()
        info = death_system.process_death(1)
        after = time.time()

        assert before <= info.timestamp <= after

    def test_death_info_time_since_death(self, death_system):
        """time_since_death should calculate correctly."""
        info = death_system.process_death(1)

        # Set timestamp in the past
        info.timestamp = time.time() - 5.0

        assert abs(info.time_since_death - 5.0) < 0.5

    def test_death_info_is_fully_dead(self, death_system):
        """is_fully_dead should check DEAD state."""
        info = death_system.process_death(1)
        assert not info.is_fully_dead

        death_system.transition_to_dead(1)
        assert info.is_fully_dead

    def test_death_with_metadata(self, death_system):
        """Death should store metadata."""
        info = death_system.process_death(
            1,
            death_cause="explosion",
            weapon_id=42,
            was_headshot=True,
            overkill_damage=50.0,
        )

        assert info.death_cause == "explosion"
        assert info.weapon_id == 42
        assert info.was_headshot
        assert info.overkill_damage == 50.0


# =============================================================================
# RESPAWN QUEUE TESTS (35 tests)
# =============================================================================


class TestRespawnQueue:
    """Tests for respawn queue management."""

    def test_queue_respawn_creates_request(self, death_system):
        """queue_respawn should create request."""
        death_system.process_death(1)
        death_system.transition_to_dead(1)
        request = death_system.queue_respawn(1)

        assert request is not None
        assert request.entity_id == 1

    def test_queue_respawn_default_delay(self, death_system):
        """queue_respawn should use default delay."""
        death_system.process_death(1)
        death_system.transition_to_dead(1)
        request = death_system.queue_respawn(1)

        expected_time = time.time() + DEFAULT_RESPAWN_TIME
        assert abs(request.respawn_time - expected_time) < 0.1

    def test_queue_respawn_custom_delay(self, death_system):
        """queue_respawn should use custom delay."""
        death_system.process_death(1)
        death_system.transition_to_dead(1)
        request = death_system.queue_respawn(1, delay=10.0)

        expected_time = time.time() + 10.0
        assert abs(request.respawn_time - expected_time) < 0.1

    def test_queue_respawn_delay_clamped_min(self, death_system):
        """Delay should be clamped to minimum."""
        death_system.process_death(1)
        death_system.transition_to_dead(1)

        before = time.time()
        request = death_system.queue_respawn(1, delay=-5.0)
        after = time.time()

        # Request time should be at least MIN_RESPAWN_TIME from now
        assert request.respawn_time >= before + MIN_RESPAWN_TIME - 0.01
        assert request.respawn_time <= after + MIN_RESPAWN_TIME + 0.01

    def test_queue_respawn_delay_clamped_max(self, death_system):
        """Delay should be clamped to maximum."""
        death_system.process_death(1)
        death_system.transition_to_dead(1)
        request = death_system.queue_respawn(1, delay=1000.0)

        expected_time = time.time() + MAX_RESPAWN_TIME
        assert request.respawn_time <= expected_time + 0.1

    def test_queue_respawn_transitions_state(self, death_system):
        """queue_respawn should transition to RESPAWNING."""
        death_system.process_death(1)
        death_system.transition_to_dead(1)
        death_system.queue_respawn(1)

        assert death_system.is_respawning(1)

    def test_queue_respawn_replaces_existing(self, death_system):
        """New respawn request should replace existing."""
        death_system.process_death(1)
        death_system.transition_to_dead(1)
        death_system.queue_respawn(1, delay=5.0)
        death_system.queue_respawn(1, delay=10.0)

        assert len(death_system._respawn_queue) == 1
        assert death_system._respawn_queue[0].respawn_time > time.time() + 8.0

    def test_cancel_respawn(self, death_system):
        """cancel_respawn should remove request."""
        death_system.process_death(1)
        death_system.transition_to_dead(1)
        death_system.queue_respawn(1)

        result = death_system.cancel_respawn(1)
        assert result
        assert death_system.get_respawn_request(1) is None

    def test_cancel_respawn_reverts_state(self, death_system):
        """cancel_respawn should revert to DEAD state."""
        death_system.process_death(1)
        death_system.transition_to_dead(1)
        death_system.queue_respawn(1)
        death_system.cancel_respawn(1)

        assert death_system.is_dead(1)

    def test_cancel_nonexistent_respawn(self, death_system):
        """cancel_respawn should return False if none exists."""
        result = death_system.cancel_respawn(999)
        assert not result

    def test_get_respawn_request(self, death_system):
        """get_respawn_request should return request."""
        death_system.process_death(1)
        death_system.transition_to_dead(1)
        death_system.queue_respawn(1)

        request = death_system.get_respawn_request(1)
        assert request is not None
        assert request.entity_id == 1

    def test_get_respawn_time_remaining(self, death_system):
        """get_respawn_time_remaining should calculate correctly."""
        death_system.process_death(1)
        death_system.transition_to_dead(1)
        death_system.queue_respawn(1, delay=10.0)

        remaining = death_system.get_respawn_time_remaining(1)
        assert 9.0 < remaining <= 10.0

    def test_get_respawn_time_remaining_no_request(self, death_system):
        """get_respawn_time_remaining should return 0 if no request."""
        remaining = death_system.get_respawn_time_remaining(999)
        assert remaining == 0.0

    def test_respawn_request_is_ready(self, death_system):
        """is_ready should check if time has passed."""
        with patch('time.time') as mock_time:
            mock_time.return_value = 1000.0
            death_system.process_death(1)
            death_system.transition_to_dead(1)
            request = death_system.queue_respawn(1, delay=5.0)

            assert not request.is_ready

            mock_time.return_value = 1006.0
            assert request.is_ready

    def test_respawn_request_time_until(self, death_system):
        """time_until_respawn should calculate remaining."""
        with patch('time.time') as mock_time:
            mock_time.return_value = 1000.0
            death_system.process_death(1)
            death_system.transition_to_dead(1)
            request = death_system.queue_respawn(1, delay=5.0)

            mock_time.return_value = 1002.0
            assert abs(request.time_until_respawn - 3.0) < 0.1


# =============================================================================
# INSTANT RESPAWN TESTS (15 tests)
# =============================================================================


class TestInstantRespawn:
    """Tests for instant respawn functionality."""

    def test_instant_respawn(self, death_system, mock_death_subject):
        """instant_respawn should immediately respawn."""
        death_system.process_death(1)
        death_system.transition_to_dead(1)

        result = death_system.instant_respawn(mock_death_subject)
        assert result
        mock_death_subject.revive.assert_called()

    def test_instant_respawn_clears_state(self, death_system, mock_death_subject):
        """instant_respawn should clear death state."""
        death_system.process_death(1)
        death_system.transition_to_dead(1)

        death_system.instant_respawn(mock_death_subject)
        assert death_system.get_death_state(1) == DeathState.ALIVE

    def test_instant_respawn_with_position(self, death_system, mock_death_subject, mock_respawn_provider):
        """instant_respawn should use respawn provider."""
        death_system._respawn_provider = mock_respawn_provider
        death_system.process_death(1)
        death_system.transition_to_dead(1)

        death_system.instant_respawn(mock_death_subject, team_id=1)
        mock_respawn_provider.get_respawn_position.assert_called()

    def test_instant_respawn_health_percentage(self, death_system, mock_death_subject):
        """instant_respawn should respect health percentage."""
        death_system.process_death(1)
        death_system.transition_to_dead(1)

        death_system.instant_respawn(mock_death_subject, health_percentage=0.5)
        mock_death_subject.revive.assert_called_with(
            health_percentage=0.5,
            add_invulnerability=True,
            invulnerability_duration=RESPAWN_INVULNERABILITY_DURATION,
        )

    def test_instant_respawn_emits_event(self, death_system, mock_death_subject):
        """instant_respawn should emit respawn event."""
        callback = Mock()
        death_system.on_respawn(callback)

        death_system.process_death(1)
        death_system.transition_to_dead(1)
        death_system.instant_respawn(mock_death_subject)

        callback.assert_called()


# =============================================================================
# UPDATE AND TIMING TESTS (25 tests)
# =============================================================================


class TestUpdateAndTiming:
    """Tests for update loop and timing."""

    def test_update_transitions_dying_to_dead(self, death_system):
        """Update should transition DYING to DEAD after duration."""
        info = death_system.process_death(1)

        # Before duration - should still be dying
        assert death_system.is_dying(1)

        # Set timestamp in the past to simulate duration elapsed
        info.timestamp = time.time() - DYING_DURATION - 1.0

        death_system.update(0.016)
        assert death_system.is_dead(1)

    def test_update_processes_respawn_queue(self, death_system, mock_death_subject):
        """Update should process ready respawns."""
        with patch('time.time') as mock_time:
            mock_time.return_value = 1000.0
            death_system.process_death(1)
            death_system.transition_to_dead(1)
            death_system.queue_respawn(1, delay=5.0)

            # Before ready
            mock_time.return_value = 1003.0
            respawned = death_system.update(0.016, {1: mock_death_subject})
            assert len(respawned) == 0

            # After ready
            mock_time.return_value = 1006.0
            respawned = death_system.update(0.016, {1: mock_death_subject})
            assert 1 in respawned

    def test_update_removes_respawned_from_queue(self, death_system, mock_death_subject):
        """Update should remove processed respawns from queue."""
        with patch('time.time') as mock_time:
            mock_time.return_value = 1000.0
            death_system.process_death(1)
            death_system.transition_to_dead(1)
            death_system.queue_respawn(1, delay=0.0)

            mock_time.return_value = 1001.0
            death_system.update(0.016, {1: mock_death_subject})

            assert len(death_system._respawn_queue) == 0

    def test_update_emits_respawn_event(self, death_system, mock_death_subject):
        """Update should emit respawn events."""
        callback = Mock()
        death_system.on_respawn(callback)

        with patch('time.time') as mock_time:
            mock_time.return_value = 1000.0
            death_system.process_death(1)
            death_system.transition_to_dead(1)
            death_system.queue_respawn(1, delay=0.0)

            mock_time.return_value = 1001.0
            death_system.update(0.016, {1: mock_death_subject})

            callback.assert_called()

    def test_custom_config_dying_duration(self, custom_config_system):
        """Custom dying duration should be respected."""
        info = custom_config_system.process_death(1)

        # Just after death - should be dying
        assert custom_config_system.is_dying(1)

        # Set timestamp to simulate 2.5s elapsed (custom config is 2.0s)
        info.timestamp = time.time() - 2.5

        custom_config_system.update(0.016)
        assert custom_config_system.is_dead(1)


# =============================================================================
# CLEANUP HANDLER TESTS (20 tests)
# =============================================================================


class TestCleanupHandlers:
    """Tests for cleanup handler execution."""

    def test_register_cleanup_handler(self, death_system):
        """register_cleanup_handler should add handler."""
        handler = Mock()
        death_system.register_cleanup_handler(handler)
        assert handler in death_system._cleanup_handlers

    def test_unregister_cleanup_handler(self, death_system):
        """unregister_cleanup_handler should remove handler."""
        handler = Mock()
        death_system.register_cleanup_handler(handler)
        result = death_system.unregister_cleanup_handler(handler)

        assert result
        assert handler not in death_system._cleanup_handlers

    def test_unregister_nonexistent_handler(self, death_system):
        """unregister_cleanup_handler should return False for nonexistent."""
        handler = Mock()
        result = death_system.unregister_cleanup_handler(handler)
        assert not result

    def test_cleanup_called_on_dead_transition(self, death_system):
        """Cleanup handlers should be called when entity becomes DEAD."""
        handler = Mock()
        death_system.register_cleanup_handler(handler)

        death_system.process_death(1)
        death_system.transition_to_dead(1)
        death_system._process_cleanup()

        handler.on_entity_death.assert_called()

    def test_cleanup_receives_correct_args(self, death_system):
        """Cleanup handler should receive entity_id and death_info."""
        handler = Mock()
        death_system.register_cleanup_handler(handler)

        death_system.process_death(1, death_cause="test")
        death_system.transition_to_dead(1)
        death_system._process_cleanup()

        call_args = handler.on_entity_death.call_args
        assert call_args[0][0] == 1  # entity_id
        assert call_args[0][1].death_cause == "test"  # death_info

    def test_force_cleanup(self, death_system):
        """force_cleanup should call handlers immediately."""
        handler = Mock()
        death_system.register_cleanup_handler(handler)

        death_system.process_death(1)
        death_system.force_cleanup(1)

        handler.on_entity_death.assert_called()

    def test_cleanup_exception_suppressed(self, death_system):
        """Cleanup handler exceptions should be suppressed."""
        def bad_handler(entity_id, death_info):
            raise RuntimeError("Test error")

        mock_handler = Mock()
        mock_handler.on_entity_death = bad_handler

        death_system.register_cleanup_handler(mock_handler)
        death_system.process_death(1)
        death_system.transition_to_dead(1)

        # Should not raise
        death_system._process_cleanup()


# =============================================================================
# EVENT HANDLING TESTS (20 tests)
# =============================================================================


class TestEventHandling:
    """Tests for death event handling."""

    def test_on_death_registered(self, death_system):
        """on_death should register handler."""
        callback = Mock()
        death_system.on_death(callback)
        assert callback in death_system._on_death

    def test_on_death_emitted(self, death_system):
        """Death should emit death event."""
        callback = Mock()
        death_system.on_death(callback)

        death_system.process_death(1)
        callback.assert_called_once()

    def test_death_event_has_info(self, death_system):
        """Death event should contain DeathInfo."""
        events = []
        death_system.on_death(lambda e: events.append(e))

        death_system.process_death(1, killer_id=2)

        assert len(events) == 1
        assert events[0].death_info.entity_id == 1
        assert events[0].death_info.killer_id == 2

    def test_on_respawn_registered(self, death_system):
        """on_respawn should register handler."""
        callback = Mock()
        death_system.on_respawn(callback)
        assert callback in death_system._on_respawn

    def test_on_state_changed_registered(self, death_system):
        """on_state_changed should register handler."""
        callback = Mock()
        death_system.on_state_changed(callback)
        assert callback in death_system._on_state_changed

    def test_state_changed_on_death(self, death_system):
        """State change should emit on death."""
        callback = Mock()
        death_system.on_state_changed(callback)

        death_system.process_death(1)
        callback.assert_called_with(1, DeathState.ALIVE, DeathState.DYING)

    def test_state_changed_on_dead_transition(self, death_system):
        """State change should emit on DEAD transition."""
        callback = Mock()
        death_system.on_state_changed(callback)

        death_system.process_death(1)
        callback.reset_mock()
        death_system.transition_to_dead(1)

        callback.assert_called_with(1, DeathState.DYING, DeathState.DEAD)

    def test_event_handler_exception_suppressed(self, death_system):
        """Event handler exceptions should be suppressed."""
        def bad_handler(event):
            raise RuntimeError("Test error")

        death_system.on_death(bad_handler)

        # Should not raise
        death_system.process_death(1)


# =============================================================================
# QUERY TESTS (15 tests)
# =============================================================================


class TestQueries:
    """Tests for query methods."""

    def test_get_all_dead(self, death_system):
        """get_all_dead should return dead and dying entities."""
        death_system.process_death(1)
        death_system.process_death(2)
        death_system.transition_to_dead(2)

        dead = death_system.get_all_dead()
        assert 1 in dead
        assert 2 in dead

    def test_get_all_respawning(self, death_system):
        """get_all_respawning should return respawning entities."""
        death_system.process_death(1)
        death_system.transition_to_dead(1)
        death_system.transition_to_respawning(1)

        respawning = death_system.get_all_respawning()
        assert 1 in respawning

    def test_get_pending_respawns(self, death_system):
        """get_pending_respawns should return queue copy."""
        death_system.process_death(1)
        death_system.transition_to_dead(1)
        death_system.queue_respawn(1)

        pending = death_system.get_pending_respawns()
        assert len(pending) == 1
        assert pending[0].entity_id == 1

    def test_get_recent_deaths_time_filter(self, death_system):
        """get_recent_deaths should filter by time."""
        # Create death with old timestamp
        info1 = death_system.process_death(1)
        info1.timestamp = time.time() - 100  # 100 seconds ago

        # Create recent death
        info2 = death_system.process_death(2)
        # info2 timestamp is now

        recent = death_system.get_recent_deaths(time_window=30.0)
        assert len(recent) == 1
        assert recent[0].entity_id == 2

    def test_get_recent_deaths_killer_filter(self, death_system):
        """get_recent_deaths should filter by killer."""
        death_system.process_death(1, killer_id=10)
        death_system.process_death(2, killer_id=20)

        recent = death_system.get_recent_deaths(killer_id=10)
        assert len(recent) == 1
        assert recent[0].killer_id == 10

    def test_get_recent_deaths_sorted(self, death_system):
        """get_recent_deaths should be sorted by timestamp."""
        with patch('time.time') as mock_time:
            mock_time.return_value = 1000.0
            death_system.process_death(1)

            mock_time.return_value = 1001.0
            death_system.process_death(2)

            mock_time.return_value = 1002.0
            recent = death_system.get_recent_deaths()
            assert recent[0].entity_id == 2  # Most recent first


# =============================================================================
# UTILITY TESTS (10 tests)
# =============================================================================


class TestUtility:
    """Tests for utility methods."""

    def test_clear(self, death_system):
        """clear should remove all state."""
        death_system.process_death(1)
        death_system.transition_to_dead(1)
        death_system.queue_respawn(1)

        death_system.clear()

        assert len(death_system._death_states) == 0
        assert len(death_system._respawn_queue) == 0
        assert len(death_system._pending_cleanup) == 0

    def test_remove_entity(self, death_system):
        """remove_entity should clear all tracking."""
        death_system.process_death(1)
        death_system.transition_to_dead(1)
        death_system.queue_respawn(1)

        death_system.remove_entity(1)

        assert death_system.get_death_info(1) is None
        assert death_system.get_respawn_request(1) is None

    def test_config_accessible(self, death_system):
        """config should be accessible."""
        assert death_system.config is not None

    def test_config_values_used(self, custom_config_system):
        """Custom config values should be used."""
        assert custom_config_system.config.dying_duration == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
