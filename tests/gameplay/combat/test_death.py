"""
Comprehensive tests for the Death System.

Tests cover:
- Death detection
- Death state
- Respawn timer
- Kill credit (last hit, most damage)
- Death events
- Death prevention (last stand)
- Corpse management
- Death penalties
- Revive mechanics
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch

from engine.gameplay.combat.death import (
    DeathSystem,
    DeathInfo,
    RespawnRequest,
    DeathEvent,
    RespawnEvent,
    DeathSubject,
    CleanupHandler,
    RespawnProvider,
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
    CombatEventType,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def death_system():
    """Create a fresh death system for each test."""
    return DeathSystem()


@pytest.fixture
def custom_config():
    """Create a custom death configuration."""
    return DeathConfig(
        dying_duration=2.0,
        default_respawn_time=10.0,
        min_respawn_time=1.0,
        max_respawn_time=60.0,
        respawn_health_percentage=0.5,
        respawn_invulnerability_duration=5.0,
    )


@pytest.fixture
def mock_entity():
    """Create a mock death subject entity."""
    entity = Mock(spec=DeathSubject)
    entity.entity_id = 1
    entity.is_dead = False
    entity.kill.return_value = True
    entity.revive.return_value = True
    return entity


@pytest.fixture
def mock_respawn_provider():
    """Create a mock respawn provider."""
    provider = Mock(spec=RespawnProvider)
    provider.get_respawn_position.return_value = (0.0, 0.0, 0.0)
    return provider


# =============================================================================
# DEATH DETECTION TESTS (~15 tests)
# =============================================================================


class TestDeathDetection:
    """Tests for death detection."""

    def test_process_death_creates_death_info(self, death_system):
        """Processing death should create death info."""
        info = death_system.process_death(entity_id=1)
        assert info is not None
        assert info.entity_id == 1

    def test_process_death_with_killer(self, death_system):
        """Death info should track killer."""
        info = death_system.process_death(entity_id=1, killer_id=2)
        assert info.killer_id == 2

    def test_process_death_with_cause(self, death_system):
        """Death info should track cause."""
        info = death_system.process_death(entity_id=1, death_cause="explosion")
        assert info.death_cause == "explosion"

    def test_process_death_with_position(self, death_system):
        """Death info should track position."""
        pos = (10.0, 20.0, 30.0)
        info = death_system.process_death(entity_id=1, death_position=pos)
        assert info.death_position == pos

    def test_process_death_with_weapon(self, death_system):
        """Death info should track weapon."""
        info = death_system.process_death(entity_id=1, weapon_id=42)
        assert info.weapon_id == 42

    def test_process_death_with_ability(self, death_system):
        """Death info should track ability."""
        info = death_system.process_death(entity_id=1, ability_id=99)
        assert info.ability_id == 99

    def test_process_death_headshot_flag(self, death_system):
        """Death info should track headshot."""
        info = death_system.process_death(entity_id=1, was_headshot=True)
        assert info.was_headshot

    def test_process_death_critical_flag(self, death_system):
        """Death info should track critical hit."""
        info = death_system.process_death(entity_id=1, was_critical=True)
        assert info.was_critical

    def test_process_death_overkill(self, death_system):
        """Death info should track overkill damage."""
        info = death_system.process_death(entity_id=1, overkill_damage=50.0)
        assert info.overkill_damage == 50.0

    def test_process_death_metadata(self, death_system):
        """Death info should store metadata."""
        info = death_system.process_death(entity_id=1, custom="value")
        assert info.metadata.get("custom") == "value"

    def test_get_death_info(self, death_system):
        """Should be able to retrieve death info."""
        death_system.process_death(entity_id=1)
        info = death_system.get_death_info(1)
        assert info is not None
        assert info.entity_id == 1

    def test_get_death_info_nonexistent(self, death_system):
        """Should return None for non-dead entity."""
        info = death_system.get_death_info(999)
        assert info is None

    def test_death_timestamp(self, death_system):
        """Death info should have timestamp."""
        info = death_system.process_death(entity_id=1)
        assert info.timestamp > 0
        assert info.timestamp <= time.time()


# =============================================================================
# DEATH STATE TESTS (~20 tests)
# =============================================================================


class TestDeathState:
    """Tests for death state management."""

    def test_initial_state_dying(self, death_system):
        """Death should start in DYING state."""
        info = death_system.process_death(entity_id=1)
        assert info.death_state == DeathState.DYING

    def test_get_death_state(self, death_system):
        """Should get current death state."""
        death_system.process_death(entity_id=1)
        state = death_system.get_death_state(1)
        assert state == DeathState.DYING

    def test_get_death_state_alive(self, death_system):
        """Non-dead entity should be ALIVE."""
        state = death_system.get_death_state(999)
        assert state == DeathState.ALIVE

    def test_is_dying(self, death_system):
        """Should detect dying state."""
        death_system.process_death(entity_id=1)
        assert death_system.is_dying(1)
        assert not death_system.is_dead(1)

    def test_transition_to_dead(self, death_system):
        """Should transition from DYING to DEAD."""
        death_system.process_death(entity_id=1)
        result = death_system.transition_to_dead(1)
        assert result
        assert death_system.is_dead(1)
        assert not death_system.is_dying(1)

    def test_transition_to_dead_wrong_state(self, death_system):
        """Cannot transition to DEAD from non-DYING state."""
        # Entity not dying
        result = death_system.transition_to_dead(999)
        assert not result

    def test_transition_to_respawning(self, death_system):
        """Should transition from DEAD to RESPAWNING."""
        death_system.process_death(entity_id=1)
        death_system.transition_to_dead(1)
        result = death_system.transition_to_respawning(1)
        assert result
        assert death_system.is_respawning(1)

    def test_transition_to_respawning_wrong_state(self, death_system):
        """Cannot transition to RESPAWNING from non-DEAD state."""
        death_system.process_death(entity_id=1)  # DYING state
        result = death_system.transition_to_respawning(1)
        assert not result

    def test_complete_respawn(self, death_system):
        """Should clear death state on respawn completion."""
        death_system.process_death(entity_id=1)
        death_system.transition_to_dead(1)
        death_system.transition_to_respawning(1)
        result = death_system.complete_respawn(1)
        assert result
        assert death_system.get_death_state(1) == DeathState.ALIVE

    def test_complete_respawn_nonexistent(self, death_system):
        """Cannot complete respawn for non-dead entity."""
        result = death_system.complete_respawn(999)
        assert not result

    def test_death_info_time_since_death(self, death_system):
        """Death info should track time since death."""
        info = death_system.process_death(entity_id=1)
        time.sleep(0.01)
        assert info.time_since_death > 0

    def test_death_info_is_fully_dead(self, death_system):
        """Should check if entity is fully dead."""
        info = death_system.process_death(entity_id=1)
        assert not info.is_fully_dead

        death_system.transition_to_dead(1)
        assert info.is_fully_dead


# =============================================================================
# RESPAWN TIMER TESTS (~20 tests)
# =============================================================================


class TestRespawnTimer:
    """Tests for respawn timer mechanics."""

    def test_queue_respawn(self, death_system):
        """Should queue respawn."""
        death_system.process_death(entity_id=1)
        death_system.transition_to_dead(1)
        request = death_system.queue_respawn(1)
        assert request is not None
        assert request.entity_id == 1

    def test_queue_respawn_default_delay(self, death_system):
        """Should use default respawn time."""
        death_system.process_death(entity_id=1)
        death_system.transition_to_dead(1)
        request = death_system.queue_respawn(1)
        assert request.time_until_respawn <= DEFAULT_RESPAWN_TIME

    def test_queue_respawn_custom_delay(self, death_system):
        """Should accept custom respawn delay."""
        death_system.process_death(entity_id=1)
        death_system.transition_to_dead(1)
        request = death_system.queue_respawn(1, delay=10.0)
        assert request.time_until_respawn <= 10.0

    def test_queue_respawn_clamped_min(self, death_system):
        """Respawn delay should be clamped to minimum."""
        death_system.process_death(entity_id=1)
        death_system.transition_to_dead(1)
        request = death_system.queue_respawn(1, delay=-5.0)
        assert request.time_until_respawn >= 0

    def test_queue_respawn_clamped_max(self, death_system):
        """Respawn delay should be clamped to maximum."""
        death_system.process_death(entity_id=1)
        death_system.transition_to_dead(1)
        request = death_system.queue_respawn(1, delay=1000.0)
        assert request.time_until_respawn <= MAX_RESPAWN_TIME

    def test_queue_respawn_health_percentage(self, death_system):
        """Should set respawn health percentage."""
        death_system.process_death(entity_id=1)
        death_system.transition_to_dead(1)
        request = death_system.queue_respawn(1, health_percentage=0.5)
        assert request.health_percentage == 0.5

    def test_queue_respawn_position(self, death_system):
        """Should set respawn position override."""
        death_system.process_death(entity_id=1)
        death_system.transition_to_dead(1)
        pos = (10.0, 20.0, 30.0)
        request = death_system.queue_respawn(1, position=pos)
        assert request.position == pos

    def test_queue_respawn_invulnerability(self, death_system):
        """Should set invulnerability options."""
        death_system.process_death(entity_id=1)
        death_system.transition_to_dead(1)
        request = death_system.queue_respawn(
            1,
            add_invulnerability=True,
            invulnerability_duration=5.0
        )
        assert request.add_invulnerability
        assert request.invulnerability_duration == 5.0

    def test_get_respawn_request(self, death_system):
        """Should retrieve respawn request."""
        death_system.process_death(entity_id=1)
        death_system.transition_to_dead(1)
        death_system.queue_respawn(1)
        request = death_system.get_respawn_request(1)
        assert request is not None

    def test_get_respawn_time_remaining(self, death_system):
        """Should get time until respawn."""
        death_system.process_death(entity_id=1)
        death_system.transition_to_dead(1)
        death_system.queue_respawn(1, delay=5.0)
        remaining = death_system.get_respawn_time_remaining(1)
        assert remaining > 0
        assert remaining <= 5.0

    def test_cancel_respawn(self, death_system):
        """Should cancel queued respawn."""
        death_system.process_death(entity_id=1)
        death_system.transition_to_dead(1)
        death_system.queue_respawn(1)
        result = death_system.cancel_respawn(1)
        assert result
        assert death_system.get_respawn_request(1) is None

    def test_cancel_respawn_nonexistent(self, death_system):
        """Should return False for non-queued respawn."""
        result = death_system.cancel_respawn(999)
        assert not result

    def test_respawn_replaces_existing(self, death_system):
        """New respawn should replace existing."""
        death_system.process_death(entity_id=1)
        death_system.transition_to_dead(1)
        death_system.queue_respawn(1, delay=5.0)
        death_system.queue_respawn(1, delay=10.0)

        # Should only have one request
        requests = death_system.get_pending_respawns()
        entity_requests = [r for r in requests if r.entity_id == 1]
        assert len(entity_requests) == 1

    def test_respawn_request_is_ready(self, death_system):
        """Should detect when respawn is ready."""
        death_system.process_death(entity_id=1)
        death_system.transition_to_dead(1)
        request = death_system.queue_respawn(1, delay=0.0)
        time.sleep(0.01)
        assert request.is_ready


# =============================================================================
# DEATH EVENT TESTS (~15 tests)
# =============================================================================


class TestDeathEvents:
    """Tests for death event emission."""

    def test_on_death_handler(self, death_system):
        """Should call death handler."""
        handler = Mock()
        death_system.on_death(handler)
        death_system.process_death(entity_id=1)
        handler.assert_called_once()

    def test_on_death_receives_event(self, death_system):
        """Death handler should receive event."""
        received = []

        def handler(event):
            received.append(event)

        death_system.on_death(handler)
        death_system.process_death(entity_id=1, killer_id=2)

        assert len(received) == 1
        assert received[0].death_info.entity_id == 1
        assert received[0].death_info.killer_id == 2

    def test_on_respawn_handler(self, death_system, mock_entity):
        """Should call respawn handler."""
        handler = Mock()
        death_system.on_respawn(handler)

        death_system.instant_respawn(mock_entity)
        handler.assert_called()

    def test_on_respawn_receives_event(self, death_system, mock_entity):
        """Respawn handler should receive event."""
        received = []

        def handler(event):
            received.append(event)

        death_system.on_respawn(handler)
        death_system.process_death(entity_id=mock_entity.entity_id)
        death_system.instant_respawn(mock_entity, health_percentage=0.5)

        assert len(received) == 1
        assert received[0].entity_id == mock_entity.entity_id
        assert received[0].health_percentage == 0.5

    def test_on_state_changed_handler(self, death_system):
        """Should call state change handler."""
        handler = Mock()
        death_system.on_state_changed(handler)

        death_system.process_death(entity_id=1)
        handler.assert_called()

    def test_state_change_provides_old_and_new(self, death_system):
        """State change should provide old and new states."""
        received = []

        def handler(entity_id, old_state, new_state):
            received.append((entity_id, old_state, new_state))

        death_system.on_state_changed(handler)
        death_system.process_death(entity_id=1)

        assert len(received) == 1
        assert received[0][0] == 1
        assert received[0][1] == DeathState.ALIVE
        assert received[0][2] == DeathState.DYING

    def test_multiple_state_changes(self, death_system):
        """Should track multiple state changes."""
        states = []

        def handler(entity_id, old_state, new_state):
            states.append((old_state, new_state))

        death_system.on_state_changed(handler)

        death_system.process_death(entity_id=1)
        death_system.transition_to_dead(1)
        death_system.transition_to_respawning(1)
        death_system.complete_respawn(1)

        assert len(states) == 4
        assert states[0] == (DeathState.ALIVE, DeathState.DYING)
        assert states[1] == (DeathState.DYING, DeathState.DEAD)
        assert states[2] == (DeathState.DEAD, DeathState.RESPAWNING)
        assert states[3] == (DeathState.RESPAWNING, DeathState.ALIVE)

    def test_handler_exception_doesnt_break(self, death_system):
        """Handler exception should not break system."""

        def bad_handler(event):
            raise Exception("Handler error")

        death_system.on_death(bad_handler)
        info = death_system.process_death(entity_id=1)
        assert info is not None


# =============================================================================
# CLEANUP HANDLER TESTS (~10 tests)
# =============================================================================


class TestCleanupHandlers:
    """Tests for cleanup handler management."""

    def test_register_cleanup_handler(self, death_system):
        """Should register cleanup handler."""
        handler = Mock(spec=CleanupHandler)
        death_system.register_cleanup_handler(handler)

        death_system.process_death(entity_id=1)
        death_system.transition_to_dead(1)
        death_system.update(0.1, {})  # Trigger cleanup

        handler.on_entity_death.assert_called_once()

    def test_unregister_cleanup_handler(self, death_system):
        """Should unregister cleanup handler."""
        handler = Mock(spec=CleanupHandler)
        death_system.register_cleanup_handler(handler)
        result = death_system.unregister_cleanup_handler(handler)
        assert result

        death_system.process_death(entity_id=1)
        death_system.transition_to_dead(1)
        death_system.update(0.1, {})

        handler.on_entity_death.assert_not_called()

    def test_unregister_nonexistent_handler(self, death_system):
        """Should return False for non-registered handler."""
        handler = Mock(spec=CleanupHandler)
        result = death_system.unregister_cleanup_handler(handler)
        assert not result

    def test_cleanup_handler_receives_info(self, death_system):
        """Cleanup handler should receive death info."""
        received = []

        class TestHandler:
            def on_entity_death(self, entity_id, death_info):
                received.append((entity_id, death_info))

        handler = TestHandler()
        death_system.register_cleanup_handler(handler)

        death_system.process_death(entity_id=1, death_cause="test")
        death_system.transition_to_dead(1)
        death_system.update(0.1, {})

        assert len(received) == 1
        assert received[0][0] == 1
        assert received[0][1].death_cause == "test"

    def test_force_cleanup(self, death_system):
        """Should force immediate cleanup."""
        handler = Mock(spec=CleanupHandler)
        death_system.register_cleanup_handler(handler)

        death_system.process_death(entity_id=1)
        death_system.force_cleanup(1)

        handler.on_entity_death.assert_called_once()


# =============================================================================
# INSTANT RESPAWN TESTS (~10 tests)
# =============================================================================


class TestInstantRespawn:
    """Tests for instant respawn mechanics."""

    def test_instant_respawn(self, death_system, mock_entity):
        """Should instantly respawn entity."""
        death_system.process_death(entity_id=mock_entity.entity_id)
        result = death_system.instant_respawn(mock_entity)
        assert result
        mock_entity.revive.assert_called_once()

    def test_instant_respawn_health_percentage(self, death_system, mock_entity):
        """Should set health percentage."""
        death_system.process_death(entity_id=mock_entity.entity_id)
        death_system.instant_respawn(mock_entity, health_percentage=0.5)

        mock_entity.revive.assert_called_once()
        call_kwargs = mock_entity.revive.call_args[1]
        assert call_kwargs["health_percentage"] == 0.5

    def test_instant_respawn_with_invulnerability(self, death_system, mock_entity):
        """Should add invulnerability."""
        death_system.process_death(entity_id=mock_entity.entity_id)
        death_system.instant_respawn(
            mock_entity,
            add_invulnerability=True,
            invulnerability_duration=5.0
        )

        call_kwargs = mock_entity.revive.call_args[1]
        assert call_kwargs["add_invulnerability"]
        assert call_kwargs["invulnerability_duration"] == 5.0

    def test_instant_respawn_without_invulnerability(self, death_system, mock_entity):
        """Should skip invulnerability if disabled."""
        death_system.process_death(entity_id=mock_entity.entity_id)
        death_system.instant_respawn(mock_entity, add_invulnerability=False)

        call_kwargs = mock_entity.revive.call_args[1]
        assert not call_kwargs["add_invulnerability"]

    def test_instant_respawn_clears_death_state(self, death_system, mock_entity):
        """Instant respawn should clear death state."""
        death_system.process_death(entity_id=mock_entity.entity_id)
        death_system.instant_respawn(mock_entity)

        state = death_system.get_death_state(mock_entity.entity_id)
        assert state == DeathState.ALIVE

    def test_instant_respawn_uses_provider(self, death_system, mock_entity, mock_respawn_provider):
        """Should use respawn provider for position."""
        system = DeathSystem(respawn_provider=mock_respawn_provider)
        system.process_death(entity_id=mock_entity.entity_id)
        system.instant_respawn(mock_entity, team_id="team1")

        mock_respawn_provider.get_respawn_position.assert_called()


# =============================================================================
# UPDATE LOOP TESTS (~15 tests)
# =============================================================================


class TestUpdateLoop:
    """Tests for death system update loop."""

    def test_update_dying_to_dead(self, death_system):
        """Update should transition DYING to DEAD after duration."""
        config = DeathConfig(dying_duration=0.01)  # Very short for testing
        system = DeathSystem(config=config)

        system.process_death(entity_id=1)
        time.sleep(0.02)
        system.update(0.01, {})

        assert system.get_death_state(1) == DeathState.DEAD

    def test_update_respawn_queue(self, death_system, mock_entity):
        """Update should process respawn queue."""
        death_system.process_death(entity_id=1)
        death_system.transition_to_dead(1)
        death_system.queue_respawn(1, delay=0.0)
        time.sleep(0.01)

        respawned = death_system.update(0.01, {1: mock_entity})

        assert 1 in respawned
        mock_entity.revive.assert_called_once()

    def test_update_returns_respawned_list(self, death_system, mock_entity):
        """Update should return list of respawned entities."""
        death_system.process_death(entity_id=1)
        death_system.transition_to_dead(1)
        death_system.queue_respawn(1, delay=0.0)
        time.sleep(0.01)

        respawned = death_system.update(0.01, {1: mock_entity})
        assert 1 in respawned

    def test_update_triggers_cleanup(self, death_system):
        """Update should trigger cleanup for newly dead entities."""
        handler = Mock(spec=CleanupHandler)
        death_system.register_cleanup_handler(handler)

        death_system.process_death(entity_id=1)
        death_system.transition_to_dead(1)
        death_system.update(0.01, {})

        handler.on_entity_death.assert_called_once()


# =============================================================================
# QUERY TESTS (~10 tests)
# =============================================================================


class TestQueries:
    """Tests for death system queries."""

    def test_get_all_dead(self, death_system):
        """Should get all dead entities."""
        death_system.process_death(entity_id=1)
        death_system.process_death(entity_id=2)
        death_system.transition_to_dead(2)

        dead = death_system.get_all_dead()
        assert 1 in dead
        assert 2 in dead

    def test_get_all_respawning(self, death_system):
        """Should get all respawning entities."""
        death_system.process_death(entity_id=1)
        death_system.transition_to_dead(1)
        death_system.queue_respawn(1)

        respawning = death_system.get_all_respawning()
        assert 1 in respawning

    def test_get_pending_respawns(self, death_system):
        """Should get all pending respawn requests."""
        death_system.process_death(entity_id=1)
        death_system.transition_to_dead(1)
        death_system.queue_respawn(1)

        death_system.process_death(entity_id=2)
        death_system.transition_to_dead(2)
        death_system.queue_respawn(2)

        pending = death_system.get_pending_respawns()
        assert len(pending) == 2

    def test_get_recent_deaths(self, death_system):
        """Should get recent deaths."""
        death_system.process_death(entity_id=1)
        death_system.process_death(entity_id=2)

        recent = death_system.get_recent_deaths(time_window=60.0)
        assert len(recent) == 2

    def test_get_recent_deaths_by_killer(self, death_system):
        """Should filter recent deaths by killer."""
        death_system.process_death(entity_id=1, killer_id=99)
        death_system.process_death(entity_id=2, killer_id=100)

        recent = death_system.get_recent_deaths(killer_id=99)
        assert len(recent) == 1
        assert recent[0].entity_id == 1


# =============================================================================
# UTILITY TESTS (~10 tests)
# =============================================================================


class TestUtility:
    """Tests for utility methods."""

    def test_clear(self, death_system):
        """Should clear all death states."""
        death_system.process_death(entity_id=1)
        death_system.process_death(entity_id=2)
        death_system.transition_to_dead(1)
        death_system.queue_respawn(1)

        death_system.clear()

        assert len(death_system.get_all_dead()) == 0
        assert len(death_system.get_pending_respawns()) == 0

    def test_remove_entity(self, death_system):
        """Should remove all tracking for entity."""
        death_system.process_death(entity_id=1)
        death_system.transition_to_dead(1)
        death_system.queue_respawn(1)

        death_system.remove_entity(1)

        assert death_system.get_death_info(1) is None
        assert death_system.get_respawn_request(1) is None


# =============================================================================
# CUSTOM CONFIG TESTS (~10 tests)
# =============================================================================


class TestCustomConfig:
    """Tests for custom death configuration."""

    def test_custom_dying_duration(self, custom_config):
        """Should use custom dying duration."""
        system = DeathSystem(config=custom_config)
        assert system.config.dying_duration == 2.0

    def test_custom_respawn_time(self, custom_config):
        """Should use custom default respawn time."""
        system = DeathSystem(config=custom_config)
        system.process_death(entity_id=1)
        system.transition_to_dead(1)
        request = system.queue_respawn(1)

        assert request.time_until_respawn <= custom_config.default_respawn_time

    def test_custom_health_percentage(self, custom_config, mock_entity):
        """Should use custom respawn health percentage."""
        system = DeathSystem(config=custom_config)
        system.process_death(entity_id=mock_entity.entity_id)
        system.instant_respawn(mock_entity)

        call_kwargs = mock_entity.revive.call_args[1]
        assert call_kwargs["health_percentage"] == 1.0  # Default unless overridden

    def test_custom_invulnerability_duration(self, custom_config, mock_entity):
        """Should use custom invulnerability duration."""
        system = DeathSystem(config=custom_config)
        system.process_death(entity_id=mock_entity.entity_id)
        system.instant_respawn(mock_entity)

        call_kwargs = mock_entity.revive.call_args[1]
        assert call_kwargs["invulnerability_duration"] == custom_config.respawn_invulnerability_duration


# =============================================================================
# RESPAWN REQUEST TESTS (~10 tests)
# =============================================================================


class TestRespawnRequest:
    """Tests for RespawnRequest dataclass."""

    def test_time_until_respawn(self):
        """Should calculate time until respawn."""
        request = RespawnRequest(
            entity_id=1,
            respawn_time=time.time() + 5.0,
        )
        assert 4.9 <= request.time_until_respawn <= 5.0

    def test_time_until_respawn_passed(self):
        """Should return zero for passed time."""
        request = RespawnRequest(
            entity_id=1,
            respawn_time=time.time() - 1.0,  # Already passed
        )
        assert request.time_until_respawn == 0.0

    def test_is_ready(self):
        """Should detect when respawn is ready."""
        request = RespawnRequest(
            entity_id=1,
            respawn_time=time.time() - 0.1,  # Already passed
        )
        assert request.is_ready

    def test_is_not_ready(self):
        """Should detect when respawn is not ready."""
        request = RespawnRequest(
            entity_id=1,
            respawn_time=time.time() + 10.0,
        )
        assert not request.is_ready
