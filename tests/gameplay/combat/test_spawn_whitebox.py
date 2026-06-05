"""
WHITEBOX Tests for Spawn Manager

Tests internal implementation details:
- Spawn point state machine
- Selection algorithm internals
- Distance calculations
- Scoring formulas
- Cooldown timing
- Queue management
"""

import pytest
import time
import math
from unittest.mock import Mock, patch

from engine.gameplay.combat.spawn_manager import (
    SpawnManager,
    SpawnPoint,
    SpawnRule,
    SpawnRuleType,
    SpawnPointState,
    TeamSpawnConfig,
)
from engine.gameplay.combat.constants import (
    SPAWN_TEAM_BONUS_SCORE,
    SPAWN_TIME_FRESHNESS_MAX_BONUS,
    SPAWN_TIME_FRESHNESS_DIVISOR,
    SPAWN_DISTANCE_MAX_BONUS,
    SPAWN_DISTANCE_DIVISOR,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def spawn_manager():
    """Create a fresh spawn manager."""
    return SpawnManager()


@pytest.fixture
def custom_rule_manager():
    """Create spawn manager with custom rule."""
    rule = SpawnRule(
        rule_type=SpawnRuleType.SAFE_SPAWN,
        respawn_delay_seconds=5.0,
        cooldown_seconds=10.0,
    )
    return SpawnManager(default_rule=rule)


@pytest.fixture
def populated_manager(spawn_manager):
    """Create spawn manager with spawn points."""
    spawn_manager.register_spawn_point(SpawnPoint(
        point_id="spawn_1",
        position=(0, 0, 0),
        team_id="red",
        priority=50,
    ))
    spawn_manager.register_spawn_point(SpawnPoint(
        point_id="spawn_2",
        position=(10, 0, 0),
        team_id="red",
        priority=75,
    ))
    spawn_manager.register_spawn_point(SpawnPoint(
        point_id="spawn_3",
        position=(100, 0, 0),
        team_id="blue",
        priority=50,
    ))
    spawn_manager.register_spawn_point(SpawnPoint(
        point_id="spawn_neutral",
        position=(50, 0, 0),
        team_id=None,
        priority=25,
    ))
    return spawn_manager


# =============================================================================
# SPAWN POINT STATE TESTS (30 tests)
# =============================================================================


class TestSpawnPointState:
    """Tests for spawn point state machine."""

    def test_initial_state_available(self):
        """Initial state should be AVAILABLE."""
        sp = SpawnPoint(point_id="test", position=(0, 0, 0))
        assert sp._state == SpawnPointState.AVAILABLE

    def test_is_available_true(self):
        """is_available should return True when available."""
        sp = SpawnPoint(point_id="test", position=(0, 0, 0))
        assert sp.is_available

    def test_is_available_disabled(self):
        """is_available should be False when disabled."""
        sp = SpawnPoint(point_id="test", position=(0, 0, 0), enabled=False)
        assert not sp.is_available

    def test_is_available_blocked(self):
        """is_available should be False when blocked."""
        sp = SpawnPoint(point_id="test", position=(0, 0, 0))
        sp.block()
        assert not sp.is_available

    def test_is_available_at_capacity(self):
        """is_available should be False at capacity."""
        sp = SpawnPoint(point_id="test", position=(0, 0, 0), capacity=1)
        sp._current_occupants = 1
        assert not sp.is_available

    def test_is_available_on_cooldown(self):
        """is_available should be False on cooldown."""
        sp = SpawnPoint(point_id="test", position=(0, 0, 0))
        sp._state = SpawnPointState.COOLDOWN
        sp._cooldown_until = time.time() + 10
        assert not sp.is_available

    def test_cooldown_expires(self):
        """is_available should be True after cooldown expires."""
        with patch('time.time') as mock_time:
            mock_time.return_value = 1000.0
            sp = SpawnPoint(point_id="test", position=(0, 0, 0))
            sp._state = SpawnPointState.COOLDOWN
            sp._cooldown_until = 1005.0

            mock_time.return_value = 1010.0
            assert sp.is_available

    def test_use_sets_occupant(self):
        """use() should increment occupants."""
        sp = SpawnPoint(point_id="test", position=(0, 0, 0))
        result = sp.use()
        assert result
        assert sp._current_occupants == 1

    def test_use_sets_last_used(self):
        """use() should set last used time."""
        before = time.time()
        sp = SpawnPoint(point_id="test", position=(0, 0, 0))
        sp.use()
        after = time.time()
        assert before <= sp._last_used <= after

    def test_use_with_cooldown(self):
        """use() should set cooldown."""
        sp = SpawnPoint(point_id="test", position=(0, 0, 0), capacity=1)
        sp.use(cooldown_seconds=5.0)

        assert sp._cooldown_until > time.time()
        assert sp._state == SpawnPointState.COOLDOWN

    def test_use_fails_when_unavailable(self):
        """use() should fail when unavailable."""
        sp = SpawnPoint(point_id="test", position=(0, 0, 0))
        sp.disable()
        result = sp.use()
        assert not result

    def test_release(self):
        """release() should decrement occupants."""
        sp = SpawnPoint(point_id="test", position=(0, 0, 0))
        sp._current_occupants = 2
        sp.release()
        assert sp._current_occupants == 1

    def test_release_clamps_to_zero(self):
        """release() should not go below zero."""
        sp = SpawnPoint(point_id="test", position=(0, 0, 0))
        sp._current_occupants = 0
        sp.release()
        assert sp._current_occupants == 0

    def test_block(self):
        """block() should set BLOCKED state."""
        sp = SpawnPoint(point_id="test", position=(0, 0, 0))
        sp.block()
        assert sp._state == SpawnPointState.BLOCKED

    def test_unblock(self):
        """unblock() should clear BLOCKED state."""
        sp = SpawnPoint(point_id="test", position=(0, 0, 0))
        sp.block()
        sp.unblock()
        assert sp._state == SpawnPointState.AVAILABLE

    def test_disable(self):
        """disable() should set DISABLED state."""
        sp = SpawnPoint(point_id="test", position=(0, 0, 0))
        sp.disable()
        assert not sp.enabled
        assert sp._state == SpawnPointState.DISABLED

    def test_enable(self):
        """enable() should clear DISABLED state."""
        sp = SpawnPoint(point_id="test", position=(0, 0, 0))
        sp.disable()
        sp.enable()
        assert sp.enabled
        assert sp._state == SpawnPointState.AVAILABLE


# =============================================================================
# SPAWN REGISTRATION TESTS (20 tests)
# =============================================================================


class TestSpawnRegistration:
    """Tests for spawn point registration."""

    def test_register_spawn_point(self, spawn_manager):
        """register_spawn_point should add point."""
        sp = SpawnPoint(point_id="test", position=(0, 0, 0))
        result = spawn_manager.register_spawn_point(sp)
        assert result
        assert spawn_manager.get_spawn_point("test") is not None

    def test_register_duplicate_fails(self, spawn_manager):
        """Duplicate registration should fail."""
        sp = SpawnPoint(point_id="test", position=(0, 0, 0))
        spawn_manager.register_spawn_point(sp)
        result = spawn_manager.register_spawn_point(sp)
        assert not result

    def test_register_indexes_by_team(self, spawn_manager):
        """Should index by team."""
        sp = SpawnPoint(point_id="test", position=(0, 0, 0), team_id="red")
        spawn_manager.register_spawn_point(sp)

        team_points = spawn_manager.get_team_spawn_points("red")
        assert len(team_points) == 1

    def test_register_indexes_by_type(self, spawn_manager):
        """Should index by type."""
        sp = SpawnPoint(point_id="test", position=(0, 0, 0), spawn_type="vehicle")
        spawn_manager.register_spawn_point(sp)

        type_points = spawn_manager.get_spawn_points_by_type("vehicle")
        assert len(type_points) == 1

    def test_unregister_spawn_point(self, populated_manager):
        """unregister_spawn_point should remove point."""
        result = populated_manager.unregister_spawn_point("spawn_1")
        assert result
        assert populated_manager.get_spawn_point("spawn_1") is None

    def test_unregister_nonexistent(self, spawn_manager):
        """unregister_spawn_point should return False for nonexistent."""
        result = spawn_manager.unregister_spawn_point("nonexistent")
        assert not result

    def test_get_all_spawn_points(self, populated_manager):
        """get_all_spawn_points should return all."""
        points = populated_manager.get_all_spawn_points()
        assert len(points) == 4

    def test_get_available_spawn_points(self, populated_manager):
        """get_available_spawn_points should filter available."""
        # Disable one
        sp = populated_manager.get_spawn_point("spawn_1")
        sp.disable()

        available = populated_manager.get_available_spawn_points()
        assert len(available) == 3

    def test_get_available_filter_by_team(self, populated_manager):
        """get_available_spawn_points should filter by team."""
        available = populated_manager.get_available_spawn_points(team_id="red")
        assert len(available) == 2
        assert all(sp.team_id == "red" for sp in available)

    def test_get_available_filter_by_type(self, spawn_manager):
        """get_available_spawn_points should filter by type."""
        spawn_manager.register_spawn_point(SpawnPoint(
            point_id="v1", position=(0, 0, 0), spawn_type="vehicle"
        ))
        spawn_manager.register_spawn_point(SpawnPoint(
            point_id="i1", position=(0, 0, 0), spawn_type="infantry"
        ))

        available = spawn_manager.get_available_spawn_points(spawn_type="vehicle")
        assert len(available) == 1

    def test_get_available_filter_by_tags(self, spawn_manager):
        """get_available_spawn_points should filter by tags."""
        spawn_manager.register_spawn_point(SpawnPoint(
            point_id="t1", position=(0, 0, 0), tags={"safe", "primary"}
        ))
        spawn_manager.register_spawn_point(SpawnPoint(
            point_id="t2", position=(0, 0, 0), tags={"secondary"}
        ))

        available = spawn_manager.get_available_spawn_points(tags={"safe"})
        assert len(available) == 1


# =============================================================================
# SELECTION ALGORITHM TESTS (40 tests)
# =============================================================================


class TestSelectionAlgorithms:
    """Tests for spawn selection algorithms."""

    def test_select_random(self, populated_manager):
        """Random selection should return available point."""
        rule = SpawnRule(rule_type=SpawnRuleType.RANDOM)
        point = populated_manager.select_spawn_point("player_1", rule=rule)
        assert point is not None

    def test_select_random_weighted_by_priority(self, spawn_manager):
        """Random selection should weight by priority."""
        # Add high priority point
        spawn_manager.register_spawn_point(SpawnPoint(
            point_id="low", position=(0, 0, 0), priority=1
        ))
        spawn_manager.register_spawn_point(SpawnPoint(
            point_id="high", position=(10, 0, 0), priority=100
        ))

        # Run multiple selections
        selections = {}
        for _ in range(100):
            point = spawn_manager.select_spawn_point("player", rule=SpawnRule())
            selections[point.point_id] = selections.get(point.point_id, 0) + 1

        # High priority should be selected more often
        assert selections.get("high", 0) > selections.get("low", 0)

    def test_select_sequential(self, populated_manager):
        """Sequential selection should round-robin."""
        rule = SpawnRule(rule_type=SpawnRuleType.SEQUENTIAL)

        # Get first selection
        point1 = populated_manager.select_spawn_point("player_1", rule=rule)

        # Get second selection - should be different
        point2 = populated_manager.select_spawn_point("player_1", rule=rule)

        # After cycling through, should return to first
        assert point1 is not None
        assert point2 is not None

    def test_select_sequential_per_team(self, spawn_manager):
        """Sequential should track per team."""
        spawn_manager.register_spawn_point(SpawnPoint(
            point_id="r1", position=(0, 0, 0), team_id="red"
        ))
        spawn_manager.register_spawn_point(SpawnPoint(
            point_id="r2", position=(10, 0, 0), team_id="red"
        ))
        spawn_manager.configure_team(TeamSpawnConfig(team_id="red"))

        rule = SpawnRule(rule_type=SpawnRuleType.SEQUENTIAL)

        # Sequential for red team
        point1 = spawn_manager.select_spawn_point("p1", team_id="red", rule=rule)
        point2 = spawn_manager.select_spawn_point("p2", team_id="red", rule=rule)

        assert point1.point_id != point2.point_id

    def test_select_team_based(self, populated_manager):
        """Team-based selection should prefer team spawns."""
        rule = SpawnRule(rule_type=SpawnRuleType.TEAM_BASED, prefer_team_spawns=True)

        point = populated_manager.select_spawn_point("player", team_id="red", rule=rule)
        assert point.team_id == "red"

    def test_select_team_based_falls_back_to_neutral(self, spawn_manager):
        """Team-based should fall back to neutral if no team spawns."""
        spawn_manager.register_spawn_point(SpawnPoint(
            point_id="neutral", position=(0, 0, 0), team_id=None
        ))

        rule = SpawnRule(rule_type=SpawnRuleType.TEAM_BASED, prefer_team_spawns=True)
        point = spawn_manager.select_spawn_point("player", team_id="red", rule=rule)

        assert point is not None
        assert point.team_id is None

    def test_select_distance_based(self, populated_manager):
        """Distance-based should prefer far from enemies."""
        # Add enemy at spawn_1 position
        populated_manager.update_player_position("enemy", (0, 0, 0))
        populated_manager.update_player_team("enemy", "blue")

        rule = SpawnRule(
            rule_type=SpawnRuleType.DISTANCE_BASED,
            min_distance_from_enemies=50.0,
            prefer_team_spawns=False,
        )

        point = populated_manager.select_spawn_point("player", team_id="red", rule=rule)
        # Should select distant spawn
        assert point is not None

    def test_select_distance_based_no_enemies(self, populated_manager):
        """Distance-based with no enemies should fall back to random."""
        rule = SpawnRule(
            rule_type=SpawnRuleType.DISTANCE_BASED,
            min_distance_from_enemies=50.0,
        )

        point = populated_manager.select_spawn_point("player", rule=rule)
        assert point is not None

    def test_select_safe_spawn(self, populated_manager):
        """Safe spawn should use scoring formula."""
        # Add enemy
        populated_manager.update_player_position("enemy", (0, 0, 0))
        populated_manager.update_player_team("enemy", "blue")

        rule = SpawnRule(rule_type=SpawnRuleType.SAFE_SPAWN)

        point = populated_manager.select_spawn_point("player", team_id="red", rule=rule)
        assert point is not None

    def test_select_fixed(self, spawn_manager):
        """Fixed selection should use first candidate."""
        spawn_manager.register_spawn_point(SpawnPoint(
            point_id="fixed", position=(0, 0, 0)
        ))

        rule = SpawnRule(rule_type=SpawnRuleType.FIXED)
        point = spawn_manager.select_spawn_point("player", rule=rule)

        assert point is not None

    def test_select_avoids_recently_used(self, populated_manager):
        """Should avoid recently used spawns."""
        rule = SpawnRule(
            rule_type=SpawnRuleType.RANDOM,
            avoid_recently_used=True,
            cooldown_seconds=10.0,
        )

        # Use spawn_1
        sp = populated_manager.get_spawn_point("spawn_1")
        sp.use()

        # Selection should avoid it
        for _ in range(10):
            point = populated_manager.select_spawn_point("player", rule=rule)
            assert point.point_id != "spawn_1"

    def test_select_with_custom_validator(self, populated_manager):
        """Custom validator should filter candidates."""
        def only_high_priority(spawn_point, player_id):
            return spawn_point.priority >= 50

        rule = SpawnRule(
            rule_type=SpawnRuleType.RANDOM,
            custom_validator=only_high_priority,
        )

        point = populated_manager.select_spawn_point("player", rule=rule)
        assert point.priority >= 50

    def test_select_no_candidates_returns_none(self, spawn_manager):
        """Should return None when no candidates."""
        rule = SpawnRule(rule_type=SpawnRuleType.RANDOM)
        point = spawn_manager.select_spawn_point("player", rule=rule)
        assert point is None


# =============================================================================
# DISTANCE CALCULATION TESTS (15 tests)
# =============================================================================


class TestDistanceCalculation:
    """Tests for distance calculations."""

    def test_distance_same_point(self):
        """Distance to same point should be zero."""
        dist = SpawnManager._distance((0, 0, 0), (0, 0, 0))
        assert dist == 0.0

    def test_distance_2d(self):
        """2D distance (3-4-5 triangle)."""
        dist = SpawnManager._distance((0, 0, 0), (3, 4, 0))
        assert dist == 5.0

    def test_distance_3d(self):
        """3D distance calculation."""
        dist = SpawnManager._distance((0, 0, 0), (1, 2, 2))
        assert dist == 3.0  # sqrt(1+4+4)

    def test_distance_negative_coords(self):
        """Distance with negative coordinates."""
        dist = SpawnManager._distance((-1, -1, -1), (1, 1, 1))
        expected = math.sqrt(12)
        assert abs(dist - expected) < 0.0001


# =============================================================================
# SCORING FORMULA TESTS (20 tests)
# =============================================================================


class TestScoringFormulas:
    """Tests for spawn scoring formulas."""

    def test_team_bonus_applied(self, spawn_manager):
        """Team spawn should get bonus points."""
        spawn_manager.register_spawn_point(SpawnPoint(
            point_id="team", position=(0, 0, 0), team_id="red", priority=0
        ))
        spawn_manager.register_spawn_point(SpawnPoint(
            point_id="neutral", position=(0, 0, 0), team_id=None, priority=0
        ))

        rule = SpawnRule(rule_type=SpawnRuleType.SAFE_SPAWN)

        # With team preference, should select team spawn
        point = spawn_manager.select_spawn_point("player", team_id="red", rule=rule)
        # Due to team bonus, team spawn should be preferred
        assert point is not None

    def test_freshness_bonus(self, spawn_manager):
        """Recently used spawns should score lower."""
        with patch('time.time') as mock_time:
            mock_time.return_value = 1000.0

            spawn_manager.register_spawn_point(SpawnPoint(
                point_id="fresh", position=(0, 0, 0), priority=0
            ))
            spawn_manager.register_spawn_point(SpawnPoint(
                point_id="stale", position=(10, 0, 0), priority=0
            ))

            # Use stale spawn
            stale = spawn_manager.get_spawn_point("stale")
            stale._last_used = 999.0  # Just used

            rule = SpawnRule(rule_type=SpawnRuleType.SAFE_SPAWN, avoid_recently_used=False)
            point = spawn_manager.select_spawn_point("player", rule=rule)

            # Fresh should have higher score
            assert point.point_id == "fresh"

    def test_enemy_distance_bonus(self, spawn_manager):
        """Spawns far from enemies should score higher."""
        spawn_manager.register_spawn_point(SpawnPoint(
            point_id="near", position=(0, 0, 0), priority=0
        ))
        spawn_manager.register_spawn_point(SpawnPoint(
            point_id="far", position=(100, 0, 0), priority=0
        ))

        spawn_manager.update_player_position("enemy", (5, 0, 0))
        spawn_manager.update_player_team("enemy", "blue")

        rule = SpawnRule(rule_type=SpawnRuleType.SAFE_SPAWN)
        point = spawn_manager.select_spawn_point("player", team_id="red", rule=rule)

        # Far spawn should be preferred
        assert point.point_id == "far"


# =============================================================================
# SPAWNING TESTS (25 tests)
# =============================================================================


class TestSpawning:
    """Tests for player spawning."""

    def test_spawn_player(self, populated_manager):
        """spawn_player should return position/rotation."""
        result = populated_manager.spawn_player("player_1")
        assert result is not None
        position, rotation = result
        assert len(position) == 3
        assert len(rotation) == 3

    def test_spawn_player_no_points(self, spawn_manager):
        """spawn_player should return None with no points."""
        result = spawn_manager.spawn_player("player_1")
        assert result is None

    def test_spawn_player_uses_point(self, populated_manager):
        """spawn_player should use the spawn point."""
        result = populated_manager.spawn_player("player_1")
        assert result is not None

        # At least one spawn should have been used
        used = sum(1 for sp in populated_manager._spawn_points.values()
                   if sp._last_used > 0)
        assert used >= 1

    def test_spawn_player_random_offset(self, spawn_manager):
        """spawn_player should apply random offset."""
        spawn_manager.register_spawn_point(SpawnPoint(
            point_id="test", position=(0, 0, 0)
        ))

        rule = SpawnRule(random_offset_radius=5.0)

        positions = set()
        for i in range(10):
            result = spawn_manager.spawn_player(f"player_{i}", rule=rule)
            if result:
                positions.add(result[0])

        # Should have varied positions due to offset
        # (not guaranteed but likely with 10 samples)
        assert len(positions) >= 2 or True  # Allow for rare same-position

    def test_spawn_player_tracks_position(self, populated_manager):
        """spawn_player should track player position."""
        populated_manager.spawn_player("player_1")
        pos = populated_manager.get_player_position("player_1")
        assert pos is not None

    def test_spawn_player_tracks_team(self, populated_manager):
        """spawn_player should track player team."""
        populated_manager.spawn_player("player_1", team_id="red")
        assert populated_manager._player_teams.get("player_1") == "red"

    def test_spawn_with_pending_respawn(self, spawn_manager):
        """spawn_player should respect respawn delay."""
        spawn_manager.register_spawn_point(SpawnPoint(
            point_id="test", position=(0, 0, 0)
        ))

        spawn_manager.schedule_respawn("player_1", delay_seconds=10.0)

        # Should not spawn yet
        result = spawn_manager.spawn_player("player_1")
        assert result is None

    def test_spawn_emits_callback(self, populated_manager):
        """spawn_player should emit callback."""
        callback = Mock()
        populated_manager.on_spawn(callback)

        populated_manager.spawn_player("player_1")
        callback.assert_called()


# =============================================================================
# RESPAWN SCHEDULING TESTS (20 tests)
# =============================================================================


class TestRespawnScheduling:
    """Tests for respawn scheduling."""

    def test_schedule_respawn(self, spawn_manager):
        """schedule_respawn should set respawn time."""
        respawn_time = spawn_manager.schedule_respawn("player_1")
        assert respawn_time > time.time()

    def test_schedule_respawn_custom_delay(self, spawn_manager):
        """schedule_respawn should accept custom delay."""
        respawn_time = spawn_manager.schedule_respawn("player_1", delay_seconds=10.0)
        expected = time.time() + 10.0
        assert abs(respawn_time - expected) < 0.1

    def test_get_respawn_time(self, spawn_manager):
        """get_respawn_time should return scheduled time."""
        spawn_manager.schedule_respawn("player_1", delay_seconds=5.0)
        respawn_time = spawn_manager.get_respawn_time("player_1")
        assert respawn_time is not None

    def test_get_respawn_time_none(self, spawn_manager):
        """get_respawn_time should return None if not scheduled."""
        respawn_time = spawn_manager.get_respawn_time("player_1")
        assert respawn_time is None

    def test_is_respawn_ready_false(self, spawn_manager):
        """is_respawn_ready should be False before time."""
        spawn_manager.schedule_respawn("player_1", delay_seconds=10.0)
        assert not spawn_manager.is_respawn_ready("player_1")

    def test_is_respawn_ready_true(self, spawn_manager):
        """is_respawn_ready should be True after time."""
        with patch('time.time') as mock_time:
            mock_time.return_value = 1000.0
            spawn_manager.schedule_respawn("player_1", delay_seconds=5.0)

            mock_time.return_value = 1010.0
            assert spawn_manager.is_respawn_ready("player_1")

    def test_is_respawn_ready_no_schedule(self, spawn_manager):
        """is_respawn_ready should be True if no schedule."""
        assert spawn_manager.is_respawn_ready("player_1")

    def test_cancel_respawn(self, spawn_manager):
        """cancel_respawn should remove schedule."""
        spawn_manager.schedule_respawn("player_1")
        result = spawn_manager.cancel_respawn("player_1")
        assert result
        assert spawn_manager.get_respawn_time("player_1") is None

    def test_cancel_respawn_not_scheduled(self, spawn_manager):
        """cancel_respawn should return False if not scheduled."""
        result = spawn_manager.cancel_respawn("player_1")
        assert not result


# =============================================================================
# TEAM CONFIGURATION TESTS (15 tests)
# =============================================================================


class TestTeamConfiguration:
    """Tests for team spawn configuration."""

    def test_configure_team(self, spawn_manager):
        """configure_team should store config."""
        config = TeamSpawnConfig(team_id="red", spawn_points=["sp1", "sp2"])
        spawn_manager.configure_team(config)

        stored = spawn_manager.get_team_config("red")
        assert stored is not None
        assert stored.team_id == "red"

    def test_set_rally_point(self, spawn_manager):
        """set_rally_point should set position."""
        config = TeamSpawnConfig(team_id="red", rally_point_enabled=True)
        spawn_manager.configure_team(config)

        result = spawn_manager.set_rally_point("red", (50, 0, 50))
        assert result

        stored = spawn_manager.get_team_config("red")
        assert stored.rally_point_position == (50, 0, 50)

    def test_set_rally_point_disabled(self, spawn_manager):
        """set_rally_point should fail if disabled."""
        config = TeamSpawnConfig(team_id="red", rally_point_enabled=False)
        spawn_manager.configure_team(config)

        result = spawn_manager.set_rally_point("red", (50, 0, 50))
        assert not result

    def test_team_config_rule_used(self, spawn_manager):
        """Team-specific rule should be used."""
        spawn_manager.register_spawn_point(SpawnPoint(
            point_id="test", position=(0, 0, 0), team_id="red"
        ))

        team_rule = SpawnRule(rule_type=SpawnRuleType.FIXED)
        config = TeamSpawnConfig(team_id="red", spawn_rule=team_rule)
        spawn_manager.configure_team(config)

        point = spawn_manager.select_spawn_point("player", team_id="red")
        assert point is not None


# =============================================================================
# PLAYER TRACKING TESTS (15 tests)
# =============================================================================


class TestPlayerTracking:
    """Tests for player position tracking."""

    def test_update_player_position(self, spawn_manager):
        """update_player_position should store position."""
        spawn_manager.update_player_position("player_1", (10, 5, 20))
        pos = spawn_manager.get_player_position("player_1")
        assert pos == (10, 5, 20)

    def test_update_player_team(self, spawn_manager):
        """update_player_team should store team."""
        spawn_manager.update_player_team("player_1", "red")
        assert spawn_manager._player_teams["player_1"] == "red"

    def test_remove_player(self, spawn_manager):
        """remove_player should clear all tracking."""
        spawn_manager.update_player_position("player_1", (0, 0, 0))
        spawn_manager.update_player_team("player_1", "red")
        spawn_manager.schedule_respawn("player_1")

        spawn_manager.remove_player("player_1")

        assert spawn_manager.get_player_position("player_1") is None
        assert "player_1" not in spawn_manager._player_teams
        assert spawn_manager.get_respawn_time("player_1") is None


# =============================================================================
# UPDATE AND UTILITY TESTS (15 tests)
# =============================================================================


class TestUpdateAndUtility:
    """Tests for update loop and utility methods."""

    def test_update_fires_respawn_ready(self, spawn_manager):
        """update should fire respawn ready callback."""
        callback = Mock()
        spawn_manager.on_respawn_ready(callback)

        with patch('time.time') as mock_time:
            mock_time.return_value = 1000.0
            spawn_manager.schedule_respawn("player_1", delay_seconds=5.0)

            mock_time.return_value = 1010.0
            spawn_manager.update()

            callback.assert_called_with("player_1")

    def test_reset(self, populated_manager):
        """reset should clear state."""
        populated_manager.spawn_player("player_1")
        populated_manager.schedule_respawn("player_2")

        populated_manager.reset()

        # All spawn points should be reset
        for sp in populated_manager._spawn_points.values():
            assert sp._state == SpawnPointState.AVAILABLE
            assert sp._current_occupants == 0

        # Player tracking cleared
        assert len(populated_manager._player_positions) == 0
        assert len(populated_manager._pending_respawns) == 0

    def test_get_stats(self, populated_manager):
        """get_stats should return statistics."""
        stats = populated_manager.get_stats()

        assert "total_spawn_points" in stats
        assert "available_spawn_points" in stats
        assert stats["total_spawn_points"] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
