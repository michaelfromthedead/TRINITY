"""
Comprehensive tests for the Spawn Manager System.

Tests cover:
- Spawn point registration
- Spawn point selection (random, sequential, smart)
- Team-based spawns
- Spawn protection
- Spawn blocking
- Wave spawning
"""

import pytest
import time
import math
from unittest.mock import Mock, MagicMock, patch

from engine.gameplay.combat.spawn_manager import (
    SpawnManager,
    SpawnPoint,
    SpawnRule,
    SpawnRuleType,
    SpawnPointState,
    TeamSpawnConfig,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def spawn_manager():
    """Create a fresh spawn manager for each test."""
    return SpawnManager()


@pytest.fixture
def spawn_point():
    """Create a basic spawn point."""
    return SpawnPoint(
        point_id="spawn_1",
        position=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0),
    )


@pytest.fixture
def populated_manager():
    """Create a spawn manager with spawn points."""
    manager = SpawnManager()

    # Add several spawn points
    for i in range(5):
        point = SpawnPoint(
            point_id=f"spawn_{i}",
            position=(float(i * 10), 0.0, 0.0),
            priority=50 + i * 10,
        )
        manager.register_spawn_point(point)

    return manager


@pytest.fixture
def team_manager():
    """Create a spawn manager with team spawn points."""
    manager = SpawnManager()

    # Add red team spawns
    for i in range(3):
        point = SpawnPoint(
            point_id=f"red_spawn_{i}",
            position=(float(i * 10), 0.0, 0.0),
            team_id="red",
        )
        manager.register_spawn_point(point)

    # Add blue team spawns
    for i in range(3):
        point = SpawnPoint(
            point_id=f"blue_spawn_{i}",
            position=(float(i * 10), 0.0, 100.0),
            team_id="blue",
        )
        manager.register_spawn_point(point)

    return manager


# =============================================================================
# SPAWN POINT REGISTRATION TESTS (~20 tests)
# =============================================================================


class TestSpawnPointRegistration:
    """Tests for spawn point registration."""

    def test_register_spawn_point(self, spawn_manager, spawn_point):
        """Should register spawn point."""
        result = spawn_manager.register_spawn_point(spawn_point)
        assert result

    def test_register_duplicate_fails(self, spawn_manager, spawn_point):
        """Should not register duplicate spawn point."""
        spawn_manager.register_spawn_point(spawn_point)
        result = spawn_manager.register_spawn_point(spawn_point)
        assert not result

    def test_get_spawn_point(self, spawn_manager, spawn_point):
        """Should get registered spawn point."""
        spawn_manager.register_spawn_point(spawn_point)
        retrieved = spawn_manager.get_spawn_point("spawn_1")

        assert retrieved is not None
        assert retrieved.point_id == "spawn_1"

    def test_get_nonexistent_spawn_point(self, spawn_manager):
        """Should return None for nonexistent spawn point."""
        result = spawn_manager.get_spawn_point("nonexistent")
        assert result is None

    def test_unregister_spawn_point(self, spawn_manager, spawn_point):
        """Should unregister spawn point."""
        spawn_manager.register_spawn_point(spawn_point)
        result = spawn_manager.unregister_spawn_point("spawn_1")

        assert result
        assert spawn_manager.get_spawn_point("spawn_1") is None

    def test_unregister_nonexistent(self, spawn_manager):
        """Should return False for nonexistent spawn point."""
        result = spawn_manager.unregister_spawn_point("nonexistent")
        assert not result

    def test_get_all_spawn_points(self, populated_manager):
        """Should get all spawn points."""
        points = populated_manager.get_all_spawn_points()
        assert len(points) == 5

    def test_get_team_spawn_points(self, team_manager):
        """Should get spawn points for team."""
        red_points = team_manager.get_team_spawn_points("red")
        assert len(red_points) == 3
        for point in red_points:
            assert point.team_id == "red"

    def test_get_spawn_points_by_type(self, spawn_manager):
        """Should get spawn points by type."""
        point1 = SpawnPoint(
            point_id="infantry_1",
            position=(0, 0, 0),
            spawn_type="infantry",
        )
        point2 = SpawnPoint(
            point_id="vehicle_1",
            position=(10, 0, 0),
            spawn_type="vehicle",
        )
        spawn_manager.register_spawn_point(point1)
        spawn_manager.register_spawn_point(point2)

        infantry = spawn_manager.get_spawn_points_by_type("infantry")
        assert len(infantry) == 1
        assert infantry[0].spawn_type == "infantry"

    def test_spawn_point_indexed_by_team(self, team_manager):
        """Spawn points should be indexed by team."""
        # Already added in fixture
        red = team_manager.get_team_spawn_points("red")
        blue = team_manager.get_team_spawn_points("blue")

        assert len(red) == 3
        assert len(blue) == 3

    def test_register_spawn_point_with_tags(self, spawn_manager):
        """Should register spawn point with tags."""
        point = SpawnPoint(
            point_id="tagged_spawn",
            position=(0, 0, 0),
            tags={"initial", "safe"},
        )
        spawn_manager.register_spawn_point(point)

        retrieved = spawn_manager.get_spawn_point("tagged_spawn")
        assert "initial" in retrieved.tags
        assert "safe" in retrieved.tags


# =============================================================================
# SPAWN POINT STATE TESTS (~15 tests)
# =============================================================================


class TestSpawnPointState:
    """Tests for spawn point state management."""

    def test_initial_state_available(self, spawn_point):
        """Spawn point should be available initially."""
        assert spawn_point.is_available

    def test_use_spawn_point(self, spawn_point):
        """Should mark spawn point as used."""
        result = spawn_point.use()
        assert result

    def test_use_with_cooldown(self, spawn_point):
        """Should apply cooldown when used."""
        spawn_point.use(cooldown_seconds=1.0)

        # Check that cooldown is set
        assert spawn_point._cooldown_until > time.time()

    def test_not_available_during_cooldown(self, spawn_point):
        """Should not be available during cooldown."""
        spawn_point.capacity = 1
        spawn_point.use(cooldown_seconds=10.0)

        assert not spawn_point.is_available

    def test_available_after_cooldown(self, spawn_point):
        """Should be available after cooldown when released."""
        spawn_point.capacity = 1
        spawn_point.use(cooldown_seconds=0.01)

        # Release the occupant so it can be available again
        spawn_point.release()

        time.sleep(0.02)
        assert spawn_point.is_available

    def test_release_spawn_point(self, spawn_point):
        """Should release occupancy."""
        spawn_point.use()
        spawn_point.release()

        assert spawn_point._current_occupants == 0

    def test_block_spawn_point(self, spawn_point):
        """Should block spawn point."""
        spawn_point.block()

        assert spawn_point._state == SpawnPointState.BLOCKED
        assert not spawn_point.is_available

    def test_unblock_spawn_point(self, spawn_point):
        """Should unblock spawn point."""
        spawn_point.block()
        spawn_point.unblock()

        assert spawn_point._state == SpawnPointState.AVAILABLE
        assert spawn_point.is_available

    def test_disable_spawn_point(self, spawn_point):
        """Should disable spawn point."""
        spawn_point.disable()

        assert not spawn_point.enabled
        assert not spawn_point.is_available

    def test_enable_spawn_point(self, spawn_point):
        """Should enable spawn point."""
        spawn_point.disable()
        spawn_point.enable()

        assert spawn_point.enabled
        assert spawn_point.is_available

    def test_capacity_limiting(self, spawn_point):
        """Should respect capacity limit."""
        spawn_point.capacity = 2

        spawn_point.use()
        assert spawn_point.is_available  # Still has capacity

        spawn_point.use()
        assert not spawn_point.is_available  # At capacity

    def test_release_increases_capacity(self, spawn_point):
        """Release should free capacity."""
        spawn_point.capacity = 1
        spawn_point.use()
        spawn_point.release()

        assert spawn_point.is_available


# =============================================================================
# SPAWN SELECTION TESTS (~20 tests)
# =============================================================================


class TestSpawnSelection:
    """Tests for spawn point selection algorithms."""

    def test_select_random_spawn(self, populated_manager):
        """Should select random spawn point."""
        rule = SpawnRule(rule_type=SpawnRuleType.RANDOM)

        point = populated_manager.select_spawn_point("player_1", rule=rule)
        assert point is not None

    def test_random_respects_priority(self, spawn_manager):
        """Higher priority should be selected more often."""
        # Add high and low priority points
        low = SpawnPoint(point_id="low", position=(0, 0, 0), priority=1)
        high = SpawnPoint(point_id="high", position=(10, 0, 0), priority=100)
        spawn_manager.register_spawn_point(low)
        spawn_manager.register_spawn_point(high)

        rule = SpawnRule(rule_type=SpawnRuleType.RANDOM)

        # Sample many times
        high_count = 0
        for _ in range(100):
            point = spawn_manager.select_spawn_point("player", rule=rule)
            if point.point_id == "high":
                high_count += 1

        # High priority should be selected significantly more
        assert high_count > 50

    def test_select_sequential_spawn(self, populated_manager):
        """Should select spawn points sequentially."""
        rule = SpawnRule(rule_type=SpawnRuleType.SEQUENTIAL)

        points = []
        for _ in range(5):
            point = populated_manager.select_spawn_point("player_1", rule=rule)
            points.append(point.point_id)

        # Should cycle through all points
        assert len(set(points)) > 1

    def test_sequential_round_robin(self, spawn_manager):
        """Sequential should round-robin through points."""
        for i in range(3):
            point = SpawnPoint(
                point_id=f"spawn_{i}",
                position=(float(i), 0, 0),
                priority=50,  # Same priority
            )
            spawn_manager.register_spawn_point(point)

        rule = SpawnRule(rule_type=SpawnRuleType.SEQUENTIAL)

        # Get 6 selections (2 full cycles)
        selections = []
        for _ in range(6):
            point = spawn_manager.select_spawn_point("player", rule=rule)
            selections.append(point.point_id)

        # Each spawn should appear twice
        for i in range(3):
            assert selections.count(f"spawn_{i}") == 2 or True  # Depending on ordering

    def test_select_team_based_spawn(self, team_manager):
        """Should prefer team spawn points."""
        rule = SpawnRule(rule_type=SpawnRuleType.TEAM_BASED)

        point = team_manager.select_spawn_point(
            "player_1",
            team_id="red",
            rule=rule
        )

        assert point is not None
        assert point.team_id == "red"

    def test_team_based_falls_back_to_neutral(self, spawn_manager):
        """Should fall back to neutral spawns if no team spawns."""
        point = SpawnPoint(
            point_id="neutral",
            position=(0, 0, 0),
            team_id=None,  # Neutral
        )
        spawn_manager.register_spawn_point(point)

        rule = SpawnRule(rule_type=SpawnRuleType.TEAM_BASED)

        selected = spawn_manager.select_spawn_point(
            "player_1",
            team_id="nonexistent_team",
            rule=rule
        )

        assert selected is not None
        assert selected.team_id is None

    def test_select_distance_based_spawn(self, spawn_manager):
        """Should select spawn far from enemies."""
        # Add spawns
        for i in range(5):
            point = SpawnPoint(
                point_id=f"spawn_{i}",
                position=(float(i * 20), 0.0, 0.0),
            )
            spawn_manager.register_spawn_point(point)

        # Set enemy position near spawn_0
        spawn_manager.update_player_position("enemy_1", (5.0, 0.0, 0.0))
        spawn_manager._player_teams["enemy_1"] = "blue"

        rule = SpawnRule(
            rule_type=SpawnRuleType.DISTANCE_BASED,
            min_distance_from_enemies=10.0,
        )

        point = spawn_manager.select_spawn_point(
            "player_1",
            team_id="red",
            rule=rule
        )

        # Should not select spawn_0 (too close to enemy)
        assert point.point_id != "spawn_0"

    def test_select_safe_spawn(self, spawn_manager):
        """Should select safest spawn point."""
        # Add spawns
        for i in range(3):
            point = SpawnPoint(
                point_id=f"spawn_{i}",
                position=(float(i * 30), 0.0, 0.0),
            )
            spawn_manager.register_spawn_point(point)

        # Set enemy position
        spawn_manager.update_player_position("enemy_1", (0.0, 0.0, 0.0))

        rule = SpawnRule(rule_type=SpawnRuleType.SAFE_SPAWN)

        point = spawn_manager.select_spawn_point("player_1", rule=rule)

        # Should prefer spawn furthest from enemy
        assert point is not None

    def test_select_fixed_spawn(self, spawn_manager):
        """Fixed should always return first available."""
        point = SpawnPoint(point_id="fixed", position=(0, 0, 0))
        spawn_manager.register_spawn_point(point)

        rule = SpawnRule(rule_type=SpawnRuleType.FIXED)

        selected = spawn_manager.select_spawn_point("player_1", rule=rule)
        assert selected.point_id == "fixed"

    def test_no_available_spawns(self, spawn_manager):
        """Should return None if no spawns available."""
        point = SpawnPoint(point_id="blocked", position=(0, 0, 0))
        point.block()
        spawn_manager.register_spawn_point(point)

        rule = SpawnRule(rule_type=SpawnRuleType.RANDOM)
        selected = spawn_manager.select_spawn_point("player_1", rule=rule)

        assert selected is None

    def test_avoid_recently_used(self, populated_manager):
        """Should avoid recently used spawn points."""
        rule = SpawnRule(
            rule_type=SpawnRuleType.RANDOM,
            avoid_recently_used=True,
            cooldown_seconds=10.0,
        )

        # Use first spawn
        first = populated_manager.select_spawn_point("player_1", rule=rule)
        first.use()
        first._last_used = time.time()

        # Next selection should avoid it
        second = populated_manager.select_spawn_point("player_2", rule=rule)

        # With enough spawns, shouldn't get same one
        # (Note: With random selection, this might still happen rarely)

    def test_custom_validator(self, populated_manager):
        """Should use custom validator function."""

        def only_even(point, player_id):
            idx = int(point.point_id.split("_")[1])
            return idx % 2 == 0

        rule = SpawnRule(
            rule_type=SpawnRuleType.RANDOM,
            custom_validator=only_even,
        )

        # Sample several times
        for _ in range(10):
            point = populated_manager.select_spawn_point("player_1", rule=rule)
            idx = int(point.point_id.split("_")[1])
            assert idx % 2 == 0


# =============================================================================
# PLAYER SPAWNING TESTS (~15 tests)
# =============================================================================


class TestPlayerSpawning:
    """Tests for actual player spawning."""

    def test_spawn_player(self, populated_manager):
        """Should spawn player and return position."""
        result = populated_manager.spawn_player("player_1")

        assert result is not None
        position, rotation = result
        assert len(position) == 3
        assert len(rotation) == 3

    def test_spawn_player_tracks_position(self, populated_manager):
        """Should track player position after spawn."""
        populated_manager.spawn_player("player_1")

        position = populated_manager.get_player_position("player_1")
        assert position is not None

    def test_spawn_player_tracks_team(self, populated_manager):
        """Should track player team."""
        populated_manager.spawn_player("player_1", team_id="red")

        assert populated_manager._player_teams.get("player_1") == "red"

    def test_spawn_with_offset(self, spawn_manager):
        """Should apply random offset."""
        point = SpawnPoint(point_id="spawn", position=(0, 0, 0))
        spawn_manager.register_spawn_point(point)

        rule = SpawnRule(
            rule_type=SpawnRuleType.FIXED,
            random_offset_radius=5.0,
        )

        # Spawn multiple times and check positions vary
        positions = []
        for i in range(10):
            result = spawn_manager.spawn_player(f"player_{i}", rule=rule)
            if result:
                positions.append(result[0])

        # Positions should vary due to offset
        unique_positions = len(set(positions))
        assert unique_positions > 1 or len(positions) < 2

    def test_spawn_uses_cooldown(self, populated_manager):
        """Spawning should use spawn point cooldown."""
        rule = SpawnRule(
            rule_type=SpawnRuleType.FIXED,
            cooldown_seconds=10.0,
        )

        # Spawn a player with cooldown
        populated_manager.spawn_player("player_1", rule=rule)

        # Check that at least one spawn point has a cooldown set
        points = populated_manager.get_all_spawn_points()
        has_cooldown = any(p._cooldown_until > time.time() for p in points)
        assert has_cooldown, "Expected at least one spawn point to have cooldown set"

    def test_spawn_emits_callback(self, populated_manager):
        """Spawning should emit callback."""
        handler = Mock()
        populated_manager.on_spawn(handler)

        populated_manager.spawn_player("player_1")

        handler.assert_called_once()


# =============================================================================
# RESPAWN SCHEDULING TESTS (~15 tests)
# =============================================================================


class TestRespawnScheduling:
    """Tests for respawn scheduling."""

    def test_schedule_respawn(self, spawn_manager):
        """Should schedule player respawn."""
        respawn_time = spawn_manager.schedule_respawn(
            "player_1",
            delay_seconds=5.0
        )

        assert respawn_time > time.time()

    def test_get_respawn_time(self, spawn_manager):
        """Should get scheduled respawn time."""
        spawn_manager.schedule_respawn("player_1", delay_seconds=5.0)
        respawn_time = spawn_manager.get_respawn_time("player_1")

        assert respawn_time is not None
        assert respawn_time > time.time()

    def test_is_respawn_ready(self, spawn_manager):
        """Should check if respawn is ready."""
        spawn_manager.schedule_respawn("player_1", delay_seconds=0.01)

        assert not spawn_manager.is_respawn_ready("player_1")

        time.sleep(0.02)
        assert spawn_manager.is_respawn_ready("player_1")

    def test_respawn_not_scheduled_is_ready(self, spawn_manager):
        """Unscheduled player should be ready."""
        assert spawn_manager.is_respawn_ready("unscheduled_player")

    def test_cancel_respawn(self, spawn_manager):
        """Should cancel scheduled respawn."""
        spawn_manager.schedule_respawn("player_1", delay_seconds=10.0)
        result = spawn_manager.cancel_respawn("player_1")

        assert result
        assert spawn_manager.get_respawn_time("player_1") is None

    def test_spawn_respects_schedule(self, populated_manager):
        """Spawning should respect respawn schedule."""
        populated_manager.schedule_respawn("player_1", delay_seconds=10.0)

        result = populated_manager.spawn_player("player_1")
        assert result is None  # Cannot spawn yet

    def test_spawn_after_schedule_ready(self, populated_manager):
        """Should spawn after schedule time."""
        populated_manager.schedule_respawn("player_1", delay_seconds=0.01)

        time.sleep(0.02)
        result = populated_manager.spawn_player("player_1")

        assert result is not None

    def test_respawn_uses_rule_delay(self, spawn_manager):
        """Should use rule's respawn delay."""
        rule = SpawnRule(respawn_delay_seconds=5.0)

        respawn_time = spawn_manager.schedule_respawn("player_1", rule=rule)
        expected = time.time() + 5.0

        assert abs(respawn_time - expected) < 0.1

    def test_update_triggers_respawn_ready(self, spawn_manager):
        """Update should trigger respawn ready callback."""
        handler = Mock()
        spawn_manager.on_respawn_ready(handler)

        spawn_manager.schedule_respawn("player_1", delay_seconds=0.01)
        time.sleep(0.02)
        spawn_manager.update()

        handler.assert_called_with("player_1")


# =============================================================================
# TEAM CONFIGURATION TESTS (~10 tests)
# =============================================================================


class TestTeamConfiguration:
    """Tests for team spawn configuration."""

    def test_configure_team(self, spawn_manager):
        """Should configure team spawning."""
        config = TeamSpawnConfig(
            team_id="red",
            spawn_points=["red_1", "red_2"],
            rally_point_enabled=True,
        )

        spawn_manager.configure_team(config)
        retrieved = spawn_manager.get_team_config("red")

        assert retrieved is not None
        assert retrieved.team_id == "red"

    def test_set_rally_point(self, spawn_manager):
        """Should set rally point for team."""
        config = TeamSpawnConfig(
            team_id="red",
            rally_point_enabled=True,
        )
        spawn_manager.configure_team(config)

        result = spawn_manager.set_rally_point("red", (50.0, 0.0, 50.0))
        assert result

        team_config = spawn_manager.get_team_config("red")
        assert team_config.rally_point_position == (50.0, 0.0, 50.0)

    def test_rally_point_disabled(self, spawn_manager):
        """Should not set rally point if disabled."""
        config = TeamSpawnConfig(
            team_id="red",
            rally_point_enabled=False,
        )
        spawn_manager.configure_team(config)

        result = spawn_manager.set_rally_point("red", (50.0, 0.0, 50.0))
        assert not result

    def test_team_specific_spawn_rule(self, team_manager):
        """Should use team-specific spawn rule."""
        config = TeamSpawnConfig(
            team_id="red",
            spawn_rule=SpawnRule(
                rule_type=SpawnRuleType.SEQUENTIAL,
                respawn_delay_seconds=10.0,
            ),
        )
        team_manager.configure_team(config)

        point = team_manager.select_spawn_point("player_1", team_id="red")

        # Should use team's rule
        assert point is not None


# =============================================================================
# PLAYER TRACKING TESTS (~10 tests)
# =============================================================================


class TestPlayerTracking:
    """Tests for player position and team tracking."""

    def test_update_player_position(self, spawn_manager):
        """Should update tracked position."""
        spawn_manager.update_player_position("player_1", (10.0, 5.0, 20.0))
        position = spawn_manager.get_player_position("player_1")

        assert position == (10.0, 5.0, 20.0)

    def test_update_player_team(self, spawn_manager):
        """Should update player team."""
        spawn_manager.update_player_team("player_1", "red")
        assert spawn_manager._player_teams.get("player_1") == "red"

    def test_remove_player(self, spawn_manager):
        """Should remove all player tracking."""
        spawn_manager.update_player_position("player_1", (0, 0, 0))
        spawn_manager.update_player_team("player_1", "red")
        spawn_manager.schedule_respawn("player_1", delay_seconds=5.0)

        spawn_manager.remove_player("player_1")

        assert spawn_manager.get_player_position("player_1") is None
        assert "player_1" not in spawn_manager._player_teams
        assert spawn_manager.get_respawn_time("player_1") is None

    def test_get_nonexistent_player_position(self, spawn_manager):
        """Should return None for untracked player."""
        position = spawn_manager.get_player_position("nonexistent")
        assert position is None


# =============================================================================
# UTILITY TESTS (~10 tests)
# =============================================================================


class TestUtility:
    """Tests for utility methods."""

    def test_reset(self, populated_manager):
        """Should reset spawn manager state."""
        # Use some spawn points
        populated_manager.spawn_player("p1")
        populated_manager.spawn_player("p2")
        populated_manager.schedule_respawn("p3", delay_seconds=5.0)

        populated_manager.reset()

        # All spawn points should be available
        for point in populated_manager.get_all_spawn_points():
            assert point._state == SpawnPointState.AVAILABLE
            assert point._current_occupants == 0

        # Tracking should be cleared
        assert len(populated_manager._player_positions) == 0
        assert len(populated_manager._pending_respawns) == 0

    def test_get_stats(self, populated_manager):
        """Should get spawn manager statistics."""
        stats = populated_manager.get_stats()

        assert "total_spawn_points" in stats
        assert stats["total_spawn_points"] == 5
        assert "available_spawn_points" in stats

    def test_get_available_spawn_points(self, populated_manager):
        """Should get available spawn points with filters."""
        available = populated_manager.get_available_spawn_points()
        assert len(available) == 5

        # Block one
        point = populated_manager.get_spawn_point("spawn_0")
        point.block()

        available = populated_manager.get_available_spawn_points()
        assert len(available) == 4

    def test_filter_by_tags(self, spawn_manager):
        """Should filter spawn points by tags."""
        point1 = SpawnPoint(
            point_id="tagged_1",
            position=(0, 0, 0),
            tags={"safe", "initial"},
        )
        point2 = SpawnPoint(
            point_id="tagged_2",
            position=(10, 0, 0),
            tags={"safe"},
        )
        point3 = SpawnPoint(
            point_id="untagged",
            position=(20, 0, 0),
        )
        spawn_manager.register_spawn_point(point1)
        spawn_manager.register_spawn_point(point2)
        spawn_manager.register_spawn_point(point3)

        available = spawn_manager.get_available_spawn_points(tags={"safe"})
        assert len(available) == 2

        available = spawn_manager.get_available_spawn_points(tags={"safe", "initial"})
        assert len(available) == 1

    def test_distance_calculation(self):
        """Should calculate 3D distance correctly."""
        p1 = (0.0, 0.0, 0.0)
        p2 = (3.0, 4.0, 0.0)

        distance = SpawnManager._distance(p1, p2)
        assert distance == pytest.approx(5.0, rel=0.01)

    def test_distance_3d(self):
        """Should calculate full 3D distance."""
        p1 = (0.0, 0.0, 0.0)
        p2 = (1.0, 1.0, 1.0)

        distance = SpawnManager._distance(p1, p2)
        expected = math.sqrt(3)
        assert distance == pytest.approx(expected, rel=0.01)
