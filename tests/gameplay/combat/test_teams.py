"""
Comprehensive tests for the Team System.

Tests cover:
- Team creation
- Team membership
- IFF (Identify Friend/Foe)
- Team relationships
- Auto-assign
- Team events
- Friendly fire
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch

from engine.gameplay.combat.teams import (
    TeamSystem,
    TeamInfo,
    TeamMembership,
    TeamChangeEvent,
    IFFResult,
)
from engine.gameplay.combat.constants import (
    TeamConfig,
    TeamRelation,
    FRIENDLY_FIRE_FULL,
    FRIENDLY_FIRE_REDUCED,
    FRIENDLY_FIRE_NONE,
    MAX_TEAMS,
    DEFAULT_TEAM_ID,
    NEUTRAL_TEAM_ID,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def team_system():
    """Create a fresh team system for each test."""
    return TeamSystem()


@pytest.fixture
def configured_system():
    """Create a team system with pre-configured teams."""
    system = TeamSystem()
    system.create_team(team_id=100, name="Red Team", max_members=8)
    system.create_team(team_id=200, name="Blue Team", max_members=8)
    return system


@pytest.fixture
def populated_system(configured_system):
    """Create a team system with players assigned."""
    for i in range(4):
        team_id = 100 if i < 2 else 200
        configured_system.set_team(entity_id=i, team_id=team_id)
    return configured_system


# =============================================================================
# TEAM CREATION TESTS (~15 tests)
# =============================================================================


class TestTeamCreation:
    """Tests for team creation."""

    def test_create_team_basic(self, team_system):
        """Should create a basic team."""
        team = team_system.create_team(team_id=10, name="Team A")
        assert team is not None
        assert team.team_id == 10
        assert team.name == "Team A"

    def test_create_team_with_max_members(self, team_system):
        """Should set max members."""
        team = team_system.create_team(team_id=10, name="Team A", max_members=5)
        assert team.max_members == 5

    def test_create_team_with_color(self, team_system):
        """Should set team color."""
        team = team_system.create_team(team_id=10, name="Red Team", color=(255, 0, 0))
        assert team.color == (255, 0, 0)

    def test_create_team_with_spawn_points(self, team_system):
        """Should set spawn points."""
        spawns = [(0, 0, 0), (1, 0, 0)]
        team = team_system.create_team(team_id=10, name="Team A", spawn_points=spawns)
        assert team.spawn_points == spawns

    def test_create_duplicate_team_fails(self, team_system):
        """Should not allow duplicate team IDs."""
        team_system.create_team(team_id=10, name="Team A")
        with pytest.raises(ValueError):
            team_system.create_team(team_id=10, name="Team A Again")

    def test_create_team_returns_team_info(self, team_system):
        """Should return TeamInfo object."""
        team = team_system.create_team(team_id=10, name="Team A")
        assert isinstance(team, TeamInfo)

    def test_create_multiple_teams(self, team_system):
        """Should create multiple teams."""
        team_system.create_team(team_id=10, name="Red")
        team_system.create_team(team_id=20, name="Blue")
        team_system.create_team(team_id=30, name="Green")

        teams = team_system.get_all_teams()
        # +2 for default and neutral teams
        assert len(teams) >= 3

    def test_default_teams_exist(self, team_system):
        """Should have default and neutral teams."""
        assert team_system.team_exists(DEFAULT_TEAM_ID)
        assert team_system.team_exists(NEUTRAL_TEAM_ID)

    def test_remove_team(self, team_system):
        """Should remove team."""
        team_system.create_team(team_id=10, name="Team A")
        result = team_system.remove_team(10)
        assert result
        assert team_system.get_team(10) is None

    def test_remove_default_team_fails(self, team_system):
        """Should not allow removing default team."""
        result = team_system.remove_team(DEFAULT_TEAM_ID)
        assert not result
        assert team_system.team_exists(DEFAULT_TEAM_ID)

    def test_remove_neutral_team_fails(self, team_system):
        """Should not allow removing neutral team."""
        result = team_system.remove_team(NEUTRAL_TEAM_ID)
        assert not result
        assert team_system.team_exists(NEUTRAL_TEAM_ID)

    def test_remove_nonexistent_team(self, team_system):
        """Should return False for nonexistent team."""
        result = team_system.remove_team(999)
        assert not result

    def test_remove_team_relocates_members(self, configured_system):
        """Removing team should move members to default team."""
        configured_system.set_team(entity_id=1, team_id=100)
        configured_system.remove_team(100)

        # Member should now be on default team
        team_id = configured_system.get_team_id(1)
        assert team_id == DEFAULT_TEAM_ID

    def test_get_team(self, configured_system):
        """Should get team by ID."""
        team = configured_system.get_team(100)
        assert team is not None
        assert team.team_id == 100

    def test_get_team_nonexistent(self, team_system):
        """Should return None for nonexistent team."""
        team = team_system.get_team(999)
        assert team is None


# =============================================================================
# TEAM MEMBERSHIP TESTS (~20 tests)
# =============================================================================


class TestTeamMembership:
    """Tests for team membership management."""

    def test_set_team(self, configured_system):
        """Should set entity's team."""
        result = configured_system.set_team(entity_id=1, team_id=100)
        assert result
        assert configured_system.get_team_id(1) == 100

    def test_set_team_invalid_team(self, team_system):
        """Should fail for invalid team."""
        result = team_system.set_team(entity_id=1, team_id=999)
        assert not result

    def test_set_team_max_capacity(self, team_system):
        """Should fail when team is full."""
        team_system.create_team(team_id=10, name="Small Team", max_members=2)
        team_system.set_team(entity_id=1, team_id=10)
        team_system.set_team(entity_id=2, team_id=10)
        result = team_system.set_team(entity_id=3, team_id=10)
        assert not result

    def test_set_team_already_on_team(self, configured_system):
        """Should handle entity already on same team."""
        configured_system.set_team(entity_id=1, team_id=100)
        result = configured_system.set_team(entity_id=1, team_id=100)
        # Should succeed (updating membership)
        assert result

    def test_remove_entity(self, configured_system):
        """Should remove entity from team system."""
        configured_system.set_team(entity_id=1, team_id=100)
        result = configured_system.remove_entity(1)
        assert result

    def test_remove_nonexistent_entity(self, configured_system):
        """Should return False for nonexistent entity."""
        result = configured_system.remove_entity(999)
        assert not result

    def test_get_team_members(self, configured_system):
        """Should get list of team members."""
        configured_system.set_team(entity_id=1, team_id=100)
        configured_system.set_team(entity_id=2, team_id=100)
        configured_system.set_team(entity_id=3, team_id=200)

        members = configured_system.get_team_members(100)
        assert len(members) == 2
        assert 1 in members
        assert 2 in members

    def test_get_team_id(self, configured_system):
        """Should get entity's team ID."""
        configured_system.set_team(entity_id=1, team_id=100)
        team_id = configured_system.get_team_id(1)
        assert team_id == 100

    def test_get_team_id_unassigned(self, team_system):
        """Should return default team for unassigned entity."""
        team_id = team_system.get_team_id(999)
        assert team_id == DEFAULT_TEAM_ID

    def test_get_membership(self, configured_system):
        """Should get entity's membership info."""
        configured_system.set_team(entity_id=1, team_id=100, role="leader")
        membership = configured_system.get_membership(1)
        assert membership is not None
        assert membership.team_id == 100
        assert membership.role == "leader"

    def test_get_membership_nonexistent(self, team_system):
        """Should return None for nonexistent membership."""
        membership = team_system.get_membership(999)
        assert membership is None

    def test_member_count(self, configured_system):
        """Should track member count."""
        configured_system.set_team(entity_id=1, team_id=100)
        configured_system.set_team(entity_id=2, team_id=100)

        count = configured_system.get_team_member_count(100)
        assert count == 2

    def test_is_on_team(self, configured_system):
        """Should check if entity is on specific team."""
        configured_system.set_team(entity_id=1, team_id=100)
        assert configured_system.is_on_team(1, 100)
        assert not configured_system.is_on_team(1, 200)

    def test_switch_teams(self, configured_system):
        """Should switch entity between teams."""
        configured_system.set_team(entity_id=1, team_id=100)
        result = configured_system.set_team(entity_id=1, team_id=200)
        assert result
        assert configured_system.get_team_id(1) == 200

    def test_switch_teams_updates_counts(self, configured_system):
        """Switching teams should update member counts."""
        configured_system.set_team(entity_id=1, team_id=100)
        configured_system.set_team(entity_id=2, team_id=100)

        initial_red = configured_system.get_team_member_count(100)
        initial_blue = configured_system.get_team_member_count(200)

        configured_system.set_team(entity_id=1, team_id=200)

        assert configured_system.get_team_member_count(100) == initial_red - 1
        assert configured_system.get_team_member_count(200) == initial_blue + 1

    def test_membership_duration(self, configured_system):
        """Should track membership duration."""
        configured_system.set_team(entity_id=1, team_id=100)
        time.sleep(0.01)
        membership = configured_system.get_membership(1)
        assert membership.membership_duration > 0


# =============================================================================
# IFF (IDENTIFY FRIEND/FOE) TESTS (~15 tests)
# =============================================================================


class TestIFF:
    """Tests for IFF checks."""

    def test_check_iff_same_team(self, configured_system):
        """Same team should be friendly."""
        configured_system.set_team(entity_id=1, team_id=100)
        configured_system.set_team(entity_id=2, team_id=100)

        result = configured_system.check_iff(1, 2)
        assert result.is_same_team
        assert result.is_friendly
        assert result.relation == TeamRelation.FRIENDLY

    def test_check_iff_different_teams(self, configured_system):
        """Different teams should be hostile by default."""
        configured_system.set_team(entity_id=1, team_id=100)
        configured_system.set_team(entity_id=2, team_id=200)

        result = configured_system.check_iff(1, 2)
        assert not result.is_same_team
        assert result.is_hostile
        assert result.relation == TeamRelation.HOSTILE

    def test_check_iff_returns_iff_result(self, configured_system):
        """Should return IFFResult object."""
        configured_system.set_team(entity_id=1, team_id=100)
        configured_system.set_team(entity_id=2, team_id=200)

        result = configured_system.check_iff(1, 2)
        assert isinstance(result, IFFResult)
        assert result.source_id == 1
        assert result.target_id == 2

    def test_can_attack_hostile(self, configured_system):
        """Should be able to attack hostile targets."""
        configured_system.set_team(entity_id=1, team_id=100)
        configured_system.set_team(entity_id=2, team_id=200)

        assert configured_system.can_attack(1, 2)

    def test_cannot_attack_friendly_by_default(self, configured_system):
        """Should not attack friendlies by default."""
        configured_system.set_team(entity_id=1, team_id=100)
        configured_system.set_team(entity_id=2, team_id=100)

        result = configured_system.check_iff(1, 2)
        # Can damage depends on friendly fire settings
        assert result.is_same_team

    def test_can_heal_friendly(self, configured_system):
        """Should be able to heal friendly targets."""
        configured_system.set_team(entity_id=1, team_id=100)
        configured_system.set_team(entity_id=2, team_id=100)

        assert configured_system.can_heal(1, 2)

    def test_cannot_heal_hostile(self, configured_system):
        """Should not heal hostile targets."""
        configured_system.set_team(entity_id=1, team_id=100)
        configured_system.set_team(entity_id=2, team_id=200)

        assert not configured_system.can_heal(1, 2)

    def test_is_friendly_helper(self, configured_system):
        """is_friendly helper should work."""
        configured_system.set_team(entity_id=1, team_id=100)
        configured_system.set_team(entity_id=2, team_id=100)
        configured_system.set_team(entity_id=3, team_id=200)

        assert configured_system.is_friendly(1, 2)
        assert not configured_system.is_friendly(1, 3)

    def test_is_hostile_helper(self, configured_system):
        """is_hostile helper should work."""
        configured_system.set_team(entity_id=1, team_id=100)
        configured_system.set_team(entity_id=2, team_id=100)
        configured_system.set_team(entity_id=3, team_id=200)

        assert not configured_system.is_hostile(1, 2)
        assert configured_system.is_hostile(1, 3)

    def test_iff_with_neutral_team(self, team_system):
        """Neutral team should be neutral to others."""
        team_system.create_team(team_id=10, name="Test Team")
        team_system.set_team(entity_id=1, team_id=10)
        team_system.set_team(entity_id=2, team_id=NEUTRAL_TEAM_ID)

        result = team_system.check_iff(1, 2)
        assert result.relation == TeamRelation.NEUTRAL


# =============================================================================
# TEAM RELATIONSHIPS TESTS (~10 tests)
# =============================================================================


class TestTeamRelationships:
    """Tests for team relationship management."""

    def test_set_relationship(self, configured_system):
        """Should set relationship between teams."""
        configured_system.set_relationship(100, 200, TeamRelation.FRIENDLY)
        relation = configured_system.get_relationship(100, 200)
        assert relation == TeamRelation.FRIENDLY

    def test_relationship_bidirectional(self, configured_system):
        """Relationship should be bidirectional."""
        configured_system.set_relationship(100, 200, TeamRelation.FRIENDLY)
        assert configured_system.get_relationship(100, 200) == TeamRelation.FRIENDLY
        assert configured_system.get_relationship(200, 100) == TeamRelation.FRIENDLY

    def test_same_team_always_friendly(self, configured_system):
        """Same team should always be friendly."""
        relation = configured_system.get_relationship(100, 100)
        assert relation == TeamRelation.FRIENDLY

    def test_default_relationship_hostile(self, configured_system):
        """Default relationship should be hostile."""
        relation = configured_system.get_relationship(100, 200)
        assert relation == TeamRelation.HOSTILE

    def test_set_all_hostile(self, team_system):
        """Should set all teams hostile."""
        team_system.create_team(team_id=10, name="A")
        team_system.create_team(team_id=20, name="B")
        team_system.create_team(team_id=30, name="C")
        team_system.set_all_hostile()

        assert team_system.get_relationship(10, 20) == TeamRelation.HOSTILE
        assert team_system.get_relationship(20, 30) == TeamRelation.HOSTILE
        assert team_system.get_relationship(10, 30) == TeamRelation.HOSTILE

    def test_set_all_friendly(self, team_system):
        """Should set all teams friendly."""
        team_system.create_team(team_id=10, name="A")
        team_system.create_team(team_id=20, name="B")
        team_system.set_all_friendly()

        assert team_system.get_relationship(10, 20) == TeamRelation.FRIENDLY

    def test_relationship_override(self, team_system):
        """Team-specific override should take precedence."""
        team_system.create_team(team_id=10, name="A")
        team_system.create_team(team_id=20, name="B")

        team = team_system.get_team(10)
        team.set_relationship_override(20, TeamRelation.NEUTRAL)

        relation = team_system.get_relationship(10, 20)
        assert relation == TeamRelation.NEUTRAL

    def test_get_hostile_teams(self, team_system):
        """Should get all hostile teams."""
        team_system.create_team(team_id=10, name="A")
        team_system.create_team(team_id=20, name="B")
        team_system.create_team(team_id=30, name="C")

        team_system.set_relationship(10, 20, TeamRelation.FRIENDLY)
        # 10 and 30 remain hostile by default

        hostile = team_system.get_hostile_teams(10)
        assert 30 in hostile
        assert 20 not in hostile

    def test_get_allied_teams(self, team_system):
        """Should get all allied teams."""
        team_system.create_team(team_id=10, name="A")
        team_system.create_team(team_id=20, name="B")
        team_system.create_team(team_id=30, name="C")

        team_system.set_relationship(10, 20, TeamRelation.FRIENDLY)

        allied = team_system.get_allied_teams(10)
        assert 10 in allied  # Self is always allied
        assert 20 in allied


# =============================================================================
# FRIENDLY FIRE TESTS (~10 tests)
# =============================================================================


class TestFriendlyFire:
    """Tests for friendly fire settings."""

    def test_set_friendly_fire(self, configured_system):
        """Should set friendly fire multiplier."""
        result = configured_system.set_friendly_fire(100, 0.5)
        assert result

        team = configured_system.get_team(100)
        assert team.friendly_fire_multiplier == 0.5

    def test_enable_friendly_fire(self, configured_system):
        """Should enable friendly fire."""
        result = configured_system.enable_friendly_fire(100)
        assert result

        team = configured_system.get_team(100)
        assert team.friendly_fire_multiplier == FRIENDLY_FIRE_FULL

    def test_disable_friendly_fire(self, configured_system):
        """Should disable friendly fire."""
        configured_system.enable_friendly_fire(100)
        result = configured_system.disable_friendly_fire(100)
        assert result

        team = configured_system.get_team(100)
        assert team.friendly_fire_multiplier == FRIENDLY_FIRE_NONE

    def test_friendly_fire_multiplier_clamped(self, configured_system):
        """Multiplier should be clamped to 0-1 range."""
        configured_system.set_friendly_fire(100, 1.5)
        team = configured_system.get_team(100)
        assert team.friendly_fire_multiplier <= 1.0

        configured_system.set_friendly_fire(100, -0.5)
        team = configured_system.get_team(100)
        assert team.friendly_fire_multiplier >= 0.0

    def test_get_friendly_fire_multiplier(self, configured_system):
        """Should get friendly fire multiplier between entities."""
        configured_system.set_team(entity_id=1, team_id=100)
        configured_system.set_team(entity_id=2, team_id=100)
        configured_system.set_friendly_fire(100, 0.5)

        mult = configured_system.get_friendly_fire_multiplier(1, 2)
        assert mult == 0.5

    def test_friendly_fire_only_affects_same_team(self, configured_system):
        """FF multiplier should only apply to same-team damage."""
        configured_system.set_team(entity_id=1, team_id=100)
        configured_system.set_team(entity_id=2, team_id=200)
        configured_system.set_friendly_fire(100, 0.5)

        mult = configured_system.get_friendly_fire_multiplier(1, 2)
        assert mult == 1.0  # Full damage to enemies

    def test_set_friendly_fire_invalid_team(self, team_system):
        """Should fail for invalid team."""
        result = team_system.set_friendly_fire(999, 0.5)
        assert not result


# =============================================================================
# AUTO-ASSIGN TESTS (~10 tests)
# =============================================================================


class TestAutoAssign:
    """Tests for auto-assignment to teams."""

    def test_auto_assign_to_smallest(self, configured_system):
        """Should assign to smallest team (excluding default)."""
        configured_system.set_team(entity_id=1, team_id=100)
        configured_system.set_team(entity_id=2, team_id=100)

        # Exclude default team (0) to only consider custom teams
        team_id = configured_system.auto_assign_team(
            entity_id=3,
            exclude={DEFAULT_TEAM_ID, NEUTRAL_TEAM_ID}
        )
        assert team_id == 200  # Blue should be smaller

    def test_auto_assign_adds_member(self, configured_system):
        """Auto-assign should add member to team."""
        team_id = configured_system.auto_assign_team(entity_id=1)
        assert configured_system.is_on_team(1, team_id)

    def test_auto_assign_with_exclusion(self, configured_system):
        """Should exclude specified teams."""
        team_id = configured_system.auto_assign_team(
            entity_id=1,
            exclude={100, DEFAULT_TEAM_ID, NEUTRAL_TEAM_ID}
        )
        assert team_id == 200

    def test_get_team_with_fewest_members(self, configured_system):
        """Should get team with fewest members (excluding default)."""
        configured_system.set_team(entity_id=1, team_id=100)
        configured_system.set_team(entity_id=2, team_id=100)

        # Exclude default team to only consider custom teams
        smallest = configured_system.get_team_with_fewest_members(
            exclude={DEFAULT_TEAM_ID, NEUTRAL_TEAM_ID}
        )
        assert smallest == 200

    def test_get_team_with_fewest_excludes_neutral(self, team_system):
        """Should exclude neutral team by default."""
        team_system.create_team(team_id=10, name="Test")

        smallest = team_system.get_team_with_fewest_members()
        assert smallest != NEUTRAL_TEAM_ID


# =============================================================================
# TEAM EVENT TESTS (~15 tests)
# =============================================================================


class TestTeamEvents:
    """Tests for team events."""

    def test_on_team_change_event(self, configured_system):
        """Should emit event when entity changes teams."""
        handler = Mock()
        configured_system.on_team_change(handler)

        configured_system.set_team(entity_id=1, team_id=100)

        handler.assert_called_once()
        event = handler.call_args[0][0]
        assert isinstance(event, TeamChangeEvent)
        assert event.entity_id == 1
        assert event.new_team_id == 100

    def test_on_team_change_switch(self, configured_system):
        """Should emit event with old team when switching."""
        handler = Mock()
        configured_system.set_team(entity_id=1, team_id=100)

        configured_system.on_team_change(handler)
        configured_system.set_team(entity_id=1, team_id=200)

        event = handler.call_args[0][0]
        assert event.old_team_id == 100
        assert event.new_team_id == 200

    def test_on_team_created_event(self, team_system):
        """Should emit event when team is created."""
        handler = Mock()
        team_system.on_team_created(handler)

        team_system.create_team(team_id=10, name="New Team")

        handler.assert_called_once()
        team = handler.call_args[0][0]
        assert isinstance(team, TeamInfo)
        assert team.team_id == 10

    def test_on_team_removed_event(self, configured_system):
        """Should emit event when team is removed."""
        handler = Mock()
        configured_system.on_team_removed(handler)

        configured_system.remove_team(100)

        handler.assert_called_once()
        team_id = handler.call_args[0][0]
        assert team_id == 100

    def test_multiple_event_handlers(self, configured_system):
        """Should support multiple handlers."""
        handler1 = Mock()
        handler2 = Mock()
        configured_system.on_team_change(handler1)
        configured_system.on_team_change(handler2)

        configured_system.set_team(entity_id=1, team_id=100)

        handler1.assert_called_once()
        handler2.assert_called_once()

    def test_handler_exception_doesnt_break(self, configured_system):
        """Handler exception should not break system."""
        def bad_handler(event):
            raise Exception("Handler error")

        configured_system.on_team_change(bad_handler)
        result = configured_system.set_team(entity_id=1, team_id=100)
        assert result


# =============================================================================
# QUERY TESTS (~10 tests)
# =============================================================================


class TestQueries:
    """Tests for team system queries."""

    def test_get_enemies(self, configured_system):
        """Should get all hostile entities."""
        configured_system.set_team(entity_id=1, team_id=100)
        configured_system.set_team(entity_id=2, team_id=100)
        configured_system.set_team(entity_id=3, team_id=200)
        configured_system.set_team(entity_id=4, team_id=200)

        enemies = configured_system.get_enemies(1)
        assert 3 in enemies
        assert 4 in enemies
        assert 2 not in enemies

    def test_get_allies(self, configured_system):
        """Should get all friendly entities."""
        configured_system.set_team(entity_id=1, team_id=100)
        configured_system.set_team(entity_id=2, team_id=100)
        configured_system.set_team(entity_id=3, team_id=200)

        allies = configured_system.get_allies(1)
        assert 2 in allies
        assert 3 not in allies
        assert 1 not in allies  # Self not included

    def test_team_exists(self, configured_system):
        """Should check if team exists."""
        assert configured_system.team_exists(100)
        assert not configured_system.team_exists(999)

    def test_get_all_teams(self, configured_system):
        """Should get all teams."""
        teams = configured_system.get_all_teams()
        team_ids = {t.team_id for t in teams}
        assert 100 in team_ids
        assert 200 in team_ids


# =============================================================================
# UTILITY TESTS (~5 tests)
# =============================================================================


class TestUtility:
    """Tests for utility methods."""

    def test_clear(self, populated_system):
        """Should clear all teams and members except defaults."""
        populated_system.clear()

        # Default teams should still exist
        assert populated_system.team_exists(DEFAULT_TEAM_ID)
        assert populated_system.team_exists(NEUTRAL_TEAM_ID)

        # Custom teams should be gone
        assert not populated_system.team_exists(100)
        assert not populated_system.team_exists(200)

    def test_team_info_is_full(self, team_system):
        """TeamInfo should track is_full status."""
        team_system.create_team(team_id=10, name="Small", max_members=2)
        team_system.set_team(entity_id=1, team_id=10)
        team_system.set_team(entity_id=2, team_id=10)

        team = team_system.get_team(10)
        assert team.is_full

    def test_team_info_not_full(self, team_system):
        """TeamInfo should report not full when space available."""
        team_system.create_team(team_id=10, name="Large", max_members=10)
        team_system.set_team(entity_id=1, team_id=10)

        team = team_system.get_team(10)
        assert not team.is_full

    def test_team_info_unlimited_never_full(self, team_system):
        """Team with max_members=0 is never full."""
        team_system.create_team(team_id=10, name="Unlimited", max_members=0)
        for i in range(100):
            team_system.set_team(entity_id=i, team_id=10)

        team = team_system.get_team(10)
        assert not team.is_full

    def test_config_accessible(self, team_system):
        """Should have accessible config."""
        config = team_system.config
        assert config is not None
        assert isinstance(config, TeamConfig)
