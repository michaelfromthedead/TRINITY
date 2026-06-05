"""
WHITEBOX Tests for Teams System

Tests internal implementation details:
- Team relationship matrix
- IFF calculation internals
- Member tracking and counts
- Friendly fire multiplier calculations
- Auto-balance algorithms
"""

import pytest
from unittest.mock import Mock

from engine.gameplay.combat.teams import (
    TeamSystem,
    TeamInfo,
    TeamMembership,
    TeamChangeEvent,
    IFFResult,
)
from engine.gameplay.combat.constants import (
    TeamRelation,
    TeamConfig,
    DEFAULT_TEAM_CONFIG,
    DEFAULT_TEAM_ID,
    NEUTRAL_TEAM_ID,
    MAX_TEAMS,
    FRIENDLY_FIRE_FULL,
    FRIENDLY_FIRE_REDUCED,
    FRIENDLY_FIRE_NONE,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def team_system():
    """Create a fresh team system."""
    return TeamSystem()


@pytest.fixture
def custom_config_system():
    """Create team system with custom config."""
    config = TeamConfig(
        max_teams=10,
        default_friendly_fire=FRIENDLY_FIRE_REDUCED,
        allow_team_changes=False,
        allow_team_damage=True,
    )
    return TeamSystem(config=config)


@pytest.fixture
def populated_system(team_system):
    """Create a team system with teams and members."""
    team_system.create_team(1, name="Red", color=(255, 0, 0))
    team_system.create_team(2, name="Blue", color=(0, 0, 255))
    team_system.set_team(100, 1)
    team_system.set_team(101, 1)
    team_system.set_team(200, 2)
    team_system.set_team(201, 2)
    return team_system


# =============================================================================
# TEAM CREATION TESTS (25 tests)
# =============================================================================


class TestTeamCreation:
    """Tests for team creation."""

    def test_default_teams_created(self, team_system):
        """Default and neutral teams should be created."""
        assert team_system.get_team(DEFAULT_TEAM_ID) is not None
        assert team_system.get_team(NEUTRAL_TEAM_ID) is not None

    def test_create_team(self, team_system):
        """create_team should create team."""
        team = team_system.create_team(1, name="Red")
        assert team is not None
        assert team.team_id == 1
        assert team.name == "Red"

    def test_create_team_with_color(self, team_system):
        """create_team should accept color."""
        team = team_system.create_team(1, color=(255, 0, 0))
        assert team.color == (255, 0, 0)

    def test_create_team_with_max_members(self, team_system):
        """create_team should accept max_members."""
        team = team_system.create_team(1, max_members=8)
        assert team.max_members == 8

    def test_create_team_duplicate_raises(self, team_system):
        """Creating duplicate team should raise ValueError."""
        team_system.create_team(1)
        with pytest.raises(ValueError):
            team_system.create_team(1)

    def test_create_team_max_teams_raises(self, team_system):
        """Exceeding max teams should raise ValueError."""
        team_system._config = TeamConfig(max_teams=3)
        # Default + Neutral = 2, can add 1 more
        team_system.create_team(1)
        with pytest.raises(ValueError):
            team_system.create_team(2)

    def test_create_team_sets_relationships(self, team_system):
        """New team should have relationships with existing."""
        team_system.create_team(1)
        team_system.create_team(2)

        # Should be hostile by default
        relation = team_system.get_relationship(1, 2)
        assert relation == TeamRelation.HOSTILE

    def test_remove_team(self, team_system):
        """remove_team should remove team."""
        team_system.create_team(1)
        result = team_system.remove_team(1)
        assert result
        assert team_system.get_team(1) is None

    def test_remove_default_team_fails(self, team_system):
        """Cannot remove default team."""
        result = team_system.remove_team(DEFAULT_TEAM_ID)
        assert not result

    def test_remove_neutral_team_fails(self, team_system):
        """Cannot remove neutral team."""
        result = team_system.remove_team(NEUTRAL_TEAM_ID)
        assert not result

    def test_remove_team_moves_members(self, populated_system):
        """Removing team should move members to default."""
        populated_system.remove_team(1)

        # Members should be on default team
        assert populated_system.get_team_id(100) == DEFAULT_TEAM_ID
        assert populated_system.get_team_id(101) == DEFAULT_TEAM_ID

    def test_get_all_teams(self, populated_system):
        """get_all_teams should return all teams."""
        teams = populated_system.get_all_teams()
        assert len(teams) >= 4  # Default, Neutral, Red, Blue

    def test_team_exists(self, team_system):
        """team_exists should check existence."""
        team_system.create_team(1)
        assert team_system.team_exists(1)
        assert not team_system.team_exists(999)


# =============================================================================
# MEMBERSHIP TESTS (30 tests)
# =============================================================================


class TestMembership:
    """Tests for team membership."""

    def test_set_team(self, team_system):
        """set_team should assign entity to team."""
        team_system.create_team(1)
        result = team_system.set_team(100, 1)
        assert result
        assert team_system.get_team_id(100) == 1

    def test_set_team_nonexistent_team(self, team_system):
        """set_team to nonexistent team should fail."""
        result = team_system.set_team(100, 999)
        assert not result

    def test_set_team_full_team(self, team_system):
        """set_team to full team should fail."""
        team = team_system.create_team(1, max_members=1)
        team_system.set_team(100, 1)
        result = team_system.set_team(101, 1)
        assert not result

    def test_set_team_updates_member_count(self, team_system):
        """set_team should update member count."""
        team_system.create_team(1)
        team_system.set_team(100, 1)
        team_system.set_team(101, 1)

        assert team_system.get_team_member_count(1) == 2

    def test_set_team_change_team(self, populated_system):
        """Changing teams should update both."""
        populated_system.set_team(100, 2)  # Move from Red to Blue

        assert populated_system.get_team_id(100) == 2
        assert populated_system.get_team_member_count(1) == 1  # Red lost one
        assert populated_system.get_team_member_count(2) == 3  # Blue gained one

    def test_set_team_changes_disabled(self, custom_config_system):
        """Team changes should respect config."""
        custom_config_system.create_team(1)
        custom_config_system.create_team(2)
        custom_config_system.set_team(100, 1)

        # Try to change
        result = custom_config_system.set_team(100, 2)
        assert not result

    def test_get_team_id_default(self, team_system):
        """Unassigned entity should be on default team."""
        team_id = team_system.get_team_id(100)
        assert team_id == DEFAULT_TEAM_ID

    def test_get_membership(self, populated_system):
        """get_membership should return membership info."""
        membership = populated_system.get_membership(100)
        assert membership is not None
        assert membership.entity_id == 100
        assert membership.team_id == 1

    def test_membership_has_role(self, team_system):
        """Membership should have role."""
        team_system.create_team(1)
        team_system.set_team(100, 1, role="leader")

        membership = team_system.get_membership(100)
        assert membership.role == "leader"

    def test_remove_entity(self, populated_system):
        """remove_entity should remove from team."""
        result = populated_system.remove_entity(100)
        assert result
        assert populated_system.get_membership(100) is None

    def test_remove_entity_updates_count(self, populated_system):
        """remove_entity should update team count."""
        count_before = populated_system.get_team_member_count(1)
        populated_system.remove_entity(100)
        count_after = populated_system.get_team_member_count(1)
        assert count_after == count_before - 1

    def test_is_on_team(self, populated_system):
        """is_on_team should check membership."""
        assert populated_system.is_on_team(100, 1)
        assert not populated_system.is_on_team(100, 2)

    def test_get_team_members(self, populated_system):
        """get_team_members should return member IDs."""
        members = populated_system.get_team_members(1)
        assert 100 in members
        assert 101 in members


# =============================================================================
# RELATIONSHIP TESTS (30 tests)
# =============================================================================


class TestRelationships:
    """Tests for team relationships."""

    def test_same_team_friendly(self, populated_system):
        """Same team should be friendly."""
        relation = populated_system.get_relationship(1, 1)
        assert relation == TeamRelation.FRIENDLY

    def test_different_teams_hostile(self, populated_system):
        """Different teams should be hostile by default."""
        relation = populated_system.get_relationship(1, 2)
        assert relation == TeamRelation.HOSTILE

    def test_neutral_team_neutral(self, team_system):
        """Neutral team should be neutral to others."""
        team_system.create_team(1)
        relation = team_system.get_relationship(1, NEUTRAL_TEAM_ID)
        assert relation == TeamRelation.NEUTRAL

    def test_set_relationship(self, populated_system):
        """set_relationship should change relation."""
        populated_system.set_relationship(1, 2, TeamRelation.FRIENDLY)
        relation = populated_system.get_relationship(1, 2)
        assert relation == TeamRelation.FRIENDLY

    def test_relationship_bidirectional(self, populated_system):
        """Relationships should be bidirectional."""
        populated_system.set_relationship(1, 2, TeamRelation.NEUTRAL)

        assert populated_system.get_relationship(1, 2) == TeamRelation.NEUTRAL
        assert populated_system.get_relationship(2, 1) == TeamRelation.NEUTRAL

    def test_set_all_hostile(self, populated_system):
        """set_all_hostile should make all teams hostile."""
        populated_system.set_relationship(1, 2, TeamRelation.FRIENDLY)
        populated_system.set_all_hostile()

        assert populated_system.get_relationship(1, 2) == TeamRelation.HOSTILE

    def test_set_all_friendly(self, populated_system):
        """set_all_friendly should make all teams friendly."""
        populated_system.set_all_friendly()
        assert populated_system.get_relationship(1, 2) == TeamRelation.FRIENDLY

    def test_team_specific_override(self, team_system):
        """Team can have relationship override."""
        team_system.create_team(1)
        team_system.create_team(2)

        team1 = team_system.get_team(1)
        team1.set_relationship_override(2, TeamRelation.FRIENDLY)

        relation = team_system.get_relationship(1, 2)
        assert relation == TeamRelation.FRIENDLY

    def test_clear_relationship_override(self, team_system):
        """Should clear relationship override."""
        team_system.create_team(1)
        team_system.create_team(2)

        team1 = team_system.get_team(1)
        team1.set_relationship_override(2, TeamRelation.FRIENDLY)
        team1.clear_relationship_override(2)

        # Should fall back to global
        relation = team_system.get_relationship(1, 2)
        assert relation == TeamRelation.HOSTILE


# =============================================================================
# IFF CALCULATION TESTS (35 tests)
# =============================================================================


class TestIFFCalculation:
    """Tests for IFF (Identify Friend/Foe) calculations."""

    def test_iff_same_team(self, populated_system):
        """IFF between same team members."""
        iff = populated_system.check_iff(100, 101)
        assert iff.is_same_team
        assert iff.is_friendly

    def test_iff_different_teams(self, populated_system):
        """IFF between different teams."""
        iff = populated_system.check_iff(100, 200)
        assert not iff.is_same_team
        assert iff.is_hostile

    def test_iff_source_and_target_ids(self, populated_system):
        """IFF should have correct source/target."""
        iff = populated_system.check_iff(100, 200)
        assert iff.source_id == 100
        assert iff.target_id == 200

    def test_iff_source_and_target_teams(self, populated_system):
        """IFF should have correct team IDs."""
        iff = populated_system.check_iff(100, 200)
        assert iff.source_team == 1
        assert iff.target_team == 2

    def test_iff_can_damage_hostile(self, populated_system):
        """Can damage hostile targets."""
        iff = populated_system.check_iff(100, 200)
        assert iff.can_damage

    def test_iff_cannot_damage_friendly(self, populated_system):
        """Cannot damage friendly by default."""
        iff = populated_system.check_iff(100, 101)
        assert not iff.can_damage

    def test_iff_can_damage_friendly_with_ff(self, team_system):
        """Can damage friendly with friendly fire enabled."""
        team_system._config = TeamConfig(allow_team_damage=True)
        team_system.create_team(1, friendly_fire_multiplier=1.0)
        team_system.set_team(100, 1)
        team_system.set_team(101, 1)

        iff = team_system.check_iff(100, 101)
        assert iff.can_damage

    def test_iff_can_heal_friendly(self, populated_system):
        """Can heal friendly targets."""
        iff = populated_system.check_iff(100, 101)
        assert iff.can_heal

    def test_iff_cannot_heal_hostile(self, populated_system):
        """Cannot heal hostile targets."""
        iff = populated_system.check_iff(100, 200)
        assert not iff.can_heal

    def test_iff_friendly_fire_multiplier_same_team(self, team_system):
        """FF multiplier for same team."""
        team_system.create_team(1, friendly_fire_multiplier=0.5)
        team_system.set_team(100, 1)
        team_system.set_team(101, 1)

        iff = team_system.check_iff(100, 101)
        assert iff.friendly_fire_multiplier == 0.5

    def test_iff_friendly_fire_multiplier_different_teams(self, populated_system):
        """FF multiplier for different teams is 1.0."""
        iff = populated_system.check_iff(100, 200)
        assert iff.friendly_fire_multiplier == 1.0

    def test_iff_is_friendly_property(self, populated_system):
        """is_friendly should check relation."""
        iff = populated_system.check_iff(100, 101)
        assert iff.is_friendly

    def test_iff_is_hostile_property(self, populated_system):
        """is_hostile should check relation."""
        iff = populated_system.check_iff(100, 200)
        assert iff.is_hostile

    def test_iff_is_neutral_property(self, team_system):
        """is_neutral should check relation."""
        team_system.create_team(1)
        team_system.set_team(100, 1)
        team_system.set_team(200, NEUTRAL_TEAM_ID)

        iff = team_system.check_iff(100, 200)
        assert iff.is_neutral

    def test_can_attack_helper(self, populated_system):
        """can_attack helper should work."""
        assert populated_system.can_attack(100, 200)
        assert not populated_system.can_attack(100, 101)

    def test_can_heal_helper(self, populated_system):
        """can_heal helper should work."""
        assert populated_system.can_heal(100, 101)
        assert not populated_system.can_heal(100, 200)

    def test_is_friendly_helper(self, populated_system):
        """is_friendly helper should work."""
        assert populated_system.is_friendly(100, 101)
        assert not populated_system.is_friendly(100, 200)

    def test_is_hostile_helper(self, populated_system):
        """is_hostile helper should work."""
        assert populated_system.is_hostile(100, 200)
        assert not populated_system.is_hostile(100, 101)

    def test_get_friendly_fire_multiplier_helper(self, team_system):
        """get_friendly_fire_multiplier helper should work."""
        team_system.create_team(1, friendly_fire_multiplier=0.25)
        team_system.set_team(100, 1)
        team_system.set_team(101, 1)

        mult = team_system.get_friendly_fire_multiplier(100, 101)
        assert mult == 0.25


# =============================================================================
# FRIENDLY FIRE TESTS (20 tests)
# =============================================================================


class TestFriendlyFire:
    """Tests for friendly fire configuration."""

    def test_default_friendly_fire(self, team_system):
        """Default should be no friendly fire."""
        team = team_system.get_team(DEFAULT_TEAM_ID)
        assert team.friendly_fire_multiplier == FRIENDLY_FIRE_NONE

    def test_set_friendly_fire(self, team_system):
        """set_friendly_fire should change multiplier."""
        team_system.create_team(1)
        result = team_system.set_friendly_fire(1, 0.5)
        assert result

        team = team_system.get_team(1)
        assert team.friendly_fire_multiplier == 0.5

    def test_set_friendly_fire_clamped_min(self, team_system):
        """Friendly fire should be clamped to 0."""
        team_system.create_team(1)
        team_system.set_friendly_fire(1, -0.5)

        team = team_system.get_team(1)
        assert team.friendly_fire_multiplier >= 0.0

    def test_set_friendly_fire_clamped_max(self, team_system):
        """Friendly fire should be clamped to 1."""
        team_system.create_team(1)
        team_system.set_friendly_fire(1, 1.5)

        team = team_system.get_team(1)
        assert team.friendly_fire_multiplier <= 1.0

    def test_enable_friendly_fire(self, team_system):
        """enable_friendly_fire should enable FF."""
        team_system.create_team(1)
        result = team_system.enable_friendly_fire(1)
        assert result

        team = team_system.get_team(1)
        assert team.friendly_fire_multiplier == FRIENDLY_FIRE_FULL

    def test_enable_friendly_fire_custom(self, team_system):
        """enable_friendly_fire with custom value."""
        team_system.create_team(1)
        team_system.enable_friendly_fire(1, 0.3)

        team = team_system.get_team(1)
        assert team.friendly_fire_multiplier == 0.3

    def test_disable_friendly_fire(self, team_system):
        """disable_friendly_fire should disable FF."""
        team_system.create_team(1, friendly_fire_multiplier=1.0)
        result = team_system.disable_friendly_fire(1)
        assert result

        team = team_system.get_team(1)
        assert team.friendly_fire_multiplier == FRIENDLY_FIRE_NONE


# =============================================================================
# EVENT HANDLING TESTS (15 tests)
# =============================================================================


class TestEventHandling:
    """Tests for event handling."""

    def test_on_team_change_registered(self, team_system):
        """on_team_change should register handler."""
        callback = Mock()
        team_system.on_team_change(callback)
        assert callback in team_system._on_team_change

    def test_team_change_emitted(self, team_system):
        """Team change should emit event."""
        callback = Mock()
        team_system.on_team_change(callback)

        team_system.create_team(1)
        team_system.set_team(100, 1)

        callback.assert_called()

    def test_team_change_event_has_correct_values(self, team_system):
        """Team change event should have correct values."""
        events = []
        team_system.on_team_change(lambda e: events.append(e))

        team_system.create_team(1)
        team_system.create_team(2)
        team_system.set_team(100, 1)
        team_system.set_team(100, 2)

        # Should have two events
        assert len(events) == 2
        change = events[1]
        assert change.entity_id == 100
        assert change.old_team_id == 1
        assert change.new_team_id == 2

    def test_on_team_created(self, team_system):
        """on_team_created should fire on team creation."""
        callback = Mock()
        team_system.on_team_created(callback)

        team_system.create_team(1, name="Test")
        callback.assert_called()

    def test_on_team_removed(self, team_system):
        """on_team_removed should fire on team removal."""
        callback = Mock()
        team_system.on_team_removed(callback)

        team_system.create_team(1)
        team_system.remove_team(1)

        callback.assert_called_with(1)


# =============================================================================
# QUERY TESTS (20 tests)
# =============================================================================


class TestQueries:
    """Tests for query methods."""

    def test_get_enemies(self, populated_system):
        """get_enemies should return hostile entities."""
        enemies = populated_system.get_enemies(100)
        assert 200 in enemies
        assert 201 in enemies
        assert 101 not in enemies

    def test_get_allies(self, populated_system):
        """get_allies should return team members."""
        allies = populated_system.get_allies(100)
        assert 101 in allies
        assert 100 not in allies  # Not self
        assert 200 not in allies

    def test_get_hostile_teams(self, populated_system):
        """get_hostile_teams should return hostile teams."""
        hostile = populated_system.get_hostile_teams(1)
        assert 2 in hostile
        assert 1 not in hostile

    def test_get_allied_teams(self, team_system):
        """get_allied_teams should return allied teams."""
        team_system.create_team(1)
        team_system.create_team(2)
        team_system.create_team(3)
        team_system.set_relationship(1, 2, TeamRelation.FRIENDLY)

        allied = team_system.get_allied_teams(1)
        assert 1 in allied  # Self
        assert 2 in allied
        assert 3 not in allied


# =============================================================================
# AUTO-BALANCE TESTS (15 tests)
# =============================================================================


class TestAutoBalance:
    """Tests for auto-balance functionality."""

    def test_get_team_with_fewest_members(self, team_system):
        """get_team_with_fewest_members should return smallest team."""
        team_system.create_team(1)
        team_system.create_team(2)
        team_system.set_team(100, 1)
        team_system.set_team(101, 1)
        team_system.set_team(200, 2)

        # Exclude neutral and default teams
        smallest = team_system.get_team_with_fewest_members(
            exclude={NEUTRAL_TEAM_ID, DEFAULT_TEAM_ID}
        )
        assert smallest == 2

    def test_get_team_with_fewest_excludes(self, team_system):
        """Should respect exclude set."""
        team_system.create_team(1)
        team_system.create_team(2)
        team_system.set_team(200, 2)

        smallest = team_system.get_team_with_fewest_members(
            exclude={2, NEUTRAL_TEAM_ID, DEFAULT_TEAM_ID}
        )
        assert smallest == 1

    def test_auto_assign_team(self, team_system):
        """auto_assign_team should assign to smallest."""
        team_system.create_team(1)
        team_system.create_team(2)
        team_system.set_team(100, 1)
        team_system.set_team(101, 1)

        # Exclude neutral and default to force assignment to created teams
        assigned = team_system.auto_assign_team(200, exclude={NEUTRAL_TEAM_ID, DEFAULT_TEAM_ID})
        assert assigned == 2
        assert team_system.get_team_id(200) == 2

    def test_auto_assign_skips_full_teams(self, team_system):
        """auto_assign should skip full teams."""
        team_system.create_team(1, max_members=0)  # Unlimited
        team_system.create_team(2, max_members=1)
        team_system.set_team(200, 2)  # Fill team 2

        assigned = team_system.auto_assign_team(100)
        assert assigned != 2


# =============================================================================
# UTILITY TESTS (10 tests)
# =============================================================================


class TestUtility:
    """Tests for utility methods."""

    def test_clear(self, populated_system):
        """clear should reset to defaults."""
        populated_system.clear()

        assert len(populated_system._memberships) == 0
        assert populated_system.team_exists(DEFAULT_TEAM_ID)
        assert populated_system.team_exists(NEUTRAL_TEAM_ID)

    def test_team_info_is_full(self, team_system):
        """TeamInfo.is_full should check capacity."""
        team = team_system.create_team(1, max_members=2)
        team_system.set_team(100, 1)
        assert not team.is_full

        team_system.set_team(101, 1)
        assert team.is_full

    def test_team_info_unlimited_not_full(self, team_system):
        """Unlimited team is never full."""
        team = team_system.create_team(1, max_members=0)
        for i in range(100):
            team_system.set_team(i, 1)
        assert not team.is_full

    def test_membership_duration(self, team_system):
        """Membership should track duration."""
        import time

        team_system.create_team(1)
        team_system.set_team(100, 1)

        membership = team_system.get_membership(100)
        assert membership.membership_duration >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
