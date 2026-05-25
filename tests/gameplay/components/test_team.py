"""
Comprehensive tests for TeamComponent.

Tests cover:
- Team assignment
- Team identification
- Friendly/enemy detection
- Neutral teams
- Team switching
- Team-based filtering
- Alliance systems
- Team events
"""

import pytest
from typing import List

from engine.gameplay.components.team import (
    TeamComponent,
    TeamRelation,
    IFFResponse,
    Team,
    Faction,
    TeamRegistry,
    get_team_registry,
    set_team_registry,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def reset_team_registry():
    """Reset team registry before each test."""
    set_team_registry(TeamRegistry())
    yield
    set_team_registry(TeamRegistry())


@pytest.fixture
def registry():
    """Get the current team registry."""
    return get_team_registry()


@pytest.fixture
def team_component():
    """Create a default team component."""
    return TeamComponent()


@pytest.fixture
def player_team_component():
    """Create a player team component."""
    return TeamComponent(
        team_id=1,
        iff_tags=IFFResponse.FRIEND | IFFResponse.PLAYER,
        entity_id="player_1"
    )


@pytest.fixture
def enemy_team_component():
    """Create an enemy team component."""
    return TeamComponent(
        team_id=2,
        iff_tags=IFFResponse.FOE | IFFResponse.AI,
        entity_id="enemy_1"
    )


@pytest.fixture
def setup_factions(registry):
    """Set up factions for testing."""
    allies = Faction(id="allies", name="Allied Forces", color=(0, 255, 0))
    axis = Faction(id="axis", name="Axis Powers", color=(255, 0, 0))
    neutral = Faction(id="neutral", name="Neutral", color=(128, 128, 128))
    registry.register_faction(allies)
    registry.register_faction(axis)
    registry.register_faction(neutral)
    return allies, axis, neutral


@pytest.fixture
def setup_teams(registry, setup_factions):
    """Set up teams for testing."""
    allies, axis, neutral = setup_factions
    team1 = registry.create_team("Alpha Squad", faction=allies, team_id=1)
    team2 = registry.create_team("Bravo Squad", faction=allies, team_id=2)
    team3 = registry.create_team("Enemy Squad", faction=axis, team_id=3)
    team4 = registry.create_team("Civilians", faction=neutral, team_id=4)
    return team1, team2, team3, team4


# =============================================================================
# FACTION TESTS
# =============================================================================


class TestFaction:
    """Tests for Faction class."""

    def test_faction_creation(self):
        """Test faction creation."""
        faction = Faction(id="test", name="Test Faction")
        assert faction.id == "test"
        assert faction.name == "Test Faction"

    def test_faction_with_color(self):
        """Test faction with custom color."""
        faction = Faction(id="test", name="Test", color=(255, 0, 0))
        assert faction.color == (255, 0, 0)

    def test_faction_with_description(self):
        """Test faction with description."""
        faction = Faction(id="test", name="Test", description="A test faction")
        assert faction.description == "A test faction"

    def test_faction_with_metadata(self):
        """Test faction with metadata."""
        faction = Faction(id="test", name="Test", metadata={"power": 100})
        assert faction.metadata["power"] == 100

    def test_faction_equality(self):
        """Test faction equality."""
        f1 = Faction(id="test", name="Test")
        f2 = Faction(id="test", name="Different Name")
        assert f1 == f2  # Same ID

    def test_faction_hash(self):
        """Test faction hashing."""
        f1 = Faction(id="test", name="Test")
        f2 = Faction(id="test", name="Test")
        assert hash(f1) == hash(f2)


# =============================================================================
# TEAM TESTS
# =============================================================================


class TestTeam:
    """Tests for Team class."""

    def test_team_creation(self):
        """Test team creation."""
        team = Team(id=1, name="Test Team")
        assert team.id == 1
        assert team.name == "Test Team"

    def test_team_with_faction(self, setup_factions):
        """Test team with faction."""
        allies, _, _ = setup_factions
        team = Team(id=1, name="Test", faction=allies)
        assert team.faction == allies

    def test_team_display_color_from_team(self):
        """Test display color from team color."""
        team = Team(id=1, name="Test", color=(255, 0, 0))
        assert team.display_color == (255, 0, 0)

    def test_team_display_color_from_faction(self, setup_factions):
        """Test display color from faction when no team color."""
        allies, _, _ = setup_factions
        team = Team(id=1, name="Test", faction=allies)
        assert team.display_color == allies.color

    def test_team_display_color_default(self):
        """Test default display color."""
        team = Team(id=1, name="Test")
        assert team.display_color == (128, 128, 128)

    def test_team_max_members(self):
        """Test team max members."""
        team = Team(id=1, name="Test", max_members=10)
        assert team.max_members == 10

    def test_team_equality(self):
        """Test team equality."""
        t1 = Team(id=1, name="Test")
        t2 = Team(id=1, name="Different")
        assert t1 == t2  # Same ID

    def test_team_hash(self):
        """Test team hashing."""
        t1 = Team(id=1, name="Test")
        t2 = Team(id=1, name="Test")
        assert hash(t1) == hash(t2)


# =============================================================================
# TEAM REGISTRY FACTION MANAGEMENT TESTS
# =============================================================================


class TestTeamRegistryFactions:
    """Tests for TeamRegistry faction management."""

    def test_register_faction(self, registry):
        """Test registering a faction."""
        faction = Faction(id="test", name="Test")
        registry.register_faction(faction)
        assert registry.get_faction("test") == faction

    def test_get_faction_not_found(self, registry):
        """Test getting non-existent faction."""
        assert registry.get_faction("nonexistent") is None

    def test_get_all_factions(self, registry, setup_factions):
        """Test getting all factions."""
        factions = registry.get_all_factions()
        assert len(factions) == 3

    def test_remove_faction(self, registry, setup_factions):
        """Test removing a faction."""
        result = registry.remove_faction("allies")
        assert result is True
        assert registry.get_faction("allies") is None

    def test_remove_faction_not_found(self, registry):
        """Test removing non-existent faction."""
        result = registry.remove_faction("nonexistent")
        assert result is False

    def test_remove_faction_clears_team_faction(self, registry, setup_factions, setup_teams):
        """Test removing faction clears from teams."""
        team1, _, _, _ = setup_teams
        assert team1.faction is not None
        registry.remove_faction("allies")
        assert team1.faction is None


# =============================================================================
# TEAM REGISTRY TEAM MANAGEMENT TESTS
# =============================================================================


class TestTeamRegistryTeams:
    """Tests for TeamRegistry team management."""

    def test_create_team(self, registry):
        """Test creating a team."""
        team = registry.create_team("Test Team")
        assert team.name == "Test Team"
        assert team.id > 0

    def test_create_team_with_id(self, registry):
        """Test creating team with specific ID."""
        team = registry.create_team("Test", team_id=100)
        assert team.id == 100

    def test_create_team_with_faction(self, registry, setup_factions):
        """Test creating team with faction."""
        allies, _, _ = setup_factions
        team = registry.create_team("Test", faction=allies)
        assert team.faction == allies

    def test_create_team_with_color(self, registry):
        """Test creating team with color."""
        team = registry.create_team("Test", color=(255, 0, 0))
        assert team.color == (255, 0, 0)

    def test_get_team(self, registry, setup_teams):
        """Test getting a team."""
        team1, _, _, _ = setup_teams
        assert registry.get_team(1) == team1

    def test_get_team_not_found(self, registry):
        """Test getting non-existent team."""
        assert registry.get_team(999) is None

    def test_get_teams_by_faction(self, registry, setup_teams):
        """Test getting teams by faction."""
        allies_teams = registry.get_teams_by_faction("allies")
        assert len(allies_teams) == 2

    def test_get_all_teams(self, registry, setup_teams):
        """Test getting all teams."""
        teams = registry.get_all_teams()
        assert len(teams) == 4

    def test_remove_team(self, registry, setup_teams):
        """Test removing a team."""
        result = registry.remove_team(1)
        assert result is True
        assert registry.get_team(1) is None

    def test_remove_team_not_found(self, registry):
        """Test removing non-existent team."""
        result = registry.remove_team(999)
        assert result is False


# =============================================================================
# TEAM REGISTRY RELATIONS TESTS
# =============================================================================


class TestTeamRegistryRelations:
    """Tests for TeamRegistry relationship management."""

    def test_set_relation(self, registry, setup_teams):
        """Test setting team relation."""
        registry.set_relation(1, 3, TeamRelation.HOSTILE)
        assert registry.get_relation(1, 3) == TeamRelation.HOSTILE

    def test_relation_bidirectional(self, registry, setup_teams):
        """Test relations are bidirectional."""
        registry.set_relation(1, 3, TeamRelation.HOSTILE)
        assert registry.get_relation(3, 1) == TeamRelation.HOSTILE

    def test_self_relation(self, registry, setup_teams):
        """Test self relation."""
        assert registry.get_relation(1, 1) == TeamRelation.SELF

    def test_same_faction_ally_default(self, registry, setup_teams):
        """Test same faction teams default to ally."""
        # Teams 1 and 2 are both in allies faction
        relation = registry.get_relation(1, 2)
        assert relation == TeamRelation.ALLY

    def test_different_faction_neutral_default(self, registry, setup_teams):
        """Test different faction teams default to neutral."""
        # Team 1 (allies) and team 4 (neutral)
        relation = registry.get_relation(1, 4)
        assert relation == TeamRelation.NEUTRAL

    def test_explicit_relation_overrides_faction(self, registry, setup_teams):
        """Test explicit relation overrides faction-based default."""
        # Teams 1 and 2 are allies by faction
        registry.set_relation(1, 2, TeamRelation.HOSTILE)
        assert registry.get_relation(1, 2) == TeamRelation.HOSTILE

    def test_clear_relations(self, registry, setup_teams):
        """Test clearing all relations."""
        registry.set_relation(1, 2, TeamRelation.HOSTILE)
        registry.clear_relations()
        # Should revert to faction-based (ally)
        assert registry.get_relation(1, 2) == TeamRelation.ALLY

    def test_get_allies(self, registry, setup_teams):
        """Test getting allied teams."""
        # Team 1 and 2 are in same faction (allies)
        allies = registry.get_allies(1)
        assert 2 in allies

    def test_get_enemies(self, registry, setup_teams):
        """Test getting enemy teams."""
        registry.set_relation(1, 3, TeamRelation.HOSTILE)
        enemies = registry.get_enemies(1)
        assert 3 in enemies

    def test_remove_team_clears_relations(self, registry, setup_teams):
        """Test removing team clears its relations."""
        registry.set_relation(1, 3, TeamRelation.HOSTILE)
        registry.remove_team(1)
        # Relation should be gone
        assert registry.get_relation(1, 3) == TeamRelation.NEUTRAL


# =============================================================================
# TEAM COMPONENT INITIALIZATION TESTS
# =============================================================================


class TestTeamComponentInitialization:
    """Tests for TeamComponent initialization."""

    def test_default_initialization(self, team_component):
        """Test default team component values."""
        assert team_component.team_id == 0
        assert team_component.iff_tags == IFFResponse.UNKNOWN

    def test_initialization_with_team_id(self):
        """Test initialization with team ID."""
        tc = TeamComponent(team_id=5)
        assert tc.team_id == 5

    def test_initialization_with_iff_tags(self):
        """Test initialization with IFF tags."""
        tc = TeamComponent(iff_tags=IFFResponse.FRIEND)
        assert tc.iff_tags == IFFResponse.FRIEND

    def test_initialization_with_entity_id(self):
        """Test initialization with entity ID."""
        tc = TeamComponent(entity_id="entity_123")
        assert tc._entity_id == "entity_123"


# =============================================================================
# TEAM MEMBERSHIP TESTS
# =============================================================================


class TestTeamMembership:
    """Tests for team membership."""

    def test_get_team(self, registry, setup_teams):
        """Test getting team object."""
        tc = TeamComponent(team_id=1)
        assert tc.team is not None
        assert tc.team.name == "Alpha Squad"

    def test_get_team_not_found(self, team_component):
        """Test getting team when not in registry."""
        assert team_component.team is None

    def test_get_faction(self, registry, setup_teams):
        """Test getting faction."""
        tc = TeamComponent(team_id=1)
        assert tc.faction is not None
        assert tc.faction.id == "allies"

    def test_get_faction_id(self, registry, setup_teams):
        """Test getting faction ID."""
        tc = TeamComponent(team_id=1)
        assert tc.faction_id == "allies"

    def test_set_team(self, registry, setup_teams):
        """Test setting team."""
        tc = TeamComponent(team_id=1)
        tc.set_team(2)
        assert tc.team_id == 2

    def test_join_team(self, registry, setup_teams):
        """Test joining a team."""
        team1, _, _, _ = setup_teams
        tc = TeamComponent()
        tc.join_team(team1)
        assert tc.team_id == team1.id

    def test_leave_team(self, registry, setup_teams):
        """Test leaving a team."""
        tc = TeamComponent(team_id=1)
        tc.leave_team()
        assert tc.team_id == 0

    def test_team_change_callback(self, registry, setup_teams):
        """Test team change callback."""
        changes = []
        tc = TeamComponent(team_id=1)
        tc.on_team_changed(lambda old, new: changes.append((old, new)))
        tc.set_team(2)
        assert len(changes) == 1
        assert changes[0] == (1, 2)

    def test_team_change_same_team_no_callback(self, registry, setup_teams):
        """Test no callback when setting same team."""
        changes = []
        tc = TeamComponent(team_id=1)
        tc.on_team_changed(lambda old, new: changes.append((old, new)))
        tc.set_team(1)  # Same team
        assert len(changes) == 0


# =============================================================================
# SECONDARY TEAM TESTS
# =============================================================================


class TestSecondaryTeams:
    """Tests for secondary team memberships."""

    def test_add_secondary_team(self, team_component):
        """Test adding secondary team."""
        team_component.add_secondary_team(2)
        assert 2 in team_component.secondary_teams

    def test_remove_secondary_team(self, team_component):
        """Test removing secondary team."""
        team_component.add_secondary_team(2)
        team_component.remove_secondary_team(2)
        assert 2 not in team_component.secondary_teams

    def test_clear_secondary_teams(self, team_component):
        """Test clearing secondary teams."""
        team_component.add_secondary_team(2)
        team_component.add_secondary_team(3)
        team_component.clear_secondary_teams()
        assert len(team_component.secondary_teams) == 0

    def test_all_teams(self, team_component):
        """Test getting all teams."""
        team_component.team_id = 1
        team_component.add_secondary_team(2)
        team_component.add_secondary_team(3)
        all_teams = team_component.all_teams
        assert 1 in all_teams
        assert 2 in all_teams
        assert 3 in all_teams

    def test_is_member_of_primary(self, team_component):
        """Test is_member_of for primary team."""
        team_component.team_id = 1
        assert team_component.is_member_of(1) is True

    def test_is_member_of_secondary(self, team_component):
        """Test is_member_of for secondary team."""
        team_component.add_secondary_team(2)
        assert team_component.is_member_of(2) is True

    def test_is_member_of_false(self, team_component):
        """Test is_member_of returns false for non-member."""
        assert team_component.is_member_of(99) is False


# =============================================================================
# IFF TAG TESTS
# =============================================================================


class TestIFFTags:
    """Tests for IFF (Identification Friend or Foe) tags."""

    def test_add_iff_tag(self, team_component):
        """Test adding IFF tag."""
        team_component.add_iff_tag(IFFResponse.FRIEND)
        assert team_component.has_iff_tag(IFFResponse.FRIEND)

    def test_remove_iff_tag(self, team_component):
        """Test removing IFF tag."""
        team_component.iff_tags = IFFResponse.FRIEND | IFFResponse.PLAYER
        team_component.remove_iff_tag(IFFResponse.FRIEND)
        assert not team_component.has_iff_tag(IFFResponse.FRIEND)
        assert team_component.has_iff_tag(IFFResponse.PLAYER)

    def test_has_iff_tag(self, player_team_component):
        """Test has_iff_tag."""
        assert player_team_component.has_iff_tag(IFFResponse.FRIEND)
        assert player_team_component.has_iff_tag(IFFResponse.PLAYER)
        assert not player_team_component.has_iff_tag(IFFResponse.FOE)

    def test_is_friendly_iff(self, player_team_component):
        """Test is_friendly_iff."""
        assert player_team_component.is_friendly_iff() is True

    def test_is_hostile_iff(self, enemy_team_component):
        """Test is_hostile_iff."""
        assert enemy_team_component.is_hostile_iff() is True

    def test_is_player(self, player_team_component):
        """Test is_player."""
        assert player_team_component.is_player() is True

    def test_is_ai(self, enemy_team_component):
        """Test is_ai."""
        assert enemy_team_component.is_ai() is True

    def test_multiple_iff_tags(self):
        """Test multiple IFF tags combined."""
        tc = TeamComponent(
            iff_tags=IFFResponse.FRIEND | IFFResponse.PLAYER | IFFResponse.OBJECTIVE
        )
        assert tc.has_iff_tag(IFFResponse.FRIEND)
        assert tc.has_iff_tag(IFFResponse.PLAYER)
        assert tc.has_iff_tag(IFFResponse.OBJECTIVE)


# =============================================================================
# RELATIONSHIP QUERY TESTS
# =============================================================================


class TestRelationshipQueries:
    """Tests for relationship queries."""

    def test_get_relation_to_same_team(self, registry, setup_teams):
        """Test relation to same team."""
        tc1 = TeamComponent(team_id=1)
        tc2 = TeamComponent(team_id=1)
        assert tc1.get_relation_to(tc2) == TeamRelation.SELF

    def test_get_relation_to_ally(self, registry, setup_teams):
        """Test relation to ally."""
        tc1 = TeamComponent(team_id=1)
        tc2 = TeamComponent(team_id=2)
        assert tc1.get_relation_to(tc2) == TeamRelation.ALLY

    def test_get_relation_to_enemy(self, registry, setup_teams):
        """Test relation to enemy."""
        registry.set_relation(1, 3, TeamRelation.HOSTILE)
        tc1 = TeamComponent(team_id=1)
        tc2 = TeamComponent(team_id=3)
        assert tc1.get_relation_to(tc2) == TeamRelation.HOSTILE

    def test_get_relation_to_neutral(self, registry, setup_teams):
        """Test relation to neutral."""
        tc1 = TeamComponent(team_id=1)
        tc2 = TeamComponent(team_id=4)
        assert tc1.get_relation_to(tc2) == TeamRelation.NEUTRAL

    def test_is_ally_same_team(self, registry, setup_teams):
        """Test is_ally for same team."""
        tc1 = TeamComponent(team_id=1)
        tc2 = TeamComponent(team_id=1)
        assert tc1.is_ally(tc2) is True

    def test_is_ally_allied_team(self, registry, setup_teams):
        """Test is_ally for allied team."""
        tc1 = TeamComponent(team_id=1)
        tc2 = TeamComponent(team_id=2)
        assert tc1.is_ally(tc2) is True

    def test_is_enemy(self, registry, setup_teams):
        """Test is_enemy."""
        registry.set_relation(1, 3, TeamRelation.HOSTILE)
        tc1 = TeamComponent(team_id=1)
        tc2 = TeamComponent(team_id=3)
        assert tc1.is_enemy(tc2) is True

    def test_is_neutral(self, registry, setup_teams):
        """Test is_neutral."""
        tc1 = TeamComponent(team_id=1)
        tc2 = TeamComponent(team_id=4)
        assert tc1.is_neutral(tc2) is True

    def test_is_same_team(self, registry, setup_teams):
        """Test is_same_team."""
        tc1 = TeamComponent(team_id=1)
        tc2 = TeamComponent(team_id=1)
        assert tc1.is_same_team(tc2) is True

    def test_is_same_team_false(self, registry, setup_teams):
        """Test is_same_team returns false."""
        tc1 = TeamComponent(team_id=1)
        tc2 = TeamComponent(team_id=2)
        assert tc1.is_same_team(tc2) is False

    def test_is_same_faction(self, registry, setup_teams):
        """Test is_same_faction."""
        tc1 = TeamComponent(team_id=1)
        tc2 = TeamComponent(team_id=2)
        assert tc1.is_same_faction(tc2) is True

    def test_is_same_faction_false(self, registry, setup_teams):
        """Test is_same_faction returns false."""
        tc1 = TeamComponent(team_id=1)
        tc2 = TeamComponent(team_id=3)
        assert tc1.is_same_faction(tc2) is False

    def test_is_same_faction_no_faction(self, team_component):
        """Test is_same_faction when no faction."""
        tc1 = TeamComponent()
        tc2 = TeamComponent()
        assert tc1.is_same_faction(tc2) is False

    def test_shares_any_team_primary(self, registry, setup_teams):
        """Test shares_any_team with primary teams."""
        tc1 = TeamComponent(team_id=1)
        tc2 = TeamComponent(team_id=1)
        assert tc1.shares_any_team(tc2) is True

    def test_shares_any_team_secondary(self, team_component):
        """Test shares_any_team with secondary teams."""
        tc1 = TeamComponent(team_id=1)
        tc1.add_secondary_team(5)
        tc2 = TeamComponent(team_id=2)
        tc2.add_secondary_team(5)
        assert tc1.shares_any_team(tc2) is True

    def test_shares_any_team_false(self, team_component):
        """Test shares_any_team returns false."""
        tc1 = TeamComponent(team_id=1)
        tc2 = TeamComponent(team_id=2)
        assert tc1.shares_any_team(tc2) is False


# =============================================================================
# CUSTOM RELATION TESTS
# =============================================================================


class TestCustomRelations:
    """Tests for custom entity-specific relations."""

    def test_set_custom_relation(self, registry, setup_teams):
        """Test setting custom relation."""
        tc1 = TeamComponent(team_id=1, entity_id="entity_1")
        tc2 = TeamComponent(team_id=2, entity_id="entity_2")
        tc1.set_custom_relation("entity_2", TeamRelation.HOSTILE)
        assert tc1.get_relation_to(tc2) == TeamRelation.HOSTILE

    def test_custom_relation_overrides_team(self, registry, setup_teams):
        """Test custom relation overrides team relation."""
        # Teams 1 and 2 are allies by faction
        tc1 = TeamComponent(team_id=1, entity_id="entity_1")
        tc2 = TeamComponent(team_id=2, entity_id="entity_2")
        tc1.set_custom_relation("entity_2", TeamRelation.HOSTILE)
        assert tc1.get_relation_to(tc2) == TeamRelation.HOSTILE

    def test_clear_custom_relation(self, registry, setup_teams):
        """Test clearing custom relation."""
        tc1 = TeamComponent(team_id=1, entity_id="entity_1")
        tc2 = TeamComponent(team_id=2, entity_id="entity_2")
        tc1.set_custom_relation("entity_2", TeamRelation.HOSTILE)
        tc1.clear_custom_relation("entity_2")
        assert tc1.get_relation_to(tc2) == TeamRelation.ALLY  # Back to faction-based

    def test_clear_all_custom_relations(self, registry, setup_teams):
        """Test clearing all custom relations."""
        tc1 = TeamComponent(team_id=1, entity_id="entity_1")
        tc1.set_custom_relation("entity_2", TeamRelation.HOSTILE)
        tc1.set_custom_relation("entity_3", TeamRelation.HOSTILE)
        tc1.clear_all_custom_relations()
        tc2 = TeamComponent(team_id=2, entity_id="entity_2")
        assert tc1.get_relation_to(tc2) == TeamRelation.ALLY


# =============================================================================
# DISPLAY HELPER TESTS
# =============================================================================


class TestDisplayHelpers:
    """Tests for display helper properties."""

    def test_team_name(self, registry, setup_teams):
        """Test team_name property."""
        tc = TeamComponent(team_id=1)
        assert tc.team_name == "Alpha Squad"

    def test_team_name_no_team(self, team_component):
        """Test team_name when no team."""
        assert team_component.team_name == "No Team"

    def test_faction_name(self, registry, setup_teams):
        """Test faction_name property."""
        tc = TeamComponent(team_id=1)
        assert tc.faction_name == "Allied Forces"

    def test_faction_name_no_faction(self, team_component):
        """Test faction_name when no faction."""
        assert team_component.faction_name == "No Faction"

    def test_team_color(self, registry, setup_teams):
        """Test team_color property."""
        tc = TeamComponent(team_id=1)
        # Should inherit faction color (allies = green)
        assert tc.team_color == (0, 255, 0)

    def test_team_color_no_team(self, team_component):
        """Test team_color when no team."""
        assert team_component.team_color == (128, 128, 128)


# =============================================================================
# CALLBACK TESTS
# =============================================================================


class TestCallbacks:
    """Tests for team change callbacks."""

    def test_on_team_changed(self, registry, setup_teams):
        """Test on_team_changed callback."""
        changes = []
        tc = TeamComponent(team_id=1)
        tc.on_team_changed(lambda old, new: changes.append((old, new)))
        tc.set_team(2)
        assert len(changes) == 1
        assert changes[0] == (1, 2)

    def test_off_team_changed(self, registry, setup_teams):
        """Test unregistering team change callback."""
        changes = []
        callback = lambda old, new: changes.append((old, new))
        tc = TeamComponent(team_id=1)
        tc.on_team_changed(callback)
        tc.off_team_changed(callback)
        tc.set_team(2)
        assert len(changes) == 0

    def test_multiple_callbacks(self, registry, setup_teams):
        """Test multiple team change callbacks."""
        count = [0]
        tc = TeamComponent(team_id=1)
        tc.on_team_changed(lambda old, new: count.__setitem__(0, count[0] + 1))
        tc.on_team_changed(lambda old, new: count.__setitem__(0, count[0] + 1))
        tc.set_team(2)
        assert count[0] == 2


# =============================================================================
# SERIALIZATION TESTS
# =============================================================================


class TestSerialization:
    """Tests for team component serialization."""

    def test_to_dict(self, player_team_component):
        """Test serialization to dictionary."""
        data = player_team_component.to_dict()
        assert "team_id" in data
        assert "iff_tags" in data
        assert "secondary_teams" in data
        assert "custom_relations" in data
        assert "entity_id" in data

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "team_id": 5,
            "iff_tags": int(IFFResponse.FRIEND | IFFResponse.PLAYER),
            "secondary_teams": [2, 3],
            "custom_relations": {"entity_99": "HOSTILE"},
            "entity_id": "test_entity",
        }
        tc = TeamComponent.from_dict(data)
        assert tc.team_id == 5
        assert tc.has_iff_tag(IFFResponse.FRIEND)
        assert tc.has_iff_tag(IFFResponse.PLAYER)
        assert 2 in tc.secondary_teams
        assert 3 in tc.secondary_teams
        assert tc._entity_id == "test_entity"

    def test_round_trip(self, player_team_component):
        """Test serialization round trip."""
        player_team_component.add_secondary_team(5)
        player_team_component.set_custom_relation("enemy_99", TeamRelation.HOSTILE)
        data = player_team_component.to_dict()
        restored = TeamComponent.from_dict(data)
        assert restored.team_id == player_team_component.team_id
        assert restored.iff_tags == player_team_component.iff_tags
        assert 5 in restored.secondary_teams

    def test_repr(self, registry, setup_teams):
        """Test string representation."""
        tc = TeamComponent(team_id=1)
        rep = repr(tc)
        assert "TeamComponent" in rep


# =============================================================================
# EDGE CASES TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_team_id_zero(self, team_component):
        """Test team ID zero (no team)."""
        assert team_component.team_id == 0
        assert team_component.team is None

    def test_very_large_team_id(self):
        """Test very large team ID."""
        tc = TeamComponent(team_id=999999)
        assert tc.team_id == 999999

    def test_many_secondary_teams(self, team_component):
        """Test many secondary teams."""
        for i in range(100):
            team_component.add_secondary_team(i + 10)
        assert len(team_component.secondary_teams) == 100

    def test_all_iff_flags(self):
        """Test all IFF flags combined."""
        all_flags = (
            IFFResponse.FRIEND |
            IFFResponse.FOE |
            IFFResponse.UNKNOWN |
            IFFResponse.CIVILIAN |
            IFFResponse.OBJECTIVE |
            IFFResponse.HAZARD |
            IFFResponse.PLAYER |
            IFFResponse.AI
        )
        tc = TeamComponent(iff_tags=all_flags)
        assert tc.has_iff_tag(IFFResponse.FRIEND)
        assert tc.has_iff_tag(IFFResponse.FOE)
        assert tc.has_iff_tag(IFFResponse.PLAYER)
        assert tc.has_iff_tag(IFFResponse.AI)

    def test_circular_faction_reference(self, registry):
        """Test no issues with faction references."""
        f1 = Faction(id="f1", name="Faction 1")
        f2 = Faction(id="f2", name="Faction 2")
        registry.register_faction(f1)
        registry.register_faction(f2)
        t1 = registry.create_team("T1", faction=f1)
        t2 = registry.create_team("T2", faction=f2)
        tc1 = TeamComponent(team_id=t1.id)
        tc2 = TeamComponent(team_id=t2.id)
        # Should work without issues
        assert tc1.is_same_faction(tc2) is False

    def test_relation_to_self(self, registry, setup_teams):
        """Test relation to self."""
        tc = TeamComponent(team_id=1, entity_id="self")
        # Comparing to same component
        assert tc.get_relation_to(tc) == TeamRelation.SELF
