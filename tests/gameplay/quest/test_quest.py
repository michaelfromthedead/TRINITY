"""
Comprehensive tests for the Quest System.

Tests cover:
- Quest state machine (UNAVAILABLE->AVAILABLE->ACTIVE->COMPLETE->TURNED_IN)
- Quest prerequisites
- Quest chains (sequential quests)
- Quest branches (choices)
- Quest failure conditions
- Quest abandonment
- Quest sharing (multiplayer)
- Repeatable quests
- Daily/weekly quests
- Quest priority/sorting
"""

import pytest
from dataclasses import dataclass
from typing import Any

from engine.gameplay.quest.quest import (
    Quest,
    QuestDefinition,
    QuestRegistry,
    QuestState,
    QuestType,
    quest,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def clear_quest_registry():
    """Clear quest registry before and after each test."""
    QuestRegistry.clear()
    yield
    QuestRegistry.clear()


@pytest.fixture
def basic_quest_def():
    """Create a basic quest definition."""
    return QuestDefinition(
        id="test_quest_1",
        name="Test Quest",
        description="A test quest",
        quest_type=QuestType.SIDE,
        level_requirement=1,
    )


@pytest.fixture
def main_quest_def():
    """Create a main story quest definition."""
    return QuestDefinition(
        id="main_quest_1",
        name="The Beginning",
        description="Start your adventure",
        quest_type=QuestType.MAIN,
        level_requirement=1,
    )


@pytest.fixture
def daily_quest_def():
    """Create a daily quest definition."""
    return QuestDefinition(
        id="daily_quest_1",
        name="Daily Challenge",
        description="Complete daily tasks",
        quest_type=QuestType.DAILY,
        level_requirement=1,
        repeatable=True,
        cooldown=86400.0,  # 24 hours
    )


@pytest.fixture
def weekly_quest_def():
    """Create a weekly quest definition."""
    return QuestDefinition(
        id="weekly_quest_1",
        name="Weekly Challenge",
        description="Complete weekly tasks",
        quest_type=QuestType.WEEKLY,
        level_requirement=1,
        repeatable=True,
        cooldown=604800.0,  # 7 days
    )


# =============================================================================
# Quest State Machine Tests
# =============================================================================

class TestQuestStateMachine:
    """Tests for quest state transitions."""

    def test_quest_initial_state_unavailable(self, basic_quest_def):
        """Test that quest starts in UNAVAILABLE state."""
        quest = Quest(definition=basic_quest_def)
        assert quest.state == QuestState.UNAVAILABLE

    def test_quest_make_available(self, basic_quest_def):
        """Test transitioning from UNAVAILABLE to AVAILABLE."""
        quest = Quest(definition=basic_quest_def)
        assert quest.make_available() is True
        assert quest.state == QuestState.AVAILABLE

    def test_quest_make_available_only_from_unavailable(self, basic_quest_def):
        """Test that make_available only works from UNAVAILABLE state."""
        quest = Quest(definition=basic_quest_def, state=QuestState.ACTIVE)
        assert quest.make_available() is False
        assert quest.state == QuestState.ACTIVE

    def test_quest_accept(self, basic_quest_def):
        """Test accepting a quest."""
        quest = Quest(definition=basic_quest_def, state=QuestState.AVAILABLE)
        assert quest.accept(timestamp=100.0) is True
        assert quest.state == QuestState.ACTIVE
        assert quest.accepted_at == 100.0

    def test_quest_accept_only_from_available(self, basic_quest_def):
        """Test that accept only works from AVAILABLE state."""
        quest = Quest(definition=basic_quest_def)
        assert quest.accept(timestamp=100.0) is False
        assert quest.state == QuestState.UNAVAILABLE

    def test_quest_complete(self, basic_quest_def):
        """Test completing a quest."""
        quest = Quest(definition=basic_quest_def, state=QuestState.ACTIVE)
        assert quest.complete(timestamp=200.0) is True
        assert quest.state == QuestState.COMPLETE
        assert quest.completed_at == 200.0

    def test_quest_complete_only_from_active(self, basic_quest_def):
        """Test that complete only works from ACTIVE state."""
        quest = Quest(definition=basic_quest_def, state=QuestState.AVAILABLE)
        assert quest.complete(timestamp=200.0) is False
        assert quest.state == QuestState.AVAILABLE

    def test_quest_turn_in(self, basic_quest_def):
        """Test turning in a quest."""
        quest = Quest(definition=basic_quest_def, state=QuestState.COMPLETE)
        assert quest.turn_in(timestamp=300.0) is True
        assert quest.state == QuestState.TURNED_IN
        assert quest.turned_in_at == 300.0
        assert quest.times_completed == 1
        assert quest.last_completed_at == 300.0

    def test_quest_turn_in_only_from_complete(self, basic_quest_def):
        """Test that turn_in only works from COMPLETE state."""
        quest = Quest(definition=basic_quest_def, state=QuestState.ACTIVE)
        assert quest.turn_in(timestamp=300.0) is False
        assert quest.state == QuestState.ACTIVE

    def test_quest_fail(self, basic_quest_def):
        """Test failing a quest."""
        quest = Quest(definition=basic_quest_def, state=QuestState.ACTIVE)
        assert quest.fail(timestamp=150.0) is True
        assert quest.state == QuestState.FAILED
        assert quest.failed_at == 150.0

    def test_quest_fail_only_from_active(self, basic_quest_def):
        """Test that fail only works from ACTIVE state."""
        quest = Quest(definition=basic_quest_def, state=QuestState.COMPLETE)
        assert quest.fail(timestamp=150.0) is False
        assert quest.state == QuestState.COMPLETE

    def test_full_quest_lifecycle(self, basic_quest_def):
        """Test full quest lifecycle from unavailable to turned in."""
        quest = Quest(definition=basic_quest_def)

        # Unavailable -> Available
        assert quest.make_available() is True
        assert quest.state == QuestState.AVAILABLE

        # Available -> Active
        assert quest.accept(timestamp=100.0) is True
        assert quest.state == QuestState.ACTIVE

        # Active -> Complete
        assert quest.complete(timestamp=200.0) is True
        assert quest.state == QuestState.COMPLETE

        # Complete -> Turned In
        assert quest.turn_in(timestamp=300.0) is True
        assert quest.state == QuestState.TURNED_IN

    def test_quest_properties(self, basic_quest_def):
        """Test quest property helpers."""
        quest = Quest(definition=basic_quest_def)

        # Test is_available
        quest.state = QuestState.AVAILABLE
        assert quest.is_available is True
        assert quest.is_active is False

        # Test is_active
        quest.state = QuestState.ACTIVE
        assert quest.is_active is True
        assert quest.is_available is False

        # Test is_complete
        quest.state = QuestState.COMPLETE
        assert quest.is_complete is True

        # Test is_finished
        quest.state = QuestState.TURNED_IN
        assert quest.is_finished is True

        quest.state = QuestState.FAILED
        assert quest.is_finished is True


# =============================================================================
# Quest Definition Tests
# =============================================================================

class TestQuestDefinition:
    """Tests for quest definition validation and creation."""

    def test_quest_definition_creation(self):
        """Test creating a valid quest definition."""
        quest_def = QuestDefinition(
            id="test_quest",
            name="Test Quest",
            description="Test description",
            quest_type=QuestType.SIDE,
            level_requirement=5,
        )
        assert quest_def.id == "test_quest"
        assert quest_def.name == "Test Quest"
        assert quest_def.quest_type == QuestType.SIDE
        assert quest_def.level_requirement == 5

    def test_quest_definition_defaults(self):
        """Test quest definition default values."""
        quest_def = QuestDefinition(
            id="test",
            name="Test",
            description="",
        )
        assert quest_def.quest_type == QuestType.SIDE
        assert quest_def.level_requirement == 1
        assert quest_def.level_cap is None
        assert quest_def.time_limit is None
        assert quest_def.repeatable is False
        assert quest_def.cooldown == 0.0
        assert quest_def.auto_accept is False
        assert quest_def.auto_complete is False
        assert quest_def.hidden is False
        assert quest_def.shareable is True
        assert quest_def.abandon_penalty is False

    def test_quest_definition_empty_id_raises(self):
        """Test that empty id raises ValueError."""
        with pytest.raises(ValueError, match="Quest id cannot be empty"):
            QuestDefinition(id="", name="Test", description="")

    def test_quest_definition_empty_name_raises(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="Quest name cannot be empty"):
            QuestDefinition(id="test", name="", description="")

    def test_quest_definition_negative_level_raises(self):
        """Test that negative level_requirement raises ValueError."""
        with pytest.raises(ValueError, match="level_requirement must be >= 0"):
            QuestDefinition(
                id="test",
                name="Test",
                description="",
                level_requirement=-1,
            )

    def test_quest_definition_invalid_level_cap_raises(self):
        """Test that level_cap < level_requirement raises ValueError."""
        with pytest.raises(ValueError, match="level_cap must be >= level_requirement"):
            QuestDefinition(
                id="test",
                name="Test",
                description="",
                level_requirement=10,
                level_cap=5,
            )

    def test_quest_definition_invalid_time_limit_raises(self):
        """Test that non-positive time_limit raises ValueError."""
        with pytest.raises(ValueError, match="time_limit must be > 0"):
            QuestDefinition(
                id="test",
                name="Test",
                description="",
                time_limit=0,
            )

        with pytest.raises(ValueError, match="time_limit must be > 0"):
            QuestDefinition(
                id="test",
                name="Test",
                description="",
                time_limit=-10,
            )

    def test_quest_definition_negative_cooldown_raises(self):
        """Test that negative cooldown raises ValueError."""
        with pytest.raises(ValueError, match="cooldown must be >= 0"):
            QuestDefinition(
                id="test",
                name="Test",
                description="",
                cooldown=-1,
            )

    def test_quest_definition_with_all_options(self):
        """Test creating a quest definition with all options set."""
        quest_def = QuestDefinition(
            id="complete_quest",
            name="Complete Quest",
            description="A fully configured quest",
            quest_type=QuestType.MAIN,
            level_requirement=10,
            level_cap=50,
            time_limit=3600.0,
            repeatable=True,
            cooldown=86400.0,
            auto_accept=True,
            auto_complete=True,
            hidden=True,
            shareable=False,
            abandon_penalty=True,
            prerequisites=["prev_quest"],
            required_items={"key_item": 1},
            required_reputation={"faction_a": 100},
            category="main_story",
            zone="starting_zone",
            giver_id="npc_001",
            turn_in_id="npc_002",
            tags={"important", "story"},
        )

        assert quest_def.level_cap == 50
        assert quest_def.time_limit == 3600.0
        assert quest_def.repeatable is True
        assert quest_def.cooldown == 86400.0
        assert "prev_quest" in quest_def.prerequisites
        assert quest_def.required_items["key_item"] == 1
        assert quest_def.required_reputation["faction_a"] == 100


# =============================================================================
# Quest Type Tests
# =============================================================================

class TestQuestTypes:
    """Tests for different quest types."""

    @pytest.mark.parametrize("quest_type", list(QuestType))
    def test_all_quest_types_valid(self, quest_type):
        """Test that all quest types can be used in quest definitions."""
        quest_def = QuestDefinition(
            id=f"quest_{quest_type.name.lower()}",
            name=f"{quest_type.name} Quest",
            description="",
            quest_type=quest_type,
        )
        assert quest_def.quest_type == quest_type

    def test_main_quest_type(self):
        """Test main quest type."""
        assert QuestType.MAIN.value is not None

    def test_side_quest_type(self):
        """Test side quest type."""
        assert QuestType.SIDE.value is not None

    def test_daily_quest_type(self):
        """Test daily quest type."""
        assert QuestType.DAILY.value is not None

    def test_weekly_quest_type(self):
        """Test weekly quest type."""
        assert QuestType.WEEKLY.value is not None

    def test_world_quest_type(self):
        """Test world quest type."""
        assert QuestType.WORLD.value is not None

    def test_dungeon_quest_type(self):
        """Test dungeon quest type."""
        assert QuestType.DUNGEON.value is not None

    def test_raid_quest_type(self):
        """Test raid quest type."""
        assert QuestType.RAID.value is not None

    def test_pvp_quest_type(self):
        """Test PvP quest type."""
        assert QuestType.PVP.value is not None

    def test_hidden_quest_type(self):
        """Test hidden quest type."""
        assert QuestType.HIDDEN.value is not None

    def test_tutorial_quest_type(self):
        """Test tutorial quest type."""
        assert QuestType.TUTORIAL.value is not None

    def test_event_quest_type(self):
        """Test event quest type."""
        assert QuestType.EVENT.value is not None

    def test_bounty_quest_type(self):
        """Test bounty quest type."""
        assert QuestType.BOUNTY.value is not None

    def test_exploration_quest_type(self):
        """Test exploration quest type."""
        assert QuestType.EXPLORATION.value is not None


# =============================================================================
# Quest Prerequisites Tests
# =============================================================================

class TestQuestPrerequisites:
    """Tests for quest prerequisite functionality."""

    def test_quest_with_no_prerequisites(self):
        """Test quest with no prerequisites."""
        quest_def = QuestDefinition(
            id="no_prereq",
            name="No Prerequisites",
            description="",
        )
        assert quest_def.prerequisites == []

    def test_quest_with_single_prerequisite(self):
        """Test quest with a single prerequisite."""
        quest_def = QuestDefinition(
            id="with_prereq",
            name="With Prerequisite",
            description="",
            prerequisites=["previous_quest"],
        )
        assert "previous_quest" in quest_def.prerequisites
        assert len(quest_def.prerequisites) == 1

    def test_quest_with_multiple_prerequisites(self):
        """Test quest with multiple prerequisites."""
        quest_def = QuestDefinition(
            id="multi_prereq",
            name="Multiple Prerequisites",
            description="",
            prerequisites=["quest_a", "quest_b", "quest_c"],
        )
        assert len(quest_def.prerequisites) == 3
        assert "quest_a" in quest_def.prerequisites
        assert "quest_b" in quest_def.prerequisites
        assert "quest_c" in quest_def.prerequisites

    def test_quest_with_item_requirements(self):
        """Test quest requiring specific items."""
        quest_def = QuestDefinition(
            id="item_req",
            name="Item Required",
            description="",
            required_items={"special_key": 1, "gold_coin": 100},
        )
        assert quest_def.required_items["special_key"] == 1
        assert quest_def.required_items["gold_coin"] == 100

    def test_quest_with_reputation_requirements(self):
        """Test quest requiring specific reputation."""
        quest_def = QuestDefinition(
            id="rep_req",
            name="Reputation Required",
            description="",
            required_reputation={"guild_a": 500, "guild_b": -100},
        )
        assert quest_def.required_reputation["guild_a"] == 500
        assert quest_def.required_reputation["guild_b"] == -100

    def test_quest_with_level_requirement(self):
        """Test quest with level requirement."""
        quest_def = QuestDefinition(
            id="level_req",
            name="Level Required",
            description="",
            level_requirement=25,
        )
        assert quest_def.level_requirement == 25

    def test_quest_with_level_cap(self):
        """Test quest with level cap."""
        quest_def = QuestDefinition(
            id="level_cap",
            name="Level Capped",
            description="",
            level_requirement=10,
            level_cap=20,
        )
        assert quest_def.level_requirement == 10
        assert quest_def.level_cap == 20


# =============================================================================
# Quest Chain Tests
# =============================================================================

class TestQuestChains:
    """Tests for sequential quest chains."""

    def test_create_quest_chain(self):
        """Test creating a chain of sequential quests."""
        quest1 = QuestDefinition(
            id="chain_1",
            name="Chain Quest 1",
            description="First quest in chain",
        )
        quest2 = QuestDefinition(
            id="chain_2",
            name="Chain Quest 2",
            description="Second quest in chain",
            prerequisites=["chain_1"],
        )
        quest3 = QuestDefinition(
            id="chain_3",
            name="Chain Quest 3",
            description="Third quest in chain",
            prerequisites=["chain_2"],
        )

        assert quest1.prerequisites == []
        assert quest2.prerequisites == ["chain_1"]
        assert quest3.prerequisites == ["chain_2"]

    def test_quest_chain_with_branching(self):
        """Test quest chain with branching paths."""
        main_quest = QuestDefinition(
            id="branch_start",
            name="Branch Start",
            description="Start of branching quests",
        )
        branch_a = QuestDefinition(
            id="branch_a",
            name="Branch A",
            description="First branch option",
            prerequisites=["branch_start"],
        )
        branch_b = QuestDefinition(
            id="branch_b",
            name="Branch B",
            description="Second branch option",
            prerequisites=["branch_start"],
        )

        # Both branches have the same prerequisite
        assert branch_a.prerequisites == branch_b.prerequisites

    def test_quest_chain_convergence(self):
        """Test quest chains that converge."""
        # Two separate quest lines
        line_a = QuestDefinition(id="line_a_end", name="Line A End", description="")
        line_b = QuestDefinition(id="line_b_end", name="Line B End", description="")

        # Quest requiring both lines complete
        convergence = QuestDefinition(
            id="convergence",
            name="Convergence Quest",
            description="Requires both lines complete",
            prerequisites=["line_a_end", "line_b_end"],
        )

        assert len(convergence.prerequisites) == 2


# =============================================================================
# Quest Failure Tests
# =============================================================================

class TestQuestFailure:
    """Tests for quest failure conditions."""

    def test_quest_fail_from_active(self, basic_quest_def):
        """Test failing a quest from active state."""
        quest = Quest(definition=basic_quest_def, state=QuestState.ACTIVE)
        result = quest.fail(timestamp=100.0)

        assert result is True
        assert quest.state == QuestState.FAILED
        assert quest.failed_at == 100.0

    def test_quest_cannot_fail_from_unavailable(self, basic_quest_def):
        """Test that quest cannot fail from unavailable state."""
        quest = Quest(definition=basic_quest_def, state=QuestState.UNAVAILABLE)
        result = quest.fail(timestamp=100.0)

        assert result is False
        assert quest.state == QuestState.UNAVAILABLE

    def test_quest_cannot_fail_from_available(self, basic_quest_def):
        """Test that quest cannot fail from available state."""
        quest = Quest(definition=basic_quest_def, state=QuestState.AVAILABLE)
        result = quest.fail(timestamp=100.0)

        assert result is False
        assert quest.state == QuestState.AVAILABLE

    def test_quest_cannot_fail_from_complete(self, basic_quest_def):
        """Test that quest cannot fail from complete state."""
        quest = Quest(definition=basic_quest_def, state=QuestState.COMPLETE)
        result = quest.fail(timestamp=100.0)

        assert result is False
        assert quest.state == QuestState.COMPLETE

    def test_quest_cannot_fail_from_turned_in(self, basic_quest_def):
        """Test that quest cannot fail from turned in state."""
        quest = Quest(definition=basic_quest_def, state=QuestState.TURNED_IN)
        result = quest.fail(timestamp=100.0)

        assert result is False
        assert quest.state == QuestState.TURNED_IN

    def test_quest_is_finished_after_failure(self, basic_quest_def):
        """Test that quest is_finished returns True after failure."""
        quest = Quest(definition=basic_quest_def, state=QuestState.FAILED)
        assert quest.is_finished is True

    def test_timed_quest_definition(self):
        """Test creating a quest with time limit."""
        quest_def = QuestDefinition(
            id="timed_quest",
            name="Timed Quest",
            description="Must complete within time limit",
            time_limit=300.0,  # 5 minutes
        )
        assert quest_def.time_limit == 300.0


# =============================================================================
# Quest Abandonment Tests
# =============================================================================

class TestQuestAbandonment:
    """Tests for quest abandonment functionality."""

    def test_abandon_active_quest(self, basic_quest_def):
        """Test abandoning an active quest."""
        quest = Quest(
            definition=basic_quest_def,
            state=QuestState.ACTIVE,
            accepted_at=100.0,
            objective_progress={"obj1": 50},
        )

        result = quest.abandon()

        assert result is True
        assert quest.state == QuestState.AVAILABLE
        assert quest.accepted_at is None
        assert quest.objective_progress == {}

    def test_abandon_complete_quest(self, basic_quest_def):
        """Test abandoning a completed (but not turned in) quest."""
        quest = Quest(
            definition=basic_quest_def,
            state=QuestState.COMPLETE,
            completed_at=200.0,
        )

        result = quest.abandon()

        assert result is True
        assert quest.state == QuestState.AVAILABLE
        assert quest.completed_at is None

    def test_cannot_abandon_unavailable_quest(self, basic_quest_def):
        """Test that unavailable quest cannot be abandoned."""
        quest = Quest(definition=basic_quest_def, state=QuestState.UNAVAILABLE)
        result = quest.abandon()

        assert result is False
        assert quest.state == QuestState.UNAVAILABLE

    def test_cannot_abandon_available_quest(self, basic_quest_def):
        """Test that available quest cannot be abandoned."""
        quest = Quest(definition=basic_quest_def, state=QuestState.AVAILABLE)
        result = quest.abandon()

        assert result is False
        assert quest.state == QuestState.AVAILABLE

    def test_cannot_abandon_turned_in_quest(self, basic_quest_def):
        """Test that turned in quest cannot be abandoned."""
        quest = Quest(definition=basic_quest_def, state=QuestState.TURNED_IN)
        result = quest.abandon()

        assert result is False
        assert quest.state == QuestState.TURNED_IN

    def test_cannot_abandon_failed_quest(self, basic_quest_def):
        """Test that failed quest cannot be abandoned."""
        quest = Quest(definition=basic_quest_def, state=QuestState.FAILED)
        result = quest.abandon()

        assert result is False
        assert quest.state == QuestState.FAILED

    def test_abandon_clears_progress(self, basic_quest_def):
        """Test that abandoning clears objective progress."""
        quest = Quest(
            definition=basic_quest_def,
            state=QuestState.ACTIVE,
            objective_progress={"kill_count": 5, "collect_count": 3},
        )

        quest.abandon()

        assert quest.objective_progress == {}

    def test_abandon_penalty_flag(self):
        """Test quest definition abandon_penalty flag."""
        quest_def = QuestDefinition(
            id="penalty_quest",
            name="Quest with Penalty",
            description="",
            abandon_penalty=True,
        )
        assert quest_def.abandon_penalty is True


# =============================================================================
# Quest Sharing Tests (Multiplayer)
# =============================================================================

class TestQuestSharing:
    """Tests for quest sharing functionality."""

    def test_shareable_quest_default(self, basic_quest_def):
        """Test that quests are shareable by default."""
        assert basic_quest_def.shareable is True

    def test_non_shareable_quest(self):
        """Test creating a non-shareable quest."""
        quest_def = QuestDefinition(
            id="personal_quest",
            name="Personal Quest",
            description="Cannot be shared",
            shareable=False,
        )
        assert quest_def.shareable is False

    def test_shareable_quest_explicit(self):
        """Test explicitly shareable quest."""
        quest_def = QuestDefinition(
            id="shareable_quest",
            name="Shareable Quest",
            description="Can be shared",
            shareable=True,
        )
        assert quest_def.shareable is True

    def test_quest_player_id_assignment(self, basic_quest_def):
        """Test quest player_id assignment."""
        quest = Quest(definition=basic_quest_def, player_id="player_123")
        assert quest.player_id == "player_123"

    def test_quest_multiple_players(self, basic_quest_def):
        """Test same quest definition with different players."""
        quest_p1 = Quest(definition=basic_quest_def, player_id="player_1")
        quest_p2 = Quest(definition=basic_quest_def, player_id="player_2")

        # Same definition, different player instances
        assert quest_p1.definition is quest_p2.definition
        assert quest_p1.player_id != quest_p2.player_id

        # Independent state
        quest_p1.state = QuestState.ACTIVE
        assert quest_p2.state == QuestState.UNAVAILABLE


# =============================================================================
# Repeatable Quest Tests
# =============================================================================

class TestRepeatableQuests:
    """Tests for repeatable quest functionality."""

    def test_repeatable_quest_can_repeat_after_turn_in(self):
        """Test that repeatable quest can be repeated after turn in."""
        quest_def = QuestDefinition(
            id="repeatable",
            name="Repeatable Quest",
            description="",
            repeatable=True,
        )
        quest = Quest(definition=quest_def, state=QuestState.TURNED_IN)

        assert quest.can_repeat is True

    def test_repeatable_quest_can_repeat_after_failure(self):
        """Test that repeatable quest can be repeated after failure."""
        quest_def = QuestDefinition(
            id="repeatable",
            name="Repeatable Quest",
            description="",
            repeatable=True,
        )
        quest = Quest(definition=quest_def, state=QuestState.FAILED)

        assert quest.can_repeat is True

    def test_non_repeatable_quest_cannot_repeat(self, basic_quest_def):
        """Test that non-repeatable quest cannot be repeated."""
        quest = Quest(definition=basic_quest_def, state=QuestState.TURNED_IN)
        assert quest.can_repeat is False

    def test_repeatable_quest_cannot_repeat_while_active(self):
        """Test that repeatable quest cannot repeat while active."""
        quest_def = QuestDefinition(
            id="repeatable",
            name="Repeatable Quest",
            description="",
            repeatable=True,
        )
        quest = Quest(definition=quest_def, state=QuestState.ACTIVE)

        assert quest.can_repeat is False

    def test_quest_reset(self):
        """Test resetting a repeatable quest."""
        quest_def = QuestDefinition(
            id="repeatable",
            name="Repeatable Quest",
            description="",
            repeatable=True,
        )
        quest = Quest(
            definition=quest_def,
            state=QuestState.TURNED_IN,
            times_completed=1,
            objective_progress={"obj": 100},
        )

        result = quest.reset()

        assert result is True
        assert quest.state == QuestState.AVAILABLE
        assert quest.accepted_at is None
        assert quest.completed_at is None
        assert quest.turned_in_at is None
        assert quest.failed_at is None
        assert quest.objective_progress == {}

    def test_reset_non_repeatable_fails(self, basic_quest_def):
        """Test that reset fails on non-repeatable quest."""
        quest = Quest(definition=basic_quest_def, state=QuestState.TURNED_IN)
        result = quest.reset()

        assert result is False
        assert quest.state == QuestState.TURNED_IN

    def test_times_completed_tracking(self):
        """Test that times_completed is tracked correctly."""
        quest_def = QuestDefinition(
            id="repeatable",
            name="Repeatable Quest",
            description="",
            repeatable=True,
        )
        quest = Quest(definition=quest_def, state=QuestState.AVAILABLE)

        # First completion
        quest.accept(timestamp=100.0)
        quest.complete(timestamp=200.0)
        quest.turn_in(timestamp=300.0)
        assert quest.times_completed == 1

        # Reset and complete again
        quest.reset()
        quest.accept(timestamp=400.0)
        quest.complete(timestamp=500.0)
        quest.turn_in(timestamp=600.0)
        assert quest.times_completed == 2

    def test_last_completed_at_tracking(self):
        """Test that last_completed_at is tracked correctly."""
        quest_def = QuestDefinition(
            id="repeatable",
            name="Repeatable Quest",
            description="",
            repeatable=True,
        )
        quest = Quest(definition=quest_def, state=QuestState.AVAILABLE)

        quest.accept(timestamp=100.0)
        quest.complete(timestamp=200.0)
        quest.turn_in(timestamp=300.0)

        assert quest.last_completed_at == 300.0


# =============================================================================
# Daily/Weekly Quest Tests
# =============================================================================

class TestDailyWeeklyQuests:
    """Tests for daily and weekly quest functionality."""

    def test_daily_quest_definition(self, daily_quest_def):
        """Test daily quest definition."""
        assert daily_quest_def.quest_type == QuestType.DAILY
        assert daily_quest_def.repeatable is True
        assert daily_quest_def.cooldown == 86400.0

    def test_weekly_quest_definition(self, weekly_quest_def):
        """Test weekly quest definition."""
        assert weekly_quest_def.quest_type == QuestType.WEEKLY
        assert weekly_quest_def.repeatable is True
        assert weekly_quest_def.cooldown == 604800.0

    def test_daily_quest_with_zero_cooldown(self):
        """Test daily quest with zero cooldown."""
        quest_def = QuestDefinition(
            id="daily_no_cooldown",
            name="Daily No Cooldown",
            description="",
            quest_type=QuestType.DAILY,
            repeatable=True,
            cooldown=0.0,
        )
        assert quest_def.cooldown == 0.0

    def test_weekly_quest_cooldown(self):
        """Test weekly quest cooldown calculation."""
        quest_def = QuestDefinition(
            id="weekly",
            name="Weekly",
            description="",
            quest_type=QuestType.WEEKLY,
            repeatable=True,
            cooldown=604800.0,  # 7 days in seconds
        )
        assert quest_def.cooldown == 7 * 24 * 60 * 60

    def test_event_quest_with_time_limit(self):
        """Test event quest with time limit."""
        quest_def = QuestDefinition(
            id="event_quest",
            name="Event Quest",
            description="Limited time event",
            quest_type=QuestType.EVENT,
            time_limit=259200.0,  # 3 days
        )
        assert quest_def.quest_type == QuestType.EVENT
        assert quest_def.time_limit == 259200.0


# =============================================================================
# Quest Registry Tests
# =============================================================================

class TestQuestRegistry:
    """Tests for quest registry functionality."""

    def test_registry_singleton(self):
        """Test that registry is a singleton."""
        reg1 = QuestRegistry.instance()
        reg2 = QuestRegistry.instance()
        assert reg1 is reg2

    def test_register_quest(self, basic_quest_def):
        """Test registering a quest definition."""
        registry = QuestRegistry.instance()
        registry.register(basic_quest_def)

        assert basic_quest_def.id in registry
        assert registry.get(basic_quest_def.id) is basic_quest_def

    def test_register_duplicate_raises(self, basic_quest_def):
        """Test that registering duplicate ID raises error."""
        registry = QuestRegistry.instance()
        registry.register(basic_quest_def)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(basic_quest_def)

    def test_unregister_quest(self, basic_quest_def):
        """Test unregistering a quest definition."""
        registry = QuestRegistry.instance()
        registry.register(basic_quest_def)

        result = registry.unregister(basic_quest_def.id)

        assert result is True
        assert basic_quest_def.id not in registry

    def test_unregister_nonexistent_quest(self):
        """Test unregistering a quest that doesn't exist."""
        registry = QuestRegistry.instance()
        result = registry.unregister("nonexistent")
        assert result is False

    def test_get_quest(self, basic_quest_def):
        """Test getting a quest by ID."""
        registry = QuestRegistry.instance()
        registry.register(basic_quest_def)

        result = registry.get(basic_quest_def.id)
        assert result is basic_quest_def

    def test_get_nonexistent_quest(self):
        """Test getting a quest that doesn't exist."""
        registry = QuestRegistry.instance()
        result = registry.get("nonexistent")
        assert result is None

    def test_get_by_type(self):
        """Test getting quests by type."""
        registry = QuestRegistry.instance()

        main_quest = QuestDefinition(
            id="main1",
            name="Main 1",
            description="",
            quest_type=QuestType.MAIN,
        )
        side_quest = QuestDefinition(
            id="side1",
            name="Side 1",
            description="",
            quest_type=QuestType.SIDE,
        )

        registry.register(main_quest)
        registry.register(side_quest)

        main_quests = registry.get_by_type(QuestType.MAIN)
        assert len(main_quests) == 1
        assert main_quests[0] is main_quest

    def test_get_by_zone(self):
        """Test getting quests by zone."""
        registry = QuestRegistry.instance()

        quest1 = QuestDefinition(
            id="zone1_quest",
            name="Zone 1 Quest",
            description="",
            zone="forest",
        )
        quest2 = QuestDefinition(
            id="zone2_quest",
            name="Zone 2 Quest",
            description="",
            zone="desert",
        )

        registry.register(quest1)
        registry.register(quest2)

        forest_quests = registry.get_by_zone("forest")
        assert len(forest_quests) == 1
        assert forest_quests[0] is quest1

    def test_get_by_giver(self):
        """Test getting quests by giver NPC."""
        registry = QuestRegistry.instance()

        quest1 = QuestDefinition(
            id="npc1_quest",
            name="NPC 1 Quest",
            description="",
            giver_id="npc_001",
        )
        quest2 = QuestDefinition(
            id="npc2_quest",
            name="NPC 2 Quest",
            description="",
            giver_id="npc_002",
        )

        registry.register(quest1)
        registry.register(quest2)

        npc1_quests = registry.get_by_giver("npc_001")
        assert len(npc1_quests) == 1
        assert npc1_quests[0] is quest1

    def test_get_by_tag(self):
        """Test getting quests by tag."""
        registry = QuestRegistry.instance()

        quest1 = QuestDefinition(
            id="tagged1",
            name="Tagged 1",
            description="",
            tags={"combat", "story"},
        )
        quest2 = QuestDefinition(
            id="tagged2",
            name="Tagged 2",
            description="",
            tags={"puzzle", "story"},
        )

        registry.register(quest1)
        registry.register(quest2)

        story_quests = registry.get_by_tag("story")
        assert len(story_quests) == 2

        combat_quests = registry.get_by_tag("combat")
        assert len(combat_quests) == 1

    def test_all_quests(self, basic_quest_def):
        """Test getting all registered quests."""
        registry = QuestRegistry.instance()
        registry.register(basic_quest_def)

        all_quests = registry.all_quests()
        assert len(all_quests) == 1
        assert basic_quest_def in all_quests

    def test_registry_len(self, basic_quest_def):
        """Test registry length."""
        registry = QuestRegistry.instance()
        assert len(registry) == 0

        registry.register(basic_quest_def)
        assert len(registry) == 1

    def test_registry_contains(self, basic_quest_def):
        """Test registry contains operator."""
        registry = QuestRegistry.instance()
        assert basic_quest_def.id not in registry

        registry.register(basic_quest_def)
        assert basic_quest_def.id in registry

    def test_registry_clear(self, basic_quest_def):
        """Test clearing the registry."""
        registry = QuestRegistry.instance()
        registry.register(basic_quest_def)
        assert len(registry) == 1

        QuestRegistry.clear()
        assert len(registry) == 0


# =============================================================================
# Quest Decorator Tests
# =============================================================================

class TestQuestDecorator:
    """Tests for the @quest decorator."""

    def test_quest_decorator_basic(self):
        """Test basic quest decorator usage."""
        @quest(
            id="decorated_quest",
            name="Decorated Quest",
            description="A decorated quest",
        )
        class MyQuest:
            pass

        assert hasattr(MyQuest, "_quest")
        assert MyQuest._quest is True
        assert MyQuest._quest_id == "decorated_quest"
        assert hasattr(MyQuest, "_quest_definition")

    def test_quest_decorator_uses_class_name(self):
        """Test that decorator uses class name if name not provided."""
        @quest(id="auto_named", description="")
        class AutoNamedQuest:
            pass

        assert AutoNamedQuest._quest_definition.name == "AutoNamedQuest"

    def test_quest_decorator_with_all_options(self):
        """Test quest decorator with all options."""
        @quest(
            id="full_quest",
            name="Full Quest",
            description="Fully configured",
            quest_type=QuestType.MAIN,
            level_requirement=10,
            prerequisites=["prev"],
            rewards=[],
        )
        class FullQuest:
            pass

        quest_def = FullQuest._quest_definition
        assert quest_def.quest_type == QuestType.MAIN
        assert quest_def.level_requirement == 10
        assert "prev" in quest_def.prerequisites

    def test_quest_decorator_empty_id_raises(self):
        """Test that decorator with empty id raises error."""
        with pytest.raises(ValueError, match="Quest id must be non-empty"):
            @quest(id="", name="Test", description="")
            class EmptyIdQuest:
                pass

    def test_quest_decorator_registers_quest(self):
        """Test that decorator registers quest in registry."""
        @quest(
            id="registered_quest",
            name="Registered",
            description="",
        )
        class RegisteredQuest:
            pass

        registry = QuestRegistry.instance()
        assert "registered_quest" in registry

    def test_quest_decorator_adds_tags(self):
        """Test that decorator adds proper tags to class."""
        @quest(
            id="tagged_quest",
            name="Tagged",
            description="",
            quest_type=QuestType.MAIN,
        )
        class TaggedQuest:
            pass

        assert TaggedQuest._tags["quest"] is True
        assert TaggedQuest._tags["quest_id"] == "tagged_quest"
        assert TaggedQuest._tags["quest_type"] == "MAIN"

    def test_quest_decorator_adds_registries(self):
        """Test that decorator adds gameplay registry."""
        @quest(id="registry_quest", name="Registry", description="")
        class RegistryQuest:
            pass

        assert "gameplay" in RegistryQuest._registries

    def test_quest_decorator_adds_applied_decorators(self):
        """Test that decorator tracks applied decorators."""
        @quest(id="applied_quest", name="Applied", description="")
        class AppliedQuest:
            pass

        assert "quest" in AppliedQuest._applied_decorators


# =============================================================================
# Quest Priority/Sorting Tests
# =============================================================================

class TestQuestPrioritySorting:
    """Tests for quest priority and sorting functionality."""

    def test_quest_sort_order_attribute(self):
        """Test quest definition with custom sort order."""
        quest_def = QuestDefinition(
            id="sorted_quest",
            name="Sorted Quest",
            description="",
            category="main",
        )
        assert quest_def.category == "main"

    def test_sort_quests_by_type(self):
        """Test sorting quests by type."""
        quests = [
            QuestDefinition(id="q1", name="Q1", description="", quest_type=QuestType.SIDE),
            QuestDefinition(id="q2", name="Q2", description="", quest_type=QuestType.MAIN),
            QuestDefinition(id="q3", name="Q3", description="", quest_type=QuestType.DAILY),
        ]

        sorted_quests = sorted(quests, key=lambda q: q.quest_type.value)
        # Order depends on enum values
        assert len(sorted_quests) == 3

    def test_sort_quests_by_level(self):
        """Test sorting quests by level requirement."""
        quests = [
            QuestDefinition(id="q1", name="Q1", description="", level_requirement=20),
            QuestDefinition(id="q2", name="Q2", description="", level_requirement=5),
            QuestDefinition(id="q3", name="Q3", description="", level_requirement=10),
        ]

        sorted_quests = sorted(quests, key=lambda q: q.level_requirement)
        assert sorted_quests[0].level_requirement == 5
        assert sorted_quests[1].level_requirement == 10
        assert sorted_quests[2].level_requirement == 20

    def test_sort_quests_by_name(self):
        """Test sorting quests alphabetically by name."""
        quests = [
            QuestDefinition(id="q1", name="Zebra Quest", description=""),
            QuestDefinition(id="q2", name="Apple Quest", description=""),
            QuestDefinition(id="q3", name="Mango Quest", description=""),
        ]

        sorted_quests = sorted(quests, key=lambda q: q.name.lower())
        assert sorted_quests[0].name == "Apple Quest"
        assert sorted_quests[1].name == "Mango Quest"
        assert sorted_quests[2].name == "Zebra Quest"

    def test_sort_quests_by_category(self):
        """Test sorting quests by category."""
        quests = [
            QuestDefinition(id="q1", name="Q1", description="", category="story"),
            QuestDefinition(id="q2", name="Q2", description="", category="combat"),
            QuestDefinition(id="q3", name="Q3", description="", category="exploration"),
        ]

        sorted_quests = sorted(quests, key=lambda q: q.category)
        assert sorted_quests[0].category == "combat"
        assert sorted_quests[1].category == "exploration"
        assert sorted_quests[2].category == "story"


# =============================================================================
# Quest ID and Name Tests
# =============================================================================

class TestQuestIdAndName:
    """Tests for quest ID and name properties."""

    def test_quest_id_property(self, basic_quest_def):
        """Test quest id property."""
        quest = Quest(definition=basic_quest_def)
        assert quest.id == basic_quest_def.id

    def test_quest_name_property(self, basic_quest_def):
        """Test quest name property."""
        quest = Quest(definition=basic_quest_def)
        assert quest.name == basic_quest_def.name

    def test_quest_definition_immutability(self, basic_quest_def):
        """Test that quest definition is accessible."""
        quest = Quest(definition=basic_quest_def)
        assert quest.definition is basic_quest_def


# =============================================================================
# Auto Accept/Complete Tests
# =============================================================================

class TestAutoAcceptComplete:
    """Tests for auto_accept and auto_complete functionality."""

    def test_auto_accept_quest_definition(self):
        """Test quest definition with auto_accept."""
        quest_def = QuestDefinition(
            id="auto_accept",
            name="Auto Accept Quest",
            description="",
            auto_accept=True,
        )
        assert quest_def.auto_accept is True

    def test_auto_complete_quest_definition(self):
        """Test quest definition with auto_complete."""
        quest_def = QuestDefinition(
            id="auto_complete",
            name="Auto Complete Quest",
            description="",
            auto_complete=True,
        )
        assert quest_def.auto_complete is True

    def test_hidden_quest_definition(self):
        """Test quest definition with hidden flag."""
        quest_def = QuestDefinition(
            id="hidden_quest",
            name="Hidden Quest",
            description="",
            hidden=True,
        )
        assert quest_def.hidden is True


# =============================================================================
# Quest Metadata Tests
# =============================================================================

class TestQuestMetadata:
    """Tests for quest metadata fields."""

    def test_quest_zone(self):
        """Test quest zone field."""
        quest_def = QuestDefinition(
            id="zone_quest",
            name="Zone Quest",
            description="",
            zone="starting_forest",
        )
        assert quest_def.zone == "starting_forest"

    def test_quest_giver_id(self):
        """Test quest giver_id field."""
        quest_def = QuestDefinition(
            id="giver_quest",
            name="Giver Quest",
            description="",
            giver_id="npc_merchant_01",
        )
        assert quest_def.giver_id == "npc_merchant_01"

    def test_quest_turn_in_id(self):
        """Test quest turn_in_id field."""
        quest_def = QuestDefinition(
            id="turnin_quest",
            name="Turn In Quest",
            description="",
            giver_id="npc_01",
            turn_in_id="npc_02",
        )
        assert quest_def.turn_in_id == "npc_02"

    def test_quest_tags(self):
        """Test quest tags field."""
        quest_def = QuestDefinition(
            id="tagged_quest",
            name="Tagged Quest",
            description="",
            tags={"epic", "story", "boss_fight"},
        )
        assert "epic" in quest_def.tags
        assert "story" in quest_def.tags
        assert "boss_fight" in quest_def.tags
        assert len(quest_def.tags) == 3


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestQuestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_quest_with_zero_level_requirement(self):
        """Test quest with level requirement of 0."""
        quest_def = QuestDefinition(
            id="zero_level",
            name="Zero Level Quest",
            description="",
            level_requirement=0,
        )
        assert quest_def.level_requirement == 0

    def test_quest_with_very_high_level_requirement(self):
        """Test quest with very high level requirement."""
        quest_def = QuestDefinition(
            id="high_level",
            name="High Level Quest",
            description="",
            level_requirement=999,
        )
        assert quest_def.level_requirement == 999

    def test_quest_with_minimal_time_limit(self):
        """Test quest with minimal positive time limit."""
        quest_def = QuestDefinition(
            id="quick_quest",
            name="Quick Quest",
            description="",
            time_limit=0.001,
        )
        assert quest_def.time_limit == 0.001

    def test_quest_with_large_time_limit(self):
        """Test quest with very large time limit."""
        quest_def = QuestDefinition(
            id="long_quest",
            name="Long Quest",
            description="",
            time_limit=31536000.0,  # One year in seconds
        )
        assert quest_def.time_limit == 31536000.0

    def test_quest_objective_progress_dict(self, basic_quest_def):
        """Test quest objective progress dictionary."""
        quest = Quest(
            definition=basic_quest_def,
            objective_progress={"kill_count": 5, "item_count": 3},
        )
        assert quest.objective_progress["kill_count"] == 5
        assert quest.objective_progress["item_count"] == 3

    def test_quest_timestamps_independence(self, basic_quest_def):
        """Test that quest timestamps are independent."""
        quest = Quest(
            definition=basic_quest_def,
            state=QuestState.AVAILABLE,
        )

        quest.accept(timestamp=100.0)
        assert quest.accepted_at == 100.0
        assert quest.completed_at is None
        assert quest.turned_in_at is None
        assert quest.failed_at is None

        quest.complete(timestamp=200.0)
        assert quest.accepted_at == 100.0
        assert quest.completed_at == 200.0

    def test_multiple_quests_same_definition(self, basic_quest_def):
        """Test multiple quest instances with same definition."""
        quest1 = Quest(definition=basic_quest_def, player_id="p1")
        quest2 = Quest(definition=basic_quest_def, player_id="p2")

        quest1.state = QuestState.ACTIVE
        quest2.state = QuestState.COMPLETE

        # States are independent
        assert quest1.state == QuestState.ACTIVE
        assert quest2.state == QuestState.COMPLETE

        # But definition is shared
        assert quest1.definition is quest2.definition
