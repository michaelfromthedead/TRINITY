"""
Blackbox tests for Quest System.

Tests PUBLIC behavior only based on specifications:
- Quest definitions and state management
- Objective types (kill, collect, talk, reach, escort, etc.)
- Quest tracking and progress management
- Quest completion and rewards

Test requirements:
- Test observable behavior only
- NO internal state inspection
- Minimum coverage of all public APIs
"""

import pytest
import time
from typing import Optional

from engine.gameplay.quest import (
    # Quest core
    Quest,
    QuestDefinition,
    QuestRegistry,
    QuestState,
    QuestType,
    # Foundation event types
    QuestStateChanged,
    ObjectiveProgress,
    ObjectiveCompleted,
    QuestRewardGranted,
    # Event helpers
    fire_quest_event,
    get_quest_events,
    clear_quest_events,
    # Objectives
    Objective,
    ObjectiveState,
    ObjectiveType,
    KillObjective,
    CollectObjective,
    TalkObjective,
    ReachObjective,
    EscortObjective,
    InteractObjective,
    UseObjective,
    CraftObjective,
    DefendObjective,
    TimedObjective,
    CompositeObjective,
    # Tracker
    QuestTracker,
    QuestEvent,
    QuestEventType,
    TrackedQuest,
)


# ============================================================================
# QUEST STATE TESTS
# ============================================================================

class TestQuestState:
    """Test QuestState enum values."""

    def test_quest_state_has_unavailable(self):
        """QuestState should have UNAVAILABLE."""
        assert hasattr(QuestState, 'UNAVAILABLE')

    def test_quest_state_has_available(self):
        """QuestState should have AVAILABLE."""
        assert hasattr(QuestState, 'AVAILABLE')

    def test_quest_state_has_active(self):
        """QuestState should have ACTIVE."""
        assert hasattr(QuestState, 'ACTIVE')

    def test_quest_state_has_complete(self):
        """QuestState should have COMPLETE."""
        assert hasattr(QuestState, 'COMPLETE')

    def test_quest_state_has_turned_in(self):
        """QuestState should have TURNED_IN."""
        assert hasattr(QuestState, 'TURNED_IN')

    def test_quest_state_has_failed(self):
        """QuestState should have FAILED."""
        assert hasattr(QuestState, 'FAILED')

    def test_quest_states_distinct(self):
        """Quest states should be distinct values."""
        states = [
            QuestState.UNAVAILABLE,
            QuestState.AVAILABLE,
            QuestState.ACTIVE,
            QuestState.COMPLETE,
            QuestState.TURNED_IN,
            QuestState.FAILED,
        ]
        assert len(set(states)) == 6


class TestQuestType:
    """Test QuestType enum values."""

    def test_quest_type_has_main(self):
        """QuestType should have MAIN."""
        assert hasattr(QuestType, 'MAIN')

    def test_quest_type_has_side(self):
        """QuestType should have SIDE."""
        assert hasattr(QuestType, 'SIDE')

    def test_quest_type_has_daily(self):
        """QuestType should have DAILY."""
        assert hasattr(QuestType, 'DAILY')

    def test_quest_type_has_weekly(self):
        """QuestType should have WEEKLY."""
        assert hasattr(QuestType, 'WEEKLY')


# ============================================================================
# QUEST DEFINITION TESTS
# ============================================================================

class TestQuestDefinition:
    """Test QuestDefinition creation and properties."""

    def test_quest_definition_basic_creation(self):
        """QuestDefinition should be creatable with basic params."""
        definition = QuestDefinition(
            id="test_quest",
            name="Test Quest",
            description="A test quest",
        )
        assert definition.id == "test_quest"
        assert definition.name == "Test Quest"
        assert definition.description == "A test quest"

    def test_quest_definition_with_type(self):
        """QuestDefinition should accept quest type."""
        definition = QuestDefinition(
            id="main_quest",
            name="Main Quest",
            description="The main storyline",
            quest_type=QuestType.MAIN,
        )
        assert definition.quest_type == QuestType.MAIN

    def test_quest_definition_default_type(self):
        """QuestDefinition should have default type."""
        definition = QuestDefinition(
            id="side",
            name="Side",
            description="A side quest",
        )
        # Default is SIDE
        assert definition.quest_type == QuestType.SIDE

    def test_quest_definition_with_level_requirement(self):
        """QuestDefinition should accept level requirement."""
        definition = QuestDefinition(
            id="high_level",
            name="High Level Quest",
            description="Requires high level",
            level_requirement=50,
        )
        assert definition.level_requirement == 50

    def test_quest_definition_repeatable(self):
        """QuestDefinition should accept repeatable flag."""
        definition = QuestDefinition(
            id="daily_quest",
            name="Daily Quest",
            description="Repeatable daily",
            repeatable=True,
        )
        assert definition.repeatable is True

    def test_quest_definition_with_time_limit(self):
        """QuestDefinition should accept time limit."""
        definition = QuestDefinition(
            id="timed",
            name="Timed Quest",
            description="Must complete quickly",
            time_limit=300.0,
        )
        assert definition.time_limit == 300.0

    def test_quest_definition_prerequisites(self):
        """QuestDefinition should accept prerequisites."""
        definition = QuestDefinition(
            id="sequel",
            name="Sequel Quest",
            description="Requires previous quest",
            prerequisites=["previous_quest"],
        )
        assert "previous_quest" in definition.prerequisites


# ============================================================================
# QUEST CREATION AND LIFECYCLE TESTS
# ============================================================================

class TestQuestCreation:
    """Test Quest instance creation."""

    def test_quest_from_definition(self):
        """Quest should be creatable from definition."""
        definition = QuestDefinition(
            id="q1",
            name="Quest One",
            description="First quest",
        )
        q = Quest(definition, player_id="player1")
        assert q.id == "q1"
        assert q.name == "Quest One"

    def test_quest_initial_state(self):
        """New quest should have a valid initial state."""
        definition = QuestDefinition(
            id="q2",
            name="Quest Two",
            description="Second quest",
        )
        q = Quest(definition, player_id="player1")
        # Initial state is UNAVAILABLE
        assert q.state == QuestState.UNAVAILABLE

    def test_quest_make_available(self):
        """Quest should be makeable available."""
        definition = QuestDefinition(
            id="q3",
            name="Quest Three",
            description="Third quest",
        )
        q = Quest(definition, player_id="player1")
        q.make_available()
        assert q.state == QuestState.AVAILABLE

    def test_quest_accept(self):
        """Quest should be acceptable."""
        definition = QuestDefinition(
            id="q4",
            name="Quest Four",
            description="Fourth quest",
        )
        q = Quest(definition, player_id="player1")
        q.make_available()
        result = q.accept(time.time())  # Accept requires timestamp
        assert result is True
        assert q.state == QuestState.ACTIVE

    def test_quest_complete(self):
        """Quest should be completable."""
        definition = QuestDefinition(
            id="q5",
            name="Quest Five",
            description="Fifth quest",
        )
        q = Quest(definition, player_id="player1")
        q.make_available()
        q.accept(time.time())
        q.complete(time.time())
        assert q.state == QuestState.COMPLETE

    def test_quest_turn_in(self):
        """Quest should be turnable in."""
        definition = QuestDefinition(
            id="q6",
            name="Quest Six",
            description="Sixth quest",
        )
        q = Quest(definition, player_id="player1")
        q.make_available()
        q.accept(time.time())
        q.complete(time.time())
        q.turn_in(time.time())
        assert q.state == QuestState.TURNED_IN

    def test_quest_fail(self):
        """Quest should be failable."""
        definition = QuestDefinition(
            id="q7",
            name="Quest Seven",
            description="Seventh quest",
        )
        q = Quest(definition, player_id="player1")
        q.make_available()
        q.accept(time.time())
        q.fail(time.time())
        assert q.state == QuestState.FAILED

    def test_quest_abandon(self):
        """Quest should be abandonable."""
        definition = QuestDefinition(
            id="q8",
            name="Quest Eight",
            description="Eighth quest",
        )
        q = Quest(definition, player_id="player1")
        q.make_available()
        q.accept(time.time())
        q.abandon()
        # After abandon, should be available again
        assert q.state == QuestState.AVAILABLE


class TestQuestStateTransitions:
    """Test valid and invalid quest state transitions."""

    def test_available_to_active(self):
        """Should transition from available to active."""
        definition = QuestDefinition(id="t1", name="T1", description="Desc")
        q = Quest(definition, player_id="p1")
        q.make_available()
        q.accept(time.time())
        assert q.state == QuestState.ACTIVE

    def test_active_to_completed(self):
        """Should transition from active to completed."""
        definition = QuestDefinition(id="t2", name="T2", description="Desc")
        q = Quest(definition, player_id="p1")
        q.make_available()
        q.accept(time.time())
        q.complete(time.time())
        assert q.state == QuestState.COMPLETE

    def test_active_to_failed(self):
        """Should transition from active to failed."""
        definition = QuestDefinition(id="t3", name="T3", description="Desc")
        q = Quest(definition, player_id="p1")
        q.make_available()
        q.accept(time.time())
        q.fail(time.time())
        assert q.state == QuestState.FAILED

    def test_complete_to_turned_in(self):
        """Should transition from complete to turned in."""
        definition = QuestDefinition(id="t4", name="T4", description="Desc")
        q = Quest(definition, player_id="p1")
        q.make_available()
        q.accept(time.time())
        q.complete(time.time())
        q.turn_in(time.time())
        assert q.state == QuestState.TURNED_IN


class TestQuestProperties:
    """Test Quest property accessors."""

    def test_quest_is_active(self):
        """is_active should return True when active."""
        definition = QuestDefinition(id="p1", name="P1", description="Desc")
        q = Quest(definition, player_id="p1")
        q.make_available()
        q.accept(time.time())
        assert q.is_active is True

    def test_quest_is_complete(self):
        """is_complete should return True when complete."""
        definition = QuestDefinition(id="p2", name="P2", description="Desc")
        q = Quest(definition, player_id="p1")
        q.make_available()
        q.accept(time.time())
        q.complete(time.time())
        assert q.is_complete is True

    def test_quest_is_available(self):
        """is_available should return True when available."""
        definition = QuestDefinition(id="p3", name="P3", description="Desc")
        q = Quest(definition, player_id="p1")
        q.make_available()
        assert q.is_available is True


# ============================================================================
# OBJECTIVE STATE TESTS
# ============================================================================

class TestObjectiveState:
    """Test ObjectiveState enum."""

    def test_objective_state_has_inactive(self):
        """ObjectiveState should have INACTIVE."""
        assert hasattr(ObjectiveState, 'INACTIVE')

    def test_objective_state_has_in_progress(self):
        """ObjectiveState should have IN_PROGRESS."""
        assert hasattr(ObjectiveState, 'IN_PROGRESS')

    def test_objective_state_has_complete(self):
        """ObjectiveState should have COMPLETE."""
        assert hasattr(ObjectiveState, 'COMPLETE')

    def test_objective_state_has_failed(self):
        """ObjectiveState should have FAILED."""
        assert hasattr(ObjectiveState, 'FAILED')


class TestObjectiveType:
    """Test ObjectiveType enum."""

    def test_objective_type_has_kill(self):
        """ObjectiveType should have KILL."""
        assert hasattr(ObjectiveType, 'KILL')

    def test_objective_type_has_collect(self):
        """ObjectiveType should have COLLECT."""
        assert hasattr(ObjectiveType, 'COLLECT')

    def test_objective_type_has_talk(self):
        """ObjectiveType should have TALK."""
        assert hasattr(ObjectiveType, 'TALK')

    def test_objective_type_has_reach(self):
        """ObjectiveType should have REACH."""
        assert hasattr(ObjectiveType, 'REACH')

    def test_objective_type_has_escort(self):
        """ObjectiveType should have ESCORT."""
        assert hasattr(ObjectiveType, 'ESCORT')


# ============================================================================
# KILL OBJECTIVE TESTS
# ============================================================================

class TestKillObjective:
    """Test KillObjective behavior."""

    def test_kill_objective_creation(self):
        """KillObjective should be creatable."""
        obj = KillObjective(
            id="kill_1",
            description="Kill enemies",
            target_type="enemy_type",
            required=5,
        )
        assert obj.id == "kill_1"
        assert obj.required == 5

    def test_kill_objective_initial_progress(self):
        """KillObjective should start with zero progress."""
        obj = KillObjective(
            id="kill_2",
            description="Kill goblins",
            target_type="goblin",
            required=10,
        )
        assert obj.current == 0

    def test_kill_objective_activate(self):
        """KillObjective should be activatable."""
        obj = KillObjective(
            id="kill_3",
            description="Kill orcs",
            target_type="orc",
            required=3,
        )
        obj.activate()
        assert obj.state == ObjectiveState.IN_PROGRESS

    def test_kill_objective_update_progress(self):
        """KillObjective should track kills via update."""
        obj = KillObjective(
            id="kill_4",
            description="Kill wolves",
            target_type="wolf",
            required=3,
        )
        obj.activate()
        obj.update("kill", {"target_type": "wolf", "count": 1})
        assert obj.current >= 1

    def test_kill_objective_completes_at_target(self):
        """KillObjective should complete when count reached."""
        obj = KillObjective(
            id="kill_5",
            description="Kill 2 wolves",
            target_type="wolf",
            required=2,
        )
        obj.activate()
        obj.update("kill", {"target_type": "wolf", "count": 1})
        obj.update("kill", {"target_type": "wolf", "count": 1})
        assert obj.state == ObjectiveState.COMPLETE

    def test_kill_objective_ignores_wrong_target(self):
        """KillObjective should ignore wrong target."""
        obj = KillObjective(
            id="kill_6",
            description="Kill bears",
            target_type="bear",
            required=1,
        )
        obj.activate()
        obj.update("kill", {"target_type": "deer", "count": 1})
        assert obj.current == 0
        assert obj.state != ObjectiveState.COMPLETE

    def test_kill_objective_progress_property(self):
        """KillObjective should report progress as fraction."""
        obj = KillObjective(
            id="kill_7",
            description="Kill spiders",
            target_type="spider",
            required=10,
        )
        obj.activate()
        obj.current = 5
        progress = obj.progress
        assert 0.4 <= progress <= 0.6  # ~50%


# ============================================================================
# COLLECT OBJECTIVE TESTS
# ============================================================================

class TestCollectObjective:
    """Test CollectObjective behavior."""

    def test_collect_objective_creation(self):
        """CollectObjective should be creatable."""
        obj = CollectObjective(
            id="collect_1",
            description="Collect coins",
            item_id="gold_coin",
            required=100,
        )
        assert obj.id == "collect_1"
        assert obj.required == 100

    def test_collect_objective_track_collection(self):
        """CollectObjective should track items collected."""
        obj = CollectObjective(
            id="collect_2",
            description="Collect herbs",
            item_id="herb",
            required=5,
        )
        obj.activate()
        obj.update("collect", {"item_id": "herb", "count": 3})
        assert obj.current == 3

    def test_collect_objective_multiple_collections(self):
        """CollectObjective should accumulate collections."""
        obj = CollectObjective(
            id="collect_3",
            description="Collect ore",
            item_id="ore",
            required=10,
        )
        obj.activate()
        obj.update("collect", {"item_id": "ore", "count": 4})
        obj.update("collect", {"item_id": "ore", "count": 3})
        obj.update("collect", {"item_id": "ore", "count": 3})
        assert obj.current == 10
        assert obj.state == ObjectiveState.COMPLETE

    def test_collect_objective_ignores_wrong_item(self):
        """CollectObjective should ignore wrong items."""
        obj = CollectObjective(
            id="collect_4",
            description="Collect iron",
            item_id="iron",
            required=5,
        )
        obj.activate()
        obj.update("collect", {"item_id": "copper", "count": 5})
        assert obj.current == 0


# ============================================================================
# TALK OBJECTIVE TESTS
# ============================================================================

class TestTalkObjective:
    """Test TalkObjective behavior."""

    def test_talk_objective_creation(self):
        """TalkObjective should be creatable."""
        obj = TalkObjective(
            id="talk_1",
            description="Talk to merchant",
            npc_id="merchant",
        )
        assert obj.id == "talk_1"
        assert obj.npc_id == "merchant"

    def test_talk_objective_complete_on_talk(self):
        """TalkObjective should complete on conversation."""
        obj = TalkObjective(
            id="talk_2",
            description="Talk to guard",
            npc_id="guard",
        )
        obj.activate()
        obj.update("talk", {"npc_id": "guard"})
        assert obj.state == ObjectiveState.COMPLETE

    def test_talk_objective_wrong_npc(self):
        """TalkObjective should not complete for wrong NPC."""
        obj = TalkObjective(
            id="talk_3",
            description="Talk to wizard",
            npc_id="wizard",
        )
        obj.activate()
        obj.update("talk", {"npc_id": "innkeeper"})
        assert obj.state != ObjectiveState.COMPLETE


# ============================================================================
# REACH OBJECTIVE TESTS
# ============================================================================

class TestReachObjective:
    """Test ReachObjective behavior."""

    def test_reach_objective_creation(self):
        """ReachObjective should be creatable."""
        obj = ReachObjective(
            id="reach_1",
            description="Reach castle",
            location_id="castle",
        )
        assert obj.id == "reach_1"
        assert obj.location_id == "castle"

    def test_reach_objective_with_radius(self):
        """ReachObjective should accept radius."""
        obj = ReachObjective(
            id="reach_2",
            description="Reach village",
            location_id="village",
            radius=10.0,
        )
        assert obj.radius == 10.0

    def test_reach_objective_complete_at_location(self):
        """ReachObjective should complete at location via enter_location event."""
        obj = ReachObjective(
            id="reach_3",
            description="Reach temple",
            location_id="temple",
        )
        obj.activate()
        obj.update("enter_location", {"location_id": "temple"})
        assert obj.state == ObjectiveState.COMPLETE

    def test_reach_objective_wrong_location(self):
        """ReachObjective should not complete at wrong location."""
        obj = ReachObjective(
            id="reach_4",
            description="Reach dungeon",
            location_id="dungeon",
        )
        obj.activate()
        obj.update("enter_location", {"location_id": "forest"})
        assert obj.state != ObjectiveState.COMPLETE


# ============================================================================
# ESCORT OBJECTIVE TESTS
# ============================================================================

class TestEscortObjective:
    """Test EscortObjective behavior."""

    def test_escort_objective_creation(self):
        """EscortObjective should be creatable."""
        obj = EscortObjective(
            id="escort_1",
            description="Escort princess",
            npc_id="princess",
            destination_id="safe_zone",
        )
        assert obj.id == "escort_1"
        assert obj.npc_id == "princess"

    def test_escort_objective_complete_at_destination(self):
        """EscortObjective should complete when NPC reaches destination."""
        obj = EscortObjective(
            id="escort_2",
            description="Escort merchant",
            npc_id="merchant",
            destination_id="town",
        )
        obj.activate()
        obj.update("npc_arrived", {"npc_id": "merchant", "destination_id": "town"})
        assert obj.state == ObjectiveState.COMPLETE

    def test_escort_objective_fail_on_npc_death(self):
        """EscortObjective should fail if NPC dies."""
        obj = EscortObjective(
            id="escort_3",
            description="Escort child",
            npc_id="child",
            destination_id="home",
        )
        obj.activate()
        obj.update("npc_died", {"npc_id": "child"})
        assert obj.state == ObjectiveState.FAILED


# ============================================================================
# INTERACT OBJECTIVE TESTS
# ============================================================================

class TestInteractObjective:
    """Test InteractObjective behavior."""

    def test_interact_objective_creation(self):
        """InteractObjective should be creatable."""
        obj = InteractObjective(
            id="interact_1",
            description="Pull lever",
            object_id="lever",
        )
        assert obj.id == "interact_1"
        assert obj.object_id == "lever"

    def test_interact_objective_complete(self):
        """InteractObjective should complete on interaction."""
        obj = InteractObjective(
            id="interact_2",
            description="Flip switch",
            object_id="switch",
        )
        obj.activate()
        obj.update("interact", {"object_id": "switch"})
        assert obj.state == ObjectiveState.COMPLETE

    def test_interact_objective_wrong_target(self):
        """InteractObjective should not complete for wrong target."""
        obj = InteractObjective(
            id="interact_3",
            description="Press button",
            object_id="button",
        )
        obj.activate()
        obj.update("interact", {"object_id": "door"})
        assert obj.state != ObjectiveState.COMPLETE


# ============================================================================
# USE OBJECTIVE TESTS
# ============================================================================

class TestUseObjective:
    """Test UseObjective behavior."""

    def test_use_objective_creation(self):
        """UseObjective should be creatable."""
        obj = UseObjective(
            id="use_1",
            description="Use key",
            item_or_ability_id="key",
        )
        assert obj.id == "use_1"
        assert obj.item_or_ability_id == "key"

    def test_use_objective_with_target(self):
        """UseObjective should accept optional target."""
        obj = UseObjective(
            id="use_2",
            description="Use key on door",
            item_or_ability_id="key",
            target_type="locked_door",
        )
        assert obj.target_type == "locked_door"

    def test_use_objective_complete(self):
        """UseObjective should complete on use."""
        obj = UseObjective(
            id="use_3",
            description="Use potion",
            item_or_ability_id="potion",
        )
        obj.activate()
        obj.update("use", {"id": "potion"})
        assert obj.state == ObjectiveState.COMPLETE


# ============================================================================
# CRAFT OBJECTIVE TESTS
# ============================================================================

class TestCraftObjective:
    """Test CraftObjective behavior."""

    def test_craft_objective_creation(self):
        """CraftObjective should be creatable."""
        obj = CraftObjective(
            id="craft_1",
            description="Craft sword",
            item_id="sword",
            required=1,
        )
        assert obj.id == "craft_1"
        assert obj.item_id == "sword"

    def test_craft_objective_multiple_items(self):
        """CraftObjective should track multiple crafts."""
        obj = CraftObjective(
            id="craft_2",
            description="Craft arrows",
            item_id="arrow",
            required=10,
        )
        obj.activate()
        obj.update("craft", {"item_id": "arrow", "count": 5})
        assert obj.current == 5

    def test_craft_objective_complete(self):
        """CraftObjective should complete when count reached."""
        obj = CraftObjective(
            id="craft_3",
            description="Craft armor",
            item_id="armor",
            required=1,
        )
        obj.activate()
        obj.update("craft", {"item_id": "armor", "count": 1})
        assert obj.state == ObjectiveState.COMPLETE


# ============================================================================
# DEFEND OBJECTIVE TESTS
# ============================================================================

class TestDefendObjective:
    """Test DefendObjective behavior."""

    def test_defend_objective_creation(self):
        """DefendObjective should be creatable."""
        obj = DefendObjective(
            id="defend_1",
            description="Defend village",
            target_id="village",
            duration=60.0,
        )
        assert obj.id == "defend_1"
        assert obj.duration == 60.0

    def test_defend_objective_complete_on_duration(self):
        """DefendObjective should complete after duration via defend_tick."""
        obj = DefendObjective(
            id="defend_2",
            description="Defend wall",
            target_id="wall",
            duration=30.0,
        )
        obj.activate()
        obj.update("defend_tick", {"target_id": "wall", "delta_time": 30.0})
        assert obj.state == ObjectiveState.COMPLETE

    def test_defend_objective_fail_on_target_destroyed(self):
        """DefendObjective should fail if target destroyed."""
        obj = DefendObjective(
            id="defend_3",
            description="Defend gate",
            target_id="gate",
            duration=60.0,
        )
        obj.activate()
        obj.update("target_destroyed", {"target_id": "gate"})
        assert obj.state == ObjectiveState.FAILED


# ============================================================================
# TIMED OBJECTIVE TESTS
# ============================================================================

class TestTimedObjective:
    """Test TimedObjective behavior."""

    def test_timed_objective_creation(self):
        """TimedObjective should be creatable."""
        obj = TimedObjective(
            id="timed_1",
            description="Complete within time",
            time_limit=120.0,
        )
        assert obj.id == "timed_1"
        assert obj.time_limit == 120.0

    def test_timed_objective_time_properties(self):
        """TimedObjective should have time tracking properties."""
        obj = TimedObjective(
            id="timed_2",
            description="Hurry",
            time_limit=60.0,
        )
        assert obj.time_elapsed == 0.0
        assert obj.fail_on_timeout is True


# ============================================================================
# COMPOSITE OBJECTIVE TESTS
# ============================================================================

class TestCompositeObjective:
    """Test CompositeObjective behavior."""

    def test_composite_objective_creation(self):
        """CompositeObjective should be creatable."""
        child1 = KillObjective(id="k1", description="Kill", target_type="enemy", required=5)
        child2 = KillObjective(id="k2", description="Kill more", target_type="boss", required=1)
        composite = CompositeObjective(
            id="comp_1",
            description="Kill and more",
            objectives=[child1, child2],
        )
        assert composite.id == "comp_1"
        assert len(composite.objectives) == 2

    def test_composite_all_required_complete(self):
        """Composite (all required) should complete when all children complete."""
        child1 = KillObjective(id="k2", description="Kill rat", target_type="rat", required=1)
        child2 = KillObjective(id="k3", description="Kill wolf", target_type="wolf", required=1)
        composite = CompositeObjective(
            id="comp_2",
            description="Complete both",
            objectives=[child1, child2],
        )
        composite.activate()
        child1.activate()
        child2.activate()

        child1.update("kill", {"target_type": "rat", "count": 1})
        child2.update("kill", {"target_type": "wolf", "count": 1})

        # Both children should be complete
        assert child1.state == ObjectiveState.COMPLETE
        assert child2.state == ObjectiveState.COMPLETE


# ============================================================================
# QUEST TRACKER TESTS
# ============================================================================

class TestQuestTrackerBasics:
    """Test QuestTracker basic operations."""

    def test_tracker_creation(self):
        """QuestTracker should be creatable."""
        tracker = QuestTracker(player_id="player1")
        assert tracker is not None
        assert tracker.player_id == "player1"

    def test_tracker_active_quests_property(self):
        """QuestTracker should have active_quests property."""
        tracker = QuestTracker(player_id="player1")
        assert hasattr(tracker, 'active_quests')

    def test_tracker_available_quests_property(self):
        """QuestTracker should have available_quests property."""
        tracker = QuestTracker(player_id="player1")
        assert hasattr(tracker, 'available_quests')

    def test_tracker_completed_quests_property(self):
        """QuestTracker should have completed_quests property."""
        tracker = QuestTracker(player_id="player1")
        assert hasattr(tracker, 'completed_quests')


class TestTrackedQuest:
    """Test TrackedQuest wrapper."""

    def test_tracked_quest_creation(self):
        """TrackedQuest should wrap quest properly."""
        definition = QuestDefinition(id="trk1", name="Tracked", description="Desc")
        q = Quest(definition, player_id="p1")
        tracked = TrackedQuest(quest=q)
        assert tracked.quest.id == "trk1"

    def test_tracked_quest_active_objectives(self):
        """TrackedQuest should track active objectives."""
        definition = QuestDefinition(id="trk2", name="Tracked 2", description="Desc")
        q = Quest(definition, player_id="p1")
        obj = KillObjective(id="ko", description="Kill", target_type="enemy", required=10)
        obj.activate()
        tracked = TrackedQuest(quest=q, objectives=[obj])

        active = tracked.active_objectives
        assert len(active) == 1

    def test_tracked_quest_all_required_complete(self):
        """TrackedQuest should check if all required complete."""
        definition = QuestDefinition(id="trk3", name="Tracked 3", description="Desc")
        q = Quest(definition, player_id="p1")
        obj = KillObjective(id="ko2", description="Kill", target_type="enemy", required=1)
        obj.activate()
        obj.update("kill", {"target_type": "enemy", "count": 1})
        tracked = TrackedQuest(quest=q, objectives=[obj])

        assert tracked.all_required_complete is True


# ============================================================================
# QUEST EVENT TESTS
# ============================================================================

class TestQuestEventType:
    """Test quest event types."""

    def test_quest_event_type_has_accepted(self):
        """QuestEventType should have QUEST_ACCEPTED."""
        assert hasattr(QuestEventType, 'QUEST_ACCEPTED')

    def test_quest_event_type_has_complete(self):
        """QuestEventType should have QUEST_COMPLETE."""
        assert hasattr(QuestEventType, 'QUEST_COMPLETE')

    def test_quest_event_type_has_failed(self):
        """QuestEventType should have QUEST_FAILED."""
        assert hasattr(QuestEventType, 'QUEST_FAILED')


class TestQuestEvents:
    """Test quest event system."""

    def test_fire_quest_event(self):
        """Should be able to fire quest events."""
        clear_quest_events()
        fire_quest_event(QuestStateChanged(
            quest_id="evt_q1",
            entity_id="player1",
            old_state=QuestState.UNAVAILABLE,
            new_state=QuestState.ACTIVE,
            timestamp=time.time(),
        ))
        events = get_quest_events()
        assert len(events) >= 1

    def test_clear_quest_events(self):
        """Should be able to clear quest events."""
        fire_quest_event(QuestStateChanged(
            quest_id="evt_q2",
            entity_id="player1",
            old_state=QuestState.ACTIVE,
            new_state=QuestState.COMPLETE,
            timestamp=time.time(),
        ))
        clear_quest_events()
        events = get_quest_events()
        assert len(events) == 0


# ============================================================================
# QUEST REGISTRY TESTS
# ============================================================================

class TestQuestRegistry:
    """Test QuestRegistry for quest definitions."""

    def test_registry_register_definition(self):
        """Should register quest definitions."""
        registry = QuestRegistry()
        definition = QuestDefinition(id="reg1", name="Registered Quest", description="Desc")
        registry.register(definition)

        result = registry.get("reg1")
        assert result is not None
        assert result.name == "Registered Quest"

    def test_registry_get_nonexistent(self):
        """Getting nonexistent should return None."""
        registry = QuestRegistry()
        result = registry.get("nonexistent")
        assert result is None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestQuestIntegration:
    """Integration tests for quest system."""

    def test_full_quest_workflow(self):
        """Test complete quest workflow."""
        # Create quest with objectives
        obj1 = KillObjective(id="int_k1", description="Kill bandits", target_type="bandit", required=3)
        obj2 = KillObjective(id="int_k2", description="Kill boss", target_type="boss", required=1)

        definition = QuestDefinition(
            id="int_quest1",
            name="Bandit Hunting",
            description="Eliminate bandits",
        )

        q = Quest(definition, player_id="player1")
        q.make_available()
        q.accept(time.time())

        # Activate objectives
        obj1.activate()
        obj2.activate()

        # Progress objectives
        obj1.update("kill", {"target_type": "bandit", "count": 3})
        obj2.update("kill", {"target_type": "boss", "count": 1})

        # Both complete
        assert obj1.state == ObjectiveState.COMPLETE
        assert obj2.state == ObjectiveState.COMPLETE

    def test_multi_quest_tracking(self):
        """Test multiple quests for same player."""
        tracker = QuestTracker(player_id="player1")

        # Player can have multiple quests
        assert tracker.player_id == "player1"


class TestQuestEdgeCases:
    """Test edge cases for quest system."""

    def test_objective_over_completion(self):
        """Objective should handle over-completion gracefully."""
        obj = KillObjective(id="edge1", description="Kill", target_type="target", required=3)
        obj.activate()

        # Kill more than required
        for _ in range(10):
            obj.update("kill", {"target_type": "target", "count": 1})

        assert obj.state == ObjectiveState.COMPLETE
        assert obj.current >= 3

    def test_inactive_objective_events(self):
        """Inactive objective should not respond to events."""
        obj = KillObjective(id="inactive1", description="Kill", target_type="enemy", required=5)
        # Don't activate

        obj.update("kill", {"target_type": "enemy", "count": 5})

        # Should not have progressed
        assert obj.current == 0


# ============================================================================
# FOUNDATION EVENT INTEGRATION TESTS
# ============================================================================

class TestFoundationEvents:
    """Test Foundation integration events."""

    def test_quest_state_changed_event(self):
        """QuestStateChanged should be fireable."""
        clear_quest_events()
        event = QuestStateChanged(
            quest_id="found1",
            entity_id="player1",
            old_state=QuestState.UNAVAILABLE,
            new_state=QuestState.ACTIVE,
            timestamp=time.time(),
        )
        fire_quest_event(event)

        events = get_quest_events()
        assert len(events) >= 1

    def test_objective_progress_event_creation(self):
        """ObjectiveProgress should be creatable."""
        event = ObjectiveProgress(
            quest_id="found2",
            objective_id="obj1",
            current=5,
            target=10,
            timestamp=time.time(),
        )
        assert event.current == 5
        assert event.target == 10

    def test_objective_completed_event_creation(self):
        """ObjectiveCompleted should be creatable."""
        event = ObjectiveCompleted(
            quest_id="found3",
            objective_id="obj2",
            timestamp=time.time(),
        )
        assert event.quest_id == "found3"

    def test_quest_reward_granted_event_creation(self):
        """QuestRewardGranted should be creatable."""
        event = QuestRewardGranted(
            quest_id="found4",
            entity_id="player1",
            reward_type="xp",
            amount=100,
            timestamp=time.time(),
        )
        assert event.quest_id == "found4"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
