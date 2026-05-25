"""
Comprehensive tests for Quest Objectives.

Tests cover:
- Kill objectives (count, type)
- Collect objectives (items, amounts)
- Interact objectives (NPCs, objects)
- Location objectives (reach area)
- Escort objectives
- Timer objectives
- Hidden objectives (revealed later)
- Optional objectives
- Objective progress events
- Parallel vs sequential objectives
"""

import pytest
from dataclasses import dataclass
from typing import Any

from engine.gameplay.quest.objectives import (
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
    ObjectiveFactory,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def kill_objective():
    """Create a basic kill objective."""
    return KillObjective(
        id="kill_wolves",
        description="Kill 10 wolves",
        target_type="wolf",
        required=10,
    )


@pytest.fixture
def collect_objective():
    """Create a basic collect objective."""
    return CollectObjective(
        id="collect_herbs",
        description="Collect 5 healing herbs",
        item_id="healing_herb",
        required=5,
    )


@pytest.fixture
def talk_objective():
    """Create a basic talk objective."""
    return TalkObjective(
        id="talk_guard",
        description="Talk to the guard captain",
        npc_id="guard_captain",
    )


@pytest.fixture
def reach_objective():
    """Create a basic reach location objective."""
    return ReachObjective(
        id="reach_town",
        description="Reach the town square",
        location_id="town_square",
    )


@pytest.fixture
def escort_objective():
    """Create a basic escort objective."""
    return EscortObjective(
        id="escort_merchant",
        description="Escort the merchant to the market",
        npc_id="merchant_01",
        destination_id="market",
    )


@pytest.fixture
def interact_objective():
    """Create a basic interact objective."""
    return InteractObjective(
        id="activate_lever",
        description="Activate the lever",
        object_id="lever_01",
    )


# =============================================================================
# Objective State Tests
# =============================================================================

class TestObjectiveState:
    """Tests for objective state management."""

    def test_objective_initial_state_inactive(self, kill_objective):
        """Test that objective starts in INACTIVE state."""
        assert kill_objective.state == ObjectiveState.INACTIVE

    def test_objective_activate(self, kill_objective):
        """Test activating an objective."""
        result = kill_objective.activate()
        assert result is True
        assert kill_objective.state == ObjectiveState.IN_PROGRESS

    def test_objective_activate_only_from_inactive(self, kill_objective):
        """Test that activate only works from INACTIVE state."""
        kill_objective.state = ObjectiveState.IN_PROGRESS
        result = kill_objective.activate()
        assert result is False

    def test_objective_complete(self, kill_objective):
        """Test completing an objective."""
        kill_objective.state = ObjectiveState.IN_PROGRESS
        result = kill_objective.complete()
        assert result is True
        assert kill_objective.state == ObjectiveState.COMPLETE

    def test_objective_complete_from_inactive(self, kill_objective):
        """Test completing objective from inactive state."""
        result = kill_objective.complete()
        assert result is True
        assert kill_objective.state == ObjectiveState.COMPLETE

    def test_objective_complete_not_from_failed(self, kill_objective):
        """Test that complete doesn't work from FAILED state."""
        kill_objective.state = ObjectiveState.FAILED
        result = kill_objective.complete()
        assert result is False

    def test_objective_fail(self, kill_objective):
        """Test failing an objective."""
        kill_objective.state = ObjectiveState.IN_PROGRESS
        result = kill_objective.fail()
        assert result is True
        assert kill_objective.state == ObjectiveState.FAILED

    def test_objective_fail_only_from_in_progress(self, kill_objective):
        """Test that fail only works from IN_PROGRESS state."""
        result = kill_objective.fail()
        assert result is False
        assert kill_objective.state == ObjectiveState.INACTIVE

    def test_objective_reset(self, kill_objective):
        """Test resetting an objective."""
        kill_objective.state = ObjectiveState.COMPLETE
        kill_objective.reset()
        assert kill_objective.state == ObjectiveState.INACTIVE

    def test_objective_is_complete_property(self, kill_objective):
        """Test is_complete property."""
        assert kill_objective.is_complete is False
        kill_objective.state = ObjectiveState.COMPLETE
        assert kill_objective.is_complete is True

    def test_objective_is_failed_property(self, kill_objective):
        """Test is_failed property."""
        assert kill_objective.is_failed is False
        kill_objective.state = ObjectiveState.FAILED
        assert kill_objective.is_failed is True

    def test_objective_is_active_property(self, kill_objective):
        """Test is_active property."""
        assert kill_objective.is_active is False
        kill_objective.state = ObjectiveState.IN_PROGRESS
        assert kill_objective.is_active is True


# =============================================================================
# Kill Objective Tests
# =============================================================================

class TestKillObjective:
    """Tests for kill objective functionality."""

    def test_kill_objective_creation(self):
        """Test creating a kill objective."""
        obj = KillObjective(
            id="kill_test",
            description="Kill test enemies",
            target_type="enemy",
            required=5,
        )
        assert obj.target_type == "enemy"
        assert obj.required == 5
        assert obj.current == 0

    def test_kill_objective_type(self, kill_objective):
        """Test kill objective type."""
        assert kill_objective.objective_type == ObjectiveType.KILL

    def test_kill_objective_progress(self, kill_objective):
        """Test kill objective progress calculation."""
        assert kill_objective.progress == 0.0
        kill_objective.current = 5
        assert kill_objective.progress == 0.5
        kill_objective.current = 10
        assert kill_objective.progress == 1.0

    def test_kill_objective_progress_text(self, kill_objective):
        """Test kill objective progress text."""
        assert kill_objective.progress_text == "0/10"
        kill_objective.current = 5
        assert kill_objective.progress_text == "5/10"

    def test_kill_objective_update_event(self, kill_objective):
        """Test kill objective update with event."""
        kill_objective.activate()
        result = kill_objective.update("kill", {"target_type": "wolf"})
        assert result is True
        assert kill_objective.current == 1

    def test_kill_objective_update_wrong_event(self, kill_objective):
        """Test kill objective ignores wrong event type."""
        kill_objective.activate()
        result = kill_objective.update("collect", {"target_type": "wolf"})
        assert result is False
        assert kill_objective.current == 0

    def test_kill_objective_update_wrong_target(self, kill_objective):
        """Test kill objective ignores wrong target type."""
        kill_objective.activate()
        result = kill_objective.update("kill", {"target_type": "bear"})
        assert result is False
        assert kill_objective.current == 0

    def test_kill_objective_update_not_active(self, kill_objective):
        """Test kill objective doesn't update when not active."""
        result = kill_objective.update("kill", {"target_type": "wolf"})
        assert result is False
        assert kill_objective.current == 0

    def test_kill_objective_auto_complete(self, kill_objective):
        """Test kill objective auto-completes when required reached."""
        kill_objective.activate()
        for _ in range(10):
            kill_objective.update("kill", {"target_type": "wolf"})
        assert kill_objective.is_complete

    def test_kill_objective_with_count(self):
        """Test kill objective with multiple kills per event."""
        obj = KillObjective(
            id="kill_batch",
            description="Kill enemies",
            target_type="enemy",
            required=10,
        )
        obj.activate()
        obj.update("kill", {"target_type": "enemy", "count": 5})
        assert obj.current == 5

    def test_kill_objective_with_location(self):
        """Test kill objective with location requirement."""
        obj = KillObjective(
            id="kill_location",
            description="Kill enemies in forest",
            target_type="enemy",
            required=5,
            location="forest",
        )
        obj.activate()

        # Wrong location
        result = obj.update("kill", {"target_type": "enemy", "location": "desert"})
        assert result is False

        # Correct location
        result = obj.update("kill", {"target_type": "enemy", "location": "forest"})
        assert result is True
        assert obj.current == 1

    def test_kill_objective_with_weapon(self):
        """Test kill objective with weapon requirement."""
        obj = KillObjective(
            id="kill_weapon",
            description="Kill with sword",
            target_type="enemy",
            required=5,
            weapon_type="sword",
        )
        obj.activate()

        # Wrong weapon
        result = obj.update("kill", {"target_type": "enemy", "weapon_type": "bow"})
        assert result is False

        # Correct weapon
        result = obj.update("kill", {"target_type": "enemy", "weapon_type": "sword"})
        assert result is True

    def test_kill_objective_add_kill(self, kill_objective):
        """Test manually adding kills."""
        kill_objective.activate()
        kill_objective.add_kill(3)
        assert kill_objective.current == 3

    def test_kill_objective_streak_tracking(self):
        """Test kill streak tracking."""
        obj = KillObjective(
            id="kill_streak",
            description="Kill with streak",
            target_type="enemy",
            required=5,
            require_streak=3,
        )
        obj.activate()

        # Build streak
        obj.update("kill", {"target_type": "enemy"})
        assert obj.kill_streak == 1
        obj.update("kill", {"target_type": "enemy"})
        assert obj.kill_streak == 2

        # Break streak with wrong target
        obj.update("kill", {"target_type": "other"})
        assert obj.kill_streak == 0

    def test_kill_objective_validation_empty_target(self):
        """Test kill objective validation with empty target_type."""
        with pytest.raises(ValueError, match="target_type cannot be empty"):
            KillObjective(
                id="invalid",
                description="Invalid",
                target_type="",
                required=5,
            )

    def test_kill_objective_validation_invalid_required(self):
        """Test kill objective validation with invalid required count."""
        with pytest.raises(ValueError, match="required must be > 0"):
            KillObjective(
                id="invalid",
                description="Invalid",
                target_type="enemy",
                required=0,
            )

    def test_kill_objective_empty_id_raises(self):
        """Test that empty id raises ValueError."""
        with pytest.raises(ValueError, match="Objective id cannot be empty"):
            KillObjective(
                id="",
                description="Invalid",
                target_type="enemy",
                required=5,
            )

    def test_kill_objective_progress_capped(self):
        """Test that progress is capped at 1.0."""
        obj = KillObjective(
            id="kill_over",
            description="Kill",
            target_type="enemy",
            required=5,
        )
        obj.current = 10  # More than required
        assert obj.progress == 1.0


# =============================================================================
# Collect Objective Tests
# =============================================================================

class TestCollectObjective:
    """Tests for collect objective functionality."""

    def test_collect_objective_creation(self):
        """Test creating a collect objective."""
        obj = CollectObjective(
            id="collect_test",
            description="Collect items",
            item_id="gold_coin",
            required=100,
        )
        assert obj.item_id == "gold_coin"
        assert obj.required == 100
        assert obj.current == 0

    def test_collect_objective_type(self, collect_objective):
        """Test collect objective type."""
        assert collect_objective.objective_type == ObjectiveType.COLLECT

    def test_collect_objective_progress(self, collect_objective):
        """Test collect objective progress calculation."""
        assert collect_objective.progress == 0.0
        collect_objective.current = 2
        assert collect_objective.progress == pytest.approx(0.4)
        collect_objective.current = 5
        assert collect_objective.progress == 1.0

    def test_collect_objective_progress_text(self, collect_objective):
        """Test collect objective progress text."""
        assert collect_objective.progress_text == "0/5"
        collect_objective.current = 3
        assert collect_objective.progress_text == "3/5"

    def test_collect_objective_update_event(self, collect_objective):
        """Test collect objective update with event."""
        collect_objective.activate()
        result = collect_objective.update("collect", {"item_id": "healing_herb"})
        assert result is True
        assert collect_objective.current == 1

    def test_collect_objective_update_wrong_item(self, collect_objective):
        """Test collect objective ignores wrong item."""
        collect_objective.activate()
        result = collect_objective.update("collect", {"item_id": "poison_herb"})
        assert result is False
        assert collect_objective.current == 0

    def test_collect_objective_with_source(self):
        """Test collect objective with source requirement."""
        obj = CollectObjective(
            id="collect_source",
            description="Collect from mining",
            item_id="ore",
            required=10,
            source_type="mining",
        )
        obj.activate()

        # Wrong source
        result = obj.update("collect", {"item_id": "ore", "source_type": "loot"})
        assert result is False

        # Correct source
        result = obj.update("collect", {"item_id": "ore", "source_type": "mining"})
        assert result is True

    def test_collect_objective_add_items(self, collect_objective):
        """Test manually adding collected items."""
        collect_objective.activate()
        collect_objective.add_items(3)
        assert collect_objective.current == 3

    def test_collect_objective_remove_items(self, collect_objective):
        """Test removing collected items."""
        collect_objective.activate()
        collect_objective.current = 4
        collect_objective.state = ObjectiveState.COMPLETE

        collect_objective.remove_items(2)
        assert collect_objective.current == 2
        # Should reactivate if was complete
        assert collect_objective.state == ObjectiveState.IN_PROGRESS

    def test_collect_objective_remove_items_min_zero(self, collect_objective):
        """Test removing items doesn't go below zero."""
        collect_objective.activate()
        collect_objective.current = 2
        collect_objective.remove_items(5)
        assert collect_objective.current == 0

    def test_collect_objective_auto_remove_flag(self):
        """Test collect objective auto_remove flag."""
        obj = CollectObjective(
            id="collect_keep",
            description="Collect and keep",
            item_id="item",
            required=5,
            auto_remove=False,
        )
        assert obj.auto_remove is False

    def test_collect_objective_validation_empty_item(self):
        """Test collect objective validation with empty item_id."""
        with pytest.raises(ValueError, match="item_id cannot be empty"):
            CollectObjective(
                id="invalid",
                description="Invalid",
                item_id="",
                required=5,
            )

    def test_collect_objective_validation_invalid_required(self):
        """Test collect objective validation with invalid required count."""
        with pytest.raises(ValueError, match="required must be > 0"):
            CollectObjective(
                id="invalid",
                description="Invalid",
                item_id="item",
                required=-1,
            )


# =============================================================================
# Talk Objective Tests
# =============================================================================

class TestTalkObjective:
    """Tests for talk objective functionality."""

    def test_talk_objective_creation(self):
        """Test creating a talk objective."""
        obj = TalkObjective(
            id="talk_test",
            description="Talk to NPC",
            npc_id="npc_001",
        )
        assert obj.npc_id == "npc_001"
        assert obj.talked is False

    def test_talk_objective_type(self, talk_objective):
        """Test talk objective type."""
        assert talk_objective.objective_type == ObjectiveType.TALK

    def test_talk_objective_progress(self, talk_objective):
        """Test talk objective progress calculation."""
        assert talk_objective.progress == 0.0
        talk_objective.talked = True
        assert talk_objective.progress == 1.0

    def test_talk_objective_progress_text(self, talk_objective):
        """Test talk objective progress text."""
        assert talk_objective.progress_text == "Incomplete"
        talk_objective.talked = True
        assert talk_objective.progress_text == "Complete"

    def test_talk_objective_update_event(self, talk_objective):
        """Test talk objective update with event."""
        talk_objective.activate()
        result = talk_objective.update("talk", {"npc_id": "guard_captain"})
        assert result is True
        assert talk_objective.talked is True
        assert talk_objective.is_complete

    def test_talk_objective_update_wrong_npc(self, talk_objective):
        """Test talk objective ignores wrong NPC."""
        talk_objective.activate()
        result = talk_objective.update("talk", {"npc_id": "random_npc"})
        assert result is False
        assert talk_objective.talked is False

    def test_talk_objective_with_dialogue(self):
        """Test talk objective with specific dialogue requirement."""
        obj = TalkObjective(
            id="talk_dialogue",
            description="Complete specific dialogue",
            npc_id="npc_001",
            dialogue_id="quest_dialogue",
        )
        obj.activate()

        # Wrong dialogue
        result = obj.update("talk", {"npc_id": "npc_001", "dialogue_id": "greeting"})
        assert result is False

        # Correct dialogue
        result = obj.update("talk", {"npc_id": "npc_001", "dialogue_id": "quest_dialogue"})
        assert result is True

    def test_talk_objective_mark_talked(self, talk_objective):
        """Test manually marking as talked."""
        talk_objective.activate()
        talk_objective.mark_talked()
        assert talk_objective.talked is True
        assert talk_objective.is_complete

    def test_talk_objective_validation_empty_npc(self):
        """Test talk objective validation with empty npc_id."""
        with pytest.raises(ValueError, match="npc_id cannot be empty"):
            TalkObjective(
                id="invalid",
                description="Invalid",
                npc_id="",
            )


# =============================================================================
# Reach Objective Tests
# =============================================================================

class TestReachObjective:
    """Tests for reach location objective functionality."""

    def test_reach_objective_creation(self):
        """Test creating a reach objective."""
        obj = ReachObjective(
            id="reach_test",
            description="Reach location",
            location_id="location_001",
        )
        assert obj.location_id == "location_001"
        assert obj.reached is False

    def test_reach_objective_type(self, reach_objective):
        """Test reach objective type."""
        assert reach_objective.objective_type == ObjectiveType.REACH

    def test_reach_objective_progress(self, reach_objective):
        """Test reach objective progress calculation."""
        assert reach_objective.progress == 0.0
        reach_objective.reached = True
        assert reach_objective.progress == 1.0

    def test_reach_objective_progress_text(self, reach_objective):
        """Test reach objective progress text."""
        assert reach_objective.progress_text == "Not reached"
        reach_objective.reached = True
        assert reach_objective.progress_text == "Reached"

    def test_reach_objective_update_enter(self, reach_objective):
        """Test reach objective update with enter event."""
        reach_objective.activate()
        result = reach_objective.update("enter_location", {"location_id": "town_square"})
        assert result is True
        assert reach_objective.reached is True
        assert reach_objective.is_complete

    def test_reach_objective_update_wrong_location(self, reach_objective):
        """Test reach objective ignores wrong location."""
        reach_objective.activate()
        result = reach_objective.update("enter_location", {"location_id": "wrong_place"})
        assert result is False
        assert reach_objective.reached is False

    def test_reach_objective_with_duration(self):
        """Test reach objective with stay duration requirement."""
        obj = ReachObjective(
            id="reach_stay",
            description="Stay in location",
            location_id="camp",
            stay_duration=10.0,
        )
        obj.activate()

        # Enter location (doesn't complete immediately)
        obj.update("enter_location", {"location_id": "camp"})
        assert obj.is_complete is False

        # Stay for partial time
        obj.update("location_tick", {"location_id": "camp", "delta_time": 5.0})
        assert obj.progress == pytest.approx(0.5)
        assert obj.is_complete is False

        # Stay for remaining time
        obj.update("location_tick", {"location_id": "camp", "delta_time": 5.0})
        assert obj.is_complete is True

    def test_reach_objective_leave_resets_time(self):
        """Test that leaving location resets stay time."""
        obj = ReachObjective(
            id="reach_stay",
            description="Stay in location",
            location_id="camp",
            stay_duration=10.0,
        )
        obj.activate()

        obj.update("enter_location", {"location_id": "camp"})
        obj.update("location_tick", {"location_id": "camp", "delta_time": 5.0})
        assert obj.time_in_area == 5.0

        obj.update("leave_location", {"location_id": "camp"})
        assert obj.time_in_area == 0.0

    def test_reach_objective_mark_reached(self, reach_objective):
        """Test manually marking as reached."""
        reach_objective.activate()
        reach_objective.mark_reached()
        assert reach_objective.reached is True
        assert reach_objective.is_complete

    def test_reach_objective_validation_empty_location(self):
        """Test reach objective validation with empty location_id."""
        with pytest.raises(ValueError, match="location_id cannot be empty"):
            ReachObjective(
                id="invalid",
                description="Invalid",
                location_id="",
            )

    def test_reach_objective_validation_invalid_radius(self):
        """Test reach objective validation with invalid radius."""
        with pytest.raises(ValueError, match="radius must be > 0"):
            ReachObjective(
                id="invalid",
                description="Invalid",
                location_id="loc",
                radius=0,
            )

    def test_reach_objective_validation_negative_duration(self):
        """Test reach objective validation with negative stay_duration."""
        with pytest.raises(ValueError, match="stay_duration must be >= 0"):
            ReachObjective(
                id="invalid",
                description="Invalid",
                location_id="loc",
                stay_duration=-1,
            )


# =============================================================================
# Escort Objective Tests
# =============================================================================

class TestEscortObjective:
    """Tests for escort objective functionality."""

    def test_escort_objective_creation(self):
        """Test creating an escort objective."""
        obj = EscortObjective(
            id="escort_test",
            description="Escort NPC",
            npc_id="npc_001",
            destination_id="destination",
        )
        assert obj.npc_id == "npc_001"
        assert obj.destination_id == "destination"
        assert obj.escorted is False

    def test_escort_objective_type(self, escort_objective):
        """Test escort objective type."""
        assert escort_objective.objective_type == ObjectiveType.ESCORT

    def test_escort_objective_progress(self, escort_objective):
        """Test escort objective progress calculation."""
        assert escort_objective.progress == 0.0
        escort_objective.escorted = True
        assert escort_objective.progress == 1.0

    def test_escort_objective_progress_text(self, escort_objective):
        """Test escort objective progress text."""
        assert "merchant_01" in escort_objective.progress_text
        assert "Health:" in escort_objective.progress_text

    def test_escort_objective_update_arrived(self, escort_objective):
        """Test escort objective update with arrival event."""
        escort_objective.activate()
        result = escort_objective.update("npc_arrived", {
            "npc_id": "merchant_01",
            "destination_id": "market",
        })
        assert result is True
        assert escort_objective.escorted is True
        assert escort_objective.is_complete

    def test_escort_objective_update_wrong_npc(self, escort_objective):
        """Test escort objective ignores wrong NPC arrival."""
        escort_objective.activate()
        result = escort_objective.update("npc_arrived", {
            "npc_id": "other_npc",
            "destination_id": "market",
        })
        assert result is False

    def test_escort_objective_update_wrong_destination(self, escort_objective):
        """Test escort objective ignores wrong destination."""
        escort_objective.activate()
        result = escort_objective.update("npc_arrived", {
            "npc_id": "merchant_01",
            "destination_id": "wrong_place",
        })
        assert result is False

    def test_escort_objective_npc_damaged(self, escort_objective):
        """Test escort objective NPC damage tracking."""
        escort_objective.activate()
        escort_objective.update("npc_damaged", {
            "npc_id": "merchant_01",
            "health_percent": 75.0,
        })
        assert escort_objective.npc_health_percent == 75.0

    def test_escort_objective_npc_died(self, escort_objective):
        """Test escort objective fails when NPC dies."""
        escort_objective.activate()
        result = escort_objective.update("npc_died", {"npc_id": "merchant_01"})
        assert result is True
        assert escort_objective.npc_health_percent == 0.0
        assert escort_objective.is_failed

    def test_escort_objective_npc_too_far(self):
        """Test escort objective fails when NPC is too far."""
        obj = EscortObjective(
            id="escort_distance",
            description="Escort with distance limit",
            npc_id="npc",
            destination_id="dest",
            distance_threshold=10.0,
        )
        obj.activate()

        result = obj.update("npc_too_far", {"npc_id": "npc", "distance": 15.0})
        assert result is True
        assert obj.is_failed

    def test_escort_objective_min_health_threshold(self):
        """Test escort objective with minimum health threshold."""
        obj = EscortObjective(
            id="escort_health",
            description="Escort with health requirement",
            npc_id="npc",
            destination_id="dest",
            min_health_percent=25.0,
        )
        obj.activate()

        # Damage but above threshold
        obj.update("npc_damaged", {"npc_id": "npc", "health_percent": 30.0})
        assert obj.is_failed is False

        # Damage below threshold
        obj.update("npc_damaged", {"npc_id": "npc", "health_percent": 20.0})
        assert obj.is_failed is True

    def test_escort_objective_validation_empty_npc(self):
        """Test escort objective validation with empty npc_id."""
        with pytest.raises(ValueError, match="npc_id cannot be empty"):
            EscortObjective(
                id="invalid",
                description="Invalid",
                npc_id="",
                destination_id="dest",
            )

    def test_escort_objective_validation_empty_destination(self):
        """Test escort objective validation with empty destination_id."""
        with pytest.raises(ValueError, match="destination_id cannot be empty"):
            EscortObjective(
                id="invalid",
                description="Invalid",
                npc_id="npc",
                destination_id="",
            )


# =============================================================================
# Interact Objective Tests
# =============================================================================

class TestInteractObjective:
    """Tests for interact objective functionality."""

    def test_interact_objective_creation(self):
        """Test creating an interact objective."""
        obj = InteractObjective(
            id="interact_test",
            description="Interact with object",
            object_id="object_001",
        )
        assert obj.object_id == "object_001"
        assert obj.interacted is False

    def test_interact_objective_type(self, interact_objective):
        """Test interact objective type."""
        assert interact_objective.objective_type == ObjectiveType.INTERACT

    def test_interact_objective_progress(self, interact_objective):
        """Test interact objective progress calculation."""
        assert interact_objective.progress == 0.0
        interact_objective.interacted = True
        assert interact_objective.progress == 1.0

    def test_interact_objective_progress_text_single(self, interact_objective):
        """Test interact objective progress text for single interaction."""
        assert interact_objective.progress_text == "Incomplete"
        interact_objective.interacted = True
        assert interact_objective.progress_text == "Done"

    def test_interact_objective_update_event(self, interact_objective):
        """Test interact objective update with event."""
        interact_objective.activate()
        result = interact_objective.update("interact", {
            "object_id": "lever_01",
            "interaction_type": "use",
        })
        assert result is True
        assert interact_objective.interacted is True
        assert interact_objective.is_complete

    def test_interact_objective_update_wrong_object(self, interact_objective):
        """Test interact objective ignores wrong object."""
        interact_objective.activate()
        result = interact_objective.update("interact", {
            "object_id": "other_lever",
            "interaction_type": "use",
        })
        assert result is False

    def test_interact_objective_update_wrong_type(self):
        """Test interact objective ignores wrong interaction type."""
        obj = InteractObjective(
            id="interact_examine",
            description="Examine object",
            object_id="object",
            interaction_type="examine",
        )
        obj.activate()

        result = obj.update("interact", {
            "object_id": "object",
            "interaction_type": "use",
        })
        assert result is False

    def test_interact_objective_multiple_times(self):
        """Test interact objective requiring multiple interactions."""
        obj = InteractObjective(
            id="interact_multi",
            description="Interact 3 times",
            object_id="object",
            times_required=3,
        )
        obj.activate()

        obj.update("interact", {"object_id": "object"})
        assert obj.times_interacted == 1
        assert obj.progress == pytest.approx(1/3)

        obj.update("interact", {"object_id": "object"})
        obj.update("interact", {"object_id": "object"})
        assert obj.is_complete

    def test_interact_objective_progress_text_multiple(self):
        """Test interact objective progress text for multiple interactions."""
        obj = InteractObjective(
            id="interact_multi",
            description="Interact multiple times",
            object_id="object",
            times_required=5,
        )
        obj.times_interacted = 2
        assert obj.progress_text == "2/5"

    def test_interact_objective_validation_empty_object(self):
        """Test interact objective validation with empty object_id."""
        with pytest.raises(ValueError, match="object_id cannot be empty"):
            InteractObjective(
                id="invalid",
                description="Invalid",
                object_id="",
            )

    def test_interact_objective_validation_invalid_times(self):
        """Test interact objective validation with invalid times_required."""
        with pytest.raises(ValueError, match="times_required must be > 0"):
            InteractObjective(
                id="invalid",
                description="Invalid",
                object_id="obj",
                times_required=0,
            )


# =============================================================================
# Use Objective Tests
# =============================================================================

class TestUseObjective:
    """Tests for use item/ability objective functionality."""

    def test_use_objective_creation(self):
        """Test creating a use objective."""
        obj = UseObjective(
            id="use_test",
            description="Use item",
            item_or_ability_id="healing_potion",
            times_required=3,
        )
        assert obj.item_or_ability_id == "healing_potion"
        assert obj.times_required == 3

    def test_use_objective_type(self):
        """Test use objective type."""
        obj = UseObjective(
            id="use_test",
            description="Use item",
            item_or_ability_id="item",
            times_required=1,
        )
        assert obj.objective_type == ObjectiveType.USE

    def test_use_objective_update(self):
        """Test use objective update with event."""
        obj = UseObjective(
            id="use_test",
            description="Use item",
            item_or_ability_id="potion",
            times_required=2,
        )
        obj.activate()

        result = obj.update("use", {"id": "potion"})
        assert result is True
        assert obj.times_used == 1

        obj.update("use", {"id": "potion"})
        assert obj.is_complete

    def test_use_objective_with_target(self):
        """Test use objective with target requirement."""
        obj = UseObjective(
            id="use_target",
            description="Use on enemy",
            item_or_ability_id="attack_spell",
            times_required=5,
            target_type="enemy",
        )
        obj.activate()

        # Wrong target
        result = obj.update("use", {"id": "attack_spell", "target_type": "ally"})
        assert result is False

        # Correct target
        result = obj.update("use", {"id": "attack_spell", "target_type": "enemy"})
        assert result is True

    def test_use_objective_validation_empty_id(self):
        """Test use objective validation with empty item_or_ability_id."""
        with pytest.raises(ValueError, match="item_or_ability_id cannot be empty"):
            UseObjective(
                id="invalid",
                description="Invalid",
                item_or_ability_id="",
                times_required=1,
            )


# =============================================================================
# Craft Objective Tests
# =============================================================================

class TestCraftObjective:
    """Tests for craft objective functionality."""

    def test_craft_objective_creation(self):
        """Test creating a craft objective."""
        obj = CraftObjective(
            id="craft_test",
            description="Craft items",
            item_id="iron_sword",
            required=5,
        )
        assert obj.item_id == "iron_sword"
        assert obj.required == 5

    def test_craft_objective_type(self):
        """Test craft objective type."""
        obj = CraftObjective(
            id="craft_test",
            description="Craft items",
            item_id="item",
            required=1,
        )
        assert obj.objective_type == ObjectiveType.CRAFT

    def test_craft_objective_update(self):
        """Test craft objective update with event."""
        obj = CraftObjective(
            id="craft_test",
            description="Craft swords",
            item_id="sword",
            required=3,
        )
        obj.activate()

        result = obj.update("craft", {"item_id": "sword"})
        assert result is True
        assert obj.current == 1

        obj.update("craft", {"item_id": "sword", "count": 2})
        assert obj.is_complete

    def test_craft_objective_with_recipe(self):
        """Test craft objective with specific recipe requirement."""
        obj = CraftObjective(
            id="craft_recipe",
            description="Craft with recipe",
            item_id="potion",
            required=1,
            recipe_id="master_recipe",
        )
        obj.activate()

        # Wrong recipe
        result = obj.update("craft", {"item_id": "potion", "recipe_id": "basic_recipe"})
        assert result is False

        # Correct recipe
        result = obj.update("craft", {"item_id": "potion", "recipe_id": "master_recipe"})
        assert result is True

    def test_craft_objective_validation_empty_item(self):
        """Test craft objective validation with empty item_id."""
        with pytest.raises(ValueError, match="item_id cannot be empty"):
            CraftObjective(
                id="invalid",
                description="Invalid",
                item_id="",
                required=1,
            )


# =============================================================================
# Defend Objective Tests
# =============================================================================

class TestDefendObjective:
    """Tests for defend objective functionality."""

    def test_defend_objective_creation(self):
        """Test creating a defend objective."""
        obj = DefendObjective(
            id="defend_test",
            description="Defend location",
            target_id="base",
            duration=60.0,
        )
        assert obj.target_id == "base"
        assert obj.duration == 60.0

    def test_defend_objective_type(self):
        """Test defend objective type."""
        obj = DefendObjective(
            id="defend_test",
            description="Defend",
            target_id="target",
            duration=30.0,
        )
        assert obj.objective_type == ObjectiveType.DEFEND

    def test_defend_objective_progress(self):
        """Test defend objective progress calculation."""
        obj = DefendObjective(
            id="defend_test",
            description="Defend",
            target_id="base",
            duration=60.0,
        )
        assert obj.progress == 0.0
        obj.time_defended = 30.0
        assert obj.progress == 0.5
        obj.time_defended = 60.0
        assert obj.progress == 1.0

    def test_defend_objective_progress_text(self):
        """Test defend objective progress text."""
        obj = DefendObjective(
            id="defend_test",
            description="Defend",
            target_id="base",
            duration=60.0,
        )
        assert "60s remaining" in obj.progress_text
        obj.time_defended = 45.0
        assert "15s remaining" in obj.progress_text

    def test_defend_objective_update_tick(self):
        """Test defend objective update with defend tick."""
        obj = DefendObjective(
            id="defend_test",
            description="Defend",
            target_id="base",
            duration=10.0,
        )
        obj.activate()

        obj.update("defend_tick", {"target_id": "base", "delta_time": 5.0})
        assert obj.time_defended == 5.0

        obj.update("defend_tick", {"target_id": "base", "delta_time": 5.0})
        assert obj.is_complete

    def test_defend_objective_target_damaged(self):
        """Test defend objective with target damage."""
        obj = DefendObjective(
            id="defend_test",
            description="Defend",
            target_id="base",
            duration=60.0,
            min_health_percent=25.0,
        )
        obj.activate()

        obj.update("target_damaged", {"target_id": "base", "health_percent": 50.0})
        assert obj.target_health_percent == 50.0
        assert obj.is_failed is False

        obj.update("target_damaged", {"target_id": "base", "health_percent": 20.0})
        assert obj.is_failed is True

    def test_defend_objective_target_destroyed(self):
        """Test defend objective fails when target destroyed."""
        obj = DefendObjective(
            id="defend_test",
            description="Defend",
            target_id="base",
            duration=60.0,
        )
        obj.activate()

        result = obj.update("target_destroyed", {"target_id": "base"})
        assert result is True
        assert obj.is_failed is True

    def test_defend_objective_validation_empty_target(self):
        """Test defend objective validation with empty target_id."""
        with pytest.raises(ValueError, match="target_id cannot be empty"):
            DefendObjective(
                id="invalid",
                description="Invalid",
                target_id="",
                duration=60.0,
            )

    def test_defend_objective_validation_invalid_duration(self):
        """Test defend objective validation with invalid duration."""
        with pytest.raises(ValueError, match="duration must be > 0"):
            DefendObjective(
                id="invalid",
                description="Invalid",
                target_id="target",
                duration=0,
            )


# =============================================================================
# Timed Objective Tests
# =============================================================================

class TestTimedObjective:
    """Tests for timed objective wrapper functionality."""

    def test_timed_objective_creation(self):
        """Test creating a timed objective."""
        inner = KillObjective(
            id="kill",
            description="Kill",
            target_type="enemy",
            required=5,
        )
        timed = TimedObjective(
            id="timed_kill",
            description="Timed kill",
            time_limit=60.0,
            inner_objective=inner,
        )
        assert timed.time_limit == 60.0
        assert timed.inner_objective is inner

    def test_timed_objective_type(self):
        """Test timed objective type."""
        timed = TimedObjective(
            id="timed",
            description="Timed",
            time_limit=60.0,
        )
        assert timed.objective_type == ObjectiveType.TIMED

    def test_timed_objective_progress_from_inner(self, kill_objective):
        """Test timed objective gets progress from inner objective."""
        timed = TimedObjective(
            id="timed",
            description="Timed",
            time_limit=60.0,
            inner_objective=kill_objective,
        )
        kill_objective.current = 5
        assert timed.progress == 0.5

    def test_timed_objective_progress_text(self, kill_objective):
        """Test timed objective progress text includes time."""
        timed = TimedObjective(
            id="timed",
            description="Timed",
            time_limit=60.0,
            inner_objective=kill_objective,
        )
        assert "(60s)" in timed.progress_text
        timed.time_elapsed = 30.0
        assert "(30s)" in timed.progress_text

    def test_timed_objective_time_remaining(self):
        """Test timed objective time_remaining property."""
        timed = TimedObjective(
            id="timed",
            description="Timed",
            time_limit=60.0,
        )
        assert timed.time_remaining == 60.0
        timed.time_elapsed = 45.0
        assert timed.time_remaining == 15.0

    def test_timed_objective_update_time_tick(self):
        """Test timed objective time tracking."""
        timed = TimedObjective(
            id="timed",
            description="Timed",
            time_limit=10.0,
        )
        timed.activate()

        timed.update("time_tick", {"delta_time": 5.0})
        assert timed.time_elapsed == 5.0

    def test_timed_objective_timeout_fail(self):
        """Test timed objective fails on timeout."""
        timed = TimedObjective(
            id="timed",
            description="Timed",
            time_limit=10.0,
            fail_on_timeout=True,
        )
        timed.activate()

        timed.update("time_tick", {"delta_time": 11.0})
        assert timed.is_failed is True

    def test_timed_objective_no_fail_on_timeout(self):
        """Test timed objective doesn't fail with fail_on_timeout=False."""
        timed = TimedObjective(
            id="timed",
            description="Timed",
            time_limit=10.0,
            fail_on_timeout=False,
        )
        timed.activate()

        timed.update("time_tick", {"delta_time": 11.0})
        assert timed.is_failed is False

    def test_timed_objective_forwards_to_inner(self):
        """Test timed objective forwards events to inner objective."""
        kill = KillObjective(
            id="kill",
            description="Kill",
            target_type="enemy",
            required=5,
        )
        timed = TimedObjective(
            id="timed",
            description="Timed",
            time_limit=60.0,
            inner_objective=kill,
        )
        timed.activate()
        kill.activate()

        result = timed.update("kill", {"target_type": "enemy"})
        assert result is True
        assert kill.current == 1

    def test_timed_objective_completes_with_inner(self):
        """Test timed objective completes when inner completes."""
        kill = KillObjective(
            id="kill",
            description="Kill",
            target_type="enemy",
            required=1,
        )
        timed = TimedObjective(
            id="timed",
            description="Timed",
            time_limit=60.0,
            inner_objective=kill,
        )
        timed.activate()
        kill.activate()

        timed.update("kill", {"target_type": "enemy"})
        assert timed.is_complete is True

    def test_timed_objective_fails_with_inner(self):
        """Test timed objective fails when inner fails."""
        escort = EscortObjective(
            id="escort",
            description="Escort",
            npc_id="npc",
            destination_id="dest",
        )
        timed = TimedObjective(
            id="timed",
            description="Timed",
            time_limit=60.0,
            inner_objective=escort,
        )
        timed.activate()
        escort.activate()

        timed.update("npc_died", {"npc_id": "npc"})
        assert timed.is_failed is True

    def test_timed_objective_set_inner(self):
        """Test setting inner objective after creation."""
        timed = TimedObjective(
            id="timed",
            description="Timed",
            time_limit=60.0,
        )
        kill = KillObjective(
            id="kill",
            description="Kill",
            target_type="enemy",
            required=5,
        )
        timed.set_inner(kill)
        assert timed.inner_objective is kill

    def test_timed_objective_validation_invalid_time_limit(self):
        """Test timed objective validation with invalid time_limit."""
        with pytest.raises(ValueError, match="time_limit must be > 0"):
            TimedObjective(
                id="invalid",
                description="Invalid",
                time_limit=0,
            )


# =============================================================================
# Composite Objective Tests
# =============================================================================

class TestCompositeObjective:
    """Tests for composite objective functionality."""

    def test_composite_objective_creation(self):
        """Test creating a composite objective."""
        obj = CompositeObjective(
            id="composite",
            description="Multiple objectives",
            mode="all",
        )
        assert obj.mode == "all"
        assert obj.objectives == []

    def test_composite_objective_type(self):
        """Test composite objective type."""
        obj = CompositeObjective(
            id="composite",
            description="Composite",
            mode="all",
        )
        assert obj.objective_type == ObjectiveType.COMPOSITE

    def test_composite_objective_add_objective(self):
        """Test adding objectives to composite."""
        composite = CompositeObjective(
            id="composite",
            description="Composite",
            mode="all",
        )
        kill = KillObjective(id="kill", description="Kill", target_type="enemy", required=5)
        collect = CollectObjective(id="collect", description="Collect", item_id="item", required=3)

        composite.add_objective(kill)
        composite.add_objective(collect)

        assert len(composite.objectives) == 2

    def test_composite_objective_progress_all_mode(self):
        """Test composite objective progress in 'all' mode."""
        composite = CompositeObjective(
            id="composite",
            description="Composite",
            mode="all",
        )
        kill = KillObjective(id="kill", description="Kill", target_type="enemy", required=10)
        collect = CollectObjective(id="collect", description="Collect", item_id="item", required=10)

        composite.add_objective(kill)
        composite.add_objective(collect)

        kill.current = 5  # 50%
        collect.current = 5  # 50%
        assert composite.progress == pytest.approx(0.5)

    def test_composite_objective_progress_any_mode(self):
        """Test composite objective progress in 'any' mode."""
        composite = CompositeObjective(
            id="composite",
            description="Composite",
            mode="any",
        )
        kill = KillObjective(id="kill", description="Kill", target_type="enemy", required=10)
        collect = CollectObjective(id="collect", description="Collect", item_id="item", required=10)

        composite.add_objective(kill)
        composite.add_objective(collect)

        kill.current = 8  # 80%
        collect.current = 3  # 30%
        assert composite.progress == pytest.approx(0.8)  # Max of both

    def test_composite_objective_progress_text(self):
        """Test composite objective progress text."""
        composite = CompositeObjective(
            id="composite",
            description="Composite",
            mode="all",
        )
        kill = KillObjective(id="kill", description="Kill", target_type="enemy", required=10)
        collect = CollectObjective(id="collect", description="Collect", item_id="item", required=10)

        composite.add_objective(kill)
        composite.add_objective(collect)

        kill.state = ObjectiveState.COMPLETE
        assert composite.progress_text == "1/2"

    def test_composite_objective_activate_all_mode(self):
        """Test composite activates all objectives in 'all' mode."""
        composite = CompositeObjective(
            id="composite",
            description="Composite",
            mode="all",
        )
        kill = KillObjective(id="kill", description="Kill", target_type="enemy", required=5)
        collect = CollectObjective(id="collect", description="Collect", item_id="item", required=3)

        composite.add_objective(kill)
        composite.add_objective(collect)

        composite.activate()

        assert kill.state == ObjectiveState.IN_PROGRESS
        assert collect.state == ObjectiveState.IN_PROGRESS

    def test_composite_objective_activate_sequential_mode(self):
        """Test composite activates only first in 'sequential' mode."""
        composite = CompositeObjective(
            id="composite",
            description="Composite",
            mode="sequential",
        )
        kill = KillObjective(id="kill", description="Kill", target_type="enemy", required=5)
        collect = CollectObjective(id="collect", description="Collect", item_id="item", required=3)

        composite.add_objective(kill)
        composite.add_objective(collect)

        composite.activate()

        assert kill.state == ObjectiveState.IN_PROGRESS
        assert collect.state == ObjectiveState.INACTIVE

    def test_composite_objective_update_all_mode(self):
        """Test composite update in 'all' mode."""
        composite = CompositeObjective(
            id="composite",
            description="Composite",
            mode="all",
        )
        kill = KillObjective(id="kill", description="Kill", target_type="enemy", required=1)
        collect = CollectObjective(id="collect", description="Collect", item_id="item", required=1)

        composite.add_objective(kill)
        composite.add_objective(collect)
        composite.activate()

        # Complete both objectives
        composite.update("kill", {"target_type": "enemy"})
        composite.update("collect", {"item_id": "item"})

        assert composite.is_complete is True

    def test_composite_objective_update_any_mode(self):
        """Test composite update in 'any' mode."""
        composite = CompositeObjective(
            id="composite",
            description="Composite",
            mode="any",
        )
        kill = KillObjective(id="kill", description="Kill", target_type="enemy", required=1)
        collect = CollectObjective(id="collect", description="Collect", item_id="item", required=1)

        composite.add_objective(kill)
        composite.add_objective(collect)
        composite.activate()

        # Complete just one
        composite.update("kill", {"target_type": "enemy"})

        assert composite.is_complete is True

    def test_composite_objective_update_sequential_mode(self):
        """Test composite update in 'sequential' mode."""
        composite = CompositeObjective(
            id="composite",
            description="Composite",
            mode="sequential",
        )
        kill = KillObjective(id="kill", description="Kill", target_type="enemy", required=1)
        collect = CollectObjective(id="collect", description="Collect", item_id="item", required=1)

        composite.add_objective(kill)
        composite.add_objective(collect)
        composite.activate()

        # First objective doesn't complete yet
        assert collect.state == ObjectiveState.INACTIVE

        # Complete first objective
        composite.update("kill", {"target_type": "enemy"})

        # Second objective now active
        assert collect.state == ObjectiveState.IN_PROGRESS

        # Complete second objective
        composite.update("collect", {"item_id": "item"})

        assert composite.is_complete is True

    def test_composite_objective_sequential_failure(self):
        """Test composite fails in 'sequential' mode when objective fails."""
        composite = CompositeObjective(
            id="composite",
            description="Composite",
            mode="sequential",
        )
        escort = EscortObjective(
            id="escort",
            description="Escort",
            npc_id="npc",
            destination_id="dest",
        )
        collect = CollectObjective(id="collect", description="Collect", item_id="item", required=1)

        composite.add_objective(escort)
        composite.add_objective(collect)
        composite.activate()

        # Fail escort
        composite.update("npc_died", {"npc_id": "npc"})

        assert composite.is_failed is True

    def test_composite_objective_validation_invalid_mode(self):
        """Test composite objective validation with invalid mode."""
        with pytest.raises(ValueError, match="mode must be"):
            CompositeObjective(
                id="invalid",
                description="Invalid",
                mode="invalid_mode",
            )

    def test_composite_objective_empty_progress(self):
        """Test composite objective with no sub-objectives has zero progress."""
        composite = CompositeObjective(
            id="composite",
            description="Composite",
            mode="all",
        )
        assert composite.progress == 0.0


# =============================================================================
# Objective Factory Tests
# =============================================================================

class TestObjectiveFactory:
    """Tests for objective factory functionality."""

    def test_factory_create_kill(self):
        """Test factory creates kill objective."""
        obj = ObjectiveFactory.create(
            "kill",
            id="kill_test",
            description="Kill test",
            target_type="enemy",
            required=5,
        )
        assert isinstance(obj, KillObjective)

    def test_factory_create_collect(self):
        """Test factory creates collect objective."""
        obj = ObjectiveFactory.create(
            "collect",
            id="collect_test",
            description="Collect test",
            item_id="item",
            required=5,
        )
        assert isinstance(obj, CollectObjective)

    def test_factory_create_talk(self):
        """Test factory creates talk objective."""
        obj = ObjectiveFactory.create(
            "talk",
            id="talk_test",
            description="Talk test",
            npc_id="npc",
        )
        assert isinstance(obj, TalkObjective)

    def test_factory_create_reach(self):
        """Test factory creates reach objective."""
        obj = ObjectiveFactory.create(
            "reach",
            id="reach_test",
            description="Reach test",
            location_id="loc",
        )
        assert isinstance(obj, ReachObjective)

    def test_factory_create_escort(self):
        """Test factory creates escort objective."""
        obj = ObjectiveFactory.create(
            "escort",
            id="escort_test",
            description="Escort test",
            npc_id="npc",
            destination_id="dest",
        )
        assert isinstance(obj, EscortObjective)

    def test_factory_create_interact(self):
        """Test factory creates interact objective."""
        obj = ObjectiveFactory.create(
            "interact",
            id="interact_test",
            description="Interact test",
            object_id="obj",
        )
        assert isinstance(obj, InteractObjective)

    def test_factory_create_use(self):
        """Test factory creates use objective."""
        obj = ObjectiveFactory.create(
            "use",
            id="use_test",
            description="Use test",
            item_or_ability_id="item",
            times_required=1,
        )
        assert isinstance(obj, UseObjective)

    def test_factory_create_craft(self):
        """Test factory creates craft objective."""
        obj = ObjectiveFactory.create(
            "craft",
            id="craft_test",
            description="Craft test",
            item_id="item",
            required=1,
        )
        assert isinstance(obj, CraftObjective)

    def test_factory_create_defend(self):
        """Test factory creates defend objective."""
        obj = ObjectiveFactory.create(
            "defend",
            id="defend_test",
            description="Defend test",
            target_id="target",
            duration=60.0,
        )
        assert isinstance(obj, DefendObjective)

    def test_factory_create_timed(self):
        """Test factory creates timed objective."""
        obj = ObjectiveFactory.create(
            "timed",
            id="timed_test",
            description="Timed test",
            time_limit=60.0,
        )
        assert isinstance(obj, TimedObjective)

    def test_factory_create_composite(self):
        """Test factory creates composite objective."""
        obj = ObjectiveFactory.create(
            "composite",
            id="composite_test",
            description="Composite test",
            mode="all",
        )
        assert isinstance(obj, CompositeObjective)

    def test_factory_create_unknown_type(self):
        """Test factory raises error for unknown type."""
        with pytest.raises(ValueError, match="Unknown objective type"):
            ObjectiveFactory.create("unknown", id="test", description="Test")

    def test_factory_from_dict(self):
        """Test factory creates objective from dictionary."""
        data = {
            "type": "kill",
            "id": "kill_dict",
            "description": "Kill from dict",
            "target_type": "enemy",
            "required": 10,
        }
        obj = ObjectiveFactory.from_dict(data)
        assert isinstance(obj, KillObjective)
        assert obj.id == "kill_dict"
        assert obj.required == 10

    def test_factory_from_dict_missing_type(self):
        """Test factory raises error for dict without type."""
        data = {
            "id": "no_type",
            "description": "No type",
        }
        with pytest.raises(ValueError, match="must include 'type'"):
            ObjectiveFactory.from_dict(data)

    def test_factory_register_custom(self):
        """Test registering custom objective type."""
        class CustomObjective(Objective):
            @property
            def progress(self):
                return 0.0

            @property
            def progress_text(self):
                return "Custom"

            def update(self, event_type, event_data):
                return False

        ObjectiveFactory.register("custom", CustomObjective)
        obj = ObjectiveFactory.create(
            "custom",
            id="custom_test",
            description="Custom test",
            objective_type=ObjectiveType.CUSTOM,
        )
        assert isinstance(obj, CustomObjective)


# =============================================================================
# Optional Objective Tests
# =============================================================================

class TestOptionalObjectives:
    """Tests for optional objective functionality."""

    def test_objective_optional_flag(self):
        """Test objective optional flag."""
        obj = KillObjective(
            id="optional_kill",
            description="Optional kill",
            target_type="enemy",
            required=5,
            optional=True,
        )
        assert obj.optional is True

    def test_objective_not_optional_by_default(self, kill_objective):
        """Test objectives are not optional by default."""
        assert kill_objective.optional is False

    def test_objective_hidden_flag(self):
        """Test objective hidden flag."""
        obj = KillObjective(
            id="hidden_kill",
            description="Hidden kill",
            target_type="enemy",
            required=5,
            hidden=True,
        )
        assert obj.hidden is True

    def test_objective_not_hidden_by_default(self, kill_objective):
        """Test objectives are not hidden by default."""
        assert kill_objective.hidden is False

    def test_objective_order_attribute(self):
        """Test objective order attribute."""
        obj = KillObjective(
            id="ordered_kill",
            description="Ordered kill",
            target_type="enemy",
            required=5,
            order=2,
        )
        assert obj.order == 2


# =============================================================================
# Objective Callback Tests
# =============================================================================

class TestObjectiveCallbacks:
    """Tests for objective callbacks."""

    def test_on_complete_callback(self, kill_objective):
        """Test on_complete callback is called."""
        completed = []

        def callback(obj):
            completed.append(obj.id)

        kill_objective.on_complete = callback
        kill_objective.complete()

        assert "kill_wolves" in completed

    def test_on_fail_callback(self):
        """Test on_fail callback is called."""
        failed = []

        def callback(obj):
            failed.append(obj.id)

        escort = EscortObjective(
            id="escort_fail",
            description="Escort",
            npc_id="npc",
            destination_id="dest",
        )
        escort.on_fail = callback
        escort.activate()
        escort.fail()

        assert "escort_fail" in failed

    def test_on_progress_callback(self, kill_objective):
        """Test on_progress callback is called."""
        progress_updates = []

        def callback(obj, progress):
            progress_updates.append((obj.id, progress))

        kill_objective.on_progress = callback
        kill_objective.activate()
        kill_objective.update("kill", {"target_type": "wolf"})

        assert len(progress_updates) == 1
        assert progress_updates[0][0] == "kill_wolves"
        assert progress_updates[0][1] == pytest.approx(0.1)


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestObjectiveEdgeCases:
    """Tests for objective edge cases and boundary conditions."""

    def test_kill_objective_negative_current_handled(self):
        """Test that negative current doesn't cause issues."""
        obj = KillObjective(
            id="kill",
            description="Kill",
            target_type="enemy",
            required=5,
        )
        obj.current = -1  # Shouldn't happen but test it
        assert obj.progress == pytest.approx(-0.2)  # Shows negative

    def test_collect_objective_remove_more_than_collected(self, collect_objective):
        """Test removing more items than collected."""
        collect_objective.current = 2
        collect_objective.remove_items(5)
        assert collect_objective.current == 0

    def test_composite_with_single_objective(self):
        """Test composite with single objective."""
        composite = CompositeObjective(
            id="single",
            description="Single",
            mode="all",
        )
        kill = KillObjective(id="kill", description="Kill", target_type="e", required=1)
        composite.add_objective(kill)
        composite.activate()

        composite.update("kill", {"target_type": "e"})
        assert composite.is_complete is True

    def test_multiple_updates_same_event(self, kill_objective):
        """Test multiple updates with same event type."""
        kill_objective.activate()

        for _ in range(15):  # More than required
            kill_objective.update("kill", {"target_type": "wolf"})

        # Should stop at complete
        assert kill_objective.is_complete
        # Current might exceed required
        assert kill_objective.current >= 10

    def test_objective_state_transitions_all(self):
        """Test all valid state transitions."""
        obj = KillObjective(id="k", description="K", target_type="e", required=1)

        # INACTIVE -> IN_PROGRESS
        assert obj.activate() is True

        # IN_PROGRESS -> COMPLETE
        assert obj.complete() is True

        # Reset
        obj.reset()
        assert obj.state == ObjectiveState.INACTIVE

        # INACTIVE -> IN_PROGRESS
        obj.activate()

        # IN_PROGRESS -> FAILED
        assert obj.fail() is True
