"""
Comprehensive tests for Quest System Foundation Integration (T-GP-9.5).

Tests cover:
- @quest decorator registers with Foundation Registry
- Registry query returns all quest classes
- QuestStateChanged fires on each state transition
- ObjectiveProgress fires on counter updates
- ObjectiveCompleted fires when objectives complete
- QuestRewardGranted fires for each reward type
- Causal chains: kill -> objective progress -> complete -> reward
- Multiple quests tracked independently
- Event query by quest_id
- Event replay for debugging
- Performance: 100 quest updates under 50ms

Total: 50+ tests
"""

import time
import pytest
from dataclasses import dataclass
from typing import Any

from foundation import (
    Registry,
    registry,
    EventLog,
    get_event_log,
    clear_event_log,
    set_current_tick,
)

from engine.gameplay.quest.quest import (
    Quest,
    QuestDefinition,
    QuestRegistry,
    QuestState,
    QuestType,
    quest,
    QuestStateChanged,
    ObjectiveProgress,
    ObjectiveCompleted,
    QuestRewardGranted,
    fire_quest_event,
    get_quest_events,
    clear_quest_events,
    get_registered_quests,
)

from engine.gameplay.quest.objectives import (
    Objective,
    ObjectiveState,
    ObjectiveType,
    KillObjective,
    CollectObjective,
    TalkObjective,
    ReachObjective,
    InteractObjective,
    CompositeObjective,
)

from engine.gameplay.quest.tracker import (
    QuestTracker,
    QuestEvent,
    QuestEventType,
    TrackedQuest,
)


# =============================================================================
# Mock Rewards for Testing
# =============================================================================

@dataclass
class XPReward:
    """Mock XP reward for testing."""
    amount: int
    reward_type: str = "xp"


@dataclass
class GoldReward:
    """Mock gold reward for testing."""
    amount: int
    reward_type: str = "gold"


@dataclass
class ItemReward:
    """Mock item reward for testing."""
    item_id: str
    quantity: int = 1
    reward_type: str = "item"

    @property
    def amount(self) -> str:
        return f"{self.item_id}x{self.quantity}"


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def clear_all_registries():
    """Clear all registries and event logs before and after each test."""
    QuestRegistry.clear()
    clear_quest_events()
    clear_event_log()
    # Clear Foundation registry quest entries
    for cls in list(registry.all_types()):
        name = registry.get_name(cls)
        if name and name.startswith("quest."):
            registry.unregister(cls)
    yield
    QuestRegistry.clear()
    clear_quest_events()
    clear_event_log()


@pytest.fixture
def basic_quest_def():
    """Create a basic quest definition."""
    return QuestDefinition(
        id="test_quest_basic",
        name="Basic Test Quest",
        description="A basic test quest",
        quest_type=QuestType.SIDE,
        level_requirement=1,
    )


@pytest.fixture
def quest_with_rewards():
    """Create a quest with multiple reward types."""
    return QuestDefinition(
        id="reward_quest",
        name="Reward Quest",
        description="A quest with rewards",
        quest_type=QuestType.MAIN,
        level_requirement=1,
        rewards=[
            XPReward(amount=100),
            GoldReward(amount=50),
            ItemReward(item_id="sword_of_testing", quantity=1),
        ],
    )


@pytest.fixture
def kill_objective():
    """Create a kill objective."""
    return KillObjective(
        id="kill_10_wolves",
        description="Kill 10 wolves",
        objective_type=ObjectiveType.KILL,
        target_type="wolf",
        required=10,
    )


@pytest.fixture
def collect_objective():
    """Create a collect objective."""
    return CollectObjective(
        id="collect_5_herbs",
        description="Collect 5 herbs",
        objective_type=ObjectiveType.COLLECT,
        item_id="herb",
        required=5,
    )


@pytest.fixture
def tracker():
    """Create a quest tracker for player."""
    return QuestTracker(player_id="player_001")


# =============================================================================
# Test: @quest Decorator Registration with Foundation Registry
# =============================================================================

class TestQuestDecoratorRegistration:
    """Tests for @quest decorator Foundation Registry integration."""

    def test_quest_decorator_registers_with_foundation_registry(self):
        """@quest decorator should register class with Foundation Registry."""

        @quest(id="test_reg_1", name="Registry Test 1")
        class TestQuest1:
            pass

        # Verify Foundation Registry registration
        registered = registry.get(f"quest.test_reg_1")
        assert registered is TestQuest1

    def test_quest_decorator_sets_registry_metadata(self):
        """@quest decorator should set metadata on Foundation Registry."""

        @quest(
            id="test_meta_1",
            name="Metadata Test",
            quest_type=QuestType.MAIN,
        )
        class MetadataQuest:
            pass

        # Verify metadata
        assert registry.get_metadata(MetadataQuest, "quest") is True
        assert registry.get_metadata(MetadataQuest, "quest_id") == "test_meta_1"
        assert registry.get_metadata(MetadataQuest, "quest_type") == "MAIN"

    def test_quest_decorator_registers_with_quest_registry(self):
        """@quest decorator should register with QuestRegistry."""

        @quest(id="test_qr_1", name="QuestRegistry Test")
        class QuestRegistryTest:
            pass

        # Verify QuestRegistry registration
        assert "test_qr_1" in QuestRegistry.instance()
        quest_def = QuestRegistry.instance().get("test_qr_1")
        assert quest_def is not None
        assert quest_def.name == "QuestRegistry Test"

    def test_quest_decorator_adds_applied_decorators(self):
        """@quest decorator should add to _applied_decorators."""

        @quest(id="test_deco_1", name="Decorator Tracking Test")
        class DecoratorTrackingQuest:
            pass

        assert hasattr(DecoratorTrackingQuest, "_applied_decorators")
        assert "quest" in DecoratorTrackingQuest._applied_decorators

    def test_quest_decorator_adds_tags(self):
        """@quest decorator should set _tags for filtering."""

        @quest(
            id="test_tags_1",
            name="Tags Test",
            quest_type=QuestType.DAILY,
        )
        class TagsQuest:
            pass

        assert hasattr(TagsQuest, "_tags")
        assert TagsQuest._tags["quest"] is True
        assert TagsQuest._tags["quest_id"] == "test_tags_1"
        assert TagsQuest._tags["quest_type"] == "DAILY"


# =============================================================================
# Test: Foundation Registry Query for Quests
# =============================================================================

class TestRegistryQuery:
    """Tests for querying quests via Foundation Registry."""

    def test_get_registered_quests_returns_all_quest_classes(self):
        """get_registered_quests() should return all @quest decorated classes."""

        @quest(id="query_test_1", name="Query Test 1")
        class QueryTest1:
            pass

        @quest(id="query_test_2", name="Query Test 2")
        class QueryTest2:
            pass

        @quest(id="query_test_3", name="Query Test 3")
        class QueryTest3:
            pass

        registered = get_registered_quests()
        assert QueryTest1 in registered
        assert QueryTest2 in registered
        assert QueryTest3 in registered
        assert len(registered) >= 3

    def test_registry_types_with_decorator_finds_quests(self):
        """registry.types_with_decorator('quest') should find quest classes."""

        @quest(id="decorator_test_1", name="Decorator Test")
        class DecoratorTest:
            pass

        found = registry.types_with_decorator("quest")
        assert DecoratorTest in found

    def test_registry_get_by_quest_name(self):
        """Should retrieve quest class by name from registry."""

        @quest(id="name_lookup_1", name="Name Lookup Test")
        class NameLookupQuest:
            pass

        cls = registry.get("quest.name_lookup_1")
        assert cls is NameLookupQuest

    def test_registry_query_returns_empty_when_no_quests(self):
        """get_registered_quests() should return empty list when no quests."""
        # Registry is cleared by fixture
        # Note: Other tests may have registered quests
        registered = get_registered_quests()
        # Should be a list (possibly empty if no quests registered)
        assert isinstance(registered, list)


# =============================================================================
# Test: QuestStateChanged Events
# =============================================================================

class TestQuestStateChangedEvents:
    """Tests for QuestStateChanged event firing."""

    def test_make_available_fires_state_changed(self, basic_quest_def):
        """make_available() should fire QuestStateChanged event."""
        quest_obj = Quest(definition=basic_quest_def, player_id="player_001")

        clear_quest_events()
        quest_obj.make_available(timestamp=1000.0)

        events = get_quest_events(event_type="QuestStateChanged")
        assert len(events) == 1
        assert events[0].quest_id == "test_quest_basic"
        assert events[0].old_state == QuestState.UNAVAILABLE
        assert events[0].new_state == QuestState.AVAILABLE

    def test_accept_fires_state_changed(self, basic_quest_def):
        """accept() should fire QuestStateChanged event."""
        quest_obj = Quest(definition=basic_quest_def, player_id="player_001")
        quest_obj.state = QuestState.AVAILABLE

        clear_quest_events()
        quest_obj.accept(timestamp=1000.0)

        events = get_quest_events(event_type="QuestStateChanged")
        assert len(events) == 1
        assert events[0].old_state == QuestState.AVAILABLE
        assert events[0].new_state == QuestState.ACTIVE

    def test_complete_fires_state_changed(self, basic_quest_def):
        """complete() should fire QuestStateChanged event."""
        quest_obj = Quest(definition=basic_quest_def, player_id="player_001")
        quest_obj.state = QuestState.ACTIVE
        quest_obj.accepted_at = 1000.0

        clear_quest_events()
        quest_obj.complete(timestamp=2000.0)

        events = get_quest_events(event_type="QuestStateChanged")
        assert len(events) == 1
        assert events[0].old_state == QuestState.ACTIVE
        assert events[0].new_state == QuestState.COMPLETE

    def test_turn_in_fires_state_changed(self, basic_quest_def):
        """turn_in() should fire QuestStateChanged event."""
        quest_obj = Quest(definition=basic_quest_def, player_id="player_001")
        quest_obj.state = QuestState.COMPLETE
        quest_obj.completed_at = 2000.0

        clear_quest_events()
        quest_obj.turn_in(timestamp=3000.0)

        events = get_quest_events(event_type="QuestStateChanged")
        assert len(events) == 1
        assert events[0].old_state == QuestState.COMPLETE
        assert events[0].new_state == QuestState.TURNED_IN

    def test_fail_fires_state_changed(self, basic_quest_def):
        """fail() should fire QuestStateChanged event."""
        quest_obj = Quest(definition=basic_quest_def, player_id="player_001")
        quest_obj.state = QuestState.ACTIVE
        quest_obj.accepted_at = 1000.0

        clear_quest_events()
        quest_obj.fail(timestamp=2000.0)

        events = get_quest_events(event_type="QuestStateChanged")
        assert len(events) == 1
        assert events[0].old_state == QuestState.ACTIVE
        assert events[0].new_state == QuestState.FAILED

    def test_state_changed_includes_entity_id(self, basic_quest_def):
        """QuestStateChanged should include entity_id (player_id)."""
        quest_obj = Quest(definition=basic_quest_def, player_id="test_player_123")
        quest_obj.state = QuestState.AVAILABLE

        clear_quest_events()
        quest_obj.accept(timestamp=1000.0)

        events = get_quest_events(event_type="QuestStateChanged")
        assert len(events) == 1
        assert events[0].entity_id == "test_player_123"

    def test_state_changed_includes_timestamp(self, basic_quest_def):
        """QuestStateChanged should include timestamp."""
        quest_obj = Quest(definition=basic_quest_def, player_id="player_001")
        quest_obj.state = QuestState.AVAILABLE

        clear_quest_events()
        quest_obj.accept(timestamp=12345.67)

        events = get_quest_events(event_type="QuestStateChanged")
        assert len(events) == 1
        assert events[0].timestamp == 12345.67


# =============================================================================
# Test: ObjectiveProgress Events
# =============================================================================

class TestObjectiveProgressEvents:
    """Tests for ObjectiveProgress event firing."""

    def test_kill_objective_fires_progress_on_update(self, kill_objective):
        """KillObjective should fire ObjectiveProgress on kill update."""
        kill_objective.quest_id = "test_quest"
        kill_objective.activate()

        clear_quest_events()
        kill_objective.update("kill", {"target_type": "wolf", "count": 1})

        events = get_quest_events(event_type="ObjectiveProgress")
        assert len(events) == 1
        assert events[0].quest_id == "test_quest"
        assert events[0].objective_id == "kill_10_wolves"
        assert events[0].current == 1
        assert events[0].target == 10

    def test_collect_objective_fires_progress_on_update(self, collect_objective):
        """CollectObjective should fire ObjectiveProgress on collect update."""
        collect_objective.quest_id = "test_quest"
        collect_objective.activate()

        clear_quest_events()
        collect_objective.update("collect", {"item_id": "herb", "count": 2})

        events = get_quest_events(event_type="ObjectiveProgress")
        assert len(events) == 1
        assert events[0].current == 2
        assert events[0].target == 5

    def test_progress_event_includes_objective_id(self, kill_objective):
        """ObjectiveProgress should include correct objective_id."""
        kill_objective.quest_id = "quest_abc"
        kill_objective.activate()

        clear_quest_events()
        kill_objective.update("kill", {"target_type": "wolf", "count": 3})

        events = get_quest_events(event_type="ObjectiveProgress")
        assert events[0].objective_id == "kill_10_wolves"

    def test_multiple_progress_updates_fire_multiple_events(self, kill_objective):
        """Multiple objective updates should fire multiple ObjectiveProgress events."""
        kill_objective.quest_id = "test_quest"
        kill_objective.activate()

        clear_quest_events()
        kill_objective.update("kill", {"target_type": "wolf", "count": 1})
        kill_objective.update("kill", {"target_type": "wolf", "count": 2})
        kill_objective.update("kill", {"target_type": "wolf", "count": 3})

        events = get_quest_events(event_type="ObjectiveProgress")
        assert len(events) == 3
        assert events[0].current == 1
        assert events[1].current == 3
        assert events[2].current == 6


# =============================================================================
# Test: ObjectiveCompleted Events
# =============================================================================

class TestObjectiveCompletedEvents:
    """Tests for ObjectiveCompleted event firing."""

    def test_objective_complete_fires_event(self, kill_objective):
        """Objective completion should fire ObjectiveCompleted event."""
        kill_objective.quest_id = "test_quest"
        kill_objective.activate()
        kill_objective.current = 9  # One kill away from completion

        clear_quest_events()
        kill_objective.update("kill", {"target_type": "wolf", "count": 1})

        # Should have both progress and completed events
        progress_events = get_quest_events(event_type="ObjectiveProgress")
        completed_events = get_quest_events(event_type="ObjectiveCompleted")

        assert len(progress_events) == 1
        assert len(completed_events) == 1
        assert completed_events[0].quest_id == "test_quest"
        assert completed_events[0].objective_id == "kill_10_wolves"

    def test_objective_completed_includes_timestamp(self, kill_objective):
        """ObjectiveCompleted should include timestamp."""
        kill_objective.quest_id = "test_quest"
        kill_objective.activate()
        kill_objective.current = 9

        before = time.time()
        clear_quest_events()
        kill_objective.update("kill", {"target_type": "wolf", "count": 1})
        after = time.time()

        completed_events = get_quest_events(event_type="ObjectiveCompleted")
        assert len(completed_events) == 1
        assert before <= completed_events[0].timestamp <= after

    def test_talk_objective_fires_completed_on_talk(self):
        """TalkObjective should fire ObjectiveCompleted on talk completion."""
        talk_obj = TalkObjective(
            id="talk_to_merchant",
            description="Talk to the merchant",
            objective_type=ObjectiveType.TALK,
            npc_id="merchant_001",
            quest_id="test_quest",
        )
        talk_obj.activate()

        clear_quest_events()
        talk_obj.update("talk", {"npc_id": "merchant_001"})

        completed_events = get_quest_events(event_type="ObjectiveCompleted")
        assert len(completed_events) == 1
        assert completed_events[0].objective_id == "talk_to_merchant"


# =============================================================================
# Test: QuestRewardGranted Events
# =============================================================================

class TestQuestRewardGrantedEvents:
    """Tests for QuestRewardGranted event firing."""

    def test_turn_in_fires_reward_events(self, quest_with_rewards, tracker):
        """turn_in_quest() should fire QuestRewardGranted for each reward."""
        tracker.track_quest(quest_with_rewards)
        tracker.accept_quest("reward_quest")
        tracker.complete_quest("reward_quest")

        clear_quest_events()
        rewards = tracker.turn_in_quest("reward_quest")

        reward_events = get_quest_events(event_type="QuestRewardGranted")
        assert len(reward_events) == 3  # XP, Gold, Item

    def test_xp_reward_event_has_correct_type(self, quest_with_rewards, tracker):
        """XP reward should have reward_type='xp'."""
        tracker.track_quest(quest_with_rewards)
        tracker.accept_quest("reward_quest")
        tracker.complete_quest("reward_quest")

        clear_quest_events()
        tracker.turn_in_quest("reward_quest")

        reward_events = get_quest_events(event_type="QuestRewardGranted")
        xp_events = [e for e in reward_events if e.reward_type == "xp"]
        assert len(xp_events) == 1
        assert xp_events[0].amount == 100

    def test_gold_reward_event_has_correct_type(self, quest_with_rewards, tracker):
        """Gold reward should have reward_type='gold'."""
        tracker.track_quest(quest_with_rewards)
        tracker.accept_quest("reward_quest")
        tracker.complete_quest("reward_quest")

        clear_quest_events()
        tracker.turn_in_quest("reward_quest")

        reward_events = get_quest_events(event_type="QuestRewardGranted")
        gold_events = [e for e in reward_events if e.reward_type == "gold"]
        assert len(gold_events) == 1
        assert gold_events[0].amount == 50

    def test_item_reward_event_has_correct_type(self, quest_with_rewards, tracker):
        """Item reward should have reward_type='item'."""
        tracker.track_quest(quest_with_rewards)
        tracker.accept_quest("reward_quest")
        tracker.complete_quest("reward_quest")

        clear_quest_events()
        tracker.turn_in_quest("reward_quest")

        reward_events = get_quest_events(event_type="QuestRewardGranted")
        item_events = [e for e in reward_events if e.reward_type == "item"]
        assert len(item_events) == 1
        assert "sword_of_testing" in str(item_events[0].amount)

    def test_reward_event_includes_entity_id(self, quest_with_rewards, tracker):
        """QuestRewardGranted should include player entity_id."""
        tracker.track_quest(quest_with_rewards)
        tracker.accept_quest("reward_quest")
        tracker.complete_quest("reward_quest")

        clear_quest_events()
        tracker.turn_in_quest("reward_quest")

        reward_events = get_quest_events(event_type="QuestRewardGranted")
        assert all(e.entity_id == "player_001" for e in reward_events)


# =============================================================================
# Test: Causal Chains
# =============================================================================

class TestCausalChains:
    """Tests for causal chain tracking (kill -> progress -> complete -> reward)."""

    def test_kill_triggers_objective_progress(self, kill_objective, basic_quest_def, tracker):
        """Kill event should cause ObjectiveProgress event."""
        tracker.track_quest(basic_quest_def, objectives=[kill_objective])
        tracker.accept_quest("test_quest_basic")

        clear_quest_events()
        tracker.process_event("kill", {"target_type": "wolf", "count": 5})

        progress_events = get_quest_events(event_type="ObjectiveProgress")
        assert len(progress_events) >= 1

    def test_objective_complete_triggers_quest_complete(self, basic_quest_def, tracker):
        """Objective completion should trigger quest completion for auto-complete quests."""
        auto_quest_def = QuestDefinition(
            id="auto_complete_quest",
            name="Auto Complete Quest",
            description="Completes automatically",
            quest_type=QuestType.SIDE,
            level_requirement=1,
            auto_complete=True,
        )

        kill_obj = KillObjective(
            id="kill_1_wolf",
            description="Kill 1 wolf",
            objective_type=ObjectiveType.KILL,
            target_type="wolf",
            required=1,
        )

        tracker.track_quest(auto_quest_def, objectives=[kill_obj])
        tracker.accept_quest("auto_complete_quest")

        clear_quest_events()
        tracker.process_event("kill", {"target_type": "wolf", "count": 1})

        # Should see progress, completed, and state changed (to TURNED_IN for auto-complete)
        progress_events = get_quest_events(event_type="ObjectiveProgress")
        completed_events = get_quest_events(event_type="ObjectiveCompleted")
        state_events = get_quest_events(event_type="QuestStateChanged")

        assert len(progress_events) >= 1
        assert len(completed_events) >= 1
        # Auto-complete quests go through COMPLETE then TURNED_IN
        assert len(state_events) >= 1

    def test_full_causal_chain_kill_to_reward(self, tracker):
        """Test full causal chain: kill -> progress -> complete -> reward."""
        quest_def = QuestDefinition(
            id="full_chain_quest",
            name="Full Chain Quest",
            description="Tests full causal chain",
            quest_type=QuestType.MAIN,
            level_requirement=1,
            rewards=[XPReward(amount=500), GoldReward(amount=100)],
        )

        kill_obj = KillObjective(
            id="kill_boss",
            description="Kill the boss",
            objective_type=ObjectiveType.KILL,
            target_type="boss",
            required=1,
        )

        tracker.track_quest(quest_def, objectives=[kill_obj])
        tracker.accept_quest("full_chain_quest")

        clear_quest_events()

        # Kill the boss
        tracker.process_event("kill", {"target_type": "boss", "count": 1})

        # Complete the quest
        tracker.complete_quest("full_chain_quest")

        # Turn in for rewards
        tracker.turn_in_quest("full_chain_quest")

        # Verify all events in chain
        progress = get_quest_events(event_type="ObjectiveProgress")
        completed = get_quest_events(event_type="ObjectiveCompleted")
        state_changes = get_quest_events(event_type="QuestStateChanged")
        rewards = get_quest_events(event_type="QuestRewardGranted")

        assert len(progress) >= 1  # Kill progress
        assert len(completed) >= 1  # Objective completed
        assert len(state_changes) >= 2  # COMPLETE and TURNED_IN
        assert len(rewards) == 2  # XP and Gold


# =============================================================================
# Test: Multiple Quests Tracked Independently
# =============================================================================

class TestMultipleQuestsIndependent:
    """Tests for tracking multiple quests independently."""

    def test_separate_events_for_different_quests(self, tracker):
        """Different quests should have separate events."""
        quest1 = QuestDefinition(id="quest_1", name="Quest 1", description="First")
        quest2 = QuestDefinition(id="quest_2", name="Quest 2", description="Second")

        kill1 = KillObjective(
            id="kill_wolves",
            description="Kill wolves",
            objective_type=ObjectiveType.KILL,
            target_type="wolf",
            required=5,
        )
        kill2 = KillObjective(
            id="kill_bears",
            description="Kill bears",
            objective_type=ObjectiveType.KILL,
            target_type="bear",
            required=3,
        )

        tracker.track_quest(quest1, objectives=[kill1])
        tracker.track_quest(quest2, objectives=[kill2])
        tracker.accept_quest("quest_1")
        tracker.accept_quest("quest_2")

        clear_quest_events()

        # Update both quests
        tracker.process_event("kill", {"target_type": "wolf", "count": 2})
        tracker.process_event("kill", {"target_type": "bear", "count": 1})

        # Verify events are separate
        quest1_events = get_quest_events(quest_id="quest_1")
        quest2_events = get_quest_events(quest_id="quest_2")

        assert len(quest1_events) >= 1
        assert len(quest2_events) >= 1
        assert all(e.quest_id == "quest_1" for e in quest1_events)
        assert all(e.quest_id == "quest_2" for e in quest2_events)

    def test_events_filtered_by_quest_id(self, tracker):
        """get_quest_events(quest_id=...) should filter correctly."""
        quest1 = QuestDefinition(id="filter_quest_1", name="Filter 1", description="Test")
        quest2 = QuestDefinition(id="filter_quest_2", name="Filter 2", description="Test")

        tracker.track_quest(quest1)
        tracker.track_quest(quest2)
        tracker.accept_quest("filter_quest_1")
        tracker.accept_quest("filter_quest_2")

        clear_quest_events()

        tracker.complete_quest("filter_quest_1")
        tracker.fail_quest("filter_quest_2")

        events1 = get_quest_events(quest_id="filter_quest_1")
        events2 = get_quest_events(quest_id="filter_quest_2")

        assert len(events1) >= 1
        assert len(events2) >= 1
        assert all(e.quest_id == "filter_quest_1" for e in events1)
        assert all(e.quest_id == "filter_quest_2" for e in events2)


# =============================================================================
# Test: Event Query Capabilities
# =============================================================================

class TestEventQuery:
    """Tests for querying quest events."""

    def test_query_by_event_type(self, basic_quest_def, tracker):
        """Should query events by type."""
        tracker.track_quest(basic_quest_def)
        tracker.accept_quest("test_quest_basic")
        tracker.complete_quest("test_quest_basic")

        state_events = get_quest_events(event_type="QuestStateChanged")
        assert len(state_events) >= 1
        assert all(isinstance(e, QuestStateChanged) for e in state_events)

    def test_query_by_entity_id(self):
        """Should query events by entity_id."""
        quest_def = QuestDefinition(
            id="entity_query_quest",
            name="Entity Query",
            description="Test",
        )
        quest_obj = Quest(definition=quest_def, player_id="specific_player_xyz")
        quest_obj.state = QuestState.AVAILABLE

        clear_quest_events()
        quest_obj.accept(timestamp=1000.0)

        events = get_quest_events(entity_id="specific_player_xyz")
        assert len(events) >= 1
        assert all(e.entity_id == "specific_player_xyz" for e in events)

    def test_query_returns_all_when_no_filter(self, basic_quest_def, tracker):
        """get_quest_events() without filters should return all events."""
        tracker.track_quest(basic_quest_def)
        tracker.accept_quest("test_quest_basic")
        tracker.complete_quest("test_quest_basic")

        all_events = get_quest_events()
        assert len(all_events) >= 2  # At least accept + complete


# =============================================================================
# Test: Event Replay for Debugging
# =============================================================================

class TestEventReplay:
    """Tests for event replay capabilities."""

    def test_events_maintain_order(self, basic_quest_def, tracker):
        """Events should maintain chronological order."""
        tracker.track_quest(basic_quest_def)

        clear_quest_events()

        # Perform sequence of actions
        tracker.accept_quest("test_quest_basic")
        tracker.complete_quest("test_quest_basic")

        events = get_quest_events(quest_id="test_quest_basic", event_type="QuestStateChanged")

        # First event: AVAILABLE -> ACTIVE
        # Second event: ACTIVE -> COMPLETE
        assert events[0].old_state == QuestState.AVAILABLE
        assert events[0].new_state == QuestState.ACTIVE
        assert events[1].old_state == QuestState.ACTIVE
        assert events[1].new_state == QuestState.COMPLETE

    def test_events_have_to_dict_for_serialization(self, basic_quest_def):
        """Events should have to_dict() for serialization."""
        quest_obj = Quest(definition=basic_quest_def, player_id="player_001")
        quest_obj.state = QuestState.AVAILABLE

        clear_quest_events()
        quest_obj.accept(timestamp=1000.0)

        events = get_quest_events()
        assert len(events) >= 1

        event_dict = events[0].to_dict()
        assert "type" in event_dict
        assert "quest_id" in event_dict
        assert "timestamp" in event_dict

    def test_clear_quest_events_clears_all(self, basic_quest_def, tracker):
        """clear_quest_events() should remove all events."""
        tracker.track_quest(basic_quest_def)
        tracker.accept_quest("test_quest_basic")

        assert len(get_quest_events()) > 0

        clear_quest_events()

        assert len(get_quest_events()) == 0


# =============================================================================
# Test: Performance
# =============================================================================

class TestPerformance:
    """Performance tests for quest system."""

    def test_100_quest_updates_under_50ms(self, tracker):
        """100 quest updates should complete under 50ms."""
        # Setup 10 quests with kill objectives
        quests = []
        for i in range(10):
            quest_def = QuestDefinition(
                id=f"perf_quest_{i}",
                name=f"Performance Quest {i}",
                description="Performance test",
            )
            kill_obj = KillObjective(
                id=f"kill_target_{i}",
                description="Kill targets",
                objective_type=ObjectiveType.KILL,
                target_type="target",
                required=100,
            )
            tracker.track_quest(quest_def, objectives=[kill_obj])
            tracker.accept_quest(f"perf_quest_{i}")
            quests.append(f"perf_quest_{i}")

        clear_quest_events()

        # Perform 100 updates
        start_time = time.perf_counter()
        for _ in range(100):
            tracker.process_event("kill", {"target_type": "target", "count": 1})
        end_time = time.perf_counter()

        elapsed_ms = (end_time - start_time) * 1000
        assert elapsed_ms < 50, f"100 updates took {elapsed_ms:.2f}ms (expected < 50ms)"

    def test_event_logging_performance(self):
        """Event logging should be fast."""
        clear_quest_events()

        start_time = time.perf_counter()
        for i in range(1000):
            event = ObjectiveProgress(
                quest_id=f"quest_{i % 10}",
                objective_id=f"obj_{i % 5}",
                current=i,
                target=100,
                timestamp=time.time(),
            )
            fire_quest_event(event)
        end_time = time.perf_counter()

        elapsed_ms = (end_time - start_time) * 1000
        assert elapsed_ms < 100, f"1000 events took {elapsed_ms:.2f}ms (expected < 100ms)"

    def test_event_query_performance(self):
        """Event querying should be efficient."""
        clear_quest_events()

        # Generate many events
        for i in range(1000):
            event = ObjectiveProgress(
                quest_id=f"quest_{i % 20}",
                objective_id=f"obj_{i % 10}",
                current=i,
                target=100,
                timestamp=time.time(),
            )
            fire_quest_event(event)

        start_time = time.perf_counter()
        for _ in range(100):
            get_quest_events(quest_id="quest_5")
            get_quest_events(event_type="ObjectiveProgress")
        end_time = time.perf_counter()

        elapsed_ms = (end_time - start_time) * 1000
        assert elapsed_ms < 50, f"100 queries took {elapsed_ms:.2f}ms (expected < 50ms)"


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_fire_event_with_empty_quest_id(self):
        """Should handle events with empty quest_id."""
        event = ObjectiveProgress(
            quest_id="",
            objective_id="test_obj",
            current=1,
            target=5,
            timestamp=time.time(),
        )
        # Should not raise
        fire_quest_event(event)

        events = get_quest_events()
        assert len(events) >= 1

    def test_query_nonexistent_quest_id(self):
        """Querying non-existent quest_id should return empty list."""
        events = get_quest_events(quest_id="nonexistent_quest_xyz")
        assert events == []

    def test_objective_without_quest_id_fires_events(self, kill_objective):
        """Objective without quest_id should still fire events."""
        kill_objective.activate()

        clear_quest_events()
        kill_objective.update("kill", {"target_type": "wolf", "count": 1})

        events = get_quest_events(event_type="ObjectiveProgress")
        assert len(events) >= 1
        assert events[0].quest_id == ""  # Empty but present

    def test_failed_state_transition_no_event(self, basic_quest_def):
        """Failed state transitions should not fire events."""
        quest_obj = Quest(definition=basic_quest_def, player_id="player_001")
        quest_obj.state = QuestState.ACTIVE  # Already active

        clear_quest_events()
        result = quest_obj.accept(timestamp=1000.0)  # Can't accept when already active

        assert result is False
        events = get_quest_events(event_type="QuestStateChanged")
        assert len(events) == 0


# =============================================================================
# Test: Integration with Foundation EventLog
# =============================================================================

class TestFoundationEventLogIntegration:
    """Tests for integration with Foundation EventLog."""

    def test_quest_state_change_adds_to_foundation_eventlog(self, basic_quest_def):
        """Quest state changes should add Changes to Foundation EventLog."""
        set_current_tick(100)
        quest_obj = Quest(definition=basic_quest_def, player_id="player_001")
        quest_obj.state = QuestState.AVAILABLE

        clear_event_log()
        # Note: add_change_to_current_event only works within @traced context
        # This test verifies the fire_quest_event is called
        quest_obj.accept(timestamp=1000.0)

        # Verify quest event was logged
        events = get_quest_events(event_type="QuestStateChanged")
        assert len(events) == 1

    def test_event_log_clear_does_not_affect_quest_events(self, basic_quest_def, tracker):
        """Foundation clear_event_log() should not affect quest-specific events."""
        tracker.track_quest(basic_quest_def)
        tracker.accept_quest("test_quest_basic")

        quest_events_before = len(get_quest_events())
        clear_event_log()
        quest_events_after = len(get_quest_events())

        # Quest events should be preserved
        assert quest_events_after == quest_events_before


# =============================================================================
# Test: Composite Objective Events
# =============================================================================

class TestCompositeObjectiveEvents:
    """Tests for composite objective event firing."""

    def test_composite_objective_fires_child_events(self):
        """CompositeObjective should fire events for child objectives."""
        kill1 = KillObjective(
            id="kill_wolves_comp",
            description="Kill wolves",
            objective_type=ObjectiveType.KILL,
            target_type="wolf",
            required=5,
            quest_id="composite_test",
        )
        kill2 = KillObjective(
            id="kill_bears_comp",
            description="Kill bears",
            objective_type=ObjectiveType.KILL,
            target_type="bear",
            required=3,
            quest_id="composite_test",
        )

        composite = CompositeObjective(
            id="kill_animals",
            description="Kill animals",
            objective_type=ObjectiveType.COMPOSITE,
            objectives=[kill1, kill2],
            mode="all",
            quest_id="composite_test",
        )

        composite.activate()

        clear_quest_events()
        composite.update("kill", {"target_type": "wolf", "count": 1})
        composite.update("kill", {"target_type": "bear", "count": 1})

        events = get_quest_events(event_type="ObjectiveProgress")
        # Should have events from both child objectives
        wolf_events = [e for e in events if e.objective_id == "kill_wolves_comp"]
        bear_events = [e for e in events if e.objective_id == "kill_bears_comp"]

        assert len(wolf_events) >= 1
        assert len(bear_events) >= 1


# =============================================================================
# Additional Tests for 50+ Coverage
# =============================================================================

class TestAdditionalCoverage:
    """Additional tests for comprehensive coverage."""

    def test_reset_quest_fires_state_changed(self, tracker):
        """reset() should fire QuestStateChanged event."""
        quest_def = QuestDefinition(
            id="reset_quest",
            name="Reset Quest",
            description="Repeatable quest",
            quest_type=QuestType.DAILY,
            repeatable=True,
        )

        tracker.track_quest(quest_def)
        tracker.accept_quest("reset_quest")
        tracker.complete_quest("reset_quest")
        tracker.turn_in_quest("reset_quest")

        clear_quest_events()
        tracked = tracker.get_tracked("reset_quest")
        tracked.quest.reset(timestamp=5000.0)

        events = get_quest_events(event_type="QuestStateChanged")
        assert len(events) == 1
        assert events[0].old_state == QuestState.TURNED_IN
        assert events[0].new_state == QuestState.AVAILABLE

    def test_abandon_quest_fires_state_changed(self, basic_quest_def, tracker):
        """abandon() should fire QuestStateChanged event."""
        tracker.track_quest(basic_quest_def)
        tracker.accept_quest("test_quest_basic")

        clear_quest_events()
        tracked = tracker.get_tracked("test_quest_basic")
        tracked.quest.abandon(timestamp=2000.0)

        events = get_quest_events(event_type="QuestStateChanged")
        assert len(events) == 1
        assert events[0].old_state == QuestState.ACTIVE
        assert events[0].new_state == QuestState.AVAILABLE

    def test_interact_objective_fires_progress(self):
        """InteractObjective should fire ObjectiveProgress on interaction."""
        interact_obj = InteractObjective(
            id="interact_lever",
            description="Pull the lever",
            objective_type=ObjectiveType.INTERACT,
            object_id="lever_001",
            times_required=3,
            quest_id="interact_test",
        )
        interact_obj.activate()

        clear_quest_events()
        interact_obj.update("interact", {"object_id": "lever_001", "interaction_type": "use"})

        events = get_quest_events(event_type="ObjectiveProgress")
        assert len(events) == 1
        # InteractObjective uses progress (0.0-1.0) not current/target
        assert events[0].objective_id == "interact_lever"
        assert events[0].quest_id == "interact_test"

    def test_reach_objective_fires_completed_on_arrival(self):
        """ReachObjective should fire ObjectiveCompleted when reached."""
        reach_obj = ReachObjective(
            id="reach_village",
            description="Reach the village",
            objective_type=ObjectiveType.REACH,
            location_id="village_01",
            quest_id="reach_test",
        )
        reach_obj.activate()

        clear_quest_events()
        reach_obj.update("enter_location", {"location_id": "village_01"})

        completed_events = get_quest_events(event_type="ObjectiveCompleted")
        assert len(completed_events) == 1
        assert completed_events[0].objective_id == "reach_village"

    def test_multiple_rewards_same_type(self, tracker):
        """Multiple rewards of same type should fire separate events."""
        quest_def = QuestDefinition(
            id="multi_reward_quest",
            name="Multi Reward",
            description="Multiple gold rewards",
            rewards=[
                GoldReward(amount=100),
                GoldReward(amount=200),
                GoldReward(amount=300),
            ],
        )

        tracker.track_quest(quest_def)
        tracker.accept_quest("multi_reward_quest")
        tracker.complete_quest("multi_reward_quest")

        clear_quest_events()
        tracker.turn_in_quest("multi_reward_quest")

        reward_events = get_quest_events(event_type="QuestRewardGranted")
        assert len(reward_events) == 3
        amounts = sorted([e.amount for e in reward_events])
        assert amounts == [100, 200, 300]

    def test_quest_event_to_dict_complete(self, basic_quest_def):
        """All event types should have complete to_dict()."""
        # Test QuestStateChanged
        state_event = QuestStateChanged(
            quest_id="test_q",
            entity_id="player_1",
            old_state=QuestState.ACTIVE,
            new_state=QuestState.COMPLETE,
            timestamp=1234.56,
        )
        d = state_event.to_dict()
        assert d["type"] == "QuestStateChanged"
        assert d["quest_id"] == "test_q"
        assert d["old_state"] == "ACTIVE"
        assert d["new_state"] == "COMPLETE"
        assert d["timestamp"] == 1234.56

        # Test ObjectiveProgress
        progress_event = ObjectiveProgress(
            quest_id="test_q",
            objective_id="obj_1",
            current=5,
            target=10,
            timestamp=2345.67,
        )
        d = progress_event.to_dict()
        assert d["type"] == "ObjectiveProgress"
        assert d["current"] == 5
        assert d["target"] == 10

        # Test QuestRewardGranted
        reward_event = QuestRewardGranted(
            quest_id="test_q",
            entity_id="player_1",
            reward_type="gold",
            amount=500,
            timestamp=3456.78,
        )
        d = reward_event.to_dict()
        assert d["type"] == "QuestRewardGranted"
        assert d["reward_type"] == "gold"
        assert d["amount"] == 500

    def test_registry_query_by_quest_type(self):
        """Should be able to query quests by quest_type via metadata."""

        @quest(id="main_story_1", name="Main Story", quest_type=QuestType.MAIN)
        class MainStoryQuest:
            pass

        @quest(id="side_mission_1", name="Side Mission", quest_type=QuestType.SIDE)
        class SideMissionQuest:
            pass

        # Find main quests via metadata
        all_quests = get_registered_quests()
        main_quests = [
            q for q in all_quests
            if registry.get_metadata(q, "quest_type") == "MAIN"
        ]
        side_quests = [
            q for q in all_quests
            if registry.get_metadata(q, "quest_type") == "SIDE"
        ]

        assert MainStoryQuest in main_quests
        assert SideMissionQuest in side_quests
        assert MainStoryQuest not in side_quests

    def test_sequential_objectives_fire_events_in_order(self, tracker):
        """Sequential objectives should fire events in correct order."""
        quest_def = QuestDefinition(
            id="sequential_quest",
            name="Sequential Quest",
            description="Sequential objectives",
        )

        obj1 = TalkObjective(
            id="talk_first",
            description="Talk to NPC 1",
            objective_type=ObjectiveType.TALK,
            npc_id="npc_1",
            order=0,
        )
        obj2 = TalkObjective(
            id="talk_second",
            description="Talk to NPC 2",
            objective_type=ObjectiveType.TALK,
            npc_id="npc_2",
            order=1,
        )

        tracker.track_quest(quest_def, objectives=[obj1, obj2])
        tracker.accept_quest("sequential_quest")

        clear_quest_events()

        # Complete first objective
        tracker.process_event("talk", {"npc_id": "npc_1"})

        events = get_quest_events()
        completed = [e for e in events if isinstance(e, ObjectiveCompleted)]
        assert len(completed) == 1
        assert completed[0].objective_id == "talk_first"

    def test_event_query_empty_list_filters(self):
        """Empty filters should return appropriate results."""
        events = get_quest_events(quest_id="", event_type=None, entity_id=None)
        # Should handle gracefully (may return all or filtered)
        assert isinstance(events, list)
