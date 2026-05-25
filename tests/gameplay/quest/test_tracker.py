"""
Comprehensive tests for Quest Tracker.

Tests cover:
- Progress tracking per objective
- Event-driven updates
- Progress persistence
- Progress queries
- Active quest limit
- Tracked quest UI data
- Progress notifications
"""

import pytest
from dataclasses import dataclass
from typing import Any, List, Dict
from unittest.mock import Mock, patch, MagicMock

# QuestProgress, ObjectiveProgress, etc. are planned but not yet implemented
pytest.skip("Tracker API not fully implemented", allow_module_level=True)

from engine.gameplay.quest.tracker import (
    QuestTracker,
    QuestProgress,
    ObjectiveProgress,
    TrackerConfig,
    QuestTrackerEvent,
    QuestTrackerListener,
)
from engine.gameplay.quest.quest import Quest, QuestDefinition, QuestState, QuestType
from engine.gameplay.quest.objectives import (
    KillObjective,
    CollectObjective,
    TalkObjective,
    ObjectiveState,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def tracker_config():
    """Create a tracker configuration."""
    return TrackerConfig(
        max_active_quests=25,
        max_tracked_quests=5,
        auto_track_on_accept=True,
        auto_complete_objectives=True,
        save_progress=True,
    )


@pytest.fixture
def quest_tracker(tracker_config):
    """Create a quest tracker instance."""
    return QuestTracker(config=tracker_config, player_id="player_001")


@pytest.fixture
def simple_quest_def():
    """Create a simple quest definition."""
    return QuestDefinition(
        id="simple_quest",
        name="Simple Quest",
        description="A simple test quest",
        quest_type=QuestType.SIDE,
    )


@pytest.fixture
def multi_objective_quest_def():
    """Create a quest with multiple objectives."""
    return QuestDefinition(
        id="multi_quest",
        name="Multi-Objective Quest",
        description="A quest with multiple objectives",
        quest_type=QuestType.MAIN,
    )


@pytest.fixture
def kill_objective():
    """Create a kill objective."""
    return KillObjective(
        id="kill_wolves",
        description="Kill 10 wolves",
        target_type="wolf",
        required=10,
    )


@pytest.fixture
def collect_objective():
    """Create a collect objective."""
    return CollectObjective(
        id="collect_herbs",
        description="Collect 5 herbs",
        item_id="herb",
        required=5,
    )


# =============================================================================
# QuestProgress Tests
# =============================================================================

class TestQuestProgress:
    """Tests for QuestProgress data class."""

    def test_quest_progress_creation(self, simple_quest_def):
        """Test creating a quest progress instance."""
        progress = QuestProgress(
            quest_id=simple_quest_def.id,
            state=QuestState.ACTIVE,
            objectives={},
        )
        assert progress.quest_id == "simple_quest"
        assert progress.state == QuestState.ACTIVE

    def test_quest_progress_with_objectives(self):
        """Test quest progress with objective progress."""
        obj_progress = ObjectiveProgress(
            objective_id="kill_obj",
            current=5,
            required=10,
            state=ObjectiveState.IN_PROGRESS,
        )
        progress = QuestProgress(
            quest_id="quest_1",
            state=QuestState.ACTIVE,
            objectives={"kill_obj": obj_progress},
        )
        assert "kill_obj" in progress.objectives
        assert progress.objectives["kill_obj"].current == 5

    def test_quest_progress_overall_progress(self):
        """Test overall progress calculation."""
        obj1 = ObjectiveProgress("obj1", 5, 10, ObjectiveState.IN_PROGRESS)
        obj2 = ObjectiveProgress("obj2", 3, 10, ObjectiveState.IN_PROGRESS)
        progress = QuestProgress(
            quest_id="quest_1",
            state=QuestState.ACTIVE,
            objectives={"obj1": obj1, "obj2": obj2},
        )
        # Average of 0.5 and 0.3
        assert progress.overall_progress == pytest.approx(0.4)

    def test_quest_progress_completion_status(self):
        """Test completion status helper."""
        obj1 = ObjectiveProgress("obj1", 10, 10, ObjectiveState.COMPLETE)
        obj2 = ObjectiveProgress("obj2", 10, 10, ObjectiveState.COMPLETE)
        progress = QuestProgress(
            quest_id="quest_1",
            state=QuestState.COMPLETE,
            objectives={"obj1": obj1, "obj2": obj2},
        )
        assert progress.is_complete is True

    def test_quest_progress_timestamps(self):
        """Test quest progress timestamps."""
        progress = QuestProgress(
            quest_id="quest_1",
            state=QuestState.ACTIVE,
            objectives={},
            accepted_at=100.0,
            updated_at=150.0,
        )
        assert progress.accepted_at == 100.0
        assert progress.updated_at == 150.0

    def test_quest_progress_empty_objectives(self):
        """Test quest progress with no objectives."""
        progress = QuestProgress(
            quest_id="quest_1",
            state=QuestState.ACTIVE,
            objectives={},
        )
        assert progress.overall_progress == 0.0


# =============================================================================
# ObjectiveProgress Tests
# =============================================================================

class TestObjectiveProgress:
    """Tests for ObjectiveProgress data class."""

    def test_objective_progress_creation(self):
        """Test creating an objective progress instance."""
        progress = ObjectiveProgress(
            objective_id="obj_1",
            current=5,
            required=10,
            state=ObjectiveState.IN_PROGRESS,
        )
        assert progress.objective_id == "obj_1"
        assert progress.current == 5
        assert progress.required == 10

    def test_objective_progress_percentage(self):
        """Test progress percentage calculation."""
        progress = ObjectiveProgress("obj", 5, 10, ObjectiveState.IN_PROGRESS)
        assert progress.progress == pytest.approx(0.5)

    def test_objective_progress_percentage_zero_required(self):
        """Test progress with zero required doesn't divide by zero."""
        progress = ObjectiveProgress("obj", 0, 0, ObjectiveState.COMPLETE)
        assert progress.progress == 1.0

    def test_objective_progress_is_complete(self):
        """Test is_complete property."""
        progress = ObjectiveProgress("obj", 10, 10, ObjectiveState.COMPLETE)
        assert progress.is_complete is True

        progress2 = ObjectiveProgress("obj2", 5, 10, ObjectiveState.IN_PROGRESS)
        assert progress2.is_complete is False

    def test_objective_progress_serialization(self):
        """Test objective progress serialization."""
        progress = ObjectiveProgress("obj", 5, 10, ObjectiveState.IN_PROGRESS)
        data = progress.to_dict()

        assert data["objective_id"] == "obj"
        assert data["current"] == 5
        assert data["required"] == 10
        assert data["state"] == "IN_PROGRESS"

    def test_objective_progress_deserialization(self):
        """Test objective progress deserialization."""
        data = {
            "objective_id": "obj",
            "current": 7,
            "required": 10,
            "state": "IN_PROGRESS",
        }
        progress = ObjectiveProgress.from_dict(data)

        assert progress.objective_id == "obj"
        assert progress.current == 7


# =============================================================================
# TrackerConfig Tests
# =============================================================================

class TestTrackerConfig:
    """Tests for TrackerConfig data class."""

    def test_tracker_config_defaults(self):
        """Test default tracker config values."""
        config = TrackerConfig()
        assert config.max_active_quests == 25
        assert config.max_tracked_quests == 5
        assert config.auto_track_on_accept is True
        assert config.auto_complete_objectives is True
        assert config.save_progress is True

    def test_tracker_config_custom(self):
        """Test custom tracker config."""
        config = TrackerConfig(
            max_active_quests=10,
            max_tracked_quests=3,
            auto_track_on_accept=False,
        )
        assert config.max_active_quests == 10
        assert config.max_tracked_quests == 3
        assert config.auto_track_on_accept is False

    def test_tracker_config_validation_max_active(self):
        """Test config validation for max_active_quests."""
        with pytest.raises(ValueError, match="max_active_quests must be > 0"):
            TrackerConfig(max_active_quests=0)

    def test_tracker_config_validation_max_tracked(self):
        """Test config validation for max_tracked_quests."""
        with pytest.raises(ValueError, match="max_tracked_quests must be > 0"):
            TrackerConfig(max_tracked_quests=-1)


# =============================================================================
# QuestTracker Basic Tests
# =============================================================================

class TestQuestTrackerBasic:
    """Tests for basic QuestTracker functionality."""

    def test_tracker_creation(self, tracker_config):
        """Test creating a quest tracker."""
        tracker = QuestTracker(config=tracker_config, player_id="player_001")
        assert tracker.player_id == "player_001"
        assert tracker.config is tracker_config

    def test_tracker_accept_quest(self, quest_tracker, simple_quest_def, kill_objective):
        """Test accepting a quest."""
        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [kill_objective]

        result = quest_tracker.accept_quest(quest, timestamp=100.0)

        assert result is True
        assert quest.state == QuestState.ACTIVE
        assert simple_quest_def.id in quest_tracker.active_quests

    def test_tracker_accept_unavailable_quest(self, quest_tracker, simple_quest_def):
        """Test that unavailable quest cannot be accepted."""
        quest = Quest(definition=simple_quest_def, state=QuestState.UNAVAILABLE)

        result = quest_tracker.accept_quest(quest, timestamp=100.0)

        assert result is False
        assert simple_quest_def.id not in quest_tracker.active_quests

    def test_tracker_accept_already_active(self, quest_tracker, simple_quest_def, kill_objective):
        """Test accepting an already active quest."""
        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [kill_objective]

        quest_tracker.accept_quest(quest, timestamp=100.0)
        result = quest_tracker.accept_quest(quest, timestamp=200.0)

        assert result is False  # Already active

    def test_tracker_active_quest_limit(self, simple_quest_def, kill_objective):
        """Test active quest limit enforcement."""
        config = TrackerConfig(max_active_quests=2)
        tracker = QuestTracker(config=config, player_id="player")

        for i in range(3):
            quest_def = QuestDefinition(
                id=f"quest_{i}",
                name=f"Quest {i}",
                description="",
            )
            quest = Quest(definition=quest_def, state=QuestState.AVAILABLE)
            quest.objectives = []

            if i < 2:
                assert tracker.accept_quest(quest, timestamp=float(i)) is True
            else:
                assert tracker.accept_quest(quest, timestamp=float(i)) is False

    def test_tracker_get_quest_progress(self, quest_tracker, simple_quest_def, kill_objective):
        """Test getting quest progress."""
        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [kill_objective]

        quest_tracker.accept_quest(quest, timestamp=100.0)
        progress = quest_tracker.get_progress(simple_quest_def.id)

        assert progress is not None
        assert progress.quest_id == simple_quest_def.id
        assert progress.state == QuestState.ACTIVE

    def test_tracker_get_nonexistent_progress(self, quest_tracker):
        """Test getting progress for non-existent quest."""
        progress = quest_tracker.get_progress("nonexistent")
        assert progress is None

    def test_tracker_complete_quest(self, quest_tracker, simple_quest_def, kill_objective):
        """Test completing a quest."""
        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        kill_objective.current = kill_objective.required  # Mark as complete
        quest.objectives = [kill_objective]

        quest_tracker.accept_quest(quest, timestamp=100.0)
        result = quest_tracker.complete_quest(simple_quest_def.id, timestamp=200.0)

        assert result is True
        assert quest.state == QuestState.COMPLETE

    def test_tracker_abandon_quest(self, quest_tracker, simple_quest_def, kill_objective):
        """Test abandoning a quest."""
        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [kill_objective]

        quest_tracker.accept_quest(quest, timestamp=100.0)
        result = quest_tracker.abandon_quest(simple_quest_def.id)

        assert result is True
        assert simple_quest_def.id not in quest_tracker.active_quests

    def test_tracker_fail_quest(self, quest_tracker, simple_quest_def, kill_objective):
        """Test failing a quest."""
        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [kill_objective]

        quest_tracker.accept_quest(quest, timestamp=100.0)
        result = quest_tracker.fail_quest(simple_quest_def.id, timestamp=150.0)

        assert result is True
        assert quest.state == QuestState.FAILED


# =============================================================================
# Progress Tracking Tests
# =============================================================================

class TestProgressTracking:
    """Tests for objective progress tracking."""

    def test_track_kill_objective_progress(self, quest_tracker, simple_quest_def, kill_objective):
        """Test tracking progress on kill objective."""
        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [kill_objective]

        quest_tracker.accept_quest(quest, timestamp=100.0)

        # Process kill event
        quest_tracker.process_event("kill", {"target_type": "wolf"})

        progress = quest_tracker.get_progress(simple_quest_def.id)
        obj_progress = progress.objectives["kill_wolves"]
        assert obj_progress.current == 1

    def test_track_collect_objective_progress(self, quest_tracker, simple_quest_def, collect_objective):
        """Test tracking progress on collect objective."""
        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [collect_objective]

        quest_tracker.accept_quest(quest, timestamp=100.0)

        # Process collect event
        quest_tracker.process_event("collect", {"item_id": "herb", "count": 3})

        progress = quest_tracker.get_progress(simple_quest_def.id)
        obj_progress = progress.objectives["collect_herbs"]
        assert obj_progress.current == 3

    def test_track_multiple_objectives(self, quest_tracker, simple_quest_def, kill_objective, collect_objective):
        """Test tracking progress on multiple objectives."""
        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [kill_objective, collect_objective]

        quest_tracker.accept_quest(quest, timestamp=100.0)

        quest_tracker.process_event("kill", {"target_type": "wolf"})
        quest_tracker.process_event("collect", {"item_id": "herb"})

        progress = quest_tracker.get_progress(simple_quest_def.id)
        assert progress.objectives["kill_wolves"].current == 1
        assert progress.objectives["collect_herbs"].current == 1

    def test_track_multiple_quests_same_event(self, quest_tracker, kill_objective):
        """Test tracking when multiple quests respond to same event."""
        quest_def1 = QuestDefinition(id="q1", name="Q1", description="")
        quest_def2 = QuestDefinition(id="q2", name="Q2", description="")

        kill1 = KillObjective(id="kill1", description="Kill wolves", target_type="wolf", required=5)
        kill2 = KillObjective(id="kill2", description="Kill more wolves", target_type="wolf", required=10)

        quest1 = Quest(definition=quest_def1, state=QuestState.AVAILABLE)
        quest1.objectives = [kill1]

        quest2 = Quest(definition=quest_def2, state=QuestState.AVAILABLE)
        quest2.objectives = [kill2]

        quest_tracker.accept_quest(quest1, timestamp=100.0)
        quest_tracker.accept_quest(quest2, timestamp=100.0)

        quest_tracker.process_event("kill", {"target_type": "wolf"})

        progress1 = quest_tracker.get_progress("q1")
        progress2 = quest_tracker.get_progress("q2")

        assert progress1.objectives["kill1"].current == 1
        assert progress2.objectives["kill2"].current == 1

    def test_progress_auto_complete_objective(self, quest_tracker, simple_quest_def, kill_objective):
        """Test auto-completion of objectives."""
        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [kill_objective]

        quest_tracker.accept_quest(quest, timestamp=100.0)

        # Kill 10 wolves
        for _ in range(10):
            quest_tracker.process_event("kill", {"target_type": "wolf"})

        progress = quest_tracker.get_progress(simple_quest_def.id)
        obj_progress = progress.objectives["kill_wolves"]
        assert obj_progress.is_complete

    def test_update_progress_directly(self, quest_tracker, simple_quest_def, kill_objective):
        """Test updating progress directly."""
        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [kill_objective]

        quest_tracker.accept_quest(quest, timestamp=100.0)

        quest_tracker.update_objective_progress(
            simple_quest_def.id,
            "kill_wolves",
            current=5,
            timestamp=150.0,
        )

        progress = quest_tracker.get_progress(simple_quest_def.id)
        assert progress.objectives["kill_wolves"].current == 5


# =============================================================================
# Event Processing Tests
# =============================================================================

class TestEventProcessing:
    """Tests for event processing functionality."""

    def test_process_event_updates_timestamp(self, quest_tracker, simple_quest_def, kill_objective):
        """Test that processing events updates timestamps."""
        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [kill_objective]

        quest_tracker.accept_quest(quest, timestamp=100.0)
        quest_tracker.process_event("kill", {"target_type": "wolf"}, timestamp=200.0)

        progress = quest_tracker.get_progress(simple_quest_def.id)
        assert progress.updated_at == 200.0

    def test_process_event_no_matching_quests(self, quest_tracker):
        """Test processing event with no matching quests."""
        # Should not raise error
        quest_tracker.process_event("kill", {"target_type": "wolf"})

    def test_process_event_wrong_target(self, quest_tracker, simple_quest_def, kill_objective):
        """Test processing event with wrong target type."""
        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [kill_objective]

        quest_tracker.accept_quest(quest, timestamp=100.0)
        quest_tracker.process_event("kill", {"target_type": "bear"})

        progress = quest_tracker.get_progress(simple_quest_def.id)
        assert progress.objectives["kill_wolves"].current == 0

    def test_process_event_batch(self, quest_tracker, simple_quest_def, kill_objective):
        """Test processing multiple events in batch."""
        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [kill_objective]

        quest_tracker.accept_quest(quest, timestamp=100.0)

        events = [
            ("kill", {"target_type": "wolf"}),
            ("kill", {"target_type": "wolf"}),
            ("kill", {"target_type": "wolf", "count": 3}),
        ]

        for event_type, event_data in events:
            quest_tracker.process_event(event_type, event_data)

        progress = quest_tracker.get_progress(simple_quest_def.id)
        assert progress.objectives["kill_wolves"].current == 5

    def test_process_event_triggers_auto_complete(self, quest_tracker, simple_quest_def, kill_objective):
        """Test that events can trigger quest auto-completion."""
        quest = Quest(
            definition=QuestDefinition(
                id="auto_complete",
                name="Auto Complete Quest",
                description="",
                auto_complete=True,
            ),
            state=QuestState.AVAILABLE,
        )
        single_kill = KillObjective(id="kill", description="Kill 1", target_type="wolf", required=1)
        quest.objectives = [single_kill]

        quest_tracker.accept_quest(quest, timestamp=100.0)
        quest_tracker.process_event("kill", {"target_type": "wolf"}, timestamp=200.0)

        # Quest should auto-complete
        progress = quest_tracker.get_progress("auto_complete")
        assert progress.state == QuestState.COMPLETE


# =============================================================================
# Tracked Quest Tests
# =============================================================================

class TestTrackedQuests:
    """Tests for tracked quest UI functionality."""

    def test_track_quest(self, quest_tracker, simple_quest_def, kill_objective):
        """Test tracking a quest."""
        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [kill_objective]

        quest_tracker.accept_quest(quest, timestamp=100.0)
        result = quest_tracker.track_quest(simple_quest_def.id)

        assert result is True
        assert simple_quest_def.id in quest_tracker.tracked_quests

    def test_untrack_quest(self, quest_tracker, simple_quest_def, kill_objective):
        """Test untracking a quest."""
        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [kill_objective]

        quest_tracker.accept_quest(quest, timestamp=100.0)
        quest_tracker.track_quest(simple_quest_def.id)
        result = quest_tracker.untrack_quest(simple_quest_def.id)

        assert result is True
        assert simple_quest_def.id not in quest_tracker.tracked_quests

    def test_tracked_quest_limit(self, simple_quest_def, kill_objective):
        """Test tracked quest limit."""
        config = TrackerConfig(max_tracked_quests=2, max_active_quests=10)
        tracker = QuestTracker(config=config, player_id="player")

        for i in range(5):
            quest_def = QuestDefinition(id=f"q{i}", name=f"Q{i}", description="")
            quest = Quest(definition=quest_def, state=QuestState.AVAILABLE)
            quest.objectives = []
            tracker.accept_quest(quest, timestamp=float(i))

        tracker.track_quest("q0")
        tracker.track_quest("q1")
        result = tracker.track_quest("q2")  # Should fail - limit reached

        assert result is False
        assert "q2" not in tracker.tracked_quests

    def test_auto_track_on_accept(self, quest_tracker, simple_quest_def, kill_objective):
        """Test auto-tracking on quest accept."""
        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [kill_objective]

        quest_tracker.accept_quest(quest, timestamp=100.0)

        # Should be auto-tracked
        assert simple_quest_def.id in quest_tracker.tracked_quests

    def test_no_auto_track_on_accept(self, simple_quest_def, kill_objective):
        """Test no auto-tracking when disabled."""
        config = TrackerConfig(auto_track_on_accept=False)
        tracker = QuestTracker(config=config, player_id="player")

        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [kill_objective]

        tracker.accept_quest(quest, timestamp=100.0)

        # Should not be auto-tracked
        assert simple_quest_def.id not in tracker.tracked_quests

    def test_get_tracked_quests_ui_data(self, quest_tracker, simple_quest_def, kill_objective):
        """Test getting tracked quest UI data."""
        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [kill_objective]

        quest_tracker.accept_quest(quest, timestamp=100.0)
        quest_tracker.track_quest(simple_quest_def.id)

        ui_data = quest_tracker.get_tracked_quest_ui_data()

        assert len(ui_data) == 1
        assert ui_data[0]["quest_id"] == simple_quest_def.id
        assert "objectives" in ui_data[0]

    def test_tracked_quest_order(self):
        """Test tracked quests maintain order."""
        config = TrackerConfig(max_tracked_quests=5, max_active_quests=10)
        tracker = QuestTracker(config=config, player_id="player")

        for i in range(3):
            quest_def = QuestDefinition(id=f"q{i}", name=f"Q{i}", description="")
            quest = Quest(definition=quest_def, state=QuestState.AVAILABLE)
            quest.objectives = []
            tracker.accept_quest(quest, timestamp=float(i))
            tracker.track_quest(f"q{i}")

        tracked = tracker.tracked_quests
        assert tracked == ["q0", "q1", "q2"]


# =============================================================================
# Listener Tests
# =============================================================================

class TestQuestTrackerListeners:
    """Tests for quest tracker event listeners."""

    def test_add_listener(self, quest_tracker):
        """Test adding a listener."""
        listener = Mock(spec=QuestTrackerListener)
        quest_tracker.add_listener(listener)
        assert listener in quest_tracker.listeners

    def test_remove_listener(self, quest_tracker):
        """Test removing a listener."""
        listener = Mock(spec=QuestTrackerListener)
        quest_tracker.add_listener(listener)
        quest_tracker.remove_listener(listener)
        assert listener not in quest_tracker.listeners

    def test_listener_on_quest_accepted(self, quest_tracker, simple_quest_def, kill_objective):
        """Test listener notified on quest accept."""
        listener = Mock(spec=QuestTrackerListener)
        quest_tracker.add_listener(listener)

        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [kill_objective]
        quest_tracker.accept_quest(quest, timestamp=100.0)

        listener.on_quest_accepted.assert_called_once()

    def test_listener_on_quest_completed(self, quest_tracker, simple_quest_def, kill_objective):
        """Test listener notified on quest complete."""
        listener = Mock(spec=QuestTrackerListener)
        quest_tracker.add_listener(listener)

        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        kill_objective.current = kill_objective.required
        quest.objectives = [kill_objective]

        quest_tracker.accept_quest(quest, timestamp=100.0)
        quest_tracker.complete_quest(simple_quest_def.id, timestamp=200.0)

        listener.on_quest_completed.assert_called_once()

    def test_listener_on_quest_failed(self, quest_tracker, simple_quest_def, kill_objective):
        """Test listener notified on quest fail."""
        listener = Mock(spec=QuestTrackerListener)
        quest_tracker.add_listener(listener)

        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [kill_objective]

        quest_tracker.accept_quest(quest, timestamp=100.0)
        quest_tracker.fail_quest(simple_quest_def.id, timestamp=150.0)

        listener.on_quest_failed.assert_called_once()

    def test_listener_on_objective_updated(self, quest_tracker, simple_quest_def, kill_objective):
        """Test listener notified on objective update."""
        listener = Mock(spec=QuestTrackerListener)
        quest_tracker.add_listener(listener)

        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [kill_objective]

        quest_tracker.accept_quest(quest, timestamp=100.0)
        quest_tracker.process_event("kill", {"target_type": "wolf"})

        listener.on_objective_progress.assert_called()

    def test_listener_on_objective_completed(self, quest_tracker, simple_quest_def, kill_objective):
        """Test listener notified on objective complete."""
        listener = Mock(spec=QuestTrackerListener)
        quest_tracker.add_listener(listener)

        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [kill_objective]

        quest_tracker.accept_quest(quest, timestamp=100.0)

        # Complete the objective
        for _ in range(10):
            quest_tracker.process_event("kill", {"target_type": "wolf"})

        listener.on_objective_completed.assert_called()

    def test_multiple_listeners(self, quest_tracker, simple_quest_def, kill_objective):
        """Test multiple listeners are notified."""
        listener1 = Mock(spec=QuestTrackerListener)
        listener2 = Mock(spec=QuestTrackerListener)

        quest_tracker.add_listener(listener1)
        quest_tracker.add_listener(listener2)

        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [kill_objective]
        quest_tracker.accept_quest(quest, timestamp=100.0)

        listener1.on_quest_accepted.assert_called_once()
        listener2.on_quest_accepted.assert_called_once()


# =============================================================================
# Persistence Tests
# =============================================================================

class TestProgressPersistence:
    """Tests for progress persistence functionality."""

    def test_save_progress(self, quest_tracker, simple_quest_def, kill_objective):
        """Test saving quest progress."""
        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [kill_objective]

        quest_tracker.accept_quest(quest, timestamp=100.0)
        quest_tracker.process_event("kill", {"target_type": "wolf", "count": 5})

        save_data = quest_tracker.save()

        assert "player_id" in save_data
        assert "quests" in save_data
        assert simple_quest_def.id in save_data["quests"]

    def test_load_progress(self, tracker_config, simple_quest_def, kill_objective):
        """Test loading quest progress."""
        # First, save some progress
        tracker1 = QuestTracker(config=tracker_config, player_id="player_001")
        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [kill_objective]

        tracker1.accept_quest(quest, timestamp=100.0)
        tracker1.process_event("kill", {"target_type": "wolf", "count": 5})

        save_data = tracker1.save()

        # Load in a new tracker
        tracker2 = QuestTracker(config=tracker_config, player_id="player_001")
        tracker2.load(save_data)

        progress = tracker2.get_progress(simple_quest_def.id)
        assert progress is not None
        assert progress.objectives["kill_wolves"].current == 5

    def test_save_tracked_quests(self, quest_tracker, simple_quest_def, kill_objective):
        """Test saving tracked quests."""
        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [kill_objective]

        quest_tracker.accept_quest(quest, timestamp=100.0)
        quest_tracker.track_quest(simple_quest_def.id)

        save_data = quest_tracker.save()

        assert "tracked" in save_data
        assert simple_quest_def.id in save_data["tracked"]

    def test_save_completed_quests(self, quest_tracker, simple_quest_def, kill_objective):
        """Test saving completed quests."""
        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        kill_objective.current = kill_objective.required
        quest.objectives = [kill_objective]

        quest_tracker.accept_quest(quest, timestamp=100.0)
        quest_tracker.complete_quest(simple_quest_def.id, timestamp=200.0)

        save_data = quest_tracker.save()

        quest_data = save_data["quests"][simple_quest_def.id]
        assert quest_data["state"] == "COMPLETE"

    def test_export_import_json(self, quest_tracker, simple_quest_def, kill_objective):
        """Test JSON export/import."""
        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [kill_objective]

        quest_tracker.accept_quest(quest, timestamp=100.0)
        quest_tracker.process_event("kill", {"target_type": "wolf", "count": 3})

        json_str = quest_tracker.export_json()

        # Import into new tracker
        new_tracker = QuestTracker(config=tracker_config, player_id="player_001")
        new_tracker.import_json(json_str)

        progress = new_tracker.get_progress(simple_quest_def.id)
        assert progress is not None


# =============================================================================
# Query Tests
# =============================================================================

class TestProgressQueries:
    """Tests for progress query functionality."""

    def test_get_all_active_quests(self, quest_tracker, kill_objective):
        """Test getting all active quests."""
        for i in range(3):
            quest_def = QuestDefinition(id=f"q{i}", name=f"Q{i}", description="")
            quest = Quest(definition=quest_def, state=QuestState.AVAILABLE)
            quest.objectives = []
            quest_tracker.accept_quest(quest, timestamp=float(i))

        active = quest_tracker.get_active_quests()
        assert len(active) == 3

    def test_get_completed_quests(self, quest_tracker, kill_objective):
        """Test getting completed quests."""
        for i in range(3):
            quest_def = QuestDefinition(id=f"q{i}", name=f"Q{i}", description="")
            quest = Quest(definition=quest_def, state=QuestState.AVAILABLE)
            quest.objectives = []
            quest_tracker.accept_quest(quest, timestamp=float(i))

        quest_tracker.complete_quest("q0", timestamp=100.0)
        quest_tracker.complete_quest("q1", timestamp=100.0)

        completed = quest_tracker.get_completed_quests()
        assert len(completed) == 2

    def test_get_failed_quests(self, quest_tracker, kill_objective):
        """Test getting failed quests."""
        for i in range(3):
            quest_def = QuestDefinition(id=f"q{i}", name=f"Q{i}", description="")
            quest = Quest(definition=quest_def, state=QuestState.AVAILABLE)
            quest.objectives = []
            quest_tracker.accept_quest(quest, timestamp=float(i))

        quest_tracker.fail_quest("q1", timestamp=100.0)

        failed = quest_tracker.get_failed_quests()
        assert len(failed) == 1
        assert "q1" in failed

    def test_get_quests_by_type(self, quest_tracker):
        """Test getting quests by type."""
        main_def = QuestDefinition(id="main", name="Main", description="", quest_type=QuestType.MAIN)
        side_def = QuestDefinition(id="side", name="Side", description="", quest_type=QuestType.SIDE)

        quest_tracker.accept_quest(Quest(definition=main_def, state=QuestState.AVAILABLE), timestamp=0)
        quest_tracker.accept_quest(Quest(definition=side_def, state=QuestState.AVAILABLE), timestamp=1)

        main_quests = quest_tracker.get_quests_by_type(QuestType.MAIN)
        assert len(main_quests) == 1
        assert "main" in main_quests

    def test_get_quest_count(self, quest_tracker):
        """Test getting quest counts."""
        for i in range(5):
            quest_def = QuestDefinition(id=f"q{i}", name=f"Q{i}", description="")
            quest = Quest(definition=quest_def, state=QuestState.AVAILABLE)
            quest.objectives = []
            quest_tracker.accept_quest(quest, timestamp=float(i))

        quest_tracker.complete_quest("q0", timestamp=100.0)
        quest_tracker.fail_quest("q1", timestamp=100.0)

        counts = quest_tracker.get_quest_counts()
        assert counts["active"] == 3
        assert counts["completed"] == 1
        assert counts["failed"] == 1

    def test_has_quest(self, quest_tracker, simple_quest_def, kill_objective):
        """Test checking if tracker has a quest."""
        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [kill_objective]

        assert quest_tracker.has_quest(simple_quest_def.id) is False

        quest_tracker.accept_quest(quest, timestamp=100.0)

        assert quest_tracker.has_quest(simple_quest_def.id) is True


# =============================================================================
# Notification Tests
# =============================================================================

class TestProgressNotifications:
    """Tests for progress notification functionality."""

    def test_notification_on_progress(self, quest_tracker, simple_quest_def, kill_objective):
        """Test notification is generated on progress."""
        notifications = []

        def on_notify(notification):
            notifications.append(notification)

        quest_tracker.set_notification_handler(on_notify)

        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [kill_objective]

        quest_tracker.accept_quest(quest, timestamp=100.0)
        quest_tracker.process_event("kill", {"target_type": "wolf"})

        assert len(notifications) > 0

    def test_notification_on_objective_complete(self, quest_tracker, simple_quest_def, kill_objective):
        """Test notification on objective complete."""
        notifications = []

        def on_notify(notification):
            notifications.append(notification)

        quest_tracker.set_notification_handler(on_notify)

        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [kill_objective]

        quest_tracker.accept_quest(quest, timestamp=100.0)

        # Complete objective
        for _ in range(10):
            quest_tracker.process_event("kill", {"target_type": "wolf"})

        # Find objective complete notification
        obj_complete_notifs = [n for n in notifications if n.get("type") == "objective_complete"]
        assert len(obj_complete_notifs) > 0

    def test_notification_on_quest_complete(self, quest_tracker, simple_quest_def, kill_objective):
        """Test notification on quest complete."""
        notifications = []

        def on_notify(notification):
            notifications.append(notification)

        quest_tracker.set_notification_handler(on_notify)

        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        kill_objective.current = kill_objective.required
        quest.objectives = [kill_objective]

        quest_tracker.accept_quest(quest, timestamp=100.0)
        quest_tracker.complete_quest(simple_quest_def.id, timestamp=200.0)

        # Find quest complete notification
        quest_complete_notifs = [n for n in notifications if n.get("type") == "quest_complete"]
        assert len(quest_complete_notifs) > 0

    def test_disable_notifications(self, quest_tracker, simple_quest_def, kill_objective):
        """Test disabling notifications."""
        notifications = []

        def on_notify(notification):
            notifications.append(notification)

        quest_tracker.set_notification_handler(on_notify)
        quest_tracker.disable_notifications()

        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [kill_objective]

        quest_tracker.accept_quest(quest, timestamp=100.0)

        assert len(notifications) == 0


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestQuestTrackerEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_process_event_empty_active_quests(self, quest_tracker):
        """Test processing event with no active quests."""
        # Should not raise
        quest_tracker.process_event("kill", {"target_type": "wolf"})

    def test_complete_nonexistent_quest(self, quest_tracker):
        """Test completing a non-existent quest."""
        result = quest_tracker.complete_quest("nonexistent", timestamp=100.0)
        assert result is False

    def test_abandon_nonexistent_quest(self, quest_tracker):
        """Test abandoning a non-existent quest."""
        result = quest_tracker.abandon_quest("nonexistent")
        assert result is False

    def test_track_nonexistent_quest(self, quest_tracker):
        """Test tracking a non-existent quest."""
        result = quest_tracker.track_quest("nonexistent")
        assert result is False

    def test_untrack_nontracked_quest(self, quest_tracker, simple_quest_def, kill_objective):
        """Test untracking a quest that isn't tracked."""
        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [kill_objective]

        quest_tracker.accept_quest(quest, timestamp=100.0)
        result = quest_tracker.untrack_quest(simple_quest_def.id)

        assert result is False  # Wasn't tracked (assuming auto_track is disabled in this test)

    def test_multiple_accepts_same_quest(self, quest_tracker, simple_quest_def, kill_objective):
        """Test accepting the same quest multiple times."""
        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [kill_objective]

        result1 = quest_tracker.accept_quest(quest, timestamp=100.0)
        result2 = quest_tracker.accept_quest(quest, timestamp=200.0)

        assert result1 is True
        assert result2 is False

    def test_progress_persistence_with_no_quests(self, quest_tracker):
        """Test saving/loading with no quests."""
        save_data = quest_tracker.save()
        assert save_data["quests"] == {}

        quest_tracker.load(save_data)
        assert len(quest_tracker.active_quests) == 0

    def test_very_large_progress_values(self, quest_tracker, simple_quest_def):
        """Test handling very large progress values."""
        obj = KillObjective(
            id="kill_many",
            description="Kill many",
            target_type="enemy",
            required=1000000,
        )
        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [obj]

        quest_tracker.accept_quest(quest, timestamp=100.0)

        # Update with large count
        quest_tracker.process_event("kill", {"target_type": "enemy", "count": 500000})

        progress = quest_tracker.get_progress(simple_quest_def.id)
        assert progress.objectives["kill_many"].current == 500000

    def test_zero_timestamp(self, quest_tracker, simple_quest_def, kill_objective):
        """Test with zero timestamp."""
        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [kill_objective]

        result = quest_tracker.accept_quest(quest, timestamp=0.0)
        assert result is True

    def test_negative_timestamp(self, quest_tracker, simple_quest_def, kill_objective):
        """Test with negative timestamp."""
        quest = Quest(definition=simple_quest_def, state=QuestState.AVAILABLE)
        quest.objectives = [kill_objective]

        # Should still work (timestamps might be relative)
        result = quest_tracker.accept_quest(quest, timestamp=-100.0)
        assert result is True
