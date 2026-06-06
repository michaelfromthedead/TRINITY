"""
Quest Tracker Module.

Provides QuestTracker for tracking quest progress, handling events,
and managing quest state transitions.

Foundation Integration (T-GP-9.5):
- Fires QuestRewardGranted events when rewards are claimed
- Implements causal chains: objective complete -> quest complete -> rewards
- All events are logged via Foundation EventLog
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from .objectives import Objective, ObjectiveState
from .quest import (
    Quest,
    QuestDefinition,
    QuestRegistry,
    QuestState,
    QuestRewardGranted,
    fire_quest_event,
)

if TYPE_CHECKING:
    from .quest_rewards import Reward

__all__ = [
    "QuestTracker",
    "QuestEvent",
    "QuestEventType",
    "QuestListener",
]


class QuestEventType:
    """Quest event type constants."""

    QUEST_AVAILABLE = "quest_available"
    QUEST_ACCEPTED = "quest_accepted"
    QUEST_PROGRESS = "quest_progress"
    QUEST_OBJECTIVE_COMPLETE = "quest_objective_complete"
    QUEST_COMPLETE = "quest_complete"
    QUEST_TURNED_IN = "quest_turned_in"
    QUEST_FAILED = "quest_failed"
    QUEST_ABANDONED = "quest_abandoned"
    QUEST_RESET = "quest_reset"


@dataclass
class QuestEvent:
    """Event data for quest-related events."""

    event_type: str
    quest_id: str
    player_id: str
    timestamp: float
    objective_id: str | None = None
    progress: float | None = None
    rewards: list[Reward] | None = None
    data: dict[str, Any] = field(default_factory=dict)


# Type alias for quest event listeners
QuestListener = Callable[[QuestEvent], None]


@dataclass
class TrackedQuest:
    """Quest with its associated objectives being tracked."""

    quest: Quest
    objectives: list[Objective] = field(default_factory=list)
    current_objective_index: int = 0  # For sequential objectives

    @property
    def active_objectives(self) -> list[Objective]:
        """Get currently active objectives."""
        return [obj for obj in self.objectives if obj.is_active]

    @property
    def complete_objectives(self) -> list[Objective]:
        """Get completed objectives."""
        return [obj for obj in self.objectives if obj.is_complete]

    @property
    def all_required_complete(self) -> bool:
        """Check if all required objectives are complete."""
        required = [obj for obj in self.objectives if not obj.optional]
        return all(obj.is_complete for obj in required)


class QuestTracker:
    """
    Tracks quest progress for a player.

    Handles:
    - Quest state management
    - Objective progress tracking
    - Event dispatching
    - Prerequisite checking
    """

    def __init__(self, player_id: str) -> None:
        self.player_id = player_id
        self._tracked: dict[str, TrackedQuest] = {}
        self._completed_quests: set[str] = set()
        self._failed_quests: set[str] = set()
        self._listeners: list[QuestListener] = []
        self._global_listeners: list[QuestListener] = []
        self._event_handlers: dict[str, list[Callable[[str, dict[str, Any]], None]]] = {}

        # Player stats for prerequisite checking
        self._player_level: int = 1
        self._player_items: dict[str, int] = {}
        self._player_reputation: dict[str, int] = {}

        # Time tracking
        self._current_time: float = 0.0

    @property
    def active_quests(self) -> list[Quest]:
        """Get all active quests."""
        return [
            tq.quest for tq in self._tracked.values()
            if tq.quest.state == QuestState.ACTIVE
        ]

    @property
    def available_quests(self) -> list[Quest]:
        """Get all available quests."""
        return [
            tq.quest for tq in self._tracked.values()
            if tq.quest.state == QuestState.AVAILABLE
        ]

    @property
    def completed_quests(self) -> list[Quest]:
        """Get all completed (but not turned in) quests."""
        return [
            tq.quest for tq in self._tracked.values()
            if tq.quest.state == QuestState.COMPLETE
        ]

    def set_time(self, timestamp: float) -> None:
        """Update the current time."""
        self._current_time = timestamp

    def set_player_level(self, level: int) -> None:
        """Set player level for prerequisite checking."""
        self._player_level = level
        self._check_availability()

    def set_player_items(self, items: dict[str, int]) -> None:
        """Set player items for prerequisite checking."""
        self._player_items = items.copy()
        self._check_availability()

    def set_player_reputation(self, reputation: dict[str, int]) -> None:
        """Set player reputation for prerequisite checking."""
        self._player_reputation = reputation.copy()
        self._check_availability()

    def add_listener(self, listener: QuestListener) -> None:
        """Add a quest event listener."""
        self._listeners.append(listener)

    def remove_listener(self, listener: QuestListener) -> None:
        """Remove a quest event listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def add_global_listener(self, listener: QuestListener) -> None:
        """Add a global listener that receives all events."""
        self._global_listeners.append(listener)

    def register_event_handler(
        self,
        event_type: str,
        handler: Callable[[str, dict[str, Any]], None]
    ) -> None:
        """Register a handler for game events."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def _emit_event(self, event: QuestEvent) -> None:
        """Emit a quest event to all listeners."""
        for listener in self._listeners:
            listener(event)
        for listener in self._global_listeners:
            listener(event)

    def _check_prerequisites(self, quest_def: QuestDefinition) -> bool:
        """Check if all prerequisites are met for a quest."""
        # Check required quests
        for prereq_id in quest_def.prerequisites:
            if prereq_id not in self._completed_quests:
                return False

        # Check level requirement
        if self._player_level < quest_def.level_requirement:
            return False

        # Check level cap
        if quest_def.level_cap is not None and self._player_level > quest_def.level_cap:
            return False

        # Check required items
        for item_id, required in quest_def.required_items.items():
            if self._player_items.get(item_id, 0) < required:
                return False

        # Check required reputation
        for faction_id, required in quest_def.required_reputation.items():
            if self._player_reputation.get(faction_id, 0) < required:
                return False

        return True

    def _check_availability(self) -> None:
        """Check and update availability for all tracked quests."""
        for tracked in self._tracked.values():
            if tracked.quest.state == QuestState.UNAVAILABLE:
                if self._check_prerequisites(tracked.quest.definition):
                    tracked.quest.make_available()
                    self._emit_event(QuestEvent(
                        event_type=QuestEventType.QUEST_AVAILABLE,
                        quest_id=tracked.quest.id,
                        player_id=self.player_id,
                        timestamp=self._current_time,
                    ))

    def track_quest(
        self,
        quest_def: QuestDefinition,
        objectives: list[Objective] | None = None
    ) -> TrackedQuest:
        """
        Start tracking a quest.

        Args:
            quest_def: The quest definition to track
            objectives: Optional list of objectives for this quest

        Returns:
            The tracked quest object
        """
        if quest_def.id in self._tracked:
            return self._tracked[quest_def.id]

        quest = Quest(
            definition=quest_def,
            player_id=self.player_id,
        )

        # Check if prerequisites are met
        if self._check_prerequisites(quest_def):
            quest.state = QuestState.AVAILABLE
        else:
            quest.state = QuestState.UNAVAILABLE

        # Set quest_id on all objectives for Foundation event firing
        obj_list = objectives or []
        for obj in obj_list:
            obj.quest_id = quest_def.id

        tracked = TrackedQuest(
            quest=quest,
            objectives=obj_list,
        )

        self._tracked[quest_def.id] = tracked

        if quest.state == QuestState.AVAILABLE:
            self._emit_event(QuestEvent(
                event_type=QuestEventType.QUEST_AVAILABLE,
                quest_id=quest.id,
                player_id=self.player_id,
                timestamp=self._current_time,
            ))

        return tracked

    def track_quest_by_id(
        self,
        quest_id: str,
        objectives: list[Objective] | None = None
    ) -> TrackedQuest | None:
        """Track a quest by its ID from the registry."""
        quest_def = QuestRegistry.instance().get(quest_id)
        if quest_def is None:
            return None
        return self.track_quest(quest_def, objectives)

    def untrack_quest(self, quest_id: str) -> bool:
        """Stop tracking a quest."""
        if quest_id not in self._tracked:
            return False

        del self._tracked[quest_id]
        return True

    def get_tracked(self, quest_id: str) -> TrackedQuest | None:
        """Get a tracked quest by ID."""
        return self._tracked.get(quest_id)

    def accept_quest(self, quest_id: str) -> bool:
        """
        Accept a quest.

        Args:
            quest_id: The quest to accept

        Returns:
            True if quest was accepted successfully
        """
        tracked = self._tracked.get(quest_id)
        if tracked is None:
            return False

        if not tracked.quest.accept(self._current_time):
            return False

        # Activate objectives
        for obj in tracked.objectives:
            if not obj.optional or obj.order == 0:
                obj.activate()

        self._emit_event(QuestEvent(
            event_type=QuestEventType.QUEST_ACCEPTED,
            quest_id=quest_id,
            player_id=self.player_id,
            timestamp=self._current_time,
        ))

        return True

    def complete_quest(self, quest_id: str) -> bool:
        """
        Mark a quest as complete (objectives done).

        Args:
            quest_id: The quest to complete

        Returns:
            True if quest was completed successfully
        """
        tracked = self._tracked.get(quest_id)
        if tracked is None:
            return False

        if not tracked.quest.complete(self._current_time):
            return False

        self._emit_event(QuestEvent(
            event_type=QuestEventType.QUEST_COMPLETE,
            quest_id=quest_id,
            player_id=self.player_id,
            timestamp=self._current_time,
        ))

        return True

    def turn_in_quest(self, quest_id: str) -> list[Reward]:
        """
        Turn in a quest and claim rewards.

        Fires QuestRewardGranted events for each reward (Foundation integration).

        Args:
            quest_id: The quest to turn in

        Returns:
            List of rewards claimed
        """
        tracked = self._tracked.get(quest_id)
        if tracked is None:
            return []

        if not tracked.quest.turn_in(self._current_time):
            return []

        self._completed_quests.add(quest_id)

        rewards = tracked.quest.definition.rewards

        self._emit_event(QuestEvent(
            event_type=QuestEventType.QUEST_TURNED_IN,
            quest_id=quest_id,
            player_id=self.player_id,
            timestamp=self._current_time,
            rewards=rewards,
        ))

        # Fire QuestRewardGranted events for each reward (Foundation integration)
        for reward in rewards:
            # Determine reward type and amount from reward object
            reward_type = getattr(reward, "reward_type", type(reward).__name__)
            amount = getattr(reward, "amount", getattr(reward, "value", 1))

            reward_event = QuestRewardGranted(
                quest_id=quest_id,
                entity_id=self.player_id,
                reward_type=reward_type,
                amount=amount,
                timestamp=self._current_time,
            )
            fire_quest_event(reward_event)

        # Check if this unlocks other quests
        self._check_availability()

        return rewards

    def fail_quest(self, quest_id: str) -> bool:
        """
        Mark a quest as failed.

        Args:
            quest_id: The quest to fail

        Returns:
            True if quest was failed successfully
        """
        tracked = self._tracked.get(quest_id)
        if tracked is None:
            return False

        if not tracked.quest.fail(self._current_time):
            return False

        self._failed_quests.add(quest_id)

        self._emit_event(QuestEvent(
            event_type=QuestEventType.QUEST_FAILED,
            quest_id=quest_id,
            player_id=self.player_id,
            timestamp=self._current_time,
        ))

        return True

    def abandon_quest(self, quest_id: str) -> bool:
        """
        Abandon a quest.

        Args:
            quest_id: The quest to abandon

        Returns:
            True if quest was abandoned successfully
        """
        tracked = self._tracked.get(quest_id)
        if tracked is None:
            return False

        if not tracked.quest.abandon():
            return False

        # Reset objectives
        for obj in tracked.objectives:
            obj.reset()

        self._emit_event(QuestEvent(
            event_type=QuestEventType.QUEST_ABANDONED,
            quest_id=quest_id,
            player_id=self.player_id,
            timestamp=self._current_time,
        ))

        return True

    def reset_quest(self, quest_id: str) -> bool:
        """
        Reset a repeatable quest.

        Args:
            quest_id: The quest to reset

        Returns:
            True if quest was reset successfully
        """
        tracked = self._tracked.get(quest_id)
        if tracked is None:
            return False

        if not tracked.quest.reset():
            return False

        # Reset objectives
        for obj in tracked.objectives:
            obj.reset()
        tracked.current_objective_index = 0

        self._emit_event(QuestEvent(
            event_type=QuestEventType.QUEST_RESET,
            quest_id=quest_id,
            player_id=self.player_id,
            timestamp=self._current_time,
        ))

        return True

    def process_event(self, event_type: str, event_data: dict[str, Any]) -> None:
        """
        Process a game event and update quest progress.

        Args:
            event_type: Type of game event
            event_data: Event-specific data
        """
        # Call registered handlers
        if event_type in self._event_handlers:
            for handler in self._event_handlers[event_type]:
                handler(event_type, event_data)

        # Update all active quests
        for tracked in self._tracked.values():
            if tracked.quest.state != QuestState.ACTIVE:
                continue

            for obj in tracked.objectives:
                if obj.state != ObjectiveState.IN_PROGRESS:
                    continue

                old_progress = obj.progress
                if obj.update(event_type, event_data):
                    # Emit progress event
                    self._emit_event(QuestEvent(
                        event_type=QuestEventType.QUEST_PROGRESS,
                        quest_id=tracked.quest.id,
                        player_id=self.player_id,
                        timestamp=self._current_time,
                        objective_id=obj.id,
                        progress=obj.progress,
                        data={"old_progress": old_progress},
                    ))

                    # Check if objective completed
                    if obj.is_complete:
                        self._emit_event(QuestEvent(
                            event_type=QuestEventType.QUEST_OBJECTIVE_COMPLETE,
                            quest_id=tracked.quest.id,
                            player_id=self.player_id,
                            timestamp=self._current_time,
                            objective_id=obj.id,
                        ))

                    # Check if objective failed (which might fail the quest)
                    if obj.is_failed and not obj.optional:
                        self.fail_quest(tracked.quest.id)

            # Check if all required objectives are complete
            if tracked.all_required_complete:
                if tracked.quest.definition.auto_complete:
                    self.complete_quest(tracked.quest.id)
                    self.turn_in_quest(tracked.quest.id)
                else:
                    self.complete_quest(tracked.quest.id)

    def update(self, delta_time: float) -> None:
        """
        Update time-based quest elements.

        Args:
            delta_time: Time elapsed since last update
        """
        self._current_time += delta_time

        # Check time limits
        for tracked in self._tracked.values():
            if tracked.quest.state != QuestState.ACTIVE:
                continue

            quest_def = tracked.quest.definition
            if quest_def.time_limit is not None:
                elapsed = self._current_time - (tracked.quest.accepted_at or 0)
                if elapsed >= quest_def.time_limit:
                    self.fail_quest(tracked.quest.id)

        # Send time tick events to objectives
        self.process_event("time_tick", {"delta_time": delta_time})

    def get_progress_summary(self, quest_id: str) -> dict[str, Any] | None:
        """
        Get a summary of quest progress.

        Args:
            quest_id: The quest to get progress for

        Returns:
            Dictionary with progress information or None if not tracked
        """
        tracked = self._tracked.get(quest_id)
        if tracked is None:
            return None

        objectives_summary = []
        for obj in tracked.objectives:
            objectives_summary.append({
                "id": obj.id,
                "description": obj.description,
                "state": obj.state.name,
                "progress": obj.progress,
                "progress_text": obj.progress_text,
                "optional": obj.optional,
            })

        total_progress = (
            sum(obj.progress for obj in tracked.objectives) / len(tracked.objectives)
            if tracked.objectives else 0.0
        )

        return {
            "quest_id": quest_id,
            "quest_name": tracked.quest.name,
            "state": tracked.quest.state.name,
            "total_progress": total_progress,
            "objectives": objectives_summary,
            "times_completed": tracked.quest.times_completed,
        }

    def serialize(self) -> dict[str, Any]:
        """Serialize tracker state for saving."""
        return {
            "player_id": self.player_id,
            "current_time": self._current_time,
            "completed_quests": list(self._completed_quests),
            "failed_quests": list(self._failed_quests),
            "player_level": self._player_level,
            "player_items": self._player_items.copy(),
            "player_reputation": self._player_reputation.copy(),
            "tracked_quests": {
                qid: {
                    "state": tq.quest.state.name,
                    "accepted_at": tq.quest.accepted_at,
                    "completed_at": tq.quest.completed_at,
                    "turned_in_at": tq.quest.turned_in_at,
                    "failed_at": tq.quest.failed_at,
                    "times_completed": tq.quest.times_completed,
                    "objective_progress": tq.quest.objective_progress,
                }
                for qid, tq in self._tracked.items()
            },
        }

    def deserialize(self, data: dict[str, Any]) -> None:
        """Deserialize tracker state from saved data."""
        self._current_time = data.get("current_time", 0.0)
        self._completed_quests = set(data.get("completed_quests", []))
        self._failed_quests = set(data.get("failed_quests", []))
        self._player_level = data.get("player_level", 1)
        self._player_items = data.get("player_items", {}).copy()
        self._player_reputation = data.get("player_reputation", {}).copy()

        # Restore tracked quest states
        for quest_id, quest_data in data.get("tracked_quests", {}).items():
            tracked = self._tracked.get(quest_id)
            if tracked is None:
                continue

            state_name = quest_data.get("state", "UNAVAILABLE")
            tracked.quest.state = QuestState[state_name]
            tracked.quest.accepted_at = quest_data.get("accepted_at")
            tracked.quest.completed_at = quest_data.get("completed_at")
            tracked.quest.turned_in_at = quest_data.get("turned_in_at")
            tracked.quest.failed_at = quest_data.get("failed_at")
            tracked.quest.times_completed = quest_data.get("times_completed", 0)
            tracked.quest.objective_progress = quest_data.get("objective_progress", {})
