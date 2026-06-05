"""
Quest System Module.

Provides quest definitions, objectives, tracking, and event management.
"""

from .quest import (
    Quest,
    QuestDefinition,
    QuestRegistry,
    QuestState,
    QuestType,
    quest,
    # Foundation event types
    QuestStateChanged,
    ObjectiveProgress,
    ObjectiveCompleted,
    QuestRewardGranted,
    # Event helpers
    fire_quest_event,
    get_quest_events,
    clear_quest_events,
    get_registered_quests,
)

from .objectives import (
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

from .tracker import (
    QuestTracker,
    QuestEvent,
    QuestEventType,
    QuestListener,
    TrackedQuest,
)

__all__ = [
    # Quest core
    "Quest",
    "QuestDefinition",
    "QuestRegistry",
    "QuestState",
    "QuestType",
    "quest",
    # Foundation event types
    "QuestStateChanged",
    "ObjectiveProgress",
    "ObjectiveCompleted",
    "QuestRewardGranted",
    # Event helpers
    "fire_quest_event",
    "get_quest_events",
    "clear_quest_events",
    "get_registered_quests",
    # Objectives
    "Objective",
    "ObjectiveState",
    "ObjectiveType",
    "KillObjective",
    "CollectObjective",
    "TalkObjective",
    "ReachObjective",
    "EscortObjective",
    "InteractObjective",
    "UseObjective",
    "CraftObjective",
    "DefendObjective",
    "TimedObjective",
    "CompositeObjective",
    "ObjectiveFactory",
    # Tracker
    "QuestTracker",
    "QuestEvent",
    "QuestEventType",
    "QuestListener",
    "TrackedQuest",
]
