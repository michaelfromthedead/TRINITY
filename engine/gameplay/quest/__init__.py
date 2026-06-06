"""
Quest System Module.

Provides the complete quest system including:
- Quest definitions and state management
- Objective types (kill, collect, talk, reach, escort, etc.)
- Quest tracking and progress management
- Quest journal UI support

Foundation Integration (T-GP-9.5):
- @quest decorator registers with Foundation Registry
- Quest events logged via EventLog
- Causal chains track objective -> quest -> reward dependencies
"""

from engine.gameplay.quest.quest import (
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

from engine.gameplay.quest.tracker import (
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
