"""
Quest Definition Module.

Provides the core Quest class and @quest decorator for defining quests
with name, description, type, level requirements, and rewards.

Foundation Integration (T-GP-9.5):
- @quest decorator registers with Foundation Registry
- Quest events are logged via EventLog
- Causal chains track objective -> quest -> reward dependencies
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, TypeVar
import time

# Foundation imports
from foundation import (
    Event as FoundationEvent,
    EventLog,
    Registry,
    registry,
    traced,
    get_event_log,
    get_current_tick,
    set_current_tick,
    add_change_to_current_event,
    EventChange,  # This is the Change from eventlog, not tracker
)

if TYPE_CHECKING:
    from .quest_rewards import Reward

__all__ = [
    "QuestState",
    "QuestType",
    "Quest",
    "QuestDefinition",
    "quest",
    "QuestRegistry",
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
]


class QuestState(Enum):
    """Quest lifecycle states."""

    UNAVAILABLE = auto()  # Prerequisites not met
    AVAILABLE = auto()    # Can be accepted
    ACTIVE = auto()       # Currently in progress
    COMPLETE = auto()     # Objectives done, not turned in
    TURNED_IN = auto()    # Completed and rewards claimed
    FAILED = auto()       # Quest failed (time limit, choices, etc.)


class QuestType(Enum):
    """Types of quests."""

    MAIN = auto()         # Main story quests
    SIDE = auto()         # Optional side quests
    DAILY = auto()        # Daily repeatable quests
    WEEKLY = auto()       # Weekly repeatable quests
    WORLD = auto()        # World/zone quests
    DUNGEON = auto()      # Dungeon-specific quests
    RAID = auto()         # Raid quests
    PVP = auto()          # PvP-related quests
    HIDDEN = auto()       # Secret/hidden quests
    TUTORIAL = auto()     # Tutorial quests
    EVENT = auto()        # Time-limited event quests
    BOUNTY = auto()       # Bounty/hunt quests
    EXPLORATION = auto()  # Exploration quests


# =============================================================================
# Foundation Event Definitions
# =============================================================================

@dataclass
class QuestStateChanged:
    """
    Event fired when a quest changes state.

    Logged to Foundation EventLog for replay and debugging.
    """
    quest_id: str
    entity_id: str  # Player/entity ID
    old_state: QuestState
    new_state: QuestState
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for EventLog."""
        return {
            "type": "QuestStateChanged",
            "quest_id": self.quest_id,
            "entity_id": self.entity_id,
            "old_state": self.old_state.name,
            "new_state": self.new_state.name,
            "timestamp": self.timestamp,
        }


@dataclass
class ObjectiveProgress:
    """
    Event fired when objective progress changes.
    """
    quest_id: str
    objective_id: str
    current: int | float
    target: int | float
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for EventLog."""
        return {
            "type": "ObjectiveProgress",
            "quest_id": self.quest_id,
            "objective_id": self.objective_id,
            "current": self.current,
            "target": self.target,
            "timestamp": self.timestamp,
        }


@dataclass
class ObjectiveCompleted:
    """
    Event fired when an objective is completed.
    """
    quest_id: str
    objective_id: str
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for EventLog."""
        return {
            "type": "ObjectiveCompleted",
            "quest_id": self.quest_id,
            "objective_id": self.objective_id,
            "timestamp": self.timestamp,
        }


@dataclass
class QuestRewardGranted:
    """
    Event fired when a quest reward is granted.
    """
    quest_id: str
    entity_id: str  # Player/entity receiving reward
    reward_type: str  # e.g., "xp", "gold", "item"
    amount: int | float | str  # Amount or item ID
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for EventLog."""
        return {
            "type": "QuestRewardGranted",
            "quest_id": self.quest_id,
            "entity_id": self.entity_id,
            "reward_type": self.reward_type,
            "amount": self.amount,
            "timestamp": self.timestamp,
        }


# Quest event log (separate from global EventLog for quest-specific queries)
_quest_event_log: list[QuestStateChanged | ObjectiveProgress | ObjectiveCompleted | QuestRewardGranted] = []


def fire_quest_event(
    event: QuestStateChanged | ObjectiveProgress | ObjectiveCompleted | QuestRewardGranted
) -> None:
    """
    Fire a quest event, recording it to both quest-specific log and Foundation EventLog.
    """
    _quest_event_log.append(event)

    # Also record to Foundation EventLog as a Change
    # We create a foundation EventChange to track the quest event
    if isinstance(event, QuestStateChanged):
        change = EventChange(
            entity=hash(event.entity_id) & 0x7FFFFFFF,  # Use hash as entity ID
            field=f"quest.{event.quest_id}.state",
            old_value=event.old_state.name if event.old_state else None,
            new_value=event.new_state.name,
        )
        add_change_to_current_event(change)


def get_quest_events(
    quest_id: str | None = None,
    entity_id: str | None = None,
    event_type: str | None = None,
) -> list[QuestStateChanged | ObjectiveProgress | ObjectiveCompleted | QuestRewardGranted]:
    """
    Query quest events with optional filters.

    Args:
        quest_id: Filter by quest ID
        entity_id: Filter by entity/player ID
        event_type: Filter by event type name

    Returns:
        List of matching quest events
    """
    result = _quest_event_log

    if quest_id is not None:
        result = [e for e in result if e.quest_id == quest_id]

    if entity_id is not None:
        result = [e for e in result if hasattr(e, "entity_id") and e.entity_id == entity_id]

    if event_type is not None:
        type_map = {
            "QuestStateChanged": QuestStateChanged,
            "ObjectiveProgress": ObjectiveProgress,
            "ObjectiveCompleted": ObjectiveCompleted,
            "QuestRewardGranted": QuestRewardGranted,
        }
        target_type = type_map.get(event_type)
        if target_type:
            result = [e for e in result if isinstance(e, target_type)]

    return result


def clear_quest_events() -> None:
    """Clear the quest event log (for testing)."""
    _quest_event_log.clear()


@dataclass
class QuestDefinition:
    """
    Immutable quest definition containing static quest data.

    This is the template/blueprint for a quest, separate from
    the player's progress on that quest.
    """

    id: str
    name: str
    description: str
    quest_type: QuestType = QuestType.SIDE
    level_requirement: int = 1
    level_cap: int | None = None  # Max level to accept quest
    time_limit: float | None = None  # Time limit in seconds
    repeatable: bool = False
    cooldown: float = 0.0  # Cooldown between repeats
    auto_accept: bool = False  # Automatically accept when available
    auto_complete: bool = False  # Automatically complete when objectives done
    hidden: bool = False  # Hidden from quest log until accepted
    shareable: bool = True  # Can share with party members
    abandon_penalty: bool = False  # Penalty for abandoning

    # Prerequisites (filled by @quest decorator)
    prerequisites: list[str] = field(default_factory=list)
    required_items: dict[str, int] = field(default_factory=dict)
    required_reputation: dict[str, int] = field(default_factory=dict)

    # Rewards (filled by @quest decorator)
    rewards: list[Reward] = field(default_factory=list)

    # Metadata
    category: str = ""  # Quest category for grouping
    zone: str = ""  # Zone where quest takes place
    giver_id: str = ""  # NPC that gives the quest
    turn_in_id: str = ""  # NPC to turn in to (defaults to giver)

    # Tags for filtering
    tags: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Validate quest definition."""
        if not self.id:
            raise ValueError("Quest id cannot be empty")
        if not self.name:
            raise ValueError("Quest name cannot be empty")
        if self.level_requirement < 0:
            raise ValueError("level_requirement must be >= 0")
        if self.level_cap is not None and self.level_cap < self.level_requirement:
            raise ValueError("level_cap must be >= level_requirement")
        if self.time_limit is not None and self.time_limit <= 0:
            raise ValueError("time_limit must be > 0")
        if self.cooldown < 0:
            raise ValueError("cooldown must be >= 0")


@dataclass
class Quest:
    """
    Active quest instance representing a player's progress on a quest.

    This tracks the current state and progress for a specific player.
    """

    definition: QuestDefinition
    state: QuestState = QuestState.UNAVAILABLE
    accepted_at: float | None = None  # Timestamp when accepted
    completed_at: float | None = None  # Timestamp when completed
    turned_in_at: float | None = None  # Timestamp when turned in
    failed_at: float | None = None  # Timestamp when failed
    times_completed: int = 0  # For repeatable quests
    last_completed_at: float | None = None  # For cooldown tracking

    # Player-specific data
    player_id: str = ""
    objective_progress: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Get quest ID."""
        return self.definition.id

    @property
    def name(self) -> str:
        """Get quest name."""
        return self.definition.name

    @property
    def is_active(self) -> bool:
        """Check if quest is active."""
        return self.state == QuestState.ACTIVE

    @property
    def is_complete(self) -> bool:
        """Check if quest is complete (but not turned in)."""
        return self.state == QuestState.COMPLETE

    @property
    def is_finished(self) -> bool:
        """Check if quest is turned in or failed."""
        return self.state in (QuestState.TURNED_IN, QuestState.FAILED)

    @property
    def is_available(self) -> bool:
        """Check if quest is available to accept."""
        return self.state == QuestState.AVAILABLE

    @property
    def can_repeat(self) -> bool:
        """Check if quest can be repeated."""
        if not self.definition.repeatable:
            return False
        if self.state not in (QuestState.TURNED_IN, QuestState.FAILED):
            return False
        return True

    def _fire_state_change(self, old_state: QuestState, new_state: QuestState, timestamp: float) -> None:
        """Fire a QuestStateChanged event."""
        event = QuestStateChanged(
            quest_id=self.id,
            entity_id=self.player_id,
            old_state=old_state,
            new_state=new_state,
            timestamp=timestamp,
        )
        fire_quest_event(event)

    def make_available(self, timestamp: float | None = None) -> bool:
        """Transition to AVAILABLE state."""
        if self.state != QuestState.UNAVAILABLE:
            return False
        old_state = self.state
        self.state = QuestState.AVAILABLE
        ts = timestamp if timestamp is not None else time.time()
        self._fire_state_change(old_state, self.state, ts)
        return True

    def accept(self, timestamp: float) -> bool:
        """Accept the quest."""
        if self.state != QuestState.AVAILABLE:
            return False
        old_state = self.state
        self.state = QuestState.ACTIVE
        self.accepted_at = timestamp
        self._fire_state_change(old_state, self.state, timestamp)
        return True

    def complete(self, timestamp: float) -> bool:
        """Mark quest as complete (objectives done)."""
        if self.state != QuestState.ACTIVE:
            return False
        old_state = self.state
        self.state = QuestState.COMPLETE
        self.completed_at = timestamp
        self._fire_state_change(old_state, self.state, timestamp)
        return True

    def turn_in(self, timestamp: float) -> bool:
        """Turn in the quest and claim rewards."""
        if self.state != QuestState.COMPLETE:
            return False
        old_state = self.state
        self.state = QuestState.TURNED_IN
        self.turned_in_at = timestamp
        self.times_completed += 1
        self.last_completed_at = timestamp
        self._fire_state_change(old_state, self.state, timestamp)
        return True

    def fail(self, timestamp: float) -> bool:
        """Mark quest as failed."""
        if self.state != QuestState.ACTIVE:
            return False
        old_state = self.state
        self.state = QuestState.FAILED
        self.failed_at = timestamp
        self._fire_state_change(old_state, self.state, timestamp)
        return True

    def reset(self, timestamp: float | None = None) -> bool:
        """Reset quest for repeat."""
        if not self.can_repeat:
            return False
        old_state = self.state
        self.state = QuestState.AVAILABLE
        self.accepted_at = None
        self.completed_at = None
        self.turned_in_at = None
        self.failed_at = None
        self.objective_progress.clear()
        ts = timestamp if timestamp is not None else time.time()
        self._fire_state_change(old_state, self.state, ts)
        return True

    def abandon(self, timestamp: float | None = None) -> bool:
        """Abandon the quest."""
        if self.state not in (QuestState.ACTIVE, QuestState.COMPLETE):
            return False
        old_state = self.state
        self.state = QuestState.AVAILABLE
        self.accepted_at = None
        self.completed_at = None
        self.objective_progress.clear()
        ts = timestamp if timestamp is not None else time.time()
        self._fire_state_change(old_state, self.state, ts)
        return True


class QuestRegistry:
    """
    Global registry for quest definitions.

    Provides lookup and management of all registered quests.
    """

    _instance: QuestRegistry | None = None

    def __init__(self) -> None:
        self._quests: dict[str, QuestDefinition] = {}
        self._by_type: dict[QuestType, list[str]] = {t: [] for t in QuestType}
        self._by_zone: dict[str, list[str]] = {}
        self._by_giver: dict[str, list[str]] = {}
        self._by_tag: dict[str, list[str]] = {}

    @classmethod
    def instance(cls) -> QuestRegistry:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def clear(cls) -> None:
        """Clear the registry (for testing)."""
        if cls._instance is not None:
            cls._instance._quests.clear()
            cls._instance._by_type = {t: [] for t in QuestType}
            cls._instance._by_zone.clear()
            cls._instance._by_giver.clear()
            cls._instance._by_tag.clear()

    def register(self, quest_def: QuestDefinition) -> None:
        """Register a quest definition."""
        if quest_def.id in self._quests:
            raise ValueError(f"Quest '{quest_def.id}' already registered")

        self._quests[quest_def.id] = quest_def
        self._by_type[quest_def.quest_type].append(quest_def.id)

        if quest_def.zone:
            if quest_def.zone not in self._by_zone:
                self._by_zone[quest_def.zone] = []
            self._by_zone[quest_def.zone].append(quest_def.id)

        if quest_def.giver_id:
            if quest_def.giver_id not in self._by_giver:
                self._by_giver[quest_def.giver_id] = []
            self._by_giver[quest_def.giver_id].append(quest_def.id)

        for tag in quest_def.tags:
            if tag not in self._by_tag:
                self._by_tag[tag] = []
            self._by_tag[tag].append(quest_def.id)

    def unregister(self, quest_id: str) -> bool:
        """Unregister a quest definition."""
        if quest_id not in self._quests:
            return False

        quest_def = self._quests.pop(quest_id)
        self._by_type[quest_def.quest_type].remove(quest_id)

        if quest_def.zone and quest_id in self._by_zone.get(quest_def.zone, []):
            self._by_zone[quest_def.zone].remove(quest_id)

        if quest_def.giver_id and quest_id in self._by_giver.get(quest_def.giver_id, []):
            self._by_giver[quest_def.giver_id].remove(quest_id)

        for tag in quest_def.tags:
            if tag in self._by_tag and quest_id in self._by_tag[tag]:
                self._by_tag[tag].remove(quest_id)

        return True

    def get(self, quest_id: str) -> QuestDefinition | None:
        """Get a quest definition by ID."""
        return self._quests.get(quest_id)

    def get_by_type(self, quest_type: QuestType) -> list[QuestDefinition]:
        """Get all quests of a specific type."""
        return [self._quests[qid] for qid in self._by_type[quest_type]]

    def get_by_zone(self, zone: str) -> list[QuestDefinition]:
        """Get all quests in a specific zone."""
        return [self._quests[qid] for qid in self._by_zone.get(zone, [])]

    def get_by_giver(self, giver_id: str) -> list[QuestDefinition]:
        """Get all quests from a specific giver."""
        return [self._quests[qid] for qid in self._by_giver.get(giver_id, [])]

    def get_by_tag(self, tag: str) -> list[QuestDefinition]:
        """Get all quests with a specific tag."""
        return [self._quests[qid] for qid in self._by_tag.get(tag, [])]

    def all_quests(self) -> list[QuestDefinition]:
        """Get all registered quests."""
        return list(self._quests.values())

    def __len__(self) -> int:
        return len(self._quests)

    def __contains__(self, quest_id: str) -> bool:
        return quest_id in self._quests


T = TypeVar("T", bound=type)


def quest(
    id: str,
    name: str | None = None,
    description: str = "",
    quest_type: QuestType = QuestType.SIDE,
    level_requirement: int = 1,
    prerequisites: list[str] | None = None,
    rewards: list[Any] | None = None,
    **kwargs: Any,
) -> Callable[[T], T]:
    """
    Decorator for defining quests.

    Registers the quest class with both the QuestRegistry and Foundation Registry,
    enabling runtime discovery via Registry.query(tag="quest").

    Usage:
        @quest(
            id="main_quest_1",
            name="The Beginning",
            description="Start your adventure",
            quest_type=QuestType.MAIN,
            level_requirement=1,
            prerequisites=["tutorial"],
            rewards=[XPReward(100), ItemReward("sword", 1)]
        )
        class MainQuest1:
            pass

    Args:
        id: Unique quest identifier
        name: Display name (defaults to class name)
        description: Quest description text
        quest_type: Type of quest
        level_requirement: Minimum level to accept
        prerequisites: List of prerequisite quest IDs
        rewards: List of Reward objects
        **kwargs: Additional QuestDefinition parameters

    Returns:
        Decorated class with quest metadata
    """
    if not id:
        raise ValueError("Quest id must be non-empty")

    def decorator(cls: T) -> T:
        quest_name = name if name is not None else cls.__name__

        quest_def = QuestDefinition(
            id=id,
            name=quest_name,
            description=description,
            quest_type=quest_type,
            level_requirement=level_requirement,
            prerequisites=prerequisites or [],
            rewards=rewards or [],
            **kwargs,
        )

        # Attach metadata to class
        cls._quest = True  # type: ignore[attr-defined]
        cls._quest_id = id  # type: ignore[attr-defined]
        cls._quest_definition = quest_def  # type: ignore[attr-defined]
        cls._quest_prerequisites = prerequisites or []  # type: ignore[attr-defined]
        cls._quest_rewards = rewards or []  # type: ignore[attr-defined]

        # Add standard metadata
        if not hasattr(cls, "_tags"):
            cls._tags = {}  # type: ignore[attr-defined]
        cls._tags["quest"] = True  # type: ignore[attr-defined]
        cls._tags["quest_id"] = id  # type: ignore[attr-defined]
        cls._tags["quest_type"] = quest_type.name  # type: ignore[attr-defined]

        if not hasattr(cls, "_registries"):
            cls._registries = set()  # type: ignore[attr-defined]
        cls._registries.add("gameplay")  # type: ignore[attr-defined]

        if not hasattr(cls, "_applied_decorators"):
            cls._applied_decorators = set()  # type: ignore[attr-defined]
        cls._applied_decorators.add("quest")  # type: ignore[attr-defined]

        # Register the quest with QuestRegistry
        QuestRegistry.instance().register(quest_def)

        # Register with Foundation Registry for runtime discovery
        # Use custom name to enable query by quest tag
        registry_name = f"quest.{id}"
        try:
            registry.register(cls, name=registry_name, track_instances=False)
            # Set metadata for Foundation Registry queries
            registry.set_metadata(cls, "quest", True)
            registry.set_metadata(cls, "quest_id", id)
            registry.set_metadata(cls, "quest_type", quest_type.name)
            registry.set_metadata(cls, "quest_definition", quest_def)
        except ValueError:
            # Already registered, skip
            pass

        return cls

    return decorator


def get_registered_quests() -> list[type]:
    """
    Query Foundation Registry for all quest-decorated classes.

    Returns:
        List of classes decorated with @quest
    """
    return registry.types_with_decorator("quest")
