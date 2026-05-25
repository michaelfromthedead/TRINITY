"""
Quest Objectives Module.

Provides objective types for quests: Kill, Collect, Talk, Reach, Escort, Interact.
Each objective type has counters, flags, and progress tracking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

from .constants import (
    DEFAULT_REACH_RADIUS,
    DEFAULT_HEALTH_PERCENT,
    MIN_HEALTH_PERCENT_DEFAULT,
    DEFAULT_ESCORT_DISTANCE_THRESHOLD,
    DEFAULT_DEFEND_DURATION,
    DEFAULT_TARGET_HEALTH_PERCENT,
    DEFAULT_TIMED_OBJECTIVE_LIMIT,
)

__all__ = [
    "ObjectiveState",
    "ObjectiveType",
    "Objective",
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
]


class ObjectiveState(Enum):
    """Objective completion states."""

    INACTIVE = auto()    # Not yet started
    IN_PROGRESS = auto()  # Currently active
    COMPLETE = auto()     # Successfully completed
    FAILED = auto()       # Failed (for timed/conditional objectives)


class ObjectiveType(Enum):
    """Types of objectives."""

    KILL = auto()
    COLLECT = auto()
    TALK = auto()
    REACH = auto()
    ESCORT = auto()
    INTERACT = auto()
    USE = auto()
    CRAFT = auto()
    DEFEND = auto()
    TIMED = auto()
    COMPOSITE = auto()
    CUSTOM = auto()


@dataclass
class Objective(ABC):
    """
    Base class for quest objectives.

    Objectives track progress toward specific goals within a quest.
    """

    id: str
    description: str
    objective_type: ObjectiveType
    state: ObjectiveState = ObjectiveState.INACTIVE
    optional: bool = False  # Optional bonus objective
    hidden: bool = False  # Hidden until certain conditions met
    order: int = 0  # Order for sequential objectives

    # Callbacks
    on_complete: Callable[[Objective], None] | None = None
    on_fail: Callable[[Objective], None] | None = None
    on_progress: Callable[[Objective, float], None] | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Objective id cannot be empty")

    @property
    @abstractmethod
    def progress(self) -> float:
        """Get progress as a value between 0.0 and 1.0."""
        ...

    @property
    @abstractmethod
    def progress_text(self) -> str:
        """Get human-readable progress text."""
        ...

    @property
    def is_complete(self) -> bool:
        """Check if objective is complete."""
        return self.state == ObjectiveState.COMPLETE

    @property
    def is_failed(self) -> bool:
        """Check if objective is failed."""
        return self.state == ObjectiveState.FAILED

    @property
    def is_active(self) -> bool:
        """Check if objective is in progress."""
        return self.state == ObjectiveState.IN_PROGRESS

    def activate(self) -> bool:
        """Activate the objective."""
        if self.state != ObjectiveState.INACTIVE:
            return False
        self.state = ObjectiveState.IN_PROGRESS
        return True

    def complete(self) -> bool:
        """Mark objective as complete."""
        if self.state not in (ObjectiveState.IN_PROGRESS, ObjectiveState.INACTIVE):
            return False
        self.state = ObjectiveState.COMPLETE
        if self.on_complete:
            self.on_complete(self)
        return True

    def fail(self) -> bool:
        """Mark objective as failed."""
        if self.state != ObjectiveState.IN_PROGRESS:
            return False
        self.state = ObjectiveState.FAILED
        if self.on_fail:
            self.on_fail(self)
        return True

    def reset(self) -> None:
        """Reset objective to inactive state."""
        self.state = ObjectiveState.INACTIVE

    @abstractmethod
    def update(self, event_type: str, event_data: dict[str, Any]) -> bool:
        """
        Update objective based on game events.

        Args:
            event_type: Type of event that occurred
            event_data: Event-specific data

        Returns:
            True if the event affected this objective
        """
        ...

    def _notify_progress(self) -> None:
        """Notify progress callback."""
        if self.on_progress:
            self.on_progress(self, self.progress)


@dataclass
class KillObjective(Objective):
    """
    Objective to kill a certain number of enemies.

    Tracks kills by target type, with optional location and weapon requirements.
    """

    target_type: str = ""  # Enemy type to kill
    current: int = 0
    required: int = 1
    location: str | None = None  # Optional location requirement
    weapon_type: str | None = None  # Optional weapon requirement
    kill_streak: int = 0  # For kill streak tracking
    require_streak: int = 0  # Required kill streak

    def __post_init__(self) -> None:
        super().__post_init__()
        self.objective_type = ObjectiveType.KILL
        if self.required <= 0:
            raise ValueError("required must be > 0")
        if not self.target_type:
            raise ValueError("target_type cannot be empty")

    @property
    def progress(self) -> float:
        return min(1.0, self.current / self.required)

    @property
    def progress_text(self) -> str:
        return f"{self.current}/{self.required}"

    def update(self, event_type: str, event_data: dict[str, Any]) -> bool:
        if event_type != "kill":
            return False
        if self.state != ObjectiveState.IN_PROGRESS:
            return False

        # Check target type
        if event_data.get("target_type") != self.target_type:
            self.kill_streak = 0
            return False

        # Check location if required
        if self.location and event_data.get("location") != self.location:
            self.kill_streak = 0
            return False

        # Check weapon if required
        if self.weapon_type and event_data.get("weapon_type") != self.weapon_type:
            self.kill_streak = 0
            return False

        self.current += event_data.get("count", 1)
        self.kill_streak += 1
        self._notify_progress()

        # Check streak requirement
        if self.require_streak > 0 and self.kill_streak < self.require_streak:
            return True

        if self.current >= self.required:
            self.complete()

        return True

    def add_kill(self, count: int = 1) -> None:
        """Manually add kills."""
        self.current += count
        self._notify_progress()
        if self.current >= self.required:
            self.complete()


@dataclass
class CollectObjective(Objective):
    """
    Objective to collect items.

    Tracks item collection with optional source requirements.
    """

    item_id: str = ""
    current: int = 0
    required: int = 1
    source_type: str | None = None  # Optional source (loot, harvest, etc.)
    auto_remove: bool = True  # Remove items when quest completes

    def __post_init__(self) -> None:
        super().__post_init__()
        self.objective_type = ObjectiveType.COLLECT
        if self.required <= 0:
            raise ValueError("required must be > 0")
        if not self.item_id:
            raise ValueError("item_id cannot be empty")

    @property
    def progress(self) -> float:
        return min(1.0, self.current / self.required)

    @property
    def progress_text(self) -> str:
        return f"{self.current}/{self.required}"

    def update(self, event_type: str, event_data: dict[str, Any]) -> bool:
        if event_type != "collect":
            return False
        if self.state != ObjectiveState.IN_PROGRESS:
            return False

        if event_data.get("item_id") != self.item_id:
            return False

        if self.source_type and event_data.get("source_type") != self.source_type:
            return False

        self.current += event_data.get("count", 1)
        self._notify_progress()

        if self.current >= self.required:
            self.complete()

        return True

    def add_items(self, count: int = 1) -> None:
        """Manually add collected items."""
        self.current += count
        self._notify_progress()
        if self.current >= self.required:
            self.complete()

    def remove_items(self, count: int = 1) -> None:
        """Remove collected items (if player loses them)."""
        self.current = max(0, self.current - count)
        self._notify_progress()
        # Re-activate if we were complete
        if self.state == ObjectiveState.COMPLETE and self.current < self.required:
            self.state = ObjectiveState.IN_PROGRESS


@dataclass
class TalkObjective(Objective):
    """
    Objective to talk to an NPC.

    Simple flag-based objective.
    """

    npc_id: str = ""
    talked: bool = False
    dialogue_id: str | None = None  # Optional specific dialogue

    def __post_init__(self) -> None:
        super().__post_init__()
        self.objective_type = ObjectiveType.TALK
        if not self.npc_id:
            raise ValueError("npc_id cannot be empty")

    @property
    def progress(self) -> float:
        return 1.0 if self.talked else 0.0

    @property
    def progress_text(self) -> str:
        return "Complete" if self.talked else "Incomplete"

    def update(self, event_type: str, event_data: dict[str, Any]) -> bool:
        if event_type != "talk":
            return False
        if self.state != ObjectiveState.IN_PROGRESS:
            return False

        if event_data.get("npc_id") != self.npc_id:
            return False

        if self.dialogue_id and event_data.get("dialogue_id") != self.dialogue_id:
            return False

        self.talked = True
        self._notify_progress()
        self.complete()
        return True

    def mark_talked(self) -> None:
        """Manually mark as talked."""
        self.talked = True
        self._notify_progress()
        self.complete()


@dataclass
class ReachObjective(Objective):
    """
    Objective to reach a location.

    Tracks player entering a specific area.
    """

    location_id: str = ""
    reached: bool = False
    radius: float = DEFAULT_REACH_RADIUS  # Detection radius
    stay_duration: float = 0.0  # How long to stay (0 = instant)
    time_in_area: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.objective_type = ObjectiveType.REACH
        if not self.location_id:
            raise ValueError("location_id cannot be empty")
        if self.radius <= 0:
            raise ValueError("radius must be > 0")
        if self.stay_duration < 0:
            raise ValueError("stay_duration must be >= 0")

    @property
    def progress(self) -> float:
        if self.stay_duration > 0:
            return min(1.0, self.time_in_area / self.stay_duration)
        return 1.0 if self.reached else 0.0

    @property
    def progress_text(self) -> str:
        if self.stay_duration > 0 and not self.reached:
            return f"{self.time_in_area:.1f}/{self.stay_duration:.1f}s"
        return "Reached" if self.reached else "Not reached"

    def update(self, event_type: str, event_data: dict[str, Any]) -> bool:
        if self.state != ObjectiveState.IN_PROGRESS:
            return False

        if event_type == "enter_location":
            if event_data.get("location_id") != self.location_id:
                return False

            if self.stay_duration <= 0:
                self.reached = True
                self._notify_progress()
                self.complete()
                return True

            self.time_in_area = 0.0
            return True

        elif event_type == "location_tick":
            if event_data.get("location_id") != self.location_id:
                self.time_in_area = 0.0
                return False

            self.time_in_area += event_data.get("delta_time", 0.0)
            self._notify_progress()

            if self.time_in_area >= self.stay_duration:
                self.reached = True
                self.complete()

            return True

        elif event_type == "leave_location":
            if event_data.get("location_id") == self.location_id:
                self.time_in_area = 0.0
                return True

        return False

    def mark_reached(self) -> None:
        """Manually mark as reached."""
        self.reached = True
        self._notify_progress()
        self.complete()


@dataclass
class EscortObjective(Objective):
    """
    Objective to escort an NPC.

    Tracks NPC arrival at destination while keeping them alive.
    """

    npc_id: str = ""
    destination_id: str = ""
    escorted: bool = False
    npc_health_percent: float = DEFAULT_HEALTH_PERCENT
    min_health_percent: float = MIN_HEALTH_PERCENT_DEFAULT  # Fail if below this
    distance_threshold: float = DEFAULT_ESCORT_DISTANCE_THRESHOLD  # Max distance from player

    def __post_init__(self) -> None:
        super().__post_init__()
        self.objective_type = ObjectiveType.ESCORT
        if not self.npc_id:
            raise ValueError("npc_id cannot be empty")
        if not self.destination_id:
            raise ValueError("destination_id cannot be empty")

    @property
    def progress(self) -> float:
        return 1.0 if self.escorted else 0.0

    @property
    def progress_text(self) -> str:
        if self.escorted:
            return "Complete"
        return f"Escort {self.npc_id} - Health: {self.npc_health_percent:.0f}%"

    def update(self, event_type: str, event_data: dict[str, Any]) -> bool:
        if self.state != ObjectiveState.IN_PROGRESS:
            return False

        if event_type == "npc_arrived":
            if event_data.get("npc_id") != self.npc_id:
                return False
            if event_data.get("destination_id") != self.destination_id:
                return False
            self.escorted = True
            self._notify_progress()
            self.complete()
            return True

        elif event_type == "npc_damaged":
            if event_data.get("npc_id") != self.npc_id:
                return False
            self.npc_health_percent = event_data.get("health_percent", 100.0)
            self._notify_progress()
            if self.npc_health_percent <= self.min_health_percent:
                self.fail()
            return True

        elif event_type == "npc_died":
            if event_data.get("npc_id") != self.npc_id:
                return False
            self.npc_health_percent = 0.0
            self.fail()
            return True

        elif event_type == "npc_too_far":
            if event_data.get("npc_id") != self.npc_id:
                return False
            distance = event_data.get("distance", 0.0)
            if distance > self.distance_threshold:
                self.fail()
                return True

        return False


@dataclass
class InteractObjective(Objective):
    """
    Objective to interact with an object.

    Tracks interactions with world objects.
    """

    object_id: str = ""
    interaction_type: str = "use"  # use, activate, examine, etc.
    interacted: bool = False
    times_required: int = 1
    times_interacted: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.objective_type = ObjectiveType.INTERACT
        if not self.object_id:
            raise ValueError("object_id cannot be empty")
        if self.times_required <= 0:
            raise ValueError("times_required must be > 0")

    @property
    def progress(self) -> float:
        return min(1.0, self.times_interacted / self.times_required)

    @property
    def progress_text(self) -> str:
        if self.times_required == 1:
            return "Done" if self.interacted else "Incomplete"
        return f"{self.times_interacted}/{self.times_required}"

    def update(self, event_type: str, event_data: dict[str, Any]) -> bool:
        if event_type != "interact":
            return False
        if self.state != ObjectiveState.IN_PROGRESS:
            return False

        if event_data.get("object_id") != self.object_id:
            return False

        if event_data.get("interaction_type", "use") != self.interaction_type:
            return False

        self.times_interacted += 1
        self._notify_progress()

        if self.times_interacted >= self.times_required:
            self.interacted = True
            self.complete()

        return True


@dataclass
class UseObjective(Objective):
    """
    Objective to use an item or ability.
    """

    item_or_ability_id: str = ""
    target_type: str | None = None  # Optional target requirement
    times_required: int = 1
    times_used: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.objective_type = ObjectiveType.USE
        if not self.item_or_ability_id:
            raise ValueError("item_or_ability_id cannot be empty")
        if self.times_required <= 0:
            raise ValueError("times_required must be > 0")

    @property
    def progress(self) -> float:
        return min(1.0, self.times_used / self.times_required)

    @property
    def progress_text(self) -> str:
        return f"{self.times_used}/{self.times_required}"

    def update(self, event_type: str, event_data: dict[str, Any]) -> bool:
        if event_type != "use":
            return False
        if self.state != ObjectiveState.IN_PROGRESS:
            return False

        if event_data.get("id") != self.item_or_ability_id:
            return False

        if self.target_type and event_data.get("target_type") != self.target_type:
            return False

        self.times_used += 1
        self._notify_progress()

        if self.times_used >= self.times_required:
            self.complete()

        return True


@dataclass
class CraftObjective(Objective):
    """
    Objective to craft items.
    """

    item_id: str = ""
    current: int = 0
    required: int = 1
    recipe_id: str | None = None  # Optional specific recipe

    def __post_init__(self) -> None:
        super().__post_init__()
        self.objective_type = ObjectiveType.CRAFT
        if not self.item_id:
            raise ValueError("item_id cannot be empty")
        if self.required <= 0:
            raise ValueError("required must be > 0")

    @property
    def progress(self) -> float:
        return min(1.0, self.current / self.required)

    @property
    def progress_text(self) -> str:
        return f"{self.current}/{self.required}"

    def update(self, event_type: str, event_data: dict[str, Any]) -> bool:
        if event_type != "craft":
            return False
        if self.state != ObjectiveState.IN_PROGRESS:
            return False

        if event_data.get("item_id") != self.item_id:
            return False

        if self.recipe_id and event_data.get("recipe_id") != self.recipe_id:
            return False

        self.current += event_data.get("count", 1)
        self._notify_progress()

        if self.current >= self.required:
            self.complete()

        return True


@dataclass
class DefendObjective(Objective):
    """
    Objective to defend a location or NPC for a duration.
    """

    target_id: str = ""  # Location or NPC to defend
    target_type: str = "location"  # location or npc
    duration: float = DEFAULT_DEFEND_DURATION  # Seconds to defend
    time_defended: float = 0.0
    target_health_percent: float = DEFAULT_TARGET_HEALTH_PERCENT
    min_health_percent: float = MIN_HEALTH_PERCENT_DEFAULT

    def __post_init__(self) -> None:
        super().__post_init__()
        self.objective_type = ObjectiveType.DEFEND
        if not self.target_id:
            raise ValueError("target_id cannot be empty")
        if self.duration <= 0:
            raise ValueError("duration must be > 0")

    @property
    def progress(self) -> float:
        return min(1.0, self.time_defended / self.duration)

    @property
    def progress_text(self) -> str:
        remaining = max(0, self.duration - self.time_defended)
        return f"{remaining:.0f}s remaining"

    def update(self, event_type: str, event_data: dict[str, Any]) -> bool:
        if self.state != ObjectiveState.IN_PROGRESS:
            return False

        if event_type == "defend_tick":
            if event_data.get("target_id") != self.target_id:
                return False

            self.time_defended += event_data.get("delta_time", 0.0)
            self._notify_progress()

            if self.time_defended >= self.duration:
                self.complete()

            return True

        elif event_type == "target_damaged":
            if event_data.get("target_id") != self.target_id:
                return False

            self.target_health_percent = event_data.get("health_percent", 100.0)

            if self.target_health_percent <= self.min_health_percent:
                self.fail()

            return True

        elif event_type == "target_destroyed":
            if event_data.get("target_id") != self.target_id:
                return False
            self.target_health_percent = 0.0
            self.fail()
            return True

        return False


@dataclass
class TimedObjective(Objective):
    """
    Wrapper for adding time limits to other objectives.
    """

    inner_objective: Objective | None = None
    time_limit: float = DEFAULT_TIMED_OBJECTIVE_LIMIT
    time_elapsed: float = 0.0
    fail_on_timeout: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        self.objective_type = ObjectiveType.TIMED
        if self.time_limit <= 0:
            raise ValueError("time_limit must be > 0")

    @property
    def progress(self) -> float:
        if self.inner_objective:
            return self.inner_objective.progress
        return 0.0

    @property
    def progress_text(self) -> str:
        remaining = max(0, self.time_limit - self.time_elapsed)
        inner_text = self.inner_objective.progress_text if self.inner_objective else ""
        return f"{inner_text} ({remaining:.0f}s)"

    @property
    def time_remaining(self) -> float:
        return max(0, self.time_limit - self.time_elapsed)

    def update(self, event_type: str, event_data: dict[str, Any]) -> bool:
        if self.state != ObjectiveState.IN_PROGRESS:
            return False

        # Handle time updates
        if event_type == "time_tick":
            self.time_elapsed += event_data.get("delta_time", 0.0)
            self._notify_progress()

            if self.time_elapsed >= self.time_limit:
                if self.fail_on_timeout:
                    self.fail()
                return True

        # Forward to inner objective
        if self.inner_objective:
            result = self.inner_objective.update(event_type, event_data)
            if self.inner_objective.is_complete:
                self.complete()
            elif self.inner_objective.is_failed:
                self.fail()
            return result

        return False

    def set_inner(self, objective: Objective) -> None:
        """Set the inner objective."""
        self.inner_objective = objective


@dataclass
class CompositeObjective(Objective):
    """
    Composite objective containing multiple sub-objectives.

    Supports sequential (in order), parallel (any order), and branching modes.
    """

    objectives: list[Objective] = field(default_factory=list)
    mode: str = "all"  # all, any, sequential
    current_index: int = 0
    fail_on_any_failure: bool = True  # Fail composite if any sub-objective fails

    def __post_init__(self) -> None:
        super().__post_init__()
        self.objective_type = ObjectiveType.COMPOSITE
        if self.mode not in ("all", "any", "sequential"):
            raise ValueError("mode must be 'all', 'any', or 'sequential'")

    @property
    def progress(self) -> float:
        if not self.objectives:
            return 0.0

        if self.mode == "any":
            return max(obj.progress for obj in self.objectives)
        else:
            return sum(obj.progress for obj in self.objectives) / len(self.objectives)

    @property
    def progress_text(self) -> str:
        complete = sum(1 for obj in self.objectives if obj.is_complete)
        return f"{complete}/{len(self.objectives)}"

    def add_objective(self, objective: Objective) -> None:
        """Add a sub-objective."""
        self.objectives.append(objective)

    def activate(self) -> bool:
        """Activate the composite and its sub-objectives."""
        if not super().activate():
            return False

        if self.mode == "sequential":
            # Only activate first objective
            if self.objectives:
                self.objectives[0].activate()
        else:
            # Activate all objectives
            for obj in self.objectives:
                obj.activate()

        return True

    def update(self, event_type: str, event_data: dict[str, Any]) -> bool:
        if self.state != ObjectiveState.IN_PROGRESS:
            return False

        affected = False

        if self.mode == "sequential":
            # Only update current objective
            if self.current_index < len(self.objectives):
                current = self.objectives[self.current_index]
                affected = current.update(event_type, event_data)

                if current.is_complete:
                    self.current_index += 1
                    if self.current_index < len(self.objectives):
                        self.objectives[self.current_index].activate()
                    else:
                        self.complete()
                elif current.is_failed:
                    self.fail()

        elif self.mode == "any":
            # Update all, complete if any is complete
            for obj in self.objectives:
                if obj.update(event_type, event_data):
                    affected = True
                if obj.is_complete:
                    self.complete()
                    break

        else:  # all
            # Update all, complete if all complete
            for obj in self.objectives:
                if obj.update(event_type, event_data):
                    affected = True

            if all(obj.is_complete for obj in self.objectives):
                self.complete()
            elif self.fail_on_any_failure and any(obj.is_failed for obj in self.objectives):
                self.fail()

        if affected:
            self._notify_progress()

        return affected


class ObjectiveFactory:
    """
    Factory for creating objectives from configuration data.
    """

    _creators: dict[str, type[Objective]] = {
        "kill": KillObjective,
        "collect": CollectObjective,
        "talk": TalkObjective,
        "reach": ReachObjective,
        "escort": EscortObjective,
        "interact": InteractObjective,
        "use": UseObjective,
        "craft": CraftObjective,
        "defend": DefendObjective,
        "timed": TimedObjective,
        "composite": CompositeObjective,
    }

    @classmethod
    def register(cls, name: str, objective_class: type[Objective]) -> None:
        """Register a custom objective type."""
        cls._creators[name] = objective_class

    @classmethod
    def create(cls, objective_type: str, **kwargs: Any) -> Objective:
        """Create an objective from type string and parameters."""
        if objective_type not in cls._creators:
            raise ValueError(f"Unknown objective type: {objective_type}")

        return cls._creators[objective_type](**kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Objective:
        """Create an objective from a dictionary configuration."""
        obj_type = data.pop("type", None)
        if not obj_type:
            raise ValueError("Objective data must include 'type'")

        return cls.create(obj_type, **data)
