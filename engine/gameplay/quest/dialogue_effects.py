"""
Dialogue Effect System.

Provides effects that can be triggered from dialogue nodes, including:
- Set variable (local, global, quest-linked)
- Give/take item (inventory manipulation)
- Update quest (state, progress)
- Change reputation (faction standing)
- Trigger game events
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

from .constants import (
    EffectPriority,
    EffectType,
    QuestState,
    MIN_REPUTATION,
    MAX_REPUTATION,
)
from .dialogue_variables import VariableScope, VariableValue


# =============================================================================
# Effect Context Protocol
# =============================================================================

class EffectContext(Protocol):
    """Protocol for effect execution context."""

    def set_variable(
        self,
        name: str,
        value: VariableValue,
        scope: VariableScope
    ) -> None:
        """Set a dialogue variable."""
        ...

    def get_variable(
        self,
        name: str,
        scope: Optional[VariableScope] = None,
        default: VariableValue = None
    ) -> VariableValue:
        """Get a dialogue variable."""
        ...

    def give_item(self, item_id: str, count: int = 1) -> bool:
        """Give item(s) to the player."""
        ...

    def take_item(self, item_id: str, count: int = 1) -> bool:
        """Take item(s) from the player."""
        ...

    def update_quest_state(self, quest_id: str, state: QuestState) -> None:
        """Update a quest's state."""
        ...

    def update_quest_progress(self, quest_id: str, progress: float) -> None:
        """Update a quest's progress (0.0 to 1.0)."""
        ...

    def change_reputation(self, faction_id: str, amount: int) -> int:
        """Change reputation with a faction, returns new value."""
        ...

    def trigger_event(self, event_name: str, data: Dict[str, Any]) -> None:
        """Trigger a game event."""
        ...

    def play_sound(self, sound_id: str) -> None:
        """Play a sound effect."""
        ...

    def play_animation(self, actor_id: str, animation_id: str) -> None:
        """Play an animation on an actor."""
        ...


# =============================================================================
# Effect Result
# =============================================================================

@dataclass
class EffectResult:
    """Result of executing an effect."""
    success: bool
    effect_type: EffectType
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    rollback_data: Optional[Dict[str, Any]] = None


# =============================================================================
# Effect Base Class
# =============================================================================

class Effect(ABC):
    """
    Abstract base class for dialogue effects.

    Effects modify game state when dialogue nodes are executed.
    """

    @property
    @abstractmethod
    def effect_type(self) -> EffectType:
        """Get the effect type."""
        pass

    @property
    def priority(self) -> EffectPriority:
        """Get the effect priority (default: NORMAL)."""
        return EffectPriority.NORMAL

    @abstractmethod
    def execute(self, context: EffectContext) -> EffectResult:
        """
        Execute the effect.

        Args:
            context: The effect execution context.

        Returns:
            EffectResult with success status and details.
        """
        pass

    def rollback(
        self,
        context: EffectContext,
        rollback_data: Dict[str, Any]
    ) -> bool:
        """
        Rollback the effect using stored data.

        Args:
            context: The effect execution context.
            rollback_data: Data from execute result.

        Returns:
            True if rollback succeeded.
        """
        # Default implementation does nothing
        return False

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the effect to a dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Effect":
        """Deserialize an effect from a dictionary."""
        pass


# =============================================================================
# Variable Effects
# =============================================================================

@dataclass
class SetVariableEffect(Effect):
    """Effect that sets a variable value."""
    variable_name: str
    value: VariableValue
    scope: VariableScope = VariableScope.LOCAL

    def __post_init__(self):
        """Validate parameters."""
        if not self.variable_name:
            raise ValueError("variable_name cannot be empty")

    @property
    def effect_type(self) -> EffectType:
        return EffectType.SET_VARIABLE

    def execute(self, context: EffectContext) -> EffectResult:
        """Execute the set variable effect."""
        try:
            # Store old value for rollback
            old_value = context.get_variable(
                self.variable_name,
                scope=self.scope
            )

            context.set_variable(
                self.variable_name,
                self.value,
                self.scope
            )

            return EffectResult(
                success=True,
                effect_type=self.effect_type,
                message=f"Set {self.variable_name} = {self.value}",
                details={
                    "variable": self.variable_name,
                    "old_value": old_value,
                    "new_value": self.value,
                    "scope": self.scope.name
                },
                rollback_data={
                    "variable": self.variable_name,
                    "old_value": old_value,
                    "scope": self.scope.name
                }
            )
        except Exception as e:
            return EffectResult(
                success=False,
                effect_type=self.effect_type,
                message=f"Failed to set variable: {e}",
                details={"error": str(e)}
            )

    def rollback(
        self,
        context: EffectContext,
        rollback_data: Dict[str, Any]
    ) -> bool:
        """Rollback by restoring the old value."""
        try:
            context.set_variable(
                rollback_data["variable"],
                rollback_data["old_value"],
                VariableScope[rollback_data["scope"]]
            )
            return True
        except Exception:
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "set_variable",
            "variable_name": self.variable_name,
            "value": self.value,
            "scope": self.scope.name
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SetVariableEffect":
        """Deserialize from dictionary."""
        return cls(
            variable_name=data["variable_name"],
            value=data["value"],
            scope=VariableScope[data.get("scope", "LOCAL")]
        )


@dataclass
class IncrementVariableEffect(Effect):
    """Effect that increments a numeric variable."""
    variable_name: str
    amount: Union[int, float] = 1
    scope: VariableScope = VariableScope.LOCAL

    def __post_init__(self):
        """Validate parameters."""
        if not self.variable_name:
            raise ValueError("variable_name cannot be empty")

    @property
    def effect_type(self) -> EffectType:
        return EffectType.INCREMENT_VARIABLE

    def execute(self, context: EffectContext) -> EffectResult:
        """Execute the increment effect."""
        try:
            old_value = context.get_variable(
                self.variable_name,
                scope=self.scope,
                default=0
            )

            if not isinstance(old_value, (int, float)):
                return EffectResult(
                    success=False,
                    effect_type=self.effect_type,
                    message=f"Cannot increment non-numeric variable: {self.variable_name}",
                    details={"current_value": old_value, "type": type(old_value).__name__}
                )

            new_value = old_value + self.amount
            context.set_variable(self.variable_name, new_value, self.scope)

            return EffectResult(
                success=True,
                effect_type=self.effect_type,
                message=f"Incremented {self.variable_name} by {self.amount}",
                details={
                    "variable": self.variable_name,
                    "old_value": old_value,
                    "new_value": new_value,
                    "amount": self.amount
                },
                rollback_data={
                    "variable": self.variable_name,
                    "old_value": old_value,
                    "scope": self.scope.name
                }
            )
        except Exception as e:
            return EffectResult(
                success=False,
                effect_type=self.effect_type,
                message=f"Failed to increment variable: {e}",
                details={"error": str(e)}
            )

    def rollback(
        self,
        context: EffectContext,
        rollback_data: Dict[str, Any]
    ) -> bool:
        """Rollback by restoring the old value."""
        try:
            context.set_variable(
                rollback_data["variable"],
                rollback_data["old_value"],
                VariableScope[rollback_data["scope"]]
            )
            return True
        except Exception:
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "increment_variable",
            "variable_name": self.variable_name,
            "amount": self.amount,
            "scope": self.scope.name
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IncrementVariableEffect":
        """Deserialize from dictionary."""
        return cls(
            variable_name=data["variable_name"],
            amount=data.get("amount", 1),
            scope=VariableScope[data.get("scope", "LOCAL")]
        )


@dataclass
class DecrementVariableEffect(Effect):
    """Effect that decrements a numeric variable."""
    variable_name: str
    amount: Union[int, float] = 1
    scope: VariableScope = VariableScope.LOCAL

    def __post_init__(self):
        """Validate parameters."""
        if not self.variable_name:
            raise ValueError("variable_name cannot be empty")

    @property
    def effect_type(self) -> EffectType:
        return EffectType.DECREMENT_VARIABLE

    def execute(self, context: EffectContext) -> EffectResult:
        """Execute the decrement effect."""
        # Reuse increment with negative amount
        increment = IncrementVariableEffect(
            self.variable_name,
            -self.amount,
            self.scope
        )
        result = increment.execute(context)
        # Update effect type in result
        result.effect_type = self.effect_type
        return result

    def rollback(
        self,
        context: EffectContext,
        rollback_data: Dict[str, Any]
    ) -> bool:
        """Rollback by restoring the old value."""
        try:
            context.set_variable(
                rollback_data["variable"],
                rollback_data["old_value"],
                VariableScope[rollback_data["scope"]]
            )
            return True
        except Exception:
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "decrement_variable",
            "variable_name": self.variable_name,
            "amount": self.amount,
            "scope": self.scope.name
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecrementVariableEffect":
        """Deserialize from dictionary."""
        return cls(
            variable_name=data["variable_name"],
            amount=data.get("amount", 1),
            scope=VariableScope[data.get("scope", "LOCAL")]
        )


# =============================================================================
# Item Effects
# =============================================================================

@dataclass
class GiveItemEffect(Effect):
    """Effect that gives item(s) to the player."""
    item_id: str
    count: int = 1

    def __post_init__(self):
        """Validate parameters."""
        if not self.item_id:
            raise ValueError("item_id cannot be empty")
        if self.count < 1:
            raise ValueError("count must be >= 1")

    @property
    def effect_type(self) -> EffectType:
        return EffectType.GIVE_ITEM

    @property
    def priority(self) -> EffectPriority:
        return EffectPriority.HIGH

    def execute(self, context: EffectContext) -> EffectResult:
        """Execute the give item effect."""
        try:
            success = context.give_item(self.item_id, self.count)

            if success:
                return EffectResult(
                    success=True,
                    effect_type=self.effect_type,
                    message=f"Gave {self.count}x {self.item_id}",
                    details={
                        "item_id": self.item_id,
                        "count": self.count
                    },
                    rollback_data={
                        "item_id": self.item_id,
                        "count": self.count
                    }
                )
            else:
                return EffectResult(
                    success=False,
                    effect_type=self.effect_type,
                    message=f"Failed to give item: inventory full or invalid item",
                    details={
                        "item_id": self.item_id,
                        "count": self.count
                    }
                )
        except Exception as e:
            return EffectResult(
                success=False,
                effect_type=self.effect_type,
                message=f"Failed to give item: {e}",
                details={"error": str(e)}
            )

    def rollback(
        self,
        context: EffectContext,
        rollback_data: Dict[str, Any]
    ) -> bool:
        """Rollback by taking the item back."""
        try:
            return context.take_item(
                rollback_data["item_id"],
                rollback_data["count"]
            )
        except Exception:
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "give_item",
            "item_id": self.item_id,
            "count": self.count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GiveItemEffect":
        """Deserialize from dictionary."""
        return cls(
            item_id=data["item_id"],
            count=data.get("count", 1)
        )


@dataclass
class TakeItemEffect(Effect):
    """Effect that takes item(s) from the player."""
    item_id: str
    count: int = 1
    required: bool = True  # If True, effect fails if player doesn't have items

    def __post_init__(self):
        """Validate parameters."""
        if not self.item_id:
            raise ValueError("item_id cannot be empty")
        if self.count < 1:
            raise ValueError("count must be >= 1")

    @property
    def effect_type(self) -> EffectType:
        return EffectType.TAKE_ITEM

    @property
    def priority(self) -> EffectPriority:
        return EffectPriority.HIGH

    def execute(self, context: EffectContext) -> EffectResult:
        """Execute the take item effect."""
        try:
            success = context.take_item(self.item_id, self.count)

            if success:
                return EffectResult(
                    success=True,
                    effect_type=self.effect_type,
                    message=f"Took {self.count}x {self.item_id}",
                    details={
                        "item_id": self.item_id,
                        "count": self.count
                    },
                    rollback_data={
                        "item_id": self.item_id,
                        "count": self.count
                    }
                )
            elif not self.required:
                return EffectResult(
                    success=True,  # Still succeeds if not required
                    effect_type=self.effect_type,
                    message=f"Could not take {self.count}x {self.item_id} (not required)",
                    details={
                        "item_id": self.item_id,
                        "count": self.count,
                        "items_taken": False
                    }
                )
            else:
                return EffectResult(
                    success=False,
                    effect_type=self.effect_type,
                    message=f"Player doesn't have {self.count}x {self.item_id}",
                    details={
                        "item_id": self.item_id,
                        "count": self.count
                    }
                )
        except Exception as e:
            return EffectResult(
                success=False,
                effect_type=self.effect_type,
                message=f"Failed to take item: {e}",
                details={"error": str(e)}
            )

    def rollback(
        self,
        context: EffectContext,
        rollback_data: Dict[str, Any]
    ) -> bool:
        """Rollback by giving the item back."""
        try:
            return context.give_item(
                rollback_data["item_id"],
                rollback_data["count"]
            )
        except Exception:
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "take_item",
            "item_id": self.item_id,
            "count": self.count,
            "required": self.required
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TakeItemEffect":
        """Deserialize from dictionary."""
        return cls(
            item_id=data["item_id"],
            count=data.get("count", 1),
            required=data.get("required", True)
        )


# =============================================================================
# Quest Effects
# =============================================================================

@dataclass
class UpdateQuestStateEffect(Effect):
    """Effect that updates a quest's state."""
    quest_id: str
    new_state: QuestState

    def __post_init__(self):
        """Validate parameters."""
        if not self.quest_id:
            raise ValueError("quest_id cannot be empty")

    @property
    def effect_type(self) -> EffectType:
        return EffectType.UPDATE_QUEST

    @property
    def priority(self) -> EffectPriority:
        return EffectPriority.HIGH

    def execute(self, context: EffectContext) -> EffectResult:
        """Execute the quest state update."""
        try:
            context.update_quest_state(self.quest_id, self.new_state)

            return EffectResult(
                success=True,
                effect_type=self.effect_type,
                message=f"Quest '{self.quest_id}' -> {self.new_state.name}",
                details={
                    "quest_id": self.quest_id,
                    "new_state": self.new_state.name
                }
            )
        except Exception as e:
            return EffectResult(
                success=False,
                effect_type=self.effect_type,
                message=f"Failed to update quest: {e}",
                details={"error": str(e)}
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "update_quest_state",
            "quest_id": self.quest_id,
            "new_state": self.new_state.name
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UpdateQuestStateEffect":
        """Deserialize from dictionary."""
        return cls(
            quest_id=data["quest_id"],
            new_state=QuestState[data["new_state"]]
        )


@dataclass
class UpdateQuestProgressEffect(Effect):
    """Effect that updates a quest's progress."""
    quest_id: str
    progress: float
    relative: bool = False  # If True, adds to current progress

    def __post_init__(self):
        """Validate parameters."""
        if not self.quest_id:
            raise ValueError("quest_id cannot be empty")
        if not self.relative and not 0.0 <= self.progress <= 1.0:
            raise ValueError("progress must be between 0.0 and 1.0 for absolute mode")

    @property
    def effect_type(self) -> EffectType:
        return EffectType.UPDATE_QUEST

    def execute(self, context: EffectContext) -> EffectResult:
        """Execute the quest progress update."""
        try:
            final_progress = self.progress
            if self.relative:
                # For relative mode, we'd need to get current progress
                # This is a simplification - real implementation would query current
                final_progress = min(1.0, max(0.0, self.progress))

            context.update_quest_progress(self.quest_id, final_progress)

            return EffectResult(
                success=True,
                effect_type=self.effect_type,
                message=f"Quest '{self.quest_id}' progress -> {final_progress:.1%}",
                details={
                    "quest_id": self.quest_id,
                    "progress": final_progress,
                    "relative": self.relative
                }
            )
        except Exception as e:
            return EffectResult(
                success=False,
                effect_type=self.effect_type,
                message=f"Failed to update quest progress: {e}",
                details={"error": str(e)}
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "update_quest_progress",
            "quest_id": self.quest_id,
            "progress": self.progress,
            "relative": self.relative
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UpdateQuestProgressEffect":
        """Deserialize from dictionary."""
        return cls(
            quest_id=data["quest_id"],
            progress=data["progress"],
            relative=data.get("relative", False)
        )


@dataclass
class StartQuestEffect(Effect):
    """Convenience effect to start a quest."""
    quest_id: str

    def __post_init__(self):
        """Validate parameters."""
        if not self.quest_id:
            raise ValueError("quest_id cannot be empty")

    @property
    def effect_type(self) -> EffectType:
        return EffectType.UPDATE_QUEST

    @property
    def priority(self) -> EffectPriority:
        return EffectPriority.HIGH

    def execute(self, context: EffectContext) -> EffectResult:
        """Start the quest."""
        try:
            context.update_quest_state(self.quest_id, QuestState.IN_PROGRESS)

            return EffectResult(
                success=True,
                effect_type=self.effect_type,
                message=f"Started quest '{self.quest_id}'",
                details={
                    "quest_id": self.quest_id,
                    "new_state": QuestState.IN_PROGRESS.name
                }
            )
        except Exception as e:
            return EffectResult(
                success=False,
                effect_type=self.effect_type,
                message=f"Failed to start quest: {e}",
                details={"error": str(e)}
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "start_quest",
            "quest_id": self.quest_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StartQuestEffect":
        """Deserialize from dictionary."""
        return cls(quest_id=data["quest_id"])


@dataclass
class CompleteQuestEffect(Effect):
    """Convenience effect to complete a quest."""
    quest_id: str

    def __post_init__(self):
        """Validate parameters."""
        if not self.quest_id:
            raise ValueError("quest_id cannot be empty")

    @property
    def effect_type(self) -> EffectType:
        return EffectType.UPDATE_QUEST

    @property
    def priority(self) -> EffectPriority:
        return EffectPriority.HIGH

    def execute(self, context: EffectContext) -> EffectResult:
        """Complete the quest."""
        try:
            context.update_quest_state(self.quest_id, QuestState.COMPLETED)

            return EffectResult(
                success=True,
                effect_type=self.effect_type,
                message=f"Completed quest '{self.quest_id}'",
                details={
                    "quest_id": self.quest_id,
                    "new_state": QuestState.COMPLETED.name
                }
            )
        except Exception as e:
            return EffectResult(
                success=False,
                effect_type=self.effect_type,
                message=f"Failed to complete quest: {e}",
                details={"error": str(e)}
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "complete_quest",
            "quest_id": self.quest_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompleteQuestEffect":
        """Deserialize from dictionary."""
        return cls(quest_id=data["quest_id"])


@dataclass
class FailQuestEffect(Effect):
    """Convenience effect to fail a quest."""
    quest_id: str

    def __post_init__(self):
        """Validate parameters."""
        if not self.quest_id:
            raise ValueError("quest_id cannot be empty")

    @property
    def effect_type(self) -> EffectType:
        return EffectType.UPDATE_QUEST

    @property
    def priority(self) -> EffectPriority:
        return EffectPriority.HIGH

    def execute(self, context: EffectContext) -> EffectResult:
        """Fail the quest."""
        try:
            context.update_quest_state(self.quest_id, QuestState.FAILED)

            return EffectResult(
                success=True,
                effect_type=self.effect_type,
                message=f"Failed quest '{self.quest_id}'",
                details={
                    "quest_id": self.quest_id,
                    "new_state": QuestState.FAILED.name
                }
            )
        except Exception as e:
            return EffectResult(
                success=False,
                effect_type=self.effect_type,
                message=f"Failed to fail quest: {e}",
                details={"error": str(e)}
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "fail_quest",
            "quest_id": self.quest_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailQuestEffect":
        """Deserialize from dictionary."""
        return cls(quest_id=data["quest_id"])


# =============================================================================
# Reputation Effects
# =============================================================================

@dataclass
class ChangeReputationEffect(Effect):
    """Effect that changes reputation with a faction."""
    faction_id: str
    amount: int
    clamp: bool = True  # Clamp to MIN/MAX_REPUTATION

    def __post_init__(self):
        """Validate parameters."""
        if not self.faction_id:
            raise ValueError("faction_id cannot be empty")

    @property
    def effect_type(self) -> EffectType:
        return EffectType.CHANGE_REPUTATION

    def execute(self, context: EffectContext) -> EffectResult:
        """Execute the reputation change."""
        try:
            new_reputation = context.change_reputation(
                self.faction_id,
                self.amount
            )

            if self.clamp:
                new_reputation = max(
                    MIN_REPUTATION,
                    min(MAX_REPUTATION, new_reputation)
                )

            return EffectResult(
                success=True,
                effect_type=self.effect_type,
                message=f"Reputation with '{self.faction_id}' "
                        f"{'increased' if self.amount >= 0 else 'decreased'} "
                        f"by {abs(self.amount)}",
                details={
                    "faction_id": self.faction_id,
                    "amount": self.amount,
                    "new_reputation": new_reputation
                },
                rollback_data={
                    "faction_id": self.faction_id,
                    "amount": -self.amount
                }
            )
        except Exception as e:
            return EffectResult(
                success=False,
                effect_type=self.effect_type,
                message=f"Failed to change reputation: {e}",
                details={"error": str(e)}
            )

    def rollback(
        self,
        context: EffectContext,
        rollback_data: Dict[str, Any]
    ) -> bool:
        """Rollback by reversing the change."""
        try:
            context.change_reputation(
                rollback_data["faction_id"],
                rollback_data["amount"]
            )
            return True
        except Exception:
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "change_reputation",
            "faction_id": self.faction_id,
            "amount": self.amount,
            "clamp": self.clamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChangeReputationEffect":
        """Deserialize from dictionary."""
        return cls(
            faction_id=data["faction_id"],
            amount=data["amount"],
            clamp=data.get("clamp", True)
        )


@dataclass
class SetReputationEffect(Effect):
    """Effect that sets reputation to a specific value."""
    faction_id: str
    value: int

    def __post_init__(self):
        """Validate parameters."""
        if not self.faction_id:
            raise ValueError("faction_id cannot be empty")
        if not MIN_REPUTATION <= self.value <= MAX_REPUTATION:
            raise ValueError(
                f"value must be between {MIN_REPUTATION} and {MAX_REPUTATION}"
            )

    @property
    def effect_type(self) -> EffectType:
        return EffectType.CHANGE_REPUTATION

    def execute(self, context: EffectContext) -> EffectResult:
        """Execute the reputation set."""
        try:
            # This is a simplification - real implementation would
            # calculate the delta needed
            context.change_reputation(self.faction_id, self.value)

            return EffectResult(
                success=True,
                effect_type=self.effect_type,
                message=f"Set reputation with '{self.faction_id}' to {self.value}",
                details={
                    "faction_id": self.faction_id,
                    "value": self.value
                }
            )
        except Exception as e:
            return EffectResult(
                success=False,
                effect_type=self.effect_type,
                message=f"Failed to set reputation: {e}",
                details={"error": str(e)}
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "set_reputation",
            "faction_id": self.faction_id,
            "value": self.value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SetReputationEffect":
        """Deserialize from dictionary."""
        return cls(
            faction_id=data["faction_id"],
            value=data["value"]
        )


# =============================================================================
# Event Effects
# =============================================================================

@dataclass
class TriggerEventEffect(Effect):
    """Effect that triggers a game event."""
    event_name: str
    event_data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate parameters."""
        if not self.event_name:
            raise ValueError("event_name cannot be empty")

    @property
    def effect_type(self) -> EffectType:
        return EffectType.TRIGGER_EVENT

    def execute(self, context: EffectContext) -> EffectResult:
        """Execute the event trigger."""
        try:
            context.trigger_event(self.event_name, self.event_data)

            return EffectResult(
                success=True,
                effect_type=self.effect_type,
                message=f"Triggered event '{self.event_name}'",
                details={
                    "event_name": self.event_name,
                    "event_data": self.event_data
                }
            )
        except Exception as e:
            return EffectResult(
                success=False,
                effect_type=self.effect_type,
                message=f"Failed to trigger event: {e}",
                details={"error": str(e)}
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "trigger_event",
            "event_name": self.event_name,
            "event_data": self.event_data
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TriggerEventEffect":
        """Deserialize from dictionary."""
        return cls(
            event_name=data["event_name"],
            event_data=data.get("event_data", {})
        )


@dataclass
class PlaySoundEffect(Effect):
    """Effect that plays a sound."""
    sound_id: str

    def __post_init__(self):
        """Validate parameters."""
        if not self.sound_id:
            raise ValueError("sound_id cannot be empty")

    @property
    def effect_type(self) -> EffectType:
        return EffectType.PLAY_SOUND

    @property
    def priority(self) -> EffectPriority:
        return EffectPriority.LOW

    def execute(self, context: EffectContext) -> EffectResult:
        """Execute the sound play."""
        try:
            context.play_sound(self.sound_id)

            return EffectResult(
                success=True,
                effect_type=self.effect_type,
                message=f"Playing sound '{self.sound_id}'",
                details={"sound_id": self.sound_id}
            )
        except Exception as e:
            return EffectResult(
                success=False,
                effect_type=self.effect_type,
                message=f"Failed to play sound: {e}",
                details={"error": str(e)}
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "play_sound",
            "sound_id": self.sound_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlaySoundEffect":
        """Deserialize from dictionary."""
        return cls(sound_id=data["sound_id"])


@dataclass
class PlayAnimationEffect(Effect):
    """Effect that plays an animation on an actor."""
    actor_id: str
    animation_id: str

    def __post_init__(self):
        """Validate parameters."""
        if not self.actor_id:
            raise ValueError("actor_id cannot be empty")
        if not self.animation_id:
            raise ValueError("animation_id cannot be empty")

    @property
    def effect_type(self) -> EffectType:
        return EffectType.PLAY_ANIMATION

    @property
    def priority(self) -> EffectPriority:
        return EffectPriority.LOW

    def execute(self, context: EffectContext) -> EffectResult:
        """Execute the animation play."""
        try:
            context.play_animation(self.actor_id, self.animation_id)

            return EffectResult(
                success=True,
                effect_type=self.effect_type,
                message=f"Playing animation '{self.animation_id}' on '{self.actor_id}'",
                details={
                    "actor_id": self.actor_id,
                    "animation_id": self.animation_id
                }
            )
        except Exception as e:
            return EffectResult(
                success=False,
                effect_type=self.effect_type,
                message=f"Failed to play animation: {e}",
                details={"error": str(e)}
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "play_animation",
            "actor_id": self.actor_id,
            "animation_id": self.animation_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlayAnimationEffect":
        """Deserialize from dictionary."""
        return cls(
            actor_id=data["actor_id"],
            animation_id=data["animation_id"]
        )


# =============================================================================
# Dialogue Control Effects
# =============================================================================

@dataclass
class StartDialogueEffect(Effect):
    """Effect that starts another dialogue."""
    dialogue_id: str
    entry_point: Optional[str] = None

    def __post_init__(self):
        """Validate parameters."""
        if not self.dialogue_id:
            raise ValueError("dialogue_id cannot be empty")

    @property
    def effect_type(self) -> EffectType:
        return EffectType.START_DIALOGUE

    def execute(self, context: EffectContext) -> EffectResult:
        """Execute the dialogue start."""
        try:
            event_data = {"dialogue_id": self.dialogue_id}
            if self.entry_point:
                event_data["entry_point"] = self.entry_point

            context.trigger_event("dialogue_start", event_data)

            return EffectResult(
                success=True,
                effect_type=self.effect_type,
                message=f"Starting dialogue '{self.dialogue_id}'",
                details=event_data
            )
        except Exception as e:
            return EffectResult(
                success=False,
                effect_type=self.effect_type,
                message=f"Failed to start dialogue: {e}",
                details={"error": str(e)}
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "type": "start_dialogue",
            "dialogue_id": self.dialogue_id
        }
        if self.entry_point:
            result["entry_point"] = self.entry_point
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StartDialogueEffect":
        """Deserialize from dictionary."""
        return cls(
            dialogue_id=data["dialogue_id"],
            entry_point=data.get("entry_point")
        )


class EndDialogueEffect(Effect):
    """Effect that ends the current dialogue."""

    @property
    def effect_type(self) -> EffectType:
        return EffectType.END_DIALOGUE

    def execute(self, context: EffectContext) -> EffectResult:
        """Execute the dialogue end."""
        try:
            context.trigger_event("dialogue_end", {})

            return EffectResult(
                success=True,
                effect_type=self.effect_type,
                message="Ending dialogue"
            )
        except Exception as e:
            return EffectResult(
                success=False,
                effect_type=self.effect_type,
                message=f"Failed to end dialogue: {e}",
                details={"error": str(e)}
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {"type": "end_dialogue"}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EndDialogueEffect":
        """Deserialize from dictionary."""
        return cls()


# =============================================================================
# Custom Effect
# =============================================================================

@dataclass
class CustomEffect(Effect):
    """
    Custom effect with user-defined execution.

    Warning: Custom effects cannot be fully serialized.
    """
    name: str
    executor: Callable[[EffectContext], EffectResult]
    description: str = ""

    @property
    def effect_type(self) -> EffectType:
        return EffectType.TRIGGER_EVENT

    def execute(self, context: EffectContext) -> EffectResult:
        """Execute the custom effect."""
        try:
            return self.executor(context)
        except Exception as e:
            return EffectResult(
                success=False,
                effect_type=self.effect_type,
                message=f"Custom effect '{self.name}' failed: {e}",
                details={"error": str(e)}
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (limited)."""
        return {
            "type": "custom",
            "name": self.name,
            "description": self.description,
            "_warning": "Custom effects cannot be fully serialized"
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CustomEffect":
        """Deserialize from dictionary (creates no-op effect)."""
        return cls(
            name=data.get("name", "unknown"),
            executor=lambda ctx: EffectResult(
                success=True,
                effect_type=EffectType.TRIGGER_EVENT,
                message="Deserialized custom effect (no-op)"
            ),
            description=data.get("description", "")
        )


# =============================================================================
# Effect Batch
# =============================================================================

@dataclass
class EffectBatch:
    """
    A batch of effects to execute together.

    Supports transactional execution with rollback on failure.
    """
    effects: List[Effect]
    transactional: bool = False

    def execute(self, context: EffectContext) -> List[EffectResult]:
        """
        Execute all effects in the batch.

        Args:
            context: The effect execution context.

        Returns:
            List of results for each effect.
        """
        # Sort by priority
        sorted_effects = sorted(
            self.effects,
            key=lambda e: e.priority.value
        )

        results: List[EffectResult] = []
        executed: List[tuple[Effect, EffectResult]] = []

        for effect in sorted_effects:
            result = effect.execute(context)
            results.append(result)

            if result.success:
                executed.append((effect, result))
            elif self.transactional:
                # Rollback all previously executed effects
                self._rollback(context, executed)
                break

        return results

    def _rollback(
        self,
        context: EffectContext,
        executed: List[tuple[Effect, EffectResult]]
    ) -> None:
        """Rollback executed effects in reverse order."""
        for effect, result in reversed(executed):
            if result.rollback_data:
                effect.rollback(context, result.rollback_data)


# =============================================================================
# Effect Factory
# =============================================================================

EFFECT_TYPES: Dict[str, type] = {
    "set_variable": SetVariableEffect,
    "increment_variable": IncrementVariableEffect,
    "decrement_variable": DecrementVariableEffect,
    "give_item": GiveItemEffect,
    "take_item": TakeItemEffect,
    "update_quest_state": UpdateQuestStateEffect,
    "update_quest_progress": UpdateQuestProgressEffect,
    "start_quest": StartQuestEffect,
    "complete_quest": CompleteQuestEffect,
    "fail_quest": FailQuestEffect,
    "change_reputation": ChangeReputationEffect,
    "set_reputation": SetReputationEffect,
    "trigger_event": TriggerEventEffect,
    "play_sound": PlaySoundEffect,
    "play_animation": PlayAnimationEffect,
    "start_dialogue": StartDialogueEffect,
    "end_dialogue": EndDialogueEffect,
    "custom": CustomEffect,
}


def effect_from_dict(data: Dict[str, Any]) -> Effect:
    """
    Create an effect from a dictionary.

    Args:
        data: Dictionary representation of an effect.

    Returns:
        Effect instance.

    Raises:
        ValueError: If effect type is unknown.
    """
    effect_type = data.get("type")

    if effect_type not in EFFECT_TYPES:
        raise ValueError(f"Unknown effect type: {effect_type}")

    return EFFECT_TYPES[effect_type].from_dict(data)


def register_effect_type(name: str, effect_class: type) -> None:
    """
    Register a custom effect type.

    Args:
        name: The type name.
        effect_class: The effect class.
    """
    if not issubclass(effect_class, Effect):
        raise TypeError("effect_class must be an Effect subclass")

    EFFECT_TYPES[name] = effect_class
