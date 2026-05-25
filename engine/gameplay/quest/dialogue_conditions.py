"""
Dialogue Condition System.

Provides condition checking for dialogue branching, including:
- Variable checks (comparison operations)
- Item checks (inventory presence/quantity)
- Quest state checks (progress, completion)
- Reputation checks (faction standing)
- Compound conditions (AND, OR, NOT, XOR)
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

from .constants import (
    ComparisonOperator,
    LogicalOperator,
    QuestState,
    ReputationLevel,
    MIN_REPUTATION,
    MAX_REPUTATION,
)
from .dialogue_variables import VariableManager, VariableScope, VariableValue


# =============================================================================
# Context Protocol
# =============================================================================

class DialogueContext(Protocol):
    """Protocol for dialogue evaluation context."""

    @property
    def variables(self) -> VariableManager:
        """Get the variable manager."""
        ...

    def get_item_count(self, item_id: str) -> int:
        """Get the count of an item in inventory."""
        ...

    def has_item(self, item_id: str, count: int = 1) -> bool:
        """Check if player has item(s)."""
        ...

    def get_quest_state(self, quest_id: str) -> QuestState:
        """Get the state of a quest."""
        ...

    def get_quest_progress(self, quest_id: str) -> float:
        """Get quest progress (0.0 to 1.0)."""
        ...

    def get_reputation(self, faction_id: str) -> int:
        """Get reputation with a faction."""
        ...

    def get_reputation_level(self, faction_id: str) -> ReputationLevel:
        """Get named reputation level with faction."""
        ...


# =============================================================================
# Condition Base Class
# =============================================================================

@dataclass
class ConditionResult:
    """Result of evaluating a condition."""
    success: bool
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class Condition(ABC):
    """
    Abstract base class for all dialogue conditions.

    Conditions are evaluated against a context to determine
    if a dialogue path should be taken.
    """

    @abstractmethod
    def evaluate(self, context: DialogueContext) -> ConditionResult:
        """
        Evaluate the condition.

        Args:
            context: The dialogue context for evaluation.

        Returns:
            ConditionResult with success status and details.
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the condition to a dictionary.

        Returns:
            Dictionary representation.
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Condition":
        """
        Deserialize a condition from a dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            Condition instance.
        """
        pass

    def __and__(self, other: "Condition") -> "AndCondition":
        """Combine conditions with AND."""
        return AndCondition([self, other])

    def __or__(self, other: "Condition") -> "OrCondition":
        """Combine conditions with OR."""
        return OrCondition([self, other])

    def __invert__(self) -> "NotCondition":
        """Negate the condition."""
        return NotCondition(self)


# =============================================================================
# Variable Conditions
# =============================================================================

@dataclass
class VariableCondition(Condition):
    """
    Condition that checks a variable against a value.

    Supports all comparison operators defined in constants.
    """
    variable_name: str
    operator: ComparisonOperator
    expected_value: VariableValue
    scope: Optional[VariableScope] = None

    def evaluate(self, context: DialogueContext) -> ConditionResult:
        """Evaluate the variable condition."""
        actual_value = context.variables.get(
            self.variable_name,
            scope=self.scope
        )

        success = self._compare(actual_value, self.expected_value)

        return ConditionResult(
            success=success,
            message=f"Variable '{self.variable_name}' "
                    f"{self.operator.value} {self.expected_value}: {success}",
            details={
                "variable": self.variable_name,
                "actual": actual_value,
                "expected": self.expected_value,
                "operator": self.operator.value
            }
        )

    def _compare(
        self,
        actual: VariableValue,
        expected: VariableValue
    ) -> bool:
        """Perform the comparison operation."""
        op = self.operator

        if op == ComparisonOperator.EQUAL:
            return actual == expected
        elif op == ComparisonOperator.NOT_EQUAL:
            return actual != expected
        elif op == ComparisonOperator.GREATER:
            return self._numeric_compare(actual, expected, lambda a, b: a > b)
        elif op == ComparisonOperator.GREATER_EQUAL:
            return self._numeric_compare(actual, expected, lambda a, b: a >= b)
        elif op == ComparisonOperator.LESS:
            return self._numeric_compare(actual, expected, lambda a, b: a < b)
        elif op == ComparisonOperator.LESS_EQUAL:
            return self._numeric_compare(actual, expected, lambda a, b: a <= b)
        elif op == ComparisonOperator.CONTAINS:
            return self._contains(actual, expected)
        elif op == ComparisonOperator.NOT_CONTAINS:
            return not self._contains(actual, expected)
        elif op == ComparisonOperator.STARTS_WITH:
            return self._string_op(actual, expected, str.startswith)
        elif op == ComparisonOperator.ENDS_WITH:
            return self._string_op(actual, expected, str.endswith)
        elif op == ComparisonOperator.MATCHES:
            return self._regex_match(actual, expected)

        return False

    def _numeric_compare(
        self,
        actual: VariableValue,
        expected: VariableValue,
        comparator: Callable[[Any, Any], bool]
    ) -> bool:
        """Compare numeric values."""
        try:
            return comparator(float(actual), float(expected))
        except (TypeError, ValueError):
            return False

    def _contains(
        self,
        actual: VariableValue,
        expected: VariableValue
    ) -> bool:
        """Check if actual contains expected."""
        if isinstance(actual, str) and isinstance(expected, str):
            return expected in actual
        if isinstance(actual, (list, tuple)):
            return expected in actual
        if isinstance(actual, dict):
            return expected in actual
        return False

    def _string_op(
        self,
        actual: VariableValue,
        expected: VariableValue,
        op: Callable[[str, str], bool]
    ) -> bool:
        """Perform string operation."""
        if isinstance(actual, str) and isinstance(expected, str):
            return op(actual, expected)
        return False

    def _regex_match(
        self,
        actual: VariableValue,
        expected: VariableValue
    ) -> bool:
        """Check if actual matches regex pattern."""
        if not isinstance(actual, str) or not isinstance(expected, str):
            return False
        try:
            return bool(re.search(expected, actual))
        except re.error:
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "type": "variable",
            "variable_name": self.variable_name,
            "operator": self.operator.value,
            "expected_value": self.expected_value
        }
        if self.scope:
            result["scope"] = self.scope.name
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VariableCondition":
        """Deserialize from dictionary."""
        scope = None
        if "scope" in data:
            scope = VariableScope[data["scope"]]

        return cls(
            variable_name=data["variable_name"],
            operator=ComparisonOperator(data["operator"]),
            expected_value=data["expected_value"],
            scope=scope
        )


@dataclass
class VariableExistsCondition(Condition):
    """Condition that checks if a variable exists."""
    variable_name: str
    scope: Optional[VariableScope] = None

    def evaluate(self, context: DialogueContext) -> ConditionResult:
        """Evaluate the existence condition."""
        exists = context.variables.has(self.variable_name, scope=self.scope)

        return ConditionResult(
            success=exists,
            message=f"Variable '{self.variable_name}' exists: {exists}",
            details={
                "variable": self.variable_name,
                "exists": exists
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "type": "variable_exists",
            "variable_name": self.variable_name
        }
        if self.scope:
            result["scope"] = self.scope.name
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VariableExistsCondition":
        """Deserialize from dictionary."""
        scope = None
        if "scope" in data:
            scope = VariableScope[data["scope"]]

        return cls(
            variable_name=data["variable_name"],
            scope=scope
        )


# =============================================================================
# Item Conditions
# =============================================================================

@dataclass
class ItemCondition(Condition):
    """
    Condition that checks for item presence or quantity.
    """
    item_id: str
    min_count: int = 1
    max_count: Optional[int] = None

    def __post_init__(self):
        """Validate item condition parameters."""
        if not self.item_id:
            raise ValueError("item_id cannot be empty")
        if self.min_count < 0:
            raise ValueError("min_count must be >= 0")
        if self.max_count is not None and self.max_count < self.min_count:
            raise ValueError("max_count must be >= min_count")

    def evaluate(self, context: DialogueContext) -> ConditionResult:
        """Evaluate the item condition."""
        count = context.get_item_count(self.item_id)

        has_minimum = count >= self.min_count
        within_maximum = (
            self.max_count is None or count <= self.max_count
        )

        success = has_minimum and within_maximum

        return ConditionResult(
            success=success,
            message=f"Item '{self.item_id}' count={count}, "
                    f"need {self.min_count}-{self.max_count or 'inf'}: {success}",
            details={
                "item_id": self.item_id,
                "count": count,
                "min_count": self.min_count,
                "max_count": self.max_count
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "type": "item",
            "item_id": self.item_id,
            "min_count": self.min_count
        }
        if self.max_count is not None:
            result["max_count"] = self.max_count
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ItemCondition":
        """Deserialize from dictionary."""
        return cls(
            item_id=data["item_id"],
            min_count=data.get("min_count", 1),
            max_count=data.get("max_count")
        )


@dataclass
class HasItemCondition(Condition):
    """Simplified condition for checking if player has an item."""
    item_id: str
    count: int = 1

    def __post_init__(self):
        """Validate parameters."""
        if not self.item_id:
            raise ValueError("item_id cannot be empty")
        if self.count < 1:
            raise ValueError("count must be >= 1")

    def evaluate(self, context: DialogueContext) -> ConditionResult:
        """Evaluate the has-item condition."""
        has_it = context.has_item(self.item_id, self.count)

        return ConditionResult(
            success=has_it,
            message=f"Has item '{self.item_id}' x{self.count}: {has_it}",
            details={
                "item_id": self.item_id,
                "required_count": self.count,
                "has_item": has_it
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "has_item",
            "item_id": self.item_id,
            "count": self.count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HasItemCondition":
        """Deserialize from dictionary."""
        return cls(
            item_id=data["item_id"],
            count=data.get("count", 1)
        )


# =============================================================================
# Quest Conditions
# =============================================================================

@dataclass
class QuestStateCondition(Condition):
    """Condition that checks quest state."""
    quest_id: str
    required_state: QuestState

    def __post_init__(self):
        """Validate parameters."""
        if not self.quest_id:
            raise ValueError("quest_id cannot be empty")

    def evaluate(self, context: DialogueContext) -> ConditionResult:
        """Evaluate the quest state condition."""
        current_state = context.get_quest_state(self.quest_id)
        success = current_state == self.required_state

        return ConditionResult(
            success=success,
            message=f"Quest '{self.quest_id}' state={current_state.name}, "
                    f"need={self.required_state.name}: {success}",
            details={
                "quest_id": self.quest_id,
                "current_state": current_state.name,
                "required_state": self.required_state.name
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "quest_state",
            "quest_id": self.quest_id,
            "required_state": self.required_state.name
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuestStateCondition":
        """Deserialize from dictionary."""
        return cls(
            quest_id=data["quest_id"],
            required_state=QuestState[data["required_state"]]
        )


@dataclass
class QuestProgressCondition(Condition):
    """Condition that checks quest progress (0.0 to 1.0)."""
    quest_id: str
    min_progress: float = 0.0
    max_progress: float = 1.0

    def __post_init__(self):
        """Validate parameters."""
        if not self.quest_id:
            raise ValueError("quest_id cannot be empty")
        if not 0.0 <= self.min_progress <= 1.0:
            raise ValueError("min_progress must be between 0.0 and 1.0")
        if not 0.0 <= self.max_progress <= 1.0:
            raise ValueError("max_progress must be between 0.0 and 1.0")
        if self.max_progress < self.min_progress:
            raise ValueError("max_progress must be >= min_progress")

    def evaluate(self, context: DialogueContext) -> ConditionResult:
        """Evaluate the quest progress condition."""
        progress = context.get_quest_progress(self.quest_id)

        success = self.min_progress <= progress <= self.max_progress

        return ConditionResult(
            success=success,
            message=f"Quest '{self.quest_id}' progress={progress:.1%}, "
                    f"need {self.min_progress:.1%}-{self.max_progress:.1%}: {success}",
            details={
                "quest_id": self.quest_id,
                "progress": progress,
                "min_progress": self.min_progress,
                "max_progress": self.max_progress
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "quest_progress",
            "quest_id": self.quest_id,
            "min_progress": self.min_progress,
            "max_progress": self.max_progress
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuestProgressCondition":
        """Deserialize from dictionary."""
        return cls(
            quest_id=data["quest_id"],
            min_progress=data.get("min_progress", 0.0),
            max_progress=data.get("max_progress", 1.0)
        )


@dataclass
class QuestCompletedCondition(Condition):
    """Simplified condition for checking if a quest is completed."""
    quest_id: str

    def __post_init__(self):
        """Validate parameters."""
        if not self.quest_id:
            raise ValueError("quest_id cannot be empty")

    def evaluate(self, context: DialogueContext) -> ConditionResult:
        """Evaluate the quest completed condition."""
        state = context.get_quest_state(self.quest_id)
        success = state == QuestState.COMPLETED

        return ConditionResult(
            success=success,
            message=f"Quest '{self.quest_id}' completed: {success}",
            details={
                "quest_id": self.quest_id,
                "state": state.name,
                "completed": success
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "quest_completed",
            "quest_id": self.quest_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuestCompletedCondition":
        """Deserialize from dictionary."""
        return cls(quest_id=data["quest_id"])


@dataclass
class QuestActiveCondition(Condition):
    """Condition for checking if a quest is active (in progress)."""
    quest_id: str

    def __post_init__(self):
        """Validate parameters."""
        if not self.quest_id:
            raise ValueError("quest_id cannot be empty")

    def evaluate(self, context: DialogueContext) -> ConditionResult:
        """Evaluate the quest active condition."""
        state = context.get_quest_state(self.quest_id)
        success = state == QuestState.IN_PROGRESS

        return ConditionResult(
            success=success,
            message=f"Quest '{self.quest_id}' active: {success}",
            details={
                "quest_id": self.quest_id,
                "state": state.name,
                "active": success
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "quest_active",
            "quest_id": self.quest_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuestActiveCondition":
        """Deserialize from dictionary."""
        return cls(quest_id=data["quest_id"])


# =============================================================================
# Reputation Conditions
# =============================================================================

@dataclass
class ReputationCondition(Condition):
    """Condition that checks reputation with a faction."""
    faction_id: str
    min_reputation: int = MIN_REPUTATION
    max_reputation: int = MAX_REPUTATION

    def __post_init__(self):
        """Validate parameters."""
        if not self.faction_id:
            raise ValueError("faction_id cannot be empty")
        if not MIN_REPUTATION <= self.min_reputation <= MAX_REPUTATION:
            raise ValueError(
                f"min_reputation must be between {MIN_REPUTATION} and {MAX_REPUTATION}"
            )
        if not MIN_REPUTATION <= self.max_reputation <= MAX_REPUTATION:
            raise ValueError(
                f"max_reputation must be between {MIN_REPUTATION} and {MAX_REPUTATION}"
            )
        if self.max_reputation < self.min_reputation:
            raise ValueError("max_reputation must be >= min_reputation")

    def evaluate(self, context: DialogueContext) -> ConditionResult:
        """Evaluate the reputation condition."""
        reputation = context.get_reputation(self.faction_id)

        success = self.min_reputation <= reputation <= self.max_reputation

        return ConditionResult(
            success=success,
            message=f"Faction '{self.faction_id}' reputation={reputation}, "
                    f"need {self.min_reputation}-{self.max_reputation}: {success}",
            details={
                "faction_id": self.faction_id,
                "reputation": reputation,
                "min_reputation": self.min_reputation,
                "max_reputation": self.max_reputation
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "reputation",
            "faction_id": self.faction_id,
            "min_reputation": self.min_reputation,
            "max_reputation": self.max_reputation
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReputationCondition":
        """Deserialize from dictionary."""
        return cls(
            faction_id=data["faction_id"],
            min_reputation=data.get("min_reputation", MIN_REPUTATION),
            max_reputation=data.get("max_reputation", MAX_REPUTATION)
        )


@dataclass
class ReputationLevelCondition(Condition):
    """Condition that checks reputation level with a faction."""
    faction_id: str
    required_level: ReputationLevel
    allow_higher: bool = True

    def __post_init__(self):
        """Validate parameters."""
        if not self.faction_id:
            raise ValueError("faction_id cannot be empty")

    def evaluate(self, context: DialogueContext) -> ConditionResult:
        """Evaluate the reputation level condition."""
        level = context.get_reputation_level(self.faction_id)

        if self.allow_higher:
            # Higher levels have higher enum values
            success = level.value >= self.required_level.value
        else:
            success = level == self.required_level

        return ConditionResult(
            success=success,
            message=f"Faction '{self.faction_id}' level={level.name}, "
                    f"need={self.required_level.name}: {success}",
            details={
                "faction_id": self.faction_id,
                "current_level": level.name,
                "required_level": self.required_level.name,
                "allow_higher": self.allow_higher
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "reputation_level",
            "faction_id": self.faction_id,
            "required_level": self.required_level.name,
            "allow_higher": self.allow_higher
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReputationLevelCondition":
        """Deserialize from dictionary."""
        return cls(
            faction_id=data["faction_id"],
            required_level=ReputationLevel[data["required_level"]],
            allow_higher=data.get("allow_higher", True)
        )


# =============================================================================
# Compound Conditions
# =============================================================================

@dataclass
class AndCondition(Condition):
    """Condition that requires all sub-conditions to be true."""
    conditions: List[Condition]

    def __post_init__(self):
        """Validate parameters."""
        if not self.conditions:
            raise ValueError("AndCondition requires at least one condition")

    def evaluate(self, context: DialogueContext) -> ConditionResult:
        """Evaluate all conditions with AND logic."""
        results = []
        all_success = True

        for condition in self.conditions:
            result = condition.evaluate(context)
            results.append(result)
            if not result.success:
                all_success = False

        return ConditionResult(
            success=all_success,
            message=f"AND({len(self.conditions)} conditions): {all_success}",
            details={
                "operator": "AND",
                "sub_results": [
                    {"success": r.success, "message": r.message}
                    for r in results
                ]
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "and",
            "conditions": [c.to_dict() for c in self.conditions]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AndCondition":
        """Deserialize from dictionary."""
        conditions = [
            condition_from_dict(c) for c in data["conditions"]
        ]
        return cls(conditions=conditions)


@dataclass
class OrCondition(Condition):
    """Condition that requires any sub-condition to be true."""
    conditions: List[Condition]

    def __post_init__(self):
        """Validate parameters."""
        if not self.conditions:
            raise ValueError("OrCondition requires at least one condition")

    def evaluate(self, context: DialogueContext) -> ConditionResult:
        """Evaluate conditions with OR logic."""
        results = []
        any_success = False

        for condition in self.conditions:
            result = condition.evaluate(context)
            results.append(result)
            if result.success:
                any_success = True

        return ConditionResult(
            success=any_success,
            message=f"OR({len(self.conditions)} conditions): {any_success}",
            details={
                "operator": "OR",
                "sub_results": [
                    {"success": r.success, "message": r.message}
                    for r in results
                ]
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "or",
            "conditions": [c.to_dict() for c in self.conditions]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrCondition":
        """Deserialize from dictionary."""
        conditions = [
            condition_from_dict(c) for c in data["conditions"]
        ]
        return cls(conditions=conditions)


@dataclass
class NotCondition(Condition):
    """Condition that negates another condition."""
    condition: Condition

    def evaluate(self, context: DialogueContext) -> ConditionResult:
        """Evaluate with NOT logic."""
        result = self.condition.evaluate(context)

        return ConditionResult(
            success=not result.success,
            message=f"NOT({result.message}): {not result.success}",
            details={
                "operator": "NOT",
                "sub_result": {
                    "success": result.success,
                    "message": result.message
                }
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "not",
            "condition": self.condition.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NotCondition":
        """Deserialize from dictionary."""
        condition = condition_from_dict(data["condition"])
        return cls(condition=condition)


@dataclass
class XorCondition(Condition):
    """Condition that requires exactly one sub-condition to be true."""
    conditions: List[Condition]

    def __post_init__(self):
        """Validate parameters."""
        if len(self.conditions) < 2:
            raise ValueError("XorCondition requires at least two conditions")

    def evaluate(self, context: DialogueContext) -> ConditionResult:
        """Evaluate with XOR logic."""
        results = []
        true_count = 0

        for condition in self.conditions:
            result = condition.evaluate(context)
            results.append(result)
            if result.success:
                true_count += 1

        success = true_count == 1

        return ConditionResult(
            success=success,
            message=f"XOR({len(self.conditions)} conditions, "
                    f"{true_count} true): {success}",
            details={
                "operator": "XOR",
                "true_count": true_count,
                "sub_results": [
                    {"success": r.success, "message": r.message}
                    for r in results
                ]
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "xor",
            "conditions": [c.to_dict() for c in self.conditions]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "XorCondition":
        """Deserialize from dictionary."""
        conditions = [
            condition_from_dict(c) for c in data["conditions"]
        ]
        return cls(conditions=conditions)


# =============================================================================
# Custom Condition (for extensibility)
# =============================================================================

@dataclass
class CustomCondition(Condition):
    """
    Custom condition with a user-defined evaluation function.

    Warning: Custom conditions cannot be serialized/deserialized.
    """
    name: str
    evaluator: Callable[[DialogueContext], bool]
    description: str = ""

    def evaluate(self, context: DialogueContext) -> ConditionResult:
        """Evaluate the custom condition."""
        try:
            success = self.evaluator(context)
            return ConditionResult(
                success=success,
                message=f"Custom '{self.name}': {success}",
                details={
                    "custom_name": self.name,
                    "description": self.description
                }
            )
        except Exception as e:
            return ConditionResult(
                success=False,
                message=f"Custom '{self.name}' error: {e}",
                details={"error": str(e)}
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (limited - no function)."""
        return {
            "type": "custom",
            "name": self.name,
            "description": self.description,
            "_warning": "Custom conditions cannot be fully serialized"
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CustomCondition":
        """Deserialize from dictionary (creates always-false condition)."""
        return cls(
            name=data.get("name", "unknown"),
            evaluator=lambda ctx: False,
            description=data.get("description", "Deserialized custom condition")
        )


# =============================================================================
# Always True/False Conditions
# =============================================================================

class AlwaysTrueCondition(Condition):
    """Condition that always evaluates to true."""

    def evaluate(self, context: DialogueContext) -> ConditionResult:
        """Always returns true."""
        return ConditionResult(
            success=True,
            message="Always true",
            details={}
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {"type": "always_true"}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlwaysTrueCondition":
        """Deserialize from dictionary."""
        return cls()


class AlwaysFalseCondition(Condition):
    """Condition that always evaluates to false."""

    def evaluate(self, context: DialogueContext) -> ConditionResult:
        """Always returns false."""
        return ConditionResult(
            success=False,
            message="Always false",
            details={}
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {"type": "always_false"}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlwaysFalseCondition":
        """Deserialize from dictionary."""
        return cls()


# =============================================================================
# Condition Factory
# =============================================================================

# Registry of condition types for deserialization
CONDITION_TYPES: Dict[str, type] = {
    "variable": VariableCondition,
    "variable_exists": VariableExistsCondition,
    "item": ItemCondition,
    "has_item": HasItemCondition,
    "quest_state": QuestStateCondition,
    "quest_progress": QuestProgressCondition,
    "quest_completed": QuestCompletedCondition,
    "quest_active": QuestActiveCondition,
    "reputation": ReputationCondition,
    "reputation_level": ReputationLevelCondition,
    "and": AndCondition,
    "or": OrCondition,
    "not": NotCondition,
    "xor": XorCondition,
    "custom": CustomCondition,
    "always_true": AlwaysTrueCondition,
    "always_false": AlwaysFalseCondition,
}


def condition_from_dict(data: Dict[str, Any]) -> Condition:
    """
    Create a condition from a dictionary.

    Args:
        data: Dictionary representation of a condition.

    Returns:
        Condition instance.

    Raises:
        ValueError: If condition type is unknown.
    """
    condition_type = data.get("type")

    if condition_type not in CONDITION_TYPES:
        raise ValueError(f"Unknown condition type: {condition_type}")

    return CONDITION_TYPES[condition_type].from_dict(data)


def register_condition_type(name: str, condition_class: type) -> None:
    """
    Register a custom condition type for deserialization.

    Args:
        name: The type name.
        condition_class: The condition class.
    """
    if not issubclass(condition_class, Condition):
        raise TypeError("condition_class must be a Condition subclass")

    CONDITION_TYPES[name] = condition_class
