"""
Utility AI System - Scoring-based decision making.

Provides a flexible utility AI implementation with:
- Multiple response curve types
- Weighted considerations
- Action selection with momentum
- Debug visualization support
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from .blackboard import Blackboard
from .constants import (
    ResponseCurveType,
    UTILITY_DEFAULT_UPDATE_RATE,
    UTILITY_MIN_SCORE_THRESHOLD,
    UTILITY_SCORE_EPSILON,
    UTILITY_MAX_CONSIDERATIONS,
    UTILITY_DEFAULT_WEIGHT,
    UTILITY_DEFAULT_MOMENTUM,
    UTILITY_LOGISTIC_CENTER,
    UTILITY_LOGISTIC_STEEPNESS,
    UTILITY_SMOOTHSTEP_COEFF_A,
    UTILITY_SMOOTHSTEP_COEFF_B,
    UTILITY_ACTION_HISTORY_SIZE,
)

T = TypeVar("T")


# =============================================================================
# Response Curves
# =============================================================================


@dataclass
class ResponseCurve:
    """
    Defines how an input value maps to a utility score.

    Supports various curve types for different behaviors.
    """
    curve_type: ResponseCurveType = ResponseCurveType.LINEAR
    slope: float = 1.0
    exponent: float = 2.0
    x_shift: float = 0.0
    y_shift: float = 0.0
    invert: bool = False
    clamp_min: float = 0.0
    clamp_max: float = 1.0

    def evaluate(self, input_value: float) -> float:
        """Evaluate the curve at the given input value."""
        # Apply x shift
        x = input_value - self.x_shift

        # Compute base value based on curve type
        if self.curve_type == ResponseCurveType.LINEAR:
            y = self.slope * x
        elif self.curve_type == ResponseCurveType.QUADRATIC:
            y = self.slope * (x ** 2)
        elif self.curve_type == ResponseCurveType.EXPONENTIAL:
            y = self.slope * (math.exp(self.exponent * x) - 1)
        elif self.curve_type == ResponseCurveType.LOGISTIC:
            try:
                y = 1.0 / (1.0 + math.exp(-self.slope * x))
            except OverflowError:
                y = 0.0 if x < 0 else 1.0
        elif self.curve_type == ResponseCurveType.SINE:
            y = math.sin(self.slope * x * math.pi / 2)
        elif self.curve_type == ResponseCurveType.INVERSE:
            if abs(x) < UTILITY_SCORE_EPSILON:
                y = self.clamp_max
            else:
                y = self.slope / x
        elif self.curve_type == ResponseCurveType.STEP:
            y = 1.0 if x >= self.slope else 0.0
        elif self.curve_type == ResponseCurveType.SMOOTHSTEP:
            if x <= 0:
                y = 0.0
            elif x >= 1:
                y = 1.0
            else:
                # Smoothstep polynomial: 3x^2 - 2x^3 = x^2(3 - 2x)
                y = x * x * (UTILITY_SMOOTHSTEP_COEFF_A - UTILITY_SMOOTHSTEP_COEFF_B * x)
        else:
            y = x

        # Apply y shift
        y = y + self.y_shift

        # Invert if needed
        if self.invert:
            y = 1.0 - y

        # Clamp to range
        return max(self.clamp_min, min(self.clamp_max, y))


class CustomResponseCurve(ResponseCurve):
    """A response curve with a custom evaluation function."""

    def __init__(
        self,
        func: Callable[[float], float],
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
    ) -> None:
        super().__init__(
            curve_type=ResponseCurveType.CUSTOM,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
        )
        self._func = func

    def evaluate(self, input_value: float) -> float:
        """Evaluate the custom curve function."""
        y = self._func(input_value)
        return max(self.clamp_min, min(self.clamp_max, y))


# Preset curves for common use cases
LINEAR_CURVE = ResponseCurve(curve_type=ResponseCurveType.LINEAR)
QUADRATIC_CURVE = ResponseCurve(curve_type=ResponseCurveType.QUADRATIC)
EXPONENTIAL_CURVE = ResponseCurve(curve_type=ResponseCurveType.EXPONENTIAL)
LOGISTIC_CURVE = ResponseCurve(curve_type=ResponseCurveType.LOGISTIC)
INVERSE_CURVE = ResponseCurve(curve_type=ResponseCurveType.INVERSE)
SMOOTHSTEP_CURVE = ResponseCurve(curve_type=ResponseCurveType.SMOOTHSTEP)


# =============================================================================
# Considerations
# =============================================================================


@dataclass
class ConsiderationContext:
    """Context passed to considerations during evaluation."""
    entity: Any = None
    blackboard: Optional[Blackboard] = None
    target: Any = None
    world_state: Dict[str, Any] = field(default_factory=dict)


class Consideration(ABC):
    """
    A single factor that influences an action's utility score.

    Each consideration evaluates an aspect of the game state and
    returns a normalized score (0-1).
    """

    def __init__(
        self,
        name: str,
        weight: float = UTILITY_DEFAULT_WEIGHT,
        curve: Optional[ResponseCurve] = None,
    ) -> None:
        self.name = name
        self.weight = weight
        self.curve = curve or LINEAR_CURVE
        self._last_raw_score = 0.0
        self._last_final_score = 0.0

    @abstractmethod
    def get_input(self, context: ConsiderationContext) -> float:
        """
        Get the raw input value for this consideration.

        Should return a value that will be passed through the response curve.
        """
        pass

    def score(self, context: ConsiderationContext) -> float:
        """
        Calculate the final weighted score for this consideration.
        """
        raw_input = self.get_input(context)
        self._last_raw_score = raw_input

        curved_score = self.curve.evaluate(raw_input)
        self._last_final_score = curved_score * self.weight

        return self._last_final_score

    @property
    def last_raw_score(self) -> float:
        """Get the last raw input value (for debugging)."""
        return self._last_raw_score

    @property
    def last_final_score(self) -> float:
        """Get the last final score (for debugging)."""
        return self._last_final_score


class BlackboardConsideration(Consideration):
    """A consideration that reads from a blackboard key."""

    def __init__(
        self,
        name: str,
        key: str,
        default: float = 0.0,
        normalize_min: float = 0.0,
        normalize_max: float = 1.0,
        weight: float = UTILITY_DEFAULT_WEIGHT,
        curve: Optional[ResponseCurve] = None,
    ) -> None:
        super().__init__(name, weight, curve)
        self.key = key
        self.default = default
        self.normalize_min = normalize_min
        self.normalize_max = normalize_max

    def get_input(self, context: ConsiderationContext) -> float:
        if context.blackboard is None:
            return self.default

        value = context.blackboard.get(self.key, self.default)

        # Normalize to 0-1 range
        range_size = self.normalize_max - self.normalize_min
        if abs(range_size) < UTILITY_SCORE_EPSILON:
            return 0.0

        normalized = (float(value) - self.normalize_min) / range_size
        return max(0.0, min(1.0, normalized))


class FunctionConsideration(Consideration):
    """A consideration that uses a custom function."""

    def __init__(
        self,
        name: str,
        func: Callable[[ConsiderationContext], float],
        weight: float = UTILITY_DEFAULT_WEIGHT,
        curve: Optional[ResponseCurve] = None,
    ) -> None:
        super().__init__(name, weight, curve)
        self._func = func

    def get_input(self, context: ConsiderationContext) -> float:
        return self._func(context)


class DistanceConsideration(Consideration):
    """A consideration based on distance to a target."""

    def __init__(
        self,
        name: str,
        max_distance: float,
        get_position: Callable[[Any], tuple],
        get_target_position: Callable[[ConsiderationContext], tuple],
        weight: float = UTILITY_DEFAULT_WEIGHT,
        curve: Optional[ResponseCurve] = None,
    ) -> None:
        super().__init__(name, weight, curve)
        self.max_distance = max_distance
        self.get_position = get_position
        self.get_target_position = get_target_position

    def get_input(self, context: ConsiderationContext) -> float:
        if context.entity is None:
            return 1.0

        pos = self.get_position(context.entity)
        target_pos = self.get_target_position(context)

        # Calculate distance
        dist_sq = sum((a - b) ** 2 for a, b in zip(pos, target_pos))
        dist = math.sqrt(dist_sq)

        # Normalize by max distance
        return min(1.0, dist / self.max_distance)


class HealthConsideration(Consideration):
    """A consideration based on entity health."""

    def __init__(
        self,
        name: str = "health",
        get_health: Optional[Callable[[Any], tuple]] = None,
        weight: float = UTILITY_DEFAULT_WEIGHT,
        curve: Optional[ResponseCurve] = None,
    ) -> None:
        super().__init__(name, weight, curve)
        self.get_health = get_health or (lambda e: (getattr(e, "health", 100), getattr(e, "max_health", 100)))

    def get_input(self, context: ConsiderationContext) -> float:
        if context.entity is None:
            return 1.0

        current, maximum = self.get_health(context.entity)
        if maximum <= 0:
            return 0.0
        return current / maximum


# =============================================================================
# Actions
# =============================================================================


@dataclass
class ActionScore:
    """The computed score for an action."""
    action: "UtilityAction"
    score: float
    consideration_scores: Dict[str, float] = field(default_factory=dict)


class UtilityAction(ABC):
    """
    An action that can be selected by the utility AI.

    Each action has a set of considerations that determine its score.
    """

    def __init__(
        self,
        name: str,
        considerations: Optional[List[Consideration]] = None,
        base_score: float = 0.0,
        cooldown: float = 0.0,
    ) -> None:
        self.name = name
        self._considerations: List[Consideration] = considerations or []
        self.base_score = base_score
        self.cooldown = cooldown
        self._last_execution_time: Optional[float] = None

        if len(self._considerations) > UTILITY_MAX_CONSIDERATIONS:
            raise ValueError(
                f"Action {name} has too many considerations "
                f"({len(self._considerations)} > {UTILITY_MAX_CONSIDERATIONS})"
            )

    @property
    def considerations(self) -> List[Consideration]:
        """Get the list of considerations."""
        return self._considerations

    def add_consideration(self, consideration: Consideration) -> "UtilityAction":
        """Add a consideration to this action."""
        if len(self._considerations) >= UTILITY_MAX_CONSIDERATIONS:
            raise ValueError("Maximum considerations reached")
        self._considerations.append(consideration)
        return self

    def is_on_cooldown(self, current_time: float) -> bool:
        """Check if this action is on cooldown."""
        if self.cooldown <= 0 or self._last_execution_time is None:
            return False
        return (current_time - self._last_execution_time) < self.cooldown

    def calculate_score(self, context: ConsiderationContext) -> ActionScore:
        """Calculate the total utility score for this action."""
        if not self._considerations:
            return ActionScore(action=self, score=self.base_score)

        # Use compensation factor for multiple considerations
        # This helps prevent score collapse when multiplying many factors
        total_score = self.base_score
        consideration_scores: Dict[str, float] = {}

        for consideration in self._considerations:
            score = consideration.score(context)
            consideration_scores[consideration.name] = score

            if score <= UTILITY_SCORE_EPSILON:
                # If any consideration is zero, action is not viable
                return ActionScore(
                    action=self,
                    score=0.0,
                    consideration_scores=consideration_scores,
                )

            # Apply compensation factor
            modification = (1 - score) * (1 - (1 / len(self._considerations)))
            total_score += score + modification * score

        # Normalize
        total_score /= len(self._considerations)

        return ActionScore(
            action=self,
            score=total_score,
            consideration_scores=consideration_scores,
        )

    @abstractmethod
    def execute(self, context: ConsiderationContext) -> bool:
        """
        Execute this action.

        Returns True if the action was executed successfully.
        """
        pass

    def on_start(self, context: ConsiderationContext) -> None:
        """Called when this action starts executing."""
        pass

    def on_end(self, context: ConsiderationContext) -> None:
        """Called when this action finishes executing."""
        pass


class FunctionAction(UtilityAction):
    """An action that executes a function."""

    def __init__(
        self,
        name: str,
        func: Callable[[ConsiderationContext], bool],
        considerations: Optional[List[Consideration]] = None,
        base_score: float = 0.0,
        cooldown: float = 0.0,
    ) -> None:
        super().__init__(name, considerations, base_score, cooldown)
        self._func = func

    def execute(self, context: ConsiderationContext) -> bool:
        return self._func(context)


# =============================================================================
# Utility AI Brain
# =============================================================================


@dataclass
class UtilityAIState:
    """Current state of the utility AI."""
    current_action: Optional[UtilityAction] = None
    current_score: float = 0.0
    all_scores: List[ActionScore] = field(default_factory=list)
    last_update_time: float = 0.0
    action_history: List[str] = field(default_factory=list)


class UtilityAI:
    """
    A utility-based AI decision maker.

    Evaluates all available actions and selects the one with the highest
    utility score.
    """

    def __init__(
        self,
        name: str = "UtilityAI",
        update_rate: float = UTILITY_DEFAULT_UPDATE_RATE,
        momentum: float = UTILITY_DEFAULT_MOMENTUM,
        min_score_threshold: float = UTILITY_MIN_SCORE_THRESHOLD,
        history_size: int = UTILITY_ACTION_HISTORY_SIZE,
        entity_id: Optional[int] = None,
    ) -> None:
        self.name = name
        self.update_rate = update_rate
        self.momentum = momentum
        self.min_score_threshold = min_score_threshold
        self.history_size = history_size
        self.entity_id = entity_id

        self._actions: List[UtilityAction] = []
        self._state = UtilityAIState()
        self._blackboard: Optional[Blackboard] = None

    @property
    def actions(self) -> List[UtilityAction]:
        """Get all registered actions."""
        return self._actions

    @property
    def state(self) -> UtilityAIState:
        """Get the current state."""
        return self._state

    @property
    def current_action(self) -> Optional[UtilityAction]:
        """Get the currently executing action."""
        return self._state.current_action

    def add_action(self, action: UtilityAction) -> "UtilityAI":
        """Add an action to the AI."""
        self._actions.append(action)
        return self

    def remove_action(self, action: UtilityAction) -> bool:
        """Remove an action from the AI."""
        if action in self._actions:
            self._actions.remove(action)
            return True
        return False

    def set_blackboard(self, blackboard: Blackboard) -> None:
        """Set the blackboard for this AI."""
        self._blackboard = blackboard

    def evaluate(
        self,
        entity: Any = None,
        target: Any = None,
        world_state: Optional[Dict[str, Any]] = None,
        current_time: float = 0.0,
    ) -> List[ActionScore]:
        """
        Evaluate all actions and return their scores.

        Returns a list of ActionScore objects sorted by score (highest first).
        """
        context = ConsiderationContext(
            entity=entity,
            blackboard=self._blackboard,
            target=target,
            world_state=world_state or {},
        )

        scores: List[ActionScore] = []

        for action in self._actions:
            # Skip actions on cooldown
            if action.is_on_cooldown(current_time):
                continue

            score = action.calculate_score(context)

            # Log consideration scores during evaluation
            if self.entity_id is not None:
                self._log_consideration_scores(score)

            # Apply momentum if this is the current action
            if action == self._state.current_action and self.momentum > 0:
                score.score += self.momentum

            scores.append(score)

        # Sort by score (highest first)
        scores.sort(key=lambda s: s.score, reverse=True)
        self._state.all_scores = scores

        return scores

    def _log_consideration_scores(self, action_score: ActionScore) -> None:
        """Log consideration scores during evaluation."""
        try:
            from .ai_events import get_ai_event_logger
            logger = get_ai_event_logger()

            for consideration_name, score in action_score.consideration_scores.items():
                logger.log_utility_score_computed(
                    entity_id=self.entity_id,
                    consideration=consideration_name,
                    score=score,
                )
        except ImportError:
            pass  # AI events module not available

    def select_action(
        self,
        entity: Any = None,
        target: Any = None,
        world_state: Optional[Dict[str, Any]] = None,
        current_time: float = 0.0,
    ) -> Optional[UtilityAction]:
        """
        Select the best action based on utility scores.

        Returns the selected action or None if no action meets the threshold.
        """
        scores = self.evaluate(entity, target, world_state, current_time)

        if not scores:
            return None

        best = scores[0]

        if best.score < self.min_score_threshold:
            return None

        # Log the action selection event
        if self.entity_id is not None:
            self._log_action_selected(best, scores)

        # Track action change
        if best.action != self._state.current_action:
            if self._state.current_action is not None:
                context = ConsiderationContext(
                    entity=entity,
                    blackboard=self._blackboard,
                    target=target,
                    world_state=world_state or {},
                )
                self._state.current_action.on_end(context)

            self._state.current_action = best.action
            self._state.current_score = best.score

            # Update history
            self._state.action_history.append(best.action.name)
            if len(self._state.action_history) > self.history_size:
                self._state.action_history.pop(0)

        return best.action

    def _log_action_selected(
        self, selected: ActionScore, all_scores: List[ActionScore]
    ) -> None:
        """Log an action selection event to the AI event logger."""
        try:
            from .ai_events import get_ai_event_logger
            logger = get_ai_event_logger()

            # Log individual consideration scores
            for consideration_name, score in selected.consideration_scores.items():
                logger.log_utility_score_computed(
                    entity_id=self.entity_id,
                    consideration=consideration_name,
                    score=score,
                )

            # Log the action selection
            all_scores_dict = {s.action.name: s.score for s in all_scores}
            logger.log_utility_action_selected(
                entity_id=self.entity_id,
                action=selected.action.name,
                score=selected.score,
                all_scores=all_scores_dict,
            )
        except ImportError:
            pass  # AI events module not available

    def update(
        self,
        entity: Any = None,
        target: Any = None,
        world_state: Optional[Dict[str, Any]] = None,
        current_time: float = 0.0,
    ) -> bool:
        """
        Update the AI: select and execute the best action.

        Returns True if an action was executed successfully.
        """
        # Check update rate
        if current_time - self._state.last_update_time < self.update_rate:
            # Continue executing current action if any
            if self._state.current_action is not None:
                context = ConsiderationContext(
                    entity=entity,
                    blackboard=self._blackboard,
                    target=target,
                    world_state=world_state or {},
                )
                return self._state.current_action.execute(context)
            return False

        self._state.last_update_time = current_time

        action = self.select_action(entity, target, world_state, current_time)

        if action is None:
            return False

        context = ConsiderationContext(
            entity=entity,
            blackboard=self._blackboard,
            target=target,
            world_state=world_state or {},
        )

        action.on_start(context)
        result = action.execute(context)
        action._last_execution_time = current_time

        return result

    def reset(self) -> None:
        """Reset the AI state."""
        self._state = UtilityAIState()

    @classmethod
    def from_registry(cls, ai_id: str, *args: Any, **kwargs: Any) -> "UtilityAI":
        """
        Create a utility AI instance from the registry by ID.

        Args:
            ai_id: The utility_id used when decorating with @utility_ai.
            *args: Positional arguments to pass to the constructor.
            **kwargs: Keyword arguments to pass to the constructor.

        Returns:
            A new instance of the registered utility AI class.

        Raises:
            ValueError: If the ID is not found in the registry.
        """
        return create_utility_ai_from_registry(ai_id, *args, **kwargs)

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about the current state."""
        return {
            "name": self.name,
            "current_action": self._state.current_action.name if self._state.current_action else None,
            "current_score": self._state.current_score,
            "all_scores": [
                {"action": s.action.name, "score": s.score}
                for s in self._state.all_scores
            ],
            "history": self._state.action_history,
        }


# Registry tag
TAG_UTILITY_AI = "utility_ai"

# Import Foundation registry
try:
    from foundation import registry
    FOUNDATION_AVAILABLE = True
except ImportError:
    registry = None
    FOUNDATION_AVAILABLE = False


def utility_ai(
    id: str,
    update_rate: float = UTILITY_DEFAULT_UPDATE_RATE,
    description: Optional[str] = None,
    track_instances: bool = False,
) -> Callable[[type], type]:
    """
    Decorator to register a class as a utility AI with the Foundation Registry.

    Args:
        id: Unique identifier for this utility AI type.
        update_rate: How often to re-evaluate actions (seconds).
        description: Optional description of what this AI does.
        track_instances: If True, track all instances via WeakSet.

    Usage:
        @utility_ai(id="combat", update_rate=0.5)
        class CombatAI(UtilityAI):
            pass

        # Query all utility AIs:
        >>> from foundation import registry
        >>> registry.query(tag="utility_ai")
    """
    def decorator(cls: type) -> type:
        if not id:
            raise ValueError("id must be non-empty")
        if update_rate <= 0:
            raise ValueError(f"update_rate must be > 0, got {update_rate}")

        # Set class attributes for introspection
        cls._utility_ai = True
        cls._utility_id = id
        cls._utility_update_rate = update_rate
        cls._utility_description = description

        # Register with Foundation Registry if available
        if FOUNDATION_AVAILABLE and registry is not None:
            registry_name = f"utility_ai.{id}"
            try:
                registry.register(cls, name=registry_name, track_instances=track_instances)
            except ValueError:
                # Already registered - fine in reload scenarios
                pass

            # Add utility_ai tag
            registry.add_tag(cls, TAG_UTILITY_AI)

            # Store metadata
            registry.set_metadata(cls, "utility_id", id)
            registry.set_metadata(cls, "update_rate", update_rate)
            registry.set_metadata(cls, "considerations", [])
            if description:
                registry.set_metadata(cls, "description", description)

        return cls
    return decorator


def get_all_utility_ai() -> List[type]:
    """Get all registered utility AI classes."""
    if FOUNDATION_AVAILABLE and registry is not None:
        return registry.query(tag=TAG_UTILITY_AI)
    return []


def get_utility_ai_by_id(ai_id: str) -> Optional[type]:
    """Get a utility AI class by its ID."""
    if FOUNDATION_AVAILABLE and registry is not None:
        results = registry.query(tag=TAG_UTILITY_AI, utility_id=ai_id)
        return results[0] if results else None
    return None


def get_utility_ai_by_update_rate(update_rate: float) -> List[type]:
    """Get utility AI classes with the specified update rate."""
    if FOUNDATION_AVAILABLE and registry is not None:
        return registry.query(tag=TAG_UTILITY_AI, update_rate=update_rate)
    return []


def create_utility_ai_from_registry(ai_id: str, *args: Any, **kwargs: Any) -> "UtilityAI":
    """Create a utility AI instance from the registry."""
    cls = get_utility_ai_by_id(ai_id)
    if cls is None:
        raise ValueError(f"Utility AI '{ai_id}' not found in registry")
    return cls(*args, **kwargs)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Response Curves
    "ResponseCurve",
    "CustomResponseCurve",
    "LINEAR_CURVE",
    "QUADRATIC_CURVE",
    "EXPONENTIAL_CURVE",
    "LOGISTIC_CURVE",
    "INVERSE_CURVE",
    "SMOOTHSTEP_CURVE",
    # Considerations
    "ConsiderationContext",
    "Consideration",
    "BlackboardConsideration",
    "FunctionConsideration",
    "DistanceConsideration",
    "HealthConsideration",
    # Actions
    "ActionScore",
    "UtilityAction",
    "FunctionAction",
    # AI
    "UtilityAIState",
    "UtilityAI",
    # Decorator
    "utility_ai",
    # Registry
    "TAG_UTILITY_AI",
    "UTILITY_DEFAULT_UPDATE_RATE",
    "create_utility_ai_from_registry",
    "get_all_utility_ai",
    "get_utility_ai_by_id",
    "get_utility_ai_by_update_rate",
]
