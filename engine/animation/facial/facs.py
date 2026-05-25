"""
Facial Action Coding System (FACS) Implementation.

Provides FACS Action Units mapping to blend shapes for anatomically
accurate facial expressions based on the Ekman FACS system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional


# =============================================================================
# Action Unit Definitions
# =============================================================================


class ActionUnit(Enum):
    """
    FACS Action Units representing individual facial muscle movements.

    Each AU corresponds to a specific facial muscle action.
    Based on the Ekman & Friesen Facial Action Coding System.
    """
    # Upper face
    AU1_INNER_BROW_RAISER = auto()    # Frontalis (medial)
    AU2_OUTER_BROW_RAISER = auto()    # Frontalis (lateral)
    AU4_BROW_LOWERER = auto()         # Corrugator supercilii, Depressor supercilii
    AU5_UPPER_LID_RAISER = auto()     # Levator palpebrae superioris
    AU6_CHEEK_RAISER = auto()         # Orbicularis oculi (orbital)
    AU7_LID_TIGHTENER = auto()        # Orbicularis oculi (palpebral)

    # Nose
    AU9_NOSE_WRINKLER = auto()        # Levator labii superioris alaeque nasi

    # Mouth
    AU10_UPPER_LIP_RAISER = auto()    # Levator labii superioris
    AU12_LIP_CORNER_PULLER = auto()   # Zygomaticus major
    AU14_DIMPLER = auto()             # Buccinator
    AU15_LIP_CORNER_DEPRESSOR = auto()# Depressor anguli oris
    AU17_CHIN_RAISER = auto()         # Mentalis
    AU20_LIP_STRETCHER = auto()       # Risorius
    AU23_LIP_TIGHTENER = auto()       # Orbicularis oris
    AU24_LIP_PRESSOR = auto()         # Orbicularis oris
    AU25_LIPS_PART = auto()           # Depressor labii inferioris, relaxed Mentalis
    AU26_JAW_DROP = auto()            # Masseter, Temporalis relaxed
    AU27_MOUTH_STRETCH = auto()       # Pterygoids, Digastric
    AU28_LIP_SUCK = auto()            # Orbicularis oris

    # Eyes
    AU43_EYES_CLOSED = auto()         # Orbicularis oculi


# Alias for convenience
AU = ActionUnit


# =============================================================================
# Action Unit Data
# =============================================================================


@dataclass
class ActionUnitData:
    """
    Data for a single Action Unit configuration.

    Attributes:
        au: The Action Unit
        intensity: Current intensity (0-1)
        blend_shapes: Mapping of blend shape names to contribution weights
        left_shapes: Left-side blend shapes (for bilateral AUs)
        right_shapes: Right-side blend shapes (for bilateral AUs)
        is_bilateral: Whether this AU has left/right variants
    """
    au: ActionUnit
    intensity: float = 0.0
    blend_shapes: dict[str, float] = field(default_factory=dict)
    left_shapes: dict[str, float] = field(default_factory=dict)
    right_shapes: dict[str, float] = field(default_factory=dict)
    is_bilateral: bool = False

    def get_blend_weights(
        self,
        left_intensity: Optional[float] = None,
        right_intensity: Optional[float] = None,
    ) -> dict[str, float]:
        """
        Get blend shape weights for current intensity.

        Args:
            left_intensity: Override for left side (bilateral AUs)
            right_intensity: Override for right side (bilateral AUs)

        Returns:
            Dictionary of blend shape names to weights
        """
        result = {}

        # Non-bilateral shapes
        for shape_name, weight in self.blend_shapes.items():
            result[shape_name] = weight * self.intensity

        # Bilateral shapes
        if self.is_bilateral:
            left_int = left_intensity if left_intensity is not None else self.intensity
            right_int = right_intensity if right_intensity is not None else self.intensity

            for shape_name, weight in self.left_shapes.items():
                result[shape_name] = weight * left_int

            for shape_name, weight in self.right_shapes.items():
                result[shape_name] = weight * right_int

        return result


# =============================================================================
# Default AU to Blend Shape Mappings
# =============================================================================


def get_default_au_mappings() -> dict[ActionUnit, ActionUnitData]:
    """
    Get default mappings from Action Units to blend shapes.

    Uses ARKit-compatible blend shape names where possible.

    Returns:
        Dictionary mapping AUs to their blend shape configurations
    """
    return {
        ActionUnit.AU1_INNER_BROW_RAISER: ActionUnitData(
            au=ActionUnit.AU1_INNER_BROW_RAISER,
            is_bilateral=False,
            blend_shapes={"browInnerUp": 1.0},
        ),
        ActionUnit.AU2_OUTER_BROW_RAISER: ActionUnitData(
            au=ActionUnit.AU2_OUTER_BROW_RAISER,
            is_bilateral=True,
            left_shapes={"browOuterUpLeft": 1.0},
            right_shapes={"browOuterUpRight": 1.0},
        ),
        ActionUnit.AU4_BROW_LOWERER: ActionUnitData(
            au=ActionUnit.AU4_BROW_LOWERER,
            is_bilateral=True,
            left_shapes={"browDownLeft": 1.0},
            right_shapes={"browDownRight": 1.0},
        ),
        ActionUnit.AU5_UPPER_LID_RAISER: ActionUnitData(
            au=ActionUnit.AU5_UPPER_LID_RAISER,
            is_bilateral=True,
            left_shapes={"eyeWideLeft": 1.0},
            right_shapes={"eyeWideRight": 1.0},
        ),
        ActionUnit.AU6_CHEEK_RAISER: ActionUnitData(
            au=ActionUnit.AU6_CHEEK_RAISER,
            is_bilateral=True,
            left_shapes={"cheekSquintLeft": 1.0},
            right_shapes={"cheekSquintRight": 1.0},
        ),
        ActionUnit.AU7_LID_TIGHTENER: ActionUnitData(
            au=ActionUnit.AU7_LID_TIGHTENER,
            is_bilateral=True,
            left_shapes={"eyeSquintLeft": 1.0},
            right_shapes={"eyeSquintRight": 1.0},
        ),
        ActionUnit.AU9_NOSE_WRINKLER: ActionUnitData(
            au=ActionUnit.AU9_NOSE_WRINKLER,
            is_bilateral=True,
            left_shapes={"noseSneerLeft": 1.0},
            right_shapes={"noseSneerRight": 1.0},
        ),
        ActionUnit.AU10_UPPER_LIP_RAISER: ActionUnitData(
            au=ActionUnit.AU10_UPPER_LIP_RAISER,
            is_bilateral=True,
            left_shapes={"mouthUpperUpLeft": 1.0},
            right_shapes={"mouthUpperUpRight": 1.0},
        ),
        ActionUnit.AU12_LIP_CORNER_PULLER: ActionUnitData(
            au=ActionUnit.AU12_LIP_CORNER_PULLER,
            is_bilateral=True,
            left_shapes={"mouthSmileLeft": 1.0},
            right_shapes={"mouthSmileRight": 1.0},
        ),
        ActionUnit.AU14_DIMPLER: ActionUnitData(
            au=ActionUnit.AU14_DIMPLER,
            is_bilateral=True,
            left_shapes={"mouthDimpleLeft": 1.0},
            right_shapes={"mouthDimpleRight": 1.0},
        ),
        ActionUnit.AU15_LIP_CORNER_DEPRESSOR: ActionUnitData(
            au=ActionUnit.AU15_LIP_CORNER_DEPRESSOR,
            is_bilateral=True,
            left_shapes={"mouthFrownLeft": 1.0},
            right_shapes={"mouthFrownRight": 1.0},
        ),
        ActionUnit.AU17_CHIN_RAISER: ActionUnitData(
            au=ActionUnit.AU17_CHIN_RAISER,
            is_bilateral=False,
            blend_shapes={"mouthShrugLower": 1.0},
        ),
        ActionUnit.AU20_LIP_STRETCHER: ActionUnitData(
            au=ActionUnit.AU20_LIP_STRETCHER,
            is_bilateral=True,
            left_shapes={"mouthStretchLeft": 1.0},
            right_shapes={"mouthStretchRight": 1.0},
        ),
        ActionUnit.AU23_LIP_TIGHTENER: ActionUnitData(
            au=ActionUnit.AU23_LIP_TIGHTENER,
            is_bilateral=False,
            blend_shapes={"mouthPucker": 0.5},
        ),
        ActionUnit.AU24_LIP_PRESSOR: ActionUnitData(
            au=ActionUnit.AU24_LIP_PRESSOR,
            is_bilateral=True,
            left_shapes={"mouthPressLeft": 1.0},
            right_shapes={"mouthPressRight": 1.0},
        ),
        ActionUnit.AU25_LIPS_PART: ActionUnitData(
            au=ActionUnit.AU25_LIPS_PART,
            is_bilateral=False,
            blend_shapes={"jawOpen": 0.3, "mouthClose": -0.5},
        ),
        ActionUnit.AU26_JAW_DROP: ActionUnitData(
            au=ActionUnit.AU26_JAW_DROP,
            is_bilateral=False,
            blend_shapes={"jawOpen": 0.7},
        ),
        ActionUnit.AU27_MOUTH_STRETCH: ActionUnitData(
            au=ActionUnit.AU27_MOUTH_STRETCH,
            is_bilateral=False,
            blend_shapes={"jawOpen": 1.0},
        ),
        ActionUnit.AU28_LIP_SUCK: ActionUnitData(
            au=ActionUnit.AU28_LIP_SUCK,
            is_bilateral=False,
            blend_shapes={"mouthRollLower": 1.0, "mouthRollUpper": 1.0},
        ),
        ActionUnit.AU43_EYES_CLOSED: ActionUnitData(
            au=ActionUnit.AU43_EYES_CLOSED,
            is_bilateral=True,
            left_shapes={"eyeBlinkLeft": 1.0},
            right_shapes={"eyeBlinkRight": 1.0},
        ),
    }


# =============================================================================
# Expression Definitions
# =============================================================================


class Expression(Enum):
    """
    Standard emotional expression presets.

    Based on Ekman's universal emotions and common expressions.
    """
    NEUTRAL = auto()
    HAPPY = auto()
    SAD = auto()
    ANGRY = auto()
    SURPRISED = auto()
    DISGUSTED = auto()
    FEARFUL = auto()
    CONTEMPT = auto()


@dataclass
class ExpressionData:
    """
    Data for an expression preset.

    Attributes:
        expression: The expression type
        au_weights: Dictionary of AU to intensity
        au_left_weights: Left-side weights for bilateral AUs
        au_right_weights: Right-side weights for bilateral AUs
    """
    expression: Expression
    au_weights: dict[ActionUnit, float] = field(default_factory=dict)
    au_left_weights: dict[ActionUnit, float] = field(default_factory=dict)
    au_right_weights: dict[ActionUnit, float] = field(default_factory=dict)


def get_default_expressions() -> dict[Expression, ExpressionData]:
    """
    Get default expression presets using FACS Action Units.

    Returns:
        Dictionary mapping Expression to ExpressionData
    """
    return {
        Expression.NEUTRAL: ExpressionData(
            expression=Expression.NEUTRAL,
            au_weights={},
        ),

        Expression.HAPPY: ExpressionData(
            expression=Expression.HAPPY,
            au_weights={
                ActionUnit.AU6_CHEEK_RAISER: 0.8,
                ActionUnit.AU12_LIP_CORNER_PULLER: 1.0,
                ActionUnit.AU25_LIPS_PART: 0.3,
            },
        ),

        Expression.SAD: ExpressionData(
            expression=Expression.SAD,
            au_weights={
                ActionUnit.AU1_INNER_BROW_RAISER: 0.8,
                ActionUnit.AU4_BROW_LOWERER: 0.4,
                ActionUnit.AU15_LIP_CORNER_DEPRESSOR: 0.7,
                ActionUnit.AU17_CHIN_RAISER: 0.5,
            },
        ),

        Expression.ANGRY: ExpressionData(
            expression=Expression.ANGRY,
            au_weights={
                ActionUnit.AU4_BROW_LOWERER: 1.0,
                ActionUnit.AU5_UPPER_LID_RAISER: 0.5,
                ActionUnit.AU7_LID_TIGHTENER: 0.7,
                ActionUnit.AU23_LIP_TIGHTENER: 0.6,
                ActionUnit.AU24_LIP_PRESSOR: 0.5,
            },
        ),

        Expression.SURPRISED: ExpressionData(
            expression=Expression.SURPRISED,
            au_weights={
                ActionUnit.AU1_INNER_BROW_RAISER: 1.0,
                ActionUnit.AU2_OUTER_BROW_RAISER: 1.0,
                ActionUnit.AU5_UPPER_LID_RAISER: 0.8,
                ActionUnit.AU26_JAW_DROP: 0.6,
            },
        ),

        Expression.DISGUSTED: ExpressionData(
            expression=Expression.DISGUSTED,
            au_weights={
                ActionUnit.AU9_NOSE_WRINKLER: 1.0,
                ActionUnit.AU10_UPPER_LIP_RAISER: 0.7,
                ActionUnit.AU4_BROW_LOWERER: 0.5,
                ActionUnit.AU7_LID_TIGHTENER: 0.4,
            },
        ),

        Expression.FEARFUL: ExpressionData(
            expression=Expression.FEARFUL,
            au_weights={
                ActionUnit.AU1_INNER_BROW_RAISER: 1.0,
                ActionUnit.AU2_OUTER_BROW_RAISER: 0.6,
                ActionUnit.AU4_BROW_LOWERER: 0.3,
                ActionUnit.AU5_UPPER_LID_RAISER: 0.9,
                ActionUnit.AU7_LID_TIGHTENER: 0.4,
                ActionUnit.AU20_LIP_STRETCHER: 0.8,
                ActionUnit.AU25_LIPS_PART: 0.5,
            },
        ),

        Expression.CONTEMPT: ExpressionData(
            expression=Expression.CONTEMPT,
            au_weights={},
            # Contempt is asymmetric - slight smile on one side
            au_left_weights={
                ActionUnit.AU12_LIP_CORNER_PULLER: 0.0,
                ActionUnit.AU14_DIMPLER: 0.0,
            },
            au_right_weights={
                ActionUnit.AU12_LIP_CORNER_PULLER: 0.5,
                ActionUnit.AU14_DIMPLER: 0.6,
            },
        ),
    }


# =============================================================================
# FACS Controller
# =============================================================================


class FACSController:
    """
    Controller for FACS-based facial animation.

    Maps Action Units to blend shapes and provides expression presets.
    """

    def __init__(
        self,
        au_mappings: Optional[dict[ActionUnit, ActionUnitData]] = None,
        expression_presets: Optional[dict[Expression, ExpressionData]] = None,
        on_weights_changed: Optional[Callable[[dict[str, float]], None]] = None,
    ) -> None:
        """
        Initialize the FACS controller.

        Args:
            au_mappings: Custom AU to blend shape mappings
            expression_presets: Custom expression presets
            on_weights_changed: Callback when blend weights change
        """
        self._au_mappings = au_mappings or get_default_au_mappings()
        self._expression_presets = expression_presets or get_default_expressions()
        self._on_weights_changed = on_weights_changed

        # Current state
        self._au_intensities: dict[ActionUnit, float] = {au: 0.0 for au in ActionUnit}
        self._au_left_intensities: dict[ActionUnit, float] = {}
        self._au_right_intensities: dict[ActionUnit, float] = {}

        # Blending
        self._current_expression: Optional[Expression] = None
        self._target_expression: Optional[Expression] = None
        self._blend_progress: float = 1.0
        self._blend_speed: float = 5.0

        self._dirty = False

    @property
    def current_expression(self) -> Optional[Expression]:
        """Get current expression."""
        return self._current_expression

    @property
    def au_intensities(self) -> dict[ActionUnit, float]:
        """Get current AU intensities."""
        return self._au_intensities.copy()

    @property
    def dirty(self) -> bool:
        """Check if state has changed."""
        return self._dirty

    def set_au_intensity(
        self,
        au: ActionUnit,
        intensity: float,
        left: Optional[float] = None,
        right: Optional[float] = None,
    ) -> None:
        """
        Set intensity for an Action Unit.

        Args:
            au: The Action Unit
            intensity: Intensity value (0-1)
            left: Left-side intensity for bilateral AUs
            right: Right-side intensity for bilateral AUs
        """
        intensity = max(0.0, min(1.0, intensity))
        self._au_intensities[au] = intensity

        if left is not None:
            self._au_left_intensities[au] = max(0.0, min(1.0, left))
        elif au in self._au_left_intensities:
            del self._au_left_intensities[au]

        if right is not None:
            self._au_right_intensities[au] = max(0.0, min(1.0, right))
        elif au in self._au_right_intensities:
            del self._au_right_intensities[au]

        self._dirty = True
        self._notify_change()

    def get_au_intensity(self, au: ActionUnit) -> float:
        """Get intensity for an Action Unit."""
        return self._au_intensities.get(au, 0.0)

    def reset_all_aus(self) -> None:
        """Reset all AU intensities to zero."""
        self._au_intensities = {au: 0.0 for au in ActionUnit}
        self._au_left_intensities.clear()
        self._au_right_intensities.clear()
        self._current_expression = None
        self._dirty = True
        self._notify_change()

    def set_expression(
        self,
        expression: Expression,
        intensity: float = 1.0,
        blend_time: float = 0.0,
    ) -> None:
        """
        Set expression preset.

        Args:
            expression: The expression to set
            intensity: Expression intensity (0-1)
            blend_time: Time to blend to expression (0 for instant)
        """
        expression_data = self._expression_presets.get(expression)
        if not expression_data:
            return

        if blend_time <= 0:
            # Instant transition
            self.reset_all_aus()

            for au, weight in expression_data.au_weights.items():
                self._au_intensities[au] = weight * intensity

            for au, weight in expression_data.au_left_weights.items():
                self._au_left_intensities[au] = weight * intensity

            for au, weight in expression_data.au_right_weights.items():
                self._au_right_intensities[au] = weight * intensity

            self._current_expression = expression
            self._target_expression = None
            self._blend_progress = 1.0
        else:
            # Set up blend
            self._target_expression = expression
            self._blend_progress = 0.0
            self._blend_speed = 1.0 / blend_time

        self._dirty = True
        self._notify_change()

    def create_expression(
        self,
        expression_name: str,
    ) -> dict[ActionUnit, float]:
        """
        Create AU weights for a named expression.

        Args:
            expression_name: Name of expression (case-insensitive)

        Returns:
            Dictionary of AU to weight
        """
        try:
            expression = Expression[expression_name.upper()]
        except KeyError:
            return {}

        expression_data = self._expression_presets.get(expression)
        if not expression_data:
            return {}

        return expression_data.au_weights.copy()

    def blend_expressions(
        self,
        expression_a: Expression,
        expression_b: Expression,
        blend_factor: float,
    ) -> dict[ActionUnit, float]:
        """
        Blend between two expressions.

        Args:
            expression_a: First expression
            expression_b: Second expression
            blend_factor: Blend factor (0 = A, 1 = B)

        Returns:
            Blended AU weights
        """
        blend_factor = max(0.0, min(1.0, blend_factor))

        data_a = self._expression_presets.get(expression_a)
        data_b = self._expression_presets.get(expression_b)

        if not data_a or not data_b:
            return {}

        result = {}
        all_aus = set(data_a.au_weights.keys()) | set(data_b.au_weights.keys())

        for au in all_aus:
            weight_a = data_a.au_weights.get(au, 0.0)
            weight_b = data_b.au_weights.get(au, 0.0)
            result[au] = weight_a * (1.0 - blend_factor) + weight_b * blend_factor

        return result

    def update(self, dt: float) -> bool:
        """
        Update expression blending.

        Args:
            dt: Delta time in seconds

        Returns:
            True if state changed
        """
        if self._target_expression is None or self._blend_progress >= 1.0:
            return False

        self._blend_progress = min(1.0, self._blend_progress + self._blend_speed * dt)

        target_data = self._expression_presets.get(self._target_expression)
        if not target_data:
            return False

        # Blend AU intensities toward target
        for au in ActionUnit:
            target_intensity = target_data.au_weights.get(au, 0.0)
            current = self._au_intensities.get(au, 0.0)
            self._au_intensities[au] = current + (target_intensity - current) * self._blend_speed * dt

        if self._blend_progress >= 1.0:
            self._current_expression = self._target_expression
            self._target_expression = None

        self._dirty = True
        self._notify_change()
        return True

    def get_blend_shape_weights(self) -> dict[str, float]:
        """
        Get current blend shape weights from AU intensities.

        Returns:
            Dictionary of blend shape names to weights
        """
        result: dict[str, float] = {}

        for au, intensity in self._au_intensities.items():
            if intensity < 0.001:
                continue

            au_data = self._au_mappings.get(au)
            if not au_data:
                continue

            # Update the AU data's intensity
            au_data.intensity = intensity

            # Get bilateral overrides
            left_int = self._au_left_intensities.get(au)
            right_int = self._au_right_intensities.get(au)

            weights = au_data.get_blend_weights(left_int, right_int)

            for shape_name, weight in weights.items():
                if shape_name in result:
                    # Combine weights (additive, clamped)
                    result[shape_name] = min(1.0, max(-1.0, result[shape_name] + weight))
                else:
                    result[shape_name] = weight

        return result

    def add_expression_preset(
        self,
        expression: Expression,
        au_weights: dict[ActionUnit, float],
        au_left_weights: Optional[dict[ActionUnit, float]] = None,
        au_right_weights: Optional[dict[ActionUnit, float]] = None,
    ) -> None:
        """
        Add or update an expression preset.

        Args:
            expression: The expression to define
            au_weights: AU intensity weights
            au_left_weights: Left-side weights for bilateral
            au_right_weights: Right-side weights for bilateral
        """
        self._expression_presets[expression] = ExpressionData(
            expression=expression,
            au_weights=au_weights,
            au_left_weights=au_left_weights or {},
            au_right_weights=au_right_weights or {},
        )

    def set_au_mapping(self, au: ActionUnit, au_data: ActionUnitData) -> None:
        """
        Set or update AU to blend shape mapping.

        Args:
            au: The Action Unit
            au_data: The mapping data
        """
        self._au_mappings[au] = au_data

    def get_active_aus(self, threshold: float = 0.001) -> list[ActionUnit]:
        """
        Get list of active Action Units.

        Args:
            threshold: Minimum intensity to consider active

        Returns:
            List of active AUs
        """
        return [au for au, intensity in self._au_intensities.items() if intensity >= threshold]

    def clear_dirty(self) -> None:
        """Clear the dirty flag."""
        self._dirty = False

    def _notify_change(self) -> None:
        """Notify change callback."""
        if self._on_weights_changed:
            self._on_weights_changed(self.get_blend_shape_weights())

    def to_dict(self) -> dict[str, Any]:
        """Serialize current state to dictionary."""
        return {
            "au_intensities": {au.name: intensity for au, intensity in self._au_intensities.items()},
            "au_left_intensities": {au.name: intensity for au, intensity in self._au_left_intensities.items()},
            "au_right_intensities": {au.name: intensity for au, intensity in self._au_right_intensities.items()},
            "current_expression": self._current_expression.name if self._current_expression else None,
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        """Restore state from dictionary."""
        if "au_intensities" in data:
            for au_name, intensity in data["au_intensities"].items():
                try:
                    au = ActionUnit[au_name]
                    self._au_intensities[au] = intensity
                except KeyError:
                    pass

        if "au_left_intensities" in data:
            for au_name, intensity in data["au_left_intensities"].items():
                try:
                    au = ActionUnit[au_name]
                    self._au_left_intensities[au] = intensity
                except KeyError:
                    pass

        if "au_right_intensities" in data:
            for au_name, intensity in data["au_right_intensities"].items():
                try:
                    au = ActionUnit[au_name]
                    self._au_right_intensities[au] = intensity
                except KeyError:
                    pass

        if data.get("current_expression"):
            try:
                self._current_expression = Expression[data["current_expression"]]
            except KeyError:
                pass

        self._dirty = True
