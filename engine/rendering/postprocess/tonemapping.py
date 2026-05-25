"""
Tone Mapping Operators

Converts HDR scene values to displayable LDR range:
- Reinhard: Simple luminance mapping
- ACES: Academy Color Encoding System (film standard)
- AgX: Modern alternative to ACES
- Filmic: Uncharted 2 style
- CustomCurve: Artist-defined curve
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from .postprocess_stack import EffectPriority, EffectSettings, PostProcessEffect


class TonemapOperator(Enum):
    """Available tonemap operators."""

    REINHARD = auto()
    REINHARD_EXTENDED = auto()
    ACES = auto()
    ACES_FITTED = auto()
    AGX = auto()
    FILMIC = auto()
    HABLE = auto()  # Uncharted 2
    NEUTRAL = auto()
    CUSTOM = auto()


@dataclass
class TonemapCurvePoint:
    """Control point for custom tonemap curve."""

    input_value: float  # Input luminance [0, inf)
    output_value: float  # Output value [0, 1]
    slope: float = 1.0  # Tangent slope at this point


@dataclass
class CustomCurveSettings:
    """Settings for custom tonemap curve."""

    points: List[TonemapCurvePoint] = field(default_factory=list)
    interpolation: str = "cubic"  # "linear" | "cubic" | "hermite"

    def __post_init__(self) -> None:
        if not self.points:
            self.points = [
                TonemapCurvePoint(0.0, 0.0, 1.0),
                TonemapCurvePoint(0.18, 0.18, 1.0),  # Middle gray
                TonemapCurvePoint(1.0, 0.8, 0.5),
                TonemapCurvePoint(10.0, 0.95, 0.1),
            ]


@dataclass
class TonemapSettings(EffectSettings):
    """Tone mapping settings."""

    operator: TonemapOperator = TonemapOperator.ACES_FITTED

    # General settings
    exposure_bias: float = 0.0  # Pre-tonemap exposure adjustment
    white_point: float = 11.2  # White level for Reinhard/Filmic
    saturation: float = 1.0  # Post-tonemap saturation

    # ACES settings
    aces_input_scale: float = 0.6  # RRT input scale
    aces_output_scale: float = 1.0  # ODT output scale

    # AgX settings
    agx_look: str = "none"  # "none" | "golden" | "punchy"
    agx_saturation: float = 1.0

    # Filmic settings
    filmic_shoulder_strength: float = 0.22
    filmic_linear_strength: float = 0.3
    filmic_linear_angle: float = 0.1
    filmic_toe_strength: float = 0.2
    filmic_toe_numerator: float = 0.01
    filmic_toe_denominator: float = 0.3

    # Custom curve
    custom_curve: CustomCurveSettings = field(default_factory=CustomCurveSettings)

    # Color correction
    color_filter: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    gamma: float = 2.2

    def __post_init__(self) -> None:
        self.priority = EffectPriority.TONEMAPPING.value

    def lerp(self, other: "TonemapSettings", t: float) -> "TonemapSettings":
        """Interpolate between two tonemap settings."""
        return TonemapSettings(
            enabled=self.enabled if t < 0.5 else other.enabled,
            weight=self.weight + (other.weight - self.weight) * t,
            operator=self.operator if t < 0.5 else other.operator,
            exposure_bias=self.exposure_bias
            + (other.exposure_bias - self.exposure_bias) * t,
            white_point=self.white_point + (other.white_point - self.white_point) * t,
            saturation=self.saturation + (other.saturation - self.saturation) * t,
            aces_input_scale=self.aces_input_scale
            + (other.aces_input_scale - self.aces_input_scale) * t,
            aces_output_scale=self.aces_output_scale
            + (other.aces_output_scale - self.aces_output_scale) * t,
            agx_look=self.agx_look if t < 0.5 else other.agx_look,
            agx_saturation=self.agx_saturation
            + (other.agx_saturation - self.agx_saturation) * t,
            color_filter=tuple(
                self.color_filter[i]
                + (other.color_filter[i] - self.color_filter[i]) * t
                for i in range(3)
            ),
            gamma=self.gamma + (other.gamma - self.gamma) * t,
        )


class TonemapFunction(ABC):
    """Abstract base for tonemap functions."""

    @abstractmethod
    def apply(
        self,
        r: float,
        g: float,
        b: float,
        settings: TonemapSettings,
    ) -> Tuple[float, float, float]:
        """Apply tonemapping to RGB values.

        Args:
            r: Red channel (HDR).
            g: Green channel (HDR).
            b: Blue channel (HDR).
            settings: Tonemap settings.

        Returns:
            Tonemapped RGB tuple [0, 1].
        """
        pass

    @staticmethod
    def luminance(r: float, g: float, b: float) -> float:
        """Calculate relative luminance.

        Args:
            r: Red channel.
            g: Green channel.
            b: Blue channel.

        Returns:
            Luminance value.
        """
        return 0.2126 * r + 0.7152 * g + 0.0722 * b


class Reinhard(TonemapFunction):
    """Reinhard tone mapping.

    Simple luminance-based mapping: L / (1 + L)
    """

    def apply(
        self,
        r: float,
        g: float,
        b: float,
        settings: TonemapSettings,
    ) -> Tuple[float, float, float]:
        """Apply Reinhard tonemapping."""
        lum = self.luminance(r, g, b)
        if lum <= 0:
            return (0.0, 0.0, 0.0)

        lum_mapped = lum / (1.0 + lum)

        scale = lum_mapped / lum
        return (
            min(1.0, r * scale),
            min(1.0, g * scale),
            min(1.0, b * scale),
        )


class ReinhardExtended(TonemapFunction):
    """Extended Reinhard with white point control.

    L_mapped = L * (1 + L/L_white^2) / (1 + L)
    """

    def apply(
        self,
        r: float,
        g: float,
        b: float,
        settings: TonemapSettings,
    ) -> Tuple[float, float, float]:
        """Apply extended Reinhard tonemapping."""
        lum = self.luminance(r, g, b)
        if lum <= 0:
            return (0.0, 0.0, 0.0)

        white_sq = settings.white_point * settings.white_point
        numerator = lum * (1.0 + lum / white_sq)
        lum_mapped = numerator / (1.0 + lum)

        scale = lum_mapped / lum
        return (
            min(1.0, r * scale),
            min(1.0, g * scale),
            min(1.0, b * scale),
        )


class ACES(TonemapFunction):
    """ACES (Academy Color Encoding System) filmic tonemapping.

    Reference implementation with RRT + ODT approximation.
    """

    # sRGB -> ACEScg matrix
    SRGB_TO_ACES = [
        [0.59719, 0.35458, 0.04823],
        [0.07600, 0.90834, 0.01566],
        [0.02840, 0.13383, 0.83777],
    ]

    # ACEScg -> sRGB matrix
    ACES_TO_SRGB = [
        [1.60475, -0.53108, -0.07367],
        [-0.10208, 1.10813, -0.00605],
        [-0.00327, -0.07276, 1.07602],
    ]

    def apply(
        self,
        r: float,
        g: float,
        b: float,
        settings: TonemapSettings,
    ) -> Tuple[float, float, float]:
        """Apply ACES tonemapping."""
        scale = settings.aces_input_scale

        r_aces = (
            self.SRGB_TO_ACES[0][0] * r * scale
            + self.SRGB_TO_ACES[0][1] * g * scale
            + self.SRGB_TO_ACES[0][2] * b * scale
        )
        g_aces = (
            self.SRGB_TO_ACES[1][0] * r * scale
            + self.SRGB_TO_ACES[1][1] * g * scale
            + self.SRGB_TO_ACES[1][2] * b * scale
        )
        b_aces = (
            self.SRGB_TO_ACES[2][0] * r * scale
            + self.SRGB_TO_ACES[2][1] * g * scale
            + self.SRGB_TO_ACES[2][2] * b * scale
        )

        r_rrt = self._rrt_odt(r_aces)
        g_rrt = self._rrt_odt(g_aces)
        b_rrt = self._rrt_odt(b_aces)

        out_scale = settings.aces_output_scale
        r_out = (
            self.ACES_TO_SRGB[0][0] * r_rrt
            + self.ACES_TO_SRGB[0][1] * g_rrt
            + self.ACES_TO_SRGB[0][2] * b_rrt
        ) * out_scale
        g_out = (
            self.ACES_TO_SRGB[1][0] * r_rrt
            + self.ACES_TO_SRGB[1][1] * g_rrt
            + self.ACES_TO_SRGB[1][2] * b_rrt
        ) * out_scale
        b_out = (
            self.ACES_TO_SRGB[2][0] * r_rrt
            + self.ACES_TO_SRGB[2][1] * g_rrt
            + self.ACES_TO_SRGB[2][2] * b_rrt
        ) * out_scale

        return (
            max(0.0, min(1.0, r_out)),
            max(0.0, min(1.0, g_out)),
            max(0.0, min(1.0, b_out)),
        )

    def _rrt_odt(self, x: float) -> float:
        """Apply RRT + ODT approximation.

        Args:
            x: Input value in ACEScg.

        Returns:
            Tonemapped value.
        """
        a = x * (x + 0.0245786) - 0.000090537
        b = x * (0.983729 * x + 0.4329510) + 0.238081
        return a / b if b != 0 else 0.0


class ACESFitted(TonemapFunction):
    """Faster ACES approximation.

    Stephen Hill's fitted curve approximation.
    """

    def apply(
        self,
        r: float,
        g: float,
        b: float,
        settings: TonemapSettings,
    ) -> Tuple[float, float, float]:
        """Apply fitted ACES tonemapping."""
        scale = settings.aces_input_scale

        return (
            self._aces_curve(r * scale),
            self._aces_curve(g * scale),
            self._aces_curve(b * scale),
        )

    def _aces_curve(self, x: float) -> float:
        """Apply fitted ACES curve.

        Uses Stephen Hill's fitted ACES coefficients from constants.

        Args:
            x: Input value.

        Returns:
            Tonemapped value [0, 1].
        """
        from .constants import TONEMAP

        a = TONEMAP.ACES_A
        b = TONEMAP.ACES_B
        c = TONEMAP.ACES_C
        d = TONEMAP.ACES_D
        e = TONEMAP.ACES_E
        # Denominator is always positive since e > 0, so division is safe
        result = (x * (a * x + b)) / (x * (c * x + d) + e)
        return max(0.0, min(1.0, result))


class AgX(TonemapFunction):
    """AgX tone mapping.

    Modern alternative to ACES with better highlight handling
    and reduced color artifacts.
    """

    def apply(
        self,
        r: float,
        g: float,
        b: float,
        settings: TonemapSettings,
    ) -> Tuple[float, float, float]:
        """Apply AgX tonemapping."""
        from .constants import TONEMAP

        log_r = self._safe_log2(r)
        log_g = self._safe_log2(g)
        log_b = self._safe_log2(b)

        min_ev = TONEMAP.AGX_MIN_EV
        max_ev = TONEMAP.AGX_MAX_EV

        log_r = (log_r - min_ev) / (max_ev - min_ev)
        log_g = (log_g - min_ev) / (max_ev - min_ev)
        log_b = (log_b - min_ev) / (max_ev - min_ev)

        out_r = self._agx_curve(log_r)
        out_g = self._agx_curve(log_g)
        out_b = self._agx_curve(log_b)

        if settings.agx_look == "punchy":
            out_r, out_g, out_b = self._apply_punchy_look(out_r, out_g, out_b)
        elif settings.agx_look == "golden":
            out_r, out_g, out_b = self._apply_golden_look(out_r, out_g, out_b)

        if settings.agx_saturation != 1.0:
            lum = self.luminance(out_r, out_g, out_b)
            sat = settings.agx_saturation
            out_r = lum + (out_r - lum) * sat
            out_g = lum + (out_g - lum) * sat
            out_b = lum + (out_b - lum) * sat

        return (
            max(0.0, min(1.0, out_r)),
            max(0.0, min(1.0, out_g)),
            max(0.0, min(1.0, out_b)),
        )

    def _safe_log2(self, x: float) -> float:
        """Safe log2 that handles zero/negative values."""
        from .constants import SAFE_LOG_MIN
        return math.log2(max(SAFE_LOG_MIN, x))

    def _agx_curve(self, x: float) -> float:
        """AgX sigmoid curve."""
        x = max(0.0, min(1.0, x))
        x2 = x * x
        x4 = x2 * x2
        return (
            15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232
        )

    def _apply_punchy_look(
        self,
        r: float,
        g: float,
        b: float,
    ) -> Tuple[float, float, float]:
        """Apply punchy contrast look."""
        slope = 1.1
        power = 1.35
        sat = 1.4

        r = pow(max(0.0, r), power) * slope
        g = pow(max(0.0, g), power) * slope
        b = pow(max(0.0, b), power) * slope

        lum = self.luminance(r, g, b)
        r = lum + (r - lum) * sat
        g = lum + (g - lum) * sat
        b = lum + (b - lum) * sat

        return (r, g, b)

    def _apply_golden_look(
        self,
        r: float,
        g: float,
        b: float,
    ) -> Tuple[float, float, float]:
        """Apply warm golden look."""
        r *= 1.05
        b *= 0.9
        return (r, g, b)


class Filmic(TonemapFunction):
    """Filmic tone mapping (Uncharted 2 / Hable).

    Customizable S-curve with shoulder and toe controls.
    """

    def apply(
        self,
        r: float,
        g: float,
        b: float,
        settings: TonemapSettings,
    ) -> Tuple[float, float, float]:
        """Apply filmic tonemapping."""
        white_scale = 1.0 / self._filmic_curve(settings.white_point, settings)

        return (
            self._filmic_curve(r, settings) * white_scale,
            self._filmic_curve(g, settings) * white_scale,
            self._filmic_curve(b, settings) * white_scale,
        )

    def _filmic_curve(self, x: float, settings: TonemapSettings) -> float:
        """Apply filmic curve.

        Args:
            x: Input value.
            settings: Curve parameters.

        Returns:
            Curved output.
        """
        A = settings.filmic_shoulder_strength
        B = settings.filmic_linear_strength
        C = settings.filmic_linear_angle
        D = settings.filmic_toe_strength
        E = settings.filmic_toe_numerator
        F = settings.filmic_toe_denominator

        return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F


class CustomCurve(TonemapFunction):
    """Artist-defined custom curve tonemapping."""

    def apply(
        self,
        r: float,
        g: float,
        b: float,
        settings: TonemapSettings,
    ) -> Tuple[float, float, float]:
        """Apply custom curve tonemapping."""
        lum = self.luminance(r, g, b)
        if lum <= 0:
            return (0.0, 0.0, 0.0)

        lum_mapped = self._evaluate_curve(lum, settings.custom_curve)

        scale = lum_mapped / lum
        return (
            min(1.0, r * scale),
            min(1.0, g * scale),
            min(1.0, b * scale),
        )

    def _evaluate_curve(self, x: float, curve: CustomCurveSettings) -> float:
        """Evaluate custom curve at a point.

        Args:
            x: Input value.
            curve: Curve definition.

        Returns:
            Interpolated output value.
        """
        points = curve.points
        if not points:
            return x

        if x <= points[0].input_value:
            return points[0].output_value
        if x >= points[-1].input_value:
            return points[-1].output_value

        for i in range(len(points) - 1):
            p0 = points[i]
            p1 = points[i + 1]

            if p0.input_value <= x <= p1.input_value:
                # Guard against division by zero when points have same input value
                input_range = p1.input_value - p0.input_value
                if input_range <= 1e-10:
                    return p0.output_value
                t = (x - p0.input_value) / input_range

                if curve.interpolation == "linear":
                    return p0.output_value + (p1.output_value - p0.output_value) * t
                else:
                    t2 = t * t
                    t3 = t2 * t
                    h00 = 2 * t3 - 3 * t2 + 1
                    h10 = t3 - 2 * t2 + t
                    h01 = -2 * t3 + 3 * t2
                    h11 = t3 - t2
                    return (
                        h00 * p0.output_value
                        + h10 * p0.slope
                        + h01 * p1.output_value
                        + h11 * p1.slope
                    )

        return x


class TonemappingEffect(PostProcessEffect[TonemapSettings]):
    """Post-process effect for tone mapping."""

    def __init__(
        self,
        settings: Optional[TonemapSettings] = None,
    ) -> None:
        """Initialize tonemapping effect.

        Args:
            settings: Tonemap settings.
        """
        super().__init__(
            name="Tonemapping",
            settings=settings or TonemapSettings(),
            priority=EffectPriority.TONEMAPPING.value,
        )

        self._operators: Dict[TonemapOperator, TonemapFunction] = {
            TonemapOperator.REINHARD: Reinhard(),
            TonemapOperator.REINHARD_EXTENDED: ReinhardExtended(),
            TonemapOperator.ACES: ACES(),
            TonemapOperator.ACES_FITTED: ACESFitted(),
            TonemapOperator.AGX: AgX(),
            TonemapOperator.FILMIC: Filmic(),
            TonemapOperator.HABLE: Filmic(),  # Same as filmic
            TonemapOperator.NEUTRAL: ACESFitted(),  # Use ACES as fallback for NEUTRAL
            TonemapOperator.CUSTOM: CustomCurve(),
        }

    def get_required_inputs(self) -> List[str]:
        """Get required input resources."""
        return ["color"]

    def get_outputs(self) -> List[str]:
        """Get output resources."""
        return ["color"]

    def setup(self, width: int, height: int) -> None:
        """Initialize tonemapping resources."""
        pass

    def execute(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        delta_time: float,
    ) -> None:
        """Execute tonemapping.

        Args:
            inputs: Input HDR color.
            outputs: Output LDR color.
            delta_time: Frame time.
        """
        if not self._settings or not self._settings.enabled:
            return

    def cleanup(self) -> None:
        """Release tonemapping resources."""
        pass

    def tonemap_value(
        self,
        r: float,
        g: float,
        b: float,
    ) -> Tuple[float, float, float]:
        """Tonemap a single RGB value (for testing/preview).

        Args:
            r: Red channel.
            g: Green channel.
            b: Blue channel.

        Returns:
            Tonemapped RGB.
        """
        if not self._settings:
            return (r, g, b)

        exposure = 2.0 ** self._settings.exposure_bias
        r *= exposure
        g *= exposure
        b *= exposure

        operator = self._operators.get(
            self._settings.operator,
            self._operators[TonemapOperator.ACES_FITTED],
        )

        r, g, b = operator.apply(r, g, b, self._settings)

        cf = self._settings.color_filter
        r *= cf[0]
        g *= cf[1]
        b *= cf[2]

        if self._settings.saturation != 1.0:
            lum = TonemapFunction.luminance(r, g, b)
            sat = self._settings.saturation
            r = lum + (r - lum) * sat
            g = lum + (g - lum) * sat
            b = lum + (b - lum) * sat

        return (
            max(0.0, min(1.0, r)),
            max(0.0, min(1.0, g)),
            max(0.0, min(1.0, b)),
        )


__all__ = [
    "TonemapOperator",
    "TonemapCurvePoint",
    "CustomCurveSettings",
    "TonemapSettings",
    "TonemapFunction",
    "Reinhard",
    "ReinhardExtended",
    "ACES",
    "ACESFitted",
    "AgX",
    "Filmic",
    "CustomCurve",
    "TonemappingEffect",
]
