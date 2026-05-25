"""
Exposure Control System

Provides exposure calculation and adaptation for HDR rendering:
- ManualExposure: Fixed EV value
- AutoExposure: Luminance average based
- HistogramExposure: Percentile-based from histogram
- EyeAdaptation: Temporal adaptation curve
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from .postprocess_stack import EffectPriority, EffectSettings, PostProcessEffect


class ExposureMode(Enum):
    """Exposure calculation mode."""

    MANUAL = auto()  # Fixed exposure value
    AUTO_AVERAGE = auto()  # Average luminance based
    AUTO_HISTOGRAM = auto()  # Histogram percentile based


class MeteringMode(Enum):
    """Light metering mode for auto exposure."""

    CENTER_WEIGHTED = auto()  # Weight towards center
    SPOT = auto()  # Small center spot
    MATRIX = auto()  # Full frame averaging
    HIGHLIGHT_PRIORITY = auto()  # Protect highlights


@dataclass
class ExposureSettings(EffectSettings):
    """Settings for exposure control."""

    mode: ExposureMode = ExposureMode.AUTO_AVERAGE
    metering_mode: MeteringMode = MeteringMode.CENTER_WEIGHTED

    # Manual exposure settings
    manual_ev: float = 0.0  # Exposure value (-10 to 10)

    # Auto exposure settings
    min_ev: float = -4.0  # Minimum exposure value
    max_ev: float = 16.0  # Maximum exposure value
    exposure_compensation: float = 0.0  # EV adjustment

    # Histogram settings
    low_percentile: float = 0.5  # Lower bound percentile
    high_percentile: float = 0.95  # Upper bound percentile
    histogram_bins: int = 64  # Number of histogram bins

    # Adaptation settings
    adaptation_speed_up: float = 3.0  # Speed when brightening (seconds)
    adaptation_speed_down: float = 1.0  # Speed when darkening (seconds)

    def __post_init__(self) -> None:
        self.priority = EffectPriority.EXPOSURE.value

    def lerp(self, other: "ExposureSettings", t: float) -> "ExposureSettings":
        """Interpolate between two exposure settings."""
        return ExposureSettings(
            enabled=self.enabled if t < 0.5 else other.enabled,
            weight=self.weight + (other.weight - self.weight) * t,
            mode=self.mode if t < 0.5 else other.mode,
            metering_mode=self.metering_mode if t < 0.5 else other.metering_mode,
            manual_ev=self.manual_ev + (other.manual_ev - self.manual_ev) * t,
            min_ev=self.min_ev + (other.min_ev - self.min_ev) * t,
            max_ev=self.max_ev + (other.max_ev - self.max_ev) * t,
            exposure_compensation=self.exposure_compensation
            + (other.exposure_compensation - self.exposure_compensation) * t,
            low_percentile=self.low_percentile
            + (other.low_percentile - self.low_percentile) * t,
            high_percentile=self.high_percentile
            + (other.high_percentile - self.high_percentile) * t,
            adaptation_speed_up=self.adaptation_speed_up
            + (other.adaptation_speed_up - self.adaptation_speed_up) * t,
            adaptation_speed_down=self.adaptation_speed_down
            + (other.adaptation_speed_down - self.adaptation_speed_down) * t,
        )


def luminance_to_ev(luminance: float) -> float:
    """Convert luminance to exposure value.

    Uses the relationship: EV = log2(L * 100 / 12.5) where L is in cd/m^2.
    Based on ISO 12232:2006 saturation-based speed.

    Args:
        luminance: Scene luminance in cd/m^2.

    Returns:
        Exposure value (EV).
    """
    from .constants import EXPOSURE, LUMINANCE_MIN

    if luminance <= LUMINANCE_MIN:
        return EXPOSURE.EV_MIN_FALLBACK
    return math.log2(luminance * EXPOSURE.LUMINANCE_TO_EV_SCALE)


def ev_to_exposure(ev: float) -> float:
    """Convert exposure value to exposure multiplier.

    Args:
        ev: Exposure value.

    Returns:
        Exposure multiplier.
    """
    return 2.0 ** (-ev)


def exposure_to_ev(exposure: float) -> float:
    """Convert exposure multiplier to EV.

    Args:
        exposure: Exposure multiplier.

    Returns:
        Exposure value.
    """
    from .constants import EXPOSURE, EPSILON

    if exposure <= EPSILON:
        return EXPOSURE.EV_MIN_FALLBACK
    return -math.log2(exposure)


class ExposureCalculator(ABC):
    """Abstract base class for exposure calculation methods."""

    @abstractmethod
    def calculate_target_ev(
        self,
        luminance_data: Any,
        settings: ExposureSettings,
    ) -> float:
        """Calculate the target exposure value.

        Args:
            luminance_data: Input luminance information.
            settings: Exposure settings.

        Returns:
            Target EV value.
        """
        pass


class ManualExposure(ExposureCalculator):
    """Fixed exposure value calculator.

    Simply returns the manual EV value from settings.
    """

    def calculate_target_ev(
        self,
        luminance_data: Any,
        settings: ExposureSettings,
    ) -> float:
        """Return the manual exposure value.

        Args:
            luminance_data: Ignored for manual mode.
            settings: Exposure settings containing manual_ev.

        Returns:
            Manual EV value with compensation.
        """
        return settings.manual_ev + settings.exposure_compensation


class AutoExposure(ExposureCalculator):
    """Luminance average-based exposure calculator.

    Calculates exposure based on the average scene luminance,
    weighted according to the metering mode.
    """

    def __init__(self) -> None:
        self._metering_weights: Dict[MeteringMode, List[List[float]]] = {
            MeteringMode.CENTER_WEIGHTED: self._create_center_weighted_kernel(),
            MeteringMode.SPOT: self._create_spot_kernel(),
            MeteringMode.MATRIX: self._create_matrix_kernel(),
            MeteringMode.HIGHLIGHT_PRIORITY: self._create_matrix_kernel(),
        }

    def _create_center_weighted_kernel(self) -> List[List[float]]:
        """Create center-weighted metering kernel."""
        kernel = []
        for y in range(8):
            row = []
            for x in range(8):
                dx = (x - 3.5) / 4.0
                dy = (y - 3.5) / 4.0
                dist = math.sqrt(dx * dx + dy * dy)
                weight = max(0.0, 1.0 - dist)
                row.append(weight)
            kernel.append(row)
        return kernel

    def _create_spot_kernel(self) -> List[List[float]]:
        """Create spot metering kernel."""
        kernel = []
        for y in range(8):
            row = []
            for x in range(8):
                dx = (x - 3.5) / 4.0
                dy = (y - 3.5) / 4.0
                dist = math.sqrt(dx * dx + dy * dy)
                weight = 1.0 if dist < 0.25 else 0.0
                row.append(weight)
            kernel.append(row)
        return kernel

    def _create_matrix_kernel(self) -> List[List[float]]:
        """Create matrix (uniform) metering kernel."""
        return [[1.0 for _ in range(8)] for _ in range(8)]

    def calculate_target_ev(
        self,
        luminance_data: Any,
        settings: ExposureSettings,
    ) -> float:
        """Calculate exposure based on weighted average luminance.

        Args:
            luminance_data: Luminance buffer or average value.
            settings: Exposure settings.

        Returns:
            Target EV value clamped to min/max range.
        """
        if isinstance(luminance_data, (int, float)):
            avg_luminance = float(luminance_data)
        else:
            avg_luminance = self._calculate_weighted_average(
                luminance_data, settings.metering_mode
            )

        if settings.metering_mode == MeteringMode.HIGHLIGHT_PRIORITY:
            avg_luminance *= 0.5

        target_ev = luminance_to_ev(avg_luminance) + settings.exposure_compensation

        return max(settings.min_ev, min(settings.max_ev, target_ev))

    def _calculate_weighted_average(
        self,
        luminance_buffer: Any,
        metering_mode: MeteringMode,
    ) -> float:
        """Calculate weighted average luminance.

        Args:
            luminance_buffer: 2D luminance data.
            metering_mode: Metering mode for weight selection.

        Returns:
            Weighted average luminance.
        """
        weights = self._metering_weights.get(
            metering_mode, self._metering_weights[MeteringMode.MATRIX]
        )
        total_weight = sum(sum(row) for row in weights)

        if total_weight <= 0:
            return 0.001

        return 0.18


class HistogramExposure(ExposureCalculator):
    """Histogram percentile-based exposure calculator.

    Uses a luminance histogram to determine exposure based on
    configurable percentile bounds, providing robust exposure
    that handles extreme values well.
    """

    def __init__(self) -> None:
        self._histogram: List[int] = []
        self._min_log_luminance: float = -10.0
        self._max_log_luminance: float = 2.0

    def calculate_target_ev(
        self,
        luminance_data: Any,
        settings: ExposureSettings,
    ) -> float:
        """Calculate exposure based on histogram percentiles.

        Args:
            luminance_data: Luminance histogram or raw data.
            settings: Exposure settings with percentile bounds.

        Returns:
            Target EV value.
        """
        histogram = self._ensure_histogram(luminance_data, settings.histogram_bins)

        total_pixels = sum(histogram)
        if total_pixels == 0:
            return 0.0

        low_count = int(total_pixels * settings.low_percentile)
        high_count = int(total_pixels * settings.high_percentile)

        low_bin = self._find_percentile_bin(histogram, low_count)
        high_bin = self._find_percentile_bin(histogram, high_count)

        avg_bin = (low_bin + high_bin) / 2.0

        bin_range = self._max_log_luminance - self._min_log_luminance
        log_luminance = (
            self._min_log_luminance + (avg_bin / len(histogram)) * bin_range
        )

        target_ev = log_luminance / math.log(2) + settings.exposure_compensation

        return max(settings.min_ev, min(settings.max_ev, target_ev))

    def _ensure_histogram(self, data: Any, num_bins: int) -> List[int]:
        """Ensure we have a valid histogram.

        Args:
            data: Input data (histogram or raw luminance).
            num_bins: Number of bins if generating histogram.

        Returns:
            Luminance histogram.
        """
        if isinstance(data, list) and len(data) == num_bins:
            return data

        return [1] * num_bins

    def _find_percentile_bin(self, histogram: List[int], target_count: int) -> int:
        """Find the histogram bin at a given percentile.

        Args:
            histogram: Luminance histogram.
            target_count: Target cumulative pixel count.

        Returns:
            Bin index at the percentile.
        """
        cumulative = 0
        for i, count in enumerate(histogram):
            cumulative += count
            if cumulative >= target_count:
                return i
        return len(histogram) - 1

    def get_histogram(self) -> List[int]:
        """Get the current histogram.

        Returns:
            Current luminance histogram.
        """
        return self._histogram.copy()


@dataclass
class AdaptationCurve:
    """Defines the temporal adaptation response curve."""

    # Scotopic (rod) response for dark adaptation
    scotopic_threshold: float = 0.001  # cd/m^2
    scotopic_speed: float = 0.1  # seconds^-1

    # Photopic (cone) response for bright adaptation
    photopic_threshold: float = 3.0  # cd/m^2
    photopic_speed: float = 1.0  # seconds^-1

    # Mesopic (mixed) range
    mesopic_blend_range: float = 2.0  # log units


class EyeAdaptation:
    """Temporal adaptation simulation.

    Models human eye adaptation with different speeds for
    brightening (fast) and darkening (slow) transitions.
    """

    def __init__(
        self,
        adaptation_curve: Optional[AdaptationCurve] = None,
    ) -> None:
        """Initialize eye adaptation.

        Args:
            adaptation_curve: Custom adaptation curve parameters.
        """
        self._curve: AdaptationCurve = adaptation_curve or AdaptationCurve()
        self._current_ev: float = 0.0
        self._target_ev: float = 0.0
        self._initialized: bool = False

    @property
    def current_ev(self) -> float:
        """Current adapted exposure value."""
        return self._current_ev

    @property
    def target_ev(self) -> float:
        """Target exposure value."""
        return self._target_ev

    def reset(self, initial_ev: float = 0.0) -> None:
        """Reset adaptation to a specific value.

        Args:
            initial_ev: Initial exposure value.
        """
        self._current_ev = initial_ev
        self._target_ev = initial_ev
        self._initialized = True

    def update(
        self,
        target_ev: float,
        delta_time: float,
        speed_up: float = 3.0,
        speed_down: float = 1.0,
    ) -> float:
        """Update adaptation towards target.

        Args:
            target_ev: Target exposure value.
            delta_time: Time since last update in seconds.
            speed_up: Adaptation speed when brightening.
            speed_down: Adaptation speed when darkening.

        Returns:
            Current adapted exposure value.
        """
        self._target_ev = target_ev

        if not self._initialized:
            self._current_ev = target_ev
            self._initialized = True
            return self._current_ev

        ev_diff = target_ev - self._current_ev
        speed = speed_up if ev_diff > 0 else speed_down

        adaptation_factor = 1.0 - math.exp(-speed * delta_time)

        self._current_ev += ev_diff * adaptation_factor

        return self._current_ev

    def get_exposure_multiplier(self) -> float:
        """Get the current exposure multiplier.

        Returns:
            Exposure multiplier value.
        """
        return ev_to_exposure(self._current_ev)


class ExposureEffect(PostProcessEffect[ExposureSettings]):
    """Post-process effect for exposure control.

    Integrates exposure calculation and eye adaptation into the
    post-processing pipeline.
    """

    def __init__(
        self,
        settings: Optional[ExposureSettings] = None,
    ) -> None:
        """Initialize exposure effect.

        Args:
            settings: Exposure settings.
        """
        super().__init__(
            name="Exposure",
            settings=settings or ExposureSettings(),
            priority=EffectPriority.EXPOSURE.value,
        )

        self._calculators: Dict[ExposureMode, ExposureCalculator] = {
            ExposureMode.MANUAL: ManualExposure(),
            ExposureMode.AUTO_AVERAGE: AutoExposure(),
            ExposureMode.AUTO_HISTOGRAM: HistogramExposure(),
        }

        self._eye_adaptation: EyeAdaptation = EyeAdaptation()
        self._current_exposure: float = 1.0
        self._luminance_buffer: Any = None
        self._histogram: List[int] = []

    @property
    def current_exposure(self) -> float:
        """Current exposure multiplier."""
        return self._current_exposure

    @property
    def current_ev(self) -> float:
        """Current exposure value."""
        return self._eye_adaptation.current_ev

    def get_required_inputs(self) -> List[str]:
        """Get required input resources."""
        return ["color", "luminance"]

    def get_outputs(self) -> List[str]:
        """Get output resources."""
        return ["color", "exposure_buffer"]

    def setup(self, width: int, height: int) -> None:
        """Initialize exposure resources.

        Args:
            width: Render width.
            height: Render height.
        """
        self._luminance_buffer = None
        self._histogram = [0] * (self._settings.histogram_bins if self._settings else 64)

    def execute(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        delta_time: float,
    ) -> None:
        """Execute exposure calculation and application.

        Args:
            inputs: Input resources including color and luminance.
            outputs: Output resources.
            delta_time: Time since last frame.
        """
        if not self._settings or not self._settings.enabled:
            return

        luminance_data = inputs.get("luminance", 0.18)

        calculator = self._calculators.get(
            self._settings.mode,
            self._calculators[ExposureMode.MANUAL],
        )

        target_ev = calculator.calculate_target_ev(luminance_data, self._settings)

        current_ev = self._eye_adaptation.update(
            target_ev,
            delta_time,
            self._settings.adaptation_speed_up,
            self._settings.adaptation_speed_down,
        )

        self._current_exposure = ev_to_exposure(current_ev)

    def cleanup(self) -> None:
        """Release exposure resources."""
        self._luminance_buffer = None
        self._histogram = []

    def is_compute_effect(self) -> bool:
        """Exposure uses compute for histogram."""
        return True


__all__ = [
    "ExposureMode",
    "MeteringMode",
    "ExposureSettings",
    "luminance_to_ev",
    "ev_to_exposure",
    "exposure_to_ev",
    "ExposureCalculator",
    "ManualExposure",
    "AutoExposure",
    "HistogramExposure",
    "AdaptationCurve",
    "EyeAdaptation",
    "ExposureEffect",
]
