"""
Temporal Denoiser with Variance-Guided Accumulation

Implements temporal denoising for ray-traced signals using reprojection,
variance-guided weighting, and neighbourhood clamping for ghost rejection.

Key Features:
- Velocity buffer reprojection for history lookup
- Variance-guided accumulation (weight by variance)
- Exponential moving average (EMA) blend
- Neighbourhood clamping (AABB/YCoCg) for ghosting rejection
- History length tracking (1-64 frames)
- Per-signal configuration (GI, reflections, shadows)

References:
- "Temporally Reliable Motion Vectors for Real-time Ray Tracing" NVIDIA
- "SVGF: Spatiotemporal Variance-Guided Filtering" Schied et al., HPG 2017
- "Ray Tracing Gems II", Chapter 25: Temporal Antialiasing & Denoising
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from engine.platform.rhi.device import Device
    from engine.platform.rhi.resources import Texture


# =============================================================================
# Constants
# =============================================================================


# Numerical safety
EPSILON = 1e-6
VARIANCE_EPSILON = 1e-4

# History length limits
MIN_HISTORY_LENGTH = 1
MAX_HISTORY_LENGTH = 64
DEFAULT_CONVERGENCE_FRAMES = 16

# Variance clip default gamma
DEFAULT_VARIANCE_GAMMA = 1.25

# Default EMA alpha values
DEFAULT_EMA_ALPHA = 0.1  # Blends 10% current, 90% history
FAST_EMA_ALPHA = 0.2     # More responsive
SLOW_EMA_ALPHA = 0.05    # More stable


# =============================================================================
# Enumerations
# =============================================================================


class TemporalQuality(IntEnum):
    """Quality presets for temporal denoising.

    Each level represents a different trade-off between
    responsiveness and stability.
    """

    LOW = 1       # Fast, aggressive, may flicker
    MEDIUM = 2    # Balanced
    HIGH = 3      # High quality, slower convergence
    ULTRA = 4     # Maximum stability, longest convergence


class TemporalTarget(Enum):
    """Target signal type for temporal denoising.

    Different signal types require different accumulation strategies.
    """

    GI = auto()           # Global illumination (diffuse indirect)
    REFLECTIONS = auto()  # Specular reflections
    SHADOWS = auto()      # Shadow visibility
    AO = auto()           # Ambient occlusion
    COMBINED = auto()     # Combined signal (GI + reflections)
    CUSTOM = auto()       # User-defined parameters


class ClampingMode(IntEnum):
    """Neighbourhood clamping strategies for ghost rejection.

    NONE: No clamping (pure accumulation, ghosts visible).
    AABB: Axis-aligned bounding box clamp (fast, decent quality).
    VARIANCE: Variance-based clamp (better quality, slower).
    YCOCG_AABB: AABB in YCoCg space (perceptually better).
    YCOCG_VARIANCE: Variance clamp in YCoCg space (best quality).
    """

    NONE = 0
    AABB = 1
    VARIANCE = 2
    YCOCG_AABB = 3
    YCOCG_VARIANCE = 4


class DisocclusionMode(IntEnum):
    """Disocclusion detection strategies.

    DEPTH_ONLY: Reject based on depth discontinuity only.
    NORMAL_ONLY: Reject based on surface normal change.
    VELOCITY: Reject based on velocity magnitude.
    COMBINED: Use all criteria (recommended).
    ADAPTIVE: Weight criteria by scene content.
    """

    DEPTH_ONLY = 0
    NORMAL_ONLY = 1
    VELOCITY = 2
    COMBINED = 3
    ADAPTIVE = 4


# =============================================================================
# Quality Preset Parameters
# =============================================================================


@dataclass(frozen=True)
class QualityPreset:
    """Parameters for a quality preset.

    Attributes:
        ema_alpha: Exponential moving average blend factor.
        target_frames: Target frames for convergence.
        variance_gamma: Variance clipping aggressiveness.
        neighbourhood_size: Size of clamping neighbourhood (3, 5, 7).
        min_history_weight: Minimum history weight after rejection.
    """

    ema_alpha: float
    target_frames: int
    variance_gamma: float
    neighbourhood_size: int
    min_history_weight: float


QUALITY_PRESETS: Dict[TemporalQuality, QualityPreset] = {
    TemporalQuality.LOW: QualityPreset(
        ema_alpha=0.2,
        target_frames=8,
        variance_gamma=1.5,
        neighbourhood_size=3,
        min_history_weight=0.7,
    ),
    TemporalQuality.MEDIUM: QualityPreset(
        ema_alpha=0.12,
        target_frames=12,
        variance_gamma=1.25,
        neighbourhood_size=3,
        min_history_weight=0.8,
    ),
    TemporalQuality.HIGH: QualityPreset(
        ema_alpha=0.08,
        target_frames=16,
        variance_gamma=1.0,
        neighbourhood_size=5,
        min_history_weight=0.85,
    ),
    TemporalQuality.ULTRA: QualityPreset(
        ema_alpha=0.05,
        target_frames=24,
        variance_gamma=0.75,
        neighbourhood_size=5,
        min_history_weight=0.9,
    ),
}


# =============================================================================
# Color Space Conversion (YCoCg)
# =============================================================================


class YCoCgConverter:
    """RGB to YCoCg color space conversion for temporal filtering.

    YCoCg provides perceptually better clamping than RGB
    by separating luminance from chrominance.
    """

    @staticmethod
    def rgb_to_ycocg(r: float, g: float, b: float) -> Tuple[float, float, float]:
        """Convert RGB to YCoCg.

        Args:
            r: Red channel [0, 1].
            g: Green channel [0, 1].
            b: Blue channel [0, 1].

        Returns:
            Tuple of (Y, Co, Cg) values.
        """
        y = 0.25 * r + 0.5 * g + 0.25 * b
        co = 0.5 * r - 0.5 * b
        cg = -0.25 * r + 0.5 * g - 0.25 * b
        return (y, co, cg)

    @staticmethod
    def ycocg_to_rgb(y: float, co: float, cg: float) -> Tuple[float, float, float]:
        """Convert YCoCg to RGB.

        Args:
            y: Luminance.
            co: Orange chrominance.
            cg: Green chrominance.

        Returns:
            Tuple of (R, G, B) values.
        """
        r = y + co - cg
        g = y + cg
        b = y - co - cg
        return (r, g, b)

    @staticmethod
    def luminance(r: float, g: float, b: float) -> float:
        """Extract luminance from RGB.

        Args:
            r: Red channel [0, 1].
            g: Green channel [0, 1].
            b: Blue channel [0, 1].

        Returns:
            Luminance value (Y channel of YCoCg).
        """
        return 0.25 * r + 0.5 * g + 0.25 * b


# =============================================================================
# Variance Estimation
# =============================================================================


@dataclass
class VarianceEstimate:
    """Local variance estimate for a pixel region.

    Attributes:
        mean: Mean color value (RGB or YCoCg).
        variance: Variance of color values.
        min_value: Minimum color value in neighbourhood.
        max_value: Maximum color value in neighbourhood.
        sample_count: Number of samples used for estimation.
        luminance_variance: Variance of luminance specifically.
    """

    mean: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    variance: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    min_value: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_value: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    sample_count: int = 0
    luminance_variance: float = 0.0

    def is_valid(self) -> bool:
        """Check if estimate has valid data.

        Returns:
            True if sample_count > 0.
        """
        return self.sample_count > 0

    def get_variance_magnitude(self) -> float:
        """Get overall variance magnitude.

        Returns:
            Sum of variance components.
        """
        return sum(self.variance)

    def get_std_dev(self) -> Tuple[float, float, float]:
        """Get standard deviation.

        Returns:
            Square root of variance for each channel.
        """
        return (
            math.sqrt(max(0.0, self.variance[0])),
            math.sqrt(max(0.0, self.variance[1])),
            math.sqrt(max(0.0, self.variance[2])),
        )


class VarianceGuided:
    """Variance-guided accumulation weight calculator.

    Computes adaptive blend weights based on local variance.
    High variance regions use more current frame data (faster response).
    Low variance regions use more history (more stable).

    References:
        SVGF: Spatiotemporal Variance-Guided Filtering
    """

    def __init__(
        self,
        base_weight: float = 0.9,
        variance_scale: float = 4.0,
        min_weight: float = 0.5,
        max_weight: float = 0.98,
    ) -> None:
        """Initialize variance-guided calculator.

        Args:
            base_weight: Base history weight when variance is zero.
            variance_scale: Scale factor for variance sensitivity.
            min_weight: Minimum allowed history weight.
            max_weight: Maximum allowed history weight.

        Raises:
            ValueError: If parameters are out of valid range.
        """
        if not (0.0 <= base_weight <= 1.0):
            raise ValueError(f"base_weight must be in [0, 1], got {base_weight}")
        if variance_scale <= 0.0:
            raise ValueError(f"variance_scale must be positive, got {variance_scale}")
        if not (0.0 <= min_weight <= max_weight <= 1.0):
            raise ValueError(
                f"Invalid weight range: [{min_weight}, {max_weight}]"
            )

        self._base_weight = base_weight
        self._variance_scale = variance_scale
        self._min_weight = min_weight
        self._max_weight = max_weight

    @property
    def base_weight(self) -> float:
        """Get base history weight."""
        return self._base_weight

    @base_weight.setter
    def base_weight(self, value: float) -> None:
        """Set base history weight."""
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"base_weight must be in [0, 1], got {value}")
        self._base_weight = value

    @property
    def variance_scale(self) -> float:
        """Get variance sensitivity scale."""
        return self._variance_scale

    @variance_scale.setter
    def variance_scale(self, value: float) -> None:
        """Set variance sensitivity scale."""
        if value <= 0.0:
            raise ValueError(f"variance_scale must be positive, got {value}")
        self._variance_scale = value

    @property
    def min_weight(self) -> float:
        """Get minimum history weight."""
        return self._min_weight

    @property
    def max_weight(self) -> float:
        """Get maximum history weight."""
        return self._max_weight

    def calculate_weight(
        self,
        variance: float,
        history_length: int = 1,
    ) -> float:
        """Calculate variance-guided history weight.

        High variance -> lower history weight (more responsive)
        Low variance -> higher history weight (more stable)

        Args:
            variance: Local luminance variance.
            history_length: Number of accumulated frames.

        Returns:
            History weight [min_weight, max_weight].
        """
        # Normalize variance
        normalized_var = variance * self._variance_scale

        # Compute weight reduction based on variance
        # Higher variance reduces history weight
        reduction = normalized_var / (1.0 + normalized_var)

        # Apply history length factor (longer history = more stable)
        history_factor = min(1.0, history_length / 16.0)

        # Final weight
        weight = self._base_weight * (1.0 - reduction * (1.0 - history_factor))

        return max(self._min_weight, min(self._max_weight, weight))

    def calculate_weight_rgb(
        self,
        variance: Tuple[float, float, float],
        history_length: int = 1,
    ) -> float:
        """Calculate weight from RGB variance.

        Args:
            variance: Variance per RGB channel.
            history_length: Number of accumulated frames.

        Returns:
            History weight [min_weight, max_weight].
        """
        # Combine RGB variance (luminance-weighted)
        lum_var = 0.25 * variance[0] + 0.5 * variance[1] + 0.25 * variance[2]
        return self.calculate_weight(lum_var, history_length)

    def get_shader_params(self) -> Dict[str, float]:
        """Get parameters for shader binding.

        Returns:
            Dictionary of variance-guided parameters.
        """
        return {
            "base_weight": self._base_weight,
            "variance_scale": self._variance_scale,
            "min_weight": self._min_weight,
            "max_weight": self._max_weight,
        }


# =============================================================================
# History Tracker
# =============================================================================


@dataclass
class HistoryEntry:
    """Single history entry for a pixel.

    Attributes:
        frame_count: Number of frames accumulated.
        last_valid_frame: Frame index of last valid sample.
        accumulated_variance: Running variance estimate.
        confidence: Confidence in history [0, 1].
    """

    frame_count: int = 0
    last_valid_frame: int = -1
    accumulated_variance: float = 0.0
    confidence: float = 0.0

    def is_valid(self) -> bool:
        """Check if history entry is valid.

        Returns:
            True if frame_count > 0.
        """
        return self.frame_count > 0

    def reset(self) -> None:
        """Reset history entry to initial state."""
        self.frame_count = 0
        self.last_valid_frame = -1
        self.accumulated_variance = 0.0
        self.confidence = 0.0


class HistoryTracker:
    """Tracks accumulation history for temporal denoising.

    Manages per-pixel history length and confidence, enabling
    adaptive blending based on how many valid frames have accumulated.

    History length is tracked in the range [1, max_history_length].
    Longer history = more stable result but slower adaptation.

    Attributes:
        max_history_length: Maximum frames to track (1-64).
        target_convergence: Target frames for "converged" state.
        current_frame: Current frame index.
    """

    def __init__(
        self,
        max_history_length: int = MAX_HISTORY_LENGTH,
        target_convergence: int = DEFAULT_CONVERGENCE_FRAMES,
    ) -> None:
        """Initialize history tracker.

        Args:
            max_history_length: Maximum history length (1-64).
            target_convergence: Target frames for convergence.

        Raises:
            ValueError: If parameters are out of valid range.
        """
        if not (MIN_HISTORY_LENGTH <= max_history_length <= MAX_HISTORY_LENGTH):
            raise ValueError(
                f"max_history_length must be in [{MIN_HISTORY_LENGTH}, {MAX_HISTORY_LENGTH}], "
                f"got {max_history_length}"
            )
        if target_convergence < 1:
            raise ValueError(
                f"target_convergence must be >= 1, got {target_convergence}"
            )

        self._max_history_length = max_history_length
        self._target_convergence = min(target_convergence, max_history_length)
        self._current_frame = 0
        self._history: Dict[Tuple[int, int], HistoryEntry] = {}
        self._global_frame_count = 0

    @property
    def max_history_length(self) -> int:
        """Get maximum history length."""
        return self._max_history_length

    @max_history_length.setter
    def max_history_length(self, value: int) -> None:
        """Set maximum history length."""
        if not (MIN_HISTORY_LENGTH <= value <= MAX_HISTORY_LENGTH):
            raise ValueError(
                f"max_history_length must be in [{MIN_HISTORY_LENGTH}, {MAX_HISTORY_LENGTH}], "
                f"got {value}"
            )
        self._max_history_length = value

    @property
    def target_convergence(self) -> int:
        """Get target convergence frame count."""
        return self._target_convergence

    @target_convergence.setter
    def target_convergence(self, value: int) -> None:
        """Set target convergence frame count."""
        if value < 1:
            raise ValueError(f"target_convergence must be >= 1, got {value}")
        self._target_convergence = min(value, self._max_history_length)

    @property
    def current_frame(self) -> int:
        """Get current frame index."""
        return self._current_frame

    @property
    def global_frame_count(self) -> int:
        """Get total frames processed."""
        return self._global_frame_count

    def advance_frame(self) -> None:
        """Advance to next frame."""
        self._current_frame += 1
        self._global_frame_count += 1

    def reset(self) -> None:
        """Reset all history tracking."""
        self._history.clear()
        self._current_frame = 0
        # Keep global_frame_count for statistics

    def invalidate(self) -> None:
        """Invalidate all history (e.g., on camera cut)."""
        self._history.clear()

    def get_entry(self, x: int, y: int) -> HistoryEntry:
        """Get history entry for a pixel.

        Args:
            x: Pixel x coordinate.
            y: Pixel y coordinate.

        Returns:
            HistoryEntry for the pixel (created if needed).
        """
        key = (x, y)
        if key not in self._history:
            self._history[key] = HistoryEntry()
        return self._history[key]

    def update_entry(
        self,
        x: int,
        y: int,
        valid: bool,
        variance: float = 0.0,
    ) -> HistoryEntry:
        """Update history entry for a pixel.

        Args:
            x: Pixel x coordinate.
            y: Pixel y coordinate.
            valid: Whether the sample was valid (not rejected).
            variance: Local variance at this pixel.

        Returns:
            Updated HistoryEntry.
        """
        entry = self.get_entry(x, y)

        if valid:
            # Increment frame count up to max
            entry.frame_count = min(
                entry.frame_count + 1, self._max_history_length
            )
            entry.last_valid_frame = self._current_frame

            # Update running variance (exponential moving average)
            alpha = 1.0 / min(entry.frame_count, 16)
            entry.accumulated_variance = (
                entry.accumulated_variance * (1.0 - alpha) + variance * alpha
            )

            # Update confidence
            entry.confidence = min(
                1.0, entry.frame_count / self._target_convergence
            )
        else:
            # Reset on rejection
            entry.frame_count = 1
            entry.accumulated_variance = variance
            entry.confidence = 1.0 / self._target_convergence

        return entry

    def get_blend_weight(self, x: int, y: int) -> float:
        """Get temporal blend weight based on history.

        Longer history = higher history weight (more stable).

        Args:
            x: Pixel x coordinate.
            y: Pixel y coordinate.

        Returns:
            History blend weight [0, 1].
        """
        entry = self.get_entry(x, y)
        if entry.frame_count == 0:
            return 0.0

        # Weight increases with history length
        # At target_convergence frames, weight is at maximum
        return min(1.0, entry.frame_count / self._target_convergence)

    def is_converged(self, x: int, y: int) -> bool:
        """Check if pixel has converged.

        Args:
            x: Pixel x coordinate.
            y: Pixel y coordinate.

        Returns:
            True if frame_count >= target_convergence.
        """
        entry = self.get_entry(x, y)
        return entry.frame_count >= self._target_convergence

    def get_convergence_progress(self) -> float:
        """Get global convergence progress.

        Returns:
            Progress [0, 1] based on global frame count.
        """
        return min(1.0, self._global_frame_count / self._target_convergence)

    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics.

        Returns:
            Dictionary of statistics.
        """
        if not self._history:
            return {
                "total_pixels": 0,
                "avg_history_length": 0.0,
                "converged_pixels": 0,
                "convergence_rate": 0.0,
            }

        total = len(self._history)
        total_length = sum(e.frame_count for e in self._history.values())
        converged = sum(
            1 for e in self._history.values()
            if e.frame_count >= self._target_convergence
        )

        return {
            "total_pixels": total,
            "avg_history_length": total_length / total if total > 0 else 0.0,
            "converged_pixels": converged,
            "convergence_rate": converged / total if total > 0 else 0.0,
        }


# =============================================================================
# Neighbourhood Clamping
# =============================================================================


class NeighbourhoodClamper:
    """Neighbourhood clamping for ghost rejection.

    Clamps history sample to the color range of the current frame's
    neighbourhood, preventing ghosting artifacts from invalid history.

    Supports multiple clamping strategies:
    - AABB: Simple min/max bounding box
    - Variance: Mean +/- gamma * stddev
    - YCoCg variants for perceptual quality
    """

    def __init__(
        self,
        mode: ClampingMode = ClampingMode.YCOCG_VARIANCE,
        gamma: float = DEFAULT_VARIANCE_GAMMA,
        neighbourhood_size: int = 3,
    ) -> None:
        """Initialize neighbourhood clamper.

        Args:
            mode: Clamping strategy.
            gamma: Variance clip gamma (stddev multiplier).
            neighbourhood_size: Size of neighbourhood (3, 5, or 7).

        Raises:
            ValueError: If parameters are invalid.
        """
        if neighbourhood_size not in (3, 5, 7):
            raise ValueError(
                f"neighbourhood_size must be 3, 5, or 7, got {neighbourhood_size}"
            )
        if gamma <= 0.0:
            raise ValueError(f"gamma must be positive, got {gamma}")

        self._mode = mode
        self._gamma = gamma
        self._neighbourhood_size = neighbourhood_size
        self._converter = YCoCgConverter()

    @property
    def mode(self) -> ClampingMode:
        """Get clamping mode."""
        return self._mode

    @mode.setter
    def mode(self, value: ClampingMode) -> None:
        """Set clamping mode."""
        self._mode = value

    @property
    def gamma(self) -> float:
        """Get variance gamma."""
        return self._gamma

    @gamma.setter
    def gamma(self, value: float) -> None:
        """Set variance gamma."""
        if value <= 0.0:
            raise ValueError(f"gamma must be positive, got {value}")
        self._gamma = value

    @property
    def neighbourhood_size(self) -> int:
        """Get neighbourhood size."""
        return self._neighbourhood_size

    @neighbourhood_size.setter
    def neighbourhood_size(self, value: int) -> None:
        """Set neighbourhood size."""
        if value not in (3, 5, 7):
            raise ValueError(
                f"neighbourhood_size must be 3, 5, or 7, got {value}"
            )
        self._neighbourhood_size = value

    def clamp_aabb(
        self,
        history: Tuple[float, float, float],
        min_color: Tuple[float, float, float],
        max_color: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """Clamp history to AABB bounds.

        Args:
            history: History color to clamp.
            min_color: Minimum color bounds.
            max_color: Maximum color bounds.

        Returns:
            Clamped color.
        """
        return (
            max(min_color[0], min(max_color[0], history[0])),
            max(min_color[1], min(max_color[1], history[1])),
            max(min_color[2], min(max_color[2], history[2])),
        )

    def clamp_variance(
        self,
        history: Tuple[float, float, float],
        mean: Tuple[float, float, float],
        std_dev: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """Clamp history using variance bounds (mean +/- gamma * stddev).

        Args:
            history: History color to clamp.
            mean: Mean color of neighbourhood.
            std_dev: Standard deviation of neighbourhood.

        Returns:
            Clamped color.
        """
        gamma = self._gamma

        min_bound = (
            mean[0] - gamma * std_dev[0],
            mean[1] - gamma * std_dev[1],
            mean[2] - gamma * std_dev[2],
        )
        max_bound = (
            mean[0] + gamma * std_dev[0],
            mean[1] + gamma * std_dev[1],
            mean[2] + gamma * std_dev[2],
        )

        return self.clamp_aabb(history, min_bound, max_bound)

    def clamp(
        self,
        history: Tuple[float, float, float],
        variance_estimate: VarianceEstimate,
    ) -> Tuple[float, float, float]:
        """Clamp history using configured mode.

        Args:
            history: History color to clamp (RGB).
            variance_estimate: Local variance estimate.

        Returns:
            Clamped color (RGB).
        """
        if self._mode == ClampingMode.NONE:
            return history

        if self._mode == ClampingMode.AABB:
            return self.clamp_aabb(
                history,
                variance_estimate.min_value,
                variance_estimate.max_value,
            )

        if self._mode == ClampingMode.VARIANCE:
            std_dev = variance_estimate.get_std_dev()
            return self.clamp_variance(
                history,
                variance_estimate.mean,
                std_dev,
            )

        # YCoCg modes
        history_ycocg = self._converter.rgb_to_ycocg(*history)
        mean_ycocg = self._converter.rgb_to_ycocg(*variance_estimate.mean)

        if self._mode == ClampingMode.YCOCG_AABB:
            min_ycocg = self._converter.rgb_to_ycocg(*variance_estimate.min_value)
            max_ycocg = self._converter.rgb_to_ycocg(*variance_estimate.max_value)
            clamped_ycocg = self.clamp_aabb(history_ycocg, min_ycocg, max_ycocg)
        else:  # YCOCG_VARIANCE
            # Compute std_dev in YCoCg space
            var_ycocg = self._converter.rgb_to_ycocg(*variance_estimate.variance)
            std_ycocg = (
                math.sqrt(max(0.0, var_ycocg[0])),
                math.sqrt(max(0.0, var_ycocg[1])),
                math.sqrt(max(0.0, var_ycocg[2])),
            )
            clamped_ycocg = self.clamp_variance(history_ycocg, mean_ycocg, std_ycocg)

        return self._converter.ycocg_to_rgb(*clamped_ycocg)

    def compute_clamp_amount(
        self,
        original: Tuple[float, float, float],
        clamped: Tuple[float, float, float],
    ) -> float:
        """Compute how much clamping was applied.

        Args:
            original: Original history color.
            clamped: Clamped history color.

        Returns:
            Clamping amount [0, 1] where 0 = no change, 1 = max change.
        """
        diff = (
            abs(original[0] - clamped[0]),
            abs(original[1] - clamped[1]),
            abs(original[2] - clamped[2]),
        )
        mag = math.sqrt(diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2)

        # Normalize by typical color range
        return min(1.0, mag / math.sqrt(3.0))


# =============================================================================
# Reprojection
# =============================================================================


@dataclass
class ReprojectionResult:
    """Result of reprojecting a pixel to history.

    Attributes:
        uv: Reprojected UV coordinates.
        valid: Whether reprojection is valid (inside frame).
        confidence: Confidence in reprojection [0, 1].
        velocity_magnitude: Magnitude of motion.
        is_disoccluded: Whether pixel was disoccluded.
    """

    uv: Tuple[float, float] = (0.0, 0.0)
    valid: bool = False
    confidence: float = 0.0
    velocity_magnitude: float = 0.0
    is_disoccluded: bool = False


class Reprojector:
    """Velocity buffer reprojection for temporal denoising.

    Uses motion vectors to find corresponding history samples.
    Detects disocclusion via depth/normal/velocity tests.
    """

    def __init__(
        self,
        disocclusion_mode: DisocclusionMode = DisocclusionMode.COMBINED,
        depth_threshold: float = 0.01,
        normal_threshold: float = 0.9,
        velocity_threshold: float = 0.05,
    ) -> None:
        """Initialize reprojector.

        Args:
            disocclusion_mode: Disocclusion detection strategy.
            depth_threshold: Relative depth threshold for rejection.
            normal_threshold: Normal dot product threshold.
            velocity_threshold: Velocity magnitude threshold.

        Raises:
            ValueError: If thresholds are invalid.
        """
        if depth_threshold <= 0.0:
            raise ValueError(f"depth_threshold must be positive, got {depth_threshold}")
        if not (0.0 <= normal_threshold <= 1.0):
            raise ValueError(
                f"normal_threshold must be in [0, 1], got {normal_threshold}"
            )
        if velocity_threshold < 0.0:
            raise ValueError(
                f"velocity_threshold must be non-negative, got {velocity_threshold}"
            )

        self._disocclusion_mode = disocclusion_mode
        self._depth_threshold = depth_threshold
        self._normal_threshold = normal_threshold
        self._velocity_threshold = velocity_threshold

    @property
    def disocclusion_mode(self) -> DisocclusionMode:
        """Get disocclusion detection mode."""
        return self._disocclusion_mode

    @disocclusion_mode.setter
    def disocclusion_mode(self, value: DisocclusionMode) -> None:
        """Set disocclusion detection mode."""
        self._disocclusion_mode = value

    @property
    def depth_threshold(self) -> float:
        """Get depth rejection threshold."""
        return self._depth_threshold

    @depth_threshold.setter
    def depth_threshold(self, value: float) -> None:
        """Set depth rejection threshold."""
        if value <= 0.0:
            raise ValueError(f"depth_threshold must be positive, got {value}")
        self._depth_threshold = value

    @property
    def normal_threshold(self) -> float:
        """Get normal rejection threshold."""
        return self._normal_threshold

    @normal_threshold.setter
    def normal_threshold(self, value: float) -> None:
        """Set normal rejection threshold."""
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"normal_threshold must be in [0, 1], got {value}")
        self._normal_threshold = value

    @property
    def velocity_threshold(self) -> float:
        """Get velocity rejection threshold."""
        return self._velocity_threshold

    @velocity_threshold.setter
    def velocity_threshold(self, value: float) -> None:
        """Set velocity rejection threshold."""
        if value < 0.0:
            raise ValueError(
                f"velocity_threshold must be non-negative, got {value}"
            )
        self._velocity_threshold = value

    def reproject(
        self,
        uv: Tuple[float, float],
        velocity: Tuple[float, float],
    ) -> ReprojectionResult:
        """Compute reprojected UV coordinates.

        Args:
            uv: Current frame UV coordinates [0, 1].
            velocity: Screen-space motion vector.

        Returns:
            ReprojectionResult with reprojected UV and validity.
        """
        # Compute reprojected UV
        reprojected_uv = (
            uv[0] - velocity[0],
            uv[1] - velocity[1],
        )

        # Check if inside frame
        valid = (
            0.0 <= reprojected_uv[0] <= 1.0
            and 0.0 <= reprojected_uv[1] <= 1.0
        )

        # Compute velocity magnitude
        vel_mag = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)

        # Confidence decreases with velocity
        confidence = 1.0 / (1.0 + vel_mag * 10.0) if valid else 0.0

        return ReprojectionResult(
            uv=reprojected_uv,
            valid=valid,
            confidence=confidence,
            velocity_magnitude=vel_mag,
            is_disoccluded=False,  # Set by check_disocclusion
        )

    def check_depth_disocclusion(
        self,
        current_depth: float,
        history_depth: float,
    ) -> float:
        """Check for depth-based disocclusion.

        Args:
            current_depth: Current frame linear depth.
            history_depth: History sample depth.

        Returns:
            Validity weight [0, 1] where 1 = valid.
        """
        if current_depth <= 0.0 or history_depth <= 0.0:
            return 0.0

        # Relative depth difference
        diff = abs(current_depth - history_depth) / max(current_depth, EPSILON)

        if diff > self._depth_threshold:
            return 0.0

        return 1.0 - (diff / self._depth_threshold)

    def check_normal_disocclusion(
        self,
        current_normal: Tuple[float, float, float],
        history_normal: Tuple[float, float, float],
    ) -> float:
        """Check for normal-based disocclusion.

        Args:
            current_normal: Current frame surface normal.
            history_normal: History sample normal.

        Returns:
            Validity weight [0, 1] where 1 = valid.
        """
        # Dot product
        dot = (
            current_normal[0] * history_normal[0]
            + current_normal[1] * history_normal[1]
            + current_normal[2] * history_normal[2]
        )

        if dot < self._normal_threshold:
            return 0.0

        # Remap to [0, 1]
        return (dot - self._normal_threshold) / (1.0 - self._normal_threshold + EPSILON)

    def check_velocity_disocclusion(
        self,
        velocity: Tuple[float, float],
    ) -> float:
        """Check for velocity-based disocclusion.

        High velocity suggests fast motion where ghosting is likely.

        Args:
            velocity: Screen-space motion vector.

        Returns:
            Validity weight [0, 1] where 1 = valid.
        """
        magnitude = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)

        if magnitude > self._velocity_threshold:
            excess = magnitude - self._velocity_threshold
            return max(0.0, 1.0 - excess * 2.0)

        return 1.0

    def check_disocclusion(
        self,
        current_depth: float,
        history_depth: float,
        current_normal: Tuple[float, float, float],
        history_normal: Tuple[float, float, float],
        velocity: Tuple[float, float],
    ) -> float:
        """Check for disocclusion using configured mode.

        Args:
            current_depth: Current frame linear depth.
            history_depth: History sample depth.
            current_normal: Current frame surface normal.
            history_normal: History sample normal.
            velocity: Screen-space motion vector.

        Returns:
            Combined validity weight [0, 1] where 1 = valid.
        """
        mode = self._disocclusion_mode

        if mode == DisocclusionMode.DEPTH_ONLY:
            return self.check_depth_disocclusion(current_depth, history_depth)

        if mode == DisocclusionMode.NORMAL_ONLY:
            return self.check_normal_disocclusion(current_normal, history_normal)

        if mode == DisocclusionMode.VELOCITY:
            return self.check_velocity_disocclusion(velocity)

        # COMBINED or ADAPTIVE: multiply all weights
        depth_weight = self.check_depth_disocclusion(current_depth, history_depth)
        normal_weight = self.check_normal_disocclusion(current_normal, history_normal)
        velocity_weight = self.check_velocity_disocclusion(velocity)

        if mode == DisocclusionMode.ADAPTIVE:
            # Weight by velocity magnitude (more motion = more strict)
            vel_mag = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
            motion_factor = min(1.0, vel_mag * 5.0)

            # Higher motion -> stricter depth/normal tests
            depth_weight = depth_weight ** (1.0 + motion_factor)
            normal_weight = normal_weight ** (1.0 + motion_factor)

        return depth_weight * normal_weight * velocity_weight


# =============================================================================
# Exponential Moving Average Blend
# =============================================================================


class EMABlender:
    """Exponential Moving Average blender for temporal accumulation.

    Blends current frame with history using:
        result = current * alpha + history * (1 - alpha)

    Alpha can be modulated by variance, motion, and history length.
    """

    def __init__(
        self,
        alpha: float = DEFAULT_EMA_ALPHA,
        variance_modulation: bool = True,
        motion_modulation: bool = True,
    ) -> None:
        """Initialize EMA blender.

        Args:
            alpha: Base blend factor (weight for current frame).
            variance_modulation: Modulate alpha by local variance.
            motion_modulation: Modulate alpha by motion magnitude.

        Raises:
            ValueError: If alpha is out of range.
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self._alpha = alpha
        self._variance_modulation = variance_modulation
        self._motion_modulation = motion_modulation

    @property
    def alpha(self) -> float:
        """Get base blend alpha."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        """Set base blend alpha."""
        if not (0.0 < value < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {value}")
        self._alpha = value

    @property
    def variance_modulation(self) -> bool:
        """Get variance modulation state."""
        return self._variance_modulation

    @variance_modulation.setter
    def variance_modulation(self, value: bool) -> None:
        """Set variance modulation state."""
        self._variance_modulation = value

    @property
    def motion_modulation(self) -> bool:
        """Get motion modulation state."""
        return self._motion_modulation

    @motion_modulation.setter
    def motion_modulation(self, value: bool) -> None:
        """Set motion modulation state."""
        self._motion_modulation = value

    def compute_alpha(
        self,
        variance: float = 0.0,
        velocity_magnitude: float = 0.0,
        history_length: int = 1,
    ) -> float:
        """Compute modulated blend alpha.

        Higher variance or motion -> higher alpha (more current frame).
        Longer history -> lower alpha (more history).

        Args:
            variance: Local luminance variance.
            velocity_magnitude: Motion vector magnitude.
            history_length: Number of accumulated frames.

        Returns:
            Modulated alpha [0.01, 0.99].
        """
        alpha = self._alpha

        if self._variance_modulation and variance > 0.0:
            # High variance -> increase alpha (faster response)
            var_factor = 1.0 + variance * 4.0
            alpha = alpha * var_factor

        if self._motion_modulation and velocity_magnitude > 0.0:
            # High motion -> increase alpha (reduce ghosting)
            motion_factor = 1.0 + velocity_magnitude * 10.0
            alpha = alpha * motion_factor

        # History length decreases alpha (longer history = more stable)
        if history_length > 1:
            history_factor = 1.0 / math.sqrt(history_length)
            alpha = alpha * history_factor

        # Clamp to valid range
        return max(0.01, min(0.99, alpha))

    def blend(
        self,
        current: Tuple[float, float, float],
        history: Tuple[float, float, float],
        alpha: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """Blend current and history colors.

        Args:
            current: Current frame color.
            history: History color.
            alpha: Optional override alpha (uses base if None).

        Returns:
            Blended color.
        """
        a = alpha if alpha is not None else self._alpha
        inv_a = 1.0 - a

        return (
            current[0] * a + history[0] * inv_a,
            current[1] * a + history[1] * inv_a,
            current[2] * a + history[2] * inv_a,
        )

    def blend_with_variance(
        self,
        current: Tuple[float, float, float],
        history: Tuple[float, float, float],
        variance: float,
        velocity_magnitude: float = 0.0,
        history_length: int = 1,
    ) -> Tuple[Tuple[float, float, float], float]:
        """Blend with variance-modulated alpha.

        Args:
            current: Current frame color.
            history: History color (after clamping).
            variance: Local luminance variance.
            velocity_magnitude: Motion vector magnitude.
            history_length: Number of accumulated frames.

        Returns:
            Tuple of (blended color, alpha used).
        """
        alpha = self.compute_alpha(variance, velocity_magnitude, history_length)
        blended = self.blend(current, history, alpha)
        return (blended, alpha)


# =============================================================================
# Temporal Denoiser Configuration
# =============================================================================


@dataclass
class TemporalDenoiseConfig:
    """Full configuration for temporal denoising.

    Attributes:
        quality: Quality preset.
        target: Signal type being denoised.
        clamping_mode: Neighbourhood clamping strategy.
        disocclusion_mode: Disocclusion detection strategy.
        variance_gamma: Variance clip gamma.
        neighbourhood_size: Clamping neighbourhood size.
        ema_alpha: EMA blend alpha (None = use preset).
        target_convergence: Target convergence frames (None = use preset).
        max_history: Maximum history length.
        depth_threshold: Depth disocclusion threshold.
        normal_threshold: Normal disocclusion threshold.
        velocity_threshold: Velocity disocclusion threshold.
        use_variance_guided: Enable variance-guided weighting.
        use_motion_modulation: Modulate blend by motion.
        anti_flicker: Enable anti-flicker filter.
    """

    quality: TemporalQuality = TemporalQuality.HIGH
    target: TemporalTarget = TemporalTarget.GI

    # Clamping
    clamping_mode: ClampingMode = ClampingMode.YCOCG_VARIANCE
    disocclusion_mode: DisocclusionMode = DisocclusionMode.COMBINED

    # Variance
    variance_gamma: Optional[float] = None
    neighbourhood_size: int = 5

    # EMA
    ema_alpha: Optional[float] = None
    target_convergence: Optional[int] = None
    max_history: int = MAX_HISTORY_LENGTH

    # Disocclusion thresholds
    depth_threshold: float = 0.01
    normal_threshold: float = 0.9
    velocity_threshold: float = 0.05

    # Modulation
    use_variance_guided: bool = True
    use_motion_modulation: bool = True
    anti_flicker: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.neighbourhood_size not in (3, 5, 7):
            raise ValueError(
                f"neighbourhood_size must be 3, 5, or 7, got {self.neighbourhood_size}"
            )
        if not (MIN_HISTORY_LENGTH <= self.max_history <= MAX_HISTORY_LENGTH):
            raise ValueError(
                f"max_history must be in [{MIN_HISTORY_LENGTH}, {MAX_HISTORY_LENGTH}], "
                f"got {self.max_history}"
            )
        if self.depth_threshold <= 0.0:
            raise ValueError(
                f"depth_threshold must be positive, got {self.depth_threshold}"
            )
        if not (0.0 <= self.normal_threshold <= 1.0):
            raise ValueError(
                f"normal_threshold must be in [0, 1], got {self.normal_threshold}"
            )
        if self.velocity_threshold < 0.0:
            raise ValueError(
                f"velocity_threshold must be non-negative, got {self.velocity_threshold}"
            )
        if self.ema_alpha is not None and not (0.0 < self.ema_alpha < 1.0):
            raise ValueError(
                f"ema_alpha must be in (0, 1), got {self.ema_alpha}"
            )
        if self.target_convergence is not None and self.target_convergence < 1:
            raise ValueError(
                f"target_convergence must be >= 1, got {self.target_convergence}"
            )
        if self.variance_gamma is not None and self.variance_gamma <= 0.0:
            raise ValueError(
                f"variance_gamma must be positive, got {self.variance_gamma}"
            )

    def get_preset(self) -> QualityPreset:
        """Get the quality preset parameters.

        Returns:
            QualityPreset for configured quality level.
        """
        return QUALITY_PRESETS[self.quality]

    def get_ema_alpha(self) -> float:
        """Get effective EMA alpha.

        Returns:
            Alpha from config or preset.
        """
        if self.ema_alpha is not None:
            return self.ema_alpha
        return self.get_preset().ema_alpha

    def get_target_convergence(self) -> int:
        """Get effective target convergence.

        Returns:
            Frames from config or preset.
        """
        if self.target_convergence is not None:
            return min(self.target_convergence, self.max_history)
        return min(self.get_preset().target_frames, self.max_history)

    def get_variance_gamma(self) -> float:
        """Get effective variance gamma.

        Returns:
            Gamma from config or preset.
        """
        if self.variance_gamma is not None:
            return self.variance_gamma
        return self.get_preset().variance_gamma

    def get_neighbourhood_size(self) -> int:
        """Get neighbourhood size.

        Returns:
            Configured neighbourhood size.
        """
        return self.neighbourhood_size


# =============================================================================
# Temporal Denoiser Statistics
# =============================================================================


@dataclass
class TemporalDenoiseStats:
    """Statistics from a temporal denoising pass.

    Attributes:
        total_pixels: Total pixels processed.
        valid_history_pixels: Pixels with valid history.
        rejected_pixels: Pixels where history was rejected.
        clamped_pixels: Pixels where history was clamped.
        converged_pixels: Pixels that have reached convergence.
        avg_alpha: Average blend alpha used.
        avg_history_length: Average history length.
        avg_variance: Average local variance.
        frame_index: Frame index for these stats.
        processing_time_ms: Time to process in milliseconds.
    """

    total_pixels: int = 0
    valid_history_pixels: int = 0
    rejected_pixels: int = 0
    clamped_pixels: int = 0
    converged_pixels: int = 0
    avg_alpha: float = 0.0
    avg_history_length: float = 0.0
    avg_variance: float = 0.0
    frame_index: int = 0
    processing_time_ms: float = 0.0

    @property
    def rejection_rate(self) -> float:
        """Get fraction of pixels with rejected history.

        Returns:
            Rejection rate [0, 1].
        """
        if self.total_pixels == 0:
            return 0.0
        return self.rejected_pixels / self.total_pixels

    @property
    def convergence_rate(self) -> float:
        """Get fraction of converged pixels.

        Returns:
            Convergence rate [0, 1].
        """
        if self.total_pixels == 0:
            return 0.0
        return self.converged_pixels / self.total_pixels

    @property
    def history_usage_rate(self) -> float:
        """Get fraction of pixels using valid history.

        Returns:
            History usage rate [0, 1].
        """
        if self.total_pixels == 0:
            return 0.0
        return self.valid_history_pixels / self.total_pixels

    @property
    def clamp_rate(self) -> float:
        """Get fraction of pixels where history was clamped.

        Returns:
            Clamp rate [0, 1].
        """
        if self.total_pixels == 0:
            return 0.0
        return self.clamped_pixels / self.total_pixels

    def reset(self) -> None:
        """Reset all statistics to zero."""
        self.total_pixels = 0
        self.valid_history_pixels = 0
        self.rejected_pixels = 0
        self.clamped_pixels = 0
        self.converged_pixels = 0
        self.avg_alpha = 0.0
        self.avg_history_length = 0.0
        self.avg_variance = 0.0
        self.processing_time_ms = 0.0


# =============================================================================
# Temporal Buffer
# =============================================================================


@dataclass
class TemporalBuffer:
    """A temporal history buffer with metadata.

    Attributes:
        texture: GPU texture resource.
        width: Buffer width in pixels.
        height: Buffer height in pixels.
        frame_index: Frame when buffer was last written.
        valid: Whether buffer contains valid data.
    """

    texture: Optional["Texture"] = None
    width: int = 0
    height: int = 0
    frame_index: int = -1
    valid: bool = False

    def is_allocated(self) -> bool:
        """Check if texture is allocated.

        Returns:
            True if texture exists.
        """
        return self.texture is not None

    def matches_dimensions(self, width: int, height: int) -> bool:
        """Check if buffer matches dimensions.

        Args:
            width: Expected width.
            height: Expected height.

        Returns:
            True if dimensions match.
        """
        return self.width == width and self.height == height

    def invalidate(self) -> None:
        """Mark buffer as invalid."""
        self.valid = False
        self.frame_index = -1

    def mark_written(self, frame_index: int) -> None:
        """Mark buffer as written.

        Args:
            frame_index: Current frame index.
        """
        self.valid = True
        self.frame_index = frame_index


@dataclass
class TemporalBufferSet:
    """Ping-pong buffer pair for temporal accumulation.

    Manages two buffers that alternate between history (read)
    and current (write) roles each frame.

    Attributes:
        buffer_a: First buffer.
        buffer_b: Second buffer.
        read_index: Index of buffer to read (0 or 1).
        frame_count: Total frames processed.
    """

    buffer_a: TemporalBuffer = field(default_factory=TemporalBuffer)
    buffer_b: TemporalBuffer = field(default_factory=TemporalBuffer)
    read_index: int = 0
    frame_count: int = 0

    @property
    def history_buffer(self) -> TemporalBuffer:
        """Get the history (read) buffer.

        Returns:
            Buffer containing previous frame's accumulated result.
        """
        return self.buffer_a if self.read_index == 0 else self.buffer_b

    @property
    def current_buffer(self) -> TemporalBuffer:
        """Get the current (write) buffer.

        Returns:
            Buffer to write this frame's accumulated result.
        """
        return self.buffer_b if self.read_index == 0 else self.buffer_a

    def swap(self) -> None:
        """Swap read and write buffers for next frame."""
        self.read_index = 1 - self.read_index
        self.frame_count += 1

    def invalidate_all(self) -> None:
        """Invalidate both buffers."""
        self.buffer_a.invalidate()
        self.buffer_b.invalidate()
        self.frame_count = 0

    def needs_resize(self, width: int, height: int) -> bool:
        """Check if buffers need resizing.

        Args:
            width: Target width.
            height: Target height.

        Returns:
            True if either buffer needs resize.
        """
        return not (
            self.buffer_a.matches_dimensions(width, height)
            and self.buffer_b.matches_dimensions(width, height)
        )

    def get_convergence_progress(self, target_frames: int = 16) -> float:
        """Get convergence progress.

        Args:
            target_frames: Target frames for convergence.

        Returns:
            Progress [0, 1] where 1 = fully converged.
        """
        if target_frames <= 0:
            return 1.0
        return min(1.0, self.frame_count / target_frames)

    def is_converged(self, target_frames: int = 16) -> bool:
        """Check if accumulation has converged.

        Args:
            target_frames: Target frames for convergence.

        Returns:
            True if frame_count >= target_frames.
        """
        return self.frame_count >= target_frames


# =============================================================================
# G-Buffer for Temporal Denoising
# =============================================================================


@dataclass
class TemporalGBuffer:
    """G-Buffer data required for temporal denoising.

    Attributes:
        depth: Linear depth buffer.
        normal: World-space normal buffer.
        velocity: Motion vector buffer (REQUIRED for temporal).
        albedo: Optional surface albedo.
    """

    depth: "Texture"
    normal: "Texture"
    velocity: "Texture"
    albedo: Optional["Texture"] = None

    def is_valid(self) -> bool:
        """Check if required G-Buffer textures are valid.

        Returns:
            True if depth, normal, and velocity are valid.
        """
        return (
            self.depth is not None
            and self.normal is not None
            and self.velocity is not None
            and self.depth.is_valid()
            and self.normal.is_valid()
            and self.velocity.is_valid()
        )

    def has_albedo(self) -> bool:
        """Check if albedo is available.

        Returns:
            True if albedo is present and valid.
        """
        return self.albedo is not None and self.albedo.is_valid()


# =============================================================================
# Temporal Denoiser
# =============================================================================


class TemporalDenoiser:
    """Temporal denoiser with variance-guided accumulation.

    Implements temporal denoising for ray-traced signals:
    1. Reproject history using velocity buffer
    2. Detect disocclusion via depth/normal/velocity tests
    3. Clamp history to current neighbourhood
    4. Compute variance-guided blend weight
    5. Blend current and history using EMA
    6. Track history length for adaptive blending

    Example:
        config = TemporalDenoiseConfig(
            quality=TemporalQuality.HIGH,
            target=TemporalTarget.GI,
        )
        denoiser = TemporalDenoiser(device, config)
        denoiser.setup(1920, 1080)

        # Each frame:
        stats = denoiser.denoise(
            noisy_input=gi_output,
            g_buffer=g_buffer,
            output=denoised_gi,
        )
    """

    def __init__(
        self,
        device: Optional["Device"] = None,
        config: Optional[TemporalDenoiseConfig] = None,
    ) -> None:
        """Initialize temporal denoiser.

        Args:
            device: RHI device for resource creation.
            config: Denoiser configuration.
        """
        self._device = device
        self._config = config or TemporalDenoiseConfig()

        # Components
        self._reprojector = Reprojector(
            disocclusion_mode=self._config.disocclusion_mode,
            depth_threshold=self._config.depth_threshold,
            normal_threshold=self._config.normal_threshold,
            velocity_threshold=self._config.velocity_threshold,
        )
        self._clamper = NeighbourhoodClamper(
            mode=self._config.clamping_mode,
            gamma=self._config.get_variance_gamma(),
            neighbourhood_size=self._config.get_neighbourhood_size(),
        )
        self._variance_guided = VarianceGuided(
            base_weight=1.0 - self._config.get_ema_alpha(),
        )
        self._blender = EMABlender(
            alpha=self._config.get_ema_alpha(),
            variance_modulation=self._config.use_variance_guided,
            motion_modulation=self._config.use_motion_modulation,
        )
        self._history_tracker = HistoryTracker(
            max_history_length=self._config.max_history,
            target_convergence=self._config.get_target_convergence(),
        )

        # Buffers
        self._buffers = TemporalBufferSet()
        self._stats = TemporalDenoiseStats()

        # State
        self._width = 0
        self._height = 0
        self._initialized = False
        self._frame_index = 0

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def device(self) -> Optional["Device"]:
        """Get RHI device."""
        return self._device

    @property
    def config(self) -> TemporalDenoiseConfig:
        """Get current configuration."""
        return self._config

    @config.setter
    def config(self, value: TemporalDenoiseConfig) -> None:
        """Set configuration and update components."""
        self._config = value
        self._update_components()

    @property
    def reprojector(self) -> Reprojector:
        """Get reprojector component."""
        return self._reprojector

    @property
    def clamper(self) -> NeighbourhoodClamper:
        """Get neighbourhood clamper component."""
        return self._clamper

    @property
    def variance_guided(self) -> VarianceGuided:
        """Get variance-guided calculator."""
        return self._variance_guided

    @property
    def blender(self) -> EMABlender:
        """Get EMA blender component."""
        return self._blender

    @property
    def history_tracker(self) -> HistoryTracker:
        """Get history tracker."""
        return self._history_tracker

    @property
    def buffers(self) -> TemporalBufferSet:
        """Get temporal buffer set."""
        return self._buffers

    @property
    def stats(self) -> TemporalDenoiseStats:
        """Get statistics from last execution."""
        return self._stats

    @property
    def is_initialized(self) -> bool:
        """Check if denoiser is initialized."""
        return self._initialized

    @property
    def width(self) -> int:
        """Get buffer width."""
        return self._width

    @property
    def height(self) -> int:
        """Get buffer height."""
        return self._height

    @property
    def frame_index(self) -> int:
        """Get current frame index."""
        return self._frame_index

    @property
    def convergence_progress(self) -> float:
        """Get convergence progress [0, 1]."""
        return self._buffers.get_convergence_progress(
            self._config.get_target_convergence()
        )

    @property
    def is_converged(self) -> bool:
        """Check if temporal accumulation has converged."""
        return self._buffers.is_converged(self._config.get_target_convergence())

    # -------------------------------------------------------------------------
    # Setup and Lifecycle
    # -------------------------------------------------------------------------

    def _update_components(self) -> None:
        """Update components from config."""
        self._reprojector.disocclusion_mode = self._config.disocclusion_mode
        self._reprojector.depth_threshold = self._config.depth_threshold
        self._reprojector.normal_threshold = self._config.normal_threshold
        self._reprojector.velocity_threshold = self._config.velocity_threshold

        self._clamper.mode = self._config.clamping_mode
        self._clamper.gamma = self._config.get_variance_gamma()
        self._clamper.neighbourhood_size = self._config.get_neighbourhood_size()

        self._variance_guided.base_weight = 1.0 - self._config.get_ema_alpha()

        self._blender.alpha = self._config.get_ema_alpha()
        self._blender.variance_modulation = self._config.use_variance_guided
        self._blender.motion_modulation = self._config.use_motion_modulation

        self._history_tracker.max_history_length = self._config.max_history
        self._history_tracker.target_convergence = self._config.get_target_convergence()

    def setup(self, width: int, height: int) -> None:
        """Initialize or resize denoiser.

        Args:
            width: Buffer width in pixels.
            height: Buffer height in pixels.

        Raises:
            ValueError: If dimensions are invalid.
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions: {width}x{height}")

        needs_resize = self._buffers.needs_resize(width, height)

        if needs_resize or not self._initialized:
            self._width = width
            self._height = height
            self._create_buffers(width, height)
            self._buffers.invalidate_all()
            self._history_tracker.invalidate()
            self._initialized = True

    def _create_buffers(self, width: int, height: int) -> None:
        """Create GPU texture resources.

        Args:
            width: Buffer width.
            height: Buffer height.
        """
        # Update buffer metadata
        self._buffers.buffer_a.width = width
        self._buffers.buffer_a.height = height

        self._buffers.buffer_b.width = width
        self._buffers.buffer_b.height = height

        # GPU texture creation would happen here with real device
        if self._device is not None:
            from engine.platform.rhi.resources import (
                Format,
                TextureDesc,
                TextureType,
                TextureUsage,
            )

            desc = TextureDesc(
                type=TextureType.TEXTURE_2D,
                format=Format.RGBA16_FLOAT,
                width=width,
                height=height,
                usage=TextureUsage.SHADER_RESOURCE | TextureUsage.UNORDERED_ACCESS,
            )

            self._buffers.buffer_a.texture = self._device.create_texture(desc)
            self._buffers.buffer_b.texture = self._device.create_texture(desc)

    def invalidate_history(self) -> None:
        """Invalidate all history buffers.

        Call on camera cuts, resolution changes, or other
        discontinuities that make history invalid.
        """
        self._buffers.invalidate_all()
        self._history_tracker.invalidate()
        self._stats.reset()

    def destroy(self) -> None:
        """Release GPU resources."""
        if self._buffers.buffer_a.texture is not None:
            self._buffers.buffer_a.texture.destroy()
            self._buffers.buffer_a.texture = None
        if self._buffers.buffer_b.texture is not None:
            self._buffers.buffer_b.texture.destroy()
            self._buffers.buffer_b.texture = None
        self._initialized = False

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.destroy()

    # -------------------------------------------------------------------------
    # Denoising
    # -------------------------------------------------------------------------

    def denoise(
        self,
        noisy_input: "Texture",
        g_buffer: TemporalGBuffer,
        output: "Texture",
        config: Optional[TemporalDenoiseConfig] = None,
    ) -> TemporalDenoiseStats:
        """Perform temporal denoising.

        Args:
            noisy_input: Input texture with noisy signal.
            g_buffer: G-Buffer with depth, normal, velocity.
            output: Output texture for denoised result.
            config: Optional override configuration.

        Returns:
            TemporalDenoiseStats with operation statistics.

        Raises:
            ValueError: If inputs are invalid.
        """
        if config is not None:
            self._config = config
            self._update_components()

        # Validate inputs
        self._validate_inputs(noisy_input, g_buffer, output)

        # Get dimensions
        input_desc = noisy_input.desc
        width = input_desc.width
        height = input_desc.height

        # Ensure setup
        if not self._initialized or self._buffers.needs_resize(width, height):
            self.setup(width, height)

        # Reset stats
        self._stats.reset()
        self._stats.frame_index = self._frame_index
        self._stats.total_pixels = width * height

        # Execute temporal pass
        self._execute_temporal_pass(noisy_input, g_buffer, output)

        # Update frame state
        self._frame_index += 1
        self._history_tracker.advance_frame()
        self._buffers.swap()
        self._buffers.current_buffer.mark_written(self._frame_index)

        return self._stats

    def _validate_inputs(
        self,
        noisy_input: "Texture",
        g_buffer: TemporalGBuffer,
        output: "Texture",
    ) -> None:
        """Validate input textures.

        Raises:
            ValueError: If any input is invalid.
        """
        if noisy_input is None or not noisy_input.is_valid():
            raise ValueError("noisy_input texture is invalid")
        if output is None or not output.is_valid():
            raise ValueError("output texture is invalid")
        if not g_buffer.is_valid():
            raise ValueError(
                "g_buffer is invalid (missing depth, normal, or velocity)"
            )

        # Check dimension match
        input_desc = noisy_input.desc
        output_desc = output.desc

        if input_desc.width != output_desc.width:
            raise ValueError(
                f"Output width ({output_desc.width}) does not match "
                f"input width ({input_desc.width})"
            )
        if input_desc.height != output_desc.height:
            raise ValueError(
                f"Output height ({output_desc.height}) does not match "
                f"input height ({input_desc.height})"
            )

    def _execute_temporal_pass(
        self,
        noisy_input: "Texture",
        g_buffer: TemporalGBuffer,
        output: "Texture",
    ) -> None:
        """Execute the temporal denoising pass.

        In real implementation, this dispatches a compute shader.
        Here we define the algorithm structure.

        Args:
            noisy_input: Noisy input texture.
            g_buffer: G-Buffer textures.
            output: Output texture.
        """
        # Shader algorithm (conceptual):
        #
        # For each pixel:
        # 1. Sample current noisy color
        # 2. Sample velocity and reproject to history UV
        # 3. Check reprojection validity (inside frame)
        # 4. Sample history at reprojected UV
        # 5. Sample depth/normal at current and history
        # 6. Check disocclusion (depth, normal, velocity)
        # 7. Compute local variance from neighbourhood
        # 8. Clamp history to neighbourhood bounds
        # 9. Compute variance-guided blend weight
        # 10. Blend current and clamped history
        # 11. Update history length
        # 12. Write result to output and current buffer

        # This is a stub - actual GPU dispatch would happen here
        _ = (
            noisy_input,
            g_buffer,
            output,
            self._buffers.history_buffer.texture,
            self._buffers.current_buffer.texture,
            self._config.get_ema_alpha(),
            self._config.get_variance_gamma(),
            self._config.get_neighbourhood_size(),
        )


# =============================================================================
# Convenience Factory Functions
# =============================================================================


def create_gi_temporal_denoiser(device: Optional["Device"] = None) -> TemporalDenoiser:
    """Create temporal denoiser optimized for global illumination.

    GI benefits from longer accumulation and aggressive variance guidance.

    Args:
        device: Optional RHI device.

    Returns:
        Configured TemporalDenoiser.
    """
    config = TemporalDenoiseConfig(
        quality=TemporalQuality.HIGH,
        target=TemporalTarget.GI,
        clamping_mode=ClampingMode.YCOCG_VARIANCE,
        variance_gamma=1.0,
        target_convergence=16,
        use_variance_guided=True,
    )
    return TemporalDenoiser(device, config)


def create_reflection_temporal_denoiser(
    device: Optional["Device"] = None,
) -> TemporalDenoiser:
    """Create temporal denoiser optimized for reflections.

    Reflections need faster response and tighter clamping.

    Args:
        device: Optional RHI device.

    Returns:
        Configured TemporalDenoiser.
    """
    config = TemporalDenoiseConfig(
        quality=TemporalQuality.MEDIUM,
        target=TemporalTarget.REFLECTIONS,
        clamping_mode=ClampingMode.YCOCG_VARIANCE,
        variance_gamma=0.75,  # Tighter clamp
        ema_alpha=0.15,  # Faster response
        target_convergence=12,
        use_variance_guided=True,
        use_motion_modulation=True,
    )
    return TemporalDenoiser(device, config)


def create_shadow_temporal_denoiser(
    device: Optional["Device"] = None,
) -> TemporalDenoiser:
    """Create temporal denoiser optimized for shadows.

    Shadows can use simpler clamping and longer accumulation.

    Args:
        device: Optional RHI device.

    Returns:
        Configured TemporalDenoiser.
    """
    config = TemporalDenoiseConfig(
        quality=TemporalQuality.HIGH,
        target=TemporalTarget.SHADOWS,
        clamping_mode=ClampingMode.VARIANCE,  # RGB is fine for shadows
        variance_gamma=1.25,
        target_convergence=16,
        use_variance_guided=True,
    )
    return TemporalDenoiser(device, config)


def create_fast_temporal_denoiser(
    device: Optional["Device"] = None,
) -> TemporalDenoiser:
    """Create fast temporal denoiser for real-time applications.

    Prioritizes responsiveness over stability.

    Args:
        device: Optional RHI device.

    Returns:
        Configured TemporalDenoiser.
    """
    config = TemporalDenoiseConfig(
        quality=TemporalQuality.LOW,
        clamping_mode=ClampingMode.AABB,  # Fastest
        ema_alpha=0.2,  # Fast response
        target_convergence=8,
        neighbourhood_size=3,  # Small neighbourhood
        use_variance_guided=False,  # Skip variance calculation
    )
    return TemporalDenoiser(device, config)


def create_quality_temporal_denoiser(
    device: Optional["Device"] = None,
) -> TemporalDenoiser:
    """Create high-quality temporal denoiser for offline rendering.

    Prioritizes stability and quality over speed.

    Args:
        device: Optional RHI device.

    Returns:
        Configured TemporalDenoiser.
    """
    config = TemporalDenoiseConfig(
        quality=TemporalQuality.ULTRA,
        clamping_mode=ClampingMode.YCOCG_VARIANCE,
        variance_gamma=0.75,  # Tight clamp
        ema_alpha=0.05,  # Very slow, stable
        target_convergence=24,
        max_history=64,
        neighbourhood_size=5,
        use_variance_guided=True,
        use_motion_modulation=True,
        anti_flicker=True,
    )
    return TemporalDenoiser(device, config)


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Core Denoiser
    "TemporalDenoiser",
    # Configuration
    "TemporalDenoiseConfig",
    "TemporalQuality",
    "TemporalTarget",
    "ClampingMode",
    "DisocclusionMode",
    "QualityPreset",
    "QUALITY_PRESETS",
    # Components
    "VarianceGuided",
    "VarianceEstimate",
    "HistoryTracker",
    "HistoryEntry",
    "NeighbourhoodClamper",
    "Reprojector",
    "ReprojectionResult",
    "EMABlender",
    # Color Space
    "YCoCgConverter",
    # Buffers
    "TemporalBuffer",
    "TemporalBufferSet",
    "TemporalGBuffer",
    # Statistics
    "TemporalDenoiseStats",
    # Constants
    "MIN_HISTORY_LENGTH",
    "MAX_HISTORY_LENGTH",
    "DEFAULT_CONVERGENCE_FRAMES",
    "DEFAULT_VARIANCE_GAMMA",
    "DEFAULT_EMA_ALPHA",
    "EPSILON",
    # Factory Functions
    "create_gi_temporal_denoiser",
    "create_reflection_temporal_denoiser",
    "create_shadow_temporal_denoiser",
    "create_fast_temporal_denoiser",
    "create_quality_temporal_denoiser",
]
