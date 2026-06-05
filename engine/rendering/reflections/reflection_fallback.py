"""Reflection Fallback Chain System (T-GIR-P8.5).

Implements per-pixel reflection technique selection and blending:
- ReflectionTechnique: Enum for available reflection methods
- TechniqueResult: Result from a single technique evaluation
- TechniqueSelector: Per-pixel technique selection based on availability
- ConfidenceBlender: Blends between techniques based on confidence
- FallbackChainConfig: Configuration for fallback behavior
- ReflectionFallbackPass: Full-screen pass evaluating fallback chain
- TransitionManager: Temporal smoothing to prevent popping

The fallback chain priority:
1. RT Reflection (highest quality, hardware ray tracing)
2. SSR (screen-space reflections, HiZ ray marching)
3. Reflection Probes (world-space, pre-captured)
4. Environment Map (final fallback, infinite distance)

References:
    - T-GIR-P4.2 SSR HiZ ray marching (ssr_temporal.py)
    - T-GIR-P5.3 Probe blending (probe_blending.py)
    - T-GIR-P8.1 RT reflection rays (rt_reflections.py)
    - Shader: reflection_fallback_chain.comp.wgsl
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from engine.core.math.vec import Vec2, Vec3

if TYPE_CHECKING:
    from engine.rendering.reflections.rt_reflections import RTReflectionPass, ReflectionOutput
    from engine.rendering.reflections.ssr_temporal import SSRTemporalReprojection
    from engine.rendering.lighting.probe_blending import ProbeBlendPass


# =============================================================================
# Constants
# =============================================================================

# Default confidence threshold for accepting a technique
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# Default blend threshold for mixing techniques
DEFAULT_BLEND_THRESHOLD = 0.3

# Default temporal transition speed (0-1, higher = faster)
DEFAULT_TRANSITION_SPEED = 0.15

# Minimum confidence to consider a result valid
MIN_VALID_CONFIDENCE = 0.01

# History length for transition smoothing (in frames)
DEFAULT_HISTORY_LENGTH = 8

# Epsilon for numerical stability
EPSILON = 1e-6


# =============================================================================
# Reflection Technique Enum
# =============================================================================


class ReflectionTechnique(IntEnum):
    """Available reflection techniques in priority order.

    Higher values indicate lower priority (fallback techniques).

    RT_REFLECTION: Hardware ray-traced reflections (highest quality)
    SSR: Screen-space reflections via HiZ ray marching
    REFLECTION_PROBE: World-space reflection probes (pre-captured)
    ENVIRONMENT_MAP: Environment/skybox map (infinite fallback)
    """

    RT_REFLECTION = 0
    """Hardware ray-traced reflections - highest quality, most expensive."""

    SSR = 1
    """Screen-space reflections - good quality, moderate cost."""

    REFLECTION_PROBE = 2
    """World-space reflection probes - pre-captured, cheap."""

    ENVIRONMENT_MAP = 3
    """Environment/skybox map - always available fallback."""

    @property
    def is_realtime(self) -> bool:
        """Check if technique generates realtime reflections."""
        return self in (ReflectionTechnique.RT_REFLECTION, ReflectionTechnique.SSR)

    @property
    def is_screen_space(self) -> bool:
        """Check if technique operates in screen space."""
        return self == ReflectionTechnique.SSR

    @property
    def quality_level(self) -> float:
        """Get relative quality level (1.0 = highest)."""
        quality_map = {
            ReflectionTechnique.RT_REFLECTION: 1.0,
            ReflectionTechnique.SSR: 0.8,
            ReflectionTechnique.REFLECTION_PROBE: 0.5,
            ReflectionTechnique.ENVIRONMENT_MAP: 0.2,
        }
        return quality_map.get(self, 0.0)

    @staticmethod
    def from_name(name: str) -> "ReflectionTechnique":
        """Get technique from string name."""
        name_map = {
            "rt": ReflectionTechnique.RT_REFLECTION,
            "rt_reflection": ReflectionTechnique.RT_REFLECTION,
            "ssr": ReflectionTechnique.SSR,
            "probe": ReflectionTechnique.REFLECTION_PROBE,
            "reflection_probe": ReflectionTechnique.REFLECTION_PROBE,
            "env": ReflectionTechnique.ENVIRONMENT_MAP,
            "environment": ReflectionTechnique.ENVIRONMENT_MAP,
            "environment_map": ReflectionTechnique.ENVIRONMENT_MAP,
        }
        return name_map.get(name.lower(), ReflectionTechnique.ENVIRONMENT_MAP)


# =============================================================================
# Technique Result
# =============================================================================


@dataclass
class TechniqueResult:
    """Result from evaluating a single reflection technique.

    Attributes:
        color: Reflected color RGB.
        confidence: Confidence in the result [0, 1].
        hit_distance: Distance to reflection hit (or inf for environment).
        technique: Which technique produced this result.
        valid: Whether the result is valid/usable.
        roughness: Surface roughness used for sampling.
    """

    color: Vec3 = field(default_factory=Vec3.zero)
    confidence: float = 0.0
    hit_distance: float = float("inf")
    technique: ReflectionTechnique = ReflectionTechnique.ENVIRONMENT_MAP
    valid: bool = True
    roughness: float = 0.0

    def __post_init__(self) -> None:
        """Validate and clamp values."""
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.hit_distance = max(0.0, self.hit_distance)
        self.roughness = max(0.0, min(1.0, self.roughness))

    @property
    def is_miss(self) -> bool:
        """Check if this represents a miss (no valid hit)."""
        return self.confidence < MIN_VALID_CONFIDENCE or not self.valid

    @property
    def is_hit(self) -> bool:
        """Check if this represents a valid hit."""
        return self.confidence >= MIN_VALID_CONFIDENCE and self.valid

    def with_confidence(self, new_confidence: float) -> "TechniqueResult":
        """Create copy with new confidence value."""
        return TechniqueResult(
            color=Vec3(self.color.x, self.color.y, self.color.z),
            confidence=new_confidence,
            hit_distance=self.hit_distance,
            technique=self.technique,
            valid=self.valid,
            roughness=self.roughness,
        )

    def with_technique(self, technique: ReflectionTechnique) -> "TechniqueResult":
        """Create copy with different technique tag."""
        return TechniqueResult(
            color=Vec3(self.color.x, self.color.y, self.color.z),
            confidence=self.confidence,
            hit_distance=self.hit_distance,
            technique=technique,
            valid=self.valid,
            roughness=self.roughness,
        )

    @staticmethod
    def miss(technique: ReflectionTechnique = ReflectionTechnique.ENVIRONMENT_MAP) -> "TechniqueResult":
        """Create a miss result."""
        return TechniqueResult(
            color=Vec3.zero(),
            confidence=0.0,
            hit_distance=float("inf"),
            technique=technique,
            valid=False,
        )

    @staticmethod
    def from_color(
        color: Vec3,
        confidence: float,
        technique: ReflectionTechnique,
        hit_distance: float = 0.0,
    ) -> "TechniqueResult":
        """Create result from color and confidence."""
        return TechniqueResult(
            color=color,
            confidence=confidence,
            hit_distance=hit_distance,
            technique=technique,
            valid=confidence >= MIN_VALID_CONFIDENCE,
        )


# =============================================================================
# Technique Selector
# =============================================================================


class TechniqueSelector:
    """Selects per-pixel technique based on availability and quality.

    Implements the fallback chain logic:
    - RT miss -> try SSR
    - SSR miss -> try probes
    - Probe miss -> use environment map

    The selector tracks which techniques are enabled and available,
    and provides methods to determine the best technique for a pixel.
    """

    def __init__(
        self,
        enable_rt: bool = True,
        enable_ssr: bool = True,
        enable_probes: bool = True,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        """Initialize technique selector.

        Args:
            enable_rt: Whether RT reflections are available.
            enable_ssr: Whether SSR is available.
            enable_probes: Whether reflection probes are available.
            confidence_threshold: Minimum confidence to accept a result.
        """
        self._enable_rt = enable_rt
        self._enable_ssr = enable_ssr
        self._enable_probes = enable_probes
        self._confidence_threshold = max(0.0, min(1.0, confidence_threshold))

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def enable_rt(self) -> bool:
        """Check if RT is enabled."""
        return self._enable_rt

    @enable_rt.setter
    def enable_rt(self, value: bool) -> None:
        """Set RT enabled state."""
        self._enable_rt = value

    @property
    def enable_ssr(self) -> bool:
        """Check if SSR is enabled."""
        return self._enable_ssr

    @enable_ssr.setter
    def enable_ssr(self, value: bool) -> None:
        """Set SSR enabled state."""
        self._enable_ssr = value

    @property
    def enable_probes(self) -> bool:
        """Check if probes are enabled."""
        return self._enable_probes

    @enable_probes.setter
    def enable_probes(self, value: bool) -> None:
        """Set probes enabled state."""
        self._enable_probes = value

    @property
    def confidence_threshold(self) -> float:
        """Get confidence threshold."""
        return self._confidence_threshold

    @confidence_threshold.setter
    def confidence_threshold(self, value: float) -> None:
        """Set confidence threshold."""
        self._confidence_threshold = max(0.0, min(1.0, value))

    # -------------------------------------------------------------------------
    # Selection Methods
    # -------------------------------------------------------------------------

    def get_priority_order(self) -> List[ReflectionTechnique]:
        """Get techniques in priority order (highest first).

        Only returns enabled techniques.

        Returns:
            List of enabled techniques in priority order.
        """
        techniques = []

        if self._enable_rt:
            techniques.append(ReflectionTechnique.RT_REFLECTION)

        if self._enable_ssr:
            techniques.append(ReflectionTechnique.SSR)

        if self._enable_probes:
            techniques.append(ReflectionTechnique.REFLECTION_PROBE)

        # Environment map is always available as final fallback
        techniques.append(ReflectionTechnique.ENVIRONMENT_MAP)

        return techniques

    def should_try_next(self, result: TechniqueResult) -> bool:
        """Check if we should try the next fallback technique.

        Args:
            result: Result from current technique.

        Returns:
            True if result is insufficient and we should try next.
        """
        if not result.valid:
            return True

        if result.confidence < self._confidence_threshold:
            return True

        return False

    def select_technique(
        self,
        rt_result: Optional[TechniqueResult] = None,
        ssr_result: Optional[TechniqueResult] = None,
        probe_result: Optional[TechniqueResult] = None,
        env_result: Optional[TechniqueResult] = None,
    ) -> Tuple[ReflectionTechnique, TechniqueResult]:
        """Select best technique based on available results.

        Evaluates results in priority order and returns the first
        one that meets the confidence threshold, or blends multiple
        if appropriate.

        Args:
            rt_result: RT reflection result (if available).
            ssr_result: SSR result (if available).
            probe_result: Probe result (if available).
            env_result: Environment map result (always available).

        Returns:
            Tuple of (selected_technique, result).
        """
        # Try RT first (highest quality)
        if self._enable_rt and rt_result is not None:
            if not self.should_try_next(rt_result):
                return (ReflectionTechnique.RT_REFLECTION, rt_result)

        # Try SSR
        if self._enable_ssr and ssr_result is not None:
            if not self.should_try_next(ssr_result):
                return (ReflectionTechnique.SSR, ssr_result)

        # Try probes
        if self._enable_probes and probe_result is not None:
            if not self.should_try_next(probe_result):
                return (ReflectionTechnique.REFLECTION_PROBE, probe_result)

        # Final fallback: environment map
        if env_result is not None:
            return (ReflectionTechnique.ENVIRONMENT_MAP, env_result)

        # No result at all - return miss
        return (ReflectionTechnique.ENVIRONMENT_MAP, TechniqueResult.miss())

    def get_next_technique(
        self,
        current: ReflectionTechnique,
    ) -> Optional[ReflectionTechnique]:
        """Get the next fallback technique after current.

        Args:
            current: Current technique.

        Returns:
            Next technique in priority order, or None if at end.
        """
        priority_order = self.get_priority_order()

        try:
            current_idx = priority_order.index(current)
            if current_idx + 1 < len(priority_order):
                return priority_order[current_idx + 1]
        except ValueError:
            pass

        return None

    def is_technique_enabled(self, technique: ReflectionTechnique) -> bool:
        """Check if a technique is enabled.

        Args:
            technique: Technique to check.

        Returns:
            True if technique is enabled.
        """
        if technique == ReflectionTechnique.RT_REFLECTION:
            return self._enable_rt
        elif technique == ReflectionTechnique.SSR:
            return self._enable_ssr
        elif technique == ReflectionTechnique.REFLECTION_PROBE:
            return self._enable_probes
        elif technique == ReflectionTechnique.ENVIRONMENT_MAP:
            return True  # Always available
        return False


# =============================================================================
# Confidence Blender
# =============================================================================


class ConfidenceBlender:
    """Blends between techniques based on confidence values.

    Provides smooth transitions when technique changes by blending
    results weighted by their confidence values.
    """

    def __init__(
        self,
        blend_threshold: float = DEFAULT_BLEND_THRESHOLD,
        transition_speed: float = DEFAULT_TRANSITION_SPEED,
    ) -> None:
        """Initialize confidence blender.

        Args:
            blend_threshold: Confidence difference threshold for blending.
            transition_speed: Speed of temporal transitions [0, 1].
        """
        self._blend_threshold = max(0.0, min(1.0, blend_threshold))
        self._transition_speed = max(0.0, min(1.0, transition_speed))

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def blend_threshold(self) -> float:
        """Get blend threshold."""
        return self._blend_threshold

    @blend_threshold.setter
    def blend_threshold(self, value: float) -> None:
        """Set blend threshold."""
        self._blend_threshold = max(0.0, min(1.0, value))

    @property
    def transition_speed(self) -> float:
        """Get transition speed."""
        return self._transition_speed

    @transition_speed.setter
    def transition_speed(self, value: float) -> None:
        """Set transition speed."""
        self._transition_speed = max(0.0, min(1.0, value))

    # -------------------------------------------------------------------------
    # Blending Methods
    # -------------------------------------------------------------------------

    def compute_blend_weight(
        self,
        primary_confidence: float,
        secondary_confidence: float,
    ) -> float:
        """Compute blend weight between two techniques.

        The weight determines how much of the secondary result to blend in.
        Returns 0.0 if primary is fully confident, approaches 1.0 as
        primary confidence drops.

        Args:
            primary_confidence: Confidence of primary (preferred) result.
            secondary_confidence: Confidence of secondary (fallback) result.

        Returns:
            Blend weight for secondary [0, 1].
        """
        if primary_confidence >= 1.0 - EPSILON:
            return 0.0

        if primary_confidence < EPSILON:
            return 1.0

        if secondary_confidence < EPSILON:
            return 0.0

        # Compute blend region
        # Below threshold: blend more secondary
        # Above threshold: use mostly primary
        if primary_confidence < self._blend_threshold:
            # Linear ramp in blend region
            t = primary_confidence / self._blend_threshold
            # Weight secondary higher as primary confidence drops
            return (1.0 - t) * (secondary_confidence / max(secondary_confidence, EPSILON))
        else:
            # Above threshold: minimal blending
            excess = (primary_confidence - self._blend_threshold) / (1.0 - self._blend_threshold + EPSILON)
            return (1.0 - excess) * 0.2 * secondary_confidence

    def lerp_colors(
        self,
        color_a: Vec3,
        color_b: Vec3,
        t: float,
    ) -> Vec3:
        """Linearly interpolate between two colors.

        Args:
            color_a: First color.
            color_b: Second color.
            t: Interpolation factor [0, 1].

        Returns:
            Interpolated color.
        """
        t = max(0.0, min(1.0, t))
        return Vec3(
            color_a.x * (1.0 - t) + color_b.x * t,
            color_a.y * (1.0 - t) + color_b.y * t,
            color_a.z * (1.0 - t) + color_b.z * t,
        )

    def blend_results(
        self,
        primary: TechniqueResult,
        secondary: TechniqueResult,
    ) -> TechniqueResult:
        """Blend two technique results based on confidence.

        Args:
            primary: Primary (preferred) result.
            secondary: Secondary (fallback) result.

        Returns:
            Blended result.
        """
        # Handle edge cases
        if not primary.valid and not secondary.valid:
            return TechniqueResult.miss()

        if not primary.valid:
            return secondary

        if not secondary.valid:
            return primary

        # Compute blend weight
        blend_weight = self.compute_blend_weight(
            primary.confidence,
            secondary.confidence,
        )

        # Blend colors
        blended_color = self.lerp_colors(
            primary.color,
            secondary.color,
            blend_weight,
        )

        # Blend confidence (weighted average)
        blended_confidence = (
            primary.confidence * (1.0 - blend_weight)
            + secondary.confidence * blend_weight
        )

        # Use primary's hit distance if it has a valid hit
        hit_distance = primary.hit_distance
        if primary.hit_distance == float("inf") and secondary.hit_distance < float("inf"):
            hit_distance = secondary.hit_distance

        # Use primary technique if weight is low, secondary if high
        technique = primary.technique if blend_weight < 0.5 else secondary.technique

        return TechniqueResult(
            color=blended_color,
            confidence=blended_confidence,
            hit_distance=hit_distance,
            technique=technique,
            valid=True,
            roughness=primary.roughness,
        )

    def blend_chain(
        self,
        results: List[TechniqueResult],
    ) -> TechniqueResult:
        """Blend a chain of technique results.

        Iteratively blends from highest to lowest priority.

        Args:
            results: List of results in priority order (highest first).

        Returns:
            Final blended result.
        """
        if not results:
            return TechniqueResult.miss()

        if len(results) == 1:
            return results[0]

        # Start with highest priority
        current = results[0]

        # Blend in each subsequent result
        for i in range(1, len(results)):
            current = self.blend_results(current, results[i])

        return current


# =============================================================================
# Fallback Chain Configuration
# =============================================================================


@dataclass
class FallbackChainConfig:
    """Configuration for the reflection fallback chain.

    Attributes:
        enable_rt: Enable RT reflections (if hardware available).
        enable_ssr: Enable screen-space reflections.
        enable_probes: Enable reflection probes.
        blend_threshold: Confidence threshold for blending.
        transition_speed: Temporal transition smoothing speed.
        confidence_threshold: Minimum confidence to accept result.
        max_blend_distance: Maximum distance for probe blending.
        roughness_threshold: Max roughness for realtime techniques.
    """

    enable_rt: bool = True
    enable_ssr: bool = True
    enable_probes: bool = True
    blend_threshold: float = DEFAULT_BLEND_THRESHOLD
    transition_speed: float = DEFAULT_TRANSITION_SPEED
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    max_blend_distance: float = 100.0
    roughness_threshold: float = 0.7

    def __post_init__(self) -> None:
        """Validate configuration values."""
        self.blend_threshold = max(0.0, min(1.0, self.blend_threshold))
        self.transition_speed = max(0.0, min(1.0, self.transition_speed))
        self.confidence_threshold = max(0.0, min(1.0, self.confidence_threshold))
        self.max_blend_distance = max(1.0, self.max_blend_distance)
        self.roughness_threshold = max(0.0, min(1.0, self.roughness_threshold))

    def validate(self) -> List[str]:
        """Validate configuration and return errors.

        Returns:
            List of error messages (empty if valid).
        """
        errors = []

        if not 0.0 <= self.blend_threshold <= 1.0:
            errors.append("blend_threshold must be in [0, 1]")

        if not 0.0 <= self.transition_speed <= 1.0:
            errors.append("transition_speed must be in [0, 1]")

        if not 0.0 <= self.confidence_threshold <= 1.0:
            errors.append("confidence_threshold must be in [0, 1]")

        if self.max_blend_distance <= 0:
            errors.append("max_blend_distance must be positive")

        return errors

    @staticmethod
    def high_quality() -> "FallbackChainConfig":
        """High quality preset with all techniques enabled."""
        return FallbackChainConfig(
            enable_rt=True,
            enable_ssr=True,
            enable_probes=True,
            blend_threshold=0.2,
            transition_speed=0.1,
            confidence_threshold=0.6,
        )

    @staticmethod
    def performance() -> "FallbackChainConfig":
        """Performance preset with RT disabled."""
        return FallbackChainConfig(
            enable_rt=False,
            enable_ssr=True,
            enable_probes=True,
            blend_threshold=0.4,
            transition_speed=0.2,
            confidence_threshold=0.4,
        )

    @staticmethod
    def minimal() -> "FallbackChainConfig":
        """Minimal preset with only probes and environment."""
        return FallbackChainConfig(
            enable_rt=False,
            enable_ssr=False,
            enable_probes=True,
            blend_threshold=0.5,
            transition_speed=0.3,
            confidence_threshold=0.3,
        )


# =============================================================================
# Transition Manager
# =============================================================================


@dataclass
class PixelHistory:
    """History data for a single pixel.

    Attributes:
        technique_history: Ring buffer of recent techniques.
        confidence_history: Ring buffer of recent confidences.
        color_history: Ring buffer of recent colors.
        frame_count: Number of frames in history.
        current_technique: Current stable technique.
        transition_progress: Progress of current transition [0, 1].
    """

    technique_history: List[ReflectionTechnique] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    color_history: List[Vec3] = field(default_factory=list)
    frame_count: int = 0
    current_technique: ReflectionTechnique = ReflectionTechnique.ENVIRONMENT_MAP
    transition_progress: float = 1.0


class TransitionManager:
    """Manages temporal transitions to prevent technique popping.

    Tracks per-pixel technique history and smooths transitions
    when the selected technique changes between frames.
    """

    def __init__(
        self,
        history_length: int = DEFAULT_HISTORY_LENGTH,
        transition_speed: float = DEFAULT_TRANSITION_SPEED,
    ) -> None:
        """Initialize transition manager.

        Args:
            history_length: Number of frames to track.
            transition_speed: Speed of transitions [0, 1].
        """
        self._history_length = max(1, history_length)
        self._transition_speed = max(0.0, min(1.0, transition_speed))

        # Per-pixel history (sparse storage)
        self._pixel_history: Dict[Tuple[int, int], PixelHistory] = {}

        # Statistics
        self._transitions_this_frame = 0
        self._total_pixels_managed = 0

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def history_length(self) -> int:
        """Get history length."""
        return self._history_length

    @history_length.setter
    def history_length(self, value: int) -> None:
        """Set history length (clears existing history)."""
        self._history_length = max(1, value)
        self._pixel_history.clear()

    @property
    def transition_speed(self) -> float:
        """Get transition speed."""
        return self._transition_speed

    @transition_speed.setter
    def transition_speed(self, value: float) -> None:
        """Set transition speed."""
        self._transition_speed = max(0.0, min(1.0, value))

    @property
    def transitions_this_frame(self) -> int:
        """Get number of transitions this frame."""
        return self._transitions_this_frame

    @property
    def total_pixels_managed(self) -> int:
        """Get total pixels with history."""
        return len(self._pixel_history)

    # -------------------------------------------------------------------------
    # History Management
    # -------------------------------------------------------------------------

    def update_history(
        self,
        x: int,
        y: int,
        technique: ReflectionTechnique,
        confidence: float,
        color: Vec3,
    ) -> None:
        """Update history for a pixel.

        Args:
            x: Pixel X coordinate.
            y: Pixel Y coordinate.
            technique: Selected technique this frame.
            confidence: Confidence of result.
            color: Reflection color.
        """
        key = (x, y)

        if key not in self._pixel_history:
            self._pixel_history[key] = PixelHistory()

        history = self._pixel_history[key]

        # Add to history buffers
        history.technique_history.append(technique)
        history.confidence_history.append(confidence)
        history.color_history.append(Vec3(color.x, color.y, color.z))

        # Trim to history length
        while len(history.technique_history) > self._history_length:
            history.technique_history.pop(0)
            history.confidence_history.pop(0)
            history.color_history.pop(0)

        history.frame_count += 1

    def get_stable_technique(
        self,
        x: int,
        y: int,
        current_technique: ReflectionTechnique,
    ) -> ReflectionTechnique:
        """Get temporally stable technique for a pixel.

        Returns the current technique if it has been consistent,
        or the most frequent recent technique if there's instability.

        Args:
            x: Pixel X coordinate.
            y: Pixel Y coordinate.
            current_technique: Currently selected technique.

        Returns:
            Stable technique selection.
        """
        key = (x, y)

        if key not in self._pixel_history:
            return current_technique

        history = self._pixel_history[key]

        if len(history.technique_history) < 2:
            return current_technique

        # Count technique frequencies
        technique_counts: Dict[ReflectionTechnique, int] = {}
        for tech in history.technique_history:
            technique_counts[tech] = technique_counts.get(tech, 0) + 1

        # Find most frequent technique
        most_frequent = max(technique_counts.items(), key=lambda x: x[1])

        # If current matches most frequent, use it
        if current_technique == most_frequent[0]:
            history.current_technique = current_technique
            return current_technique

        # If most frequent is dominant (>50%), use it for stability
        if most_frequent[1] > len(history.technique_history) / 2:
            if history.current_technique != most_frequent[0]:
                self._transitions_this_frame += 1
            history.current_technique = most_frequent[0]
            return most_frequent[0]

        # Otherwise use current
        if history.current_technique != current_technique:
            self._transitions_this_frame += 1
        history.current_technique = current_technique
        return current_technique

    def smooth_transition(
        self,
        x: int,
        y: int,
        current_result: TechniqueResult,
    ) -> TechniqueResult:
        """Apply temporal smoothing to prevent popping.

        Blends between previous and current results based on
        transition speed to smooth technique changes.

        Args:
            x: Pixel X coordinate.
            y: Pixel Y coordinate.
            current_result: Current frame result.

        Returns:
            Smoothed result.
        """
        key = (x, y)

        if key not in self._pixel_history:
            return current_result

        history = self._pixel_history[key]

        if len(history.color_history) < 2:
            return current_result

        # Get previous color
        prev_color = history.color_history[-2] if len(history.color_history) >= 2 else current_result.color

        # Check if technique changed
        technique_changed = False
        if len(history.technique_history) >= 2:
            technique_changed = history.technique_history[-1] != history.technique_history[-2]

        if technique_changed:
            # Start new transition
            history.transition_progress = 0.0

        # Update transition progress
        if history.transition_progress < 1.0:
            history.transition_progress = min(
                1.0,
                history.transition_progress + self._transition_speed,
            )

        # Blend colors
        t = history.transition_progress
        smoothed_color = Vec3(
            prev_color.x * (1.0 - t) + current_result.color.x * t,
            prev_color.y * (1.0 - t) + current_result.color.y * t,
            prev_color.z * (1.0 - t) + current_result.color.z * t,
        )

        return TechniqueResult(
            color=smoothed_color,
            confidence=current_result.confidence,
            hit_distance=current_result.hit_distance,
            technique=current_result.technique,
            valid=current_result.valid,
            roughness=current_result.roughness,
        )

    def reset_frame_stats(self) -> None:
        """Reset per-frame statistics."""
        self._transitions_this_frame = 0

    def clear_history(self) -> None:
        """Clear all pixel history."""
        self._pixel_history.clear()

    def clear_pixel(self, x: int, y: int) -> None:
        """Clear history for a specific pixel."""
        key = (x, y)
        if key in self._pixel_history:
            del self._pixel_history[key]

    def get_pixel_history(self, x: int, y: int) -> Optional[PixelHistory]:
        """Get history for a pixel.

        Args:
            x: Pixel X coordinate.
            y: Pixel Y coordinate.

        Returns:
            PixelHistory or None if not tracked.
        """
        return self._pixel_history.get((x, y))


# =============================================================================
# Reflection Fallback Pass
# =============================================================================


@dataclass
class FallbackPassOutput:
    """Output from fallback pass at a single pixel.

    Attributes:
        color: Final blended reflection color.
        technique: Primary technique used.
        confidence: Final confidence value.
        hit_distance: Reflection hit distance.
    """

    color: Vec3
    technique: ReflectionTechnique
    confidence: float
    hit_distance: float


class ReflectionFallbackPass:
    """Full-screen pass evaluating the reflection fallback chain.

    Reads results from RT, SSR, and probe passes, evaluates the
    fallback chain per pixel, and outputs the final blended
    reflection color plus a technique mask for debugging.

    Usage:
        config = FallbackChainConfig()
        pass_obj = ReflectionFallbackPass(config)
        pass_obj.set_inputs(rt_buffer, ssr_buffer, probe_buffer, env_buffer)
        pass_obj.execute(width, height)
        final = pass_obj.get_final_buffer()
        mask = pass_obj.get_technique_mask()
    """

    def __init__(
        self,
        config: Optional[FallbackChainConfig] = None,
    ) -> None:
        """Initialize fallback pass.

        Args:
            config: Fallback chain configuration.
        """
        self._config = config or FallbackChainConfig()

        # Create components
        self._selector = TechniqueSelector(
            enable_rt=self._config.enable_rt,
            enable_ssr=self._config.enable_ssr,
            enable_probes=self._config.enable_probes,
            confidence_threshold=self._config.confidence_threshold,
        )
        self._blender = ConfidenceBlender(
            blend_threshold=self._config.blend_threshold,
            transition_speed=self._config.transition_speed,
        )
        self._transition_manager = TransitionManager(
            transition_speed=self._config.transition_speed,
        )

        # Input samplers (functions that sample the input buffers)
        self._rt_sampler: Optional[Callable[[int, int], TechniqueResult]] = None
        self._ssr_sampler: Optional[Callable[[int, int], TechniqueResult]] = None
        self._probe_sampler: Optional[Callable[[int, int], TechniqueResult]] = None
        self._env_sampler: Optional[Callable[[int, int], TechniqueResult]] = None

        # Output buffers
        self._output_width = 0
        self._output_height = 0
        self._final_buffer: List[FallbackPassOutput] = []
        self._technique_mask: List[ReflectionTechnique] = []

        # Statistics
        self._stats: Dict[str, int] = {
            "rt_pixels": 0,
            "ssr_pixels": 0,
            "probe_pixels": 0,
            "env_pixels": 0,
            "blended_pixels": 0,
            "total_pixels": 0,
        }
        self._initialized = False

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def config(self) -> FallbackChainConfig:
        """Get configuration."""
        return self._config

    @config.setter
    def config(self, value: FallbackChainConfig) -> None:
        """Set configuration and update components."""
        self._config = value
        self._selector.enable_rt = value.enable_rt
        self._selector.enable_ssr = value.enable_ssr
        self._selector.enable_probes = value.enable_probes
        self._selector.confidence_threshold = value.confidence_threshold
        self._blender.blend_threshold = value.blend_threshold
        self._blender.transition_speed = value.transition_speed
        self._transition_manager.transition_speed = value.transition_speed

    @property
    def output_width(self) -> int:
        """Get output width."""
        return self._output_width

    @property
    def output_height(self) -> int:
        """Get output height."""
        return self._output_height

    @property
    def is_initialized(self) -> bool:
        """Check if pass is initialized."""
        return self._initialized

    # -------------------------------------------------------------------------
    # Input Configuration
    # -------------------------------------------------------------------------

    def set_rt_sampler(
        self,
        sampler: Optional[Callable[[int, int], TechniqueResult]],
    ) -> None:
        """Set RT result sampler function.

        Args:
            sampler: Function(x, y) -> TechniqueResult.
        """
        self._rt_sampler = sampler

    def set_ssr_sampler(
        self,
        sampler: Optional[Callable[[int, int], TechniqueResult]],
    ) -> None:
        """Set SSR result sampler function.

        Args:
            sampler: Function(x, y) -> TechniqueResult.
        """
        self._ssr_sampler = sampler

    def set_probe_sampler(
        self,
        sampler: Optional[Callable[[int, int], TechniqueResult]],
    ) -> None:
        """Set probe result sampler function.

        Args:
            sampler: Function(x, y) -> TechniqueResult.
        """
        self._probe_sampler = sampler

    def set_env_sampler(
        self,
        sampler: Optional[Callable[[int, int], TechniqueResult]],
    ) -> None:
        """Set environment map sampler function.

        Args:
            sampler: Function(x, y) -> TechniqueResult.
        """
        self._env_sampler = sampler

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    def execute(self, width: int, height: int) -> None:
        """Execute the fallback pass.

        Processes all pixels, evaluating the fallback chain and
        producing final blended reflections.

        Args:
            width: Output width in pixels.
            height: Output height in pixels.
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions: {width}x{height}")

        self._output_width = width
        self._output_height = height

        # Reset statistics
        self._stats = {
            "rt_pixels": 0,
            "ssr_pixels": 0,
            "probe_pixels": 0,
            "env_pixels": 0,
            "blended_pixels": 0,
            "total_pixels": 0,
        }
        self._transition_manager.reset_frame_stats()

        # Allocate output buffers
        total_pixels = width * height
        self._final_buffer = []
        self._technique_mask = []

        # Process each pixel
        for y in range(height):
            for x in range(width):
                output = self._process_pixel(x, y)
                self._final_buffer.append(output)
                self._technique_mask.append(output.technique)

        self._stats["total_pixels"] = total_pixels
        self._initialized = True

    def _process_pixel(self, x: int, y: int) -> FallbackPassOutput:
        """Process a single pixel through the fallback chain.

        Args:
            x: Pixel X coordinate.
            y: Pixel Y coordinate.

        Returns:
            FallbackPassOutput for this pixel.
        """
        # Sample all available techniques
        rt_result = None
        ssr_result = None
        probe_result = None
        env_result = None

        if self._config.enable_rt and self._rt_sampler is not None:
            rt_result = self._rt_sampler(x, y)
            rt_result = rt_result.with_technique(ReflectionTechnique.RT_REFLECTION)

        if self._config.enable_ssr and self._ssr_sampler is not None:
            ssr_result = self._ssr_sampler(x, y)
            ssr_result = ssr_result.with_technique(ReflectionTechnique.SSR)

        if self._config.enable_probes and self._probe_sampler is not None:
            probe_result = self._probe_sampler(x, y)
            probe_result = probe_result.with_technique(ReflectionTechnique.REFLECTION_PROBE)

        if self._env_sampler is not None:
            env_result = self._env_sampler(x, y)
            env_result = env_result.with_technique(ReflectionTechnique.ENVIRONMENT_MAP)
        else:
            # Default environment fallback
            env_result = TechniqueResult(
                color=Vec3(0.2, 0.3, 0.5),  # Sky blue
                confidence=1.0,
                technique=ReflectionTechnique.ENVIRONMENT_MAP,
                valid=True,
            )

        # Collect valid results for blending
        results: List[TechniqueResult] = []
        if rt_result is not None and rt_result.valid:
            results.append(rt_result)
        if ssr_result is not None and ssr_result.valid:
            results.append(ssr_result)
        if probe_result is not None and probe_result.valid:
            results.append(probe_result)
        if env_result is not None and env_result.valid:
            results.append(env_result)

        # Select primary technique
        selected_tech, primary_result = self._selector.select_technique(
            rt_result=rt_result,
            ssr_result=ssr_result,
            probe_result=probe_result,
            env_result=env_result,
        )

        # Blend if multiple valid results
        if len(results) > 1:
            final_result = self._blender.blend_chain(results)
            self._stats["blended_pixels"] += 1
        else:
            final_result = primary_result

        # Apply temporal smoothing
        stable_tech = self._transition_manager.get_stable_technique(x, y, selected_tech)
        final_result = self._transition_manager.smooth_transition(x, y, final_result)

        # Update history
        self._transition_manager.update_history(
            x, y,
            selected_tech,
            final_result.confidence,
            final_result.color,
        )

        # Update technique statistics
        if selected_tech == ReflectionTechnique.RT_REFLECTION:
            self._stats["rt_pixels"] += 1
        elif selected_tech == ReflectionTechnique.SSR:
            self._stats["ssr_pixels"] += 1
        elif selected_tech == ReflectionTechnique.REFLECTION_PROBE:
            self._stats["probe_pixels"] += 1
        else:
            self._stats["env_pixels"] += 1

        return FallbackPassOutput(
            color=final_result.color,
            technique=stable_tech,
            confidence=final_result.confidence,
            hit_distance=final_result.hit_distance,
        )

    # -------------------------------------------------------------------------
    # Output Access
    # -------------------------------------------------------------------------

    def get_final_buffer(self) -> List[FallbackPassOutput]:
        """Get the final blended reflection buffer.

        Returns:
            List of FallbackPassOutput, one per pixel (row-major).
        """
        return self._final_buffer

    def get_final_at(self, x: int, y: int) -> FallbackPassOutput:
        """Get final output at pixel coordinates.

        Args:
            x: Pixel X coordinate.
            y: Pixel Y coordinate.

        Returns:
            FallbackPassOutput at that pixel.
        """
        if not self._final_buffer:
            return FallbackPassOutput(
                color=Vec3.zero(),
                technique=ReflectionTechnique.ENVIRONMENT_MAP,
                confidence=0.0,
                hit_distance=float("inf"),
            )

        if x < 0 or x >= self._output_width or y < 0 or y >= self._output_height:
            return FallbackPassOutput(
                color=Vec3.zero(),
                technique=ReflectionTechnique.ENVIRONMENT_MAP,
                confidence=0.0,
                hit_distance=float("inf"),
            )

        idx = y * self._output_width + x
        return self._final_buffer[idx]

    def get_technique_mask(self) -> List[ReflectionTechnique]:
        """Get the technique mask buffer.

        Returns:
            List of ReflectionTechnique, one per pixel (row-major).
        """
        return self._technique_mask

    def get_technique_at(self, x: int, y: int) -> ReflectionTechnique:
        """Get technique used at pixel coordinates.

        Args:
            x: Pixel X coordinate.
            y: Pixel Y coordinate.

        Returns:
            ReflectionTechnique at that pixel.
        """
        if not self._technique_mask:
            return ReflectionTechnique.ENVIRONMENT_MAP

        if x < 0 or x >= self._output_width or y < 0 or y >= self._output_height:
            return ReflectionTechnique.ENVIRONMENT_MAP

        idx = y * self._output_width + x
        return self._technique_mask[idx]

    def get_statistics(self) -> Dict[str, Any]:
        """Get pass execution statistics.

        Returns:
            Dict with pixel counts per technique, transitions, etc.
        """
        stats = dict(self._stats)
        stats["transitions"] = self._transition_manager.transitions_this_frame
        stats["pixels_with_history"] = self._transition_manager.total_pixels_managed

        # Calculate percentages
        total = max(1, stats["total_pixels"])
        stats["rt_percent"] = stats["rt_pixels"] / total * 100.0
        stats["ssr_percent"] = stats["ssr_pixels"] / total * 100.0
        stats["probe_percent"] = stats["probe_pixels"] / total * 100.0
        stats["env_percent"] = stats["env_pixels"] / total * 100.0
        stats["blend_percent"] = stats["blended_pixels"] / total * 100.0

        return stats

    def invalidate_history(self) -> None:
        """Invalidate all temporal history.

        Call on camera cuts or major scene changes.
        """
        self._transition_manager.clear_history()


# =============================================================================
# WGSL Shader Generation
# =============================================================================


def generate_fallback_chain_wgsl(config: FallbackChainConfig) -> str:
    """Generate WGSL compute shader for fallback chain.

    Args:
        config: Fallback chain configuration.

    Returns:
        WGSL shader source for reflection_fallback_chain.comp.wgsl.
    """
    return f"""// Reflection Fallback Chain Compute Shader
// Generated for T-GIR-P8.5 Reflection Fallback Chain
// reflection_fallback_chain.comp.wgsl

// Constants
const ENABLE_RT: bool = {"true" if config.enable_rt else "false"};
const ENABLE_SSR: bool = {"true" if config.enable_ssr else "false"};
const ENABLE_PROBES: bool = {"true" if config.enable_probes else "false"};
const CONFIDENCE_THRESHOLD: f32 = {config.confidence_threshold:.6f};
const BLEND_THRESHOLD: f32 = {config.blend_threshold:.6f};
const TRANSITION_SPEED: f32 = {config.transition_speed:.6f};
const EPSILON: f32 = 1e-6;

// Technique IDs (must match ReflectionTechnique enum)
const TECH_RT: u32 = 0u;
const TECH_SSR: u32 = 1u;
const TECH_PROBE: u32 = 2u;
const TECH_ENV: u32 = 3u;

// Technique result structure
struct TechniqueResult {{
    color: vec3<f32>,
    confidence: f32,
    hit_distance: f32,
    technique: u32,
    valid: u32,
    _pad: u32,
}};

// Uniforms
struct Uniforms {{
    output_size: vec2<u32>,
    frame_index: u32,
    _pad: u32,
}};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var rt_buffer: texture_2d<f32>;
@group(0) @binding(2) var rt_confidence: texture_2d<f32>;
@group(0) @binding(3) var ssr_buffer: texture_2d<f32>;
@group(0) @binding(4) var ssr_confidence: texture_2d<f32>;
@group(0) @binding(5) var probe_buffer: texture_2d<f32>;
@group(0) @binding(6) var probe_confidence: texture_2d<f32>;
@group(0) @binding(7) var env_buffer: texture_2d<f32>;
@group(0) @binding(8) var<storage, read_write> output_color: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> output_technique: array<u32>;
@group(0) @binding(10) var<storage, read> history_color: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read> history_technique: array<u32>;

// Check if we should try next technique (confidence below threshold)
fn should_try_next(confidence: f32) -> bool {{
    return confidence < CONFIDENCE_THRESHOLD;
}}

// Compute blend weight between primary and secondary
fn compute_blend_weight(primary_conf: f32, secondary_conf: f32) -> f32 {{
    if (primary_conf >= 1.0 - EPSILON) {{
        return 0.0;
    }}
    if (primary_conf < EPSILON) {{
        return 1.0;
    }}
    if (secondary_conf < EPSILON) {{
        return 0.0;
    }}

    if (primary_conf < BLEND_THRESHOLD) {{
        let t = primary_conf / BLEND_THRESHOLD;
        return (1.0 - t) * (secondary_conf / max(secondary_conf, EPSILON));
    }} else {{
        let excess = (primary_conf - BLEND_THRESHOLD) / (1.0 - BLEND_THRESHOLD + EPSILON);
        return (1.0 - excess) * 0.2 * secondary_conf;
    }}
}}

// Blend two colors based on weight
fn blend_colors(a: vec3<f32>, b: vec3<f32>, weight: f32) -> vec3<f32> {{
    let t = clamp(weight, 0.0, 1.0);
    return mix(a, b, t);
}}

// Sample RT buffer
fn sample_rt(pixel: vec2<i32>) -> TechniqueResult {{
    var result: TechniqueResult;
    if (ENABLE_RT) {{
        let color = textureLoad(rt_buffer, pixel, 0);
        let conf = textureLoad(rt_confidence, pixel, 0).r;
        result.color = color.rgb;
        result.confidence = conf;
        result.hit_distance = color.a;
        result.technique = TECH_RT;
        result.valid = select(0u, 1u, conf > EPSILON);
    }} else {{
        result.valid = 0u;
    }}
    return result;
}}

// Sample SSR buffer
fn sample_ssr(pixel: vec2<i32>) -> TechniqueResult {{
    var result: TechniqueResult;
    if (ENABLE_SSR) {{
        let color = textureLoad(ssr_buffer, pixel, 0);
        let conf = textureLoad(ssr_confidence, pixel, 0).r;
        result.color = color.rgb;
        result.confidence = conf;
        result.hit_distance = color.a;
        result.technique = TECH_SSR;
        result.valid = select(0u, 1u, conf > EPSILON);
    }} else {{
        result.valid = 0u;
    }}
    return result;
}}

// Sample probe buffer
fn sample_probes(pixel: vec2<i32>) -> TechniqueResult {{
    var result: TechniqueResult;
    if (ENABLE_PROBES) {{
        let color = textureLoad(probe_buffer, pixel, 0);
        let conf = textureLoad(probe_confidence, pixel, 0).r;
        result.color = color.rgb;
        result.confidence = conf;
        result.hit_distance = 1000.0; // Probes don't have accurate distance
        result.technique = TECH_PROBE;
        result.valid = select(0u, 1u, conf > EPSILON);
    }} else {{
        result.valid = 0u;
    }}
    return result;
}}

// Sample environment map
fn sample_environment(pixel: vec2<i32>) -> TechniqueResult {{
    var result: TechniqueResult;
    let color = textureLoad(env_buffer, pixel, 0);
    result.color = color.rgb;
    result.confidence = 1.0; // Environment is always valid
    result.hit_distance = 1e10; // Infinite distance
    result.technique = TECH_ENV;
    result.valid = 1u;
    return result;
}}

// Blend result with previous (for fallback)
fn blend_with_previous(prev: TechniqueResult, current: TechniqueResult) -> TechniqueResult {{
    var result: TechniqueResult;

    if (prev.valid == 0u) {{
        return current;
    }}
    if (current.valid == 0u) {{
        return prev;
    }}

    let weight = compute_blend_weight(prev.confidence, current.confidence);
    result.color = blend_colors(prev.color, current.color, weight);
    result.confidence = prev.confidence * (1.0 - weight) + current.confidence * weight;
    result.hit_distance = select(prev.hit_distance, current.hit_distance, weight > 0.5);
    result.technique = select(prev.technique, current.technique, weight > 0.5);
    result.valid = 1u;

    return result;
}}

// Apply temporal smoothing
fn apply_temporal(current: vec4<f32>, history: vec4<f32>, technique_changed: bool) -> vec4<f32> {{
    var t = TRANSITION_SPEED;
    if (!technique_changed) {{
        t = 0.5; // Faster blend when technique is stable
    }}
    return mix(history, current, t);
}}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let pixel = vec2<i32>(global_id.xy);

    if (global_id.x >= uniforms.output_size.x || global_id.y >= uniforms.output_size.y) {{
        return;
    }}

    let idx = global_id.y * uniforms.output_size.x + global_id.x;

    // Sample all techniques
    var rt_result = sample_rt(pixel);
    var ssr_result = sample_ssr(pixel);
    var probe_result = sample_probes(pixel);
    var env_result = sample_environment(pixel);

    // Evaluate fallback chain
    var final_result: TechniqueResult;
    final_result.valid = 0u;

    // Try RT first
    if (ENABLE_RT && rt_result.valid == 1u && !should_try_next(rt_result.confidence)) {{
        final_result = rt_result;
    }}
    // Try SSR
    else if (ENABLE_SSR && ssr_result.valid == 1u && !should_try_next(ssr_result.confidence)) {{
        final_result = blend_with_previous(rt_result, ssr_result);
    }}
    // Try probes
    else if (ENABLE_PROBES && probe_result.valid == 1u && !should_try_next(probe_result.confidence)) {{
        final_result = blend_with_previous(ssr_result, probe_result);
    }}
    // Fallback to environment
    else {{
        final_result = blend_with_previous(probe_result, env_result);
    }}

    // Apply temporal smoothing
    let history = history_color[idx];
    let history_tech = history_technique[idx];
    let technique_changed = (history_tech != final_result.technique);

    let current_color = vec4<f32>(final_result.color, final_result.confidence);
    let smoothed = apply_temporal(current_color, history, technique_changed);

    // Write outputs
    output_color[idx] = smoothed;
    output_technique[idx] = final_result.technique;
}}
"""


# =============================================================================
# Utility Functions
# =============================================================================


def evaluate_fallback_chain(
    config: FallbackChainConfig,
    rt_result: Optional[TechniqueResult] = None,
    ssr_result: Optional[TechniqueResult] = None,
    probe_result: Optional[TechniqueResult] = None,
    env_result: Optional[TechniqueResult] = None,
) -> TechniqueResult:
    """Evaluate the fallback chain for a single pixel (CPU reference).

    This is the reference implementation that matches the shader logic.

    Args:
        config: Fallback chain configuration.
        rt_result: RT reflection result (if available).
        ssr_result: SSR result (if available).
        probe_result: Probe result (if available).
        env_result: Environment map result.

    Returns:
        Final blended TechniqueResult.
    """
    threshold = config.confidence_threshold

    # Try RT first (highest quality) - only if enabled
    if config.enable_rt and rt_result is not None:
        if rt_result.valid and rt_result.confidence >= threshold:
            return rt_result

    # Try SSR
    if config.enable_ssr and ssr_result is not None:
        if ssr_result.valid and ssr_result.confidence >= threshold:
            # Blend with RT if it had partial hit AND RT is enabled
            if config.enable_rt and rt_result is not None and rt_result.valid:
                blender = ConfidenceBlender(config.blend_threshold)
                return blender.blend_results(rt_result, ssr_result)
            return ssr_result

    # Try probes
    if config.enable_probes and probe_result is not None:
        if probe_result.valid and probe_result.confidence >= threshold:
            # Blend with previous (SSR if enabled)
            if config.enable_ssr and ssr_result is not None and ssr_result.valid:
                blender = ConfidenceBlender(config.blend_threshold)
                return blender.blend_results(ssr_result, probe_result)
            return probe_result

    # Final fallback: environment map
    if env_result is not None and env_result.valid:
        # Blend with previous (probes if enabled)
        if config.enable_probes and probe_result is not None and probe_result.valid:
            blender = ConfidenceBlender(config.blend_threshold)
            return blender.blend_results(probe_result, env_result)
        return env_result

    # No valid result
    return TechniqueResult.miss()


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Constants
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_BLEND_THRESHOLD",
    "DEFAULT_TRANSITION_SPEED",
    "MIN_VALID_CONFIDENCE",
    "DEFAULT_HISTORY_LENGTH",
    # Enums
    "ReflectionTechnique",
    # Data structures
    "TechniqueResult",
    "FallbackPassOutput",
    "PixelHistory",
    # Configuration
    "FallbackChainConfig",
    # Core classes
    "TechniqueSelector",
    "ConfidenceBlender",
    "TransitionManager",
    "ReflectionFallbackPass",
    # Utilities
    "generate_fallback_chain_wgsl",
    "evaluate_fallback_chain",
]
