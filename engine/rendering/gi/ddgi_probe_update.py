"""DDGI Probe Update System for TRINITY (T-GIR-P2.4).

This module implements the probe update pipeline for Dynamic Diffuse Global
Illumination (DDGI), including:
    - Irradiance accumulation with distance-weighted gaussian
    - Visibility minimum-distance storage for occlusion
    - Temporal accumulation with configurable convergence
    - Importance-based update scheduling

The probe update process:
    1. Ray tracing produces hit data (radiance, distance, direction)
    2. IrradianceAccumulator accumulates radiance samples into SH
    3. VisibilityStorage tracks minimum distances per direction
    4. TemporalAccumulator blends new data with history
    5. ImportanceScheduler determines next-frame update priorities

References:
    - DDGI Paper (JCGT 2019): "Dynamic Diffuse Global Illumination with
      Ray-Traced Irradiance Fields"
    - RTXGI SDK: NVIDIA reference implementation
    - UE5 Lumen: Multi-bounce GI with importance sampling
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from engine.core.math.vec import Vec3
from engine.rendering.gi.sh_math import (
    SHCoefficientsL2,
    sh_basis_l2,
    sh_project_l2,
    fibonacci_sphere_directions,
)


# ============================================================================
# Constants
# ============================================================================

# Default temporal blend factors
DEFAULT_IRRADIANCE_BLEND = 0.97  # High persistence for stable lighting
DEFAULT_VISIBILITY_BLEND = 0.95  # Slightly faster adaptation for shadows

# Gaussian distance weighting parameters
DEFAULT_GAUSSIAN_SIGMA = 0.5  # Controls falloff sharpness
DEFAULT_CONFIDENCE_THRESHOLD = 0.1  # Minimum confidence to contribute

# Update scheduling constants
IMPORTANCE_CRITICAL = 1.0  # Update every frame
IMPORTANCE_HIGH = 0.75  # Update frequently
IMPORTANCE_MEDIUM = 0.5  # Update occasionally
IMPORTANCE_LOW = 0.25  # Update rarely

# Visibility encoding
VISIBILITY_MAX_DISTANCE = 1000.0  # Maximum trackable distance
VISIBILITY_MISS_DISTANCE = 10000.0  # Distance for ray misses


# ============================================================================
# Ray Hit Data
# ============================================================================


@dataclass
class ProbeRayHit:
    """Result of a single ray traced from a probe.

    Attributes:
        direction: Normalized ray direction (world space)
        radiance: HDR radiance at hit point (RGB)
        distance: Distance to hit (or VISIBILITY_MISS_DISTANCE for miss)
        hit_normal: Surface normal at hit point (optional)
        hit_backface: Whether the ray hit a backface
    """

    direction: Vec3
    radiance: Vec3
    distance: float
    hit_normal: Optional[Vec3] = None
    hit_backface: bool = False

    def is_miss(self) -> bool:
        """Check if this ray missed all geometry."""
        return self.distance >= VISIBILITY_MISS_DISTANCE * 0.9

    def is_valid(self) -> bool:
        """Check if this is a valid contributing sample."""
        return not self.hit_backface and self.distance > 0


# ============================================================================
# Distance Weighting
# ============================================================================


@dataclass
class DistanceWeightConfig:
    """Configuration for distance-based sample weighting.

    Attributes:
        sigma: Gaussian sigma for distance falloff
        min_distance: Minimum valid sample distance
        max_distance: Maximum distance for full weight
        confidence_power: Exponent for confidence calculation
    """

    sigma: float = DEFAULT_GAUSSIAN_SIGMA
    min_distance: float = 0.01  # 1cm minimum
    max_distance: float = 100.0  # 100m maximum for full weight
    confidence_power: float = 2.0

    def compute_weight(self, distance: float, mean_distance: float) -> float:
        """Compute distance-based gaussian weight.

        The weight is higher for samples near the mean distance,
        reducing influence of outliers (very close or very far hits).

        Args:
            distance: Sample distance
            mean_distance: Mean distance across samples

        Returns:
            Weight in range [0, 1]
        """
        if distance < self.min_distance:
            return 0.0
        if mean_distance < self.min_distance:
            return 1.0

        # Gaussian falloff centered on mean distance
        normalized_diff = (distance - mean_distance) / (mean_distance * self.sigma)
        return math.exp(-0.5 * normalized_diff * normalized_diff)

    def compute_confidence(self, sample_count: int, variance: float) -> float:
        """Compute confidence based on sample count and variance.

        Args:
            sample_count: Number of valid samples
            variance: Distance variance across samples

        Returns:
            Confidence in range [0, 1]
        """
        # More samples = higher confidence
        count_confidence = min(1.0, sample_count / 64.0)

        # Lower variance = higher confidence
        variance_confidence = 1.0 / (1.0 + variance ** self.confidence_power)

        return count_confidence * variance_confidence


def compute_distance_statistics(
    hits: List[ProbeRayHit],
) -> Tuple[float, float, int]:
    """Compute distance statistics for a set of ray hits.

    Args:
        hits: List of ray hit results

    Returns:
        Tuple of (mean_distance, variance, valid_count)
    """
    valid_distances = [
        h.distance for h in hits
        if h.is_valid() and not h.is_miss()
    ]

    if not valid_distances:
        return (VISIBILITY_MAX_DISTANCE, 0.0, 0)

    mean = sum(valid_distances) / len(valid_distances)
    variance = sum((d - mean) ** 2 for d in valid_distances) / len(valid_distances)

    return (mean, variance, len(valid_distances))


# ============================================================================
# Irradiance Accumulator
# ============================================================================


@dataclass
class IrradianceAccumulatorConfig:
    """Configuration for irradiance accumulation.

    Attributes:
        use_distance_weighting: Enable distance-based gaussian weighting
        distance_config: Distance weighting parameters
        min_radiance: Minimum radiance to clamp negative values
        max_radiance: Maximum radiance for firefly clamping
        backface_rejection: Reject backface hits
        confidence_threshold: Minimum confidence for accumulation
    """

    use_distance_weighting: bool = True
    distance_config: DistanceWeightConfig = field(
        default_factory=DistanceWeightConfig
    )
    min_radiance: float = 0.0
    max_radiance: float = 100.0  # HDR clamp
    backface_rejection: bool = True
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD


class IrradianceAccumulator:
    """Accumulates ray hit radiance into spherical harmonics.

    The accumulator processes ray hits from probe tracing and converts
    them into SH coefficients suitable for efficient irradiance lookup.

    Features:
        - Distance-weighted gaussian accumulation
        - Confidence-based sample weighting
        - Backface rejection
        - HDR radiance clamping

    Usage:
        accumulator = IrradianceAccumulator(config)
        sh_coeffs = accumulator.accumulate(ray_hits)
    """

    def __init__(self, config: Optional[IrradianceAccumulatorConfig] = None):
        """Initialize the accumulator.

        Args:
            config: Accumulation configuration (uses defaults if None)
        """
        self.config = config or IrradianceAccumulatorConfig()
        self._sample_count = 0
        self._total_weight = 0.0
        self._confidence = 0.0

    @property
    def sample_count(self) -> int:
        """Number of valid samples in last accumulation."""
        return self._sample_count

    @property
    def confidence(self) -> float:
        """Confidence level of last accumulation."""
        return self._confidence

    def accumulate(self, hits: List[ProbeRayHit]) -> SHCoefficientsL2:
        """Accumulate ray hits into SH coefficients.

        Args:
            hits: List of ray hit results from probe tracing

        Returns:
            SH coefficients representing accumulated irradiance
        """
        result = SHCoefficientsL2.zero()
        self._sample_count = 0
        self._total_weight = 0.0

        if not hits:
            self._confidence = 0.0
            return result

        # Compute distance statistics for weighting
        mean_dist, variance, valid_count = compute_distance_statistics(hits)

        for hit in hits:
            if not self._should_include_hit(hit):
                continue

            # Compute weight
            weight = self._compute_hit_weight(hit, mean_dist)
            if weight < 1e-6:
                continue

            # Clamp radiance
            radiance = self._clamp_radiance(hit.radiance)

            # Project into SH
            direction = np.array(
                [hit.direction.x, hit.direction.y, hit.direction.z],
                dtype=np.float32
            )
            color = np.array(
                [radiance.x, radiance.y, radiance.z],
                dtype=np.float32
            )

            projected = sh_project_l2(direction, color * weight)
            result.add(projected)

            self._sample_count += 1
            self._total_weight += weight

        # Normalize by total weight
        if self._total_weight > 0:
            result.scale(4.0 * math.pi / self._total_weight)

        # Compute confidence
        self._confidence = self.config.distance_config.compute_confidence(
            self._sample_count, variance
        )

        return result

    def _should_include_hit(self, hit: ProbeRayHit) -> bool:
        """Check if a hit should be included in accumulation."""
        if self.config.backface_rejection and hit.hit_backface:
            return False
        if not hit.is_valid():
            return False
        return True

    def _compute_hit_weight(self, hit: ProbeRayHit, mean_distance: float) -> float:
        """Compute the weight for a single hit."""
        if not self.config.use_distance_weighting:
            return 1.0

        if hit.is_miss():
            # Sky/miss samples get reduced weight
            return 0.5

        return self.config.distance_config.compute_weight(
            hit.distance, mean_distance
        )

    def _clamp_radiance(self, radiance: Vec3) -> Vec3:
        """Clamp radiance to valid range."""
        return Vec3(
            max(self.config.min_radiance, min(self.config.max_radiance, radiance.x)),
            max(self.config.min_radiance, min(self.config.max_radiance, radiance.y)),
            max(self.config.min_radiance, min(self.config.max_radiance, radiance.z)),
        )


# ============================================================================
# Visibility Storage
# ============================================================================


@dataclass
class VisibilityStorageConfig:
    """Configuration for visibility (depth) storage.

    Attributes:
        resolution: Number of directions to store (typically matches ray count)
        use_chebyshev: Use Chebyshev inequality for soft shadows
        depth_sharpness: Sharpness of depth comparison (higher = sharper shadows)
        bias: Bias to prevent self-shadowing
    """

    resolution: int = 256  # Directions per probe
    use_chebyshev: bool = True
    depth_sharpness: float = 50.0
    bias: float = 0.01


class VisibilityStorage:
    """Stores minimum distance per direction for occlusion queries.

    The visibility storage maintains a low-resolution depth map around
    each probe, enabling efficient soft shadow computation during sampling.

    Storage format:
        - Mean distance per octahedral-encoded direction
        - Variance for Chebyshev soft shadows (optional)

    Usage:
        storage = VisibilityStorage(config)
        storage.update_from_hits(hits)
        occlusion = storage.compute_occlusion(direction, query_distance)
    """

    def __init__(self, config: Optional[VisibilityStorageConfig] = None):
        """Initialize visibility storage.

        Args:
            config: Storage configuration (uses defaults if None)
        """
        self.config = config or VisibilityStorageConfig()

        # Mean distance per direction
        self._depth_mean = np.full(
            self.config.resolution,
            VISIBILITY_MAX_DISTANCE,
            dtype=np.float32
        )

        # Depth variance for Chebyshev (mean of squared distances)
        self._depth_mean_sq = np.full(
            self.config.resolution,
            VISIBILITY_MAX_DISTANCE ** 2,
            dtype=np.float32
        )

        # Direction lookup (precomputed)
        self._directions = self._generate_directions()

    def _generate_directions(self) -> NDArray[np.float32]:
        """Generate uniformly distributed directions."""
        return fibonacci_sphere_directions(self.config.resolution)

    def update_from_hits(self, hits: List[ProbeRayHit]) -> None:
        """Update visibility from ray hit results.

        Uses minimum distance per direction bucket to track closest
        geometry for shadow queries.

        Args:
            hits: Ray hit results from probe tracing
        """
        # Reset to maximum distance
        new_depth = np.full(
            self.config.resolution,
            VISIBILITY_MAX_DISTANCE,
            dtype=np.float32
        )

        for hit in hits:
            if not hit.is_valid():
                continue

            # Find closest direction bucket
            direction = np.array(
                [hit.direction.x, hit.direction.y, hit.direction.z],
                dtype=np.float32
            )
            bucket = self._find_direction_bucket(direction)

            # Store minimum distance
            distance = min(hit.distance, VISIBILITY_MAX_DISTANCE)
            new_depth[bucket] = min(new_depth[bucket], distance)

        # Update statistics
        self._depth_mean = new_depth
        self._depth_mean_sq = new_depth ** 2

    def _find_direction_bucket(self, direction: NDArray[np.float32]) -> int:
        """Find the closest direction bucket for a given direction."""
        dots = np.dot(self._directions, direction)
        return int(np.argmax(dots))

    def compute_occlusion(
        self,
        direction: Vec3,
        query_distance: float,
    ) -> float:
        """Compute occlusion factor for a direction and distance.

        Args:
            direction: Query direction (normalized)
            query_distance: Distance to query point

        Returns:
            Occlusion factor in [0, 1] (0 = fully occluded, 1 = visible)
        """
        dir_array = np.array(
            [direction.x, direction.y, direction.z],
            dtype=np.float32
        )
        bucket = self._find_direction_bucket(dir_array)

        stored_distance = self._depth_mean[bucket] + self.config.bias

        if self.config.use_chebyshev:
            return self._chebyshev_occlusion(
                bucket, query_distance, stored_distance
            )
        else:
            # Hard shadow
            return 1.0 if query_distance <= stored_distance else 0.0

    def _chebyshev_occlusion(
        self,
        bucket: int,
        query_distance: float,
        stored_distance: float,
    ) -> float:
        """Compute soft occlusion using Chebyshev inequality."""
        if query_distance <= stored_distance:
            return 1.0

        # Variance
        mean = self._depth_mean[bucket]
        mean_sq = self._depth_mean_sq[bucket]
        variance = max(0.0001, mean_sq - mean * mean)

        # Chebyshev upper bound
        d = query_distance - mean
        p_max = variance / (variance + d * d)

        # Apply sharpness
        return max(0.0, p_max ** self.config.depth_sharpness)

    def get_mean_depth(self, direction: Vec3) -> float:
        """Get stored mean depth for a direction."""
        dir_array = np.array(
            [direction.x, direction.y, direction.z],
            dtype=np.float32
        )
        bucket = self._find_direction_bucket(dir_array)
        return float(self._depth_mean[bucket])

    def to_bytes(self) -> bytes:
        """Convert visibility data to bytes for GPU upload."""
        # Pack as float32 pairs: (mean, mean_sq)
        data = np.column_stack([self._depth_mean, self._depth_mean_sq])
        return data.astype(np.float32).tobytes()

    @classmethod
    def from_bytes(cls, data: bytes, config: VisibilityStorageConfig) -> VisibilityStorage:
        """Create visibility storage from GPU buffer data."""
        storage = cls(config)
        arr = np.frombuffer(data, dtype=np.float32).reshape(-1, 2)
        storage._depth_mean = arr[:, 0].copy()
        storage._depth_mean_sq = arr[:, 1].copy()
        return storage


# ============================================================================
# Temporal Accumulation
# ============================================================================


@dataclass
class TemporalAccumulatorConfig:
    """Configuration for temporal accumulation.

    Attributes:
        irradiance_blend: Blend factor for irradiance (0 = no history, 1 = only history)
        visibility_blend: Blend factor for visibility
        min_frames: Minimum frames before stable convergence
        max_frames: Maximum accumulation window
        adaptive_blend: Enable variance-based adaptive blending
    """

    irradiance_blend: float = DEFAULT_IRRADIANCE_BLEND
    visibility_blend: float = DEFAULT_VISIBILITY_BLEND
    min_frames: int = 8
    max_frames: int = 32
    adaptive_blend: bool = True


class TemporalAccumulator:
    """Temporal accumulation for stable probe data.

    Blends new probe data with history using exponential moving average,
    with optional variance-based adaptive blending for faster response
    to lighting changes.

    The accumulator maintains:
        - Historical SH coefficients for irradiance
        - Frame counter for convergence tracking
        - Variance estimate for adaptive blending
    """

    def __init__(self, config: Optional[TemporalAccumulatorConfig] = None):
        """Initialize temporal accumulator.

        Args:
            config: Accumulation configuration
        """
        self.config = config or TemporalAccumulatorConfig()

        self._history: Optional[SHCoefficientsL2] = None
        self._frame_count = 0
        self._variance_estimate = 1.0
        self._last_blend = 0.0

    @property
    def frame_count(self) -> int:
        """Number of frames accumulated."""
        return self._frame_count

    @property
    def is_converged(self) -> bool:
        """Check if accumulation has converged."""
        return self._frame_count >= self.config.min_frames

    @property
    def convergence_factor(self) -> float:
        """Get convergence factor [0, 1] based on frame count."""
        if self.config.max_frames <= self.config.min_frames:
            return 1.0 if self._frame_count >= self.config.min_frames else 0.0

        progress = (self._frame_count - self.config.min_frames) / (
            self.config.max_frames - self.config.min_frames
        )
        return max(0.0, min(1.0, progress))

    def accumulate(
        self,
        new_irradiance: SHCoefficientsL2,
        confidence: float = 1.0,
    ) -> SHCoefficientsL2:
        """Accumulate new irradiance data with history.

        Args:
            new_irradiance: Newly computed SH coefficients
            confidence: Confidence in new data [0, 1]

        Returns:
            Blended SH coefficients
        """
        self._frame_count += 1

        if self._history is None:
            self._history = SHCoefficientsL2(new_irradiance.coeffs.copy())
            self._last_blend = 0.0
            return self._history

        # Compute blend factor
        blend = self._compute_blend_factor(new_irradiance, confidence)
        self._last_blend = blend

        # Lerp between history and new data
        # blend = 0.97 means: result = 0.97 * history + 0.03 * new
        result = self._history.lerp(new_irradiance, 1.0 - blend)

        self._history = SHCoefficientsL2(result.coeffs.copy())
        return result

    def _compute_blend_factor(
        self,
        new_irradiance: SHCoefficientsL2,
        confidence: float,
    ) -> float:
        """Compute adaptive blend factor."""
        base_blend = self.config.irradiance_blend

        if not self.config.adaptive_blend:
            return base_blend

        # Ramp up blend during initial convergence
        if self._frame_count < self.config.min_frames:
            t = self._frame_count / self.config.min_frames
            # Start with lower blend (faster adaptation), increase over time
            return base_blend * t + 0.5 * (1 - t)

        # Estimate variance between new and history
        diff = new_irradiance.coeffs - self._history.coeffs
        variance = float(np.mean(diff ** 2))

        # Smooth variance estimate
        self._variance_estimate = (
            0.95 * self._variance_estimate + 0.05 * variance
        )

        # Reduce blend (faster adaptation) when variance is high
        if self._variance_estimate > 0.1:
            # Lighting changed significantly, adapt faster
            return max(0.8, base_blend - 0.1)

        # Adjust by confidence
        return base_blend * confidence + 0.5 * (1 - confidence)

    def reset(self) -> None:
        """Reset accumulation history."""
        self._history = None
        self._frame_count = 0
        self._variance_estimate = 1.0
        self._last_blend = 0.0

    def force_update(self, irradiance: SHCoefficientsL2) -> None:
        """Force update history without blending."""
        self._history = SHCoefficientsL2(irradiance.coeffs.copy())
        self._frame_count = 1


# ============================================================================
# Importance Scheduler
# ============================================================================


class ProbeImportance(Enum):
    """Probe importance levels for update scheduling."""

    CRITICAL = auto()  # Update every frame (near camera, high variance)
    HIGH = auto()       # Update frequently (visible, moderate variance)
    MEDIUM = auto()     # Update occasionally (visible but stable)
    LOW = auto()        # Update rarely (far from camera, low variance)
    DORMANT = auto()    # Skip updates (outside view frustum)


@dataclass
class ImportanceSchedulerConfig:
    """Configuration for importance-based probe scheduling.

    Attributes:
        camera_distance_critical: Distance for critical importance
        camera_distance_high: Distance for high importance
        camera_distance_medium: Distance for medium importance
        variance_threshold_high: Variance for high importance
        variance_threshold_medium: Variance for medium importance
        critical_update_rate: Frames between critical probe updates
        high_update_rate: Frames between high importance updates
        medium_update_rate: Frames between medium importance updates
        low_update_rate: Frames between low importance updates
    """

    camera_distance_critical: float = 10.0
    camera_distance_high: float = 30.0
    camera_distance_medium: float = 60.0
    variance_threshold_high: float = 0.1
    variance_threshold_medium: float = 0.02
    critical_update_rate: int = 1  # Every frame
    high_update_rate: int = 2      # Every 2 frames
    medium_update_rate: int = 8    # Every 8 frames
    low_update_rate: int = 16      # Every 16 frames


@dataclass
class ProbeUpdateState:
    """Update state for a single probe.

    Attributes:
        probe_id: Unique probe identifier
        importance: Current importance level
        last_update_frame: Frame index of last update
        variance_estimate: Estimated temporal variance
        camera_distance: Distance to camera
        in_frustum: Whether probe is in view frustum
    """

    probe_id: int
    importance: ProbeImportance = ProbeImportance.MEDIUM
    last_update_frame: int = 0
    variance_estimate: float = 1.0
    camera_distance: float = 0.0
    in_frustum: bool = True


class ImportanceScheduler:
    """Schedules probe updates based on importance.

    The scheduler prioritizes probes based on:
        - Distance to camera (closer = more important)
        - Temporal variance (changing lighting = more important)
        - Visibility (in frustum = more important)
        - Recent update history (stale = more important)

    This enables efficient budget utilization by updating important
    probes frequently while allowing stable distant probes to update
    less often.
    """

    def __init__(self, config: Optional[ImportanceSchedulerConfig] = None):
        """Initialize the importance scheduler.

        Args:
            config: Scheduler configuration
        """
        self.config = config or ImportanceSchedulerConfig()
        self._probe_states: dict[int, ProbeUpdateState] = {}
        self._current_frame = 0

    def register_probe(self, probe_id: int) -> None:
        """Register a probe for scheduling.

        Args:
            probe_id: Unique probe identifier
        """
        self._probe_states[probe_id] = ProbeUpdateState(probe_id=probe_id)

    def unregister_probe(self, probe_id: int) -> None:
        """Remove a probe from scheduling."""
        self._probe_states.pop(probe_id, None)

    def update_probe_info(
        self,
        probe_id: int,
        camera_distance: float,
        in_frustum: bool,
        variance: float,
    ) -> None:
        """Update probe information for importance calculation.

        Args:
            probe_id: Probe identifier
            camera_distance: Distance from camera to probe
            in_frustum: Whether probe is visible
            variance: Recent temporal variance
        """
        if probe_id not in self._probe_states:
            self.register_probe(probe_id)

        state = self._probe_states[probe_id]
        state.camera_distance = camera_distance
        state.in_frustum = in_frustum
        state.variance_estimate = variance
        state.importance = self._compute_importance(state)

    def _compute_importance(self, state: ProbeUpdateState) -> ProbeImportance:
        """Compute importance level for a probe."""
        if not state.in_frustum:
            return ProbeImportance.DORMANT

        # Distance-based importance
        if state.camera_distance < self.config.camera_distance_critical:
            distance_importance = ProbeImportance.CRITICAL
        elif state.camera_distance < self.config.camera_distance_high:
            distance_importance = ProbeImportance.HIGH
        elif state.camera_distance < self.config.camera_distance_medium:
            distance_importance = ProbeImportance.MEDIUM
        else:
            distance_importance = ProbeImportance.LOW

        # Variance-based importance boost
        if state.variance_estimate > self.config.variance_threshold_high:
            variance_importance = ProbeImportance.CRITICAL
        elif state.variance_estimate > self.config.variance_threshold_medium:
            variance_importance = ProbeImportance.HIGH
        else:
            variance_importance = ProbeImportance.LOW

        # Take higher importance of the two
        if variance_importance.value < distance_importance.value:
            return variance_importance
        return distance_importance

    def get_probes_to_update(
        self,
        frame_index: int,
        max_updates: int,
    ) -> List[int]:
        """Get list of probe IDs that should update this frame.

        Args:
            frame_index: Current frame index
            max_updates: Maximum number of probes to update

        Returns:
            List of probe IDs to update, sorted by priority
        """
        self._current_frame = frame_index

        candidates: List[Tuple[float, int]] = []

        for probe_id, state in self._probe_states.items():
            if not self._should_update(state, frame_index):
                continue

            # Priority score (lower = higher priority)
            priority = self._compute_priority_score(state, frame_index)
            candidates.append((priority, probe_id))

        # Sort by priority (lower scores first)
        candidates.sort(key=lambda x: x[0])

        # Return top N probe IDs
        return [probe_id for _, probe_id in candidates[:max_updates]]

    def _should_update(self, state: ProbeUpdateState, frame_index: int) -> bool:
        """Check if a probe should be considered for update."""
        if state.importance == ProbeImportance.DORMANT:
            return False

        # Check update rate based on importance
        frames_since_update = frame_index - state.last_update_frame

        rates = {
            ProbeImportance.CRITICAL: self.config.critical_update_rate,
            ProbeImportance.HIGH: self.config.high_update_rate,
            ProbeImportance.MEDIUM: self.config.medium_update_rate,
            ProbeImportance.LOW: self.config.low_update_rate,
        }

        required_rate = rates.get(state.importance, self.config.low_update_rate)
        return frames_since_update >= required_rate

    def _compute_priority_score(
        self,
        state: ProbeUpdateState,
        frame_index: int,
    ) -> float:
        """Compute priority score for update ordering."""
        # Base score from importance level
        importance_scores = {
            ProbeImportance.CRITICAL: 0.0,
            ProbeImportance.HIGH: 1.0,
            ProbeImportance.MEDIUM: 2.0,
            ProbeImportance.LOW: 3.0,
            ProbeImportance.DORMANT: 100.0,
        }
        score = importance_scores.get(state.importance, 10.0)

        # Boost priority for stale probes
        frames_since_update = frame_index - state.last_update_frame
        staleness_boost = min(1.0, frames_since_update / 32.0)
        score -= staleness_boost * 0.5

        # Boost priority for high variance
        variance_boost = min(1.0, state.variance_estimate * 10.0)
        score -= variance_boost * 0.3

        # Boost priority for closer probes
        distance_boost = max(0.0, 1.0 - state.camera_distance / 100.0)
        score -= distance_boost * 0.2

        return score

    def mark_updated(self, probe_id: int, frame_index: int) -> None:
        """Mark a probe as updated.

        Args:
            probe_id: Probe that was updated
            frame_index: Frame index of update
        """
        if probe_id in self._probe_states:
            self._probe_states[probe_id].last_update_frame = frame_index

    def get_statistics(self) -> dict:
        """Get scheduler statistics."""
        importance_counts = {level: 0 for level in ProbeImportance}

        for state in self._probe_states.values():
            importance_counts[state.importance] += 1

        return {
            "total_probes": len(self._probe_states),
            "importance_distribution": {
                level.name: count for level, count in importance_counts.items()
            },
            "current_frame": self._current_frame,
        }


# ============================================================================
# Main Probe Updater
# ============================================================================


@dataclass
class DDGIProbeUpdaterConfig:
    """Configuration for the DDGI probe update system.

    Attributes:
        rays_per_probe: Number of rays traced per probe per update
        irradiance_config: Irradiance accumulation settings
        visibility_config: Visibility storage settings
        temporal_config: Temporal accumulation settings
        scheduler_config: Importance scheduling settings
        enable_visibility: Whether to compute visibility data
        enable_temporal: Whether to use temporal accumulation
        enable_scheduling: Whether to use importance-based scheduling
    """

    rays_per_probe: int = 256
    irradiance_config: IrradianceAccumulatorConfig = field(
        default_factory=IrradianceAccumulatorConfig
    )
    visibility_config: VisibilityStorageConfig = field(
        default_factory=VisibilityStorageConfig
    )
    temporal_config: TemporalAccumulatorConfig = field(
        default_factory=TemporalAccumulatorConfig
    )
    scheduler_config: ImportanceSchedulerConfig = field(
        default_factory=ImportanceSchedulerConfig
    )
    enable_visibility: bool = True
    enable_temporal: bool = True
    enable_scheduling: bool = True


@dataclass
class ProbeData:
    """Complete data for a single probe.

    Attributes:
        probe_id: Unique identifier
        position: World position
        irradiance: SH coefficients for irradiance
        visibility: Visibility/depth storage
        temporal: Temporal accumulator state
    """

    probe_id: int
    position: Vec3
    irradiance: SHCoefficientsL2 = field(default_factory=SHCoefficientsL2.zero)
    visibility: Optional[VisibilityStorage] = None
    temporal: Optional[TemporalAccumulator] = None


class DDGIProbeUpdater:
    """Main orchestrator for DDGI probe updates.

    The probe updater manages the complete probe update pipeline:
        1. Schedule which probes to update based on importance
        2. Accumulate ray hit radiance into SH coefficients
        3. Store visibility data for soft shadows
        4. Apply temporal accumulation for stability
        5. Track statistics for performance monitoring

    Usage:
        updater = DDGIProbeUpdater(config)

        # Register probes
        for probe in probe_grid:
            updater.register_probe(probe.id, probe.position)

        # Each frame
        probes_to_update = updater.get_probes_to_update(frame_index, budget)

        for probe_id in probes_to_update:
            hits = trace_rays_for_probe(probe_id)
            updater.update_probe(probe_id, hits)
    """

    def __init__(self, config: Optional[DDGIProbeUpdaterConfig] = None):
        """Initialize the probe updater.

        Args:
            config: Update configuration
        """
        self.config = config or DDGIProbeUpdaterConfig()

        self._probes: dict[int, ProbeData] = {}
        self._accumulator = IrradianceAccumulator(self.config.irradiance_config)
        self._scheduler = ImportanceScheduler(self.config.scheduler_config)

        self._frame_index = 0
        self._total_updates = 0
        self._last_update_count = 0

    @property
    def probe_count(self) -> int:
        """Number of registered probes."""
        return len(self._probes)

    @property
    def frame_index(self) -> int:
        """Current frame index."""
        return self._frame_index

    def register_probe(
        self,
        probe_id: int,
        position: Vec3,
    ) -> None:
        """Register a probe for updates.

        Args:
            probe_id: Unique probe identifier
            position: World-space probe position
        """
        probe = ProbeData(probe_id=probe_id, position=position)

        if self.config.enable_visibility:
            probe.visibility = VisibilityStorage(self.config.visibility_config)

        if self.config.enable_temporal:
            probe.temporal = TemporalAccumulator(self.config.temporal_config)

        self._probes[probe_id] = probe

        if self.config.enable_scheduling:
            self._scheduler.register_probe(probe_id)

    def unregister_probe(self, probe_id: int) -> None:
        """Remove a probe from updates."""
        self._probes.pop(probe_id, None)
        self._scheduler.unregister_probe(probe_id)

    def get_probes_to_update(
        self,
        frame_index: int,
        max_updates: int,
    ) -> List[int]:
        """Get list of probes that should update this frame.

        Args:
            frame_index: Current frame index
            max_updates: Maximum probes to update (budget)

        Returns:
            List of probe IDs to update
        """
        self._frame_index = frame_index

        if not self.config.enable_scheduling:
            # Update all probes if scheduling disabled
            return list(self._probes.keys())[:max_updates]

        return self._scheduler.get_probes_to_update(frame_index, max_updates)

    def update_probe_camera_info(
        self,
        probe_id: int,
        camera_position: Vec3,
        in_frustum: bool,
    ) -> None:
        """Update camera-relative information for a probe.

        Args:
            probe_id: Probe identifier
            camera_position: Camera world position
            in_frustum: Whether probe is in view frustum
        """
        if probe_id not in self._probes:
            return

        probe = self._probes[probe_id]
        distance = (probe.position - camera_position).length()

        # Get variance estimate from temporal accumulator
        variance = 1.0
        if probe.temporal is not None:
            variance = probe.temporal._variance_estimate

        self._scheduler.update_probe_info(
            probe_id, distance, in_frustum, variance
        )

    def update_probe(
        self,
        probe_id: int,
        hits: List[ProbeRayHit],
    ) -> SHCoefficientsL2:
        """Update a probe with new ray hit data.

        Args:
            probe_id: Probe to update
            hits: Ray hit results from tracing

        Returns:
            Updated SH coefficients for the probe
        """
        if probe_id not in self._probes:
            raise ValueError(f"Unknown probe ID: {probe_id}")

        probe = self._probes[probe_id]

        # Accumulate irradiance
        new_irradiance = self._accumulator.accumulate(hits)
        confidence = self._accumulator.confidence

        # Update visibility
        if probe.visibility is not None:
            probe.visibility.update_from_hits(hits)

        # Apply temporal accumulation
        if probe.temporal is not None:
            probe.irradiance = probe.temporal.accumulate(
                new_irradiance, confidence
            )
        else:
            probe.irradiance = new_irradiance

        # Mark as updated in scheduler
        if self.config.enable_scheduling:
            self._scheduler.mark_updated(probe_id, self._frame_index)

        self._total_updates += 1
        self._last_update_count += 1

        return probe.irradiance

    def get_probe_irradiance(self, probe_id: int) -> SHCoefficientsL2:
        """Get current irradiance SH coefficients for a probe."""
        if probe_id not in self._probes:
            return SHCoefficientsL2.zero()
        return self._probes[probe_id].irradiance

    def get_probe_visibility(self, probe_id: int) -> Optional[VisibilityStorage]:
        """Get visibility storage for a probe."""
        if probe_id not in self._probes:
            return None
        return self._probes[probe_id].visibility

    def get_probe_convergence(self, probe_id: int) -> float:
        """Get temporal convergence factor for a probe."""
        if probe_id not in self._probes:
            return 0.0
        probe = self._probes[probe_id]
        if probe.temporal is None:
            return 1.0
        return probe.temporal.convergence_factor

    def reset_probe(self, probe_id: int) -> None:
        """Reset a probe's accumulated data."""
        if probe_id not in self._probes:
            return
        probe = self._probes[probe_id]
        probe.irradiance = SHCoefficientsL2.zero()
        if probe.temporal is not None:
            probe.temporal.reset()

    def reset_all(self) -> None:
        """Reset all probe data."""
        for probe_id in self._probes:
            self.reset_probe(probe_id)

    def begin_frame(self, frame_index: int) -> None:
        """Begin a new frame.

        Args:
            frame_index: New frame index
        """
        self._frame_index = frame_index
        self._last_update_count = 0

    def end_frame(self) -> dict:
        """End frame and return statistics."""
        return {
            "frame_index": self._frame_index,
            "updates_this_frame": self._last_update_count,
            "total_updates": self._total_updates,
            "probe_count": len(self._probes),
            "scheduler_stats": self._scheduler.get_statistics(),
        }

    def iter_probes(self) -> Iterator[ProbeData]:
        """Iterate over all registered probes."""
        yield from self._probes.values()

    # ========================================================================
    # GPU Data Export
    # ========================================================================

    def build_irradiance_buffer(self) -> bytes:
        """Build GPU buffer containing all probe irradiance SH data.

        Returns:
            Bytes containing packed SH coefficients for all probes
        """
        data = []
        for probe in self._probes.values():
            data.append(probe.irradiance.to_bytes())
        return b"".join(data)

    def build_visibility_buffer(self) -> bytes:
        """Build GPU buffer containing all probe visibility data.

        Returns:
            Bytes containing packed visibility data
        """
        data = []
        for probe in self._probes.values():
            if probe.visibility is not None:
                data.append(probe.visibility.to_bytes())
        return b"".join(data)


# ============================================================================
# Utility Functions
# ============================================================================


def create_test_ray_hits(
    direction_count: int = 256,
    scene_type: str = "uniform",
) -> List[ProbeRayHit]:
    """Create test ray hits for testing.

    Args:
        direction_count: Number of rays
        scene_type: Type of test scene ("uniform", "gradient", "mixed")

    Returns:
        List of synthetic ray hits
    """
    directions = fibonacci_sphere_directions(direction_count)
    hits = []

    for dir_arr in directions:
        direction = Vec3(float(dir_arr[0]), float(dir_arr[1]), float(dir_arr[2]))

        if scene_type == "uniform":
            radiance = Vec3(0.5, 0.5, 0.5)
            distance = 10.0
        elif scene_type == "gradient":
            # Gradient based on y direction
            t = (direction.y + 1.0) * 0.5
            radiance = Vec3(t, t, t)
            distance = 5.0 + 10.0 * t
        elif scene_type == "mixed":
            # Some hits, some misses
            if direction.y > 0:
                radiance = Vec3(0.8, 0.7, 0.5)  # Sky
                distance = VISIBILITY_MISS_DISTANCE
            else:
                radiance = Vec3(0.1, 0.2, 0.1)  # Ground
                distance = 5.0
        else:
            radiance = Vec3(0.5, 0.5, 0.5)
            distance = 10.0

        hits.append(ProbeRayHit(
            direction=direction,
            radiance=radiance,
            distance=distance,
        ))

    return hits


def estimate_update_cost(
    probe_count: int,
    rays_per_probe: int,
    ms_per_million_rays: float = 0.5,
) -> float:
    """Estimate probe update cost in milliseconds.

    Args:
        probe_count: Number of probes to update
        rays_per_probe: Rays per probe
        ms_per_million_rays: Hardware-dependent cost factor

    Returns:
        Estimated time in milliseconds
    """
    total_rays = probe_count * rays_per_probe
    return (total_rays / 1_000_000.0) * ms_per_million_rays
