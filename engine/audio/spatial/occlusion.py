"""Audio Occlusion and Obstruction System.

Handles detection and response for sounds blocked by geometry:
- Multi-ray occlusion detection
- Material-aware transmission
- Low-pass filtering for occluded sounds
- Volume reduction based on occlusion type
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional, Tuple

from engine.audio.spatial.config import (
    OCCLUSION_INTERPOLATION_TIME,
    OCCLUSION_LOW_PASS_FREQ,
    OCCLUSION_MAX_RAYS,
    OCCLUSION_UPDATE_RATE,
    OCCLUSION_VOLUME_REDUCTION_DB,
    OBSTRUCTION_VOLUME_REDUCTION_DB,
)
from engine.core.math.vec import Vec3


class OcclusionType(Enum):
    """Types of audio occlusion."""

    NONE = auto()
    """No occlusion - direct line of sight."""

    PARTIAL = auto()
    """Some rays blocked - partial occlusion."""

    FULL = auto()
    """All rays blocked - full occlusion."""

    OBSTRUCTION = auto()
    """Direct path blocked but sound travels around obstacles."""


@dataclass
class RaycastHit:
    """Result of a geometry raycast."""

    hit: bool
    """Whether the ray hit geometry."""

    distance: float
    """Distance to hit point (if hit)."""

    point: Vec3 = field(default_factory=Vec3)
    """Hit point in world space."""

    normal: Vec3 = field(default_factory=lambda: Vec3(0.0, 1.0, 0.0))
    """Surface normal at hit point."""

    material_id: Optional[str] = None
    """Material identifier for the hit surface."""

    transmission: float = 0.0
    """How much sound passes through (0 = opaque, 1 = transparent)."""


# Type alias for raycast callback function
RaycastFunction = Callable[[Vec3, Vec3], Optional[RaycastHit]]


@dataclass
class OcclusionResult:
    """Result of occlusion detection."""

    occlusion_type: OcclusionType
    """Type of occlusion detected."""

    occlusion_factor: float
    """Occlusion amount (0.0 = no occlusion, 1.0 = full)."""

    low_pass_frequency: float
    """Low-pass filter cutoff frequency (Hz)."""

    volume_reduction_db: float
    """Volume reduction in decibels."""

    blocked_rays: int
    """Number of blocked rays."""

    total_rays: int
    """Total number of rays cast."""

    average_transmission: float = 0.0
    """Average transmission through blocking surfaces."""


@dataclass
class OcclusionSettings:
    """Per-source occlusion settings."""

    enabled: bool = True
    """Whether occlusion is enabled for this source."""

    num_rays: int = OCCLUSION_MAX_RAYS
    """Number of rays to cast for multi-ray occlusion."""

    update_rate: float = OCCLUSION_UPDATE_RATE
    """How often to update occlusion state (Hz)."""

    interpolation_time: float = OCCLUSION_INTERPOLATION_TIME
    """Time to interpolate occlusion changes (seconds)."""

    use_transmission: bool = True
    """Whether to account for material transmission."""

    ray_spread: float = 0.5
    """Spread of rays around direct path (meters)."""


class OcclusionDetector:
    """Detects occlusion between sound source and listener."""

    def __init__(self, num_rays: int = OCCLUSION_MAX_RAYS) -> None:
        """Initialize the occlusion detector.

        Args:
            num_rays: Number of rays to cast for detection.
        """
        self._num_rays = min(max(1, num_rays), OCCLUSION_MAX_RAYS)
        self._raycast_func: Optional[RaycastFunction] = None
        self._ray_spread = 0.5  # meters

    @property
    def num_rays(self) -> int:
        """Get number of rays used for detection."""
        return self._num_rays

    @num_rays.setter
    def num_rays(self, value: int) -> None:
        """Set number of rays used for detection."""
        self._num_rays = min(max(1, value), OCCLUSION_MAX_RAYS)

    @property
    def ray_spread(self) -> float:
        """Get ray spread distance."""
        return self._ray_spread

    @ray_spread.setter
    def ray_spread(self, value: float) -> None:
        """Set ray spread distance."""
        self._ray_spread = max(0.0, value)

    def set_raycast_function(self, func: Optional[RaycastFunction]) -> None:
        """Set the raycast function for geometry queries.

        Args:
            func: Function that takes (origin: Vec3, direction: Vec3) and
                  returns RaycastHit or None.
        """
        self._raycast_func = func

    def detect(
        self,
        source_pos: Vec3,
        listener_pos: Vec3,
        settings: Optional[OcclusionSettings] = None
    ) -> OcclusionResult:
        """Detect occlusion between source and listener.

        Args:
            source_pos: Sound source position.
            listener_pos: Listener position.
            settings: Optional per-source settings.

        Returns:
            OcclusionResult with occlusion information.
        """
        if self._raycast_func is None:
            return OcclusionResult(
                occlusion_type=OcclusionType.NONE,
                occlusion_factor=0.0,
                low_pass_frequency=20000.0,
                volume_reduction_db=0.0,
                blocked_rays=0,
                total_rays=self._num_rays,
                average_transmission=0.0
            )

        num_rays = settings.num_rays if settings else self._num_rays
        use_transmission = settings.use_transmission if settings else True
        spread = settings.ray_spread if settings else self._ray_spread

        blocked = 0
        total_transmission = 0.0

        # Generate ray origins around the source
        ray_origins = self._generate_ray_origins(source_pos, listener_pos, num_rays, spread)

        for origin in ray_origins:
            direction = listener_pos - origin
            distance = direction.length()

            if distance < 0.0001:
                continue

            direction = direction.normalized()
            hit = self._raycast_func(origin, direction)

            if hit is not None and hit.hit and hit.distance < distance:
                blocked += 1
                if use_transmission:
                    total_transmission += hit.transmission

        # Calculate occlusion factor
        occlusion_factor = blocked / num_rays if num_rays > 0 else 0.0
        avg_transmission = total_transmission / max(blocked, 1)

        # Determine occlusion type
        if blocked == 0:
            occ_type = OcclusionType.NONE
        elif blocked == num_rays:
            occ_type = OcclusionType.FULL
        elif blocked == 1 and num_rays > 1:
            # Only one ray blocked suggests obstruction
            occ_type = OcclusionType.OBSTRUCTION
        else:
            occ_type = OcclusionType.PARTIAL

        # Calculate effective occlusion accounting for transmission
        effective_occlusion = occlusion_factor * (1.0 - avg_transmission)

        # Calculate audio response
        low_pass = self._calculate_low_pass(effective_occlusion)
        volume_db = self._calculate_volume_reduction(occ_type, effective_occlusion)

        return OcclusionResult(
            occlusion_type=occ_type,
            occlusion_factor=effective_occlusion,
            low_pass_frequency=low_pass,
            volume_reduction_db=volume_db,
            blocked_rays=blocked,
            total_rays=num_rays,
            average_transmission=avg_transmission
        )

    def _generate_ray_origins(
        self,
        source: Vec3,
        listener: Vec3,
        num_rays: int,
        spread: float
    ) -> List[Vec3]:
        """Generate ray origins for multi-ray occlusion.

        Creates a pattern of rays around the direct path to detect
        partial occlusion and potential diffraction paths.

        Args:
            source: Source position.
            listener: Listener position.
            num_rays: Number of rays to generate.
            spread: Distance to spread rays from center.

        Returns:
            List of ray origin positions.
        """
        origins = [source]  # Always include direct path

        if num_rays <= 1:
            return origins

        # Calculate orthogonal vectors for spread pattern
        direction = listener - source
        if direction.length() < 0.0001:
            return origins

        direction = direction.normalized()

        # Find a perpendicular vector
        if abs(direction.y) < 0.9:
            up = Vec3(0.0, 1.0, 0.0)
        else:
            up = Vec3(1.0, 0.0, 0.0)

        right = Vec3(
            direction.y * up.z - direction.z * up.y,
            direction.z * up.x - direction.x * up.z,
            direction.x * up.y - direction.y * up.x
        ).normalized()

        up_vec = Vec3(
            right.y * direction.z - right.z * direction.y,
            right.z * direction.x - right.x * direction.z,
            right.x * direction.y - right.y * direction.x
        ).normalized()

        # Generate offset rays in a pattern
        offsets = [
            (spread, 0.0),
            (-spread, 0.0),
            (0.0, spread),
            (0.0, -spread),
            (spread * 0.707, spread * 0.707),
            (-spread * 0.707, spread * 0.707),
            (spread * 0.707, -spread * 0.707),
            (-spread * 0.707, -spread * 0.707),
        ]

        for i, (r_off, u_off) in enumerate(offsets):
            if len(origins) >= num_rays:
                break
            offset_pos = Vec3(
                source.x + right.x * r_off + up_vec.x * u_off,
                source.y + right.y * r_off + up_vec.y * u_off,
                source.z + right.z * r_off + up_vec.z * u_off
            )
            origins.append(offset_pos)

        return origins[:num_rays]

    def _calculate_low_pass(self, occlusion: float) -> float:
        """Calculate low-pass filter frequency based on occlusion.

        Higher occlusion results in lower cutoff frequency, simulating
        how high frequencies are absorbed more by obstacles.

        Args:
            occlusion: Occlusion factor (0-1).

        Returns:
            Low-pass filter cutoff frequency in Hz.
        """
        max_freq = 20000.0
        min_freq = OCCLUSION_LOW_PASS_FREQ

        # Exponential interpolation for more natural response
        t = occlusion * occlusion  # Quadratic curve
        return max_freq - (max_freq - min_freq) * t

    def _calculate_volume_reduction(
        self,
        occ_type: OcclusionType,
        factor: float
    ) -> float:
        """Calculate volume reduction in dB.

        Args:
            occ_type: Type of occlusion.
            factor: Occlusion factor (0-1).

        Returns:
            Volume reduction in decibels (negative value).
        """
        if occ_type == OcclusionType.NONE:
            return 0.0
        elif occ_type == OcclusionType.OBSTRUCTION:
            return -OBSTRUCTION_VOLUME_REDUCTION_DB * factor
        else:
            return -OCCLUSION_VOLUME_REDUCTION_DB * factor


@dataclass
class OcclusionState:
    """State for smoothed occlusion processing on a single source."""

    source_id: int = 0
    """Source identifier."""

    current_result: OcclusionResult = field(default_factory=lambda: OcclusionResult(
        occlusion_type=OcclusionType.NONE,
        occlusion_factor=0.0,
        low_pass_frequency=20000.0,
        volume_reduction_db=0.0,
        blocked_rays=0,
        total_rays=1
    ))
    """Current interpolated occlusion result."""

    target_result: OcclusionResult = field(default_factory=lambda: OcclusionResult(
        occlusion_type=OcclusionType.NONE,
        occlusion_factor=0.0,
        low_pass_frequency=20000.0,
        volume_reduction_db=0.0,
        blocked_rays=0,
        total_rays=1
    ))
    """Target occlusion result."""

    interpolation_time: float = OCCLUSION_INTERPOLATION_TIME
    """Time constant for interpolation."""

    time_since_update: float = 0.0
    """Time since last full occlusion update."""

    update_interval: float = 1.0 / OCCLUSION_UPDATE_RATE
    """Interval between full updates."""


class OcclusionProcessor:
    """Processor for occlusion with smoothing and rate limiting.

    Handles periodic occlusion checks and smooth interpolation
    to prevent audio artifacts from sudden occlusion changes.
    """

    def __init__(
        self,
        detector: Optional[OcclusionDetector] = None,
        update_rate: float = OCCLUSION_UPDATE_RATE,
        interpolation_time: float = OCCLUSION_INTERPOLATION_TIME
    ) -> None:
        """Initialize the occlusion processor.

        Args:
            detector: Occlusion detector to use.
            update_rate: How often to update occlusion (Hz).
            interpolation_time: Smoothing time constant (seconds).
        """
        self._detector = detector or OcclusionDetector()
        self._update_rate = max(1.0, update_rate)
        self._interpolation_time = max(0.0, interpolation_time)
        self._states: dict[int, OcclusionState] = {}

    @property
    def detector(self) -> OcclusionDetector:
        """Get the occlusion detector."""
        return self._detector

    @detector.setter
    def detector(self, value: OcclusionDetector) -> None:
        """Set the occlusion detector."""
        self._detector = value

    @property
    def update_rate(self) -> float:
        """Get update rate in Hz."""
        return self._update_rate

    @update_rate.setter
    def update_rate(self, value: float) -> None:
        """Set update rate in Hz."""
        self._update_rate = max(1.0, value)
        for state in self._states.values():
            state.update_interval = 1.0 / self._update_rate

    @property
    def interpolation_time(self) -> float:
        """Get interpolation time constant."""
        return self._interpolation_time

    @interpolation_time.setter
    def interpolation_time(self, value: float) -> None:
        """Set interpolation time constant."""
        self._interpolation_time = max(0.0, value)
        for state in self._states.values():
            state.interpolation_time = self._interpolation_time

    def get_or_create_state(self, source_id: int) -> OcclusionState:
        """Get or create state for a source."""
        if source_id not in self._states:
            self._states[source_id] = OcclusionState(
                source_id=source_id,
                interpolation_time=self._interpolation_time,
                update_interval=1.0 / self._update_rate
            )
        return self._states[source_id]

    def remove_state(self, source_id: int) -> None:
        """Remove state for a source."""
        self._states.pop(source_id, None)

    def clear_states(self) -> None:
        """Clear all states."""
        self._states.clear()

    def update(
        self,
        source_id: int,
        source_pos: Vec3,
        listener_pos: Vec3,
        dt: float,
        settings: Optional[OcclusionSettings] = None
    ) -> OcclusionResult:
        """Update occlusion state and return current result.

        Args:
            source_id: Source identifier.
            source_pos: Current source position.
            listener_pos: Current listener position.
            dt: Time delta since last update (seconds).
            settings: Optional per-source settings.

        Returns:
            Current interpolated occlusion result.
        """
        state = self.get_or_create_state(source_id)

        if dt <= 0.0:
            return state.current_result

        state.time_since_update += dt

        # Check if we need a new occlusion detection
        if state.time_since_update >= state.update_interval:
            state.time_since_update = 0.0
            state.target_result = self._detector.detect(source_pos, listener_pos, settings)

        # Interpolate towards target
        if state.interpolation_time > 0.0:
            alpha = 1.0 - math.exp(-dt / state.interpolation_time)
        else:
            alpha = 1.0

        current = state.current_result
        target = state.target_result

        # Interpolate numeric values
        new_factor = current.occlusion_factor + alpha * (target.occlusion_factor - current.occlusion_factor)
        new_low_pass = current.low_pass_frequency + alpha * (target.low_pass_frequency - current.low_pass_frequency)
        new_volume = current.volume_reduction_db + alpha * (target.volume_reduction_db - current.volume_reduction_db)

        state.current_result = OcclusionResult(
            occlusion_type=target.occlusion_type,
            occlusion_factor=new_factor,
            low_pass_frequency=new_low_pass,
            volume_reduction_db=new_volume,
            blocked_rays=target.blocked_rays,
            total_rays=target.total_rays,
            average_transmission=target.average_transmission
        )

        return state.current_result

    def get_current_result(self, source_id: int) -> OcclusionResult:
        """Get current occlusion result for a source without updating."""
        state = self._states.get(source_id)
        if state:
            return state.current_result
        return OcclusionResult(
            occlusion_type=OcclusionType.NONE,
            occlusion_factor=0.0,
            low_pass_frequency=20000.0,
            volume_reduction_db=0.0,
            blocked_rays=0,
            total_rays=1
        )


def db_to_linear(db: float) -> float:
    """Convert decibels to linear gain.

    Args:
        db: Value in decibels.

    Returns:
        Linear gain value.
    """
    return math.pow(10.0, db / 20.0)


def linear_to_db(linear: float) -> float:
    """Convert linear gain to decibels.

    Args:
        linear: Linear gain value.

    Returns:
        Value in decibels.
    """
    if linear <= 0.0:
        return -100.0
    return 20.0 * math.log10(linear)
