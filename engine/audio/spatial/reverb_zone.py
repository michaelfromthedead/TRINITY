"""Reverb Zones with Blending Support.

Implements reverb zones for environmental audio:
- Volume triggers (enter/exit detection)
- Zone blending for smooth transitions
- Priority system for overlapping zones
- Preset and custom reverb parameters
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from engine.audio.spatial.config import (
    DEFAULT_REVERB_PREDELAY,
    DEFAULT_REVERB_WET_MIX,
    MAX_REVERB_ZONES,
    REVERB_BLEND_TIME,
    REVERB_MAX_ROOM_SIZE,
    REVERB_MAX_RT60,
    REVERB_MIN_ROOM_SIZE,
    REVERB_MIN_RT60,
    ReverbPreset,
)
from engine.core.math.vec import Vec3


@dataclass
class ReverbParameters:
    """Parameters for a reverb effect."""

    room_size: float = 50.0
    """Room size parameter (affects diffusion and density)."""

    rt60: float = 1.5
    """Reverb decay time RT60 in seconds (time for -60dB decay)."""

    damping: float = 0.5
    """High frequency damping (0 = bright, 1 = dark)."""

    diffusion: float = 0.8
    """Diffusion/density of reflections (0 = sparse, 1 = dense)."""

    wet_mix: float = DEFAULT_REVERB_WET_MIX
    """Wet/dry mix (0 = dry only, 1 = wet only)."""

    predelay: float = DEFAULT_REVERB_PREDELAY
    """Pre-delay before reverb starts (seconds)."""

    early_reflections: float = 0.5
    """Early reflections level (0-1)."""

    late_reverb: float = 0.7
    """Late reverb level (0-1)."""

    low_shelf_freq: float = 200.0
    """Low shelf filter frequency (Hz)."""

    low_shelf_gain: float = 0.0
    """Low shelf gain (dB)."""

    high_shelf_freq: float = 4000.0
    """High shelf filter frequency (Hz)."""

    high_shelf_gain: float = -6.0
    """High shelf gain (dB, typically negative for natural decay)."""

    def __post_init__(self) -> None:
        self.room_size = max(REVERB_MIN_ROOM_SIZE, min(REVERB_MAX_ROOM_SIZE, self.room_size))
        self.rt60 = max(REVERB_MIN_RT60, min(REVERB_MAX_RT60, self.rt60))
        self.damping = max(0.0, min(1.0, self.damping))
        self.diffusion = max(0.0, min(1.0, self.diffusion))
        self.wet_mix = max(0.0, min(1.0, self.wet_mix))
        self.predelay = max(0.0, min(0.5, self.predelay))
        self.early_reflections = max(0.0, min(1.0, self.early_reflections))
        self.late_reverb = max(0.0, min(1.0, self.late_reverb))

    def lerp(self, other: ReverbParameters, t: float) -> ReverbParameters:
        """Linearly interpolate between this and another parameter set."""
        t = max(0.0, min(1.0, t))
        return ReverbParameters(
            room_size=self.room_size + t * (other.room_size - self.room_size),
            rt60=self.rt60 + t * (other.rt60 - self.rt60),
            damping=self.damping + t * (other.damping - self.damping),
            diffusion=self.diffusion + t * (other.diffusion - self.diffusion),
            wet_mix=self.wet_mix + t * (other.wet_mix - self.wet_mix),
            predelay=self.predelay + t * (other.predelay - self.predelay),
            early_reflections=self.early_reflections + t * (other.early_reflections - self.early_reflections),
            late_reverb=self.late_reverb + t * (other.late_reverb - self.late_reverb),
            low_shelf_freq=self.low_shelf_freq + t * (other.low_shelf_freq - self.low_shelf_freq),
            low_shelf_gain=self.low_shelf_gain + t * (other.low_shelf_gain - self.low_shelf_gain),
            high_shelf_freq=self.high_shelf_freq + t * (other.high_shelf_freq - self.high_shelf_freq),
            high_shelf_gain=self.high_shelf_gain + t * (other.high_shelf_gain - self.high_shelf_gain),
        )


# Preset reverb parameters
REVERB_PRESET_PARAMS: Dict[ReverbPreset, ReverbParameters] = {
    ReverbPreset.NONE: ReverbParameters(room_size=1.0, rt60=0.0, wet_mix=0.0),
    ReverbPreset.SMALL_ROOM: ReverbParameters(room_size=10.0, rt60=0.4, damping=0.6, wet_mix=0.25, predelay=0.005),
    ReverbPreset.MEDIUM_ROOM: ReverbParameters(room_size=30.0, rt60=0.8, damping=0.5, wet_mix=0.3, predelay=0.015),
    ReverbPreset.LARGE_ROOM: ReverbParameters(room_size=60.0, rt60=1.2, damping=0.4, wet_mix=0.35, predelay=0.025),
    ReverbPreset.BATHROOM: ReverbParameters(room_size=8.0, rt60=0.8, damping=0.2, diffusion=0.9, wet_mix=0.4, predelay=0.002),
    ReverbPreset.CAVE: ReverbParameters(room_size=80.0, rt60=3.0, damping=0.7, diffusion=0.6, wet_mix=0.5, predelay=0.04),
    ReverbPreset.CATHEDRAL: ReverbParameters(room_size=100.0, rt60=5.0, damping=0.3, diffusion=0.8, wet_mix=0.45, predelay=0.05),
    ReverbPreset.ARENA: ReverbParameters(room_size=90.0, rt60=4.0, damping=0.4, diffusion=0.7, wet_mix=0.4, predelay=0.06),
    ReverbPreset.HANGAR: ReverbParameters(room_size=100.0, rt60=6.0, damping=0.5, diffusion=0.5, wet_mix=0.35, predelay=0.07),
    ReverbPreset.OUTDOOR: ReverbParameters(room_size=200.0, rt60=0.3, damping=0.8, diffusion=0.3, wet_mix=0.1, predelay=0.03),
    ReverbPreset.UNDERWATER: ReverbParameters(room_size=50.0, rt60=2.0, damping=0.9, diffusion=0.9, wet_mix=0.6, predelay=0.01,
                                               high_shelf_freq=2000.0, high_shelf_gain=-12.0),
}


def get_preset_parameters(preset: ReverbPreset) -> ReverbParameters:
    """Get reverb parameters for a preset."""
    return REVERB_PRESET_PARAMS.get(preset, REVERB_PRESET_PARAMS[ReverbPreset.MEDIUM_ROOM])


@dataclass
class ReverbZone:
    """A spatial zone with reverb properties."""

    zone_id: int = 0
    """Unique identifier for this zone."""

    name: str = ""
    """Human-readable name."""

    center: Vec3 = field(default_factory=Vec3.zero)
    """Center position of the zone."""

    half_extents: Vec3 = field(default_factory=lambda: Vec3(10.0, 10.0, 10.0))
    """Half-size of the zone bounding box."""

    preset: ReverbPreset = ReverbPreset.MEDIUM_ROOM
    """Reverb preset to use."""

    parameters: Optional[ReverbParameters] = None
    """Custom parameters (overrides preset if set)."""

    fade_distance: float = 2.0
    """Distance over which to fade in/out at zone boundaries."""

    priority: int = 0
    """Priority for overlapping zones (higher = takes precedence)."""

    active: bool = True
    """Whether the zone is active."""

    def __post_init__(self) -> None:
        self.half_extents = Vec3(abs(self.half_extents.x), abs(self.half_extents.y), abs(self.half_extents.z))
        self.fade_distance = max(0.0, self.fade_distance)

    def get_parameters(self) -> ReverbParameters:
        """Get effective reverb parameters."""
        if self.parameters is not None:
            return self.parameters
        return get_preset_parameters(self.preset)

    @property
    def min_corner(self) -> Vec3:
        """Get minimum corner of bounding box."""
        return self.center - self.half_extents

    @property
    def max_corner(self) -> Vec3:
        """Get maximum corner of bounding box."""
        return self.center + self.half_extents

    def contains(self, point: Vec3) -> bool:
        """Check if a point is inside the zone."""
        min_c = self.min_corner
        max_c = self.max_corner
        return (
            min_c.x <= point.x <= max_c.x and
            min_c.y <= point.y <= max_c.y and
            min_c.z <= point.z <= max_c.z
        )

    def get_blend_factor(self, point: Vec3) -> float:
        """Get blend factor (0-1) based on point position.

        Returns 1.0 when fully inside (beyond fade distance from edges),
        interpolates to 0.0 at the edge, and 0.0 outside.
        """
        if not self.active:
            return 0.0

        min_c = self.min_corner
        max_c = self.max_corner

        # Check if outside
        if not (min_c.x <= point.x <= max_c.x and
                min_c.y <= point.y <= max_c.y and
                min_c.z <= point.z <= max_c.z):
            return 0.0

        if self.fade_distance <= 0.0:
            return 1.0

        # Calculate distance to nearest edge
        dist_to_edges = [
            point.x - min_c.x,
            max_c.x - point.x,
            point.y - min_c.y,
            max_c.y - point.y,
            point.z - min_c.z,
            max_c.z - point.z,
        ]

        min_dist = min(dist_to_edges)

        if min_dist >= self.fade_distance:
            return 1.0

        # Smoothstep fade
        t = min_dist / self.fade_distance
        return t * t * (3.0 - 2.0 * t)

    def signed_distance(self, point: Vec3) -> float:
        """Get signed distance to zone (negative if inside)."""
        min_c = self.min_corner
        max_c = self.max_corner

        # Distance to each face
        dx = max(min_c.x - point.x, point.x - max_c.x, 0.0)
        dy = max(min_c.y - point.y, point.y - max_c.y, 0.0)
        dz = max(min_c.z - point.z, point.z - max_c.z, 0.0)

        outside_dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        if outside_dist > 0:
            return outside_dist

        # Inside - return negative of distance to nearest face
        inside_dist = min(
            point.x - min_c.x,
            max_c.x - point.x,
            point.y - min_c.y,
            max_c.y - point.y,
            point.z - min_c.z,
            max_c.z - point.z,
        )
        return -inside_dist


@dataclass
class ReverbZoneState:
    """State for a listener in reverb zones."""

    active_zones: List[Tuple[ReverbZone, float]] = field(default_factory=list)
    """Currently active zones with their blend factors."""

    current_params: ReverbParameters = field(default_factory=ReverbParameters)
    """Current blended reverb parameters."""

    target_params: ReverbParameters = field(default_factory=ReverbParameters)
    """Target reverb parameters (for smooth transitions)."""

    blend_time: float = REVERB_BLEND_TIME
    """Time to blend between parameter changes."""

    blend_progress: float = 1.0
    """Progress of current blend (0-1)."""


class ReverbZoneManager:
    """Manages reverb zones and listener transitions."""

    def __init__(self, blend_time: float = REVERB_BLEND_TIME) -> None:
        self._zones: Dict[int, ReverbZone] = {}
        self._next_id = 1
        self._blend_time = blend_time
        self._listener_states: Dict[int, ReverbZoneState] = {}
        self._default_params = get_preset_parameters(ReverbPreset.NONE)
        self._on_zone_enter: Optional[Callable[[int, int], None]] = None
        self._on_zone_exit: Optional[Callable[[int, int], None]] = None

    @property
    def blend_time(self) -> float:
        """Get default blend time."""
        return self._blend_time

    @blend_time.setter
    def blend_time(self, value: float) -> None:
        self._blend_time = max(0.0, value)

    @property
    def default_parameters(self) -> ReverbParameters:
        """Get default reverb parameters (when outside all zones)."""
        return self._default_params

    @default_parameters.setter
    def default_parameters(self, value: ReverbParameters) -> None:
        self._default_params = value

    def set_callbacks(
        self,
        on_enter: Optional[Callable[[int, int], None]] = None,
        on_exit: Optional[Callable[[int, int], None]] = None
    ) -> None:
        """Set zone enter/exit callbacks.

        Args:
            on_enter: Called with (listener_id, zone_id) when entering.
            on_exit: Called with (listener_id, zone_id) when exiting.
        """
        self._on_zone_enter = on_enter
        self._on_zone_exit = on_exit

    def add_zone(
        self,
        center: Vec3,
        half_extents: Vec3,
        preset: ReverbPreset = ReverbPreset.MEDIUM_ROOM,
        parameters: Optional[ReverbParameters] = None,
        fade_distance: float = 2.0,
        priority: int = 0,
        name: str = ""
    ) -> int:
        """Add a reverb zone.

        Returns:
            Zone ID.
        """
        zone_id = self._next_id
        self._next_id += 1

        zone = ReverbZone(
            zone_id=zone_id,
            name=name or f"Zone_{zone_id}",
            center=center,
            half_extents=half_extents,
            preset=preset,
            parameters=parameters,
            fade_distance=fade_distance,
            priority=priority
        )
        self._zones[zone_id] = zone
        return zone_id

    def remove_zone(self, zone_id: int) -> bool:
        """Remove a reverb zone."""
        if zone_id in self._zones:
            del self._zones[zone_id]
            return True
        return False

    def get_zone(self, zone_id: int) -> Optional[ReverbZone]:
        """Get a zone by ID."""
        return self._zones.get(zone_id)

    def get_zones(self) -> List[ReverbZone]:
        """Get all zones."""
        return list(self._zones.values())

    def set_zone_active(self, zone_id: int, active: bool) -> bool:
        """Enable or disable a zone."""
        zone = self._zones.get(zone_id)
        if zone:
            zone.active = active
            return True
        return False

    def _get_or_create_state(self, listener_id: int) -> ReverbZoneState:
        """Get or create state for a listener."""
        if listener_id not in self._listener_states:
            self._listener_states[listener_id] = ReverbZoneState(
                current_params=self._default_params,
                target_params=self._default_params,
                blend_time=self._blend_time
            )
        return self._listener_states[listener_id]

    def update(self, listener_id: int, listener_pos: Vec3, dt: float) -> ReverbParameters:
        """Update reverb for a listener and return current parameters.

        Args:
            listener_id: Listener identifier.
            listener_pos: Listener world position.
            dt: Time delta (seconds).

        Returns:
            Current blended reverb parameters.
        """
        state = self._get_or_create_state(listener_id)

        # Find active zones and their blend factors
        new_active: List[Tuple[ReverbZone, float]] = []
        for zone in self._zones.values():
            if not zone.active:
                continue

            blend = zone.get_blend_factor(listener_pos)
            if blend > 0.0:
                new_active.append((zone, blend))

        # Sort by priority (higher first)
        new_active.sort(key=lambda x: x[0].priority, reverse=True)

        # Limit to max zones
        new_active = new_active[:MAX_REVERB_ZONES]

        # Check for enter/exit events
        old_zone_ids = {z.zone_id for z, _ in state.active_zones}
        new_zone_ids = {z.zone_id for z, _ in new_active}

        if self._on_zone_enter:
            for zone, _ in new_active:
                if zone.zone_id not in old_zone_ids:
                    self._on_zone_enter(listener_id, zone.zone_id)

        if self._on_zone_exit:
            for zone, _ in state.active_zones:
                if zone.zone_id not in new_zone_ids:
                    self._on_zone_exit(listener_id, zone.zone_id)

        state.active_zones = new_active

        # Calculate target parameters (blend active zones)
        if not new_active:
            state.target_params = self._default_params
        else:
            # Weighted blend of all active zones
            total_weight = sum(blend for _, blend in new_active)
            if total_weight > 0.0:
                blended = get_preset_parameters(ReverbPreset.NONE)
                for zone, blend in new_active:
                    weight = blend / total_weight
                    params = zone.get_parameters()
                    blended = blended.lerp(params, weight)
                state.target_params = blended

        # Smoothly transition to target
        if state.blend_progress < 1.0 or state.current_params != state.target_params:
            if dt > 0.0 and state.blend_time > 0.0:
                state.blend_progress = min(1.0, state.blend_progress + dt / state.blend_time)
            else:
                state.blend_progress = 1.0

            state.current_params = state.current_params.lerp(state.target_params, state.blend_progress)

            # Reset blend progress if target changed
            if state.current_params != state.target_params:
                state.blend_progress = 0.0

        return state.current_params

    def get_current_parameters(self, listener_id: int) -> ReverbParameters:
        """Get current reverb parameters for a listener without updating."""
        state = self._listener_states.get(listener_id)
        return state.current_params if state else self._default_params

    def get_active_zones(self, listener_id: int) -> List[Tuple[int, float]]:
        """Get active zones and their blend factors for a listener.

        Returns:
            List of (zone_id, blend_factor) tuples.
        """
        state = self._listener_states.get(listener_id)
        if not state:
            return []
        return [(zone.zone_id, blend) for zone, blend in state.active_zones]


def create_reverb_zone(
    center: Vec3,
    size: Vec3,
    preset: ReverbPreset = ReverbPreset.MEDIUM_ROOM,
    **kwargs
) -> ReverbZone:
    """Convenience function to create a reverb zone.

    Args:
        center: Zone center position.
        size: Full size of the zone (will be halved for half_extents).
        preset: Reverb preset.
        **kwargs: Additional ReverbZone parameters.

    Returns:
        New ReverbZone instance.
    """
    return ReverbZone(
        center=center,
        half_extents=Vec3(size.x / 2, size.y / 2, size.z / 2),
        preset=preset,
        **kwargs
    )
