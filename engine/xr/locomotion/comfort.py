"""
Comfort settings and vignette system for XR locomotion.

Provides motion sickness mitigation through vignette effects, snap turn,
tunneling, and other comfort features.

Components:
    ComfortVignette     - Dynamic vignette effect for comfort
    XRComfortSettings   - Global comfort preferences resource
    ComfortPreset       - Pre-configured comfort presets

Decorators:
    @xr_comfort         - Configure comfort features for a component

Based on XR_CONTEXT.md specifications:
    - Comfort options: Vignette, Snap turn, Tunneling, Stable horizon
    - Vignette settings: enabled, intensity (0-1)
    - Snap turn settings: enabled, angle (15-90)
    - Teleport settings: fade_enabled, fade_duration (0-0.5)
    - Seated mode support
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Callable, Optional

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry
from engine.xr.config import XR_CONFIG
from engine.xr.utils.markers import Tracked, Range, Observable, Transient, Immutable

if TYPE_CHECKING:
    pass


# =============================================================================
# ENUMS
# =============================================================================


class ComfortLevel(Enum):
    """Pre-defined comfort level presets."""

    NONE = "none"  # No comfort features (for VR veterans)
    LOW = "low"  # Minimal comfort features
    MEDIUM = "medium"  # Balanced comfort
    HIGH = "high"  # Maximum comfort features
    CUSTOM = "custom"  # User-defined settings


class VignetteShape(Enum):
    """Shape of the comfort vignette."""

    CIRCULAR = "circular"  # Standard circular vignette
    ELLIPTICAL = "elliptical"  # Wider horizontal for peripheral vision
    RECTANGULAR = "rectangular"  # Rectangular tunnel effect


class VignetteTrigger(Enum):
    """What triggers vignette activation."""

    VELOCITY = "velocity"  # Linear movement speed
    ROTATION = "rotation"  # Turning/rotation speed
    BOTH = "both"  # Either velocity or rotation
    MANUAL = "manual"  # Manually controlled


class TunnelingMode(Enum):
    """Tunneling effect mode for extreme comfort."""

    DISABLED = "disabled"
    MILD = "mild"  # Subtle tunnel effect
    MODERATE = "moderate"  # Noticeable tunnel
    STRONG = "strong"  # Aggressive tunneling


class PlayMode(Enum):
    """Player physical play mode."""

    STANDING = "standing"  # Room-scale standing
    SEATED = "seated"  # Seated gameplay
    ROOMSCALE = "roomscale"  # Full room-scale tracking


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class VignetteState:
    """Current state of the comfort vignette."""

    intensity: float = 0.0  # Current intensity 0-1
    target_intensity: float = 0.0  # Target intensity for smoothing
    is_active: bool = False
    triggered_by: VignetteTrigger = VignetteTrigger.VELOCITY


@dataclass
class ComfortMetrics:
    """Metrics tracked for adaptive comfort."""

    cumulative_rotation: float = 0.0  # Total rotation since last reset
    cumulative_velocity: float = 0.0  # Total velocity since last reset
    time_in_motion: float = 0.0  # Seconds spent moving
    session_time: float = 0.0  # Total session time
    comfort_events: int = 0  # Number of times comfort triggered


# =============================================================================
# COMFORT VIGNETTE COMPONENT
# =============================================================================


@dataclass
class ComfortVignette:
    """
    Dynamic vignette effect for motion comfort.

    Applies a darkening vignette around the edges of vision during
    movement to reduce motion sickness.

    Attributes:
        enabled: Whether vignette is active
        intensity: Base vignette intensity (0-1)
        shape: Vignette shape
        trigger: What triggers vignette
        velocity_threshold: Velocity to start vignette
        angular_velocity_threshold: Angular velocity threshold (deg/s)
        inner_radius: Inner radius of vignette (0-1)
        outer_radius: Outer radius of vignette (0-1)
        fade_in_speed: Speed to fade in vignette
        fade_out_speed: Speed to fade out vignette
        color: Vignette color (RGB)
    """

    # Enable/disable
    enabled: Annotated[bool, Tracked] = True

    # Intensity
    base_intensity: Annotated[float, Tracked, Range(0.0, 1.0)] = XR_CONFIG.locomotion.VIGNETTE_INTENSITY_DEFAULT
    current_intensity: Annotated[float, Tracked, Range(0.0, 1.0)] = 0.0
    max_intensity: Annotated[float, Tracked, Range(0.0, 1.0)] = XR_CONFIG.locomotion.VIGNETTE_INTENSITY_MAX

    # Shape and appearance
    shape: Annotated[VignetteShape, Tracked] = VignetteShape.CIRCULAR
    inner_radius: Annotated[float, Tracked, Range(0.0, 1.0)] = XR_CONFIG.locomotion.VIGNETTE_RADIUS_INNER
    outer_radius: Annotated[float, Tracked, Range(0.0, 1.0)] = XR_CONFIG.locomotion.VIGNETTE_RADIUS_OUTER
    feather: Annotated[float, Tracked, Range(0.0, 0.5)] = 0.2
    color: Annotated[tuple[float, float, float], Tracked] = (0.0, 0.0, 0.0)

    # Trigger settings
    trigger: Annotated[VignetteTrigger, Tracked] = VignetteTrigger.BOTH
    velocity_threshold: Annotated[float, Tracked, Range(0.0, 5.0)] = 0.5
    angular_velocity_threshold: Annotated[float, Tracked, Range(0.0, 180.0)] = XR_CONFIG.locomotion.ANGULAR_VELOCITY_LOW_THRESHOLD_DPS

    # Animation
    fade_in_speed: Annotated[float, Tracked, Range(1.0, 20.0)] = 8.0
    fade_out_speed: Annotated[float, Tracked, Range(1.0, 20.0)] = 4.0

    # Adaptive intensity
    adaptive_enabled: Annotated[bool, Tracked] = True
    adaptive_scale: Annotated[float, Tracked, Range(0.0, 2.0)] = 1.0

    # Internal state
    _state: Annotated[VignetteState, Transient] = field(default_factory=VignetteState)
    _velocity_contribution: Annotated[float, Transient] = 0.0
    _angular_contribution: Annotated[float, Transient] = 0.0

    def update(
        self,
        linear_velocity: float,
        angular_velocity: float,
        delta_time: float,
    ) -> float:
        """
        Update vignette based on current movement.

        Args:
            linear_velocity: Current linear speed (m/s)
            angular_velocity: Current angular speed (deg/s)
            delta_time: Time since last frame

        Returns:
            Current vignette intensity
        """
        if not self.enabled:
            self.current_intensity = 0.0
            return 0.0

        target = 0.0
        triggered = False
        trigger_type = VignetteTrigger.MANUAL

        # Check velocity trigger
        if self.trigger in (VignetteTrigger.VELOCITY, VignetteTrigger.BOTH):
            if linear_velocity > self.velocity_threshold:
                velocity_factor = min(
                    (linear_velocity - self.velocity_threshold) / 2.0, 1.0
                )
                self._velocity_contribution = velocity_factor * self.base_intensity
                target = max(target, self._velocity_contribution)
                triggered = True
                trigger_type = VignetteTrigger.VELOCITY
            else:
                self._velocity_contribution = 0.0

        # Check angular velocity trigger
        if self.trigger in (VignetteTrigger.ROTATION, VignetteTrigger.BOTH):
            if angular_velocity > self.angular_velocity_threshold:
                angular_factor = min(
                    (angular_velocity - self.angular_velocity_threshold) / 60.0, 1.0
                )
                self._angular_contribution = angular_factor * self.base_intensity
                target = max(target, self._angular_contribution)
                triggered = True
                if trigger_type == VignetteTrigger.VELOCITY:
                    trigger_type = VignetteTrigger.BOTH
                else:
                    trigger_type = VignetteTrigger.ROTATION
            else:
                self._angular_contribution = 0.0

        # Apply adaptive scaling
        if self.adaptive_enabled:
            target *= self.adaptive_scale

        # Clamp to max intensity
        target = min(target, self.max_intensity)

        # Update state
        self._state.target_intensity = target
        self._state.is_active = triggered
        self._state.triggered_by = trigger_type

        # Smooth transition
        if target > self.current_intensity:
            self.current_intensity = min(
                self.current_intensity + self.fade_in_speed * delta_time,
                target,
            )
        else:
            self.current_intensity = max(
                self.current_intensity - self.fade_out_speed * delta_time,
                target,
            )

        return self.current_intensity

    def get_shader_params(self) -> dict[str, Any]:
        """
        Get parameters for vignette shader.

        Returns:
            Dictionary of shader parameters
        """
        return {
            "intensity": self.current_intensity,
            "inner_radius": self.inner_radius,
            "outer_radius": self.outer_radius,
            "feather": self.feather,
            "color": self.color,
            "shape": self.shape.value,
        }

    def set_intensity_override(self, intensity: float) -> None:
        """
        Manually set vignette intensity (for manual trigger mode).

        Args:
            intensity: Intensity value 0-1
        """
        if self.trigger == VignetteTrigger.MANUAL:
            self._state.target_intensity = min(intensity, self.max_intensity)
            self._state.is_active = intensity > 0


# =============================================================================
# XR COMFORT SETTINGS RESOURCE
# =============================================================================


@dataclass
class XRComfortSettings:
    """
    Global XR comfort preferences resource.

    Singleton resource containing all user comfort preferences.

    Attributes:
        comfort_level: Overall comfort preset
        snap_turn_enabled: Use snap turn instead of smooth
        snap_turn_angle: Degrees per snap turn (15-90)
        smooth_turn_speed: Speed for smooth turn (deg/s)
        vignette_enabled: Enable comfort vignette
        vignette_intensity: Base vignette intensity
        teleport_fade_enabled: Fade during teleport
        teleport_fade_duration: Fade duration (0-0.5s)
        seated_mode: Playing seated
        seated_height_offset: Height offset for seated play
    """

    # Comfort level preset
    comfort_level: Annotated[ComfortLevel, Tracked, Observable] = ComfortLevel.MEDIUM

    # Turn settings
    snap_turn_enabled: Annotated[bool, Tracked] = True
    snap_turn_angle: Annotated[float, Tracked, Range(15.0, 90.0)] = XR_CONFIG.locomotion.SNAP_TURN_ANGLE_DEGREES
    smooth_turn_speed: Annotated[float, Tracked, Range(30.0, 180.0)] = XR_CONFIG.locomotion.SMOOTH_TURN_SPEED_DPS

    # Vignette settings
    vignette_enabled: Annotated[bool, Tracked] = True
    vignette_intensity: Annotated[float, Tracked, Range(0.0, 1.0)] = XR_CONFIG.locomotion.VIGNETTE_INTENSITY_DEFAULT
    vignette_on_turn: Annotated[bool, Tracked] = True
    vignette_on_move: Annotated[bool, Tracked] = True

    # Teleport settings
    teleport_fade_enabled: Annotated[bool, Tracked] = True
    teleport_fade_duration: Annotated[float, Tracked, Range(0.0, 0.5)] = XR_CONFIG.locomotion.TELEPORT_FADE_DURATION_SECONDS

    # Tunneling
    tunneling_mode: Annotated[TunnelingMode, Tracked] = TunnelingMode.DISABLED

    # Stable horizon
    stable_horizon_enabled: Annotated[bool, Tracked] = False
    horizon_lock_roll: Annotated[bool, Tracked] = False
    horizon_lock_pitch: Annotated[bool, Tracked] = False

    # Seated mode
    seated_mode: Annotated[bool, Tracked] = False
    seated_height_offset: Annotated[float, Tracked, Range(-1.0, 1.0)] = 0.0
    play_mode: Annotated[PlayMode, Tracked] = PlayMode.STANDING

    # Movement settings
    smooth_locomotion_enabled: Annotated[bool, Tracked] = True
    teleport_enabled: Annotated[bool, Tracked] = True
    climbing_enabled: Annotated[bool, Tracked] = True

    # Speed reduction for comfort
    movement_speed_scale: Annotated[float, Tracked, Range(0.25, 1.0)] = 1.0

    # Controller dead zones
    thumbstick_dead_zone: Annotated[float, Tracked, Range(0.0, 0.5)] = 0.1

    # Metrics tracking
    track_comfort_metrics: Annotated[bool, Tracked] = True
    _metrics: Annotated[ComfortMetrics, Transient] = field(default_factory=ComfortMetrics)

    def apply_preset(self, level: ComfortLevel) -> None:
        """
        Apply a comfort preset.

        Args:
            level: Comfort level to apply
        """
        self.comfort_level = level

        if level == ComfortLevel.NONE:
            self.snap_turn_enabled = False
            self.vignette_enabled = False
            self.teleport_fade_enabled = False
            self.tunneling_mode = TunnelingMode.DISABLED
            self.stable_horizon_enabled = False
            self.movement_speed_scale = 1.0

        elif level == ComfortLevel.LOW:
            self.snap_turn_enabled = False
            self.vignette_enabled = True
            self.vignette_intensity = 0.3
            self.teleport_fade_enabled = True
            self.teleport_fade_duration = 0.05
            self.tunneling_mode = TunnelingMode.DISABLED
            self.stable_horizon_enabled = False
            self.movement_speed_scale = 1.0

        elif level == ComfortLevel.MEDIUM:
            self.snap_turn_enabled = True
            self.snap_turn_angle = 45.0
            self.vignette_enabled = True
            self.vignette_intensity = 0.5
            self.teleport_fade_enabled = True
            self.teleport_fade_duration = 0.1
            self.tunneling_mode = TunnelingMode.MILD
            self.stable_horizon_enabled = False
            self.movement_speed_scale = 0.8
            self.smooth_locomotion_enabled = True  # Enable smooth locomotion at medium

        elif level == ComfortLevel.HIGH:
            self.snap_turn_enabled = True
            self.snap_turn_angle = 30.0
            self.vignette_enabled = True
            self.vignette_intensity = 0.7
            self.vignette_on_turn = True
            self.vignette_on_move = True
            self.teleport_fade_enabled = True
            self.teleport_fade_duration = 0.2
            self.tunneling_mode = TunnelingMode.MODERATE
            self.stable_horizon_enabled = True
            self.horizon_lock_roll = True
            self.movement_speed_scale = 0.6
            self.smooth_locomotion_enabled = False  # Teleport only

    def get_effective_turn_type(self) -> str:
        """Get the effective turn type based on settings."""
        return "snap" if self.snap_turn_enabled else "smooth"

    def get_effective_turn_speed(self) -> float:
        """Get the effective turn speed/angle."""
        if self.snap_turn_enabled:
            return self.snap_turn_angle
        return self.smooth_turn_speed

    def update_metrics(
        self,
        rotation_delta: float,
        velocity: float,
        delta_time: float,
        comfort_triggered: bool,
    ) -> None:
        """
        Update comfort tracking metrics.

        Args:
            rotation_delta: Rotation this frame (degrees)
            velocity: Linear velocity this frame
            delta_time: Time since last frame
            comfort_triggered: Whether comfort effect was triggered
        """
        if not self.track_comfort_metrics:
            return

        self._metrics.cumulative_rotation += abs(rotation_delta)
        self._metrics.cumulative_velocity += velocity * delta_time
        if velocity > 0.1 or abs(rotation_delta) > 1.0:
            self._metrics.time_in_motion += delta_time
        self._metrics.session_time += delta_time
        if comfort_triggered:
            self._metrics.comfort_events += 1

    def get_metrics(self) -> ComfortMetrics:
        """Get current comfort metrics."""
        return self._metrics

    def reset_metrics(self) -> None:
        """Reset comfort tracking metrics."""
        self._metrics = ComfortMetrics()


# =============================================================================
# COMFORT PRESET CONFIGURATION
# =============================================================================


@dataclass
class ComfortPreset:
    """
    Pre-configured comfort preset for quick application.

    Use to define game-specific comfort presets.
    """

    name: str
    description: str

    # Turn
    snap_turn: bool = True
    snap_angle: float = 45.0
    smooth_speed: float = 90.0

    # Vignette
    vignette: bool = True
    vignette_intensity: float = 0.5

    # Teleport
    teleport_fade: bool = True
    fade_duration: float = 0.1

    # Extra
    tunneling: TunnelingMode = TunnelingMode.DISABLED
    stable_horizon: bool = False
    speed_scale: float = 1.0

    def apply_to(self, settings: XRComfortSettings) -> None:
        """
        Apply this preset to comfort settings.

        Args:
            settings: Settings to modify
        """
        settings.snap_turn_enabled = self.snap_turn
        settings.snap_turn_angle = self.snap_angle
        settings.smooth_turn_speed = self.smooth_speed
        settings.vignette_enabled = self.vignette
        settings.vignette_intensity = self.vignette_intensity
        settings.teleport_fade_enabled = self.teleport_fade
        settings.teleport_fade_duration = self.fade_duration
        settings.tunneling_mode = self.tunneling
        settings.stable_horizon_enabled = self.stable_horizon
        settings.movement_speed_scale = self.speed_scale
        settings.comfort_level = ComfortLevel.CUSTOM


# Pre-defined presets
COMFORT_PRESETS: dict[str, ComfortPreset] = {
    "veteran": ComfortPreset(
        name="Veteran",
        description="For experienced VR users - no comfort features",
        snap_turn=False,
        vignette=False,
        teleport_fade=False,
        tunneling=TunnelingMode.DISABLED,
        stable_horizon=False,
        speed_scale=1.0,
    ),
    "intermediate": ComfortPreset(
        name="Intermediate",
        description="Balanced comfort for regular VR users",
        snap_turn=True,
        snap_angle=45.0,
        vignette=True,
        vignette_intensity=0.4,
        teleport_fade=True,
        fade_duration=0.1,
        tunneling=TunnelingMode.MILD,
        stable_horizon=False,
        speed_scale=0.9,
    ),
    "comfortable": ComfortPreset(
        name="Comfortable",
        description="Enhanced comfort for sensitive users",
        snap_turn=True,
        snap_angle=30.0,
        vignette=True,
        vignette_intensity=0.6,
        teleport_fade=True,
        fade_duration=0.15,
        tunneling=TunnelingMode.MODERATE,
        stable_horizon=True,
        speed_scale=0.7,
    ),
    "maximum": ComfortPreset(
        name="Maximum Comfort",
        description="Maximum comfort - teleport only movement",
        snap_turn=True,
        snap_angle=30.0,
        vignette=True,
        vignette_intensity=0.8,
        teleport_fade=True,
        fade_duration=0.25,
        tunneling=TunnelingMode.STRONG,
        stable_horizon=True,
        speed_scale=0.5,
    ),
    "seated": ComfortPreset(
        name="Seated Play",
        description="Optimized for seated gameplay",
        snap_turn=True,
        snap_angle=45.0,
        vignette=True,
        vignette_intensity=0.5,
        teleport_fade=True,
        fade_duration=0.1,
        tunneling=TunnelingMode.MILD,
        stable_horizon=True,
        speed_scale=0.8,
    ),
}


def get_preset(name: str) -> Optional[ComfortPreset]:
    """Get a comfort preset by name."""
    return COMFORT_PRESETS.get(name)


def list_presets() -> list[str]:
    """List available comfort preset names."""
    return list(COMFORT_PRESETS.keys())


# =============================================================================
# DECORATOR: @xr_comfort
# =============================================================================


def _validate_xr_comfort(
    comfort_type: str = "vignette", **_: Any
) -> None:
    """Validate xr_comfort decorator parameters."""
    valid_types = {"vignette", "locomotion", "turn", "teleport", "general"}
    if comfort_type not in valid_types:
        raise ValueError(
            f"@xr_comfort: 'comfort_type' must be one of {valid_types}, "
            f"got '{comfort_type}'"
        )


def _xr_comfort_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for xr_comfort decorator."""
    comfort_type = params.get("comfort_type", "general")
    settings = params.get("settings", {})
    return [
        Step(Op.TAG, {"key": "xr_comfort", "value": True}),
        Step(Op.TAG, {"key": "comfort_type", "value": comfort_type}),
        Step(Op.TAG, {"key": "comfort_settings", "value": settings}),
        Step(Op.REGISTER, {"registry": "xr"}),
    ]


def _after_xr_comfort(target: Any, params: dict[str, Any]) -> Any:
    """Post-processing for xr_comfort decorator."""
    target._xr_comfort = True
    target._comfort_type = params.get("comfort_type", "general")
    target._comfort_settings = params.get("settings", {})
    return None


xr_comfort = make_decorator(
    name="xr_comfort",
    steps=_xr_comfort_steps,
    doc="Configure XR comfort features for a component.",
    validate=_validate_xr_comfort,
    after_steps=_after_xr_comfort,
)


# =============================================================================
# REGISTRY REGISTRATION
# =============================================================================

_REGISTRY_ENTRIES: list[tuple[str, Any, tuple[str, ...]]] = [
    ("xr_comfort", xr_comfort, ("class", "function")),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.GAMEPLAY,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.GAMEPLAY].append(_spec)


# =============================================================================
# COMFORT SYSTEM MANAGER
# =============================================================================


class ComfortManager:
    """
    Manages comfort features across the XR session.

    Coordinates vignette, turn settings, and other comfort features
    based on user preferences.
    """

    def __init__(self, settings: XRComfortSettings):
        """
        Initialize comfort manager.

        Args:
            settings: Global comfort settings resource
        """
        self.settings = settings
        self.vignette = ComfortVignette(
            enabled=settings.vignette_enabled,
            base_intensity=settings.vignette_intensity,
        )

    def update(
        self,
        linear_velocity: float,
        angular_velocity: float,
        delta_time: float,
    ) -> dict[str, Any]:
        """
        Update all comfort systems.

        Args:
            linear_velocity: Current linear speed
            angular_velocity: Current angular speed (deg/s)
            delta_time: Time since last frame

        Returns:
            Dictionary of comfort state data
        """
        # Update vignette
        vignette_active = False
        if self.settings.vignette_enabled:
            # Determine if triggers apply
            check_velocity = self.settings.vignette_on_move
            check_angular = self.settings.vignette_on_turn

            trigger = VignetteTrigger.MANUAL
            if check_velocity and check_angular:
                trigger = VignetteTrigger.BOTH
            elif check_velocity:
                trigger = VignetteTrigger.VELOCITY
            elif check_angular:
                trigger = VignetteTrigger.ROTATION

            self.vignette.trigger = trigger
            self.vignette.base_intensity = self.settings.vignette_intensity
            intensity = self.vignette.update(
                linear_velocity, angular_velocity, delta_time
            )
            vignette_active = intensity > 0.01

        # Update metrics
        self.settings.update_metrics(
            angular_velocity * delta_time,
            linear_velocity,
            delta_time,
            vignette_active,
        )

        return {
            "vignette_intensity": self.vignette.current_intensity,
            "vignette_active": vignette_active,
            "vignette_params": self.vignette.get_shader_params(),
            "snap_turn_enabled": self.settings.snap_turn_enabled,
            "snap_turn_angle": self.settings.snap_turn_angle,
            "smooth_turn_speed": self.settings.smooth_turn_speed,
            "tunneling_mode": self.settings.tunneling_mode.value,
            "speed_scale": self.settings.movement_speed_scale,
        }

    def apply_preset(self, preset_name: str) -> bool:
        """
        Apply a named comfort preset.

        Args:
            preset_name: Name of preset to apply

        Returns:
            True if preset was found and applied
        """
        preset = get_preset(preset_name)
        if preset is None:
            return False

        preset.apply_to(self.settings)
        self.vignette.enabled = self.settings.vignette_enabled
        self.vignette.base_intensity = self.settings.vignette_intensity
        return True


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ComfortLevel",
    "VignetteShape",
    "VignetteTrigger",
    "TunnelingMode",
    "PlayMode",
    # Data classes
    "VignetteState",
    "ComfortMetrics",
    # Components
    "ComfortVignette",
    "XRComfortSettings",
    "ComfortPreset",
    # Manager
    "ComfortManager",
    # Presets
    "COMFORT_PRESETS",
    "get_preset",
    "list_presets",
    # Decorator
    "xr_comfort",
    # Type markers
    "Tracked",
    "Range",
    "Observable",
    "Transient",
    "Immutable",
]
