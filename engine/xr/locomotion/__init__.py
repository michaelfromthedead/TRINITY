"""
XR Locomotion Module

Provides locomotion systems for XR (VR/AR/MR) environments including:
- Teleport locomotion with arc visualization
- Smooth continuous locomotion with snap/smooth turn
- Climbing locomotion
- Comfort features (vignette, tunneling, etc.)

Based on XR_CONTEXT.md specifications:
    - Teleportation: Instant, Fade, Dash, Arc visualizer
    - Smooth locomotion: Thumbstick, Arm swing, Head-directed, Hand-directed
    - Physical: Room-scale, Climbing mechanics
    - Comfort options: Vignette, Snap turn, Tunneling, Stable horizon

Example usage:
    from engine.xr.locomotion import (
        TeleportLocomotion,
        SmoothLocomotion,
        ClimbingLocomotion,
        XRComfortSettings,
        ComfortManager,
    )

    # Create teleport locomotion
    teleport = TeleportLocomotion(
        style=TeleportStyle.FADE,
        fade_duration=0.1,
        max_distance=10.0,
    )

    # Create smooth locomotion
    smooth = SmoothLocomotion(
        move_speed=3.0,
        turn_type=TurnType.SNAP,
        snap_angle=45.0,
        vignette_enabled=True,
    )

    # Create comfort settings
    comfort_settings = XRComfortSettings()
    comfort_settings.apply_preset(ComfortLevel.MEDIUM)

    # Use comfort manager
    manager = ComfortManager(comfort_settings)
    state = manager.update(velocity, angular_velocity, delta_time)

Decorators:
    @xr_teleport_area  - Mark as valid teleport destination
    @xr_locomotion     - Configure locomotion behavior
    @xr_climbable      - Mark as climbable surface
    @xr_comfort        - Configure comfort features
"""

from __future__ import annotations

# Teleport locomotion
from engine.xr.locomotion.teleport import (
    ArcPoint,
    ArcSegmentType,
    TeleportArcCalculator,
    TeleportLocomotion,
    TeleportLocomotionProvider,
    TeleportResult,
    TeleportState,
    TeleportStyle,
    TeleportTarget,
    xr_teleport_area,
)

# Smooth locomotion
from engine.xr.locomotion.smooth import (
    ArmSwingData,
    MovementInput,
    MovementMode,
    MovementResult,
    MovementState,
    SmoothLocomotion,
    SmoothLocomotionProvider,
    StrafeBehavior,
    TurnSettings,
    TurnType,
    xr_locomotion,
)

# Climbing locomotion
from engine.xr.locomotion.climbing import (
    ClimbableType,
    ClimbableVolume,
    ClimbingInput,
    ClimbingLocomotion,
    ClimbingLocomotionProvider,
    ClimbingMovement,
    ClimbingState,
    GrabHandState,
    GrabPoint,
    GrabState,
    MantleType,
    xr_climbable,
)

# Comfort system
from engine.xr.locomotion.comfort import (
    COMFORT_PRESETS,
    ComfortLevel,
    ComfortManager,
    ComfortMetrics,
    ComfortPreset,
    ComfortVignette,
    PlayMode,
    TunnelingMode,
    VignetteShape,
    VignetteState,
    VignetteTrigger,
    XRComfortSettings,
    get_preset,
    list_presets,
    xr_comfort,
)


# =============================================================================
# ABSTRACT LOCOMOTION PROVIDER
# =============================================================================


class LocomotionProvider:
    """
    Abstract base class for locomotion providers.

    Implement this to create custom locomotion systems that integrate
    multiple locomotion types.
    """

    def __init__(self) -> None:
        """Initialize the locomotion provider."""
        self._teleport: TeleportLocomotion | None = None
        self._smooth: SmoothLocomotion | None = None
        self._climbing: ClimbingLocomotion | None = None
        self._comfort_settings: XRComfortSettings | None = None
        self._comfort_manager: ComfortManager | None = None

    def initialize(
        self,
        teleport: TeleportLocomotion | None = None,
        smooth: SmoothLocomotion | None = None,
        climbing: ClimbingLocomotion | None = None,
        comfort_settings: XRComfortSettings | None = None,
    ) -> None:
        """
        Initialize with locomotion components.

        Args:
            teleport: Teleport locomotion component
            smooth: Smooth locomotion component
            climbing: Climbing locomotion component
            comfort_settings: Comfort settings resource
        """
        self._teleport = teleport
        self._smooth = smooth
        self._climbing = climbing
        self._comfort_settings = comfort_settings

        if comfort_settings:
            self._comfort_manager = ComfortManager(comfort_settings)

    def update(self, delta_time: float) -> None:
        """
        Update all locomotion systems.

        Args:
            delta_time: Time since last frame
        """
        if self._teleport:
            self._teleport.update(delta_time)
        if self._smooth:
            self._smooth.update(delta_time)
        if self._climbing:
            self._climbing.update(delta_time)

    def get_current_vignette_intensity(self) -> float:
        """Get current comfort vignette intensity."""
        if self._comfort_manager:
            return self._comfort_manager.vignette.current_intensity
        if self._smooth:
            return self._smooth.get_current_vignette_intensity()
        return 0.0

    @property
    def teleport(self) -> TeleportLocomotion | None:
        """Get teleport locomotion component."""
        return self._teleport

    @property
    def smooth(self) -> SmoothLocomotion | None:
        """Get smooth locomotion component."""
        return self._smooth

    @property
    def climbing(self) -> ClimbingLocomotion | None:
        """Get climbing locomotion component."""
        return self._climbing

    @property
    def comfort(self) -> XRComfortSettings | None:
        """Get comfort settings."""
        return self._comfort_settings


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_teleport_locomotion(
    style: TeleportStyle = TeleportStyle.FADE,
    fade_duration: float = 0.1,
    max_distance: float = 10.0,
    **kwargs,
) -> TeleportLocomotion:
    """
    Factory function to create teleport locomotion.

    Args:
        style: Teleport transition style
        fade_duration: Duration of fade transition
        max_distance: Maximum teleport distance

    Returns:
        Configured TeleportLocomotion instance
    """
    return TeleportLocomotion(
        style=style,
        fade_duration=fade_duration,
        max_distance=max_distance,
        **kwargs,
    )


def create_smooth_locomotion(
    move_speed: float = 3.0,
    turn_type: TurnType = TurnType.SNAP,
    snap_angle: float = 45.0,
    vignette_enabled: bool = True,
    **kwargs,
) -> SmoothLocomotion:
    """
    Factory function to create smooth locomotion.

    Args:
        move_speed: Forward movement speed
        turn_type: Snap or smooth turning
        snap_angle: Angle for snap turns
        vignette_enabled: Enable comfort vignette

    Returns:
        Configured SmoothLocomotion instance
    """
    return SmoothLocomotion(
        move_speed=move_speed,
        turn_type=turn_type,
        snap_angle=snap_angle,
        vignette_enabled=vignette_enabled,
        **kwargs,
    )


def create_climbing_locomotion(
    stamina_enabled: bool = True,
    mantle_enabled: bool = True,
    **kwargs,
) -> ClimbingLocomotion:
    """
    Factory function to create climbing locomotion.

    Args:
        stamina_enabled: Enable stamina system
        mantle_enabled: Enable mantling

    Returns:
        Configured ClimbingLocomotion instance
    """
    return ClimbingLocomotion(
        stamina_enabled=stamina_enabled,
        mantle_enabled=mantle_enabled,
        **kwargs,
    )


def create_comfort_settings(
    preset: ComfortLevel = ComfortLevel.MEDIUM,
) -> XRComfortSettings:
    """
    Factory function to create comfort settings with preset.

    Args:
        preset: Comfort level preset to apply

    Returns:
        Configured XRComfortSettings instance
    """
    settings = XRComfortSettings()
    settings.apply_preset(preset)
    return settings


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # --- Teleport ---
    "TeleportLocomotion",
    "TeleportLocomotionProvider",
    "TeleportTarget",
    "TeleportArcCalculator",
    "TeleportStyle",
    "TeleportState",
    "TeleportResult",
    "ArcPoint",
    "ArcSegmentType",
    "xr_teleport_area",
    # --- Smooth ---
    "SmoothLocomotion",
    "SmoothLocomotionProvider",
    "TurnSettings",
    "MovementMode",
    "MovementState",
    "MovementInput",
    "MovementResult",
    "TurnType",
    "StrafeBehavior",
    "ArmSwingData",
    "xr_locomotion",
    # --- Climbing ---
    "ClimbingLocomotion",
    "ClimbingLocomotionProvider",
    "ClimbableVolume",
    "GrabPoint",
    "ClimbingState",
    "GrabHandState",
    "ClimbableType",
    "MantleType",
    "GrabState",
    "ClimbingInput",
    "ClimbingMovement",
    "xr_climbable",
    # --- Comfort ---
    "ComfortVignette",
    "XRComfortSettings",
    "ComfortPreset",
    "ComfortManager",
    "ComfortMetrics",
    "ComfortLevel",
    "VignetteShape",
    "VignetteTrigger",
    "TunnelingMode",
    "PlayMode",
    "VignetteState",
    "COMFORT_PRESETS",
    "get_preset",
    "list_presets",
    "xr_comfort",
    # --- Provider ---
    "LocomotionProvider",
    # --- Factory ---
    "create_teleport_locomotion",
    "create_smooth_locomotion",
    "create_climbing_locomotion",
    "create_comfort_settings",
]
