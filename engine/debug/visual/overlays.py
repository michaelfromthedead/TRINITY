"""
Debug overlays for visualizing engine subsystems.

Provides a system for enabling/disabling debug overlays for
different engine subsystems (physics, navigation, rendering,
AI, audio, network). Each overlay type has its own render
callback that can be registered.

Usage:
    from engine.debug.visual import DebugOverlay, OverlayType

    # Enable physics overlay
    DebugOverlay.enable(OverlayType.PHYSICS)

    # Register custom render callback
    DebugOverlay.register_callback(OverlayType.AI, render_ai_debug)

    # Check if overlay is enabled
    if DebugOverlay.is_enabled(OverlayType.NAVIGATION):
        render_navmesh()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, Flag, auto
from typing import Callable, Dict, List, Optional, Set, Any


class OverlayType(Flag):
    """
    Types of debug overlays available.

    Can be combined using bitwise OR for enabling multiple overlays:
        DebugOverlay.enable(OverlayType.PHYSICS | OverlayType.AI)
    """
    NONE = 0
    PHYSICS = auto()
    NAVIGATION = auto()
    RENDERING = auto()
    AI = auto()
    AUDIO = auto()
    NETWORK = auto()
    ANIMATION = auto()
    PARTICLES = auto()
    CULLING = auto()
    LOD = auto()
    STREAMING = auto()
    MEMORY = auto()
    PERFORMANCE = auto()
    CUSTOM = auto()

    # Convenience combinations
    ALL = (PHYSICS | NAVIGATION | RENDERING | AI | AUDIO | NETWORK |
           ANIMATION | PARTICLES | CULLING | LOD | STREAMING |
           MEMORY | PERFORMANCE | CUSTOM)
    GAMEPLAY = PHYSICS | AI | ANIMATION
    GRAPHICS = RENDERING | PARTICLES | CULLING | LOD


# Render callback type: (world, camera, delta_time) -> None
OverlayCallback = Callable[[Any, Any, float], None]


@dataclass(slots=True)
class OverlayConfig:
    """
    Configuration for a specific overlay type.

    Attributes:
        enabled: Whether the overlay is currently enabled
        opacity: Overlay opacity (0.0 - 1.0)
        callbacks: List of render callbacks for this overlay
        priority: Render priority (higher = rendered last)
        category: Display category name for UI
        description: Description of what the overlay shows
    """
    enabled: bool = False
    opacity: float = 1.0
    callbacks: List[OverlayCallback] = field(default_factory=list)
    priority: int = 0
    category: str = "General"
    description: str = ""


@dataclass(slots=True)
class PhysicsOverlaySettings:
    """Settings specific to physics debug overlay."""
    show_collision_shapes: bool = True
    show_contact_points: bool = True
    show_joints: bool = True
    show_raycasts: bool = True
    show_velocities: bool = False
    show_center_of_mass: bool = False
    show_inertia_tensor: bool = False
    show_sleep_state: bool = False
    contact_point_size: float = 5.0
    velocity_scale: float = 0.1


@dataclass(slots=True)
class NavigationOverlaySettings:
    """Settings specific to navigation debug overlay."""
    show_navmesh: bool = True
    show_paths: bool = True
    show_off_mesh_links: bool = True
    show_agents: bool = True
    show_obstacles: bool = True
    show_regions: bool = False
    navmesh_opacity: float = 0.3
    path_thickness: float = 2.0


@dataclass(slots=True)
class AIOverlaySettings:
    """Settings specific to AI debug overlay."""
    show_perception_cones: bool = True
    show_behavior_tree: bool = False
    show_eqs_results: bool = False
    show_blackboard: bool = False
    show_active_goals: bool = True
    show_debug_strings: bool = True
    perception_cone_opacity: float = 0.2


@dataclass(slots=True)
class RenderingOverlaySettings:
    """Settings specific to rendering debug overlay."""
    show_bounds: bool = True
    show_wireframe: bool = False
    show_normals: bool = False
    show_tangents: bool = False
    show_uv_seams: bool = False
    show_vertex_colors: bool = False
    normal_scale: float = 0.1
    bounds_color_static: tuple = (0.0, 1.0, 0.0, 0.3)
    bounds_color_dynamic: tuple = (1.0, 1.0, 0.0, 0.3)


@dataclass(slots=True)
class AudioOverlaySettings:
    """Settings specific to audio debug overlay."""
    show_sound_positions: bool = True
    show_attenuation_spheres: bool = True
    show_active_voices: bool = True
    show_listener_position: bool = True
    show_reverb_zones: bool = False
    attenuation_sphere_opacity: float = 0.2


@dataclass(slots=True)
class NetworkOverlaySettings:
    """Settings specific to network debug overlay."""
    show_replication_status: bool = True
    show_ownership: bool = True
    show_relevancy: bool = False
    show_bandwidth: bool = True
    show_packet_flow: bool = False
    show_prediction_errors: bool = False


class DebugOverlay:
    """
    Static class for managing debug overlays.

    Provides enable/disable control for different overlay types,
    registration of render callbacks, and per-overlay configuration.

    Usage:
        # Enable overlays
        DebugOverlay.enable(OverlayType.PHYSICS)
        DebugOverlay.toggle(OverlayType.AI)

        # Register callback
        DebugOverlay.register_callback(OverlayType.PHYSICS, render_physics)

        # Render all enabled overlays
        DebugOverlay.render(world, camera, dt)
    """

    _configs: Dict[OverlayType, OverlayConfig] = {}
    _global_enabled: bool = True
    _global_opacity: float = 1.0

    # Per-overlay-type settings
    _physics_settings: PhysicsOverlaySettings = PhysicsOverlaySettings()
    _navigation_settings: NavigationOverlaySettings = NavigationOverlaySettings()
    _ai_settings: AIOverlaySettings = AIOverlaySettings()
    _rendering_settings: RenderingOverlaySettings = RenderingOverlaySettings()
    _audio_settings: AudioOverlaySettings = AudioOverlaySettings()
    _network_settings: NetworkOverlaySettings = NetworkOverlaySettings()

    @classmethod
    def _ensure_config(cls, overlay_type: OverlayType) -> OverlayConfig:
        """Ensure configuration exists for an overlay type."""
        if overlay_type not in cls._configs:
            # Set up default descriptions
            descriptions = {
                OverlayType.PHYSICS: "Physics collision shapes, contacts, and forces",
                OverlayType.NAVIGATION: "NavMesh, paths, and navigation agents",
                OverlayType.RENDERING: "Render bounds, wireframes, and buffers",
                OverlayType.AI: "Perception, behavior trees, and blackboard",
                OverlayType.AUDIO: "Sound sources and attenuation zones",
                OverlayType.NETWORK: "Replication status and network state",
                OverlayType.ANIMATION: "Skeleton, bones, and animation state",
                OverlayType.PARTICLES: "Particle emitters and bounds",
                OverlayType.CULLING: "Frustum culling and occlusion",
                OverlayType.LOD: "Level of detail transitions",
                OverlayType.STREAMING: "Asset streaming state",
                OverlayType.MEMORY: "Memory usage visualization",
                OverlayType.PERFORMANCE: "Performance metrics overlay",
                OverlayType.CUSTOM: "Custom debug visualization",
            }
            categories = {
                OverlayType.PHYSICS: "Simulation",
                OverlayType.NAVIGATION: "Simulation",
                OverlayType.AI: "Simulation",
                OverlayType.RENDERING: "Graphics",
                OverlayType.PARTICLES: "Graphics",
                OverlayType.CULLING: "Graphics",
                OverlayType.LOD: "Graphics",
                OverlayType.AUDIO: "Audio",
                OverlayType.NETWORK: "Network",
                OverlayType.ANIMATION: "Animation",
                OverlayType.STREAMING: "Resources",
                OverlayType.MEMORY: "System",
                OverlayType.PERFORMANCE: "System",
                OverlayType.CUSTOM: "Custom",
            }
            cls._configs[overlay_type] = OverlayConfig(
                description=descriptions.get(overlay_type, ""),
                category=categories.get(overlay_type, "General")
            )
        return cls._configs[overlay_type]

    @classmethod
    def set_global_enabled(cls, enabled: bool) -> None:
        """
        Enable or disable all overlays globally.

        Args:
            enabled: If False, no overlays are rendered
        """
        cls._global_enabled = enabled

    @classmethod
    def is_global_enabled(cls) -> bool:
        """Return whether overlays are globally enabled."""
        return cls._global_enabled

    @classmethod
    def set_global_opacity(cls, opacity: float) -> None:
        """
        Set global opacity multiplier for all overlays.

        Args:
            opacity: Opacity value (0.0 - 1.0)
        """
        if not 0.0 <= opacity <= 1.0:
            raise ValueError(f"Opacity must be between 0.0 and 1.0, got {opacity}")
        cls._global_opacity = opacity

    @classmethod
    def get_global_opacity(cls) -> float:
        """Return global opacity multiplier."""
        return cls._global_opacity

    @classmethod
    def enable(cls, overlay_type: OverlayType) -> None:
        """
        Enable an overlay type.

        Args:
            overlay_type: The overlay type(s) to enable
        """
        # Handle Flag combinations
        for member in OverlayType:
            if member in overlay_type and member != OverlayType.NONE:
                config = cls._ensure_config(member)
                config.enabled = True

    @classmethod
    def disable(cls, overlay_type: OverlayType) -> None:
        """
        Disable an overlay type.

        Args:
            overlay_type: The overlay type(s) to disable
        """
        for member in OverlayType:
            if member in overlay_type and member != OverlayType.NONE:
                config = cls._ensure_config(member)
                config.enabled = False

    @classmethod
    def toggle(cls, overlay_type: OverlayType) -> bool:
        """
        Toggle an overlay type on/off.

        Args:
            overlay_type: The overlay type to toggle

        Returns:
            New enabled state
        """
        config = cls._ensure_config(overlay_type)
        config.enabled = not config.enabled
        return config.enabled

    @classmethod
    def is_enabled(cls, overlay_type: OverlayType) -> bool:
        """
        Check if an overlay type is enabled.

        Args:
            overlay_type: The overlay type to check

        Returns:
            True if enabled and global overlays are enabled
        """
        if not cls._global_enabled:
            return False
        config = cls._ensure_config(overlay_type)
        return config.enabled

    @classmethod
    def get_enabled_overlays(cls) -> Set[OverlayType]:
        """
        Get all currently enabled overlay types.

        Returns:
            Set of enabled overlay types
        """
        if not cls._global_enabled:
            return set()
        return {
            overlay_type
            for overlay_type, config in cls._configs.items()
            if config.enabled
        }

    @classmethod
    def set_opacity(cls, overlay_type: OverlayType, opacity: float) -> None:
        """
        Set opacity for a specific overlay type.

        Args:
            overlay_type: The overlay type
            opacity: Opacity value (0.0 - 1.0)
        """
        if not 0.0 <= opacity <= 1.0:
            raise ValueError(f"Opacity must be between 0.0 and 1.0, got {opacity}")
        config = cls._ensure_config(overlay_type)
        config.opacity = opacity

    @classmethod
    def get_opacity(cls, overlay_type: OverlayType) -> float:
        """
        Get effective opacity for an overlay type.

        Returns:
            Overlay opacity * global opacity
        """
        config = cls._ensure_config(overlay_type)
        return config.opacity * cls._global_opacity

    @classmethod
    def set_priority(cls, overlay_type: OverlayType, priority: int) -> None:
        """
        Set render priority for an overlay type.

        Higher priority overlays are rendered last (on top).

        Args:
            overlay_type: The overlay type
            priority: Priority value
        """
        config = cls._ensure_config(overlay_type)
        config.priority = priority

    @classmethod
    def register_callback(
        cls,
        overlay_type: OverlayType,
        callback: OverlayCallback
    ) -> None:
        """
        Register a render callback for an overlay type.

        Callbacks are called when the overlay is rendered:
            callback(world, camera, delta_time)

        Args:
            overlay_type: The overlay type
            callback: Render callback function
        """
        config = cls._ensure_config(overlay_type)
        if callback not in config.callbacks:
            config.callbacks.append(callback)

    @classmethod
    def unregister_callback(
        cls,
        overlay_type: OverlayType,
        callback: OverlayCallback
    ) -> bool:
        """
        Unregister a render callback.

        Args:
            overlay_type: The overlay type
            callback: Callback to remove

        Returns:
            True if callback was found and removed
        """
        config = cls._ensure_config(overlay_type)
        if callback in config.callbacks:
            config.callbacks.remove(callback)
            return True
        return False

    @classmethod
    def clear_callbacks(cls, overlay_type: Optional[OverlayType] = None) -> None:
        """
        Clear all callbacks for an overlay type, or all overlays.

        Args:
            overlay_type: Specific overlay, or None for all
        """
        if overlay_type is None:
            for config in cls._configs.values():
                config.callbacks.clear()
        else:
            config = cls._ensure_config(overlay_type)
            config.callbacks.clear()

    @classmethod
    def render(cls, world: Any, camera: Any, delta_time: float) -> None:
        """
        Render all enabled overlays.

        Calls all registered callbacks for enabled overlays,
        sorted by priority.

        Args:
            world: World/scene context
            camera: Active camera
            delta_time: Frame delta time
        """
        if not cls._global_enabled:
            return

        # Get enabled overlays sorted by priority
        enabled = [
            (overlay_type, config)
            for overlay_type, config in cls._configs.items()
            if config.enabled
        ]
        enabled.sort(key=lambda x: x[1].priority)

        # Call callbacks
        for overlay_type, config in enabled:
            for callback in config.callbacks:
                try:
                    callback(world, camera, delta_time)
                except Exception as e:
                    # Log error but continue with other callbacks
                    # In production, this would use the engine's logging system
                    print(f"Error in overlay callback for {overlay_type}: {e}")

    @classmethod
    def get_physics_settings(cls) -> PhysicsOverlaySettings:
        """Get physics overlay settings."""
        return cls._physics_settings

    @classmethod
    def get_navigation_settings(cls) -> NavigationOverlaySettings:
        """Get navigation overlay settings."""
        return cls._navigation_settings

    @classmethod
    def get_ai_settings(cls) -> AIOverlaySettings:
        """Get AI overlay settings."""
        return cls._ai_settings

    @classmethod
    def get_rendering_settings(cls) -> RenderingOverlaySettings:
        """Get rendering overlay settings."""
        return cls._rendering_settings

    @classmethod
    def get_audio_settings(cls) -> AudioOverlaySettings:
        """Get audio overlay settings."""
        return cls._audio_settings

    @classmethod
    def get_network_settings(cls) -> NetworkOverlaySettings:
        """Get network overlay settings."""
        return cls._network_settings

    @classmethod
    def enable_all(cls) -> None:
        """Enable all overlay types."""
        cls.enable(OverlayType.ALL)

    @classmethod
    def disable_all(cls) -> None:
        """Disable all overlay types."""
        for config in cls._configs.values():
            config.enabled = False

    @classmethod
    def reset(cls) -> None:
        """Reset all overlay state to defaults."""
        cls._configs.clear()
        cls._global_enabled = True
        cls._global_opacity = 1.0
        cls._physics_settings = PhysicsOverlaySettings()
        cls._navigation_settings = NavigationOverlaySettings()
        cls._ai_settings = AIOverlaySettings()
        cls._rendering_settings = RenderingOverlaySettings()
        cls._audio_settings = AudioOverlaySettings()
        cls._network_settings = NetworkOverlaySettings()

    @classmethod
    def get_overlay_info(cls, overlay_type: OverlayType) -> Dict[str, Any]:
        """
        Get information about an overlay type.

        Returns:
            Dictionary with overlay metadata
        """
        config = cls._ensure_config(overlay_type)
        return {
            "type": overlay_type.name,
            "enabled": config.enabled,
            "opacity": config.opacity,
            "priority": config.priority,
            "category": config.category,
            "description": config.description,
            "callback_count": len(config.callbacks)
        }

    @classmethod
    def get_all_overlay_info(cls) -> List[Dict[str, Any]]:
        """
        Get information about all overlay types.

        Returns:
            List of dictionaries with overlay metadata
        """
        result = []
        for member in OverlayType:
            if member not in (OverlayType.NONE, OverlayType.ALL,
                            OverlayType.GAMEPLAY, OverlayType.GRAPHICS):
                result.append(cls.get_overlay_info(member))
        return result


# Module-level exports
__all__ = [
    'OverlayType',
    'OverlayCallback',
    'OverlayConfig',
    'PhysicsOverlaySettings',
    'NavigationOverlaySettings',
    'AIOverlaySettings',
    'RenderingOverlaySettings',
    'AudioOverlaySettings',
    'NetworkOverlaySettings',
    'DebugOverlay',
]
