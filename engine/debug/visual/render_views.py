"""
Debug render view modes for visualizing different render buffers.

Provides control over the active render view mode, allowing
visualization of wireframe, unlit, base color, normals, roughness,
metallic, AO, overdraw, and shader complexity.

Usage:
    from engine.debug.visual import RenderViewMode, set_view_mode, get_view_mode

    # Switch to wireframe mode
    set_view_mode(RenderViewMode.WIREFRAME)

    # Get current mode
    mode = get_view_mode()

    # Use render view manager for more control
    RenderViewManager.set_mode(RenderViewMode.NORMALS)
    RenderViewManager.set_overlay_strength(0.5)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple


class RenderViewMode(Enum):
    """
    Available debug render view modes.

    Each mode visualizes a different aspect of the rendering pipeline.
    """
    NORMAL = auto()             # Standard lit rendering
    WIREFRAME = auto()          # Mesh edges only
    UNLIT = auto()              # No lighting, base color only
    BASE_COLOR = auto()         # Albedo texture only
    NORMALS = auto()            # World-space normals (RGB)
    ROUGHNESS = auto()          # Roughness channel
    METALLIC = auto()           # Metallic channel
    AO = auto()                 # Ambient occlusion
    OVERDRAW = auto()           # Pixel overdraw heatmap
    SHADER_COMPLEXITY = auto()  # Shader instruction count
    EMISSIVE = auto()           # Emissive channel
    SPECULAR = auto()           # Specular contribution
    DIFFUSE = auto()            # Diffuse contribution
    DEPTH = auto()              # Depth buffer visualization
    STENCIL = auto()            # Stencil buffer visualization
    MOTION_VECTORS = auto()     # Motion vector field
    LIGHTMAP = auto()           # Lightmap UVs or values
    UV_CHECKER = auto()         # UV checker pattern
    VERTEX_COLORS = auto()      # Per-vertex colors
    LOD_COLORING = auto()       # LOD level visualization
    MIPMAP_LEVEL = auto()       # Mipmap level visualization
    TANGENT_SPACE = auto()      # Tangent vectors
    BITANGENT_SPACE = auto()    # Bitangent vectors
    LIGHT_ONLY = auto()         # Light contribution only
    REFLECTIONS = auto()        # Reflection probes/SSR
    SHADOWS = auto()            # Shadow maps visualization


@dataclass(slots=True)
class RenderViewConfig:
    """
    Configuration for a render view mode.

    Attributes:
        mode: The render view mode
        name: Display name
        description: Description of what is visualized
        category: Category for grouping in UI
        requires_gbuffer: Whether mode requires deferred GBuffer
        overlay_strength: How strong the overlay effect is (0-1)
        custom_shader: Optional custom shader name
    """
    mode: RenderViewMode
    name: str
    description: str
    category: str = "General"
    requires_gbuffer: bool = False
    overlay_strength: float = 1.0
    custom_shader: Optional[str] = None


# Callback type for mode change notifications
ModeChangeCallback = Callable[[RenderViewMode, RenderViewMode], None]


class RenderViewManager:
    """
    Manages the active render view mode and related settings.

    Provides a central point for controlling debug render views,
    with support for mode change callbacks and per-mode configuration.
    """

    _current_mode: RenderViewMode = RenderViewMode.NORMAL
    _previous_mode: RenderViewMode = RenderViewMode.NORMAL
    _overlay_strength: float = 1.0
    _enabled: bool = True
    _callbacks: List[ModeChangeCallback] = []
    _configs: Dict[RenderViewMode, RenderViewConfig] = {}

    # Initialize default configurations
    _default_configs: List[RenderViewConfig] = [
        RenderViewConfig(
            RenderViewMode.NORMAL,
            "Normal",
            "Standard lit rendering",
            "Standard"
        ),
        RenderViewConfig(
            RenderViewMode.WIREFRAME,
            "Wireframe",
            "Mesh edges only, no fill",
            "Geometry"
        ),
        RenderViewConfig(
            RenderViewMode.UNLIT,
            "Unlit",
            "Base color without lighting",
            "Lighting"
        ),
        RenderViewConfig(
            RenderViewMode.BASE_COLOR,
            "Base Color",
            "Albedo/diffuse texture only",
            "Material",
            requires_gbuffer=True
        ),
        RenderViewConfig(
            RenderViewMode.NORMALS,
            "Normals",
            "World-space surface normals (RGB)",
            "Material",
            requires_gbuffer=True
        ),
        RenderViewConfig(
            RenderViewMode.ROUGHNESS,
            "Roughness",
            "Surface roughness value (grayscale)",
            "Material",
            requires_gbuffer=True
        ),
        RenderViewConfig(
            RenderViewMode.METALLIC,
            "Metallic",
            "Metallic value (grayscale)",
            "Material",
            requires_gbuffer=True
        ),
        RenderViewConfig(
            RenderViewMode.AO,
            "Ambient Occlusion",
            "SSAO and baked AO",
            "Material",
            requires_gbuffer=True
        ),
        RenderViewConfig(
            RenderViewMode.OVERDRAW,
            "Overdraw",
            "Pixel overdraw heatmap (red = high)",
            "Performance"
        ),
        RenderViewConfig(
            RenderViewMode.SHADER_COMPLEXITY,
            "Shader Complexity",
            "Shader instruction count heatmap",
            "Performance"
        ),
        RenderViewConfig(
            RenderViewMode.EMISSIVE,
            "Emissive",
            "Emissive/self-illumination channel",
            "Material",
            requires_gbuffer=True
        ),
        RenderViewConfig(
            RenderViewMode.SPECULAR,
            "Specular",
            "Specular lighting contribution",
            "Lighting",
            requires_gbuffer=True
        ),
        RenderViewConfig(
            RenderViewMode.DIFFUSE,
            "Diffuse",
            "Diffuse lighting contribution",
            "Lighting",
            requires_gbuffer=True
        ),
        RenderViewConfig(
            RenderViewMode.DEPTH,
            "Depth Buffer",
            "Depth values (near=white, far=black)",
            "Buffers"
        ),
        RenderViewConfig(
            RenderViewMode.STENCIL,
            "Stencil Buffer",
            "Stencil buffer values",
            "Buffers"
        ),
        RenderViewConfig(
            RenderViewMode.MOTION_VECTORS,
            "Motion Vectors",
            "Per-pixel motion for TAA/motion blur",
            "Buffers",
            requires_gbuffer=True
        ),
        RenderViewConfig(
            RenderViewMode.LIGHTMAP,
            "Lightmap",
            "Baked lightmap UVs or values",
            "Lighting"
        ),
        RenderViewConfig(
            RenderViewMode.UV_CHECKER,
            "UV Checker",
            "Checker pattern on UV coordinates",
            "Geometry"
        ),
        RenderViewConfig(
            RenderViewMode.VERTEX_COLORS,
            "Vertex Colors",
            "Per-vertex color attribute",
            "Geometry"
        ),
        RenderViewConfig(
            RenderViewMode.LOD_COLORING,
            "LOD Levels",
            "Color-coded LOD level visualization",
            "Performance"
        ),
        RenderViewConfig(
            RenderViewMode.MIPMAP_LEVEL,
            "Mipmap Level",
            "Texture mipmap level visualization",
            "Performance"
        ),
        RenderViewConfig(
            RenderViewMode.TANGENT_SPACE,
            "Tangent Vectors",
            "Per-vertex tangent direction",
            "Geometry",
            requires_gbuffer=True
        ),
        RenderViewConfig(
            RenderViewMode.BITANGENT_SPACE,
            "Bitangent Vectors",
            "Per-vertex bitangent direction",
            "Geometry",
            requires_gbuffer=True
        ),
        RenderViewConfig(
            RenderViewMode.LIGHT_ONLY,
            "Light Only",
            "Lighting contribution (no albedo)",
            "Lighting",
            requires_gbuffer=True
        ),
        RenderViewConfig(
            RenderViewMode.REFLECTIONS,
            "Reflections",
            "Reflection probes and SSR",
            "Lighting",
            requires_gbuffer=True
        ),
        RenderViewConfig(
            RenderViewMode.SHADOWS,
            "Shadow Maps",
            "Shadow map visualization",
            "Lighting"
        ),
    ]

    @classmethod
    def _ensure_configs(cls) -> None:
        """Initialize default configurations if not already done."""
        if not cls._configs:
            for config in cls._default_configs:
                cls._configs[config.mode] = config

    @classmethod
    def set_mode(cls, mode: RenderViewMode) -> None:
        """
        Set the active render view mode.

        Args:
            mode: The render view mode to activate
        """
        if not cls._enabled and mode != RenderViewMode.NORMAL:
            return

        cls._ensure_configs()
        old_mode = cls._current_mode
        if old_mode == mode:
            return

        cls._previous_mode = old_mode
        cls._current_mode = mode

        # Notify callbacks
        for callback in cls._callbacks:
            try:
                callback(old_mode, mode)
            except Exception as e:
                print(f"Error in render view mode callback: {e}")

    @classmethod
    def get_mode(cls) -> RenderViewMode:
        """
        Get the current render view mode.

        Returns:
            The active render view mode
        """
        return cls._current_mode

    @classmethod
    def get_previous_mode(cls) -> RenderViewMode:
        """
        Get the previous render view mode.

        Returns:
            The previously active render view mode
        """
        return cls._previous_mode

    @classmethod
    def toggle_mode(cls, mode: RenderViewMode) -> bool:
        """
        Toggle between a mode and normal view.

        Args:
            mode: The mode to toggle

        Returns:
            True if mode is now active, False if switched to NORMAL
        """
        if cls._current_mode == mode:
            cls.set_mode(RenderViewMode.NORMAL)
            return False
        else:
            cls.set_mode(mode)
            return True

    @classmethod
    def cycle_mode(cls, forward: bool = True) -> RenderViewMode:
        """
        Cycle to the next or previous render view mode.

        Args:
            forward: If True, cycle forward; if False, cycle backward

        Returns:
            The new active mode
        """
        modes = list(RenderViewMode)
        current_idx = modes.index(cls._current_mode)

        if forward:
            new_idx = (current_idx + 1) % len(modes)
        else:
            new_idx = (current_idx - 1) % len(modes)

        cls.set_mode(modes[new_idx])
        return cls._current_mode

    @classmethod
    def set_enabled(cls, enabled: bool) -> None:
        """
        Enable or disable render view mode changes.

        When disabled, only NORMAL mode is allowed.

        Args:
            enabled: If False, forces NORMAL mode
        """
        cls._enabled = enabled
        if not enabled and cls._current_mode != RenderViewMode.NORMAL:
            cls.set_mode(RenderViewMode.NORMAL)

    @classmethod
    def is_enabled(cls) -> bool:
        """Return whether render view changes are enabled."""
        return cls._enabled

    @classmethod
    def set_overlay_strength(cls, strength: float) -> None:
        """
        Set the overlay strength for debug views.

        Args:
            strength: Strength value (0.0 - 1.0)
        """
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"Strength must be between 0.0 and 1.0, got {strength}")
        cls._overlay_strength = strength

    @classmethod
    def get_overlay_strength(cls) -> float:
        """Return the current overlay strength."""
        return cls._overlay_strength

    @classmethod
    def register_callback(cls, callback: ModeChangeCallback) -> None:
        """
        Register a callback for mode changes.

        Callback signature: (old_mode, new_mode) -> None

        Args:
            callback: Callback function
        """
        if callback not in cls._callbacks:
            cls._callbacks.append(callback)

    @classmethod
    def unregister_callback(cls, callback: ModeChangeCallback) -> bool:
        """
        Unregister a mode change callback.

        Args:
            callback: Callback to remove

        Returns:
            True if callback was found and removed
        """
        if callback in cls._callbacks:
            cls._callbacks.remove(callback)
            return True
        return False

    @classmethod
    def get_config(cls, mode: RenderViewMode) -> RenderViewConfig:
        """
        Get configuration for a render view mode.

        Args:
            mode: The render view mode

        Returns:
            Configuration for the mode
        """
        cls._ensure_configs()
        return cls._configs[mode]

    @classmethod
    def get_all_configs(cls) -> List[RenderViewConfig]:
        """
        Get all render view mode configurations.

        Returns:
            List of all configurations
        """
        cls._ensure_configs()
        return list(cls._configs.values())

    @classmethod
    def get_modes_by_category(cls, category: str) -> List[RenderViewMode]:
        """
        Get all modes in a category.

        Args:
            category: Category name

        Returns:
            List of modes in the category
        """
        cls._ensure_configs()
        return [
            config.mode
            for config in cls._configs.values()
            if config.category == category
        ]

    @classmethod
    def get_categories(cls) -> List[str]:
        """
        Get all unique categories.

        Returns:
            List of category names
        """
        cls._ensure_configs()
        return sorted(set(config.category for config in cls._configs.values()))

    @classmethod
    def is_gbuffer_required(cls, mode: Optional[RenderViewMode] = None) -> bool:
        """
        Check if current or specified mode requires GBuffer.

        Args:
            mode: Mode to check, or None for current mode

        Returns:
            True if GBuffer is required
        """
        cls._ensure_configs()
        check_mode = mode if mode is not None else cls._current_mode
        return cls._configs[check_mode].requires_gbuffer

    @classmethod
    def reset(cls) -> None:
        """Reset to default state."""
        cls._current_mode = RenderViewMode.NORMAL
        cls._previous_mode = RenderViewMode.NORMAL
        cls._overlay_strength = 1.0
        cls._enabled = True
        cls._callbacks.clear()

    @classmethod
    def get_mode_info(cls, mode: Optional[RenderViewMode] = None) -> Dict[str, Any]:
        """
        Get information about a render view mode.

        Args:
            mode: Mode to get info for, or None for current mode

        Returns:
            Dictionary with mode information
        """
        cls._ensure_configs()
        check_mode = mode if mode is not None else cls._current_mode
        config = cls._configs[check_mode]
        return {
            "mode": config.mode.name,
            "name": config.name,
            "description": config.description,
            "category": config.category,
            "requires_gbuffer": config.requires_gbuffer,
            "overlay_strength": cls._overlay_strength,
            "is_current": check_mode == cls._current_mode
        }


# Module-level convenience functions
def set_view_mode(mode: RenderViewMode) -> None:
    """
    Set the active render view mode.

    Args:
        mode: The render view mode to activate
    """
    RenderViewManager.set_mode(mode)


def get_view_mode() -> RenderViewMode:
    """
    Get the current render view mode.

    Returns:
        The active render view mode
    """
    return RenderViewManager.get_mode()


def toggle_view_mode(mode: RenderViewMode) -> bool:
    """
    Toggle between a mode and normal view.

    Args:
        mode: The mode to toggle

    Returns:
        True if mode is now active
    """
    return RenderViewManager.toggle_mode(mode)


def cycle_view_mode(forward: bool = True) -> RenderViewMode:
    """
    Cycle to the next or previous render view mode.

    Args:
        forward: If True, cycle forward

    Returns:
        The new active mode
    """
    return RenderViewManager.cycle_mode(forward)


# Module-level exports
__all__ = [
    'RenderViewMode',
    'RenderViewConfig',
    'RenderViewManager',
    'ModeChangeCallback',
    'set_view_mode',
    'get_view_mode',
    'toggle_view_mode',
    'cycle_view_mode',
]
