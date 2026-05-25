"""
Configuration constants for the Visual Debugging subsystem.

This module centralizes all configurable values that were previously
hardcoded throughout the debug visual implementation. This includes
draw settings, gizmo parameters, and performance limits.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DebugDrawConfig:
    """Configuration for debug draw primitives."""

    # Epsilon for vector normalization to avoid division by zero
    vector_normalize_epsilon: float = 1e-10

    # Maximum number of primitives allowed in batch (0 = unlimited)
    # When exceeded, oldest frame primitives are dropped
    max_primitives: int = 10000

    # Warning threshold - log warning when this count is exceeded
    primitive_warning_threshold: int = 5000

    # Default arrow head size as fraction of arrow length
    default_arrow_head_size: float = 0.15

    # Default coordinate axes arrow head size
    coordinate_axes_head_size: float = 0.15

    # Default line thickness in pixels
    default_line_thickness: float = 1.0

    # Default sphere segments for wireframe rendering
    default_sphere_segments: int = 16

    # Default circle segments for wireframe rendering
    default_circle_segments: int = 32


@dataclass(frozen=True)
class GizmoConfig:
    """Configuration for interactive gizmos."""

    # Epsilon for vector normalization
    vector_normalize_epsilon: float = 1e-10

    # TransformGizmo plane handle settings
    translate_plane_size_ratio: float = 0.3
    translate_plane_offset_ratio: float = 0.4
    translate_plane_thickness: float = 0.01

    # Scale gizmo box sizes as ratio of gizmo size
    scale_handle_box_ratio: float = 0.08
    scale_center_box_multiplier: float = 1.5

    # Hit test radius as ratio of gizmo size
    hit_test_radius_ratio: float = 0.15

    # BoundsGizmo settings
    bounds_axis_size_ratio: float = 0.5
    bounds_label_offset: float = 0.2
    bounds_label_scale: float = 0.8

    # LightGizmo settings
    light_center_sphere_radius: float = 0.1
    light_attenuation_alpha: float = 0.3
    light_attenuation_line_alpha: float = 0.5
    light_arrow_length_ratio: float = 0.5
    light_arrow_head_size: float = 0.15

    # Directional light gizmo settings
    directional_arrow_length: float = 2.0
    directional_arrow_spacing: float = 0.5
    directional_arrow_head_size: float = 0.2

    # CameraGizmo settings
    camera_icon_sphere_radius: float = 0.1
    camera_default_aspect_ratio: float = 16.0 / 9.0


@dataclass(frozen=True)
class OverlayConfig:
    """Configuration for debug overlays."""

    # Default opacity for overlays
    default_opacity: float = 1.0

    # Default render priority (higher = rendered last/on top)
    default_priority: int = 0


# Default configuration instances
DEBUG_DRAW_CONFIG = DebugDrawConfig()
GIZMO_CONFIG = GizmoConfig()
OVERLAY_CONFIG = OverlayConfig()
