"""
Visual debugging subsystem for the game engine.

Provides debug drawing primitives, overlays, render view modes,
and interactive gizmos for development and debugging.

Usage:
    from engine.debug.visual import DebugDraw, Color, DebugOverlay, OverlayType
    from engine.debug.visual import RenderViewMode, set_view_mode
    from engine.debug.visual import TransformGizmo, BoundsGizmo, LightGizmo

    # Draw debug primitives
    DebugDraw.line(start=(0, 0, 0), end=(10, 0, 0), color=Color.RED)
    DebugDraw.sphere(center=(5, 5, 5), radius=1.0, color=Color.GREEN, duration=2.0)

    # Enable overlays
    DebugOverlay.enable(OverlayType.PHYSICS)

    # Switch render view mode
    set_view_mode(RenderViewMode.WIREFRAME)

    # Use gizmos
    gizmo = TransformGizmo()
    gizmo.set_target((0, 0, 0))
    gizmo.render(camera)
"""

from .config import (
    DEBUG_DRAW_CONFIG,
    GIZMO_CONFIG,
    OVERLAY_CONFIG,
    DebugDrawConfig,
    GizmoConfig,
    OverlayConfig,
)
from .draw import (
    Color,
    DebugDraw,
    DebugDrawBatch,
    DrawOptions,
    DrawPrimitive,
    DrawPrimitiveType,
    Quat,
    Vec3,
)
from .gizmos import (
    BaseGizmo,
    BoundsGizmo,
    CameraGizmo,
    GizmoAxis,
    GizmoChangeCallback,
    GizmoSpace,
    GizmoState,
    GizmoStyle,
    GizmoType,
    LightGizmo,
    TransformGizmo,
)
from .overlays import (
    AIOverlaySettings,
    AudioOverlaySettings,
    DebugOverlay,
    NavigationOverlaySettings,
    NetworkOverlaySettings,
    OverlayCallback,
    OverlayConfig,
    OverlayType,
    PhysicsOverlaySettings,
    RenderingOverlaySettings,
)
from .render_views import (
    ModeChangeCallback,
    RenderViewConfig,
    RenderViewManager,
    RenderViewMode,
    cycle_view_mode,
    get_view_mode,
    set_view_mode,
    toggle_view_mode,
)

__all__ = [
    # config.py
    'DebugDrawConfig',
    'GizmoConfig',
    'OverlayConfig',
    'DEBUG_DRAW_CONFIG',
    'GIZMO_CONFIG',
    'OVERLAY_CONFIG',
    # draw.py
    'Color',
    'DrawOptions',
    'DrawPrimitiveType',
    'DrawPrimitive',
    'DebugDrawBatch',
    'DebugDraw',
    'Vec3',
    'Quat',
    # overlays.py
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
    # render_views.py
    'RenderViewMode',
    'RenderViewConfig',
    'RenderViewManager',
    'ModeChangeCallback',
    'set_view_mode',
    'get_view_mode',
    'toggle_view_mode',
    'cycle_view_mode',
    # gizmos.py
    'GizmoType',
    'GizmoSpace',
    'GizmoAxis',
    'GizmoStyle',
    'GizmoState',
    'GizmoChangeCallback',
    'BaseGizmo',
    'TransformGizmo',
    'BoundsGizmo',
    'LightGizmo',
    'CameraGizmo',
]
