"""
Debug Systems - Comprehensive debugging and visualization tools for the AI Game Engine.

This module provides a complete suite of debug tools including:
- Debug draw primitives (lines, spheres, boxes, arrows, text, planes)
- Screen overlays with filtering and categories
- Debug cameras (free-fly, orbit, debug switching)
- Gameplay debugging (AI visualization, nav mesh, trigger volumes)
- Physics debugging (collision shapes, contact points, raycasts)
- Render debugging (wireframe, bounding boxes, LOD, overdraw)
- In-game debug console with commands
- Debug menu system with categories
- Variable watch window with breakpoints

Usage:
    from engine.tooling.debug import DebugDraw, DebugConsole, DebugMenu

    # Draw debug primitives
    debug = DebugDraw.get_instance()
    debug.draw_line(start, end, color=DebugColor.RED)

    # Use decorators
    @debug_draw(category=DebugCategory.PHYSICS)
    def visualize_physics():
        ...

    @cheat(name="god_mode")
    def enable_god_mode():
        ...
"""

from .debug_draw import (
    DebugDraw,
    DebugCategory,
    DebugColor,
    DepthTestMode,
    DrawPrimitive,
    debug_draw,
)

from .debug_overlays import (
    DebugOverlay,
    OverlayManager,
    OverlayPosition,
    OverlayVisibility,
)

from .debug_camera import (
    DebugCamera,
    FreeFlyCamera,
    OrbitCamera,
    DebugCameraController,
    CameraMode,
)

from .gameplay_debug import (
    GameplayDebugger,
    AIVisualization,
    NavMeshDisplay,
    TriggerVolumeVisualizer,
)

from .physics_debug import (
    PhysicsDebugger,
    CollisionShapeVisualizer,
    ContactPointDisplay,
    RaycastVisualizer,
)

from .render_debug import (
    RenderDebugger,
    WireframeMode,
    BoundingBoxDisplay,
    LODVisualization,
    OverdrawHeatmap,
)

from .debug_console import (
    DebugConsole,
    ConsoleCommand,
    cheat,
)

from .debug_menu import (
    DebugMenu,
    MenuCategory,
    MenuItem,
    MenuToggle,
    MenuSlider,
    MenuAction,
)

from .watch_variables import (
    WatchWindow,
    WatchVariable,
    Breakpoint,
    ConditionalWatch,
    VariableTracker,
)

__all__ = [
    # Debug Draw
    "DebugDraw",
    "DebugCategory",
    "DebugColor",
    "DepthTestMode",
    "DrawPrimitive",
    "debug_draw",
    # Overlays
    "DebugOverlay",
    "OverlayManager",
    "OverlayPosition",
    "OverlayVisibility",
    # Debug Camera
    "DebugCamera",
    "FreeFlyCamera",
    "OrbitCamera",
    "DebugCameraController",
    "CameraMode",
    # Gameplay Debug
    "GameplayDebugger",
    "AIVisualization",
    "NavMeshDisplay",
    "TriggerVolumeVisualizer",
    # Physics Debug
    "PhysicsDebugger",
    "CollisionShapeVisualizer",
    "ContactPointDisplay",
    "RaycastVisualizer",
    # Render Debug
    "RenderDebugger",
    "WireframeMode",
    "BoundingBoxDisplay",
    "LODVisualization",
    "OverdrawHeatmap",
    # Console
    "DebugConsole",
    "ConsoleCommand",
    "cheat",
    # Menu
    "DebugMenu",
    "MenuCategory",
    "MenuItem",
    "MenuToggle",
    "MenuSlider",
    "MenuAction",
    # Watch
    "WatchWindow",
    "WatchVariable",
    "Breakpoint",
    "ConditionalWatch",
    "VariableTracker",
]
