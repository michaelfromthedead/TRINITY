"""
Modes - Editor modes for different editing workflows.

Provides:
- Mode system with enter/exit lifecycle
- Select mode for object manipulation
- Paint mode for texture/vertex painting
- Sculpt mode for mesh sculpting
- Placement mode for object placement
- Sequence mode for animation editing
- Mode-specific tools
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

from engine.tooling.editor.app_shell import editor, reloadable


class ModeType(Enum):
    """Types of editor modes."""
    SELECT = auto()
    PAINT = auto()
    SCULPT = auto()
    PLACEMENT = auto()
    SEQUENCE = auto()


@editor(category="Modes")
@reloadable()
class ModeContext:
    """Context information passed to editor modes."""
    __slots__ = ("viewport", "selection", "gizmo_manager", "command_manager",
                 "scene", "delta_time", "mouse_position", "modifier_keys")

    def __init__(self, viewport: Any = None, selection: Any = None,
                 gizmo_manager: Any = None, command_manager: Any = None,
                 scene: Any = None):
        self.viewport = viewport
        self.selection = selection
        self.gizmo_manager = gizmo_manager
        self.command_manager = command_manager
        self.scene = scene
        self.delta_time = 0.0
        self.mouse_position = (0, 0)
        self.modifier_keys = set()


@editor(category="Modes")
@reloadable()
class ModeTool(ABC):
    """Base class for mode-specific tools."""
    __slots__ = ("id", "name", "icon", "tooltip", "enabled", "active",
                 "shortcut", "on_activated", "on_deactivated")

    def __init__(self, id: str, name: str, icon: str = ""):
        self.id = id
        self.name = name
        self.icon = icon
        self.tooltip: str = ""
        self.enabled: bool = True
        self.active: bool = False
        self.shortcut: Optional[str] = None
        self.on_activated: Optional[Callable[[], None]] = None
        self.on_deactivated: Optional[Callable[[], None]] = None

    def activate(self) -> None:
        """Activate this tool."""
        self.active = True
        if self.on_activated:
            self.on_activated()

    def deactivate(self) -> None:
        """Deactivate this tool."""
        self.active = False
        if self.on_deactivated:
            self.on_deactivated()

    @abstractmethod
    def on_mouse_down(self, ctx: ModeContext, x: int, y: int, button: int) -> bool:
        """Handle mouse down. Returns True if handled."""
        pass

    @abstractmethod
    def on_mouse_up(self, ctx: ModeContext, x: int, y: int, button: int) -> bool:
        """Handle mouse up. Returns True if handled."""
        pass

    @abstractmethod
    def on_mouse_move(self, ctx: ModeContext, x: int, y: int) -> bool:
        """Handle mouse move. Returns True if handled."""
        pass

    def on_key_down(self, ctx: ModeContext, key: str) -> bool:
        """Handle key down. Returns True if handled."""
        return False

    def on_key_up(self, ctx: ModeContext, key: str) -> bool:
        """Handle key up. Returns True if handled."""
        return False


@editor(category="Modes")
@reloadable()
class EditorMode(ABC):
    """Base class for editor modes."""
    __slots__ = ("mode_type", "name", "icon", "_tools", "_active_tool",
                 "_context", "on_enter", "on_exit", "enabled")

    def __init__(self, mode_type: ModeType, name: str = ""):
        self.mode_type = mode_type
        self.name = name or mode_type.name.title()
        self.icon: str = ""
        self._tools: dict[str, ModeTool] = {}
        self._active_tool: Optional[ModeTool] = None
        self._context: Optional[ModeContext] = None
        self.on_enter: Optional[Callable[[], None]] = None
        self.on_exit: Optional[Callable[[], None]] = None
        self.enabled: bool = True

    @property
    def tools(self) -> list[ModeTool]:
        """Get all tools for this mode."""
        return list(self._tools.values())

    @property
    def active_tool(self) -> Optional[ModeTool]:
        """Get the active tool."""
        return self._active_tool

    @property
    def context(self) -> Optional[ModeContext]:
        """Get the mode context."""
        return self._context

    def add_tool(self, tool: ModeTool) -> None:
        """Add a tool to this mode."""
        self._tools[tool.id] = tool

    def remove_tool(self, tool_id: str) -> Optional[ModeTool]:
        """Remove a tool from this mode."""
        tool = self._tools.pop(tool_id, None)
        if tool and tool == self._active_tool:
            self._active_tool = None
        return tool

    def get_tool(self, tool_id: str) -> Optional[ModeTool]:
        """Get a tool by ID."""
        return self._tools.get(tool_id)

    def set_active_tool(self, tool_id: str) -> bool:
        """Set the active tool. Returns True if successful."""
        tool = self._tools.get(tool_id)
        if tool and tool.enabled:
            if self._active_tool:
                self._active_tool.deactivate()
            self._active_tool = tool
            tool.activate()
            return True
        return False

    def enter(self, context: ModeContext) -> None:
        """Enter this mode."""
        self._context = context
        self._on_enter()
        if self.on_enter:
            self.on_enter()

    def exit(self) -> None:
        """Exit this mode."""
        self._on_exit()
        if self.on_exit:
            self.on_exit()
        if self._active_tool:
            self._active_tool.deactivate()
        self._context = None

    def update(self, delta_time: float) -> None:
        """Update the mode each frame."""
        if self._context:
            self._context.delta_time = delta_time
            self._on_update(delta_time)

    @abstractmethod
    def _on_enter(self) -> None:
        """Called when entering the mode. Override for setup."""
        pass

    @abstractmethod
    def _on_exit(self) -> None:
        """Called when exiting the mode. Override for cleanup."""
        pass

    def _on_update(self, delta_time: float) -> None:
        """Called each frame. Override for continuous updates."""
        pass

    def on_mouse_down(self, x: int, y: int, button: int) -> bool:
        """Handle mouse down. Returns True if handled."""
        if self._active_tool and self._context:
            return self._active_tool.on_mouse_down(self._context, x, y, button)
        return False

    def on_mouse_up(self, x: int, y: int, button: int) -> bool:
        """Handle mouse up. Returns True if handled."""
        if self._active_tool and self._context:
            return self._active_tool.on_mouse_up(self._context, x, y, button)
        return False

    def on_mouse_move(self, x: int, y: int) -> bool:
        """Handle mouse move. Returns True if handled."""
        if self._context:
            self._context.mouse_position = (x, y)
        if self._active_tool and self._context:
            return self._active_tool.on_mouse_move(self._context, x, y)
        return False

    def on_key_down(self, key: str) -> bool:
        """Handle key down. Returns True if handled."""
        if self._active_tool and self._context:
            return self._active_tool.on_key_down(self._context, key)
        return False

    def on_key_up(self, key: str) -> bool:
        """Handle key up. Returns True if handled."""
        if self._active_tool and self._context:
            return self._active_tool.on_key_up(self._context, key)
        return False


# Concrete tool implementations

@editor(category="Modes")
class SelectTool(ModeTool):
    """Tool for selecting objects."""
    __slots__ = ("_is_dragging", "_drag_start", "_drag_mode")

    def __init__(self):
        super().__init__("select", "Select", "cursor")
        self.tooltip = "Select objects (Q)"
        self.shortcut = "Q"
        self._is_dragging = False
        self._drag_start: Optional[tuple[int, int]] = None
        self._drag_mode: str = "single"  # single, marquee, add, toggle

    def on_mouse_down(self, ctx: ModeContext, x: int, y: int, button: int) -> bool:
        if button == 0:  # Left button
            self._is_dragging = True
            self._drag_start = (x, y)
            # Determine mode based on modifiers
            if "shift" in ctx.modifier_keys:
                self._drag_mode = "add"
            elif "ctrl" in ctx.modifier_keys:
                self._drag_mode = "toggle"
            else:
                self._drag_mode = "single"
            return True
        return False

    def on_mouse_up(self, ctx: ModeContext, x: int, y: int, button: int) -> bool:
        if button == 0 and self._is_dragging:
            self._is_dragging = False
            # Perform selection based on drag distance
            if self._drag_start:
                dx = abs(x - self._drag_start[0])
                dy = abs(y - self._drag_start[1])
                if dx < 5 and dy < 5:
                    # Click selection
                    if ctx.viewport:
                        hit = ctx.viewport.pick_at(x, y)
                        if hit and ctx.selection:
                            if self._drag_mode == "add":
                                ctx.selection.add_to_selection(hit)
                            elif self._drag_mode == "toggle":
                                ctx.selection.toggle_selection(hit)
                            else:
                                ctx.selection.select(hit)
                        elif ctx.selection and self._drag_mode == "single":
                            ctx.selection.clear_selection()
            self._drag_start = None
            return True
        return False

    def on_mouse_move(self, ctx: ModeContext, x: int, y: int) -> bool:
        return False


@editor(category="Modes")
class MoveTool(ModeTool):
    """Tool for moving objects."""
    __slots__ = ("_is_moving", "_start_position")

    def __init__(self):
        super().__init__("move", "Move", "move")
        self.tooltip = "Move objects (W)"
        self.shortcut = "W"
        self._is_moving = False
        self._start_position: Optional[tuple[float, float, float]] = None

    def on_mouse_down(self, ctx: ModeContext, x: int, y: int, button: int) -> bool:
        if button == 0 and ctx.gizmo_manager:
            # Try to start gizmo interaction
            axis = ctx.gizmo_manager.hit_test(x, y, ctx.viewport.camera.position if ctx.viewport else (0, 0, 0))
            if axis:
                self._is_moving = True
                ctx.gizmo_manager.begin_transform(axis, x, y)
                return True
        return False

    def on_mouse_up(self, ctx: ModeContext, x: int, y: int, button: int) -> bool:
        if button == 0 and self._is_moving:
            self._is_moving = False
            if ctx.gizmo_manager:
                ctx.gizmo_manager.end_transform()
            return True
        return False

    def on_mouse_move(self, ctx: ModeContext, x: int, y: int) -> bool:
        if self._is_moving and ctx.gizmo_manager:
            delta = ctx.gizmo_manager.update_transform(
                x, y, ctx.viewport.camera.position if ctx.viewport else (0, 0, 0)
            )
            if delta:
                # Apply transform to selection
                pass
            return True
        return False


@editor(category="Modes")
class RotateTool(ModeTool):
    """Tool for rotating objects."""
    __slots__ = ("_is_rotating",)

    def __init__(self):
        super().__init__("rotate", "Rotate", "rotate")
        self.tooltip = "Rotate objects (E)"
        self.shortcut = "E"
        self._is_rotating = False

    def on_mouse_down(self, ctx: ModeContext, x: int, y: int, button: int) -> bool:
        if button == 0:
            self._is_rotating = True
            return True
        return False

    def on_mouse_up(self, ctx: ModeContext, x: int, y: int, button: int) -> bool:
        if button == 0 and self._is_rotating:
            self._is_rotating = False
            return True
        return False

    def on_mouse_move(self, ctx: ModeContext, x: int, y: int) -> bool:
        return self._is_rotating


@editor(category="Modes")
class ScaleTool(ModeTool):
    """Tool for scaling objects."""
    __slots__ = ("_is_scaling",)

    def __init__(self):
        super().__init__("scale", "Scale", "scale")
        self.tooltip = "Scale objects (R)"
        self.shortcut = "R"
        self._is_scaling = False

    def on_mouse_down(self, ctx: ModeContext, x: int, y: int, button: int) -> bool:
        if button == 0:
            self._is_scaling = True
            return True
        return False

    def on_mouse_up(self, ctx: ModeContext, x: int, y: int, button: int) -> bool:
        if button == 0 and self._is_scaling:
            self._is_scaling = False
            return True
        return False

    def on_mouse_move(self, ctx: ModeContext, x: int, y: int) -> bool:
        return self._is_scaling


# Concrete mode implementations

@editor(category="Modes")
@reloadable()
class SelectMode(EditorMode):
    """Mode for selecting and transforming objects."""
    __slots__ = ()

    def __init__(self):
        super().__init__(ModeType.SELECT, "Select")
        self.icon = "cursor"
        # Add default tools
        self.add_tool(SelectTool())
        self.add_tool(MoveTool())
        self.add_tool(RotateTool())
        self.add_tool(ScaleTool())

    def _on_enter(self) -> None:
        """Set up select mode."""
        # Set default tool to select
        self.set_active_tool("select")

    def _on_exit(self) -> None:
        """Clean up select mode."""
        pass


@editor(category="Modes")
class BrushTool(ModeTool):
    """Tool for brush-based painting."""
    __slots__ = ("brush_size", "brush_strength", "brush_falloff",
                 "_is_painting", "_stroke_points")

    def __init__(self):
        super().__init__("brush", "Brush", "brush")
        self.tooltip = "Paint brush (B)"
        self.shortcut = "B"
        self.brush_size: float = 10.0
        self.brush_strength: float = 1.0
        self.brush_falloff: str = "smooth"  # smooth, linear, constant
        self._is_painting = False
        self._stroke_points: list[tuple[int, int]] = []

    def on_mouse_down(self, ctx: ModeContext, x: int, y: int, button: int) -> bool:
        if button == 0:
            self._is_painting = True
            self._stroke_points = [(x, y)]
            return True
        return False

    def on_mouse_up(self, ctx: ModeContext, x: int, y: int, button: int) -> bool:
        if button == 0 and self._is_painting:
            self._is_painting = False
            # Apply stroke
            self._stroke_points = []
            return True
        return False

    def on_mouse_move(self, ctx: ModeContext, x: int, y: int) -> bool:
        if self._is_painting:
            self._stroke_points.append((x, y))
            return True
        return False


@editor(category="Modes")
@reloadable()
class PaintMode(EditorMode):
    """Mode for painting textures or vertex colors."""
    __slots__ = ("paint_target", "blend_mode")

    def __init__(self):
        super().__init__(ModeType.PAINT, "Paint")
        self.icon = "paintbrush"
        self.paint_target: str = "texture"  # texture, vertex_color, weight
        self.blend_mode: str = "normal"
        self.add_tool(BrushTool())

    def _on_enter(self) -> None:
        self.set_active_tool("brush")

    def _on_exit(self) -> None:
        pass


@editor(category="Modes")
class SculptBrushTool(ModeTool):
    """Tool for sculpting brushes."""
    __slots__ = ("brush_type", "brush_size", "brush_strength",
                 "_is_sculpting")

    def __init__(self, brush_type: str = "grab"):
        super().__init__(f"sculpt_{brush_type}", brush_type.title(), brush_type)
        self.brush_type = brush_type
        self.brush_size: float = 20.0
        self.brush_strength: float = 0.5
        self._is_sculpting = False

    def on_mouse_down(self, ctx: ModeContext, x: int, y: int, button: int) -> bool:
        if button == 0:
            self._is_sculpting = True
            return True
        return False

    def on_mouse_up(self, ctx: ModeContext, x: int, y: int, button: int) -> bool:
        if button == 0 and self._is_sculpting:
            self._is_sculpting = False
            return True
        return False

    def on_mouse_move(self, ctx: ModeContext, x: int, y: int) -> bool:
        return self._is_sculpting


@editor(category="Modes")
@reloadable()
class SculptMode(EditorMode):
    """Mode for mesh sculpting."""
    __slots__ = ("symmetry", "symmetry_axis")

    def __init__(self):
        super().__init__(ModeType.SCULPT, "Sculpt")
        self.icon = "sculpt"
        self.symmetry: bool = True
        self.symmetry_axis: str = "x"

        # Add sculpt brushes
        for brush_type in ["grab", "smooth", "flatten", "inflate", "pinch", "clay"]:
            self.add_tool(SculptBrushTool(brush_type))

    def _on_enter(self) -> None:
        self.set_active_tool("sculpt_grab")

    def _on_exit(self) -> None:
        pass


@editor(category="Modes")
class PlacementTool(ModeTool):
    """Tool for placing objects."""
    __slots__ = ("object_to_place", "placement_mode", "align_to_surface",
                 "_preview_object")

    def __init__(self):
        super().__init__("place", "Place Object", "place")
        self.tooltip = "Place objects in scene"
        self.object_to_place: Any = None
        self.placement_mode: str = "click"  # click, paint, line
        self.align_to_surface: bool = True
        self._preview_object: Any = None

    def on_mouse_down(self, ctx: ModeContext, x: int, y: int, button: int) -> bool:
        if button == 0 and self.object_to_place:
            # Place object at position
            return True
        return False

    def on_mouse_up(self, ctx: ModeContext, x: int, y: int, button: int) -> bool:
        return False

    def on_mouse_move(self, ctx: ModeContext, x: int, y: int) -> bool:
        # Update preview position
        return self.object_to_place is not None


@editor(category="Modes")
@reloadable()
class PlacementMode(EditorMode):
    """Mode for placing objects in the scene."""
    __slots__ = ("random_rotation", "random_scale_range")

    def __init__(self):
        super().__init__(ModeType.PLACEMENT, "Placement")
        self.icon = "place"
        self.random_rotation: bool = False
        self.random_scale_range: tuple[float, float] = (0.8, 1.2)
        self.add_tool(PlacementTool())

    def _on_enter(self) -> None:
        self.set_active_tool("place")

    def _on_exit(self) -> None:
        pass


@editor(category="Modes")
class TimelineTool(ModeTool):
    """Tool for timeline manipulation."""
    __slots__ = ("_is_scrubbing",)

    def __init__(self):
        super().__init__("timeline", "Timeline", "timeline")
        self.tooltip = "Scrub timeline"
        self._is_scrubbing = False

    def on_mouse_down(self, ctx: ModeContext, x: int, y: int, button: int) -> bool:
        if button == 0:
            self._is_scrubbing = True
            return True
        return False

    def on_mouse_up(self, ctx: ModeContext, x: int, y: int, button: int) -> bool:
        if button == 0 and self._is_scrubbing:
            self._is_scrubbing = False
            return True
        return False

    def on_mouse_move(self, ctx: ModeContext, x: int, y: int) -> bool:
        return self._is_scrubbing


@editor(category="Modes")
@reloadable()
class SequenceMode(EditorMode):
    """Mode for animation sequencing."""
    __slots__ = ("current_time", "playback_speed", "loop_mode",
                 "auto_key", "_is_playing")

    def __init__(self):
        super().__init__(ModeType.SEQUENCE, "Sequence")
        self.icon = "sequence"
        self.current_time: float = 0.0
        self.playback_speed: float = 1.0
        self.loop_mode: str = "none"  # none, loop, ping_pong
        self.auto_key: bool = False
        self._is_playing = False
        self.add_tool(TimelineTool())

    def _on_enter(self) -> None:
        self.set_active_tool("timeline")

    def _on_exit(self) -> None:
        self._is_playing = False

    def _on_update(self, delta_time: float) -> None:
        if self._is_playing:
            self.current_time += delta_time * self.playback_speed

    def play(self) -> None:
        """Start playback."""
        self._is_playing = True

    def pause(self) -> None:
        """Pause playback."""
        self._is_playing = False

    def stop(self) -> None:
        """Stop playback and reset to start."""
        self._is_playing = False
        self.current_time = 0.0

    def toggle_play(self) -> bool:
        """Toggle playback. Returns new playing state."""
        self._is_playing = not self._is_playing
        return self._is_playing

    @property
    def is_playing(self) -> bool:
        """Check if playing."""
        return self._is_playing


@editor(category="Modes")
@reloadable(preserve=["_modes", "_active_mode"])
class ModeManager:
    """Manages editor modes and mode switching."""
    __slots__ = ("_modes", "_active_mode", "_context", "_previous_mode",
                 "on_mode_changed", "on_tool_changed")

    def __init__(self):
        self._modes: dict[ModeType, EditorMode] = {}
        self._active_mode: Optional[EditorMode] = None
        self._context: Optional[ModeContext] = None
        self._previous_mode: Optional[ModeType] = None
        self.on_mode_changed: Optional[Callable[[EditorMode], None]] = None
        self.on_tool_changed: Optional[Callable[[ModeTool], None]] = None

        # Register default modes
        self.register_mode(SelectMode())
        self.register_mode(PaintMode())
        self.register_mode(SculptMode())
        self.register_mode(PlacementMode())
        self.register_mode(SequenceMode())

    @property
    def modes(self) -> list[EditorMode]:
        """Get all registered modes."""
        return list(self._modes.values())

    @property
    def active_mode(self) -> Optional[EditorMode]:
        """Get the active mode."""
        return self._active_mode

    @property
    def active_mode_type(self) -> Optional[ModeType]:
        """Get the active mode type."""
        return self._active_mode.mode_type if self._active_mode else None

    def set_context(self, context: ModeContext) -> None:
        """Set the mode context."""
        self._context = context

    def register_mode(self, mode: EditorMode) -> None:
        """Register a mode."""
        self._modes[mode.mode_type] = mode

    def unregister_mode(self, mode_type: ModeType) -> Optional[EditorMode]:
        """Unregister a mode."""
        mode = self._modes.pop(mode_type, None)
        if mode == self._active_mode:
            self._active_mode = None
        return mode

    def get_mode(self, mode_type: ModeType) -> Optional[EditorMode]:
        """Get a mode by type."""
        return self._modes.get(mode_type)

    def set_mode(self, mode_type: ModeType) -> bool:
        """Set the active mode. Returns True if successful."""
        mode = self._modes.get(mode_type)
        if mode and mode.enabled:
            if self._active_mode:
                self._previous_mode = self._active_mode.mode_type
                self._active_mode.exit()

            self._active_mode = mode
            if self._context:
                mode.enter(self._context)

            if self.on_mode_changed:
                self.on_mode_changed(mode)
            return True
        return False

    def toggle_previous_mode(self) -> bool:
        """Toggle back to the previous mode."""
        if self._previous_mode:
            return self.set_mode(self._previous_mode)
        return False

    def update(self, delta_time: float) -> None:
        """Update the active mode."""
        if self._active_mode:
            self._active_mode.update(delta_time)

    def on_mouse_down(self, x: int, y: int, button: int) -> bool:
        """Forward mouse down to active mode."""
        if self._active_mode:
            return self._active_mode.on_mouse_down(x, y, button)
        return False

    def on_mouse_up(self, x: int, y: int, button: int) -> bool:
        """Forward mouse up to active mode."""
        if self._active_mode:
            return self._active_mode.on_mouse_up(x, y, button)
        return False

    def on_mouse_move(self, x: int, y: int) -> bool:
        """Forward mouse move to active mode."""
        if self._active_mode:
            return self._active_mode.on_mouse_move(x, y)
        return False

    def on_key_down(self, key: str) -> bool:
        """Forward key down to active mode."""
        if self._active_mode:
            return self._active_mode.on_key_down(key)
        return False

    def on_key_up(self, key: str) -> bool:
        """Forward key up to active mode."""
        if self._active_mode:
            return self._active_mode.on_key_up(key)
        return False

    def set_active_tool(self, tool_id: str) -> bool:
        """Set active tool in current mode."""
        if self._active_mode:
            result = self._active_mode.set_active_tool(tool_id)
            if result and self.on_tool_changed and self._active_mode.active_tool:
                self.on_tool_changed(self._active_mode.active_tool)
            return result
        return False
