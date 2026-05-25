"""
Comprehensive tests for the Editor Modes system.

Tests cover:
- Mode lifecycle (enter, exit, update)
- Mode-specific tools
- Mode context
- Select, Paint, Sculpt, Placement, Sequence modes
- Mode manager
"""
import pytest
import sys

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.tooling.editor.modes import (
    EditorMode,
    ModeManager,
    ModeType,
    ModeContext,
    ModeTool,
    SelectMode,
    PaintMode,
    SculptMode,
    PlacementMode,
    SequenceMode,
    SelectTool,
    MoveTool,
    BrushTool,
)


class ConcreteModeTool(ModeTool):
    """Concrete implementation of ModeTool for testing."""

    def __init__(self, id: str = "test"):
        super().__init__(id, "Test Tool", "test_icon")
        self.mouse_down_calls = []
        self.mouse_up_calls = []
        self.mouse_move_calls = []

    def on_mouse_down(self, ctx, x, y, button):
        self.mouse_down_calls.append((x, y, button))
        return True

    def on_mouse_up(self, ctx, x, y, button):
        self.mouse_up_calls.append((x, y, button))
        return True

    def on_mouse_move(self, ctx, x, y):
        self.mouse_move_calls.append((x, y))
        return True


class ConcreteEditorMode(EditorMode):
    """Concrete implementation of EditorMode for testing."""

    def __init__(self):
        super().__init__(ModeType.SELECT, "Test Mode")
        self.entered = False
        self.exited = False

    def _on_enter(self):
        self.entered = True

    def _on_exit(self):
        self.exited = True


class TestModeContext:
    """Tests for ModeContext class."""

    def test_context_creation(self):
        """ModeContext should be created with defaults."""
        ctx = ModeContext()
        assert ctx.viewport is None
        assert ctx.selection is None
        assert ctx.delta_time == 0.0
        assert ctx.mouse_position == (0, 0)
        assert len(ctx.modifier_keys) == 0

    def test_context_with_references(self):
        """ModeContext can hold references."""
        viewport = object()
        selection = object()

        ctx = ModeContext(viewport=viewport, selection=selection)

        assert ctx.viewport is viewport
        assert ctx.selection is selection


class TestModeTool:
    """Tests for ModeTool class."""

    def test_tool_creation(self):
        """ModeTool should be created with defaults."""
        tool = ConcreteModeTool("select")
        assert tool.id == "select"
        assert tool.enabled is True
        assert tool.active is False

    def test_tool_activate_deactivate(self):
        """Tool can be activated and deactivated."""
        tool = ConcreteModeTool()
        activated = []
        deactivated = []
        tool.on_activated = lambda: activated.append(True)
        tool.on_deactivated = lambda: deactivated.append(True)

        tool.activate()
        assert tool.active is True
        assert len(activated) == 1

        tool.deactivate()
        assert tool.active is False
        assert len(deactivated) == 1

    def test_tool_mouse_events(self):
        """Tool handles mouse events."""
        tool = ConcreteModeTool()
        ctx = ModeContext()

        tool.on_mouse_down(ctx, 100, 200, 0)
        assert len(tool.mouse_down_calls) == 1
        assert tool.mouse_down_calls[0] == (100, 200, 0)

        tool.on_mouse_up(ctx, 100, 200, 0)
        assert len(tool.mouse_up_calls) == 1

        tool.on_mouse_move(ctx, 150, 250)
        assert len(tool.mouse_move_calls) == 1


class TestEditorMode:
    """Tests for EditorMode base class."""

    def test_mode_creation(self):
        """EditorMode should be created with type."""
        mode = ConcreteEditorMode()
        assert mode.mode_type == ModeType.SELECT
        assert mode.enabled is True

    def test_mode_enter_exit(self):
        """Mode has enter/exit lifecycle."""
        mode = ConcreteEditorMode()
        ctx = ModeContext()

        mode.enter(ctx)
        assert mode.entered is True
        assert mode.context is ctx

        mode.exit()
        assert mode.exited is True
        assert mode.context is None

    def test_mode_callbacks(self):
        """Mode triggers callbacks."""
        mode = ConcreteEditorMode()
        ctx = ModeContext()
        entered = []
        exited = []
        mode.on_enter = lambda: entered.append(True)
        mode.on_exit = lambda: exited.append(True)

        mode.enter(ctx)
        assert len(entered) == 1

        mode.exit()
        assert len(exited) == 1

    def test_mode_add_tool(self):
        """Tools can be added to mode."""
        mode = ConcreteEditorMode()
        tool = ConcreteModeTool("tool1")

        mode.add_tool(tool)
        assert tool in mode.tools
        assert mode.get_tool("tool1") == tool

    def test_mode_remove_tool(self):
        """Tools can be removed from mode."""
        mode = ConcreteEditorMode()
        tool = ConcreteModeTool("tool1")

        mode.add_tool(tool)
        removed = mode.remove_tool("tool1")

        assert removed == tool
        assert mode.get_tool("tool1") is None

    def test_mode_set_active_tool(self):
        """Active tool can be set."""
        mode = ConcreteEditorMode()
        tool1 = ConcreteModeTool("tool1")
        tool2 = ConcreteModeTool("tool2")

        mode.add_tool(tool1)
        mode.add_tool(tool2)

        assert mode.set_active_tool("tool2") is True
        assert mode.active_tool == tool2
        assert tool2.active is True

    def test_mode_set_active_tool_nonexistent(self):
        """Setting nonexistent tool returns False."""
        mode = ConcreteEditorMode()

        assert mode.set_active_tool("nonexistent") is False

    def test_mode_forward_mouse_events(self):
        """Mode forwards events to active tool."""
        mode = ConcreteEditorMode()
        ctx = ModeContext()
        tool = ConcreteModeTool("tool1")

        mode.add_tool(tool)
        mode.enter(ctx)
        mode.set_active_tool("tool1")

        mode.on_mouse_down(100, 200, 0)
        assert len(tool.mouse_down_calls) == 1

        mode.on_mouse_move(150, 250)
        assert len(tool.mouse_move_calls) == 1

    def test_mode_update(self):
        """Mode update is called."""
        mode = ConcreteEditorMode()
        ctx = ModeContext()
        mode.enter(ctx)

        mode.update(0.016)
        assert ctx.delta_time == 0.016


class TestSelectMode:
    """Tests for SelectMode class."""

    def test_select_mode_creation(self):
        """SelectMode should be created with tools."""
        mode = SelectMode()
        assert mode.mode_type == ModeType.SELECT
        assert len(mode.tools) >= 4  # select, move, rotate, scale

    def test_select_mode_has_standard_tools(self):
        """SelectMode has standard transform tools."""
        mode = SelectMode()

        assert mode.get_tool("select") is not None
        assert mode.get_tool("move") is not None
        assert mode.get_tool("rotate") is not None
        assert mode.get_tool("scale") is not None

    def test_select_mode_default_tool(self):
        """SelectMode defaults to select tool on enter."""
        mode = SelectMode()
        ctx = ModeContext()

        mode.enter(ctx)
        assert mode.active_tool.id == "select"


class TestSelectTool:
    """Tests for SelectTool class."""

    def test_select_tool_creation(self):
        """SelectTool should be created properly."""
        tool = SelectTool()
        assert tool.id == "select"
        assert tool.shortcut == "Q"

    def test_select_tool_drag_state(self):
        """SelectTool tracks drag state."""
        tool = SelectTool()
        ctx = ModeContext()

        tool.on_mouse_down(ctx, 100, 100, 0)
        assert tool._is_dragging is True
        assert tool._drag_start == (100, 100)

        tool.on_mouse_up(ctx, 100, 100, 0)
        assert tool._is_dragging is False


class TestMoveTool:
    """Tests for MoveTool class."""

    def test_move_tool_creation(self):
        """MoveTool should be created properly."""
        tool = MoveTool()
        assert tool.id == "move"
        assert tool.shortcut == "W"


class TestPaintMode:
    """Tests for PaintMode class."""

    def test_paint_mode_creation(self):
        """PaintMode should be created with brush tool."""
        mode = PaintMode()
        assert mode.mode_type == ModeType.PAINT
        assert mode.get_tool("brush") is not None

    def test_paint_mode_default_settings(self):
        """PaintMode has default settings."""
        mode = PaintMode()
        assert mode.paint_target == "texture"
        assert mode.blend_mode == "normal"


class TestBrushTool:
    """Tests for BrushTool class."""

    def test_brush_tool_creation(self):
        """BrushTool should be created with defaults."""
        tool = BrushTool()
        assert tool.id == "brush"
        assert tool.brush_size == 10.0
        assert tool.brush_strength == 1.0

    def test_brush_tool_stroke(self):
        """BrushTool records stroke points."""
        tool = BrushTool()
        ctx = ModeContext()

        tool.on_mouse_down(ctx, 100, 100, 0)
        assert tool._is_painting is True

        tool.on_mouse_move(ctx, 110, 110)
        tool.on_mouse_move(ctx, 120, 120)

        assert len(tool._stroke_points) >= 2

        tool.on_mouse_up(ctx, 120, 120, 0)
        assert tool._is_painting is False


class TestSculptMode:
    """Tests for SculptMode class."""

    def test_sculpt_mode_creation(self):
        """SculptMode should be created with brushes."""
        mode = SculptMode()
        assert mode.mode_type == ModeType.SCULPT

    def test_sculpt_mode_has_brushes(self):
        """SculptMode has standard sculpt brushes."""
        mode = SculptMode()

        brush_types = ["grab", "smooth", "flatten", "inflate", "pinch", "clay"]
        for brush_type in brush_types:
            assert mode.get_tool(f"sculpt_{brush_type}") is not None

    def test_sculpt_mode_symmetry_settings(self):
        """SculptMode has symmetry settings."""
        mode = SculptMode()
        assert mode.symmetry is True
        assert mode.symmetry_axis == "x"


class TestPlacementMode:
    """Tests for PlacementMode class."""

    def test_placement_mode_creation(self):
        """PlacementMode should be created properly."""
        mode = PlacementMode()
        assert mode.mode_type == ModeType.PLACEMENT
        assert mode.get_tool("place") is not None

    def test_placement_mode_settings(self):
        """PlacementMode has randomization settings."""
        mode = PlacementMode()
        assert mode.random_rotation is False
        assert mode.random_scale_range == (0.8, 1.2)


class TestSequenceMode:
    """Tests for SequenceMode class."""

    def test_sequence_mode_creation(self):
        """SequenceMode should be created properly."""
        mode = SequenceMode()
        assert mode.mode_type == ModeType.SEQUENCE
        assert mode.current_time == 0.0

    def test_sequence_mode_playback(self):
        """SequenceMode has playback controls."""
        mode = SequenceMode()
        ctx = ModeContext()
        mode.enter(ctx)

        assert mode.is_playing is False

        mode.play()
        assert mode.is_playing is True

        mode.pause()
        assert mode.is_playing is False

    def test_sequence_mode_toggle_play(self):
        """SequenceMode playback can be toggled."""
        mode = SequenceMode()
        ctx = ModeContext()
        mode.enter(ctx)

        result = mode.toggle_play()
        assert result is True
        assert mode.is_playing is True

        result = mode.toggle_play()
        assert result is False
        assert mode.is_playing is False

    def test_sequence_mode_stop(self):
        """SequenceMode stop resets time."""
        mode = SequenceMode()
        ctx = ModeContext()
        mode.enter(ctx)

        mode.current_time = 5.0
        mode.play()

        mode.stop()
        assert mode.is_playing is False
        assert mode.current_time == 0.0

    def test_sequence_mode_update_during_playback(self):
        """SequenceMode updates time during playback."""
        mode = SequenceMode()
        ctx = ModeContext()
        mode.enter(ctx)

        mode.play()
        mode.update(1.0)

        assert mode.current_time == pytest.approx(1.0)

    def test_sequence_mode_playback_speed(self):
        """SequenceMode respects playback speed."""
        mode = SequenceMode()
        ctx = ModeContext()
        mode.enter(ctx)

        mode.playback_speed = 2.0
        mode.play()
        mode.update(1.0)

        assert mode.current_time == pytest.approx(2.0)


class TestModeManager:
    """Tests for ModeManager class."""

    def test_manager_creation(self):
        """ModeManager should be created with default modes."""
        manager = ModeManager()
        assert len(manager.modes) >= 5  # Default modes

    def test_manager_has_default_modes(self):
        """ModeManager has all default modes."""
        manager = ModeManager()

        assert manager.get_mode(ModeType.SELECT) is not None
        assert manager.get_mode(ModeType.PAINT) is not None
        assert manager.get_mode(ModeType.SCULPT) is not None
        assert manager.get_mode(ModeType.PLACEMENT) is not None
        assert manager.get_mode(ModeType.SEQUENCE) is not None

    def test_manager_set_mode(self):
        """ModeManager can set active mode."""
        manager = ModeManager()
        ctx = ModeContext()
        manager.set_context(ctx)

        assert manager.set_mode(ModeType.PAINT) is True
        assert manager.active_mode_type == ModeType.PAINT

    def test_manager_set_mode_callback(self):
        """ModeManager triggers callback on mode change."""
        manager = ModeManager()
        ctx = ModeContext()
        manager.set_context(ctx)

        changes = []
        manager.on_mode_changed = lambda m: changes.append(m.mode_type)

        manager.set_mode(ModeType.PAINT)
        assert len(changes) == 1
        assert changes[0] == ModeType.PAINT

    def test_manager_toggle_previous_mode(self):
        """ModeManager can toggle back to previous mode."""
        manager = ModeManager()
        ctx = ModeContext()
        manager.set_context(ctx)

        manager.set_mode(ModeType.SELECT)
        manager.set_mode(ModeType.PAINT)

        assert manager.toggle_previous_mode() is True
        assert manager.active_mode_type == ModeType.SELECT

    def test_manager_update(self):
        """ModeManager forwards update to active mode."""
        manager = ModeManager()
        ctx = ModeContext()
        manager.set_context(ctx)

        manager.set_mode(ModeType.SEQUENCE)
        seq_mode = manager.active_mode
        seq_mode.play()

        manager.update(1.0)

        assert seq_mode.current_time > 0

    def test_manager_forward_mouse_events(self):
        """ModeManager forwards mouse events."""
        manager = ModeManager()
        ctx = ModeContext()
        manager.set_context(ctx)

        manager.set_mode(ModeType.SELECT)

        # Should not raise
        manager.on_mouse_down(100, 100, 0)
        manager.on_mouse_move(150, 150)
        manager.on_mouse_up(150, 150, 0)

    def test_manager_forward_key_events(self):
        """ModeManager forwards key events."""
        manager = ModeManager()
        ctx = ModeContext()
        manager.set_context(ctx)

        manager.set_mode(ModeType.SELECT)

        # Should not raise
        manager.on_key_down("w")
        manager.on_key_up("w")

    def test_manager_register_mode(self):
        """ModeManager can register new modes."""
        manager = ModeManager()
        custom_mode = ConcreteEditorMode()

        manager.register_mode(custom_mode)
        assert manager.get_mode(ModeType.SELECT) is not None

    def test_manager_unregister_mode(self):
        """ModeManager can unregister modes."""
        manager = ModeManager()

        removed = manager.unregister_mode(ModeType.PAINT)
        assert removed is not None
        assert manager.get_mode(ModeType.PAINT) is None

    def test_manager_set_active_tool(self):
        """ModeManager can set active tool in current mode."""
        manager = ModeManager()
        ctx = ModeContext()
        manager.set_context(ctx)

        manager.set_mode(ModeType.SELECT)
        assert manager.set_active_tool("move") is True
        assert manager.active_mode.active_tool.id == "move"

    def test_manager_tool_changed_callback(self):
        """ModeManager triggers callback on tool change."""
        manager = ModeManager()
        ctx = ModeContext()
        manager.set_context(ctx)

        changes = []
        manager.on_tool_changed = lambda t: changes.append(t.id)

        manager.set_mode(ModeType.SELECT)
        manager.set_active_tool("move")

        assert len(changes) == 1
        assert changes[0] == "move"


class TestModeType:
    """Tests for ModeType enumeration."""

    def test_all_mode_types_exist(self):
        """All required mode types exist."""
        types = [
            ModeType.SELECT,
            ModeType.PAINT,
            ModeType.SCULPT,
            ModeType.PLACEMENT,
            ModeType.SEQUENCE,
        ]

        for mtype in types:
            assert isinstance(mtype, ModeType)
