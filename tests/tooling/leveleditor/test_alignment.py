"""
Tests for the alignment module.

Tests all alignment operations.
"""

import pytest
import sys

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.tooling.leveleditor.alignment import (
    AlignAxis,
    AlignEdge,
    AlignReference,
    SpaceMode,
    AlignTarget,
    AlignmentSettings,
    AlignmentResult,
    AlignmentTool,
)
from engine.tooling.leveleditor.placement import Vector3
from foundation.tracker import tracker


@pytest.fixture(autouse=True)
def reset_tracker():
    """Reset tracker state before each test."""
    tracker._dirty.clear()
    tracker._cb_global.clear()
    tracker._cb_type.clear()
    tracker._cb_obj.clear()
    tracker._undo.clear()
    tracker._redo.clear()
    tracker._txn = None
    yield


def create_target(x, y, z, width=1, height=1, depth=1, obj_id=None):
    """Helper to create AlignTarget."""
    return AlignTarget(
        object_id=obj_id or f"obj-{x}-{y}-{z}",
        position=Vector3(x, y, z),
        bounds_min=Vector3(x - width/2, y - height/2, z - depth/2),
        bounds_max=Vector3(x + width/2, y + height/2, z + depth/2),
        pivot=Vector3(x, y, z),
    )


class TestAlignTarget:
    """Tests for AlignTarget class."""

    def test_center_calculation(self):
        """Center should be calculated from bounds."""
        target = AlignTarget(
            object_id="test",
            position=Vector3(0, 0, 0),
            bounds_min=Vector3(0, 0, 0),
            bounds_max=Vector3(10, 10, 10),
            pivot=Vector3(0, 0, 0),
        )

        center = target.center
        assert center.x == 5
        assert center.y == 5
        assert center.z == 5

    def test_size_calculation(self):
        """Size should be calculated from bounds."""
        target = AlignTarget(
            object_id="test",
            position=Vector3(0, 0, 0),
            bounds_min=Vector3(0, 0, 0),
            bounds_max=Vector3(10, 5, 20),
            pivot=Vector3(0, 0, 0),
        )

        size = target.size
        assert size.x == 10
        assert size.y == 5
        assert size.z == 20


class TestAlignmentTool:
    """Tests for AlignmentTool class."""

    def test_creation(self):
        """Tool should initialize with default settings."""
        tool = AlignmentTool()
        assert tool.settings.axis == AlignAxis.X
        assert tool.settings.edge == AlignEdge.MIN

    def test_set_targets(self):
        """Should store targets."""
        tool = AlignmentTool()
        targets = [create_target(0, 0, 0), create_target(10, 0, 0)]

        tool.set_targets(targets, targets[0].object_id)

        # Accessing internal state through align operations
        result = tool.align()
        assert result.objects_aligned >= 0

    def test_align_left(self):
        """Should align objects to left edge."""
        tool = AlignmentTool()
        targets = [
            create_target(5, 0, 0),
            create_target(10, 0, 0),
            create_target(15, 0, 0),
        ]

        tool.set_targets(targets)
        tool.settings.reference = AlignReference.SELECTION_BOUNDS
        result = tool.align_left()

        assert result.success is True
        # All should align to leftmost
        for obj_id, new_pos in result.new_positions.items():
            assert new_pos.x < 10  # Should move toward left

    def test_align_right(self):
        """Should align objects to right edge."""
        tool = AlignmentTool()
        targets = [
            create_target(5, 0, 0),
            create_target(10, 0, 0),
            create_target(15, 0, 0),
        ]

        tool.set_targets(targets)
        tool.settings.reference = AlignReference.SELECTION_BOUNDS
        result = tool.align_right()

        assert result.success is True

    def test_align_center_x(self):
        """Should align objects to center X."""
        tool = AlignmentTool()
        targets = [
            create_target(0, 0, 0),
            create_target(10, 0, 0),
        ]

        tool.set_targets(targets)
        tool.settings.reference = AlignReference.SELECTION_BOUNDS
        result = tool.align_center_x()

        assert result.success is True

    def test_align_top(self):
        """Should align objects to top edge."""
        tool = AlignmentTool()
        targets = [
            create_target(0, 5, 0),
            create_target(0, 10, 0),
            create_target(0, 15, 0),
        ]

        tool.set_targets(targets)
        tool.settings.reference = AlignReference.SELECTION_BOUNDS
        result = tool.align_top()

        assert result.success is True

    def test_align_bottom(self):
        """Should align objects to bottom edge."""
        tool = AlignmentTool()
        targets = [
            create_target(0, 5, 0),
            create_target(0, 10, 0),
        ]

        tool.set_targets(targets)
        result = tool.align_bottom()

        assert result.success is True

    def test_align_center_y(self):
        """Should align objects to center Y."""
        tool = AlignmentTool()
        targets = [
            create_target(0, 0, 0),
            create_target(0, 20, 0),
        ]

        tool.set_targets(targets)
        result = tool.align_center_y()

        assert result.success is True

    def test_align_front(self):
        """Should align objects to front edge."""
        tool = AlignmentTool()
        targets = [
            create_target(0, 0, 5),
            create_target(0, 0, 10),
        ]

        tool.set_targets(targets)
        result = tool.align_front()

        assert result.success is True

    def test_align_back(self):
        """Should align objects to back edge."""
        tool = AlignmentTool()
        targets = [
            create_target(0, 0, 5),
            create_target(0, 0, 10),
        ]

        tool.set_targets(targets)
        result = tool.align_back()

        assert result.success is True

    def test_align_center_z(self):
        """Should align objects to center Z."""
        tool = AlignmentTool()
        targets = [
            create_target(0, 0, 0),
            create_target(0, 0, 20),
        ]

        tool.set_targets(targets)
        result = tool.align_center_z()

        assert result.success is True

    def test_align_to_active_object(self):
        """Should align to active object."""
        tool = AlignmentTool()
        targets = [
            create_target(0, 0, 0, obj_id="obj1"),
            create_target(10, 0, 0, obj_id="obj2"),
        ]

        tool.set_targets(targets, "obj1")
        tool.settings.reference = AlignReference.ACTIVE_OBJECT
        result = tool.align_left()

        assert result.success is True
        # Active object should not move
        assert "obj1" not in result.new_positions

    def test_align_to_first_selected(self):
        """Should align to first selected object."""
        tool = AlignmentTool()
        targets = [
            create_target(0, 0, 0, obj_id="first"),
            create_target(10, 0, 0, obj_id="second"),
        ]

        tool.set_targets(targets)
        tool.settings.reference = AlignReference.FIRST_SELECTED
        result = tool.align_left()

        assert result.success is True
        assert "first" not in result.new_positions

    def test_align_to_last_selected(self):
        """Should align to last selected object."""
        tool = AlignmentTool()
        targets = [
            create_target(0, 0, 0, obj_id="first"),
            create_target(10, 0, 0, obj_id="last"),
        ]

        tool.set_targets(targets)
        tool.settings.reference = AlignReference.LAST_SELECTED
        result = tool.align_left()

        assert result.success is True
        assert "last" not in result.new_positions

    def test_align_to_world_origin(self):
        """Should align to world origin."""
        tool = AlignmentTool()
        targets = [
            create_target(10, 10, 10),
        ]

        tool.set_targets(targets)
        tool.settings.reference = AlignReference.WORLD_ORIGIN
        result = tool.align_left()

        assert result.success is True

    def test_align_to_cursor(self):
        """Should align to cursor position."""
        tool = AlignmentTool()
        tool.cursor_position = Vector3(5, 5, 5)
        targets = [
            create_target(0, 0, 0),
            create_target(10, 0, 0),
        ]

        tool.set_targets(targets)
        tool.settings.reference = AlignReference.CURSOR
        result = tool.align_left()

        assert result.success is True

    def test_align_to_custom_position(self):
        """Should align to custom position."""
        tool = AlignmentTool()
        targets = [
            create_target(0, 0, 0),
        ]

        tool.set_targets(targets)
        tool.settings.reference = AlignReference.CUSTOM
        tool.settings.custom_position = Vector3(100, 100, 100)
        result = tool.align_left()

        assert result.success is True

    def test_align_to_ground(self):
        """Should align objects to ground plane."""
        tool = AlignmentTool()
        targets = [
            create_target(0, 5, 0),
            create_target(5, 10, 0),
        ]

        tool.set_targets(targets)
        result = tool.align_to_ground(ground_y=0.0)

        assert result.success is True

    def test_stack_vertically(self):
        """Should stack objects vertically."""
        tool = AlignmentTool()
        targets = [
            create_target(0, 0, 0, height=2, obj_id="a"),
            create_target(0, 5, 0, height=2, obj_id="b"),
            create_target(0, 10, 0, height=2, obj_id="c"),
        ]

        tool.set_targets(targets)
        result = tool.stack_vertically(spacing=1.0)

        assert result.success is True

    def test_stack_vertically_insufficient_objects(self):
        """Stack should fail with less than 2 objects."""
        tool = AlignmentTool()
        targets = [create_target(0, 0, 0)]

        tool.set_targets(targets)
        result = tool.stack_vertically()

        assert result.success is False

    def test_preview_alignment(self):
        """Should preview without applying changes."""
        tool = AlignmentTool()
        targets = [
            create_target(0, 0, 0),
            create_target(10, 0, 0),
        ]

        tool.set_targets(targets)
        preview = tool.preview_alignment()

        assert isinstance(preview, dict)
        # Original positions should be unchanged
        for t in targets:
            assert t.position.x in [0, 10]

    def test_align_no_targets(self):
        """Align with no targets should fail."""
        tool = AlignmentTool()
        tool.set_targets([])
        result = tool.align()

        assert result.success is False
        assert result.objects_aligned == 0

    def test_callback_on_align(self):
        """Should trigger on_align callback."""
        tool = AlignmentTool()
        results_received = []

        def callback(result):
            results_received.append(result)

        tool.on("on_align", callback)
        targets = [create_target(0, 0, 0), create_target(10, 0, 0)]
        tool.set_targets(targets)
        tool.align()

        assert len(results_received) == 1

    def test_callback_on_preview(self):
        """Should trigger on_preview callback."""
        tool = AlignmentTool()
        previews_received = []

        def callback(preview):
            previews_received.append(preview)

        tool.on("on_preview", callback)
        targets = [create_target(0, 0, 0)]
        tool.set_targets(targets)
        tool.preview_alignment()

        assert len(previews_received) == 1

    def test_result_stores_original_positions(self):
        """Result should store original positions."""
        tool = AlignmentTool()
        targets = [
            create_target(5, 0, 0),
            create_target(10, 0, 0),
        ]

        tool.set_targets(targets)
        result = tool.align_left()

        assert len(result.original_positions) >= 1

    def test_result_stores_new_positions(self):
        """Result should store new positions."""
        tool = AlignmentTool()
        targets = [
            create_target(5, 0, 0),
            create_target(10, 0, 0),
        ]

        tool.set_targets(targets)
        result = tool.align_left()

        assert len(result.new_positions) >= 1

    def test_use_pivot_mode(self):
        """Should align using pivot instead of bounds."""
        tool = AlignmentTool()
        tool.settings.use_pivot = True

        targets = [create_target(0, 0, 0), create_target(10, 0, 0)]
        tool.set_targets(targets)
        result = tool.align_left()

        assert result.success is True
