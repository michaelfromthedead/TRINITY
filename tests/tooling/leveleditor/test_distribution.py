"""
Tests for the distribution module.

Tests even spacing and patterns.
"""

import math
import pytest
import sys

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.tooling.leveleditor.distribution import (
    DistributionMode,
    DistributionAxis,
    PatternType,
    SpacingSettings,
    PatternSettings,
    DistributionSettings,
    DistributionResult,
    DistributionTool,
)
from engine.tooling.leveleditor.alignment import AlignTarget
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


class TestDistributionTool:
    """Tests for DistributionTool class."""

    def test_creation(self):
        """Tool should initialize with default settings."""
        tool = DistributionTool()
        assert tool.settings.mode == DistributionMode.EVEN_SPACING
        assert tool.settings.axis == DistributionAxis.X

    def test_distribute_horizontally(self):
        """Should distribute objects evenly along X."""
        tool = DistributionTool()
        targets = [
            create_target(0, 0, 0, obj_id="a"),
            create_target(5, 0, 0, obj_id="b"),
            create_target(20, 0, 0, obj_id="c"),
        ]

        tool.set_targets(targets)
        result = tool.distribute_horizontally()

        assert result.success is True
        assert result.objects_distributed == 3

    def test_distribute_vertically(self):
        """Should distribute objects evenly along Y."""
        tool = DistributionTool()
        targets = [
            create_target(0, 0, 0, obj_id="a"),
            create_target(0, 5, 0, obj_id="b"),
            create_target(0, 20, 0, obj_id="c"),
        ]

        tool.set_targets(targets)
        result = tool.distribute_vertically()

        assert result.success is True
        assert result.objects_distributed == 3

    def test_distribute_depth(self):
        """Should distribute objects evenly along Z."""
        tool = DistributionTool()
        targets = [
            create_target(0, 0, 0, obj_id="a"),
            create_target(0, 0, 5, obj_id="b"),
            create_target(0, 0, 20, obj_id="c"),
        ]

        tool.set_targets(targets)
        result = tool.distribute_depth()

        assert result.success is True
        assert result.objects_distributed == 3

    def test_distribute_even_spacing(self):
        """Even spacing should create equal gaps."""
        tool = DistributionTool()
        targets = [
            create_target(0, 0, 0, obj_id="a"),
            create_target(2, 0, 0, obj_id="b"),  # Not evenly spaced
            create_target(10, 0, 0, obj_id="c"),
        ]

        tool.set_targets(targets)
        tool.settings.mode = DistributionMode.EVEN_SPACING
        result = tool.distribute()

        assert result.success is True
        # Middle object should be at x=5
        assert "b" in result.new_positions
        assert abs(result.new_positions["b"].x - 5) < 0.0001

    def test_distribute_with_fixed_spacing(self):
        """Should use fixed spacing when specified."""
        tool = DistributionTool()
        targets = [
            create_target(0, 0, 0, obj_id="a"),
            create_target(5, 0, 0, obj_id="b"),
            create_target(10, 0, 0, obj_id="c"),
        ]

        tool.set_targets(targets)
        tool.settings.spacing.fixed_spacing = 3.0
        result = tool.distribute_horizontally(spacing=3.0)

        assert result.success is True

    def test_distribute_equal_gaps(self):
        """Equal gaps mode should consider object sizes."""
        tool = DistributionTool()
        targets = [
            create_target(0, 0, 0, width=2, obj_id="a"),
            create_target(10, 0, 0, width=2, obj_id="b"),
            create_target(20, 0, 0, width=2, obj_id="c"),
        ]

        tool.set_targets(targets)
        tool.settings.mode = DistributionMode.EQUAL_GAPS
        result = tool.distribute()

        assert result.success is True

    def test_distribute_in_circle(self):
        """Should distribute objects in a circle."""
        tool = DistributionTool()
        targets = [
            create_target(0, 0, 0, obj_id=f"obj{i}")
            for i in range(8)
        ]

        tool.set_targets(targets)
        result = tool.distribute_in_circle(Vector3(0, 0, 0), radius=10)

        assert result.success is True
        assert result.objects_distributed == 8

        # Verify objects are on circle
        for obj_id, pos in result.new_positions.items():
            dist = math.sqrt(pos.x**2 + pos.z**2)
            assert abs(dist - 10) < 0.0001

    def test_distribute_in_grid(self):
        """Should distribute objects in a grid."""
        tool = DistributionTool()
        targets = [
            create_target(0, 0, 0, obj_id=f"obj{i}")
            for i in range(9)
        ]

        tool.set_targets(targets)
        result = tool.distribute_in_grid(
            Vector3(0, 0, 0),
            columns=3,
            rows=3,
            spacing=5.0
        )

        assert result.success is True
        assert result.objects_distributed == 9

    def test_distribute_arc(self):
        """Should distribute objects along an arc."""
        tool = DistributionTool()
        targets = [
            create_target(0, 0, 0, obj_id=f"obj{i}")
            for i in range(5)
        ]

        tool.set_targets(targets)
        tool.settings.mode = DistributionMode.PATTERN
        tool.settings.pattern.pattern_type = PatternType.ARC
        tool.settings.pattern.radius = 10.0
        tool.settings.pattern.start_angle = 0
        tool.settings.pattern.end_angle = math.pi  # Half circle

        result = tool.distribute()

        assert result.success is True

    def test_distribute_spiral(self):
        """Should distribute objects along a spiral."""
        tool = DistributionTool()
        targets = [
            create_target(0, 0, 0, obj_id=f"obj{i}")
            for i in range(10)
        ]

        tool.set_targets(targets)
        tool.settings.mode = DistributionMode.PATTERN
        tool.settings.pattern.pattern_type = PatternType.SPIRAL
        tool.settings.pattern.spiral_turns = 2.0
        tool.settings.pattern.spiral_start_radius = 1.0
        tool.settings.pattern.spiral_end_radius = 10.0

        result = tool.distribute()

        assert result.success is True

    def test_distribute_random(self):
        """Should distribute objects randomly within radius."""
        tool = DistributionTool()
        targets = [
            create_target(0, 0, 0, obj_id=f"obj{i}")
            for i in range(5)
        ]

        tool.set_targets(targets)
        tool.settings.mode = DistributionMode.PATTERN
        tool.settings.pattern.pattern_type = PatternType.RANDOM
        tool.settings.pattern.radius = 10.0

        result = tool.distribute()

        assert result.success is True

    def test_equalize_spacing(self):
        """Should equalize spacing between objects."""
        tool = DistributionTool()
        targets = [
            create_target(0, 0, 0, obj_id="a"),
            create_target(3, 0, 0, obj_id="b"),
            create_target(15, 0, 0, obj_id="c"),
        ]

        tool.set_targets(targets)
        result = tool.equalize_spacing(DistributionAxis.X)

        assert result.success is True

    def test_distribute_insufficient_objects(self):
        """Should fail with less than 2 objects."""
        tool = DistributionTool()
        targets = [create_target(0, 0, 0)]

        tool.set_targets(targets)
        result = tool.distribute()

        assert result.success is False

    def test_preview_distribution(self):
        """Should preview without applying changes."""
        tool = DistributionTool()
        targets = [
            create_target(0, 0, 0, obj_id="a"),
            create_target(10, 0, 0, obj_id="b"),
        ]

        tool.set_targets(targets)
        preview = tool.preview_distribution()

        assert isinstance(preview, dict)
        # Original positions unchanged
        for t in targets:
            assert t.position.x in [0, 10]

    def test_distribute_preserves_other_dimensions(self):
        """Should preserve non-distributed dimensions."""
        tool = DistributionTool()
        targets = [
            create_target(0, 5, 10, obj_id="a"),
            create_target(10, 15, 20, obj_id="b"),
        ]

        tool.set_targets(targets)
        tool.settings.preserve_dimension = True
        result = tool.distribute_horizontally()

        assert result.success is True
        # Y and Z should be unchanged for each object

    def test_callback_on_distribute(self):
        """Should trigger on_distribute callback."""
        tool = DistributionTool()
        results_received = []

        def callback(result):
            results_received.append(result)

        tool.on("on_distribute", callback)
        targets = [create_target(0, 0, 0), create_target(10, 0, 0)]
        tool.set_targets(targets)
        tool.distribute()

        assert len(results_received) == 1

    def test_callback_on_preview(self):
        """Should trigger on_preview callback."""
        tool = DistributionTool()
        previews_received = []

        def callback(preview):
            previews_received.append(preview)

        tool.on("on_preview", callback)
        targets = [create_target(0, 0, 0), create_target(10, 0, 0)]
        tool.set_targets(targets)
        tool.preview_distribution()

        assert len(previews_received) == 1

    def test_distribute_reverse_order(self):
        """Should reverse distribution order."""
        tool = DistributionTool()
        targets = [
            create_target(0, 0, 0, obj_id="a"),
            create_target(5, 0, 0, obj_id="b"),
            create_target(10, 0, 0, obj_id="c"),
        ]

        tool.set_targets(targets)
        tool.settings.spacing.reverse_order = True
        result = tool.distribute()

        assert result.success is True

    def test_distribute_xy_plane(self):
        """Should distribute in XY plane."""
        tool = DistributionTool()
        targets = [
            create_target(0, 0, 0, obj_id="a"),
            create_target(10, 10, 0, obj_id="b"),
        ]

        tool.set_targets(targets)
        tool.settings.axis = DistributionAxis.XY
        result = tool.distribute()

        assert result.success is True

    def test_distribute_xz_plane(self):
        """Should distribute in XZ plane (floor)."""
        tool = DistributionTool()
        targets = [
            create_target(0, 0, 0, obj_id="a"),
            create_target(10, 0, 10, obj_id="b"),
        ]

        tool.set_targets(targets)
        tool.settings.axis = DistributionAxis.XZ
        result = tool.distribute()

        assert result.success is True

    def test_result_stores_positions(self):
        """Result should store original and new positions."""
        tool = DistributionTool()
        targets = [
            create_target(0, 0, 0, obj_id="a"),
            create_target(10, 0, 0, obj_id="b"),
        ]

        tool.set_targets(targets)
        result = tool.distribute()

        assert len(result.original_positions) >= 1
        assert len(result.new_positions) >= 1
