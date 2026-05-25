"""Tests for terrain sculpting tools."""

import pytest
import math
from engine.tooling.terrain.sculpt_tools import (
    SculptMode,
    BrushShape,
    FalloffCurve,
    BrushSettings,
    TerrainBrush,
    SculptOperation,
    TerrainData,
    TerrainSculptTool,
)


class TestBrushSettings:
    """Tests for BrushSettings."""

    def test_default_settings(self):
        """Test default brush settings."""
        settings = BrushSettings()
        assert settings.size == 10.0
        assert settings.strength == 0.5
        assert settings.falloff == 0.5
        assert settings.shape == BrushShape.CIRCLE
        assert settings.falloff_curve == FalloffCurve.SMOOTH

    def test_custom_settings(self):
        """Test custom brush settings."""
        settings = BrushSettings(
            size=20.0,
            strength=0.8,
            falloff=0.3,
            shape=BrushShape.SQUARE,
            falloff_curve=FalloffCurve.LINEAR,
        )
        assert settings.size == 20.0
        assert settings.strength == 0.8
        assert settings.falloff == 0.3
        assert settings.shape == BrushShape.SQUARE


class TestTerrainBrush:
    """Tests for TerrainBrush."""

    def test_circular_brush_center(self):
        """Test brush influence at center."""
        brush = TerrainBrush()
        influence = brush.get_influence(5.0, 5.0, 5.0, 5.0)
        assert influence == brush.settings.strength

    def test_circular_brush_outside(self):
        """Test brush influence outside radius."""
        brush = TerrainBrush(settings=BrushSettings(size=10.0))
        influence = brush.get_influence(0.0, 0.0, 20.0, 20.0)
        assert influence == 0.0

    def test_square_brush_inside(self):
        """Test square brush inside area."""
        brush = TerrainBrush(settings=BrushSettings(shape=BrushShape.SQUARE, size=10.0))
        assert brush.is_in_brush(5.0, 5.0, 5.0, 5.0)
        assert brush.is_in_brush(2.0, 2.0, 5.0, 5.0)

    def test_square_brush_outside(self):
        """Test square brush outside area."""
        brush = TerrainBrush(settings=BrushSettings(shape=BrushShape.SQUARE, size=10.0))
        assert not brush.is_in_brush(20.0, 20.0, 5.0, 5.0)

    def test_falloff_linear(self):
        """Test linear falloff curve."""
        brush = TerrainBrush(settings=BrushSettings(
            falloff_curve=FalloffCurve.LINEAR,
            falloff=1.0,
        ))
        # At max distance, falloff should be 0
        falloff = brush.get_falloff(5.0, 5.0)
        assert falloff == 0.0
        # At center, falloff should be 1
        falloff = brush.get_falloff(0.0, 5.0)
        assert falloff == 1.0

    def test_falloff_smooth(self):
        """Test smooth falloff curve."""
        brush = TerrainBrush(settings=BrushSettings(
            falloff_curve=FalloffCurve.SMOOTH,
            falloff=1.0,
        ))
        falloff = brush.get_falloff(2.5, 5.0)
        assert 0.0 < falloff < 1.0

    def test_falloff_sphere(self):
        """Test sphere falloff curve."""
        brush = TerrainBrush(settings=BrushSettings(
            falloff_curve=FalloffCurve.SPHERE,
            falloff=1.0,
        ))
        falloff = brush.get_falloff(0.0, 5.0)
        assert falloff == 1.0

    def test_falloff_tip(self):
        """Test tip falloff curve."""
        brush = TerrainBrush(settings=BrushSettings(
            falloff_curve=FalloffCurve.TIP,
            falloff=1.0,
        ))
        falloff = brush.get_falloff(5.0, 5.0)
        assert falloff == 0.0

    def test_falloff_flat(self):
        """Test flat falloff curve."""
        brush = TerrainBrush(settings=BrushSettings(
            falloff_curve=FalloffCurve.FLAT,
            falloff=1.0,
        ))
        # Flat falloff should be 1 until 90% of radius
        falloff = brush.get_falloff(4.0, 5.0)
        assert falloff == 1.0


class TestTerrainData:
    """Tests for TerrainData."""

    def test_creation(self):
        """Test terrain data creation."""
        terrain = TerrainData(64, 64)
        assert terrain.width == 64
        assert terrain.height == 64

    def test_get_height(self):
        """Test getting height values."""
        terrain = TerrainData(64, 64)
        assert terrain.get_height(0, 0) == 0.0
        assert terrain.get_height(32, 32) == 0.0

    def test_set_height(self):
        """Test setting height values."""
        terrain = TerrainData(64, 64)
        terrain.set_height(10, 10, 0.5)
        assert terrain.get_height(10, 10) == 0.5

    def test_out_of_bounds(self):
        """Test out of bounds access."""
        terrain = TerrainData(64, 64)
        assert terrain.get_height(-1, 0) == 0.0
        assert terrain.get_height(100, 100) == 0.0

    def test_dirty_chunks(self):
        """Test dirty chunk tracking."""
        terrain = TerrainData(64, 64, chunk_size=16)
        terrain.set_height(10, 10, 0.5)
        dirty = terrain.get_dirty_chunks()
        assert (0, 0) in dirty

    def test_clear_dirty_chunks(self):
        """Test clearing dirty chunks."""
        terrain = TerrainData(64, 64)
        terrain.set_height(10, 10, 0.5)
        terrain.clear_dirty_chunks()
        assert len(terrain.get_dirty_chunks()) == 0

    def test_average_height(self):
        """Test average height calculation."""
        terrain = TerrainData(64, 64)
        for y in range(5, 10):
            for x in range(5, 10):
                terrain.set_height(x, y, 1.0)

        avg = terrain.get_average_height(7, 7, 2)
        assert avg > 0.0


class TestTerrainSculptTool:
    """Tests for TerrainSculptTool."""

    def setup_method(self):
        """Set up test fixtures."""
        self.terrain = TerrainData(64, 64)
        self.tool = TerrainSculptTool(self.terrain)

    def test_creation(self):
        """Test sculpt tool creation."""
        assert self.tool.terrain == self.terrain
        assert self.tool.mode == SculptMode.RAISE

    def test_mode_change(self):
        """Test changing sculpt mode."""
        self.tool.mode = SculptMode.LOWER
        assert self.tool.mode == SculptMode.LOWER

    def test_raise_mode(self):
        """Test raise sculpt mode."""
        self.tool.mode = SculptMode.RAISE
        self.tool.apply(32.0, 32.0)
        assert self.terrain.get_height(32, 32) > 0.0

    def test_lower_mode(self):
        """Test lower sculpt mode."""
        self.terrain.set_height(32, 32, 1.0)
        self.tool.mode = SculptMode.LOWER
        self.tool.apply(32.0, 32.0)
        assert self.terrain.get_height(32, 32) < 1.0

    def test_smooth_mode(self):
        """Test smooth sculpt mode."""
        # Create a spike
        self.terrain.set_height(32, 32, 1.0)
        self.tool.mode = SculptMode.SMOOTH
        self.tool.apply(32.0, 32.0)
        # Spike should be reduced
        assert self.terrain.get_height(32, 32) < 1.0

    def test_flatten_mode(self):
        """Test flatten sculpt mode."""
        self.terrain.set_height(32, 32, 1.0)
        self.tool.mode = SculptMode.FLATTEN
        self.tool.apply(32.0, 32.0)

    def test_noise_mode(self):
        """Test noise sculpt mode."""
        self.tool.mode = SculptMode.NOISE
        self.tool.set_noise_params(42, 1.0)
        self.tool.apply(32.0, 32.0)
        # Should have modified heights

    def test_level_mode(self):
        """Test level sculpt mode."""
        self.terrain.set_height(32, 32, 0.0)
        self.tool.mode = SculptMode.LEVEL
        self.tool.set_level_height(0.5)
        self.tool.apply(32.0, 32.0)
        assert self.terrain.get_height(32, 32) > 0.0

    def test_stamp_mode(self):
        """Test stamp sculpt mode."""
        stamp_data = [[0.5, 0.5], [0.5, 0.5]]
        self.tool.set_stamp_data(stamp_data)
        self.tool.mode = SculptMode.STAMP
        self.tool.apply(32.0, 32.0)

    def test_undo(self):
        """Test undo operation."""
        original = self.terrain.get_height(32, 32)
        self.tool.mode = SculptMode.RAISE
        self.tool.apply(32.0, 32.0)
        raised = self.terrain.get_height(32, 32)
        assert raised > original

        self.tool.undo()
        assert self.terrain.get_height(32, 32) == original

    def test_redo(self):
        """Test redo operation."""
        self.tool.mode = SculptMode.RAISE
        self.tool.apply(32.0, 32.0)
        raised = self.terrain.get_height(32, 32)

        self.tool.undo()
        self.tool.redo()
        assert self.terrain.get_height(32, 32) == raised

    def test_can_undo(self):
        """Test can_undo check."""
        assert not self.tool.can_undo()
        self.tool.apply(32.0, 32.0)
        assert self.tool.can_undo()

    def test_can_redo(self):
        """Test can_redo check."""
        assert not self.tool.can_redo()
        self.tool.apply(32.0, 32.0)
        self.tool.undo()
        assert self.tool.can_redo()

    def test_clear_history(self):
        """Test clearing history."""
        self.tool.apply(32.0, 32.0)
        self.tool.clear_history()
        assert not self.tool.can_undo()

    def test_set_brush(self):
        """Test setting custom brush."""
        brush = TerrainBrush(settings=BrushSettings(size=20.0))
        self.tool.set_brush(brush)
        assert self.tool.brush.settings.size == 20.0

    def test_dirty_chunks_after_apply(self):
        """Test dirty chunk tracking after apply."""
        self.tool.apply(32.0, 32.0)
        dirty = self.tool.get_dirty_chunks()
        assert len(dirty) > 0
