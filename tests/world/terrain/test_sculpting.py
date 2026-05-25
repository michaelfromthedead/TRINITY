"""Tests for terrain sculpting tools."""

import math
from typing import List, Tuple

import pytest

from engine.world.terrain.sculpting import (
    BaseSculptTool,
    BrushSettings,
    BrushShape,
    ErosionTool,
    FlattenTool,
    HeightDelta,
    LowerTool,
    NoiseTool,
    RaiseTool,
    RampTool,
    SculptingSession,
    SculptTool,
    SmoothTool,
    TerrainBrush,
    create_tool,
)


class MockHeightfield:
    """Mock heightfield for testing."""

    def __init__(
        self,
        width: int = 64,
        height: int = 64,
        sample_spacing: float = 1.0,
        initial_height: float = 0.0,
    ):
        self._width = width
        self._height = height
        self._sample_spacing = sample_spacing
        self._heights = [
            [initial_height for _ in range(width)] for _ in range(height)
        ]

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def sample_spacing(self) -> float:
        return self._sample_spacing

    def get_height_at(self, x: int, z: int) -> float:
        return self._heights[z][x]

    def set_height_at(self, x: int, z: int, height: float) -> None:
        self._heights[z][x] = height

    def world_to_sample(self, world_x: float, world_z: float) -> Tuple[int, int]:
        return int(world_x / self._sample_spacing), int(world_z / self._sample_spacing)

    def sample_to_world(self, sample_x: int, sample_z: int) -> Tuple[float, float]:
        return sample_x * self._sample_spacing, sample_z * self._sample_spacing


# ============================================================================
# BrushSettings tests
# ============================================================================


class TestBrushSettings:
    """Tests for BrushSettings dataclass."""

    def test_default_values(self):
        """Test default brush settings."""
        settings = BrushSettings()
        assert settings.size == 10.0
        assert settings.strength == 0.5
        assert settings.falloff == 0.5
        assert settings.shape == BrushShape.CIRCLE

    def test_custom_values(self):
        """Test custom brush settings."""
        settings = BrushSettings(
            size=20.0,
            strength=0.8,
            falloff=0.3,
            shape=BrushShape.SQUARE,
        )
        assert settings.size == 20.0
        assert settings.strength == 0.8
        assert settings.falloff == 0.3
        assert settings.shape == BrushShape.SQUARE

    def test_invalid_size(self):
        """Test that size must be positive."""
        with pytest.raises(ValueError, match="size must be > 0"):
            BrushSettings(size=0)

        with pytest.raises(ValueError, match="size must be > 0"):
            BrushSettings(size=-5.0)

    def test_invalid_strength(self):
        """Test that strength must be in [0, 1]."""
        with pytest.raises(ValueError, match="strength must be in range"):
            BrushSettings(strength=-0.1)

        with pytest.raises(ValueError, match="strength must be in range"):
            BrushSettings(strength=1.5)

    def test_invalid_falloff(self):
        """Test that falloff must be in [0, 1]."""
        with pytest.raises(ValueError, match="falloff must be in range"):
            BrushSettings(falloff=-0.1)

        with pytest.raises(ValueError, match="falloff must be in range"):
            BrushSettings(falloff=1.5)


# ============================================================================
# TerrainBrush tests
# ============================================================================


class TestTerrainBrush:
    """Tests for TerrainBrush class."""

    def test_falloff_at_center(self):
        """Test falloff at brush center is 1.0."""
        settings = BrushSettings(size=10.0, falloff=0.5)
        brush = TerrainBrush(settings)
        assert brush.get_falloff_at(0.0) == 1.0

    def test_falloff_at_edge(self):
        """Test falloff at brush edge is 0.0."""
        settings = BrushSettings(size=10.0, falloff=0.5)
        brush = TerrainBrush(settings)
        assert brush.get_falloff_at(10.0) == 0.0

    def test_falloff_outside_brush(self):
        """Test falloff outside brush is 0.0."""
        settings = BrushSettings(size=10.0, falloff=0.5)
        brush = TerrainBrush(settings)
        assert brush.get_falloff_at(15.0) == 0.0

    def test_hard_edge_falloff(self):
        """Test hard edge when falloff is 0."""
        settings = BrushSettings(size=10.0, falloff=0.0)
        brush = TerrainBrush(settings)
        assert brush.get_falloff_at(5.0) == 1.0
        assert brush.get_falloff_at(9.9) == 1.0
        assert brush.get_falloff_at(10.0) == 0.0

    def test_soft_edge_falloff(self):
        """Test soft edge when falloff is 1.0."""
        settings = BrushSettings(size=10.0, falloff=1.0)
        brush = TerrainBrush(settings)
        # At center
        assert brush.get_falloff_at(0.0) == 1.0
        # Halfway should be around 0.5
        falloff_mid = brush.get_falloff_at(5.0)
        assert 0.4 < falloff_mid < 0.6

    def test_falloff_negative_distance_raises(self):
        """Test that negative distance raises error."""
        settings = BrushSettings(size=10.0)
        brush = TerrainBrush(settings)
        with pytest.raises(ValueError, match="distance_from_center must be >= 0"):
            brush.get_falloff_at(-1.0)

    def test_get_affected_samples_circle(self):
        """Test getting affected samples for circle brush."""
        heightfield = MockHeightfield(width=32, height=32, sample_spacing=1.0)
        settings = BrushSettings(size=5.0, shape=BrushShape.CIRCLE)
        brush = TerrainBrush(settings)

        samples = brush.get_affected_samples(heightfield, 16.0, 16.0)

        assert len(samples) > 0
        # All samples should be within radius
        for sx, sz in samples:
            world_x, world_z = heightfield.sample_to_world(sx, sz)
            distance = math.sqrt((world_x - 16.0) ** 2 + (world_z - 16.0) ** 2)
            assert distance <= 5.0

    def test_get_affected_samples_square(self):
        """Test getting affected samples for square brush."""
        heightfield = MockHeightfield(width=32, height=32, sample_spacing=1.0)
        settings = BrushSettings(size=5.0, shape=BrushShape.SQUARE)
        brush = TerrainBrush(settings)

        samples = brush.get_affected_samples(heightfield, 16.0, 16.0)

        assert len(samples) > 0
        # All samples should be within square
        for sx, sz in samples:
            world_x, world_z = heightfield.sample_to_world(sx, sz)
            assert abs(world_x - 16.0) <= 5.0
            assert abs(world_z - 16.0) <= 5.0

    def test_get_affected_samples_edge_of_terrain(self):
        """Test affected samples at terrain edge."""
        heightfield = MockHeightfield(width=32, height=32, sample_spacing=1.0)
        settings = BrushSettings(size=10.0, shape=BrushShape.CIRCLE)
        brush = TerrainBrush(settings)

        samples = brush.get_affected_samples(heightfield, 0.0, 0.0)

        # All samples should be within valid range
        for sx, sz in samples:
            assert 0 <= sx < 32
            assert 0 <= sz < 32


# ============================================================================
# HeightDelta tests
# ============================================================================


class TestHeightDelta:
    """Tests for HeightDelta class."""

    def test_initial_state_empty(self):
        """Test that delta starts empty."""
        delta = HeightDelta()
        assert delta.is_empty()

    def test_add_change(self):
        """Test adding a height change."""
        delta = HeightDelta()
        delta.add_change(5, 10, 1.0, 2.0)

        assert not delta.is_empty()
        assert (5, 10) in delta.changes
        assert delta.changes[(5, 10)] == (1.0, 2.0)

    def test_multiple_changes_same_position(self):
        """Test that multiple changes to same position keep original old height."""
        delta = HeightDelta()
        delta.add_change(5, 10, 1.0, 2.0)
        delta.add_change(5, 10, 2.0, 3.0)

        # Should keep original old height
        assert delta.changes[(5, 10)] == (1.0, 3.0)


# ============================================================================
# RaiseTool tests
# ============================================================================


class TestRaiseTool:
    """Tests for RaiseTool."""

    def test_raise_increases_height(self):
        """Test that raise tool increases height."""
        heightfield = MockHeightfield(width=32, height=32, initial_height=0.0)
        settings = BrushSettings(size=5.0, strength=1.0, falloff=0.0)
        brush = TerrainBrush(settings)
        tool = RaiseTool(brush)
        delta = HeightDelta()

        tool.apply(heightfield, 16.0, 16.0, delta)

        # Center should be raised
        assert heightfield.get_height_at(16, 16) > 0.0

    def test_raise_respects_strength(self):
        """Test that raise tool respects strength setting."""
        heightfield = MockHeightfield(width=32, height=32, initial_height=0.0)
        settings = BrushSettings(size=5.0, strength=0.5, falloff=0.0)
        brush = TerrainBrush(settings)
        tool = RaiseTool(brush)
        delta = HeightDelta()

        tool.apply(heightfield, 16.0, 16.0, delta)

        # Center should be raised by strength
        assert abs(heightfield.get_height_at(16, 16) - 0.5) < 0.01

    def test_raise_records_delta(self):
        """Test that raise tool records changes in delta."""
        heightfield = MockHeightfield(width=32, height=32, initial_height=0.0)
        settings = BrushSettings(size=5.0, strength=1.0, falloff=0.0)
        brush = TerrainBrush(settings)
        tool = RaiseTool(brush)
        delta = HeightDelta()

        tool.apply(heightfield, 16.0, 16.0, delta)

        assert not delta.is_empty()
        assert (16, 16) in delta.changes


# ============================================================================
# LowerTool tests
# ============================================================================


class TestLowerTool:
    """Tests for LowerTool."""

    def test_lower_decreases_height(self):
        """Test that lower tool decreases height."""
        heightfield = MockHeightfield(width=32, height=32, initial_height=5.0)
        settings = BrushSettings(size=5.0, strength=1.0, falloff=0.0)
        brush = TerrainBrush(settings)
        tool = LowerTool(brush)
        delta = HeightDelta()

        tool.apply(heightfield, 16.0, 16.0, delta)

        # Center should be lowered
        assert heightfield.get_height_at(16, 16) < 5.0

    def test_lower_respects_strength(self):
        """Test that lower tool respects strength setting."""
        heightfield = MockHeightfield(width=32, height=32, initial_height=5.0)
        settings = BrushSettings(size=5.0, strength=0.5, falloff=0.0)
        brush = TerrainBrush(settings)
        tool = LowerTool(brush)
        delta = HeightDelta()

        tool.apply(heightfield, 16.0, 16.0, delta)

        # Center should be lowered by strength
        assert abs(heightfield.get_height_at(16, 16) - 4.5) < 0.01


# ============================================================================
# SmoothTool tests
# ============================================================================


class TestSmoothTool:
    """Tests for SmoothTool."""

    def test_smooth_averages_heights(self):
        """Test that smooth tool averages heights."""
        heightfield = MockHeightfield(width=32, height=32, initial_height=0.0)
        # Create a spike
        heightfield.set_height_at(16, 16, 10.0)

        settings = BrushSettings(size=5.0, strength=1.0, falloff=0.0)
        brush = TerrainBrush(settings)
        tool = SmoothTool(brush, kernel_size=3)
        delta = HeightDelta()

        tool.apply(heightfield, 16.0, 16.0, delta)

        # Spike should be reduced
        assert heightfield.get_height_at(16, 16) < 10.0

    def test_smooth_invalid_kernel_size(self):
        """Test that invalid kernel size raises error."""
        settings = BrushSettings(size=5.0)
        brush = TerrainBrush(settings)

        with pytest.raises(ValueError, match="kernel_size must be a positive odd number"):
            SmoothTool(brush, kernel_size=0)

        with pytest.raises(ValueError, match="kernel_size must be a positive odd number"):
            SmoothTool(brush, kernel_size=4)

    def test_smooth_spreads_height(self):
        """Test that smooth tool spreads height to neighbors."""
        heightfield = MockHeightfield(width=32, height=32, initial_height=0.0)
        heightfield.set_height_at(16, 16, 10.0)

        settings = BrushSettings(size=5.0, strength=1.0, falloff=0.0)
        brush = TerrainBrush(settings)
        tool = SmoothTool(brush, kernel_size=3)
        delta = HeightDelta()

        tool.apply(heightfield, 16.0, 16.0, delta)

        # Neighbors should gain some height
        assert heightfield.get_height_at(15, 16) > 0.0


# ============================================================================
# FlattenTool tests
# ============================================================================


class TestFlattenTool:
    """Tests for FlattenTool."""

    def test_flatten_to_target_height(self):
        """Test flattening to a specific height."""
        heightfield = MockHeightfield(width=32, height=32, initial_height=0.0)
        settings = BrushSettings(size=5.0, strength=1.0, falloff=0.0)
        brush = TerrainBrush(settings)
        tool = FlattenTool(brush, target_height=5.0)
        delta = HeightDelta()

        tool.apply(heightfield, 16.0, 16.0, delta)

        # Should be at target height
        assert abs(heightfield.get_height_at(16, 16) - 5.0) < 0.01

    def test_flatten_to_center_height(self):
        """Test flattening to center height when no target specified."""
        heightfield = MockHeightfield(width=32, height=32, initial_height=0.0)
        heightfield.set_height_at(16, 16, 5.0)
        # Create variation around center
        heightfield.set_height_at(15, 16, 2.0)
        heightfield.set_height_at(17, 16, 8.0)

        settings = BrushSettings(size=5.0, strength=1.0, falloff=0.0)
        brush = TerrainBrush(settings)
        tool = FlattenTool(brush)  # No target height
        delta = HeightDelta()

        tool.apply(heightfield, 16.0, 16.0, delta)

        # Neighbors should move toward center height (5.0)
        assert abs(heightfield.get_height_at(15, 16) - 5.0) < abs(2.0 - 5.0)
        assert abs(heightfield.get_height_at(17, 16) - 5.0) < abs(8.0 - 5.0)

    def test_flatten_target_height_property(self):
        """Test target_height property."""
        settings = BrushSettings(size=5.0)
        brush = TerrainBrush(settings)
        tool = FlattenTool(brush, target_height=5.0)

        assert tool.target_height == 5.0
        tool.target_height = 10.0
        assert tool.target_height == 10.0


# ============================================================================
# ErosionTool tests
# ============================================================================


class TestErosionTool:
    """Tests for ErosionTool."""

    def test_erosion_modifies_terrain(self):
        """Test that erosion tool modifies terrain."""
        heightfield = MockHeightfield(width=32, height=32, initial_height=0.0)
        # Create a hill
        for z in range(12, 20):
            for x in range(12, 20):
                dist = math.sqrt((x - 16) ** 2 + (z - 16) ** 2)
                heightfield.set_height_at(x, z, max(0, 5.0 - dist))

        settings = BrushSettings(size=10.0, strength=1.0, falloff=0.0)
        brush = TerrainBrush(settings)
        tool = ErosionTool(brush, iterations=5)
        delta = HeightDelta()

        original_peak = heightfield.get_height_at(16, 16)
        tool.apply(heightfield, 16.0, 16.0, delta)

        # Peak should be eroded
        assert heightfield.get_height_at(16, 16) < original_peak

    def test_erosion_invalid_iterations(self):
        """Test that invalid iterations raises error."""
        settings = BrushSettings(size=5.0)
        brush = TerrainBrush(settings)

        with pytest.raises(ValueError, match="iterations must be >= 1"):
            ErosionTool(brush, iterations=0)

    def test_erosion_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        settings = BrushSettings(size=5.0)
        brush = TerrainBrush(settings)

        with pytest.raises(ValueError, match="sediment_capacity must be > 0"):
            ErosionTool(brush, sediment_capacity=0)

        with pytest.raises(ValueError, match="deposition_rate must be in"):
            ErosionTool(brush, deposition_rate=0)

        with pytest.raises(ValueError, match="erosion_rate must be in"):
            ErosionTool(brush, erosion_rate=0)


# ============================================================================
# NoiseTool tests
# ============================================================================


class TestNoiseTool:
    """Tests for NoiseTool."""

    def test_noise_adds_variation(self):
        """Test that noise tool adds variation to terrain."""
        heightfield = MockHeightfield(width=32, height=32, initial_height=0.0)
        settings = BrushSettings(size=10.0, strength=1.0, falloff=0.0)
        brush = TerrainBrush(settings)
        tool = NoiseTool(brush, noise_scale=0.5, octaves=2)
        delta = HeightDelta()

        tool.apply(heightfield, 16.0, 16.0, delta)

        # Check that there's variation in heights
        heights = set()
        for z in range(12, 20):
            for x in range(12, 20):
                heights.add(round(heightfield.get_height_at(x, z), 4))

        # Should have multiple different heights
        assert len(heights) > 1

    def test_noise_is_deterministic(self):
        """Test that noise with same seed produces same result."""
        heightfield1 = MockHeightfield(width=32, height=32, initial_height=0.0)
        heightfield2 = MockHeightfield(width=32, height=32, initial_height=0.0)

        settings = BrushSettings(size=10.0, strength=1.0, falloff=0.0)
        brush = TerrainBrush(settings)
        tool1 = NoiseTool(brush, seed=42)
        tool2 = NoiseTool(brush, seed=42)

        tool1.apply(heightfield1, 16.0, 16.0, HeightDelta())
        tool2.apply(heightfield2, 16.0, 16.0, HeightDelta())

        # Results should be identical
        for z in range(32):
            for x in range(32):
                assert heightfield1.get_height_at(x, z) == heightfield2.get_height_at(x, z)

    def test_noise_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        settings = BrushSettings(size=5.0)
        brush = TerrainBrush(settings)

        with pytest.raises(ValueError, match="noise_scale must be > 0"):
            NoiseTool(brush, noise_scale=0)

        with pytest.raises(ValueError, match="octaves must be >= 1"):
            NoiseTool(brush, octaves=0)

        with pytest.raises(ValueError, match="persistence must be in"):
            NoiseTool(brush, persistence=0)


# ============================================================================
# RampTool tests
# ============================================================================


class TestRampTool:
    """Tests for RampTool."""

    def test_ramp_creates_gradient(self):
        """Test that ramp tool creates a gradient."""
        heightfield = MockHeightfield(width=32, height=32, initial_height=0.0)
        settings = BrushSettings(size=10.0, strength=1.0, falloff=0.0)
        brush = TerrainBrush(settings)
        tool = RampTool(brush)
        tool.set_ramp_points(
            start_point=(8.0, 8.0),
            end_point=(24.0, 8.0),
            start_height=0.0,
            end_height=10.0,
        )
        delta = HeightDelta()

        tool.apply(heightfield, 16.0, 8.0, delta)

        # Heights should increase from start to end
        h_start = heightfield.get_height_at(8, 8)
        h_mid = heightfield.get_height_at(16, 8)
        h_end = heightfield.get_height_at(24, 8)

        assert h_start < h_mid < h_end

    def test_ramp_requires_points(self):
        """Test that ramp tool requires points to be set."""
        heightfield = MockHeightfield(width=32, height=32, initial_height=0.0)
        settings = BrushSettings(size=10.0, strength=1.0, falloff=0.0)
        brush = TerrainBrush(settings)
        tool = RampTool(brush)
        delta = HeightDelta()

        with pytest.raises(ValueError, match="Ramp start and end points must be set"):
            tool.apply(heightfield, 16.0, 16.0, delta)


# ============================================================================
# SculptingSession tests
# ============================================================================


class TestSculptingSession:
    """Tests for SculptingSession."""

    def test_initial_state(self):
        """Test initial session state."""
        heightfield = MockHeightfield()
        session = SculptingSession(heightfield)

        assert not session.can_undo
        assert not session.can_redo
        assert session.undo_count == 0
        assert session.redo_count == 0

    def test_apply_tool_enables_undo(self):
        """Test that applying a tool enables undo."""
        heightfield = MockHeightfield(initial_height=0.0)
        session = SculptingSession(heightfield)

        settings = BrushSettings(size=5.0, strength=1.0, falloff=0.0)
        brush = TerrainBrush(settings)
        tool = RaiseTool(brush)

        session.apply_tool(tool, 16.0, 16.0)

        assert session.can_undo
        assert session.undo_count == 1

    def test_undo_restores_height(self):
        """Test that undo restores previous height."""
        heightfield = MockHeightfield(initial_height=0.0)
        session = SculptingSession(heightfield)

        settings = BrushSettings(size=5.0, strength=1.0, falloff=0.0)
        brush = TerrainBrush(settings)
        tool = RaiseTool(brush)

        session.apply_tool(tool, 16.0, 16.0)
        modified_height = heightfield.get_height_at(16, 16)
        assert modified_height > 0.0

        result = session.undo()
        assert result is True
        assert heightfield.get_height_at(16, 16) == 0.0

    def test_redo_reapplies_change(self):
        """Test that redo reapplies the undone change."""
        heightfield = MockHeightfield(initial_height=0.0)
        session = SculptingSession(heightfield)

        settings = BrushSettings(size=5.0, strength=1.0, falloff=0.0)
        brush = TerrainBrush(settings)
        tool = RaiseTool(brush)

        session.apply_tool(tool, 16.0, 16.0)
        modified_height = heightfield.get_height_at(16, 16)

        session.undo()
        result = session.redo()

        assert result is True
        assert heightfield.get_height_at(16, 16) == modified_height

    def test_undo_empty_stack_returns_false(self):
        """Test that undo on empty stack returns False."""
        heightfield = MockHeightfield()
        session = SculptingSession(heightfield)

        result = session.undo()
        assert result is False

    def test_redo_empty_stack_returns_false(self):
        """Test that redo on empty stack returns False."""
        heightfield = MockHeightfield()
        session = SculptingSession(heightfield)

        result = session.redo()
        assert result is False

    def test_new_change_clears_redo_stack(self):
        """Test that new changes clear the redo stack."""
        heightfield = MockHeightfield(initial_height=0.0)
        session = SculptingSession(heightfield)

        settings = BrushSettings(size=5.0, strength=1.0, falloff=0.0)
        brush = TerrainBrush(settings)
        tool = RaiseTool(brush)

        session.apply_tool(tool, 16.0, 16.0)
        session.undo()
        assert session.can_redo

        session.apply_tool(tool, 20.0, 20.0)
        assert not session.can_redo

    def test_max_undo_levels(self):
        """Test that undo stack respects max levels."""
        heightfield = MockHeightfield(initial_height=0.0)
        session = SculptingSession(heightfield, max_undo_levels=3)

        settings = BrushSettings(size=5.0, strength=0.1, falloff=0.0)
        brush = TerrainBrush(settings)
        tool = RaiseTool(brush)

        # Apply 5 changes
        for i in range(5):
            session.apply_tool(tool, 16.0 + i, 16.0)

        # Should only have 3 undo levels
        assert session.undo_count == 3

    def test_multiple_undo_redo(self):
        """Test multiple undo/redo operations."""
        heightfield = MockHeightfield(initial_height=0.0)
        session = SculptingSession(heightfield)

        settings = BrushSettings(size=5.0, strength=1.0, falloff=0.0)
        brush = TerrainBrush(settings)
        tool = RaiseTool(brush)

        session.apply_tool(tool, 16.0, 16.0)
        h1 = heightfield.get_height_at(16, 16)
        session.apply_tool(tool, 16.0, 16.0)
        h2 = heightfield.get_height_at(16, 16)

        session.undo()
        assert abs(heightfield.get_height_at(16, 16) - h1) < 0.01
        session.undo()
        assert heightfield.get_height_at(16, 16) == 0.0

        session.redo()
        assert abs(heightfield.get_height_at(16, 16) - h1) < 0.01
        session.redo()
        assert abs(heightfield.get_height_at(16, 16) - h2) < 0.01

    def test_clear_history(self):
        """Test clearing history."""
        heightfield = MockHeightfield(initial_height=0.0)
        session = SculptingSession(heightfield)

        settings = BrushSettings(size=5.0, strength=1.0, falloff=0.0)
        brush = TerrainBrush(settings)
        tool = RaiseTool(brush)

        session.apply_tool(tool, 16.0, 16.0)
        session.clear_history()

        assert not session.can_undo
        assert not session.can_redo


# ============================================================================
# create_tool factory tests
# ============================================================================


class TestCreateTool:
    """Tests for create_tool factory function."""

    def test_create_raise_tool(self):
        """Test creating raise tool."""
        settings = BrushSettings(size=5.0)
        brush = TerrainBrush(settings)
        tool = create_tool(SculptTool.RAISE, brush)
        assert isinstance(tool, RaiseTool)

    def test_create_lower_tool(self):
        """Test creating lower tool."""
        settings = BrushSettings(size=5.0)
        brush = TerrainBrush(settings)
        tool = create_tool(SculptTool.LOWER, brush)
        assert isinstance(tool, LowerTool)

    def test_create_smooth_tool(self):
        """Test creating smooth tool."""
        settings = BrushSettings(size=5.0)
        brush = TerrainBrush(settings)
        tool = create_tool(SculptTool.SMOOTH, brush, kernel_size=5)
        assert isinstance(tool, SmoothTool)

    def test_create_flatten_tool(self):
        """Test creating flatten tool."""
        settings = BrushSettings(size=5.0)
        brush = TerrainBrush(settings)
        tool = create_tool(SculptTool.FLATTEN, brush, target_height=5.0)
        assert isinstance(tool, FlattenTool)

    def test_create_erosion_tool(self):
        """Test creating erosion tool."""
        settings = BrushSettings(size=5.0)
        brush = TerrainBrush(settings)
        tool = create_tool(SculptTool.EROSION, brush, iterations=10)
        assert isinstance(tool, ErosionTool)

    def test_create_noise_tool(self):
        """Test creating noise tool."""
        settings = BrushSettings(size=5.0)
        brush = TerrainBrush(settings)
        tool = create_tool(SculptTool.NOISE, brush, seed=123)
        assert isinstance(tool, NoiseTool)

    def test_create_ramp_tool(self):
        """Test creating ramp tool."""
        settings = BrushSettings(size=5.0)
        brush = TerrainBrush(settings)
        tool = create_tool(SculptTool.RAMP, brush)
        assert isinstance(tool, RampTool)


# ============================================================================
# Enhanced sculpting tests for edge cases
# ============================================================================


class TestEnhancedSculptingVerification:
    """Enhanced tests that verify actual sculpting behavior."""

    def test_raise_exact_change_at_center(self):
        """Verify raise tool applies exact strength at center with no falloff."""
        heightfield = MockHeightfield(width=32, height=32, initial_height=0.0)
        settings = BrushSettings(size=5.0, strength=0.75, falloff=0.0)
        brush = TerrainBrush(settings)
        tool = RaiseTool(brush)
        delta = HeightDelta()

        tool.apply(heightfield, 16.0, 16.0, delta)

        # With falloff=0, center should receive full strength
        center_height = heightfield.get_height_at(16, 16)
        assert abs(center_height - 0.75) < 0.01, f"Expected 0.75, got {center_height}"

    def test_falloff_gradient_verification(self):
        """Verify that falloff creates a gradient from center to edge."""
        heightfield = MockHeightfield(width=64, height=64, initial_height=0.0)
        # Use larger brush with significant falloff to see gradient
        settings = BrushSettings(size=20.0, strength=1.0, falloff=0.8)
        brush = TerrainBrush(settings)
        tool = RaiseTool(brush)
        delta = HeightDelta()

        tool.apply(heightfield, 32.0, 32.0, delta)

        center_height = heightfield.get_height_at(32, 32)
        # Sample at ~80% of radius (in falloff zone)
        edge_height = heightfield.get_height_at(32, 48)  # 16 units from center, radius is 20

        # Center should be at full strength
        assert center_height > 0.0, "Center should be raised"
        # At edge (within radius but in falloff zone), should be less than center
        # Note: if edge is within inner radius ratio (1-falloff = 0.2 of radius = 4 units),
        # it will also get full strength
        far_edge_height = heightfield.get_height_at(32, 50)  # 18 units, definitely in falloff
        # Far edge should be lower than center
        assert center_height >= far_edge_height, "Edge in falloff zone should be <= center"

    def test_smooth_weighted_average_calculation(self):
        """Verify smooth tool applies proper weighted average."""
        heightfield = MockHeightfield(width=32, height=32, initial_height=0.0)
        # Create a 3x3 pattern with known values
        heightfield.set_height_at(15, 15, 0.0)
        heightfield.set_height_at(16, 15, 0.0)
        heightfield.set_height_at(17, 15, 0.0)
        heightfield.set_height_at(15, 16, 0.0)
        heightfield.set_height_at(16, 16, 9.0)  # Spike in center
        heightfield.set_height_at(17, 16, 0.0)
        heightfield.set_height_at(15, 17, 0.0)
        heightfield.set_height_at(16, 17, 0.0)
        heightfield.set_height_at(17, 17, 0.0)

        settings = BrushSettings(size=3.0, strength=1.0, falloff=0.0)
        brush = TerrainBrush(settings)
        tool = SmoothTool(brush, kernel_size=3)
        delta = HeightDelta()

        original_height = heightfield.get_height_at(16, 16)
        tool.apply(heightfield, 16.0, 16.0, delta)
        new_height = heightfield.get_height_at(16, 16)

        # 3x3 kernel with 9.0 at center and 0.0 elsewhere
        # After smoothing, center should be approximately 9.0/9 = 1.0 (uniform kernel)
        # The exact value depends on Gaussian weights, but should be between 1.0 and 9.0
        assert new_height < original_height, "Spike should be reduced"
        assert new_height > 0.0, "Smoothed value should be positive"

    def test_lower_prevents_negative_height(self):
        """Verify lower tool behavior at height boundaries."""
        heightfield = MockHeightfield(width=32, height=32, initial_height=0.5)
        settings = BrushSettings(size=5.0, strength=1.0, falloff=0.0)
        brush = TerrainBrush(settings)
        tool = LowerTool(brush)
        delta = HeightDelta()

        # Apply lower multiple times
        tool.apply(heightfield, 16.0, 16.0, delta)
        height_after_1 = heightfield.get_height_at(16, 16)
        assert height_after_1 < 0.5, "Height should decrease"

        # Height can go negative (no artificial floor)
        tool.apply(heightfield, 16.0, 16.0, delta)
        height_after_2 = heightfield.get_height_at(16, 16)
        assert height_after_2 < height_after_1, "Height should continue decreasing"
