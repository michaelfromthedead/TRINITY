"""Tests for Radiance Cache System (T-GIR-P2.7).

This module provides comprehensive tests for the radiance cache system
including grid operations, temporal accumulation, SH evaluation,
GPU buffer packing, and integration with probe systems.

Test Categories:
    - RadianceCacheConfig: Configuration validation and presets
    - CacheCell: SH storage, accumulation, state transitions
    - CacheGrid: 3D grid operations, coordinate transforms
    - RadianceCache: Temporal updates, sampling, invalidation
    - RadianceCacheUpdater: Shader dispatch, WGSL generation
    - Integration: End-to-end cache workflows
"""

from __future__ import annotations

import math
import struct
from typing import List, Tuple

import pytest

from engine.core.math.geometry import AABB
from engine.core.math.vec import Vec3
from engine.rendering.gi.radiance_cache import (
    # Constants
    DEFAULT_GRID_WIDTH,
    DEFAULT_GRID_HEIGHT,
    DEFAULT_GRID_DEPTH,
    DEFAULT_BLEND_FACTOR,
    DEFAULT_HYSTERESIS_FRAMES,
    DEFAULT_VARIANCE_THRESHOLD,
    SH_L1_COEFFICIENTS,
    GPU_ALIGNMENT,
    # Enums
    CacheUpdateMode,
    CacheQuality,
    CellState,
    # Classes
    RadianceCacheConfig,
    CacheCell,
    CacheGrid,
    RadianceCache,
    RadianceCacheUpdater,
    # Functions
    estimate_cache_memory,
    recommend_cache_quality,
    create_constant_probe_sampler,
    create_gradient_probe_sampler,
    generate_radiance_cache_sampling_wgsl,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def unit_bounds() -> AABB:
    """Unit cube centered at origin."""
    return AABB(Vec3(-1.0, -1.0, -1.0), Vec3(1.0, 1.0, 1.0))


@pytest.fixture
def large_bounds() -> AABB:
    """Large 100m cube centered at origin."""
    return AABB(Vec3(-50.0, -50.0, -50.0), Vec3(50.0, 50.0, 50.0))


@pytest.fixture
def default_config() -> RadianceCacheConfig:
    """Default cache configuration."""
    return RadianceCacheConfig()


@pytest.fixture
def small_config() -> RadianceCacheConfig:
    """Small grid for fast tests."""
    return RadianceCacheConfig(width=4, height=4, depth=4)


@pytest.fixture
def small_grid(unit_bounds: AABB, small_config: RadianceCacheConfig) -> CacheGrid:
    """Small cache grid for testing."""
    return CacheGrid(bounds=unit_bounds, config=small_config)


@pytest.fixture
def default_cell() -> CacheCell:
    """Default empty cache cell."""
    return CacheCell()


@pytest.fixture
def small_cache(unit_bounds: AABB, small_config: RadianceCacheConfig) -> RadianceCache:
    """Small radiance cache for testing."""
    return RadianceCache.create(unit_bounds, small_config)


# ============================================================================
# Constants Tests
# ============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_default_grid_dimensions(self):
        """Default grid is 64x64x32."""
        assert DEFAULT_GRID_WIDTH == 64
        assert DEFAULT_GRID_HEIGHT == 64
        assert DEFAULT_GRID_DEPTH == 32

    def test_default_blend_factor(self):
        """Default blend factor is 5%."""
        assert DEFAULT_BLEND_FACTOR == 0.05

    def test_default_hysteresis_frames(self):
        """Default hysteresis is 4 frames."""
        assert DEFAULT_HYSTERESIS_FRAMES == 4

    def test_default_variance_threshold(self):
        """Default variance threshold is 0.001."""
        assert DEFAULT_VARIANCE_THRESHOLD == 0.001

    def test_sh_l1_coefficients(self):
        """SH L1 has 4 coefficients."""
        assert SH_L1_COEFFICIENTS == 4

    def test_gpu_alignment(self):
        """GPU alignment is 16 bytes."""
        assert GPU_ALIGNMENT == 16


# ============================================================================
# CacheUpdateMode Tests
# ============================================================================


class TestCacheUpdateMode:
    """Tests for CacheUpdateMode enum."""

    def test_full_mode_exists(self):
        """FULL update mode exists."""
        assert CacheUpdateMode.FULL is not None

    def test_partial_mode_exists(self):
        """PARTIAL update mode exists."""
        assert CacheUpdateMode.PARTIAL is not None

    def test_adaptive_mode_exists(self):
        """ADAPTIVE update mode exists."""
        assert CacheUpdateMode.ADAPTIVE is not None


# ============================================================================
# CacheQuality Tests
# ============================================================================


class TestCacheQuality:
    """Tests for CacheQuality enum."""

    def test_low_quality_exists(self):
        """LOW quality preset exists."""
        assert CacheQuality.LOW is not None

    def test_medium_quality_exists(self):
        """MEDIUM quality preset exists."""
        assert CacheQuality.MEDIUM is not None

    def test_high_quality_exists(self):
        """HIGH quality preset exists."""
        assert CacheQuality.HIGH is not None

    def test_ultra_quality_exists(self):
        """ULTRA quality preset exists."""
        assert CacheQuality.ULTRA is not None

    def test_quality_ordering(self):
        """Quality levels are distinct."""
        qualities = [CacheQuality.LOW, CacheQuality.MEDIUM, CacheQuality.HIGH, CacheQuality.ULTRA]
        assert len(set(qualities)) == 4


# ============================================================================
# CellState Tests
# ============================================================================


class TestCellState:
    """Tests for CellState enum."""

    def test_empty_state_exists(self):
        """EMPTY state exists."""
        assert CellState.EMPTY is not None

    def test_accumulating_state_exists(self):
        """ACCUMULATING state exists."""
        assert CellState.ACCUMULATING is not None

    def test_stable_state_exists(self):
        """STABLE state exists."""
        assert CellState.STABLE is not None

    def test_stale_state_exists(self):
        """STALE state exists."""
        assert CellState.STALE is not None


# ============================================================================
# RadianceCacheConfig Tests
# ============================================================================


class TestRadianceCacheConfig:
    """Tests for RadianceCacheConfig."""

    def test_default_width(self, default_config: RadianceCacheConfig):
        """Default width is 64."""
        assert default_config.width == 64

    def test_default_height(self, default_config: RadianceCacheConfig):
        """Default height is 64."""
        assert default_config.height == 64

    def test_default_depth(self, default_config: RadianceCacheConfig):
        """Default depth is 32."""
        assert default_config.depth == 32

    def test_default_blend_factor(self, default_config: RadianceCacheConfig):
        """Default blend factor is 0.05."""
        assert default_config.blend_factor == 0.05

    def test_default_hysteresis_frames(self, default_config: RadianceCacheConfig):
        """Default hysteresis is 4 frames."""
        assert default_config.hysteresis_frames == 4

    def test_default_variance_threshold(self, default_config: RadianceCacheConfig):
        """Default variance threshold is 0.001."""
        assert default_config.variance_threshold == 0.001

    def test_default_update_mode(self, default_config: RadianceCacheConfig):
        """Default update mode is FULL."""
        assert default_config.update_mode == CacheUpdateMode.FULL

    def test_total_cells(self, default_config: RadianceCacheConfig):
        """Total cells is product of dimensions."""
        expected = 64 * 64 * 32
        assert default_config.total_cells == expected

    def test_small_config_total_cells(self, small_config: RadianceCacheConfig):
        """Small config has 64 cells."""
        assert small_config.total_cells == 64

    def test_cell_size_bytes(self, default_config: RadianceCacheConfig):
        """Cell size is 64 bytes (SH + metadata)."""
        # 4 SH * 3 channels * 4 bytes + 16 bytes metadata = 64 bytes
        assert default_config.cell_size_bytes == 64

    def test_total_memory_bytes(self, default_config: RadianceCacheConfig):
        """Total memory is cells * cell_size."""
        expected = default_config.total_cells * default_config.cell_size_bytes
        assert default_config.total_memory_bytes == expected

    def test_invalid_width_raises(self):
        """Width < 1 raises ValueError."""
        with pytest.raises(ValueError, match="width must be >= 1"):
            RadianceCacheConfig(width=0)

    def test_invalid_height_raises(self):
        """Height < 1 raises ValueError."""
        with pytest.raises(ValueError, match="height must be >= 1"):
            RadianceCacheConfig(height=0)

    def test_invalid_depth_raises(self):
        """Depth < 1 raises ValueError."""
        with pytest.raises(ValueError, match="depth must be >= 1"):
            RadianceCacheConfig(depth=0)

    def test_invalid_blend_factor_low_raises(self):
        """Blend factor < 0 raises ValueError."""
        with pytest.raises(ValueError, match="blend_factor must be in"):
            RadianceCacheConfig(blend_factor=-0.1)

    def test_invalid_blend_factor_high_raises(self):
        """Blend factor > 1 raises ValueError."""
        with pytest.raises(ValueError, match="blend_factor must be in"):
            RadianceCacheConfig(blend_factor=1.5)

    def test_invalid_hysteresis_raises(self):
        """Negative hysteresis raises ValueError."""
        with pytest.raises(ValueError, match="hysteresis_frames must be >= 0"):
            RadianceCacheConfig(hysteresis_frames=-1)

    def test_invalid_variance_threshold_raises(self):
        """Negative variance threshold raises ValueError."""
        with pytest.raises(ValueError, match="variance_threshold must be >= 0"):
            RadianceCacheConfig(variance_threshold=-0.001)

    def test_from_quality_low(self):
        """LOW quality preset has 32x32x16 grid."""
        config = RadianceCacheConfig.from_quality(CacheQuality.LOW)
        assert config.width == 32
        assert config.height == 32
        assert config.depth == 16

    def test_from_quality_medium(self):
        """MEDIUM quality preset has 64x64x32 grid."""
        config = RadianceCacheConfig.from_quality(CacheQuality.MEDIUM)
        assert config.width == 64
        assert config.height == 64
        assert config.depth == 32

    def test_from_quality_high(self):
        """HIGH quality preset has 128x128x64 grid."""
        config = RadianceCacheConfig.from_quality(CacheQuality.HIGH)
        assert config.width == 128
        assert config.height == 128
        assert config.depth == 64
        assert config.enable_mip_chain is True

    def test_from_quality_ultra(self):
        """ULTRA quality preset has 256x256x128 grid."""
        config = RadianceCacheConfig.from_quality(CacheQuality.ULTRA)
        assert config.width == 256
        assert config.height == 256
        assert config.depth == 128


# ============================================================================
# CacheCell Tests
# ============================================================================


class TestCacheCell:
    """Tests for CacheCell."""

    def test_initial_state_empty(self, default_cell: CacheCell):
        """New cell starts in EMPTY state."""
        assert default_cell.state == CellState.EMPTY

    def test_initial_sh_zeroed(self, default_cell: CacheCell):
        """New cell has zeroed SH coefficients."""
        assert all(c == 0.0 for c in default_cell.sh_r)
        assert all(c == 0.0 for c in default_cell.sh_g)
        assert all(c == 0.0 for c in default_cell.sh_b)

    def test_initial_frames_accumulated_zero(self, default_cell: CacheCell):
        """New cell has zero frames accumulated."""
        assert default_cell.frames_accumulated == 0

    def test_initial_variance_zero(self, default_cell: CacheCell):
        """New cell has zero variance."""
        assert default_cell.variance == 0.0

    def test_reset_clears_state(self, default_cell: CacheCell):
        """reset() returns cell to EMPTY state."""
        default_cell.state = CellState.STABLE
        default_cell.frames_accumulated = 10
        default_cell.reset()
        assert default_cell.state == CellState.EMPTY
        assert default_cell.frames_accumulated == 0

    def test_reset_clears_sh(self, default_cell: CacheCell):
        """reset() zeros SH coefficients."""
        default_cell.sh_r = [1.0, 2.0, 3.0, 4.0]
        default_cell.reset()
        assert all(c == 0.0 for c in default_cell.sh_r)

    def test_get_dc_radiance_zeroed(self, default_cell: CacheCell):
        """DC radiance of zeroed cell is zero."""
        radiance = default_cell.get_dc_radiance()
        assert radiance.x == 0.0
        assert radiance.y == 0.0
        assert radiance.z == 0.0

    def test_get_dc_radiance_scales_by_pi(self):
        """DC radiance scales DC coefficient by pi."""
        cell = CacheCell()
        cell.sh_r = [1.0, 0.0, 0.0, 0.0]
        cell.sh_g = [2.0, 0.0, 0.0, 0.0]
        cell.sh_b = [3.0, 0.0, 0.0, 0.0]
        radiance = cell.get_dc_radiance()
        assert abs(radiance.x - math.pi * 1.0) < 0.001
        assert abs(radiance.y - math.pi * 2.0) < 0.001
        assert abs(radiance.z - math.pi * 3.0) < 0.001

    def test_evaluate_sh_at_origin(self, default_cell: CacheCell):
        """Evaluating zeroed SH returns zero."""
        result = default_cell.evaluate_sh(Vec3(0.0, 1.0, 0.0))
        assert result.x == 0.0
        assert result.y == 0.0
        assert result.z == 0.0

    def test_evaluate_sh_dc_only(self):
        """SH evaluation with only DC returns constant."""
        cell = CacheCell()
        cell.sh_r = [1.0, 0.0, 0.0, 0.0]
        cell.sh_g = [0.5, 0.0, 0.0, 0.0]
        cell.sh_b = [0.0, 0.0, 0.0, 0.0]

        # DC coefficient is direction-independent
        c0 = 0.2821
        result1 = cell.evaluate_sh(Vec3(0.0, 1.0, 0.0))
        result2 = cell.evaluate_sh(Vec3(1.0, 0.0, 0.0))

        assert abs(result1.x - result2.x) < 0.001
        assert abs(result1.x - c0 * 1.0) < 0.001

    def test_evaluate_sh_directional(self):
        """SH evaluation with directional terms varies with direction."""
        cell = CacheCell()
        # Y1-1 = y direction (index 1)
        cell.sh_r = [0.0, 1.0, 0.0, 0.0]

        up = cell.evaluate_sh(Vec3(0.0, 1.0, 0.0))
        down = cell.evaluate_sh(Vec3(0.0, -1.0, 0.0))

        # Up should be positive, down should be clamped to 0
        assert up.x > 0.0
        assert down.x == 0.0  # Clamped negative values

    def test_evaluate_sh_clamps_negative(self):
        """Negative SH results are clamped to zero."""
        cell = CacheCell()
        cell.sh_r = [-1.0, 0.0, 0.0, 0.0]  # Negative DC
        result = cell.evaluate_sh(Vec3(0.0, 1.0, 0.0))
        assert result.x == 0.0

    def test_accumulate_first_sample_copies(self, default_cell: CacheCell):
        """First accumulation copies values directly."""
        new_sh_r = [1.0, 0.0, 0.0, 0.0]
        new_sh_g = [2.0, 0.0, 0.0, 0.0]
        new_sh_b = [3.0, 0.0, 0.0, 0.0]

        default_cell.accumulate(new_sh_r, new_sh_g, new_sh_b, 0.05, 1)

        assert default_cell.sh_r[0] == 1.0
        assert default_cell.sh_g[0] == 2.0
        assert default_cell.sh_b[0] == 3.0
        assert default_cell.state == CellState.ACCUMULATING
        assert default_cell.frames_accumulated == 1

    def test_accumulate_blends_subsequent(self, default_cell: CacheCell):
        """Subsequent accumulations blend values."""
        sh1 = [1.0, 0.0, 0.0, 0.0]
        sh2 = [2.0, 0.0, 0.0, 0.0]

        default_cell.accumulate(sh1, sh1, sh1, 0.5, 1)
        default_cell.accumulate(sh2, sh2, sh2, 0.5, 2)

        # With 50% blend: (1.0 * 0.5) + (2.0 * 0.5) = 1.5
        assert abs(default_cell.sh_r[0] - 1.5) < 0.001

    def test_accumulate_increments_frame_count(self, default_cell: CacheCell):
        """Accumulation increments frame counter."""
        sh = [1.0, 0.0, 0.0, 0.0]

        default_cell.accumulate(sh, sh, sh, 0.05, 1)
        assert default_cell.frames_accumulated == 1

        default_cell.accumulate(sh, sh, sh, 0.05, 2)
        assert default_cell.frames_accumulated == 2

    def test_update_state_empty_stays_empty(self, default_cell: CacheCell):
        """EMPTY cell stays EMPTY after state update."""
        default_cell.update_state(4, 0.001)
        assert default_cell.state == CellState.EMPTY

    def test_update_state_becomes_stable(self):
        """Cell becomes STABLE after enough frames with low variance."""
        cell = CacheCell()
        cell.state = CellState.ACCUMULATING
        cell.frames_accumulated = 10
        cell.variance = 0.0001  # Below threshold

        cell.update_state(4, 0.001)

        assert cell.state == CellState.STABLE

    def test_update_state_stays_accumulating(self):
        """Cell stays ACCUMULATING if variance is high."""
        cell = CacheCell()
        cell.state = CellState.ACCUMULATING
        cell.frames_accumulated = 10
        cell.variance = 0.1  # Above threshold

        cell.update_state(4, 0.001)

        assert cell.state == CellState.ACCUMULATING

    def test_update_state_stable_to_accumulating(self):
        """STABLE cell returns to ACCUMULATING if variance increases."""
        cell = CacheCell()
        cell.state = CellState.STABLE
        cell.frames_accumulated = 10
        cell.variance = 0.1  # Above threshold

        cell.update_state(4, 0.001)

        assert cell.state == CellState.ACCUMULATING

    def test_to_bytes_size(self, default_cell: CacheCell):
        """to_bytes() returns 64 bytes."""
        data = default_cell.to_bytes()
        assert len(data) == 64

    def test_from_bytes_roundtrip(self):
        """from_bytes(to_bytes()) roundtrips correctly."""
        original = CacheCell()
        original.sh_r = [1.0, 2.0, 3.0, 4.0]
        original.sh_g = [5.0, 6.0, 7.0, 8.0]
        original.sh_b = [9.0, 10.0, 11.0, 12.0]
        original.state = CellState.STABLE
        original.frames_accumulated = 42
        original.variance = 0.123

        data = original.to_bytes()
        restored = CacheCell.from_bytes(data)

        assert restored.sh_r == original.sh_r
        assert restored.sh_g == original.sh_g
        assert restored.sh_b == original.sh_b
        assert restored.state == original.state
        assert restored.frames_accumulated == original.frames_accumulated
        assert abs(restored.variance - original.variance) < 0.0001


# ============================================================================
# CacheGrid Tests
# ============================================================================


class TestCacheGrid:
    """Tests for CacheGrid."""

    def test_creates_correct_cell_count(self, small_grid: CacheGrid):
        """Grid creates correct number of cells."""
        assert len(small_grid.cells) == 64  # 4*4*4

    def test_cell_size_calculation(self, small_grid: CacheGrid):
        """Cell size is bounds / dimensions."""
        # Unit bounds: 2x2x2, grid: 4x4x4
        # Cell size: 0.5 x 0.5 x 0.5
        size = small_grid.cell_size
        assert abs(size.x - 0.5) < 0.001
        assert abs(size.y - 0.5) < 0.001
        assert abs(size.z - 0.5) < 0.001

    def test_dimensions_property(self, small_grid: CacheGrid):
        """dimensions property returns (w, h, d)."""
        assert small_grid.dimensions == (4, 4, 4)

    def test_cell_index_corner_zero(self, small_grid: CacheGrid):
        """Cell index at (0, 0, 0) is 0."""
        assert small_grid.cell_index(0, 0, 0) == 0

    def test_cell_index_z_major(self, small_grid: CacheGrid):
        """Cell index uses Z-major ordering."""
        # Index 16 should be at z=1 for 4x4 grid
        assert small_grid.cell_index(0, 0, 1) == 16

    def test_cell_index_out_of_bounds_x(self, small_grid: CacheGrid):
        """Out of bounds X raises IndexError."""
        with pytest.raises(IndexError, match="ix=4"):
            small_grid.cell_index(4, 0, 0)

    def test_cell_index_out_of_bounds_y(self, small_grid: CacheGrid):
        """Out of bounds Y raises IndexError."""
        with pytest.raises(IndexError, match="iy=4"):
            small_grid.cell_index(0, 4, 0)

    def test_cell_index_out_of_bounds_z(self, small_grid: CacheGrid):
        """Out of bounds Z raises IndexError."""
        with pytest.raises(IndexError, match="iz=4"):
            small_grid.cell_index(0, 0, 4)

    def test_cell_coords_roundtrip(self, small_grid: CacheGrid):
        """cell_coords(cell_index(...)) roundtrips."""
        for iz in range(4):
            for iy in range(4):
                for ix in range(4):
                    flat = small_grid.cell_index(ix, iy, iz)
                    coords = small_grid.cell_coords(flat)
                    assert coords == (ix, iy, iz)

    def test_cell_coords_out_of_bounds(self, small_grid: CacheGrid):
        """Out of bounds flat index raises IndexError."""
        with pytest.raises(IndexError):
            small_grid.cell_coords(64)

    def test_world_to_cell_center(self, small_grid: CacheGrid):
        """Origin maps to cell (2, 2, 2) in 4x4x4 grid with -1 to 1 bounds."""
        # Origin is at center of bounds, which is cell (2, 2, 2)
        # Actually: cell 0 is at -1 to -0.5, cell 1 is -0.5 to 0, etc.
        # Origin (0,0,0) is at the boundary of cells 1/2
        # int(0 + 1) / 0.5 = 2
        ix, iy, iz = small_grid.world_to_cell(Vec3(0.0, 0.0, 0.0))
        assert ix == 2
        assert iy == 2
        assert iz == 2

    def test_world_to_cell_corner(self, small_grid: CacheGrid):
        """Min corner maps to cell (0, 0, 0)."""
        ix, iy, iz = small_grid.world_to_cell(Vec3(-1.0, -1.0, -1.0))
        assert ix == 0
        assert iy == 0
        assert iz == 0

    def test_world_to_cell_max_corner(self, small_grid: CacheGrid):
        """Max corner maps to last cell."""
        ix, iy, iz = small_grid.world_to_cell(Vec3(0.99, 0.99, 0.99))
        assert ix == 3
        assert iy == 3
        assert iz == 3

    def test_world_to_cell_clamps(self, small_grid: CacheGrid):
        """Out of bounds positions are clamped."""
        ix, iy, iz = small_grid.world_to_cell(Vec3(100.0, 100.0, 100.0))
        assert ix == 3
        assert iy == 3
        assert iz == 3

    def test_cell_to_world_center(self, small_grid: CacheGrid):
        """Cell (0,0,0) center is at (-0.75, -0.75, -0.75)."""
        # Cell 0 spans -1 to -0.5, center is at -0.75
        pos = small_grid.cell_to_world(0, 0, 0)
        assert abs(pos.x - (-0.75)) < 0.001
        assert abs(pos.y - (-0.75)) < 0.001
        assert abs(pos.z - (-0.75)) < 0.001

    def test_get_cell_returns_cell(self, small_grid: CacheGrid):
        """get_cell returns a CacheCell."""
        cell = small_grid.get_cell(0, 0, 0)
        assert isinstance(cell, CacheCell)

    def test_set_cell_updates_grid(self, small_grid: CacheGrid):
        """set_cell updates the cell at coordinates."""
        new_cell = CacheCell()
        new_cell.sh_r = [999.0, 0.0, 0.0, 0.0]
        small_grid.set_cell(1, 1, 1, new_cell)

        retrieved = small_grid.get_cell(1, 1, 1)
        assert retrieved.sh_r[0] == 999.0

    def test_sample_inside_bounds(self, small_grid: CacheGrid):
        """sample() returns zero for empty grid."""
        result = small_grid.sample(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))
        assert result.x == 0.0
        assert result.y == 0.0
        assert result.z == 0.0

    def test_sample_outside_bounds(self, small_grid: CacheGrid):
        """sample() outside bounds returns zero."""
        result = small_grid.sample(Vec3(100.0, 100.0, 100.0), Vec3(0.0, 1.0, 0.0))
        assert result.x == 0.0

    def test_sample_with_data(self, small_grid: CacheGrid):
        """sample() returns interpolated data from filled cells."""
        # Fill all cells with constant value
        for cell in small_grid.cells:
            cell.sh_r = [1.0, 0.0, 0.0, 0.0]
            cell.sh_g = [0.5, 0.0, 0.0, 0.0]
            cell.sh_b = [0.25, 0.0, 0.0, 0.0]
            cell.state = CellState.STABLE

        result = small_grid.sample(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))

        # Should get SH evaluation of constant field
        assert result.x > 0.0
        assert result.y > 0.0
        assert result.z > 0.0

    def test_sample_dc_inside(self, small_grid: CacheGrid):
        """sample_dc() returns DC radiance."""
        cell = small_grid.get_cell(2, 2, 2)
        cell.sh_r = [1.0, 0.0, 0.0, 0.0]
        cell.state = CellState.STABLE

        result = small_grid.sample_dc(Vec3(0.0, 0.0, 0.0))
        assert result.x > 0.0  # pi * 1.0

    def test_sample_dc_outside(self, small_grid: CacheGrid):
        """sample_dc() outside bounds returns zero."""
        result = small_grid.sample_dc(Vec3(100.0, 0.0, 0.0))
        assert result.x == 0.0

    def test_clear_resets_all_cells(self, small_grid: CacheGrid):
        """clear() resets all cells to EMPTY."""
        for cell in small_grid.cells:
            cell.state = CellState.STABLE
            cell.sh_r = [1.0, 0.0, 0.0, 0.0]

        small_grid.clear()

        for cell in small_grid.cells:
            assert cell.state == CellState.EMPTY
            assert all(c == 0.0 for c in cell.sh_r)

    def test_clear_resets_frame_index(self, small_grid: CacheGrid):
        """clear() resets frame_index to 0."""
        small_grid.frame_index = 100
        small_grid.clear()
        assert small_grid.frame_index == 0

    def test_iter_cells_count(self, small_grid: CacheGrid):
        """iter_cells() yields all cells."""
        count = sum(1 for _ in small_grid.iter_cells())
        assert count == 64

    def test_iter_cells_coordinates(self, small_grid: CacheGrid):
        """iter_cells() yields correct coordinates."""
        for ix, iy, iz, cell in small_grid.iter_cells():
            assert 0 <= ix < 4
            assert 0 <= iy < 4
            assert 0 <= iz < 4
            assert cell is small_grid.get_cell(ix, iy, iz)

    def test_get_statistics_total_cells(self, small_grid: CacheGrid):
        """get_statistics() includes total_cells."""
        stats = small_grid.get_statistics()
        assert stats["total_cells"] == 64

    def test_get_statistics_dimensions(self, small_grid: CacheGrid):
        """get_statistics() includes dimensions."""
        stats = small_grid.get_statistics()
        assert stats["dimensions"] == (4, 4, 4)

    def test_get_statistics_state_counts(self, small_grid: CacheGrid):
        """get_statistics() counts cell states."""
        small_grid.cells[0].state = CellState.STABLE
        small_grid.cells[1].state = CellState.STABLE

        stats = small_grid.get_statistics()
        assert stats["state_counts"]["EMPTY"] == 62
        assert stats["state_counts"]["STABLE"] == 2

    def test_to_bytes_size(self, small_grid: CacheGrid):
        """to_bytes() returns correct total size."""
        data = small_grid.to_bytes()
        expected = 64 * 64  # 64 cells * 64 bytes each
        assert len(data) == expected


# ============================================================================
# RadianceCache Tests
# ============================================================================


class TestRadianceCache:
    """Tests for RadianceCache."""

    def test_create_default_config(self, unit_bounds: AABB):
        """create() with None config uses defaults."""
        cache = RadianceCache.create(unit_bounds)
        assert cache.config.width == 64

    def test_create_custom_config(self, unit_bounds: AABB, small_config: RadianceCacheConfig):
        """create() uses provided config."""
        cache = RadianceCache.create(unit_bounds, small_config)
        assert cache.config.width == 4

    def test_create_with_quality(self, unit_bounds: AABB):
        """create_with_quality() uses quality preset."""
        cache = RadianceCache.create_with_quality(unit_bounds, CacheQuality.LOW)
        assert cache.config.width == 32

    def test_initial_not_valid(self, small_cache: RadianceCache):
        """New cache is not valid."""
        assert small_cache.is_valid is False

    def test_initial_needs_update(self, small_cache: RadianceCache):
        """New cache needs full update."""
        assert small_cache.needs_full_update is True

    def test_update_from_probes_marks_valid(self, small_cache: RadianceCache):
        """update_from_probes() marks cache as valid."""
        sampler = create_constant_probe_sampler(Vec3(1.0, 1.0, 1.0))
        small_cache.update_from_probes(sampler)
        assert small_cache.is_valid is True
        assert small_cache.needs_full_update is False

    def test_update_from_probes_increments_frame(self, small_cache: RadianceCache):
        """update_from_probes() increments frame index."""
        sampler = create_constant_probe_sampler(Vec3(1.0, 1.0, 1.0))
        small_cache.update_from_probes(sampler)
        assert small_cache.grid.frame_index == 1

    def test_update_cell_single(self, small_cache: RadianceCache):
        """update_cell() updates single cell."""
        sh = [1.0, 0.0, 0.0, 0.0]
        small_cache.update_cell(0, 0, 0, sh, sh, sh)

        cell = small_cache.grid.get_cell(0, 0, 0)
        assert cell.sh_r[0] == 1.0
        assert cell.state == CellState.ACCUMULATING

    def test_sample_invalid_returns_zero(self, small_cache: RadianceCache):
        """sample() on invalid cache returns zero."""
        result = small_cache.sample(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))
        assert result.x == 0.0

    def test_sample_valid_cache(self, small_cache: RadianceCache):
        """sample() on valid cache returns data."""
        sampler = create_constant_probe_sampler(Vec3(1.0, 0.5, 0.25))
        small_cache.update_from_probes(sampler)

        result = small_cache.sample(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))
        assert result.x > 0.0

    def test_sample_dc_invalid_returns_zero(self, small_cache: RadianceCache):
        """sample_dc() on invalid cache returns zero."""
        result = small_cache.sample_dc(Vec3(0.0, 0.0, 0.0))
        assert result.x == 0.0

    def test_invalidate_clears_valid(self, small_cache: RadianceCache):
        """invalidate() clears is_valid flag."""
        sampler = create_constant_probe_sampler(Vec3(1.0, 1.0, 1.0))
        small_cache.update_from_probes(sampler)
        small_cache.invalidate()
        assert small_cache.is_valid is False
        assert small_cache.needs_full_update is True

    def test_clear_resets_everything(self, small_cache: RadianceCache):
        """clear() resets cache completely."""
        sampler = create_constant_probe_sampler(Vec3(1.0, 1.0, 1.0))
        small_cache.update_from_probes(sampler)
        small_cache.clear()

        assert small_cache.is_valid is False
        assert small_cache.grid.frame_index == 0
        for cell in small_cache.grid.cells:
            assert cell.state == CellState.EMPTY

    def test_get_bounds(self, small_cache: RadianceCache, unit_bounds: AABB):
        """get_bounds() returns cache bounds."""
        bounds = small_cache.get_bounds()
        assert bounds.min.x == unit_bounds.min.x
        assert bounds.max.x == unit_bounds.max.x

    def test_get_statistics_includes_valid(self, small_cache: RadianceCache):
        """get_statistics() includes validity flags."""
        stats = small_cache.get_statistics()
        assert "is_valid" in stats
        assert "needs_full_update" in stats


# ============================================================================
# RadianceCacheUpdater Tests
# ============================================================================


class TestRadianceCacheUpdater:
    """Tests for RadianceCacheUpdater."""

    def test_default_workgroup_size(self, small_cache: RadianceCache):
        """Default workgroup is (8, 8, 4)."""
        updater = RadianceCacheUpdater(cache=small_cache)
        assert updater.workgroup_size == (8, 8, 4)

    def test_get_dispatch_size(self, small_cache: RadianceCache):
        """get_dispatch_size() computes dispatch dimensions."""
        updater = RadianceCacheUpdater(cache=small_cache)
        dx, dy, dz = updater.get_dispatch_size()

        # 4x4x4 grid with 8x8x4 workgroup: ceil(4/8)=1, ceil(4/4)=1
        assert dx == 1
        assert dy == 1
        assert dz == 1

    def test_get_dispatch_size_large(self, large_bounds: AABB):
        """get_dispatch_size() for larger grid."""
        config = RadianceCacheConfig(width=64, height=64, depth=32)
        cache = RadianceCache.create(large_bounds, config)
        updater = RadianceCacheUpdater(cache=cache)
        dx, dy, dz = updater.get_dispatch_size()

        # 64/8=8, 64/8=8, 32/4=8
        assert dx == 8
        assert dy == 8
        assert dz == 8

    def test_get_shader_uniforms_size(self, small_cache: RadianceCache):
        """get_shader_uniforms() returns 60 bytes."""
        updater = RadianceCacheUpdater(cache=small_cache)
        uniforms = updater.get_shader_uniforms()
        # 3f+I + 3f+I + 3f+I + f+I+I = 12+4 + 12+4 + 12+4 + 4+4+4 = 60 bytes
        assert len(uniforms) == 60

    def test_step_full_update(self, small_cache: RadianceCache):
        """step() with cells_per_frame=0 updates all cells."""
        updater = RadianceCacheUpdater(cache=small_cache, cells_per_frame=0)
        count = updater.step()
        assert count == 64  # All cells

    def test_step_partial_update(self, small_cache: RadianceCache):
        """step() with cells_per_frame limits cells updated."""
        updater = RadianceCacheUpdater(cache=small_cache, cells_per_frame=16)
        count = updater.step()
        assert count == 16
        assert updater.current_offset == 16

    def test_step_wraps_around(self, small_cache: RadianceCache):
        """step() wraps around when reaching end."""
        updater = RadianceCacheUpdater(cache=small_cache, cells_per_frame=32)
        updater.step()  # 0-32
        updater.step()  # 32-64
        count = updater.step()  # wrap to 0
        assert count == 32
        assert updater.current_offset == 32

    def test_reset_clears_offset(self, small_cache: RadianceCache):
        """reset() clears current_offset."""
        updater = RadianceCacheUpdater(cache=small_cache, cells_per_frame=16)
        updater.step()
        updater.reset()
        assert updater.current_offset == 0

    def test_generate_wgsl_shader_not_empty(self, small_cache: RadianceCache):
        """generate_wgsl_shader() returns non-empty string."""
        updater = RadianceCacheUpdater(cache=small_cache)
        shader = updater.generate_wgsl_shader()
        assert len(shader) > 100
        assert "@compute" in shader
        assert "CacheCell" in shader


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_estimate_cache_memory_low(self):
        """estimate_cache_memory() for LOW quality."""
        memory = estimate_cache_memory(CacheQuality.LOW)
        # 32*32*16 * 64 bytes = 1,048,576 bytes
        assert memory == 32 * 32 * 16 * 64

    def test_estimate_cache_memory_medium(self):
        """estimate_cache_memory() for MEDIUM quality."""
        memory = estimate_cache_memory(CacheQuality.MEDIUM)
        assert memory == 64 * 64 * 32 * 64

    def test_recommend_cache_quality_high_budget(self):
        """recommend_cache_quality() with large budget returns ULTRA."""
        quality = recommend_cache_quality(1000.0)  # 1000 MB
        assert quality == CacheQuality.ULTRA

    def test_recommend_cache_quality_low_budget(self):
        """recommend_cache_quality() with tiny budget returns LOW."""
        quality = recommend_cache_quality(0.5)  # 0.5 MB
        assert quality == CacheQuality.LOW

    def test_create_constant_probe_sampler(self):
        """create_constant_probe_sampler() returns constant values."""
        sampler = create_constant_probe_sampler(Vec3(1.0, 2.0, 3.0))

        sh_r, sh_g, sh_b = sampler(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))

        # DC coefficient should encode the radiance
        assert len(sh_r) == 4
        assert len(sh_g) == 4
        assert len(sh_b) == 4

        # R > G > B (scaled by DC)
        assert sh_r[0] < sh_g[0] < sh_b[0]

    def test_create_gradient_probe_sampler(self, unit_bounds: AABB):
        """create_gradient_probe_sampler() varies across bounds."""
        sampler = create_gradient_probe_sampler(
            Vec3(0.0, 0.0, 0.0),
            Vec3(1.0, 1.0, 1.0),
            unit_bounds,
        )

        # At min bound, should be ~0
        sh_r_min, _, _ = sampler(Vec3(-1.0, -1.0, -1.0), Vec3(0.0, 1.0, 0.0))
        # At max bound, should be ~1
        sh_r_max, _, _ = sampler(Vec3(1.0, 1.0, 1.0), Vec3(0.0, 1.0, 0.0))

        assert sh_r_max[0] > sh_r_min[0]

    def test_generate_radiance_cache_sampling_wgsl(self):
        """generate_radiance_cache_sampling_wgsl() returns valid WGSL."""
        wgsl = generate_radiance_cache_sampling_wgsl()
        assert "sample_radiance_cache" in wgsl
        assert "sample_radiance_cache_dc" in wgsl
        assert "CacheSampleUniforms" in wgsl


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_update_workflow(self, unit_bounds: AABB):
        """Complete workflow: create, update, sample."""
        # Create cache
        cache = RadianceCache.create_with_quality(unit_bounds, CacheQuality.LOW)
        assert not cache.is_valid

        # Update from probes
        sampler = create_constant_probe_sampler(Vec3(0.5, 0.5, 0.5))
        cache.update_from_probes(sampler)
        assert cache.is_valid

        # Sample
        result = cache.sample(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))
        assert result.x > 0.0

    def test_temporal_accumulation(self, unit_bounds: AABB):
        """Multiple updates converge to stable values."""
        config = RadianceCacheConfig(width=4, height=4, depth=4, blend_factor=0.5)
        cache = RadianceCache.create(unit_bounds, config)

        sampler = create_constant_probe_sampler(Vec3(1.0, 1.0, 1.0))

        # Update multiple times
        for _ in range(10):
            cache.update_from_probes(sampler)

        # Check cells are stable
        stats = cache.get_statistics()
        assert stats["state_counts"]["EMPTY"] == 0

    def test_gradient_sampling(self, unit_bounds: AABB):
        """Gradient sampler produces varying cache values."""
        config = RadianceCacheConfig(width=4, height=4, depth=4)
        cache = RadianceCache.create(unit_bounds, config)

        sampler = create_gradient_probe_sampler(
            Vec3(0.0, 0.0, 0.0),
            Vec3(1.0, 1.0, 1.0),
            unit_bounds,
        )
        cache.update_from_probes(sampler)

        # Sample at different positions
        low = cache.sample_dc(Vec3(-0.5, -0.5, -0.5))
        high = cache.sample_dc(Vec3(0.5, 0.5, 0.5))

        # Higher position should have more radiance
        assert high.x > low.x

    def test_updater_shader_generation(self, unit_bounds: AABB):
        """Updater generates valid shader code."""
        cache = RadianceCache.create_with_quality(unit_bounds, CacheQuality.MEDIUM)
        updater = RadianceCacheUpdater(cache=cache)

        shader = updater.generate_wgsl_shader()

        # Check shader structure
        assert "struct CacheUniforms" in shader
        assert "struct CacheCell" in shader
        assert "@compute @workgroup_size(8, 8, 4)" in shader
        assert "fn main" in shader
        assert "cell_index" in shader
        assert "cell_to_world" in shader

    def test_cache_invalidate_and_revalidate(self, unit_bounds: AABB):
        """Cache can be invalidated and re-validated."""
        cache = RadianceCache.create_with_quality(unit_bounds, CacheQuality.LOW)
        sampler = create_constant_probe_sampler(Vec3(1.0, 1.0, 1.0))

        # Initial update
        cache.update_from_probes(sampler)
        assert cache.is_valid

        # Invalidate
        cache.invalidate()
        assert not cache.is_valid
        assert cache.needs_full_update

        # Re-validate
        cache.update_from_probes(sampler)
        assert cache.is_valid

    def test_cache_clear_workflow(self, unit_bounds: AABB):
        """Cache clear allows fresh start."""
        cache = RadianceCache.create_with_quality(unit_bounds, CacheQuality.LOW)
        sampler = create_constant_probe_sampler(Vec3(1.0, 1.0, 1.0))

        # Update
        cache.update_from_probes(sampler)
        assert cache.grid.frame_index > 0

        # Clear
        cache.clear()
        assert cache.grid.frame_index == 0
        assert not cache.is_valid

        # Sample returns zero
        result = cache.sample(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))
        assert result.x == 0.0

    def test_partial_update_coverage(self, unit_bounds: AABB):
        """Partial updates eventually cover all cells."""
        config = RadianceCacheConfig(width=4, height=4, depth=4)
        cache = RadianceCache.create(unit_bounds, config)
        updater = RadianceCacheUpdater(cache=cache, cells_per_frame=16)

        # 64 cells / 16 per frame = 4 steps to cover all
        total = 0
        for _ in range(4):
            total += updater.step()

        assert total == 64

    def test_gpu_buffer_roundtrip(self, unit_bounds: AABB):
        """Grid to_bytes produces correct size buffer."""
        config = RadianceCacheConfig(width=4, height=4, depth=4)
        cache = RadianceCache.create(unit_bounds, config)

        # Update with data
        sampler = create_constant_probe_sampler(Vec3(1.0, 0.5, 0.25))
        cache.update_from_probes(sampler)

        # Pack to bytes
        data = cache.grid.to_bytes()

        # Should be 64 cells * 64 bytes = 4096 bytes
        assert len(data) == 4096

        # Unpack first cell and verify
        first_cell = CacheCell.from_bytes(data[:64])
        assert first_cell.state == CellState.ACCUMULATING


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_cell_grid(self, unit_bounds: AABB):
        """1x1x1 grid works correctly."""
        config = RadianceCacheConfig(width=1, height=1, depth=1)
        cache = RadianceCache.create(unit_bounds, config)

        sampler = create_constant_probe_sampler(Vec3(1.0, 1.0, 1.0))
        cache.update_from_probes(sampler)

        result = cache.sample_dc(Vec3(0.0, 0.0, 0.0))
        assert result.x > 0.0

    def test_zero_blend_factor(self, unit_bounds: AABB):
        """Zero blend factor keeps original values."""
        config = RadianceCacheConfig(width=2, height=2, depth=2, blend_factor=0.0)
        cache = RadianceCache.create(unit_bounds, config)

        # First update
        sampler1 = create_constant_probe_sampler(Vec3(1.0, 1.0, 1.0))
        cache.update_from_probes(sampler1)

        value1 = cache.grid.get_cell(0, 0, 0).sh_r[0]

        # Second update with different value
        sampler2 = create_constant_probe_sampler(Vec3(2.0, 2.0, 2.0))
        cache.update_from_probes(sampler2)

        value2 = cache.grid.get_cell(0, 0, 0).sh_r[0]

        # Should not change (blend = 0)
        assert value1 == value2

    def test_one_blend_factor(self, unit_bounds: AABB):
        """Blend factor of 1.0 replaces immediately."""
        config = RadianceCacheConfig(width=2, height=2, depth=2, blend_factor=1.0)
        cache = RadianceCache.create(unit_bounds, config)

        # First update
        sampler1 = create_constant_probe_sampler(Vec3(1.0, 1.0, 1.0))
        cache.update_from_probes(sampler1)

        # Second update with different value
        sampler2 = create_constant_probe_sampler(Vec3(2.0, 2.0, 2.0))
        cache.update_from_probes(sampler2)

        # Should have new value (blend = 1)
        cell = cache.grid.get_cell(0, 0, 0)
        dc_scale = 1.0 / (2.0 * math.sqrt(math.pi))
        expected = 2.0 / dc_scale
        assert abs(cell.sh_r[0] - expected) < 0.001

    def test_negative_position_sampling(self, unit_bounds: AABB):
        """Sampling at negative positions works."""
        config = RadianceCacheConfig(width=4, height=4, depth=4)
        cache = RadianceCache.create(unit_bounds, config)

        sampler = create_constant_probe_sampler(Vec3(1.0, 1.0, 1.0))
        cache.update_from_probes(sampler)

        result = cache.sample_dc(Vec3(-0.5, -0.5, -0.5))
        assert result.x > 0.0

    def test_boundary_sampling(self, unit_bounds: AABB):
        """Sampling exactly at boundaries works."""
        config = RadianceCacheConfig(width=4, height=4, depth=4)
        cache = RadianceCache.create(unit_bounds, config)

        sampler = create_constant_probe_sampler(Vec3(1.0, 1.0, 1.0))
        cache.update_from_probes(sampler)

        # At exact boundary
        result = cache.sample_dc(Vec3(-1.0, -1.0, -1.0))
        assert result.x >= 0.0

    def test_max_corner_sampling(self, unit_bounds: AABB):
        """Sampling at max corner (just inside) works."""
        config = RadianceCacheConfig(width=4, height=4, depth=4)
        cache = RadianceCache.create(unit_bounds, config)

        sampler = create_constant_probe_sampler(Vec3(1.0, 1.0, 1.0))
        cache.update_from_probes(sampler)

        # Just inside max corner
        result = cache.sample_dc(Vec3(0.99, 0.99, 0.99))
        assert result.x > 0.0

    def test_zero_variance_threshold(self, unit_bounds: AABB):
        """Very small variance threshold allows cells to become stable quickly."""
        config = RadianceCacheConfig(
            width=2, height=2, depth=2,
            variance_threshold=0.001,  # Small but non-zero threshold
            hysteresis_frames=1,
            blend_factor=1.0,  # Use 100% blend to get exact values
        )
        cache = RadianceCache.create(unit_bounds, config)

        # Constant data with 100% blend means zero variance after second frame
        sampler = create_constant_probe_sampler(Vec3(1.0, 1.0, 1.0))
        cache.update_from_probes(sampler)

        # First frame is ACCUMULATING (first sample always has high variance)
        assert cache.grid.get_cell(0, 0, 0).state == CellState.ACCUMULATING

        # Second frame: with 100% blend, new sample == old sample, so diff_sq = 0
        # variance = variance * 0.0 + diff_sq * 1.0 = 0.0
        cache.update_from_probes(sampler)
        # Variance converged to 0 since constant data and 100% blend
        cell = cache.grid.get_cell(0, 0, 0)
        assert cell.variance == 0.0  # Zero variance with constant input
        # 0.0 < 0.001 and frames >= 1, so STABLE
        assert cell.state == CellState.STABLE
