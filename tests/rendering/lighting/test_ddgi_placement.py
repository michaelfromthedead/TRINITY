"""Tests for DDGI camera-relative probe placement system (T-GIR-P2.1).

Tests cover:
- Quality presets and configuration
- Grid origin computation with snapping
- Probe index to world position roundtrip
- Camera scrolling and origin updates
- GPU data serialization matching Rust struct layout
- Memory estimation per preset
"""

from __future__ import annotations

import math
import struct
import pytest

from engine.core.math.geometry import AABB
from engine.core.math.vec import Vec3
from engine.rendering.lighting.gi_ddgi import (
    DDGIQualityPreset,
    DDGIConfig,
    DDGICameraRelativeGrid,
    get_preset_params,
    estimate_gpu_memory,
    _QUALITY_PARAMS,
)


# ============================================================================
# Quality Preset Tests
# ============================================================================


class TestDDGIQualityPreset:
    """Tests for DDGIQualityPreset enum."""

    def test_preset_enum_values(self) -> None:
        """Test all preset enum values exist."""
        assert DDGIQualityPreset.LOW is not None
        assert DDGIQualityPreset.MEDIUM is not None
        assert DDGIQualityPreset.HIGH is not None
        assert DDGIQualityPreset.ULTRA is not None

    def test_preset_params_low(self) -> None:
        """Test LOW preset parameters."""
        dims, spacing, rays = get_preset_params(DDGIQualityPreset.LOW)
        assert dims == (16, 16, 4)
        assert spacing == pytest.approx(4.0)
        assert rays == 32

    def test_preset_params_medium(self) -> None:
        """Test MEDIUM preset parameters."""
        dims, spacing, rays = get_preset_params(DDGIQualityPreset.MEDIUM)
        assert dims == (24, 24, 6)
        assert spacing == pytest.approx(3.0)
        assert rays == 48

    def test_preset_params_high(self) -> None:
        """Test HIGH preset parameters."""
        dims, spacing, rays = get_preset_params(DDGIQualityPreset.HIGH)
        assert dims == (32, 32, 8)
        assert spacing == pytest.approx(2.0)
        assert rays == 64

    def test_preset_params_ultra(self) -> None:
        """Test ULTRA preset parameters."""
        dims, spacing, rays = get_preset_params(DDGIQualityPreset.ULTRA)
        assert dims == (48, 48, 12)
        assert spacing == pytest.approx(1.5)
        assert rays == 128

    def test_all_presets_have_params(self) -> None:
        """Test all presets have parameters defined."""
        for preset in DDGIQualityPreset:
            params = get_preset_params(preset)
            assert len(params) == 3
            assert all(d > 0 for d in params[0])
            assert params[1] > 0
            assert params[2] > 0


# ============================================================================
# Memory Estimation Tests
# ============================================================================


class TestMemoryEstimation:
    """Tests for GPU memory estimation."""

    def test_memory_low_preset(self) -> None:
        """Test LOW preset memory (~200KB)."""
        mem = estimate_gpu_memory(DDGIQualityPreset.LOW)
        # 1024 probes * (192 + 16) + 64 = 212,992
        assert 200_000 < mem < 250_000

    def test_memory_medium_preset(self) -> None:
        """Test MEDIUM preset memory (~650KB)."""
        mem = estimate_gpu_memory(DDGIQualityPreset.MEDIUM)
        # 3456 probes * (192 + 16) + 64 = 718,912
        assert 600_000 < mem < 800_000

    def test_memory_high_preset(self) -> None:
        """Test HIGH preset memory (~1.5MB)."""
        mem = estimate_gpu_memory(DDGIQualityPreset.HIGH)
        # 8192 probes * (192 + 16) + 64 = 1,703,936
        assert 1_500_000 < mem < 2_000_000

    def test_memory_ultra_preset(self) -> None:
        """Test ULTRA preset memory (~5MB)."""
        mem = estimate_gpu_memory(DDGIQualityPreset.ULTRA)
        # 27648 probes * (192 + 16) + 64 = 5,750,848
        assert 5_000_000 < mem < 6_000_000

    def test_memory_increases_with_quality(self) -> None:
        """Test memory increases with quality level."""
        low = estimate_gpu_memory(DDGIQualityPreset.LOW)
        medium = estimate_gpu_memory(DDGIQualityPreset.MEDIUM)
        high = estimate_gpu_memory(DDGIQualityPreset.HIGH)
        ultra = estimate_gpu_memory(DDGIQualityPreset.ULTRA)

        assert low < medium < high < ultra


# ============================================================================
# DDGIConfig Tests
# ============================================================================


class TestDDGIConfig:
    """Tests for DDGIConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = DDGIConfig()
        assert config.preset == DDGIQualityPreset.HIGH
        assert config.probe_spacing is None
        assert config.grid_dimensions is None
        assert config.hysteresis == pytest.approx(0.97)
        assert config.update_fraction == pytest.approx(0.125)

    def test_get_dimensions_from_preset(self) -> None:
        """Test dimensions come from preset by default."""
        config = DDGIConfig(preset=DDGIQualityPreset.LOW)
        assert config.get_dimensions() == (16, 16, 4)

    def test_get_dimensions_override(self) -> None:
        """Test dimension override."""
        config = DDGIConfig(grid_dimensions=(10, 10, 5))
        assert config.get_dimensions() == (10, 10, 5)

    def test_get_spacing_from_preset(self) -> None:
        """Test spacing comes from preset by default."""
        config = DDGIConfig(preset=DDGIQualityPreset.MEDIUM)
        assert config.get_spacing() == pytest.approx(3.0)

    def test_get_spacing_override(self) -> None:
        """Test spacing override."""
        config = DDGIConfig(probe_spacing=1.0)
        assert config.get_spacing() == pytest.approx(1.0)

    def test_get_rays_per_probe_from_preset(self) -> None:
        """Test rays per probe comes from preset by default."""
        config = DDGIConfig(preset=DDGIQualityPreset.ULTRA)
        assert config.get_rays_per_probe() == 128

    def test_get_rays_per_probe_override(self) -> None:
        """Test rays per probe override."""
        config = DDGIConfig(rays_per_probe=256)
        assert config.get_rays_per_probe() == 256

    def test_total_probes(self) -> None:
        """Test total probe calculation."""
        config = DDGIConfig(preset=DDGIQualityPreset.HIGH)
        assert config.total_probes() == 32 * 32 * 8

    def test_estimated_memory_matches_function(self) -> None:
        """Test estimated memory matches standalone function."""
        config = DDGIConfig(preset=DDGIQualityPreset.LOW)
        assert config.estimated_memory_bytes() == estimate_gpu_memory(DDGIQualityPreset.LOW)

    def test_validation_passes_for_valid_config(self) -> None:
        """Test validation passes for valid config."""
        config = DDGIConfig()
        errors = config.validate()
        assert len(errors) == 0

    def test_validation_fails_for_zero_dimensions(self) -> None:
        """Test validation fails for zero dimensions."""
        config = DDGIConfig(grid_dimensions=(0, 10, 10))
        errors = config.validate()
        assert any("positive" in e.lower() for e in errors)

    def test_validation_fails_for_large_dimensions(self) -> None:
        """Test validation fails for dimensions > 64."""
        config = DDGIConfig(grid_dimensions=(100, 100, 100))
        errors = config.validate()
        assert any("64" in e for e in errors)

    def test_validation_fails_for_negative_spacing(self) -> None:
        """Test validation fails for negative spacing."""
        config = DDGIConfig(probe_spacing=-1.0)
        errors = config.validate()
        assert any("positive" in e.lower() for e in errors)

    def test_validation_warns_for_small_spacing(self) -> None:
        """Test validation warns for spacing < 0.5."""
        config = DDGIConfig(probe_spacing=0.1)
        errors = config.validate()
        assert any("0.5" in e for e in errors)

    def test_validation_fails_for_invalid_hysteresis(self) -> None:
        """Test validation fails for hysteresis outside [0, 1]."""
        config = DDGIConfig()
        config.hysteresis = 1.5
        errors = config.validate()
        assert any("hysteresis" in e.lower() for e in errors)


# ============================================================================
# Grid Origin Computation Tests
# ============================================================================


class TestGridOriginComputation:
    """Tests for grid origin computation with snapping."""

    def test_origin_at_world_origin(self) -> None:
        """Test origin computation when camera at world origin."""
        grid = DDGICameraRelativeGrid(config=DDGIConfig(preset=DDGIQualityPreset.HIGH))
        camera_pos = Vec3(0, 0, 0)
        origin = grid.compute_grid_origin(camera_pos)

        # Grid should be centered on camera
        spacing = 2.0
        dims = (32, 32, 8)
        expected_x = -(dims[0] - 1) * spacing / 2
        expected_y = -(dims[1] - 1) * spacing / 2
        expected_z = -(dims[2] - 1) * spacing / 2

        # Origin should be snapped to spacing
        assert origin.x % spacing == pytest.approx(0.0, abs=1e-6)
        assert origin.y % spacing == pytest.approx(0.0, abs=1e-6)
        assert origin.z % spacing == pytest.approx(0.0, abs=1e-6)

    def test_origin_snapped_to_spacing(self) -> None:
        """Test origin is snapped to spacing multiples."""
        config = DDGIConfig(preset=DDGIQualityPreset.LOW)  # 4m spacing
        grid = DDGICameraRelativeGrid(config=config)

        # Camera at odd position
        camera_pos = Vec3(7.3, 2.1, 5.9)
        origin = grid.compute_grid_origin(camera_pos)

        # Origin should be multiple of 4m
        assert origin.x % 4.0 == pytest.approx(0.0, abs=1e-6)
        assert origin.y % 4.0 == pytest.approx(0.0, abs=1e-6)
        assert origin.z % 4.0 == pytest.approx(0.0, abs=1e-6)

    def test_origin_centered_on_camera(self) -> None:
        """Test grid is approximately centered on camera."""
        config = DDGIConfig(preset=DDGIQualityPreset.HIGH)  # 2m spacing, 32x32x8
        grid = DDGICameraRelativeGrid(config=config)

        camera_pos = Vec3(100, 50, 200)
        origin = grid.compute_grid_origin(camera_pos)

        # Grid center should be within one cell of camera
        spacing = 2.0
        dims = config.get_dimensions()
        center = Vec3(
            origin.x + (dims[0] - 1) * spacing / 2,
            origin.y + (dims[1] - 1) * spacing / 2,
            origin.z + (dims[2] - 1) * spacing / 2,
        )

        assert abs(center.x - camera_pos.x) <= spacing
        assert abs(center.y - camera_pos.y) <= spacing
        assert abs(center.z - camera_pos.z) <= spacing


# ============================================================================
# Probe Index Tests
# ============================================================================


class TestProbeIndexing:
    """Tests for probe index conversions."""

    def test_world_to_probe_index_at_origin(self) -> None:
        """Test world to probe index at grid origin."""
        config = DDGIConfig(preset=DDGIQualityPreset.HIGH)
        grid = DDGICameraRelativeGrid(config=config, origin=Vec3(0, 0, 0))

        idx = grid.world_to_probe_index(Vec3(0, 0, 0))
        assert idx == (0, 0, 0)

    def test_world_to_probe_index_offset(self) -> None:
        """Test world to probe index with offset."""
        config = DDGIConfig(preset=DDGIQualityPreset.HIGH)  # 2m spacing
        grid = DDGICameraRelativeGrid(config=config, origin=Vec3(0, 0, 0))

        # Position 4m along X should be probe index 2
        idx = grid.world_to_probe_index(Vec3(4, 0, 0))
        assert idx == (2, 0, 0)

    def test_world_to_probe_index_clamps(self) -> None:
        """Test world to probe index clamps to grid bounds."""
        config = DDGIConfig(preset=DDGIQualityPreset.HIGH)  # 32x32x8
        grid = DDGICameraRelativeGrid(config=config, origin=Vec3(0, 0, 0))

        # Position way outside should clamp
        idx = grid.world_to_probe_index(Vec3(1000, 1000, 1000))
        assert idx == (31, 31, 7)

        idx = grid.world_to_probe_index(Vec3(-1000, -1000, -1000))
        assert idx == (0, 0, 0)

    def test_probe_world_position_at_origin(self) -> None:
        """Test probe world position at index (0,0,0)."""
        config = DDGIConfig(preset=DDGIQualityPreset.HIGH)
        grid = DDGICameraRelativeGrid(config=config, origin=Vec3(10, 20, 30))

        pos = grid.get_probe_world_position(0, 0, 0)
        assert pos == Vec3(10, 20, 30)

    def test_probe_world_position_offset(self) -> None:
        """Test probe world position with offset."""
        config = DDGIConfig(preset=DDGIQualityPreset.HIGH)  # 2m spacing
        grid = DDGICameraRelativeGrid(config=config, origin=Vec3(0, 0, 0))

        pos = grid.get_probe_world_position(5, 3, 2)
        assert pos == Vec3(10, 6, 4)

    def test_index_world_roundtrip(self) -> None:
        """Test roundtrip from index to world to index."""
        config = DDGIConfig(preset=DDGIQualityPreset.HIGH)
        grid = DDGICameraRelativeGrid(config=config, origin=Vec3(0, 0, 0))

        for ix in range(0, 32, 8):
            for iy in range(0, 32, 8):
                for iz in range(0, 8, 2):
                    pos = grid.get_probe_world_position(ix, iy, iz)
                    idx = grid.world_to_probe_index(pos)
                    assert idx == (ix, iy, iz), f"Roundtrip failed for ({ix}, {iy}, {iz})"

    def test_probe_index_to_linear(self) -> None:
        """Test 3D to linear index conversion."""
        config = DDGIConfig(preset=DDGIQualityPreset.HIGH)  # 32x32x8
        grid = DDGICameraRelativeGrid(config=config)

        # Linear index: x + y * dim_x + z * dim_x * dim_y
        assert grid.probe_index_to_linear(0, 0, 0) == 0
        assert grid.probe_index_to_linear(1, 0, 0) == 1
        assert grid.probe_index_to_linear(0, 1, 0) == 32
        assert grid.probe_index_to_linear(0, 0, 1) == 32 * 32
        assert grid.probe_index_to_linear(3, 2, 1) == 3 + 2 * 32 + 1 * 32 * 32


# ============================================================================
# Camera Scrolling Tests
# ============================================================================


class TestCameraScrolling:
    """Tests for camera-relative scrolling."""

    def test_first_update_initializes_origin(self) -> None:
        """Test first camera update initializes origin."""
        config = DDGIConfig(preset=DDGIQualityPreset.HIGH)
        grid = DDGICameraRelativeGrid(config=config)

        scrolled = grid.update_for_camera(Vec3(100, 50, 200))
        assert scrolled is True
        assert grid.origin != Vec3.zero()

    def test_small_movement_no_scroll(self) -> None:
        """Test small camera movement doesn't trigger scroll."""
        config = DDGIConfig(preset=DDGIQualityPreset.HIGH)  # 2m spacing
        grid = DDGICameraRelativeGrid(config=config)

        grid.update_for_camera(Vec3(0, 0, 0))
        initial_origin = Vec3(grid.origin.x, grid.origin.y, grid.origin.z)

        # Move less than one cell
        scrolled = grid.update_for_camera(Vec3(0.5, 0, 0))
        assert scrolled is False
        assert grid.origin == initial_origin

    def test_large_movement_triggers_scroll(self) -> None:
        """Test large camera movement triggers scroll."""
        config = DDGIConfig(preset=DDGIQualityPreset.HIGH)  # 2m spacing
        grid = DDGICameraRelativeGrid(config=config)

        grid.update_for_camera(Vec3(0, 0, 0))
        initial_origin = Vec3(grid.origin.x, grid.origin.y, grid.origin.z)

        # Move more than one cell
        scrolled = grid.update_for_camera(Vec3(10, 0, 0))
        assert scrolled is True
        assert grid.origin != initial_origin

    def test_scroll_offset_wraps(self) -> None:
        """Test scroll offset wraps correctly."""
        config = DDGIConfig(preset=DDGIQualityPreset.LOW)  # 16x16x4
        grid = DDGICameraRelativeGrid(config=config)

        grid.update_for_camera(Vec3(0, 0, 0))

        # Force many scrolls to test wrapping
        for i in range(20):
            grid.update_for_camera(Vec3(i * 10, 0, 0))

        # Scroll offset should be within [0, dim)
        assert 0 <= grid.scroll_offset[0] < 16
        assert 0 <= grid.scroll_offset[1] < 16
        assert 0 <= grid.scroll_offset[2] < 4

    def test_scrolled_probe_index(self) -> None:
        """Test scrolled probe index application."""
        config = DDGIConfig(preset=DDGIQualityPreset.LOW)  # 16x16x4
        grid = DDGICameraRelativeGrid(config=config)
        grid.scroll_offset = (5, 3, 1)

        # Base index (0, 0, 0) with offset (5, 3, 1) should give (5, 3, 1)
        scrolled = grid.get_scrolled_probe_index(0, 0, 0)
        assert scrolled == (5, 3, 1)

        # Test wrapping
        scrolled = grid.get_scrolled_probe_index(15, 15, 3)
        # (15 + 5) % 16 = 4, (15 + 3) % 16 = 2, (3 + 1) % 4 = 0
        assert scrolled == (4, 2, 0)


# ============================================================================
# GPU Data Serialization Tests
# ============================================================================


class TestGPUDataSerialization:
    """Tests for GPU data serialization matching Rust struct layout."""

    def test_gpu_data_size(self) -> None:
        """Test GPU data is exactly 64 bytes."""
        config = DDGIConfig(preset=DDGIQualityPreset.HIGH)
        grid = DDGICameraRelativeGrid(config=config, origin=Vec3(1, 2, 3))

        data = grid.build_gpu_data()
        assert len(data) == 64

    def test_gpu_data_origin_offset(self) -> None:
        """Test origin is at correct offset in GPU data."""
        config = DDGIConfig(preset=DDGIQualityPreset.HIGH)
        grid = DDGICameraRelativeGrid(config=config, origin=Vec3(1.0, 2.0, 3.0))

        data = grid.build_gpu_data()

        # Origin is first 12 bytes (3 floats)
        origin = struct.unpack_from("<3f", data, 0)
        assert origin == pytest.approx((1.0, 2.0, 3.0))

    def test_gpu_data_cell_size_offset(self) -> None:
        """Test cell_size is at correct offset in GPU data."""
        config = DDGIConfig(preset=DDGIQualityPreset.HIGH)  # 2m spacing
        grid = DDGICameraRelativeGrid(config=config, origin=Vec3(0, 0, 0))

        data = grid.build_gpu_data()

        # cell_size starts at offset 16 (after origin + pad)
        cell_size = struct.unpack_from("<3f", data, 16)
        assert cell_size == pytest.approx((2.0, 2.0, 2.0))

    def test_gpu_data_dimensions_offset(self) -> None:
        """Test dimensions is at correct offset in GPU data."""
        config = DDGIConfig(preset=DDGIQualityPreset.HIGH)  # 32x32x8
        grid = DDGICameraRelativeGrid(config=config, origin=Vec3(0, 0, 0))

        data = grid.build_gpu_data()

        # dimensions starts at offset 32 (after origin + pad + cell_size + pad)
        dims = struct.unpack_from("<3I", data, 32)
        assert dims == (32, 32, 8)

    def test_gpu_data_total_probes_offset(self) -> None:
        """Test total_probes is at correct offset in GPU data."""
        config = DDGIConfig(preset=DDGIQualityPreset.HIGH)  # 8192 probes
        grid = DDGICameraRelativeGrid(config=config, origin=Vec3(0, 0, 0))

        data = grid.build_gpu_data()

        # total_probes at offset 44 (after dimensions)
        total = struct.unpack_from("<I", data, 44)[0]
        assert total == 8192

    def test_gpu_data_scroll_offset(self) -> None:
        """Test scroll_offset is at correct offset in GPU data."""
        config = DDGIConfig(preset=DDGIQualityPreset.HIGH)
        grid = DDGICameraRelativeGrid(config=config, origin=Vec3(0, 0, 0))
        grid.scroll_offset = (5, -3, 7)

        data = grid.build_gpu_data()

        # scroll_offset at offset 48 (after total_probes)
        scroll = struct.unpack_from("<3i", data, 48)
        assert scroll == (5, -3, 7)

    def test_gpu_data_frame_index_offset(self) -> None:
        """Test frame_index is at correct offset in GPU data."""
        config = DDGIConfig(preset=DDGIQualityPreset.HIGH)
        grid = DDGICameraRelativeGrid(config=config, origin=Vec3(0, 0, 0))
        grid.frame_index = 12345

        data = grid.build_gpu_data()

        # frame_index at offset 60 (last 4 bytes)
        frame = struct.unpack_from("<I", data, 60)[0]
        assert frame == 12345

    def test_gpu_data_matches_rust_layout(self) -> None:
        """Test GPU data layout matches Rust ProbeGridGpu struct."""
        config = DDGIConfig(
            preset=DDGIQualityPreset.LOW,  # 16x16x4, 4m spacing
        )
        grid = DDGICameraRelativeGrid(config=config, origin=Vec3(1.5, 2.5, 3.5))
        grid.scroll_offset = (1, 2, 3)
        grid.frame_index = 42

        data = grid.build_gpu_data()

        # Unpack entire struct
        # Format: 3f f 3f f 3I I 3i I
        unpacked = struct.unpack("<3ff3ff3II3iI", data)

        # origin: [0:3]
        assert unpacked[0:3] == pytest.approx((1.5, 2.5, 3.5))
        # _pad0: [3]
        assert unpacked[3] == pytest.approx(0.0)
        # cell_size: [4:7]
        assert unpacked[4:7] == pytest.approx((4.0, 4.0, 4.0))
        # _pad1: [7]
        assert unpacked[7] == pytest.approx(0.0)
        # dimensions: [8:11]
        assert unpacked[8:11] == (16, 16, 4)
        # total_probes: [11]
        assert unpacked[11] == 1024
        # scroll_offset: [12:15]
        assert unpacked[12:15] == (1, 2, 3)
        # frame_index: [15]
        assert unpacked[15] == 42


# ============================================================================
# Frame Advancement and Update Scheduling Tests
# ============================================================================


class TestFrameAdvancement:
    """Tests for frame advancement and probe update scheduling."""

    def test_advance_frame(self) -> None:
        """Test frame index advances."""
        config = DDGIConfig(preset=DDGIQualityPreset.HIGH)
        grid = DDGICameraRelativeGrid(config=config)

        assert grid.frame_index == 0
        grid.advance_frame()
        assert grid.frame_index == 1
        grid.advance_frame()
        assert grid.frame_index == 2

    def test_get_probes_to_update(self) -> None:
        """Test probe update scheduling."""
        config = DDGIConfig(preset=DDGIQualityPreset.HIGH)  # 8192 probes, 1/8 update
        grid = DDGICameraRelativeGrid(config=config)

        probes = grid.get_probes_to_update()

        # Should update 1/8 of 8192 = 1024 probes
        assert len(probes) == 1024

        # All indices should be valid
        for ix, iy, iz in probes:
            assert 0 <= ix < 32
            assert 0 <= iy < 32
            assert 0 <= iz < 8

    def test_probes_to_update_cycles(self) -> None:
        """Test all probes are updated over multiple frames."""
        config = DDGIConfig(preset=DDGIQualityPreset.LOW)  # 1024 probes, 1/8 update
        grid = DDGICameraRelativeGrid(config=config)

        updated = set()
        for _ in range(8):  # 8 frames should cover all probes
            probes = grid.get_probes_to_update()
            for idx in probes:
                updated.add(idx)
            grid.advance_frame()

        # All probes should have been updated
        assert len(updated) == 1024


# ============================================================================
# Bounds Tests
# ============================================================================


class TestGridBounds:
    """Tests for grid bounds computation."""

    def test_get_bounds(self) -> None:
        """Test AABB bounds computation."""
        config = DDGIConfig(preset=DDGIQualityPreset.HIGH)  # 32x32x8, 2m spacing
        grid = DDGICameraRelativeGrid(config=config, origin=Vec3(0, 0, 0))

        bounds = grid.get_bounds()

        assert bounds.min == Vec3(0, 0, 0)
        # Extent: (dim - 1) * spacing
        assert bounds.max == Vec3(62, 62, 14)

    def test_bounds_with_offset_origin(self) -> None:
        """Test bounds with non-zero origin."""
        config = DDGIConfig(preset=DDGIQualityPreset.LOW)  # 16x16x4, 4m spacing
        grid = DDGICameraRelativeGrid(config=config, origin=Vec3(10, 20, 30))

        bounds = grid.get_bounds()

        assert bounds.min == Vec3(10, 20, 30)
        # Extent: (16-1)*4=60, (16-1)*4=60, (4-1)*4=12
        assert bounds.max == Vec3(70, 80, 42)


# ============================================================================
# Upload Tracking Tests
# ============================================================================


class TestUploadTracking:
    """Tests for GPU upload state tracking."""

    def test_needs_upload_initially(self) -> None:
        """Test grid needs upload initially."""
        config = DDGIConfig(preset=DDGIQualityPreset.HIGH)
        grid = DDGICameraRelativeGrid(config=config)

        assert grid.needs_gpu_upload() is True

    def test_mark_uploaded(self) -> None:
        """Test marking upload complete."""
        config = DDGIConfig(preset=DDGIQualityPreset.HIGH)
        grid = DDGICameraRelativeGrid(config=config)

        grid.mark_uploaded()
        assert grid.needs_gpu_upload() is False

    def test_scroll_sets_needs_upload(self) -> None:
        """Test scrolling sets needs_upload flag."""
        config = DDGIConfig(preset=DDGIQualityPreset.HIGH)
        grid = DDGICameraRelativeGrid(config=config)

        grid.update_for_camera(Vec3(0, 0, 0))
        grid.mark_uploaded()
        assert grid.needs_gpu_upload() is False

        # Scroll grid
        grid.update_for_camera(Vec3(100, 0, 0))
        assert grid.needs_gpu_upload() is True
