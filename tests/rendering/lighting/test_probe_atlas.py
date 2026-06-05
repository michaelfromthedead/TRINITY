"""Tests for probe atlas management system.

Covers:
- AtlasSlot: allocation, deallocation, UV coordinates, dirty flags
- AtlasLayout: grid dimensions, UV calculations, mip regions
- AtlasConfig: validation, format conversion
- ProbeAtlas: slot management, probe mapping, texture updates
- AtlasUpdater: batch updates, face updates, mip chain updates
- AtlasSampler: bilinear filtering, roughness mip sampling, direction sampling
- ProbeAtlasManager: multi-atlas management, auto-allocation
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import pytest

from engine.core.math.vec import Vec3
from engine.rendering.lighting.baked_probes import (
    BakedProbeConstants,
    CubemapData,
    CubemapFace,
    CubemapFaceData,
    CubemapMipChain,
    HDRPixel,
    MipLevel,
)
from engine.rendering.lighting.probe_atlas import (
    AtlasConstants,
    AtlasFormat,
    AtlasSlot,
    AtlasLayout,
    AtlasConfig,
    ProbeAtlas,
    PendingUpdate,
    AtlasUpdater,
    AtlasSampler,
    ProbeAtlasManager,
)


# -----------------------------------------------------------------------------
# AtlasSlot Tests
# -----------------------------------------------------------------------------

class TestAtlasSlot:
    """Tests for atlas slot."""

    def test_slot_creation(self) -> None:
        """Test creating an atlas slot."""
        slot = AtlasSlot(grid_x=2, grid_y=3)
        assert slot.grid_x == 2
        assert slot.grid_y == 3
        assert slot.probe_id == -1
        assert slot.is_occupied() is False

    def test_slot_is_occupied_initially_false(self) -> None:
        """Test that new slot is not occupied."""
        slot = AtlasSlot(grid_x=0, grid_y=0)
        assert slot.is_occupied() is False

    def test_slot_assign_success(self) -> None:
        """Test assigning a probe to an empty slot."""
        slot = AtlasSlot(grid_x=0, grid_y=0)
        result = slot.assign(probe_id=42)
        assert result is True
        assert slot.probe_id == 42
        assert slot.is_occupied() is True

    def test_slot_assign_fails_when_occupied(self) -> None:
        """Test assigning to an occupied slot fails."""
        slot = AtlasSlot(grid_x=0, grid_y=0)
        slot.assign(probe_id=1)
        result = slot.assign(probe_id=2)
        assert result is False
        assert slot.probe_id == 1

    def test_slot_release_returns_old_id(self) -> None:
        """Test releasing a slot returns the old probe ID."""
        slot = AtlasSlot(grid_x=0, grid_y=0)
        slot.assign(probe_id=99)
        old_id = slot.release()
        assert old_id == 99
        assert slot.probe_id == -1
        assert slot.is_occupied() is False

    def test_slot_release_empty_returns_negative(self) -> None:
        """Test releasing empty slot returns -1."""
        slot = AtlasSlot(grid_x=0, grid_y=0)
        old_id = slot.release()
        assert old_id == -1

    def test_slot_dirty_flag(self) -> None:
        """Test dirty flag operations."""
        slot = AtlasSlot(grid_x=0, grid_y=0)
        assert slot.is_dirty() is False
        slot.mark_dirty()
        assert slot.is_dirty() is True
        slot.mark_clean()
        assert slot.is_dirty() is False

    def test_slot_assign_marks_dirty(self) -> None:
        """Test that assigning marks slot as dirty."""
        slot = AtlasSlot(grid_x=0, grid_y=0)
        slot.assign(probe_id=1)
        assert slot.is_dirty() is True

    def test_slot_get_face_uv_positive_x(self) -> None:
        """Test getting UV coordinates for +X face."""
        layout = AtlasLayout(grid_rows=4, grid_cols=4, probe_resolution=128, padding=2)
        slot = AtlasSlot(grid_x=0, grid_y=0)
        uvs = slot.get_face_uv(CubemapFace.POSITIVE_X, layout)
        assert len(uvs) == 4
        assert uvs[0] >= 0 and uvs[0] <= 1  # u_min
        assert uvs[1] >= 0 and uvs[1] <= 1  # v_min
        assert uvs[2] > uvs[0]  # u_max > u_min
        assert uvs[3] > uvs[1]  # v_max > v_min

    def test_slot_get_face_uv_all_faces(self) -> None:
        """Test getting UV coordinates for all faces."""
        layout = AtlasLayout(grid_rows=4, grid_cols=4, probe_resolution=128, padding=2)
        slot = AtlasSlot(grid_x=1, grid_y=1)
        uvs = slot.get_all_face_uvs(layout)
        assert len(uvs) == 6
        for face in CubemapFace:
            assert face in uvs
            assert len(uvs[face]) == 4


# -----------------------------------------------------------------------------
# AtlasLayout Tests
# -----------------------------------------------------------------------------

class TestAtlasLayout:
    """Tests for atlas layout."""

    def test_layout_creation_default(self) -> None:
        """Test creating layout with defaults."""
        layout = AtlasLayout()
        assert layout.grid_rows == AtlasConstants.DEFAULT_GRID_ROWS
        assert layout.grid_cols == AtlasConstants.DEFAULT_GRID_COLS
        assert layout.probe_resolution == AtlasConstants.DEFAULT_PROBE_RESOLUTION
        assert layout.padding == AtlasConstants.DEFAULT_PADDING

    def test_layout_max_probes(self) -> None:
        """Test max probes calculation."""
        layout = AtlasLayout(grid_rows=4, grid_cols=4)
        assert layout.max_probes == 16

    def test_layout_slot_dimensions(self) -> None:
        """Test slot width and height calculation."""
        layout = AtlasLayout(probe_resolution=128, padding=2)
        # 3 faces wide + 2*padding
        expected_width = 128 * 3 + 2 * 2
        # 2 faces tall + 2*padding
        expected_height = 128 * 2 + 2 * 2
        assert layout.slot_width == expected_width
        assert layout.slot_height == expected_height

    def test_layout_get_slot_valid(self) -> None:
        """Test getting valid slot position."""
        layout = AtlasLayout(grid_rows=4, grid_cols=4, probe_resolution=128, padding=2)
        pos = layout.get_slot(0, 0)
        assert pos == (0, 0)
        pos = layout.get_slot(1, 0)
        assert pos == (layout.slot_width, 0)

    def test_layout_get_slot_out_of_bounds(self) -> None:
        """Test getting slot out of bounds returns None."""
        layout = AtlasLayout(grid_rows=4, grid_cols=4)
        assert layout.get_slot(-1, 0) is None
        assert layout.get_slot(0, -1) is None
        assert layout.get_slot(4, 0) is None
        assert layout.get_slot(0, 4) is None

    def test_layout_get_total_size_without_mips(self) -> None:
        """Test total atlas size without mips."""
        layout = AtlasLayout(
            grid_rows=4,
            grid_cols=4,
            probe_resolution=128,
            padding=2,
            include_mips=False,
        )
        width, height = layout.get_total_size()
        assert width == 4 * layout.slot_width
        assert height == 4 * layout.slot_height

    def test_layout_get_total_size_with_mips(self) -> None:
        """Test total atlas size with mips includes extra space."""
        layout = AtlasLayout(
            grid_rows=4,
            grid_cols=4,
            probe_resolution=128,
            padding=2,
            include_mips=True,
        )
        width, height = layout.get_total_size()
        base_width = 4 * layout.slot_width
        # Width should be larger to accommodate mips
        assert width > base_width

    def test_layout_compute_face_uvs_in_range(self) -> None:
        """Test computed face UVs are in [0, 1] range."""
        layout = AtlasLayout(grid_rows=4, grid_cols=4, probe_resolution=128)
        uvs = layout.compute_face_uvs(0, 0, 0, 0)
        u_min, v_min, u_max, v_max = uvs
        assert 0 <= u_min <= 1
        assert 0 <= v_min <= 1
        assert 0 <= u_max <= 1
        assert 0 <= v_max <= 1

    def test_layout_compute_face_uvs_different_slots(self) -> None:
        """Test face UVs differ between slots."""
        layout = AtlasLayout(grid_rows=4, grid_cols=4, probe_resolution=128)
        uvs_00 = layout.compute_face_uvs(0, 0, 0, 0)
        uvs_11 = layout.compute_face_uvs(1, 1, 0, 0)
        assert uvs_00 != uvs_11

    def test_layout_grid_to_slot_index(self) -> None:
        """Test grid to slot index conversion."""
        layout = AtlasLayout(grid_rows=4, grid_cols=4)
        assert layout.grid_to_slot_index(0, 0) == 0
        assert layout.grid_to_slot_index(1, 0) == 1
        assert layout.grid_to_slot_index(0, 1) == 4
        assert layout.grid_to_slot_index(3, 3) == 15

    def test_layout_slot_index_to_grid(self) -> None:
        """Test slot index to grid conversion."""
        layout = AtlasLayout(grid_rows=4, grid_cols=4)
        assert layout.slot_index_to_grid(0) == (0, 0)
        assert layout.slot_index_to_grid(1) == (1, 0)
        assert layout.slot_index_to_grid(4) == (0, 1)
        assert layout.slot_index_to_grid(15) == (3, 3)

    def test_layout_get_mip_region_base_level(self) -> None:
        """Test getting mip region for base level."""
        layout = AtlasLayout(grid_rows=4, grid_cols=4, probe_resolution=128, include_mips=True)
        region = layout.get_mip_region(0)
        assert region is not None
        assert region[0] == 0  # x starts at 0
        assert region[1] == 0  # y starts at 0

    def test_layout_get_mip_region_invalid_level(self) -> None:
        """Test getting mip region for invalid level returns None."""
        layout = AtlasLayout(grid_rows=4, grid_cols=4, probe_resolution=128, include_mips=True)
        region = layout.get_mip_region(100)
        assert region is None

    def test_layout_get_mip_region_without_mips(self) -> None:
        """Test getting mip region when mips disabled returns None."""
        layout = AtlasLayout(grid_rows=4, grid_cols=4, probe_resolution=128, include_mips=False)
        region = layout.get_mip_region(1)
        assert region is None


# -----------------------------------------------------------------------------
# AtlasConfig Tests
# -----------------------------------------------------------------------------

class TestAtlasConfig:
    """Tests for atlas configuration."""

    def test_config_creation_default(self) -> None:
        """Test creating config with defaults."""
        config = AtlasConfig()
        assert config.grid_size == (4, 4)
        assert config.probe_resolution == 128
        assert config.include_mips is True

    def test_config_grid_properties(self) -> None:
        """Test grid property accessors."""
        config = AtlasConfig(grid_size=(3, 5))
        assert config.grid_rows == 3
        assert config.grid_cols == 5
        assert config.max_probes == 15

    def test_config_to_layout(self) -> None:
        """Test converting config to layout."""
        config = AtlasConfig(
            grid_size=(2, 3),
            probe_resolution=64,
            include_mips=False,
            padding=4,
        )
        layout = config.to_layout()
        assert layout.grid_rows == 2
        assert layout.grid_cols == 3
        assert layout.probe_resolution == 64
        assert layout.include_mips is False
        assert layout.padding == 4

    def test_config_get_atlas_format(self) -> None:
        """Test getting atlas format enum."""
        config = AtlasConfig(format="rgba16f")
        assert config.get_atlas_format() == AtlasFormat.RGBA16F
        config = AtlasConfig(format="rgba32f")
        assert config.get_atlas_format() == AtlasFormat.RGBA32F

    def test_config_format_case_insensitive(self) -> None:
        """Test format string is case insensitive."""
        config = AtlasConfig(format="RGBA16F")
        assert config.get_atlas_format() == AtlasFormat.RGBA16F

    def test_config_validates_resolution(self) -> None:
        """Test that config validates resolution."""
        config = AtlasConfig(probe_resolution=1)
        assert config.probe_resolution >= AtlasConstants.MIN_MIP_RESOLUTION

    def test_config_validates_padding(self) -> None:
        """Test that config validates padding."""
        config = AtlasConfig(padding=-5)
        assert config.padding >= 0


# -----------------------------------------------------------------------------
# ProbeAtlas Tests
# -----------------------------------------------------------------------------

class TestProbeAtlas:
    """Tests for probe atlas."""

    def test_atlas_creation(self) -> None:
        """Test creating a probe atlas."""
        atlas = ProbeAtlas()
        assert atlas.capacity == 16
        assert atlas.allocated_count == 0
        assert atlas.free_count == 16

    def test_atlas_allocate_probe(self) -> None:
        """Test allocating a probe."""
        atlas = ProbeAtlas()
        slot = atlas.allocate_probe(1)
        assert slot is not None
        assert slot.probe_id == 1
        assert atlas.allocated_count == 1
        assert atlas.free_count == 15

    def test_atlas_allocate_probe_duplicate_fails(self) -> None:
        """Test allocating duplicate probe returns None."""
        atlas = ProbeAtlas()
        atlas.allocate_probe(1)
        slot = atlas.allocate_probe(1)
        assert slot is None
        assert atlas.allocated_count == 1

    def test_atlas_deallocate_probe(self) -> None:
        """Test deallocating a probe."""
        atlas = ProbeAtlas()
        atlas.allocate_probe(1)
        result = atlas.deallocate_probe(1)
        assert result is True
        assert atlas.allocated_count == 0
        assert atlas.free_count == 16

    def test_atlas_deallocate_nonexistent(self) -> None:
        """Test deallocating nonexistent probe returns False."""
        atlas = ProbeAtlas()
        result = atlas.deallocate_probe(999)
        assert result is False

    def test_atlas_has_probe(self) -> None:
        """Test checking if probe is in atlas."""
        atlas = ProbeAtlas()
        assert atlas.has_probe(1) is False
        atlas.allocate_probe(1)
        assert atlas.has_probe(1) is True

    def test_atlas_get_slot(self) -> None:
        """Test getting slot for a probe."""
        atlas = ProbeAtlas()
        atlas.allocate_probe(1)
        slot = atlas.get_slot(1)
        assert slot is not None
        assert slot.probe_id == 1

    def test_atlas_get_slot_nonexistent(self) -> None:
        """Test getting slot for nonexistent probe returns None."""
        atlas = ProbeAtlas()
        slot = atlas.get_slot(999)
        assert slot is None

    def test_atlas_get_slot_at(self) -> None:
        """Test getting slot at grid position."""
        atlas = ProbeAtlas()
        slot = atlas.get_slot_at(0, 0)
        assert slot is not None
        assert slot.grid_x == 0
        assert slot.grid_y == 0

    def test_atlas_get_slot_at_out_of_bounds(self) -> None:
        """Test getting slot at invalid position returns None."""
        atlas = ProbeAtlas()
        assert atlas.get_slot_at(-1, 0) is None
        assert atlas.get_slot_at(100, 0) is None

    def test_atlas_is_full(self) -> None:
        """Test checking if atlas is full."""
        config = AtlasConfig(grid_size=(2, 2))
        atlas = ProbeAtlas(config=config)
        assert atlas.is_full() is False
        for i in range(4):
            atlas.allocate_probe(i)
        assert atlas.is_full() is True

    def test_atlas_get_probe_uvs(self) -> None:
        """Test getting UV coordinates for a probe."""
        atlas = ProbeAtlas()
        atlas.allocate_probe(1)
        uvs = atlas.get_probe_uvs(1)
        assert uvs is not None
        assert len(uvs) == 6
        for face in CubemapFace:
            assert face in uvs

    def test_atlas_get_probe_uvs_nonexistent(self) -> None:
        """Test getting UVs for nonexistent probe returns None."""
        atlas = ProbeAtlas()
        uvs = atlas.get_probe_uvs(999)
        assert uvs is None

    def test_atlas_get_face_uvs(self) -> None:
        """Test getting UV coordinates for specific face."""
        atlas = ProbeAtlas()
        atlas.allocate_probe(1)
        uvs = atlas.get_face_uvs(1, CubemapFace.POSITIVE_X)
        assert uvs is not None
        assert len(uvs) == 4

    def test_atlas_update_slot(self) -> None:
        """Test updating a slot with cubemap data."""
        atlas = ProbeAtlas()
        atlas.allocate_probe(1)
        cubemap = CubemapData(resolution=128)
        result = atlas.update_slot(1, cubemap)
        assert result is True

    def test_atlas_update_slot_nonexistent(self) -> None:
        """Test updating nonexistent slot returns False."""
        atlas = ProbeAtlas()
        cubemap = CubemapData(resolution=128)
        result = atlas.update_slot(999, cubemap)
        assert result is False

    def test_atlas_get_dirty_slots(self) -> None:
        """Test getting list of dirty slots."""
        atlas = ProbeAtlas()
        atlas.allocate_probe(1)
        dirty = atlas.get_dirty_slots()
        assert len(dirty) == 1
        assert dirty[0].probe_id == 1

    def test_atlas_clear_dirty_flags(self) -> None:
        """Test clearing dirty flags."""
        atlas = ProbeAtlas()
        atlas.allocate_probe(1)
        atlas.clear_dirty_flags()
        dirty = atlas.get_dirty_slots()
        assert len(dirty) == 0

    def test_atlas_clear(self) -> None:
        """Test clearing all probes."""
        atlas = ProbeAtlas()
        atlas.allocate_probe(1)
        atlas.allocate_probe(2)
        atlas.clear()
        assert atlas.allocated_count == 0
        assert atlas.free_count == atlas.capacity

    def test_atlas_get_allocated_probes(self) -> None:
        """Test getting list of allocated probe IDs."""
        atlas = ProbeAtlas()
        atlas.allocate_probe(5)
        atlas.allocate_probe(10)
        probes = atlas.get_allocated_probes()
        assert len(probes) == 2
        assert 5 in probes
        assert 10 in probes

    def test_atlas_width_height(self) -> None:
        """Test atlas texture dimensions."""
        config = AtlasConfig(grid_size=(4, 4), probe_resolution=128)
        atlas = ProbeAtlas(config=config)
        assert atlas.width > 0
        assert atlas.height > 0


# -----------------------------------------------------------------------------
# AtlasUpdater Tests
# -----------------------------------------------------------------------------

class TestAtlasUpdater:
    """Tests for atlas updater."""

    def test_updater_creation(self) -> None:
        """Test creating an updater."""
        atlas = ProbeAtlas()
        updater = AtlasUpdater(atlas)
        assert updater.pending_count == 0

    def test_updater_update_probe(self) -> None:
        """Test queuing a probe update."""
        atlas = ProbeAtlas()
        atlas.allocate_probe(1)
        updater = AtlasUpdater(atlas)
        cubemap = CubemapData(resolution=128)
        result = updater.update_probe(1, cubemap)
        assert result is True
        assert updater.pending_count == 1

    def test_updater_update_probe_nonexistent(self) -> None:
        """Test updating nonexistent probe returns False."""
        atlas = ProbeAtlas()
        updater = AtlasUpdater(atlas)
        cubemap = CubemapData(resolution=128)
        result = updater.update_probe(999, cubemap)
        assert result is False
        assert updater.pending_count == 0

    def test_updater_update_face(self) -> None:
        """Test queuing a face update."""
        atlas = ProbeAtlas()
        atlas.allocate_probe(1)
        updater = AtlasUpdater(atlas)
        face_data = CubemapFaceData(face=CubemapFace.POSITIVE_X, resolution=128)
        result = updater.update_face(1, CubemapFace.POSITIVE_X, face_data)
        assert result is True
        assert updater.pending_count == 1

    def test_updater_update_face_nonexistent(self) -> None:
        """Test updating face for nonexistent probe returns False."""
        atlas = ProbeAtlas()
        updater = AtlasUpdater(atlas)
        face_data = CubemapFaceData(face=CubemapFace.POSITIVE_X, resolution=128)
        result = updater.update_face(999, CubemapFace.POSITIVE_X, face_data)
        assert result is False

    def test_updater_flush_updates(self) -> None:
        """Test flushing pending updates."""
        atlas = ProbeAtlas()
        atlas.allocate_probe(1)
        updater = AtlasUpdater(atlas)
        cubemap = CubemapData(resolution=128)
        updater.update_probe(1, cubemap)
        count = updater.flush_updates()
        assert count == 1
        assert updater.pending_count == 0

    def test_updater_flush_multiple_updates(self) -> None:
        """Test flushing multiple updates."""
        atlas = ProbeAtlas()
        atlas.allocate_probe(1)
        atlas.allocate_probe(2)
        updater = AtlasUpdater(atlas)
        cubemap = CubemapData(resolution=128)
        updater.update_probe(1, cubemap)
        updater.update_probe(2, cubemap)
        count = updater.flush_updates()
        assert count == 2

    def test_updater_clear_pending(self) -> None:
        """Test clearing pending updates without applying."""
        atlas = ProbeAtlas()
        atlas.allocate_probe(1)
        updater = AtlasUpdater(atlas)
        cubemap = CubemapData(resolution=128)
        updater.update_probe(1, cubemap)
        updater.clear_pending()
        assert updater.pending_count == 0

    def test_updater_auto_flush_at_batch_size(self) -> None:
        """Test auto-flush when batch size is reached."""
        atlas = ProbeAtlas()
        for i in range(5):
            atlas.allocate_probe(i)
        updater = AtlasUpdater(atlas, batch_size=3)
        cubemap = CubemapData(resolution=128)
        for i in range(3):
            updater.update_probe(i, cubemap)
        # Should have auto-flushed
        assert updater.pending_count == 0

    def test_updater_update_mip_chain(self) -> None:
        """Test updating with mip chain."""
        atlas = ProbeAtlas()
        atlas.allocate_probe(1)
        updater = AtlasUpdater(atlas)

        # Create mip chain
        base_cubemap = CubemapData(resolution=128)
        mip_chain = CubemapMipChain(
            base_resolution=128,
            mip_count=2,
            mips=[
                MipLevel(level=0, resolution=128, cubemap=base_cubemap),
                MipLevel(level=1, resolution=64, cubemap=CubemapData(resolution=64)),
            ],
        )

        result = updater.update_mip_chain(1, mip_chain)
        assert result is True
        assert updater.pending_count == 2


# -----------------------------------------------------------------------------
# AtlasSampler Tests
# -----------------------------------------------------------------------------

class TestAtlasSampler:
    """Tests for atlas sampler."""

    @pytest.fixture
    def populated_atlas(self) -> ProbeAtlas:
        """Create an atlas with texture data."""
        config = AtlasConfig(grid_size=(2, 2), probe_resolution=64)
        atlas = ProbeAtlas(config=config)
        atlas.allocate_probe(1)

        # Create cubemap with known data
        cubemap = CubemapData(resolution=64)
        for face in CubemapFace:
            face_data = cubemap.get_face(face)
            for y in range(64):
                for x in range(64):
                    # Gradient based on position
                    pixel = HDRPixel(x / 64.0, y / 64.0, 0.5)
                    face_data.set_pixel(x, y, pixel)

        atlas.update_slot(1, cubemap)
        return atlas

    def test_sampler_creation(self, populated_atlas: ProbeAtlas) -> None:
        """Test creating a sampler."""
        sampler = AtlasSampler(populated_atlas)
        assert sampler is not None

    def test_sampler_sample_basic(self, populated_atlas: ProbeAtlas) -> None:
        """Test basic sampling."""
        sampler = AtlasSampler(populated_atlas)
        pixel = sampler.sample(1, CubemapFace.POSITIVE_X, 0.5, 0.5)
        assert isinstance(pixel, HDRPixel)

    def test_sampler_sample_corner_uvs(self, populated_atlas: ProbeAtlas) -> None:
        """Test sampling at corner UVs."""
        sampler = AtlasSampler(populated_atlas)
        # Top-left
        pixel = sampler.sample(1, CubemapFace.POSITIVE_X, 0.0, 0.0)
        assert isinstance(pixel, HDRPixel)
        # Bottom-right
        pixel = sampler.sample(1, CubemapFace.POSITIVE_X, 1.0, 1.0)
        assert isinstance(pixel, HDRPixel)

    def test_sampler_sample_with_roughness_zero(self, populated_atlas: ProbeAtlas) -> None:
        """Test sampling with zero roughness uses base mip."""
        sampler = AtlasSampler(populated_atlas)
        pixel = sampler.sample_with_roughness(1, CubemapFace.POSITIVE_X, 0.5, 0.5, 0.0)
        assert isinstance(pixel, HDRPixel)

    def test_sampler_sample_with_roughness_high(self, populated_atlas: ProbeAtlas) -> None:
        """Test sampling with high roughness."""
        sampler = AtlasSampler(populated_atlas)
        pixel = sampler.sample_with_roughness(1, CubemapFace.POSITIVE_X, 0.5, 0.5, 1.0)
        assert isinstance(pixel, HDRPixel)

    def test_sampler_sample_direction(self, populated_atlas: ProbeAtlas) -> None:
        """Test sampling using direction vector."""
        sampler = AtlasSampler(populated_atlas)
        direction = Vec3(1, 0, 0)  # +X direction
        pixel = sampler.sample_direction(1, direction)
        assert isinstance(pixel, HDRPixel)

    def test_sampler_sample_direction_with_roughness(self, populated_atlas: ProbeAtlas) -> None:
        """Test direction sampling with roughness."""
        sampler = AtlasSampler(populated_atlas)
        direction = Vec3(0, 1, 0)  # +Y direction
        pixel = sampler.sample_direction(1, direction, roughness=0.5)
        assert isinstance(pixel, HDRPixel)

    def test_sampler_direction_to_face_positive_x(self, populated_atlas: ProbeAtlas) -> None:
        """Test direction maps to correct face (+X)."""
        sampler = AtlasSampler(populated_atlas)
        face, u, v = sampler._direction_to_face_uv(Vec3(1, 0, 0))
        assert face == CubemapFace.POSITIVE_X

    def test_sampler_direction_to_face_negative_x(self, populated_atlas: ProbeAtlas) -> None:
        """Test direction maps to correct face (-X)."""
        sampler = AtlasSampler(populated_atlas)
        face, u, v = sampler._direction_to_face_uv(Vec3(-1, 0, 0))
        assert face == CubemapFace.NEGATIVE_X

    def test_sampler_direction_to_face_positive_y(self, populated_atlas: ProbeAtlas) -> None:
        """Test direction maps to correct face (+Y)."""
        sampler = AtlasSampler(populated_atlas)
        face, u, v = sampler._direction_to_face_uv(Vec3(0, 1, 0))
        assert face == CubemapFace.POSITIVE_Y

    def test_sampler_direction_to_face_negative_y(self, populated_atlas: ProbeAtlas) -> None:
        """Test direction maps to correct face (-Y)."""
        sampler = AtlasSampler(populated_atlas)
        face, u, v = sampler._direction_to_face_uv(Vec3(0, -1, 0))
        assert face == CubemapFace.NEGATIVE_Y

    def test_sampler_direction_to_face_positive_z(self, populated_atlas: ProbeAtlas) -> None:
        """Test direction maps to correct face (+Z)."""
        sampler = AtlasSampler(populated_atlas)
        face, u, v = sampler._direction_to_face_uv(Vec3(0, 0, 1))
        assert face == CubemapFace.POSITIVE_Z

    def test_sampler_direction_to_face_negative_z(self, populated_atlas: ProbeAtlas) -> None:
        """Test direction maps to correct face (-Z)."""
        sampler = AtlasSampler(populated_atlas)
        face, u, v = sampler._direction_to_face_uv(Vec3(0, 0, -1))
        assert face == CubemapFace.NEGATIVE_Z

    def test_sampler_direction_uvs_in_range(self, populated_atlas: ProbeAtlas) -> None:
        """Test that direction UVs are in valid range."""
        sampler = AtlasSampler(populated_atlas)
        directions = [
            Vec3(1, 0.5, 0.5),
            Vec3(-0.5, 1, 0.3),
            Vec3(0.2, -0.8, 0.1),
            Vec3(0.3, 0.4, 1),
            Vec3(-0.2, -0.3, -1),
        ]
        for direction in directions:
            face, u, v = sampler._direction_to_face_uv(direction)
            assert 0 <= u <= 1, f"U out of range for direction {direction}"
            assert 0 <= v <= 1, f"V out of range for direction {direction}"


# -----------------------------------------------------------------------------
# ProbeAtlasManager Tests
# -----------------------------------------------------------------------------

class TestProbeAtlasManager:
    """Tests for probe atlas manager."""

    def test_manager_creation(self) -> None:
        """Test creating a manager."""
        manager = ProbeAtlasManager()
        assert manager.atlas_count == 0
        assert manager.total_capacity == 0

    def test_manager_create_atlas(self) -> None:
        """Test creating an atlas."""
        manager = ProbeAtlasManager()
        atlas = manager.create_atlas()
        assert atlas is not None
        assert manager.atlas_count == 1

    def test_manager_create_atlas_with_config(self) -> None:
        """Test creating atlas with custom config."""
        manager = ProbeAtlasManager()
        config = AtlasConfig(grid_size=(2, 2))
        atlas = manager.create_atlas(config)
        assert atlas.capacity == 4

    def test_manager_allocate_probe(self) -> None:
        """Test allocating a probe."""
        manager = ProbeAtlasManager()
        slot = manager.allocate_probe(1)
        assert slot is not None
        assert manager.total_allocated == 1
        assert manager.atlas_count == 1

    def test_manager_allocate_probe_creates_atlas(self) -> None:
        """Test that allocating creates atlas if needed."""
        manager = ProbeAtlasManager()
        assert manager.atlas_count == 0
        manager.allocate_probe(1)
        assert manager.atlas_count == 1

    def test_manager_allocate_duplicate_fails(self) -> None:
        """Test allocating duplicate probe returns None."""
        manager = ProbeAtlasManager()
        manager.allocate_probe(1)
        slot = manager.allocate_probe(1)
        assert slot is None

    def test_manager_deallocate_probe(self) -> None:
        """Test deallocating a probe."""
        manager = ProbeAtlasManager()
        manager.allocate_probe(1)
        result = manager.deallocate_probe(1)
        assert result is True
        assert manager.total_allocated == 0

    def test_manager_deallocate_nonexistent(self) -> None:
        """Test deallocating nonexistent probe returns False."""
        manager = ProbeAtlasManager()
        result = manager.deallocate_probe(999)
        assert result is False

    def test_manager_get_probe_atlas(self) -> None:
        """Test getting atlas containing a probe."""
        manager = ProbeAtlasManager()
        manager.allocate_probe(1)
        atlas = manager.get_probe_atlas(1)
        assert atlas is not None
        assert atlas.has_probe(1)

    def test_manager_get_probe_atlas_nonexistent(self) -> None:
        """Test getting atlas for nonexistent probe returns None."""
        manager = ProbeAtlasManager()
        atlas = manager.get_probe_atlas(999)
        assert atlas is None

    def test_manager_get_probe_uvs(self) -> None:
        """Test getting UVs through manager."""
        manager = ProbeAtlasManager()
        manager.allocate_probe(1)
        uvs = manager.get_probe_uvs(1)
        assert uvs is not None
        assert len(uvs) == 6

    def test_manager_update_probe(self) -> None:
        """Test updating probe through manager."""
        manager = ProbeAtlasManager()
        manager.allocate_probe(1)
        cubemap = CubemapData(resolution=128)
        result = manager.update_probe(1, cubemap)
        assert result is True

    def test_manager_update_probe_nonexistent(self) -> None:
        """Test updating nonexistent probe returns False."""
        manager = ProbeAtlasManager()
        cubemap = CubemapData(resolution=128)
        result = manager.update_probe(999, cubemap)
        assert result is False

    def test_manager_get_updater(self) -> None:
        """Test getting updater for atlas."""
        manager = ProbeAtlasManager()
        atlas = manager.create_atlas()
        updater = manager.get_updater(atlas)
        assert updater is not None
        assert isinstance(updater, AtlasUpdater)

    def test_manager_get_updater_same_atlas(self) -> None:
        """Test getting same updater for same atlas."""
        manager = ProbeAtlasManager()
        atlas = manager.create_atlas()
        updater1 = manager.get_updater(atlas)
        updater2 = manager.get_updater(atlas)
        assert updater1 is updater2

    def test_manager_flush_all_updates(self) -> None:
        """Test flushing all updates."""
        manager = ProbeAtlasManager()
        manager.allocate_probe(1)
        atlas = manager.get_probe_atlas(1)
        updater = manager.get_updater(atlas)
        cubemap = CubemapData(resolution=128)
        updater.update_probe(1, cubemap)
        count = manager.flush_all_updates()
        assert count == 1

    def test_manager_clear(self) -> None:
        """Test clearing all atlases."""
        manager = ProbeAtlasManager()
        manager.allocate_probe(1)
        manager.allocate_probe(2)
        manager.clear()
        assert manager.atlas_count == 0
        assert manager.total_allocated == 0

    def test_manager_get_all_atlases(self) -> None:
        """Test getting all atlases."""
        manager = ProbeAtlasManager()
        manager.create_atlas()
        manager.create_atlas()
        atlases = manager.get_all_atlases()
        assert len(atlases) == 2

    def test_manager_create_sampler(self) -> None:
        """Test creating sampler for atlas."""
        manager = ProbeAtlasManager()
        atlas = manager.create_atlas()
        sampler = manager.create_sampler(atlas)
        assert sampler is not None
        assert isinstance(sampler, AtlasSampler)

    def test_manager_multi_atlas_overflow(self) -> None:
        """Test probes overflow to new atlas."""
        config = AtlasConfig(grid_size=(2, 2))  # 4 probes per atlas
        manager = ProbeAtlasManager(default_config=config)
        for i in range(6):
            manager.allocate_probe(i)
        assert manager.atlas_count == 2
        assert manager.total_allocated == 6

    def test_manager_total_capacity(self) -> None:
        """Test total capacity across atlases."""
        config = AtlasConfig(grid_size=(2, 2))
        manager = ProbeAtlasManager(default_config=config)
        manager.create_atlas()
        manager.create_atlas()
        assert manager.total_capacity == 8


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------

class TestProbeAtlasIntegration:
    """Integration tests for probe atlas system."""

    def test_full_workflow(self) -> None:
        """Test complete workflow: allocate, update, sample."""
        # Create manager and atlas
        config = AtlasConfig(grid_size=(4, 4), probe_resolution=64)
        manager = ProbeAtlasManager(default_config=config)

        # Allocate probes
        for i in range(4):
            manager.allocate_probe(i)

        # Update probes with cubemap data
        for i in range(4):
            cubemap = CubemapData(resolution=64)
            # Fill with test pattern
            for face in CubemapFace:
                face_data = cubemap.get_face(face)
                for y in range(64):
                    for x in range(64):
                        pixel = HDRPixel(i / 4.0, x / 64.0, y / 64.0)
                        face_data.set_pixel(x, y, pixel)
            manager.update_probe(i, cubemap)

        # Sample from probes
        atlas = manager.get_probe_atlas(0)
        sampler = manager.create_sampler(atlas)

        for i in range(4):
            pixel = sampler.sample(i, CubemapFace.POSITIVE_X, 0.5, 0.5)
            assert pixel is not None

    def test_batch_update_workflow(self) -> None:
        """Test batch update workflow."""
        manager = ProbeAtlasManager()
        atlas = manager.create_atlas()
        updater = manager.get_updater(atlas)

        # Allocate probes
        for i in range(4):
            atlas.allocate_probe(i)

        # Queue updates
        for i in range(4):
            cubemap = CubemapData(resolution=128)
            updater.update_probe(i, cubemap)

        # Flush
        count = updater.flush_updates()
        assert count == 4

    def test_mip_sampling_workflow(self) -> None:
        """Test mip level sampling workflow."""
        config = AtlasConfig(grid_size=(2, 2), probe_resolution=64, include_mips=True)
        atlas = ProbeAtlas(config=config)
        atlas.allocate_probe(1)

        # Create and update cubemap
        cubemap = CubemapData(resolution=64)
        atlas.update_slot(1, cubemap)

        # Sample at different roughness levels
        sampler = AtlasSampler(atlas)
        for roughness in [0.0, 0.25, 0.5, 0.75, 1.0]:
            pixel = sampler.sample_with_roughness(
                1, CubemapFace.POSITIVE_X, 0.5, 0.5, roughness
            )
            assert pixel is not None

    def test_uv_no_bleeding(self) -> None:
        """Test that UVs don't bleed between probes."""
        config = AtlasConfig(grid_size=(2, 2), probe_resolution=64, padding=2)
        atlas = ProbeAtlas(config=config)

        # Allocate adjacent probes
        atlas.allocate_probe(1)  # Will get slot (0,0)
        atlas.allocate_probe(2)  # Will get slot (1,0)

        # Get UVs
        uvs1 = atlas.get_face_uvs(1, CubemapFace.POSITIVE_X)
        uvs2 = atlas.get_face_uvs(2, CubemapFace.POSITIVE_X)

        # Ensure no overlap
        assert uvs1 is not None
        assert uvs2 is not None
        # UV max of probe 1 should not exceed UV min of probe 2
        assert uvs1[2] < uvs2[0]  # u_max < next u_min (or they're in different rows)

    def test_full_capacity_handling(self) -> None:
        """Test handling full capacity atlas."""
        config = AtlasConfig(grid_size=(2, 2))  # Only 4 slots
        atlas = ProbeAtlas(config=config)

        # Fill atlas
        for i in range(4):
            slot = atlas.allocate_probe(i)
            assert slot is not None

        # Atlas should be full
        assert atlas.is_full() is True

        # Next allocation should fail
        slot = atlas.allocate_probe(5)
        assert slot is None

        # Deallocate one
        atlas.deallocate_probe(0)
        assert atlas.is_full() is False

        # Now allocation should succeed
        slot = atlas.allocate_probe(5)
        assert slot is not None
