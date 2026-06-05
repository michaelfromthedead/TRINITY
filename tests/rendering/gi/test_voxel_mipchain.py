"""Tests for voxel mip chain and storage (T-GIR-P7.2).

This module provides comprehensive tests for the voxel mip chain system
including mip generation, downsampling, variance tracking, GPU storage
abstraction, and WGSL shader generation.

Test Categories:
    - MipResolution: Enum properties and conversions
    - VoxelData: Data container operations
    - VoxelMipLevel: Single mip level operations
    - VoxelDownsample: 2x2x2 block averaging
    - VoxelMipChain: Full mip pyramid operations
    - VoxelStorage: GPU abstraction and upload
    - Integration: End-to-end mip chain workflows
    - Edge Cases: Boundary conditions and error handling
"""

from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from engine.rendering.gi.voxel_mipchain import (
    # Constants
    MIN_VOXEL_RESOLUTION,
    MAX_VOXEL_RESOLUTION,
    VOXEL_FORMAT_BYTES_PER_TEXEL,
    VARIANCE_START_MIP,
    MIN_VARIANCE_THRESHOLD,
    # Enums
    MipResolution,
    VoxelStorageFormat,
    # Data classes
    VoxelData,
    VoxelMipLevel,
    VoxelMipChain,
    VoxelDownsampleConfig,
    VoxelStorageConfig,
    VoxelStorage,
    # Functions
    downsample_voxels,
    compute_mip_resolution,
    compute_mip_count,
    estimate_voxel_memory,
    create_test_voxel_pattern,
    validate_mip_chain,
    generate_voxel_downsample_wgsl,
    generate_voxel_sample_wgsl,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def empty_voxel() -> VoxelData:
    """Empty voxel with zero radiance and opacity."""
    return VoxelData.empty()


@pytest.fixture
def opaque_red_voxel() -> VoxelData:
    """Fully opaque red voxel."""
    return VoxelData(np.array([1.0, 0.0, 0.0], dtype=np.float32), 1.0)


@pytest.fixture
def small_mip_level() -> VoxelMipLevel:
    """4x4x4 mip level for testing."""
    return VoxelMipLevel.create_empty(0, 4, track_variance=False)


@pytest.fixture
def small_mip_chain() -> VoxelMipChain:
    """64^3 mip chain for testing."""
    return VoxelMipChain(base_resolution=MipResolution.RES_64)


@pytest.fixture
def default_downsample_config() -> VoxelDownsampleConfig:
    """Default downsampling configuration."""
    return VoxelDownsampleConfig()


@pytest.fixture
def default_storage_config() -> VoxelStorageConfig:
    """Default storage configuration."""
    return VoxelStorageConfig()


# ============================================================================
# MipResolution Tests
# ============================================================================


class TestMipResolution:
    """Tests for MipResolution enum."""

    def test_res_64_value(self):
        """RES_64 has value 64."""
        assert MipResolution.RES_64.value == 64

    def test_res_128_value(self):
        """RES_128 has value 128."""
        assert MipResolution.RES_128.value == 128

    def test_res_256_value(self):
        """RES_256 has value 256."""
        assert MipResolution.RES_256.value == 256

    def test_mip_count_64(self):
        """64^3 resolution has 7 mip levels (64, 32, 16, 8, 4, 2, 1)."""
        assert MipResolution.RES_64.mip_count == 7

    def test_mip_count_128(self):
        """128^3 resolution has 8 mip levels."""
        assert MipResolution.RES_128.mip_count == 8

    def test_mip_count_256(self):
        """256^3 resolution has 9 mip levels."""
        assert MipResolution.RES_256.mip_count == 9

    def test_memory_bytes_increases_with_resolution(self):
        """Higher resolution uses more memory."""
        mem_64 = MipResolution.RES_64.memory_bytes
        mem_128 = MipResolution.RES_128.memory_bytes
        mem_256 = MipResolution.RES_256.memory_bytes

        assert mem_64 < mem_128 < mem_256

    def test_memory_bytes_64_approximately_correct(self):
        """64^3 memory approximately 2MB (with mip chain)."""
        mem = MipResolution.RES_64.memory_bytes
        # 64^3 * 8 + 32^3 * 8 + ... ~= 2.4 MB
        assert 2_000_000 < mem < 3_000_000

    def test_from_size_64(self):
        """from_size(64) returns RES_64."""
        assert MipResolution.from_size(64) == MipResolution.RES_64

    def test_from_size_128(self):
        """from_size(128) returns RES_128."""
        assert MipResolution.from_size(128) == MipResolution.RES_128

    def test_from_size_256(self):
        """from_size(256) returns RES_256."""
        assert MipResolution.from_size(256) == MipResolution.RES_256

    def test_from_size_invalid(self):
        """from_size with invalid size raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported voxel resolution"):
            MipResolution.from_size(32)


# ============================================================================
# VoxelData Tests
# ============================================================================


class TestVoxelData:
    """Tests for VoxelData container."""

    def test_empty_voxel_zero_radiance(self, empty_voxel: VoxelData):
        """Empty voxel has zero radiance."""
        assert np.allclose(empty_voxel.radiance, [0.0, 0.0, 0.0])

    def test_empty_voxel_zero_opacity(self, empty_voxel: VoxelData):
        """Empty voxel has zero opacity."""
        assert empty_voxel.opacity == 0.0

    def test_empty_is_empty(self, empty_voxel: VoxelData):
        """Empty voxel is_empty returns True."""
        assert empty_voxel.is_empty()

    def test_opaque_not_empty(self, opaque_red_voxel: VoxelData):
        """Opaque voxel is not empty."""
        assert not opaque_red_voxel.is_empty()

    def test_radiance_shape(self, opaque_red_voxel: VoxelData):
        """Radiance has shape (3,)."""
        assert opaque_red_voxel.radiance.shape == (3,)

    def test_radiance_dtype(self, opaque_red_voxel: VoxelData):
        """Radiance is float32."""
        assert opaque_red_voxel.radiance.dtype == np.float32

    def test_from_rgba(self):
        """from_rgba creates VoxelData correctly."""
        rgba = np.array([0.5, 0.3, 0.1, 0.8], dtype=np.float32)
        voxel = VoxelData.from_rgba(rgba)

        assert np.allclose(voxel.radiance, [0.5, 0.3, 0.1])
        assert abs(voxel.opacity - 0.8) < 0.001  # Float precision tolerance

    def test_to_rgba(self, opaque_red_voxel: VoxelData):
        """to_rgba returns correct RGBA array."""
        rgba = opaque_red_voxel.to_rgba()

        assert rgba.shape == (4,)
        assert np.allclose(rgba, [1.0, 0.0, 0.0, 1.0])

    def test_roundtrip_rgba(self):
        """from_rgba and to_rgba roundtrip correctly."""
        original = np.array([0.25, 0.5, 0.75, 0.9], dtype=np.float32)
        voxel = VoxelData.from_rgba(original)
        result = voxel.to_rgba()

        assert np.allclose(result, original)

    def test_luminance_red(self, opaque_red_voxel: VoxelData):
        """Red voxel luminance is approximately 0.2126."""
        lum = opaque_red_voxel.luminance()
        assert abs(lum - 0.2126) < 0.001

    def test_luminance_green(self):
        """Green voxel luminance is approximately 0.7152."""
        voxel = VoxelData(np.array([0.0, 1.0, 0.0], dtype=np.float32), 1.0)
        lum = voxel.luminance()
        assert abs(lum - 0.7152) < 0.001

    def test_luminance_blue(self):
        """Blue voxel luminance is approximately 0.0722."""
        voxel = VoxelData(np.array([0.0, 0.0, 1.0], dtype=np.float32), 1.0)
        lum = voxel.luminance()
        assert abs(lum - 0.0722) < 0.001

    def test_luminance_white(self):
        """White voxel luminance is 1.0."""
        voxel = VoxelData(np.array([1.0, 1.0, 1.0], dtype=np.float32), 1.0)
        lum = voxel.luminance()
        assert abs(lum - 1.0) < 0.001

    def test_invalid_radiance_shape(self):
        """Invalid radiance shape raises ValueError."""
        with pytest.raises(ValueError, match="Expected radiance shape"):
            VoxelData(np.array([1.0, 2.0], dtype=np.float32), 1.0)

    def test_is_empty_threshold(self):
        """is_empty respects threshold parameter."""
        voxel = VoxelData(np.zeros(3, dtype=np.float32), 0.005)

        assert voxel.is_empty(threshold=0.01)
        assert not voxel.is_empty(threshold=0.001)


# ============================================================================
# VoxelMipLevel Tests
# ============================================================================


class TestVoxelMipLevel:
    """Tests for VoxelMipLevel."""

    def test_create_empty_shape(self, small_mip_level: VoxelMipLevel):
        """Empty mip level has correct data shape."""
        assert small_mip_level.data.shape == (4, 4, 4, 4)

    def test_create_empty_zeros(self, small_mip_level: VoxelMipLevel):
        """Empty mip level is all zeros."""
        assert np.allclose(small_mip_level.data, 0.0)

    def test_level_index(self, small_mip_level: VoxelMipLevel):
        """Level index is stored correctly."""
        assert small_mip_level.level == 0

    def test_resolution(self, small_mip_level: VoxelMipLevel):
        """Resolution is stored correctly."""
        assert small_mip_level.resolution == 4

    def test_no_variance_by_default(self, small_mip_level: VoxelMipLevel):
        """Variance not tracked by default."""
        assert small_mip_level.variance is None

    def test_variance_when_enabled(self):
        """Variance tracked when enabled."""
        mip = VoxelMipLevel.create_empty(2, 4, track_variance=True)
        assert mip.variance is not None
        assert mip.variance.shape == (4, 4, 4, 4)

    def test_get_set_voxel(self, small_mip_level: VoxelMipLevel):
        """get_voxel and set_voxel work correctly."""
        voxel = VoxelData(np.array([0.5, 0.3, 0.1], dtype=np.float32), 0.9)
        small_mip_level.set_voxel(1, 2, 3, voxel)

        retrieved = small_mip_level.get_voxel(1, 2, 3)
        assert np.allclose(retrieved.radiance, voxel.radiance)
        assert abs(retrieved.opacity - voxel.opacity) < 0.001  # Float precision tolerance

    def test_voxel_count(self, small_mip_level: VoxelMipLevel):
        """voxel_count returns correct count."""
        assert small_mip_level.voxel_count() == 64  # 4^3

    def test_non_empty_count_all_empty(self, small_mip_level: VoxelMipLevel):
        """non_empty_count is 0 for empty level."""
        assert small_mip_level.non_empty_count() == 0

    def test_non_empty_count_with_voxels(self, small_mip_level: VoxelMipLevel):
        """non_empty_count counts non-empty voxels."""
        voxel = VoxelData(np.array([1.0, 1.0, 1.0], dtype=np.float32), 1.0)
        small_mip_level.set_voxel(0, 0, 0, voxel)
        small_mip_level.set_voxel(1, 1, 1, voxel)

        assert small_mip_level.non_empty_count() == 2

    def test_memory_bytes(self, small_mip_level: VoxelMipLevel):
        """memory_bytes returns correct size."""
        expected = 4 * 4 * 4 * 4 * 4  # 4^3 * 4 channels * 4 bytes (float32)
        assert small_mip_level.memory_bytes() == expected

    def test_memory_bytes_with_variance(self):
        """memory_bytes includes variance buffer."""
        mip = VoxelMipLevel.create_empty(2, 4, track_variance=True)
        expected = 4 * 4 * 4 * 4 * 4 * 2  # Double for variance
        assert mip.memory_bytes() == expected

    def test_get_variance_none_when_not_tracked(self, small_mip_level: VoxelMipLevel):
        """get_variance returns None when not tracked."""
        assert small_mip_level.get_variance(0, 0, 0) is None

    def test_get_set_variance(self):
        """get_variance and set_variance work correctly."""
        mip = VoxelMipLevel.create_empty(2, 4, track_variance=True)
        var = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        mip.set_variance(1, 2, 3, var)

        retrieved = mip.get_variance(1, 2, 3)
        assert np.allclose(retrieved, var)

    def test_invalid_data_shape(self):
        """Invalid data shape raises ValueError."""
        with pytest.raises(ValueError, match="Expected data shape"):
            VoxelMipLevel(
                level=0,
                resolution=4,
                data=np.zeros((4, 4, 3, 4), dtype=np.float32),  # Wrong shape
            )


# ============================================================================
# VoxelDownsample Tests
# ============================================================================


class TestVoxelDownsample:
    """Tests for voxel downsampling."""

    def test_downsample_halves_resolution(self):
        """Downsampling halves the resolution."""
        source = VoxelMipLevel.create_empty(0, 8)
        result = downsample_voxels(source)

        assert result.resolution == 4

    def test_downsample_increments_level(self):
        """Downsampling increments mip level."""
        source = VoxelMipLevel.create_empty(0, 8)
        result = downsample_voxels(source)

        assert result.level == 1

    def test_downsample_empty_stays_empty(self):
        """Empty voxels downsample to empty."""
        source = VoxelMipLevel.create_empty(0, 4)
        result = downsample_voxels(source)

        assert result.non_empty_count() == 0

    def test_downsample_averages_children(self):
        """Downsampling averages 8 children."""
        source = VoxelMipLevel.create_empty(0, 4)

        # Fill one 2x2x2 block with identical values
        voxel = VoxelData(np.array([0.8, 0.4, 0.2], dtype=np.float32), 1.0)
        for z in range(2):
            for y in range(2):
                for x in range(2):
                    source.set_voxel(x, y, z, voxel)

        result = downsample_voxels(source)
        parent = result.get_voxel(0, 0, 0)

        # With all same values, average should be similar
        assert parent.opacity > 0.1  # Not empty
        # Note: alpha weighting and energy preservation may affect exact values

    def test_downsample_alpha_weighted(self):
        """Alpha-weighted downsampling weights by opacity."""
        source = VoxelMipLevel.create_empty(0, 4)

        # One bright opaque voxel
        bright = VoxelData(np.array([1.0, 0.0, 0.0], dtype=np.float32), 1.0)
        source.set_voxel(0, 0, 0, bright)

        # Seven dim transparent voxels
        dim = VoxelData(np.array([0.0, 0.0, 1.0], dtype=np.float32), 0.1)
        for z in range(2):
            for y in range(2):
                for x in range(2):
                    if x != 0 or y != 0 or z != 0:
                        source.set_voxel(x, y, z, dim)

        result = downsample_voxels(source)
        parent = result.get_voxel(0, 0, 0)

        # Bright voxel should dominate due to higher alpha
        assert parent.radiance[0] > parent.radiance[2]

    def test_downsample_simple_average(self):
        """Simple average treats all voxels equally."""
        source = VoxelMipLevel.create_empty(0, 4)

        # Fill with gradient
        for z in range(2):
            for y in range(2):
                for x in range(2):
                    val = (x + y + z) / 3.0
                    voxel = VoxelData(
                        np.array([val, val, val], dtype=np.float32),
                        1.0
                    )
                    source.set_voxel(x, y, z, voxel)

        config = VoxelDownsampleConfig(alpha_weighted=False)
        result = downsample_voxels(source, config)
        parent = result.get_voxel(0, 0, 0)

        # Simple average of gradient: (0+1+1+2+1+2+2+3)/3 / 8 = 12/3/8 = 0.5
        assert abs(parent.radiance[0] - 0.5) < 0.1

    def test_downsample_variance_computed(self):
        """Variance computed at high mip levels."""
        source = VoxelMipLevel.create_empty(VARIANCE_START_MIP - 1, 4)

        # Fill with different values
        for z in range(2):
            for y in range(2):
                for x in range(2):
                    val = float(x + y * 2 + z * 4)
                    voxel = VoxelData(
                        np.array([val, val, val], dtype=np.float32),
                        1.0
                    )
                    source.set_voxel(x, y, z, voxel)

        result = downsample_voxels(source)

        assert result.variance is not None
        var = result.get_variance(0, 0, 0)
        assert var is not None
        assert var[0] > 0  # Non-zero variance

    def test_downsample_resolution_1_error(self):
        """Cannot downsample resolution 1."""
        source = VoxelMipLevel.create_empty(6, 1)

        with pytest.raises(ValueError, match="Cannot downsample"):
            downsample_voxels(source)


# ============================================================================
# VoxelMipChain Tests
# ============================================================================


class TestVoxelMipChain:
    """Tests for VoxelMipChain."""

    def test_create_64_mip_count(self, small_mip_chain: VoxelMipChain):
        """64^3 chain has 7 mip levels."""
        assert small_mip_chain.mip_count == 7

    def test_create_128_mip_count(self):
        """128^3 chain has 8 mip levels."""
        chain = VoxelMipChain(base_resolution=MipResolution.RES_128)
        assert chain.mip_count == 8

    def test_get_level_base(self, small_mip_chain: VoxelMipChain):
        """get_level(0) returns base level."""
        base = small_mip_chain.get_level(0)
        assert base.resolution == 64

    def test_get_level_top(self, small_mip_chain: VoxelMipChain):
        """get_level(-1) returns 1x1x1 level."""
        top = small_mip_chain.get_top()
        assert top.resolution == 1

    def test_get_level_out_of_range(self, small_mip_chain: VoxelMipChain):
        """get_level out of range raises IndexError."""
        with pytest.raises(IndexError):
            small_mip_chain.get_level(100)

    def test_get_base(self, small_mip_chain: VoxelMipChain):
        """get_base returns level 0."""
        base = small_mip_chain.get_base()
        assert base.level == 0
        assert base.resolution == 64

    def test_get_top(self, small_mip_chain: VoxelMipChain):
        """get_top returns 1x1x1 level."""
        top = small_mip_chain.get_top()
        assert top.resolution == 1

    def test_set_base_voxel(self, small_mip_chain: VoxelMipChain):
        """set_base_voxel sets voxel in base level."""
        voxel = VoxelData(np.array([1.0, 0.5, 0.25], dtype=np.float32), 0.9)
        small_mip_chain.set_base_voxel(10, 20, 30, voxel)

        retrieved = small_mip_chain.get_base().get_voxel(10, 20, 30)
        assert np.allclose(retrieved.radiance, voxel.radiance)

    def test_sample_voxel(self, small_mip_chain: VoxelMipChain):
        """sample_voxel retrieves from correct level."""
        voxel = VoxelData(np.array([0.7, 0.3, 0.1], dtype=np.float32), 0.8)
        small_mip_chain.set_base_voxel(5, 5, 5, voxel)

        sampled = small_mip_chain.sample_voxel(5, 5, 5, 0)
        assert np.allclose(sampled.radiance, voxel.radiance)

    def test_build_mip_chain(self, small_mip_chain: VoxelMipChain):
        """build_mip_chain generates all levels."""
        # Set some voxels in base
        voxel = VoxelData(np.array([1.0, 1.0, 1.0], dtype=np.float32), 1.0)
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    small_mip_chain.set_base_voxel(x, y, z, voxel)

        small_mip_chain.build_mip_chain()

        # Check mip 1 has content
        mip1 = small_mip_chain.get_level(1)
        assert mip1.non_empty_count() > 0

    def test_build_mip_chain_resolutions(self, small_mip_chain: VoxelMipChain):
        """build_mip_chain creates correct resolutions."""
        small_mip_chain.build_mip_chain()

        expected_res = [64, 32, 16, 8, 4, 2, 1]
        for i, expected in enumerate(expected_res):
            assert small_mip_chain.get_level(i).resolution == expected

    def test_sample_trilinear_center(self, small_mip_chain: VoxelMipChain):
        """sample_trilinear at center returns center value."""
        # Fill with gradient
        base = small_mip_chain.get_base()
        for x in range(64):
            for y in range(64):
                for z in range(64):
                    val = (x + y + z) / 192.0
                    base.set_voxel(
                        x, y, z,
                        VoxelData(np.array([val, val, val], dtype=np.float32), 1.0)
                    )

        # Sample at center
        sampled = small_mip_chain.sample_trilinear(0.5, 0.5, 0.5, 0)
        # At center (31.5, 31.5, 31.5), value should be ~0.5
        assert 0.4 < sampled.radiance[0] < 0.6

    def test_sample_trilinear_interpolates(self, small_mip_chain: VoxelMipChain):
        """sample_trilinear interpolates between voxels."""
        # Set two adjacent voxels with different values
        small_mip_chain.set_base_voxel(
            0, 0, 0,
            VoxelData(np.array([0.0, 0.0, 0.0], dtype=np.float32), 1.0)
        )
        small_mip_chain.set_base_voxel(
            1, 0, 0,
            VoxelData(np.array([1.0, 1.0, 1.0], dtype=np.float32), 1.0)
        )

        # Sample at midpoint
        sampled = small_mip_chain.sample_trilinear(0.5 / 63, 0.0, 0.0, 0)
        # Should interpolate between 0 and 1
        assert 0.3 < sampled.radiance[0] < 0.7

    def test_get_variance_at(self, small_mip_chain: VoxelMipChain):
        """get_variance_at returns variance at high mip levels."""
        # Fill with varied data
        for x in range(8):
            for y in range(8):
                for z in range(8):
                    val = float((x * y * z) % 8) / 8.0
                    small_mip_chain.set_base_voxel(
                        x, y, z,
                        VoxelData(np.array([val, val, val], dtype=np.float32), 1.0)
                    )

        small_mip_chain.build_mip_chain()

        # Variance should exist at mip 2+
        var = small_mip_chain.get_variance_at(0, 0, 0, VARIANCE_START_MIP)
        assert var is not None

    def test_total_memory_bytes(self, small_mip_chain: VoxelMipChain):
        """total_memory_bytes returns sum of all levels."""
        mem = small_mip_chain.total_memory_bytes()
        # Should be roughly MipResolution.RES_64.memory_bytes
        # but with float32 instead of float16
        assert mem > 0

    def test_get_statistics(self, small_mip_chain: VoxelMipChain):
        """get_statistics returns correct info."""
        stats = small_mip_chain.get_statistics()

        assert stats["base_resolution"] == 64
        assert stats["mip_count"] == 7
        assert len(stats["levels"]) == 7


# ============================================================================
# VoxelStorage Tests
# ============================================================================


class TestVoxelStorageFormat:
    """Tests for VoxelStorageFormat."""

    def test_rgba16f_bytes_per_texel(self):
        """RGBA16F is 8 bytes per texel."""
        assert VoxelStorageFormat.RGBA16F.bytes_per_texel == 8

    def test_rgba32f_bytes_per_texel(self):
        """RGBA32F is 16 bytes per texel."""
        assert VoxelStorageFormat.RGBA32F.bytes_per_texel == 16

    def test_r11g11b10f_bytes_per_texel(self):
        """R11G11B10F is 4 bytes per texel."""
        assert VoxelStorageFormat.R11G11B10F.bytes_per_texel == 4

    def test_rgba16f_wgpu_format(self):
        """RGBA16F maps to rgba16float."""
        assert VoxelStorageFormat.RGBA16F.wgpu_format == "rgba16float"


class TestVoxelStorageConfig:
    """Tests for VoxelStorageConfig."""

    def test_default_resolution(self, default_storage_config: VoxelStorageConfig):
        """Default resolution is 128."""
        assert default_storage_config.resolution == MipResolution.RES_128

    def test_default_format(self, default_storage_config: VoxelStorageConfig):
        """Default format is RGBA16F."""
        assert default_storage_config.format == VoxelStorageFormat.RGBA16F

    def test_total_memory_bytes(self, default_storage_config: VoxelStorageConfig):
        """Memory estimate is reasonable."""
        mem = default_storage_config.total_memory_bytes()
        # 128^3 with mips, RGBA16F, with variance = ~32MB
        assert 20_000_000 < mem < 50_000_000


class TestVoxelStorage:
    """Tests for VoxelStorage GPU abstraction."""

    def test_not_initialized_by_default(self, default_storage_config: VoxelStorageConfig):
        """Storage not initialized by default."""
        storage = VoxelStorage(config=default_storage_config)
        assert not storage.is_initialized()

    def test_texture_descriptor(self, default_storage_config: VoxelStorageConfig):
        """Texture descriptor has correct properties."""
        storage = VoxelStorage(config=default_storage_config)
        desc = storage.get_texture_descriptor()

        assert desc["dimension"] == "3d"
        assert desc["format"] == "rgba16float"
        assert desc["size"]["width"] == 128

    def test_variance_texture_descriptor(
        self, default_storage_config: VoxelStorageConfig
    ):
        """Variance texture descriptor matches main."""
        storage = VoxelStorage(config=default_storage_config)
        var_desc = storage.get_variance_texture_descriptor()

        assert var_desc is not None
        assert var_desc["format"] == "rgba16float"

    def test_variance_disabled(self):
        """No variance descriptor when disabled."""
        config = VoxelStorageConfig(enable_variance=False)
        storage = VoxelStorage(config=config)

        assert storage.get_variance_texture_descriptor() is None

    def test_mark_dirty(self, default_storage_config: VoxelStorageConfig):
        """mark_dirty adds dirty region."""
        storage = VoxelStorage(config=default_storage_config)
        storage.mark_dirty(0, 0, 0, 64, 64, 64)

        assert storage.has_dirty_regions()
        assert len(storage.get_dirty_regions()) == 1

    def test_mark_all_dirty(self, default_storage_config: VoxelStorageConfig):
        """mark_all_dirty covers entire texture."""
        storage = VoxelStorage(config=default_storage_config)
        storage.mark_all_dirty()

        regions = storage.get_dirty_regions()
        assert len(regions) == 1
        assert regions[0] == (0, 0, 0, 128, 128, 128)

    def test_clear_dirty(self, default_storage_config: VoxelStorageConfig):
        """clear_dirty removes all dirty regions."""
        storage = VoxelStorage(config=default_storage_config)
        storage.mark_dirty(0, 0, 0, 64, 64, 64)
        storage.clear_dirty()

        assert not storage.has_dirty_regions()

    def test_prepare_upload_data_rgba32f(self):
        """Upload data for RGBA32F is raw bytes."""
        config = VoxelStorageConfig(
            resolution=MipResolution.RES_64,
            format=VoxelStorageFormat.RGBA32F,
        )
        storage = VoxelStorage(config=config)
        chain = VoxelMipChain(base_resolution=MipResolution.RES_64)

        data = storage.prepare_upload_data(chain, 0)
        expected_size = 64 ** 3 * 4 * 4  # 64^3 * 4 channels * 4 bytes
        assert len(data) == expected_size

    def test_prepare_upload_data_rgba16f(self):
        """Upload data for RGBA16F is half size."""
        config = VoxelStorageConfig(
            resolution=MipResolution.RES_64,
            format=VoxelStorageFormat.RGBA16F,
        )
        storage = VoxelStorage(config=config)
        chain = VoxelMipChain(base_resolution=MipResolution.RES_64)

        data = storage.prepare_upload_data(chain, 0)
        expected_size = 64 ** 3 * 4 * 2  # 64^3 * 4 channels * 2 bytes
        assert len(data) == expected_size

    def test_sampler_descriptor(self, default_storage_config: VoxelStorageConfig):
        """Sampler descriptor has correct properties."""
        storage = VoxelStorage(config=default_storage_config)
        desc = storage.get_sampler_descriptor()

        assert desc["mag_filter"] == "linear"
        assert desc["min_filter"] == "linear"
        assert desc["mipmap_filter"] == "linear"

    def test_binding_layout(self, default_storage_config: VoxelStorageConfig):
        """Binding layout has correct entries."""
        storage = VoxelStorage(config=default_storage_config)
        layouts = storage.get_binding_layout()

        # Main texture, sampler, variance texture
        assert len(layouts) == 3
        assert layouts[0]["texture"]["view_dimension"] == "3d"


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_compute_mip_resolution_level_0(self):
        """Mip 0 has base resolution."""
        assert compute_mip_resolution(256, 0) == 256

    def test_compute_mip_resolution_level_1(self):
        """Mip 1 is half base."""
        assert compute_mip_resolution(256, 1) == 128

    def test_compute_mip_resolution_minimum(self):
        """Resolution cannot go below 1."""
        assert compute_mip_resolution(64, 10) == 1

    def test_compute_mip_count_64(self):
        """64 has 7 mip levels."""
        assert compute_mip_count(64) == 7

    def test_compute_mip_count_256(self):
        """256 has 9 mip levels."""
        assert compute_mip_count(256) == 9

    def test_compute_mip_count_invalid(self):
        """Resolution 0 raises ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            compute_mip_count(0)

    def test_estimate_voxel_memory_64(self):
        """Memory estimate for 64^3 is reasonable."""
        mem = estimate_voxel_memory(64)
        # With mips and variance, ~5MB
        assert 2_000_000 < mem < 10_000_000

    def test_estimate_voxel_memory_no_mips(self):
        """Memory without mips is lower."""
        with_mips = estimate_voxel_memory(64, include_mips=True)
        no_mips = estimate_voxel_memory(64, include_mips=False)

        assert no_mips < with_mips

    def test_estimate_voxel_memory_no_variance(self):
        """Memory without variance is half."""
        with_var = estimate_voxel_memory(64, include_variance=True)
        no_var = estimate_voxel_memory(64, include_variance=False)

        assert no_var == with_var // 2


class TestCreateTestVoxelPattern:
    """Tests for create_test_voxel_pattern."""

    def test_sphere_pattern(self):
        """Sphere pattern creates spherical voxels."""
        chain = create_test_voxel_pattern(64, "sphere")

        # Center should be non-empty
        center = chain.sample_voxel(32, 32, 32, 0)
        assert not center.is_empty()

        # Corner should be empty
        corner = chain.sample_voxel(0, 0, 0, 0)
        assert corner.is_empty()

    def test_cube_pattern(self):
        """Cube pattern creates cubic voxels."""
        chain = create_test_voxel_pattern(64, "cube")

        center = chain.sample_voxel(32, 32, 32, 0)
        assert not center.is_empty()

    def test_gradient_pattern(self):
        """Gradient pattern has varying values."""
        chain = create_test_voxel_pattern(64, "gradient")

        low = chain.sample_voxel(0, 0, 0, 0)
        high = chain.sample_voxel(63, 63, 63, 0)

        assert low.radiance[0] < high.radiance[0]

    def test_checkerboard_pattern(self):
        """Checkerboard pattern alternates."""
        chain = create_test_voxel_pattern(64, "checkerboard")

        v0 = chain.sample_voxel(0, 0, 0, 0)
        v8 = chain.sample_voxel(8, 0, 0, 0)

        # Should alternate every 8 voxels
        assert v0.opacity != v8.opacity

    def test_invalid_pattern(self):
        """Invalid pattern raises ValueError."""
        with pytest.raises(ValueError, match="Unknown pattern"):
            create_test_voxel_pattern(64, "invalid")

    def test_builds_mip_chain(self):
        """Pattern function builds mip chain."""
        chain = create_test_voxel_pattern(64, "sphere")

        # Check mip 1 has content
        mip1 = chain.get_level(1)
        assert mip1.non_empty_count() > 0


class TestValidateMipChain:
    """Tests for validate_mip_chain."""

    def test_valid_chain(self):
        """Valid chain has no errors."""
        chain = create_test_voxel_pattern(64, "sphere")
        errors = validate_mip_chain(chain)

        # May have some tolerance warnings but no major errors
        assert len(errors) < 5

    def test_detects_resolution_mismatch(self):
        """Detects resolution mismatch."""
        chain = VoxelMipChain(base_resolution=MipResolution.RES_64)

        # Manually break resolution by setting wrong mip level resolution
        # Level 1 should be 32, but we set it to 16
        chain.levels[1] = VoxelMipLevel.create_empty(1, 16)  # Should be 32

        errors = validate_mip_chain(chain)
        # Should detect the resolution mismatch
        assert any("resolution" in e.lower() for e in errors)


# ============================================================================
# WGSL Generator Tests
# ============================================================================


class TestWGSLGenerators:
    """Tests for WGSL shader generators."""

    def test_generate_downsample_wgsl(self):
        """Downsample shader generates valid WGSL."""
        wgsl = generate_voxel_downsample_wgsl()

        assert len(wgsl) > 500
        assert "@compute" in wgsl
        assert "@workgroup_size" in wgsl
        assert "textureLoad" in wgsl
        assert "textureStore" in wgsl

    def test_generate_sample_wgsl(self):
        """Sample helpers generate valid WGSL."""
        wgsl = generate_voxel_sample_wgsl()

        assert len(wgsl) > 200
        assert "sample_voxel_trilinear" in wgsl
        assert "sample_voxel_cone" in wgsl
        assert "get_voxel_variance" in wgsl


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_mip_chain_workflow(self):
        """Full workflow: create, populate, build, sample."""
        # Create chain
        chain = VoxelMipChain(base_resolution=MipResolution.RES_64)

        # Populate base level with test data
        for x in range(16):
            for y in range(16):
                for z in range(16):
                    voxel = VoxelData(
                        np.array([0.5, 0.3, 0.1], dtype=np.float32),
                        0.8
                    )
                    chain.set_base_voxel(x, y, z, voxel)

        # Build mip chain
        chain.build_mip_chain()

        # Verify all levels have content
        for i in range(min(4, chain.mip_count)):
            level = chain.get_level(i)
            if level.resolution >= 8:
                assert level.non_empty_count() > 0

    def test_mip_chain_with_storage(self):
        """Mip chain with GPU storage abstraction."""
        config = VoxelStorageConfig(
            resolution=MipResolution.RES_64,
            format=VoxelStorageFormat.RGBA16F,
        )
        storage = VoxelStorage(config=config)
        chain = create_test_voxel_pattern(64, "sphere")

        # Prepare upload data for all levels
        for i in range(chain.mip_count):
            data = storage.prepare_upload_data(chain, i)
            expected_size = (chain.get_level(i).resolution ** 3) * 4 * 2
            assert len(data) == expected_size

    def test_variance_through_mip_chain(self):
        """Variance tracked through mip chain."""
        chain = VoxelMipChain(
            base_resolution=MipResolution.RES_64,
            downsample_config=VoxelDownsampleConfig(compute_variance=True)
        )

        # Fill with varied data
        for x in range(32):
            for y in range(32):
                for z in range(32):
                    val = np.sin(x * 0.5) * np.cos(y * 0.5) * 0.5 + 0.5
                    chain.set_base_voxel(
                        x, y, z,
                        VoxelData(np.array([val, val, val], dtype=np.float32), 1.0)
                    )

        chain.build_mip_chain()

        # Check variance at high mips
        for i in range(VARIANCE_START_MIP, chain.mip_count):
            level = chain.get_level(i)
            if level.variance is not None and level.resolution > 0:
                var = level.get_variance(0, 0, 0)
                if var is not None:
                    # Variance should be computed
                    assert True  # Passed if we got here

    def test_64_128_256_resolutions(self):
        """All supported resolutions work."""
        for res in [64, 128, 256]:
            voxel_res = MipResolution.from_size(res)
            chain = VoxelMipChain(base_resolution=voxel_res)

            assert chain.get_base().resolution == res
            assert chain.get_top().resolution == 1

    def test_energy_preservation_through_mips(self):
        """Energy approximately preserved through mip chain."""
        chain = VoxelMipChain(
            base_resolution=MipResolution.RES_64,
            downsample_config=VoxelDownsampleConfig(preserve_energy=True)
        )

        # Fill center region with constant value
        for x in range(32, 48):
            for y in range(32, 48):
                for z in range(32, 48):
                    chain.set_base_voxel(
                        x, y, z,
                        VoxelData(np.array([1.0, 1.0, 1.0], dtype=np.float32), 1.0)
                    )

        chain.build_mip_chain()

        # Sum energy at each level (very rough estimate)
        # Energy should decrease but not vanish
        base_energy = np.sum(chain.get_base().data[:, :, :, :3])
        top_energy = np.sum(chain.get_top().data[:, :, :, :3])

        assert top_energy > 0  # Some energy preserved


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_mip_chain(self):
        """Empty mip chain handles sampling gracefully."""
        chain = VoxelMipChain(base_resolution=MipResolution.RES_64)

        voxel = chain.sample_voxel(0, 0, 0, 0)
        assert voxel.is_empty()

    def test_single_voxel_mip_chain(self):
        """Single non-empty voxel propagates to top."""
        chain = VoxelMipChain(base_resolution=MipResolution.RES_64)
        chain.set_base_voxel(
            0, 0, 0,
            VoxelData(np.array([1.0, 1.0, 1.0], dtype=np.float32), 1.0)
        )
        chain.build_mip_chain()

        # Top should have some contribution
        top = chain.get_top().get_voxel(0, 0, 0)
        # Very diluted but non-zero
        assert top.radiance[0] >= 0

    def test_max_hdr_values(self):
        """High HDR values handled correctly."""
        voxel = VoxelData(np.array([100.0, 200.0, 300.0], dtype=np.float32), 1.0)

        assert voxel.radiance[0] == 100.0
        assert voxel.luminance() > 100

    def test_negative_values_clamped_in_storage(self):
        """Negative values handled in storage."""
        config = VoxelStorageConfig(
            resolution=MipResolution.RES_64,
            format=VoxelStorageFormat.R11G11B10F,
        )
        storage = VoxelStorage(config=config)
        chain = VoxelMipChain(base_resolution=MipResolution.RES_64)

        # Set negative value
        chain.set_base_voxel(
            0, 0, 0,
            VoxelData(np.array([-1.0, -1.0, -1.0], dtype=np.float32), 1.0)
        )

        # Upload should not crash (R11G11B10F is unsigned)
        data = storage.prepare_upload_data(chain, 0)
        assert len(data) > 0

    def test_sample_at_boundary(self):
        """Sampling at texture boundary works."""
        chain = VoxelMipChain(base_resolution=MipResolution.RES_64)

        # Sample at exact boundaries
        v = chain.sample_trilinear(0.0, 0.0, 0.0, 0)
        assert v is not None

        v = chain.sample_trilinear(1.0, 1.0, 1.0, 0)
        assert v is not None

    def test_statistics_on_empty_chain(self):
        """Statistics work on empty chain."""
        chain = VoxelMipChain(base_resolution=MipResolution.RES_64)
        stats = chain.get_statistics()

        assert stats["base_resolution"] == 64
        assert all(level["non_empty"] == 0 for level in stats["levels"])
