"""
Tests for heightfield terrain data (heightfield.py).

Tests cover:
- HeightfieldConfig validation
- Heightfield creation and data management
- Bilinear interpolation accuracy
- Normal calculation
- Compression/decompression
- Boundary conditions
- Height range clamping
- Numerical precision edge cases
"""

import pytest
import math

from engine.world.terrain.heightfield import (
    HeightfieldPrecision,
    HeightfieldConfig,
    Heightfield,
)
from engine.world.terrain.constants import (
    DEFAULT_RESOLUTION,
    DEFAULT_HEIGHT_RANGE,
    DEFAULT_SCALE,
    MIN_RESOLUTION,
    BITS_16_MAX_VALUE,
    HEIGHT_EPSILON,
)


# =============================================================================
# HeightfieldConfig Tests
# =============================================================================


class TestHeightfieldConfig:
    """Tests for HeightfieldConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HeightfieldConfig()
        assert config.resolution == 65
        assert config.precision == HeightfieldPrecision.BITS_16
        assert config.height_range == (-500.0, 500.0)
        assert config.scale == 1.0

    def test_custom_resolution(self):
        """Test custom resolution setting."""
        config = HeightfieldConfig(resolution=129)
        assert config.resolution == 129

    def test_custom_precision(self):
        """Test custom precision setting."""
        config = HeightfieldConfig(precision=HeightfieldPrecision.BITS_32)
        assert config.precision == HeightfieldPrecision.BITS_32

    def test_custom_height_range(self):
        """Test custom height range setting."""
        config = HeightfieldConfig(height_range=(-100.0, 1000.0))
        assert config.height_range == (-100.0, 1000.0)

    def test_custom_scale(self):
        """Test custom scale setting."""
        config = HeightfieldConfig(scale=2.5)
        assert config.scale == 2.5

    def test_invalid_resolution_below_minimum(self):
        """Test that resolution below 2 raises error."""
        with pytest.raises(ValueError, match="resolution must be >= 2"):
            HeightfieldConfig(resolution=1)

    def test_invalid_resolution_zero(self):
        """Test that zero resolution raises error."""
        with pytest.raises(ValueError, match="resolution must be >= 2"):
            HeightfieldConfig(resolution=0)

    def test_invalid_scale_zero(self):
        """Test that zero scale raises error."""
        with pytest.raises(ValueError, match="scale must be > 0"):
            HeightfieldConfig(scale=0.0)

    def test_invalid_scale_negative(self):
        """Test that negative scale raises error."""
        with pytest.raises(ValueError, match="scale must be > 0"):
            HeightfieldConfig(scale=-1.0)

    def test_invalid_height_range_equal(self):
        """Test that equal height range raises error."""
        with pytest.raises(ValueError, match="height_range"):
            HeightfieldConfig(height_range=(0.0, 0.0))

    def test_invalid_height_range_inverted(self):
        """Test that inverted height range raises error."""
        with pytest.raises(ValueError, match="height_range"):
            HeightfieldConfig(height_range=(100.0, -100.0))

    def test_minimum_valid_resolution(self):
        """Test minimum valid resolution of 2."""
        config = HeightfieldConfig(resolution=2)
        assert config.resolution == 2

    def test_large_resolution(self):
        """Test large resolution value."""
        config = HeightfieldConfig(resolution=4097)
        assert config.resolution == 4097


# =============================================================================
# Heightfield Creation Tests
# =============================================================================


class TestHeightfieldCreation:
    """Tests for Heightfield initialization."""

    def test_default_creation(self):
        """Test creation with default config."""
        hf = Heightfield()
        assert hf.config.resolution == 65

    def test_custom_config_creation(self):
        """Test creation with custom config."""
        config = HeightfieldConfig(resolution=33, scale=2.0)
        hf = Heightfield(config)
        assert hf.config.resolution == 33
        assert hf.config.scale == 2.0

    def test_initial_heights_are_zero(self):
        """Test that initial heights are all zero."""
        config = HeightfieldConfig(resolution=5)
        hf = Heightfield(config)
        for z in range(5):
            for x in range(5):
                assert hf.get_raw_height_at(x, z) == 0.0

    def test_data_dimensions(self):
        """Test that internal data has correct dimensions."""
        config = HeightfieldConfig(resolution=17)
        hf = Heightfield(config)
        assert len(hf._data) == 17
        for row in hf._data:
            assert len(row) == 17


# =============================================================================
# Height Access Tests
# =============================================================================


class TestHeightAccess:
    """Tests for height get/set operations."""

    def test_set_and_get_height(self):
        """Test basic set and get operations."""
        config = HeightfieldConfig(resolution=5)
        hf = Heightfield(config)
        assert hf.set_height_at(2, 2, 50.0)
        assert hf.get_raw_height_at(2, 2) == 50.0

    def test_set_out_of_bounds_x(self):
        """Test set at invalid X index."""
        config = HeightfieldConfig(resolution=5)
        hf = Heightfield(config)
        assert not hf.set_height_at(5, 0, 10.0)
        assert not hf.set_height_at(-1, 0, 10.0)

    def test_set_out_of_bounds_z(self):
        """Test set at invalid Z index."""
        config = HeightfieldConfig(resolution=5)
        hf = Heightfield(config)
        assert not hf.set_height_at(0, 5, 10.0)
        assert not hf.set_height_at(0, -1, 10.0)

    def test_get_out_of_bounds(self):
        """Test get at invalid index returns None."""
        config = HeightfieldConfig(resolution=5)
        hf = Heightfield(config)
        assert hf.get_raw_height_at(5, 0) is None
        assert hf.get_raw_height_at(0, 5) is None
        assert hf.get_raw_height_at(-1, 0) is None

    def test_height_clamping_max(self):
        """Test height is clamped to max range."""
        config = HeightfieldConfig(resolution=5, height_range=(-100.0, 100.0))
        hf = Heightfield(config)
        hf.set_height_at(0, 0, 500.0)
        assert hf.get_raw_height_at(0, 0) == 100.0

    def test_height_clamping_min(self):
        """Test height is clamped to min range."""
        config = HeightfieldConfig(resolution=5, height_range=(-100.0, 100.0))
        hf = Heightfield(config)
        hf.set_height_at(0, 0, -500.0)
        assert hf.get_raw_height_at(0, 0) == -100.0

    def test_height_within_range(self):
        """Test height within range is not clamped."""
        config = HeightfieldConfig(resolution=5, height_range=(-100.0, 100.0))
        hf = Heightfield(config)
        hf.set_height_at(0, 0, 50.0)
        assert hf.get_raw_height_at(0, 0) == 50.0

    def test_boundary_indices(self):
        """Test accessing boundary sample indices."""
        config = HeightfieldConfig(resolution=5)
        hf = Heightfield(config)
        # Corners
        assert hf.set_height_at(0, 0, 1.0)
        assert hf.set_height_at(4, 0, 2.0)
        assert hf.set_height_at(0, 4, 3.0)
        assert hf.set_height_at(4, 4, 4.0)
        assert hf.get_raw_height_at(0, 0) == 1.0
        assert hf.get_raw_height_at(4, 0) == 2.0
        assert hf.get_raw_height_at(0, 4) == 3.0
        assert hf.get_raw_height_at(4, 4) == 4.0


# =============================================================================
# Bilinear Interpolation Tests
# =============================================================================


class TestBilinearInterpolation:
    """Tests for bilinear interpolation in get_height_at."""

    def test_exact_sample_position(self):
        """Test that exact sample positions return exact values."""
        config = HeightfieldConfig(resolution=5, scale=1.0)
        hf = Heightfield(config)
        hf.set_height_at(2, 2, 100.0)
        # At exact sample position (scale=1.0, so sample 2,2 is at world 2.0,2.0)
        assert abs(hf.get_height_at(2.0, 2.0) - 100.0) < 1e-6

    def test_interpolation_midpoint_x(self):
        """Test interpolation at X midpoint."""
        config = HeightfieldConfig(resolution=3, scale=1.0)
        hf = Heightfield(config)
        hf.set_height_at(0, 0, 0.0)
        hf.set_height_at(1, 0, 100.0)
        # Midpoint between 0 and 100 should be 50
        height = hf.get_height_at(0.5, 0.0)
        assert abs(height - 50.0) < 1e-6

    def test_interpolation_midpoint_z(self):
        """Test interpolation at Z midpoint."""
        config = HeightfieldConfig(resolution=3, scale=1.0)
        hf = Heightfield(config)
        hf.set_height_at(0, 0, 0.0)
        hf.set_height_at(0, 1, 100.0)
        # Midpoint between 0 and 100 should be 50
        height = hf.get_height_at(0.0, 0.5)
        assert abs(height - 50.0) < 1e-6

    def test_interpolation_center_of_quad(self):
        """Test bilinear interpolation at center of a quad."""
        config = HeightfieldConfig(resolution=3, scale=1.0)
        hf = Heightfield(config)
        # Set corners of one cell
        hf.set_height_at(0, 0, 0.0)
        hf.set_height_at(1, 0, 100.0)
        hf.set_height_at(0, 1, 100.0)
        hf.set_height_at(1, 1, 200.0)
        # Center should be average = (0+100+100+200)/4 = 100
        height = hf.get_height_at(0.5, 0.5)
        assert abs(height - 100.0) < 1e-6

    def test_interpolation_asymmetric_quad(self):
        """Test bilinear interpolation with asymmetric corner values."""
        config = HeightfieldConfig(resolution=3, scale=1.0)
        hf = Heightfield(config)
        hf.set_height_at(0, 0, 0.0)
        hf.set_height_at(1, 0, 10.0)
        hf.set_height_at(0, 1, 20.0)
        hf.set_height_at(1, 1, 30.0)
        # At (0.25, 0.75):
        # h0 = 0.0 * 0.75 + 10.0 * 0.25 = 2.5
        # h1 = 20.0 * 0.75 + 30.0 * 0.25 = 22.5
        # result = 2.5 * 0.25 + 22.5 * 0.75 = 0.625 + 16.875 = 17.5
        height = hf.get_height_at(0.25, 0.75)
        assert abs(height - 17.5) < 1e-6

    def test_interpolation_with_scale(self):
        """Test interpolation with non-unit scale."""
        config = HeightfieldConfig(resolution=3, scale=2.0)
        hf = Heightfield(config)
        hf.set_height_at(0, 0, 0.0)
        hf.set_height_at(1, 0, 100.0)
        # With scale=2.0, sample 1 is at world position 2.0
        # World position 1.0 is at sample 0.5
        height = hf.get_height_at(1.0, 0.0)
        assert abs(height - 50.0) < 1e-6

    def test_interpolation_clamped_negative_x(self):
        """Test that negative X positions are clamped."""
        config = HeightfieldConfig(resolution=3, scale=1.0)
        hf = Heightfield(config)
        hf.set_height_at(0, 0, 100.0)
        # Negative position should clamp to edge
        height = hf.get_height_at(-1.0, 0.0)
        assert abs(height - 100.0) < 1e-6

    def test_interpolation_clamped_beyond_max(self):
        """Test that positions beyond max are clamped."""
        config = HeightfieldConfig(resolution=3, scale=1.0)
        hf = Heightfield(config)
        hf.set_height_at(2, 2, 100.0)
        # Position beyond max (2.0) should clamp to edge
        height = hf.get_height_at(10.0, 10.0)
        assert abs(height - 100.0) < 1e-6

    def test_interpolation_quarter_points(self):
        """Test interpolation at quarter points."""
        config = HeightfieldConfig(resolution=3, scale=1.0)
        hf = Heightfield(config)
        hf.set_height_at(0, 0, 0.0)
        hf.set_height_at(1, 0, 100.0)
        # 0.25 of the way should give 25
        height = hf.get_height_at(0.25, 0.0)
        assert abs(height - 25.0) < 1e-6
        # 0.75 of the way should give 75
        height = hf.get_height_at(0.75, 0.0)
        assert abs(height - 75.0) < 1e-6


# =============================================================================
# Normal Calculation Tests
# =============================================================================


class TestNormalCalculation:
    """Tests for surface normal calculation."""

    def test_flat_surface_normal(self):
        """Test normal on flat surface points up."""
        config = HeightfieldConfig(resolution=5, scale=1.0)
        hf = Heightfield(config)
        # All heights at 0 = flat surface
        normal = hf.get_normal_at(2.0, 2.0)
        assert abs(normal[0]) < 1e-6
        assert abs(normal[1] - 1.0) < 1e-6
        assert abs(normal[2]) < 1e-6

    def test_normal_is_normalized(self):
        """Test that normal is unit length."""
        config = HeightfieldConfig(resolution=5, scale=1.0)
        hf = Heightfield(config)
        # Create a slope
        for x in range(5):
            for z in range(5):
                hf.set_height_at(x, z, x * 10.0)
        normal = hf.get_normal_at(2.0, 2.0)
        length = math.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
        assert abs(length - 1.0) < 1e-6

    def test_normal_points_upward(self):
        """Test that Y component of normal is always positive."""
        config = HeightfieldConfig(resolution=5, scale=1.0)
        hf = Heightfield(config)
        # Create steep slope
        for x in range(5):
            for z in range(5):
                hf.set_height_at(x, z, x * 100.0 + z * 50.0)
        normal = hf.get_normal_at(2.0, 2.0)
        assert normal[1] > 0

    def test_normal_x_slope(self):
        """Test normal on X-facing slope."""
        config = HeightfieldConfig(resolution=5, scale=1.0)
        hf = Heightfield(config)
        # Slope in X direction: height increases with X
        for x in range(5):
            for z in range(5):
                hf.set_height_at(x, z, x * 10.0)
        normal = hf.get_normal_at(2.0, 2.0)
        # Normal should point somewhat in -X direction
        assert normal[0] < 0

    def test_normal_z_slope(self):
        """Test normal on Z-facing slope."""
        config = HeightfieldConfig(resolution=5, scale=1.0)
        hf = Heightfield(config)
        # Slope in Z direction: height increases with Z
        for x in range(5):
            for z in range(5):
                hf.set_height_at(x, z, z * 10.0)
        normal = hf.get_normal_at(2.0, 2.0)
        # Normal should point somewhat in -Z direction
        assert normal[2] < 0

    def test_normal_at_edge(self):
        """Test normal calculation at edge handles boundary correctly."""
        config = HeightfieldConfig(resolution=5, scale=1.0)
        hf = Heightfield(config)
        # Should not crash at edges
        normal = hf.get_normal_at(0.0, 0.0)
        assert len(normal) == 3
        normal = hf.get_normal_at(4.0, 4.0)
        assert len(normal) == 3


# =============================================================================
# Sample Region Tests
# =============================================================================


class TestSampleRegion:
    """Tests for sample_region method."""

    def test_sample_full_region(self):
        """Test sampling entire heightfield."""
        config = HeightfieldConfig(resolution=5)
        hf = Heightfield(config)
        for z in range(5):
            for x in range(5):
                hf.set_height_at(x, z, x + z * 10.0)
        region = hf.sample_region(0, 0, 4, 4)
        assert len(region) == 5
        assert len(region[0]) == 5
        assert region[0][0] == 0.0
        assert region[0][4] == 4.0
        assert region[4][0] == 40.0
        assert region[4][4] == 44.0

    def test_sample_sub_region(self):
        """Test sampling a sub-region."""
        config = HeightfieldConfig(resolution=5)
        hf = Heightfield(config)
        for z in range(5):
            for x in range(5):
                hf.set_height_at(x, z, x + z * 10.0)
        region = hf.sample_region(1, 1, 3, 3)
        assert len(region) == 3
        assert len(region[0]) == 3
        assert region[0][0] == 11.0  # (1, 1)
        assert region[2][2] == 33.0  # (3, 3)

    def test_sample_single_cell(self):
        """Test sampling a single cell."""
        config = HeightfieldConfig(resolution=5)
        hf = Heightfield(config)
        hf.set_height_at(2, 2, 100.0)
        region = hf.sample_region(2, 2, 2, 2)
        assert len(region) == 1
        assert len(region[0]) == 1
        assert region[0][0] == 100.0

    def test_sample_clamped_bounds(self):
        """Test sampling with out-of-bounds coordinates gets clamped."""
        config = HeightfieldConfig(resolution=5)
        hf = Heightfield(config)
        region = hf.sample_region(-1, -1, 10, 10)
        assert len(region) == 5
        assert len(region[0]) == 5

    def test_sample_invalid_region(self):
        """Test sampling with inverted bounds returns empty."""
        config = HeightfieldConfig(resolution=5)
        hf = Heightfield(config)
        region = hf.sample_region(3, 3, 1, 1)
        assert region == []


# =============================================================================
# Import/Export Tests
# =============================================================================


class TestImportExport:
    """Tests for import_from_data and export_to_data."""

    def test_export_empty_heightfield(self):
        """Test exporting empty heightfield."""
        config = HeightfieldConfig(resolution=5)
        hf = Heightfield(config)
        data = hf.export_to_data()
        assert len(data) == 5
        for row in data:
            assert len(row) == 5
            for h in row:
                assert h == 0.0

    def test_export_with_values(self):
        """Test exporting heightfield with values."""
        config = HeightfieldConfig(resolution=3)
        hf = Heightfield(config)
        hf.set_height_at(0, 0, 10.0)
        hf.set_height_at(2, 2, 50.0)
        data = hf.export_to_data()
        assert data[0][0] == 10.0
        assert data[2][2] == 50.0

    def test_import_valid_data(self):
        """Test importing valid data."""
        config = HeightfieldConfig(resolution=3)
        hf = Heightfield(config)
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        assert hf.import_from_data(data)
        assert hf.get_raw_height_at(0, 0) == 1.0
        assert hf.get_raw_height_at(2, 2) == 9.0

    def test_import_wrong_row_count(self):
        """Test importing data with wrong row count fails."""
        config = HeightfieldConfig(resolution=3)
        hf = Heightfield(config)
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]  # Only 2 rows
        assert not hf.import_from_data(data)

    def test_import_wrong_column_count(self):
        """Test importing data with wrong column count fails."""
        config = HeightfieldConfig(resolution=3)
        hf = Heightfield(config)
        data = [[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]]  # Only 2 columns
        assert not hf.import_from_data(data)

    def test_import_clamps_heights(self):
        """Test that import clamps heights to range."""
        config = HeightfieldConfig(resolution=2, height_range=(-10.0, 10.0))
        hf = Heightfield(config)
        data = [[100.0, -100.0], [0.0, 5.0]]
        assert hf.import_from_data(data)
        assert hf.get_raw_height_at(0, 0) == 10.0
        assert hf.get_raw_height_at(1, 0) == -10.0
        assert hf.get_raw_height_at(0, 1) == 0.0
        assert hf.get_raw_height_at(1, 1) == 5.0

    def test_roundtrip_export_import(self):
        """Test export then import preserves data."""
        config = HeightfieldConfig(resolution=5)
        hf1 = Heightfield(config)
        for z in range(5):
            for x in range(5):
                hf1.set_height_at(x, z, x * z * 1.5)
        data = hf1.export_to_data()
        hf2 = Heightfield(config)
        assert hf2.import_from_data(data)
        for z in range(5):
            for x in range(5):
                assert abs(hf1.get_raw_height_at(x, z) - hf2.get_raw_height_at(x, z)) < 1e-6


# =============================================================================
# Compression/Decompression Tests
# =============================================================================


class TestCompression:
    """Tests for compress and decompress methods."""

    def test_compress_returns_bytes(self):
        """Test that compress returns bytes."""
        config = HeightfieldConfig(resolution=5)
        hf = Heightfield(config)
        data = hf.compress()
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_decompress_returns_heightfield(self):
        """Test that decompress returns Heightfield."""
        config = HeightfieldConfig(resolution=5)
        hf = Heightfield(config)
        data = hf.compress()
        hf2 = Heightfield.decompress(data)
        assert isinstance(hf2, Heightfield)

    def test_roundtrip_compression_16bit(self):
        """Test compress/decompress roundtrip with 16-bit precision."""
        config = HeightfieldConfig(
            resolution=5,
            precision=HeightfieldPrecision.BITS_16,
            height_range=(-100.0, 100.0)
        )
        hf = Heightfield(config)
        for z in range(5):
            for x in range(5):
                hf.set_height_at(x, z, (x - 2) * 20.0 + (z - 2) * 10.0)

        data = hf.compress()
        hf2 = Heightfield.decompress(data)

        # 16-bit has some quantization error
        for z in range(5):
            for x in range(5):
                h1 = hf.get_raw_height_at(x, z)
                h2 = hf2.get_raw_height_at(x, z)
                assert abs(h1 - h2) < 0.01  # Within quantization tolerance

    def test_roundtrip_compression_32bit(self):
        """Test compress/decompress roundtrip with 32-bit precision."""
        config = HeightfieldConfig(
            resolution=5,
            precision=HeightfieldPrecision.BITS_32,
            height_range=(-100.0, 100.0)
        )
        hf = Heightfield(config)
        for z in range(5):
            for x in range(5):
                hf.set_height_at(x, z, (x - 2) * 20.123 + (z - 2) * 10.456)

        data = hf.compress()
        hf2 = Heightfield.decompress(data)

        # 32-bit should be very accurate
        for z in range(5):
            for x in range(5):
                h1 = hf.get_raw_height_at(x, z)
                h2 = hf2.get_raw_height_at(x, z)
                assert abs(h1 - h2) < 1e-5

    def test_compressed_size_is_smaller(self):
        """Test that compressed data is smaller than uncompressed."""
        config = HeightfieldConfig(resolution=65)
        hf = Heightfield(config)
        # Fill with smooth data that compresses well
        for z in range(65):
            for x in range(65):
                hf.set_height_at(x, z, math.sin(x * 0.1) * math.cos(z * 0.1) * 50.0)

        compressed = hf.compress()
        # Uncompressed would be 65*65*2 = 8450 bytes for 16-bit
        assert len(compressed) < 65 * 65 * 2

    def test_decompress_preserves_config(self):
        """Test that decompression preserves configuration."""
        config = HeightfieldConfig(
            resolution=33,
            precision=HeightfieldPrecision.BITS_32,
            height_range=(-200.0, 500.0),
            scale=2.5
        )
        hf = Heightfield(config)
        data = hf.compress()
        hf2 = Heightfield.decompress(data)

        assert hf2.config.resolution == 33
        assert hf2.config.precision == HeightfieldPrecision.BITS_32
        assert hf2.config.height_range == (-200.0, 500.0)
        assert hf2.config.scale == 2.5

    def test_decompress_invalid_data_too_short(self):
        """Test decompress with too short data raises error."""
        with pytest.raises(ValueError, match="too short"):
            Heightfield.decompress(b"short")

    def test_decompress_corrupted_data(self):
        """Test decompress with corrupted compressed data raises error."""
        config = HeightfieldConfig(resolution=5)
        hf = Heightfield(config)
        data = hf.compress()
        # Corrupt the compressed portion
        corrupted = data[:29] + b"corrupted data here" * 10
        with pytest.raises(ValueError, match="decompress"):
            Heightfield.decompress(corrupted)


# =============================================================================
# Bounds Tests
# =============================================================================


class TestBounds:
    """Tests for get_bounds method."""

    def test_bounds_empty_heightfield(self):
        """Test bounds of empty heightfield."""
        config = HeightfieldConfig(resolution=5)
        hf = Heightfield(config)
        min_h, max_h = hf.get_bounds()
        assert min_h == 0.0
        assert max_h == 0.0

    def test_bounds_with_values(self):
        """Test bounds with various heights."""
        config = HeightfieldConfig(resolution=5)
        hf = Heightfield(config)
        hf.set_height_at(0, 0, -50.0)
        hf.set_height_at(2, 2, 100.0)
        hf.set_height_at(4, 4, 25.0)
        min_h, max_h = hf.get_bounds()
        assert min_h == -50.0
        assert max_h == 100.0

    def test_bounds_are_cached(self):
        """Test that bounds are cached until data changes."""
        config = HeightfieldConfig(resolution=5)
        hf = Heightfield(config)
        hf.set_height_at(0, 0, 50.0)
        bounds1 = hf.get_bounds()
        bounds2 = hf.get_bounds()
        assert bounds1 == bounds2
        # Change data
        hf.set_height_at(1, 1, 100.0)
        bounds3 = hf.get_bounds()
        assert bounds3[1] == 100.0


# =============================================================================
# Utility Method Tests
# =============================================================================


class TestUtilityMethods:
    """Tests for utility methods."""

    def test_get_world_size(self):
        """Test get_world_size calculation."""
        config = HeightfieldConfig(resolution=65, scale=1.0)
        hf = Heightfield(config)
        width, depth = hf.get_world_size()
        assert width == 64.0
        assert depth == 64.0

    def test_get_world_size_with_scale(self):
        """Test get_world_size with non-unit scale."""
        config = HeightfieldConfig(resolution=33, scale=2.0)
        hf = Heightfield(config)
        width, depth = hf.get_world_size()
        assert width == 64.0  # 32 edges * 2.0 scale
        assert depth == 64.0

    def test_fill(self):
        """Test fill method."""
        config = HeightfieldConfig(resolution=5)
        hf = Heightfield(config)
        hf.fill(50.0)
        for z in range(5):
            for x in range(5):
                assert hf.get_raw_height_at(x, z) == 50.0

    def test_fill_clamps_height(self):
        """Test that fill clamps to height range."""
        config = HeightfieldConfig(resolution=5, height_range=(-10.0, 10.0))
        hf = Heightfield(config)
        hf.fill(100.0)
        assert hf.get_raw_height_at(0, 0) == 10.0

    def test_copy(self):
        """Test copy creates independent copy."""
        config = HeightfieldConfig(resolution=5)
        hf1 = Heightfield(config)
        hf1.set_height_at(2, 2, 100.0)
        hf2 = hf1.copy()
        assert hf2.get_raw_height_at(2, 2) == 100.0
        # Modify copy shouldn't affect original
        hf2.set_height_at(2, 2, 50.0)
        assert hf1.get_raw_height_at(2, 2) == 100.0
        assert hf2.get_raw_height_at(2, 2) == 50.0

    def test_equality(self):
        """Test heightfield equality comparison."""
        config = HeightfieldConfig(resolution=5)
        hf1 = Heightfield(config)
        hf2 = Heightfield(config)
        hf1.set_height_at(0, 0, 10.0)
        hf2.set_height_at(0, 0, 10.0)
        assert hf1 == hf2

    def test_inequality_different_heights(self):
        """Test heightfield inequality with different heights."""
        config = HeightfieldConfig(resolution=5)
        hf1 = Heightfield(config)
        hf2 = Heightfield(config)
        hf1.set_height_at(0, 0, 10.0)
        hf2.set_height_at(0, 0, 20.0)
        assert hf1 != hf2

    def test_inequality_different_config(self):
        """Test heightfield inequality with different config."""
        hf1 = Heightfield(HeightfieldConfig(resolution=5))
        hf2 = Heightfield(HeightfieldConfig(resolution=9))
        assert hf1 != hf2

    def test_inequality_different_type(self):
        """Test heightfield inequality with different type."""
        hf = Heightfield()
        assert hf != "not a heightfield"
        assert hf != 42
        assert hf != None


# =============================================================================
# Numerical Precision Tests
# =============================================================================


class TestNumericalPrecision:
    """Tests for numerical precision and edge cases."""

    def test_interpolation_exact_boundary_values(self):
        """Test interpolation at exact sample boundaries returns exact values."""
        config = HeightfieldConfig(resolution=5, scale=1.0)
        hf = Heightfield(config)

        # Set specific heights at all corners
        expected_heights = [
            (0, 0, 10.0), (1, 0, 20.0), (2, 0, 30.0), (3, 0, 40.0), (4, 0, 50.0),
            (0, 1, 15.0), (1, 1, 25.0), (2, 1, 35.0), (3, 1, 45.0), (4, 1, 55.0),
        ]
        for x, z, h in expected_heights:
            hf.set_height_at(x, z, h)

        # Verify exact positions return exact values (within epsilon)
        for x, z, expected in expected_heights:
            actual = hf.get_height_at(float(x), float(z))
            assert abs(actual - expected) < HEIGHT_EPSILON, \
                f"At ({x}, {z}): expected {expected}, got {actual}"

    def test_interpolation_symmetry(self):
        """Test bilinear interpolation is symmetric."""
        config = HeightfieldConfig(resolution=3, scale=1.0)
        hf = Heightfield(config)

        # Set symmetric values
        hf.set_height_at(0, 0, 0.0)
        hf.set_height_at(1, 0, 100.0)
        hf.set_height_at(0, 1, 100.0)
        hf.set_height_at(1, 1, 0.0)

        # Center should be average: (0+100+100+0)/4 = 50
        center = hf.get_height_at(0.5, 0.5)
        assert abs(center - 50.0) < HEIGHT_EPSILON

        # Test symmetry: (0.25, 0.75) should equal (0.75, 0.25)
        h1 = hf.get_height_at(0.25, 0.75)
        h2 = hf.get_height_at(0.75, 0.25)
        assert abs(h1 - h2) < HEIGHT_EPSILON

    def test_very_small_scale(self):
        """Test heightfield with very small scale doesn't cause numerical issues."""
        config = HeightfieldConfig(resolution=5, scale=0.001)
        hf = Heightfield(config)
        hf.set_height_at(2, 2, 100.0)

        # Query at sample position (scale 0.001 means position 0.002 is sample 2)
        height = hf.get_height_at(0.002, 0.002)
        assert abs(height - 100.0) < HEIGHT_EPSILON

        # Normal should still be computable
        normal = hf.get_normal_at(0.002, 0.002)
        assert normal is not None
        assert len(normal) == 3

        # Normal length should be 1 (unit normal)
        length = math.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
        assert abs(length - 1.0) < HEIGHT_EPSILON

    def test_very_large_scale(self):
        """Test heightfield with very large scale works correctly."""
        config = HeightfieldConfig(resolution=5, scale=1000.0)
        hf = Heightfield(config)
        hf.set_height_at(2, 2, 100.0)

        # Sample 2 is at world position 2000
        height = hf.get_height_at(2000.0, 2000.0)
        assert abs(height - 100.0) < HEIGHT_EPSILON

    def test_16bit_quantization_precision(self):
        """Test 16-bit quantization precision across height range."""
        config = HeightfieldConfig(
            resolution=3,
            precision=HeightfieldPrecision.BITS_16,
            height_range=(-1000.0, 1000.0)
        )
        hf = Heightfield(config)

        # Test range is 2000, quantization step is 2000/65535 ~ 0.0305
        test_values = [-1000.0, -500.0, -0.1, 0.0, 0.1, 500.0, 1000.0]

        for i, h in enumerate(test_values):
            hf.set_height_at(i % 3, i // 3, h)

        # Compress and decompress
        data = hf.compress()
        hf2 = Heightfield.decompress(data)

        # Max error should be about 0.0305 (half a quantization step)
        max_expected_error = 2000.0 / BITS_16_MAX_VALUE

        for i, expected in enumerate(test_values):
            actual = hf2.get_raw_height_at(i % 3, i // 3)
            error = abs(actual - expected)
            assert error < max_expected_error, \
                f"Quantization error {error} exceeds expected {max_expected_error} for {expected}"

    def test_zero_height_range_compression(self):
        """Test compression handles zero height range (all same height)."""
        # This should raise ValueError due to invalid height_range
        with pytest.raises(ValueError, match="height_range"):
            HeightfieldConfig(
                resolution=5,
                precision=HeightfieldPrecision.BITS_16,
                height_range=(0.0, 0.0)  # Invalid range
            )

    def test_flat_terrain_compression_accuracy(self):
        """Test compression of flat terrain preserves exact values."""
        config = HeightfieldConfig(
            resolution=5,
            precision=HeightfieldPrecision.BITS_16,
            height_range=(-100.0, 100.0)
        )
        hf = Heightfield(config)
        hf.fill(0.0)  # All zeros

        data = hf.compress()
        hf2 = Heightfield.decompress(data)

        for z in range(5):
            for x in range(5):
                h = hf2.get_raw_height_at(x, z)
                # 0.0 should be at the middle of the range, perfectly representable
                # Actually 0 is at normalized = 0.5, which is 32767 or 32768 in 16-bit
                assert abs(h) < 0.01  # Within tolerance

    def test_extreme_height_values(self):
        """Test handling of extreme height values."""
        config = HeightfieldConfig(
            resolution=3,
            height_range=(-1e6, 1e6),
            scale=1.0
        )
        hf = Heightfield(config)

        # Set extreme values
        hf.set_height_at(0, 0, -1e6)
        hf.set_height_at(2, 0, 1e6)

        # Verify clamping
        assert hf.get_raw_height_at(0, 0) == -1e6
        assert hf.get_raw_height_at(2, 0) == 1e6

        # Try to set beyond range
        hf.set_height_at(1, 1, 2e6)
        assert hf.get_raw_height_at(1, 1) == 1e6  # Clamped

    def test_normal_on_steep_slope(self):
        """Test normal calculation on very steep slopes."""
        config = HeightfieldConfig(resolution=5, scale=1.0)
        hf = Heightfield(config)

        # Create a very steep slope: 100 units rise over 1 unit
        for x in range(5):
            for z in range(5):
                hf.set_height_at(x, z, x * 100.0)

        normal = hf.get_normal_at(2.0, 2.0)

        # Normal should be unit length
        length = math.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
        assert abs(length - 1.0) < HEIGHT_EPSILON

        # Y component should still be positive (pointing up)
        assert normal[1] > 0

        # X component should be negative (slope goes up in +X direction)
        assert normal[0] < 0

    def test_interpolation_continuity(self):
        """Test that interpolation is continuous across cell boundaries."""
        config = HeightfieldConfig(resolution=5, scale=1.0)
        hf = Heightfield(config)

        # Create smooth terrain
        for x in range(5):
            for z in range(5):
                hf.set_height_at(x, z, math.sin(x * 0.5) * math.cos(z * 0.5) * 10.0)

        # Sample along a line crossing cell boundaries
        prev_height = hf.get_height_at(0.0, 2.0)

        for i in range(1, 40):
            x = i * 0.1
            height = hf.get_height_at(x, 2.0)

            # Height change should be continuous (no jumps)
            delta = abs(height - prev_height)
            assert delta < 2.0, f"Discontinuity at x={x}: delta={delta}"
            prev_height = height

    def test_constants_match_defaults(self):
        """Test that constants module values match default config."""
        config = HeightfieldConfig()

        assert config.resolution == DEFAULT_RESOLUTION
        assert config.height_range == DEFAULT_HEIGHT_RANGE
        assert config.scale == DEFAULT_SCALE
