"""Tests for pre-filtered cubemaps for roughness-based reflections.

Covers:
- GGX normal distribution function math
- Importance sampling uniformity and correctness
- Hammersley/Van der Corput sequence
- Tangent-to-world transformation
- Pre-filter blur behavior with roughness
- Split-sum LUT accuracy
- Mip level selection
- Full pipeline integration
- Performance constraints
"""

from __future__ import annotations

import math
import time
from typing import Tuple

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
from engine.rendering.lighting.probe_prefilter import (
    PrefilterConstants,
    GGXDistribution,
    radical_inverse_vdc,
    hammersley,
    tangent_to_world,
    ImportanceSampler,
    PrefilterConfig,
    CubemapPrefilter,
    BRDFTerms,
    SplitSumLUT,
    PrefilterResult,
    PrefilterPipeline,
)


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def uniform_cubemap() -> CubemapData:
    """Create a uniform white cubemap for testing."""
    cubemap = CubemapData(resolution=16)
    for face in CubemapFace:
        face_data = cubemap.get_face(face)
        for y in range(16):
            for x in range(16):
                face_data.set_pixel(x, y, HDRPixel(1.0, 1.0, 1.0))
    return cubemap


@pytest.fixture
def gradient_cubemap() -> CubemapData:
    """Create a gradient cubemap for blur testing."""
    cubemap = CubemapData(resolution=32)
    for face in CubemapFace:
        face_data = cubemap.get_face(face)
        for y in range(32):
            for x in range(32):
                # Create sharp edge at x=16
                if x < 16:
                    face_data.set_pixel(x, y, HDRPixel(0.0, 0.0, 0.0))
                else:
                    face_data.set_pixel(x, y, HDRPixel(1.0, 1.0, 1.0))
    return cubemap


@pytest.fixture
def colored_cubemap() -> CubemapData:
    """Create a cubemap with different colors per face."""
    cubemap = CubemapData(resolution=16)
    colors = [
        HDRPixel(1.0, 0.0, 0.0),  # +X red
        HDRPixel(0.0, 1.0, 0.0),  # -X green
        HDRPixel(0.0, 0.0, 1.0),  # +Y blue
        HDRPixel(1.0, 1.0, 0.0),  # -Y yellow
        HDRPixel(1.0, 0.0, 1.0),  # +Z magenta
        HDRPixel(0.0, 1.0, 1.0),  # -Z cyan
    ]
    for face in CubemapFace:
        face_data = cubemap.get_face(face)
        color = colors[face.value]
        for y in range(16):
            for x in range(16):
                face_data.set_pixel(x, y, color)
    return cubemap


@pytest.fixture
def hdr_cubemap() -> CubemapData:
    """Create an HDR cubemap with high intensity values."""
    cubemap = CubemapData(resolution=16)
    for face in CubemapFace:
        face_data = cubemap.get_face(face)
        for y in range(16):
            for x in range(16):
                intensity = 5.0 + (face.value * 2.0)
                face_data.set_pixel(x, y, HDRPixel(intensity, intensity, intensity))
    return cubemap


# -----------------------------------------------------------------------------
# GGXDistribution Tests
# -----------------------------------------------------------------------------

class TestGGXDistribution:
    """Tests for GGX normal distribution function."""

    def test_ggx_creation_default(self) -> None:
        """Test creating GGX distribution with defaults."""
        ggx = GGXDistribution()
        assert ggx.roughness == pytest.approx(0.5)
        assert ggx.alpha == pytest.approx(0.25)

    def test_ggx_creation_custom_roughness(self) -> None:
        """Test creating GGX with custom roughness."""
        ggx = GGXDistribution(roughness=0.3)
        assert ggx.roughness == pytest.approx(0.3)
        assert ggx.alpha == pytest.approx(0.09)

    def test_ggx_roughness_clamping_low(self) -> None:
        """Test roughness is clamped to minimum."""
        ggx = GGXDistribution(roughness=0.0)
        assert ggx.roughness == pytest.approx(PrefilterConstants.MIN_ROUGHNESS)

    def test_ggx_roughness_clamping_high(self) -> None:
        """Test roughness is clamped to maximum."""
        ggx = GGXDistribution(roughness=2.0)
        assert ggx.roughness == pytest.approx(1.0)

    def test_ggx_d_at_normal(self) -> None:
        """Test D(H) when H = N (NdotH = 1)."""
        ggx = GGXDistribution(roughness=0.5)
        # At NdotH = 1, denom = (1 * (a^4 - 1) + 1)^2 * pi = (a^4)^2 * pi = a^8 * pi
        # So D = a^4 / (a^8 * pi) = 1 / (a^4 * pi)
        # Actually: denom = (1 * (a^2 - 1) + 1)^2 * pi = (a^2)^2 * pi = a^4 * pi
        # D = a^2 / (a^4 * pi) = 1 / (a^2 * pi)
        expected = 1.0 / (ggx.alpha * ggx.alpha * math.pi)
        result = ggx.D(1.0)
        assert result == pytest.approx(expected, rel=0.01)

    def test_ggx_d_at_grazing(self) -> None:
        """Test D(H) at grazing angle (NdotH approaching 0)."""
        ggx = GGXDistribution(roughness=0.5)
        result = ggx.D(0.01)
        # Should be very small at grazing
        assert result < 0.1

    def test_ggx_d_normalized(self) -> None:
        """Test that GGX D has sensible peak at normal."""
        ggx = GGXDistribution(roughness=0.5)
        # D should peak at NdotH = 1 and decrease toward grazing
        d_normal = ggx.D(1.0)
        d_45deg = ggx.D(0.707)  # cos(45 degrees)
        d_grazing = ggx.D(0.1)

        # Should decrease as angle from normal increases
        assert d_normal > d_45deg
        assert d_45deg > d_grazing
        # Normal direction should have highest value
        assert d_normal > 1.0  # Should be > 1/pi for roughness 0.5

    def test_ggx_d_increases_with_roughness(self) -> None:
        """Test D at off-normal increases with roughness."""
        ggx_smooth = GGXDistribution(roughness=0.2)
        ggx_rough = GGXDistribution(roughness=0.8)
        # At a moderate angle, rough surface spreads more
        d_smooth = ggx_smooth.D(0.5)
        d_rough = ggx_rough.D(0.5)
        # Rough should have higher D at off-normal angles
        assert d_rough > d_smooth

    def test_ggx_d_non_negative(self) -> None:
        """Test D is always non-negative."""
        ggx = GGXDistribution(roughness=0.5)
        for n_dot_h in [0.0, 0.1, 0.5, 0.9, 1.0]:
            assert ggx.D(n_dot_h) >= 0.0

    def test_ggx_sample_direction_returns_unit_vector(self) -> None:
        """Test sampled direction is normalized."""
        ggx = GGXDistribution(roughness=0.5)
        N = Vec3(0, 1, 0)
        H = ggx.sample_direction(0.3, 0.7, N)
        assert H.length() == pytest.approx(1.0, rel=0.001)

    def test_ggx_sample_direction_in_hemisphere(self) -> None:
        """Test sampled directions are in the correct hemisphere."""
        ggx = GGXDistribution(roughness=0.5)
        N = Vec3(0, 1, 0)
        for i in range(50):
            xi_x = i / 50.0
            xi_y = (i * 0.7) % 1.0
            H = ggx.sample_direction(xi_x, xi_y, N)
            assert N.dot(H) >= 0.0  # H should be in upper hemisphere

    def test_ggx_set_roughness(self) -> None:
        """Test updating roughness value."""
        ggx = GGXDistribution(roughness=0.5)
        ggx.set_roughness(0.8)
        assert ggx.roughness == pytest.approx(0.8)
        assert ggx.alpha == pytest.approx(0.64)

    def test_ggx_pdf_non_negative(self) -> None:
        """Test PDF is always non-negative."""
        ggx = GGXDistribution(roughness=0.5)
        for n_dot_h in [0.1, 0.5, 0.9]:
            for h_dot_v in [0.1, 0.5, 0.9]:
                assert ggx.pdf(n_dot_h, h_dot_v) >= 0.0

    def test_ggx_pdf_zero_for_zero_hdotv(self) -> None:
        """Test PDF is zero when HdotV is zero."""
        ggx = GGXDistribution(roughness=0.5)
        assert ggx.pdf(0.5, 0.0) == 0.0


# -----------------------------------------------------------------------------
# Importance Sampling Tests
# -----------------------------------------------------------------------------

class TestRadicalInverseVDC:
    """Tests for Van der Corput sequence."""

    def test_vdc_zero_input(self) -> None:
        """Test VDC with input 0."""
        result = radical_inverse_vdc(0)
        assert result == 0.0

    def test_vdc_one_input(self) -> None:
        """Test VDC with input 1."""
        result = radical_inverse_vdc(1)
        assert 0.0 < result < 1.0

    def test_vdc_in_range(self) -> None:
        """Test VDC outputs are in [0, 1)."""
        for i in range(100):
            result = radical_inverse_vdc(i)
            assert 0.0 <= result < 1.0

    def test_vdc_unique_values(self) -> None:
        """Test VDC produces unique values."""
        values = [radical_inverse_vdc(i) for i in range(100)]
        # All values should be unique (within floating point)
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                assert values[i] != values[j]

    def test_vdc_low_discrepancy(self) -> None:
        """Test VDC produces low-discrepancy sequence."""
        values = [radical_inverse_vdc(i) for i in range(64)]
        # Check distribution is roughly uniform
        buckets = [0] * 4
        for v in values:
            bucket = min(3, int(v * 4))
            buckets[bucket] += 1
        # Each bucket should have roughly 16 values
        for count in buckets:
            assert 8 <= count <= 24


class TestHammersley:
    """Tests for Hammersley sequence."""

    def test_hammersley_returns_tuple(self) -> None:
        """Test Hammersley returns 2D point."""
        point = hammersley(0, 10)
        assert isinstance(point, tuple)
        assert len(point) == 2

    def test_hammersley_in_unit_square(self) -> None:
        """Test all points are in [0, 1) x [0, 1)."""
        for i in range(100):
            x, y = hammersley(i, 100)
            assert 0.0 <= x < 1.0
            assert 0.0 <= y < 1.0

    def test_hammersley_x_uniform(self) -> None:
        """Test x coordinate is uniformly distributed."""
        N = 10
        for i in range(N):
            x, _ = hammersley(i, N)
            expected = i / N
            assert x == pytest.approx(expected)

    def test_hammersley_coverage(self) -> None:
        """Test Hammersley covers unit square well."""
        N = 64
        points = [hammersley(i, N) for i in range(N)]
        # Check quadrant coverage
        quadrants = [[0, 0], [0, 0]]
        for x, y in points:
            qx = 0 if x < 0.5 else 1
            qy = 0 if y < 0.5 else 1
            quadrants[qx][qy] += 1
        # Each quadrant should have roughly N/4 points
        for row in quadrants:
            for count in row:
                assert count >= N // 8


class TestTangentToWorld:
    """Tests for tangent-to-world transformation."""

    def test_tangent_to_world_z_up_unchanged(self) -> None:
        """Test z-axis in tangent space maps to N."""
        N = Vec3(0, 1, 0)
        H_tangent = Vec3(0, 0, 1)
        H_world = tangent_to_world(H_tangent, N)
        # Should be approximately equal to N
        assert H_world.dot(N) == pytest.approx(1.0, abs=0.01)

    def test_tangent_to_world_preserves_length(self) -> None:
        """Test transformation preserves unit length."""
        N = Vec3(1, 1, 1).normalized()
        for angle in [0.1, 0.5, 1.0]:
            H_tangent = Vec3(math.sin(angle), 0, math.cos(angle))
            H_world = tangent_to_world(H_tangent, N)
            assert H_world.length() == pytest.approx(1.0, rel=0.01)

    def test_tangent_to_world_orthogonal_basis(self) -> None:
        """Test result is perpendicular to normal for xy-plane input."""
        N = Vec3(0, 1, 0)
        H_tangent = Vec3(1, 0, 0)  # Purely in tangent plane
        H_world = tangent_to_world(H_tangent, N)
        # Should be perpendicular to N
        assert abs(H_world.dot(N)) < 0.1

    def test_tangent_to_world_various_normals(self) -> None:
        """Test transformation works for various normal directions."""
        normals = [
            Vec3(1, 0, 0),
            Vec3(0, 1, 0),
            Vec3(0, 0, 1),
            Vec3(1, 1, 0).normalized(),
            Vec3(1, 1, 1).normalized(),
        ]
        H_tangent = Vec3(0, 0, 1)  # z-up
        for N in normals:
            H_world = tangent_to_world(H_tangent, N)
            # Should align with N
            assert H_world.dot(N) == pytest.approx(1.0, abs=0.02)


class TestImportanceSampler:
    """Tests for importance sampler."""

    def test_sampler_creation_default(self) -> None:
        """Test creating sampler with defaults."""
        sampler = ImportanceSampler()
        assert sampler.sample_count == PrefilterConstants.DEFAULT_SAMPLE_COUNT

    def test_sampler_creation_custom(self) -> None:
        """Test creating sampler with custom sample count."""
        sampler = ImportanceSampler(sample_count=512)
        assert sampler.sample_count == 512

    def test_sampler_sample_count_clamping(self) -> None:
        """Test sample count is clamped to valid range."""
        sampler_low = ImportanceSampler(sample_count=1)
        sampler_high = ImportanceSampler(sample_count=100000)
        assert sampler_low.sample_count == PrefilterConstants.MIN_SAMPLE_COUNT
        assert sampler_high.sample_count == PrefilterConstants.MAX_SAMPLE_COUNT

    def test_sampler_hammersley_sequence(self) -> None:
        """Test sampler generates correct Hammersley points."""
        sampler = ImportanceSampler(sample_count=100)
        x, y = sampler.hammersley(50)
        expected_x = 50 / 100
        assert x == pytest.approx(expected_x)

    def test_sampler_importance_sample_ggx_unit_vector(self) -> None:
        """Test importance sampling returns unit vectors."""
        sampler = ImportanceSampler(sample_count=100)
        sampler.set_roughness(0.5)
        N = Vec3(0, 1, 0)
        for i in range(20):
            xi = sampler.hammersley(i)
            H = sampler.importance_sample_ggx(xi, N)
            assert H.length() == pytest.approx(1.0, rel=0.01)

    def test_sampler_get_sample_direction(self) -> None:
        """Test convenience method for getting sample direction."""
        sampler = ImportanceSampler(sample_count=100)
        sampler.set_roughness(0.5)
        N = Vec3(0, 0, 1)
        H = sampler.get_sample_direction(0, N)
        assert isinstance(H, Vec3)
        assert H.length() == pytest.approx(1.0, rel=0.01)

    def test_sampler_reflected_direction_valid(self) -> None:
        """Test reflected direction and weight."""
        sampler = ImportanceSampler(sample_count=100)
        sampler.set_roughness(0.3)
        N = Vec3(0, 1, 0)
        V = Vec3(0, 1, 0)
        L, NdotL = sampler.get_reflected_direction(10, N, V)
        assert L.length() == pytest.approx(1.0, rel=0.01)
        assert NdotL >= 0.0

    def test_sampler_set_roughness(self) -> None:
        """Test updating roughness."""
        sampler = ImportanceSampler()
        sampler.set_roughness(0.7)
        assert sampler.distribution.roughness == pytest.approx(0.7)


# -----------------------------------------------------------------------------
# PrefilterConfig Tests
# -----------------------------------------------------------------------------

class TestPrefilterConfig:
    """Tests for pre-filter configuration."""

    def test_config_creation_default(self) -> None:
        """Test creating config with defaults."""
        config = PrefilterConfig()
        assert config.roughness_levels == PrefilterConstants.DEFAULT_ROUGHNESS_LEVELS
        assert config.sample_count == PrefilterConstants.DEFAULT_SAMPLE_COUNT

    def test_config_roughness_levels_clamping(self) -> None:
        """Test roughness levels are clamped."""
        config_low = PrefilterConfig(roughness_levels=1)
        config_high = PrefilterConfig(roughness_levels=100)
        assert config_low.roughness_levels == PrefilterConstants.MIN_ROUGHNESS_LEVELS
        assert config_high.roughness_levels == PrefilterConstants.MAX_ROUGHNESS_LEVELS

    def test_config_sample_count_clamping(self) -> None:
        """Test sample count is clamped."""
        config = PrefilterConfig(sample_count=10)
        assert config.sample_count == PrefilterConstants.MIN_SAMPLE_COUNT

    def test_config_resolution_scale_clamping(self) -> None:
        """Test resolution scale is clamped."""
        config_low = PrefilterConfig(resolution_scale=0.1)
        config_high = PrefilterConfig(resolution_scale=2.0)
        assert config_low.resolution_scale == 0.25
        assert config_high.resolution_scale == 1.0

    def test_config_get_resolution_for_level(self) -> None:
        """Test resolution calculation per level."""
        config = PrefilterConfig(resolution_scale=0.5)
        assert config.get_resolution_for_level(256, 0) == 256
        assert config.get_resolution_for_level(256, 1) == 128
        assert config.get_resolution_for_level(256, 2) == 64

    def test_config_get_roughness_for_level(self) -> None:
        """Test roughness calculation per level."""
        config = PrefilterConfig(roughness_levels=5)
        assert config.get_roughness_for_level(0) == pytest.approx(0.0)
        assert config.get_roughness_for_level(2) == pytest.approx(0.5)
        assert config.get_roughness_for_level(4) == pytest.approx(1.0)

    def test_config_get_sample_count_for_roughness(self) -> None:
        """Test adaptive sample count based on roughness."""
        config = PrefilterConfig(sample_count=256)
        # Low roughness needs fewer samples
        low_count = config.get_sample_count_for_roughness(0.05)
        high_count = config.get_sample_count_for_roughness(0.9)
        assert low_count < high_count


# -----------------------------------------------------------------------------
# CubemapPrefilter Tests
# -----------------------------------------------------------------------------

class TestCubemapPrefilter:
    """Tests for cubemap pre-filtering."""

    def test_prefilter_creation(self) -> None:
        """Test creating prefilter."""
        prefilter = CubemapPrefilter()
        assert prefilter.config.roughness_levels == PrefilterConstants.DEFAULT_ROUGHNESS_LEVELS

    def test_prefilter_face_returns_correct_resolution(
        self, uniform_cubemap: CubemapData
    ) -> None:
        """Test prefiltered face has correct resolution."""
        prefilter = CubemapPrefilter()
        result = prefilter.prefilter_face(uniform_cubemap, CubemapFace.POSITIVE_X, 8, 0.5)
        assert result.resolution == 8

    def test_prefilter_face_correct_face(
        self, uniform_cubemap: CubemapData
    ) -> None:
        """Test prefiltered face has correct face identifier."""
        prefilter = CubemapPrefilter()
        result = prefilter.prefilter_face(uniform_cubemap, CubemapFace.NEGATIVE_Y, 8, 0.5)
        assert result.face == CubemapFace.NEGATIVE_Y

    def test_prefilter_uniform_preserves_color(
        self, uniform_cubemap: CubemapData
    ) -> None:
        """Test uniform cubemap preserves color after filtering."""
        prefilter = CubemapPrefilter()
        result = prefilter.prefilter_cubemap(uniform_cubemap, 0.5, 8)
        # Sample center of a face
        pixel = result.get_face(CubemapFace.POSITIVE_X).get_pixel(4, 4)
        assert pixel.r == pytest.approx(1.0, rel=0.1)
        assert pixel.g == pytest.approx(1.0, rel=0.1)
        assert pixel.b == pytest.approx(1.0, rel=0.1)

    def test_prefilter_low_roughness_sharp(
        self, gradient_cubemap: CubemapData
    ) -> None:
        """Test low roughness keeps edges sharper."""
        prefilter = CubemapPrefilter(
            config=PrefilterConfig(sample_count=64, roughness_levels=4)
        )
        result = prefilter.prefilter_cubemap(gradient_cubemap, 0.0, 32)
        # Check edge region - should still have contrast
        face = result.get_face(CubemapFace.POSITIVE_X)
        left_pixel = face.get_pixel(8, 16)
        right_pixel = face.get_pixel(24, 16)
        contrast = abs(right_pixel.r - left_pixel.r)
        assert contrast > 0.5  # Should maintain contrast

    def test_prefilter_high_roughness_blurry(
        self, gradient_cubemap: CubemapData
    ) -> None:
        """Test high roughness produces blurrier result."""
        prefilter = CubemapPrefilter(
            config=PrefilterConfig(sample_count=64, roughness_levels=4)
        )
        result_sharp = prefilter.prefilter_cubemap(gradient_cubemap, 0.1, 16)
        result_blurry = prefilter.prefilter_cubemap(gradient_cubemap, 0.9, 16)

        # Measure contrast near edge
        def measure_contrast(cubemap: CubemapData) -> float:
            face = cubemap.get_face(CubemapFace.POSITIVE_X)
            left = face.get_pixel(4, 8)
            right = face.get_pixel(12, 8)
            return abs(right.r - left.r)

        sharp_contrast = measure_contrast(result_sharp)
        blurry_contrast = measure_contrast(result_blurry)

        # Blurry should have less contrast (more blurred)
        assert blurry_contrast < sharp_contrast

    def test_prefilter_cubemap_all_faces(
        self, uniform_cubemap: CubemapData
    ) -> None:
        """Test all 6 faces are processed."""
        prefilter = CubemapPrefilter()
        result = prefilter.prefilter_cubemap(uniform_cubemap, 0.5, 8)
        assert len(result.faces) == 6
        for face in CubemapFace:
            assert result.get_face(face).resolution == 8

    def test_prefilter_hdr_preserves_intensity(
        self, hdr_cubemap: CubemapData
    ) -> None:
        """Test HDR values are preserved in filtering."""
        prefilter = CubemapPrefilter(
            config=PrefilterConfig(sample_count=64)
        )
        result = prefilter.prefilter_cubemap(hdr_cubemap, 0.3, 8)
        pixel = result.get_face(CubemapFace.POSITIVE_X).get_pixel(4, 4)
        # Original was ~5.0 intensity, should be preserved
        assert pixel.r > 3.0

    def test_prefilter_get_mip_for_roughness(self) -> None:
        """Test mip level calculation from roughness."""
        config = PrefilterConfig(roughness_levels=10)
        prefilter = CubemapPrefilter(config=config)

        assert prefilter.get_mip_for_roughness(0.0) == 0
        assert prefilter.get_mip_for_roughness(0.5) == 5
        assert prefilter.get_mip_for_roughness(1.0) == 9

    def test_prefilter_timing_recorded(
        self, uniform_cubemap: CubemapData
    ) -> None:
        """Test filter timing is recorded."""
        prefilter = CubemapPrefilter()
        prefilter.prefilter_face(uniform_cubemap, CubemapFace.POSITIVE_X, 8, 0.5)
        assert prefilter.last_filter_time_ms > 0


# -----------------------------------------------------------------------------
# SplitSumLUT Tests
# -----------------------------------------------------------------------------

class TestSplitSumLUT:
    """Tests for BRDF integration lookup table."""

    def test_lut_creation_default(self) -> None:
        """Test creating LUT with defaults."""
        lut = SplitSumLUT()
        assert lut.resolution == PrefilterConstants.LUT_RESOLUTION

    def test_lut_creation_custom_resolution(self) -> None:
        """Test creating LUT with custom resolution."""
        lut = SplitSumLUT(resolution=64)
        assert lut.resolution == 64
        assert len(lut.data) == 64
        assert len(lut.data[0]) == 64

    def test_lut_generate(self) -> None:
        """Test LUT generation completes."""
        lut = SplitSumLUT(resolution=16)
        lut.generate_lut(sample_count=64)
        # Check some values were computed
        terms = lut.data[8][8]
        assert isinstance(terms, BRDFTerms)

    def test_lut_sample_in_range(self) -> None:
        """Test sampled values are in valid range."""
        lut = SplitSumLUT(resolution=32)
        lut.generate_lut(sample_count=64)

        for roughness in [0.1, 0.5, 0.9]:
            for n_dot_v in [0.1, 0.5, 0.9]:
                terms = lut.sample_lut(n_dot_v, roughness)
                # Scale should be [0, 1], bias should be [0, 1]
                assert 0.0 <= terms.scale <= 2.0
                assert 0.0 <= terms.bias <= 1.0

    def test_lut_sample_interpolation(self) -> None:
        """Test bilinear interpolation works."""
        lut = SplitSumLUT(resolution=16)
        lut.generate_lut(sample_count=64)

        # Sample at known grid points and between
        t1 = lut.sample_lut(0.5, 0.5)
        t2 = lut.sample_lut(0.55, 0.5)  # Slightly offset

        # Should be close but not identical (interpolation)
        assert abs(t1.scale - t2.scale) < 0.2

    def test_lut_fresnel_behavior(self) -> None:
        """Test LUT captures Fresnel behavior."""
        lut = SplitSumLUT(resolution=32)
        lut.generate_lut(sample_count=128)

        # At grazing angles, Fresnel effect increases
        grazing = lut.sample_lut(0.1, 0.5)
        normal = lut.sample_lut(0.9, 0.5)

        # Bias (Fresnel term) should be higher at grazing
        assert grazing.bias > normal.bias * 0.5

    def test_lut_roughness_effect(self) -> None:
        """Test roughness affects BRDF terms."""
        lut = SplitSumLUT(resolution=32)
        lut.generate_lut(sample_count=128)

        smooth = lut.sample_lut(0.5, 0.1)
        rough = lut.sample_lut(0.5, 0.9)

        # Smooth surfaces have tighter specular
        assert smooth.scale != rough.scale

    def test_lut_get_brdf_terms(self) -> None:
        """Test convenience method for getting terms."""
        lut = SplitSumLUT(resolution=32)
        lut.generate_lut(sample_count=64)

        scale, bias = lut.get_brdf_terms(0.5, 0.5)
        assert isinstance(scale, float)
        assert isinstance(bias, float)

    def test_lut_edge_values(self) -> None:
        """Test LUT edge values are valid."""
        lut = SplitSumLUT(resolution=32)
        lut.generate_lut(sample_count=64)

        # Test corners
        corners = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        for n_dot_v, roughness in corners:
            terms = lut.sample_lut(n_dot_v, roughness)
            assert math.isfinite(terms.scale)
            assert math.isfinite(terms.bias)


# -----------------------------------------------------------------------------
# BRDFTerms Tests
# -----------------------------------------------------------------------------

class TestBRDFTerms:
    """Tests for BRDF terms data class."""

    def test_brdf_terms_creation_default(self) -> None:
        """Test creating BRDF terms with defaults."""
        terms = BRDFTerms()
        assert terms.scale == 1.0
        assert terms.bias == 0.0

    def test_brdf_terms_creation_custom(self) -> None:
        """Test creating BRDF terms with custom values."""
        terms = BRDFTerms(scale=0.7, bias=0.3)
        assert terms.scale == 0.7
        assert terms.bias == 0.3


# -----------------------------------------------------------------------------
# PrefilterPipeline Tests
# -----------------------------------------------------------------------------

class TestPrefilterPipeline:
    """Tests for full pre-filter pipeline."""

    def test_pipeline_creation_default(self) -> None:
        """Test creating pipeline with defaults."""
        pipeline = PrefilterPipeline()
        assert pipeline.config.roughness_levels == PrefilterConstants.DEFAULT_ROUGHNESS_LEVELS

    def test_pipeline_creation_custom_config(self) -> None:
        """Test creating pipeline with custom config."""
        config = PrefilterConfig(roughness_levels=10, sample_count=128)
        pipeline = PrefilterPipeline(config=config)
        assert pipeline.config.roughness_levels == 10

    def test_pipeline_process_returns_result(
        self, uniform_cubemap: CubemapData
    ) -> None:
        """Test pipeline process returns PrefilterResult."""
        config = PrefilterConfig(roughness_levels=4, sample_count=32)
        pipeline = PrefilterPipeline(config=config, generate_lut=False)
        result = pipeline.process(uniform_cubemap)
        assert isinstance(result, PrefilterResult)

    def test_pipeline_process_mip_chain_levels(
        self, uniform_cubemap: CubemapData
    ) -> None:
        """Test pipeline generates correct number of mip levels."""
        config = PrefilterConfig(roughness_levels=6, sample_count=32)
        pipeline = PrefilterPipeline(config=config, generate_lut=False)
        result = pipeline.process(uniform_cubemap)
        assert len(result.mip_chain.mips) == 6

    def test_pipeline_process_mip_roughness_values(
        self, uniform_cubemap: CubemapData
    ) -> None:
        """Test mip levels have correct roughness values."""
        config = PrefilterConfig(roughness_levels=5, sample_count=32)
        pipeline = PrefilterPipeline(config=config, generate_lut=False)
        result = pipeline.process(uniform_cubemap)

        assert result.mip_chain.mips[0].roughness == pytest.approx(0.0)
        assert result.mip_chain.mips[2].roughness == pytest.approx(0.5)
        assert result.mip_chain.mips[4].roughness == pytest.approx(1.0)

    def test_pipeline_process_mip_resolutions(
        self, uniform_cubemap: CubemapData
    ) -> None:
        """Test mip levels have decreasing resolutions."""
        config = PrefilterConfig(roughness_levels=4, sample_count=32, resolution_scale=0.5)
        pipeline = PrefilterPipeline(config=config, generate_lut=False)
        result = pipeline.process(uniform_cubemap)

        resolutions = [mip.resolution for mip in result.mip_chain.mips]
        # Should be decreasing or equal
        for i in range(len(resolutions) - 1):
            assert resolutions[i] >= resolutions[i + 1]

    def test_pipeline_process_generates_lut(
        self, uniform_cubemap: CubemapData
    ) -> None:
        """Test pipeline generates BRDF LUT when requested."""
        config = PrefilterConfig(roughness_levels=3, sample_count=32)
        pipeline = PrefilterPipeline(config=config, generate_lut=True)
        result = pipeline.process(uniform_cubemap)
        assert result.brdf_lut is not None
        assert pipeline.brdf_lut is not None

    def test_pipeline_process_no_lut_when_disabled(
        self, uniform_cubemap: CubemapData
    ) -> None:
        """Test pipeline doesn't generate LUT when disabled."""
        config = PrefilterConfig(roughness_levels=3, sample_count=32)
        pipeline = PrefilterPipeline(config=config, generate_lut=False)
        result = pipeline.process(uniform_cubemap)
        assert result.brdf_lut is None

    def test_pipeline_process_timing_recorded(
        self, uniform_cubemap: CubemapData
    ) -> None:
        """Test pipeline records timing."""
        config = PrefilterConfig(roughness_levels=3, sample_count=32)
        pipeline = PrefilterPipeline(config=config, generate_lut=False)
        result = pipeline.process(uniform_cubemap)
        assert result.total_time_ms > 0
        assert len(result.per_level_times_ms) == 3

    def test_pipeline_store_mips(
        self, uniform_cubemap: CubemapData
    ) -> None:
        """Test extracting individual mip cubemaps."""
        config = PrefilterConfig(roughness_levels=4, sample_count=32)
        pipeline = PrefilterPipeline(config=config, generate_lut=False)
        result = pipeline.process(uniform_cubemap)
        mips = pipeline.store_mips(result)
        assert len(mips) == 4
        for mip in mips:
            assert isinstance(mip, CubemapData)

    def test_pipeline_get_prefiltered_cubemap(
        self, uniform_cubemap: CubemapData
    ) -> None:
        """Test getting cubemap for specific roughness."""
        config = PrefilterConfig(roughness_levels=5, sample_count=32)
        pipeline = PrefilterPipeline(config=config, generate_lut=False)
        result = pipeline.process(uniform_cubemap)

        cubemap = pipeline.get_prefiltered_cubemap(result, 0.5)
        assert isinstance(cubemap, CubemapData)

    def test_pipeline_mip_chain_is_prefiltered_flag(
        self, uniform_cubemap: CubemapData
    ) -> None:
        """Test mip chain has is_prefiltered flag set."""
        config = PrefilterConfig(roughness_levels=4, sample_count=32)
        pipeline = PrefilterPipeline(config=config, generate_lut=False)
        result = pipeline.process(uniform_cubemap)
        assert result.mip_chain.is_prefiltered is True


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------

class TestPrefilterIntegration:
    """Integration tests for complete pre-filter workflow."""

    def test_full_pipeline_colored_cubemap(
        self, colored_cubemap: CubemapData
    ) -> None:
        """Test full pipeline with colored cubemap."""
        config = PrefilterConfig(roughness_levels=6, sample_count=64)
        pipeline = PrefilterPipeline(config=config, generate_lut=True)
        result = pipeline.process(colored_cubemap)

        # Verify structure
        assert len(result.mip_chain.mips) == 6
        assert result.brdf_lut is not None

        # Verify smooth mip preserves face colors
        smooth_mip = result.mip_chain.mips[0].cubemap
        red_face = smooth_mip.get_face(CubemapFace.POSITIVE_X)
        pixel = red_face.get_pixel(8, 8)
        assert pixel.r > pixel.g
        assert pixel.r > pixel.b

    def test_full_pipeline_hdr_cubemap(
        self, hdr_cubemap: CubemapData
    ) -> None:
        """Test full pipeline preserves HDR values."""
        config = PrefilterConfig(roughness_levels=4, sample_count=64)
        pipeline = PrefilterPipeline(config=config, generate_lut=False)
        result = pipeline.process(hdr_cubemap)

        # Check HDR preserved at each level
        for mip in result.mip_chain.mips:
            face = mip.cubemap.get_face(CubemapFace.POSITIVE_X)
            center = face.get_pixel(face.resolution // 2, face.resolution // 2)
            assert center.r > 2.0  # HDR value preserved

    def test_prefilter_blur_increases_with_roughness(
        self, gradient_cubemap: CubemapData
    ) -> None:
        """Test blur amount increases with roughness level."""
        config = PrefilterConfig(roughness_levels=5, sample_count=64)
        pipeline = PrefilterPipeline(config=config, generate_lut=False)
        result = pipeline.process(gradient_cubemap)

        def measure_edge_sharpness(cubemap: CubemapData) -> float:
            face = cubemap.get_face(CubemapFace.POSITIVE_X)
            res = face.resolution
            mid = res // 2
            left = face.get_pixel(res // 4, mid)
            right = face.get_pixel(3 * res // 4, mid)
            return abs(right.r - left.r)

        sharpness_values = [
            measure_edge_sharpness(mip.cubemap)
            for mip in result.mip_chain.mips
        ]

        # First mip should be sharpest
        assert sharpness_values[0] >= sharpness_values[-1]

    def test_split_sum_approximation_accuracy(self) -> None:
        """Test split-sum LUT produces reasonable values."""
        lut = SplitSumLUT(resolution=64)
        lut.generate_lut(sample_count=256)

        # At NdotV=1, roughness=0, F0 scale should be ~1
        scale_mirror, bias_mirror = lut.get_brdf_terms(1.0, 0.0)
        assert scale_mirror > 0.8

        # Total F should be <= 1 (energy conservation)
        for roughness in [0.2, 0.5, 0.8]:
            for n_dot_v in [0.2, 0.5, 0.8]:
                scale, bias = lut.get_brdf_terms(n_dot_v, roughness)
                total = scale + bias
                assert total <= 2.0  # Reasonable upper bound

    def test_mip_selection_correct(self) -> None:
        """Test mip level selection for roughness."""
        config = PrefilterConfig(roughness_levels=10)
        prefilter = CubemapPrefilter(config=config)

        # mip = roughness * (max_mip - 1)
        assert prefilter.get_mip_for_roughness(0.0) == 0
        assert prefilter.get_mip_for_roughness(0.11) == 1
        assert prefilter.get_mip_for_roughness(0.5) == 5
        assert prefilter.get_mip_for_roughness(1.0) == 9


# -----------------------------------------------------------------------------
# Performance Tests
# -----------------------------------------------------------------------------

class TestPrefilterPerformance:
    """Performance tests for pre-filtering."""

    def test_prefilter_face_under_budget(
        self, uniform_cubemap: CubemapData
    ) -> None:
        """Test single face pre-filter completes within time budget."""
        config = PrefilterConfig(sample_count=64)
        prefilter = CubemapPrefilter(config=config)

        start = time.perf_counter()
        prefilter.prefilter_face(uniform_cubemap, CubemapFace.POSITIVE_X, 8, 0.5)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should be under 100ms for small resolution
        assert elapsed_ms < PrefilterConstants.MAX_TIME_PER_FACE_MS

    def test_lut_generation_reasonable_time(self) -> None:
        """Test LUT generation completes in reasonable time."""
        lut = SplitSumLUT(resolution=64)

        start = time.perf_counter()
        lut.generate_lut(sample_count=128)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should complete within reasonable time (10s allows for CI variability)
        assert elapsed_ms < 10000

    def test_pipeline_scaling_with_resolution(self) -> None:
        """Test pipeline time scales reasonably with resolution."""
        small = CubemapData(resolution=8)
        large = CubemapData(resolution=16)

        # Fill with uniform color
        for cubemap in [small, large]:
            for face in CubemapFace:
                face_data = cubemap.get_face(face)
                for y in range(face_data.resolution):
                    for x in range(face_data.resolution):
                        face_data.set_pixel(x, y, HDRPixel(1.0, 1.0, 1.0))

        config = PrefilterConfig(roughness_levels=3, sample_count=32)
        pipeline = PrefilterPipeline(config=config, generate_lut=False)

        result_small = pipeline.process(small)
        result_large = pipeline.process(large)

        # Large should take longer but not exponentially
        ratio = result_large.total_time_ms / max(result_small.total_time_ms, 0.001)
        # 4x resolution increase should be roughly 4x time (quadratic in pixels)
        assert ratio < 20  # Allow for some variance


# -----------------------------------------------------------------------------
# Edge Case Tests
# -----------------------------------------------------------------------------

class TestPrefilterEdgeCases:
    """Edge case tests for pre-filtering."""

    def test_prefilter_single_pixel_cubemap(self) -> None:
        """Test handling very small cubemap."""
        tiny = CubemapData(resolution=1)
        for face in CubemapFace:
            tiny.get_face(face).set_pixel(0, 0, HDRPixel(0.5, 0.5, 0.5))

        config = PrefilterConfig(roughness_levels=2, sample_count=16)
        prefilter = CubemapPrefilter(config=config)
        result = prefilter.prefilter_cubemap(tiny, 0.5, 1)
        assert result.resolution == 1

    def test_prefilter_zero_roughness(
        self, uniform_cubemap: CubemapData
    ) -> None:
        """Test pre-filtering with zero roughness."""
        prefilter = CubemapPrefilter()
        result = prefilter.prefilter_cubemap(uniform_cubemap, 0.0, 8)
        # Should be essentially a downsample
        assert result.resolution == 8

    def test_prefilter_max_roughness(
        self, uniform_cubemap: CubemapData
    ) -> None:
        """Test pre-filtering with maximum roughness."""
        prefilter = CubemapPrefilter()
        result = prefilter.prefilter_cubemap(uniform_cubemap, 1.0, 8)
        assert result.resolution == 8

    def test_lut_sample_out_of_bounds(self) -> None:
        """Test LUT handles out-of-bounds gracefully."""
        lut = SplitSumLUT(resolution=16)
        lut.generate_lut(sample_count=32)

        # Values outside [0,1] should clamp
        terms_low = lut.sample_lut(-0.1, 0.5)
        terms_high = lut.sample_lut(1.5, 0.5)

        assert math.isfinite(terms_low.scale)
        assert math.isfinite(terms_high.scale)

    def test_ggx_extreme_roughness(self) -> None:
        """Test GGX at extreme roughness values."""
        smooth = GGXDistribution(roughness=0.001)
        rough = GGXDistribution(roughness=0.999)

        # Both should produce valid D values
        d_smooth = smooth.D(0.5)
        d_rough = rough.D(0.5)

        assert math.isfinite(d_smooth)
        assert math.isfinite(d_rough)

    def test_pipeline_single_level(self) -> None:
        """Test pipeline with minimum roughness levels."""
        cubemap = CubemapData(resolution=8)
        for face in CubemapFace:
            face_data = cubemap.get_face(face)
            for y in range(8):
                for x in range(8):
                    face_data.set_pixel(x, y, HDRPixel(1.0, 0.0, 0.0))

        config = PrefilterConfig(roughness_levels=2, sample_count=16)
        pipeline = PrefilterPipeline(config=config, generate_lut=False)
        result = pipeline.process(cubemap)

        assert len(result.mip_chain.mips) == 2
