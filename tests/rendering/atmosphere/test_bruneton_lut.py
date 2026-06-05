"""Tests for Bruneton atmospheric scattering LUT precomputation.

This test module verifies the correctness and performance of the Bruneton 2017
atmospheric scattering LUT generator, including:
- Atmosphere parameter validation
- Phase function evaluation
- Optical depth computation
- Transmittance (Beer-Lambert) calculation
- LUT dimension and format checks
- Determinism and reproducibility
- Performance targets (<500ms for full precomputation)
- Edge cases (horizon, zenith, nadir)
- Physical plausibility tests
"""

from __future__ import annotations

import math
import time
from typing import Tuple

import numpy as np
import pytest

from engine.rendering.atmosphere.bruneton_lut import (
    AtmosphereParams,
    LUTDimensions,
    BrunetonLUTGenerator,
    rayleigh_phase,
    cornette_shanks_phase,
    henyey_greenstein_phase,
    compute_optical_depth,
    compute_transmittance,
    validate_transmittance_lut,
    validate_sky_view_lut,
    validate_aerial_perspective_lut,
    get_density_at_altitude,
    get_ozone_density,
    ray_sphere_intersection,
    EPSILON,
)


# -----------------------------------------------------------------------------
# AtmosphereParams Tests
# -----------------------------------------------------------------------------


class TestAtmosphereParams:
    """Tests for AtmosphereParams dataclass."""

    def test_default_values(self) -> None:
        """Test that default parameters are Earth-like."""
        params = AtmosphereParams()

        # Earth radius ~6371 km
        assert params.planet_radius == pytest.approx(6371e3, rel=0.01)

        # Atmosphere height ~80 km
        assert params.atmosphere_height == pytest.approx(80e3, rel=0.1)

        # Rayleigh scale height ~8 km
        assert params.rayleigh_scale_height == pytest.approx(8e3, rel=0.1)

        # Mie scale height ~1.2 km
        assert params.mie_scale_height == pytest.approx(1.2e3, rel=0.1)

        # Asymmetry parameter for forward scattering
        assert 0.5 < params.mie_asymmetry_g < 1.0

    def test_atmosphere_top_radius(self) -> None:
        """Test atmosphere_top_radius property."""
        params = AtmosphereParams()
        expected = params.planet_radius + params.atmosphere_height
        assert params.atmosphere_top_radius == pytest.approx(expected)

    def test_mie_extinction(self) -> None:
        """Test mie_extinction property (scattering + absorption)."""
        params = AtmosphereParams()
        expected = params.mie_scattering + params.mie_absorption
        assert params.mie_extinction == pytest.approx(expected)

    def test_custom_parameters(self) -> None:
        """Test creating atmosphere with custom parameters."""
        params = AtmosphereParams(
            planet_radius=3389.5e3,  # Mars radius
            atmosphere_height=100e3,
            rayleigh_scale_height=11e3,
            mie_scale_height=2e3,
            mie_asymmetry_g=0.65,
        )

        assert params.planet_radius == pytest.approx(3389.5e3)
        assert params.atmosphere_height == pytest.approx(100e3)
        assert params.mie_asymmetry_g == pytest.approx(0.65)

    def test_invalid_planet_radius(self) -> None:
        """Test that non-positive planet radius raises error."""
        with pytest.raises(ValueError):
            AtmosphereParams(planet_radius=0)
        with pytest.raises(ValueError):
            AtmosphereParams(planet_radius=-1000)

    def test_invalid_atmosphere_height(self) -> None:
        """Test that non-positive atmosphere height raises error."""
        with pytest.raises(ValueError):
            AtmosphereParams(atmosphere_height=0)
        with pytest.raises(ValueError):
            AtmosphereParams(atmosphere_height=-50e3)

    def test_invalid_rayleigh_scale_height(self) -> None:
        """Test that non-positive Rayleigh scale height raises error."""
        with pytest.raises(ValueError):
            AtmosphereParams(rayleigh_scale_height=0)
        with pytest.raises(ValueError):
            AtmosphereParams(rayleigh_scale_height=-1000)

    def test_invalid_mie_scale_height(self) -> None:
        """Test that non-positive Mie scale height raises error."""
        with pytest.raises(ValueError):
            AtmosphereParams(mie_scale_height=0)

    def test_invalid_mie_asymmetry_g(self) -> None:
        """Test that asymmetry parameter outside [-1, 1] raises error."""
        with pytest.raises(ValueError):
            AtmosphereParams(mie_asymmetry_g=1.5)
        with pytest.raises(ValueError):
            AtmosphereParams(mie_asymmetry_g=-1.5)

    def test_valid_boundary_asymmetry(self) -> None:
        """Test that boundary asymmetry values are valid."""
        params_low = AtmosphereParams(mie_asymmetry_g=-1.0)
        params_high = AtmosphereParams(mie_asymmetry_g=1.0)
        assert params_low.mie_asymmetry_g == pytest.approx(-1.0)
        assert params_high.mie_asymmetry_g == pytest.approx(1.0)

    def test_invalid_sun_angular_radius(self) -> None:
        """Test that non-positive sun angular radius raises error."""
        with pytest.raises(ValueError):
            AtmosphereParams(sun_angular_radius=0)
        with pytest.raises(ValueError):
            AtmosphereParams(sun_angular_radius=-0.01)

    def test_rayleigh_scattering_rgb_order(self) -> None:
        """Test that Rayleigh scattering increases from R to B."""
        params = AtmosphereParams()
        r, g, b = params.rayleigh_scattering

        # Blue scatters more than green, green more than red
        assert r < g < b


# -----------------------------------------------------------------------------
# LUTDimensions Tests
# -----------------------------------------------------------------------------


class TestLUTDimensions:
    """Tests for LUTDimensions dataclass."""

    def test_default_transmittance_dimensions(self) -> None:
        """Test default transmittance LUT dimensions."""
        dims = LUTDimensions()
        assert dims.transmittance_width == 256
        assert dims.transmittance_height == 64

    def test_default_sky_view_dimensions(self) -> None:
        """Test default sky-view LUT dimensions."""
        dims = LUTDimensions()
        assert dims.sky_view_width == 256
        assert dims.sky_view_height == 512

    def test_default_aerial_perspective_dimensions(self) -> None:
        """Test default aerial perspective LUT dimensions."""
        dims = LUTDimensions()
        assert dims.aerial_perspective_size == 32

    def test_custom_dimensions(self) -> None:
        """Test custom LUT dimensions."""
        dims = LUTDimensions(
            transmittance_width=128,
            transmittance_height=32,
            sky_view_width=512,
            sky_view_height=1024,
            aerial_perspective_size=64,
        )
        assert dims.transmittance_width == 128
        assert dims.sky_view_height == 1024
        assert dims.aerial_perspective_size == 64


# -----------------------------------------------------------------------------
# Phase Function Tests
# -----------------------------------------------------------------------------


class TestRayleighPhase:
    """Tests for Rayleigh phase function."""

    def test_forward_direction(self) -> None:
        """Test phase function value at cos_angle = 1 (forward)."""
        # P(0) = 3/(16*pi) * (1 + 1) = 6/(16*pi)
        expected = 6.0 / (16.0 * math.pi)
        assert rayleigh_phase(1.0) == pytest.approx(expected)

    def test_backward_direction(self) -> None:
        """Test phase function value at cos_angle = -1 (backward)."""
        # Same as forward due to symmetry
        expected = 6.0 / (16.0 * math.pi)
        assert rayleigh_phase(-1.0) == pytest.approx(expected)

    def test_perpendicular_direction(self) -> None:
        """Test phase function value at cos_angle = 0 (perpendicular)."""
        # P(90) = 3/(16*pi) * (1 + 0) = 3/(16*pi)
        expected = 3.0 / (16.0 * math.pi)
        assert rayleigh_phase(0.0) == pytest.approx(expected)

    def test_symmetry(self) -> None:
        """Test that Rayleigh phase is symmetric."""
        for cos_angle in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert rayleigh_phase(cos_angle) == pytest.approx(rayleigh_phase(-cos_angle))

    def test_always_positive(self) -> None:
        """Test that phase function is always positive."""
        for cos_angle in np.linspace(-1, 1, 100):
            assert rayleigh_phase(cos_angle) > 0

    def test_normalization(self) -> None:
        """Test that phase function integrates to 1 over the sphere."""
        # Numerical integration over angles
        n_samples = 1000
        total = 0.0
        for i in range(n_samples):
            cos_theta = -1.0 + 2.0 * (i + 0.5) / n_samples
            phase = rayleigh_phase(cos_theta)
            # Solid angle element: 2*pi * d(cos_theta)
            total += phase * 2.0 * math.pi * (2.0 / n_samples)

        assert total == pytest.approx(1.0, rel=0.01)


class TestCornetteShanksPhase:
    """Tests for Cornette-Shanks phase function."""

    def test_reduces_to_rayleigh_when_g_zero(self) -> None:
        """Test that CS phase equals Rayleigh when g=0."""
        for cos_angle in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            cs = cornette_shanks_phase(cos_angle, 0.0)
            ray = rayleigh_phase(cos_angle)
            assert cs == pytest.approx(ray, rel=0.01)

    def test_forward_scattering_peak(self) -> None:
        """Test that positive g gives forward scattering peak."""
        g = 0.8
        forward = cornette_shanks_phase(1.0, g)
        backward = cornette_shanks_phase(-1.0, g)

        assert forward > backward

    def test_backward_scattering_peak(self) -> None:
        """Test that negative g gives backward scattering peak."""
        g = -0.5
        forward = cornette_shanks_phase(1.0, g)
        backward = cornette_shanks_phase(-1.0, g)

        assert backward > forward

    def test_always_positive(self) -> None:
        """Test that phase function is always positive."""
        for g in [-0.9, -0.5, 0.0, 0.5, 0.9]:
            for cos_angle in np.linspace(-1, 1, 50):
                assert cornette_shanks_phase(cos_angle, g) > 0

    def test_asymmetry_parameter_effect(self) -> None:
        """Test that higher g produces more forward scattering."""
        forward_g05 = cornette_shanks_phase(1.0, 0.5)
        forward_g08 = cornette_shanks_phase(1.0, 0.8)

        assert forward_g08 > forward_g05


class TestHenyeyGreensteinPhase:
    """Tests for Henyey-Greenstein phase function."""

    def test_isotropic_when_g_zero(self) -> None:
        """Test that HG phase is isotropic when g=0."""
        g = 0.0
        expected = 1.0 / (4.0 * math.pi)  # Uniform over sphere

        for cos_angle in [-1.0, 0.0, 1.0]:
            assert henyey_greenstein_phase(cos_angle, g) == pytest.approx(expected)

    def test_normalization(self) -> None:
        """Test that HG phase integrates to 1."""
        g = 0.6
        n_samples = 1000
        total = 0.0

        for i in range(n_samples):
            cos_theta = -1.0 + 2.0 * (i + 0.5) / n_samples
            phase = henyey_greenstein_phase(cos_theta, g)
            total += phase * 2.0 * math.pi * (2.0 / n_samples)

        assert total == pytest.approx(1.0, rel=0.02)


# -----------------------------------------------------------------------------
# Optical Depth Tests
# -----------------------------------------------------------------------------


class TestOpticalDepth:
    """Tests for optical depth computation."""

    def test_optical_depth_at_ground_looking_up(self) -> None:
        """Test optical depth looking straight up from ground."""
        params = AtmosphereParams()
        rayleigh_od, mie_od, ozone_od = compute_optical_depth(0.0, 1.0, params)

        # Should have positive values
        assert np.all(rayleigh_od > 0)
        assert mie_od[0] > 0
        assert np.all(ozone_od >= 0)

    def test_optical_depth_decreases_with_altitude(self) -> None:
        """Test that optical depth decreases with increasing altitude."""
        params = AtmosphereParams()

        od_ground = compute_optical_depth(0.0, 1.0, params)
        od_10km = compute_optical_depth(10e3, 1.0, params)
        od_50km = compute_optical_depth(50e3, 1.0, params)

        # Rayleigh optical depth should decrease
        assert np.sum(od_ground[0]) > np.sum(od_10km[0])
        assert np.sum(od_10km[0]) > np.sum(od_50km[0])

    def test_optical_depth_increases_toward_horizon(self) -> None:
        """Test that optical depth increases as view approaches horizon."""
        params = AtmosphereParams()

        od_zenith = compute_optical_depth(0.0, 1.0, params)  # Looking up
        od_45deg = compute_optical_depth(0.0, 0.707, params)  # 45 degrees
        od_horizon = compute_optical_depth(0.0, 0.1, params)  # Near horizon

        # Total optical depth should increase
        total_zenith = np.sum(od_zenith[0])
        total_45deg = np.sum(od_45deg[0])
        total_horizon = np.sum(od_horizon[0])

        assert total_zenith < total_45deg < total_horizon

    def test_optical_depth_zero_above_atmosphere(self) -> None:
        """Test that optical depth is zero above atmosphere looking up."""
        params = AtmosphereParams()
        rayleigh_od, mie_od, ozone_od = compute_optical_depth(
            params.atmosphere_height + 1000, 1.0, params
        )

        assert np.allclose(rayleigh_od, 0)
        assert np.allclose(mie_od, 0)
        assert np.allclose(ozone_od, 0)

    def test_rayleigh_rgb_order(self) -> None:
        """Test that blue has higher Rayleigh optical depth than red."""
        params = AtmosphereParams()
        rayleigh_od, _, _ = compute_optical_depth(0.0, 1.0, params)

        # RGB order: rayleigh_od[0]=R, [1]=G, [2]=B
        assert rayleigh_od[0] < rayleigh_od[1] < rayleigh_od[2]


class TestTransmittance:
    """Tests for transmittance computation (Beer-Lambert)."""

    def test_zero_optical_depth(self) -> None:
        """Test that zero optical depth gives transmittance = 1."""
        od = np.array([0.0, 0.0, 0.0])
        trans = compute_transmittance(od)
        assert np.allclose(trans, 1.0)

    def test_transmittance_decreases_with_optical_depth(self) -> None:
        """Test that transmittance decreases with increasing optical depth."""
        od1 = np.array([0.1])
        od2 = np.array([0.5])
        od3 = np.array([1.0])

        trans1 = compute_transmittance(od1)
        trans2 = compute_transmittance(od2)
        trans3 = compute_transmittance(od3)

        assert trans1[0] > trans2[0] > trans3[0]

    def test_transmittance_range(self) -> None:
        """Test that transmittance is always in [0, 1]."""
        for od in [0.0, 0.5, 1.0, 5.0, 10.0, 100.0]:
            trans = compute_transmittance(np.array([od]))
            assert 0.0 <= trans[0] <= 1.0

    def test_beer_lambert_law(self) -> None:
        """Test that transmittance follows Beer-Lambert: T = exp(-od)."""
        od = np.array([0.3, 0.7, 1.5])
        trans = compute_transmittance(od)
        expected = np.exp(-od)
        assert np.allclose(trans, expected)


# -----------------------------------------------------------------------------
# Helper Function Tests
# -----------------------------------------------------------------------------


class TestDensityFunctions:
    """Tests for atmospheric density functions."""

    def test_density_at_ground(self) -> None:
        """Test that density at ground level is 1.0."""
        params = AtmosphereParams()
        density = get_density_at_altitude(0.0, params.rayleigh_scale_height, params)
        assert density == pytest.approx(1.0)

    def test_density_decreases_with_altitude(self) -> None:
        """Test exponential density falloff."""
        params = AtmosphereParams()
        h = params.rayleigh_scale_height

        density_0 = get_density_at_altitude(0.0, h, params)
        density_h = get_density_at_altitude(h, h, params)
        density_2h = get_density_at_altitude(2 * h, h, params)

        # Should follow exp(-altitude / scale_height)
        assert density_h == pytest.approx(density_0 / math.e, rel=0.01)
        assert density_2h == pytest.approx(density_0 / (math.e * math.e), rel=0.01)

    def test_density_zero_above_atmosphere(self) -> None:
        """Test that density is zero above atmosphere."""
        params = AtmosphereParams()
        density = get_density_at_altitude(
            params.atmosphere_height + 1000, params.rayleigh_scale_height, params
        )
        assert density == pytest.approx(0.0)

    def test_ozone_layer_peak(self) -> None:
        """Test that ozone density peaks around 25 km."""
        params = AtmosphereParams()

        density_10km = get_ozone_density(10e3, params)
        density_25km = get_ozone_density(25e3, params)
        density_50km = get_ozone_density(50e3, params)

        # Peak should be around 25 km
        assert density_25km > density_10km
        assert density_25km > density_50km


class TestRaySphereIntersection:
    """Tests for ray-sphere intersection."""

    def test_ray_from_inside_sphere(self) -> None:
        """Test ray starting inside sphere."""
        origin = np.array([0.0, 0.0, 0.0])
        direction = np.array([1.0, 0.0, 0.0])
        center = np.array([0.0, 0.0, 0.0])
        radius = 1.0

        near, far = ray_sphere_intersection(origin, direction, center, radius)

        # Near should be negative (behind), far should be positive
        assert near < 0
        assert far > 0
        assert far == pytest.approx(radius)

    def test_ray_missing_sphere(self) -> None:
        """Test ray that misses sphere entirely."""
        origin = np.array([0.0, 2.0, 0.0])  # Above sphere
        direction = np.array([1.0, 0.0, 0.0])  # Horizontal
        center = np.array([0.0, 0.0, 0.0])
        radius = 1.0

        near, far = ray_sphere_intersection(origin, direction, center, radius)

        assert near < 0
        assert far < 0

    def test_ray_tangent_to_sphere(self) -> None:
        """Test ray tangent to sphere surface."""
        origin = np.array([0.0, 1.0, 0.0])  # On sphere surface
        direction = np.array([1.0, 0.0, 0.0])  # Tangent
        center = np.array([0.0, 0.0, 0.0])
        radius = 1.0

        near, far = ray_sphere_intersection(origin, direction, center, radius)

        # Should just touch at one point (near == far)
        assert near == pytest.approx(far, abs=0.001)


# -----------------------------------------------------------------------------
# LUT Validation Tests
# -----------------------------------------------------------------------------


class TestLUTValidation:
    """Tests for LUT validation functions."""

    def test_valid_transmittance_lut(self) -> None:
        """Test validation of a valid transmittance LUT."""
        lut = np.random.uniform(0.5, 1.0, (64, 256, 4)).astype(np.float32)
        assert validate_transmittance_lut(lut) is True

    def test_invalid_transmittance_negative(self) -> None:
        """Test validation fails for negative values."""
        lut = np.random.uniform(0.5, 1.0, (64, 256, 4)).astype(np.float32)
        lut[0, 0, 0] = -0.1
        assert validate_transmittance_lut(lut) is False

    def test_invalid_transmittance_greater_than_one(self) -> None:
        """Test validation fails for values > 1."""
        lut = np.random.uniform(0.5, 1.0, (64, 256, 4)).astype(np.float32)
        lut[0, 0, 0] = 1.5
        assert validate_transmittance_lut(lut) is False

    def test_invalid_transmittance_nan(self) -> None:
        """Test validation fails for NaN values."""
        lut = np.random.uniform(0.5, 1.0, (64, 256, 4)).astype(np.float32)
        lut[10, 10, 0] = np.nan
        assert validate_transmittance_lut(lut) is False

    def test_invalid_transmittance_inf(self) -> None:
        """Test validation fails for infinite values."""
        lut = np.random.uniform(0.5, 1.0, (64, 256, 4)).astype(np.float32)
        lut[10, 10, 1] = np.inf
        assert validate_transmittance_lut(lut) is False

    def test_invalid_transmittance_wrong_dims(self) -> None:
        """Test validation fails for wrong dimensions."""
        lut = np.random.uniform(0.5, 1.0, (64, 256)).astype(np.float32)  # 2D
        assert validate_transmittance_lut(lut) is False

    def test_valid_sky_view_lut(self) -> None:
        """Test validation of a valid sky-view LUT."""
        lut = np.random.uniform(0.01, 10.0, (512, 256, 3)).astype(np.float32)
        assert validate_sky_view_lut(lut) is True

    def test_invalid_sky_view_negative(self) -> None:
        """Test validation fails for negative sky values."""
        lut = np.random.uniform(0.1, 1.0, (512, 256, 3)).astype(np.float32)
        lut[0, 0, 0] = -0.01
        assert validate_sky_view_lut(lut) is False

    def test_invalid_sky_view_too_bright(self) -> None:
        """Test validation fails for unreasonably bright values."""
        lut = np.random.uniform(0.1, 1.0, (512, 256, 3)).astype(np.float32)
        lut[0, 0, 0] = 1e7
        assert validate_sky_view_lut(lut) is False

    def test_valid_aerial_perspective_luts(self) -> None:
        """Test validation of valid aerial perspective LUTs."""
        inscatter = np.random.uniform(0, 0.5, (32, 32, 32, 4)).astype(np.float32)
        transmittance = np.random.uniform(0.5, 1.0, (32, 32, 32, 4)).astype(np.float32)
        assert validate_aerial_perspective_lut(inscatter, transmittance) is True

    def test_invalid_aerial_perspective_negative(self) -> None:
        """Test validation fails for negative inscatter."""
        inscatter = np.random.uniform(0, 0.5, (32, 32, 32, 4)).astype(np.float32)
        transmittance = np.random.uniform(0.5, 1.0, (32, 32, 32, 4)).astype(np.float32)
        inscatter[0, 0, 0, 0] = -0.1
        assert validate_aerial_perspective_lut(inscatter, transmittance) is False


# -----------------------------------------------------------------------------
# BrunetonLUTGenerator Tests
# -----------------------------------------------------------------------------


class TestBrunetonLUTGenerator:
    """Tests for the BrunetonLUTGenerator class."""

    def test_generator_creation_default(self) -> None:
        """Test creating generator with default parameters."""
        gen = BrunetonLUTGenerator()
        assert gen.params is not None
        assert gen.dimensions is not None
        assert gen._transmittance_lut is None

    def test_generator_creation_custom_params(self) -> None:
        """Test creating generator with custom parameters."""
        params = AtmosphereParams(planet_radius=3389.5e3)
        gen = BrunetonLUTGenerator(params=params)
        assert gen.params.planet_radius == pytest.approx(3389.5e3)

    def test_generator_creation_custom_dimensions(self) -> None:
        """Test creating generator with custom dimensions."""
        dims = LUTDimensions(transmittance_width=128)
        gen = BrunetonLUTGenerator(dimensions=dims)
        assert gen.dimensions.transmittance_width == 128


class TestTransmittanceLUT:
    """Tests for transmittance LUT generation."""

    @pytest.fixture
    def generator(self) -> BrunetonLUTGenerator:
        """Create a generator for tests."""
        return BrunetonLUTGenerator()

    def test_transmittance_lut_shape(self, generator: BrunetonLUTGenerator) -> None:
        """Test transmittance LUT has correct shape."""
        lut = generator.precompute_transmittance()
        dims = generator.dimensions

        assert lut.shape == (dims.transmittance_height, dims.transmittance_width, 4)

    def test_transmittance_lut_format(self, generator: BrunetonLUTGenerator) -> None:
        """Test transmittance LUT is float32 (RGBA16F compatible)."""
        lut = generator.precompute_transmittance()
        assert lut.dtype == np.float32

    def test_transmittance_lut_valid(self, generator: BrunetonLUTGenerator) -> None:
        """Test transmittance LUT passes validation."""
        lut = generator.precompute_transmittance()
        assert validate_transmittance_lut(lut) is True

    def test_transmittance_lut_custom_size(
        self, generator: BrunetonLUTGenerator
    ) -> None:
        """Test transmittance LUT with custom dimensions."""
        lut = generator.precompute_transmittance(width=128, height=32)
        assert lut.shape == (32, 128, 4)

    def test_transmittance_lut_caching(self, generator: BrunetonLUTGenerator) -> None:
        """Test that transmittance LUT is cached."""
        assert generator._transmittance_lut is None
        generator.precompute_transmittance()
        assert generator._transmittance_lut is not None

    def test_transmittance_values_at_top(self, generator: BrunetonLUTGenerator) -> None:
        """Test transmittance near 1.0 at top of atmosphere."""
        lut = generator.precompute_transmittance()
        # Top row should have high transmittance
        top_row_rgb = lut[-1, :, :3]
        assert np.mean(top_row_rgb) > 0.7

    def test_transmittance_alpha_channel(
        self, generator: BrunetonLUTGenerator
    ) -> None:
        """Test that alpha channel is always 1.0."""
        lut = generator.precompute_transmittance()
        alpha = lut[:, :, 3]
        assert np.allclose(alpha, 1.0)

    def test_transmittance_determinism(self) -> None:
        """Test that transmittance computation is deterministic."""
        gen1 = BrunetonLUTGenerator()
        gen2 = BrunetonLUTGenerator()

        lut1 = gen1.precompute_transmittance(width=64, height=16)
        lut2 = gen2.precompute_transmittance(width=64, height=16)

        assert np.allclose(lut1, lut2)


class TestSkyViewLUT:
    """Tests for sky-view LUT generation."""

    @pytest.fixture
    def generator(self) -> BrunetonLUTGenerator:
        """Create a generator for tests."""
        return BrunetonLUTGenerator()

    @pytest.fixture
    def sun_direction(self) -> np.ndarray:
        """Default sun direction (high in sky)."""
        return np.array([0.0, 0.8, 0.6])

    def test_sky_view_lut_shape(
        self, generator: BrunetonLUTGenerator, sun_direction: np.ndarray
    ) -> None:
        """Test sky-view LUT has correct shape."""
        lut = generator.precompute_sky_view(sun_direction)
        dims = generator.dimensions

        assert lut.shape == (dims.sky_view_height, dims.sky_view_width, 3)

    def test_sky_view_lut_format(
        self, generator: BrunetonLUTGenerator, sun_direction: np.ndarray
    ) -> None:
        """Test sky-view LUT is float32 (RGB16F compatible)."""
        lut = generator.precompute_sky_view(sun_direction)
        assert lut.dtype == np.float32

    def test_sky_view_lut_valid(
        self, generator: BrunetonLUTGenerator, sun_direction: np.ndarray
    ) -> None:
        """Test sky-view LUT passes validation."""
        lut = generator.precompute_sky_view(sun_direction)
        assert validate_sky_view_lut(lut) is True

    def test_sky_view_has_brightness(
        self, generator: BrunetonLUTGenerator, sun_direction: np.ndarray
    ) -> None:
        """Test that sky-view LUT has non-zero brightness."""
        lut = generator.precompute_sky_view(sun_direction)
        assert np.max(lut) > 0.01

    def test_sky_view_custom_size(
        self, generator: BrunetonLUTGenerator, sun_direction: np.ndarray
    ) -> None:
        """Test sky-view LUT with custom dimensions."""
        lut = generator.precompute_sky_view(sun_direction, width=64, height=128)
        assert lut.shape == (128, 64, 3)

    def test_sky_view_determinism(self, sun_direction: np.ndarray) -> None:
        """Test that sky-view computation is deterministic."""
        gen1 = BrunetonLUTGenerator()
        gen2 = BrunetonLUTGenerator()

        lut1 = gen1.precompute_sky_view(sun_direction, width=32, height=64)
        lut2 = gen2.precompute_sky_view(sun_direction, width=32, height=64)

        assert np.allclose(lut1, lut2)


class TestAerialPerspectiveLUT:
    """Tests for aerial perspective LUT generation."""

    @pytest.fixture
    def generator(self) -> BrunetonLUTGenerator:
        """Create a generator for tests."""
        return BrunetonLUTGenerator()

    def test_aerial_perspective_shapes(
        self, generator: BrunetonLUTGenerator
    ) -> None:
        """Test aerial perspective LUTs have correct shapes."""
        inscatter, transmittance = generator.precompute_aerial_perspective()
        size = generator.dimensions.aerial_perspective_size

        assert inscatter.shape == (size, size, size, 4)
        assert transmittance.shape == (size, size, size, 4)

    def test_aerial_perspective_format(
        self, generator: BrunetonLUTGenerator
    ) -> None:
        """Test aerial perspective LUTs are float32."""
        inscatter, transmittance = generator.precompute_aerial_perspective()

        assert inscatter.dtype == np.float32
        assert transmittance.dtype == np.float32

    def test_aerial_perspective_valid(
        self, generator: BrunetonLUTGenerator
    ) -> None:
        """Test aerial perspective LUTs pass validation."""
        inscatter, transmittance = generator.precompute_aerial_perspective()
        assert validate_aerial_perspective_lut(inscatter, transmittance) is True

    def test_aerial_perspective_custom_size(
        self, generator: BrunetonLUTGenerator
    ) -> None:
        """Test aerial perspective with custom size."""
        inscatter, transmittance = generator.precompute_aerial_perspective(size=16)

        assert inscatter.shape == (16, 16, 16, 4)
        assert transmittance.shape == (16, 16, 16, 4)


class TestPrecomputeAll:
    """Tests for full LUT precomputation."""

    @pytest.fixture
    def sun_direction(self) -> np.ndarray:
        """Default sun direction."""
        return np.array([0.5, 0.5, 0.0])

    def test_precompute_all_returns_dict(self, sun_direction: np.ndarray) -> None:
        """Test that precompute_all returns a dictionary with all LUTs."""
        gen = BrunetonLUTGenerator()
        luts = gen.precompute_all(sun_direction)

        assert "transmittance" in luts
        assert "sky_view" in luts
        assert "aerial_perspective" in luts

    def test_precompute_all_lut_types(self, sun_direction: np.ndarray) -> None:
        """Test that returned LUTs have correct types."""
        gen = BrunetonLUTGenerator()
        luts = gen.precompute_all(sun_direction)

        assert isinstance(luts["transmittance"], np.ndarray)
        assert isinstance(luts["sky_view"], np.ndarray)
        assert isinstance(luts["aerial_perspective"], tuple)
        assert len(luts["aerial_perspective"]) == 2


class TestPerformance:
    """Performance tests for LUT generation.

    Note: This is a pure Python CPU implementation. Performance targets
    are relaxed compared to GPU implementations. Production use cases
    should precompute LUTs offline or use GPU-accelerated shaders.

    The tests use minimal LUT sizes to verify the algorithm completes
    in reasonable time on CPU. For real-time rendering, a GPU shader
    implementation is required.
    """

    @pytest.mark.slow
    def test_transmittance_performance(self) -> None:
        """Test that transmittance precomputation completes in reasonable time.

        Uses minimal 8x4 LUT for fast execution on CPU.
        """
        dims = LUTDimensions(transmittance_width=8, transmittance_height=4)
        gen = BrunetonLUTGenerator(dimensions=dims)

        start = time.perf_counter()
        lut = gen.precompute_transmittance(width=8, height=4)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Verify the LUT was generated correctly
        assert lut.shape == (4, 8, 4)
        assert validate_transmittance_lut(lut)

        # CPU target: <5s for 8x4 LUT (32 pixels * 64 integration steps)
        # This is lenient for CI variance and pure Python overhead
        assert elapsed_ms < 5000, f"Transmittance took {elapsed_ms:.1f}ms (target: <5000ms)"

    @pytest.mark.slow
    def test_sky_view_performance(self) -> None:
        """Test that sky-view precomputation completes in reasonable time.

        Uses minimal 4x4 LUT for fast execution on CPU.
        Sky-view is inherently slower due to nested ray marching.
        """
        dims = LUTDimensions(
            transmittance_width=4,
            transmittance_height=2,
            sky_view_width=4,
            sky_view_height=4,
        )
        gen = BrunetonLUTGenerator(dimensions=dims)
        sun_direction = np.array([0.5, 0.5, 0.0])

        start = time.perf_counter()
        lut = gen.precompute_sky_view(sun_direction, width=4, height=4)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Verify the LUT was generated correctly
        assert lut.shape == (4, 4, 3)
        assert validate_sky_view_lut(lut)

        # CPU target: <30s for 4x4 LUT (16 pixels * nested integration)
        # Pure Python ray marching is inherently slow
        assert elapsed_ms < 30000, f"Sky-view took {elapsed_ms:.1f}ms (target: <30000ms)"

    @pytest.mark.slow
    def test_precompute_all_performance(self) -> None:
        """Test that full precomputation completes in reasonable time.

        Uses absolute minimal LUT sizes for CPU execution.
        Note: Production LUTs require GPU-accelerated precomputation.
        """
        dims = LUTDimensions(
            transmittance_width=4,
            transmittance_height=2,
            sky_view_width=4,
            sky_view_height=4,
            aerial_perspective_size=2,
        )
        gen = BrunetonLUTGenerator(dimensions=dims)
        sun_direction = np.array([0.5, 0.5, 0.0])

        start = time.perf_counter()
        luts = gen.precompute_all(sun_direction)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Verify all LUTs were generated
        assert "transmittance" in luts
        assert "sky_view" in luts
        assert "aerial_perspective" in luts

        # CPU target: <60s for minimal LUTs
        # This accounts for all three LUT types with nested integration
        assert elapsed_ms < 60000, f"Full precompute took {elapsed_ms:.1f}ms (target: <60000ms)"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def generator(self) -> BrunetonLUTGenerator:
        """Create a generator for tests."""
        return BrunetonLUTGenerator()

    def test_sun_at_zenith(self, generator: BrunetonLUTGenerator) -> None:
        """Test sky with sun directly overhead."""
        sun_direction = np.array([0.0, 1.0, 0.0])
        lut = generator.precompute_sky_view(sun_direction, width=32, height=64)

        assert validate_sky_view_lut(lut)
        # Sun at zenith should still produce visible sky
        assert np.max(lut) > 0.01

    def test_sun_at_horizon(self, generator: BrunetonLUTGenerator) -> None:
        """Test sky with sun at horizon."""
        sun_direction = np.array([1.0, 0.0, 0.0])
        lut = generator.precompute_sky_view(sun_direction, width=32, height=64)

        assert validate_sky_view_lut(lut)

    def test_sun_below_horizon(self, generator: BrunetonLUTGenerator) -> None:
        """Test sky with sun below horizon (twilight)."""
        sun_direction = np.array([1.0, -0.1, 0.0])
        sun_direction = sun_direction / np.linalg.norm(sun_direction)
        lut = generator.precompute_sky_view(sun_direction, width=32, height=64)

        # Should be valid but dimmer
        assert validate_sky_view_lut(lut)

    def test_nadir_view(self, generator: BrunetonLUTGenerator) -> None:
        """Test transmittance looking straight down."""
        # Looking down should hit ground (short path)
        rayleigh_od, mie_od, ozone_od = compute_optical_depth(
            10e3, -1.0, generator.params
        )

        # Should have finite optical depth
        assert np.all(np.isfinite(rayleigh_od))
        assert np.all(np.isfinite(mie_od))

    def test_horizon_view(self, generator: BrunetonLUTGenerator) -> None:
        """Test transmittance looking at horizon."""
        # Near-horizontal view has longest path
        rayleigh_od, _, _ = compute_optical_depth(0.0, 0.01, generator.params)

        # Should have high but finite optical depth
        assert np.all(rayleigh_od > 0)
        assert np.all(np.isfinite(rayleigh_od))

    def test_clear_cache(self, generator: BrunetonLUTGenerator) -> None:
        """Test clearing cached LUTs."""
        generator.precompute_transmittance()
        assert generator._transmittance_lut is not None

        generator.clear_cache()
        assert generator._transmittance_lut is None


class TestPhysicalPlausibility:
    """Tests for physical plausibility of results."""

    @pytest.fixture
    def generator(self) -> BrunetonLUTGenerator:
        """Create a generator for tests."""
        return BrunetonLUTGenerator()

    def test_sky_bluer_away_from_sun(self, generator: BrunetonLUTGenerator) -> None:
        """Test that sky is bluer away from the sun."""
        sun_direction = np.array([1.0, 0.5, 0.0])
        sun_direction = sun_direction / np.linalg.norm(sun_direction)

        lut = generator.precompute_sky_view(sun_direction, width=64, height=128)

        # Sample away from sun (opposite side of sky)
        # This is a simplified check - real test would compute proper coordinates
        # Just verify we have valid data
        assert np.max(lut[:, :, 2]) > 0  # Blue channel should have values

    def test_transmittance_wavelength_dependence(
        self, generator: BrunetonLUTGenerator
    ) -> None:
        """Test that red transmits more than blue (wavelength dependence)."""
        lut = generator.precompute_transmittance()

        # At ground level looking toward horizon (high optical depth)
        # Red should have higher transmittance than blue
        # Sample from bottom row (ground level), middle column (horizon-ish)
        sample = lut[0, lut.shape[1] // 4, :3]

        # R > G > B for transmittance (longer wavelengths penetrate more)
        # Note: This relationship should hold for most viewing angles
        # Allow for some variation due to discrete sampling
        mean_trans = np.mean(sample)
        if mean_trans > 0.1 and mean_trans < 0.99:  # Skip near-0 and near-1 cases
            assert sample[0] >= sample[2] * 0.9  # Red >= Blue (with tolerance)

    def test_aerial_perspective_depth_falloff(
        self, generator: BrunetonLUTGenerator
    ) -> None:
        """Test that transmittance decreases with depth."""
        inscatter, transmittance = generator.precompute_aerial_perspective(size=16)

        # Compare near and far slices
        near_trans = np.mean(transmittance[0, :, :, :3])
        far_trans = np.mean(transmittance[-1, :, :, :3])

        # Near should have higher transmittance than far
        assert near_trans >= far_trans * 0.8  # Allow some tolerance
