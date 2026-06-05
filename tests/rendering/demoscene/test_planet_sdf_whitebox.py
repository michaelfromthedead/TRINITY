"""
Whitebox tests for PlanetSDF (T-DEMO-4.9).

Tests internal implementation details of spherical terrain generation:
- Spherical coordinate conversion
- FBM noise sampling in spherical space
- Terrain height calculation
- Ocean level clipping
- Crater formation mathematics
- Polar singularity handling
- WGSL code generation internals

WHITEBOX coverage plan:
  Path A: SphericalCoord.from_cartesian basic conversion
  Path B: SphericalCoord.from_cartesian at poles (phi = +/- pi/2)
  Path C: SphericalCoord.from_cartesian at origin singularity
  Path D: SphericalCoord.to_cartesian roundtrip
  Path E: SphericalCoord.normalized_uv range [0, 1]
  Path F: _hash21 determinism
  Path G: _hash21 range [0, 1)
  Path H: _hash31 determinism
  Path I: _hash31 range [0, 1)
  Path J: _value_noise_2d range [-1, 1]
  Path K: _value_noise_2d continuity
  Path L: _value_noise_3d range [-1, 1]
  Path M: _value_noise_3d continuity
  Path N: _fbm_2d zero octaves returns 0
  Path O: _fbm_2d single octave equals base noise
  Path P: _fbm_2d amplitude normalization
  Path Q: _fbm_3d determinism
  Path R: _fbm_spherical no seams at theta = +/- pi
  Path S: _fbm_spherical pole behavior
  Path T: PlanetSDF._sample_terrain_height amplitude scaling
  Path U: PlanetSDF._sample_terrain_height ocean clipping
  Path V: PlanetSDF._sample_terrain_height mountain modulation
  Path W: PlanetSDF._evaluate_crater distance calculation
  Path X: PlanetSDF.evaluate center singularity
  Path Y: PlanetSDF.evaluate radial displacement
  Path Z: PlanetSDF.evaluate_atmosphere shell distance
  Path AA: PlanetConfig validation errors
  Path AB: CraterConfig validation errors
  Path AC: PlanetSDF.add_crater and clear_craters
  Path AD: PlanetSDF.is_ocean boundary detection
  Path AE: PlanetSDF.is_continent noise threshold
  Path AF: PlanetSDF.sample_normal gradient direction
  Path AG: PlanetSDF.to_wgsl output contains required elements
  Path AH: PlanetSDF dirty tracking
"""

from __future__ import annotations

import math
from typing import List, Tuple

import pytest

# Module under test
from engine.rendering.demoscene.planet_sdf import (
    PlanetSDF,
    PlanetConfig,
    CraterConfig,
    SphericalCoord,
    _hash21,
    _hash31,
    _wgsl_fract,
    _smoothstep,
    _lerp,
    _value_noise_2d,
    _value_noise_3d,
    _fbm_2d,
    _fbm_3d,
    _fbm_spherical,
    PI,
    TWO_PI,
    HALF_PI,
)
from engine.rendering.demoscene.sdf_ast import Vec3


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def default_planet() -> PlanetSDF:
    """Create a default planet for testing."""
    return PlanetSDF(planet_radius=1.0, terrain_amplitude=0.1)


@pytest.fixture
def flat_planet() -> PlanetSDF:
    """Create a planet with no terrain (smooth sphere)."""
    return PlanetSDF(
        planet_radius=1.0,
        terrain_amplitude=0.0,
        noise_octaves=1,
    )


@pytest.fixture
def planet_with_craters() -> PlanetSDF:
    """Create a planet with configured craters."""
    config = PlanetConfig(
        planet_radius=1.0,
        terrain_amplitude=0.05,
        craters=[
            CraterConfig(center=Vec3(1.0, 0.0, 0.0), radius=0.1),
            CraterConfig(center=Vec3(0.0, 1.0, 0.0), radius=0.15),
        ],
    )
    return PlanetSDF(config=config)


# =============================================================================
# Path A: SphericalCoord.from_cartesian basic conversion
# =============================================================================

class TestSphericalCoordFromCartesian:
    """Tests for cartesian to spherical conversion."""

    def test_unit_x_axis(self) -> None:
        """Point on +X axis should have theta=0, phi=0."""
        coord = SphericalCoord.from_cartesian(1.0, 0.0, 0.0)
        assert coord.r == pytest.approx(1.0, abs=1e-10)
        assert coord.theta == pytest.approx(0.0, abs=1e-10)
        assert coord.phi == pytest.approx(0.0, abs=1e-10)

    def test_unit_z_axis(self) -> None:
        """Point on +Z axis should have theta=pi/2, phi=0."""
        coord = SphericalCoord.from_cartesian(0.0, 0.0, 1.0)
        assert coord.r == pytest.approx(1.0, abs=1e-10)
        assert coord.theta == pytest.approx(HALF_PI, abs=1e-10)
        assert coord.phi == pytest.approx(0.0, abs=1e-10)

    def test_negative_x_axis(self) -> None:
        """Point on -X axis should have theta=pi or -pi."""
        coord = SphericalCoord.from_cartesian(-1.0, 0.0, 0.0)
        assert coord.r == pytest.approx(1.0, abs=1e-10)
        assert abs(coord.theta) == pytest.approx(PI, abs=1e-10)
        assert coord.phi == pytest.approx(0.0, abs=1e-10)

    def test_arbitrary_point(self) -> None:
        """Arbitrary point should have correct radius."""
        coord = SphericalCoord.from_cartesian(1.0, 2.0, 2.0)
        expected_r = math.sqrt(1 + 4 + 4)  # sqrt(9) = 3
        assert coord.r == pytest.approx(expected_r, abs=1e-10)


# =============================================================================
# Path B: SphericalCoord.from_cartesian at poles
# =============================================================================

class TestSphericalCoordPoles:
    """Tests for polar singularities."""

    def test_north_pole(self) -> None:
        """Point at +Y (north pole) should have phi=pi/2."""
        coord = SphericalCoord.from_cartesian(0.0, 1.0, 0.0)
        assert coord.r == pytest.approx(1.0, abs=1e-10)
        assert coord.phi == pytest.approx(HALF_PI, abs=1e-10)

    def test_south_pole(self) -> None:
        """Point at -Y (south pole) should have phi=-pi/2."""
        coord = SphericalCoord.from_cartesian(0.0, -1.0, 0.0)
        assert coord.r == pytest.approx(1.0, abs=1e-10)
        assert coord.phi == pytest.approx(-HALF_PI, abs=1e-10)

    def test_near_north_pole(self) -> None:
        """Point near north pole should have phi close to pi/2."""
        coord = SphericalCoord.from_cartesian(0.001, 0.9999995, 0.0)
        assert coord.phi > 0.99 * HALF_PI


# =============================================================================
# Path C: SphericalCoord.from_cartesian at origin singularity
# =============================================================================

class TestSphericalCoordOrigin:
    """Tests for origin singularity handling."""

    def test_exact_origin(self) -> None:
        """Origin should return r=0, arbitrary theta/phi."""
        coord = SphericalCoord.from_cartesian(0.0, 0.0, 0.0)
        assert coord.r == pytest.approx(0.0, abs=1e-10)
        # theta and phi are arbitrary at origin

    def test_near_origin(self) -> None:
        """Very small radius should not cause numerical issues."""
        coord = SphericalCoord.from_cartesian(1e-12, 1e-12, 1e-12)
        assert coord.r == pytest.approx(0.0, abs=1e-10)


# =============================================================================
# Path D: SphericalCoord.to_cartesian roundtrip
# =============================================================================

class TestSphericalCoordRoundtrip:
    """Tests for coordinate roundtrip conversion."""

    @pytest.mark.parametrize("x,y,z", [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0),
        (0.5, 0.5, 0.5),
        (0.1, 0.9, 0.2),
        (-0.3, -0.4, 0.5),
    ])
    def test_roundtrip_preserves_coordinates(self, x: float, y: float, z: float) -> None:
        """Converting to spherical and back should preserve coordinates."""
        coord = SphericalCoord.from_cartesian(x, y, z)
        x2, y2, z2 = coord.to_cartesian()
        assert x2 == pytest.approx(x, abs=1e-9)
        assert y2 == pytest.approx(y, abs=1e-9)
        assert z2 == pytest.approx(z, abs=1e-9)


# =============================================================================
# Path E: SphericalCoord.normalized_uv range [0, 1]
# =============================================================================

class TestSphericalCoordNormalizedUV:
    """Tests for normalized UV coordinate generation."""

    @pytest.mark.parametrize("theta,phi", [
        (0.0, 0.0),
        (PI, 0.0),
        (-PI, 0.0),
        (0.0, HALF_PI),
        (0.0, -HALF_PI),
        (PI / 4, PI / 4),
    ])
    def test_uv_in_unit_range(self, theta: float, phi: float) -> None:
        """UV coordinates should be in [0, 1] range."""
        coord = SphericalCoord(1.0, theta, phi)
        u, v = coord.normalized_uv()
        assert 0.0 <= u <= 1.0, f"u={u} out of range"
        assert 0.0 <= v <= 1.0, f"v={v} out of range"


# =============================================================================
# Path F: _hash21 determinism
# =============================================================================

class TestHash21Determinism:
    """Tests for 2D hash function determinism."""

    def test_same_input_same_output(self) -> None:
        """Same input should always produce same output."""
        h1 = _hash21(1.5, 2.5)
        h2 = _hash21(1.5, 2.5)
        assert h1 == h2

    def test_different_inputs_different_outputs(self) -> None:
        """Different inputs should (usually) produce different outputs."""
        h1 = _hash21(1.0, 2.0)
        h2 = _hash21(1.0, 2.001)
        assert h1 != h2


# =============================================================================
# Path G: _hash21 range [0, 1)
# =============================================================================

class TestHash21Range:
    """Tests for 2D hash function range."""

    @pytest.mark.parametrize("x,y", [
        (0.0, 0.0),
        (1.0, 1.0),
        (-1.0, -1.0),
        (100.0, 200.0),
        (-50.5, 75.25),
        (0.001, 0.999),
    ])
    def test_output_in_unit_range(self, x: float, y: float) -> None:
        """Hash output should be in [0, 1)."""
        h = _hash21(x, y)
        assert 0.0 <= h < 1.0, f"hash21({x}, {y}) = {h} out of range"


# =============================================================================
# Path H: _hash31 determinism
# =============================================================================

class TestHash31Determinism:
    """Tests for 3D hash function determinism."""

    def test_same_input_same_output(self) -> None:
        """Same input should always produce same output."""
        h1 = _hash31(1.5, 2.5, 3.5)
        h2 = _hash31(1.5, 2.5, 3.5)
        assert h1 == h2

    def test_different_z_different_output(self) -> None:
        """Changing z should change output."""
        h1 = _hash31(1.0, 2.0, 3.0)
        h2 = _hash31(1.0, 2.0, 3.001)
        assert h1 != h2


# =============================================================================
# Path I: _hash31 range [0, 1)
# =============================================================================

class TestHash31Range:
    """Tests for 3D hash function range."""

    @pytest.mark.parametrize("x,y,z", [
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (-1.0, -1.0, -1.0),
        (100.0, 200.0, 300.0),
        (-50.5, 75.25, -12.125),
    ])
    def test_output_in_unit_range(self, x: float, y: float, z: float) -> None:
        """Hash output should be in [0, 1)."""
        h = _hash31(x, y, z)
        assert 0.0 <= h < 1.0, f"hash31({x}, {y}, {z}) = {h} out of range"


# =============================================================================
# Path J: _value_noise_2d range [-1, 1]
# =============================================================================

class TestValueNoise2DRange:
    """Tests for 2D value noise range."""

    def test_output_range_at_grid_points(self) -> None:
        """Noise at integer grid points should be in [-1, 1]."""
        for i in range(10):
            for j in range(10):
                n = _value_noise_2d(float(i), float(j))
                assert -1.0 <= n <= 1.0

    def test_output_range_at_fractional_points(self) -> None:
        """Noise at fractional points should be in [-1, 1]."""
        for i in range(100):
            x = i * 0.37
            y = i * 0.41
            n = _value_noise_2d(x, y)
            assert -1.0 <= n <= 1.0


# =============================================================================
# Path K: _value_noise_2d continuity
# =============================================================================

class TestValueNoise2DContinuity:
    """Tests for 2D value noise continuity."""

    def test_small_step_small_change(self) -> None:
        """Small step should produce small change (Lipschitz continuity)."""
        x, y = 1.5, 2.5
        epsilon = 0.001
        n0 = _value_noise_2d(x, y)
        n1 = _value_noise_2d(x + epsilon, y)
        n2 = _value_noise_2d(x, y + epsilon)

        # Change should be proportional to step size
        assert abs(n1 - n0) < 0.1  # Reasonable continuity
        assert abs(n2 - n0) < 0.1


# =============================================================================
# Path L: _value_noise_3d range [-1, 1]
# =============================================================================

class TestValueNoise3DRange:
    """Tests for 3D value noise range."""

    def test_output_range_sampling(self) -> None:
        """Sample noise at various points, all should be in [-1, 1]."""
        for i in range(50):
            x = i * 0.37 - 10
            y = i * 0.41 - 10
            z = i * 0.29 - 10
            n = _value_noise_3d(x, y, z)
            assert -1.0 <= n <= 1.0, f"noise3d({x}, {y}, {z}) = {n} out of range"


# =============================================================================
# Path M: _value_noise_3d continuity
# =============================================================================

class TestValueNoise3DContinuity:
    """Tests for 3D value noise continuity."""

    def test_continuous_along_x(self) -> None:
        """Noise should be continuous along X axis."""
        y, z = 2.5, 3.5
        for i in range(10):
            x = i * 0.1
            n0 = _value_noise_3d(x, y, z)
            n1 = _value_noise_3d(x + 0.001, y, z)
            assert abs(n1 - n0) < 0.1


# =============================================================================
# Path N: _fbm_2d zero octaves returns 0
# =============================================================================

class TestFBM2DZeroOctaves:
    """Tests for FBM with zero octaves."""

    def test_zero_octaves_returns_zero(self) -> None:
        """Zero octaves should return 0.0."""
        result = _fbm_2d(1.5, 2.5, octaves=0, frequency=1.0, lacunarity=2.0, persistence=0.5)
        assert result == 0.0


# =============================================================================
# Path O: _fbm_2d single octave equals base noise
# =============================================================================

class TestFBM2DSingleOctave:
    """Tests for FBM with single octave."""

    def test_single_octave_is_scaled_noise(self) -> None:
        """Single octave FBM should equal base noise at given frequency."""
        x, y = 1.5, 2.5
        freq = 2.0
        fbm_val = _fbm_2d(x, y, octaves=1, frequency=freq, lacunarity=2.0, persistence=0.5)
        base_val = _value_noise_2d(x * freq, y * freq)
        assert fbm_val == pytest.approx(base_val, abs=1e-10)


# =============================================================================
# Path P: _fbm_2d amplitude normalization
# =============================================================================

class TestFBM2DAmplitudeNormalization:
    """Tests for FBM amplitude normalization."""

    def test_output_approximately_in_unit_range(self) -> None:
        """FBM output should be approximately in [-1, 1] after normalization."""
        for i in range(50):
            x = i * 0.37
            y = i * 0.41
            result = _fbm_2d(x, y, octaves=6, frequency=1.0, lacunarity=2.0, persistence=0.5)
            assert -1.5 <= result <= 1.5  # Some overshoot allowed


# =============================================================================
# Path Q: _fbm_3d determinism
# =============================================================================

class TestFBM3DDeterminism:
    """Tests for 3D FBM determinism."""

    def test_same_input_same_output(self) -> None:
        """Same inputs should produce same output."""
        r1 = _fbm_3d(1.5, 2.5, 3.5, octaves=4, frequency=1.0, lacunarity=2.0, persistence=0.5)
        r2 = _fbm_3d(1.5, 2.5, 3.5, octaves=4, frequency=1.0, lacunarity=2.0, persistence=0.5)
        assert r1 == r2


# =============================================================================
# Path R: _fbm_spherical no seams at theta = +/- pi
# =============================================================================

class TestFBMSphericalSeams:
    """Tests for spherical FBM seam handling."""

    def test_no_seam_at_theta_boundary(self) -> None:
        """FBM should be continuous across theta = +/- pi boundary."""
        phi = 0.0  # Equator
        seed = 42

        # Sample just before and after the seam
        n1 = _fbm_spherical(PI - 0.01, phi, 4, 2.0, 2.0, 0.5, seed)
        n2 = _fbm_spherical(-PI + 0.01, phi, 4, 2.0, 2.0, 0.5, seed)

        # Should be nearly equal (continuous)
        assert abs(n1 - n2) < 0.1, f"Seam discontinuity: {n1} vs {n2}"


# =============================================================================
# Path S: _fbm_spherical pole behavior
# =============================================================================

class TestFBMSphericalPoles:
    """Tests for spherical FBM at poles."""

    def test_north_pole_consistent(self) -> None:
        """All theta values at north pole should give similar results."""
        phi = HALF_PI - 0.001  # Near north pole
        seed = 42

        values = [_fbm_spherical(theta, phi, 4, 2.0, 2.0, 0.5, seed) for theta in [0, PI/2, PI, -PI/2]]

        # Values should be similar (converging at pole)
        mean = sum(values) / len(values)
        for v in values:
            assert abs(v - mean) < 0.2  # Reasonable convergence


# =============================================================================
# Path T: PlanetSDF._sample_terrain_height amplitude scaling
# =============================================================================

class TestPlanetTerrainAmplitude:
    """Tests for terrain amplitude scaling."""

    def test_zero_amplitude_flat_terrain(self) -> None:
        """Zero amplitude should produce zero terrain height."""
        planet = PlanetSDF(terrain_amplitude=0.0)
        height = planet._sample_terrain_height(0.0, 0.0)
        assert height == 0.0

    def test_amplitude_scales_height(self) -> None:
        """Terrain height should scale with amplitude."""
        # Use negative ocean level to avoid clipping
        config1 = PlanetConfig(terrain_amplitude=0.1, noise_seed=42, ocean_level=-1.0)
        config2 = PlanetConfig(terrain_amplitude=0.2, noise_seed=42, ocean_level=-1.0)
        planet1 = PlanetSDF(config=config1)
        planet2 = PlanetSDF(config=config2)

        # Sample at location that has non-zero terrain
        h1 = planet1._sample_terrain_height(1.5, 0.3)
        h2 = planet2._sample_terrain_height(1.5, 0.3)

        # h2 should be approximately 2x h1 (before ocean clipping)
        # The ratio should be close to 2.0
        if abs(h1) > 0.001:  # Only test if there's significant terrain
            ratio = abs(h2) / abs(h1)
            assert 1.5 < ratio < 2.5  # Should be approximately 2x


# =============================================================================
# Path U: PlanetSDF._sample_terrain_height ocean clipping
# =============================================================================

class TestPlanetOceanClipping:
    """Tests for ocean level clipping."""

    def test_height_clipped_to_ocean_level(self) -> None:
        """Terrain below ocean level should be clipped."""
        planet = PlanetSDF(
            terrain_amplitude=0.1,
            ocean_level=0.05,  # High ocean level
            noise_seed=42,
        )

        # Sample many points; all should be >= ocean level
        for i in range(20):
            theta = i * 0.3
            phi = i * 0.15 - 1.5
            height = planet._sample_terrain_height(theta, phi)
            assert height >= planet.ocean_level, f"Height {height} below ocean {planet.ocean_level}"


# =============================================================================
# Path V: PlanetSDF._sample_terrain_height mountain modulation
# =============================================================================

class TestPlanetMountainModulation:
    """Tests for mountain region modulation."""

    def test_mountain_scale_increases_amplitude(self) -> None:
        """Mountain regions should have increased amplitude."""
        config = PlanetConfig(
            terrain_amplitude=0.1,
            mountain_amplitude_scale=3.0,
            mountain_threshold=0.0,  # All terrain is "mountain"
        )
        planet = PlanetSDF(config=config)

        # Compare with planet without mountain scaling
        config2 = PlanetConfig(
            terrain_amplitude=0.1,
            mountain_amplitude_scale=1.0,
        )
        planet2 = PlanetSDF(config=config2)

        # Sample at same location - mountain scaled should be larger
        h1 = abs(planet._sample_terrain_height(1.0, 0.5))
        h2 = abs(planet2._sample_terrain_height(1.0, 0.5))

        # Mountain version should have larger magnitude
        assert h1 >= h2


# =============================================================================
# Path W: PlanetSDF._evaluate_crater distance calculation
# =============================================================================

class TestPlanetCraterDistance:
    """Tests for crater SDF evaluation."""

    def test_crater_center_inside(self) -> None:
        """Point at crater center should be inside crater sphere."""
        crater = CraterConfig(center=Vec3(1.0, 0.0, 0.0), radius=0.2)
        planet = PlanetSDF()

        # Evaluate at crater center
        sdf = planet._evaluate_crater(1.0, 0.0, 0.0, crater)
        assert sdf < 0  # Inside the crater sphere

    def test_crater_edge(self) -> None:
        """Point at crater edge should have SDF near zero."""
        crater = CraterConfig(center=Vec3(1.0, 0.0, 0.0), radius=0.2)
        planet = PlanetSDF()

        # Point at edge (1.2, 0, 0) is exactly 0.2 from center
        sdf = planet._evaluate_crater(1.2, 0.0, 0.0, crater)
        assert sdf == pytest.approx(0.0, abs=1e-10)


# =============================================================================
# Path X: PlanetSDF.evaluate center singularity
# =============================================================================

class TestPlanetCenterSingularity:
    """Tests for center singularity handling."""

    def test_origin_returns_negative_radius(self, default_planet: PlanetSDF) -> None:
        """Point at origin should return -radius (inside planet)."""
        sdf = default_planet.evaluate_xyz(0.0, 0.0, 0.0)
        assert sdf == pytest.approx(-default_planet.planet_radius, abs=1e-10)

    def test_near_origin_no_crash(self, default_planet: PlanetSDF) -> None:
        """Very small position should not cause numerical issues."""
        sdf = default_planet.evaluate_xyz(1e-15, 1e-15, 1e-15)
        assert math.isfinite(sdf)


# =============================================================================
# Path Y: PlanetSDF.evaluate radial displacement
# =============================================================================

class TestPlanetRadialDisplacement:
    """Tests for radial terrain displacement."""

    def test_flat_planet_is_sphere(self, flat_planet: PlanetSDF) -> None:
        """Planet with zero amplitude should be a perfect sphere."""
        # Test points at radius should have SDF near zero
        for angle in [0, PI/4, PI/2, PI, -PI/4]:
            x = math.cos(angle)
            z = math.sin(angle)
            sdf = flat_planet.evaluate_xyz(x, 0.0, z)
            assert sdf == pytest.approx(0.0, abs=1e-6)

    def test_point_far_outside(self, default_planet: PlanetSDF) -> None:
        """Point far from planet should have large positive SDF."""
        sdf = default_planet.evaluate_xyz(10.0, 0.0, 0.0)
        assert sdf > 8.0  # Far outside


# =============================================================================
# Path Z: PlanetSDF.evaluate_atmosphere shell distance
# =============================================================================

class TestPlanetAtmosphere:
    """Tests for atmosphere shell SDF."""

    def test_no_atmosphere_returns_inf(self, default_planet: PlanetSDF) -> None:
        """Planet without atmosphere should return infinity."""
        sdf = default_planet.evaluate_atmosphere(2.0, 0.0, 0.0)
        assert sdf == float('inf')

    def test_atmosphere_shell(self) -> None:
        """Atmosphere shell should have correct radius."""
        planet = PlanetSDF(
            planet_radius=1.0,
            terrain_amplitude=0.1,
            atmosphere_thickness=0.2,
        )
        atmo_radius = 1.0 + 0.1 + 0.2  # planet + terrain + atmo

        # Point at atmosphere boundary
        sdf = planet.evaluate_atmosphere(atmo_radius, 0.0, 0.0)
        assert sdf == pytest.approx(0.0, abs=1e-10)


# =============================================================================
# Path AA: PlanetConfig validation errors
# =============================================================================

class TestPlanetConfigValidation:
    """Tests for configuration validation."""

    def test_negative_radius_raises(self) -> None:
        """Negative planet radius should raise ValueError."""
        with pytest.raises(ValueError, match="radius must be positive"):
            PlanetConfig(planet_radius=-1.0)

    def test_negative_amplitude_raises(self) -> None:
        """Negative terrain amplitude should raise ValueError."""
        with pytest.raises(ValueError, match="amplitude must be non-negative"):
            PlanetConfig(terrain_amplitude=-0.1)

    def test_zero_octaves_raises(self) -> None:
        """Zero noise octaves should raise ValueError."""
        with pytest.raises(ValueError, match="octaves must be >= 1"):
            PlanetConfig(noise_octaves=0)

    def test_persistence_out_of_range_raises(self) -> None:
        """Persistence outside (0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="persistence must be in"):
            PlanetConfig(noise_persistence=0.0)
        with pytest.raises(ValueError, match="persistence must be in"):
            PlanetConfig(noise_persistence=1.5)


# =============================================================================
# Path AB: CraterConfig validation errors
# =============================================================================

class TestCraterConfigValidation:
    """Tests for crater configuration validation."""

    def test_negative_radius_raises(self) -> None:
        """Negative crater radius should raise ValueError."""
        with pytest.raises(ValueError, match="radius must be positive"):
            CraterConfig(center=Vec3(1, 0, 0), radius=-0.1)

    def test_depth_out_of_range_raises(self) -> None:
        """Depth outside [0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="depth must be in"):
            CraterConfig(center=Vec3(1, 0, 0), radius=0.1, depth=1.5)


# =============================================================================
# Path AC: PlanetSDF.add_crater and clear_craters
# =============================================================================

class TestPlanetCraterManagement:
    """Tests for crater management."""

    def test_add_crater(self) -> None:
        """Adding a crater should increase crater count."""
        planet = PlanetSDF()
        initial_count = len(planet.craters)

        planet.add_crater(CraterConfig(center=Vec3(1, 0, 0), radius=0.1))
        assert len(planet.craters) == initial_count + 1

    def test_clear_craters(self) -> None:
        """Clearing craters should remove all craters."""
        planet = PlanetSDF(crater_count=5)
        assert len(planet.craters) >= 5

        planet.clear_craters()
        assert len(planet.craters) == 0


# =============================================================================
# Path AD: PlanetSDF.is_ocean boundary detection
# =============================================================================

class TestPlanetOceanDetection:
    """Tests for ocean/land detection."""

    def test_is_ocean_detects_low_terrain(self) -> None:
        """is_ocean should return True for terrain below ocean level."""
        # Use ocean_level=0 which is the middle of noise range [-1, 1] * amplitude
        # This should give roughly 50% ocean coverage
        planet = PlanetSDF(
            terrain_amplitude=0.1,
            ocean_level=0.0,  # Middle of range
        )

        # Sample many points with varied coordinates
        ocean_count = sum(
            1 for i in range(100)
            if planet.is_ocean(i * 0.37, (i * 0.23) - 1.5)  # Better distribution
        )

        # Some should be ocean, some land (roughly 50/50 with ocean_level=0)
        assert 10 < ocean_count < 90


# =============================================================================
# Path AE: PlanetSDF.is_continent noise threshold
# =============================================================================

class TestPlanetContinentDetection:
    """Tests for continent detection."""

    def test_is_continent_binary(self) -> None:
        """is_continent should return boolean."""
        planet = PlanetSDF()
        result = planet.is_continent(0.5, 0.3)
        assert isinstance(result, bool)

    def test_continent_distribution(self) -> None:
        """Continent distribution should have some variety."""
        # Use config to set continent_threshold
        config = PlanetConfig(continent_threshold=0.0)
        planet = PlanetSDF(config=config)

        continent_count = sum(
            1 for i in range(100)
            if planet.is_continent(i * 0.37, (i * 0.23) - 1.5)  # Better distribution
        )

        # Should have mix of continent and ocean (roughly 50/50 with threshold=0)
        assert 10 < continent_count < 90


# =============================================================================
# Path AF: PlanetSDF.sample_normal gradient direction
# =============================================================================

class TestPlanetNormal:
    """Tests for surface normal calculation."""

    def test_normal_points_outward_on_sphere(self, flat_planet: PlanetSDF) -> None:
        """Normal on flat sphere should point radially outward."""
        # Test point on +X axis
        p = Vec3(1.0, 0.0, 0.0)
        normal = flat_planet.sample_normal(p)

        # Should point in +X direction
        assert normal.x > 0.9
        assert abs(normal.y) < 0.1
        assert abs(normal.z) < 0.1

    def test_normal_is_normalized(self, default_planet: PlanetSDF) -> None:
        """Surface normal should be unit length."""
        p = Vec3(1.0, 0.5, 0.3)
        normal = default_planet.sample_normal(p)
        length = normal.length()
        assert length == pytest.approx(1.0, abs=1e-6)


# =============================================================================
# Path AG: PlanetSDF.to_wgsl output contains required elements
# =============================================================================

class TestPlanetWGSLOutput:
    """Tests for WGSL code generation."""

    def test_contains_function_name(self, default_planet: PlanetSDF) -> None:
        """Generated WGSL should contain the function name."""
        wgsl = default_planet.to_wgsl("my_planet")
        assert "fn my_planet(" in wgsl

    def test_contains_constants(self, default_planet: PlanetSDF) -> None:
        """Generated WGSL should contain planet constants."""
        wgsl = default_planet.to_wgsl()
        assert "PLANET_RADIUS" in wgsl
        assert "TERRAIN_AMPLITUDE" in wgsl
        assert "NOISE_OCTAVES" in wgsl

    def test_contains_noise_functions(self, default_planet: PlanetSDF) -> None:
        """Generated WGSL should contain noise helper functions."""
        wgsl = default_planet.to_wgsl()
        assert "fn hash31(" in wgsl
        assert "fn value_noise_3d(" in wgsl
        assert "fn fbm_spherical(" in wgsl

    def test_contains_normal_function(self, default_planet: PlanetSDF) -> None:
        """Generated WGSL should contain normal calculation."""
        wgsl = default_planet.to_wgsl("sdf_planet")
        assert "fn sdf_planet_normal(" in wgsl

    def test_atmosphere_code_when_enabled(self) -> None:
        """Atmosphere code should be included when thickness > 0."""
        planet = PlanetSDF(atmosphere_thickness=0.2)
        wgsl = planet.to_wgsl()
        assert "ATMOSPHERE_RADIUS" in wgsl
        assert "_atmosphere(" in wgsl

    def test_crater_array_when_present(self, planet_with_craters: PlanetSDF) -> None:
        """Crater array should be included when craters exist."""
        wgsl = planet_with_craters.to_wgsl()
        assert "CRATERS:" in wgsl
        assert "CRATER_COUNT:" in wgsl


# =============================================================================
# Path AH: PlanetSDF dirty tracking
# =============================================================================

class TestPlanetDirtyTracking:
    """Tests for dirty flag tracking."""

    def test_initial_state_dirty(self) -> None:
        """New planet should have dirty flags."""
        planet = PlanetSDF()
        assert planet.tracker.is_dirty

    def test_clear_and_modify(self) -> None:
        """Modifying after clear should set dirty again."""
        planet = PlanetSDF()
        planet.tracker.clear()
        assert not planet.tracker.is_dirty

        planet.terrain_amplitude = 0.2
        assert planet.tracker.is_dirty


# =============================================================================
# Additional Edge Case Tests
# =============================================================================

class TestPlanetEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_large_coordinates(self, default_planet: PlanetSDF) -> None:
        """Should handle very large coordinates without overflow."""
        sdf = default_planet.evaluate_xyz(1e10, 1e10, 1e10)
        assert math.isfinite(sdf)
        assert sdf > 0  # Far outside

    def test_many_craters(self) -> None:
        """Should handle many craters efficiently."""
        planet = PlanetSDF(crater_count=50)
        assert len(planet.craters) == 50

        # Evaluation should still work
        sdf = planet.evaluate_xyz(1.5, 0.0, 0.0)
        assert math.isfinite(sdf)

    def test_clone_independence(self, default_planet: PlanetSDF) -> None:
        """Cloned planet should be independent of original."""
        clone = default_planet.clone()
        clone.terrain_amplitude = 0.5

        assert default_planet.terrain_amplitude != clone.terrain_amplitude

    def test_spherical_symmetry(self, flat_planet: PlanetSDF) -> None:
        """Flat planet should be spherically symmetric."""
        sdf1 = flat_planet.evaluate_xyz(1.0, 0.0, 0.0)
        sdf2 = flat_planet.evaluate_xyz(0.0, 1.0, 0.0)
        sdf3 = flat_planet.evaluate_xyz(0.0, 0.0, 1.0)

        assert sdf1 == pytest.approx(sdf2, abs=1e-6)
        assert sdf2 == pytest.approx(sdf3, abs=1e-6)
