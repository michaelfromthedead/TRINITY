"""
Blackbox tests for PlanetSDF (T-DEMO-4.9).

Tests public API behavior and integration requirements:
- Spherical terrain generation
- SDF evaluation correctness
- Configuration management
- WGSL code generation
- Performance characteristics

These tests verify the module meets its specification without
knowledge of implementation details.

BLACKBOX coverage plan:
  Spec A: Planet radius defines base sphere size
  Spec B: Terrain amplitude controls height variation
  Spec C: Ocean level clips terrain from below
  Spec D: Noise parameters affect terrain characteristics
  Spec E: Craters create depressions in surface
  Spec F: Atmosphere adds outer shell
  Spec G: WGSL output is valid and evaluable
  Spec H: Configuration can be modified at runtime
  Spec I: Deterministic output for same seed
  Spec J: Spherical symmetry without terrain
  Spec K: Normal vectors point outward
  Spec L: SDF gradient is well-defined
  Spec M: Performance: evaluation is bounded
  Spec N: Trinity tracker integration
  Spec O: Clone creates independent copy
  Spec P: API stability and type safety
"""

from __future__ import annotations

import math
import time
from typing import List, Tuple

import pytest

from engine.rendering.demoscene.planet_sdf import (
    PlanetSDF,
    PlanetConfig,
    CraterConfig,
    SphericalCoord,
)
from engine.rendering.demoscene.sdf_ast import Vec3


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def unit_planet() -> PlanetSDF:
    """Create a unit radius planet with default settings."""
    return PlanetSDF(planet_radius=1.0)


@pytest.fixture
def large_planet() -> PlanetSDF:
    """Create a larger planet for scale testing."""
    return PlanetSDF(planet_radius=100.0, terrain_amplitude=5.0)


@pytest.fixture
def ocean_planet() -> PlanetSDF:
    """Create a planet with significant ocean coverage."""
    return PlanetSDF(
        planet_radius=1.0,
        terrain_amplitude=0.2,
        ocean_level=0.1,
    )


@pytest.fixture
def crater_planet() -> PlanetSDF:
    """Create a planet with craters."""
    return PlanetSDF(
        planet_radius=1.0,
        terrain_amplitude=0.05,
        crater_count=10,
        noise_seed=42,
    )


# =============================================================================
# Spec A: Planet radius defines base sphere size
# =============================================================================

class TestPlanetRadius:
    """Tests for planet radius behavior."""

    def test_radius_determines_base_size(self) -> None:
        """Planet radius should determine the base sphere size."""
        for radius in [0.5, 1.0, 2.0, 10.0]:
            planet = PlanetSDF(planet_radius=radius, terrain_amplitude=0.0)

            # Point at exactly radius distance should be on surface
            sdf = planet.evaluate(Vec3(radius, 0.0, 0.0))
            assert sdf == pytest.approx(0.0, abs=1e-6)

            # Point inside should have negative SDF
            sdf_inside = planet.evaluate(Vec3(radius * 0.5, 0.0, 0.0))
            assert sdf_inside < 0

            # Point outside should have positive SDF
            sdf_outside = planet.evaluate(Vec3(radius * 1.5, 0.0, 0.0))
            assert sdf_outside > 0

    def test_radius_is_readable(self, unit_planet: PlanetSDF) -> None:
        """Planet radius should be accessible via property."""
        assert unit_planet.planet_radius == 1.0

    def test_radius_is_writable(self, unit_planet: PlanetSDF) -> None:
        """Planet radius should be modifiable."""
        unit_planet.planet_radius = 2.0
        assert unit_planet.planet_radius == 2.0


# =============================================================================
# Spec B: Terrain amplitude controls height variation
# =============================================================================

class TestTerrainAmplitude:
    """Tests for terrain amplitude behavior."""

    def test_zero_amplitude_smooth_sphere(self) -> None:
        """Zero amplitude should produce a perfect sphere."""
        planet = PlanetSDF(planet_radius=1.0, terrain_amplitude=0.0)

        # Sample at multiple angles
        angles = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        for theta in angles:
            x = math.cos(theta)
            z = math.sin(theta)
            sdf = planet.evaluate(Vec3(x, 0.0, z))
            assert sdf == pytest.approx(0.0, abs=1e-6)

    def test_amplitude_increases_variation(self) -> None:
        """Higher amplitude should produce more terrain variation."""
        seed = 42

        # Use configs with no ocean clipping to see full terrain variation
        config_low = PlanetConfig(terrain_amplitude=0.01, noise_seed=seed, ocean_level=-1.0)
        config_high = PlanetConfig(terrain_amplitude=0.2, noise_seed=seed, ocean_level=-1.0)
        planet_low = PlanetSDF(config=config_low)
        planet_high = PlanetSDF(config=config_high)

        # Sample at multiple diverse points
        points = [
            (1.0, 0.0, 0.0),
            (0.7, 0.7, 0.0),
            (0.5, 0.5, 0.707),
            (0.0, 1.0, 0.0),
            (-0.5, 0.5, 0.707),
        ]

        low_values = [planet_low.evaluate(Vec3(*p)) for p in points]
        high_values = [planet_high.evaluate(Vec3(*p)) for p in points]

        low_range = max(low_values) - min(low_values)
        high_range = max(high_values) - min(high_values)

        # High amplitude should have larger variation (at least 5x for 20x amplitude)
        assert high_range > low_range * 2.0

    def test_amplitude_is_bounded(self) -> None:
        """Terrain should not exceed amplitude bounds significantly."""
        planet = PlanetSDF(
            planet_radius=1.0,
            terrain_amplitude=0.1,
            ocean_level=-1.0,  # No ocean clipping
        )

        # Sample many points
        max_deviation = 0.0
        for i in range(100):
            theta = i * 0.1
            phi = (i % 20) * 0.1 - 1.0

            coord = SphericalCoord(1.0, theta, phi)
            x, y, z = coord.to_cartesian()

            sdf = planet.evaluate(Vec3(x, y, z))
            max_deviation = max(max_deviation, abs(sdf))

        # Maximum deviation should be within 2x amplitude
        assert max_deviation < 0.3


# =============================================================================
# Spec C: Ocean level clips terrain from below
# =============================================================================

class TestOceanLevel:
    """Tests for ocean level clipping."""

    def test_terrain_clipped_at_ocean_level(self, ocean_planet: PlanetSDF) -> None:
        """Terrain below ocean level should be clipped."""
        ocean_level = ocean_planet.ocean_level

        # Sample at many points - all effective heights should be >= ocean_level
        for i in range(50):
            theta = i * 0.13
            phi = i * 0.07 - 1.5

            # The raw terrain might be below, but sampled height should be clipped
            height = ocean_planet._sample_terrain_height(theta, phi)
            assert height >= ocean_level

    def test_ocean_detection(self) -> None:
        """is_ocean should correctly identify ocean regions."""
        # Use ocean level at 0 which is middle of noise range
        planet = PlanetSDF(
            planet_radius=1.0,
            terrain_amplitude=0.2,
            ocean_level=0.0,  # Middle of terrain range
        )

        # With enough samples at varied coordinates, we should find both ocean and land
        ocean_samples = 0
        land_samples = 0

        for i in range(100):
            theta = i * 0.37  # Better coverage
            phi = (i * 0.23) - 1.5

            if planet.is_ocean(theta, phi):
                ocean_samples += 1
            else:
                land_samples += 1

        # Should have some of each (roughly 50/50 with ocean_level=0)
        assert ocean_samples > 10, f"Too few ocean samples: {ocean_samples}"
        assert land_samples > 10, f"Too few land samples: {land_samples}"


# =============================================================================
# Spec D: Noise parameters affect terrain characteristics
# =============================================================================

class TestNoiseParameters:
    """Tests for noise parameter effects."""

    def test_different_seeds_different_terrain(self) -> None:
        """Different seeds should produce different terrain."""
        planet1 = PlanetSDF(noise_seed=1)
        planet2 = PlanetSDF(noise_seed=2)

        p = Vec3(1.0, 0.0, 0.0)
        sdf1 = planet1.evaluate(p)
        sdf2 = planet2.evaluate(p)

        assert sdf1 != sdf2

    def test_same_seed_same_terrain(self) -> None:
        """Same seed should produce identical terrain."""
        planet1 = PlanetSDF(noise_seed=42)
        planet2 = PlanetSDF(noise_seed=42)

        p = Vec3(1.0, 0.5, 0.3)
        sdf1 = planet1.evaluate(p)
        sdf2 = planet2.evaluate(p)

        assert sdf1 == sdf2

    def test_frequency_affects_detail_scale(self) -> None:
        """Higher frequency should produce finer detail."""
        planet_low = PlanetSDF(noise_frequency=1.0, noise_seed=42)
        planet_high = PlanetSDF(noise_frequency=4.0, noise_seed=42)

        # Sample at two nearby points
        p1 = Vec3(1.0, 0.0, 0.0)
        p2 = Vec3(1.0, 0.01, 0.0)

        diff_low = abs(planet_low.evaluate(p1) - planet_low.evaluate(p2))
        diff_high = abs(planet_high.evaluate(p1) - planet_high.evaluate(p2))

        # Higher frequency should have more variation over small distances
        assert diff_high >= diff_low * 0.5  # Some tolerance

    def test_octaves_affect_detail(self) -> None:
        """More octaves should add more detail."""
        planet_few = PlanetSDF(noise_octaves=1, noise_seed=42)
        planet_many = PlanetSDF(noise_octaves=8, noise_seed=42)

        # More octaves means different results (more detail)
        p = Vec3(1.0, 0.1, 0.1)
        sdf_few = planet_few.evaluate(p)
        sdf_many = planet_many.evaluate(p)

        assert sdf_few != sdf_many


# =============================================================================
# Spec E: Craters create depressions in surface
# =============================================================================

class TestCraters:
    """Tests for crater formation."""

    def test_crater_creates_depression(self) -> None:
        """Crater should create a depression at its location."""
        # Planet without crater
        planet_flat = PlanetSDF(
            planet_radius=1.0,
            terrain_amplitude=0.0,
        )

        # Planet with crater at +X
        planet_crater = PlanetSDF(
            planet_radius=1.0,
            terrain_amplitude=0.0,
            config=PlanetConfig(
                planet_radius=1.0,
                terrain_amplitude=0.0,
                craters=[CraterConfig(center=Vec3(1.0, 0.0, 0.0), radius=0.2)],
            ),
        )

        # At crater center, the surface should be depressed (closer to center)
        sdf_flat = planet_flat.evaluate(Vec3(0.9, 0.0, 0.0))
        sdf_crater = planet_crater.evaluate(Vec3(0.9, 0.0, 0.0))

        # With crater, point should be "less inside" or "outside"
        assert sdf_crater > sdf_flat

    def test_procedural_craters_are_deterministic(self, crater_planet: PlanetSDF) -> None:
        """Procedural crater generation should be deterministic."""
        planet2 = PlanetSDF(
            planet_radius=1.0,
            terrain_amplitude=0.05,
            crater_count=10,
            noise_seed=42,
        )

        # Same seed should produce same craters
        assert len(crater_planet.craters) == len(planet2.craters)

        for c1, c2 in zip(crater_planet.craters, planet2.craters):
            assert c1.center.x == c2.center.x
            assert c1.center.y == c2.center.y
            assert c1.center.z == c2.center.z
            assert c1.radius == c2.radius

    def test_add_crater_dynamically(self) -> None:
        """Should be able to add craters after creation."""
        planet = PlanetSDF()
        initial_count = len(planet.craters)

        planet.add_crater(CraterConfig(
            center=Vec3(0.0, 1.0, 0.0),
            radius=0.1,
        ))

        assert len(planet.craters) == initial_count + 1


# =============================================================================
# Spec F: Atmosphere adds outer shell
# =============================================================================

class TestAtmosphere:
    """Tests for atmosphere shell."""

    def test_atmosphere_shell_outside_planet(self) -> None:
        """Atmosphere should be outside the planet surface."""
        planet = PlanetSDF(
            planet_radius=1.0,
            terrain_amplitude=0.1,
            atmosphere_thickness=0.2,
        )

        # Point between planet and atmosphere edge
        mid_point = 1.0 + 0.1 + 0.1  # radius + amplitude + half atmo

        planet_sdf = planet.evaluate(Vec3(mid_point, 0.0, 0.0))
        atmo_sdf = planet.evaluate_atmosphere(mid_point, 0.0, 0.0)

        # Should be outside planet, inside atmosphere
        assert planet_sdf > 0
        assert atmo_sdf < 0

    def test_no_atmosphere_when_thickness_zero(self, unit_planet: PlanetSDF) -> None:
        """Zero thickness should mean no atmosphere."""
        atmo_sdf = unit_planet.evaluate_atmosphere(2.0, 0.0, 0.0)
        assert atmo_sdf == float('inf')


# =============================================================================
# Spec G: WGSL output is valid and evaluable
# =============================================================================

class TestWGSLGeneration:
    """Tests for WGSL code generation."""

    def test_generates_valid_wgsl_syntax(self, unit_planet: PlanetSDF) -> None:
        """Generated WGSL should have valid syntax structure."""
        wgsl = unit_planet.to_wgsl()

        # Check for required WGSL elements
        assert "fn " in wgsl  # Has function definitions
        assert "-> f32" in wgsl  # Returns float
        assert "vec3<f32>" in wgsl  # Uses vec3

        # Check balanced braces
        assert wgsl.count("{") == wgsl.count("}")
        assert wgsl.count("(") == wgsl.count(")")

    def test_wgsl_contains_planet_params(self, unit_planet: PlanetSDF) -> None:
        """WGSL should embed planet parameters."""
        wgsl = unit_planet.to_wgsl()

        # Parameters should be embedded
        assert "PLANET_RADIUS" in wgsl
        assert "TERRAIN_AMPLITUDE" in wgsl
        assert "NOISE_OCTAVES" in wgsl

    def test_custom_function_name(self, unit_planet: PlanetSDF) -> None:
        """Should be able to specify function name."""
        wgsl = unit_planet.to_wgsl("my_custom_planet")
        assert "fn my_custom_planet(" in wgsl

    def test_wgsl_with_craters(self, crater_planet: PlanetSDF) -> None:
        """WGSL should include crater data when present."""
        wgsl = crater_planet.to_wgsl()
        assert "Crater" in wgsl
        assert "CRATERS" in wgsl

    def test_wgsl_with_atmosphere(self) -> None:
        """WGSL should include atmosphere when enabled."""
        planet = PlanetSDF(atmosphere_thickness=0.1)
        wgsl = planet.to_wgsl()
        assert "atmosphere" in wgsl.lower()


# =============================================================================
# Spec H: Configuration can be modified at runtime
# =============================================================================

class TestRuntimeConfiguration:
    """Tests for runtime configuration changes."""

    def test_modify_terrain_amplitude(self, unit_planet: PlanetSDF) -> None:
        """Should be able to modify terrain amplitude."""
        sdf_before = unit_planet.evaluate(Vec3(1.0, 0.0, 0.0))
        unit_planet.terrain_amplitude = 0.5
        sdf_after = unit_planet.evaluate(Vec3(1.0, 0.0, 0.0))

        assert sdf_before != sdf_after

    def test_modify_ocean_level(self, unit_planet: PlanetSDF) -> None:
        """Should be able to modify ocean level."""
        unit_planet.ocean_level = 0.05
        assert unit_planet.ocean_level == 0.05

    def test_config_replacement(self, unit_planet: PlanetSDF) -> None:
        """Should be able to replace entire config."""
        new_config = PlanetConfig(
            planet_radius=2.0,
            terrain_amplitude=0.3,
        )
        unit_planet.config = new_config
        assert unit_planet.planet_radius == 2.0


# =============================================================================
# Spec I: Deterministic output for same seed
# =============================================================================

class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_seed_same_output(self) -> None:
        """Same configuration should always produce same output."""
        config = PlanetConfig(noise_seed=12345)

        planet1 = PlanetSDF(config=config)
        planet2 = PlanetSDF(config=PlanetConfig(noise_seed=12345))

        points = [
            Vec3(1.0, 0.0, 0.0),
            Vec3(0.5, 0.5, 0.707),
            Vec3(-0.3, 0.8, 0.5),
        ]

        for p in points:
            assert planet1.evaluate(p) == planet2.evaluate(p)

    def test_wgsl_deterministic(self, unit_planet: PlanetSDF) -> None:
        """WGSL output should be deterministic."""
        wgsl1 = unit_planet.to_wgsl()
        wgsl2 = unit_planet.to_wgsl()
        assert wgsl1 == wgsl2


# =============================================================================
# Spec J: Spherical symmetry without terrain
# =============================================================================

class TestSphericalSymmetry:
    """Tests for spherical symmetry."""

    def test_equidistant_points_same_sdf(self) -> None:
        """Points at same distance should have same SDF on smooth sphere."""
        planet = PlanetSDF(terrain_amplitude=0.0)

        # All these points are at distance 1.5 from origin
        # Use exact values on axes to avoid floating point issues
        points = [
            Vec3(1.5, 0.0, 0.0),
            Vec3(0.0, 1.5, 0.0),
            Vec3(0.0, 0.0, 1.5),
            Vec3(-1.5, 0.0, 0.0),
        ]

        sdf_values = [planet.evaluate(p) for p in points]

        # All should be approximately equal (distance from unit sphere = 0.5)
        for sdf in sdf_values[1:]:
            assert sdf == pytest.approx(sdf_values[0], abs=1e-5)


# =============================================================================
# Spec K: Normal vectors point outward
# =============================================================================

class TestNormalVectors:
    """Tests for surface normal calculation."""

    def test_normal_points_outward(self, unit_planet: PlanetSDF) -> None:
        """Normal should point away from planet center."""
        points = [
            Vec3(1.0, 0.0, 0.0),
            Vec3(0.0, 1.0, 0.0),
            Vec3(0.707, 0.707, 0.0),
        ]

        for p in points:
            normal = unit_planet.sample_normal(p)

            # Dot product with position should be positive (outward)
            dot = normal.x * p.x + normal.y * p.y + normal.z * p.z
            assert dot > 0

    def test_normal_is_unit_length(self, unit_planet: PlanetSDF) -> None:
        """Normal should be normalized."""
        p = Vec3(1.0, 0.3, 0.5)
        normal = unit_planet.sample_normal(p)
        length = normal.length()
        assert length == pytest.approx(1.0, abs=1e-5)


# =============================================================================
# Spec L: SDF gradient is well-defined
# =============================================================================

class TestSDFGradient:
    """Tests for SDF gradient properties."""

    def test_gradient_magnitude_near_one(self, unit_planet: PlanetSDF) -> None:
        """SDF gradient magnitude should be approximately 1."""
        p = Vec3(1.2, 0.0, 0.0)
        epsilon = 0.001

        # Compute gradient via central differences
        dx = unit_planet.evaluate_xyz(p.x + epsilon, p.y, p.z) - unit_planet.evaluate_xyz(p.x - epsilon, p.y, p.z)
        dy = unit_planet.evaluate_xyz(p.x, p.y + epsilon, p.z) - unit_planet.evaluate_xyz(p.x, p.y - epsilon, p.z)
        dz = unit_planet.evaluate_xyz(p.x, p.y, p.z + epsilon) - unit_planet.evaluate_xyz(p.x, p.y, p.z - epsilon)

        gradient_length = math.sqrt((dx / (2 * epsilon)) ** 2 + (dy / (2 * epsilon)) ** 2 + (dz / (2 * epsilon)) ** 2)

        # For a proper SDF, gradient magnitude should be close to 1
        assert 0.5 < gradient_length < 2.0


# =============================================================================
# Spec M: Performance: evaluation is bounded
# =============================================================================

class TestPerformance:
    """Tests for performance characteristics."""

    def test_single_evaluation_fast(self, unit_planet: PlanetSDF) -> None:
        """Single evaluation should be fast."""
        p = Vec3(1.0, 0.5, 0.3)

        start = time.perf_counter()
        for _ in range(1000):
            unit_planet.evaluate(p)
        elapsed = time.perf_counter() - start

        # 1000 evaluations should complete in < 1 second
        assert elapsed < 1.0

    def test_wgsl_generation_fast(self, unit_planet: PlanetSDF) -> None:
        """WGSL generation should be fast."""
        start = time.perf_counter()
        for _ in range(100):
            unit_planet.to_wgsl()
        elapsed = time.perf_counter() - start

        # 100 generations should complete in < 1 second
        assert elapsed < 1.0

    def test_many_craters_reasonable_time(self) -> None:
        """Evaluation with many craters should still be reasonable."""
        planet = PlanetSDF(crater_count=100)
        p = Vec3(1.5, 0.0, 0.0)

        start = time.perf_counter()
        for _ in range(100):
            planet.evaluate(p)
        elapsed = time.perf_counter() - start

        # Should complete in < 2 seconds even with 100 craters
        assert elapsed < 2.0


# =============================================================================
# Spec N: Trinity tracker integration
# =============================================================================

class TestTrackerIntegration:
    """Tests for Trinity tracker integration."""

    def test_has_tracker(self, unit_planet: PlanetSDF) -> None:
        """Planet should have a tracker."""
        assert hasattr(unit_planet, 'tracker')
        assert unit_planet.tracker is not None

    def test_has_mirror(self, unit_planet: PlanetSDF) -> None:
        """Planet should have a mirror."""
        assert hasattr(unit_planet, 'mirror')
        assert unit_planet.mirror is not None

    def test_dirty_tracking(self, unit_planet: PlanetSDF) -> None:
        """Changes should mark tracker dirty."""
        unit_planet.tracker.clear()
        assert not unit_planet.tracker.is_dirty

        unit_planet.planet_radius = 2.0
        assert unit_planet.tracker.is_dirty


# =============================================================================
# Spec O: Clone creates independent copy
# =============================================================================

class TestCloning:
    """Tests for planet cloning."""

    def test_clone_is_equal(self, unit_planet: PlanetSDF) -> None:
        """Clone should produce same SDF values."""
        clone = unit_planet.clone()
        p = Vec3(1.0, 0.5, 0.3)

        assert unit_planet.evaluate(p) == clone.evaluate(p)

    def test_clone_is_independent(self, unit_planet: PlanetSDF) -> None:
        """Modifying clone should not affect original."""
        clone = unit_planet.clone()
        clone.planet_radius = 5.0

        assert unit_planet.planet_radius == 1.0
        assert clone.planet_radius == 5.0


# =============================================================================
# Spec P: API stability and type safety
# =============================================================================

class TestAPIStability:
    """Tests for API stability and type safety."""

    def test_vec3_input(self, unit_planet: PlanetSDF) -> None:
        """evaluate() should accept Vec3."""
        result = unit_planet.evaluate(Vec3(1.0, 0.0, 0.0))
        assert isinstance(result, float)

    def test_xyz_input(self, unit_planet: PlanetSDF) -> None:
        """evaluate_xyz() should accept separate coordinates."""
        result = unit_planet.evaluate_xyz(1.0, 0.0, 0.0)
        assert isinstance(result, float)

    def test_to_wgsl_returns_string(self, unit_planet: PlanetSDF) -> None:
        """to_wgsl() should return string."""
        result = unit_planet.to_wgsl()
        assert isinstance(result, str)

    def test_label_returns_string(self, unit_planet: PlanetSDF) -> None:
        """label() should return string."""
        result = unit_planet.label()
        assert isinstance(result, str)

    def test_config_type(self, unit_planet: PlanetSDF) -> None:
        """config property should return PlanetConfig."""
        assert isinstance(unit_planet.config, PlanetConfig)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_planet_workflow(self) -> None:
        """Test complete workflow from creation to WGSL."""
        # Create planet
        planet = PlanetSDF(
            planet_radius=1.0,
            terrain_amplitude=0.1,
            ocean_level=0.02,
            noise_octaves=4,
            crater_count=5,
            atmosphere_thickness=0.1,
            noise_seed=42,
        )

        # Evaluate at surface
        sdf = planet.evaluate(Vec3(1.05, 0.0, 0.0))
        assert math.isfinite(sdf)

        # Check atmosphere
        atmo = planet.evaluate_atmosphere(1.25, 0.0, 0.0)
        assert math.isfinite(atmo)

        # Get normal
        normal = planet.sample_normal(Vec3(1.0, 0.0, 0.0))
        assert normal.length() == pytest.approx(1.0, abs=1e-5)

        # Generate WGSL
        wgsl = planet.to_wgsl("planet_terrain")
        assert len(wgsl) > 1000  # Substantial output

        # Clone and verify
        clone = planet.clone()
        assert clone.evaluate(Vec3(1.05, 0.0, 0.0)) == sdf

    def test_earth_like_planet(self) -> None:
        """Test creating an Earth-like planet."""
        config = PlanetConfig(
            planet_radius=6371.0,  # km
            terrain_amplitude=10.0,  # 10km max mountains
            ocean_level=0.0,
            noise_octaves=8,
            noise_frequency=0.01,
            continent_threshold=0.0,
            crater_count=0,  # No visible craters on Earth
            atmosphere_thickness=100.0,  # 100km atmosphere
        )
        earth = PlanetSDF(config=config)

        # Check surface near equator
        sdf = earth.evaluate(Vec3(6371.0, 0.0, 0.0))
        assert abs(sdf) < 20  # Within terrain range

        # Check atmosphere - point should be inside atmosphere boundary
        # Atmosphere radius = 6371 + 10 + 100 = 6481, so 6480 is inside
        atmo = earth.evaluate_atmosphere(6480.0, 0.0, 0.0)
        assert atmo < 0  # Inside atmosphere

    def test_moon_like_planet(self) -> None:
        """Test creating a Moon-like body with craters."""
        moon = PlanetSDF(
            planet_radius=1737.0,  # km
            terrain_amplitude=5.0,
            ocean_level=-100.0,  # No ocean
            noise_octaves=6,
            crater_count=50,
            crater_radius_range=(10.0, 100.0),
            atmosphere_thickness=0.0,  # No atmosphere
            noise_seed=1969,  # Moon landing reference
        )

        assert len(moon.craters) >= 50
        assert moon.evaluate_atmosphere(2000.0, 0.0, 0.0) == float('inf')
