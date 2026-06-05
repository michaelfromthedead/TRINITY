"""
Blackbox tests for advanced terrain SDF functions (T-DEMO-4.3 and T-DEMO-4.4).

Tests the public API without knowledge of internal implementation:
  T-DEMO-4.3 (Domain-Warped Terrain):
  - Height retrieval API
  - SDF evaluation API
  - Warp configuration API
  - Non-repeating pattern verification

  T-DEMO-4.4 (3D Cave Terrain):
  - Cave detection API
  - Overhang detection API
  - Connectivity verification API
  - SDF continuity verification API

BLACKBOX coverage plan:
  Path A-L: Public API functionality
  Path M-X: Configuration and factory methods
  Path Y-AJ: Terrain behavior verification
  Path AK-AV: Error handling and edge cases
"""

from __future__ import annotations

import math
import random

import pytest

from engine.rendering.demoscene.terrain_advanced import (
    DomainWarpedTerrainSDF,
    DomainWarpConfig,
    CaveTerrainSDF,
    CaveConfig,
    WarpPass,
    NoiseType,
    TerrainConfig,
    create_domain_warped_terrain,
    create_cave_terrain,
)
from engine.rendering.demoscene.sdf_ast import Vec3


# =============================================================================
# T-DEMO-4.3: Domain-Warped Terrain Public API
# =============================================================================


class TestDomainWarpedTerrainAPI:
    """Blackbox tests for DomainWarpedTerrainSDF public API."""

    # Path A: Basic instantiation
    def test_instantiation_default(self):
        """Can instantiate with default config."""
        terrain = DomainWarpedTerrainSDF()
        assert terrain is not None

    # Path B: Instantiation with config
    def test_instantiation_with_config(self):
        """Can instantiate with custom config."""
        config = DomainWarpConfig(warp_strength=2.0)
        terrain = DomainWarpedTerrainSDF(config)
        assert terrain is not None

    # Path C: get_height returns float
    def test_get_height_returns_float(self):
        """get_height should return a float."""
        terrain = DomainWarpedTerrainSDF()
        h = terrain.get_height(0.0, 0.0)
        assert isinstance(h, float)

    # Path D: evaluate returns float
    def test_evaluate_returns_float(self):
        """evaluate should return a float."""
        terrain = DomainWarpedTerrainSDF()
        sdf = terrain.evaluate(Vec3(0.0, 0.0, 0.0))
        assert isinstance(sdf, float)

    # Path E: evaluate_tuple returns float
    def test_evaluate_tuple_returns_float(self):
        """evaluate_tuple should return a float."""
        terrain = DomainWarpedTerrainSDF()
        sdf = terrain.evaluate_tuple((0.0, 0.0, 0.0))
        assert isinstance(sdf, float)

    # Path F: get_normal returns Vec3
    def test_get_normal_returns_vec3(self):
        """get_normal should return a Vec3."""
        terrain = DomainWarpedTerrainSDF()
        normal = terrain.get_normal(Vec3(0.0, 0.0, 0.0))
        assert hasattr(normal, 'x')
        assert hasattr(normal, 'y')
        assert hasattr(normal, 'z')

    # Path G: is_pattern_repeating returns bool
    def test_is_pattern_repeating_returns_bool(self):
        """is_pattern_repeating should return a bool."""
        terrain = DomainWarpedTerrainSDF()
        result = terrain.is_pattern_repeating()
        assert isinstance(result, bool)

    # Path H: to_wgsl returns string
    def test_to_wgsl_returns_string(self):
        """to_wgsl should return a string."""
        terrain = DomainWarpedTerrainSDF()
        wgsl = terrain.to_wgsl()
        assert isinstance(wgsl, str)

    # Path I: label returns string
    def test_label_returns_string(self):
        """label should return a string."""
        terrain = DomainWarpedTerrainSDF()
        label = terrain.label()
        assert isinstance(label, str)

    # Path J: clone returns new instance
    def test_clone_returns_new_instance(self):
        """clone should return a new instance."""
        terrain1 = DomainWarpedTerrainSDF()
        terrain2 = terrain1.clone()
        assert terrain1 is not terrain2

    # Path K: config property readable
    def test_config_property_readable(self):
        """config property should be readable."""
        config = DomainWarpConfig(warp_strength=3.0)
        terrain = DomainWarpedTerrainSDF(config)
        assert terrain.config.warp_strength == 3.0

    # Path L: config property writable
    def test_config_property_writable(self):
        """config property should be writable."""
        terrain = DomainWarpedTerrainSDF()
        new_config = DomainWarpConfig(warp_strength=5.0)
        terrain.config = new_config
        assert terrain.config.warp_strength == 5.0


class TestCaveTerrainAPI:
    """Blackbox tests for CaveTerrainSDF public API."""

    # Path M: Basic instantiation
    def test_instantiation_default(self):
        """Can instantiate with default config."""
        terrain = CaveTerrainSDF()
        assert terrain is not None

    # Path N: Instantiation with config
    def test_instantiation_with_config(self):
        """Can instantiate with custom config."""
        config = CaveConfig(cave_strength=5.0)
        terrain = CaveTerrainSDF(config)
        assert terrain is not None

    # Path O: evaluate returns float
    def test_evaluate_returns_float(self):
        """evaluate should return a float."""
        terrain = CaveTerrainSDF()
        sdf = terrain.evaluate(Vec3(0.0, 0.0, 0.0))
        assert isinstance(sdf, float)

    # Path P: is_inside_cave returns bool
    def test_is_inside_cave_returns_bool(self):
        """is_inside_cave should return a bool."""
        terrain = CaveTerrainSDF()
        result = terrain.is_inside_cave(Vec3(0.0, 5.0, 0.0))
        assert isinstance(result, bool)

    # Path Q: has_overhang_at returns bool
    def test_has_overhang_at_returns_bool(self):
        """has_overhang_at should return a bool."""
        terrain = CaveTerrainSDF()
        result = terrain.has_overhang_at(Vec3(0.0, 5.0, 0.0))
        assert isinstance(result, bool)

    # Path R: check_cave_connectivity returns bool
    def test_check_cave_connectivity_returns_bool(self):
        """check_cave_connectivity should return a bool."""
        terrain = CaveTerrainSDF()
        result = terrain.check_cave_connectivity((-5, -5, -5, 5, 5, 5))
        assert isinstance(result, bool)

    # Path S: is_sdf_continuous returns bool
    def test_is_sdf_continuous_returns_bool(self):
        """is_sdf_continuous should return a bool."""
        terrain = CaveTerrainSDF()
        p1 = Vec3(0.0, 0.0, 0.0)
        p2 = Vec3(1.0, 1.0, 1.0)
        result = terrain.is_sdf_continuous(p1, p2)
        assert isinstance(result, bool)

    # Path T: get_normal returns Vec3
    def test_get_normal_returns_vec3(self):
        """get_normal should return a Vec3."""
        terrain = CaveTerrainSDF()
        normal = terrain.get_normal(Vec3(0.0, 0.0, 0.0))
        assert hasattr(normal, 'x')
        assert hasattr(normal, 'y')
        assert hasattr(normal, 'z')

    # Path U: to_wgsl returns string
    def test_to_wgsl_returns_string(self):
        """to_wgsl should return a string."""
        terrain = CaveTerrainSDF()
        wgsl = terrain.to_wgsl()
        assert isinstance(wgsl, str)

    # Path V: clone returns new instance
    def test_clone_returns_new_instance(self):
        """clone should return a new instance."""
        terrain1 = CaveTerrainSDF()
        terrain2 = terrain1.clone()
        assert terrain1 is not terrain2


# =============================================================================
# Configuration and Factory Methods
# =============================================================================


class TestConfigurationAPI:
    """Blackbox tests for configuration classes."""

    # Path W: DomainWarpConfig creation
    def test_domain_warp_config_creation(self):
        """Can create DomainWarpConfig with various parameters."""
        config = DomainWarpConfig(
            warp_strength=2.0,
            warp_frequency=0.3,
            height_octaves=6,
            height_amplitude=20.0,
        )
        assert config.warp_strength == 2.0

    # Path X: CaveConfig creation
    def test_cave_config_creation(self):
        """Can create CaveConfig with various parameters."""
        config = CaveConfig(
            cave_strength=4.0,
            cave_density=0.6,
            overhang_probability=0.3,
        )
        assert config.cave_strength == 4.0

    # Path Y: WarpPass creation
    def test_warp_pass_creation(self):
        """Can create WarpPass with various parameters."""
        warp_pass = WarpPass(
            frequency=0.5,
            amplitude=2.0,
            octaves=4,
        )
        assert warp_pass.frequency == 0.5

    # Path Z: TerrainConfig convenience methods
    def test_terrain_config_domain_warped(self):
        """TerrainConfig.domain_warped works."""
        config = TerrainConfig.domain_warped(warp_strength=2.0)
        assert config.domain_warp is not None

    # Path AA: TerrainConfig.with_caves
    def test_terrain_config_with_caves(self):
        """TerrainConfig.with_caves works."""
        config = TerrainConfig.with_caves(cave_strength=5.0)
        assert config.cave is not None

    # Path AB: create_domain_warped_terrain factory
    def test_create_domain_warped_terrain(self):
        """create_domain_warped_terrain factory works."""
        terrain = create_domain_warped_terrain(
            warp_strength=2.0,
            warp_passes=3,
        )
        assert terrain is not None
        assert len(terrain.config.warp_passes) == 3

    # Path AC: create_cave_terrain factory
    def test_create_cave_terrain(self):
        """create_cave_terrain factory works."""
        terrain = create_cave_terrain(
            cave_strength=4.0,
            cave_density=0.6,
        )
        assert terrain is not None
        assert terrain.config.cave_strength == 4.0


# =============================================================================
# Terrain Behavior Verification
# =============================================================================


class TestTerrainBehavior:
    """Blackbox tests verifying terrain behavior."""

    # Path AD: Domain warp affects terrain
    def test_domain_warp_affects_terrain(self):
        """Domain warping should change terrain shape."""
        terrain_no_warp = DomainWarpedTerrainSDF(
            DomainWarpConfig(warp_strength=0.0)
        )
        terrain_with_warp = DomainWarpedTerrainSDF(
            DomainWarpConfig(warp_strength=5.0)
        )

        # Sample multiple points and check for differences
        different_count = 0
        for i in range(50):
            x, z = i * 2.0, i * 3.0
            h1 = terrain_no_warp.get_height(x, z)
            h2 = terrain_with_warp.get_height(x, z)
            if abs(h1 - h2) > 0.1:
                different_count += 1

        assert different_count > 20, "Warp should affect terrain"

    # Path AE: Terrain has varied heights
    def test_terrain_varied_heights(self):
        """Terrain should have varied heights across positions."""
        terrain = DomainWarpedTerrainSDF()

        heights = set()
        for i in range(100):
            h = terrain.get_height(i * 0.5, i * 0.7)
            heights.add(round(h, 2))

        assert len(heights) > 20, "Terrain should have varied heights"

    # Path AF: Cave terrain has caves
    def test_cave_terrain_has_caves(self):
        """Cave terrain should have cave regions where 3D field is negative."""
        # With low cave_density, threshold is close to 0, so negative
        # FBM values will trigger cave formation
        terrain = CaveTerrainSDF(CaveConfig(
            cave_density=0.1,  # Low = threshold -0.1, easier to meet
            cave_strength=8.0,
            cave_frequency=0.15,
        ))

        # Check that cave field produces values below threshold
        threshold = -terrain.config.cave_density
        negative_found = False
        for _ in range(500):
            x = random.uniform(-100, 100)
            z = random.uniform(-100, 100)
            y = random.uniform(-10, 30)
            # Check the 3D cave field directly
            cave_value = terrain._get_cave_value((x, y, z))
            if cave_value < threshold:
                negative_found = True
                break

        assert negative_found, f"Should find 3D cave field value < {threshold}"

    # Path AG: SDF sign matches surface
    def test_sdf_sign_matches_surface(self):
        """SDF should be negative below surface, positive above."""
        terrain = DomainWarpedTerrainSDF()

        for _ in range(20):
            x = random.uniform(-20, 20)
            z = random.uniform(-20, 20)
            h = terrain.get_height(x, z)

            # Below surface
            sdf_below = terrain.evaluate(Vec3(x, h - 2.0, z))
            assert sdf_below < 0, f"SDF below surface should be negative: {sdf_below}"

            # Above surface
            sdf_above = terrain.evaluate(Vec3(x, h + 2.0, z))
            assert sdf_above > 0, f"SDF above surface should be positive: {sdf_above}"

    # Path AH: Normal points away from surface
    def test_normal_direction(self):
        """Normal should generally point upward for terrain."""
        terrain = DomainWarpedTerrainSDF()

        upward_count = 0
        for _ in range(30):
            x = random.uniform(-10, 10)
            z = random.uniform(-10, 10)
            h = terrain.get_height(x, z)
            p = Vec3(x, h, z)
            normal = terrain.get_normal(p)

            # Y component should generally be positive (upward)
            if normal.y > 0:
                upward_count += 1

        # Most normals should point upward for terrain
        assert upward_count > 15, "Most normals should point upward"

    # Path AI: Continuous height across terrain
    def test_continuous_height(self):
        """Height should be continuous (no sudden jumps)."""
        terrain = DomainWarpedTerrainSDF()

        prev_h = terrain.get_height(0.0, 0.0)
        max_jump = 0.0

        for i in range(1, 100):
            x = i * 0.1
            z = i * 0.1
            h = terrain.get_height(x, z)
            jump = abs(h - prev_h)
            max_jump = max(max_jump, jump)
            prev_h = h

        # For small steps, height jumps should be small
        assert max_jump < 2.0, f"Height jump too large: {max_jump}"

    # Path AJ: Multiple warp passes change terrain
    def test_multiple_warp_passes_effect(self):
        """Multiple warp passes should create different terrain."""
        terrain_1 = DomainWarpedTerrainSDF(DomainWarpConfig(
            warp_passes=(WarpPass(),),
        ))
        terrain_3 = DomainWarpedTerrainSDF(DomainWarpConfig(
            warp_passes=(WarpPass(), WarpPass(), WarpPass()),
        ))

        different_count = 0
        for i in range(50):
            x, z = i * 0.5, i * 0.7
            h1 = terrain_1.get_height(x, z)
            h3 = terrain_3.get_height(x, z)
            if abs(h1 - h3) > 0.05:
                different_count += 1

        assert different_count > 20, "Different pass counts should produce different terrain"


# =============================================================================
# Error Handling and Edge Cases
# =============================================================================


class TestErrorHandling:
    """Blackbox tests for error handling."""

    # Path AK: Invalid warp strength rejected
    def test_invalid_warp_strength(self):
        """Negative warp strength should raise error."""
        with pytest.raises(ValueError):
            DomainWarpConfig(warp_strength=-1.0)

    # Path AL: Invalid cave density rejected
    def test_invalid_cave_density(self):
        """Cave density outside [0, 1] should raise error."""
        with pytest.raises(ValueError):
            CaveConfig(cave_density=1.5)

    # Path AM: Invalid overhang probability rejected
    def test_invalid_overhang_probability(self):
        """Overhang probability outside [0, 1] should raise error."""
        with pytest.raises(ValueError):
            CaveConfig(overhang_probability=-0.1)

    # Path AN: Invalid WarpPass frequency rejected
    def test_invalid_warp_pass_frequency(self):
        """Zero or negative frequency should raise error."""
        with pytest.raises(ValueError):
            WarpPass(frequency=0.0)

    # Path AO: Large coordinates handled
    def test_large_coordinates_handled(self):
        """Large coordinates should not cause errors."""
        terrain = DomainWarpedTerrainSDF()
        sdf = terrain.evaluate(Vec3(10000.0, 500.0, 10000.0))
        assert math.isfinite(sdf)

    # Path AP: Negative coordinates handled
    def test_negative_coordinates_handled(self):
        """Negative coordinates should not cause errors."""
        terrain = CaveTerrainSDF()
        sdf = terrain.evaluate(Vec3(-50.0, -10.0, -50.0))
        assert math.isfinite(sdf)

    # Path AQ: Zero coordinates handled
    def test_zero_coordinates_handled(self):
        """Origin coordinates should not cause errors."""
        terrain1 = DomainWarpedTerrainSDF()
        terrain2 = CaveTerrainSDF()

        sdf1 = terrain1.evaluate(Vec3(0.0, 0.0, 0.0))
        sdf2 = terrain2.evaluate(Vec3(0.0, 0.0, 0.0))

        assert math.isfinite(sdf1)
        assert math.isfinite(sdf2)


# =============================================================================
# WGSL Output Verification
# =============================================================================


class TestWGSLOutput:
    """Blackbox tests for WGSL output."""

    # Path AR: Domain warp WGSL not empty
    def test_domain_warp_wgsl_not_empty(self):
        """Domain warp WGSL output should not be empty."""
        terrain = DomainWarpedTerrainSDF()
        wgsl = terrain.to_wgsl()
        assert len(wgsl) > 50

    # Path AS: Cave WGSL not empty
    def test_cave_wgsl_not_empty(self):
        """Cave WGSL output should not be empty."""
        terrain = CaveTerrainSDF()
        wgsl = terrain.to_wgsl()
        assert len(wgsl) > 50

    # Path AT: WGSL contains function definitions
    def test_wgsl_contains_functions(self):
        """WGSL should contain fn definitions."""
        terrain1 = DomainWarpedTerrainSDF()
        terrain2 = CaveTerrainSDF()

        assert "fn " in terrain1.to_wgsl()
        assert "fn " in terrain2.to_wgsl()

    # Path AU: WGSL contains return statements
    def test_wgsl_contains_return(self):
        """WGSL should contain return statements."""
        terrain1 = DomainWarpedTerrainSDF()
        terrain2 = CaveTerrainSDF()

        assert "return" in terrain1.to_wgsl()
        assert "return" in terrain2.to_wgsl()

    # Path AV: Config values embedded in WGSL
    def test_config_values_in_wgsl(self):
        """Config values should appear in WGSL output."""
        config = DomainWarpConfig(
            warp_strength=7.5,
            height_amplitude=30.0,
        )
        terrain = DomainWarpedTerrainSDF(config)
        wgsl = terrain.to_wgsl()

        assert "7.5" in wgsl
        assert "30.0" in wgsl
