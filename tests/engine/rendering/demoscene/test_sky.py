"""
Tests for sky.py (T-DEMO-3.10).

Tests sky color functions including:
  - Gradient sky interpolation (horizon to zenith)
  - Solid color mode
  - Triple gradient with below-horizon color
  - Procedural sky with sun disc
  - WGSL code generation

Acceptance Criteria:
  - Gradient sky correctly interpolates horizon to zenith
  - Solid color mode works
  - Miss rays receive sky color
  - 15+ tests covering sky variations
"""

from __future__ import annotations

import math
import pytest

from engine.rendering.demoscene.sky import (
    SkyMode,
    Vec3,
    SkyConfig,
    SkySettingsNode,
    sky_solid,
    sky_gradient,
    sky_gradient_triple,
    sky_procedural,
    generate_sky_wgsl,
    create_sunset_sky,
    create_daytime_sky,
    create_night_sky,
)
from engine.rendering.demoscene.ast_nodes import Vec3Node, FloatNode


# =============================================================================
# Vec3 Helper Tests
# =============================================================================


class TestVec3:
    """Tests for Vec3 helper class."""

    def test_creation(self):
        """Basic Vec3 creation."""
        v = Vec3(1.0, 2.0, 3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_from_tuple(self):
        """Create from tuple."""
        v = Vec3.from_tuple((1.0, 2.0, 3.0))
        assert v.as_tuple() == (1.0, 2.0, 3.0)

    def test_from_node(self):
        """Create from Vec3Node."""
        node = Vec3Node(1.0, 2.0, 3.0)
        v = Vec3.from_node(node)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_lerp_at_0(self):
        """Lerp at t=0 returns self."""
        a = Vec3(1.0, 0.0, 0.0)
        b = Vec3(0.0, 1.0, 0.0)
        result = a.lerp(b, 0.0)
        assert abs(result.x - 1.0) < 1e-6
        assert abs(result.y - 0.0) < 1e-6

    def test_lerp_at_1(self):
        """Lerp at t=1 returns other."""
        a = Vec3(1.0, 0.0, 0.0)
        b = Vec3(0.0, 1.0, 0.0)
        result = a.lerp(b, 1.0)
        assert abs(result.x - 0.0) < 1e-6
        assert abs(result.y - 1.0) < 1e-6

    def test_lerp_at_half(self):
        """Lerp at t=0.5 returns midpoint."""
        a = Vec3(0.0, 0.0, 0.0)
        b = Vec3(1.0, 1.0, 1.0)
        result = a.lerp(b, 0.5)
        assert abs(result.x - 0.5) < 1e-6
        assert abs(result.y - 0.5) < 1e-6
        assert abs(result.z - 0.5) < 1e-6

    def test_clamp_in_range(self):
        """Clamp values already in range."""
        v = Vec3(0.5, 0.5, 0.5)
        result = v.clamp()
        assert result.x == 0.5
        assert result.y == 0.5
        assert result.z == 0.5

    def test_clamp_below(self):
        """Clamp values below range."""
        v = Vec3(-0.5, -1.0, -2.0)
        result = v.clamp()
        assert result.x == 0.0
        assert result.y == 0.0
        assert result.z == 0.0

    def test_clamp_above(self):
        """Clamp values above range."""
        v = Vec3(1.5, 2.0, 3.0)
        result = v.clamp()
        assert result.x == 1.0
        assert result.y == 1.0
        assert result.z == 1.0


# =============================================================================
# Sky Solid Tests
# =============================================================================


class TestSkySolid:
    """Tests for solid color sky."""

    def test_solid_returns_color(self):
        """Solid sky returns the solid color."""
        direction = Vec3(0.0, 1.0, 0.0)
        color = Vec3(0.5, 0.5, 0.5)
        result = sky_solid(direction, color)
        assert result.x == 0.5
        assert result.y == 0.5
        assert result.z == 0.5

    def test_solid_ignores_direction(self):
        """Direction doesn't affect solid color."""
        color = Vec3(1.0, 0.0, 0.0)
        r1 = sky_solid(Vec3(0.0, 1.0, 0.0), color)
        r2 = sky_solid(Vec3(0.0, -1.0, 0.0), color)
        r3 = sky_solid(Vec3(1.0, 0.0, 0.0), color)
        assert r1.as_tuple() == r2.as_tuple() == r3.as_tuple()


# =============================================================================
# Sky Gradient Tests
# =============================================================================


class TestSkyGradient:
    """Tests for two-color gradient sky."""

    def test_gradient_at_horizon(self):
        """Y=0 should return horizon color."""
        direction = Vec3(1.0, 0.0, 0.0).normalized() if hasattr(Vec3, 'normalized') else Vec3(1.0, 0.0, 0.0)
        # With y=0, should get horizon color
        horizon = Vec3(1.0, 0.0, 0.0)
        zenith = Vec3(0.0, 0.0, 1.0)
        result = sky_gradient(Vec3(1.0, 0.0, 0.0), horizon, zenith)
        assert abs(result.x - 1.0) < 1e-6
        assert abs(result.y - 0.0) < 1e-6
        assert abs(result.z - 0.0) < 1e-6

    def test_gradient_at_zenith(self):
        """Y=1 should return zenith color."""
        direction = Vec3(0.0, 1.0, 0.0)
        horizon = Vec3(1.0, 0.0, 0.0)
        zenith = Vec3(0.0, 0.0, 1.0)
        result = sky_gradient(direction, horizon, zenith)
        assert abs(result.x - 0.0) < 1e-6
        assert abs(result.y - 0.0) < 1e-6
        assert abs(result.z - 1.0) < 1e-6

    def test_gradient_below_horizon(self):
        """Y<0 should clamp to horizon color."""
        direction = Vec3(0.0, -1.0, 0.0)
        horizon = Vec3(1.0, 0.5, 0.0)
        zenith = Vec3(0.0, 0.0, 1.0)
        result = sky_gradient(direction, horizon, zenith)
        # max(0, -1) = 0, so should get horizon
        assert abs(result.x - 1.0) < 1e-6
        assert abs(result.y - 0.5) < 1e-6
        assert abs(result.z - 0.0) < 1e-6

    def test_gradient_half(self):
        """Y=0.5 should interpolate halfway."""
        direction = Vec3(0.0, 0.5, 0.866)  # normalized roughly
        horizon = Vec3(0.0, 0.0, 0.0)
        zenith = Vec3(1.0, 1.0, 1.0)
        result = sky_gradient(direction, horizon, zenith)
        # t = 0.5
        assert abs(result.x - 0.5) < 1e-6
        assert abs(result.y - 0.5) < 1e-6
        assert abs(result.z - 0.5) < 1e-6


# =============================================================================
# Sky Gradient Triple Tests
# =============================================================================


class TestSkyGradientTriple:
    """Tests for three-color gradient sky."""

    def test_triple_above_horizon(self):
        """Y > 0 should interpolate horizon to zenith."""
        direction = Vec3(0.0, 1.0, 0.0)
        below = Vec3(0.1, 0.0, 0.0)
        horizon = Vec3(0.5, 0.5, 0.0)
        zenith = Vec3(0.0, 0.0, 1.0)
        result = sky_gradient_triple(direction, below, horizon, zenith)
        # Y=1, so should get zenith
        assert abs(result.z - 1.0) < 1e-6

    def test_triple_at_horizon(self):
        """Y=0 should return horizon color."""
        direction = Vec3(1.0, 0.0, 0.0)
        below = Vec3(0.1, 0.0, 0.0)
        horizon = Vec3(0.5, 0.5, 0.0)
        zenith = Vec3(0.0, 0.0, 1.0)
        result = sky_gradient_triple(direction, below, horizon, zenith)
        assert abs(result.x - 0.5) < 1e-6
        assert abs(result.y - 0.5) < 1e-6

    def test_triple_below_horizon(self):
        """Y < 0 should interpolate below to horizon."""
        direction = Vec3(0.0, -1.0, 0.0)
        below = Vec3(0.0, 0.0, 0.0)
        horizon = Vec3(1.0, 1.0, 1.0)
        zenith = Vec3(0.0, 0.0, 1.0)
        result = sky_gradient_triple(direction, below, horizon, zenith)
        # Y=-1 maps to t=0, so should get below color
        assert abs(result.x - 0.0) < 1e-6

    def test_triple_mid_below_horizon(self):
        """Y=-0.5 should interpolate halfway below."""
        direction = Vec3(0.0, -0.5, 0.866)
        below = Vec3(0.0, 0.0, 0.0)
        horizon = Vec3(1.0, 1.0, 1.0)
        zenith = Vec3(0.0, 0.0, 1.0)
        result = sky_gradient_triple(direction, below, horizon, zenith)
        # Y=-0.5, t = -0.5 + 1 = 0.5
        assert abs(result.x - 0.5) < 1e-6


# =============================================================================
# Sky Procedural Tests
# =============================================================================


class TestSkyProcedural:
    """Tests for procedural sky with sun."""

    def test_procedural_base_gradient(self):
        """Base gradient should be applied."""
        direction = Vec3(0.0, 1.0, 0.0)  # Straight up
        base = Vec3(0.5, 0.5, 0.0)
        sun_color = Vec3(1.0, 0.0, 0.0)
        sun_dir = Vec3(-1.0, 0.0, 0.0)  # Away from direction
        result = sky_procedural(direction, 0.0, base, sun_color, sun_dir)
        # Zenith should be darker blue (top color is 0.1, 0.2, 0.4)
        # Since sun is opposite direction, no sun contribution
        assert result.z > 0  # Has some blue

    def test_procedural_sun_contribution(self):
        """Looking at sun should add sun color."""
        sun_dir = Vec3(0.0, 1.0, 0.0)  # Normalized sun direction
        direction = sun_dir  # Looking directly at sun
        base = Vec3(0.0, 0.0, 0.0)
        sun_color = Vec3(1.0, 1.0, 1.0)
        result = sky_procedural(direction, 0.0, base, sun_color, sun_dir)
        # Should have sun contribution (dot = 1)
        # intensity = 1^64 = 1
        # Result should be clamped to 1.0
        assert result.x > 0.5


# =============================================================================
# SkyConfig Tests
# =============================================================================


class TestSkyConfig:
    """Tests for SkyConfig class."""

    def test_default_mode(self):
        """Default mode is gradient."""
        config = SkyConfig()
        assert config.mode == SkyMode.GRADIENT

    def test_evaluate_solid(self):
        """Evaluate solid mode."""
        config = SkyConfig(
            mode=SkyMode.SOLID,
            solid_color=(0.5, 0.5, 0.5),
        )
        result = config.evaluate(Vec3(0.0, 1.0, 0.0))
        assert abs(result.x - 0.5) < 1e-6

    def test_evaluate_gradient(self):
        """Evaluate gradient mode."""
        config = SkyConfig(
            mode=SkyMode.GRADIENT,
            horizon_color=(1.0, 0.0, 0.0),
            zenith_color=(0.0, 0.0, 1.0),
        )
        # Looking up
        result = config.evaluate(Vec3(0.0, 1.0, 0.0))
        assert result.z > result.x  # More blue at zenith

    def test_evaluate_with_sun(self):
        """Evaluate gradient with sun enabled."""
        config = SkyConfig(
            mode=SkyMode.GRADIENT,
            sun_enabled=True,
            sun_direction=(0.0, 1.0, 0.0),
            sun_color=(1.0, 1.0, 1.0),
            sun_power=4.0,  # Lower power for visible effect
        )
        # Looking at sun
        result = config.evaluate(Vec3(0.0, 1.0, 0.0))
        # Should have sun contribution
        assert result.x > 0


# =============================================================================
# WGSL Generation Tests
# =============================================================================


class TestWGSLGeneration:
    """Tests for WGSL code generation."""

    def test_generate_solid(self):
        """Generate solid sky WGSL."""
        config = SkyConfig(
            mode=SkyMode.SOLID,
            solid_color=(0.1, 0.2, 0.3),
        )
        wgsl = config.generate_sky_wgsl()
        assert "fn sky_color" in wgsl
        assert "0.1" in wgsl
        assert "0.2" in wgsl
        assert "0.3" in wgsl

    def test_generate_gradient(self):
        """Generate gradient sky WGSL."""
        config = SkyConfig(
            mode=SkyMode.GRADIENT,
            horizon_color=(0.8, 0.6, 0.4),
            zenith_color=(0.1, 0.2, 0.5),
        )
        wgsl = config.generate_sky_wgsl()
        assert "fn sky_color" in wgsl
        assert "max(0.0, direction.y)" in wgsl
        assert "mix" in wgsl

    def test_generate_gradient_with_sun(self):
        """Generate gradient with sun WGSL."""
        config = SkyConfig(
            mode=SkyMode.GRADIENT,
            sun_enabled=True,
            sun_direction=(0.5, 0.7, 0.5),
            sun_power=64.0,
        )
        wgsl = config.generate_sky_wgsl()
        assert "sun_dir" in wgsl
        assert "dot(direction, sun_dir)" in wgsl
        assert "64.0" in wgsl

    def test_generate_triple_gradient(self):
        """Generate triple gradient WGSL."""
        config = SkyConfig(
            mode=SkyMode.GRADIENT_TRIPLE,
            below_horizon_color=(0.05, 0.05, 0.08),
            horizon_color=(0.8, 0.6, 0.4),
            zenith_color=(0.1, 0.2, 0.5),
        )
        wgsl = config.generate_sky_wgsl()
        assert "below" in wgsl
        assert "direction.y < 0.0" in wgsl

    def test_generate_procedural(self):
        """Generate procedural sky WGSL."""
        config = SkyConfig(mode=SkyMode.PROCEDURAL)
        wgsl = config.generate_sky_wgsl()
        assert "sun_dir" in wgsl
        assert "pow" in wgsl

    def test_generate_custom_function_name(self):
        """Custom function name."""
        config = SkyConfig(mode=SkyMode.SOLID)
        wgsl = config.generate_sky_wgsl(function_name="get_background")
        assert "fn get_background" in wgsl

    def test_convenience_function(self):
        """Test generate_sky_wgsl convenience function."""
        wgsl = generate_sky_wgsl(
            mode=SkyMode.GRADIENT,
            horizon_color=(0.8, 0.6, 0.4),
            zenith_color=(0.1, 0.2, 0.5),
        )
        assert "fn sky_color" in wgsl
        assert "0.8" in wgsl


# =============================================================================
# Preset Tests
# =============================================================================


class TestPresets:
    """Tests for sky presets."""

    def test_sunset_preset(self):
        """Sunset sky preset."""
        config = create_sunset_sky()
        assert config.mode == SkyMode.GRADIENT_TRIPLE
        assert config.sun_enabled
        # Should have warm horizon
        assert config.horizon_color[0] > 0.5  # Red-ish

    def test_daytime_preset(self):
        """Daytime sky preset."""
        config = create_daytime_sky()
        assert config.mode == SkyMode.GRADIENT
        assert config.sun_enabled
        # Should have blue zenith
        assert config.zenith_color[2] > 0.5

    def test_night_preset(self):
        """Night sky preset."""
        config = create_night_sky()
        assert config.mode == SkyMode.GRADIENT
        assert not config.sun_enabled
        # Should be dark
        assert max(config.horizon_color) < 0.1

    def test_presets_generate_wgsl(self):
        """All presets should generate valid WGSL."""
        for preset_fn in [create_sunset_sky, create_daytime_sky, create_night_sky]:
            config = preset_fn()
            wgsl = config.generate_sky_wgsl()
            assert "fn sky_color" in wgsl


# =============================================================================
# SkySettingsNode Tests
# =============================================================================


class TestSkySettingsNode:
    """Tests for SkySettingsNode integration."""

    def test_default_values(self):
        """Default SkySettingsNode values."""
        node = SkySettingsNode()
        assert node.mode == SkyMode.GRADIENT
        assert not node.sun_enabled

    def test_to_sky_config(self):
        """Convert to SkyConfig."""
        node = SkySettingsNode(
            mode=SkyMode.SOLID,
            solid_color=Vec3Node(0.5, 0.5, 0.5),
        )
        config = node.to_sky_config()
        assert config.mode == SkyMode.SOLID
        assert config.solid_color == (0.5, 0.5, 0.5)

    def test_custom_sun_settings(self):
        """Custom sun settings."""
        node = SkySettingsNode(
            sun_enabled=True,
            sun_power=FloatNode(32.0),
            sun_direction=Vec3Node(0.5, 0.8, 0.3),
        )
        config = node.to_sky_config()
        assert config.sun_enabled
        assert config.sun_power == 32.0
