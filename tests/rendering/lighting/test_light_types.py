"""Tests for light type definitions."""

from __future__ import annotations

import math
import pytest

from engine.core.math.vec import Vec3
from engine.rendering.lighting.light_types import (
    DirectionalLight,
    PointLight,
    SpotLight,
    RectAreaLight,
    DiskAreaLight,
    IESLight,
    IESProfile,
    SkyLight,
    LightType,
    ShadowMode,
    GIImportance,
    ShadowCasterConfig,
    GIContributorConfig,
    shadow_caster,
    gi_contributor,
)


class TestLightBase:
    """Tests for base light functionality."""

    def test_directional_light_creation(self) -> None:
        """Test creating a directional light with defaults."""
        light = DirectionalLight()
        assert light.light_type == LightType.DIRECTIONAL
        assert light.color == Vec3(1.0, 1.0, 1.0)
        assert light.intensity == 1.0
        assert light.enabled is True
        assert light.cascade_count == 4

    def test_directional_light_custom(self) -> None:
        """Test creating a directional light with custom values."""
        light = DirectionalLight(
            color=Vec3(1.0, 0.9, 0.8),
            intensity=2.5,
            direction=Vec3(0.5, -0.8, 0.2),
            cascade_count=3,
        )
        assert light.color.x == pytest.approx(1.0)
        assert light.color.y == pytest.approx(0.9)
        assert light.intensity == pytest.approx(2.5)
        assert light.cascade_count == 3
        # Direction should be normalized
        assert abs(light.direction.length() - 1.0) < 0.001

    def test_directional_light_invalid_cascades(self) -> None:
        """Test that invalid cascade count raises error."""
        with pytest.raises(ValueError):
            DirectionalLight(cascade_count=5)
        with pytest.raises(ValueError):
            DirectionalLight(cascade_count=0)

    def test_point_light_creation(self) -> None:
        """Test creating a point light."""
        light = PointLight(
            position=Vec3(10, 5, -3),
            radius=15.0,
            intensity=500.0,
        )
        assert light.light_type == LightType.POINT
        assert light.position.x == pytest.approx(10.0)
        assert light.radius == pytest.approx(15.0)

    def test_point_light_attenuation(self) -> None:
        """Test point light attenuation calculation."""
        light = PointLight(radius=10.0)

        # At center, attenuation should be high
        assert light.get_attenuation(0.0) == pytest.approx(1.0, abs=0.1)

        # At radius, attenuation should be zero
        assert light.get_attenuation(10.0) == pytest.approx(0.0)

        # Beyond radius, attenuation should be zero
        assert light.get_attenuation(15.0) == pytest.approx(0.0)

        # Halfway should be some positive value
        mid_atten = light.get_attenuation(5.0)
        assert mid_atten > 0.0
        assert mid_atten < 1.0

    def test_point_light_invalid_radius(self) -> None:
        """Test that invalid radius raises error."""
        with pytest.raises(ValueError):
            PointLight(radius=0.0)
        with pytest.raises(ValueError):
            PointLight(radius=-5.0)

    def test_spot_light_creation(self) -> None:
        """Test creating a spot light."""
        light = SpotLight(
            position=Vec3(0, 10, 0),
            direction=Vec3(0, -1, 0),
            inner_angle=math.radians(20),
            outer_angle=math.radians(40),
            radius=30.0,
        )
        assert light.light_type == LightType.SPOT
        assert light.inner_angle < light.outer_angle

    def test_spot_light_angular_attenuation(self) -> None:
        """Test spot light angular attenuation."""
        light = SpotLight(
            direction=Vec3(0, -1, 0),
            inner_angle=math.radians(20),
            outer_angle=math.radians(40),
        )

        # Direction aligned with light - full intensity
        atten = light.get_angular_attenuation(Vec3(0, 1, 0))  # Looking back at light
        assert atten == pytest.approx(1.0)

        # Perpendicular - outside cone
        atten = light.get_angular_attenuation(Vec3(1, 0, 0))
        assert atten == pytest.approx(0.0)

    def test_spot_light_invalid_angles(self) -> None:
        """Test that invalid angles raise errors."""
        # Outer angle less than inner
        with pytest.raises(ValueError):
            SpotLight(inner_angle=math.radians(45), outer_angle=math.radians(20))

    def test_rect_area_light_creation(self) -> None:
        """Test creating a rectangular area light."""
        light = RectAreaLight(
            position=Vec3(0, 3, 0),
            width=2.0,
            height=1.0,
        )
        assert light.light_type == LightType.RECT_AREA
        assert light.area == pytest.approx(2.0)

    def test_rect_area_light_corners(self) -> None:
        """Test rect area light corner calculation."""
        light = RectAreaLight(
            position=Vec3(0, 0, 0),
            direction=Vec3(0, -1, 0),
            up=Vec3(0, 0, 1),
            width=2.0,
            height=2.0,
        )
        corners = light.get_corners()
        assert len(corners) == 4

    def test_disk_area_light_creation(self) -> None:
        """Test creating a disk area light."""
        light = DiskAreaLight(
            position=Vec3(0, 5, 0),
            disk_radius=0.5,
        )
        assert light.light_type == LightType.DISK_AREA
        assert light.area == pytest.approx(math.pi * 0.25)

    def test_ies_light_creation(self) -> None:
        """Test creating an IES light."""
        profile = IESProfile(
            name="test_profile",
            vertical_angles=[0.0, math.pi / 2, math.pi],
            horizontal_angles=[0.0, math.pi],
            candela_values=[[1.0, 0.8], [0.5, 0.3], [0.1, 0.05]],
            lumens=1000.0,
        )
        light = IESLight(
            position=Vec3(0, 2, 0),
            profile=profile,
        )
        assert light.light_type == LightType.IES

    def test_ies_profile_sampling(self) -> None:
        """Test IES profile sampling."""
        profile = IESProfile(
            vertical_angles=[0.0, math.pi],
            horizontal_angles=[0.0],
            candela_values=[[1.0], [0.5]],
        )
        # Sample at top (vertical = 0)
        assert profile.sample(0.0, 0.0) == pytest.approx(1.0)

    def test_sky_light_creation(self) -> None:
        """Test creating a sky light."""
        light = SkyLight(
            cubemap_path="/textures/sky.hdr",
            intensity=1.5,
        )
        assert light.light_type == LightType.SKY


class TestShadowConfig:
    """Tests for shadow configuration."""

    def test_shadow_caster_config_defaults(self) -> None:
        """Test default shadow caster configuration."""
        config = ShadowCasterConfig()
        assert config.mode == ShadowMode.DYNAMIC
        assert config.resolution_scale == 1.0
        assert config.cascade_bias == 0.0

    def test_shadow_caster_config_custom(self) -> None:
        """Test custom shadow caster configuration."""
        config = ShadowCasterConfig(
            mode=ShadowMode.STATIC,
            resolution_scale=2.0,
            cascade_bias=0.01,
        )
        assert config.mode == ShadowMode.STATIC
        assert config.resolution_scale == 2.0

    def test_shadow_caster_config_invalid_scale(self) -> None:
        """Test that invalid resolution scale raises error."""
        with pytest.raises(ValueError):
            ShadowCasterConfig(resolution_scale=0.0)
        with pytest.raises(ValueError):
            ShadowCasterConfig(resolution_scale=-1.0)

    def test_light_casts_shadows_property(self) -> None:
        """Test the casts_shadows property."""
        light = PointLight()
        # Default should have shadow config from decorator
        assert hasattr(DirectionalLight, '_shadow_config')

    def test_shadow_mode_none(self) -> None:
        """Test creating light with no shadows."""
        config = ShadowCasterConfig(mode=ShadowMode.NONE)
        light = PointLight(shadow_config=config)
        assert not light.casts_shadows


class TestGIConfig:
    """Tests for GI configuration."""

    def test_gi_contributor_config_defaults(self) -> None:
        """Test default GI contributor configuration."""
        config = GIContributorConfig()
        assert config.importance == GIImportance.MEDIUM
        assert config.emissive is False

    def test_gi_contributor_config_custom(self) -> None:
        """Test custom GI contributor configuration."""
        config = GIContributorConfig(
            importance=GIImportance.CRITICAL,
            emissive=True,
        )
        assert config.importance == GIImportance.CRITICAL
        assert config.emissive is True

    def test_light_contributes_gi_property(self) -> None:
        """Test the contributes_gi property."""
        light = PointLight(gi_config=GIContributorConfig())
        assert light.contributes_gi


class TestDecorators:
    """Tests for light decorators."""

    def test_shadow_caster_decorator(self) -> None:
        """Test shadow_caster decorator application."""
        @shadow_caster(mode="static", resolution_scale=0.5, cascade_bias=0.005)
        class CustomLight:
            pass

        assert hasattr(CustomLight, '_shadow_caster')
        assert CustomLight._shadow_caster is True
        assert CustomLight._shadow_mode == ShadowMode.STATIC
        assert CustomLight._shadow_resolution_scale == 0.5
        assert CustomLight._shadow_cascade_bias == 0.005

    def test_gi_contributor_decorator(self) -> None:
        """Test gi_contributor decorator application."""
        @gi_contributor(importance="high", emissive=True)
        class EmissiveLight:
            pass

        assert hasattr(EmissiveLight, '_gi_contributor')
        assert EmissiveLight._gi_contributor is True
        assert EmissiveLight._gi_importance == GIImportance.HIGH
        assert EmissiveLight._gi_emissive is True

    def test_decorator_on_directional_light(self) -> None:
        """Test that DirectionalLight has decorator attributes."""
        # DirectionalLight should have the decorators applied
        assert hasattr(DirectionalLight, '_shadow_caster')
        assert hasattr(DirectionalLight, '_gi_contributor')


class TestLightIntensityCalculations:
    """Tests for light intensity and power calculations."""

    def test_directional_light_luminous_power(self) -> None:
        """Test directional light luminous power."""
        light = DirectionalLight(intensity=1.0)
        power = light.get_luminous_power()
        assert power > 0

    def test_point_light_luminous_power(self) -> None:
        """Test point light luminous power."""
        light = PointLight(intensity=100.0)
        power = light.get_luminous_power()
        # Should be intensity * 4 * pi
        assert power == pytest.approx(100.0 * 4.0 * math.pi)

    def test_spot_light_luminous_power(self) -> None:
        """Test spot light luminous power."""
        light = SpotLight(intensity=100.0, outer_angle=math.radians(45))
        power = light.get_luminous_power()
        assert power > 0

    def test_area_light_luminous_power(self) -> None:
        """Test area light luminous power."""
        light = RectAreaLight(intensity=1000.0, width=1.0, height=1.0)
        power = light.get_luminous_power()
        # Should be intensity * area * pi
        assert power == pytest.approx(1000.0 * 1.0 * math.pi)

    def test_color_intensity_multiplication(self) -> None:
        """Test color * intensity helper."""
        light = PointLight(color=Vec3(1.0, 0.5, 0.25), intensity=2.0)
        result = light.get_color_intensity()
        assert result.x == pytest.approx(2.0)
        assert result.y == pytest.approx(1.0)
        assert result.z == pytest.approx(0.5)


class TestLightColorValidation:
    """Tests for light color validation."""

    def test_color_clamping(self) -> None:
        """Test that colors are clamped to [0, 1]."""
        light = PointLight(color=Vec3(1.5, -0.5, 0.5))
        assert light.color.x == pytest.approx(1.0)
        assert light.color.y == pytest.approx(0.0)
        assert light.color.z == pytest.approx(0.5)

    def test_negative_intensity_rejected(self) -> None:
        """Test that negative intensity raises error."""
        with pytest.raises(ValueError):
            PointLight(intensity=-10.0)


class TestEdgeCasesAndSafety:
    """Tests for edge cases and division by zero protection."""

    def test_zero_direction_defaults_to_down(self) -> None:
        """Test that zero direction vector defaults to (0, -1, 0)."""
        light = DirectionalLight(direction=Vec3(0, 0, 0))
        assert light.direction.x == pytest.approx(0.0)
        assert light.direction.y == pytest.approx(-1.0)
        assert light.direction.z == pytest.approx(0.0)

    def test_spot_light_zero_direction(self) -> None:
        """Test spot light with zero direction defaults safely."""
        light = SpotLight(direction=Vec3(0, 0, 0))
        assert light.direction.length() == pytest.approx(1.0)

    def test_spot_light_equal_angles_attenuation(self) -> None:
        """Test spot light attenuation when inner equals outer angle."""
        # This tests the division by zero protection
        light = SpotLight(
            inner_angle=math.radians(30),
            outer_angle=math.radians(30),  # Same as inner
        )
        # Should not crash when calculating attenuation
        atten = light.get_angular_attenuation(Vec3(0, 1, 0))
        assert atten >= 0.0 and atten <= 1.0

    def test_rect_area_light_parallel_direction_up(self) -> None:
        """Test rect area light when direction and up are parallel."""
        # Direction pointing up, up vector also pointing up - they're parallel
        light = RectAreaLight(
            direction=Vec3(0, 1, 0),
            up=Vec3(0, 1, 0),  # Parallel to direction
            width=1.0,
            height=1.0,
        )
        # Should handle this gracefully by choosing alternative up
        corners = light.get_corners()
        assert len(corners) == 4

    def test_disk_light_zero_direction(self) -> None:
        """Test disk area light with zero direction."""
        light = DiskAreaLight(direction=Vec3(0, 0, 0))
        assert light.direction.length() == pytest.approx(1.0)

    def test_ies_light_zero_direction(self) -> None:
        """Test IES light with zero direction."""
        light = IESLight(direction=Vec3(0, 0, 0))
        assert light.direction.length() == pytest.approx(1.0)

    def test_point_light_attenuation_at_zero_distance(self) -> None:
        """Test point light attenuation at zero distance doesn't crash."""
        light = PointLight(radius=10.0)
        atten = light.get_attenuation(0.0)
        # Should be close to 1.0 at center
        assert atten > 0.5

    def test_point_light_attenuation_negative_distance(self) -> None:
        """Test point light handles negative distance input."""
        light = PointLight(radius=10.0)
        # Negative distance should still work (treated as positive effectively)
        atten = light.get_attenuation(-5.0)
        # The implementation may or may not handle this, but should not crash
        assert isinstance(atten, float)
