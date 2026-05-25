"""Tests for advanced shading models.

Tests SubsurfaceScattering, ClearCoat, Anisotropy, Sheen,
Iridescence, Transmission, and AdvancedShadingModel.
"""
import math
import pytest

from engine.core.math.vec import Vec2, Vec3, Vec4
from engine.rendering.materials.advanced_models import (
    AdvancedShadingModel,
    Anisotropy,
    ClearCoat,
    Iridescence,
    ShadingModelType,
    Sheen,
    SubsurfaceProfile,
    SubsurfaceScattering,
    Transmission,
)


class TestSubsurfaceProfile:
    """Test SubsurfaceProfile diffusion profiles."""

    def test_default_profile(self):
        """Test default profile creation."""
        profile = SubsurfaceProfile()
        assert profile.name == "Default"
        assert profile.scatter_radius == 1.0
        assert profile.scatter_color == Vec3(1.0, 0.2, 0.1)

    def test_custom_profile(self):
        """Test custom profile creation."""
        profile = SubsurfaceProfile(
            name="CustomSkin",
            scatter_radius=0.5,
            scatter_color=Vec3(0.8, 0.4, 0.2),
            falloff_color=Vec3(0.9, 0.5, 0.3),
        )
        assert profile.name == "CustomSkin"
        assert profile.scatter_radius == 0.5
        assert profile.falloff_color == Vec3(0.9, 0.5, 0.3)

    def test_diffusion_profile_samples(self):
        """Test diffusion profile generation."""
        profile = SubsurfaceProfile(scatter_radius=1.0)
        samples = profile.get_diffusion_profile(num_samples=16)
        assert len(samples) == 16
        # All samples should be non-negative
        assert all(s >= 0 for s in samples)
        # Should be normalized (sum to ~1.0)
        assert abs(sum(samples) - 1.0) < 0.01

    def test_diffusion_profile_few_samples(self):
        """Test diffusion profile with minimal samples."""
        profile = SubsurfaceProfile(scatter_radius=0.5)
        samples = profile.get_diffusion_profile(num_samples=4)
        assert len(samples) == 4
        assert abs(sum(samples) - 1.0) < 0.01

    def test_zero_scatter_radius(self):
        """Test diffusion profile with zero radius raises error."""
        profile = SubsurfaceProfile(scatter_radius=0.0)
        # Zero radius causes division by zero in the Burley profile
        with pytest.raises((ZeroDivisionError, ValueError)):
            profile.get_diffusion_profile(num_samples=8)

    def test_to_shader_data(self):
        """Test conversion to shader-ready format."""
        profile = SubsurfaceProfile(
            name="TestProfile",
            scatter_radius=0.8,
            scatter_color=Vec3(0.5, 0.3, 0.1),
            boundary_color_bleed=0.7,
        )
        data = profile.to_shader_data()
        assert data["scatterRadius"] == 0.8
        assert data["scatterColor"] == (0.5, 0.3, 0.1)
        assert data["boundaryColorBleed"] == 0.7


class TestSubsurfaceScattering:
    """Test SubsurfaceScattering shading model."""

    def test_default_creation(self):
        """Test default SSS creation."""
        sss = SubsurfaceScattering()
        assert sss.opacity == 1.0
        assert sss.subsurface_color == Vec3(1.0, 1.0, 1.0)
        assert not sss.enable_transmission

    def test_with_custom_profile(self):
        """Test SSS with custom profile."""
        profile = SubsurfaceProfile(
            name="Skin",
            scatter_radius=0.5,
        )
        sss = SubsurfaceScattering(
            profile=profile,
            subsurface_color=Vec3(0.9, 0.5, 0.3),
            opacity=0.8,
        )
        assert sss.profile.name == "Skin"
        assert sss.opacity == 0.8

    def test_opacity_clamp(self):
        """Test opacity clamping to [0, 1]."""
        sss = SubsurfaceScattering(opacity=2.5)
        assert sss.opacity == 1.0

        sss.opacity = -0.5
        assert sss.opacity == 0.0

    def test_enable_transmission(self):
        """Test enabling transmission."""
        sss = SubsurfaceScattering(enable_transmission=True)
        assert sss.enable_transmission

        sss.enable_transmission = False
        assert not sss.enable_transmission

    def test_dirty_on_profile_change(self):
        """Test dirty flag on profile change."""
        sss = SubsurfaceScattering()
        assert sss._dirty

        sss._dirty = False
        sss.subsurface_color = Vec3(0.5, 0.5, 0.5)
        assert sss._dirty

    def test_to_shader_data(self):
        """Test shader data output."""
        sss = SubsurfaceScattering(opacity=0.7)
        data = sss.to_shader_data()
        assert "profile" in data
        assert data["opacity"] == 0.7

    def test_get_shader_defines_default(self):
        """Test shader defines without transmission."""
        sss = SubsurfaceScattering()
        defines = sss.get_shader_defines()
        assert "HAS_SUBSURFACE_SCATTERING" in defines
        assert "HAS_SSS_TRANSMISSION" not in defines

    def test_get_shader_defines_transmission(self):
        """Test shader defines with transmission enabled."""
        sss = SubsurfaceScattering(enable_transmission=True)
        defines = sss.get_shader_defines()
        assert "HAS_SUBSURFACE_SCATTERING" in defines
        assert "HAS_SSS_TRANSMISSION" in defines

    def test_skin_preset(self):
        """Test the skin preset profile."""
        sss = SubsurfaceScattering()
        assert SubsurfaceScattering.SKIN_PROFILE.name == "Skin"
        assert SubsurfaceScattering.SKIN_PROFILE.scatter_color == Vec3(0.48, 0.25, 0.17)

    def test_wax_preset(self):
        """Test the wax preset profile."""
        assert SubsurfaceScattering.WAX_PROFILE.name == "Wax"

    def test_jade_preset(self):
        """Test the jade preset profile."""
        assert SubsurfaceScattering.JADE_PROFILE.name == "Jade"

    def test_milk_preset(self):
        """Test the milk preset profile."""
        assert SubsurfaceScattering.MILK_PROFILE.name == "Milk"


class TestClearCoat:
    """Test ClearCoat shading model."""

    def test_default_creation(self):
        """Test default clear coat creation."""
        coat = ClearCoat()
        assert coat.intensity == 1.0
        assert coat.roughness == 0.0
        assert coat.ior == 1.5
        assert coat.normal_map is None
        assert coat.tint == Vec3(1.0, 1.0, 1.0)

    def test_custom_values(self):
        """Test custom clear coat values."""
        coat = ClearCoat(
            intensity=0.5,
            roughness=0.3,
            ior=1.6,
            tint=Vec3(0.9, 0.8, 0.7),
        )
        assert coat.intensity == 0.5
        assert coat.roughness == 0.3
        assert coat.ior == 1.6

    def test_intensity_clamp(self):
        """Test intensity clamping to [0, 1]."""
        coat = ClearCoat(intensity=2.0)
        assert coat.intensity == 1.0

    def test_ior_clamp(self):
        """Test IOR clamping to [1, 3]."""
        coat = ClearCoat(ior=5.0)
        assert coat.ior == 3.0

        coat = ClearCoat(ior=0.5)
        assert coat.ior == 1.0

    def test_roughness_clamp(self):
        """Test roughness clamping to [0, 1]."""
        coat = ClearCoat(roughness=1.5)
        assert coat.roughness == 1.0

    def test_validate_valid(self):
        """Test validation passes for valid values."""
        coat = ClearCoat(intensity=0.5, roughness=0.3, ior=1.5)
        is_valid, errors = coat.validate()
        assert is_valid
        assert len(errors) == 0

    def test_validate_post_clamp(self):
        """Test validation passes after __post_init__ clamps values."""
        # __post_init__ clamps to valid range, so validate should pass
        coat = ClearCoat(intensity=1.5, ior=0.5)
        is_valid, errors = coat.validate()
        assert is_valid  # __post_init__ already clamped to valid range
        assert len(errors) == 0

    def test_to_shader_data(self):
        """Test conversion to shader data."""
        coat = ClearCoat(intensity=0.8, roughness=0.2, ior=1.5)
        data = coat.to_shader_data()
        assert data["clearCoatIntensity"] == 0.8
        assert data["clearCoatRoughness"] == 0.2
        assert data["clearCoatF0"] is not None
        assert data["hasClearCoatNormal"] is False

    def test_to_shader_data_normal_map(self):
        """Test shader data with normal map."""
        coat = ClearCoat(normal_map="textures/coat_normal.png")
        data = coat.to_shader_data()
        assert data["hasClearCoatNormal"] is True

    def test_get_shader_defines(self):
        """Test shader defines."""
        coat = ClearCoat()
        defines = coat.get_shader_defines()
        assert "HAS_CLEAR_COAT" in defines

    def test_get_shader_defines_with_normal(self):
        """Test shader defines with normal map."""
        coat = ClearCoat(normal_map="coat_n.png")
        defines = coat.get_shader_defines()
        assert "HAS_CLEAR_COAT" in defines
        assert "HAS_CLEAR_COAT_NORMAL" in defines

    def test_f0_calculation(self):
        """Test F0 calculation from IOR."""
        coat = ClearCoat(ior=1.5)
        f0 = ((1.5 - 1.0) / (1.5 + 1.0)) ** 2
        data = coat.to_shader_data()
        assert abs(data["clearCoatF0"] - f0) < 0.001


class TestAnisotropy:
    """Test Anisotropy reflection model."""

    def test_default_creation(self):
        """Test default anisotropy creation."""
        aniso = Anisotropy()
        assert aniso.strength == 0.0
        assert aniso.rotation == 0.0
        assert aniso.tangent_map is None

    def test_custom_strength(self):
        """Test custom anisotropy strength."""
        aniso = Anisotropy(strength=0.8)
        assert aniso.strength == 0.8

    def test_strength_clamp(self):
        """Test strength clamping to [-1, 1]."""
        aniso = Anisotropy(strength=2.0)
        assert aniso.strength == 1.0

        aniso = Anisotropy(strength=-2.0)
        assert aniso.strength == -1.0

    def test_rotation_wrap(self):
        """Test rotation wraps correctly."""
        aniso = Anisotropy(rotation=3.0 * math.pi)
        assert 0 <= aniso.rotation < 2.0 * math.pi

    def test_anisotropic_roughness_positive(self):
        """Test anisotropic roughness with positive strength."""
        aniso = Anisotropy(strength=0.5)
        rough_t, rough_b = aniso.get_anisotropic_roughness(0.5)
        # Positive strength: rough_t > rough_b
        assert rough_t > rough_b

    def test_anisotropic_roughness_negative(self):
        """Test anisotropic roughness with negative strength."""
        aniso = Anisotropy(strength=-0.5)
        rough_t, rough_b = aniso.get_anisotropic_roughness(0.5)
        # Negative strength: rough_t < rough_b
        assert rough_t < rough_b

    def test_anisotropic_roughness_zero(self):
        """Test anisotropic roughness with zero strength (isotropic)."""
        aniso = Anisotropy(strength=0.0)
        rough_t, rough_b = aniso.get_anisotropic_roughness(0.5)
        assert rough_t == rough_b

    def test_anisotropic_roughness_max(self):
        """Test anisotropic roughness with max strength."""
        aniso = Anisotropy(strength=1.0)
        rough_t, rough_b = aniso.get_anisotropic_roughness(0.5)
        assert rough_t > rough_b
        assert abs(rough_t * rough_b - 0.25) < 0.001  # product ~= base^2

    def test_anisotropic_roughness_min(self):
        """Test anisotropic roughness with minimum strength."""
        aniso = Anisotropy(strength=-1.0)
        rough_t, rough_b = aniso.get_anisotropic_roughness(0.5)
        assert rough_t < rough_b

    def test_to_shader_data(self):
        """Test conversion to shader data."""
        aniso = Anisotropy(strength=0.7, rotation=math.pi / 4)
        data = aniso.to_shader_data()
        assert data["anisotropyStrength"] == 0.7
        assert len(data["anisotropyDirection"]) == 2
        assert abs(data["anisotropyDirection"][0] - math.cos(math.pi / 4)) < 0.001
        assert abs(data["anisotropyDirection"][1] - math.sin(math.pi / 4)) < 0.001

    def test_to_shader_data_tangent_map(self):
        """Test shader data with tangent map."""
        aniso = Anisotropy(tangent_map="tangents.png")
        data = aniso.to_shader_data()
        assert data["hasAnisotropyTangentMap"] is True

    def test_get_shader_defines_default(self):
        """Test shader defines without tangent map."""
        aniso = Anisotropy()
        defines = aniso.get_shader_defines()
        assert "HAS_ANISOTROPY" in defines
        assert "HAS_ANISOTROPY_TANGENT_MAP" not in defines

    def test_get_shader_defines_with_map(self):
        """Test shader defines with tangent map."""
        aniso = Anisotropy(tangent_map="t.png")
        defines = aniso.get_shader_defines()
        assert "HAS_ANISOTROPY" in defines
        assert "HAS_ANISOTROPY_TANGENT_MAP" in defines


class TestSheen:
    """Test Sheen shading model for fabrics."""

    def test_default_creation(self):
        """Test default sheen creation."""
        sheen = Sheen()
        assert sheen.color == Vec3(1.0, 1.0, 1.0)
        assert sheen.roughness == 0.5
        assert sheen.intensity == 1.0

    def test_custom_values(self):
        """Test custom sheen values."""
        sheen = Sheen(
            color=Vec3(0.8, 0.6, 0.4),
            roughness=0.3,
            intensity=0.7,
        )
        assert sheen.color == Vec3(0.8, 0.6, 0.4)
        assert sheen.roughness == 0.3
        assert sheen.intensity == 0.7

    def test_roughness_clamp(self):
        """Test roughness clamping to [0, 1]."""
        sheen = Sheen(roughness=1.5)
        assert sheen.roughness == 1.0

        sheen = Sheen(roughness=-0.5)
        assert sheen.roughness == 0.0

    def test_intensity_clamp(self):
        """Test intensity clamping to [0, 2]."""
        sheen = Sheen(intensity=3.0)
        assert sheen.intensity == 2.0

        sheen = Sheen(intensity=-1.0)
        assert sheen.intensity == 0.0

    def test_to_shader_data(self):
        """Test conversion to shader data."""
        sheen = Sheen(color=Vec3(0.5, 0.3, 0.1), roughness=0.4)
        data = sheen.to_shader_data()
        assert data["sheenColor"] == (0.5, 0.3, 0.1)
        assert data["sheenRoughness"] == 0.4
        assert "sheenIntensity" in data

    def test_get_shader_defines(self):
        """Test shader defines."""
        sheen = Sheen()
        defines = sheen.get_shader_defines()
        assert "HAS_SHEEN" in defines


class TestIridescence:
    """Test thin film iridescence model."""

    def test_default_creation(self):
        """Test default iridescence creation."""
        iri = Iridescence()
        assert iri.intensity == 1.0
        assert iri.ior == 1.3
        assert iri.thickness_min == 100.0
        assert iri.thickness_max == 400.0

    def test_custom_values(self):
        """Test custom iridescence values."""
        iri = Iridescence(
            intensity=0.7,
            ior=1.5,
            thickness_min=200.0,
            thickness_max=600.0,
        )
        assert iri.intensity == 0.7
        assert iri.thickness_max == 600.0

    def test_intensity_clamp(self):
        """Test intensity clamping to [0, 1]."""
        iri = Iridescence(intensity=2.0)
        assert iri.intensity == 1.0

    def test_thickness_invariant(self):
        """Test thickness_max is always >= thickness_min."""
        iri = Iridescence(thickness_min=500.0, thickness_max=100.0)
        assert iri.thickness_max >= iri.thickness_min

    def test_get_interference_color(self):
        """Test interference color calculation."""
        iri = Iridescence(thickness_min=300.0, thickness_max=300.0)
        color = iri.get_interference_color(300.0, 1.0)
        assert isinstance(color, Vec3)
        # Color components should be in [0, 1]
        assert all(0.0 <= c <= 1.0 for c in [color.x, color.y, color.z])

    def test_interference_color_different_angles(self):
        """Test interference changes with view angle."""
        iri = Iridescence(thickness_min=300.0, thickness_max=300.0)
        color_front = iri.get_interference_color(300.0, 1.0)
        color_side = iri.get_interference_color(300.0, 0.3)
        # Different angles should produce different colors
        assert color_front != color_side

    def test_to_shader_data(self):
        """Test conversion to shader data."""
        iri = Iridescence(intensity=0.8, ior=1.4)
        data = iri.to_shader_data()
        assert data["iridescenceIntensity"] == 0.8
        assert data["iridescenceIOR"] == 1.4
        assert data["hasIridescenceThicknessMap"] is False

    def test_to_shader_data_with_thickness_map(self):
        """Test shader data with thickness map."""
        iri = Iridescence(thickness_map="thickness.png")
        data = iri.to_shader_data()
        assert data["hasIridescenceThicknessMap"] is True

    def test_get_shader_defines_default(self):
        """Test shader defines without thickness map."""
        iri = Iridescence()
        defines = iri.get_shader_defines()
        assert "HAS_IRIDESCENCE" in defines
        assert "HAS_IRIDESCENCE_THICKNESS_MAP" not in defines

    def test_get_shader_defines_with_map(self):
        """Test shader defines with thickness map."""
        iri = Iridescence(thickness_map="t.png")
        defines = iri.get_shader_defines()
        assert "HAS_IRIDESCENCE" in defines
        assert "HAS_IRIDESCENCE_THICKNESS_MAP" in defines


class TestTransmission:
    """Test transmission model for transparent materials."""

    def test_default_creation(self):
        """Test default transmission creation."""
        trans = Transmission()
        assert trans.factor == 1.0
        assert trans.ior == 1.5
        assert trans.color == Vec3(1.0, 1.0, 1.0)
        assert trans.thickness == 0.0
        assert trans.roughness == 0.0

    def test_custom_values(self):
        """Test custom transmission values."""
        trans = Transmission(
            factor=0.8,
            ior=1.6,
            color=Vec3(0.9, 0.9, 1.0),
            thickness=2.0,
            roughness=0.1,
        )
        assert trans.factor == 0.8
        assert trans.roughness == 0.1

    def test_factor_clamp(self):
        """Test factor clamping to [0, 1]."""
        trans = Transmission(factor=2.0)
        assert trans.factor == 1.0

    def test_roughness_clamp(self):
        """Test roughness clamping to [0, 1]."""
        trans = Transmission(roughness=1.5)
        assert trans.roughness == 1.0

    def test_get_fresnel_mix(self):
        """Test Fresnel mixing calculation."""
        trans = Transmission(ior=1.5)
        # At normal incidence, reflection should be close to F0
        f0 = ((1.5 - 1.0) / (1.5 + 1.0)) ** 2
        reflection = trans.get_fresnel_mix(1.0)
        assert abs(reflection - f0) < 0.001

        # At grazing angle, reflection approaches 1.0
        reflection_grazing = trans.get_fresnel_mix(0.0)
        assert abs(reflection_grazing - 1.0) < 0.001

    def test_get_attenuation_infinite_distance(self):
        """Test attenuation with infinite distance."""
        trans = Transmission(attenuation_distance=float("inf"))
        atten = trans.get_attenuation(100.0)
        assert atten == Vec3(1.0, 1.0, 1.0)

    def test_get_attenuation_finite_distance(self):
        """Test attenuation with finite distance."""
        trans = Transmission(
            attenuation_color=Vec3(0.5, 0.5, 0.5),
            attenuation_distance=1.0,
        )
        atten = trans.get_attenuation(0.0)
        assert atten == Vec3(1.0, 1.0, 1.0)

        atten = trans.get_attenuation(1.0)
        assert all(0.4 < c < 0.6 for c in [atten.x, atten.y, atten.z])

    def test_to_shader_data(self):
        """Test conversion to shader data."""
        trans = Transmission(factor=0.7, ior=1.4)
        data = trans.to_shader_data()
        assert data["transmissionFactor"] == 0.7
        assert data["transmissionIOR"] == 1.4

    def test_get_shader_defines_default(self):
        """Test shader defines without volume."""
        trans = Transmission()
        defines = trans.get_shader_defines()
        assert "HAS_TRANSMISSION" in defines
        assert "HAS_TRANSMISSION_VOLUME" not in defines

    def test_get_shader_defines_with_volume(self):
        """Test shader defines with thickness (volume)."""
        trans = Transmission(thickness=1.0)
        defines = trans.get_shader_defines()
        assert "HAS_TRANSMISSION" in defines
        assert "HAS_TRANSMISSION_VOLUME" in defines


class TestAdvancedShadingModel:
    """Test AdvancedShadingModel container."""

    def test_empty_container(self):
        """Test empty advanced shading model."""
        model = AdvancedShadingModel()
        active = model.get_active_models()
        assert len(active) == 0
        defines = model.get_all_shader_defines()
        assert len(defines) == 0
        data = model.to_shader_data()
        assert len(data) == 0

    def test_with_subsurface(self):
        """Test with subsurface scattering."""
        model = AdvancedShadingModel(
            subsurface=SubsurfaceScattering(),
        )
        active = model.get_active_models()
        assert ShadingModelType.SUBSURFACE in active
        defines = model.get_all_shader_defines()
        assert "HAS_SUBSURFACE_SCATTERING" in defines

    def test_with_clear_coat(self):
        """Test with clear coat."""
        model = AdvancedShadingModel(
            clear_coat=ClearCoat(),
        )
        active = model.get_active_models()
        assert ShadingModelType.CLEAR_COAT in active

    def test_with_anisotropy(self):
        """Test with anisotropy."""
        model = AdvancedShadingModel(
            anisotropy=Anisotropy(strength=0.5),
        )
        active = model.get_active_models()
        assert ShadingModelType.ANISOTROPY in active

    def test_with_sheen(self):
        """Test with sheen."""
        model = AdvancedShadingModel(sheen=Sheen())
        active = model.get_active_models()
        assert ShadingModelType.SHEEN in active

    def test_with_iridescence(self):
        """Test with iridescence."""
        model = AdvancedShadingModel(iridescence=Iridescence())
        active = model.get_active_models()
        assert ShadingModelType.IRIDESCENCE in active

    def test_with_transmission(self):
        """Test with transmission."""
        model = AdvancedShadingModel(transmission=Transmission())
        active = model.get_active_models()
        assert ShadingModelType.TRANSMISSION in active

    def test_multiple_models(self):
        """Test with multiple shading models combined."""
        model = AdvancedShadingModel(
            clear_coat=ClearCoat(),
            anisotropy=Anisotropy(strength=0.3),
            sheen=Sheen(),
        )
        active = model.get_active_models()
        assert len(active) == 3
        defines = model.get_all_shader_defines()
        assert "HAS_CLEAR_COAT" in defines
        assert "HAS_ANISOTROPY" in defines
        assert "HAS_SHEEN" in defines

    def test_to_shader_data_multiple(self):
        """Test shader data with multiple models."""
        model = AdvancedShadingModel(
            subsurface=SubsurfaceScattering(opacity=0.8),
            clear_coat=ClearCoat(intensity=0.5),
        )
        data = model.to_shader_data()
        assert "subsurface" in data
        assert "clearCoat" in data
        assert "anisotropy" not in data
