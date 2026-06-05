"""Tests for subsystem quality capabilities (T-CC-0.5, T-CC-0.6)."""

import pytest

from trinity.types import QualityTier

from engine.rendering.quality.capabilities import QualityCapabilities
from engine.rendering.quality.subsystems.atmosphere import AtmosphereCapabilities
from engine.rendering.quality.subsystems.demoscene import DemosceneCapabilities
from engine.rendering.quality.subsystems.gi import GICapabilities
from engine.rendering.quality.subsystems.lighting import LightingCapabilities
from engine.rendering.quality.subsystems.materials import MaterialsCapabilities
from engine.rendering.quality.subsystems.particles import ParticlesCapabilities
from engine.rendering.quality.subsystems.postprocess import PostProcessCapabilities
from engine.rendering.quality.subsystems.raytracing import RayTracingCapabilities
from engine.rendering.quality.subsystems.reflections import ReflectionsCapabilities
from engine.rendering.quality.subsystems.shadows import ShadowsCapabilities
from engine.rendering.quality.subsystems.terrain import TerrainCapabilities


class TestMaterialsCapabilities:
    """Test MaterialsCapabilities (S3)."""

    def test_implements_protocol(self):
        """Test implements QualityCapabilities."""
        caps = MaterialsCapabilities()
        assert isinstance(caps, QualityCapabilities)

    def test_subsystem_name(self):
        """Test subsystem name."""
        caps = MaterialsCapabilities()
        assert caps.subsystem_name == "materials"

    def test_low_tier_basic_pbr(self):
        """Test LOW tier has basic PBR only."""
        caps = MaterialsCapabilities()
        features = caps.get_features(QualityTier.LOW)
        assert features.has_feature("base_color")
        assert features.has_feature("normal_mapping")
        assert not features.has_feature("subsurface_scattering")

    def test_ultra_tier_all_features(self):
        """Test ULTRA tier has all features."""
        caps = MaterialsCapabilities()
        features = caps.get_features(QualityTier.ULTRA)
        assert features.has_feature("subsurface_scattering")
        assert features.has_feature("iridescence")
        assert features.has_feature("transmission")

    def test_budget_increases_with_tier(self):
        """Test budget increases with tier."""
        caps = MaterialsCapabilities()
        low = caps.get_budget(QualityTier.LOW)
        ultra = caps.get_budget(QualityTier.ULTRA)
        assert ultra.gpu_time_ms > low.gpu_time_ms
        assert ultra.memory_mb > low.memory_mb

    def test_fallback_chain_subsurface(self):
        """Test subsurface fallback chain."""
        caps = MaterialsCapabilities()
        chain = caps.get_fallback_chain("subsurface")
        assert chain is not None
        assert chain.primary == "separable_sss"


class TestLightingCapabilities:
    """Test LightingCapabilities (S4)."""

    def test_implements_protocol(self):
        """Test implements QualityCapabilities."""
        caps = LightingCapabilities()
        assert isinstance(caps, QualityCapabilities)

    def test_subsystem_name(self):
        """Test subsystem name."""
        caps = LightingCapabilities()
        assert caps.subsystem_name == "lighting"

    def test_low_tier_limited_lights(self):
        """Test LOW tier has limited lights."""
        caps = LightingCapabilities()
        features = caps.get_features(QualityTier.LOW)
        assert features.get_param("max_lights") == 8
        assert not features.has_feature("clustered_lighting")

    def test_high_tier_deferred(self):
        """Test HIGH tier has deferred shading."""
        caps = LightingCapabilities()
        features = caps.get_features(QualityTier.HIGH)
        assert features.has_feature("deferred_shading")
        assert features.has_feature("area_lights")

    def test_ultra_tier_unlimited(self):
        """Test ULTRA tier has unlimited lights."""
        caps = LightingCapabilities()
        features = caps.get_features(QualityTier.ULTRA)
        assert features.get_param("max_lights") == -1


class TestShadowsCapabilities:
    """Test ShadowsCapabilities (S5)."""

    def test_implements_protocol(self):
        """Test implements QualityCapabilities."""
        caps = ShadowsCapabilities()
        assert isinstance(caps, QualityCapabilities)

    def test_subsystem_name(self):
        """Test subsystem name."""
        caps = ShadowsCapabilities()
        assert caps.subsystem_name == "shadows"

    def test_low_tier_single_cascade(self):
        """Test LOW tier has single cascade."""
        caps = ShadowsCapabilities()
        features = caps.get_features(QualityTier.LOW)
        assert features.get_param("cascade_count") == 1

    def test_high_tier_vsm(self):
        """Test HIGH tier has VSM."""
        caps = ShadowsCapabilities()
        features = caps.get_features(QualityTier.HIGH)
        assert features.has_feature("vsm")
        assert features.has_feature("contact_shadows")

    def test_ultra_tier_ray_traced(self):
        """Test ULTRA tier has ray-traced shadows."""
        caps = ShadowsCapabilities()
        features = caps.get_features(QualityTier.ULTRA)
        assert features.has_feature("ray_traced_shadows")

    def test_resolution_increases_with_tier(self):
        """Test shadow resolution increases with tier."""
        caps = ShadowsCapabilities()
        low = caps.get_resolution(QualityTier.LOW)
        ultra = caps.get_resolution(QualityTier.ULTRA)
        assert ultra.shadow_resolution > low.shadow_resolution


class TestGICapabilities:
    """Test GICapabilities (S6)."""

    def test_implements_protocol(self):
        """Test implements QualityCapabilities."""
        caps = GICapabilities()
        assert isinstance(caps, QualityCapabilities)

    def test_subsystem_name(self):
        """Test subsystem name."""
        caps = GICapabilities()
        assert caps.subsystem_name == "gi"

    def test_low_tier_baked_only(self):
        """Test LOW tier uses baked lightmaps only."""
        caps = GICapabilities()
        features = caps.get_features(QualityTier.LOW)
        assert features.has_feature("baked_lightmaps")
        assert not features.has_feature("ssao")
        assert not features.has_feature("ddgi")

    def test_high_tier_ddgi(self):
        """Test HIGH tier has DDGI."""
        caps = GICapabilities()
        features = caps.get_features(QualityTier.HIGH)
        assert features.has_feature("ddgi")
        assert features.has_feature("ssr")

    def test_ultra_tier_rtgi(self):
        """Test ULTRA tier has ray-traced GI."""
        caps = GICapabilities()
        features = caps.get_features(QualityTier.ULTRA)
        assert features.has_feature("rtgi")
        assert features.has_feature("rt_reflections")

    def test_fallback_chain_ao(self):
        """Test ambient occlusion fallback chain."""
        caps = GICapabilities()
        chain = caps.get_fallback_chain("ambient_occlusion")
        assert chain is not None
        assert chain.primary == "gtao"


class TestPostProcessCapabilities:
    """Test PostProcessCapabilities (S8)."""

    def test_implements_protocol(self):
        """Test implements QualityCapabilities."""
        caps = PostProcessCapabilities()
        assert isinstance(caps, QualityCapabilities)

    def test_subsystem_name(self):
        """Test subsystem name."""
        caps = PostProcessCapabilities()
        assert caps.subsystem_name == "postprocess"

    def test_low_tier_basic(self):
        """Test LOW tier has basic effects."""
        caps = PostProcessCapabilities()
        features = caps.get_features(QualityTier.LOW)
        assert features.has_feature("tonemapping")
        assert features.has_feature("bloom")
        assert not features.has_feature("dof")

    def test_medium_tier_dof_taa(self):
        """Test MEDIUM tier has DOF and TAA."""
        caps = PostProcessCapabilities()
        features = caps.get_features(QualityTier.MEDIUM)
        assert features.has_feature("dof")
        assert features.has_feature("taa")

    def test_ultra_tier_upscaling(self):
        """Test ULTRA tier has temporal upscaling."""
        caps = PostProcessCapabilities()
        features = caps.get_features(QualityTier.ULTRA)
        assert features.has_feature("temporal_upscaling")
        assert features.has_feature("bokeh_dof")


class TestAtmosphereCapabilities:
    """Test AtmosphereCapabilities (S11)."""

    def test_implements_protocol(self):
        """Test implements QualityCapabilities."""
        caps = AtmosphereCapabilities()
        assert isinstance(caps, QualityCapabilities)

    def test_subsystem_name(self):
        """Test subsystem name."""
        caps = AtmosphereCapabilities()
        assert caps.subsystem_name == "atmosphere"

    def test_low_tier_gradient(self):
        """Test LOW tier uses gradient sky."""
        caps = AtmosphereCapabilities()
        features = caps.get_features(QualityTier.LOW)
        assert features.has_feature("gradient_sky")
        assert not features.has_feature("bruneton_scattering")

    def test_high_tier_volumetric_clouds(self):
        """Test HIGH tier has volumetric clouds."""
        caps = AtmosphereCapabilities()
        features = caps.get_features(QualityTier.HIGH)
        assert features.has_feature("volumetric_clouds")
        assert features.has_feature("god_rays")

    def test_ultra_tier_full(self):
        """Test ULTRA tier has all features."""
        caps = AtmosphereCapabilities()
        features = caps.get_features(QualityTier.ULTRA)
        assert features.has_feature("cloud_shadows")
        assert features.has_feature("temporal_reprojection")
        assert features.has_feature("multiple_scattering")


class TestReflectionsCapabilities:
    """Test ReflectionsCapabilities (S7)."""

    def test_implements_protocol(self):
        """Test implements QualityCapabilities."""
        caps = ReflectionsCapabilities()
        assert isinstance(caps, QualityCapabilities)

    def test_subsystem_name(self):
        """Test subsystem name."""
        caps = ReflectionsCapabilities()
        assert caps.subsystem_name == "reflections"

    def test_low_tier_env_map_only(self):
        """Test LOW tier uses environment map only."""
        caps = ReflectionsCapabilities()
        features = caps.get_features(QualityTier.LOW)
        assert features.has_feature("environment_map")
        assert not features.has_feature("ssr")

    def test_high_tier_ssr(self):
        """Test HIGH tier has SSR."""
        caps = ReflectionsCapabilities()
        features = caps.get_features(QualityTier.HIGH)
        assert features.has_feature("ssr")
        assert features.has_feature("planar_reflections")

    def test_ultra_tier_rt(self):
        """Test ULTRA tier has RT reflections."""
        caps = ReflectionsCapabilities()
        features = caps.get_features(QualityTier.ULTRA)
        assert features.has_feature("rt_reflections")


class TestParticlesCapabilities:
    """Test ParticlesCapabilities (S9)."""

    def test_implements_protocol(self):
        """Test implements QualityCapabilities."""
        caps = ParticlesCapabilities()
        assert isinstance(caps, QualityCapabilities)

    def test_subsystem_name(self):
        """Test subsystem name."""
        caps = ParticlesCapabilities()
        assert caps.subsystem_name == "particles"

    def test_low_tier_cpu_sim(self):
        """Test LOW tier uses CPU simulation."""
        caps = ParticlesCapabilities()
        features = caps.get_features(QualityTier.LOW)
        assert features.has_feature("cpu_simulation")
        assert features.get_param("max_particles") == 1000

    def test_high_tier_gpu_sim(self):
        """Test HIGH tier uses GPU simulation."""
        caps = ParticlesCapabilities()
        features = caps.get_features(QualityTier.HIGH)
        assert features.has_feature("gpu_simulation")
        assert features.has_feature("trails")

    def test_ultra_tier_million_particles(self):
        """Test ULTRA tier supports 1M particles."""
        caps = ParticlesCapabilities()
        features = caps.get_features(QualityTier.ULTRA)
        assert features.get_param("max_particles") == 1000000


class TestRayTracingCapabilities:
    """Test RayTracingCapabilities (S10)."""

    def test_implements_protocol(self):
        """Test implements QualityCapabilities."""
        caps = RayTracingCapabilities()
        assert isinstance(caps, QualityCapabilities)

    def test_subsystem_name(self):
        """Test subsystem name."""
        caps = RayTracingCapabilities()
        assert caps.subsystem_name == "raytracing"

    def test_low_tier_no_rt(self):
        """Test LOW tier has no RT."""
        caps = RayTracingCapabilities()
        features = caps.get_features(QualityTier.LOW)
        assert features.get_param("rt_available") is False
        assert not features.has_feature("rt_shadows")

    def test_high_tier_inline_queries(self):
        """Test HIGH tier has inline ray queries."""
        caps = RayTracingCapabilities()
        features = caps.get_features(QualityTier.HIGH)
        assert features.has_feature("inline_ray_query")
        assert features.has_feature("rt_shadows")

    def test_ultra_tier_full_rt(self):
        """Test ULTRA tier has full RT."""
        caps = RayTracingCapabilities()
        features = caps.get_features(QualityTier.ULTRA)
        assert features.has_feature("rt_gi")
        assert features.has_feature("rt_reflections")


class TestTerrainCapabilities:
    """Test TerrainCapabilities (S12)."""

    def test_implements_protocol(self):
        """Test implements QualityCapabilities."""
        caps = TerrainCapabilities()
        assert isinstance(caps, QualityCapabilities)

    def test_subsystem_name(self):
        """Test subsystem name."""
        caps = TerrainCapabilities()
        assert caps.subsystem_name == "terrain"

    def test_low_tier_basic(self):
        """Test LOW tier has basic terrain."""
        caps = TerrainCapabilities()
        features = caps.get_features(QualityTier.LOW)
        assert features.has_feature("heightmap_terrain")
        assert not features.has_feature("tessellation")

    def test_high_tier_fft_ocean(self):
        """Test HIGH tier has FFT ocean."""
        caps = TerrainCapabilities()
        features = caps.get_features(QualityTier.HIGH)
        assert features.has_feature("fft_ocean")
        assert features.has_feature("tessellation")

    def test_ultra_tier_full(self):
        """Test ULTRA tier has all features."""
        caps = TerrainCapabilities()
        features = caps.get_features(QualityTier.ULTRA)
        assert features.has_feature("underwater")
        assert features.has_feature("caustics")


class TestDemosceneCapabilities:
    """Test DemosceneCapabilities (S13)."""

    def test_implements_protocol(self):
        """Test implements QualityCapabilities."""
        caps = DemosceneCapabilities()
        assert isinstance(caps, QualityCapabilities)

    def test_subsystem_name(self):
        """Test subsystem name."""
        caps = DemosceneCapabilities()
        assert caps.subsystem_name == "demoscene"

    def test_low_tier_basic_sdf(self):
        """Test LOW tier has basic SDF."""
        caps = DemosceneCapabilities()
        features = caps.get_features(QualityTier.LOW)
        assert features.has_feature("sphere_tracing")
        assert features.get_param("max_steps") == 64

    def test_high_tier_reflections(self):
        """Test HIGH tier has reflections."""
        caps = DemosceneCapabilities()
        features = caps.get_features(QualityTier.HIGH)
        assert features.has_feature("reflections")
        assert features.has_feature("dof")

    def test_ultra_tier_fractals(self):
        """Test ULTRA tier has fractals."""
        caps = DemosceneCapabilities()
        features = caps.get_features(QualityTier.ULTRA)
        assert features.has_feature("fractals")
        assert features.has_feature("volumetric_effects")


class TestSubsystemConsistency:
    """Test consistency across all subsystems."""

    @pytest.fixture
    def all_subsystems(self):
        """Create all subsystem capabilities."""
        return [
            MaterialsCapabilities(),
            LightingCapabilities(),
            ShadowsCapabilities(),
            GICapabilities(),
            PostProcessCapabilities(),
            AtmosphereCapabilities(),
            ReflectionsCapabilities(),
            ParticlesCapabilities(),
            RayTracingCapabilities(),
            TerrainCapabilities(),
            DemosceneCapabilities(),
        ]

    def test_all_have_unique_names(self, all_subsystems):
        """Test all subsystems have unique names."""
        names = [s.subsystem_name for s in all_subsystems]
        assert len(names) == len(set(names))

    def test_all_support_all_tiers(self, all_subsystems):
        """Test all subsystems support all quality tiers."""
        for subsystem in all_subsystems:
            for tier in QualityTier:
                # Should not raise
                features = subsystem.get_features(tier)
                budget = subsystem.get_budget(tier)
                assert features is not None
                assert budget is not None

    def test_budgets_increase_monotonically(self, all_subsystems):
        """Test budgets generally increase with tier."""
        for subsystem in all_subsystems:
            prev_time = 0
            for tier in QualityTier:
                budget = subsystem.get_budget(tier)
                assert budget.gpu_time_ms >= prev_time
                prev_time = budget.gpu_time_ms

    def test_ultra_has_most_features(self, all_subsystems):
        """Test ULTRA tier has the most features."""
        for subsystem in all_subsystems:
            ultra = subsystem.get_features(QualityTier.ULTRA)
            for tier in [QualityTier.LOW, QualityTier.MEDIUM, QualityTier.HIGH]:
                other = subsystem.get_features(tier)
                # ULTRA should have at least as many features
                assert len(ultra.enabled_features) >= len(other.enabled_features)
