"""
Tests for Trinity Pattern - Tier 42: RENDERING Decorators
"""

import pytest

from trinity.decorators.rendering import (
    VALID_BLEND_MODE,
    VALID_CAPTURE_MODE,
    VALID_GI_IMPORTANCE,
    VALID_MATERIAL_DOMAIN,
    VALID_SHADOW_MODE,
    GIContributorConfig,
    MaterialBlendConfig,
    MaterialDomainConfig,
    ReflectionProbeConfig,
    RenderLayerConfig,
    ShadowCasterConfig,
    gi_contributor,
    material_blend,
    material_domain,
    reflection_probe,
    render_layer,
    shadow_caster,
)
from trinity.decorators.registry import registry


class TestGIContributor:
    """Test @gi_contributor decorator."""

    def test_basic_application(self):
        """Test basic decorator application with defaults."""

        @gi_contributor()
        class TestObject:
            pass

        assert hasattr(TestObject, "_gi_contributor")
        assert TestObject._gi_contributor is True
        assert TestObject._gi_importance == "medium"
        assert TestObject._gi_emissive is False
        assert isinstance(TestObject._gi_config, GIContributorConfig)
        assert TestObject._gi_config.importance == "medium"
        assert TestObject._gi_config.emissive is False

    def test_custom_params(self):
        """Test decorator with custom parameters."""

        @gi_contributor(importance="critical", emissive=True)
        class HighGI:
            pass

        assert HighGI._gi_importance == "critical"
        assert HighGI._gi_emissive is True
        assert HighGI._gi_config.importance == "critical"
        assert HighGI._gi_config.emissive is True

    def test_invalid_importance(self):
        """Test validation of importance parameter."""
        with pytest.raises(ValueError, match="Invalid importance"):

            @gi_contributor(importance="invalid")
            class BadGI:
                pass

    def test_all_valid_importances(self):
        """Test all valid importance values."""
        for importance in VALID_GI_IMPORTANCE:

            @gi_contributor(importance=importance)
            class TestGI:
                pass

            assert TestGI._gi_importance == importance

    def test_registry_registration(self):
        """Test that decorator is registered properly."""
        spec = registry.get("gi_contributor")
        assert spec is not None
        assert spec.name == "gi_contributor"
        assert spec.tier.name == "RENDERING"

    def test_tags_applied(self):
        """Test that tags are applied."""

        @gi_contributor()
        class Tagged:
            pass

        assert hasattr(Tagged, "_tags")
        assert Tagged._tags.get("gi_contributor") is True


class TestShadowCaster:
    """Test @shadow_caster decorator."""

    def test_basic_application(self):
        """Test basic decorator application with defaults."""

        @shadow_caster()
        class TestObject:
            pass

        assert hasattr(TestObject, "_shadow_caster")
        assert TestObject._shadow_caster is True
        assert TestObject._shadow_mode == "dynamic"
        assert TestObject._shadow_resolution_scale == 1.0
        assert TestObject._shadow_cascade_bias == 0.0

    def test_custom_params(self):
        """Test decorator with custom parameters."""

        @shadow_caster(mode="static", resolution_scale=2.0, cascade_bias=0.1)
        class StaticShadow:
            pass

        assert StaticShadow._shadow_mode == "static"
        assert StaticShadow._shadow_resolution_scale == 2.0
        assert StaticShadow._shadow_cascade_bias == 0.1

    def test_invalid_mode(self):
        """Test validation of mode parameter."""
        with pytest.raises(ValueError, match="Invalid mode"):

            @shadow_caster(mode="invalid")
            class BadMode:
                pass

    def test_invalid_resolution_scale(self):
        """Test validation of resolution_scale parameter."""
        with pytest.raises(ValueError, match="resolution_scale must be > 0"):

            @shadow_caster(resolution_scale=0)
            class BadScale:
                pass

        with pytest.raises(ValueError, match="resolution_scale must be > 0"):

            @shadow_caster(resolution_scale=-1.0)
            class NegScale:
                pass

    def test_all_valid_modes(self):
        """Test all valid shadow modes."""
        for mode in VALID_SHADOW_MODE:

            @shadow_caster(mode=mode)
            class TestShadow:
                pass

            assert TestShadow._shadow_mode == mode


class TestReflectionProbe:
    """Test @reflection_probe decorator."""

    def test_basic_application(self):
        """Test basic decorator application with defaults."""

        @reflection_probe()
        class TestObject:
            pass

        assert hasattr(TestObject, "_reflection_probe")
        assert TestObject._reflection_probe is True
        assert TestObject._reflection_capture_mode == "baked"
        assert TestObject._reflection_resolution == 256
        assert TestObject._reflection_update_rate == 0.0

    def test_custom_params(self):
        """Test decorator with custom parameters."""

        @reflection_probe(capture_mode="realtime", resolution=512, update_rate=30.0)
        class RealtimeProbe:
            pass

        assert RealtimeProbe._reflection_capture_mode == "realtime"
        assert RealtimeProbe._reflection_resolution == 512
        assert RealtimeProbe._reflection_update_rate == 30.0

    def test_invalid_capture_mode(self):
        """Test validation of capture_mode parameter."""
        with pytest.raises(ValueError, match="Invalid capture_mode"):

            @reflection_probe(capture_mode="invalid")
            class BadMode:
                pass

    def test_invalid_resolution(self):
        """Test validation of resolution parameter."""
        with pytest.raises(ValueError, match="resolution must be > 0"):

            @reflection_probe(resolution=0)
            class BadRes:
                pass

    def test_all_valid_capture_modes(self):
        """Test all valid capture modes."""
        for mode in VALID_CAPTURE_MODE:

            @reflection_probe(capture_mode=mode)
            class TestProbe:
                pass

            assert TestProbe._reflection_capture_mode == mode


class TestMaterialDomain:
    """Test @material_domain decorator."""

    def test_basic_application(self):
        """Test basic decorator application."""

        @material_domain(domain="surface")
        class SurfaceMat:
            pass

        assert hasattr(SurfaceMat, "_material_domain")
        assert SurfaceMat._material_domain is True
        assert SurfaceMat._material_domain_type == "surface"

    def test_all_valid_domains(self):
        """Test all valid material domains."""
        for domain in VALID_MATERIAL_DOMAIN:

            @material_domain(domain=domain)
            class TestMat:
                pass

            assert TestMat._material_domain_type == domain

    def test_invalid_domain(self):
        """Test validation of domain parameter."""
        with pytest.raises(ValueError, match="Invalid domain"):

            @material_domain(domain="invalid")
            class BadDomain:
                pass


class TestMaterialBlend:
    """Test @material_blend decorator."""

    def test_basic_application(self):
        """Test basic decorator application."""

        @material_blend(mode="opaque")
        class OpaqueMat:
            pass

        assert hasattr(OpaqueMat, "_material_blend")
        assert OpaqueMat._material_blend is True
        assert OpaqueMat._material_blend_mode == "opaque"

    def test_all_valid_modes(self):
        """Test all valid blend modes."""
        for mode in VALID_BLEND_MODE:

            @material_blend(mode=mode)
            class TestMat:
                pass

            assert TestMat._material_blend_mode == mode

    def test_invalid_mode(self):
        """Test validation of mode parameter."""
        with pytest.raises(ValueError, match="Invalid mode"):

            @material_blend(mode="invalid")
            class BadMode:
                pass


class TestRenderLayer:
    """Test @render_layer decorator."""

    def test_basic_application(self):
        """Test basic decorator application."""

        @render_layer(layer="main", order=0)
        class MainLayer:
            pass

        assert hasattr(MainLayer, "_render_layer")
        assert MainLayer._render_layer is True
        assert MainLayer._render_layer_name == "main"
        assert MainLayer._render_layer_order == 0

    def test_custom_params(self):
        """Test decorator with custom parameters."""

        @render_layer(layer="ui", order=100)
        class UILayer:
            pass

        assert UILayer._render_layer_name == "ui"
        assert UILayer._render_layer_order == 100

    def test_empty_layer_validation(self):
        """Test validation of empty layer name."""
        with pytest.raises(ValueError, match="layer must be a non-empty string"):

            @render_layer(layer="")
            class EmptyLayer:
                pass

    def test_negative_order(self):
        """Test negative order values are allowed."""

        @render_layer(layer="background", order=-10)
        class BackLayer:
            pass

        assert BackLayer._render_layer_order == -10


class TestDecoratorComposition:
    """Test combining multiple rendering decorators."""

    def test_multiple_decorators(self):
        """Test applying multiple decorators to same class."""

        @render_layer(layer="main")
        @shadow_caster(mode="static")
        @gi_contributor(importance="high")
        class ComplexObject:
            pass

        assert ComplexObject._render_layer is True
        assert ComplexObject._shadow_caster is True
        assert ComplexObject._gi_contributor is True

    def test_material_decorators_together(self):
        """Test material domain and blend mode together."""

        @material_blend(mode="translucent")
        @material_domain(domain="surface")
        class TranslucentSurface:
            pass

        assert TranslucentSurface._material_domain_type == "surface"
        assert TranslucentSurface._material_blend_mode == "translucent"

    def test_full_rendering_stack(self):
        """Test applying all rendering decorators."""

        @render_layer(layer="main", order=0)
        @material_blend(mode="opaque")
        @material_domain(domain="surface")
        @reflection_probe(capture_mode="baked")
        @shadow_caster(mode="dynamic")
        @gi_contributor(importance="critical")
        class FullyRendered:
            pass

        # Verify all decorators applied
        assert hasattr(FullyRendered, "_applied_decorators")
        applied = FullyRendered._applied_decorators
        assert "gi_contributor" in applied
        assert "shadow_caster" in applied
        assert "reflection_probe" in applied
        assert "material_domain" in applied
        assert "material_blend" in applied
        assert "render_layer" in applied


class TestRegistryIntegration:
    """Test integration with decorator registry."""

    def test_all_decorators_registered(self):
        """Test that all rendering decorators are registered."""
        decorators = [
            "gi_contributor",
            "shadow_caster",
            "reflection_probe",
            "material_domain",
            "material_blend",
            "render_layer",
        ]

        for name in decorators:
            spec = registry.get(name)
            assert spec is not None, f"Decorator {name} not registered"
            assert spec.tier.name == "RENDERING"

    def test_rendering_tier_contains_decorators(self):
        """Test that RENDERING tier contains our decorators."""
        from trinity.decorators.registry import Tier

        rendering_specs = registry.by_tier(Tier.RENDERING)
        names = [spec.name for spec in rendering_specs]

        assert "gi_contributor" in names
        assert "shadow_caster" in names
        assert "reflection_probe" in names
        assert "material_domain" in names
        assert "material_blend" in names
        assert "render_layer" in names
