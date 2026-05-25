"""
Tests for Tier 45: PARTICLES_VFX decorators.
"""

import pytest

from trinity.decorators.particles_vfx import (
    VALID_PARTICLE_SIMULATIONS,
    VALID_PARTICLE_STAGES,
    VALID_TEXTURE_MODES,
    VALID_VFX_TRIGGERS,
    decal,
    gpu_particle,
    particle_emitter,
    particle_module,
    trail,
    vfx_event,
)
from trinity.decorators.registry import Tier, registry


class TestParticleEmitter:
    """Test @particle_emitter decorator."""

    def test_basic_application(self):
        """Test basic decorator application with default params."""

        @particle_emitter()
        class BasicEmitter:
            pass

        assert hasattr(BasicEmitter, "_particle_emitter")
        assert BasicEmitter._particle_emitter is True
        assert BasicEmitter._particle_max_particles == 1000
        assert BasicEmitter._particle_simulation == "auto"
        assert BasicEmitter._particle_budget_category is None

    def test_custom_params(self):
        """Test custom parameters."""

        @particle_emitter(
            max_particles=5000,
            simulation="gpu",
            budget_category="high_quality",
        )
        class CustomEmitter:
            pass

        assert CustomEmitter._particle_max_particles == 5000
        assert CustomEmitter._particle_simulation == "gpu"
        assert CustomEmitter._particle_budget_category == "high_quality"

    def test_all_simulation_modes(self):
        """Test all valid simulation modes."""
        for sim_mode in VALID_PARTICLE_SIMULATIONS:

            @particle_emitter(simulation=sim_mode)
            class EmitterTest:
                pass

            assert EmitterTest._particle_simulation == sim_mode

    def test_invalid_max_particles(self):
        """Test invalid max_particles raises ValueError."""
        with pytest.raises(ValueError, match="max_particles must be > 0"):

            @particle_emitter(max_particles=0)
            class EmitterBad:
                pass

        with pytest.raises(ValueError, match="max_particles must be > 0"):

            @particle_emitter(max_particles=-100)
            class EmitterBad2:
                pass

    def test_invalid_simulation(self):
        """Test invalid simulation raises ValueError."""
        with pytest.raises(ValueError, match="Invalid simulation"):

            @particle_emitter(simulation="invalid")
            class EmitterBad:
                pass

    def test_tags(self):
        """Test that tags are applied."""

        @particle_emitter(max_particles=2000, simulation="cpu")
        class EmitterTest:
            pass

        assert EmitterTest._tags["particle_emitter"] is True
        assert EmitterTest._tags["particle_max_particles"] == 2000
        assert EmitterTest._tags["particle_simulation"] == "cpu"

    def test_registry_registration(self):
        """Test decorator is registered in registry."""
        spec = registry.get("particle_emitter")
        assert spec is not None
        assert spec.name == "particle_emitter"
        assert spec.tier == Tier.PARTICLES_VFX

    def test_applied_decorators_tracking(self):
        """Test decorator application is tracked."""

        @particle_emitter()
        class EmitterTest:
            pass

        assert hasattr(EmitterTest, "_applied_decorators")
        assert "particle_emitter" in EmitterTest._applied_decorators


class TestParticleModule:
    """Test @particle_module decorator."""

    def test_basic_application(self):
        """Test basic decorator application."""

        @particle_module(stage="spawn")
        class SpawnModule:
            pass

        assert hasattr(SpawnModule, "_particle_module")
        assert SpawnModule._particle_module is True
        assert SpawnModule._particle_module_stage == "spawn"
        assert SpawnModule._particle_module_lod_range == (0, 3)

    def test_custom_lod_range(self):
        """Test custom LOD range."""

        @particle_module(stage="update", lod_range=(1, 2))
        class UpdateModule:
            pass

        assert UpdateModule._particle_module_stage == "update"
        assert UpdateModule._particle_module_lod_range == (1, 2)

    def test_all_stages(self):
        """Test all valid stages."""
        for stage in VALID_PARTICLE_STAGES:

            @particle_module(stage=stage)
            class ModuleTest:
                pass

            assert ModuleTest._particle_module_stage == stage

    def test_invalid_stage(self):
        """Test invalid stage raises ValueError."""
        with pytest.raises(ValueError, match="Invalid stage"):

            @particle_module(stage="invalid")
            class ModuleBad:
                pass

    def test_invalid_lod_range_type(self):
        """Test invalid LOD range type raises ValueError."""
        with pytest.raises(ValueError, match="lod_range must be a tuple"):

            @particle_module(stage="spawn", lod_range=[0, 3])
            class ModuleBad:
                pass

    def test_invalid_lod_range_order(self):
        """Test invalid LOD range order raises ValueError."""
        with pytest.raises(ValueError, match="lod_range\\[0\\] must be <= lod_range\\[1\\]"):

            @particle_module(stage="spawn", lod_range=(3, 1))
            class ModuleBad:
                pass

    def test_tags(self):
        """Test that tags are applied."""

        @particle_module(stage="render", lod_range=(0, 2))
        class ModuleTest:
            pass

        assert ModuleTest._tags["particle_module"] is True
        assert ModuleTest._tags["particle_module_stage"] == "render"
        assert ModuleTest._tags["particle_module_lod_range"] == (0, 2)

    def test_registry_registration(self):
        """Test decorator is registered in registry."""
        spec = registry.get("particle_module")
        assert spec is not None
        assert spec.tier == Tier.PARTICLES_VFX


class TestVFXEvent:
    """Test @vfx_event decorator."""

    def test_basic_application(self):
        """Test basic decorator application."""

        @vfx_event(trigger="spawn")
        class SpawnVFX:
            pass

        assert hasattr(SpawnVFX, "_vfx_event")
        assert SpawnVFX._vfx_event is True
        assert SpawnVFX._vfx_event_trigger == "spawn"

    def test_all_triggers(self):
        """Test all valid triggers."""
        for trigger in VALID_VFX_TRIGGERS:

            @vfx_event(trigger=trigger)
            class VFXTest:
                pass

            assert VFXTest._vfx_event_trigger == trigger

    def test_invalid_trigger(self):
        """Test invalid trigger raises ValueError."""
        with pytest.raises(ValueError, match="Invalid trigger"):

            @vfx_event(trigger="invalid")
            class VFXBad:
                pass

    def test_tags(self):
        """Test that tags are applied."""

        @vfx_event(trigger="collision")
        class VFXTest:
            pass

        assert VFXTest._tags["vfx_event"] is True
        assert VFXTest._tags["vfx_event_trigger"] == "collision"

    def test_registry_registration(self):
        """Test decorator is registered in registry."""
        spec = registry.get("vfx_event")
        assert spec is not None
        assert spec.tier == Tier.PARTICLES_VFX


class TestGPUParticle:
    """Test @gpu_particle decorator."""

    def test_basic_application(self):
        """Test basic decorator application."""

        @gpu_particle(attributes=["position", "velocity"])
        class GPUParticleSystem:
            pass

        assert hasattr(GPUParticleSystem, "_gpu_particle")
        assert GPUParticleSystem._gpu_particle is True
        assert GPUParticleSystem._gpu_particle_attributes == ["position", "velocity"]
        assert GPUParticleSystem._gpu_particle_compute_shader is None

    def test_with_compute_shader(self):
        """Test with compute shader."""

        @gpu_particle(
            attributes=["position", "velocity", "color"],
            compute_shader="particle_update.comp",
        )
        class CustomGPUParticle:
            pass

        assert CustomGPUParticle._gpu_particle_attributes == [
            "position",
            "velocity",
            "color",
        ]
        assert CustomGPUParticle._gpu_particle_compute_shader == "particle_update.comp"

    def test_empty_attributes_validation(self):
        """Test empty attributes raises ValueError."""
        with pytest.raises(ValueError, match="attributes must be a non-empty list"):

            @gpu_particle(attributes=[])
            class GPUBad:
                pass

    def test_non_list_attributes_validation(self):
        """Test non-list attributes raises ValueError."""
        with pytest.raises(ValueError, match="attributes must be a non-empty list"):

            @gpu_particle(attributes="position")
            class GPUBad:
                pass

    def test_tags(self):
        """Test that tags are applied."""

        @gpu_particle(attributes=["pos", "vel"], compute_shader="test.comp")
        class GPUTest:
            pass

        assert GPUTest._tags["gpu_particle"] is True
        assert GPUTest._tags["gpu_particle_attributes"] == ["pos", "vel"]
        assert GPUTest._tags["gpu_particle_compute_shader"] == "test.comp"

    def test_registry_registration(self):
        """Test decorator is registered in registry."""
        spec = registry.get("gpu_particle")
        assert spec is not None
        assert spec.tier == Tier.PARTICLES_VFX


class TestTrail:
    """Test @trail decorator."""

    def test_basic_application(self):
        """Test basic decorator application."""

        @trail()
        class BasicTrail:
            pass

        assert hasattr(BasicTrail, "_trail")
        assert BasicTrail._trail is True
        assert BasicTrail._trail_width == 0.1
        assert BasicTrail._trail_fade_time == 1.0
        assert BasicTrail._trail_texture_mode == "stretch"

    def test_custom_params(self):
        """Test custom parameters."""

        @trail(width=0.5, fade_time=2.0, texture_mode="tile")
        class CustomTrail:
            pass

        assert CustomTrail._trail_width == 0.5
        assert CustomTrail._trail_fade_time == 2.0
        assert CustomTrail._trail_texture_mode == "tile"

    def test_all_texture_modes(self):
        """Test all valid texture modes."""
        for mode in VALID_TEXTURE_MODES:

            @trail(texture_mode=mode)
            class TrailTest:
                pass

            assert TrailTest._trail_texture_mode == mode

    def test_invalid_width(self):
        """Test invalid width raises ValueError."""
        with pytest.raises(ValueError, match="width must be > 0"):

            @trail(width=0)
            class TrailBad:
                pass

        with pytest.raises(ValueError, match="width must be > 0"):

            @trail(width=-0.1)
            class TrailBad2:
                pass

    def test_invalid_fade_time(self):
        """Test invalid fade_time raises ValueError."""
        with pytest.raises(ValueError, match="fade_time must be > 0"):

            @trail(fade_time=0)
            class TrailBad:
                pass

    def test_invalid_texture_mode(self):
        """Test invalid texture_mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid texture_mode"):

            @trail(texture_mode="invalid")
            class TrailBad:
                pass

    def test_tags(self):
        """Test that tags are applied."""

        @trail(width=0.2, fade_time=1.5, texture_mode="stretch")
        class TrailTest:
            pass

        assert TrailTest._tags["trail"] is True
        assert TrailTest._tags["trail_width"] == 0.2
        assert TrailTest._tags["trail_fade_time"] == 1.5
        assert TrailTest._tags["trail_texture_mode"] == "stretch"

    def test_registry_registration(self):
        """Test decorator is registered in registry."""
        spec = registry.get("trail")
        assert spec is not None
        assert spec.tier == Tier.PARTICLES_VFX


class TestDecal:
    """Test @decal decorator."""

    def test_basic_application(self):
        """Test basic decorator application."""

        @decal()
        class BasicDecal:
            pass

        assert hasattr(BasicDecal, "_decal")
        assert BasicDecal._decal is True
        assert BasicDecal._decal_lifetime is None
        assert BasicDecal._decal_fade_time == 1.0
        assert BasicDecal._decal_channel == 0

    def test_custom_params(self):
        """Test custom parameters."""

        @decal(lifetime=5.0, fade_time=0.5, channel=2)
        class CustomDecal:
            pass

        assert CustomDecal._decal_lifetime == 5.0
        assert CustomDecal._decal_fade_time == 0.5
        assert CustomDecal._decal_channel == 2

    def test_none_lifetime(self):
        """Test None lifetime is allowed."""

        @decal(lifetime=None, fade_time=2.0)
        class PermanentDecal:
            pass

        assert PermanentDecal._decal_lifetime is None
        assert PermanentDecal._decal_fade_time == 2.0

    def test_invalid_lifetime(self):
        """Test invalid lifetime raises ValueError."""
        with pytest.raises(ValueError, match="lifetime must be > 0 or None"):

            @decal(lifetime=0)
            class DecalBad:
                pass

        with pytest.raises(ValueError, match="lifetime must be > 0 or None"):

            @decal(lifetime=-1.0)
            class DecalBad2:
                pass

    def test_invalid_fade_time(self):
        """Test invalid fade_time raises ValueError."""
        with pytest.raises(ValueError, match="fade_time must be >= 0"):

            @decal(fade_time=-0.1)
            class DecalBad:
                pass

    def test_zero_fade_time(self):
        """Test zero fade time is allowed."""

        @decal(fade_time=0.0)
        class InstantDecal:
            pass

        assert InstantDecal._decal_fade_time == 0.0

    def test_invalid_channel(self):
        """Test invalid channel raises ValueError."""
        with pytest.raises(ValueError, match="channel must be >= 0"):

            @decal(channel=-1)
            class DecalBad:
                pass

    def test_tags(self):
        """Test that tags are applied."""

        @decal(lifetime=3.0, fade_time=0.8, channel=1)
        class DecalTest:
            pass

        assert DecalTest._tags["decal"] is True
        assert DecalTest._tags["decal_lifetime"] == 3.0
        assert DecalTest._tags["decal_fade_time"] == 0.8
        assert DecalTest._tags["decal_channel"] == 1

    def test_registry_registration(self):
        """Test decorator is registered in registry."""
        spec = registry.get("decal")
        assert spec is not None
        assert spec.tier == Tier.PARTICLES_VFX


class TestComposition:
    """Test decorator composition."""

    def test_emitter_with_modules(self):
        """Test particle emitter with multiple modules."""

        @particle_module(stage="render")
        @particle_module(stage="update")
        @particle_module(stage="spawn")
        @particle_emitter(max_particles=2000, simulation="gpu")
        class ComplexParticleSystem:
            pass

        # All decorators applied
        assert ComplexParticleSystem._particle_emitter is True
        assert ComplexParticleSystem._particle_module is True

        # Parameters preserved
        assert ComplexParticleSystem._particle_max_particles == 2000
        assert ComplexParticleSystem._particle_simulation == "gpu"

    def test_gpu_particle_with_trail(self):
        """Test combining GPU particles with trail."""

        @trail(width=0.3, fade_time=2.0)
        @gpu_particle(attributes=["position", "velocity"])
        class GPUTrail:
            pass

        assert GPUTrail._gpu_particle is True
        assert GPUTrail._trail is True
        assert GPUTrail._gpu_particle_attributes == ["position", "velocity"]
        assert GPUTrail._trail_width == 0.3

    def test_vfx_event_with_decal(self):
        """Test VFX event triggering decals."""

        @decal(lifetime=2.0, channel=1)
        @vfx_event(trigger="collision")
        class ImpactDecal:
            pass

        assert ImpactDecal._vfx_event is True
        assert ImpactDecal._decal is True
        assert ImpactDecal._vfx_event_trigger == "collision"
        assert ImpactDecal._decal_lifetime == 2.0


class TestRegistryIntegration:
    """Test registry integration for all decorators."""

    def test_all_decorators_registered(self):
        """Test all decorators are registered in tier 45."""
        tier_specs = registry.by_tier(Tier.PARTICLES_VFX)
        decorator_names = {spec.name for spec in tier_specs}

        expected = {
            "particle_emitter",
            "particle_module",
            "vfx_event",
            "gpu_particle",
            "trail",
            "decal",
        }

        assert expected.issubset(decorator_names)

    def test_decorator_metadata(self):
        """Test decorator metadata is correct."""
        for name in [
            "particle_emitter",
            "particle_module",
            "vfx_event",
            "gpu_particle",
            "trail",
            "decal",
        ]:
            spec = registry.get(name)
            assert spec is not None
            assert spec.tier == Tier.PARTICLES_VFX
            assert spec.foundation is False
            assert "class" in spec.target_types
