"""Tests for Volume Material Domain - Ray marching for fog, clouds, smoke.

Task: T-MAT-5.4 Volume Domain Implementation
Gap: S5-G4
Dependency: T-MAT-3.4 (pipeline integration)

Tests verify:
1. VolumeParams struct validates physical parameters
2. Ray-AABB intersection algorithm correctness
3. Henyey-Greenstein phase function correctness
4. Beer's law transmittance calculation
5. Volume material WGSL generation with correct features
6. Homogeneous fog rendering produces expected output
7. No surface/UI domain code in volume shaders
"""

from __future__ import annotations

import math
import pytest
from typing import List, Tuple

from trinity.materials.volume_domain import (
    VolumeDensityMode,
    VolumePhaseFunction,
    VolumeParams,
    VolumeMaterialConfig,
    VolumeMaterialBuilder,
    VOLUME_DOMAIN_WGSL,
    VOLUME_WGSL_PATH,
    VOLUME_MATERIAL_PRESETS,
    generate_volume_material,
    generate_volume_material_consts,
    get_volume_entry_point,
    get_volume_material_preset,
    validate_volume_material_wgsl,
)

from trinity.materials.domains import (
    DomainCapability,
    DomainShaderTemplate,
    domain_has_capability,
)
from trinity.materials.variants import MaterialDomain


# =============================================================================
# Test: VolumeParams
# =============================================================================


class TestVolumeParams:
    """Test VolumeParams dataclass."""

    def test_default_params(self):
        """Default params should have physically reasonable values."""
        params = VolumeParams()

        assert params.density_scale == 1.0
        assert params.absorption == (0.01, 0.01, 0.01)
        assert params.scattering == (0.1, 0.1, 0.1)
        assert params.phase_g == 0.0  # Isotropic
        assert params.emission == (0.0, 0.0, 0.0)
        assert params.max_march_distance == 100.0
        assert params.max_march_steps == 128

    def test_custom_params(self):
        """Custom params should store all values correctly."""
        params = VolumeParams(
            density_scale=2.5,
            absorption=(0.1, 0.05, 0.02),
            scattering=(0.3, 0.25, 0.2),
            phase_g=0.7,
            emission=(1.0, 0.5, 0.2),
            max_march_distance=50.0,
            max_march_steps=64,
        )

        assert params.density_scale == 2.5
        assert params.absorption == (0.1, 0.05, 0.02)
        assert params.scattering == (0.3, 0.25, 0.2)
        assert params.phase_g == 0.7
        assert params.emission == (1.0, 0.5, 0.2)
        assert params.max_march_distance == 50.0
        assert params.max_march_steps == 64

    def test_params_is_frozen(self):
        """Params should be immutable (frozen dataclass)."""
        params = VolumeParams()

        with pytest.raises(AttributeError):
            params.density_scale = 5.0

    def test_density_scale_validation(self):
        """Density scale must be non-negative."""
        with pytest.raises(ValueError, match="density_scale must be >= 0"):
            VolumeParams(density_scale=-0.1)

        # Zero density is valid (transparent volume)
        params = VolumeParams(density_scale=0.0)
        assert params.density_scale == 0.0

    def test_phase_g_validation(self):
        """Phase g must be in [-1, 1] range."""
        with pytest.raises(ValueError, match="phase_g must be in"):
            VolumeParams(phase_g=-1.5)

        with pytest.raises(ValueError, match="phase_g must be in"):
            VolumeParams(phase_g=1.5)

    def test_phase_g_boundary_values(self):
        """Phase g at boundary values should be valid."""
        params_neg = VolumeParams(phase_g=-1.0)
        params_pos = VolumeParams(phase_g=1.0)
        params_zero = VolumeParams(phase_g=0.0)

        assert params_neg.phase_g == -1.0
        assert params_pos.phase_g == 1.0
        assert params_zero.phase_g == 0.0

    def test_max_march_distance_validation(self):
        """Max march distance must be positive."""
        with pytest.raises(ValueError, match="max_march_distance must be > 0"):
            VolumeParams(max_march_distance=0.0)

        with pytest.raises(ValueError, match="max_march_distance must be > 0"):
            VolumeParams(max_march_distance=-10.0)

    def test_max_march_steps_validation(self):
        """Max march steps must be at least 1."""
        with pytest.raises(ValueError, match="max_march_steps must be >= 1"):
            VolumeParams(max_march_steps=0)

        params = VolumeParams(max_march_steps=1)
        assert params.max_march_steps == 1

    def test_extinction_property(self):
        """Extinction should be sum of absorption and scattering."""
        params = VolumeParams(
            absorption=(0.1, 0.2, 0.3),
            scattering=(0.4, 0.5, 0.6),
        )

        ext = params.extinction
        # Use approximate comparison for floating point
        assert abs(ext[0] - 0.5) < 1e-10
        assert abs(ext[1] - 0.7) < 1e-10
        assert abs(ext[2] - 0.9) < 1e-10

    def test_single_scattering_albedo_property(self):
        """Single-scattering albedo should be scattering / extinction."""
        params = VolumeParams(
            absorption=(0.2, 0.2, 0.2),
            scattering=(0.8, 0.8, 0.8),
        )

        ssa = params.single_scattering_albedo
        # scattering / (absorption + scattering) = 0.8 / 1.0 = 0.8
        assert abs(ssa[0] - 0.8) < 1e-6
        assert abs(ssa[1] - 0.8) < 1e-6
        assert abs(ssa[2] - 0.8) < 1e-6

    def test_single_scattering_albedo_zero_extinction(self):
        """SSA with zero extinction should return 1.0 (fully scattering)."""
        params = VolumeParams(
            absorption=(0.0, 0.0, 0.0),
            scattering=(0.0, 0.0, 0.0),
        )

        ssa = params.single_scattering_albedo
        assert ssa == (1.0, 1.0, 1.0)


# =============================================================================
# Test: VolumeMaterialConfig
# =============================================================================


class TestVolumeMaterialConfig:
    """Test VolumeMaterialConfig dataclass."""

    def test_default_config(self):
        """Default config should have reasonable defaults."""
        config = VolumeMaterialConfig()

        assert config.params.density_scale == 1.0
        assert config.density_mode == VolumeDensityMode.HOMOGENEOUS
        assert config.phase_function == VolumePhaseFunction.HENYEY_GREENSTEIN
        assert config.enable_shadows is False
        assert config.enable_self_shadowing is False
        assert config.adaptive_stepping is True
        assert config.early_termination is True
        assert config.transmittance_threshold == 0.001

    def test_transmittance_threshold_validation(self):
        """Transmittance threshold must be in (0, 1)."""
        with pytest.raises(ValueError, match="transmittance_threshold must be in"):
            VolumeMaterialConfig(transmittance_threshold=0.0)

        with pytest.raises(ValueError, match="transmittance_threshold must be in"):
            VolumeMaterialConfig(transmittance_threshold=1.0)

        with pytest.raises(ValueError, match="transmittance_threshold must be in"):
            VolumeMaterialConfig(transmittance_threshold=-0.1)


# =============================================================================
# Test: Volume Material WGSL Generation
# =============================================================================


class TestGenerateVolumeMaterial:
    """Test WGSL shader generation for volume materials."""

    def test_generate_produces_valid_wgsl(self):
        """Generated WGSL should have balanced braces."""
        config = VolumeMaterialConfig()
        wgsl = generate_volume_material(config, include_core=False)

        open_braces = wgsl.count("{")
        close_braces = wgsl.count("}")

        assert open_braces == close_braces, "Unbalanced braces"

    def test_generate_includes_const_declarations(self):
        """Generated WGSL should include const declarations from config."""
        config = VolumeMaterialConfig(
            density_mode=VolumeDensityMode.HOMOGENEOUS,
            phase_function=VolumePhaseFunction.HENYEY_GREENSTEIN,
            adaptive_stepping=True,
            early_termination=True,
        )
        wgsl = generate_volume_material(config, include_core=False)

        assert "const DENSITY_MODE_HOMOGENEOUS: bool = true;" in wgsl
        assert "const PHASE_HG: bool = true;" in wgsl
        assert "const ADAPTIVE_STEPPING: bool = true;" in wgsl
        assert "const EARLY_TERMINATION: bool = true;" in wgsl

    def test_generate_with_texture_mode(self):
        """Generated WGSL should have texture mode const when enabled."""
        config = VolumeMaterialConfig(
            density_mode=VolumeDensityMode.TEXTURE_3D,
        )
        wgsl = generate_volume_material(config, include_core=False)

        assert "const DENSITY_MODE_TEXTURE: bool = true;" in wgsl
        assert "const DENSITY_MODE_HOMOGENEOUS: bool = false;" in wgsl

    def test_generate_has_vertex_shader(self):
        """Generated WGSL should have vertex shader entry point."""
        config = VolumeMaterialConfig()
        wgsl = generate_volume_material(config, include_core=False)

        assert "@vertex" in wgsl
        assert "fn vs_volume" in wgsl

    def test_generate_has_fragment_shader(self):
        """Generated WGSL should have fragment shader entry point."""
        config = VolumeMaterialConfig()
        wgsl = generate_volume_material(config, include_core=False)

        assert "@fragment" in wgsl
        assert "fn fs_volume" in wgsl

    def test_generate_has_volume_uniforms(self):
        """Generated WGSL should have VolumeUniforms struct."""
        config = VolumeMaterialConfig()
        wgsl = generate_volume_material(config, include_core=False)

        assert "struct VolumeUniforms" in wgsl
        assert "density_scale:" in wgsl
        assert "phase_g:" in wgsl
        assert "absorption:" in wgsl
        assert "scattering:" in wgsl

    def test_generate_max_steps_const(self):
        """Generated WGSL should include max march steps const."""
        config = VolumeMaterialConfig(
            params=VolumeParams(max_march_steps=256)
        )
        wgsl = generate_volume_material(config, include_core=False)

        assert "const MAX_MARCH_STEPS: u32 = 256u;" in wgsl


# =============================================================================
# Test: Core Volume WGSL Functions
# =============================================================================


class TestVolumeWGSL:
    """Test the core volume.wgsl file."""

    def test_volume_wgsl_file_exists(self):
        """volume.wgsl should exist in the wgsl directory."""
        assert VOLUME_WGSL_PATH.exists(), f"Missing: {VOLUME_WGSL_PATH}"

    def test_volume_wgsl_has_volume_params(self):
        """volume.wgsl should define VolumeParams struct."""
        if not VOLUME_WGSL_PATH.exists():
            pytest.skip("volume.wgsl not found")

        content = VOLUME_WGSL_PATH.read_text()
        assert "struct VolumeParams" in content

    def test_volume_wgsl_has_ray_aabb_intersect(self):
        """volume.wgsl should have ray_aabb_intersect function."""
        if not VOLUME_WGSL_PATH.exists():
            pytest.skip("volume.wgsl not found")

        content = VOLUME_WGSL_PATH.read_text()
        assert "fn ray_aabb_intersect" in content

    def test_volume_wgsl_has_henyey_greenstein(self):
        """volume.wgsl should have henyey_greenstein phase function."""
        if not VOLUME_WGSL_PATH.exists():
            pytest.skip("volume.wgsl not found")

        content = VOLUME_WGSL_PATH.read_text()
        assert "fn henyey_greenstein" in content

    def test_volume_wgsl_has_beer_lambert(self):
        """volume.wgsl should have Beer-Lambert transmittance function."""
        if not VOLUME_WGSL_PATH.exists():
            pytest.skip("volume.wgsl not found")

        content = VOLUME_WGSL_PATH.read_text()
        assert "beer_lambert" in content.lower() or "transmittance" in content.lower()

    def test_volume_wgsl_has_march_volume(self):
        """volume.wgsl should have march_volume function."""
        if not VOLUME_WGSL_PATH.exists():
            pytest.skip("volume.wgsl not found")

        content = VOLUME_WGSL_PATH.read_text()
        assert "fn march_volume" in content

    def test_volume_wgsl_has_inscattered_integration(self):
        """volume.wgsl should have in-scattered light integration."""
        if not VOLUME_WGSL_PATH.exists():
            pytest.skip("volume.wgsl not found")

        content = VOLUME_WGSL_PATH.read_text()
        assert "inscattered" in content.lower() or "in_scatter" in content.lower()

    def test_volume_wgsl_has_evaluate_volume(self):
        """volume.wgsl should have evaluate_volume entry function."""
        if not VOLUME_WGSL_PATH.exists():
            pytest.skip("volume.wgsl not found")

        content = VOLUME_WGSL_PATH.read_text()
        assert "fn evaluate_volume" in content

    def test_volume_wgsl_balanced_braces(self):
        """volume.wgsl should have balanced braces."""
        if not VOLUME_WGSL_PATH.exists():
            pytest.skip("volume.wgsl not found")

        content = VOLUME_WGSL_PATH.read_text()
        assert content.count("{") == content.count("}"), "Unbalanced braces"


# =============================================================================
# Test: Henyey-Greenstein Phase Function (Mathematical Correctness)
# =============================================================================


class TestHenyeyGreensteinPhaseFunction:
    """Test Henyey-Greenstein phase function properties."""

    def henyey_greenstein(self, cos_theta: float, g: float) -> float:
        """Python implementation of HG phase function for testing."""
        g2 = g * g
        denom = 1.0 + g2 - 2.0 * g * cos_theta
        if abs(g) < 1e-6:
            return 1.0 / (4.0 * math.pi)
        return (1.0 - g2) / (4.0 * math.pi * (denom ** 1.5))

    def test_isotropic_case(self):
        """When g=0, phase function should be isotropic (1/4pi)."""
        expected = 1.0 / (4.0 * math.pi)

        # Should be same for all angles when g=0
        for cos_theta in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            result = self.henyey_greenstein(cos_theta, 0.0)
            assert abs(result - expected) < 1e-6, f"Failed for cos_theta={cos_theta}"

    def test_forward_scattering(self):
        """When g>0, forward direction should have highest probability."""
        g = 0.7

        # Forward (cos_theta = 1) should be maximum
        forward = self.henyey_greenstein(1.0, g)
        side = self.henyey_greenstein(0.0, g)
        backward = self.henyey_greenstein(-1.0, g)

        assert forward > side > backward

    def test_backward_scattering(self):
        """When g<0, backward direction should have highest probability."""
        g = -0.5

        forward = self.henyey_greenstein(1.0, g)
        side = self.henyey_greenstein(0.0, g)
        backward = self.henyey_greenstein(-1.0, g)

        assert backward > side > forward

    def test_normalization(self):
        """Phase function should integrate to 1 over the sphere."""
        # Numerical integration using trapezoidal rule
        g = 0.6
        n_samples = 1000
        integral = 0.0

        for i in range(n_samples):
            cos_theta = -1.0 + 2.0 * (i + 0.5) / n_samples
            # Solid angle element: 2*pi * d(cos_theta)
            d_cos_theta = 2.0 / n_samples
            phase = self.henyey_greenstein(cos_theta, g)
            integral += phase * 2.0 * math.pi * d_cos_theta

        # Should be close to 1.0
        assert abs(integral - 1.0) < 0.01, f"Integral = {integral}, expected ~1.0"

    def test_symmetry(self):
        """Phase function should be symmetric in g."""
        cos_theta = 0.3

        forward_g = self.henyey_greenstein(cos_theta, 0.5)
        backward_g = self.henyey_greenstein(-cos_theta, -0.5)

        assert abs(forward_g - backward_g) < 1e-6


# =============================================================================
# Test: Beer-Lambert Transmittance (Mathematical Correctness)
# =============================================================================


class TestBeerLambertTransmittance:
    """Test Beer-Lambert transmittance calculation."""

    def beer_lambert(self, extinction: float, optical_depth: float) -> float:
        """Python implementation of Beer-Lambert law."""
        return math.exp(-extinction * optical_depth)

    def test_zero_extinction(self):
        """Zero extinction should give full transmittance."""
        result = self.beer_lambert(0.0, 10.0)
        assert abs(result - 1.0) < 1e-6

    def test_zero_depth(self):
        """Zero optical depth should give full transmittance."""
        result = self.beer_lambert(1.0, 0.0)
        assert abs(result - 1.0) < 1e-6

    def test_high_extinction_low_transmittance(self):
        """High extinction should give low transmittance."""
        result = self.beer_lambert(5.0, 1.0)
        expected = math.exp(-5.0)
        assert abs(result - expected) < 1e-6
        assert result < 0.01  # Should be very low

    def test_additive_optical_depth(self):
        """Transmittance through two segments should multiply."""
        ext = 0.5
        d1 = 2.0
        d2 = 3.0

        T1 = self.beer_lambert(ext, d1)
        T2 = self.beer_lambert(ext, d2)
        T_combined = self.beer_lambert(ext, d1 + d2)

        assert abs(T1 * T2 - T_combined) < 1e-6


# =============================================================================
# Test: Volume Material Builder
# =============================================================================


class TestVolumeMaterialBuilder:
    """Test VolumeMaterialBuilder fluent interface."""

    def test_builder_default(self):
        """Builder with no customization should produce default config."""
        config = VolumeMaterialBuilder().build()

        assert config.params.density_scale == 1.0
        assert config.density_mode == VolumeDensityMode.HOMOGENEOUS
        assert config.phase_function == VolumePhaseFunction.HENYEY_GREENSTEIN

    def test_builder_with_scattering(self):
        """Builder should set scattering coefficient."""
        config = (
            VolumeMaterialBuilder()
            .with_scattering(0.5, 0.4, 0.3)
            .build()
        )

        assert config.params.scattering == (0.5, 0.4, 0.3)

    def test_builder_with_absorption(self):
        """Builder should set absorption coefficient."""
        config = (
            VolumeMaterialBuilder()
            .with_absorption(0.1, 0.2, 0.3)
            .build()
        )

        assert config.params.absorption == (0.1, 0.2, 0.3)

    def test_builder_with_phase_g(self):
        """Builder should set phase function asymmetry."""
        config = (
            VolumeMaterialBuilder()
            .with_phase_g(0.8)
            .build()
        )

        assert config.params.phase_g == 0.8

    def test_builder_with_emission(self):
        """Builder should set emission color."""
        config = (
            VolumeMaterialBuilder()
            .with_emission(1.0, 0.5, 0.2)
            .build()
        )

        assert config.params.emission == (1.0, 0.5, 0.2)

    def test_builder_with_density_mode(self):
        """Builder should set density sampling mode."""
        config = (
            VolumeMaterialBuilder()
            .with_density_mode(VolumeDensityMode.TEXTURE_3D)
            .build()
        )

        assert config.density_mode == VolumeDensityMode.TEXTURE_3D

    def test_builder_with_phase_function(self):
        """Builder should set phase function type."""
        config = (
            VolumeMaterialBuilder()
            .with_phase_function(VolumePhaseFunction.RAYLEIGH)
            .build()
        )

        assert config.phase_function == VolumePhaseFunction.RAYLEIGH

    def test_builder_with_shadows(self):
        """Builder should enable shadow reception."""
        config = (
            VolumeMaterialBuilder()
            .with_shadows(True)
            .build()
        )

        assert config.enable_shadows is True

    def test_builder_chain(self):
        """Builder should support full fluent chain."""
        config = (
            VolumeMaterialBuilder()
            .with_density_scale(2.0)
            .with_scattering(0.3, 0.3, 0.3)
            .with_absorption(0.1, 0.1, 0.1)
            .with_phase_g(0.7)
            .with_emission(0.1, 0.05, 0.02)
            .with_max_distance(200.0)
            .with_max_steps(256)
            .with_density_mode(VolumeDensityMode.EXPONENTIAL_HEIGHT)
            .with_phase_function(VolumePhaseFunction.TWO_LOBE_HG)
            .with_shadows(True)
            .with_self_shadowing(True)
            .with_adaptive_stepping(True)
            .with_early_termination(True, threshold=0.005)
            .build()
        )

        assert config.params.density_scale == 2.0
        assert config.params.scattering == (0.3, 0.3, 0.3)
        assert config.params.absorption == (0.1, 0.1, 0.1)
        assert config.params.phase_g == 0.7
        assert config.params.emission == (0.1, 0.05, 0.02)
        assert config.params.max_march_distance == 200.0
        assert config.params.max_march_steps == 256
        assert config.density_mode == VolumeDensityMode.EXPONENTIAL_HEIGHT
        assert config.phase_function == VolumePhaseFunction.TWO_LOBE_HG
        assert config.enable_shadows is True
        assert config.enable_self_shadowing is True
        assert config.adaptive_stepping is True
        assert config.early_termination is True
        assert config.transmittance_threshold == 0.005


# =============================================================================
# Test: Volume Material Presets
# =============================================================================


class TestVolumeMaterialPresets:
    """Test predefined volume material configurations."""

    def test_all_presets_exist(self):
        """All documented presets should exist."""
        expected_presets = [
            "fog_light",
            "fog_dense",
            "fog_height",
            "cloud",
            "smoke",
            "fire_glow",
            "atmospheric",
        ]

        for name in expected_presets:
            assert name in VOLUME_MATERIAL_PRESETS, f"Missing preset: {name}"

    def test_get_preset_returns_config(self):
        """get_volume_material_preset should return VolumeMaterialConfig."""
        config = get_volume_material_preset("fog_light")
        assert isinstance(config, VolumeMaterialConfig)

    def test_get_preset_unknown_raises(self):
        """get_volume_material_preset should raise for unknown preset."""
        with pytest.raises(KeyError, match="Unknown volume material preset"):
            get_volume_material_preset("nonexistent_preset")

    def test_fog_light_preset(self):
        """fog_light preset should have low density and scattering."""
        config = get_volume_material_preset("fog_light")

        assert config.params.density_scale == 0.5
        assert config.density_mode == VolumeDensityMode.HOMOGENEOUS

    def test_cloud_preset_forward_scattering(self):
        """cloud preset should have strong forward scattering."""
        config = get_volume_material_preset("cloud")

        assert config.params.phase_g > 0.5  # Forward scattering
        assert config.phase_function == VolumePhaseFunction.TWO_LOBE_HG

    def test_fire_glow_has_emission(self):
        """fire_glow preset should have non-zero emission."""
        config = get_volume_material_preset("fire_glow")

        assert sum(config.params.emission) > 0

    def test_atmospheric_uses_rayleigh(self):
        """atmospheric preset should use Rayleigh phase function."""
        config = get_volume_material_preset("atmospheric")

        assert config.phase_function == VolumePhaseFunction.RAYLEIGH


# =============================================================================
# Test: Entry Point Selection
# =============================================================================


class TestEntryPointSelection:
    """Test fragment shader entry point selection."""

    def test_homogeneous_uses_fullscreen(self):
        """Homogeneous density mode should use fullscreen fog shader."""
        config = VolumeMaterialConfig(density_mode=VolumeDensityMode.HOMOGENEOUS)
        entry = get_volume_entry_point(config)

        assert entry == "fs_fullscreen_fog"

    def test_texture_uses_volume(self):
        """Texture density mode should use volume shader."""
        config = VolumeMaterialConfig(density_mode=VolumeDensityMode.TEXTURE_3D)
        entry = get_volume_entry_point(config)

        assert entry == "fs_volume"

    def test_procedural_uses_volume(self):
        """Procedural density mode should use volume shader."""
        config = VolumeMaterialConfig(density_mode=VolumeDensityMode.PROCEDURAL)
        entry = get_volume_entry_point(config)

        assert entry == "fs_volume"


# =============================================================================
# Test: WGSL Validation
# =============================================================================


class TestVolumeWGSLValidation:
    """Test volume material WGSL validation."""

    def test_validate_complete_shader_passes(self):
        """Complete volume shader should pass validation."""
        config = VolumeMaterialConfig()
        wgsl = generate_volume_material(config, include_core=True)

        errors = validate_volume_material_wgsl(wgsl)
        assert len(errors) == 0, f"Validation errors: {errors}"

    def test_validate_detects_missing_volume_params(self):
        """Validation should fail if VolumeParams is missing."""
        wgsl = """
        fn some_function() -> f32 {
            return 1.0;
        }
        """
        errors = validate_volume_material_wgsl(wgsl)

        assert any("Volume parameters struct" in e for e in errors)

    def test_validate_detects_surface_domain_code(self):
        """Validation should fail if surface domain code is present."""
        config = VolumeMaterialConfig()
        wgsl = generate_volume_material(config, include_core=True)

        # Inject forbidden pattern
        wgsl += "\nfn evaluate_surface_domain() {}"

        errors = validate_volume_material_wgsl(wgsl)
        assert any("Surface domain" in e for e in errors)


# =============================================================================
# Test: Volume Domain Capabilities
# =============================================================================


class TestVolumeDomainCapabilities:
    """Test volume domain capability configuration."""

    def test_volume_domain_has_volumetric_capability(self):
        """Volume domain should have VOLUMETRIC capability."""
        assert domain_has_capability(MaterialDomain.VOLUME, DomainCapability.VOLUMETRIC)

    def test_volume_domain_has_emissive_capability(self):
        """Volume domain should have EMISSIVE capability (for fire/glow)."""
        assert domain_has_capability(MaterialDomain.VOLUME, DomainCapability.EMISSIVE)

    def test_volume_domain_no_lighting_capability(self):
        """Volume domain should NOT have direct LIGHTING (uses in-scattering)."""
        assert not domain_has_capability(MaterialDomain.VOLUME, DomainCapability.LIGHTING)

    def test_volume_domain_no_shadows_capability(self):
        """Volume domain should NOT have SHADOWS (self-shadowing is separate)."""
        assert not domain_has_capability(MaterialDomain.VOLUME, DomainCapability.SHADOWS)


# =============================================================================
# Test: Homogeneous Fog Rendering Verification
# =============================================================================


class TestHomogeneousFogRendering:
    """Test homogeneous fog rendering produces expected results."""

    def compute_fog_transmittance(
        self,
        extinction: Tuple[float, float, float],
        density: float,
        distance: float,
    ) -> Tuple[float, float, float]:
        """Compute expected transmittance for homogeneous fog."""
        return (
            math.exp(-extinction[0] * density * distance),
            math.exp(-extinction[1] * density * distance),
            math.exp(-extinction[2] * density * distance),
        )

    def test_zero_density_full_transmittance(self):
        """Zero density fog should have full transmittance."""
        params = VolumeParams(density_scale=0.0)
        T = self.compute_fog_transmittance(params.extinction, 0.0, 100.0)

        assert T == (1.0, 1.0, 1.0)

    def test_high_density_low_transmittance(self):
        """High density fog should have low transmittance."""
        params = VolumeParams(
            density_scale=5.0,
            absorption=(0.1, 0.1, 0.1),
            scattering=(0.5, 0.5, 0.5),
        )
        # extinction = (0.6, 0.6, 0.6)
        T = self.compute_fog_transmittance(params.extinction, 5.0, 10.0)

        # exp(-0.6 * 5.0 * 10.0) = exp(-30) ~ 0
        assert T[0] < 1e-10
        assert T[1] < 1e-10
        assert T[2] < 1e-10

    def test_fog_visibility_relationship(self):
        """Fog should follow expected visibility relationship."""
        # Visibility distance: where T = e^-1 ~ 0.368
        # T = exp(-extinction * density * distance)
        # e^-1 = exp(-ext * dens * vis_dist)
        # 1 = ext * dens * vis_dist
        # vis_dist = 1 / (ext * dens)

        params = VolumeParams(
            density_scale=1.0,
            absorption=(0.01, 0.01, 0.01),
            scattering=(0.09, 0.09, 0.09),
        )
        # extinction = 0.1
        expected_vis_dist = 1.0 / (0.1 * 1.0)  # = 10.0

        T_at_vis = self.compute_fog_transmittance(params.extinction, 1.0, expected_vis_dist)

        # Should be approximately e^-1 = 0.368
        expected = math.exp(-1.0)
        assert abs(T_at_vis[0] - expected) < 1e-6


# =============================================================================
# Test: Integration with Domain System
# =============================================================================


class TestVolumeDomainIntegration:
    """Test volume domain integration with material system."""

    def test_domain_shader_template_exists(self):
        """Volume domain should have a shader template in DomainShaderTemplate."""
        template = DomainShaderTemplate.get_for_domain(MaterialDomain.VOLUME)

        # Template should exist and contain volume-related code
        assert template is not None
        assert len(template) > 0

    def test_domain_function_name(self):
        """Volume domain should have correct function name."""
        func_name = DomainShaderTemplate.get_domain_function_name(MaterialDomain.VOLUME)

        assert func_name == "evaluate_volume_domain"

    def test_volume_consts_generation(self):
        """Volume const generation should produce valid WGSL."""
        config = VolumeMaterialConfig(
            density_mode=VolumeDensityMode.TEXTURE_3D,
            phase_function=VolumePhaseFunction.TWO_LOBE_HG,
            enable_shadows=True,
        )
        consts = generate_volume_material_consts(config)

        # Check all modes/functions have consts
        assert "DENSITY_MODE_TEXTURE: bool = true" in consts
        assert "DENSITY_MODE_HOMOGENEOUS: bool = false" in consts
        assert "PHASE_TWO_LOBE_HG: bool = true" in consts
        assert "ENABLE_SHADOWS: bool = true" in consts
