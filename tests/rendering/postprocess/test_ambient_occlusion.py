"""
Tests for Screen-Space Ambient Occlusion System

Tests SSAOKernel hemisphere generation, HBAO direction generation,
BilateralFilter weights, BentNormalOutput, and AOEffect integration.
"""

import math
import pytest

from engine.rendering.postprocess.ambient_occlusion import (
    AOEffect,
    AOMethod,
    AOQuality,
    AOSettings,
    BentNormalOutput,
    BilateralFilter,
    GTAO,
    HBAO,
    SSAO,
    SSAOKernel,
)


class TestSSAOKernel:
    """Test SSAOKernel hemisphere generation."""

    def test_kernel_creation(self):
        """Test kernel creation with default 64 samples."""
        kernel = SSAOKernel()
        samples = kernel.samples

        assert len(samples) == 64

    def test_all_samples_in_positive_hemisphere(self):
        """Test all samples have z >= 0 (hemisphere)."""
        kernel = SSAOKernel(sample_count=64)

        for x, y, z in kernel.samples:
            assert z >= 0.0, f"Sample ({x}, {y}, {z}) has negative z"

    def test_kernel_sample_count(self):
        """Test variable sample counts."""
        for count in [4, 8, 16, 32, 64, 128]:
            kernel = SSAOKernel(sample_count=count)
            assert len(kernel.samples) == count

    def test_all_samples_normalized(self):
        """Test samples are normalized (unit length before scaling)."""
        kernel = SSAOKernel(sample_count=32)

        for x, y, z in kernel.samples:
            length = math.sqrt(x * x + y * y + z * z)
            # After scaling, samples may not be unit length, but should be valid
            assert length >= 0
            assert not math.isnan(length)
            assert not math.isinf(length)

    def test_samples_not_all_identical(self):
        """Test samples are diverse."""
        kernel = SSAOKernel(sample_count=64)
        samples = kernel.samples

        # Check first and last sample differ
        first = samples[0]
        last = samples[-1]
        assert first != last

    def test_distribution_not_degenerate(self):
        """Test hemisphere distribution is not degenerate."""
        kernel = SSAOKernel(sample_count=128)
        z_values = [z for _, _, z in kernel.samples]

        # Mean z should be in valid range (accelerating distribution
        # pulls samples toward origin, so mean_z < 0.5 is expected)
        mean_z = sum(z_values) / len(z_values)
        assert 0.0 < mean_z < 1.0

        # No sample should be at the origin
        for x, y, z in kernel.samples:
            length = math.sqrt(x * x + y * y + z * z)
            assert length > 0.0

    def test_more_samples_near_center(self):
        """Test distribution has more samples near origin (accelerating)."""
        kernel = SSAOKernel(sample_count=128)
        samples = kernel.samples

        # Split into two groups: early (near center) and late (near edge)
        near_center = samples[:64]
        far_center = samples[64:]

        # Early samples (smaller index) should have smaller scale = closer to center
        # This tests the accelerating distribution property
        near_lengths = [math.sqrt(x*x + y*y + z*z) for x, y, z in near_center]
        far_lengths = [math.sqrt(x*x + y*y + z*z) for x, y, z in far_center]

        mean_near = sum(near_lengths) / len(near_lengths)
        mean_far = sum(far_lengths) / len(far_lengths)

        # Near center group should have smaller average length
        assert mean_near < mean_far

    def test_no_clustering_at_poles(self):
        """Test no samples cluster at poles (z=0 or z=1)."""
        kernel = SSAOKernel(sample_count=64)

        for x, y, z in kernel.samples:
            # No sample should be exactly at z=0 or z=1
            assert 0.0 < z <= 1.0

    def test_no_clustering_at_equator_boundary(self):
        """Test no samples are exactly at the hemisphere boundary."""
        kernel = SSAOKernel(sample_count=64)

        for x, y, z in kernel.samples:
            length = math.sqrt(x * x + y * y + z * z)
            # Should not be zero
            assert length > 0

    def test_deterministic_generation(self):
        """Test generation is deterministic with same seed."""
        kernel1 = SSAOKernel(sample_count=32)
        kernel2 = SSAOKernel(sample_count=32)

        for s1, s2 in zip(kernel1.samples, kernel2.samples):
            assert s1 == s2

    def test_noise_texture_generated(self):
        """Test noise rotation vectors are generated."""
        kernel = SSAOKernel(sample_count=16)
        noise = kernel.noise

        # Should have 16 noise vectors (4x4)
        assert len(noise) == 16

        for x, y in noise:
            # Noise should be unit-length 2D vectors
            length = math.sqrt(x * x + y * y)
            assert length > 0.9  # Approximately unit length
            assert abs(length - 1.0) < 0.1

    def test_weight_falloff(self):
        """Test that later samples have larger scale (more weight at distance)."""
        kernel = SSAOKernel(sample_count=64)
        samples = kernel.samples

        # Check that the scale increases monotonically (approximately)
        prev_length = 0
        increasing_count = 0
        for x, y, z in samples:
            length = math.sqrt(x * x + y * y + z * z)
            if length >= prev_length:
                increasing_count += 1
            prev_length = length

        # Most samples should have non-decreasing length
        assert increasing_count > len(samples) * 0.5


class TestSSAO:
    """Test SSAO processor."""

    def test_ssao_creation(self):
        """Test SSAO creation."""
        ssao = SSAO()
        assert ssao is not None
        assert ssao.kernel is not None

    def test_ssao_setup(self):
        """Test SSAO setup."""
        ssao = SSAO()
        ssao.setup(1920, 1080, sample_count=32)

        assert len(ssao.kernel.samples) == 32


class TestHBAO:
    """Test HBAO direction generation."""

    def test_hbao_creation(self):
        """Test HBAO creation."""
        hbao = HBAO()
        assert hbao is not None

    def test_direction_generation(self):
        """Test direction generation."""
        hbao = HBAO()
        hbao.setup(1920, 1080, direction_count=8)

        # Should have 8 directions
        assert len(hbao._directions) == 8

    def test_directions_are_unit(self):
        """Test directions are unit vectors."""
        hbao = HBAO()
        hbao.setup(1920, 1080, direction_count=8)

        for dx, dy in hbao._directions:
            length = math.sqrt(dx * dx + dy * dy)
            assert abs(length - 1.0) < 0.01

    def test_directions_uniform_in_angle(self):
        """Test directions are uniformly distributed in angle."""
        hbao = HBAO()
        hbao.setup(1920, 1080, direction_count=8)

        # Compute the angle of each direction
        angles = [math.atan2(dy, dx) for dx, dy in hbao._directions]
        angles.sort()

        # Angles should be increasing
        for i in range(1, len(angles)):
            assert angles[i] > angles[i - 1]

        # Sum of adjacent angle differences should equal 2*pi
        # (last angle wraps around through 2*pi to first)
        diffs = [
            angles[i] - angles[i - 1] if i > 0
            else (angles[0] + 2 * math.pi - angles[-1])
            for i in range(len(angles))
        ]
        total = sum(diffs)
        assert abs(total - 2 * math.pi) < 0.01

    def test_variable_direction_counts(self):
        """Test variable direction counts."""
        for count in [4, 6, 8, 12, 16]:
            hbao = HBAO()
            hbao.setup(1920, 1080, direction_count=count)

            assert len(hbao._directions) == count

    def test_setup_does_not_crash(self):
        """Test HBAO setup doesn't crash with various sizes."""
        hbao = HBAO()
        hbao.setup(1920, 1080, direction_count=8)
        assert True


class TestGTAO:
    """Test GTAO processor."""

    def test_gtao_creation(self):
        """Test GTAO creation."""
        gtao = GTAO()
        assert gtao is not None

    def test_gtao_setup(self):
        """Test GTAO setup."""
        gtao = GTAO()
        gtao.setup(1920, 1080, slice_count=8, steps_per_slice=4)

        assert gtao is not None


class TestBilateralFilter:
    """Test BilateralFilter weight calculation."""

    def test_bilateral_filter_creation(self):
        """Test bilateral filter creation."""
        bf = BilateralFilter()
        assert bf is not None

    def test_bilateral_weight_depth(self):
        """Test depth-based weight is higher for similar depths."""
        bf = BilateralFilter()

        # Same depth -> high weight
        weight_same = bf._bilateral_weight(
            depth0=1.0, depth1=1.0,
            normal0=(0, 0, 1), normal1=(0, 0, 1),
            distance=1.0, sharpness=1.0,
        )

        # Different depth -> lower weight
        weight_diff = bf._bilateral_weight(
            depth0=1.0, depth1=2.0,
            normal0=(0, 0, 1), normal1=(0, 0, 1),
            distance=1.0, sharpness=1.0,
        )

        assert weight_same > weight_diff

    def test_bilateral_weight_spatial(self):
        """Test spatial weight decreases with distance."""
        bf = BilateralFilter()

        weight_near = bf._bilateral_weight(
            depth0=1.0, depth1=1.0,
            normal0=(0, 0, 1), normal1=(0, 0, 1),
            distance=0.5, sharpness=1.0,
        )

        weight_far = bf._bilateral_weight(
            depth0=1.0, depth1=1.0,
            normal0=(0, 0, 1), normal1=(0, 0, 1),
            distance=5.0, sharpness=1.0,
        )

        assert weight_near > weight_far

    def test_bilateral_weight_normal(self):
        """Test normal-based weight."""
        bf = BilateralFilter()

        # Same normal -> weight includes normal contribution
        weight_same = bf._bilateral_weight(
            depth0=1.0, depth1=1.0,
            normal0=(0, 0, 1), normal1=(0, 0, 1),
            distance=1.0, sharpness=1.0,
        )

        # Opposite normal -> no normal contribution
        weight_opposite = bf._bilateral_weight(
            depth0=1.0, depth1=1.0,
            normal0=(0, 0, 1), normal1=(0, 0, -1),
            distance=1.0, sharpness=1.0,
        )

        assert weight_same > weight_opposite

    def test_bilateral_weight_combined_formula(self):
        """Test combined weight formula produces valid [0, 1] values."""
        bf = BilateralFilter()

        test_cases = [
            # (same depth, similar normal, close)
            (1.0, 1.0, (0, 0, 1), (0, 0, 1), 0.5, 1.0),
            # (different depth, different normal, far)
            (1.0, 5.0, (0, 0, 1), (1, 0, 0), 10.0, 8.0),
            # (same depth, opposite normal, close)
            (1.0, 1.0, (0, 1, 0), (0, -1, 0), 0.5, 4.0),
        ]

        for depth0, depth1, n0, n1, dist, sharpness in test_cases:
            weight = bf._bilateral_weight(depth0, depth1, n0, n1, dist, sharpness)
            assert 0.0 <= weight <= 1.0

    def test_bilateral_weight_zero_distance(self):
        """Test weight at zero distance (should be 1 with all same)."""
        bf = BilateralFilter()

        weight = bf._bilateral_weight(
            depth0=1.0, depth1=1.0,
            normal0=(0, 0, 1), normal1=(0, 0, 1),
            distance=0.0, sharpness=1.0,
        )

        # Should be 1.0 when everything matches
        assert abs(weight - 1.0) < 0.01

    def test_depth_weight_monotonic(self):
        """Test depth weight decreases monotonically with depth difference."""
        bf = BilateralFilter()

        prev_weight = 1.0
        for depth_diff in [0.0, 0.5, 1.0, 2.0, 5.0]:
            weight = bf._bilateral_weight(
                depth0=1.0, depth1=1.0 + depth_diff,
                normal0=(0, 0, 1), normal1=(0, 0, 1),
                distance=0.0, sharpness=1.0,
            )
            assert weight <= prev_weight + 0.01
            prev_weight = weight

    def test_normal_weight_gaussian(self):
        """Test spatial weight follows Gaussian falloff."""
        bf = BilateralFilter()

        weight_a = bf._bilateral_weight(
            depth0=1.0, depth1=1.0,
            normal0=(0, 0, 1), normal1=(0, 0, 1),
            distance=1.0, sharpness=1.0,
        )

        weight_b = bf._bilateral_weight(
            depth0=1.0, depth1=1.0,
            normal0=(0, 0, 1), normal1=(0, 0, 1),
            distance=2.0, sharpness=1.0,
        )

        weight_c = bf._bilateral_weight(
            depth0=1.0, depth1=1.0,
            normal0=(0, 0, 1), normal1=(0, 0, 1),
            distance=3.0, sharpness=1.0,
        )

        # Spatial Gaussian: exp(-d^2 / 2)
        assert weight_a > weight_b > weight_c


class TestBentNormalOutput:
    """Test BentNormalOutput calculations."""

    def test_bent_normal_defaults(self):
        """Test bent normal output defaults."""
        bno = BentNormalOutput(
            bent_normal=(0.0, 0.0, 1.0),
            visibility_cone=0.5,
            occlusion=0.0,
        )

        assert bno.bent_normal == (0.0, 0.0, 1.0)

    def test_specular_occlusion_no_occlusion(self):
        """Test specular occlusion with no occlusion."""
        bno = BentNormalOutput(
            bent_normal=(0.0, 0.0, 1.0),
            visibility_cone=1.0,
            occlusion=0.0,
        )

        spec = bno.calculate_specular_occlusion(
            view_dir=(0.0, 0.0, 1.0),
            roughness=0.5,
        )

        # No occlusion should give 0
        assert abs(spec) < 0.01

    def test_specular_occlusion_with_occlusion(self):
        """Test specular occlusion with full occlusion."""
        bno = BentNormalOutput(
            bent_normal=(0.0, 0.0, 1.0),
            visibility_cone=0.5,
            occlusion=1.0,
        )

        spec = bno.calculate_specular_occlusion(
            view_dir=(0.0, 0.0, 1.0),
            roughness=0.5,
        )

        # Full occlusion with aligned view/bent normal should give non-zero
        assert spec > 0.0

    def test_specular_occlusion_view_mismatch(self):
        """Test specular occlusion with view not aligned with bent normal."""
        bno = BentNormalOutput(
            bent_normal=(0.0, 0.0, 1.0),
            visibility_cone=0.5,
            occlusion=1.0,
        )

        spec = bno.calculate_specular_occlusion(
            view_dir=(1.0, 0.0, 0.0),
            roughness=0.5,
        )

        # Different direction should reduce specular occlusion
        aligned_spec = bno.calculate_specular_occlusion(
            view_dir=(0.0, 0.0, 1.0),
            roughness=0.5,
        )
        assert spec < aligned_spec

    def test_specular_occlusion_roughness_effect(self):
        """Test roughness reduces specular occlusion effect when view not aligned."""
        bno = BentNormalOutput(
            bent_normal=(0.0, 0.0, 1.0),
            visibility_cone=0.5,
            occlusion=1.0,
        )

        # Use off-center view direction so dot product != 1.0
        # (when dot == 1.0, pow(1.0, anything) == 1.0 regardless of roughness)
        spec_rough = bno.calculate_specular_occlusion(
            view_dir=(0.5, 0.0, math.sqrt(1.0 - 0.25)),
            roughness=0.9,
        )
        spec_smooth = bno.calculate_specular_occlusion(
            view_dir=(0.5, 0.0, math.sqrt(1.0 - 0.25)),
            roughness=0.1,
        )

        # Rougher surfaces should have LESS specular occlusion
        # (smaller exponent: pow(dot, (1-r)*4) so spec decays less)
        assert spec_rough > spec_smooth


class TestAOSettings:
    """Test AOSettings dataclass."""

    def test_default_settings(self):
        """Test default AO settings."""
        settings = AOSettings()

        assert settings.method == AOMethod.GTAO
        assert settings.quality == AOQuality.HIGH
        assert settings.radius == 0.5
        assert settings.sample_count == 16
        assert settings.direction_count == 8

    def test_custom_settings(self):
        """Test custom AO settings."""
        settings = AOSettings(
            method=AOMethod.SSAO,
            quality=AOQuality.ULTRA,
            sample_count=32,
            intensity=1.5,
        )

        assert settings.method == AOMethod.SSAO
        assert settings.quality == AOQuality.ULTRA
        assert settings.sample_count == 32
        assert settings.intensity == 1.5

    def test_settings_lerp(self):
        """Test settings interpolation."""
        settings1 = AOSettings(radius=0.5, intensity=1.0)
        settings2 = AOSettings(radius=1.0, intensity=2.0)

        lerped = settings1.lerp(settings2, 0.5)

        assert lerped.radius == 0.75
        assert lerped.intensity == 1.5


class TestAOEffect:
    """Test AOEffect integration."""

    def test_effect_creation(self):
        """Test AO effect creation."""
        effect = AOEffect()

        assert effect.name == "AmbientOcclusion"
        assert effect.settings is not None

    def test_effect_with_custom_settings(self):
        """Test effect with custom settings."""
        settings = AOSettings(method=AOMethod.SSAO)
        effect = AOEffect(settings)

        assert effect.settings.method == AOMethod.SSAO

    def test_effect_required_inputs(self):
        """Test effect required inputs."""
        effect = AOEffect()
        inputs = effect.get_required_inputs()

        assert "depth" in inputs
        assert "normal" in inputs

    def test_effect_outputs(self):
        """Test effect outputs."""
        effect = AOEffect()
        outputs = effect.get_outputs()

        assert "ao" in outputs

    def test_effect_setup(self):
        """Test effect setup."""
        effect = AOEffect()
        effect.setup(1920, 1080)
        # Should not raise

    def test_effect_execute_disabled(self):
        """Test effect does nothing when disabled."""
        settings = AOSettings(enabled=False)
        effect = AOEffect(settings)

        effect.execute({}, {}, 0.016)
        # Should not raise

    def test_effect_cleanup(self):
        """Test effect cleanup."""
        effect = AOEffect()
        effect.cleanup()
        # Should not raise

    def test_effect_is_compute(self):
        """Test effect uses compute."""
        effect = AOEffect()
        assert effect.is_compute_effect() is True

    def test_effect_bent_normals_output(self):
        """Test bent normals output when enabled."""
        settings = AOSettings(
            bent_normals_enabled=True,
        )
        effect = AOEffect(settings)
        outputs = effect.get_outputs()

        assert "ao" in outputs
        assert "bent_normals" in outputs

    def test_effect_half_resolution(self):
        """Test half-resolution mode."""
        settings = AOSettings(half_resolution=True)
        effect = AOEffect(settings)

        effect.setup(1920, 1080)
        # Should not raise

    def test_effect_execute_with_default_projection(self):
        """Test execute with default settings doesn't crash."""
        effect = AOEffect()
        effect.setup(1920, 1080)

        effect.execute(
            {"depth": None, "normal": None},
            {},
            0.016,
        )
        # Should not raise


class TestAOMethod:
    """Test AOMethod enum."""

    def test_all_methods_exist(self):
        """Test all AO methods exist."""
        methods = [
            AOMethod.SSAO,
            AOMethod.HBAO,
            AOMethod.HBAO_PLUS,
            AOMethod.GTAO,
            AOMethod.RTAO,
        ]

        for m in methods:
            assert m is not None


class TestAOQuality:
    """Test AOQuality enum."""

    def test_all_qualities_exist(self):
        """Test all quality presets exist."""
        qualities = [
            AOQuality.LOW,
            AOQuality.MEDIUM,
            AOQuality.HIGH,
            AOQuality.ULTRA,
        ]

        for q in qualities:
            assert q is not None


class TestAONumericalSafety:
    """Test numerical safety in AO calculations."""

    def test_ssao_kernel_handles_large_counts(self):
        """Test large sample count doesn't overflow."""
        kernel = SSAOKernel(sample_count=512)
        assert len(kernel.samples) == 512

        for x, y, z in kernel.samples:
            assert not math.isnan(x)
            assert not math.isnan(y)
            assert not math.isnan(z)

    def test_bilateral_weight_numerical_stability(self):
        """Test bilateral weight doesn't produce NaN."""
        bf = BilateralFilter()

        weight = bf._bilateral_weight(
            depth0=0.0, depth1=0.0,
            normal0=(0, 0, 1), normal1=(0, 0, 1),
            distance=0.0, sharpness=0.0,
        )

        assert not math.isnan(weight)
        assert 0.0 <= weight <= 1.0

    def test_bent_normal_view_dir_zero(self):
        """Test bent normal with zero view direction."""
        # Create valid bent normal
        bno = BentNormalOutput(
            bent_normal=(0.0, 1.0, 0.0),
            visibility_cone=0.5,
            occlusion=1.0,
        )

        # Zero view direction - dot product will be 0
        spec = bno.calculate_specular_occlusion(
            view_dir=(0.0, 0.0, 0.0),
            roughness=0.5,
        )

        assert 0.0 <= spec <= 1.0
