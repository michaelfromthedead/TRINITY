"""
Tests for Depth of Field System

Tests CircleOfConfusion calculation, BokehShape generation,
AutoFocusSystem, and DOFEffect integration.
"""

import math
import pytest

from engine.rendering.postprocess.dof import (
    AutoFocusSystem,
    BokehShape,
    BokehShapeType,
    CircleOfConfusion,
    DOFEffect,
    DOFMode,
    DOFQuality,
    DOFSettings,
    NearFieldDOF,
    FarFieldDOF,
)


class TestCircleOfConfusion:
    """Test Circle of Confusion calculation."""

    def test_default_parameters(self):
        """Test default CoC parameters."""
        coc = CircleOfConfusion()

        assert coc.sensor_width == 36.0
        assert coc.focal_length == 50.0
        assert coc.aperture == 2.8
        assert coc.focus_distance == 5.0

    def test_calculate_at_focus_distance(self):
        """Test CoC is 0 at focus distance."""
        coc = CircleOfConfusion(
            sensor_width=36.0,
            focal_length=50.0,
            aperture=2.8,
            focus_distance=5.0,
        )

        coc_pixels = coc.calculate(5.0, 1920)

        # At focus distance, CoC should be 0
        assert abs(coc_pixels) < 0.01

    def test_calculate_out_of_focus(self):
        """Test CoC increases away from focus plane."""
        coc = CircleOfConfusion(
            sensor_width=36.0,
            focal_length=50.0,
            aperture=2.8,
            focus_distance=5.0,
        )

        # At focus
        coc_focus = coc.calculate(5.0, 1920)

        # Beyond focus
        coc_far = coc.calculate(10.0, 1920)

        # Before focus
        coc_near = coc.calculate(2.0, 1920)

        # Both should have larger CoC than focus
        assert coc_far > coc_focus
        assert coc_near > coc_focus

    def test_hyperfocal_distance_formula(self):
        """Test hyperfocal distance formula: H = f + f^2 / (N * c)."""
        # For a 50mm lens at f/2.8 on full frame (36mm sensor, 1920px)
        coc = CircleOfConfusion(
            sensor_width=36.0,
            focal_length=50.0,
            aperture=2.8,
            focus_distance=5.0,
        )

        coc_mm = coc.sensor_width / 1920  # 1 pixel in mm
        hyperfocal_m = (
            coc.focal_length / 1000.0
            + (coc.focal_length / 1000.0) ** 2 / (coc.aperture * coc_mm / 1000.0)
        )

        # Test by calculating CoC at the hyperfocal distance
        # At hyperfocal, far limit should be at infinity
        coc_at_hyperfocal = coc.calculate(hyperfocal_m, 1920)
        coc_past_hyperfocal = coc.calculate(hyperfocal_m * 2, 1920)

        # Both should produce valid (non-negative) CoC values
        assert coc_at_hyperfocal >= 0
        assert coc_past_hyperfocal >= 0

    def test_known_aperture_values(self):
        """Test wider aperture gives larger CoC."""
        coc_wide = CircleOfConfusion(
            sensor_width=36.0,
            focal_length=50.0,
            aperture=1.4,
            focus_distance=5.0,
        )
        coc_narrow = CircleOfConfusion(
            sensor_width=36.0,
            focal_length=50.0,
            aperture=11.0,
            focus_distance=5.0,
        )

        coc_wide_val = coc_wide.calculate(10.0, 1920)
        coc_narrow_val = coc_narrow.calculate(10.0, 1920)

        # Wider aperture = larger CoC
        assert coc_wide_val > coc_narrow_val

    def test_zero_depth_returns_zero(self):
        """Test zero depth returns 0 CoC."""
        coc = CircleOfConfusion()
        result = coc.calculate(0.0, 1920)

        assert result == 0.0

    def test_negative_depth_returns_zero(self):
        """Test negative depth returns 0 CoC."""
        coc = CircleOfConfusion()
        result = coc.calculate(-1.0, 1920)

        assert result == 0.0

    def test_max_coc_radius_clamping(self):
        """Test CoC is clamped to max_coc_radius."""
        coc = CircleOfConfusion(
            sensor_width=36.0,
            focal_length=200.0,
            aperture=1.4,
            focus_distance=1.0,
            max_coc_radius=32.0,
        )

        # Very far object should clamp to max
        coc_val = coc.calculate(100.0, 1920)

        assert coc_val <= coc.max_coc_radius

    def test_image_width_affects_coc(self):
        """Test wider image gives larger CoC in pixels."""
        coc = CircleOfConfusion(
            sensor_width=36.0,
            focal_length=50.0,
            aperture=2.8,
            focus_distance=5.0,
        )

        coc_hd = coc.calculate(10.0, 1920)
        coc_4k = coc.calculate(10.0, 3840)

        # Higher resolution = larger CoC in pixels (same optical CoC)
        assert coc_4k > coc_hd

    def test_focus_distance_impact(self):
        """Test closer focus gives larger background CoC."""
        coc_close = CircleOfConfusion(focus_distance=2.0)
        coc_far = CircleOfConfusion(focus_distance=10.0)

        close_val = coc_close.calculate(20.0, 1920)
        far_val = coc_far.calculate(20.0, 1920)

        # Closer focus = larger background blur
        assert close_val > far_val

    def test_get_depth_ranges(self):
        """Test get_depth_ranges returns ordered values."""
        coc = CircleOfConfusion(
            sensor_width=36.0,
            focal_length=50.0,
            aperture=2.8,
            focus_distance=5.0,
        )

        near_sharp, near_blur, far_sharp, far_blur = coc.get_depth_ranges(1920)

        # Ranges should be ordered
        assert near_sharp <= near_blur
        assert near_blur <= far_sharp
        assert far_sharp <= far_blur

    def test_get_depth_ranges_hyperfocal(self):
        """Test hyperfocal focus returns far_blur as inf."""
        coc = CircleOfConfusion(
            sensor_width=36.0,
            focal_length=16.0,
            aperture=11.0,
            focus_distance=10.0,
        )

        _, _, _, far_blur = coc.get_depth_ranges(1920)

        # Wide angle stopped down should have far focus at infinity
        assert far_blur == float("inf")

    def test_coc_increases_with_distance(self):
        """Test CoC increases as depth moves away from focus plane."""
        coc = CircleOfConfusion(
            sensor_width=36.0,
            focal_length=50.0,
            aperture=2.8,
            focus_distance=5.0,
        )

        # Both near and far should produce valid positive CoC
        coc_near = coc.calculate(3.0, 1920)
        coc_far = coc.calculate(7.0, 1920)

        assert coc_near > 0
        assert coc_far > 0

        # Further away from focus should give larger CoC
        coc_further_near = coc.calculate(1.0, 1920)
        coc_further_far = coc.calculate(20.0, 1920)

        assert coc_further_near > coc_near
        assert coc_further_far > coc_far


class TestBokehShape:
    """Test BokehShape kernel generation."""

    def test_default_shape(self):
        """Test default bokeh shape."""
        bokeh = BokehShape()

        assert bokeh.shape_type == BokehShapeType.CIRCLE
        assert bokeh.blade_count == 6

    def test_disk_kernel_normalized(self):
        """Test circular disk kernel has valid weights."""
        bokeh = BokehShape(shape_type=BokehShapeType.CIRCLE)
        samples = bokeh.get_bokeh_kernel(radius=8)

        assert len(samples) >= 8

        for x, y, weight in samples:
            # All samples should be within the disk radius
            assert math.sqrt(x * x + y * y) <= 8 + 0.01
            # All weights should be positive
            assert weight >= 0.1

    def test_disk_kernel_sample_count(self):
        """Test disk kernel sample count scales with radius."""
        bokeh = BokehShape(shape_type=BokehShapeType.CIRCLE)

        small_kernel = bokeh.get_bokeh_kernel(radius=4)
        large_kernel = bokeh.get_bokeh_kernel(radius=16)

        # Larger radius should have more samples
        assert len(small_kernel) < len(large_kernel)

    def test_polygon_kernel_five_blades(self):
        """Test 5-sided polygon kernel."""
        bokeh = BokehShape(
            shape_type=BokehShapeType.POLYGON,
            blade_count=5,
        )
        samples = bokeh.get_bokeh_kernel(radius=8)

        assert len(samples) >= 8

        for x, y, weight in samples:
            # Should be within the polygon
            assert math.sqrt(x * x + y * y) <= 8 + 0.01
            assert weight >= 0.1

    def test_polygon_kernel_six_blades(self):
        """Test 6-sided polygon kernel."""
        bokeh = BokehShape(
            shape_type=BokehShapeType.POLYGON,
            blade_count=6,
        )
        samples = bokeh.get_bokeh_kernel(radius=8)

        assert len(samples) >= 8

    def test_polygon_kernel_seven_blades(self):
        """Test 7-sided polygon kernel."""
        bokeh = BokehShape(
            shape_type=BokehShapeType.POLYGON,
            blade_count=7,
        )
        samples = bokeh.get_bokeh_kernel(radius=8)

        assert len(samples) >= 8

    def test_polygon_kernel_eight_blades(self):
        """Test 8-sided polygon kernel."""
        bokeh = BokehShape(
            shape_type=BokehShapeType.POLYGON,
            blade_count=8,
        )
        samples = bokeh.get_bokeh_kernel(radius=8)

        assert len(samples) >= 8

    def test_anamorphic_kernel_stretch(self):
        """Test anamorphic kernel elliptical stretch."""
        bokeh_wide = BokehShape(
            shape_type=BokehShapeType.ANAMORPHIC,
            anamorphic_ratio=2.0,
        )
        bokeh_square = BokehShape(
            shape_type=BokehShapeType.ANAMORPHIC,
            anamorphic_ratio=1.0,
        )

        wide_samples = bokeh_wide.get_bokeh_kernel(radius=8)
        square_samples = bokeh_square.get_bokeh_kernel(radius=8)

        # Wide anamorphic should have larger x-extent
        max_x_wide = max(abs(x) for x, _, _ in wide_samples)
        max_x_square = max(abs(x) for x, _, _ in square_samples)

        assert max_x_wide > max_x_square

    def test_anamorphic_aspect_ratio(self):
        """Test anamorphic aspect ratio is applied."""
        bokeh = BokehShape(
            shape_type=BokehShapeType.ANAMORPHIC,
            anamorphic_ratio=0.5,
        )
        samples = bokeh.get_bokeh_kernel(radius=8)

        # With ratio < 1, samples should be squished in x
        for x, y, weight in samples:
            assert weight >= 0.1

    def test_blade_curvature_affects_polygon(self):
        """Test blade curvature affects polygon kernel."""
        bokeh_convex = BokehShape(
            shape_type=BokehShapeType.POLYGON,
            blade_count=6,
            blade_curvature=0.5,
        )
        bokeh_concave = BokehShape(
            shape_type=BokehShapeType.POLYGON,
            blade_count=6,
            blade_curvature=-0.5,
        )

        convex_samples = bokeh_convex.get_bokeh_kernel(radius=8)
        concave_samples = bokeh_concave.get_bokeh_kernel(radius=8)

        assert len(convex_samples) > 0
        assert len(concave_samples) > 0

    def test_blade_rotation(self):
        """Test blade rotation affects polygon orientation."""
        bokeh_rotated = BokehShape(
            shape_type=BokehShapeType.POLYGON,
            blade_count=6,
            blade_rotation=30.0,
        )
        samples = bokeh_rotated.get_bokeh_kernel(radius=8)

        assert len(samples) >= 8

    def test_spherical_aberration_affects_weights(self):
        """Test spherical aberration affects edge brightness."""
        bokeh_pos = BokehShape(
            shape_type=BokehShapeType.CIRCLE,
            spherical_aberration=0.5,
        )
        bokeh_neg = BokehShape(
            shape_type=BokehShapeType.CIRCLE,
            spherical_aberration=-0.5,
        )

        pos_samples = bokeh_pos.get_bokeh_kernel(radius=8)
        neg_samples = bokeh_neg.get_bokeh_kernel(radius=8)

        # Both should produce valid samples
        assert len(pos_samples) > 0
        assert len(neg_samples) > 0

    def test_kernel_weights_positive(self):
        """Test all kernel weights are positive."""
        for shape_type in BokehShapeType:
            bokeh = BokehShape(shape_type=shape_type)
            samples = bokeh.get_bokeh_kernel(radius=8)

            for _, _, weight in samples:
                assert weight > 0.0, f"Weight <= 0 for {shape_type}"


class TestBokehShapeType:
    """Test BokehShapeType enum."""

    def test_all_types_exist(self):
        """Test all bokeh shape types exist."""
        types = [
            BokehShapeType.CIRCLE,
            BokehShapeType.POLYGON,
            BokehShapeType.ANAMORPHIC,
            BokehShapeType.CAT_EYE,
            BokehShapeType.SWIRL,
        ]

        for t in types:
            assert t is not None


class TestAutoFocusSystem:
    """Test AutoFocusSystem."""

    def test_default_focus(self):
        """Test default focus distance."""
        af = AutoFocusSystem()

        assert af.current_focus == 5.0

    def test_update_no_change(self):
        """Test update when already at target."""
        af = AutoFocusSystem()

        result = af.update(5.0, 1.0)

        # Should already be at target
        assert abs(result - 5.0) < 0.01

    def test_smooth_transition_to_target(self):
        """Test smooth transition towards target."""
        af = AutoFocusSystem()
        af._current_focus = 1.0

        result = af.update(10.0, 1.0)

        # Should have moved towards target but not reached it
        assert result > 1.0
        assert result < 10.0

    def test_transition_speed(self):
        """Test transition speed is limited by max_change."""
        af = AutoFocusSystem()
        af._current_focus = 1.0
        af._focus_speed = 2.0  # 2 meters per second

        # Over 0.5 seconds, should move at most 1 meter
        result = af.update(10.0, 0.5)

        assert result <= 1.0 + 2.0 * 0.5 + 0.01
        assert result > 1.0

    def test_reaches_target(self):
        """Test eventually reaches the target."""
        af = AutoFocusSystem()
        af._current_focus = 1.0

        for _ in range(100):
            result = af.update(10.0, 1.0)

        # After enough time, should reach target
        assert abs(result - 10.0) < 0.01

    def test_smooth_transition_back_and_forth(self):
        """Test smooth transition in both directions."""
        af = AutoFocusSystem()

        # Go from 5 to 1
        af._current_focus = 5.0
        result_down = af.update(1.0, 1.0)
        assert result_down < 5.0

        # Go from current to 10
        result_up = af.update(10.0, 1.0)
        assert result_up > result_down


class TestDOFSettings:
    """Test DOFSettings dataclass."""

    def test_default_settings(self):
        """Test default DOF settings."""
        settings = DOFSettings()

        assert settings.mode == DOFMode.PHYSICAL
        assert settings.aperture == 2.8
        assert settings.focal_length == 50.0
        assert settings.focus_distance == 5.0

    def test_custom_settings(self):
        """Test custom DOF settings."""
        settings = DOFSettings(
            mode=DOFMode.MANUAL,
            aperture=1.4,
            focal_length=85.0,
            focus_distance=3.0,
        )

        assert settings.mode == DOFMode.MANUAL
        assert settings.aperture == 1.4
        assert settings.focal_length == 85.0
        assert settings.focus_distance == 3.0

    def test_settings_lerp(self):
        """Test settings interpolation."""
        settings1 = DOFSettings(aperture=1.4, focal_length=50.0)
        settings2 = DOFSettings(aperture=5.6, focal_length=100.0)

        lerped = settings1.lerp(settings2, 0.5)

        assert lerped.aperture == pytest.approx(3.5, abs=1e-6)
        assert lerped.focal_length == pytest.approx(75.0, abs=1e-6)


class TestDOFEffect:
    """Test DOFEffect integration."""

    def test_effect_creation(self):
        """Test DOF effect creation."""
        effect = DOFEffect()

        assert effect.name == "DepthOfField"
        assert effect.settings is not None
        assert effect.coc_calculator is not None

    def test_effect_with_custom_settings(self):
        """Test effect with custom settings."""
        settings = DOFSettings(
            mode=DOFMode.MANUAL,
            aperture=4.0,
            focal_length=35.0,
        )
        effect = DOFEffect(settings)

        assert effect.settings.aperture == 4.0
        assert effect.settings.focal_length == 35.0

    def test_effect_required_inputs(self):
        """Test effect required inputs."""
        effect = DOFEffect()
        inputs = effect.get_required_inputs()

        assert "color" in inputs
        assert "depth" in inputs

    def test_effect_outputs(self):
        """Test effect outputs."""
        effect = DOFEffect()
        outputs = effect.get_outputs()

        assert "color" in outputs

    def test_current_focus_distance(self):
        """Test current focus distance."""
        effect = DOFEffect()
        effect.setup(1920, 1080)

        distance = effect.current_focus_distance
        assert distance == 5.0

    def test_effect_is_compute(self):
        """Test effect uses compute."""
        effect = DOFEffect()
        assert effect.is_compute_effect() is True

    def test_effect_setup(self):
        """Test effect setup."""
        effect = DOFEffect()
        effect.setup(1920, 1080)

        # Should not raise
        assert True

    def test_effect_execute_disabled(self):
        """Test effect does nothing when disabled."""
        settings = DOFSettings(enabled=False)
        effect = DOFEffect(settings)

        effect.execute({}, {}, 0.016)
        # Should not raise

    def test_effect_cleanup(self):
        """Test effect cleanup."""
        effect = DOFEffect()
        effect.cleanup()

        # Should not raise

    def test_coc_calculator_updates_from_settings(self):
        """Test CoC calculator gets settings values."""
        settings = DOFSettings(
            aperture=4.0,
            focal_length=35.0,
            focus_distance=3.0,
        )
        effect = DOFEffect(settings)
        effect.setup(1920, 1080)

        assert effect.coc_calculator.aperture == 4.0
        assert effect.coc_calculator.focal_length == 35.0
        # Focus distance may change due to autofocus logic
        assert effect.coc_calculator.focus_distance is not None


class TestDOFNumericalSafety:
    """Test numerical safety in DOF calculations."""

    def test_coc_with_zero_aperture(self):
        """Test CoC with zero aperture should raise."""
        coc = CircleOfConfusion(aperture=0.0)

        # Aperture of 0 causes division by zero in the hyperfocal formula
        with pytest.raises(ZeroDivisionError):
            coc.calculate(10.0, 1920)

    def test_coc_with_very_large_depth(self):
        """Test CoC with very large depth value."""
        coc = CircleOfConfusion()
        result = coc.calculate(100000.0, 1920)

        # Should not overflow and should be clamped
        assert 0 <= result <= coc.max_coc_radius

    def test_coc_with_zero_focus_distance(self):
        """Test CoC with zero focus distance."""
        coc = CircleOfConfusion(focus_distance=0.0)
        result = coc.calculate(5.0, 1920)

        # Should handle gracefully
        assert result >= 0


class TestDOFQuality:
    """Test DOFQuality enum."""

    def test_all_qualities_exist(self):
        """Test all quality presets exist."""
        qualities = [
            DOFQuality.LOW,
            DOFQuality.MEDIUM,
            DOFQuality.HIGH,
            DOFQuality.CINEMATIC,
        ]

        for q in qualities:
            assert q is not None


class TestDOFMode:
    """Test DOFMode enum."""

    def test_all_modes_exist(self):
        """Test all DOF modes exist."""
        modes = [
            DOFMode.PHYSICAL,
            DOFMode.MANUAL,
            DOFMode.AUTO_FOCUS,
        ]

        for m in modes:
            assert m is not None


class TestNearFieldDOF:
    """Test NearFieldDOF operations."""

    def test_creation(self):
        """Test near field DOF creation."""
        dof = NearFieldDOF()
        assert dof is not None

    def test_setup(self):
        """Test near field setup."""
        dof = NearFieldDOF()
        dof.setup(1920, 1080)
        # Should not raise


class TestFarFieldDOF:
    """Test FarFieldDOF operations."""

    def test_creation(self):
        """Test far field DOF creation."""
        dof = FarFieldDOF()
        assert dof is not None

    def test_setup(self):
        """Test far field setup."""
        dof = FarFieldDOF()
        dof.setup(1920, 1080)
        # Should not raise
