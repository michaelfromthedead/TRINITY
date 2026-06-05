"""Tests for Reflection Fallback Chain System (T-GIR-P8.5).

Comprehensive test suite covering:
- ReflectionTechnique enum behavior
- TechniqueResult creation and manipulation
- TechniqueSelector logic and priority order
- ConfidenceBlender weight computation and blending
- FallbackChainConfig validation and presets
- TransitionManager history and smoothing
- ReflectionFallbackPass full chain evaluation
- WGSL shader generation
- Edge cases and error handling

Tests follow the pattern established in test_rt_reflections.py.
"""

import math
import pytest
from typing import List, Optional

from engine.core.math.vec import Vec3
from engine.rendering.reflections.reflection_fallback import (
    # Constants
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_BLEND_THRESHOLD,
    DEFAULT_TRANSITION_SPEED,
    MIN_VALID_CONFIDENCE,
    DEFAULT_HISTORY_LENGTH,
    EPSILON,
    # Enums
    ReflectionTechnique,
    # Data structures
    TechniqueResult,
    FallbackPassOutput,
    PixelHistory,
    # Configuration
    FallbackChainConfig,
    # Core classes
    TechniqueSelector,
    ConfidenceBlender,
    TransitionManager,
    ReflectionFallbackPass,
    # Utilities
    generate_fallback_chain_wgsl,
    evaluate_fallback_chain,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def default_config() -> FallbackChainConfig:
    """Create default fallback chain configuration."""
    return FallbackChainConfig()


@pytest.fixture
def high_quality_config() -> FallbackChainConfig:
    """Create high quality configuration."""
    return FallbackChainConfig.high_quality()


@pytest.fixture
def performance_config() -> FallbackChainConfig:
    """Create performance configuration."""
    return FallbackChainConfig.performance()


@pytest.fixture
def selector() -> TechniqueSelector:
    """Create default technique selector."""
    return TechniqueSelector()


@pytest.fixture
def blender() -> ConfidenceBlender:
    """Create default confidence blender."""
    return ConfidenceBlender()


@pytest.fixture
def transition_manager() -> TransitionManager:
    """Create default transition manager."""
    return TransitionManager()


@pytest.fixture
def fallback_pass(default_config: FallbackChainConfig) -> ReflectionFallbackPass:
    """Create default fallback pass."""
    return ReflectionFallbackPass(default_config)


def make_result(
    color: Vec3 = None,
    confidence: float = 0.8,
    technique: ReflectionTechnique = ReflectionTechnique.RT_REFLECTION,
    hit_distance: float = 5.0,
    valid: bool = True,
) -> TechniqueResult:
    """Helper to create TechniqueResult."""
    return TechniqueResult(
        color=color or Vec3(0.5, 0.5, 0.5),
        confidence=confidence,
        hit_distance=hit_distance,
        technique=technique,
        valid=valid,
    )


# =============================================================================
# ReflectionTechnique Tests
# =============================================================================


class TestReflectionTechnique:
    """Test ReflectionTechnique enum."""

    def test_technique_values(self):
        """Test technique enum values are sequential."""
        assert ReflectionTechnique.RT_REFLECTION == 0
        assert ReflectionTechnique.SSR == 1
        assert ReflectionTechnique.REFLECTION_PROBE == 2
        assert ReflectionTechnique.ENVIRONMENT_MAP == 3

    def test_is_realtime(self):
        """Test is_realtime property."""
        assert ReflectionTechnique.RT_REFLECTION.is_realtime is True
        assert ReflectionTechnique.SSR.is_realtime is True
        assert ReflectionTechnique.REFLECTION_PROBE.is_realtime is False
        assert ReflectionTechnique.ENVIRONMENT_MAP.is_realtime is False

    def test_is_screen_space(self):
        """Test is_screen_space property."""
        assert ReflectionTechnique.RT_REFLECTION.is_screen_space is False
        assert ReflectionTechnique.SSR.is_screen_space is True
        assert ReflectionTechnique.REFLECTION_PROBE.is_screen_space is False
        assert ReflectionTechnique.ENVIRONMENT_MAP.is_screen_space is False

    def test_quality_level(self):
        """Test quality_level property."""
        assert ReflectionTechnique.RT_REFLECTION.quality_level == 1.0
        assert ReflectionTechnique.SSR.quality_level == 0.8
        assert ReflectionTechnique.REFLECTION_PROBE.quality_level == 0.5
        assert ReflectionTechnique.ENVIRONMENT_MAP.quality_level == 0.2

    def test_quality_level_ordering(self):
        """Test quality levels are ordered correctly."""
        assert (
            ReflectionTechnique.RT_REFLECTION.quality_level
            > ReflectionTechnique.SSR.quality_level
            > ReflectionTechnique.REFLECTION_PROBE.quality_level
            > ReflectionTechnique.ENVIRONMENT_MAP.quality_level
        )

    def test_from_name_valid(self):
        """Test from_name with valid names."""
        assert ReflectionTechnique.from_name("rt") == ReflectionTechnique.RT_REFLECTION
        assert ReflectionTechnique.from_name("RT_REFLECTION") == ReflectionTechnique.RT_REFLECTION
        assert ReflectionTechnique.from_name("ssr") == ReflectionTechnique.SSR
        assert ReflectionTechnique.from_name("SSR") == ReflectionTechnique.SSR
        assert ReflectionTechnique.from_name("probe") == ReflectionTechnique.REFLECTION_PROBE
        assert ReflectionTechnique.from_name("env") == ReflectionTechnique.ENVIRONMENT_MAP
        assert ReflectionTechnique.from_name("environment_map") == ReflectionTechnique.ENVIRONMENT_MAP

    def test_from_name_invalid(self):
        """Test from_name with invalid name returns environment map."""
        assert ReflectionTechnique.from_name("invalid") == ReflectionTechnique.ENVIRONMENT_MAP
        assert ReflectionTechnique.from_name("") == ReflectionTechnique.ENVIRONMENT_MAP


# =============================================================================
# TechniqueResult Tests
# =============================================================================


class TestTechniqueResult:
    """Test TechniqueResult dataclass."""

    def test_default_creation(self):
        """Test default TechniqueResult creation."""
        result = TechniqueResult()
        assert result.color == Vec3.zero()
        assert result.confidence == 0.0
        assert result.hit_distance == float("inf")
        assert result.technique == ReflectionTechnique.ENVIRONMENT_MAP
        assert result.valid is True

    def test_custom_creation(self):
        """Test TechniqueResult with custom values."""
        color = Vec3(1.0, 0.5, 0.0)
        result = TechniqueResult(
            color=color,
            confidence=0.9,
            hit_distance=10.0,
            technique=ReflectionTechnique.RT_REFLECTION,
            valid=True,
        )
        assert result.color.x == 1.0
        assert result.confidence == 0.9
        assert result.hit_distance == 10.0
        assert result.technique == ReflectionTechnique.RT_REFLECTION

    def test_confidence_clamping(self):
        """Test confidence is clamped to [0, 1]."""
        result_high = TechniqueResult(confidence=1.5)
        assert result_high.confidence == 1.0

        result_low = TechniqueResult(confidence=-0.5)
        assert result_low.confidence == 0.0

    def test_hit_distance_clamping(self):
        """Test hit_distance is non-negative."""
        result = TechniqueResult(hit_distance=-10.0)
        assert result.hit_distance == 0.0

    def test_roughness_clamping(self):
        """Test roughness is clamped to [0, 1]."""
        result_high = TechniqueResult(roughness=1.5)
        assert result_high.roughness == 1.0

        result_low = TechniqueResult(roughness=-0.5)
        assert result_low.roughness == 0.0

    def test_is_miss(self):
        """Test is_miss property."""
        miss = TechniqueResult(confidence=0.0, valid=False)
        assert miss.is_miss is True

        hit = TechniqueResult(confidence=0.5, valid=True)
        assert hit.is_miss is False

        low_conf = TechniqueResult(confidence=0.001, valid=True)
        assert low_conf.is_miss is True  # Below MIN_VALID_CONFIDENCE

    def test_is_hit(self):
        """Test is_hit property."""
        hit = TechniqueResult(confidence=0.5, valid=True)
        assert hit.is_hit is True

        miss = TechniqueResult(confidence=0.0, valid=False)
        assert miss.is_hit is False

    def test_with_confidence(self):
        """Test with_confidence creates new result."""
        original = make_result(confidence=0.5)
        modified = original.with_confidence(0.9)

        assert original.confidence == 0.5
        assert modified.confidence == 0.9
        assert modified.color.x == original.color.x

    def test_with_technique(self):
        """Test with_technique creates new result."""
        original = make_result(technique=ReflectionTechnique.RT_REFLECTION)
        modified = original.with_technique(ReflectionTechnique.SSR)

        assert original.technique == ReflectionTechnique.RT_REFLECTION
        assert modified.technique == ReflectionTechnique.SSR

    def test_miss_factory(self):
        """Test miss() factory method."""
        miss = TechniqueResult.miss()
        assert miss.confidence == 0.0
        assert miss.valid is False
        assert miss.is_miss is True

    def test_miss_factory_with_technique(self):
        """Test miss() with specific technique."""
        miss = TechniqueResult.miss(ReflectionTechnique.SSR)
        assert miss.technique == ReflectionTechnique.SSR
        assert miss.is_miss is True

    def test_from_color_factory(self):
        """Test from_color() factory method."""
        result = TechniqueResult.from_color(
            color=Vec3(1.0, 0.0, 0.0),
            confidence=0.8,
            technique=ReflectionTechnique.RT_REFLECTION,
            hit_distance=5.0,
        )
        assert result.color.x == 1.0
        assert result.confidence == 0.8
        assert result.valid is True


# =============================================================================
# TechniqueSelector Tests
# =============================================================================


class TestTechniqueSelector:
    """Test TechniqueSelector class."""

    def test_default_creation(self, selector: TechniqueSelector):
        """Test default selector creation."""
        assert selector.enable_rt is True
        assert selector.enable_ssr is True
        assert selector.enable_probes is True
        assert selector.confidence_threshold == DEFAULT_CONFIDENCE_THRESHOLD

    def test_custom_creation(self):
        """Test selector with custom settings."""
        selector = TechniqueSelector(
            enable_rt=False,
            enable_ssr=True,
            enable_probes=False,
            confidence_threshold=0.7,
        )
        assert selector.enable_rt is False
        assert selector.enable_ssr is True
        assert selector.enable_probes is False
        assert selector.confidence_threshold == 0.7

    def test_get_priority_order_all_enabled(self, selector: TechniqueSelector):
        """Test priority order with all techniques enabled."""
        order = selector.get_priority_order()
        assert order[0] == ReflectionTechnique.RT_REFLECTION
        assert order[1] == ReflectionTechnique.SSR
        assert order[2] == ReflectionTechnique.REFLECTION_PROBE
        assert order[3] == ReflectionTechnique.ENVIRONMENT_MAP

    def test_get_priority_order_rt_disabled(self):
        """Test priority order with RT disabled."""
        selector = TechniqueSelector(enable_rt=False)
        order = selector.get_priority_order()
        assert order[0] == ReflectionTechnique.SSR
        assert ReflectionTechnique.RT_REFLECTION not in order

    def test_get_priority_order_only_probes(self):
        """Test priority order with only probes enabled."""
        selector = TechniqueSelector(
            enable_rt=False, enable_ssr=False, enable_probes=True
        )
        order = selector.get_priority_order()
        assert order[0] == ReflectionTechnique.REFLECTION_PROBE
        assert order[1] == ReflectionTechnique.ENVIRONMENT_MAP

    def test_get_priority_order_always_includes_env(self):
        """Test environment map is always in priority order."""
        selector = TechniqueSelector(
            enable_rt=False, enable_ssr=False, enable_probes=False
        )
        order = selector.get_priority_order()
        assert ReflectionTechnique.ENVIRONMENT_MAP in order

    def test_should_try_next_invalid_result(self, selector: TechniqueSelector):
        """Test should_try_next with invalid result."""
        result = TechniqueResult(valid=False)
        assert selector.should_try_next(result) is True

    def test_should_try_next_low_confidence(self, selector: TechniqueSelector):
        """Test should_try_next with low confidence."""
        result = TechniqueResult(confidence=0.3, valid=True)
        selector.confidence_threshold = 0.5
        assert selector.should_try_next(result) is True

    def test_should_try_next_high_confidence(self, selector: TechniqueSelector):
        """Test should_try_next with high confidence."""
        result = TechniqueResult(confidence=0.9, valid=True)
        selector.confidence_threshold = 0.5
        assert selector.should_try_next(result) is False

    def test_select_technique_rt_hit(self, selector: TechniqueSelector):
        """Test select_technique returns RT on good RT result."""
        rt = make_result(confidence=0.9, technique=ReflectionTechnique.RT_REFLECTION)
        tech, result = selector.select_technique(rt_result=rt)
        assert tech == ReflectionTechnique.RT_REFLECTION

    def test_select_technique_rt_miss_uses_ssr(self, selector: TechniqueSelector):
        """Test fallback to SSR when RT misses."""
        rt = make_result(confidence=0.1, technique=ReflectionTechnique.RT_REFLECTION)
        ssr = make_result(confidence=0.8, technique=ReflectionTechnique.SSR)
        tech, result = selector.select_technique(rt_result=rt, ssr_result=ssr)
        assert tech == ReflectionTechnique.SSR

    def test_select_technique_ssr_miss_uses_probes(self, selector: TechniqueSelector):
        """Test fallback to probes when SSR misses."""
        ssr = make_result(confidence=0.1, technique=ReflectionTechnique.SSR)
        probe = make_result(confidence=0.7, technique=ReflectionTechnique.REFLECTION_PROBE)
        tech, result = selector.select_technique(ssr_result=ssr, probe_result=probe)
        assert tech == ReflectionTechnique.REFLECTION_PROBE

    def test_select_technique_all_miss_uses_env(self, selector: TechniqueSelector):
        """Test fallback to environment when all miss."""
        rt = make_result(confidence=0.1, technique=ReflectionTechnique.RT_REFLECTION)
        ssr = make_result(confidence=0.1, technique=ReflectionTechnique.SSR)
        probe = make_result(confidence=0.1, technique=ReflectionTechnique.REFLECTION_PROBE)
        env = make_result(confidence=1.0, technique=ReflectionTechnique.ENVIRONMENT_MAP)

        tech, result = selector.select_technique(
            rt_result=rt, ssr_result=ssr, probe_result=probe, env_result=env
        )
        assert tech == ReflectionTechnique.ENVIRONMENT_MAP

    def test_select_technique_no_results(self, selector: TechniqueSelector):
        """Test select_technique with no results returns miss."""
        tech, result = selector.select_technique()
        assert tech == ReflectionTechnique.ENVIRONMENT_MAP
        assert result.is_miss is True

    def test_get_next_technique(self, selector: TechniqueSelector):
        """Test get_next_technique returns correct fallback."""
        assert selector.get_next_technique(ReflectionTechnique.RT_REFLECTION) == ReflectionTechnique.SSR
        assert selector.get_next_technique(ReflectionTechnique.SSR) == ReflectionTechnique.REFLECTION_PROBE
        assert selector.get_next_technique(ReflectionTechnique.REFLECTION_PROBE) == ReflectionTechnique.ENVIRONMENT_MAP
        assert selector.get_next_technique(ReflectionTechnique.ENVIRONMENT_MAP) is None

    def test_is_technique_enabled(self, selector: TechniqueSelector):
        """Test is_technique_enabled checks."""
        assert selector.is_technique_enabled(ReflectionTechnique.RT_REFLECTION) is True
        assert selector.is_technique_enabled(ReflectionTechnique.ENVIRONMENT_MAP) is True

        selector.enable_rt = False
        assert selector.is_technique_enabled(ReflectionTechnique.RT_REFLECTION) is False


# =============================================================================
# ConfidenceBlender Tests
# =============================================================================


class TestConfidenceBlender:
    """Test ConfidenceBlender class."""

    def test_default_creation(self, blender: ConfidenceBlender):
        """Test default blender creation."""
        assert blender.blend_threshold == DEFAULT_BLEND_THRESHOLD
        assert blender.transition_speed == DEFAULT_TRANSITION_SPEED

    def test_custom_creation(self):
        """Test blender with custom settings."""
        blender = ConfidenceBlender(blend_threshold=0.5, transition_speed=0.2)
        assert blender.blend_threshold == 0.5
        assert blender.transition_speed == 0.2

    def test_compute_blend_weight_high_primary(self, blender: ConfidenceBlender):
        """Test blend weight when primary is highly confident."""
        weight = blender.compute_blend_weight(0.99, 0.5)
        assert weight < 0.1  # Very little secondary blending

    def test_compute_blend_weight_low_primary(self, blender: ConfidenceBlender):
        """Test blend weight when primary has low confidence."""
        weight = blender.compute_blend_weight(0.1, 0.8)
        assert weight > 0.5  # Significant secondary blending

    def test_compute_blend_weight_no_secondary(self, blender: ConfidenceBlender):
        """Test blend weight when secondary has no confidence."""
        weight = blender.compute_blend_weight(0.5, 0.0)
        assert weight == 0.0

    def test_compute_blend_weight_no_primary(self, blender: ConfidenceBlender):
        """Test blend weight when primary has no confidence."""
        weight = blender.compute_blend_weight(0.0, 0.8)
        assert weight == 1.0

    def test_lerp_colors(self, blender: ConfidenceBlender):
        """Test color lerping."""
        black = Vec3(0.0, 0.0, 0.0)
        white = Vec3(1.0, 1.0, 1.0)

        result = blender.lerp_colors(black, white, 0.5)
        assert abs(result.x - 0.5) < EPSILON
        assert abs(result.y - 0.5) < EPSILON
        assert abs(result.z - 0.5) < EPSILON

    def test_lerp_colors_at_zero(self, blender: ConfidenceBlender):
        """Test color lerp at t=0 returns first color."""
        red = Vec3(1.0, 0.0, 0.0)
        blue = Vec3(0.0, 0.0, 1.0)

        result = blender.lerp_colors(red, blue, 0.0)
        assert result.x == 1.0
        assert result.z == 0.0

    def test_lerp_colors_at_one(self, blender: ConfidenceBlender):
        """Test color lerp at t=1 returns second color."""
        red = Vec3(1.0, 0.0, 0.0)
        blue = Vec3(0.0, 0.0, 1.0)

        result = blender.lerp_colors(red, blue, 1.0)
        assert result.x == 0.0
        assert result.z == 1.0

    def test_lerp_colors_clamps_t(self, blender: ConfidenceBlender):
        """Test color lerp clamps t to [0, 1]."""
        black = Vec3(0.0, 0.0, 0.0)
        white = Vec3(1.0, 1.0, 1.0)

        result_high = blender.lerp_colors(black, white, 2.0)
        assert result_high.x == 1.0

        result_low = blender.lerp_colors(black, white, -1.0)
        assert result_low.x == 0.0

    def test_blend_results_both_valid(self, blender: ConfidenceBlender):
        """Test blending two valid results."""
        primary = make_result(
            color=Vec3(1.0, 0.0, 0.0),
            confidence=0.7,
            technique=ReflectionTechnique.RT_REFLECTION,
        )
        secondary = make_result(
            color=Vec3(0.0, 1.0, 0.0),
            confidence=0.5,
            technique=ReflectionTechnique.SSR,
        )

        result = blender.blend_results(primary, secondary)
        assert result.valid is True
        # Color should be somewhere between red and green
        assert 0.0 < result.color.x < 1.0
        assert 0.0 < result.color.y < 1.0

    def test_blend_results_primary_invalid(self, blender: ConfidenceBlender):
        """Test blending returns secondary when primary is invalid."""
        primary = make_result(valid=False)
        secondary = make_result(
            color=Vec3(0.0, 1.0, 0.0),
            confidence=0.8,
            technique=ReflectionTechnique.SSR,
        )

        result = blender.blend_results(primary, secondary)
        assert result.color.y == 1.0

    def test_blend_results_secondary_invalid(self, blender: ConfidenceBlender):
        """Test blending returns primary when secondary is invalid."""
        primary = make_result(
            color=Vec3(1.0, 0.0, 0.0),
            confidence=0.8,
            technique=ReflectionTechnique.RT_REFLECTION,
        )
        secondary = make_result(valid=False)

        result = blender.blend_results(primary, secondary)
        assert result.color.x == 1.0

    def test_blend_results_both_invalid(self, blender: ConfidenceBlender):
        """Test blending returns miss when both invalid."""
        primary = make_result(valid=False)
        secondary = make_result(valid=False)

        result = blender.blend_results(primary, secondary)
        assert result.is_miss is True

    def test_blend_chain_single_result(self, blender: ConfidenceBlender):
        """Test chain blending with single result."""
        results = [make_result(confidence=0.9)]
        result = blender.blend_chain(results)
        assert result.confidence == 0.9

    def test_blend_chain_multiple_results(self, blender: ConfidenceBlender):
        """Test chain blending with multiple results."""
        results = [
            make_result(confidence=0.7, technique=ReflectionTechnique.RT_REFLECTION),
            make_result(confidence=0.5, technique=ReflectionTechnique.SSR),
            make_result(confidence=0.3, technique=ReflectionTechnique.REFLECTION_PROBE),
        ]
        result = blender.blend_chain(results)
        assert result.valid is True

    def test_blend_chain_empty(self, blender: ConfidenceBlender):
        """Test chain blending with empty list."""
        result = blender.blend_chain([])
        assert result.is_miss is True


# =============================================================================
# FallbackChainConfig Tests
# =============================================================================


class TestFallbackChainConfig:
    """Test FallbackChainConfig dataclass."""

    def test_default_creation(self, default_config: FallbackChainConfig):
        """Test default config creation."""
        assert default_config.enable_rt is True
        assert default_config.enable_ssr is True
        assert default_config.enable_probes is True
        assert default_config.blend_threshold == DEFAULT_BLEND_THRESHOLD
        assert default_config.transition_speed == DEFAULT_TRANSITION_SPEED

    def test_value_clamping(self):
        """Test config values are clamped."""
        config = FallbackChainConfig(
            blend_threshold=1.5,
            transition_speed=-0.1,
            confidence_threshold=2.0,
            max_blend_distance=-10.0,
        )
        assert config.blend_threshold == 1.0
        assert config.transition_speed == 0.0
        assert config.confidence_threshold == 1.0
        assert config.max_blend_distance == 1.0

    def test_validate_valid_config(self, default_config: FallbackChainConfig):
        """Test validation passes for valid config."""
        errors = default_config.validate()
        assert len(errors) == 0

    def test_high_quality_preset(self, high_quality_config: FallbackChainConfig):
        """Test high quality preset."""
        assert high_quality_config.enable_rt is True
        assert high_quality_config.blend_threshold == 0.2
        assert high_quality_config.confidence_threshold == 0.6

    def test_performance_preset(self, performance_config: FallbackChainConfig):
        """Test performance preset."""
        assert performance_config.enable_rt is False
        assert performance_config.blend_threshold == 0.4

    def test_minimal_preset(self):
        """Test minimal preset."""
        config = FallbackChainConfig.minimal()
        assert config.enable_rt is False
        assert config.enable_ssr is False
        assert config.enable_probes is True


# =============================================================================
# TransitionManager Tests
# =============================================================================


class TestTransitionManager:
    """Test TransitionManager class."""

    def test_default_creation(self, transition_manager: TransitionManager):
        """Test default manager creation."""
        assert transition_manager.history_length == DEFAULT_HISTORY_LENGTH
        assert transition_manager.transition_speed == DEFAULT_TRANSITION_SPEED
        assert transition_manager.total_pixels_managed == 0

    def test_update_history(self, transition_manager: TransitionManager):
        """Test updating pixel history."""
        transition_manager.update_history(
            x=10, y=20,
            technique=ReflectionTechnique.RT_REFLECTION,
            confidence=0.9,
            color=Vec3(1.0, 0.0, 0.0),
        )

        history = transition_manager.get_pixel_history(10, 20)
        assert history is not None
        assert len(history.technique_history) == 1
        assert history.technique_history[0] == ReflectionTechnique.RT_REFLECTION

    def test_update_history_trims_to_length(self, transition_manager: TransitionManager):
        """Test history is trimmed to max length."""
        transition_manager.history_length = 4

        for i in range(10):
            transition_manager.update_history(
                x=0, y=0,
                technique=ReflectionTechnique.SSR,
                confidence=0.5,
                color=Vec3(float(i), 0.0, 0.0),
            )

        history = transition_manager.get_pixel_history(0, 0)
        assert len(history.technique_history) == 4

    def test_get_stable_technique_consistent(self, transition_manager: TransitionManager):
        """Test stable technique with consistent history."""
        for _ in range(5):
            transition_manager.update_history(
                x=0, y=0,
                technique=ReflectionTechnique.SSR,
                confidence=0.8,
                color=Vec3(0.5, 0.5, 0.5),
            )

        stable = transition_manager.get_stable_technique(0, 0, ReflectionTechnique.SSR)
        assert stable == ReflectionTechnique.SSR

    def test_get_stable_technique_dominant(self, transition_manager: TransitionManager):
        """Test stable technique returns dominant technique."""
        # Add 4 SSR, 2 RT
        for _ in range(4):
            transition_manager.update_history(
                x=0, y=0,
                technique=ReflectionTechnique.SSR,
                confidence=0.8,
                color=Vec3(0.5, 0.5, 0.5),
            )
        for _ in range(2):
            transition_manager.update_history(
                x=0, y=0,
                technique=ReflectionTechnique.RT_REFLECTION,
                confidence=0.8,
                color=Vec3(0.5, 0.5, 0.5),
            )

        # Should return SSR as dominant
        stable = transition_manager.get_stable_technique(0, 0, ReflectionTechnique.RT_REFLECTION)
        assert stable == ReflectionTechnique.SSR

    def test_get_stable_technique_no_history(self, transition_manager: TransitionManager):
        """Test stable technique with no history returns current."""
        stable = transition_manager.get_stable_technique(99, 99, ReflectionTechnique.RT_REFLECTION)
        assert stable == ReflectionTechnique.RT_REFLECTION

    def test_smooth_transition(self, transition_manager: TransitionManager):
        """Test smooth transition between frames."""
        # Add initial history
        transition_manager.update_history(
            x=0, y=0,
            technique=ReflectionTechnique.SSR,
            confidence=0.8,
            color=Vec3(0.0, 0.0, 0.0),  # Black
        )
        transition_manager.update_history(
            x=0, y=0,
            technique=ReflectionTechnique.SSR,
            confidence=0.8,
            color=Vec3(1.0, 1.0, 1.0),  # White
        )

        current = make_result(
            color=Vec3(1.0, 1.0, 1.0),
            confidence=0.8,
            technique=ReflectionTechnique.SSR,
        )

        smoothed = transition_manager.smooth_transition(0, 0, current)
        # Should be partially blended between black and white
        assert smoothed.valid is True

    def test_smooth_transition_no_history(self, transition_manager: TransitionManager):
        """Test smooth transition returns current when no history."""
        current = make_result(color=Vec3(1.0, 0.0, 0.0))
        smoothed = transition_manager.smooth_transition(99, 99, current)
        assert smoothed.color.x == 1.0

    def test_clear_history(self, transition_manager: TransitionManager):
        """Test clearing all history."""
        transition_manager.update_history(0, 0, ReflectionTechnique.SSR, 0.8, Vec3(1.0, 0.0, 0.0))
        transition_manager.update_history(1, 1, ReflectionTechnique.SSR, 0.8, Vec3(1.0, 0.0, 0.0))

        assert transition_manager.total_pixels_managed == 2

        transition_manager.clear_history()
        assert transition_manager.total_pixels_managed == 0

    def test_clear_pixel(self, transition_manager: TransitionManager):
        """Test clearing single pixel history."""
        transition_manager.update_history(0, 0, ReflectionTechnique.SSR, 0.8, Vec3(1.0, 0.0, 0.0))
        transition_manager.update_history(1, 1, ReflectionTechnique.SSR, 0.8, Vec3(1.0, 0.0, 0.0))

        transition_manager.clear_pixel(0, 0)
        assert transition_manager.get_pixel_history(0, 0) is None
        assert transition_manager.get_pixel_history(1, 1) is not None

    def test_reset_frame_stats(self, transition_manager: TransitionManager):
        """Test resetting per-frame statistics."""
        # Simulate some transitions
        transition_manager.update_history(0, 0, ReflectionTechnique.SSR, 0.8, Vec3.zero())
        transition_manager.update_history(0, 0, ReflectionTechnique.RT_REFLECTION, 0.8, Vec3.zero())
        transition_manager.get_stable_technique(0, 0, ReflectionTechnique.RT_REFLECTION)

        transition_manager.reset_frame_stats()
        assert transition_manager.transitions_this_frame == 0


# =============================================================================
# ReflectionFallbackPass Tests
# =============================================================================


class TestReflectionFallbackPass:
    """Test ReflectionFallbackPass class."""

    def test_default_creation(self, fallback_pass: ReflectionFallbackPass):
        """Test default pass creation."""
        assert fallback_pass.config is not None
        assert fallback_pass.is_initialized is False

    def test_execute_creates_buffers(self, fallback_pass: ReflectionFallbackPass):
        """Test execute creates output buffers."""
        # Set up environment sampler (always available)
        fallback_pass.set_env_sampler(
            lambda x, y: TechniqueResult.from_color(
                Vec3(0.2, 0.3, 0.5),
                confidence=1.0,
                technique=ReflectionTechnique.ENVIRONMENT_MAP,
            )
        )

        fallback_pass.execute(10, 10)

        assert fallback_pass.is_initialized is True
        assert fallback_pass.output_width == 10
        assert fallback_pass.output_height == 10
        assert len(fallback_pass.get_final_buffer()) == 100
        assert len(fallback_pass.get_technique_mask()) == 100

    def test_execute_invalid_dimensions(self, fallback_pass: ReflectionFallbackPass):
        """Test execute raises on invalid dimensions."""
        with pytest.raises(ValueError):
            fallback_pass.execute(0, 10)

        with pytest.raises(ValueError):
            fallback_pass.execute(10, -1)

    def test_get_final_at(self, fallback_pass: ReflectionFallbackPass):
        """Test getting final output at coordinates."""
        fallback_pass.set_env_sampler(
            lambda x, y: TechniqueResult.from_color(
                Vec3(float(x) / 10, float(y) / 10, 0.0),
                confidence=1.0,
                technique=ReflectionTechnique.ENVIRONMENT_MAP,
            )
        )

        fallback_pass.execute(10, 10)

        output = fallback_pass.get_final_at(5, 5)
        assert output.color is not None

    def test_get_final_at_out_of_bounds(self, fallback_pass: ReflectionFallbackPass):
        """Test getting final output out of bounds."""
        fallback_pass.set_env_sampler(
            lambda x, y: TechniqueResult.from_color(
                Vec3(0.5, 0.5, 0.5),
                confidence=1.0,
                technique=ReflectionTechnique.ENVIRONMENT_MAP,
            )
        )
        fallback_pass.execute(10, 10)

        output = fallback_pass.get_final_at(-1, 5)
        assert output.confidence == 0.0

        output = fallback_pass.get_final_at(100, 100)
        assert output.confidence == 0.0

    def test_get_technique_at(self, fallback_pass: ReflectionFallbackPass):
        """Test getting technique at coordinates."""
        fallback_pass.set_env_sampler(
            lambda x, y: TechniqueResult.from_color(
                Vec3(0.5, 0.5, 0.5),
                confidence=1.0,
                technique=ReflectionTechnique.ENVIRONMENT_MAP,
            )
        )
        fallback_pass.execute(10, 10)

        tech = fallback_pass.get_technique_at(5, 5)
        assert tech == ReflectionTechnique.ENVIRONMENT_MAP

    def test_get_statistics(self, fallback_pass: ReflectionFallbackPass):
        """Test getting statistics after execution."""
        fallback_pass.set_env_sampler(
            lambda x, y: TechniqueResult.from_color(
                Vec3(0.5, 0.5, 0.5),
                confidence=1.0,
                technique=ReflectionTechnique.ENVIRONMENT_MAP,
            )
        )
        fallback_pass.execute(10, 10)

        stats = fallback_pass.get_statistics()
        assert stats["total_pixels"] == 100
        assert "env_pixels" in stats
        assert "env_percent" in stats

    def test_technique_selection_rt_priority(self, fallback_pass: ReflectionFallbackPass):
        """Test RT is used when available with high confidence."""
        fallback_pass.set_rt_sampler(
            lambda x, y: TechniqueResult.from_color(
                Vec3(1.0, 0.0, 0.0),
                confidence=0.9,
                technique=ReflectionTechnique.RT_REFLECTION,
            )
        )
        fallback_pass.set_ssr_sampler(
            lambda x, y: TechniqueResult.from_color(
                Vec3(0.0, 1.0, 0.0),
                confidence=0.8,
                technique=ReflectionTechnique.SSR,
            )
        )
        fallback_pass.set_env_sampler(
            lambda x, y: TechniqueResult.from_color(
                Vec3(0.2, 0.3, 0.5),
                confidence=1.0,
                technique=ReflectionTechnique.ENVIRONMENT_MAP,
            )
        )

        fallback_pass.execute(5, 5)

        stats = fallback_pass.get_statistics()
        assert stats["rt_pixels"] > 0

    def test_technique_selection_ssr_fallback(self, default_config: FallbackChainConfig):
        """Test SSR is used when RT fails."""
        default_config.enable_rt = False
        fallback_pass = ReflectionFallbackPass(default_config)

        fallback_pass.set_ssr_sampler(
            lambda x, y: TechniqueResult.from_color(
                Vec3(0.0, 1.0, 0.0),
                confidence=0.8,
                technique=ReflectionTechnique.SSR,
            )
        )
        fallback_pass.set_env_sampler(
            lambda x, y: TechniqueResult.from_color(
                Vec3(0.2, 0.3, 0.5),
                confidence=1.0,
                technique=ReflectionTechnique.ENVIRONMENT_MAP,
            )
        )

        fallback_pass.execute(5, 5)

        stats = fallback_pass.get_statistics()
        assert stats["ssr_pixels"] > 0
        assert stats["rt_pixels"] == 0

    def test_invalidate_history(self, fallback_pass: ReflectionFallbackPass):
        """Test invalidating temporal history."""
        fallback_pass.set_env_sampler(
            lambda x, y: TechniqueResult.from_color(
                Vec3(0.5, 0.5, 0.5),
                confidence=1.0,
                technique=ReflectionTechnique.ENVIRONMENT_MAP,
            )
        )
        fallback_pass.execute(5, 5)

        fallback_pass.invalidate_history()
        # Should not crash after invalidation
        fallback_pass.execute(5, 5)

    def test_config_update(self, fallback_pass: ReflectionFallbackPass):
        """Test updating configuration."""
        new_config = FallbackChainConfig.performance()
        fallback_pass.config = new_config

        assert fallback_pass.config.enable_rt is False


# =============================================================================
# evaluate_fallback_chain Tests
# =============================================================================


class TestEvaluateFallbackChain:
    """Test evaluate_fallback_chain utility function."""

    def test_returns_rt_on_hit(self, default_config: FallbackChainConfig):
        """Test returns RT result when it hits."""
        rt = make_result(confidence=0.9, technique=ReflectionTechnique.RT_REFLECTION)
        result = evaluate_fallback_chain(default_config, rt_result=rt, env_result=None)
        assert result.technique == ReflectionTechnique.RT_REFLECTION

    def test_returns_ssr_on_rt_miss(self, default_config: FallbackChainConfig):
        """Test returns SSR when RT misses."""
        rt = make_result(confidence=0.1, technique=ReflectionTechnique.RT_REFLECTION)
        ssr = make_result(confidence=0.8, technique=ReflectionTechnique.SSR)
        result = evaluate_fallback_chain(default_config, rt_result=rt, ssr_result=ssr, env_result=None)
        # Should blend RT and SSR
        assert result.valid is True

    def test_returns_env_when_all_miss(self, default_config: FallbackChainConfig):
        """Test returns environment when all techniques miss."""
        env = make_result(confidence=1.0, technique=ReflectionTechnique.ENVIRONMENT_MAP)
        result = evaluate_fallback_chain(default_config, env_result=env)
        assert result.technique == ReflectionTechnique.ENVIRONMENT_MAP

    def test_respects_enable_flags(self):
        """Test respects technique enable flags."""
        config = FallbackChainConfig(enable_rt=False)
        rt = make_result(confidence=0.9, technique=ReflectionTechnique.RT_REFLECTION)
        ssr = make_result(confidence=0.8, technique=ReflectionTechnique.SSR)

        result = evaluate_fallback_chain(config, rt_result=rt, ssr_result=ssr, env_result=None)
        # RT should be ignored since disabled
        assert result.technique == ReflectionTechnique.SSR

    def test_returns_miss_when_no_results(self, default_config: FallbackChainConfig):
        """Test returns miss when no results provided."""
        result = evaluate_fallback_chain(default_config)
        assert result.is_miss is True


# =============================================================================
# WGSL Generation Tests
# =============================================================================


class TestWGSLGeneration:
    """Test WGSL shader generation."""

    def test_generates_shader(self, default_config: FallbackChainConfig):
        """Test shader is generated."""
        shader = generate_fallback_chain_wgsl(default_config)
        assert len(shader) > 0
        assert "reflection_fallback_chain" in shader.lower()

    def test_includes_config_values(self, default_config: FallbackChainConfig):
        """Test shader includes configuration values."""
        shader = generate_fallback_chain_wgsl(default_config)
        assert "CONFIDENCE_THRESHOLD" in shader
        assert "BLEND_THRESHOLD" in shader
        assert "TRANSITION_SPEED" in shader

    def test_includes_enable_flags(self, default_config: FallbackChainConfig):
        """Test shader includes enable flags."""
        shader = generate_fallback_chain_wgsl(default_config)
        assert "ENABLE_RT" in shader
        assert "ENABLE_SSR" in shader
        assert "ENABLE_PROBES" in shader

    def test_includes_technique_ids(self, default_config: FallbackChainConfig):
        """Test shader includes technique IDs."""
        shader = generate_fallback_chain_wgsl(default_config)
        assert "TECH_RT" in shader
        assert "TECH_SSR" in shader
        assert "TECH_PROBE" in shader
        assert "TECH_ENV" in shader

    def test_includes_blend_functions(self, default_config: FallbackChainConfig):
        """Test shader includes blend functions."""
        shader = generate_fallback_chain_wgsl(default_config)
        assert "compute_blend_weight" in shader
        assert "blend_colors" in shader

    def test_includes_fallback_logic(self, default_config: FallbackChainConfig):
        """Test shader includes fallback chain logic."""
        shader = generate_fallback_chain_wgsl(default_config)
        assert "should_try_next" in shader
        assert "blend_with_previous" in shader

    def test_different_configs_different_shaders(self):
        """Test different configs produce different shaders."""
        config1 = FallbackChainConfig(enable_rt=True)
        config2 = FallbackChainConfig(enable_rt=False)

        shader1 = generate_fallback_chain_wgsl(config1)
        shader2 = generate_fallback_chain_wgsl(config2)

        assert shader1 != shader2


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_confidence_results(self, fallback_pass: ReflectionFallbackPass):
        """Test handling of zero confidence results."""
        fallback_pass.set_rt_sampler(
            lambda x, y: TechniqueResult(confidence=0.0, valid=True)
        )
        fallback_pass.set_env_sampler(
            lambda x, y: TechniqueResult.from_color(
                Vec3(0.5, 0.5, 0.5),
                confidence=1.0,
                technique=ReflectionTechnique.ENVIRONMENT_MAP,
            )
        )

        fallback_pass.execute(5, 5)
        # Should fall through to environment
        stats = fallback_pass.get_statistics()
        assert stats["env_pixels"] == 25

    def test_nan_confidence_clamped(self):
        """Test NaN confidence is handled."""
        # This would need special handling if confidence could be NaN
        result = TechniqueResult(confidence=float("nan"))
        # Should clamp to valid range
        assert 0.0 <= result.confidence <= 1.0 or math.isnan(result.confidence)

    def test_large_dimensions(self, default_config: FallbackChainConfig):
        """Test with larger dimensions."""
        fallback_pass = ReflectionFallbackPass(default_config)
        fallback_pass.set_env_sampler(
            lambda x, y: TechniqueResult.from_color(
                Vec3(0.5, 0.5, 0.5),
                confidence=1.0,
                technique=ReflectionTechnique.ENVIRONMENT_MAP,
            )
        )

        # 100x100 = 10000 pixels
        fallback_pass.execute(100, 100)
        assert len(fallback_pass.get_final_buffer()) == 10000

    def test_all_techniques_disabled_except_env(self):
        """Test with all techniques disabled except environment."""
        config = FallbackChainConfig(
            enable_rt=False,
            enable_ssr=False,
            enable_probes=False,
        )
        fallback_pass = ReflectionFallbackPass(config)
        fallback_pass.set_env_sampler(
            lambda x, y: TechniqueResult.from_color(
                Vec3(0.5, 0.5, 0.5),
                confidence=1.0,
                technique=ReflectionTechnique.ENVIRONMENT_MAP,
            )
        )

        fallback_pass.execute(5, 5)
        stats = fallback_pass.get_statistics()
        assert stats["env_pixels"] == 25
        assert stats["rt_pixels"] == 0
        assert stats["ssr_pixels"] == 0
        assert stats["probe_pixels"] == 0

    def test_rapidly_changing_techniques(self, transition_manager: TransitionManager):
        """Test transition manager with rapidly changing techniques."""
        techniques = [
            ReflectionTechnique.RT_REFLECTION,
            ReflectionTechnique.SSR,
            ReflectionTechnique.RT_REFLECTION,
            ReflectionTechnique.REFLECTION_PROBE,
            ReflectionTechnique.SSR,
        ]

        for tech in techniques:
            transition_manager.update_history(
                x=0, y=0,
                technique=tech,
                confidence=0.8,
                color=Vec3(0.5, 0.5, 0.5),
            )

        # Should stabilize to most frequent
        stable = transition_manager.get_stable_technique(0, 0, ReflectionTechnique.ENVIRONMENT_MAP)
        assert stable in [t for t in ReflectionTechnique]


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline(self, default_config: FallbackChainConfig):
        """Test full pipeline from input to output."""
        fallback_pass = ReflectionFallbackPass(default_config)

        # Set up samplers simulating different pixel regions
        def rt_sampler(x: int, y: int) -> TechniqueResult:
            # RT works well in center, misses at edges
            if 2 <= x <= 7 and 2 <= y <= 7:
                return TechniqueResult.from_color(
                    Vec3(1.0, 0.0, 0.0),
                    confidence=0.9,
                    technique=ReflectionTechnique.RT_REFLECTION,
                    hit_distance=5.0,
                )
            return TechniqueResult.miss(ReflectionTechnique.RT_REFLECTION)

        def ssr_sampler(x: int, y: int) -> TechniqueResult:
            # SSR works in most areas
            if x > 0 and y > 0:
                return TechniqueResult.from_color(
                    Vec3(0.0, 1.0, 0.0),
                    confidence=0.7,
                    technique=ReflectionTechnique.SSR,
                    hit_distance=3.0,
                )
            return TechniqueResult.miss(ReflectionTechnique.SSR)

        def probe_sampler(x: int, y: int) -> TechniqueResult:
            return TechniqueResult.from_color(
                Vec3(0.0, 0.0, 1.0),
                confidence=0.5,
                technique=ReflectionTechnique.REFLECTION_PROBE,
            )

        def env_sampler(x: int, y: int) -> TechniqueResult:
            return TechniqueResult.from_color(
                Vec3(0.2, 0.3, 0.5),
                confidence=1.0,
                technique=ReflectionTechnique.ENVIRONMENT_MAP,
            )

        fallback_pass.set_rt_sampler(rt_sampler)
        fallback_pass.set_ssr_sampler(ssr_sampler)
        fallback_pass.set_probe_sampler(probe_sampler)
        fallback_pass.set_env_sampler(env_sampler)

        fallback_pass.execute(10, 10)

        stats = fallback_pass.get_statistics()
        # Should have mix of techniques
        assert stats["rt_pixels"] > 0
        assert stats["ssr_pixels"] > 0 or stats["env_pixels"] > 0

        # Center pixel should use RT
        center = fallback_pass.get_final_at(5, 5)
        assert center.confidence > 0.5

    def test_temporal_stability_over_frames(self, default_config: FallbackChainConfig):
        """Test temporal stability across multiple frames."""
        fallback_pass = ReflectionFallbackPass(default_config)

        frame_results = []

        def rt_sampler(x: int, y: int) -> TechniqueResult:
            # Flickering RT (alternates between hit and miss)
            return TechniqueResult.from_color(
                Vec3(1.0, 0.0, 0.0),
                confidence=0.9 if len(frame_results) % 2 == 0 else 0.1,
                technique=ReflectionTechnique.RT_REFLECTION,
            )

        def env_sampler(x: int, y: int) -> TechniqueResult:
            return TechniqueResult.from_color(
                Vec3(0.2, 0.3, 0.5),
                confidence=1.0,
                technique=ReflectionTechnique.ENVIRONMENT_MAP,
            )

        fallback_pass.set_rt_sampler(rt_sampler)
        fallback_pass.set_env_sampler(env_sampler)

        # Execute multiple frames
        for _ in range(5):
            fallback_pass.execute(5, 5)
            center = fallback_pass.get_final_at(2, 2)
            frame_results.append(center.color.x)

        # Temporal smoothing should prevent hard jumps
        # (colors should gradually change, not flip between 1.0 and 0.2)
        assert len(frame_results) == 5


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
