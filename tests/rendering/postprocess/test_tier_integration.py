"""Tests for T-CC-1.2: Quality tier integration for Post-Processing subsystem."""
import pytest
from trinity.types import QualityTier
from engine.rendering.postprocess.tier_integration import (
    PostProcessEffect,
    TonemapOperator,
    AAMethod,
    UpscalingMethod,
    BloomConfig,
    DOFConfig,
    MotionBlurConfig,
    TAAConfig,
    UpscalingConfig,
    PostProcessTierConfig,
    EffectTimingStats,
    TierChangeListener,
    PostProcessTierManager,
    create_low_tier_config,
    create_medium_tier_config,
    create_high_tier_config,
    create_ultra_tier_config,
    TIER_CONFIGS,
    get_tier_for_effects,
    estimate_postprocess_memory,
    create_effect_pass_list,
)


class TestEnums:
    """Tests for enum definitions."""

    def test_postprocess_effect_count(self):
        effects = list(PostProcessEffect)
        assert len(effects) >= 10

    def test_tonemap_operators(self):
        assert TonemapOperator.REINHARD in TonemapOperator
        assert TonemapOperator.ACES_FITTED in TonemapOperator

    def test_aa_methods(self):
        assert AAMethod.NONE in AAMethod
        assert AAMethod.TAA in AAMethod
        assert AAMethod.FXAA in AAMethod

    def test_upscaling_methods(self):
        assert UpscalingMethod.FSR2 in UpscalingMethod
        assert UpscalingMethod.DLSS in UpscalingMethod


class TestBloomConfig:
    """Tests for BloomConfig."""

    def test_default_values(self):
        cfg = BloomConfig()
        assert cfg.enabled is True
        assert cfg.iterations == 5
        assert cfg.threshold == 0.8

    def test_custom_values(self):
        cfg = BloomConfig(enabled=False, iterations=8, threshold=0.5)
        assert cfg.enabled is False
        assert cfg.iterations == 8


class TestDOFConfig:
    """Tests for DOFConfig."""

    def test_default_disabled(self):
        cfg = DOFConfig()
        assert cfg.enabled is False

    def test_bokeh_option(self):
        cfg = DOFConfig(enabled=True, use_bokeh=True)
        assert cfg.use_bokeh is True


class TestMotionBlurConfig:
    """Tests for MotionBlurConfig."""

    def test_default_disabled(self):
        cfg = MotionBlurConfig()
        assert cfg.enabled is False

    def test_samples(self):
        cfg = MotionBlurConfig(samples=16)
        assert cfg.samples == 16


class TestTAAConfig:
    """Tests for TAAConfig."""

    def test_default_disabled(self):
        cfg = TAAConfig()
        assert cfg.enabled is False

    def test_jitter_samples(self):
        cfg = TAAConfig(enabled=True, jitter_samples=16)
        assert cfg.jitter_samples == 16


class TestUpscalingConfig:
    """Tests for UpscalingConfig."""

    def test_default_disabled(self):
        cfg = UpscalingConfig()
        assert cfg.enabled is False
        assert cfg.method == UpscalingMethod.NONE

    def test_fsr2(self):
        cfg = UpscalingConfig(enabled=True, method=UpscalingMethod.FSR2)
        assert cfg.method == UpscalingMethod.FSR2


class TestPostProcessTierConfig:
    """Tests for PostProcessTierConfig."""

    def test_uses_temporal_effects_taa(self):
        cfg = PostProcessTierConfig(
            tier=QualityTier.MEDIUM,
            taa=TAAConfig(enabled=True),
        )
        assert cfg.uses_temporal_effects is True

    def test_uses_temporal_effects_upscaling(self):
        cfg = PostProcessTierConfig(
            tier=QualityTier.ULTRA,
            upscaling=UpscalingConfig(enabled=True, method=UpscalingMethod.FSR2),
        )
        assert cfg.uses_temporal_effects is True

    def test_uses_temporal_effects_false(self):
        cfg = PostProcessTierConfig(tier=QualityTier.LOW)
        assert cfg.uses_temporal_effects is False

    def test_effect_count(self):
        cfg = PostProcessTierConfig(
            tier=QualityTier.LOW,
            enabled_effects={PostProcessEffect.TONEMAPPING, PostProcessEffect.BLOOM},
        )
        assert cfg.effect_count == 2


class TestTierConfigs:
    """Tests for tier config factory functions."""

    def test_low_tier_config(self):
        cfg = create_low_tier_config()
        assert cfg.tier == QualityTier.LOW
        assert PostProcessEffect.TONEMAPPING in cfg.enabled_effects
        assert PostProcessEffect.BLOOM in cfg.enabled_effects
        assert PostProcessEffect.FXAA in cfg.enabled_effects
        assert cfg.aa_method == AAMethod.FXAA
        assert cfg.tonemap_operator == TonemapOperator.REINHARD
        assert cfg.render_scale == 0.75

    def test_low_tier_no_dof(self):
        cfg = create_low_tier_config()
        assert PostProcessEffect.DOF not in cfg.enabled_effects
        assert cfg.dof.enabled is False

    def test_medium_tier_config(self):
        cfg = create_medium_tier_config()
        assert cfg.tier == QualityTier.MEDIUM
        assert PostProcessEffect.DOF in cfg.enabled_effects
        assert PostProcessEffect.TAA in cfg.enabled_effects
        assert cfg.aa_method == AAMethod.TAA
        assert cfg.tonemap_operator == TonemapOperator.ACES

    def test_high_tier_config(self):
        cfg = create_high_tier_config()
        assert cfg.tier == QualityTier.HIGH
        assert PostProcessEffect.MOTION_BLUR in cfg.enabled_effects
        assert PostProcessEffect.COLOR_GRADING in cfg.enabled_effects
        assert PostProcessEffect.AUTO_EXPOSURE in cfg.enabled_effects
        assert cfg.motion_blur.enabled is True

    def test_ultra_tier_config(self):
        cfg = create_ultra_tier_config()
        assert cfg.tier == QualityTier.ULTRA
        assert PostProcessEffect.BOKEH_DOF in cfg.enabled_effects
        assert PostProcessEffect.LENS_FLARE in cfg.enabled_effects
        assert PostProcessEffect.TEMPORAL_UPSCALING in cfg.enabled_effects
        assert cfg.upscaling.enabled is True
        assert cfg.upscaling.method == UpscalingMethod.FSR2
        assert cfg.dof.use_bokeh is True

    def test_all_tiers_in_dict(self):
        for tier in QualityTier:
            assert tier in TIER_CONFIGS

    def test_tier_gpu_budgets_increase(self):
        low = create_low_tier_config()
        medium = create_medium_tier_config()
        high = create_high_tier_config()
        ultra = create_ultra_tier_config()
        assert low.gpu_time_budget_ms < medium.gpu_time_budget_ms
        assert medium.gpu_time_budget_ms < high.gpu_time_budget_ms
        assert high.gpu_time_budget_ms < ultra.gpu_time_budget_ms


class TestEffectTimingStats:
    """Tests for effect timing statistics."""

    def test_initial_values(self):
        stats = EffectTimingStats(PostProcessEffect.BLOOM)
        assert stats.gpu_time_ms == 0.0
        assert stats.invocation_count == 0

    def test_avg_time(self):
        stats = EffectTimingStats(PostProcessEffect.BLOOM, gpu_time_ms=10.0, invocation_count=5)
        assert stats.avg_time_ms == 2.0

    def test_avg_time_zero_count(self):
        stats = EffectTimingStats(PostProcessEffect.BLOOM)
        assert stats.avg_time_ms == 0.0  # Should not divide by zero


class TestTierChangeListener:
    """Tests for tier change listener."""

    def test_listener_receives_notification(self):
        class TestListener(TierChangeListener):
            def __init__(self):
                self.changes = []

            def on_tier_changed(self, old_tier, new_tier, config):
                self.changes.append((old_tier, new_tier, config.tier))

        mgr = PostProcessTierManager(QualityTier.LOW)
        listener = TestListener()
        mgr.add_listener(listener)
        mgr.set_tier(QualityTier.HIGH)

        assert len(listener.changes) == 1
        assert listener.changes[0] == (QualityTier.LOW, QualityTier.HIGH, QualityTier.HIGH)


class TestPostProcessTierManager:
    """Tests for PostProcessTierManager."""

    def test_default_tier(self):
        mgr = PostProcessTierManager()
        assert mgr.current_tier == QualityTier.MEDIUM

    def test_custom_initial_tier(self):
        mgr = PostProcessTierManager(QualityTier.HIGH)
        assert mgr.current_tier == QualityTier.HIGH

    def test_set_tier(self):
        mgr = PostProcessTierManager(QualityTier.LOW)
        mgr.set_tier(QualityTier.ULTRA)
        assert mgr.current_tier == QualityTier.ULTRA

    def test_set_same_tier_no_op(self):
        mgr = PostProcessTierManager(QualityTier.MEDIUM)
        changes = []
        listener = type("L", (), {"on_tier_changed": lambda s, o, n, c: changes.append(1)})()
        mgr.add_listener(listener)
        mgr.set_tier(QualityTier.MEDIUM)
        assert len(changes) == 0

    def test_is_effect_enabled(self):
        mgr = PostProcessTierManager(QualityTier.LOW)
        assert mgr.is_effect_enabled(PostProcessEffect.BLOOM)
        assert not mgr.is_effect_enabled(PostProcessEffect.DOF)

    def test_get_bloom_config(self):
        mgr = PostProcessTierManager(QualityTier.LOW)
        bloom = mgr.get_bloom_config()
        assert bloom.iterations == 3

    def test_get_dof_config(self):
        mgr = PostProcessTierManager(QualityTier.HIGH)
        dof = mgr.get_dof_config()
        assert dof.enabled is True
        assert dof.samples == 16

    def test_get_motion_blur_config(self):
        mgr = PostProcessTierManager(QualityTier.HIGH)
        mb = mgr.get_motion_blur_config()
        assert mb.enabled is True

    def test_get_taa_config(self):
        mgr = PostProcessTierManager(QualityTier.MEDIUM)
        taa = mgr.get_taa_config()
        assert taa.enabled is True

    def test_get_upscaling_config(self):
        mgr = PostProcessTierManager(QualityTier.ULTRA)
        up = mgr.get_upscaling_config()
        assert up.enabled is True
        assert up.method == UpscalingMethod.FSR2

    def test_get_tonemap_operator(self):
        low = PostProcessTierManager(QualityTier.LOW)
        high = PostProcessTierManager(QualityTier.HIGH)
        assert low.get_tonemap_operator() == TonemapOperator.REINHARD
        assert high.get_tonemap_operator() == TonemapOperator.ACES_FITTED

    def test_get_aa_method(self):
        low = PostProcessTierManager(QualityTier.LOW)
        assert low.get_aa_method() == AAMethod.FXAA

    def test_get_render_scale(self):
        low = PostProcessTierManager(QualityTier.LOW)
        assert low.get_render_scale() == 0.75


class TestEffectOverrides:
    """Tests for effect overrides."""

    def test_override_effect_enable(self):
        mgr = PostProcessTierManager(QualityTier.LOW)
        assert not mgr.is_effect_enabled(PostProcessEffect.DOF)
        mgr.override_effect(PostProcessEffect.DOF, True)
        assert mgr.is_effect_enabled(PostProcessEffect.DOF)

    def test_override_effect_disable(self):
        mgr = PostProcessTierManager(QualityTier.HIGH)
        assert mgr.is_effect_enabled(PostProcessEffect.MOTION_BLUR)
        mgr.override_effect(PostProcessEffect.MOTION_BLUR, False)
        assert not mgr.is_effect_enabled(PostProcessEffect.MOTION_BLUR)

    def test_clear_overrides(self):
        mgr = PostProcessTierManager(QualityTier.LOW)
        mgr.override_effect(PostProcessEffect.DOF, True)
        mgr.clear_overrides()
        assert not mgr.is_effect_enabled(PostProcessEffect.DOF)


class TestTimingAndBudget:
    """Tests for timing and budget tracking."""

    def test_record_effect_timing(self):
        mgr = PostProcessTierManager(QualityTier.MEDIUM)
        mgr.record_effect_timing(PostProcessEffect.BLOOM, 0.5)
        mgr.record_effect_timing(PostProcessEffect.BLOOM, 0.5)
        assert mgr._timing_stats[PostProcessEffect.BLOOM].invocation_count == 2

    def test_get_total_frame_time(self):
        mgr = PostProcessTierManager(QualityTier.MEDIUM)
        mgr.record_effect_timing(PostProcessEffect.BLOOM, 0.5)
        mgr.record_effect_timing(PostProcessEffect.DOF, 0.3)
        assert mgr.get_total_frame_time() == pytest.approx(0.8)

    def test_check_budget_within(self):
        mgr = PostProcessTierManager(QualityTier.LOW)  # 1ms budget
        mgr.record_effect_timing(PostProcessEffect.BLOOM, 0.5)
        assert mgr.check_budget() is True

    def test_check_budget_exceeded(self):
        mgr = PostProcessTierManager(QualityTier.LOW)  # 1ms budget
        mgr.record_effect_timing(PostProcessEffect.BLOOM, 2.0)
        assert mgr.check_budget() is False

    def test_reset_frame_stats(self):
        mgr = PostProcessTierManager(QualityTier.MEDIUM)
        mgr.record_effect_timing(PostProcessEffect.BLOOM, 0.5)
        mgr.reset_frame_stats()
        assert mgr.get_total_frame_time() == 0.0

    def test_should_auto_downgrade(self):
        mgr = PostProcessTierManager(QualityTier.HIGH)
        for _ in range(15):
            mgr.record_effect_timing(PostProcessEffect.BLOOM, 5.0)  # Exceeds 3ms budget
            mgr.check_budget()
        assert mgr.should_auto_downgrade(consecutive_frames=10) is True

    def test_auto_downgrade(self):
        mgr = PostProcessTierManager(QualityTier.HIGH)
        result = mgr.auto_downgrade()
        assert result is True
        assert mgr.current_tier == QualityTier.MEDIUM

    def test_auto_downgrade_from_low(self):
        mgr = PostProcessTierManager(QualityTier.LOW)
        result = mgr.auto_downgrade()
        assert result is False
        assert mgr.current_tier == QualityTier.LOW


class TestEnabledEffectsList:
    """Tests for enabled effects list generation."""

    def test_get_enabled_effects_list_low(self):
        mgr = PostProcessTierManager(QualityTier.LOW)
        effects = mgr.get_enabled_effects_list()
        assert PostProcessEffect.BLOOM in effects
        assert PostProcessEffect.TONEMAPPING in effects
        assert PostProcessEffect.FXAA in effects

    def test_effects_in_render_order(self):
        mgr = PostProcessTierManager(QualityTier.HIGH)
        effects = mgr.get_enabled_effects_list()
        bloom_idx = effects.index(PostProcessEffect.BLOOM)
        tonemap_idx = effects.index(PostProcessEffect.TONEMAPPING)
        # Bloom should come before tonemapping
        assert bloom_idx < tonemap_idx


class TestStatusDict:
    """Tests for status dictionary."""

    def test_get_status_dict(self):
        mgr = PostProcessTierManager(QualityTier.HIGH)
        status = mgr.get_status_dict()
        assert status["tier"] == "HIGH"
        assert status["bloom_enabled"] is True
        assert status["dof_enabled"] is True
        assert status["motion_blur_enabled"] is True
        assert status["render_scale"] == 1.0


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_tier_for_effects_low(self):
        effects = {PostProcessEffect.TONEMAPPING, PostProcessEffect.BLOOM}
        tier = get_tier_for_effects(effects)
        assert tier == QualityTier.LOW

    def test_get_tier_for_effects_medium(self):
        effects = {PostProcessEffect.DOF, PostProcessEffect.TAA}
        tier = get_tier_for_effects(effects)
        assert tier == QualityTier.MEDIUM

    def test_get_tier_for_effects_ultra(self):
        effects = {PostProcessEffect.LENS_FLARE, PostProcessEffect.BOKEH_DOF}
        tier = get_tier_for_effects(effects)
        assert tier == QualityTier.ULTRA


class TestMemoryEstimation:
    """Tests for memory estimation."""

    def test_estimate_low_tier_memory(self):
        cfg = create_low_tier_config()
        memory = estimate_postprocess_memory(cfg, 1280, 720)
        assert memory > 0

    def test_estimate_high_tier_memory(self):
        cfg = create_high_tier_config()
        memory = estimate_postprocess_memory(cfg, 1920, 1080)
        low_memory = estimate_postprocess_memory(create_low_tier_config(), 1920, 1080)
        # High tier needs more memory
        assert memory > low_memory

    def test_estimate_includes_taa_history(self):
        with_taa = PostProcessTierConfig(tier=QualityTier.MEDIUM, taa=TAAConfig(enabled=True))
        without_taa = PostProcessTierConfig(tier=QualityTier.LOW, taa=TAAConfig(enabled=False))
        mem_with = estimate_postprocess_memory(with_taa, 1920, 1080)
        mem_without = estimate_postprocess_memory(without_taa, 1920, 1080)
        assert mem_with > mem_without


class TestEffectPassList:
    """Tests for effect pass list generation."""

    def test_create_effect_pass_list_low(self):
        cfg = create_low_tier_config()
        passes = create_effect_pass_list(cfg)
        pass_names = [p["name"] for p in passes]
        assert "tonemap" in pass_names

    def test_create_effect_pass_list_has_bloom(self):
        cfg = create_medium_tier_config()
        passes = create_effect_pass_list(cfg)
        pass_names = [p["name"] for p in passes]
        assert "bloom_downsample" in pass_names
        assert "bloom_upsample" in pass_names

    def test_create_effect_pass_list_has_dof(self):
        cfg = create_high_tier_config()
        passes = create_effect_pass_list(cfg)
        pass_names = [p["name"] for p in passes]
        assert "dof" in pass_names

    def test_create_effect_pass_list_has_taa(self):
        cfg = create_medium_tier_config()
        passes = create_effect_pass_list(cfg)
        pass_names = [p["name"] for p in passes]
        assert "taa" in pass_names


class TestIntegration:
    """Integration tests."""

    def test_full_tier_progression(self):
        mgr = PostProcessTierManager(QualityTier.LOW)
        for tier in [QualityTier.MEDIUM, QualityTier.HIGH, QualityTier.ULTRA]:
            mgr.set_tier(tier)
            assert mgr.current_tier == tier
            assert mgr.config.tier == tier

    def test_frame_lifecycle(self):
        mgr = PostProcessTierManager(QualityTier.MEDIUM)

        # Simulate frame
        mgr.reset_frame_stats()
        mgr.record_effect_timing(PostProcessEffect.BLOOM, 0.3)
        mgr.record_effect_timing(PostProcessEffect.DOF, 0.4)
        mgr.record_effect_timing(PostProcessEffect.TAA, 0.2)

        total = mgr.get_total_frame_time()
        assert total == pytest.approx(0.9)
        assert mgr.check_budget() is True  # Under 2ms budget

    def test_budget_exceeded_downgrade(self):
        mgr = PostProcessTierManager(QualityTier.ULTRA)

        # Simulate many over-budget frames
        for _ in range(15):
            mgr.reset_frame_stats()
            mgr.record_effect_timing(PostProcessEffect.BLOOM, 10.0)
            mgr.check_budget()

        if mgr.should_auto_downgrade():
            mgr.auto_downgrade()

        assert mgr.current_tier == QualityTier.HIGH
