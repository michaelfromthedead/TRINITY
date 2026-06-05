"""T-CC-1.2: Quality tier integration for Post-Processing subsystem (S8).

Wires quality tier settings into runtime post-processing configuration:
- Low: tonemap + bloom + FXAA
- Medium: + DOF + TAA
- High: Full effects + motion blur + color grading
- Ultra: + upscaling + bokeh DOF + lens flare
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Set, Tuple, Any

from trinity.types import QualityTier


class PostProcessEffect(Enum):
    """Available post-processing effects."""
    TONEMAPPING = auto()
    BLOOM = auto()
    DOF = auto()
    BOKEH_DOF = auto()
    MOTION_BLUR = auto()
    FXAA = auto()
    TAA = auto()
    SMAA = auto()
    CHROMATIC_ABERRATION = auto()
    FILM_GRAIN = auto()
    VIGNETTE = auto()
    AUTO_EXPOSURE = auto()
    COLOR_GRADING = auto()
    LENS_FLARE = auto()
    TEMPORAL_UPSCALING = auto()
    AMBIENT_OCCLUSION = auto()


class TonemapOperator(Enum):
    """Tonemapping operators."""
    LINEAR = auto()
    REINHARD = auto()
    REINHARD_EXTENDED = auto()
    ACES = auto()
    ACES_FITTED = auto()
    UNCHARTED2 = auto()
    NEUTRAL = auto()


class AAMethod(Enum):
    """Anti-aliasing methods."""
    NONE = auto()
    FXAA = auto()
    SMAA = auto()
    TAA = auto()
    MSAA_2X = auto()
    MSAA_4X = auto()


class UpscalingMethod(Enum):
    """Upscaling methods."""
    NONE = auto()
    BILINEAR = auto()
    FSR1 = auto()
    FSR2 = auto()
    DLSS = auto()


@dataclass
class BloomConfig:
    """Bloom effect configuration."""
    enabled: bool = True
    iterations: int = 5
    threshold: float = 0.8
    intensity: float = 1.0
    soft_knee: float = 0.5


@dataclass
class DOFConfig:
    """Depth of field configuration."""
    enabled: bool = False
    samples: int = 8
    focus_distance: float = 10.0
    aperture: float = 2.8
    use_bokeh: bool = False
    bokeh_shape: str = "circular"


@dataclass
class MotionBlurConfig:
    """Motion blur configuration."""
    enabled: bool = False
    samples: int = 8
    intensity: float = 1.0
    max_velocity: float = 32.0


@dataclass
class TAAConfig:
    """Temporal anti-aliasing configuration."""
    enabled: bool = False
    jitter_samples: int = 8
    history_weight: float = 0.9
    sharpening: float = 0.0


@dataclass
class UpscalingConfig:
    """Upscaling configuration."""
    enabled: bool = False
    method: UpscalingMethod = UpscalingMethod.NONE
    render_scale: float = 1.0
    sharpness: float = 0.5


@dataclass
class PostProcessTierConfig:
    """Complete post-processing configuration for a quality tier."""
    tier: QualityTier
    enabled_effects: Set[PostProcessEffect] = field(default_factory=set)
    tonemap_operator: TonemapOperator = TonemapOperator.REINHARD
    aa_method: AAMethod = AAMethod.NONE
    bloom: BloomConfig = field(default_factory=BloomConfig)
    dof: DOFConfig = field(default_factory=DOFConfig)
    motion_blur: MotionBlurConfig = field(default_factory=MotionBlurConfig)
    taa: TAAConfig = field(default_factory=TAAConfig)
    upscaling: UpscalingConfig = field(default_factory=UpscalingConfig)
    render_scale: float = 1.0
    gpu_time_budget_ms: float = 1.0
    memory_budget_mb: int = 32

    @property
    def uses_temporal_effects(self) -> bool:
        return self.taa.enabled or self.upscaling.method in (
            UpscalingMethod.FSR2, UpscalingMethod.DLSS
        )

    @property
    def effect_count(self) -> int:
        return len(self.enabled_effects)


def create_low_tier_config() -> PostProcessTierConfig:
    """Create Low tier post-process config: tonemap + bloom + FXAA."""
    return PostProcessTierConfig(
        tier=QualityTier.LOW,
        enabled_effects={
            PostProcessEffect.TONEMAPPING,
            PostProcessEffect.BLOOM,
            PostProcessEffect.FXAA,
        },
        tonemap_operator=TonemapOperator.REINHARD,
        aa_method=AAMethod.FXAA,
        bloom=BloomConfig(enabled=True, iterations=3, threshold=1.0),
        dof=DOFConfig(enabled=False),
        motion_blur=MotionBlurConfig(enabled=False),
        taa=TAAConfig(enabled=False),
        upscaling=UpscalingConfig(enabled=False),
        render_scale=0.75,
        gpu_time_budget_ms=1.0,
        memory_budget_mb=32,
    )


def create_medium_tier_config() -> PostProcessTierConfig:
    """Create Medium tier post-process config: + DOF + TAA + vignette."""
    return PostProcessTierConfig(
        tier=QualityTier.MEDIUM,
        enabled_effects={
            PostProcessEffect.TONEMAPPING,
            PostProcessEffect.BLOOM,
            PostProcessEffect.DOF,
            PostProcessEffect.TAA,
            PostProcessEffect.VIGNETTE,
        },
        tonemap_operator=TonemapOperator.ACES,
        aa_method=AAMethod.TAA,
        bloom=BloomConfig(enabled=True, iterations=5, threshold=0.8),
        dof=DOFConfig(enabled=True, samples=8),
        motion_blur=MotionBlurConfig(enabled=False),
        taa=TAAConfig(enabled=True, jitter_samples=8),
        upscaling=UpscalingConfig(enabled=False),
        render_scale=1.0,
        gpu_time_budget_ms=2.0,
        memory_budget_mb=64,
    )


def create_high_tier_config() -> PostProcessTierConfig:
    """Create High tier post-process config: full stack."""
    return PostProcessTierConfig(
        tier=QualityTier.HIGH,
        enabled_effects={
            PostProcessEffect.TONEMAPPING,
            PostProcessEffect.BLOOM,
            PostProcessEffect.DOF,
            PostProcessEffect.MOTION_BLUR,
            PostProcessEffect.TAA,
            PostProcessEffect.CHROMATIC_ABERRATION,
            PostProcessEffect.VIGNETTE,
            PostProcessEffect.AUTO_EXPOSURE,
            PostProcessEffect.COLOR_GRADING,
        },
        tonemap_operator=TonemapOperator.ACES_FITTED,
        aa_method=AAMethod.TAA,
        bloom=BloomConfig(enabled=True, iterations=6, threshold=0.6),
        dof=DOFConfig(enabled=True, samples=16),
        motion_blur=MotionBlurConfig(enabled=True, samples=8),
        taa=TAAConfig(enabled=True, jitter_samples=16, sharpening=0.5),
        upscaling=UpscalingConfig(enabled=False),
        render_scale=1.0,
        gpu_time_budget_ms=3.0,
        memory_budget_mb=128,
    )


def create_ultra_tier_config() -> PostProcessTierConfig:
    """Create Ultra tier post-process config: + upscaling + bokeh + lens flare."""
    return PostProcessTierConfig(
        tier=QualityTier.ULTRA,
        enabled_effects={
            PostProcessEffect.TONEMAPPING,
            PostProcessEffect.BLOOM,
            PostProcessEffect.BOKEH_DOF,
            PostProcessEffect.MOTION_BLUR,
            PostProcessEffect.TAA,
            PostProcessEffect.CHROMATIC_ABERRATION,
            PostProcessEffect.FILM_GRAIN,
            PostProcessEffect.VIGNETTE,
            PostProcessEffect.AUTO_EXPOSURE,
            PostProcessEffect.COLOR_GRADING,
            PostProcessEffect.LENS_FLARE,
            PostProcessEffect.TEMPORAL_UPSCALING,
        },
        tonemap_operator=TonemapOperator.ACES_FITTED,
        aa_method=AAMethod.TAA,
        bloom=BloomConfig(enabled=True, iterations=8, threshold=0.5, soft_knee=0.7),
        dof=DOFConfig(enabled=True, samples=32, use_bokeh=True, bokeh_shape="circular"),
        motion_blur=MotionBlurConfig(enabled=True, samples=16),
        taa=TAAConfig(enabled=True, jitter_samples=32, sharpening=0.5),
        upscaling=UpscalingConfig(enabled=True, method=UpscalingMethod.FSR2, render_scale=0.67),
        render_scale=1.0,
        gpu_time_budget_ms=5.0,
        memory_budget_mb=256,
    )


TIER_CONFIGS: Dict[QualityTier, Callable[[], PostProcessTierConfig]] = {
    QualityTier.LOW: create_low_tier_config,
    QualityTier.MEDIUM: create_medium_tier_config,
    QualityTier.HIGH: create_high_tier_config,
    QualityTier.ULTRA: create_ultra_tier_config,
}


@dataclass
class EffectTimingStats:
    """Timing statistics for post-process effects."""
    effect: PostProcessEffect
    gpu_time_ms: float = 0.0
    invocation_count: int = 0

    @property
    def avg_time_ms(self) -> float:
        return self.gpu_time_ms / max(1, self.invocation_count)


class TierChangeListener:
    """Protocol for tier change notifications."""
    def on_tier_changed(self, old_tier: QualityTier, new_tier: QualityTier, config: PostProcessTierConfig) -> None:
        pass


class PostProcessTierManager:
    """Manages quality tier integration for post-processing."""

    def __init__(self, initial_tier: QualityTier = QualityTier.MEDIUM):
        self._current_tier = initial_tier
        self._config = TIER_CONFIGS[initial_tier]()
        self._listeners: List[TierChangeListener] = []
        self._effect_overrides: Dict[PostProcessEffect, bool] = {}
        self._timing_stats: Dict[PostProcessEffect, EffectTimingStats] = {}
        self._frame_budget_exceeded_count: int = 0

    @property
    def current_tier(self) -> QualityTier:
        return self._current_tier

    @property
    def config(self) -> PostProcessTierConfig:
        return self._config

    def set_tier(self, tier: QualityTier) -> None:
        """Set the quality tier and update configuration."""
        if tier == self._current_tier:
            return

        old_tier = self._current_tier
        self._current_tier = tier
        self._config = TIER_CONFIGS[tier]()
        self._apply_overrides()
        self._frame_budget_exceeded_count = 0

        for listener in self._listeners:
            listener.on_tier_changed(old_tier, tier, self._config)

    def _apply_overrides(self) -> None:
        """Apply user overrides to the current config."""
        for effect, enabled in self._effect_overrides.items():
            if enabled:
                self._config.enabled_effects.add(effect)
            else:
                self._config.enabled_effects.discard(effect)

    def add_listener(self, listener: TierChangeListener) -> None:
        if listener not in self._listeners:
            self._listeners.append(listener)

    def remove_listener(self, listener: TierChangeListener) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def override_effect(self, effect: PostProcessEffect, enabled: bool) -> None:
        """Override an effect setting regardless of tier."""
        self._effect_overrides[effect] = enabled
        self._apply_overrides()

    def clear_overrides(self) -> None:
        """Clear all effect overrides."""
        self._effect_overrides.clear()
        self._config = TIER_CONFIGS[self._current_tier]()

    def is_effect_enabled(self, effect: PostProcessEffect) -> bool:
        """Check if an effect is enabled in current config."""
        return effect in self._config.enabled_effects

    def get_bloom_config(self) -> BloomConfig:
        return self._config.bloom

    def get_dof_config(self) -> DOFConfig:
        return self._config.dof

    def get_motion_blur_config(self) -> MotionBlurConfig:
        return self._config.motion_blur

    def get_taa_config(self) -> TAAConfig:
        return self._config.taa

    def get_upscaling_config(self) -> UpscalingConfig:
        return self._config.upscaling

    def get_tonemap_operator(self) -> TonemapOperator:
        return self._config.tonemap_operator

    def get_aa_method(self) -> AAMethod:
        return self._config.aa_method

    def get_render_scale(self) -> float:
        return self._config.render_scale

    def get_gpu_budget_ms(self) -> float:
        return self._config.gpu_time_budget_ms

    def get_memory_budget_mb(self) -> int:
        return self._config.memory_budget_mb

    def record_effect_timing(self, effect: PostProcessEffect, gpu_time_ms: float) -> None:
        """Record timing for an effect execution."""
        if effect not in self._timing_stats:
            self._timing_stats[effect] = EffectTimingStats(effect)
        stats = self._timing_stats[effect]
        stats.gpu_time_ms += gpu_time_ms
        stats.invocation_count += 1

    def get_total_frame_time(self) -> float:
        """Get total GPU time for all post-process effects."""
        return sum(s.gpu_time_ms for s in self._timing_stats.values())

    def check_budget(self) -> bool:
        """Check if within GPU time budget. Returns False if exceeded."""
        total = self.get_total_frame_time()
        if total > self._config.gpu_time_budget_ms:
            self._frame_budget_exceeded_count += 1
            return False
        return True

    def reset_frame_stats(self) -> None:
        """Reset per-frame timing statistics."""
        for stats in self._timing_stats.values():
            stats.gpu_time_ms = 0.0
            stats.invocation_count = 0

    def should_auto_downgrade(self, consecutive_frames: int = 10) -> bool:
        """Check if tier should be automatically downgraded due to budget."""
        return self._frame_budget_exceeded_count >= consecutive_frames

    def auto_downgrade(self) -> bool:
        """Attempt to downgrade tier. Returns True if downgraded."""
        tier_order = [QualityTier.ULTRA, QualityTier.HIGH, QualityTier.MEDIUM, QualityTier.LOW]
        current_idx = tier_order.index(self._current_tier)
        if current_idx < len(tier_order) - 1:
            self.set_tier(tier_order[current_idx + 1])
            return True
        return False

    def get_enabled_effects_list(self) -> List[PostProcessEffect]:
        """Get list of enabled effects in render order."""
        render_order = [
            PostProcessEffect.AUTO_EXPOSURE,
            PostProcessEffect.AMBIENT_OCCLUSION,
            PostProcessEffect.DOF,
            PostProcessEffect.BOKEH_DOF,
            PostProcessEffect.MOTION_BLUR,
            PostProcessEffect.BLOOM,
            PostProcessEffect.LENS_FLARE,
            PostProcessEffect.COLOR_GRADING,
            PostProcessEffect.TONEMAPPING,
            PostProcessEffect.CHROMATIC_ABERRATION,
            PostProcessEffect.FILM_GRAIN,
            PostProcessEffect.VIGNETTE,
            PostProcessEffect.FXAA,
            PostProcessEffect.SMAA,
            PostProcessEffect.TAA,
            PostProcessEffect.TEMPORAL_UPSCALING,
        ]
        return [e for e in render_order if e in self._config.enabled_effects]

    def get_status_dict(self) -> Dict[str, Any]:
        """Get status information as a dictionary."""
        return {
            "tier": self._current_tier.name,
            "effect_count": self._config.effect_count,
            "tonemap": self._config.tonemap_operator.name,
            "aa_method": self._config.aa_method.name,
            "bloom_enabled": self._config.bloom.enabled,
            "dof_enabled": self._config.dof.enabled,
            "motion_blur_enabled": self._config.motion_blur.enabled,
            "taa_enabled": self._config.taa.enabled,
            "upscaling_enabled": self._config.upscaling.enabled,
            "render_scale": self._config.render_scale,
            "gpu_budget_ms": self._config.gpu_time_budget_ms,
            "frame_budget_exceeded": self._frame_budget_exceeded_count,
        }


def get_tier_for_effects(required_effects: Set[PostProcessEffect]) -> QualityTier:
    """Suggest minimum tier that supports the required effects."""
    for tier in [QualityTier.LOW, QualityTier.MEDIUM, QualityTier.HIGH, QualityTier.ULTRA]:
        config = TIER_CONFIGS[tier]()
        if required_effects.issubset(config.enabled_effects):
            return tier
    return QualityTier.ULTRA


def estimate_postprocess_memory(
    config: PostProcessTierConfig,
    screen_width: int,
    screen_height: int,
) -> int:
    """Estimate post-processing memory usage in bytes."""
    memory = 0
    pixels = screen_width * screen_height

    # HDR color buffer (always needed)
    memory += pixels * 8  # RGBA16F

    # Bloom (mip chain)
    if config.bloom.enabled:
        bloom_pixels = pixels
        for _ in range(config.bloom.iterations):
            bloom_pixels //= 4
            memory += bloom_pixels * 8  # RGBA16F per level

    # DOF
    if PostProcessEffect.DOF in config.enabled_effects:
        memory += pixels * 8  # CoC buffer + blur buffers

    # Motion blur
    if config.motion_blur.enabled:
        memory += pixels * 4  # Velocity buffer (RG16F)

    # TAA history
    if config.taa.enabled:
        memory += pixels * 8 * 2  # Two history buffers

    # Upscaling
    if config.upscaling.enabled:
        internal_pixels = int(pixels * config.upscaling.render_scale ** 2)
        memory += internal_pixels * 8  # Internal render target

    return memory


def create_effect_pass_list(config: PostProcessTierConfig) -> List[Dict[str, Any]]:
    """Create a list of post-process passes for the frame graph."""
    passes = []

    if PostProcessEffect.AUTO_EXPOSURE in config.enabled_effects:
        passes.append({
            "name": "auto_exposure",
            "type": "compute",
            "inputs": ["hdr_color"],
            "outputs": ["exposure_value"],
        })

    if config.bloom.enabled:
        passes.append({
            "name": "bloom_downsample",
            "type": "compute",
            "iterations": config.bloom.iterations,
            "inputs": ["hdr_color"],
            "outputs": ["bloom_mips"],
        })
        passes.append({
            "name": "bloom_upsample",
            "type": "compute",
            "iterations": config.bloom.iterations,
            "inputs": ["bloom_mips"],
            "outputs": ["bloom_result"],
        })

    if config.dof.enabled:
        passes.append({
            "name": "dof",
            "type": "fragment",
            "bokeh": config.dof.use_bokeh,
            "samples": config.dof.samples,
            "inputs": ["hdr_color", "depth"],
            "outputs": ["dof_result"],
        })

    if config.motion_blur.enabled:
        passes.append({
            "name": "motion_blur",
            "type": "fragment",
            "samples": config.motion_blur.samples,
            "inputs": ["hdr_color", "velocity"],
            "outputs": ["motion_blur_result"],
        })

    passes.append({
        "name": "tonemap",
        "type": "fragment",
        "operator": config.tonemap_operator.name,
        "inputs": ["hdr_color", "bloom_result"],
        "outputs": ["ldr_color"],
    })

    if config.taa.enabled:
        passes.append({
            "name": "taa",
            "type": "fragment",
            "jitter_samples": config.taa.jitter_samples,
            "inputs": ["ldr_color", "depth", "velocity", "history"],
            "outputs": ["taa_result", "new_history"],
        })

    return passes
