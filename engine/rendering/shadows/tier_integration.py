"""T-CC-1.4: Quality tier integration for Shadows subsystem (S5).

Wires quality tier settings into runtime shadow configuration:
- Low: PCF 512, 1 cascade
- Medium: PCF 1024, 2 cascades
- High: VSM 2048, 4 cascades + contact shadows
- Ultra: RT shadows + PCSS 4096
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Set, Any

from trinity.types import QualityTier


class ShadowFilterMethod(Enum):
    """Shadow filtering methods."""
    HARD = auto()       # No filtering
    PCF = auto()        # Percentage Closer Filtering
    VSM = auto()        # Variance Shadow Maps
    PCSS = auto()       # Percentage Closer Soft Shadows
    RT = auto()         # Ray-traced shadows


class ShadowFeature(Enum):
    """Shadow features that can be enabled per tier."""
    DIRECTIONAL_SHADOW = auto()
    SPOT_SHADOW = auto()
    POINT_SHADOW = auto()
    AREA_SHADOW = auto()
    CASCADED_SHADOWS = auto()
    CONTACT_SHADOWS = auto()
    PCF_FILTERING = auto()
    VSM_FILTERING = auto()
    PCSS_FILTERING = auto()
    RAY_TRACED_SHADOWS = auto()


@dataclass
class CascadeConfig:
    """Cascade shadow map configuration."""
    count: int = 4
    split_lambda: float = 0.5
    blend_distance: float = 5.0
    stabilize: bool = True


@dataclass
class PCFConfig:
    """PCF filtering configuration."""
    samples: int = 9
    radius: float = 1.0
    use_rotated_poisson: bool = True


@dataclass
class VSMConfig:
    """VSM filtering configuration."""
    blur_samples: int = 5
    min_variance: float = 0.0001
    light_bleed_reduction: float = 0.2


@dataclass
class PCSSConfig:
    """PCSS filtering configuration."""
    blocker_samples: int = 16
    pcf_samples: int = 32
    light_size: float = 0.02


@dataclass
class ContactShadowConfig:
    """Contact shadow (screen-space) configuration."""
    enabled: bool = False
    steps: int = 16
    ray_length: float = 0.1
    thickness: float = 0.05


@dataclass
class RTShadowConfig:
    """Ray-traced shadow configuration."""
    enabled: bool = False
    rays_per_pixel: int = 2
    denoise: bool = True
    soft_shadows: bool = True


@dataclass
class ShadowTierConfig:
    """Complete shadow configuration for a quality tier."""
    tier: QualityTier
    resolution: int = 1024
    filter_method: ShadowFilterMethod = ShadowFilterMethod.PCF
    enabled_features: Set[ShadowFeature] = field(default_factory=set)
    cascade_config: CascadeConfig = field(default_factory=CascadeConfig)
    pcf_config: PCFConfig = field(default_factory=PCFConfig)
    vsm_config: VSMConfig = field(default_factory=VSMConfig)
    pcss_config: PCSSConfig = field(default_factory=PCSSConfig)
    contact_config: ContactShadowConfig = field(default_factory=ContactShadowConfig)
    rt_config: RTShadowConfig = field(default_factory=RTShadowConfig)
    shadow_distance: float = 100.0
    shadow_fade: float = 0.85
    max_spot_shadows: int = 4
    max_point_shadows: int = 2
    gpu_time_budget_ms: float = 1.0
    memory_budget_mb: int = 32

    @property
    def uses_cascades(self) -> bool:
        return ShadowFeature.CASCADED_SHADOWS in self.enabled_features

    @property
    def uses_ray_tracing(self) -> bool:
        return ShadowFeature.RAY_TRACED_SHADOWS in self.enabled_features

    @property
    def supports_soft_shadows(self) -> bool:
        return self.filter_method in (ShadowFilterMethod.PCSS, ShadowFilterMethod.RT)


def create_low_tier_config() -> ShadowTierConfig:
    """Create Low tier shadow config: PCF 512, 1 cascade."""
    return ShadowTierConfig(
        tier=QualityTier.LOW,
        resolution=512,
        filter_method=ShadowFilterMethod.PCF,
        enabled_features={
            ShadowFeature.DIRECTIONAL_SHADOW,
            ShadowFeature.PCF_FILTERING,
        },
        cascade_config=CascadeConfig(count=1),
        pcf_config=PCFConfig(samples=4, use_rotated_poisson=False),
        contact_config=ContactShadowConfig(enabled=False),
        rt_config=RTShadowConfig(enabled=False),
        shadow_distance=50.0,
        shadow_fade=0.9,
        max_spot_shadows=2,
        max_point_shadows=0,
        gpu_time_budget_ms=0.5,
        memory_budget_mb=8,
    )


def create_medium_tier_config() -> ShadowTierConfig:
    """Create Medium tier shadow config: PCF 1024, 2 cascades."""
    return ShadowTierConfig(
        tier=QualityTier.MEDIUM,
        resolution=1024,
        filter_method=ShadowFilterMethod.PCF,
        enabled_features={
            ShadowFeature.DIRECTIONAL_SHADOW,
            ShadowFeature.SPOT_SHADOW,
            ShadowFeature.CASCADED_SHADOWS,
            ShadowFeature.PCF_FILTERING,
        },
        cascade_config=CascadeConfig(count=2),
        pcf_config=PCFConfig(samples=9),
        contact_config=ContactShadowConfig(enabled=False),
        rt_config=RTShadowConfig(enabled=False),
        shadow_distance=100.0,
        shadow_fade=0.85,
        max_spot_shadows=4,
        max_point_shadows=1,
        gpu_time_budget_ms=1.0,
        memory_budget_mb=32,
    )


def create_high_tier_config() -> ShadowTierConfig:
    """Create High tier shadow config: VSM 2048, 4 cascades + contact."""
    return ShadowTierConfig(
        tier=QualityTier.HIGH,
        resolution=2048,
        filter_method=ShadowFilterMethod.VSM,
        enabled_features={
            ShadowFeature.DIRECTIONAL_SHADOW,
            ShadowFeature.SPOT_SHADOW,
            ShadowFeature.POINT_SHADOW,
            ShadowFeature.CASCADED_SHADOWS,
            ShadowFeature.VSM_FILTERING,
            ShadowFeature.CONTACT_SHADOWS,
        },
        cascade_config=CascadeConfig(count=4),
        vsm_config=VSMConfig(blur_samples=5),
        contact_config=ContactShadowConfig(enabled=True, steps=16),
        rt_config=RTShadowConfig(enabled=False),
        shadow_distance=200.0,
        shadow_fade=0.8,
        max_spot_shadows=8,
        max_point_shadows=4,
        gpu_time_budget_ms=2.0,
        memory_budget_mb=64,
    )


def create_ultra_tier_config() -> ShadowTierConfig:
    """Create Ultra tier shadow config: RT + PCSS 4096."""
    return ShadowTierConfig(
        tier=QualityTier.ULTRA,
        resolution=4096,
        filter_method=ShadowFilterMethod.RT,
        enabled_features={
            ShadowFeature.DIRECTIONAL_SHADOW,
            ShadowFeature.SPOT_SHADOW,
            ShadowFeature.POINT_SHADOW,
            ShadowFeature.AREA_SHADOW,
            ShadowFeature.CASCADED_SHADOWS,
            ShadowFeature.PCSS_FILTERING,
            ShadowFeature.RAY_TRACED_SHADOWS,
            ShadowFeature.CONTACT_SHADOWS,
        },
        cascade_config=CascadeConfig(count=4),
        pcss_config=PCSSConfig(blocker_samples=16, pcf_samples=32),
        contact_config=ContactShadowConfig(enabled=True, steps=32),
        rt_config=RTShadowConfig(enabled=True, rays_per_pixel=2, denoise=True),
        shadow_distance=500.0,
        shadow_fade=0.75,
        max_spot_shadows=16,
        max_point_shadows=8,
        gpu_time_budget_ms=3.0,
        memory_budget_mb=128,
    )


TIER_CONFIGS: Dict[QualityTier, Callable[[], ShadowTierConfig]] = {
    QualityTier.LOW: create_low_tier_config,
    QualityTier.MEDIUM: create_medium_tier_config,
    QualityTier.HIGH: create_high_tier_config,
    QualityTier.ULTRA: create_ultra_tier_config,
}


@dataclass
class ShadowUsageStats:
    """Tracks shadow caster usage."""
    active_spot_shadows: int = 0
    active_point_shadows: int = 0
    cascade_renders_this_frame: int = 0


class TierChangeListener:
    """Protocol for tier change notifications."""
    def on_tier_changed(self, old_tier: QualityTier, new_tier: QualityTier, config: ShadowTierConfig) -> None:
        pass


class ShadowTierManager:
    """Manages quality tier integration for the shadows subsystem."""

    def __init__(self, initial_tier: QualityTier = QualityTier.MEDIUM):
        self._current_tier = initial_tier
        self._config = TIER_CONFIGS[initial_tier]()
        self._listeners: List[TierChangeListener] = []
        self._feature_overrides: Dict[ShadowFeature, bool] = {}
        self._usage_stats = ShadowUsageStats()
        self._rt_available = False

    @property
    def current_tier(self) -> QualityTier:
        return self._current_tier

    @property
    def config(self) -> ShadowTierConfig:
        return self._config

    @property
    def usage_stats(self) -> ShadowUsageStats:
        return self._usage_stats

    def set_rt_available(self, available: bool) -> None:
        """Set whether ray tracing is available on this hardware."""
        self._rt_available = available

    def set_tier(self, tier: QualityTier) -> None:
        """Set the quality tier and update configuration."""
        if tier == self._current_tier:
            return

        old_tier = self._current_tier
        self._current_tier = tier
        self._config = TIER_CONFIGS[tier]()
        self._apply_overrides()
        self._usage_stats = ShadowUsageStats()

        for listener in self._listeners:
            listener.on_tier_changed(old_tier, tier, self._config)

    def _apply_overrides(self) -> None:
        """Apply user overrides to the current config."""
        for feature, enabled in self._feature_overrides.items():
            if enabled:
                self._config.enabled_features.add(feature)
            else:
                self._config.enabled_features.discard(feature)

    def add_listener(self, listener: TierChangeListener) -> None:
        if listener not in self._listeners:
            self._listeners.append(listener)

    def remove_listener(self, listener: TierChangeListener) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def override_feature(self, feature: ShadowFeature, enabled: bool) -> None:
        """Override a feature setting regardless of tier."""
        self._feature_overrides[feature] = enabled
        self._apply_overrides()

    def clear_overrides(self) -> None:
        """Clear all feature overrides."""
        self._feature_overrides.clear()
        self._config = TIER_CONFIGS[self._current_tier]()

    def is_feature_enabled(self, feature: ShadowFeature) -> bool:
        """Check if a feature is enabled in current config."""
        return feature in self._config.enabled_features

    def get_resolution(self) -> int:
        """Get shadow map resolution."""
        return self._config.resolution

    def get_filter_method(self) -> ShadowFilterMethod:
        """Get active shadow filter method."""
        # Fall back from RT if not available
        if self._config.filter_method == ShadowFilterMethod.RT and not self._rt_available:
            return ShadowFilterMethod.PCSS
        return self._config.filter_method

    def get_cascade_config(self) -> CascadeConfig:
        return self._config.cascade_config

    def get_cascade_count(self) -> int:
        if self._config.uses_cascades:
            return self._config.cascade_config.count
        return 1

    def get_pcf_config(self) -> PCFConfig:
        return self._config.pcf_config

    def get_vsm_config(self) -> VSMConfig:
        return self._config.vsm_config

    def get_pcss_config(self) -> PCSSConfig:
        return self._config.pcss_config

    def get_contact_config(self) -> ContactShadowConfig:
        return self._config.contact_config

    def get_rt_config(self) -> RTShadowConfig:
        return self._config.rt_config

    def get_shadow_distance(self) -> float:
        return self._config.shadow_distance

    def get_shadow_fade(self) -> float:
        return self._config.shadow_fade

    def can_add_spot_shadow(self) -> bool:
        return self._usage_stats.active_spot_shadows < self._config.max_spot_shadows

    def can_add_point_shadow(self) -> bool:
        return self._usage_stats.active_point_shadows < self._config.max_point_shadows

    def register_spot_shadow(self) -> bool:
        if not self.can_add_spot_shadow():
            return False
        self._usage_stats.active_spot_shadows += 1
        return True

    def register_point_shadow(self) -> bool:
        if not self.can_add_point_shadow():
            return False
        self._usage_stats.active_point_shadows += 1
        return True

    def unregister_spot_shadow(self) -> None:
        if self._usage_stats.active_spot_shadows > 0:
            self._usage_stats.active_spot_shadows -= 1

    def unregister_point_shadow(self) -> None:
        if self._usage_stats.active_point_shadows > 0:
            self._usage_stats.active_point_shadows -= 1

    def begin_frame(self) -> None:
        """Reset per-frame stats."""
        self._usage_stats.cascade_renders_this_frame = 0

    def record_cascade_render(self) -> None:
        """Record a cascade render pass."""
        self._usage_stats.cascade_renders_this_frame += 1

    def get_gpu_budget_ms(self) -> float:
        return self._config.gpu_time_budget_ms

    def get_memory_budget_mb(self) -> int:
        return self._config.memory_budget_mb

    def select_fallback_filter(self) -> ShadowFilterMethod:
        """Select best available filter method based on hardware."""
        fallback_chain = [
            ShadowFilterMethod.RT,
            ShadowFilterMethod.PCSS,
            ShadowFilterMethod.VSM,
            ShadowFilterMethod.PCF,
            ShadowFilterMethod.HARD,
        ]

        for method in fallback_chain:
            if method == ShadowFilterMethod.RT and not self._rt_available:
                continue
            if self._can_use_filter(method):
                return method
        return ShadowFilterMethod.HARD

    def _can_use_filter(self, method: ShadowFilterMethod) -> bool:
        """Check if filter method is supported by current tier."""
        method_features = {
            ShadowFilterMethod.RT: ShadowFeature.RAY_TRACED_SHADOWS,
            ShadowFilterMethod.PCSS: ShadowFeature.PCSS_FILTERING,
            ShadowFilterMethod.VSM: ShadowFeature.VSM_FILTERING,
            ShadowFilterMethod.PCF: ShadowFeature.PCF_FILTERING,
            ShadowFilterMethod.HARD: None,
        }
        feature = method_features.get(method)
        if feature is None:
            return True
        return feature in self._config.enabled_features

    def get_status_dict(self) -> Dict[str, Any]:
        """Get status information as a dictionary."""
        return {
            "tier": self._current_tier.name,
            "resolution": self._config.resolution,
            "filter_method": self.get_filter_method().name,
            "cascade_count": self.get_cascade_count(),
            "uses_cascades": self._config.uses_cascades,
            "uses_ray_tracing": self._config.uses_ray_tracing and self._rt_available,
            "supports_soft_shadows": self._config.supports_soft_shadows,
            "contact_shadows_enabled": self._config.contact_config.enabled,
            "shadow_distance": self._config.shadow_distance,
            "active_spot_shadows": self._usage_stats.active_spot_shadows,
            "active_point_shadows": self._usage_stats.active_point_shadows,
            "rt_available": self._rt_available,
        }


def get_tier_for_resolution(resolution: int) -> QualityTier:
    """Suggest minimum tier for the desired resolution."""
    if resolution <= 512:
        return QualityTier.LOW
    elif resolution <= 1024:
        return QualityTier.MEDIUM
    elif resolution <= 2048:
        return QualityTier.HIGH
    else:
        return QualityTier.ULTRA


def get_tier_for_filter(method: ShadowFilterMethod) -> QualityTier:
    """Suggest minimum tier for the desired filter method."""
    tier_map = {
        ShadowFilterMethod.HARD: QualityTier.LOW,
        ShadowFilterMethod.PCF: QualityTier.LOW,
        ShadowFilterMethod.VSM: QualityTier.HIGH,
        ShadowFilterMethod.PCSS: QualityTier.ULTRA,
        ShadowFilterMethod.RT: QualityTier.ULTRA,
    }
    return tier_map.get(method, QualityTier.LOW)


def estimate_shadow_memory(config: ShadowTierConfig) -> int:
    """Estimate shadow system memory usage in bytes."""
    memory = 0

    # Directional shadow cascades
    cascade_size = config.resolution * config.resolution
    if config.filter_method == ShadowFilterMethod.VSM:
        bytes_per_pixel = 8  # RG32F
    else:
        bytes_per_pixel = 4  # D32F
    memory += cascade_size * bytes_per_pixel * config.cascade_config.count

    # Spot shadows
    spot_size = config.resolution // 2
    memory += spot_size * spot_size * bytes_per_pixel * config.max_spot_shadows

    # Point shadows (cube maps)
    point_size = config.resolution // 4
    memory += point_size * point_size * 6 * bytes_per_pixel * config.max_point_shadows

    # RT shadow buffers
    if config.rt_config.enabled:
        # Estimate 1080p shadow buffer
        memory += 1920 * 1080 * 4 * 2  # Two buffers for denoising

    return memory


def calculate_cascade_splits(
    near: float,
    far: float,
    count: int,
    lambda_: float = 0.5,
) -> List[float]:
    """Calculate cascade split distances using logarithmic distribution."""
    splits = []
    for i in range(count + 1):
        uniform = near + (far - near) * (i / count)
        log = near * ((far / near) ** (i / count))
        split = lambda_ * log + (1 - lambda_) * uniform
        splits.append(split)
    return splits
