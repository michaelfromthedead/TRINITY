"""T-CC-1.1: Quality tier integration for Lighting subsystem (S4).

Wires quality tier settings into runtime lighting configuration:
- Low: 8 lights forward
- Medium: 64 lights clustered
- High: 256 lights tiled+clustered
- Ultra: Unlimited GPU-driven
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Set, Tuple, Any

from trinity.types import QualityTier


class LightCullingMode(Enum):
    """Light culling algorithm selection."""
    NONE = auto()           # No culling, iterate all lights
    PER_OBJECT = auto()     # Simple per-object light list
    CLUSTERED = auto()      # 3D froxel-based clustering
    TILED = auto()          # 2D tile-based (compute shader)
    TILED_CLUSTERED = auto()  # Combined tiled + clustered
    GPU_DRIVEN = auto()     # Full GPU-driven with indirect dispatch


class LightingFeature(Enum):
    """Lighting features that can be enabled per tier."""
    DIRECTIONAL_LIGHT = auto()
    POINT_LIGHTS = auto()
    SPOT_LIGHTS = auto()
    AREA_LIGHTS = auto()
    LIGHT_COOKIES = auto()
    IES_PROFILES = auto()
    LTC_AREA_LIGHTS = auto()
    VOLUMETRIC_LIGHTING = auto()
    LIGHT_PROBES = auto()
    EMISSIVE_LIGHTING = auto()
    CLUSTERED_LIGHTING = auto()
    DEFERRED_SHADING = auto()
    FORWARD_SHADING = auto()


@dataclass
class ClusterConfig:
    """Configuration for clustered lighting."""
    tiles_x: int = 16
    tiles_y: int = 8
    slices_z: int = 24
    use_exponential_depth: bool = True


@dataclass
class LightingTierConfig:
    """Complete lighting configuration for a quality tier."""
    tier: QualityTier
    max_lights: int
    max_point_lights: int
    max_spot_lights: int
    max_area_lights: int = 0
    culling_mode: LightCullingMode = LightCullingMode.NONE
    cluster_config: Optional[ClusterConfig] = None
    enabled_features: Set[LightingFeature] = field(default_factory=set)
    volumetric_samples: int = 0
    shadow_resolution: int = 512
    gpu_time_budget_ms: float = 1.0
    memory_budget_mb: int = 32

    @property
    def uses_clustering(self) -> bool:
        return self.culling_mode in (
            LightCullingMode.CLUSTERED,
            LightCullingMode.TILED_CLUSTERED,
            LightCullingMode.GPU_DRIVEN,
        )

    @property
    def uses_deferred(self) -> bool:
        return LightingFeature.DEFERRED_SHADING in self.enabled_features

    @property
    def supports_area_lights(self) -> bool:
        return LightingFeature.AREA_LIGHTS in self.enabled_features


def create_low_tier_config() -> LightingTierConfig:
    """Create Low tier lighting config: 8 lights, forward, per-object culling."""
    return LightingTierConfig(
        tier=QualityTier.LOW,
        max_lights=8,
        max_point_lights=4,
        max_spot_lights=4,
        max_area_lights=0,
        culling_mode=LightCullingMode.PER_OBJECT,
        cluster_config=None,
        enabled_features={
            LightingFeature.DIRECTIONAL_LIGHT,
            LightingFeature.POINT_LIGHTS,
            LightingFeature.FORWARD_SHADING,
        },
        shadow_resolution=512,
        gpu_time_budget_ms=1.0,
        memory_budget_mb=32,
    )


def create_medium_tier_config() -> LightingTierConfig:
    """Create Medium tier lighting config: 64 lights, clustered forward."""
    return LightingTierConfig(
        tier=QualityTier.MEDIUM,
        max_lights=64,
        max_point_lights=32,
        max_spot_lights=32,
        max_area_lights=0,
        culling_mode=LightCullingMode.CLUSTERED,
        cluster_config=ClusterConfig(tiles_x=16, tiles_y=8, slices_z=24),
        enabled_features={
            LightingFeature.DIRECTIONAL_LIGHT,
            LightingFeature.POINT_LIGHTS,
            LightingFeature.SPOT_LIGHTS,
            LightingFeature.CLUSTERED_LIGHTING,
            LightingFeature.FORWARD_SHADING,
            LightingFeature.LIGHT_COOKIES,
        },
        shadow_resolution=1024,
        gpu_time_budget_ms=2.0,
        memory_budget_mb=64,
    )


def create_high_tier_config() -> LightingTierConfig:
    """Create High tier lighting config: 256 lights, tiled+clustered deferred."""
    return LightingTierConfig(
        tier=QualityTier.HIGH,
        max_lights=256,
        max_point_lights=128,
        max_spot_lights=64,
        max_area_lights=32,
        culling_mode=LightCullingMode.TILED_CLUSTERED,
        cluster_config=ClusterConfig(tiles_x=16, tiles_y=8, slices_z=32),
        enabled_features={
            LightingFeature.DIRECTIONAL_LIGHT,
            LightingFeature.POINT_LIGHTS,
            LightingFeature.SPOT_LIGHTS,
            LightingFeature.AREA_LIGHTS,
            LightingFeature.CLUSTERED_LIGHTING,
            LightingFeature.DEFERRED_SHADING,
            LightingFeature.LIGHT_COOKIES,
            LightingFeature.IES_PROFILES,
            LightingFeature.LTC_AREA_LIGHTS,
        },
        shadow_resolution=2048,
        gpu_time_budget_ms=3.0,
        memory_budget_mb=128,
    )


def create_ultra_tier_config() -> LightingTierConfig:
    """Create Ultra tier lighting config: Unlimited, GPU-driven with volumetric."""
    return LightingTierConfig(
        tier=QualityTier.ULTRA,
        max_lights=-1,  # Unlimited
        max_point_lights=-1,
        max_spot_lights=-1,
        max_area_lights=-1,
        culling_mode=LightCullingMode.GPU_DRIVEN,
        cluster_config=ClusterConfig(tiles_x=16, tiles_y=8, slices_z=48),
        enabled_features={
            LightingFeature.DIRECTIONAL_LIGHT,
            LightingFeature.POINT_LIGHTS,
            LightingFeature.SPOT_LIGHTS,
            LightingFeature.AREA_LIGHTS,
            LightingFeature.CLUSTERED_LIGHTING,
            LightingFeature.DEFERRED_SHADING,
            LightingFeature.LIGHT_COOKIES,
            LightingFeature.IES_PROFILES,
            LightingFeature.LTC_AREA_LIGHTS,
            LightingFeature.VOLUMETRIC_LIGHTING,
            LightingFeature.LIGHT_PROBES,
            LightingFeature.EMISSIVE_LIGHTING,
        },
        volumetric_samples=64,
        shadow_resolution=4096,
        gpu_time_budget_ms=5.0,
        memory_budget_mb=256,
    )


TIER_CONFIGS: Dict[QualityTier, Callable[[], LightingTierConfig]] = {
    QualityTier.LOW: create_low_tier_config,
    QualityTier.MEDIUM: create_medium_tier_config,
    QualityTier.HIGH: create_high_tier_config,
    QualityTier.ULTRA: create_ultra_tier_config,
}


@dataclass
class LightBudgetState:
    """Tracks current light budget usage."""
    point_lights_used: int = 0
    spot_lights_used: int = 0
    area_lights_used: int = 0

    @property
    def total_lights(self) -> int:
        return self.point_lights_used + self.spot_lights_used + self.area_lights_used

    def reset(self) -> None:
        self.point_lights_used = 0
        self.spot_lights_used = 0
        self.area_lights_used = 0


class TierChangeListener:
    """Protocol for tier change notifications."""
    def on_tier_changed(self, old_tier: QualityTier, new_tier: QualityTier, config: LightingTierConfig) -> None:
        pass


class LightingTierManager:
    """Manages quality tier integration for the lighting subsystem."""

    def __init__(self, initial_tier: QualityTier = QualityTier.MEDIUM):
        self._current_tier = initial_tier
        self._config = TIER_CONFIGS[initial_tier]()
        self._budget_state = LightBudgetState()
        self._listeners: List[TierChangeListener] = []
        self._feature_overrides: Dict[LightingFeature, bool] = {}
        self._parameter_overrides: Dict[str, Any] = {}

    @property
    def current_tier(self) -> QualityTier:
        return self._current_tier

    @property
    def config(self) -> LightingTierConfig:
        return self._config

    @property
    def budget_state(self) -> LightBudgetState:
        return self._budget_state

    def set_tier(self, tier: QualityTier) -> None:
        """Set the quality tier and update configuration."""
        if tier == self._current_tier:
            return

        old_tier = self._current_tier
        self._current_tier = tier
        self._config = TIER_CONFIGS[tier]()
        self._apply_overrides()
        self._budget_state.reset()

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

    def override_feature(self, feature: LightingFeature, enabled: bool) -> None:
        """Override a feature setting regardless of tier."""
        self._feature_overrides[feature] = enabled
        self._apply_overrides()

    def clear_overrides(self) -> None:
        """Clear all feature overrides."""
        self._feature_overrides.clear()
        self._config = TIER_CONFIGS[self._current_tier]()

    def is_feature_enabled(self, feature: LightingFeature) -> bool:
        """Check if a feature is enabled in current config."""
        return feature in self._config.enabled_features

    def can_add_light(self, light_type: str) -> bool:
        """Check if another light of the given type can be added."""
        config = self._config
        budget = self._budget_state

        if config.max_lights != -1 and budget.total_lights >= config.max_lights:
            return False

        if light_type == "point":
            if config.max_point_lights == -1:
                return True
            return budget.point_lights_used < config.max_point_lights
        elif light_type == "spot":
            if config.max_spot_lights == -1:
                return True
            return budget.spot_lights_used < config.max_spot_lights
        elif light_type == "area":
            if not config.supports_area_lights:
                return False
            if config.max_area_lights == -1:
                return True
            return budget.area_lights_used < config.max_area_lights

        return False

    def register_light(self, light_type: str) -> bool:
        """Register a light and return True if within budget."""
        if not self.can_add_light(light_type):
            return False

        if light_type == "point":
            self._budget_state.point_lights_used += 1
        elif light_type == "spot":
            self._budget_state.spot_lights_used += 1
        elif light_type == "area":
            self._budget_state.area_lights_used += 1

        return True

    def unregister_light(self, light_type: str) -> None:
        """Unregister a light from the budget."""
        if light_type == "point" and self._budget_state.point_lights_used > 0:
            self._budget_state.point_lights_used -= 1
        elif light_type == "spot" and self._budget_state.spot_lights_used > 0:
            self._budget_state.spot_lights_used -= 1
        elif light_type == "area" and self._budget_state.area_lights_used > 0:
            self._budget_state.area_lights_used -= 1

    def begin_frame(self) -> None:
        """Reset per-frame state."""
        self._budget_state.reset()

    def get_cluster_config(self) -> Optional[ClusterConfig]:
        """Get the cluster config for current tier, or None if not clustered."""
        if self._config.uses_clustering:
            return self._config.cluster_config
        return None

    def get_culling_mode(self) -> LightCullingMode:
        """Get the light culling mode for current tier."""
        return self._config.culling_mode

    def get_shadow_resolution(self) -> int:
        """Get the shadow map resolution for current tier."""
        return self._config.shadow_resolution

    def get_gpu_budget_ms(self) -> float:
        """Get the GPU time budget in milliseconds."""
        return self._config.gpu_time_budget_ms

    def get_memory_budget_mb(self) -> int:
        """Get the memory budget in megabytes."""
        return self._config.memory_budget_mb

    def get_volumetric_samples(self) -> int:
        """Get the number of volumetric lighting samples."""
        return self._config.volumetric_samples

    def get_status_dict(self) -> Dict[str, Any]:
        """Get status information as a dictionary."""
        return {
            "tier": self._current_tier.name,
            "max_lights": self._config.max_lights,
            "culling_mode": self._config.culling_mode.name,
            "uses_deferred": self._config.uses_deferred,
            "supports_area_lights": self._config.supports_area_lights,
            "lights_used": self._budget_state.total_lights,
            "point_lights": self._budget_state.point_lights_used,
            "spot_lights": self._budget_state.spot_lights_used,
            "area_lights": self._budget_state.area_lights_used,
            "shadow_resolution": self._config.shadow_resolution,
            "volumetric_samples": self._config.volumetric_samples,
        }


def get_tier_for_light_count(light_count: int) -> QualityTier:
    """Suggest a quality tier based on desired light count."""
    if light_count <= 8:
        return QualityTier.LOW
    elif light_count <= 64:
        return QualityTier.MEDIUM
    elif light_count <= 256:
        return QualityTier.HIGH
    else:
        return QualityTier.ULTRA


def estimate_lighting_memory(config: LightingTierConfig, screen_width: int, screen_height: int) -> int:
    """Estimate lighting memory usage in bytes."""
    memory = 0

    # Light data buffers
    light_struct_size = 96  # bytes per light
    if config.max_lights > 0:
        memory += config.max_lights * light_struct_size
    else:
        memory += 4096 * light_struct_size  # Estimate for unlimited

    # Cluster data
    if config.cluster_config:
        cc = config.cluster_config
        cluster_count = cc.tiles_x * cc.tiles_y * cc.slices_z
        # Each cluster stores up to 256 light indices (2 bytes each)
        memory += cluster_count * 256 * 2

    # Shadow maps
    shadow_size = config.shadow_resolution * config.shadow_resolution
    if config.max_point_lights > 0:
        memory += min(config.max_point_lights, 16) * 6 * shadow_size * 4  # Cube map, 4 bytes depth
    if config.max_spot_lights > 0:
        memory += min(config.max_spot_lights, 16) * shadow_size * 4

    # G-buffer for deferred (if applicable)
    if config.uses_deferred:
        gbuffer_size = screen_width * screen_height
        memory += gbuffer_size * 16  # 4 MRTs at 4 bytes each

    return memory
