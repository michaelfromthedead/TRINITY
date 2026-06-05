"""T-CC-1.3: Quality tier integration for Materials subsystem (S3).

Wires quality tier settings into runtime material configuration:
- Low: 1 variant, basic PBR only
- Medium: 3 variants, standard PBR + AO + emissive
- High: 10 variants, + SSS + clear coat + anisotropy
- Ultra: Unlimited variants, all advanced shading models
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Set, Tuple, Any, FrozenSet

from trinity.types import QualityTier


class MaterialFeature(Enum):
    """Material features that can be enabled per tier."""
    BASE_COLOR = auto()
    NORMAL_MAPPING = auto()
    ROUGHNESS_METALLIC = auto()
    AMBIENT_OCCLUSION = auto()
    EMISSIVE = auto()
    DETAIL_MAPS = auto()
    SUBSURFACE_SCATTERING = auto()
    CLEAR_COAT = auto()
    ANISOTROPY = auto()
    SHEEN = auto()
    TRANSMISSION = auto()
    IRIDESCENCE = auto()
    PARALLAX_MAPPING = auto()
    TESSELLATION = auto()
    DISPLACEMENT_MAPPING = auto()


class TextureFilterMode(Enum):
    """Texture filtering modes."""
    NEAREST = auto()
    BILINEAR = auto()
    TRILINEAR = auto()


@dataclass
class TextureConfig:
    """Texture quality configuration."""
    max_size: int = 1024
    lod_bias: float = 0.0
    filter_mode: TextureFilterMode = TextureFilterMode.TRILINEAR
    anisotropic_level: int = 4


@dataclass
class SSSConfig:
    """Subsurface scattering configuration."""
    enabled: bool = False
    samples: int = 8
    method: str = "separable"  # separable, simple, diffuse_wrap


@dataclass
class ParallaxConfig:
    """Parallax mapping configuration."""
    enabled: bool = False
    steps: int = 16
    method: str = "pom"  # pom, steep, simple


@dataclass
class MaterialTierConfig:
    """Complete material configuration for a quality tier."""
    tier: QualityTier
    max_variants: int = 1
    enabled_features: Set[MaterialFeature] = field(default_factory=set)
    texture_config: TextureConfig = field(default_factory=TextureConfig)
    sss_config: SSSConfig = field(default_factory=SSSConfig)
    parallax_config: ParallaxConfig = field(default_factory=ParallaxConfig)
    gpu_time_budget_ms: float = 0.5
    memory_budget_mb: int = 64

    @property
    def supports_advanced_shading(self) -> bool:
        advanced = {
            MaterialFeature.SUBSURFACE_SCATTERING,
            MaterialFeature.CLEAR_COAT,
            MaterialFeature.ANISOTROPY,
            MaterialFeature.SHEEN,
            MaterialFeature.TRANSMISSION,
            MaterialFeature.IRIDESCENCE,
        }
        return bool(self.enabled_features & advanced)

    @property
    def feature_count(self) -> int:
        return len(self.enabled_features)

    @property
    def unlimited_variants(self) -> bool:
        return self.max_variants < 0


def create_low_tier_config() -> MaterialTierConfig:
    """Create Low tier material config: 1 variant, basic PBR."""
    return MaterialTierConfig(
        tier=QualityTier.LOW,
        max_variants=1,
        enabled_features={
            MaterialFeature.BASE_COLOR,
            MaterialFeature.NORMAL_MAPPING,
            MaterialFeature.ROUGHNESS_METALLIC,
        },
        texture_config=TextureConfig(
            max_size=512,
            lod_bias=2.0,
            filter_mode=TextureFilterMode.BILINEAR,
            anisotropic_level=1,
        ),
        sss_config=SSSConfig(enabled=False),
        parallax_config=ParallaxConfig(enabled=False),
        gpu_time_budget_ms=0.5,
        memory_budget_mb=64,
    )


def create_medium_tier_config() -> MaterialTierConfig:
    """Create Medium tier material config: 3 variants, standard PBR."""
    return MaterialTierConfig(
        tier=QualityTier.MEDIUM,
        max_variants=3,
        enabled_features={
            MaterialFeature.BASE_COLOR,
            MaterialFeature.NORMAL_MAPPING,
            MaterialFeature.ROUGHNESS_METALLIC,
            MaterialFeature.AMBIENT_OCCLUSION,
            MaterialFeature.EMISSIVE,
            MaterialFeature.DETAIL_MAPS,
        },
        texture_config=TextureConfig(
            max_size=1024,
            lod_bias=1.0,
            filter_mode=TextureFilterMode.TRILINEAR,
            anisotropic_level=4,
        ),
        sss_config=SSSConfig(enabled=False),
        parallax_config=ParallaxConfig(enabled=False),
        gpu_time_budget_ms=1.0,
        memory_budget_mb=128,
    )


def create_high_tier_config() -> MaterialTierConfig:
    """Create High tier material config: 10 variants, advanced shading."""
    return MaterialTierConfig(
        tier=QualityTier.HIGH,
        max_variants=10,
        enabled_features={
            MaterialFeature.BASE_COLOR,
            MaterialFeature.NORMAL_MAPPING,
            MaterialFeature.ROUGHNESS_METALLIC,
            MaterialFeature.AMBIENT_OCCLUSION,
            MaterialFeature.EMISSIVE,
            MaterialFeature.DETAIL_MAPS,
            MaterialFeature.SUBSURFACE_SCATTERING,
            MaterialFeature.CLEAR_COAT,
            MaterialFeature.ANISOTROPY,
            MaterialFeature.PARALLAX_MAPPING,
        },
        texture_config=TextureConfig(
            max_size=2048,
            lod_bias=0.0,
            filter_mode=TextureFilterMode.TRILINEAR,
            anisotropic_level=8,
        ),
        sss_config=SSSConfig(enabled=True, samples=8, method="separable"),
        parallax_config=ParallaxConfig(enabled=True, steps=16, method="pom"),
        gpu_time_budget_ms=2.0,
        memory_budget_mb=256,
    )


def create_ultra_tier_config() -> MaterialTierConfig:
    """Create Ultra tier material config: unlimited variants, all features."""
    return MaterialTierConfig(
        tier=QualityTier.ULTRA,
        max_variants=-1,  # Unlimited
        enabled_features={
            MaterialFeature.BASE_COLOR,
            MaterialFeature.NORMAL_MAPPING,
            MaterialFeature.ROUGHNESS_METALLIC,
            MaterialFeature.AMBIENT_OCCLUSION,
            MaterialFeature.EMISSIVE,
            MaterialFeature.DETAIL_MAPS,
            MaterialFeature.SUBSURFACE_SCATTERING,
            MaterialFeature.CLEAR_COAT,
            MaterialFeature.ANISOTROPY,
            MaterialFeature.SHEEN,
            MaterialFeature.TRANSMISSION,
            MaterialFeature.IRIDESCENCE,
            MaterialFeature.PARALLAX_MAPPING,
            MaterialFeature.TESSELLATION,
            MaterialFeature.DISPLACEMENT_MAPPING,
        },
        texture_config=TextureConfig(
            max_size=4096,
            lod_bias=-0.5,
            filter_mode=TextureFilterMode.TRILINEAR,
            anisotropic_level=16,
        ),
        sss_config=SSSConfig(enabled=True, samples=16, method="separable"),
        parallax_config=ParallaxConfig(enabled=True, steps=32, method="pom"),
        gpu_time_budget_ms=4.0,
        memory_budget_mb=512,
    )


TIER_CONFIGS: Dict[QualityTier, Callable[[], MaterialTierConfig]] = {
    QualityTier.LOW: create_low_tier_config,
    QualityTier.MEDIUM: create_medium_tier_config,
    QualityTier.HIGH: create_high_tier_config,
    QualityTier.ULTRA: create_ultra_tier_config,
}


@dataclass
class VariantUsageStats:
    """Tracks material variant usage."""
    active_variants: int = 0
    total_materials: int = 0
    variant_breakdown: Dict[str, int] = field(default_factory=dict)


class TierChangeListener:
    """Protocol for tier change notifications."""
    def on_tier_changed(self, old_tier: QualityTier, new_tier: QualityTier, config: MaterialTierConfig) -> None:
        pass


class MaterialTierManager:
    """Manages quality tier integration for the materials subsystem."""

    def __init__(self, initial_tier: QualityTier = QualityTier.MEDIUM):
        self._current_tier = initial_tier
        self._config = TIER_CONFIGS[initial_tier]()
        self._listeners: List[TierChangeListener] = []
        self._feature_overrides: Dict[MaterialFeature, bool] = {}
        self._usage_stats = VariantUsageStats()

    @property
    def current_tier(self) -> QualityTier:
        return self._current_tier

    @property
    def config(self) -> MaterialTierConfig:
        return self._config

    @property
    def usage_stats(self) -> VariantUsageStats:
        return self._usage_stats

    def set_tier(self, tier: QualityTier) -> None:
        """Set the quality tier and update configuration."""
        if tier == self._current_tier:
            return

        old_tier = self._current_tier
        self._current_tier = tier
        self._config = TIER_CONFIGS[tier]()
        self._apply_overrides()

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

    def override_feature(self, feature: MaterialFeature, enabled: bool) -> None:
        """Override a feature setting regardless of tier."""
        self._feature_overrides[feature] = enabled
        self._apply_overrides()

    def clear_overrides(self) -> None:
        """Clear all feature overrides."""
        self._feature_overrides.clear()
        self._config = TIER_CONFIGS[self._current_tier]()

    def is_feature_enabled(self, feature: MaterialFeature) -> bool:
        """Check if a feature is enabled in current config."""
        return feature in self._config.enabled_features

    def get_max_variants(self) -> int:
        """Get maximum allowed material variants."""
        return self._config.max_variants

    def can_add_variant(self) -> bool:
        """Check if another variant can be added."""
        if self._config.unlimited_variants:
            return True
        return self._usage_stats.active_variants < self._config.max_variants

    def register_variant(self, variant_name: str) -> bool:
        """Register a material variant. Returns False if at limit."""
        if not self.can_add_variant():
            return False
        self._usage_stats.active_variants += 1
        self._usage_stats.variant_breakdown[variant_name] = (
            self._usage_stats.variant_breakdown.get(variant_name, 0) + 1
        )
        return True

    def unregister_variant(self, variant_name: str) -> None:
        """Unregister a material variant."""
        if self._usage_stats.active_variants > 0:
            self._usage_stats.active_variants -= 1
        if variant_name in self._usage_stats.variant_breakdown:
            count = self._usage_stats.variant_breakdown[variant_name]
            if count > 1:
                self._usage_stats.variant_breakdown[variant_name] = count - 1
            else:
                del self._usage_stats.variant_breakdown[variant_name]

    def register_material(self) -> None:
        """Register a material instance."""
        self._usage_stats.total_materials += 1

    def unregister_material(self) -> None:
        """Unregister a material instance."""
        if self._usage_stats.total_materials > 0:
            self._usage_stats.total_materials -= 1

    def get_texture_config(self) -> TextureConfig:
        return self._config.texture_config

    def get_sss_config(self) -> SSSConfig:
        return self._config.sss_config

    def get_parallax_config(self) -> ParallaxConfig:
        return self._config.parallax_config

    def get_max_texture_size(self) -> int:
        return self._config.texture_config.max_size

    def get_lod_bias(self) -> float:
        return self._config.texture_config.lod_bias

    def get_anisotropic_level(self) -> int:
        return self._config.texture_config.anisotropic_level

    def get_gpu_budget_ms(self) -> float:
        return self._config.gpu_time_budget_ms

    def get_memory_budget_mb(self) -> int:
        return self._config.memory_budget_mb

    def get_required_features(self, material_type: str) -> Set[MaterialFeature]:
        """Get features required for a material type, filtered by tier."""
        material_requirements: Dict[str, Set[MaterialFeature]] = {
            "standard": {MaterialFeature.BASE_COLOR, MaterialFeature.NORMAL_MAPPING, MaterialFeature.ROUGHNESS_METALLIC},
            "skin": {MaterialFeature.BASE_COLOR, MaterialFeature.NORMAL_MAPPING, MaterialFeature.SUBSURFACE_SCATTERING},
            "car_paint": {MaterialFeature.BASE_COLOR, MaterialFeature.CLEAR_COAT},
            "fabric": {MaterialFeature.BASE_COLOR, MaterialFeature.ANISOTROPY, MaterialFeature.SHEEN},
            "glass": {MaterialFeature.BASE_COLOR, MaterialFeature.TRANSMISSION},
            "hair": {MaterialFeature.BASE_COLOR, MaterialFeature.ANISOTROPY},
        }
        required = material_requirements.get(material_type, set())
        return required & self._config.enabled_features

    def select_fallback_variant(self, requested: str) -> str:
        """Select best available variant based on tier constraints."""
        fallback_chains: Dict[str, List[str]] = {
            "skin": ["skin_sss", "skin_simple", "standard"],
            "car_paint": ["car_paint_clearcoat", "car_paint_basic", "standard"],
            "fabric": ["fabric_sheen", "fabric_basic", "standard"],
            "glass": ["glass_transmission", "glass_simple", "standard"],
            "hair": ["hair_anisotropic", "hair_simple", "standard"],
        }

        chain = fallback_chains.get(requested, [requested, "standard"])
        for variant in chain:
            if self._can_use_variant(variant):
                return variant
        return "standard"

    def _can_use_variant(self, variant: str) -> bool:
        """Check if a variant can be used with current tier."""
        variant_features: Dict[str, Set[MaterialFeature]] = {
            "skin_sss": {MaterialFeature.SUBSURFACE_SCATTERING},
            "car_paint_clearcoat": {MaterialFeature.CLEAR_COAT},
            "fabric_sheen": {MaterialFeature.SHEEN},
            "glass_transmission": {MaterialFeature.TRANSMISSION},
            "hair_anisotropic": {MaterialFeature.ANISOTROPY},
            "standard": set(),
        }
        required = variant_features.get(variant, set())
        return required.issubset(self._config.enabled_features)

    def get_status_dict(self) -> Dict[str, Any]:
        """Get status information as a dictionary."""
        return {
            "tier": self._current_tier.name,
            "max_variants": self._config.max_variants,
            "active_variants": self._usage_stats.active_variants,
            "total_materials": self._usage_stats.total_materials,
            "supports_advanced_shading": self._config.supports_advanced_shading,
            "feature_count": self._config.feature_count,
            "max_texture_size": self._config.texture_config.max_size,
            "anisotropic_level": self._config.texture_config.anisotropic_level,
            "sss_enabled": self._config.sss_config.enabled,
            "parallax_enabled": self._config.parallax_config.enabled,
        }


def get_tier_for_features(required_features: Set[MaterialFeature]) -> QualityTier:
    """Suggest minimum tier that supports the required features."""
    for tier in [QualityTier.LOW, QualityTier.MEDIUM, QualityTier.HIGH, QualityTier.ULTRA]:
        config = TIER_CONFIGS[tier]()
        if required_features.issubset(config.enabled_features):
            return tier
    return QualityTier.ULTRA


def get_tier_for_variant_count(variant_count: int) -> QualityTier:
    """Suggest minimum tier that supports the variant count."""
    if variant_count <= 1:
        return QualityTier.LOW
    elif variant_count <= 3:
        return QualityTier.MEDIUM
    elif variant_count <= 10:
        return QualityTier.HIGH
    else:
        return QualityTier.ULTRA


def estimate_material_memory(
    config: MaterialTierConfig,
    material_count: int,
    avg_textures_per_material: int = 5,
) -> int:
    """Estimate material system memory usage in bytes."""
    memory = 0

    # Texture memory
    tex_size = config.texture_config.max_size
    bytes_per_texture = tex_size * tex_size * 4  # RGBA8
    memory += material_count * avg_textures_per_material * bytes_per_texture

    # Uniform buffers (material parameters)
    bytes_per_material = 256  # Typical UBO size
    memory += material_count * bytes_per_material

    # Shader variants
    bytes_per_variant = 64 * 1024  # 64KB average compiled shader
    if config.unlimited_variants:
        variant_estimate = min(material_count, 50)  # Cap estimate
    else:
        variant_estimate = config.max_variants
    memory += variant_estimate * bytes_per_variant

    return memory


def create_shader_permutation_key(features: Set[MaterialFeature]) -> int:
    """Create a bitmask key for shader permutation lookup."""
    key = 0
    feature_bits = {
        MaterialFeature.BASE_COLOR: 1 << 0,
        MaterialFeature.NORMAL_MAPPING: 1 << 1,
        MaterialFeature.ROUGHNESS_METALLIC: 1 << 2,
        MaterialFeature.AMBIENT_OCCLUSION: 1 << 3,
        MaterialFeature.EMISSIVE: 1 << 4,
        MaterialFeature.DETAIL_MAPS: 1 << 5,
        MaterialFeature.SUBSURFACE_SCATTERING: 1 << 6,
        MaterialFeature.CLEAR_COAT: 1 << 7,
        MaterialFeature.ANISOTROPY: 1 << 8,
        MaterialFeature.SHEEN: 1 << 9,
        MaterialFeature.TRANSMISSION: 1 << 10,
        MaterialFeature.IRIDESCENCE: 1 << 11,
        MaterialFeature.PARALLAX_MAPPING: 1 << 12,
        MaterialFeature.TESSELLATION: 1 << 13,
        MaterialFeature.DISPLACEMENT_MAPPING: 1 << 14,
    }
    for feature in features:
        key |= feature_bits.get(feature, 0)
    return key
