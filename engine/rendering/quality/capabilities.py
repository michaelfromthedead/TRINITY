"""
Quality capabilities trait for subsystems (T-CC-0.4).

Defines the interface that rendering subsystems implement to declare
their tier-specific features, budgets, resolutions, and fallback chains.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Optional, Protocol, TypeVar, runtime_checkable

from trinity.types import QualityTier


@dataclass
class TierFeatureSet:
    """Feature set enabled at a specific quality tier."""

    tier: QualityTier
    enabled_features: frozenset[str] = field(default_factory=frozenset)
    disabled_features: frozenset[str] = field(default_factory=frozenset)
    parameters: dict[str, Any] = field(default_factory=dict)

    def has_feature(self, feature: str) -> bool:
        """Check if feature is enabled at this tier."""
        return feature in self.enabled_features

    def get_param(self, name: str, default: Any = None) -> Any:
        """Get tier-specific parameter value."""
        return self.parameters.get(name, default)


@dataclass
class TierBudget:
    """Resource budget for a quality tier."""

    tier: QualityTier
    gpu_time_ms: float = 2.0  # Max GPU time in milliseconds
    memory_mb: int = 128  # Max GPU memory in megabytes
    draw_calls: int = 100  # Max draw calls
    triangles: int = 100000  # Max triangles

    def exceeds_budget(
        self,
        gpu_time_ms: float = 0,
        memory_mb: int = 0,
        draw_calls: int = 0,
        triangles: int = 0,
    ) -> bool:
        """Check if usage exceeds budget."""
        return (
            gpu_time_ms > self.gpu_time_ms
            or memory_mb > self.memory_mb
            or draw_calls > self.draw_calls
            or triangles > self.triangles
        )


@dataclass
class TierResolution:
    """Resolution settings for a quality tier."""

    tier: QualityTier
    render_scale: float = 1.0  # 0.5 = half resolution, 1.0 = native
    shadow_resolution: int = 1024
    reflection_resolution: int = 512
    gi_resolution: int = 256
    texture_max_size: int = 2048

    def scaled_resolution(self, width: int, height: int) -> tuple[int, int]:
        """Get scaled render resolution."""
        return (
            max(1, int(width * self.render_scale)),
            max(1, int(height * self.render_scale)),
        )


@dataclass
class FallbackChain:
    """Fallback chain for graceful degradation."""

    primary: str
    fallbacks: tuple[str, ...] = ()

    def get_fallback(self, tier: QualityTier) -> str:
        """Get appropriate implementation for tier."""
        if tier == QualityTier.ULTRA or not self.fallbacks:
            return self.primary
        # Map tier to fallback index: HIGH→0, MEDIUM→1, LOW→2
        fallback_idx = QualityTier.ULTRA.value - tier.value - 1
        # Clamp to available fallbacks
        fallback_idx = min(fallback_idx, len(self.fallbacks) - 1)
        return self.fallbacks[fallback_idx]


@runtime_checkable
class QualityCapabilities(Protocol):
    """
    Protocol for subsystems that support quality tier configuration.

    Subsystems implement this to declare their tier-specific behavior,
    enabling the QualityManager to configure rendering appropriately.
    """

    @property
    def subsystem_name(self) -> str:
        """Unique identifier for this subsystem (e.g., 'lighting', 'shadows')."""
        ...

    def get_features(self, tier: QualityTier) -> TierFeatureSet:
        """Get feature set enabled at the given tier."""
        ...

    def get_budget(self, tier: QualityTier) -> TierBudget:
        """Get resource budget for the given tier."""
        ...

    def get_resolution(self, tier: QualityTier) -> TierResolution:
        """Get resolution settings for the given tier."""
        ...

    def get_fallback_chain(self, feature: str) -> Optional[FallbackChain]:
        """Get fallback chain for a specific feature."""
        ...


class BaseQualityCapabilities(ABC):
    """
    Base implementation of QualityCapabilities with common functionality.

    Subsystems can extend this to simplify implementation.
    """

    _features: dict[QualityTier, TierFeatureSet]
    _budgets: dict[QualityTier, TierBudget]
    _resolutions: dict[QualityTier, TierResolution]
    _fallbacks: dict[str, FallbackChain]

    def __init__(self):
        self._features = {}
        self._budgets = {}
        self._resolutions = {}
        self._fallbacks = {}
        self._init_tier_configs()

    @property
    @abstractmethod
    def subsystem_name(self) -> str:
        """Unique identifier for this subsystem."""
        ...

    @abstractmethod
    def _init_tier_configs(self) -> None:
        """Initialize tier configurations. Override to set up features/budgets."""
        ...

    def get_features(self, tier: QualityTier) -> TierFeatureSet:
        """Get feature set for tier, falling back to lower tiers if not defined."""
        if tier in self._features:
            return self._features[tier]
        # Fall back to lower tier
        for t in reversed(list(QualityTier)):
            if t.value <= tier.value and t in self._features:
                return self._features[t]
        return TierFeatureSet(tier=tier)

    def get_budget(self, tier: QualityTier) -> TierBudget:
        """Get budget for tier, falling back to lower tiers if not defined."""
        if tier in self._budgets:
            return self._budgets[tier]
        for t in reversed(list(QualityTier)):
            if t.value <= tier.value and t in self._budgets:
                return self._budgets[t]
        return TierBudget(tier=tier)

    def get_resolution(self, tier: QualityTier) -> TierResolution:
        """Get resolution for tier, falling back to lower tiers if not defined."""
        if tier in self._resolutions:
            return self._resolutions[tier]
        for t in reversed(list(QualityTier)):
            if t.value <= tier.value and t in self._resolutions:
                return self._resolutions[t]
        return TierResolution(tier=tier)

    def get_fallback_chain(self, feature: str) -> Optional[FallbackChain]:
        """Get fallback chain for a feature."""
        return self._fallbacks.get(feature)

    def _set_features(self, tier: QualityTier, features: TierFeatureSet) -> None:
        """Set feature set for a tier."""
        self._features[tier] = features

    def _set_budget(self, tier: QualityTier, budget: TierBudget) -> None:
        """Set budget for a tier."""
        self._budgets[tier] = budget

    def _set_resolution(self, tier: QualityTier, resolution: TierResolution) -> None:
        """Set resolution for a tier."""
        self._resolutions[tier] = resolution

    def _set_fallback(self, feature: str, chain: FallbackChain) -> None:
        """Set fallback chain for a feature."""
        self._fallbacks[feature] = chain

    def supports_tier(self, tier: QualityTier) -> bool:
        """Check if this subsystem supports the given tier."""
        return tier in self._features

    def list_features(self, tier: QualityTier) -> list[str]:
        """List all enabled features at the given tier."""
        features = self.get_features(tier)
        return sorted(features.enabled_features)

    def compare_tiers(
        self, low_tier: QualityTier, high_tier: QualityTier
    ) -> dict[str, Any]:
        """Compare features between two tiers."""
        low_features = self.get_features(low_tier)
        high_features = self.get_features(high_tier)

        return {
            "added": high_features.enabled_features - low_features.enabled_features,
            "removed": low_features.enabled_features - high_features.enabled_features,
            "low_tier": low_tier,
            "high_tier": high_tier,
        }


# Type variable for generic subsystem capabilities
T = TypeVar("T", bound=QualityCapabilities)


class QualityCapabilitiesRegistry:
    """
    Registry for subsystem quality capabilities.

    Allows the QualityManager to discover and configure all subsystems.
    """

    _instance: Optional["QualityCapabilitiesRegistry"] = None
    _subsystems: dict[str, QualityCapabilities]

    def __new__(cls) -> "QualityCapabilitiesRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._subsystems = {}
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the registry (for testing)."""
        cls._instance = None

    def register(self, capabilities: QualityCapabilities) -> None:
        """Register a subsystem's quality capabilities."""
        self._subsystems[capabilities.subsystem_name] = capabilities

    def unregister(self, name: str) -> None:
        """Unregister a subsystem."""
        self._subsystems.pop(name, None)

    def get(self, name: str) -> Optional[QualityCapabilities]:
        """Get capabilities for a named subsystem."""
        return self._subsystems.get(name)

    def list_subsystems(self) -> list[str]:
        """List all registered subsystem names."""
        return sorted(self._subsystems.keys())

    def get_all(self) -> dict[str, QualityCapabilities]:
        """Get all registered capabilities."""
        return dict(self._subsystems)

    def get_features_for_tier(
        self, tier: QualityTier
    ) -> dict[str, TierFeatureSet]:
        """Get all subsystem features for a tier."""
        return {
            name: caps.get_features(tier)
            for name, caps in self._subsystems.items()
        }

    def get_total_budget(self, tier: QualityTier) -> TierBudget:
        """Get combined budget across all subsystems."""
        total = TierBudget(
            tier=tier,
            gpu_time_ms=0,
            memory_mb=0,
            draw_calls=0,
            triangles=0,
        )
        for caps in self._subsystems.values():
            budget = caps.get_budget(tier)
            total.gpu_time_ms += budget.gpu_time_ms
            total.memory_mb += budget.memory_mb
            total.draw_calls += budget.draw_calls
            total.triangles += budget.triangles
        return total
