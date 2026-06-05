"""Quality tier management and capability scoring for Trinity rendering."""

from .capabilities import (
    BaseQualityCapabilities,
    FallbackChain,
    QualityCapabilities,
    QualityCapabilitiesRegistry,
    TierBudget,
    TierFeatureSet,
    TierResolution,
)
from .capability_scorer import (
    AdapterInfo,
    CapabilityScorer,
    FeatureFlags,
    GPULimits,
)
from .quality_manager import QualityManager

__all__ = [
    "AdapterInfo",
    "BaseQualityCapabilities",
    "CapabilityScorer",
    "FallbackChain",
    "FeatureFlags",
    "GPULimits",
    "QualityCapabilities",
    "QualityCapabilitiesRegistry",
    "QualityManager",
    "TierBudget",
    "TierFeatureSet",
    "TierResolution",
]
