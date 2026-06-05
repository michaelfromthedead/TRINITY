"""Shader variant pruning based on quality tier (T-CC-0.7).

Prunes shader permutations to only compile variants needed for the selected
quality tier, reducing shader compilation time and memory usage.

Example:
    # Full permutation set (550+ variants)
    permutation = ShaderPermutation(name="PBR")
    permutation.add_feature("normal_map")
    permutation.add_feature("parallax")
    permutation.add_feature("tessellation")
    permutation.add_feature("ray_traced_shadows")

    # Pruned to LOW tier (e.g., 50 variants)
    pruner = ShaderVariantPruner()
    valid_keys = pruner.get_valid_keys_for_tier(permutation, QualityTier.LOW)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, FrozenSet, Set

from trinity.types import QualityTier

from .capabilities import QualityCapabilitiesRegistry, QualityCapabilities, TierFeatureSet

if TYPE_CHECKING:
    from engine.rendering.materials.shader_compiler import (
        PermutationKey,
        ShaderPermutation,
    )

__all__ = [
    "ShaderVariantPruner",
    "VariantPruningConfig",
    "VariantPruningResult",
    "FeatureMapping",
]


@dataclass(slots=True)
class FeatureMapping:
    """Maps shader features to subsystem capabilities.

    Shader features (like "HAS_NORMAL_MAP") map to capability features
    (like "normal_mapping") which are tier-gated.
    """

    shader_feature: str
    capability_feature: str
    subsystem: str
    min_tier: QualityTier = QualityTier.LOW

    def is_available_at_tier(self, tier: QualityTier) -> bool:
        """Check if this feature is available at the given tier."""
        return tier.value >= self.min_tier.value


@dataclass(slots=True)
class VariantPruningConfig:
    """Configuration for shader variant pruning."""

    max_variants_per_shader: int = 128
    max_total_variants: int = 1024
    always_include_base_variant: bool = True
    include_debug_variants: bool = False
    compile_next_tier_up: bool = False

    @classmethod
    def for_tier(cls, tier: QualityTier) -> VariantPruningConfig:
        """Create config appropriate for a quality tier."""
        configs = {
            QualityTier.LOW: cls(
                max_variants_per_shader=32,
                max_total_variants=256,
                include_debug_variants=False,
            ),
            QualityTier.MEDIUM: cls(
                max_variants_per_shader=64,
                max_total_variants=512,
                include_debug_variants=False,
            ),
            QualityTier.HIGH: cls(
                max_variants_per_shader=128,
                max_total_variants=1024,
                include_debug_variants=False,
            ),
            QualityTier.ULTRA: cls(
                max_variants_per_shader=256,
                max_total_variants=2048,
                include_debug_variants=True,
            ),
        }
        return configs.get(tier, cls())


@dataclass(slots=True)
class VariantPruningResult:
    """Result of variant pruning operation."""

    original_count: int
    pruned_count: int
    excluded_features: set[str]
    included_features: set[str]

    @property
    def reduction_percent(self) -> float:
        """Percentage of variants pruned."""
        if self.original_count == 0:
            return 0.0
        return 100.0 * (1.0 - self.pruned_count / self.original_count)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "original_count": self.original_count,
            "pruned_count": self.pruned_count,
            "reduction_percent": round(self.reduction_percent, 1),
            "excluded_features": list(self.excluded_features),
            "included_features": list(self.included_features),
        }


# Standard feature mappings for common shader features
STANDARD_FEATURE_MAPPINGS: list[FeatureMapping] = [
    # Material features
    FeatureMapping("normal_map", "normal_mapping", "materials", QualityTier.LOW),
    FeatureMapping("parallax", "parallax_mapping", "materials", QualityTier.MEDIUM),
    FeatureMapping("tessellation", "tessellation", "materials", QualityTier.HIGH),
    FeatureMapping("displacement", "displacement_mapping", "materials", QualityTier.HIGH),
    FeatureMapping("subsurface", "subsurface_scattering", "materials", QualityTier.HIGH),
    FeatureMapping("clear_coat", "clear_coat", "materials", QualityTier.MEDIUM),
    FeatureMapping("anisotropy", "anisotropic_brdf", "materials", QualityTier.HIGH),

    # Lighting features
    FeatureMapping("clustered_lighting", "clustered_forward", "lighting", QualityTier.MEDIUM),
    FeatureMapping("area_lights", "area_lights", "lighting", QualityTier.HIGH),
    FeatureMapping("ies_profiles", "ies_profiles", "lighting", QualityTier.HIGH),
    FeatureMapping("volumetric_lighting", "volumetric", "lighting", QualityTier.HIGH),

    # Shadow features
    FeatureMapping("pcf_shadows", "pcf", "shadows", QualityTier.LOW),
    FeatureMapping("vsm_shadows", "vsm", "shadows", QualityTier.MEDIUM),
    FeatureMapping("pcss_shadows", "pcss", "shadows", QualityTier.HIGH),
    FeatureMapping("ray_traced_shadows", "ray_traced", "shadows", QualityTier.ULTRA),
    FeatureMapping("contact_shadows", "contact_shadows", "shadows", QualityTier.HIGH),

    # GI features
    FeatureMapping("ssao", "ssao", "gi", QualityTier.LOW),
    FeatureMapping("gtao", "gtao", "gi", QualityTier.MEDIUM),
    FeatureMapping("ddgi", "ddgi", "gi", QualityTier.HIGH),
    FeatureMapping("rtgi", "rtgi", "gi", QualityTier.ULTRA),

    # Reflection features
    FeatureMapping("ssr", "ssr", "reflections", QualityTier.MEDIUM),
    FeatureMapping("probe_reflections", "probe", "reflections", QualityTier.LOW),
    FeatureMapping("planar_reflections", "planar", "reflections", QualityTier.HIGH),
    FeatureMapping("rt_reflections", "rt_reflections", "reflections", QualityTier.ULTRA),

    # Post-process features
    FeatureMapping("bloom", "bloom", "postprocess", QualityTier.LOW),
    FeatureMapping("dof", "dof", "postprocess", QualityTier.MEDIUM),
    FeatureMapping("motion_blur", "motion_blur", "postprocess", QualityTier.MEDIUM),
    FeatureMapping("taa", "taa", "postprocess", QualityTier.MEDIUM),
    FeatureMapping("dlss", "dlss", "postprocess", QualityTier.ULTRA),
    FeatureMapping("fsr", "fsr", "postprocess", QualityTier.HIGH),

    # Atmosphere features
    FeatureMapping("volumetric_fog", "volumetric_fog", "atmosphere", QualityTier.HIGH),
    FeatureMapping("aerial_perspective", "aerial_perspective", "atmosphere", QualityTier.HIGH),
    FeatureMapping("god_rays", "god_rays", "atmosphere", QualityTier.MEDIUM),

    # Particle features
    FeatureMapping("gpu_particles", "gpu_simulation", "particles", QualityTier.MEDIUM),
    FeatureMapping("soft_particles", "soft_blending", "particles", QualityTier.LOW),
    FeatureMapping("particle_shadows", "particle_shadows", "particles", QualityTier.HIGH),
    FeatureMapping("particle_lighting", "particle_lighting", "particles", QualityTier.MEDIUM),

    # Ray tracing features
    FeatureMapping("rt_acceleration", "hw_rt", "raytracing", QualityTier.ULTRA),
    FeatureMapping("rt_ao", "rt_ao", "raytracing", QualityTier.ULTRA),
]


class ShaderVariantPruner:
    """
    Prunes shader variants based on quality tier.

    Filters permutation keys to only include features that are:
    1. Available at the current quality tier
    2. Supported by hardware capabilities
    3. Within variant count budget
    """

    __slots__ = ("_registry", "_config", "_feature_mappings", "_tier")

    def __init__(
        self,
        tier: QualityTier = QualityTier.HIGH,
        config: VariantPruningConfig | None = None,
        registry: QualityCapabilitiesRegistry | None = None,
    ):
        self._tier = tier
        self._config = config or VariantPruningConfig.for_tier(tier)
        self._registry = registry or QualityCapabilitiesRegistry()
        self._feature_mappings: dict[str, FeatureMapping] = {
            m.shader_feature: m for m in STANDARD_FEATURE_MAPPINGS
        }

    @property
    def tier(self) -> QualityTier:
        """Get current quality tier."""
        return self._tier

    @tier.setter
    def tier(self, value: QualityTier) -> None:
        """Set quality tier and update config."""
        self._tier = value
        self._config = VariantPruningConfig.for_tier(value)

    @property
    def config(self) -> VariantPruningConfig:
        """Get pruning configuration."""
        return self._config

    def register_mapping(self, mapping: FeatureMapping) -> None:
        """Register a custom feature mapping."""
        self._feature_mappings[mapping.shader_feature] = mapping

    def register_mappings(self, mappings: list[FeatureMapping]) -> None:
        """Register multiple custom feature mappings."""
        for m in mappings:
            self._feature_mappings[m.shader_feature] = m

    def get_available_features(self) -> set[str]:
        """Get all shader features available at current tier."""
        available = set()

        for shader_feature, mapping in self._feature_mappings.items():
            if mapping.is_available_at_tier(self._tier):
                # Check subsystem capabilities if registry has the subsystem
                caps = self._registry.get(mapping.subsystem)
                if caps is not None:
                    tier_features = caps.tier_features(self._tier)
                    if mapping.capability_feature in tier_features.enabled_features:
                        available.add(shader_feature)
                else:
                    # No subsystem registered, use min_tier check only
                    available.add(shader_feature)

        return available

    def get_excluded_features(self) -> set[str]:
        """Get shader features excluded at current tier."""
        all_features = set(self._feature_mappings.keys())
        available = self.get_available_features()
        return all_features - available

    def is_feature_available(self, shader_feature: str) -> bool:
        """Check if a shader feature is available at current tier."""
        mapping = self._feature_mappings.get(shader_feature)
        if mapping is None:
            # Unknown feature, allow it
            return True
        return mapping.is_available_at_tier(self._tier)

    def prune_permutation_key(
        self,
        key: "PermutationKey",
    ) -> "PermutationKey":
        """Prune features from a permutation key based on tier.

        Args:
            key: Original permutation key

        Returns:
            New key with unavailable features removed
        """
        from engine.rendering.materials.shader_compiler import PermutationKey

        available = self.get_available_features()

        # Keep features that are either:
        # 1. Available at this tier
        # 2. Not in our mapping (unknown features are allowed)
        pruned_features = frozenset(
            f for f in key.features
            if f in available or f not in self._feature_mappings
        )

        return PermutationKey(pruned_features)

    def get_valid_keys_for_tier(
        self,
        permutation: "ShaderPermutation",
    ) -> list["PermutationKey"]:
        """Get valid permutation keys filtered by quality tier.

        Args:
            permutation: Shader permutation configuration

        Returns:
            List of valid keys for the current tier
        """
        all_keys = permutation.get_valid_keys()
        available = self.get_available_features()

        pruned_keys = []
        seen_keys: set[FrozenSet[str]] = set()

        for key in all_keys:
            # Prune unavailable features from each key
            pruned = self.prune_permutation_key(key)

            # Skip duplicates (different original keys may prune to same result)
            if pruned.features in seen_keys:
                continue
            seen_keys.add(pruned.features)

            # Validate pruned key is still valid
            is_valid, _ = permutation.validate_key(pruned)
            if is_valid:
                pruned_keys.append(pruned)
            elif self._config.always_include_base_variant:
                # Try base variant with only required features
                from engine.rendering.materials.shader_compiler import PermutationKey
                base_key = PermutationKey(frozenset(permutation.required))
                if base_key.features not in seen_keys:
                    seen_keys.add(base_key.features)
                    pruned_keys.append(base_key)

        # Enforce variant count limit
        if len(pruned_keys) > self._config.max_variants_per_shader:
            pruned_keys = pruned_keys[:self._config.max_variants_per_shader]

        return pruned_keys

    def prune_permutation(
        self,
        permutation: "ShaderPermutation",
    ) -> VariantPruningResult:
        """Prune a shader permutation and return statistics.

        Args:
            permutation: Shader permutation configuration

        Returns:
            Pruning result with statistics
        """
        original_keys = permutation.get_valid_keys()
        pruned_keys = self.get_valid_keys_for_tier(permutation)

        return VariantPruningResult(
            original_count=len(original_keys),
            pruned_count=len(pruned_keys),
            excluded_features=self.get_excluded_features() & permutation.features,
            included_features=self.get_available_features() & permutation.features,
        )

    def estimate_variant_reduction(
        self,
        permutations: list["ShaderPermutation"],
    ) -> dict[str, VariantPruningResult]:
        """Estimate variant reduction for multiple permutations.

        Args:
            permutations: List of shader permutations

        Returns:
            Dictionary mapping permutation name to pruning result
        """
        results = {}
        for perm in permutations:
            results[perm.name] = self.prune_permutation(perm)
        return results

    def total_variant_stats(
        self,
        permutations: list["ShaderPermutation"],
    ) -> dict[str, Any]:
        """Get total variant statistics across all permutations.

        Args:
            permutations: List of shader permutations

        Returns:
            Statistics dictionary
        """
        results = self.estimate_variant_reduction(permutations)

        total_original = sum(r.original_count for r in results.values())
        total_pruned = sum(r.pruned_count for r in results.values())

        all_excluded = set()
        all_included = set()
        for r in results.values():
            all_excluded |= r.excluded_features
            all_included |= r.included_features

        return {
            "tier": self._tier.name,
            "shader_count": len(permutations),
            "total_original_variants": total_original,
            "total_pruned_variants": total_pruned,
            "total_reduction_percent": round(
                100.0 * (1.0 - total_pruned / total_original) if total_original > 0 else 0,
                1
            ),
            "excluded_features": list(all_excluded),
            "included_features": list(all_included),
            "per_shader_results": {
                name: result.to_dict() for name, result in results.items()
            },
        }


def create_pruner_for_tier(
    tier: QualityTier,
    registry: QualityCapabilitiesRegistry | None = None,
) -> ShaderVariantPruner:
    """Factory function to create a pruner for a specific tier."""
    return ShaderVariantPruner(
        tier=tier,
        config=VariantPruningConfig.for_tier(tier),
        registry=registry,
    )
