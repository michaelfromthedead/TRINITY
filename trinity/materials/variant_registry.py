"""Quality-driven variant compilation registry for material shaders.

This module implements MaterialMeta-based triple compilation for quality tiers,
enabling runtime selection of the appropriate shader variant based on device
capabilities and quality settings.

The registry stores compiled variants keyed by:
1. Material name
2. Variant key (combination of domain + blend + quality)

Runtime selection is O(1) dictionary lookup.

Task: T-MAT-5.1 Quality-Driven Variant Compilation
Gap: S3-G9 (HIGH)
Dependency: T-MAT-2.4 (quality tier variants in quality.py)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple, TYPE_CHECKING

from trinity.materials.variants import (
    MaterialDomain,
    BlendMode,
    QualityTier,
    VariantConfig,
)

if TYPE_CHECKING:
    from trinity.materials.compiler import MaterialCompiler


@dataclass(frozen=True, slots=True)
class CompiledVariant:
    """A compiled shader variant with metadata.

    Immutable dataclass representing a single compiled shader variant.
    The frozen=True enables use as a dictionary key if needed.

    Attributes:
        wgsl: The compiled WGSL shader source.
        variant_key: Unique string key identifying this variant.
        quality: Quality tier (LOW, MEDIUM, HIGH).
        domain: Material domain (SURFACE, VOLUME, etc.).
        blend: Blend mode (OPAQUE, MASKED, etc.).
        hash_key: Integer hash for fast lookup.
    """

    wgsl: str
    variant_key: str
    quality: QualityTier
    domain: MaterialDomain
    blend: BlendMode
    hash_key: int

    @classmethod
    def from_config(cls, wgsl: str, config: VariantConfig) -> "CompiledVariant":
        """Create a CompiledVariant from a VariantConfig.

        Args:
            wgsl: Compiled WGSL shader source.
            config: The VariantConfig used for compilation.

        Returns:
            New CompiledVariant instance.
        """
        variant_key = (
            f"{config.domain.value}_{config.blend.value}_{config.quality.value}"
        )
        return cls(
            wgsl=wgsl,
            variant_key=variant_key,
            quality=config.quality,
            domain=config.domain,
            blend=config.blend,
            hash_key=config.get_variant_key(),
        )


class MaterialVariantRegistry:
    """Registry for compiled material variants.

    This registry stores pre-compiled shader variants for each registered
    material, enabling O(1) runtime selection based on quality tier and
    other variant dimensions.

    The registry supports:
    - Registering materials with all 75 variants (5 domains x 5 blends x 3 qualities)
    - Fast variant selection by quality tier
    - Optional filtering by domain and blend mode
    - Iteration over registered materials and their variants

    Example::

        registry = MaterialVariantRegistry()

        # Register a material (compiles all 75 variants)
        compiler = MaterialCompiler()
        registry.register_material("gold", compiler, GoldMaterial)

        # Select variant at runtime based on quality setting
        variant = registry.select_variant("gold", QualityTier.HIGH)
        if variant:
            gpu.compile_shader(variant.wgsl)

    Thread Safety:
        The registry is NOT thread-safe. External synchronization is required
        for concurrent access during registration. Selection is read-only and
        safe for concurrent reads.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        # material_name -> (variant_key -> CompiledVariant)
        self._variants: Dict[str, Dict[str, CompiledVariant]] = {}
        # material_name -> class reference (for re-compilation)
        self._material_classes: Dict[str, type] = {}

    def register_material(
        self,
        name: str,
        compiler: "MaterialCompiler",
        material_class: type,
        domains: Optional[List[MaterialDomain]] = None,
        blends: Optional[List[BlendMode]] = None,
        qualities: Optional[List[QualityTier]] = None,
    ) -> int:
        """Compile and register all variants for a material.

        Args:
            name: Unique material name for registry lookup.
            compiler: MaterialCompiler instance for compilation.
            material_class: Material class to compile.
            domains: Optional list of domains to compile. If None, all domains.
            blends: Optional list of blend modes to compile. If None, all blends.
            qualities: Optional list of quality tiers to compile. If None, all tiers.

        Returns:
            Number of variants compiled and registered.

        Example::

            # Register all 75 variants
            count = registry.register_material("gold", compiler, GoldMaterial)
            # count == 75

            # Register only quality variants for SURFACE domain
            count = registry.register_material(
                "gold_surface",
                compiler,
                GoldMaterial,
                domains=[MaterialDomain.SURFACE],
                blends=[BlendMode.OPAQUE],
            )
            # count == 3 (one per quality tier)
        """
        # Default to all variants if not specified
        if domains is None:
            domains = list(MaterialDomain)
        if blends is None:
            blends = list(BlendMode)
        if qualities is None:
            qualities = list(QualityTier)

        # Store material class reference
        self._material_classes[name] = material_class

        # Initialize variant dict for this material
        if name not in self._variants:
            self._variants[name] = {}

        count = 0
        for domain in domains:
            for blend in blends:
                for quality in qualities:
                    config = VariantConfig(
                        domain=domain,
                        blend=blend,
                        quality=quality,
                    )

                    # Compile with this configuration
                    wgsl = compiler.compile_with_variants(material_class, config)

                    # Create compiled variant
                    variant = CompiledVariant.from_config(wgsl, config)

                    # Store by variant key
                    self._variants[name][variant.variant_key] = variant
                    count += 1

        return count

    def register_precompiled(
        self,
        name: str,
        variants: List[Tuple[VariantConfig, str]],
    ) -> int:
        """Register pre-compiled variants without re-compilation.

        Useful for loading cached variants from disk.

        Args:
            name: Unique material name.
            variants: List of (config, wgsl) tuples.

        Returns:
            Number of variants registered.
        """
        if name not in self._variants:
            self._variants[name] = {}

        count = 0
        for config, wgsl in variants:
            variant = CompiledVariant.from_config(wgsl, config)
            self._variants[name][variant.variant_key] = variant
            count += 1

        return count

    def select_variant(
        self,
        material_name: str,
        quality: QualityTier,
        domain: Optional[MaterialDomain] = None,
        blend: Optional[BlendMode] = None,
    ) -> Optional[CompiledVariant]:
        """Select the best variant for given quality tier.

        Selection priority:
        1. Exact match (quality + domain + blend)
        2. Quality + domain (any blend)
        3. Quality only (any domain/blend)
        4. None if no matching variant exists

        Args:
            material_name: Name of the registered material.
            quality: Required quality tier.
            domain: Optional domain filter. Defaults to SURFACE if unspecified
                    when doing exact match.
            blend: Optional blend mode filter. Defaults to OPAQUE if unspecified
                   when doing exact match.

        Returns:
            CompiledVariant if found, None otherwise.

        Example::

            # Get HIGH quality SURFACE OPAQUE variant
            variant = registry.select_variant("gold", QualityTier.HIGH)

            # Get specific variant
            variant = registry.select_variant(
                "gold",
                QualityTier.MEDIUM,
                domain=MaterialDomain.VOLUME,
                blend=BlendMode.TRANSLUCENT,
            )
        """
        if material_name not in self._variants:
            return None

        variants = self._variants[material_name]

        # Default to SURFACE + OPAQUE for common case
        target_domain = domain if domain is not None else MaterialDomain.SURFACE
        target_blend = blend if blend is not None else BlendMode.OPAQUE

        # Try exact match first
        exact_key = f"{target_domain.value}_{target_blend.value}_{quality.value}"
        if exact_key in variants:
            return variants[exact_key]

        # If domain was specified but blend wasn't, find any blend
        if domain is not None and blend is None:
            for v in variants.values():
                if v.quality == quality and v.domain == domain:
                    return v

        # If only quality was specified, find any matching
        if domain is None and blend is None:
            for v in variants.values():
                if v.quality == quality:
                    return v

        # Fallback: any variant with matching quality
        for v in variants.values():
            if v.quality == quality:
                return v

        return None

    def select_variant_exact(
        self,
        material_name: str,
        quality: QualityTier,
        domain: MaterialDomain,
        blend: BlendMode,
    ) -> Optional[CompiledVariant]:
        """Select a variant with exact match only.

        Unlike select_variant(), this method requires an exact match
        for all three dimensions and returns None if not found.

        Args:
            material_name: Name of the registered material.
            quality: Required quality tier.
            domain: Required domain.
            blend: Required blend mode.

        Returns:
            CompiledVariant if exact match found, None otherwise.
        """
        if material_name not in self._variants:
            return None

        exact_key = f"{domain.value}_{blend.value}_{quality.value}"
        return self._variants[material_name].get(exact_key)

    def get_variant_count(self, material_name: str) -> int:
        """Get number of variants for a material.

        Args:
            material_name: Name of the registered material.

        Returns:
            Number of variants, or 0 if material not registered.
        """
        return len(self._variants.get(material_name, {}))

    def get_all_variants(self, material_name: str) -> List[CompiledVariant]:
        """Get all variants for a material.

        Args:
            material_name: Name of the registered material.

        Returns:
            List of all CompiledVariant instances, empty if not registered.
        """
        if material_name not in self._variants:
            return []
        return list(self._variants[material_name].values())

    def get_quality_variants(
        self,
        material_name: str,
        quality: QualityTier,
    ) -> List[CompiledVariant]:
        """Get all variants at a specific quality tier.

        Args:
            material_name: Name of the registered material.
            quality: Quality tier to filter by.

        Returns:
            List of variants matching the quality tier.
        """
        if material_name not in self._variants:
            return []
        return [
            v for v in self._variants[material_name].values()
            if v.quality == quality
        ]

    def has_material(self, material_name: str) -> bool:
        """Check if a material is registered.

        Args:
            material_name: Name to check.

        Returns:
            True if material is registered with at least one variant.
        """
        return material_name in self._variants and len(self._variants[material_name]) > 0

    def list_materials(self) -> List[str]:
        """Get list of all registered material names.

        Returns:
            List of material names.
        """
        return list(self._variants.keys())

    def unregister_material(self, material_name: str) -> bool:
        """Remove a material and all its variants from the registry.

        Args:
            material_name: Name of the material to remove.

        Returns:
            True if material was removed, False if not found.
        """
        if material_name in self._variants:
            del self._variants[material_name]
            self._material_classes.pop(material_name, None)
            return True
        return False

    def clear(self) -> None:
        """Remove all registered materials."""
        self._variants.clear()
        self._material_classes.clear()

    def iter_variants(
        self,
        material_name: str,
    ) -> Iterator[CompiledVariant]:
        """Iterate over all variants for a material.

        Args:
            material_name: Name of the registered material.

        Yields:
            CompiledVariant instances.
        """
        if material_name in self._variants:
            yield from self._variants[material_name].values()

    def get_material_class(self, material_name: str) -> Optional[type]:
        """Get the material class for a registered material.

        Useful for re-compilation or inspection.

        Args:
            material_name: Name of the registered material.

        Returns:
            Material class if registered, None otherwise.
        """
        return self._material_classes.get(material_name)


def select_material_variant(
    registry: MaterialVariantRegistry,
    material: str,
    quality: QualityTier,
) -> Optional[str]:
    """Select material variant WGSL for quality tier.

    Convenience function for simple variant selection.

    Args:
        registry: MaterialVariantRegistry instance.
        material: Material name.
        quality: Quality tier.

    Returns:
        WGSL shader source if found, None otherwise.
    """
    variant = registry.select_variant(material, quality)
    return variant.wgsl if variant else None


def create_quality_optimized_registry(
    compiler: "MaterialCompiler",
    materials: Dict[str, type],
    domain: MaterialDomain = MaterialDomain.SURFACE,
    blend: BlendMode = BlendMode.OPAQUE,
) -> MaterialVariantRegistry:
    """Create a registry optimized for quality-only switching.

    Registers each material with only the 3 quality tier variants
    for a specific domain/blend combination. Useful when only
    quality switching is needed at runtime.

    Args:
        compiler: MaterialCompiler instance.
        materials: Dict of material_name -> material_class.
        domain: Domain to compile for (default: SURFACE).
        blend: Blend mode to compile for (default: OPAQUE).

    Returns:
        MaterialVariantRegistry with 3 variants per material.

    Example::

        materials = {
            "gold": GoldMaterial,
            "silver": SilverMaterial,
        }
        registry = create_quality_optimized_registry(compiler, materials)
        # 2 materials x 3 qualities = 6 variants total
    """
    registry = MaterialVariantRegistry()

    for name, material_class in materials.items():
        registry.register_material(
            name,
            compiler,
            material_class,
            domains=[domain],
            blends=[blend],
            qualities=list(QualityTier),
        )

    return registry


__all__ = [
    "CompiledVariant",
    "MaterialVariantRegistry",
    "select_material_variant",
    "create_quality_optimized_registry",
]
