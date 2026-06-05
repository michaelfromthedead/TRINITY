"""Fallback selection logic for startup tier assignment (T-CC-0.13).

Determines the correct quality tier and fallback chain based on
GPU capabilities detected via wgpu adapter query at engine startup.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from trinity.types import QualityTier

from .capability_scorer import AdapterInfo, CapabilityScorer, GPUBackend
from .gles_capabilities import GLESCapabilities, GLESVersion, GLESWorkaroundRegistry
from .quality_manager import QualityManager, QualityManagerConfig

if TYPE_CHECKING:
    from .capabilities import QualityCapabilities

__all__ = [
    "FallbackSelector",
    "FallbackChainResult",
    "StartupCapabilityCheck",
]


@dataclass(slots=True)
class FallbackChainResult:
    """Result of fallback chain selection."""

    selected_tier: QualityTier
    capability_score: float
    gles_version: GLESVersion | None
    required_workarounds: list[str]
    available_features: set[str]
    unavailable_features: set[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "selected_tier": self.selected_tier.name,
            "capability_score": self.capability_score,
            "gles_version": self.gles_version.name if self.gles_version else None,
            "required_workarounds": self.required_workarounds,
            "available_features": list(self.available_features),
            "unavailable_features": list(self.unavailable_features),
            "warnings": self.warnings,
        }


class StartupCapabilityCheck:
    """
    Performs capability check at engine startup.

    Queries wgpu adapter and determines appropriate quality tier
    and fallback chain.
    """

    __slots__ = ("_adapter_info", "_scorer", "_gles_caps")

    def __init__(self, adapter_info: AdapterInfo | dict[str, Any] | None = None):
        if isinstance(adapter_info, dict):
            self._adapter_info = AdapterInfo.from_dict(adapter_info)
        else:
            self._adapter_info = adapter_info or AdapterInfo()

        self._scorer = CapabilityScorer(self._adapter_info)

        # Detect GLES capabilities if applicable
        if self._adapter_info.backend == GPUBackend.OPENGLES:
            self._gles_caps = GLESCapabilities.detect_from_adapter({
                "backend": "OpenGL ES",
                "features": {
                    "compute_shader": self._adapter_info.features.compute_shader,
                    "geometry_shader": False,  # Not in standard FeatureFlags
                    "texture_compression_etc2": self._adapter_info.features.texture_compression_etc2,
                    "texture_compression_astc": self._adapter_info.features.texture_compression_astc,
                },
                "limits": {},
            })
        else:
            self._gles_caps = None

    @property
    def capability_score(self) -> float:
        """Get overall capability score (0.0-1.0)."""
        return self._scorer.score()

    @property
    def adapter_info(self) -> AdapterInfo:
        """Get adapter info."""
        return self._adapter_info

    @property
    def gles_capabilities(self) -> GLESCapabilities | None:
        """Get GLES capabilities if on OpenGL ES backend."""
        return self._gles_caps

    def select_tier(self) -> QualityTier:
        """Select appropriate quality tier based on capabilities."""
        score = self.capability_score

        # GLES 3.0 devices are limited to LOW or MEDIUM
        if self._gles_caps and not self._gles_caps.has_compute:
            return QualityTier.LOW

        # Use standard tier selection
        return QualityTier.from_score(score)

    def get_required_workarounds(self) -> list[str]:
        """Get list of required workarounds."""
        if not self._gles_caps:
            return []

        workarounds = GLESWorkaroundRegistry.list_required(self._gles_caps)
        return [w.feature for w in workarounds]

    def get_available_features(self) -> set[str]:
        """Get set of available GPU features."""
        features = set()
        flags = self._adapter_info.features

        if flags.compute_shader:
            features.add("compute_shader")
        if flags.storage_buffers:
            features.add("storage_buffers")
        if flags.ray_tracing:
            features.add("ray_tracing")
        if flags.ray_query:
            features.add("ray_query")
        if flags.mesh_shader:
            features.add("mesh_shader")
        if flags.bindless:
            features.add("bindless")
        if flags.indirect_draw:
            features.add("indirect_draw")
        if flags.texture_compression_bc:
            features.add("texture_compression_bc")
        if flags.texture_compression_etc2:
            features.add("texture_compression_etc2")
        if flags.texture_compression_astc:
            features.add("texture_compression_astc")

        return features

    def get_unavailable_features(self) -> set[str]:
        """Get set of unavailable features that affect quality."""
        all_features = {
            "compute_shader",
            "storage_buffers",
            "ray_tracing",
            "ray_query",
            "mesh_shader",
            "bindless",
            "indirect_draw",
        }
        return all_features - self.get_available_features()

    def get_warnings(self) -> list[str]:
        """Get warnings about capability limitations."""
        warnings = []

        if self._gles_caps:
            if not self._gles_caps.has_compute:
                warnings.append(
                    "GLES 3.0 detected: No compute shaders. "
                    "GPU culling, clustered lighting, and GPU particles disabled."
                )
            if not self._gles_caps.has_astc_compression:
                warnings.append(
                    "ASTC compression not available. Using ETC2 fallback."
                )

        if self._adapter_info.device_type.name == "INTEGRATED":
            warnings.append(
                "Integrated GPU detected. Performance may be limited."
            )

        if not self._adapter_info.features.ray_query:
            warnings.append(
                "Ray query not available. RT shadows/reflections disabled."
            )

        return warnings

    def perform(self) -> FallbackChainResult:
        """Perform full capability check and return result."""
        return FallbackChainResult(
            selected_tier=self.select_tier(),
            capability_score=self.capability_score,
            gles_version=self._gles_caps.version if self._gles_caps else None,
            required_workarounds=self.get_required_workarounds(),
            available_features=self.get_available_features(),
            unavailable_features=self.get_unavailable_features(),
            warnings=self.get_warnings(),
        )


class FallbackSelector:
    """
    Selects appropriate tier and fallback chain at engine startup.

    Integrates CapabilityScorer, GLESCapabilities, and QualityManager
    to determine the optimal rendering configuration.
    """

    __slots__ = ("_check", "_manager", "_subsystem_overrides")

    def __init__(
        self,
        adapter_info: AdapterInfo | dict[str, Any] | None = None,
        manager_config: QualityManagerConfig | None = None,
    ):
        self._check = StartupCapabilityCheck(adapter_info)
        self._manager = QualityManager(config=manager_config)
        self._subsystem_overrides: dict[str, QualityTier] = {}

    @property
    def capability_check(self) -> StartupCapabilityCheck:
        """Get the capability check."""
        return self._check

    @property
    def quality_manager(self) -> QualityManager:
        """Get the quality manager."""
        return self._manager

    def initialize(self) -> FallbackChainResult:
        """
        Initialize quality system from adapter capabilities.

        Returns the fallback chain result describing the selected
        configuration.
        """
        result = self._check.perform()

        # Set base tier in manager
        self._manager.set_tier(result.selected_tier)

        # Apply GLES-specific overrides
        if self._check.gles_capabilities:
            gles = self._check.gles_capabilities
            if not gles.has_compute:
                # Force LOW tier for subsystems that require compute
                self._manager.set_override("gpu_compute", QualityTier.LOW, locked=True)
                self._manager.set_override("particles", QualityTier.LOW, locked=True)
                self._manager.set_override("gi", QualityTier.LOW, locked=True)

        return result

    def get_fallback_for_feature(self, feature: str) -> str | None:
        """
        Get fallback strategy for a specific feature.

        Returns the workaround strategy string or None if no workaround needed.
        """
        if self._check.gles_capabilities:
            if self._check.gles_capabilities.requires_workaround(feature):
                return GLESWorkaroundRegistry.get_strategy(feature)
        return None

    def should_use_fallback(self, feature: str) -> bool:
        """Check if a fallback should be used for a feature."""
        unavailable = self._check.get_unavailable_features()
        return feature in unavailable or (
            self._check.gles_capabilities is not None
            and self._check.gles_capabilities.requires_workaround(feature)
        )

    def get_effective_tier(self, subsystem: str) -> QualityTier:
        """Get effective tier for a subsystem after overrides."""
        return self._manager.get_tier(subsystem)

    def log_configuration(self) -> str:
        """Generate configuration log for debugging."""
        result = self._check.perform()
        lines = [
            "=== Quality Tier Selection ===",
            f"Adapter: {self._check.adapter_info.name}",
            f"Backend: {self._check.adapter_info.backend.name}",
            f"Device Type: {self._check.adapter_info.device_type.name}",
            f"Capability Score: {result.capability_score:.2f}",
            f"Selected Tier: {result.selected_tier.name}",
        ]

        if result.gles_version:
            lines.append(f"GLES Version: {result.gles_version.name}")

        if result.required_workarounds:
            lines.append(f"Required Workarounds: {', '.join(result.required_workarounds)}")

        if result.warnings:
            lines.append("Warnings:")
            for warning in result.warnings:
                lines.append(f"  - {warning}")

        return "\n".join(lines)
