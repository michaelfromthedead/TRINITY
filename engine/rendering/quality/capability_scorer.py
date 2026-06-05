"""
GPU capability scoring from adapter information (T-CC-0.2).

Provides scoring of GPU capabilities based on wgpu adapter limits
and feature flags. Used by QualityManager to select appropriate tier.
"""

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional


class GPUBackend(IntEnum):
    """GPU backend type (from wgpu)."""

    UNKNOWN = 0
    VULKAN = auto()
    D3D12 = auto()
    D3D11 = auto()
    METAL = auto()
    OPENGL = auto()
    OPENGLES = auto()
    WEBGPU = auto()


class GPUDeviceType(IntEnum):
    """GPU device type (from wgpu)."""

    UNKNOWN = 0
    INTEGRATED = auto()
    DISCRETE = auto()
    VIRTUAL = auto()
    CPU = auto()


@dataclass
class FeatureFlags:
    """GPU feature flags for capability detection."""

    compute_shader: bool = True
    storage_buffers: bool = True
    texture_compression_bc: bool = False
    texture_compression_etc2: bool = False
    texture_compression_astc: bool = False
    ray_tracing: bool = False
    ray_query: bool = False
    mesh_shader: bool = False
    bindless: bool = False
    variable_rate_shading: bool = False
    depth_clip_control: bool = True
    timestamp_query: bool = True
    indirect_draw: bool = True
    multi_draw_indirect: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, bool]) -> "FeatureFlags":
        """Create from dictionary of feature names to booleans."""
        known_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


@dataclass
class GPULimits:
    """GPU limits affecting quality tier selection."""

    max_texture_dimension_2d: int = 8192
    max_texture_dimension_3d: int = 2048
    max_texture_array_layers: int = 256
    max_bind_groups: int = 4
    max_bindings_per_bind_group: int = 1000
    max_dynamic_uniform_buffers_per_pipeline_layout: int = 8
    max_dynamic_storage_buffers_per_pipeline_layout: int = 4
    max_sampled_textures_per_shader_stage: int = 16
    max_samplers_per_shader_stage: int = 16
    max_storage_buffers_per_shader_stage: int = 8
    max_storage_textures_per_shader_stage: int = 4
    max_uniform_buffers_per_shader_stage: int = 12
    max_uniform_buffer_binding_size: int = 65536
    max_storage_buffer_binding_size: int = 134217728  # 128MB
    max_vertex_buffers: int = 8
    max_vertex_attributes: int = 16
    max_vertex_buffer_array_stride: int = 2048
    max_inter_stage_shader_components: int = 60
    max_compute_workgroup_storage_size: int = 16384
    max_compute_invocations_per_workgroup: int = 256
    max_compute_workgroup_size_x: int = 256
    max_compute_workgroup_size_y: int = 256
    max_compute_workgroup_size_z: int = 64
    max_compute_workgroups_per_dimension: int = 65535

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> "GPULimits":
        """Create from dictionary of limit names to values."""
        known_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    @property
    def vram_estimate_mb(self) -> int:
        """Estimate VRAM from limits (heuristic)."""
        storage = self.max_storage_buffer_binding_size
        if storage >= 2147483648:  # 2GB+
            return 8192
        elif storage >= 1073741824:  # 1GB+
            return 4096
        elif storage >= 268435456:  # 256MB+
            return 2048
        elif storage >= 134217728:  # 128MB+
            return 1024
        else:
            return 512


@dataclass
class AdapterInfo:
    """GPU adapter information from wgpu."""

    name: str = "Unknown"
    vendor: str = "Unknown"
    device_id: int = 0
    vendor_id: int = 0
    backend: GPUBackend = GPUBackend.UNKNOWN
    device_type: GPUDeviceType = GPUDeviceType.UNKNOWN
    driver: str = ""
    driver_info: str = ""
    features: FeatureFlags = field(default_factory=FeatureFlags)
    limits: GPULimits = field(default_factory=GPULimits)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdapterInfo":
        """Create from dictionary (e.g., from wgpu adapter.info)."""
        features = data.get("features", {})
        limits = data.get("limits", {})

        backend_str = data.get("backend", "unknown").upper()
        backend = getattr(GPUBackend, backend_str, GPUBackend.UNKNOWN)

        device_type_str = data.get("device_type", "unknown").upper()
        device_type = getattr(GPUDeviceType, device_type_str, GPUDeviceType.UNKNOWN)

        return cls(
            name=data.get("name", "Unknown"),
            vendor=data.get("vendor", "Unknown"),
            device_id=data.get("device_id", 0),
            vendor_id=data.get("vendor_id", 0),
            backend=backend,
            device_type=device_type,
            driver=data.get("driver", ""),
            driver_info=data.get("driver_info", ""),
            features=FeatureFlags.from_dict(features)
            if isinstance(features, dict)
            else features,
            limits=GPULimits.from_dict(limits)
            if isinstance(limits, dict)
            else limits,
        )


class CapabilityScorer:
    """
    Scores GPU capabilities to produce a 0.0-1.0 capability score.

    The score is used by QualityManager to select the appropriate
    quality tier for the detected hardware.
    """

    # Scoring weights (should sum to approximately 1.0)
    WEIGHT_DEVICE_TYPE = 0.25
    WEIGHT_FEATURES = 0.35
    WEIGHT_LIMITS = 0.25
    WEIGHT_BACKEND = 0.15

    def __init__(self, adapter_info: Optional[AdapterInfo] = None):
        self._adapter = adapter_info or AdapterInfo()

    @property
    def adapter_info(self) -> AdapterInfo:
        """Get the adapter info being scored."""
        return self._adapter

    def score(self) -> float:
        """
        Compute overall capability score (0.0 to 1.0).

        Higher scores indicate more capable hardware.
        """
        device_score = self._score_device_type()
        feature_score = self._score_features()
        limit_score = self._score_limits()
        backend_score = self._score_backend()

        total = (
            device_score * self.WEIGHT_DEVICE_TYPE
            + feature_score * self.WEIGHT_FEATURES
            + limit_score * self.WEIGHT_LIMITS
            + backend_score * self.WEIGHT_BACKEND
        )

        return max(0.0, min(1.0, total))

    def _score_device_type(self) -> float:
        """Score based on GPU device type."""
        scores = {
            GPUDeviceType.DISCRETE: 1.0,
            GPUDeviceType.VIRTUAL: 0.8,
            GPUDeviceType.INTEGRATED: 0.5,
            GPUDeviceType.CPU: 0.2,
            GPUDeviceType.UNKNOWN: 0.3,
        }
        return scores.get(self._adapter.device_type, 0.3)

    def _score_features(self) -> float:
        """Score based on supported features."""
        f = self._adapter.features
        score = 0.0

        # Essential features (baseline)
        if f.compute_shader:
            score += 0.15
        if f.storage_buffers:
            score += 0.15
        if f.indirect_draw:
            score += 0.1

        # Advanced features
        if f.ray_tracing:
            score += 0.2
        elif f.ray_query:
            score += 0.1
        if f.mesh_shader:
            score += 0.1
        if f.bindless:
            score += 0.1
        if f.multi_draw_indirect:
            score += 0.1

        # Texture compression (pick one)
        if f.texture_compression_bc:
            score += 0.05
        elif f.texture_compression_astc:
            score += 0.05
        elif f.texture_compression_etc2:
            score += 0.03

        # Nice-to-have
        if f.variable_rate_shading:
            score += 0.05

        return min(1.0, score)

    def _score_limits(self) -> float:
        """Score based on GPU limits."""
        lim = self._adapter.limits
        score = 0.0

        # Texture size
        if lim.max_texture_dimension_2d >= 16384:
            score += 0.2
        elif lim.max_texture_dimension_2d >= 8192:
            score += 0.15
        elif lim.max_texture_dimension_2d >= 4096:
            score += 0.1

        # Compute capabilities
        if lim.max_compute_invocations_per_workgroup >= 1024:
            score += 0.2
        elif lim.max_compute_invocations_per_workgroup >= 256:
            score += 0.1

        # Storage buffer size (VRAM indicator)
        vram = lim.vram_estimate_mb
        if vram >= 8192:
            score += 0.3
        elif vram >= 4096:
            score += 0.25
        elif vram >= 2048:
            score += 0.2
        elif vram >= 1024:
            score += 0.1
        else:
            score += 0.05

        # Bind group limits
        if lim.max_bindings_per_bind_group >= 1000:
            score += 0.15
        elif lim.max_bindings_per_bind_group >= 500:
            score += 0.1

        # Storage buffers per stage
        if lim.max_storage_buffers_per_shader_stage >= 8:
            score += 0.15
        elif lim.max_storage_buffers_per_shader_stage >= 4:
            score += 0.1

        return min(1.0, score)

    def _score_backend(self) -> float:
        """Score based on backend type."""
        scores = {
            GPUBackend.VULKAN: 1.0,
            GPUBackend.D3D12: 1.0,
            GPUBackend.METAL: 0.95,
            GPUBackend.WEBGPU: 0.85,
            GPUBackend.D3D11: 0.7,
            GPUBackend.OPENGL: 0.5,
            GPUBackend.OPENGLES: 0.3,
            GPUBackend.UNKNOWN: 0.5,
        }
        return scores.get(self._adapter.backend, 0.5)

    def explain(self) -> dict[str, float]:
        """Return breakdown of score components for debugging."""
        return {
            "device_type": self._score_device_type(),
            "features": self._score_features(),
            "limits": self._score_limits(),
            "backend": self._score_backend(),
            "total": self.score(),
        }
