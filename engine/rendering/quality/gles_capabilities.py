"""GLES 3.1 capability detection and workarounds (T-CC-0.9).

Detects GLES version and provides workarounds for missing features.
GLES 3.0 lacks compute shaders; GLES 3.1 adds them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Callable

from .capability_scorer import FeatureFlags, GPUBackend, GPULimits

__all__ = [
    "GLESVersion",
    "GLESCapabilities",
    "GLESWorkaround",
    "GLESWorkaroundRegistry",
]


class GLESVersion(IntEnum):
    """OpenGL ES version levels."""

    GLES_20 = 20   # WebGL 1.0 equivalent
    GLES_30 = 30   # WebGL 2.0 equivalent, no compute
    GLES_31 = 31   # Compute shaders added
    GLES_32 = 32   # Geometry/tessellation shaders


@dataclass(slots=True)
class GLESCapabilities:
    """
    GLES-specific capability detection.

    Identifies GLES version and available features, providing workarounds
    for features not available on the detected version.
    """

    version: GLESVersion = GLESVersion.GLES_30
    has_compute: bool = False
    has_ssbo: bool = False
    has_image_load_store: bool = False
    has_geometry_shader: bool = False
    has_tessellation: bool = False
    has_texture_buffer: bool = False
    has_etc2_compression: bool = True
    has_astc_compression: bool = False
    max_compute_work_group_size: tuple[int, int, int] = (0, 0, 0)
    max_compute_work_group_count: tuple[int, int, int] = (0, 0, 0)
    max_compute_shared_memory: int = 0

    @classmethod
    def from_version(cls, version: GLESVersion) -> "GLESCapabilities":
        """Create capabilities from GLES version."""
        if version >= GLESVersion.GLES_31:
            return cls(
                version=version,
                has_compute=True,
                has_ssbo=True,
                has_image_load_store=True,
                has_geometry_shader=version >= GLESVersion.GLES_32,
                has_tessellation=version >= GLESVersion.GLES_32,
                has_texture_buffer=version >= GLESVersion.GLES_32,
                max_compute_work_group_size=(1024, 1024, 64),
                max_compute_work_group_count=(65535, 65535, 65535),
                max_compute_shared_memory=32768,
            )
        elif version == GLESVersion.GLES_30:
            return cls(
                version=version,
                has_compute=False,
                has_ssbo=False,
                has_image_load_store=False,
                has_geometry_shader=False,
                has_tessellation=False,
                has_texture_buffer=False,
                max_compute_work_group_size=(0, 0, 0),
                max_compute_work_group_count=(0, 0, 0),
                max_compute_shared_memory=0,
            )
        else:  # GLES 2.0
            return cls(
                version=version,
                has_compute=False,
                has_ssbo=False,
                has_image_load_store=False,
                has_geometry_shader=False,
                has_tessellation=False,
                has_texture_buffer=False,
                has_etc2_compression=False,
                max_compute_work_group_size=(0, 0, 0),
                max_compute_work_group_count=(0, 0, 0),
                max_compute_shared_memory=0,
            )

    @classmethod
    def detect_from_adapter(cls, adapter_info: dict[str, Any]) -> "GLESCapabilities":
        """
        Detect GLES capabilities from adapter info.

        Args:
            adapter_info: Dictionary with 'backend', 'features', 'limits' keys.
        """
        backend = adapter_info.get("backend", "")
        features = adapter_info.get("features", {})
        limits = adapter_info.get("limits", {})

        # Detect version from features
        if features.get("compute_shader", False):
            if features.get("geometry_shader", False):
                version = GLESVersion.GLES_32
            else:
                version = GLESVersion.GLES_31
        elif features.get("texture_compression_etc2", False):
            version = GLESVersion.GLES_30
        else:
            version = GLESVersion.GLES_20

        caps = cls.from_version(version)

        # Override with actual limits if provided
        if "max_compute_work_group_size_x" in limits:
            caps.max_compute_work_group_size = (
                limits.get("max_compute_work_group_size_x", 0),
                limits.get("max_compute_work_group_size_y", 0),
                limits.get("max_compute_work_group_size_z", 0),
            )

        if "max_compute_shared_memory_size" in limits:
            caps.max_compute_shared_memory = limits["max_compute_shared_memory_size"]

        caps.has_astc_compression = features.get("texture_compression_astc", False)

        return caps

    def to_feature_flags(self) -> FeatureFlags:
        """Convert to generic FeatureFlags."""
        return FeatureFlags(
            compute_shader=self.has_compute,
            storage_buffers=self.has_ssbo,
            texture_compression_etc2=self.has_etc2_compression,
            texture_compression_astc=self.has_astc_compression,
            ray_tracing=False,
            ray_query=False,
            mesh_shader=False,
            bindless=False,
            indirect_draw=self.has_compute,
            multi_draw_indirect=False,
        )

    def requires_workaround(self, feature: str) -> bool:
        """Check if a feature requires a workaround on this GLES version."""
        workaround_map = {
            "compute_shader": not self.has_compute,
            "ssbo": not self.has_ssbo,
            "image_load_store": not self.has_image_load_store,
            "geometry_shader": not self.has_geometry_shader,
            "tessellation": not self.has_tessellation,
            "texture_buffer": not self.has_texture_buffer,
            "gpu_culling": not self.has_compute,
            "gpu_particles": not self.has_compute,
            "clustered_lighting": not self.has_compute,
        }
        return workaround_map.get(feature, False)


@dataclass(slots=True)
class GLESWorkaround:
    """
    Defines a workaround for a missing GLES feature.

    Contains the feature name, minimum GLES version that has it natively,
    and the workaround strategy for older versions.
    """

    feature: str
    min_version: GLESVersion
    workaround_strategy: str
    performance_impact: str
    implementation_notes: str


# Registry of GLES workarounds
GLES_WORKAROUNDS: list[GLESWorkaround] = [
    GLESWorkaround(
        feature="compute_shader",
        min_version=GLESVersion.GLES_31,
        workaround_strategy="Use vertex/fragment shader ping-pong with render targets",
        performance_impact="2-4x slower than compute; requires extra draw calls and texture reads",
        implementation_notes=(
            "GPU particles: Use transform feedback or texture-based particle state. "
            "GPU culling: Fall back to CPU frustum culling or use hierarchical-Z on CPU. "
            "Light clustering: Use forward rendering with limited light count per draw."
        ),
    ),
    GLESWorkaround(
        feature="ssbo",
        min_version=GLESVersion.GLES_31,
        workaround_strategy="Use uniform buffer objects (UBO) with packing",
        performance_impact="Limited to 16KB per UBO; requires multiple draw calls for large data",
        implementation_notes=(
            "Pack data tightly; use vec4 alignment. Split large buffers into multiple UBOs. "
            "For large arrays, use texture buffers or instanced attributes."
        ),
    ),
    GLESWorkaround(
        feature="image_load_store",
        min_version=GLESVersion.GLES_31,
        workaround_strategy="Use framebuffer render targets with MRT",
        performance_impact="No random-access writes; must render full quad per output",
        implementation_notes=(
            "For post-processing: render to FBO instead of image store. "
            "For scatter writes: use point primitives with gl_PointSize."
        ),
    ),
    GLESWorkaround(
        feature="gpu_culling",
        min_version=GLESVersion.GLES_31,
        workaround_strategy="CPU frustum + distance culling with spatial partitioning",
        performance_impact="CPU-bound; use BVH or octree for large object counts",
        implementation_notes=(
            "Implement BVH traversal on CPU. Use view-distance LOD selection. "
            "Consider hybrid: coarse CPU cull + fine GPU cull if compute available."
        ),
    ),
    GLESWorkaround(
        feature="gpu_particles",
        min_version=GLESVersion.GLES_31,
        workaround_strategy="Transform feedback or texture-based particle system",
        performance_impact="Limited particle count (~10K vs 100K+ with compute)",
        implementation_notes=(
            "Use transform feedback to update particle positions. "
            "Store particle state in RGBA32F texture. "
            "Render with instanced point sprites."
        ),
    ),
    GLESWorkaround(
        feature="clustered_lighting",
        min_version=GLESVersion.GLES_31,
        workaround_strategy="Forward rendering with per-object light list",
        performance_impact="Limited to 8-16 lights per object; CPU overhead",
        implementation_notes=(
            "Sort lights by influence radius. Assign nearest N lights per object. "
            "Use deferred lighting only on GLES 3.1+ with compute."
        ),
    ),
    GLESWorkaround(
        feature="geometry_shader",
        min_version=GLESVersion.GLES_32,
        workaround_strategy="Pre-expand geometry on CPU or use instancing",
        performance_impact="Higher memory usage; CPU preprocessing required",
        implementation_notes=(
            "For billboards: use instanced quads with vertex shader rotation. "
            "For wireframe: pre-generate line strips. "
            "For shadow volume extrusion: pre-compute on CPU."
        ),
    ),
    GLESWorkaround(
        feature="tessellation",
        min_version=GLESVersion.GLES_32,
        workaround_strategy="Pre-tessellated meshes with LOD",
        performance_impact="Higher memory; less adaptive detail",
        implementation_notes=(
            "Generate multiple LOD meshes offline. "
            "Use displacement mapping with parallax occlusion if available. "
            "For terrain: use clipmap-based LOD."
        ),
    ),
]


class GLESWorkaroundRegistry:
    """Registry for looking up GLES workarounds."""

    _workarounds: dict[str, GLESWorkaround] = {w.feature: w for w in GLES_WORKAROUNDS}

    @classmethod
    def get(cls, feature: str) -> GLESWorkaround | None:
        """Get workaround for a feature."""
        return cls._workarounds.get(feature)

    @classmethod
    def list_required(cls, caps: GLESCapabilities) -> list[GLESWorkaround]:
        """List all workarounds required for given capabilities."""
        required = []
        for workaround in GLES_WORKAROUNDS:
            if caps.requires_workaround(workaround.feature):
                required.append(workaround)
        return required

    @classmethod
    def get_strategy(cls, feature: str) -> str | None:
        """Get workaround strategy string for a feature."""
        workaround = cls._workarounds.get(feature)
        return workaround.workaround_strategy if workaround else None

    @classmethod
    def all_workarounds(cls) -> list[GLESWorkaround]:
        """Get all registered workarounds."""
        return list(cls._workarounds.values())
