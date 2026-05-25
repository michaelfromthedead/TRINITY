"""XR Rendering module.

Provides comprehensive XR rendering support including:
- Stereo rendering (Multi-View, Instanced, Sequential)
- Foveated rendering (Fixed, Dynamic, Contrast-Adaptive)
- Reprojection (ATW, ASW, Hybrid)
- Compositor layer management
- Hidden area mesh optimization

Performance targets:
- 90Hz/120Hz frame rates
- 11ms frame time budget
- Minimal motion-to-photon latency
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Tuple

# Stereo rendering
from .stereo import (
    StereoMethod,
    ProjectionType,
    IPDMode,
    EyeIndex,
    ViewFrustum,
    EyeView,
    StereoConfig,
    StereoRenderTarget,
    StereoRenderer,
    MultiViewStereoRenderer,
    InstancedStereoRenderer,
    SequentialStereoRenderer,
    create_stereo_renderer,
)

# Foveated rendering
from .foveated import (
    FoveationType,
    FoveationRegion,
    ShadingRate,
    FoveationRegionConfig,
    GazePoint,
    FoveationConfig,
    FoveationMetrics,
    FoveatedRenderer,
    FixedFoveatedRenderer,
    DynamicFoveatedRenderer,
    ContrastAdaptiveFoveatedRenderer,
    create_foveated_renderer,
)

# Reprojection
from .reprojection import (
    ReprojectionMode,
    PredictionMethod,
    Pose,
    PoseVelocity,
    MotionVector,
    ReprojectionConfig,
    ReprojectionMetrics,
    ReprojectedFrame,
    XRReprojection,
    ATWReprojection,
    ASWReprojection,
    HybridReprojection,
    create_reprojection,
)

# Compositor
from .compositor import (
    LayerType,
    BlendMode,
    LayerFlags,
    LayerPose,
    Viewport,
    ProjectionLayerConfig,
    QuadLayerConfig,
    CylinderLayerConfig,
    CubemapLayerConfig,
    EquirectLayerConfig,
    Layer,
    CompositorConfig,
    CompositorMetrics,
    CompositorLayer,
    XRCompositor,
    NullCompositorLayer,
    NullXRCompositor,
    ProjectionLayer,
    QuadLayer,
    CylinderLayer,
    create_compositor,
)

# Hidden area mesh
from .hidden_area import (
    HiddenAreaType,
    MeshFormat,
    Vertex2D,
    Triangle,
    HiddenAreaMeshData,
    HiddenAreaConfig,
    HiddenAreaMetrics,
    HiddenAreaMask,
    NullHiddenAreaMask,
    StencilHiddenAreaMask,
    DepthHiddenAreaMask,
    CombinedHiddenAreaMask,
    create_hidden_area_mask,
)


class XRDisplayMode(Enum):
    """XR display mode."""
    VR = auto()         # Full VR immersion
    AR = auto()         # Augmented reality passthrough
    MR = auto()         # Mixed reality
    SPECTATOR = auto()  # 2D spectator view


class RefreshRate(Enum):
    """Supported XR refresh rates."""
    HZ_72 = 72
    HZ_80 = 80
    HZ_90 = 90
    HZ_120 = 120
    HZ_144 = 144


@dataclass
class XRRenderSettings:
    """Unified XR rendering settings resource.

    Combines all rendering configuration into a single resource
    that can be serialized and applied atomically.
    """
    # Display settings
    display_mode: XRDisplayMode = XRDisplayMode.VR
    refresh_rate: RefreshRate = RefreshRate.HZ_90
    resolution_scale: float = 1.0

    # Stereo configuration
    stereo: StereoConfig = field(default_factory=StereoConfig)

    # Foveation configuration
    foveation: FoveationConfig = field(default_factory=FoveationConfig)

    # Reprojection configuration
    reprojection: ReprojectionConfig = field(default_factory=ReprojectionConfig)

    # Compositor configuration
    compositor: CompositorConfig = field(default_factory=CompositorConfig)

    # Hidden area configuration
    hidden_area: HiddenAreaConfig = field(default_factory=HiddenAreaConfig)

    # Performance settings
    target_frame_time_ms: float = 11.11  # ~90Hz
    max_frame_time_ms: float = 13.89     # ~72Hz fallback
    late_latch_enabled: bool = True
    dynamic_resolution: bool = True

    # Quality presets
    @classmethod
    def low_quality(cls) -> 'XRRenderSettings':
        """Low quality preset for performance-constrained systems."""
        return cls(
            resolution_scale=0.7,
            stereo=StereoConfig(method=StereoMethod.SEQUENTIAL),
            foveation=FoveationConfig(
                type=FoveationType.FIXED,
                fovea_rate=ShadingRate.HALF,
                parafoveal_rate=ShadingRate.QUARTER,
                peripheral_rate=ShadingRate.QUARTER
            ),
            reprojection=ReprojectionConfig(mode=ReprojectionMode.ATW),
            dynamic_resolution=True
        )

    @classmethod
    def medium_quality(cls) -> 'XRRenderSettings':
        """Medium quality preset for balanced performance."""
        return cls(
            resolution_scale=1.0,
            stereo=StereoConfig(method=StereoMethod.INSTANCED),
            foveation=FoveationConfig(
                type=FoveationType.FIXED,
                fovea_rate=ShadingRate.FULL,
                parafoveal_rate=ShadingRate.HALF,
                peripheral_rate=ShadingRate.QUARTER
            ),
            reprojection=ReprojectionConfig(mode=ReprojectionMode.ATW),
            dynamic_resolution=True
        )

    @classmethod
    def high_quality(cls) -> 'XRRenderSettings':
        """High quality preset for capable systems."""
        return cls(
            resolution_scale=1.2,
            refresh_rate=RefreshRate.HZ_120,
            stereo=StereoConfig(method=StereoMethod.MULTI_VIEW),
            foveation=FoveationConfig(
                type=FoveationType.DYNAMIC,
                fovea_rate=ShadingRate.FULL,
                parafoveal_rate=ShadingRate.FULL,
                peripheral_rate=ShadingRate.HALF
            ),
            reprojection=ReprojectionConfig(mode=ReprojectionMode.HYBRID),
            dynamic_resolution=False
        )

    @classmethod
    def ultra_quality(cls) -> 'XRRenderSettings':
        """Ultra quality preset with all features enabled."""
        return cls(
            resolution_scale=1.5,
            refresh_rate=RefreshRate.HZ_144,
            stereo=StereoConfig(
                method=StereoMethod.MULTI_VIEW,
                projection_type=ProjectionType.ASYMMETRIC
            ),
            foveation=FoveationConfig(
                type=FoveationType.CONTRAST_ADAPTIVE,
                fovea_rate=ShadingRate.FULL,
                parafoveal_rate=ShadingRate.FULL,
                peripheral_rate=ShadingRate.FULL
            ),
            reprojection=ReprojectionConfig(mode=ReprojectionMode.ASW),
            hidden_area=HiddenAreaConfig(type=HiddenAreaType.COMBINED),
            dynamic_resolution=False
        )


class XRRenderPipeline:
    """High-level XR rendering pipeline orchestrator.

    Coordinates stereo, foveation, reprojection, compositor, and
    hidden area components into a unified pipeline.
    """

    def __init__(self, settings: Optional[XRRenderSettings] = None):
        """Initialize XR render pipeline.

        Args:
            settings: Render settings (uses medium quality default)
        """
        self._settings = settings or XRRenderSettings.medium_quality()
        self._stereo = create_stereo_renderer(self._settings.stereo)
        self._foveation = create_foveated_renderer(self._settings.foveation)
        self._reprojection = create_reprojection(self._settings.reprojection)
        self._compositor = create_compositor(self._settings.compositor)
        self._hidden_area = create_hidden_area_mask(self._settings.hidden_area)
        self._frame_count = 0

    @property
    def settings(self) -> XRRenderSettings:
        """Get current render settings."""
        return self._settings

    def apply_settings(self, settings: XRRenderSettings) -> None:
        """Apply new render settings.

        Args:
            settings: New settings to apply
        """
        self._settings = settings
        self._stereo.configure(settings.stereo)
        self._foveation.configure(settings.foveation)
        self._reprojection.configure(settings.reprojection)
        self._compositor.configure(settings.compositor)
        self._hidden_area.configure(settings.hidden_area)

    @property
    def stereo(self) -> StereoRenderer:
        """Get stereo renderer."""
        return self._stereo

    @property
    def foveation(self) -> FoveatedRenderer:
        """Get foveated renderer."""
        return self._foveation

    @property
    def reprojection(self) -> XRReprojection:
        """Get reprojection handler."""
        return self._reprojection

    @property
    def compositor(self) -> XRCompositor:
        """Get compositor."""
        return self._compositor

    @property
    def hidden_area(self) -> HiddenAreaMask:
        """Get hidden area mask."""
        return self._hidden_area

    def begin_frame(self) -> None:
        """Begin XR frame rendering."""
        self._frame_count += 1
        self._stereo.begin_frame()
        self._foveation.begin_frame()
        self._compositor.begin_frame()

    def begin_eye(self, eye: EyeIndex) -> None:
        """Begin rendering for an eye.

        Args:
            eye: Which eye to render
        """
        self._stereo.begin_eye(eye)
        self._hidden_area.begin_mask(eye.value)

    def end_eye(self, eye: EyeIndex) -> None:
        """End rendering for an eye.

        Args:
            eye: Which eye was rendered
        """
        self._hidden_area.end_mask(eye.value)
        self._stereo.end_eye(eye)

    def submit_frame(self, render_pose: Pose) -> None:
        """Submit rendered frame for reprojection.

        Args:
            render_pose: Pose used for rendering
        """
        self._reprojection.submit_frame(render_pose, self._frame_count)
        self._compositor.submit_layers()

    def end_frame(self) -> None:
        """End XR frame rendering."""
        self._stereo.end_frame()
        self._foveation.end_frame()
        self._compositor.end_frame()

    def get_frame_time_budget_ms(self) -> float:
        """Get frame time budget based on refresh rate."""
        hz = self._settings.refresh_rate.value
        return 1000.0 / hz


# Convenience function to create full pipeline
def create_xr_render_pipeline(
    quality: str = "medium"
) -> XRRenderPipeline:
    """Create XR render pipeline with quality preset.

    Args:
        quality: Quality preset ("low", "medium", "high", "ultra")

    Returns:
        Configured XR render pipeline
    """
    presets = {
        "low": XRRenderSettings.low_quality,
        "medium": XRRenderSettings.medium_quality,
        "high": XRRenderSettings.high_quality,
        "ultra": XRRenderSettings.ultra_quality,
    }

    settings_factory = presets.get(quality.lower(), XRRenderSettings.medium_quality)
    return XRRenderPipeline(settings_factory())


__all__ = [
    # Stereo
    "StereoMethod",
    "ProjectionType",
    "IPDMode",
    "EyeIndex",
    "ViewFrustum",
    "EyeView",
    "StereoConfig",
    "StereoRenderTarget",
    "StereoRenderer",
    "MultiViewStereoRenderer",
    "InstancedStereoRenderer",
    "SequentialStereoRenderer",
    "create_stereo_renderer",
    # Foveated
    "FoveationType",
    "FoveationRegion",
    "ShadingRate",
    "FoveationRegionConfig",
    "GazePoint",
    "FoveationConfig",
    "FoveationMetrics",
    "FoveatedRenderer",
    "FixedFoveatedRenderer",
    "DynamicFoveatedRenderer",
    "ContrastAdaptiveFoveatedRenderer",
    "create_foveated_renderer",
    # Reprojection
    "ReprojectionMode",
    "PredictionMethod",
    "Pose",
    "PoseVelocity",
    "MotionVector",
    "ReprojectionConfig",
    "ReprojectionMetrics",
    "ReprojectedFrame",
    "XRReprojection",
    "ATWReprojection",
    "ASWReprojection",
    "HybridReprojection",
    "create_reprojection",
    # Compositor
    "LayerType",
    "BlendMode",
    "LayerFlags",
    "LayerPose",
    "Viewport",
    "ProjectionLayerConfig",
    "QuadLayerConfig",
    "CylinderLayerConfig",
    "CubemapLayerConfig",
    "EquirectLayerConfig",
    "Layer",
    "CompositorConfig",
    "CompositorMetrics",
    "CompositorLayer",
    "XRCompositor",
    "NullCompositorLayer",
    "NullXRCompositor",
    "ProjectionLayer",
    "QuadLayer",
    "CylinderLayer",
    "create_compositor",
    # Hidden area
    "HiddenAreaType",
    "MeshFormat",
    "Vertex2D",
    "Triangle",
    "HiddenAreaMeshData",
    "HiddenAreaConfig",
    "HiddenAreaMetrics",
    "HiddenAreaMask",
    "NullHiddenAreaMask",
    "StencilHiddenAreaMask",
    "DepthHiddenAreaMask",
    "CombinedHiddenAreaMask",
    "create_hidden_area_mask",
    # High-level
    "XRDisplayMode",
    "RefreshRate",
    "XRRenderSettings",
    "XRRenderPipeline",
    "create_xr_render_pipeline",
]
