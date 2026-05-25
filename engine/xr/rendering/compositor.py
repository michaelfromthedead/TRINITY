"""XR compositor layer management.

Handles composition of multiple rendering layers:
- Projection: Main 3D scene (stereo pair)
- Quad: UI panels, floating windows
- Cylinder: Curved UI, panoramic displays
- Cubemap: Skyboxes, 360 video
- Equirect: Equirectangular 360 content

Manages layer ordering, blending, and submission to XR runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Tuple, Any
import threading


class LayerType(Enum):
    """Compositor layer type."""
    PROJECTION = auto()   # Main stereo projection
    QUAD = auto()         # Flat 2D panel in 3D space
    CYLINDER = auto()     # Cylindrical surface
    CUBEMAP = auto()      # 6-face cube environment
    EQUIRECT = auto()     # Equirectangular panorama


class BlendMode(Enum):
    """Layer blending mode."""
    OPAQUE = auto()           # No transparency
    ALPHA_BLEND = auto()      # Standard alpha blending
    PREMULTIPLIED = auto()    # Premultiplied alpha
    ADDITIVE = auto()         # Additive blending


class LayerFlags(Enum):
    """Layer behavior flags."""
    HEAD_LOCKED = auto()      # Locked to HMD
    WORLD_LOCKED = auto()     # Fixed in world space
    DEPTH_TEST = auto()       # Participates in depth testing
    UNPREMULTIPLIED = auto()  # Alpha not premultiplied
    STATIC = auto()           # Content rarely changes
    CHROMATIC_ABERRATION = auto()  # Apply CA correction


@dataclass
class LayerPose:
    """3D pose for layer positioning."""
    position: Tuple[float, float, float] = (0.0, 0.0, -1.0)
    orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # quaternion
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class Viewport:
    """Rendering viewport."""
    x: int = 0
    y: int = 0
    width: int = 1920
    height: int = 2160


@dataclass
class ProjectionLayerConfig:
    """Configuration for projection (main scene) layer."""
    near_z: float = 0.1
    far_z: float = 1000.0
    viewports: List[Viewport] = field(default_factory=lambda: [Viewport(), Viewport()])
    depth_info_enabled: bool = True


@dataclass
class QuadLayerConfig:
    """Configuration for quad layer."""
    size: Tuple[float, float] = (1.0, 1.0)  # Width, height in meters
    pose: LayerPose = field(default_factory=LayerPose)
    subimage_rect: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h in texture
    blend_mode: BlendMode = BlendMode.ALPHA_BLEND
    eye_visibility: int = 3  # Bitmask: 1=left, 2=right, 3=both


@dataclass
class CylinderLayerConfig:
    """Configuration for cylinder layer."""
    radius: float = 2.0
    central_angle: float = 1.5708  # 90 degrees in radians
    aspect_ratio: float = 2.0
    pose: LayerPose = field(default_factory=LayerPose)
    blend_mode: BlendMode = BlendMode.ALPHA_BLEND
    eye_visibility: int = 3


@dataclass
class CubemapLayerConfig:
    """Configuration for cubemap layer."""
    pose: LayerPose = field(default_factory=LayerPose)
    blend_mode: BlendMode = BlendMode.OPAQUE
    # Face order: +X, -X, +Y, -Y, +Z, -Z


@dataclass
class EquirectLayerConfig:
    """Configuration for equirectangular layer."""
    pose: LayerPose = field(default_factory=LayerPose)
    central_horizontal_angle: float = 6.283  # 360 degrees
    upper_vertical_angle: float = 1.571      # 90 degrees
    lower_vertical_angle: float = -1.571     # -90 degrees
    blend_mode: BlendMode = BlendMode.OPAQUE


@dataclass
class Layer:
    """Compositor layer."""
    id: int
    name: str
    type: LayerType
    config: Any  # Type-specific config
    texture_handle: int = 0
    depth_handle: int = 0
    priority: int = 0  # Lower = rendered first (background)
    visible: bool = True
    flags: List[LayerFlags] = field(default_factory=list)


@dataclass
class CompositorConfig:
    """Compositor configuration."""
    max_layers: int = 16
    default_blend_mode: BlendMode = BlendMode.ALPHA_BLEND
    depth_testing_enabled: bool = True
    chromatic_aberration_correction: bool = True
    hidden_area_mask_enabled: bool = True


@dataclass
class CompositorMetrics:
    """Compositor performance metrics."""
    active_layers: int = 0
    layers_composited: int = 0
    blend_operations: int = 0
    submit_time_ms: float = 0.0


class CompositorLayer(ABC):
    """Abstract compositor layer interface."""

    @property
    @abstractmethod
    def layer(self) -> Layer:
        """Get layer data."""
        pass

    @abstractmethod
    def set_texture(self, texture_handle: int, depth_handle: int = 0) -> None:
        """Set layer textures.

        Args:
            texture_handle: Color texture handle
            depth_handle: Optional depth texture handle
        """
        pass

    @abstractmethod
    def set_visible(self, visible: bool) -> None:
        """Set layer visibility."""
        pass

    @abstractmethod
    def update_config(self, config: Any) -> None:
        """Update layer configuration."""
        pass


class XRCompositor(ABC):
    """Abstract XR compositor interface."""

    @property
    @abstractmethod
    def config(self) -> CompositorConfig:
        """Get compositor configuration."""
        pass

    @abstractmethod
    def configure(self, config: CompositorConfig) -> None:
        """Update compositor configuration."""
        pass

    @abstractmethod
    def create_projection_layer(self, name: str,
                                 config: ProjectionLayerConfig) -> CompositorLayer:
        """Create main projection layer.

        Args:
            name: Layer name
            config: Projection layer configuration

        Returns:
            Created layer
        """
        pass

    @abstractmethod
    def create_quad_layer(self, name: str, config: QuadLayerConfig) -> CompositorLayer:
        """Create quad UI layer."""
        pass

    @abstractmethod
    def create_cylinder_layer(self, name: str,
                               config: CylinderLayerConfig) -> CompositorLayer:
        """Create cylinder layer."""
        pass

    @abstractmethod
    def create_cubemap_layer(self, name: str,
                              config: CubemapLayerConfig) -> CompositorLayer:
        """Create cubemap environment layer."""
        pass

    @abstractmethod
    def create_equirect_layer(self, name: str,
                               config: EquirectLayerConfig) -> CompositorLayer:
        """Create equirectangular panorama layer."""
        pass

    @abstractmethod
    def destroy_layer(self, layer_id: int) -> bool:
        """Destroy a layer.

        Args:
            layer_id: Layer to destroy

        Returns:
            True if destroyed successfully
        """
        pass

    @abstractmethod
    def get_layer(self, layer_id: int) -> Optional[CompositorLayer]:
        """Get layer by ID."""
        pass

    @abstractmethod
    def get_layers(self) -> List[CompositorLayer]:
        """Get all layers in priority order."""
        pass

    @abstractmethod
    def set_layer_priority(self, layer_id: int, priority: int) -> None:
        """Set layer render priority."""
        pass

    @abstractmethod
    def begin_frame(self) -> None:
        """Begin compositor frame."""
        pass

    @abstractmethod
    def submit_layers(self) -> None:
        """Submit all visible layers to XR runtime."""
        pass

    @abstractmethod
    def end_frame(self) -> None:
        """End compositor frame."""
        pass

    @abstractmethod
    def get_metrics(self) -> CompositorMetrics:
        """Get compositor metrics."""
        pass


class NullCompositorLayer(CompositorLayer):
    """Null implementation of compositor layer."""

    def __init__(self, layer: Layer):
        self._layer = layer

    @property
    def layer(self) -> Layer:
        return self._layer

    def set_texture(self, texture_handle: int, depth_handle: int = 0) -> None:
        self._layer.texture_handle = texture_handle
        self._layer.depth_handle = depth_handle

    def set_visible(self, visible: bool) -> None:
        self._layer.visible = visible

    def update_config(self, config: Any) -> None:
        self._layer.config = config


class NullXRCompositor(XRCompositor):
    """Null implementation of XR compositor for testing."""

    def __init__(self, config: Optional[CompositorConfig] = None):
        self._config = config or CompositorConfig()
        self._layers: Dict[int, NullCompositorLayer] = {}
        self._next_layer_id = 1
        self._metrics = CompositorMetrics()
        self._lock = threading.Lock()
        self._frame_active = False

    @property
    def config(self) -> CompositorConfig:
        return self._config

    def configure(self, config: CompositorConfig) -> None:
        with self._lock:
            self._config = config

    def create_projection_layer(self, name: str,
                                 config: ProjectionLayerConfig) -> CompositorLayer:
        return self._create_layer(name, LayerType.PROJECTION, config, priority=0)

    def create_quad_layer(self, name: str, config: QuadLayerConfig) -> CompositorLayer:
        return self._create_layer(name, LayerType.QUAD, config, priority=100)

    def create_cylinder_layer(self, name: str,
                               config: CylinderLayerConfig) -> CompositorLayer:
        return self._create_layer(name, LayerType.CYLINDER, config, priority=100)

    def create_cubemap_layer(self, name: str,
                              config: CubemapLayerConfig) -> CompositorLayer:
        return self._create_layer(name, LayerType.CUBEMAP, config, priority=-100)

    def create_equirect_layer(self, name: str,
                               config: EquirectLayerConfig) -> CompositorLayer:
        return self._create_layer(name, LayerType.EQUIRECT, config, priority=-100)

    def _create_layer(self, name: str, layer_type: LayerType,
                      config: Any, priority: int) -> CompositorLayer:
        with self._lock:
            if len(self._layers) >= self._config.max_layers:
                raise RuntimeError(f"Maximum layers ({self._config.max_layers}) exceeded")

            layer_id = self._next_layer_id
            self._next_layer_id += 1

            layer = Layer(
                id=layer_id,
                name=name,
                type=layer_type,
                config=config,
                priority=priority
            )

            compositor_layer = NullCompositorLayer(layer)
            self._layers[layer_id] = compositor_layer

            return compositor_layer

    def destroy_layer(self, layer_id: int) -> bool:
        with self._lock:
            if layer_id in self._layers:
                del self._layers[layer_id]
                return True
            return False

    def get_layer(self, layer_id: int) -> Optional[CompositorLayer]:
        return self._layers.get(layer_id)

    def get_layers(self) -> List[CompositorLayer]:
        with self._lock:
            sorted_layers = sorted(
                self._layers.values(),
                key=lambda l: l.layer.priority
            )
            return list(sorted_layers)

    def set_layer_priority(self, layer_id: int, priority: int) -> None:
        with self._lock:
            if layer_id in self._layers:
                self._layers[layer_id].layer.priority = priority

    def begin_frame(self) -> None:
        self._frame_active = True
        self._metrics.layers_composited = 0
        self._metrics.blend_operations = 0

    def submit_layers(self) -> None:
        if not self._frame_active:
            return

        with self._lock:
            visible_layers = [l for l in self._layers.values() if l.layer.visible]
            visible_layers.sort(key=lambda l: l.layer.priority)

            self._metrics.active_layers = len(visible_layers)
            self._metrics.layers_composited = len(visible_layers)

            # Count blend operations (layers with non-opaque blending)
            for layer in visible_layers:
                if hasattr(layer.layer.config, 'blend_mode'):
                    if layer.layer.config.blend_mode != BlendMode.OPAQUE:
                        self._metrics.blend_operations += 1

    def end_frame(self) -> None:
        self._frame_active = False

    def get_metrics(self) -> CompositorMetrics:
        return self._metrics


class ProjectionLayer(NullCompositorLayer):
    """Specialized projection layer with stereo support."""

    def __init__(self, layer: Layer):
        super().__init__(layer)
        self._left_texture = 0
        self._right_texture = 0
        self._left_depth = 0
        self._right_depth = 0

    def set_eye_textures(self, left_texture: int, right_texture: int,
                         left_depth: int = 0, right_depth: int = 0) -> None:
        """Set per-eye textures for stereo rendering."""
        self._left_texture = left_texture
        self._right_texture = right_texture
        self._left_depth = left_depth
        self._right_depth = right_depth

    @property
    def left_texture(self) -> int:
        return self._left_texture

    @property
    def right_texture(self) -> int:
        return self._right_texture


class QuadLayer(NullCompositorLayer):
    """Specialized quad layer for UI panels."""

    def __init__(self, layer: Layer):
        super().__init__(layer)

    def set_pose(self, pose: LayerPose) -> None:
        """Update quad pose."""
        if isinstance(self._layer.config, QuadLayerConfig):
            self._layer.config.pose = pose

    def set_size(self, width: float, height: float) -> None:
        """Update quad size in meters."""
        if isinstance(self._layer.config, QuadLayerConfig):
            self._layer.config.size = (width, height)


class CylinderLayer(NullCompositorLayer):
    """Specialized cylinder layer for curved UI."""

    def __init__(self, layer: Layer):
        super().__init__(layer)

    def set_pose(self, pose: LayerPose) -> None:
        """Update cylinder pose."""
        if isinstance(self._layer.config, CylinderLayerConfig):
            self._layer.config.pose = pose

    def set_curvature(self, radius: float, angle: float) -> None:
        """Update cylinder curvature.

        Args:
            radius: Distance from center axis
            angle: Central angle in radians
        """
        if isinstance(self._layer.config, CylinderLayerConfig):
            self._layer.config.radius = radius
            self._layer.config.central_angle = angle


def create_compositor(config: Optional[CompositorConfig] = None) -> XRCompositor:
    """Factory function to create XR compositor.

    Args:
        config: Compositor configuration

    Returns:
        Configured compositor instance
    """
    return NullXRCompositor(config or CompositorConfig())
