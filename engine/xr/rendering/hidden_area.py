"""Hidden area mesh optimization for XR displays.

Manages the hidden area mesh (HAM) - regions of the display that are
not visible through the HMD optics. Rendering to these areas is wasted
work that can be skipped.

Supports:
- Per-eye hidden area meshes
- Stencil-based masking
- Early-z rejection
- Integration with foveated rendering
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Tuple
import math
import threading


class HiddenAreaType(Enum):
    """Type of hidden area mesh."""
    NONE = auto()          # No hidden area mask
    STENCIL = auto()       # Use stencil buffer
    DEPTH = auto()         # Write to depth buffer (early-z)
    MESH = auto()          # Custom mesh geometry
    COMBINED = auto()      # Stencil + depth


class MeshFormat(Enum):
    """Hidden area mesh data format."""
    TRIANGLE_LIST = auto()  # List of triangles
    TRIANGLE_FAN = auto()   # Fan from center
    LINE_LOOP = auto()      # Outline only


@dataclass
class Vertex2D:
    """2D vertex in normalized device coordinates."""
    x: float  # -1 to 1
    y: float  # -1 to 1


@dataclass
class Triangle:
    """Triangle defined by three vertices."""
    v0: Vertex2D
    v1: Vertex2D
    v2: Vertex2D


@dataclass
class HiddenAreaMeshData:
    """Hidden area mesh geometry data."""
    vertices: List[Vertex2D] = field(default_factory=list)
    indices: List[int] = field(default_factory=list)
    format: MeshFormat = MeshFormat.TRIANGLE_LIST
    vertex_count: int = 0
    triangle_count: int = 0


@dataclass
class HiddenAreaConfig:
    """Hidden area mask configuration."""
    type: HiddenAreaType = HiddenAreaType.STENCIL
    enabled: bool = True
    stencil_reference: int = 0xFF
    stencil_mask: int = 0xFF
    depth_value: float = 1.0  # Far plane
    invert_mask: bool = False  # If true, hidden area is visible area
    padding_pixels: int = 0    # Expand mask boundary


@dataclass
class HiddenAreaMetrics:
    """Hidden area optimization metrics."""
    pixels_hidden_left: int = 0
    pixels_hidden_right: int = 0
    pixel_savings_percent: float = 0.0
    triangles_per_eye: int = 0


class HiddenAreaMask(ABC):
    """Abstract hidden area mask interface."""

    @property
    @abstractmethod
    def config(self) -> HiddenAreaConfig:
        """Get mask configuration."""
        pass

    @abstractmethod
    def configure(self, config: HiddenAreaConfig) -> None:
        """Update mask configuration."""
        pass

    @abstractmethod
    def set_mesh_data(self, eye_index: int, mesh: HiddenAreaMeshData) -> None:
        """Set hidden area mesh for an eye.

        Args:
            eye_index: 0 for left, 1 for right
            mesh: Mesh geometry data
        """
        pass

    @abstractmethod
    def get_mesh_data(self, eye_index: int) -> HiddenAreaMeshData:
        """Get hidden area mesh for an eye.

        Args:
            eye_index: 0 for left, 1 for right

        Returns:
            Mesh geometry data
        """
        pass

    @abstractmethod
    def generate_default_mesh(self, eye_index: int, resolution: Tuple[int, int]) -> HiddenAreaMeshData:
        """Generate default hidden area mesh.

        Args:
            eye_index: 0 for left, 1 for right
            resolution: Target resolution (width, height)

        Returns:
            Generated mesh data
        """
        pass

    @abstractmethod
    def is_point_hidden(self, eye_index: int, x: float, y: float) -> bool:
        """Check if a point is in the hidden area.

        Args:
            eye_index: 0 for left, 1 for right
            x: Normalized x coordinate (-1 to 1)
            y: Normalized y coordinate (-1 to 1)

        Returns:
            True if point is hidden
        """
        pass

    @abstractmethod
    def get_visible_area_ratio(self, eye_index: int) -> float:
        """Get ratio of visible to total pixels.

        Args:
            eye_index: 0 for left, 1 for right

        Returns:
            Ratio (0 to 1)
        """
        pass

    @abstractmethod
    def begin_mask(self, eye_index: int) -> None:
        """Begin hidden area masking for an eye."""
        pass

    @abstractmethod
    def end_mask(self, eye_index: int) -> None:
        """End hidden area masking."""
        pass

    @abstractmethod
    def get_metrics(self) -> HiddenAreaMetrics:
        """Get optimization metrics."""
        pass


class NullHiddenAreaMask(HiddenAreaMask):
    """Null implementation of hidden area mask for testing."""

    def __init__(self, config: Optional[HiddenAreaConfig] = None):
        self._config = config or HiddenAreaConfig()
        self._meshes: List[HiddenAreaMeshData] = [
            HiddenAreaMeshData(),
            HiddenAreaMeshData()
        ]
        self._metrics = HiddenAreaMetrics()
        self._lock = threading.Lock()

    @property
    def config(self) -> HiddenAreaConfig:
        return self._config

    def configure(self, config: HiddenAreaConfig) -> None:
        with self._lock:
            self._config = config

    def set_mesh_data(self, eye_index: int, mesh: HiddenAreaMeshData) -> None:
        if 0 <= eye_index < 2:
            with self._lock:
                self._meshes[eye_index] = mesh
                self._update_metrics()

    def get_mesh_data(self, eye_index: int) -> HiddenAreaMeshData:
        if 0 <= eye_index < 2:
            return self._meshes[eye_index]
        return HiddenAreaMeshData()

    def generate_default_mesh(self, eye_index: int, resolution: Tuple[int, int]) -> HiddenAreaMeshData:
        """Generate a typical VR hidden area mesh.

        Creates a mesh that masks the corners of the screen that are
        typically outside the visible lens area.
        """
        # Generate corner triangles for typical VR lens shape
        # This is an approximation - real meshes come from the runtime
        mesh = HiddenAreaMeshData(format=MeshFormat.TRIANGLE_LIST)

        # Corner radius (normalized) - typical VR lenses hide ~10-15% corners
        r = 0.15

        # Top-left corner
        self._add_corner_triangles(mesh, -1.0, 1.0, r, quadrant=0)

        # Top-right corner
        self._add_corner_triangles(mesh, 1.0, 1.0, r, quadrant=1)

        # Bottom-right corner
        self._add_corner_triangles(mesh, 1.0, -1.0, r, quadrant=2)

        # Bottom-left corner
        self._add_corner_triangles(mesh, -1.0, -1.0, r, quadrant=3)

        mesh.vertex_count = len(mesh.vertices)
        mesh.triangle_count = len(mesh.indices) // 3

        return mesh

    def _add_corner_triangles(self, mesh: HiddenAreaMeshData, cx: float, cy: float,
                               radius: float, quadrant: int, segments: int = 8) -> None:
        """Add triangles for one corner."""
        base_index = len(mesh.vertices)

        # Corner vertex
        mesh.vertices.append(Vertex2D(cx, cy))

        # Arc vertices
        start_angle = quadrant * (math.pi / 2) + math.pi
        for i in range(segments + 1):
            angle = start_angle + (math.pi / 2) * i / segments
            x = cx + radius * math.cos(angle) * (1 if cx < 0 else -1)
            y = cy + radius * math.sin(angle) * (1 if cy > 0 else -1)
            mesh.vertices.append(Vertex2D(x, y))

        # Create triangles
        for i in range(segments):
            mesh.indices.extend([
                base_index,
                base_index + i + 1,
                base_index + i + 2
            ])

    def is_point_hidden(self, eye_index: int, x: float, y: float) -> bool:
        """Check if point is inside any hidden area triangle."""
        if not self._config.enabled:
            return False

        mesh = self._meshes[eye_index] if 0 <= eye_index < 2 else None
        if not mesh or not mesh.vertices:
            return False

        # Check against each triangle
        for i in range(0, len(mesh.indices), 3):
            if i + 2 < len(mesh.indices):
                v0 = mesh.vertices[mesh.indices[i]]
                v1 = mesh.vertices[mesh.indices[i + 1]]
                v2 = mesh.vertices[mesh.indices[i + 2]]

                if self._point_in_triangle(x, y, v0, v1, v2):
                    return not self._config.invert_mask

        return self._config.invert_mask

    def _point_in_triangle(self, px: float, py: float,
                           v0: Vertex2D, v1: Vertex2D, v2: Vertex2D) -> bool:
        """Check if point is inside triangle using barycentric coordinates."""
        def sign(p1: Vertex2D, p2: Vertex2D, p3: Vertex2D) -> float:
            return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)

        p = Vertex2D(px, py)

        d1 = sign(p, v0, v1)
        d2 = sign(p, v1, v2)
        d3 = sign(p, v2, v0)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)

    def get_visible_area_ratio(self, eye_index: int) -> float:
        """Calculate visible area ratio by sampling."""
        if not self._config.enabled:
            return 1.0

        mesh = self._meshes[eye_index] if 0 <= eye_index < 2 else None
        if not mesh or not mesh.vertices:
            return 1.0

        # Sample grid
        samples = 32
        visible = 0
        total = samples * samples

        for yi in range(samples):
            for xi in range(samples):
                x = -1.0 + 2.0 * xi / (samples - 1)
                y = -1.0 + 2.0 * yi / (samples - 1)

                if not self.is_point_hidden(eye_index, x, y):
                    visible += 1

        return visible / total

    def begin_mask(self, eye_index: int) -> None:
        """Begin masking - in null impl, just validates state."""
        pass

    def end_mask(self, eye_index: int) -> None:
        """End masking."""
        pass

    def get_metrics(self) -> HiddenAreaMetrics:
        return self._metrics

    def _update_metrics(self) -> None:
        """Update optimization metrics."""
        # Calculate triangle counts
        for i, mesh in enumerate(self._meshes):
            if mesh.indices:
                self._metrics.triangles_per_eye = len(mesh.indices) // 3

        # Calculate pixel savings (approximate)
        left_ratio = self.get_visible_area_ratio(0)
        right_ratio = self.get_visible_area_ratio(1)

        avg_visible = (left_ratio + right_ratio) / 2
        self._metrics.pixel_savings_percent = (1.0 - avg_visible) * 100


class StencilHiddenAreaMask(NullHiddenAreaMask):
    """Hidden area mask using stencil buffer."""

    def __init__(self, config: Optional[HiddenAreaConfig] = None):
        if config is None:
            config = HiddenAreaConfig(type=HiddenAreaType.STENCIL)
        else:
            config.type = HiddenAreaType.STENCIL
        super().__init__(config)
        self._stencil_active = False

    def begin_mask(self, eye_index: int) -> None:
        """Set up stencil state for masking."""
        if not self._config.enabled:
            return

        self._stencil_active = True
        # In a real implementation, this would:
        # 1. Enable stencil testing
        # 2. Clear stencil to reference value
        # 3. Render hidden area mesh writing to stencil
        # 4. Set stencil test to reject pixels matching reference

    def end_mask(self, eye_index: int) -> None:
        """Restore stencil state."""
        self._stencil_active = False


class DepthHiddenAreaMask(NullHiddenAreaMask):
    """Hidden area mask using depth buffer for early-z rejection."""

    def __init__(self, config: Optional[HiddenAreaConfig] = None):
        if config is None:
            config = HiddenAreaConfig(type=HiddenAreaType.DEPTH)
        else:
            config.type = HiddenAreaType.DEPTH
        super().__init__(config)

    def begin_mask(self, eye_index: int) -> None:
        """Write depth values to hidden areas."""
        if not self._config.enabled:
            return

        # In a real implementation, this would:
        # 1. Disable color writes
        # 2. Render hidden area mesh at far depth
        # 3. Re-enable color writes
        # Early-z hardware will then skip hidden fragments

    def end_mask(self, eye_index: int) -> None:
        """Restore depth state."""
        pass


class CombinedHiddenAreaMask(NullHiddenAreaMask):
    """Combined stencil + depth hidden area mask."""

    def __init__(self, config: Optional[HiddenAreaConfig] = None):
        if config is None:
            config = HiddenAreaConfig(type=HiddenAreaType.COMBINED)
        else:
            config.type = HiddenAreaType.COMBINED
        super().__init__(config)
        self._stencil_mask = StencilHiddenAreaMask(config)
        self._depth_mask = DepthHiddenAreaMask(config)

    def set_mesh_data(self, eye_index: int, mesh: HiddenAreaMeshData) -> None:
        super().set_mesh_data(eye_index, mesh)
        self._stencil_mask.set_mesh_data(eye_index, mesh)
        self._depth_mask.set_mesh_data(eye_index, mesh)

    def begin_mask(self, eye_index: int) -> None:
        """Apply both masking techniques."""
        self._stencil_mask.begin_mask(eye_index)
        self._depth_mask.begin_mask(eye_index)

    def end_mask(self, eye_index: int) -> None:
        self._stencil_mask.end_mask(eye_index)
        self._depth_mask.end_mask(eye_index)


def create_hidden_area_mask(config: Optional[HiddenAreaConfig] = None) -> HiddenAreaMask:
    """Factory function to create hidden area mask.

    Args:
        config: Mask configuration

    Returns:
        Configured hidden area mask instance
    """
    if config is None:
        config = HiddenAreaConfig()

    if not config.enabled or config.type == HiddenAreaType.NONE:
        return NullHiddenAreaMask(HiddenAreaConfig(enabled=False))

    if config.type == HiddenAreaType.STENCIL:
        return StencilHiddenAreaMask(config)
    elif config.type == HiddenAreaType.DEPTH:
        return DepthHiddenAreaMask(config)
    elif config.type == HiddenAreaType.COMBINED:
        return CombinedHiddenAreaMask(config)
    else:
        return NullHiddenAreaMask(config)
