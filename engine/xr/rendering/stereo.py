"""Stereo rendering for XR displays.

Supports multiple rendering methods:
- Multi-View: Single draw call, geometry shader replicates
- Instanced Stereo: Single draw call with instancing
- Sequential: Traditional two-pass rendering

Also handles projection types and IPD management.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Tuple, TYPE_CHECKING
import math
import threading

from engine.xr.utils.math_utils import multiply_quaternions

if TYPE_CHECKING:
    from ...platform.rhi.resources import Texture, TextureDesc


class StereoMethod(Enum):
    """Stereo rendering method."""
    MULTI_VIEW = auto()      # OVR_multiview2 - single draw, geometry replication
    INSTANCED = auto()       # Instanced stereo - single draw with gl_InstanceID
    SEQUENTIAL = auto()      # Two separate render passes


class ProjectionType(Enum):
    """Stereo projection type."""
    SYMMETRIC = auto()       # Standard symmetric frustum
    CANTED = auto()          # Displays angled toward each other (e.g., Index)
    ASYMMETRIC = auto()      # Per-eye asymmetric frustum (most accurate)


class IPDMode(Enum):
    """Inter-pupillary distance handling mode."""
    HARDWARE = auto()        # Physical lens adjustment
    SOFTWARE = auto()        # View matrix offset only
    WORLD_SCALE = auto()     # Scale world to match user IPD


class EyeIndex(Enum):
    """Eye identifier."""
    LEFT = 0
    RIGHT = 1


@dataclass
class ViewFrustum:
    """Asymmetric view frustum angles in radians."""
    left: float = -0.785398    # -45 degrees
    right: float = 0.785398    # +45 degrees
    top: float = 0.785398      # +45 degrees
    bottom: float = -0.785398  # -45 degrees


@dataclass
class EyeView:
    """Per-eye view configuration."""
    eye: EyeIndex
    position_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_offset: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # quaternion
    frustum: ViewFrustum = field(default_factory=ViewFrustum)
    viewport_x: int = 0
    viewport_y: int = 0
    viewport_width: int = 1920
    viewport_height: int = 2160


@dataclass
class StereoConfig:
    """Stereo rendering configuration."""
    method: StereoMethod = StereoMethod.MULTI_VIEW
    projection_type: ProjectionType = ProjectionType.ASYMMETRIC
    ipd_mode: IPDMode = IPDMode.SOFTWARE
    ipd_meters: float = 0.063  # 63mm default IPD
    world_scale: float = 1.0
    near_plane: float = 0.1
    far_plane: float = 1000.0
    canting_angle: float = 0.0  # Radians, for canted displays


@dataclass
class StereoRenderTarget:
    """Stereo render target descriptor."""
    width: int = 1920
    height: int = 2160
    array_layers: int = 2  # One per eye
    format: str = "RGBA8_UNORM"
    depth_format: str = "D32_FLOAT"
    sample_count: int = 1


class StereoRenderer(ABC):
    """Abstract stereo renderer interface."""

    @property
    @abstractmethod
    def config(self) -> StereoConfig:
        """Get current stereo configuration."""
        pass

    @abstractmethod
    def configure(self, config: StereoConfig) -> None:
        """Update stereo configuration."""
        pass

    @abstractmethod
    def get_eye_view(self, eye: EyeIndex) -> EyeView:
        """Get view configuration for specified eye."""
        pass

    @abstractmethod
    def get_view_matrix(self, eye: EyeIndex, head_position: Tuple[float, float, float],
                        head_orientation: Tuple[float, float, float, float]) -> List[float]:
        """Calculate view matrix for specified eye.

        Args:
            eye: Which eye to calculate for
            head_position: World space head position (x, y, z)
            head_orientation: Head orientation quaternion (x, y, z, w)

        Returns:
            4x4 view matrix as 16-element list (column-major)
        """
        pass

    @abstractmethod
    def get_projection_matrix(self, eye: EyeIndex) -> List[float]:
        """Calculate projection matrix for specified eye.

        Args:
            eye: Which eye to calculate for

        Returns:
            4x4 projection matrix as 16-element list (column-major)
        """
        pass

    @abstractmethod
    def begin_frame(self) -> None:
        """Begin stereo frame rendering."""
        pass

    @abstractmethod
    def begin_eye(self, eye: EyeIndex) -> None:
        """Begin rendering for specified eye."""
        pass

    @abstractmethod
    def end_eye(self, eye: EyeIndex) -> None:
        """End rendering for specified eye."""
        pass

    @abstractmethod
    def end_frame(self) -> None:
        """End stereo frame rendering."""
        pass


class MultiViewStereoRenderer(StereoRenderer):
    """Multi-view stereo rendering using OVR_multiview2.

    Renders both eyes in a single draw call using geometry shader
    or hardware multi-view extension.
    """

    def __init__(self, config: Optional[StereoConfig] = None):
        """Initialize multi-view stereo renderer.

        Args:
            config: Initial stereo configuration
        """
        self._config = config or StereoConfig(method=StereoMethod.MULTI_VIEW)
        self._eye_views = self._calculate_eye_views()
        self._frame_active = False
        self._lock = threading.Lock()

    @property
    def config(self) -> StereoConfig:
        """Get current stereo configuration."""
        return self._config

    def configure(self, config: StereoConfig) -> None:
        """Update stereo configuration."""
        with self._lock:
            self._config = config
            self._config.method = StereoMethod.MULTI_VIEW
            self._eye_views = self._calculate_eye_views()

    def get_eye_view(self, eye: EyeIndex) -> EyeView:
        """Get view configuration for specified eye."""
        return self._eye_views[eye.value]

    def get_view_matrix(self, eye: EyeIndex, head_position: Tuple[float, float, float],
                        head_orientation: Tuple[float, float, float, float]) -> List[float]:
        """Calculate view matrix for specified eye."""
        eye_view = self._eye_views[eye.value]

        # Apply eye offset to head position
        eye_offset = self._apply_rotation_to_vector(
            head_orientation, eye_view.position_offset
        )
        eye_position = (
            head_position[0] + eye_offset[0],
            head_position[1] + eye_offset[1],
            head_position[2] + eye_offset[2]
        )

        # Apply canting if configured
        eye_orientation = head_orientation
        if self._config.projection_type == ProjectionType.CANTED:
            canting = self._config.canting_angle
            if eye == EyeIndex.RIGHT:
                canting = -canting
            eye_orientation = self._apply_y_rotation(head_orientation, canting)

        return self._create_view_matrix(eye_position, eye_orientation)

    def get_projection_matrix(self, eye: EyeIndex) -> List[float]:
        """Calculate projection matrix for specified eye."""
        eye_view = self._eye_views[eye.value]
        frustum = eye_view.frustum

        return self._create_asymmetric_projection(
            frustum.left, frustum.right,
            frustum.bottom, frustum.top,
            self._config.near_plane, self._config.far_plane
        )

    def begin_frame(self) -> None:
        """Begin stereo frame rendering."""
        self._frame_active = True

    def begin_eye(self, eye: EyeIndex) -> None:
        """Begin rendering for specified eye.

        In multi-view mode, this is mostly a no-op since both eyes
        render simultaneously.
        """
        pass

    def end_eye(self, eye: EyeIndex) -> None:
        """End rendering for specified eye."""
        pass

    def end_frame(self) -> None:
        """End stereo frame rendering."""
        self._frame_active = False

    def _calculate_eye_views(self) -> List[EyeView]:
        """Calculate eye views from current configuration."""
        half_ipd = self._config.ipd_meters * self._config.world_scale / 2.0

        left_view = EyeView(
            eye=EyeIndex.LEFT,
            position_offset=(-half_ipd, 0.0, 0.0)
        )
        right_view = EyeView(
            eye=EyeIndex.RIGHT,
            position_offset=(half_ipd, 0.0, 0.0)
        )

        return [left_view, right_view]

    def _apply_rotation_to_vector(
        self, quat: Tuple[float, float, float, float],
        vec: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """Apply quaternion rotation to vector."""
        qx, qy, qz, qw = quat
        vx, vy, vz = vec

        # Quaternion rotation: q * v * q^-1
        # Optimized formula
        ix = qw * vx + qy * vz - qz * vy
        iy = qw * vy + qz * vx - qx * vz
        iz = qw * vz + qx * vy - qy * vx
        iw = -qx * vx - qy * vy - qz * vz

        rx = ix * qw + iw * -qx + iy * -qz - iz * -qy
        ry = iy * qw + iw * -qy + iz * -qx - ix * -qz
        rz = iz * qw + iw * -qz + ix * -qy - iy * -qx

        return (rx, ry, rz)

    def _apply_y_rotation(
        self, quat: Tuple[float, float, float, float], angle: float
    ) -> Tuple[float, float, float, float]:
        """Apply Y-axis rotation to quaternion."""
        half_angle = angle / 2.0
        sin_half = math.sin(half_angle)
        cos_half = math.cos(half_angle)

        # Y-axis rotation quaternion
        ry = (0.0, sin_half, 0.0, cos_half)

        # Multiply quaternions using shared utility
        return multiply_quaternions(quat, ry)

    def _create_view_matrix(
        self, position: Tuple[float, float, float],
        orientation: Tuple[float, float, float, float]
    ) -> List[float]:
        """Create view matrix from position and orientation."""
        # Convert quaternion to rotation matrix
        qx, qy, qz, qw = orientation

        # Rotation matrix elements
        xx = qx * qx
        yy = qy * qy
        zz = qz * qz
        xy = qx * qy
        xz = qx * qz
        yz = qy * qz
        wx = qw * qx
        wy = qw * qy
        wz = qw * qz

        r00 = 1.0 - 2.0 * (yy + zz)
        r01 = 2.0 * (xy - wz)
        r02 = 2.0 * (xz + wy)

        r10 = 2.0 * (xy + wz)
        r11 = 1.0 - 2.0 * (xx + zz)
        r12 = 2.0 * (yz - wx)

        r20 = 2.0 * (xz - wy)
        r21 = 2.0 * (yz + wx)
        r22 = 1.0 - 2.0 * (xx + yy)

        # View matrix is inverse of camera transform
        # Transpose rotation and negate position
        px, py, pz = position

        # Translation in view space
        tx = -(r00 * px + r10 * py + r20 * pz)
        ty = -(r01 * px + r11 * py + r21 * pz)
        tz = -(r02 * px + r12 * py + r22 * pz)

        # Column-major 4x4 matrix
        return [
            r00, r01, r02, 0.0,
            r10, r11, r12, 0.0,
            r20, r21, r22, 0.0,
            tx, ty, tz, 1.0
        ]

    def _create_asymmetric_projection(
        self, left: float, right: float,
        bottom: float, top: float,
        near: float, far: float
    ) -> List[float]:
        """Create asymmetric projection matrix from frustum angles."""
        tan_left = math.tan(left)
        tan_right = math.tan(right)
        tan_bottom = math.tan(bottom)
        tan_top = math.tan(top)

        x = 2.0 / (tan_right - tan_left)
        y = 2.0 / (tan_top - tan_bottom)

        a = (tan_right + tan_left) / (tan_right - tan_left)
        b = (tan_top + tan_bottom) / (tan_top - tan_bottom)
        c = -(far + near) / (far - near)
        d = -(2.0 * far * near) / (far - near)

        # Column-major 4x4 matrix
        return [
            x, 0.0, 0.0, 0.0,
            0.0, y, 0.0, 0.0,
            a, b, c, -1.0,
            0.0, 0.0, d, 0.0
        ]


class InstancedStereoRenderer(StereoRenderer):
    """Instanced stereo rendering.

    Uses GPU instancing to render both eyes with a single draw call,
    selecting view/projection matrices via gl_InstanceID.
    """

    def __init__(self, config: Optional[StereoConfig] = None):
        """Initialize instanced stereo renderer."""
        self._config = config or StereoConfig(method=StereoMethod.INSTANCED)
        self._eye_views = self._calculate_eye_views()
        self._current_eye: Optional[EyeIndex] = None
        self._lock = threading.Lock()

    @property
    def config(self) -> StereoConfig:
        return self._config

    def configure(self, config: StereoConfig) -> None:
        with self._lock:
            self._config = config
            self._config.method = StereoMethod.INSTANCED
            self._eye_views = self._calculate_eye_views()

    def get_eye_view(self, eye: EyeIndex) -> EyeView:
        return self._eye_views[eye.value]

    def get_view_matrix(self, eye: EyeIndex, head_position: Tuple[float, float, float],
                        head_orientation: Tuple[float, float, float, float]) -> List[float]:
        # Delegate to shared implementation
        renderer = MultiViewStereoRenderer(self._config)
        renderer._eye_views = self._eye_views
        return renderer.get_view_matrix(eye, head_position, head_orientation)

    def get_projection_matrix(self, eye: EyeIndex) -> List[float]:
        renderer = MultiViewStereoRenderer(self._config)
        renderer._eye_views = self._eye_views
        return renderer.get_projection_matrix(eye)

    def begin_frame(self) -> None:
        pass

    def begin_eye(self, eye: EyeIndex) -> None:
        self._current_eye = eye

    def end_eye(self, eye: EyeIndex) -> None:
        self._current_eye = None

    def end_frame(self) -> None:
        pass

    def _calculate_eye_views(self) -> List[EyeView]:
        half_ipd = self._config.ipd_meters * self._config.world_scale / 2.0
        return [
            EyeView(eye=EyeIndex.LEFT, position_offset=(-half_ipd, 0.0, 0.0)),
            EyeView(eye=EyeIndex.RIGHT, position_offset=(half_ipd, 0.0, 0.0))
        ]


class SequentialStereoRenderer(StereoRenderer):
    """Sequential (two-pass) stereo rendering.

    Traditional approach rendering left eye completely, then right eye.
    Most compatible but highest draw call overhead.
    """

    def __init__(self, config: Optional[StereoConfig] = None):
        """Initialize sequential stereo renderer."""
        self._config = config or StereoConfig(method=StereoMethod.SEQUENTIAL)
        self._eye_views = self._calculate_eye_views()
        self._current_eye: Optional[EyeIndex] = None
        self._lock = threading.Lock()

    @property
    def config(self) -> StereoConfig:
        return self._config

    def configure(self, config: StereoConfig) -> None:
        with self._lock:
            self._config = config
            self._config.method = StereoMethod.SEQUENTIAL
            self._eye_views = self._calculate_eye_views()

    def get_eye_view(self, eye: EyeIndex) -> EyeView:
        return self._eye_views[eye.value]

    def get_view_matrix(self, eye: EyeIndex, head_position: Tuple[float, float, float],
                        head_orientation: Tuple[float, float, float, float]) -> List[float]:
        renderer = MultiViewStereoRenderer(self._config)
        renderer._eye_views = self._eye_views
        return renderer.get_view_matrix(eye, head_position, head_orientation)

    def get_projection_matrix(self, eye: EyeIndex) -> List[float]:
        renderer = MultiViewStereoRenderer(self._config)
        renderer._eye_views = self._eye_views
        return renderer.get_projection_matrix(eye)

    def begin_frame(self) -> None:
        pass

    def begin_eye(self, eye: EyeIndex) -> None:
        self._current_eye = eye

    def end_eye(self, eye: EyeIndex) -> None:
        self._current_eye = None

    def end_frame(self) -> None:
        pass

    def _calculate_eye_views(self) -> List[EyeView]:
        half_ipd = self._config.ipd_meters * self._config.world_scale / 2.0
        return [
            EyeView(eye=EyeIndex.LEFT, position_offset=(-half_ipd, 0.0, 0.0)),
            EyeView(eye=EyeIndex.RIGHT, position_offset=(half_ipd, 0.0, 0.0))
        ]


def create_stereo_renderer(config: Optional[StereoConfig] = None) -> StereoRenderer:
    """Factory function to create appropriate stereo renderer.

    Args:
        config: Stereo configuration (determines renderer type)

    Returns:
        Configured stereo renderer instance
    """
    if config is None:
        config = StereoConfig()

    if config.method == StereoMethod.MULTI_VIEW:
        return MultiViewStereoRenderer(config)
    elif config.method == StereoMethod.INSTANCED:
        return InstancedStereoRenderer(config)
    else:
        return SequentialStereoRenderer(config)
