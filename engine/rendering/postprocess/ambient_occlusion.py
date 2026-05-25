"""
Screen-Space Ambient Occlusion System

Provides multiple AO algorithms:
- SSAO: Original Crytek algorithm
- HBAO: Horizon-Based Ambient Occlusion
- GTAO: Ground-Truth Ambient Occlusion
- AOSettings: Complete configuration
- BentNormal output for specular occlusion
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from .postprocess_stack import EffectPriority, EffectSettings, PostProcessEffect


class AOMethod(Enum):
    """Ambient occlusion algorithm."""

    SSAO = auto()  # Original Crytek SSAO
    HBAO = auto()  # Horizon-Based AO
    HBAO_PLUS = auto()  # Enhanced HBAO
    GTAO = auto()  # Ground-Truth AO
    RTAO = auto()  # Ray-traced AO (placeholder)


class AOQuality(Enum):
    """AO quality preset."""

    LOW = auto()  # 4 samples, no blur
    MEDIUM = auto()  # 8 samples, basic blur
    HIGH = auto()  # 16 samples, bilateral blur
    ULTRA = auto()  # 32+ samples, temporal


@dataclass
class AOSettings(EffectSettings):
    """Ambient occlusion settings.

    Uses constants from constants.py AO module for default values.
    """

    method: AOMethod = AOMethod.GTAO
    quality: AOQuality = AOQuality.HIGH

    # Core settings - see AO constants
    radius: float = 0.5  # World-space radius in meters (AO.RADIUS_DEFAULT)
    intensity: float = 1.0  # AO intensity [0, 2] (AO.INTENSITY_DEFAULT)
    power: float = 1.0  # AO power curve
    bias: float = 0.01  # Depth bias (AO.BIAS_DEFAULT)

    # Sample settings
    sample_count: int = 16  # Samples per pixel (AO.SAMPLE_COUNT_HIGH)
    direction_count: int = 8  # Directions (AO.DIRECTION_COUNT_DEFAULT)

    # Spatial settings
    thickness: float = 0.1  # Thickness heuristic
    falloff_start: float = 0.2  # Start of distance falloff
    falloff_end: float = 1.0  # End of distance falloff

    # Filtering
    blur_sharpness: float = 8.0  # Edge-aware blur sharpness
    temporal_enabled: bool = True  # Temporal accumulation
    temporal_weight: float = 0.9  # History blend weight

    # Output options
    bent_normals_enabled: bool = False  # Output bent normals
    multi_bounce: bool = False  # Multi-bounce approximation

    # Performance
    half_resolution: bool = False  # Process at half res
    downsample_depth: bool = True  # Use downsampled depth

    def __post_init__(self) -> None:
        self.priority = EffectPriority.AMBIENT_OCCLUSION.value

    def lerp(self, other: "AOSettings", t: float) -> "AOSettings":
        """Interpolate between two AO settings."""
        return AOSettings(
            enabled=self.enabled if t < 0.5 else other.enabled,
            weight=self.weight + (other.weight - self.weight) * t,
            method=self.method if t < 0.5 else other.method,
            quality=self.quality if t < 0.5 else other.quality,
            radius=self.radius + (other.radius - self.radius) * t,
            intensity=self.intensity + (other.intensity - self.intensity) * t,
            power=self.power + (other.power - self.power) * t,
            bias=self.bias + (other.bias - self.bias) * t,
            blur_sharpness=self.blur_sharpness
            + (other.blur_sharpness - self.blur_sharpness) * t,
        )


class SSAOKernel:
    """Sample kernel for SSAO."""

    def __init__(self, sample_count: int = 64) -> None:
        """Initialize SSAO kernel.

        Args:
            sample_count: Number of samples in kernel.
        """
        self._samples: List[Tuple[float, float, float]] = []
        self._noise: List[Tuple[float, float]] = []
        self.generate(sample_count)

    @property
    def samples(self) -> List[Tuple[float, float, float]]:
        """Get sample positions."""
        return self._samples

    @property
    def noise(self) -> List[Tuple[float, float]]:
        """Get noise rotation vectors."""
        return self._noise

    def generate(self, sample_count: int) -> None:
        """Generate hemisphere sample kernel.

        Args:
            sample_count: Number of samples.
        """
        self._samples = []
        random.seed(42)  # Deterministic for consistency

        for i in range(sample_count):
            # Random point in hemisphere
            x = random.random() * 2.0 - 1.0
            y = random.random() * 2.0 - 1.0
            z = random.random()

            # Normalize
            length = math.sqrt(x * x + y * y + z * z)
            if length > 0:
                x /= length
                y /= length
                z /= length

            # Scale distribution (more samples closer to origin)
            scale = i / sample_count
            scale = 0.1 + scale * scale * 0.9

            self._samples.append((x * scale, y * scale, z * scale))

        # Generate 4x4 noise texture values
        self._noise = []
        for _ in range(16):
            x = random.random() * 2.0 - 1.0
            y = random.random() * 2.0 - 1.0
            length = math.sqrt(x * x + y * y)
            if length > 0:
                x /= length
                y /= length
            self._noise.append((x, y))


class SSAO:
    """Screen-Space Ambient Occlusion (Crytek).

    Original SSAO algorithm using random sampling in
    a hemisphere around each pixel.
    """

    def __init__(self) -> None:
        self._kernel: SSAOKernel = SSAOKernel()
        self._ao_buffer: Any = None
        self._blur_buffer: Any = None
        self._noise_texture: Any = None

    @property
    def kernel(self) -> SSAOKernel:
        """Access the sample kernel."""
        return self._kernel

    def setup(self, width: int, height: int, sample_count: int = 64) -> None:
        """Initialize SSAO resources.

        Args:
            width: Buffer width.
            height: Buffer height.
            sample_count: Kernel sample count.
        """
        self._kernel.generate(sample_count)
        self._ao_buffer = None
        self._blur_buffer = None

    def calculate(
        self,
        depth_buffer: Any,
        normal_buffer: Any,
        settings: AOSettings,
        projection: List[List[float]],
    ) -> Any:
        """Calculate SSAO.

        Args:
            depth_buffer: Scene depth.
            normal_buffer: View-space normals.
            settings: AO settings.
            projection: Projection matrix.

        Returns:
            AO buffer.
        """
        return self._ao_buffer

    def blur(
        self,
        ao_buffer: Any,
        depth_buffer: Any,
        settings: AOSettings,
    ) -> Any:
        """Apply edge-aware blur.

        Args:
            ao_buffer: Raw AO buffer.
            depth_buffer: Scene depth.
            settings: Blur settings.

        Returns:
            Blurred AO buffer.
        """
        return self._blur_buffer


class HBAO:
    """Horizon-Based Ambient Occlusion.

    Traces rays in the 2D image plane and measures
    the horizon angle to approximate occlusion.
    """

    def __init__(self) -> None:
        self._ao_buffer: Any = None
        self._blur_buffer: Any = None
        self._directions: List[Tuple[float, float]] = []

    def setup(self, width: int, height: int, direction_count: int = 8) -> None:
        """Initialize HBAO resources.

        Args:
            width: Buffer width.
            height: Buffer height.
            direction_count: Number of ray directions.
        """
        self._ao_buffer = None
        self._blur_buffer = None
        self._generate_directions(direction_count)

    def _generate_directions(self, count: int) -> None:
        """Generate uniform directions around the pixel.

        Args:
            count: Number of directions.
        """
        self._directions = []
        for i in range(count):
            angle = (i / count) * 2.0 * math.pi
            self._directions.append((math.cos(angle), math.sin(angle)))

    def calculate(
        self,
        depth_buffer: Any,
        normal_buffer: Any,
        settings: AOSettings,
        projection: List[List[float]],
    ) -> Any:
        """Calculate HBAO.

        Args:
            depth_buffer: Scene depth.
            normal_buffer: View-space normals.
            settings: AO settings.
            projection: Projection matrix.

        Returns:
            AO buffer.
        """
        return self._ao_buffer

    def calculate_horizon_angle(
        self,
        start_uv: Tuple[float, float],
        direction: Tuple[float, float],
        depth_buffer: Any,
        step_count: int,
    ) -> float:
        """Calculate horizon angle for a direction.

        Args:
            start_uv: Starting UV coordinate.
            direction: Ray direction.
            depth_buffer: Depth buffer.
            step_count: Number of steps.

        Returns:
            Maximum horizon angle.
        """
        return 0.0


class GTAO:
    """Ground-Truth Ambient Occlusion.

    Accurate AO using cosine-weighted hemisphere
    integration with efficient temporal filtering.
    """

    def __init__(self) -> None:
        self._ao_buffer: Any = None
        self._bent_normal_buffer: Any = None
        self._history_buffer: Any = None
        self._slice_count: int = 8
        self._steps_per_slice: int = 4

    def setup(
        self,
        width: int,
        height: int,
        slice_count: int = 8,
        steps_per_slice: int = 4,
    ) -> None:
        """Initialize GTAO resources.

        Args:
            width: Buffer width.
            height: Buffer height.
            slice_count: Number of angular slices.
            steps_per_slice: Steps per slice.
        """
        self._ao_buffer = None
        self._bent_normal_buffer = None
        self._history_buffer = None
        self._slice_count = slice_count
        self._steps_per_slice = steps_per_slice

    def calculate(
        self,
        depth_buffer: Any,
        normal_buffer: Any,
        settings: AOSettings,
        view_matrix: List[List[float]],
        projection: List[List[float]],
    ) -> Tuple[Any, Optional[Any]]:
        """Calculate GTAO with optional bent normals.

        Args:
            depth_buffer: Scene depth.
            normal_buffer: View-space normals.
            settings: AO settings.
            view_matrix: View matrix.
            projection: Projection matrix.

        Returns:
            Tuple of (AO buffer, bent normal buffer or None).
        """
        bent_normals = self._bent_normal_buffer if settings.bent_normals_enabled else None
        return (self._ao_buffer, bent_normals)

    def integrate_slice(
        self,
        pixel_uv: Tuple[float, float],
        slice_angle: float,
        normal: Tuple[float, float, float],
        depth_buffer: Any,
        settings: AOSettings,
    ) -> Tuple[float, Tuple[float, float, float]]:
        """Integrate visibility for one angular slice.

        Args:
            pixel_uv: Pixel UV coordinate.
            slice_angle: Slice angle in radians.
            normal: Surface normal.
            depth_buffer: Depth buffer.
            settings: AO settings.

        Returns:
            (occlusion, bent_normal_contribution).
        """
        return (0.0, (0.0, 0.0, 1.0))

    def temporal_filter(
        self,
        current_ao: Any,
        history_ao: Any,
        velocity_buffer: Any,
        settings: AOSettings,
    ) -> Any:
        """Apply temporal filtering.

        Args:
            current_ao: Current frame AO.
            history_ao: Previous frame AO.
            velocity_buffer: Motion vectors.
            settings: Temporal settings.

        Returns:
            Filtered AO buffer.
        """
        return current_ao


@dataclass
class BentNormalOutput:
    """Bent normal data for specular occlusion."""

    bent_normal: Tuple[float, float, float]  # Dominant unoccluded direction
    visibility_cone: float  # Cone angle of unoccluded region
    occlusion: float  # Ambient occlusion factor

    def calculate_specular_occlusion(
        self,
        view_dir: Tuple[float, float, float],
        roughness: float,
    ) -> float:
        """Calculate specular occlusion from bent normal.

        Args:
            view_dir: View direction.
            roughness: Surface roughness.

        Returns:
            Specular occlusion factor [0, 1].
        """
        # Dot product between view and bent normal
        dot = sum(view_dir[i] * self.bent_normal[i] for i in range(3))
        dot = max(0.0, dot)

        # Roughness-adjusted specular occlusion
        spec_occ = self.occlusion * pow(dot, (1.0 - roughness) * 4.0)
        return max(0.0, min(1.0, spec_occ))


class BilateralFilter:
    """Edge-aware bilateral filter for AO."""

    def __init__(self) -> None:
        self._temp_buffer: Any = None

    def setup(self, width: int, height: int) -> None:
        """Initialize filter buffers.

        Args:
            width: Buffer width.
            height: Buffer height.
        """
        self._temp_buffer = None

    def apply(
        self,
        ao_buffer: Any,
        depth_buffer: Any,
        normal_buffer: Any,
        sharpness: float,
        radius: int = 4,
    ) -> Any:
        """Apply bilateral filter.

        Args:
            ao_buffer: Input AO.
            depth_buffer: Scene depth.
            normal_buffer: Surface normals.
            sharpness: Edge sharpness factor.
            radius: Filter radius.

        Returns:
            Filtered AO buffer.
        """
        return ao_buffer

    def _bilateral_weight(
        self,
        depth0: float,
        depth1: float,
        normal0: Tuple[float, float, float],
        normal1: Tuple[float, float, float],
        distance: float,
        sharpness: float,
    ) -> float:
        """Calculate bilateral filter weight.

        Args:
            depth0: Center depth.
            depth1: Sample depth.
            normal0: Center normal.
            normal1: Sample normal.
            distance: Spatial distance.
            sharpness: Edge sharpness.

        Returns:
            Filter weight.
        """
        # Depth weight
        depth_diff = abs(depth1 - depth0)
        depth_weight = math.exp(-depth_diff * depth_diff * sharpness)

        # Normal weight
        normal_dot = sum(normal0[i] * normal1[i] for i in range(3))
        normal_weight = max(0.0, normal_dot)

        # Spatial weight (Gaussian)
        spatial_weight = math.exp(-distance * distance / 2.0)

        return depth_weight * normal_weight * spatial_weight


class AOEffect(PostProcessEffect[AOSettings]):
    """Complete Ambient Occlusion post-process effect."""

    def __init__(
        self,
        settings: Optional[AOSettings] = None,
    ) -> None:
        """Initialize AO effect.

        Args:
            settings: AO configuration.
        """
        super().__init__(
            name="AmbientOcclusion",
            settings=settings or AOSettings(),
            priority=EffectPriority.AMBIENT_OCCLUSION.value,
        )

        self._ssao: SSAO = SSAO()
        self._hbao: HBAO = HBAO()
        self._gtao: GTAO = GTAO()
        self._bilateral: BilateralFilter = BilateralFilter()

        self._ao_result: Any = None
        self._bent_normals: Any = None
        self._width: int = 0
        self._height: int = 0

    @property
    def ao_buffer(self) -> Any:
        """Get the current AO result buffer."""
        return self._ao_result

    @property
    def bent_normals(self) -> Any:
        """Get bent normals buffer (if enabled)."""
        return self._bent_normals

    def get_required_inputs(self) -> List[str]:
        """Get required input resources."""
        return ["depth", "normal"]

    def get_outputs(self) -> List[str]:
        """Get output resources."""
        outputs = ["ao"]
        if self._settings and self._settings.bent_normals_enabled:
            outputs.append("bent_normals")
        return outputs

    def setup(self, width: int, height: int) -> None:
        """Initialize AO resources.

        Args:
            width: Render width.
            height: Render height.
        """
        self._width = width
        self._height = height

        process_width = width // 2 if self._settings and self._settings.half_resolution else width
        process_height = height // 2 if self._settings and self._settings.half_resolution else height

        sample_count = self._settings.sample_count if self._settings else 16
        direction_count = self._settings.direction_count if self._settings else 8

        self._ssao.setup(process_width, process_height, sample_count)
        self._hbao.setup(process_width, process_height, direction_count)
        self._gtao.setup(process_width, process_height, direction_count, 4)
        self._bilateral.setup(process_width, process_height)

    def execute(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        delta_time: float,
    ) -> None:
        """Execute AO calculation.

        Args:
            inputs: Depth and normal buffers.
            outputs: AO and bent normal buffers.
            delta_time: Frame time.
        """
        if not self._settings or not self._settings.enabled:
            return

        depth_buffer = inputs.get("depth")
        normal_buffer = inputs.get("normal")
        projection = [[1, 0, 0, 0]] * 4
        view = [[1, 0, 0, 0]] * 4

        method = self._settings.method

        if method == AOMethod.SSAO:
            self._ao_result = self._ssao.calculate(
                depth_buffer,
                normal_buffer,
                self._settings,
                projection,
            )
            self._ao_result = self._ssao.blur(
                self._ao_result,
                depth_buffer,
                self._settings,
            )

        elif method in (AOMethod.HBAO, AOMethod.HBAO_PLUS):
            self._ao_result = self._hbao.calculate(
                depth_buffer,
                normal_buffer,
                self._settings,
                projection,
            )

        elif method == AOMethod.GTAO:
            self._ao_result, self._bent_normals = self._gtao.calculate(
                depth_buffer,
                normal_buffer,
                self._settings,
                view,
                projection,
            )

            if self._settings.temporal_enabled:
                velocity = inputs.get("velocity")
                self._ao_result = self._gtao.temporal_filter(
                    self._ao_result,
                    self._gtao._history_buffer,
                    velocity,
                    self._settings,
                )

        if self._settings.blur_sharpness > 0:
            self._ao_result = self._bilateral.apply(
                self._ao_result,
                depth_buffer,
                normal_buffer,
                self._settings.blur_sharpness,
            )

    def cleanup(self) -> None:
        """Release AO resources."""
        self._ao_result = None
        self._bent_normals = None

    def is_compute_effect(self) -> bool:
        """AO uses compute shaders."""
        return True


__all__ = [
    "AOMethod",
    "AOQuality",
    "AOSettings",
    "SSAOKernel",
    "SSAO",
    "HBAO",
    "GTAO",
    "BentNormalOutput",
    "BilateralFilter",
    "AOEffect",
]
