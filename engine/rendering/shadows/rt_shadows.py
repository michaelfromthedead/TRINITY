"""
RT Shadow Dispatch System

Provides ray-traced shadow rendering dispatch:
- RTShadowQuality: Quality presets controlling rays per pixel
- RTShadowParams: Shadow ray configuration parameters
- RTShadowDispatcher: Main RT shadow dispatch class
- ShadowFallbackDispatcher: Fallback shadow techniques for non-RT hardware
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from engine.platform.rhi.device import Device
    from engine.platform.rhi.resources import Buffer, Texture
    from engine.platform.rhi.raytracing import TLASHandle


# =============================================================================
# Quality Presets
# =============================================================================


class RTShadowQuality(IntEnum):
    """RT shadow quality preset.

    Controls the number of shadow rays traced per pixel.
    Higher ray counts provide softer, more accurate shadows at increased cost.
    """

    LOW = 1       # 1 ray/pixel - hard shadows, fastest
    MEDIUM = 2    # 2 rays/pixel - slight softness
    HIGH = 4      # 4 rays/pixel - soft penumbra
    ULTRA = 8     # 8 rays/pixel - smooth, high quality


# =============================================================================
# Shadow Parameters
# =============================================================================


@dataclass
class RTShadowParams:
    """RT shadow configuration parameters.

    Controls ray tracing behavior for shadow computation including
    quality, distance limits, and alpha testing settings.

    Attributes:
        quality: Quality preset controlling ray count per pixel.
        max_distance: Maximum shadow ray distance in world units.
        bias: Depth bias to prevent shadow acne (self-shadowing artifacts).
        alpha_test_enabled: Whether to evaluate alpha for semi-transparent geometry.
        alpha_cutoff: Alpha threshold for alpha testing (0.0-1.0).
    """

    quality: RTShadowQuality = RTShadowQuality.MEDIUM
    max_distance: float = 1000.0
    bias: float = 0.001
    alpha_test_enabled: bool = True
    alpha_cutoff: float = 0.5

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if not isinstance(self.quality, RTShadowQuality):
            raise TypeError(
                f"quality must be RTShadowQuality, got {type(self.quality).__name__}"
            )
        if self.max_distance <= 0.0:
            raise ValueError(f"max_distance must be positive, got {self.max_distance}")
        if self.bias < 0.0:
            raise ValueError(f"bias must be non-negative, got {self.bias}")
        if not 0.0 <= self.alpha_cutoff <= 1.0:
            raise ValueError(
                f"alpha_cutoff must be in [0.0, 1.0], got {self.alpha_cutoff}"
            )

    def get_shader_params(self) -> Tuple[float, float, float]:
        """Get shader-bindable parameters.

        Returns:
            Tuple of (max_distance, bias, alpha_cutoff).
        """
        return (self.max_distance, self.bias, self.alpha_cutoff)

    def with_quality(self, quality: RTShadowQuality) -> "RTShadowParams":
        """Create a copy with different quality setting.

        Args:
            quality: New quality level.

        Returns:
            New RTShadowParams with updated quality.
        """
        return RTShadowParams(
            quality=quality,
            max_distance=self.max_distance,
            bias=self.bias,
            alpha_test_enabled=self.alpha_test_enabled,
            alpha_cutoff=self.alpha_cutoff,
        )


# =============================================================================
# RT Shadow Dispatcher
# =============================================================================


class RTShadowDispatcher:
    """Ray-traced shadow dispatcher.

    Dispatches RT shadow ray generation shaders for computing accurate
    shadows using hardware ray tracing. Falls back gracefully when
    RT is unavailable.

    Example:
        dispatcher = RTShadowDispatcher(device)
        if dispatcher.supports_rt():
            params = RTShadowParams(quality=RTShadowQuality.HIGH)
            dispatcher.dispatch_shadows(tlas, depth, normal, lights, output, params)
    """

    def __init__(self, device: "Device") -> None:
        """Initialize the RT shadow dispatcher.

        Args:
            device: RHI device for resource creation and capability query.
        """
        self._device = device
        self._rt_supported: Optional[bool] = None
        self._pipeline = None  # RT pipeline state (lazy init)
        self._initialized = False

    @property
    def device(self) -> "Device":
        """Get the RHI device.

        Returns:
            The device used for dispatching.
        """
        return self._device

    @property
    def is_initialized(self) -> bool:
        """Check if dispatcher has been initialized.

        Returns:
            True if RT pipeline has been created.
        """
        return self._initialized

    def supports_rt(self) -> bool:
        """Check if device supports RT shadows.

        Queries the device for ray tracing support. Result is cached
        after first query.

        Returns:
            True if device supports hardware ray tracing.
        """
        if self._rt_supported is None:
            # Query device capabilities
            try:
                # Access the underlying adapter to query features
                if hasattr(self._device, '_adapter'):
                    features = self._device._adapter.query_features()
                    self._rt_supported = features.ray_tracing
                else:
                    # Fallback: assume RT not supported if we can't query
                    self._rt_supported = False
            except Exception:
                self._rt_supported = False
        return self._rt_supported

    def get_ray_count_per_pixel(self, quality: RTShadowQuality) -> int:
        """Return ray count for quality level.

        Args:
            quality: Quality preset to query.

        Returns:
            Number of shadow rays per pixel.
        """
        return int(quality)

    def get_total_ray_count(
        self, quality: RTShadowQuality, width: int, height: int
    ) -> int:
        """Calculate total ray count for given resolution.

        Args:
            quality: Quality preset.
            width: Output width in pixels.
            height: Output height in pixels.

        Returns:
            Total number of shadow rays to trace.
        """
        return self.get_ray_count_per_pixel(quality) * width * height

    def dispatch_shadows(
        self,
        tlas: "TLASHandle",
        depth_buffer: "Texture",
        normal_buffer: "Texture",
        light_buffer: "Buffer",
        output: "Texture",
        params: Optional[RTShadowParams] = None,
    ) -> None:
        """Dispatch RT shadow ray generation.

        Traces shadow rays from visible surfaces toward lights using
        the provided TLAS for intersection testing.

        Args:
            tlas: Top-level acceleration structure for scene geometry.
            depth_buffer: Depth buffer for reconstructing world positions.
            normal_buffer: Normal buffer for ray origin offset.
            light_buffer: Buffer containing light data (positions, directions).
            output: Output texture for shadow mask (0=shadowed, 1=lit).
            params: Shadow parameters. Uses defaults if None.

        Raises:
            RuntimeError: If RT is not supported by the device.
            ValueError: If input textures are invalid.
        """
        if not self.supports_rt():
            raise RuntimeError("RT shadows not supported on this device")

        if params is None:
            params = RTShadowParams()

        # Validate inputs
        if tlas is None:
            raise ValueError("tlas cannot be None")
        if depth_buffer is None or not depth_buffer.is_valid():
            raise ValueError("depth_buffer is invalid")
        if normal_buffer is None or not normal_buffer.is_valid():
            raise ValueError("normal_buffer is invalid")
        if output is None or not output.is_valid():
            raise ValueError("output texture is invalid")

        # Get dimensions
        depth_desc = depth_buffer.desc
        output_desc = output.desc

        # Validate dimension match
        if output_desc.width != depth_desc.width or output_desc.height != depth_desc.height:
            raise ValueError(
                f"Output dimensions ({output_desc.width}x{output_desc.height}) "
                f"do not match depth buffer ({depth_desc.width}x{depth_desc.height})"
            )

        # Dispatch RT shadow shader
        self._dispatch_rt_shadows(
            tlas=tlas,
            depth_buffer=depth_buffer,
            normal_buffer=normal_buffer,
            light_buffer=light_buffer,
            output=output,
            params=params,
            width=depth_desc.width,
            height=depth_desc.height,
        )

    def _dispatch_rt_shadows(
        self,
        tlas: "TLASHandle",
        depth_buffer: "Texture",
        normal_buffer: "Texture",
        light_buffer: "Buffer",
        output: "Texture",
        params: RTShadowParams,
        width: int,
        height: int,
    ) -> None:
        """Internal dispatch of RT shadow shader.

        Args:
            tlas: Acceleration structure.
            depth_buffer: Depth texture.
            normal_buffer: Normal texture.
            light_buffer: Light buffer.
            output: Output shadow mask.
            params: Shadow parameters.
            width: Output width.
            height: Output height.
        """
        # Stub implementation: In real implementation, would:
        # 1. Initialize RT pipeline if not done
        # 2. Bind TLAS, depth, normal, light buffers
        # 3. Set shader parameters (max_distance, bias, alpha settings)
        # 4. Dispatch ray generation shader with (width * height * ray_count) rays
        # 5. Handle alpha testing if enabled

        ray_count = self.get_ray_count_per_pixel(params.quality)
        total_rays = width * height * ray_count

        # Store dispatch info for verification
        self._last_dispatch = {
            "width": width,
            "height": height,
            "ray_count": ray_count,
            "total_rays": total_rays,
            "alpha_test": params.alpha_test_enabled,
        }
        self._initialized = True

    def destroy(self) -> None:
        """Release dispatcher resources."""
        self._pipeline = None
        self._initialized = False


# =============================================================================
# Fallback Shadow Dispatcher
# =============================================================================


class ShadowTechnique:
    """Enumeration of shadow techniques."""

    RT_SHADOWS = "rt_shadows"
    CSM = "csm"  # Cascaded Shadow Maps
    CONTACT_SHADOWS = "contact_shadows"
    VSM = "vsm"  # Variance Shadow Maps


class ShadowFallbackDispatcher:
    """Fallback shadow dispatcher for non-RT hardware.

    Provides traditional shadow techniques when hardware ray tracing
    is unavailable, including Cascaded Shadow Maps (CSM) and
    screen-space contact shadows.

    Example:
        fallback = ShadowFallbackDispatcher(device)
        technique = fallback.select_best_technique(rt_available=False)
        if technique == ShadowTechnique.CSM:
            fallback.dispatch_csm(...)
    """

    # CSM configuration
    DEFAULT_CASCADE_COUNT = 4
    DEFAULT_CASCADE_SPLITS = [0.05, 0.15, 0.35, 1.0]

    # Contact shadow configuration
    DEFAULT_CONTACT_RAY_LENGTH = 0.5
    DEFAULT_CONTACT_STEP_COUNT = 16

    def __init__(self, device: "Device") -> None:
        """Initialize fallback shadow dispatcher.

        Args:
            device: RHI device for resource creation.
        """
        self._device = device
        self._csm_initialized = False
        self._contact_initialized = False
        self._cascade_count = self.DEFAULT_CASCADE_COUNT
        self._cascade_splits = list(self.DEFAULT_CASCADE_SPLITS)

    @property
    def device(self) -> "Device":
        """Get the RHI device."""
        return self._device

    @property
    def cascade_count(self) -> int:
        """Get CSM cascade count."""
        return self._cascade_count

    @cascade_count.setter
    def cascade_count(self, value: int) -> None:
        """Set CSM cascade count.

        Args:
            value: Number of cascades (1-8).

        Raises:
            ValueError: If value out of range.
        """
        if not 1 <= value <= 8:
            raise ValueError(f"cascade_count must be in [1, 8], got {value}")
        self._cascade_count = value

    def configure_csm(
        self,
        cascade_count: int = DEFAULT_CASCADE_COUNT,
        cascade_splits: Optional[List[float]] = None,
    ) -> None:
        """Configure CSM parameters.

        Args:
            cascade_count: Number of shadow map cascades.
            cascade_splits: Split distances as fractions of far plane.

        Raises:
            ValueError: If cascade_splits length doesn't match cascade_count.
        """
        self.cascade_count = cascade_count

        if cascade_splits is not None:
            if len(cascade_splits) != cascade_count:
                raise ValueError(
                    f"cascade_splits length ({len(cascade_splits)}) must match "
                    f"cascade_count ({cascade_count})"
                )
            # Validate monotonically increasing
            for i in range(1, len(cascade_splits)):
                if cascade_splits[i] <= cascade_splits[i - 1]:
                    raise ValueError("cascade_splits must be monotonically increasing")
            self._cascade_splits = list(cascade_splits)
        else:
            # Generate default splits for cascade count
            self._cascade_splits = self._generate_cascade_splits(cascade_count)

    def _generate_cascade_splits(self, count: int) -> List[float]:
        """Generate default cascade splits using logarithmic distribution.

        Args:
            count: Number of cascades.

        Returns:
            List of split distances as fractions.
        """
        # Practical split distances (log-linear blend)
        splits = []
        for i in range(count):
            fraction = (i + 1) / count
            # Blend between linear and logarithmic
            linear = fraction
            log = 0.01 * (10.0 ** (fraction * 2.0)) / 100.0
            split = 0.5 * linear + 0.5 * log
            splits.append(min(split, 1.0))
        return splits

    def dispatch_csm(
        self,
        scene_bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        light_direction: Tuple[float, float, float],
        view_matrix: List[float],
        projection_matrix: List[float],
        shadow_maps: List["Texture"],
        output: "Texture",
    ) -> None:
        """Dispatch Cascaded Shadow Map rendering.

        Args:
            scene_bounds: World-space AABB as (min_point, max_point).
            light_direction: Normalized light direction vector.
            view_matrix: 4x4 camera view matrix (16 floats row-major).
            projection_matrix: 4x4 camera projection matrix.
            shadow_maps: List of shadow map textures (one per cascade).
            output: Output shadow mask texture.

        Raises:
            ValueError: If shadow_maps count doesn't match cascade_count.
        """
        if len(shadow_maps) != self._cascade_count:
            raise ValueError(
                f"shadow_maps count ({len(shadow_maps)}) must match "
                f"cascade_count ({self._cascade_count})"
            )

        # Validate textures
        for i, sm in enumerate(shadow_maps):
            if sm is None or not sm.is_valid():
                raise ValueError(f"shadow_maps[{i}] is invalid")

        if output is None or not output.is_valid():
            raise ValueError("output texture is invalid")

        # Stub implementation: In real implementation, would:
        # 1. For each cascade:
        #    a. Compute light-space view/projection matrix
        #    b. Render scene depth to shadow map
        # 2. Sample all cascades to produce final shadow mask

        self._csm_initialized = True
        self._last_csm_dispatch = {
            "cascade_count": self._cascade_count,
            "splits": self._cascade_splits,
        }

    def dispatch_contact_shadows(
        self,
        depth_buffer: "Texture",
        normal_buffer: "Texture",
        light_direction: Tuple[float, float, float],
        output: "Texture",
        ray_length: float = DEFAULT_CONTACT_RAY_LENGTH,
        step_count: int = DEFAULT_CONTACT_STEP_COUNT,
    ) -> None:
        """Dispatch screen-space contact shadow rendering.

        Traces short rays in screen space to detect contact shadows
        that CSM might miss due to resolution limits.

        Args:
            depth_buffer: Scene depth buffer.
            normal_buffer: Scene normal buffer.
            light_direction: Normalized light direction.
            output: Output shadow mask texture.
            ray_length: Screen-space ray length (0.0-1.0).
            step_count: Number of ray march steps.

        Raises:
            ValueError: If ray_length or step_count out of range.
        """
        if not 0.0 < ray_length <= 1.0:
            raise ValueError(f"ray_length must be in (0.0, 1.0], got {ray_length}")
        if not 1 <= step_count <= 64:
            raise ValueError(f"step_count must be in [1, 64], got {step_count}")

        if depth_buffer is None or not depth_buffer.is_valid():
            raise ValueError("depth_buffer is invalid")
        if output is None or not output.is_valid():
            raise ValueError("output texture is invalid")

        # Stub implementation: In real implementation, would:
        # 1. For each pixel, march ray in light direction
        # 2. Sample depth buffer along ray
        # 3. Detect occlusion when ray goes behind depth

        self._contact_initialized = True
        self._last_contact_dispatch = {
            "ray_length": ray_length,
            "step_count": step_count,
        }

    def select_best_technique(self, rt_available: bool) -> str:
        """Select best available shadow technique.

        Args:
            rt_available: Whether hardware RT is available.

        Returns:
            Name of the recommended shadow technique.
        """
        if rt_available:
            return ShadowTechnique.RT_SHADOWS
        # For non-RT, prefer CSM with contact shadows as enhancement
        return ShadowTechnique.CSM

    def get_supported_techniques(self) -> List[str]:
        """Get list of supported shadow techniques.

        Returns:
            List of technique names this dispatcher supports.
        """
        return [
            ShadowTechnique.CSM,
            ShadowTechnique.CONTACT_SHADOWS,
            ShadowTechnique.VSM,
        ]

    def destroy(self) -> None:
        """Release dispatcher resources."""
        self._csm_initialized = False
        self._contact_initialized = False


# =============================================================================
# Convenience Functions
# =============================================================================


def create_shadow_dispatcher(
    device: "Device",
) -> Union[RTShadowDispatcher, ShadowFallbackDispatcher]:
    """Factory function to create appropriate shadow dispatcher.

    Creates an RT shadow dispatcher if the device supports ray tracing,
    otherwise returns a fallback dispatcher with CSM support.

    Args:
        device: RHI device for capability query and resource creation.

    Returns:
        RTShadowDispatcher if RT supported, else ShadowFallbackDispatcher.
    """
    rt_dispatcher = RTShadowDispatcher(device)
    if rt_dispatcher.supports_rt():
        return rt_dispatcher
    return ShadowFallbackDispatcher(device)


def estimate_shadow_cost(
    params: RTShadowParams,
    resolution: Tuple[int, int],
    technique: str = ShadowTechnique.RT_SHADOWS,
) -> dict:
    """Estimate performance cost for shadow rendering.

    Provides rough estimates for shadow rendering workload based on
    quality settings and resolution.

    Args:
        params: Shadow parameters (quality affects ray count).
        resolution: Output resolution as (width, height).
        technique: Shadow technique to estimate.

    Returns:
        Dictionary with cost estimates:
        - ray_count: Total rays to trace (RT only)
        - memory_mb: Estimated memory usage
        - relative_cost: Relative cost (1.0 = baseline)
    """
    width, height = resolution
    pixel_count = width * height

    if technique == ShadowTechnique.RT_SHADOWS:
        ray_count = int(params.quality) * pixel_count
        # Rough memory estimate: ray payload + output
        memory_mb = (ray_count * 32 + pixel_count * 4) / (1024 * 1024)
        # Relative cost scales with ray count
        relative_cost = float(params.quality) / 2.0  # MEDIUM = 1.0
        # Alpha testing adds ~20% overhead
        if params.alpha_test_enabled:
            relative_cost *= 1.2
    elif technique == ShadowTechnique.CSM:
        ray_count = 0  # No rays for rasterized shadows
        # CSM memory: 4 cascades * resolution * depth format
        cascade_resolution = 2048
        cascade_count = 4
        memory_mb = (cascade_count * cascade_resolution * cascade_resolution * 4) / (1024 * 1024)
        relative_cost = 0.4  # CSM typically cheaper than RT
    else:
        ray_count = 0
        memory_mb = pixel_count * 4 / (1024 * 1024)
        relative_cost = 0.2

    return {
        "ray_count": ray_count,
        "memory_mb": round(memory_mb, 2),
        "relative_cost": round(relative_cost, 2),
        "pixels": pixel_count,
        "technique": technique,
    }


def create_default_params() -> RTShadowParams:
    """Create default RT shadow parameters.

    Returns:
        RTShadowParams with balanced defaults.
    """
    return RTShadowParams()


def create_quality_params(quality: RTShadowQuality) -> RTShadowParams:
    """Create parameters for a specific quality level.

    Args:
        quality: Desired quality level.

    Returns:
        RTShadowParams configured for the given quality.
    """
    # Adjust bias based on quality (finer bias for higher quality)
    bias_scale = {
        RTShadowQuality.LOW: 0.002,
        RTShadowQuality.MEDIUM: 0.001,
        RTShadowQuality.HIGH: 0.0005,
        RTShadowQuality.ULTRA: 0.00025,
    }

    return RTShadowParams(
        quality=quality,
        bias=bias_scale.get(quality, 0.001),
    )
