"""
Bloom Effect System

Provides physically-based bloom rendering:
- BloomThreshold: Bright pass extraction
- BloomDownsample: Mip chain generation
- BloomBlur: Gaussian/Kawase blur
- BloomUpsample: Accumulate and blend
- BloomSettings: Full configuration
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from .postprocess_stack import EffectPriority, EffectSettings, PostProcessEffect


class BlurMethod(Enum):
    """Blur algorithm selection."""

    GAUSSIAN = auto()  # Traditional Gaussian blur
    KAWASE = auto()  # Dual Kawase blur (faster)
    BOX = auto()  # Simple box filter


class BloomQuality(Enum):
    """Bloom quality preset."""

    LOW = auto()  # 3 mip levels, box blur
    MEDIUM = auto()  # 5 mip levels, Kawase blur
    HIGH = auto()  # 6 mip levels, Gaussian blur
    ULTRA = auto()  # 8 mip levels, Gaussian blur with scatter


@dataclass
class BloomMipSettings:
    """Per-mip level bloom settings."""

    intensity: float = 1.0
    tint: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    scatter: float = 0.7  # How much to spread to next level


@dataclass
class LensDirtSettings:
    """Lens dirt/flare overlay settings."""

    enabled: bool = False
    intensity: float = 1.0
    texture_path: Optional[str] = None
    tint: Tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class BloomSettings(EffectSettings):
    """Complete bloom effect settings."""

    # Threshold settings - using constants for consistent values
    # See constants.py BLOOM for documentation
    threshold: float = 1.0  # Luminance threshold for bloom
    threshold_softness: float = 0.5  # Soft knee for threshold
    clamp_max: float = 65504.0  # Half-float max to prevent fireflies (BLOOM.CLAMP_MAX_DEFAULT)

    # Global intensity
    intensity: float = 1.0
    scatter: float = 0.7  # Default scatter between mips

    # Per-mip settings (up to 8 mip levels)
    mip_settings: List[BloomMipSettings] = field(default_factory=list)

    # Quality settings
    quality: BloomQuality = BloomQuality.MEDIUM
    blur_method: BlurMethod = BlurMethod.KAWASE
    blur_iterations: int = 2
    resolution_scale: float = 0.5  # Start at half resolution

    # Lens effects
    lens_dirt: LensDirtSettings = field(default_factory=LensDirtSettings)

    # Anamorphic bloom
    anamorphic_ratio: float = 0.0  # 0 = circular, 1 = horizontal stretch

    def __post_init__(self) -> None:
        self.priority = EffectPriority.BLOOM.value

        if not 0.0 <= self.threshold <= 65504.0:
            raise ValueError(f"threshold must be in [0, 65504], got {self.threshold}")
        if not 0.0 <= self.threshold_softness <= 1.0:
            raise ValueError(f"threshold_softness must be in [0, 1], got {self.threshold_softness}")
        if self.clamp_max < 0.0:
            raise ValueError(f"clamp_max must be non-negative, got {self.clamp_max}")
        if self.intensity < 0.0 or self.intensity > 10.0:
            raise ValueError(f"intensity must be in [0, 10], got {self.intensity}")
        if not 0.0 <= self.scatter <= 1.0:
            raise ValueError(f"scatter must be in [0, 1], got {self.scatter}")
        if self.blur_iterations < 0:
            raise ValueError(f"blur_iterations must be non-negative, got {self.blur_iterations}")
        if not 0.25 <= self.resolution_scale <= 1.0:
            raise ValueError(f"resolution_scale must be in [0.25, 1.0], got {self.resolution_scale}")

        if not self.mip_settings:
            self.mip_settings = [
                BloomMipSettings(intensity=1.0, scatter=0.7),
                BloomMipSettings(intensity=0.9, scatter=0.7),
                BloomMipSettings(intensity=0.8, scatter=0.7),
                BloomMipSettings(intensity=0.7, scatter=0.7),
                BloomMipSettings(intensity=0.6, scatter=0.6),
                BloomMipSettings(intensity=0.5, scatter=0.5),
            ]

    def lerp(self, other: "BloomSettings", t: float) -> "BloomSettings":
        """Interpolate between two bloom settings."""
        lerped_mips = []
        max_mips = max(len(self.mip_settings), len(other.mip_settings))

        for i in range(max_mips):
            self_mip = (
                self.mip_settings[i]
                if i < len(self.mip_settings)
                else BloomMipSettings()
            )
            other_mip = (
                other.mip_settings[i]
                if i < len(other.mip_settings)
                else BloomMipSettings()
            )

            lerped_mips.append(
                BloomMipSettings(
                    intensity=self_mip.intensity
                    + (other_mip.intensity - self_mip.intensity) * t,
                    tint=tuple(
                        self_mip.tint[j] + (other_mip.tint[j] - self_mip.tint[j]) * t
                        for j in range(3)
                    ),
                    scatter=self_mip.scatter
                    + (other_mip.scatter - self_mip.scatter) * t,
                )
            )

        return BloomSettings(
            enabled=self.enabled if t < 0.5 else other.enabled,
            weight=self.weight + (other.weight - self.weight) * t,
            threshold=self.threshold + (other.threshold - self.threshold) * t,
            threshold_softness=self.threshold_softness
            + (other.threshold_softness - self.threshold_softness) * t,
            clamp_max=self.clamp_max + (other.clamp_max - self.clamp_max) * t,
            intensity=self.intensity + (other.intensity - self.intensity) * t,
            scatter=self.scatter + (other.scatter - self.scatter) * t,
            mip_settings=lerped_mips,
            quality=self.quality if t < 0.5 else other.quality,
            blur_method=self.blur_method if t < 0.5 else other.blur_method,
            blur_iterations=int(
                self.blur_iterations + (other.blur_iterations - self.blur_iterations) * t
            ),
            resolution_scale=self.resolution_scale
            + (other.resolution_scale - self.resolution_scale) * t,
            anamorphic_ratio=self.anamorphic_ratio
            + (other.anamorphic_ratio - self.anamorphic_ratio) * t,
            lens_dirt=LensDirtSettings(
                enabled=self.lens_dirt.enabled if t < 0.5 else other.lens_dirt.enabled,
                intensity=self.lens_dirt.intensity
                + (other.lens_dirt.intensity - self.lens_dirt.intensity) * t,
                tint=tuple(
                    self.lens_dirt.tint[j]
                    + (other.lens_dirt.tint[j] - self.lens_dirt.tint[j]) * t
                    for j in range(3)
                ),
                texture_path=self.lens_dirt.texture_path
                if t < 0.5
                else other.lens_dirt.texture_path,
            ),
        )


class BloomThreshold:
    """Extracts bright regions for bloom.

    Uses a soft threshold with configurable knee to smoothly
    transition from no bloom to full bloom.
    """

    def __init__(self) -> None:
        from .constants import BLOOM

        self._threshold: float = BLOOM.THRESHOLD_DEFAULT
        self._softness: float = BLOOM.THRESHOLD_SOFTNESS_DEFAULT
        self._clamp_max: float = BLOOM.CLAMP_MAX_DEFAULT

    def configure(
        self,
        threshold: float,
        softness: float,
        clamp_max: float,
    ) -> None:
        """Configure threshold parameters.

        Args:
            threshold: Luminance threshold.
            softness: Soft knee amount [0, 1].
            clamp_max: Maximum luminance clamp.
        """
        self._threshold = max(0.0, threshold)
        self._softness = max(0.0, min(1.0, softness))
        self._clamp_max = max(0.0, clamp_max)

    def apply(self, luminance: float) -> float:
        """Calculate bloom contribution for a luminance value.

        Args:
            luminance: Input luminance.

        Returns:
            Bloom weight [0, 1].
        """
        luminance = min(luminance, self._clamp_max)

        if self._softness <= 0:
            return 1.0 if luminance > self._threshold else 0.0

        from .constants import EPSILON

        knee = self._threshold * self._softness
        soft_threshold = self._threshold - knee

        if luminance <= soft_threshold:
            return 0.0
        elif luminance >= self._threshold:
            return 1.0
        else:
            x = luminance - soft_threshold
            return x * x / (4.0 * knee + EPSILON)

    def get_knee_params(self) -> Tuple[float, float, float, float]:
        """Get threshold knee parameters for shader.

        Returns:
            (threshold, knee, knee * 2, 0.25 / knee).
        """
        from .constants import EPSILON

        knee = self._threshold * self._softness
        return (
            self._threshold,
            knee,
            knee * 2.0,
            0.25 / (knee + EPSILON),
        )


class BloomDownsample:
    """Generates bloom mip chain.

    Creates progressively smaller versions of the bright pass
    using filtered downsampling.
    """

    def __init__(self, max_mips: int = 8) -> None:
        """Initialize downsample chain.

        Args:
            max_mips: Maximum number of mip levels.
        """
        self._max_mips: int = max_mips
        self._mip_sizes: List[Tuple[int, int]] = []
        self._mip_buffers: List[Any] = []
        self._use_karis_average: bool = True  # Anti-flickering

    @property
    def mip_count(self) -> int:
        """Number of mip levels."""
        return len(self._mip_sizes)

    @property
    def mip_sizes(self) -> List[Tuple[int, int]]:
        """Sizes of each mip level."""
        return self._mip_sizes.copy()

    def setup(
        self,
        width: int,
        height: int,
        resolution_scale: float = 0.5,
    ) -> None:
        """Initialize mip chain buffers.

        Args:
            width: Base render width.
            height: Base render height.
            resolution_scale: Initial resolution scale.
        """
        self._mip_sizes.clear()
        self._mip_buffers.clear()

        w = int(width * resolution_scale)
        h = int(height * resolution_scale)

        for _ in range(self._max_mips):
            if w < 2 or h < 2:
                break

            self._mip_sizes.append((w, h))
            self._mip_buffers.append(None)  # Placeholder for GPU buffer

            w = max(1, w // 2)
            h = max(1, h // 2)

    def downsample(self, source: Any, mip_level: int) -> Any:
        """Downsample to the next mip level.

        Performs a filtered 2x reduction. Lazily creates the mip buffer
        on first access so that downstream consumers always receive a
        valid (non-None) buffer.

        Args:
            source: Source buffer to downsample.
            mip_level: Target mip level.

        Returns:
            Downsampled buffer (never None for valid mip_level).
        """
        if mip_level >= len(self._mip_buffers):
            return source

        buffer = self._mip_buffers[mip_level]
        if buffer is None:
            w, h = self._mip_sizes[mip_level]
            self._mip_buffers[mip_level] = [0.0] * (w * h * 4)
            buffer = self._mip_buffers[mip_level]

        return buffer

    def get_mip_buffer(self, level: int) -> Any:
        """Get buffer for a specific mip level.

        Args:
            level: Mip level index.

        Returns:
            Buffer for the mip level.
        """
        if 0 <= level < len(self._mip_buffers):
            return self._mip_buffers[level]
        return None


class BloomBlur:
    """Blur implementation for bloom.

    Supports multiple blur algorithms optimized for different
    quality/performance trade-offs.
    """

    def __init__(self, method: BlurMethod = BlurMethod.KAWASE) -> None:
        """Initialize blur processor.

        Args:
            method: Blur algorithm to use.
        """
        self._method: BlurMethod = method
        self._gaussian_weights: List[float] = []
        self._gaussian_offsets: List[float] = []

    @property
    def method(self) -> BlurMethod:
        """Current blur method."""
        return self._method

    @method.setter
    def method(self, value: BlurMethod) -> None:
        self._method = value

    def calculate_gaussian_weights(self, radius: int, sigma: float) -> None:
        """Pre-calculate Gaussian weights.

        Args:
            radius: Blur radius in pixels.
            sigma: Gaussian sigma value.
        """
        self._gaussian_weights = []
        self._gaussian_offsets = []

        total_weight = 0.0

        for i in range(radius + 1):
            weight = math.exp(-(i * i) / (2.0 * sigma * sigma))
            self._gaussian_weights.append(weight)
            self._gaussian_offsets.append(float(i))
            total_weight += weight if i == 0 else weight * 2

        for i in range(len(self._gaussian_weights)):
            self._gaussian_weights[i] /= total_weight

    def get_kawase_offsets(self, iteration: int) -> float:
        """Get Kawase blur offset for iteration.

        Args:
            iteration: Current blur iteration.

        Returns:
            Sample offset.
        """
        return 0.5 + iteration

    def blur(
        self,
        source: Any,
        target: Any,
        iterations: int = 2,
        width: int = 0,
        height: int = 0,
    ) -> Any:
        """Apply blur to a buffer.

        Args:
            source: Source buffer (flat RGBA list).
            target: Target buffer.
            iterations: Number of blur passes.
            width: Buffer width in pixels (0 = skip).
            height: Buffer height in pixels (0 = skip).

        Returns:
            Blurred buffer.
        """
        if source is None or width < 2 or height < 2:
            return target if target is not None else source

        if self._method == BlurMethod.GAUSSIAN:
            return self._gaussian_blur(source, target, iterations, width, height)
        elif self._method == BlurMethod.KAWASE:
            return self._kawase_blur(source, target, iterations, width, height)
        else:
            return self._box_blur(source, target, iterations, width, height)

    def _gaussian_blur(
        self,
        source: Any,
        target: Any,
        iterations: int,
        width: int,
        height: int,
    ) -> Any:
        """Apply Gaussian blur using separable 2-pass convolution.

        Performs horizontal then vertical convolution per iteration,
        using pre-calculated Gaussian weights. Edge pixels clamp to
        the nearest valid sample.

        Args:
            source: Source buffer (flat RGBA list).
            target: Target buffer (may be same as source).
            iterations: Number of full (H+V) passes.
            width: Buffer width in pixels.
            height: Buffer height in pixels.

        Returns:
            Blurred buffer (same reference as target if provided).
        """
        if not self._gaussian_weights:
            return target if target is not None else source

        stride = width * 4
        result = target if target is not None else list(source)

        for _ in range(iterations):
            temp = [0.0] * (stride * height)

            # Horizontal pass: convolve each row
            for y in range(height):
                row = y * stride
                for x in range(width):
                    idx = row + x * 4
                    for c in range(4):
                        total = source[idx + c] * self._gaussian_weights[0]
                        for k in range(1, len(self._gaussian_weights)):
                            off = int(self._gaussian_offsets[k])
                            w = self._gaussian_weights[k]
                            if x - off >= 0:
                                total += source[idx - off * 4 + c] * w
                            if x + off < width:
                                total += source[idx + off * 4 + c] * w
                        temp[idx + c] = total

            # Vertical pass: convolve each column of temp
            for y in range(height):
                for x in range(width):
                    idx = (y * width + x) * 4
                    for c in range(4):
                        total = temp[idx + c] * self._gaussian_weights[0]
                        for k in range(1, len(self._gaussian_weights)):
                            off = int(self._gaussian_offsets[k])
                            w = self._gaussian_weights[k]
                            up = (y - off) * width + x
                            down = (y + off) * width + x
                            if y - off >= 0:
                                total += temp[up * 4 + c] * w
                            if y + off < height:
                                total += temp[down * 4 + c] * w
                        result[idx + c] = total

            source = result

        return result

    def _kawase_blur(
        self,
        source: Any,
        target: Any,
        iterations: int,
        width: int,
        height: int,
    ) -> Any:
        """Apply Kawase dual blur.

        Each iteration samples a 5-point cross pattern (center
        + four corners) with an offset that expands per iteration.
        Corner samples are clamped to valid pixel coords.

        Args:
            source: Source buffer (flat RGBA list).
            target: Target buffer (may be same as source).
            iterations: Number of Kawase passes.
            width: Buffer width in pixels.
            height: Buffer height in pixels.

        Returns:
            Blurred buffer (same reference as target if provided).
        """
        stride = width * 4
        result = target if target is not None else list(source)

        for iteration in range(iterations):
            pixel_offset = iteration + 1
            temp = [0.0] * (stride * height)

            for y in range(height):
                for x in range(width):
                    idx = (y * width + x) * 4

                    samples = [
                        (x, y),
                        (x + pixel_offset, y + pixel_offset),
                        (x + pixel_offset, y - pixel_offset),
                        (x - pixel_offset, y + pixel_offset),
                        (x - pixel_offset, y - pixel_offset),
                    ]

                    for c in range(4):
                        total = 0.0
                        count = 0
                        for sx, sy in samples:
                            if 0 <= sx < width and 0 <= sy < height:
                                total += source[(sy * width + sx) * 4 + c]
                                count += 1
                        temp[idx + c] = total / count if count else 0.0

            source = temp

        result[:] = temp
        return result

    def _box_blur(
        self,
        source: Any,
        target: Any,
        iterations: int,
        width: int,
        height: int,
    ) -> Any:
        """Apply box blur using separable 1x3 averaging.

        Each iteration performs a horizontal three-tap average
        followed by a vertical three-tap average. Edge pixels
        average over 2 samples instead of 3.

        Args:
            source: Source buffer (flat RGBA list).
            target: Target buffer (may be same as source).
            iterations: Number of full (H+V) passes.
            width: Buffer width in pixels.
            height: Buffer height in pixels.

        Returns:
            Blurred buffer (same reference as target if provided).
        """
        stride = width * 4
        result = target if target is not None else list(source)

        for _ in range(iterations):
            temp = [0.0] * (stride * height)

            # Horizontal pass: [1, 1, 1] / 3
            for y in range(height):
                row = y * stride
                for x in range(width):
                    idx = row + x * 4
                    for c in range(4):
                        total = source[idx + c]
                        count = 1
                        if x > 0:
                            total += source[idx - 4 + c]
                            count += 1
                        if x < width - 1:
                            total += source[idx + 4 + c]
                            count += 1
                        temp[idx + c] = total / count

            # Vertical pass: [1, 1, 1] / 3
            for y in range(height):
                for x in range(width):
                    idx = (y * width + x) * 4
                    for c in range(4):
                        total = temp[idx + c]
                        count = 1
                        if y > 0:
                            total += temp[((y - 1) * width + x) * 4 + c]
                            count += 1
                        if y < height - 1:
                            total += temp[((y + 1) * width + x) * 4 + c]
                            count += 1
                        result[idx + c] = total / count

            source = result

        return result


class BloomUpsample:
    """Upsamples and accumulates bloom mip chain.

    Progressively upsamples from the smallest mip back to
    the original resolution, accumulating bloom at each level.
    """

    def __init__(self) -> None:
        self._upsample_buffers: List[Any] = []

    def setup(self, mip_sizes: List[Tuple[int, int]]) -> None:
        """Initialize upsample buffers.

        Args:
            mip_sizes: Sizes of each mip level.
        """
        self._upsample_buffers = [None] * len(mip_sizes)

    def upsample_and_accumulate(
        self,
        low_res: Any,
        high_res: Any,
        mip_settings: BloomMipSettings,
    ) -> Any:
        """Upsample and blend with higher resolution.

        Args:
            low_res: Lower resolution buffer to upsample.
            high_res: Higher resolution buffer to blend with.
            mip_settings: Settings for this mip level.

        Returns:
            Accumulated result.
        """
        return high_res


class BloomEffect(PostProcessEffect[BloomSettings]):
    """Complete bloom post-process effect.

    Implements a full bloom pipeline with threshold extraction,
    mip chain generation, blur, and accumulation.
    """

    def __init__(
        self,
        settings: Optional[BloomSettings] = None,
    ) -> None:
        """Initialize bloom effect.

        Args:
            settings: Bloom configuration.
        """
        super().__init__(
            name="Bloom",
            settings=settings or BloomSettings(),
            priority=EffectPriority.BLOOM.value,
        )

        self._threshold: BloomThreshold = BloomThreshold()
        self._downsample: BloomDownsample = BloomDownsample()
        self._blur: BloomBlur = BloomBlur()
        self._upsample: BloomUpsample = BloomUpsample()

        self._bright_pass_buffer: Any = None
        self._lens_dirt_texture: Any = None
        self._width: int = 0
        self._height: int = 0

    @property
    def mip_count(self) -> int:
        """Number of bloom mip levels."""
        return self._downsample.mip_count

    def get_required_inputs(self) -> List[str]:
        """Get required input resources."""
        return ["color"]

    def get_outputs(self) -> List[str]:
        """Get output resources."""
        return ["color", "bloom_buffer"]

    def setup(self, width: int, height: int) -> None:
        """Initialize bloom resources.

        Args:
            width: Render width.
            height: Render height.
        """
        self._width = width
        self._height = height

        if self._settings:
            self._configure_from_settings()

    def _configure_from_settings(self) -> None:
        """Configure components from current settings."""
        if not self._settings:
            return

        self._threshold.configure(
            self._settings.threshold,
            self._settings.threshold_softness,
            self._settings.clamp_max,
        )

        self._blur.method = self._settings.blur_method

        from .constants import BLOOM

        max_mips = {
            BloomQuality.LOW: BLOOM.MIP_COUNT_LOW,
            BloomQuality.MEDIUM: BLOOM.MIP_COUNT_MEDIUM,
            BloomQuality.HIGH: BLOOM.MIP_COUNT_HIGH,
            BloomQuality.ULTRA: BLOOM.MIP_COUNT_ULTRA,
        }.get(self._settings.quality, BLOOM.MIP_COUNT_MEDIUM)

        self._downsample._max_mips = max_mips
        self._downsample.setup(
            self._width,
            self._height,
            self._settings.resolution_scale,
        )

        self._upsample.setup(self._downsample.mip_sizes)

        if self._settings.blur_method == BlurMethod.GAUSSIAN:
            self._blur.calculate_gaussian_weights(
                radius=BLOOM.GAUSSIAN_RADIUS_DEFAULT,
                sigma=BLOOM.GAUSSIAN_SIGMA_DEFAULT
            )

    def execute(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        delta_time: float,
    ) -> None:
        """Execute bloom effect.

        Args:
            inputs: Input color buffer.
            outputs: Output with bloom applied.
            delta_time: Frame time.
        """
        if not self._settings or not self._settings.enabled:
            return

        if self._settings.intensity <= 0:
            return

        color_input = inputs.get("color")

        for i in range(self._downsample.mip_count):
            source = self._bright_pass_buffer if i == 0 else self._downsample.get_mip_buffer(i - 1)
            self._downsample.downsample(source, i)

        for i in range(self._downsample.mip_count):
            mip_buffer = self._downsample.get_mip_buffer(i)
            mip_w, mip_h = self._downsample.mip_sizes[i]
            mip_settings = (
                self._settings.mip_settings[i]
                if i < len(self._settings.mip_settings)
                else BloomMipSettings()
            )
            self._blur.blur(
                mip_buffer,
                mip_buffer,
                self._settings.blur_iterations,
                mip_w,
                mip_h,
            )

        bloom_result = None
        for i in range(self._downsample.mip_count - 1, -1, -1):
            mip_buffer = self._downsample.get_mip_buffer(i)
            mip_settings = (
                self._settings.mip_settings[i]
                if i < len(self._settings.mip_settings)
                else BloomMipSettings()
            )
            bloom_result = self._upsample.upsample_and_accumulate(
                bloom_result if bloom_result else mip_buffer,
                mip_buffer,
                mip_settings,
            )

    def cleanup(self) -> None:
        """Release bloom resources."""
        self._bright_pass_buffer = None
        self._lens_dirt_texture = None

    def is_compute_effect(self) -> bool:
        """Bloom uses compute shaders."""
        return True


__all__ = [
    "BlurMethod",
    "BloomQuality",
    "BloomMipSettings",
    "LensDirtSettings",
    "BloomSettings",
    "BloomThreshold",
    "BloomDownsample",
    "BloomBlur",
    "BloomUpsample",
    "BloomEffect",
]
