"""
HDR (High Dynamic Range) display support.

Provides HDR capability detection and color space management with
sensible defaults for headless operation.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from ..constants import (
    HDR_DEFAULT_MIN_LUMINANCE, HDR_DEFAULT_MAX_LUMINANCE,
    HDR_DEFAULT_MAX_FULL_FRAME_LUMINANCE, HDR_METADATA_DEFAULT_MAX_CLL
)


class ColorSpace(Enum):
    """Color space options."""
    SRGB = auto()
    SCRGB = auto()
    HDR10 = auto()
    PQ = auto()
    DOLBY_VISION = auto()


@dataclass(slots=True)
class HDRCapabilities:
    """HDR capabilities information."""
    supported: bool
    min_luminance: float  # cd/m²
    max_luminance: float  # cd/m²
    max_full_frame_luminance: float  # cd/m²
    color_space: ColorSpace


class DisplayHDR:
    """
    HDR display management.

    Provides HDR capability detection and color space management with
    a headless backend for testing.
    """

    __slots__ = (
        "_capabilities",
        "_current_color_space",
        "_metadata",
    )

    def __init__(self, simulate_hdr: bool = False):
        """
        Initialize HDR manager.

        Args:
            simulate_hdr: Simulate HDR support for testing
        """
        if simulate_hdr:
            self._capabilities = HDRCapabilities(
                supported=True,
                min_luminance=HDR_DEFAULT_MIN_LUMINANCE,
                max_luminance=HDR_DEFAULT_MAX_LUMINANCE,
                max_full_frame_luminance=HDR_DEFAULT_MAX_FULL_FRAME_LUMINANCE,
                color_space=ColorSpace.HDR10
            )
        else:
            self._capabilities = HDRCapabilities(
                supported=False,
                min_luminance=0.0,
                max_luminance=HDR_METADATA_DEFAULT_MAX_CLL,
                max_full_frame_luminance=HDR_METADATA_DEFAULT_MAX_CLL,
                color_space=ColorSpace.SRGB
            )

        self._current_color_space = self._capabilities.color_space
        self._metadata: dict[str, float] = {}

    def is_supported(self) -> bool:
        """
        Check if HDR is supported on this instance.

        Returns:
            True if HDR is supported
        """
        return self._capabilities.supported

    def get_capabilities(self) -> HDRCapabilities:
        """
        Get HDR capabilities of the display.

        Returns:
            HDRCapabilities object with display information
        """
        return self._capabilities

    def set_color_space(self, color_space: ColorSpace) -> bool:
        """
        Set the color space for rendering.

        Args:
            color_space: Target color space

        Returns:
            True if color space was set successfully
        """
        if not self._capabilities.supported and color_space != ColorSpace.SRGB:
            return False

        self._current_color_space = color_space
        return True

    def set_metadata(
        self,
        max_content_light_level: Optional[float] = None,
        max_frame_average_light_level: Optional[float] = None,
        min_mastering_luminance: Optional[float] = None,
        max_mastering_luminance: Optional[float] = None
    ) -> None:
        """
        Set HDR metadata for content.

        Args:
            max_content_light_level: Maximum content light level (cd/m²)
            max_frame_average_light_level: Maximum frame average light level (cd/m²)
            min_mastering_luminance: Minimum mastering display luminance (cd/m²)
            max_mastering_luminance: Maximum mastering display luminance (cd/m²)
        """
        if max_content_light_level is not None:
            self._metadata["max_cll"] = max_content_light_level
        if max_frame_average_light_level is not None:
            self._metadata["max_fall"] = max_frame_average_light_level
        if min_mastering_luminance is not None:
            self._metadata["min_mdl"] = min_mastering_luminance
        if max_mastering_luminance is not None:
            self._metadata["max_mdl"] = max_mastering_luminance

    @property
    def current_color_space(self) -> ColorSpace:
        """Get current color space."""
        return self._current_color_space

    @property
    def metadata(self) -> dict[str, float]:
        """Get current HDR metadata."""
        return self._metadata.copy()
