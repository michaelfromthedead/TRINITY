"""Variable Rate Shading (VRS) utilities for XR rendering.

This module provides shared utilities for working with VRS shading rates
across different foveated rendering implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.xr.rendering.foveated import ShadingRate


# Shading rate to VRS integer value mapping
_SHADING_RATE_VALUES = {
    "FULL": 0,
    "HALF_X": 1,
    "HALF_Y": 2,
    "HALF": 3,
    "QUARTER_X": 4,
    "QUARTER_Y": 5,
    "QUARTER": 6,
}

# Shading rate to pixel coverage multiplier mapping
_SHADING_RATE_MULTIPLIERS = {
    "FULL": 1.0,
    "HALF_X": 0.5,
    "HALF_Y": 0.5,
    "HALF": 0.25,
    "QUARTER_X": 0.25,
    "QUARTER_Y": 0.25,
    "QUARTER": 0.0625,
}


def shading_rate_to_int(rate: "ShadingRate") -> int:
    """Convert shading rate enum to VRS integer value.

    The integer values correspond to the VRS tile size encoding
    used by graphics APIs.

    Args:
        rate: ShadingRate enum value

    Returns:
        Integer VRS value (0-6)
    """
    return _SHADING_RATE_VALUES.get(rate.name, 0)


def get_rate_multiplier(rate: "ShadingRate") -> float:
    """Get pixel coverage multiplier for a shading rate.

    Returns the fraction of pixels that are actually shaded at
    this rate compared to full resolution.

    Args:
        rate: ShadingRate enum value

    Returns:
        Multiplier from 0.0 to 1.0
    """
    return _SHADING_RATE_MULTIPLIERS.get(rate.name, 1.0)


class ShadingRateUtils:
    """Utilities for VRS shading rate operations.

    This class provides static methods for working with VRS shading rates.
    Can be used as a mixin or via static method calls.
    """

    @staticmethod
    def shading_rate_to_int(rate: "ShadingRate") -> int:
        """Convert shading rate enum to VRS integer value.

        Args:
            rate: ShadingRate enum value

        Returns:
            Integer VRS value (0-6)
        """
        return shading_rate_to_int(rate)

    @staticmethod
    def get_rate_multiplier(rate: "ShadingRate") -> float:
        """Get pixel coverage multiplier for a shading rate.

        Args:
            rate: ShadingRate enum value

        Returns:
            Multiplier from 0.0 to 1.0
        """
        return get_rate_multiplier(rate)
