"""
Variable Refresh Rate (VRR) support.

Provides VRR detection and management for adaptive sync technologies
like FreeSync, G-Sync, and HDMI VRR.
"""

from dataclasses import dataclass
from enum import Enum, auto

from ..constants import VRR_DEFAULT_MIN_HZ, VRR_DEFAULT_MAX_HZ, VRR_DEFAULT_FIXED_HZ


class VRRType(Enum):
    """Variable refresh rate technology types."""
    NONE = auto()
    FREESYNC = auto()
    GSYNC = auto()
    GSYNC_COMPATIBLE = auto()
    HDMI_VRR = auto()
    ADAPTIVE_SYNC = auto()


@dataclass(slots=True)
class RefreshRange:
    """Refresh rate range for VRR."""
    min_hz: int
    max_hz: int


class VariableRefresh:
    """
    Variable refresh rate management.

    Provides VRR detection and control with a headless backend for testing.
    """

    __slots__ = (
        "_supported",
        "_enabled",
        "_vrr_type",
        "_refresh_range",
    )

    def __init__(self, simulate_vrr: bool = False):
        """
        Initialize VRR manager.

        Args:
            simulate_vrr: Simulate VRR support for testing
        """
        if simulate_vrr:
            self._supported = True
            self._vrr_type = VRRType.FREESYNC
            self._refresh_range = RefreshRange(min_hz=VRR_DEFAULT_MIN_HZ, max_hz=VRR_DEFAULT_MAX_HZ)
        else:
            self._supported = False
            self._vrr_type = VRRType.NONE
            self._refresh_range = RefreshRange(min_hz=VRR_DEFAULT_FIXED_HZ, max_hz=VRR_DEFAULT_FIXED_HZ)

        self._enabled = False

    def is_supported(self) -> bool:
        """
        Check if VRR is supported on this instance.

        Returns:
            True if VRR is supported
        """
        return self._supported

    def enable(self, enabled: bool) -> bool:
        """
        Enable or disable VRR.

        Args:
            enabled: True to enable VRR, False to disable

        Returns:
            True if VRR state was changed successfully
        """
        if not self._supported and enabled:
            return False

        self._enabled = enabled
        return True

    def get_range(self) -> RefreshRange:
        """
        Get the supported refresh rate range.

        Returns:
            RefreshRange object with min/max refresh rates
        """
        return self._refresh_range

    @property
    def vrr_type(self) -> VRRType:
        """Get the VRR technology type."""
        return self._vrr_type

    @property
    def enabled(self) -> bool:
        """Check if VRR is currently enabled."""
        return self._enabled

    @property
    def supported(self) -> bool:
        """Check if VRR is supported."""
        return self._supported
