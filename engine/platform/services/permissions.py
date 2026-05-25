"""Platform permissions system (stub implementation)."""
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)


class Permission(Enum):
    """Permission types."""
    STORAGE = auto()
    CAMERA = auto()
    MICROPHONE = auto()
    LOCATION = auto()
    NETWORK = auto()


class PermissionStatus(Enum):
    """Permission status."""
    GRANTED = auto()
    DENIED = auto()
    NOT_REQUESTED = auto()


def request(permission: Permission) -> PermissionStatus:
    """
    Request permission from user.

    Args:
        permission: Permission to request

    Returns:
        Current permission status
    """
    # Stub implementation - grant all permissions
    # In real implementation, would show platform-specific permission dialog
    logger.debug(f"Stub permission system: auto-granting {permission}")
    return PermissionStatus.GRANTED


def check(permission: Permission) -> PermissionStatus:
    """
    Check permission status.

    Args:
        permission: Permission to check

    Returns:
        Current permission status
    """
    # Stub implementation - all permissions granted
    logger.debug(f"Stub permission system: auto-granting {permission}")
    return PermissionStatus.GRANTED
