"""Platform detection utilities."""
from dataclasses import dataclass
from enum import Enum, auto
import sys
import platform


class PlatformType(Enum):
    """Platform type enumeration."""
    WINDOWS = auto()
    LINUX = auto()
    MACOS = auto()
    IOS = auto()
    ANDROID = auto()
    WEB = auto()
    PS5 = auto()
    XBOX = auto()
    SWITCH = auto()


@dataclass
class PlatformInfo:
    """Platform information."""
    type: PlatformType
    name: str
    version: str
    arch: str
    is_console: bool
    is_mobile: bool
    is_desktop: bool


def detect() -> PlatformInfo:
    """
    Detect current platform.

    Returns:
        PlatformInfo describing the current platform
    """
    system = platform.system()
    machine = platform.machine()
    release = platform.release()

    # Detect platform type
    if system == "Windows":
        platform_type = PlatformType.WINDOWS
        name = "Windows"
        is_desktop = True
        is_mobile = False
        is_console = False
    elif system == "Linux":
        # Could be Linux desktop, Android, or console
        # For now, assume desktop Linux
        # (In real implementation, would check for Android, PS5, etc.)
        platform_type = PlatformType.LINUX
        name = "Linux"
        is_desktop = True
        is_mobile = False
        is_console = False
    elif system == "Darwin":
        # macOS or iOS
        if machine.startswith("iP"):
            platform_type = PlatformType.IOS
            name = "iOS"
            is_desktop = False
            is_mobile = True
            is_console = False
        else:
            platform_type = PlatformType.MACOS
            name = "macOS"
            is_desktop = True
            is_mobile = False
            is_console = False
    else:
        # Default to Linux for unknown platforms
        platform_type = PlatformType.LINUX
        name = system
        is_desktop = True
        is_mobile = False
        is_console = False

    return PlatformInfo(
        type=platform_type,
        name=name,
        version=release,
        arch=machine,
        is_console=is_console,
        is_mobile=is_mobile,
        is_desktop=is_desktop
    )
