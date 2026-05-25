"""Platform target definitions and capabilities.

Provides platform-specific build targets for Windows, Linux, Mac,
mobile platforms (Android, iOS), and consoles (PS5, Xbox, Switch).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Type
import platform
import sys


class Platform(Enum):
    """Supported platforms."""
    WINDOWS = auto()
    LINUX = auto()
    MACOS = auto()
    ANDROID = auto()
    IOS = auto()
    PS5 = auto()
    XBOX_SERIES = auto()
    SWITCH = auto()
    WEBGL = auto()
    STADIA = auto()  # Legacy/deprecated


class Architecture(Enum):
    """CPU architectures."""
    X86 = auto()
    X64 = auto()
    ARM32 = auto()
    ARM64 = auto()
    WASM = auto()
    UNIVERSAL = auto()  # Fat binaries (macOS)


class GraphicsAPI(Enum):
    """Graphics APIs."""
    DIRECTX_11 = auto()
    DIRECTX_12 = auto()
    VULKAN = auto()
    METAL = auto()
    OPENGL = auto()
    OPENGL_ES = auto()
    WEBGPU = auto()
    GNM = auto()        # PlayStation
    GXM = auto()        # PlayStation Vita (legacy)
    NVN = auto()        # Nintendo Switch


@dataclass
class PlatformCapabilities:
    """Platform capability flags and limits."""
    # Graphics
    max_texture_size: int = 4096
    supports_compute: bool = True
    supports_raytracing: bool = False
    supports_mesh_shaders: bool = False
    supports_variable_rate_shading: bool = False
    graphics_apis: List[GraphicsAPI] = field(default_factory=list)

    # Memory
    max_addressable_memory: int = 8 * 1024 * 1024 * 1024  # 8 GB
    recommended_memory: int = 4 * 1024 * 1024 * 1024      # 4 GB

    # Storage
    supports_async_io: bool = True
    supports_memory_mapped_files: bool = True

    # Threading
    max_threads: int = 16
    supports_fiber: bool = False

    # Audio
    max_audio_channels: int = 256
    supports_spatial_audio: bool = True

    # Features
    supports_touch: bool = False
    supports_gamepad: bool = True
    supports_keyboard: bool = True
    supports_mouse: bool = True
    supports_vr: bool = False
    supports_achievements: bool = True
    supports_cloud_saves: bool = True


@dataclass
class PlatformTarget(ABC):
    """Abstract base for platform targets."""
    name: str
    platform: Platform
    architecture: Architecture
    capabilities: PlatformCapabilities
    sdk_version: str = ""
    min_os_version: str = ""
    toolchain: str = ""

    @abstractmethod
    def get_compiler_flags(self) -> List[str]:
        """Get platform-specific compiler flags."""
        pass

    @abstractmethod
    def get_linker_flags(self) -> List[str]:
        """Get platform-specific linker flags."""
        pass

    @abstractmethod
    def get_defines(self) -> Dict[str, str]:
        """Get platform-specific preprocessor defines."""
        pass

    @abstractmethod
    def get_executable_extension(self) -> str:
        """Get the executable file extension."""
        pass

    @abstractmethod
    def get_shared_library_extension(self) -> str:
        """Get the shared library file extension."""
        pass

    @abstractmethod
    def get_static_library_extension(self) -> str:
        """Get the static library file extension."""
        pass

    def get_texture_format(self) -> str:
        """Get preferred texture compression format."""
        return "DXT"

    def get_audio_format(self) -> str:
        """Get preferred audio format."""
        return "OGG"

    def validate_sdk(self) -> bool:
        """Validate SDK installation."""
        return True  # Override in subclasses


@dataclass
class WindowsTarget(PlatformTarget):
    """Windows platform target."""

    def __init__(self, architecture: Architecture = Architecture.X64):
        caps = PlatformCapabilities(
            max_texture_size=16384,
            supports_raytracing=True,
            supports_mesh_shaders=True,
            supports_variable_rate_shading=True,
            graphics_apis=[GraphicsAPI.DIRECTX_12, GraphicsAPI.DIRECTX_11, GraphicsAPI.VULKAN],
            supports_fiber=True,
            supports_vr=True,
        )
        super().__init__(
            name="Windows",
            platform=Platform.WINDOWS,
            architecture=architecture,
            capabilities=caps,
            sdk_version="10.0.22621.0",
            min_os_version="10.0",
            toolchain="MSVC",
        )

    def get_compiler_flags(self) -> List[str]:
        flags = ["/W4", "/MP", "/permissive-", "/Zc:__cplusplus"]
        if self.architecture == Architecture.X64:
            flags.append("/arch:AVX2")
        return flags

    def get_linker_flags(self) -> List[str]:
        return ["/DYNAMICBASE", "/NXCOMPAT", "/HIGHENTROPYVA"]

    def get_defines(self) -> Dict[str, str]:
        defines = {
            "WIN32": "1",
            "_WINDOWS": "1",
            "UNICODE": "1",
            "_UNICODE": "1",
            "NOMINMAX": "1",
        }
        if self.architecture == Architecture.X64:
            defines["WIN64"] = "1"
        return defines

    def get_executable_extension(self) -> str:
        return ".exe"

    def get_shared_library_extension(self) -> str:
        return ".dll"

    def get_static_library_extension(self) -> str:
        return ".lib"

    def get_texture_format(self) -> str:
        return "BC7"


@dataclass
class LinuxTarget(PlatformTarget):
    """Linux platform target."""

    def __init__(self, architecture: Architecture = Architecture.X64):
        caps = PlatformCapabilities(
            max_texture_size=16384,
            supports_raytracing=True,
            supports_mesh_shaders=True,
            graphics_apis=[GraphicsAPI.VULKAN, GraphicsAPI.OPENGL],
            supports_vr=True,
        )
        super().__init__(
            name="Linux",
            platform=Platform.LINUX,
            architecture=architecture,
            capabilities=caps,
            min_os_version="Ubuntu 20.04",
            toolchain="GCC/Clang",
        )

    def get_compiler_flags(self) -> List[str]:
        flags = ["-Wall", "-Wextra", "-pthread", "-fPIC"]
        if self.architecture == Architecture.X64:
            flags.extend(["-march=x86-64-v2", "-mavx2"])
        return flags

    def get_linker_flags(self) -> List[str]:
        return ["-pthread", "-Wl,-z,now", "-Wl,-z,relro"]

    def get_defines(self) -> Dict[str, str]:
        defines = {
            "LINUX": "1",
            "_GNU_SOURCE": "1",
        }
        if self.architecture == Architecture.X64:
            defines["__x86_64__"] = "1"
        return defines

    def get_executable_extension(self) -> str:
        return ""

    def get_shared_library_extension(self) -> str:
        return ".so"

    def get_static_library_extension(self) -> str:
        return ".a"

    def get_texture_format(self) -> str:
        return "BC7"


@dataclass
class MacTarget(PlatformTarget):
    """macOS platform target."""

    def __init__(self, architecture: Architecture = Architecture.ARM64):
        caps = PlatformCapabilities(
            max_texture_size=16384,
            supports_raytracing=True,
            supports_mesh_shaders=True,
            graphics_apis=[GraphicsAPI.METAL, GraphicsAPI.VULKAN],
            supports_vr=True,
        )
        super().__init__(
            name="macOS",
            platform=Platform.MACOS,
            architecture=architecture,
            capabilities=caps,
            sdk_version="14.0",
            min_os_version="12.0",
            toolchain="Clang/Apple",
        )

    def get_compiler_flags(self) -> List[str]:
        flags = ["-Wall", "-Wextra", "-fPIC"]
        if self.architecture == Architecture.ARM64:
            flags.append("-arch arm64")
        elif self.architecture == Architecture.X64:
            flags.append("-arch x86_64")
        elif self.architecture == Architecture.UNIVERSAL:
            flags.extend(["-arch arm64", "-arch x86_64"])
        return flags

    def get_linker_flags(self) -> List[str]:
        flags = [f"-mmacosx-version-min={self.min_os_version}"]
        if self.architecture == Architecture.ARM64:
            flags.append("-arch arm64")
        elif self.architecture == Architecture.X64:
            flags.append("-arch x86_64")
        elif self.architecture == Architecture.UNIVERSAL:
            flags.extend(["-arch arm64", "-arch x86_64"])
        return flags

    def get_defines(self) -> Dict[str, str]:
        return {
            "__APPLE__": "1",
            "MACOS": "1",
        }

    def get_executable_extension(self) -> str:
        return ""

    def get_shared_library_extension(self) -> str:
        return ".dylib"

    def get_static_library_extension(self) -> str:
        return ".a"

    def get_texture_format(self) -> str:
        return "ASTC"


@dataclass
class AndroidTarget(PlatformTarget):
    """Android platform target."""

    def __init__(self, architecture: Architecture = Architecture.ARM64):
        caps = PlatformCapabilities(
            max_texture_size=4096,
            supports_raytracing=False,
            supports_mesh_shaders=False,
            graphics_apis=[GraphicsAPI.VULKAN, GraphicsAPI.OPENGL_ES],
            max_addressable_memory=4 * 1024 * 1024 * 1024,
            supports_touch=True,
            supports_keyboard=False,
            supports_mouse=False,
            supports_vr=True,
        )
        super().__init__(
            name="Android",
            platform=Platform.ANDROID,
            architecture=architecture,
            capabilities=caps,
            sdk_version="34",
            min_os_version="API 26",
            toolchain="NDK",
        )
        self.ndk_version = "r26"

    def get_compiler_flags(self) -> List[str]:
        flags = ["-Wall", "-fPIC", "-DANDROID"]
        if self.architecture == Architecture.ARM64:
            flags.append("-march=armv8-a")
        elif self.architecture == Architecture.ARM32:
            flags.append("-march=armv7-a")
        return flags

    def get_linker_flags(self) -> List[str]:
        return ["-llog", "-landroid", "-lEGL", "-lGLESv3"]

    def get_defines(self) -> Dict[str, str]:
        return {
            "ANDROID": "1",
            "__ANDROID__": "1",
            f"__ANDROID_API__": "26",
        }

    def get_executable_extension(self) -> str:
        return ""

    def get_shared_library_extension(self) -> str:
        return ".so"

    def get_static_library_extension(self) -> str:
        return ".a"

    def get_texture_format(self) -> str:
        return "ASTC"

    def get_audio_format(self) -> str:
        return "AAC"


@dataclass
class iOSTarget(PlatformTarget):
    """iOS platform target."""

    def __init__(self, architecture: Architecture = Architecture.ARM64):
        caps = PlatformCapabilities(
            max_texture_size=4096,
            supports_raytracing=True,
            supports_mesh_shaders=True,
            graphics_apis=[GraphicsAPI.METAL],
            max_addressable_memory=4 * 1024 * 1024 * 1024,
            supports_touch=True,
            supports_gamepad=True,
            supports_keyboard=False,
            supports_mouse=False,
        )
        super().__init__(
            name="iOS",
            platform=Platform.IOS,
            architecture=architecture,
            capabilities=caps,
            sdk_version="17.0",
            min_os_version="15.0",
            toolchain="Clang/Apple",
        )

    def get_compiler_flags(self) -> List[str]:
        return [
            "-Wall", "-Wextra", "-fPIC",
            "-arch arm64",
            f"-miphoneos-version-min={self.min_os_version}",
        ]

    def get_linker_flags(self) -> List[str]:
        return [
            "-arch arm64",
            f"-miphoneos-version-min={self.min_os_version}",
            "-framework UIKit",
            "-framework Metal",
            "-framework MetalKit",
        ]

    def get_defines(self) -> Dict[str, str]:
        return {
            "__APPLE__": "1",
            "IOS": "1",
            "TARGET_OS_IPHONE": "1",
        }

    def get_executable_extension(self) -> str:
        return ""

    def get_shared_library_extension(self) -> str:
        return ".dylib"

    def get_static_library_extension(self) -> str:
        return ".a"

    def get_texture_format(self) -> str:
        return "ASTC"

    def get_audio_format(self) -> str:
        return "AAC"


@dataclass
class PS5Target(PlatformTarget):
    """PlayStation 5 platform target."""

    def __init__(self):
        caps = PlatformCapabilities(
            max_texture_size=16384,
            supports_raytracing=True,
            supports_mesh_shaders=True,
            supports_variable_rate_shading=True,
            graphics_apis=[GraphicsAPI.GNM],
            max_addressable_memory=16 * 1024 * 1024 * 1024,
            max_threads=16,
            supports_vr=True,
        )
        super().__init__(
            name="PlayStation 5",
            platform=Platform.PS5,
            architecture=Architecture.X64,
            capabilities=caps,
            sdk_version="8.0",
            toolchain="Prospero SDK",
        )

    def get_compiler_flags(self) -> List[str]:
        return ["-Wall", "-Wextra", "-fPIC", "-march=znver2"]

    def get_linker_flags(self) -> List[str]:
        return ["-lSceGnm", "-lSceAudioOut", "-lSceUserService"]

    def get_defines(self) -> Dict[str, str]:
        return {
            "__PROSPERO__": "1",
            "PS5": "1",
            "ORBIS": "1",
        }

    def get_executable_extension(self) -> str:
        return ".elf"

    def get_shared_library_extension(self) -> str:
        return ".prx"

    def get_static_library_extension(self) -> str:
        return ".a"

    def get_texture_format(self) -> str:
        return "GNF"

    def get_audio_format(self) -> str:
        return "AT9"


@dataclass
class XboxSeriesTarget(PlatformTarget):
    """Xbox Series X|S platform target."""

    def __init__(self, is_series_x: bool = True):
        max_mem = 16 * 1024 * 1024 * 1024 if is_series_x else 8 * 1024 * 1024 * 1024
        caps = PlatformCapabilities(
            max_texture_size=16384 if is_series_x else 8192,
            supports_raytracing=True,
            supports_mesh_shaders=True,
            supports_variable_rate_shading=True,
            graphics_apis=[GraphicsAPI.DIRECTX_12],
            max_addressable_memory=max_mem,
            max_threads=16 if is_series_x else 8,
        )
        name = "Xbox Series X" if is_series_x else "Xbox Series S"
        super().__init__(
            name=name,
            platform=Platform.XBOX_SERIES,
            architecture=Architecture.X64,
            capabilities=caps,
            sdk_version="GDK 2023.10",
            toolchain="MSVC/GDK",
        )
        self.is_series_x = is_series_x

    def get_compiler_flags(self) -> List[str]:
        return ["/W4", "/MP", "/arch:AVX2"]

    def get_linker_flags(self) -> List[str]:
        return ["/DYNAMICBASE"]

    def get_defines(self) -> Dict[str, str]:
        defines = {
            "_GAMING_XBOX": "1",
            "_GAMING_XBOX_SCARLETT": "1",
        }
        if self.is_series_x:
            defines["XBOX_SERIES_X"] = "1"
        else:
            defines["XBOX_SERIES_S"] = "1"
        return defines

    def get_executable_extension(self) -> str:
        return ".exe"

    def get_shared_library_extension(self) -> str:
        return ".dll"

    def get_static_library_extension(self) -> str:
        return ".lib"

    def get_texture_format(self) -> str:
        return "BC7"

    def get_audio_format(self) -> str:
        return "XMA2"


@dataclass
class SwitchTarget(PlatformTarget):
    """Nintendo Switch platform target."""

    def __init__(self):
        caps = PlatformCapabilities(
            max_texture_size=4096,
            supports_raytracing=False,
            supports_mesh_shaders=False,
            graphics_apis=[GraphicsAPI.NVN, GraphicsAPI.VULKAN],
            max_addressable_memory=4 * 1024 * 1024 * 1024,
            max_threads=4,
            supports_touch=True,
        )
        super().__init__(
            name="Nintendo Switch",
            platform=Platform.SWITCH,
            architecture=Architecture.ARM64,
            capabilities=caps,
            sdk_version="17.0.0",
            toolchain="Nintendo SDK",
        )

    def get_compiler_flags(self) -> List[str]:
        return ["-Wall", "-Wextra", "-fPIC", "-march=armv8-a+crc+crypto"]

    def get_linker_flags(self) -> List[str]:
        return ["-lnn_gfx", "-lnn_audio"]

    def get_defines(self) -> Dict[str, str]:
        return {
            "__SWITCH__": "1",
            "NN_NINTENDO_SDK": "1",
        }

    def get_executable_extension(self) -> str:
        return ".nso"

    def get_shared_library_extension(self) -> str:
        return ".nro"

    def get_static_library_extension(self) -> str:
        return ".a"

    def get_texture_format(self) -> str:
        return "ASTC"

    def get_audio_format(self) -> str:
        return "OPUS"


# Platform registry
_PLATFORM_TARGETS: Dict[Platform, Type[PlatformTarget]] = {
    Platform.WINDOWS: WindowsTarget,
    Platform.LINUX: LinuxTarget,
    Platform.MACOS: MacTarget,
    Platform.ANDROID: AndroidTarget,
    Platform.IOS: iOSTarget,
    Platform.PS5: PS5Target,
    Platform.XBOX_SERIES: XboxSeriesTarget,
    Platform.SWITCH: SwitchTarget,
}


def get_current_platform() -> Platform:
    """Detect and return the current platform."""
    system = platform.system().lower()
    if system == "windows":
        return Platform.WINDOWS
    elif system == "linux":
        # Check for Android
        try:
            with open("/system/build.prop", "r") as f:
                return Platform.ANDROID
        except FileNotFoundError:
            return Platform.LINUX
    elif system == "darwin":
        return Platform.MACOS
    else:
        return Platform.LINUX  # Default fallback


def get_supported_platforms() -> List[Platform]:
    """Get list of all supported platforms."""
    return list(_PLATFORM_TARGETS.keys())


def create_target(platform: Platform, **kwargs) -> PlatformTarget:
    """Create a platform target instance."""
    target_class = _PLATFORM_TARGETS.get(platform)
    if target_class is None:
        raise ValueError(f"Unsupported platform: {platform}")
    return target_class(**kwargs)
