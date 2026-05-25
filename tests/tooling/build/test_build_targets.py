"""Tests for platform target definitions."""
import pytest
from engine.tooling.build.build_targets import (
    Platform,
    Architecture,
    GraphicsAPI,
    PlatformCapabilities,
    PlatformTarget,
    WindowsTarget,
    LinuxTarget,
    MacTarget,
    AndroidTarget,
    iOSTarget,
    PS5Target,
    XboxSeriesTarget,
    SwitchTarget,
    get_current_platform,
    get_supported_platforms,
    create_target,
)


class TestPlatformEnum:
    """Tests for Platform enum."""

    def test_all_platforms_exist(self):
        """Test all platforms exist."""
        assert Platform.WINDOWS
        assert Platform.LINUX
        assert Platform.MACOS
        assert Platform.ANDROID
        assert Platform.IOS
        assert Platform.PS5
        assert Platform.XBOX_SERIES
        assert Platform.SWITCH
        assert Platform.WEBGL

    def test_platform_uniqueness(self):
        """Test platforms are unique."""
        values = [p.value for p in Platform]
        assert len(values) == len(set(values))


class TestArchitectureEnum:
    """Tests for Architecture enum."""

    def test_all_architectures_exist(self):
        """Test all architectures exist."""
        assert Architecture.X86
        assert Architecture.X64
        assert Architecture.ARM32
        assert Architecture.ARM64
        assert Architecture.WASM
        assert Architecture.UNIVERSAL


class TestGraphicsAPIEnum:
    """Tests for GraphicsAPI enum."""

    def test_all_graphics_apis_exist(self):
        """Test all graphics APIs exist."""
        assert GraphicsAPI.DIRECTX_11
        assert GraphicsAPI.DIRECTX_12
        assert GraphicsAPI.VULKAN
        assert GraphicsAPI.METAL
        assert GraphicsAPI.OPENGL
        assert GraphicsAPI.OPENGL_ES
        assert GraphicsAPI.WEBGPU


class TestPlatformCapabilities:
    """Tests for PlatformCapabilities dataclass."""

    def test_default_values(self):
        """Test default capability values."""
        caps = PlatformCapabilities()
        assert caps.max_texture_size == 4096
        assert caps.supports_compute is True
        assert caps.supports_raytracing is False
        assert caps.supports_async_io is True
        assert caps.supports_keyboard is True

    def test_custom_values(self):
        """Test custom capability values."""
        caps = PlatformCapabilities(
            max_texture_size=16384,
            supports_raytracing=True,
            supports_vr=True,
        )
        assert caps.max_texture_size == 16384
        assert caps.supports_raytracing is True
        assert caps.supports_vr is True


class TestWindowsTarget:
    """Tests for WindowsTarget."""

    def test_default_creation(self):
        """Test creating default Windows target."""
        target = WindowsTarget()
        assert target.name == "Windows"
        assert target.platform == Platform.WINDOWS
        assert target.architecture == Architecture.X64

    def test_x86_architecture(self):
        """Test creating x86 Windows target."""
        target = WindowsTarget(architecture=Architecture.X86)
        assert target.architecture == Architecture.X86

    def test_compiler_flags(self):
        """Test compiler flags."""
        target = WindowsTarget()
        flags = target.get_compiler_flags()
        assert "/W4" in flags
        assert "/MP" in flags

    def test_linker_flags(self):
        """Test linker flags."""
        target = WindowsTarget()
        flags = target.get_linker_flags()
        assert "/DYNAMICBASE" in flags

    def test_defines(self):
        """Test preprocessor defines."""
        target = WindowsTarget()
        defines = target.get_defines()
        assert "WIN32" in defines
        assert "UNICODE" in defines
        assert "NOMINMAX" in defines

    def test_file_extensions(self):
        """Test file extensions."""
        target = WindowsTarget()
        assert target.get_executable_extension() == ".exe"
        assert target.get_shared_library_extension() == ".dll"
        assert target.get_static_library_extension() == ".lib"

    def test_capabilities(self):
        """Test Windows capabilities."""
        target = WindowsTarget()
        assert target.capabilities.supports_raytracing is True
        assert target.capabilities.supports_mesh_shaders is True
        assert GraphicsAPI.DIRECTX_12 in target.capabilities.graphics_apis


class TestLinuxTarget:
    """Tests for LinuxTarget."""

    def test_default_creation(self):
        """Test creating default Linux target."""
        target = LinuxTarget()
        assert target.name == "Linux"
        assert target.platform == Platform.LINUX
        assert target.architecture == Architecture.X64

    def test_compiler_flags(self):
        """Test compiler flags."""
        target = LinuxTarget()
        flags = target.get_compiler_flags()
        assert "-Wall" in flags
        assert "-pthread" in flags
        assert "-fPIC" in flags

    def test_defines(self):
        """Test preprocessor defines."""
        target = LinuxTarget()
        defines = target.get_defines()
        assert "LINUX" in defines

    def test_file_extensions(self):
        """Test file extensions."""
        target = LinuxTarget()
        assert target.get_executable_extension() == ""
        assert target.get_shared_library_extension() == ".so"
        assert target.get_static_library_extension() == ".a"

    def test_capabilities(self):
        """Test Linux capabilities."""
        target = LinuxTarget()
        assert GraphicsAPI.VULKAN in target.capabilities.graphics_apis


class TestMacTarget:
    """Tests for MacTarget."""

    def test_default_creation(self):
        """Test creating default Mac target."""
        target = MacTarget()
        assert target.name == "macOS"
        assert target.platform == Platform.MACOS
        assert target.architecture == Architecture.ARM64

    def test_universal_architecture(self):
        """Test creating universal Mac target."""
        target = MacTarget(architecture=Architecture.UNIVERSAL)
        flags = target.get_compiler_flags()
        assert "-arch arm64" in flags
        assert "-arch x86_64" in flags

    def test_file_extensions(self):
        """Test file extensions."""
        target = MacTarget()
        assert target.get_shared_library_extension() == ".dylib"

    def test_texture_format(self):
        """Test texture format."""
        target = MacTarget()
        assert target.get_texture_format() == "ASTC"

    def test_capabilities(self):
        """Test Mac capabilities."""
        target = MacTarget()
        assert GraphicsAPI.METAL in target.capabilities.graphics_apis


class TestAndroidTarget:
    """Tests for AndroidTarget."""

    def test_default_creation(self):
        """Test creating default Android target."""
        target = AndroidTarget()
        assert target.platform == Platform.ANDROID
        assert target.architecture == Architecture.ARM64

    def test_arm32_architecture(self):
        """Test creating ARM32 Android target."""
        target = AndroidTarget(architecture=Architecture.ARM32)
        flags = target.get_compiler_flags()
        assert "-march=armv7-a" in flags

    def test_defines(self):
        """Test preprocessor defines."""
        target = AndroidTarget()
        defines = target.get_defines()
        assert "ANDROID" in defines

    def test_capabilities(self):
        """Test Android capabilities."""
        target = AndroidTarget()
        assert target.capabilities.supports_touch is True
        assert target.capabilities.supports_keyboard is False

    def test_audio_format(self):
        """Test audio format."""
        target = AndroidTarget()
        assert target.get_audio_format() == "AAC"


class TestiOSTarget:
    """Tests for iOSTarget."""

    def test_default_creation(self):
        """Test creating default iOS target."""
        target = iOSTarget()
        assert target.platform == Platform.IOS
        assert target.architecture == Architecture.ARM64

    def test_defines(self):
        """Test preprocessor defines."""
        target = iOSTarget()
        defines = target.get_defines()
        assert "IOS" in defines
        assert "__APPLE__" in defines

    def test_capabilities(self):
        """Test iOS capabilities."""
        target = iOSTarget()
        assert target.capabilities.supports_touch is True
        assert GraphicsAPI.METAL in target.capabilities.graphics_apis


class TestPS5Target:
    """Tests for PS5Target."""

    def test_creation(self):
        """Test creating PS5 target."""
        target = PS5Target()
        assert target.platform == Platform.PS5
        assert target.architecture == Architecture.X64

    def test_defines(self):
        """Test preprocessor defines."""
        target = PS5Target()
        defines = target.get_defines()
        assert "PS5" in defines
        assert "__PROSPERO__" in defines

    def test_file_extensions(self):
        """Test file extensions."""
        target = PS5Target()
        assert target.get_executable_extension() == ".elf"
        assert target.get_shared_library_extension() == ".prx"

    def test_capabilities(self):
        """Test PS5 capabilities."""
        target = PS5Target()
        assert target.capabilities.supports_raytracing is True
        assert target.capabilities.supports_vr is True


class TestXboxSeriesTarget:
    """Tests for XboxSeriesTarget."""

    def test_series_x_creation(self):
        """Test creating Xbox Series X target."""
        target = XboxSeriesTarget(is_series_x=True)
        assert "Xbox Series X" in target.name

    def test_series_s_creation(self):
        """Test creating Xbox Series S target."""
        target = XboxSeriesTarget(is_series_x=False)
        assert "Xbox Series S" in target.name

    def test_defines(self):
        """Test preprocessor defines."""
        target_x = XboxSeriesTarget(is_series_x=True)
        defines_x = target_x.get_defines()
        assert "XBOX_SERIES_X" in defines_x

        target_s = XboxSeriesTarget(is_series_x=False)
        defines_s = target_s.get_defines()
        assert "XBOX_SERIES_S" in defines_s

    def test_capabilities_difference(self):
        """Test capability differences between X and S."""
        target_x = XboxSeriesTarget(is_series_x=True)
        target_s = XboxSeriesTarget(is_series_x=False)

        assert target_x.capabilities.max_threads > target_s.capabilities.max_threads


class TestSwitchTarget:
    """Tests for SwitchTarget."""

    def test_creation(self):
        """Test creating Switch target."""
        target = SwitchTarget()
        assert target.platform == Platform.SWITCH
        assert target.architecture == Architecture.ARM64

    def test_defines(self):
        """Test preprocessor defines."""
        target = SwitchTarget()
        defines = target.get_defines()
        assert "__SWITCH__" in defines

    def test_capabilities(self):
        """Test Switch capabilities."""
        target = SwitchTarget()
        assert target.capabilities.supports_touch is True
        assert target.capabilities.supports_raytracing is False


class TestPlatformUtilities:
    """Tests for platform utility functions."""

    def test_get_current_platform(self):
        """Test getting current platform."""
        platform = get_current_platform()
        assert platform in Platform

    def test_get_supported_platforms(self):
        """Test getting supported platforms."""
        platforms = get_supported_platforms()
        assert Platform.WINDOWS in platforms
        assert Platform.LINUX in platforms
        assert Platform.MACOS in platforms

    def test_create_target_windows(self):
        """Test creating Windows target via factory."""
        target = create_target(Platform.WINDOWS)
        assert isinstance(target, WindowsTarget)

    def test_create_target_linux(self):
        """Test creating Linux target via factory."""
        target = create_target(Platform.LINUX)
        assert isinstance(target, LinuxTarget)

    def test_create_target_invalid(self):
        """Test creating invalid target raises error."""
        with pytest.raises(ValueError):
            create_target(Platform.STADIA)  # No provider for deprecated platform
