"""Blackbox tests for the Platform Layer.

These tests verify PUBLIC API behavior ONLY. They do NOT depend on:
- Internal implementation details
- Private attributes or methods
- Actual hardware (uses null/mock backends)

Coverage Areas:
  1. BackendRegistry: Generic registry for pluggable backends
  2. Platform Bootstrap: Platform initialization and shutdown
  3. Platform Services: Platform detection, app lifecycle, permissions
  4. Audio System: Audio devices, spatial audio, backends
  5. Input System: Input manager, devices, events
  6. Window System: Window, display, cursor, HDR, VRR
  7. GPU System: Low latency features
  8. RHI Integration: Adapter, device, resources (uses null backend)
  9. Constants: Platform constants validation
  10. Thread Safety: Concurrent access patterns

Minimum test count: 150+
"""
import pytest
import threading
import time
from dataclasses import dataclass
from typing import Any
from concurrent.futures import ThreadPoolExecutor

# =============================================================================
# Section 1: BackendRegistry Tests (20 tests)
# =============================================================================

from engine.platform.registry import BackendRegistry


class DummyBackend:
    """Dummy backend for testing."""
    def __init__(self, value: int = 0):
        self.value = value


class AnotherBackend:
    """Another dummy backend for testing."""
    def __init__(self, name: str = "default"):
        self.name = name


class TestBackendRegistryBasics:
    """Test basic BackendRegistry operations."""

    def test_create_empty_registry(self):
        """Empty registry can be created."""
        registry: BackendRegistry[DummyBackend] = BackendRegistry()
        assert registry.list() == []
        assert registry.default() is None

    def test_register_single_backend(self):
        """Single backend can be registered."""
        registry: BackendRegistry[DummyBackend] = BackendRegistry()
        registry.register("test", DummyBackend)
        assert "test" in registry.list()

    def test_register_multiple_backends(self):
        """Multiple backends can be registered."""
        registry: BackendRegistry[DummyBackend] = BackendRegistry()
        registry.register("a", DummyBackend)
        registry.register("b", DummyBackend)
        registry.register("c", DummyBackend)
        backends = registry.list()
        assert len(backends) == 3
        assert "a" in backends
        assert "b" in backends
        assert "c" in backends

    def test_register_with_default(self):
        """Backend can be registered as default."""
        registry: BackendRegistry[DummyBackend] = BackendRegistry()
        registry.register("default", DummyBackend, set_default=True)
        assert registry.default() == "default"

    def test_register_changes_default(self):
        """Registering new default replaces old default."""
        registry: BackendRegistry[DummyBackend] = BackendRegistry()
        registry.register("first", DummyBackend, set_default=True)
        registry.register("second", DummyBackend, set_default=True)
        assert registry.default() == "second"

    def test_get_registered_backend(self):
        """Can get registered backend class."""
        registry: BackendRegistry[DummyBackend] = BackendRegistry()
        registry.register("test", DummyBackend)
        cls = registry.get("test")
        assert cls is DummyBackend

    def test_get_unregistered_backend_returns_none(self):
        """Get returns None for unregistered backend."""
        registry: BackendRegistry[DummyBackend] = BackendRegistry()
        assert registry.get("nonexistent") is None

    def test_list_returns_sorted(self):
        """List returns sorted backend names."""
        registry: BackendRegistry[DummyBackend] = BackendRegistry()
        registry.register("zebra", DummyBackend)
        registry.register("alpha", DummyBackend)
        registry.register("middle", DummyBackend)
        assert registry.list() == ["alpha", "middle", "zebra"]


class TestBackendRegistryCreate:
    """Test BackendRegistry.create() method."""

    def test_create_with_name(self):
        """Create backend by explicit name."""
        registry: BackendRegistry[DummyBackend] = BackendRegistry()
        registry.register("test", DummyBackend)
        instance = registry.create("test")
        assert isinstance(instance, DummyBackend)

    def test_create_with_default(self):
        """Create default backend when no name given."""
        registry: BackendRegistry[DummyBackend] = BackendRegistry()
        registry.register("default", DummyBackend, set_default=True)
        instance = registry.create()
        assert isinstance(instance, DummyBackend)

    def test_create_unknown_raises_valueerror(self):
        """Create raises ValueError for unknown backend."""
        registry: BackendRegistry[DummyBackend] = BackendRegistry()
        registry.register("known", DummyBackend, set_default=True)
        with pytest.raises(ValueError, match="Unknown backend"):
            registry.create("unknown")

    def test_create_no_default_raises_valueerror(self):
        """Create raises ValueError when no default set."""
        registry: BackendRegistry[DummyBackend] = BackendRegistry()
        registry.register("test", DummyBackend)  # Not set as default
        with pytest.raises(ValueError, match="No default backend"):
            registry.create()

    def test_create_with_args(self):
        """Create passes positional args to constructor."""
        registry: BackendRegistry[DummyBackend] = BackendRegistry()
        registry.register("test", DummyBackend, set_default=True)
        instance = registry.create(value=42)
        assert instance.value == 42

    def test_create_with_kwargs(self):
        """Create passes keyword args to constructor."""
        registry: BackendRegistry[DummyBackend] = BackendRegistry()
        registry.register("test", DummyBackend, set_default=True)
        # Pass constructor kwarg (value) not name
        instance = registry.create(value=42)
        assert instance.value == 42

    def test_create_explicit_name_overrides_default(self):
        """Explicit name takes precedence over default."""
        registry: BackendRegistry[DummyBackend] = BackendRegistry()
        registry.register("default", DummyBackend, set_default=True)
        registry.register("other", AnotherBackend)
        instance = registry.create("other")
        assert isinstance(instance, AnotherBackend)


class TestBackendRegistryThreadSafety:
    """Test thread safety of BackendRegistry."""

    def test_concurrent_registration(self):
        """Concurrent registrations are thread-safe."""
        registry: BackendRegistry[DummyBackend] = BackendRegistry()

        def register_backend(name: str):
            registry.register(name, DummyBackend)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(register_backend, f"backend_{i}") for i in range(100)]
            for f in futures:
                f.result()

        assert len(registry.list()) == 100

    def test_concurrent_get(self):
        """Concurrent gets are thread-safe."""
        registry: BackendRegistry[DummyBackend] = BackendRegistry()
        registry.register("test", DummyBackend)

        results = []

        def get_backend():
            cls = registry.get("test")
            results.append(cls is DummyBackend)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_backend) for _ in range(100)]
            for f in futures:
                f.result()

        assert all(results)

    def test_concurrent_create(self):
        """Concurrent creates are thread-safe."""
        registry: BackendRegistry[DummyBackend] = BackendRegistry()
        registry.register("test", DummyBackend, set_default=True)

        instances = []
        lock = threading.Lock()

        def create_backend():
            instance = registry.create()
            with lock:
                instances.append(instance)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_backend) for _ in range(100)]
            for f in futures:
                f.result()

        assert len(instances) == 100
        assert all(isinstance(i, DummyBackend) for i in instances)


# =============================================================================
# Section 2: Platform Bootstrap Tests (10 tests)
# =============================================================================

from engine.platform import (
    bootstrap_platform,
    shutdown_platform,
    create_graphics_device,
    get_lifecycle,
    get_platform_info,
    get_low_latency,
    BackendRegistry,
)


class TestPlatformBootstrap:
    """Test platform bootstrap and shutdown."""

    def test_bootstrap_can_be_called(self):
        """bootstrap_platform can be called without error."""
        # Note: May have already been called, which is fine
        bootstrap_platform()
        # If we get here, it succeeded

    def test_get_platform_info_returns_info(self):
        """get_platform_info returns platform information."""
        info = get_platform_info()
        assert info is not None
        # Check observable attributes exist
        assert hasattr(info, 'name')
        assert hasattr(info, 'version')
        assert hasattr(info, 'arch')

    def test_platform_info_name_is_string(self):
        """Platform info name is a non-empty string."""
        info = get_platform_info()
        assert isinstance(info.name, str)
        assert len(info.name) > 0

    def test_platform_info_arch_is_string(self):
        """Platform info arch is a non-empty string."""
        info = get_platform_info()
        assert isinstance(info.arch, str)
        assert len(info.arch) > 0

    def test_create_graphics_device_returns_device_or_none(self):
        """create_graphics_device returns device or None."""
        bootstrap_platform()
        device = create_graphics_device()
        # Either returns a device or None (headless environment)
        assert device is None or hasattr(device, 'shutdown')

    def test_get_lifecycle_returns_lifecycle_or_none(self):
        """get_lifecycle returns lifecycle manager or None."""
        bootstrap_platform()
        lifecycle = get_lifecycle()
        assert lifecycle is None or hasattr(lifecycle, 'shutdown')

    def test_get_low_latency_returns_low_latency_or_none(self):
        """get_low_latency returns low latency manager or None."""
        bootstrap_platform()
        low_latency = get_low_latency()
        # LowLatency uses is_available method, not is_supported
        assert low_latency is None or hasattr(low_latency, 'is_available')

    def test_shutdown_can_be_called(self):
        """shutdown_platform can be called without error."""
        # Shutdown and re-bootstrap for other tests
        shutdown_platform()
        bootstrap_platform()

    def test_multiple_bootstrap_is_idempotent(self):
        """Multiple bootstrap calls are idempotent."""
        bootstrap_platform()
        bootstrap_platform()
        bootstrap_platform()
        # No error raised

    def test_shutdown_after_shutdown_is_idempotent(self):
        """Multiple shutdown calls are idempotent."""
        shutdown_platform()
        shutdown_platform()
        bootstrap_platform()


# =============================================================================
# Section 3: Platform Services Tests (20 tests)
# =============================================================================

from engine.platform.services import (
    PlatformType,
    PlatformInfo,
    detect,
    AppState,
    AppLifecycle,
    Permission,
    PermissionStatus,
    request,
    check,
    ServiceType,
    ServiceProvider,
    NullServiceProvider,
    create_service_provider,
)


class TestPlatformDetection:
    """Test platform detection."""

    def test_detect_returns_platform_info(self):
        """detect() returns PlatformInfo."""
        info = detect()
        assert isinstance(info, PlatformInfo)

    def test_platform_info_has_name(self):
        """PlatformInfo has name attribute."""
        info = detect()
        assert hasattr(info, 'name')
        assert isinstance(info.name, str)

    def test_platform_info_has_version(self):
        """PlatformInfo has version attribute."""
        info = detect()
        assert hasattr(info, 'version')
        assert isinstance(info.version, str)

    def test_platform_info_has_arch(self):
        """PlatformInfo has arch attribute."""
        info = detect()
        assert hasattr(info, 'arch')
        assert isinstance(info.arch, str)

    def test_platform_type_enum_has_linux(self):
        """PlatformType enum has LINUX value."""
        assert hasattr(PlatformType, 'LINUX')

    def test_platform_type_enum_has_windows(self):
        """PlatformType enum has WINDOWS value."""
        assert hasattr(PlatformType, 'WINDOWS')

    def test_platform_type_enum_has_macos(self):
        """PlatformType enum has MACOS value."""
        assert hasattr(PlatformType, 'MACOS')


class TestAppLifecycle:
    """Test application lifecycle management."""

    def test_app_lifecycle_can_be_created(self):
        """AppLifecycle can be instantiated."""
        lifecycle = AppLifecycle()
        assert lifecycle is not None

    def test_app_lifecycle_has_shutdown_method(self):
        """AppLifecycle has shutdown method."""
        lifecycle = AppLifecycle()
        assert hasattr(lifecycle, 'shutdown')
        assert callable(lifecycle.shutdown)

    def test_app_state_enum_has_running(self):
        """AppState enum has RUNNING value."""
        assert hasattr(AppState, 'RUNNING')

    def test_app_state_enum_has_suspended(self):
        """AppState enum has SUSPENDED value."""
        assert hasattr(AppState, 'SUSPENDED')

    def test_app_state_enum_has_shutting_down(self):
        """AppState enum has SHUTTING_DOWN value."""
        assert hasattr(AppState, 'SHUTTING_DOWN')


class TestPermissions:
    """Test permission system."""

    def test_permission_enum_exists(self):
        """Permission enum exists."""
        assert Permission is not None

    def test_permission_status_enum_exists(self):
        """PermissionStatus enum exists."""
        assert PermissionStatus is not None

    def test_request_function_exists(self):
        """request function exists."""
        assert callable(request)

    def test_check_function_exists(self):
        """check function exists."""
        assert callable(check)


class TestServiceProvider:
    """Test service provider system."""

    def test_null_service_provider_can_be_created(self):
        """NullServiceProvider can be instantiated."""
        provider = NullServiceProvider()
        assert provider is not None

    def test_create_service_provider_returns_provider(self):
        """create_service_provider returns a ServiceProvider."""
        provider = create_service_provider()
        assert provider is not None
        assert isinstance(provider, ServiceProvider)

    def test_service_type_enum_exists(self):
        """ServiceType enum exists."""
        assert ServiceType is not None


# =============================================================================
# Section 4: Audio System Tests (25 tests)
# =============================================================================

from engine.platform.audio import (
    AudioDevice,
    AudioDeviceInfo,
    AudioDeviceType,
    AudioFormat,
    AudioBackend,
    NullAudioBackend,
    SpatialAudioEngine,
    SpatialAudioAPI,
    SpatialSource,
    SpatialListener,
    ReverbPreset,
    Vec3,
    register_backend,
    get_backend,
    get_default_backend,
    list_backends,
    create_backend,
)


class TestAudioDeviceType:
    """Test AudioDeviceType enum."""

    def test_device_type_has_playback(self):
        """AudioDeviceType has PLAYBACK value."""
        assert hasattr(AudioDeviceType, 'PLAYBACK')

    def test_device_type_has_capture(self):
        """AudioDeviceType has CAPTURE value."""
        assert hasattr(AudioDeviceType, 'CAPTURE')

    def test_device_type_values_are_distinct(self):
        """AudioDeviceType values are distinct."""
        assert AudioDeviceType.PLAYBACK != AudioDeviceType.CAPTURE


class TestAudioBackendRegistry:
    """Test audio backend registry."""

    def test_list_backends_returns_list(self):
        """list_backends returns a list."""
        backends = list_backends()
        assert isinstance(backends, list)

    def test_null_backend_is_registered(self):
        """Null audio backend is registered."""
        backends = list_backends()
        assert "null" in backends

    def test_get_backend_returns_backend(self):
        """get_backend returns backend class or None."""
        backend = get_backend("null")
        assert backend is not None

    def test_create_backend_returns_instance(self):
        """create_backend returns backend instance."""
        backend = create_backend("null")
        assert isinstance(backend, AudioBackend)


class TestNullAudioBackend:
    """Test NullAudioBackend."""

    def test_null_backend_can_be_instantiated(self):
        """NullAudioBackend can be instantiated."""
        backend = NullAudioBackend()
        assert backend is not None

    def test_null_backend_implements_audio_backend(self):
        """NullAudioBackend implements AudioBackend interface."""
        backend = NullAudioBackend()
        assert isinstance(backend, AudioBackend)


class TestSpatialAudioVec3:
    """Test Vec3 spatial audio helper."""

    def test_vec3_creation_default(self):
        """Vec3 can be created with defaults."""
        v = Vec3()
        assert hasattr(v, 'x')
        assert hasattr(v, 'y')
        assert hasattr(v, 'z')

    def test_vec3_creation_with_values(self):
        """Vec3 can be created with custom values."""
        v = Vec3(1.0, 2.0, 3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_vec3_zero_values(self):
        """Vec3 with zeros works correctly."""
        v = Vec3(0.0, 0.0, 0.0)
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0

    def test_vec3_negative_values(self):
        """Vec3 with negative values works correctly."""
        v = Vec3(-1.0, -2.0, -3.0)
        assert v.x == -1.0
        assert v.y == -2.0
        assert v.z == -3.0


class TestSpatialSource:
    """Test SpatialSource."""

    def test_spatial_source_creation_default(self):
        """SpatialSource can be created with defaults."""
        source = SpatialSource()
        assert source is not None

    def test_spatial_source_has_position(self):
        """SpatialSource has position attribute."""
        source = SpatialSource()
        assert hasattr(source, 'position')

    def test_spatial_source_with_position(self):
        """SpatialSource can be created with custom position."""
        pos = Vec3(10.0, 0.0, 0.0)
        source = SpatialSource(position=pos)
        assert source.position.x == 10.0

    def test_spatial_source_has_velocity(self):
        """SpatialSource has velocity attribute."""
        source = SpatialSource()
        assert hasattr(source, 'velocity')


class TestSpatialListener:
    """Test SpatialListener."""

    def test_spatial_listener_creation_default(self):
        """SpatialListener can be created with defaults."""
        listener = SpatialListener()
        assert listener is not None

    def test_spatial_listener_has_position(self):
        """SpatialListener has position attribute."""
        listener = SpatialListener()
        assert hasattr(listener, 'position')

    def test_spatial_listener_has_orientation(self):
        """SpatialListener has orientation attributes."""
        listener = SpatialListener()
        assert hasattr(listener, 'forward')
        assert hasattr(listener, 'up')


class TestSpatialAudioEngine:
    """Test SpatialAudioEngine."""

    def test_spatial_engine_creation(self):
        """SpatialAudioEngine can be created."""
        engine = SpatialAudioEngine()
        assert engine is not None

    def test_spatial_engine_has_create_source(self):
        """SpatialAudioEngine has create_source method."""
        engine = SpatialAudioEngine()
        assert hasattr(engine, 'create_source')
        assert callable(engine.create_source)

    def test_spatial_engine_has_update_listener(self):
        """SpatialAudioEngine has update_listener method."""
        engine = SpatialAudioEngine()
        assert hasattr(engine, 'update_listener')
        assert callable(engine.update_listener)


class TestReverbPreset:
    """Test ReverbPreset enum."""

    def test_reverb_preset_has_none(self):
        """ReverbPreset has NONE value."""
        assert hasattr(ReverbPreset, 'NONE')

    def test_reverb_preset_has_small_room(self):
        """ReverbPreset has SMALL_ROOM value."""
        assert hasattr(ReverbPreset, 'SMALL_ROOM')

    def test_reverb_preset_has_large_hall(self):
        """ReverbPreset has LARGE_HALL value."""
        assert hasattr(ReverbPreset, 'LARGE_HALL')


# =============================================================================
# Section 5: Input System Tests (25 tests)
# =============================================================================

from engine.platform.input import (
    InputManager,
    InputDevice,
    InputDeviceType,
    InputEvent,
    Keyboard,
    KeyCode,
    KeyState,
    Mouse,
    MouseButton,
    Gamepad,
    GamepadAxis,
    GamepadButton,
    GamepadTrigger,
    TouchDevice,
    TouchPoint,
    TouchPhase,
    PenDevice,
    Haptics,
    HapticType,
    HapticEffect,
    XRController,
    XRHand,
    XRButton,
    HandJoint,
    Pose,
    JointPose,
)


class TestInputDeviceType:
    """Test InputDeviceType enum."""

    def test_device_type_has_keyboard(self):
        """InputDeviceType has KEYBOARD value."""
        assert hasattr(InputDeviceType, 'KEYBOARD')

    def test_device_type_has_mouse(self):
        """InputDeviceType has MOUSE value."""
        assert hasattr(InputDeviceType, 'MOUSE')

    def test_device_type_has_gamepad(self):
        """InputDeviceType has GAMEPAD value."""
        assert hasattr(InputDeviceType, 'GAMEPAD')

    def test_device_type_has_touch(self):
        """InputDeviceType has TOUCH value."""
        assert hasattr(InputDeviceType, 'TOUCH')


class TestKeyboard:
    """Test Keyboard class."""

    def test_keyboard_can_be_instantiated(self):
        """Keyboard can be instantiated."""
        keyboard = Keyboard()
        assert keyboard is not None

    def test_keyboard_has_is_key_down(self):
        """Keyboard has is_key_down method."""
        keyboard = Keyboard()
        assert hasattr(keyboard, 'is_key_down')
        assert callable(keyboard.is_key_down)


class TestKeyCode:
    """Test KeyCode enum."""

    def test_keycode_has_a(self):
        """KeyCode has A value."""
        assert hasattr(KeyCode, 'A')

    def test_keycode_has_space(self):
        """KeyCode has SPACE value."""
        assert hasattr(KeyCode, 'SPACE')

    def test_keycode_has_escape(self):
        """KeyCode has ESCAPE value."""
        assert hasattr(KeyCode, 'ESCAPE')

    def test_keycode_has_enter(self):
        """KeyCode has ENTER value."""
        assert hasattr(KeyCode, 'ENTER')


class TestKeyState:
    """Test KeyState enum."""

    def test_keystate_has_pressed(self):
        """KeyState has PRESSED value."""
        assert hasattr(KeyState, 'PRESSED')

    def test_keystate_has_released(self):
        """KeyState has RELEASED value."""
        assert hasattr(KeyState, 'RELEASED')


class TestMouse:
    """Test Mouse class."""

    def test_mouse_can_be_instantiated(self):
        """Mouse can be instantiated."""
        mouse = Mouse()
        assert mouse is not None

    def test_mouse_has_position(self):
        """Mouse has position attribute or method."""
        mouse = Mouse()
        assert hasattr(mouse, 'position') or hasattr(mouse, 'get_position')


class TestMouseButton:
    """Test MouseButton enum."""

    def test_mouse_button_has_left(self):
        """MouseButton has LEFT value."""
        assert hasattr(MouseButton, 'LEFT')

    def test_mouse_button_has_right(self):
        """MouseButton has RIGHT value."""
        assert hasattr(MouseButton, 'RIGHT')

    def test_mouse_button_has_middle(self):
        """MouseButton has MIDDLE value."""
        assert hasattr(MouseButton, 'MIDDLE')


class TestGamepad:
    """Test Gamepad class."""

    def test_gamepad_can_be_instantiated(self):
        """Gamepad can be instantiated."""
        gamepad = Gamepad()
        assert gamepad is not None


class TestGamepadAxis:
    """Test GamepadAxis enum."""

    def test_gamepad_axis_has_left_x(self):
        """GamepadAxis has LEFT_X value."""
        assert hasattr(GamepadAxis, 'LEFT_X')

    def test_gamepad_axis_has_right_x(self):
        """GamepadAxis has RIGHT_X value."""
        assert hasattr(GamepadAxis, 'RIGHT_X')


class TestGamepadButton:
    """Test GamepadButton enum."""

    def test_gamepad_button_has_a(self):
        """GamepadButton has A value."""
        assert hasattr(GamepadButton, 'A')

    def test_gamepad_button_has_b(self):
        """GamepadButton has B value."""
        assert hasattr(GamepadButton, 'B')


class TestTouchDevice:
    """Test TouchDevice class."""

    def test_touch_device_can_be_instantiated(self):
        """TouchDevice can be instantiated."""
        touch = TouchDevice()
        assert touch is not None


class TestTouchPhase:
    """Test TouchPhase enum."""

    def test_touch_phase_has_began(self):
        """TouchPhase has BEGAN value."""
        assert hasattr(TouchPhase, 'BEGAN')

    def test_touch_phase_has_moved(self):
        """TouchPhase has MOVED value."""
        assert hasattr(TouchPhase, 'MOVED')

    def test_touch_phase_has_ended(self):
        """TouchPhase has ENDED value."""
        assert hasattr(TouchPhase, 'ENDED')


class TestHaptics:
    """Test Haptics class."""

    def test_haptics_can_be_instantiated(self):
        """Haptics can be instantiated."""
        haptics = Haptics()
        assert haptics is not None


class TestHapticType:
    """Test HapticType enum."""

    def test_haptic_type_has_rumble(self):
        """HapticType has RUMBLE value."""
        assert hasattr(HapticType, 'RUMBLE')


# =============================================================================
# Section 6: Window System Tests (25 tests)
# =============================================================================

from engine.platform.window import (
    Window,
    WindowConfig,
    WindowStyle,
    FullscreenMode,
    WindowState,
    WindowEvent,
    WindowEventType,
    Rect,
    Display,
    DisplayMode,
    DisplayInfo,
    CursorManager,
    CursorType,
    DisplayHDR,
    HDRCapabilities,
    ColorSpace,
    VariableRefresh,
    VRRType,
    RefreshRange,
)


class TestWindowStyle:
    """Test WindowStyle enum."""

    def test_window_style_has_windowed(self):
        """WindowStyle has WINDOWED value."""
        assert hasattr(WindowStyle, 'WINDOWED')

    def test_window_style_has_borderless(self):
        """WindowStyle has BORDERLESS value."""
        assert hasattr(WindowStyle, 'BORDERLESS')

    def test_window_style_has_fullscreen_exclusive(self):
        """WindowStyle has FULLSCREEN_EXCLUSIVE value."""
        assert hasattr(WindowStyle, 'FULLSCREEN_EXCLUSIVE')


class TestFullscreenMode:
    """Test FullscreenMode enum."""

    def test_fullscreen_mode_has_exclusive(self):
        """FullscreenMode has EXCLUSIVE value."""
        assert hasattr(FullscreenMode, 'EXCLUSIVE')

    def test_fullscreen_mode_has_borderless(self):
        """FullscreenMode has BORDERLESS value."""
        assert hasattr(FullscreenMode, 'BORDERLESS')


class TestWindowState:
    """Test WindowState enum."""

    def test_window_state_has_normal(self):
        """WindowState has NORMAL value."""
        assert hasattr(WindowState, 'NORMAL')

    def test_window_state_has_minimized(self):
        """WindowState has MINIMIZED value."""
        assert hasattr(WindowState, 'MINIMIZED')

    def test_window_state_has_maximized(self):
        """WindowState has MAXIMIZED value."""
        assert hasattr(WindowState, 'MAXIMIZED')


class TestWindowEventType:
    """Test WindowEventType enum."""

    def test_window_event_type_has_resize(self):
        """WindowEventType has RESIZE value."""
        assert hasattr(WindowEventType, 'RESIZE')

    def test_window_event_type_has_close(self):
        """WindowEventType has CLOSE value."""
        assert hasattr(WindowEventType, 'CLOSE')

    def test_window_event_type_has_focus(self):
        """WindowEventType has FOCUS value."""
        assert hasattr(WindowEventType, 'FOCUS')


class TestRect:
    """Test Rect dataclass."""

    def test_rect_can_be_created(self):
        """Rect can be created."""
        rect = Rect(x=0, y=0, width=100, height=100)
        assert rect is not None

    def test_rect_has_dimensions(self):
        """Rect has x, y, width, height."""
        rect = Rect(x=10, y=20, width=800, height=600)
        assert rect.x == 10
        assert rect.y == 20
        assert rect.width == 800
        assert rect.height == 600


class TestWindowConfig:
    """Test WindowConfig dataclass."""

    def test_window_config_can_be_created(self):
        """WindowConfig can be created."""
        config = WindowConfig()
        assert config is not None

    def test_window_config_has_title(self):
        """WindowConfig has title attribute."""
        config = WindowConfig()
        assert hasattr(config, 'title')

    def test_window_config_has_width_height(self):
        """WindowConfig has width and height attributes."""
        config = WindowConfig()
        assert hasattr(config, 'width')
        assert hasattr(config, 'height')


class TestCursorType:
    """Test CursorType enum."""

    def test_cursor_type_has_arrow(self):
        """CursorType has ARROW value."""
        assert hasattr(CursorType, 'ARROW')

    def test_cursor_type_has_hand(self):
        """CursorType has HAND value."""
        assert hasattr(CursorType, 'HAND')

    def test_cursor_type_has_ibeam(self):
        """CursorType has IBEAM value."""
        assert hasattr(CursorType, 'IBEAM')


class TestCursorManager:
    """Test CursorManager class."""

    def test_cursor_manager_can_be_instantiated(self):
        """CursorManager can be instantiated."""
        manager = CursorManager()
        assert manager is not None


class TestDisplayHDR:
    """Test DisplayHDR class."""

    def test_display_hdr_can_be_instantiated(self):
        """DisplayHDR can be instantiated."""
        hdr = DisplayHDR()
        assert hdr is not None


class TestColorSpace:
    """Test ColorSpace enum."""

    def test_color_space_has_srgb(self):
        """ColorSpace has SRGB value."""
        assert hasattr(ColorSpace, 'SRGB')

    def test_color_space_has_hdr10(self):
        """ColorSpace has HDR10 value."""
        assert hasattr(ColorSpace, 'HDR10')


class TestVariableRefresh:
    """Test VariableRefresh class."""

    def test_variable_refresh_can_be_instantiated(self):
        """VariableRefresh can be instantiated."""
        vrr = VariableRefresh()
        assert vrr is not None


class TestVRRType:
    """Test VRRType enum."""

    def test_vrr_type_has_none(self):
        """VRRType has NONE value."""
        assert hasattr(VRRType, 'NONE')

    def test_vrr_type_has_freesync(self):
        """VRRType has FREESYNC value."""
        assert hasattr(VRRType, 'FREESYNC')

    def test_vrr_type_has_gsync(self):
        """VRRType has GSYNC value."""
        assert hasattr(VRRType, 'GSYNC')


# =============================================================================
# Section 7: GPU System Tests (10 tests)
# =============================================================================

from engine.platform.gpu import (
    LowLatencyAPI,
    LowLatencyConfig,
    LowLatency,
)


class TestLowLatencyAPI:
    """Test LowLatencyAPI enum."""

    def test_low_latency_api_has_none(self):
        """LowLatencyAPI has NONE value."""
        assert hasattr(LowLatencyAPI, 'NONE')

    def test_low_latency_api_has_nvidia_reflex(self):
        """LowLatencyAPI has NVIDIA_REFLEX value."""
        assert hasattr(LowLatencyAPI, 'NVIDIA_REFLEX')

    def test_low_latency_api_has_amd_antilag(self):
        """LowLatencyAPI has AMD_ANTILAG value."""
        assert hasattr(LowLatencyAPI, 'AMD_ANTILAG')


class TestLowLatencyConfig:
    """Test LowLatencyConfig dataclass."""

    def test_low_latency_config_can_be_created(self):
        """LowLatencyConfig can be created."""
        config = LowLatencyConfig()
        assert config is not None


class TestLowLatency:
    """Test LowLatency class."""

    def test_low_latency_can_be_instantiated(self):
        """LowLatency can be instantiated."""
        low_latency = LowLatency()
        assert low_latency is not None

    def test_low_latency_has_is_available(self):
        """LowLatency has is_available attribute."""
        low_latency = LowLatency()
        assert hasattr(low_latency, 'is_available')

    def test_low_latency_is_available_is_bool(self):
        """LowLatency.is_available is boolean property."""
        low_latency = LowLatency()
        # is_available is a property, not a method
        result = low_latency.is_available
        assert isinstance(result, bool)

    def test_low_latency_has_enable(self):
        """LowLatency has enable method."""
        low_latency = LowLatency()
        assert hasattr(low_latency, 'enable')

    def test_low_latency_has_disable(self):
        """LowLatency has disable method."""
        low_latency = LowLatency()
        assert hasattr(low_latency, 'disable')

    def test_low_latency_has_set_marker(self):
        """LowLatency has set_marker method."""
        low_latency = LowLatency()
        assert hasattr(low_latency, 'set_marker')
        assert callable(low_latency.set_marker)


# =============================================================================
# Section 8: Constants Validation Tests (15 tests)
# =============================================================================

from engine.platform import constants


class TestOSConstants:
    """Test OS abstraction constants."""

    def test_default_page_size(self):
        """DEFAULT_PAGE_SIZE is a reasonable power of 2."""
        assert constants.DEFAULT_PAGE_SIZE >= 4096
        # Check power of 2
        assert constants.DEFAULT_PAGE_SIZE & (constants.DEFAULT_PAGE_SIZE - 1) == 0

    def test_default_cache_line_size(self):
        """DEFAULT_CACHE_LINE_SIZE is a reasonable power of 2."""
        assert constants.DEFAULT_CACHE_LINE_SIZE >= 32
        assert constants.DEFAULT_CACHE_LINE_SIZE & (constants.DEFAULT_CACHE_LINE_SIZE - 1) == 0

    def test_ticks_per_second(self):
        """TICKS_PER_SECOND is nanoseconds."""
        assert constants.TICKS_PER_SECOND == 1_000_000_000

    def test_nanos_per_milli(self):
        """NANOS_PER_MILLI is correct."""
        assert constants.NANOS_PER_MILLI == 1_000_000


class TestWindowConstants:
    """Test window and display constants."""

    def test_default_window_dimensions(self):
        """Default window dimensions are reasonable."""
        assert constants.DEFAULT_WINDOW_WIDTH >= 640
        assert constants.DEFAULT_WINDOW_HEIGHT >= 480

    def test_standard_resolutions_exist(self):
        """STANDARD_RESOLUTIONS is a non-empty list."""
        assert isinstance(constants.STANDARD_RESOLUTIONS, list)
        assert len(constants.STANDARD_RESOLUTIONS) > 0

    def test_standard_refresh_rates_exist(self):
        """STANDARD_REFRESH_RATES is a non-empty list."""
        assert isinstance(constants.STANDARD_REFRESH_RATES, list)
        assert len(constants.STANDARD_REFRESH_RATES) > 0
        assert 60 in constants.STANDARD_REFRESH_RATES


class TestHDRConstants:
    """Test HDR constants."""

    def test_hdr_min_luminance(self):
        """HDR_DEFAULT_MIN_LUMINANCE is a small positive number."""
        assert constants.HDR_DEFAULT_MIN_LUMINANCE > 0
        assert constants.HDR_DEFAULT_MIN_LUMINANCE < 1

    def test_hdr_max_luminance(self):
        """HDR_DEFAULT_MAX_LUMINANCE is a large positive number."""
        assert constants.HDR_DEFAULT_MAX_LUMINANCE > 100


class TestVRRConstants:
    """Test VRR constants."""

    def test_vrr_default_min_hz(self):
        """VRR_DEFAULT_MIN_HZ is reasonable."""
        assert constants.VRR_DEFAULT_MIN_HZ >= 30
        assert constants.VRR_DEFAULT_MIN_HZ < 60

    def test_vrr_default_max_hz(self):
        """VRR_DEFAULT_MAX_HZ is reasonable."""
        assert constants.VRR_DEFAULT_MAX_HZ >= 60


class TestAudioConstants:
    """Test audio constants."""

    def test_default_sample_rate(self):
        """DEFAULT_AUDIO_SAMPLE_RATE is standard."""
        assert constants.DEFAULT_AUDIO_SAMPLE_RATE == 48000

    def test_fallback_sample_rate(self):
        """FALLBACK_AUDIO_SAMPLE_RATE is CD quality."""
        assert constants.FALLBACK_AUDIO_SAMPLE_RATE == 44100

    def test_default_channels(self):
        """DEFAULT_AUDIO_CHANNELS is stereo."""
        assert constants.DEFAULT_AUDIO_CHANNELS == 2


class TestInputConstants:
    """Test input constants."""

    def test_default_gamepad_deadzone(self):
        """DEFAULT_GAMEPAD_DEADZONE is reasonable."""
        assert constants.DEFAULT_GAMEPAD_DEADZONE > 0
        assert constants.DEFAULT_GAMEPAD_DEADZONE < 0.5


class TestRHIConstants:
    """Test RHI constants."""

    def test_handle_ranges_non_overlapping(self):
        """Handle ranges are non-overlapping."""
        assert constants.BUFFER_HANDLE_START < constants.TEXTURE_HANDLE_START
        assert constants.TEXTURE_HANDLE_START < constants.SAMPLER_HANDLE_START
        assert constants.SAMPLER_HANDLE_START < constants.SHADER_HANDLE_START
        assert constants.SHADER_HANDLE_START < constants.PIPELINE_HANDLE_START

    def test_gpu_address_start(self):
        """GPU_ADDRESS_START is at 4GB boundary."""
        assert constants.GPU_ADDRESS_START == 0x100000000  # 4GB


# =============================================================================
# Section 9: RHI Integration Tests (15 tests)
# =============================================================================

from engine.platform.rhi import (
    Adapter,
    AdapterInfo,
    AdapterType,
    Device,
    DeviceConfig,
    NullAdapter,
    NullDevice,
    QueueType,
    Buffer,
    BufferDesc,
    BufferUsage,
    MemoryType,
    Texture,
    TextureDesc,
    Format,
    NullCommandList,
    NullQueue,
    NullFence,
    NullSwapchain,
    SwapchainDesc,
    PresentMode,
)


class TestRHIAdapterType:
    """Test AdapterType enum."""

    def test_adapter_type_has_discrete(self):
        """AdapterType has DISCRETE value."""
        assert hasattr(AdapterType, 'DISCRETE')

    def test_adapter_type_has_integrated(self):
        """AdapterType has INTEGRATED value."""
        assert hasattr(AdapterType, 'INTEGRATED')

    def test_adapter_type_has_software(self):
        """AdapterType has SOFTWARE value."""
        assert hasattr(AdapterType, 'SOFTWARE')


class TestNullAdapter:
    """Test NullAdapter."""

    def test_null_adapter_can_be_created(self):
        """NullAdapter can be instantiated."""
        adapter = NullAdapter(AdapterType.DISCRETE)
        assert adapter is not None

    def test_null_adapter_enumerate(self):
        """NullAdapter.enumerate returns list."""
        adapters = NullAdapter.enumerate()
        assert isinstance(adapters, list)
        assert len(adapters) >= 1


class TestNullDevice:
    """Test NullDevice."""

    def test_null_device_can_be_created(self):
        """NullDevice can be created from adapter."""
        adapter = NullAdapter(AdapterType.DISCRETE)
        config = DeviceConfig(adapter=adapter)
        device = NullDevice.create(adapter, config)
        assert device is not None

    def test_null_device_has_shutdown(self):
        """NullDevice has shutdown method."""
        adapter = NullAdapter(AdapterType.DISCRETE)
        config = DeviceConfig(adapter=adapter)
        device = NullDevice.create(adapter, config)
        assert hasattr(device, 'shutdown')

    def test_null_device_get_queue(self):
        """NullDevice can get queue."""
        adapter = NullAdapter(AdapterType.DISCRETE)
        config = DeviceConfig(adapter=adapter)
        device = NullDevice.create(adapter, config)
        queue = device.get_queue(QueueType.GRAPHICS)
        assert queue is not None


class TestQueueType:
    """Test QueueType enum."""

    def test_queue_type_has_graphics(self):
        """QueueType has GRAPHICS value."""
        assert hasattr(QueueType, 'GRAPHICS')

    def test_queue_type_has_compute(self):
        """QueueType has COMPUTE value."""
        assert hasattr(QueueType, 'COMPUTE')

    def test_queue_type_has_transfer(self):
        """QueueType has TRANSFER value."""
        assert hasattr(QueueType, 'TRANSFER')


class TestBufferUsage:
    """Test BufferUsage flags."""

    def test_buffer_usage_has_vertex(self):
        """BufferUsage has VERTEX value."""
        assert hasattr(BufferUsage, 'VERTEX')

    def test_buffer_usage_has_index(self):
        """BufferUsage has INDEX value."""
        assert hasattr(BufferUsage, 'INDEX')

    def test_buffer_usage_has_constant(self):
        """BufferUsage has CONSTANT value."""
        assert hasattr(BufferUsage, 'CONSTANT')


class TestFormat:
    """Test Format enum."""

    def test_format_has_rgba8(self):
        """Format has RGBA8_UNORM value."""
        assert hasattr(Format, 'RGBA8_UNORM')

    def test_format_has_rgba16_float(self):
        """Format has RGBA16_FLOAT value."""
        assert hasattr(Format, 'RGBA16_FLOAT')


# =============================================================================
# Section 10: Integration Tests (10 tests)
# =============================================================================

class TestPlatformIntegration:
    """Integration tests for platform layer."""

    def test_bootstrap_provides_services(self):
        """Bootstrap initializes all services."""
        bootstrap_platform()
        info = get_platform_info()
        assert info is not None

    def test_registry_backend_create_lifecycle(self):
        """Registry -> create -> use lifecycle works."""
        registry: BackendRegistry[DummyBackend] = BackendRegistry()
        registry.register("test", DummyBackend, set_default=True)
        instance = registry.create()
        assert isinstance(instance, DummyBackend)

    def test_spatial_audio_source_listener_workflow(self):
        """Spatial audio source-listener workflow."""
        engine = SpatialAudioEngine()
        source = SpatialSource(position=Vec3(10.0, 0.0, 0.0))
        listener = SpatialListener(position=Vec3(0.0, 0.0, 0.0))
        # Use update_listener instead of set_listener
        engine.update_listener(listener)
        # Source creation should work
        handle = engine.create_source(source)
        assert handle is not None

    def test_rhi_adapter_to_device_workflow(self):
        """RHI adapter -> device -> queue workflow."""
        adapters = NullAdapter.enumerate()
        assert len(adapters) >= 1
        adapter = adapters[0]
        config = DeviceConfig(adapter=adapter)
        device = NullDevice.create(adapter, config)
        assert device is not None
        queue = device.get_queue(QueueType.GRAPHICS)
        assert queue is not None

    def test_window_config_initialization(self):
        """WindowConfig can be customized."""
        config = WindowConfig(
            title="Test Window",
            width=1920,
            height=1080,
        )
        assert config.title == "Test Window"
        assert config.width == 1920
        assert config.height == 1080

    def test_constants_form_coherent_system(self):
        """Constants form a coherent system."""
        # Audio constants are coherent
        assert constants.DEFAULT_AUDIO_SAMPLE_RATE > constants.FALLBACK_AUDIO_SAMPLE_RATE
        # VRR range is valid
        assert constants.VRR_DEFAULT_MIN_HZ < constants.VRR_DEFAULT_MAX_HZ
        # HDR range is valid
        assert constants.HDR_DEFAULT_MIN_LUMINANCE < constants.HDR_DEFAULT_MAX_LUMINANCE

    def test_input_types_are_complete(self):
        """Input system provides all required types."""
        # All input device types exist
        assert InputDeviceType.KEYBOARD is not None
        assert InputDeviceType.MOUSE is not None
        assert InputDeviceType.GAMEPAD is not None
        # All key states exist
        assert KeyState.PRESSED is not None
        assert KeyState.RELEASED is not None

    def test_low_latency_api_detection(self):
        """Low latency API detection works."""
        low_latency = LowLatency()
        # is_available is a property, not a method
        available = low_latency.is_available
        # Should return a boolean
        assert isinstance(available, bool)

    def test_service_provider_workflow(self):
        """Service provider creation workflow."""
        provider = create_service_provider()
        assert isinstance(provider, ServiceProvider)

    def test_multi_backend_registry_isolation(self):
        """Multiple registries are isolated."""
        registry1: BackendRegistry[DummyBackend] = BackendRegistry()
        registry2: BackendRegistry[DummyBackend] = BackendRegistry()

        registry1.register("only_in_1", DummyBackend)
        registry2.register("only_in_2", DummyBackend)

        assert "only_in_1" in registry1.list()
        assert "only_in_1" not in registry2.list()
        assert "only_in_2" in registry2.list()
        assert "only_in_2" not in registry1.list()


# =============================================================================
# Cleanup fixture to ensure platform is in good state
# =============================================================================

@pytest.fixture(autouse=True, scope="module")
def ensure_platform_bootstrapped():
    """Ensure platform is bootstrapped for all tests."""
    bootstrap_platform()
    yield
    # Don't shutdown - other tests may need it


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
