"""Platform abstraction layer and bootstrap."""
from typing import Optional
import logging
import threading

logger = logging.getLogger(__name__)

# Import submodules
from . import rhi
from . import services
from . import gpu
from .registry import BackendRegistry

# Singleton instances
_bootstrap_lock = threading.Lock()
_bootstrapped = False
_device: Optional[rhi.Device] = None
_lifecycle: Optional[services.AppLifecycle] = None
_low_latency: Optional[gpu.LowLatency] = None


def bootstrap_platform() -> None:
    """
    Bootstrap platform subsystems in priority order.

    This initializes:
    1. Platform detection
    2. Application lifecycle
    3. Graphics device (null backend)
    4. Low latency features
    """
    global _bootstrapped, _device, _lifecycle, _low_latency

    with _bootstrap_lock:
        if _bootstrapped:
            return

        # Detect platform
        platform_info = services.detect()
        logger.info(f"Platform: {platform_info.name} {platform_info.version} ({platform_info.arch})")

        # Initialize lifecycle
        _lifecycle = services.AppLifecycle()

        # Initialize graphics device (null backend)
        adapters = rhi.NullAdapter.enumerate()
        if adapters:
            adapter = adapters[0]  # Use first adapter
            config = rhi.DeviceConfig(adapter=adapter, enable_debug=True)
            _device = rhi.NullDevice.create(adapter, config)

        # Initialize low latency features
        _low_latency = gpu.LowLatency()

        _bootstrapped = True


def shutdown_platform() -> None:
    """Shutdown platform subsystems."""
    global _bootstrapped, _device, _lifecycle

    with _bootstrap_lock:
        if not _bootstrapped:
            return

        if _lifecycle:
            _lifecycle.shutdown()

        if _device:
            _device.shutdown()

        _bootstrapped = False


# Public API functions

def create_graphics_device() -> Optional[rhi.Device]:
    """Get or create graphics device."""
    if not _bootstrapped:
        bootstrap_platform()
    return _device


def get_lifecycle() -> Optional[services.AppLifecycle]:
    """Get application lifecycle manager."""
    if not _bootstrapped:
        bootstrap_platform()
    return _lifecycle


def get_platform_info() -> services.PlatformInfo:
    """Get platform information."""
    return services.detect()


def get_low_latency() -> Optional[gpu.LowLatency]:
    """Get low latency manager."""
    if not _bootstrapped:
        bootstrap_platform()
    return _low_latency


# Export main APIs
__all__ = [
    # Bootstrap
    'bootstrap_platform',
    'shutdown_platform',
    # API accessors
    'create_graphics_device',
    'get_lifecycle',
    'get_platform_info',
    'get_low_latency',
    # Classes
    'BackendRegistry',
    # Submodules
    'rhi',
    'services',
    'gpu',
]
