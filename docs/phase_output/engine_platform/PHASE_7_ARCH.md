# PHASE 7 ARCHITECTURE: GPU Low-Latency and Services Completion

## Phase Overview

Phase 7 addresses the remaining stub implementations: GPU low-latency features and mobile platform services. These are lower priority but necessary for complete platform coverage.

## Current State (from Investigation)

### GPU (98 lines - STUB)

| Component | Status | Notes |
|-----------|--------|-------|
| LowLatency | STUB | is_available returns False, sleep uses time.sleep |
| LowLatencyAPI | Enum | NONE, NVIDIA_REFLEX, AMD_ANTILAG |
| LowLatencyConfig | Dataclass | enabled, boost, min_interval_us |

### Services (265 lines - MIXED)

| Component | Status | Notes |
|-----------|--------|-------|
| platform_detect.py | REAL | Desktop detection works |
| app_lifecycle.py | REAL | Thread-safe singleton |
| permissions.py | STUB | Auto-grants all permissions |

## Architectural Decisions

### ADR-P7-001: NVIDIA Reflex Integration

**Status:** Proposed (Windows Only)

**Context:**
NVIDIA Reflex requires NVAPI library. Available on Windows with NVIDIA drivers.

**Decision:**
Use ctypes to load nvapi64.dll:

```python
# engine/platform/gpu/reflex.py
import ctypes
from ctypes import wintypes

class NVAPIReflex:
    def __init__(self):
        try:
            self._nvapi = ctypes.WinDLL("nvapi64.dll")
            self._available = self._initialize()
        except OSError:
            self._available = False

    def _initialize(self) -> bool:
        # NvAPI_Initialize
        result = self._nvapi.nvapi_QueryInterface(0x0150E828)  # NvAPI_Initialize
        return result == 0  # NVAPI_OK

    def is_available(self) -> bool:
        return self._available

    def set_sleep_mode(self, enabled: bool, min_interval_us: int) -> bool:
        # NvAPI_D3D_SetSleepMode
        ...

    def sleep(self) -> None:
        # NvAPI_D3D_Sleep
        ...

    def set_marker(self, marker_type: int, frame_id: int) -> None:
        # NvAPI_D3D_SetLatencyMarker
        ...
```

**Consequences:**
- Windows-only (NVAPI not available elsewhere)
- Requires NVIDIA GPU and recent drivers
- Graceful fallback on non-NVIDIA systems

### ADR-P7-002: AMD Anti-Lag Integration

**Status:** Proposed (Windows Only)

**Context:**
AMD Anti-Lag requires AGS library. Available on Windows with AMD drivers.

**Decision:**
Use ctypes to load amd_ags_x64.dll:

```python
# engine/platform/gpu/antilag.py
import ctypes

class AMDAntiLag:
    def __init__(self):
        try:
            self._ags = ctypes.CDLL("amd_ags_x64.dll")
            self._available = self._initialize()
        except OSError:
            self._available = False

    def _initialize(self) -> bool:
        # agsInit
        ...

    def is_available(self) -> bool:
        return self._available

    def set_enabled(self, enabled: bool) -> bool:
        # agsDriverExtensionsDX12_SetAntiLag
        ...
```

**Consequences:**
- Windows-only
- Requires AMD GPU and recent drivers
- Graceful fallback on non-AMD systems

### ADR-P7-003: Unified Low-Latency Interface

**Status:** Proposed

**Context:**
Need unified API that works with either Reflex or Anti-Lag.

**Decision:**
LowLatency class detects and uses available API:

```python
# engine/platform/gpu/low_latency.py

class LowLatency:
    def __init__(self):
        self._backend: Optional[LowLatencyBackend] = None
        self._api = LowLatencyAPI.NONE

        # Try NVIDIA first
        try:
            from .reflex import NVAPIReflex
            reflex = NVAPIReflex()
            if reflex.is_available():
                self._backend = reflex
                self._api = LowLatencyAPI.NVIDIA_REFLEX
                return
        except ImportError:
            pass

        # Try AMD
        try:
            from .antilag import AMDAntiLag
            antilag = AMDAntiLag()
            if antilag.is_available():
                self._backend = antilag
                self._api = LowLatencyAPI.AMD_ANTILAG
                return
        except ImportError:
            pass

    @property
    def is_available(self) -> bool:
        return self._backend is not None

    @property
    def current_api(self) -> LowLatencyAPI:
        return self._api

    def sleep(self) -> None:
        if self._backend:
            self._backend.sleep()
        else:
            # Fallback to regular sleep (current behavior)
            time.sleep(self._config.min_interval_us / 1_000_000)
```

**Consequences:**
- Automatic API detection
- Single interface for both vendors
- Fallback when neither available

### ADR-P7-004: Mobile Permissions Strategy

**Status:** Proposed

**Context:**
Mobile platforms require runtime permissions. Current implementation auto-grants.

**Decision:**
Platform-specific permission backends:

```python
# engine/platform/services/permissions.py

class PermissionBackend(ABC):
    @abstractmethod
    def request(self, permission: Permission) -> PermissionStatus: ...
    @abstractmethod
    def check(self, permission: Permission) -> PermissionStatus: ...

class StubPermissionBackend(PermissionBackend):
    """Auto-grants all permissions (desktop default)."""
    def request(self, permission: Permission) -> PermissionStatus:
        return PermissionStatus.GRANTED

    def check(self, permission: Permission) -> PermissionStatus:
        return PermissionStatus.GRANTED

class AndroidPermissionBackend(PermissionBackend):
    """Android runtime permissions via pyjnius."""
    def request(self, permission: Permission) -> PermissionStatus:
        from jnius import autoclass
        PythonActivity = autoclass('org.kivy.android.PythonActivity')
        # Request permission via Android API
        ...

class IOSPermissionBackend(PermissionBackend):
    """iOS permissions via pyobjc."""
    def request(self, permission: Permission) -> PermissionStatus:
        # Use pyobjc to call iOS permission APIs
        ...
```

**Consequences:**
- Desktop continues to auto-grant
- Mobile gets real permission dialogs
- Requires platform-specific dependencies (pyjnius, pyobjc)

### ADR-P7-005: Platform Detection Enhancement

**Status:** Proposed

**Context:**
Platform detection missing Android, console, and web detection.

**Decision:**
Enhanced detection logic:

```python
def detect() -> PlatformInfo:
    system = platform.system()
    machine = platform.machine()

    # Android detection
    if 'ANDROID_ROOT' in os.environ or hasattr(sys, 'getandroidapilevel'):
        return PlatformInfo(
            type=PlatformType.ANDROID,
            is_mobile=True,
            # ...
        )

    # Web/WASM detection
    if sys.platform == 'emscripten':
        return PlatformInfo(
            type=PlatformType.WEB,
            # ...
        )

    # Console detection (build-time flags)
    if os.environ.get('TRINITY_PLATFORM') == 'PS5':
        return PlatformInfo(type=PlatformType.PS5, is_console=True, ...)

    # Desktop detection (existing logic)
    ...
```

**Consequences:**
- Android detected via environment
- Web detected via sys.platform
- Console via environment variable (set at build time)

## Component Diagram

```
engine/platform/gpu/
    |
    +-- low_latency.py      # Unified LowLatency interface
    +-- reflex.py           # NEW: NVIDIA Reflex (Windows)
    +-- antilag.py          # NEW: AMD Anti-Lag (Windows)
    +-- backends/
            +-- __init__.py

engine/platform/services/
    |
    +-- platform_detect.py  # Enhanced detection
    +-- app_lifecycle.py    # Unchanged
    +-- permissions.py      # Backend-based
    +-- backends/
            +-- stub.py     # Desktop auto-grant
            +-- android.py  # Android permissions (pyjnius)
            +-- ios.py      # iOS permissions (pyobjc)
```

## File Changes Required

### New Files

| File | Purpose |
|------|---------|
| engine/platform/gpu/reflex.py | NVIDIA Reflex ctypes wrapper |
| engine/platform/gpu/antilag.py | AMD Anti-Lag ctypes wrapper |
| engine/platform/services/backends/__init__.py | Permission backend registry |
| engine/platform/services/backends/stub.py | Desktop stub backend |
| engine/platform/services/backends/android.py | Android permissions |
| engine/platform/services/backends/ios.py | iOS permissions |

### Modified Files

| File | Changes |
|------|---------|
| engine/platform/gpu/low_latency.py | Use Reflex/Anti-Lag backends |
| engine/platform/services/platform_detect.py | Add Android, web, console detection |
| engine/platform/services/permissions.py | Use permission backends |

## Dependencies

### Optional Python Packages

| Package | Platform | Purpose |
|---------|----------|---------|
| pyjnius | Android | JNI bridge for Android APIs |
| pyobjc | iOS/macOS | Objective-C bridge |

### Native Libraries (via ctypes)

| Library | Platform | Purpose |
|---------|----------|---------|
| nvapi64.dll | Windows | NVIDIA Reflex |
| amd_ags_x64.dll | Windows | AMD Anti-Lag |

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| NVAPI/AGS not documented | Use existing open-source wrappers as reference |
| Library version mismatch | Version detection, graceful fallback |
| Mobile permissions complex | Start with common permissions (camera, storage) |
| pyjnius/pyobjc unavailable | Conditional imports, stub fallback |

## Phase Exit Criteria

1. NVIDIA Reflex works on Windows + NVIDIA GPU
2. AMD Anti-Lag works on Windows + AMD GPU
3. LowLatency gracefully falls back when neither available
4. Android platform detection works
5. Web platform detection works
6. Permission stubs still work for desktop
7. Mobile permission framework in place (even if not fully tested)
