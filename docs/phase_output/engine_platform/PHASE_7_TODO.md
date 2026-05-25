# PHASE 7 TODO: GPU Low-Latency and Services Completion

## Summary

Complete GPU low-latency features (Reflex, Anti-Lag) and mobile platform services.

**Estimated Effort:** 16-24 hours
**Dependencies:** Phase 1
**Blocking:** None (optional features)

---

## Tasks

### T-P7-001: Create NVIDIA Reflex Backend

**Priority:** P1 (Important)
**Estimate:** 4 hours

Create `engine/platform/gpu/reflex.py`:

```python
import ctypes
from ctypes import wintypes
from typing import Optional

class NVAPIReflex:
    """NVIDIA Reflex via NVAPI (Windows only)."""

    def __init__(self):
        self._available = False
        self._nvapi = None
        try:
            self._nvapi = ctypes.WinDLL("nvapi64.dll")
            self._available = self._initialize()
        except (OSError, AttributeError):
            pass

    def _initialize(self) -> bool:
        # NVAPI initialization
        ...

    def is_available(self) -> bool:
        return self._available

    def set_sleep_mode(self, enabled: bool, min_interval_us: int) -> bool: ...
    def sleep(self) -> None: ...
    def set_marker(self, marker_type: int, frame_id: int) -> None: ...
```

**Acceptance Criteria:**
- [ ] Loads nvapi64.dll successfully
- [ ] is_available() returns True on NVIDIA systems
- [ ] sleep() reduces input latency
- [ ] Graceful failure on non-NVIDIA systems

---

### T-P7-002: Create AMD Anti-Lag Backend

**Priority:** P1 (Important)
**Estimate:** 4 hours

Create `engine/platform/gpu/antilag.py`:

```python
import ctypes
from typing import Optional

class AMDAntiLag:
    """AMD Anti-Lag via AGS (Windows only)."""

    def __init__(self):
        self._available = False
        self._ags = None
        self._context = None
        try:
            self._ags = ctypes.CDLL("amd_ags_x64.dll")
            self._available = self._initialize()
        except (OSError, AttributeError):
            pass

    def _initialize(self) -> bool:
        # AGS initialization
        ...

    def is_available(self) -> bool:
        return self._available

    def set_enabled(self, enabled: bool) -> bool: ...
```

**Acceptance Criteria:**
- [ ] Loads amd_ags_x64.dll successfully
- [ ] is_available() returns True on AMD systems
- [ ] set_enabled() toggles Anti-Lag
- [ ] Graceful failure on non-AMD systems

---

### T-P7-003: Integrate Backends into LowLatency

**Priority:** P0 (Blocking)
**Estimate:** 2 hours

Modify `engine/platform/gpu/low_latency.py`:

```python
class LowLatency:
    def __init__(self, config: LowLatencyConfig | None = None):
        self._config = config or LowLatencyConfig()
        self._backend = None
        self._api = LowLatencyAPI.NONE

        # Detection order: Reflex -> Anti-Lag -> None
        self._detect_backend()

    def _detect_backend(self) -> None:
        try:
            from .reflex import NVAPIReflex
            reflex = NVAPIReflex()
            if reflex.is_available():
                self._backend = reflex
                self._api = LowLatencyAPI.NVIDIA_REFLEX
                return
        except Exception:
            pass

        try:
            from .antilag import AMDAntiLag
            antilag = AMDAntiLag()
            if antilag.is_available():
                self._backend = antilag
                self._api = LowLatencyAPI.AMD_ANTILAG
                return
        except Exception:
            pass

    @property
    def is_available(self) -> bool:
        return self._backend is not None
```

**Acceptance Criteria:**
- [ ] Auto-detects NVIDIA Reflex
- [ ] Auto-detects AMD Anti-Lag
- [ ] current_api property correct
- [ ] Falls back gracefully

---

### T-P7-004: Enhance Platform Detection

**Priority:** P1 (Important)
**Estimate:** 1.5 hours

Modify `engine/platform/services/platform_detect.py`:

```python
import os
import sys
import platform

def detect() -> PlatformInfo:
    system = platform.system()
    machine = platform.machine()

    # Android
    if 'ANDROID_ROOT' in os.environ:
        return PlatformInfo(
            type=PlatformType.ANDROID,
            name="Android",
            version=os.environ.get('ANDROID_API_LEVEL', ''),
            arch=machine,
            is_mobile=True,
            is_desktop=False,
            is_console=False
        )

    # Emscripten/Web
    if sys.platform == 'emscripten':
        return PlatformInfo(
            type=PlatformType.WEB,
            name="WebAssembly",
            # ...
        )

    # Console (via environment variable)
    console_platform = os.environ.get('TRINITY_PLATFORM')
    if console_platform == 'PS5':
        return PlatformInfo(type=PlatformType.PS5, is_console=True, ...)
    elif console_platform == 'XBOX':
        return PlatformInfo(type=PlatformType.XBOX, is_console=True, ...)
    elif console_platform == 'SWITCH':
        return PlatformInfo(type=PlatformType.SWITCH, is_console=True, ...)

    # Desktop (existing logic)
    ...
```

**Acceptance Criteria:**
- [ ] Android detected via ANDROID_ROOT
- [ ] Web detected via sys.platform
- [ ] Console detected via TRINITY_PLATFORM env var
- [ ] Existing desktop detection unchanged

---

### T-P7-005: Create Permission Backend ABC

**Priority:** P1 (Important)
**Estimate:** 1 hour

Create `engine/platform/services/backends/__init__.py`:

```python
from abc import ABC, abstractmethod
from ..permissions import Permission, PermissionStatus

class PermissionBackend(ABC):
    @abstractmethod
    def request(self, permission: Permission) -> PermissionStatus: ...

    @abstractmethod
    def check(self, permission: Permission) -> PermissionStatus: ...

    @abstractmethod
    def request_multiple(self, permissions: list[Permission]) -> dict[Permission, PermissionStatus]:
        return {p: self.request(p) for p in permissions}
```

**Acceptance Criteria:**
- [ ] ABC defines required methods
- [ ] Default implementation for request_multiple
- [ ] Type hints complete

---

### T-P7-006: Create Stub Permission Backend

**Priority:** P0 (Blocking)
**Estimate:** 30 minutes

Create `engine/platform/services/backends/stub.py`:

```python
from . import PermissionBackend
from ..permissions import Permission, PermissionStatus

class StubPermissionBackend(PermissionBackend):
    """Auto-grants all permissions. Used on desktop platforms."""

    def request(self, permission: Permission) -> PermissionStatus:
        return PermissionStatus.GRANTED

    def check(self, permission: Permission) -> PermissionStatus:
        return PermissionStatus.GRANTED
```

**Acceptance Criteria:**
- [ ] All permissions granted
- [ ] Used as default on desktop
- [ ] Existing behavior preserved

---

### T-P7-007: Create Android Permission Backend

**Priority:** P2 (Nice to have)
**Estimate:** 3 hours

Create `engine/platform/services/backends/android.py`:

```python
from . import PermissionBackend
from ..permissions import Permission, PermissionStatus

ANDROID_PERMISSION_MAP = {
    Permission.STORAGE: "android.permission.WRITE_EXTERNAL_STORAGE",
    Permission.CAMERA: "android.permission.CAMERA",
    Permission.MICROPHONE: "android.permission.RECORD_AUDIO",
    Permission.LOCATION: "android.permission.ACCESS_FINE_LOCATION",
    Permission.NETWORK: "android.permission.INTERNET",  # Granted at install
}

class AndroidPermissionBackend(PermissionBackend):
    def __init__(self):
        from jnius import autoclass
        self._Activity = autoclass('org.kivy.android.PythonActivity')
        self._ContextCompat = autoclass('androidx.core.content.ContextCompat')
        self._ActivityCompat = autoclass('androidx.core.app.ActivityCompat')

    def check(self, permission: Permission) -> PermissionStatus:
        android_perm = ANDROID_PERMISSION_MAP.get(permission)
        if not android_perm:
            return PermissionStatus.GRANTED  # Unknown = granted

        result = self._ContextCompat.checkSelfPermission(
            self._Activity.mActivity,
            android_perm
        )
        if result == 0:  # PERMISSION_GRANTED
            return PermissionStatus.GRANTED
        return PermissionStatus.DENIED

    def request(self, permission: Permission) -> PermissionStatus:
        # Use ActivityCompat.requestPermissions
        ...
```

**Acceptance Criteria:**
- [ ] Uses pyjnius correctly
- [ ] Maps Permission enum to Android permissions
- [ ] Returns accurate status
- [ ] Falls back to stub if pyjnius unavailable

---

### T-P7-008: Integrate Permission Backends

**Priority:** P0 (Blocking)
**Estimate:** 1 hour

Modify `engine/platform/services/permissions.py`:

```python
from .backends import PermissionBackend
from .backends.stub import StubPermissionBackend
from .platform_detect import detect, PlatformType

_backend: PermissionBackend | None = None

def _get_backend() -> PermissionBackend:
    global _backend
    if _backend is None:
        platform = detect()
        if platform.type == PlatformType.ANDROID:
            try:
                from .backends.android import AndroidPermissionBackend
                _backend = AndroidPermissionBackend()
            except ImportError:
                _backend = StubPermissionBackend()
        elif platform.type == PlatformType.IOS:
            # Similar for iOS
            _backend = StubPermissionBackend()
        else:
            _backend = StubPermissionBackend()
    return _backend

def request(permission: Permission) -> PermissionStatus:
    return _get_backend().request(permission)

def check(permission: Permission) -> PermissionStatus:
    return _get_backend().check(permission)
```

**Acceptance Criteria:**
- [ ] Auto-selects backend by platform
- [ ] Desktop uses stub
- [ ] Android uses Android backend when available
- [ ] Backward compatible API

---

### T-P7-009: Write Low-Latency Tests

**Priority:** P1 (Important)
**Estimate:** 1.5 hours

Create `tests/platform/gpu/test_low_latency.py`:

```python
import pytest
from engine.platform.gpu import LowLatency, LowLatencyAPI, LowLatencyConfig

def test_detection_does_not_crash():
    ll = LowLatency()
    # Should not raise even without GPU

def test_is_available_returns_bool():
    ll = LowLatency()
    assert isinstance(ll.is_available, bool)

def test_current_api_valid():
    ll = LowLatency()
    assert ll.current_api in LowLatencyAPI

def test_sleep_does_not_crash():
    ll = LowLatency(LowLatencyConfig(min_interval_us=1000))
    ll.sleep()  # Should not raise

@pytest.mark.skipif(True, reason="Requires NVIDIA GPU")
def test_reflex_available():
    from engine.platform.gpu.reflex import NVAPIReflex
    reflex = NVAPIReflex()
    assert reflex.is_available()
```

**Acceptance Criteria:**
- [ ] Tests pass without GPU
- [ ] No crashes on any platform
- [ ] Optional tests for Reflex/Anti-Lag

---

### T-P7-010: Write Platform Detection Tests

**Priority:** P1 (Important)
**Estimate:** 1 hour

Update `tests/platform/services/test_platform_detect.py`:

```python
import os
import pytest
from unittest.mock import patch
from engine.platform.services.platform_detect import detect, PlatformType

def test_detect_returns_info():
    info = detect()
    assert info.type in PlatformType
    assert isinstance(info.is_desktop, bool)

@patch.dict(os.environ, {'ANDROID_ROOT': '/system'})
def test_detect_android():
    info = detect()
    assert info.type == PlatformType.ANDROID
    assert info.is_mobile

@patch.dict(os.environ, {'TRINITY_PLATFORM': 'PS5'})
def test_detect_console():
    info = detect()
    assert info.type == PlatformType.PS5
    assert info.is_console
```

**Acceptance Criteria:**
- [ ] Current platform detected
- [ ] Android detection tested via mock
- [ ] Console detection tested via mock

---

### T-P7-011: Write Permission Tests

**Priority:** P1 (Important)
**Estimate:** 1 hour

Create `tests/platform/services/test_permissions.py`:

```python
import pytest
from engine.platform.services.permissions import (
    request, check, Permission, PermissionStatus
)

def test_request_returns_status():
    status = request(Permission.STORAGE)
    assert status in PermissionStatus

def test_check_returns_status():
    status = check(Permission.CAMERA)
    assert status in PermissionStatus

def test_desktop_grants_all():
    # On desktop, stub backend grants everything
    for perm in Permission:
        assert check(perm) == PermissionStatus.GRANTED
```

**Acceptance Criteria:**
- [ ] request() returns valid status
- [ ] check() returns valid status
- [ ] Desktop behavior verified

---

## Task Dependency Graph

```
T-P7-001 (NVIDIA Reflex)
    |
    +-- T-P7-003 (LowLatency Integration)
                |
                +-- T-P7-009 (Low-Latency Tests)

T-P7-002 (AMD Anti-Lag)
    |
    +-- T-P7-003 (LowLatency Integration)

T-P7-004 (Platform Detection)
    |
    +-- T-P7-008 (Permission Integration)
    |
    +-- T-P7-010 (Detection Tests)

T-P7-005 (Permission ABC)
    |
    +-- T-P7-006 (Stub Backend)
    |       |
    |       +-- T-P7-008 (Integration)
    |
    +-- T-P7-007 (Android Backend)
            |
            +-- T-P7-008 (Integration)

T-P7-011 (Permission Tests) -- after T-P7-008
```

## Verification Commands

```bash
# Test low-latency (should not crash even without GPU)
uv run python -c "
from engine.platform.gpu import LowLatency, LowLatencyConfig
ll = LowLatency(LowLatencyConfig(enabled=True))
print(f'Available: {ll.is_available}')
print(f'API: {ll.current_api}')
ll.sleep()
print('Sleep completed')
"

# Test platform detection
uv run python -c "
from engine.platform.services.platform_detect import detect
info = detect()
print(f'Platform: {info.type}')
print(f'Desktop: {info.is_desktop}')
print(f'Mobile: {info.is_mobile}')
"

# Test permissions
uv run python -c "
from engine.platform.services.permissions import request, check, Permission
for p in Permission:
    print(f'{p.name}: {check(p)}')
"

# Run all Phase 7 tests
uv run pytest tests/platform/gpu/test_low_latency.py tests/platform/services/ -v
```

## Completion Checklist

- [ ] T-P7-001: NVIDIA Reflex backend created
- [ ] T-P7-002: AMD Anti-Lag backend created
- [ ] T-P7-003: LowLatency uses backends
- [ ] T-P7-004: Platform detection enhanced
- [ ] T-P7-005: Permission ABC created
- [ ] T-P7-006: Stub backend created
- [ ] T-P7-007: Android backend created
- [ ] T-P7-008: Permissions use backends
- [ ] T-P7-009: Low-latency tests pass
- [ ] T-P7-010: Detection tests pass
- [ ] T-P7-011: Permission tests pass
