"""RHI Synchronization primitives."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
import threading
import time as _time


class ResourceState(Enum):
    """Resource state for barriers."""
    UNDEFINED = auto()
    COMMON = auto()
    RENDER_TARGET = auto()
    DEPTH_WRITE = auto()
    DEPTH_READ = auto()
    SHADER_RESOURCE = auto()
    UNORDERED_ACCESS = auto()
    COPY_SRC = auto()
    COPY_DST = auto()
    PRESENT = auto()


class BarrierType(Enum):
    """Barrier type."""
    TRANSITION = auto()
    UAV = auto()
    ALIASING = auto()


@dataclass
class BarrierDesc:
    """Barrier descriptor."""
    type: BarrierType
    resource: 'Any' = None
    state_before: ResourceState = ResourceState.UNDEFINED
    state_after: ResourceState = ResourceState.UNDEFINED


class Fence(ABC):
    """Abstract synchronization fence."""

    @classmethod
    @abstractmethod
    def create(cls, device: 'Device', initial: int = 0) -> Fence:
        """Create fence."""
        pass

    @property
    @abstractmethod
    def value(self) -> int:
        """Get current fence value."""
        pass

    @abstractmethod
    def wait(self, value: int, timeout_ms: int = -1) -> bool:
        """Wait for fence to reach value."""
        pass

    @abstractmethod
    def is_complete(self, value: int) -> bool:
        """Check if fence has reached value."""
        pass

    @abstractmethod
    def signal(self, value: int) -> None:
        """Signal fence to value (CPU-side)."""
        pass


class NullFence(Fence):
    """Null implementation of Fence."""

    def __init__(self, device: 'Device', initial: int = 0):
        self._device = device
        self._value = initial
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    @classmethod
    def create(cls, device: 'Device', initial: int = 0) -> Fence:
        """Create fence."""
        return cls(device, initial)

    @property
    def value(self) -> int:
        """Get current fence value."""
        with self._lock:
            return self._value

    def wait(self, value: int, timeout_ms: int = -1) -> bool:
        """Wait for fence to reach value."""
        with self._condition:
            if timeout_ms < 0:
                # Infinite wait
                while self._value < value:
                    self._condition.wait()
                return True
            else:
                # Timed wait with deadline tracking
                deadline = _time.monotonic() + (timeout_ms / 1000.0)
                while self._value < value:
                    remaining = deadline - _time.monotonic()
                    if remaining <= 0:
                        return False
                    if not self._condition.wait(remaining):
                        if self._value < value:
                            return False
                return True

    def is_complete(self, value: int) -> bool:
        """Check if fence has reached value."""
        with self._lock:
            return self._value >= value

    def signal(self, value: int) -> None:
        """Signal fence to value (CPU-side)."""
        with self._condition:
            self._value = value
            self._condition.notify_all()


# Forward declarations
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from .device import Device
