"""RHI Descriptor binding system."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional
import threading


class DescriptorType(Enum):
    """Descriptor type."""
    CBV = auto()      # Constant buffer view
    SRV = auto()      # Shader resource view
    UAV = auto()      # Unordered access view
    SAMPLER = auto()  # Sampler


@dataclass
class DescriptorHandle:
    """Descriptor handle."""
    heap_index: int
    offset: int


class DescriptorHeap(ABC):
    """Abstract descriptor heap."""

    @classmethod
    @abstractmethod
    def create(cls, device: 'Device', desc_type: DescriptorType, count: int) -> DescriptorHeap:
        """Create descriptor heap."""
        pass

    @abstractmethod
    def allocate(self) -> Optional[DescriptorHandle]:
        """Allocate descriptor handle."""
        pass

    @abstractmethod
    def free(self, handle: DescriptorHandle) -> None:
        """Free descriptor handle."""
        pass


class NullDescriptorHeap(DescriptorHeap):
    """Null implementation of DescriptorHeap."""

    _next_heap_index = 0
    _heap_lock = threading.Lock()

    def __init__(self, device: 'Device', desc_type: DescriptorType, count: int):
        self._device = device
        self._type = desc_type
        self._count = count
        self._next_offset = 0
        self._free_list: List[int] = []
        self._lock = threading.Lock()

        with NullDescriptorHeap._heap_lock:
            self._heap_index = NullDescriptorHeap._next_heap_index
            NullDescriptorHeap._next_heap_index += 1

    @classmethod
    def create(cls, device: 'Device', desc_type: DescriptorType, count: int) -> DescriptorHeap:
        """Create descriptor heap."""
        return cls(device, desc_type, count)

    def allocate(self) -> Optional[DescriptorHandle]:
        """Allocate descriptor handle."""
        with self._lock:
            # Try to reuse from free list
            if self._free_list:
                offset = self._free_list.pop()
                return DescriptorHandle(self._heap_index, offset)

            # Allocate new if space available
            if self._next_offset < self._count:
                offset = self._next_offset
                self._next_offset += 1
                return DescriptorHandle(self._heap_index, offset)

            # Heap is full
            return None

    def free(self, handle: DescriptorHandle) -> None:
        """Free descriptor handle."""
        with self._lock:
            if handle.heap_index == self._heap_index:
                self._free_list.append(handle.offset)


# Forward declarations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .device import Device
