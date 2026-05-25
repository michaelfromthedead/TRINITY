"""RHI Device and Adapter abstraction."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List, ClassVar
import threading

from ..constants import (
    NULL_VENDOR_ID, NULL_DEVICE_ID,
    DEFAULT_DISCRETE_VRAM, DEFAULT_SHARED_MEMORY,
    DEFAULT_MAX_TEXTURE_SIZE, DEFAULT_MAX_BUFFER_SIZE
)


class AdapterType(Enum):
    """GPU adapter type."""
    DISCRETE = auto()
    INTEGRATED = auto()
    SOFTWARE = auto()


@dataclass
class AdapterInfo:
    """Information about a GPU adapter."""
    name: str
    dedicated_video_memory: int
    dedicated_system_memory: int
    shared_system_memory: int
    adapter_type: AdapterType
    vendor_id: int
    device_id: int


@dataclass
class FeatureSupport:
    """GPU feature support flags."""
    ray_tracing: bool
    mesh_shaders: bool
    bindless: bool
    compute: bool
    max_texture_size: int
    max_buffer_size: int


@dataclass
class FormatSupport:
    """Format support capabilities."""
    renderable: bool
    filterable: bool
    blendable: bool
    storage: bool
    multisample: bool


class QueueType(Enum):
    """Command queue type."""
    GRAPHICS = auto()
    COMPUTE = auto()
    TRANSFER = auto()


@dataclass
class DeviceConfig:
    """Device creation configuration."""
    adapter: Adapter
    enable_debug: bool = False
    enable_validation: bool = False


class Adapter(ABC):
    """Abstract GPU adapter."""

    @classmethod
    @abstractmethod
    def enumerate(cls) -> List[Adapter]:
        """Enumerate available adapters."""
        pass

    @abstractmethod
    def info(self) -> AdapterInfo:
        """Get adapter information."""
        pass

    @abstractmethod
    def query_features(self) -> FeatureSupport:
        """Query supported features."""
        pass

    @abstractmethod
    def query_format_support(self, format: 'Format') -> FormatSupport:
        """Query format support."""
        pass


class Device(ABC):
    """Abstract GPU device."""

    @classmethod
    @abstractmethod
    def create(cls, adapter: Adapter, config: DeviceConfig) -> Device:
        """Create device from adapter."""
        pass

    @abstractmethod
    def get_queue(self, queue_type: QueueType) -> 'Queue':
        """Get command queue by type."""
        pass

    @abstractmethod
    def create_buffer(self, desc: 'BufferDesc') -> 'Buffer':
        """Create buffer resource."""
        pass

    @abstractmethod
    def create_texture(self, desc: 'TextureDesc') -> 'Texture':
        """Create texture resource."""
        pass

    @abstractmethod
    def create_sampler(self, desc: 'SamplerDesc') -> 'Sampler':
        """Create sampler."""
        pass

    @abstractmethod
    def create_graphics_pipeline(self, desc: 'GraphicsPipelineDesc') -> 'PipelineState':
        """Create graphics pipeline state."""
        pass

    @abstractmethod
    def create_compute_pipeline(self, desc: 'ComputePipelineDesc') -> 'PipelineState':
        """Create compute pipeline state."""
        pass

    @abstractmethod
    def wait_idle(self) -> None:
        """Wait for device to be idle."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown device."""
        pass


class NullAdapter(Adapter):
    """Null implementation of Adapter for testing."""

    _instance_count: ClassVar[int] = 0
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, adapter_type: AdapterType = AdapterType.SOFTWARE):
        with NullAdapter._lock:
            self._id = NullAdapter._instance_count
            NullAdapter._instance_count += 1
        self._adapter_type = adapter_type
        self._info = AdapterInfo(
            name=f"Null Adapter {self._id}",
            dedicated_video_memory=DEFAULT_DISCRETE_VRAM,
            dedicated_system_memory=0,
            shared_system_memory=DEFAULT_SHARED_MEMORY,
            adapter_type=adapter_type,
            vendor_id=NULL_VENDOR_ID,
            device_id=NULL_DEVICE_ID
        )

    @classmethod
    def enumerate(cls) -> List[Adapter]:
        """Enumerate available adapters."""
        return [
            cls(AdapterType.DISCRETE),
            cls(AdapterType.INTEGRATED),
            cls(AdapterType.SOFTWARE)
        ]

    def info(self) -> AdapterInfo:
        """Get adapter information."""
        return self._info

    def query_features(self) -> FeatureSupport:
        """Query supported features."""
        return FeatureSupport(
            ray_tracing=self._adapter_type == AdapterType.DISCRETE,
            mesh_shaders=self._adapter_type == AdapterType.DISCRETE,
            bindless=True,
            compute=True,
            max_texture_size=DEFAULT_MAX_TEXTURE_SIZE,
            max_buffer_size=DEFAULT_MAX_BUFFER_SIZE
        )

    def query_format_support(self, format: 'Format') -> FormatSupport:
        """Query format support."""
        # Most formats are fully supported in null impl
        return FormatSupport(
            renderable=True,
            filterable=True,
            blendable=True,
            storage=True,
            multisample=True
        )


class NullDevice(Device):
    """Null implementation of Device for testing."""

    def __init__(self, adapter: Adapter, config: DeviceConfig):
        self._adapter = adapter
        self._config = config
        self._queues = {
            QueueType.GRAPHICS: None,
            QueueType.COMPUTE: None,
            QueueType.TRANSFER: None
        }
        self._shutdown_called = False

    @classmethod
    def create(cls, adapter: Adapter, config: DeviceConfig) -> Device:
        """Create device from adapter."""
        return cls(adapter, config)

    def get_queue(self, queue_type: QueueType) -> 'Queue':
        """Get command queue by type."""
        from .commands import Queue, NullQueue
        if self._queues[queue_type] is None:
            self._queues[queue_type] = NullQueue(queue_type)
        return self._queues[queue_type]

    def create_buffer(self, desc: 'BufferDesc') -> 'Buffer':
        """Create buffer resource."""
        from .resources import Buffer, NullBuffer
        return NullBuffer(desc)

    def create_texture(self, desc: 'TextureDesc') -> 'Texture':
        """Create texture resource."""
        from .resources import Texture, NullTexture
        return NullTexture(desc)

    def create_sampler(self, desc: 'SamplerDesc') -> 'Sampler':
        """Create sampler."""
        from .resources import Sampler, NullSampler
        return NullSampler(desc)

    def create_graphics_pipeline(self, desc: 'GraphicsPipelineDesc') -> 'PipelineState':
        """Create graphics pipeline state."""
        from .pipeline import PipelineState, NullPipelineState, PipelineType
        return NullPipelineState(desc, PipelineType.GRAPHICS)

    def create_compute_pipeline(self, desc: 'ComputePipelineDesc') -> 'PipelineState':
        """Create compute pipeline state."""
        from .pipeline import PipelineState, NullPipelineState, PipelineType
        return NullPipelineState(desc, PipelineType.COMPUTE)

    def wait_idle(self) -> None:
        """Wait for device to be idle."""
        # Null implementation - nothing to wait for
        pass

    def shutdown(self) -> None:
        """Shutdown device."""
        self._shutdown_called = True
        # Clean up resources using proper API
        for queue in self._queues.values():
            if queue is not None:
                queue.shutdown()


# Forward declarations for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .resources import Buffer, BufferDesc, Texture, TextureDesc, Sampler, SamplerDesc, Format
    from .pipeline import PipelineState, GraphicsPipelineDesc, ComputePipelineDesc
    from .commands import Queue
