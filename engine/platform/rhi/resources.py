"""RHI GPU Resources - buffers, textures, samplers."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, Flag, auto
from typing import Optional
import threading

from ..constants import (
    BUFFER_HANDLE_START, TEXTURE_HANDLE_START,
    SAMPLER_HANDLE_START, DEFAULT_MAX_LOD
)


class BufferUsage(Flag):
    """Buffer usage flags."""
    VERTEX = auto()
    INDEX = auto()
    CONSTANT = auto()
    STORAGE = auto()
    INDIRECT = auto()
    COPY_SRC = auto()
    COPY_DST = auto()


class MemoryType(Enum):
    """Memory type for resource allocation."""
    DEFAULT = auto()  # GPU-only memory
    UPLOAD = auto()   # CPU to GPU
    READBACK = auto() # GPU to CPU


class Format(Enum):
    """Texture/buffer formats."""
    R8_UNORM = auto()
    RG8_UNORM = auto()
    RGBA8_UNORM = auto()
    RGBA16_FLOAT = auto()
    RGBA32_FLOAT = auto()
    R32_FLOAT = auto()
    R32_UINT = auto()
    R16_UINT = auto()
    D32_FLOAT = auto()
    D24_S8 = auto()
    BC7_UNORM = auto()


class TextureType(Enum):
    """Texture dimensionality."""
    TEXTURE_1D = auto()
    TEXTURE_2D = auto()
    TEXTURE_3D = auto()
    TEXTURE_CUBE = auto()
    TEXTURE_ARRAY = auto()


class TextureUsage(Flag):
    """Texture usage flags."""
    SHADER_RESOURCE = auto()
    RENDER_TARGET = auto()
    DEPTH_STENCIL = auto()
    UNORDERED_ACCESS = auto()


class SampleCount(Enum):
    """MSAA sample count."""
    X1 = 1
    X2 = 2
    X4 = 4
    X8 = 8


class FilterMode(Enum):
    """Texture filtering mode."""
    NEAREST = auto()
    LINEAR = auto()


class AddressMode(Enum):
    """Texture address mode."""
    WRAP = auto()
    CLAMP = auto()
    MIRROR = auto()
    BORDER = auto()


class CompareOp(Enum):
    """Comparison operation."""
    NEVER = auto()
    LESS = auto()
    EQUAL = auto()
    LESS_EQUAL = auto()
    GREATER = auto()
    NOT_EQUAL = auto()
    GREATER_EQUAL = auto()
    ALWAYS = auto()


@dataclass
class BufferDesc:
    """Buffer descriptor."""
    size: int
    usage: BufferUsage
    memory_type: MemoryType = MemoryType.DEFAULT
    stride: int = 0


@dataclass
class TextureDesc:
    """Texture descriptor."""
    type: TextureType
    format: Format
    width: int
    height: int = 1
    depth: int = 1
    mip_levels: int = 1
    array_size: int = 1
    sample_count: SampleCount = SampleCount.X1
    usage: TextureUsage = TextureUsage.SHADER_RESOURCE


@dataclass
class SamplerDesc:
    """Sampler descriptor."""
    min_filter: FilterMode = FilterMode.LINEAR
    mag_filter: FilterMode = FilterMode.LINEAR
    mip_filter: FilterMode = FilterMode.LINEAR
    address_u: AddressMode = AddressMode.WRAP
    address_v: AddressMode = AddressMode.WRAP
    address_w: AddressMode = AddressMode.WRAP
    mip_lod_bias: float = 0.0
    max_anisotropy: int = 1
    compare_op: Optional[CompareOp] = None
    min_lod: float = 0.0
    max_lod: float = DEFAULT_MAX_LOD


class Buffer(ABC):
    """Abstract GPU buffer."""

    @property
    @abstractmethod
    def handle(self) -> int:
        """Get native handle."""
        pass

    @property
    @abstractmethod
    def desc(self) -> BufferDesc:
        """Get buffer descriptor."""
        pass

    @abstractmethod
    def destroy(self) -> None:
        """Destroy buffer."""
        pass

    @abstractmethod
    def is_valid(self) -> bool:
        """Check if buffer is valid."""
        pass


class Texture(ABC):
    """Abstract GPU texture."""

    @property
    @abstractmethod
    def handle(self) -> int:
        """Get native handle."""
        pass

    @property
    @abstractmethod
    def desc(self) -> TextureDesc:
        """Get texture descriptor."""
        pass

    @abstractmethod
    def destroy(self) -> None:
        """Destroy texture."""
        pass

    @abstractmethod
    def is_valid(self) -> bool:
        """Check if texture is valid."""
        pass


class Sampler(ABC):
    """Abstract GPU sampler."""

    @property
    @abstractmethod
    def handle(self) -> int:
        """Get native handle."""
        pass

    @property
    @abstractmethod
    def desc(self) -> SamplerDesc:
        """Get sampler descriptor."""
        pass

    @abstractmethod
    def destroy(self) -> None:
        """Destroy sampler."""
        pass

    @abstractmethod
    def is_valid(self) -> bool:
        """Check if sampler is valid."""
        pass


class NullBuffer(Buffer):
    """Null implementation of Buffer."""

    _next_handle = BUFFER_HANDLE_START
    _lock = threading.Lock()

    def __init__(self, desc: BufferDesc):
        self._desc = desc
        with NullBuffer._lock:
            self._handle = NullBuffer._next_handle
            NullBuffer._next_handle += 1
        self._valid = True

    @property
    def handle(self) -> int:
        """Get native handle."""
        return self._handle

    @property
    def desc(self) -> BufferDesc:
        """Get buffer descriptor."""
        return self._desc

    def destroy(self) -> None:
        """Destroy buffer."""
        self._valid = False

    def is_valid(self) -> bool:
        """Check if buffer is valid."""
        return self._valid


class NullTexture(Texture):
    """Null implementation of Texture."""

    _next_handle = TEXTURE_HANDLE_START
    _lock = threading.Lock()

    def __init__(self, desc: TextureDesc):
        self._desc = desc
        with NullTexture._lock:
            self._handle = NullTexture._next_handle
            NullTexture._next_handle += 1
        self._valid = True

    @property
    def handle(self) -> int:
        """Get native handle."""
        return self._handle

    @property
    def desc(self) -> TextureDesc:
        """Get texture descriptor."""
        return self._desc

    def destroy(self) -> None:
        """Destroy texture."""
        self._valid = False

    def is_valid(self) -> bool:
        """Check if texture is valid."""
        return self._valid


class NullSampler(Sampler):
    """Null implementation of Sampler."""

    _next_handle = SAMPLER_HANDLE_START
    _lock = threading.Lock()

    def __init__(self, desc: SamplerDesc):
        self._desc = desc
        with NullSampler._lock:
            self._handle = NullSampler._next_handle
            NullSampler._next_handle += 1
        self._valid = True

    @property
    def handle(self) -> int:
        """Get native handle."""
        return self._handle

    @property
    def desc(self) -> SamplerDesc:
        """Get sampler descriptor."""
        return self._desc

    def destroy(self) -> None:
        """Destroy sampler."""
        self._valid = False

    def is_valid(self) -> bool:
        """Check if sampler is valid."""
        return self._valid
