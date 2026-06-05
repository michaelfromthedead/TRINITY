"""Low-tier GPU memory budget manager.

T-CC-0.11: Enforces 256MB GPU memory limit with constraints:
- Max texture: 1024x1024
- Max render target: 1280x720
- ETC2/ASTC compressed textures
- Reduced draw call budget
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Protocol, Tuple, Callable
import math


class TextureFormat(Enum):
    """Texture compression formats."""
    RGBA8 = auto()      # 32 bpp uncompressed
    RGB8 = auto()       # 24 bpp uncompressed
    ETC2_RGB = auto()   # 4 bpp compressed
    ETC2_RGBA = auto()  # 8 bpp compressed
    ASTC_4x4 = auto()   # 8 bpp compressed
    ASTC_6x6 = auto()   # ~3.5 bpp compressed
    ASTC_8x8 = auto()   # 2 bpp compressed
    BC1 = auto()        # 4 bpp DXT1
    BC3 = auto()        # 8 bpp DXT5
    BC7 = auto()        # 8 bpp
    R8 = auto()         # 8 bpp single channel
    RG8 = auto()        # 16 bpp two channel
    R16F = auto()       # 16 bpp half float
    RGBA16F = auto()    # 64 bpp half float
    RGBA32F = auto()    # 128 bpp full float


BITS_PER_PIXEL: Dict[TextureFormat, float] = {
    TextureFormat.RGBA8: 32,
    TextureFormat.RGB8: 24,
    TextureFormat.ETC2_RGB: 4,
    TextureFormat.ETC2_RGBA: 8,
    TextureFormat.ASTC_4x4: 8,
    TextureFormat.ASTC_6x6: 3.56,
    TextureFormat.ASTC_8x8: 2,
    TextureFormat.BC1: 4,
    TextureFormat.BC3: 8,
    TextureFormat.BC7: 8,
    TextureFormat.R8: 8,
    TextureFormat.RG8: 16,
    TextureFormat.R16F: 16,
    TextureFormat.RGBA16F: 64,
    TextureFormat.RGBA32F: 128,
}


MOBILE_FORMATS = {TextureFormat.ETC2_RGB, TextureFormat.ETC2_RGBA,
                  TextureFormat.ASTC_4x4, TextureFormat.ASTC_6x6, TextureFormat.ASTC_8x8}


@dataclass(frozen=True)
class MemoryBudgetConfig:
    """Configuration for memory budget constraints."""
    max_gpu_memory_mb: int = 256
    max_texture_size: int = 1024
    max_render_target_width: int = 1280
    max_render_target_height: int = 720
    max_draw_calls_per_frame: int = 500
    max_simultaneous_textures: int = 64
    require_compressed_textures: bool = True
    allowed_formats: Tuple[TextureFormat, ...] = (
        TextureFormat.ETC2_RGB,
        TextureFormat.ETC2_RGBA,
        TextureFormat.ASTC_4x4,
        TextureFormat.ASTC_6x6,
        TextureFormat.ASTC_8x8,
        TextureFormat.R8,
        TextureFormat.RG8,
    )
    mipmap_budget_ratio: float = 1.33  # Full mip chain adds ~33%


@dataclass
class TextureAllocation:
    """Tracks a single texture allocation."""
    name: str
    width: int
    height: int
    format: TextureFormat
    mip_levels: int = 1
    array_layers: int = 1

    @property
    def size_bytes(self) -> int:
        bpp = BITS_PER_PIXEL.get(self.format, 32)
        base_size = int((self.width * self.height * bpp) / 8)
        mip_multiplier = 1.0
        if self.mip_levels > 1:
            mip_multiplier = sum(1 / (4 ** i) for i in range(self.mip_levels))
        return int(base_size * mip_multiplier * self.array_layers)


@dataclass
class RenderTargetAllocation:
    """Tracks a render target allocation."""
    name: str
    width: int
    height: int
    format: TextureFormat
    samples: int = 1

    @property
    def size_bytes(self) -> int:
        bpp = BITS_PER_PIXEL.get(self.format, 32)
        return int((self.width * self.height * bpp * self.samples) / 8)


@dataclass
class BufferAllocation:
    """Tracks a GPU buffer allocation."""
    name: str
    size_bytes: int
    usage: str = "generic"


class AllocationResult(Enum):
    """Result of an allocation attempt."""
    SUCCESS = auto()
    EXCEEDS_BUDGET = auto()
    EXCEEDS_SIZE_LIMIT = auto()
    INVALID_FORMAT = auto()
    EXCEEDS_COUNT_LIMIT = auto()


@dataclass
class AllocationResponse:
    """Response from allocation attempt."""
    result: AllocationResult
    message: str
    suggested_size: Optional[Tuple[int, int]] = None
    suggested_format: Optional[TextureFormat] = None


@dataclass
class MemoryStats:
    """Current memory usage statistics."""
    texture_memory_bytes: int = 0
    render_target_memory_bytes: int = 0
    buffer_memory_bytes: int = 0
    texture_count: int = 0
    render_target_count: int = 0
    buffer_count: int = 0

    @property
    def total_bytes(self) -> int:
        return self.texture_memory_bytes + self.render_target_memory_bytes + self.buffer_memory_bytes

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 * 1024)


class EvictionPolicy(Protocol):
    """Protocol for eviction policies."""
    def select_for_eviction(
        self,
        textures: Dict[str, TextureAllocation],
        needed_bytes: int
    ) -> List[str]: ...


class LRUEvictionPolicy:
    """Least recently used eviction policy."""

    def __init__(self):
        self._access_order: List[str] = []

    def record_access(self, name: str) -> None:
        if name in self._access_order:
            self._access_order.remove(name)
        self._access_order.append(name)

    def select_for_eviction(
        self,
        textures: Dict[str, TextureAllocation],
        needed_bytes: int
    ) -> List[str]:
        evict_list: List[str] = []
        freed = 0
        for name in self._access_order:
            if name in textures:
                evict_list.append(name)
                freed += textures[name].size_bytes
                if freed >= needed_bytes:
                    break
        return evict_list


class SizeBasedEvictionPolicy:
    """Evict largest textures first."""

    def select_for_eviction(
        self,
        textures: Dict[str, TextureAllocation],
        needed_bytes: int
    ) -> List[str]:
        sorted_textures = sorted(
            textures.items(),
            key=lambda x: x[1].size_bytes,
            reverse=True
        )
        evict_list: List[str] = []
        freed = 0
        for name, alloc in sorted_textures:
            evict_list.append(name)
            freed += alloc.size_bytes
            if freed >= needed_bytes:
                break
        return evict_list


class MemoryBudgetManager:
    """Manages GPU memory budget for low-tier rendering."""

    def __init__(
        self,
        config: Optional[MemoryBudgetConfig] = None,
        eviction_policy: Optional[EvictionPolicy] = None
    ):
        self._config = config or MemoryBudgetConfig()
        self._eviction_policy = eviction_policy or LRUEvictionPolicy()
        self._textures: Dict[str, TextureAllocation] = {}
        self._render_targets: Dict[str, RenderTargetAllocation] = {}
        self._buffers: Dict[str, BufferAllocation] = {}
        self._draw_call_count: int = 0
        self._frame_number: int = 0
        self._eviction_callbacks: List[Callable[[str], None]] = []

    @property
    def config(self) -> MemoryBudgetConfig:
        return self._config

    @property
    def stats(self) -> MemoryStats:
        return MemoryStats(
            texture_memory_bytes=sum(t.size_bytes for t in self._textures.values()),
            render_target_memory_bytes=sum(rt.size_bytes for rt in self._render_targets.values()),
            buffer_memory_bytes=sum(b.size_bytes for b in self._buffers.values()),
            texture_count=len(self._textures),
            render_target_count=len(self._render_targets),
            buffer_count=len(self._buffers),
        )

    @property
    def available_bytes(self) -> int:
        budget = self._config.max_gpu_memory_mb * 1024 * 1024
        return max(0, budget - self.stats.total_bytes)

    @property
    def utilization_percent(self) -> float:
        budget = self._config.max_gpu_memory_mb * 1024 * 1024
        return (self.stats.total_bytes / budget) * 100 if budget > 0 else 0

    def register_eviction_callback(self, callback: Callable[[str], None]) -> None:
        self._eviction_callbacks.append(callback)

    def _notify_eviction(self, name: str) -> None:
        for cb in self._eviction_callbacks:
            cb(name)

    def validate_texture(
        self,
        width: int,
        height: int,
        format: TextureFormat
    ) -> AllocationResponse:
        max_size = self._config.max_texture_size
        if width > max_size or height > max_size:
            suggested = (min(width, max_size), min(height, max_size))
            return AllocationResponse(
                AllocationResult.EXCEEDS_SIZE_LIMIT,
                f"Texture size {width}x{height} exceeds max {max_size}x{max_size}",
                suggested_size=suggested,
            )

        if self._config.require_compressed_textures:
            if format not in self._config.allowed_formats:
                suggested = TextureFormat.ETC2_RGBA if format in (
                    TextureFormat.RGBA8, TextureFormat.RGBA16F, TextureFormat.RGBA32F
                ) else TextureFormat.ETC2_RGB
                return AllocationResponse(
                    AllocationResult.INVALID_FORMAT,
                    f"Format {format.name} not allowed, use compressed format",
                    suggested_format=suggested,
                )

        if len(self._textures) >= self._config.max_simultaneous_textures:
            return AllocationResponse(
                AllocationResult.EXCEEDS_COUNT_LIMIT,
                f"Texture count {len(self._textures)} exceeds limit {self._config.max_simultaneous_textures}",
            )

        return AllocationResponse(AllocationResult.SUCCESS, "Validation passed")

    def validate_render_target(
        self,
        width: int,
        height: int,
        format: TextureFormat
    ) -> AllocationResponse:
        max_w = self._config.max_render_target_width
        max_h = self._config.max_render_target_height
        if width > max_w or height > max_h:
            suggested = (min(width, max_w), min(height, max_h))
            return AllocationResponse(
                AllocationResult.EXCEEDS_SIZE_LIMIT,
                f"RT size {width}x{height} exceeds max {max_w}x{max_h}",
                suggested_size=suggested,
            )
        return AllocationResponse(AllocationResult.SUCCESS, "Validation passed")

    def allocate_texture(
        self,
        name: str,
        width: int,
        height: int,
        format: TextureFormat,
        mip_levels: int = 1,
        array_layers: int = 1,
        auto_evict: bool = True,
    ) -> AllocationResponse:
        validation = self.validate_texture(width, height, format)
        if validation.result != AllocationResult.SUCCESS:
            return validation

        alloc = TextureAllocation(name, width, height, format, mip_levels, array_layers)

        if alloc.size_bytes > self.available_bytes:
            if auto_evict:
                needed = alloc.size_bytes - self.available_bytes
                evict_list = self._eviction_policy.select_for_eviction(self._textures, needed)
                for evict_name in evict_list:
                    self.free_texture(evict_name)
                    self._notify_eviction(evict_name)
                if alloc.size_bytes > self.available_bytes:
                    return AllocationResponse(
                        AllocationResult.EXCEEDS_BUDGET,
                        f"Texture {alloc.size_bytes} bytes exceeds remaining budget after eviction",
                    )
            else:
                return AllocationResponse(
                    AllocationResult.EXCEEDS_BUDGET,
                    f"Texture {alloc.size_bytes} bytes exceeds available {self.available_bytes}",
                )

        self._textures[name] = alloc
        if isinstance(self._eviction_policy, LRUEvictionPolicy):
            self._eviction_policy.record_access(name)

        return AllocationResponse(AllocationResult.SUCCESS, f"Allocated {alloc.size_bytes} bytes")

    def allocate_render_target(
        self,
        name: str,
        width: int,
        height: int,
        format: TextureFormat,
        samples: int = 1,
    ) -> AllocationResponse:
        validation = self.validate_render_target(width, height, format)
        if validation.result != AllocationResult.SUCCESS:
            return validation

        alloc = RenderTargetAllocation(name, width, height, format, samples)

        if alloc.size_bytes > self.available_bytes:
            return AllocationResponse(
                AllocationResult.EXCEEDS_BUDGET,
                f"RT {alloc.size_bytes} bytes exceeds available {self.available_bytes}",
            )

        self._render_targets[name] = alloc
        return AllocationResponse(AllocationResult.SUCCESS, f"Allocated {alloc.size_bytes} bytes")

    def allocate_buffer(
        self,
        name: str,
        size_bytes: int,
        usage: str = "generic",
    ) -> AllocationResponse:
        if size_bytes > self.available_bytes:
            return AllocationResponse(
                AllocationResult.EXCEEDS_BUDGET,
                f"Buffer {size_bytes} bytes exceeds available {self.available_bytes}",
            )

        self._buffers[name] = BufferAllocation(name, size_bytes, usage)
        return AllocationResponse(AllocationResult.SUCCESS, f"Allocated {size_bytes} bytes")

    def free_texture(self, name: str) -> bool:
        if name in self._textures:
            del self._textures[name]
            return True
        return False

    def free_render_target(self, name: str) -> bool:
        if name in self._render_targets:
            del self._render_targets[name]
            return True
        return False

    def free_buffer(self, name: str) -> bool:
        if name in self._buffers:
            del self._buffers[name]
            return True
        return False

    def access_texture(self, name: str) -> bool:
        if name in self._textures:
            if isinstance(self._eviction_policy, LRUEvictionPolicy):
                self._eviction_policy.record_access(name)
            return True
        return False

    def begin_frame(self) -> None:
        self._draw_call_count = 0
        self._frame_number += 1

    def record_draw_call(self, count: int = 1) -> bool:
        self._draw_call_count += count
        return self._draw_call_count <= self._config.max_draw_calls_per_frame

    @property
    def draw_call_budget_remaining(self) -> int:
        return max(0, self._config.max_draw_calls_per_frame - self._draw_call_count)

    @property
    def draw_calls_this_frame(self) -> int:
        return self._draw_call_count

    def get_texture(self, name: str) -> Optional[TextureAllocation]:
        return self._textures.get(name)

    def get_render_target(self, name: str) -> Optional[RenderTargetAllocation]:
        return self._render_targets.get(name)

    def get_buffer(self, name: str) -> Optional[BufferAllocation]:
        return self._buffers.get(name)

    def clear_all(self) -> None:
        self._textures.clear()
        self._render_targets.clear()
        self._buffers.clear()
        self._draw_call_count = 0


def create_low_tier_budget() -> MemoryBudgetManager:
    """Factory for standard low-tier budget manager."""
    return MemoryBudgetManager(MemoryBudgetConfig())


def create_medium_tier_budget() -> MemoryBudgetManager:
    """Factory for medium-tier budget (512MB, larger textures)."""
    return MemoryBudgetManager(MemoryBudgetConfig(
        max_gpu_memory_mb=512,
        max_texture_size=2048,
        max_render_target_width=1920,
        max_render_target_height=1080,
        max_draw_calls_per_frame=2000,
        max_simultaneous_textures=256,
        require_compressed_textures=False,
        allowed_formats=tuple(TextureFormat),
    ))


def create_high_tier_budget() -> MemoryBudgetManager:
    """Factory for high-tier budget (2GB, no practical limits)."""
    return MemoryBudgetManager(MemoryBudgetConfig(
        max_gpu_memory_mb=2048,
        max_texture_size=4096,
        max_render_target_width=3840,
        max_render_target_height=2160,
        max_draw_calls_per_frame=10000,
        max_simultaneous_textures=1024,
        require_compressed_textures=False,
        allowed_formats=tuple(TextureFormat),
    ))


def estimate_mipmap_memory(width: int, height: int, format: TextureFormat) -> int:
    """Estimate total memory for full mipmap chain."""
    bpp = BITS_PER_PIXEL.get(format, 32)
    total = 0
    w, h = width, height
    while w >= 1 and h >= 1:
        total += int((w * h * bpp) / 8)
        w //= 2
        h //= 2
    return total


def suggest_texture_size(
    original_width: int,
    original_height: int,
    max_size: int,
    preserve_aspect: bool = True,
) -> Tuple[int, int]:
    """Suggest texture size that fits within limit."""
    if original_width <= max_size and original_height <= max_size:
        return (original_width, original_height)

    if preserve_aspect:
        scale = min(max_size / original_width, max_size / original_height)
        return (
            int(original_width * scale),
            int(original_height * scale),
        )
    return (min(original_width, max_size), min(original_height, max_size))


def power_of_two_size(size: int) -> int:
    """Round up to next power of two."""
    return 1 << (size - 1).bit_length() if size > 0 else 1
