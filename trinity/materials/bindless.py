"""Bindless Texture Arrays for TRINITY Material System (T-MAT-5.7).

This module provides Python-side management of bindless texture arrays using
WebGPU's binding_array feature. Textures are indexed via the material table's
texture_index fields rather than individual per-draw bindings.

Core Components:
    TextureSlot: Metadata for a single texture in the bindless array
    BindlessTextureArray: Manager for the GPU bindless texture array
    BindlessCapabilities: Runtime capability detection for bindless support

WebGPU Requirements:
    - `binding_array<texture_2d<f32>>` support
    - `maxTexturesPerShaderStage` >= desired array size
    - Fallback to traditional bindings when limits exceeded

Integration Points:
    - MaterialTable: texture_index fields reference BindlessTextureArray slots
    - TextureTable (Rust): CPU-side metadata shadows GPU texture array
    - PipelineIntegration: Bindless pipelines use shared bind group layout

Example::

    from trinity.materials.bindless import BindlessTextureArray, TextureSlot

    # Create bindless array with capability detection
    array = BindlessTextureArray(max_textures=1024)

    # Register textures
    albedo_slot = array.register(
        TextureSlot(
            name="player_albedo",
            width=1024,
            height=1024,
            format=TextureFormat.RGBA8_UNORM_SRGB,
            mip_levels=10,
        )
    )

    # Access in shader via material table
    # material_table[idx].albedo_texture_id == albedo_slot.index

Notes:
    The bindless array is capped at MAX_BINDLESS_TEXTURES (4096) to match
    typical hardware limits. When bindless is unavailable, the fallback mode
    uses traditional per-draw texture bindings with a performance cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Dict, List, Optional, Callable, Any, Tuple, Set
import hashlib


# =============================================================================
# Constants
# =============================================================================

# Maximum number of textures in the bindless array (matches hardware limits)
MAX_BINDLESS_TEXTURES: int = 4096

# Default initial capacity for the bindless texture array
DEFAULT_BINDLESS_CAPACITY: int = 1024

# Sentinel value indicating no texture bound (matches WGSL shader conventions)
INVALID_TEXTURE_INDEX: int = 0xFFFFFFFF

# Minimum WebGPU limit for bindless support
MIN_TEXTURES_FOR_BINDLESS: int = 16


# =============================================================================
# Texture Format Enum
# =============================================================================


class TextureFormat(IntEnum):
    """Texture format identifiers (matches wgpu::TextureFormat subset)."""
    RGBA8_UNORM = 0
    RGBA8_UNORM_SRGB = 1
    RGBA8_SNORM = 2
    RGBA16_FLOAT = 3
    RGBA32_FLOAT = 4
    R8_UNORM = 5
    RG8_UNORM = 6
    R16_FLOAT = 7
    RG16_FLOAT = 8
    R32_FLOAT = 9
    RG32_FLOAT = 10
    DEPTH32_FLOAT = 11
    DEPTH24_PLUS = 12
    DEPTH24_PLUS_STENCIL8 = 13
    BC1_RGBA_UNORM = 14
    BC1_RGBA_UNORM_SRGB = 15
    BC3_RGBA_UNORM = 16
    BC3_RGBA_UNORM_SRGB = 17
    BC5_RG_UNORM = 18
    BC7_RGBA_UNORM = 19
    BC7_RGBA_UNORM_SRGB = 20

    def to_wgsl(self) -> str:
        """Convert to WGSL format string for texture declarations."""
        format_map = {
            TextureFormat.RGBA8_UNORM: "rgba8unorm",
            TextureFormat.RGBA8_UNORM_SRGB: "rgba8unorm-srgb",
            TextureFormat.RGBA8_SNORM: "rgba8snorm",
            TextureFormat.RGBA16_FLOAT: "rgba16float",
            TextureFormat.RGBA32_FLOAT: "rgba32float",
            TextureFormat.R8_UNORM: "r8unorm",
            TextureFormat.RG8_UNORM: "rg8unorm",
            TextureFormat.R16_FLOAT: "r16float",
            TextureFormat.RG16_FLOAT: "rg16float",
            TextureFormat.R32_FLOAT: "r32float",
            TextureFormat.RG32_FLOAT: "rg32float",
            TextureFormat.DEPTH32_FLOAT: "depth32float",
            TextureFormat.DEPTH24_PLUS: "depth24plus",
            TextureFormat.DEPTH24_PLUS_STENCIL8: "depth24plus-stencil8",
        }
        return format_map.get(self, "rgba8unorm")

    def is_srgb(self) -> bool:
        """Return True if this format uses sRGB color space."""
        return self in (
            TextureFormat.RGBA8_UNORM_SRGB,
            TextureFormat.BC1_RGBA_UNORM_SRGB,
            TextureFormat.BC3_RGBA_UNORM_SRGB,
            TextureFormat.BC7_RGBA_UNORM_SRGB,
        )

    def is_compressed(self) -> bool:
        """Return True if this is a block-compressed format."""
        return self >= TextureFormat.BC1_RGBA_UNORM

    def bytes_per_pixel(self) -> float:
        """Return bytes per pixel (fractional for compressed formats)."""
        bpp_map = {
            TextureFormat.RGBA8_UNORM: 4.0,
            TextureFormat.RGBA8_UNORM_SRGB: 4.0,
            TextureFormat.RGBA8_SNORM: 4.0,
            TextureFormat.RGBA16_FLOAT: 8.0,
            TextureFormat.RGBA32_FLOAT: 16.0,
            TextureFormat.R8_UNORM: 1.0,
            TextureFormat.RG8_UNORM: 2.0,
            TextureFormat.R16_FLOAT: 2.0,
            TextureFormat.RG16_FLOAT: 4.0,
            TextureFormat.R32_FLOAT: 4.0,
            TextureFormat.RG32_FLOAT: 8.0,
            TextureFormat.DEPTH32_FLOAT: 4.0,
            TextureFormat.DEPTH24_PLUS: 4.0,
            TextureFormat.DEPTH24_PLUS_STENCIL8: 4.0,
            # BC formats: bits per 4x4 block / 16 pixels
            TextureFormat.BC1_RGBA_UNORM: 0.5,
            TextureFormat.BC1_RGBA_UNORM_SRGB: 0.5,
            TextureFormat.BC3_RGBA_UNORM: 1.0,
            TextureFormat.BC3_RGBA_UNORM_SRGB: 1.0,
            TextureFormat.BC5_RG_UNORM: 1.0,
            TextureFormat.BC7_RGBA_UNORM: 1.0,
            TextureFormat.BC7_RGBA_UNORM_SRGB: 1.0,
        }
        return bpp_map.get(self, 4.0)


# =============================================================================
# Sampler Configuration
# =============================================================================


class FilterMode(IntEnum):
    """Texture filtering modes."""
    NEAREST = 0
    LINEAR = 1


class AddressMode(IntEnum):
    """Texture addressing modes for out-of-bounds UVs."""
    REPEAT = 0
    MIRROR_REPEAT = 1
    CLAMP_TO_EDGE = 2

    def to_wgsl(self) -> str:
        """Convert to WGSL address mode string."""
        return ["repeat", "mirror_repeat", "clamp_to_edge"][self]


@dataclass
class SamplerConfig:
    """Configuration for a texture sampler.

    Attributes:
        mag_filter: Magnification filter mode.
        min_filter: Minification filter mode.
        mipmap_filter: Mipmap filter mode.
        address_u: U-axis address mode.
        address_v: V-axis address mode.
        address_w: W-axis address mode (for 3D/array textures).
        lod_min_clamp: Minimum LOD level.
        lod_max_clamp: Maximum LOD level.
        max_anisotropy: Maximum anisotropic filtering level (1-16).
    """
    mag_filter: FilterMode = FilterMode.LINEAR
    min_filter: FilterMode = FilterMode.LINEAR
    mipmap_filter: FilterMode = FilterMode.LINEAR
    address_u: AddressMode = AddressMode.REPEAT
    address_v: AddressMode = AddressMode.REPEAT
    address_w: AddressMode = AddressMode.REPEAT
    lod_min_clamp: float = 0.0
    lod_max_clamp: float = 32.0
    max_anisotropy: int = 1

    def content_hash(self) -> str:
        """Compute a hash for sampler deduplication."""
        data = (
            f"{self.mag_filter},{self.min_filter},{self.mipmap_filter},"
            f"{self.address_u},{self.address_v},{self.address_w},"
            f"{self.lod_min_clamp},{self.lod_max_clamp},{self.max_anisotropy}"
        )
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    @classmethod
    def linear_repeat(cls) -> "SamplerConfig":
        """Standard linear filtering with repeat wrapping."""
        return cls()

    @classmethod
    def linear_clamp(cls) -> "SamplerConfig":
        """Linear filtering with edge clamping."""
        return cls(
            address_u=AddressMode.CLAMP_TO_EDGE,
            address_v=AddressMode.CLAMP_TO_EDGE,
            address_w=AddressMode.CLAMP_TO_EDGE,
        )

    @classmethod
    def nearest_repeat(cls) -> "SamplerConfig":
        """Point filtering with repeat wrapping."""
        return cls(
            mag_filter=FilterMode.NEAREST,
            min_filter=FilterMode.NEAREST,
            mipmap_filter=FilterMode.NEAREST,
        )

    @classmethod
    def anisotropic(cls, level: int = 16) -> "SamplerConfig":
        """Anisotropic filtering with repeat wrapping."""
        return cls(max_anisotropy=min(max(level, 1), 16))


# =============================================================================
# Texture Slot
# =============================================================================


@dataclass
class TextureSlot:
    """Metadata for a single texture in the bindless array.

    Each TextureSlot describes one texture registered in the
    BindlessTextureArray. The slot's `index` field is assigned during
    registration and is used by shaders to sample the texture.

    Attributes:
        name: Human-readable identifier for debugging.
        width: Texture width in pixels.
        height: Texture height in pixels.
        format: Pixel format (TextureFormat enum).
        mip_levels: Number of mipmap levels (1 = no mipmaps).
        layer_count: Number of array layers (1 for 2D textures).
        sampler: Sampler configuration for this texture.
        index: Assigned bindless array index (set by BindlessTextureArray).
        valid: Whether the slot contains valid texture data.
        source_path: Optional path to source texture file.
        content_hash: Optional content hash for deduplication.
        gpu_handle: Optional GPU resource handle (wgpu::Texture).
    """
    name: str
    width: int
    height: int
    format: TextureFormat = TextureFormat.RGBA8_UNORM
    mip_levels: int = 1
    layer_count: int = 1
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    index: int = INVALID_TEXTURE_INDEX
    valid: bool = False
    source_path: Optional[str] = None
    content_hash: Optional[str] = None
    gpu_handle: Optional[Any] = None

    def byte_size(self, include_mipmaps: bool = True) -> int:
        """Calculate total memory size in bytes.

        Args:
            include_mipmaps: Include mipmap chain in calculation.

        Returns:
            Total bytes for this texture.
        """
        base_size = int(self.width * self.height * self.format.bytes_per_pixel())
        base_size *= self.layer_count

        if not include_mipmaps or self.mip_levels <= 1:
            return base_size

        # Mipmap chain: sum of 1/4^n for n=0..mip_levels-1 = 4/3 * base
        # More accurate: sum each level
        total = base_size
        w, h = self.width // 2, self.height // 2
        for _ in range(1, self.mip_levels):
            if w < 1:
                w = 1
            if h < 1:
                h = 1
            total += int(w * h * self.format.bytes_per_pixel() * self.layer_count)
            w //= 2
            h //= 2
        return total

    def max_mip_levels(self) -> int:
        """Calculate the maximum possible mip levels for this texture size."""
        import math
        return int(math.floor(math.log2(max(self.width, self.height)))) + 1

    def is_power_of_two(self) -> bool:
        """Return True if dimensions are powers of two."""
        return (
            (self.width & (self.width - 1)) == 0
            and (self.height & (self.height - 1)) == 0
        )

    def aspect_ratio(self) -> float:
        """Return width/height aspect ratio."""
        return self.width / self.height if self.height > 0 else 1.0

    def __hash__(self) -> int:
        return hash((self.name, self.index))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TextureSlot):
            return NotImplemented
        return self.name == other.name and self.index == other.index


# =============================================================================
# Bindless Capabilities
# =============================================================================


@dataclass
class BindlessCapabilities:
    """Runtime capability detection for bindless texture support.

    This class queries the GPU adapter limits to determine whether
    bindless texture arrays are supported and what limits apply.

    Attributes:
        max_textures_per_shader_stage: WebGPU limit for texture bindings.
        max_samplers_per_shader_stage: WebGPU limit for sampler bindings.
        max_storage_textures_per_shader_stage: Storage texture limit.
        supports_binding_array: Whether binding_array is supported.
        supports_partially_bound: Whether partially bound arrays are allowed.
        effective_max_bindless: Practical limit for bindless array size.
        fallback_mode: Whether fallback to traditional bindings is active.
    """
    max_textures_per_shader_stage: int = 16
    max_samplers_per_shader_stage: int = 16
    max_storage_textures_per_shader_stage: int = 8
    supports_binding_array: bool = False
    supports_partially_bound: bool = False
    effective_max_bindless: int = 0
    fallback_mode: bool = True

    @classmethod
    def detect(cls, adapter_limits: Optional[Dict[str, int]] = None) -> "BindlessCapabilities":
        """Detect bindless capabilities from adapter limits.

        Args:
            adapter_limits: Dictionary of WebGPU adapter limits.
                If None, uses conservative defaults (fallback mode).

        Returns:
            BindlessCapabilities instance with detected values.
        """
        if adapter_limits is None:
            # Conservative fallback: assume minimal WebGPU support
            return cls()

        max_tex = adapter_limits.get("maxTexturesPerShaderStage", 16)
        max_samp = adapter_limits.get("maxSamplersPerShaderStage", 16)
        max_storage = adapter_limits.get("maxStorageTexturesPerShaderStage", 8)

        # Check for binding_array support (indicated by high texture limit)
        supports_binding_array = max_tex >= MIN_TEXTURES_FOR_BINDLESS
        supports_partially_bound = adapter_limits.get("partiallyBoundArrays", False)

        # Effective limit: min of texture limit and our hard cap
        effective_max = min(max_tex, MAX_BINDLESS_TEXTURES) if supports_binding_array else 0

        return cls(
            max_textures_per_shader_stage=max_tex,
            max_samplers_per_shader_stage=max_samp,
            max_storage_textures_per_shader_stage=max_storage,
            supports_binding_array=supports_binding_array,
            supports_partially_bound=supports_partially_bound,
            effective_max_bindless=effective_max,
            fallback_mode=not supports_binding_array,
        )

    def is_bindless_viable(self, required_textures: int) -> bool:
        """Check if bindless mode can accommodate the required textures.

        Args:
            required_textures: Number of textures needed.

        Returns:
            True if bindless can handle the requirement.
        """
        return (
            self.supports_binding_array
            and required_textures <= self.effective_max_bindless
        )


# =============================================================================
# Bindless Texture Array Statistics
# =============================================================================


@dataclass
class BindlessArrayStats:
    """Statistics for bindless texture array usage.

    Attributes:
        total_slots: Maximum number of slots in the array.
        used_slots: Number of slots currently in use.
        free_slots: Number of slots available in free list.
        total_memory_bytes: Total GPU memory used by all textures.
        largest_texture_bytes: Size of the largest texture.
        format_histogram: Count of textures by format.
        mip_level_histogram: Count of textures by mip level count.
    """
    total_slots: int = 0
    used_slots: int = 0
    free_slots: int = 0
    total_memory_bytes: int = 0
    largest_texture_bytes: int = 0
    format_histogram: Dict[TextureFormat, int] = field(default_factory=dict)
    mip_level_histogram: Dict[int, int] = field(default_factory=dict)

    def utilization(self) -> float:
        """Return slot utilization as percentage [0.0, 100.0]."""
        if self.total_slots == 0:
            return 0.0
        return (self.used_slots / self.total_slots) * 100.0

    def memory_mb(self) -> float:
        """Return total memory in megabytes."""
        return self.total_memory_bytes / (1024 * 1024)


# =============================================================================
# Bindless Texture Array
# =============================================================================


class BindlessTextureArray:
    """Manager for the GPU bindless texture array.

    This class manages a bindless texture array using WebGPU's binding_array
    feature. Textures are registered and assigned indices that can be used
    by shaders to sample from the array without per-draw binding changes.

    The array uses a free-list allocation strategy for O(1) slot reuse.
    When bindless is not supported, the class falls back to traditional
    per-draw bindings with reduced performance.

    Attributes:
        max_textures: Maximum number of textures in the array.
        capabilities: Detected bindless capabilities.
        fallback_mode: Whether using traditional bindings (no bindless).

    Example::

        # Create with capability detection
        array = BindlessTextureArray(max_textures=1024)

        # Register a texture
        slot = TextureSlot(
            name="brick_albedo",
            width=512,
            height=512,
            format=TextureFormat.RGBA8_UNORM_SRGB,
            mip_levels=9,
        )
        index = array.register(slot)

        # Use in material: material.albedo_texture_id = index

        # Remove when done
        array.remove(index)
    """

    def __init__(
        self,
        max_textures: int = DEFAULT_BINDLESS_CAPACITY,
        capabilities: Optional[BindlessCapabilities] = None,
        on_fallback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Create a new bindless texture array.

        Args:
            max_textures: Maximum number of textures to support.
            capabilities: Pre-detected capabilities (detects if None).
            on_fallback: Callback when falling back to traditional bindings.
        """
        self._max_textures = min(max_textures, MAX_BINDLESS_TEXTURES)
        self._capabilities = capabilities or BindlessCapabilities.detect()
        self._on_fallback = on_fallback

        # Slot storage
        self._slots: Dict[int, TextureSlot] = {}
        self._name_to_index: Dict[str, int] = {}
        self._hash_to_index: Dict[str, int] = {}

        # Free-list for O(1) allocation
        self._free_list: List[int] = []
        self._next_index: int = 0

        # Sampler deduplication
        self._sampler_cache: Dict[str, int] = {}
        self._next_sampler_index: int = 0

        # Dirty tracking for GPU upload
        self._dirty_indices: Set[int] = set()

        # Check fallback mode
        self._fallback_mode = self._capabilities.fallback_mode
        if self._fallback_mode and self._on_fallback:
            self._on_fallback(
                f"Bindless textures unavailable: using fallback mode "
                f"(max_textures_per_stage={self._capabilities.max_textures_per_shader_stage})"
            )

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def max_textures(self) -> int:
        """Maximum number of textures the array can hold."""
        return self._max_textures

    @property
    def capabilities(self) -> BindlessCapabilities:
        """Runtime bindless capabilities."""
        return self._capabilities

    @property
    def fallback_mode(self) -> bool:
        """True if using traditional bindings instead of bindless."""
        return self._fallback_mode

    @property
    def count(self) -> int:
        """Number of textures currently registered."""
        return len(self._slots)

    @property
    def is_empty(self) -> bool:
        """True if no textures are registered."""
        return len(self._slots) == 0

    @property
    def is_full(self) -> bool:
        """True if the array has reached maximum capacity."""
        return (
            self._next_index >= self._max_textures
            and len(self._free_list) == 0
        )

    @property
    def dirty_count(self) -> int:
        """Number of slots needing GPU upload."""
        return len(self._dirty_indices)

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    def register(self, slot: TextureSlot) -> int:
        """Register a texture in the bindless array.

        Assigns an index to the texture slot and marks it for GPU upload.
        If a texture with the same name already exists, it is replaced.

        Args:
            slot: TextureSlot describing the texture.

        Returns:
            The assigned bindless array index.

        Raises:
            RuntimeError: If the array is full and no free slots available.
        """
        # Check for existing texture with same name
        if slot.name in self._name_to_index:
            existing_index = self._name_to_index[slot.name]
            self._update_slot(existing_index, slot)
            return existing_index

        # Check for duplicate by content hash
        if slot.content_hash and slot.content_hash in self._hash_to_index:
            return self._hash_to_index[slot.content_hash]

        # Allocate new index
        if self._free_list:
            index = self._free_list.pop()
        elif self._next_index < self._max_textures:
            index = self._next_index
            self._next_index += 1
        else:
            raise RuntimeError(
                f"Bindless texture array full: {self._max_textures} slots used"
            )

        # Update slot and register
        slot.index = index
        slot.valid = True
        self._slots[index] = slot
        self._name_to_index[slot.name] = index
        if slot.content_hash:
            self._hash_to_index[slot.content_hash] = index

        # Register sampler
        self._register_sampler(slot.sampler)

        # Mark dirty for GPU upload
        self._dirty_indices.add(index)

        return index

    def _update_slot(self, index: int, slot: TextureSlot) -> None:
        """Update an existing slot with new texture data."""
        old_slot = self._slots.get(index)
        if old_slot and old_slot.content_hash:
            self._hash_to_index.pop(old_slot.content_hash, None)

        slot.index = index
        slot.valid = True
        self._slots[index] = slot
        self._name_to_index[slot.name] = index
        if slot.content_hash:
            self._hash_to_index[slot.content_hash] = index

        self._register_sampler(slot.sampler)
        self._dirty_indices.add(index)

    def _register_sampler(self, config: SamplerConfig) -> int:
        """Register a sampler configuration, deduplicating identical configs."""
        hash_key = config.content_hash()
        if hash_key in self._sampler_cache:
            return self._sampler_cache[hash_key]

        index = self._next_sampler_index
        self._next_sampler_index += 1
        self._sampler_cache[hash_key] = index
        return index

    def remove(self, index: int) -> bool:
        """Remove a texture from the bindless array.

        The slot is zeroed and returned to the free-list for reuse.

        Args:
            index: Bindless array index to remove.

        Returns:
            True if the texture was removed, False if not found.
        """
        slot = self._slots.pop(index, None)
        if slot is None:
            return False

        # Clear name and hash mappings
        self._name_to_index.pop(slot.name, None)
        if slot.content_hash:
            self._hash_to_index.pop(slot.content_hash, None)

        # Return to free-list
        self._free_list.append(index)
        self._dirty_indices.add(index)

        return True

    def remove_by_name(self, name: str) -> bool:
        """Remove a texture by name.

        Args:
            name: Texture name to remove.

        Returns:
            True if the texture was removed, False if not found.
        """
        index = self._name_to_index.get(name)
        if index is None:
            return False
        return self.remove(index)

    # -------------------------------------------------------------------------
    # Access
    # -------------------------------------------------------------------------

    def get(self, index: int) -> Optional[TextureSlot]:
        """Get a texture slot by index.

        Args:
            index: Bindless array index.

        Returns:
            TextureSlot if found, None otherwise.
        """
        return self._slots.get(index)

    def get_by_name(self, name: str) -> Optional[TextureSlot]:
        """Get a texture slot by name.

        Args:
            name: Texture name.

        Returns:
            TextureSlot if found, None otherwise.
        """
        index = self._name_to_index.get(name)
        if index is None:
            return None
        return self._slots.get(index)

    def get_index(self, name: str) -> int:
        """Get the bindless array index for a texture name.

        Args:
            name: Texture name.

        Returns:
            Bindless array index, or INVALID_TEXTURE_INDEX if not found.
        """
        return self._name_to_index.get(name, INVALID_TEXTURE_INDEX)

    def contains(self, index: int) -> bool:
        """Check if an index has a registered texture."""
        return index in self._slots

    def contains_name(self, name: str) -> bool:
        """Check if a texture with the given name is registered."""
        return name in self._name_to_index

    def iter_slots(self):
        """Iterate over all registered texture slots."""
        return iter(self._slots.values())

    def iter_indices(self):
        """Iterate over all registered indices."""
        return iter(self._slots.keys())

    # -------------------------------------------------------------------------
    # Dirty Tracking
    # -------------------------------------------------------------------------

    def mark_dirty(self, index: int) -> None:
        """Mark a slot as needing GPU upload.

        Args:
            index: Bindless array index to mark dirty.
        """
        if index in self._slots:
            self._dirty_indices.add(index)

    def clear_dirty(self) -> Set[int]:
        """Clear and return the set of dirty indices.

        Returns:
            Set of indices that were marked dirty.
        """
        dirty = self._dirty_indices.copy()
        self._dirty_indices.clear()
        return dirty

    def any_dirty(self) -> bool:
        """Return True if any slots need GPU upload."""
        return len(self._dirty_indices) > 0

    def get_dirty_indices(self) -> Set[int]:
        """Get the current set of dirty indices (does not clear)."""
        return self._dirty_indices.copy()

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_stats(self) -> BindlessArrayStats:
        """Compute current statistics for the bindless array.

        Returns:
            BindlessArrayStats with current usage information.
        """
        stats = BindlessArrayStats(
            total_slots=self._max_textures,
            used_slots=len(self._slots),
            free_slots=len(self._free_list),
        )

        format_hist: Dict[TextureFormat, int] = {}
        mip_hist: Dict[int, int] = {}
        largest = 0

        for slot in self._slots.values():
            size = slot.byte_size()
            stats.total_memory_bytes += size
            if size > largest:
                largest = size

            format_hist[slot.format] = format_hist.get(slot.format, 0) + 1
            mip_hist[slot.mip_levels] = mip_hist.get(slot.mip_levels, 0) + 1

        stats.largest_texture_bytes = largest
        stats.format_histogram = format_hist
        stats.mip_level_histogram = mip_hist

        return stats

    # -------------------------------------------------------------------------
    # WGSL Generation
    # -------------------------------------------------------------------------

    def generate_wgsl_declarations(
        self,
        group: int = 1,
        texture_binding: int = 0,
        sampler_binding: int = 1,
    ) -> str:
        """Generate WGSL binding declarations for the bindless array.

        Args:
            group: Binding group for the declarations.
            texture_binding: Binding index for the texture array.
            sampler_binding: Binding index for the sampler array.

        Returns:
            WGSL binding declaration string.
        """
        if self._fallback_mode:
            return self._generate_fallback_wgsl(group, texture_binding)

        num_samplers = len(self._sampler_cache)
        if num_samplers == 0:
            num_samplers = 1  # At least one sampler

        return f"""// Bindless texture array (T-MAT-5.7)
@group({group}) @binding({texture_binding})
var textures: binding_array<texture_2d<f32>, {self._max_textures}>;

@group({group}) @binding({sampler_binding})
var samplers: binding_array<sampler, {num_samplers}>;
"""

    def _generate_fallback_wgsl(self, group: int, base_binding: int) -> str:
        """Generate WGSL for fallback mode (traditional bindings)."""
        lines = [f"// Fallback texture bindings (bindless unavailable)"]
        for i, slot in enumerate(self._slots.values()):
            tex_binding = base_binding + i * 2
            samp_binding = base_binding + i * 2 + 1
            lines.append(
                f"@group({group}) @binding({tex_binding}) "
                f"var {slot.name}_tex: texture_2d<f32>;"
            )
            lines.append(
                f"@group({group}) @binding({samp_binding}) "
                f"var {slot.name}_sampler: sampler;"
            )
        return "\n".join(lines)

    def generate_wgsl_sample_function(self) -> str:
        """Generate WGSL helper function for sampling bindless textures.

        Returns:
            WGSL function definition string.
        """
        if self._fallback_mode:
            return self._generate_fallback_sample_function()

        return """/// Sample from the bindless texture array.
/// @param texture_index: Index into the bindless texture array.
/// @param sampler_index: Index into the sampler array.
/// @param uv: Texture coordinates.
/// @returns: Sampled RGBA color.
fn sample_bindless(texture_index: u32, sampler_index: u32, uv: vec2<f32>) -> vec4<f32> {
    // Validate index bounds (u32::MAX = no texture)
    if texture_index == 0xFFFFFFFFu {
        return vec4<f32>(1.0, 0.0, 1.0, 1.0); // Magenta error color
    }
    return textureSample(textures[texture_index], samplers[sampler_index], uv);
}

/// Sample from bindless array with default sampler (index 0).
fn sample_bindless_default(texture_index: u32, uv: vec2<f32>) -> vec4<f32> {
    return sample_bindless(texture_index, 0u, uv);
}

/// Sample from bindless array with LOD selection.
fn sample_bindless_lod(
    texture_index: u32,
    sampler_index: u32,
    uv: vec2<f32>,
    lod: f32
) -> vec4<f32> {
    if texture_index == 0xFFFFFFFFu {
        return vec4<f32>(1.0, 0.0, 1.0, 1.0);
    }
    return textureSampleLevel(textures[texture_index], samplers[sampler_index], uv, lod);
}

/// Sample from bindless array with gradient for anisotropic filtering.
fn sample_bindless_grad(
    texture_index: u32,
    sampler_index: u32,
    uv: vec2<f32>,
    ddx: vec2<f32>,
    ddy: vec2<f32>
) -> vec4<f32> {
    if texture_index == 0xFFFFFFFFu {
        return vec4<f32>(1.0, 0.0, 1.0, 1.0);
    }
    return textureSampleGrad(textures[texture_index], samplers[sampler_index], uv, ddx, ddy);
}
"""

    def _generate_fallback_sample_function(self) -> str:
        """Generate sample function for fallback mode."""
        return """// Fallback sampling - textures bound individually
// Use traditional textureSample() calls with named textures
"""

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_bytes(self) -> bytes:
        """Serialize texture metadata for GPU upload.

        Returns byte representation matching the Rust TextureTableEntry
        format (24 bytes per entry).

        Returns:
            Raw bytes for GPU buffer upload.
        """
        import struct

        # TextureTableEntry layout: width, height, mip_levels, format, layer_count, flags
        # Each field is u32 (4 bytes), total 24 bytes per entry
        ENTRY_SIZE = 24

        # Create buffer for all slots up to next_index
        buffer = bytearray(self._next_index * ENTRY_SIZE)

        for index, slot in self._slots.items():
            offset = index * ENTRY_SIZE
            flags = 1 if slot.valid else 0  # bit 0 = valid

            struct.pack_into(
                "<IIIIII",  # 6 x uint32, little-endian
                buffer,
                offset,
                slot.width,
                slot.height,
                slot.mip_levels,
                int(slot.format),
                slot.layer_count,
                flags,
            )

        return bytes(buffer)

    # -------------------------------------------------------------------------
    # Clearing
    # -------------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all textures from the array."""
        self._slots.clear()
        self._name_to_index.clear()
        self._hash_to_index.clear()
        self._free_list.clear()
        self._next_index = 0
        self._sampler_cache.clear()
        self._next_sampler_index = 0
        self._dirty_indices.clear()

    def reserve(self, count: int) -> None:
        """Reserve capacity for additional textures.

        This is a hint for pre-allocation and does not affect the
        maximum texture limit.

        Args:
            count: Number of additional textures to reserve space for.
        """
        # Python dicts grow automatically, so this is a no-op
        # but we track intent for potential future optimization
        pass


# =============================================================================
# Factory Functions
# =============================================================================


def create_bindless_array(
    max_textures: int = DEFAULT_BINDLESS_CAPACITY,
    adapter_limits: Optional[Dict[str, int]] = None,
) -> BindlessTextureArray:
    """Create a bindless texture array with automatic capability detection.

    Args:
        max_textures: Maximum number of textures to support.
        adapter_limits: Optional WebGPU adapter limits dictionary.

    Returns:
        Configured BindlessTextureArray instance.
    """
    caps = BindlessCapabilities.detect(adapter_limits)
    return BindlessTextureArray(max_textures=max_textures, capabilities=caps)


def create_default_slots() -> Dict[str, TextureSlot]:
    """Create default texture slots for fallback rendering.

    Returns dictionary of 1x1 default textures:
    - white: (1, 1, 1, 1) for missing albedo
    - black: (0, 0, 0, 1) for missing metallic
    - normal: (0.5, 0.5, 1, 1) for flat normals
    - gray: (0.5, 0.5, 0.5, 1) for missing roughness

    Returns:
        Dictionary mapping name to TextureSlot.
    """
    defaults = {
        "white": TextureSlot(
            name="white",
            width=1,
            height=1,
            format=TextureFormat.RGBA8_UNORM,
            mip_levels=1,
        ),
        "black": TextureSlot(
            name="black",
            width=1,
            height=1,
            format=TextureFormat.RGBA8_UNORM,
            mip_levels=1,
        ),
        "normal": TextureSlot(
            name="normal",
            width=1,
            height=1,
            format=TextureFormat.RGBA8_UNORM,
            mip_levels=1,
        ),
        "gray": TextureSlot(
            name="gray",
            width=1,
            height=1,
            format=TextureFormat.RGBA8_UNORM,
            mip_levels=1,
        ),
    }
    return defaults


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Constants
    "MAX_BINDLESS_TEXTURES",
    "DEFAULT_BINDLESS_CAPACITY",
    "INVALID_TEXTURE_INDEX",
    "MIN_TEXTURES_FOR_BINDLESS",
    # Enums
    "TextureFormat",
    "FilterMode",
    "AddressMode",
    # Data classes
    "SamplerConfig",
    "TextureSlot",
    "BindlessCapabilities",
    "BindlessArrayStats",
    # Main class
    "BindlessTextureArray",
    # Factory functions
    "create_bindless_array",
    "create_default_slots",
]
