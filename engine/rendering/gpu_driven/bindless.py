"""
Bindless Resource Management for GPU-driven rendering.

Implements bindless textures and buffers using descriptor indexing,
enabling GPU-driven rendering without per-draw binding changes.

References:
- RENDERING_CONTEXT.md Section 6.2 GPU-Driven Rendering Pipeline
"""

from __future__ import annotations

import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Generic, Optional, TypeVar


# =============================================================================
# RESOURCE HANDLE
# =============================================================================


class ResourceType(IntEnum):
    """Types of bindless resources."""
    TEXTURE_2D = auto()
    TEXTURE_3D = auto()
    TEXTURE_CUBE = auto()
    TEXTURE_2D_ARRAY = auto()
    BUFFER = auto()
    SAMPLER = auto()
    ACCELERATION_STRUCTURE = auto()


@dataclass(frozen=True, slots=True)
class ResourceHandle:
    """
    Handle for bindless resource access.

    The handle contains an index into the descriptor heap/table,
    enabling GPU shaders to access resources without explicit bindings.
    """
    index: int  # Index in descriptor table
    resource_type: ResourceType
    generation: int = 0  # For handle validation

    # Invalid handle constant
    INVALID_INDEX: int = 0xFFFFFFFF

    @classmethod
    def invalid(cls) -> "ResourceHandle":
        """Create an invalid handle."""
        return cls(
            index=cls.INVALID_INDEX,
            resource_type=ResourceType.TEXTURE_2D,
            generation=0,
        )

    @property
    def is_valid(self) -> bool:
        """Check if handle is valid."""
        return self.index != self.INVALID_INDEX

    def to_shader_index(self) -> int:
        """Get index for use in shaders."""
        return self.index

    def to_bytes(self) -> bytes:
        """Pack handle for GPU buffer."""
        return struct.pack("<I", self.index)


# =============================================================================
# TEXTURE DESCRIPTOR
# =============================================================================


class TextureFormat(IntEnum):
    """Common texture formats."""
    RGBA8_UNORM = auto()
    RGBA8_SRGB = auto()
    RGBA16_FLOAT = auto()
    RGBA32_FLOAT = auto()
    R8_UNORM = auto()
    R16_FLOAT = auto()
    R32_FLOAT = auto()
    RG8_UNORM = auto()
    RG16_FLOAT = auto()
    RG32_FLOAT = auto()
    BC1_UNORM = auto()  # DXT1
    BC3_UNORM = auto()  # DXT5
    BC5_UNORM = auto()  # Normal maps
    BC7_UNORM = auto()  # High quality
    DEPTH32_FLOAT = auto()
    DEPTH24_STENCIL8 = auto()


@dataclass
class TextureDescriptor:
    """
    Descriptor for a bindless texture.

    Contains all information needed to sample the texture in shaders.
    """
    # Texture identification
    name: str = ""
    handle: ResourceHandle = field(default_factory=ResourceHandle.invalid)

    # Dimensions
    width: int = 1
    height: int = 1
    depth: int = 1
    mip_levels: int = 1
    array_layers: int = 1

    # Format
    format: TextureFormat = TextureFormat.RGBA8_UNORM

    # Resource type
    resource_type: ResourceType = ResourceType.TEXTURE_2D

    # GPU resource ID (platform-specific)
    gpu_resource_id: int = 0

    # Sampler index (for combined image/sampler)
    sampler_index: int = 0

    @property
    def is_array(self) -> bool:
        return self.array_layers > 1

    @property
    def is_3d(self) -> bool:
        return self.resource_type == ResourceType.TEXTURE_3D

    @property
    def is_cube(self) -> bool:
        return self.resource_type == ResourceType.TEXTURE_CUBE


# =============================================================================
# BUFFER DESCRIPTOR
# =============================================================================


class BufferUsage(IntEnum):
    """Buffer usage flags."""
    VERTEX = 1 << 0
    INDEX = 1 << 1
    UNIFORM = 1 << 2
    STORAGE = 1 << 3
    INDIRECT = 1 << 4
    TRANSFER_SRC = 1 << 5
    TRANSFER_DST = 1 << 6


@dataclass
class BufferDescriptor:
    """
    Descriptor for a bindless buffer.

    Enables shader access to arbitrary buffer data via index.
    """
    # Buffer identification
    name: str = ""
    handle: ResourceHandle = field(default_factory=ResourceHandle.invalid)

    # Size and structure
    size_bytes: int = 0
    stride_bytes: int = 0  # Element stride for structured buffers

    # Usage
    usage: int = BufferUsage.STORAGE

    # GPU resource ID
    gpu_resource_id: int = 0

    # GPU address (for buffer device address)
    gpu_address: int = 0

    @property
    def element_count(self) -> int:
        """Get number of elements (for structured buffers)."""
        if self.stride_bytes == 0:
            return 0
        return self.size_bytes // self.stride_bytes


# =============================================================================
# SAMPLER DESCRIPTOR
# =============================================================================


class FilterMode(IntEnum):
    """Texture filter modes."""
    NEAREST = auto()
    LINEAR = auto()


class AddressMode(IntEnum):
    """Texture address modes."""
    REPEAT = auto()
    MIRRORED_REPEAT = auto()
    CLAMP_TO_EDGE = auto()
    CLAMP_TO_BORDER = auto()


@dataclass
class SamplerDescriptor:
    """
    Descriptor for a bindless sampler.
    """
    # Sampler identification
    name: str = ""
    handle: ResourceHandle = field(default_factory=ResourceHandle.invalid)

    # Filtering
    min_filter: FilterMode = FilterMode.LINEAR
    mag_filter: FilterMode = FilterMode.LINEAR
    mip_filter: FilterMode = FilterMode.LINEAR

    # Addressing
    address_u: AddressMode = AddressMode.REPEAT
    address_v: AddressMode = AddressMode.REPEAT
    address_w: AddressMode = AddressMode.REPEAT

    # Anisotropy
    max_anisotropy: float = 1.0

    # LOD
    min_lod: float = 0.0
    max_lod: float = 1000.0
    lod_bias: float = 0.0

    # Comparison (for shadow maps)
    compare_enable: bool = False
    compare_op: int = 0  # Platform-specific


# =============================================================================
# BINDLESS TEXTURE MANAGER
# =============================================================================


@dataclass
class BindlessTextureManagerConfig:
    """Configuration for bindless texture manager."""
    max_textures: int = 16384
    max_samplers: int = 2048
    enable_validation: bool = True


class BindlessTextureManager:
    """
    Manages bindless texture descriptors.

    Maintains a global texture descriptor heap that shaders can index into
    using texture handles. This eliminates per-draw texture binding overhead.

    Usage:
    1. Register textures to get handles
    2. Pass handles to GPU via instance data or material buffer
    3. Shaders use handles to sample textures from global descriptor array
    """

    def __init__(self, config: Optional[BindlessTextureManagerConfig] = None) -> None:
        self._config = config or BindlessTextureManagerConfig()

        # Texture descriptors by index
        self._textures: dict[int, TextureDescriptor] = {}
        self._samplers: dict[int, SamplerDescriptor] = {}

        # Free list for recycling indices
        self._free_texture_indices: list[int] = list(
            range(self._config.max_textures - 1, -1, -1)
        )
        self._free_sampler_indices: list[int] = list(
            range(self._config.max_samplers - 1, -1, -1)
        )

        # Generation counters for handle validation
        self._texture_generations: dict[int, int] = {}
        self._sampler_generations: dict[int, int] = {}

        # Name lookup
        self._texture_by_name: dict[str, int] = {}
        self._sampler_by_name: dict[str, int] = {}

    @property
    def config(self) -> BindlessTextureManagerConfig:
        return self._config

    @property
    def texture_count(self) -> int:
        return len(self._textures)

    @property
    def sampler_count(self) -> int:
        return len(self._samplers)

    def register_texture(
        self,
        name: str,
        width: int,
        height: int,
        format: TextureFormat = TextureFormat.RGBA8_UNORM,
        mip_levels: int = 1,
        array_layers: int = 1,
        resource_type: ResourceType = ResourceType.TEXTURE_2D,
        gpu_resource_id: int = 0,
    ) -> ResourceHandle:
        """
        Register a texture and get a bindless handle.

        Args:
            name: Texture name for lookup
            width: Texture width
            height: Texture height
            format: Texture format
            mip_levels: Number of mip levels
            array_layers: Number of array layers
            resource_type: Type of texture resource
            gpu_resource_id: Platform-specific GPU resource ID

        Returns:
            Handle for bindless access
        """
        if not self._free_texture_indices:
            return ResourceHandle.invalid()

        # Allocate index
        index = self._free_texture_indices.pop()
        generation = self._texture_generations.get(index, 0) + 1
        self._texture_generations[index] = generation

        # Create handle
        handle = ResourceHandle(
            index=index,
            resource_type=resource_type,
            generation=generation,
        )

        # Create descriptor
        descriptor = TextureDescriptor(
            name=name,
            handle=handle,
            width=width,
            height=height,
            mip_levels=mip_levels,
            array_layers=array_layers,
            format=format,
            resource_type=resource_type,
            gpu_resource_id=gpu_resource_id,
        )

        self._textures[index] = descriptor
        self._texture_by_name[name] = index

        return handle

    def unregister_texture(self, handle: ResourceHandle) -> bool:
        """
        Unregister a texture and free its handle.

        Returns True if texture was unregistered.
        """
        if not self._validate_texture_handle(handle):
            return False

        index = handle.index
        descriptor = self._textures.pop(index, None)
        if descriptor:
            self._texture_by_name.pop(descriptor.name, None)
            self._free_texture_indices.append(index)
            return True

        return False

    def get_texture(self, handle: ResourceHandle) -> Optional[TextureDescriptor]:
        """Get texture descriptor by handle."""
        if not self._validate_texture_handle(handle):
            return None
        return self._textures.get(handle.index)

    def get_texture_by_name(self, name: str) -> Optional[TextureDescriptor]:
        """Get texture descriptor by name."""
        index = self._texture_by_name.get(name)
        if index is None:
            return None
        return self._textures.get(index)

    def register_sampler(
        self,
        name: str,
        min_filter: FilterMode = FilterMode.LINEAR,
        mag_filter: FilterMode = FilterMode.LINEAR,
        address_mode: AddressMode = AddressMode.REPEAT,
        max_anisotropy: float = 1.0,
    ) -> ResourceHandle:
        """
        Register a sampler and get a bindless handle.
        """
        if not self._free_sampler_indices:
            return ResourceHandle.invalid()

        index = self._free_sampler_indices.pop()
        generation = self._sampler_generations.get(index, 0) + 1
        self._sampler_generations[index] = generation

        handle = ResourceHandle(
            index=index,
            resource_type=ResourceType.SAMPLER,
            generation=generation,
        )

        descriptor = SamplerDescriptor(
            name=name,
            handle=handle,
            min_filter=min_filter,
            mag_filter=mag_filter,
            address_u=address_mode,
            address_v=address_mode,
            address_w=address_mode,
            max_anisotropy=max_anisotropy,
        )

        self._samplers[index] = descriptor
        self._sampler_by_name[name] = index

        return handle

    def get_sampler(self, handle: ResourceHandle) -> Optional[SamplerDescriptor]:
        """Get sampler descriptor by handle."""
        if handle.resource_type != ResourceType.SAMPLER:
            return None
        return self._samplers.get(handle.index)

    def _validate_texture_handle(self, handle: ResourceHandle) -> bool:
        """Validate a texture handle."""
        if not handle.is_valid:
            return False
        if handle.index not in self._textures:
            return False
        if self._config.enable_validation:
            expected_gen = self._texture_generations.get(handle.index, 0)
            if handle.generation != expected_gen:
                return False
        return True

    def build_descriptor_table(self) -> list[TextureDescriptor]:
        """
        Build a contiguous descriptor table for GPU upload.

        Returns list indexed by handle.index.
        """
        max_index = max(self._textures.keys(), default=-1) + 1
        table: list[Optional[TextureDescriptor]] = [None] * max_index

        for index, descriptor in self._textures.items():
            table[index] = descriptor

        # Replace None with placeholder
        placeholder = TextureDescriptor(name="__invalid__")
        return [d if d is not None else placeholder for d in table]


# =============================================================================
# BINDLESS BUFFER MANAGER
# =============================================================================


@dataclass
class BindlessBufferManagerConfig:
    """Configuration for bindless buffer manager."""
    max_buffers: int = 8192
    enable_validation: bool = True
    use_buffer_device_address: bool = True


class BindlessBufferManager:
    """
    Manages bindless buffer descriptors.

    Enables shaders to access arbitrary buffers using handles/indices.
    Supports both descriptor indexing and buffer device address approaches.
    """

    def __init__(self, config: Optional[BindlessBufferManagerConfig] = None) -> None:
        self._config = config or BindlessBufferManagerConfig()

        self._buffers: dict[int, BufferDescriptor] = {}
        self._free_indices: list[int] = list(
            range(self._config.max_buffers - 1, -1, -1)
        )
        self._generations: dict[int, int] = {}
        self._buffer_by_name: dict[str, int] = {}

    @property
    def config(self) -> BindlessBufferManagerConfig:
        return self._config

    @property
    def buffer_count(self) -> int:
        return len(self._buffers)

    def register_buffer(
        self,
        name: str,
        size_bytes: int,
        stride_bytes: int = 0,
        usage: int = BufferUsage.STORAGE,
        gpu_resource_id: int = 0,
        gpu_address: int = 0,
    ) -> ResourceHandle:
        """
        Register a buffer and get a bindless handle.

        Args:
            name: Buffer name for lookup
            size_bytes: Buffer size in bytes
            stride_bytes: Element stride for structured buffers
            usage: Buffer usage flags
            gpu_resource_id: Platform-specific GPU resource ID
            gpu_address: GPU virtual address (for BDA)

        Returns:
            Handle for bindless access
        """
        if not self._free_indices:
            return ResourceHandle.invalid()

        index = self._free_indices.pop()
        generation = self._generations.get(index, 0) + 1
        self._generations[index] = generation

        handle = ResourceHandle(
            index=index,
            resource_type=ResourceType.BUFFER,
            generation=generation,
        )

        descriptor = BufferDescriptor(
            name=name,
            handle=handle,
            size_bytes=size_bytes,
            stride_bytes=stride_bytes,
            usage=usage,
            gpu_resource_id=gpu_resource_id,
            gpu_address=gpu_address,
        )

        self._buffers[index] = descriptor
        self._buffer_by_name[name] = index

        return handle

    def unregister_buffer(self, handle: ResourceHandle) -> bool:
        """Unregister a buffer and free its handle."""
        if not self._validate_handle(handle):
            return False

        index = handle.index
        descriptor = self._buffers.pop(index, None)
        if descriptor:
            self._buffer_by_name.pop(descriptor.name, None)
            self._free_indices.append(index)
            return True

        return False

    def get_buffer(self, handle: ResourceHandle) -> Optional[BufferDescriptor]:
        """Get buffer descriptor by handle."""
        if not self._validate_handle(handle):
            return None
        return self._buffers.get(handle.index)

    def get_buffer_by_name(self, name: str) -> Optional[BufferDescriptor]:
        """Get buffer descriptor by name."""
        index = self._buffer_by_name.get(name)
        if index is None:
            return None
        return self._buffers.get(index)

    def get_gpu_address(self, handle: ResourceHandle) -> int:
        """Get GPU virtual address for buffer device address access."""
        buffer = self.get_buffer(handle)
        return buffer.gpu_address if buffer else 0

    def _validate_handle(self, handle: ResourceHandle) -> bool:
        """Validate a buffer handle."""
        if not handle.is_valid:
            return False
        if handle.resource_type != ResourceType.BUFFER:
            return False
        if handle.index not in self._buffers:
            return False
        if self._config.enable_validation:
            expected_gen = self._generations.get(handle.index, 0)
            if handle.generation != expected_gen:
                return False
        return True

    def build_address_buffer(self) -> bytes:
        """
        Build a buffer of GPU addresses for shader access.

        Returns packed buffer addresses for BDA-style access.
        """
        max_index = max(self._buffers.keys(), default=-1) + 1
        data = bytearray(max_index * 8)  # 64-bit addresses

        for index, descriptor in self._buffers.items():
            offset = index * 8
            struct.pack_into("<Q", data, offset, descriptor.gpu_address)

        return bytes(data)


# =============================================================================
# MATERIAL RESOURCE TABLE
# =============================================================================


@dataclass
class MaterialResources:
    """
    Resource bindings for a material.

    Contains handles to all textures used by a material.
    """
    material_id: int = 0

    # Standard PBR textures
    albedo_texture: ResourceHandle = field(default_factory=ResourceHandle.invalid)
    normal_texture: ResourceHandle = field(default_factory=ResourceHandle.invalid)
    metallic_roughness_texture: ResourceHandle = field(default_factory=ResourceHandle.invalid)
    ao_texture: ResourceHandle = field(default_factory=ResourceHandle.invalid)
    emissive_texture: ResourceHandle = field(default_factory=ResourceHandle.invalid)

    # Additional textures
    detail_texture: ResourceHandle = field(default_factory=ResourceHandle.invalid)
    subsurface_texture: ResourceHandle = field(default_factory=ResourceHandle.invalid)

    # Sampler
    sampler: ResourceHandle = field(default_factory=ResourceHandle.invalid)

    def to_gpu_format(self) -> bytes:
        """Pack material resources for GPU buffer."""
        return struct.pack(
            "<8I",
            self.albedo_texture.index if self.albedo_texture.is_valid else 0xFFFFFFFF,
            self.normal_texture.index if self.normal_texture.is_valid else 0xFFFFFFFF,
            self.metallic_roughness_texture.index if self.metallic_roughness_texture.is_valid else 0xFFFFFFFF,
            self.ao_texture.index if self.ao_texture.is_valid else 0xFFFFFFFF,
            self.emissive_texture.index if self.emissive_texture.is_valid else 0xFFFFFFFF,
            self.detail_texture.index if self.detail_texture.is_valid else 0xFFFFFFFF,
            self.subsurface_texture.index if self.subsurface_texture.is_valid else 0xFFFFFFFF,
            self.sampler.index if self.sampler.is_valid else 0xFFFFFFFF,
        )

    @classmethod
    def byte_size(cls) -> int:
        return 32  # 8 * sizeof(uint32)


class MaterialResourceTable:
    """
    Table of material resource bindings for GPU access.

    Enables shaders to look up material textures by material ID.
    """

    def __init__(self, max_materials: int = 4096) -> None:
        self._max_materials = max_materials
        self._materials: dict[int, MaterialResources] = {}

    @property
    def material_count(self) -> int:
        return len(self._materials)

    def register_material(self, resources: MaterialResources) -> None:
        """Register material resource bindings."""
        self._materials[resources.material_id] = resources

    def unregister_material(self, material_id: int) -> bool:
        """Unregister material resources."""
        return self._materials.pop(material_id, None) is not None

    def get_material(self, material_id: int) -> Optional[MaterialResources]:
        """Get material resources by ID."""
        return self._materials.get(material_id)

    def build_gpu_buffer(self) -> bytes:
        """Build GPU buffer containing all material resource bindings."""
        data = bytearray(self._max_materials * MaterialResources.byte_size())

        for mat_id, resources in self._materials.items():
            if mat_id < self._max_materials:
                offset = mat_id * MaterialResources.byte_size()
                packed = resources.to_gpu_format()
                data[offset:offset + len(packed)] = packed

        return bytes(data)


# =============================================================================
# BINDLESS RESOURCE SYSTEM
# =============================================================================


class BindlessResourceSystem:
    """
    Unified bindless resource management system.

    Coordinates texture, buffer, and sampler management for
    efficient GPU-driven rendering.
    """

    def __init__(
        self,
        texture_config: Optional[BindlessTextureManagerConfig] = None,
        buffer_config: Optional[BindlessBufferManagerConfig] = None,
        max_materials: int = 4096,
    ) -> None:
        self._texture_manager = BindlessTextureManager(texture_config)
        self._buffer_manager = BindlessBufferManager(buffer_config)
        self._material_table = MaterialResourceTable(max_materials)

    @property
    def textures(self) -> BindlessTextureManager:
        return self._texture_manager

    @property
    def buffers(self) -> BindlessBufferManager:
        return self._buffer_manager

    @property
    def materials(self) -> MaterialResourceTable:
        return self._material_table

    def build_all_buffers(self) -> tuple[bytes, bytes, bytes]:
        """
        Build all GPU buffers for bindless rendering.

        Returns:
            Tuple of (material_buffer, address_buffer, texture_descriptor_count)
        """
        material_buffer = self._material_table.build_gpu_buffer()
        address_buffer = self._buffer_manager.build_address_buffer()

        # For texture descriptors, we'd typically use platform-specific
        # descriptor heap updates. This returns a count for now.
        tex_count = self._texture_manager.texture_count

        return material_buffer, address_buffer, struct.pack("<I", tex_count)


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Resource types
    "ResourceType",
    "ResourceHandle",
    # Texture
    "TextureFormat",
    "TextureDescriptor",
    "BindlessTextureManagerConfig",
    "BindlessTextureManager",
    # Buffer
    "BufferUsage",
    "BufferDescriptor",
    "BindlessBufferManagerConfig",
    "BindlessBufferManager",
    # Sampler
    "FilterMode",
    "AddressMode",
    "SamplerDescriptor",
    # Material
    "MaterialResources",
    "MaterialResourceTable",
    # System
    "BindlessResourceSystem",
]
