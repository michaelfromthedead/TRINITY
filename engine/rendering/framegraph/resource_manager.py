"""
Resource management for the Frame Graph.

This module implements resource allocation and tracking for the frame graph system,
as specified in RENDERING_CONTEXT.md Section 6.1.

Resource Types (from spec):
- Transient resources: Allocated per-frame, aliased across passes for memory efficiency
- History resources: Persisted across frames (TAA history, GI accumulators)
- External resources: Swap chain backbuffer, imported textures

The resource manager handles:
- Resource allocation and deallocation
- Memory aliasing for transient resources
- Lifetime tracking for history resources
- External resource registration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional
from uuid import uuid4


class ResourceType(Enum):
    """Classification of frame graph resources."""

    TRANSIENT = auto()
    """Per-frame resource that can be aliased with other transients."""

    HISTORY = auto()
    """Persisted across frames (TAA history, GI accumulators)."""

    EXTERNAL = auto()
    """External resources like swap chain backbuffer, imported textures."""


class ResourceFormat(Enum):
    """Common GPU resource formats."""

    R8_UNORM = auto()
    R8G8B8A8_UNORM = auto()
    R8G8B8A8_SRGB = auto()
    R11G11B10_FLOAT = auto()
    R16G16B16A16_FLOAT = auto()
    R32G32B32A32_FLOAT = auto()
    R32_FLOAT = auto()
    R32G32_FLOAT = auto()
    D24_UNORM_S8_UINT = auto()
    D32_FLOAT = auto()
    D32_FLOAT_S8_UINT = auto()
    BC1_UNORM = auto()
    BC3_UNORM = auto()
    BC5_UNORM = auto()
    BC6H_FLOAT = auto()
    BC7_UNORM = auto()


class ResourceState(Enum):
    """GPU resource states for barrier management.

    These states determine how a resource can be accessed and what
    barriers are needed for state transitions.
    """

    UNDEFINED = auto()
    """Initial state, contents undefined."""

    RENDER_TARGET = auto()
    """Written as a render target (color attachment)."""

    DEPTH_WRITE = auto()
    """Written as depth/stencil attachment."""

    DEPTH_READ = auto()
    """Read-only depth/stencil."""

    SHADER_RESOURCE = auto()
    """Read in a shader (SRV)."""

    UNORDERED_ACCESS = auto()
    """Read/write in compute (UAV)."""

    COPY_SOURCE = auto()
    """Source of a copy operation."""

    COPY_DEST = auto()
    """Destination of a copy operation."""

    PRESENT = auto()
    """Ready for presentation."""

    INDIRECT_ARGUMENT = auto()
    """Used as indirect draw/dispatch arguments."""

    VERTEX_BUFFER = auto()
    """Bound as vertex buffer."""

    INDEX_BUFFER = auto()
    """Bound as index buffer."""

    CONSTANT_BUFFER = auto()
    """Bound as constant/uniform buffer."""

    ACCELERATION_STRUCTURE = auto()
    """Ray tracing acceleration structure."""


@dataclass
class ResourceDescriptor:
    """Describes a frame graph resource's properties.

    This descriptor is used to declare what kind of resource is needed
    by a pass, allowing the resource manager to allocate or alias appropriately.
    """

    name: str
    """Unique name for this resource within the frame graph."""

    resource_type: ResourceType = ResourceType.TRANSIENT
    """Classification determining lifetime and aliasing behavior."""

    format: ResourceFormat = ResourceFormat.R8G8B8A8_UNORM
    """Pixel/data format of the resource."""

    width: int = 0
    """Width in pixels (0 = derive from render target)."""

    height: int = 0
    """Height in pixels (0 = derive from render target)."""

    depth: int = 1
    """Depth for 3D textures, array layers for 2D arrays."""

    mip_levels: int = 1
    """Number of mipmap levels."""

    sample_count: int = 1
    """MSAA sample count (1 = no MSAA)."""

    is_texture: bool = True
    """True for textures, False for buffers."""

    buffer_size: int = 0
    """Size in bytes for buffer resources."""

    clear_value: Optional[tuple] = None
    """Optional clear value (color or depth/stencil)."""


@dataclass
class ResourceHandle:
    """Handle for referencing a frame graph resource.

    ResourceHandles are lightweight references that allow passes to
    declare dependencies without directly accessing the underlying
    GPU resource. The actual resource is resolved during frame graph
    compilation and execution.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    """Unique identifier for this handle."""

    descriptor: Optional[ResourceDescriptor] = None
    """The resource descriptor, if this handle owns the resource."""

    version: int = 0
    """Version number for tracking write operations."""

    _producer_pass: Optional[str] = None
    """Name of the pass that produces/writes this resource."""

    @property
    def name(self) -> str:
        """Get the resource name from the descriptor."""
        if self.descriptor:
            return self.descriptor.name
        return f"unnamed_{self.id[:8]}"

    @property
    def resource_type(self) -> ResourceType:
        """Get the resource type from the descriptor."""
        if self.descriptor:
            return self.descriptor.resource_type
        return ResourceType.TRANSIENT

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResourceHandle):
            return False
        return self.id == other.id


@dataclass
class TransientResource:
    """A per-frame resource that can be aliased across passes.

    Transient resources are allocated from a memory pool and can share
    the same underlying memory with other transient resources that don't
    have overlapping lifetimes. This is a key optimization for reducing
    GPU memory usage.

    Per RENDERING_CONTEXT.md Section 6.1:
    "Transient resources - allocated per-frame, aliased across passes"
    """

    handle: ResourceHandle
    """The handle for this resource."""

    first_use_pass: int = -1
    """Index of the first pass that uses this resource."""

    last_use_pass: int = -1
    """Index of the last pass that uses this resource."""

    alias_group: int = -1
    """Group ID for resources that can share memory (-1 = no aliasing)."""

    allocated_offset: int = 0
    """Offset within the alias group's memory allocation."""

    size_bytes: int = 0
    """Size of this resource in bytes."""

    current_state: ResourceState = ResourceState.UNDEFINED
    """Current GPU state of the resource."""

    def overlaps_with(self, other: TransientResource) -> bool:
        """Check if this resource's lifetime overlaps with another.

        Resources with overlapping lifetimes cannot be aliased to the
        same memory.
        """
        if self.first_use_pass == -1 or other.first_use_pass == -1:
            return False
        return not (
            self.last_use_pass < other.first_use_pass or
            other.last_use_pass < self.first_use_pass
        )


@dataclass
class HistoryResource:
    """A resource persisted across frames.

    History resources maintain their contents between frames, which is
    essential for temporal effects like:
    - TAA (Temporal Anti-Aliasing) history buffers
    - GI accumulators (DDGI, Lumen radiance cache)
    - Motion vectors from previous frame
    - Reprojection buffers

    Per RENDERING_CONTEXT.md Section 6.1:
    "History resources - persisted across frames (TAA history, GI accumulators)"
    """

    handle: ResourceHandle
    """The handle for this resource."""

    frame_count: int = 0
    """Number of frames this resource has existed."""

    double_buffered: bool = True
    """If True, maintains two copies for read-while-write."""

    current_index: int = 0
    """Current buffer index for double-buffered resources."""

    current_state: ResourceState = ResourceState.UNDEFINED
    """Current GPU state of the resource."""

    _gpu_resources: list = field(default_factory=list)
    """Underlying GPU resource(s) - platform-specific."""

    def swap_buffers(self) -> None:
        """Swap the current buffer index for double-buffered resources."""
        if self.double_buffered:
            self.current_index = 1 - self.current_index
            self.frame_count += 1


@dataclass
class ExternalResource:
    """A resource imported from outside the frame graph.

    External resources are not managed by the frame graph's memory
    allocator. They include:
    - Swap chain backbuffer
    - Imported textures from asset system
    - Resources shared with other systems

    Per RENDERING_CONTEXT.md Section 6.1:
    "External resources - swap chain backbuffer, imported textures"
    """

    handle: ResourceHandle
    """The handle for this resource."""

    gpu_resource: Any = None
    """The actual GPU resource (platform-specific)."""

    current_state: ResourceState = ResourceState.UNDEFINED
    """Current GPU state of the resource."""

    is_backbuffer: bool = False
    """True if this is the swap chain backbuffer."""

    read_only: bool = False
    """True if this resource should not be written to."""


class ResourceManager:
    """Manages resource allocation and aliasing for the frame graph.

    The ResourceManager is responsible for:
    1. Creating resource handles for passes to reference
    2. Tracking resource lifetimes during compilation
    3. Performing memory aliasing for transient resources
    4. Managing history resource double-buffering
    5. Registering external resources

    This implements the resource management described in
    RENDERING_CONTEXT.md Section 6.1.
    """

    def __init__(self) -> None:
        """Initialize the resource manager."""
        self._transients: dict[str, TransientResource] = {}
        self._history: dict[str, HistoryResource] = {}
        self._externals: dict[str, ExternalResource] = {}
        self._handles: dict[str, ResourceHandle] = {}
        self._alias_groups: dict[int, list[TransientResource]] = {}
        self._next_alias_group: int = 0

    def create_transient(
        self,
        name: str,
        format: ResourceFormat = ResourceFormat.R8G8B8A8_UNORM,
        width: int = 0,
        height: int = 0,
        depth: int = 1,
        mip_levels: int = 1,
        sample_count: int = 1,
        clear_value: Optional[tuple] = None,
    ) -> ResourceHandle:
        """Create a transient resource that can be aliased.

        Args:
            name: Unique name for this resource.
            format: Pixel format.
            width: Width in pixels (0 = derive from render target).
            height: Height in pixels (0 = derive from render target).
            depth: Depth for 3D textures.
            mip_levels: Number of mipmap levels.
            sample_count: MSAA sample count.
            clear_value: Optional clear value.

        Returns:
            A ResourceHandle for referencing this resource.

        Raises:
            ValueError: If a resource with this name already exists.
        """
        if name in self._handles:
            raise ValueError(f"Resource '{name}' already exists")

        descriptor = ResourceDescriptor(
            name=name,
            resource_type=ResourceType.TRANSIENT,
            format=format,
            width=width,
            height=height,
            depth=depth,
            mip_levels=mip_levels,
            sample_count=sample_count,
            clear_value=clear_value,
        )

        handle = ResourceHandle(descriptor=descriptor)
        self._handles[name] = handle

        transient = TransientResource(handle=handle)
        self._transients[name] = transient

        return handle

    def create_history(
        self,
        name: str,
        format: ResourceFormat = ResourceFormat.R8G8B8A8_UNORM,
        width: int = 0,
        height: int = 0,
        double_buffered: bool = True,
    ) -> ResourceHandle:
        """Create a history resource persisted across frames.

        Args:
            name: Unique name for this resource.
            format: Pixel format.
            width: Width in pixels.
            height: Height in pixels.
            double_buffered: Whether to maintain two copies.

        Returns:
            A ResourceHandle for referencing this resource.

        Raises:
            ValueError: If a resource with this name already exists.
        """
        if name in self._handles:
            raise ValueError(f"Resource '{name}' already exists")

        descriptor = ResourceDescriptor(
            name=name,
            resource_type=ResourceType.HISTORY,
            format=format,
            width=width,
            height=height,
        )

        handle = ResourceHandle(descriptor=descriptor)
        self._handles[name] = handle

        history = HistoryResource(
            handle=handle,
            double_buffered=double_buffered,
        )
        self._history[name] = history

        return handle

    def register_external(
        self,
        name: str,
        gpu_resource: Any,
        format: ResourceFormat = ResourceFormat.R8G8B8A8_UNORM,
        width: int = 0,
        height: int = 0,
        is_backbuffer: bool = False,
        read_only: bool = False,
    ) -> ResourceHandle:
        """Register an external resource (backbuffer, imported texture).

        Args:
            name: Unique name for this resource.
            gpu_resource: The actual GPU resource (platform-specific).
            format: Pixel format.
            width: Width in pixels.
            height: Height in pixels.
            is_backbuffer: True if this is the swap chain backbuffer.
            read_only: True if this resource should not be written to.

        Returns:
            A ResourceHandle for referencing this resource.

        Raises:
            ValueError: If a resource with this name already exists.
        """
        if name in self._handles:
            raise ValueError(f"Resource '{name}' already exists")

        descriptor = ResourceDescriptor(
            name=name,
            resource_type=ResourceType.EXTERNAL,
            format=format,
            width=width,
            height=height,
        )

        handle = ResourceHandle(descriptor=descriptor)
        self._handles[name] = handle

        external = ExternalResource(
            handle=handle,
            gpu_resource=gpu_resource,
            is_backbuffer=is_backbuffer,
            read_only=read_only,
        )
        self._externals[name] = external

        return handle

    def create_buffer(
        self,
        name: str,
        size_bytes: int,
        resource_type: ResourceType = ResourceType.TRANSIENT,
    ) -> ResourceHandle:
        """Create a buffer resource.

        Args:
            name: Unique name for this resource.
            size_bytes: Size of the buffer in bytes.
            resource_type: Type of resource (TRANSIENT or HISTORY).

        Returns:
            A ResourceHandle for referencing this resource.

        Raises:
            ValueError: If a resource with this name already exists or
                       if size_bytes is not positive.
        """
        if name in self._handles:
            raise ValueError(f"Resource '{name}' already exists")

        if size_bytes <= 0:
            raise ValueError(f"Buffer size must be positive, got {size_bytes}")

        descriptor = ResourceDescriptor(
            name=name,
            resource_type=resource_type,
            is_texture=False,
            buffer_size=size_bytes,
        )

        handle = ResourceHandle(descriptor=descriptor)
        self._handles[name] = handle

        if resource_type == ResourceType.TRANSIENT:
            transient = TransientResource(handle=handle, size_bytes=size_bytes)
            self._transients[name] = transient
        elif resource_type == ResourceType.HISTORY:
            history = HistoryResource(handle=handle)
            self._history[name] = history

        return handle

    def get_handle(self, name: str) -> Optional[ResourceHandle]:
        """Get a resource handle by name.

        Args:
            name: The resource name.

        Returns:
            The ResourceHandle, or None if not found.
        """
        return self._handles.get(name)

    def update_lifetime(
        self,
        handle: ResourceHandle,
        pass_index: int,
    ) -> None:
        """Update the lifetime of a transient resource.

        Called during compilation to track when resources are used.

        Args:
            handle: The resource handle.
            pass_index: The index of the pass using this resource.
        """
        name = handle.name
        if name in self._transients:
            transient = self._transients[name]
            if transient.first_use_pass == -1:
                transient.first_use_pass = pass_index
            transient.last_use_pass = max(transient.last_use_pass, pass_index)

    def compute_aliasing(self) -> None:
        """Compute memory aliasing for transient resources.

        This algorithm groups non-overlapping transient resources so they
        can share the same underlying memory allocation. This is a key
        optimization for reducing GPU memory usage.

        The algorithm:
        1. Sort transients by first_use_pass
        2. For each transient, try to find an existing alias group
        3. If no compatible group, create a new one
        """
        self._alias_groups.clear()
        self._next_alias_group = 0

        # Sort transients by first use
        sorted_transients = sorted(
            self._transients.values(),
            key=lambda t: (t.first_use_pass, t.last_use_pass),
        )

        for transient in sorted_transients:
            if transient.first_use_pass == -1:
                # Resource never used, skip
                continue

            # Try to find a compatible alias group
            found_group = False
            for group_id, group_members in self._alias_groups.items():
                can_alias = True
                for member in group_members:
                    if transient.overlaps_with(member):
                        can_alias = False
                        break

                if can_alias:
                    transient.alias_group = group_id
                    group_members.append(transient)
                    found_group = True
                    break

            if not found_group:
                # Create new alias group
                group_id = self._next_alias_group
                self._next_alias_group += 1
                transient.alias_group = group_id
                self._alias_groups[group_id] = [transient]

    def get_transient(self, name: str) -> Optional[TransientResource]:
        """Get a transient resource by name."""
        return self._transients.get(name)

    def get_history(self, name: str) -> Optional[HistoryResource]:
        """Get a history resource by name."""
        return self._history.get(name)

    def get_external(self, name: str) -> Optional[ExternalResource]:
        """Get an external resource by name."""
        return self._externals.get(name)

    def get_alias_group(self, group_id: int) -> list[TransientResource]:
        """Get all resources in an alias group."""
        return self._alias_groups.get(group_id, [])

    def get_alias_group_count(self) -> int:
        """Get the number of alias groups."""
        return len(self._alias_groups)

    def begin_frame(self) -> None:
        """Called at the start of each frame.

        Resets transient resources and swaps history buffers.
        """
        # Reset transient resource states
        for transient in self._transients.values():
            transient.current_state = ResourceState.UNDEFINED
            transient.first_use_pass = -1
            transient.last_use_pass = -1
            transient.alias_group = -1

        # Swap history buffers
        for history in self._history.values():
            history.swap_buffers()

    def clear(self) -> None:
        """Clear all resources."""
        self._transients.clear()
        self._history.clear()
        self._externals.clear()
        self._handles.clear()
        self._alias_groups.clear()
        self._next_alias_group = 0
