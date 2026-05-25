"""
Pass node definitions for the Frame Graph.

This module implements the different pass types that can be added to a frame graph,
as specified in RENDERING_CONTEXT.md Section 6.1.

Pass Types (from spec):
- Graphics Pass (rasterization)
- Compute Pass (dispatch)
- Copy Pass (transfer)
- Ray Tracing Pass

Each pass declares its read/write resource dependencies, enabling the frame graph
to build a dependency graph, compute execution order, and insert barriers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

from .resource_manager import ResourceHandle, ResourceState


class PassType(Enum):
    """Types of render passes in the frame graph."""

    GRAPHICS = auto()
    """Rasterization pass (draw calls, render targets)."""

    COMPUTE = auto()
    """Compute dispatch pass."""

    COPY = auto()
    """Transfer/copy operation."""

    RAY_TRACING = auto()
    """Ray tracing pass."""


class PassFlags(Enum):
    """Optional flags for pass behavior."""

    NONE = 0
    """No special flags."""

    ASYNC_COMPUTE = 1 << 0
    """Can run on async compute queue."""

    NO_CULL = 1 << 1
    """Never cull this pass, even if outputs are unused."""

    SIDE_EFFECTS = 1 << 2
    """Has side effects (e.g., writes to swap chain)."""


@dataclass
class ResourceAccess:
    """Describes how a pass accesses a resource."""

    handle: ResourceHandle
    """The resource being accessed."""

    state: ResourceState
    """The state the resource needs to be in."""

    is_write: bool = False
    """True if this is a write access."""

    subresource: Optional[int] = None
    """Optional subresource index (mip level, array slice)."""


@dataclass
class ColorAttachment:
    """Configuration for a color render target attachment."""

    handle: ResourceHandle
    """The render target resource."""

    load_op: str = "clear"
    """Load operation: 'clear', 'load', or 'dont_care'."""

    store_op: str = "store"
    """Store operation: 'store' or 'dont_care'."""

    clear_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Clear color if load_op is 'clear'."""

    resolve_target: Optional[ResourceHandle] = None
    """Optional MSAA resolve target."""


@dataclass
class DepthStencilAttachment:
    """Configuration for a depth/stencil render target attachment."""

    handle: ResourceHandle
    """The depth/stencil resource."""

    depth_load_op: str = "clear"
    """Depth load operation: 'clear', 'load', or 'dont_care'."""

    depth_store_op: str = "store"
    """Depth store operation: 'store' or 'dont_care'."""

    stencil_load_op: str = "dont_care"
    """Stencil load operation: 'clear', 'load', or 'dont_care'."""

    stencil_store_op: str = "dont_care"
    """Stencil store operation: 'store' or 'dont_care'."""

    clear_depth: float = 1.0
    """Clear depth value if depth_load_op is 'clear'."""

    clear_stencil: int = 0
    """Clear stencil value if stencil_load_op is 'clear'."""

    read_only: bool = False
    """If True, depth/stencil is read-only (depth testing without writing)."""


@dataclass
class PassNode(ABC):
    """Base class for all frame graph pass types.

    A PassNode represents a single unit of GPU work. Passes declare their
    resource dependencies (reads and writes), which allows the frame graph
    to determine execution order and insert necessary barriers.

    Per RENDERING_CONTEXT.md Section 6.1:
    "Declare passes -> Build dependency graph -> Cull unused passes
     -> Schedule async compute -> Insert barriers -> Execute"
    """

    name: str
    """Unique name for this pass within the frame graph."""

    pass_type: PassType = PassType.GRAPHICS
    """The type of GPU work this pass performs."""

    enabled: bool = True
    """If False, this pass is skipped during execution."""

    flags: int = PassFlags.NONE.value
    """Combination of PassFlags."""

    reads: list[ResourceAccess] = field(default_factory=list)
    """Resources this pass reads from."""

    writes: list[ResourceAccess] = field(default_factory=list)
    """Resources this pass writes to."""

    _execute_callback: Optional[Callable] = None
    """The callback to execute this pass's GPU work."""

    _culled: bool = False
    """True if this pass was culled during compilation."""

    _execution_index: int = -1
    """Execution order index assigned during compilation."""

    def read(
        self,
        handle: ResourceHandle,
        state: ResourceState = ResourceState.SHADER_RESOURCE,
        subresource: Optional[int] = None,
    ) -> PassNode:
        """Declare a resource read dependency.

        Args:
            handle: The resource to read.
            state: The state needed for reading.
            subresource: Optional subresource index.

        Returns:
            Self for method chaining.
        """
        self.reads.append(ResourceAccess(
            handle=handle,
            state=state,
            is_write=False,
            subresource=subresource,
        ))
        return self

    def write(
        self,
        handle: ResourceHandle,
        state: ResourceState = ResourceState.RENDER_TARGET,
        subresource: Optional[int] = None,
    ) -> PassNode:
        """Declare a resource write dependency.

        Args:
            handle: The resource to write.
            state: The state needed for writing.
            subresource: Optional subresource index.

        Returns:
            Self for method chaining.

        Note:
            Writing to a resource increments its version number.
        """
        self.writes.append(ResourceAccess(
            handle=handle,
            state=state,
            is_write=True,
            subresource=subresource,
        ))
        handle.version += 1
        handle._producer_pass = self.name
        return self

    def set_execute(self, callback: Callable) -> PassNode:
        """Set the execution callback for this pass.

        Args:
            callback: Function to call when executing this pass.

        Returns:
            Self for method chaining.
        """
        self._execute_callback = callback
        return self

    def has_flag(self, flag: PassFlags) -> bool:
        """Check if this pass has a specific flag."""
        return bool(self.flags & flag.value)

    def set_flag(self, flag: PassFlags) -> PassNode:
        """Set a flag on this pass.

        Args:
            flag: The flag to set.

        Returns:
            Self for method chaining.
        """
        self.flags |= flag.value
        return self

    def clear_flag(self, flag: PassFlags) -> PassNode:
        """Clear a flag from this pass.

        Args:
            flag: The flag to clear.

        Returns:
            Self for method chaining.
        """
        self.flags &= ~flag.value
        return self

    def get_read_handles(self) -> list[ResourceHandle]:
        """Get all resource handles this pass reads."""
        return [access.handle for access in self.reads]

    def get_write_handles(self) -> list[ResourceHandle]:
        """Get all resource handles this pass writes."""
        return [access.handle for access in self.writes]

    def get_all_handles(self) -> list[ResourceHandle]:
        """Get all resource handles this pass uses."""
        return self.get_read_handles() + self.get_write_handles()

    @abstractmethod
    def execute(self, context: Any) -> None:
        """Execute this pass's GPU work.

        Args:
            context: Platform-specific rendering context.
        """
        pass


@dataclass
class GraphicsPass(PassNode):
    """A rasterization pass (draw calls, render targets).

    Graphics passes render geometry to one or more render targets using
    the graphics pipeline. They can have color and depth/stencil attachments.

    Per RENDERING_CONTEXT.md Frame Graph Pass Types:
    "Graphics Pass (rasterization)"
    """

    pass_type: PassType = field(default=PassType.GRAPHICS, init=False)

    color_attachments: list[ColorAttachment] = field(default_factory=list)
    """Color render target attachments."""

    depth_stencil: Optional[DepthStencilAttachment] = None
    """Optional depth/stencil attachment."""

    viewport: Optional[tuple[int, int, int, int]] = None
    """Optional viewport (x, y, width, height)."""

    scissor: Optional[tuple[int, int, int, int]] = None
    """Optional scissor rect (x, y, width, height)."""

    def add_color_attachment(
        self,
        handle: ResourceHandle,
        load_op: str = "clear",
        store_op: str = "store",
        clear_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        resolve_target: Optional[ResourceHandle] = None,
    ) -> GraphicsPass:
        """Add a color render target attachment.

        Args:
            handle: The render target resource.
            load_op: Load operation ('clear', 'load', 'dont_care').
            store_op: Store operation ('store', 'dont_care').
            clear_color: Clear color if load_op is 'clear'.
            resolve_target: Optional MSAA resolve target.

        Returns:
            Self for method chaining.
        """
        attachment = ColorAttachment(
            handle=handle,
            load_op=load_op,
            store_op=store_op,
            clear_color=clear_color,
            resolve_target=resolve_target,
        )
        self.color_attachments.append(attachment)

        # Declare the write dependency
        self.write(handle, ResourceState.RENDER_TARGET)

        if resolve_target:
            self.write(resolve_target, ResourceState.RENDER_TARGET)

        return self

    def set_depth_stencil(
        self,
        handle: ResourceHandle,
        depth_load_op: str = "clear",
        depth_store_op: str = "store",
        clear_depth: float = 1.0,
        read_only: bool = False,
    ) -> GraphicsPass:
        """Set the depth/stencil attachment.

        Args:
            handle: The depth/stencil resource.
            depth_load_op: Depth load operation.
            depth_store_op: Depth store operation.
            clear_depth: Clear depth value.
            read_only: If True, depth is read-only.

        Returns:
            Self for method chaining.
        """
        self.depth_stencil = DepthStencilAttachment(
            handle=handle,
            depth_load_op=depth_load_op,
            depth_store_op=depth_store_op,
            clear_depth=clear_depth,
            read_only=read_only,
        )

        # Declare the dependency
        if read_only:
            self.read(handle, ResourceState.DEPTH_READ)
        else:
            self.write(handle, ResourceState.DEPTH_WRITE)

        return self

    def execute(self, context: Any) -> None:
        """Execute this graphics pass.

        Args:
            context: Platform-specific rendering context.
        """
        if self._execute_callback:
            self._execute_callback(context)


@dataclass
class ComputePass(PassNode):
    """A compute dispatch pass.

    Compute passes dispatch compute shaders to process data on the GPU.
    They can read from and write to textures and buffers.

    Per RENDERING_CONTEXT.md Frame Graph Pass Types:
    "Compute Pass (dispatch)"
    """

    pass_type: PassType = field(default=PassType.COMPUTE, init=False)

    dispatch_size: tuple[int, int, int] = (1, 1, 1)
    """Dispatch size (groups_x, groups_y, groups_z)."""

    indirect_buffer: Optional[ResourceHandle] = None
    """Optional indirect dispatch buffer."""

    def set_dispatch_size(
        self,
        groups_x: int,
        groups_y: int,
        groups_z: int = 1,
    ) -> ComputePass:
        """Set the dispatch size.

        Args:
            groups_x: Number of work groups in X.
            groups_y: Number of work groups in Y.
            groups_z: Number of work groups in Z.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If any dimension is not positive.
        """
        if groups_x <= 0 or groups_y <= 0 or groups_z <= 0:
            raise ValueError(
                f"Dispatch dimensions must be positive, got ({groups_x}, {groups_y}, {groups_z})"
            )
        self.dispatch_size = (groups_x, groups_y, groups_z)
        return self

    def set_indirect(self, buffer: ResourceHandle) -> ComputePass:
        """Set indirect dispatch buffer.

        Args:
            buffer: Buffer containing dispatch arguments.

        Returns:
            Self for method chaining.
        """
        self.indirect_buffer = buffer
        self.read(buffer, ResourceState.INDIRECT_ARGUMENT)
        return self

    def read_texture(
        self,
        handle: ResourceHandle,
        subresource: Optional[int] = None,
    ) -> ComputePass:
        """Declare a texture read (SRV).

        Args:
            handle: The texture to read.
            subresource: Optional mip level.

        Returns:
            Self for method chaining.
        """
        self.read(handle, ResourceState.SHADER_RESOURCE, subresource)
        return self

    def write_texture(
        self,
        handle: ResourceHandle,
        subresource: Optional[int] = None,
    ) -> ComputePass:
        """Declare a texture write (UAV).

        Args:
            handle: The texture to write.
            subresource: Optional mip level.

        Returns:
            Self for method chaining.
        """
        self.write(handle, ResourceState.UNORDERED_ACCESS, subresource)
        return self

    def read_buffer(self, handle: ResourceHandle) -> ComputePass:
        """Declare a buffer read (SRV).

        Args:
            handle: The buffer to read.

        Returns:
            Self for method chaining.
        """
        self.read(handle, ResourceState.SHADER_RESOURCE)
        return self

    def write_buffer(self, handle: ResourceHandle) -> ComputePass:
        """Declare a buffer write (UAV).

        Args:
            handle: The buffer to write.

        Returns:
            Self for method chaining.
        """
        self.write(handle, ResourceState.UNORDERED_ACCESS)
        return self

    def execute(self, context: Any) -> None:
        """Execute this compute pass.

        Args:
            context: Platform-specific rendering context.
        """
        if self._execute_callback:
            self._execute_callback(context)


@dataclass
class CopyPass(PassNode):
    """A transfer/copy operation pass.

    Copy passes perform data transfers between resources, such as:
    - Texture to texture copies
    - Buffer to buffer copies
    - Buffer to texture uploads
    - Texture to buffer readbacks

    Per RENDERING_CONTEXT.md Frame Graph Pass Types:
    "Copy Pass (transfer)"
    """

    pass_type: PassType = field(default=PassType.COPY, init=False)

    source: Optional[ResourceHandle] = None
    """Source resource for the copy."""

    destination: Optional[ResourceHandle] = None
    """Destination resource for the copy."""

    source_region: Optional[tuple] = None
    """Optional source region (x, y, z, width, height, depth)."""

    dest_offset: Optional[tuple] = None
    """Optional destination offset (x, y, z)."""

    def set_copy(
        self,
        source: ResourceHandle,
        destination: ResourceHandle,
        source_region: Optional[tuple] = None,
        dest_offset: Optional[tuple] = None,
    ) -> CopyPass:
        """Set the copy operation parameters.

        Args:
            source: Source resource.
            destination: Destination resource.
            source_region: Optional source region.
            dest_offset: Optional destination offset.

        Returns:
            Self for method chaining.
        """
        self.source = source
        self.destination = destination
        self.source_region = source_region
        self.dest_offset = dest_offset

        self.read(source, ResourceState.COPY_SOURCE)
        self.write(destination, ResourceState.COPY_DEST)

        return self

    def execute(self, context: Any) -> None:
        """Execute this copy pass.

        Args:
            context: Platform-specific rendering context.
        """
        if self._execute_callback:
            self._execute_callback(context)


@dataclass
class RayTracingPass(PassNode):
    """A ray tracing pass.

    Ray tracing passes dispatch ray tracing shaders that traverse
    acceleration structures to trace rays against scene geometry.

    Per RENDERING_CONTEXT.md Frame Graph Pass Types:
    "Ray Tracing Pass"

    Also see Section 6.11 for ray tracing architecture details.
    """

    pass_type: PassType = field(default=PassType.RAY_TRACING, init=False)

    dispatch_width: int = 1
    """Width of the ray dispatch (typically screen width)."""

    dispatch_height: int = 1
    """Height of the ray dispatch (typically screen height)."""

    dispatch_depth: int = 1
    """Depth of the ray dispatch."""

    tlas: Optional[ResourceHandle] = None
    """Top-Level Acceleration Structure (scene BVH)."""

    shader_binding_table: Optional[ResourceHandle] = None
    """Shader Binding Table for hit/miss shaders."""

    max_recursion_depth: int = 1
    """Maximum ray recursion depth."""

    def set_dispatch_dimensions(
        self,
        width: int,
        height: int,
        depth: int = 1,
    ) -> RayTracingPass:
        """Set the ray dispatch dimensions.

        Args:
            width: Dispatch width.
            height: Dispatch height.
            depth: Dispatch depth.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If any dimension is not positive.
        """
        if width <= 0 or height <= 0 or depth <= 0:
            raise ValueError(
                f"Ray dispatch dimensions must be positive, got ({width}, {height}, {depth})"
            )
        self.dispatch_width = width
        self.dispatch_height = height
        self.dispatch_depth = depth
        return self

    def set_acceleration_structure(
        self,
        tlas: ResourceHandle,
    ) -> RayTracingPass:
        """Set the top-level acceleration structure.

        Args:
            tlas: TLAS resource handle.

        Returns:
            Self for method chaining.
        """
        self.tlas = tlas
        self.read(tlas, ResourceState.ACCELERATION_STRUCTURE)
        return self

    def set_shader_binding_table(
        self,
        sbt: ResourceHandle,
    ) -> RayTracingPass:
        """Set the shader binding table.

        Args:
            sbt: SBT resource handle.

        Returns:
            Self for method chaining.
        """
        self.shader_binding_table = sbt
        self.read(sbt, ResourceState.SHADER_RESOURCE)
        return self

    def write_output(
        self,
        handle: ResourceHandle,
    ) -> RayTracingPass:
        """Declare the ray tracing output texture.

        Args:
            handle: Output texture handle.

        Returns:
            Self for method chaining.
        """
        self.write(handle, ResourceState.UNORDERED_ACCESS)
        return self

    def execute(self, context: Any) -> None:
        """Execute this ray tracing pass.

        Args:
            context: Platform-specific rendering context.
        """
        if self._execute_callback:
            self._execute_callback(context)


def create_pass(
    name: str,
    pass_type: PassType,
    **kwargs,
) -> PassNode:
    """Factory function to create a pass of the specified type.

    Args:
        name: The pass name.
        pass_type: The type of pass to create.
        **kwargs: Additional arguments for the pass constructor.

    Returns:
        A new PassNode of the specified type.

    Raises:
        ValueError: If the pass type is unknown.
    """
    pass_classes = {
        PassType.GRAPHICS: GraphicsPass,
        PassType.COMPUTE: ComputePass,
        PassType.COPY: CopyPass,
        PassType.RAY_TRACING: RayTracingPass,
    }

    if pass_type not in pass_classes:
        raise ValueError(f"Unknown pass type: {pass_type}")

    return pass_classes[pass_type](name=name, **kwargs)
