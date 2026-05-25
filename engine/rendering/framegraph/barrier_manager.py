"""
Automatic barrier insertion for the Frame Graph.

This module implements automatic resource barrier insertion between passes,
as specified in RENDERING_CONTEXT.md Section 6.1.

Barrier Management (from spec):
"Automatic barrier insertion between passes"

The barrier manager analyzes resource state transitions between passes
and inserts appropriate barriers (e.g., RENDER_TARGET -> SHADER_RESOURCE).

Integration Pattern from Section 11:
"Pass A writes texture as RTV (render target)
 Pass B reads same texture as SRV (shader resource)
 Frame graph automatically inserts:
   Barrier(texture, RENDER_TARGET -> SHADER_RESOURCE) between A and B"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

from .pass_node import PassNode, ResourceAccess
from .resource_manager import (
    ResourceHandle,
    ResourceManager,
    ResourceState,
    ResourceType,
)


class BarrierType(Enum):
    """Types of GPU resource barriers."""

    TRANSITION = auto()
    """Resource state transition (e.g., RTV -> SRV)."""

    UAV = auto()
    """UAV barrier for read-after-write hazards."""

    ALIASING = auto()
    """Aliasing barrier when memory is reused."""


class PipelineStage(Enum):
    """GPU pipeline stages for synchronization."""

    TOP_OF_PIPE = auto()
    """Start of the pipeline."""

    VERTEX_INPUT = auto()
    """Vertex input assembly."""

    VERTEX_SHADER = auto()
    """Vertex shader stage."""

    FRAGMENT_SHADER = auto()
    """Fragment/pixel shader stage."""

    EARLY_FRAGMENT_TESTS = auto()
    """Early depth/stencil tests."""

    LATE_FRAGMENT_TESTS = auto()
    """Late depth/stencil tests."""

    COLOR_ATTACHMENT_OUTPUT = auto()
    """Color attachment output."""

    COMPUTE_SHADER = auto()
    """Compute shader stage."""

    TRANSFER = auto()
    """Transfer/copy operations."""

    RAY_TRACING_SHADER = auto()
    """Ray tracing shaders."""

    ACCELERATION_STRUCTURE_BUILD = auto()
    """Acceleration structure build."""

    BOTTOM_OF_PIPE = auto()
    """End of the pipeline."""

    ALL_GRAPHICS = auto()
    """All graphics pipeline stages."""

    ALL_COMMANDS = auto()
    """All pipeline stages."""


class AccessFlags(Enum):
    """Memory access flags for barriers."""

    NONE = 0

    VERTEX_ATTRIBUTE_READ = 1 << 0
    INDEX_READ = 1 << 1
    UNIFORM_READ = 1 << 2
    INPUT_ATTACHMENT_READ = 1 << 3
    SHADER_READ = 1 << 4
    SHADER_WRITE = 1 << 5
    COLOR_ATTACHMENT_READ = 1 << 6
    COLOR_ATTACHMENT_WRITE = 1 << 7
    DEPTH_STENCIL_READ = 1 << 8
    DEPTH_STENCIL_WRITE = 1 << 9
    TRANSFER_READ = 1 << 10
    TRANSFER_WRITE = 1 << 11
    HOST_READ = 1 << 12
    HOST_WRITE = 1 << 13
    MEMORY_READ = 1 << 14
    MEMORY_WRITE = 1 << 15
    ACCELERATION_STRUCTURE_READ = 1 << 16
    ACCELERATION_STRUCTURE_WRITE = 1 << 17


@dataclass
class Barrier:
    """Describes a resource barrier to be inserted.

    Barriers ensure proper synchronization between GPU operations and
    resource state transitions.
    """

    handle: ResourceHandle
    """The resource this barrier is for."""

    barrier_type: BarrierType
    """The type of barrier."""

    old_state: ResourceState
    """The resource state before the barrier."""

    new_state: ResourceState
    """The resource state after the barrier."""

    src_stage: PipelineStage = PipelineStage.ALL_COMMANDS
    """Source pipeline stage (wait for this stage)."""

    dst_stage: PipelineStage = PipelineStage.ALL_COMMANDS
    """Destination pipeline stage (block until barrier done)."""

    src_access: int = 0
    """Source access flags (AccessFlags combination)."""

    dst_access: int = 0
    """Destination access flags (AccessFlags combination)."""

    subresource: Optional[int] = None
    """Optional subresource index (None = all subresources)."""

    def __repr__(self) -> str:
        return (
            f"Barrier({self.handle.name}: "
            f"{self.old_state.name} -> {self.new_state.name})"
        )


@dataclass
class BarrierBatch:
    """A batch of barriers to execute together.

    Batching barriers is more efficient than executing them individually.
    """

    barriers: list[Barrier] = field(default_factory=list)
    """The barriers in this batch."""

    before_pass: Optional[str] = None
    """Name of the pass this batch is executed before."""

    def add(self, barrier: Barrier) -> None:
        """Add a barrier to this batch."""
        self.barriers.append(barrier)

    def is_empty(self) -> bool:
        """Check if this batch has no barriers."""
        return len(self.barriers) == 0


# Mapping from resource state to optimal pipeline stages
_STATE_TO_STAGE: dict[ResourceState, tuple[PipelineStage, int]] = {
    ResourceState.UNDEFINED: (
        PipelineStage.TOP_OF_PIPE,
        AccessFlags.NONE.value,
    ),
    ResourceState.RENDER_TARGET: (
        PipelineStage.COLOR_ATTACHMENT_OUTPUT,
        AccessFlags.COLOR_ATTACHMENT_WRITE.value,
    ),
    ResourceState.DEPTH_WRITE: (
        PipelineStage.LATE_FRAGMENT_TESTS,
        AccessFlags.DEPTH_STENCIL_WRITE.value,
    ),
    ResourceState.DEPTH_READ: (
        PipelineStage.EARLY_FRAGMENT_TESTS,
        AccessFlags.DEPTH_STENCIL_READ.value,
    ),
    ResourceState.SHADER_RESOURCE: (
        PipelineStage.FRAGMENT_SHADER,
        AccessFlags.SHADER_READ.value,
    ),
    ResourceState.UNORDERED_ACCESS: (
        PipelineStage.COMPUTE_SHADER,
        AccessFlags.SHADER_READ.value | AccessFlags.SHADER_WRITE.value,
    ),
    ResourceState.COPY_SOURCE: (
        PipelineStage.TRANSFER,
        AccessFlags.TRANSFER_READ.value,
    ),
    ResourceState.COPY_DEST: (
        PipelineStage.TRANSFER,
        AccessFlags.TRANSFER_WRITE.value,
    ),
    ResourceState.PRESENT: (
        PipelineStage.BOTTOM_OF_PIPE,
        AccessFlags.NONE.value,
    ),
    ResourceState.INDIRECT_ARGUMENT: (
        PipelineStage.VERTEX_INPUT,
        AccessFlags.SHADER_READ.value,
    ),
    ResourceState.VERTEX_BUFFER: (
        PipelineStage.VERTEX_INPUT,
        AccessFlags.VERTEX_ATTRIBUTE_READ.value,
    ),
    ResourceState.INDEX_BUFFER: (
        PipelineStage.VERTEX_INPUT,
        AccessFlags.INDEX_READ.value,
    ),
    ResourceState.CONSTANT_BUFFER: (
        PipelineStage.VERTEX_SHADER,
        AccessFlags.UNIFORM_READ.value,
    ),
    ResourceState.ACCELERATION_STRUCTURE: (
        PipelineStage.RAY_TRACING_SHADER,
        AccessFlags.ACCELERATION_STRUCTURE_READ.value,
    ),
}


def _get_stage_and_access(
    state: ResourceState,
) -> tuple[PipelineStage, int]:
    """Get the optimal pipeline stage and access flags for a resource state."""
    return _STATE_TO_STAGE.get(
        state,
        (PipelineStage.ALL_COMMANDS, AccessFlags.MEMORY_READ.value),
    )


def _needs_barrier(
    old_state: ResourceState,
    new_state: ResourceState,
) -> bool:
    """Determine if a barrier is needed between two states.

    Some transitions don't need explicit barriers:
    - Same state to same state (no change)
    - Undefined to any state (no previous data to protect)
    """
    if old_state == new_state:
        return False

    if old_state == ResourceState.UNDEFINED:
        # Transitioning from undefined doesn't need synchronization
        # but may need layout transition
        return True

    return True


@dataclass
class ResourceStateTracker:
    """Tracks the current state of each resource.

    Used by the barrier manager to determine what transitions are needed.
    """

    _states: dict[str, ResourceState] = field(default_factory=dict)
    """Map of resource name to current state."""

    _subresource_states: dict[str, dict[int, ResourceState]] = field(
        default_factory=dict
    )
    """Map of resource name to subresource states."""

    def get_state(
        self,
        handle: ResourceHandle,
        subresource: Optional[int] = None,
    ) -> ResourceState:
        """Get the current state of a resource.

        Args:
            handle: The resource handle.
            subresource: Optional subresource index.

        Returns:
            The current resource state.
        """
        name = handle.name

        if subresource is not None:
            sub_states = self._subresource_states.get(name, {})
            if subresource in sub_states:
                return sub_states[subresource]

        return self._states.get(name, ResourceState.UNDEFINED)

    def set_state(
        self,
        handle: ResourceHandle,
        state: ResourceState,
        subresource: Optional[int] = None,
    ) -> None:
        """Set the current state of a resource.

        Args:
            handle: The resource handle.
            state: The new state.
            subresource: Optional subresource index.
        """
        name = handle.name

        if subresource is not None:
            if name not in self._subresource_states:
                self._subresource_states[name] = {}
            self._subresource_states[name][subresource] = state
        else:
            self._states[name] = state
            # Clear subresource states when setting whole-resource state
            if name in self._subresource_states:
                del self._subresource_states[name]

    def clear(self) -> None:
        """Clear all tracked states."""
        self._states.clear()
        self._subresource_states.clear()


class BarrierManager:
    """Manages automatic barrier insertion for the frame graph.

    The BarrierManager analyzes resource state transitions between passes
    and generates the necessary barriers to ensure correct GPU synchronization.

    Per RENDERING_CONTEXT.md Section 6.1:
    "Automatic barrier insertion between passes"

    And Section 11 Integration Pattern:
    "Frame graph automatically inserts:
     Barrier(texture, RENDER_TARGET -> SHADER_RESOURCE) between A and B"
    """

    def __init__(self, resource_manager: ResourceManager) -> None:
        """Initialize the barrier manager.

        Args:
            resource_manager: The frame graph's resource manager.
        """
        self._resource_manager = resource_manager
        self._state_tracker = ResourceStateTracker()
        self._barrier_batches: list[BarrierBatch] = []

    def analyze_passes(self, passes: list[PassNode]) -> list[BarrierBatch]:
        """Analyze passes and generate barrier batches.

        This is the main entry point for barrier analysis. It walks through
        all passes in execution order and generates barriers for any
        resource state transitions.

        Args:
            passes: List of passes in execution order.

        Returns:
            List of barrier batches, one per pass.
        """
        self._state_tracker.clear()
        self._barrier_batches.clear()

        # Initialize external resource states
        for name, external in self._resource_manager._externals.items():
            self._state_tracker.set_state(
                external.handle,
                external.current_state,
            )

        # Analyze each pass
        for pass_node in passes:
            if pass_node._culled:
                continue

            batch = self._analyze_pass(pass_node)
            self._barrier_batches.append(batch)

        return self._barrier_batches

    def _analyze_pass(self, pass_node: PassNode) -> BarrierBatch:
        """Analyze a single pass and generate its barrier batch.

        Args:
            pass_node: The pass to analyze.

        Returns:
            A barrier batch for this pass.
        """
        batch = BarrierBatch(before_pass=pass_node.name)

        # Process reads first (they need resources in read state)
        for access in pass_node.reads:
            barrier = self._create_transition_barrier(
                access.handle,
                access.state,
                access.subresource,
            )
            if barrier:
                batch.add(barrier)

        # Process writes (may need UAV barriers or transitions)
        for access in pass_node.writes:
            barrier = self._create_transition_barrier(
                access.handle,
                access.state,
                access.subresource,
            )
            if barrier:
                batch.add(barrier)

        # Check for UAV barriers (read-after-write on same resource)
        uav_barriers = self._check_uav_hazards(pass_node)
        for barrier in uav_barriers:
            batch.add(barrier)

        return batch

    def _create_transition_barrier(
        self,
        handle: ResourceHandle,
        required_state: ResourceState,
        subresource: Optional[int] = None,
    ) -> Optional[Barrier]:
        """Create a transition barrier if needed.

        Args:
            handle: The resource handle.
            required_state: The state needed by the pass.
            subresource: Optional subresource index.

        Returns:
            A Barrier if transition is needed, None otherwise.
        """
        current_state = self._state_tracker.get_state(handle, subresource)

        if not _needs_barrier(current_state, required_state):
            return None

        # Get pipeline stages and access flags
        src_stage, src_access = _get_stage_and_access(current_state)
        dst_stage, dst_access = _get_stage_and_access(required_state)

        barrier = Barrier(
            handle=handle,
            barrier_type=BarrierType.TRANSITION,
            old_state=current_state,
            new_state=required_state,
            src_stage=src_stage,
            dst_stage=dst_stage,
            src_access=src_access,
            dst_access=dst_access,
            subresource=subresource,
        )

        # Update tracked state
        self._state_tracker.set_state(handle, required_state, subresource)

        return barrier

    def _check_uav_hazards(self, pass_node: PassNode) -> list[Barrier]:
        """Check for UAV read-after-write hazards.

        When a resource is written as UAV and then read as UAV in
        a subsequent pass, a UAV barrier is needed.

        Args:
            pass_node: The pass to check.

        Returns:
            List of UAV barriers needed.
        """
        barriers: list[Barrier] = []

        for access in pass_node.reads:
            if access.state == ResourceState.UNORDERED_ACCESS:
                current = self._state_tracker.get_state(
                    access.handle,
                    access.subresource,
                )
                if current == ResourceState.UNORDERED_ACCESS:
                    # UAV-to-UAV requires explicit barrier
                    barrier = Barrier(
                        handle=access.handle,
                        barrier_type=BarrierType.UAV,
                        old_state=current,
                        new_state=access.state,
                        src_stage=PipelineStage.COMPUTE_SHADER,
                        dst_stage=PipelineStage.COMPUTE_SHADER,
                        src_access=AccessFlags.SHADER_WRITE.value,
                        dst_access=AccessFlags.SHADER_READ.value,
                        subresource=access.subresource,
                    )
                    barriers.append(barrier)

        return barriers

    def create_aliasing_barrier(
        self,
        old_resource: ResourceHandle,
        new_resource: ResourceHandle,
    ) -> Barrier:
        """Create an aliasing barrier when memory is reused.

        Aliasing barriers are needed when transient resources that share
        the same memory are used in sequence.

        Args:
            old_resource: The resource being "deallocated".
            new_resource: The resource being "allocated".

        Returns:
            An aliasing barrier.
        """
        return Barrier(
            handle=new_resource,
            barrier_type=BarrierType.ALIASING,
            old_state=ResourceState.UNDEFINED,
            new_state=ResourceState.UNDEFINED,
            src_stage=PipelineStage.ALL_COMMANDS,
            dst_stage=PipelineStage.ALL_COMMANDS,
            src_access=AccessFlags.MEMORY_WRITE.value,
            dst_access=AccessFlags.MEMORY_READ.value | AccessFlags.MEMORY_WRITE.value,
        )

    def get_final_states(self) -> dict[str, ResourceState]:
        """Get the final states of all resources after all passes.

        Returns:
            Dictionary mapping resource name to final state.
        """
        return dict(self._state_tracker._states)

    def prepare_for_present(
        self,
        backbuffer: ResourceHandle,
    ) -> Optional[Barrier]:
        """Create a barrier to prepare backbuffer for presentation.

        Args:
            backbuffer: The backbuffer resource handle.

        Returns:
            A Barrier if transition is needed, None otherwise.
        """
        return self._create_transition_barrier(
            backbuffer,
            ResourceState.PRESENT,
        )

    def reset(self) -> None:
        """Reset the barrier manager for a new frame."""
        self._state_tracker.clear()
        self._barrier_batches.clear()
