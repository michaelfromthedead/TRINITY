"""
RHIContext protocol definition for rendering context abstraction.

This module defines the RHIContext Protocol that all rendering contexts
must satisfy. It provides the abstract interface between the frame graph
and the underlying rendering hardware interface (RHI).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType, Protocol, Sequence, runtime_checkable

from .async_scheduler import QueueType
from .barrier_manager import Barrier
from .pass_node import PassNode
from .resource_manager import ResourceDescriptor


AllocationHandle = NewType("AllocationHandle", int)
"""Opaque handle for a transient resource allocation."""


@dataclass
class FenceOp:
    """A fence operation for GPU queue synchronization.

    Fence operations synchronize work between different GPU queues
    (graphics, compute, copy).
    """

    operation: str = "signal"
    """Type of operation: 'signal' or 'wait'."""

    fence_value: int = 0
    """The fence value to signal or wait for."""

    target_queue: QueueType = QueueType.GRAPHICS
    """The queue this fence operation targets."""


@runtime_checkable
class RHIContext(Protocol):
    """Protocol for rendering contexts used by the frame graph.

    Any rendering backend (Vulkan, DirectX 12, Metal, WebGPU) must
    satisfy this protocol to be used with the frame graph execution
    pipeline.

    The protocol ensures the minimal interface needed by the frame graph:
    - Barrier execution for resource state transitions
    - Transient resource allocation
    - Pass lifecycle (begin/end)
    - Queue submission with fences
    """

    def execute_barriers(self, barriers: Sequence[Barrier]) -> None:
        """Execute resource state transition barriers.

        Args:
            barriers: Sequence of barriers to execute.
        """
        ...

    def allocate_transient(self, desc: ResourceDescriptor) -> AllocationHandle:
        """Allocate a transient resource.

        Args:
            desc: Description of the resource to allocate.

        Returns:
            An opaque handle for the allocated resource.
        """
        ...

    def begin_pass(self, pass_node: PassNode) -> None:
        """Begin a render pass.

        Args:
            pass_node: The pass to begin.
        """
        ...

    def end_pass(self, pass_node: PassNode) -> None:
        """End a render pass.

        Args:
            pass_node: The pass to end.
        """
        ...

    def submit_queue(self, queue: QueueType, fences: Sequence[FenceOp]) -> None:
        """Submit work on a queue with fence synchronization.

        Args:
            queue: The queue to submit work on.
            fences: Fence operations for synchronization.
        """
        ...
