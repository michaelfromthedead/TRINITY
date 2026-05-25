"""
Mock rendering context for testing the Frame Graph.

Provides a mock implementation of the RHIContext protocol that records
all method calls for test assertions and defaults to no-op behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from engine.rendering.framegraph import (
    AllocationHandle,
    Barrier,
    FenceOp,
    PassNode,
    QueueType,
    ResourceDescriptor,
)


@dataclass
class MockContext:
    """Mock rendering context satisfying the RHIContext protocol.

    Records all method calls for test assertions. All methods default
    to no-op behavior (no exceptions).

    Usage:
        ctx = MockContext()

        # Default no-op
        ctx.execute_barriers([])

        # Assert after frame graph execution
        assert ctx.barrier_count > 0
        assert ctx.pass_begin_count == expected

    Helpers:
        assert_barriers_executed()  - checks barriers were called
        assert_queue_submitted()    - checks queue submission
        reset()                     - clears recorded calls
    """

    # Recorded calls
    barrier_calls: list[list[Barrier]] = field(default_factory=list)
    """Record of each execute_barriers call with its barriers."""

    alloc_calls: list[ResourceDescriptor] = field(default_factory=list)
    """Record of each allocate_transient call with its descriptor."""

    pass_begin_calls: list[PassNode] = field(default_factory=list)
    """Record of each begin_pass call with its pass node."""

    pass_end_calls: list[PassNode] = field(default_factory=list)
    """Record of each end_pass call with its pass node."""

    submit_calls: list[tuple[QueueType, list[FenceOp]]] = field(default_factory=list)
    """Record of each submit_queue call."""

    # Helper counters (computed from recorded calls)
    @property
    def barrier_count(self) -> int:
        """Total number of individual barriers across all calls."""
        return sum(len(batch) for batch in self.barrier_calls)

    @property
    def barrier_batch_count(self) -> int:
        """Number of execute_barriers invocations."""
        return len(self.barrier_calls)

    @property
    def pass_begin_count(self) -> int:
        """Number of begin_pass invocations."""
        return len(self.pass_begin_calls)

    @property
    def pass_end_count(self) -> int:
        """Number of end_pass invocations."""
        return len(self.pass_end_calls)

    @property
    def submit_count(self) -> int:
        """Number of submit_queue invocations."""
        return len(self.submit_calls)

    # RHIContext protocol implementation

    def execute_barriers(self, barriers: Sequence[Barrier]) -> None:
        """Record barrier execution.

        Args:
            barriers: The barriers that were executed.
        """
        self.barrier_calls.append(list(barriers))

    def allocate_transient(self, desc: ResourceDescriptor) -> AllocationHandle:
        """Record transient allocation and return a dummy handle.

        Args:
            desc: Resource descriptor.

        Returns:
            A dummy AllocationHandle (value 1).
        """
        self.alloc_calls.append(desc)
        return AllocationHandle(1)

    def begin_pass(self, pass_node: PassNode) -> None:
        """Record pass begin.

        Args:
            pass_node: The pass being started.
        """
        self.pass_begin_calls.append(pass_node)

    def end_pass(self, pass_node: PassNode) -> None:
        """Record pass end.

        Args:
            pass_node: The pass being ended.
        """
        self.pass_end_calls.append(pass_node)

    def submit_queue(
        self,
        queue: QueueType,
        fences: Sequence[FenceOp],
    ) -> None:
        """Record queue submission.

        Args:
            queue: The target queue.
            fences: Fence operations.
        """
        self.submit_calls.append((queue, list(fences)))

    # Helper assertions

    def assert_barriers_executed(self, min_count: int = 1) -> None:
        """Assert that at least min_count barriers were executed.

        Args:
            min_count: Minimum number of barriers expected.

        Raises:
            AssertionError: If fewer barriers were recorded.
        """
        assert self.barrier_count >= min_count, (
            f"Expected at least {min_count} barriers, "
            f"got {self.barrier_count}"
        )

    def assert_queue_submitted(
        self,
        queue: QueueType | None = None,
        min_count: int = 1,
    ) -> None:
        """Assert that queue submissions were recorded.

        Args:
            queue: Optional filter for specific queue type.
            min_count: Minimum number of submissions expected.

        Raises:
            AssertionError: If fewer submissions were recorded.
        """
        if queue is not None:
            matching = sum(
                1 for q, _ in self.submit_calls if q == queue
            )
            assert matching >= min_count, (
                f"Expected at least {min_count} submissions on "
                f"{queue}, got {matching}"
            )
        else:
            assert self.submit_count >= min_count, (
                f"Expected at least {min_count} submissions, "
                f"got {self.submit_count}"
            )

    def assert_barrier_for_resource(self, resource_name: str) -> None:
        """Assert that a barrier was executed for a specific resource.

        Args:
            resource_name: Name of the resource to check.

        Raises:
            AssertionError: If no barrier references the resource.
        """
        for batch in self.barrier_calls:
            for barrier in batch:
                if barrier.handle.name == resource_name:
                    return
        raise AssertionError(
            f"No barrier found for resource '{resource_name}'"
        )

    def assert_pass_sequence(self, expected_names: list[str]) -> None:
        """Assert that passes began in the expected order.

        Args:
            expected_names: Expected pass names in order.

        Raises:
            AssertionError: If pass sequence doesn't match.
        """
        actual = [p.name for p in self.pass_begin_calls]
        assert actual == expected_names, (
            f"Expected pass sequence {expected_names}, got {actual}"
        )

    def reset(self) -> None:
        """Reset all recorded calls and counters."""
        self.barrier_calls.clear()
        self.alloc_calls.clear()
        self.pass_begin_calls.clear()
        self.pass_end_calls.clear()
        self.submit_calls.clear()
