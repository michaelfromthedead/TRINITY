"""
Async compute scheduling for the Frame Graph.

This module implements async compute scheduling to enable parallel execution
of compute work alongside graphics, as specified in RENDERING_CONTEXT.md Section 6.1.

Async Compute Scheduling (from spec):
"Schedule async compute"

The scheduler identifies passes that can run on the async compute queue
and schedules them to execute in parallel with the graphics pipeline,
maximizing GPU utilization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from .config import ASYNC_SCHEDULER_CONFIG
from .pass_node import PassFlags, PassNode, PassType
from .resource_manager import ResourceHandle


class QueueType(Enum):
    """GPU command queue types."""

    GRAPHICS = auto()
    """Main graphics queue (supports all operations)."""

    COMPUTE = auto()
    """Async compute queue (compute only)."""

    COPY = auto()
    """Transfer/copy queue."""


@dataclass
class SyncPoint:
    """A synchronization point between queues.

    Sync points are used to coordinate work between different GPU queues,
    ensuring proper ordering of dependent operations.
    """

    name: str
    """Name of this sync point."""

    signal_queue: QueueType
    """Queue that signals completion."""

    wait_queue: QueueType
    """Queue that waits for completion."""

    signal_pass: str
    """Name of the pass that signals."""

    wait_pass: str
    """Name of the pass that waits."""

    fence_value: int = 0
    """Fence value for GPU synchronization."""


@dataclass
class ScheduledPass:
    """A pass with scheduling information assigned."""

    pass_node: PassNode
    """The underlying pass node."""

    queue: QueueType = QueueType.GRAPHICS
    """The queue this pass executes on."""

    sync_before: list[SyncPoint] = field(default_factory=list)
    """Sync points to wait for before execution."""

    sync_after: list[SyncPoint] = field(default_factory=list)
    """Sync points to signal after execution."""

    parallel_group: int = -1
    """Group ID for passes that can execute in parallel (-1 = sequential)."""


@dataclass
class QueueTimeline:
    """Timeline of work scheduled on a specific queue."""

    queue_type: QueueType
    """The queue type."""

    passes: list[ScheduledPass] = field(default_factory=list)
    """Passes scheduled on this queue."""

    current_fence_value: int = 0
    """Current fence value for this queue."""


class AsyncScheduler:
    """Schedules async compute passes for parallel execution.

    The AsyncScheduler analyzes the frame graph's pass dependencies and
    identifies opportunities to run compute passes on the async compute
    queue in parallel with graphics work.

    Per RENDERING_CONTEXT.md Section 6.1:
    "Declare passes -> Build dependency graph -> Cull unused passes
     -> Schedule async compute -> Insert barriers -> Execute"

    Benefits of async compute:
    - Better GPU utilization by overlapping compute and graphics
    - Hide latency of compute passes behind graphics work
    - Useful for effects like SSAO, DOF, bloom that can run independently
    """

    def __init__(self) -> None:
        """Initialize the async scheduler."""
        self._graphics_timeline = QueueTimeline(queue_type=QueueType.GRAPHICS)
        self._compute_timeline = QueueTimeline(queue_type=QueueType.COMPUTE)
        self._copy_timeline = QueueTimeline(queue_type=QueueType.COPY)
        self._sync_points: list[SyncPoint] = []
        self._scheduled_passes: list[ScheduledPass] = []
        self._next_fence_value: int = 1

    def schedule(
        self,
        passes: list[PassNode],
        enable_async_compute: bool = True,
    ) -> list[ScheduledPass]:
        """Schedule passes across GPU queues.

        Args:
            passes: List of passes in dependency order.
            enable_async_compute: Whether to use async compute.

        Returns:
            List of scheduled passes with queue assignments.
        """
        self._reset()

        for pass_node in passes:
            if pass_node._culled:
                continue

            scheduled = self._schedule_pass(pass_node, enable_async_compute)
            self._scheduled_passes.append(scheduled)

        # Add cross-queue synchronization
        self._compute_sync_points()

        return self._scheduled_passes

    def _reset(self) -> None:
        """Reset the scheduler for a new frame."""
        self._graphics_timeline = QueueTimeline(queue_type=QueueType.GRAPHICS)
        self._compute_timeline = QueueTimeline(queue_type=QueueType.COMPUTE)
        self._copy_timeline = QueueTimeline(queue_type=QueueType.COPY)
        self._sync_points.clear()
        self._scheduled_passes.clear()
        self._next_fence_value = 1

    def _schedule_pass(
        self,
        pass_node: PassNode,
        enable_async: bool,
    ) -> ScheduledPass:
        """Schedule a single pass.

        Args:
            pass_node: The pass to schedule.
            enable_async: Whether async compute is enabled.

        Returns:
            A scheduled pass with queue assignment.
        """
        scheduled = ScheduledPass(pass_node=pass_node)

        # Determine the appropriate queue
        queue = self._determine_queue(pass_node, enable_async)
        scheduled.queue = queue

        # Add to appropriate timeline
        if queue == QueueType.GRAPHICS:
            self._graphics_timeline.passes.append(scheduled)
        elif queue == QueueType.COMPUTE:
            self._compute_timeline.passes.append(scheduled)
        else:
            self._copy_timeline.passes.append(scheduled)

        return scheduled

    def _determine_queue(
        self,
        pass_node: PassNode,
        enable_async: bool,
    ) -> QueueType:
        """Determine which queue a pass should run on.

        Args:
            pass_node: The pass to analyze.
            enable_async: Whether async compute is enabled.

        Returns:
            The appropriate queue type.
        """
        # Copy passes go to copy queue
        if pass_node.pass_type == PassType.COPY:
            return QueueType.COPY

        # If async not enabled, everything goes to graphics
        if not enable_async:
            return QueueType.GRAPHICS

        # Graphics and ray tracing must use graphics queue
        if pass_node.pass_type in (PassType.GRAPHICS, PassType.RAY_TRACING):
            return QueueType.GRAPHICS

        # Compute passes can potentially run async
        if pass_node.pass_type == PassType.COMPUTE:
            # Check if explicitly marked for async
            if pass_node.has_flag(PassFlags.ASYNC_COMPUTE):
                return QueueType.COMPUTE

            # Check if it's safe to run async (no graphics dependencies)
            if self._can_run_async(pass_node):
                return QueueType.COMPUTE

        return QueueType.GRAPHICS

    def _can_run_async(self, pass_node: PassNode) -> bool:
        """Determine if a compute pass can safely run on async queue.

        A pass can run async if:
        - It's a compute pass
        - It doesn't have immediate dependencies on graphics output
        - It doesn't write to resources immediately needed by graphics

        Args:
            pass_node: The pass to analyze.

        Returns:
            True if the pass can run on async compute.
        """
        if pass_node.pass_type != PassType.COMPUTE:
            return False

        # Check read dependencies
        for access in pass_node.reads:
            # If reading a resource written by recent graphics pass,
            # need synchronization
            if self._has_recent_graphics_write(access.handle):
                return False

        # Check write dependencies
        for access in pass_node.writes:
            # If writing a resource read by pending graphics pass,
            # need synchronization
            if self._has_pending_graphics_read(access.handle):
                return False

        return True

    def _has_recent_graphics_write(self, handle: ResourceHandle) -> bool:
        """Check if a resource was recently written by graphics queue.

        Args:
            handle: The resource handle.

        Returns:
            True if there's a recent graphics write.
        """
        # Look at last few graphics passes (configurable window)
        window_size = ASYNC_SCHEDULER_CONFIG.recent_graphics_write_window
        recent_passes = self._graphics_timeline.passes[-window_size:]
        for scheduled in reversed(recent_passes):
            write_handles = scheduled.pass_node.get_write_handles()
            if handle in write_handles:
                return True
        return False

    def _has_pending_graphics_read(self, handle: ResourceHandle) -> bool:
        """Check if a resource will be read by upcoming graphics pass.

        This is a heuristic - in practice we'd need the full dependency graph.

        Args:
            handle: The resource handle.

        Returns:
            True if there's a pending graphics read.
        """
        # This would need lookahead in a full implementation
        return False

    def _compute_sync_points(self) -> None:
        """Compute synchronization points between queues.

        Adds sync points where:
        - Compute writes data that graphics will read
        - Graphics writes data that compute will read
        """
        # Track resources written by each queue
        graphics_writes: dict[str, ScheduledPass] = {}
        compute_writes: dict[str, ScheduledPass] = {}

        # Build write maps
        for scheduled in self._graphics_timeline.passes:
            for handle in scheduled.pass_node.get_write_handles():
                graphics_writes[handle.name] = scheduled

        for scheduled in self._compute_timeline.passes:
            for handle in scheduled.pass_node.get_write_handles():
                compute_writes[handle.name] = scheduled

        # Find compute -> graphics dependencies
        for scheduled in self._graphics_timeline.passes:
            for handle in scheduled.pass_node.get_read_handles():
                if handle.name in compute_writes:
                    writer = compute_writes[handle.name]
                    sync = self._create_sync_point(
                        writer,
                        scheduled,
                        QueueType.COMPUTE,
                        QueueType.GRAPHICS,
                    )
                    self._sync_points.append(sync)
                    writer.sync_after.append(sync)
                    scheduled.sync_before.append(sync)

        # Find graphics -> compute dependencies
        for scheduled in self._compute_timeline.passes:
            for handle in scheduled.pass_node.get_read_handles():
                if handle.name in graphics_writes:
                    writer = graphics_writes[handle.name]
                    sync = self._create_sync_point(
                        writer,
                        scheduled,
                        QueueType.GRAPHICS,
                        QueueType.COMPUTE,
                    )
                    self._sync_points.append(sync)
                    writer.sync_after.append(sync)
                    scheduled.sync_before.append(sync)

    def _create_sync_point(
        self,
        signal_pass: ScheduledPass,
        wait_pass: ScheduledPass,
        signal_queue: QueueType,
        wait_queue: QueueType,
    ) -> SyncPoint:
        """Create a synchronization point.

        Args:
            signal_pass: The pass that signals.
            wait_pass: The pass that waits.
            signal_queue: The signaling queue.
            wait_queue: The waiting queue.

        Returns:
            A new SyncPoint.
        """
        fence_value = self._next_fence_value
        self._next_fence_value += 1

        return SyncPoint(
            name=f"sync_{signal_pass.pass_node.name}_to_{wait_pass.pass_node.name}",
            signal_queue=signal_queue,
            wait_queue=wait_queue,
            signal_pass=signal_pass.pass_node.name,
            wait_pass=wait_pass.pass_node.name,
            fence_value=fence_value,
        )

    def get_graphics_passes(self) -> list[ScheduledPass]:
        """Get passes scheduled on the graphics queue."""
        return self._graphics_timeline.passes

    def get_compute_passes(self) -> list[ScheduledPass]:
        """Get passes scheduled on the compute queue."""
        return self._compute_timeline.passes

    def get_copy_passes(self) -> list[ScheduledPass]:
        """Get passes scheduled on the copy queue."""
        return self._copy_timeline.passes

    def get_sync_points(self) -> list[SyncPoint]:
        """Get all synchronization points."""
        return self._sync_points

    def get_parallel_groups(self) -> dict[int, list[ScheduledPass]]:
        """Get passes grouped by parallel execution opportunity.

        Returns:
            Dictionary mapping group ID to list of passes that can
            execute in parallel.
        """
        groups: dict[int, list[ScheduledPass]] = {}
        group_id = 0

        # Group consecutive async compute passes
        current_group: list[ScheduledPass] = []

        for scheduled in self._scheduled_passes:
            if scheduled.queue == QueueType.COMPUTE:
                current_group.append(scheduled)
                scheduled.parallel_group = group_id
            else:
                if current_group:
                    groups[group_id] = current_group
                    current_group = []
                    group_id += 1

        if current_group:
            groups[group_id] = current_group

        return groups

    def estimate_overlap_benefit(self) -> float:
        """Estimate the benefit of async compute overlap.

        Returns:
            Estimated percentage improvement (0.0 - 1.0).
        """
        total_passes = len(self._scheduled_passes)
        if total_passes == 0:
            return 0.0

        async_passes = len(self._compute_timeline.passes)
        if async_passes == 0:
            return 0.0

        # Simple heuristic: benefit based on ratio of async work
        # In practice, this would need timing data
        overlap_ratio = async_passes / total_passes
        multiplier = ASYNC_SCHEDULER_CONFIG.overlap_benefit_multiplier
        max_benefit = ASYNC_SCHEDULER_CONFIG.max_overlap_benefit
        return min(overlap_ratio * multiplier, max_benefit)


def identify_async_candidates(passes: list[PassNode]) -> list[PassNode]:
    """Identify passes that are good candidates for async compute.

    Good candidates are:
    - Compute passes with high workload
    - Passes with limited dependencies
    - Passes that don't immediately feed graphics output

    Args:
        passes: List of passes to analyze.

    Returns:
        List of passes suitable for async compute.
    """
    candidates: list[PassNode] = []

    for pass_node in passes:
        if pass_node.pass_type != PassType.COMPUTE:
            continue

        # Already marked for async
        if pass_node.has_flag(PassFlags.ASYNC_COMPUTE):
            candidates.append(pass_node)
            continue

        # Heuristics for async suitability
        # 1. Fewer reads = more independent
        # 2. Not writing to render targets
        max_reads = ASYNC_SCHEDULER_CONFIG.async_candidate_max_reads
        if len(pass_node.reads) <= max_reads:
            is_render_target_writer = any(
                access.handle.name in ("backbuffer", "color_target", "hdr_target")
                for access in pass_node.writes
            )
            if not is_render_target_writer:
                candidates.append(pass_node)

    return candidates
