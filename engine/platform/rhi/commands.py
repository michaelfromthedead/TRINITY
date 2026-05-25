"""RHI Command recording and submission."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Any
import threading


@dataclass
class Command:
    """Recorded command."""
    type: str
    args: dict


class CommandList(ABC):
    """Abstract command list for recording GPU commands."""

    @abstractmethod
    def begin(self) -> None:
        """Begin command recording."""
        pass

    @abstractmethod
    def end(self) -> None:
        """End command recording."""
        pass

    @abstractmethod
    def barrier(self, resource: Any, state_before: 'ResourceState', state_after: 'ResourceState') -> None:
        """Insert resource barrier."""
        pass

    @abstractmethod
    def begin_render_pass(
        self,
        render_targets: List['Texture'],
        depth_target: Optional['Texture'] = None,
        clear_color: Optional[tuple] = None,
        clear_depth: Optional[float] = None
    ) -> None:
        """Begin render pass."""
        pass

    @abstractmethod
    def end_render_pass(self) -> None:
        """End render pass."""
        pass

    @abstractmethod
    def set_pipeline(self, pipeline: 'PipelineState') -> None:
        """Set pipeline state."""
        pass

    @abstractmethod
    def set_viewport(self, x: float, y: float, w: float, h: float, min_depth: float, max_depth: float) -> None:
        """Set viewport."""
        pass

    @abstractmethod
    def set_scissor(self, x: int, y: int, w: int, h: int) -> None:
        """Set scissor rectangle."""
        pass

    @abstractmethod
    def set_vertex_buffer(self, slot: int, buffer: 'Buffer', offset: int, stride: int) -> None:
        """Set vertex buffer."""
        pass

    @abstractmethod
    def set_index_buffer(self, buffer: 'Buffer', offset: int, format: 'Format') -> None:
        """Set index buffer."""
        pass

    @abstractmethod
    def draw(self, vertex_count: int, instance_count: int, first_vertex: int, first_instance: int) -> None:
        """Draw primitives."""
        pass

    @abstractmethod
    def draw_indexed(
        self,
        index_count: int,
        instance_count: int,
        first_index: int,
        vertex_offset: int,
        first_instance: int
    ) -> None:
        """Draw indexed primitives."""
        pass

    @abstractmethod
    def dispatch(self, x: int, y: int, z: int) -> None:
        """Dispatch compute."""
        pass

    @abstractmethod
    def copy_buffer(self, dst: 'Buffer', dst_offset: int, src: 'Buffer', src_offset: int, size: int) -> None:
        """Copy buffer data."""
        pass

    @property
    @abstractmethod
    def recorded_commands(self) -> List[Command]:
        """Get recorded commands for testing/validation."""
        pass


class Queue(ABC):
    """Abstract command queue."""

    @abstractmethod
    def submit(self, command_lists: List[CommandList], signal_fence: Optional['Fence'] = None) -> None:
        """Submit command lists for execution."""
        pass

    @abstractmethod
    def wait(self, fence: 'Fence') -> None:
        """Wait for fence."""
        pass

    @abstractmethod
    def signal(self, fence: 'Fence') -> None:
        """Signal fence."""
        pass


class NullCommandList(CommandList):
    """Null implementation of CommandList."""

    def __init__(self):
        self._commands: List[Command] = []
        self._recording = False

    def begin(self) -> None:
        """Begin command recording."""
        self._recording = True
        self._commands.clear()

    def end(self) -> None:
        """End command recording."""
        self._recording = False

    def barrier(self, resource: Any, state_before: 'ResourceState', state_after: 'ResourceState') -> None:
        """Insert resource barrier."""
        if self._recording:
            self._commands.append(Command(
                type="barrier",
                args={"resource": resource, "state_before": state_before, "state_after": state_after}
            ))

    def begin_render_pass(
        self,
        render_targets: List['Texture'],
        depth_target: Optional['Texture'] = None,
        clear_color: Optional[tuple] = None,
        clear_depth: Optional[float] = None
    ) -> None:
        """Begin render pass."""
        if self._recording:
            self._commands.append(Command(
                type="begin_render_pass",
                args={
                    "render_targets": render_targets,
                    "depth_target": depth_target,
                    "clear_color": clear_color,
                    "clear_depth": clear_depth
                }
            ))

    def end_render_pass(self) -> None:
        """End render pass."""
        if self._recording:
            self._commands.append(Command(type="end_render_pass", args={}))

    def set_pipeline(self, pipeline: 'PipelineState') -> None:
        """Set pipeline state."""
        if self._recording:
            self._commands.append(Command(type="set_pipeline", args={"pipeline": pipeline}))

    def set_viewport(self, x: float, y: float, w: float, h: float, min_depth: float, max_depth: float) -> None:
        """Set viewport."""
        if self._recording:
            self._commands.append(Command(
                type="set_viewport",
                args={"x": x, "y": y, "w": w, "h": h, "min_depth": min_depth, "max_depth": max_depth}
            ))

    def set_scissor(self, x: int, y: int, w: int, h: int) -> None:
        """Set scissor rectangle."""
        if self._recording:
            self._commands.append(Command(type="set_scissor", args={"x": x, "y": y, "w": w, "h": h}))

    def set_vertex_buffer(self, slot: int, buffer: 'Buffer', offset: int, stride: int) -> None:
        """Set vertex buffer."""
        if self._recording:
            self._commands.append(Command(
                type="set_vertex_buffer",
                args={"slot": slot, "buffer": buffer, "offset": offset, "stride": stride}
            ))

    def set_index_buffer(self, buffer: 'Buffer', offset: int, format: 'Format') -> None:
        """Set index buffer."""
        if self._recording:
            self._commands.append(Command(
                type="set_index_buffer",
                args={"buffer": buffer, "offset": offset, "format": format}
            ))

    def draw(self, vertex_count: int, instance_count: int, first_vertex: int, first_instance: int) -> None:
        """Draw primitives."""
        if self._recording:
            self._commands.append(Command(
                type="draw",
                args={
                    "vertex_count": vertex_count,
                    "instance_count": instance_count,
                    "first_vertex": first_vertex,
                    "first_instance": first_instance
                }
            ))

    def draw_indexed(
        self,
        index_count: int,
        instance_count: int,
        first_index: int,
        vertex_offset: int,
        first_instance: int
    ) -> None:
        """Draw indexed primitives."""
        if self._recording:
            self._commands.append(Command(
                type="draw_indexed",
                args={
                    "index_count": index_count,
                    "instance_count": instance_count,
                    "first_index": first_index,
                    "vertex_offset": vertex_offset,
                    "first_instance": first_instance
                }
            ))

    def dispatch(self, x: int, y: int, z: int) -> None:
        """Dispatch compute."""
        if self._recording:
            self._commands.append(Command(type="dispatch", args={"x": x, "y": y, "z": z}))

    def copy_buffer(self, dst: 'Buffer', dst_offset: int, src: 'Buffer', src_offset: int, size: int) -> None:
        """Copy buffer data."""
        if self._recording:
            self._commands.append(Command(
                type="copy_buffer",
                args={"dst": dst, "dst_offset": dst_offset, "src": src, "src_offset": src_offset, "size": size}
            ))

    @property
    def recorded_commands(self) -> List[Command]:
        """Get recorded commands for testing/validation."""
        return self._commands.copy()


class NullQueue(Queue):
    """Null implementation of Queue."""

    def __init__(self, queue_type: 'QueueType'):
        self._queue_type = queue_type
        self._submitted_commands: List[List[Command]] = []
        self._shutdown = False

    def submit(self, command_lists: List[CommandList], signal_fence: Optional['Fence'] = None) -> None:
        """Submit command lists for execution."""
        for cmd_list in command_lists:
            self._submitted_commands.append(cmd_list.recorded_commands)

        # Signal fence if provided using proper API
        if signal_fence:
            signal_fence.signal(signal_fence.value + 1)

    def wait(self, fence: 'Fence') -> None:
        """Wait for fence."""
        # Null implementation - nothing to wait for
        pass

    def signal(self, fence: 'Fence') -> None:
        """Signal fence using proper API."""
        fence.signal(fence.value + 1)

    def shutdown(self) -> None:
        """Shutdown the queue."""
        self._shutdown = True


# Forward declarations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .resources import Buffer, Texture, Format
    from .pipeline import PipelineState
    from .sync import Fence, ResourceState
    from .device import QueueType
