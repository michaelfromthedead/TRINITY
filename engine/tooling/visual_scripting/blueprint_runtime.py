"""
FlowForge Blueprint Runtime - Blueprint execution engine with VM and JIT compilation.

Provides the execution engine for blueprints:
- Virtual machine for interpreting blueprints
- Execution scheduling and flow control
- Event dispatching
- Latent action management (delays, timers)
- Performance monitoring
- Optional JIT compilation hooks
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

from .data_types import BlueprintType, convert_value
from .execution_context import (
    ExecutionContext,
    ExecutionState,
    StackFrame,
    acquire_context,
    release_context,
)
from .graph_editor import BlueprintGraph, Connection
from .node_types import Node, Pin, PinDirection, PinKind


class VMInstruction(Enum):
    """Virtual machine instruction types."""
    NOP = auto()
    PUSH = auto()
    POP = auto()
    LOAD_VAR = auto()
    STORE_VAR = auto()
    LOAD_PIN = auto()
    STORE_PIN = auto()
    CALL_NODE = auto()
    JUMP = auto()
    JUMP_IF = auto()
    JUMP_IF_NOT = auto()
    RETURN = auto()
    YIELD = auto()  # For latent nodes
    HALT = auto()


@dataclass
class VMOp:
    """A single VM operation."""
    instruction: VMInstruction
    operand: Any = None
    node_id: Optional[str] = None
    pin_id: Optional[str] = None


@dataclass
class LatentAction:
    """A pending latent action (e.g., delay)."""
    id: str
    node_id: str
    resume_time: float
    context_snapshot: Dict[str, Any]
    resume_pin_id: Optional[str] = None


@dataclass
class ExecutionStats:
    """Statistics for a single execution."""
    start_time: float = 0.0
    end_time: float = 0.0
    node_count: int = 0
    instruction_count: int = 0
    latent_actions: int = 0
    peak_stack_depth: int = 0
    errors: int = 0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class EventDispatcher:
    """Manages event subscriptions and dispatching."""

    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = {}
        self._pending_events: List[Tuple[str, Dict[str, Any]]] = []

    def subscribe(self, event_name: str, handler: Callable) -> None:
        """Subscribe to an event."""
        if event_name not in self._handlers:
            self._handlers[event_name] = []
        self._handlers[event_name].append(handler)

    def unsubscribe(self, event_name: str, handler: Callable) -> bool:
        """Unsubscribe from an event."""
        if event_name in self._handlers:
            try:
                self._handlers[event_name].remove(handler)
                return True
            except ValueError:
                pass
        return False

    def dispatch(self, event_name: str, params: Optional[Dict[str, Any]] = None) -> int:
        """Dispatch an event to all handlers."""
        handlers = self._handlers.get(event_name, [])
        for handler in handlers:
            handler(params or {})
        return len(handlers)

    def queue_event(self, event_name: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Queue an event for later dispatch."""
        self._pending_events.append((event_name, params or {}))

    def process_queued(self) -> int:
        """Process all queued events."""
        count = 0
        while self._pending_events:
            event_name, params = self._pending_events.pop(0)
            count += self.dispatch(event_name, params)
        return count

    def clear(self) -> None:
        """Clear all handlers and queued events."""
        self._handlers.clear()
        self._pending_events.clear()


class BlueprintVM:
    """Virtual machine for executing blueprints."""

    def __init__(self, graph: BlueprintGraph):
        self.graph = graph
        self._context: Optional[ExecutionContext] = None
        self._current_node: Optional[Node] = None
        self._instruction_pointer = 0
        self._value_stack: List[Any] = []
        self._latent_actions: Dict[str, LatentAction] = {}
        self._stats = ExecutionStats()

        # Compiled bytecode (if JIT compiled)
        self._bytecode: List[VMOp] = []
        self._is_compiled = False

        # Execution limits
        self.max_iterations = 100000
        self.max_stack_depth = 1000

    def execute_from_entry(
        self,
        entry_node_id: str,
        context: Optional[ExecutionContext] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> ExecutionStats:
        """Execute the graph starting from an entry node."""
        self._stats = ExecutionStats(start_time=time.time())

        # Get or create context
        if context:
            self._context = context
        else:
            self._context = acquire_context(blueprint_id=self.graph.id)

        # Set parameters
        if params:
            for name, value in params.items():
                self._context.set_variable(name, value)

        # Get entry node
        entry_node = self.graph.get_node(entry_node_id)
        if not entry_node:
            self._stats.errors += 1
            self._stats.end_time = time.time()
            return self._stats

        self._context.begin_execution()

        try:
            # Execute from entry
            self._execute_node(entry_node)

            # Process any pending latent actions
            self._process_latent_actions()

        except Exception as e:
            self._stats.errors += 1
            self._context.report_error(
                message=str(e),
                node_id=self._current_node.id if self._current_node else "",
                exception=e
            )
        finally:
            self._context.end_execution(self._stats.errors == 0)

        self._stats.end_time = time.time()

        # Release context if we created it
        if not context:
            release_context(self._context)
            self._context = None

        return self._stats

    def _execute_node(self, node: Node) -> None:
        """Execute a single node."""
        if not self._context.is_running():
            return

        if not self._context.increment_instruction():
            self._context.report_error(
                "Maximum instruction limit reached",
                node.id
            )
            return

        self._current_node = node
        self._context.current_node_id = node.id
        self._stats.node_count += 1
        self._stats.instruction_count += 1

        # Check stack depth
        if self._context.get_stack_depth() > self.max_stack_depth:
            self._context.report_error(
                "Maximum stack depth exceeded",
                node.id
            )
            return

        self._stats.peak_stack_depth = max(
            self._stats.peak_stack_depth,
            self._context.get_stack_depth()
        )

        # Resolve input data pins (pull data from connected nodes)
        self._resolve_input_pins(node)

        # Execute the node
        next_pin_id = node.execute(self._context)

        # Handle latent nodes
        if node.get_metadata().is_latent and self._context.state == ExecutionState.WAITING:
            self._stats.latent_actions += 1
            return

        # Follow execution wire
        if next_pin_id:
            self._follow_execution_wire(node, next_pin_id)

    def _resolve_input_pins(self, node: Node) -> None:
        """Resolve all input data pins by pulling from connected nodes."""
        for pin in node.input_pins.values():
            if pin.kind == PinKind.DATA and pin.is_connected:
                # Find the connection
                for conn in self.graph.get_incoming_connections(node.id):
                    if conn.target_pin_id == pin.id:
                        # Get value from source
                        source_node = self.graph.get_node(conn.source_node_id)
                        if source_node:
                            source_pin = None
                            for sp in source_node.output_pins.values():
                                if sp.id == conn.source_pin_id:
                                    source_pin = sp
                                    break

                            if source_pin:
                                # For pure nodes, execute them to get value
                                if source_node.get_metadata().is_pure:
                                    self._resolve_input_pins(source_node)
                                    source_node.execute(self._context)

                                # Get and convert value
                                value = source_pin.get_value()
                                if pin.data_type != source_pin.data_type:
                                    value = convert_value(
                                        value,
                                        source_pin.data_type,
                                        pin.data_type
                                    )
                                pin.set_value(value)
                        break

    def _follow_execution_wire(self, node: Node, pin_id: str) -> None:
        """Follow an execution wire to the next node."""
        # Find the connection from this output pin
        for conn in self.graph.get_outgoing_connections(node.id):
            if conn.source_pin_id == pin_id and conn.is_execution:
                # Get the target node
                target_node = self.graph.get_node(conn.target_node_id)
                if target_node:
                    self._execute_node(target_node)
                break

    def _process_latent_actions(self) -> None:
        """Process pending latent actions."""
        current_time = time.time()
        completed = []

        for action_id, action in self._latent_actions.items():
            if current_time >= action.resume_time:
                completed.append(action_id)

                # Resume execution
                if action.resume_pin_id:
                    node = self.graph.get_node(action.node_id)
                    if node:
                        self._follow_execution_wire(node, action.resume_pin_id)

        # Remove completed actions
        for action_id in completed:
            del self._latent_actions[action_id]

    def schedule_latent(
        self,
        node_id: str,
        delay: float,
        resume_pin_id: str
    ) -> str:
        """Schedule a latent action (e.g., delay)."""
        import uuid
        action_id = str(uuid.uuid4())

        action = LatentAction(
            id=action_id,
            node_id=node_id,
            resume_time=time.time() + delay,
            context_snapshot=self._context.snapshot() if self._context else {},
            resume_pin_id=resume_pin_id
        )

        self._latent_actions[action_id] = action
        return action_id

    def cancel_latent(self, action_id: str) -> bool:
        """Cancel a latent action."""
        if action_id in self._latent_actions:
            del self._latent_actions[action_id]
            return True
        return False

    def tick(self, delta_time: float) -> None:
        """Update the VM (process latent actions)."""
        self._process_latent_actions()

    def reset(self) -> None:
        """Reset the VM state."""
        self._current_node = None
        self._instruction_pointer = 0
        self._value_stack.clear()
        self._latent_actions.clear()
        self._stats = ExecutionStats()


class BlueprintRuntime:
    """High-level runtime for executing blueprints."""

    def __init__(self):
        self._vms: Dict[str, BlueprintVM] = {}
        self._event_dispatcher = EventDispatcher()
        self._active_blueprints: Set[str] = set()
        self._paused_blueprints: Set[str] = set()

        # Global context
        self._global_context = ExecutionContext()
        self._delta_time: float = 0.016

        # Performance tracking
        self._total_stats = ExecutionStats()
        self._frame_stats: List[ExecutionStats] = []
        self._max_frame_stats = 60

    def register_blueprint(self, graph: BlueprintGraph) -> BlueprintVM:
        """Register a blueprint for execution."""
        vm = BlueprintVM(graph)
        self._vms[graph.id] = vm
        return vm

    def unregister_blueprint(self, graph_id: str) -> bool:
        """Unregister a blueprint."""
        if graph_id in self._vms:
            del self._vms[graph_id]
            self._active_blueprints.discard(graph_id)
            self._paused_blueprints.discard(graph_id)
            return True
        return False

    def get_vm(self, graph_id: str) -> Optional[BlueprintVM]:
        """Get the VM for a blueprint."""
        return self._vms.get(graph_id)

    # =========================================================================
    # EXECUTION
    # =========================================================================

    def execute_event(
        self,
        graph_id: str,
        event_name: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[ExecutionStats]:
        """Execute a specific event in a blueprint."""
        vm = self._vms.get(graph_id)
        if not vm:
            return None

        # Find the event node
        for entry_id in vm.graph.entry_points:
            entry_node = vm.graph.get_node(entry_id)
            if entry_node:
                meta = entry_node.get_metadata()
                if event_name in meta.display_name:
                    return vm.execute_from_entry(entry_id, params=params)

        return None

    def begin_play(self, graph_id: str) -> Optional[ExecutionStats]:
        """Trigger BeginPlay event."""
        self._active_blueprints.add(graph_id)
        return self.execute_event(graph_id, "BeginPlay")

    def end_play(self, graph_id: str) -> None:
        """Trigger EndPlay and deactivate."""
        self.execute_event(graph_id, "EndPlay")
        self._active_blueprints.discard(graph_id)

    def tick(self, delta_time: float) -> ExecutionStats:
        """Tick all active blueprints."""
        self._delta_time = delta_time
        frame_stats = ExecutionStats(start_time=time.time())

        for graph_id in list(self._active_blueprints):
            if graph_id in self._paused_blueprints:
                continue

            vm = self._vms.get(graph_id)
            if vm:
                # Update latent actions
                vm.tick(delta_time)

                # Execute Tick event
                stats = self.execute_event(graph_id, "Tick", {"DeltaTime": delta_time})
                if stats:
                    frame_stats.node_count += stats.node_count
                    frame_stats.instruction_count += stats.instruction_count

        frame_stats.end_time = time.time()

        # Track frame stats
        self._frame_stats.append(frame_stats)
        while len(self._frame_stats) > self._max_frame_stats:
            self._frame_stats.pop(0)

        # Update total stats
        self._total_stats.node_count += frame_stats.node_count
        self._total_stats.instruction_count += frame_stats.instruction_count

        return frame_stats

    def pause_blueprint(self, graph_id: str) -> bool:
        """Pause a blueprint's execution."""
        if graph_id in self._active_blueprints:
            self._paused_blueprints.add(graph_id)
            return True
        return False

    def resume_blueprint(self, graph_id: str) -> bool:
        """Resume a paused blueprint."""
        if graph_id in self._paused_blueprints:
            self._paused_blueprints.discard(graph_id)
            return True
        return False

    def is_active(self, graph_id: str) -> bool:
        """Check if a blueprint is active."""
        return graph_id in self._active_blueprints

    def is_paused(self, graph_id: str) -> bool:
        """Check if a blueprint is paused."""
        return graph_id in self._paused_blueprints

    # =========================================================================
    # EVENTS
    # =========================================================================

    def dispatch_event(
        self,
        event_name: str,
        params: Optional[Dict[str, Any]] = None
    ) -> int:
        """Dispatch a global event to all blueprints."""
        count = 0
        for graph_id in self._active_blueprints:
            if graph_id not in self._paused_blueprints:
                result = self.execute_event(graph_id, event_name, params)
                if result:
                    count += 1
        return count

    def subscribe_event(self, event_name: str, handler: Callable) -> None:
        """Subscribe to a global event."""
        self._event_dispatcher.subscribe(event_name, handler)

    def unsubscribe_event(self, event_name: str, handler: Callable) -> bool:
        """Unsubscribe from a global event."""
        return self._event_dispatcher.unsubscribe(event_name, handler)

    # =========================================================================
    # INPUT HANDLING
    # =========================================================================

    def input_action(
        self,
        action_name: str,
        pressed: bool,
        params: Optional[Dict[str, Any]] = None
    ) -> int:
        """Handle an input action."""
        event_params = params or {}
        event_params["ActionName"] = action_name
        event_params["Pressed"] = pressed

        return self.dispatch_event("InputAction", event_params)

    def input_axis(
        self,
        axis_name: str,
        value: float
    ) -> int:
        """Handle an input axis."""
        return self.dispatch_event("InputAxis", {
            "AxisName": axis_name,
            "Value": value
        })

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get runtime statistics."""
        avg_frame_time = 0.0
        if self._frame_stats:
            avg_frame_time = sum(s.duration for s in self._frame_stats) / len(self._frame_stats)

        return {
            "active_blueprints": len(self._active_blueprints),
            "paused_blueprints": len(self._paused_blueprints),
            "total_vms": len(self._vms),
            "total_nodes_executed": self._total_stats.node_count,
            "total_instructions": self._total_stats.instruction_count,
            "average_frame_time_ms": avg_frame_time * 1000,
            "frame_stats_count": len(self._frame_stats)
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._total_stats = ExecutionStats()
        self._frame_stats.clear()


# Global runtime instance
_runtime: Optional[BlueprintRuntime] = None


def get_runtime() -> BlueprintRuntime:
    """Get the global blueprint runtime."""
    global _runtime
    if _runtime is None:
        _runtime = BlueprintRuntime()
    return _runtime


def execute_blueprint(
    graph: BlueprintGraph,
    event_name: str = "BeginPlay",
    params: Optional[Dict[str, Any]] = None
) -> ExecutionStats:
    """Execute a blueprint with the global runtime."""
    runtime = get_runtime()
    vm = runtime.register_blueprint(graph)

    # Find and execute the event
    for entry_id in graph.entry_points:
        entry_node = graph.get_node(entry_id)
        if entry_node:
            meta = entry_node.get_metadata()
            if event_name in meta.display_name:
                return vm.execute_from_entry(entry_id, params=params)

    return ExecutionStats()
