"""
FlowForge Blueprint Debugger - Blueprint debugging with breakpoints, stepping, and watch.

Provides comprehensive debugging capabilities:
- Breakpoint management (conditional, hit count)
- Step execution (into, over, out)
- Variable watching and inspection
- Call stack visualization
- Execution history and replay
- Performance profiling per node
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .execution_context import ExecutionContext, ExecutionState, StackFrame
from .graph_editor import BlueprintGraph
from .node_types import Node, Pin


class BreakpointType(Enum):
    """Types of breakpoints."""
    UNCONDITIONAL = auto()
    CONDITIONAL = auto()
    HIT_COUNT = auto()
    LOG_POINT = auto()  # Log but don't break


class StepMode(Enum):
    """Step execution modes."""
    NONE = auto()
    INTO = auto()  # Step into function calls
    OVER = auto()  # Step over function calls
    OUT = auto()   # Step out of current function


@dataclass
class Breakpoint:
    """A debugging breakpoint."""
    id: str
    node_id: str
    bp_type: BreakpointType = BreakpointType.UNCONDITIONAL
    is_enabled: bool = True
    condition: str = ""  # Expression to evaluate
    hit_count_target: int = 0
    current_hit_count: int = 0
    log_message: str = ""
    actions: List[str] = field(default_factory=list)

    def should_break(self, context: ExecutionContext) -> bool:
        """Determine if this breakpoint should trigger."""
        if not self.is_enabled:
            return False

        self.current_hit_count += 1

        if self.bp_type == BreakpointType.UNCONDITIONAL:
            return True

        elif self.bp_type == BreakpointType.HIT_COUNT:
            return self.current_hit_count >= self.hit_count_target

        elif self.bp_type == BreakpointType.CONDITIONAL:
            return self._evaluate_condition(context)

        elif self.bp_type == BreakpointType.LOG_POINT:
            # Log but don't break
            return False

        return False

    def _evaluate_condition(self, context: ExecutionContext) -> bool:
        """Evaluate the breakpoint condition."""
        if not self.condition:
            return True

        try:
            # Create evaluation context
            eval_globals = {
                "get_var": context.get_variable,
                "has_var": context.has_variable,
            }
            return bool(eval(self.condition, eval_globals))
        except Exception:
            return False

    def reset_hit_count(self) -> None:
        """Reset the hit count."""
        self.current_hit_count = 0


@dataclass
class WatchExpression:
    """A watched variable or expression."""
    id: str
    expression: str
    name: str = ""
    is_enabled: bool = True
    last_value: Any = None
    last_update_time: float = 0.0
    error: Optional[str] = None

    def evaluate(self, context: ExecutionContext) -> Any:
        """Evaluate the watch expression."""
        if not self.is_enabled:
            return self.last_value

        try:
            # Create evaluation context
            eval_globals = {
                "get_var": context.get_variable,
                "has_var": context.has_variable,
                "stack_depth": context.get_stack_depth,
            }

            # Check if it's a simple variable name
            if self.expression.isidentifier():
                value = context.get_variable(self.expression)
            else:
                value = eval(self.expression, eval_globals)

            self.last_value = value
            self.last_update_time = time.time()
            self.error = None
            return value

        except Exception as e:
            self.error = str(e)
            return None


@dataclass
class ExecutionHistoryEntry:
    """An entry in the execution history."""
    timestamp: float
    node_id: str
    node_name: str
    action: str  # "enter", "exit", "break", etc.
    variables: Dict[str, Any] = field(default_factory=dict)
    stack_depth: int = 0


@dataclass
class NodeProfile:
    """Profiling data for a node."""
    node_id: str
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    last_time: float = 0.0

    @property
    def avg_time(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.total_time / self.call_count

    def record(self, duration: float) -> None:
        self.call_count += 1
        self.total_time += duration
        self.last_time = duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)


class DebugState(Enum):
    """State of the debugger."""
    DETACHED = auto()
    ATTACHED = auto()
    RUNNING = auto()
    PAUSED = auto()
    STEPPING = auto()


class BlueprintDebugger:
    """Debugger for blueprint execution."""

    def __init__(self, graph: BlueprintGraph):
        self.graph = graph
        self._state = DebugState.DETACHED
        self._context: Optional[ExecutionContext] = None

        # Breakpoints
        self._breakpoints: Dict[str, Breakpoint] = {}
        self._breakpoints_by_node: Dict[str, List[str]] = {}

        # Watch expressions
        self._watches: Dict[str, WatchExpression] = {}

        # Stepping
        self._step_mode = StepMode.NONE
        self._step_target_depth: int = 0

        # History
        self._history: List[ExecutionHistoryEntry] = []
        self._max_history = 1000
        self._recording = False

        # Profiling
        self._profiles: Dict[str, NodeProfile] = {}
        self._profiling_enabled = False
        self._current_node_start: float = 0.0

        # Callbacks
        self.on_breakpoint_hit: Optional[Callable[[Breakpoint, Node], None]] = None
        self.on_step_complete: Optional[Callable[[Node], None]] = None
        self.on_state_changed: Optional[Callable[[DebugState], None]] = None
        self.on_watch_updated: Optional[Callable[[WatchExpression], None]] = None

        # Current state
        self._current_node_id: Optional[str] = None
        self._paused_at_node: Optional[str] = None

    # =========================================================================
    # ATTACH/DETACH
    # =========================================================================

    def attach(self, context: ExecutionContext) -> bool:
        """Attach the debugger to an execution context."""
        if self._state != DebugState.DETACHED:
            return False

        self._context = context
        self._state = DebugState.ATTACHED
        self._notify_state_change()
        return True

    def detach(self) -> bool:
        """Detach the debugger."""
        if self._state == DebugState.DETACHED:
            return False

        self._context = None
        self._state = DebugState.DETACHED
        self._step_mode = StepMode.NONE
        self._notify_state_change()
        return True

    def is_attached(self) -> bool:
        """Check if debugger is attached."""
        return self._state != DebugState.DETACHED

    # =========================================================================
    # BREAKPOINTS
    # =========================================================================

    def add_breakpoint(
        self,
        node_id: str,
        bp_type: BreakpointType = BreakpointType.UNCONDITIONAL,
        condition: str = "",
        hit_count: int = 0,
        log_message: str = ""
    ) -> Breakpoint:
        """Add a breakpoint at a node."""
        import uuid
        bp_id = str(uuid.uuid4())

        bp = Breakpoint(
            id=bp_id,
            node_id=node_id,
            bp_type=bp_type,
            condition=condition,
            hit_count_target=hit_count,
            log_message=log_message
        )

        self._breakpoints[bp_id] = bp

        if node_id not in self._breakpoints_by_node:
            self._breakpoints_by_node[node_id] = []
        self._breakpoints_by_node[node_id].append(bp_id)

        return bp

    def remove_breakpoint(self, bp_id: str) -> bool:
        """Remove a breakpoint."""
        bp = self._breakpoints.pop(bp_id, None)
        if bp:
            if bp.node_id in self._breakpoints_by_node:
                self._breakpoints_by_node[bp.node_id].remove(bp_id)
            return True
        return False

    def get_breakpoint(self, bp_id: str) -> Optional[Breakpoint]:
        """Get a breakpoint by ID."""
        return self._breakpoints.get(bp_id)

    def get_breakpoints_at_node(self, node_id: str) -> List[Breakpoint]:
        """Get all breakpoints at a node."""
        bp_ids = self._breakpoints_by_node.get(node_id, [])
        return [self._breakpoints[bp_id] for bp_id in bp_ids if bp_id in self._breakpoints]

    def enable_breakpoint(self, bp_id: str) -> bool:
        """Enable a breakpoint."""
        bp = self._breakpoints.get(bp_id)
        if bp:
            bp.is_enabled = True
            return True
        return False

    def disable_breakpoint(self, bp_id: str) -> bool:
        """Disable a breakpoint."""
        bp = self._breakpoints.get(bp_id)
        if bp:
            bp.is_enabled = False
            return True
        return False

    def toggle_breakpoint(self, bp_id: str) -> Optional[bool]:
        """Toggle a breakpoint. Returns new state or None if not found."""
        bp = self._breakpoints.get(bp_id)
        if bp:
            bp.is_enabled = not bp.is_enabled
            return bp.is_enabled
        return None

    def clear_all_breakpoints(self) -> int:
        """Remove all breakpoints. Returns count removed."""
        count = len(self._breakpoints)
        self._breakpoints.clear()
        self._breakpoints_by_node.clear()
        return count

    def has_breakpoint_at(self, node_id: str) -> bool:
        """Check if there's a breakpoint at a node."""
        return node_id in self._breakpoints_by_node and len(self._breakpoints_by_node[node_id]) > 0

    # =========================================================================
    # EXECUTION CONTROL
    # =========================================================================

    def on_node_enter(self, node: Node) -> bool:
        """Called when execution enters a node. Returns True if should pause."""
        self._current_node_id = node.id

        if self._profiling_enabled:
            self._current_node_start = time.time()

        # Record history
        if self._recording:
            self._record_history(node, "enter")

        # Check breakpoints
        for bp in self.get_breakpoints_at_node(node.id):
            if bp.should_break(self._context):
                self._pause_at_breakpoint(bp, node)
                return True

        # Check step mode
        if self._should_pause_for_step():
            self._pause_at_step(node)
            return True

        return False

    def on_node_exit(self, node: Node) -> None:
        """Called when execution exits a node."""
        if self._profiling_enabled:
            duration = time.time() - self._current_node_start
            self._record_profile(node.id, duration)

        if self._recording:
            self._record_history(node, "exit")

    def _pause_at_breakpoint(self, bp: Breakpoint, node: Node) -> None:
        """Pause execution at a breakpoint."""
        self._state = DebugState.PAUSED
        self._paused_at_node = node.id

        if self._context:
            self._context.pause_execution()

        if self.on_breakpoint_hit:
            self.on_breakpoint_hit(bp, node)

        self._notify_state_change()

    def _pause_at_step(self, node: Node) -> None:
        """Pause execution for stepping."""
        self._state = DebugState.PAUSED
        self._paused_at_node = node.id
        self._step_mode = StepMode.NONE

        if self._context:
            self._context.pause_execution()

        if self.on_step_complete:
            self.on_step_complete(node)

        self._notify_state_change()

    def _should_pause_for_step(self) -> bool:
        """Check if should pause for current step mode."""
        if self._step_mode == StepMode.NONE:
            return False

        if not self._context:
            return False

        current_depth = self._context.get_stack_depth()

        if self._step_mode == StepMode.INTO:
            return True

        elif self._step_mode == StepMode.OVER:
            return current_depth <= self._step_target_depth

        elif self._step_mode == StepMode.OUT:
            return current_depth < self._step_target_depth

        return False

    def resume(self) -> bool:
        """Resume execution."""
        if self._state != DebugState.PAUSED:
            return False

        self._state = DebugState.RUNNING
        self._step_mode = StepMode.NONE
        self._paused_at_node = None

        if self._context:
            self._context.resume_execution()

        self._notify_state_change()
        return True

    def step_into(self) -> bool:
        """Step into the next node."""
        if self._state != DebugState.PAUSED:
            return False

        self._step_mode = StepMode.INTO
        self._state = DebugState.STEPPING

        if self._context:
            self._context.resume_execution()

        self._notify_state_change()
        return True

    def step_over(self) -> bool:
        """Step over the current node."""
        if self._state != DebugState.PAUSED:
            return False

        self._step_mode = StepMode.OVER
        self._step_target_depth = self._context.get_stack_depth() if self._context else 0
        self._state = DebugState.STEPPING

        if self._context:
            self._context.resume_execution()

        self._notify_state_change()
        return True

    def step_out(self) -> bool:
        """Step out of the current function."""
        if self._state != DebugState.PAUSED:
            return False

        self._step_mode = StepMode.OUT
        self._step_target_depth = self._context.get_stack_depth() if self._context else 0
        self._state = DebugState.STEPPING

        if self._context:
            self._context.resume_execution()

        self._notify_state_change()
        return True

    def stop(self) -> bool:
        """Stop execution completely."""
        if self._state == DebugState.DETACHED:
            return False

        if self._context:
            self._context.end_execution(success=False)

        self._state = DebugState.ATTACHED
        self._step_mode = StepMode.NONE
        self._paused_at_node = None
        self._notify_state_change()
        return True

    # =========================================================================
    # WATCH EXPRESSIONS
    # =========================================================================

    def add_watch(self, expression: str, name: str = "") -> WatchExpression:
        """Add a watch expression."""
        import uuid
        watch_id = str(uuid.uuid4())

        watch = WatchExpression(
            id=watch_id,
            expression=expression,
            name=name or expression
        )

        self._watches[watch_id] = watch

        # Evaluate immediately if attached
        if self._context:
            watch.evaluate(self._context)

        return watch

    def remove_watch(self, watch_id: str) -> bool:
        """Remove a watch expression."""
        return self._watches.pop(watch_id, None) is not None

    def get_watch(self, watch_id: str) -> Optional[WatchExpression]:
        """Get a watch by ID."""
        return self._watches.get(watch_id)

    def get_all_watches(self) -> List[WatchExpression]:
        """Get all watch expressions."""
        return list(self._watches.values())

    def update_watches(self) -> None:
        """Update all watch values."""
        if not self._context:
            return

        for watch in self._watches.values():
            old_value = watch.last_value
            watch.evaluate(self._context)

            if watch.last_value != old_value and self.on_watch_updated:
                self.on_watch_updated(watch)

    def evaluate_expression(self, expression: str) -> Tuple[Any, Optional[str]]:
        """Evaluate an expression. Returns (value, error)."""
        if not self._context:
            return (None, "Debugger not attached")

        temp_watch = WatchExpression(id="temp", expression=expression)
        value = temp_watch.evaluate(self._context)
        return (value, temp_watch.error)

    # =========================================================================
    # INSPECTION
    # =========================================================================

    def get_call_stack(self) -> List[Dict[str, Any]]:
        """Get the current call stack."""
        if not self._context:
            return []

        stack = []
        for frame in self._context.get_call_stack():
            node = self.graph.get_node(frame.node_id)
            stack.append({
                "function": frame.function_name,
                "node_id": frame.node_id,
                "node_name": node.get_metadata().display_name if node else "Unknown",
                "locals": {k: v.value for k, v in frame.local_variables.items()},
                "execution_count": frame.execution_count
            })
        return stack

    def get_local_variables(self, frame_index: int = -1) -> Dict[str, Any]:
        """Get local variables for a stack frame."""
        if not self._context:
            return {}

        stack = self._context.get_call_stack()
        if not stack or frame_index >= len(stack):
            return {}

        frame = stack[frame_index]
        return {k: v.value for k, v in frame.local_variables.items()}

    def get_all_variables(self) -> Dict[str, Dict[str, Any]]:
        """Get all variables by scope."""
        if not self._context:
            return {}

        return self._context.snapshot().get("variables", {})

    def inspect_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a node."""
        node = self.graph.get_node(node_id)
        if not node:
            return None

        meta = node.get_metadata()
        profile = self._profiles.get(node_id)

        return {
            "id": node.id,
            "name": meta.display_name,
            "category": meta.category.name,
            "position": node.position,
            "is_disabled": node.is_disabled,
            "input_pins": [
                {
                    "name": pin.name,
                    "type": pin.data_type.type_name() if pin.data_type else "Unknown",
                    "value": pin.get_value(),
                    "is_connected": pin.is_connected
                }
                for pin in node.input_pins.values()
            ],
            "output_pins": [
                {
                    "name": pin.name,
                    "type": pin.data_type.type_name() if pin.data_type else "Unknown",
                    "value": pin.get_value(),
                    "is_connected": pin.is_connected
                }
                for pin in node.output_pins.values()
            ],
            "breakpoints": [
                {
                    "id": bp.id,
                    "type": bp.bp_type.name,
                    "enabled": bp.is_enabled,
                    "hit_count": bp.current_hit_count
                }
                for bp in self.get_breakpoints_at_node(node_id)
            ],
            "profile": {
                "call_count": profile.call_count,
                "avg_time_ms": profile.avg_time * 1000,
                "total_time_ms": profile.total_time * 1000
            } if profile else None
        }

    # =========================================================================
    # HISTORY
    # =========================================================================

    def start_recording(self) -> None:
        """Start recording execution history."""
        self._recording = True
        self._history.clear()

    def stop_recording(self) -> None:
        """Stop recording execution history."""
        self._recording = False

    def is_recording(self) -> bool:
        """Check if recording."""
        return self._recording

    def get_history(self, limit: Optional[int] = None) -> List[ExecutionHistoryEntry]:
        """Get execution history."""
        if limit:
            return self._history[-limit:]
        return list(self._history)

    def clear_history(self) -> None:
        """Clear execution history."""
        self._history.clear()

    def _record_history(self, node: Node, action: str) -> None:
        """Record a history entry."""
        entry = ExecutionHistoryEntry(
            timestamp=time.time(),
            node_id=node.id,
            node_name=node.get_metadata().display_name,
            action=action,
            variables=self._context.snapshot()["variables"] if self._context else {},
            stack_depth=self._context.get_stack_depth() if self._context else 0
        )

        self._history.append(entry)

        while len(self._history) > self._max_history:
            self._history.pop(0)

    # =========================================================================
    # PROFILING
    # =========================================================================

    def enable_profiling(self) -> None:
        """Enable performance profiling."""
        self._profiling_enabled = True
        self._profiles.clear()

    def disable_profiling(self) -> None:
        """Disable performance profiling."""
        self._profiling_enabled = False

    def is_profiling(self) -> bool:
        """Check if profiling is enabled."""
        return self._profiling_enabled

    def get_profile(self, node_id: str) -> Optional[NodeProfile]:
        """Get profile for a node."""
        return self._profiles.get(node_id)

    def get_all_profiles(self) -> Dict[str, NodeProfile]:
        """Get all profiles."""
        return dict(self._profiles)

    def get_hottest_nodes(self, limit: int = 10) -> List[NodeProfile]:
        """Get nodes with highest total execution time."""
        sorted_profiles = sorted(
            self._profiles.values(),
            key=lambda p: p.total_time,
            reverse=True
        )
        return sorted_profiles[:limit]

    def clear_profiles(self) -> None:
        """Clear all profiling data."""
        self._profiles.clear()

    def _record_profile(self, node_id: str, duration: float) -> None:
        """Record profile data for a node."""
        if node_id not in self._profiles:
            self._profiles[node_id] = NodeProfile(node_id=node_id)
        self._profiles[node_id].record(duration)

    # =========================================================================
    # STATE
    # =========================================================================

    def get_state(self) -> DebugState:
        """Get the current debugger state."""
        return self._state

    def is_paused(self) -> bool:
        """Check if execution is paused."""
        return self._state == DebugState.PAUSED

    def get_paused_node(self) -> Optional[Node]:
        """Get the node where execution is paused."""
        if self._paused_at_node:
            return self.graph.get_node(self._paused_at_node)
        return None

    def _notify_state_change(self) -> None:
        """Notify listeners of state change."""
        if self.on_state_changed:
            self.on_state_changed(self._state)

    def get_debug_info(self) -> Dict[str, Any]:
        """Get complete debug information."""
        return {
            "state": self._state.name,
            "is_attached": self.is_attached(),
            "paused_at": self._paused_at_node,
            "step_mode": self._step_mode.name,
            "breakpoint_count": len(self._breakpoints),
            "enabled_breakpoints": sum(1 for bp in self._breakpoints.values() if bp.is_enabled),
            "watch_count": len(self._watches),
            "history_entries": len(self._history),
            "is_recording": self._recording,
            "is_profiling": self._profiling_enabled,
            "profiled_nodes": len(self._profiles)
        }
