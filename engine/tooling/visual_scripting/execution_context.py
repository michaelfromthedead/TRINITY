"""
FlowForge Execution Context - Execution context with variables, call stack, and scope.

Provides the runtime context for blueprint execution:
- Variable storage and scoping (local, instance, class)
- Call stack management
- Execution state tracking
- Context isolation for function calls
- Watch expressions for debugging
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

from .data_types import BlueprintType, WildcardType


class VariableScope(Enum):
    """Scope levels for variables."""
    LOCAL = auto()      # Local to current function/macro
    INSTANCE = auto()   # Instance variable (per-object)
    CLASS = auto()      # Class/static variable (shared)
    GLOBAL = auto()     # Global scope
    TEMPORARY = auto()  # Temporary (execution-only)


@dataclass
class Variable:
    """A blueprint variable."""
    name: str
    data_type: Type[BlueprintType]
    value: Any
    scope: VariableScope = VariableScope.INSTANCE
    is_const: bool = False
    is_exposed: bool = False  # Editable in details panel
    category: str = ""
    tooltip: str = ""
    replication: str = "None"  # None, Replicated, RepNotify

    def get_value(self) -> Any:
        """Get the variable value."""
        return self.value

    def set_value(self, new_value: Any) -> bool:
        """Set the variable value. Returns False if const."""
        if self.is_const:
            return False
        self.value = self.data_type.coerce(new_value)
        return True

    def clone(self) -> Variable:
        """Create a copy of this variable."""
        return Variable(
            name=self.name,
            data_type=self.data_type,
            value=copy.deepcopy(self.value),
            scope=self.scope,
            is_const=self.is_const,
            is_exposed=self.is_exposed,
            category=self.category,
            tooltip=self.tooltip,
            replication=self.replication
        )


@dataclass
class StackFrame:
    """A single frame in the call stack."""
    function_name: str
    node_id: str
    local_variables: Dict[str, Variable] = field(default_factory=dict)
    return_value: Any = None
    start_time: float = field(default_factory=time.time)
    execution_count: int = 0
    is_latent: bool = False  # Frame is paused (e.g., delay node)
    latent_resume_time: float = 0.0

    def get_local(self, name: str) -> Optional[Variable]:
        """Get a local variable."""
        return self.local_variables.get(name)

    def set_local(self, name: str, value: Any, data_type: Type[BlueprintType] = WildcardType) -> Variable:
        """Set or create a local variable."""
        if name in self.local_variables:
            self.local_variables[name].set_value(value)
        else:
            self.local_variables[name] = Variable(
                name=name,
                data_type=data_type,
                value=data_type.coerce(value),
                scope=VariableScope.LOCAL
            )
        return self.local_variables[name]

    def elapsed_time(self) -> float:
        """Get time since frame was created."""
        return time.time() - self.start_time


class ExecutionState(Enum):
    """State of blueprint execution."""
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()  # Breakpoint or step mode
    WAITING = auto()  # Latent operation
    COMPLETED = auto()
    ERROR = auto()


@dataclass
class ExecutionError:
    """Error during execution."""
    message: str
    node_id: str
    pin_id: Optional[str] = None
    stack_trace: List[str] = field(default_factory=list)
    exception: Optional[Exception] = None


class ExecutionContext:
    """Runtime context for blueprint execution."""

    def __init__(
        self,
        blueprint_id: str = "",
        owner_object: Any = None,
        delta_time: float = 0.016
    ):
        self.blueprint_id = blueprint_id
        self.owner_object = owner_object
        self.delta_time = delta_time

        # Variable storage by scope
        self._local_variables: Dict[str, Variable] = {}
        self._instance_variables: Dict[str, Variable] = {}
        self._class_variables: Dict[str, Variable] = {}
        self._global_variables: Dict[str, Variable] = {}
        self._temporary_variables: Dict[str, Variable] = {}

        # Call stack
        self._call_stack: List[StackFrame] = []

        # Execution state
        self.state = ExecutionState.IDLE
        self.current_node_id: Optional[str] = None
        self.errors: List[ExecutionError] = []

        # Timing
        self.start_time: float = 0.0
        self.total_execution_time: float = 0.0
        self.instruction_count: int = 0
        self.max_instructions: int = 1000000  # Safety limit

        # Debug output
        self.output_log: List[str] = []

    # =========================================================================
    # VARIABLE ACCESS
    # =========================================================================

    def get_variable(self, name: str, scope: Optional[VariableScope] = None) -> Any:
        """Get a variable value by name.

        If scope is None, searches from most local to most global.
        """
        var = self._find_variable(name, scope)
        if var:
            return var.get_value()
        return None

    def set_variable(
        self,
        name: str,
        value: Any,
        scope: VariableScope = VariableScope.INSTANCE,
        data_type: Type[BlueprintType] = WildcardType
    ) -> bool:
        """Set a variable value."""
        var = self._find_variable(name, scope)
        if var:
            return var.set_value(value)

        # Create new variable
        new_var = Variable(
            name=name,
            data_type=data_type,
            value=data_type.coerce(value),
            scope=scope
        )
        self._get_scope_dict(scope)[name] = new_var
        return True

    def declare_variable(
        self,
        name: str,
        data_type: Type[BlueprintType],
        initial_value: Any = None,
        scope: VariableScope = VariableScope.INSTANCE,
        is_const: bool = False,
        is_exposed: bool = False,
        category: str = "",
        tooltip: str = ""
    ) -> Variable:
        """Declare a new variable with full options."""
        value = initial_value if initial_value is not None else data_type.default_value
        var = Variable(
            name=name,
            data_type=data_type,
            value=data_type.coerce(value),
            scope=scope,
            is_const=is_const,
            is_exposed=is_exposed,
            category=category,
            tooltip=tooltip
        )
        self._get_scope_dict(scope)[name] = var
        return var

    def has_variable(self, name: str, scope: Optional[VariableScope] = None) -> bool:
        """Check if a variable exists."""
        return self._find_variable(name, scope) is not None

    def delete_variable(self, name: str, scope: Optional[VariableScope] = None) -> bool:
        """Delete a variable."""
        if scope:
            scope_dict = self._get_scope_dict(scope)
            if name in scope_dict:
                del scope_dict[name]
                return True
            return False

        # Search all scopes
        for s in VariableScope:
            scope_dict = self._get_scope_dict(s)
            if name in scope_dict:
                del scope_dict[name]
                return True
        return False

    def get_all_variables(self, scope: Optional[VariableScope] = None) -> Dict[str, Variable]:
        """Get all variables in scope(s)."""
        if scope:
            return dict(self._get_scope_dict(scope))

        # Combine all scopes (local shadows outer)
        result = {}
        for s in reversed(list(VariableScope)):
            result.update(self._get_scope_dict(s))
        return result

    def _find_variable(self, name: str, scope: Optional[VariableScope] = None) -> Optional[Variable]:
        """Find a variable by name."""
        if scope:
            return self._get_scope_dict(scope).get(name)

        # Search from most local to most global
        for s in VariableScope:
            var = self._get_scope_dict(s).get(name)
            if var:
                return var
        return None

    def _get_scope_dict(self, scope: VariableScope) -> Dict[str, Variable]:
        """Get the dictionary for a scope."""
        if scope == VariableScope.LOCAL:
            # Local scope is in current stack frame
            if self._call_stack:
                return self._call_stack[-1].local_variables
            return self._local_variables
        elif scope == VariableScope.INSTANCE:
            return self._instance_variables
        elif scope == VariableScope.CLASS:
            return self._class_variables
        elif scope == VariableScope.GLOBAL:
            return self._global_variables
        else:
            return self._temporary_variables

    # =========================================================================
    # CALL STACK
    # =========================================================================

    def push_frame(
        self,
        function_name: str,
        node_id: str,
        initial_locals: Optional[Dict[str, Any]] = None
    ) -> StackFrame:
        """Push a new stack frame."""
        frame = StackFrame(
            function_name=function_name,
            node_id=node_id
        )

        if initial_locals:
            for name, value in initial_locals.items():
                frame.set_local(name, value)

        self._call_stack.append(frame)
        return frame

    def pop_frame(self) -> Optional[StackFrame]:
        """Pop the current stack frame."""
        if self._call_stack:
            return self._call_stack.pop()
        return None

    def get_current_frame(self) -> Optional[StackFrame]:
        """Get the current (top) stack frame."""
        if self._call_stack:
            return self._call_stack[-1]
        return None

    def get_call_stack(self) -> List[StackFrame]:
        """Get the entire call stack."""
        return list(self._call_stack)

    def get_stack_depth(self) -> int:
        """Get the current call stack depth."""
        return len(self._call_stack)

    def get_stack_trace(self) -> List[str]:
        """Get a human-readable stack trace."""
        trace = []
        for i, frame in enumerate(reversed(self._call_stack)):
            trace.append(f"  [{len(self._call_stack) - 1 - i}] {frame.function_name} @ {frame.node_id}")
        return trace

    # =========================================================================
    # EXECUTION CONTROL
    # =========================================================================

    def begin_execution(self) -> None:
        """Start a new execution cycle."""
        self.state = ExecutionState.RUNNING
        self.start_time = time.time()
        self.instruction_count = 0
        self.errors.clear()

    def end_execution(self, success: bool = True) -> None:
        """End the current execution cycle."""
        self.total_execution_time = time.time() - self.start_time
        self.state = ExecutionState.COMPLETED if success else ExecutionState.ERROR

    def pause_execution(self) -> None:
        """Pause execution (for debugging)."""
        self.state = ExecutionState.PAUSED

    def resume_execution(self) -> None:
        """Resume paused execution."""
        if self.state == ExecutionState.PAUSED:
            self.state = ExecutionState.RUNNING

    def wait_for_latent(self, duration: float = 0.0) -> None:
        """Enter waiting state for latent operation."""
        self.state = ExecutionState.WAITING
        if self._call_stack:
            frame = self._call_stack[-1]
            frame.is_latent = True
            frame.latent_resume_time = time.time() + duration

    def check_latent_complete(self) -> bool:
        """Check if latent operation has completed."""
        if not self._call_stack:
            return True

        frame = self._call_stack[-1]
        if not frame.is_latent:
            return True

        if time.time() >= frame.latent_resume_time:
            frame.is_latent = False
            self.state = ExecutionState.RUNNING
            return True

        return False

    def increment_instruction(self) -> bool:
        """Increment instruction count. Returns False if limit reached."""
        self.instruction_count += 1
        return self.instruction_count < self.max_instructions

    def report_error(
        self,
        message: str,
        node_id: str,
        pin_id: Optional[str] = None,
        exception: Optional[Exception] = None
    ) -> None:
        """Report an execution error."""
        error = ExecutionError(
            message=message,
            node_id=node_id,
            pin_id=pin_id,
            stack_trace=self.get_stack_trace(),
            exception=exception
        )
        self.errors.append(error)
        self.state = ExecutionState.ERROR

    def is_running(self) -> bool:
        """Check if execution is currently running."""
        return self.state == ExecutionState.RUNNING

    # =========================================================================
    # DEBUG OUTPUT
    # =========================================================================

    def print_string(self, text: str, to_screen: bool = True, to_log: bool = True) -> None:
        """Print a debug string."""
        if to_log:
            self.output_log.append(text)
        if to_screen and self.owner_object and hasattr(self.owner_object, "display_debug"):
            self.owner_object.display_debug(text)

    def clear_output(self) -> None:
        """Clear the output log."""
        self.output_log.clear()

    # =========================================================================
    # CONTEXT CLONING
    # =========================================================================

    def create_child_context(self) -> ExecutionContext:
        """Create a child context for function calls."""
        child = ExecutionContext(
            blueprint_id=self.blueprint_id,
            owner_object=self.owner_object,
            delta_time=self.delta_time
        )
        # Share class and global variables
        child._class_variables = self._class_variables
        child._global_variables = self._global_variables
        # Copy instance variables (could share depending on use case)
        child._instance_variables = {
            k: v.clone() for k, v in self._instance_variables.items()
        }
        return child

    def snapshot(self) -> Dict[str, Any]:
        """Create a snapshot of the current context state."""
        return {
            "blueprint_id": self.blueprint_id,
            "state": self.state.name,
            "current_node_id": self.current_node_id,
            "instruction_count": self.instruction_count,
            "stack_depth": self.get_stack_depth(),
            "variables": {
                "local": {k: v.value for k, v in self._local_variables.items()},
                "instance": {k: v.value for k, v in self._instance_variables.items()},
                "class": {k: v.value for k, v in self._class_variables.items()},
            },
            "call_stack": [
                {
                    "function": f.function_name,
                    "node": f.node_id,
                    "locals": {k: v.value for k, v in f.local_variables.items()}
                }
                for f in self._call_stack
            ],
            "errors": [
                {
                    "message": e.message,
                    "node": e.node_id,
                    "trace": e.stack_trace
                }
                for e in self.errors
            ]
        }


class ExecutionContextPool:
    """Pool of reusable execution contexts."""

    def __init__(self, initial_size: int = 10, max_size: int = 100):
        self._pool: List[ExecutionContext] = []
        self._max_size = max_size
        self._in_use: Set[int] = set()

        # Pre-allocate initial contexts
        for _ in range(initial_size):
            self._pool.append(ExecutionContext())

    def acquire(
        self,
        blueprint_id: str = "",
        owner_object: Any = None,
        delta_time: float = 0.016
    ) -> ExecutionContext:
        """Acquire a context from the pool."""
        # Find an available context
        for i, ctx in enumerate(self._pool):
            if i not in self._in_use:
                self._in_use.add(i)
                ctx.blueprint_id = blueprint_id
                ctx.owner_object = owner_object
                ctx.delta_time = delta_time
                ctx.state = ExecutionState.IDLE
                return ctx

        # Create new context if pool not at max
        if len(self._pool) < self._max_size:
            ctx = ExecutionContext(blueprint_id, owner_object, delta_time)
            self._pool.append(ctx)
            self._in_use.add(len(self._pool) - 1)
            return ctx

        # Pool exhausted - create temporary context
        return ExecutionContext(blueprint_id, owner_object, delta_time)

    def release(self, context: ExecutionContext) -> None:
        """Release a context back to the pool."""
        try:
            idx = self._pool.index(context)
            if idx in self._in_use:
                self._in_use.discard(idx)
                # Reset context
                context._call_stack.clear()
                context._local_variables.clear()
                context._temporary_variables.clear()
                context.errors.clear()
                context.output_log.clear()
                context.state = ExecutionState.IDLE
        except ValueError:
            # Context not from pool, just let it be garbage collected
            pass

    def get_pool_stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        return {
            "total": len(self._pool),
            "in_use": len(self._in_use),
            "available": len(self._pool) - len(self._in_use),
            "max_size": self._max_size
        }


# Global context pool
_context_pool: Optional[ExecutionContextPool] = None


def get_context_pool() -> ExecutionContextPool:
    """Get the global context pool."""
    global _context_pool
    if _context_pool is None:
        _context_pool = ExecutionContextPool()
    return _context_pool


def acquire_context(
    blueprint_id: str = "",
    owner_object: Any = None,
    delta_time: float = 0.016
) -> ExecutionContext:
    """Acquire a context from the global pool."""
    return get_context_pool().acquire(blueprint_id, owner_object, delta_time)


def release_context(context: ExecutionContext) -> None:
    """Release a context to the global pool."""
    get_context_pool().release(context)
