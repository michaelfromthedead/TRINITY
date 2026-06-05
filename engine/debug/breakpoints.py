"""T-CC-4.2: Conditional breakpoints and value watches for time-travel debugging.

This module extends the time-travel debugging system with:
- ConditionalBreakpoint: Stops execution when a condition becomes true
- ValueWatch: Tracks value changes across simulation ticks
- BinarySearch: Efficiently finds the exact tick where a value changed
- ExpressionEvaluator: Safe evaluation of Python expressions
- BreakpointManager: Manages breakpoints and watches

Example:
    from engine.debug.breakpoints import (
        BreakpointManager,
        ConditionalBreakpoint,
        ValueWatch,
    )

    manager = BreakpointManager(time_travel)

    # Stop when player health drops below 50
    bp = manager.add_breakpoint(
        condition="entity.health < 50",
        name="low_health_breakpoint",
    )

    # Watch player position across ticks
    watch = manager.add_watch(
        expression="entity.position.x",
        entity_id=player_id,
        name="player_x_position",
    )

    # Find when value changed
    change_tick = manager.binary_search_change(
        expression="entity.is_grounded",
        entity_id=player_id,
        start_tick=100,
        end_tick=200,
    )

Dependencies:
    - T-CC-4.1 (time_travel): TimeTravel, TickSnapshot, SnapshotRingBuffer
"""
from __future__ import annotations

import ast
import copy
import hashlib
import operator
import re
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)

if TYPE_CHECKING:
    from engine.debug.time_travel import TimeTravel, TickSnapshot


__all__ = [
    # Core types
    "ConditionalBreakpoint",
    "BreakpointState",
    "BreakpointHit",
    "BreakpointConfig",
    # Value watches
    "ValueWatch",
    "WatchState",
    "WatchRecord",
    "ValueChange",
    # Expression evaluation
    "ExpressionEvaluator",
    "ExpressionContext",
    "EvaluationResult",
    "ExpressionError",
    # Binary search
    "BinarySearchResult",
    "ChangeSearcher",
    # Manager
    "BreakpointManager",
    "ManagerConfig",
    "ManagerEvent",
    # Persistence
    "BreakpointSerializer",
]


# =============================================================================
# EXPRESSION EVALUATION
# =============================================================================


class ExpressionError(Exception):
    """Error during expression evaluation."""

    def __init__(self, message: str, expression: str, position: int = -1) -> None:
        super().__init__(message)
        self.expression = expression
        self.position = position


@dataclass(slots=True)
class ExpressionContext:
    """Context for expression evaluation.

    Provides variables and entities that expressions can reference.

    Attributes:
        variables: Named variables available to expressions.
        entity: The current entity being evaluated (if any).
        tick: Current simulation tick.
        state: Current snapshot state (if available).
    """
    variables: Dict[str, Any] = field(default_factory=dict)
    entity: Optional[Any] = None
    tick: int = 0
    state: Optional[Dict[str, Any]] = None

    def get(self, name: str) -> Any:
        """Get a value by name from the context."""
        # Check direct variables first
        if name in self.variables:
            return self.variables[name]

        # Check entity attributes
        if self.entity is not None and hasattr(self.entity, name):
            return getattr(self.entity, name)

        # Check state dict
        if self.state is not None and name in self.state:
            return self.state[name]

        raise KeyError(f"Name '{name}' not found in context")

    def has(self, name: str) -> bool:
        """Check if a name exists in the context."""
        try:
            self.get(name)
            return True
        except KeyError:
            return False


@dataclass(slots=True, frozen=True)
class EvaluationResult:
    """Result of evaluating an expression.

    Attributes:
        value: The evaluated value.
        success: Whether evaluation succeeded.
        error: Error message if evaluation failed.
        expression: The original expression.
        duration_us: Evaluation time in microseconds.
    """
    value: Any
    success: bool
    error: Optional[str] = None
    expression: str = ""
    duration_us: float = 0.0


class ExpressionEvaluator:
    """Safe evaluator for Python expressions.

    Evaluates expressions in a sandboxed environment without access to
    dangerous built-ins or operations. Supports attribute access, comparisons,
    arithmetic, and boolean operations.

    Example:
        evaluator = ExpressionEvaluator()

        context = ExpressionContext(
            variables={"x": 10, "y": 20},
            entity=player,
        )

        result = evaluator.evaluate("x + y > 25", context)
        if result.success:
            print(f"Result: {result.value}")  # True
    """

    # Allowed operators
    _BINARY_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.LShift: operator.lshift,
        ast.RShift: operator.rshift,
        ast.BitOr: operator.or_,
        ast.BitXor: operator.xor,
        ast.BitAnd: operator.and_,
    }

    _UNARY_OPS = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
        ast.Not: operator.not_,
        ast.Invert: operator.invert,
    }

    _COMPARE_OPS = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Is: operator.is_,
        ast.IsNot: operator.is_not,
        ast.In: lambda a, b: a in b,
        ast.NotIn: lambda a, b: a not in b,
    }

    # Allowed built-in functions
    _SAFE_BUILTINS = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "float": float,
        "int": int,
        "len": len,
        "max": max,
        "min": min,
        "pow": pow,
        "round": round,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "list": list,
        "set": set,
        "dict": dict,
        "sorted": sorted,
        "reversed": reversed,
        "enumerate": enumerate,
        "zip": zip,
        "range": range,
        "True": True,
        "False": False,
        "None": None,
    }

    __slots__ = ("_cache", "_max_cache_size", "_max_depth", "_max_iterations")

    def __init__(
        self,
        max_cache_size: int = 1000,
        max_depth: int = 50,
        max_iterations: int = 10000,
    ) -> None:
        """Initialize the expression evaluator.

        Args:
            max_cache_size: Maximum number of cached parsed expressions.
            max_depth: Maximum AST recursion depth.
            max_iterations: Maximum iterations for loops/comprehensions.
        """
        self._cache: Dict[str, ast.Expression] = {}
        self._max_cache_size = max_cache_size
        self._max_depth = max_depth
        self._max_iterations = max_iterations

    def evaluate(
        self,
        expression: str,
        context: ExpressionContext,
    ) -> EvaluationResult:
        """Evaluate an expression in the given context.

        Args:
            expression: The Python expression to evaluate.
            context: The context providing variables and entities.

        Returns:
            The evaluation result.
        """
        start_time = time.perf_counter()

        try:
            # Parse expression (with caching)
            parsed = self._parse(expression)

            # Evaluate the AST
            value = self._eval_node(parsed.body, context, depth=0)

            duration_us = (time.perf_counter() - start_time) * 1_000_000
            return EvaluationResult(
                value=value,
                success=True,
                expression=expression,
                duration_us=duration_us,
            )

        except ExpressionError:
            raise
        except Exception as e:
            duration_us = (time.perf_counter() - start_time) * 1_000_000
            return EvaluationResult(
                value=None,
                success=False,
                error=str(e),
                expression=expression,
                duration_us=duration_us,
            )

    def validate(self, expression: str) -> Tuple[bool, Optional[str]]:
        """Validate an expression without evaluating it.

        Args:
            expression: The expression to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        try:
            self._parse(expression)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except ExpressionError as e:
            return False, str(e)

    def get_referenced_names(self, expression: str) -> Set[str]:
        """Get all variable names referenced in an expression.

        Args:
            expression: The expression to analyze.

        Returns:
            Set of variable names.
        """
        try:
            parsed = self._parse(expression)
            names: Set[str] = set()
            self._collect_names(parsed.body, names)
            return names
        except Exception:
            return set()

    def clear_cache(self) -> None:
        """Clear the parsed expression cache."""
        self._cache.clear()

    def _parse(self, expression: str) -> ast.Expression:
        """Parse an expression string into an AST."""
        if expression in self._cache:
            return self._cache[expression]

        # Parse as expression
        try:
            parsed = ast.parse(expression, mode="eval")
        except SyntaxError as e:
            raise ExpressionError(
                f"Invalid expression syntax: {e}",
                expression,
                e.offset or -1,
            )

        # Validate AST nodes are allowed
        self._validate_ast(parsed)

        # Cache if room
        if len(self._cache) < self._max_cache_size:
            self._cache[expression] = parsed

        return parsed

    def _validate_ast(self, node: ast.AST) -> None:
        """Validate that AST contains only allowed nodes."""
        allowed_nodes = (
            ast.Expression,
            ast.BoolOp,
            ast.BinOp,
            ast.UnaryOp,
            ast.Lambda,
            ast.IfExp,
            ast.Dict,
            ast.Set,
            ast.ListComp,
            ast.SetComp,
            ast.DictComp,
            ast.GeneratorExp,
            ast.Compare,
            ast.Call,
            ast.Constant,
            ast.Attribute,
            ast.Subscript,
            ast.Name,
            ast.List,
            ast.Tuple,
            ast.Slice,
            ast.And,
            ast.Or,
            ast.Load,
            ast.Store,  # Needed for comprehension loop variables
            ast.comprehension,
            ast.arguments,
            ast.arg,
        )

        # Add operator nodes
        allowed_nodes += tuple(self._BINARY_OPS.keys())
        allowed_nodes += tuple(self._UNARY_OPS.keys())
        allowed_nodes += tuple(self._COMPARE_OPS.keys())

        for child in ast.walk(node):
            if not isinstance(child, allowed_nodes):
                raise ExpressionError(
                    f"Disallowed AST node: {type(child).__name__}",
                    "",
                )

    def _eval_node(
        self,
        node: ast.AST,
        context: ExpressionContext,
        depth: int,
    ) -> Any:
        """Recursively evaluate an AST node."""
        if depth > self._max_depth:
            raise ExpressionError("Maximum recursion depth exceeded", "")

        # Constants
        if isinstance(node, ast.Constant):
            return node.value

        # Names (variables)
        if isinstance(node, ast.Name):
            name = node.id
            if name in self._SAFE_BUILTINS:
                return self._SAFE_BUILTINS[name]
            return context.get(name)

        # Attribute access
        if isinstance(node, ast.Attribute):
            value = self._eval_node(node.value, context, depth + 1)
            return getattr(value, node.attr)

        # Subscript (indexing)
        if isinstance(node, ast.Subscript):
            value = self._eval_node(node.value, context, depth + 1)
            index = self._eval_node(node.slice, context, depth + 1)
            return value[index]

        # Slice
        if isinstance(node, ast.Slice):
            lower = self._eval_node(node.lower, context, depth + 1) if node.lower else None
            upper = self._eval_node(node.upper, context, depth + 1) if node.upper else None
            step = self._eval_node(node.step, context, depth + 1) if node.step else None
            return slice(lower, upper, step)

        # Binary operations
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, context, depth + 1)
            right = self._eval_node(node.right, context, depth + 1)
            op_func = self._BINARY_OPS.get(type(node.op))
            if op_func is None:
                raise ExpressionError(f"Unknown operator: {type(node.op).__name__}", "")
            return op_func(left, right)

        # Unary operations
        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand, context, depth + 1)
            op_func = self._UNARY_OPS.get(type(node.op))
            if op_func is None:
                raise ExpressionError(f"Unknown operator: {type(node.op).__name__}", "")
            return op_func(operand)

        # Boolean operations (and, or)
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                for value in node.values:
                    result = self._eval_node(value, context, depth + 1)
                    if not result:
                        return result
                return result
            elif isinstance(node.op, ast.Or):
                for value in node.values:
                    result = self._eval_node(value, context, depth + 1)
                    if result:
                        return result
                return result

        # Comparisons
        if isinstance(node, ast.Compare):
            left = self._eval_node(node.left, context, depth + 1)
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval_node(comparator, context, depth + 1)
                op_func = self._COMPARE_OPS.get(type(op))
                if op_func is None:
                    raise ExpressionError(f"Unknown comparison: {type(op).__name__}", "")
                if not op_func(left, right):
                    return False
                left = right
            return True

        # Conditional expression (ternary)
        if isinstance(node, ast.IfExp):
            test = self._eval_node(node.test, context, depth + 1)
            if test:
                return self._eval_node(node.body, context, depth + 1)
            else:
                return self._eval_node(node.orelse, context, depth + 1)

        # Function calls
        if isinstance(node, ast.Call):
            func = self._eval_node(node.func, context, depth + 1)

            # Validate callable is allowed
            if func not in self._SAFE_BUILTINS.values():
                # Check if it's a method on an object
                if isinstance(node.func, ast.Attribute):
                    pass  # Allow method calls on objects
                else:
                    raise ExpressionError(f"Function not allowed: {func}", "")

            args = [self._eval_node(arg, context, depth + 1) for arg in node.args]
            kwargs = {
                kw.arg: self._eval_node(kw.value, context, depth + 1)
                for kw in node.keywords
                if kw.arg is not None
            }
            return func(*args, **kwargs)

        # List literal
        if isinstance(node, ast.List):
            return [self._eval_node(elt, context, depth + 1) for elt in node.elts]

        # Tuple literal
        if isinstance(node, ast.Tuple):
            return tuple(self._eval_node(elt, context, depth + 1) for elt in node.elts)

        # Dict literal
        if isinstance(node, ast.Dict):
            return {
                self._eval_node(k, context, depth + 1): self._eval_node(v, context, depth + 1)
                for k, v in zip(node.keys, node.values)
                if k is not None
            }

        # Set literal
        if isinstance(node, ast.Set):
            return {self._eval_node(elt, context, depth + 1) for elt in node.elts}

        # List comprehension
        if isinstance(node, ast.ListComp):
            return self._eval_comprehension(node, context, depth, list)

        # Set comprehension
        if isinstance(node, ast.SetComp):
            return self._eval_comprehension(node, context, depth, set)

        # Dict comprehension
        if isinstance(node, ast.DictComp):
            return self._eval_dict_comprehension(node, context, depth)

        # Generator expression (convert to list)
        if isinstance(node, ast.GeneratorExp):
            return self._eval_comprehension(node, context, depth, list)

        raise ExpressionError(f"Cannot evaluate node: {type(node).__name__}", "")

    def _eval_comprehension(
        self,
        node: Union[ast.ListComp, ast.SetComp, ast.GeneratorExp],
        context: ExpressionContext,
        depth: int,
        container_type: type,
    ) -> Any:
        """Evaluate a list/set comprehension or generator expression."""
        results: List[Any] = []
        iterations = 0

        def iterate(generators: List[ast.comprehension], local_ctx: ExpressionContext) -> None:
            nonlocal iterations

            if not generators:
                if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
                    results.append(self._eval_node(node.elt, local_ctx, depth + 1))
                return

            gen = generators[0]
            iterable = self._eval_node(gen.iter, local_ctx, depth + 1)

            for item in iterable:
                iterations += 1
                if iterations > self._max_iterations:
                    raise ExpressionError("Maximum iterations exceeded in comprehension", "")

                # Create new context with loop variable
                new_ctx = ExpressionContext(
                    variables={**local_ctx.variables, gen.target.id: item}
                    if isinstance(gen.target, ast.Name)
                    else local_ctx.variables,
                    entity=local_ctx.entity,
                    tick=local_ctx.tick,
                    state=local_ctx.state,
                )

                # Check conditions
                if all(
                    self._eval_node(cond, new_ctx, depth + 1)
                    for cond in gen.ifs
                ):
                    iterate(generators[1:], new_ctx)

        iterate(list(node.generators), context)
        return container_type(results)

    def _eval_dict_comprehension(
        self,
        node: ast.DictComp,
        context: ExpressionContext,
        depth: int,
    ) -> Dict[Any, Any]:
        """Evaluate a dict comprehension."""
        results: Dict[Any, Any] = {}
        iterations = 0

        def iterate(generators: List[ast.comprehension], local_ctx: ExpressionContext) -> None:
            nonlocal iterations

            if not generators:
                key = self._eval_node(node.key, local_ctx, depth + 1)
                value = self._eval_node(node.value, local_ctx, depth + 1)
                results[key] = value
                return

            gen = generators[0]
            iterable = self._eval_node(gen.iter, local_ctx, depth + 1)

            for item in iterable:
                iterations += 1
                if iterations > self._max_iterations:
                    raise ExpressionError("Maximum iterations exceeded in comprehension", "")

                new_ctx = ExpressionContext(
                    variables={**local_ctx.variables, gen.target.id: item}
                    if isinstance(gen.target, ast.Name)
                    else local_ctx.variables,
                    entity=local_ctx.entity,
                    tick=local_ctx.tick,
                    state=local_ctx.state,
                )

                if all(
                    self._eval_node(cond, new_ctx, depth + 1)
                    for cond in gen.ifs
                ):
                    iterate(generators[1:], new_ctx)

        iterate(list(node.generators), context)
        return results

    def _collect_names(self, node: ast.AST, names: Set[str]) -> None:
        """Collect all Name nodes from an AST."""
        if isinstance(node, ast.Name):
            if node.id not in self._SAFE_BUILTINS:
                names.add(node.id)

        for child in ast.iter_child_nodes(node):
            self._collect_names(child, names)


# =============================================================================
# BREAKPOINT TYPES
# =============================================================================


class BreakpointState(IntEnum):
    """State of a breakpoint."""
    DISABLED = 0
    ENABLED = 1
    HIT = 2
    ERROR = 3


@dataclass(slots=True, frozen=True)
class BreakpointConfig:
    """Configuration for a breakpoint.

    Attributes:
        hit_count_threshold: Number of hits before breaking (0 = every hit).
        ignore_count: Number of times to ignore before first break.
        log_message: Optional message to log on hit instead of breaking.
        temporary: If True, breakpoint is deleted after first hit.
    """
    hit_count_threshold: int = 0
    ignore_count: int = 0
    log_message: Optional[str] = None
    temporary: bool = False


@dataclass(slots=True, frozen=True)
class BreakpointHit:
    """Information about a breakpoint being hit.

    Attributes:
        breakpoint_id: ID of the breakpoint that was hit.
        tick: The tick at which the breakpoint was hit.
        condition_value: The value of the condition expression.
        context: The evaluation context at hit time.
        hit_count: Total number of times this breakpoint has been hit.
    """
    breakpoint_id: str
    tick: int
    condition_value: Any
    context: ExpressionContext
    hit_count: int


@dataclass(slots=True)
class ConditionalBreakpoint:
    """A breakpoint that triggers when a condition becomes true.

    Breakpoints are evaluated each tick and trigger when their condition
    expression evaluates to True. They can be configured to require
    multiple hits, log instead of breaking, or auto-delete after first hit.

    Attributes:
        id: Unique identifier for the breakpoint.
        name: Human-readable name.
        condition: Python expression that triggers the breakpoint.
        entity_id: Optional entity ID to evaluate condition against.
        config: Breakpoint configuration.
        state: Current state of the breakpoint.
        hit_count: Number of times the breakpoint has been hit.
        last_hit_tick: Tick of the most recent hit.
        created_at: Timestamp when breakpoint was created.
    """
    id: str
    name: str
    condition: str
    entity_id: Optional[int] = None
    config: BreakpointConfig = field(default_factory=BreakpointConfig)
    state: BreakpointState = BreakpointState.ENABLED
    hit_count: int = 0
    last_hit_tick: int = -1
    created_at: float = field(default_factory=time.time)
    _previous_value: Optional[bool] = field(default=None, repr=False)

    @property
    def is_enabled(self) -> bool:
        """Check if breakpoint is enabled."""
        return self.state == BreakpointState.ENABLED

    @property
    def is_conditional_on_entity(self) -> bool:
        """Check if breakpoint requires a specific entity."""
        return self.entity_id is not None

    def enable(self) -> None:
        """Enable the breakpoint."""
        if self.state != BreakpointState.ERROR:
            self.state = BreakpointState.ENABLED

    def disable(self) -> None:
        """Disable the breakpoint."""
        self.state = BreakpointState.DISABLED

    def reset(self) -> None:
        """Reset hit count and state."""
        self.hit_count = 0
        self.last_hit_tick = -1
        self._previous_value = None
        if self.state == BreakpointState.HIT:
            self.state = BreakpointState.ENABLED

    def evaluate(
        self,
        evaluator: ExpressionEvaluator,
        context: ExpressionContext,
        tick: int,
    ) -> Optional[BreakpointHit]:
        """Evaluate the breakpoint condition.

        Args:
            evaluator: The expression evaluator.
            context: The evaluation context.
            tick: The current tick.

        Returns:
            BreakpointHit if condition triggers, None otherwise.
        """
        if self.state != BreakpointState.ENABLED:
            return None

        # Update context tick
        context.tick = tick

        # Evaluate condition
        result = evaluator.evaluate(self.condition, context)

        if not result.success:
            self.state = BreakpointState.ERROR
            return None

        current_value = bool(result.value)

        # Check if condition just became true (edge detection)
        should_trigger = False
        if current_value:
            # Trigger only on transition from False to True
            if self._previous_value is False or self._previous_value is None:
                should_trigger = True

        self._previous_value = current_value

        if not should_trigger:
            return None

        # Check ignore count
        if self.config.ignore_count > 0 and self.hit_count < self.config.ignore_count:
            self.hit_count += 1
            return None

        # Check hit count threshold
        if self.config.hit_count_threshold > 0:
            self.hit_count += 1
            if self.hit_count % self.config.hit_count_threshold != 0:
                return None
        else:
            self.hit_count += 1

        # Record hit
        self.last_hit_tick = tick

        # Handle temporary breakpoint - disable after first hit
        if self.config.temporary:
            self.state = BreakpointState.DISABLED
        # Check if this is a log-only breakpoint
        elif self.config.log_message is not None:
            # Don't change state, just log
            pass
        else:
            self.state = BreakpointState.HIT

        return BreakpointHit(
            breakpoint_id=self.id,
            tick=tick,
            condition_value=result.value,
            context=context,
            hit_count=self.hit_count,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "condition": self.condition,
            "entity_id": self.entity_id,
            "config": {
                "hit_count_threshold": self.config.hit_count_threshold,
                "ignore_count": self.config.ignore_count,
                "log_message": self.config.log_message,
                "temporary": self.config.temporary,
            },
            "state": self.state.value,
            "hit_count": self.hit_count,
            "last_hit_tick": self.last_hit_tick,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConditionalBreakpoint":
        """Deserialize from dictionary."""
        config_data = data.get("config", {})
        config = BreakpointConfig(
            hit_count_threshold=config_data.get("hit_count_threshold", 0),
            ignore_count=config_data.get("ignore_count", 0),
            log_message=config_data.get("log_message"),
            temporary=config_data.get("temporary", False),
        )
        return cls(
            id=data["id"],
            name=data["name"],
            condition=data["condition"],
            entity_id=data.get("entity_id"),
            config=config,
            state=BreakpointState(data.get("state", BreakpointState.ENABLED.value)),
            hit_count=data.get("hit_count", 0),
            last_hit_tick=data.get("last_hit_tick", -1),
            created_at=data.get("created_at", time.time()),
        )


# =============================================================================
# VALUE WATCHES
# =============================================================================


class WatchState(IntEnum):
    """State of a value watch."""
    DISABLED = 0
    ACTIVE = 1
    ERROR = 2


@dataclass(slots=True, frozen=True)
class WatchRecord:
    """A single recorded value from a watch.

    Attributes:
        tick: The tick when the value was recorded.
        value: The recorded value.
        timestamp: Real-world timestamp of recording.
    """
    tick: int
    value: Any
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True, frozen=True)
class ValueChange:
    """Record of a value change detected by a watch.

    Attributes:
        tick: The tick when the change was detected.
        old_value: The previous value.
        new_value: The new value.
        expression: The watched expression.
    """
    tick: int
    old_value: Any
    new_value: Any
    expression: str


@dataclass(slots=True)
class ValueWatch:
    """Watches an expression value across simulation ticks.

    Value watches track how an expression's value changes over time,
    recording the value at each tick and detecting changes.

    Attributes:
        id: Unique identifier for the watch.
        name: Human-readable name.
        expression: Python expression to evaluate.
        entity_id: Optional entity ID for entity-specific watches.
        state: Current state of the watch.
        max_history: Maximum number of records to keep.
        record_interval: Record every N ticks (1 = every tick).
        detect_changes_only: Only record when value changes.
    """
    id: str
    name: str
    expression: str
    entity_id: Optional[int] = None
    state: WatchState = WatchState.ACTIVE
    max_history: int = 1000
    record_interval: int = 1
    detect_changes_only: bool = False
    _history: List[WatchRecord] = field(default_factory=list, repr=False)
    _changes: List[ValueChange] = field(default_factory=list, repr=False)
    _last_recorded_tick: int = field(default=-1, repr=False)
    _last_value: Any = field(default=None, repr=False)
    _has_recorded: bool = field(default=False, repr=False)

    @property
    def is_active(self) -> bool:
        """Check if watch is active."""
        return self.state == WatchState.ACTIVE

    @property
    def history(self) -> List[WatchRecord]:
        """Get recorded history."""
        return self._history.copy()

    @property
    def changes(self) -> List[ValueChange]:
        """Get detected value changes."""
        return self._changes.copy()

    @property
    def current_value(self) -> Any:
        """Get the most recently recorded value."""
        return self._last_value

    @property
    def record_count(self) -> int:
        """Number of records in history."""
        return len(self._history)

    @property
    def change_count(self) -> int:
        """Number of detected changes."""
        return len(self._changes)

    def enable(self) -> None:
        """Enable the watch."""
        if self.state != WatchState.ERROR:
            self.state = WatchState.ACTIVE

    def disable(self) -> None:
        """Disable the watch."""
        self.state = WatchState.DISABLED

    def clear_history(self) -> None:
        """Clear recorded history."""
        self._history.clear()
        self._changes.clear()
        self._last_recorded_tick = -1
        self._last_value = None
        self._has_recorded = False

    def record(
        self,
        evaluator: ExpressionEvaluator,
        context: ExpressionContext,
        tick: int,
    ) -> Optional[ValueChange]:
        """Record the current value.

        Args:
            evaluator: The expression evaluator.
            context: The evaluation context.
            tick: The current tick.

        Returns:
            ValueChange if value changed, None otherwise.
        """
        if self.state != WatchState.ACTIVE:
            return None

        # Check recording interval
        if self._last_recorded_tick >= 0:
            if (tick - self._last_recorded_tick) < self.record_interval:
                return None

        # Update context tick
        context.tick = tick

        # Evaluate expression
        result = evaluator.evaluate(self.expression, context)

        if not result.success:
            self.state = WatchState.ERROR
            return None

        value = result.value
        change: Optional[ValueChange] = None

        # Check for value change
        if self._has_recorded:
            if value != self._last_value:
                change = ValueChange(
                    tick=tick,
                    old_value=self._last_value,
                    new_value=value,
                    expression=self.expression,
                )
                self._changes.append(change)

                # Trim changes if needed
                if len(self._changes) > self.max_history:
                    self._changes = self._changes[-self.max_history:]

        # Record if not in changes-only mode, or if there was a change
        should_record = (not self.detect_changes_only) or (change is not None)

        if should_record:
            record = WatchRecord(tick=tick, value=value)
            self._history.append(record)

            # Trim history if needed
            if len(self._history) > self.max_history:
                self._history = self._history[-self.max_history:]

            self._last_recorded_tick = tick

        self._last_value = value
        self._has_recorded = True

        return change

    def get_value_at_tick(self, tick: int) -> Optional[WatchRecord]:
        """Get the recorded value at a specific tick.

        Args:
            tick: The tick to look up.

        Returns:
            The record if found, None otherwise.
        """
        for record in self._history:
            if record.tick == tick:
                return record
        return None

    def get_value_range(
        self,
        start_tick: int,
        end_tick: int,
    ) -> List[WatchRecord]:
        """Get all records in a tick range.

        Args:
            start_tick: Start of range (inclusive).
            end_tick: End of range (inclusive).

        Returns:
            List of records in the range.
        """
        return [
            record
            for record in self._history
            if start_tick <= record.tick <= end_tick
        ]

    def get_changes_in_range(
        self,
        start_tick: int,
        end_tick: int,
    ) -> List[ValueChange]:
        """Get all value changes in a tick range.

        Args:
            start_tick: Start of range (inclusive).
            end_tick: End of range (inclusive).

        Returns:
            List of changes in the range.
        """
        return [
            change
            for change in self._changes
            if start_tick <= change.tick <= end_tick
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (without history)."""
        return {
            "id": self.id,
            "name": self.name,
            "expression": self.expression,
            "entity_id": self.entity_id,
            "state": self.state.value,
            "max_history": self.max_history,
            "record_interval": self.record_interval,
            "detect_changes_only": self.detect_changes_only,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValueWatch":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            expression=data["expression"],
            entity_id=data.get("entity_id"),
            state=WatchState(data.get("state", WatchState.ACTIVE.value)),
            max_history=data.get("max_history", 1000),
            record_interval=data.get("record_interval", 1),
            detect_changes_only=data.get("detect_changes_only", False),
        )


# =============================================================================
# BINARY SEARCH FOR VALUE CHANGES
# =============================================================================


@dataclass(slots=True, frozen=True)
class BinarySearchResult:
    """Result of a binary search for value change.

    Attributes:
        found: Whether a change point was found.
        tick: The tick where the value changed (or -1 if not found).
        old_value: The value before the change.
        new_value: The value after the change.
        ticks_searched: Number of ticks examined during search.
        snapshots_restored: Number of snapshots restored.
        duration_ms: Search duration in milliseconds.
    """
    found: bool
    tick: int
    old_value: Any
    new_value: Any
    ticks_searched: int
    snapshots_restored: int
    duration_ms: float


class ChangeSearcher:
    """Binary search implementation for finding when a value changed.

    Uses the time-travel snapshot system to efficiently find the exact
    tick where an expression's value changed from one value to another.

    Example:
        searcher = ChangeSearcher(time_travel)

        # Find when player became grounded
        result = searcher.find_change(
            expression="entity.is_grounded",
            target_value=True,
            start_tick=100,
            end_tick=500,
            context=context,
        )

        if result.found:
            print(f"Player became grounded at tick {result.tick}")
    """

    __slots__ = ("_time_travel", "_evaluator")

    def __init__(
        self,
        time_travel: "TimeTravel",
        evaluator: Optional[ExpressionEvaluator] = None,
    ) -> None:
        """Initialize the change searcher.

        Args:
            time_travel: The time travel system for seeking.
            evaluator: Optional expression evaluator (creates default if None).
        """
        self._time_travel = time_travel
        self._evaluator = evaluator or ExpressionEvaluator()

    def find_change(
        self,
        expression: str,
        start_tick: int,
        end_tick: int,
        context_factory: Callable[[int], ExpressionContext],
        target_value: Optional[Any] = None,
    ) -> BinarySearchResult:
        """Find the tick where an expression's value changed.

        Uses binary search to efficiently find the exact tick where
        the expression's value changed. If target_value is specified,
        searches for when the value became that specific value.

        Args:
            expression: The expression to evaluate.
            start_tick: Start of search range.
            end_tick: End of search range.
            context_factory: Function that creates context for a given tick.
            target_value: Optional specific value to search for.

        Returns:
            BinarySearchResult with the change point information.
        """
        start_time = time.perf_counter()
        ticks_searched = 0
        snapshots_restored = 0

        # Get value at start
        start_context = context_factory(start_tick)
        start_result = self._evaluator.evaluate(expression, start_context)
        if not start_result.success:
            return BinarySearchResult(
                found=False,
                tick=-1,
                old_value=None,
                new_value=None,
                ticks_searched=1,
                snapshots_restored=0,
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )
        start_value = start_result.value
        ticks_searched += 1

        # Get value at end
        end_context = context_factory(end_tick)
        end_result = self._evaluator.evaluate(expression, end_context)
        if not end_result.success:
            return BinarySearchResult(
                found=False,
                tick=-1,
                old_value=None,
                new_value=None,
                ticks_searched=2,
                snapshots_restored=0,
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )
        end_value = end_result.value
        ticks_searched += 1

        # Check if values are the same (no change in range)
        if start_value == end_value:
            return BinarySearchResult(
                found=False,
                tick=-1,
                old_value=start_value,
                new_value=end_value,
                ticks_searched=ticks_searched,
                snapshots_restored=0,
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Check target_value if specified
        if target_value is not None:
            if end_value != target_value:
                return BinarySearchResult(
                    found=False,
                    tick=-1,
                    old_value=start_value,
                    new_value=end_value,
                    ticks_searched=ticks_searched,
                    snapshots_restored=0,
                    duration_ms=(time.perf_counter() - start_time) * 1000,
                )

        # Binary search for the change point
        low = start_tick
        high = end_tick
        left_value = start_value
        right_value = end_value

        while low < high - 1:
            mid = (low + high) // 2

            # Seek to mid tick
            if self._time_travel.seek_to_tick(mid):
                snapshots_restored += 1

            mid_context = context_factory(mid)
            mid_result = self._evaluator.evaluate(expression, mid_context)
            ticks_searched += 1

            if not mid_result.success:
                # Error evaluating - can't continue binary search
                break

            mid_value = mid_result.value

            if mid_value == left_value:
                # Change is in the right half
                low = mid
                left_value = mid_value
            else:
                # Change is in the left half (or at mid)
                high = mid
                right_value = mid_value

        # The change happened at high tick
        duration_ms = (time.perf_counter() - start_time) * 1000

        return BinarySearchResult(
            found=True,
            tick=high,
            old_value=left_value,
            new_value=right_value,
            ticks_searched=ticks_searched,
            snapshots_restored=snapshots_restored,
            duration_ms=duration_ms,
        )

    def find_all_changes(
        self,
        expression: str,
        start_tick: int,
        end_tick: int,
        context_factory: Callable[[int], ExpressionContext],
        step: int = 1,
    ) -> List[ValueChange]:
        """Find all value changes in a tick range.

        This is a linear search that finds every tick where the value changed.
        Use find_change for more efficient single-change detection.

        Args:
            expression: The expression to evaluate.
            start_tick: Start of search range.
            end_tick: End of search range.
            context_factory: Function that creates context for a given tick.
            step: Tick step size (1 = check every tick).

        Returns:
            List of ValueChange objects for each change detected.
        """
        changes: List[ValueChange] = []
        last_value: Optional[Any] = None
        first = True

        for tick in range(start_tick, end_tick + 1, step):
            context = context_factory(tick)
            result = self._evaluator.evaluate(expression, context)

            if not result.success:
                continue

            value = result.value

            if not first and value != last_value:
                changes.append(ValueChange(
                    tick=tick,
                    old_value=last_value,
                    new_value=value,
                    expression=expression,
                ))

            last_value = value
            first = False

        return changes


# =============================================================================
# BREAKPOINT MANAGER
# =============================================================================


class ManagerEvent(Enum):
    """Events emitted by the breakpoint manager."""
    BREAKPOINT_ADDED = auto()
    BREAKPOINT_REMOVED = auto()
    BREAKPOINT_HIT = auto()
    BREAKPOINT_ERROR = auto()
    WATCH_ADDED = auto()
    WATCH_REMOVED = auto()
    WATCH_CHANGED = auto()
    WATCH_ERROR = auto()


@dataclass(slots=True, frozen=True)
class ManagerConfig:
    """Configuration for the breakpoint manager.

    Attributes:
        max_breakpoints: Maximum number of breakpoints.
        max_watches: Maximum number of watches.
        evaluate_watches_every_tick: Evaluate watches every tick.
        persist_on_change: Auto-persist when breakpoints/watches change.
    """
    max_breakpoints: int = 100
    max_watches: int = 100
    evaluate_watches_every_tick: bool = True
    persist_on_change: bool = False


class BreakpointManager:
    """Manages breakpoints and value watches for debugging.

    Provides a high-level API for adding, removing, and evaluating
    breakpoints and watches. Integrates with the time-travel system
    for seeking to breakpoint locations.

    Example:
        manager = BreakpointManager(time_travel)

        # Add a conditional breakpoint
        bp = manager.add_breakpoint(
            condition="player.health < 50",
            name="low_health",
        )

        # Add a value watch
        watch = manager.add_watch(
            expression="player.position.x",
            name="player_x",
        )

        # Each tick, evaluate breakpoints and watches
        manager.update(tick, get_context)

        # Check for hits
        for hit in manager.get_hits():
            print(f"Breakpoint {hit.breakpoint_id} hit at tick {hit.tick}")
    """

    __slots__ = (
        "_time_travel",
        "_evaluator",
        "_config",
        "_breakpoints",
        "_watches",
        "_hits",
        "_event_handlers",
        "_next_bp_id",
        "_next_watch_id",
    )

    def __init__(
        self,
        time_travel: "TimeTravel",
        config: Optional[ManagerConfig] = None,
        evaluator: Optional[ExpressionEvaluator] = None,
    ) -> None:
        """Initialize the breakpoint manager.

        Args:
            time_travel: The time travel system.
            config: Manager configuration.
            evaluator: Optional expression evaluator.
        """
        self._time_travel = time_travel
        self._config = config or ManagerConfig()
        self._evaluator = evaluator or ExpressionEvaluator()
        self._breakpoints: Dict[str, ConditionalBreakpoint] = {}
        self._watches: Dict[str, ValueWatch] = {}
        self._hits: List[BreakpointHit] = []
        self._event_handlers: List[Callable[[ManagerEvent, Any], None]] = []
        self._next_bp_id = 1
        self._next_watch_id = 1

    @property
    def breakpoint_count(self) -> int:
        """Number of registered breakpoints."""
        return len(self._breakpoints)

    @property
    def watch_count(self) -> int:
        """Number of registered watches."""
        return len(self._watches)

    @property
    def has_pending_hits(self) -> bool:
        """Check if there are unprocessed breakpoint hits."""
        return len(self._hits) > 0

    # -------------------------------------------------------------------------
    # Breakpoint Management
    # -------------------------------------------------------------------------

    def add_breakpoint(
        self,
        condition: str,
        name: Optional[str] = None,
        entity_id: Optional[int] = None,
        config: Optional[BreakpointConfig] = None,
    ) -> ConditionalBreakpoint:
        """Add a new conditional breakpoint.

        Args:
            condition: Python expression that triggers breakpoint.
            name: Optional human-readable name.
            entity_id: Optional entity ID for entity-specific conditions.
            config: Optional breakpoint configuration.

        Returns:
            The created breakpoint.

        Raises:
            ValueError: If max breakpoints reached or condition invalid.
        """
        if len(self._breakpoints) >= self._config.max_breakpoints:
            raise ValueError(
                f"Maximum breakpoints reached ({self._config.max_breakpoints})"
            )

        # Validate condition
        is_valid, error = self._evaluator.validate(condition)
        if not is_valid:
            raise ValueError(f"Invalid condition: {error}")

        bp_id = f"bp_{self._next_bp_id}"
        self._next_bp_id += 1

        if name is None:
            name = f"Breakpoint {self._next_bp_id - 1}"

        breakpoint = ConditionalBreakpoint(
            id=bp_id,
            name=name,
            condition=condition,
            entity_id=entity_id,
            config=config or BreakpointConfig(),
        )

        self._breakpoints[bp_id] = breakpoint
        self._emit_event(ManagerEvent.BREAKPOINT_ADDED, breakpoint)

        return breakpoint

    def remove_breakpoint(self, breakpoint_id: str) -> bool:
        """Remove a breakpoint by ID.

        Args:
            breakpoint_id: The breakpoint ID to remove.

        Returns:
            True if removed, False if not found.
        """
        breakpoint = self._breakpoints.pop(breakpoint_id, None)
        if breakpoint is not None:
            self._emit_event(ManagerEvent.BREAKPOINT_REMOVED, breakpoint)
            return True
        return False

    def get_breakpoint(self, breakpoint_id: str) -> Optional[ConditionalBreakpoint]:
        """Get a breakpoint by ID."""
        return self._breakpoints.get(breakpoint_id)

    def list_breakpoints(self) -> List[ConditionalBreakpoint]:
        """Get all registered breakpoints."""
        return list(self._breakpoints.values())

    def enable_breakpoint(self, breakpoint_id: str) -> bool:
        """Enable a breakpoint."""
        bp = self._breakpoints.get(breakpoint_id)
        if bp is not None:
            bp.enable()
            return True
        return False

    def disable_breakpoint(self, breakpoint_id: str) -> bool:
        """Disable a breakpoint."""
        bp = self._breakpoints.get(breakpoint_id)
        if bp is not None:
            bp.disable()
            return True
        return False

    def enable_all_breakpoints(self) -> None:
        """Enable all breakpoints."""
        for bp in self._breakpoints.values():
            bp.enable()

    def disable_all_breakpoints(self) -> None:
        """Disable all breakpoints."""
        for bp in self._breakpoints.values():
            bp.disable()

    def clear_breakpoints(self) -> None:
        """Remove all breakpoints."""
        self._breakpoints.clear()

    # -------------------------------------------------------------------------
    # Watch Management
    # -------------------------------------------------------------------------

    def add_watch(
        self,
        expression: str,
        name: Optional[str] = None,
        entity_id: Optional[int] = None,
        max_history: int = 1000,
        record_interval: int = 1,
        detect_changes_only: bool = False,
    ) -> ValueWatch:
        """Add a new value watch.

        Args:
            expression: Python expression to watch.
            name: Optional human-readable name.
            entity_id: Optional entity ID for entity-specific watches.
            max_history: Maximum records to keep.
            record_interval: Record every N ticks.
            detect_changes_only: Only record when value changes.

        Returns:
            The created watch.

        Raises:
            ValueError: If max watches reached or expression invalid.
        """
        if len(self._watches) >= self._config.max_watches:
            raise ValueError(
                f"Maximum watches reached ({self._config.max_watches})"
            )

        # Validate expression
        is_valid, error = self._evaluator.validate(expression)
        if not is_valid:
            raise ValueError(f"Invalid expression: {error}")

        watch_id = f"watch_{self._next_watch_id}"
        self._next_watch_id += 1

        if name is None:
            name = f"Watch {self._next_watch_id - 1}"

        watch = ValueWatch(
            id=watch_id,
            name=name,
            expression=expression,
            entity_id=entity_id,
            max_history=max_history,
            record_interval=record_interval,
            detect_changes_only=detect_changes_only,
        )

        self._watches[watch_id] = watch
        self._emit_event(ManagerEvent.WATCH_ADDED, watch)

        return watch

    def remove_watch(self, watch_id: str) -> bool:
        """Remove a watch by ID.

        Args:
            watch_id: The watch ID to remove.

        Returns:
            True if removed, False if not found.
        """
        watch = self._watches.pop(watch_id, None)
        if watch is not None:
            self._emit_event(ManagerEvent.WATCH_REMOVED, watch)
            return True
        return False

    def get_watch(self, watch_id: str) -> Optional[ValueWatch]:
        """Get a watch by ID."""
        return self._watches.get(watch_id)

    def list_watches(self) -> List[ValueWatch]:
        """Get all registered watches."""
        return list(self._watches.values())

    def enable_watch(self, watch_id: str) -> bool:
        """Enable a watch."""
        watch = self._watches.get(watch_id)
        if watch is not None:
            watch.enable()
            return True
        return False

    def disable_watch(self, watch_id: str) -> bool:
        """Disable a watch."""
        watch = self._watches.get(watch_id)
        if watch is not None:
            watch.disable()
            return True
        return False

    def clear_watches(self) -> None:
        """Remove all watches."""
        self._watches.clear()

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    def update(
        self,
        tick: int,
        context_factory: Callable[[Optional[int]], ExpressionContext],
    ) -> List[BreakpointHit]:
        """Evaluate all breakpoints and watches for the current tick.

        Args:
            tick: The current simulation tick.
            context_factory: Function that creates context, optionally for an entity_id.

        Returns:
            List of breakpoint hits that occurred.
        """
        hits: List[BreakpointHit] = []

        # Evaluate breakpoints
        for bp in self._breakpoints.values():
            if not bp.is_enabled:
                continue

            # Get context for this breakpoint
            context = context_factory(bp.entity_id)

            hit = bp.evaluate(self._evaluator, context, tick)
            if hit is not None:
                hits.append(hit)
                self._hits.append(hit)
                self._emit_event(ManagerEvent.BREAKPOINT_HIT, hit)

            if bp.state == BreakpointState.ERROR:
                self._emit_event(ManagerEvent.BREAKPOINT_ERROR, bp)

        # Evaluate watches
        if self._config.evaluate_watches_every_tick:
            for watch in self._watches.values():
                if not watch.is_active:
                    continue

                context = context_factory(watch.entity_id)
                change = watch.record(self._evaluator, context, tick)

                if change is not None:
                    self._emit_event(ManagerEvent.WATCH_CHANGED, change)

                if watch.state == WatchState.ERROR:
                    self._emit_event(ManagerEvent.WATCH_ERROR, watch)

        return hits

    def get_hits(self) -> List[BreakpointHit]:
        """Get all pending breakpoint hits."""
        return self._hits.copy()

    def pop_hit(self) -> Optional[BreakpointHit]:
        """Pop the next pending breakpoint hit."""
        if self._hits:
            return self._hits.pop(0)
        return None

    def clear_hits(self) -> None:
        """Clear all pending hits."""
        self._hits.clear()

    # -------------------------------------------------------------------------
    # Binary Search Integration
    # -------------------------------------------------------------------------

    def binary_search_change(
        self,
        expression: str,
        start_tick: int,
        end_tick: int,
        context_factory: Callable[[int], ExpressionContext],
        target_value: Optional[Any] = None,
    ) -> BinarySearchResult:
        """Binary search for when an expression's value changed.

        Args:
            expression: The expression to search.
            start_tick: Start of search range.
            end_tick: End of search range.
            context_factory: Function that creates context for a tick.
            target_value: Optional specific value to search for.

        Returns:
            BinarySearchResult with change information.
        """
        searcher = ChangeSearcher(self._time_travel, self._evaluator)
        return searcher.find_change(
            expression=expression,
            start_tick=start_tick,
            end_tick=end_tick,
            context_factory=context_factory,
            target_value=target_value,
        )

    def seek_to_breakpoint(self, breakpoint_id: str) -> bool:
        """Seek to the tick where a breakpoint was last hit.

        Args:
            breakpoint_id: The breakpoint ID.

        Returns:
            True if seek succeeded, False otherwise.
        """
        bp = self._breakpoints.get(breakpoint_id)
        if bp is None or bp.last_hit_tick < 0:
            return False

        return self._time_travel.seek_to_tick(bp.last_hit_tick)

    # -------------------------------------------------------------------------
    # Events
    # -------------------------------------------------------------------------

    def on_event(self, handler: Callable[[ManagerEvent, Any], None]) -> None:
        """Register an event handler.

        Args:
            handler: Callback receiving (event, data).
        """
        self._event_handlers.append(handler)

    def remove_event_handler(
        self,
        handler: Callable[[ManagerEvent, Any], None],
    ) -> None:
        """Remove an event handler."""
        try:
            self._event_handlers.remove(handler)
        except ValueError:
            pass

    def _emit_event(self, event: ManagerEvent, data: Any) -> None:
        """Emit an event to all handlers."""
        for handler in self._event_handlers:
            try:
                handler(event, data)
            except Exception:
                pass  # Don't let handler errors break manager

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize manager state to dictionary."""
        return {
            "breakpoints": [bp.to_dict() for bp in self._breakpoints.values()],
            "watches": [w.to_dict() for w in self._watches.values()],
            "next_bp_id": self._next_bp_id,
            "next_watch_id": self._next_watch_id,
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Restore manager state from dictionary."""
        self._breakpoints.clear()
        self._watches.clear()

        for bp_data in data.get("breakpoints", []):
            bp = ConditionalBreakpoint.from_dict(bp_data)
            self._breakpoints[bp.id] = bp

        for watch_data in data.get("watches", []):
            watch = ValueWatch.from_dict(watch_data)
            self._watches[watch.id] = watch

        self._next_bp_id = data.get("next_bp_id", 1)
        self._next_watch_id = data.get("next_watch_id", 1)

    def clear_all(self) -> None:
        """Clear all breakpoints, watches, and hits."""
        self._breakpoints.clear()
        self._watches.clear()
        self._hits.clear()


# =============================================================================
# SERIALIZATION
# =============================================================================


class BreakpointSerializer:
    """Handles persistence of breakpoints and watches to files.

    Example:
        serializer = BreakpointSerializer()

        # Save to file
        serializer.save(manager, "/path/to/breakpoints.json")

        # Load from file
        serializer.load(manager, "/path/to/breakpoints.json")
    """

    @staticmethod
    def save(manager: BreakpointManager, path: str) -> None:
        """Save manager state to a JSON file.

        Args:
            manager: The breakpoint manager.
            path: File path to save to.
        """
        import json
        data = manager.to_dict()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(manager: BreakpointManager, path: str) -> None:
        """Load manager state from a JSON file.

        Args:
            manager: The breakpoint manager.
            path: File path to load from.
        """
        import json
        with open(path, "r") as f:
            data = json.load(f)
        manager.from_dict(data)

    @staticmethod
    def export_breakpoints(
        breakpoints: List[ConditionalBreakpoint],
    ) -> str:
        """Export breakpoints to JSON string."""
        import json
        return json.dumps(
            [bp.to_dict() for bp in breakpoints],
            indent=2,
        )

    @staticmethod
    def import_breakpoints(
        json_str: str,
    ) -> List[ConditionalBreakpoint]:
        """Import breakpoints from JSON string."""
        import json
        data = json.loads(json_str)
        return [ConditionalBreakpoint.from_dict(bp_data) for bp_data in data]

    @staticmethod
    def export_watches(watches: List[ValueWatch]) -> str:
        """Export watches to JSON string."""
        import json
        return json.dumps(
            [w.to_dict() for w in watches],
            indent=2,
        )

    @staticmethod
    def import_watches(json_str: str) -> List[ValueWatch]:
        """Import watches from JSON string."""
        import json
        data = json.loads(json_str)
        return [ValueWatch.from_dict(w_data) for w_data in data]
