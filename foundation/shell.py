"""
Shell - Live code execution system. Core Foundation Layer 3.
Enables interactive exploration, AI collaboration, and runtime manipulation.
"""
from __future__ import annotations
import io
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from foundation.mirror import mirror, ObjectMirror, ClassMirror, FieldInfo
from foundation.serializer import to_dict, from_dict, to_file, from_file, deep_copy
from foundation.registry import registry
from foundation.tracker import tracker
from foundation.inspector import inspector


@dataclass
class ExecutionResult:
    """Result of executing code in the shell."""
    success: bool
    value: Any = None
    output: str = ""
    error: str | None = None
    error_type: str | None = None


class Shell:
    """
    Interactive Python shell with pre-populated namespace.

    Provides:
        - Code execution (expressions return values, statements execute silently)
        - Pre-populated namespace with all core systems
        - Context binding (set 'self' to an object)
        - History management with optional persistence
    """
    __slots__ = ("_namespace", "_history", "_bound_object")

    def __init__(self) -> None:
        self._namespace: dict[str, Any] = {}
        self._history: list[str] = []
        self._bound_object: Any = None
        self.reset_namespace()

    # --- Execution ---

    def execute(self, code: str) -> ExecutionResult:
        """
        Execute Python code in the shell namespace.

        - Expressions return their value
        - Statements execute silently (return None)
        - Errors are caught and returned (non-fatal)

        Args:
            code: Python code to execute

        Returns:
            ExecutionResult with success status, value, output, and any error info
        """
        self._history.append(code)

        # Capture stdout
        old_stdout = sys.stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            # Try as expression first (returns value)
            try:
                result = eval(code, self._namespace)
                output = captured_output.getvalue()
                self._update_result_vars(result)
                return ExecutionResult(success=True, value=result, output=output)
            except SyntaxError:
                # Not an expression, try as statement
                exec(code, self._namespace)
                output = captured_output.getvalue()
                return ExecutionResult(success=True, value=None, output=output)
        except Exception as e:
            output = captured_output.getvalue()
            return ExecutionResult(
                success=False,
                value=None,
                output=output,
                error=str(e),
                error_type=type(e).__name__
            )
        finally:
            sys.stdout = old_stdout

    def _update_result_vars(self, result: Any) -> None:
        """Update _, __, ___ with recent results."""
        if result is not None:
            self._namespace["___"] = self._namespace.get("__")
            self._namespace["__"] = self._namespace.get("_")
            self._namespace["_"] = result

    # --- Namespace ---

    @property
    def namespace(self) -> dict[str, Any]:
        """Get the execution namespace."""
        return self._namespace

    def reset_namespace(self) -> None:
        """Reset namespace to default state with core systems and utilities."""
        self._namespace.clear()

        # Core systems
        self._namespace.update({
            "mirror": mirror,
            "registry": registry,
            "tracker": tracker,
            "serializer": sys.modules.get("foundation.serializer"),
            "inspector": inspector,
            "shell": self,
        })

        # Convenience functions
        self._namespace.update({
            "inspect": lambda obj: inspector.inspect(obj),
            "save": lambda obj, path: to_file(obj, path),
            "load": lambda path: from_file(path),
            "copy": lambda obj: deep_copy(obj),
            "types": lambda: registry.all_types(),
            "instances": lambda cls: list(registry.instances(cls)),
            "dirty": lambda: tracker.all_dirty(),
            "undo": lambda: tracker.undo(),
            "redo": lambda: tracker.redo(),
        })

        # Result variables
        self._namespace.update({"_": None, "__": None, "___": None})

        # Restore bound object if any
        if self._bound_object is not None:
            self._namespace["self"] = self._bound_object

    # --- Context Binding ---

    def bind(self, obj: Any) -> None:
        """
        Bind an object as 'self' in the namespace.

        Useful for exploring a specific instance interactively.

        Args:
            obj: The object to bind as 'self'
        """
        self._bound_object = obj
        self._namespace["self"] = obj

    def unbind(self) -> None:
        """Remove the 'self' binding from the namespace."""
        self._bound_object = None
        self._namespace.pop("self", None)

    @property
    def bound_object(self) -> Any | None:
        """Get the currently bound object, or None if not bound."""
        return self._bound_object

    # --- History ---

    @property
    def history(self) -> list[str]:
        """Get all executed inputs."""
        return self._history.copy()

    def clear_history(self) -> None:
        """Clear the execution history."""
        self._history.clear()

    def save_history(self, path: str | Path) -> None:
        """Save history to a file."""
        Path(path).write_text("\n".join(self._history))

    def load_history(self, path: str | Path) -> None:
        """Load history from a file, appending to current history."""
        p = Path(path)
        if p.exists():
            lines = p.read_text().strip().split("\n")
            self._history.extend(line for line in lines if line)


# Module-level singleton
shell = Shell()


def inspect(obj: Any) -> Any:
    """Convenience function to inspect an object."""
    return inspector.inspect(obj)


__all__ = [
    "ExecutionResult",
    "Shell",
    "shell",
    "inspect",
]
