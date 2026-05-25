"""
ShellLang REPL - Interactive shell with feedback system.

Provides:
    Feedback     Echo system for operation feedback
    Shell        Interactive REPL with namespace setup
    echo         Module-level feedback singleton
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from foundation.shelllang.core import World

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_PROMPT = ">>> "
DEFAULT_HISTORY_COUNT = 10
EXIT_COMMANDS = frozenset({"exit", "quit", "q"})
HELP_COMMANDS = frozenset({"help", "?", "h"})


# =============================================================================
# FEEDBACK SYSTEM
# =============================================================================


class Feedback:
    """
    Simple feedback system that can be enabled/disabled.

    Provides immediate feedback on mutations for human users.
    """

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._history: list[str] = []
        self._callback: Optional[Callable[[str], None]] = None

    def __call__(self, message: str) -> None:
        """Echo a message if feedback is enabled."""
        if self._enabled:
            self._history.append(message)
            if self._callback:
                self._callback(message)
            else:
                print(f"  → {message}")

    def enable(self) -> None:
        """Enable feedback."""
        self._enabled = True

    def disable(self) -> None:
        """Disable feedback."""
        self._enabled = False

    def set_callback(self, callback: Optional[Callable[[str], None]]) -> None:
        """Set a custom callback for feedback messages."""
        self._callback = callback

    def history(self, n: int = DEFAULT_HISTORY_COUNT) -> list[str]:
        """Get the last n feedback messages."""
        return self._history[-n:]

    def clear_history(self) -> None:
        """Clear feedback history."""
        self._history.clear()


# Module-level feedback instance
echo = Feedback()


# =============================================================================
# SHELL
# =============================================================================


class Shell:
    """
    Interactive REPL for ShellLang.

    Sets up a namespace with all sugar functions and runs an interactive loop.
    """

    def __init__(
        self,
        world: "World",
        registry: Dict[str, Type],
        feedback: Optional[Feedback] = None,
    ) -> None:
        self._world = world
        self._registry = registry
        self._feedback = feedback or echo
        self._namespace: Dict[str, Any] = {}
        self._running = False

        # Set up the sugar module with our world
        self._setup_sugar()
        self._setup_namespace()

    def _setup_sugar(self) -> None:
        """Configure the sugar module with our world and registry."""
        from foundation.shelllang import sugar

        sugar.set_world(self._world)
        sugar.set_echo(self._feedback)
        sugar.set_registry(self._registry)

    def _setup_namespace(self) -> None:
        """Build the execution namespace."""
        from foundation.shelllang.sugar import (
            EntityProxy,
            QueryResult,
            TimeManager,
            TypeQuery,
        )
        from foundation.shelllang.ai import AIInterface

        # Create instances
        time_manager = TimeManager()
        ai = AIInterface(self._world, self._registry)

        # Base namespace
        self._namespace = {
            # World operations
            "world": self._world,
            "create": self._world.create,
            "destroy": self._world.destroy,
            "query": self._world.query,
            "snap": self._world.snap,
            "restore": self._world.restore,
            # Time operations
            "mark": time_manager.mark,
            "rewind": time_manager.rewind,
            "undo": time_manager.undo,
            "redo": time_manager.redo,
            "history": time_manager.history,
            # Sugar classes
            "EntityProxy": EntityProxy,
            "QueryResult": QueryResult,
            "TypeQuery": TypeQuery,
            # AI interface
            "ai": ai,
            # Feedback
            "echo": self._feedback,
        }

        # Add all registered component types to namespace
        for name, cls in self._registry.items():
            self._namespace[name] = cls
            # Also add TypeQuery for each component
            self._namespace[f"{name}s"] = TypeQuery(cls)

    def execute(self, code: str) -> Any:
        """
        Execute a line of code in the shell namespace.

        Args:
            code: Python code to execute.

        Returns:
            The result of the expression, or None for statements.
        """
        code = code.strip()

        if not code:
            return None

        if code in EXIT_COMMANDS:
            self._running = False
            return "Goodbye!"

        if code in HELP_COMMANDS:
            return self._help()

        try:
            # Try as expression first
            result = eval(code, self._namespace)
            return result
        except SyntaxError:
            # Try as statement
            exec(code, self._namespace)
            return None

    def run(self, prompt: str = DEFAULT_PROMPT) -> None:
        """
        Run the interactive REPL loop.

        Args:
            prompt: The prompt string to display.
        """
        self._running = True
        print("ShellLang REPL. Type 'help' for commands, 'quit' to exit.")

        while self._running:
            try:
                line = input(prompt)
                result = self.execute(line)
                if result is not None:
                    print(result)
            except KeyboardInterrupt:
                print("\nInterrupted. Type 'quit' to exit.")
            except EOFError:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

    def exit(self) -> None:
        """Exit the REPL."""
        self._running = False

    def _help(self) -> str:
        """Return help text."""
        return """ShellLang Commands:

Entity Operations:
    e = create()           Create a new entity
    destroy(e)             Destroy an entity
    query(Component)       Find entities with component

Component Access (via EntityProxy):
    e.health.current       Get field value
    e.health.current = 50  Set field value (tracked)

Queries (via TypeQuery):
    Enemys.all             All entities with Enemy component
    Enemys.where(fn)       Filter by predicate
    Enemys.near(e, dist)   Filter by distance

Bulk Operations:
    result.set(health__current=100)
    result.destroy()
    result.each(fn)

Time Travel:
    mark("name")           Create named snapshot
    rewind("name")         Restore to snapshot
    undo()                 Undo last change
    redo()                 Redo undone change
    history()              Show recent changes

AI Interface:
    ai.execute({"op": "query", ...})
    ai.validate({"op": "set", ...})
    ai.dry_run({"op": "spawn", ...})

Type 'quit' or 'exit' to leave."""

    @property
    def namespace(self) -> Dict[str, Any]:
        """Get the current namespace (for testing/debugging)."""
        return self._namespace


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "Feedback",
    "Shell",
    "echo",
    # Constants
    "DEFAULT_PROMPT",
    "DEFAULT_HISTORY_COUNT",
    "EXIT_COMMANDS",
    "HELP_COMMANDS",
]
