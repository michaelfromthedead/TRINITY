"""Command Aliases - Map short names to command sequences.

This module provides command aliasing with:
- Simple alias registration
- Alias expansion on execute
- Multiple command support (via semicolon)
- Argument substitution

Example:
    >>> registry = AliasRegistry()
    >>> registry.register("noclip", "god; fly")
    >>> registry.expand("noclip")
    'god; fly'
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import ALIAS as ALIAS_CONFIG


class AliasError(Exception):
    """Base exception for alias errors."""
    pass


class AliasRecursionError(AliasError):
    """Raised when alias expansion would cause infinite recursion."""
    pass


@dataclass
class Alias:
    """A command alias.

    Attributes:
        name: The alias name.
        expansion: The command(s) to expand to.
        description: Optional description.
    """
    name: str
    expansion: str
    description: str = ""


class AliasLimitError(AliasError):
    """Raised when alias limit is reached."""
    pass


class AliasRegistry:
    """Registry for command aliases.

    Aliases allow mapping short names to longer command sequences.
    Aliases can expand to multiple commands separated by semicolons.

    Example:
        >>> registry = AliasRegistry()
        >>> registry.register("god_mode", "god; infinite_ammo; infinite_health")
        >>> registry.expand("god_mode")
        'god; infinite_ammo; infinite_health'
    """

    def __init__(self, max_aliases: Optional[int] = None, max_recursion_depth: Optional[int] = None) -> None:
        """Initialize the alias registry.

        Args:
            max_aliases: Maximum number of aliases allowed (default from config).
            max_recursion_depth: Maximum recursion depth for alias expansion (default from config).
        """
        self._aliases: Dict[str, Alias] = {}
        self._max_aliases = max_aliases if max_aliases is not None else ALIAS_CONFIG.MAX_ALIASES
        self._max_recursion_depth = max_recursion_depth if max_recursion_depth is not None else ALIAS_CONFIG.MAX_RECURSION_DEPTH

    def register(
        self,
        name: str,
        expansion: str,
        description: str = "",
    ) -> Alias:
        """Register a new alias.

        Args:
            name: The alias name (cannot contain spaces).
            expansion: The command(s) to expand to.
            description: Optional description.

        Returns:
            The registered Alias object.

        Raises:
            ValueError: If the name is invalid or already exists.
        """
        # Validate name
        if not name:
            raise ValueError("Alias name cannot be empty")
        if " " in name:
            raise ValueError("Alias name cannot contain spaces")
        if name in self._aliases:
            raise ValueError(f"Alias '{name}' already exists")
        if len(self._aliases) >= self._max_aliases:
            raise AliasLimitError(
                f"Maximum number of aliases ({self._max_aliases}) reached"
            )

        alias = Alias(name=name, expansion=expansion, description=description)
        self._aliases[name] = alias
        return alias

    def unregister(self, name: str) -> bool:
        """Unregister an alias.

        Args:
            name: The alias name to remove.

        Returns:
            True if the alias was found and removed, False otherwise.
        """
        if name in self._aliases:
            del self._aliases[name]
            return True
        return False

    def get(self, name: str) -> Optional[Alias]:
        """Get an alias by name.

        Args:
            name: The alias name.

        Returns:
            The Alias if found, None otherwise.
        """
        return self._aliases.get(name)

    def update(self, name: str, expansion: str) -> bool:
        """Update an existing alias.

        Args:
            name: The alias name to update.
            expansion: The new expansion.

        Returns:
            True if the alias was found and updated, False otherwise.
        """
        if name in self._aliases:
            self._aliases[name].expansion = expansion
            return True
        return False

    def expand(
        self,
        input_text: str,
        depth: int = 0,
    ) -> str:
        """Expand aliases in the input text.

        Recursively expands aliases up to MAX_RECURSION_DEPTH.

        Args:
            input_text: The input text to expand.
            depth: Current recursion depth (internal use).

        Returns:
            The expanded input text.

        Raises:
            AliasRecursionError: If recursion limit is exceeded.
        """
        if depth >= self._max_recursion_depth:
            raise AliasRecursionError(
                f"Alias recursion depth exceeded ({self._max_recursion_depth})"
            )

        # Split input into commands
        commands = input_text.split(";")
        expanded_commands = []

        for command in commands:
            command = command.strip()
            if not command:
                continue

            # Split command into name and arguments
            parts = command.split(None, 1)
            name = parts[0]
            args = parts[1] if len(parts) > 1 else ""

            # Check if it's an alias
            alias = self.get(name)
            if alias:
                # Substitute arguments
                expansion = self._substitute_args(alias.expansion, args)
                # Recursively expand
                expansion = self.expand(expansion, depth + 1)
                expanded_commands.append(expansion)
            else:
                expanded_commands.append(command)

        return "; ".join(expanded_commands)

    def _substitute_args(self, expansion: str, args: str) -> str:
        """Substitute argument placeholders in expansion.

        Supports placeholders:
        - $* - All arguments
        - $1, $2, ... - Individual arguments
        - $@ - Same as $*

        Args:
            expansion: The expansion string with placeholders.
            args: The argument string.

        Returns:
            The expansion with placeholders substituted.
        """
        # Parse arguments
        arg_list = args.split() if args else []

        # Substitute $* and $@
        result = expansion.replace("$*", args).replace("$@", args)

        # Substitute numbered arguments
        for i, arg in enumerate(arg_list, 1):
            result = result.replace(f"${i}", arg)

        # Remove unused placeholders (for missing arguments)
        result = re.sub(r'\$\d+', '', result)

        return result.strip()

    def all(self) -> List[Alias]:
        """Get all registered aliases.

        Returns:
            List of all aliases, sorted by name.
        """
        return sorted(self._aliases.values(), key=lambda a: a.name)

    def find(self, pattern: str) -> List[Alias]:
        """Find aliases matching a pattern.

        Args:
            pattern: A glob pattern to match.

        Returns:
            List of matching aliases.
        """
        import fnmatch
        return [
            alias for name, alias in self._aliases.items()
            if fnmatch.fnmatch(name, pattern)
        ]

    def clear(self) -> None:
        """Remove all aliases."""
        self._aliases.clear()

    def export(self) -> Dict[str, str]:
        """Export all aliases as a dictionary.

        Returns:
            Dictionary mapping alias names to expansions.
        """
        return {name: alias.expansion for name, alias in self._aliases.items()}

    def import_aliases(self, aliases: Dict[str, str]) -> int:
        """Import aliases from a dictionary.

        Args:
            aliases: Dictionary mapping names to expansions.

        Returns:
            Number of aliases successfully imported.
        """
        count = 0
        for name, expansion in aliases.items():
            try:
                if name in self._aliases:
                    self.update(name, expansion)
                else:
                    self.register(name, expansion)
                count += 1
            except ValueError:
                pass
        return count

    def __len__(self) -> int:
        """Return the number of registered aliases."""
        return len(self._aliases)

    def __contains__(self, name: str) -> bool:
        """Check if an alias is registered."""
        return name in self._aliases

    def __iter__(self):
        """Iterate over alias names."""
        return iter(self._aliases)
