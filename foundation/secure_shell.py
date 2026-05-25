"""
SecureShell - Shell with capability enforcement.
Part of Core Foundation Layer 3.

Extends the base Shell class to enforce capability-based security
restrictions on code execution and entity manipulation.
"""
from __future__ import annotations
from typing import Any, Optional

from foundation.shell import Shell, ExecutionResult
from foundation.capabilities import (
    Capability,
    CapabilitySet,
    CapabilityError,
    SecureContext,
    CAPS_READONLY,
)


class SecureShell(Shell):
    """
    Shell that enforces capability restrictions.

    Extends the base Shell with capability-based security. All operations
    check for required capabilities before proceeding.

    The shell maintains its own capability set and executes all code
    within a SecureContext using those capabilities.

    Examples:
        >>> caps = CapabilitySet(Capability.READ | Capability.EXECUTE)
        >>> shell = SecureShell(caps)
        >>> shell.execute("1 + 1")  # Works (EXECUTE capability)
        ExecutionResult(success=True, value=2, ...)

        >>> readonly_shell = SecureShell(CapabilitySet(Capability.READ))
        >>> readonly_shell.execute("x = 1")  # Fails (no EXECUTE capability)
        ExecutionResult(success=False, error="EXECUTE capability required", ...)
    """
    __slots__ = ("_capabilities",)

    def __init__(self, capabilities: Optional[CapabilitySet] = None) -> None:
        """
        Create a SecureShell with the specified capabilities.

        Args:
            capabilities: The capabilities for this shell. Defaults to READONLY.
        """
        super().__init__()
        self._capabilities = capabilities if capabilities is not None else CAPS_READONLY

    @property
    def capabilities(self) -> CapabilitySet:
        """Get the capability set for this shell."""
        return self._capabilities

    def execute(self, code: str) -> ExecutionResult:
        """
        Execute code within a capability context.

        Requires EXECUTE capability. All code runs with the shell's
        capability set active.

        Args:
            code: Python code to execute

        Returns:
            ExecutionResult with success status and any output/errors
        """
        # Check EXECUTE capability
        if not self._capabilities.has(Capability.EXECUTE):
            return ExecutionResult(
                success=False,
                value=None,
                output="",
                error="EXECUTE capability required",
                error_type="CapabilityError"
            )

        # Execute within secure context
        with SecureContext(self._capabilities):
            return super().execute(code)

    def create_entity(self, entity_type: type, **kwargs: Any) -> Any:
        """
        Create a new entity (requires CREATE capability).

        Args:
            entity_type: The type of entity to create
            **kwargs: Arguments to pass to the entity constructor

        Returns:
            The created entity

        Raises:
            CapabilityError: If CREATE capability is not present
        """
        if not self._capabilities.has(Capability.CREATE):
            raise CapabilityError(
                "CREATE capability required",
                required=Capability.CREATE,
                available=self._capabilities
            )

        with SecureContext(self._capabilities):
            return entity_type(**kwargs)

    def delete_entity(self, entity: Any) -> None:
        """
        Delete an entity (requires DELETE capability).

        Note: This method validates the DELETE capability but actual
        deletion logic must be implemented by the entity management
        system (e.g., World, EntityManager). This ensures the capability
        check happens at the shell boundary.

        Args:
            entity: The entity to delete

        Raises:
            CapabilityError: If DELETE capability is not present
            NotImplementedError: Always (deletion requires entity manager)
        """
        if not self._capabilities.has(Capability.DELETE):
            raise CapabilityError(
                "DELETE capability required",
                required=Capability.DELETE,
                available=self._capabilities
            )

        # Capability validated - actual deletion requires entity management system
        raise NotImplementedError(
            "Entity deletion requires an entity manager. "
            "Use world.delete(entity) or entity_manager.remove(entity) instead."
        )

    def read_field(self, obj: Any, field: str) -> Any:
        """
        Read a field from an object (requires READ capability).

        Args:
            obj: The object to read from
            field: The field name to read

        Returns:
            The field value

        Raises:
            CapabilityError: If READ capability is not present
        """
        if not self._capabilities.has(Capability.READ):
            raise CapabilityError(
                "READ capability required",
                required=Capability.READ,
                available=self._capabilities
            )

        with SecureContext(self._capabilities):
            return getattr(obj, field)

    def write_field(self, obj: Any, field: str, value: Any) -> None:
        """
        Write a field on an object (requires WRITE capability).

        Args:
            obj: The object to modify
            field: The field name to write
            value: The value to set

        Raises:
            CapabilityError: If WRITE capability is not present
        """
        if not self._capabilities.has(Capability.WRITE):
            raise CapabilityError(
                "WRITE capability required",
                required=Capability.WRITE,
                available=self._capabilities
            )

        with SecureContext(self._capabilities):
            setattr(obj, field, value)

    def with_capabilities(self, capabilities: CapabilitySet) -> SecureShell:
        """
        Create a new SecureShell with different capabilities.

        The new shell shares history with this shell but has
        different capabilities.

        Args:
            capabilities: The capabilities for the new shell

        Returns:
            A new SecureShell with the specified capabilities
        """
        new_shell = SecureShell(capabilities)
        # Share relevant state
        new_shell._history = self._history
        new_shell._bound_object = self._bound_object
        if self._bound_object is not None:
            new_shell._namespace["self"] = self._bound_object
        return new_shell

    def restrict_to(self, capabilities: Capability) -> SecureShell:
        """
        Create a new SecureShell with restricted capabilities.

        The new shell can only have capabilities that both the
        original shell has AND the specified capabilities.

        Args:
            capabilities: The maximum capabilities for the new shell

        Returns:
            A new SecureShell with restricted capabilities
        """
        restricted = CapabilitySet(
            self._capabilities.capabilities & capabilities
        )
        return self.with_capabilities(restricted)


# Factory functions for common secure shell configurations

def create_readonly_shell() -> SecureShell:
    """Create a shell with only READ capability."""
    return SecureShell(CapabilitySet(Capability.READ))


def create_sandbox_shell() -> SecureShell:
    """Create a shell with READ, WRITE, CREATE, and EXECUTE but no external access."""
    return SecureShell(CapabilitySet(
        Capability.READ | Capability.WRITE | Capability.CREATE | Capability.EXECUTE
    ))


def create_full_shell() -> SecureShell:
    """Create a shell with all capabilities (use with caution)."""
    from foundation.capabilities import CAPS_FULL
    return SecureShell(CAPS_FULL)


__all__ = [
    "SecureShell",
    "create_readonly_shell",
    "create_sandbox_shell",
    "create_full_shell",
]
