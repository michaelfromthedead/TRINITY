"""
Authority validation system for network security.

This module provides authority-based access control for networked game entities,
ensuring that only authorized callers can modify entity state, spawn entities,
or destroy them.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional, Set
import threading


class Authority(Enum):
    """Authority levels for network operations."""
    SERVER = auto()  # Full authority - can do anything
    CLIENT = auto()  # Limited authority - needs validation
    OWNER = auto()   # Ownership authority - can modify owned entities


class AuthorityError(Exception):
    """Exception raised when an authority violation occurs."""

    def __init__(
        self,
        message: str,
        caller_authority: Authority,
        required_authority: Authority,
        operation: str,
        entity_id: Optional[str] = None
    ):
        """
        Initialize an authority error.

        Args:
            message: Human-readable error description
            caller_authority: The authority level of the caller
            required_authority: The authority level required for the operation
            operation: The operation that was attempted
            entity_id: Optional entity identifier involved
        """
        super().__init__(message)
        self.caller_authority = caller_authority
        self.required_authority = required_authority
        self.operation = operation
        self.entity_id = entity_id


@dataclass
class FieldAuthority:
    """Defines authority requirements for a specific field."""
    write_authority: Authority = Authority.SERVER
    read_authority: Authority = Authority.CLIENT
    owner_can_write: bool = False


@dataclass
class EntityAuthority:
    """Defines authority requirements for an entity type."""
    spawn_authority: Authority = Authority.SERVER
    destroy_authority: Authority = Authority.SERVER
    owner_can_destroy: bool = False
    field_authorities: Dict[str, FieldAuthority] = field(default_factory=dict)
    default_field_authority: FieldAuthority = field(default_factory=FieldAuthority)


@dataclass
class Caller:
    """Represents a caller attempting an operation."""
    id: str
    authority: Authority
    owned_entities: Set[str] = field(default_factory=set)

    def owns(self, entity_id: str) -> bool:
        """Check if this caller owns the given entity."""
        return entity_id in self.owned_entities


@dataclass
class Entity:
    """Represents a networked entity."""
    id: str
    entity_type: str
    owner_id: Optional[str] = None
    fields: Dict[str, Any] = field(default_factory=dict)


class AuthorityValidator:
    """
    Validates authority for network operations.

    This class enforces access control rules for entity operations including
    field writes, entity spawning, and entity destruction.

    Thread-safe: Uses internal locking for concurrent access.
    """

    def __init__(self):
        """Initialize the authority validator."""
        self._entity_authorities: Dict[str, EntityAuthority] = {}
        self._default_authority = EntityAuthority()
        self._lock = threading.RLock()
        self._custom_validators: Dict[str, Callable[[Entity, str, Caller], bool]] = {}

    def register_entity_type(
        self,
        entity_type: str,
        authority: EntityAuthority
    ) -> None:
        """
        Register authority requirements for an entity type.

        Args:
            entity_type: The type name of the entity
            authority: The authority configuration for this entity type
        """
        with self._lock:
            self._entity_authorities[entity_type] = authority

    def register_custom_validator(
        self,
        operation: str,
        validator: Callable[[Entity, str, Caller], bool]
    ) -> None:
        """
        Register a custom validation function for an operation.

        Args:
            operation: The operation name (e.g., "write", "spawn", "destroy")
            validator: A function that returns True if the operation is allowed
        """
        with self._lock:
            self._custom_validators[operation] = validator

    def _get_entity_authority(self, entity_type: str) -> EntityAuthority:
        """Get authority configuration for an entity type."""
        return self._entity_authorities.get(entity_type, self._default_authority)

    def is_server(self, caller: Caller) -> bool:
        """
        Check if the caller has server authority.

        Args:
            caller: The caller to check

        Returns:
            True if the caller has server authority
        """
        return caller.authority == Authority.SERVER

    def is_owner(self, entity: Entity, caller: Caller) -> bool:
        """
        Check if the caller owns the given entity.

        Args:
            entity: The entity to check ownership of
            caller: The caller to verify

        Returns:
            True if the caller owns the entity
        """
        if entity.owner_id is None:
            return False
        return entity.owner_id == caller.id or caller.owns(entity.id)

    def validate_write(
        self,
        entity: Entity,
        field_name: str,
        caller: Caller,
        raise_on_failure: bool = False
    ) -> bool:
        """
        Validate if a caller can write to an entity field.

        Args:
            entity: The entity being modified
            field_name: The name of the field being written
            caller: The caller attempting the write
            raise_on_failure: If True, raise AuthorityError on failure

        Returns:
            True if the write is authorized

        Raises:
            AuthorityError: If raise_on_failure is True and validation fails
        """
        with self._lock:
            # Server can always write
            if self.is_server(caller):
                return True

            # Check custom validator
            if "write" in self._custom_validators:
                if self._custom_validators["write"](entity, field_name, caller):
                    return True

            # Get authority configuration
            entity_auth = self._get_entity_authority(entity.entity_type)
            field_auth = entity_auth.field_authorities.get(
                field_name,
                entity_auth.default_field_authority
            )

            # Check if owner can write
            if field_auth.owner_can_write and self.is_owner(entity, caller):
                return True

            # Check authority level
            if caller.authority == field_auth.write_authority:
                return True

            # Validation failed
            if raise_on_failure:
                raise AuthorityError(
                    f"Caller '{caller.id}' with authority {caller.authority.name} "
                    f"cannot write to field '{field_name}' on entity '{entity.id}' "
                    f"(requires {field_auth.write_authority.name})",
                    caller_authority=caller.authority,
                    required_authority=field_auth.write_authority,
                    operation="write",
                    entity_id=entity.id
                )

            return False

    def validate_spawn(
        self,
        entity_type: str,
        caller: Caller,
        raise_on_failure: bool = False
    ) -> bool:
        """
        Validate if a caller can spawn an entity of the given type.

        Args:
            entity_type: The type of entity to spawn
            caller: The caller attempting to spawn
            raise_on_failure: If True, raise AuthorityError on failure

        Returns:
            True if the spawn is authorized

        Raises:
            AuthorityError: If raise_on_failure is True and validation fails
        """
        with self._lock:
            # Server can always spawn
            if self.is_server(caller):
                return True

            # Check custom validator
            if "spawn" in self._custom_validators:
                # For spawn, we don't have an entity yet, pass None
                dummy_entity = Entity(id="", entity_type=entity_type)
                if self._custom_validators["spawn"](dummy_entity, "", caller):
                    return True

            # Get authority configuration
            entity_auth = self._get_entity_authority(entity_type)

            # Check authority level
            if caller.authority == entity_auth.spawn_authority:
                return True

            # Validation failed
            if raise_on_failure:
                raise AuthorityError(
                    f"Caller '{caller.id}' with authority {caller.authority.name} "
                    f"cannot spawn entity type '{entity_type}' "
                    f"(requires {entity_auth.spawn_authority.name})",
                    caller_authority=caller.authority,
                    required_authority=entity_auth.spawn_authority,
                    operation="spawn",
                    entity_id=None
                )

            return False

    def validate_destroy(
        self,
        entity: Entity,
        caller: Caller,
        raise_on_failure: bool = False
    ) -> bool:
        """
        Validate if a caller can destroy an entity.

        Args:
            entity: The entity to destroy
            caller: The caller attempting destruction
            raise_on_failure: If True, raise AuthorityError on failure

        Returns:
            True if the destruction is authorized

        Raises:
            AuthorityError: If raise_on_failure is True and validation fails
        """
        with self._lock:
            # Server can always destroy
            if self.is_server(caller):
                return True

            # Check custom validator
            if "destroy" in self._custom_validators:
                if self._custom_validators["destroy"](entity, "", caller):
                    return True

            # Get authority configuration
            entity_auth = self._get_entity_authority(entity.entity_type)

            # Check if owner can destroy
            if entity_auth.owner_can_destroy and self.is_owner(entity, caller):
                return True

            # Check authority level
            if caller.authority == entity_auth.destroy_authority:
                return True

            # Validation failed
            if raise_on_failure:
                raise AuthorityError(
                    f"Caller '{caller.id}' with authority {caller.authority.name} "
                    f"cannot destroy entity '{entity.id}' "
                    f"(requires {entity_auth.destroy_authority.name})",
                    caller_authority=caller.authority,
                    required_authority=entity_auth.destroy_authority,
                    operation="destroy",
                    entity_id=entity.id
                )

            return False

    def validate_batch_writes(
        self,
        entity: Entity,
        field_names: Set[str],
        caller: Caller
    ) -> Dict[str, bool]:
        """
        Validate multiple field writes at once.

        Args:
            entity: The entity being modified
            field_names: Set of field names being written
            caller: The caller attempting the writes

        Returns:
            Dictionary mapping field names to validation results
        """
        results = {}
        for field_name in field_names:
            results[field_name] = self.validate_write(entity, field_name, caller)
        return results

    def get_writable_fields(
        self,
        entity: Entity,
        caller: Caller
    ) -> Set[str]:
        """
        Get all fields that a caller can write to on an entity.

        Args:
            entity: The entity to check
            caller: The caller to check permissions for

        Returns:
            Set of field names the caller can write to
        """
        with self._lock:
            writable = set()

            # Server can write everything
            if self.is_server(caller):
                entity_auth = self._get_entity_authority(entity.entity_type)
                return set(entity_auth.field_authorities.keys()) | set(entity.fields.keys())

            # Check each known field
            entity_auth = self._get_entity_authority(entity.entity_type)
            for field_name in entity_auth.field_authorities.keys():
                if self.validate_write(entity, field_name, caller):
                    writable.add(field_name)

            return writable
