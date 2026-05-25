"""
ShellLang AI Interface - Structured commands for AI agents.

Provides:
    AIInterface     Structured JSON command execution
        execute()   Execute a command
        validate()  Validate a command without executing
        dry_run()   Preview effects without executing

Command format:
    {
        "op": "query" | "set" | "spawn" | "destroy" | "snap" | "restore" | "inspect" | "schema",
        ...operation-specific fields...
    }
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Type, TYPE_CHECKING

from foundation.shelllang.core import Entity

if TYPE_CHECKING:
    from foundation.shelllang.core import World

# =============================================================================
# CONSTANTS
# =============================================================================

VALID_OPERATIONS = frozenset({
    "query",
    "set",
    "spawn",
    "destroy",
    "snap",
    "restore",
    "inspect",
    "schema",
    "list_types",
    "count",
})

DEFAULT_QUERY_LIMIT = 100


# =============================================================================
# AI INTERFACE
# =============================================================================


class AIInterface:
    """
    Structured command interface for AI agents.

    AI agents send JSON-like dict commands and receive structured responses.
    All operations can be validated before execution and previewed with dry_run.
    """

    def __init__(self, world: "World", registry: Dict[str, Type]) -> None:
        self._world = world
        self._registry = registry

    # =========================================================================
    # MAIN ENTRY POINTS
    # =========================================================================

    def execute(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a structured command and return the result.

        Args:
            command: Dict with 'op' key and operation-specific fields.

        Returns:
            Dict with operation results or error information.
        """
        op = command.get("op")

        if op == "query":
            return self._query(command)
        elif op == "set":
            return self._set(command)
        elif op == "spawn":
            return self._spawn(command)
        elif op == "destroy":
            return self._destroy(command)
        elif op == "snap":
            return self._snap(command)
        elif op == "restore":
            return self._restore(command)
        elif op == "inspect":
            return self._inspect(command)
        elif op == "schema":
            return self._schema(command)
        elif op == "list_types":
            return self._list_types(command)
        elif op == "count":
            return self._count(command)
        else:
            return {"error": f"Unknown operation: {op}"}

    def validate(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a command without executing it.

        Args:
            command: Dict with 'op' key and operation-specific fields.

        Returns:
            Dict with 'valid' bool and optional 'error' message.
        """
        op = command.get("op")

        if not op:
            return {"valid": False, "error": "Missing 'op' field"}

        if op not in VALID_OPERATIONS:
            return {"valid": False, "error": f"Unknown operation: {op}"}

        # Operation-specific validation
        if op == "set":
            required = ("entity", "component", "field", "value")
            missing = [k for k in required if k not in command]
            if missing:
                return {"valid": False, "error": f"set requires: {', '.join(required)}. Missing: {missing}"}

            # Check component type exists
            comp_name = command["component"]
            if comp_name not in self._registry and comp_name.title() not in self._registry:
                return {"valid": False, "error": f"Unknown component type: {comp_name}"}

            # Check entity exists
            entity_id = command["entity"]
            if not self._world.exists(Entity(entity_id)):
                return {"valid": False, "error": f"Entity {entity_id} does not exist"}

        elif op == "spawn":
            comp_name = command.get("component")
            if comp_name and comp_name not in self._registry and comp_name.title() not in self._registry:
                return {"valid": False, "error": f"Unknown component type: {comp_name}"}

        elif op == "destroy":
            if "entity" not in command:
                return {"valid": False, "error": "destroy requires 'entity' field"}

        elif op == "schema":
            type_name = command.get("type")
            if not type_name:
                return {"valid": False, "error": "schema requires 'type' field"}
            if type_name not in self._registry and type_name.title() not in self._registry:
                return {"valid": False, "error": f"Unknown type: {type_name}"}

        elif op == "restore":
            if "snapshot" not in command:
                return {"valid": False, "error": "restore requires 'snapshot' field"}

        return {"valid": True}

    def dry_run(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preview effects of a command without executing it.

        Args:
            command: Dict with 'op' key and operation-specific fields.

        Returns:
            Dict describing what would happen if the command were executed.
        """
        validation = self.validate(command)
        if not validation["valid"]:
            return validation

        op = command.get("op")

        if op == "query":
            result = self._query(command)
            return {"would_return": result}

        elif op == "set":
            entity_id = command["entity"]
            comp_name = command["component"]
            field_name = command["field"]
            new_value = command["value"]

            C = self._registry.get(comp_name) or self._registry.get(comp_name.title())
            e = Entity(entity_id)
            component = self._world.get(e, C)

            if component is None:
                return {"error": f"Entity {entity_id} does not have component {comp_name}"}

            old_value = getattr(component, field_name, None)
            return {
                "would_change": {
                    "entity": entity_id,
                    "field": f"{comp_name}.{field_name}",
                    "from": old_value,
                    "to": new_value,
                }
            }

        elif op == "spawn":
            return {
                "would_create": {
                    "component": command.get("component"),
                    "fields": command.get("fields", {}),
                }
            }

        elif op == "destroy":
            entity_id = command["entity"]
            entity_info = self._entity_to_dict(entity_id)
            return {"would_destroy": entity_info}

        elif op == "inspect":
            return self._inspect(command)

        elif op == "schema":
            return self._schema(command)

        return {"preview": "not available for this operation"}

    # =========================================================================
    # OPERATION IMPLEMENTATIONS
    # =========================================================================

    def _query(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a query command."""

        components = cmd.get("components", [])
        where = cmd.get("where", {})
        limit = cmd.get("limit", DEFAULT_QUERY_LIMIT)

        # Get component types
        Cs = []
        for name in components:
            C = self._registry.get(name) or self._registry.get(name.title())
            if C:
                Cs.append(C)

        # Execute query
        if Cs:
            entities = self._world.query(*Cs)
        else:
            entities = list(self._world.entities)

        # Apply where filters
        for field_path, condition in where.items():
            filtered = []
            for e in entities:
                value = self._get_field_value(e, field_path)
                if value is not None and self._check_condition(value, condition):
                    filtered.append(e)
            entities = filtered

        # Apply limit
        entities = entities[:limit]

        return {
            "entities": [self._entity_to_dict(e.id) for e in entities],
            "count": len(entities),
        }

    def _set(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a set command."""

        entity_id = cmd["entity"]
        comp_name = cmd["component"]
        field_name = cmd["field"]
        new_value = cmd["value"]

        C = self._registry.get(comp_name) or self._registry.get(comp_name.title())
        e = Entity(entity_id)

        component = self._world.get(e, C)
        if component is None:
            return {"error": f"Entity {entity_id} does not have component {comp_name}"}

        old_value = getattr(component, field_name, None)
        self._world.set(e, C, field_name, new_value)

        return {
            "entity": entity_id,
            "component": comp_name,
            "field": field_name,
            "old": old_value,
            "new": new_value,
        }

    def _spawn(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a spawn command."""
        comp_name = cmd.get("component")
        fields = cmd.get("fields", {})

        # Create entity
        e = self._world.create()

        # Attach component if specified
        if comp_name:
            C = self._registry.get(comp_name) or self._registry.get(comp_name.title())
            if C:
                try:
                    component = C(**fields)
                except TypeError:
                    # Try creating without args then setting fields
                    component = C()
                    for name, value in fields.items():
                        setattr(component, name, value)
                self._world.attach(e, component)

        return {
            "entity": e.id,
            "component": comp_name,
            "fields": fields,
        }

    def _destroy(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a destroy command."""

        entity_id = cmd["entity"]
        e = Entity(entity_id)

        if not self._world.exists(e):
            return {"error": f"Entity {entity_id} does not exist"}

        self._world.destroy(e)
        return {"destroyed": entity_id}

    def _snap(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a snapshot command."""
        name = cmd.get("name")
        snapshot = self._world.snap(name)
        return {
            "snapshot": {
                "name": snapshot.name,
                "entity_count": len(snapshot.entities),
            }
        }

    def _restore(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a restore command."""
        # This would need snapshot storage - simplified version
        return {"error": "restore requires snapshot storage (not implemented in basic AI interface)"}

    def _inspect(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an inspect command."""

        entity_id = cmd.get("entity")
        if entity_id is None:
            return {"error": "inspect requires 'entity' field"}

        e = Entity(entity_id)
        if not self._world.exists(e):
            return {"error": f"Entity {entity_id} does not exist"}

        components = {}
        for C in self._world.components_of(e):
            comp = self._world.get(e, C)
            if comp:
                components[C.__name__] = self._component_to_dict(comp)

        return {
            "entity": entity_id,
            "components": components,
        }

    def _schema(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a schema command."""
        type_name = cmd.get("type")
        T = self._registry.get(type_name) or self._registry.get(type_name.title())

        if not T:
            return {"error": f"Unknown type: {type_name}"}

        # Extract schema from annotations
        annotations = getattr(T, "__annotations__", {})
        fields = {}

        for name, typ in annotations.items():
            if name.startswith("_"):
                continue

            field_info = {
                "type": getattr(typ, "__name__", str(typ)),
            }

            # Get default value if present
            default = getattr(T, name, None)
            if default is not None and not callable(default):
                field_info["default"] = default

            fields[name] = field_info

        return {
            "name": type_name,
            "fields": fields,
        }

    def _list_types(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """List all registered component types."""
        return {
            "types": list(self._registry.keys()),
            "count": len(self._registry),
        }

    def _count(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """Count entities matching criteria."""
        result = self._query(cmd)
        return {"count": result["count"]}

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _entity_to_dict(self, entity_id: int) -> Dict[str, Any]:
        """Convert an entity to a dict representation."""

        e = Entity(entity_id)
        if not self._world.exists(e):
            return {"id": entity_id, "exists": False}

        components = [C.__name__ for C in self._world.components_of(e)]
        return {
            "id": entity_id,
            "components": components,
        }

    def _component_to_dict(self, component: Any) -> Dict[str, Any]:
        """Convert a component to a dict representation."""
        result = {}
        for name in dir(component):
            if name.startswith("_"):
                continue
            try:
                value = getattr(component, name)
                if callable(value):
                    continue
                # Try to make it JSON-serializable
                if isinstance(value, (int, float, str, bool, type(None))):
                    result[name] = value
                elif isinstance(value, (list, tuple)):
                    result[name] = list(value)
                elif isinstance(value, dict):
                    result[name] = dict(value)
                else:
                    result[name] = str(value)
            except (AttributeError, TypeError):
                pass
        return result

    def _get_field_value(self, e: "Entity", field_path: str) -> Any:
        """Get a field value from an entity using dot notation."""
        parts = field_path.split(".")

        if len(parts) == 1:
            # Just component name - return the component itself
            comp_name = parts[0]
            C = self._registry.get(comp_name) or self._registry.get(comp_name.title())
            if C:
                return self._world.get(e, C)
            return None

        # component.field format
        comp_name = parts[0]
        field_name = parts[1]

        C = self._registry.get(comp_name) or self._registry.get(comp_name.title())
        if not C:
            return None

        component = self._world.get(e, C)
        if component is None:
            return None

        return getattr(component, field_name, None)

    def _check_condition(self, value: Any, condition: Dict[str, Any]) -> bool:
        """Check if a value matches a condition."""
        for op, target in condition.items():
            if op == "==" or op == "eq":
                if value != target:
                    return False
            elif op == "!=" or op == "ne":
                if value == target:
                    return False
            elif op == "<" or op == "lt":
                if not (value < target):
                    return False
            elif op == "<=" or op == "le":
                if not (value <= target):
                    return False
            elif op == ">" or op == "gt":
                if not (value > target):
                    return False
            elif op == ">=" or op == "ge":
                if not (value >= target):
                    return False
            elif op == "in":
                if value not in target:
                    return False
            elif op == "not_in":
                if value in target:
                    return False
        return True


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "AIInterface",
    "VALID_OPERATIONS",
    "DEFAULT_QUERY_LIMIT",
]
