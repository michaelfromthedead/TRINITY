"""Material instances - Material instances with parameter overrides."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
import uuid
import copy

from .material_parameters import (
    MaterialParameter, ParameterCollection, ParameterType,
    ScalarParameter, VectorParameter, ColorParameter, TextureParameter,
    BooleanParameter, IntegerParameter
)


class InstanceState(Enum):
    """State of material instance."""
    VALID = auto()
    DIRTY = auto()
    INVALID = auto()


@dataclass
class ParameterOverride:
    """Override for a single parameter."""
    parameter_name: str
    value: Any
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter_name": self.parameter_name,
            "value": self.value,
            "enabled": self.enabled
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParameterOverride':
        return cls(
            parameter_name=data["parameter_name"],
            value=data["value"],
            enabled=data.get("enabled", True)
        )


class MaterialDefinition:
    """
    Base material definition that instances are created from.

    Contains the default parameters and shader information.
    """

    def __init__(self, name: str, shader_path: str = ""):
        self._id = str(uuid.uuid4())
        self._name = name
        self._shader_path = shader_path
        self._parameters = ParameterCollection()
        self._version = 1
        self._instances: Set[str] = set()  # Track instance IDs

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def shader_path(self) -> str:
        return self._shader_path

    @shader_path.setter
    def shader_path(self, value: str) -> None:
        self._shader_path = value
        self._version += 1

    @property
    def parameters(self) -> ParameterCollection:
        return self._parameters

    @property
    def version(self) -> int:
        return self._version

    @property
    def instance_count(self) -> int:
        return len(self._instances)

    def add_parameter(self, param: MaterialParameter) -> None:
        """Add a parameter to the definition."""
        self._parameters.add(param)
        self._version += 1

    def remove_parameter(self, name: str) -> Optional[MaterialParameter]:
        """Remove a parameter from the definition."""
        param = self._parameters.remove(name)
        if param:
            self._version += 1
        return param

    def get_parameter(self, name: str) -> Optional[MaterialParameter]:
        """Get a parameter by name."""
        return self._parameters.get(name)

    def register_instance(self, instance_id: str) -> None:
        """Register an instance of this material."""
        self._instances.add(instance_id)

    def unregister_instance(self, instance_id: str) -> None:
        """Unregister an instance."""
        self._instances.discard(instance_id)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self._id,
            "name": self._name,
            "shader_path": self._shader_path,
            "parameters": self._parameters.to_dict(),
            "version": self._version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MaterialDefinition':
        """Deserialize from dictionary."""
        definition = cls(data["name"], data.get("shader_path", ""))
        definition._id = data.get("id", str(uuid.uuid4()))
        definition._parameters = ParameterCollection.from_dict(data.get("parameters", {}))
        definition._version = data.get("version", 1)
        return definition


class MaterialInstance:
    """
    Instance of a material with parameter overrides.

    Instances share the shader and default parameters from their
    parent definition but can override individual parameter values.
    """

    def __init__(self, name: str, parent: MaterialDefinition):
        self._id = str(uuid.uuid4())
        self._name = name
        self._parent = parent
        self._overrides: Dict[str, ParameterOverride] = {}
        self._state = InstanceState.VALID
        self._parent_version = parent.version
        self._tags: List[str] = []
        self._metadata: Dict[str, Any] = {}

        # Callbacks
        self._on_changed: List[Callable[[], None]] = []

        # Register with parent
        parent.register_instance(self._id)

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value
        self._notify_changed()

    @property
    def parent(self) -> MaterialDefinition:
        return self._parent

    @property
    def state(self) -> InstanceState:
        # Check if parent has changed
        if self._parent_version != self._parent.version:
            self._state = InstanceState.DIRTY
        return self._state

    @property
    def tags(self) -> List[str]:
        return self._tags.copy()

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata.copy()

    @property
    def override_count(self) -> int:
        return len([o for o in self._overrides.values() if o.enabled])

    def set_override(self, parameter_name: str, value: Any) -> bool:
        """
        Set an override for a parameter.

        Args:
            parameter_name: Name of parameter to override
            value: Override value

        Returns:
            True if override was set successfully
        """
        param = self._parent.get_parameter(parameter_name)
        if param is None:
            return False

        self._overrides[parameter_name] = ParameterOverride(
            parameter_name=parameter_name,
            value=value,
            enabled=True
        )
        self._notify_changed()
        return True

    def clear_override(self, parameter_name: str) -> bool:
        """Clear an override for a parameter."""
        if parameter_name in self._overrides:
            del self._overrides[parameter_name]
            self._notify_changed()
            return True
        return False

    def clear_all_overrides(self) -> None:
        """Clear all parameter overrides."""
        self._overrides.clear()
        self._notify_changed()

    def has_override(self, parameter_name: str) -> bool:
        """Check if parameter has an override."""
        override = self._overrides.get(parameter_name)
        return override is not None and override.enabled

    def get_override(self, parameter_name: str) -> Optional[ParameterOverride]:
        """Get override for a parameter."""
        return self._overrides.get(parameter_name)

    def get_effective_value(self, parameter_name: str) -> Any:
        """
        Get the effective value for a parameter.

        Returns override value if set, otherwise default from parent.
        """
        override = self._overrides.get(parameter_name)
        if override and override.enabled:
            return override.value

        param = self._parent.get_parameter(parameter_name)
        if param:
            return param.value
        return None

    def get_all_effective_values(self) -> Dict[str, Any]:
        """Get effective values for all parameters."""
        values = {}
        for param in self._parent.parameters:
            values[param.name] = self.get_effective_value(param.name)
        return values

    def get_override_names(self) -> List[str]:
        """Get names of all overridden parameters."""
        return [name for name, o in self._overrides.items() if o.enabled]

    def enable_override(self, parameter_name: str) -> bool:
        """Enable an override (if it exists)."""
        override = self._overrides.get(parameter_name)
        if override:
            override.enabled = True
            self._notify_changed()
            return True
        return False

    def disable_override(self, parameter_name: str) -> bool:
        """Disable an override without removing it."""
        override = self._overrides.get(parameter_name)
        if override:
            override.enabled = False
            self._notify_changed()
            return True
        return False

    def refresh(self) -> None:
        """Refresh instance after parent changes."""
        self._parent_version = self._parent.version

        # Remove overrides for parameters that no longer exist
        invalid_overrides = []
        for name in self._overrides:
            if self._parent.get_parameter(name) is None:
                invalid_overrides.append(name)

        for name in invalid_overrides:
            del self._overrides[name]

        self._state = InstanceState.VALID
        self._notify_changed()

    def add_tag(self, tag: str) -> None:
        """Add a tag to the instance."""
        if tag not in self._tags:
            self._tags.append(tag)

    def remove_tag(self, tag: str) -> bool:
        """Remove a tag from the instance."""
        if tag in self._tags:
            self._tags.remove(tag)
            return True
        return False

    def has_tag(self, tag: str) -> bool:
        """Check if instance has a tag."""
        return tag in self._tags

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value."""
        self._metadata[key] = value

    def get_metadata(self, key: str) -> Any:
        """Get metadata value."""
        return self._metadata.get(key)

    def on_changed(self, callback: Callable[[], None]) -> None:
        """Register callback for when instance changes."""
        self._on_changed.append(callback)

    def _notify_changed(self) -> None:
        """Notify listeners of changes."""
        for callback in self._on_changed:
            callback()

    def clone(self, new_name: str = "") -> 'MaterialInstance':
        """Create a copy of this instance."""
        instance = MaterialInstance(
            new_name or f"{self._name}_copy",
            self._parent
        )
        for name, override in self._overrides.items():
            instance._overrides[name] = ParameterOverride(
                parameter_name=override.parameter_name,
                value=copy.deepcopy(override.value),
                enabled=override.enabled
            )
        instance._tags = self._tags.copy()
        instance._metadata = copy.deepcopy(self._metadata)
        return instance

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self._id,
            "name": self._name,
            "parent_id": self._parent.id,
            "overrides": {name: o.to_dict() for name, o in self._overrides.items()},
            "tags": self._tags,
            "metadata": self._metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], parent: MaterialDefinition) -> 'MaterialInstance':
        """Deserialize from dictionary."""
        instance = cls(data["name"], parent)
        instance._id = data.get("id", str(uuid.uuid4()))
        for name, override_data in data.get("overrides", {}).items():
            instance._overrides[name] = ParameterOverride.from_dict(override_data)
        instance._tags = data.get("tags", [])
        instance._metadata = data.get("metadata", {})
        return instance

    def __del__(self):
        """Unregister from parent on deletion."""
        if hasattr(self, '_parent') and self._parent:
            self._parent.unregister_instance(self._id)


class MaterialInstanceManager:
    """
    Manages material definitions and instances.

    Provides lookup, creation, and lifecycle management for
    material definitions and their instances.
    """

    def __init__(self):
        self._definitions: Dict[str, MaterialDefinition] = {}
        self._instances: Dict[str, MaterialInstance] = {}
        self._definition_by_name: Dict[str, str] = {}  # name -> id
        self._instance_by_name: Dict[str, str] = {}  # name -> id

    @property
    def definition_count(self) -> int:
        return len(self._definitions)

    @property
    def instance_count(self) -> int:
        return len(self._instances)

    # ========================================================================
    # Definition Management
    # ========================================================================

    def create_definition(self, name: str, shader_path: str = "") -> MaterialDefinition:
        """Create a new material definition."""
        definition = MaterialDefinition(name, shader_path)
        self._definitions[definition.id] = definition
        self._definition_by_name[name] = definition.id
        return definition

    def get_definition(self, id: str) -> Optional[MaterialDefinition]:
        """Get definition by ID."""
        return self._definitions.get(id)

    def get_definition_by_name(self, name: str) -> Optional[MaterialDefinition]:
        """Get definition by name."""
        def_id = self._definition_by_name.get(name)
        if def_id:
            return self._definitions.get(def_id)
        return None

    def remove_definition(self, id: str) -> bool:
        """Remove a definition and all its instances."""
        definition = self._definitions.get(id)
        if definition is None:
            return False

        # Remove all instances of this definition
        instances_to_remove = [
            inst_id for inst_id, inst in self._instances.items()
            if inst.parent.id == id
        ]
        for inst_id in instances_to_remove:
            self.remove_instance(inst_id)

        # Remove definition
        del self._definitions[id]
        if definition.name in self._definition_by_name:
            del self._definition_by_name[definition.name]

        return True

    def get_all_definitions(self) -> List[MaterialDefinition]:
        """Get all definitions."""
        return list(self._definitions.values())

    # ========================================================================
    # Instance Management
    # ========================================================================

    def create_instance(
        self,
        name: str,
        definition_id: str
    ) -> Optional[MaterialInstance]:
        """Create a new material instance."""
        definition = self._definitions.get(definition_id)
        if definition is None:
            return None

        instance = MaterialInstance(name, definition)
        self._instances[instance.id] = instance
        self._instance_by_name[name] = instance.id
        return instance

    def create_instance_from_definition(
        self,
        name: str,
        definition: MaterialDefinition
    ) -> MaterialInstance:
        """Create instance directly from definition object."""
        # Ensure definition is registered
        if definition.id not in self._definitions:
            self._definitions[definition.id] = definition
            self._definition_by_name[definition.name] = definition.id

        instance = MaterialInstance(name, definition)
        self._instances[instance.id] = instance
        self._instance_by_name[name] = instance.id
        return instance

    def get_instance(self, id: str) -> Optional[MaterialInstance]:
        """Get instance by ID."""
        return self._instances.get(id)

    def get_instance_by_name(self, name: str) -> Optional[MaterialInstance]:
        """Get instance by name."""
        inst_id = self._instance_by_name.get(name)
        if inst_id:
            return self._instances.get(inst_id)
        return None

    def remove_instance(self, id: str) -> bool:
        """Remove an instance."""
        instance = self._instances.get(id)
        if instance is None:
            return False

        del self._instances[id]
        if instance.name in self._instance_by_name:
            del self._instance_by_name[instance.name]

        return True

    def get_all_instances(self) -> List[MaterialInstance]:
        """Get all instances."""
        return list(self._instances.values())

    def get_instances_of_definition(self, definition_id: str) -> List[MaterialInstance]:
        """Get all instances of a specific definition."""
        return [
            inst for inst in self._instances.values()
            if inst.parent.id == definition_id
        ]

    def get_instances_by_tag(self, tag: str) -> List[MaterialInstance]:
        """Get all instances with a specific tag."""
        return [inst for inst in self._instances.values() if inst.has_tag(tag)]

    # ========================================================================
    # Serialization
    # ========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize all definitions and instances."""
        return {
            "definitions": {id: d.to_dict() for id, d in self._definitions.items()},
            "instances": {id: i.to_dict() for id, i in self._instances.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MaterialInstanceManager':
        """Deserialize from dictionary."""
        manager = cls()

        # Load definitions first
        for id, def_data in data.get("definitions", {}).items():
            definition = MaterialDefinition.from_dict(def_data)
            manager._definitions[definition.id] = definition
            manager._definition_by_name[definition.name] = definition.id

        # Load instances
        for id, inst_data in data.get("instances", {}).items():
            parent_id = inst_data.get("parent_id")
            parent = manager._definitions.get(parent_id)
            if parent:
                instance = MaterialInstance.from_dict(inst_data, parent)
                manager._instances[instance.id] = instance
                manager._instance_by_name[instance.name] = instance.id

        return manager

    def clear(self) -> None:
        """Clear all definitions and instances."""
        self._instances.clear()
        self._definitions.clear()
        self._definition_by_name.clear()
        self._instance_by_name.clear()
