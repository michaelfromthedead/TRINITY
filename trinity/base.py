"""
Base classes for Trinity Pattern types.

These are the classes that users inherit from when creating components,
systems, resources, events, assets, protocols, and states.
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional

from trinity.constants import DEFAULT_RESOURCE_PRIORITY
from trinity.metaclasses.asset_meta import AssetMeta
from trinity.metaclasses.component_meta import ComponentMeta
from trinity.metaclasses.event_meta import EventMeta
from trinity.metaclasses.protocol_meta import ProtocolMeta
from trinity.metaclasses.resource_meta import ResourceMeta
from trinity.metaclasses.state_meta import StateMeta
from trinity.metaclasses.system_meta import SystemMeta


class Component(metaclass=ComponentMeta):
    """
    Base class for ECS components.

    Components are data containers that hold entity state.
    They should be data-only with no complex logic.

    Usage:
        class Health(Component):
            current: float = 100.0
            maximum: float = 100.0

        class Position(Component):
            x: float = 0.0
            y: float = 0.0
            z: float = 0.0
    """

    # These are set by ComponentMeta
    _component_id: ClassVar[int]
    _component_name: ClassVar[str]
    _field_types: ClassVar[dict[str, type]]
    _field_descriptors: ClassVar[dict[str, Any]]
    _field_offsets: ClassVar[dict[str, int]]
    _field_defaults: ClassVar[dict[str, Any]]

    # Optional configuration (set by decorators)
    _track_changes: ClassVar[bool] = False
    _network_config: ClassVar[Optional[Any]] = None
    _serialization_config: ClassVar[Optional[Any]] = None

    def __init__(self, **kwargs: Any) -> None:
        """Initialize component with optional field values."""
        # Initialize dirty tracking if enabled
        if self._track_changes:
            self._dirty_fields: set[str] = set()

        # Initialize network queue if networked
        if self._network_config is not None:
            self._network_queue: list[dict[str, Any]] = []

        # Set initial values from kwargs
        for name, value in kwargs.items():
            if name in self._field_types:
                setattr(self, name, value)
            else:
                raise TypeError(
                    f"{type(self).__name__}() got unexpected keyword argument '{name}'"
                )

    def __repr__(self) -> str:
        """Generate a readable representation."""
        fields = []
        for name in self._field_types:
            try:
                value = getattr(self, name)
                fields.append(f"{name}={value!r}")
            except AttributeError:
                pass
        return f"{type(self).__name__}({', '.join(fields)})"


class System(metaclass=SystemMeta):
    """
    Base class for ECS systems.

    Systems contain logic that operates on components.
    They declare what components they read and write.

    Usage:
        class MovementSystem(System):
            _reads = (Position, Velocity)
            _writes = (Position,)

            def execute(self, entities):
                for entity in entities:
                    entity.position.x += entity.velocity.x
                    entity.position.y += entity.velocity.y
    """

    # These are set by SystemMeta
    _system_id: ClassVar[int]
    _system_name: ClassVar[str]
    _dependencies: ClassVar[set[int]]
    _can_parallelize: ClassVar[bool]

    # Configuration (set by decorators or class definition)
    _reads: ClassVar[tuple[type, ...]] = ()
    _writes: ClassVar[tuple[type, ...]] = ()
    _resources: ClassVar[tuple[type, ...]] = ()
    _exclusive: ClassVar[bool] = False
    _priority: ClassVar[int] = 0

    def execute(self, *args: Any, **kwargs: Any) -> None:
        """
        Execute the system logic.

        Override this method to implement system behavior.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.execute() must be implemented"
        )


class Resource(metaclass=ResourceMeta):
    """
    Base class for global singleton resources.

    Resources are global state containers that systems can access.
    Only one instance of each resource type can exist.

    Usage:
        class TimeResource(Resource):
            delta_time: float = 0.0
            total_time: float = 0.0
            tick: int = 0

        # Access the singleton
        time = TimeResource()
    """

    # These are set by ResourceMeta
    _resource_id: ClassVar[int]
    _resource_name: ClassVar[str]

    # Configuration (set by decorators or class definition)
    _resource_priority: ClassVar[int] = DEFAULT_RESOURCE_PRIORITY
    _resource_dependencies: ClassVar[tuple[type, ...]] = ()

    def shutdown(self) -> None:
        """
        Clean up resource on shutdown.

        Override this method to perform cleanup when the engine shuts down.
        """
        pass


class Event(metaclass=EventMeta):
    """
    Base class for event types.

    Events are data-only objects used for communication between systems.
    They cannot have methods (except __init__, __repr__, etc.).

    Usage:
        class DamageEvent(Event):
            target_id: int
            amount: float
            source_id: int

        class DeathEvent(DamageEvent):
            killer_id: int
    """

    # These are set by EventMeta
    _event_id: ClassVar[int]
    _event_name: ClassVar[str]
    _event_fields: ClassVar[dict[str, type]]
    _event_parent_ids: ClassVar[tuple[int, ...]]

    # Configuration (set by decorators or class definition)
    _event_priority: ClassVar[int] = 0
    _event_channels: ClassVar[tuple[str, ...]] = ()
    _event_pooled: ClassVar[bool] = False

    def __init__(self, **kwargs: Any) -> None:
        """Initialize event with field values."""
        for name, value in kwargs.items():
            if name in self._event_fields:
                setattr(self, name, value)
            else:
                raise TypeError(
                    f"{type(self).__name__}() got unexpected keyword argument '{name}'"
                )

    def __repr__(self) -> str:
        """Generate a readable representation."""
        fields = []
        for name in self._event_fields:
            if hasattr(self, name):
                value = getattr(self, name)
                fields.append(f"{name}={value!r}")
        return f"{type(self).__name__}({', '.join(fields)})"


class Asset(metaclass=AssetMeta):
    """
    Base class for asset handle types.

    Assets represent external resources like textures, models, sounds.
    Each asset type declares what file extensions it handles.

    Usage:
        class Texture(Asset):
            _asset_extensions = ('.png', '.jpg', '.jpeg', '.tga')

            width: int = 0
            height: int = 0
            data: bytes = b''

        class Model(Asset):
            _asset_extensions = ('.obj', '.fbx', '.gltf')
    """

    # These are set by AssetMeta
    _asset_id: ClassVar[int]
    _asset_name: ClassVar[str]
    _asset_type_code: ClassVar[str]

    # Required configuration
    _asset_extensions: ClassVar[tuple[str, ...]]  # Must be defined by subclass

    # Optional configuration
    _asset_loader: ClassVar[Optional[type]] = None
    _asset_hot_reload: ClassVar[bool] = False
    _asset_priority: ClassVar[int] = 0


class Protocol(metaclass=ProtocolMeta):
    """
    Base class for network protocol definitions.

    Protocols define the network communication format.
    They must specify a version number.

    Usage:
        class GameProtocol(Protocol):
            _protocol_version = 1
            _protocol_min_version = 1
    """

    # These are set by ProtocolMeta
    _protocol_id: ClassVar[int]
    _protocol_qualified_name: ClassVar[str]

    # Required configuration
    _protocol_version: ClassVar[int]  # Must be defined by subclass

    # Optional configuration
    _protocol_min_version: ClassVar[int] = 1
    _protocol_messages: ClassVar[dict[int, type]] = {}
    _protocol_name: ClassVar[str] = ""


class State(metaclass=StateMeta):
    """
    Base class for state machine states.

    States define behavior in a state machine.
    They can declare allowed transitions and lifecycle hooks.

    Usage:
        class IdleState(State):
            _state_transitions = {'walking', 'running', 'jumping'}

            def on_enter(self, entity):
                entity.velocity = 0

            def on_exit(self, entity):
                pass
    """

    # These are set by StateMeta
    _state_id: ClassVar[int]
    _state_name: ClassVar[str]
    _state_qualified_name: ClassVar[str]

    # Optional configuration
    _state_transitions: ClassVar[set[str]] = set()
    _state_machine_cls: ClassVar[Optional[type]] = None
    _state_on_enter: ClassVar[Optional[callable]] = None
    _state_on_exit: ClassVar[Optional[callable]] = None
