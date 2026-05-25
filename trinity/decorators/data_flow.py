"""
Data flow decorators — built from Ops.

These decorators manage data serialization, networking, snapshots, and versioning
for game state synchronization and persistence.

Every decorator here is a named list of Steps, created by make_decorator.
The Steps do the real work. The decorators are just configuration.

Decorators:
    @serializable - Data serialization marker
    @networked    - Network replication configuration
    @snapshot     - State snapshot/restore with history
    @versioned    - Version migration support
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypeVar

from trinity.decorators.base import validate_target_type
from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import Tier, registry

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# VALID VALUES
# =============================================================================

VALID_SERIALIZATION_FORMATS = frozenset({"binary", "json", "msgpack"})
VALID_NETWORK_RELEVANCE = frozenset({"global", "spatial", "owner"})
VALID_NETWORK_AUTHORITY = frozenset({"server", "client", "owner"})
VALID_INTERPOLATION_MODES = frozenset({"linear", "hermite", "none"})


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================


@dataclass(frozen=True)
class SerializableConfig:
    """Configuration for @serializable decorator."""

    format: str = "binary"
    version: int = 1


@dataclass(frozen=True)
class NetworkedConfig:
    """Configuration for @networked decorator."""

    relevance: str = "spatial"
    authority: str = "server"
    priority: int = 0
    unreliable: bool = False
    delta: bool = False
    predicted: bool = False
    interpolated: str = "none"


@dataclass(frozen=True)
class SnapshotConfig:
    """Configuration for @snapshot decorator."""

    history_frames: int = 60


@dataclass(frozen=True)
class VersionedConfig:
    """Configuration for @versioned decorator."""

    version: int = 1
    migrations: dict = field(default_factory=dict)


# =============================================================================
# VALIDATORS
# =============================================================================


def _validate_serializable(format: str = "binary", version: int = 1, **_: Any) -> None:
    if format not in VALID_SERIALIZATION_FORMATS:
        raise ValueError(
            f"@serializable: invalid format '{format}'. "
            f"Valid formats: {sorted(VALID_SERIALIZATION_FORMATS)}"
        )
    if not isinstance(version, int) or version < 1:
        raise ValueError(f"@serializable: version must be positive integer, got {version}")


def _validate_networked(
    relevance: str = "spatial",
    authority: str = "server",
    priority: int = 0,
    interpolated: str = "none",
    **_: Any,
) -> None:
    if relevance not in VALID_NETWORK_RELEVANCE:
        raise ValueError(
            f"@networked: invalid relevance '{relevance}'. "
            f"Valid values: {sorted(VALID_NETWORK_RELEVANCE)}"
        )
    if authority not in VALID_NETWORK_AUTHORITY:
        raise ValueError(
            f"@networked: invalid authority '{authority}'. "
            f"Valid values: {sorted(VALID_NETWORK_AUTHORITY)}"
        )
    if not isinstance(priority, int):
        raise ValueError(f"@networked: priority must be integer, got {type(priority)}")
    if interpolated not in VALID_INTERPOLATION_MODES:
        raise ValueError(
            f"@networked: invalid interpolated '{interpolated}'. "
            f"Valid modes: {sorted(VALID_INTERPOLATION_MODES)}"
        )


def _validate_snapshot(history_frames: int = 60, **_: Any) -> None:
    if not isinstance(history_frames, int) or history_frames < 1:
        raise ValueError(
            f"@snapshot: history_frames must be positive integer, got {history_frames}"
        )


def _validate_versioned(version: int = 1, migrations: Any = None, **_: Any) -> None:
    if not isinstance(version, int) or version < 1:
        raise ValueError(f"@versioned: version must be positive integer, got {version}")
    if migrations is not None and not isinstance(migrations, dict):
        raise ValueError(f"@versioned: migrations must be dict, got {type(migrations)}")


# =============================================================================
# STEP BUILDERS
# =============================================================================


def _serializable_steps(params: dict[str, Any]) -> list[Step]:
    fmt = params.get("format", "binary")
    ver = params.get("version", 1)
    return [
        Step(Op.TAG, {"key": "serializable", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "serializable_config",
                "value": SerializableConfig(format=fmt, version=ver),
            },
        ),
        Step(Op.REGISTER, {"registry": "data_flow"}),
        Step(Op.DESCRIBE, {}),
    ]


def _networked_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "networked", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "networked_config",
                "value": NetworkedConfig(
                    relevance=params.get("relevance", "spatial"),
                    authority=params.get("authority", "server"),
                    priority=params.get("priority", 0),
                    unreliable=params.get("unreliable", False),
                    delta=params.get("delta", False),
                    predicted=params.get("predicted", False),
                    interpolated=params.get("interpolated", "none"),
                ),
            },
        ),
        Step(Op.REGISTER, {"registry": "data_flow"}),
    ]


def _snapshot_steps(params: dict[str, Any]) -> list[Step]:
    history = params.get("history_frames", 60)
    return [
        Step(Op.TAG, {"key": "snapshot", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "snapshot_config",
                "value": SnapshotConfig(history_frames=history),
            },
        ),
        Step(Op.REGISTER, {"registry": "data_flow"}),
    ]


def _versioned_steps(params: dict[str, Any]) -> list[Step]:
    ver = params.get("version", 1)
    migrations = params.get("migrations", {})
    return [
        Step(Op.TAG, {"key": "versioned", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "versioned_config",
                "value": VersionedConfig(version=ver, migrations=migrations),
            },
        ),
        Step(Op.REGISTER, {"registry": "data_flow"}),
        Step(Op.VALIDATE, {"check": "requires_serializable"}),
    ]


# =============================================================================
# AFTER-STEPS (domain behavior that isn't an Op)
# =============================================================================


def _after_serializable(target: Any, params: dict[str, Any]) -> Any:
    """Attach serialization attributes and methods."""
    validate_target_type(target, "serializable", ("class",))
    config = target._tags.get("serializable_config")
    target._serializable = True
    target._serializable_format = config.format
    target._serializable_version = config.version

    # Extract fields from annotations if available
    if hasattr(target, "__annotations__"):
        target._serializable_fields = list(target.__annotations__.keys())
    else:
        target._serializable_fields = []

    # Add serialize/deserialize classmethods (stub implementations)
    @classmethod
    def serialize(cls, obj: Any) -> dict:
        """Serialize object to dictionary (stub implementation)."""
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj)}")
        data = {"__version__": cls._serializable_version, "__type__": cls.__name__}
        for field in cls._serializable_fields:
            if hasattr(obj, field):
                data[field] = getattr(obj, field)
        return data

    @classmethod
    def deserialize(cls, data: dict) -> Any:
        """Deserialize object from dictionary (stub implementation)."""
        if data.get("__type__") != cls.__name__:
            raise ValueError(f"Type mismatch: expected {cls.__name__}, got {data.get('__type__')}")
        # Create instance with default constructor
        obj = cls.__new__(cls)
        for field in cls._serializable_fields:
            if field in data:
                setattr(obj, field, data[field])
        return obj

    target.serialize = serialize
    target.deserialize = deserialize

    return None


def _after_networked(target: Any, params: dict[str, Any]) -> Any:
    """Attach network replication attributes and methods."""
    validate_target_type(target, "networked", ("class",))
    config = target._tags.get("networked_config")
    target._networked = True
    target._networked_relevance = config.relevance
    target._networked_authority = config.authority
    target._networked_priority = config.priority
    target._networked_unreliable = config.unreliable
    target._networked_delta = config.delta
    target._networked_predicted = config.predicted
    target._networked_interpolated = config.interpolated

    # Add network serialization methods (stub implementations)
    def _serialize_net(self) -> dict:
        """Serialize for network transmission (stub implementation)."""
        data = {"__type__": self.__class__.__name__}
        if hasattr(self, "_serializable_fields"):
            for field in self._serializable_fields:
                if hasattr(self, field):
                    data[field] = getattr(self, field)
        return data

    def _deserialize_net(self, data: dict) -> None:
        """Deserialize from network data (stub implementation)."""
        for key, value in data.items():
            if key != "__type__":
                setattr(self, key, value)

    target._serialize_net = _serialize_net
    target._deserialize_net = _deserialize_net

    return None


def _after_snapshot(target: Any, params: dict[str, Any]) -> Any:
    """Attach snapshot history and methods."""
    validate_target_type(target, "snapshot", ("class",))

    # Validate that @serializable is present
    if not hasattr(target, "_serializable"):
        raise TypeError(
            f"@snapshot requires @serializable to be applied first on {target.__name__}"
        )

    config = target._tags.get("snapshot_config")
    target._snapshot = True
    target._snapshot_history_frames = config.history_frames

    # Add snapshot methods (stub implementations using ring buffer)
    def snapshot_save(self) -> int:
        """Save current state to history and return frame number (stub implementation)."""
        if not hasattr(self, "_snapshot_history"):
            self._snapshot_history = []
            self._snapshot_frame = 0

        # Serialize current state
        state = self.__class__.serialize(self)

        # Ring buffer logic
        if len(self._snapshot_history) >= self._snapshot_history_frames:
            self._snapshot_history.pop(0)

        self._snapshot_history.append(state)
        self._snapshot_frame += 1
        return self._snapshot_frame - 1

    def snapshot_restore(self, frame: int) -> bool:
        """Restore state from history frame (stub implementation)."""
        if not hasattr(self, "_snapshot_history"):
            return False

        if frame < 0 or frame >= len(self._snapshot_history):
            return False

        state = self._snapshot_history[frame]
        # Deserialize into self
        for key, value in state.items():
            if key not in ("__version__", "__type__"):
                setattr(self, key, value)
        return True

    target.snapshot_save = snapshot_save
    target.snapshot_restore = snapshot_restore

    return None


def _after_versioned(target: Any, params: dict[str, Any]) -> Any:
    """Attach versioning attributes."""
    validate_target_type(target, "versioned", ("class",))

    # Validate that @serializable is present
    if not hasattr(target, "_serializable"):
        raise TypeError(
            f"@versioned requires @serializable to be applied first on {target.__name__}"
        )

    config = target._tags.get("versioned_config")
    target._versioned = True
    target._versioned_version = config.version
    target._versioned_migrations = config.migrations

    return None


# =============================================================================
# DECORATOR DEFINITIONS
# =============================================================================


serializable = make_decorator(
    name="serializable",
    steps=_serializable_steps,
    doc="Mark class for data serialization with format and version support.",
    validate=_validate_serializable,
    after_steps=_after_serializable,
)

networked = make_decorator(
    name="networked",
    steps=_networked_steps,
    doc="Configure network replication with relevance, authority, and interpolation settings.",
    validate=_validate_networked,
    after_steps=_after_networked,
)

snapshot = make_decorator(
    name="snapshot",
    steps=_snapshot_steps,
    doc="Enable state snapshot/restore with configurable history. Requires @serializable.",
    validate=_validate_snapshot,
    after_steps=_after_snapshot,
)

versioned = make_decorator(
    name="versioned",
    steps=_versioned_steps,
    doc="Add version tracking and migration support. Requires @serializable.",
    validate=_validate_versioned,
    after_steps=_after_versioned,
)


# =============================================================================
# REGISTRY REGISTRATION
# =============================================================================
# Register decorator *definitions* with the registry for discoverability.
# This is separate from the Op-based REGISTER step which tracks applied instances.

from trinity.decorators.registry import DecoratorSpec

_REGISTRY_ENTRIES: list[tuple[str, Any, tuple[str, ...]]] = [
    ("serializable", serializable, ("class",)),
    ("networked", networked, ("class",)),
    ("snapshot", snapshot, ("class",)),
    ("versioned", versioned, ("class",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.DATA_FLOW,
            func=_func,
            unique=True,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.DATA_FLOW].append(_spec)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Decorators
    "serializable",
    "networked",
    "snapshot",
    "versioned",
    # Configuration classes
    "SerializableConfig",
    "NetworkedConfig",
    "SnapshotConfig",
    "VersionedConfig",
    # Valid values
    "VALID_SERIALIZATION_FORMATS",
    "VALID_NETWORK_RELEVANCE",
    "VALID_NETWORK_AUTHORITY",
    "VALID_INTERPOLATION_MODES",
]
