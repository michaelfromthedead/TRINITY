"""
Trinity Pattern - Tier 33: SAVE_SYSTEM Decorators

Save game management, persistence, and synchronization decorators.
All decorators use the ops-based system via make_decorator().
"""

from __future__ import annotations

from typing import Any, Literal, TypeVar

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

# Valid constants
VALID_CONFLICT_RESOLUTIONS = frozenset({"newest", "ask_player", "merge"})

# Type variable for decorators
T = TypeVar("T")


# ============================================================================
# Step builders
# ============================================================================


def _save_slot_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @save_slot decorator."""
    max_slots = params.get("max_slots", 10)
    auto_save = params.get("auto_save", True)
    auto_save_interval = params.get("auto_save_interval", 300.0)

    return [
        Step(Op.TAG, {"key": "save_slot", "value": True}),
        Step(Op.TAG, {"key": "save_max_slots", "value": max_slots}),
        Step(Op.TAG, {"key": "save_auto_save", "value": auto_save}),
        Step(Op.TAG, {"key": "save_auto_save_interval", "value": auto_save_interval}),
        Step(Op.REGISTER, {"registry": "save_system"}),
    ]


def _atomic_save_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @atomic_save decorator."""
    return [
        Step(Op.TAG, {"key": "atomic_save", "value": True}),
        Step(Op.REGISTER, {"registry": "save_system"}),
    ]


def _cloud_sync_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @cloud_sync decorator."""
    platform = params.get("platform", "")
    conflict_resolution = params.get("conflict_resolution", "newest")

    return [
        Step(Op.TAG, {"key": "cloud_sync", "value": True}),
        Step(Op.TAG, {"key": "cloud_platform", "value": platform}),
        Step(Op.TAG, {"key": "cloud_conflict_resolution", "value": conflict_resolution}),
        Step(Op.REGISTER, {"registry": "save_system"}),
    ]


def _save_migration_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @save_migration decorator."""
    from_version = params.get("from_version", 0)
    to_version = params.get("to_version", 1)

    return [
        Step(Op.TAG, {"key": "save_migration", "value": True}),
        Step(Op.TAG, {"key": "migration_from_version", "value": from_version}),
        Step(Op.TAG, {"key": "migration_to_version", "value": to_version}),
        Step(Op.REGISTER, {"registry": "save_system"}),
    ]


# ============================================================================
# Validators
# ============================================================================


def _validate_save_slot_params(**kwargs: Any) -> None:
    """Validate @save_slot parameters."""
    max_slots = kwargs.get("max_slots", 10)
    if not isinstance(max_slots, int) or max_slots <= 0:
        raise ValueError(f"max_slots must be a positive integer, got {max_slots}")

    auto_save_interval = kwargs.get("auto_save_interval", 300.0)
    if not isinstance(auto_save_interval, (int, float)) or auto_save_interval <= 0:
        raise ValueError(
            f"auto_save_interval must be a positive number, got {auto_save_interval}"
        )


def _validate_cloud_sync_params(**kwargs: Any) -> None:
    """Validate @cloud_sync parameters."""
    platform = kwargs.get("platform", "")
    if not platform or not isinstance(platform, str):
        raise ValueError("platform must be a non-empty string")

    conflict_resolution = kwargs.get("conflict_resolution", "newest")
    if conflict_resolution not in VALID_CONFLICT_RESOLUTIONS:
        raise ValueError(
            f"Invalid conflict_resolution '{conflict_resolution}'. "
            f"Must be one of {VALID_CONFLICT_RESOLUTIONS}"
        )


def _validate_save_migration_params(**kwargs: Any) -> None:
    """Validate @save_migration parameters."""
    from_version = kwargs.get("from_version", 0)
    to_version = kwargs.get("to_version", 1)

    if not isinstance(from_version, int) or from_version < 0:
        raise ValueError(f"from_version must be >= 0, got {from_version}")

    if not isinstance(to_version, int) or to_version <= from_version:
        raise ValueError(
            f"to_version must be > from_version ({from_version}), got {to_version}"
        )


# ============================================================================
# After-apply functions
# ============================================================================


def _save_slot_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @save_slot is applied."""
    max_slots = params.get("max_slots", 10)
    auto_save = params.get("auto_save", True)
    auto_save_interval = params.get("auto_save_interval", 300.0)

    obj._save_slot = True
    obj._save_max_slots = max_slots
    obj._save_auto_save = auto_save
    obj._save_auto_save_interval = auto_save_interval


def _atomic_save_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @atomic_save is applied."""
    obj._atomic_save = True


def _cloud_sync_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @cloud_sync is applied."""
    platform = params.get("platform", "")
    conflict_resolution = params.get("conflict_resolution", "newest")

    obj._cloud_sync = True
    obj._cloud_platform = platform
    obj._cloud_conflict_resolution = conflict_resolution


def _save_migration_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @save_migration is applied."""
    from_version = params.get("from_version", 0)
    to_version = params.get("to_version", 1)

    obj._save_migration = True
    obj._migration_from_version = from_version
    obj._migration_to_version = to_version


# ============================================================================
# Decorator creation
# ============================================================================

save_slot = make_decorator(
    name="save_slot",
    steps=_save_slot_steps,
    doc="Save slot management with optional auto-save configuration.",
    validate=_validate_save_slot_params,
    after_steps=_save_slot_after_apply,
)

atomic_save = make_decorator(
    name="atomic_save",
    steps=_atomic_save_steps,
    doc="Atomic save operation using write-temp-then-rename pattern.",
    after_steps=_atomic_save_after_apply,
)

cloud_sync = make_decorator(
    name="cloud_sync",
    steps=_cloud_sync_steps,
    doc="Cloud save synchronization with conflict resolution strategies.",
    validate=_validate_cloud_sync_params,
    after_steps=_cloud_sync_after_apply,
)

save_migration = make_decorator(
    name="save_migration",
    steps=_save_migration_steps,
    doc="Save file migration between versions.",
    validate=_validate_save_migration_params,
    after_steps=_save_migration_after_apply,
)


# ============================================================================
# Registry registration
# ============================================================================

_REGISTRY_ENTRIES = [
    ("save_slot", save_slot, ("class",)),
    ("atomic_save", atomic_save, ("class",)),
    ("cloud_sync", cloud_sync, ("class",)),
    ("save_migration", save_migration, ("class",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.SAVE_SYSTEM,
            func=_func,
            unique=_name != "save_migration",  # Allow multiple migrations
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.SAVE_SYSTEM].append(_spec)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "save_slot",
    "atomic_save",
    "cloud_sync",
    "save_migration",
    "VALID_CONFLICT_RESOLUTIONS",
]
