"""
Spatial decorators — built from Ops.

Decorators for spatial indexing and partitioning configuration.

Decorators:
    @spatial      - Configure spatial indexing structure
    @partitioned  - Configure spatial partitioning
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import Tier, registry

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# VALID VALUES
# =============================================================================

VALID_SPATIAL_STRUCTURES = frozenset({"grid", "quadtree", "octree", "bvh", "hash"})
VALID_PARTITION_DIMENSIONS = frozenset({2, 3})


# =============================================================================
# VALIDATORS
# =============================================================================


def _validate_spatial(structure: str = "", cell_size: float = 1.0, **_: Any) -> None:
    if not structure:
        raise ValueError("@spatial: 'structure' parameter is required")
    if structure not in VALID_SPATIAL_STRUCTURES:
        raise ValueError(
            f"@spatial: invalid structure '{structure}'. "
            f"Valid structures: {sorted(VALID_SPATIAL_STRUCTURES)}"
        )
    if cell_size <= 0:
        raise ValueError(
            f"@spatial: cell_size must be > 0, got {cell_size}"
        )


def _validate_partitioned(
    dimensions: int = 2, max_entities: int = 1000, **_: Any
) -> None:
    if dimensions not in VALID_PARTITION_DIMENSIONS:
        raise ValueError(
            f"@partitioned: invalid dimensions {dimensions}. "
            f"Valid dimensions: {sorted(VALID_PARTITION_DIMENSIONS)}"
        )
    if max_entities <= 0:
        raise ValueError(
            f"@partitioned: max_entities must be > 0, got {max_entities}"
        )


# =============================================================================
# STEP BUILDERS
# =============================================================================


def _spatial_steps(params: dict[str, Any]) -> list[Step]:
    structure = params.get("structure", "")
    cell_size = params.get("cell_size", 1.0)
    return [
        Step(Op.TAG, {"key": "spatial", "value": True}),
        Step(Op.TAG, {"key": "spatial_structure", "value": structure}),
        Step(Op.TAG, {"key": "spatial_cell_size", "value": cell_size}),
        Step(Op.REGISTER, {"registry": "spatial"}),
    ]


def _partitioned_steps(params: dict[str, Any]) -> list[Step]:
    dimensions = params.get("dimensions", 2)
    max_entities = params.get("max_entities", 1000)
    return [
        Step(Op.TAG, {"key": "partitioned", "value": True}),
        Step(Op.TAG, {"key": "partition_dimensions", "value": dimensions}),
        Step(Op.TAG, {"key": "partition_max_entities", "value": max_entities}),
        Step(Op.REGISTER, {"registry": "spatial"}),
    ]


# =============================================================================
# AFTER-STEPS
# =============================================================================


def _after_spatial(target: Any, params: dict[str, Any]) -> Any:
    target._spatial = True
    target._spatial_structure = params.get("structure", "")
    target._spatial_cell_size = params.get("cell_size", 1.0)
    return None


def _after_partitioned(target: Any, params: dict[str, Any]) -> Any:
    target._partitioned = True
    target._partition_dimensions = params.get("dimensions", 2)
    target._partition_max_entities = params.get("max_entities", 1000)
    return None


# =============================================================================
# DECORATOR DEFINITIONS
# =============================================================================

spatial = make_decorator(
    name="spatial",
    steps=_spatial_steps,
    doc="Configure spatial indexing structure for the target.",
    validate=_validate_spatial,
    after_steps=_after_spatial,
)

partitioned = make_decorator(
    name="partitioned",
    steps=_partitioned_steps,
    doc="Configure spatial partitioning for the target.",
    validate=_validate_partitioned,
    after_steps=_after_partitioned,
)


# =============================================================================
# REGISTRY REGISTRATION
# =============================================================================

from trinity.decorators.registry import DecoratorSpec

_REGISTRY_ENTRIES: list[tuple[str, Any, tuple[str, ...]]] = [
    ("spatial", spatial, ("class",)),
    ("partitioned", partitioned, ("class",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.SPATIAL,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.SPATIAL].append(_spec)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "spatial",
    "partitioned",
    "VALID_SPATIAL_STRUCTURES",
    "VALID_PARTITION_DIMENSIONS",
]
