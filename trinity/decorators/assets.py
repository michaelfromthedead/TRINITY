"""
Trinity Pattern - Tier 8: ASSETS Decorators

Asset management, cooking, and residency control decorators.
All decorators use the ops-based system via make_decorator().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, TypeVar

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

# Valid constants
VALID_COMPRESSION = frozenset({"none", "lz4", "zstd"})
VALID_RESIDENCY_PRIORITIES = frozenset({"critical", "high", "normal", "low", "evictable"})

# Type variable for decorators
T = TypeVar("T")


# ============================================================================
# Configuration dataclasses
# ============================================================================


@dataclass(frozen=True)
class AssetConfig:
    """Asset configuration."""

    extensions: tuple[str, ...]
    loader: Optional[Callable] = None


@dataclass(frozen=True)
class CookConfig:
    """Cook configuration for asset optimization."""

    platform: Optional[str] = None
    compression: str = "lz4"
    strip_debug: bool = True


@dataclass(frozen=True)
class ResidencyConfig:
    """Residency configuration for memory management."""

    priority: str
    min_mip: int = 0


@dataclass(frozen=True)
class ImportSettingsConfig:
    """Import settings for external assets."""

    scale: float = 1.0
    axis_conversion: tuple[str, str, str] = ("X", "Y", "Z")
    merge_meshes: bool = False


# ============================================================================
# Step builders
# ============================================================================


def _asset_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @asset decorator."""
    extensions = params.get("extensions", ())
    loader = params.get("loader")

    return [
        Step(Op.TAG, {"key": "asset", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "asset_config",
                "value": AssetConfig(extensions=tuple(extensions), loader=loader),
            },
        ),
        Step(Op.REGISTER, {"registry": "assets"}),
        Step(Op.DESCRIBE, {}),
    ]


def _preload_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @preload decorator."""
    priority = params.get("priority", 0)

    return [
        Step(Op.TAG, {"key": "preload", "value": True}),
        Step(Op.TAG, {"key": "preload_priority", "value": priority}),
        Step(Op.REGISTER, {"registry": "assets"}),
    ]


def _cook_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @cook decorator."""
    platform = params.get("platform")
    compression = params.get("compression", "lz4")
    strip_debug = params.get("strip_debug", True)

    return [
        Step(Op.TAG, {"key": "cook", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "cook_config",
                "value": CookConfig(
                    platform=platform,
                    compression=compression,
                    strip_debug=strip_debug,
                ),
            },
        ),
        Step(Op.REGISTER, {"registry": "assets"}),
    ]


def _residency_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @residency decorator."""
    priority = params.get("priority", "normal")  # Default for introspection
    min_mip = params.get("min_mip", 0)

    return [
        Step(Op.TAG, {"key": "residency", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "residency_config",
                "value": ResidencyConfig(priority=priority, min_mip=min_mip),
            },
        ),
        Step(Op.REGISTER, {"registry": "assets"}),
        Step(
            Op.VALIDATE,
            {
                "check": "valid_priority",
                "validator": lambda obj: getattr(
                    obj._residency_config, "priority", None
                )
                in VALID_RESIDENCY_PRIORITIES,
            },
        ),
    ]


def _import_settings_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @import_settings decorator."""
    scale = params.get("scale", 1.0)
    axis_conversion = params.get("axis_conversion", ("X", "Y", "Z"))
    merge_meshes = params.get("merge_meshes", False)

    return [
        Step(Op.TAG, {"key": "import_settings", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "import_settings_config",
                "value": ImportSettingsConfig(
                    scale=scale,
                    axis_conversion=tuple(axis_conversion),
                    merge_meshes=merge_meshes,
                ),
            },
        ),
        Step(Op.REGISTER, {"registry": "assets"}),
    ]


# ============================================================================
# Validators
# ============================================================================


def _validate_asset_params(**kwargs: Any) -> None:
    """Validate @asset parameters."""
    extensions = kwargs.get("extensions", ())
    if not extensions:
        raise ValueError("extensions must be a non-empty tuple or list")
    if not isinstance(extensions, (tuple, list)):
        raise TypeError("extensions must be a tuple or list")


def _validate_cook_params(**kwargs: Any) -> None:
    """Validate @cook parameters."""
    compression = kwargs.get("compression", "lz4")
    if compression not in VALID_COMPRESSION:
        raise ValueError(
            f"Invalid compression '{compression}'. Must be one of {VALID_COMPRESSION}"
        )


def _validate_residency_params(**kwargs: Any) -> None:
    """Validate @residency parameters."""
    priority = kwargs.get("priority")
    if priority not in VALID_RESIDENCY_PRIORITIES:
        raise ValueError(
            f"Invalid priority '{priority}'. Must be one of {VALID_RESIDENCY_PRIORITIES}"
        )

    min_mip = kwargs.get("min_mip", 0)
    if min_mip < 0:
        raise ValueError(f"min_mip must be >= 0, got {min_mip}")


# ============================================================================
# After-apply functions
# ============================================================================


def _asset_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @asset is applied."""
    extensions = params.get("extensions", ())
    loader = params.get("loader")

    obj._asset = True
    obj._asset_extensions = tuple(extensions)
    obj._asset_loader = loader
    obj._asset_config = AssetConfig(extensions=tuple(extensions), loader=loader)


def _preload_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @preload is applied."""
    priority = params.get("priority", 0)

    obj._preload = True
    obj._preload_priority = priority


def _cook_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @cook is applied."""
    platform = params.get("platform")
    compression = params.get("compression", "lz4")
    strip_debug = params.get("strip_debug", True)

    obj._cook = True
    obj._cook_platform = platform
    obj._cook_compression = compression
    obj._cook_strip_debug = strip_debug
    obj._cook_config = CookConfig(
        platform=platform, compression=compression, strip_debug=strip_debug
    )


def _residency_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @residency is applied."""
    priority = params.get("priority", "normal")  # Default for introspection
    min_mip = params.get("min_mip", 0)

    obj._residency = True
    obj._residency_priority = priority
    obj._residency_min_mip = min_mip
    obj._residency_config = ResidencyConfig(priority=priority, min_mip=min_mip)


def _import_settings_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @import_settings is applied."""
    scale = params.get("scale", 1.0)
    axis_conversion = params.get("axis_conversion", ("X", "Y", "Z"))
    merge_meshes = params.get("merge_meshes", False)

    obj._import_settings = True
    obj._import_scale = scale
    obj._import_axis_conversion = tuple(axis_conversion)
    obj._import_merge_meshes = merge_meshes
    obj._import_settings_config = ImportSettingsConfig(
        scale=scale, axis_conversion=tuple(axis_conversion), merge_meshes=merge_meshes
    )


# ============================================================================
# Decorator creation
# ============================================================================

asset = make_decorator(
    name="asset",
    steps=_asset_steps,
    validate=_validate_asset_params,
    after_steps=_asset_after_apply,
)

preload = make_decorator(
    name="preload",
    steps=_preload_steps,
    after_steps=_preload_after_apply,
)

cook = make_decorator(
    name="cook",
    steps=_cook_steps,
    validate=_validate_cook_params,
    after_steps=_cook_after_apply,
)

residency = make_decorator(
    name="residency",
    steps=_residency_steps,
    validate=_validate_residency_params,
    after_steps=_residency_after_apply,
)

import_settings = make_decorator(
    name="import_settings",
    steps=_import_settings_steps,
    after_steps=_import_settings_after_apply,
)


# ============================================================================
# Registry registration
# ============================================================================

_REGISTRY_ENTRIES = [
    ("asset", asset, ("class",)),
    ("preload", preload, ("class",)),
    ("cook", cook, ("class",)),
    ("residency", residency, ("class",)),
    ("import_settings", import_settings, ("class",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.ASSETS,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.ASSETS].append(_spec)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "asset",
    "preload",
    "cook",
    "residency",
    "import_settings",
    "AssetConfig",
    "CookConfig",
    "ResidencyConfig",
    "ImportSettingsConfig",
    "VALID_COMPRESSION",
    "VALID_RESIDENCY_PRIORITIES",
]
