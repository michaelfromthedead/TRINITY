"""
Trinity Pattern - Tier 41: BUILD_DEPLOY Decorators

Build configuration and deployment decorators.
All decorators use the ops-based system via make_decorator().
"""

from __future__ import annotations

from typing import Any, TypeVar

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

# Type variable for decorators
T = TypeVar("T")


# ============================================================================
# Step builders
# ============================================================================


def _build_only_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @build_only decorator."""
    configurations = params.get("configurations", {"debug"})

    return [
        Step(Op.TAG, {"key": "build_only", "value": True}),
        Step(Op.TAG, {"key": "build_configurations", "value": frozenset(configurations)}),
        Step(Op.REGISTER, {"registry": "build_deploy"}),
    ]


def _strip_in_release_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @strip_in_release decorator."""
    return [
        Step(Op.TAG, {"key": "strip_in_release", "value": True}),
        Step(Op.REGISTER, {"registry": "build_deploy"}),
    ]


def _asset_bundle_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @asset_bundle decorator."""
    name = params.get("name", "")
    platforms = params.get("platforms")

    return [
        Step(Op.TAG, {"key": "asset_bundle", "value": True}),
        Step(Op.TAG, {"key": "asset_bundle_name", "value": name}),
        Step(
            Op.TAG,
            {
                "key": "asset_bundle_platforms",
                "value": frozenset(platforms) if platforms else None,
            },
        ),
        Step(Op.REGISTER, {"registry": "build_deploy"}),
    ]


def _feature_flag_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @feature_flag decorator."""
    id = params.get("id", "")
    default = params.get("default", False)

    return [
        Step(Op.TAG, {"key": "feature_flag", "value": True}),
        Step(Op.TAG, {"key": "feature_flag_id", "value": id}),
        Step(Op.TAG, {"key": "feature_flag_default", "value": default}),
        Step(Op.REGISTER, {"registry": "build_deploy"}),
    ]


# ============================================================================
# Validators
# ============================================================================


def _validate_build_only_params(**kwargs: Any) -> None:
    """Validate @build_only parameters."""
    configurations = kwargs.get("configurations", {"debug"})
    if not configurations:
        raise ValueError("configurations must be a non-empty set")
    if not isinstance(configurations, (set, frozenset)):
        raise TypeError("configurations must be a set")


def _validate_asset_bundle_params(**kwargs: Any) -> None:
    """Validate @asset_bundle parameters."""
    name = kwargs.get("name", "")
    if not name:
        raise ValueError("name must be a non-empty string")
    if not isinstance(name, str):
        raise TypeError("name must be a string")


def _validate_feature_flag_params(**kwargs: Any) -> None:
    """Validate @feature_flag parameters."""
    id = kwargs.get("id", "")
    if not id:
        raise ValueError("id must be a non-empty string")
    if not isinstance(id, str):
        raise TypeError("id must be a string")


# ============================================================================
# After-apply functions
# ============================================================================


def _build_only_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @build_only is applied."""
    configurations = params.get("configurations", {"debug"})

    obj._build_only = True
    obj._build_configurations = frozenset(configurations)


def _strip_in_release_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @strip_in_release is applied."""
    obj._strip_in_release = True


def _asset_bundle_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @asset_bundle is applied."""
    name = params.get("name", "")
    platforms = params.get("platforms")

    obj._asset_bundle = True
    obj._asset_bundle_name = name
    obj._asset_bundle_platforms = frozenset(platforms) if platforms else None


def _feature_flag_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @feature_flag is applied."""
    id = params.get("id", "")
    default = params.get("default", False)

    obj._feature_flag = True
    obj._feature_flag_id = id
    obj._feature_flag_default = default


# ============================================================================
# Decorator creation
# ============================================================================

build_only = make_decorator(
    name="build_only",
    steps=_build_only_steps,
    doc="Mark code to only exist in certain build configurations (e.g., debug).",
    validate=_validate_build_only_params,
    after_steps=_build_only_after_apply,
)

strip_in_release = make_decorator(
    name="strip_in_release",
    steps=_strip_in_release_steps,
    doc="Marker decorator to remove code from release builds.",
    after_steps=_strip_in_release_after_apply,
)

asset_bundle = make_decorator(
    name="asset_bundle",
    steps=_asset_bundle_steps,
    doc="Group assets into named bundles for platform-specific builds.",
    validate=_validate_asset_bundle_params,
    after_steps=_asset_bundle_after_apply,
)

feature_flag = make_decorator(
    name="feature_flag",
    steps=_feature_flag_steps,
    doc="Runtime feature toggles with configurable defaults.",
    validate=_validate_feature_flag_params,
    after_steps=_feature_flag_after_apply,
)


# ============================================================================
# Registry registration
# ============================================================================

_REGISTRY_ENTRIES = [
    ("build_only", build_only, ("class", "function")),
    ("strip_in_release", strip_in_release, ("class", "function")),
    ("asset_bundle", asset_bundle, ("class",)),
    ("feature_flag", feature_flag, ("class", "function")),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.BUILD_DEPLOY,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.BUILD_DEPLOY].append(_spec)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "build_only",
    "strip_in_release",
    "asset_bundle",
    "feature_flag",
]
