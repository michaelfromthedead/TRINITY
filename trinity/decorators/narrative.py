"""
Trinity Pattern - Tier 34: NARRATIVE Decorators

Dialogue and conversation system decorators.
All decorators use the ops-based system via make_decorator().
"""

from __future__ import annotations

from typing import Any, TypeVar

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

# Type variable for decorators
T = TypeVar("T")


# ============================================================================
# Validators
# ============================================================================


def _validate_dialogue_params(**kwargs: Any) -> None:
    """Validate @dialogue parameters."""
    id_param = kwargs.get("id", "")
    if not id_param or not isinstance(id_param, str):
        raise ValueError("id must be a non-empty string")


def _validate_conversation_params(**kwargs: Any) -> None:
    """Validate @conversation parameters."""
    id_param = kwargs.get("id", "")
    if not id_param or not isinstance(id_param, str):
        raise ValueError("id must be a non-empty string")

    start_node = kwargs.get("start_node", "")
    if not start_node or not isinstance(start_node, str):
        raise ValueError("start_node must be a non-empty string")


def _validate_voice_over_params(**kwargs: Any) -> None:
    """Validate @voice_over parameters."""
    audio_asset = kwargs.get("audio_asset", "")
    if not audio_asset or not isinstance(audio_asset, str):
        raise ValueError("audio_asset must be a non-empty string")


# ============================================================================
# Step builders
# ============================================================================


def _dialogue_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @dialogue decorator."""
    id_param = params.get("id", "")
    speaker = params.get("speaker")

    return [
        Step(Op.TAG, {"key": "dialogue", "value": True}),
        Step(Op.TAG, {"key": "dialogue_id", "value": id_param}),
        Step(Op.TAG, {"key": "dialogue_speaker", "value": speaker}),
        Step(Op.REGISTER, {"registry": "narrative"}),
        Step(Op.DESCRIBE, {}),
    ]


def _conversation_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @conversation decorator."""
    id_param = params.get("id", "")
    start_node = params.get("start_node", "")

    return [
        Step(Op.TAG, {"key": "conversation", "value": True}),
        Step(Op.TAG, {"key": "conversation_id", "value": id_param}),
        Step(Op.TAG, {"key": "conversation_start_node", "value": start_node}),
        Step(Op.REGISTER, {"registry": "narrative"}),
        Step(Op.DESCRIBE, {}),
    ]


def _voice_over_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @voice_over decorator."""
    audio_asset = params.get("audio_asset", "")
    lip_sync = params.get("lip_sync")

    return [
        Step(Op.TAG, {"key": "voice_over", "value": True}),
        Step(Op.TAG, {"key": "voice_over_audio_asset", "value": audio_asset}),
        Step(Op.TAG, {"key": "voice_over_lip_sync", "value": lip_sync}),
        Step(Op.REGISTER, {"registry": "narrative"}),
    ]


# ============================================================================
# After-apply functions
# ============================================================================


def _dialogue_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @dialogue is applied."""
    id_param = params.get("id", "")
    speaker = params.get("speaker")

    obj._dialogue = True
    obj._dialogue_id = id_param
    obj._dialogue_speaker = speaker


def _conversation_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @conversation is applied."""
    id_param = params.get("id", "")
    start_node = params.get("start_node", "")

    obj._conversation = True
    obj._conversation_id = id_param
    obj._conversation_start_node = start_node


def _voice_over_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @voice_over is applied."""
    audio_asset = params.get("audio_asset", "")
    lip_sync = params.get("lip_sync")

    obj._voice_over = True
    obj._voice_over_audio_asset = audio_asset
    obj._voice_over_lip_sync = lip_sync


# ============================================================================
# Decorator creation
# ============================================================================

dialogue = make_decorator(
    name="dialogue",
    steps=_dialogue_steps,
    doc="Dialogue node with id and optional speaker.",
    validate=_validate_dialogue_params,
    after_steps=_dialogue_after_apply,
)

conversation = make_decorator(
    name="conversation",
    steps=_conversation_steps,
    doc="Dialogue tree container with id and start node.",
    validate=_validate_conversation_params,
    after_steps=_conversation_after_apply,
)

voice_over = make_decorator(
    name="voice_over",
    steps=_voice_over_steps,
    doc="Voice acting with audio asset and optional lip sync.",
    validate=_validate_voice_over_params,
    after_steps=_voice_over_after_apply,
)


# ============================================================================
# Registry registration
# ============================================================================

_REGISTRY_ENTRIES = [
    ("dialogue", dialogue, ("class",)),
    ("conversation", conversation, ("class",)),
    ("voice_over", voice_over, ("class",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.NARRATIVE,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.NARRATIVE].append(_spec)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "dialogue",
    "conversation",
    "voice_over",
]
