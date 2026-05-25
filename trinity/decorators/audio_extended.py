"""
Trinity Pattern - Tier 49: AUDIO_EXTENDED Decorators

Advanced audio decorators for DSP, voice management, occlusion, reverb,
adaptive music, and mixing.
All decorators use the ops-based system via make_decorator().
"""

from __future__ import annotations

from typing import Any, Literal, TypeVar

from engine.audio.core.config import PRIORITY_NORMAL
from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

# Valid constants
VALID_OCCLUSION_METHODS = frozenset({"raycast", "propagation", "baked"})
VALID_MUSIC_TRANSITION_TYPES = frozenset(
    {"immediate", "next_beat", "next_bar", "crossfade"}
)

# Type variable for decorators
T = TypeVar("T")


# ============================================================================
# Step builders
# ============================================================================


def _dsp_node_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @dsp_node decorator."""
    inputs = params.get("inputs", 1)
    outputs = params.get("outputs", 1)
    latency_samples = params.get("latency_samples", 0)

    return [
        Step(Op.TAG, {"key": "dsp_node", "value": True}),
        Step(Op.TAG, {"key": "dsp_inputs", "value": inputs}),
        Step(Op.TAG, {"key": "dsp_outputs", "value": outputs}),
        Step(Op.TAG, {"key": "dsp_latency_samples", "value": latency_samples}),
        Step(Op.REGISTER, {"registry": "audio_extended"}),
    ]


def _voice_priority_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @voice_priority decorator."""
    priority = params.get("priority", PRIORITY_NORMAL)
    virtualize = params.get("virtualize", True)
    steal_oldest = params.get("steal_oldest", True)

    return [
        Step(Op.TAG, {"key": "voice_priority", "value": True}),
        Step(Op.TAG, {"key": "voice_priority_value", "value": priority}),
        Step(Op.TAG, {"key": "voice_virtualize", "value": virtualize}),
        Step(Op.TAG, {"key": "voice_steal_oldest", "value": steal_oldest}),
        Step(Op.REGISTER, {"registry": "audio_extended"}),
    ]


def _occlusion_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @occlusion decorator."""
    method = params.get("method", "raycast")
    max_occlusion = params.get("max_occlusion", 1.0)

    return [
        Step(Op.TAG, {"key": "occlusion", "value": True}),
        Step(Op.TAG, {"key": "occlusion_method", "value": method}),
        Step(Op.TAG, {"key": "occlusion_max", "value": max_occlusion}),
        Step(Op.REGISTER, {"registry": "audio_extended"}),
    ]


def _reverb_zone_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @reverb_zone decorator."""
    preset = params.get("preset")
    fade_distance = params.get("fade_distance", 5.0)

    return [
        Step(Op.TAG, {"key": "reverb_zone", "value": True}),
        Step(Op.TAG, {"key": "reverb_preset", "value": preset}),
        Step(Op.TAG, {"key": "reverb_fade_distance", "value": fade_distance}),
        Step(Op.REGISTER, {"registry": "audio_extended"}),
    ]


def _music_stem_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @music_stem decorator."""
    group = params.get("group", "")
    layer = params.get("layer", 0)
    sync_to_beat = params.get("sync_to_beat", True)

    return [
        Step(Op.TAG, {"key": "music_stem", "value": True}),
        Step(Op.TAG, {"key": "music_stem_group", "value": group}),
        Step(Op.TAG, {"key": "music_stem_layer", "value": layer}),
        Step(Op.TAG, {"key": "music_stem_sync_to_beat", "value": sync_to_beat}),
        Step(Op.REGISTER, {"registry": "audio_extended"}),
    ]


def _music_transition_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @music_transition decorator."""
    from_state = params.get("from_state", "")
    to_state = params.get("to_state", "")
    transition_type = params.get("type", "immediate")
    duration_beats = params.get("duration_beats", 0.0)

    return [
        Step(Op.TAG, {"key": "music_transition", "value": True}),
        Step(Op.TAG, {"key": "music_from_state", "value": from_state}),
        Step(Op.TAG, {"key": "music_to_state", "value": to_state}),
        Step(Op.TAG, {"key": "music_transition_type", "value": transition_type}),
        Step(Op.TAG, {"key": "music_duration_beats", "value": duration_beats}),
        Step(Op.REGISTER, {"registry": "audio_extended"}),
    ]


def _audio_snapshot_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @audio_snapshot decorator."""
    bus_overrides = params.get("bus_overrides", {})
    crossfade_time = params.get("crossfade_time", 0.5)

    return [
        Step(Op.TAG, {"key": "audio_snapshot", "value": True}),
        Step(Op.TAG, {"key": "snapshot_bus_overrides", "value": dict(bus_overrides)}),
        Step(Op.TAG, {"key": "snapshot_crossfade_time", "value": crossfade_time}),
        Step(Op.REGISTER, {"registry": "audio_extended"}),
    ]


def _sidechain_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @sidechain decorator."""
    source_bus = params.get("source_bus", "")
    attack = params.get("attack", 0.01)
    release = params.get("release", 0.1)
    ratio = params.get("ratio", 4.0)

    return [
        Step(Op.TAG, {"key": "sidechain", "value": True}),
        Step(Op.TAG, {"key": "sidechain_source_bus", "value": source_bus}),
        Step(Op.TAG, {"key": "sidechain_attack", "value": attack}),
        Step(Op.TAG, {"key": "sidechain_release", "value": release}),
        Step(Op.TAG, {"key": "sidechain_ratio", "value": ratio}),
        Step(Op.REGISTER, {"registry": "audio_extended"}),
    ]


# ============================================================================
# Validators
# ============================================================================


def _validate_dsp_node_params(**kwargs: Any) -> None:
    """Validate @dsp_node parameters."""
    inputs = kwargs.get("inputs", 1)
    if inputs <= 0:
        raise ValueError(f"inputs must be > 0, got {inputs}")

    outputs = kwargs.get("outputs", 1)
    if outputs <= 0:
        raise ValueError(f"outputs must be > 0, got {outputs}")

    latency_samples = kwargs.get("latency_samples", 0)
    if latency_samples < 0:
        raise ValueError(f"latency_samples must be >= 0, got {latency_samples}")


def _validate_voice_priority_params(**kwargs: Any) -> None:
    """Validate @voice_priority parameters."""
    priority = kwargs.get("priority", PRIORITY_NORMAL)
    if not isinstance(priority, int):
        raise ValueError(
            f"@voice_priority: 'priority' must be an int, "
            f"got {type(priority).__name__}"
        )
    if priority < 0 or priority > 100:
        raise ValueError(
            f"@voice_priority: 'priority' must be between 0 and 100, "
            f"got {priority}"
        )
    virtualize = kwargs.get("virtualize", True)
    if not isinstance(virtualize, bool):
        raise ValueError(
            f"@voice_priority: 'virtualize' must be a bool, "
            f"got {type(virtualize).__name__}"
        )
    steal_oldest = kwargs.get("steal_oldest", True)
    if not isinstance(steal_oldest, bool):
        raise ValueError(
            f"@voice_priority: 'steal_oldest' must be a bool, "
            f"got {type(steal_oldest).__name__}"
        )


def _validate_occlusion_params(**kwargs: Any) -> None:
    """Validate @occlusion parameters."""
    method = kwargs.get("method")
    if method not in VALID_OCCLUSION_METHODS:
        raise ValueError(
            f"Invalid method '{method}'. Must be one of {sorted(VALID_OCCLUSION_METHODS)}"
        )

    max_occlusion = kwargs.get("max_occlusion", 1.0)
    if not 0 <= max_occlusion <= 1:
        raise ValueError(f"max_occlusion must be between 0 and 1, got {max_occlusion}")


def _validate_reverb_zone_params(**kwargs: Any) -> None:
    """Validate @reverb_zone parameters."""
    fade_distance = kwargs.get("fade_distance", 5.0)
    if fade_distance <= 0:
        raise ValueError(f"fade_distance must be > 0, got {fade_distance}")


def _validate_music_stem_params(**kwargs: Any) -> None:
    """Validate @music_stem parameters."""
    group = kwargs.get("group")
    if not group:
        raise ValueError("group must be a non-empty string")

    layer = kwargs.get("layer", 0)
    if layer < 0:
        raise ValueError(f"layer must be >= 0, got {layer}")


def _validate_music_transition_params(**kwargs: Any) -> None:
    """Validate @music_transition parameters."""
    from_state = kwargs.get("from_state")
    if not from_state:
        raise ValueError("from_state must be a non-empty string")

    to_state = kwargs.get("to_state")
    if not to_state:
        raise ValueError("to_state must be a non-empty string")

    transition_type = kwargs.get("type")
    if transition_type not in VALID_MUSIC_TRANSITION_TYPES:
        raise ValueError(
            f"Invalid type '{transition_type}'. Must be one of {sorted(VALID_MUSIC_TRANSITION_TYPES)}"
        )

    duration_beats = kwargs.get("duration_beats", 0.0)
    if duration_beats < 0:
        raise ValueError(f"duration_beats must be >= 0, got {duration_beats}")


def _validate_audio_snapshot_params(**kwargs: Any) -> None:
    """Validate @audio_snapshot parameters."""
    bus_overrides = kwargs.get("bus_overrides")
    if not bus_overrides:
        raise ValueError("bus_overrides must be a non-empty dict")

    crossfade_time = kwargs.get("crossfade_time", 0.5)
    if crossfade_time < 0:
        raise ValueError(f"crossfade_time must be >= 0, got {crossfade_time}")


def _validate_sidechain_params(**kwargs: Any) -> None:
    """Validate @sidechain parameters."""
    source_bus = kwargs.get("source_bus")
    if not source_bus:
        raise ValueError("source_bus must be a non-empty string")

    attack = kwargs.get("attack", 0.01)
    if attack <= 0:
        raise ValueError(f"attack must be > 0, got {attack}")

    release = kwargs.get("release", 0.1)
    if release <= 0:
        raise ValueError(f"release must be > 0, got {release}")

    ratio = kwargs.get("ratio", 4.0)
    if ratio < 1:
        raise ValueError(f"ratio must be >= 1, got {ratio}")


# ============================================================================
# After-apply functions
# ============================================================================


def _dsp_node_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @dsp_node is applied."""
    obj._dsp_node = True
    obj._dsp_inputs = params.get("inputs", 1)
    obj._dsp_outputs = params.get("outputs", 1)
    obj._dsp_latency_samples = params.get("latency_samples", 0)


def _voice_priority_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @voice_priority is applied."""
    obj._voice_priority = True
    obj._voice_priority_value = params.get("priority", PRIORITY_NORMAL)
    obj._voice_virtualize = params.get("virtualize", True)
    obj._voice_steal_oldest = params.get("steal_oldest", True)


def _occlusion_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @occlusion is applied."""
    obj._occlusion = True
    obj._occlusion_method = params.get("method", "raycast")
    obj._occlusion_max = params.get("max_occlusion", 1.0)


def _reverb_zone_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @reverb_zone is applied."""
    obj._reverb_zone = True
    obj._reverb_preset = params.get("preset")
    obj._reverb_fade_distance = params.get("fade_distance", 5.0)


def _music_stem_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @music_stem is applied."""
    obj._music_stem = True
    obj._music_stem_group = params.get("group", "")
    obj._music_stem_layer = params.get("layer", 0)
    obj._music_stem_sync_to_beat = params.get("sync_to_beat", True)


def _music_transition_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @music_transition is applied."""
    obj._music_transition = True
    obj._music_from_state = params.get("from_state", "")
    obj._music_to_state = params.get("to_state", "")
    obj._music_transition_type = params.get("type", "immediate")
    obj._music_duration_beats = params.get("duration_beats", 0.0)


def _audio_snapshot_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @audio_snapshot is applied."""
    obj._audio_snapshot = True
    obj._snapshot_bus_overrides = dict(params.get("bus_overrides", {}))
    obj._snapshot_crossfade_time = params.get("crossfade_time", 0.5)


def _sidechain_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @sidechain is applied."""
    obj._sidechain = True
    obj._sidechain_source_bus = params.get("source_bus", "")
    obj._sidechain_attack = params.get("attack", 0.01)
    obj._sidechain_release = params.get("release", 0.1)
    obj._sidechain_ratio = params.get("ratio", 4.0)


# ============================================================================
# Decorator creation
# ============================================================================

dsp_node = make_decorator(
    name="dsp_node",
    steps=_dsp_node_steps,
    doc="DSP processing node with configurable inputs, outputs, and latency.",
    validate=_validate_dsp_node_params,
    after_steps=_dsp_node_after_apply,
)

voice_priority = make_decorator(
    name="voice_priority",
    steps=_voice_priority_steps,
    doc="Voice management with priority-based virtualization and stealing.",
    validate=_validate_voice_priority_params,
    after_steps=_voice_priority_after_apply,
)

occlusion = make_decorator(
    name="occlusion",
    steps=_occlusion_steps,
    doc="Sound occlusion with raycast, propagation, or baked methods.",
    validate=_validate_occlusion_params,
    after_steps=_occlusion_after_apply,
)

reverb_zone = make_decorator(
    name="reverb_zone",
    steps=_reverb_zone_steps,
    doc="Reverb environment with preset and distance-based fading.",
    validate=_validate_reverb_zone_params,
    after_steps=_reverb_zone_after_apply,
)

music_stem = make_decorator(
    name="music_stem",
    steps=_music_stem_steps,
    doc="Adaptive music stem with group, layer, and beat synchronization.",
    validate=_validate_music_stem_params,
    after_steps=_music_stem_after_apply,
)

music_transition = make_decorator(
    name="music_transition",
    steps=_music_transition_steps,
    doc="Music state transition with timing and crossfade control.",
    validate=_validate_music_transition_params,
    after_steps=_music_transition_after_apply,
)

audio_snapshot = make_decorator(
    name="audio_snapshot",
    steps=_audio_snapshot_steps,
    doc="Mixer snapshot with bus overrides and crossfade timing.",
    validate=_validate_audio_snapshot_params,
    after_steps=_audio_snapshot_after_apply,
)

sidechain = make_decorator(
    name="sidechain",
    steps=_sidechain_steps,
    doc="Sidechain compression with configurable attack, release, and ratio.",
    validate=_validate_sidechain_params,
    after_steps=_sidechain_after_apply,
)


# ============================================================================
# Registry registration
# ============================================================================

_REGISTRY_ENTRIES = [
    ("dsp_node", dsp_node, ("class",)),
    ("voice_priority", voice_priority, ("class",)),
    ("occlusion", occlusion, ("class",)),
    ("reverb_zone", reverb_zone, ("class",)),
    ("music_stem", music_stem, ("class",)),
    ("music_transition", music_transition, ("class",)),
    ("audio_snapshot", audio_snapshot, ("class",)),
    ("sidechain", sidechain, ("class",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.AUDIO_EXTENDED,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.AUDIO_EXTENDED].append(_spec)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "dsp_node",
    "voice_priority",
    "occlusion",
    "reverb_zone",
    "music_stem",
    "music_transition",
    "audio_snapshot",
    "sidechain",
    "VALID_OCCLUSION_METHODS",
    "VALID_MUSIC_TRANSITION_TYPES",
]
