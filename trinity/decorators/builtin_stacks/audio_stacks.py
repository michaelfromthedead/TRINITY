"""Audio domain composite stacks."""
from __future__ import annotations
from trinity.decorators.stacks import Stack, parameterized_stack, stack


@parameterized_stack
def adaptive_audio(
    crossfade_time: float = 0.5,
    stem_group: str = "music",
) -> Stack:
    """Adaptive music with stems, snapshots, and transitions."""
    from trinity.decorators.audio_extended import audio_snapshot, music_stem, music_transition
    from trinity.decorators.data_flow import serializable
    return stack(
        music_stem(group=stem_group),
        music_transition(from_state="explore", to_state="combat", type="crossfade"),
        audio_snapshot(bus_overrides={"master": 0.8}, crossfade_time=crossfade_time),
        serializable(format="binary"),
    )


__all__ = ["adaptive_audio"]
