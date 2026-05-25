"""
Virtual Voice Tracker

Tracks virtualized voices: position/time preservation, urgency scoring,
and promotion logic for the VoiceManager.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .config import (
    VIRTUAL_VOICE_MAX_TIME_SECONDS,
    VIRTUAL_VOICE_URGENCY_PRIORITY_WEIGHT,
    VIRTUAL_VOICE_URGENCY_TIME_WEIGHT,
    VIRTUAL_VOICE_URGENCY_RISE_WEIGHT,
    VIRTUAL_VOICE_FORCE_PROMOTE_GRACE_MS,
    AudioCategory,
)
from .audio_source import AudioSource


@dataclass
class VirtualVoiceState:
    """
    State snapshot for a virtualized voice.

    Captures sample position at virtualization time and tracks priority
    changes so the tracker can compute urgency for promotion ordering.
    """

    voice_id: int
    source_id: str
    position_samples: int = 0
    position_at_virtualization: int = 0
    virtual_start_time: float = 0.0
    last_update_time: float = 0.0
    accumulated_virtual_time: float = 0.0
    priority_at_virtualization: int = 50
    current_priority: int = 50
    peak_priority: int = 50
    urgency_score: float = 0.0
    category: AudioCategory = AudioCategory.SFX
    force_promote: bool = False


class VirtualVoiceTracker:
    """
    Tracks every virtualized voice, advances its sample position on each
    update tick, computes urgency, and supplies promotion candidates.

    Stats are aggregated so the VoiceManager can report them in get_stats().
    """

    def __init__(self) -> None:
        self._states: dict[int, VirtualVoiceState] = {}

        # Aggregate stats
        self.total_virtualized: int = 0
        self.total_promoted: int = 0
        self.total_force_promoted: int = 0
        self.peak_virtual_count: int = 0
        self._last_promotion_time: float = 0.0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def track_virtualization(
        self,
        voice_id: int,
        source: AudioSource,
        position_samples: int,
    ) -> None:
        """
        Register a newly-virtualized voice.

        Args:
            voice_id: The voice slot that was freed.
            source: The AudioSource that was virtualized.
            position_samples: Current playback sample position.
        """
        priority = source.priority
        now = time.time()
        state = VirtualVoiceState(
            voice_id=voice_id,
            source_id=source.id,
            position_samples=position_samples,
            position_at_virtualization=position_samples,
            virtual_start_time=now,
            last_update_time=now,
            accumulated_virtual_time=0.0,
            priority_at_virtualization=priority,
            current_priority=priority,
            peak_priority=priority,
            urgency_score=0.0,
            category=source.category,
            force_promote=False,
        )
        self._states[voice_id] = state
        self.total_virtualized += 1
        self.peak_virtual_count = max(self.peak_virtual_count, len(self._states))

    # ------------------------------------------------------------------
    # Per-tick update
    # ------------------------------------------------------------------

    def update(self, delta_time: float, source_lookup: dict[int, AudioSource]) -> None:
        """
        Advance sample positions for all tracked virtual voices and
        recompute urgency scores.

        Args:
            delta_time: Seconds since last tick.
            source_lookup: Mapping of voice_id -> AudioSource for currently
                           virtual voices.
        """
        if delta_time <= 0:
            return

        delta_ms = delta_time * 1000.0
        now = time.time()

        to_remove: list[int] = []

        for voice_id, state in self._states.items():
            source = source_lookup.get(voice_id)

            if source is None:
                # Source no longer tracked — mark for removal
                to_remove.append(voice_id)
                continue

            # Accumulate virtual time
            elapsed_since_update = now - state.last_update_time
            state.accumulated_virtual_time += max(0.0, elapsed_since_update)
            state.last_update_time = now

            # Advance sample position (pitch-adjusted)
            sample_rate = source.clip.sample_rate if source.clip else 48000
            pitch = source.pitch
            samples_per_ms = (sample_rate / 1000.0) * pitch
            samples_advanced = int(delta_ms * samples_per_ms)
            state.position_samples += samples_advanced

            # Track priority changes
            old_priority = state.current_priority
            state.current_priority = source.priority
            state.peak_priority = max(state.peak_priority, state.current_priority)

            # Compute urgency
            state.urgency_score = self._compute_urgency(state)

            # Force-promote check
            if state.accumulated_virtual_time >= VIRTUAL_VOICE_MAX_TIME_SECONDS:
                state.force_promote = True

        for vid in to_remove:
            self._states.pop(vid, None)

    # ------------------------------------------------------------------
    # Urgency computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_urgency(state: VirtualVoiceState) -> float:
        """
        Compute a promotion-urgency score for a virtual voice.

        Higher values = should be promoted sooner.  Three components:

        1. **Priority score** — how important this voice is
        2. **Time score** — quadratic ramp so urgency accelerates
        3. **Priority-rise score** — voices whose importance grew
           while virtual should jump the queue

        Force-promote returns infinity.
        """
        if state.force_promote:
            return float("inf")

        max_time = VIRTUAL_VOICE_MAX_TIME_SECONDS
        time_ratio = min(state.accumulated_virtual_time / max_time, 1.0)

        # Priority component (normalised to 0-1)
        priority_score = state.current_priority / 100.0

        # Quadratic time component
        time_score = time_ratio * time_ratio

        # Priority-rise component
        rise = state.peak_priority - state.priority_at_virtualization
        priority_rise_score = min(max(rise / 100.0, 0.0), 1.0)

        urgency = (
            VIRTUAL_VOICE_URGENCY_PRIORITY_WEIGHT * priority_score
            + VIRTUAL_VOICE_URGENCY_TIME_WEIGHT * time_score
            + VIRTUAL_VOICE_URGENCY_RISE_WEIGHT * priority_rise_score
        )

        return urgency

    # ------------------------------------------------------------------
    # Promotion
    # ------------------------------------------------------------------

    def get_promotion_candidates(self, max_count: int) -> list[VirtualVoiceState]:
        """
        Return up to *max_count* virtual voices sorted by descending
        urgency (most urgent first).

        Force-promote candidates (urgency = inf) always appear at the
        front, sorted by accumulated virtual time descending.
        """
        if not self._states:
            return []

        candidates = sorted(
            self._states.values(),
            key=lambda s: (
                0 if s.force_promote else 1,   # force-promote first
                -s.accumulated_virtual_time if s.force_promote else -s.urgency_score,
            ),
        )
        return candidates[:max_count]

    def on_promoted(self, voice_id: int) -> None:
        """
        Called when a virtual voice has been successfully promoted back
        to a real voice.
        """
        state = self._states.pop(voice_id, None)
        self.total_promoted += 1
        if state is not None and state.force_promote:
            self.total_force_promoted += 1

    def on_released(self, voice_id: int) -> None:
        """
        Called when a virtual voice is released (stopped / stolen).

        Does NOT count as a promotion.  Removes the state entry.
        """
        self._states.pop(voice_id, None)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_state(self, voice_id: int) -> Optional[VirtualVoiceState]:
        """Return the tracked state for a voice, or None."""
        return self._states.get(voice_id)

    @property
    def virtual_count(self) -> int:
        return len(self._states)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """
        Return aggregate + per-voice diagnostic data.

        The VoiceManager merges this into its own get_stats() output.
        """
        per_voice = []
        for state in sorted(
            self._states.values(),
            key=lambda s: -s.urgency_score,
        ):
            per_voice.append({
                "voice_id": state.voice_id,
                "source_id": state.source_id,
                "position_samples": state.position_samples,
                "virtual_time_s": round(state.accumulated_virtual_time, 3),
                "priority": state.current_priority,
                "peak_priority": state.peak_priority,
                "urgency": round(state.urgency_score, 3),
                "force_promote": state.force_promote,
                "category": state.category.name,
            })

        return {
            "total_virtualized": self.total_virtualized,
            "total_promoted": self.total_promoted,
            "total_force_promoted": self.total_force_promoted,
            "currently_virtual": len(self._states),
            "peak_virtual_count": self.peak_virtual_count,
            "per_voice": per_voice,
        }
