"""
Lip Synchronization Module.

Provides phoneme-to-viseme mapping and lip sync animation control
for speech synthesis and audio-driven facial animation.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, Sequence


# =============================================================================
# Viseme Definitions
# =============================================================================


class Viseme(Enum):
    """
    Standard viseme set for lip synchronization.

    Based on the Preston Blair phoneme chart and common speech synthesis standards.
    """
    SIL = auto()   # Silence / rest position
    PP = auto()    # p, b, m (bilabial)
    FF = auto()    # f, v (labiodental)
    TH = auto()    # th (dental fricative)
    DD = auto()    # t, d, n, l (alveolar)
    KK = auto()    # k, g, ng (velar)
    CH = auto()    # ch, j, sh, zh (palato-alveolar)
    SS = auto()    # s, z (alveolar fricative)
    NN = auto()    # n (nasal)
    RR = auto()    # r (approximant)
    AA = auto()    # a, ah (open vowel)
    EE = auto()    # e, eh (front mid vowel)
    II = auto()    # i, ee (front close vowel)
    OO = auto()    # o, oh (back mid vowel)
    UU = auto()    # u, oo (back close vowel)


# =============================================================================
# Phoneme to Viseme Mapping
# =============================================================================


# IPA and common phoneme representations to viseme mapping
PHONEME_TO_VISEME: dict[str, Viseme] = {
    # Silence
    "sil": Viseme.SIL,
    "sp": Viseme.SIL,
    "": Viseme.SIL,

    # Bilabials (p, b, m)
    "p": Viseme.PP,
    "b": Viseme.PP,
    "m": Viseme.PP,
    "P": Viseme.PP,
    "B": Viseme.PP,
    "M": Viseme.PP,

    # Labiodentals (f, v)
    "f": Viseme.FF,
    "v": Viseme.FF,
    "F": Viseme.FF,
    "V": Viseme.FF,

    # Dental fricatives (th)
    "th": Viseme.TH,
    "dh": Viseme.TH,
    "TH": Viseme.TH,
    "DH": Viseme.TH,

    # Alveolar (t, d)
    "t": Viseme.DD,
    "d": Viseme.DD,
    "T": Viseme.DD,
    "D": Viseme.DD,

    # Alveolar nasal
    "n": Viseme.NN,
    "N": Viseme.NN,

    # Alveolar lateral
    "l": Viseme.DD,
    "L": Viseme.DD,

    # Alveolar fricative (s, z)
    "s": Viseme.SS,
    "z": Viseme.SS,
    "S": Viseme.SS,
    "Z": Viseme.SS,

    # Palato-alveolar (sh, zh, ch, j)
    "sh": Viseme.CH,
    "zh": Viseme.CH,
    "ch": Viseme.CH,
    "jh": Viseme.CH,
    "SH": Viseme.CH,
    "ZH": Viseme.CH,
    "CH": Viseme.CH,
    "JH": Viseme.CH,

    # Velar (k, g, ng)
    "k": Viseme.KK,
    "g": Viseme.KK,
    "ng": Viseme.KK,
    "K": Viseme.KK,
    "G": Viseme.KK,
    "NG": Viseme.KK,

    # Approximant (r)
    "r": Viseme.RR,
    "R": Viseme.RR,
    "er": Viseme.RR,
    "ER": Viseme.RR,

    # Glides (w, y)
    "w": Viseme.UU,
    "y": Viseme.II,
    "W": Viseme.UU,
    "Y": Viseme.II,

    # Glottal
    "h": Viseme.SIL,
    "hh": Viseme.SIL,
    "H": Viseme.SIL,
    "HH": Viseme.SIL,

    # Vowels - Open
    "aa": Viseme.AA,
    "ah": Viseme.AA,
    "ae": Viseme.AA,
    "ax": Viseme.AA,
    "AA": Viseme.AA,
    "AH": Viseme.AA,
    "AE": Viseme.AA,
    "AX": Viseme.AA,

    # Vowels - Front mid
    "eh": Viseme.EE,
    "ey": Viseme.EE,
    "EH": Viseme.EE,
    "EY": Viseme.EE,

    # Vowels - Front close
    "iy": Viseme.II,
    "ih": Viseme.II,
    "IY": Viseme.II,
    "IH": Viseme.II,

    # Vowels - Back mid
    "ao": Viseme.OO,
    "ow": Viseme.OO,
    "oy": Viseme.OO,
    "AO": Viseme.OO,
    "OW": Viseme.OO,
    "OY": Viseme.OO,

    # Vowels - Back close
    "uw": Viseme.UU,
    "uh": Viseme.UU,
    "UW": Viseme.UU,
    "UH": Viseme.UU,

    # Diphthongs
    "aw": Viseme.AA,  # Start of diphthong
    "ay": Viseme.AA,
    "AW": Viseme.AA,
    "AY": Viseme.AA,
}


def phoneme_to_viseme(phoneme: str) -> Viseme:
    """
    Convert a phoneme to its corresponding viseme.

    Args:
        phoneme: Phoneme string (IPA or common representation)

    Returns:
        Corresponding Viseme
    """
    # Strip stress markers if present (e.g., "AA1" -> "AA")
    clean_phoneme = phoneme.rstrip("0123456789")
    return PHONEME_TO_VISEME.get(clean_phoneme, Viseme.SIL)


# =============================================================================
# Viseme to Blend Shape Mapping
# =============================================================================


@dataclass
class VisemeMapping:
    """
    Mapping from visemes to blend shape weights.

    Attributes:
        viseme: The viseme
        blend_shapes: Dictionary of blend shape names to weights
    """
    viseme: Viseme
    blend_shapes: dict[str, float] = field(default_factory=dict)

    def get_weights(self, intensity: float = 1.0) -> dict[str, float]:
        """
        Get blend shape weights scaled by intensity.

        Args:
            intensity: Intensity multiplier (0-1)

        Returns:
            Scaled blend shape weights
        """
        return {name: weight * intensity for name, weight in self.blend_shapes.items()}


def get_default_viseme_mappings() -> dict[Viseme, VisemeMapping]:
    """
    Get default viseme to blend shape mappings.

    Uses ARKit-compatible blend shape names.

    Returns:
        Dictionary of viseme mappings
    """
    return {
        Viseme.SIL: VisemeMapping(
            viseme=Viseme.SIL,
            blend_shapes={
                "jawOpen": 0.0,
                "mouthClose": 0.0,
            },
        ),
        Viseme.PP: VisemeMapping(
            viseme=Viseme.PP,
            blend_shapes={
                "mouthClose": 0.8,
                "mouthPucker": 0.3,
                "mouthPressLeft": 0.5,
                "mouthPressRight": 0.5,
            },
        ),
        Viseme.FF: VisemeMapping(
            viseme=Viseme.FF,
            blend_shapes={
                "mouthFunnel": 0.3,
                "mouthLowerDownLeft": 0.2,
                "mouthLowerDownRight": 0.2,
            },
        ),
        Viseme.TH: VisemeMapping(
            viseme=Viseme.TH,
            blend_shapes={
                "jawOpen": 0.15,
                "tongueOut": 0.3,
            },
        ),
        Viseme.DD: VisemeMapping(
            viseme=Viseme.DD,
            blend_shapes={
                "jawOpen": 0.2,
                "mouthClose": 0.0,
            },
        ),
        Viseme.KK: VisemeMapping(
            viseme=Viseme.KK,
            blend_shapes={
                "jawOpen": 0.25,
                "mouthClose": 0.0,
            },
        ),
        Viseme.CH: VisemeMapping(
            viseme=Viseme.CH,
            blend_shapes={
                "jawOpen": 0.15,
                "mouthFunnel": 0.4,
                "mouthPucker": 0.3,
            },
        ),
        Viseme.SS: VisemeMapping(
            viseme=Viseme.SS,
            blend_shapes={
                "jawOpen": 0.1,
                "mouthStretchLeft": 0.3,
                "mouthStretchRight": 0.3,
            },
        ),
        Viseme.NN: VisemeMapping(
            viseme=Viseme.NN,
            blend_shapes={
                "jawOpen": 0.15,
                "mouthClose": 0.3,
            },
        ),
        Viseme.RR: VisemeMapping(
            viseme=Viseme.RR,
            blend_shapes={
                "jawOpen": 0.2,
                "mouthFunnel": 0.3,
            },
        ),
        Viseme.AA: VisemeMapping(
            viseme=Viseme.AA,
            blend_shapes={
                "jawOpen": 0.6,
                "mouthFunnel": 0.0,
            },
        ),
        Viseme.EE: VisemeMapping(
            viseme=Viseme.EE,
            blend_shapes={
                "jawOpen": 0.3,
                "mouthSmileLeft": 0.4,
                "mouthSmileRight": 0.4,
            },
        ),
        Viseme.II: VisemeMapping(
            viseme=Viseme.II,
            blend_shapes={
                "jawOpen": 0.2,
                "mouthSmileLeft": 0.5,
                "mouthSmileRight": 0.5,
                "mouthStretchLeft": 0.3,
                "mouthStretchRight": 0.3,
            },
        ),
        Viseme.OO: VisemeMapping(
            viseme=Viseme.OO,
            blend_shapes={
                "jawOpen": 0.4,
                "mouthFunnel": 0.5,
                "mouthPucker": 0.3,
            },
        ),
        Viseme.UU: VisemeMapping(
            viseme=Viseme.UU,
            blend_shapes={
                "jawOpen": 0.25,
                "mouthFunnel": 0.6,
                "mouthPucker": 0.5,
            },
        ),
    }


# =============================================================================
# Phoneme Events
# =============================================================================


@dataclass
class PhonemeEvent:
    """
    A phoneme event in an audio timeline.

    Attributes:
        phoneme: The phoneme string
        start_time: Start time in seconds
        end_time: End time in seconds
        confidence: Recognition confidence (0-1)
    """
    phoneme: str
    start_time: float
    end_time: float
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        return self.end_time - self.start_time

    @property
    def mid_time(self) -> float:
        """Get midpoint time."""
        return (self.start_time + self.end_time) / 2.0


@dataclass
class VisemeEvent:
    """
    A viseme event in a lip sync timeline.

    Attributes:
        viseme: The viseme
        start_time: Start time in seconds
        end_time: End time in seconds
        weight: Viseme weight/intensity
    """
    viseme: Viseme
    start_time: float
    end_time: float
    weight: float = 1.0

    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        return self.end_time - self.start_time


# =============================================================================
# Coarticulation
# =============================================================================


@dataclass
class CoarticulationSettings:
    """
    Settings for coarticulation (phoneme blending).

    Attributes:
        anticipation_time: Time to start anticipating next phoneme
        carryover_time: Time for previous phoneme influence
        blend_curve: Blend curve type ("linear", "ease_in_out", "ease_in", "ease_out")
    """
    anticipation_time: float = 0.05
    carryover_time: float = 0.03
    blend_curve: str = "ease_in_out"

    def calculate_blend(self, t: float) -> float:
        """
        Calculate blend value for a normalized time.

        Args:
            t: Normalized time (0-1)

        Returns:
            Blend value (0-1)
        """
        t = max(0.0, min(1.0, t))

        if self.blend_curve == "ease_in_out":
            # Smoothstep
            return t * t * (3.0 - 2.0 * t)
        elif self.blend_curve == "ease_in":
            return t * t
        elif self.blend_curve == "ease_out":
            return 1.0 - (1.0 - t) * (1.0 - t)
        else:  # linear
            return t


def apply_coarticulation(
    events: list[VisemeEvent],
    settings: CoarticulationSettings,
) -> list[tuple[float, Viseme, float, Viseme, float]]:
    """
    Apply coarticulation to viseme events.

    Returns timeline of (time, prev_viseme, prev_weight, next_viseme, next_weight).

    Args:
        events: List of viseme events
        settings: Coarticulation settings

    Returns:
        List of blended viseme states
    """
    if not events:
        return []

    result = []
    sorted_events = sorted(events, key=lambda e: e.start_time)

    for i, event in enumerate(sorted_events):
        prev_event = sorted_events[i - 1] if i > 0 else None
        next_event = sorted_events[i + 1] if i < len(sorted_events) - 1 else None

        # Sample multiple points within the event
        num_samples = max(2, int(event.duration * 100))  # 100 samples per second
        dt = event.duration / num_samples

        for s in range(num_samples + 1):
            sample_time = event.start_time + s * dt

            # Calculate influence from previous phoneme (carryover)
            prev_weight = 0.0
            if prev_event:
                carryover_end = event.start_time + settings.carryover_time
                if sample_time < carryover_end:
                    t = (sample_time - event.start_time) / settings.carryover_time
                    prev_weight = (1.0 - settings.calculate_blend(t)) * prev_event.weight

            # Calculate influence from next phoneme (anticipation)
            next_weight = 0.0
            if next_event:
                anticipation_start = event.end_time - settings.anticipation_time
                if sample_time > anticipation_start:
                    t = (sample_time - anticipation_start) / settings.anticipation_time
                    next_weight = settings.calculate_blend(t) * next_event.weight

            # Current phoneme weight
            current_weight = event.weight - prev_weight - next_weight
            current_weight = max(0.0, current_weight)

            prev_viseme = prev_event.viseme if prev_event else Viseme.SIL
            next_viseme = next_event.viseme if next_event else Viseme.SIL

            result.append((
                sample_time,
                prev_viseme, prev_weight,
                event.viseme, current_weight,
                next_viseme, next_weight,
            ))

    return result


# =============================================================================
# Lip Sync Controller
# =============================================================================


class LipSyncController:
    """
    Controller for lip synchronization animation.

    Processes phoneme events and outputs blend shape weights.
    """

    def __init__(
        self,
        viseme_mappings: Optional[dict[Viseme, VisemeMapping]] = None,
        coarticulation_settings: Optional[CoarticulationSettings] = None,
        on_weights_changed: Optional[Callable[[dict[str, float]], None]] = None,
    ) -> None:
        """
        Initialize the lip sync controller.

        Args:
            viseme_mappings: Custom viseme to blend shape mappings
            coarticulation_settings: Coarticulation settings
            on_weights_changed: Callback when blend weights change
        """
        self._viseme_mappings = viseme_mappings or get_default_viseme_mappings()
        self._coarticulation = coarticulation_settings or CoarticulationSettings()
        self._on_weights_changed = on_weights_changed

        # Timeline
        self._viseme_timeline: list[VisemeEvent] = []
        self._timeline_times: list[float] = []  # For binary search

        # Current state
        self._current_time: float = 0.0
        self._current_viseme: Viseme = Viseme.SIL
        self._current_weights: dict[str, float] = {}
        self._is_playing: bool = False

        # Blend settings
        self._blend_time: float = 0.05  # Time to blend between visemes
        self._intensity: float = 1.0

        self._dirty = False

    @property
    def current_viseme(self) -> Viseme:
        """Get current viseme."""
        return self._current_viseme

    @property
    def current_time(self) -> float:
        """Get current playback time."""
        return self._current_time

    @property
    def is_playing(self) -> bool:
        """Check if lip sync is playing."""
        return self._is_playing

    @property
    def duration(self) -> float:
        """Get timeline duration."""
        if not self._viseme_timeline:
            return 0.0
        return self._viseme_timeline[-1].end_time

    @property
    def intensity(self) -> float:
        """Get animation intensity."""
        return self._intensity

    @intensity.setter
    def intensity(self, value: float) -> None:
        """Set animation intensity."""
        self._intensity = max(0.0, min(1.0, value))

    @property
    def blend_time(self) -> float:
        """Get blend time between visemes."""
        return self._blend_time

    @blend_time.setter
    def blend_time(self, value: float) -> None:
        """Set blend time between visemes."""
        self._blend_time = max(0.0, value)

    def process_audio_events(
        self,
        phoneme_events: Sequence[PhonemeEvent],
    ) -> list[VisemeEvent]:
        """
        Process phoneme events to create viseme timeline.

        Args:
            phoneme_events: List of phoneme events from audio analysis

        Returns:
            List of viseme events
        """
        viseme_events = []

        for event in phoneme_events:
            viseme = phoneme_to_viseme(event.phoneme)

            viseme_events.append(VisemeEvent(
                viseme=viseme,
                start_time=event.start_time,
                end_time=event.end_time,
                weight=event.confidence,
            ))

        return viseme_events

    def set_timeline(self, viseme_events: Sequence[VisemeEvent]) -> None:
        """
        Set the viseme timeline.

        Args:
            viseme_events: List of viseme events
        """
        self._viseme_timeline = sorted(list(viseme_events), key=lambda e: e.start_time)
        self._timeline_times = [e.start_time for e in self._viseme_timeline]
        self._current_time = 0.0
        self._is_playing = False
        self._dirty = True

    def set_phoneme_timeline(self, phoneme_events: Sequence[PhonemeEvent]) -> None:
        """
        Set timeline from phoneme events.

        Args:
            phoneme_events: List of phoneme events
        """
        viseme_events = self.process_audio_events(phoneme_events)
        self.set_timeline(viseme_events)

    def add_phoneme_event(self, event: PhonemeEvent) -> None:
        """
        Add a single phoneme event to the timeline.

        Args:
            event: The phoneme event to add
        """
        viseme = phoneme_to_viseme(event.phoneme)
        viseme_event = VisemeEvent(
            viseme=viseme,
            start_time=event.start_time,
            end_time=event.end_time,
            weight=event.confidence,
        )
        # Insert in sorted order
        idx = bisect.bisect_left(self._timeline_times, event.start_time)
        self._viseme_timeline.insert(idx, viseme_event)
        self._timeline_times.insert(idx, event.start_time)
        self._dirty = True

    def get_viseme_events(self) -> list[VisemeEvent]:
        """
        Get the current viseme timeline.

        Returns:
            List of viseme events in the timeline
        """
        return list(self._viseme_timeline)

    def play(self) -> None:
        """Start or resume playback."""
        self._is_playing = True

    def pause(self) -> None:
        """Pause playback."""
        self._is_playing = False

    def stop(self) -> None:
        """Stop playback and reset to beginning."""
        self._is_playing = False
        self._current_time = 0.0
        self._current_viseme = Viseme.SIL
        self._current_weights = {}
        self._dirty = True
        self._notify_change()

    def seek(self, time: float) -> None:
        """
        Seek to a specific time.

        Args:
            time: Time in seconds
        """
        self._current_time = max(0.0, time)
        self._update_viseme_at_time(self._current_time)

    def update(self, dt: Optional[float] = None, *, time: Optional[float] = None) -> dict[str, float]:
        """
        Update lip sync animation.

        Can be called with either delta time (dt) or absolute time.
        If 'time' is provided, seeks to that time and returns weights.
        If 'dt' is provided, advances playback by that amount.
        If neither is provided and playing, returns current weights.

        Args:
            dt: Delta time in seconds (optional)
            time: Absolute time in seconds (optional, keyword-only)

        Returns:
            Current blend shape weights
        """
        # If absolute time provided, seek directly
        if time is not None:
            return self._update_viseme_at_time(time)

        # If no timeline, return empty/current weights
        if not self._viseme_timeline:
            return self._current_weights

        # If delta time provided but not playing, still update for convenience
        if dt is not None:
            if not self._is_playing:
                # Allow direct time-based queries even when not playing
                return self._update_viseme_at_time(self._current_time + dt)
            self._current_time += dt
        elif not self._is_playing:
            return self._current_weights

        # Check if we've finished
        if self._current_time >= self.duration:
            self._current_time = self.duration
            self._is_playing = False
            self._current_viseme = Viseme.SIL
            self._current_weights = {}
            self._dirty = True
            self._notify_change()
            return self._current_weights

        return self._update_viseme_at_time(self._current_time)

    def _update_viseme_at_time(self, time: float) -> dict[str, float]:
        """
        Update viseme state for a specific time.

        Args:
            time: Current time

        Returns:
            Blend shape weights
        """
        if not self._viseme_timeline:
            return {}

        # Find current and adjacent viseme events
        idx = bisect.bisect_right(self._timeline_times, time) - 1
        idx = max(0, min(idx, len(self._viseme_timeline) - 1))

        current_event = self._viseme_timeline[idx]
        prev_event = self._viseme_timeline[idx - 1] if idx > 0 else None
        next_event = self._viseme_timeline[idx + 1] if idx < len(self._viseme_timeline) - 1 else None

        # Calculate blended weights with coarticulation
        weights: dict[str, float] = {}

        # Current viseme contribution
        current_mapping = self._viseme_mappings.get(current_event.viseme)
        if current_mapping:
            time_in_event = time - current_event.start_time
            event_duration = current_event.duration

            # Calculate fade in/out for current viseme
            # Guard against zero or negative duration
            min_duration = 0.001  # 1ms minimum
            if event_duration < min_duration:
                # Zero/negative duration: apply full weight instantly
                current_weight = current_event.weight * self._intensity
            elif self._blend_time <= 0:
                # No blending: full weight
                current_weight = current_event.weight * self._intensity
            else:
                # Normal case: calculate fade in/out
                fade_in = min(1.0, time_in_event / self._blend_time)
                time_remaining = event_duration - time_in_event
                fade_out = min(1.0, time_remaining / self._blend_time)
                current_weight = min(fade_in, fade_out) * current_event.weight * self._intensity

            for shape_name, shape_weight in current_mapping.blend_shapes.items():
                weights[shape_name] = weights.get(shape_name, 0.0) + shape_weight * current_weight

        # Carryover from previous viseme
        carryover_time = self._coarticulation.carryover_time
        if prev_event and carryover_time > 0 and time < current_event.start_time + carryover_time:
            prev_mapping = self._viseme_mappings.get(prev_event.viseme)
            if prev_mapping:
                t = (time - current_event.start_time) / carryover_time
                carryover_weight = (1.0 - self._coarticulation.calculate_blend(t)) * prev_event.weight * self._intensity

                for shape_name, shape_weight in prev_mapping.blend_shapes.items():
                    weights[shape_name] = weights.get(shape_name, 0.0) + shape_weight * carryover_weight

        # Anticipation of next viseme
        anticipation_time = self._coarticulation.anticipation_time
        if next_event and anticipation_time > 0:
            anticipation_start = current_event.end_time - anticipation_time
            if time > anticipation_start:
                next_mapping = self._viseme_mappings.get(next_event.viseme)
                if next_mapping:
                    t = (time - anticipation_start) / anticipation_time
                    anticipation_weight = self._coarticulation.calculate_blend(t) * next_event.weight * self._intensity

                    for shape_name, shape_weight in next_mapping.blend_shapes.items():
                        weights[shape_name] = weights.get(shape_name, 0.0) + shape_weight * anticipation_weight

        # Clamp weights
        for name in weights:
            weights[name] = max(0.0, min(1.0, weights[name]))

        if weights != self._current_weights:
            self._current_viseme = current_event.viseme
            self._current_weights = weights
            self._dirty = True
            self._notify_change()

        return weights

    def get_blend_weights(self) -> dict[str, float]:
        """Get current blend shape weights."""
        return self._current_weights.copy()

    def set_viseme_mapping(self, viseme: Viseme, mapping: VisemeMapping) -> None:
        """
        Set or update a viseme mapping.

        Args:
            viseme: The viseme
            mapping: The blend shape mapping
        """
        self._viseme_mappings[viseme] = mapping

    def get_viseme_at_time(self, time: float) -> Viseme:
        """
        Get the viseme at a specific time.

        Args:
            time: Time in seconds

        Returns:
            Viseme at that time
        """
        if not self._viseme_timeline:
            return Viseme.SIL

        idx = bisect.bisect_right(self._timeline_times, time) - 1
        idx = max(0, min(idx, len(self._viseme_timeline) - 1))

        return self._viseme_timeline[idx].viseme

    def clear_dirty(self) -> None:
        """Clear the dirty flag."""
        self._dirty = False

    @property
    def dirty(self) -> bool:
        """Check if state has changed."""
        return self._dirty

    def _notify_change(self) -> None:
        """Notify change callback."""
        if self._on_weights_changed:
            self._on_weights_changed(self._current_weights.copy())

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "current_time": self._current_time,
            "is_playing": self._is_playing,
            "current_viseme": self._current_viseme.name,
            "intensity": self._intensity,
            "blend_time": self._blend_time,
            "timeline_count": len(self._viseme_timeline),
        }


# =============================================================================
# Utility Functions
# =============================================================================


def create_phoneme_events_from_text(
    text: str,
    start_time: float = 0.0,
    phoneme_duration: float = 0.08,
) -> list[PhonemeEvent]:
    """
    Create simple phoneme events from text (for testing).

    This is a very basic approximation - real lip sync should use
    audio analysis or text-to-phoneme conversion.

    Args:
        text: Input text
        start_time: Start time offset
        phoneme_duration: Duration per phoneme

    Returns:
        List of phoneme events
    """
    # Simple character to phoneme mapping (very approximate)
    char_to_phoneme = {
        'a': 'aa', 'b': 'b', 'c': 'k', 'd': 'd', 'e': 'eh',
        'f': 'f', 'g': 'g', 'h': 'hh', 'i': 'ih', 'j': 'jh',
        'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 'o': 'ao',
        'p': 'p', 'q': 'k', 'r': 'r', 's': 's', 't': 't',
        'u': 'uw', 'v': 'v', 'w': 'w', 'x': 'k', 'y': 'y', 'z': 'z',
    }

    events = []
    current_time = start_time

    for char in text.lower():
        if char in char_to_phoneme:
            phoneme = char_to_phoneme[char]
            events.append(PhonemeEvent(
                phoneme=phoneme,
                start_time=current_time,
                end_time=current_time + phoneme_duration,
                confidence=1.0,
            ))
            current_time += phoneme_duration
        elif char == ' ':
            # Short silence for spaces
            events.append(PhonemeEvent(
                phoneme='sil',
                start_time=current_time,
                end_time=current_time + phoneme_duration * 0.5,
                confidence=1.0,
            ))
            current_time += phoneme_duration * 0.5

    return events
