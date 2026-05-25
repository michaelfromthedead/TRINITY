"""Speaker Configuration and Channel Routing.

Implements speaker layouts and channel routing:
- Stereo (2.0), Quad (4.0), 5.1, 7.1, Atmos
- Downmix and upmix matrices
- Channel routing and mapping
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from engine.audio.spatial.config import SPEAKER_ANGLES, SpeakerLayout
from engine.core.math.vec import Vec3


class ChannelName(Enum):
    """Standard audio channel names."""

    # Basic channels
    MONO = auto()
    LEFT = auto()
    RIGHT = auto()
    CENTER = auto()
    LFE = auto()

    # Surround channels
    SURROUND_LEFT = auto()
    SURROUND_RIGHT = auto()
    SIDE_LEFT = auto()
    SIDE_RIGHT = auto()
    BACK_LEFT = auto()
    BACK_RIGHT = auto()

    # Height channels
    TOP_FRONT_LEFT = auto()
    TOP_FRONT_RIGHT = auto()
    TOP_BACK_LEFT = auto()
    TOP_BACK_RIGHT = auto()
    TOP_CENTER = auto()


# Channel indices for each layout
CHANNEL_INDICES: Dict[SpeakerLayout, Dict[ChannelName, int]] = {
    SpeakerLayout.MONO: {
        ChannelName.MONO: 0,
    },
    SpeakerLayout.STEREO: {
        ChannelName.LEFT: 0,
        ChannelName.RIGHT: 1,
    },
    SpeakerLayout.BINAURAL: {
        ChannelName.LEFT: 0,
        ChannelName.RIGHT: 1,
    },
    SpeakerLayout.QUAD: {
        ChannelName.LEFT: 0,
        ChannelName.RIGHT: 1,
        ChannelName.BACK_LEFT: 2,
        ChannelName.BACK_RIGHT: 3,
    },
    SpeakerLayout.SURROUND_5_1: {
        ChannelName.LEFT: 0,
        ChannelName.RIGHT: 1,
        ChannelName.CENTER: 2,
        ChannelName.LFE: 3,
        ChannelName.SURROUND_LEFT: 4,
        ChannelName.SURROUND_RIGHT: 5,
    },
    SpeakerLayout.SURROUND_7_1: {
        ChannelName.LEFT: 0,
        ChannelName.RIGHT: 1,
        ChannelName.CENTER: 2,
        ChannelName.LFE: 3,
        ChannelName.SIDE_LEFT: 4,
        ChannelName.SIDE_RIGHT: 5,
        ChannelName.BACK_LEFT: 6,
        ChannelName.BACK_RIGHT: 7,
    },
    SpeakerLayout.ATMOS_5_1_2: {
        ChannelName.LEFT: 0,
        ChannelName.RIGHT: 1,
        ChannelName.CENTER: 2,
        ChannelName.LFE: 3,
        ChannelName.SURROUND_LEFT: 4,
        ChannelName.SURROUND_RIGHT: 5,
        ChannelName.TOP_FRONT_LEFT: 6,
        ChannelName.TOP_FRONT_RIGHT: 7,
    },
    SpeakerLayout.ATMOS_7_1_4: {
        ChannelName.LEFT: 0,
        ChannelName.RIGHT: 1,
        ChannelName.CENTER: 2,
        ChannelName.LFE: 3,
        ChannelName.SIDE_LEFT: 4,
        ChannelName.SIDE_RIGHT: 5,
        ChannelName.BACK_LEFT: 6,
        ChannelName.BACK_RIGHT: 7,
        ChannelName.TOP_FRONT_LEFT: 8,
        ChannelName.TOP_FRONT_RIGHT: 9,
        ChannelName.TOP_BACK_LEFT: 10,
        ChannelName.TOP_BACK_RIGHT: 11,
    },
}


def get_channel_count(layout: SpeakerLayout) -> int:
    """Get number of channels for a speaker layout."""
    counts = {
        SpeakerLayout.MONO: 1,
        SpeakerLayout.STEREO: 2,
        SpeakerLayout.BINAURAL: 2,
        SpeakerLayout.QUAD: 4,
        SpeakerLayout.SURROUND_5_1: 6,
        SpeakerLayout.SURROUND_7_1: 8,
        SpeakerLayout.ATMOS_5_1_2: 8,
        SpeakerLayout.ATMOS_7_1_4: 12,
    }
    return counts.get(layout, 2)


@dataclass
class SpeakerPosition:
    """Position of a speaker in the listening environment."""

    channel: ChannelName
    """Channel this speaker represents."""

    azimuth: float = 0.0
    """Horizontal angle in degrees (0 = front, positive = right)."""

    elevation: float = 0.0
    """Vertical angle in degrees (0 = ear level, positive = above)."""

    distance: float = 1.0
    """Distance from listener in meters."""

    is_lfe: bool = False
    """Whether this is a Low Frequency Effects (subwoofer) channel."""

    def get_direction(self) -> Vec3:
        """Get unit direction vector to this speaker."""
        az_rad = math.radians(self.azimuth)
        el_rad = math.radians(self.elevation)
        return Vec3(
            math.sin(az_rad) * math.cos(el_rad),
            math.sin(el_rad),
            -math.cos(az_rad) * math.cos(el_rad)
        )


@dataclass
class SpeakerConfiguration:
    """Complete speaker configuration for a listening environment."""

    layout: SpeakerLayout = SpeakerLayout.STEREO
    """Speaker layout type."""

    speakers: List[SpeakerPosition] = field(default_factory=list)
    """Individual speaker positions."""

    lfe_crossover: float = 80.0
    """LFE crossover frequency in Hz."""

    bass_management: bool = True
    """Whether to use bass management (send low frequencies to LFE)."""

    def __post_init__(self) -> None:
        if not self.speakers:
            self._create_default_speakers()

    def _create_default_speakers(self) -> None:
        """Create default speaker positions for the layout."""
        self.speakers = []
        angles = SPEAKER_ANGLES.get(self.layout, [])
        channels = CHANNEL_INDICES.get(self.layout, {})

        # Reverse lookup: index -> channel name
        index_to_channel = {v: k for k, v in channels.items()}

        for i, (az, el) in enumerate(angles):
            channel = index_to_channel.get(i, ChannelName.MONO)
            is_lfe = channel == ChannelName.LFE

            speaker = SpeakerPosition(
                channel=channel,
                azimuth=az,
                elevation=el,
                distance=1.0,
                is_lfe=is_lfe
            )
            self.speakers.append(speaker)

    @property
    def channel_count(self) -> int:
        """Get number of channels."""
        return len(self.speakers)

    def get_channel_index(self, channel: ChannelName) -> Optional[int]:
        """Get index of a channel, or None if not in layout."""
        indices = CHANNEL_INDICES.get(self.layout, {})
        return indices.get(channel)

    def get_speaker(self, channel: ChannelName) -> Optional[SpeakerPosition]:
        """Get speaker position for a channel."""
        for speaker in self.speakers:
            if speaker.channel == channel:
                return speaker
        return None


# Downmix matrix type: source_channels -> (dest_channels -> coefficient)
DownmixMatrix = List[List[float]]

# Standard downmix coefficients
_MINUS_3DB = 0.707107  # 1/sqrt(2)
_MINUS_6DB = 0.5
_MINUS_12DB = 0.25

# Downmix from 5.1 to stereo
DOWNMIX_5_1_TO_STEREO: DownmixMatrix = [
    # L, R, C, LFE, SL, SR -> L, R
    [1.0, 0.0],           # L -> L
    [0.0, 1.0],           # R -> R
    [_MINUS_3DB, _MINUS_3DB],  # C -> L,R
    [0.0, 0.0],           # LFE (discarded or separate)
    [_MINUS_3DB, 0.0],    # SL -> L
    [0.0, _MINUS_3DB],    # SR -> R
]

# Downmix from 7.1 to stereo
DOWNMIX_7_1_TO_STEREO: DownmixMatrix = [
    [1.0, 0.0],
    [0.0, 1.0],
    [_MINUS_3DB, _MINUS_3DB],
    [0.0, 0.0],
    [_MINUS_3DB, 0.0],    # Side L
    [0.0, _MINUS_3DB],    # Side R
    [_MINUS_6DB, 0.0],    # Back L
    [0.0, _MINUS_6DB],    # Back R
]

# Downmix from 7.1 to 5.1
DOWNMIX_7_1_TO_5_1: DownmixMatrix = [
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # L
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # R
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # C
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # LFE
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # Side L -> SL
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Side R -> SR
    [0.0, 0.0, 0.0, 0.0, _MINUS_3DB, 0.0],  # Back L -> SL
    [0.0, 0.0, 0.0, 0.0, 0.0, _MINUS_3DB],  # Back R -> SR
]

# Downmix from quad to stereo
DOWNMIX_QUAD_TO_STEREO: DownmixMatrix = [
    [1.0, 0.0],
    [0.0, 1.0],
    [_MINUS_3DB, 0.0],
    [0.0, _MINUS_3DB],
]

# Upmix from stereo to 5.1
UPMIX_STEREO_TO_5_1: DownmixMatrix = [
    # L, R -> L, R, C, LFE, SL, SR
    [1.0, 0.0, _MINUS_6DB, 0.0, _MINUS_6DB, 0.0],  # L
    [0.0, 1.0, _MINUS_6DB, 0.0, 0.0, _MINUS_6DB],  # R
]

# Upmix from mono to stereo
UPMIX_MONO_TO_STEREO: DownmixMatrix = [
    [_MINUS_3DB, _MINUS_3DB],  # Mono -> L, R
]


def apply_mix_matrix(
    input_samples: List[List[float]],
    matrix: DownmixMatrix
) -> List[List[float]]:
    """Apply a mix matrix to convert between channel configurations.

    Args:
        input_samples: List of channels, each containing samples.
        matrix: Mix matrix (input_channels x output_channels).

    Returns:
        Output samples with transformed channel configuration.
    """
    if not input_samples or not matrix:
        return []

    num_input_channels = len(input_samples)
    num_samples = len(input_samples[0])
    num_output_channels = len(matrix[0]) if matrix else 0

    # Initialize output
    output = [[0.0] * num_samples for _ in range(num_output_channels)]

    # Apply matrix
    for in_ch in range(min(num_input_channels, len(matrix))):
        for out_ch in range(num_output_channels):
            coeff = matrix[in_ch][out_ch]
            if coeff != 0.0:
                for i in range(num_samples):
                    output[out_ch][i] += input_samples[in_ch][i] * coeff

    return output


class ChannelRouter:
    """Routes audio channels between different speaker configurations."""

    def __init__(
        self,
        source_layout: SpeakerLayout,
        dest_layout: SpeakerLayout
    ) -> None:
        self._source_layout = source_layout
        self._dest_layout = dest_layout
        self._matrix = self._build_matrix()

    @property
    def source_layout(self) -> SpeakerLayout:
        """Get source layout."""
        return self._source_layout

    @property
    def dest_layout(self) -> SpeakerLayout:
        """Get destination layout."""
        return self._dest_layout

    @property
    def source_channels(self) -> int:
        """Get number of source channels."""
        return get_channel_count(self._source_layout)

    @property
    def dest_channels(self) -> int:
        """Get number of destination channels."""
        return get_channel_count(self._dest_layout)

    def _build_matrix(self) -> DownmixMatrix:
        """Build the routing matrix."""
        # Check for identity (same layout)
        if self._source_layout == self._dest_layout:
            n = get_channel_count(self._source_layout)
            return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

        # Standard downmix matrices
        if self._source_layout == SpeakerLayout.SURROUND_5_1:
            if self._dest_layout == SpeakerLayout.STEREO:
                return DOWNMIX_5_1_TO_STEREO
            elif self._dest_layout == SpeakerLayout.MONO:
                # 5.1 -> stereo -> mono
                stereo = apply_mix_matrix(
                    [[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]],  # Just L channel
                    DOWNMIX_5_1_TO_STEREO
                )
                return [[_MINUS_3DB, _MINUS_3DB]]  # Combine stereo to mono

        if self._source_layout == SpeakerLayout.SURROUND_7_1:
            if self._dest_layout == SpeakerLayout.STEREO:
                return DOWNMIX_7_1_TO_STEREO
            elif self._dest_layout == SpeakerLayout.SURROUND_5_1:
                return DOWNMIX_7_1_TO_5_1

        if self._source_layout == SpeakerLayout.QUAD:
            if self._dest_layout == SpeakerLayout.STEREO:
                return DOWNMIX_QUAD_TO_STEREO

        if self._source_layout == SpeakerLayout.STEREO:
            if self._dest_layout == SpeakerLayout.MONO:
                return UPMIX_MONO_TO_STEREO  # Works both ways
            elif self._dest_layout == SpeakerLayout.SURROUND_5_1:
                return UPMIX_STEREO_TO_5_1

        if self._source_layout == SpeakerLayout.MONO:
            if self._dest_layout == SpeakerLayout.STEREO:
                return UPMIX_MONO_TO_STEREO

        # Fallback: identity or truncation
        src_n = get_channel_count(self._source_layout)
        dst_n = get_channel_count(self._dest_layout)
        return [
            [1.0 if i == j else 0.0 for j in range(dst_n)]
            for i in range(src_n)
        ]

    def route(self, input_samples: List[List[float]]) -> List[List[float]]:
        """Route samples from source to destination configuration.

        Args:
            input_samples: Input samples [channel][sample].

        Returns:
            Output samples [channel][sample].
        """
        return apply_mix_matrix(input_samples, self._matrix)


@dataclass
class VirtualSpeaker:
    """A virtual speaker for object-based audio (like Atmos).

    Represents a sound object that can be freely positioned in 3D
    space rather than tied to physical speaker channels.
    """

    object_id: int = 0
    """Unique identifier for this object."""

    position: Vec3 = field(default_factory=Vec3.zero)
    """Position in 3D space."""

    gain: float = 1.0
    """Object gain/volume."""

    size: float = 0.0
    """Object size (0 = point source, larger = more spread)."""

    priority: int = 0
    """Rendering priority (higher = more important)."""

    def get_speaker_gains(self, config: SpeakerConfiguration) -> List[float]:
        """Calculate gains for physical speakers based on position.

        Uses VBAP-like algorithm to distribute object to nearest speakers.
        """
        gains = [0.0] * config.channel_count

        # Calculate direction from listener
        distance = self.position.length()
        if distance < 0.0001:
            # At listener position - equal to all speakers
            for i in range(config.channel_count):
                if not config.speakers[i].is_lfe:
                    gains[i] = self.gain / math.sqrt(config.channel_count)
            return gains

        direction = self.position / distance
        azimuth = math.degrees(math.atan2(direction.x, -direction.z))
        elevation = math.degrees(math.atan2(direction.y, math.sqrt(direction.x**2 + direction.z**2)))

        # Find nearest speakers (excluding LFE)
        speaker_distances: List[Tuple[int, float]] = []
        for i, speaker in enumerate(config.speakers):
            if speaker.is_lfe:
                continue
            speaker_dir = speaker.get_direction()
            dot = direction.dot(speaker_dir)
            angular_dist = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
            speaker_distances.append((i, angular_dist))

        speaker_distances.sort(key=lambda x: x[1])

        # Use two nearest speakers for panning
        if len(speaker_distances) >= 2:
            idx1, dist1 = speaker_distances[0]
            idx2, dist2 = speaker_distances[1]

            total = dist1 + dist2
            if total > 0.0001:
                w1 = 1.0 - dist1 / total
                w2 = 1.0 - dist2 / total
                norm = math.sqrt(w1**2 + w2**2)
                gains[idx1] = (w1 / norm) * self.gain
                gains[idx2] = (w2 / norm) * self.gain
            else:
                gains[idx1] = self.gain
        elif speaker_distances:
            gains[speaker_distances[0][0]] = self.gain

        # Apply size (spread to more speakers)
        if self.size > 0.0:
            spread_amount = self.gain * self.size * 0.3
            for i in range(config.channel_count):
                if not config.speakers[i].is_lfe:
                    gains[i] = max(gains[i], spread_amount)

        return gains


def create_speaker_config(
    layout: SpeakerLayout,
    custom_positions: Optional[List[Tuple[float, float]]] = None
) -> SpeakerConfiguration:
    """Create a speaker configuration.

    Args:
        layout: Speaker layout type.
        custom_positions: Optional custom (azimuth, elevation) positions.

    Returns:
        Speaker configuration.
    """
    config = SpeakerConfiguration(layout=layout)

    if custom_positions:
        channels = CHANNEL_INDICES.get(layout, {})
        index_to_channel = {v: k for k, v in channels.items()}

        config.speakers = []
        for i, (az, el) in enumerate(custom_positions):
            channel = index_to_channel.get(i, ChannelName.MONO)
            config.speakers.append(SpeakerPosition(
                channel=channel,
                azimuth=az,
                elevation=el,
                is_lfe=(channel == ChannelName.LFE)
            ))

    return config
