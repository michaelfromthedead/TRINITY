"""
Voice Chat System Module.

Provides voice communication management including channel handling,
proximity-based voice, muting, and volume controls.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable, Any
from threading import Lock
import logging
import time
import math

from .config import SOCIAL_CONFIG

logger = logging.getLogger(__name__)


class VoiceChannel(Enum):
    """Voice communication channel types."""
    TEAM = auto()       # Team-only voice
    SQUAD = auto()      # Squad/party voice
    PROXIMITY = auto()  # Distance-based voice
    GLOBAL = auto()     # Everyone can hear
    PRIVATE = auto()    # Direct player-to-player


class VoiceQuality(Enum):
    """Voice transmission quality levels."""
    LOW = auto()        # Lower bandwidth, reduced quality
    MEDIUM = auto()     # Balanced
    HIGH = auto()       # High quality, higher bandwidth
    ULTRA = auto()      # Highest quality


@dataclass
class VoiceState:
    """Voice state for a participant."""
    is_muted: bool = False          # Self-muted (can't transmit)
    is_deafened: bool = False       # Self-deafened (can't receive)
    is_server_muted: bool = False   # Server-enforced mute
    volume: float = SOCIAL_CONFIG.VoiceChat.VOLUME_DEFAULT             # Output volume (0.0 to 2.0)
    input_volume: float = SOCIAL_CONFIG.VoiceChat.INPUT_VOLUME_DEFAULT       # Input sensitivity (0.0 to 2.0)
    noise_gate: float = SOCIAL_CONFIG.VoiceChat.NOISE_GATE_DEFAULT        # Threshold for voice activation
    push_to_talk: bool = False      # PTT mode enabled
    is_transmitting: bool = False   # Currently transmitting
    quality: VoiceQuality = VoiceQuality.MEDIUM

    def __post_init__(self) -> None:
        """Validate state values."""
        self.volume = max(
            SOCIAL_CONFIG.VoiceChat.VOLUME_MIN,
            min(SOCIAL_CONFIG.VoiceChat.VOLUME_MAX, self.volume)
        )
        self.input_volume = max(
            SOCIAL_CONFIG.VoiceChat.VOLUME_MIN,
            min(SOCIAL_CONFIG.VoiceChat.VOLUME_MAX, self.input_volume)
        )
        self.noise_gate = max(
            SOCIAL_CONFIG.VoiceChat.NOISE_GATE_MIN,
            min(SOCIAL_CONFIG.VoiceChat.NOISE_GATE_MAX, self.noise_gate)
        )


@dataclass
class VoiceParticipant:
    """A participant in voice chat."""
    player_id: str
    display_name: str
    channel: VoiceChannel
    state: VoiceState = field(default_factory=VoiceState)
    position: Optional[tuple[float, float, float]] = None  # For proximity
    join_time: float = field(default_factory=time.time)

    # Per-player volume overrides (other player -> volume)
    # Note: Using field(default_factory=dict) is correct for mutable defaults in dataclasses
    _volume_overrides: dict[str, float] = field(default_factory=dict)

    # Per-player mutes (players this participant has muted)
    # Note: Using field(default_factory=set) is correct for mutable defaults in dataclasses
    _muted_players: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Ensure mutable fields are properly initialized."""
        # Defensive copy in case someone passes a mutable directly
        if self._volume_overrides is not None:
            self._volume_overrides = dict(self._volume_overrides)
        else:
            self._volume_overrides = {}
        if self._muted_players is not None:
            self._muted_players = set(self._muted_players)
        else:
            self._muted_players = set()

    def get_volume_for_player(self, player_id: str) -> float:
        """Get the effective volume for hearing a specific player."""
        return self._volume_overrides.get(player_id, self.state.volume)

    def is_player_muted(self, player_id: str) -> bool:
        """Check if this participant has muted another player."""
        return player_id in self._muted_players


@dataclass
class VoiceChannelInfo:
    """Information about a voice channel."""
    channel_type: VoiceChannel
    channel_id: str
    participants: list[str] = field(default_factory=list)
    max_participants: int = SOCIAL_CONFIG.VoiceChat.CHANNEL_MAX_PARTICIPANTS
    is_moderated: bool = False
    moderators: set[str] = field(default_factory=set)


class ProximityVoice:
    """
    Handles proximity-based voice communication.

    Calculates volume attenuation based on distance between players.
    """

    def __init__(
        self,
        max_distance: Optional[float] = None,
        min_distance: Optional[float] = None,
        falloff_exponent: Optional[float] = None,
        use_occlusion: bool = False
    ) -> None:
        """
        Initialize proximity voice settings.

        Args:
            max_distance: Maximum distance for hearing (units).
            min_distance: Distance below which volume is 100%.
            falloff_exponent: Rate of volume falloff (2.0 = inverse square).
            use_occlusion: Whether to check for obstacles.
        """
        self.max_distance = max_distance if max_distance is not None else SOCIAL_CONFIG.VoiceChat.PROXIMITY_MAX_DISTANCE
        self.min_distance = min_distance if min_distance is not None else SOCIAL_CONFIG.VoiceChat.PROXIMITY_MIN_DISTANCE
        self.falloff_exponent = falloff_exponent if falloff_exponent is not None else SOCIAL_CONFIG.VoiceChat.PROXIMITY_FALLOFF_EXPONENT
        self.use_occlusion = use_occlusion

        # Occlusion callback (returns 0.0-1.0 obstruction factor)
        self._occlusion_callback: Optional[
            Callable[[tuple[float, float, float], tuple[float, float, float]], float]
        ] = None

    def set_occlusion_callback(
        self,
        callback: Callable[[tuple[float, float, float], tuple[float, float, float]], float]
    ) -> None:
        """Set callback for checking occlusion between two positions."""
        self._occlusion_callback = callback

    def calculate_distance(
        self,
        pos1: tuple[float, float, float],
        pos2: tuple[float, float, float]
    ) -> float:
        """Calculate 3D distance between two positions."""
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        dz = pos2[2] - pos1[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def calculate_attenuation(
        self,
        distance: float,
        source_pos: Optional[tuple[float, float, float]] = None,
        listener_pos: Optional[tuple[float, float, float]] = None
    ) -> float:
        """
        Calculate volume attenuation based on distance.

        Args:
            distance: Distance between speaker and listener.
            source_pos: Speaker's position (for occlusion).
            listener_pos: Listener's position (for occlusion).

        Returns:
            Volume multiplier (0.0 to 1.0).
        """
        if distance >= self.max_distance:
            return 0.0

        if distance <= self.min_distance:
            base_attenuation = 1.0
        else:
            # Calculate distance-based attenuation
            effective_distance = distance - self.min_distance
            max_effective = self.max_distance - self.min_distance

            # Inverse power law falloff
            normalized = effective_distance / max_effective
            base_attenuation = max(0.0, 1.0 - pow(normalized, 1.0 / self.falloff_exponent))

        # Apply occlusion if enabled
        if self.use_occlusion and source_pos and listener_pos and self._occlusion_callback:
            occlusion_factor = self._occlusion_callback(source_pos, listener_pos)
            base_attenuation *= (1.0 - occlusion_factor)

        return base_attenuation

    def get_audible_players(
        self,
        listener_pos: tuple[float, float, float],
        speaker_positions: dict[str, tuple[float, float, float]]
    ) -> dict[str, float]:
        """
        Get all audible players and their volumes for a listener.

        Args:
            listener_pos: The listener's position.
            speaker_positions: Dict mapping player_id to position.

        Returns:
            Dict mapping player_id to volume multiplier (only audible players).
        """
        audible: dict[str, float] = {}

        for player_id, speaker_pos in speaker_positions.items():
            distance = self.calculate_distance(listener_pos, speaker_pos)
            attenuation = self.calculate_attenuation(
                distance, speaker_pos, listener_pos
            )

            if attenuation > 0.0:
                audible[player_id] = attenuation

        return audible


class VoiceChatManager:
    """
    Manages voice communication for all players.

    Handles channel management, muting, volume controls,
    and active speaker tracking.

    Thread-safe for concurrent access.
    """

    def __init__(self) -> None:
        """Initialize the voice chat manager."""
        self._participants: dict[str, VoiceParticipant] = {}
        self._channels: dict[str, VoiceChannelInfo] = {}
        self._lock = Lock()

        # Proximity voice handler
        self._proximity = ProximityVoice()

        # Callbacks
        self._on_join: Optional[Callable[[str, VoiceChannel], None]] = None
        self._on_leave: Optional[Callable[[str], None]] = None
        self._on_speaking_start: Optional[Callable[[str], None]] = None
        self._on_speaking_stop: Optional[Callable[[str], None]] = None

        # Active speakers (recently transmitted)
        self._active_speakers: dict[str, float] = {}  # player_id -> last_spoke_time
        self._speaker_timeout = SOCIAL_CONFIG.VoiceChat.SPEAKER_TIMEOUT_SECONDS  # Seconds before no longer "active"

    def set_on_join(self, callback: Callable[[str, VoiceChannel], None]) -> None:
        """Set callback for player joining voice."""
        self._on_join = callback

    def set_on_leave(self, callback: Callable[[str], None]) -> None:
        """Set callback for player leaving voice."""
        self._on_leave = callback

    def set_on_speaking_start(self, callback: Callable[[str], None]) -> None:
        """Set callback for when a player starts speaking."""
        self._on_speaking_start = callback

    def set_on_speaking_stop(self, callback: Callable[[str], None]) -> None:
        """Set callback for when a player stops speaking."""
        self._on_speaking_stop = callback

    @property
    def proximity_voice(self) -> ProximityVoice:
        """Get the proximity voice handler."""
        return self._proximity

    def join_channel(
        self,
        player_id: str,
        display_name: str,
        channel: VoiceChannel,
        channel_id: Optional[str] = None,
        position: Optional[tuple[float, float, float]] = None
    ) -> bool:
        """
        Join a voice channel.

        Args:
            player_id: The player joining.
            display_name: The player's display name.
            channel: The channel type to join.
            channel_id: Specific channel ID (for team/squad channels).
            position: Initial position (for proximity voice).

        Returns:
            True if successfully joined, False otherwise.
        """
        with self._lock:
            # Leave current channel if in one
            if player_id in self._participants:
                self._leave_internal(player_id)

            participant = VoiceParticipant(
                player_id=player_id,
                display_name=display_name,
                channel=channel,
                position=position
            )
            self._participants[player_id] = participant

            # Track channel membership
            cid = channel_id or channel.name
            if cid not in self._channels:
                self._channels[cid] = VoiceChannelInfo(
                    channel_type=channel,
                    channel_id=cid
                )
            self._channels[cid].participants.append(player_id)

            if self._on_join:
                self._on_join(player_id, channel)

            return True

    def _leave_internal(self, player_id: str) -> None:
        """Internal leave without lock (caller must hold lock)."""
        if player_id not in self._participants:
            return

        participant = self._participants[player_id]

        # Remove from channel tracking
        for channel_info in self._channels.values():
            if player_id in channel_info.participants:
                channel_info.participants.remove(player_id)

        del self._participants[player_id]

        # Remove from active speakers
        if player_id in self._active_speakers:
            del self._active_speakers[player_id]

    def leave_channel(self, player_id: str) -> bool:
        """
        Leave the current voice channel.

        Args:
            player_id: The player leaving.

        Returns:
            True if successfully left, False if not in voice.
        """
        with self._lock:
            if player_id not in self._participants:
                return False

            self._leave_internal(player_id)

            if self._on_leave:
                self._on_leave(player_id)

            return True

    def get_participant(self, player_id: str) -> Optional[VoiceParticipant]:
        """Get a participant by player ID."""
        with self._lock:
            return self._participants.get(player_id)

    def get_channel_participants(
        self,
        channel: VoiceChannel,
        channel_id: Optional[str] = None
    ) -> list[VoiceParticipant]:
        """Get all participants in a channel."""
        with self._lock:
            cid = channel_id or channel.name
            channel_info = self._channels.get(cid)

            if not channel_info:
                return []

            return [
                self._participants[pid]
                for pid in channel_info.participants
                if pid in self._participants
            ]

    def mute_self(self, player_id: str, muted: bool = True) -> bool:
        """
        Toggle self-mute for a player.

        Args:
            player_id: The player to mute/unmute.
            muted: Whether to mute.

        Returns:
            True if updated, False if player not found.
        """
        with self._lock:
            participant = self._participants.get(player_id)
            if not participant:
                return False

            participant.state.is_muted = muted

            if muted and participant.state.is_transmitting:
                participant.state.is_transmitting = False
                if self._on_speaking_stop:
                    self._on_speaking_stop(player_id)

            return True

    def deafen_self(self, player_id: str, deafened: bool = True) -> bool:
        """
        Toggle self-deafen for a player.

        Args:
            player_id: The player to deafen/undeafen.
            deafened: Whether to deafen.

        Returns:
            True if updated, False if player not found.
        """
        with self._lock:
            participant = self._participants.get(player_id)
            if not participant:
                return False

            participant.state.is_deafened = deafened
            return True

    def mute_player(
        self,
        target_id: str,
        muted_by: str,
        server_mute: bool = False
    ) -> bool:
        """
        Mute another player.

        Args:
            target_id: The player to mute.
            muted_by: The player doing the muting.
            server_mute: If True, applies server-enforced mute.

        Returns:
            True if muted, False otherwise.
        """
        with self._lock:
            if server_mute:
                # Server mute - affects the target's transmission
                target = self._participants.get(target_id)
                if not target:
                    return False
                target.state.is_server_muted = True
                return True
            else:
                # Personal mute - only affects what muter hears
                muter = self._participants.get(muted_by)
                if not muter:
                    return False
                muter._muted_players.add(target_id)
                return True

    def unmute_player(
        self,
        target_id: str,
        unmuted_by: str,
        server_mute: bool = False
    ) -> bool:
        """
        Unmute another player.

        Args:
            target_id: The player to unmute.
            unmuted_by: The player doing the unmuting.
            server_mute: If True, removes server-enforced mute.

        Returns:
            True if unmuted, False otherwise.
        """
        with self._lock:
            if server_mute:
                target = self._participants.get(target_id)
                if not target:
                    return False
                target.state.is_server_muted = False
                return True
            else:
                muter = self._participants.get(unmuted_by)
                if not muter:
                    return False
                muter._muted_players.discard(target_id)
                return True

    def set_volume(
        self,
        player_id: str,
        target_id: Optional[str] = None,
        volume: float = SOCIAL_CONFIG.VoiceChat.VOLUME_DEFAULT
    ) -> bool:
        """
        Set volume level.

        Args:
            player_id: The player setting volume.
            target_id: Specific player to adjust (None for global).
            volume: Volume level (0.0 to 2.0).

        Returns:
            True if updated, False otherwise.
        """
        volume = max(
            SOCIAL_CONFIG.VoiceChat.VOLUME_MIN,
            min(SOCIAL_CONFIG.VoiceChat.VOLUME_MAX, volume)
        )

        with self._lock:
            participant = self._participants.get(player_id)
            if not participant:
                return False

            if target_id:
                # Per-player volume
                participant._volume_overrides[target_id] = volume
            else:
                # Global volume
                participant.state.volume = volume

            return True

    def update_position(
        self,
        player_id: str,
        position: tuple[float, float, float]
    ) -> bool:
        """
        Update a player's position for proximity voice.

        Args:
            player_id: The player to update.
            position: New 3D position.

        Returns:
            True if updated, False if player not found.
        """
        with self._lock:
            participant = self._participants.get(player_id)
            if not participant:
                return False

            participant.position = position
            return True

    def set_transmitting(
        self,
        player_id: str,
        transmitting: bool = True
    ) -> bool:
        """
        Set whether a player is currently transmitting.

        Args:
            player_id: The player.
            transmitting: Whether transmitting.

        Returns:
            True if updated, False if player not found.
        """
        with self._lock:
            participant = self._participants.get(player_id)
            if not participant:
                return False

            # Can't transmit if muted
            if transmitting and (
                participant.state.is_muted or
                participant.state.is_server_muted
            ):
                return False

            was_transmitting = participant.state.is_transmitting
            participant.state.is_transmitting = transmitting

            if transmitting:
                self._active_speakers[player_id] = time.time()
                if not was_transmitting and self._on_speaking_start:
                    self._on_speaking_start(player_id)
            else:
                if was_transmitting and self._on_speaking_stop:
                    self._on_speaking_stop(player_id)

            return True

    def get_active_speakers(
        self,
        channel: Optional[VoiceChannel] = None,
        channel_id: Optional[str] = None
    ) -> list[str]:
        """
        Get list of currently active speakers.

        Args:
            channel: Filter by channel type.
            channel_id: Filter by specific channel.

        Returns:
            List of player IDs who are actively speaking.
        """
        with self._lock:
            current_time = time.time()
            active: list[str] = []

            for player_id, last_spoke in list(self._active_speakers.items()):
                if current_time - last_spoke > self._speaker_timeout:
                    del self._active_speakers[player_id]
                    continue

                participant = self._participants.get(player_id)
                if not participant:
                    continue

                # Apply filters
                if channel and participant.channel != channel:
                    continue

                if channel_id:
                    cid = channel_id
                    channel_info = self._channels.get(cid)
                    if not channel_info or player_id not in channel_info.participants:
                        continue

                active.append(player_id)

            return active

    def get_audible_players(
        self,
        listener_id: str
    ) -> dict[str, float]:
        """
        Get all players audible to a listener with their volumes.

        Handles channel membership, muting, and proximity.

        Args:
            listener_id: The listening player's ID.

        Returns:
            Dict mapping player_id to effective volume.
        """
        with self._lock:
            listener = self._participants.get(listener_id)
            if not listener or listener.state.is_deafened:
                return {}

            audible: dict[str, float] = {}

            for player_id, participant in self._participants.items():
                if player_id == listener_id:
                    continue

                # Check if muted by listener or server
                if listener.is_player_muted(player_id):
                    continue

                if participant.state.is_muted or participant.state.is_server_muted:
                    continue

                if not participant.state.is_transmitting:
                    continue

                # Calculate base volume
                base_volume = listener.get_volume_for_player(player_id)

                # Handle channel-specific logic
                if participant.channel == VoiceChannel.PROXIMITY:
                    if listener.channel != VoiceChannel.PROXIMITY:
                        continue

                    # Both in proximity - calculate distance-based volume
                    if participant.position and listener.position:
                        distance = self._proximity.calculate_distance(
                            listener.position, participant.position
                        )
                        attenuation = self._proximity.calculate_attenuation(
                            distance, participant.position, listener.position
                        )
                        if attenuation > 0:
                            audible[player_id] = base_volume * attenuation

                elif participant.channel == VoiceChannel.GLOBAL:
                    # Global is always audible
                    audible[player_id] = base_volume

                else:
                    # Team/Squad - must be in same channel
                    for cid, channel_info in self._channels.items():
                        if (player_id in channel_info.participants and
                                listener_id in channel_info.participants):
                            audible[player_id] = base_volume
                            break

            return audible

    def set_quality(self, player_id: str, quality: VoiceQuality) -> bool:
        """
        Set voice quality for a player.

        Args:
            player_id: The player.
            quality: Quality level.

        Returns:
            True if updated, False if player not found.
        """
        with self._lock:
            participant = self._participants.get(player_id)
            if not participant:
                return False

            participant.state.quality = quality
            return True

    def set_push_to_talk(self, player_id: str, enabled: bool = True) -> bool:
        """
        Enable/disable push-to-talk mode.

        Args:
            player_id: The player.
            enabled: Whether to enable PTT.

        Returns:
            True if updated, False if player not found.
        """
        with self._lock:
            participant = self._participants.get(player_id)
            if not participant:
                return False

            participant.state.push_to_talk = enabled
            return True

    def get_stats(self) -> dict[str, Any]:
        """Get voice chat statistics."""
        with self._lock:
            channel_counts: dict[str, int] = {}
            for cid, info in self._channels.items():
                channel_counts[cid] = len(info.participants)

            return {
                "total_participants": len(self._participants),
                "active_speakers": len(self._active_speakers),
                "channels": channel_counts
            }
