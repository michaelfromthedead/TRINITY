"""
Basic music playback system with linear, loop, and shuffle modes.

Provides the foundation for music playback in the game engine.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable, List, Any
import threading
import random
import time

from .config import (
    DEFAULT_VOLUME,
    MIN_VOLUME,
    MAX_VOLUME,
    DEFAULT_BPM,
    PLAYBACK_MODE_LINEAR,
    PLAYBACK_MODE_LOOP,
    PLAYBACK_MODE_SHUFFLE,
    PLAYBACK_MODE_ADAPTIVE,
    VALID_PLAYBACK_MODES,
    CROSSFADE_DEFAULT_DURATION,
)
from .music_timing import MusicClock, TimeSignature


class PlaybackState(Enum):
    """Music playback states."""
    STOPPED = auto()
    PLAYING = auto()
    PAUSED = auto()
    FADING_IN = auto()
    FADING_OUT = auto()
    TRANSITIONING = auto()


@dataclass
class TrackInfo:
    """Information about a music track.

    Attributes:
        track_id: Unique identifier for the track
        name: Display name
        path: File path or resource identifier
        duration_ms: Track duration in milliseconds
        bpm: Beats per minute
        time_signature: Musical time signature
        loop_start_ms: Start of loop region
        loop_end_ms: End of loop region
        intro_end_ms: End of intro section
        outro_start_ms: Start of outro section
        metadata: Additional track metadata
    """
    track_id: str
    name: str
    path: str
    duration_ms: float
    bpm: float = DEFAULT_BPM
    time_signature: tuple[int, int] = (4, 4)
    loop_start_ms: float = 0.0
    loop_end_ms: float = 0.0
    intro_end_ms: float = 0.0
    outro_start_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        # Set loop_end to duration if not specified
        if self.loop_end_ms == 0.0:
            self.loop_end_ms = self.duration_ms
        if self.outro_start_ms == 0.0:
            self.outro_start_ms = self.duration_ms


@dataclass
class PlaybackPosition:
    """Current playback position information.

    Attributes:
        time_ms: Current time in milliseconds
        normalized: Position as 0.0-1.0
        beat: Current beat number
        bar: Current bar number
        in_intro: Whether in intro section
        in_outro: Whether in outro section
        in_loop: Whether in loop region
    """
    time_ms: float
    normalized: float
    beat: int
    bar: int
    in_intro: bool
    in_outro: bool
    in_loop: bool


class Playlist:
    """A playlist of tracks.

    Attributes:
        name: Playlist name
        tracks: List of tracks
    """

    def __init__(self, name: str):
        """Initialize playlist.

        Args:
            name: Playlist name
        """
        self.name = name
        self._tracks: List[TrackInfo] = []
        self._current_index = 0
        self._shuffle_order: Optional[List[int]] = None
        self._lock = threading.RLock()

    def add_track(self, track: TrackInfo):
        """Add a track to the playlist.

        Args:
            track: Track to add
        """
        with self._lock:
            self._tracks.append(track)

    def remove_track(self, track_id: str) -> bool:
        """Remove a track from the playlist.

        Args:
            track_id: ID of track to remove

        Returns:
            True if track was found and removed
        """
        with self._lock:
            for i, track in enumerate(self._tracks):
                if track.track_id == track_id:
                    self._tracks.pop(i)
                    if self._current_index >= len(self._tracks):
                        self._current_index = max(0, len(self._tracks) - 1)
                    return True
            return False

    def get_track(self, index: int) -> Optional[TrackInfo]:
        """Get track at index.

        Args:
            index: Track index

        Returns:
            TrackInfo or None
        """
        with self._lock:
            if 0 <= index < len(self._tracks):
                return self._tracks[index]
            return None

    def get_track_by_id(self, track_id: str) -> Optional[TrackInfo]:
        """Get track by ID.

        Args:
            track_id: Track identifier

        Returns:
            TrackInfo or None
        """
        with self._lock:
            for track in self._tracks:
                if track.track_id == track_id:
                    return track
            return None

    def get_current_track(self) -> Optional[TrackInfo]:
        """Get current track.

        Returns:
            Current TrackInfo or None
        """
        return self.get_track(self._current_index)

    def get_next_track(self, mode: str = PLAYBACK_MODE_LINEAR) -> Optional[TrackInfo]:
        """Get next track based on playback mode.

        Args:
            mode: Playback mode

        Returns:
            Next TrackInfo or None
        """
        with self._lock:
            if len(self._tracks) == 0:
                return None

            if mode == PLAYBACK_MODE_SHUFFLE:
                if self._shuffle_order is None:
                    self._generate_shuffle_order()
                # Find current position in shuffle order
                try:
                    shuffle_pos = self._shuffle_order.index(self._current_index)
                    next_shuffle_pos = (shuffle_pos + 1) % len(self._shuffle_order)
                    self._current_index = self._shuffle_order[next_shuffle_pos]
                except ValueError:
                    self._current_index = self._shuffle_order[0]
            elif mode == PLAYBACK_MODE_LOOP:
                # Loop stays on same track
                pass
            else:  # LINEAR or ADAPTIVE
                self._current_index = (self._current_index + 1) % len(self._tracks)

            return self._tracks[self._current_index]

    def get_previous_track(self, mode: str = PLAYBACK_MODE_LINEAR) -> Optional[TrackInfo]:
        """Get previous track based on playback mode.

        Args:
            mode: Playback mode

        Returns:
            Previous TrackInfo or None
        """
        with self._lock:
            if len(self._tracks) == 0:
                return None

            if mode == PLAYBACK_MODE_SHUFFLE:
                if self._shuffle_order is None:
                    self._generate_shuffle_order()
                try:
                    shuffle_pos = self._shuffle_order.index(self._current_index)
                    prev_shuffle_pos = (shuffle_pos - 1) % len(self._shuffle_order)
                    self._current_index = self._shuffle_order[prev_shuffle_pos]
                except ValueError:
                    self._current_index = self._shuffle_order[-1]
            elif mode == PLAYBACK_MODE_LOOP:
                pass
            else:
                self._current_index = (self._current_index - 1) % len(self._tracks)

            return self._tracks[self._current_index]

    def set_current_index(self, index: int):
        """Set current track index.

        Args:
            index: Track index
        """
        with self._lock:
            if 0 <= index < len(self._tracks):
                self._current_index = index

    def shuffle(self):
        """Generate new shuffle order."""
        self._generate_shuffle_order()

    def _generate_shuffle_order(self):
        """Generate random shuffle order."""
        with self._lock:
            self._shuffle_order = list(range(len(self._tracks)))
            random.shuffle(self._shuffle_order)

    def clear(self):
        """Clear all tracks from playlist."""
        with self._lock:
            self._tracks.clear()
            self._current_index = 0
            self._shuffle_order = None

    @property
    def track_count(self) -> int:
        """Get number of tracks in playlist."""
        return len(self._tracks)

    @property
    def current_index(self) -> int:
        """Get current track index."""
        return self._current_index

    @property
    def tracks(self) -> List[TrackInfo]:
        """Get list of all tracks."""
        with self._lock:
            return self._tracks.copy()


class MusicPlayer:
    """Basic music player with playback modes.

    Provides linear, loop, shuffle, and adaptive playback modes.
    """

    def __init__(self):
        """Initialize music player."""
        self._playlist: Optional[Playlist] = None
        self._current_track: Optional[TrackInfo] = None
        self._state = PlaybackState.STOPPED
        self._mode = PLAYBACK_MODE_LINEAR
        self._volume = DEFAULT_VOLUME
        self._clock = MusicClock()
        self._lock = threading.RLock()

        # Fade state
        self._fade_start_volume = 0.0
        self._fade_target_volume = 0.0
        self._fade_duration_ms = 0.0
        self._fade_start_time = 0.0

        # Callbacks
        self._on_track_end: Optional[Callable[[TrackInfo], None]] = None
        self._on_track_start: Optional[Callable[[TrackInfo], None]] = None
        self._on_state_change: Optional[Callable[[PlaybackState], None]] = None

        # Update thread
        self._running = False
        self._update_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def state(self) -> PlaybackState:
        """Get current playback state."""
        return self._state

    @property
    def mode(self) -> str:
        """Get current playback mode."""
        return self._mode

    @mode.setter
    def mode(self, value: str):
        """Set playback mode."""
        if value not in VALID_PLAYBACK_MODES:
            raise ValueError(f"Invalid playback mode: {value}")
        with self._lock:
            self._mode = value

    @property
    def volume(self) -> float:
        """Get current volume."""
        return self._volume

    @volume.setter
    def volume(self, value: float):
        """Set volume (0.0 to 1.0)."""
        if value < MIN_VOLUME or value > MAX_VOLUME:
            raise ValueError(f"Volume must be between {MIN_VOLUME} and {MAX_VOLUME}")
        with self._lock:
            self._volume = value

    @property
    def current_track(self) -> Optional[TrackInfo]:
        """Get currently playing track."""
        return self._current_track

    @property
    def clock(self) -> MusicClock:
        """Get the music clock."""
        return self._clock

    @property
    def is_playing(self) -> bool:
        """Check if currently playing."""
        return self._state == PlaybackState.PLAYING

    @property
    def is_paused(self) -> bool:
        """Check if paused."""
        return self._state == PlaybackState.PAUSED

    @property
    def is_stopped(self) -> bool:
        """Check if stopped."""
        return self._state == PlaybackState.STOPPED

    def set_playlist(self, playlist: Playlist):
        """Set the active playlist.

        Args:
            playlist: Playlist to use
        """
        with self._lock:
            self._playlist = playlist

    def get_playlist(self) -> Optional[Playlist]:
        """Get the active playlist."""
        return self._playlist

    def play(self, track: Optional[TrackInfo] = None):
        """Start playback.

        Args:
            track: Optional track to play (uses playlist current if None)
        """
        with self._lock:
            if track is not None:
                self._current_track = track
            elif self._playlist is not None:
                self._current_track = self._playlist.get_current_track()

            if self._current_track is None:
                return

            # Configure clock for this track
            self._clock.bpm = self._current_track.bpm
            self._clock.time_signature = TimeSignature.from_tuple(
                self._current_track.time_signature
            )

            if self._state == PlaybackState.PAUSED:
                self._clock.resume()
            else:
                self._clock.stop()
                self._clock.start()

            self._set_state(PlaybackState.PLAYING)

            if self._on_track_start is not None:
                self._on_track_start(self._current_track)

    def pause(self):
        """Pause playback."""
        with self._lock:
            if self._state == PlaybackState.PLAYING:
                self._clock.pause()
                self._set_state(PlaybackState.PAUSED)

    def resume(self):
        """Resume playback from pause."""
        with self._lock:
            if self._state == PlaybackState.PAUSED:
                self._clock.resume()
                self._set_state(PlaybackState.PLAYING)

    def stop(self):
        """Stop playback."""
        with self._lock:
            self._clock.stop()
            self._set_state(PlaybackState.STOPPED)

    def seek(self, time_ms: float):
        """Seek to a position.

        Args:
            time_ms: Target time in milliseconds
        """
        with self._lock:
            if self._current_track is not None:
                time_ms = max(0, min(time_ms, self._current_track.duration_ms))
            self._clock.seek(time_ms)

    def seek_to_bar(self, bar: int, beat: float = 0.0):
        """Seek to a specific bar and beat.

        Args:
            bar: Bar number
            beat: Beat within bar
        """
        self._clock.seek_to_bar(bar, beat)

    def next_track(self) -> Optional[TrackInfo]:
        """Skip to next track.

        Returns:
            New current track or None
        """
        with self._lock:
            if self._playlist is None:
                return None

            track = self._playlist.get_next_track(self._mode)
            if track is not None:
                was_playing = self._state == PlaybackState.PLAYING
                self.stop()
                if was_playing:
                    self.play(track)
                else:
                    self._current_track = track
            return track

    def previous_track(self) -> Optional[TrackInfo]:
        """Skip to previous track.

        Returns:
            New current track or None
        """
        with self._lock:
            if self._playlist is None:
                return None

            track = self._playlist.get_previous_track(self._mode)
            if track is not None:
                was_playing = self._state == PlaybackState.PLAYING
                self.stop()
                if was_playing:
                    self.play(track)
                else:
                    self._current_track = track
            return track

    def fade_in(self, duration_ms: float = CROSSFADE_DEFAULT_DURATION * 1000):
        """Start fade in.

        Args:
            duration_ms: Fade duration in milliseconds
        """
        with self._lock:
            self._fade_start_volume = 0.0
            self._fade_target_volume = self._volume
            self._fade_duration_ms = duration_ms
            self._fade_start_time = time.perf_counter() * 1000
            self._volume = 0.0
            self._set_state(PlaybackState.FADING_IN)

    def fade_out(self, duration_ms: float = CROSSFADE_DEFAULT_DURATION * 1000):
        """Start fade out.

        Args:
            duration_ms: Fade duration in milliseconds
        """
        with self._lock:
            self._fade_start_volume = self._volume
            self._fade_target_volume = 0.0
            self._fade_duration_ms = duration_ms
            self._fade_start_time = time.perf_counter() * 1000
            self._set_state(PlaybackState.FADING_OUT)

    def get_position(self) -> PlaybackPosition:
        """Get current playback position.

        Returns:
            PlaybackPosition with current state
        """
        with self._lock:
            time_ms = self._clock.get_time_ms()

            if self._current_track is None:
                return PlaybackPosition(
                    time_ms=0,
                    normalized=0.0,
                    beat=0,
                    bar=0,
                    in_intro=False,
                    in_outro=False,
                    in_loop=False,
                )

            duration = self._current_track.duration_ms
            normalized = time_ms / duration if duration > 0 else 0.0

            beat_info = self._clock.get_beat_info()

            return PlaybackPosition(
                time_ms=time_ms,
                normalized=min(1.0, normalized),
                beat=beat_info.total_beats,
                bar=beat_info.bar,
                in_intro=time_ms < self._current_track.intro_end_ms,
                in_outro=time_ms >= self._current_track.outro_start_ms,
                in_loop=(self._current_track.loop_start_ms <= time_ms <
                        self._current_track.loop_end_ms),
            )

    def _set_state(self, state: PlaybackState):
        """Set playback state and trigger callback.

        Args:
            state: New state
        """
        old_state = self._state
        self._state = state
        if self._on_state_change is not None and old_state != state:
            self._on_state_change(state)

    def _update(self):
        """Update player state (called from update loop)."""
        with self._lock:
            # Handle fades
            if self._state in (PlaybackState.FADING_IN, PlaybackState.FADING_OUT):
                self._update_fade()

            # Check for track end
            if self._state == PlaybackState.PLAYING and self._current_track is not None:
                time_ms = self._clock.get_time_ms()

                if self._mode == PLAYBACK_MODE_LOOP:
                    # Loop within loop region
                    if time_ms >= self._current_track.loop_end_ms:
                        self._clock.seek(self._current_track.loop_start_ms)
                elif time_ms >= self._current_track.duration_ms:
                    self._handle_track_end()

    def _update_fade(self):
        """Update fade state."""
        current_time = time.perf_counter() * 1000
        elapsed = current_time - self._fade_start_time
        progress = 1.0 if self._fade_duration_ms <= 0 else min(1.0, elapsed / self._fade_duration_ms)

        # Linear interpolation
        self._volume = (
            self._fade_start_volume +
            (self._fade_target_volume - self._fade_start_volume) * progress
        )

        if progress >= 1.0:
            if self._state == PlaybackState.FADING_IN:
                self._set_state(PlaybackState.PLAYING)
            elif self._state == PlaybackState.FADING_OUT:
                self.stop()

    def _handle_track_end(self):
        """Handle track ending."""
        if self._current_track is not None and self._on_track_end is not None:
            self._on_track_end(self._current_track)

        if self._mode == PLAYBACK_MODE_LINEAR:
            next_track = self.next_track()
            if next_track is None:
                self.stop()
        elif self._mode == PLAYBACK_MODE_SHUFFLE:
            self.next_track()

    def set_on_track_end(self, callback: Optional[Callable[[TrackInfo], None]]):
        """Set callback for track end events.

        Args:
            callback: Function to call when track ends
        """
        self._on_track_end = callback

    def set_on_track_start(self, callback: Optional[Callable[[TrackInfo], None]]):
        """Set callback for track start events.

        Args:
            callback: Function to call when track starts
        """
        self._on_track_start = callback

    def set_on_state_change(self, callback: Optional[Callable[[PlaybackState], None]]):
        """Set callback for state change events.

        Args:
            callback: Function to call when state changes
        """
        self._on_state_change = callback

    def start_update_loop(self, interval_ms: float = 10.0):
        """Start the update loop thread.

        Args:
            interval_ms: Update interval in milliseconds
        """
        if self._running:
            return

        self._running = True
        self._stop_event.clear()

        def update_loop():
            while not self._stop_event.is_set():
                self._update()
                time.sleep(interval_ms / 1000.0)

        self._update_thread = threading.Thread(target=update_loop, daemon=True)
        self._update_thread.start()

    def stop_update_loop(self):
        """Stop the update loop thread."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._update_thread is not None:
            self._update_thread.join(timeout=1.0)
            self._update_thread = None
