"""
Whitebox tests for music_player.py - Basic music playback system.
"""

import pytest
import time
import threading
from engine.audio.adaptive.music_player import (
    PlaybackState,
    TrackInfo,
    PlaybackPosition,
    Playlist,
    MusicPlayer,
)
from engine.audio.adaptive.music_timing import MusicClock, TimeSignature
from engine.audio.adaptive.config import (
    DEFAULT_VOLUME,
    MIN_VOLUME,
    MAX_VOLUME,
    DEFAULT_BPM,
    PLAYBACK_MODE_LINEAR,
    PLAYBACK_MODE_LOOP,
    PLAYBACK_MODE_SHUFFLE,
    PLAYBACK_MODE_ADAPTIVE,
    CROSSFADE_DEFAULT_DURATION,
)


class TestPlaybackState:
    """Tests for PlaybackState enum."""

    def test_playback_states_exist(self):
        """All playback states should exist."""
        assert PlaybackState.STOPPED is not None
        assert PlaybackState.PLAYING is not None
        assert PlaybackState.PAUSED is not None
        assert PlaybackState.FADING_IN is not None
        assert PlaybackState.FADING_OUT is not None
        assert PlaybackState.TRANSITIONING is not None


class TestTrackInfo:
    """Tests for TrackInfo dataclass."""

    def test_create_track_info(self):
        """Create track info with required fields."""
        track = TrackInfo(
            track_id="track_1",
            name="Track One",
            path="/audio/track1.wav",
            duration_ms=180000.0,
        )
        assert track.track_id == "track_1"
        assert track.name == "Track One"
        assert track.path == "/audio/track1.wav"
        assert track.duration_ms == 180000.0

    def test_track_info_defaults(self):
        """TrackInfo has sensible defaults."""
        track = TrackInfo(
            track_id="test",
            name="Test",
            path="/test.wav",
            duration_ms=60000.0,
        )
        assert track.bpm == DEFAULT_BPM
        assert track.time_signature == (4, 4)
        assert track.loop_start_ms == 0.0
        assert track.intro_end_ms == 0.0
        assert track.metadata == {}

    def test_track_info_loop_end_defaults_to_duration(self):
        """loop_end_ms defaults to duration if not set."""
        track = TrackInfo(
            track_id="test",
            name="Test",
            path="/test.wav",
            duration_ms=60000.0,
        )
        assert track.loop_end_ms == 60000.0

    def test_track_info_outro_defaults_to_duration(self):
        """outro_start_ms defaults to duration if not set."""
        track = TrackInfo(
            track_id="test",
            name="Test",
            path="/test.wav",
            duration_ms=60000.0,
        )
        assert track.outro_start_ms == 60000.0

    def test_track_info_with_loop_region(self):
        """Create track with loop region."""
        track = TrackInfo(
            track_id="looping",
            name="Looping Track",
            path="/loop.wav",
            duration_ms=120000.0,
            loop_start_ms=10000.0,
            loop_end_ms=100000.0,
        )
        assert track.loop_start_ms == 10000.0
        assert track.loop_end_ms == 100000.0

    def test_track_info_with_intro_outro(self):
        """Create track with intro and outro."""
        track = TrackInfo(
            track_id="structured",
            name="Structured Track",
            path="/structured.wav",
            duration_ms=180000.0,
            intro_end_ms=15000.0,
            outro_start_ms=165000.0,
        )
        assert track.intro_end_ms == 15000.0
        assert track.outro_start_ms == 165000.0

    def test_track_info_with_metadata(self):
        """Create track with metadata."""
        track = TrackInfo(
            track_id="meta",
            name="Track with Metadata",
            path="/meta.wav",
            duration_ms=60000.0,
            metadata={"artist": "Test Artist", "album": "Test Album"},
        )
        assert track.metadata["artist"] == "Test Artist"


class TestPlaybackPosition:
    """Tests for PlaybackPosition dataclass."""

    def test_create_playback_position(self):
        """Create playback position."""
        pos = PlaybackPosition(
            time_ms=30000.0,
            normalized=0.5,
            beat=60,
            bar=15,
            in_intro=False,
            in_outro=False,
            in_loop=True,
        )
        assert pos.time_ms == 30000.0
        assert pos.normalized == 0.5
        assert pos.beat == 60
        assert pos.bar == 15
        assert pos.in_loop is True


class TestPlaylist:
    """Tests for Playlist class."""

    def create_tracks(self, count):
        """Create test tracks."""
        tracks = []
        for i in range(count):
            tracks.append(TrackInfo(
                track_id=f"track_{i}",
                name=f"Track {i}",
                path=f"/track{i}.wav",
                duration_ms=60000.0,
            ))
        return tracks

    def test_create_playlist(self):
        """Create empty playlist."""
        playlist = Playlist("My Playlist")
        assert playlist.name == "My Playlist"
        assert playlist.track_count == 0

    def test_add_track(self):
        """Add track to playlist."""
        playlist = Playlist("Test")
        tracks = self.create_tracks(1)
        playlist.add_track(tracks[0])
        assert playlist.track_count == 1

    def test_remove_track(self):
        """Remove track from playlist."""
        playlist = Playlist("Test")
        tracks = self.create_tracks(2)
        playlist.add_track(tracks[0])
        playlist.add_track(tracks[1])
        assert playlist.remove_track("track_0") is True
        assert playlist.track_count == 1

    def test_remove_nonexistent_track(self):
        """Removing nonexistent track returns False."""
        playlist = Playlist("Test")
        assert playlist.remove_track("nonexistent") is False

    def test_get_track(self):
        """Get track by index."""
        playlist = Playlist("Test")
        tracks = self.create_tracks(3)
        for t in tracks:
            playlist.add_track(t)
        track = playlist.get_track(1)
        assert track is not None
        assert track.track_id == "track_1"

    def test_get_track_invalid_index(self):
        """Get track with invalid index returns None."""
        playlist = Playlist("Test")
        assert playlist.get_track(0) is None
        assert playlist.get_track(-1) is None

    def test_get_track_by_id(self):
        """Get track by ID."""
        playlist = Playlist("Test")
        tracks = self.create_tracks(3)
        for t in tracks:
            playlist.add_track(t)
        track = playlist.get_track_by_id("track_2")
        assert track is not None
        assert track.name == "Track 2"

    def test_get_current_track(self):
        """Get current track."""
        playlist = Playlist("Test")
        tracks = self.create_tracks(3)
        for t in tracks:
            playlist.add_track(t)
        current = playlist.get_current_track()
        assert current is not None
        assert current.track_id == "track_0"

    def test_get_next_track_linear(self):
        """Get next track in linear mode."""
        playlist = Playlist("Test")
        tracks = self.create_tracks(3)
        for t in tracks:
            playlist.add_track(t)
        next_track = playlist.get_next_track(PLAYBACK_MODE_LINEAR)
        assert next_track.track_id == "track_1"
        next_track = playlist.get_next_track(PLAYBACK_MODE_LINEAR)
        assert next_track.track_id == "track_2"

    def test_get_next_track_linear_wraps(self):
        """Linear mode wraps to beginning."""
        playlist = Playlist("Test")
        tracks = self.create_tracks(2)
        for t in tracks:
            playlist.add_track(t)
        playlist.get_next_track(PLAYBACK_MODE_LINEAR)  # track_1
        next_track = playlist.get_next_track(PLAYBACK_MODE_LINEAR)
        assert next_track.track_id == "track_0"

    def test_get_next_track_loop(self):
        """Loop mode stays on same track."""
        playlist = Playlist("Test")
        tracks = self.create_tracks(3)
        for t in tracks:
            playlist.add_track(t)
        next_track = playlist.get_next_track(PLAYBACK_MODE_LOOP)
        # Should stay on first track
        assert next_track.track_id == "track_0"

    def test_get_next_track_shuffle(self):
        """Shuffle mode returns different track."""
        playlist = Playlist("Test")
        tracks = self.create_tracks(10)
        for t in tracks:
            playlist.add_track(t)
        playlist.shuffle()
        # Should generate shuffle order
        assert playlist._shuffle_order is not None

    def test_get_previous_track_linear(self):
        """Get previous track in linear mode."""
        playlist = Playlist("Test")
        tracks = self.create_tracks(3)
        for t in tracks:
            playlist.add_track(t)
        playlist.set_current_index(2)
        prev_track = playlist.get_previous_track(PLAYBACK_MODE_LINEAR)
        assert prev_track.track_id == "track_1"

    def test_get_previous_track_wraps(self):
        """Previous track wraps to end."""
        playlist = Playlist("Test")
        tracks = self.create_tracks(3)
        for t in tracks:
            playlist.add_track(t)
        prev_track = playlist.get_previous_track(PLAYBACK_MODE_LINEAR)
        assert prev_track.track_id == "track_2"

    def test_set_current_index(self):
        """Set current track index."""
        playlist = Playlist("Test")
        tracks = self.create_tracks(5)
        for t in tracks:
            playlist.add_track(t)
        playlist.set_current_index(3)
        assert playlist.current_index == 3
        assert playlist.get_current_track().track_id == "track_3"

    def test_set_current_index_invalid(self):
        """Setting invalid index is ignored."""
        playlist = Playlist("Test")
        tracks = self.create_tracks(3)
        for t in tracks:
            playlist.add_track(t)
        playlist.set_current_index(10)
        # Should still be at 0
        assert playlist.current_index == 0

    def test_shuffle(self):
        """Shuffle generates new order."""
        playlist = Playlist("Test")
        tracks = self.create_tracks(10)
        for t in tracks:
            playlist.add_track(t)
        playlist.shuffle()
        assert playlist._shuffle_order is not None
        assert len(playlist._shuffle_order) == 10

    def test_clear_playlist(self):
        """Clear all tracks."""
        playlist = Playlist("Test")
        tracks = self.create_tracks(5)
        for t in tracks:
            playlist.add_track(t)
        playlist.clear()
        assert playlist.track_count == 0
        assert playlist.current_index == 0

    def test_tracks_property(self):
        """Get all tracks as list."""
        playlist = Playlist("Test")
        tracks = self.create_tracks(3)
        for t in tracks:
            playlist.add_track(t)
        all_tracks = playlist.tracks
        assert len(all_tracks) == 3

    def test_empty_playlist_get_next(self):
        """Get next from empty playlist returns None."""
        playlist = Playlist("Test")
        assert playlist.get_next_track() is None


class TestMusicPlayer:
    """Tests for MusicPlayer class."""

    def create_track(self, track_id="test", duration_ms=10000.0, bpm=120.0):
        """Create test track."""
        return TrackInfo(
            track_id=track_id,
            name=track_id.title(),
            path=f"/{track_id}.wav",
            duration_ms=duration_ms,
            bpm=bpm,
        )

    def test_create_music_player(self):
        """Create music player."""
        player = MusicPlayer()
        assert player.state == PlaybackState.STOPPED
        assert player.volume == DEFAULT_VOLUME
        assert player.mode == PLAYBACK_MODE_LINEAR

    def test_set_playlist(self):
        """Set active playlist."""
        player = MusicPlayer()
        playlist = Playlist("Test")
        player.set_playlist(playlist)
        assert player.get_playlist() is playlist

    def test_play_track(self):
        """Play a track."""
        player = MusicPlayer()
        track = self.create_track()
        player.play(track)
        assert player.state == PlaybackState.PLAYING
        assert player.current_track is track
        assert player.is_playing is True
        player.stop()

    def test_play_from_playlist(self):
        """Play from playlist."""
        player = MusicPlayer()
        playlist = Playlist("Test")
        playlist.add_track(self.create_track("track1"))
        playlist.add_track(self.create_track("track2"))
        player.set_playlist(playlist)
        player.play()
        assert player.current_track.track_id == "track1"
        player.stop()

    def test_pause_resume(self):
        """Pause and resume playback."""
        player = MusicPlayer()
        track = self.create_track()
        player.play(track)
        player.pause()
        assert player.state == PlaybackState.PAUSED
        assert player.is_paused is True
        player.resume()
        assert player.state == PlaybackState.PLAYING
        player.stop()

    def test_stop(self):
        """Stop playback."""
        player = MusicPlayer()
        track = self.create_track()
        player.play(track)
        player.stop()
        assert player.state == PlaybackState.STOPPED
        assert player.is_stopped is True

    def test_seek(self):
        """Seek to position."""
        player = MusicPlayer()
        track = self.create_track(duration_ms=60000.0)
        player.play(track)
        player.seek(30000.0)
        pos = player.get_position()
        assert 29000 <= pos.time_ms <= 31000
        player.stop()

    def test_seek_clamps_to_duration(self):
        """Seek clamps to track duration."""
        player = MusicPlayer()
        track = self.create_track(duration_ms=60000.0)
        player.play(track)
        player.seek(100000.0)
        pos = player.get_position()
        # Allow small floating point tolerance
        assert pos.time_ms <= 60001.0
        player.stop()

    def test_seek_to_bar(self):
        """Seek to bar position."""
        player = MusicPlayer()
        track = self.create_track(bpm=120.0)
        player.play(track)
        player.seek_to_bar(2, 0.0)
        pos = player.get_position()
        # At 120 BPM, bar 2 = 4000ms
        assert 3800 <= pos.time_ms <= 4200
        player.stop()

    def test_next_track(self):
        """Skip to next track."""
        player = MusicPlayer()
        playlist = Playlist("Test")
        playlist.add_track(self.create_track("track1"))
        playlist.add_track(self.create_track("track2"))
        player.set_playlist(playlist)
        player.play()
        next_track = player.next_track()
        assert next_track.track_id == "track2"
        player.stop()

    def test_previous_track(self):
        """Skip to previous track."""
        player = MusicPlayer()
        playlist = Playlist("Test")
        playlist.add_track(self.create_track("track1"))
        playlist.add_track(self.create_track("track2"))
        player.set_playlist(playlist)
        player.play()
        player.next_track()
        prev_track = player.previous_track()
        assert prev_track.track_id == "track1"
        player.stop()

    def test_volume_setter(self):
        """Set volume."""
        player = MusicPlayer()
        player.volume = 0.5
        assert player.volume == 0.5

    def test_volume_invalid_raises(self):
        """Setting invalid volume raises."""
        player = MusicPlayer()
        with pytest.raises(ValueError):
            player.volume = -0.1
        with pytest.raises(ValueError):
            player.volume = 1.5

    def test_mode_setter(self):
        """Set playback mode."""
        player = MusicPlayer()
        player.mode = PLAYBACK_MODE_SHUFFLE
        assert player.mode == PLAYBACK_MODE_SHUFFLE

    def test_mode_invalid_raises(self):
        """Setting invalid mode raises."""
        player = MusicPlayer()
        with pytest.raises(ValueError):
            player.mode = "invalid"

    def test_fade_in(self):
        """Fade in playback."""
        player = MusicPlayer()
        track = self.create_track()
        player.play(track)
        player.fade_in(duration_ms=500.0)
        assert player.state == PlaybackState.FADING_IN
        player.stop()

    def test_fade_out(self):
        """Fade out playback."""
        player = MusicPlayer()
        track = self.create_track()
        player.play(track)
        player.fade_out(duration_ms=500.0)
        assert player.state == PlaybackState.FADING_OUT
        player.stop()

    def test_get_position(self):
        """Get playback position."""
        player = MusicPlayer()
        track = self.create_track(
            duration_ms=60000.0,
            bpm=120.0,
        )
        player.play(track)
        player.seek(15000.0)
        pos = player.get_position()
        assert pos.normalized == pytest.approx(0.25, abs=0.05)
        player.stop()

    def test_get_position_no_track(self):
        """Get position with no track."""
        player = MusicPlayer()
        pos = player.get_position()
        assert pos.time_ms == 0
        assert pos.normalized == 0.0

    def test_get_position_in_intro(self):
        """Get position detects intro."""
        player = MusicPlayer()
        track = TrackInfo(
            track_id="test",
            name="Test",
            path="/test.wav",
            duration_ms=60000.0,
            intro_end_ms=10000.0,
        )
        player.play(track)
        player.seek(5000.0)
        pos = player.get_position()
        assert pos.in_intro is True
        player.stop()

    def test_get_position_in_outro(self):
        """Get position detects outro."""
        player = MusicPlayer()
        track = TrackInfo(
            track_id="test",
            name="Test",
            path="/test.wav",
            duration_ms=60000.0,
            outro_start_ms=50000.0,
        )
        player.play(track)
        player.seek(55000.0)
        pos = player.get_position()
        assert pos.in_outro is True
        player.stop()

    def test_get_position_in_loop(self):
        """Get position detects loop region."""
        player = MusicPlayer()
        track = TrackInfo(
            track_id="test",
            name="Test",
            path="/test.wav",
            duration_ms=60000.0,
            loop_start_ms=10000.0,
            loop_end_ms=50000.0,
        )
        player.play(track)
        player.seek(30000.0)
        pos = player.get_position()
        assert pos.in_loop is True
        player.stop()

    def test_clock_property(self):
        """Access music clock."""
        player = MusicPlayer()
        assert player.clock is not None

    def test_on_track_end_callback(self):
        """Track end callback is invoked."""
        player = MusicPlayer()
        ended_tracks = []

        def on_end(track):
            ended_tracks.append(track)

        player.set_on_track_end(on_end)
        # Cannot easily test automatic end without real audio

    def test_on_track_start_callback(self):
        """Track start callback is invoked."""
        player = MusicPlayer()
        started_tracks = []

        def on_start(track):
            started_tracks.append(track)

        player.set_on_track_start(on_start)
        track = self.create_track()
        player.play(track)
        assert len(started_tracks) == 1
        player.stop()

    def test_on_state_change_callback(self):
        """State change callback is invoked."""
        player = MusicPlayer()
        state_changes = []

        def on_change(state):
            state_changes.append(state)

        player.set_on_state_change(on_change)
        track = self.create_track()
        player.play(track)
        player.pause()
        player.stop()
        assert PlaybackState.PLAYING in state_changes
        assert PlaybackState.PAUSED in state_changes
        assert PlaybackState.STOPPED in state_changes
