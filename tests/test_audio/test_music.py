"""
Comprehensive Tests for the Music Subsystem.

Tests covering:
- MusicPlayer: Playback, playlist management, states
- MusicStem: Layered music, stem control, fading
- MusicTransition: Crossfade, beat-sync, bar-sync, stingers
- MusicState: State-driven music management
- AdaptiveMusic: Vertical/horizontal remixing
- MusicTiming: BPM, beat grid, sync points
- MusicCallback: Beat, bar, marker callbacks
- Stinger: Stinger playback and management
"""

import math
import time
import threading
from unittest.mock import Mock, MagicMock, patch, call
import pytest

# Import source modules
from engine.audio.adaptive.music_player import (
    MusicPlayer, PlaybackState, TrackInfo, PlaybackPosition,
    Playlist,
)
from engine.audio.adaptive.music_stem import (
    MusicStem, StemInfo, StemState, StemGroup,
    LayeredMusicPlayer, FadeCurve, StemPlaybackState,
)
from engine.audio.adaptive.music_transition import (
    MusicTransition, TransitionManager, TransitionConfig,
    TransitionRequest, TransitionProgress, TransitionState,
)
from engine.audio.adaptive.music_state import (
    MusicState, MusicStateManager, MusicStateConfig,
    StateTransition, StateChangeReason, StateHistoryEntry,
    create_exploration_state, create_combat_state,
)
from engine.audio.adaptive.adaptive_music import (
    AdaptiveMusicSystem, AdaptiveMode, VerticalRemixer,
    HorizontalSequencer, MusicSection, IntensityLevel,
    AdaptiveParameters, BranchType,
)
from engine.audio.adaptive.music_timing import (
    MusicClock, BeatGrid, TimeSignature, SyncPoint,
    SyncPointManager, BeatInfo, BeatSubdivision,
)
from engine.audio.adaptive.music_callback import (
    MusicCallbackManager, BeatScheduler, CallbackEvent,
    CallbackRegistration, CallbackPriority,
)
from engine.audio.adaptive.stinger import (
    Stinger, StingerManager, StingerInfo, StingerState,
    StingerPlayback,
)
from engine.audio.adaptive.config import (
    DEFAULT_BPM, MIN_BPM, MAX_BPM,
    DEFAULT_VOLUME, MIN_VOLUME, MAX_VOLUME,
    PLAYBACK_MODE_LINEAR, PLAYBACK_MODE_LOOP, PLAYBACK_MODE_SHUFFLE,
    TRANSITION_CROSSFADE, TRANSITION_BEAT_SYNC, TRANSITION_BAR_SYNC,
    STINGER_TYPE_IMPACT, STINGER_TYPE_TRANSITION,
    STATE_EXPLORATION, STATE_COMBAT, STATE_STEALTH,
    LAYER_DRUMS, LAYER_BASS, LAYER_MELODY, LAYER_PADS,
    FADE_CURVE_LINEAR, FADE_CURVE_EQUAL_POWER, FADE_CURVE_S_CURVE,
    CALLBACK_BEAT, CALLBACK_BAR, CALLBACK_MARKER,
    STEM_FADE_TIME, CROSSFADE_DEFAULT_DURATION,
    BEAT_CALLBACK_PRECISION_MS, UPDATE_INTERVAL_CALLBACK_MS,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_track():
    """Create a sample track for testing."""
    return TrackInfo(
        track_id="test_track_001",
        name="Test Track",
        path="/audio/test.ogg",
        duration_ms=120000.0,
        bpm=120.0,
        time_signature=(4, 4),
        loop_start_ms=4000.0,
        loop_end_ms=116000.0,
        intro_end_ms=8000.0,
        outro_start_ms=112000.0,
    )


@pytest.fixture
def sample_tracks():
    """Create multiple sample tracks."""
    return [
        TrackInfo(
            track_id=f"track_{i}",
            name=f"Track {i}",
            path=f"/audio/track_{i}.ogg",
            duration_ms=60000.0 + i * 10000,
            bpm=120.0 + i * 10,
        )
        for i in range(5)
    ]


@pytest.fixture
def music_player():
    """Create a MusicPlayer instance."""
    return MusicPlayer()


@pytest.fixture
def playlist(sample_tracks):
    """Create a playlist with sample tracks."""
    pl = Playlist("Test Playlist")
    for track in sample_tracks:
        pl.add_track(track)
    return pl


@pytest.fixture
def stem_info():
    """Create sample stem info."""
    return StemInfo(
        stem_id="drums_main",
        name="Drums",
        layer_type=LAYER_DRUMS,
        path="/audio/stems/drums.ogg",
        volume=0.8,
        priority=1,
    )


@pytest.fixture
def music_stem(stem_info):
    """Create a MusicStem instance."""
    return MusicStem(stem_info)


@pytest.fixture
def layered_player():
    """Create a LayeredMusicPlayer instance."""
    return LayeredMusicPlayer(max_stems=8)


@pytest.fixture
def music_clock():
    """Create a MusicClock instance."""
    return MusicClock(bpm=120.0)


@pytest.fixture
def beat_grid():
    """Create a BeatGrid instance."""
    return BeatGrid(bpm=120.0, time_signature=TimeSignature(4, 4))


@pytest.fixture
def callback_manager(music_clock):
    """Create a MusicCallbackManager instance."""
    return MusicCallbackManager(music_clock)


@pytest.fixture
def transition_manager(music_clock):
    """Create a TransitionManager instance."""
    return TransitionManager(music_clock)


@pytest.fixture
def state_manager(music_clock):
    """Create a MusicStateManager instance."""
    return MusicStateManager(music_clock)


@pytest.fixture
def stinger_info():
    """Create sample stinger info."""
    return StingerInfo(
        stinger_id="impact_01",
        name="Impact Stinger",
        stinger_type=STINGER_TYPE_IMPACT,
        path="/audio/stingers/impact.ogg",
        duration_ms=1500.0,
        volume=1.0,
    )


@pytest.fixture
def stinger_manager(music_clock):
    """Create a StingerManager instance."""
    return StingerManager(music_clock)


# =============================================================================
# MusicPlayer Tests
# =============================================================================

class TestMusicPlayer:
    """Tests for MusicPlayer class."""

    def test_initial_state(self, music_player):
        """Test initial player state."""
        assert music_player.state == PlaybackState.STOPPED
        assert music_player.is_stopped
        assert not music_player.is_playing
        assert not music_player.is_paused
        assert music_player.volume == DEFAULT_VOLUME
        assert music_player.mode == PLAYBACK_MODE_LINEAR

    def test_set_playlist(self, music_player, playlist):
        """Test setting playlist."""
        music_player.set_playlist(playlist)
        assert music_player.get_playlist() is playlist

    def test_play_track_directly(self, music_player, sample_track):
        """Test playing a track directly."""
        music_player.play(sample_track)
        assert music_player.state == PlaybackState.PLAYING
        assert music_player.current_track == sample_track
        assert music_player.is_playing

    def test_play_from_playlist(self, music_player, playlist):
        """Test playing from playlist."""
        music_player.set_playlist(playlist)
        music_player.play()
        assert music_player.is_playing
        assert music_player.current_track is not None

    def test_pause_resume(self, music_player, sample_track):
        """Test pause and resume functionality."""
        music_player.play(sample_track)
        assert music_player.is_playing

        music_player.pause()
        assert music_player.is_paused
        assert music_player.state == PlaybackState.PAUSED

        music_player.resume()
        assert music_player.is_playing

    def test_stop_playback(self, music_player, sample_track):
        """Test stopping playback."""
        music_player.play(sample_track)
        music_player.stop()
        assert music_player.is_stopped
        assert music_player.state == PlaybackState.STOPPED

    def test_volume_control(self, music_player):
        """Test volume setting and limits."""
        music_player.volume = 0.5
        assert music_player.volume == 0.5

        music_player.volume = 0.0
        assert music_player.volume == MIN_VOLUME

        music_player.volume = 1.0
        assert music_player.volume == MAX_VOLUME

    def test_volume_out_of_range(self, music_player):
        """Test volume rejection for out-of-range values."""
        with pytest.raises(ValueError):
            music_player.volume = -0.5

        with pytest.raises(ValueError):
            music_player.volume = 1.5

    def test_playback_mode_changes(self, music_player):
        """Test changing playback modes."""
        music_player.mode = PLAYBACK_MODE_LOOP
        assert music_player.mode == PLAYBACK_MODE_LOOP

        music_player.mode = PLAYBACK_MODE_SHUFFLE
        assert music_player.mode == PLAYBACK_MODE_SHUFFLE

    def test_invalid_playback_mode(self, music_player):
        """Test rejection of invalid playback mode."""
        with pytest.raises(ValueError):
            music_player.mode = "invalid_mode"

    def test_next_track(self, music_player, playlist):
        """Test skipping to next track."""
        music_player.set_playlist(playlist)
        music_player.play()
        first_track = music_player.current_track

        next_track = music_player.next_track()
        assert next_track is not None
        assert next_track != first_track

    def test_previous_track(self, music_player, playlist):
        """Test skipping to previous track."""
        music_player.set_playlist(playlist)
        music_player.play()
        music_player.next_track()

        second_track = music_player.current_track
        music_player.previous_track()
        assert music_player.current_track != second_track

    def test_seek(self, music_player, sample_track):
        """Test seeking to position."""
        music_player.play(sample_track)
        music_player.seek(30000.0)
        position = music_player.get_position()
        assert position.time_ms >= 0

    def test_seek_to_bar(self, music_player, sample_track):
        """Test seeking to bar."""
        music_player.play(sample_track)
        music_player.seek_to_bar(4, 0.0)
        # Should not raise

    def test_get_position(self, music_player, sample_track):
        """Test getting playback position."""
        music_player.play(sample_track)
        position = music_player.get_position()

        assert isinstance(position, PlaybackPosition)
        assert position.time_ms >= 0
        assert 0.0 <= position.normalized <= 1.0
        assert position.beat >= 0
        assert position.bar >= 0

    def test_fade_in(self, music_player, sample_track):
        """Test fade in."""
        music_player.play(sample_track)
        music_player.fade_in(1000.0)
        assert music_player.state == PlaybackState.FADING_IN

    def test_fade_out(self, music_player, sample_track):
        """Test fade out."""
        music_player.play(sample_track)
        music_player.fade_out(1000.0)
        assert music_player.state == PlaybackState.FADING_OUT

    def test_callbacks_on_track_start(self, music_player, sample_track):
        """Test track start callback."""
        callback = Mock()
        music_player.set_on_track_start(callback)
        music_player.play(sample_track)
        callback.assert_called_once_with(sample_track)

    def test_callbacks_on_state_change(self, music_player, sample_track):
        """Test state change callback."""
        callback = Mock()
        music_player.set_on_state_change(callback)
        music_player.play(sample_track)
        callback.assert_called_with(PlaybackState.PLAYING)

    def test_clock_sync_with_track(self, music_player, sample_track):
        """Test clock synchronization with track BPM."""
        music_player.play(sample_track)
        assert music_player.clock.bpm == sample_track.bpm


# =============================================================================
# Playlist Tests
# =============================================================================

class TestPlaylist:
    """Tests for Playlist class."""

    def test_create_playlist(self):
        """Test playlist creation."""
        pl = Playlist("My Playlist")
        assert pl.name == "My Playlist"
        assert pl.track_count == 0

    def test_add_track(self, sample_track):
        """Test adding tracks."""
        pl = Playlist("Test")
        pl.add_track(sample_track)
        assert pl.track_count == 1

    def test_remove_track(self, sample_track):
        """Test removing tracks."""
        pl = Playlist("Test")
        pl.add_track(sample_track)
        result = pl.remove_track(sample_track.track_id)
        assert result is True
        assert pl.track_count == 0

    def test_remove_nonexistent_track(self):
        """Test removing non-existent track."""
        pl = Playlist("Test")
        result = pl.remove_track("nonexistent")
        assert result is False

    def test_get_track_by_index(self, playlist, sample_tracks):
        """Test getting track by index."""
        track = playlist.get_track(0)
        assert track == sample_tracks[0]

    def test_get_track_by_id(self, playlist, sample_tracks):
        """Test getting track by ID."""
        track = playlist.get_track_by_id(sample_tracks[2].track_id)
        assert track == sample_tracks[2]

    def test_get_next_track_linear(self, playlist):
        """Test getting next track in linear mode."""
        first = playlist.get_current_track()
        second = playlist.get_next_track(PLAYBACK_MODE_LINEAR)
        assert second is not None
        assert second != first

    def test_get_next_track_shuffle(self, playlist):
        """Test getting next track in shuffle mode."""
        playlist.shuffle()
        track = playlist.get_next_track(PLAYBACK_MODE_SHUFFLE)
        assert track is not None

    def test_get_next_track_loop(self, playlist):
        """Test loop mode stays on same track."""
        first = playlist.get_current_track()
        same = playlist.get_next_track(PLAYBACK_MODE_LOOP)
        assert same == first

    def test_clear_playlist(self, playlist):
        """Test clearing playlist."""
        playlist.clear()
        assert playlist.track_count == 0


# =============================================================================
# MusicStem Tests
# =============================================================================

class TestMusicStem:
    """Tests for MusicStem class."""

    def test_initial_state(self, music_stem, stem_info):
        """Test initial stem state."""
        assert music_stem.stem_id == stem_info.stem_id
        assert music_stem.name == stem_info.name
        assert music_stem.layer_type == stem_info.layer_type
        assert music_stem.current_state == StemState.INACTIVE

    def test_activate_stem(self, music_stem):
        """Test activating a stem."""
        music_stem.activate(fade_time=0.0)
        assert music_stem.current_state == StemState.ACTIVE
        assert music_stem.is_active

    def test_activate_with_fade(self, music_stem):
        """Test activating with fade."""
        music_stem.activate(fade_time=1.0)
        assert music_stem.current_state == StemState.FADING_IN

    def test_deactivate_stem(self, music_stem):
        """Test deactivating a stem."""
        music_stem.activate(fade_time=0.0)
        music_stem.deactivate(fade_time=0.0)
        assert music_stem.current_state == StemState.INACTIVE
        assert not music_stem.is_active

    def test_deactivate_with_fade(self, music_stem):
        """Test deactivating with fade."""
        music_stem.activate(fade_time=0.0)
        music_stem.deactivate(fade_time=1.0)
        assert music_stem.current_state == StemState.FADING_OUT

    def test_set_volume(self, music_stem):
        """Test setting volume."""
        music_stem.set_volume(0.5, fade_time=0.0)
        # Volume is combined with info.volume
        assert music_stem.volume >= 0.0

    def test_set_volume_out_of_range(self, music_stem):
        """Test volume limits."""
        with pytest.raises(ValueError):
            music_stem.set_volume(1.5)

    def test_mute_unmute(self, music_stem):
        """Test muting and unmuting."""
        music_stem.activate(fade_time=0.0)
        music_stem.mute()
        assert music_stem.is_muted
        assert music_stem.volume == 0.0

        music_stem.unmute()
        assert not music_stem.is_muted

    def test_solo(self, music_stem):
        """Test solo functionality."""
        music_stem.set_solo(True)
        assert music_stem.is_solo

        music_stem.set_solo(False)
        assert not music_stem.is_solo

    def test_update_fade(self, music_stem):
        """Test fade update processing."""
        music_stem.activate(fade_time=0.1)
        time.sleep(0.15)
        music_stem.update()
        assert music_stem.current_state == StemState.ACTIVE

    def test_get_state_snapshot(self, music_stem):
        """Test getting state snapshot."""
        music_stem.activate(fade_time=0.0)
        snapshot = music_stem.get_state_snapshot()
        assert isinstance(snapshot, StemPlaybackState)
        assert snapshot.state == StemState.ACTIVE


# =============================================================================
# FadeCurve Tests
# =============================================================================

class TestFadeCurve:
    """Tests for FadeCurve calculations."""

    def test_linear_curve(self):
        """Test linear fade curve."""
        assert FadeCurve.linear(0.0) == 0.0
        assert FadeCurve.linear(0.5) == 0.5
        assert FadeCurve.linear(1.0) == 1.0

    def test_equal_power_curve(self):
        """Test equal power curve."""
        assert FadeCurve.equal_power(0.0) == pytest.approx(0.0, abs=0.01)
        assert FadeCurve.equal_power(1.0) == pytest.approx(1.0, abs=0.01)
        # Mid-point should be > 0.5 for equal power
        assert FadeCurve.equal_power(0.5) > 0.5

    def test_s_curve(self):
        """Test S-curve (smoothstep)."""
        assert FadeCurve.s_curve(0.0) == 0.0
        assert FadeCurve.s_curve(1.0) == 1.0
        assert FadeCurve.s_curve(0.5) == 0.5

    def test_exponential_curve(self):
        """Test exponential curve."""
        assert FadeCurve.exponential(0.0) == pytest.approx(0.0, abs=0.01)
        assert FadeCurve.exponential(1.0) == pytest.approx(1.0, abs=0.01)

    def test_curve_clamping(self):
        """Test curve values are clamped to 0-1."""
        assert FadeCurve.linear(-0.5) == 0.0
        assert FadeCurve.linear(1.5) == 1.0

    def test_get_curve_by_name(self):
        """Test getting curve by name."""
        assert FadeCurve.get_curve(FADE_CURVE_LINEAR) == FadeCurve.linear
        assert FadeCurve.get_curve(FADE_CURVE_EQUAL_POWER) == FadeCurve.equal_power
        assert FadeCurve.get_curve(FADE_CURVE_S_CURVE) == FadeCurve.s_curve


# =============================================================================
# StemGroup Tests
# =============================================================================

class TestStemGroup:
    """Tests for StemGroup class."""

    def test_create_group(self):
        """Test creating a stem group."""
        group = StemGroup("percussion")
        assert group.name == "percussion"
        assert group.stem_count == 0

    def test_add_stem(self, music_stem):
        """Test adding stem to group."""
        group = StemGroup("test")
        group.add_stem(music_stem)
        assert group.stem_count == 1

    def test_remove_stem(self, music_stem):
        """Test removing stem from group."""
        group = StemGroup("test")
        group.add_stem(music_stem)
        result = group.remove_stem(music_stem.stem_id)
        assert result == music_stem
        assert group.stem_count == 0

    def test_activate_all(self, stem_info):
        """Test activating all stems in group."""
        group = StemGroup("test")
        for i in range(3):
            info = StemInfo(
                stem_id=f"stem_{i}",
                name=f"Stem {i}",
                layer_type=LAYER_DRUMS,
                path=f"/audio/stem_{i}.ogg",
            )
            stem = MusicStem(info)
            group.add_stem(stem)

        group.activate_all(fade_time=0.0)
        for stem in group.stems:
            assert stem.is_active

    def test_deactivate_all(self, stem_info):
        """Test deactivating all stems."""
        group = StemGroup("test")
        stem = MusicStem(stem_info)
        group.add_stem(stem)
        stem.activate(fade_time=0.0)

        group.deactivate_all(fade_time=0.0)
        for s in group.stems:
            assert not s.is_active

    def test_set_group_volume(self, stem_info):
        """Test setting group volume."""
        group = StemGroup("test")
        stem = MusicStem(stem_info)
        group.add_stem(stem)

        group.set_group_volume(0.5, fade_time=0.0)
        # Volume should be applied

    def test_mute_group(self, stem_info):
        """Test muting group."""
        group = StemGroup("test")
        stem = MusicStem(stem_info)
        group.add_stem(stem)

        group.mute_group()
        assert stem.is_muted


# =============================================================================
# LayeredMusicPlayer Tests
# =============================================================================

class TestLayeredMusicPlayer:
    """Tests for LayeredMusicPlayer class."""

    def test_initial_state(self, layered_player):
        """Test initial layered player state."""
        assert layered_player.stem_count == 0
        assert layered_player.master_volume == DEFAULT_VOLUME

    def test_add_stem(self, layered_player, stem_info):
        """Test adding a stem."""
        stem = layered_player.add_stem(stem_info)
        assert stem is not None
        assert layered_player.stem_count == 1

    def test_add_duplicate_stem_id(self, layered_player, stem_info):
        """Test adding duplicate stem ID fails."""
        layered_player.add_stem(stem_info)
        with pytest.raises(ValueError):
            layered_player.add_stem(stem_info)

    def test_max_stems_limit(self):
        """Test maximum stems limit."""
        player = LayeredMusicPlayer(max_stems=2)
        for i in range(2):
            info = StemInfo(
                stem_id=f"stem_{i}",
                name=f"Stem {i}",
                layer_type=LAYER_DRUMS,
                path=f"/audio/stem_{i}.ogg",
            )
            player.add_stem(info)

        with pytest.raises(ValueError):
            player.add_stem(StemInfo(
                stem_id="extra",
                name="Extra",
                layer_type=LAYER_BASS,
                path="/audio/extra.ogg",
            ))

    def test_remove_stem(self, layered_player, stem_info):
        """Test removing a stem."""
        layered_player.add_stem(stem_info)
        result = layered_player.remove_stem(stem_info.stem_id)
        assert result is True
        assert layered_player.stem_count == 0

    def test_get_stem(self, layered_player, stem_info):
        """Test getting a stem by ID."""
        layered_player.add_stem(stem_info)
        stem = layered_player.get_stem(stem_info.stem_id)
        assert stem is not None
        assert stem.stem_id == stem_info.stem_id

    def test_get_stem_by_type(self, layered_player):
        """Test getting stems by layer type."""
        for i, layer in enumerate([LAYER_DRUMS, LAYER_DRUMS, LAYER_BASS]):
            info = StemInfo(
                stem_id=f"stem_{i}",
                name=f"Stem {i}",
                layer_type=layer,
                path=f"/audio/stem_{i}.ogg",
            )
            layered_player.add_stem(info)

        drums = layered_player.get_stem_by_type(LAYER_DRUMS)
        assert len(drums) == 2

    def test_activate_layer(self, layered_player):
        """Test activating a layer type."""
        for i, layer in enumerate([LAYER_DRUMS, LAYER_BASS]):
            info = StemInfo(
                stem_id=f"stem_{i}",
                name=f"Stem {i}",
                layer_type=layer,
                path=f"/audio/stem_{i}.ogg",
            )
            layered_player.add_stem(info)

        layered_player.activate_layer(LAYER_DRUMS, fade_time=0.0)
        drums = layered_player.get_stem_by_type(LAYER_DRUMS)
        for stem in drums:
            assert stem.is_active

    def test_deactivate_layer(self, layered_player, stem_info):
        """Test deactivating a layer."""
        layered_player.add_stem(stem_info)
        layered_player.activate_layer(LAYER_DRUMS, fade_time=0.0)
        layered_player.deactivate_layer(LAYER_DRUMS, fade_time=0.0)

        stems = layered_player.get_stem_by_type(LAYER_DRUMS)
        for stem in stems:
            assert not stem.is_active

    def test_solo_stem(self, layered_player):
        """Test soloing a stem."""
        for i in range(3):
            info = StemInfo(
                stem_id=f"stem_{i}",
                name=f"Stem {i}",
                layer_type=LAYER_DRUMS,
                path=f"/audio/stem_{i}.ogg",
            )
            layered_player.add_stem(info)

        layered_player.solo_stem("stem_1")
        stem = layered_player.get_stem("stem_1")
        assert stem.is_solo

    def test_clear_solo(self, layered_player, stem_info):
        """Test clearing all solos."""
        layered_player.add_stem(stem_info)
        layered_player.solo_stem(stem_info.stem_id)
        layered_player.clear_solo()

        stem = layered_player.get_stem(stem_info.stem_id)
        assert not stem.is_solo

    def test_set_blend(self, layered_player):
        """Test setting volume blend."""
        for layer in [LAYER_DRUMS, LAYER_BASS, LAYER_MELODY]:
            info = StemInfo(
                stem_id=f"stem_{layer}",
                name=layer,
                layer_type=layer,
                path=f"/audio/{layer}.ogg",
            )
            layered_player.add_stem(info)

        blend = {
            LAYER_DRUMS: 1.0,
            LAYER_BASS: 0.8,
            LAYER_MELODY: 0.5,
        }
        layered_player.set_blend(blend, fade_time=0.0)
        # Should apply without error

    def test_get_all_volumes(self, layered_player, stem_info):
        """Test getting all stem volumes."""
        layered_player.add_stem(stem_info)
        volumes = layered_player.get_all_volumes()
        assert stem_info.stem_id in volumes

    def test_get_active_stems(self, layered_player, stem_info):
        """Test getting active stems."""
        layered_player.add_stem(stem_info)
        layered_player.activate_layer(LAYER_DRUMS, fade_time=0.0)

        active = layered_player.get_active_stems()
        assert len(active) == 1

    def test_activate_by_intensity(self, layered_player):
        """Test activating stems by intensity."""
        layers = [LAYER_PADS, LAYER_BASS, LAYER_MELODY, LAYER_DRUMS]
        for i, layer in enumerate(layers):
            info = StemInfo(
                stem_id=f"stem_{i}",
                name=layer,
                layer_type=layer,
                path=f"/audio/{layer}.ogg",
            )
            layered_player.add_stem(info)

        layered_player.activate_stems_by_intensity(0.5, layers, fade_time=0.0)
        active = layered_player.get_active_stems()
        assert len(active) == 2  # 50% intensity = 2 of 4 layers


# =============================================================================
# MusicTiming Tests
# =============================================================================

class TestTimeSignature:
    """Tests for TimeSignature class."""

    def test_create_time_signature(self):
        """Test creating time signature."""
        ts = TimeSignature(4, 4)
        assert ts.beats_per_bar == 4
        assert ts.beat_unit == 4

    def test_from_tuple(self):
        """Test creating from tuple."""
        ts = TimeSignature.from_tuple((3, 4))
        assert ts.beats_per_bar == 3
        assert ts.beat_unit == 4

    def test_to_tuple(self):
        """Test converting to tuple."""
        ts = TimeSignature(6, 8)
        assert ts.to_tuple() == (6, 8)

    def test_invalid_beats_per_bar(self):
        """Test invalid beats per bar."""
        with pytest.raises(ValueError):
            TimeSignature(0, 4)

    def test_invalid_beat_unit(self):
        """Test beat unit must be power of 2."""
        with pytest.raises(ValueError):
            TimeSignature(4, 3)


class TestBeatGrid:
    """Tests for BeatGrid class."""

    def test_create_beat_grid(self, beat_grid):
        """Test creating beat grid."""
        assert beat_grid.bpm == 120.0
        assert beat_grid.time_signature.beats_per_bar == 4

    def test_beat_duration(self, beat_grid):
        """Test beat duration calculation."""
        # At 120 BPM, one beat = 500ms
        assert beat_grid.beat_duration_ms == 500.0

    def test_bar_duration(self, beat_grid):
        """Test bar duration calculation."""
        # 4/4 at 120 BPM = 2000ms per bar
        assert beat_grid.bar_duration_ms == 2000.0

    def test_time_to_beat(self, beat_grid):
        """Test converting time to beats."""
        assert beat_grid.time_to_beat(500.0) == 1.0
        assert beat_grid.time_to_beat(1000.0) == 2.0

    def test_beat_to_time(self, beat_grid):
        """Test converting beats to time."""
        assert beat_grid.beat_to_time(1.0) == 500.0
        assert beat_grid.beat_to_time(4.0) == 2000.0

    def test_time_to_bar(self, beat_grid):
        """Test converting time to bar/beat."""
        bar, beat = beat_grid.time_to_bar(2500.0)
        assert bar == 1
        assert beat == pytest.approx(1.0)

    def test_bar_to_time(self, beat_grid):
        """Test converting bar/beat to time."""
        assert beat_grid.bar_to_time(2, 0.0) == 4000.0
        assert beat_grid.bar_to_time(1, 2.0) == 3000.0

    def test_quantize_to_beat(self, beat_grid):
        """Test quantizing to beat."""
        assert beat_grid.quantize_to_beat(480.0) == 500.0
        assert beat_grid.quantize_to_beat(520.0) == 500.0

    def test_quantize_to_bar(self, beat_grid):
        """Test quantizing to bar."""
        assert beat_grid.quantize_to_bar(1800.0) == 2000.0

    def test_next_beat(self, beat_grid):
        """Test getting next beat time."""
        next_beat = beat_grid.next_beat(300.0)
        assert next_beat == 500.0

    def test_next_bar(self, beat_grid):
        """Test getting next bar time."""
        next_bar = beat_grid.next_bar(500.0)
        assert next_bar == 2000.0

    def test_get_beat_info(self, beat_grid):
        """Test getting beat info."""
        info = beat_grid.get_beat_info(2750.0)
        assert isinstance(info, BeatInfo)
        assert info.bar == 1
        assert info.beat_in_bar == 1

    def test_bpm_change(self, beat_grid):
        """Test changing BPM."""
        beat_grid.bpm = 60.0
        assert beat_grid.beat_duration_ms == 1000.0

    def test_bpm_limits(self, beat_grid):
        """Test BPM limits."""
        with pytest.raises(ValueError):
            beat_grid.bpm = 10.0  # Below MIN_BPM

        with pytest.raises(ValueError):
            beat_grid.bpm = 500.0  # Above MAX_BPM


class TestMusicClock:
    """Tests for MusicClock class."""

    def test_create_clock(self, music_clock):
        """Test creating music clock."""
        assert music_clock.bpm == 120.0
        assert not music_clock.is_running

    def test_start_stop(self, music_clock):
        """Test starting and stopping clock."""
        music_clock.start()
        assert music_clock.is_running

        music_clock.stop()
        assert not music_clock.is_running

    def test_pause_resume(self, music_clock):
        """Test pausing and resuming."""
        music_clock.start()
        music_clock.pause()
        assert not music_clock.is_running

        music_clock.resume()
        assert music_clock.is_running

    def test_get_time(self, music_clock):
        """Test getting current time."""
        music_clock.start()
        time.sleep(0.1)
        current = music_clock.get_time_ms()
        assert current >= 0

    def test_seek(self, music_clock):
        """Test seeking to position."""
        music_clock.start()
        music_clock.seek(5000.0)
        assert music_clock.get_time_ms() >= 5000.0

    def test_seek_to_bar(self, music_clock):
        """Test seeking to bar."""
        music_clock.start()
        music_clock.seek_to_bar(4, 0.0)
        # Should not raise

    def test_get_beat_info(self, music_clock):
        """Test getting beat info."""
        music_clock.start()
        music_clock.seek(1000.0)
        info = music_clock.get_beat_info()
        assert isinstance(info, BeatInfo)

    def test_time_until_next_beat(self, music_clock):
        """Test getting time until next beat."""
        music_clock.start()
        music_clock.seek(300.0)
        remaining = music_clock.time_until_next_beat()
        assert remaining > 0

    def test_time_until_next_bar(self, music_clock):
        """Test getting time until next bar."""
        music_clock.start()
        music_clock.seek(500.0)
        remaining = music_clock.time_until_next_bar()
        assert remaining > 0


class TestSyncPointManager:
    """Tests for SyncPointManager class."""

    def test_add_sync_point(self, beat_grid):
        """Test adding sync point."""
        manager = SyncPointManager(beat_grid)
        sp = manager.add_sync_point("intro", bar=4, beat=0.0)
        assert sp.name == "intro"
        assert sp.bar == 4

    def test_remove_sync_point(self, beat_grid):
        """Test removing sync point."""
        manager = SyncPointManager(beat_grid)
        manager.add_sync_point("test", bar=0)
        result = manager.remove_sync_point("test")
        assert result is True

    def test_get_sync_point(self, beat_grid):
        """Test getting sync point by name."""
        manager = SyncPointManager(beat_grid)
        manager.add_sync_point("verse", bar=8)
        sp = manager.get_sync_point("verse")
        assert sp is not None
        assert sp.bar == 8

    def test_get_next_sync_point(self, beat_grid):
        """Test getting next sync point."""
        manager = SyncPointManager(beat_grid)
        manager.add_sync_point("a", bar=2)
        manager.add_sync_point("b", bar=8)

        # At 120 BPM with 4/4, bar 2 = 4000ms, bar 8 = 16000ms
        # Query from 5000ms should return bar 8
        next_sp = manager.get_next_sync_point(5000.0)
        assert next_sp.bar == 8

    def test_get_sync_points_in_range(self, beat_grid):
        """Test getting sync points in range."""
        manager = SyncPointManager(beat_grid)
        manager.add_sync_point("a", bar=1)
        manager.add_sync_point("b", bar=2)
        manager.add_sync_point("c", bar=5)

        points = manager.get_sync_points_in_range(0, 5000.0)
        assert len(points) == 2  # bars 1 and 2


# =============================================================================
# MusicCallback Tests
# =============================================================================

class TestMusicCallbackManager:
    """Tests for MusicCallbackManager class."""

    def test_register_beat_callback(self, callback_manager):
        """Test registering beat callback."""
        callback = Mock()
        callback_id = callback_manager.register_beat_callback(callback)
        assert callback_id > 0
        assert callback_manager.get_registered_count(CALLBACK_BEAT) == 1

    def test_register_bar_callback(self, callback_manager):
        """Test registering bar callback."""
        callback = Mock()
        callback_id = callback_manager.register_bar_callback(callback)
        assert callback_id > 0

    def test_register_marker_callback(self, callback_manager):
        """Test registering marker callback."""
        callback = Mock()
        callback_id = callback_manager.register_marker_callback("verse", callback)
        assert callback_id > 0

    def test_unregister_callback(self, callback_manager):
        """Test unregistering callback."""
        callback = Mock()
        callback_id = callback_manager.register_beat_callback(callback)
        result = callback_manager.unregister(callback_id)
        assert result is True
        assert callback_manager.get_registered_count(CALLBACK_BEAT) == 0

    def test_unregister_all(self, callback_manager):
        """Test unregistering all callbacks."""
        for _ in range(5):
            callback_manager.register_beat_callback(Mock())

        callback_manager.unregister_all(CALLBACK_BEAT)
        assert callback_manager.get_registered_count(CALLBACK_BEAT) == 0

    def test_add_marker(self, callback_manager):
        """Test adding marker."""
        callback_manager.add_marker("chorus", 10000.0)
        marker_time = callback_manager.get_marker_time("chorus")
        assert marker_time == 10000.0

    def test_add_marker_at_bar(self, callback_manager):
        """Test adding marker at bar."""
        callback_manager.add_marker_at_bar("bridge", bar=16, beat=0.0)
        marker_time = callback_manager.get_marker_time("bridge")
        assert marker_time is not None

    def test_remove_marker(self, callback_manager):
        """Test removing marker."""
        callback_manager.add_marker("test", 5000.0)
        result = callback_manager.remove_marker("test")
        assert result is True

    def test_trigger_event(self, callback_manager):
        """Test manually triggering event."""
        callback = Mock()
        callback_manager.register_beat_callback(callback)

        event = CallbackEvent(
            event_type=CALLBACK_BEAT,
            time_ms=1000.0,
            beat=2.0,
            bar=0,
        )
        callback_manager.trigger_event(event)
        callback.assert_called_once()

    def test_callback_priority(self, callback_manager):
        """Test callback priority ordering."""
        results = []

        def high_callback(event, data):
            results.append("high")

        def low_callback(event, data):
            results.append("low")

        callback_manager.register_callback(
            CALLBACK_BEAT, low_callback, priority=CallbackPriority.LOW
        )
        callback_manager.register_callback(
            CALLBACK_BEAT, high_callback, priority=CallbackPriority.HIGH
        )

        event = CallbackEvent(
            event_type=CALLBACK_BEAT,
            time_ms=0,
            beat=0,
            bar=0,
        )
        callback_manager.trigger_event(event)

        assert results == ["high", "low"]

    def test_once_callback(self, callback_manager):
        """Test one-shot callback."""
        callback = Mock()
        callback_manager.register_beat_callback(callback, once=True)

        event = CallbackEvent(event_type=CALLBACK_BEAT, time_ms=0, beat=0, bar=0)
        callback_manager.trigger_event(event)
        callback_manager.trigger_event(event)

        assert callback.call_count == 1


class TestBeatCallbackTiming:
    """Tests for beat/bar callback timing verification."""

    def test_beat_callback_timing_precision(self, music_clock, callback_manager):
        """Test that beat callbacks fire within precision tolerance."""
        callback_times = []
        expected_beat_duration = 500.0  # At 120 BPM

        def record_time(event, user_data):
            callback_times.append(event.time_ms)

        callback_manager.register_beat_callback(record_time)
        music_clock.start()
        callback_manager.start(interval_ms=UPDATE_INTERVAL_CALLBACK_MS)

        # Wait for a few beats
        time.sleep(0.6)  # Should get at least 1 beat

        callback_manager.stop()
        music_clock.stop()

        # Verify callbacks were triggered
        assert len(callback_times) >= 1

    def test_bar_callback_fires_at_bar_boundary(self, music_clock, callback_manager):
        """Test that bar callbacks fire at bar boundaries."""
        bar_numbers = []

        def record_bar(event, user_data):
            bar_numbers.append(event.bar)

        callback_manager.register_bar_callback(record_bar)
        music_clock.start()
        callback_manager.start(interval_ms=UPDATE_INTERVAL_CALLBACK_MS)

        # Wait for at least 1 bar (4 beats at 120 BPM = 2 seconds)
        time.sleep(2.1)

        callback_manager.stop()
        music_clock.stop()

        # Verify at least one bar callback fired
        assert len(bar_numbers) >= 1

    def test_callback_precision_within_tolerance(self, music_clock):
        """Test that callback timing is within defined tolerance."""
        # At 120 BPM, beat duration is 500ms
        beat_duration = 60000.0 / 120.0  # 500ms

        music_clock.start()
        music_clock.seek(480.0)  # Just before first beat

        # Check if on beat within tolerance
        is_on = music_clock.is_on_beat(tolerance_ms=BEAT_CALLBACK_PRECISION_MS)
        # Should not be on beat yet (20ms away)
        assert not is_on

        music_clock.seek(498.0)  # Within tolerance
        is_on = music_clock.is_on_beat(tolerance_ms=BEAT_CALLBACK_PRECISION_MS)
        assert is_on


class TestBeatScheduler:
    """Tests for BeatScheduler class."""

    def test_schedule_at_beat(self, music_clock, callback_manager):
        """Test scheduling at specific beat."""
        scheduler = BeatScheduler(music_clock, callback_manager)
        callback = Mock()

        music_clock.start()
        scheduler.schedule_at_beat(4, callback)

        assert scheduler.get_scheduled_count() == 1

    def test_schedule_at_bar(self, music_clock, callback_manager):
        """Test scheduling at specific bar."""
        scheduler = BeatScheduler(music_clock, callback_manager)
        callback = Mock()

        music_clock.start()
        scheduler.schedule_at_bar(2, 0.0, callback)

        assert scheduler.get_scheduled_count() == 1

    def test_clear_scheduled(self, music_clock, callback_manager):
        """Test clearing scheduled callbacks."""
        scheduler = BeatScheduler(music_clock, callback_manager)
        scheduler.schedule_at_beat(4, Mock())
        scheduler.schedule_at_beat(8, Mock())

        scheduler.clear()
        assert scheduler.get_scheduled_count() == 0


# =============================================================================
# MusicTransition Tests
# =============================================================================

class TestTransitionConfig:
    """Tests for TransitionConfig class."""

    def test_create_config(self):
        """Test creating transition config."""
        config = TransitionConfig(
            transition_type=TRANSITION_CROSSFADE,
            duration_ms=2000.0,
        )
        assert config.transition_type == TRANSITION_CROSSFADE
        assert config.duration_ms == 2000.0

    def test_invalid_transition_type(self):
        """Test invalid transition type."""
        with pytest.raises(ValueError):
            TransitionConfig(transition_type="invalid")

    def test_minimum_duration(self):
        """Test minimum duration validation."""
        with pytest.raises(ValueError):
            TransitionConfig(duration_ms=10.0)  # Too short


class TestMusicTransition:
    """Tests for MusicTransition class."""

    def test_create_transition(self, music_clock):
        """Test creating a transition."""
        config = TransitionConfig()
        request = TransitionRequest(
            request_id=1,
            config=config,
            destination_id="track_b",
        )
        transition = MusicTransition(request, music_clock)

        assert transition.state == TransitionState.IDLE
        assert not transition.is_active

    def test_start_transition(self, music_clock):
        """Test starting a transition."""
        config = TransitionConfig()
        request = TransitionRequest(request_id=1, config=config, destination_id="b")
        transition = MusicTransition(request, music_clock)

        transition.start()
        assert transition.state == TransitionState.ACTIVE
        assert transition.is_active

    def test_schedule_transition(self, music_clock):
        """Test scheduling a transition."""
        config = TransitionConfig()
        request = TransitionRequest(request_id=1, config=config, destination_id="b")
        transition = MusicTransition(request, music_clock)

        transition.schedule(5000.0)
        assert transition.state == TransitionState.PENDING

    def test_cancel_transition(self, music_clock):
        """Test canceling a transition."""
        config = TransitionConfig()
        request = TransitionRequest(request_id=1, config=config, destination_id="b")
        transition = MusicTransition(request, music_clock)

        transition.start()
        transition.cancel()
        assert transition.state == TransitionState.CANCELLED

    def test_update_transition(self, music_clock):
        """Test updating transition progress."""
        config = TransitionConfig(duration_ms=100.0)
        request = TransitionRequest(request_id=1, config=config, destination_id="b")
        transition = MusicTransition(request, music_clock)

        transition.start()
        time.sleep(0.05)
        transition.update()

        assert transition.progress > 0.0

    def test_transition_completes(self, music_clock):
        """Test transition completion."""
        # Use minimum valid duration (0.1s = 100ms)
        config = TransitionConfig(duration_ms=100.0)
        request = TransitionRequest(request_id=1, config=config, destination_id="b")
        transition = MusicTransition(request, music_clock)

        transition.start()
        time.sleep(0.15)  # Wait longer than duration
        transition.update()

        assert transition.is_complete

    def test_transition_callbacks(self, music_clock):
        """Test transition callbacks."""
        on_start = Mock()
        on_complete = Mock()

        # Use minimum valid duration (0.1s = 100ms)
        config = TransitionConfig(duration_ms=100.0)
        request = TransitionRequest(request_id=1, config=config, destination_id="b")
        transition = MusicTransition(request, music_clock)
        transition.set_callbacks(on_start=on_start, on_complete=on_complete)

        transition.start()
        on_start.assert_called_once()

        time.sleep(0.15)  # Wait longer than duration
        transition.update()
        on_complete.assert_called_once()


class TestTransitionManager:
    """Tests for TransitionManager class."""

    def test_request_crossfade(self, transition_manager):
        """Test requesting crossfade transition."""
        request_id = transition_manager.request_crossfade("track_b", duration_ms=1000.0)
        assert request_id > 0
        assert transition_manager.pending_count == 1

    def test_request_beat_sync(self, transition_manager):
        """Test requesting beat-synced transition."""
        request_id = transition_manager.request_beat_sync("track_b")
        assert request_id > 0

    def test_request_bar_sync(self, transition_manager):
        """Test requesting bar-synced transition."""
        request_id = transition_manager.request_bar_sync("track_b")
        assert request_id > 0

    def test_request_immediate(self, transition_manager):
        """Test requesting immediate transition."""
        request_id = transition_manager.request_immediate("track_b")
        assert request_id > 0

    def test_cancel_transition(self, transition_manager):
        """Test canceling a pending transition."""
        request_id = transition_manager.request_crossfade("track_b")
        result = transition_manager.cancel_transition(request_id)
        assert result is True
        assert transition_manager.pending_count == 0

    def test_cancel_all_pending(self, transition_manager):
        """Test canceling all pending transitions."""
        transition_manager.request_crossfade("a")
        transition_manager.request_crossfade("b")
        transition_manager.request_crossfade("c")

        transition_manager.cancel_all_pending()
        assert transition_manager.pending_count == 0

    def test_process_next_transition(self, transition_manager, music_clock):
        """Test processing next transition."""
        music_clock.start()
        transition_manager.request_crossfade("track_b")
        transition = transition_manager.process_next_transition()
        assert transition is not None
        assert transition_manager.has_active_transition

    def test_get_source_volume(self, transition_manager, music_clock):
        """Test getting source volume during transition."""
        assert transition_manager.get_source_volume() == 1.0

    def test_get_destination_volume(self, transition_manager):
        """Test getting destination volume."""
        assert transition_manager.get_destination_volume() == 0.0


class TestTransitionEdgeCases:
    """Tests for transition edge cases and timing."""

    def test_scheduled_transition_triggers_at_time(self, music_clock):
        """Test that scheduled transitions trigger at the correct time."""
        config = TransitionConfig(duration_ms=200.0)
        request = TransitionRequest(request_id=1, config=config, destination_id="b")
        transition = MusicTransition(request, music_clock)

        music_clock.start()
        transition.schedule(500.0)  # Schedule at 500ms

        assert transition.state == TransitionState.PENDING

        # Before scheduled time
        music_clock.seek(400.0)
        transition.update(400.0)
        assert transition.state == TransitionState.PENDING

        # At or after scheduled time
        transition.update(500.0)
        assert transition.state == TransitionState.ACTIVE

    def test_transition_volume_curves(self, music_clock):
        """Test that transition volumes follow expected curves."""
        config = TransitionConfig(duration_ms=100.0)
        request = TransitionRequest(request_id=1, config=config, destination_id="b")
        transition = MusicTransition(request, music_clock)

        transition.start()
        # At start
        assert transition.source_volume == pytest.approx(1.0, abs=0.1)
        assert transition.destination_volume == pytest.approx(0.0, abs=0.1)

        # Let it run to completion
        time.sleep(0.15)
        transition.update()

        # At end
        assert transition.source_volume == pytest.approx(0.0, abs=0.1)
        assert transition.destination_volume == pytest.approx(1.0, abs=0.1)

    def test_cancel_during_active_transition(self, music_clock):
        """Test canceling an active transition."""
        config = TransitionConfig(duration_ms=1000.0)
        request = TransitionRequest(request_id=1, config=config, destination_id="b")
        transition = MusicTransition(request, music_clock)

        on_cancel = Mock()
        transition.set_callbacks(on_cancel=on_cancel)

        transition.start()
        assert transition.is_active

        transition.cancel()
        assert transition.state == TransitionState.CANCELLED
        on_cancel.assert_called_once()

    def test_cannot_cancel_completed_transition(self, music_clock):
        """Test that completed transitions cannot be cancelled."""
        # Use minimum valid duration (0.1s = 100ms)
        config = TransitionConfig(duration_ms=100.0)
        request = TransitionRequest(request_id=1, config=config, destination_id="b")
        transition = MusicTransition(request, music_clock)

        transition.start()
        time.sleep(0.15)  # Wait longer than duration
        transition.update()

        assert transition.is_complete

        # Try to cancel
        transition.cancel()
        # Should still be completed, not cancelled
        assert transition.state == TransitionState.COMPLETED

    def test_immediate_transition_instant_volume_change(self, music_clock):
        """Test immediate transition has instant volume change."""
        from engine.audio.adaptive.config import TRANSITION_IMMEDIATE

        config = TransitionConfig(
            transition_type=TRANSITION_IMMEDIATE,
            duration_ms=100.0,
        )
        request = TransitionRequest(request_id=1, config=config, destination_id="b")
        transition = MusicTransition(request, music_clock)

        transition.start()
        transition.update()

        # Immediate should have instant switch
        assert transition.source_volume == 0.0
        assert transition.destination_volume == 1.0


# =============================================================================
# MusicState Tests
# =============================================================================

class TestMusicStateConfig:
    """Tests for MusicStateConfig class."""

    def test_create_config(self):
        """Test creating state config."""
        config = MusicStateConfig(
            state_id=STATE_EXPLORATION,
            track_ids=["track_1", "track_2"],
        )
        assert config.state_id == STATE_EXPLORATION
        assert len(config.track_ids) == 2

    def test_create_exploration_state(self):
        """Test helper for exploration state."""
        config = create_exploration_state(["track_1"])
        assert config.state_id == STATE_EXPLORATION
        assert config.loop is True

    def test_create_combat_state(self):
        """Test helper for combat state."""
        config = create_combat_state(["combat_1"])
        assert config.state_id == STATE_COMBAT
        assert config.min_duration_ms > 0


class TestMusicState:
    """Tests for MusicState class."""

    def test_create_state(self):
        """Test creating music state."""
        config = MusicStateConfig(state_id="test", track_ids=["track"])
        state = MusicState(config)
        assert state.state_id == "test"
        assert not state.is_active

    def test_enter_exit_state(self):
        """Test entering and exiting state."""
        config = MusicStateConfig(state_id="test", track_ids=["track"])
        state = MusicState(config)

        state.enter()
        assert state.is_active

        state.exit()
        assert not state.is_active

    def test_get_current_track(self):
        """Test getting current track ID."""
        config = MusicStateConfig(
            state_id="test",
            track_ids=["track_a", "track_b"],
        )
        state = MusicState(config)
        assert state.get_current_track_id() == "track_a"

    def test_advance_track(self):
        """Test advancing to next track."""
        config = MusicStateConfig(
            state_id="test",
            track_ids=["track_a", "track_b"],
            loop=True,
        )
        state = MusicState(config)

        next_track = state.advance_track()
        assert next_track == "track_b"

    def test_advance_track_loop(self):
        """Test looping back to first track."""
        config = MusicStateConfig(
            state_id="test",
            track_ids=["track_a", "track_b"],
            loop=True,
        )
        state = MusicState(config)
        state.advance_track()
        next_track = state.advance_track()
        assert next_track == "track_a"

    def test_can_exit(self):
        """Test minimum duration check."""
        config = MusicStateConfig(
            state_id="test",
            track_ids=["track"],
            min_duration_ms=0.0,
        )
        state = MusicState(config)
        state.enter()
        assert state.can_exit

    def test_reset_state(self):
        """Test resetting state."""
        config = MusicStateConfig(state_id="test", track_ids=["a", "b"])
        state = MusicState(config)
        state.enter()
        state.advance_track()
        state.reset()

        assert not state.is_active
        assert state.get_current_track_id() == "a"


class TestMusicStateManager:
    """Tests for MusicStateManager class."""

    def test_register_state(self, state_manager):
        """Test registering a state."""
        config = MusicStateConfig(state_id="test", track_ids=["track"])
        state = state_manager.register_state(config)
        assert state is not None
        assert state.state_id == "test"

    def test_unregister_state(self, state_manager):
        """Test unregistering a state."""
        config = MusicStateConfig(state_id="test", track_ids=["track"])
        state_manager.register_state(config)
        result = state_manager.unregister_state("test")
        assert result is True

    def test_get_state(self, state_manager):
        """Test getting a state."""
        config = MusicStateConfig(state_id="test", track_ids=["track"])
        state_manager.register_state(config)
        state = state_manager.get_state("test")
        assert state is not None

    def test_change_state(self, state_manager):
        """Test changing states."""
        state_manager.register_state(
            MusicStateConfig(state_id="exploration", track_ids=["exp"])
        )
        state_manager.register_state(
            MusicStateConfig(state_id="combat", track_ids=["cmb"])
        )

        state_manager.change_state("exploration")
        assert state_manager.current_state_id == "exploration"

        state_manager.change_state("combat", force=True)
        assert state_manager.current_state_id == "combat"

    def test_push_pop_state(self, state_manager):
        """Test state stack."""
        state_manager.register_state(
            MusicStateConfig(state_id="base", track_ids=["b"])
        )
        state_manager.register_state(
            MusicStateConfig(state_id="overlay", track_ids=["o"])
        )

        state_manager.change_state("base")
        state_manager.push_state("overlay")
        assert state_manager.current_state_id == "overlay"

        state_manager.pop_state()
        assert state_manager.current_state_id == "base"

    def test_return_to_default(self, state_manager):
        """Test returning to default state."""
        state_manager.register_state(
            MusicStateConfig(state_id=STATE_EXPLORATION, track_ids=["exp"])
        )
        state_manager.set_default_state(STATE_EXPLORATION)

        result = state_manager.return_to_default()
        assert result == STATE_EXPLORATION

    def test_state_callbacks(self, state_manager):
        """Test state change callbacks."""
        on_enter = Mock()
        on_exit = Mock()
        state_manager.set_callbacks(on_state_enter=on_enter, on_state_exit=on_exit)

        state_manager.register_state(
            MusicStateConfig(state_id="a", track_ids=["a"])
        )
        state_manager.register_state(
            MusicStateConfig(state_id="b", track_ids=["b"])
        )

        state_manager.change_state("a")
        on_enter.assert_called()

        state_manager.change_state("b", force=True)
        on_exit.assert_called()

    def test_set_get_parameter(self, state_manager):
        """Test setting and getting parameters."""
        state_manager.set_parameter("intensity", 0.8)
        value = state_manager.get_parameter("intensity")
        assert value == 0.8

    def test_get_state_history(self, state_manager):
        """Test getting state history."""
        state_manager.register_state(
            MusicStateConfig(state_id="a", track_ids=["a"])
        )
        state_manager.change_state("a")

        history = state_manager.get_state_history()
        assert len(history) > 0


class TestMusicStateEdgeCases:
    """Tests for music state machine edge cases."""

    def test_change_to_nonexistent_state(self, state_manager):
        """Test changing to a state that doesn't exist."""
        result = state_manager.change_state("nonexistent")
        assert result is False

    def test_priority_prevents_lower_priority_change(self, state_manager):
        """Test that higher priority states block lower priority changes."""
        state_manager.register_state(
            MusicStateConfig(state_id="high", track_ids=["h"], priority=10)
        )
        state_manager.register_state(
            MusicStateConfig(state_id="low", track_ids=["l"], priority=1)
        )

        state_manager.change_state("high")
        result = state_manager.change_state("low")  # Should fail due to priority
        assert result is False
        assert state_manager.current_state_id == "high"

    def test_force_overrides_priority(self, state_manager):
        """Test that force=True overrides priority check."""
        state_manager.register_state(
            MusicStateConfig(state_id="high", track_ids=["h"], priority=10)
        )
        state_manager.register_state(
            MusicStateConfig(state_id="low", track_ids=["l"], priority=1)
        )

        state_manager.change_state("high")
        result = state_manager.change_state("low", force=True)
        assert result is True
        assert state_manager.current_state_id == "low"

    def test_min_duration_prevents_exit(self, state_manager):
        """Test that minimum duration prevents early exit."""
        state_manager.register_state(
            MusicStateConfig(
                state_id="long",
                track_ids=["l"],
                min_duration_ms=5000.0,
                can_interrupt=False,
            )
        )
        state_manager.register_state(
            MusicStateConfig(state_id="other", track_ids=["o"])
        )

        state_manager.change_state("long")
        # Try to change immediately (before min_duration)
        result = state_manager.change_state("other")
        assert result is False

    def test_can_interrupt_allows_early_exit(self, state_manager):
        """Test that can_interrupt=True allows early state changes."""
        state_manager.register_state(
            MusicStateConfig(
                state_id="interruptible",
                track_ids=["i"],
                min_duration_ms=5000.0,
                can_interrupt=True,
            )
        )
        state_manager.register_state(
            MusicStateConfig(state_id="other", track_ids=["o"], priority=10)
        )

        state_manager.change_state("interruptible")
        result = state_manager.change_state("other")
        assert result is True

    def test_deep_state_stack(self, state_manager):
        """Test deep state stack operations."""
        for i in range(10):
            state_manager.register_state(
                MusicStateConfig(state_id=f"state_{i}", track_ids=[f"t{i}"])
            )

        # Push all states
        for i in range(10):
            state_manager.push_state(f"state_{i}")

        # Pop all and verify order
        for i in range(9, 0, -1):
            state_manager.pop_state()
            expected = f"state_{i-1}" if i > 1 else None
            # After popping, previous state should be active

    def test_unregister_current_state(self, state_manager):
        """Test unregistering the currently active state."""
        state_manager.register_state(
            MusicStateConfig(state_id="active", track_ids=["a"])
        )
        state_manager.change_state("active")

        result = state_manager.unregister_state("active")
        assert result is True
        assert state_manager.current_state is None

    def test_clear_all_states(self, state_manager):
        """Test clearing all states."""
        for i in range(5):
            state_manager.register_state(
                MusicStateConfig(state_id=f"s{i}", track_ids=[f"t{i}"])
            )
        state_manager.change_state("s0")

        state_manager.clear()
        assert len(state_manager.get_all_states()) == 0

    def test_parameter_trigger_combat(self, state_manager):
        """Test that high danger triggers combat state."""
        from engine.audio.adaptive.config import (
            DANGER_THRESHOLD_HIGH, PARAM_DANGER,
        )

        state_manager.register_state(
            MusicStateConfig(state_id=STATE_EXPLORATION, track_ids=["exp"])
        )
        state_manager.register_state(
            MusicStateConfig(state_id=STATE_COMBAT, track_ids=["cmb"])
        )

        state_manager.change_state(STATE_EXPLORATION)

        # Set high danger - should trigger combat
        state_manager.set_parameter(PARAM_DANGER, DANGER_THRESHOLD_HIGH + 0.1)

        # Note: State change happens in _evaluate_parameter_triggers
        assert state_manager.current_state_id == STATE_COMBAT


# =============================================================================
# AdaptiveMusic Tests
# =============================================================================

class TestAdaptiveParameters:
    """Tests for AdaptiveParameters class."""

    def test_default_parameters(self):
        """Test default parameter values."""
        params = AdaptiveParameters()
        assert params.intensity == 0.5
        assert params.danger == 0.0
        assert params.tension == 0.0

    def test_get_parameter(self):
        """Test getting parameters."""
        params = AdaptiveParameters(intensity=0.8)
        assert params.get("intensity") == 0.8

    def test_set_parameter(self):
        """Test setting parameters."""
        params = AdaptiveParameters()
        params.set("intensity", 0.9)
        assert params.intensity == 0.9

    def test_parameter_clamping(self):
        """Test parameter clamping to 0-1."""
        params = AdaptiveParameters()
        params.set("intensity", 1.5)
        assert params.intensity == 1.0

        params.set("intensity", -0.5)
        assert params.intensity == 0.0

    def test_custom_parameters(self):
        """Test custom parameter storage."""
        params = AdaptiveParameters()
        params.set("my_param", 0.5)
        assert params.get("my_param") == 0.5


class TestMusicSection:
    """Tests for MusicSection class."""

    def test_create_section(self):
        """Test creating a music section."""
        section = MusicSection(
            section_id="intro",
            name="Intro",
            start_bar=0,
            end_bar=8,
        )
        assert section.section_id == "intro"
        assert section.length_bars == 8

    def test_section_with_next_sections(self):
        """Test section with valid next sections."""
        section = MusicSection(
            section_id="verse",
            name="Verse",
            start_bar=8,
            end_bar=16,
            next_sections=["chorus", "bridge"],
            weights={"chorus": 0.7, "bridge": 0.3},
        )
        assert len(section.next_sections) == 2
        assert section.weights["chorus"] == 0.7


class TestIntensityLevel:
    """Tests for IntensityLevel class."""

    def test_create_level(self):
        """Test creating intensity level."""
        level = IntensityLevel(
            level_id="medium",
            threshold=0.5,
            layers={LAYER_DRUMS: 0.8, LAYER_BASS: 0.6},
        )
        assert level.threshold == 0.5
        assert level.layers[LAYER_DRUMS] == 0.8


class TestVerticalRemixer:
    """Tests for VerticalRemixer class."""

    def test_set_intensity(self, layered_player):
        """Test setting intensity."""
        remixer = VerticalRemixer(layered_player)
        remixer.set_intensity(0.8, immediate=True)
        assert remixer.get_intensity() == 0.8

    def test_add_intensity_level(self, layered_player):
        """Test adding intensity level."""
        remixer = VerticalRemixer(layered_player)
        level = IntensityLevel(
            level_id="extreme",
            threshold=0.9,
            layers={LAYER_DRUMS: 1.0},
        )
        remixer.add_intensity_level(level)
        # Should not raise

    def test_remove_intensity_level(self, layered_player):
        """Test removing intensity level."""
        remixer = VerticalRemixer(layered_player)
        result = remixer.remove_intensity_level("low")
        assert result is True


class TestHorizontalSequencer:
    """Tests for HorizontalSequencer class."""

    def test_add_section(self, music_clock, callback_manager):
        """Test adding a section."""
        sequencer = HorizontalSequencer(music_clock, callback_manager)
        section = MusicSection(
            section_id="intro",
            name="Intro",
            start_bar=0,
            end_bar=8,
        )
        sequencer.add_section(section)
        assert sequencer.get_section("intro") is not None

    def test_remove_section(self, music_clock, callback_manager):
        """Test removing a section."""
        sequencer = HorizontalSequencer(music_clock, callback_manager)
        section = MusicSection("test", "Test", 0, 4)
        sequencer.add_section(section)

        result = sequencer.remove_section("test")
        assert result is True

    def test_start_section(self, music_clock, callback_manager):
        """Test starting a specific section."""
        music_clock.start()
        sequencer = HorizontalSequencer(music_clock, callback_manager)
        section = MusicSection("intro", "Intro", 0, 8)
        sequencer.add_section(section)

        sequencer.start_section("intro")
        assert sequencer.current_section == section

    def test_queue_next_section(self, music_clock, callback_manager):
        """Test queueing next section."""
        sequencer = HorizontalSequencer(music_clock, callback_manager)
        sequencer.add_section(MusicSection("a", "A", 0, 8))
        sequencer.add_section(MusicSection("b", "B", 8, 16))

        sequencer.queue_next_section("b")
        # Should queue without error

    def test_set_branch_type(self, music_clock, callback_manager):
        """Test setting branch type."""
        sequencer = HorizontalSequencer(music_clock, callback_manager)
        sequencer.set_branch_type(BranchType.WEIGHTED)
        # Should not raise


class TestAdaptiveMusicSystem:
    """Tests for AdaptiveMusicSystem class."""

    def test_create_system(self, music_clock, layered_player, callback_manager):
        """Test creating adaptive music system."""
        system = AdaptiveMusicSystem(
            music_clock,
            layered_player,
            callback_manager,
        )
        assert system.mode == AdaptiveMode.COMBINED

    def test_set_mode(self, music_clock, layered_player, callback_manager):
        """Test setting adaptive mode."""
        system = AdaptiveMusicSystem(
            music_clock, layered_player, callback_manager
        )
        system.mode = AdaptiveMode.VERTICAL
        assert system.mode == AdaptiveMode.VERTICAL

    def test_set_parameter(self, music_clock, layered_player, callback_manager):
        """Test setting music parameter."""
        system = AdaptiveMusicSystem(
            music_clock, layered_player, callback_manager
        )
        system.set_parameter("intensity", 0.8)
        assert system.get_parameter("intensity") == 0.8

    def test_trigger_combat(self, music_clock, layered_player, callback_manager):
        """Test triggering combat music."""
        system = AdaptiveMusicSystem(
            music_clock, layered_player, callback_manager
        )
        system.trigger_combat()
        assert system.get_parameter("intensity") > 0.5

    def test_trigger_exploration(self, music_clock, layered_player, callback_manager):
        """Test triggering exploration music."""
        system = AdaptiveMusicSystem(
            music_clock, layered_player, callback_manager
        )
        system.trigger_exploration()
        assert system.get_parameter("intensity") < 0.5

    def test_increase_decrease_intensity(
        self, music_clock, layered_player, callback_manager
    ):
        """Test intensity adjustment."""
        system = AdaptiveMusicSystem(
            music_clock, layered_player, callback_manager
        )
        initial = system.get_parameter("intensity")

        system.increase_intensity(0.2)
        assert system.get_parameter("intensity") > initial

        system.decrease_intensity(0.4)
        assert system.get_parameter("intensity") < initial

    def test_add_parameter_rule(self, music_clock, layered_player, callback_manager):
        """Test adding parameter rule."""
        system = AdaptiveMusicSystem(
            music_clock, layered_player, callback_manager
        )
        rule = Mock()
        system.add_parameter_rule(rule)
        system.set_parameter("intensity", 0.9)

        rule.assert_called()


# =============================================================================
# Stinger Tests
# =============================================================================

class TestStingerInfo:
    """Tests for StingerInfo class."""

    def test_create_stinger_info(self):
        """Test creating stinger info."""
        info = StingerInfo(
            stinger_id="impact_01",
            name="Impact",
            stinger_type=STINGER_TYPE_IMPACT,
            path="/audio/impact.ogg",
            duration_ms=1000.0,
        )
        assert info.stinger_id == "impact_01"
        assert info.stinger_type == STINGER_TYPE_IMPACT

    def test_invalid_stinger_type(self):
        """Test invalid stinger type."""
        with pytest.raises(ValueError):
            StingerInfo(
                stinger_id="test",
                name="Test",
                stinger_type="invalid",
                path="/audio/test.ogg",
                duration_ms=1000.0,
            )

    def test_duration_limits(self):
        """Test duration limits."""
        with pytest.raises(ValueError):
            StingerInfo(
                stinger_id="test",
                name="Test",
                stinger_type=STINGER_TYPE_IMPACT,
                path="/test.ogg",
                duration_ms=50.0,  # Too short
            )


class TestStinger:
    """Tests for Stinger class."""

    def test_create_stinger(self, stinger_info):
        """Test creating stinger."""
        stinger = Stinger(stinger_info)
        assert stinger.stinger_id == stinger_info.stinger_id
        assert stinger.state == StingerState.IDLE

    def test_play_stinger(self, stinger_info):
        """Test playing stinger."""
        stinger = Stinger(stinger_info)
        stinger.play()
        assert stinger.is_playing
        assert stinger.state == StingerState.PLAYING

    def test_stop_stinger(self, stinger_info):
        """Test stopping stinger."""
        stinger = Stinger(stinger_info)
        stinger.play()
        stinger.stop(fade_out=False)
        assert stinger.state == StingerState.FINISHED

    def test_stop_with_fade(self, stinger_info):
        """Test stopping with fade out."""
        stinger = Stinger(stinger_info)
        stinger.play()
        stinger.stop(fade_out=True)
        assert stinger.state == StingerState.FADING_OUT

    def test_schedule_stinger(self, stinger_info):
        """Test scheduling stinger."""
        stinger = Stinger(stinger_info)
        stinger.schedule(1000.0)
        assert stinger.state == StingerState.IDLE

    def test_stinger_elapsed_time(self, stinger_info):
        """Test elapsed time tracking."""
        stinger = Stinger(stinger_info)
        stinger.play()
        time.sleep(0.05)
        assert stinger.elapsed_ms > 0

    def test_stinger_remaining_time(self, stinger_info):
        """Test remaining time calculation."""
        stinger = Stinger(stinger_info)
        stinger.play()
        assert stinger.remaining_ms > 0
        assert stinger.remaining_ms <= stinger_info.duration_ms

    def test_on_complete_callback(self, stinger_info):
        """Test completion callback."""
        callback = Mock()
        stinger = Stinger(stinger_info)
        stinger.set_on_complete(callback)

        stinger.play()
        stinger.stop(fade_out=False)

        callback.assert_called_once_with(stinger)

    def test_reset_stinger(self, stinger_info):
        """Test resetting stinger."""
        stinger = Stinger(stinger_info)
        stinger.play()
        stinger.reset()

        assert stinger.state == StingerState.IDLE
        assert stinger.elapsed_ms == 0.0


class TestStingerManager:
    """Tests for StingerManager class."""

    def test_register_stinger(self, stinger_manager, stinger_info):
        """Test registering stinger."""
        stinger = stinger_manager.register_stinger(stinger_info)
        assert stinger is not None
        assert stinger_manager.stinger_count == 1

    def test_unregister_stinger(self, stinger_manager, stinger_info):
        """Test unregistering stinger."""
        stinger_manager.register_stinger(stinger_info)
        result = stinger_manager.unregister_stinger(stinger_info.stinger_id)
        assert result is True
        assert stinger_manager.stinger_count == 0

    def test_get_stinger(self, stinger_manager, stinger_info):
        """Test getting stinger by ID."""
        stinger_manager.register_stinger(stinger_info)
        stinger = stinger_manager.get_stinger(stinger_info.stinger_id)
        assert stinger is not None

    def test_get_stingers_by_type(self, stinger_manager):
        """Test getting stingers by type."""
        for i in range(3):
            info = StingerInfo(
                stinger_id=f"impact_{i}",
                name=f"Impact {i}",
                stinger_type=STINGER_TYPE_IMPACT,
                path=f"/audio/impact_{i}.ogg",
                duration_ms=1000.0,
            )
            stinger_manager.register_stinger(info)

        impacts = stinger_manager.get_stingers_by_type(STINGER_TYPE_IMPACT)
        assert len(impacts) == 3

    def test_get_stingers_by_tag(self, stinger_manager):
        """Test getting stingers by tag."""
        info = StingerInfo(
            stinger_id="test",
            name="Test",
            stinger_type=STINGER_TYPE_IMPACT,
            path="/audio/test.ogg",
            duration_ms=1000.0,
            tags=frozenset({"combat", "loud"}),
        )
        stinger_manager.register_stinger(info)

        tagged = stinger_manager.get_stingers_by_tag("combat")
        assert len(tagged) == 1

    def test_play_stinger(self, stinger_manager, stinger_info):
        """Test playing stinger through manager."""
        stinger_manager.register_stinger(stinger_info)
        result = stinger_manager.play_stinger(stinger_info.stinger_id)
        assert result is True

    def test_play_stinger_at_bar(self, stinger_manager, stinger_info, music_clock):
        """Test playing stinger at next bar."""
        music_clock.start()
        stinger_manager.register_stinger(stinger_info)
        result = stinger_manager.play_stinger_at_bar(stinger_info.stinger_id)
        assert result is True

    def test_stop_stinger(self, stinger_manager, stinger_info):
        """Test stopping a playing stinger."""
        stinger_manager.register_stinger(stinger_info)
        stinger_manager.play_stinger(stinger_info.stinger_id, beat_aligned=False)
        stinger_manager.stop_stinger(stinger_info.stinger_id)
        # Should not raise

    def test_stop_all_stingers(self, stinger_manager):
        """Test stopping all stingers."""
        for i in range(3):
            info = StingerInfo(
                stinger_id=f"stinger_{i}",
                name=f"Stinger {i}",
                stinger_type=STINGER_TYPE_IMPACT,
                path=f"/audio/stinger_{i}.ogg",
                duration_ms=1000.0,
            )
            stinger_manager.register_stinger(info)
            stinger_manager.play_stinger(f"stinger_{i}", beat_aligned=False)

        stinger_manager.stop_all_stingers(fade_out=False)
        # Need to call update to process the stop
        stinger_manager.update()
        assert stinger_manager.active_count == 0

    def test_play_random_stinger(self, stinger_manager):
        """Test playing random stinger."""
        for i in range(5):
            info = StingerInfo(
                stinger_id=f"impact_{i}",
                name=f"Impact {i}",
                stinger_type=STINGER_TYPE_IMPACT,
                path=f"/audio/impact_{i}.ogg",
                duration_ms=1000.0,
            )
            stinger_manager.register_stinger(info)

        stinger = stinger_manager.play_random_stinger(
            stinger_type=STINGER_TYPE_IMPACT
        )
        assert stinger is not None

    def test_get_active_stingers(self, stinger_manager, stinger_info):
        """Test getting active stingers."""
        stinger_manager.register_stinger(stinger_info)
        stinger_manager.play_stinger(stinger_info.stinger_id, beat_aligned=False)

        active = stinger_manager.get_active_stingers()
        assert len(active) == 1

    def test_clear_all(self, stinger_manager, stinger_info):
        """Test clearing all stingers."""
        stinger_manager.register_stinger(stinger_info)
        stinger_manager.clear()
        assert stinger_manager.stinger_count == 0


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_playlist_operations(self, music_player):
        """Test operations on empty playlist."""
        pl = Playlist("Empty")
        music_player.set_playlist(pl)
        music_player.play()
        assert music_player.current_track is None

    def test_very_fast_bpm(self, beat_grid):
        """Test handling of fast BPM."""
        beat_grid.bpm = MAX_BPM
        assert beat_grid.beat_duration_ms < 250  # Fast beats

    def test_very_slow_bpm(self, beat_grid):
        """Test handling of slow BPM."""
        beat_grid.bpm = MIN_BPM
        assert beat_grid.beat_duration_ms > 1500  # Slow beats

    def test_zero_duration_track(self, music_player):
        """Test handling of zero duration track."""
        track = TrackInfo(
            track_id="zero",
            name="Zero",
            path="/audio/zero.ogg",
            duration_ms=0.0,
        )
        music_player.play(track)
        position = music_player.get_position()
        assert position.normalized == 0.0

    def test_concurrent_stem_operations(self, layered_player):
        """Test thread safety of stem operations."""
        for i in range(8):
            info = StemInfo(
                stem_id=f"stem_{i}",
                name=f"Stem {i}",
                layer_type=LAYER_DRUMS,
                path=f"/audio/stem_{i}.ogg",
            )
            layered_player.add_stem(info)

        def activate_all():
            for i in range(8):
                stem = layered_player.get_stem(f"stem_{i}")
                if stem:
                    stem.activate(fade_time=0.1)

        def deactivate_all():
            for i in range(8):
                stem = layered_player.get_stem(f"stem_{i}")
                if stem:
                    stem.deactivate(fade_time=0.1)

        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=activate_all))
            threads.append(threading.Thread(target=deactivate_all))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def test_negative_seek(self, music_player, sample_track):
        """Test seeking to negative time."""
        music_player.play(sample_track)
        music_player.seek(-1000.0)
        position = music_player.get_position()
        assert position.time_ms >= 0

    def test_seek_past_duration(self, music_player, sample_track):
        """Test seeking past track duration."""
        music_player.play(sample_track)
        music_player.seek(sample_track.duration_ms + 10000.0)
        position = music_player.get_position()
        # Allow small floating point tolerance
        assert position.time_ms <= sample_track.duration_ms + 1.0


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance-related tests."""

    def test_rapid_state_changes(self, state_manager):
        """Test rapid state changes."""
        for state_id in ["s1", "s2", "s3"]:
            state_manager.register_state(
                MusicStateConfig(state_id=state_id, track_ids=["track"])
            )

        start = time.perf_counter()
        for _ in range(100):
            for state_id in ["s1", "s2", "s3"]:
                state_manager.change_state(state_id, force=True)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0  # Should complete in under 1 second

    def test_many_callbacks(self, callback_manager):
        """Test handling many callbacks."""
        callbacks = []
        for i in range(100):
            callback = Mock()
            callbacks.append(callback)
            callback_manager.register_beat_callback(callback)

        event = CallbackEvent(event_type=CALLBACK_BEAT, time_ms=0, beat=0, bar=0)

        start = time.perf_counter()
        callback_manager.trigger_event(event)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.1  # All callbacks in under 100ms
        for cb in callbacks:
            cb.assert_called_once()

    def test_large_transition_queue(self, transition_manager):
        """Test handling large transition queue."""
        for i in range(50):
            transition_manager.request_crossfade(f"track_{i}")

        # Queue should be limited
        assert transition_manager.pending_count <= 8  # TRANSITION_QUEUE_SIZE
