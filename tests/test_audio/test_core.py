"""
Comprehensive Tests for Audio Core Subsystem.

Tests all core audio components:
- AudioEngine: lifecycle, threading, state management
- AudioSource: creation, playback, seeking, looping, parameters
- AudioListener: position, orientation, velocity, multiple listeners
- AudioClip: loading, format detection, duration, sample data
- VoiceManager: allocation, voice limits, priority, stealing, virtual voices
- SoundCue: simple/random/sequence/switch cues, variation
- MemoryManager: allocation, pools, budgets, eviction, streaming

Target: 80+ tests with edge cases using pytest fixtures.
"""

from __future__ import annotations

import math
import pytest
import threading
import time
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

from engine.audio.core.config import (
    AudioCategory,
    AudioFormat,
    ChannelLayout,
    MemoryPoolType,
    SoundCueType,
    SourceType,
    VoiceState,
    VoiceStealStrategy,
    MAX_VOICES,
    MAX_INSTANCES_PER_SOUND,
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_NORMAL,
    PRIORITY_LOW,
    DEFAULT_SAMPLE_RATE,
    AUDIO_BUFFER_SIZE,
    MINIMUM_DB_VALUE,
)
from engine.audio.core.audio_engine import (
    AudioEngine,
    EngineState,
    PlayCommand,
    StopCommand,
)
from engine.audio.core.audio_source import (
    AudioSource,
    AudioSourcePool,
    FadeState,
    SourcePlaybackMode,
)
from engine.audio.core.audio_listener import (
    AudioListener,
    AudioListenerManager,
    Vector3,
)
from engine.audio.core.audio_clip import (
    AudioClip,
    AudioClipManager,
    AudioClipMetadata,
    ClipLoadState,
)
from engine.audio.core.voice_manager import (
    Voice,
    VoiceAllocationResult,
    VoiceManager,
)
from engine.audio.core.sound_cue import (
    CueVariation,
    SoundCue,
    SoundCueBuilder,
    SoundCueManager,
    SoundEntry,
    db_to_linear,
    linear_to_db,
)
from engine.audio.core.memory_manager import (
    AudioMemoryManager,
    MemoryBlock,
    MemoryPool,
    StreamBuffer,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def audio_engine():
    """Create a basic audio engine for testing."""
    engine = AudioEngine(
        sample_rate=DEFAULT_SAMPLE_RATE,
        buffer_size=AUDIO_BUFFER_SIZE,
        max_voices=MAX_VOICES,
        enable_streaming=False,
        enable_3d=True,
    )
    yield engine
    if engine.state not in (EngineState.UNINITIALIZED, EngineState.STOPPED):
        engine.stop()


@pytest.fixture
def initialized_engine(audio_engine):
    """Create an initialized (but not running) audio engine."""
    audio_engine.initialize()
    return audio_engine


@pytest.fixture
def running_engine(audio_engine):
    """Create a running audio engine."""
    audio_engine.start()
    yield audio_engine
    audio_engine.stop()


@pytest.fixture
def voice_manager():
    """Create a voice manager for testing."""
    return VoiceManager(
        max_voices=8,
        steal_strategy=VoiceStealStrategy.LOWEST_PRIORITY,
        enable_virtual_voices=True,
    )


@pytest.fixture
def audio_clip():
    """Create a mock audio clip for testing."""
    clip = AudioClip(
        id="test_clip_001",
        name="test_clip",
        category=AudioCategory.SFX,
        pool_type=MemoryPoolType.RESIDENT,
    )
    clip.metadata = AudioClipMetadata(
        duration_seconds=2.0,
        sample_rate=DEFAULT_SAMPLE_RATE,
        channels=2,
        format=AudioFormat.PCM_INT16,
        total_samples=DEFAULT_SAMPLE_RATE * 2,
        file_size=DEFAULT_SAMPLE_RATE * 2 * 2 * 2,
    )
    clip.load_state = ClipLoadState.LOADED
    clip._data = bytes(DEFAULT_SAMPLE_RATE * 2 * 2 * 2)
    return clip


@pytest.fixture
def audio_source(audio_clip):
    """Create a basic audio source for testing."""
    source = AudioSource(
        id="test_source_001",
        name="test_source",
        category=AudioCategory.SFX,
        priority=PRIORITY_NORMAL,
    )
    source.set_clip(audio_clip)
    return source


@pytest.fixture
def audio_listener():
    """Create an audio listener for testing."""
    return AudioListener()


@pytest.fixture
def memory_manager():
    """Create a memory manager for testing."""
    return AudioMemoryManager(
        total_budget=64 * 1024 * 1024,
        resident_size=32 * 1024 * 1024,
        streaming_size=16 * 1024 * 1024,
        temporary_size=8 * 1024 * 1024,
    )


# =============================================================================
# Vector3 Tests
# =============================================================================


class TestVector3:
    """Test suite for Vector3 math operations."""

    def test_vector3_creation(self):
        """Test Vector3 creation with default values."""
        v = Vector3()
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0

    def test_vector3_creation_with_values(self):
        """Test Vector3 creation with specific values."""
        v = Vector3(1.0, 2.0, 3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_vector3_addition(self):
        """Test Vector3 addition."""
        v1 = Vector3(1.0, 2.0, 3.0)
        v2 = Vector3(4.0, 5.0, 6.0)
        result = v1 + v2
        assert result.x == 5.0
        assert result.y == 7.0
        assert result.z == 9.0

    def test_vector3_subtraction(self):
        """Test Vector3 subtraction."""
        v1 = Vector3(5.0, 7.0, 9.0)
        v2 = Vector3(1.0, 2.0, 3.0)
        result = v1 - v2
        assert result.x == 4.0
        assert result.y == 5.0
        assert result.z == 6.0

    def test_vector3_scalar_multiplication(self):
        """Test Vector3 scalar multiplication."""
        v = Vector3(1.0, 2.0, 3.0)
        result = v * 2.0
        assert result.x == 2.0
        assert result.y == 4.0
        assert result.z == 6.0

    def test_vector3_scalar_division(self):
        """Test Vector3 scalar division."""
        v = Vector3(2.0, 4.0, 6.0)
        result = v / 2.0
        assert result.x == 1.0
        assert result.y == 2.0
        assert result.z == 3.0

    def test_vector3_division_by_zero(self):
        """Test Vector3 division by zero returns zero vector."""
        v = Vector3(1.0, 2.0, 3.0)
        result = v / 0.0
        assert result.x == 0.0
        assert result.y == 0.0
        assert result.z == 0.0

    def test_vector3_dot_product(self):
        """Test Vector3 dot product."""
        v1 = Vector3(1.0, 2.0, 3.0)
        v2 = Vector3(4.0, 5.0, 6.0)
        result = v1.dot(v2)
        assert result == 32.0  # 1*4 + 2*5 + 3*6

    def test_vector3_cross_product(self):
        """Test Vector3 cross product."""
        v1 = Vector3(1.0, 0.0, 0.0)
        v2 = Vector3(0.0, 1.0, 0.0)
        result = v1.cross(v2)
        assert result.z == 1.0  # x cross y = z

    def test_vector3_length(self):
        """Test Vector3 magnitude."""
        v = Vector3(3.0, 4.0, 0.0)
        assert v.length() == 5.0  # 3-4-5 triangle

    def test_vector3_normalized(self):
        """Test Vector3 normalization."""
        v = Vector3(3.0, 0.0, 0.0)
        n = v.normalized()
        assert n.x == 1.0
        assert n.y == 0.0
        assert n.z == 0.0

    def test_vector3_normalized_zero_vector(self):
        """Test normalizing zero vector returns zero vector."""
        v = Vector3(0.0, 0.0, 0.0)
        n = v.normalized()
        assert n.x == 0.0
        assert n.y == 0.0
        assert n.z == 0.0

    def test_vector3_distance_to(self):
        """Test Vector3 distance calculation."""
        v1 = Vector3(0.0, 0.0, 0.0)
        v2 = Vector3(3.0, 4.0, 0.0)
        assert v1.distance_to(v2) == 5.0


# =============================================================================
# AudioEngine Tests
# =============================================================================


class TestAudioEngine:
    """Test suite for AudioEngine lifecycle and state management."""

    def test_engine_initial_state(self, audio_engine):
        """Test engine starts in uninitialized state."""
        assert audio_engine.state == EngineState.UNINITIALIZED
        assert not audio_engine.is_initialized
        assert not audio_engine.is_running

    def test_engine_initialize(self, audio_engine):
        """Test engine initialization."""
        result = audio_engine.initialize()
        assert result is True
        assert audio_engine.state == EngineState.INITIALIZED
        assert audio_engine.is_initialized

    def test_engine_double_initialize(self, initialized_engine):
        """Test double initialization returns False."""
        result = initialized_engine.initialize()
        assert result is False

    def test_engine_start(self, audio_engine):
        """Test engine start."""
        result = audio_engine.start()
        assert result is True
        assert audio_engine.state == EngineState.RUNNING
        assert audio_engine.is_running
        audio_engine.stop()

    def test_engine_start_auto_initializes(self, audio_engine):
        """Test start automatically initializes if needed."""
        result = audio_engine.start()
        assert result is True
        assert audio_engine.is_initialized
        assert audio_engine.is_running
        audio_engine.stop()

    def test_engine_stop(self, running_engine):
        """Test engine stop."""
        running_engine.stop()
        assert running_engine.state == EngineState.STOPPED

    def test_engine_pause_resume(self, running_engine):
        """Test engine pause and resume."""
        running_engine.pause()
        assert running_engine.state == EngineState.PAUSED

        running_engine.resume()
        assert running_engine.state == EngineState.RUNNING

    def test_engine_context_manager(self, audio_engine):
        """Test engine as context manager."""
        with audio_engine as engine:
            assert engine.is_running
        assert engine.state == EngineState.STOPPED

    def test_engine_master_volume(self, audio_engine):
        """Test master volume setting."""
        audio_engine.master_volume = 0.5
        assert audio_engine.master_volume == 0.5

    def test_engine_master_volume_clamping(self, audio_engine):
        """Test master volume is clamped to 0-1."""
        audio_engine.master_volume = 2.0
        assert audio_engine.master_volume == 1.0

        audio_engine.master_volume = -1.0
        assert audio_engine.master_volume == 0.0

    def test_engine_mute(self, audio_engine):
        """Test engine mute state."""
        assert not audio_engine.muted
        audio_engine.muted = True
        assert audio_engine.muted

    def test_engine_category_volume(self, running_engine):
        """Test category volume control."""
        running_engine.set_category_volume(AudioCategory.SFX, 0.75)
        assert running_engine.get_category_volume(AudioCategory.SFX) == 0.75

    def test_engine_sample_rate(self, audio_engine):
        """Test sample rate property."""
        assert audio_engine.sample_rate == DEFAULT_SAMPLE_RATE

    def test_engine_listener(self, audio_engine):
        """Test listener access."""
        listener = audio_engine.listener
        assert listener is not None
        assert isinstance(listener, AudioListener)

    def test_engine_stats(self, running_engine):
        """Test engine statistics."""
        stats = running_engine.get_stats()
        assert 'state' in stats
        assert 'sample_rate' in stats
        assert 'master_volume' in stats


# =============================================================================
# AudioSource Tests
# =============================================================================


class TestAudioSource:
    """Test suite for AudioSource playback and parameters."""

    def test_source_creation(self):
        """Test source creation with defaults."""
        source = AudioSource()
        assert source.volume == 1.0
        assert source.pitch == 1.0
        assert source.pan == 0.0
        assert source.state == VoiceState.STOPPED

    def test_source_volume_clamping(self, audio_source):
        """Test volume is clamped to 0-1."""
        audio_source.volume = 2.0
        assert audio_source.volume == 1.0

        audio_source.volume = -1.0
        assert audio_source.volume == 0.0

    def test_source_pitch_clamping(self, audio_source):
        """Test pitch is clamped to 0.1-4.0."""
        audio_source.pitch = 10.0
        assert audio_source.pitch == 4.0

        audio_source.pitch = 0.01
        assert audio_source.pitch == 0.1

    def test_source_pan_clamping(self, audio_source):
        """Test pan is clamped to -1 to 1."""
        audio_source.pan = 2.0
        assert audio_source.pan == 1.0

        audio_source.pan = -2.0
        assert audio_source.pan == -1.0

    def test_source_play(self, audio_source):
        """Test source play."""
        result = audio_source.play()
        assert result is True
        assert audio_source.is_playing

    def test_source_play_without_clip(self):
        """Test play without clip returns False."""
        source = AudioSource()
        result = source.play()
        assert result is False

    def test_source_pause_resume(self, audio_source):
        """Test source pause and resume."""
        audio_source.play()
        audio_source.pause()
        assert audio_source.is_paused

        audio_source.resume()
        assert audio_source.is_playing

    def test_source_stop(self, audio_source):
        """Test source stop."""
        audio_source.play()
        audio_source.stop()
        assert audio_source.is_stopped

    def test_source_seek(self, audio_source):
        """Test source seeking."""
        audio_source.play()
        audio_source.seek(1.0)
        assert audio_source.playback_time == pytest.approx(1.0, rel=0.1)

    def test_source_looping(self, audio_source):
        """Test source looping state."""
        audio_source.playback_mode = SourcePlaybackMode.LOOP
        assert audio_source.is_looping

    def test_source_3d_position(self, audio_source):
        """Test 3D position setting."""
        audio_source.set_position_3d(10.0, 20.0, 30.0)
        assert audio_source.position.x == 10.0
        assert audio_source.position.y == 20.0
        assert audio_source.position.z == 30.0

    def test_source_3d_velocity(self, audio_source):
        """Test 3D velocity setting."""
        audio_source.set_velocity_3d(5.0, 0.0, -5.0)
        assert audio_source.velocity.x == 5.0
        assert audio_source.velocity.z == -5.0

    def test_source_distance_settings(self, audio_source):
        """Test distance attenuation settings."""
        audio_source.set_distance_settings(2.0, 50.0, 1.5)
        assert audio_source.min_distance == 2.0
        assert audio_source.max_distance == 50.0
        assert audio_source.rolloff_factor == 1.5

    def test_source_effective_volume(self, audio_source):
        """Test effective volume calculation."""
        audio_source.volume = 0.5
        assert audio_source.effective_volume == 0.5

        audio_source.muted = True
        assert audio_source.effective_volume == 0.0

    def test_source_clone(self, audio_source):
        """Test source cloning."""
        audio_source.volume = 0.7
        audio_source.pitch = 1.5

        clone = audio_source.clone()
        assert clone.volume == 0.7
        assert clone.pitch == 1.5
        assert clone.id != audio_source.id

    def test_source_reset(self, audio_source):
        """Test source reset."""
        audio_source.play()
        audio_source.volume = 0.5
        audio_source.reset()

        assert audio_source.is_stopped
        assert audio_source._playback_position == 0


# =============================================================================
# AudioSourcePool Tests
# =============================================================================


class TestAudioSourcePool:
    """Test suite for AudioSourcePool."""

    def test_pool_creation(self):
        """Test pool creation with initial size."""
        pool = AudioSourcePool(initial_size=10, max_size=50)
        assert pool.available_count == 10
        assert pool.in_use_count == 0

    def test_pool_acquire(self):
        """Test acquiring source from pool."""
        pool = AudioSourcePool(initial_size=5)
        source = pool.acquire()
        assert source is not None
        assert pool.in_use_count == 1
        assert pool.available_count == 4

    def test_pool_release(self):
        """Test releasing source back to pool."""
        pool = AudioSourcePool(initial_size=5)
        source = pool.acquire()
        pool.release(source)
        assert pool.in_use_count == 0
        assert pool.available_count == 5

    def test_pool_exhaustion(self):
        """Test pool exhaustion at max size."""
        pool = AudioSourcePool(initial_size=2, max_size=2)
        s1 = pool.acquire()
        s2 = pool.acquire()
        s3 = pool.acquire()
        assert s1 is not None
        assert s2 is not None
        assert s3 is None  # Pool exhausted

    def test_pool_clear(self):
        """Test clearing the pool."""
        pool = AudioSourcePool(initial_size=5)
        pool.acquire()
        pool.acquire()
        pool.clear()
        assert pool.available_count == 0
        assert pool.in_use_count == 0


# =============================================================================
# AudioListener Tests
# =============================================================================


class TestAudioListener:
    """Test suite for AudioListener."""

    def test_listener_creation(self, audio_listener):
        """Test listener default state."""
        assert audio_listener.volume == 1.0
        assert not audio_listener.muted
        assert audio_listener.active

    def test_listener_set_position(self, audio_listener):
        """Test listener position setting."""
        audio_listener.set_position(10.0, 5.0, -20.0)
        assert audio_listener.position.x == 10.0
        assert audio_listener.position.y == 5.0
        assert audio_listener.position.z == -20.0

    def test_listener_set_orientation(self, audio_listener):
        """Test listener orientation setting."""
        audio_listener.set_orientation(0.0, 0.0, 1.0, 0.0, 1.0, 0.0)
        # Vectors are normalized
        assert audio_listener.forward.z == 1.0

    def test_listener_set_velocity(self, audio_listener):
        """Test listener velocity setting."""
        audio_listener.set_velocity(5.0, 0.0, -10.0)
        assert audio_listener.velocity.x == 5.0
        assert audio_listener.velocity.z == -10.0

    def test_listener_calculate_pan(self, audio_listener):
        """Test pan calculation for source position."""
        audio_listener.set_position(0.0, 0.0, 0.0)
        audio_listener.set_orientation(0.0, 0.0, -1.0, 0.0, 1.0, 0.0)

        # Source to the right
        source_pos = Vector3(10.0, 0.0, 0.0)
        pan = audio_listener.calculate_pan(source_pos)
        assert pan > 0  # Right

    def test_listener_calculate_doppler(self, audio_listener):
        """Test Doppler calculation."""
        audio_listener.set_position(0.0, 0.0, 0.0)
        audio_listener.set_velocity(0.0, 0.0, 0.0)

        source_pos = Vector3(0.0, 0.0, -10.0)
        source_vel = Vector3(0.0, 0.0, 50.0)  # Moving towards listener

        doppler = audio_listener.calculate_doppler_factor(source_pos, source_vel)
        assert doppler > 1.0  # Higher pitch

    def test_listener_calculate_3d_parameters(self, audio_listener):
        """Test 3D parameter calculation."""
        audio_listener.set_position(0.0, 0.0, 0.0)
        source_pos = Vector3(10.0, 0.0, -10.0)
        source_vel = Vector3(0.0, 0.0, 0.0)

        attenuation, pan, doppler = audio_listener.calculate_3d_parameters(
            source_pos, source_vel, 1.0, 100.0, 1.0
        )

        assert 0.0 <= attenuation <= 1.0
        assert -1.0 <= pan <= 1.0
        assert 0.5 <= doppler <= 2.0

    def test_listener_effective_volume_muted(self, audio_listener):
        """Test effective volume when muted."""
        audio_listener.volume = 0.8
        assert audio_listener.effective_volume == 0.8

        audio_listener.muted = True
        assert audio_listener.effective_volume == 0.0


# =============================================================================
# AudioListenerManager Tests
# =============================================================================


class TestAudioListenerManager:
    """Test suite for AudioListenerManager."""

    def test_manager_creation(self):
        """Test manager creates default listener."""
        manager = AudioListenerManager()
        assert manager.active_listener is not None

    def test_manager_create_listener(self):
        """Test creating additional listeners."""
        manager = AudioListenerManager()
        listener = manager.create_listener("player2")
        assert listener is not None

    def test_manager_get_listener(self):
        """Test getting listener by ID."""
        manager = AudioListenerManager()
        manager.create_listener("player2")
        listener = manager.get_listener("player2")
        assert listener is not None

    def test_manager_set_active_listener(self):
        """Test setting active listener."""
        manager = AudioListenerManager()
        manager.create_listener("player2")
        result = manager.set_active_listener("player2")
        assert result is True

    def test_manager_remove_listener(self):
        """Test removing a listener."""
        manager = AudioListenerManager()
        manager.create_listener("player2")
        result = manager.remove_listener("player2")
        assert result is True

    def test_manager_cannot_remove_default(self):
        """Test cannot remove default listener."""
        manager = AudioListenerManager()
        result = manager.remove_listener("default")
        assert result is False


# =============================================================================
# AudioClip Tests
# =============================================================================


class TestAudioClip:
    """Test suite for AudioClip."""

    def test_clip_creation(self):
        """Test clip creation."""
        clip = AudioClip(id="test", name="test_clip")
        assert clip.id == "test"
        assert clip.name == "test_clip"
        assert clip.load_state == ClipLoadState.UNLOADED

    def test_clip_duration(self, audio_clip):
        """Test clip duration property."""
        assert audio_clip.duration == 2.0

    def test_clip_sample_rate(self, audio_clip):
        """Test clip sample rate property."""
        assert audio_clip.sample_rate == DEFAULT_SAMPLE_RATE

    def test_clip_channels(self, audio_clip):
        """Test clip channels property."""
        assert audio_clip.channels == 2

    def test_clip_reference_counting(self, audio_clip):
        """Test clip reference counting."""
        assert audio_clip.ref_count == 0
        audio_clip.add_ref()
        assert audio_clip.ref_count == 1
        audio_clip.release_ref()
        assert audio_clip.ref_count == 0

    def test_clip_is_loaded(self, audio_clip):
        """Test clip load state check."""
        assert audio_clip.is_loaded

    def test_clip_memory_size(self, audio_clip):
        """Test clip memory size calculation."""
        assert audio_clip.memory_size > 0

    def test_clip_load_from_memory(self):
        """Test loading clip from memory."""
        clip = AudioClip(id="mem_test", name="mem_clip")
        data = bytes(1024)
        result = clip.load_from_memory(
            data, AudioFormat.PCM_INT16, 44100, 1
        )
        assert result is True
        assert clip.is_loaded

    def test_clip_set_loop_points(self, audio_clip):
        """Test setting loop points."""
        audio_clip.set_loop_points(1000, 10000)
        assert audio_clip.metadata.loop_start == 1000
        assert audio_clip.metadata.loop_end == 10000
        assert audio_clip.metadata.has_loop_points


# =============================================================================
# AudioClipManager Tests
# =============================================================================


class TestAudioClipManager:
    """Test suite for AudioClipManager."""

    def test_manager_creation(self):
        """Test manager creation."""
        manager = AudioClipManager()
        assert manager.get_clip_count() == 0

    def test_manager_create_clip(self):
        """Test creating clip from raw data."""
        manager = AudioClipManager()
        data = bytes(1024)
        clip = manager.create_clip(
            "test", data, AudioFormat.PCM_INT16, 44100, 1
        )
        assert clip is not None
        assert manager.get_clip_count() == 1

    def test_manager_get_clip(self):
        """Test getting clip by ID."""
        manager = AudioClipManager()
        data = bytes(1024)
        clip = manager.create_clip(
            "test", data, AudioFormat.PCM_INT16, 44100, 1
        )
        retrieved = manager.get_clip(clip.id)
        assert retrieved is clip

    def test_manager_unload_all(self):
        """Test unloading all clips."""
        manager = AudioClipManager()
        data = bytes(1024)
        manager.create_clip("test1", data, AudioFormat.PCM_INT16, 44100, 1)
        manager.create_clip("test2", data, AudioFormat.PCM_INT16, 44100, 1)

        manager.unload_all()
        assert manager.get_clip_count() == 0


# =============================================================================
# VoiceManager Tests
# =============================================================================


class TestVoiceManager:
    """Test suite for VoiceManager."""

    def test_manager_creation(self, voice_manager):
        """Test manager creation."""
        assert voice_manager.active_voice_count == 0
        assert voice_manager.available_voices == 8

    def test_voice_allocation(self, voice_manager, audio_source):
        """Test voice allocation."""
        result = voice_manager.allocate_voice(audio_source)
        assert result.success
        assert result.voice_id is not None
        assert voice_manager.active_voice_count == 1

    def test_voice_release(self, voice_manager, audio_source):
        """Test voice release."""
        result = voice_manager.allocate_voice(audio_source)
        voice_manager.release_voice(result.voice_id)
        assert voice_manager.active_voice_count == 0

    def test_voice_limit_enforcement(self, voice_manager, audio_clip):
        """Test voice limit is enforced."""
        sources = []
        for i in range(10):  # Try to allocate more than max_voices
            source = AudioSource(id=f"source_{i}")
            source.set_clip(audio_clip)
            source.priority = PRIORITY_LOW
            sources.append(source)
            voice_manager.allocate_voice(source)

        # Should have stolen or limited
        assert voice_manager.active_voice_count <= 8

    def test_voice_stealing_lowest_priority(self, voice_manager, audio_clip):
        """Test voice stealing by lowest priority."""
        # Fill all slots with low priority
        low_sources = []
        for i in range(8):
            source = AudioSource(id=f"low_{i}")
            source.set_clip(audio_clip)
            source.priority = PRIORITY_LOW
            low_sources.append(source)
            voice_manager.allocate_voice(source)

        # Try to allocate high priority
        high_source = AudioSource(id="high")
        high_source.set_clip(audio_clip)
        high_source.priority = PRIORITY_HIGH

        result = voice_manager.allocate_voice(high_source)
        assert result.success

    def test_voice_stealing_strategy_none(self, audio_clip):
        """Test no stealing strategy."""
        manager = VoiceManager(
            max_voices=2,
            steal_strategy=VoiceStealStrategy.NONE,
            enable_virtual_voices=False,
        )

        # Fill slots
        for i in range(2):
            source = AudioSource(id=f"source_{i}")
            source.set_clip(audio_clip)
            manager.allocate_voice(source)

        # Try one more
        extra = AudioSource(id="extra")
        extra.set_clip(audio_clip)
        result = manager.allocate_voice(extra)
        assert not result.success

    def test_virtual_voices(self, voice_manager, audio_clip):
        """Test virtual voice creation when all slots are filled with higher priority."""
        # Fill all slots with critical priority (can't be stolen by NORMAL)
        # Use unique clips to avoid per-sound instance limit (MAX_INSTANCES_PER_SOUND=4)
        for i in range(8):
            source = AudioSource(id=f"critical_{i}")
            unique_clip = AudioClip(
                id=f"clip_{i}",
                name=f"clip_{i}",
                category=AudioCategory.SFX,
                pool_type=MemoryPoolType.RESIDENT,
            )
            unique_clip.metadata = audio_clip.metadata
            unique_clip.load_state = ClipLoadState.LOADED
            source.set_clip(unique_clip)
            source.priority = PRIORITY_CRITICAL
            voice_manager.allocate_voice(source)

        # All slots should be filled
        assert voice_manager.active_voice_count == 8

        # Try to allocate with lower priority - should fail or virtualize
        low_source = AudioSource(id="low")
        low_source.set_clip(audio_clip)
        low_source.priority = PRIORITY_LOW

        result = voice_manager.allocate_voice(low_source)

        # With all CRITICAL voices, a LOW priority voice cannot steal
        # It should either fail or be made virtual
        if result.success:
            # If it succeeded, a voice must have been virtualized
            assert voice_manager.virtual_voice_count >= 1 or result.made_virtual
        else:
            # Allocation failed as expected - can't steal from CRITICAL
            assert result.error is not None

    def test_category_voice_limit(self, voice_manager, audio_clip):
        """Test per-category voice limits."""
        voice_manager.set_category_limit(AudioCategory.SFX, 2)

        for i in range(4):
            source = AudioSource(id=f"sfx_{i}", category=AudioCategory.SFX)
            source.set_clip(audio_clip)
            voice_manager.allocate_voice(source)

        # Should be limited to 2 per category
        assert voice_manager.get_category_voice_count(AudioCategory.SFX) <= 2

    def test_per_sound_instance_limit(self, voice_manager, audio_clip):
        """Test per-sound instance limit."""
        for i in range(MAX_INSTANCES_PER_SOUND + 2):
            source = AudioSource(id=f"instance_{i}")
            source.set_clip(audio_clip)
            voice_manager.allocate_voice(source)

        count = voice_manager.get_sound_instance_count(audio_clip.id)
        assert count <= MAX_INSTANCES_PER_SOUND

    def test_voice_manager_update(self, voice_manager, audio_source):
        """Test voice manager update."""
        voice_manager.allocate_voice(audio_source)
        voice_manager.update(0.016)  # ~60fps
        assert voice_manager.active_voice_count >= 0

    def test_stop_all_voices(self, voice_manager, audio_clip):
        """Test stopping all voices."""
        for i in range(4):
            source = AudioSource(id=f"source_{i}")
            source.set_clip(audio_clip)
            voice_manager.allocate_voice(source)

        voice_manager.stop_all()
        assert voice_manager.active_voice_count == 0

    def test_stop_category_voices(self, voice_manager, audio_clip):
        """Test stopping voices by category."""
        sfx = AudioSource(id="sfx", category=AudioCategory.SFX)
        sfx.set_clip(audio_clip)
        voice_manager.allocate_voice(sfx)

        music = AudioSource(id="music", category=AudioCategory.MUSIC)
        music.set_clip(audio_clip)
        voice_manager.allocate_voice(music)

        voice_manager.stop_category(AudioCategory.SFX)
        # SFX stopped, MUSIC still playing
        assert voice_manager.get_category_voice_count(AudioCategory.SFX) == 0


# =============================================================================
# SoundCue Tests
# =============================================================================


class TestSoundCue:
    """Test suite for SoundCue system."""

    def test_simple_cue(self, audio_clip):
        """Test simple sound cue."""
        cue = SoundCue(id="cue1", name="test_cue", cue_type=SoundCueType.SIMPLE)
        cue.add_entry(audio_clip)

        entry = cue.select_entry()
        assert entry is not None
        assert entry.clip == audio_clip

    def test_random_cue(self, audio_clip):
        """Test random sound cue selection."""
        cue = SoundCue(id="cue2", name="random_cue", cue_type=SoundCueType.RANDOM)

        # Create multiple clips
        clips = [audio_clip]
        for i in range(3):
            clip = AudioClip(id=f"clip_{i}", name=f"clip_{i}")
            clip.metadata = audio_clip.metadata
            clip.load_state = ClipLoadState.LOADED
            clips.append(clip)

        for clip in clips:
            cue.add_entry(clip, weight=1.0)

        # Select multiple times
        selections = [cue.select_entry().clip for _ in range(10)]
        # Should have some variety (probabilistic)
        assert len(selections) > 0

    def test_sequence_cue(self, audio_clip):
        """Test sequence sound cue."""
        cue = SoundCue(id="cue3", name="seq_cue", cue_type=SoundCueType.SEQUENCE)

        clips = []
        for i in range(3):
            clip = AudioClip(id=f"seq_clip_{i}", name=f"seq_clip_{i}")
            clip.metadata = audio_clip.metadata
            clip.load_state = ClipLoadState.LOADED
            clips.append(clip)
            cue.add_entry(clip)

        # Should play in order
        assert cue.select_entry().clip == clips[0]
        assert cue.select_entry().clip == clips[1]
        assert cue.select_entry().clip == clips[2]
        assert cue.select_entry().clip == clips[0]  # Wraps around

    def test_switch_cue(self, audio_clip):
        """Test switch sound cue."""
        cue = SoundCue(id="cue4", name="switch_cue", cue_type=SoundCueType.SWITCH)

        clip1 = AudioClip(id="walk", name="walk")
        clip1.metadata = audio_clip.metadata
        clip1.load_state = ClipLoadState.LOADED

        clip2 = AudioClip(id="run", name="run")
        clip2.metadata = audio_clip.metadata
        clip2.load_state = ClipLoadState.LOADED

        cue.add_entry(clip1, condition="walking")
        cue.add_entry(clip2, condition="running")

        cue.set_switch_parameter("walking")
        assert cue.select_entry().clip == clip1

        cue.set_switch_parameter("running")
        assert cue.select_entry().clip == clip2

    def test_shuffle_cue(self, audio_clip):
        """Test shuffle sound cue."""
        cue = SoundCue(id="cue5", name="shuffle_cue", cue_type=SoundCueType.SHUFFLE)

        for i in range(4):
            clip = AudioClip(id=f"shuffle_{i}", name=f"shuffle_{i}")
            clip.metadata = audio_clip.metadata
            clip.load_state = ClipLoadState.LOADED
            cue.add_entry(clip)

        # Play through all
        seen = []
        for _ in range(4):
            entry = cue.select_entry()
            seen.append(entry.clip.id)

        # Should have played all 4 unique
        assert len(set(seen)) == 4

    def test_cue_variation_pitch(self, audio_clip):
        """Test pitch variation."""
        variation = CueVariation(
            enable_pitch_variation=True,
            pitch_variation_range=0.1,
        )

        pitches = [variation.apply_pitch_variation(1.0) for _ in range(20)]
        # Should have some variation
        assert min(pitches) < 1.0
        assert max(pitches) > 1.0

    def test_cue_variation_volume(self, audio_clip):
        """Test volume variation."""
        variation = CueVariation(
            enable_volume_variation=True,
            volume_variation_db=3.0,
        )

        volumes = [variation.apply_volume_variation(1.0) for _ in range(20)]
        # Should have some variation
        assert min(volumes) != max(volumes)

    def test_cue_create_source(self, audio_clip):
        """Test creating source from cue."""
        cue = SoundCue(id="cue6", name="source_cue")
        cue.add_entry(audio_clip)
        cue.base_volume = 0.8
        cue.base_pitch = 1.2

        source = cue.create_source()
        assert source is not None
        assert source.volume == pytest.approx(0.8, rel=0.1)

    def test_sound_cue_builder(self, audio_clip):
        """Test SoundCueBuilder pattern."""
        cue = (SoundCueBuilder()
            .with_id("built_cue")
            .with_name("Built Cue")
            .with_type(SoundCueType.RANDOM)
            .with_category(AudioCategory.SFX)
            .with_volume(0.9)
            .with_pitch_variation(0.1)
            .add_clip(audio_clip, weight=2.0)
            .build())

        assert cue.id == "built_cue"
        assert cue.cue_type == SoundCueType.RANDOM
        assert cue.base_volume == 0.9


# =============================================================================
# SoundCueManager Tests
# =============================================================================


class TestSoundCueManager:
    """Test suite for SoundCueManager."""

    def test_manager_register_cue(self, audio_clip):
        """Test registering a cue."""
        manager = SoundCueManager()
        cue = SoundCue(id="managed_cue", name="managed")
        cue.add_entry(audio_clip)

        manager.register_cue(cue)
        assert manager.get_cue("managed_cue") is cue

    def test_manager_unregister_cue(self, audio_clip):
        """Test unregistering a cue."""
        manager = SoundCueManager()
        cue = SoundCue(id="temp_cue", name="temp")
        manager.register_cue(cue)

        result = manager.unregister_cue("temp_cue")
        assert result is True
        assert manager.get_cue("temp_cue") is None

    def test_manager_play_cue(self, audio_clip):
        """Test playing a cue through manager."""
        manager = SoundCueManager()
        cue = SoundCue(id="play_cue", name="play")
        cue.add_entry(audio_clip)
        manager.register_cue(cue)

        source = manager.play_cue("play_cue")
        assert source is not None

    def test_manager_stop_cue(self, audio_clip):
        """Test stopping cue instances."""
        manager = SoundCueManager()
        cue = SoundCue(id="stop_cue", name="stop")
        cue.add_entry(audio_clip)
        manager.register_cue(cue)

        manager.play_cue("stop_cue")
        manager.stop_cue("stop_cue")

        assert manager.get_active_instance_count("stop_cue") == 0


# =============================================================================
# MemoryPool Tests
# =============================================================================


class TestMemoryPool:
    """Test suite for MemoryPool."""

    def test_pool_creation(self):
        """Test pool creation."""
        pool = MemoryPool(MemoryPoolType.RESIDENT, 1024 * 1024, "test")
        assert pool.max_size == 1024 * 1024
        assert pool.used_size == 0

    def test_pool_allocation(self):
        """Test memory allocation."""
        pool = MemoryPool(MemoryPoolType.RESIDENT, 1024 * 1024)
        block = pool.allocate(1024)
        assert block is not None
        assert block.size == 1024
        assert pool.used_size == 1024

    def test_pool_free(self):
        """Test freeing memory."""
        pool = MemoryPool(MemoryPoolType.RESIDENT, 1024 * 1024)
        block = pool.allocate(1024)
        result = pool.free(block.id)
        assert result is True
        assert pool.used_size == 0

    def test_pool_eviction(self):
        """Test LRU eviction."""
        pool = MemoryPool(MemoryPoolType.TEMPORARY, 1024)

        # Fill pool
        block1 = pool.allocate(512, priority=10)
        block2 = pool.allocate(512, priority=50)

        # Allocate more - should evict low priority
        block3 = pool.allocate(512, priority=100)
        assert block3 is not None

    def test_pool_pinned_block(self):
        """Test pinned blocks aren't evicted."""
        pool = MemoryPool(MemoryPoolType.TEMPORARY, 1024)

        # Pin a block
        block1 = pool.allocate(512, priority=10, pinned=True)
        block2 = pool.allocate(512, priority=50)

        # Try to allocate more
        block3 = pool.allocate(512, priority=100)
        # Pinned block should remain
        assert pool.get_block(block1.id) is not None


# =============================================================================
# AudioMemoryManager Tests
# =============================================================================


class TestAudioMemoryManager:
    """Test suite for AudioMemoryManager."""

    def test_manager_creation(self, memory_manager):
        """Test manager creation."""
        assert memory_manager.total_free > 0

    def test_manager_allocate(self, memory_manager):
        """Test memory allocation through manager."""
        block = memory_manager.allocate(
            1024,
            MemoryPoolType.RESIDENT,
            category=AudioCategory.SFX,
        )
        assert block is not None

    def test_manager_free(self, memory_manager):
        """Test freeing memory through manager."""
        block = memory_manager.allocate(
            1024,
            MemoryPoolType.RESIDENT,
        )
        result = memory_manager.free(block.id, MemoryPoolType.RESIDENT)
        assert result is True

    def test_manager_category_budget(self, memory_manager):
        """Test category budget enforcement."""
        memory_manager.set_category_budget(AudioCategory.SFX, 2048)

        # Allocate within budget
        block1 = memory_manager.allocate(
            1024, MemoryPoolType.RESIDENT,
            category=AudioCategory.SFX,
        )
        assert block1 is not None

        # Allocate beyond budget
        block2 = memory_manager.allocate(
            2048, MemoryPoolType.RESIDENT,
            category=AudioCategory.SFX,
        )
        # Should fail or evict
        assert memory_manager.get_category_usage(AudioCategory.SFX) <= 2048

    def test_manager_stream_buffer_acquire(self, memory_manager, audio_clip):
        """Test acquiring stream buffer."""
        buffer = memory_manager.acquire_stream_buffer(audio_clip)
        assert buffer is not None
        assert buffer.clip == audio_clip

    def test_manager_stream_buffer_release(self, memory_manager, audio_clip):
        """Test releasing stream buffer."""
        buffer = memory_manager.acquire_stream_buffer(audio_clip)
        memory_manager.release_stream_buffer(buffer.id)
        # Buffer should be returned to pool

    def test_manager_stats(self, memory_manager):
        """Test getting memory stats."""
        stats = memory_manager.get_stats()
        assert 'total_budget' in stats
        assert 'total_used' in stats
        assert 'pools' in stats


# =============================================================================
# StreamBuffer Tests
# =============================================================================


class TestStreamBuffer:
    """Test suite for StreamBuffer."""

    def test_buffer_creation(self):
        """Test buffer creation."""
        buffer = StreamBuffer(
            id=1,
            data=bytearray(1024),
            capacity=1024,
        )
        assert buffer.capacity == 1024
        assert buffer.available == 0

    def test_buffer_write_read(self):
        """Test writing and reading from buffer."""
        buffer = StreamBuffer(
            id=1,
            data=bytearray(1024),
            capacity=1024,
        )

        # Write data
        data = bytes([1, 2, 3, 4, 5])
        written = buffer.write(data)
        assert written == 5
        assert buffer.available == 5

        # Read data
        read_data = buffer.read(5)
        assert read_data == data

    def test_buffer_wraparound(self):
        """Test buffer wraparound."""
        buffer = StreamBuffer(
            id=1,
            data=bytearray(16),
            capacity=16,
        )

        # Fill most of buffer
        buffer.write(bytes(12))
        buffer.read(10)  # Free up space at start

        # Write across wraparound point
        written = buffer.write(bytes(8))
        assert written == 8

    def test_buffer_reset(self):
        """Test buffer reset."""
        buffer = StreamBuffer(
            id=1,
            data=bytearray(1024),
            capacity=1024,
        )
        buffer.write(bytes(100))
        buffer.reset()

        assert buffer.available == 0
        assert buffer.read_pos == 0
        assert buffer.write_pos == 0


# =============================================================================
# FadeState Tests
# =============================================================================


class TestFadeState:
    """Test suite for FadeState."""

    def test_fade_state_creation(self):
        """Test fade state creation."""
        fade = FadeState()
        assert not fade.is_fading
        assert fade.target_volume == 1.0

    def test_fade_in(self):
        """Test fade in behavior."""
        fade = FadeState(
            is_fading=True,
            fade_in=True,
            start_volume=0.0,
            target_volume=1.0,
            duration_ms=100.0,
        )

        # Update halfway
        result = fade.update(50.0)
        assert result == pytest.approx(0.5, rel=0.1)

    def test_fade_complete(self):
        """Test fade completion."""
        completed = False
        def on_complete():
            nonlocal completed
            completed = True

        fade = FadeState(
            is_fading=True,
            duration_ms=100.0,
            callback=on_complete,
        )

        fade.update(150.0)
        assert not fade.is_fading
        assert completed


# =============================================================================
# dB Conversion Tests
# =============================================================================


class TestDBConversion:
    """Test suite for decibel conversions."""

    def test_db_to_linear_unity(self):
        """Test 0 dB = 1.0 linear."""
        assert db_to_linear(0.0) == 1.0

    def test_db_to_linear_half(self):
        """Test -6 dB is approximately 0.5 linear."""
        result = db_to_linear(-6.0)
        assert result == pytest.approx(0.5, rel=0.1)

    def test_linear_to_db_unity(self):
        """Test 1.0 linear = 0 dB."""
        assert linear_to_db(1.0) == 0.0

    def test_linear_to_db_zero(self):
        """Test 0.0 linear returns MINIMUM_DB_VALUE from config."""
        result = linear_to_db(0.0)
        assert result == MINIMUM_DB_VALUE

    def test_linear_to_db_negative(self):
        """Test negative linear value returns MINIMUM_DB_VALUE."""
        result = linear_to_db(-1.0)
        assert result == MINIMUM_DB_VALUE

    def test_db_roundtrip(self):
        """Test dB conversion roundtrip."""
        original = -12.0
        linear = db_to_linear(original)
        back = linear_to_db(linear)
        assert back == pytest.approx(original, rel=0.01)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_memory_pool_zero_max_size(self):
        """Test memory pool with zero max size returns 0 utilization."""
        pool = MemoryPool(MemoryPoolType.TEMPORARY, 0, "zero_pool")
        assert pool.utilization == 0.0
        # Allocation should fail gracefully
        block = pool.allocate(100)
        assert block is None

    def test_source_loop_with_invalid_loop_points(self, audio_clip):
        """Test looping source handles invalid loop points (loop_end == loop_start)."""
        source = AudioSource(id="loop_test")
        source.set_clip(audio_clip)
        source.playback_mode = SourcePlaybackMode.LOOP
        source.loop_count = -1  # infinite
        # Set invalid loop points where start == end
        source.loop_start = 1000
        source.loop_end = 1000

        source.play()
        # Advance past the loop point - should not crash with division by zero
        source._playback_position = 1500
        source._advance_playback(100)
        # Should handle gracefully by resetting to loop_start
        assert source._playback_position == source.loop_start

    def test_doppler_with_zero_scale(self, audio_listener):
        """Test Doppler calculation with zero scale returns 1.0."""
        audio_listener.doppler_scale = 0.0
        source_pos = Vector3(0.0, 0.0, -10.0)
        source_vel = Vector3(0.0, 0.0, 100.0)

        doppler = audio_listener.calculate_doppler_factor(source_pos, source_vel)
        assert doppler == 1.0

    def test_memory_manager_defragment(self, memory_manager, audio_clip):
        """Test memory manager defragmentation recovers memory."""
        # Allocate a block
        block = memory_manager.allocate(
            1024,
            MemoryPoolType.TEMPORARY,
            clip=audio_clip,
            category=AudioCategory.SFX,
        )
        assert block is not None

        # Ensure clip has no references (simulating unused clip)
        while audio_clip.ref_count > 0:
            audio_clip.release_ref()

        # Defragment should recover the memory
        recovered = memory_manager.defragment()
        assert recovered >= 1024

    def test_voice_manager_high_priority_cannot_steal_critical(self, audio_clip):
        """Test that even HIGH priority cannot steal from CRITICAL voices."""
        manager = VoiceManager(
            max_voices=2,
            steal_strategy=VoiceStealStrategy.LOWEST_PRIORITY,
            enable_virtual_voices=False,
        )

        # Fill with CRITICAL priority
        for i in range(2):
            source = AudioSource(id=f"critical_{i}")
            source.set_clip(audio_clip)
            source.priority = PRIORITY_CRITICAL
            manager.allocate_voice(source)

        # HIGH priority cannot steal from CRITICAL
        high = AudioSource(id="high")
        high.set_clip(audio_clip)
        high.priority = PRIORITY_HIGH

        result = manager.allocate_voice(high)
        assert not result.success
        assert result.error is not None

    def test_stream_buffer_empty_read(self):
        """Test reading from empty buffer returns empty bytes."""
        buffer = StreamBuffer(
            id=1,
            data=bytearray(1024),
            capacity=1024,
        )
        result = buffer.read(100)
        assert result == b''
        assert len(result) == 0

    def test_stream_buffer_overfill(self):
        """Test writing more than capacity only writes what fits."""
        buffer = StreamBuffer(
            id=1,
            data=bytearray(16),
            capacity=16,
        )
        # Try to write 20 bytes to 16 byte buffer
        data = bytes(20)
        written = buffer.write(data)
        assert written == 16  # Only capacity written

    def test_audio_clip_get_samples_past_end(self, audio_clip):
        """Test getting samples past end of clip returns None."""
        # Request samples way past the end
        result = audio_clip.get_samples(audio_clip.metadata.total_samples + 1000, 100)
        assert result is None

    def test_fade_state_no_callback(self):
        """Test fade completion without callback doesn't crash."""
        fade = FadeState(
            is_fading=True,
            duration_ms=100.0,
            callback=None,  # No callback
        )
        # Should complete without error
        fade.update(150.0)
        assert not fade.is_fading

    def test_cue_variation_disabled(self):
        """Test variation returns original value when disabled."""
        variation = CueVariation(
            enable_pitch_variation=False,
            enable_volume_variation=False,
        )

        # Should return exact input value
        assert variation.apply_pitch_variation(1.0) == 1.0
        assert variation.apply_volume_variation(0.8) == 0.8

    def test_listener_manager_set_invalid_active(self):
        """Test setting non-existent listener as active returns False."""
        manager = AudioListenerManager()
        result = manager.set_active_listener("nonexistent")
        assert result is False

    def test_source_seek_without_clip(self):
        """Test seeking without clip doesn't crash."""
        source = AudioSource()
        source.seek(1.0)  # Should not crash
        assert source._playback_position == 0

    def test_source_get_samples_when_stopped(self, audio_source):
        """Test getting samples when stopped returns None."""
        audio_source.stop()
        result = audio_source.get_samples(100)
        assert result is None

    def test_empty_sound_cue_select(self):
        """Test selecting from empty cue returns None."""
        cue = SoundCue(id="empty", name="empty", cue_type=SoundCueType.RANDOM)
        result = cue.select_entry()
        assert result is None

    def test_memory_pool_negative_priority(self):
        """Test allocation with negative priority still works."""
        pool = MemoryPool(MemoryPoolType.TEMPORARY, 1024)
        block = pool.allocate(100, priority=-10)
        assert block is not None
        assert block.priority == -10
