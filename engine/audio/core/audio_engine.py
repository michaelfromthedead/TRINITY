"""
Audio Engine

Main audio engine with threading model for game audio.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable, Any
from enum import IntEnum, auto
from contextlib import contextmanager

from .config import (
    MAX_VOICES,
    DEFAULT_SAMPLE_RATE,
    AUDIO_BUFFER_SIZE,
    AUDIO_TICK_RATE_MS,
    STREAM_TICK_RATE_MS,
    SOURCE_POOL_INITIAL_SIZE,
    AudioCategory,
    VoiceState,
    VoiceStealStrategy,
    MemoryPoolType,
    CATEGORY_VOICE_LIMITS,
)
from .audio_clip import AudioClip, AudioClipManager
from .audio_source import AudioSource, AudioSourcePool
from .audio_listener import AudioListener, AudioListenerManager, Vector3
from .voice_manager import VoiceManager, VoiceAllocationResult
from .sound_cue import SoundCue, SoundCueManager
from .memory_manager import AudioMemoryManager
from ..mixing import CATEGORY_TO_BUS
from ..mixing.mix_bus import MixBus
from ..mixing.mixer import Mixer, MixerConfig


class EngineState(IntEnum):
    """Audio engine states."""
    UNINITIALIZED = 0
    INITIALIZED = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPING = auto()
    STOPPED = auto()


class AudioCommand:
    """Base class for audio commands sent to audio thread."""
    pass


@dataclass
class PlayCommand(AudioCommand):
    """Command to play a source."""
    source: AudioSource
    fade_in_ms: float = 0


@dataclass
class StopCommand(AudioCommand):
    """Command to stop a source."""
    source: AudioSource
    fade_out_ms: float = 0


@dataclass
class PauseCommand(AudioCommand):
    """Command to pause a source."""
    source: AudioSource


@dataclass
class ResumeCommand(AudioCommand):
    """Command to resume a source."""
    source: AudioSource


@dataclass
class SetVolumeCommand(AudioCommand):
    """Command to set source volume."""
    source: AudioSource
    volume: float


@dataclass
class SetPositionCommand(AudioCommand):
    """Command to set 3D position."""
    source: AudioSource
    position: Vector3


@dataclass
class UpdateListenerCommand(AudioCommand):
    """Command to update listener state."""
    position: Vector3
    forward: Vector3
    up: Vector3
    velocity: Vector3


class AudioEngine:
    """
    Main audio engine for game audio.

    Threading Model:
    - Game Thread: Sends commands, manages high-level state
    - Audio Thread: Processes commands, mixes audio, updates voices
    - Stream Thread: Handles file I/O for streaming audio
    - Decode Thread: Decodes compressed audio formats

    Pipeline:
    Source -> Decode -> Process -> Spatialize -> Output
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        buffer_size: int = AUDIO_BUFFER_SIZE,
        max_voices: int = MAX_VOICES,
        enable_streaming: bool = True,
        enable_3d: bool = True
    ) -> None:
        """
        Initialize audio engine.

        Args:
            sample_rate: Output sample rate
            buffer_size: Audio buffer size in samples
            max_voices: Maximum simultaneous voices
            enable_streaming: Enable streaming playback
            enable_3d: Enable 3D audio processing
        """
        self._sample_rate = sample_rate
        self._buffer_size = buffer_size
        self._max_voices = max_voices
        self._enable_streaming = enable_streaming
        self._enable_3d = enable_3d

        # State
        self._state = EngineState.UNINITIALIZED
        self._state_lock = threading.RLock()

        # Managers
        self._clip_manager = AudioClipManager()
        self._listener_manager = AudioListenerManager()
        self._voice_manager = VoiceManager(
            max_voices=max_voices,
            steal_strategy=VoiceStealStrategy.LOWEST_PRIORITY,
            enable_virtual_voices=True
        )
        self._cue_manager = SoundCueManager()
        self._memory_manager = AudioMemoryManager()
        self._source_pool = AudioSourcePool(initial_size=SOURCE_POOL_INITIAL_SIZE, max_size=max_voices)

        # Active sources
        self._active_sources: Dict[str, AudioSource] = {}
        self._sources_lock = threading.RLock()

        # Command queue (game thread -> audio thread)
        self._command_queue: queue.Queue[AudioCommand] = queue.Queue()

        # Category volumes
        self._category_volumes: Dict[AudioCategory, float] = {
            cat: 1.0 for cat in AudioCategory
        }
        self._master_volume = 1.0
        self._muted = False

        # Threads
        self._audio_thread: Optional[threading.Thread] = None
        self._stream_thread: Optional[threading.Thread] = None
        self._decode_thread: Optional[threading.Thread] = None

        # Thread control
        self._audio_running = threading.Event()
        self._stream_running = threading.Event()
        self._decode_running = threading.Event()

        # Timing
        self._last_update_time = 0.0
        self._delta_time = 0.0
        self._frame_count = 0

        # Callbacks
        self.on_source_complete: Optional[Callable[[AudioSource], None]] = None
        self.on_voice_stolen: Optional[Callable[[AudioSource, AudioSource], None]] = None
        self.on_error: Optional[Callable[[str, Exception], None]] = None

        # Mixer
        self._mixer: Optional[Mixer] = None

        # Statistics
        self._stats = {
            'frames_processed': 0,
            'commands_processed': 0,
            'voices_active': 0,
            'cpu_usage': 0.0,
            'buffer_underruns': 0,
        }

    @property
    def state(self) -> EngineState:
        """Get current engine state."""
        with self._state_lock:
            return self._state

    @property
    def sample_rate(self) -> int:
        """Get output sample rate."""
        return self._sample_rate

    @property
    def is_running(self) -> bool:
        """Check if engine is running."""
        return self._state == EngineState.RUNNING

    @property
    def is_initialized(self) -> bool:
        """Check if engine is initialized."""
        return self._state != EngineState.UNINITIALIZED

    @property
    def master_volume(self) -> float:
        """Get master volume."""
        return self._master_volume

    @master_volume.setter
    def master_volume(self, value: float) -> None:
        """Set master volume."""
        self._master_volume = max(0.0, min(1.0, value))

    @property
    def muted(self) -> bool:
        """Check if muted."""
        return self._muted

    @muted.setter
    def muted(self, value: bool) -> None:
        """Set mute state."""
        self._muted = value

    @property
    def listener(self) -> AudioListener:
        """Get the active listener."""
        return self._listener_manager.active_listener

    @property
    def clip_manager(self) -> AudioClipManager:
        """Get clip manager."""
        return self._clip_manager

    @property
    def mixer(self) -> Optional[Mixer]:
        """Get the audio mixer."""
        return self._mixer

    @property
    def cue_manager(self) -> SoundCueManager:
        """Get cue manager."""
        return self._cue_manager

    @property
    def memory_manager(self) -> AudioMemoryManager:
        """Get memory manager."""
        return self._memory_manager

    @property
    def voice_manager(self) -> VoiceManager:
        """Get voice manager."""
        return self._voice_manager

    def initialize(self) -> bool:
        """
        Initialize the audio engine.

        Returns:
            True if initialized successfully
        """
        with self._state_lock:
            if self._state != EngineState.UNINITIALIZED:
                return False

            try:
                # Set up voice manager callbacks
                self._voice_manager.on_voice_stolen = self._on_voice_stolen

                # Initialize mixer
                self._mixer = Mixer(MixerConfig(sample_rate=self._sample_rate))
                self._mixer.initialize()

                self._state = EngineState.INITIALIZED
                return True

            except Exception as e:
                if self.on_error:
                    self.on_error("initialize", e)
                return False

    def start(self) -> bool:
        """
        Start the audio engine.

        Returns:
            True if started successfully
        """
        with self._state_lock:
            if self._state == EngineState.RUNNING:
                return True
            if self._state == EngineState.UNINITIALIZED:
                if not self.initialize():
                    return False

            try:
                # Start audio thread
                self._audio_running.set()
                self._audio_thread = threading.Thread(
                    target=self._audio_thread_func,
                    name="AudioThread",
                    daemon=True
                )
                self._audio_thread.start()

                # Start stream thread if enabled
                if self._enable_streaming:
                    self._stream_running.set()
                    self._stream_thread = threading.Thread(
                        target=self._stream_thread_func,
                        name="StreamThread",
                        daemon=True
                    )
                    self._stream_thread.start()

                self._last_update_time = time.time()
                self._state = EngineState.RUNNING
                return True

            except Exception as e:
                if self.on_error:
                    self.on_error("start", e)
                return False

    def stop(self) -> None:
        """Stop the audio engine."""
        with self._state_lock:
            if self._state not in (EngineState.RUNNING, EngineState.PAUSED):
                return

            self._state = EngineState.STOPPING

        # Stop threads
        self._audio_running.clear()
        self._stream_running.clear()
        self._decode_running.clear()

        # Wait for threads
        if self._audio_thread:
            self._audio_thread.join(timeout=1.0)
        if self._stream_thread:
            self._stream_thread.join(timeout=1.0)
        if self._decode_thread:
            self._decode_thread.join(timeout=1.0)

        # Stop all sources
        with self._sources_lock:
            for source in self._active_sources.values():
                source.stop()
            self._active_sources.clear()

        with self._state_lock:
            self._state = EngineState.STOPPED

    def pause(self) -> None:
        """Pause the audio engine."""
        with self._state_lock:
            if self._state == EngineState.RUNNING:
                self._state = EngineState.PAUSED
                self._voice_manager.pause_all()

    def resume(self) -> None:
        """Resume the audio engine."""
        with self._state_lock:
            if self._state == EngineState.PAUSED:
                self._state = EngineState.RUNNING
                self._voice_manager.resume_all()

    def update(self, delta_time: float = 0.0) -> None:
        """
        Update from game thread.

        Args:
            delta_time: Time since last update (auto-calculated if 0)
        """
        if self._state != EngineState.RUNNING:
            return

        # Calculate delta time if not provided
        current_time = time.time()
        if delta_time <= 0:
            delta_time = current_time - self._last_update_time
        self._last_update_time = current_time
        self._delta_time = delta_time

        # Update listener
        self._listener_manager.update(delta_time)

        # Send listener update to audio thread
        listener = self.listener
        self._command_queue.put(UpdateListenerCommand(
            position=Vector3(listener.position.x, listener.position.y, listener.position.z),
            forward=Vector3(listener.forward.x, listener.forward.y, listener.forward.z),
            up=Vector3(listener.up.x, listener.up.y, listener.up.z),
            velocity=Vector3(listener.velocity.x, listener.velocity.y, listener.velocity.z)
        ))

        # Update cue manager
        self._cue_manager.update(delta_time)

        self._frame_count += 1

    # =========================================================================
    # Source Management
    # =========================================================================

    def play(
        self,
        clip: AudioClip,
        volume: float = 1.0,
        pitch: float = 1.0,
        category: AudioCategory = AudioCategory.SFX,
        priority: int = 50,
        fade_in_ms: float = 0,
        looping: bool = False
    ) -> Optional[AudioSource]:
        """
        Play an audio clip.

        Args:
            clip: The audio clip to play
            volume: Playback volume (0-1)
            pitch: Playback pitch (0.1-4.0)
            category: Audio category
            priority: Voice priority
            fade_in_ms: Fade in duration
            looping: Whether to loop

        Returns:
            AudioSource or None if playback failed
        """
        if self._state != EngineState.RUNNING:
            return None

        source = self._source_pool.acquire()
        if not source:
            return None

        source.set_clip(clip)
        source.volume = volume
        source.pitch = pitch
        source.category = category
        source.priority = priority
        source.loop_count = -1 if looping else 0

        return self._play_source(source, fade_in_ms)

    def play_3d(
        self,
        clip: AudioClip,
        position: Vector3,
        volume: float = 1.0,
        pitch: float = 1.0,
        category: AudioCategory = AudioCategory.SFX,
        priority: int = 50,
        min_distance: float = 1.0,
        max_distance: float = 100.0,
        fade_in_ms: float = 0,
        looping: bool = False
    ) -> Optional[AudioSource]:
        """
        Play an audio clip in 3D space.

        Args:
            clip: The audio clip
            position: World position
            volume: Base volume
            pitch: Playback pitch
            category: Audio category
            priority: Voice priority
            min_distance: Full volume distance
            max_distance: Inaudible distance
            fade_in_ms: Fade in duration
            looping: Whether to loop

        Returns:
            AudioSource or None
        """
        source = self.play(clip, volume, pitch, category, priority, fade_in_ms, looping)
        if source:
            source.is_3d = True
            source.position = position
            source.min_distance = min_distance
            source.max_distance = max_distance

        return source

    def play_cue(
        self,
        cue: SoundCue,
        position: Optional[Vector3] = None,
        fade_in_ms: float = 0
    ) -> Optional[AudioSource]:
        """
        Play a sound cue.

        Args:
            cue: The sound cue
            position: Optional 3D position
            fade_in_ms: Fade in duration

        Returns:
            AudioSource or None
        """
        if self._state != EngineState.RUNNING:
            return None

        source = cue.create_source()
        if not source:
            return None

        if position and cue.is_3d:
            source.position = position

        return self._play_source(source, fade_in_ms)

    def _play_source(self, source: AudioSource, fade_in_ms: float = 0) -> Optional[AudioSource]:
        """Internal method to play a source."""
        # Route source to mixer bus by category
        mixer = self._mixer
        if mixer is not None:
            bus_name = CATEGORY_TO_BUS.get(source.category.name)
            if bus_name:
                mixer.route_source_to_bus(source.id, bus_name)

        # Allocate voice
        result = self._voice_manager.allocate_voice(source)
        if not result.success:
            self._source_pool.release(source)
            return None

        # Track source
        with self._sources_lock:
            self._active_sources[source.id] = source

        # Queue play command
        self._command_queue.put(PlayCommand(source, fade_in_ms))

        return source

    def stop_source(self, source: AudioSource, fade_out_ms: float = 0) -> None:
        """
        Stop a specific source.

        Args:
            source: Source to stop
            fade_out_ms: Fade out duration
        """
        self._command_queue.put(StopCommand(source, fade_out_ms))

    def pause_source(self, source: AudioSource) -> None:
        """Pause a specific source."""
        self._command_queue.put(PauseCommand(source))

    def resume_source(self, source: AudioSource) -> None:
        """Resume a specific source."""
        self._command_queue.put(ResumeCommand(source))

    def stop_all(self, fade_out_ms: float = 0) -> None:
        """Stop all sources."""
        with self._sources_lock:
            for source in list(self._active_sources.values()):
                self.stop_source(source, fade_out_ms)

    def stop_category(self, category: AudioCategory, fade_out_ms: float = 0) -> None:
        """Stop all sources in a category."""
        with self._sources_lock:
            for source in list(self._active_sources.values()):
                if source.category == category:
                    self.stop_source(source, fade_out_ms)

    # =========================================================================
    # Volume Control
    # =========================================================================

    def set_category_volume(self, category: AudioCategory, volume: float) -> None:
        """Set volume for an audio category."""
        self._category_volumes[category] = max(0.0, min(1.0, volume))

    def get_category_volume(self, category: AudioCategory) -> float:
        """Get volume for an audio category."""
        return self._category_volumes.get(category, 1.0)

    def get_effective_volume(self, source: AudioSource) -> float:
        """Calculate effective volume for a source."""
        if self._muted:
            return 0.0

        base = source.effective_volume
        category_vol = self._category_volumes.get(source.category, 1.0)
        master = self._master_volume

        return base * category_vol * master

    # =========================================================================
    # Listener
    # =========================================================================

    def set_listener_position(self, x: float, y: float, z: float) -> None:
        """Set listener position."""
        self.listener.set_position(x, y, z)

    def set_listener_orientation(
        self,
        forward_x: float, forward_y: float, forward_z: float,
        up_x: float, up_y: float, up_z: float
    ) -> None:
        """Set listener orientation."""
        self.listener.set_orientation(
            forward_x, forward_y, forward_z,
            up_x, up_y, up_z
        )

    def set_listener_velocity(self, x: float, y: float, z: float) -> None:
        """Set listener velocity."""
        self.listener.set_velocity(x, y, z)

    # =========================================================================
    # Thread Functions
    # =========================================================================

    def _audio_thread_func(self) -> None:
        """Audio thread main function."""
        tick_interval = AUDIO_TICK_RATE_MS / 1000.0

        while self._audio_running.is_set():
            start_time = time.time()

            try:
                # Process commands
                self._process_commands()

                # Update voice manager
                self._voice_manager.update(tick_interval)

                # Process audio (mixing would happen here)
                self._process_audio(tick_interval)

                # Run mixer tick pipeline
                mixer = self._mixer
                if mixer is not None:
                    try:
                        mixer.update(tick_interval)
                        mixer.tick(self._buffer_size)
                    except Exception:
                        pass

                # Update statistics
                self._stats['frames_processed'] += 1
                self._stats['voices_active'] = self._voice_manager.active_voice_count

            except Exception as e:
                if self.on_error:
                    self.on_error("audio_thread", e)

            # Sleep for remainder of tick
            elapsed = time.time() - start_time
            sleep_time = max(0, tick_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _stream_thread_func(self) -> None:
        """Stream thread main function for I/O."""
        tick_interval = STREAM_TICK_RATE_MS / 1000.0

        while self._stream_running.is_set():
            start_time = time.time()

            try:
                # Process prefetch queue
                self._memory_manager.process_prefetch()

                # Fill streaming buffers
                self._fill_stream_buffers()

            except Exception as e:
                if self.on_error:
                    self.on_error("stream_thread", e)

            elapsed = time.time() - start_time
            sleep_time = max(0, tick_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _process_commands(self) -> None:
        """Process queued commands."""
        while True:
            try:
                command = self._command_queue.get_nowait()
            except queue.Empty:
                break

            self._stats['commands_processed'] += 1

            if isinstance(command, PlayCommand):
                self._execute_play(command)
            elif isinstance(command, StopCommand):
                self._execute_stop(command)
            elif isinstance(command, PauseCommand):
                command.source.pause()
            elif isinstance(command, ResumeCommand):
                command.source.resume()
            elif isinstance(command, SetVolumeCommand):
                command.source.volume = command.volume
            elif isinstance(command, SetPositionCommand):
                command.source.position = command.position
            elif isinstance(command, UpdateListenerCommand):
                self._voice_manager.set_listener_position(command.position)

    def _execute_play(self, command: PlayCommand) -> None:
        """Execute play command."""
        source = command.source
        source.play(command.fade_in_ms)

    def _execute_stop(self, command: StopCommand) -> None:
        """Execute stop command."""
        source = command.source
        source.stop(command.fade_out_ms)

        # Unroute source from mixer
        mixer = self._mixer
        if mixer is not None:
            mixer.unroute_source(source.id)

        # Release voice
        if source.voice_id is not None:
            self._voice_manager.release_voice(source.voice_id)

        # Return to pool
        with self._sources_lock:
            if source.id in self._active_sources:
                del self._active_sources[source.id]
        self._source_pool.release(source)

        if self.on_source_complete:
            self.on_source_complete(source)

    def _process_audio(self, delta_time: float) -> None:
        """Process audio mixing and spatialization."""
        delta_ms = delta_time * 1000.0

        with self._sources_lock:
            completed = []

            for source_id, source in self._active_sources.items():
                if source.is_stopped:
                    completed.append(source_id)
                    continue

                # Update source
                source.update(delta_ms)

                # Calculate 3D parameters if applicable
                if source.is_3d and self._enable_3d:
                    attenuation, pan, doppler = self.listener.calculate_3d_parameters(
                        source.position,
                        source.velocity,
                        source.min_distance,
                        source.max_distance,
                        source.rolloff_factor
                    )
                    source.calculate_output_volumes(attenuation, pan, doppler)

            # Clean up completed sources
            for source_id in completed:
                source = self._active_sources.pop(source_id)
                if source.voice_id is not None:
                    self._voice_manager.release_voice(source.voice_id)
                self._source_pool.release(source)
                if self.on_source_complete:
                    self.on_source_complete(source)

    def _fill_stream_buffers(self) -> None:
        """Fill streaming buffers with data."""
        # Implementation would read from files and fill buffers
        pass

    def _on_voice_stolen(self, victim: AudioSource, thief: AudioSource) -> None:
        """Handle voice stolen callback."""
        if self.on_voice_stolen:
            self.on_voice_stolen(victim, thief)

    # =========================================================================
    # Statistics and Debugging
    # =========================================================================

    def get_stats(self) -> dict:
        """Get engine statistics."""
        with self._sources_lock:
            active_count = len(self._active_sources)

        return {
            'state': self._state.name,
            'sample_rate': self._sample_rate,
            'buffer_size': self._buffer_size,
            'master_volume': self._master_volume,
            'muted': self._muted,
            'active_sources': active_count,
            'voice_stats': self._voice_manager.get_stats(),
            'memory_stats': self._memory_manager.get_stats(),
            'frame_count': self._frame_count,
            **self._stats,
        }

    def get_active_sources(self) -> List[AudioSource]:
        """Get list of active sources."""
        with self._sources_lock:
            return list(self._active_sources.values())

    @contextmanager
    def batch_operations(self):
        """
        Context manager for batching operations.

        Commands sent during batch are processed together.
        """
        yield
        # Commands are already queued, no special handling needed

    def __enter__(self) -> 'AudioEngine':
        """Context manager enter - start engine."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stop engine."""
        self.stop()

    def shutdown(self) -> None:
        """Fully shutdown the engine."""
        self.stop()

        # Shutdown mixer
        if self._mixer is not None:
            self._mixer.shutdown()
            self._mixer = None

        # Clean up managers
        self._clip_manager.unload_all()
        self._memory_manager.clear_all()
        self._source_pool.clear()

        with self._state_lock:
            self._state = EngineState.UNINITIALIZED
