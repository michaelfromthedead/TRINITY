# Investigation: engine/audio/core

## Summary
The audio core is a REAL IMPLEMENTATION of a game audio engine with comprehensive voice management, memory pooling, 3D spatial audio, streaming support, and sound cue variation systems. It follows production game engine patterns (voice stealing, virtual voices, priority-based allocation) and includes full threading models for audio/stream/decode threads. However, actual audio output/backend integration is not present in this directory - mixing produces no output (the `_fill_stream_buffers` method is stubbed).

## Files
| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `__init__.py` | 27 | Complete | Exports voice priority bridge APIs |
| `config.py` | 301 | Complete | All constants centralized (no magic numbers) |
| `audio_engine.py` | 825 | Complete | Main engine with threading model, command queue |
| `audio_source.py` | 705 | Complete | Full source with 3D, fades, looping, pooling |
| `audio_clip.py` | 588 | Complete | Clip loading, format detection (WAV/OGG/FLAC/MP3), metadata |
| `audio_listener.py` | 507 | Complete | 3D listener with Doppler, pan, attenuation |
| `voice_manager.py` | 658 | Complete | Voice allocation, limits, stealing, virtual voices |
| `virtual_voice.py` | 294 | Complete | Virtual voice tracking with urgency-based promotion |
| `memory_manager.py` | 795 | Complete | Memory pools, LRU eviction, streaming buffers |
| `sound_cue.py` | 623 | Complete | Cue system with random/sequence/shuffle/switch |
| `voice_priority_bridge.py` | ~200 | Complete | Bridge for @voice_priority decorator |

**Total: ~5,523 lines** in audio core alone.

## Audio Components
- `AudioEngine` - Main engine with game/audio/stream thread separation
- `AudioSource` - Playable source with volume, pitch, pan, 3D position, fades
- `AudioSourcePool` - Object pooling to reduce allocations (32 initial, 128 max)
- `AudioClip` - Loaded audio with format detection and metadata
- `AudioClipManager` - Clip caching and reference counting
- `AudioListener` - 3D listener with orientation and Doppler
- `AudioListenerManager` - Multi-listener support (split-screen)
- `VoiceManager` - Voice allocation with per-category limits
- `VirtualVoiceTracker` - Urgency scoring for promotion
- `SoundCue` - Variation container (random/sequence/shuffle/switch)
- `SoundCueManager` - Cue registry and instance tracking
- `AudioMemoryManager` - Memory pools (resident/streaming/temporary)
- `StreamBuffer` - Ring buffer for streaming audio

## Audio Implementation
- Real audio buffers? **YES** - `StreamBuffer` ring buffers with read/write positions
- Real mixing? **PARTIAL** - Command queue processes play/stop/fade but `_process_audio` does not output samples to hardware
- Real DSP? **YES** - 3D spatialization (attenuation, pan, Doppler), fade interpolation, pitch-adjusted playback
- Backend integration? **NO** - No actual audio output driver (OpenAL/SDL/WASAPI/etc.) present in this directory

## Verdict
**PARTIAL IMPLEMENTATION** - The audio core is a production-grade audio subsystem architecture with real voice management, memory pooling, 3D spatial audio math, and sound cue variation. The only missing piece is the actual audio output backend. The architecture is designed for one to be plugged in (the audio thread's `_process_audio` method would feed samples to a backend).

## Evidence

### Voice Stealing with Priority (voice_manager.py:281-302)
```python
def _steal_voice(self, requester: AudioSource) -> Optional[VoiceAllocationResult]:
    """Attempt to steal a voice based on strategy."""
    if self._steal_strategy == VoiceStealStrategy.NONE:
        return None

    candidates = self._get_steal_candidates(requester)
    if not candidates:
        return None

    # Sort by stealability (lowest priority first)
    if self._steal_strategy == VoiceStealStrategy.OLDEST:
        candidates.sort(key=lambda v: v.start_time)
    elif self._steal_strategy == VoiceStealStrategy.QUIETEST:
        candidates.sort(key=lambda v: v.volume)
    elif self._steal_strategy == VoiceStealStrategy.FARTHEST:
        candidates.sort(key=lambda v: -v.distance)
    elif self._steal_strategy == VoiceStealStrategy.LOWEST_PRIORITY:
        candidates.sort(key=lambda v: v.priority)
```

### 3D Audio Calculations (audio_listener.py:319-354)
```python
def calculate_3d_parameters(
    self,
    source_position: Vector3,
    source_velocity: Vector3,
    min_distance: float,
    max_distance: float,
    rolloff: float
) -> Tuple[float, float, float]:
    """
    Calculate 3D audio parameters for a source.
    Returns:
        Tuple of (attenuation, pan, doppler_factor)
    """
    distance = self.get_distance_to(source_position)
    pan = self.calculate_pan(source_position)
    doppler = self.calculate_doppler_factor(source_position, source_velocity)

    # Calculate distance attenuation (inverse distance)
    if distance <= min_distance:
        attenuation = 1.0
    elif distance >= max_distance:
        attenuation = 0.0
    else:
        # Inverse distance clamped
        attenuation = min_distance / (min_distance + rolloff * (distance - min_distance))
```

### Memory Pool LRU Eviction (memory_manager.py:269-298)
```python
def _evict(self, needed_size: int) -> int:
    """Evict blocks to free up space."""
    freed = 0

    # Get eviction candidates (unpinned, sorted by priority and LRU)
    candidates = [
        block for block in self._blocks.values()
        if not block.pinned
    ]
    candidates.sort()  # Uses __lt__ for eviction order

    for block in candidates:
        if freed >= needed_size:
            break

        freed += block.size
        self._used_size -= block.size
        del self._blocks[block.id]
```

### Stubbed Backend (audio_engine.py:758-761)
```python
def _fill_stream_buffers(self) -> None:
    """Fill streaming buffers with data."""
    # Implementation would read from files and fill buffers
    pass
```

## Architecture Notes
- **Threading**: Game thread sends commands via queue, audio thread processes at 200Hz, stream thread handles I/O at 100Hz
- **Voice limits**: 64 total, 48 SFX, 8 music, 8 VO, 16 ambient per category
- **Memory budget**: 256MB total (128MB SFX, 64MB music, 64MB VO, 32MB ambient)
- **Streaming**: Ring buffers with low/high watermarks for prefetch
- **Virtual voices**: Track position during virtualization, urgency-based promotion when slots free
