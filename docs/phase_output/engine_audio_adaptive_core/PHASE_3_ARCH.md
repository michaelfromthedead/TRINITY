# PHASE 3 ARCHITECTURE: Backend Integration

**Phase**: 3 of 3
**Status**: NOT STARTED (stubbed methods)
**Dependencies**: Phase 2 complete
**Classification**: GAP

---

## 1. Overview

Phase 3 bridges the Python audio middleware to platform audio output. This phase requires implementing the stubbed methods in AudioEngine to produce actual audio output.

---

## 2. The Gap

### 2.1 Stubbed Method (audio_engine.py:758-761)

```python
def _fill_stream_buffers(self) -> None:
    """Fill streaming buffers with data."""
    # Implementation would read from files and fill buffers
    pass
```

This method should:
1. Check streaming buffer watermarks
2. Read compressed audio data from files
3. Decode via decode thread
4. Fill StreamBuffer ring buffers

### 2.2 Missing: Audio Output Binding

The audio thread's `_process_audio()` method produces no samples. It should:
1. Mix active voices
2. Apply 3D spatialization
3. Write samples to platform audio buffer

---

## 3. Integration Options

### 3.2 Option B: ctypes/OpenAL

**Description**: Bind directly to OpenAL via ctypes.

**Advantages**:
- Cross-platform (OpenAL-Soft)
- Well-documented API
- No Rust dependency

**Architecture**:
```
Python AudioEngine
        |
        v
ctypes OpenAL bindings
        |
        v
OpenAL-Soft
        |
        v
Platform API
```

**Effort**: Medium (1-2 weeks)

### 3.3 Option C: SDL_mixer

**Description**: Use SDL2 audio via pygame or similar.

**Advantages**:
- Simple API
- Well-tested
- Existing Python bindings

**Architecture**:
```
Python AudioEngine
        |
        v
pygame.mixer / pysdl2
        |
        v
SDL2 Audio
        |
        v
Platform API
```

**Effort**: Small (days)

### 3.4 Option D: miniaudio

**Description**: Use miniaudio Python bindings.

**Advantages**:
- Single-file library
- Low latency
- Good Python support

**Architecture**:
```
Python AudioEngine
        |
        v
miniaudio-python
        |
        v
miniaudio
        |
        v
Platform API
```

**Effort**: Small (days)

---

## 5. Interface Contract

### 5.1 Backend Must Implement

```python
class AudioBackend(Protocol):
    def initialize(self, sample_rate: int, channels: int, buffer_size: int) -> bool:
        """Initialize audio output."""
        ...

    def shutdown(self) -> None:
        """Shutdown audio output."""
        ...

    def write_samples(self, samples: bytes, format: AudioFormat) -> int:
        """Write samples to output buffer. Returns bytes written."""
        ...

    def get_buffer_status(self) -> BufferStatus:
        """Get current buffer fill level."""
        ...
```

### 5.2 AudioEngine Modifications

```python
class AudioEngine:
    def __init__(self, backend: AudioBackend = None):
        self._backend = backend or DefaultBackend()

    def _process_audio(self) -> None:
        # Mix active voices into buffer
        mixed = self._mix_voices()

        # Apply master effects
        processed = self._apply_master_chain(mixed)

        # Output
        self._backend.write_samples(processed)
```

---

## 6. Streaming Integration

### 6.1 _fill_stream_buffers Implementation

```python
def _fill_stream_buffers(self) -> None:
    for source in self._streaming_sources:
        buffer = source.stream_buffer
        if buffer.fill_level < buffer.low_watermark:
            # Read from file
            chunk = source.clip.read_chunk(STREAM_CHUNK_SIZE)
            # Decode if compressed
            if source.clip.is_compressed:
                chunk = self._decode_queue.submit(chunk)
            # Fill buffer
            buffer.write(chunk)
```

### 6.2 Decode Thread Integration

The decode thread already exists conceptually. Implementation:
1. Thread pool for decode tasks
2. Priority queue (music > VO > SFX for streaming)
3. Decoded chunks returned to audio thread

---

## 7. Testing Strategy

### 7.1 Unit Tests

- Backend mock for testing AudioEngine without audio output
- Verify command processing produces expected backend calls

### 7.2 Integration Tests

- Actual audio output verification (manual or frequency analysis)
- Latency measurement
- Buffer underrun detection

### 7.3 Performance Tests

- 64 simultaneous voices
- Streaming while mixing
- Memory budget compliance

---

## 8. Rollout Plan

### 8.1 Phase 3a: miniaudio POC

1. Add miniaudio dependency
2. Implement MiniaudioBackend
3. Wire _fill_stream_buffers
4. Basic playback test

**Duration**: 2-3 days

### 8.2 Phase 3b: Streaming

1. Implement streaming buffer fill
2. Add decode thread pool
3. Test with large files

**Duration**: 3-5 days

### 8.3 Phase 3c: Full Integration

1. All voice types playing
2. 3D audio verified
3. Stress testing

**Duration**: 1 week
