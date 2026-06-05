# RECOMMENDATIONS.md: engine/audio/adaptive + engine/audio/core

---

## Rust Bridge Requirements

### High Priority

| Component | Reason | Estimated Effort |
|-----------|--------|------------------|
| **Voice Mixer** | 5ms audio tick is latency-critical; Rust SIMD can eliminate Python GIL contention | 2-3 days |
| **Streaming I/O** | `_fill_stream_buffers()` is a stub; Rust async I/O with memory-mapped files needed | 3-4 days |
| **Memory Allocator** | Unify Python's AudioMemoryManager with Rust allocator for GPU upload staging | 2 days |

### Medium Priority

| Component | Reason | Estimated Effort |
|-----------|--------|------------------|
| **Doppler/Attenuation** | SIMD batch calculation for N sources at once | 1 day |
| **Beat Grid** | Lock-free beat tracking with sub-millisecond precision | 1 day |
| **Voice Steal Sort** | Priority heap in Rust for O(1) steal candidate | 1 day |

### Low Priority

| Component | Reason | Estimated Effort |
|-----------|--------|------------------|
| **Crossfade Curves** | Math-only, low call frequency | 0.5 days |
| **State Machine** | Lookup table, not performance-critical | 0.5 days |
| **Callback Dispatch** | Event-driven, not hot path | 0.5 days |

---

## Integration Strategy

### Phase 1: Core Bridge (Week 1)

1. Create `renderer-backend/src/audio_bridge.rs` with:
   ```rust
   #[pyfunction]
   fn process_voices(sources: Vec<PyRef<AudioSource>>, ...) -> Vec<OutputSample>;
   
   #[pyfunction]
   fn allocate_voice(priority: i32, category: AudioCategory) -> VoiceAllocationResult;
   
   #[pyfunction]
   fn fill_stream_buffers(buffer_ids: Vec<u32>) -> Result<(), AudioError>;
   ```

2. Expose via PyO3 to `engine.audio.core.audio_engine`

3. Replace hot-path methods with Rust calls:
   - `_process_audio()` -> `process_voices()`
   - `_fill_stream_buffers()` -> `fill_stream_buffers()`

### Phase 2: Memory Unification (Week 2)

1. Bridge `AudioMemoryManager` pools to Rust:
   ```rust
   struct SharedAudioPool {
       resident: Arc<Mutex<Pool<64_MB>>>,
       streaming: Arc<Mutex<Pool<32_MB>>>,
       temporary: Arc<Mutex<Pool<16_MB>>>,
   }
   ```

2. Expose pool handles to Python via PyO3

3. Use shared pools for GPU staging buffers in renderer

### Phase 3: GPU DSP (Week 3)

1. Move DSP effects (from `engine/audio/dsp`) to WGSL compute shaders:
   - `reverb.py` -> `reverb.wgsl`
   - `dynamics.py` -> `compressor.wgsl`
   - `filters.py` -> `biquad.wgsl`

2. Batch process all active voices per frame

---

## Testing Strategy

### Unit Tests

| Test | Target | Method |
|------|--------|--------|
| Crossfade curves | music_stem.py:FadeCurve | Assert equal_power(0.5) == sin(pi/4) |
| Beat quantization | music_timing.py:BeatGrid | Assert quantize_to_beat(125ms) == 125ms at 120 BPM |
| Voice stealing | voice_manager.py | Mock 64 voices, assert lowest priority stolen |
| LRU eviction | memory_manager.py | Fill pool, assert oldest unpinned evicted |
| Doppler | audio_listener.py | Assert approaching source > 1.0 pitch |

### Integration Tests

| Test | Target | Method |
|------|--------|--------|
| Command queue | audio_engine.py | Send 1000 commands, assert all processed |
| Adaptive intensity | adaptive_music.py | Set intensity 0->1, assert correct stems active |
| State transitions | music_state.py | Combat->Exploration->Boss, assert priority respected |

### Blackbox Tests (Post-Rust Bridge)

| Test | Target | Method |
|------|--------|--------|
| Rust voice mixer | audio_bridge.rs | Compare Python vs Rust output samples |
| Streaming latency | fill_stream_buffers | Assert <10ms buffer fill time |
| Memory fragmentation | SharedAudioPool | 10000 alloc/free cycles, assert no leak |

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| GIL contention at 5ms tick | High | Rust bridge for voice processing |
| Memory fragmentation | Medium | Rust allocator with defragment() |
| Beat callback jitter | Medium | Lock-free MusicClock in Rust |
| Streaming underruns | High | Double-buffering with prefetch |
| State machine deadlock | Low | Already uses RLock correctly |

### Dependencies

| Dependency | Version | Risk |
|------------|---------|------|
| Python threading | 3.x | Low (well-understood) |
| PyO3 | 0.21+ | Low (mature) |
| wgpu-py | N/A | Medium (for GPU DSP) |

---

## Recommended Priority Order

1. **Rust Voice Mixer** - Immediate, unblocks all audio performance
2. **Streaming I/O** - Required for large audio files
3. **Memory Unification** - Required for GPU staging
4. **GPU DSP** - Nice-to-have, batch processing gain
5. **Beat Grid** - Nice-to-have, sub-ms precision

Total estimated effort: 2-3 weeks for full Rust bridge.
