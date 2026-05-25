# PHASE 3 TODO: Backend Integration

**Phase**: 3 of 3
**Status**: NOT STARTED
**Prerequisites**: Phase 2 stable

---

## Milestone: Phase 3a - miniaudio POC

### T-AUDIO-3.1: Add miniaudio Dependency

**Priority**: Critical
**Description**: Add miniaudio Python package to project dependencies.

**Acceptance Criteria**:
- [ ] miniaudio added to requirements/dependencies
- [ ] Import succeeds on all target platforms
- [ ] Basic playback test works standalone

**Effort**: Trivial (30 minutes)

---

### T-AUDIO-3.2: Implement MiniaudioBackend

**Priority**: Critical
**Description**: Create AudioBackend implementation using miniaudio.

**Acceptance Criteria**:
- [ ] MiniaudioBackend class implements AudioBackend protocol
- [ ] initialize() creates miniaudio device
- [ ] shutdown() cleans up device
- [ ] write_samples() outputs audio
- [ ] get_buffer_status() returns fill level

**Effort**: Small (4-6 hours)

---

### T-AUDIO-3.3: Wire AudioEngine to Backend

**Priority**: Critical
**Description**: Connect AudioEngine to MiniaudioBackend.

**Acceptance Criteria**:
- [ ] AudioEngine accepts backend parameter
- [ ] _process_audio() calls backend.write_samples()
- [ ] Engine start/stop calls backend init/shutdown
- [ ] Basic play() produces audible output

**Effort**: Small (2-4 hours)

---

### T-AUDIO-3.4: Implement _fill_stream_buffers

**Priority**: Critical
**Description**: Implement the stubbed streaming method.

**Acceptance Criteria**:
- [ ] Check watermark levels
- [ ] Read chunks from AudioClip
- [ ] Fill StreamBuffer ring buffers
- [ ] Handle EOF and looping

**Effort**: Medium (4-6 hours)

---

## Milestone: Phase 3b - Streaming

### T-AUDIO-3.5: Decode Thread Pool

**Priority**: High
**Description**: Implement thread pool for compressed audio decoding.

**Acceptance Criteria**:
- [ ] ThreadPoolExecutor for decode tasks
- [ ] Priority queue (music > VO > SFX)
- [ ] Decoded chunks returned via future
- [ ] Integration with _fill_stream_buffers

**Effort**: Medium (6-8 hours)

---

### T-AUDIO-3.6: Large File Streaming Test

**Priority**: High
**Description**: Verify streaming works with large music files.

**Acceptance Criteria**:
- [ ] 5+ minute music file streams without hiccups
- [ ] Memory stays within streaming pool budget
- [ ] No audible gaps during buffer refill
- [ ] Loop points work correctly

**Effort**: Small (2-3 hours testing)

---

## Milestone: Phase 3c - Full Integration

### T-AUDIO-3.7: Voice Mixing

**Priority**: High
**Description**: Implement multi-voice mixing in _process_audio.

**Acceptance Criteria**:
- [ ] Mix all active voices into single buffer
- [ ] Apply per-voice volume/pan
- [ ] Apply 3D spatialization
- [ ] Apply Doppler pitch shift

**Effort**: Medium (6-8 hours)

---

### T-AUDIO-3.8: Master Output Chain

**Priority**: Medium
**Description**: Apply master volume and limiting before output.

**Acceptance Criteria**:
- [ ] Master volume control
- [ ] Limiter to prevent clipping
- [ ] Fade in/out on engine start/stop

**Effort**: Small (2-4 hours)

---

### T-AUDIO-3.9: Stress Test 64 Voices

**Priority**: High
**Description**: Verify system handles maximum voice count.

**Acceptance Criteria**:
- [ ] 64 voices playing simultaneously
- [ ] No buffer underruns
- [ ] Audio thread stays under 5ms budget
- [ ] Voice stealing kicks in correctly at limit

**Effort**: Small (2-3 hours testing)

---

### T-AUDIO-3.10: 3D Audio Verification

**Priority**: High
**Description**: Verify 3D audio calculations produce correct output.

**Acceptance Criteria**:
- [ ] Distance attenuation audible
- [ ] Pan tracks source position
- [ ] Doppler audible on moving sources
- [ ] Matches expected values from V-AUDIO-2.3

**Effort**: Small (2-3 hours testing)

---

## Milestone: Phase 3d - Rust Backend (Future)

### T-AUDIO-3.11: Rust AudioBackend Crate

**Priority**: Low (Future)
**Description**: Implement audio backend in Rust.

**Acceptance Criteria**:
- [ ] Rust crate with platform audio output
- [ ] Same buffer interface as MiniaudioBackend
- [ ] Lower latency than Python implementation

**Effort**: Large (2-3 weeks)

---

### T-AUDIO-3.13: Performance Comparison

**Priority**: Low (Future)
**Description**: Benchmark Rust vs miniaudio backend.

**Acceptance Criteria**:
- [ ] Latency measurement
- [ ] CPU usage comparison
- [ ] Voice count stress test
- [ ] Decision on default backend

**Effort**: Small (days)

---

## Verification Tasks

### V-AUDIO-3.1: End-to-End Playback

**Description**: Play a sound and hear it.

**Steps**:
1. Initialize AudioEngine with backend
2. Load test clip
3. Call engine.play(clip)
4. Verify audio output (ears)

---

### V-AUDIO-3.2: Adaptive Music Integration

**Description**: Verify adaptive music plays through backend.

**Steps**:
1. Initialize AdaptiveMusicSystem
2. Start playback
3. Change intensity
4. Verify stem volumes change audibly

---

### V-AUDIO-3.3: Buffer Underrun Detection

**Description**: Verify system handles high load without underruns.

**Steps**:
1. Play 64 voices
2. Trigger heavy game logic
3. Monitor for audio glitches
4. Check underrun counter

---

## Dependencies

| Task | Depends On | Blocks |
|------|------------|--------|
| T-AUDIO-3.1 | Nothing | All Phase 3 |
| T-AUDIO-3.2 | T-AUDIO-3.1 | T-AUDIO-3.3 |
| T-AUDIO-3.3 | T-AUDIO-3.2 | T-AUDIO-3.4 |
| T-AUDIO-3.4 | T-AUDIO-3.3 | T-AUDIO-3.5 |
| T-AUDIO-3.5 | T-AUDIO-3.4 | T-AUDIO-3.6 |
| T-AUDIO-3.6 | T-AUDIO-3.5 | Phase 3c |
| T-AUDIO-3.7 | T-AUDIO-3.4 | T-AUDIO-3.9 |
| T-AUDIO-3.8 | T-AUDIO-3.7 | Nothing |
| T-AUDIO-3.9 | T-AUDIO-3.7 | Nothing |
| T-AUDIO-3.10 | T-AUDIO-3.7 | Nothing |
| T-AUDIO-3.11 | Phase 3c complete | T-AUDIO-3.12 |
| T-AUDIO-3.12 | T-AUDIO-3.11 | T-AUDIO-3.13 |
| T-AUDIO-3.13 | T-AUDIO-3.12 | Production decision |
