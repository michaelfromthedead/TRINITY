# PHASE 2 TODO: Audio Core Engine

**Phase**: 2 of 3
**Status**: PARTIAL (backend integration required)

---

## Completed Tasks

- [x] AudioEngine with threading model
- [x] Command queue pattern
- [x] VoiceManager with 4 stealing strategies
- [x] VirtualVoiceTracker with urgency scoring
- [x] AudioMemoryManager with 3 pools
- [x] LRU eviction with priority weighting
- [x] StreamBuffer ring buffers
- [x] AudioSource with 3D properties
- [x] AudioSourcePool object reuse
- [x] AudioClip format detection (WAV/OGG/FLAC/MP3)
- [x] AudioListener 3D calculations
- [x] AudioListenerManager multi-listener
- [x] SoundCue with 4 play modes
- [x] SoundCueBuilder fluent API
- [x] VoicePriorityBridge decorator integration
- [x] Thread safety with locks
- [x] Configuration centralization

---

## Remaining Tasks

### T-AUDIO-2.1: Full OGG Parser

**Priority**: High
**Location**: `audio_clip.py:329-334`
**Description**: Replace simplified OGG parsing with full header reading.

**Acceptance Criteria**:
- [ ] Read sample rate from OGG headers
- [ ] Read channel count from OGG headers
- [ ] Handle Vorbis comments for metadata
- [ ] Support Opus in OGG container

**Effort**: Medium (4-6 hours)

---

### T-AUDIO-2.2: Remove Duplicate Constants

**Priority**: Low
**Location**: `config.py:233-260, 254-260, 293-300`
**Description**: Remove copy-pasted VIRTUAL_VOICE_* constant blocks.

**Acceptance Criteria**:
- [ ] Single definition of each constant
- [ ] All references updated
- [ ] Tests pass

**Effort**: Trivial (15 minutes)

---

### T-AUDIO-2.3: Memory Pool Metrics

**Priority**: Medium
**Location**: `memory_manager.py`
**Description**: Add metrics collection for pool usage, evictions, fragmentation.

**Acceptance Criteria**:
- [ ] Track allocations/frees per pool
- [ ] Track eviction count and freed bytes
- [ ] Track fragmentation ratio
- [ ] Expose metrics via API

**Effort**: Small (2-3 hours)

---

### T-AUDIO-2.4: Voice Virtualization Metrics

**Priority**: Medium
**Location**: `voice_manager.py`
**Description**: Track virtualization frequency and promotion success.

**Acceptance Criteria**:
- [ ] Track voices virtualized per second
- [ ] Track promotion success rate
- [ ] Track average virtualization duration
- [ ] Alert on high virtualization rate

**Effort**: Small (2-3 hours)

---

### T-AUDIO-2.5: Streaming Prefetch Tuning

**Priority**: Low
**Location**: `memory_manager.py`
**Description**: Make watermark thresholds configurable per category.

**Acceptance Criteria**:
- [ ] Low/high watermarks configurable
- [ ] Music uses larger buffers than SFX
- [ ] VO uses prioritized prefetch

**Effort**: Small (1-2 hours)

---

## Verification Tasks

### V-AUDIO-2.1: Voice Stealing Correctness

**Description**: Verify each stealing strategy selects correct victim.

**Steps**:
1. Fill all 64 voices
2. Request new voice with OLDEST strategy
3. Verify oldest voice stolen
4. Repeat for QUIETEST, FARTHEST, LOWEST_PRIORITY

---

### V-AUDIO-2.2: LRU Eviction Order

**Description**: Verify eviction respects priority and access time.

**Steps**:
1. Fill pool with blocks of varying priority
2. Access some blocks recently
3. Request allocation that requires eviction
4. Verify low-priority, old-access blocks evicted first

---

### V-AUDIO-2.3: 3D Audio Calculations

**Description**: Verify Doppler and attenuation math.

**Steps**:
1. Place source at known distance
2. Calculate expected attenuation
3. Verify AudioListener.calculate_3d_parameters matches
4. Move source with known velocity
5. Verify Doppler factor matches expected

---

### V-AUDIO-2.4: Sound Cue Variation

**Description**: Verify variation produces expected distribution.

**Steps**:
1. Create cue with 10% pitch variation
2. Play 1000 times, collect pitch values
3. Verify normal distribution within range
4. Verify no repeats within history depth

---

## Dependencies

| Task | Depends On | Blocks |
|------|------------|--------|
| T-AUDIO-2.1 | Nothing | OGG playback correctness |
| T-AUDIO-2.2 | Nothing | Nothing |
| T-AUDIO-2.3 | Nothing | Performance tuning |
| T-AUDIO-2.4 | Nothing | Performance tuning |
| T-AUDIO-2.5 | Nothing | Streaming quality |

---

## Phase 3 Prerequisite

Before Phase 3 (Backend Integration) can proceed, Phase 2 must have:
- [x] Stable command queue interface
- [x] Stable memory pool interface
- [x] Stable streaming buffer interface
- [ ] T-AUDIO-2.1 (OGG parser) - recommended but not blocking
