# PROJECT: engine_audio_adaptive_core

**Subsystem**: Audio (Adaptive Music + Core Engine)
**Total Lines**: ~10,600
**Classification**: MOSTLY REAL (Adaptive: REAL, Core: PARTIAL)

---

## 1. Scope

### 1.1 What This Subsystem Is

The audio subsystem provides:
- **Adaptive Music System**: Dynamic game music with vertical layering (intensity-based stem control) and horizontal re-sequencing (section-based branching)
- **Core Audio Engine**: Voice management, memory pooling, 3D spatial audio, format handling, and sound cue variation

### 1.2 What This Subsystem Is Not

- **Not an audio backend**: No actual audio output driver (OpenAL/SDL/WASAPI) - this is the middleware layer above platform APIs
- **Not a DSP library**: While it includes spatial DSP (Doppler, attenuation), it does not include general-purpose audio effects (EQ, compression, etc.) - those are in `engine/audio/dsp`
- **Not the mixing layer**: Mixing and HDR audio are handled in `engine/audio/mixing`

### 1.3 Boundaries

| Layer | This Subsystem | Adjacent Subsystem |
|-------|---------------|-------------------|
| Above | Game logic integration | Gameplay systems |
| Below | Audio backend/output | engine/platform/audio |
| Peer | Format handling | engine/audio/dsp for effects |
| Peer | Sound cues | engine/audio/mixing for final mix |

---

## 2. Goals

### 2.1 Primary Goals

1. **Provide production-quality adaptive music** comparable to commercial middleware (FMOD/Wwise)
2. **Manage voice allocation efficiently** with priority-based stealing and virtual voices
3. **Enable memory-efficient audio** through pooling, streaming, and LRU eviction
4. **Support 3D spatial audio** with proper Doppler and distance attenuation

### 2.2 Quality Attributes

| Attribute | Target | Evidence |
|-----------|--------|----------|
| Latency | 5ms tick (200Hz audio thread) | audio_engine.py threading model |
| Memory | 256MB total budget with pooling | memory_manager.py configuration |
| Voices | 64 concurrent with virtual fallback | voice_manager.py implementation |
| Precision | 5ms beat callback accuracy | music_callback.py documentation |

---

## 3. Constraints

### 3.1 Technical Constraints

1. **No audio output**: The `_fill_stream_buffers` method is stubbed - integration with platform audio backend required
2. **Simplified format parsing**: OGG uses default sample rate; full parser needed for production
3. **Threading model assumed**: Code assumes game/audio/stream/decode thread separation exists

### 3.2 Design Constraints

1. **Command queue pattern mandatory**: Game thread must never directly manipulate audio state
2. **Voice limits enforced**: 64 total, per-category caps prevent runaway resource usage
3. **Memory budgets respected**: LRU eviction kicks in automatically when pools exhaust

### 3.3 Integration Constraints

1. **Requires platform audio layer**: Must integrate with `engine/platform/audio` for actual output
2. **Assumes ECS context**: Audio sources attach to game entities
3. **Listener tracking**: Game must call `set_listener_position` each frame

---

## 4. Stakeholders

| Stakeholder | Concern |
|-------------|---------|
| Sound Designer | Adaptive music authoring, stem configuration, state machine setup |
| Game Programmer | API usability, play/stop/fade calls, listener tracking |
| Performance Engineer | Memory budgets, voice limits, thread tick rates |
| Platform Engineer | Backend integration, format support, streaming |

---

## 5. Risks

### 5.1 Identified Risks

| Risk | Severity | Mitigation Status |
|------|----------|-------------------|
| Backend stub blocks audio output | HIGH | Requires platform integration work |
| OGG parsing incomplete | MEDIUM | Works for most files, needs full parser |
| Duplicate constants in config | LOW | Copy-paste error, no functional impact |
| music_state track-end integration incomplete | LOW | Comment acknowledges gap |

### 5.2 Open Questions

1. How will Rust backend (if any) integrate with Python memory pools?
2. Which platform audio API (OpenAL/SDL/WASAPI/CoreAudio) will be used?
3. Will DSP effects chain integrate before or after spatial processing?

---

## 6. Success Criteria

### 6.1 Functional Criteria

- [ ] Adaptive music responds to gameplay intensity in real-time
- [ ] Voice stealing degrades gracefully under pressure
- [ ] 3D audio pans and attenuates correctly
- [ ] Beat-synced transitions quantize to musical boundaries

### 6.2 Non-Functional Criteria

- [ ] Audio thread meets 5ms tick budget consistently
- [ ] Memory usage stays within 256MB budget
- [ ] Voice virtualization maintains positional tracking
- [ ] No audible pops/clicks during voice stealing

---

## 7. Definition of Done

This subsystem is considered complete when:
1. Audio backend integration provides actual audio output
2. OGG parser reads sample rate from headers
3. music_state track-end integration is wired
4. All 4 voice stealing strategies verified under load
5. Beat callback precision validated at 5ms target
