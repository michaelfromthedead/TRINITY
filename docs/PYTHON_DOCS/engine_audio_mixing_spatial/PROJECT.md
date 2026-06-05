# PROJECT: Audio Mixing and Spatial Subsystems

**RDC Project Definition**
**Date**: 2026-05-23

---

## Scope

Implementation and maintenance of the TRINITY engine's audio mixing and spatial processing subsystems:

1. **Audio Mixing** (`engine/audio/mixing/`)
   - Hierarchical bus-based mixing with DSP chains
   - Ducking with envelope followers
   - Sidechain compression
   - HDR audio dynamic range management
   - Mix snapshot system for state transitions

2. **Audio Spatial** (`engine/audio/spatial/`)
   - 3D positioning with multiple source types
   - Distance attenuation curves
   - Spatialization (stereo, surround, VBAP, Ambisonics, HRTF)
   - Occlusion and propagation
   - Reverb zones with RT60 calculation
   - Acoustic material system

---

## Goals

### Primary Goals

1. **Real-Time Performance**: All audio processing must complete within frame budget for 60fps minimum
2. **Thread Safety**: Concurrent access from audio thread and game thread without data races
3. **Professional Quality**: Industry-standard algorithms (Woodworth HRTF, Sabine RT60, VBAP)
4. **Extensibility**: Factory patterns and plugin architecture for custom effects and spatializers

### Secondary Goals

1. **Memory Efficiency**: Pooled buffers, no runtime allocations in hot paths
2. **Configuration-Driven**: All parameters exposed through config.py modules
3. **Debug Support**: Comprehensive logging and profiling hooks
4. **Platform Agnosticism**: Pure Python/NumPy implementation portable across platforms

---

## Constraints

### Technical Constraints

1. **NumPy Dependency**: All audio buffer operations use NumPy for vectorization
2. **Thread Model**: Audio runs on dedicated thread; game thread schedules changes
3. **Buffer Size**: Fixed buffer sizes per tick (typically 256-1024 samples)
4. **Sample Rate**: Support for 44.1kHz and 48kHz minimum

### Design Constraints

1. **No Blocking Calls**: Audio tick cannot block waiting for game state
2. **Deterministic Processing**: Given same inputs, produce identical outputs
3. **Graceful Degradation**: Fallback paths when full processing is too expensive
4. **Lock Ordering**: Documented lock hierarchy to prevent deadlocks

### Integration Constraints

1. **DSP Chain**: Mixing depends on `../dsp/dsp_graph` for effect chains
2. **Filters**: Mixing depends on `../dsp/filters` for LP/HP filtering
3. **Raycast**: Spatial occlusion requires geometry raycast callback
4. **Listener Sync**: Spatial requires frame-accurate listener position updates

---

## Dependencies

### Internal Dependencies

| Dependency | Used By | Purpose |
|------------|---------|---------|
| `engine/audio/dsp/dsp_graph` | Mixing | Effect chain processing |
| `engine/audio/dsp/filters` | Mixing | Low-pass, high-pass filters |
| `engine/audio/core/config` | Both | Shared audio constants |
| Geometry system | Spatial | Raycast for occlusion |

### External Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| NumPy | >= 1.20 | Audio buffer operations |
| Python threading | stdlib | RLock for thread safety |

---

## Non-Goals

1. **Audio Decoding**: File format decoding handled by core audio, not mixing/spatial
2. **Platform Audio API**: OS audio output handled by platform layer
3. **Asset Pipeline**: Audio asset processing handled by tooling
4. **Network Sync**: Networked audio synchronization handled by networking layer

---

## Success Criteria

1. All 20+ algorithms implemented and tested
2. Zero stub or placeholder implementations
3. Thread-safe operation under concurrent load
4. Mathematical correctness validated against published formulas
5. Factory functions for all major component types
6. Complete configuration exposure via config.py modules
