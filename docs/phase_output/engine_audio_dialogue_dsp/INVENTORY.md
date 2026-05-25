# INVENTORY: engine_audio_dialogue_dsp

**RDC Workflow Inventory**
**Generated:** 2026-05-23
**Subsystem:** Audio Dialogue and DSP Processing

---

## Source Documents (Temporal Order)

| Order | Document | Path | Investigation Date | Lines |
|-------|----------|------|-------------------|-------|
| 1 | engine_audio_dialogue.md | docs/investigation/engine_audio_dialogue.md | 2026-05-22 | 127 |
| 2 | engine_audio_dsp.md | docs/investigation/engine_audio_dsp.md | 2026-05-22 | 174 |
| 3 | engine_audio_dialogue_dsp.md | docs/investigation/engine_audio_dialogue_dsp.md | 2026-05-22 | 240 |

---

## Temporal Ordering Rationale

All three documents were produced during the same archaeological investigation session (2026-05-22). The ordering is based on document structure:

1. **engine_audio_dialogue.md** - Focused investigation of dialogue subsystem only (11 files, 6,267 lines)
2. **engine_audio_dsp.md** - Focused investigation of DSP subsystem only (11 files, ~7,920 lines)
3. **engine_audio_dialogue_dsp.md** - Consolidated synthesis of both subsystems (12,194 lines combined)

The consolidated document (order 3) was produced after the individual investigations and serves as the synthesis document.

---

## Source Code Coverage

### engine/audio/dialogue/ (11 files, 6,267 lines)
- `__init__.py` (258 lines)
- `config.py` (235 lines)
- `vo_line.py` (341 lines)
- `vo_queue.py` (573 lines)
- `localization.py` (580 lines)
- `subtitle_sync.py` (636 lines)
- `vo_streaming.py` (707 lines)
- `dialogue_manager.py` (710 lines)
- `conversation.py` (725 lines)
- `vo_processing.py` (728 lines)
- `contextual_dialogue.py` (774 lines)

### engine/audio/dsp/ (11 files, ~7,920 lines)
- `__init__.py` (249 lines)
- `config.py` (~450 lines)
- `dsp_node.py` (493 lines)
- `distortion.py` (455 lines)
- `pitch_time.py` (580 lines)
- `special_fx.py` (741 lines)
- `dsp_graph.py` (~800 lines)
- `reverb.py` (856 lines)
- `time_effects.py` (972 lines)
- `filters.py` (973 lines)
- `dynamics.py` (1,351 lines)

---

## Classification Status

All source documents classify the audio subsystems as **REAL IMPLEMENTATION** - not stubs, not scaffolding.
