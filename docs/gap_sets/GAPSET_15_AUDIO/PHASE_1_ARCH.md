# Phase 1 Architecture: Audio Device Abstraction and Core Types

## Purpose
Low-level audio I/O abstraction layer providing platform-independent device access, buffer management, and command dispatch between game thread and audio thread.

## Current Implementation
All core types exist as Python dataclasses/enums in `core/config.py`:
- `AudioFormat` enum (PCM_INT16/INT24/FLOAT32, ADPCM, VORBIS, OPUS, MP3, AAC)
- `ChannelLayout` enum (MONO, STEREO, SURROUND_5_1, SURROUND_7_1)
- `AudioCategory` enum (MASTER, SFX, MUSIC, VOICE_OVER, AMBIENT, UI)
- `VoiceState` enum (STOPPED, PLAYING, PAUSED, STOPPING, VIRTUAL)
- `SourceType` enum (ONE_SHOT, LOOPING, STREAMING)
- Audio buffer sizes, sample rates, memory budgets, voice limits

## Missing (6 tasks)
| Task | Component | Priority |
|------|-----------|----------|
| T-AU-1.5 | WASAPI backend | High |
| T-AU-1.6 | Core Audio backend | High |
| T-AU-1.7 | ALSA backend | High |
| T-AU-1.8 | PulseAudio backend | Medium |
| T-AU-1.10 | SPSC ring buffer | High |
| T-AU-1.11 | MPSC ring buffer | High |

## Partially Complete (1 task)
| Task | Status | Gap |
|------|--------|-----|
| T-AU-1.9 | AudioOutput sink | Engine exists, no lock-free fill slot; uses `queue.Queue` |

## Architecture Decision
The key decision is whether to implement backends natively (Rust/C++) or use an intermediate library (portaudio, SDL, sounddevice). The current Python code would need either (a) a C extension module wrapping platform APIs, or (b) bindings to an existing audio library.
