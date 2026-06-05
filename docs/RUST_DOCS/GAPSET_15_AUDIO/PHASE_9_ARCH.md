# Phase 9 Architecture: Dialogue System

## Purpose
Complete voice-over dialogue pipeline: priority queue, streaming, processing (radio/distance/reverb/spatial), conversation branching, contextual line selection (barks/ambient), localization, subtitles, and lip sync.

## Current Implementation
**8/11 tasks complete (~90% of the system).**

### Core Data (`dialogue/vo_line.py`) [x]
- `VOLine`: line_id, audio_asset, text, speaker_id, priority, interruptible, context_type, tags, conditions, weight, cooldown, callbacks
- `LipSyncData`: phonemes [(time, phoneme)], visemes [(time, viseme_id)], blend_shapes {name: [(time, value)]}
- `SubtitleData`: text, speaker_name/color, start/end_time, screen position
- `VOLineState`: PENDING->LOADING->READY->PLAYING->COMPLETED/INTERRUPTED/FAILED
- `create_vo_line()` factory function

### VO Queue (`dialogue/vo_queue.py`) [x]
- `VOQueue`: heapq priority queue, sort_key=(-priority, enqueue_time)
- `enqueue(line, timeout_ms, force)`: configurable max_size, timeout-based auto-dequeue
- `interrupt_for(incoming_priority)`: returns interrupted lines that can be resumed
- `start_line(line)`: checks max_simultaneous, prevents duplicate same-speaker
- `end_line(line, interrupted)`: cleanup, callback dispatch
- `remove_by_speaker()`, `remove_by_tag()`, `remove_below_priority()`
- `get_ducking_level()`: overlap ducking at -6dB
- `VOQueueManager`: named queues with independent config, pause/resume/clear

### VO Streaming (`dialogue/vo_streaming.py`) [x]
- `VOCache`: LRU with max_size_mb (default 32), eviction threshold, hit/miss tracking
- `StreamHandle`: state machine (IDLE->LOADING->BUFFERING->READY->STREAMING->PAUSED->COMPLETED->ERROR)
- `VOStreamManager`: streaming start/stop, preload queue, anticipated line preloading
- Memory management: `trim_cache()`, `clear_cache()`, memory usage stats

### VO Processing (`dialogue/vo_processing.py`) [x]
- `RadioEffect`: band-pass (300-3400Hz) + distortion + noise/crackle/static, configurable poor/normal/good quality
- `DistanceFilter`: linear/log/exponential attenuation, configurable start_distance/max_distance
- `ReverbSettings`: 8 environment presets (ROOM/HALL/CAVE/ARENA/OUTDOORS/UNDERWATER/CUSTOM/NONE)
- `SpatialSettings`: position, min/max_distance, spread, doppler, `calculate_pan()`
- `VOProcessor`: per-source processing state, `get_processed_params()` for mixer integration
- Factory: `create_radio_preset()`, `create_telephone_preset()`, `create_megaphone_preset()`

### Conversations (`dialogue/conversation.py`) [x]
- `ConversationNode`: line, next_nodes, branch_options, conditions, enter/exit callbacks
- `Conversation`: node graph, state machine (INACTIVE->STARTING->ACTIVE->WAITING->COMPLETED)
- `ConversationManager`: multiple conversations, priority-based interruption, gap timing, skip, choices
- Helpers: `create_linear_conversation()`, `create_branching_conversation()`

### Contextual Dialogue (`dialogue/contextual_dialogue.py`) [x]
- `CooldownTracker`: per-line/speaker/category cooldowns, remaining time query
- `LinePool`: 5 selection modes (RANDOM/SEQUENTIAL/WEIGHTED/SHUFFLE/CONDITIONAL)
- `ContextualDialogueManager`: pool registry, selection, play recording, game state binding
- `BarkSystem`: bark pools per type, speaker filtering, enable/disable
- `AmbientVOSystem`: timer-based triggering (min/max interval), zone entry/exit
- Helper: `create_bark_lines()`

### Localization (`dialogue/localization.py`) [x]
- 10 supported languages: en, es, fr, de, it, pt, ru, ja, ko, zh
- `LocalizedAsset`: per-language path/duration/subtitle/lip_sync variants
- `AudioBank`: per-language asset collection with load/unload
- `LocalizationManager`: `set_language()`, fallback chain, `localize_line()`, `switch_language_banks()`
- Helpers: `create_localized_asset()`, `create_audio_bank()`

### Subtitles (`dialogue/subtitle_sync.py`) [x]
- `SubtitleStyle`: font family/size/weight, text/background/outline/shadow colors, opacity
- `ActiveSubtitle`: state (HIDDEN->FADING_IN->VISIBLE->FADING_OUT), opacity animation
- `SubtitleTrack`: timed cue points, sorted by time, `get_cue_at_time()`
- `SubtitleManager`: max_lines (default 3), display duration calculation, `sync_with_playback()`
- Priority-ordered display, speaker styles, fade in/out

### Dialogue Manager (`dialogue/dialogue_manager.py`) [x]
- `DialogueManager`: central orchestrator
- Integrates: VOQueue + VOStreamManager + VOProcessor + SubtitleManager + LocalizationManager + ConversationManager + BarkSystem + AmbientVOSystem
- `play_line(line, position, radio_effect, force)`: full pipeline
- `start_conversation(conversation_id)`, `advance_conversation()`, `make_conversation_choice()`
- `trigger_bark(bark_type, speaker_id)`: contextual bark
- `register_ambient_zone()`, `enter_ambient_zone()`, `exit_ambient_zone()`
- `update(delta_ms)`: per-frame tick, drives conversations + ambient VO + subtitles

### Missing (2 partial, 1 missing)
| Task | Gap |
|------|-----|
| T-AU-9.4 [~] | VOPriority.CRITICAL=100 defined but not wired to voice pool reservation |
| T-AU-9.7 [~] | AudioBank/LocalizedAsset dataclasses exist, CSV/JSON manifest parsing is stub |
| T-AU-9.9 [-] | LipSyncData model complete, animation blend shape driver not implemented |
