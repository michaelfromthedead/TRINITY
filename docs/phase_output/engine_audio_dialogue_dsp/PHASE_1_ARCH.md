# PHASE 1 ARCH: Dialogue Subsystem

**Architecture Specification**
**Phase:** 1 of 2
**Scope:** engine/audio/dialogue/

---

## 1. Phase Overview

Phase 1 covers the **Dialogue Subsystem** - the complete voice-over and conversation management system. This phase is architecturally independent from Phase 2 (DSP) but may call DSP processors for spatial audio effects.

**Total Implementation:** 5,433 lines across 11 files

---

## 2. Module Architecture

```
engine/audio/dialogue/
+-- __init__.py (258)         # Public API exports
+-- config.py (235)           # Configuration constants
+-- vo_line.py (341)          # VO line data structure
+-- vo_queue.py (573)         # Priority queue scheduler
+-- vo_streaming.py (707)     # LRU cache and streaming
+-- dialogue_manager.py (710) # Central orchestrator
+-- conversation.py (725)     # Branching dialogue trees
+-- vo_processing.py (728)    # Audio effects for VO
+-- contextual_dialogue.py (774) # Barks and ambient VO
+-- subtitle_sync.py (636)    # Subtitle timing and display
+-- localization.py (580)     # Multi-language support
```

---

## 3. Component Specifications

### 3.1 VOLine (vo_line.py)

**Purpose:** Immutable representation of a voice-over line.

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| audio_asset | str | Path to audio file |
| text | str | Spoken text (for subtitles) |
| speaker | str | Character ID |
| duration | float | Playback duration in seconds |
| priority | int | Queue priority (higher = more important) |
| conditions | dict | Game state conditions for playback |
| lip_sync_data | LipSyncData | Viseme timing data |
| subtitle_data | SubtitleData | Subtitle cue points |

**States:**
- PENDING: Queued, not yet playing
- PLAYING: Currently playing
- COMPLETED: Finished normally
- INTERRUPTED: Preempted by higher priority

### 3.2 VOQueue (vo_queue.py)

**Purpose:** Priority-based scheduling of VO lines.

**Data Structure:** Min-heap with composite sort key `(neg_priority, time)`

**Capacity:** Configurable max simultaneous VO (default: 2)

**Operations:**
| Operation | Complexity | Description |
|-----------|------------|-------------|
| enqueue | O(log n) | Add line to queue |
| dequeue | O(log n) | Get next line to play |
| interrupt_for | O(n) | Interrupt lower-priority active lines |
| expire_old | O(n) | Remove timed-out entries |

**Thread Safety:** RLock for all operations

### 3.3 VOStreamManager (vo_streaming.py)

**Purpose:** Audio caching and streaming with memory management.

**Cache Strategy:** LRU (Least Recently Used)

**Configuration:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| max_size_bytes | 64 MB | Maximum cache size |
| eviction_threshold | 0.9 | Evict when 90% full |
| preload_ahead | 3 | Number of lines to preload |

**Stream States:**
- IDLE: No active stream
- BUFFERING: Loading data
- STREAMING: Playing from cache
- COMPLETE: Playback finished
- ERROR: Load or playback failed

### 3.4 ConversationManager (conversation.py)

**Purpose:** Branching dialogue tree execution.

**State Machine:**
```
INACTIVE -> STARTING -> ACTIVE <-> WAITING -> COMPLETED
                          |
                          v
                       PAUSED
```

**Node Types:**
- LinearNode: Single next node
- BranchNode: Multiple choices for player
- ConditionalNode: Auto-select based on game state

**Features:**
- Multiple simultaneous conversations (max: 4)
- Participant tracking (who is talking)
- on_enter/on_exit callbacks per node
- Auto-advance option for non-interactive sequences

### 3.5 ContextualDialogue (contextual_dialogue.py)

**Purpose:** Event-triggered short VO (barks, ambient).

**Subsystems:**

#### BarkSystem
- Trigger types: reload, enemy_spotted, low_health, etc.
- Selection modes: random, sequential, weighted, shuffle, conditional
- Per-bark, per-speaker, per-category cooldowns

#### AmbientVOSystem
- Zone-based trigger regions
- Interval randomization (min/max delay)
- Weather/time-of-day conditions

### 3.6 LocalizationManager (localization.py)

**Purpose:** Multi-language audio support.

**Languages:** en, es, fr, de, it, ja, ko, zh, pt, ru

**Fallback Chain Example:**
```
es-MX -> es -> en
zh-TW -> zh -> en
```

**AudioBank:**
- Collections of localized assets by category
- Hot-swap on language change
- Unload unused languages to save memory

### 3.7 SubtitleManager (subtitle_sync.py)

**Purpose:** Synchronized subtitle display.

**Features:**
- Timed cue points aligned to audio
- Fade-in/fade-out animations
- Multi-speaker support (color coding)
- Reading speed calculation for timing

**Styling:**
| Property | Options |
|----------|---------|
| Position | top, center, bottom |
| Font | configurable |
| Color | per-speaker |
| Background | solid, transparent, blur |
| Animation | fade, slide, pop |

### 3.8 VOProcessor (vo_processing.py)

**Purpose:** Audio effects specific to dialogue.

**Effects:**
| Effect | Parameters | Use Case |
|--------|------------|----------|
| Radio | band 300-3400Hz, noise, distortion | Comms VO |
| Distance | distance, attenuation curve | Far speakers |
| Environment | reverb preset | Indoor/outdoor |
| Spatial | listener pos, speaker pos | 3D positioning |

**Environment Presets:**
| Preset | Room Size | Damping | Decay |
|--------|-----------|---------|-------|
| outdoor | 0.2 | 0.8 | 0.5s |
| cave | 0.9 | 0.2 | 3.0s |
| church | 0.8 | 0.3 | 4.0s |

---

## 4. Data Flow

```
Game Event (e.g., player enters area)
    |
    v
ContextualDialogue.trigger("ambient_vo", zone="forest")
    |
    v
LinePool.select(mode="weighted", conditions=game_state)
    |
    v
LocalizationManager.get_localized(line_id, language="es")
    |
    v
VOQueue.enqueue(line, priority=5)
    |
    v
VOStreamManager.preload(line.audio_asset)
    |
    v
DialogueManager.tick() -> dequeue and play
    |
    v
VOProcessor.apply_environment("outdoor")
    |
    v
SubtitleManager.show(line.subtitle_data)
    |
    v
Audio Output
```

---

## 5. Thread Safety Model

| Component | Lock Type | Contention Level |
|-----------|-----------|------------------|
| VOQueue | RLock | Medium (game thread + audio thread) |
| VOStreamManager | RLock | Low (preload is async) |
| CooldownTracker | RLock | Low (writes infrequent) |
| ConversationManager | RLock | Low (single conversation per context) |

---

## 6. Dependencies

### Internal
| Dependency | Usage |
|------------|-------|
| engine/audio/core/audio_source.py | Audio playback |
| engine/audio/dsp/filters.py | Radio effect bandpass |
| engine/audio/dsp/reverb.py | Environment reverb |

### External
| Dependency | Usage |
|------------|-------|
| threading | RLock synchronization |
| dataclasses | Data structures |
| heapq | Priority queue |
| uuid | Line identifiers |
