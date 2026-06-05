# PHASE 1 ARCH: Audio Mixing Core

**RDC Phase Architecture**
**Phase**: Foundation Mixing Infrastructure

---

## Phase Overview

Establish the core mixing infrastructure including the bus hierarchy, 8-stage tick pipeline, and basic signal routing.

---

## Components

### 1.1 MixBus

**Purpose**: Hierarchical signal routing node with volume, filters, and DSP chain.

**State**:
- `name: str` - Unique bus identifier
- `volume_db: float` - Volume in decibels (-inf to 0)
- `muted: bool` - Bypass all audio
- `soloed: bool` - Solo mode flag
- `low_pass_enabled: bool` - Low-pass filter active
- `low_pass_freq: float` - Cutoff frequency
- `high_pass_enabled: bool` - High-pass filter active
- `high_pass_freq: float` - Cutoff frequency
- `parent: Optional[MixBus]` - Parent in hierarchy
- `children: List[MixBus]` - Child buses

**Operations**:
- `add_child(bus)` - Add child bus
- `remove_child(bus)` - Remove child bus
- `set_volume_db(db)` - Set volume
- `read_acc_buffer(samples)` - Read accumulated input
- `process_audio(samples)` - Apply volume/filters/DSP
- `write_output(data)` - Write to accumulation buffer

### 1.2 Mixer

**Purpose**: Central coordinator running the 8-stage tick pipeline.

**State**:
- `buses: Dict[str, MixBus]` - All registered buses
- `master: MixBus` - Root of hierarchy
- `_lock: RLock` - Thread synchronization

**8-Stage Pipeline**:
```
Stage 1: Compute DFS post-order (leaf-to-root)
Stage 2: Clear accumulation buffers
Stage 2b: Generate source impulses
Stage 3: Bottom-up bus processing
Stage 4: PRE_FADER aux sends
Stage 5: Process bus (volume + effects + filters)
Stage 5b: POST_FADER aux sends
Stage 6: Ducking adjustments
Stage 7: HDR + sidechain
Stage 8: Hard clip [-1.0, 1.0]
```

### 1.3 BusRouter

**Purpose**: Manage aux sends and direct outputs.

**Send Types**:
- `PRE_FADER` - Tap raw audio before volume
- `POST_FADER` - Tap processed audio after volume

**Operations**:
- `create_send(from_bus, to_bus, type, gain)` - Create aux send
- `remove_send(send_id)` - Remove aux send
- `set_send_gain(send_id, gain)` - Adjust send level
- `get_sends_for_bus(bus)` - Query sends from/to bus

---

## Data Flow

```
Sources -> Child Buses -> Parent Buses -> Master -> Output
              |                |
              v                v
         PRE_FADER         POST_FADER
           Sends             Sends
              |                |
              +-----> Aux Buses (reverb, etc.)
```

---

## Thread Safety

1. `Mixer._lock` protects bus hierarchy mutations
2. `MixBus._acc_lock` protects per-bus accumulation buffers
3. Lock ordering: always acquire `Mixer._lock` before `MixBus._acc_lock`

---

## Integration Points

- **DSP Chain**: `MixBus` holds reference to effect chain from `../dsp/dsp_graph`
- **Filters**: `MixBus` uses `LowPassFilter`, `HighPassFilter` from `../dsp/filters`
- **Sources**: `Mixer.tick()` receives impulses from audio engine

---

## Success Criteria

1. Bus hierarchy traversal in correct DFS post-order
2. PRE_FADER sends tap raw signal
3. POST_FADER sends tap processed signal
4. Thread-safe concurrent access from audio and game threads
5. No audio glitches from lock contention
