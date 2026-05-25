# GAPSET_15_AUDIO — Task Checklist

> **Task ID Format**: `T-AU-{PHASE}.{N}`
> **Status**: 0/134 tasks complete
> **Legend**: `[ ]` = Not started, `[x]` = Complete

---

## Phase 1: Audio Device Abstraction and Core Types (12 tasks)

### Core Type Definitions

- [ ] **T-AU-1.1** — Define `SampleFormat` enum (F32, S16, S32, U8) with sample size and conversion methods
  - **Acceptance**: Enum with byte_size(), is_float(), convert_to_f32() methods; 10+ test cases covering all conversion pairs
  - **Deps**: Foundation math.rs
  - **Effort**: 1 day

- [ ] **T-AU-1.2** — Define `AudioBuffer` type (interleaved and planar variants, channel count, frame count, format)
  - **Acceptance**: Buffer supports interleaved<->planar conversion, channel extraction, sub-buffer view; 15+ test cases
  - **Deps**: T-AU-1.1
  - **Effort**: 2 days

- [ ] **T-AU-1.3** — Define `AudioConfig` struct (sample_rate, channels, format, period_size, device_name)
  - **Acceptance**: Config validates sample rate (8k-192k), channels (1-8), period size (32-4096, power of 2); 8+ test cases
  - **Deps**: T-AU-1.1
  - **Effort**: 0.5 day

- [ ] **T-AU-1.4** — Define `AudioDevice` trait/interface with open, start, stop, close, buffer_size, sample_rate
  - **Acceptance**: Trait defined in Python with Rust implementation stub; all methods documented; 5+ mock test cases
  - **Deps**: T-AU-1.3
  - **Effort**: 1 day

### Platform Implementations

- [ ] **T-AU-1.5** — Implement WASAPI audio device backend (Windows)
  - **Acceptance**: Opens default output device, starts/stops callback, handles device change notification, recovers from XRUN; tested on Windows 10/11
  - **Deps**: T-AU-1.4, Foundation memory.rs (lock-free ring buffer)
  - **Effort**: 5 days

- [ ] **T-AU-1.6** — Implement Core Audio device backend (macOS/iOS)
  - **Acceptance**: Opens default output via AudioUnit, starts/stops render callback, handles format negotiation; tested on macOS 14+
  - **Deps**: T-AU-1.4
  - **Effort**: 4 days

- [ ] **T-AU-1.7** — Implement ALSA device backend (Linux)
  - **Acceptance**: Opens default PCM device, starts/stops callback in RW mode, handles period wakeup, XRUN recovery; tested on ALSA 1.2+
  - **Deps**: T-AU-1.4
  - **Effort**: 4 days

- [ ] **T-AU-1.8** — Implement PulseAudio device backend (Linux fallback)
  - **Acceptance**: Opens stream via simple API or async API, starts/stops, handles stream underflow; tested on PulseAudio 16+
  - **Deps**: T-AU-1.4
  - **Effort**: 2 days

### Audio Output Infrastructure

- [ ] **T-AU-1.9** — Implement `AudioOutput` sink: owns device, provides lock-free fill slot, pulls from ring buffer
  - **Acceptance**: AudioOutput renders periods without underrun at 256 frames/48kHz; 10+ test cases for fill/pull/overrun
  - **Deps**: T-AU-1.5 through T-AU-1.8 (at least one platform backend)
  - **Effort**: 2 days

- [ ] **T-AU-1.10** — Build lock-free ring buffer (single-producer, single-consumer) for game-to-audio command queue
  - **Acceptance**: SPSC queue with fixed capacity (default 256 commands); no allocations in push/pop; 20+ concurrent test cases
  - **Deps**: Foundation memory.rs
  - **Effort**: 2 days

- [ ] **T-AU-1.11** — Build lock-free ring buffer for stream-to-audio PCM delivery
  - **Acceptance**: MPSC queue (multiple decode threads push, audio thread pops); block-free on consumer; 20+ concurrent test cases
  - **Deps**: Foundation memory.rs
  - **Effort**: 2 days

- [ ] **T-AU-1.12** — Integrate `@sound` decorator with audio type system (ComponentMeta, Registry registration)
  - **Acceptance**: `@sound(bank="test")` on a Component class registers it with Registry, TAG(sound=True); 5+ test cases
  - **Deps**: T-AU-1.1 through T-AU-1.3, Foundation ComponentMeta, Foundation Registry
  - **Effort**: 1 day

---

## Phase 2: Mixer Graph and Voice Management (18 tasks)

### Mix Bus Implementation

- [ ] **T-AU-2.1** — Implement `MixBus` resource: volume, pitch, mute, solo, low-pass, high-pass, effects chain
  - **Acceptance**: MixBus has all properties tracked; gain computation from volume + mute + solo; per-bus LPF/HPF with configurable cutoff; 15+ test cases
  - **Deps**: T-AU-1.2 (AudioBuffer), Foundation ResourceMeta
  - **Effort**: 3 days

- [ ] **T-AU-2.2** — Implement master bus hierarchy: Master -> SFX/Music/Dialogue/Ambient with submix routing
  - **Acceptance**: Tree of MixBus nodes; each node accumulates children into its output; master bus sends to AudioOutput; 10+ test cases for routing correctness
  - **Deps**: T-AU-2.1
  - **Effort**: 2 days

- [ ] **T-AU-2.3** — Wire `@audio_bus` decorator to MixBus resource, register via ResourceMeta
  - **Acceptance**: `@audio_bus(name="sfx", volume=0.9)` creates a MixBus registered in Foundation Registry; test by enumerating buses at startup
  - **Deps**: T-AU-2.1, Foundation ResourceMeta, Foundation Registry
  - **Effort**: 1 day

- [ ] **T-AU-2.4** — Implement bus gain computation (pre-computed per-period gain per bus node)
  - **Acceptance**: Gain computed from volume * mute_flag + parent gain; pre-computed per-period before mixing loop; 8+ test cases
  - **Deps**: T-AU-2.1
  - **Effort**: 1 day

- [ ] **T-AU-2.5** — Implement mix snapshot system (`@audio_snapshot`): named presets, crossfade blending
  - **Acceptance**: MixSnapshot stores bus volume overrides; apply/recall creates smooth crossfade over crossfade_time; 8+ test cases
  - **Deps**: T-AU-2.2, Foundation Serilizer
  - **Effort**: 2 days

- [ ] **T-AU-2.6** — Implement HDR audio: virtual dB scale, audible window, dynamic shifting
  - **Acceptance**: Loud events (above threshold) push window up; quiet moments let window drop; configurable window size and shift speed; 10+ test cases
  - **Deps**: T-AU-2.2
  - **Effort**: 3 days

### Voice Management

- [ ] **T-AU-2.7** — Implement voice pool: fixed-size allocation, O(1) acquire/release
  - **Acceptance**: Pool allocates N voices at startup; acquire returns free voice, release returns to pool; O(1) amortized; 10+ test cases including exhaustion
  - **Deps**: Foundation memory.rs
  - **Effort**: 1 day

- [ ] **T-AU-2.8** — Implement priority-based voice stealing (lowest priority first, tiebreak by age)
  - **Acceptance**: When pool exhausted, voice with lowest priority stolen; same priority steals oldest; O(log V) or better; 12+ test cases
  - **Deps**: T-AU-2.7
  - **Effort**: 2 days

- [ ] **T-AU-2.9** — Implement per-category and per-sound voice limits
  - **Acceptance**: Category limits (max N footsteps) enforced; per-sound limits (max 3 gunfire from same weapon) enforced; category tracking O(1); 8+ test cases
  - **Deps**: T-AU-2.8
  - **Effort**: 1 day

- [ ] **T-AU-2.10** — Implement virtual voices: track position/time when inaudible, resume when voice available
  - **Acceptance**: Virtualized voice tracks position/time but produces no audio; when priority sufficiently high, voice becomes real; 6+ test cases
  - **Deps**: T-AU-2.8
  - **Effort**: 2 days

- [ ] **T-AU-2.11** — Wire `@voice_priority` decorator to voice management
  - **Acceptance**: `@voice_priority(priority=5, virtualize=True)` on a Component sets its voice parameters; registered in Registry; 4+ test cases
  - **Deps**: T-AU-2.8, Foundation ComponentMeta
  - **Effort**: 1 day

### Sidechain

- [ ] **T-AU-2.12** — Implement sidechain compression: envelope follower on source bus, gain reduction on target bus
  - **Acceptance**: SidechainCompressor links two buses; source bus envelope drives gain reduction on target bus; attack/release/ratio configurable; 10+ test cases
  - **Deps**: T-AU-2.2
  - **Effort**: 3 days

- [ ] **T-AU-2.13** — Wire `@sidechain` decorator to sidechain compressor
  - **Acceptance**: `@sidechain(source_bus="sfx", attack=0.01, release=0.1, ratio=4.0)` on a Resource configures sidechain; test by verifying envelope follower output
  - **Deps**: T-AU-2.12, Foundation ResourceMeta
  - **Effort**: 1 day

### Mixer Integration

- [ ] **T-AU-2.14** — Implement mixer tick: accumulate voices into buses, process bus effects, output to AudioOutput
  - **Acceptance**: Mixer tick processes all active voices into their assigned buses; per-bus effects applied; final mix sent to AudioOutput; no missed periods; 15+ test cases
  - **Deps**: T-AU-2.2, T-AU-2.7, T-AU-1.9
  - **Effort**: 3 days

- [ ] **T-AU-2.15** — Implement voice-to-bus routing (each voice assigned to one bus at creation)
  - **Acceptance**: Voice created with bus_id; voice output added to bus accumulator during mixer tick; 5+ test cases
  - **Deps**: T-AU-2.14
  - **Effort**: 1 day

- [ ] **T-AU-2.16** — Implement per-bus effect chain (ordered list of DSP nodes)
  - **Acceptance**: Bus has ordered list of DSP node references; each node processes bus output during mixer tick; chain bypass supported; 8+ test cases
  - **Deps**: T-AU-2.14, Phase 7 DSP nodes (stubs acceptable for Phase 2)
  - **Effort**: 2 days

- [ ] **T-AU-2.17** — Implement `@audio_bus` effects parameter (link to bus effect chain)
  - **Acceptance**: `@audio_bus(name="sfx", effects=["compressor"])` adds DSP node references to bus
  - **Deps**: T-AU-2.16
  - **Effort**: 0.5 day

- [ ] **T-AU-2.18** — Implement mix bus configuration persistence via Session (volumes, mute/solo, snapshots)
  - **Acceptance**: Bus volumes, mute/solo, and active snapshot saved and restored across sessions; 6+ test cases
  - **Deps**: T-AU-2.5, Foundation Session, Foundation Serilizer
  - **Effort**: 1 day

---

## Phase 3: Sound Playback Engine (14 tasks)

### AudioSource Component

- [ ] **T-AU-3.1** — Implement `AudioSource` component: volume, pitch, playing, sound bank reference
  - **Acceptance**: Component with tracked volume (0-1), pitch (0.1-4.0), playing (bool), bank (str); uses TrackedDescriptor, RangeDescriptor; registered via ComponentMeta; 8+ test cases
  - **Deps**: Foundation ComponentMeta, Foundation Tracker, T-AU-1.12
  - **Effort**: 1 day

- [ ] **T-AU-3.2** — Wire `@sound(bank, preload)` decorator to AudioSource configuration
  - **Acceptance**: Decorated class gets sound_bank and sound_preload tags; Registry enumerates all sound components; 5+ test cases
  - **Deps**: T-AU-3.1, Foundation Registry
  - **Effort**: 1 day

- [ ] **T-AU-3.3** — Implement `@tracked` / TrackedDescriptor integration for AudioSource volume/pitch/playing
  - **Acceptance**: Writing to volume/pitch/playing marks component dirty; AudioUpdateSystem reads dirty fields; Tracker integration verified; 8+ test cases
  - **Deps**: T-AU-3.1, Foundation Tracker
  - **Effort**: 1 day

### AudioClip Asset

- [ ] **T-AU-3.4** — Implement `AudioClip` asset type: sample_rate, channels, duration, compressed flag, PCM data
  - **Acceptance**: Asset with @asset(extensions=[".wav", ".ogg", ".mp3"]) loads format-agnostic PCM; metadata fields populated; 10+ test cases
  - **Deps**: Foundation AssetMeta, T-AU-1.2
  - **Effort**: 2 days

- [ ] **T-AU-3.5** — Implement WAV format loader (uncompressed PCM, read header, extract data)
  - **Acceptance**: Loads 16-bit and 24-bit WAV files; converts to f32 planar; handles mono/stereo; 8+ test cases with reference files
  - **Deps**: T-AU-3.4
  - **Effort**: 1 day

- [ ] **T-AU-3.6** — Implement OGG/Vorbis format loader (libvorbis or pure-Rust decoder)
  - **Acceptance**: Loads OGG files of any bitrate; outputs f32 planar; handles multi-channel; 6+ test cases with reference files
  - **Deps**: T-AU-3.4
  - **Effort**: 3 days

- [ ] **T-AU-3.7** — Implement FLAC format loader (libFLAC or pure-Rust decoder)
  - **Acceptance**: Loads FLAC files at any compression level; outputs f32 planar; 4+ test cases with reference files
  - **Deps**: T-AU-3.4
  - **Effort**: 2 days

- [ ] **T-AU-3.8** — Implement MP3 format loader (libmpg123 or minimp3)
  - **Acceptance**: Loads CBR and VBR MP3 files; outputs f32 planar; 4+ test cases with reference files
  - **Deps**: T-AU-3.4
  - **Effort**: 2 days

- [ ] **T-AU-3.9** — Implement Opus format loader (libopusfile)
  - **Acceptance**: Loads Opus files in Ogg container; outputs f32 planar; 4+ test cases with reference files
  - **Deps**: T-AU-3.4
  - **Effort**: 2 days

### Playback Command Path

- [ ] **T-AU-3.10** — Implement lock-free command queue for game-to-audio thread: Play, Stop, SetVolume, SetPitch
  - **Acceptance**: Commands serialized to queue; audio thread processes one batch per tick; no allocations in push/pop; 12+ concurrent test cases
  - **Deps**: T-AU-1.10
  - **Effort**: 2 days

- [ ] **T-AU-3.11** — Implement `AudioUpdateSystem` (@system, phase="audio"): polls dirty sources, pushes commands
  - **Acceptance**: System runs each frame; collects dirty AudioSource components; pushes Play/Stop/Param commands for changed fields; 10+ test cases
  - **Deps**: T-AU-3.3, T-AU-3.10, Foundation SystemMeta
  - **Effort**: 2 days

### Sound Cues

- [ ] **T-AU-3.12** — Implement sound cues: Simple, Random (weighted, no-repeat), Sequence (ordered), Switch (parameter-driven)
  - **Acceptance**: Each cue type resolves to an AudioClip; Random respects weights and no-repeat; Sequence supports forward/reverse/ping-pong; Switch selects by parameter; 15+ test cases
  - **Deps**: T-AU-3.4
  - **Effort**: 3 days

- [ ] **T-AU-3.13** — Implement variation system: pitch randomization, volume randomization, start offset
  - **Acceptance**: Per-play randomization applied at voice creation; pitch +/- N semitones; volume +/- N dB; offset skips N seconds; 8+ test cases
  - **Deps**: T-AU-3.10 (variation baked into Play command)
  - **Effort**: 1 day

### AudioListener

- [ ] **T-AU-3.14** — Implement `AudioListener` component: position, velocity, orientation
  - **Acceptance**: Component registered via ComponentMeta; one active listener per scene; position/velocity/orientation tracked; 6+ test cases
  - **Deps**: Foundation ComponentMeta, Foundation Tracker
  - **Effort**: 1 day

---

## Phase 4: Stream and Decode Thread Architecture (10 tasks)

### Thread Infrastructure

- [ ] **T-AU-4.1** — Implement decode thread pool with configurable worker count and format plugin interface
  - **Acceptance**: Pool spawns N workers; workers pull decode jobs from queue; format plugin trait defined; default N = max(1, num_cpus - 2); 10+ test cases
  - **Deps**: Foundation task_system.rs
  - **Effort**: 3 days

- [ ] **T-AU-4.2** — Implement decode job queue (MPSC): stream threads push, decode workers pop
  - **Acceptance**: Lock-free MPSC queue; bounded capacity; no allocations; 12+ concurrent test cases
  - **Deps**: Foundation memory.rs
  - **Effort**: 1 day

- [ ] **T-AU-4.3** — Implement stream thread: async file I/O, chunked reading, stream state machine
  - **Acceptance**: Stream thread opens files asynchronously; reads configurable chunk sizes; state machine: Idle->Opening->Reading->Decoding->Playing->Draining->Idle; 10+ test cases
  - **Deps**: T-AU-4.1, Foundation task_system.rs
  - **Effort**: 3 days

### Memory Pool Management

- [ ] **T-AU-4.4** — Implement resident memory pool: preload, LRU eviction, preload-marked never-evicted
  - **Acceptance**: Sounds with `preload=True` loaded at startup; LRU eviction on pool full; preload-marked sounds never evicted; 8+ test cases
  - **Deps**: Foundation memory.rs (pool allocator)
  - **Effort**: 2 days

- [ ] **T-AU-4.5** — Implement streaming pool: per-stream ring buffer, chunk management
  - **Acceptance**: Each active stream has a ring buffer (default 256 KB, 2 chunks); stream thread fills, audio thread consumes; underrun detection and recovery; 10+ test cases
  - **Deps**: T-AU-4.3, T-AU-1.11
  - **Effort**: 2 days

- [ ] **T-AU-4.6** — Implement temporary pool: one-shot decode buffer, allocation/recycle, timed release
  - **Acceptance**: Temporary buffers allocated for one-shot decodes; recycled when voice stops; unused buffers released after configurable timeout; 6+ test cases
  - **Deps**: Foundation memory.rs
  - **Effort**: 1 day

### Streaming Playback

- [ ] **T-AU-4.7** — Implement streaming source type: ring-buffer-fed voice, continuous playback
  - **Acceptance**: Streaming voice reads from stream pool ring buffer; triggers next chunk load when below threshold; seamless transition between chunks; 8+ test cases
  - **Deps**: T-AU-4.5, T-AU-3.10 (Play command for streaming)
  - **Effort**: 2 days

- [ ] **T-AU-4.8** — Implement decode format plugin interface: register decoder per format, format-agnostic PCM output
  - **Acceptance**: Plugin trait with decode(chunk) -> Result<AudioBuffer>; decoders registered by file extension; all produce f32 planar; 6+ test cases
  - **Deps**: T-AU-3.5 through T-AU-3.9 (format loaders as plugins)
  - **Effort**: 2 days

- [ ] **T-AU-4.9** — Wire AudioClip streaming flag to stream manager
  - **Acceptance**: AudioClip with stream=True triggers stream thread loading; clip with stream=False loads entirely into resident pool; 4+ test cases
  - **Deps**: T-AU-4.5, T-AU-3.4
  - **Effort**: 1 day

- [ ] **T-AU-4.10** — Implement audio thread tick that never blocks on I/O
  - **Acceptance**: Verify with instrumentation that audio thread makes zero file I/O calls during tick; PCM data is always pre-decoded; 24-hour stress test with no underrun
  - **Deps**: T-AU-4.1 through T-AU-4.9
  - **Effort**: 3 days (testing + verification)

---

## Phase 5: Spatial Audio System (15 tasks)

### Attenuation

- [ ] **T-AU-5.1** — Implement attenuation curves: linear, logarithmic, inverse, custom spline
  - **Acceptance**: Each curve type maps distance to gain; custom spline supports N control points; output clamped [0, 1]; 12+ test cases
  - **Deps**: T-AU-3.14 (listener position)
  - **Effort**: 2 days

- [ ] **T-AU-5.2** — Wire `@spatial_audio(falloff, max_distance)` decorator to attenuation configuration
  - **Acceptance**: Decorated component receives falloff type and max_distance; Registry queryable by spatial properties; 4+ test cases
  - **Deps**: T-AU-5.1, Foundation ComponentMeta
  - **Effort**: 1 day

### Positioning

- [ ] **T-AU-5.3** — Implement point source positioning (direction vector, distance from listener)
  - **Acceptance**: Given listener and source transforms, compute azimuth/elevation/distance; 8+ test cases
  - **Deps**: T-AU-3.14, Foundation math.rs (Vec3)
  - **Effort**: 1 day

- [ ] **T-AU-5.4** — Implement area, line, and volume source positioning (closest-point, fade-to-edge)
  - **Acceptance**: Area source fades volume from center to edges; line source uses closest-point along segment; volume source uses penetration depth; 10+ test cases
  - **Deps**: T-AU-5.3, Foundation math.rs
  - **Effort**: 3 days

### Spatialization Methods

- [ ] **T-AU-5.5** — Implement stereo panning (equal-power, sin/cos pan law)
  - **Acceptance**: Source angle mapped to L/R gain; equal-power (3 dB) pan law; mono-to-stereo and stereo-to-stereo; 8+ test cases
  - **Deps**: T-AU-5.3
  - **Effort**: 1 day

- [ ] **T-AU-5.6** — Implement dummy HRTF: ITD (interaural time delay) + frequency-dependent level (head shadow)
  - **Acceptance**: ITD computed from azimuth (Woodworth formula); head shadow LPF on far ear; functional and distinguishable from plain panning; 8+ test cases
  - **Deps**: T-AU-5.3, Foundation math.rs
  - **Effort**: 3 days

- [ ] **T-AU-5.7** — Implement measured HRTF via SOFA convolution with head-related impulse responses
  - **Acceptance**: SOFA file loaded as HRIR set; nearest-neighbor or bilinear interpolation over azimuth/elevation; partitioned convolution; tested with CIPIC or similar dataset; 10+ test cases
  - **Deps**: T-AU-5.6, T-AU-7.5 (convolution)
  - **Effort**: 5 days

- [ ] **T-AU-5.8** — Implement VBAP: speaker triplet selection, gain computation per speaker
  - **Acceptance**: Speaker layout defined at startup; active triplet selected per source direction; gains normalized (sum^2 = 1); 10+ test cases for 2.0, 5.1, 7.1
  - **Deps**: T-AU-5.3, Foundation math.rs (Vec3)
  - **Effort**: 4 days

- [ ] **T-AU-5.9** — Implement FOA ambisonics: WXYZ encoding, decoder to speaker layout
  - **Acceptance**: Source encoded to B-format (W, X, Y, Z); decoded to speaker feeds via decoder matrix; ambisonic rotation supported; 8+ test cases
  - **Deps**: T-AU-5.3, Foundation math.rs
  - **Effort**: 4 days

- [ ] **T-AU-5.10** — Implement HOA ambisonics (N=3): higher-order encoding/decoding, binaural rendering via HRTF
  - **Acceptance**: N-order ambisonic encoding (N+1)^2 channels; decoder to speaker layout; binaural decoding via HRTF convolution; 6+ test cases
  - **Deps**: T-AU-5.9, T-AU-5.7
  - **Effort**: 5 days

### Occlusion

- [ ] **T-AU-5.11** — Implement single-ray occlusion test (ray from source to listener via physics system)
  - **Acceptance**: Ray cast between source and listener positions; returns occlusion amount [0, 1]; applies frequency-band attenuation; 6+ test cases
  - **Deps**: T-AU-5.3, Physics system (ray cast)
  - **Effort**: 2 days

- [ ] **T-AU-5.12** — Implement multi-ray occlusion (N rays from source volume to listener)
  - **Acceptance**: Configurable N rays (4-32); occlusion = fraction of blocked rays; optional cone distribution; 8+ test cases
  - **Deps**: T-AU-5.11
  - **Effort**: 2 days

- [ ] **T-AU-5.13** — Wire `@occlusion(method, max_occlusion)` decorator to occlusion settings
  - **Acceptance**: Decorated component configures occlusion method and max_occlusion; 4+ test cases
  - **Deps**: T-AU-5.12, Foundation ComponentMeta
  - **Effort**: 1 day

### Spatial Update System

- [ ] **T-AU-5.14** — Implement `SpatialUpdateSystem`: update listener/source positions, compute per-source spatial parameters
  - **Acceptance**: System runs each frame; updates listener from AudioListener component; for each spatial source, computes attenuation, direction, occlusion; pushes spatial parameters to audio thread; 10+ test cases
  - **Deps**: T-AU-5.1 through T-AU-5.13, Foundation SystemMeta
  - **Effort**: 3 days

- [ ] **T-AU-5.15** — Implement spatial gain application in mixer (spatial gain multiplies voice gain before bus accumulation)
  - **Acceptance**: Each voice has spatial_gain factor; mixer tick applies spatial_gain before accumulating to bus; 6+ test cases
  - **Deps**: T-AU-5.14, T-AU-2.14
  - **Effort**: 1 day

---

## Phase 6: Acoustic Simulation (15 tasks)

### Algorithmic Reverb

- [ ] **T-AU-6.1** — Implement feedback delay network (FDN) reverb with configurable RT60, room size, damping, diffusion
  - **Acceptance**: FDN with 8-16 delay lines; RT60 per frequency band (low, mid, high); room size scales delay length; damping absorbs high frequencies per reflection; diffusion controls echo density; 12+ test cases
  - **Deps**: T-AU-1.2 (AudioBuffer)
  - **Effort**: 5 days

- [ ] **T-AU-6.2** — Implement Schroeder-Moorer reverb (comb filters + all-pass filters, simpler alternative)
  - **Acceptance**: 4-8 comb filters in parallel summed into 2-4 all-pass filters in series; configurable RT60 and pre-delay; lower CPU than FDN; 8+ test cases
  - **Deps**: T-AU-1.2
  - **Effort**: 3 days

- [ ] **T-AU-6.3** — Implement reverb zone detection: which reverb zone(s) contain source and listener
  - **Acceptance**: Zone query by world position; supports overlapping zones with blend; returns active zone list and blend weights; 10+ test cases
  - **Deps**: T-AU-5.3 (source/listener positioning), Foundation Registry (zone query)
  - **Effort**: 2 days

- [ ] **T-AU-6.4** — Wire `@reverb_zone(preset, fade_distance)` decorator to reverb zone components
  - **Acceptance**: Decorated component registers reverb zone with preset and fade_distance; 4+ test cases
  - **Deps**: T-AU-6.3, Foundation ComponentMeta
  - **Effort**: 1 day

### Convolution Reverb

- [ ] **T-AU-6.5** — Implement convolution engine: impulse response loading, partitioned convolution
  - **Acceptance**: Loads IR from WAV file; non-uniform partitioned convolution (e.g., 512/1024/2048 sample partitions); IR normalization; 8+ test cases
  - **Deps**: T-AU-1.2, Foundation math.rs (FFT)
  - **Effort**: 5 days

- [ ] **T-AU-6.6** — Implement real-time IR swap with crossfade (seamless reverb preset change)
  - **Acceptance**: Two convolution engines running; crossfade between old and new IR over configurable time (default 50 ms); no audible click or pop; 6+ test cases
  - **Deps**: T-AU-6.5
  - **Effort**: 2 days

- [ ] **T-AU-6.7** — Implement hybrid reverb: algorithmic early reflections + convolution tail
  - **Acceptance**: Early reflections (1st-3rd order) from algorithmic reverb; late decay from convolution IR; crosspoint at configurable time (default 80 ms); 8+ test cases
  - **Deps**: T-AU-6.1, T-AU-6.5
  - **Effort**: 4 days

### Propagation

- [ ] **T-AU-6.8** — Implement early reflections: 1st-3rd order geometric reflection paths
  - **Acceptance**: Image-source method for specular reflections; up to 3rd order; each reflection adds delay, attenuation, and frequency filtering based on material; 10+ test cases
  - **Deps**: T-AU-5.3, T-AU-6.10 (acoustic materials), Physics system
  - **Effort**: 5 days

- [ ] **T-AU-6.9** — Implement diffraction: Huygens-Fresnel approximation for sound bending around corners
  - **Acceptance**: Edge diffraction computed via secondary source at obstructing geometry edge; low-pass characteristic of diffracted sound (more bass passes around corners); 6+ test cases
  - **Deps**: T-AU-6.8, Physics system
  - **Effort**: 4 days

- [ ] **T-AU-6.10** — Implement transmission: through-wall filtering based on material properties
  - **Acceptance**: Wall material provides per-band transmission coefficient; transmitted sound is frequency-filtered and attenuated; 6+ test cases
  - **Deps**: T-AU-6.11 (acoustic materials), Physics system
  - **Effort**: 2 days

### Acoustic Materials

- [ ] **T-AU-6.11** — Define acoustic material presets with absorption, reflection, transmission coefficients (per frequency band)
  - **Acceptance**: Material struct with coefficients for low (125 Hz), mid (500 Hz), high (4000 Hz) bands; presets: concrete, wood, glass, carpet, curtain, brick, metal, water; interpolation between materials; 10+ test cases
  - **Deps**: None
  - **Effort**: 2 days

- [ ] **T-AU-6.12** — Wire material assignment to world geometry (physics material -> acoustic material mapping)
  - **Acceptance**: Physics material lookup maps to acoustic material; occlusion/propagation queries use material coefficients; 4+ test cases
  - **Deps**: T-AU-6.11, Physics system
  - **Effort**: 1 day

### Reverb Integration

- [ ] **T-AU-6.13** — Implement dry/wet mix for acoustic simulation (blend based on distance, occlusion, zone blend)
  - **Acceptance**: Dry signal passes directly to bus; wet signal processed through reverb + propagation; blend controlled by distance (more wet at distance), occlusion (more wet when occluded), zone blend weight; 10+ test cases
  - **Deps**: T-AU-6.3, T-AU-6.1
  - **Effort**: 3 days

- [ ] **T-AU-6.14** — Implement reverb send in mixer (voice sends to reverb bus with configurable send level)
  - **Acceptance**: Each voice has reverb send level [0, 1]; mixer accumulates sends to reverb aux bus; reverb bus feeds back into master or submix; 8+ test cases
  - **Deps**: T-AU-2.14, T-AU-6.13
  - **Effort**: 2 days

- [ ] **T-AU-6.15** — Implement baked occlusion (offline precomputed per zone-pair for static geometry)
  - **Acceptance**: Baked occlusion table (per-zone-pair, per-frequency-band); runtime lookup is O(1); fallback to real-time raycast for dynamic objects; 6+ test cases
  - **Deps**: T-AU-5.11, T-AU-6.11
  - **Effort**: 3 days

---

## Phase 7: DSP Effect Chain (20 tasks)

### DSP Graph Framework

- [ ] **T-AU-7.1** — Implement `DSPNode` interface with process(), configure(), reset()
  - **Acceptance**: Interface defined and documented; all DSP effects implement it; process takes input buffer and returns output buffer; 5+ test cases
  - **Deps**: T-AU-1.2
  - **Effort**: 1 day

- [ ] **T-AU-7.2** — Wire `@dsp_node(inputs, outputs, latency_samples)` decorator to DSP node registration
  - **Acceptance**: Decorated class registered as DSP node type; Registry enumerable; 4+ test cases
  - **Deps**: T-AU-7.1, Foundation ComponentMeta
  - **Effort**: 1 day

- [ ] **T-AU-7.3** — Implement DSP node chain execution (ordered list of nodes, each feeding next)
  - **Acceptance**: Chain of N nodes processes buffer sequentially; in-place processing where supported; chain bypass supported (passthrough); 8+ test cases
  - **Deps**: T-AU-7.1
  - **Effort**: 1 day

- [ ] **T-AU-7.4** — Implement real-time parameter modulation (LFO, envelope follower, game state binding)
  - **Acceptance**: Modulator can drive any f32 parameter; LFO (sine, triangle, saw, square, random) with configurable rate/depth; envelope follower from audio signal; game state parameter binding; 12+ test cases
  - **Deps**: T-AU-7.1
  - **Effort**: 4 days

### Filters

- [ ] **T-AU-7.5** — Implement biquad filter (Direct Form I): LPF, HPF, BPF, Notch, Shelf, Parametric EQ
  - **Acceptance**: All 6 filter types; configurable cutoff, Q, gain (for shelf/PEQ); coefficients recomputed on parameter change; 20+ test cases with reference frequency response
  - **Deps**: T-AU-7.1
  - **Effort**: 4 days

### Dynamics

- [ ] **T-AU-7.6** — Implement compressor: RMS/peak detection, feed-forward, configurable threshold/ratio/attack/release/knee
  - **Acceptance**: Compression curve with soft knee; attack 0.1-100 ms; release 10-1000 ms; ratio 1:1 to 20:1; gain reduction metering output; 15+ test cases
  - **Deps**: T-AU-7.1
  - **Effort**: 4 days

- [ ] **T-AU-7.7** — Implement limiter: hard knee, fast attack (0.01-5 ms), ceiling, lookahead
  - **Acceptance**: Brickwall limiting at configurable ceiling; lookahead (up to 10 ms); overshoot prevention; 8+ test cases
  - **Deps**: T-AU-7.6
  - **Effort**: 2 days

- [ ] **T-AU-7.8** — Implement gate/expander: downward expansion, hysteresis (open/close thresholds)
  - **Acceptance**: Gate closes below threshold with hysteresis band; expander ratio 1:1 to 1:inf; attack/release/hold controls; 10+ test cases
  - **Deps**: T-AU-7.1
  - **Effort**: 3 days

### Time-Based Effects

- [ ] **T-AU-7.9** — Implement delay: configurable delay time, feedback, mix, ping-pong mode
  - **Acceptance**: Delay time 1-2000 ms; feedback 0-0.99; ping-pong alternately L/R; tempo-sync mode (beat subdivisions); 10+ test cases
  - **Deps**: T-AU-7.1
  - **Effort**: 2 days

- [ ] **T-AU-7.10** — Implement chorus: modulated delay lines, configurable depth/rate/mix
  - **Acceptance**: 2-4 voices with slightly detuned LFO modulation; depth 0-100%; rate 0.1-5 Hz; mix dry/wet; 8+ test cases
  - **Deps**: T-AU-7.1, T-AU-7.4 (LFO modulator)
  - **Effort**: 3 days

- [ ] **T-AU-7.11** — Implement flanger: modulated short delay (<10 ms), feedback, depth/rate/resonance
  - **Acceptance**: Delay 0.1-10 ms; LFO modulates delay time; feedback creates comb filtering sweep; resonance controls feedback amount; 8+ test cases
  - **Deps**: T-AU-7.1, T-AU-7.4
  - **Effort**: 2 days

- [ ] **T-AU-7.12** — Implement phaser: all-pass filter chain, LFO modulation, feedback
  - **Acceptance**: 4-12 all-pass stages in series; LFO modulates all-pass center frequencies; feedback for resonance; 8+ test cases
  - **Deps**: T-AU-7.1, T-AU-7.4
  - **Effort**: 3 days

### Distortion

- [ ] **T-AU-7.13** — Implement hard clip: threshold-based symmetric/asymmetric clipping
  - **Acceptance**: Threshold [0, 1]; symmetric and asymmetric modes; DC offset removal; 6+ test cases
  - **Deps**: T-AU-7.1
  - **Effort**: 1 day

- [ ] **T-AU-7.14** — Implement soft clip: tanh/sigmoid/arctan waveshaping
  - **Acceptance**: Three waveshaping functions with configurable drive; smooth transition from clean to distorted; 6+ test cases
  - **Deps**: T-AU-7.1
  - **Effort**: 1 day

- [ ] **T-AU-7.15** — Implement waveshaping: arbitrary transfer function (lookup table)
  - **Acceptance**: Transfer function defined by N-point curve or mathematical expression; LUT-based processing (efficient); 6+ test cases
  - **Deps**: T-AU-7.1
  - **Effort**: 2 days

- [ ] **T-AU-7.16** — Implement bitcrush: sample rate reduction + bit depth reduction
  - **Acceptance**: Sample rate reduction factor [1, 256]; bit depth [1, 24]; configurable order (rate then depth or depth then rate); 6+ test cases
  - **Deps**: T-AU-7.1
  - **Effort**: 1 day

- [ ] **T-AU-7.17** — Implement foldback distortion: phase-inverted folding at threshold
  - **Acceptance**: Signal folds back when exceeding threshold; multiple fold regions; aliasing reduction (oversample + filter); 6+ test cases
  - **Deps**: T-AU-7.1
  - **Effort**: 2 days

### Pitch/Time Effects

- [ ] **T-AU-7.18** — Implement simple pitch shift: sample-rate offset with linear interpolation
  - **Acceptance**: Pitch ratio [0.5, 2.0]; linear interpolation between samples; with/without formant preservation flag; 8+ test cases
  - **Deps**: T-AU-7.1
  - **Effort**: 1 day

- [ ] **T-AU-7.19** — Implement granular pitch shift: overlapping grain envelope, configurable grain size
  - **Acceptance**: Grain size 20-100 ms; overlap 2-8 grains; window function (Hann, Blackman); pitch ratio [0.5, 2.0] independent of time stretch; 10+ test cases
  - **Deps**: T-AU-7.1
  - **Effort**: 4 days

- [ ] **T-AU-7.20** — Implement phase vocoder: FFT-based pitch shift with formant preservation
  - **Acceptance**: FFT size 1024-4096; hop size 25-75%; phase locking for transients; formant preservation via spectral envelope scaling; 8+ test cases
  - **Deps**: T-AU-7.1, Foundation math.rs (FFT)
  - **Effort**: 5 days

---

## Phase 8: Adaptive Music Engine (14 tasks)

### Music System Core

- [ ] **T-AU-8.1** — Implement music system: beat clock (BPM, time signature), beat/bar event firing
  - **Acceptance**: Beat clock generates beats and bars from BPM and time signature; `BeatHit` and `BarHit` events fired on each boundary; clock resynchronizable; 8+ test cases
  - **Deps**: Foundation EventLog, Foundation EventMeta
  - **Effort**: 2 days

- [ ] **T-AU-8.2** — Implement MusicStateMachine with validated state transitions (StateMeta)
  - **Acceptance**: States defined via `@state` with `_valid_transitions`; invalid transitions rejected at registration; current state queryable; state change fires event; 10+ test cases
  - **Deps**: Foundation StateMeta, Foundation EventLog
  - **Effort**: 2 days

- [ ] **T-AU-8.3** — Wire `@state` / StateMeta to music state registration and validation
  - **Acceptance**: `@state` on MusicState class registers valid transitions; StateMeta validates at class creation time; 6+ test cases
  - **Deps**: T-AU-8.2, Foundation StateMeta
  - **Effort**: 1 day

### Horizontal Resequencing

- [ ] **T-AU-8.4** — Implement horizontal resequencing: state-to-segment mapping, segment selection (random, ordered, intensity-based)
  - **Acceptance**: Each state maps to N segment clips; segment selected by mode (random with weights, sequential, intensity threshold); segment plays to completion before next selection; 10+ test cases
  - **Deps**: T-AU-3.4 (AudioClip for segments), T-AU-8.2
  - **Effort**: 3 days

- [ ] **T-AU-8.5** — Implement segment queuing and seamless transition (beat-synced segment boundaries)
  - **Acceptance**: Next segment queued before current ends; transition at segment boundary (not mid-segment); optional overlap with configurable crossfade; 8+ test cases
  - **Deps**: T-AU-8.4, T-AU-8.1
  - **Effort**: 2 days

### Vertical Remixing

- [ ] **T-AU-8.6** — Implement vertical remixing: stem group, layer index, intensity-gated volume
  - **Acceptance**: Stems organized in layers (0 = base, N = highest); layer volume driven by intensity [0, 1]; configurable fade-in/fade-out over N beats; 10+ test cases
  - **Deps**: T-AU-8.1, T-AU-3.4
  - **Effort**: 3 days

- [ ] **T-AU-8.7** — Wire `@music_stem(group, layer, sync_to_beat)` decorator to stem registration
  - **Acceptance**: Decorated class registered as stem; group, layer, sync_to_beat tagged; Registry queryable; 4+ test cases
  - **Deps**: T-AU-8.6, Foundation ResourceMeta
  - **Effort**: 1 day

### Music Transitions

- [ ] **T-AU-8.8** — Implement music transitions: immediate, next_beat, next_bar, crossfade (with duration_beats)
  - **Acceptance**: immediate switches instantly; next_beat quantizes to nearest beat; next_bar quantizes to bar boundary; crossfade blends over duration_beats; no audible glitch at boundaries; 12+ test cases
  - **Deps**: T-AU-8.1, T-AU-8.4, T-AU-8.6
  - **Effort**: 4 days

- [ ] **T-AU-8.9** — Wire `@music_transition(from_state, to_state, type, duration_beats)` decorator to transition rules
  - **Acceptance**: Decorated class registered as transition rule; lookup by (from_state, to_state) pair; 6+ test cases
  - **Deps**: T-AU-8.8, Foundation ResourceMeta
  - **Effort**: 1 day

### Stingers

- [ ] **T-AU-8.10** — Implement stinger system: one-shot musical accents triggered by game events
  - **Acceptance**: Stingers registered with trigger event type; on trigger, stinger queued on next beat boundary; stinger plays as overlay without changing current state; 8+ test cases
  - **Deps**: T-AU-8.1, T-AU-3.10 (play command)
  - **Effort**: 2 days

### Composite Decorators and Integration

- [ ] **T-AU-8.11** — Implement `@adaptive_audio` composite stack (stem + transition + snapshot + serializable)
  - **Acceptance**: Single decorator expands to @music_stem + @music_transition + @audio_snapshot + @serializable; all constituent decorators applied; 4+ test cases
  - **Deps**: T-AU-8.7, T-AU-8.9, T-AU-2.5, Foundation Serilizer
  - **Effort**: 1 day

- [ ] **T-AU-8.12** — Implement `MusicUpdateSystem`: drive beat clock, transition logic, stem volume per frame
  - **Acceptance**: System runs each frame; advances beat clock; checks for pending state transitions; updates stem volumes based on intensity; fires BeatHit/BarHit events; 10+ test cases
  - **Deps**: T-AU-8.1 through T-AU-8.10, Foundation SystemMeta
  - **Effort**: 3 days

- [ ] **T-AU-8.13** — Implement music state persistence via Session (current state, intensity)
  - **Acceptance**: Current music state and intensity save/load across sessions; 4+ test cases
  - **Deps**: T-AU-8.2, Foundation Session, Foundation Serilizer
  - **Effort**: 1 day

- [ ] **T-AU-8.14** — Implement `@audio_snapshot` integration with music system (state-specific bus overrides)
  - **Acceptance**: Each music state maps to an audio snapshot (bus volume overrides); snapshot applied on state entry; snapshot crossfade time configurable per transition; 6+ test cases
  - **Deps**: T-AU-8.8, T-AU-2.5
  - **Effort**: 2 days

---

## Phase 9: Dialogue System (11 tasks)

### VO Management

- [ ] **T-AU-9.1** — Implement VO queue: priority-sorted, configurable depth (default 8), per-category buckets
  - **Acceptance**: Queue depth configurable; items sorted by priority; per-category buckets with independent depth limits; O(log N) insert; 12+ test cases
  - **Deps**: None
  - **Effort**: 2 days

- [ ] **T-AU-9.2** — Implement interruption rules: none (current plays to end), low-only (interrupt if priority higher than threshold), always
  - **Acceptance**: Current VO interrupted based on rule + incoming priority; interrupted VO can resume or be discarded; 10+ test cases
  - **Deps**: T-AU-9.1
  - **Effort**: 2 days

- [ ] **T-AU-9.3** — Implement overlap handling: replace (new replaces current), queue (new goes to queue), discard (new silently dropped)
  - **Acceptance**: Each request specifies behavior; replace stops current and plays new; queue enqueues; discard returns without action; 8+ test cases
  - **Deps**: T-AU-9.1
  - **Effort**: 1 day

- [ ] **T-AU-9.4** — Register dialogue voices with highest voice priority tier (reserved voice slots)
  - **Acceptance**: Dialogue manager reserves N voice slots at highest priority; dialogue Always takes precedence over game audio; configurable reservation count; 6+ test cases
  - **Deps**: T-AU-2.7 (voice pool), T-AU-3.10 (play command)
  - **Effort**: 1 day

### Line Selection

- [ ] **T-AU-9.5** — Implement line selection modes: Random, Sequential, Cooldown-based, Conditional
  - **Acceptance**: Random picks uniformly; Sequential plays in order; Cooldown excludes recently-played lines; Conditional filters by game state expression; 12+ test cases
  - **Deps**: T-AU-9.1
  - **Effort**: 3 days

- [ ] **T-AU-9.6** — Implement line type categories: Barks, Conversations, Ambient VO, Narration
  - **Acceptance**: Each type has configurable behavior: bark cooldown, conversation sequential guarantee, ambient low priority, narration high priority + pause others; 8+ test cases
  - **Deps**: T-AU-9.5
  - **Effort**: 2 days

- [ ] **T-AU-9.7** — Implement dialogue bank format (CSV/JSON manifest for line ID, language, file, priority, category, selection mode, cooldown)
  - **Acceptance**: DialogueBank loaded from manifest; line lookup by ID; all fields parsed and validated; 8+ test cases
  - **Deps**: T-AU-9.5, Foundation AssetMeta
  - **Effort**: 2 days

### Lipsync

- [ ] **T-AU-9.8** — Implement pre-baked lipsync data format: per-phoneme timing (phoneme ID, start time, end time) per line
  - **Acceptance**: Lipsync data loaded as sidecar file; phoneme stream per dialogue line; timestamps in seconds; format: JSON or binary; 6+ test cases
  - **Deps**: None
  - **Effort**: 2 days

- [ ] **T-AU-9.9** — Integrate lipsync data with animation system (phoneme stream -> blend shape driver)
  - **Acceptance**: Animation system receives phoneme stream; blend shapes driven by current phoneme + interpolation to next; configurable transition time between phonemes; 6+ test cases
  - **Deps**: T-AU-9.8, Animation system (blend shape interface)
  - **Effort**: 3 days

### Localization

- [ ] **T-AU-9.10** — Implement localization: language code per clip, language-aware line lookup, fallback chain
  - **Acceptance**: Dialogue bank includes language codes; lookup by (language, line_id); fallback: requested -> default -> subtitles-only; language switch at runtime supported; 8+ test cases
  - **Deps**: T-AU-9.7
  - **Effort**: 2 days

### Dialogue Manager

- [ ] **T-AU-9.11** — Implement `DialogueManager` as system: processes dialogue requests, manages queue, drives playback and lipsync
  - **Acceptance**: DialogueManager receives requests; resolves lines; manages queue with priority/interruption; plays via AudioSource; sends lipsync data to animation; fires `DialogueEvent` for EventLog; 15+ test cases
  - **Deps**: T-AU-9.1 through T-AU-9.10, Foundation EventLog, Foundation SystemMeta
  - **Effort**: 4 days

---

## Summary

| Phase | Tasks | Est. Effort |
|-------|-------|-------------|
| 1. Audio Device Abstraction & Core Types | 12 | 26.5 days |
| 2. Mixer Graph & Voice Management | 18 | 29.5 days |
| 3. Sound Playback Engine | 14 | 23 days |
| 4. Stream & Decode Thread Architecture | 10 | 20 days |
| 5. Spatial Audio System | 15 | 37 days |
| 6. Acoustic Simulation | 15 | 43 days |
| 7. DSP Effect Chain | 20 | 49 days |
| 8. Adaptive Music Engine | 14 | 27 days |
| 9. Dialogue System | 11 | 24 days |
| **Total** | **134** | **279 days** |
