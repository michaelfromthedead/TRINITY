# Phase 4 Architecture: Stream and Decode Thread Architecture

## Purpose
Multi-threaded audio streaming infrastructure: decode workers, stream I/O thread, memory pool management, and non-blocking audio tick.

## Current Implementation
**No tasks fully complete.** All existing streaming infrastructure is Python-level prototype in `dialogue/vo_streaming.py`:

**Partial [~] (5 tasks):**
- `VOStreamManager` with stream state machine (IDLE/LOADING/BUFFERING/READY/STREAMING/PAUSED/COMPLETED/ERROR) — Python, not thread-safe at C level
- `VOCache` with LRU eviction, max_size_mb, hit rate tracking — Python OrderedDict, not lock-free
- `StreamHandle` with buffer_fill_percent tracking — Python dataclass
- Preload queue with anticipated line system — Python list, blocking
- Streaming source type (`SourceType.STREAMING` enum exists)

**Missing [-] (5 tasks):**
- Decode thread pool with configurable worker count
- MPSC decode job queue (stream threads -> decode workers)
- Temporary memory pool (one-shot decode buffer with timed release)
- Format plugin interface per decoder
- Audio tick that never blocks on I/O (requires all pre-decoded data)

## Architecture
```
Thread Hierarchy:
  Game Thread -> SPSC command queue -> Audio Thread (tick, never blocks)
  Stream Thread (async file I/O) -> MPSC decode jobs -> Decode Workers (N = cpu-2)
  Decode Workers -> MPSC decoded PCM -> Audio Thread (stream ring buffer)

Memory Pools:
  Resident (64MB): preloaded sounds, LRU eviction, preload-marked pinned
  Streaming (32MB): per-stream ring buffers, chunk management
  Temporary (16MB): one-shot decodes, timed release after voice stop

Stream State Machine:
  Idle -> Opening -> Reading -> Decoding -> Playing -> Draining -> Idle
  (with seamless chunk transition and underrun recovery)
```
