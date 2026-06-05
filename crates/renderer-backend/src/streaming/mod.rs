// SPDX-License-Identifier: MIT
//
// streaming/mod.rs -- 3-Thread Streaming Architecture (T-AS-5.1)
//
// This module provides a background streaming system for asset loading
// with lock-free communication between threads.
//
// # Architecture
//
// The streaming system uses three threads:
// 1. **Game Thread** - Submits streaming requests and priority updates
// 2. **Streaming Thread** - Performs async I/O and data preparation
// 3. **Render Thread** - Consumes GPU upload commands
//
// # Communication
//
// Thread communication uses lock-free data structures:
// - SPSC Queue: Game -> Streaming (request submission)
// - MPSC Queue: Any -> Streaming (priority updates)
// - Ring Buffer: Streaming -> Render (GPU upload commands)
//
// # Example
//
// ```ignore
// use renderer_backend::streaming::{
//     StreamingThread, StreamingConfig, StreamRequest, GpuUploadCommand,
// };
//
// // Create and start streaming thread
// let streaming = StreamingThread::new(StreamingConfig::default());
//
// // Game thread: submit requests
// streaming.submit_request(StreamRequest::new(1, 0, "texture.dds".into()));
//
// // Game thread: update priorities
// streaming.update_priority(PriorityUpdate::new(1, 10));
//
// // Render thread: consume upload commands
// while let Some(cmd) = streaming.pop_upload_command() {
//     match cmd {
//         GpuUploadCommand::UploadTexture { asset_id, data, .. } => {
//             // Upload texture to GPU
//         }
//         GpuUploadCommand::AssetReady { asset_id } => {
//             // Mark asset as ready
//         }
//         _ => {}
//     }
// }
//
// // Shutdown
// streaming.shutdown();
// ```
//
// # Thread Safety
//
// - `submit_request()`: Single producer only (game thread)
// - `update_priority()`, `cancel_request()`: Any thread
// - `pop_upload_command()`: Single consumer only (render thread)
// - `stats()`, `shutdown()`: Any thread

pub mod budget;
pub mod budget_lod;
pub mod jobs;
pub mod predictive;
pub mod priority_queue;
pub mod queues;
pub mod remote_cache;
pub mod streaming_thread;

// Re-export main types
pub use queues::{
    AtomicLoadState, AtomicRefCount, GpuUploadCommand, LoadState, MpscQueue, PriorityUpdate,
    RingBuffer, SpscQueue, StreamRequest,
};

pub use streaming_thread::{
    StreamingConfig, StreamingStats, StreamingThread, StreamingThreadHandle,
};

pub use priority_queue::{
    BinaryHeap, LockFreeSkipList, PriorityEntry, PriorityFactors, PriorityQueueStats,
    PriorityTier, PriorityUpdateMsg, PriorityWeights, StreamingPriorityQueue,
};

pub use budget::{
    AssetFootprint, AssetId, AssetType, BudgetConfig, BudgetManager, BudgetUsage,
    EvictionCandidate, DEFAULT_GLOBAL_BUDGET, DEFAULT_IO_BUDGET_PER_FRAME,
    DEFAULT_MESH_BUDGET, DEFAULT_SHADER_BUDGET, DEFAULT_TARGET_FRAME_TIME_MS,
    DEFAULT_TEXTURE_BUDGET,
};

pub use predictive::{
    AssetBounds, CameraState, Frustum, Plane, PredictionConfig, PredictionResult,
    PredictiveLoader, Quat, Vec3, AABB,
};

pub use jobs::{
    compress_lz4, compress_zlib, compress_zstd, decompress, decompress_auto, decompress_lz4,
    decompress_zlib, decompress_zstd, deserialize_asset, deserialize_mesh, deserialize_texture,
    deserialize_shader, deserialize_animation, deserialize_audio,
    AnimationData, AudioData, CompressionFormat, DecompressJob,
    DeserializedAsset, DeserializeJob, IoDecompressPipeline, JobError, JobHandle, JobPriority,
    JobResult, JobStatus, JobSystemConfig, MeshData, ShaderData, StreamingDecompressor,
    StreamingJobManager, TextureData,
};

pub use remote_cache::{
    CacheResult, CacheStats, HttpClient, HttpMethod, HttpRequest, HttpResponse,
    MockHttpClient, RemoteCache, RemoteCacheConfig, RemoteCacheError, SyncStats,
    DEFAULT_BATCH_SIZE, DEFAULT_MAX_RETRIES, DEFAULT_RECONNECT_INTERVAL_MS,
    DEFAULT_TIMEOUT_MS, DEFAULT_UPLOAD_QUEUE_CAPACITY,
};

pub use budget_lod::{
    AssetLodState, BudgetAwareLodSelector, LodBudget, LodBudgetConfig,
    DEFAULT_PRIORITY_WEIGHT, DEFAULT_REDUCTION_STEP, DEFAULT_TEXEL_BUDGET,
    DEFAULT_TRIANGLE_BUDGET,
};

// ---------------------------------------------------------------------------
// Integration tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};

    /// Full 3-thread simulation test.
    #[test]
    fn three_thread_integration() {
        let streaming = Arc::new(StreamingThread::new_default());

        let game_handle = {
            let s = Arc::clone(&streaming);
            thread::spawn(move || {
                // Game thread: submit requests
                for i in 0..50 {
                    let request = StreamRequest::new(i, (50 - i) as u32, format!("asset_{}.bin", i));
                    while s.submit_request(request.clone()).is_err() {
                        thread::yield_now();
                    }

                    // Occasionally update priorities
                    if i % 10 == 5 {
                        s.update_priority(PriorityUpdate::new(i - 5, 0));
                    }
                }
            })
        };

        let render_handle = {
            let s = Arc::clone(&streaming);
            thread::spawn(move || {
                // Render thread: consume uploads
                let mut ready_count = 0;
                let start = Instant::now();

                while ready_count < 50 && start.elapsed() < Duration::from_secs(5) {
                    match s.pop_upload_command() {
                        Some(GpuUploadCommand::AssetReady { .. }) => {
                            ready_count += 1;
                        }
                        Some(_) => {}
                        None => thread::yield_now(),
                    }
                }

                ready_count
            })
        };

        game_handle.join().unwrap();
        let ready_count = render_handle.join().unwrap();

        // All assets should have been processed
        assert_eq!(ready_count, 50);

        streaming.shutdown();
    }

    /// Test high-throughput streaming.
    #[test]
    fn high_throughput_streaming() {
        let config = StreamingConfig {
            request_queue_capacity: 8192,
            upload_buffer_capacity: 1024,
            max_requests_per_tick: 128,
            max_uploads_per_tick: 64,
            idle_sleep_ms: 0,
            ..Default::default()
        };

        let streaming = Arc::new(StreamingThread::new(config));

        let items = 1000;

        let producer = {
            let s = Arc::clone(&streaming);
            thread::spawn(move || {
                for i in 0..items {
                    let request = StreamRequest::new(i, 0, format!("asset_{}.bin", i));
                    while s.submit_request(request.clone()).is_err() {
                        thread::yield_now();
                    }
                }
            })
        };

        let consumer = {
            let s = Arc::clone(&streaming);
            thread::spawn(move || {
                let mut count = 0;
                let start = Instant::now();

                while count < items * 2 && start.elapsed() < Duration::from_secs(10) {
                    if s.pop_upload_command().is_some() {
                        count += 1;
                    } else {
                        thread::yield_now();
                    }
                }

                count
            })
        };

        producer.join().unwrap();
        let consumed = consumer.join().unwrap();

        // Should have consumed uploads and ready signals
        assert!(consumed >= items as u64);

        streaming.shutdown();
    }

    /// Test cancellation during streaming.
    #[test]
    fn cancellation_during_streaming() {
        let streaming = StreamingThread::new_default();

        // Submit many requests
        for i in 0..20 {
            let request = StreamRequest::new(i, 0, format!("asset_{}.bin", i));
            let _ = streaming.submit_request(request);
        }

        // Cancel half immediately
        for i in 0..10 {
            streaming.cancel_request(i);
        }

        // Wait for processing
        thread::sleep(Duration::from_millis(100));

        // Drain uploads
        while streaming.pop_upload_command().is_some() {}

        streaming.shutdown();
    }

    /// Test reference counting across threads.
    #[test]
    fn reference_counting_threaded() {
        let ref_count = Arc::new(AtomicRefCount::new());
        let mut handles = vec![];

        // Spawn threads that increment/decrement
        for _ in 0..4 {
            let rc = Arc::clone(&ref_count);
            handles.push(thread::spawn(move || {
                for _ in 0..1000 {
                    rc.increment();
                }
                for _ in 0..1000 {
                    rc.decrement();
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // Final count should be 1 (initial)
        assert_eq!(ref_count.get(), 1);
    }

    /// Test load state transitions across threads.
    #[test]
    fn load_state_concurrent_transitions() {
        let state = Arc::new(AtomicLoadState::new(LoadState::None));

        let state1 = Arc::clone(&state);
        let state2 = Arc::clone(&state);

        // Two threads try to transition
        let h1 = thread::spawn(move || {
            state1
                .transition(
                    LoadState::None,
                    LoadState::Queued,
                    std::sync::atomic::Ordering::AcqRel,
                    std::sync::atomic::Ordering::Acquire,
                )
                .is_ok()
        });

        let h2 = thread::spawn(move || {
            state2
                .transition(
                    LoadState::None,
                    LoadState::Queued,
                    std::sync::atomic::Ordering::AcqRel,
                    std::sync::atomic::Ordering::Acquire,
                )
                .is_ok()
        });

        let r1 = h1.join().unwrap();
        let r2 = h2.join().unwrap();

        // Exactly one should succeed
        assert!(r1 ^ r2);

        // State should be Queued
        assert_eq!(
            state.load(std::sync::atomic::Ordering::Relaxed),
            LoadState::Queued
        );
    }
}
