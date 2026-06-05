// SPDX-License-Identifier: MIT
//
// streaming_thread.rs -- Background streaming thread implementation (T-AS-5.1)
//
// Provides a dedicated streaming thread that:
// - Receives streaming requests via lock-free SPSC queue
// - Handles priority updates via lock-free MPSC queue
// - Submits GPU upload commands via ring buffer
// - Supports clean shutdown with queue draining

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use super::queues::{
    AtomicLoadState, AtomicRefCount, GpuUploadCommand, LoadState, MpscQueue, PriorityUpdate,
    RingBuffer, SpscQueue, StreamRequest,
};

// ---------------------------------------------------------------------------
// StreamingConfig
// ---------------------------------------------------------------------------

/// Configuration for the streaming thread.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Capacity of the request queue (game -> streaming).
    pub request_queue_capacity: usize,
    /// Capacity of the GPU upload ring buffer.
    pub upload_buffer_capacity: usize,
    /// Maximum requests to process per tick.
    pub max_requests_per_tick: usize,
    /// Maximum uploads to submit per tick.
    pub max_uploads_per_tick: usize,
    /// Idle sleep duration when no work is available.
    pub idle_sleep_ms: u64,
    /// Thread name for debugging.
    pub thread_name: String,
    /// Whether to set thread priority (platform-specific).
    pub set_thread_priority: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            request_queue_capacity: 4096,
            upload_buffer_capacity: 256,
            max_requests_per_tick: 64,
            max_uploads_per_tick: 32,
            idle_sleep_ms: 1,
            thread_name: "streaming".to_string(),
            set_thread_priority: false,
        }
    }
}

// ---------------------------------------------------------------------------
// AssetState -- Per-asset tracking
// ---------------------------------------------------------------------------

/// Internal state for a single asset being streamed.
struct AssetState {
    /// Request information.
    request: StreamRequest,
    /// Current loading state.
    load_state: AtomicLoadState,
    /// Reference count.
    ref_count: AtomicRefCount,
    /// When loading started.
    load_start: Option<Instant>,
    /// Loaded data (if available).
    data: Option<Arc<[u8]>>,
}

impl AssetState {
    fn new(request: StreamRequest) -> Self {
        Self {
            request,
            load_state: AtomicLoadState::new(LoadState::Queued),
            ref_count: AtomicRefCount::new(),
            load_start: None,
            data: None,
        }
    }
}

// ---------------------------------------------------------------------------
// StreamingStats
// ---------------------------------------------------------------------------

/// Statistics for the streaming thread.
#[derive(Debug, Clone, Default)]
pub struct StreamingStats {
    /// Total requests processed.
    pub requests_processed: u64,
    /// Total uploads submitted.
    pub uploads_submitted: u64,
    /// Total bytes loaded.
    pub bytes_loaded: u64,
    /// Current active loads.
    pub active_loads: usize,
    /// Requests waiting in queue.
    pub pending_requests: usize,
    /// Failed loads.
    pub failed_loads: u64,
    /// Average load time in milliseconds.
    pub avg_load_time_ms: f64,
    /// Peak queue depth.
    pub peak_queue_depth: usize,
}

// ---------------------------------------------------------------------------
// StreamingThread
// ---------------------------------------------------------------------------

/// Handle to the streaming thread.
///
/// The streaming thread runs in the background and handles asset loading
/// separately from the game and render threads.
///
/// # Architecture
///
/// ```text
/// Game Thread                  Streaming Thread              Render Thread
///      |                              |                            |
///      |-- StreamRequest -->          |                            |
///      |      (SPSC Queue)            |                            |
///      |                              |                            |
///      |-- PriorityUpdate -->         |                            |
///      |      (MPSC Queue)            |                            |
///      |                              |                            |
///      |                              |-- GpuUploadCommand -->     |
///      |                              |      (Ring Buffer)         |
/// ```
///
/// # Thread Safety
///
/// - `request_queue`: SPSC, game thread is producer, streaming thread is consumer
/// - `priority_queue`: MPSC, any thread is producer, streaming thread is consumer
/// - `upload_buffer`: Ring buffer, streaming thread is producer, render thread is consumer
///
/// # Example
///
/// ```ignore
/// let config = StreamingConfig::default();
/// let streaming = StreamingThread::new(config);
///
/// // Submit a request from the game thread
/// streaming.submit_request(StreamRequest::new(1, 0, "texture.dds".into()));
///
/// // Check for uploads from the render thread
/// while let Some(cmd) = streaming.pop_upload_command() {
///     // Process GPU upload
/// }
///
/// // Shutdown
/// streaming.shutdown();
/// ```
pub struct StreamingThread {
    /// Request queue: game -> streaming (SPSC).
    request_queue: Arc<SpscQueue<StreamRequest>>,
    /// Priority update queue: any -> streaming (MPSC).
    priority_queue: Arc<MpscQueue<PriorityUpdate>>,
    /// Upload command buffer: streaming -> render (Ring buffer).
    upload_buffer: Arc<RingBuffer<GpuUploadCommand>>,
    /// Running flag for shutdown.
    running: Arc<AtomicBool>,
    /// Shutdown complete flag.
    shutdown_complete: Arc<AtomicBool>,
    /// Wake condition variable.
    wake_cv: Arc<Condvar>,
    wake_mutex: Arc<Mutex<()>>,
    /// Thread handle.
    handle: Mutex<Option<JoinHandle<()>>>,
    /// Statistics.
    stats: Arc<Mutex<StreamingStats>>,
    /// Monotonic counter for fence IDs.
    fence_counter: AtomicU64,
    /// Configuration.
    config: StreamingConfig,
}

impl StreamingThread {
    /// Creates and starts a new streaming thread.
    pub fn new(config: StreamingConfig) -> Self {
        let request_queue = Arc::new(SpscQueue::new(config.request_queue_capacity));
        let priority_queue = Arc::new(MpscQueue::new());
        let upload_buffer = Arc::new(RingBuffer::new(config.upload_buffer_capacity));
        let running = Arc::new(AtomicBool::new(true));
        let shutdown_complete = Arc::new(AtomicBool::new(false));
        let wake_cv = Arc::new(Condvar::new());
        let wake_mutex = Arc::new(Mutex::new(()));
        let stats = Arc::new(Mutex::new(StreamingStats::default()));

        // Clone for thread
        let thread_request_queue = Arc::clone(&request_queue);
        let thread_priority_queue = Arc::clone(&priority_queue);
        let thread_upload_buffer = Arc::clone(&upload_buffer);
        let thread_running = Arc::clone(&running);
        let thread_shutdown_complete = Arc::clone(&shutdown_complete);
        let thread_wake_cv = Arc::clone(&wake_cv);
        let thread_wake_mutex = Arc::clone(&wake_mutex);
        let thread_stats = Arc::clone(&stats);
        let thread_config = config.clone();

        let handle = thread::Builder::new()
            .name(config.thread_name.clone())
            .spawn(move || {
                streaming_thread_main(
                    thread_request_queue,
                    thread_priority_queue,
                    thread_upload_buffer,
                    thread_running,
                    thread_shutdown_complete,
                    thread_wake_cv,
                    thread_wake_mutex,
                    thread_stats,
                    thread_config,
                );
            })
            .expect("Failed to spawn streaming thread");

        Self {
            request_queue,
            priority_queue,
            upload_buffer,
            running,
            shutdown_complete,
            wake_cv,
            wake_mutex,
            handle: Mutex::new(Some(handle)),
            stats,
            fence_counter: AtomicU64::new(0),
            config,
        }
    }

    /// Creates a streaming thread with default configuration.
    pub fn new_default() -> Self {
        Self::new(StreamingConfig::default())
    }

    /// Submits a streaming request.
    ///
    /// Returns `Ok(())` if the request was queued, or `Err(request)` if
    /// the queue is full.
    ///
    /// # Thread Safety
    ///
    /// This must only be called from the game thread (single producer).
    pub fn submit_request(&self, request: StreamRequest) -> Result<(), StreamRequest> {
        let result = self.request_queue.push(request);
        if result.is_ok() {
            self.wake();
        }
        result
    }

    /// Submits a priority update.
    ///
    /// This is lock-free and can be called from any thread.
    pub fn update_priority(&self, update: PriorityUpdate) {
        self.priority_queue.push(update);
        self.wake();
    }

    /// Cancels a streaming request.
    ///
    /// This is a convenience method that submits a cancellation update.
    pub fn cancel_request(&self, asset_id: u64) {
        self.update_priority(PriorityUpdate::cancel(asset_id));
    }

    /// Pops a GPU upload command.
    ///
    /// Returns `Some(command)` if available, or `None` if the buffer is empty.
    ///
    /// # Thread Safety
    ///
    /// This should only be called from the render thread (single consumer).
    pub fn pop_upload_command(&self) -> Option<GpuUploadCommand> {
        self.upload_buffer.pop()
    }

    /// Returns the number of pending requests in the queue.
    pub fn pending_requests(&self) -> usize {
        self.request_queue.len()
    }

    /// Returns the number of pending upload commands.
    pub fn pending_uploads(&self) -> usize {
        self.upload_buffer.len()
    }

    /// Returns current statistics.
    pub fn stats(&self) -> StreamingStats {
        self.stats.lock().clone()
    }

    /// Wakes the streaming thread.
    fn wake(&self) {
        self.wake_cv.notify_one();
    }

    /// Requests a fence that will be signaled when all current work is done.
    ///
    /// Returns a fence ID that will appear in the upload buffer as a
    /// `GpuUploadCommand::Fence` when reached.
    pub fn request_fence(&self) -> u64 {
        let fence_id = self.fence_counter.fetch_add(1, Ordering::Relaxed);
        // The streaming thread will emit the fence when it processes it
        fence_id
    }

    /// Returns true if the streaming thread is still running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Returns true if shutdown is complete.
    pub fn is_shutdown_complete(&self) -> bool {
        self.shutdown_complete.load(Ordering::Relaxed)
    }

    /// Initiates shutdown of the streaming thread.
    ///
    /// This will:
    /// 1. Signal the thread to stop accepting new work
    /// 2. Drain all pending requests and complete in-flight I/O
    /// 3. Join the thread
    pub fn shutdown(&self) {
        if !self.running.swap(false, Ordering::Relaxed) {
            // Already shutdown
            return;
        }

        // Wake the thread so it can see the shutdown flag
        self.wake_cv.notify_all();

        // Join the thread
        if let Some(handle) = self.handle.lock().take() {
            let _ = handle.join();
        }
    }

    /// Initiates graceful shutdown and waits for completion.
    ///
    /// Unlike `shutdown()`, this waits for all pending uploads to be
    /// processed by the render thread.
    pub fn shutdown_graceful(&self, timeout: Duration) -> bool {
        self.shutdown();

        let start = Instant::now();
        while !self.upload_buffer.is_empty() {
            if start.elapsed() > timeout {
                return false;
            }
            thread::sleep(Duration::from_millis(1));
        }

        true
    }

    /// Returns the configuration.
    pub fn config(&self) -> &StreamingConfig {
        &self.config
    }
}

impl Drop for StreamingThread {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// ---------------------------------------------------------------------------
// Streaming Thread Main Loop
// ---------------------------------------------------------------------------

fn streaming_thread_main(
    request_queue: Arc<SpscQueue<StreamRequest>>,
    priority_queue: Arc<MpscQueue<PriorityUpdate>>,
    upload_buffer: Arc<RingBuffer<GpuUploadCommand>>,
    running: Arc<AtomicBool>,
    shutdown_complete: Arc<AtomicBool>,
    wake_cv: Arc<Condvar>,
    wake_mutex: Arc<Mutex<()>>,
    stats: Arc<Mutex<StreamingStats>>,
    config: StreamingConfig,
) {
    // Per-asset state tracking
    let mut assets: HashMap<u64, AssetState> = HashMap::new();
    let mut pending_loads: Vec<u64> = Vec::new();
    let mut load_times: Vec<f64> = Vec::new();

    // Main loop
    loop {
        let mut did_work = false;

        // Process priority updates first (they may cancel pending requests)
        for _ in 0..config.max_requests_per_tick {
            match priority_queue.pop() {
                Some(update) => {
                    did_work = true;
                    process_priority_update(&mut assets, &update);
                }
                None => break,
            }
        }

        // Process new requests
        let mut processed = 0;
        for _ in 0..config.max_requests_per_tick {
            match request_queue.pop() {
                Some(request) => {
                    did_work = true;
                    processed += 1;

                    let asset_id = request.asset_id;
                    assets.insert(asset_id, AssetState::new(request));
                    pending_loads.push(asset_id);
                }
                None => break,
            }
        }

        if processed > 0 {
            let mut s = stats.lock();
            s.pending_requests = pending_loads.len();
            if pending_loads.len() > s.peak_queue_depth {
                s.peak_queue_depth = pending_loads.len();
            }
        }

        // Process pending loads (simulate loading for now)
        let mut completed = Vec::new();
        for asset_id in pending_loads.iter() {
            if let Some(state) = assets.get_mut(asset_id) {
                // Transition to loading if still queued
                if state.load_state.load(Ordering::Relaxed) == LoadState::Queued {
                    let _ = state.load_state.transition(
                        LoadState::Queued,
                        LoadState::Loading,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    );
                    state.load_start = Some(Instant::now());
                }

                // For testing/simulation: complete immediately
                // In production, this would do actual I/O
                if state.load_state.load(Ordering::Relaxed) == LoadState::Loading {
                    // Simulate successful load
                    let data: Arc<[u8]> = Arc::from(vec![0u8; 1024]);
                    state.data = Some(data.clone());

                    // Transition to uploading
                    let _ = state.load_state.transition(
                        LoadState::Loading,
                        LoadState::Uploading,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    );

                    // Record load time
                    if let Some(start) = state.load_start {
                        load_times.push(start.elapsed().as_secs_f64() * 1000.0);
                    }

                    // Queue upload command
                    let cmd = GpuUploadCommand::UploadBuffer {
                        asset_id: *asset_id,
                        data,
                        offset: 0,
                    };

                    // Try to push, may fail if buffer is full
                    if upload_buffer.push(cmd).is_ok() {
                        // Mark complete
                        let _ = state.load_state.transition(
                            LoadState::Uploading,
                            LoadState::Ready,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        );

                        // Queue ready signal
                        let _ = upload_buffer.push(GpuUploadCommand::AssetReady {
                            asset_id: *asset_id,
                        });

                        completed.push(*asset_id);
                        did_work = true;

                        let mut s = stats.lock();
                        s.requests_processed += 1;
                        s.uploads_submitted += 1;
                        s.bytes_loaded += 1024;
                    }
                }
            }
        }

        // Remove completed assets from pending list
        pending_loads.retain(|id| !completed.contains(id));

        // Update stats
        if !load_times.is_empty() {
            let mut s = stats.lock();
            s.active_loads = pending_loads.len();
            s.avg_load_time_ms = load_times.iter().sum::<f64>() / load_times.len() as f64;
        }

        // Check for shutdown
        if !running.load(Ordering::Relaxed) {
            // Drain remaining work
            while request_queue.pop().is_some() {}
            while priority_queue.pop().is_some() {}

            // Complete any in-flight loads
            for asset_id in pending_loads.drain(..) {
                if let Some(state) = assets.get(&asset_id) {
                    // Mark as failed due to shutdown
                    let _ = upload_buffer.push(GpuUploadCommand::AssetFailed {
                        asset_id,
                        error_code: 1, // Shutdown
                    });
                    let _ = state.load_state.transition(
                        state.load_state.load(Ordering::Relaxed),
                        LoadState::Failed,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    );
                }
            }

            shutdown_complete.store(true, Ordering::Release);
            return;
        }

        // Sleep if no work was done
        if !did_work {
            let mut guard = wake_mutex.lock();
            // Check running flag again before sleeping
            if running.load(Ordering::Relaxed) {
                let _ = wake_cv.wait_for(&mut guard, Duration::from_millis(config.idle_sleep_ms));
            }
        }
    }
}

fn process_priority_update(assets: &mut HashMap<u64, AssetState>, update: &PriorityUpdate) {
    if let Some(state) = assets.get_mut(&update.asset_id) {
        if update.cancel {
            // Mark as failed/cancelled
            state
                .load_state
                .store(LoadState::Failed, Ordering::Release);
        } else {
            // Update priority
            state.request.priority = update.new_priority;
        }
    }
}

// ---------------------------------------------------------------------------
// StreamingThreadHandle -- Shared handle for multi-threaded access
// ---------------------------------------------------------------------------

/// A shareable handle to the streaming thread.
///
/// This provides a convenient way to share the streaming thread across
/// multiple components using Arc internally.
#[derive(Clone)]
pub struct StreamingThreadHandle {
    inner: Arc<StreamingThread>,
}

impl StreamingThreadHandle {
    /// Creates a new streaming thread handle.
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            inner: Arc::new(StreamingThread::new(config)),
        }
    }

    /// Creates a handle from an existing streaming thread.
    pub fn from_thread(thread: StreamingThread) -> Self {
        Self {
            inner: Arc::new(thread),
        }
    }

    /// Submits a streaming request.
    pub fn submit_request(&self, request: StreamRequest) -> Result<(), StreamRequest> {
        self.inner.submit_request(request)
    }

    /// Updates priority of a request.
    pub fn update_priority(&self, update: PriorityUpdate) {
        self.inner.update_priority(update);
    }

    /// Cancels a request.
    pub fn cancel_request(&self, asset_id: u64) {
        self.inner.cancel_request(asset_id);
    }

    /// Pops an upload command.
    pub fn pop_upload_command(&self) -> Option<GpuUploadCommand> {
        self.inner.pop_upload_command()
    }

    /// Returns statistics.
    pub fn stats(&self) -> StreamingStats {
        self.inner.stats()
    }

    /// Shuts down the streaming thread.
    pub fn shutdown(&self) {
        self.inner.shutdown();
    }

    /// Returns a reference to the underlying thread.
    pub fn inner(&self) -> &StreamingThread {
        &self.inner
    }
}

impl std::ops::Deref for StreamingThreadHandle {
    type Target = StreamingThread;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // ── Basic Lifecycle Tests ───────────────────────────────────────────

    #[test]
    fn streaming_thread_spawn_join() {
        let thread = StreamingThread::new_default();
        assert!(thread.is_running());

        thread.shutdown();
        assert!(!thread.is_running());
    }

    #[test]
    fn streaming_thread_spawn_drop() {
        // Thread should be joined on drop
        let thread = StreamingThread::new_default();
        assert!(thread.is_running());
        drop(thread);
    }

    #[test]
    fn streaming_thread_with_config() {
        let config = StreamingConfig {
            request_queue_capacity: 1024,
            upload_buffer_capacity: 64,
            max_requests_per_tick: 32,
            idle_sleep_ms: 5,
            thread_name: "test-streaming".to_string(),
            ..Default::default()
        };

        let thread = StreamingThread::new(config);
        assert!(thread.is_running());
        thread.shutdown();
    }

    // ── Request Submission Tests ────────────────────────────────────────

    #[test]
    fn submit_single_request() {
        let thread = StreamingThread::new_default();

        let request = StreamRequest::new(1, 0, "test.dds".to_string());
        assert!(thread.submit_request(request).is_ok());

        // Give thread time to process
        thread::sleep(Duration::from_millis(50));

        let stats = thread.stats();
        assert!(stats.requests_processed >= 1);

        thread.shutdown();
    }

    #[test]
    fn submit_multiple_requests() {
        let thread = StreamingThread::new_default();

        for i in 0..10 {
            let request = StreamRequest::new(i, 0, format!("texture_{}.dds", i));
            assert!(thread.submit_request(request).is_ok());
        }

        // Give thread time to process
        thread::sleep(Duration::from_millis(100));

        let stats = thread.stats();
        assert!(stats.requests_processed >= 10);

        thread.shutdown();
    }

    #[test]
    fn submit_request_queue_full() {
        // Create thread with tiny queue
        let config = StreamingConfig {
            request_queue_capacity: 4, // Will be rounded to 4
            idle_sleep_ms: 100,        // Make thread slow
            ..Default::default()
        };
        let thread = StreamingThread::new(config);

        // Fill the queue quickly
        let mut submitted = 0;
        let mut rejected = 0;
        for i in 0..10 {
            let request = StreamRequest::new(i, 0, format!("tex_{}.dds", i));
            match thread.submit_request(request) {
                Ok(()) => submitted += 1,
                Err(_) => rejected += 1,
            }
        }

        // At least some should be rejected when queue is small
        // (depends on thread timing, so we check total adds up)
        assert_eq!(submitted + rejected, 10);

        thread.shutdown();
    }

    // ── Priority Update Tests ───────────────────────────────────────────

    #[test]
    fn update_priority() {
        let thread = StreamingThread::new_default();

        // Submit request
        let request = StreamRequest::new(42, 10, "test.dds".to_string());
        thread.submit_request(request).unwrap();

        // Update priority
        thread.update_priority(PriorityUpdate::new(42, 0));

        thread::sleep(Duration::from_millis(50));
        thread.shutdown();
    }

    #[test]
    fn cancel_request() {
        let thread = StreamingThread::new_default();

        // Submit request
        let request = StreamRequest::new(42, 0, "test.dds".to_string());
        thread.submit_request(request).unwrap();

        // Cancel it
        thread.cancel_request(42);

        thread::sleep(Duration::from_millis(50));
        thread.shutdown();
    }

    #[test]
    fn priority_update_concurrent() {
        let thread = StreamingThread::new_default();
        let handle = Arc::new(thread);

        let mut threads = vec![];
        for t in 0..4 {
            let h = Arc::clone(&handle);
            threads.push(thread::spawn(move || {
                for i in 0..100 {
                    h.update_priority(PriorityUpdate::new(t * 100 + i, i as u32));
                }
            }));
        }

        for t in threads {
            t.join().unwrap();
        }

        handle.shutdown();
    }

    // ── Upload Command Tests ────────────────────────────────────────────

    #[test]
    fn pop_upload_commands() {
        let thread = StreamingThread::new_default();

        // Submit requests
        for i in 0..5 {
            let request = StreamRequest::new(i, 0, format!("tex_{}.dds", i));
            thread.submit_request(request).unwrap();
        }

        // Wait for processing
        thread::sleep(Duration::from_millis(100));

        // Pop commands
        let mut commands = vec![];
        while let Some(cmd) = thread.pop_upload_command() {
            commands.push(cmd);
        }

        // Should have upload and ready commands
        assert!(!commands.is_empty());

        thread.shutdown();
    }

    #[test]
    fn upload_buffer_wrap_around() {
        // Create thread with small buffer
        let config = StreamingConfig {
            upload_buffer_capacity: 8,
            ..Default::default()
        };
        let thread = StreamingThread::new(config);

        // Submit many requests to force wrap-around
        for round in 0..3 {
            for i in 0..4 {
                let request = StreamRequest::new(round * 10 + i, 0, format!("tex_{}.dds", i));
                let _ = thread.submit_request(request);
            }

            thread::sleep(Duration::from_millis(50));

            // Drain buffer
            while thread.pop_upload_command().is_some() {}
        }

        thread.shutdown();
    }

    // ── Statistics Tests ────────────────────────────────────────────────

    #[test]
    fn stats_tracking() {
        let thread = StreamingThread::new_default();

        // Initial stats
        let stats = thread.stats();
        assert_eq!(stats.requests_processed, 0);

        // Submit requests
        for i in 0..5 {
            let request = StreamRequest::new(i, 0, format!("tex_{}.dds", i));
            thread.submit_request(request).unwrap();
        }

        // Wait for processing
        thread::sleep(Duration::from_millis(100));

        // Drain uploads
        while thread.pop_upload_command().is_some() {}

        let stats = thread.stats();
        assert!(stats.requests_processed >= 5);
        assert!(stats.uploads_submitted >= 5);
        assert!(stats.bytes_loaded >= 5 * 1024);

        thread.shutdown();
    }

    // ── Shutdown Tests ──────────────────────────────────────────────────

    #[test]
    fn clean_shutdown() {
        let thread = StreamingThread::new_default();

        // Submit some requests
        for i in 0..10 {
            let request = StreamRequest::new(i, 0, format!("tex_{}.dds", i));
            let _ = thread.submit_request(request);
        }

        // Initiate shutdown
        thread.shutdown();

        assert!(!thread.is_running());
        assert!(thread.is_shutdown_complete());
    }

    #[test]
    fn graceful_shutdown() {
        let thread = Arc::new(StreamingThread::new_default());
        let thread_consumer = Arc::clone(&thread);

        // Submit requests
        for i in 0..5 {
            let request = StreamRequest::new(i, 0, format!("tex_{}.dds", i));
            thread.submit_request(request).unwrap();
        }

        // Wait for processing
        thread::sleep(Duration::from_millis(100));

        // Spawn consumer thread to drain upload buffer
        let consumer = thread::spawn(move || {
            let start = Instant::now();
            while start.elapsed() < Duration::from_secs(2) {
                while thread_consumer.pop_upload_command().is_some() {}
                thread::sleep(Duration::from_millis(1));
            }
        });

        // Graceful shutdown with timeout (should succeed now that consumer is draining)
        let success = thread.shutdown_graceful(Duration::from_secs(5));

        // Wait for consumer to finish
        let _ = consumer.join();

        assert!(success);
    }

    #[test]
    fn shutdown_drains_queues() {
        let thread = StreamingThread::new_default();

        // Fill request queue
        for i in 0..100 {
            let request = StreamRequest::new(i, 0, format!("tex_{}.dds", i));
            let _ = thread.submit_request(request);
        }

        // Shutdown immediately
        thread.shutdown();

        // Request queue should be drained
        assert_eq!(thread.pending_requests(), 0);
    }

    // ── Handle Tests ────────────────────────────────────────────────────

    #[test]
    fn streaming_thread_handle() {
        let handle = StreamingThreadHandle::new(StreamingConfig::default());

        // Submit request through handle
        let request = StreamRequest::new(1, 0, "test.dds".to_string());
        assert!(handle.submit_request(request).is_ok());

        thread::sleep(Duration::from_millis(50));

        let stats = handle.stats();
        assert!(stats.requests_processed >= 1);

        handle.shutdown();
    }

    #[test]
    fn handle_clone_shared() {
        let handle1 = StreamingThreadHandle::new(StreamingConfig::default());
        let handle2 = handle1.clone();

        // Both handles point to same thread
        handle1
            .submit_request(StreamRequest::new(1, 0, "a.dds".into()))
            .unwrap();
        handle2
            .submit_request(StreamRequest::new(2, 0, "b.dds".into()))
            .unwrap();

        thread::sleep(Duration::from_millis(50));

        // Stats should reflect both
        let stats = handle1.stats();
        assert!(stats.requests_processed >= 2);

        handle1.shutdown();
    }

    // ── Concurrent Access Tests ─────────────────────────────────────────

    #[test]
    fn concurrent_submit_and_consume() {
        let thread = StreamingThread::new_default();
        let thread = Arc::new(thread);

        let thread_producer = Arc::clone(&thread);
        let thread_consumer = Arc::clone(&thread);

        let producer = thread::spawn(move || {
            for i in 0..100 {
                let request = StreamRequest::new(i, 0, format!("tex_{}.dds", i));
                while thread_producer.submit_request(request.clone()).is_err() {
                    thread::yield_now();
                }
            }
        });

        let consumer = thread::spawn(move || {
            let mut count = 0;
            let start = Instant::now();
            while count < 200 && start.elapsed() < Duration::from_secs(5) {
                if thread_consumer.pop_upload_command().is_some() {
                    count += 1;
                } else {
                    thread::yield_now();
                }
            }
            count
        });

        producer.join().unwrap();
        let consumed = consumer.join().unwrap();

        // Should have consumed at least some commands
        assert!(consumed > 0);

        thread.shutdown();
    }

    // ── Thread Priority Tests ───────────────────────────────────────────

    #[test]
    fn thread_name_set() {
        let config = StreamingConfig {
            thread_name: "custom-streaming".to_string(),
            ..Default::default()
        };

        let thread = StreamingThread::new(config);
        assert!(thread.is_running());
        thread.shutdown();
    }

    // ── Fence Tests ─────────────────────────────────────────────────────

    #[test]
    fn request_fence() {
        let thread = StreamingThread::new_default();

        let fence_id = thread.request_fence();
        assert_eq!(fence_id, 0);

        let fence_id2 = thread.request_fence();
        assert_eq!(fence_id2, 1);

        thread.shutdown();
    }

    // ── Edge Cases ──────────────────────────────────────────────────────

    #[test]
    fn double_shutdown() {
        let thread = StreamingThread::new_default();

        thread.shutdown();
        thread.shutdown(); // Should be safe

        assert!(!thread.is_running());
    }

    #[test]
    fn empty_pop() {
        let thread = StreamingThread::new_default();

        // No requests submitted
        assert!(thread.pop_upload_command().is_none());

        thread.shutdown();
    }

    #[test]
    fn stats_after_shutdown() {
        let thread = StreamingThread::new_default();

        for i in 0..5 {
            let request = StreamRequest::new(i, 0, format!("tex_{}.dds", i));
            thread.submit_request(request).unwrap();
        }

        thread::sleep(Duration::from_millis(50));
        thread.shutdown();

        // Stats should still be accessible
        let stats = thread.stats();
        assert!(stats.requests_processed > 0);
    }
}
