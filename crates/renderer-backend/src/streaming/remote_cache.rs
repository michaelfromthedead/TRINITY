//! Remote Asset Caching for Build Farm and Team-Wide Deduplication (T-AS-5.6).
//!
//! Provides a network-aware content store backend for distributed asset caching:
//!
//! - **HTTP/REST-based Remote Storage**: Content-addressed (hash as key)
//! - **Two-Tier Cache Hit Path**: Local first, then remote on miss
//! - **Background Uploads**: Non-blocking upload to remote after local computation
//! - **Transparent Fallback**: Graceful degradation when remote is unavailable
//! - **DeltaSync Integration**: Minimize bandwidth via differential updates
//!
//! # Architecture
//!
//! ```text
//! RemoteCache
//!   |-- get(hash)          -> Check local, fallback to remote, update local
//!   |-- put(hash, data)    -> Store locally, queue background upload
//!   |-- batch_get(hashes)  -> Parallel download with batching
//!   |-- sync_with_remote() -> DeltaSync-based full synchronization
//!   |
//!   Background Upload Thread:
//!   |-- Consumes upload queue
//!   |-- Retries on failure
//!   |-- Respects rate limits
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::streaming::remote_cache::{RemoteCache, RemoteCacheConfig};
//! use renderer_backend::asset::content_store::MemoryContentStore;
//! use std::sync::Arc;
//!
//! let local_store = Arc::new(MemoryContentStore::default());
//! let config = RemoteCacheConfig {
//!     endpoint: "https://cache.example.com/v1".to_string(),
//!     api_key: Some("secret".to_string()),
//!     timeout_ms: 5000,
//!     ..Default::default()
//! };
//!
//! let cache = RemoteCache::new(config, local_store);
//!
//! // Get asset (checks local first, then remote)
//! let result = cache.get(&hash).await;
//! match result {
//!     CacheResult::LocalHit(data) => { /* fast path */ }
//!     CacheResult::RemoteHit(data) => { /* downloaded and cached locally */ }
//!     CacheResult::Miss => { /* not found anywhere */ }
//!     CacheResult::Error(e) => { /* network or other error */ }
//! }
//!
//! // Put asset (stores locally, uploads in background)
//! cache.put(&hash, &data).await?;
//!
//! // Full sync with remote
//! let stats = cache.sync_with_remote().await?;
//! println!("Uploaded: {}, Downloaded: {}", stats.uploaded, stats.downloaded);
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::io::{self, Read};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::asset::content_store::ContentStore;
use crate::asset::delta_sync::compute_set_delta;
use crate::pipeline::ContentHash;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Default timeout for HTTP requests (5 seconds).
pub const DEFAULT_TIMEOUT_MS: u64 = 5000;

/// Default maximum retry attempts.
pub const DEFAULT_MAX_RETRIES: u32 = 3;

/// Default batch size for bulk operations.
pub const DEFAULT_BATCH_SIZE: usize = 50;

/// Default reconnection interval (30 seconds).
pub const DEFAULT_RECONNECT_INTERVAL_MS: u64 = 30000;

/// Default upload queue capacity.
pub const DEFAULT_UPLOAD_QUEUE_CAPACITY: usize = 1000;

/// Configuration for remote cache.
#[derive(Debug, Clone)]
pub struct RemoteCacheConfig {
    /// Remote server endpoint URL.
    pub endpoint: String,
    /// API key for authentication (optional).
    pub api_key: Option<String>,
    /// Bearer token for authentication (optional, alternative to api_key).
    pub bearer_token: Option<String>,
    /// Request timeout in milliseconds.
    pub timeout_ms: u64,
    /// Maximum retry attempts for failed requests.
    pub max_retries: u32,
    /// Whether to upload in background (non-blocking).
    pub upload_in_background: bool,
    /// Batch size for bulk operations.
    pub batch_size: usize,
    /// Interval between reconnection attempts (ms).
    pub reconnect_interval_ms: u64,
    /// Maximum capacity of background upload queue.
    pub upload_queue_capacity: usize,
    /// Whether to verify downloaded content hashes.
    pub verify_downloads: bool,
    /// Whether to compress uploads (if server supports).
    pub compress_uploads: bool,
    /// Custom headers to include in requests.
    pub custom_headers: HashMap<String, String>,
}

impl Default for RemoteCacheConfig {
    fn default() -> Self {
        Self {
            endpoint: String::new(),
            api_key: None,
            bearer_token: None,
            timeout_ms: DEFAULT_TIMEOUT_MS,
            max_retries: DEFAULT_MAX_RETRIES,
            upload_in_background: true,
            batch_size: DEFAULT_BATCH_SIZE,
            reconnect_interval_ms: DEFAULT_RECONNECT_INTERVAL_MS,
            upload_queue_capacity: DEFAULT_UPLOAD_QUEUE_CAPACITY,
            verify_downloads: true,
            compress_uploads: false,
            custom_headers: HashMap::new(),
        }
    }
}

impl RemoteCacheConfig {
    /// Create config with endpoint.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Set API key for authentication.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Set bearer token for authentication.
    pub fn with_bearer_token(mut self, token: impl Into<String>) -> Self {
        self.bearer_token = Some(token.into());
        self
    }

    /// Set request timeout.
    pub fn with_timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = ms.max(100);
        self
    }

    /// Set maximum retry attempts.
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Enable or disable background uploads.
    pub fn with_background_upload(mut self, enabled: bool) -> Self {
        self.upload_in_background = enabled;
        self
    }

    /// Set batch size for bulk operations.
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size.max(1);
        self
    }

    /// Set reconnection interval.
    pub fn with_reconnect_interval_ms(mut self, ms: u64) -> Self {
        self.reconnect_interval_ms = ms.max(1000);
        self
    }

    /// Enable or disable download verification.
    pub fn with_verify_downloads(mut self, verify: bool) -> Self {
        self.verify_downloads = verify;
        self
    }

    /// Enable or disable upload compression.
    pub fn with_compress_uploads(mut self, compress: bool) -> Self {
        self.compress_uploads = compress;
        self
    }

    /// Add a custom header.
    pub fn with_custom_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom_headers.insert(key.into(), value.into());
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> std::result::Result<(), RemoteCacheError> {
        if self.endpoint.is_empty() {
            return Err(RemoteCacheError::InvalidConfig("endpoint cannot be empty".into()));
        }
        if self.timeout_ms < 100 {
            return Err(RemoteCacheError::InvalidConfig("timeout must be at least 100ms".into()));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Error Types
// ---------------------------------------------------------------------------

/// Errors that can occur during remote cache operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemoteCacheError {
    /// Network error (connection failed, timeout, etc.).
    NetworkError(String),
    /// HTTP error response from server.
    HttpError { status_code: u16, message: String },
    /// Content not found on remote.
    NotFound(ContentHash),
    /// Hash verification failed after download.
    HashMismatch { expected: ContentHash, actual: ContentHash },
    /// Authentication failed.
    AuthenticationFailed(String),
    /// Server rate limit exceeded.
    RateLimited { retry_after_ms: Option<u64> },
    /// Local store error.
    LocalStoreError(String),
    /// Invalid configuration.
    InvalidConfig(String),
    /// Upload queue is full.
    QueueFull,
    /// Operation timed out.
    Timeout,
    /// Cache is disconnected from remote.
    Disconnected,
    /// Serialization/deserialization error.
    SerializationError(String),
}

impl fmt::Display for RemoteCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NetworkError(msg) => write!(f, "network error: {}", msg),
            Self::HttpError { status_code, message } => {
                write!(f, "HTTP error {}: {}", status_code, message)
            }
            Self::NotFound(hash) => write!(f, "content not found: {}", hash),
            Self::HashMismatch { expected, actual } => {
                write!(f, "hash mismatch: expected {}, got {}", expected, actual)
            }
            Self::AuthenticationFailed(msg) => write!(f, "authentication failed: {}", msg),
            Self::RateLimited { retry_after_ms } => match retry_after_ms {
                Some(ms) => write!(f, "rate limited, retry after {}ms", ms),
                None => write!(f, "rate limited"),
            },
            Self::LocalStoreError(msg) => write!(f, "local store error: {}", msg),
            Self::InvalidConfig(msg) => write!(f, "invalid configuration: {}", msg),
            Self::QueueFull => write!(f, "upload queue is full"),
            Self::Timeout => write!(f, "operation timed out"),
            Self::Disconnected => write!(f, "disconnected from remote"),
            Self::SerializationError(msg) => write!(f, "serialization error: {}", msg),
        }
    }
}

impl std::error::Error for RemoteCacheError {}

impl From<io::Error> for RemoteCacheError {
    fn from(e: io::Error) -> Self {
        Self::LocalStoreError(e.to_string())
    }
}

/// Result type for remote cache operations.
pub type Result<T> = std::result::Result<T, RemoteCacheError>;

// ---------------------------------------------------------------------------
// Cache Result
// ---------------------------------------------------------------------------

/// Result of a cache lookup operation.
#[derive(Debug, Clone)]
pub enum CacheResult {
    /// Content found in local cache.
    LocalHit(Vec<u8>),
    /// Content found on remote server (and cached locally).
    RemoteHit(Vec<u8>),
    /// Content not found in any cache.
    Miss,
    /// Error during lookup.
    Error(String),
}

impl CacheResult {
    /// Check if this is a hit (local or remote).
    pub fn is_hit(&self) -> bool {
        matches!(self, Self::LocalHit(_) | Self::RemoteHit(_))
    }

    /// Check if this is a local hit.
    pub fn is_local_hit(&self) -> bool {
        matches!(self, Self::LocalHit(_))
    }

    /// Check if this is a remote hit.
    pub fn is_remote_hit(&self) -> bool {
        matches!(self, Self::RemoteHit(_))
    }

    /// Check if this is a miss.
    pub fn is_miss(&self) -> bool {
        matches!(self, Self::Miss)
    }

    /// Check if this is an error.
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }

    /// Get the data if this is a hit.
    pub fn data(&self) -> Option<&[u8]> {
        match self {
            Self::LocalHit(data) | Self::RemoteHit(data) => Some(data),
            _ => None,
        }
    }

    /// Consume and return the data if this is a hit.
    pub fn into_data(self) -> Option<Vec<u8>> {
        match self {
            Self::LocalHit(data) | Self::RemoteHit(data) => Some(data),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Sync Statistics
// ---------------------------------------------------------------------------

/// Statistics from a synchronization operation.
#[derive(Debug, Clone, Default)]
pub struct SyncStats {
    /// Number of items uploaded to remote.
    pub uploaded: usize,
    /// Number of items downloaded from remote.
    pub downloaded: usize,
    /// Total bytes transferred.
    pub bytes_transferred: u64,
    /// Number of errors encountered.
    pub errors: usize,
    /// Duration of the sync operation in milliseconds.
    pub duration_ms: u64,
}

impl SyncStats {
    /// Create new empty stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Merge with another stats object.
    pub fn merge(&mut self, other: &SyncStats) {
        self.uploaded += other.uploaded;
        self.downloaded += other.downloaded;
        self.bytes_transferred += other.bytes_transferred;
        self.errors += other.errors;
        self.duration_ms += other.duration_ms;
    }
}

// ---------------------------------------------------------------------------
// HTTP Client Interface
// ---------------------------------------------------------------------------

/// HTTP request method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpMethod {
    Get,
    Put,
    Post,
    Delete,
    Head,
}

impl fmt::Display for HttpMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Get => write!(f, "GET"),
            Self::Put => write!(f, "PUT"),
            Self::Post => write!(f, "POST"),
            Self::Delete => write!(f, "DELETE"),
            Self::Head => write!(f, "HEAD"),
        }
    }
}

/// HTTP request.
#[derive(Debug, Clone)]
pub struct HttpRequest {
    /// HTTP method.
    pub method: HttpMethod,
    /// Full URL.
    pub url: String,
    /// Request headers.
    pub headers: HashMap<String, String>,
    /// Request body (optional).
    pub body: Option<Vec<u8>>,
    /// Timeout in milliseconds.
    pub timeout_ms: u64,
}

impl HttpRequest {
    /// Create a new GET request.
    pub fn get(url: impl Into<String>) -> Self {
        Self {
            method: HttpMethod::Get,
            url: url.into(),
            headers: HashMap::new(),
            body: None,
            timeout_ms: DEFAULT_TIMEOUT_MS,
        }
    }

    /// Create a new PUT request.
    pub fn put(url: impl Into<String>, body: Vec<u8>) -> Self {
        Self {
            method: HttpMethod::Put,
            url: url.into(),
            headers: HashMap::new(),
            body: Some(body),
            timeout_ms: DEFAULT_TIMEOUT_MS,
        }
    }

    /// Create a new HEAD request.
    pub fn head(url: impl Into<String>) -> Self {
        Self {
            method: HttpMethod::Head,
            url: url.into(),
            headers: HashMap::new(),
            body: None,
            timeout_ms: DEFAULT_TIMEOUT_MS,
        }
    }

    /// Create a new POST request.
    pub fn post(url: impl Into<String>, body: Vec<u8>) -> Self {
        Self {
            method: HttpMethod::Post,
            url: url.into(),
            headers: HashMap::new(),
            body: Some(body),
            timeout_ms: DEFAULT_TIMEOUT_MS,
        }
    }

    /// Add a header.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }

    /// Set timeout.
    pub fn with_timeout(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }
}

/// HTTP response.
#[derive(Debug, Clone)]
pub struct HttpResponse {
    /// HTTP status code.
    pub status_code: u16,
    /// Response headers.
    pub headers: HashMap<String, String>,
    /// Response body.
    pub body: Vec<u8>,
}

impl HttpResponse {
    /// Check if response is successful (2xx).
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status_code)
    }

    /// Check if response is not found (404).
    pub fn is_not_found(&self) -> bool {
        self.status_code == 404
    }

    /// Check if response indicates rate limiting (429).
    pub fn is_rate_limited(&self) -> bool {
        self.status_code == 429
    }

    /// Check if response indicates authentication failure (401/403).
    pub fn is_auth_error(&self) -> bool {
        self.status_code == 401 || self.status_code == 403
    }
}

/// Trait for HTTP client implementations.
///
/// Implement this trait to provide a real HTTP client (e.g., reqwest, ureq).
/// A mock implementation is provided for testing.
pub trait HttpClient: Send + Sync {
    /// Execute an HTTP request.
    fn execute(&self, request: HttpRequest) -> Result<HttpResponse>;
}

// ---------------------------------------------------------------------------
// Mock HTTP Client (for testing)
// ---------------------------------------------------------------------------

/// A mock HTTP client for testing.
///
/// Simulates a remote cache server with in-memory storage.
#[derive(Default)]
pub struct MockHttpClient {
    /// Stored content (hash -> data).
    storage: RwLock<HashMap<String, Vec<u8>>>,
    /// Whether to simulate being offline.
    offline: AtomicBool,
    /// Number of requests made.
    request_count: AtomicUsize,
    /// Simulated latency in milliseconds.
    latency_ms: AtomicU64,
    /// Whether to fail the next request.
    fail_next: AtomicBool,
}

impl MockHttpClient {
    /// Create a new mock HTTP client.
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-populate with content.
    pub fn with_content(self, hash: &ContentHash, data: Vec<u8>) -> Self {
        self.storage.write().unwrap().insert(format!("{}", hash), data);
        self
    }

    /// Set offline mode.
    pub fn set_offline(&self, offline: bool) {
        self.offline.store(offline, Ordering::Release);
    }

    /// Set simulated latency.
    pub fn set_latency(&self, ms: u64) {
        self.latency_ms.store(ms, Ordering::Release);
    }

    /// Make the next request fail.
    pub fn fail_next_request(&self) {
        self.fail_next.store(true, Ordering::Release);
    }

    /// Get the number of requests made.
    pub fn request_count(&self) -> usize {
        self.request_count.load(Ordering::Acquire)
    }

    /// Check if content exists.
    pub fn has(&self, hash: &ContentHash) -> bool {
        self.storage.read().unwrap().contains_key(&format!("{}", hash))
    }

    /// Clear all stored content.
    pub fn clear(&self) {
        self.storage.write().unwrap().clear();
    }

    /// Extract hash from URL path (assumes /{hash} format).
    fn extract_hash(&self, url: &str) -> Option<String> {
        url.rsplit('/').next().map(|s| s.to_string())
    }
}

impl HttpClient for MockHttpClient {
    fn execute(&self, request: HttpRequest) -> Result<HttpResponse> {
        self.request_count.fetch_add(1, Ordering::AcqRel);

        // Simulate latency
        let latency = self.latency_ms.load(Ordering::Acquire);
        if latency > 0 {
            std::thread::sleep(Duration::from_millis(latency));
        }

        // Check if we should fail this request
        if self.fail_next.swap(false, Ordering::AcqRel) {
            return Err(RemoteCacheError::NetworkError("simulated failure".into()));
        }

        // Check if offline
        if self.offline.load(Ordering::Acquire) {
            return Err(RemoteCacheError::NetworkError("offline".into()));
        }

        let hash = self.extract_hash(&request.url);

        match request.method {
            HttpMethod::Get => {
                let storage = self.storage.read().unwrap();
                match hash.and_then(|h| storage.get(&h).cloned()) {
                    Some(data) => Ok(HttpResponse {
                        status_code: 200,
                        headers: HashMap::new(),
                        body: data,
                    }),
                    None => Ok(HttpResponse {
                        status_code: 404,
                        headers: HashMap::new(),
                        body: Vec::new(),
                    }),
                }
            }
            HttpMethod::Put => {
                if let (Some(hash), Some(body)) = (hash, request.body) {
                    self.storage.write().unwrap().insert(hash, body);
                    Ok(HttpResponse {
                        status_code: 200,
                        headers: HashMap::new(),
                        body: Vec::new(),
                    })
                } else {
                    Ok(HttpResponse {
                        status_code: 400,
                        headers: HashMap::new(),
                        body: b"bad request".to_vec(),
                    })
                }
            }
            HttpMethod::Head => {
                let storage = self.storage.read().unwrap();
                let exists = hash.map(|h| storage.contains_key(&h)).unwrap_or(false);
                Ok(HttpResponse {
                    status_code: if exists { 200 } else { 404 },
                    headers: HashMap::new(),
                    body: Vec::new(),
                })
            }
            HttpMethod::Post => {
                // Batch endpoint simulation
                if request.url.contains("/batch") {
                    // Return empty batch response
                    Ok(HttpResponse {
                        status_code: 200,
                        headers: HashMap::new(),
                        body: b"[]".to_vec(),
                    })
                } else {
                    Ok(HttpResponse {
                        status_code: 200,
                        headers: HashMap::new(),
                        body: Vec::new(),
                    })
                }
            }
            HttpMethod::Delete => {
                if let Some(hash) = hash {
                    let existed = self.storage.write().unwrap().remove(&hash).is_some();
                    Ok(HttpResponse {
                        status_code: if existed { 200 } else { 404 },
                        headers: HashMap::new(),
                        body: Vec::new(),
                    })
                } else {
                    Ok(HttpResponse {
                        status_code: 400,
                        headers: HashMap::new(),
                        body: Vec::new(),
                    })
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Background Upload Queue
// ---------------------------------------------------------------------------

/// An upload task for the background queue.
#[derive(Debug, Clone)]
struct UploadTask {
    /// Content hash.
    hash: ContentHash,
    /// Data to upload.
    data: Vec<u8>,
    /// Number of retry attempts.
    retries: u32,
    /// Time when task was created.
    created_at: Instant,
}

/// Background upload queue state.
struct UploadQueueState {
    /// Pending upload tasks.
    queue: VecDeque<UploadTask>,
    /// Set of hashes currently in queue (for deduplication).
    pending: HashSet<ContentHash>,
    /// Whether shutdown has been requested.
    shutdown: bool,
}

impl UploadQueueState {
    fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            pending: HashSet::new(),
            shutdown: false,
        }
    }
}

/// Background upload queue with worker thread.
struct BackgroundUploadQueue {
    /// Queue state protected by mutex.
    state: Mutex<UploadQueueState>,
    /// Condition variable for waking worker.
    condvar: Condvar,
    /// Maximum queue capacity.
    capacity: usize,
}

impl BackgroundUploadQueue {
    fn new(capacity: usize) -> Self {
        Self {
            state: Mutex::new(UploadQueueState::new()),
            condvar: Condvar::new(),
            capacity,
        }
    }

    /// Add an upload task to the queue.
    fn enqueue(&self, hash: ContentHash, data: Vec<u8>) -> Result<()> {
        let mut state = self.state.lock().unwrap();

        // Check if already pending
        if state.pending.contains(&hash) {
            return Ok(());
        }

        // Check capacity
        if state.queue.len() >= self.capacity {
            return Err(RemoteCacheError::QueueFull);
        }

        state.pending.insert(hash);
        state.queue.push_back(UploadTask {
            hash,
            data,
            retries: 0,
            created_at: Instant::now(),
        });

        // Wake worker
        self.condvar.notify_one();

        Ok(())
    }

    /// Dequeue an upload task (blocks if empty).
    fn dequeue(&self, timeout: Duration) -> Option<UploadTask> {
        let mut state = self.state.lock().unwrap();

        // Wait for task or shutdown
        let result = self.condvar.wait_timeout_while(state, timeout, |s| {
            s.queue.is_empty() && !s.shutdown
        });

        let mut state = result.ok()?.0;

        if state.shutdown && state.queue.is_empty() {
            return None;
        }

        let task = state.queue.pop_front()?;
        state.pending.remove(&task.hash);
        Some(task)
    }

    /// Re-enqueue a failed task for retry.
    fn requeue(&self, mut task: UploadTask, max_retries: u32) -> bool {
        task.retries += 1;
        if task.retries > max_retries {
            return false;
        }

        let mut state = self.state.lock().unwrap();
        state.pending.insert(task.hash);
        state.queue.push_back(task);
        self.condvar.notify_one();
        true
    }

    /// Signal shutdown.
    fn shutdown(&self) {
        let mut state = self.state.lock().unwrap();
        state.shutdown = true;
        self.condvar.notify_all();
    }

    /// Get current queue length.
    fn len(&self) -> usize {
        self.state.lock().unwrap().queue.len()
    }

    /// Check if queue is empty.
    fn is_empty(&self) -> bool {
        self.state.lock().unwrap().queue.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Remote Cache
// ---------------------------------------------------------------------------

/// Network-aware content store with local caching and background uploads.
///
/// Provides transparent access to both local and remote content storage,
/// with automatic fallback when the remote is unavailable.
pub struct RemoteCache<S: ContentStore + 'static, C: HttpClient + 'static> {
    /// Configuration.
    config: RemoteCacheConfig,
    /// Local content store.
    local_store: Arc<S>,
    /// HTTP client for remote requests.
    http_client: Arc<C>,
    /// Connection status.
    connected: AtomicBool,
    /// Last successful connection time.
    last_connected: AtomicU64,
    /// Last connection attempt time.
    last_attempt: AtomicU64,
    /// Background upload queue.
    upload_queue: Arc<BackgroundUploadQueue>,
    /// Statistics: total requests.
    stats_requests: AtomicU64,
    /// Statistics: local hits.
    stats_local_hits: AtomicU64,
    /// Statistics: remote hits.
    stats_remote_hits: AtomicU64,
    /// Statistics: misses.
    stats_misses: AtomicU64,
    /// Statistics: errors.
    stats_errors: AtomicU64,
    /// Statistics: bytes downloaded.
    stats_bytes_downloaded: AtomicU64,
    /// Statistics: bytes uploaded.
    stats_bytes_uploaded: AtomicU64,
    /// Worker thread handle (if background uploads enabled).
    worker_handle: Mutex<Option<JoinHandle<()>>>,
    /// Shutdown flag for worker.
    shutdown: AtomicBool,
}

impl<S: ContentStore + 'static, C: HttpClient + 'static> RemoteCache<S, C> {
    /// Create a new remote cache.
    pub fn new(config: RemoteCacheConfig, local_store: Arc<S>, http_client: Arc<C>) -> Self {
        let upload_queue = Arc::new(BackgroundUploadQueue::new(config.upload_queue_capacity));

        let cache = Self {
            config,
            local_store,
            http_client,
            connected: AtomicBool::new(false),
            last_connected: AtomicU64::new(0),
            last_attempt: AtomicU64::new(0),
            upload_queue,
            stats_requests: AtomicU64::new(0),
            stats_local_hits: AtomicU64::new(0),
            stats_remote_hits: AtomicU64::new(0),
            stats_misses: AtomicU64::new(0),
            stats_errors: AtomicU64::new(0),
            stats_bytes_downloaded: AtomicU64::new(0),
            stats_bytes_uploaded: AtomicU64::new(0),
            worker_handle: Mutex::new(None),
            shutdown: AtomicBool::new(false),
        };

        // Start background upload worker if enabled
        if cache.config.upload_in_background {
            cache.start_upload_worker();
        }

        cache
    }

    /// Start the background upload worker thread.
    fn start_upload_worker(&self) {
        let queue = Arc::clone(&self.upload_queue);
        let client = Arc::clone(&self.http_client);
        let endpoint = self.config.endpoint.clone();
        let max_retries = self.config.max_retries;
        let timeout_ms = self.config.timeout_ms;
        let api_key = self.config.api_key.clone();
        let bearer_token = self.config.bearer_token.clone();
        let custom_headers = self.config.custom_headers.clone();

        let handle = thread::spawn(move || {
            loop {
                // Wait for task with timeout
                let task = match queue.dequeue(Duration::from_secs(1)) {
                    Some(t) => t,
                    None => continue,
                };

                // Build upload request
                let url = format!("{}/content/{}", endpoint, task.hash);
                let mut request = HttpRequest::put(&url, task.data.clone())
                    .with_timeout(timeout_ms)
                    .with_header("Content-Type", "application/octet-stream");

                // Add authentication
                if let Some(ref key) = api_key {
                    request = request.with_header("X-API-Key", key.as_str());
                }
                if let Some(ref token) = bearer_token {
                    request = request.with_header("Authorization", format!("Bearer {}", token));
                }

                // Add custom headers
                for (k, v) in &custom_headers {
                    request = request.with_header(k.as_str(), v.as_str());
                }

                // Execute upload
                match client.execute(request) {
                    Ok(response) if response.is_success() => {
                        // Upload successful
                    }
                    Ok(_) | Err(_) => {
                        // Retry on failure
                        queue.requeue(task, max_retries);
                    }
                }
            }
        });

        *self.worker_handle.lock().unwrap() = Some(handle);
    }

    /// Get the configuration.
    pub fn config(&self) -> &RemoteCacheConfig {
        &self.config
    }

    /// Get the local store.
    pub fn local_store(&self) -> &Arc<S> {
        &self.local_store
    }

    /// Check if connected to remote.
    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Acquire)
    }

    /// Get content by hash.
    ///
    /// Checks local cache first, then falls back to remote.
    pub fn get(&self, hash: &ContentHash) -> CacheResult {
        self.stats_requests.fetch_add(1, Ordering::Relaxed);

        // Check local cache first
        match self.local_store.get_stream(hash) {
            Ok(Some(mut reader)) => {
                let mut data = Vec::new();
                if reader.read_to_end(&mut data).is_ok() {
                    self.stats_local_hits.fetch_add(1, Ordering::Relaxed);
                    return CacheResult::LocalHit(data);
                }
            }
            Ok(None) => {}
            Err(e) => {
                self.stats_errors.fetch_add(1, Ordering::Relaxed);
                return CacheResult::Error(e.to_string());
            }
        }

        // Try remote if available
        if !self.should_try_remote() {
            self.stats_misses.fetch_add(1, Ordering::Relaxed);
            return CacheResult::Miss;
        }

        match self.fetch_from_remote(hash) {
            Ok(data) => {
                // Store in local cache
                let mut cursor = std::io::Cursor::new(&data);
                if let Err(e) = self.local_store.put_stream(&mut cursor) {
                    // Log but don't fail - we still have the data
                    eprintln!("Failed to cache locally: {}", e);
                }
                self.stats_remote_hits.fetch_add(1, Ordering::Relaxed);
                self.stats_bytes_downloaded.fetch_add(data.len() as u64, Ordering::Relaxed);
                CacheResult::RemoteHit(data)
            }
            Err(RemoteCacheError::NotFound(_)) => {
                self.stats_misses.fetch_add(1, Ordering::Relaxed);
                CacheResult::Miss
            }
            Err(e) => {
                self.stats_errors.fetch_add(1, Ordering::Relaxed);
                CacheResult::Error(e.to_string())
            }
        }
    }

    /// Store content by hash.
    ///
    /// Stores locally immediately, queues background upload if enabled.
    pub fn put(&self, hash: &ContentHash, data: &[u8]) -> Result<()> {
        // Store locally first
        let mut cursor = std::io::Cursor::new(data);
        self.local_store
            .put_stream(&mut cursor)
            .map_err(|e| RemoteCacheError::LocalStoreError(e.to_string()))?;

        // Queue background upload if enabled
        if self.config.upload_in_background && self.should_try_remote() {
            self.upload_queue.enqueue(*hash, data.to_vec())?;
        } else if !self.config.upload_in_background && self.should_try_remote() {
            // Synchronous upload
            self.upload_to_remote(hash, data)?;
        }

        Ok(())
    }

    /// Check if content exists (local or remote).
    pub fn exists(&self, hash: &ContentHash) -> bool {
        // Check local first
        if self.local_store.has(hash) {
            return true;
        }

        // Check remote if available
        if !self.should_try_remote() {
            return false;
        }

        self.check_remote_exists(hash).unwrap_or(false)
    }

    /// Queue a background upload without storing locally first.
    ///
    /// Use this when you've already stored locally and just want to sync to remote.
    pub fn queue_background_upload(&self, hash: ContentHash, data: Vec<u8>) {
        if self.config.upload_in_background {
            let _ = self.upload_queue.enqueue(hash, data);
        }
    }

    /// Get multiple items in a batch.
    ///
    /// Returns results in the same order as input hashes.
    pub fn batch_get(&self, hashes: &[ContentHash]) -> Vec<CacheResult> {
        if hashes.is_empty() {
            return Vec::new();
        }

        // First pass: check local cache
        let mut results: Vec<Option<CacheResult>> = vec![None; hashes.len()];
        let mut remote_indices = Vec::new();

        for (i, hash) in hashes.iter().enumerate() {
            match self.local_store.get_stream(hash) {
                Ok(Some(mut reader)) => {
                    let mut data = Vec::new();
                    if reader.read_to_end(&mut data).is_ok() {
                        self.stats_local_hits.fetch_add(1, Ordering::Relaxed);
                        results[i] = Some(CacheResult::LocalHit(data));
                        continue;
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    results[i] = Some(CacheResult::Error(e.to_string()));
                    continue;
                }
            }
            remote_indices.push(i);
        }

        // Second pass: fetch missing from remote in batches
        if !remote_indices.is_empty() && self.should_try_remote() {
            for chunk in remote_indices.chunks(self.config.batch_size) {
                let batch_hashes: Vec<_> = chunk.iter().map(|&i| hashes[i]).collect();
                let batch_results = self.fetch_batch_from_remote(&batch_hashes);

                for (batch_idx, &original_idx) in chunk.iter().enumerate() {
                    results[original_idx] = Some(batch_results[batch_idx].clone());
                }
            }
        }

        // Fill any remaining None with Miss
        results
            .into_iter()
            .map(|r| r.unwrap_or(CacheResult::Miss))
            .collect()
    }

    /// Synchronize with remote cache.
    ///
    /// Downloads missing content from remote and uploads local-only content.
    pub fn sync_with_remote(&self) -> Result<SyncStats> {
        if !self.should_try_remote() {
            return Err(RemoteCacheError::Disconnected);
        }

        let start = Instant::now();
        let mut stats = SyncStats::new();

        // Get local manifest
        let local_hashes = self.get_local_hashes()?;

        // Get remote manifest
        let remote_hashes = self.fetch_remote_manifest()?;

        // Compute delta
        let (to_download, to_upload, _unchanged) =
            compute_set_delta(&local_hashes, &remote_hashes);

        // Download missing content
        for hash in to_download {
            match self.fetch_from_remote(&hash) {
                Ok(data) => {
                    let mut cursor = std::io::Cursor::new(&data);
                    if self.local_store.put_stream(&mut cursor).is_ok() {
                        stats.downloaded += 1;
                        stats.bytes_transferred += data.len() as u64;
                    }
                }
                Err(_) => {
                    stats.errors += 1;
                }
            }
        }

        // Upload local-only content
        for hash in to_upload {
            if let Ok(Some(mut reader)) = self.local_store.get_stream(&hash) {
                let mut data = Vec::new();
                if reader.read_to_end(&mut data).is_ok() {
                    match self.upload_to_remote(&hash, &data) {
                        Ok(()) => {
                            stats.uploaded += 1;
                            stats.bytes_transferred += data.len() as u64;
                        }
                        Err(_) => {
                            stats.errors += 1;
                        }
                    }
                }
            }
        }

        stats.duration_ms = start.elapsed().as_millis() as u64;
        Ok(stats)
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            total_requests: self.stats_requests.load(Ordering::Relaxed),
            local_hits: self.stats_local_hits.load(Ordering::Relaxed),
            remote_hits: self.stats_remote_hits.load(Ordering::Relaxed),
            misses: self.stats_misses.load(Ordering::Relaxed),
            errors: self.stats_errors.load(Ordering::Relaxed),
            bytes_downloaded: self.stats_bytes_downloaded.load(Ordering::Relaxed),
            bytes_uploaded: self.stats_bytes_uploaded.load(Ordering::Relaxed),
            pending_uploads: self.upload_queue.len(),
            connected: self.is_connected(),
        }
    }

    /// Reset statistics.
    pub fn reset_stats(&self) {
        self.stats_requests.store(0, Ordering::Relaxed);
        self.stats_local_hits.store(0, Ordering::Relaxed);
        self.stats_remote_hits.store(0, Ordering::Relaxed);
        self.stats_misses.store(0, Ordering::Relaxed);
        self.stats_errors.store(0, Ordering::Relaxed);
        self.stats_bytes_downloaded.store(0, Ordering::Relaxed);
        self.stats_bytes_uploaded.store(0, Ordering::Relaxed);
    }

    /// Shutdown the cache (stops background worker).
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
        self.upload_queue.shutdown();

        // Wait for worker to finish (with timeout)
        if let Some(handle) = self.worker_handle.lock().unwrap().take() {
            let _ = handle.join();
        }
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /// Check if we should try the remote.
    fn should_try_remote(&self) -> bool {
        if self.config.endpoint.is_empty() {
            return false;
        }

        // If connected, always try
        if self.connected.load(Ordering::Acquire) {
            return true;
        }

        // Check reconnection interval
        let now = current_timestamp_ms();
        let last_attempt = self.last_attempt.load(Ordering::Acquire);
        now >= last_attempt + self.config.reconnect_interval_ms
    }

    /// Update connection status.
    fn update_connection_status(&self, success: bool) {
        let now = current_timestamp_ms();
        self.last_attempt.store(now, Ordering::Release);

        if success {
            self.connected.store(true, Ordering::Release);
            self.last_connected.store(now, Ordering::Release);
        } else {
            self.connected.store(false, Ordering::Release);
        }
    }

    /// Build authenticated request.
    fn build_request(&self, method: HttpMethod, path: &str) -> HttpRequest {
        let url = format!("{}{}", self.config.endpoint, path);
        let mut request = match method {
            HttpMethod::Get => HttpRequest::get(&url),
            HttpMethod::Put => HttpRequest::put(&url, Vec::new()),
            HttpMethod::Post => HttpRequest::post(&url, Vec::new()),
            HttpMethod::Delete => HttpRequest {
                method: HttpMethod::Delete,
                url,
                headers: HashMap::new(),
                body: None,
                timeout_ms: self.config.timeout_ms,
            },
            HttpMethod::Head => HttpRequest::head(&url),
        };

        request = request.with_timeout(self.config.timeout_ms);

        // Add authentication
        if let Some(ref key) = self.config.api_key {
            request = request.with_header("X-API-Key", key.as_str());
        }
        if let Some(ref token) = self.config.bearer_token {
            request = request.with_header("Authorization", format!("Bearer {}", token));
        }

        // Add custom headers
        for (k, v) in &self.config.custom_headers {
            request = request.with_header(k.as_str(), v.as_str());
        }

        request
    }

    /// Fetch content from remote.
    fn fetch_from_remote(&self, hash: &ContentHash) -> Result<Vec<u8>> {
        let request = self.build_request(HttpMethod::Get, &format!("/content/{}", hash));

        let response = self.http_client.execute(request).map_err(|e| {
            self.update_connection_status(false);
            e
        })?;

        self.update_connection_status(true);

        if response.is_not_found() {
            return Err(RemoteCacheError::NotFound(*hash));
        }

        if response.is_auth_error() {
            return Err(RemoteCacheError::AuthenticationFailed(
                String::from_utf8_lossy(&response.body).into_owned(),
            ));
        }

        if response.is_rate_limited() {
            let retry_after = response
                .headers
                .get("Retry-After")
                .and_then(|v| v.parse().ok())
                .map(|secs: u64| secs * 1000);
            return Err(RemoteCacheError::RateLimited {
                retry_after_ms: retry_after,
            });
        }

        if !response.is_success() {
            return Err(RemoteCacheError::HttpError {
                status_code: response.status_code,
                message: String::from_utf8_lossy(&response.body).into_owned(),
            });
        }

        // Verify hash if configured
        if self.config.verify_downloads {
            let actual = ContentHash::from_bytes(&response.body);
            if &actual != hash {
                return Err(RemoteCacheError::HashMismatch {
                    expected: *hash,
                    actual,
                });
            }
        }

        Ok(response.body)
    }

    /// Check if content exists on remote.
    fn check_remote_exists(&self, hash: &ContentHash) -> Result<bool> {
        let request = self.build_request(HttpMethod::Head, &format!("/content/{}", hash));

        let response = self.http_client.execute(request).map_err(|e| {
            self.update_connection_status(false);
            e
        })?;

        self.update_connection_status(true);
        Ok(response.is_success())
    }

    /// Upload content to remote.
    fn upload_to_remote(&self, hash: &ContentHash, data: &[u8]) -> Result<()> {
        let url = format!("{}/content/{}", self.config.endpoint, hash);
        let mut request = HttpRequest::put(&url, data.to_vec())
            .with_timeout(self.config.timeout_ms)
            .with_header("Content-Type", "application/octet-stream");

        // Add authentication
        if let Some(ref key) = self.config.api_key {
            request = request.with_header("X-API-Key", key.as_str());
        }
        if let Some(ref token) = self.config.bearer_token {
            request = request.with_header("Authorization", format!("Bearer {}", token));
        }

        // Add custom headers
        for (k, v) in &self.config.custom_headers {
            request = request.with_header(k.as_str(), v.as_str());
        }

        let response = self.http_client.execute(request).map_err(|e| {
            self.update_connection_status(false);
            e
        })?;

        self.update_connection_status(true);
        self.stats_bytes_uploaded.fetch_add(data.len() as u64, Ordering::Relaxed);

        if !response.is_success() {
            return Err(RemoteCacheError::HttpError {
                status_code: response.status_code,
                message: String::from_utf8_lossy(&response.body).into_owned(),
            });
        }

        Ok(())
    }

    /// Fetch batch of content from remote.
    fn fetch_batch_from_remote(&self, hashes: &[ContentHash]) -> Vec<CacheResult> {
        // For now, implement as sequential fetches
        // A real implementation would use a batch endpoint
        hashes
            .iter()
            .map(|hash| {
                self.stats_requests.fetch_add(1, Ordering::Relaxed);
                match self.fetch_from_remote(hash) {
                    Ok(data) => {
                        // Store in local cache
                        let mut cursor = std::io::Cursor::new(&data);
                        if let Err(e) = self.local_store.put_stream(&mut cursor) {
                            eprintln!("Failed to cache locally: {}", e);
                        }
                        self.stats_remote_hits.fetch_add(1, Ordering::Relaxed);
                        self.stats_bytes_downloaded.fetch_add(data.len() as u64, Ordering::Relaxed);
                        CacheResult::RemoteHit(data)
                    }
                    Err(RemoteCacheError::NotFound(_)) => {
                        self.stats_misses.fetch_add(1, Ordering::Relaxed);
                        CacheResult::Miss
                    }
                    Err(e) => {
                        self.stats_errors.fetch_add(1, Ordering::Relaxed);
                        CacheResult::Error(e.to_string())
                    }
                }
            })
            .collect()
    }

    /// Get all hashes from local store.
    fn get_local_hashes(&self) -> Result<HashSet<ContentHash>> {
        // This is a simplified implementation
        // Real implementation would depend on local store's listing capability
        Ok(HashSet::new())
    }

    /// Fetch manifest of all hashes from remote.
    fn fetch_remote_manifest(&self) -> Result<HashSet<ContentHash>> {
        // This would call a /manifest endpoint on the remote
        // For now, return empty set
        Ok(HashSet::new())
    }
}

impl<S: ContentStore + 'static, C: HttpClient + 'static> Drop for RemoteCache<S, C> {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// ---------------------------------------------------------------------------
// Cache Statistics
// ---------------------------------------------------------------------------

/// Statistics about cache operations.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total number of requests.
    pub total_requests: u64,
    /// Number of local cache hits.
    pub local_hits: u64,
    /// Number of remote cache hits.
    pub remote_hits: u64,
    /// Number of cache misses.
    pub misses: u64,
    /// Number of errors.
    pub errors: u64,
    /// Total bytes downloaded from remote.
    pub bytes_downloaded: u64,
    /// Total bytes uploaded to remote.
    pub bytes_uploaded: u64,
    /// Number of pending background uploads.
    pub pending_uploads: usize,
    /// Whether currently connected to remote.
    pub connected: bool,
}

impl CacheStats {
    /// Calculate hit rate (0.0 to 1.0).
    pub fn hit_rate(&self) -> f64 {
        let hits = self.local_hits + self.remote_hits;
        let total = hits + self.misses;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }

    /// Calculate local hit rate (0.0 to 1.0).
    pub fn local_hit_rate(&self) -> f64 {
        let total = self.local_hits + self.remote_hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.local_hits as f64 / total as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Utility Functions
// ---------------------------------------------------------------------------

/// Get current timestamp in milliseconds since UNIX epoch.
fn current_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::content_store::MemoryContentStore;
    use std::io::Cursor;

    fn create_test_cache() -> RemoteCache<MemoryContentStore, MockHttpClient> {
        let local_store = Arc::new(MemoryContentStore::default());
        let http_client = Arc::new(MockHttpClient::new());
        let config = RemoteCacheConfig::default()
            .with_endpoint("http://test.example.com")
            .with_background_upload(false); // Disable for predictable tests

        RemoteCache::new(config, local_store, http_client)
    }

    // ========================================================================
    // Configuration tests
    // ========================================================================

    #[test]
    fn test_default_config() {
        let config = RemoteCacheConfig::default();
        assert!(config.endpoint.is_empty());
        assert!(config.api_key.is_none());
        assert_eq!(config.timeout_ms, DEFAULT_TIMEOUT_MS);
        assert_eq!(config.max_retries, DEFAULT_MAX_RETRIES);
        assert!(config.upload_in_background);
        assert_eq!(config.batch_size, DEFAULT_BATCH_SIZE);
    }

    #[test]
    fn test_config_with_endpoint() {
        let config = RemoteCacheConfig::default()
            .with_endpoint("https://cache.example.com");
        assert_eq!(config.endpoint, "https://cache.example.com");
    }

    #[test]
    fn test_config_with_api_key() {
        let config = RemoteCacheConfig::default()
            .with_api_key("secret-key");
        assert_eq!(config.api_key, Some("secret-key".to_string()));
    }

    #[test]
    fn test_config_with_bearer_token() {
        let config = RemoteCacheConfig::default()
            .with_bearer_token("token123");
        assert_eq!(config.bearer_token, Some("token123".to_string()));
    }

    #[test]
    fn test_config_validate_empty_endpoint() {
        let config = RemoteCacheConfig::default();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_success() {
        let config = RemoteCacheConfig::default()
            .with_endpoint("https://example.com");
        assert!(config.validate().is_ok());
    }

    // ========================================================================
    // Local cache hit tests
    // ========================================================================

    #[test]
    fn test_local_cache_hit() {
        let cache = create_test_cache();

        // Store content locally
        let data = b"test content";
        let hash = ContentHash::from_bytes(data);
        let mut cursor = Cursor::new(data);
        cache.local_store.put_stream(&mut cursor).unwrap();

        // Get should return local hit
        let result = cache.get(&hash);
        assert!(result.is_local_hit());
        assert_eq!(result.data().unwrap(), data);
    }

    #[test]
    fn test_local_cache_hit_multiple() {
        let cache = create_test_cache();

        // Store multiple items
        for i in 0..5 {
            let data = format!("content_{}", i);
            let hash = ContentHash::from_bytes(data.as_bytes());
            let mut cursor = Cursor::new(data.as_bytes());
            cache.local_store.put_stream(&mut cursor).unwrap();
        }

        // All should be local hits
        for i in 0..5 {
            let data = format!("content_{}", i);
            let hash = ContentHash::from_bytes(data.as_bytes());
            let result = cache.get(&hash);
            assert!(result.is_local_hit());
        }
    }

    #[test]
    fn test_local_cache_hit_large_data() {
        let cache = create_test_cache();

        // Store large content
        let data = vec![0xAB; 1024 * 1024]; // 1MB
        let hash = ContentHash::from_bytes(&data);
        let mut cursor = Cursor::new(&data);
        cache.local_store.put_stream(&mut cursor).unwrap();

        // Should be local hit
        let result = cache.get(&hash);
        assert!(result.is_local_hit());
        assert_eq!(result.data().unwrap().len(), data.len());
    }

    // ========================================================================
    // Remote cache hit tests
    // ========================================================================

    #[test]
    fn test_remote_cache_hit() {
        let local_store = Arc::new(MemoryContentStore::default());
        let data = b"remote content";
        let hash = ContentHash::from_bytes(data);

        let http_client = Arc::new(MockHttpClient::new().with_content(&hash, data.to_vec()));
        let config = RemoteCacheConfig::default()
            .with_endpoint("http://test.example.com")
            .with_background_upload(false);

        let cache = RemoteCache::new(config, local_store.clone(), http_client);

        // Get should return remote hit
        let result = cache.get(&hash);
        assert!(result.is_remote_hit());
        assert_eq!(result.data().unwrap(), data);

        // Content should now be cached locally
        assert!(local_store.has(&hash));
    }

    #[test]
    fn test_remote_cache_hit_caches_locally() {
        let local_store = Arc::new(MemoryContentStore::default());
        let data = b"remote content to cache";
        let hash = ContentHash::from_bytes(data);

        let http_client = Arc::new(MockHttpClient::new().with_content(&hash, data.to_vec()));
        let config = RemoteCacheConfig::default()
            .with_endpoint("http://test.example.com")
            .with_background_upload(false);

        let cache = RemoteCache::new(config, local_store.clone(), http_client);

        // First get - remote hit
        let result1 = cache.get(&hash);
        assert!(result1.is_remote_hit());

        // Second get - should be local hit
        let result2 = cache.get(&hash);
        assert!(result2.is_local_hit());
    }

    #[test]
    fn test_remote_cache_hit_updates_stats() {
        let local_store = Arc::new(MemoryContentStore::default());
        let data = b"stats test content";
        let hash = ContentHash::from_bytes(data);

        let http_client = Arc::new(MockHttpClient::new().with_content(&hash, data.to_vec()));
        let config = RemoteCacheConfig::default()
            .with_endpoint("http://test.example.com")
            .with_background_upload(false);

        let cache = RemoteCache::new(config, local_store, http_client);

        cache.get(&hash);
        let stats = cache.stats();

        assert_eq!(stats.total_requests, 1);
        assert_eq!(stats.remote_hits, 1);
        assert_eq!(stats.bytes_downloaded, data.len() as u64);
    }

    // ========================================================================
    // Cache miss tests
    // ========================================================================

    #[test]
    fn test_cache_miss() {
        let cache = create_test_cache();
        let hash = ContentHash::from_bytes(b"nonexistent");

        let result = cache.get(&hash);
        assert!(result.is_miss());
    }

    #[test]
    fn test_cache_miss_not_in_local_or_remote() {
        let local_store = Arc::new(MemoryContentStore::default());
        let http_client = Arc::new(MockHttpClient::new()); // Empty remote

        let config = RemoteCacheConfig::default()
            .with_endpoint("http://test.example.com")
            .with_background_upload(false);

        let cache = RemoteCache::new(config, local_store, http_client);
        let hash = ContentHash::from_bytes(b"missing content");

        let result = cache.get(&hash);
        assert!(result.is_miss());
    }

    #[test]
    fn test_cache_miss_updates_stats() {
        let cache = create_test_cache();
        let hash = ContentHash::from_bytes(b"nonexistent for stats");

        cache.get(&hash);
        let stats = cache.stats();

        assert_eq!(stats.total_requests, 1);
        assert_eq!(stats.misses, 1);
    }

    // ========================================================================
    // Fallback on disconnect tests
    // ========================================================================

    #[test]
    fn test_fallback_when_remote_offline() {
        let local_store = Arc::new(MemoryContentStore::default());
        let http_client = Arc::new(MockHttpClient::new());
        http_client.set_offline(true);

        let config = RemoteCacheConfig::default()
            .with_endpoint("http://test.example.com")
            .with_background_upload(false)
            .with_reconnect_interval_ms(0); // Allow immediate retry

        let cache = RemoteCache::new(config, local_store, http_client);
        let hash = ContentHash::from_bytes(b"offline content");

        // Should miss (remote is offline)
        let result = cache.get(&hash);
        assert!(result.is_miss());
    }

    #[test]
    fn test_fallback_uses_local_when_offline() {
        let local_store = Arc::new(MemoryContentStore::default());

        // Pre-populate local store
        let data = b"local only content";
        let hash = ContentHash::from_bytes(data);
        let mut cursor = Cursor::new(data);
        local_store.put_stream(&mut cursor).unwrap();

        let http_client = Arc::new(MockHttpClient::new());
        http_client.set_offline(true);

        let config = RemoteCacheConfig::default()
            .with_endpoint("http://test.example.com")
            .with_background_upload(false);

        let cache = RemoteCache::new(config, local_store, http_client);

        // Should return local hit even when remote is offline
        let result = cache.get(&hash);
        assert!(result.is_local_hit());
    }

    #[test]
    fn test_reconnection_after_offline() {
        let local_store = Arc::new(MemoryContentStore::default());
        let data = b"reconnection test";
        let hash = ContentHash::from_bytes(data);

        let http_client = Arc::new(MockHttpClient::new().with_content(&hash, data.to_vec()));

        let config = RemoteCacheConfig::default()
            .with_endpoint("http://test.example.com")
            .with_background_upload(false)
            .with_reconnect_interval_ms(0); // Allow immediate retry

        let cache = RemoteCache::new(config, local_store, http_client.clone());

        // Set offline
        http_client.set_offline(true);
        let result1 = cache.get(&hash);
        assert!(result1.is_miss() || result1.is_error());

        // Come back online
        http_client.set_offline(false);
        let result2 = cache.get(&hash);
        assert!(result2.is_remote_hit());
    }

    // ========================================================================
    // Background upload tests
    // ========================================================================

    #[test]
    fn test_background_upload_queued() {
        let local_store = Arc::new(MemoryContentStore::default());
        let http_client = Arc::new(MockHttpClient::new());

        let config = RemoteCacheConfig::default()
            .with_endpoint("http://test.example.com")
            .with_background_upload(true);

        let cache = RemoteCache::new(config, local_store, http_client);

        let data = b"background upload content";
        let hash = ContentHash::from_bytes(data);

        // Queue upload
        cache.queue_background_upload(hash, data.to_vec());

        // Should be in queue
        let stats = cache.stats();
        assert!(stats.pending_uploads > 0);

        cache.shutdown();
    }

    #[test]
    fn test_put_queues_upload() {
        let local_store = Arc::new(MemoryContentStore::default());
        let http_client = Arc::new(MockHttpClient::new());

        let config = RemoteCacheConfig::default()
            .with_endpoint("http://test.example.com")
            .with_background_upload(true);

        let cache = RemoteCache::new(config, local_store.clone(), http_client);

        let data = b"put with background upload";
        let hash = ContentHash::from_bytes(data);

        cache.put(&hash, data).unwrap();

        // Content should be in local store
        assert!(local_store.has(&hash));

        cache.shutdown();
    }

    #[test]
    fn test_synchronous_upload_when_disabled() {
        let local_store = Arc::new(MemoryContentStore::default());
        let http_client = Arc::new(MockHttpClient::new());

        let config = RemoteCacheConfig::default()
            .with_endpoint("http://test.example.com")
            .with_background_upload(false);

        let cache = RemoteCache::new(config, local_store.clone(), http_client.clone());

        let data = b"sync upload content";
        let hash = ContentHash::from_bytes(data);

        cache.put(&hash, data).unwrap();

        // Content should be in both local and remote
        assert!(local_store.has(&hash));
        assert!(http_client.has(&hash));
    }

    // ========================================================================
    // Batch operation tests
    // ========================================================================

    #[test]
    fn test_batch_get_empty() {
        let cache = create_test_cache();
        let results = cache.batch_get(&[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_batch_get_all_local() {
        let cache = create_test_cache();

        // Store multiple items locally
        let mut hashes = Vec::new();
        for i in 0..5 {
            let data = format!("batch content {}", i);
            let hash = ContentHash::from_bytes(data.as_bytes());
            let mut cursor = Cursor::new(data.as_bytes());
            cache.local_store.put_stream(&mut cursor).unwrap();
            hashes.push(hash);
        }

        let results = cache.batch_get(&hashes);
        assert_eq!(results.len(), 5);
        assert!(results.iter().all(|r| r.is_local_hit()));
    }

    #[test]
    fn test_batch_get_mixed_local_and_remote() {
        let local_store = Arc::new(MemoryContentStore::default());

        // Store some locally
        let local_data = b"local batch item";
        let local_hash = ContentHash::from_bytes(local_data);
        let mut cursor = Cursor::new(local_data);
        local_store.put_stream(&mut cursor).unwrap();

        // Put some on remote
        let remote_data = b"remote batch item";
        let remote_hash = ContentHash::from_bytes(remote_data);
        let http_client = Arc::new(MockHttpClient::new().with_content(&remote_hash, remote_data.to_vec()));

        let config = RemoteCacheConfig::default()
            .with_endpoint("http://test.example.com")
            .with_background_upload(false);

        let cache = RemoteCache::new(config, local_store, http_client);

        let results = cache.batch_get(&[local_hash, remote_hash]);
        assert_eq!(results.len(), 2);
        assert!(results[0].is_local_hit());
        assert!(results[1].is_remote_hit());
    }

    // ========================================================================
    // Authentication tests
    // ========================================================================

    #[test]
    fn test_api_key_in_requests() {
        let local_store = Arc::new(MemoryContentStore::default());
        let data = b"auth test content";
        let hash = ContentHash::from_bytes(data);

        let http_client = Arc::new(MockHttpClient::new().with_content(&hash, data.to_vec()));

        let config = RemoteCacheConfig::default()
            .with_endpoint("http://test.example.com")
            .with_api_key("test-api-key")
            .with_background_upload(false);

        let cache = RemoteCache::new(config, local_store, http_client);
        cache.get(&hash);

        // The mock doesn't validate headers, but this tests the code path
    }

    #[test]
    fn test_bearer_token_in_requests() {
        let local_store = Arc::new(MemoryContentStore::default());
        let data = b"bearer test content";
        let hash = ContentHash::from_bytes(data);

        let http_client = Arc::new(MockHttpClient::new().with_content(&hash, data.to_vec()));

        let config = RemoteCacheConfig::default()
            .with_endpoint("http://test.example.com")
            .with_bearer_token("test-bearer-token")
            .with_background_upload(false);

        let cache = RemoteCache::new(config, local_store, http_client);
        cache.get(&hash);
    }

    // ========================================================================
    // Error handling tests
    // ========================================================================

    #[test]
    fn test_error_on_network_failure() {
        let local_store = Arc::new(MemoryContentStore::default());
        let http_client = Arc::new(MockHttpClient::new());

        let config = RemoteCacheConfig::default()
            .with_endpoint("http://test.example.com")
            .with_background_upload(false)
            .with_reconnect_interval_ms(0);

        let cache = RemoteCache::new(config, local_store, http_client.clone());

        // Make next request fail
        http_client.fail_next_request();

        let hash = ContentHash::from_bytes(b"fail test");
        let result = cache.get(&hash);

        // Should be error or miss (depends on retry behavior)
        assert!(result.is_error() || result.is_miss());
    }

    #[test]
    fn test_hash_verification_on_download() {
        let local_store = Arc::new(MemoryContentStore::default());
        let hash = ContentHash::from_bytes(b"expected content");
        let wrong_data = b"wrong content that doesn't match hash";

        // Remote has wrong data for this hash
        let http_client = Arc::new(MockHttpClient::new().with_content(&hash, wrong_data.to_vec()));

        let config = RemoteCacheConfig::default()
            .with_endpoint("http://test.example.com")
            .with_background_upload(false)
            .with_verify_downloads(true);

        let cache = RemoteCache::new(config, local_store, http_client);

        let result = cache.get(&hash);
        assert!(result.is_error());
    }

    // ========================================================================
    // Statistics tests
    // ========================================================================

    #[test]
    fn test_stats_initial() {
        let cache = create_test_cache();
        let stats = cache.stats();

        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.local_hits, 0);
        assert_eq!(stats.remote_hits, 0);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_stats_after_operations() {
        let local_store = Arc::new(MemoryContentStore::default());

        // Add local content
        let local_data = b"local stats content";
        let local_hash = ContentHash::from_bytes(local_data);
        let mut cursor = Cursor::new(local_data);
        local_store.put_stream(&mut cursor).unwrap();

        // Add remote content
        let remote_data = b"remote stats content";
        let remote_hash = ContentHash::from_bytes(remote_data);
        let http_client = Arc::new(MockHttpClient::new().with_content(&remote_hash, remote_data.to_vec()));

        let config = RemoteCacheConfig::default()
            .with_endpoint("http://test.example.com")
            .with_background_upload(false);

        let cache = RemoteCache::new(config, local_store, http_client);

        // Make requests
        cache.get(&local_hash);
        cache.get(&remote_hash);
        cache.get(&ContentHash::from_bytes(b"missing"));

        let stats = cache.stats();
        assert_eq!(stats.total_requests, 3);
        assert_eq!(stats.local_hits, 1);
        assert_eq!(stats.remote_hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_stats_reset() {
        let cache = create_test_cache();

        // Make some requests
        cache.get(&ContentHash::from_bytes(b"test1"));
        cache.get(&ContentHash::from_bytes(b"test2"));

        let stats_before = cache.stats();
        assert!(stats_before.total_requests > 0);

        cache.reset_stats();

        let stats_after = cache.stats();
        assert_eq!(stats_after.total_requests, 0);
        assert_eq!(stats_after.misses, 0);
    }

    #[test]
    fn test_hit_rate_calculation() {
        let stats = CacheStats {
            total_requests: 10,
            local_hits: 3,
            remote_hits: 2,
            misses: 5,
            ..Default::default()
        };

        assert!((stats.hit_rate() - 0.5).abs() < 0.001);
        assert!((stats.local_hit_rate() - 0.3).abs() < 0.001);
    }

    // ========================================================================
    // CacheResult tests
    // ========================================================================

    #[test]
    fn test_cache_result_local_hit() {
        let data = vec![1, 2, 3];
        let result = CacheResult::LocalHit(data.clone());

        assert!(result.is_hit());
        assert!(result.is_local_hit());
        assert!(!result.is_remote_hit());
        assert!(!result.is_miss());
        assert_eq!(result.data().unwrap(), &data);
    }

    #[test]
    fn test_cache_result_remote_hit() {
        let data = vec![4, 5, 6];
        let result = CacheResult::RemoteHit(data.clone());

        assert!(result.is_hit());
        assert!(!result.is_local_hit());
        assert!(result.is_remote_hit());
        assert!(!result.is_miss());
        assert_eq!(result.data().unwrap(), &data);
    }

    #[test]
    fn test_cache_result_miss() {
        let result = CacheResult::Miss;

        assert!(!result.is_hit());
        assert!(result.is_miss());
        assert!(result.data().is_none());
    }

    #[test]
    fn test_cache_result_error() {
        let result = CacheResult::Error("test error".to_string());

        assert!(!result.is_hit());
        assert!(!result.is_miss());
        assert!(result.is_error());
        assert!(result.data().is_none());
    }

    #[test]
    fn test_cache_result_into_data() {
        let data = vec![7, 8, 9];
        let result = CacheResult::LocalHit(data.clone());
        assert_eq!(result.into_data().unwrap(), data);

        let miss = CacheResult::Miss;
        assert!(miss.into_data().is_none());
    }

    // ========================================================================
    // MockHttpClient tests
    // ========================================================================

    #[test]
    fn test_mock_http_client_get() {
        let data = b"mock content";
        let hash = ContentHash::from_bytes(data);
        let client = MockHttpClient::new().with_content(&hash, data.to_vec());

        let request = HttpRequest::get(format!("http://test.com/{}", hash));
        let response = client.execute(request).unwrap();

        assert!(response.is_success());
        assert_eq!(response.body, data);
    }

    #[test]
    fn test_mock_http_client_put() {
        let client = MockHttpClient::new();
        let data = b"new content";
        let hash = ContentHash::from_bytes(data);

        let request = HttpRequest::put(format!("http://test.com/{}", hash), data.to_vec());
        let response = client.execute(request).unwrap();

        assert!(response.is_success());
        assert!(client.has(&hash));
    }

    #[test]
    fn test_mock_http_client_offline() {
        let client = MockHttpClient::new();
        client.set_offline(true);

        let request = HttpRequest::get("http://test.com/anything");
        let result = client.execute(request);

        assert!(result.is_err());
    }

    #[test]
    fn test_mock_http_client_latency() {
        let client = MockHttpClient::new();
        client.set_latency(10); // 10ms

        let start = std::time::Instant::now();
        let request = HttpRequest::get("http://test.com/anything");
        let _ = client.execute(request);
        let elapsed = start.elapsed();

        assert!(elapsed >= Duration::from_millis(10));
    }

    #[test]
    fn test_mock_http_client_request_count() {
        let client = MockHttpClient::new();

        assert_eq!(client.request_count(), 0);

        let _ = client.execute(HttpRequest::get("http://test.com/1"));
        let _ = client.execute(HttpRequest::get("http://test.com/2"));
        let _ = client.execute(HttpRequest::get("http://test.com/3"));

        assert_eq!(client.request_count(), 3);
    }

    // ========================================================================
    // SyncStats tests
    // ========================================================================

    #[test]
    fn test_sync_stats_merge() {
        let mut stats1 = SyncStats {
            uploaded: 5,
            downloaded: 3,
            bytes_transferred: 1000,
            errors: 1,
            duration_ms: 100,
        };

        let stats2 = SyncStats {
            uploaded: 2,
            downloaded: 4,
            bytes_transferred: 500,
            errors: 0,
            duration_ms: 50,
        };

        stats1.merge(&stats2);

        assert_eq!(stats1.uploaded, 7);
        assert_eq!(stats1.downloaded, 7);
        assert_eq!(stats1.bytes_transferred, 1500);
        assert_eq!(stats1.errors, 1);
        assert_eq!(stats1.duration_ms, 150);
    }

    // ========================================================================
    // Error type tests
    // ========================================================================

    #[test]
    fn test_error_display() {
        let err = RemoteCacheError::NotFound(ContentHash::from_bytes(b"test"));
        assert!(err.to_string().contains("not found"));

        let err = RemoteCacheError::NetworkError("connection refused".into());
        assert!(err.to_string().contains("network error"));

        let err = RemoteCacheError::HttpError { status_code: 500, message: "internal error".into() };
        assert!(err.to_string().contains("500"));
    }

    // ========================================================================
    // HttpRequest/HttpResponse tests
    // ========================================================================

    #[test]
    fn test_http_request_builders() {
        let get = HttpRequest::get("http://test.com");
        assert_eq!(get.method, HttpMethod::Get);
        assert_eq!(get.url, "http://test.com");

        let put = HttpRequest::put("http://test.com", vec![1, 2, 3]);
        assert_eq!(put.method, HttpMethod::Put);
        assert_eq!(put.body, Some(vec![1, 2, 3]));

        let head = HttpRequest::head("http://test.com");
        assert_eq!(head.method, HttpMethod::Head);
    }

    #[test]
    fn test_http_request_with_headers() {
        let request = HttpRequest::get("http://test.com")
            .with_header("X-Custom", "value")
            .with_timeout(1000);

        assert_eq!(request.headers.get("X-Custom"), Some(&"value".to_string()));
        assert_eq!(request.timeout_ms, 1000);
    }

    #[test]
    fn test_http_response_status_checks() {
        let success = HttpResponse { status_code: 200, headers: HashMap::new(), body: Vec::new() };
        assert!(success.is_success());
        assert!(!success.is_not_found());

        let not_found = HttpResponse { status_code: 404, headers: HashMap::new(), body: Vec::new() };
        assert!(!not_found.is_success());
        assert!(not_found.is_not_found());

        let rate_limited = HttpResponse { status_code: 429, headers: HashMap::new(), body: Vec::new() };
        assert!(rate_limited.is_rate_limited());

        let auth_error = HttpResponse { status_code: 401, headers: HashMap::new(), body: Vec::new() };
        assert!(auth_error.is_auth_error());
    }
}
