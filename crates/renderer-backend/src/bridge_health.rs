//! Bridge Health Monitoring and Error Handling (T-TL-1.5)
//!
//! This module provides production-grade error handling, timeout detection,
//! connection health monitoring, and reconnection logic for the Python-Rust bridge.
//!
//! # Features
//!
//! - **Typed Error Responses**: Extended error codes with categories and recovery hints
//! - **Timeout Handling**: Per-request timeout tracking with automatic cleanup
//! - **Health Monitoring**: Connection state machine with latency measurement
//! - **Reconnection Logic**: Exponential backoff with jitter and state preservation
//!
//! # Architecture
//!
//! ```text
//! HealthMonitor
//!     |
//!     +-- ConnectionState (Connected | Degraded | Disconnected)
//!     |
//!     +-- LatencyTracker (rolling average)
//!     |
//!     +-- Metrics (requests/sec, errors/sec)
//!
//! TimeoutTracker
//!     |
//!     +-- Per-request timeout configuration
//!     |
//!     +-- Automatic timeout detection
//!     |
//!     +-- Cleanup of expired requests
//!
//! ReconnectionManager
//!     |
//!     +-- Exponential backoff with jitter
//!     |
//!     +-- Pending request preservation
//!     |
//!     +-- State recovery
//! ```

use crate::bridge_protocol::BridgeError;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Error Categories and Codes
// ---------------------------------------------------------------------------

/// Error category for classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCategory {
    /// Connection-related errors (network, transport).
    Connection,
    /// Protocol-level errors (version mismatch, serialization).
    Protocol,
    /// Request validation errors (invalid params, missing fields).
    Validation,
    /// Resource errors (not found, exhausted).
    Resource,
    /// Permission/authorization errors.
    Permission,
    /// Internal/system errors.
    Internal,
    /// Timeout errors.
    Timeout,
    /// Cancelled operation.
    Cancelled,
}

impl ErrorCategory {
    /// Whether errors in this category are typically retryable.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            ErrorCategory::Connection | ErrorCategory::Timeout | ErrorCategory::Internal
        )
    }

    /// Suggested retry delay in milliseconds for this category.
    pub fn suggested_retry_delay_ms(&self) -> Option<u64> {
        match self {
            ErrorCategory::Connection => Some(1000),
            ErrorCategory::Timeout => Some(500),
            ErrorCategory::Internal => Some(100),
            _ => None,
        }
    }
}

/// Numeric error codes for programmatic handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum ErrorCode {
    // Connection errors (1xxx)
    NotConnected = 1001,
    ConnectionLost = 1002,
    ConnectionRefused = 1003,
    HandshakeFailed = 1004,

    // Protocol errors (2xxx)
    VersionMismatch = 2001,
    SerializationFailed = 2002,
    DeserializationFailed = 2003,
    InvalidMessage = 2004,
    MalformedPayload = 2005,

    // Validation errors (3xxx)
    InvalidParams = 3001,
    MissingField = 3002,
    TypeMismatch = 3003,
    OutOfRange = 3004,

    // Resource errors (4xxx)
    UnknownNamespace = 4001,
    UnknownMethod = 4002,
    EntityNotFound = 4003,
    ComponentNotFound = 4004,
    AssetNotFound = 4005,
    TypeNotRegistered = 4006,
    ResourceExhausted = 4007,

    // Permission errors (5xxx)
    PermissionDenied = 5001,
    Unauthorized = 5002,

    // Internal errors (6xxx)
    InternalError = 6001,
    SystemError = 6002,
    UnexpectedState = 6003,

    // Timeout errors (7xxx)
    RequestTimeout = 7001,
    OperationTimeout = 7002,

    // Cancelled (8xxx)
    RequestCancelled = 8001,
    ShutdownInProgress = 8002,
}

impl ErrorCode {
    /// Get the error category for this code.
    pub fn category(&self) -> ErrorCategory {
        match *self as u32 {
            1001..=1999 => ErrorCategory::Connection,
            2001..=2999 => ErrorCategory::Protocol,
            3001..=3999 => ErrorCategory::Validation,
            4001..=4999 => ErrorCategory::Resource,
            5001..=5999 => ErrorCategory::Permission,
            6001..=6999 => ErrorCategory::Internal,
            7001..=7999 => ErrorCategory::Timeout,
            8001..=8999 => ErrorCategory::Cancelled,
            _ => ErrorCategory::Internal,
        }
    }

    /// Get a human-readable description of this error code.
    pub fn description(&self) -> &'static str {
        match self {
            ErrorCode::NotConnected => "Bridge connection not established",
            ErrorCode::ConnectionLost => "Connection to bridge lost unexpectedly",
            ErrorCode::ConnectionRefused => "Connection refused by remote end",
            ErrorCode::HandshakeFailed => "Protocol handshake failed",
            ErrorCode::VersionMismatch => "Protocol version mismatch between client and server",
            ErrorCode::SerializationFailed => "Failed to serialize data for transport",
            ErrorCode::DeserializationFailed => "Failed to deserialize received data",
            ErrorCode::InvalidMessage => "Received invalid or malformed message",
            ErrorCode::MalformedPayload => "Message payload is malformed",
            ErrorCode::InvalidParams => "Invalid parameters provided",
            ErrorCode::MissingField => "Required field is missing",
            ErrorCode::TypeMismatch => "Type mismatch in parameter or field",
            ErrorCode::OutOfRange => "Value is out of acceptable range",
            ErrorCode::UnknownNamespace => "Unknown namespace in request",
            ErrorCode::UnknownMethod => "Unknown method in namespace",
            ErrorCode::EntityNotFound => "Entity does not exist",
            ErrorCode::ComponentNotFound => "Component does not exist on entity",
            ErrorCode::AssetNotFound => "Asset not found",
            ErrorCode::TypeNotRegistered => "Type not registered in schema",
            ErrorCode::ResourceExhausted => "System resources exhausted",
            ErrorCode::PermissionDenied => "Permission denied for operation",
            ErrorCode::Unauthorized => "Not authorized to perform operation",
            ErrorCode::InternalError => "Internal server error",
            ErrorCode::SystemError => "System-level error occurred",
            ErrorCode::UnexpectedState => "Unexpected state encountered",
            ErrorCode::RequestTimeout => "Request timed out",
            ErrorCode::OperationTimeout => "Operation timed out",
            ErrorCode::RequestCancelled => "Request was cancelled",
            ErrorCode::ShutdownInProgress => "Shutdown in progress, request rejected",
        }
    }
}

impl From<&BridgeError> for ErrorCode {
    fn from(error: &BridgeError) -> Self {
        match error {
            BridgeError::NotConnected => ErrorCode::NotConnected,
            BridgeError::Timeout(_) => ErrorCode::RequestTimeout,
            BridgeError::Serialization(_) => ErrorCode::SerializationFailed,
            BridgeError::Deserialization(_) => ErrorCode::DeserializationFailed,
            BridgeError::UnknownNamespace(_) => ErrorCode::UnknownNamespace,
            BridgeError::UnknownMethod(_) => ErrorCode::UnknownMethod,
            BridgeError::InvalidParams(_) => ErrorCode::InvalidParams,
            BridgeError::Internal(_) => ErrorCode::InternalError,
            BridgeError::VersionMismatch { .. } => ErrorCode::VersionMismatch,
            BridgeError::TypeNotRegistered(_) => ErrorCode::TypeNotRegistered,
            BridgeError::EntityNotFound(_) => ErrorCode::EntityNotFound,
            BridgeError::ComponentNotFound(_) => ErrorCode::ComponentNotFound,
            BridgeError::AssetNotFound(_) => ErrorCode::AssetNotFound,
            BridgeError::PermissionDenied(_) => ErrorCode::PermissionDenied,
            BridgeError::ResourceExhausted(_) => ErrorCode::ResourceExhausted,
            BridgeError::Cancelled => ErrorCode::RequestCancelled,
        }
    }
}

// ---------------------------------------------------------------------------
// Error Response
// ---------------------------------------------------------------------------

/// Structured error response with extended metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// Numeric error code.
    pub code: u32,
    /// Error category.
    pub category: ErrorCategory,
    /// Human-readable error message.
    pub message: String,
    /// Optional detailed description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
    /// Recovery hint for the client.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recovery_hint: Option<String>,
    /// Whether this error is retryable.
    pub retryable: bool,
    /// Suggested retry delay in milliseconds (if retryable).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_after_ms: Option<u64>,
    /// Request ID that caused this error (if applicable).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<u64>,
    /// Timestamp when error occurred (milliseconds since epoch).
    pub timestamp_ms: u64,
}

impl ErrorResponse {
    /// Create an error response from a BridgeError.
    pub fn from_bridge_error(error: &BridgeError, request_id: Option<u64>) -> Self {
        let code = ErrorCode::from(error);
        let category = code.category();

        let recovery_hint = match error {
            BridgeError::NotConnected => Some("Establish connection before sending requests".into()),
            BridgeError::Timeout(_) => Some("Increase timeout or retry the request".into()),
            BridgeError::VersionMismatch { expected, got } => Some(format!(
                "Update client to protocol version {} (current: {})",
                expected, got
            )),
            BridgeError::EntityNotFound(id) => {
                Some(format!("Verify entity {} exists before accessing", id))
            }
            BridgeError::ComponentNotFound(c) => {
                Some(format!("Add component '{}' to entity first", c))
            }
            BridgeError::ResourceExhausted(_) => {
                Some("Wait for resources to become available or reduce load".into())
            }
            _ => None,
        };

        Self {
            code: code as u32,
            category,
            message: error.to_string(),
            details: Some(code.description().into()),
            recovery_hint,
            retryable: category.is_retryable(),
            retry_after_ms: category.suggested_retry_delay_ms(),
            request_id,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        }
    }

    /// Create a timeout error response.
    pub fn timeout(request_id: u64, namespace: &str, method: &str, timeout_ms: u64) -> Self {
        Self {
            code: ErrorCode::RequestTimeout as u32,
            category: ErrorCategory::Timeout,
            message: format!(
                "Request {}.{} timed out after {}ms",
                namespace, method, timeout_ms
            ),
            details: Some(ErrorCode::RequestTimeout.description().into()),
            recovery_hint: Some("Increase timeout or check server responsiveness".into()),
            retryable: true,
            retry_after_ms: Some(100),
            request_id: Some(request_id),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        }
    }

    /// Create a connection lost error response.
    pub fn connection_lost(pending_requests: usize) -> Self {
        Self {
            code: ErrorCode::ConnectionLost as u32,
            category: ErrorCategory::Connection,
            message: format!(
                "Connection lost with {} pending requests",
                pending_requests
            ),
            details: Some(ErrorCode::ConnectionLost.description().into()),
            recovery_hint: Some("Reconnect and retry failed requests".into()),
            retryable: true,
            retry_after_ms: Some(1000),
            request_id: None,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        }
    }

    /// Serialize to JSON bytes for Python consumption.
    pub fn to_json(&self) -> Result<Vec<u8>, BridgeError> {
        serde_json::to_vec(self).map_err(|e| BridgeError::Serialization(e.to_string()))
    }

    /// Deserialize from JSON bytes.
    pub fn from_json(data: &[u8]) -> Result<Self, BridgeError> {
        serde_json::from_slice(data).map_err(|e| BridgeError::Deserialization(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Timeout Configuration
// ---------------------------------------------------------------------------

/// Timeout configuration for namespace/method combinations.
#[derive(Debug, Clone)]
pub struct TimeoutConfig {
    /// Default timeout for all requests.
    pub default_timeout: Duration,
    /// Per-namespace default timeouts.
    namespace_timeouts: HashMap<String, Duration>,
    /// Per-method timeouts (namespace.method -> timeout).
    method_timeouts: HashMap<(String, String), Duration>,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            default_timeout: Duration::from_secs(30),
            namespace_timeouts: HashMap::new(),
            method_timeouts: HashMap::new(),
        }
    }
}

impl TimeoutConfig {
    /// Create a new timeout configuration with default values.
    pub fn new() -> Self {
        let mut config = Self::default();

        // Set sensible defaults for known namespaces
        config.set_namespace_timeout("bridge", Duration::from_secs(5));
        config.set_namespace_timeout("entity", Duration::from_secs(10));
        config.set_namespace_timeout("component", Duration::from_secs(10));
        config.set_namespace_timeout("frame", Duration::from_secs(1));
        config.set_namespace_timeout("profiler", Duration::from_secs(5));
        config.set_namespace_timeout("editor", Duration::from_secs(5));
        config.set_namespace_timeout("material", Duration::from_secs(30));
        config.set_namespace_timeout("asset", Duration::from_secs(60));

        // Special cases for potentially long operations
        config.set_method_timeout("asset", "load", Duration::from_secs(120));
        config.set_method_timeout("material", "compile", Duration::from_secs(60));

        config
    }

    /// Set default timeout for a namespace.
    pub fn set_namespace_timeout(&mut self, namespace: &str, timeout: Duration) {
        self.namespace_timeouts.insert(namespace.to_string(), timeout);
    }

    /// Set timeout for a specific method.
    pub fn set_method_timeout(&mut self, namespace: &str, method: &str, timeout: Duration) {
        self.method_timeouts
            .insert((namespace.to_string(), method.to_string()), timeout);
    }

    /// Get the configured timeout for a namespace/method.
    pub fn get_timeout(&self, namespace: &str, method: &str) -> Duration {
        // Check method-specific first
        if let Some(timeout) = self
            .method_timeouts
            .get(&(namespace.to_string(), method.to_string()))
        {
            return *timeout;
        }

        // Then namespace default
        if let Some(timeout) = self.namespace_timeouts.get(namespace) {
            return *timeout;
        }

        // Fall back to global default
        self.default_timeout
    }
}

// ---------------------------------------------------------------------------
// Timeout Tracker
// ---------------------------------------------------------------------------

/// Pending request with timeout information.
#[derive(Debug, Clone)]
pub struct PendingRequest {
    /// Request ID.
    pub id: u64,
    /// Namespace.
    pub namespace: String,
    /// Method.
    pub method: String,
    /// When the request was started.
    pub start_time: Instant,
    /// Timeout for this request.
    pub timeout: Duration,
    /// Whether this request has been marked as timed out.
    pub timed_out: bool,
}

impl PendingRequest {
    /// Check if this request has exceeded its timeout.
    pub fn is_expired(&self) -> bool {
        self.start_time.elapsed() > self.timeout
    }

    /// Get remaining time before timeout.
    pub fn remaining(&self) -> Duration {
        self.timeout.saturating_sub(self.start_time.elapsed())
    }

    /// Get elapsed time since start.
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
}

/// Tracks pending requests with timeout detection.
#[derive(Debug)]
pub struct TimeoutTracker {
    /// Pending requests: id -> request info.
    pending: RwLock<HashMap<u64, PendingRequest>>,
    /// Timeout configuration.
    config: RwLock<TimeoutConfig>,
    /// Total requests tracked.
    total_requests: AtomicU64,
    /// Total timeouts.
    total_timeouts: AtomicU64,
}

impl TimeoutTracker {
    /// Create a new timeout tracker with default configuration.
    pub fn new() -> Self {
        Self {
            pending: RwLock::new(HashMap::new()),
            config: RwLock::new(TimeoutConfig::new()),
            total_requests: AtomicU64::new(0),
            total_timeouts: AtomicU64::new(0),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: TimeoutConfig) -> Self {
        Self {
            pending: RwLock::new(HashMap::new()),
            config: RwLock::new(config),
            total_requests: AtomicU64::new(0),
            total_timeouts: AtomicU64::new(0),
        }
    }

    /// Register a new pending request.
    pub fn register(&self, id: u64, namespace: &str, method: &str) {
        let timeout = self.config.read().get_timeout(namespace, method);
        self.register_with_timeout(id, namespace, method, timeout);
    }

    /// Register a request with explicit timeout.
    pub fn register_with_timeout(
        &self,
        id: u64,
        namespace: &str,
        method: &str,
        timeout: Duration,
    ) {
        let request = PendingRequest {
            id,
            namespace: namespace.to_string(),
            method: method.to_string(),
            start_time: Instant::now(),
            timeout,
            timed_out: false,
        };

        self.pending.write().insert(id, request);
        self.total_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Complete a request and return elapsed time.
    pub fn complete(&self, id: u64) -> Option<Duration> {
        self.pending.write().remove(&id).map(|r| r.elapsed())
    }

    /// Check for timed-out requests and return their IDs.
    pub fn check_timeouts(&self) -> Vec<PendingRequest> {
        let mut pending = self.pending.write();
        let mut timed_out = Vec::new();

        for request in pending.values_mut() {
            if request.is_expired() && !request.timed_out {
                request.timed_out = true;
                timed_out.push(request.clone());
                self.total_timeouts.fetch_add(1, Ordering::Relaxed);
            }
        }

        timed_out
    }

    /// Remove timed-out requests from tracking.
    pub fn cleanup_expired(&self) -> Vec<PendingRequest> {
        let mut pending = self.pending.write();
        let expired: Vec<_> = pending
            .iter()
            .filter(|(_, r)| r.timed_out)
            .map(|(id, r)| (*id, r.clone()))
            .collect();

        for (id, _) in &expired {
            pending.remove(id);
        }

        expired.into_iter().map(|(_, r)| r).collect()
    }

    /// Get the number of pending requests.
    pub fn pending_count(&self) -> usize {
        self.pending.read().len()
    }

    /// Get all pending request IDs.
    pub fn pending_ids(&self) -> Vec<u64> {
        self.pending.read().keys().cloned().collect()
    }

    /// Get a specific pending request.
    pub fn get(&self, id: u64) -> Option<PendingRequest> {
        self.pending.read().get(&id).cloned()
    }

    /// Get total requests tracked.
    pub fn total_requests(&self) -> u64 {
        self.total_requests.load(Ordering::Relaxed)
    }

    /// Get total timeouts.
    pub fn total_timeouts(&self) -> u64 {
        self.total_timeouts.load(Ordering::Relaxed)
    }

    /// Get timeout configuration (read-only).
    pub fn config(&self) -> TimeoutConfig {
        self.config.read().clone()
    }

    /// Update timeout configuration.
    pub fn set_config(&self, config: TimeoutConfig) {
        *self.config.write() = config;
    }
}

impl Default for TimeoutTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Connection State
// ---------------------------------------------------------------------------

/// Connection state machine states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConnectionState {
    /// Connection is healthy.
    Connected,
    /// Connection is experiencing issues but still functional.
    Degraded,
    /// Connection is lost.
    Disconnected,
    /// Reconnection in progress.
    Reconnecting,
}

impl ConnectionState {
    /// Whether the connection can process requests.
    pub fn is_operational(&self) -> bool {
        matches!(self, ConnectionState::Connected | ConnectionState::Degraded)
    }

    /// Whether the connection is fully healthy.
    pub fn is_healthy(&self) -> bool {
        matches!(self, ConnectionState::Connected)
    }
}

// ---------------------------------------------------------------------------
// Latency Tracker
// ---------------------------------------------------------------------------

/// Rolling average latency tracker.
#[derive(Debug)]
pub struct LatencyTracker {
    /// Window of recent latencies (microseconds).
    samples: RwLock<Vec<u64>>,
    /// Maximum samples to keep.
    max_samples: usize,
    /// Sum of all samples for fast average.
    sum_us: AtomicU64,
    /// Total samples recorded.
    total_samples: AtomicU64,
    /// Min latency observed.
    min_us: AtomicU64,
    /// Max latency observed.
    max_us: AtomicU64,
}

impl LatencyTracker {
    /// Create a new latency tracker.
    pub fn new(max_samples: usize) -> Self {
        Self {
            samples: RwLock::new(Vec::with_capacity(max_samples)),
            max_samples,
            sum_us: AtomicU64::new(0),
            total_samples: AtomicU64::new(0),
            min_us: AtomicU64::new(u64::MAX),
            max_us: AtomicU64::new(0),
        }
    }

    /// Record a latency sample.
    pub fn record(&self, latency: Duration) {
        let latency_us = latency.as_micros() as u64;

        let mut samples = self.samples.write();

        // Remove oldest if at capacity
        if samples.len() >= self.max_samples {
            if let Some(old) = samples.first() {
                self.sum_us.fetch_sub(*old, Ordering::Relaxed);
            }
            samples.remove(0);
        }

        samples.push(latency_us);
        self.sum_us.fetch_add(latency_us, Ordering::Relaxed);
        self.total_samples.fetch_add(1, Ordering::Relaxed);

        // Update min/max
        let _ = self.min_us.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |min| {
            if latency_us < min {
                Some(latency_us)
            } else {
                None
            }
        });
        let _ = self.max_us.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |max| {
            if latency_us > max {
                Some(latency_us)
            } else {
                None
            }
        });
    }

    /// Get average latency in microseconds.
    pub fn average_us(&self) -> u64 {
        let samples = self.samples.read();
        if samples.is_empty() {
            return 0;
        }
        self.sum_us.load(Ordering::Relaxed) / samples.len() as u64
    }

    /// Get average latency as Duration.
    pub fn average(&self) -> Duration {
        Duration::from_micros(self.average_us())
    }

    /// Get minimum latency observed.
    pub fn min(&self) -> Duration {
        let min = self.min_us.load(Ordering::Relaxed);
        if min == u64::MAX {
            Duration::ZERO
        } else {
            Duration::from_micros(min)
        }
    }

    /// Get maximum latency observed.
    pub fn max(&self) -> Duration {
        Duration::from_micros(self.max_us.load(Ordering::Relaxed))
    }

    /// Get total samples recorded.
    pub fn total_samples(&self) -> u64 {
        self.total_samples.load(Ordering::Relaxed)
    }

    /// Get current sample count in window.
    pub fn sample_count(&self) -> usize {
        self.samples.read().len()
    }

    /// Get P50 latency (median).
    pub fn p50(&self) -> Duration {
        self.percentile(50)
    }

    /// Get P95 latency.
    pub fn p95(&self) -> Duration {
        self.percentile(95)
    }

    /// Get P99 latency.
    pub fn p99(&self) -> Duration {
        self.percentile(99)
    }

    /// Get percentile latency.
    pub fn percentile(&self, p: u8) -> Duration {
        let samples = self.samples.read();
        if samples.is_empty() {
            return Duration::ZERO;
        }

        let mut sorted: Vec<_> = samples.iter().cloned().collect();
        sorted.sort_unstable();

        let idx = ((p as usize * sorted.len()) / 100).min(sorted.len() - 1);
        Duration::from_micros(sorted[idx])
    }

    /// Reset the tracker.
    pub fn reset(&self) {
        self.samples.write().clear();
        self.sum_us.store(0, Ordering::Relaxed);
        self.total_samples.store(0, Ordering::Relaxed);
        self.min_us.store(u64::MAX, Ordering::Relaxed);
        self.max_us.store(0, Ordering::Relaxed);
    }
}

impl Default for LatencyTracker {
    fn default() -> Self {
        Self::new(1000)
    }
}

// ---------------------------------------------------------------------------
// Health Metrics
// ---------------------------------------------------------------------------

/// Health metrics snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMetrics {
    /// Current connection state.
    pub state: ConnectionState,
    /// Requests per second (rolling).
    pub requests_per_sec: f64,
    /// Errors per second (rolling).
    pub errors_per_sec: f64,
    /// Error rate (errors / total).
    pub error_rate: f64,
    /// Average latency in milliseconds.
    pub avg_latency_ms: f64,
    /// P50 latency in milliseconds.
    pub p50_latency_ms: f64,
    /// P95 latency in milliseconds.
    pub p95_latency_ms: f64,
    /// P99 latency in milliseconds.
    pub p99_latency_ms: f64,
    /// Min latency in milliseconds.
    pub min_latency_ms: f64,
    /// Max latency in milliseconds.
    pub max_latency_ms: f64,
    /// Pending request count.
    pub pending_requests: u32,
    /// Total requests since start.
    pub total_requests: u64,
    /// Total errors since start.
    pub total_errors: u64,
    /// Total timeouts since start.
    pub total_timeouts: u64,
    /// Uptime in milliseconds.
    pub uptime_ms: u64,
    /// Last heartbeat received (milliseconds since epoch).
    pub last_heartbeat_ms: u64,
    /// Consecutive heartbeat failures.
    pub heartbeat_failures: u32,
}

// ---------------------------------------------------------------------------
// Health Monitor
// ---------------------------------------------------------------------------

/// Threshold configuration for health monitoring.
#[derive(Debug, Clone)]
pub struct HealthThresholds {
    /// Latency threshold to consider degraded (milliseconds).
    pub degraded_latency_ms: u64,
    /// Error rate threshold to consider degraded.
    pub degraded_error_rate: f64,
    /// Max heartbeat failures before disconnected.
    pub max_heartbeat_failures: u32,
    /// Heartbeat interval.
    pub heartbeat_interval: Duration,
}

impl Default for HealthThresholds {
    fn default() -> Self {
        Self {
            degraded_latency_ms: 500,
            degraded_error_rate: 0.1,
            max_heartbeat_failures: 3,
            heartbeat_interval: Duration::from_secs(5),
        }
    }
}

/// Connection health monitor with heartbeat tracking.
#[derive(Debug)]
pub struct HealthMonitor {
    /// Current connection state.
    state: RwLock<ConnectionState>,
    /// Latency tracker.
    latency: LatencyTracker,
    /// Request counter (rolling window).
    request_counter: AtomicU64,
    /// Error counter (rolling window).
    error_counter: AtomicU64,
    /// Total requests.
    total_requests: AtomicU64,
    /// Total errors.
    total_errors: AtomicU64,
    /// Last heartbeat time.
    last_heartbeat: RwLock<Instant>,
    /// Consecutive heartbeat failures.
    heartbeat_failures: AtomicU32,
    /// Start time.
    start_time: Instant,
    /// Last window reset time.
    last_window_reset: RwLock<Instant>,
    /// Window duration for rate calculation.
    window_duration: Duration,
    /// Health thresholds.
    thresholds: RwLock<HealthThresholds>,
}

impl HealthMonitor {
    /// Create a new health monitor.
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            state: RwLock::new(ConnectionState::Disconnected),
            latency: LatencyTracker::new(1000),
            request_counter: AtomicU64::new(0),
            error_counter: AtomicU64::new(0),
            total_requests: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
            last_heartbeat: RwLock::new(now),
            heartbeat_failures: AtomicU32::new(0),
            start_time: now,
            last_window_reset: RwLock::new(now),
            window_duration: Duration::from_secs(10),
            thresholds: RwLock::new(HealthThresholds::default()),
        }
    }

    /// Create with custom thresholds.
    pub fn with_thresholds(thresholds: HealthThresholds) -> Self {
        let monitor = Self::new();
        *monitor.thresholds.write() = thresholds;
        monitor
    }

    /// Record a successful request.
    pub fn record_request(&self, latency: Duration) {
        self.request_counter.fetch_add(1, Ordering::Relaxed);
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.latency.record(latency);
        self.maybe_reset_window();
        self.update_state();
    }

    /// Record an error.
    pub fn record_error(&self) {
        self.error_counter.fetch_add(1, Ordering::Relaxed);
        self.total_errors.fetch_add(1, Ordering::Relaxed);
        self.maybe_reset_window();
        self.update_state();
    }

    /// Record heartbeat received.
    pub fn heartbeat_received(&self) {
        *self.last_heartbeat.write() = Instant::now();
        self.heartbeat_failures.store(0, Ordering::Relaxed);
        self.update_state();
    }

    /// Record heartbeat failure.
    pub fn heartbeat_failed(&self) {
        self.heartbeat_failures.fetch_add(1, Ordering::Relaxed);
        self.update_state();
    }

    /// Mark as connected.
    pub fn set_connected(&self) {
        *self.state.write() = ConnectionState::Connected;
        self.heartbeat_failures.store(0, Ordering::Relaxed);
        *self.last_heartbeat.write() = Instant::now();
    }

    /// Mark as disconnected.
    pub fn set_disconnected(&self) {
        *self.state.write() = ConnectionState::Disconnected;
    }

    /// Mark as reconnecting.
    pub fn set_reconnecting(&self) {
        *self.state.write() = ConnectionState::Reconnecting;
    }

    /// Get current connection state.
    pub fn state(&self) -> ConnectionState {
        *self.state.read()
    }

    /// Check if connection is operational.
    pub fn is_operational(&self) -> bool {
        self.state().is_operational()
    }

    /// Get current metrics snapshot.
    pub fn metrics(&self) -> HealthMetrics {
        let window_elapsed = self.last_window_reset.read().elapsed().as_secs_f64();
        let requests = self.request_counter.load(Ordering::Relaxed) as f64;
        let errors = self.error_counter.load(Ordering::Relaxed) as f64;

        let requests_per_sec = if window_elapsed > 0.0 {
            requests / window_elapsed
        } else {
            0.0
        };

        let errors_per_sec = if window_elapsed > 0.0 {
            errors / window_elapsed
        } else {
            0.0
        };

        let total_reqs = self.total_requests.load(Ordering::Relaxed) as f64;
        let total_errs = self.total_errors.load(Ordering::Relaxed) as f64;
        let error_rate = if total_reqs > 0.0 {
            total_errs / total_reqs
        } else {
            0.0
        };

        HealthMetrics {
            state: self.state(),
            requests_per_sec,
            errors_per_sec,
            error_rate,
            avg_latency_ms: self.latency.average().as_secs_f64() * 1000.0,
            p50_latency_ms: self.latency.p50().as_secs_f64() * 1000.0,
            p95_latency_ms: self.latency.p95().as_secs_f64() * 1000.0,
            p99_latency_ms: self.latency.p99().as_secs_f64() * 1000.0,
            min_latency_ms: self.latency.min().as_secs_f64() * 1000.0,
            max_latency_ms: self.latency.max().as_secs_f64() * 1000.0,
            pending_requests: 0, // Would be filled by TimeoutTracker
            total_requests: self.total_requests.load(Ordering::Relaxed),
            total_errors: self.total_errors.load(Ordering::Relaxed),
            total_timeouts: 0, // Would be filled by TimeoutTracker
            uptime_ms: self.start_time.elapsed().as_millis() as u64,
            last_heartbeat_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0)
                - self.last_heartbeat.read().elapsed().as_millis() as u64,
            heartbeat_failures: self.heartbeat_failures.load(Ordering::Relaxed),
        }
    }

    /// Get latency tracker.
    pub fn latency(&self) -> &LatencyTracker {
        &self.latency
    }

    /// Get uptime.
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Reset window counters if window has elapsed.
    fn maybe_reset_window(&self) {
        let mut last_reset = self.last_window_reset.write();
        if last_reset.elapsed() > self.window_duration {
            self.request_counter.store(0, Ordering::Relaxed);
            self.error_counter.store(0, Ordering::Relaxed);
            *last_reset = Instant::now();
        }
    }

    /// Update connection state based on current metrics.
    fn update_state(&self) {
        let thresholds = self.thresholds.read();
        let current_state = *self.state.read();

        // Don't update if we're reconnecting
        if current_state == ConnectionState::Reconnecting {
            return;
        }

        // Check heartbeat failures
        let failures = self.heartbeat_failures.load(Ordering::Relaxed);
        if failures >= thresholds.max_heartbeat_failures {
            *self.state.write() = ConnectionState::Disconnected;
            return;
        }

        // Check if we're connected
        if current_state == ConnectionState::Disconnected {
            return;
        }

        // Calculate current error rate and latency
        let total = self.total_requests.load(Ordering::Relaxed) as f64;
        let errors = self.total_errors.load(Ordering::Relaxed) as f64;
        let error_rate = if total > 0.0 { errors / total } else { 0.0 };
        let avg_latency_ms = self.latency.average().as_millis() as u64;

        // Determine state
        let new_state = if error_rate > thresholds.degraded_error_rate
            || avg_latency_ms > thresholds.degraded_latency_ms
            || failures > 0
        {
            ConnectionState::Degraded
        } else {
            ConnectionState::Connected
        };

        *self.state.write() = new_state;
    }
}

impl Default for HealthMonitor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Reconnection Manager
// ---------------------------------------------------------------------------

/// Configuration for reconnection behavior.
#[derive(Debug, Clone)]
pub struct ReconnectionConfig {
    /// Initial delay before first retry.
    pub initial_delay: Duration,
    /// Maximum delay between retries.
    pub max_delay: Duration,
    /// Multiplier for exponential backoff.
    pub multiplier: f64,
    /// Maximum number of retry attempts (0 = unlimited).
    pub max_attempts: u32,
    /// Whether to add jitter to delay.
    pub jitter: bool,
    /// Jitter factor (0.0 - 1.0).
    pub jitter_factor: f64,
}

impl Default for ReconnectionConfig {
    fn default() -> Self {
        Self {
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            multiplier: 2.0,
            max_attempts: 10,
            jitter: true,
            jitter_factor: 0.25,
        }
    }
}

/// State of a reconnection attempt.
#[derive(Debug, Clone)]
pub struct ReconnectionState {
    /// Current attempt number (0 = first attempt).
    pub attempt: u32,
    /// Current delay before next attempt.
    pub current_delay: Duration,
    /// When the last attempt was made.
    pub last_attempt: Option<Instant>,
    /// Whether reconnection is in progress.
    pub in_progress: bool,
    /// Pending requests to retry after reconnection.
    pub pending_requests: Vec<PendingRequest>,
}

impl Default for ReconnectionState {
    fn default() -> Self {
        Self {
            attempt: 0,
            current_delay: Duration::ZERO,
            last_attempt: None,
            in_progress: false,
            pending_requests: Vec::new(),
        }
    }
}

/// Manages reconnection with exponential backoff.
#[derive(Debug)]
pub struct ReconnectionManager {
    /// Configuration.
    config: RwLock<ReconnectionConfig>,
    /// Current state.
    state: RwLock<ReconnectionState>,
    /// Simple RNG state for jitter.
    rng_state: AtomicU64,
}

impl ReconnectionManager {
    /// Create a new reconnection manager.
    pub fn new() -> Self {
        Self {
            config: RwLock::new(ReconnectionConfig::default()),
            state: RwLock::new(ReconnectionState::default()),
            rng_state: AtomicU64::new(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(12345),
            ),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: ReconnectionConfig) -> Self {
        let manager = Self::new();
        *manager.config.write() = config;
        manager
    }

    /// Start a reconnection sequence.
    pub fn start(&self, pending_requests: Vec<PendingRequest>) {
        let mut state = self.state.write();
        state.attempt = 0;
        state.current_delay = self.config.read().initial_delay;
        state.last_attempt = None;
        state.in_progress = true;
        state.pending_requests = pending_requests;
    }

    /// Get the delay before next attempt.
    pub fn next_delay(&self) -> Option<Duration> {
        let state = self.state.read();
        if !state.in_progress {
            return None;
        }

        let config = self.config.read();
        if config.max_attempts > 0 && state.attempt >= config.max_attempts {
            return None;
        }

        Some(self.calculate_delay(state.attempt))
    }

    /// Record an attempt and advance state.
    pub fn record_attempt(&self) {
        let mut state = self.state.write();
        state.attempt += 1;
        state.last_attempt = Some(Instant::now());
        state.current_delay = self.calculate_delay(state.attempt);
    }

    /// Mark reconnection as successful.
    pub fn success(&self) -> Vec<PendingRequest> {
        let mut state = self.state.write();
        state.in_progress = false;
        let pending = std::mem::take(&mut state.pending_requests);
        state.attempt = 0;
        state.current_delay = Duration::ZERO;
        pending
    }

    /// Mark reconnection as failed (exhausted retries).
    pub fn failed(&self) -> Vec<PendingRequest> {
        let mut state = self.state.write();
        state.in_progress = false;
        std::mem::take(&mut state.pending_requests)
    }

    /// Reset the manager.
    pub fn reset(&self) {
        *self.state.write() = ReconnectionState::default();
    }

    /// Get current state.
    pub fn state(&self) -> ReconnectionState {
        self.state.read().clone()
    }

    /// Check if reconnection is in progress.
    pub fn is_reconnecting(&self) -> bool {
        self.state.read().in_progress
    }

    /// Check if max attempts exhausted.
    pub fn is_exhausted(&self) -> bool {
        let state = self.state.read();
        let config = self.config.read();
        config.max_attempts > 0 && state.attempt >= config.max_attempts
    }

    /// Get current attempt number.
    pub fn attempt(&self) -> u32 {
        self.state.read().attempt
    }

    /// Calculate delay for given attempt with backoff and jitter.
    fn calculate_delay(&self, attempt: u32) -> Duration {
        let config = self.config.read();

        // Exponential backoff
        let base_delay =
            config.initial_delay.as_secs_f64() * config.multiplier.powi(attempt as i32);
        let capped_delay = base_delay.min(config.max_delay.as_secs_f64());

        // Add jitter if enabled
        let final_delay = if config.jitter {
            let jitter_range = capped_delay * config.jitter_factor;
            let jitter = self.random_f64() * jitter_range * 2.0 - jitter_range;
            (capped_delay + jitter).max(0.0)
        } else {
            capped_delay
        };

        Duration::from_secs_f64(final_delay)
    }

    /// Simple PRNG for jitter (xorshift64).
    fn random_f64(&self) -> f64 {
        let state = self.rng_state.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |mut x| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            Some(x)
        });

        let val = state.unwrap_or(12345);
        (val as f64) / (u64::MAX as f64)
    }
}

impl Default for ReconnectionManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Integrated Bridge Health System
// ---------------------------------------------------------------------------

/// Integrated bridge health system combining all monitoring components.
#[derive(Debug)]
pub struct BridgeHealthSystem {
    /// Health monitor.
    pub monitor: HealthMonitor,
    /// Timeout tracker.
    pub timeouts: TimeoutTracker,
    /// Reconnection manager.
    pub reconnection: ReconnectionManager,
}

impl BridgeHealthSystem {
    /// Create a new bridge health system with defaults.
    pub fn new() -> Self {
        Self {
            monitor: HealthMonitor::new(),
            timeouts: TimeoutTracker::new(),
            reconnection: ReconnectionManager::new(),
        }
    }

    /// Create with custom configurations.
    pub fn with_config(
        health_thresholds: HealthThresholds,
        timeout_config: TimeoutConfig,
        reconnection_config: ReconnectionConfig,
    ) -> Self {
        Self {
            monitor: HealthMonitor::with_thresholds(health_thresholds),
            timeouts: TimeoutTracker::with_config(timeout_config),
            reconnection: ReconnectionManager::with_config(reconnection_config),
        }
    }

    /// Register a request for tracking.
    pub fn register_request(&self, id: u64, namespace: &str, method: &str) {
        self.timeouts.register(id, namespace, method);
    }

    /// Complete a request and record metrics.
    pub fn complete_request(&self, id: u64) -> Option<Duration> {
        let elapsed = self.timeouts.complete(id);
        if let Some(duration) = elapsed {
            self.monitor.record_request(duration);
        }
        elapsed
    }

    /// Record an error for a request.
    pub fn record_error(&self, id: u64) {
        self.timeouts.complete(id);
        self.monitor.record_error();
    }

    /// Check for and handle timeouts.
    pub fn check_timeouts(&self) -> Vec<ErrorResponse> {
        let timed_out = self.timeouts.check_timeouts();
        timed_out
            .iter()
            .map(|r| {
                ErrorResponse::timeout(r.id, &r.namespace, &r.method, r.timeout.as_millis() as u64)
            })
            .collect()
    }

    /// Handle connection loss.
    pub fn on_connection_lost(&self) -> ErrorResponse {
        self.monitor.set_disconnected();

        // Collect pending requests for reconnection
        let pending: Vec<_> = self
            .timeouts
            .pending_ids()
            .iter()
            .filter_map(|id| self.timeouts.get(*id))
            .collect();

        let pending_count = pending.len();
        self.reconnection.start(pending);

        ErrorResponse::connection_lost(pending_count)
    }

    /// Handle successful reconnection.
    pub fn on_reconnected(&self) -> Vec<PendingRequest> {
        self.monitor.set_connected();
        self.reconnection.success()
    }

    /// Get comprehensive health metrics.
    pub fn health_metrics(&self) -> HealthMetrics {
        let mut metrics = self.monitor.metrics();
        metrics.pending_requests = self.timeouts.pending_count() as u32;
        metrics.total_timeouts = self.timeouts.total_timeouts();
        metrics
    }

    /// Check if the system is healthy enough to process requests.
    pub fn is_operational(&self) -> bool {
        self.monitor.is_operational()
    }
}

impl Default for BridgeHealthSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ===== ErrorCategory tests =====

    #[test]
    fn test_error_category_is_retryable() {
        assert!(ErrorCategory::Connection.is_retryable());
        assert!(ErrorCategory::Timeout.is_retryable());
        assert!(ErrorCategory::Internal.is_retryable());
        assert!(!ErrorCategory::Validation.is_retryable());
        assert!(!ErrorCategory::Permission.is_retryable());
        assert!(!ErrorCategory::Resource.is_retryable());
    }

    #[test]
    fn test_error_category_suggested_retry_delay() {
        assert_eq!(ErrorCategory::Connection.suggested_retry_delay_ms(), Some(1000));
        assert_eq!(ErrorCategory::Timeout.suggested_retry_delay_ms(), Some(500));
        assert_eq!(ErrorCategory::Internal.suggested_retry_delay_ms(), Some(100));
        assert_eq!(ErrorCategory::Validation.suggested_retry_delay_ms(), None);
    }

    // ===== ErrorCode tests =====

    #[test]
    fn test_error_code_category() {
        assert_eq!(ErrorCode::NotConnected.category(), ErrorCategory::Connection);
        assert_eq!(ErrorCode::VersionMismatch.category(), ErrorCategory::Protocol);
        assert_eq!(ErrorCode::InvalidParams.category(), ErrorCategory::Validation);
        assert_eq!(ErrorCode::EntityNotFound.category(), ErrorCategory::Resource);
        assert_eq!(ErrorCode::PermissionDenied.category(), ErrorCategory::Permission);
        assert_eq!(ErrorCode::InternalError.category(), ErrorCategory::Internal);
        assert_eq!(ErrorCode::RequestTimeout.category(), ErrorCategory::Timeout);
        assert_eq!(ErrorCode::RequestCancelled.category(), ErrorCategory::Cancelled);
    }

    #[test]
    fn test_error_code_description() {
        assert!(!ErrorCode::NotConnected.description().is_empty());
        assert!(!ErrorCode::EntityNotFound.description().is_empty());
    }

    #[test]
    fn test_error_code_from_bridge_error() {
        assert_eq!(ErrorCode::from(&BridgeError::NotConnected), ErrorCode::NotConnected);
        assert_eq!(
            ErrorCode::from(&BridgeError::Timeout("test".into())),
            ErrorCode::RequestTimeout
        );
        assert_eq!(
            ErrorCode::from(&BridgeError::EntityNotFound(123)),
            ErrorCode::EntityNotFound
        );
        assert_eq!(ErrorCode::from(&BridgeError::Cancelled), ErrorCode::RequestCancelled);
    }

    // ===== ErrorResponse tests =====

    #[test]
    fn test_error_response_from_bridge_error() {
        let error = BridgeError::EntityNotFound(42);
        let response = ErrorResponse::from_bridge_error(&error, Some(100));

        assert_eq!(response.code, ErrorCode::EntityNotFound as u32);
        assert_eq!(response.category, ErrorCategory::Resource);
        assert!(response.message.contains("42"));
        assert_eq!(response.request_id, Some(100));
        assert!(!response.retryable);
    }

    #[test]
    fn test_error_response_timeout() {
        let response = ErrorResponse::timeout(1, "entity", "spawn", 5000);

        assert_eq!(response.code, ErrorCode::RequestTimeout as u32);
        assert_eq!(response.category, ErrorCategory::Timeout);
        assert!(response.message.contains("entity.spawn"));
        assert!(response.message.contains("5000"));
        assert!(response.retryable);
        assert_eq!(response.request_id, Some(1));
    }

    #[test]
    fn test_error_response_connection_lost() {
        let response = ErrorResponse::connection_lost(5);

        assert_eq!(response.code, ErrorCode::ConnectionLost as u32);
        assert!(response.message.contains("5"));
        assert!(response.retryable);
    }

    #[test]
    fn test_error_response_serialization() {
        let error = BridgeError::NotConnected;
        let response = ErrorResponse::from_bridge_error(&error, None);

        let json = response.to_json().unwrap();
        let deserialized = ErrorResponse::from_json(&json).unwrap();

        assert_eq!(deserialized.code, response.code);
        assert_eq!(deserialized.category, response.category);
    }

    // ===== TimeoutConfig tests =====

    #[test]
    fn test_timeout_config_default() {
        let config = TimeoutConfig::new();
        assert_eq!(config.default_timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_timeout_config_namespace_timeout() {
        let config = TimeoutConfig::new();
        // "bridge" is configured with 5s default
        assert_eq!(config.get_timeout("bridge", "health"), Duration::from_secs(5));
    }

    #[test]
    fn test_timeout_config_method_timeout() {
        let config = TimeoutConfig::new();
        // "asset.load" is configured with 120s
        assert_eq!(config.get_timeout("asset", "load"), Duration::from_secs(120));
    }

    #[test]
    fn test_timeout_config_fallback() {
        let config = TimeoutConfig::new();
        // Unknown namespace falls back to default
        assert_eq!(config.get_timeout("unknown", "method"), Duration::from_secs(30));
    }

    #[test]
    fn test_timeout_config_custom() {
        let mut config = TimeoutConfig::default();
        config.set_namespace_timeout("custom", Duration::from_secs(99));
        config.set_method_timeout("custom", "special", Duration::from_secs(999));

        assert_eq!(config.get_timeout("custom", "normal"), Duration::from_secs(99));
        assert_eq!(config.get_timeout("custom", "special"), Duration::from_secs(999));
    }

    // ===== PendingRequest tests =====

    #[test]
    fn test_pending_request_is_expired() {
        let request = PendingRequest {
            id: 1,
            namespace: "test".into(),
            method: "method".into(),
            start_time: Instant::now() - Duration::from_secs(10),
            timeout: Duration::from_secs(5),
            timed_out: false,
        };

        assert!(request.is_expired());
    }

    #[test]
    fn test_pending_request_not_expired() {
        let request = PendingRequest {
            id: 1,
            namespace: "test".into(),
            method: "method".into(),
            start_time: Instant::now(),
            timeout: Duration::from_secs(5),
            timed_out: false,
        };

        assert!(!request.is_expired());
    }

    #[test]
    fn test_pending_request_remaining() {
        let request = PendingRequest {
            id: 1,
            namespace: "test".into(),
            method: "method".into(),
            start_time: Instant::now(),
            timeout: Duration::from_secs(5),
            timed_out: false,
        };

        assert!(request.remaining() <= Duration::from_secs(5));
        assert!(request.remaining() > Duration::from_secs(4));
    }

    // ===== TimeoutTracker tests =====

    #[test]
    fn test_timeout_tracker_register_complete() {
        let tracker = TimeoutTracker::new();
        tracker.register(1, "entity", "spawn");

        assert_eq!(tracker.pending_count(), 1);

        let elapsed = tracker.complete(1);
        assert!(elapsed.is_some());
        assert_eq!(tracker.pending_count(), 0);
    }

    #[test]
    fn test_timeout_tracker_check_timeouts() {
        let tracker = TimeoutTracker::new();
        tracker.register_with_timeout(1, "test", "method", Duration::from_millis(1));

        std::thread::sleep(Duration::from_millis(10));

        let timed_out = tracker.check_timeouts();
        assert_eq!(timed_out.len(), 1);
        assert_eq!(timed_out[0].id, 1);
    }

    #[test]
    fn test_timeout_tracker_cleanup_expired() {
        let tracker = TimeoutTracker::new();
        tracker.register_with_timeout(1, "test", "method", Duration::from_millis(1));

        std::thread::sleep(Duration::from_millis(10));

        // First check marks as timed out
        tracker.check_timeouts();

        // Cleanup removes them
        let cleaned = tracker.cleanup_expired();
        assert_eq!(cleaned.len(), 1);
        assert_eq!(tracker.pending_count(), 0);
    }

    #[test]
    fn test_timeout_tracker_multiple_requests() {
        let tracker = TimeoutTracker::new();

        for i in 0..10 {
            tracker.register(i, "test", "method");
        }

        assert_eq!(tracker.pending_count(), 10);
        assert_eq!(tracker.pending_ids().len(), 10);
    }

    #[test]
    fn test_timeout_tracker_stats() {
        let tracker = TimeoutTracker::new();

        tracker.register(1, "test", "method");
        tracker.complete(1);

        assert_eq!(tracker.total_requests(), 1);
        assert_eq!(tracker.total_timeouts(), 0);
    }

    // ===== ConnectionState tests =====

    #[test]
    fn test_connection_state_is_operational() {
        assert!(ConnectionState::Connected.is_operational());
        assert!(ConnectionState::Degraded.is_operational());
        assert!(!ConnectionState::Disconnected.is_operational());
        assert!(!ConnectionState::Reconnecting.is_operational());
    }

    #[test]
    fn test_connection_state_is_healthy() {
        assert!(ConnectionState::Connected.is_healthy());
        assert!(!ConnectionState::Degraded.is_healthy());
        assert!(!ConnectionState::Disconnected.is_healthy());
    }

    // ===== LatencyTracker tests =====

    #[test]
    fn test_latency_tracker_record() {
        let tracker = LatencyTracker::new(100);

        tracker.record(Duration::from_millis(10));
        tracker.record(Duration::from_millis(20));
        tracker.record(Duration::from_millis(30));

        assert_eq!(tracker.sample_count(), 3);
        assert_eq!(tracker.total_samples(), 3);
    }

    #[test]
    fn test_latency_tracker_average() {
        let tracker = LatencyTracker::new(100);

        tracker.record(Duration::from_millis(10));
        tracker.record(Duration::from_millis(20));
        tracker.record(Duration::from_millis(30));

        // Average should be ~20ms = 20000us
        let avg_us = tracker.average_us();
        assert!(avg_us >= 19000 && avg_us <= 21000);
    }

    #[test]
    fn test_latency_tracker_min_max() {
        let tracker = LatencyTracker::new(100);

        tracker.record(Duration::from_millis(10));
        tracker.record(Duration::from_millis(50));
        tracker.record(Duration::from_millis(30));

        assert!(tracker.min() >= Duration::from_millis(9) && tracker.min() <= Duration::from_millis(11));
        assert!(tracker.max() >= Duration::from_millis(49) && tracker.max() <= Duration::from_millis(51));
    }

    #[test]
    fn test_latency_tracker_percentiles() {
        let tracker = LatencyTracker::new(100);

        for i in 1..=100 {
            tracker.record(Duration::from_millis(i));
        }

        // P50 should be around 50ms
        let p50 = tracker.p50().as_millis();
        assert!(p50 >= 40 && p50 <= 60);

        // P99 should be around 99ms
        let p99 = tracker.p99().as_millis();
        assert!(p99 >= 90);
    }

    #[test]
    fn test_latency_tracker_respects_capacity() {
        let tracker = LatencyTracker::new(5);

        for i in 1..=10 {
            tracker.record(Duration::from_millis(i));
        }

        // Should only keep last 5
        assert_eq!(tracker.sample_count(), 5);
    }

    #[test]
    fn test_latency_tracker_reset() {
        let tracker = LatencyTracker::new(100);
        tracker.record(Duration::from_millis(10));

        tracker.reset();

        assert_eq!(tracker.sample_count(), 0);
        assert_eq!(tracker.average_us(), 0);
    }

    // ===== HealthMonitor tests =====

    #[test]
    fn test_health_monitor_initial_state() {
        let monitor = HealthMonitor::new();
        assert_eq!(monitor.state(), ConnectionState::Disconnected);
    }

    #[test]
    fn test_health_monitor_set_connected() {
        let monitor = HealthMonitor::new();
        monitor.set_connected();
        assert_eq!(monitor.state(), ConnectionState::Connected);
    }

    #[test]
    fn test_health_monitor_record_request() {
        let monitor = HealthMonitor::new();
        monitor.set_connected();

        monitor.record_request(Duration::from_millis(10));

        let metrics = monitor.metrics();
        assert_eq!(metrics.total_requests, 1);
    }

    #[test]
    fn test_health_monitor_record_error() {
        let monitor = HealthMonitor::new();
        monitor.set_connected();

        monitor.record_error();

        let metrics = monitor.metrics();
        assert_eq!(metrics.total_errors, 1);
    }

    #[test]
    fn test_health_monitor_heartbeat() {
        let monitor = HealthMonitor::new();
        monitor.set_connected();

        monitor.heartbeat_received();

        let metrics = monitor.metrics();
        assert_eq!(metrics.heartbeat_failures, 0);
    }

    #[test]
    fn test_health_monitor_heartbeat_failure() {
        let monitor = HealthMonitor::new();
        monitor.set_connected();

        monitor.heartbeat_failed();

        let metrics = monitor.metrics();
        assert_eq!(metrics.heartbeat_failures, 1);
    }

    #[test]
    fn test_health_monitor_degraded_on_errors() {
        let thresholds = HealthThresholds {
            degraded_error_rate: 0.1,
            ..Default::default()
        };
        let monitor = HealthMonitor::with_thresholds(thresholds);
        monitor.set_connected();

        // Generate errors to exceed threshold
        for _ in 0..20 {
            monitor.record_error();
        }
        for _ in 0..80 {
            monitor.record_request(Duration::from_millis(1));
        }

        // Should be degraded due to error rate
        assert_eq!(monitor.state(), ConnectionState::Degraded);
    }

    #[test]
    fn test_health_monitor_disconnected_on_heartbeat_failures() {
        let thresholds = HealthThresholds {
            max_heartbeat_failures: 3,
            ..Default::default()
        };
        let monitor = HealthMonitor::with_thresholds(thresholds);
        monitor.set_connected();

        monitor.heartbeat_failed();
        monitor.heartbeat_failed();
        monitor.heartbeat_failed();

        assert_eq!(monitor.state(), ConnectionState::Disconnected);
    }

    // ===== ReconnectionConfig tests =====

    #[test]
    fn test_reconnection_config_default() {
        let config = ReconnectionConfig::default();
        assert_eq!(config.initial_delay, Duration::from_millis(100));
        assert_eq!(config.max_attempts, 10);
        assert!(config.jitter);
    }

    // ===== ReconnectionManager tests =====

    #[test]
    fn test_reconnection_manager_start() {
        let manager = ReconnectionManager::new();

        manager.start(vec![]);

        assert!(manager.is_reconnecting());
        assert_eq!(manager.attempt(), 0);
    }

    #[test]
    fn test_reconnection_manager_record_attempt() {
        let manager = ReconnectionManager::new();
        manager.start(vec![]);

        manager.record_attempt();

        assert_eq!(manager.attempt(), 1);
    }

    #[test]
    fn test_reconnection_manager_success() {
        let manager = ReconnectionManager::new();

        let pending = vec![PendingRequest {
            id: 1,
            namespace: "test".into(),
            method: "method".into(),
            start_time: Instant::now(),
            timeout: Duration::from_secs(5),
            timed_out: false,
        }];

        manager.start(pending);
        let recovered = manager.success();

        assert!(!manager.is_reconnecting());
        assert_eq!(recovered.len(), 1);
    }

    #[test]
    fn test_reconnection_manager_exhausted() {
        let config = ReconnectionConfig {
            max_attempts: 3,
            ..Default::default()
        };
        let manager = ReconnectionManager::with_config(config);
        manager.start(vec![]);

        manager.record_attempt();
        manager.record_attempt();
        manager.record_attempt();

        assert!(manager.is_exhausted());
    }

    #[test]
    fn test_reconnection_manager_backoff() {
        let config = ReconnectionConfig {
            initial_delay: Duration::from_millis(100),
            multiplier: 2.0,
            max_delay: Duration::from_secs(10),
            jitter: false,
            ..Default::default()
        };
        let manager = ReconnectionManager::with_config(config);
        manager.start(vec![]);

        let delay0 = manager.next_delay().unwrap();
        assert_eq!(delay0, Duration::from_millis(100));

        manager.record_attempt();
        let delay1 = manager.next_delay().unwrap();
        assert_eq!(delay1, Duration::from_millis(200));

        manager.record_attempt();
        let delay2 = manager.next_delay().unwrap();
        assert_eq!(delay2, Duration::from_millis(400));
    }

    #[test]
    fn test_reconnection_manager_max_delay_cap() {
        let config = ReconnectionConfig {
            initial_delay: Duration::from_secs(1),
            multiplier: 10.0,
            max_delay: Duration::from_secs(5),
            jitter: false,
            ..Default::default()
        };
        let manager = ReconnectionManager::with_config(config);
        manager.start(vec![]);

        manager.record_attempt();
        manager.record_attempt();

        let delay = manager.next_delay().unwrap();
        assert!(delay <= Duration::from_secs(5));
    }

    #[test]
    fn test_reconnection_manager_reset() {
        let manager = ReconnectionManager::new();
        manager.start(vec![]);
        manager.record_attempt();

        manager.reset();

        assert!(!manager.is_reconnecting());
        assert_eq!(manager.attempt(), 0);
    }

    // ===== BridgeHealthSystem tests =====

    #[test]
    fn test_bridge_health_system_new() {
        let system = BridgeHealthSystem::new();
        assert!(!system.is_operational());
    }

    #[test]
    fn test_bridge_health_system_request_lifecycle() {
        let system = BridgeHealthSystem::new();
        system.monitor.set_connected();

        system.register_request(1, "entity", "spawn");
        std::thread::sleep(Duration::from_millis(1));
        let elapsed = system.complete_request(1);

        assert!(elapsed.is_some());
        assert!(elapsed.unwrap() >= Duration::from_millis(1));
    }

    #[test]
    fn test_bridge_health_system_record_error() {
        let system = BridgeHealthSystem::new();
        system.monitor.set_connected();

        system.register_request(1, "entity", "spawn");
        system.record_error(1);

        let metrics = system.health_metrics();
        assert_eq!(metrics.total_errors, 1);
    }

    #[test]
    fn test_bridge_health_system_check_timeouts() {
        let system = BridgeHealthSystem::new();

        system.timeouts.register_with_timeout(1, "test", "method", Duration::from_millis(1));
        std::thread::sleep(Duration::from_millis(10));

        let errors = system.check_timeouts();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].code, ErrorCode::RequestTimeout as u32);
    }

    #[test]
    fn test_bridge_health_system_connection_lost() {
        let system = BridgeHealthSystem::new();
        system.monitor.set_connected();
        system.register_request(1, "test", "method");

        let error = system.on_connection_lost();

        assert_eq!(error.code, ErrorCode::ConnectionLost as u32);
        assert!(!system.is_operational());
        assert!(system.reconnection.is_reconnecting());
    }

    #[test]
    fn test_bridge_health_system_reconnected() {
        let system = BridgeHealthSystem::new();

        let pending = vec![PendingRequest {
            id: 1,
            namespace: "test".into(),
            method: "method".into(),
            start_time: Instant::now(),
            timeout: Duration::from_secs(5),
            timed_out: false,
        }];

        system.reconnection.start(pending);
        let recovered = system.on_reconnected();

        assert!(system.is_operational());
        assert_eq!(recovered.len(), 1);
    }

    #[test]
    fn test_bridge_health_system_metrics() {
        let system = BridgeHealthSystem::new();
        system.monitor.set_connected();

        system.register_request(1, "test", "method");
        system.complete_request(1);

        let metrics = system.health_metrics();
        assert_eq!(metrics.total_requests, 1);
        assert_eq!(metrics.state, ConnectionState::Connected);
    }
}
