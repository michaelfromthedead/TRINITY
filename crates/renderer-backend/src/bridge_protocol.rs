//! Three-Channel Bridge Protocol for Python-Rust Communication
//!
//! This module implements a general-purpose bridge protocol for communication
//! between Python and Rust components of the TRINITY engine. The protocol uses
//! three distinct channels:
//!
//! - **Type Channel**: Schema definitions, type metadata, registration
//! - **Data Channel**: Bulk data transfer, serialized payloads
//! - **Command Channel**: RPC-style function calls with request/response
//!
//! # Architecture
//!
//! ```text
//! Python (trinity.bridge)              Rust (bridge_protocol)
//! =====================              =====================
//!       │                                    │
//!       ├─── Type Channel ──────────────────►│ TypeSchema registration
//!       │    (schema sync)                   │
//!       │                                    │
//!       ├─── Data Channel ──────────────────►│ Bulk transfers
//!       │    (buffers, arrays)               │
//!       │                                    │
//!       ├─── Command Channel ───────────────►│ RPC dispatch
//!       │    (request/response)              │
//!       │◄──────────────────────────────────┤
//!       │                                    │
//! ```
//!
//! # Endpoint Namespaces
//!
//! | Namespace    | Description                        |
//! |--------------|------------------------------------|
//! | `bridge.*`   | Connection lifecycle, health       |
//! | `entity.*`   | Spawn, destroy, query entities     |
//! | `component.*`| Get, set, list components          |
//! | `frame.*`    | Frame timing, statistics           |
//! | `profiler.*` | GPU/CPU timing data                |
//! | `editor.*`   | Selection, viewport, panels        |
//! | `material.*` | Material parameters, compilation   |
//! | `asset.*`    | Asset loading, hot-reload triggers |

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Protocol Version
// ---------------------------------------------------------------------------

/// Protocol version for compatibility checking.
pub const PROTOCOL_VERSION: u32 = 1;

/// Protocol magic number for message validation.
pub const PROTOCOL_MAGIC: u32 = 0x5452_494E; // "TRIN"

// ---------------------------------------------------------------------------
// Channel Types
// ---------------------------------------------------------------------------

/// The three communication channels in the bridge protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChannelKind {
    /// Type channel: schema definitions and type metadata.
    Type,
    /// Data channel: bulk data transfer.
    Data,
    /// Command channel: RPC-style function calls.
    Command,
}

impl ChannelKind {
    /// Returns the channel priority (lower = higher priority).
    pub fn priority(&self) -> u8 {
        match self {
            ChannelKind::Command => 0,
            ChannelKind::Type => 1,
            ChannelKind::Data => 2,
        }
    }
}

// ---------------------------------------------------------------------------
// Message Envelope
// ---------------------------------------------------------------------------

/// A message envelope wrapping payload data for transport.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeMessage {
    /// Unique message ID for tracking and correlation.
    pub id: u64,
    /// The channel this message belongs to.
    pub channel: ChannelKind,
    /// Timestamp when the message was created (milliseconds since epoch).
    pub timestamp_ms: u64,
    /// Serialized payload data.
    #[serde(with = "serde_bytes")]
    pub payload: Vec<u8>,
}

impl BridgeMessage {
    /// Create a new bridge message with auto-generated ID.
    pub fn new(channel: ChannelKind, payload: Vec<u8>) -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            channel,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            payload,
        }
    }

    /// Create a message with a specific ID.
    pub fn with_id(id: u64, channel: ChannelKind, payload: Vec<u8>) -> Self {
        Self {
            id,
            channel,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            payload,
        }
    }

    /// Serialize the message to JSON bytes.
    pub fn to_json(&self) -> Result<Vec<u8>, BridgeError> {
        serde_json::to_vec(self).map_err(|e| BridgeError::Serialization(e.to_string()))
    }

    /// Deserialize a message from JSON bytes.
    pub fn from_json(data: &[u8]) -> Result<Self, BridgeError> {
        serde_json::from_slice(data).map_err(|e| BridgeError::Deserialization(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Error Types
// ---------------------------------------------------------------------------

/// Errors that can occur in the bridge protocol.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "message")]
pub enum BridgeError {
    /// Connection not established.
    NotConnected,
    /// Connection timeout.
    Timeout(String),
    /// Serialization error.
    Serialization(String),
    /// Deserialization error.
    Deserialization(String),
    /// Unknown namespace.
    UnknownNamespace(String),
    /// Unknown method.
    UnknownMethod(String),
    /// Invalid parameters.
    InvalidParams(String),
    /// Internal error.
    Internal(String),
    /// Version mismatch.
    VersionMismatch { expected: u32, got: u32 },
    /// Type not registered.
    TypeNotRegistered(String),
    /// Entity not found.
    EntityNotFound(u64),
    /// Component not found.
    ComponentNotFound(String),
    /// Asset not found.
    AssetNotFound(String),
    /// Permission denied.
    PermissionDenied(String),
    /// Resource exhausted.
    ResourceExhausted(String),
    /// Request cancelled.
    Cancelled,
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BridgeError::NotConnected => write!(f, "Bridge not connected"),
            BridgeError::Timeout(msg) => write!(f, "Timeout: {}", msg),
            BridgeError::Serialization(msg) => write!(f, "Serialization error: {}", msg),
            BridgeError::Deserialization(msg) => write!(f, "Deserialization error: {}", msg),
            BridgeError::UnknownNamespace(ns) => write!(f, "Unknown namespace: {}", ns),
            BridgeError::UnknownMethod(m) => write!(f, "Unknown method: {}", m),
            BridgeError::InvalidParams(msg) => write!(f, "Invalid parameters: {}", msg),
            BridgeError::Internal(msg) => write!(f, "Internal error: {}", msg),
            BridgeError::VersionMismatch { expected, got } => {
                write!(f, "Version mismatch: expected {}, got {}", expected, got)
            }
            BridgeError::TypeNotRegistered(t) => write!(f, "Type not registered: {}", t),
            BridgeError::EntityNotFound(id) => write!(f, "Entity not found: {}", id),
            BridgeError::ComponentNotFound(c) => write!(f, "Component not found: {}", c),
            BridgeError::AssetNotFound(a) => write!(f, "Asset not found: {}", a),
            BridgeError::PermissionDenied(msg) => write!(f, "Permission denied: {}", msg),
            BridgeError::ResourceExhausted(msg) => write!(f, "Resource exhausted: {}", msg),
            BridgeError::Cancelled => write!(f, "Request cancelled"),
        }
    }
}

impl std::error::Error for BridgeError {}

// ---------------------------------------------------------------------------
// Command Request/Response
// ---------------------------------------------------------------------------

/// A command request sent over the Command channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandRequest {
    /// Request ID for correlation.
    pub id: u64,
    /// Namespace (e.g., "entity", "component", "frame").
    pub namespace: String,
    /// Method name within the namespace.
    pub method: String,
    /// Parameters as JSON value.
    pub params: serde_json::Value,
    /// Optional timeout in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
}

impl CommandRequest {
    /// Create a new command request.
    pub fn new(
        namespace: impl Into<String>,
        method: impl Into<String>,
        params: serde_json::Value,
    ) -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            namespace: namespace.into(),
            method: method.into(),
            params,
            timeout_ms: None,
        }
    }

    /// Set a timeout for this request.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout_ms = Some(timeout.as_millis() as u64);
        self
    }

    /// Serialize to JSON bytes.
    pub fn to_json(&self) -> Result<Vec<u8>, BridgeError> {
        serde_json::to_vec(self).map_err(|e| BridgeError::Serialization(e.to_string()))
    }

    /// Deserialize from JSON bytes.
    pub fn from_json(data: &[u8]) -> Result<Self, BridgeError> {
        serde_json::from_slice(data).map_err(|e| BridgeError::Deserialization(e.to_string()))
    }
}

/// A command response sent over the Command channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandResponse {
    /// Request ID this response correlates to.
    pub id: u64,
    /// Result: either success value or error.
    #[serde(flatten)]
    pub result: CommandResult,
    /// Execution time in microseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_us: Option<u64>,
}

/// The result of a command execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status")]
pub enum CommandResult {
    /// Successful execution with return value.
    #[serde(rename = "ok")]
    Ok { value: serde_json::Value },
    /// Failed execution with error.
    #[serde(rename = "error")]
    Error { error: BridgeError },
}

impl CommandResponse {
    /// Create a successful response.
    pub fn ok(id: u64, value: serde_json::Value) -> Self {
        Self {
            id,
            result: CommandResult::Ok { value },
            execution_us: None,
        }
    }

    /// Create an error response.
    pub fn error(id: u64, error: BridgeError) -> Self {
        Self {
            id,
            result: CommandResult::Error { error },
            execution_us: None,
        }
    }

    /// Add execution time to the response.
    pub fn with_execution_time(mut self, duration: Duration) -> Self {
        self.execution_us = Some(duration.as_micros() as u64);
        self
    }

    /// Check if the response indicates success.
    pub fn is_ok(&self) -> bool {
        matches!(self.result, CommandResult::Ok { .. })
    }

    /// Get the success value, if any.
    pub fn value(&self) -> Option<&serde_json::Value> {
        match &self.result {
            CommandResult::Ok { value } => Some(value),
            CommandResult::Error { .. } => None,
        }
    }

    /// Get the error, if any.
    pub fn get_error(&self) -> Option<&BridgeError> {
        match &self.result {
            CommandResult::Ok { .. } => None,
            CommandResult::Error { error } => Some(error),
        }
    }

    /// Serialize to JSON bytes.
    pub fn to_json(&self) -> Result<Vec<u8>, BridgeError> {
        serde_json::to_vec(self).map_err(|e| BridgeError::Serialization(e.to_string()))
    }

    /// Deserialize from JSON bytes.
    pub fn from_json(data: &[u8]) -> Result<Self, BridgeError> {
        serde_json::from_slice(data).map_err(|e| BridgeError::Deserialization(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Type Schema (Type Channel)
// ---------------------------------------------------------------------------

/// Schema for a registered type in the bridge protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeSchema {
    /// Unique type name.
    pub name: String,
    /// Version of this schema.
    pub version: u32,
    /// Field definitions.
    pub fields: Vec<FieldSchema>,
    /// Optional documentation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc: Option<String>,
}

/// Schema for a field within a type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSchema {
    /// Field name.
    pub name: String,
    /// Field type.
    pub field_type: FieldType,
    /// Whether the field is optional.
    #[serde(default)]
    pub optional: bool,
    /// Default value (JSON).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,
    /// Optional documentation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc: Option<String>,
}

/// Supported field types in the schema system.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "params")]
pub enum FieldType {
    /// Boolean.
    Bool,
    /// 32-bit signed integer.
    I32,
    /// 64-bit signed integer.
    I64,
    /// 32-bit unsigned integer.
    U32,
    /// 64-bit unsigned integer.
    U64,
    /// 32-bit float.
    F32,
    /// 64-bit float.
    F64,
    /// UTF-8 string.
    String,
    /// Byte array.
    Bytes,
    /// Array of another type.
    Array { element: Box<FieldType> },
    /// Map from string keys to values.
    Map { value: Box<FieldType> },
    /// Reference to another registered type.
    Ref { type_name: String },
    /// 2D vector (f32).
    Vec2,
    /// 3D vector (f32).
    Vec3,
    /// 4D vector (f32).
    Vec4,
    /// 4x4 matrix (f32).
    Mat4,
    /// Quaternion (f32).
    Quat,
    /// RGBA color (f32).
    Color,
    /// Entity ID.
    EntityId,
    /// Asset handle.
    AssetHandle,
}

impl TypeSchema {
    /// Create a new type schema.
    pub fn new(name: impl Into<String>, version: u32) -> Self {
        Self {
            name: name.into(),
            version,
            fields: Vec::new(),
            doc: None,
        }
    }

    /// Add a field to the schema.
    pub fn with_field(mut self, field: FieldSchema) -> Self {
        self.fields.push(field);
        self
    }

    /// Add documentation.
    pub fn with_doc(mut self, doc: impl Into<String>) -> Self {
        self.doc = Some(doc.into());
        self
    }

    /// Serialize to JSON bytes.
    pub fn to_json(&self) -> Result<Vec<u8>, BridgeError> {
        serde_json::to_vec(self).map_err(|e| BridgeError::Serialization(e.to_string()))
    }

    /// Deserialize from JSON bytes.
    pub fn from_json(data: &[u8]) -> Result<Self, BridgeError> {
        serde_json::from_slice(data).map_err(|e| BridgeError::Deserialization(e.to_string()))
    }
}

impl FieldSchema {
    /// Create a required field.
    pub fn required(name: impl Into<String>, field_type: FieldType) -> Self {
        Self {
            name: name.into(),
            field_type,
            optional: false,
            default: None,
            doc: None,
        }
    }

    /// Create an optional field.
    pub fn optional(name: impl Into<String>, field_type: FieldType) -> Self {
        Self {
            name: name.into(),
            field_type,
            optional: true,
            default: None,
            doc: None,
        }
    }

    /// Add a default value.
    pub fn with_default(mut self, default: serde_json::Value) -> Self {
        self.default = Some(default);
        self
    }

    /// Add documentation.
    pub fn with_doc(mut self, doc: impl Into<String>) -> Self {
        self.doc = Some(doc.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Data Transfer (Data Channel)
// ---------------------------------------------------------------------------

/// Header for bulk data transfer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataHeader {
    /// Transfer ID for correlation.
    pub transfer_id: u64,
    /// Type name of the data being transferred.
    pub type_name: String,
    /// Total size in bytes.
    pub total_bytes: u64,
    /// Number of elements (for arrays).
    pub element_count: u64,
    /// Compression method (if any).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compression: Option<CompressionMethod>,
    /// Checksum of uncompressed data (CRC32).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checksum: Option<u32>,
}

/// Supported compression methods for data transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CompressionMethod {
    /// No compression.
    None,
    /// LZ4 compression.
    Lz4,
    /// Zstd compression.
    Zstd,
}

impl DataHeader {
    /// Create a new data header.
    pub fn new(transfer_id: u64, type_name: impl Into<String>, total_bytes: u64) -> Self {
        Self {
            transfer_id,
            type_name: type_name.into(),
            total_bytes,
            element_count: 1,
            compression: None,
            checksum: None,
        }
    }

    /// Set element count.
    pub fn with_element_count(mut self, count: u64) -> Self {
        self.element_count = count;
        self
    }

    /// Set compression method.
    pub fn with_compression(mut self, method: CompressionMethod) -> Self {
        self.compression = Some(method);
        self
    }

    /// Set checksum.
    pub fn with_checksum(mut self, checksum: u32) -> Self {
        self.checksum = Some(checksum);
        self
    }
}

// ---------------------------------------------------------------------------
// Endpoint Definitions
// ---------------------------------------------------------------------------

/// Namespace for bridge lifecycle operations.
pub mod bridge_ns {
    use super::*;

    /// Handshake request to establish connection.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct HandshakeRequest {
        pub protocol_version: u32,
        pub client_name: String,
        pub capabilities: Vec<String>,
    }

    /// Handshake response.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct HandshakeResponse {
        pub protocol_version: u32,
        pub server_name: String,
        pub capabilities: Vec<String>,
        pub session_id: u64,
    }

    /// Health check response.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct HealthResponse {
        pub status: String,
        pub uptime_ms: u64,
        pub pending_requests: u32,
        pub memory_used_bytes: u64,
    }

    /// Version info response.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct VersionInfo {
        pub protocol_version: u32,
        pub engine_version: String,
        pub build_date: String,
        pub features: Vec<String>,
    }
}

/// Namespace for entity operations.
pub mod entity_ns {
    use super::*;

    /// Entity spawn request.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SpawnRequest {
        pub archetype: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub parent: Option<u64>,
        pub components: HashMap<String, serde_json::Value>,
    }

    /// Entity spawn response.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SpawnResponse {
        pub entity_id: u64,
        pub generation: u32,
    }

    /// Entity destroy request.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct DestroyRequest {
        pub entity_id: u64,
        #[serde(default)]
        pub recursive: bool,
    }

    /// Entity query request.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct QueryRequest {
        pub components: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub filter: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub limit: Option<u32>,
    }

    /// Entity query response.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct QueryResponse {
        pub entities: Vec<EntityData>,
        pub total_count: u64,
    }

    /// Entity data in query results.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct EntityData {
        pub id: u64,
        pub generation: u32,
        pub components: HashMap<String, serde_json::Value>,
    }
}

/// Namespace for component operations.
pub mod component_ns {
    use super::*;

    /// Get component request.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GetRequest {
        pub entity_id: u64,
        pub component: String,
    }

    /// Set component request.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SetRequest {
        pub entity_id: u64,
        pub component: String,
        pub value: serde_json::Value,
    }

    /// List components request.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ListRequest {
        pub entity_id: u64,
    }

    /// List components response.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ListResponse {
        pub components: Vec<String>,
    }
}

/// Namespace for frame timing operations.
pub mod frame_ns {
    use super::*;

    /// Frame statistics.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FrameStats {
        pub frame_number: u64,
        pub delta_time_ms: f64,
        pub fps: f64,
        pub frame_time_ms: f64,
        pub cpu_time_ms: f64,
        pub gpu_time_ms: f64,
        pub draw_calls: u32,
        pub triangles: u64,
        pub vertices: u64,
    }

    /// Timing breakdown.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TimingBreakdown {
        pub passes: HashMap<String, f64>,
        pub total_ms: f64,
    }
}

/// Namespace for profiler operations.
pub mod profiler_ns {
    use super::*;

    /// Profiler marker.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Marker {
        pub name: String,
        pub start_us: u64,
        pub end_us: u64,
        pub category: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub parent_id: Option<u64>,
    }

    /// GPU timing query result.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GpuTiming {
        pub pass_name: String,
        pub duration_us: u64,
        pub timestamp_start: u64,
        pub timestamp_end: u64,
    }

    /// CPU timing result.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CpuTiming {
        pub function_name: String,
        pub duration_us: u64,
        pub call_count: u32,
    }

    /// Memory statistics.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MemoryStats {
        pub gpu_buffer_bytes: u64,
        pub gpu_texture_bytes: u64,
        pub cpu_heap_bytes: u64,
        pub allocations: u32,
        pub deallocations: u32,
    }
}

/// Namespace for editor operations.
pub mod editor_ns {
    use super::*;

    /// Selection state.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Selection {
        pub entities: Vec<u64>,
        pub primary: Option<u64>,
    }

    /// Viewport configuration.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Viewport {
        pub width: u32,
        pub height: u32,
        pub camera_position: [f32; 3],
        pub camera_rotation: [f32; 4],
        pub fov_degrees: f32,
        pub near_clip: f32,
        pub far_clip: f32,
    }

    /// Panel state.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PanelState {
        pub panel_id: String,
        pub visible: bool,
        pub position: [f32; 2],
        pub size: [f32; 2],
    }

    /// Gizmo state.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GizmoState {
        pub mode: String,
        pub space: String,
        pub snap_enabled: bool,
        pub snap_value: f32,
    }
}

/// Namespace for material operations.
pub mod material_ns {
    use super::*;

    /// Material parameter.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Parameter {
        pub name: String,
        pub value: serde_json::Value,
        pub param_type: String,
    }

    /// Material definition.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MaterialDef {
        pub name: String,
        pub shader: String,
        pub parameters: Vec<Parameter>,
        pub render_states: HashMap<String, serde_json::Value>,
    }

    /// Compilation result.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CompilationResult {
        pub success: bool,
        pub shader_hash: Option<String>,
        pub errors: Vec<String>,
        pub warnings: Vec<String>,
    }
}

/// Namespace for asset operations.
pub mod asset_ns {
    use super::*;

    /// Asset load request.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LoadRequest {
        pub path: String,
        pub asset_type: String,
        #[serde(default)]
        pub async_load: bool,
        #[serde(default)]
        pub priority: i32,
    }

    /// Asset load response.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LoadResponse {
        pub handle: u64,
        pub status: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub error: Option<String>,
    }

    /// Hot-reload trigger.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct HotReloadTrigger {
        pub paths: Vec<String>,
        pub force: bool,
    }

    /// Asset status query.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct AssetStatus {
        pub handle: u64,
        pub path: String,
        pub loaded: bool,
        pub size_bytes: u64,
        pub last_modified: u64,
    }
}

// ---------------------------------------------------------------------------
// Request Tracker
// ---------------------------------------------------------------------------

/// Tracks pending requests for timeout and correlation.
#[derive(Debug)]
pub struct RequestTracker {
    /// Pending requests: id -> (namespace, method, start_time, timeout).
    pending: parking_lot::RwLock<HashMap<u64, PendingRequest>>,
}

#[derive(Debug)]
struct PendingRequest {
    namespace: String,
    method: String,
    start_time: Instant,
    timeout: Option<Duration>,
}

impl RequestTracker {
    /// Create a new request tracker.
    pub fn new() -> Self {
        Self {
            pending: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    /// Register a pending request.
    pub fn register(&self, request: &CommandRequest) {
        let mut pending = self.pending.write();
        pending.insert(
            request.id,
            PendingRequest {
                namespace: request.namespace.clone(),
                method: request.method.clone(),
                start_time: Instant::now(),
                timeout: request.timeout_ms.map(Duration::from_millis),
            },
        );
    }

    /// Complete a request and return elapsed time.
    pub fn complete(&self, id: u64) -> Option<Duration> {
        let mut pending = self.pending.write();
        pending.remove(&id).map(|p| p.start_time.elapsed())
    }

    /// Check for timed-out requests.
    pub fn check_timeouts(&self) -> Vec<u64> {
        let pending = self.pending.read();
        let now = Instant::now();
        pending
            .iter()
            .filter_map(|(id, req)| {
                if let Some(timeout) = req.timeout {
                    if now.duration_since(req.start_time) > timeout {
                        return Some(*id);
                    }
                }
                None
            })
            .collect()
    }

    /// Get the number of pending requests.
    pub fn pending_count(&self) -> usize {
        self.pending.read().len()
    }
}

impl Default for RequestTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Type Registry
// ---------------------------------------------------------------------------

/// Registry for type schemas.
#[derive(Debug, Default)]
pub struct TypeRegistry {
    /// Registered types: name -> schema.
    types: parking_lot::RwLock<HashMap<String, TypeSchema>>,
}

impl TypeRegistry {
    /// Create a new type registry.
    pub fn new() -> Self {
        Self {
            types: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    /// Register a type schema.
    pub fn register(&self, schema: TypeSchema) -> Result<(), BridgeError> {
        let mut types = self.types.write();
        if let Some(existing) = types.get(&schema.name) {
            if existing.version >= schema.version {
                return Err(BridgeError::Internal(format!(
                    "Type {} version {} already registered (current: {})",
                    schema.name, schema.version, existing.version
                )));
            }
        }
        types.insert(schema.name.clone(), schema);
        Ok(())
    }

    /// Get a type schema by name.
    pub fn get(&self, name: &str) -> Option<TypeSchema> {
        self.types.read().get(name).cloned()
    }

    /// List all registered type names.
    pub fn list(&self) -> Vec<String> {
        self.types.read().keys().cloned().collect()
    }

    /// Check if a type is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.types.read().contains_key(name)
    }

    /// Get the number of registered types.
    pub fn len(&self) -> usize {
        self.types.read().len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.types.read().is_empty()
    }
}

// ---------------------------------------------------------------------------
// Protocol Dispatcher
// ---------------------------------------------------------------------------

/// Handler function type for command dispatch.
pub type CommandHandler =
    Box<dyn Fn(serde_json::Value) -> Result<serde_json::Value, BridgeError> + Send + Sync>;

/// Dispatcher for routing commands to handlers.
pub struct ProtocolDispatcher {
    /// Handlers: (namespace, method) -> handler.
    handlers: parking_lot::RwLock<HashMap<(String, String), CommandHandler>>,
    /// Request tracker.
    tracker: RequestTracker,
    /// Type registry.
    types: TypeRegistry,
}

impl ProtocolDispatcher {
    /// Create a new protocol dispatcher.
    pub fn new() -> Self {
        Self {
            handlers: parking_lot::RwLock::new(HashMap::new()),
            tracker: RequestTracker::new(),
            types: TypeRegistry::new(),
        }
    }

    /// Register a command handler.
    pub fn register_handler<F>(
        &self,
        namespace: impl Into<String>,
        method: impl Into<String>,
        handler: F,
    ) where
        F: Fn(serde_json::Value) -> Result<serde_json::Value, BridgeError> + Send + Sync + 'static,
    {
        let mut handlers = self.handlers.write();
        handlers.insert((namespace.into(), method.into()), Box::new(handler));
    }

    /// Dispatch a command request.
    pub fn dispatch(&self, request: CommandRequest) -> CommandResponse {
        self.tracker.register(&request);
        let start = Instant::now();

        let result = {
            let handlers = self.handlers.read();
            let key = (request.namespace.clone(), request.method.clone());

            if let Some(handler) = handlers.get(&key) {
                handler(request.params)
            } else {
                // Check if namespace exists
                let namespace_exists = handlers
                    .keys()
                    .any(|(ns, _)| ns == &request.namespace);

                if namespace_exists {
                    Err(BridgeError::UnknownMethod(request.method.clone()))
                } else {
                    Err(BridgeError::UnknownNamespace(request.namespace.clone()))
                }
            }
        };

        self.tracker.complete(request.id);

        let response = match result {
            Ok(value) => CommandResponse::ok(request.id, value),
            Err(error) => CommandResponse::error(request.id, error),
        };

        response.with_execution_time(start.elapsed())
    }

    /// Get the type registry.
    pub fn types(&self) -> &TypeRegistry {
        &self.types
    }

    /// Get the request tracker.
    pub fn tracker(&self) -> &RequestTracker {
        &self.tracker
    }

    /// List all registered handlers.
    pub fn list_handlers(&self) -> Vec<(String, String)> {
        self.handlers.read().keys().cloned().collect()
    }
}

impl Default for ProtocolDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for ProtocolDispatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProtocolDispatcher")
            .field("handler_count", &self.handlers.read().len())
            .field("pending_requests", &self.tracker.pending_count())
            .field("registered_types", &self.types.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Helper Module: serde_bytes for Vec<u8>
// ---------------------------------------------------------------------------

mod serde_bytes {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(bytes: &Vec<u8>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Serialize as base64 string for JSON compatibility
        let encoded = base64_encode(bytes);
        encoded.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        base64_decode(&s).map_err(serde::de::Error::custom)
    }

    fn base64_encode(bytes: &[u8]) -> String {
        const ALPHABET: &[u8] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut result = String::with_capacity((bytes.len() + 2) / 3 * 4);

        for chunk in bytes.chunks(3) {
            let mut n = 0u32;
            for (i, &b) in chunk.iter().enumerate() {
                n |= (b as u32) << (16 - i * 8);
            }

            result.push(ALPHABET[(n >> 18 & 0x3F) as usize] as char);
            result.push(ALPHABET[(n >> 12 & 0x3F) as usize] as char);

            if chunk.len() > 1 {
                result.push(ALPHABET[(n >> 6 & 0x3F) as usize] as char);
            } else {
                result.push('=');
            }

            if chunk.len() > 2 {
                result.push(ALPHABET[(n & 0x3F) as usize] as char);
            } else {
                result.push('=');
            }
        }

        result
    }

    fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
        fn decode_char(c: char) -> Result<u8, String> {
            match c {
                'A'..='Z' => Ok(c as u8 - b'A'),
                'a'..='z' => Ok(c as u8 - b'a' + 26),
                '0'..='9' => Ok(c as u8 - b'0' + 52),
                '+' => Ok(62),
                '/' => Ok(63),
                '=' => Ok(0),
                _ => Err(format!("Invalid base64 character: {}", c)),
            }
        }

        let s = s.trim();
        if s.is_empty() {
            return Ok(Vec::new());
        }

        let mut result = Vec::with_capacity(s.len() * 3 / 4);
        let chars: Vec<char> = s.chars().collect();

        for chunk in chars.chunks(4) {
            if chunk.len() != 4 {
                return Err("Invalid base64 length".to_string());
            }

            let a = decode_char(chunk[0])?;
            let b = decode_char(chunk[1])?;
            let c = decode_char(chunk[2])?;
            let d = decode_char(chunk[3])?;

            let n = ((a as u32) << 18) | ((b as u32) << 12) | ((c as u32) << 6) | (d as u32);

            result.push((n >> 16 & 0xFF) as u8);
            if chunk[2] != '=' {
                result.push((n >> 8 & 0xFF) as u8);
            }
            if chunk[3] != '=' {
                result.push((n & 0xFF) as u8);
            }
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -- ChannelKind tests --------------------------------------------------

    #[test]
    fn test_channel_kind_priority() {
        assert_eq!(ChannelKind::Command.priority(), 0);
        assert_eq!(ChannelKind::Type.priority(), 1);
        assert_eq!(ChannelKind::Data.priority(), 2);
    }

    #[test]
    fn test_channel_kind_serialization() {
        let json = serde_json::to_string(&ChannelKind::Command).unwrap();
        assert_eq!(json, "\"command\"");

        let deserialized: ChannelKind = serde_json::from_str("\"type\"").unwrap();
        assert_eq!(deserialized, ChannelKind::Type);
    }

    #[test]
    fn test_channel_kind_all_variants() {
        assert_eq!(ChannelKind::Type, ChannelKind::Type);
        assert_eq!(ChannelKind::Data, ChannelKind::Data);
        assert_eq!(ChannelKind::Command, ChannelKind::Command);
        assert_ne!(ChannelKind::Type, ChannelKind::Data);
    }

    // -- BridgeMessage tests ------------------------------------------------

    #[test]
    fn test_bridge_message_new() {
        let msg = BridgeMessage::new(ChannelKind::Command, vec![1, 2, 3]);
        assert!(msg.id > 0);
        assert_eq!(msg.channel, ChannelKind::Command);
        assert_eq!(msg.payload, vec![1, 2, 3]);
        assert!(msg.timestamp_ms > 0);
    }

    #[test]
    fn test_bridge_message_with_id() {
        let msg = BridgeMessage::with_id(42, ChannelKind::Data, vec![4, 5, 6]);
        assert_eq!(msg.id, 42);
        assert_eq!(msg.channel, ChannelKind::Data);
    }

    #[test]
    fn test_bridge_message_unique_ids() {
        let msg1 = BridgeMessage::new(ChannelKind::Command, vec![]);
        let msg2 = BridgeMessage::new(ChannelKind::Command, vec![]);
        assert_ne!(msg1.id, msg2.id);
    }

    #[test]
    fn test_bridge_message_serialization() {
        let msg = BridgeMessage::with_id(100, ChannelKind::Type, vec![7, 8, 9]);
        let json = msg.to_json().unwrap();
        let deserialized = BridgeMessage::from_json(&json).unwrap();
        assert_eq!(deserialized.id, 100);
        assert_eq!(deserialized.channel, ChannelKind::Type);
        assert_eq!(deserialized.payload, vec![7, 8, 9]);
    }

    #[test]
    fn test_bridge_message_empty_payload() {
        let msg = BridgeMessage::new(ChannelKind::Data, vec![]);
        let json = msg.to_json().unwrap();
        let deserialized = BridgeMessage::from_json(&json).unwrap();
        assert!(deserialized.payload.is_empty());
    }

    #[test]
    fn test_bridge_message_large_payload() {
        let payload: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        let msg = BridgeMessage::new(ChannelKind::Data, payload.clone());
        let json = msg.to_json().unwrap();
        let deserialized = BridgeMessage::from_json(&json).unwrap();
        assert_eq!(deserialized.payload, payload);
    }

    // -- BridgeError tests --------------------------------------------------

    #[test]
    fn test_bridge_error_display() {
        assert_eq!(BridgeError::NotConnected.to_string(), "Bridge not connected");
        assert_eq!(
            BridgeError::Timeout("request".to_string()).to_string(),
            "Timeout: request"
        );
        assert_eq!(
            BridgeError::UnknownNamespace("foo".to_string()).to_string(),
            "Unknown namespace: foo"
        );
    }

    #[test]
    fn test_bridge_error_serialization() {
        let error = BridgeError::VersionMismatch { expected: 1, got: 2 };
        let json = serde_json::to_string(&error).unwrap();
        let deserialized: BridgeError = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, error);
    }

    #[test]
    fn test_bridge_error_all_variants() {
        let errors = vec![
            BridgeError::NotConnected,
            BridgeError::Timeout("test".into()),
            BridgeError::Serialization("test".into()),
            BridgeError::Deserialization("test".into()),
            BridgeError::UnknownNamespace("ns".into()),
            BridgeError::UnknownMethod("method".into()),
            BridgeError::InvalidParams("params".into()),
            BridgeError::Internal("internal".into()),
            BridgeError::VersionMismatch { expected: 1, got: 2 },
            BridgeError::TypeNotRegistered("type".into()),
            BridgeError::EntityNotFound(123),
            BridgeError::ComponentNotFound("comp".into()),
            BridgeError::AssetNotFound("asset".into()),
            BridgeError::PermissionDenied("perm".into()),
            BridgeError::ResourceExhausted("res".into()),
            BridgeError::Cancelled,
        ];

        for error in errors {
            let json = serde_json::to_string(&error).unwrap();
            let deserialized: BridgeError = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized, error);
        }
    }

    // -- CommandRequest tests -----------------------------------------------

    #[test]
    fn test_command_request_new() {
        let req = CommandRequest::new("entity", "spawn", json!({"name": "test"}));
        assert!(req.id > 0);
        assert_eq!(req.namespace, "entity");
        assert_eq!(req.method, "spawn");
        assert_eq!(req.params["name"], "test");
        assert!(req.timeout_ms.is_none());
    }

    #[test]
    fn test_command_request_with_timeout() {
        let req = CommandRequest::new("bridge", "health", json!({}))
            .with_timeout(Duration::from_secs(5));
        assert_eq!(req.timeout_ms, Some(5000));
    }

    #[test]
    fn test_command_request_serialization() {
        let req = CommandRequest::new("component", "get", json!({"entity_id": 42}));
        let json = req.to_json().unwrap();
        let deserialized = CommandRequest::from_json(&json).unwrap();
        assert_eq!(deserialized.namespace, "component");
        assert_eq!(deserialized.method, "get");
        assert_eq!(deserialized.params["entity_id"], 42);
    }

    #[test]
    fn test_command_request_unique_ids() {
        let req1 = CommandRequest::new("a", "b", json!({}));
        let req2 = CommandRequest::new("a", "b", json!({}));
        assert_ne!(req1.id, req2.id);
    }

    // -- CommandResponse tests ----------------------------------------------

    #[test]
    fn test_command_response_ok() {
        let resp = CommandResponse::ok(1, json!({"result": "success"}));
        assert!(resp.is_ok());
        assert_eq!(resp.value(), Some(&json!({"result": "success"})));
        assert!(resp.get_error().is_none());
    }

    #[test]
    fn test_command_response_error() {
        let resp = CommandResponse::error(2, BridgeError::NotConnected);
        assert!(!resp.is_ok());
        assert!(resp.value().is_none());
        assert_eq!(resp.get_error(), Some(&BridgeError::NotConnected));
    }

    #[test]
    fn test_command_response_with_execution_time() {
        let resp =
            CommandResponse::ok(3, json!({})).with_execution_time(Duration::from_micros(500));
        assert_eq!(resp.execution_us, Some(500));
    }

    #[test]
    fn test_command_response_serialization() {
        let resp = CommandResponse::ok(4, json!({"value": 123}));
        let json = resp.to_json().unwrap();
        let deserialized = CommandResponse::from_json(&json).unwrap();
        assert_eq!(deserialized.id, 4);
        assert!(deserialized.is_ok());
    }

    // -- TypeSchema tests ---------------------------------------------------

    #[test]
    fn test_type_schema_new() {
        let schema = TypeSchema::new("Transform", 1);
        assert_eq!(schema.name, "Transform");
        assert_eq!(schema.version, 1);
        assert!(schema.fields.is_empty());
    }

    #[test]
    fn test_type_schema_with_fields() {
        let schema = TypeSchema::new("Position", 1)
            .with_field(FieldSchema::required("x", FieldType::F32))
            .with_field(FieldSchema::required("y", FieldType::F32))
            .with_field(FieldSchema::required("z", FieldType::F32))
            .with_doc("3D position component");

        assert_eq!(schema.fields.len(), 3);
        assert_eq!(schema.doc, Some("3D position component".to_string()));
    }

    #[test]
    fn test_type_schema_serialization() {
        let schema = TypeSchema::new("Test", 2)
            .with_field(FieldSchema::optional("value", FieldType::I32).with_default(json!(0)));
        let json = schema.to_json().unwrap();
        let deserialized = TypeSchema::from_json(&json).unwrap();
        assert_eq!(deserialized.name, "Test");
        assert_eq!(deserialized.version, 2);
        assert_eq!(deserialized.fields.len(), 1);
        assert!(deserialized.fields[0].optional);
    }

    // -- FieldSchema tests --------------------------------------------------

    #[test]
    fn test_field_schema_required() {
        let field = FieldSchema::required("name", FieldType::String);
        assert_eq!(field.name, "name");
        assert!(!field.optional);
        assert!(field.default.is_none());
    }

    #[test]
    fn test_field_schema_optional() {
        let field = FieldSchema::optional("count", FieldType::U32);
        assert!(field.optional);
    }

    #[test]
    fn test_field_schema_with_default() {
        let field = FieldSchema::optional("limit", FieldType::I32).with_default(json!(100));
        assert_eq!(field.default, Some(json!(100)));
    }

    #[test]
    fn test_field_schema_with_doc() {
        let field = FieldSchema::required("id", FieldType::U64).with_doc("Unique identifier");
        assert_eq!(field.doc, Some("Unique identifier".to_string()));
    }

    // -- FieldType tests ----------------------------------------------------

    #[test]
    fn test_field_type_primitives() {
        let types = vec![
            FieldType::Bool,
            FieldType::I32,
            FieldType::I64,
            FieldType::U32,
            FieldType::U64,
            FieldType::F32,
            FieldType::F64,
            FieldType::String,
            FieldType::Bytes,
        ];

        for t in types {
            let json = serde_json::to_string(&t).unwrap();
            let deserialized: FieldType = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized, t);
        }
    }

    #[test]
    fn test_field_type_array() {
        let arr_type = FieldType::Array {
            element: Box::new(FieldType::F32),
        };
        let json = serde_json::to_string(&arr_type).unwrap();
        let deserialized: FieldType = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, arr_type);
    }

    #[test]
    fn test_field_type_map() {
        let map_type = FieldType::Map {
            value: Box::new(FieldType::String),
        };
        let json = serde_json::to_string(&map_type).unwrap();
        let deserialized: FieldType = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, map_type);
    }

    #[test]
    fn test_field_type_ref() {
        let ref_type = FieldType::Ref {
            type_name: "Transform".to_string(),
        };
        let json = serde_json::to_string(&ref_type).unwrap();
        let deserialized: FieldType = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, ref_type);
    }

    #[test]
    fn test_field_type_math_types() {
        let types = vec![
            FieldType::Vec2,
            FieldType::Vec3,
            FieldType::Vec4,
            FieldType::Mat4,
            FieldType::Quat,
            FieldType::Color,
        ];

        for t in types {
            let json = serde_json::to_string(&t).unwrap();
            let deserialized: FieldType = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized, t);
        }
    }

    #[test]
    fn test_field_type_special() {
        assert_eq!(FieldType::EntityId, FieldType::EntityId);
        assert_eq!(FieldType::AssetHandle, FieldType::AssetHandle);
    }

    // -- DataHeader tests ---------------------------------------------------

    #[test]
    fn test_data_header_new() {
        let header = DataHeader::new(1, "Mesh", 1024);
        assert_eq!(header.transfer_id, 1);
        assert_eq!(header.type_name, "Mesh");
        assert_eq!(header.total_bytes, 1024);
        assert_eq!(header.element_count, 1);
    }

    #[test]
    fn test_data_header_with_options() {
        let header = DataHeader::new(2, "Texture", 4096)
            .with_element_count(100)
            .with_compression(CompressionMethod::Lz4)
            .with_checksum(0xDEADBEEF);

        assert_eq!(header.element_count, 100);
        assert_eq!(header.compression, Some(CompressionMethod::Lz4));
        assert_eq!(header.checksum, Some(0xDEADBEEF));
    }

    #[test]
    fn test_compression_method_serialization() {
        let methods = vec![
            CompressionMethod::None,
            CompressionMethod::Lz4,
            CompressionMethod::Zstd,
        ];

        for method in methods {
            let json = serde_json::to_string(&method).unwrap();
            let deserialized: CompressionMethod = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized, method);
        }
    }

    // -- RequestTracker tests -----------------------------------------------

    #[test]
    fn test_request_tracker_register_complete() {
        let tracker = RequestTracker::new();
        let req = CommandRequest::new("test", "method", json!({}));
        let id = req.id;

        tracker.register(&req);
        assert_eq!(tracker.pending_count(), 1);

        let elapsed = tracker.complete(id);
        assert!(elapsed.is_some());
        assert_eq!(tracker.pending_count(), 0);
    }

    #[test]
    fn test_request_tracker_timeout_check() {
        let tracker = RequestTracker::new();
        let req =
            CommandRequest::new("test", "method", json!({})).with_timeout(Duration::from_millis(1));
        let id = req.id;

        tracker.register(&req);

        // Wait for timeout
        std::thread::sleep(Duration::from_millis(10));

        let timed_out = tracker.check_timeouts();
        assert!(timed_out.contains(&id));
    }

    #[test]
    fn test_request_tracker_no_timeout() {
        let tracker = RequestTracker::new();
        let req = CommandRequest::new("test", "method", json!({})); // No timeout

        tracker.register(&req);

        let timed_out = tracker.check_timeouts();
        assert!(timed_out.is_empty());
    }

    // -- TypeRegistry tests -------------------------------------------------

    #[test]
    fn test_type_registry_register_get() {
        let registry = TypeRegistry::new();
        let schema = TypeSchema::new("Entity", 1);

        registry.register(schema.clone()).unwrap();
        assert!(registry.contains("Entity"));
        assert!(!registry.contains("Unknown"));

        let retrieved = registry.get("Entity").unwrap();
        assert_eq!(retrieved.name, "Entity");
        assert_eq!(retrieved.version, 1);
    }

    #[test]
    fn test_type_registry_list() {
        let registry = TypeRegistry::new();
        registry.register(TypeSchema::new("A", 1)).unwrap();
        registry.register(TypeSchema::new("B", 1)).unwrap();

        let types = registry.list();
        assert_eq!(types.len(), 2);
        assert!(types.contains(&"A".to_string()));
        assert!(types.contains(&"B".to_string()));
    }

    #[test]
    fn test_type_registry_version_upgrade() {
        let registry = TypeRegistry::new();
        registry.register(TypeSchema::new("X", 1)).unwrap();
        registry.register(TypeSchema::new("X", 2)).unwrap(); // Upgrade OK

        let schema = registry.get("X").unwrap();
        assert_eq!(schema.version, 2);
    }

    #[test]
    fn test_type_registry_version_downgrade_fails() {
        let registry = TypeRegistry::new();
        registry.register(TypeSchema::new("Y", 2)).unwrap();

        let result = registry.register(TypeSchema::new("Y", 1)); // Downgrade
        assert!(result.is_err());
    }

    #[test]
    fn test_type_registry_len_is_empty() {
        let registry = TypeRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);

        registry.register(TypeSchema::new("Z", 1)).unwrap();
        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 1);
    }

    // -- ProtocolDispatcher tests -------------------------------------------

    #[test]
    fn test_dispatcher_register_dispatch() {
        let dispatcher = ProtocolDispatcher::new();

        dispatcher.register_handler("test", "echo", |params| Ok(params));

        let req = CommandRequest::new("test", "echo", json!({"message": "hello"}));
        let resp = dispatcher.dispatch(req);

        assert!(resp.is_ok());
        assert_eq!(resp.value().unwrap()["message"], "hello");
    }

    #[test]
    fn test_dispatcher_unknown_namespace() {
        let dispatcher = ProtocolDispatcher::new();

        let req = CommandRequest::new("unknown", "method", json!({}));
        let resp = dispatcher.dispatch(req);

        assert!(!resp.is_ok());
        assert!(matches!(
            resp.get_error(),
            Some(BridgeError::UnknownNamespace(_))
        ));
    }

    #[test]
    fn test_dispatcher_unknown_method() {
        let dispatcher = ProtocolDispatcher::new();
        dispatcher.register_handler("ns", "exists", |_| Ok(json!({})));

        let req = CommandRequest::new("ns", "missing", json!({}));
        let resp = dispatcher.dispatch(req);

        assert!(!resp.is_ok());
        assert!(matches!(resp.get_error(), Some(BridgeError::UnknownMethod(_))));
    }

    #[test]
    fn test_dispatcher_handler_error() {
        let dispatcher = ProtocolDispatcher::new();
        dispatcher.register_handler("fail", "always", |_| {
            Err(BridgeError::Internal("intentional".to_string()))
        });

        let req = CommandRequest::new("fail", "always", json!({}));
        let resp = dispatcher.dispatch(req);

        assert!(!resp.is_ok());
        assert!(matches!(resp.get_error(), Some(BridgeError::Internal(_))));
    }

    #[test]
    fn test_dispatcher_execution_time() {
        let dispatcher = ProtocolDispatcher::new();
        dispatcher.register_handler("time", "test", |_| {
            std::thread::sleep(Duration::from_millis(10));
            Ok(json!({}))
        });

        let req = CommandRequest::new("time", "test", json!({}));
        let resp = dispatcher.dispatch(req);

        assert!(resp.execution_us.unwrap() >= 10_000); // At least 10ms
    }

    #[test]
    fn test_dispatcher_list_handlers() {
        let dispatcher = ProtocolDispatcher::new();
        dispatcher.register_handler("a", "x", |_| Ok(json!({})));
        dispatcher.register_handler("a", "y", |_| Ok(json!({})));
        dispatcher.register_handler("b", "z", |_| Ok(json!({})));

        let handlers = dispatcher.list_handlers();
        assert_eq!(handlers.len(), 3);
    }

    // -- Namespace endpoint structure tests ---------------------------------

    #[test]
    fn test_bridge_ns_handshake() {
        let req = bridge_ns::HandshakeRequest {
            protocol_version: PROTOCOL_VERSION,
            client_name: "test".to_string(),
            capabilities: vec!["debug".to_string()],
        };
        let json = serde_json::to_string(&req).unwrap();
        let deserialized: bridge_ns::HandshakeRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.protocol_version, PROTOCOL_VERSION);
    }

    #[test]
    fn test_entity_ns_spawn() {
        let req = entity_ns::SpawnRequest {
            archetype: "Player".to_string(),
            name: Some("Hero".to_string()),
            parent: None,
            components: HashMap::new(),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("Player"));
    }

    #[test]
    fn test_component_ns_set() {
        let req = component_ns::SetRequest {
            entity_id: 123,
            component: "Transform".to_string(),
            value: json!({"position": [0, 0, 0]}),
        };
        let json = serde_json::to_string(&req).unwrap();
        let deserialized: component_ns::SetRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.entity_id, 123);
    }

    #[test]
    fn test_frame_ns_stats() {
        let stats = frame_ns::FrameStats {
            frame_number: 1000,
            delta_time_ms: 16.67,
            fps: 60.0,
            frame_time_ms: 16.5,
            cpu_time_ms: 8.0,
            gpu_time_ms: 8.5,
            draw_calls: 150,
            triangles: 500_000,
            vertices: 1_000_000,
        };
        let json = serde_json::to_string(&stats).unwrap();
        let deserialized: frame_ns::FrameStats = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.frame_number, 1000);
    }

    #[test]
    fn test_profiler_ns_marker() {
        let marker = profiler_ns::Marker {
            name: "RenderPass".to_string(),
            start_us: 0,
            end_us: 5000,
            category: "GPU".to_string(),
            parent_id: None,
        };
        let json = serde_json::to_string(&marker).unwrap();
        assert!(json.contains("RenderPass"));
    }

    #[test]
    fn test_editor_ns_selection() {
        let selection = editor_ns::Selection {
            entities: vec![1, 2, 3],
            primary: Some(1),
        };
        let json = serde_json::to_string(&selection).unwrap();
        let deserialized: editor_ns::Selection = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.entities.len(), 3);
    }

    #[test]
    fn test_material_ns_param() {
        let param = material_ns::Parameter {
            name: "albedo".to_string(),
            value: json!([1.0, 0.5, 0.0, 1.0]),
            param_type: "color".to_string(),
        };
        let json = serde_json::to_string(&param).unwrap();
        assert!(json.contains("albedo"));
    }

    #[test]
    fn test_asset_ns_load() {
        let req = asset_ns::LoadRequest {
            path: "assets/mesh.gltf".to_string(),
            asset_type: "mesh".to_string(),
            async_load: true,
            priority: 10,
        };
        let json = serde_json::to_string(&req).unwrap();
        let deserialized: asset_ns::LoadRequest = serde_json::from_str(&json).unwrap();
        assert!(deserialized.async_load);
    }

    // -- Integration tests --------------------------------------------------

    #[test]
    fn test_full_command_roundtrip() {
        let dispatcher = ProtocolDispatcher::new();

        // Register schema
        let schema = TypeSchema::new("Vec3", 1)
            .with_field(FieldSchema::required("x", FieldType::F32))
            .with_field(FieldSchema::required("y", FieldType::F32))
            .with_field(FieldSchema::required("z", FieldType::F32));
        dispatcher.types().register(schema).unwrap();

        // Register handler
        dispatcher.register_handler("entity", "spawn", |params| {
            let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("unnamed");
            Ok(json!({
                "entity_id": 42,
                "name": name
            }))
        });

        // Create and dispatch request
        let req = CommandRequest::new("entity", "spawn", json!({ "name": "TestEntity" }));
        let resp = dispatcher.dispatch(req);

        assert!(resp.is_ok());
        assert_eq!(resp.value().unwrap()["entity_id"], 42);
        assert_eq!(resp.value().unwrap()["name"], "TestEntity");
    }

    #[test]
    fn test_message_channel_routing() {
        let type_msg = BridgeMessage::new(ChannelKind::Type, vec![]);
        let data_msg = BridgeMessage::new(ChannelKind::Data, vec![]);
        let cmd_msg = BridgeMessage::new(ChannelKind::Command, vec![]);

        // Verify channel assignment
        assert_eq!(type_msg.channel, ChannelKind::Type);
        assert_eq!(data_msg.channel, ChannelKind::Data);
        assert_eq!(cmd_msg.channel, ChannelKind::Command);

        // Verify priority ordering
        assert!(cmd_msg.channel.priority() < type_msg.channel.priority());
        assert!(type_msg.channel.priority() < data_msg.channel.priority());
    }

    #[test]
    fn test_concurrent_request_tracking() {
        let tracker = RequestTracker::new();
        let mut ids = Vec::new();

        // Register multiple requests
        for i in 0..10 {
            let req = CommandRequest::new("test", format!("method_{}", i), json!({}));
            ids.push(req.id);
            tracker.register(&req);
        }

        assert_eq!(tracker.pending_count(), 10);

        // Complete in reverse order
        for id in ids.iter().rev() {
            let elapsed = tracker.complete(*id);
            assert!(elapsed.is_some());
        }

        assert_eq!(tracker.pending_count(), 0);
    }

    #[test]
    fn test_protocol_version_constant() {
        assert_eq!(PROTOCOL_VERSION, 1);
        assert_eq!(PROTOCOL_MAGIC, 0x5452_494E);
    }

    #[test]
    fn test_dispatcher_debug() {
        let dispatcher = ProtocolDispatcher::new();
        dispatcher.register_handler("a", "b", |_| Ok(json!({})));

        let debug_str = format!("{:?}", dispatcher);
        assert!(debug_str.contains("ProtocolDispatcher"));
        assert!(debug_str.contains("handler_count"));
    }
}
