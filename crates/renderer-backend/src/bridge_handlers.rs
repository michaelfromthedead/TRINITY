//! Bridge endpoint handlers for Python-Rust communication.
//!
//! This module implements the handler functions for each namespace defined in
//! [`bridge_protocol`](crate::bridge_protocol). Handlers wire the protocol to
//! engine subsystems like [`ComponentStore`], [`Editor`], and [`GPUProfiler`].
//!
//! # Architecture
//!
//! ```text
//! CommandRequest -> ProtocolDispatcher -> Handler Function -> Engine Subsystem
//!                                                          -> CommandResponse
//! ```
//!
//! # Namespaces
//!
//! | Namespace    | Handlers                                        | Status       |
//! |--------------|-------------------------------------------------|--------------|
//! | `bridge.*`   | handshake, health, version                      | Implemented  |
//! | `entity.*`   | spawn, destroy, query                           | Implemented  |
//! | `component.*`| get, set, list                                  | Implemented  |
//! | `frame.*`    | stats, history                                  | Implemented  |
//! | `profiler.*` | start_marker, end_marker, gpu_timing, memory    | Implemented  |
//! | `editor.*`   | select, get_selection, set_viewport             | Implemented  |
//! | `material.*` | (stubs)                                         | Stub         |
//! | `asset.*`    | (stubs)                                         | Stub         |

use crate::bridge_protocol::{
    bridge_ns, component_ns, editor_ns, entity_ns, frame_ns, profiler_ns, BridgeError,
    CommandHandler, ProtocolDispatcher, PROTOCOL_VERSION,
};
use crate::component_store::{global_component_store, ComponentStore};
use crate::editor::{Editor, EditorState};
use crate::gpu_profiler::{FrameProfile, PassTiming};
use crate::type_registry::TypeRegistry;

use parking_lot::RwLock;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Engine Version
// ---------------------------------------------------------------------------

/// Current engine version.
pub const ENGINE_VERSION: &str = "0.1.0-alpha";

/// Build date (set at compile time or defaulted).
pub const BUILD_DATE: &str = "2026-05-26";

// ---------------------------------------------------------------------------
// BridgeContext
// ---------------------------------------------------------------------------

/// Shared context for bridge handlers, providing access to engine subsystems.
///
/// This struct holds references to the component store, editor state, and
/// profiling data. It is wrapped in `Arc` for safe sharing across handlers.
pub struct BridgeContext {
    /// Component store for ECS operations.
    pub component_store: Arc<RwLock<ComponentStore>>,
    /// Editor state for selection and viewport.
    pub editor: RwLock<Editor>,
    /// Frame timing ring buffer.
    pub frame_history: RwLock<FrameHistory>,
    /// Memory statistics.
    pub memory_stats: RwLock<MemoryStats>,
    /// Active profiler markers.
    pub profiler_markers: RwLock<ProfilerMarkers>,
    /// Session state.
    pub session: RwLock<SessionState>,
    /// Connection start time.
    pub start_time: Instant,
}

impl BridgeContext {
    /// Create a new bridge context wrapping the given component store.
    pub fn new(component_store: Arc<RwLock<ComponentStore>>) -> Self {
        let editor = Editor::new(component_store.clone());
        Self {
            component_store: component_store.clone(),
            editor: RwLock::new(editor),
            frame_history: RwLock::new(FrameHistory::new(120)),
            memory_stats: RwLock::new(MemoryStats::default()),
            profiler_markers: RwLock::new(ProfilerMarkers::new()),
            session: RwLock::new(SessionState::default()),
            start_time: Instant::now(),
        }
    }

    /// Get uptime in milliseconds.
    pub fn uptime_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }
}

// ---------------------------------------------------------------------------
// Frame History
// ---------------------------------------------------------------------------

/// Ring buffer of recent frame timing data.
#[derive(Debug)]
pub struct FrameHistory {
    /// Ring buffer of frame stats.
    frames: VecDeque<frame_ns::FrameStats>,
    /// Maximum number of frames to store.
    capacity: usize,
    /// Current frame number.
    frame_counter: AtomicU64,
}

impl FrameHistory {
    /// Create a new frame history with given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            frames: VecDeque::with_capacity(capacity),
            capacity,
            frame_counter: AtomicU64::new(0),
        }
    }

    /// Record a new frame's timing data.
    pub fn record(&mut self, stats: frame_ns::FrameStats) {
        if self.frames.len() >= self.capacity {
            self.frames.pop_front();
        }
        self.frames.push_back(stats);
        self.frame_counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Get the latest frame stats.
    pub fn latest(&self) -> Option<&frame_ns::FrameStats> {
        self.frames.back()
    }

    /// Get the last N frames.
    pub fn last_n(&self, n: usize) -> Vec<&frame_ns::FrameStats> {
        self.frames.iter().rev().take(n).collect()
    }

    /// Get all frames in the history.
    pub fn all(&self) -> &VecDeque<frame_ns::FrameStats> {
        &self.frames
    }

    /// Current frame number.
    pub fn frame_number(&self) -> u64 {
        self.frame_counter.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Memory Stats
// ---------------------------------------------------------------------------

/// Memory allocation statistics.
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// GPU buffer memory in bytes.
    pub gpu_buffer_bytes: u64,
    /// GPU texture memory in bytes.
    pub gpu_texture_bytes: u64,
    /// CPU heap memory in bytes.
    pub cpu_heap_bytes: u64,
    /// Number of allocations.
    pub allocations: u32,
    /// Number of deallocations.
    pub deallocations: u32,
}

impl From<&MemoryStats> for profiler_ns::MemoryStats {
    fn from(stats: &MemoryStats) -> Self {
        profiler_ns::MemoryStats {
            gpu_buffer_bytes: stats.gpu_buffer_bytes,
            gpu_texture_bytes: stats.gpu_texture_bytes,
            cpu_heap_bytes: stats.cpu_heap_bytes,
            allocations: stats.allocations,
            deallocations: stats.deallocations,
        }
    }
}

// ---------------------------------------------------------------------------
// Profiler Markers
// ---------------------------------------------------------------------------

/// Active profiler markers for CPU timing.
#[derive(Debug)]
pub struct ProfilerMarkers {
    /// Active markers: name -> (start_time, category).
    active: HashMap<String, (Instant, String)>,
    /// Completed markers.
    completed: Vec<profiler_ns::Marker>,
    /// Next marker ID.
    next_id: AtomicU64,
    /// GPU pass timings.
    gpu_timings: Vec<profiler_ns::GpuTiming>,
}

impl ProfilerMarkers {
    /// Create a new profiler markers container.
    pub fn new() -> Self {
        Self {
            active: HashMap::new(),
            completed: Vec::new(),
            next_id: AtomicU64::new(1),
            gpu_timings: Vec::new(),
        }
    }

    /// Start a named profiling marker.
    pub fn start(&mut self, name: &str, category: &str) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.active
            .insert(name.to_string(), (Instant::now(), category.to_string()));
        id
    }

    /// End a named profiling marker.
    pub fn end(&mut self, name: &str) -> Option<profiler_ns::Marker> {
        if let Some((start, category)) = self.active.remove(name) {
            let end = Instant::now();
            let start_us = 0; // Relative timing
            let end_us = end.duration_since(start).as_micros() as u64;

            let marker = profiler_ns::Marker {
                name: name.to_string(),
                start_us,
                end_us,
                category,
                parent_id: None,
            };
            self.completed.push(marker.clone());
            Some(marker)
        } else {
            None
        }
    }

    /// Record GPU timing data.
    pub fn record_gpu_timing(&mut self, timing: profiler_ns::GpuTiming) {
        self.gpu_timings.push(timing);
    }

    /// Get all GPU timings and clear.
    pub fn take_gpu_timings(&mut self) -> Vec<profiler_ns::GpuTiming> {
        std::mem::take(&mut self.gpu_timings)
    }

    /// Clear completed markers.
    pub fn clear_completed(&mut self) {
        self.completed.clear();
    }
}

impl Default for ProfilerMarkers {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Session State
// ---------------------------------------------------------------------------

/// Session state for connection tracking.
#[derive(Debug, Clone)]
pub struct SessionState {
    /// Session ID.
    pub session_id: u64,
    /// Client name.
    pub client_name: String,
    /// Negotiated capabilities.
    pub capabilities: Vec<String>,
    /// Connection established.
    pub connected: bool,
    /// Pending request count.
    pub pending_requests: u32,
}

impl Default for SessionState {
    fn default() -> Self {
        static NEXT_SESSION: AtomicU64 = AtomicU64::new(1);
        Self {
            session_id: NEXT_SESSION.fetch_add(1, Ordering::Relaxed),
            client_name: String::new(),
            capabilities: Vec::new(),
            connected: false,
            pending_requests: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Handler Registration
// ---------------------------------------------------------------------------

/// Register all bridge handlers on the given dispatcher.
///
/// This wires each namespace's handlers to the dispatcher, providing access
/// to the engine subsystems through the shared `BridgeContext`.
pub fn register_handlers(dispatcher: &ProtocolDispatcher, context: Arc<BridgeContext>) {
    // bridge.* namespace
    register_bridge_handlers(dispatcher, context.clone());

    // entity.* namespace
    register_entity_handlers(dispatcher, context.clone());

    // component.* namespace
    register_component_handlers(dispatcher, context.clone());

    // frame.* namespace
    register_frame_handlers(dispatcher, context.clone());

    // profiler.* namespace
    register_profiler_handlers(dispatcher, context.clone());

    // editor.* namespace
    register_editor_handlers(dispatcher, context.clone());

    // material.* namespace (stubs)
    register_material_handlers(dispatcher, context.clone());

    // asset.* namespace (stubs)
    register_asset_handlers(dispatcher, context);
}

// ---------------------------------------------------------------------------
// bridge.* Handlers
// ---------------------------------------------------------------------------

fn register_bridge_handlers(dispatcher: &ProtocolDispatcher, context: Arc<BridgeContext>) {
    // bridge.handshake
    let ctx = context.clone();
    dispatcher.register_handler("bridge", "handshake", move |params: Value| {
        let request: bridge_ns::HandshakeRequest =
            serde_json::from_value(params).map_err(|e| BridgeError::InvalidParams(e.to_string()))?;

        // Check protocol version
        if request.protocol_version != PROTOCOL_VERSION {
            return Err(BridgeError::VersionMismatch {
                expected: PROTOCOL_VERSION,
                got: request.protocol_version,
            });
        }

        // Update session state
        let mut session = ctx.session.write();
        session.client_name = request.client_name;
        session.capabilities = request.capabilities.clone();
        session.connected = true;

        // Build server capabilities
        let server_caps = vec![
            "entity".to_string(),
            "component".to_string(),
            "frame".to_string(),
            "profiler".to_string(),
            "editor".to_string(),
        ];

        let response = bridge_ns::HandshakeResponse {
            protocol_version: PROTOCOL_VERSION,
            server_name: "TRINITY Renderer Backend".to_string(),
            capabilities: server_caps,
            session_id: session.session_id,
        };

        serde_json::to_value(response).map_err(|e| BridgeError::Serialization(e.to_string()))
    });

    // bridge.health
    let ctx = context.clone();
    dispatcher.register_handler("bridge", "health", move |_params: Value| {
        let session = ctx.session.read();
        let memory_stats = ctx.memory_stats.read();

        let response = bridge_ns::HealthResponse {
            status: if session.connected {
                "healthy".to_string()
            } else {
                "not_connected".to_string()
            },
            uptime_ms: ctx.uptime_ms(),
            pending_requests: session.pending_requests,
            memory_used_bytes: memory_stats.cpu_heap_bytes
                + memory_stats.gpu_buffer_bytes
                + memory_stats.gpu_texture_bytes,
        };

        serde_json::to_value(response).map_err(|e| BridgeError::Serialization(e.to_string()))
    });

    // bridge.version
    dispatcher.register_handler("bridge", "version", move |_params: Value| {
        let response = bridge_ns::VersionInfo {
            protocol_version: PROTOCOL_VERSION,
            engine_version: ENGINE_VERSION.to_string(),
            build_date: BUILD_DATE.to_string(),
            features: vec![
                "ecs".to_string(),
                "profiling".to_string(),
                "editor".to_string(),
                "hot_reload".to_string(),
            ],
        };

        serde_json::to_value(response).map_err(|e| BridgeError::Serialization(e.to_string()))
    });
}

// ---------------------------------------------------------------------------
// entity.* Handlers
// ---------------------------------------------------------------------------

fn register_entity_handlers(dispatcher: &ProtocolDispatcher, context: Arc<BridgeContext>) {
    static NEXT_ENTITY_ID: AtomicU64 = AtomicU64::new(1);
    static ENTITY_GENERATION: AtomicU64 = AtomicU64::new(1);

    // entity.spawn
    let ctx = context.clone();
    dispatcher.register_handler("entity", "spawn", move |params: Value| {
        let request: entity_ns::SpawnRequest =
            serde_json::from_value(params).map_err(|e| BridgeError::InvalidParams(e.to_string()))?;

        let entity_id = NEXT_ENTITY_ID.fetch_add(1, Ordering::Relaxed);
        let generation = ENTITY_GENERATION.load(Ordering::Relaxed) as u32;

        // Prepare component data
        let mut component_ids: Vec<u32> = Vec::new();
        let mut component_data: Vec<(u32, Vec<u8>)> = Vec::new();

        for (comp_name, value) in &request.components {
            // Map component name to ID (simplified: use hash)
            let comp_id = hash_component_name(comp_name);
            component_ids.push(comp_id);

            // Serialize component value to bytes
            let bytes = serde_json::to_vec(&value)
                .map_err(|e| BridgeError::Serialization(e.to_string()))?;
            component_data.push((comp_id, bytes));
        }

        // Spawn entity in store
        {
            let mut store = ctx.component_store.write();
            store.spawn(entity_id, &component_ids, &component_data);
        }

        let response = entity_ns::SpawnResponse {
            entity_id,
            generation,
        };

        serde_json::to_value(response).map_err(|e| BridgeError::Serialization(e.to_string()))
    });

    // entity.destroy
    let ctx = context.clone();
    dispatcher.register_handler("entity", "destroy", move |params: Value| {
        let request: entity_ns::DestroyRequest =
            serde_json::from_value(params).map_err(|e| BridgeError::InvalidParams(e.to_string()))?;

        {
            let mut store = ctx.component_store.write();

            // Check if entity exists
            if !store.entity_index.contains_key(&request.entity_id) {
                return Err(BridgeError::EntityNotFound(request.entity_id));
            }

            store.despawn(request.entity_id);
        }

        // Increment generation for future entities
        ENTITY_GENERATION.fetch_add(1, Ordering::Relaxed);

        Ok(json!({ "success": true }))
    });

    // entity.query
    let ctx = context.clone();
    dispatcher.register_handler("entity", "query", move |params: Value| {
        let request: entity_ns::QueryRequest =
            serde_json::from_value(params).map_err(|e| BridgeError::InvalidParams(e.to_string()))?;

        let store = ctx.component_store.read();

        // Map component names to IDs
        let component_ids: Vec<u32> = request
            .components
            .iter()
            .map(|name| hash_component_name(name))
            .collect();

        // Query entities
        let mut entity_ids = store.query(&component_ids);

        // Apply limit
        if let Some(limit) = request.limit {
            entity_ids.truncate(limit as usize);
        }

        // Build response with component data
        let mut entities = Vec::new();
        for entity_id in &entity_ids {
            if let Some((arch_id, row)) = store.entity_index.get(entity_id) {
                if let Some(archetype) = store.archetypes.get(arch_id) {
                    let mut components = HashMap::new();

                    for (col_idx, comp_id) in archetype.component_ids.iter().enumerate() {
                        if let Some(info) = store.registry.get(*comp_id) {
                            let stride = info.size;
                            let col = &archetype.columns[col_idx];
                            let start = row * stride;
                            let end = (start + stride).min(col.len());

                            if start < end {
                                let bytes = &col[start..end];
                                // Try to deserialize as JSON value, fallback to raw bytes
                                let value = serde_json::from_slice(bytes)
                                    .unwrap_or_else(|_| json!({"raw": bytes.to_vec()}));
                                components.insert(info.name.clone(), value);
                            }
                        }
                    }

                    entities.push(entity_ns::EntityData {
                        id: *entity_id,
                        generation: 1, // Simplified
                        components,
                    });
                }
            }
        }

        let response = entity_ns::QueryResponse {
            entities,
            total_count: entity_ids.len() as u64,
        };

        serde_json::to_value(response).map_err(|e| BridgeError::Serialization(e.to_string()))
    });
}

// ---------------------------------------------------------------------------
// component.* Handlers
// ---------------------------------------------------------------------------

fn register_component_handlers(dispatcher: &ProtocolDispatcher, context: Arc<BridgeContext>) {
    // component.get
    let ctx = context.clone();
    dispatcher.register_handler("component", "get", move |params: Value| {
        let request: component_ns::GetRequest =
            serde_json::from_value(params).map_err(|e| BridgeError::InvalidParams(e.to_string()))?;

        let store = ctx.component_store.read();

        // Check if entity exists
        let (arch_id, row) = store
            .entity_index
            .get(&request.entity_id)
            .ok_or_else(|| BridgeError::EntityNotFound(request.entity_id))?;

        let archetype = store
            .archetypes
            .get(arch_id)
            .ok_or_else(|| BridgeError::Internal("Archetype not found".to_string()))?;

        // Find component
        let comp_id = hash_component_name(&request.component);
        let col_idx = archetype
            .component_ids
            .iter()
            .position(|c| *c == comp_id)
            .ok_or_else(|| BridgeError::ComponentNotFound(request.component.clone()))?;

        // Get component data
        let info = store
            .registry
            .get(comp_id)
            .ok_or_else(|| BridgeError::ComponentNotFound(request.component.clone()))?;

        let stride = info.size;
        let col = &archetype.columns[col_idx];
        let start = row * stride;
        let end = (start + stride).min(col.len());

        if start >= end {
            return Ok(json!(null));
        }

        let bytes = &col[start..end];

        // Try to deserialize as JSON, fallback to raw bytes
        let value =
            serde_json::from_slice(bytes).unwrap_or_else(|_| json!({ "raw": bytes.to_vec() }));

        Ok(value)
    });

    // component.set
    let ctx = context.clone();
    dispatcher.register_handler("component", "set", move |params: Value| {
        let request: component_ns::SetRequest =
            serde_json::from_value(params).map_err(|e| BridgeError::InvalidParams(e.to_string()))?;

        let mut store = ctx.component_store.write();

        // Check if entity exists
        if !store.entity_index.contains_key(&request.entity_id) {
            return Err(BridgeError::EntityNotFound(request.entity_id));
        }

        let comp_id = hash_component_name(&request.component);

        // Serialize value to bytes
        let bytes = serde_json::to_vec(&request.value)
            .map_err(|e| BridgeError::Serialization(e.to_string()))?;

        // Write field at offset 0
        store.write_field(request.entity_id, comp_id, 0, &bytes);

        Ok(json!({ "success": true }))
    });

    // component.list
    let ctx = context.clone();
    dispatcher.register_handler("component", "list", move |params: Value| {
        let request: component_ns::ListRequest =
            serde_json::from_value(params).map_err(|e| BridgeError::InvalidParams(e.to_string()))?;

        let store = ctx.component_store.read();

        // Check if entity exists
        let (arch_id, _row) = store
            .entity_index
            .get(&request.entity_id)
            .ok_or_else(|| BridgeError::EntityNotFound(request.entity_id))?;

        let archetype = store
            .archetypes
            .get(arch_id)
            .ok_or_else(|| BridgeError::Internal("Archetype not found".to_string()))?;

        // Get component names
        let mut components = Vec::new();
        for comp_id in &archetype.component_ids {
            if let Some(info) = store.registry.get(*comp_id) {
                components.push(info.name.clone());
            } else {
                components.push(format!("unknown_{}", comp_id));
            }
        }

        let response = component_ns::ListResponse { components };

        serde_json::to_value(response).map_err(|e| BridgeError::Serialization(e.to_string()))
    });
}

// ---------------------------------------------------------------------------
// frame.* Handlers
// ---------------------------------------------------------------------------

fn register_frame_handlers(dispatcher: &ProtocolDispatcher, context: Arc<BridgeContext>) {
    // frame.stats
    let ctx = context.clone();
    dispatcher.register_handler("frame", "stats", move |_params: Value| {
        let history = ctx.frame_history.read();

        if let Some(stats) = history.latest() {
            serde_json::to_value(stats).map_err(|e| BridgeError::Serialization(e.to_string()))
        } else {
            // Return default stats if no frames recorded yet
            let default_stats = frame_ns::FrameStats {
                frame_number: history.frame_number(),
                delta_time_ms: 16.67,
                fps: 60.0,
                frame_time_ms: 16.67,
                cpu_time_ms: 0.0,
                gpu_time_ms: 0.0,
                draw_calls: 0,
                triangles: 0,
                vertices: 0,
            };
            serde_json::to_value(default_stats)
                .map_err(|e| BridgeError::Serialization(e.to_string()))
        }
    });

    // frame.history
    let ctx = context.clone();
    dispatcher.register_handler("frame", "history", move |params: Value| {
        let count = params
            .get("count")
            .and_then(|v| v.as_u64())
            .unwrap_or(60) as usize;

        let history = ctx.frame_history.read();
        let frames: Vec<_> = history.last_n(count).into_iter().cloned().collect();

        Ok(json!({
            "frames": frames,
            "count": frames.len(),
            "total_frames": history.frame_number()
        }))
    });
}

// ---------------------------------------------------------------------------
// profiler.* Handlers
// ---------------------------------------------------------------------------

fn register_profiler_handlers(dispatcher: &ProtocolDispatcher, context: Arc<BridgeContext>) {
    // profiler.start_marker
    let ctx = context.clone();
    dispatcher.register_handler("profiler", "start_marker", move |params: Value| {
        let name = params
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| BridgeError::InvalidParams("Missing 'name' parameter".to_string()))?;

        let category = params
            .get("category")
            .and_then(|v| v.as_str())
            .unwrap_or("default");

        let mut markers = ctx.profiler_markers.write();
        let marker_id = markers.start(name, category);

        Ok(json!({
            "marker_id": marker_id,
            "name": name,
            "category": category
        }))
    });

    // profiler.end_marker
    let ctx = context.clone();
    dispatcher.register_handler("profiler", "end_marker", move |params: Value| {
        let name = params
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| BridgeError::InvalidParams("Missing 'name' parameter".to_string()))?;

        let mut markers = ctx.profiler_markers.write();

        if let Some(marker) = markers.end(name) {
            serde_json::to_value(marker).map_err(|e| BridgeError::Serialization(e.to_string()))
        } else {
            Err(BridgeError::Internal(format!(
                "Marker '{}' not found or already ended",
                name
            )))
        }
    });

    // profiler.gpu_timing
    let ctx = context.clone();
    dispatcher.register_handler("profiler", "gpu_timing", move |_params: Value| {
        let mut markers = ctx.profiler_markers.write();
        let timings = markers.take_gpu_timings();

        Ok(json!({
            "timings": timings,
            "count": timings.len()
        }))
    });

    // profiler.memory
    let ctx = context.clone();
    dispatcher.register_handler("profiler", "memory", move |_params: Value| {
        let stats = ctx.memory_stats.read();
        let response: profiler_ns::MemoryStats = (&*stats).into();

        serde_json::to_value(response).map_err(|e| BridgeError::Serialization(e.to_string()))
    });
}

// ---------------------------------------------------------------------------
// editor.* Handlers
// ---------------------------------------------------------------------------

fn register_editor_handlers(dispatcher: &ProtocolDispatcher, context: Arc<BridgeContext>) {
    // editor.select
    let ctx = context.clone();
    dispatcher.register_handler("editor", "select", move |params: Value| {
        let entity_id = params
            .get("entity_id")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                BridgeError::InvalidParams("Missing 'entity_id' parameter".to_string())
            })?;

        // Verify entity exists
        {
            let store = ctx.component_store.read();
            if !store.entity_index.contains_key(&entity_id) {
                return Err(BridgeError::EntityNotFound(entity_id));
            }
        }

        let mut editor = ctx.editor.write();
        editor.select_entity(entity_id);

        Ok(json!({
            "selected": entity_id,
            "success": true
        }))
    });

    // editor.get_selection
    let ctx = context.clone();
    dispatcher.register_handler("editor", "get_selection", move |_params: Value| {
        let editor = ctx.editor.read();

        let response = editor_ns::Selection {
            entities: editor
                .state
                .selected_entity
                .map(|e| vec![e])
                .unwrap_or_default(),
            primary: editor.state.selected_entity,
        };

        serde_json::to_value(response).map_err(|e| BridgeError::Serialization(e.to_string()))
    });

    // editor.set_viewport
    let ctx = context.clone();
    dispatcher.register_handler("editor", "set_viewport", move |params: Value| {
        let viewport: editor_ns::Viewport = serde_json::from_value(params)
            .map_err(|e| BridgeError::InvalidParams(e.to_string()))?;

        // For now, just acknowledge the viewport settings
        // In a full implementation, this would update camera/rendering state
        Ok(json!({
            "success": true,
            "viewport": {
                "width": viewport.width,
                "height": viewport.height,
                "fov": viewport.fov_degrees
            }
        }))
    });

    // editor.deselect
    let ctx = context.clone();
    dispatcher.register_handler("editor", "deselect", move |_params: Value| {
        let mut editor = ctx.editor.write();
        editor.deselect();

        Ok(json!({
            "success": true,
            "selected": null
        }))
    });
}

// ---------------------------------------------------------------------------
// material.* Handlers (Stubs)
// ---------------------------------------------------------------------------

fn register_material_handlers(dispatcher: &ProtocolDispatcher, _context: Arc<BridgeContext>) {
    // material.get
    dispatcher.register_handler("material", "get", |_params: Value| {
        Err(BridgeError::Internal(
            "material.get not implemented".to_string(),
        ))
    });

    // material.set
    dispatcher.register_handler("material", "set", |_params: Value| {
        Err(BridgeError::Internal(
            "material.set not implemented".to_string(),
        ))
    });

    // material.compile
    dispatcher.register_handler("material", "compile", |_params: Value| {
        Err(BridgeError::Internal(
            "material.compile not implemented".to_string(),
        ))
    });

    // material.list
    dispatcher.register_handler("material", "list", |_params: Value| {
        Err(BridgeError::Internal(
            "material.list not implemented".to_string(),
        ))
    });
}

// ---------------------------------------------------------------------------
// asset.* Handlers (Stubs)
// ---------------------------------------------------------------------------

fn register_asset_handlers(dispatcher: &ProtocolDispatcher, _context: Arc<BridgeContext>) {
    // asset.load
    dispatcher.register_handler("asset", "load", |_params: Value| {
        Err(BridgeError::Internal(
            "asset.load not implemented".to_string(),
        ))
    });

    // asset.unload
    dispatcher.register_handler("asset", "unload", |_params: Value| {
        Err(BridgeError::Internal(
            "asset.unload not implemented".to_string(),
        ))
    });

    // asset.status
    dispatcher.register_handler("asset", "status", |_params: Value| {
        Err(BridgeError::Internal(
            "asset.status not implemented".to_string(),
        ))
    });

    // asset.hot_reload
    dispatcher.register_handler("asset", "hot_reload", |_params: Value| {
        Err(BridgeError::Internal(
            "asset.hot_reload not implemented".to_string(),
        ))
    });
}

// ---------------------------------------------------------------------------
// Utility Functions
// ---------------------------------------------------------------------------

/// Hash a component name to a u32 ID.
///
/// This is a simple hash for demonstration. In production, component IDs
/// would be registered in the TypeRegistry.
fn hash_component_name(name: &str) -> u32 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    name.hash(&mut hasher);
    hasher.finish() as u32
}

/// Create a default BridgeContext using the global component store.
///
/// # Panics
///
/// Panics if the global component store has not been initialized.
pub fn default_bridge_context() -> Arc<BridgeContext> {
    Arc::new(BridgeContext::new(global_component_store().clone()))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component_store::ComponentStore;
    use crate::type_registry::{ComponentTypeInfo, TypeRegistry};

    // ===== Test Helpers =====

    fn make_test_registry() -> Arc<TypeRegistry> {
        let registry = TypeRegistry::new();
        registry.register(ComponentTypeInfo {
            id: hash_component_name("Position"),
            name: "Position".into(),
            size: 64,
            fields: vec![],
            flags: 0,
            archetype_id: None,
        });
        registry.register(ComponentTypeInfo {
            id: hash_component_name("Velocity"),
            name: "Velocity".into(),
            size: 32,
            fields: vec![],
            flags: 0,
            archetype_id: None,
        });
        registry.register(ComponentTypeInfo {
            id: hash_component_name("Health"),
            name: "Health".into(),
            size: 16,
            fields: vec![],
            flags: 0,
            archetype_id: None,
        });
        Arc::new(registry)
    }

    fn make_test_context() -> Arc<BridgeContext> {
        let registry = make_test_registry();
        let store = Arc::new(RwLock::new(ComponentStore::new(registry)));
        Arc::new(BridgeContext::new(store))
    }

    fn make_dispatcher_with_handlers() -> (ProtocolDispatcher, Arc<BridgeContext>) {
        let context = make_test_context();
        let dispatcher = ProtocolDispatcher::new();
        register_handlers(&dispatcher, context.clone());
        (dispatcher, context)
    }

    // ===== SECTION 1: FrameHistory tests =====

    #[test]
    fn frame_history_new_creates_empty() {
        let history = FrameHistory::new(10);
        assert!(history.latest().is_none());
        assert_eq!(history.frame_number(), 0);
    }

    #[test]
    fn frame_history_record_stores_frame() {
        let mut history = FrameHistory::new(10);
        let stats = frame_ns::FrameStats {
            frame_number: 1,
            delta_time_ms: 16.67,
            fps: 60.0,
            frame_time_ms: 16.67,
            cpu_time_ms: 8.0,
            gpu_time_ms: 8.67,
            draw_calls: 100,
            triangles: 50000,
            vertices: 100000,
        };

        history.record(stats.clone());

        assert_eq!(history.frame_number(), 1);
        let latest = history.latest().unwrap();
        assert_eq!(latest.frame_number, 1);
    }

    #[test]
    fn frame_history_respects_capacity() {
        let mut history = FrameHistory::new(3);

        for i in 0..5 {
            history.record(frame_ns::FrameStats {
                frame_number: i,
                delta_time_ms: 16.67,
                fps: 60.0,
                frame_time_ms: 16.67,
                cpu_time_ms: 0.0,
                gpu_time_ms: 0.0,
                draw_calls: 0,
                triangles: 0,
                vertices: 0,
            });
        }

        // Only last 3 should remain
        assert_eq!(history.all().len(), 3);
        assert_eq!(history.all()[0].frame_number, 2);
    }

    #[test]
    fn frame_history_last_n_returns_correct_count() {
        let mut history = FrameHistory::new(10);

        for i in 0..5 {
            history.record(frame_ns::FrameStats {
                frame_number: i,
                delta_time_ms: 16.67,
                fps: 60.0,
                frame_time_ms: 16.67,
                cpu_time_ms: 0.0,
                gpu_time_ms: 0.0,
                draw_calls: 0,
                triangles: 0,
                vertices: 0,
            });
        }

        let last_3 = history.last_n(3);
        assert_eq!(last_3.len(), 3);
        assert_eq!(last_3[0].frame_number, 4); // Most recent first
    }

    // ===== SECTION 2: MemoryStats tests =====

    #[test]
    fn memory_stats_default_is_zero() {
        let stats = MemoryStats::default();
        assert_eq!(stats.gpu_buffer_bytes, 0);
        assert_eq!(stats.gpu_texture_bytes, 0);
        assert_eq!(stats.cpu_heap_bytes, 0);
        assert_eq!(stats.allocations, 0);
        assert_eq!(stats.deallocations, 0);
    }

    #[test]
    fn memory_stats_converts_to_protocol() {
        let stats = MemoryStats {
            gpu_buffer_bytes: 1000,
            gpu_texture_bytes: 2000,
            cpu_heap_bytes: 3000,
            allocations: 10,
            deallocations: 5,
        };

        let proto: profiler_ns::MemoryStats = (&stats).into();
        assert_eq!(proto.gpu_buffer_bytes, 1000);
        assert_eq!(proto.gpu_texture_bytes, 2000);
    }

    // ===== SECTION 3: ProfilerMarkers tests =====

    #[test]
    fn profiler_markers_start_returns_id() {
        let mut markers = ProfilerMarkers::new();
        let id = markers.start("test", "category");
        assert!(id > 0);
    }

    #[test]
    fn profiler_markers_end_returns_marker() {
        let mut markers = ProfilerMarkers::new();
        markers.start("test", "category");

        // Small delay
        std::thread::sleep(Duration::from_millis(1));

        let marker = markers.end("test");
        assert!(marker.is_some());
        let marker = marker.unwrap();
        assert_eq!(marker.name, "test");
        assert_eq!(marker.category, "category");
        assert!(marker.end_us > 0);
    }

    #[test]
    fn profiler_markers_end_unknown_returns_none() {
        let mut markers = ProfilerMarkers::new();
        assert!(markers.end("nonexistent").is_none());
    }

    #[test]
    fn profiler_markers_gpu_timing_stored() {
        let mut markers = ProfilerMarkers::new();
        markers.record_gpu_timing(profiler_ns::GpuTiming {
            pass_name: "GBuffer".to_string(),
            duration_us: 5000,
            timestamp_start: 0,
            timestamp_end: 5000,
        });

        let timings = markers.take_gpu_timings();
        assert_eq!(timings.len(), 1);
        assert_eq!(timings[0].pass_name, "GBuffer");
    }

    // ===== SECTION 4: SessionState tests =====

    #[test]
    fn session_state_default_has_unique_id() {
        let s1 = SessionState::default();
        let s2 = SessionState::default();
        assert_ne!(s1.session_id, s2.session_id);
    }

    #[test]
    fn session_state_default_not_connected() {
        let state = SessionState::default();
        assert!(!state.connected);
        assert!(state.client_name.is_empty());
    }

    // ===== SECTION 5: BridgeContext tests =====

    #[test]
    fn bridge_context_uptime_increases() {
        let ctx = make_test_context();
        let uptime1 = ctx.uptime_ms();
        std::thread::sleep(Duration::from_millis(10));
        let uptime2 = ctx.uptime_ms();
        assert!(uptime2 >= uptime1);
    }

    // ===== SECTION 6: bridge.* handler tests =====

    #[test]
    fn handler_bridge_version_returns_info() {
        let (dispatcher, _ctx) = make_dispatcher_with_handlers();

        let request = crate::bridge_protocol::CommandRequest::new("bridge", "version", json!({}));
        let response = dispatcher.dispatch(request);

        assert!(response.is_ok());
        let value = response.value().unwrap();
        assert_eq!(value["protocol_version"], PROTOCOL_VERSION);
        assert!(value["engine_version"].as_str().is_some());
    }

    #[test]
    fn handler_bridge_health_returns_status() {
        let (dispatcher, _ctx) = make_dispatcher_with_handlers();

        let request = crate::bridge_protocol::CommandRequest::new("bridge", "health", json!({}));
        let response = dispatcher.dispatch(request);

        assert!(response.is_ok());
        let value = response.value().unwrap();
        assert!(value["status"].as_str().is_some());
        assert!(value["uptime_ms"].as_u64().is_some());
    }

    #[test]
    fn handler_bridge_handshake_success() {
        let (dispatcher, _ctx) = make_dispatcher_with_handlers();

        let request = crate::bridge_protocol::CommandRequest::new(
            "bridge",
            "handshake",
            json!({
                "protocol_version": PROTOCOL_VERSION,
                "client_name": "TestClient",
                "capabilities": ["debug"]
            }),
        );
        let response = dispatcher.dispatch(request);

        assert!(response.is_ok());
        let value = response.value().unwrap();
        assert_eq!(value["protocol_version"], PROTOCOL_VERSION);
        assert!(value["session_id"].as_u64().is_some());
    }

    #[test]
    fn handler_bridge_handshake_version_mismatch() {
        let (dispatcher, _ctx) = make_dispatcher_with_handlers();

        let request = crate::bridge_protocol::CommandRequest::new(
            "bridge",
            "handshake",
            json!({
                "protocol_version": 999,
                "client_name": "TestClient",
                "capabilities": []
            }),
        );
        let response = dispatcher.dispatch(request);

        assert!(!response.is_ok());
        assert!(matches!(
            response.get_error(),
            Some(BridgeError::VersionMismatch { .. })
        ));
    }

    // ===== SECTION 7: entity.* handler tests =====

    #[test]
    fn handler_entity_spawn_creates_entity() {
        let (dispatcher, ctx) = make_dispatcher_with_handlers();

        let request = crate::bridge_protocol::CommandRequest::new(
            "entity",
            "spawn",
            json!({
                "archetype": "Player",
                "name": "TestEntity",
                "components": {
                    "Position": {"x": 0, "y": 0, "z": 0}
                }
            }),
        );
        let response = dispatcher.dispatch(request);

        assert!(response.is_ok());
        let value = response.value().unwrap();
        let entity_id = value["entity_id"].as_u64().unwrap();
        assert!(entity_id > 0);

        // Verify entity exists in store
        let store = ctx.component_store.read();
        assert!(store.entity_index.contains_key(&entity_id));
    }

    #[test]
    fn handler_entity_destroy_removes_entity() {
        let (dispatcher, ctx) = make_dispatcher_with_handlers();

        // Spawn first
        let spawn_req = crate::bridge_protocol::CommandRequest::new(
            "entity",
            "spawn",
            json!({
                "archetype": "Test",
                "components": {}
            }),
        );
        let spawn_resp = dispatcher.dispatch(spawn_req);
        let entity_id = spawn_resp.value().unwrap()["entity_id"].as_u64().unwrap();

        // Destroy
        let destroy_req = crate::bridge_protocol::CommandRequest::new(
            "entity",
            "destroy",
            json!({
                "entity_id": entity_id,
                "recursive": false
            }),
        );
        let destroy_resp = dispatcher.dispatch(destroy_req);

        assert!(destroy_resp.is_ok());

        // Verify entity gone
        let store = ctx.component_store.read();
        assert!(!store.entity_index.contains_key(&entity_id));
    }

    #[test]
    fn handler_entity_destroy_nonexistent_fails() {
        let (dispatcher, _ctx) = make_dispatcher_with_handlers();

        let request = crate::bridge_protocol::CommandRequest::new(
            "entity",
            "destroy",
            json!({
                "entity_id": 999999,
                "recursive": false
            }),
        );
        let response = dispatcher.dispatch(request);

        assert!(!response.is_ok());
        assert!(matches!(
            response.get_error(),
            Some(BridgeError::EntityNotFound(999999))
        ));
    }

    #[test]
    fn handler_entity_query_returns_matching() {
        let (dispatcher, _ctx) = make_dispatcher_with_handlers();

        // Spawn some entities
        for i in 0..3 {
            dispatcher.dispatch(crate::bridge_protocol::CommandRequest::new(
                "entity",
                "spawn",
                json!({
                    "archetype": "Test",
                    "name": format!("Entity{}", i),
                    "components": {
                        "Position": {"x": i}
                    }
                }),
            ));
        }

        // Query
        let request = crate::bridge_protocol::CommandRequest::new(
            "entity",
            "query",
            json!({
                "components": ["Position"]
            }),
        );
        let response = dispatcher.dispatch(request);

        assert!(response.is_ok());
        let value = response.value().unwrap();
        assert!(value["total_count"].as_u64().unwrap() >= 3);
    }

    // ===== SECTION 8: component.* handler tests =====

    #[test]
    fn handler_component_get_returns_data() {
        let (dispatcher, _ctx) = make_dispatcher_with_handlers();

        // Spawn entity with component
        let spawn_resp = dispatcher.dispatch(crate::bridge_protocol::CommandRequest::new(
            "entity",
            "spawn",
            json!({
                "archetype": "Test",
                "components": {
                    "Position": {"x": 10, "y": 20, "z": 30}
                }
            }),
        ));
        let entity_id = spawn_resp.value().unwrap()["entity_id"].as_u64().unwrap();

        // Get component
        let request = crate::bridge_protocol::CommandRequest::new(
            "component",
            "get",
            json!({
                "entity_id": entity_id,
                "component": "Position"
            }),
        );
        let response = dispatcher.dispatch(request);

        assert!(response.is_ok());
    }

    #[test]
    fn handler_component_set_updates_data() {
        let (dispatcher, _ctx) = make_dispatcher_with_handlers();

        // Spawn
        let spawn_resp = dispatcher.dispatch(crate::bridge_protocol::CommandRequest::new(
            "entity",
            "spawn",
            json!({
                "archetype": "Test",
                "components": {
                    "Position": {"x": 0, "y": 0, "z": 0}
                }
            }),
        ));
        let entity_id = spawn_resp.value().unwrap()["entity_id"].as_u64().unwrap();

        // Set
        let request = crate::bridge_protocol::CommandRequest::new(
            "component",
            "set",
            json!({
                "entity_id": entity_id,
                "component": "Position",
                "value": {"x": 100, "y": 200, "z": 300}
            }),
        );
        let response = dispatcher.dispatch(request);

        assert!(response.is_ok());
        assert_eq!(response.value().unwrap()["success"], true);
    }

    #[test]
    fn handler_component_list_returns_names() {
        let (dispatcher, _ctx) = make_dispatcher_with_handlers();

        // Spawn with multiple components
        let spawn_resp = dispatcher.dispatch(crate::bridge_protocol::CommandRequest::new(
            "entity",
            "spawn",
            json!({
                "archetype": "Test",
                "components": {
                    "Position": {},
                    "Velocity": {}
                }
            }),
        ));
        let entity_id = spawn_resp.value().unwrap()["entity_id"].as_u64().unwrap();

        // List
        let request = crate::bridge_protocol::CommandRequest::new(
            "component",
            "list",
            json!({
                "entity_id": entity_id
            }),
        );
        let response = dispatcher.dispatch(request);

        assert!(response.is_ok());
        let components = response.value().unwrap()["components"].as_array().unwrap();
        assert_eq!(components.len(), 2);
    }

    // ===== SECTION 9: frame.* handler tests =====

    #[test]
    fn handler_frame_stats_returns_data() {
        let (dispatcher, ctx) = make_dispatcher_with_handlers();

        // Record a frame
        {
            let mut history = ctx.frame_history.write();
            history.record(frame_ns::FrameStats {
                frame_number: 1,
                delta_time_ms: 16.67,
                fps: 60.0,
                frame_time_ms: 16.67,
                cpu_time_ms: 8.0,
                gpu_time_ms: 8.67,
                draw_calls: 150,
                triangles: 100000,
                vertices: 200000,
            });
        }

        let request = crate::bridge_protocol::CommandRequest::new("frame", "stats", json!({}));
        let response = dispatcher.dispatch(request);

        assert!(response.is_ok());
        let value = response.value().unwrap();
        assert_eq!(value["frame_number"], 1);
        assert_eq!(value["draw_calls"], 150);
    }

    #[test]
    fn handler_frame_history_returns_frames() {
        let (dispatcher, ctx) = make_dispatcher_with_handlers();

        // Record frames
        {
            let mut history = ctx.frame_history.write();
            for i in 0..5 {
                history.record(frame_ns::FrameStats {
                    frame_number: i,
                    delta_time_ms: 16.67,
                    fps: 60.0,
                    frame_time_ms: 16.67,
                    cpu_time_ms: 0.0,
                    gpu_time_ms: 0.0,
                    draw_calls: 0,
                    triangles: 0,
                    vertices: 0,
                });
            }
        }

        let request =
            crate::bridge_protocol::CommandRequest::new("frame", "history", json!({"count": 3}));
        let response = dispatcher.dispatch(request);

        assert!(response.is_ok());
        let value = response.value().unwrap();
        assert_eq!(value["count"], 3);
    }

    // ===== SECTION 10: profiler.* handler tests =====

    #[test]
    fn handler_profiler_start_marker_creates_marker() {
        let (dispatcher, _ctx) = make_dispatcher_with_handlers();

        let request = crate::bridge_protocol::CommandRequest::new(
            "profiler",
            "start_marker",
            json!({
                "name": "TestMarker",
                "category": "test"
            }),
        );
        let response = dispatcher.dispatch(request);

        assert!(response.is_ok());
        let value = response.value().unwrap();
        assert!(value["marker_id"].as_u64().is_some());
        assert_eq!(value["name"], "TestMarker");
    }

    #[test]
    fn handler_profiler_end_marker_completes() {
        let (dispatcher, _ctx) = make_dispatcher_with_handlers();

        // Start
        dispatcher.dispatch(crate::bridge_protocol::CommandRequest::new(
            "profiler",
            "start_marker",
            json!({
                "name": "TestMarker",
                "category": "test"
            }),
        ));

        // End
        let request = crate::bridge_protocol::CommandRequest::new(
            "profiler",
            "end_marker",
            json!({
                "name": "TestMarker"
            }),
        );
        let response = dispatcher.dispatch(request);

        assert!(response.is_ok());
        let value = response.value().unwrap();
        assert_eq!(value["name"], "TestMarker");
    }

    #[test]
    fn handler_profiler_memory_returns_stats() {
        let (dispatcher, _ctx) = make_dispatcher_with_handlers();

        let request = crate::bridge_protocol::CommandRequest::new("profiler", "memory", json!({}));
        let response = dispatcher.dispatch(request);

        assert!(response.is_ok());
        let value = response.value().unwrap();
        assert!(value["gpu_buffer_bytes"].as_u64().is_some());
    }

    // ===== SECTION 11: editor.* handler tests =====

    #[test]
    fn handler_editor_select_sets_selection() {
        let (dispatcher, ctx) = make_dispatcher_with_handlers();

        // Spawn entity first
        let spawn_resp = dispatcher.dispatch(crate::bridge_protocol::CommandRequest::new(
            "entity",
            "spawn",
            json!({
                "archetype": "Test",
                "components": {}
            }),
        ));
        let entity_id = spawn_resp.value().unwrap()["entity_id"].as_u64().unwrap();

        // Select
        let request = crate::bridge_protocol::CommandRequest::new(
            "editor",
            "select",
            json!({
                "entity_id": entity_id
            }),
        );
        let response = dispatcher.dispatch(request);

        assert!(response.is_ok());

        // Verify
        let editor = ctx.editor.read();
        assert_eq!(editor.state.selected_entity, Some(entity_id));
    }

    #[test]
    fn handler_editor_select_nonexistent_fails() {
        let (dispatcher, _ctx) = make_dispatcher_with_handlers();

        let request = crate::bridge_protocol::CommandRequest::new(
            "editor",
            "select",
            json!({
                "entity_id": 999999
            }),
        );
        let response = dispatcher.dispatch(request);

        assert!(!response.is_ok());
        assert!(matches!(
            response.get_error(),
            Some(BridgeError::EntityNotFound(999999))
        ));
    }

    #[test]
    fn handler_editor_get_selection_returns_current() {
        let (dispatcher, ctx) = make_dispatcher_with_handlers();

        // Spawn and select
        let spawn_resp = dispatcher.dispatch(crate::bridge_protocol::CommandRequest::new(
            "entity",
            "spawn",
            json!({ "archetype": "Test", "components": {} }),
        ));
        let entity_id = spawn_resp.value().unwrap()["entity_id"].as_u64().unwrap();

        {
            let mut editor = ctx.editor.write();
            editor.select_entity(entity_id);
        }

        let request =
            crate::bridge_protocol::CommandRequest::new("editor", "get_selection", json!({}));
        let response = dispatcher.dispatch(request);

        assert!(response.is_ok());
        let value = response.value().unwrap();
        assert_eq!(value["primary"], entity_id);
    }

    #[test]
    fn handler_editor_deselect_clears_selection() {
        let (dispatcher, ctx) = make_dispatcher_with_handlers();

        // Select something first
        {
            let mut editor = ctx.editor.write();
            editor.select_entity(123);
        }

        let request =
            crate::bridge_protocol::CommandRequest::new("editor", "deselect", json!({}));
        let response = dispatcher.dispatch(request);

        assert!(response.is_ok());

        let editor = ctx.editor.read();
        assert!(editor.state.selected_entity.is_none());
    }

    #[test]
    fn handler_editor_set_viewport_succeeds() {
        let (dispatcher, _ctx) = make_dispatcher_with_handlers();

        let request = crate::bridge_protocol::CommandRequest::new(
            "editor",
            "set_viewport",
            json!({
                "width": 1920,
                "height": 1080,
                "camera_position": [0.0, 10.0, -20.0],
                "camera_rotation": [0.0, 0.0, 0.0, 1.0],
                "fov_degrees": 60.0,
                "near_clip": 0.1,
                "far_clip": 1000.0
            }),
        );
        let response = dispatcher.dispatch(request);

        assert!(response.is_ok());
        assert_eq!(response.value().unwrap()["success"], true);
    }

    // ===== SECTION 12: material.* stub tests =====

    #[test]
    fn handler_material_get_returns_not_implemented() {
        let (dispatcher, _ctx) = make_dispatcher_with_handlers();

        let request = crate::bridge_protocol::CommandRequest::new("material", "get", json!({}));
        let response = dispatcher.dispatch(request);

        assert!(!response.is_ok());
        assert!(matches!(
            response.get_error(),
            Some(BridgeError::Internal(_))
        ));
    }

    // ===== SECTION 13: asset.* stub tests =====

    #[test]
    fn handler_asset_load_returns_not_implemented() {
        let (dispatcher, _ctx) = make_dispatcher_with_handlers();

        let request = crate::bridge_protocol::CommandRequest::new("asset", "load", json!({}));
        let response = dispatcher.dispatch(request);

        assert!(!response.is_ok());
        assert!(matches!(
            response.get_error(),
            Some(BridgeError::Internal(_))
        ));
    }

    // ===== SECTION 14: hash_component_name tests =====

    #[test]
    fn hash_component_name_deterministic() {
        let h1 = hash_component_name("Position");
        let h2 = hash_component_name("Position");
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_component_name_different_for_different_names() {
        let h1 = hash_component_name("Position");
        let h2 = hash_component_name("Velocity");
        assert_ne!(h1, h2);
    }

    // ===== SECTION 15: Integration tests =====

    #[test]
    fn integration_spawn_query_destroy_lifecycle() {
        let (dispatcher, _ctx) = make_dispatcher_with_handlers();

        // Spawn
        let spawn_resp = dispatcher.dispatch(crate::bridge_protocol::CommandRequest::new(
            "entity",
            "spawn",
            json!({
                "archetype": "Player",
                "components": {
                    "Position": {"x": 0, "y": 0, "z": 0},
                    "Health": {"value": 100}
                }
            }),
        ));
        assert!(spawn_resp.is_ok());
        let entity_id = spawn_resp.value().unwrap()["entity_id"].as_u64().unwrap();

        // Query
        let query_resp = dispatcher.dispatch(crate::bridge_protocol::CommandRequest::new(
            "entity",
            "query",
            json!({ "components": ["Position", "Health"] }),
        ));
        assert!(query_resp.is_ok());
        let entities = query_resp.value().unwrap()["entities"].as_array().unwrap();
        assert!(entities.iter().any(|e| e["id"] == entity_id));

        // Destroy
        let destroy_resp = dispatcher.dispatch(crate::bridge_protocol::CommandRequest::new(
            "entity",
            "destroy",
            json!({ "entity_id": entity_id, "recursive": false }),
        ));
        assert!(destroy_resp.is_ok());

        // Query again - should not contain entity
        let query_resp2 = dispatcher.dispatch(crate::bridge_protocol::CommandRequest::new(
            "entity",
            "query",
            json!({ "components": ["Position"] }),
        ));
        let entities2 = query_resp2.value().unwrap()["entities"].as_array().unwrap();
        assert!(!entities2.iter().any(|e| e["id"] == entity_id));
    }

    #[test]
    fn integration_editor_select_get_deselect() {
        let (dispatcher, _ctx) = make_dispatcher_with_handlers();

        // Spawn
        let spawn_resp = dispatcher.dispatch(crate::bridge_protocol::CommandRequest::new(
            "entity",
            "spawn",
            json!({ "archetype": "Test", "components": {} }),
        ));
        let entity_id = spawn_resp.value().unwrap()["entity_id"].as_u64().unwrap();

        // Select
        let select_resp = dispatcher.dispatch(crate::bridge_protocol::CommandRequest::new(
            "editor",
            "select",
            json!({ "entity_id": entity_id }),
        ));
        assert!(select_resp.is_ok());

        // Get selection
        let get_resp = dispatcher.dispatch(crate::bridge_protocol::CommandRequest::new(
            "editor",
            "get_selection",
            json!({}),
        ));
        assert_eq!(get_resp.value().unwrap()["primary"], entity_id);

        // Deselect
        dispatcher.dispatch(crate::bridge_protocol::CommandRequest::new(
            "editor",
            "deselect",
            json!({}),
        ));

        // Get selection again
        let get_resp2 = dispatcher.dispatch(crate::bridge_protocol::CommandRequest::new(
            "editor",
            "get_selection",
            json!({}),
        ));
        assert!(get_resp2.value().unwrap()["primary"].is_null());
    }

    #[test]
    fn integration_profiler_marker_lifecycle() {
        let (dispatcher, _ctx) = make_dispatcher_with_handlers();

        // Start marker
        let start_resp = dispatcher.dispatch(crate::bridge_protocol::CommandRequest::new(
            "profiler",
            "start_marker",
            json!({ "name": "RenderLoop", "category": "gpu" }),
        ));
        assert!(start_resp.is_ok());

        // Small delay
        std::thread::sleep(Duration::from_millis(1));

        // End marker
        let end_resp = dispatcher.dispatch(crate::bridge_protocol::CommandRequest::new(
            "profiler",
            "end_marker",
            json!({ "name": "RenderLoop" }),
        ));
        assert!(end_resp.is_ok());
        let marker = end_resp.value().unwrap();
        assert_eq!(marker["name"], "RenderLoop");
        assert!(marker["end_us"].as_u64().unwrap() > 0);
    }
}
