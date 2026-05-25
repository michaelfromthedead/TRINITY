//! Trinity introspection commands
//!
//! Provides Tauri commands for introspecting the Trinity ECS runtime
//! through the Python sidecar. These commands allow the frontend to:
//! - Check Trinity availability and version
//! - List registered types from the Trinity Registry
//! - Query active component instances
//! - Retrieve recent events from the EventLog

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Mutex;
use tauri::State;

use crate::sidecar::{PythonSidecar, SidecarError};

// =============================================================================
// Response Types
// =============================================================================

/// Status of the Trinity runtime
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrinityStatus {
    /// Whether Trinity is available and responding
    pub available: bool,
    /// Trinity version if available
    pub version: Option<String>,
    /// Error message if Trinity is not available
    pub error: Option<String>,
}

/// Entry from the Trinity Registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryEntry {
    /// Name of the registered type
    pub name: String,
    /// Kind of type: "component", "system", "resource", "event"
    pub type_kind: String,
    /// Module where the type is defined
    pub module: String,
    /// Optional additional metadata
    #[serde(default)]
    pub metadata: Option<Value>,
}

/// Response containing registry entries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryListResponse {
    /// List of registry entries
    pub entries: Vec<RegistryEntry>,
    /// Total count of entries
    pub total: usize,
}

/// Information about active component instances
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceInfo {
    /// Component type name
    pub component: String,
    /// Number of active instances
    pub count: u32,
    /// Optional instance data (entity IDs, sample data, etc.)
    #[serde(default)]
    pub data: Option<Value>,
}

/// Response containing instance information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstancesQueryResponse {
    /// List of instance information by component type
    pub instances: Vec<InstanceInfo>,
    /// Total number of entities in the world
    pub total_entities: u32,
}

/// Entry from the Trinity EventLog
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventEntry {
    /// Type/name of the event
    pub event_type: String,
    /// ISO 8601 timestamp when the event occurred
    pub timestamp: String,
    /// Event payload data
    pub data: Value,
    /// Optional event ID for correlation
    #[serde(default)]
    pub event_id: Option<String>,
}

/// Response containing recent events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventsRecentResponse {
    /// List of recent events (most recent first)
    pub events: Vec<EventEntry>,
    /// Total number of events in the log
    pub total_in_log: usize,
}

// =============================================================================
// Sidecar State Wrapper
// =============================================================================

/// Thread-safe wrapper for the Python sidecar
pub struct SidecarState(pub Mutex<Option<PythonSidecar>>);

impl SidecarState {
    /// Create a new empty sidecar state
    pub fn new() -> Self {
        Self(Mutex::new(None))
    }

    /// Create sidecar state with an already-spawned sidecar
    pub fn with_sidecar(sidecar: PythonSidecar) -> Self {
        Self(Mutex::new(Some(sidecar)))
    }

    /// Ensure the sidecar is running, spawning it if necessary
    fn ensure_running(&self) -> Result<(), String> {
        let mut guard = self.0.lock().map_err(|e| format!("Lock poisoned: {}", e))?;

        match &mut *guard {
            Some(sidecar) => {
                // Check if existing sidecar is still running
                if !sidecar.is_running() {
                    // Try to restart
                    sidecar
                        .restart_if_crashed()
                        .map_err(|e| format!("Failed to restart sidecar: {}", e))?;
                }
                Ok(())
            }
            None => {
                // Spawn new sidecar
                let sidecar = PythonSidecar::spawn()
                    .map_err(|e| format!("Failed to spawn sidecar: {}", e))?;
                *guard = Some(sidecar);
                Ok(())
            }
        }
    }

    /// Send a request to the sidecar, with automatic retry on crash
    pub fn send_request(&self, method: &str, params: Value) -> Result<Value, String> {
        self.ensure_running()?;

        let mut guard = self.0.lock().map_err(|e| format!("Lock poisoned: {}", e))?;

        let sidecar = guard
            .as_mut()
            .ok_or_else(|| "Sidecar not initialized".to_string())?;

        match sidecar.send_request(method, params.clone()) {
            Ok(result) => Ok(result),
            Err(e) => {
                // If the sidecar crashed, attempt recovery and retry once
                let err_str = format!("{}", e);
                if err_str.contains("NotRunning")
                    || err_str.contains("Broken pipe")
                    || err_str.contains("Failed to write")
                    || err_str.contains("Failed to read")
                {
                    eprintln!(
                        "[SidecarState] Sidecar communication failed ({}), attempting restart...",
                        err_str
                    );
                    match sidecar.restart_if_crashed() {
                        Ok(true) => {
                            // Retry the request after successful restart
                            sidecar
                                .send_request(method, params)
                                .map_err(|e2| {
                                    format!(
                                        "Sidecar request failed after restart: {}. Original error: {}",
                                        e2, err_str
                                    )
                                })
                        }
                        Ok(false) => {
                            // Already running but request failed
                            Err(format!("Sidecar request failed: {}", err_str))
                        }
                        Err(restart_err) => {
                            Err(format!(
                                "Sidecar crashed and restart failed: {}. Original error: {}",
                                restart_err, err_str
                            ))
                        }
                    }
                } else {
                    Err(format!("Sidecar request failed: {}", err_str))
                }
            }
        }
    }
}

impl Default for SidecarState {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tauri Commands
// =============================================================================

/// Connection result for trinity_connect
#[derive(Debug, Clone, Serialize)]
pub struct TrinityConnectionResult {
    pub success: bool,
    pub error: Option<String>,
    #[serde(rename = "sessionId")]
    pub session_id: Option<String>,
}

/// Connect to the Trinity runtime by ensuring the Python sidecar is running
#[tauri::command]
pub async fn trinity_connect(
    sidecar: State<'_, SidecarState>,
) -> Result<TrinityConnectionResult, String> {
    match sidecar.send_request("trinity.status", serde_json::json!({})) {
        Ok(result) => {
            let available = result
                .get("available")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            if available {
                Ok(TrinityConnectionResult {
                    success: true,
                    error: None,
                    session_id: Some("sidecar-session".to_string()),
                })
            } else {
                let error = result
                    .get("error")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| "Trinity runtime not available".to_string());
                Ok(TrinityConnectionResult {
                    success: false,
                    error: Some(error),
                    session_id: None,
                })
            }
        }
        Err(e) => Ok(TrinityConnectionResult {
            success: false,
            error: Some(e),
            session_id: None,
        }),
    }
}

/// Check if Trinity is available and get its status
///
/// This command queries the Python sidecar to check if the Trinity
/// runtime is loaded and available for introspection.
#[tauri::command]
pub async fn trinity_status(sidecar: State<'_, SidecarState>) -> Result<TrinityStatus, String> {
    match sidecar.send_request("trinity.status", serde_json::json!({})) {
        Ok(result) => {
            // Parse the response from Python
            let available = result
                .get("available")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let version = result
                .get("version")
                .and_then(|v| v.as_str())
                .map(String::from);
            let error = result
                .get("error")
                .and_then(|v| v.as_str())
                .map(String::from);

            Ok(TrinityStatus {
                available,
                version,
                error,
            })
        }
        Err(e) => {
            // Sidecar communication failed - Trinity is not available
            Ok(TrinityStatus {
                available: false,
                version: None,
                error: Some(e),
            })
        }
    }
}

/// Get all registered types from the Trinity Registry
///
/// Returns a list of all components, systems, resources, and events
/// that have been registered with the Trinity runtime.
///
/// # Arguments
/// * `type_filter` - Optional filter for type kind ("component", "system", "resource", "event")
#[tauri::command]
pub async fn trinity_registry_list(
    sidecar: State<'_, SidecarState>,
    type_filter: Option<String>,
) -> Result<RegistryListResponse, String> {
    let params = serde_json::json!({
        "type_filter": type_filter,
    });

    let result = sidecar.send_request("trinity.registry.list", params)?;

    // Parse the response
    let entries: Vec<RegistryEntry> = result
        .get("entries")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();

    let total = entries.len();

    Ok(RegistryListResponse { entries, total })
}

/// Query active component instances in the Trinity world
///
/// Returns information about how many instances of each component type
/// exist in the current ECS world.
///
/// # Arguments
/// * `component_filter` - Optional filter for specific component type(s)
/// * `include_data` - Whether to include sample instance data
#[tauri::command]
pub async fn trinity_instances_query(
    sidecar: State<'_, SidecarState>,
    component_filter: Option<Vec<String>>,
    include_data: Option<bool>,
) -> Result<InstancesQueryResponse, String> {
    let params = serde_json::json!({
        "component_filter": component_filter,
        "include_data": include_data.unwrap_or(false),
    });

    let result = sidecar.send_request("trinity.instances.query", params)?;

    // Parse the response
    let instances: Vec<InstanceInfo> = result
        .get("instances")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();

    let total_entities = result
        .get("total_entities")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;

    Ok(InstancesQueryResponse {
        instances,
        total_entities,
    })
}

/// Get recent events from the Trinity EventLog
///
/// Returns the most recent events that have been logged by the
/// Trinity runtime, useful for debugging and monitoring.
///
/// # Arguments
/// * `limit` - Maximum number of events to return (default: 50)
/// * `event_type_filter` - Optional filter for specific event type(s)
/// * `since` - Optional ISO 8601 timestamp to get events after
#[tauri::command]
pub async fn trinity_events_recent(
    sidecar: State<'_, SidecarState>,
    limit: Option<usize>,
    event_type_filter: Option<Vec<String>>,
    since: Option<String>,
) -> Result<EventsRecentResponse, String> {
    let params = serde_json::json!({
        "limit": limit.unwrap_or(50),
        "event_type_filter": event_type_filter,
        "since": since,
    });

    let result = sidecar.send_request("trinity.events.recent", params)?;

    // Parse the response
    let events: Vec<EventEntry> = result
        .get("events")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();

    let total_in_log = result
        .get("total_in_log")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    Ok(EventsRecentResponse {
        events,
        total_in_log,
    })
}

// =============================================================================
// Inspection Types and Commands
// =============================================================================

/// Hierarchy entry for class inheritance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchyEntry {
    /// Class name
    pub name: String,
    /// Module path
    pub module: Option<String>,
    /// Whether this is a Trinity base class
    #[serde(rename = "isTrinityBase")]
    pub is_trinity_base: bool,
}

/// Decorator entry for the decorator chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoratorEntry {
    /// Decorator name
    pub name: String,
    /// Tier level (1-5)
    pub tier: Option<i32>,
    /// Tier name
    #[serde(rename = "tierName")]
    pub tier_name: Option<String>,
    /// Whether it's a foundation decorator
    pub foundation: Option<bool>,
    /// Documentation
    pub doc: Option<String>,
    /// Arguments passed to the decorator
    pub args: Option<Value>,
}

/// Source location information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceLocation {
    /// File path
    pub file: String,
    /// Line number
    pub line: Option<i32>,
}

/// Detailed inspection result for a Trinity type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspectionResult {
    /// Whether the inspection was successful
    pub success: bool,
    /// Error message if inspection failed
    pub error: Option<String>,
    /// The type name
    pub name: Option<String>,
    /// Fully qualified name
    #[serde(rename = "qualifiedName")]
    pub qualified_name: Option<String>,
    /// Type category
    pub category: Option<String>,
    /// Module where the type is defined
    pub module: Option<String>,
    /// Documentation string
    pub doc: Option<String>,
    /// Source file information
    pub source: Option<SourceLocation>,
    /// Metaclass name
    pub metaclass: Option<String>,
    /// Component hierarchy (base classes)
    pub hierarchy: Option<Vec<HierarchyEntry>>,
    /// Decorator chain
    pub decorators: Option<Vec<DecoratorEntry>>,
    /// Field types for components
    #[serde(rename = "fieldTypes")]
    pub field_types: Option<Value>,
    /// Field defaults
    #[serde(rename = "fieldDefaults")]
    pub field_defaults: Option<Value>,
    /// Additional metadata
    pub metadata: Option<Value>,
}

/// Request payload for trinity_inspect
#[derive(Debug, Clone, Deserialize)]
pub struct InspectRequest {
    /// The type name to inspect
    #[serde(rename = "typeName")]
    pub type_name: String,
}

/// Request payload for trinity_inspector_get
#[derive(Debug, Clone, Deserialize)]
pub struct InspectorGetRequest {
    /// Type of target ("type", "instance", "decorator")
    #[serde(rename = "targetType")]
    pub target_type: String,
    /// Qualified name for type or decorator lookup
    #[serde(rename = "qualifiedName")]
    pub qualified_name: Option<String>,
    /// Instance ID (required for "instance" target_type)
    #[serde(rename = "targetId")]
    pub target_id: Option<i64>,
}

/// Inspect a Trinity type by name
///
/// Returns detailed information about a registered Trinity type including
/// hierarchy, decorators, metaclass, fields, and source location.
#[tauri::command]
pub async fn trinity_inspect(
    sidecar: State<'_, SidecarState>,
    request: InspectRequest,
) -> Result<InspectionResult, String> {
    let params = serde_json::json!({
        "type_name": request.type_name,
    });

    match sidecar.send_request("trinity.inspect", params) {
        Ok(result) => {
            // Parse the successful response
            let success = result
                .get("success")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            if !success {
                let error = result
                    .get("error")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| "Unknown error".to_string());
                return Ok(InspectionResult {
                    success: false,
                    error: Some(error),
                    name: None,
                    qualified_name: None,
                    category: None,
                    module: None,
                    doc: None,
                    source: None,
                    metaclass: None,
                    hierarchy: None,
                    decorators: None,
                    field_types: None,
                    field_defaults: None,
                    metadata: None,
                });
            }

            // Extract fields from result
            let name = result.get("name").and_then(|v| v.as_str()).map(String::from);
            let qualified_name = result
                .get("qualified_name")
                .and_then(|v| v.as_str())
                .map(String::from);
            let category = result
                .get("category")
                .and_then(|v| v.as_str())
                .map(String::from);
            let module = result.get("module").and_then(|v| v.as_str()).map(String::from);
            let doc = result.get("doc").and_then(|v| v.as_str()).map(String::from);
            let metaclass = result
                .get("metaclass")
                .and_then(|v| v.as_str())
                .map(String::from);

            // Parse source location
            let source = result.get("source").and_then(|s| {
                let file = s.get("file").and_then(|v| v.as_str()).map(String::from)?;
                let line = s.get("line").and_then(|v| v.as_i64()).map(|l| l as i32);
                Some(SourceLocation { file, line })
            });

            // Parse hierarchy - build from bases if present
            let hierarchy = build_hierarchy_from_result(&result);

            // Parse decorators - get from trinity.decorators.list if needed
            let decorators = build_decorators_from_result(&result);

            // Get field types and defaults from metadata
            let field_types = result.get("field_types").cloned();
            let field_defaults = result.get("field_defaults").cloned();
            let metadata = result.get("metadata").cloned();

            Ok(InspectionResult {
                success: true,
                error: None,
                name,
                qualified_name,
                category,
                module,
                doc,
                source,
                metaclass,
                hierarchy,
                decorators,
                field_types,
                field_defaults,
                metadata,
            })
        }
        Err(e) => Ok(InspectionResult {
            success: false,
            error: Some(e),
            name: None,
            qualified_name: None,
            category: None,
            module: None,
            doc: None,
            source: None,
            metaclass: None,
            hierarchy: None,
            decorators: None,
            field_types: None,
            field_defaults: None,
            metadata: None,
        }),
    }
}

/// Get detailed inspector information for a target
///
/// Unified API that can inspect types, instances, or decorators.
#[tauri::command]
pub async fn trinity_inspector_get(
    sidecar: State<'_, SidecarState>,
    request: InspectorGetRequest,
) -> Result<InspectionResult, String> {
    let params = serde_json::json!({
        "target_type": request.target_type,
        "qualified_name": request.qualified_name,
        "target_id": request.target_id,
    });

    match sidecar.send_request("trinity.inspector.get", params) {
        Ok(result) => {
            // Same parsing as trinity_inspect
            let success = result
                .get("success")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            if !success {
                let error = result
                    .get("error")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| "Unknown error".to_string());
                return Ok(InspectionResult {
                    success: false,
                    error: Some(error),
                    name: None,
                    qualified_name: None,
                    category: None,
                    module: None,
                    doc: None,
                    source: None,
                    metaclass: None,
                    hierarchy: None,
                    decorators: None,
                    field_types: None,
                    field_defaults: None,
                    metadata: None,
                });
            }

            let name = result.get("name").and_then(|v| v.as_str()).map(String::from);
            let qualified_name = result
                .get("qualified_name")
                .and_then(|v| v.as_str())
                .map(String::from);
            let category = result
                .get("category")
                .and_then(|v| v.as_str())
                .map(String::from);
            let module = result.get("module").and_then(|v| v.as_str()).map(String::from);
            let doc = result.get("doc").and_then(|v| v.as_str()).map(String::from);
            let metaclass = result
                .get("metaclass")
                .and_then(|v| v.as_str())
                .map(String::from);

            let source = result.get("source").and_then(|s| {
                let file = s.get("file").and_then(|v| v.as_str()).map(String::from)?;
                let line = s.get("line").and_then(|v| v.as_i64()).map(|l| l as i32);
                Some(SourceLocation { file, line })
            });

            let hierarchy = build_hierarchy_from_result(&result);
            let decorators = build_decorators_from_result(&result);
            let field_types = result.get("field_types").cloned();
            let field_defaults = result.get("field_defaults").cloned();
            let metadata = result.get("metadata").cloned();

            Ok(InspectionResult {
                success: true,
                error: None,
                name,
                qualified_name,
                category,
                module,
                doc,
                source,
                metaclass,
                hierarchy,
                decorators,
                field_types,
                field_defaults,
                metadata,
            })
        }
        Err(e) => Ok(InspectionResult {
            success: false,
            error: Some(e),
            name: None,
            qualified_name: None,
            category: None,
            module: None,
            doc: None,
            source: None,
            metaclass: None,
            hierarchy: None,
            decorators: None,
            field_types: None,
            field_defaults: None,
            metadata: None,
        }),
    }
}

/// Build hierarchy entries from inspection result
fn build_hierarchy_from_result(result: &Value) -> Option<Vec<HierarchyEntry>> {
    // Try to get hierarchy from direct field or build from bases
    if let Some(hierarchy_arr) = result.get("hierarchy").and_then(|v| v.as_array()) {
        let entries: Vec<HierarchyEntry> = hierarchy_arr
            .iter()
            .filter_map(|entry| {
                let name = entry.get("name").and_then(|v| v.as_str())?;
                let module = entry.get("module").and_then(|v| v.as_str()).map(String::from);
                let is_trinity_base = entry
                    .get("isTrinityBase")
                    .or_else(|| entry.get("is_trinity_base"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                Some(HierarchyEntry {
                    name: name.to_string(),
                    module,
                    is_trinity_base,
                })
            })
            .collect();
        if !entries.is_empty() {
            return Some(entries);
        }
    }

    // Fall back to building from bases array
    if let Some(bases_arr) = result.get("bases").and_then(|v| v.as_array()) {
        let trinity_bases = ["Component", "System", "Resource", "Event", "Entity", "World"];
        let entries: Vec<HierarchyEntry> = bases_arr
            .iter()
            .filter_map(|base| {
                let name = base.as_str().map(String::from).or_else(|| {
                    base.get("name").and_then(|v| v.as_str()).map(String::from)
                })?;
                let module = base.get("module").and_then(|v| v.as_str()).map(String::from);
                let is_trinity_base = trinity_bases.contains(&name.as_str());
                Some(HierarchyEntry {
                    name,
                    module,
                    is_trinity_base,
                })
            })
            .collect();
        if !entries.is_empty() {
            return Some(entries);
        }
    }

    None
}

/// Build decorator entries from inspection result
fn build_decorators_from_result(result: &Value) -> Option<Vec<DecoratorEntry>> {
    // Try to get decorators array
    if let Some(decorators_arr) = result.get("decorators").and_then(|v| v.as_array()) {
        let entries: Vec<DecoratorEntry> = decorators_arr
            .iter()
            .filter_map(|dec| {
                let name = dec
                    .as_str()
                    .map(String::from)
                    .or_else(|| dec.get("name").and_then(|v| v.as_str()).map(String::from))?;
                let tier = dec.get("tier").and_then(|v| v.as_i64()).map(|t| t as i32);
                let tier_name = dec
                    .get("tier_name")
                    .or_else(|| dec.get("tierName"))
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let foundation = dec.get("foundation").and_then(|v| v.as_bool());
                let doc = dec.get("doc").and_then(|v| v.as_str()).map(String::from);
                let args = dec.get("args").cloned();

                Some(DecoratorEntry {
                    name,
                    tier,
                    tier_name,
                    foundation,
                    doc,
                    args,
                })
            })
            .collect();
        if !entries.is_empty() {
            return Some(entries);
        }
    }

    // If no decorators found but we have category, create a synthetic decorator
    if let Some(category) = result.get("category").and_then(|v| v.as_str()) {
        let decorator_name = category.to_string();
        return Some(vec![DecoratorEntry {
            name: decorator_name,
            tier: Some(1),
            tier_name: Some("FOUNDATION".to_string()),
            foundation: Some(true),
            doc: None,
            args: None,
        }]);
    }

    None
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trinity_status_serialization() {
        let status = TrinityStatus {
            available: true,
            version: Some("0.1.0".to_string()),
            error: None,
        };

        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("\"available\":true"));
        assert!(json.contains("\"version\":\"0.1.0\""));

        let deserialized: TrinityStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.available, true);
        assert_eq!(deserialized.version, Some("0.1.0".to_string()));
    }

    #[test]
    fn test_registry_entry_serialization() {
        let entry = RegistryEntry {
            name: "Position".to_string(),
            type_kind: "component".to_string(),
            module: "game.components".to_string(),
            metadata: Some(serde_json::json!({"fields": ["x", "y", "z"]})),
        };

        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("\"name\":\"Position\""));
        assert!(json.contains("\"type_kind\":\"component\""));

        let deserialized: RegistryEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "Position");
        assert_eq!(deserialized.type_kind, "component");
    }

    #[test]
    fn test_instance_info_serialization() {
        let info = InstanceInfo {
            component: "Velocity".to_string(),
            count: 42,
            data: Some(serde_json::json!({"entities": [1, 2, 3]})),
        };

        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("\"component\":\"Velocity\""));
        assert!(json.contains("\"count\":42"));

        let deserialized: InstanceInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.component, "Velocity");
        assert_eq!(deserialized.count, 42);
    }

    #[test]
    fn test_event_entry_serialization() {
        let event = EventEntry {
            event_type: "EntitySpawned".to_string(),
            timestamp: "2024-01-15T10:30:00Z".to_string(),
            data: serde_json::json!({"entity_id": 123}),
            event_id: Some("evt-001".to_string()),
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"event_type\":\"EntitySpawned\""));
        assert!(json.contains("\"timestamp\":\"2024-01-15T10:30:00Z\""));

        let deserialized: EventEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.event_type, "EntitySpawned");
        assert_eq!(deserialized.event_id, Some("evt-001".to_string()));
    }

    #[test]
    fn test_sidecar_state_default() {
        let state = SidecarState::default();
        let guard = state.0.lock().unwrap();
        assert!(guard.is_none());
    }
}
