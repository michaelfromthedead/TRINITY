//! T-TL-1.1: Three-Channel Bridge Protocol Schema
//!
//! Formal schema for the Type/Data/Command three-channel bridge protocol
//! connecting the Rust sidecar, the Python Trinity/Foundation runtime, and
//! the TypeScript frontend.
//!
//! # Protocol Architecture
//!
//! The bridge defines four logical channel groups, each with a distinct
//! performance and semantic profile:
//!
//! | Channel    | Phase           | Hot Path? | Endpoints |
//! |------------|-----------------|-----------|-----------|
//! | Type       | Import/definition | No       | 5         |
//! | Data       | Per-frame field  | Yes      | 5         |
//! | Command    | Structural world | No       | 6         |
//! | System     | Introspection    | No       | 6         |
//!
//! # Transport
//!
//! All endpoints are transported over JSON-RPC 2.0 (line-delimited JSON
//! over stdin/stdout for the Python sidecar, or Tauri `invoke()` for the
//! frontend).  Every type in this module derives `Serialize + Deserialize`
//! so it can cross any of those boundaries.
//!
//! # Naming Convention
//!
//! Rust fields use `snake_case` internally.  Frontend-facing fields that
//! are part of a public JSON contract use `#[serde(rename = "camelCase")]`.
//! Fields that are only consumed internally by Rust use plain snake_case
//! with no rename attribute.
//!
//! # Schema Version
//!
//! This module is the single source of truth for the bridge protocol.
//! Every endpoint MUST be added here first, then mirrored in:
//!   - TypeScript: flowforge/packages/core/bridge/types.ts (type exports)
//!   - Python:     tests/integration/_omega_mock.py (mock implementation)

use serde::{Deserialize, Serialize};
use serde_json::Value;

// =============================================================================
// TYPE CHANNEL -- Component/System/Resource/Event type registration
// =============================================================================

/// Channel identifier for routing (used in JSON-RPC "method" prefix).
pub const TYPE_CHANNEL_PREFIX: &str = "type";

/// Kind of type in the registry (extensible for forward compatibility).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TypeKind {
    Component,
    System,
    Resource,
    Event,
    /// Forward-compatible catch-all for future type kinds.
    #[serde(untagged)]
    Custom(String),
}

impl TypeKind {
    /// Return the string representation of this kind.
    pub fn as_str(&self) -> &str {
        match self {
            TypeKind::Component => "component",
            TypeKind::System => "system",
            TypeKind::Resource => "resource",
            TypeKind::Event => "event",
            TypeKind::Custom(s) => s.as_str(),
        }
    }
}

/// Single field description in a type layout.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FieldDescriptor {
    /// Field name (e.g. "x", "health", "name").
    pub name: String,
    /// Rust type code (e.g. "f32", "i32", "u8", "string").
    #[serde(rename = "typeCode")]
    pub type_code: String,
    /// Byte offset from the start of the component's storage block.
    pub offset: u32,
    /// Size in bytes (0 for variable-length types like string).
    #[serde(default)]
    pub size: u32,
}

/// T-TYPE-01: Register a component/system/resource/event type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeRegisterRequest {
    /// Unique component ID (auto-assigned by ComponentMeta).
    #[serde(rename = "componentId")]
    pub component_id: u32,
    /// Fully qualified type name (e.g. "game.components.Position").
    pub name: String,
    /// Kind of type.
    #[serde(rename = "typeKind")]
    pub type_kind: TypeKind,
    /// Total storage size in bytes for fixed-layout types.
    #[serde(rename = "totalSize")]
    pub total_size: u32,
    /// Ordered list of field descriptors.
    pub fields: Vec<FieldDescriptor>,
    /// Module path where the type is defined.
    #[serde(default)]
    pub module: String,
    /// Optional metadata (JSON blob for extensibility).
    #[serde(default)]
    pub metadata: Option<Value>,
}

/// Response from type_register.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeRegisterResponse {
    pub success: bool,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(rename = "componentId")]
    pub component_id: u32,
}

/// T-TYPE-02: List all registered types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeListRequest {
    /// Optional filter by type kind.
    #[serde(rename = "typeFilter")]
    pub type_filter: Option<String>,
    /// Optional pagination offset.
    #[serde(default)]
    pub offset: u64,
    /// Optional pagination limit (default: all).
    #[serde(default)]
    pub limit: Option<u64>,
}

/// Single entry in the registry listing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TypeRegistryEntry {
    #[serde(rename = "componentId")]
    pub component_id: u32,
    pub name: String,
    #[serde(rename = "typeKind")]
    pub type_kind: String,
    pub module: String,
    #[serde(rename = "totalSize")]
    pub total_size: u32,
    pub fields: Vec<FieldDescriptor>,
    #[serde(default)]
    pub metadata: Option<Value>,
}

/// Response from type_list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeListResponse {
    pub entries: Vec<TypeRegistryEntry>,
    pub total: u64,
    #[serde(default)]
    pub offset: u64,
}

/// T-TYPE-03: Get a specific type by component ID.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeGetRequest {
    #[serde(rename = "componentId")]
    pub component_id: u32,
}

/// T-TYPE-04: Remove a type registration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeRemoveRequest {
    #[serde(rename = "componentId")]
    pub component_id: u32,
}

/// T-TYPE-05: Count registered types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeCountRequest {
    /// Optional filter by type kind.
    #[serde(rename = "typeFilter")]
    pub type_filter: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeCountResponse {
    pub count: u64,
}

// =============================================================================
// DATA CHANNEL -- Per-frame field reads and writes
// =============================================================================
// The Data channel is the hottest path in the bridge. Every per-frame
// component field access routes through one of these endpoints.
//
// Performance targets (T-CORE-5.6):
//   - 1M field reads in < 100 ms (Rust) / < 750 ms (Python mock)
//   - 1M field writes in < 200 ms (Rust) / < 750 ms (Python mock)
//   - GIL release allows concurrent read/write without corruption
//
// Error semantics:
//   - Read of an unwritten key raises RuntimeError (not Option/None)
//   - Write to any valid key always succeeds (creates or overwrites)
//   - Delete is idempotent: deleting a non-existent key does NOT raise

/// Channel identifier for routing.
pub const DATA_CHANNEL_PREFIX: &str = "data";

/// Key that identifies a single field value in the component store.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct FieldKey {
    #[serde(rename = "entityId")]
    pub entity_id: u64,
    #[serde(rename = "componentId")]
    pub component_id: u32,
    pub offset: u32,
}

/// T-DATA-01: Read a single field value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentReadRequest {
    pub key: FieldKey,
    /// Expected Python type for deserialization hint.
    #[serde(rename = "fieldType", default)]
    pub field_type: Option<String>,
}

/// T-DATA-02: Write a single field value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentWriteRequest {
    pub key: FieldKey,
    /// Value to write (must be JSON-serializable).
    pub value: Value,
}

/// T-DATA-03: Delete a single field value (idempotent).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentDeleteRequest {
    pub key: FieldKey,
}

/// T-DATA-04: Batch read multiple fields in one call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentBatchReadRequest {
    pub keys: Vec<FieldKey>,
}

/// A single read result in a batch response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldReadResult {
    pub key: FieldKey,
    #[serde(default)]
    pub value: Option<Value>,
    #[serde(default)]
    pub error: Option<String>,
}

/// T-DATA-04 response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentBatchReadResponse {
    pub results: Vec<FieldReadResult>,
}

/// T-DATA-05: Batch write multiple fields in one call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentBatchWriteRequest {
    /// Pairs of (key, value) to write.
    pub writes: Vec<ComponentWriteRequest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentBatchWriteResponse {
    pub written: u64,
}

// =============================================================================
// COMMAND CHANNEL -- Structural world operations
// =============================================================================

/// Channel identifier for routing.
pub const COMMAND_CHANNEL_PREFIX: &str = "command";

/// A single field-initializer pair for component data during spawn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldInit {
    pub offset: u32,
    pub value: Value,
}

/// Component data block for spawning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentBlock {
    #[serde(rename = "componentId")]
    pub component_id: u32,
    /// Field initializers (offset + value pairs).
    pub fields: Vec<FieldInit>,
}

/// T-CMD-01: Create a new world handle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldCreateRequest {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldCreateResponse {
    #[serde(rename = "worldHandle")]
    pub world_handle: u32,
}

/// T-CMD-02: Spawn an entity with initial component data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldSpawnRequest {
    /// World handle (0 for single-world mode).
    #[serde(rename = "worldHandle")]
    pub world_handle: u32,
    /// Component blocks to attach to the new entity.
    pub components: Vec<ComponentBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldSpawnResponse {
    #[serde(rename = "entityId")]
    pub entity_id: u64,
}

/// T-CMD-03: Despawn an entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldDespawnRequest {
    #[serde(rename = "worldHandle")]
    pub world_handle: u32,
    #[serde(rename = "entityId")]
    pub entity_id: u64,
}

/// T-CMD-04: Query entities by component set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldQueryRequest {
    #[serde(rename = "worldHandle")]
    pub world_handle: u32,
    /// Return entities that have ALL of these component IDs.
    #[serde(rename = "componentIds")]
    pub component_ids: Vec<u32>,
    /// Optional maximum results.
    #[serde(default)]
    pub limit: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldQueryResponse {
    #[serde(rename = "entityIds")]
    pub entity_ids: Vec<u64>,
    pub total: u64,
}

/// T-CMD-05: Reset world state (clear all entities and data).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldResetRequest {
    #[serde(rename = "worldHandle")]
    pub world_handle: u32,
}

/// T-CMD-06: Get world statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldStatsRequest {
    #[serde(rename = "worldHandle")]
    pub world_handle: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldStatsResponse {
    #[serde(rename = "entityCount")]
    pub entity_count: u64,
    #[serde(rename = "archetypeCount")]
    pub archetype_count: u64,
    #[serde(rename = "componentTypeCount")]
    pub component_type_count: u64,
    #[serde(rename = "totalFields")]
    pub total_fields: u64,
}

// =============================================================================
// SYSTEM CHANNEL -- Introspection, management, and diagnostics
// =============================================================================

/// Channel identifier for routing.
pub const SYSTEM_CHANNEL_PREFIX: &str = "system";

/// T-SYS-01: Trinity connection status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrinityConnectRequest {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrinityConnectResponse {
    pub success: bool,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(rename = "sessionId")]
    pub session_id: Option<String>,
}

/// T-SYS-02: Runtime status check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrinityStatusRequest {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrinityStatusResponse {
    pub available: bool,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(rename = "uptimeMs", default)]
    pub uptime_ms: Option<u64>,
}

/// T-SYS-03: Inspect a type in detail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrinityInspectRequest {
    #[serde(rename = "typeName")]
    pub type_name: String,
}

/// Class hierarchy entry for inspection results.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HierarchyEntry {
    pub name: String,
    #[serde(default)]
    pub module: Option<String>,
    #[serde(rename = "isTrinityBase", default)]
    pub is_trinity_base: bool,
}

/// Decorator chain entry for inspection results.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DecoratorEntry {
    pub name: String,
    #[serde(default)]
    pub tier: Option<i32>,
    #[serde(rename = "tierName")]
    pub tier_name: Option<String>,
    #[serde(default)]
    pub foundation: Option<bool>,
    #[serde(default)]
    pub doc: Option<String>,
    #[serde(default)]
    pub args: Option<Value>,
}

/// Source location for inspection results.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SourceLocation {
    pub file: String,
    #[serde(default)]
    pub line: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrinityInspectResponse {
    pub success: bool,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(rename = "qualifiedName")]
    pub qualified_name: Option<String>,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub module: Option<String>,
    #[serde(default)]
    pub doc: Option<String>,
    #[serde(default)]
    pub source: Option<SourceLocation>,
    #[serde(default)]
    pub metaclass: Option<String>,
    #[serde(default)]
    pub hierarchy: Option<Vec<HierarchyEntry>>,
    #[serde(default)]
    pub decorators: Option<Vec<DecoratorEntry>>,
    #[serde(rename = "fieldTypes")]
    pub field_types: Option<Value>,
    #[serde(rename = "fieldDefaults")]
    pub field_defaults: Option<Value>,
    #[serde(default)]
    pub metadata: Option<Value>,
}

/// T-SYS-04: Get inspector target details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspectorGetRequest {
    /// Target type: "type", "instance", or "decorator".
    #[serde(rename = "targetType")]
    pub target_type: String,
    /// Qualified name for type or decorator lookup.
    #[serde(rename = "qualifiedName")]
    pub qualified_name: Option<String>,
    /// Instance ID for instance target.
    #[serde(rename = "targetId")]
    pub target_id: Option<i64>,
}

/// T-SYS-05: Get recent events from the event log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventsRecentRequest {
    /// Max events to return.
    #[serde(default)]
    pub limit: u64,
    /// Optional filter by event type(s).
    #[serde(rename = "eventTypeFilter", default)]
    pub event_type_filter: Option<Vec<String>>,
    /// ISO 8601 timestamp filter (events after this time).
    #[serde(default)]
    pub since: Option<String>,
}

/// Single event entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EventEntry {
    #[serde(rename = "eventType")]
    pub event_type: String,
    pub timestamp: String,
    pub data: Value,
    #[serde(rename = "eventId", default)]
    pub event_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventsRecentResponse {
    pub events: Vec<EventEntry>,
    #[serde(rename = "totalInLog")]
    pub total_in_log: u64,
}

/// T-SYS-06: Compute content-addressed checksum of the entire store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChecksumRequest {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChecksumResponse {
    /// SHA-256 hex digest of the store contents.
    pub checksum: String,
    /// Number of entities included.
    #[serde(rename = "entityCount")]
    pub entity_count: u64,
}

// =============================================================================
// PROTOCOL METADATA -- Route table for dispatching endpoints
// =============================================================================

/// Full routing table mapping method names to their channel and type.
pub const METHOD_TABLE: &[(&str, &str, &str)] = &[
    // TYPE CHANNEL (5 endpoints)
    ("type.register",        "TypeRegisterRequest",        "TypeRegisterResponse"),
    ("type.list",            "TypeListRequest",            "TypeListResponse"),
    ("type.get",             "TypeGetRequest",             "TypeRegistryEntry"),
    ("type.remove",          "TypeRemoveRequest",          "TypeRegisterResponse"),
    ("type.count",           "TypeCountRequest",           "TypeCountResponse"),
    // DATA CHANNEL (5 endpoints)
    ("data.read",            "ComponentReadRequest",       "Value"),
    ("data.write",           "ComponentWriteRequest",      "Value"),
    ("data.delete",          "ComponentDeleteRequest",     "Value"),
    ("data.batch_read",      "ComponentBatchReadRequest",  "ComponentBatchReadResponse"),
    ("data.batch_write",     "ComponentBatchWriteRequest", "ComponentBatchWriteResponse"),
    // COMMAND CHANNEL (6 endpoints)
    ("command.create",       "WorldCreateRequest",         "WorldCreateResponse"),
    ("command.spawn",        "WorldSpawnRequest",          "WorldSpawnResponse"),
    ("command.despawn",      "WorldDespawnRequest",        "Value"),
    ("command.query",        "WorldQueryRequest",          "WorldQueryResponse"),
    ("command.reset",        "WorldResetRequest",          "Value"),
    ("command.stats",        "WorldStatsRequest",          "WorldStatsResponse"),
    // SYSTEM CHANNEL (6 endpoints)
    ("system.connect",       "TrinityConnectRequest",      "TrinityConnectResponse"),
    ("system.status",        "TrinityStatusRequest",       "TrinityStatusResponse"),
    ("system.inspect",       "TrinityInspectRequest",      "TrinityInspectResponse"),
    ("system.inspector_get", "InspectorGetRequest",        "TrinityInspectResponse"),
    ("system.events_recent", "EventsRecentRequest",        "EventsRecentResponse"),
    ("system.checksum",      "ChecksumRequest",            "ChecksumResponse"),
];

/// Count of endpoints per channel.
pub const TYPE_CHANNEL_ENDPOINTS: usize   = 5;
pub const DATA_CHANNEL_ENDPOINTS: usize   = 5;
pub const COMMAND_CHANNEL_ENDPOINTS: usize = 6;
pub const SYSTEM_CHANNEL_ENDPOINTS: usize = 6;

/// Total endpoints across all four channels.
pub const TOTAL_ENDPOINTS: usize =
    TYPE_CHANNEL_ENDPOINTS
    + DATA_CHANNEL_ENDPOINTS
    + COMMAND_CHANNEL_ENDPOINTS
    + SYSTEM_CHANNEL_ENDPOINTS;

/// Determine the channel group from a method name.
pub fn channel_for_method(method: &str) -> Option<&'static str> {
    // Use format!("{}.", prefix) to avoid false positives:
    //   "type.register" -> type  (correct)
    //   "typewriter"    -> None  (false-positive prevented by trailing dot)
    if method.starts_with(&format!("{}.", TYPE_CHANNEL_PREFIX)) {
        Some("type")
    } else if method.starts_with(&format!("{}.", DATA_CHANNEL_PREFIX)) {
        Some("data")
    } else if method.starts_with(&format!("{}.", COMMAND_CHANNEL_PREFIX)) {
        Some("command")
    } else if method.starts_with(&format!("{}.", SYSTEM_CHANNEL_PREFIX)) {
        Some("system")
    } else {
        None
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // TYPE CHANNEL serialization tests
    // -------------------------------------------------------------------------

    #[test]
    fn type_register_request_round_trip() {
        let req = TypeRegisterRequest {
            component_id: 1,
            name: "game.components.Position".into(),
            type_kind: TypeKind::Component,
            total_size: 12,
            fields: vec![
                FieldDescriptor { name: "x".into(), type_code: "f32".into(), offset: 0, size: 4 },
                FieldDescriptor { name: "y".into(), type_code: "f32".into(), offset: 4, size: 4 },
                FieldDescriptor { name: "z".into(), type_code: "f32".into(), offset: 8, size: 4 },
            ],
            module: "game.components".into(),
            metadata: None,
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"componentId\":1"));
        assert!(json.contains("\"typeCode\":\"f32\""));

        let deserialized: TypeRegisterRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.component_id, 1);
        assert_eq!(deserialized.fields.len(), 3);
        assert_eq!(deserialized.fields[1].name, "y");
    }

    #[test]
    fn type_kind_round_trip() {
        for kind in &[
            TypeKind::Component,
            TypeKind::System,
            TypeKind::Resource,
            TypeKind::Event,
        ] {
            let json = serde_json::to_string(kind).unwrap();
            let deserialized: TypeKind = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized, *kind);
        }
    }

    #[test]
    fn type_kind_custom_round_trip() {
        let custom = TypeKind::Custom("state".into());
        let json = serde_json::to_string(&custom).unwrap();
        let deserialized: TypeKind = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, custom);
    }

    #[test]
    fn field_descriptor_round_trip() {
        let fd = FieldDescriptor {
            name: "current_hp".into(),
            type_code: "f32".into(),
            offset: 0,
            size: 4,
        };

        let json = serde_json::to_string(&fd).unwrap();
        assert!(json.contains("\"typeCode\":\"f32\""));
        assert!(json.contains("\"offset\":0"));

        let deserialized: FieldDescriptor = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "current_hp");
    }

    #[test]
    fn type_list_request_round_trip() {
        let req = TypeListRequest {
            type_filter: Some("component".into()),
            offset: 0,
            limit: Some(50),
        };

        let json = serde_json::to_string(&req).unwrap();
        let deserialized: TypeListRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.type_filter, Some("component".into()));
        assert_eq!(deserialized.limit, Some(50));
    }

    #[test]
    fn type_list_response_round_trip() {
        let resp = TypeListResponse {
            entries: vec![
                TypeRegistryEntry {
                    component_id: 1,
                    name: "Pos".into(),
                    type_kind: "component".into(),
                    module: "ecs".into(),
                    total_size: 12,
                    fields: vec![],
                    metadata: None,
                },
            ],
            total: 1,
            offset: 0,
        };

        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: TypeListResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.entries.len(), 1);
        assert_eq!(deserialized.total, 1);
    }

    #[test]
    fn type_register_response_round_trip() {
        let resp = TypeRegisterResponse {
            success: true,
            error: None,
            component_id: 1,
        };

        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: TypeRegisterResponse = serde_json::from_str(&json).unwrap();
        assert!(deserialized.success);
        assert_eq!(deserialized.component_id, 1);
    }

    // -------------------------------------------------------------------------
    // DATA CHANNEL serialization tests
    // -------------------------------------------------------------------------

    #[test]
    fn field_key_round_trip() {
        let key = FieldKey { entity_id: 42, component_id: 1, offset: 0 };
        let json = serde_json::to_string(&key).unwrap();
        assert!(json.contains("\"entityId\":42"));

        let deserialized: FieldKey = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.entity_id, 42);
        assert_eq!(deserialized.offset, 0);
    }

    #[test]
    fn component_read_write_round_trip() {
        let write = ComponentWriteRequest {
            key: FieldKey { entity_id: 1, component_id: 1, offset: 0 },
            value: serde_json::json!(42),
        };

        let json = serde_json::to_string(&write).unwrap();
        let deserialized: ComponentWriteRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.value, 42);

        let read = ComponentReadRequest {
            key: FieldKey { entity_id: 1, component_id: 1, offset: 0 },
            field_type: Some("i32".into()),
        };

        let json = serde_json::to_string(&read).unwrap();
        let deserialized: ComponentReadRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.field_type, Some("i32".into()));
    }

    #[test]
    fn component_delete_request() {
        let req = ComponentDeleteRequest {
            key: FieldKey { entity_id: 1, component_id: 1, offset: 8 },
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"entityId\":1"));
        assert!(json.contains("\"offset\":8"));
    }

    #[test]
    fn component_batch_read_round_trip() {
        let req = ComponentBatchReadRequest {
            keys: vec![
                FieldKey { entity_id: 1, component_id: 1, offset: 0 },
                FieldKey { entity_id: 1, component_id: 1, offset: 4 },
            ],
        };

        let json = serde_json::to_string(&req).unwrap();
        let deserialized: ComponentBatchReadRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.keys.len(), 2);
    }

    #[test]
    fn component_batch_write_round_trip() {
        let req = ComponentBatchWriteRequest {
            writes: vec![
                ComponentWriteRequest {
                    key: FieldKey { entity_id: 1, component_id: 1, offset: 0 },
                    value: serde_json::json!(10.0),
                },
                ComponentWriteRequest {
                    key: FieldKey { entity_id: 1, component_id: 1, offset: 4 },
                    value: serde_json::json!(20.0),
                },
            ],
        };

        let json = serde_json::to_string(&req).unwrap();
        let deserialized: ComponentBatchWriteRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.writes.len(), 2);
    }

    #[test]
    fn field_read_result_round_trip() {
        let result = FieldReadResult {
            key: FieldKey { entity_id: 1, component_id: 1, offset: 0 },
            value: Some(serde_json::json!(3.14)),
            error: None,
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: FieldReadResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.value, Some(serde_json::json!(3.14)));
        assert_eq!(deserialized.error, None);
    }

    // -------------------------------------------------------------------------
    // COMMAND CHANNEL serialization tests
    // -------------------------------------------------------------------------

    #[test]
    fn world_spawn_round_trip() {
        let req = WorldSpawnRequest {
            world_handle: 0,
            components: vec![
                ComponentBlock {
                    component_id: 1,
                    fields: vec![
                        FieldInit { offset: 0, value: serde_json::json!(1.0) },
                        FieldInit { offset: 4, value: serde_json::json!(2.0) },
                    ],
                },
            ],
        };

        let json = serde_json::to_string(&req).unwrap();
        let deserialized: WorldSpawnRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.world_handle, 0);
        assert_eq!(deserialized.components.len(), 1);
        assert_eq!(deserialized.components[0].fields.len(), 2);
    }

    #[test]
    fn world_spawn_response_round_trip() {
        let resp = WorldSpawnResponse { entity_id: 42 };
        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: WorldSpawnResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.entity_id, 42);
    }

    #[test]
    fn world_query_round_trip() {
        let req = WorldQueryRequest {
            world_handle: 0,
            component_ids: vec![1, 2, 3],
            limit: Some(100),
        };

        let json = serde_json::to_string(&req).unwrap();
        let deserialized: WorldQueryRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.component_ids.len(), 3);
        assert_eq!(deserialized.limit, Some(100));
    }

    #[test]
    fn world_query_response_round_trip() {
        let resp = WorldQueryResponse {
            entity_ids: vec![1, 2, 3],
            total: 3,
        };

        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: WorldQueryResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.entity_ids.len(), 3);
        assert_eq!(deserialized.total, 3);
    }

    #[test]
    fn world_despawn_request() {
        let req = WorldDespawnRequest { world_handle: 0, entity_id: 99 };
        let json = serde_json::to_string(&req).unwrap();
        let deserialized: WorldDespawnRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.entity_id, 99);
        assert_eq!(deserialized.world_handle, 0);
    }

    #[test]
    fn world_stats_response_round_trip() {
        let resp = WorldStatsResponse {
            entity_count: 1000,
            archetype_count: 25,
            component_type_count: 10,
            total_fields: 5000,
        };

        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: WorldStatsResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.entity_count, 1000);
        assert_eq!(deserialized.archetype_count, 25);
    }

    // -------------------------------------------------------------------------
    // SYSTEM CHANNEL serialization tests
    // -------------------------------------------------------------------------

    #[test]
    fn trinity_connect_response_round_trip() {
        let resp = TrinityConnectResponse {
            success: true,
            error: None,
            session_id: Some("sess-001".into()),
        };

        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"sessionId\":\"sess-001\""));

        let deserialized: TrinityConnectResponse = serde_json::from_str(&json).unwrap();
        assert!(deserialized.success);
        assert_eq!(deserialized.session_id, Some("sess-001".into()));
    }

    #[test]
    fn trinity_status_response_round_trip() {
        let resp = TrinityStatusResponse {
            available: true,
            version: Some("0.1.0".into()),
            error: None,
            uptime_ms: Some(12345),
        };

        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: TrinityStatusResponse = serde_json::from_str(&json).unwrap();
        assert!(deserialized.available);
        assert_eq!(deserialized.version, Some("0.1.0".into()));
        assert_eq!(deserialized.uptime_ms, Some(12345));
    }

    #[test]
    fn trinity_inspect_response_round_trip() {
        let resp = TrinityInspectResponse {
            success: true,
            error: None,
            name: Some("Health".into()),
            qualified_name: Some("game.components.Health".into()),
            category: Some("component".into()),
            module: Some("game.components".into()),
            doc: Some("Player health".into()),
            source: Some(SourceLocation {
                file: "health.py".into(),
                line: Some(10),
            }),
            metaclass: Some("ComponentMeta".into()),
            hierarchy: Some(vec![
                HierarchyEntry {
                    name: "Component".into(),
                    module: Some("trinity.base".into()),
                    is_trinity_base: true,
                },
            ]),
            decorators: Some(vec![
                DecoratorEntry {
                    name: "component".into(),
                    tier: Some(1),
                    tier_name: Some("FOUNDATION".into()),
                    foundation: Some(true),
                    doc: None,
                    args: None,
                },
            ]),
            field_types: Some(serde_json::json!({"current": "f32", "max_hp": "f32"})),
            field_defaults: Some(serde_json::json!({"current": 100.0, "max_hp": 100.0})),
            metadata: Some(serde_json::json!({"component_id": 3})),
        };

        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: TrinityInspectResponse = serde_json::from_str(&json).unwrap();
        assert!(deserialized.success);
        assert_eq!(deserialized.name.as_deref(), Some("Health"));
        assert_eq!(deserialized.hierarchy.as_ref().unwrap().len(), 1);
        assert!(deserialized.hierarchy.as_ref().unwrap()[0].is_trinity_base);
    }

    #[test]
    fn events_recent_round_trip() {
        let req = EventsRecentRequest {
            limit: 50,
            event_type_filter: Some(vec!["Spawn".into(), "Despawn".into()]),
            since: Some("2026-05-21T00:00:00Z".into()),
        };

        let json = serde_json::to_string(&req).unwrap();
        let deserialized: EventsRecentRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.limit, 50);
        assert_eq!(deserialized.event_type_filter.as_ref().unwrap().len(), 2);

        let resp = EventsRecentResponse {
            events: vec![
                EventEntry {
                    event_type: "Spawn".into(),
                    timestamp: "2026-05-21T12:00:00Z".into(),
                    data: serde_json::json!({"entity_id": 1}),
                    event_id: Some("evt-001".into()),
                },
            ],
            total_in_log: 1,
        };

        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: EventsRecentResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.events.len(), 1);
        assert_eq!(deserialized.total_in_log, 1);
    }

    #[test]
    fn checksum_response_round_trip() {
        let resp = ChecksumResponse {
            checksum: "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890".into(),
            entity_count: 42,
        };

        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: ChecksumResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.checksum.len(), 64);
        assert_eq!(deserialized.entity_count, 42);
    }

    // -------------------------------------------------------------------------
    // PROTOCOL METADATA tests
    // -------------------------------------------------------------------------

    #[test]
    fn method_table_has_all_endpoints() {
        let type_eps: Vec<_> = METHOD_TABLE.iter().filter(|(m, _, _)| m.starts_with("type.")).collect();
        let data_eps: Vec<_> = METHOD_TABLE.iter().filter(|(m, _, _)| m.starts_with("data.")).collect();
        let cmd_eps: Vec<_> = METHOD_TABLE.iter().filter(|(m, _, _)| m.starts_with("command.")).collect();
        let sys_eps: Vec<_> = METHOD_TABLE.iter().filter(|(m, _, _)| m.starts_with("system.")).collect();

        assert_eq!(type_eps.len(), TYPE_CHANNEL_ENDPOINTS);
        assert_eq!(data_eps.len(), DATA_CHANNEL_ENDPOINTS);
        assert_eq!(cmd_eps.len(), COMMAND_CHANNEL_ENDPOINTS);
        assert_eq!(sys_eps.len(), SYSTEM_CHANNEL_ENDPOINTS);

        assert_eq!(METHOD_TABLE.len(), TOTAL_ENDPOINTS);
    }

    #[test]
    fn channel_for_method_routes_correctly() {
        assert_eq!(channel_for_method("type.register"), Some("type"));
        assert_eq!(channel_for_method("data.read"), Some("data"));
        assert_eq!(channel_for_method("command.spawn"), Some("command"));
        assert_eq!(channel_for_method("system.status"), Some("system"));
        assert_eq!(channel_for_method("unknown.method"), None);
    }

    #[test]
    fn total_endpoints_22() {
        assert_eq!(TOTAL_ENDPOINTS, 22);
    }

    #[test]
    fn all_method_table_entries_have_valid_channels() {
        for (method, _req, _resp) in METHOD_TABLE {
            assert!(
                channel_for_method(method).is_some(),
                "Method '{}' has no valid channel route",
                method
            );
        }
    }

    #[test]
    fn method_names_use_dot_notation() {
        for (method, _req, _resp) in METHOD_TABLE {
            let parts: Vec<&str> = method.split('.').collect();
            assert_eq!(parts.len(), 2, "Method '{}' should have exactly 2 parts (channel.endpoint)", method);
            assert!(!parts[0].is_empty(), "Method '{}' channel part must be non-empty", method);
            assert!(!parts[1].is_empty(), "Method '{}' endpoint part must be non-empty", method);
        }
    }

    // -------------------------------------------------------------------------
    // EDGE CASE tests
    // -------------------------------------------------------------------------

    #[test]
    fn type_register_with_zero_fields() {
        let req = TypeRegisterRequest {
            component_id: 99,
            name: "Tag".into(),
            type_kind: TypeKind::Component,
            total_size: 0,
            fields: vec![],
            module: "tags".into(),
            metadata: None,
        };

        let json = serde_json::to_string(&req).unwrap();
        let deserialized: TypeRegisterRequest = serde_json::from_str(&json).unwrap();
        assert!(deserialized.fields.is_empty());
        assert_eq!(deserialized.total_size, 0);
    }

    #[test]
    fn type_register_with_unicode() {
        let req = TypeRegisterRequest {
            component_id: 1,
            name: "PlayerComponent".into(),
            type_kind: TypeKind::Component,
            total_size: 4,
            fields: vec![
                FieldDescriptor {
                    name: "玩家名称".into(),
                    type_code: "string".into(),
                    offset: 0,
                    size: 0,
                },
            ],
            module: "game.components".into(),
            metadata: Some(serde_json::json!({"description": "玩家组件"})),
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("玩家名称"));
        assert!(json.contains("玩家组件"));

        let deserialized: TypeRegisterRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.fields[0].name, "玩家名称");
    }

    #[test]
    fn inspect_response_minimal() {
        let json = r#"{"success":true}"#;
        let resp: TrinityInspectResponse = serde_json::from_str(json).unwrap();
        assert!(resp.success);
        assert_eq!(resp.error, None);
        assert_eq!(resp.name, None);
        assert_eq!(resp.hierarchy, None);
    }

    #[test]
    fn large_type_list_response() {
        let mut entries = Vec::with_capacity(10_000);
        for i in 0..10_000 {
            entries.push(TypeRegistryEntry {
                component_id: i as u32,
                name: format!("StressComp{}", i),
                type_kind: match i % 4 {
                    0 => "component".into(),
                    1 => "system".into(),
                    2 => "resource".into(),
                    _ => "event".into(),
                },
                module: format!("stress.test.comp{}", i % 100),
                total_size: 4,
                fields: vec![FieldDescriptor {
                    name: "x".into(),
                    type_code: "i32".into(),
                    offset: 0,
                    size: 4,
                }],
                metadata: None,
            });
        }

        let resp = TypeListResponse { entries, total: 10_000, offset: 0 };
        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: TypeListResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.entries.len(), 10_000);
        assert_eq!(deserialized.total, 10_000);
    }

    #[test]
    fn large_batch_read_response() {
        let mut results = Vec::with_capacity(10_000);
        for i in 0..10_000 {
            results.push(FieldReadResult {
                key: FieldKey { entity_id: i, component_id: 1, offset: 0 },
                value: Some(serde_json::json!(i)),
                error: None,
            });
        }

        let resp = ComponentBatchReadResponse { results };
        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: ComponentBatchReadResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.results.len(), 10_000);
    }

    #[test]
    fn events_response_with_empty_array() {
        let resp = EventsRecentResponse {
            events: vec![],
            total_in_log: 0,
        };

        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: EventsRecentResponse = serde_json::from_str(&json).unwrap();
        assert!(deserialized.events.is_empty());
        assert_eq!(deserialized.total_in_log, 0);
    }

    #[test]
    fn send_sync_bounds() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<FieldKey>();
        assert_sync::<FieldKey>();
        assert_send::<TypeRegistryEntry>();
        assert_sync::<TypeRegistryEntry>();
        assert_send::<FieldDescriptor>();
        assert_sync::<FieldDescriptor>();
        assert_send::<HierarchyEntry>();
        assert_sync::<HierarchyEntry>();
        assert_send::<DecoratorEntry>();
        assert_sync::<DecoratorEntry>();
        assert_send::<SourceLocation>();
        assert_sync::<SourceLocation>();
        assert_send::<EventEntry>();
        assert_sync::<EventEntry>();
    }

    #[test]
    fn clone_bounds() {
        fn assert_clone<T: Clone>() {}

        assert_clone::<FieldKey>();
        assert_clone::<TypeRegistryEntry>();
        assert_clone::<FieldDescriptor>();
        assert_clone::<HierarchyEntry>();
        assert_clone::<DecoratorEntry>();
        assert_clone::<SourceLocation>();
        assert_clone::<EventEntry>();
    }

    #[test]
    fn unknown_fields_deserialize_to_none() {
        // Forward-compat: extra fields in JSON are ignored
        let json = r#"{"success":true,"name":"Health","extraField":"ignored"}"#;
        let resp: TrinityInspectResponse = serde_json::from_str(json).unwrap();
        assert!(resp.success);
        assert_eq!(resp.name, Some("Health".into()));
    }

    #[test]
    fn camel_case_serde_renames() {
        let entry = HierarchyEntry {
            name: "Component".into(),
            module: None,
            is_trinity_base: true,
        };
        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("isTrinityBase"));
        assert!(!json.contains("is_trinity_base"));

        let decorator = DecoratorEntry {
            name: "component".into(),
            tier: Some(1),
            tier_name: Some("FOUNDATION".into()),
            foundation: None,
            doc: None,
            args: None,
        };
        let json = serde_json::to_string(&decorator).unwrap();
        assert!(json.contains("tierName"));
    }

    #[test]
    fn component_block_field_init() {
        let block = ComponentBlock {
            component_id: 1,
            fields: vec![
                FieldInit { offset: 0, value: serde_json::json!(3.14) },
                FieldInit { offset: 4, value: serde_json::json!("hello") },
                FieldInit { offset: 8, value: serde_json::json!(true) },
            ],
        };

        let json = serde_json::to_string(&block).unwrap();
        let deserialized: ComponentBlock = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.fields.len(), 3);
        assert_eq!(deserialized.fields[0].offset, 0);
        assert_eq!(deserialized.fields[0].value, serde_json::json!(3.14));
        assert_eq!(deserialized.fields[1].value, serde_json::json!("hello"));
        assert_eq!(deserialized.fields[2].value, serde_json::json!(true));
    }

    #[test]
    fn world_create_response() {
        let resp = WorldCreateResponse { world_handle: 0 };
        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: WorldCreateResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.world_handle, 0);
    }

    #[test]
    fn world_reset_request() {
        let req = WorldResetRequest { world_handle: 0 };
        let json = serde_json::to_string(&req).unwrap();
        let deserialized: WorldResetRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.world_handle, 0);
    }

    #[test]
    fn trinity_connect_request() {
        let req = TrinityConnectRequest {};
        let json = serde_json::to_string(&req).unwrap();
        let deserialized: TrinityConnectRequest = serde_json::from_str(&json).unwrap();
        assert!(serde_json::from_str::<TrinityConnectRequest>(&json).is_ok());
    }

    #[test]
    fn checksum_request() {
        let json = r#"{}"#;
        let req: ChecksumRequest = serde_json::from_str(json).unwrap();
        assert!(serde_json::to_string(&req).is_ok());
    }

    #[test]
    fn type_count_response() {
        let resp = TypeCountResponse { count: 42 };
        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: TypeCountResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.count, 42);
    }

    #[test]
    fn inspector_get_request() {
        let req = InspectorGetRequest {
            target_type: "type".into(),
            qualified_name: Some("game.components.Health".into()),
            target_id: None,
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"targetType\":\"type\""));
        assert!(json.contains("\"qualifiedName\":\"game.components.Health\""));

        let deserialized: InspectorGetRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.target_type, "type");
        assert_eq!(deserialized.qualified_name, Some("game.components.Health".into()));
    }

    #[test]
    fn field_key_hash_equality() {
        let a = FieldKey { entity_id: 1, component_id: 2, offset: 4 };
        let b = FieldKey { entity_id: 1, component_id: 2, offset: 4 };
        let c = FieldKey { entity_id: 1, component_id: 2, offset: 8 };

        assert_eq!(a, b);
        assert_ne!(a, c);

        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(a.clone());
        set.insert(b.clone()); // duplicate - no-op
        assert_eq!(set.len(), 1);
        set.insert(c);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn batch_write_response() {
        let resp = ComponentBatchWriteResponse { written: 5 };
        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: ComponentBatchWriteResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.written, 5);
    }

    #[test]
    fn empty_batch_read_response() {
        let resp = ComponentBatchReadResponse { results: vec![] };
        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: ComponentBatchReadResponse = serde_json::from_str(&json).unwrap();
        assert!(deserialized.results.is_empty());
    }

    // =========================================================================
    // WHITEBOX TESTS -- Internal structure, invariants, and code-path coverage
    // =========================================================================
    //
    // These tests examine the bridge protocol beyond simple serialization
    // round-trips. They validate:
    //   1. Channel routing dispatch logic (every code path, edge cases)
    //   2. TypeKind internal method correctness (as_str for all variants)
    //   3. Trait bounds for every protocol type (Debug, Clone, PartialEq,
    //      Send, Sync)
    //   4. METHOD_TABLE structural integrity (uniqueness, channel consistency)
    //   5. Serde attribute correctness (field visibility, defaults, renames)
    //   6. Hash/equality semantics for key types
    //   7. Boundary conditions (max values, empty collections, negative paths)
    //   8. Missing required field deserialization errors

    // -------------------------------------------------------------------------
    // SECTION W1 -- Channel Routing: channel_for_method exhaustive coverage
    // -------------------------------------------------------------------------

    /// Every method in METHOD_TABLE must route to its correct channel.
    /// This is a whitebox structural test of the dispatch function.
    #[test]
    fn channel_for_method_routes_all_type_endpoints() {
        for method in &["type.register", "type.list", "type.get", "type.remove", "type.count"] {
            assert_eq!(channel_for_method(method), Some("type"));
        }
    }

    #[test]
    fn channel_for_method_routes_all_data_endpoints() {
        for method in &["data.read", "data.write", "data.delete", "data.batch_read", "data.batch_write"] {
            assert_eq!(channel_for_method(method), Some("data"));
        }
    }

    #[test]
    fn channel_for_method_routes_all_command_endpoints() {
        for method in &[
            "command.create", "command.spawn", "command.despawn",
            "command.query", "command.reset", "command.stats",
        ] {
            assert_eq!(channel_for_method(method), Some("command"));
        }
    }

    #[test]
    fn channel_for_method_routes_all_system_endpoints() {
        for method in &[
            "system.connect", "system.status", "system.inspect",
            "system.inspector_get", "system.events_recent", "system.checksum",
        ] {
            assert_eq!(channel_for_method(method), Some("system"));
        }
    }

    /// channel_for_method with only a channel prefix (no dot or endpoint)
    /// returns None because the dispatch requires "prefix." dot-separated
    /// notation to avoid false positives (e.g. "typewriter").
    #[test]
    fn channel_for_method_prefix_only_returns_none() {
        assert_eq!(channel_for_method("type"), None);
        assert_eq!(channel_for_method("data"), None);
        assert_eq!(channel_for_method("command"), None);
        assert_eq!(channel_for_method("system"), None);
    }

    /// channel_for_method with empty string returns None.
    #[test]
    fn channel_for_method_empty_string_returns_none() {
        assert_eq!(channel_for_method(""), None);
    }

    /// channel_for_method with unknown prefix returns None.
    #[test]
    fn channel_for_method_unknown_prefix_returns_none() {
        assert_eq!(channel_for_method("unknown"), None);
        assert_eq!(channel_for_method("rpc"), None);
        assert_eq!(channel_for_method("ws"), None);
        assert_eq!(channel_for_method("http"), None);
    }

    /// channel_for_method with method-like strings that have extra dots
    /// after the prefix must still route correctly.
    #[test]
    fn channel_for_method_extra_dot_segments() {
        // starts_with("type.") matches "type." at the beginning
        assert_eq!(channel_for_method("type.register.extended"), Some("type"));
        assert_eq!(channel_for_method("data.write.batch"), Some("data"));
        assert_eq!(channel_for_method("command.spawn.v2"), Some("command"));
    }

    /// channel_for_method must not match partial channel names (e.g.
    /// "typewriter" should not match "type").
    #[test]
    fn channel_for_method_no_false_positive_prefix() {
        // "typewriter" starts with "type" not "type."
        assert_eq!(channel_for_method("typewriter.action"), None);
        assert_eq!(channel_for_method("database.query"), None);
        assert_eq!(channel_for_method("commandant.action"), None);
        assert_eq!(channel_for_method("systemic.check"), None);
    }

    // -------------------------------------------------------------------------
    // SECTION W2 -- TypeKind internal method correctness
    // -------------------------------------------------------------------------

    /// TypeKind::as_str must return the canonical string for every built-in
    /// variant. This is a whitebox test of the match arms.
    #[test]
    fn type_kind_as_str_built_in_variants() {
        assert_eq!(TypeKind::Component.as_str(), "component");
        assert_eq!(TypeKind::System.as_str(), "system");
        assert_eq!(TypeKind::Resource.as_str(), "resource");
        assert_eq!(TypeKind::Event.as_str(), "event");
    }

    /// TypeKind::Custom as_str delegates to the inner string.
    #[test]
    fn type_kind_as_str_custom_variant() {
        assert_eq!(TypeKind::Custom("state".into()).as_str(), "state");
        assert_eq!(TypeKind::Custom("".into()).as_str(), "");
        let long = "a".repeat(1000);
        assert_eq!(TypeKind::Custom(long.clone()).as_str(), long);
    }

    /// TypeKind PartialEq must distinguish between all four built-in variants
    /// and the Custom variant.
    #[test]
    fn type_kind_partial_eq_distinguishes_all_variants() {
        assert_eq!(TypeKind::Component, TypeKind::Component);
        assert_eq!(TypeKind::System, TypeKind::System);
        assert_eq!(TypeKind::Resource, TypeKind::Resource);
        assert_eq!(TypeKind::Event, TypeKind::Event);

        assert_ne!(TypeKind::Component, TypeKind::System);
        assert_ne!(TypeKind::Component, TypeKind::Resource);
        assert_ne!(TypeKind::Component, TypeKind::Event);
        assert_ne!(TypeKind::System, TypeKind::Resource);
        assert_ne!(TypeKind::System, TypeKind::Event);
        assert_ne!(TypeKind::Resource, TypeKind::Event);

        assert_eq!(TypeKind::Custom("x".into()), TypeKind::Custom("x".into()));
        assert_ne!(TypeKind::Custom("x".into()), TypeKind::Custom("y".into()));
        assert_ne!(TypeKind::Component, TypeKind::Custom("component".into()));
    }

    // -------------------------------------------------------------------------
    // SECTION W3 -- Trait bounds for every protocol type
    // -------------------------------------------------------------------------

    /// Compile-time assertion: every type used in the bridge protocol must
    /// implement Debug + Clone for testability and logging. Some types also
    /// require PartialEq for equality comparisons in tests.
    #[test]
    fn all_protocol_types_debug_clone() {
        fn assert_traits<T: std::fmt::Debug + Clone>() {}

        assert_traits::<TypeKind>();
        assert_traits::<FieldDescriptor>();
        assert_traits::<TypeRegisterRequest>();
        assert_traits::<TypeRegisterResponse>();
        assert_traits::<TypeListRequest>();
        assert_traits::<TypeRegistryEntry>();
        assert_traits::<TypeListResponse>();
        assert_traits::<TypeGetRequest>();
        assert_traits::<TypeRemoveRequest>();
        assert_traits::<TypeCountRequest>();
        assert_traits::<TypeCountResponse>();
        assert_traits::<FieldKey>();
        assert_traits::<ComponentReadRequest>();
        assert_traits::<ComponentWriteRequest>();
        assert_traits::<ComponentDeleteRequest>();
        assert_traits::<ComponentBatchReadRequest>();
        assert_traits::<FieldReadResult>();
        assert_traits::<ComponentBatchReadResponse>();
        assert_traits::<ComponentBatchWriteRequest>();
        assert_traits::<ComponentBatchWriteResponse>();
        assert_traits::<FieldInit>();
        assert_traits::<ComponentBlock>();
        assert_traits::<WorldCreateRequest>();
        assert_traits::<WorldCreateResponse>();
        assert_traits::<WorldSpawnRequest>();
        assert_traits::<WorldSpawnResponse>();
        assert_traits::<WorldDespawnRequest>();
        assert_traits::<WorldQueryRequest>();
        assert_traits::<WorldQueryResponse>();
        assert_traits::<WorldResetRequest>();
        assert_traits::<WorldStatsRequest>();
        assert_traits::<WorldStatsResponse>();
        assert_traits::<TrinityConnectRequest>();
        assert_traits::<TrinityConnectResponse>();
        assert_traits::<TrinityStatusRequest>();
        assert_traits::<TrinityStatusResponse>();
        assert_traits::<TrinityInspectRequest>();
        assert_traits::<HierarchyEntry>();
        assert_traits::<DecoratorEntry>();
        assert_traits::<SourceLocation>();
        assert_traits::<TrinityInspectResponse>();
        assert_traits::<InspectorGetRequest>();
        assert_traits::<EventsRecentRequest>();
        assert_traits::<EventEntry>();
        assert_traits::<EventsRecentResponse>();
        assert_traits::<ChecksumRequest>();
        assert_traits::<ChecksumResponse>();
    }

    /// Compile-time assertion: every type used in the bridge protocol must
    /// implement Send + Sync for GIL release / thread-safe transport.
    #[test]
    fn all_protocol_types_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<TypeKind>();
        assert_sync::<TypeKind>();
        assert_send::<FieldDescriptor>();
        assert_sync::<FieldDescriptor>();
        assert_send::<TypeRegisterRequest>();
        assert_sync::<TypeRegisterRequest>();
        assert_send::<TypeRegisterResponse>();
        assert_sync::<TypeRegisterResponse>();
        assert_send::<TypeListRequest>();
        assert_sync::<TypeListRequest>();
        assert_send::<TypeRegistryEntry>();
        assert_sync::<TypeRegistryEntry>();
        assert_send::<TypeListResponse>();
        assert_sync::<TypeListResponse>();
        assert_send::<TypeGetRequest>();
        assert_sync::<TypeGetRequest>();
        assert_send::<TypeRemoveRequest>();
        assert_sync::<TypeRemoveRequest>();
        assert_send::<TypeCountRequest>();
        assert_sync::<TypeCountRequest>();
        assert_send::<TypeCountResponse>();
        assert_sync::<TypeCountResponse>();
        assert_send::<FieldKey>();
        assert_sync::<FieldKey>();
        assert_send::<ComponentReadRequest>();
        assert_sync::<ComponentReadRequest>();
        assert_send::<ComponentWriteRequest>();
        assert_sync::<ComponentWriteRequest>();
        assert_send::<ComponentDeleteRequest>();
        assert_sync::<ComponentDeleteRequest>();
        assert_send::<ComponentBatchReadRequest>();
        assert_sync::<ComponentBatchReadRequest>();
        assert_send::<FieldReadResult>();
        assert_sync::<FieldReadResult>();
        assert_send::<ComponentBatchReadResponse>();
        assert_sync::<ComponentBatchReadResponse>();
        assert_send::<ComponentBatchWriteRequest>();
        assert_sync::<ComponentBatchWriteRequest>();
        assert_send::<ComponentBatchWriteResponse>();
        assert_sync::<ComponentBatchWriteResponse>();
        assert_send::<FieldInit>();
        assert_sync::<FieldInit>();
        assert_send::<ComponentBlock>();
        assert_sync::<ComponentBlock>();
        assert_send::<WorldCreateRequest>();
        assert_sync::<WorldCreateRequest>();
        assert_send::<WorldCreateResponse>();
        assert_sync::<WorldCreateResponse>();
        assert_send::<WorldSpawnRequest>();
        assert_sync::<WorldSpawnRequest>();
        assert_send::<WorldSpawnResponse>();
        assert_sync::<WorldSpawnResponse>();
        assert_send::<WorldDespawnRequest>();
        assert_sync::<WorldDespawnRequest>();
        assert_send::<WorldQueryRequest>();
        assert_sync::<WorldQueryRequest>();
        assert_send::<WorldQueryResponse>();
        assert_sync::<WorldQueryResponse>();
        assert_send::<WorldResetRequest>();
        assert_sync::<WorldResetRequest>();
        assert_send::<WorldStatsRequest>();
        assert_sync::<WorldStatsRequest>();
        assert_send::<WorldStatsResponse>();
        assert_sync::<WorldStatsResponse>();
        assert_send::<TrinityConnectRequest>();
        assert_sync::<TrinityConnectRequest>();
        assert_send::<TrinityConnectResponse>();
        assert_sync::<TrinityConnectResponse>();
        assert_send::<TrinityStatusRequest>();
        assert_sync::<TrinityStatusRequest>();
        assert_send::<TrinityStatusResponse>();
        assert_sync::<TrinityStatusResponse>();
        assert_send::<TrinityInspectRequest>();
        assert_sync::<TrinityInspectRequest>();
        assert_send::<HierarchyEntry>();
        assert_sync::<HierarchyEntry>();
        assert_send::<DecoratorEntry>();
        assert_sync::<DecoratorEntry>();
        assert_send::<SourceLocation>();
        assert_sync::<SourceLocation>();
        assert_send::<TrinityInspectResponse>();
        assert_sync::<TrinityInspectResponse>();
        assert_send::<InspectorGetRequest>();
        assert_sync::<InspectorGetRequest>();
        assert_send::<EventsRecentRequest>();
        assert_sync::<EventsRecentRequest>();
        assert_send::<EventEntry>();
        assert_sync::<EventEntry>();
        assert_send::<EventsRecentResponse>();
        assert_sync::<EventsRecentResponse>();
        assert_send::<ChecksumRequest>();
        assert_sync::<ChecksumRequest>();
        assert_send::<ChecksumResponse>();
        assert_sync::<ChecksumResponse>();
    }

    // -------------------------------------------------------------------------
    // SECTION W4 -- METHOD_TABLE structural integrity
    // -------------------------------------------------------------------------

    /// Every method name in METHOD_TABLE must be unique (no duplicates).
    #[test]
    fn method_table_all_names_unique() {
        let mut seen = std::collections::HashSet::new();
        for (method, _req, _resp) in METHOD_TABLE {
            assert!(
                seen.insert(method),
                "Duplicate method name in METHOD_TABLE: '{}'",
                method
            );
        }
        assert_eq!(seen.len(), METHOD_TABLE.len());
    }

    /// The per-channel endpoint count constants must match the actual number
    /// of entries in METHOD_TABLE for each channel.
    #[test]
    fn method_table_channel_counts_match_constants() {
        let type_count = METHOD_TABLE.iter().filter(|(m, _, _)| m.starts_with("type.")).count();
        let data_count = METHOD_TABLE.iter().filter(|(m, _, _)| m.starts_with("data.")).count();
        let cmd_count = METHOD_TABLE.iter().filter(|(m, _, _)| m.starts_with("command.")).count();
        let sys_count = METHOD_TABLE.iter().filter(|(m, _, _)| m.starts_with("system.")).count();

        assert_eq!(type_count, TYPE_CHANNEL_ENDPOINTS,
            "Expected {} type endpoints, found {}", TYPE_CHANNEL_ENDPOINTS, type_count);
        assert_eq!(data_count, DATA_CHANNEL_ENDPOINTS,
            "Expected {} data endpoints, found {}", DATA_CHANNEL_ENDPOINTS, data_count);
        assert_eq!(cmd_count, COMMAND_CHANNEL_ENDPOINTS,
            "Expected {} command endpoints, found {}", COMMAND_CHANNEL_ENDPOINTS, cmd_count);
        assert_eq!(sys_count, SYSTEM_CHANNEL_ENDPOINTS,
            "Expected {} system endpoints, found {}", SYSTEM_CHANNEL_ENDPOINTS, sys_count);

        assert_eq!(type_count + data_count + cmd_count + sys_count, TOTAL_ENDPOINTS);
    }

    /// Every METHOD_TABLE entry's second and third fields (request/response
    /// type names) must be non-empty strings.
    #[test]
    fn method_table_type_names_are_non_empty() {
        for (method, req_type, resp_type) in METHOD_TABLE {
            assert!(!req_type.is_empty(),
                "Method '{}' has empty request type name", method);
            assert!(!resp_type.is_empty(),
                "Method '{}' has empty response type name", method);
        }
    }

    /// Every method in METHOD_TABLE must have a channel prefix that is one
    /// of the four known channel constants.
    #[test]
    fn method_table_channels_are_known() {
        let known_prefixes = &["type.", "data.", "command.", "system."];
        for (method, _req, _resp) in METHOD_TABLE {
            let has_known_prefix = known_prefixes.iter().any(|p| method.starts_with(p));
            assert!(has_known_prefix,
                "Method '{}' does not start with a known channel prefix", method);
        }
    }

    /// The METHOD_TABLE must be sorted in a stable order (type, data, command,
    /// system) for deterministic dispatch.
    #[test]
    fn method_table_entries_are_in_channel_order() {
        let channels: Vec<&str> = METHOD_TABLE.iter()
            .map(|(m, _, _)| channel_for_method(m).unwrap_or("unknown"))
            .collect();

        let expected_order = ["type", "data", "command", "system"];
        let mut seen_idx = 0;

        for &channel in channels.iter() {
            let pos = expected_order.iter().position(|&c| c == channel).unwrap();
            assert!(pos >= seen_idx,
                "Method table out of order: '{}' appears after '{}'",
                channel, expected_order[seen_idx]);
            seen_idx = pos;
        }
    }

    // -------------------------------------------------------------------------
    // SECTION W5 -- Serde attribute correctness
    // -------------------------------------------------------------------------

    /// FieldDescriptor's `size` field has #[serde(default)], so it must
    /// deserialize as 0 when omitted from JSON.
    #[test]
    fn field_descriptor_size_defaults_to_zero() {
        let json = r#"{"name":"test","typeCode":"f32","offset":0}"#;
        let fd: FieldDescriptor = serde_json::from_str(json).unwrap();
        assert_eq!(fd.size, 0, "size must default to 0 when omitted");
    }

    /// FieldDescriptor's `metadata` field has #[serde(default)], so it must
    /// deserialize as None when omitted from JSON.
    #[test]
    fn type_register_metadata_defaults_to_none() {
        let json = r#"{"componentId":1,"name":"Test","typeKind":"component","totalSize":4,"fields":[],"module":"m"}"#;
        let req: TypeRegisterRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.metadata, None);
    }

    /// Every struct with #[serde(rename = "camelCase")] must produce the
    /// camelCase key and NOT the snake_case key in serialized JSON.
    #[test]
    fn serde_rename_fields_use_camel_case_not_snake_case() {
        // FieldDescriptor: type_code -> typeCode
        let fd = FieldDescriptor { name: "hp".into(), type_code: "f32".into(), offset: 0, size: 4 };
        let json = serde_json::to_string(&fd).unwrap();
        assert!(json.contains("\"typeCode\":"), "FieldDescriptor should serialize typeCode: {}", json);
        assert!(!json.contains("\"type_code\":"), "FieldDescriptor should NOT serialize type_code: {}", json);

        // FieldKey: entity_id -> entityId, component_id -> componentId
        let fk = FieldKey { entity_id: 1, component_id: 2, offset: 0 };
        let json = serde_json::to_string(&fk).unwrap();
        assert!(json.contains("\"entityId\":"), "FieldKey should serialize entityId: {}", json);
        assert!(json.contains("\"componentId\":"), "FieldKey should serialize componentId: {}", json);
        assert!(!json.contains("\"entity_id\":"), "FieldKey should NOT serialize entity_id: {}", json);
        assert!(!json.contains("\"component_id\":"), "FieldKey should NOT serialize component_id: {}", json);

        // WorldStatsResponse: entity_count -> entityCount, etc.
        let stats = WorldStatsResponse {
            entity_count: 100, archetype_count: 5, component_type_count: 3, total_fields: 400,
        };
        let json = serde_json::to_string(&stats).unwrap();
        assert!(json.contains("\"entityCount\":"));
        assert!(json.contains("\"archetypeCount\":"));
        assert!(json.contains("\"componentTypeCount\":"));
        assert!(json.contains("\"totalFields\":"));
    }

    /// All request/response types with camelCase fields: verify the specific
    /// rename attributes produce correct output.
    #[test]
    fn serde_rename_comprehensive_audit() {
        // TypeRegisterRequest renames: componentId, typeKind, totalSize
        let req = TypeRegisterRequest {
            component_id: 1, name: "Pos".into(), type_kind: TypeKind::Component,
            total_size: 12, fields: vec![], module: "m".into(), metadata: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"componentId\":"), "Missing componentId in TypeRegisterRequest");
        assert!(json.contains("\"typeKind\":"), "Missing typeKind in TypeRegisterRequest");
        assert!(json.contains("\"totalSize\":"), "Missing totalSize in TypeRegisterRequest");

        // TypeListRequest: typeFilter
        let req = TypeListRequest { type_filter: None, offset: 0, limit: None };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"typeFilter\":"), "Missing typeFilter in TypeListRequest");

        // TypeRegistryEntry: componentId, typeKind, totalSize
        let entry = TypeRegistryEntry {
            component_id: 1, name: "Pos".into(), type_kind: "component".into(),
            module: "m".into(), total_size: 12, fields: vec![], metadata: None,
        };
        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("\"componentId\":"), "Missing componentId in TypeRegistryEntry");

        // TypeGetRequest: componentId
        let req = TypeGetRequest { component_id: 1 };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"componentId\":"), "Missing componentId in TypeGetRequest");

        // TypeRemoveRequest: componentId
        let req = TypeRemoveRequest { component_id: 1 };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"componentId\":"), "Missing componentId in TypeRemoveRequest");

        // TypeCountRequest: typeFilter
        let req = TypeCountRequest { type_filter: None };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"typeFilter\":"), "Missing typeFilter in TypeCountRequest");

        // ComponentReadRequest: fieldType
        let req = ComponentReadRequest { key: FieldKey { entity_id: 1, component_id: 1, offset: 0 }, field_type: None };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"fieldType\":"), "Missing fieldType in ComponentReadRequest");
    }

    // -------------------------------------------------------------------------
    // SECTION W6 -- FieldKey semantic properties
    // -------------------------------------------------------------------------

    /// FieldKey equality must compare all three fields.
    #[test]
    fn field_key_equality_all_fields() {
        let a = FieldKey { entity_id: 1, component_id: 2, offset: 4 };
        let b = FieldKey { entity_id: 1, component_id: 2, offset: 4 };
        let c = FieldKey { entity_id: 2, component_id: 2, offset: 4 };
        let d = FieldKey { entity_id: 1, component_id: 3, offset: 4 };
        let e = FieldKey { entity_id: 1, component_id: 2, offset: 8 };

        assert_eq!(a, b);
        assert_ne!(a, c); // different entity_id
        assert_ne!(a, d); // different component_id
        assert_ne!(a, e); // different offset
    }

    /// FieldKey Hash + Eq must be consistent with HashMap semantics.
    #[test]
    fn field_key_hash_map_semantics() {
        use std::collections::HashMap;
        let mut map = HashMap::new();

        let k1 = FieldKey { entity_id: 1, component_id: 2, offset: 4 };
        let k2 = FieldKey { entity_id: 1, component_id: 2, offset: 4 }; // same as k1
        let k3 = FieldKey { entity_id: 1, component_id: 2, offset: 8 }; // different offset

        map.insert(k1.clone(), "value1");
        // Insert with equal key should overwrite
        map.insert(k2.clone(), "value2");
        assert_eq!(map.len(), 1, "Equal keys must map to same slot");
        assert_eq!(map.get(&k1), Some(&"value2"), "Second insert must overwrite first");

        // Different key must create new entry
        map.insert(k3, "value3");
        assert_eq!(map.len(), 2, "Different key must create new entry");
    }

    /// FieldKey with boundary values (u64::MAX, u32::MAX) must not panic
    /// during hash or equality operations.
    #[test]
    fn field_key_boundary_values() {
        let k = FieldKey {
            entity_id: u64::MAX,
            component_id: u32::MAX,
            offset: u32::MAX,
        };

        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(k.clone());
        assert!(set.contains(&k));
        assert_eq!(set.len(), 1);

        // Round-trip through JSON
        let json = serde_json::to_string(&k).unwrap();
        let deserialized: FieldKey = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, k);
    }

    // -------------------------------------------------------------------------
    // SECTION W7 -- Negative paths and error handling
    // -------------------------------------------------------------------------

    /// Missing required field (component_id with no default) must fail
    /// deserialization.
    #[test]
    fn type_get_request_missing_required_field_fails() {
        let json = r#"{}"#;
        let result: Result<TypeGetRequest, _> = serde_json::from_str(json);
        assert!(result.is_err(), "Missing required 'componentId' must fail deserialization");
    }

    /// TypeKind deserialization of unknown string value falls through to
    /// the #[serde(untagged)] Custom variant.
    #[test]
    fn type_kind_unknown_value_deserializes_to_custom() {
        let json = r#""state_machine""#;
        let kind: TypeKind = serde_json::from_str(json).unwrap();
        assert_eq!(kind, TypeKind::Custom("state_machine".into()));
    }

    /// TypeKind deserialization of a number fails (expects string).
    #[test]
    fn type_kind_wrong_type_fails_deserialization() {
        let json = r#"42"#;
        let result: Result<TypeKind, _> = serde_json::from_str(json);
        assert!(result.is_err(), "Number should not deserialize as TypeKind");
    }

    /// FieldDescriptor with negative offset must still deserialize (u32
    /// will reject it, but serde may accept negative numbers).
    #[test]
    fn field_descriptor_negative_offset_fails() {
        let json = r#"{"name":"x","typeCode":"f32","offset":-1,"size":4}"#;
        let result: Result<FieldDescriptor, _> = serde_json::from_str(json);
        // Negative values are invalid for u32
        assert!(result.is_err(), "Negative offset must fail deserialization for u32");
    }

    /// TrinityInspectResponse requires the `success` field (no #[serde(default)]
    /// on bool). Missing it must fail deserialization.
    #[test]
    fn trinity_inspect_response_success_is_required() {
        let json = r#"{"name":"Test"}"#;
        let result: Result<TrinityInspectResponse, _> = serde_json::from_str(json);
        assert!(
            result.is_err(),
            "Missing required 'success' field must fail deserialization"
        );
        assert!(
            result.unwrap_err().to_string().contains("missing field `success`"),
            "Error must mention 'missing field `success`'"
        );
    }

    // -------------------------------------------------------------------------
    // SECTION W8 -- Boundary conditions
    // -------------------------------------------------------------------------

    /// WorldStatsResponse with all-zero stats.
    #[test]
    fn world_stats_response_all_zeros() {
        let resp = WorldStatsResponse {
            entity_count: 0, archetype_count: 0,
            component_type_count: 0, total_fields: 0,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: WorldStatsResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.entity_count, 0);
        assert_eq!(deserialized.archetype_count, 0);
        assert_eq!(deserialized.component_type_count, 0);
        assert_eq!(deserialized.total_fields, 0);
    }

    /// WorldQueryRequest with empty component_ids vector.
    #[test]
    fn world_query_empty_component_ids() {
        let req = WorldQueryRequest {
            world_handle: 0, component_ids: vec![], limit: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let deserialized: WorldQueryRequest = serde_json::from_str(&json).unwrap();
        assert!(deserialized.component_ids.is_empty());
    }

    /// ComponentBlock with empty fields.
    #[test]
    fn component_block_empty_fields() {
        let block = ComponentBlock { component_id: 1, fields: vec![] };
        let json = serde_json::to_string(&block).unwrap();
        let deserialized: ComponentBlock = serde_json::from_str(&json).unwrap();
        assert!(deserialized.fields.is_empty());
    }

    /// EventsRecentRequest with no limit should serialize limit as 0 (default).
    #[test]
    fn events_recent_request_default_limit_is_zero() {
        let req = EventsRecentRequest {
            limit: 0, event_type_filter: None, since: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let deserialized: EventsRecentRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.limit, 0);
        assert_eq!(deserialized.event_type_filter, None);
        assert_eq!(deserialized.since, None);
    }

    /// InspectorGetRequest with all three target types.
    #[test]
    fn inspector_get_all_target_types() {
        for target in &["type", "instance", "decorator"] {
            let req = InspectorGetRequest {
                target_type: target.to_string(),
                qualified_name: None,
                target_id: None,
            };
            let json = serde_json::to_string(&req).unwrap();
            let deserialized: InspectorGetRequest = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized.target_type, *target);
        }
    }

    /// InspectorGetRequest with instance target and target_id.
    #[test]
    fn inspector_get_instance_target() {
        let req = InspectorGetRequest {
            target_type: "instance".into(),
            qualified_name: None,
            target_id: Some(42),
        };
        let json = serde_json::to_string(&req).unwrap();
        let deserialized: InspectorGetRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.target_type, "instance");
        assert_eq!(deserialized.target_id, Some(42));
    }

    // -------------------------------------------------------------------------
    // SECTION W9 -- Structural type round-trip completeness
    // -------------------------------------------------------------------------

    /// Every type in the Type channel must round-trip through JSON.
    /// Tests types that are not covered by individual round-trip tests above.
    #[test]
    fn type_get_request_round_trip() {
        let req = TypeGetRequest { component_id: 42 };
        let json = serde_json::to_string(&req).unwrap();
        let deserialized: TypeGetRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.component_id, 42);
    }

    #[test]
    fn type_remove_request_round_trip() {
        let req = TypeRemoveRequest { component_id: 7 };
        let json = serde_json::to_string(&req).unwrap();
        let deserialized: TypeRemoveRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.component_id, 7);
    }

    #[test]
    fn type_count_request_round_trip() {
        let req = TypeCountRequest { type_filter: Some("system".into()) };
        let json = serde_json::to_string(&req).unwrap();
        let deserialized: TypeCountRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.type_filter, Some("system".into()));
    }

    /// WorldStatsRequest round-trip.
    #[test]
    fn world_stats_request_round_trip() {
        let req = WorldStatsRequest { world_handle: 0 };
        let json = serde_json::to_string(&req).unwrap();
        let deserialized: WorldStatsRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.world_handle, 0);
    }

    /// TrinityStatusRequest round-trip.
    #[test]
    fn trinity_status_request_round_trip() {
        let req = TrinityStatusRequest {};
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(json, "{}");
        let deserialized: TrinityStatusRequest = serde_json::from_str(&json).unwrap();
        assert!(serde_json::to_string(&deserialized).is_ok());
    }

    /// TrinityInspectRequest round-trip.
    #[test]
    fn trinity_inspect_request_round_trip() {
        let req = TrinityInspectRequest { type_name: "Health".into() };
        let json = serde_json::to_string(&req).unwrap();
        let deserialized: TrinityInspectRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.type_name, "Health");
    }

    /// ChecksumRequest with empty object.
    #[test]
    fn checksum_request_empty_object() {
        let req = ChecksumRequest {};
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(json, "{}");
    }

    // -------------------------------------------------------------------------
    // SECTION W10 -- Constants verification
    // -------------------------------------------------------------------------

    /// The channel prefix constants must not be empty.
    #[test]
    fn channel_prefixes_are_non_empty() {
        assert!(!TYPE_CHANNEL_PREFIX.is_empty());
        assert!(!DATA_CHANNEL_PREFIX.is_empty());
        assert!(!COMMAND_CHANNEL_PREFIX.is_empty());
        assert!(!SYSTEM_CHANNEL_PREFIX.is_empty());
    }

    /// The channel prefix constants must match the expected dispatch logic
    /// (they are used in starts_with matching).
    #[test]
    fn channel_prefixes_match_dispatch_logic() {
        // channel_for_method checks starts_with("type") etc.
        // So the prefix must be the exact channel name.
        assert_eq!(TYPE_CHANNEL_PREFIX, "type");
        assert_eq!(DATA_CHANNEL_PREFIX, "data");
        assert_eq!(COMMAND_CHANNEL_PREFIX, "command");
        assert_eq!(SYSTEM_CHANNEL_PREFIX, "system");
    }

    /// The per-channel endpoint constants must sum to TOTAL_ENDPOINTS.
    #[test]
    fn endpoint_counts_sum_to_total() {
        let sum = TYPE_CHANNEL_ENDPOINTS + DATA_CHANNEL_ENDPOINTS
                + COMMAND_CHANNEL_ENDPOINTS + SYSTEM_CHANNEL_ENDPOINTS;
        assert_eq!(sum, TOTAL_ENDPOINTS);
        assert_eq!(TOTAL_ENDPOINTS, 22);
    }
}
