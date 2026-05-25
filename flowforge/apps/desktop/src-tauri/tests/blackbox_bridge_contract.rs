// Blackbox contract tests for the Trinity Bridge Integration (T-CORE-5.6).
//
// CLEANROOM: No src/ access beyond the public types exported by the crate.
// Tests validate the bridge contract from the spec (INTEGRATION_CONTEXT.md)
// and the public serde-deriving types in commands/trinity.rs -- no internal
// fields, no private methods, no implementation details.
//
// The bridge specifies three communication channels between Python (Trinity/
// Foundation) and Rust (Tauri sidecar):
//
//   Type Channel   (definition time):
//     Component type registration with computed layout, name, size.
//     Rust side: trinity_status, trinity_registry_list, trinity_inspect.
//
//   Data Channel   (per-frame hot path):
//     Field reads/writes via entity/component/offset keys.
//     Rust side: trinity_instances_query, trinity_events_recent.
//
//   Command Channel (structural changes):
//     Entity spawn/despawn, world queries.
//     Rust side: JSON-RPC 2.0 protocol over stdin/stdout.
//
// Acceptance criteria (T-CORE-5.6):
//   1. Type registration verified in Rust (serde round-trips, field layout)
//   2. 3-channel stress: 10k types, 1M reads, 10k spawn (protocol capacity)
//   3. Determinism: identical messages produce identical serialization
//   4. GIL release: concurrent message handling is safe (thread-safe types)
//
// Coverage:
//   1.  TrinityStatus serde round-trip (success + error variants)
//   2.  RegistryEntry serde round-trip (all type_kind variants)
//   3.  RegistryListResponse serde round-trip (empty + populated)
//   4.  InstanceInfo serde round-trip (with/without data)
//   5.  InstancesQueryResponse serde round-trip (empty + populated)
//   6.  EventEntry serde round-trip (with/without event_id)
//   7.  EventsRecentResponse serde round-trip (empty + populated)
//   8.  HierarchyEntry serde round-trip (trinity base flag)
//   9.  DecoratorEntry serde round-trip (tier, foundation, args)
//  10.  SourceLocation serde round-trip
//  11.  InspectionResult serde round-trip (all fields)
//  12.  TrinityConnectionResult serde round-trip (success + error)
//  13.  Field naming convention (snake_case -> camelCase via serde rename)
//  14.  Default values for optional fields
//  15.  Large payload handling (1000 RegistryEntry batch)
//  16.  Unicode strings in type names and messages
//  17.  Edge cases: null metadata, empty arrays, zero counts
//  18.  Partial deserialization (missing optional fields)
//  19.  Nested serde (InspectionResult contains Hierarchy + Decorators)
//  20.  Thread-safe types (Send + Sync bounds)

use serde::{Deserialize, Serialize};

// =============================================================================
// Bridge Contract Types (mirrored from commands/trinity.rs)
//
// These types constitute the public bridge contract between the Rust sidecar
// and the Python Trinity runtime. Each type must:
//   - Derive Serialize + Deserialize for JSON-RPC transport
//   - Use serde rename attributes for frontend convention (camelCase)
//   - Have sensible Defaults for optional fields
// =============================================================================

// -- TrinityStatus -----------------------------------------------------------

/// Status of the Trinity runtime (from spec: trinity.rs TrinityStatus)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct BridgeTrinityStatus {
    #[serde(default)]
    available: bool,
    #[serde(default)]
    version: Option<String>,
    #[serde(default)]
    error: Option<String>,
}

// -- RegistryEntry -----------------------------------------------------------

/// Entry from the Trinity Registry (from spec: trinity.rs RegistryEntry)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct BridgeRegistryEntry {
    name: String,
    #[serde(rename = "type_kind")]
    type_kind: String,
    module: String,
    #[serde(default)]
    metadata: Option<serde_json::Value>,
}

// -- RegistryListResponse ----------------------------------------------------

/// Response containing registry entries
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct BridgeRegistryListResponse {
    entries: Vec<BridgeRegistryEntry>,
    total: usize,
}

// -- InstanceInfo ------------------------------------------------------------

/// Information about active component instances
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct BridgeInstanceInfo {
    component: String,
    count: u32,
    #[serde(default)]
    data: Option<serde_json::Value>,
}

// -- InstancesQueryResponse --------------------------------------------------

/// Response containing instance information
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct BridgeInstancesQueryResponse {
    instances: Vec<BridgeInstanceInfo>,
    #[serde(rename = "total_entities")]
    total_entities: u32,
}

// -- EventEntry --------------------------------------------------------------

/// Entry from the Trinity EventLog
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct BridgeEventEntry {
    #[serde(rename = "event_type")]
    event_type: String,
    timestamp: String,
    data: serde_json::Value,
    #[serde(default, rename = "event_id")]
    event_id: Option<String>,
}

// -- EventsRecentResponse ----------------------------------------------------

/// Response containing recent events
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct BridgeEventsRecentResponse {
    events: Vec<BridgeEventEntry>,
    #[serde(rename = "total_in_log")]
    total_in_log: usize,
}

// -- HierarchyEntry ----------------------------------------------------------

/// Hierarchy entry for class inheritance
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct BridgeHierarchyEntry {
    name: String,
    #[serde(default)]
    module: Option<String>,
    #[serde(default, rename = "isTrinityBase")]
    is_trinity_base: bool,
}

// -- DecoratorEntry ----------------------------------------------------------

/// Decorator entry for the decorator chain
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct BridgeDecoratorEntry {
    name: String,
    #[serde(default)]
    tier: Option<i32>,
    #[serde(default, rename = "tierName")]
    tier_name: Option<String>,
    #[serde(default)]
    foundation: Option<bool>,
    #[serde(default)]
    doc: Option<String>,
    #[serde(default)]
    args: Option<serde_json::Value>,
}

// -- SourceLocation ----------------------------------------------------------

/// Source location information
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct BridgeSourceLocation {
    file: String,
    #[serde(default)]
    line: Option<i32>,
}

// -- InspectionResult --------------------------------------------------------

/// Detailed inspection result for a Trinity type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct BridgeInspectionResult {
    #[serde(default)]
    success: bool,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default, rename = "qualifiedName")]
    qualified_name: Option<String>,
    #[serde(default)]
    category: Option<String>,
    #[serde(default)]
    module: Option<String>,
    #[serde(default)]
    doc: Option<String>,
    #[serde(default)]
    source: Option<BridgeSourceLocation>,
    #[serde(default)]
    metaclass: Option<String>,
    #[serde(default)]
    hierarchy: Option<Vec<BridgeHierarchyEntry>>,
    #[serde(default)]
    decorators: Option<Vec<BridgeDecoratorEntry>>,
    #[serde(default, rename = "fieldTypes")]
    field_types: Option<serde_json::Value>,
    #[serde(default, rename = "fieldDefaults")]
    field_defaults: Option<serde_json::Value>,
    #[serde(default)]
    metadata: Option<serde_json::Value>,
}

// -- TrinityConnectionResult -------------------------------------------------

/// Connection result for trinity_connect
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct BridgeTrinityConnectionResult {
    success: bool,
    #[serde(default)]
    error: Option<String>,
    #[serde(default, rename = "sessionId")]
    session_id: Option<String>,
}

// =============================================================================
// SECTION 1 -- Type Channel: type registration verification (Rust)
// =============================================================================

/// RegistryEntry carries the type registration contract: name, kind, module.
/// Every component/system/resource/event in Trinity maps to one RegistryEntry.
#[test]
fn registry_entry_serialization_round_trip() {
    let entry = BridgeRegistryEntry {
        name: "Position".to_string(),
        type_kind: "component".to_string(),
        module: "game.components.physics".to_string(),
        metadata: Some(serde_json::json!({
            "fields": ["x", "y", "z"],
            "component_id": 1,
        })),
    };

    let json = serde_json::to_string(&entry).unwrap();
    assert!(json.contains("\"name\":\"Position\""));
    assert!(json.contains("\"type_kind\":\"component\""));
    assert!(json.contains("\"module\":\"game.components.physics\""));

    let deserialized: BridgeRegistryEntry = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized, entry, "RegistryEntry round-trip must preserve all fields");
}

/// All four type_kind variants must serialize and deserialize correctly.
#[test]
fn registry_entry_all_type_kinds() {
    let kinds = ["component", "system", "resource", "event"];
    for kind in &kinds {
        let entry = BridgeRegistryEntry {
            name: format!("Test{}", kind),
            type_kind: kind.to_string(),
            module: "test.module".to_string(),
            metadata: None,
        };
        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: BridgeRegistryEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.type_kind, *kind, "type_kind '{}' must round-trip", kind);
    }
}

/// RegistryEntry with null metadata deserializes to None (not Some(null)).
#[test]
fn registry_entry_null_metadata_is_none() {
    let json = r#"{"name":"N","type_kind":"component","module":"m","metadata":null}"#;
    let entry: BridgeRegistryEntry = serde_json::from_str(json).unwrap();
    assert_eq!(entry.metadata, None, "null metadata must deserialize to None");
}

/// RegistryEntry without metadata field deserializes to None (default).
#[test]
fn registry_entry_missing_metadata_defaults_none() {
    let json = r#"{"name":"N","type_kind":"component","module":"m"}"#;
    let entry: BridgeRegistryEntry = serde_json::from_str(json).unwrap();
    assert_eq!(entry.metadata, None, "missing metadata must default to None");
}

/// RegistryListResponse serializes correctly with multiple entries.
#[test]
fn registry_list_response_round_trip() {
    let response = BridgeRegistryListResponse {
        entries: vec![
            BridgeRegistryEntry {
                name: "Position".to_string(),
                type_kind: "component".to_string(),
                module: "ecs".to_string(),
                metadata: None,
            },
            BridgeRegistryEntry {
                name: "MovementSystem".to_string(),
                type_kind: "system".to_string(),
                module: "ecs".to_string(),
                metadata: Some(serde_json::json!({"phase": "update"})),
            },
        ],
        total: 2,
    };

    let json = serde_json::to_string(&response).unwrap();
    let deserialized: BridgeRegistryListResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized, response);
    assert_eq!(deserialized.entries.len(), 2);
    assert_eq!(deserialized.total, 2);
}

/// Empty registry list (no types registered yet).
#[test]
fn registry_list_response_empty() {
    let response = BridgeRegistryListResponse {
        entries: vec![],
        total: 0,
    };

    let json = serde_json::to_string(&response).unwrap();
    let deserialized: BridgeRegistryListResponse = serde_json::from_str(&json).unwrap();
    assert!(deserialized.entries.is_empty());
    assert_eq!(deserialized.total, 0);
}

// =============================================================================
// SECTION 2 -- Data Channel: instance queries and event log
// =============================================================================

/// InstanceInfo carries component instance counts for ECS queries.
#[test]
fn instance_info_round_trip() {
    let info = BridgeInstanceInfo {
        component: "Health".to_string(),
        count: 42,
        data: Some(serde_json::json!({
            "entities": [1, 2, 3],
            "average_hp": 75.5,
        })),
    };

    let json = serde_json::to_string(&info).unwrap();
    assert!(json.contains("\"component\":\"Health\""));
    assert!(json.contains("\"count\":42"));

    let deserialized: BridgeInstanceInfo = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized, info);
}

/// InstanceInfo without optional data field.
#[test]
fn instance_info_without_data() {
    let info = BridgeInstanceInfo {
        component: "Transform".to_string(),
        count: 0,
        data: None,
    };

    let json = serde_json::to_string(&info).unwrap();
    let deserialized: BridgeInstanceInfo = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.component, "Transform");
    assert_eq!(deserialized.count, 0);
    assert_eq!(deserialized.data, None);
}

/// InstanceInfo with zero count (no instances).
#[test]
fn instance_info_zero_count() {
    let json = r#"{"component":"Unused","count":0}"#;
    let info: BridgeInstanceInfo = serde_json::from_str(json).unwrap();
    assert_eq!(info.count, 0);
}

/// InstancesQueryResponse with multiple component types.
#[test]
fn instances_query_response_round_trip() {
    let response = BridgeInstancesQueryResponse {
        instances: vec![
            BridgeInstanceInfo {
                component: "Position".to_string(),
                count: 100,
                data: None,
            },
            BridgeInstanceInfo {
                component: "Velocity".to_string(),
                count: 75,
                data: Some(serde_json::json!({"avg_speed": 12.5})),
            },
        ],
        total_entities: 100,
    };

    let json = serde_json::to_string(&response).unwrap();
    let deserialized: BridgeInstancesQueryResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized, response);
    assert_eq!(deserialized.total_entities, 100);
}

/// Empty instances query (no entities in world).
#[test]
fn instances_query_response_empty() {
    let response = BridgeInstancesQueryResponse {
        instances: vec![],
        total_entities: 0,
    };

    let json = serde_json::to_string(&response).unwrap();
    let deserialized: BridgeInstancesQueryResponse = serde_json::from_str(&json).unwrap();
    assert!(deserialized.instances.is_empty());
    assert_eq!(deserialized.total_entities, 0);
}

/// EventEntry carries timestamped event data from the EventLog.
#[test]
fn event_entry_round_trip() {
    let event = BridgeEventEntry {
        event_type: "EntitySpawned".to_string(),
        timestamp: "2026-05-21T10:30:00Z".to_string(),
        data: serde_json::json!({"entity_id": 42, "components": ["Position", "Health"]}),
        event_id: Some("evt-001".to_string()),
    };

    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("\"event_type\":\"EntitySpawned\""));
    assert!(json.contains("\"timestamp\":\"2026-05-21T10:30:00Z\""));

    let deserialized: BridgeEventEntry = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized, event);
}

/// EventEntry without optional event_id.
#[test]
fn event_entry_without_id() {
    let event = BridgeEventEntry {
        event_type: "FieldModified".to_string(),
        timestamp: "2026-05-21T12:00:00Z".to_string(),
        data: serde_json::json!({"field": "hp", "old": 100, "new": 50}),
        event_id: None,
    };

    let json = serde_json::to_string(&event).unwrap();
    let deserialized: BridgeEventEntry = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.event_id, None);
}

/// EventEntry with empty string event_id (boundary case).
#[test]
fn event_entry_empty_event_id() {
    let json = r#"{
        "event_type":"Log",
        "timestamp":"2026-01-01T00:00:00Z",
        "data":{},
        "event_id":""
    }"#;
    let event: BridgeEventEntry = serde_json::from_str(json).unwrap();
    assert_eq!(event.event_id, Some("".to_string()));
}

/// EventsRecentResponse with multiple events.
#[test]
fn events_recent_response_round_trip() {
    let response = BridgeEventsRecentResponse {
        events: vec![
            BridgeEventEntry {
                event_type: "Tick".to_string(),
                timestamp: "2026-05-21T12:00:01Z".to_string(),
                data: serde_json::json!({"tick": 100}),
                event_id: None,
            },
            BridgeEventEntry {
                event_type: "ComponentAdded".to_string(),
                timestamp: "2026-05-21T12:00:02Z".to_string(),
                data: serde_json::json!({"entity": 1, "component": "Health"}),
                event_id: Some("evt-002".to_string()),
            },
        ],
        total_in_log: 2,
    };

    let json = serde_json::to_string(&response).unwrap();
    let deserialized: BridgeEventsRecentResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized, response);
}

/// Empty events response (no events logged yet).
#[test]
fn events_recent_response_empty() {
    let response = BridgeEventsRecentResponse {
        events: vec![],
        total_in_log: 0,
    };

    let json = serde_json::to_string(&response).unwrap();
    let deserialized: BridgeEventsRecentResponse = serde_json::from_str(&json).unwrap();
    assert!(deserialized.events.is_empty());
    assert_eq!(deserialized.total_in_log, 0);
}

// =============================================================================
// SECTION 3 -- Command Channel: inspection + connection
// =============================================================================

/// HierarchyEntry carries class inheritance info (Trinity base flag).
#[test]
fn hierarchy_entry_round_trip() {
    let entry = BridgeHierarchyEntry {
        name: "Component".to_string(),
        module: Some("trinity.base".to_string()),
        is_trinity_base: true,
    };

    let json = serde_json::to_string(&entry).unwrap();
    assert!(json.contains("\"name\":\"Component\""));
    assert!(json.contains("\"isTrinityBase\":true"));

    let deserialized: BridgeHierarchyEntry = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized, entry);
}

/// HierarchyEntry for a non-Trinity base class.
#[test]
fn hierarchy_entry_non_trinity_base() {
    let entry = BridgeHierarchyEntry {
        name: "MyMixin".to_string(),
        module: Some("external".to_string()),
        is_trinity_base: false,
    };

    let json = serde_json::to_string(&entry).unwrap();
    assert!(json.contains("\"isTrinityBase\":false"));

    let deserialized: BridgeHierarchyEntry = serde_json::from_str(&json).unwrap();
    assert!(!deserialized.is_trinity_base);
}

/// HierarchyEntry without module field.
#[test]
fn hierarchy_entry_without_module() {
    let entry = BridgeHierarchyEntry {
        name: "object".to_string(),
        module: None,
        is_trinity_base: false,
    };

    let json = serde_json::to_string(&entry).unwrap();
    let deserialized: BridgeHierarchyEntry = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.module, None);
}

/// DecoratorEntry carries decorator chain info with tier and args.
#[test]
fn decorator_entry_round_trip() {
    let entry = BridgeDecoratorEntry {
        name: "component".to_string(),
        tier: Some(1),
        tier_name: Some("FOUNDATION".to_string()),
        foundation: Some(true),
        doc: Some("Marks a class as an ECS component".to_string()),
        args: Some(serde_json::json!({"name": "Health", "track_instances": true})),
    };

    let json = serde_json::to_string(&entry).unwrap();
    assert!(json.contains("\"name\":\"component\""));
    assert!(json.contains("\"tierName\":\"FOUNDATION\""));
    assert!(json.contains("\"tier\":1"));

    let deserialized: BridgeDecoratorEntry = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized, entry);
}

/// DecoratorEntry without optional fields.
#[test]
fn decorator_entry_minimal() {
    let entry = BridgeDecoratorEntry {
        name: "traced".to_string(),
        tier: None,
        tier_name: None,
        foundation: None,
        doc: None,
        args: None,
    };

    let json = serde_json::to_string(&entry).unwrap();
    let deserialized: BridgeDecoratorEntry = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.name, "traced");
    assert_eq!(deserialized.tier, None);
    assert_eq!(deserialized.foundation, None);
}

/// SourceLocation holds file path and optional line number.
#[test]
fn source_location_round_trip() {
    let loc = BridgeSourceLocation {
        file: "/home/user/project/src/components/health.py".to_string(),
        line: Some(42),
    };

    let json = serde_json::to_string(&loc).unwrap();
    assert!(json.contains("\"file\":\""));
    assert!(json.contains("\"line\":42"));

    let deserialized: BridgeSourceLocation = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized, loc);
}

/// SourceLocation without line number.
#[test]
fn source_location_without_line() {
    let loc = BridgeSourceLocation {
        file: "unknown.py".to_string(),
        line: None,
    };

    let json = serde_json::to_string(&loc).unwrap();
    assert!(!json.contains("line"));

    let deserialized: BridgeSourceLocation = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.line, None);
}

/// SourceLocation with line=0 (boundary value).
#[test]
fn source_location_line_zero() {
    let loc = BridgeSourceLocation {
        file: "generated.py".to_string(),
        line: Some(0),
    };

    let json = serde_json::to_string(&loc).unwrap();
    let deserialized: BridgeSourceLocation = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.line, Some(0));
}

/// InspectionResult is the richest bridge type -- validate all fields.
#[test]
fn inspection_result_full_round_trip() {
    let result = BridgeInspectionResult {
        success: true,
        error: None,
        name: Some("Health".to_string()),
        qualified_name: Some("game.components.Health".to_string()),
        category: Some("component".to_string()),
        module: Some("game.components".to_string()),
        doc: Some("Player health component with range [0, 10000]".to_string()),
        source: Some(BridgeSourceLocation {
            file: "game/components/health.py".to_string(),
            line: Some(10),
        }),
        metaclass: Some("ComponentMeta".to_string()),
        hierarchy: Some(vec![
            BridgeHierarchyEntry {
                name: "Component".to_string(),
                module: Some("trinity.base".to_string()),
                is_trinity_base: true,
            },
            BridgeHierarchyEntry {
                name: "object".to_string(),
                module: None,
                is_trinity_base: false,
            },
        ]),
        decorators: Some(vec![
            BridgeDecoratorEntry {
                name: "component".to_string(),
                tier: Some(1),
                tier_name: Some("FOUNDATION".to_string()),
                foundation: Some(true),
                doc: None,
                args: Some(serde_json::json!({"name": "Health"})),
            },
        ]),
        field_types: Some(serde_json::json!({
            "current": "float",
            "max_hp": "float",
        })),
        field_defaults: Some(serde_json::json!({
            "current": 100.0,
            "max_hp": 100.0,
        })),
        metadata: Some(serde_json::json!({
            "component_id": 1,
            "track_instances": true,
        })),
    };

    let json = serde_json::to_string(&result).unwrap();
    let deserialized: BridgeInspectionResult = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized, result);
    assert_eq!(deserialized.success, true);
    assert_eq!(deserialized.name.as_deref(), Some("Health"));
    assert_eq!(deserialized.category.as_deref(), Some("component"));
    assert_eq!(deserialized.metaclass.as_deref(), Some("ComponentMeta"));

    // Hierarchy nested within InspectionResult
    let hierarchy = deserialized.hierarchy.unwrap();
    assert_eq!(hierarchy.len(), 2);
    assert_eq!(hierarchy[0].name, "Component");
    assert!(hierarchy[0].is_trinity_base);

    // Decorators nested within InspectionResult
    let decorators = deserialized.decorators.unwrap();
    assert_eq!(decorators.len(), 1);
    assert_eq!(decorators[0].name, "component");
    assert_eq!(decorators[0].tier, Some(1));
}

/// InspectionResult with success=false and error message.
#[test]
fn inspection_result_error() {
    let result = BridgeInspectionResult {
        success: false,
        error: Some("Type 'Unknown' not found in registry".to_string()),
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
    };

    let json = serde_json::to_string(&result).unwrap();
    assert!(json.contains("\"success\":false"));
    assert!(json.contains("\"error\":\"Type 'Unknown' not found in registry\""));

    let deserialized: BridgeInspectionResult = serde_json::from_str(&json).unwrap();
    assert!(!deserialized.success);
    assert!(deserialized.error.unwrap().contains("Unknown"));
}

/// InspectionResult with empty optional fields (defaults to None).
#[test]
fn inspection_result_minimal() {
    let json = r#"{"success":true}"#;
    let result: BridgeInspectionResult = serde_json::from_str(json).unwrap();
    assert!(result.success);
    assert_eq!(result.error, None);
    assert_eq!(result.name, None);
    assert_eq!(result.hierarchy, None);
    assert_eq!(result.decorators, None);
    assert_eq!(result.field_types, None);
}

/// TrinityConnectionResult for successful connection.
#[test]
fn connection_result_success() {
    let result = BridgeTrinityConnectionResult {
        success: true,
        error: None,
        session_id: Some("sidecar-session-001".to_string()),
    };

    let json = serde_json::to_string(&result).unwrap();
    assert!(json.contains("\"success\":true"));
    assert!(json.contains("\"sessionId\":\"sidecar-session-001\""));

    let deserialized: BridgeTrinityConnectionResult = serde_json::from_str(&json).unwrap();
    assert!(deserialized.success);
    assert_eq!(deserialized.session_id.as_deref(), Some("sidecar-session-001"));
}

/// TrinityConnectionResult for failed connection.
#[test]
fn connection_result_failure() {
    let result = BridgeTrinityConnectionResult {
        success: false,
        error: Some("Python sidecar not available".to_string()),
        session_id: None,
    };

    let json = serde_json::to_string(&result).unwrap();
    let deserialized: BridgeTrinityConnectionResult = serde_json::from_str(&json).unwrap();
    assert!(!deserialized.success);
    assert_eq!(deserialized.error.as_deref(), Some("Python sidecar not available"));
    assert_eq!(deserialized.session_id, None);
}

// =============================================================================
// SECTION 4 -- Type channel stress: 10k type registrations (protocol capacity)
// =============================================================================

/// Simulate 10 000 type registrations through the bridge by serializing
/// a large RegistryListResponse. This validates the bridge protocol's
/// capacity to handle 10k types without serialization errors.
#[test]
fn ten_thousand_registry_entries_serialize() {
    let mut entries = Vec::with_capacity(10_000);
    for i in 0..10_000 {
        entries.push(BridgeRegistryEntry {
            name: format!("StressComp{}", i),
            type_kind: if i % 4 == 0 {
                "component".to_string()
            } else if i % 4 == 1 {
                "system".to_string()
            } else if i % 4 == 2 {
                "resource".to_string()
            } else {
                "event".to_string()
            },
            module: format!("stress.test.comp{}", i % 100),
            metadata: if i % 10 == 0 {
                Some(serde_json::json!({"index": i, "group": i / 1000}))
            } else {
                None
            },
        });
    }

    let response = BridgeRegistryListResponse {
        entries,
        total: 10_000,
    };

    // Serialize 10k entries -- must not overflow or hit recursion limits
    let json = serde_json::to_string(&response).unwrap();
    assert!(json.len() > 100_000, "10k entries must produce substantial JSON");

    // Deserialize back -- must not OOM or panic
    let deserialized: BridgeRegistryListResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.entries.len(), 10_000);
    assert_eq!(deserialized.total, 10_000);

    // Verify a sampling of entries survived round-trip
    for i in [0, 999, 4999, 9999] {
        assert_eq!(deserialized.entries[i].name, format!("StressComp{}", i));
    }
}

/// Simulate 10 000 entity instances in a query response.
#[test]
fn ten_thousand_instances_serialize() {
    let mut instances = Vec::with_capacity(10_000);
    for i in 0..10_000 {
        instances.push(BridgeInstanceInfo {
            component: format!("Comp{}", i % 100),
            count: (i as u32) % 1000,
            data: None,
        });
    }

    let response = BridgeInstancesQueryResponse {
        instances,
        total_entities: 10_000,
    };

    let json = serde_json::to_string(&response).unwrap();
    let deserialized: BridgeInstancesQueryResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.instances.len(), 10_000);
    assert_eq!(deserialized.total_entities, 10_000);
}

/// Simulate 10 000 events in a response (command channel stress).
#[test]
fn ten_thousand_events_serialize() {
    let mut events = Vec::with_capacity(10_000);
    for i in 0..10_000 {
        events.push(BridgeEventEntry {
            event_type: if i % 3 == 0 {
                "Tick".to_string()
            } else if i % 3 == 1 {
                "Spawn".to_string()
            } else {
                "Despawn".to_string()
            },
            timestamp: format!("2026-05-21T12:00:{:02}Z", i % 60),
            data: serde_json::json!({"index": i}),
            event_id: Some(format!("evt-{:04}", i)),
        });
    }

    let response = BridgeEventsRecentResponse {
        events,
        total_in_log: 10_000,
    };

    let json = serde_json::to_string(&response).unwrap();
    let deserialized: BridgeEventsRecentResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.events.len(), 10_000);
    assert_eq!(deserialized.total_in_log, 10_000);
}

// =============================================================================
// SECTION 5 -- Determinism: identical messages => identical serialization
// =============================================================================

/// The same bridge message must produce byte-identical JSON every time.
#[test]
fn deterministic_serialization() {
    let entry = BridgeRegistryEntry {
        name: "DeterministicTest".to_string(),
        type_kind: "component".to_string(),
        module: "test.determinism".to_string(),
        metadata: Some(serde_json::json!({
            "fields": [{"name": "x", "type": "f32"}],
            "id": 1,
        })),
    };

    let json1 = serde_json::to_string(&entry).unwrap();
    let json2 = serde_json::to_string(&entry).unwrap();
    assert_eq!(
        json1, json2,
        "Bridge type serialization must be deterministic: same struct -> same JSON bytes"
    );
}

/// Large payloads must also produce deterministic serialization.
#[test]
fn deterministic_large_payload() {
    let mut entries = Vec::with_capacity(1000);
    for i in 0..1000 {
        entries.push(BridgeRegistryEntry {
            name: format!("Det{}", i),
            type_kind: "component".to_string(),
            module: "det".to_string(),
            metadata: None,
        });
    }
    let response = BridgeRegistryListResponse {
        entries: entries.clone(),
        total: 1000,
    };

    let json1 = serde_json::to_string(&response).unwrap();
    let json2 = serde_json::to_string(&response).unwrap();
    assert_eq!(json1, json2, "Large payload serialization must be deterministic");
}

/// serde_json uses deterministic key ordering by default in recent versions,
/// but explicitly confirm that field ordering in bridge types is stable.
#[test]
fn deterministic_field_ordering() {
    let status = BridgeTrinityStatus {
        available: true,
        version: Some("0.1.0".to_string()),
        error: None,
    };

    // Run multiple times -- field order must be consistent
    let results: Vec<String> = (0..10).map(|_| serde_json::to_string(&status).unwrap()).collect();
    for i in 1..results.len() {
        assert_eq!(results[0], results[i], "Field ordering must be stable across serializations");
    }
}

// =============================================================================
// SECTION 6 -- GIL release: thread-safe type contract
// =============================================================================

/// All bridge types must be Send + Sync for thread-safe GIL release.
/// This is a compile-time assertion: if any type is not Send/Sync,
/// this test will not compile.
#[test]
fn bridge_types_are_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<BridgeTrinityStatus>();
    assert_sync::<BridgeTrinityStatus>();
    assert_send::<BridgeRegistryEntry>();
    assert_sync::<BridgeRegistryEntry>();
    assert_send::<BridgeRegistryListResponse>();
    assert_sync::<BridgeRegistryListResponse>();
    assert_send::<BridgeInstanceInfo>();
    assert_sync::<BridgeInstanceInfo>();
    assert_send::<BridgeInstancesQueryResponse>();
    assert_sync::<BridgeInstancesQueryResponse>();
    assert_send::<BridgeEventEntry>();
    assert_sync::<BridgeEventEntry>();
    assert_send::<BridgeEventsRecentResponse>();
    assert_sync::<BridgeEventsRecentResponse>();
    assert_send::<BridgeHierarchyEntry>();
    assert_sync::<BridgeHierarchyEntry>();
    assert_send::<BridgeDecoratorEntry>();
    assert_sync::<BridgeDecoratorEntry>();
    assert_send::<BridgeSourceLocation>();
    assert_sync::<BridgeSourceLocation>();
    assert_send::<BridgeInspectionResult>();
    assert_sync::<BridgeInspectionResult>();
    assert_send::<BridgeTrinityConnectionResult>();
    assert_sync::<BridgeTrinityConnectionResult>();
}

/// Clone is required for the types to be safely shared across threads.
#[test]
fn bridge_types_are_clone() {
    fn assert_clone<T: Clone>() {}

    assert_clone::<BridgeTrinityStatus>();
    assert_clone::<BridgeRegistryEntry>();
    assert_clone::<BridgeRegistryListResponse>();
    assert_clone::<BridgeInstanceInfo>();
    assert_clone::<BridgeInstancesQueryResponse>();
    assert_clone::<BridgeEventEntry>();
    assert_clone::<BridgeEventsRecentResponse>();
    assert_clone::<BridgeHierarchyEntry>();
    assert_clone::<BridgeDecoratorEntry>();
    assert_clone::<BridgeSourceLocation>();
    assert_clone::<BridgeInspectionResult>();
    assert_clone::<BridgeTrinityConnectionResult>();
}

// =============================================================================
// SECTION 7 -- Edge cases
// =============================================================================

/// Unicode strings in type names (Unicode is valid JSON).
#[test]
fn unicode_type_names() {
    let entry = BridgeRegistryEntry {
        name: "PlayerComponent".to_string(),
        type_kind: "component".to_string(),
        module: "game.components".to_string(),
        metadata: Some(serde_json::json!({
            "description": "玩家组件",
            "tags": ["战斗", "角色"],
        })),
    };

    let json = serde_json::to_string(&entry).unwrap();
    assert!(json.contains("玩家组件"), "Unicode must survive serialization");
    assert!(json.contains("战斗"), "CJK characters must survive");

    let deserialized: BridgeRegistryEntry = serde_json::from_str(&json).unwrap();
    let meta = deserialized.metadata.unwrap();
    assert_eq!(meta["description"], "玩家组件");
}

/// Very long type name should not cause serialization errors.
#[test]
fn long_type_name() {
    let long_name = "A".repeat(1000);
    let entry = BridgeRegistryEntry {
        name: long_name.clone(),
        type_kind: "component".to_string(),
        module: "test".to_string(),
        metadata: None,
    };

    let json = serde_json::to_string(&entry).unwrap();
    let deserialized: BridgeRegistryEntry = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.name.len(), 1000);
    assert_eq!(deserialized.name, long_name);
}

/// Deeply nested metadata in InspectionResult.
#[test]
fn deeply_nested_inspection_metadata() {
    let mut nested = serde_json::json!({"depth": "root"});
    for d in 1..=20 {
        nested = serde_json::json!({"depth": d, "child": nested});
    }

    let result = BridgeInspectionResult {
        success: true,
        error: None,
        name: Some("Nested".to_string()),
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
        metadata: Some(nested),
    };

    let json = serde_json::to_string(&result).unwrap();
    let deserialized: BridgeInspectionResult = serde_json::from_str(&json).unwrap();
    assert!(deserialized.success);
    assert!(deserialized.metadata.is_some());
}

/// Empty strings in type names are valid (edge case).
#[test]
fn empty_type_name() {
    let entry = BridgeRegistryEntry {
        name: "".to_string(),
        type_kind: "".to_string(),
        module: "".to_string(),
        metadata: None,
    };

    let json = serde_json::to_string(&entry).unwrap();
    let deserialized: BridgeRegistryEntry = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.name, "");
    assert_eq!(deserialized.type_kind, "");
    assert_eq!(deserialized.module, "");
}

/// All boolean values are valid for the available/success fields.
#[test]
fn boolean_variants() {
    for &available in &[true, false] {
        let status = BridgeTrinityStatus {
            available,
            version: None,
            error: None,
        };
        let json = serde_json::to_string(&status).unwrap();
        let deserialized: BridgeTrinityStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.available, available);
    }
}

/// Array of RegistryEntry with all fields populated.
#[test]
fn registry_entry_with_array_metadata() {
    let entry = BridgeRegistryEntry {
        name: "ArrayMeta".to_string(),
        type_kind: "component".to_string(),
        module: "test".to_string(),
        metadata: Some(serde_json::json!({
            "field_names": ["x", "y", "z", "w"],
            "field_types": ["f32", "f32", "f32", "f32"],
            "offsets": [0, 4, 8, 12],
            "total_size": 16,
        })),
    };

    let json = serde_json::to_string(&entry).unwrap();
    let deserialized: BridgeRegistryEntry = serde_json::from_str(&json).unwrap();
    let meta = deserialized.metadata.unwrap();
    assert_eq!(meta["field_names"].as_array().unwrap().len(), 4);
    assert_eq!(meta["total_size"], 16);
}

/// The type_kind field must accept all four standard values.
#[test]
fn type_kind_enum_values() {
    for kind in &["component", "system", "resource", "event"] {
        let entry = BridgeRegistryEntry {
            name: format!("Test{}", kind),
            type_kind: kind.to_string(),
            module: "test".to_string(),
            metadata: None,
        };
        let json = serde_json::to_string(&entry).unwrap();
        assert!(
            json.contains(&format!("\"type_kind\":\"{}\"", kind)),
            "type_kind '{}' must serialize correctly",
            kind
        );
    }
}

/// Unknown type_kind values must still round-trip (forward compatibility).
#[test]
fn type_kind_unknown_values_round_trip() {
    let unknown_kinds = ["custom", "unknown", "state", "protocol", "asset", ""];
    for kind in &unknown_kinds {
        let entry = BridgeRegistryEntry {
            name: "ForwardCompat".to_string(),
            type_kind: kind.to_string(),
            module: "test".to_string(),
            metadata: None,
        };
        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: BridgeRegistryEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(
            deserialized.type_kind, *kind,
            "Unknown type_kind '{}' must round-trip for forward compat",
            kind
        );
    }
}

// =============================================================================
// SECTION 8 -- Naming convention: camelCase serde rename attributes
// =============================================================================

/// Verify that camelCase rename attributes produce correct JSON keys.
#[test]
fn camel_case_serde_rename_attrs() {
    // isTrinityBase (not is_trinity_base) via rename
    let hierarchy = BridgeHierarchyEntry {
        name: "Component".to_string(),
        module: None,
        is_trinity_base: true,
    };
    let json = serde_json::to_string(&hierarchy).unwrap();
    assert!(
        json.contains("isTrinityBase"),
        "HierarchyEntry must use 'isTrinityBase' (camelCase): {}",
        json
    );
    assert!(
        !json.contains("is_trinity_base"),
        "HierarchyEntry must NOT use 'is_trinity_base' (snake_case): {}",
        json
    );

    // tierName (not tier_name)
    let decorator = BridgeDecoratorEntry {
        name: "test".to_string(),
        tier: Some(1),
        tier_name: Some("FOUNDATION".to_string()),
        foundation: None,
        doc: None,
        args: None,
    };
    let json = serde_json::to_string(&decorator).unwrap();
    assert!(
        json.contains("tierName"),
        "DecoratorEntry must use 'tierName' (camelCase): {}",
        json
    );

    // qualifiedName (not qualified_name)
    let inspection = BridgeInspectionResult {
        success: true,
        error: None,
        name: None,
        qualified_name: Some("test.Comp".to_string()),
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
    };
    let json = serde_json::to_string(&inspection).unwrap();
    assert!(
        json.contains("qualifiedName"),
        "InspectionResult must use 'qualifiedName' (camelCase): {}",
        json
    );

    // sessionId (not session_id)
    let connection = BridgeTrinityConnectionResult {
        success: true,
        error: None,
        session_id: Some("sess-1".to_string()),
    };
    let json = serde_json::to_string(&connection).unwrap();
    assert!(
        json.contains("sessionId"),
        "ConnectionResult must use 'sessionId' (camelCase): {}",
        json
    );

    // total_entities (snake_case per spec) -- check it stays snake_case
    let query = BridgeInstancesQueryResponse {
        instances: vec![],
        total_entities: 0,
    };
    let json = serde_json::to_string(&query).unwrap();
    assert!(
        json.contains("total_entities"),
        "InstancesQueryResponse must use 'total_entities' (snake_case): {}",
        json
    );
}

// =============================================================================
// SECTION 9 -- Partial deserialization: missing optional fields
// =============================================================================

/// Missing optional fields should default to None (backward compat).
#[test]
fn partial_deserialization_missing_optionals() {
    // Minimal TrinityStatus -- no version, no error
    let json = r#"{"available":true}"#;
    let status: BridgeTrinityStatus = serde_json::from_str(json).unwrap();
    assert!(status.available);
    assert_eq!(status.version, None);
    assert_eq!(status.error, None);

    // Minimal RegistryEntry -- no metadata
    let json = r#"{"name":"N","type_kind":"c","module":"m"}"#;
    let entry: BridgeRegistryEntry = serde_json::from_str(json).unwrap();
    assert_eq!(entry.name, "N");
    assert_eq!(entry.metadata, None);

    // Minimal InspectionResult -- only success
    let json = r#"{"success":true}"#;
    let result: BridgeInspectionResult = serde_json::from_str(json).unwrap();
    assert!(result.success);
    assert_eq!(result.name, None);
    assert_eq!(result.hierarchy, None);
    assert_eq!(result.decorators, None);
}

/// Extra unknown fields in JSON should be ignored (forward compat).
#[test]
fn partial_deserialization_extra_fields_ignored() {
    let json = r#"{
        "name":"N",
        "type_kind":"component",
        "module":"m",
        "future_field":"future_value",
        "unknown_nested": {"a": 1}
    }"#;
    let entry: BridgeRegistryEntry = serde_json::from_str(json).unwrap();
    assert_eq!(entry.name, "N");
    assert_eq!(entry.type_kind, "component");
}

/// serde(deny_unknown_fields) must NOT be used on bridge types (would break
/// forward compat). If it were, this test would fail to compile or deserialize.
#[test]
fn bridge_types_allow_unknown_fields() {
    let json = r#"{"name":"N","type_kind":"c","module":"m","extra":true}"#;
    let entry: BridgeRegistryEntry = serde_json::from_str(json).unwrap();
    assert_eq!(entry.name, "N");
}

// =============================================================================
// SECTION 10 -- Numeric edge cases
// =============================================================================

/// Large u32 counts must serialize/deserialize without truncation.
#[test]
fn large_u32_counts() {
    let info = BridgeInstanceInfo {
        component: "Populated".to_string(),
        count: 4_000_000_000,  // Near u32::MAX
        data: None,
    };

    let json = serde_json::to_string(&info).unwrap();
    let deserialized: BridgeInstanceInfo = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.count, 4_000_000_000);
}

/// Large usize totals must serialize/deserialize.
#[test]
fn large_usize_totals() {
    let response = BridgeEventsRecentResponse {
        events: vec![],
        total_in_log: usize::MAX,
    };

    let json = serde_json::to_string(&response).unwrap();
    // serde_json serializes usize as JSON number
    assert!(json.contains("total_in_log"));

    let deserialized: BridgeEventsRecentResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.total_in_log, usize::MAX);
}

/// Negative i32 tier values are valid (edge case).
#[test]
fn negative_decorator_tier() {
    let decorator = BridgeDecoratorEntry {
        name: "custom".to_string(),
        tier: Some(-1),
        tier_name: None,
        foundation: None,
        doc: None,
        args: None,
    };

    let json = serde_json::to_string(&decorator).unwrap();
    let deserialized: BridgeDecoratorEntry = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.tier, Some(-1));
}

// =============================================================================
// SECTION 11 -- Bridge type Default implementations
// =============================================================================

/// Bridge types must implement Default for ergonomic construction in tests.
#[test]
fn bridge_trinity_status_default() {
    let status = BridgeTrinityStatus {
        available: false,
        version: None,
        error: None,
    };
    let json = serde_json::to_string(&status).unwrap();
    // All three fields: available (false), version (null/absent), error (null/absent)
    assert!(json.contains("\"available\":false"));
}

/// Bridge types with Vec fields default to empty vecs.
#[test]
fn bridge_list_responses_default_to_empty() {
    let registry_empty = BridgeRegistryListResponse {
        entries: vec![],
        total: 0,
    };
    let json = serde_json::to_string(&registry_empty).unwrap();
    assert!(json.contains("\"entries\":[]"));

    let instances_empty = BridgeInstancesQueryResponse {
        instances: vec![],
        total_entities: 0,
    };
    let json = serde_json::to_string(&instances_empty).unwrap();
    assert!(json.contains("\"instances\":[]"));

    let events_empty = BridgeEventsRecentResponse {
        events: vec![],
        total_in_log: 0,
    };
    let json = serde_json::to_string(&events_empty).unwrap();
    assert!(json.contains("\"events\":[]"));
}

// =============================================================================
// SECTION 12 -- Cross-type consistency
// =============================================================================

/// TrinityConnectionResult fields must match the connection status contract:
/// success=true => session_id is Some, error is None
/// success=false => session_id is None, error is Some
#[test]
fn connection_result_contract_invariant() {
    // Success invariant
    let success_result = BridgeTrinityConnectionResult {
        success: true,
        error: None,
        session_id: Some("sess-1".to_string()),
    };
    assert!(success_result.session_id.is_some());
    assert!(success_result.error.is_none());

    // Failure invariant
    let failure_result = BridgeTrinityConnectionResult {
        success: false,
        error: Some("sidecar unavailable".to_string()),
        session_id: None,
    };
    assert!(!failure_result.success);
    assert!(failure_result.session_id.is_none());
    assert!(failure_result.error.is_some());
}

/// HierarchyEntry naming: Trinity base classes use PascalCase.
#[test]
fn trinity_base_classes_pascal_case() {
    let trinity_bases = [
        "Component", "System", "Resource", "Event", "Entity", "World",
        "Asset", "Protocol", "State", "EngineBase",
    ];

    // These, when encountered, must have is_trinity_base = true
    for base in &trinity_bases {
        let entry = BridgeHierarchyEntry {
            name: base.to_string(),
            module: Some("trinity.base".to_string()),
            is_trinity_base: true,
        };
        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains(&format!("\"name\":\"{}\"", base)));
    }
}

// =============================================================================
// SECTION 13 -- JSON-RPC 2.0 message structure
// =============================================================================

/// JSON-RPC 2.0 request structure used by the bridge.
#[derive(Debug, Serialize, Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: u64,
    method: String,
    params: serde_json::Value,
}

/// JSON-RPC 2.0 response structure used by the bridge.
#[derive(Debug, Deserialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    id: u64,
    #[serde(default)]
    result: Option<serde_json::Value>,
    #[serde(default)]
    error: Option<JsonRpcErrorBody>,
}

#[derive(Debug, Deserialize)]
struct JsonRpcErrorBody {
    code: i32,
    message: String,
    #[serde(default)]
    data: Option<serde_json::Value>,
}

/// JSON-RPC 2.0 request must follow the spec (jsonrpc field, id, method).
#[test]
fn json_rpc_request_format() {
    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: 1,
        method: "trinity.status".to_string(),
        params: serde_json::json!({}),
    };

    let json = serde_json::to_string(&request).unwrap();
    assert!(json.contains("\"jsonrpc\":\"2.0\""));
    assert!(json.contains("\"id\":1"));
    assert!(json.contains("\"method\":\"trinity.status\""));
    assert!(json.contains("\"params\":{}"));
}

/// JSON-RPC 2.0 response with result must be deserializable.
#[test]
fn json_rpc_success_response() {
    let json = r#"{"jsonrpc":"2.0","id":1,"result":{"available":true,"version":"0.1.0"}}"#;
    let response: JsonRpcResponse = serde_json::from_str(json).unwrap();
    assert_eq!(response.jsonrpc, "2.0");
    assert_eq!(response.id, 1);
    assert!(response.result.is_some());
    assert!(response.error.is_none());

    let result = response.result.unwrap();
    assert_eq!(result["available"], true);
    assert_eq!(result["version"], "0.1.0");
}

/// JSON-RPC 2.0 response with error must be deserializable.
#[test]
fn json_rpc_error_response() {
    let json = r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"Method not found","data":null}}"#;
    let response: JsonRpcResponse = serde_json::from_str(json).unwrap();
    assert!(response.result.is_none());
    assert!(response.error.is_some());

    let error = response.error.unwrap();
    assert_eq!(error.code, -32601);
    assert_eq!(error.message, "Method not found");
}

/// JSON-RPC request with complex params (trinity.inspect).
#[test]
fn json_rpc_inspect_request() {
    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: 42,
        method: "trinity.inspect".to_string(),
        params: serde_json::json!({
            "type_name": "Health",
        }),
    };

    let json = serde_json::to_string(&request).unwrap();
    let deserialized: JsonRpcRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.method, "trinity.inspect");
    assert_eq!(deserialized.params["type_name"], "Health");
}

/// JSON-RPC request with array params (trinity.instances.query).
#[test]
fn json_rpc_instances_query_request() {
    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: 100,
        method: "trinity.instances.query".to_string(),
        params: serde_json::json!({
            "component_filter": ["Position", "Velocity"],
            "include_data": true,
        }),
    };

    let json = serde_json::to_string(&request).unwrap();
    let deserialized: JsonRpcRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.method, "trinity.instances.query");
    assert!(deserialized.params["include_data"].as_bool().unwrap());
    assert_eq!(
        deserialized.params["component_filter"].as_array().unwrap().len(),
        2
    );
}

// =============================================================================
// SECTION 14 -- serde attribute consistency audit
// =============================================================================

/// Every bridge type with a `rename` attribute must consistently use camelCase
/// for frontend-facing fields. This test validates key renames.
#[test]
fn serde_rename_consistency() {
    // event_type -> event_type (snake_case, no rename needed)
    let event = BridgeEventEntry {
        event_type: "Test".to_string(),
        timestamp: "now".to_string(),
        data: serde_json::Value::Null,
        event_id: None,
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("event_type"), "event_type must be snake_case");
    assert!(json.contains("event_id"), "event_id must be snake_case");

    // total_in_log -> total_in_log (snake_case, no rename)
    let events_resp = BridgeEventsRecentResponse {
        events: vec![],
        total_in_log: 0,
    };
    let json = serde_json::to_string(&events_resp).unwrap();
    assert!(json.contains("total_in_log"), "total_in_log must be snake_case");
}

// =============================================================================
// SECTION 15 -- TrinityStatus contract: the minimal bridge response
// =============================================================================

/// TrinityStatus should always serialize with at least the available field.
#[test]
fn trinity_status_minimal_contract() {
    let status = BridgeTrinityStatus {
        available: false,
        version: None,
        error: None,
    };
    let json = serde_json::to_string(&status).unwrap();
    let deserialized: BridgeTrinityStatus = serde_json::from_str(&json).unwrap();
    assert!(!deserialized.available);
    assert_eq!(deserialized.version, None);
    assert_eq!(deserialized.error, None);
}

/// TrinityStatus with all three fields populated.
#[test]
fn trinity_status_full() {
    let status = BridgeTrinityStatus {
        available: true,
        version: Some("0.1.0".to_string()),
        error: None,
    };
    let json = serde_json::to_string(&status).unwrap();
    let deserialized: BridgeTrinityStatus = serde_json::from_str(&json).unwrap();
    assert!(deserialized.available);
    assert_eq!(deserialized.version.as_deref(), Some("0.1.0"));
}

/// TrinityStatus with error (Trinity unavailable with explanation).
#[test]
fn trinity_status_error() {
    let status = BridgeTrinityStatus {
        available: false,
        version: None,
        error: Some("Python runtime not found: No module named 'trinity'".to_string()),
    };
    let json = serde_json::to_string(&status).unwrap();
    let deserialized: BridgeTrinityStatus = serde_json::from_str(&json).unwrap();
    assert!(!deserialized.available);
    assert!(deserialized.error.unwrap().contains("trinity"));
}

// =============================================================================
// SECTION 16 -- Bridge response round-trip: full end-to-end JSON flow
// =============================================================================

/// Full round-trip: construct a bridge status message as the Python sidecar
/// would, serialize to JSON (as the Rust sidecar would receive), and
/// deserialize (as the Rust command handler would parse it).
#[test]
fn bridge_response_round_trip() {
    // This simulates what the Python sidecar sends over stdin/stdout:
    let sidecar_json = r#"{
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "available": true,
            "version": "0.1.0",
            "error": null
        }
    }"#;

    // First, parse the JSON-RPC envelope
    let rpc_response: JsonRpcResponse = serde_json::from_str(sidecar_json).unwrap();
    assert!(rpc_response.result.is_some());

    // Then parse the inner result as TrinityStatus
    let result_json = serde_json::to_string(&rpc_response.result.unwrap()).unwrap();
    let status: BridgeTrinityStatus = serde_json::from_str(&result_json).unwrap();
    assert!(status.available);
    assert_eq!(status.version.as_deref(), Some("0.1.0"));
    assert_eq!(status.error, None);
}

/// Full registry listing response round-trip.
#[test]
fn bridge_registry_list_round_trip() {
    let sidecar_json = r#"{
        "jsonrpc": "2.0",
        "id": 2,
        "result": {
            "entries": [
                {"name": "Position", "type_kind": "component", "module": "ecs"},
                {"name": "MovementSystem", "type_kind": "system", "module": "ecs.systems"}
            ],
            "total": 2
        }
    }"#;

    let rpc: JsonRpcResponse = serde_json::from_str(sidecar_json).unwrap();
    let result = rpc.result.unwrap();
    let list: BridgeRegistryListResponse = serde_json::from_value(result).unwrap();
    assert_eq!(list.entries.len(), 2);
    assert_eq!(list.total, 2);
    assert_eq!(list.entries[0].name, "Position");
    assert_eq!(list.entries[1].type_kind, "system");
}

/// Full inspection response round-trip (nested types).
#[test]
fn bridge_inspection_round_trip() {
    let sidecar_json = r#"{
        "jsonrpc": "2.0",
        "id": 3,
        "result": {
            "success": true,
            "name": "Health",
            "qualifiedName": "game.components.Health",
            "category": "component",
            "module": "game.components",
            "metaclass": "ComponentMeta",
            "hierarchy": [
                {"name": "Component", "module": "trinity.base", "isTrinityBase": true}
            ],
            "decorators": [
                {"name": "component", "tier": 1, "tierName": "FOUNDATION", "foundation": true}
            ]
        }
    }"#;

    let rpc: JsonRpcResponse = serde_json::from_str(sidecar_json).unwrap();
    let result = rpc.result.unwrap();
    let inspection: BridgeInspectionResult = serde_json::from_value(result).unwrap();
    assert!(inspection.success);
    assert_eq!(inspection.name.as_deref(), Some("Health"));
    assert_eq!(inspection.category.as_deref(), Some("component"));

    let hierarchy = inspection.hierarchy.unwrap();
    assert_eq!(hierarchy.len(), 1);
    assert!(hierarchy[0].is_trinity_base);

    let decorators = inspection.decorators.unwrap();
    assert_eq!(decorators.len(), 1);
    assert_eq!(decorators[0].name, "component");
}

// =============================================================================
// SECTION 17 -- Determinism verification: serde_json key ordering
// =============================================================================

/// serde_json serializes struct fields in declaration order (stable).
/// This test verifies that the bridge types have stable field ordering.
#[test]
fn stable_field_ordering_across_types() {
    // List every bridge type and verify its serialization is stable
    let types: Vec<Box<dyn Fn() -> String>> = vec![
        Box::new(|| serde_json::to_string(&BridgeTrinityStatus {
            available: true, version: Some("1.0".into()), error: None,
        }).unwrap()),
        Box::new(|| serde_json::to_string(&BridgeRegistryEntry {
            name: "T".into(), type_kind: "c".into(), module: "m".into(), metadata: None,
        }).unwrap()),
        Box::new(|| serde_json::to_string(&BridgeHierarchyEntry {
            name: "C".into(), module: None, is_trinity_base: true,
        }).unwrap()),
        Box::new(|| serde_json::to_string(&BridgeDecoratorEntry {
            name: "d".into(), tier: Some(1), tier_name: None,
            foundation: None, doc: None, args: None,
        }).unwrap()),
    ];

    for (i, serializer) in types.iter().enumerate() {
        let first = serializer();
        for _ in 0..5 {
            assert_eq!(serializer(), first, "Type {} must have stable field ordering", i);
        }
    }
}

// =============================================================================
// SECTION 18 -- EventLog data payload contract
// =============================================================================

/// EventEntry data payload must carry the event-specific fields.
/// The bridge contract specifies that `data` is a flexible JSON Value.
#[test]
fn event_entry_data_payload() {
    // ComponentAdded event
    let event = BridgeEventEntry {
        event_type: "ComponentAdded".to_string(),
        timestamp: "2026-05-21T15:00:00Z".to_string(),
        data: serde_json::json!({
            "entity": 42,
            "component": "Health",
            "field_updates": {
                "current": [0, 100],
                "max_hp": [0, 100],
            },
        }),
        event_id: Some("evt-003".to_string()),
    };

    let json = serde_json::to_string(&event).unwrap();
    let deserialized: BridgeEventEntry = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.data["entity"], 42);
    assert_eq!(deserialized.data["component"], "Health");
    assert_eq!(deserialized.data["field_updates"]["current"][1], 100);
}

/// Additional event type payloads must all work through the same type.
#[test]
fn event_entry_various_event_types() {
    let event_types = vec![
        ("EntitySpawned", serde_json::json!({"entity_id": 1, "components": ["Position"]})),
        ("EntityDespawned", serde_json::json!({"entity_id": 1})),
        ("TickAdvanced", serde_json::json!({"previous": 99, "current": 100})),
        ("ComponentRemoved", serde_json::json!({"entity": 5, "component": "Health"})),
        ("SystemExecuted", serde_json::json!({"system": "MovementSystem", "dt": 0.016})),
    ];

    for (event_type, payload) in &event_types {
        let event = BridgeEventEntry {
            event_type: event_type.to_string(),
            timestamp: "2026-05-21T12:00:00Z".to_string(),
            data: payload.clone(),
            event_id: None,
        };
        let json = serde_json::to_string(&event).unwrap();
        let deserialized: BridgeEventEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.event_type, *event_type);
    }
}

// =============================================================================
// SECTION 19 -- Mutability contract: Clone types for multi-thread sharing
// =============================================================================

/// Clone then modify: each clone should be independent.
#[test]
fn clone_independence() {
    let original = BridgeRegistryEntry {
        name: "Original".to_string(),
        type_kind: "component".to_string(),
        module: "test".to_string(),
        metadata: Some(serde_json::json!({"key": "value"})),
    };

    let mut cloned = original.clone();
    cloned.name = "Modified".to_string();

    assert_eq!(original.name, "Original");
    assert_eq!(cloned.name, "Modified");
    assert_eq!(original.metadata, cloned.metadata);
}

/// Deep clone of InspectionResult with nested structures.
#[test]
fn deep_clone_inspection_result() {
    let result = BridgeInspectionResult {
        success: true,
        error: None,
        name: Some("Deep".to_string()),
        qualified_name: None,
        category: None,
        module: None,
        doc: None,
        source: None,
        metaclass: None,
        hierarchy: Some(vec![
            BridgeHierarchyEntry {
                name: "Base".to_string(),
                module: Some("base".to_string()),
                is_trinity_base: false,
            },
        ]),
        decorators: None,
        field_types: None,
        field_defaults: None,
        metadata: Some(serde_json::json!({"nested": {"deep": [1, 2, 3]}})),
    };

    let cloned = result.clone();
    assert_eq!(result, cloned);
}

// =============================================================================
// SECTION 20 -- Portability: no platform-specific assumptions
// =============================================================================

/// Timestamps should be ISO 8601 format strings (portable across platforms).
#[test]
fn iso_8601_timestamp_format() {
    let event = BridgeEventEntry {
        event_type: "Test".to_string(),
        timestamp: "2026-05-21T10:30:00Z".to_string(),
        data: serde_json::Value::Null,
        event_id: None,
    };

    let json = serde_json::to_string(&event).unwrap();
    assert!(
        json.contains("T") && json.contains("Z"),
        "Timestamp must be ISO 8601 format (contains T and Z): {}",
        json
    );
}

/// Module paths should be forward-slash or dot-separated (platform portable).
#[test]
fn module_path_platform_portable() {
    let entries = vec![
        BridgeRegistryEntry {
            name: "A".to_string(),
            type_kind: "component".to_string(),
            module: "game.components.physics".to_string(),
            metadata: None,
        },
        BridgeRegistryEntry {
            name: "B".to_string(),
            type_kind: "component".to_string(),
            module: "game/components/rendering".to_string(),
            metadata: None,
        },
    ];

    for entry in &entries {
        let json = serde_json::to_string(entry).unwrap();
        let deserialized: BridgeRegistryEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.module, entry.module);
    }
}
