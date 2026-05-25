//! Node definition commands

use crate::state::AppState;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use tauri::State;

/// Node definitions response (matches ComfyUI object_info format)
pub type NodeDefinitions = HashMap<String, NodeDefinition>;

/// Node definition
#[derive(Debug, Clone, Serialize)]
pub struct NodeDefinition {
    pub input: NodeInputs,
    pub output: Vec<String>,
    pub output_name: Vec<String>,
    pub category: String,
    pub display_name: String,
    pub description: String,
}

/// Node inputs
#[derive(Debug, Clone, Serialize)]
pub struct NodeInputs {
    pub required: HashMap<String, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optional: Option<HashMap<String, Value>>,
}

/// Search request
#[derive(Debug, Deserialize)]
pub struct SearchNodesRequest {
    pub query: String,
    pub limit: Option<usize>,
}

/// Get all node definitions (equivalent to /object_info)
#[tauri::command]
pub async fn get_object_info(state: State<'_, AppState>) -> Result<NodeDefinitions, String> {
    tracing::debug!("Getting object info");

    // TODO: Load from Bun engine
    // For now, return some example nodes

    let mut definitions = HashMap::new();

    // Example: Add node
    definitions.insert(
        "Math/Add".to_string(),
        NodeDefinition {
            input: NodeInputs {
                required: {
                    let mut map = HashMap::new();
                    map.insert(
                        "a".to_string(),
                        serde_json::json!(["FLOAT", { "default": 0, "min": -1000, "max": 1000 }]),
                    );
                    map.insert(
                        "b".to_string(),
                        serde_json::json!(["FLOAT", { "default": 0, "min": -1000, "max": 1000 }]),
                    );
                    map
                },
                optional: None,
            },
            output: vec!["FLOAT".to_string()],
            output_name: vec!["result".to_string()],
            category: "Math/Arithmetic".to_string(),
            display_name: "Add".to_string(),
            description: "Adds two numbers together".to_string(),
        },
    );

    Ok(definitions)
}

/// Get a specific node definition
#[tauri::command]
pub async fn get_node_definition(
    node_type: String,
    state: State<'_, AppState>,
) -> Result<Option<NodeDefinition>, String> {
    let definitions = get_object_info(state).await?;
    Ok(definitions.get(&node_type).cloned())
}

/// Search nodes by query
#[tauri::command]
pub async fn search_nodes(
    request: SearchNodesRequest,
    state: State<'_, AppState>,
) -> Result<NodeDefinitions, String> {
    let all_definitions = get_object_info(state).await?;
    let query = request.query.to_lowercase();
    let limit = request.limit.unwrap_or(20);

    let filtered: NodeDefinitions = all_definitions
        .into_iter()
        .filter(|(key, def)| {
            key.to_lowercase().contains(&query)
                || def.display_name.to_lowercase().contains(&query)
                || def.description.to_lowercase().contains(&query)
                || def.category.to_lowercase().contains(&query)
        })
        .take(limit)
        .collect();

    Ok(filtered)
}
