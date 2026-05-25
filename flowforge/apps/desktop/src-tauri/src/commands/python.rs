//! Python file and Trinity integration commands

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::State;

use crate::commands::trinity::SidecarState;

/// Node in the visual graph
#[derive(Debug, Serialize)]
pub struct GraphNode {
    pub id: String,
    #[serde(rename = "type")]
    pub node_type: String, // "component", "system", "resource", "event"
    pub name: String,
    pub position: (f64, f64),
    pub data: Value,
    pub source: SourceLocation,
}

#[derive(Debug, Serialize)]
pub struct SourceLocation {
    pub file: String,
    pub line: u32,
}

#[derive(Debug, Serialize)]
pub struct GraphEdge {
    pub id: String,
    pub source: String,
    pub target: String,
    #[serde(rename = "type")]
    pub edge_type: String, // "reference", "inheritance", "query"
}

#[derive(Debug, Serialize)]
pub struct NodeGraph {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

/// Parse a Python file by delegating to the Python sidecar
#[tauri::command]
pub async fn parse_python_file(
    sidecar: State<'_, SidecarState>,
    path: String,
) -> Result<NodeGraph, String> {
    // Validate file extension
    if !path.ends_with(".py") {
        let ext = std::path::Path::new(&path)
            .extension()
            .map(|e| format!(".{}", e.to_string_lossy()))
            .unwrap_or_else(|| "(no extension)".to_string());
        return Err(format!(
            "Cannot parse \"{}\": only Python (.py) files are supported, got {} file",
            std::path::Path::new(&path)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| path.clone()),
            ext
        ));
    }

    // Check file exists before sending to sidecar
    if !std::path::Path::new(&path).exists() {
        return Err(format!("File not found: {}", path));
    }

    let params = serde_json::json!({ "path": path });
    let result = sidecar.send_request("parse_python_file", params)?;

    // Check success
    let success = result.get("success").and_then(|v| v.as_bool()).unwrap_or(false);
    if !success {
        let errors = result
            .get("errors")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|e| e.as_str())
                    .collect::<Vec<_>>()
                    .join("; ")
            })
            .unwrap_or_else(|| "Unknown parse error".to_string());
        return Err(errors);
    }

    // Graph data is nested under "graph" key
    let graph_data = result.get("graph").unwrap_or(&result);

    // Parse nodes from response
    let nodes: Vec<GraphNode> = graph_data
        .get("nodes")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|n| {
                    let id = n.get("id")?.as_str()?.to_string();
                    let node_type = n
                        .get("type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("component")
                        .to_string();
                    let name = n.get("name")?.as_str()?.to_string();
                    let pos = n.get("position");
                    let x = pos
                        .and_then(|p| p.get(0).or_else(|| p.get("x")))
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    let y = pos
                        .and_then(|p| p.get(1).or_else(|| p.get("y")))
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    let data = n.get("data").cloned().unwrap_or(Value::Null);
                    let file = n
                        .get("source")
                        .and_then(|s| s.get("file"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let line = n
                        .get("source")
                        .and_then(|s| s.get("line"))
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as u32;
                    Some(GraphNode {
                        id,
                        node_type,
                        name,
                        position: (x, y),
                        data,
                        source: SourceLocation { file, line },
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    // Parse edges from response
    let edges: Vec<GraphEdge> = graph_data
        .get("edges")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|e| {
                    let id = e.get("id")?.as_str()?.to_string();
                    let source = e.get("source")?.as_str()?.to_string();
                    let target = e.get("target")?.as_str()?.to_string();
                    let edge_type = e
                        .get("type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("reference")
                        .to_string();
                    Some(GraphEdge {
                        id,
                        source,
                        target,
                        edge_type,
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    Ok(NodeGraph { nodes, edges })
}

/// Read a Python file's content
#[tauri::command]
pub async fn read_python_file(path: String) -> Result<PythonFileContent, String> {
    let content =
        std::fs::read_to_string(&path).map_err(|e| format!("Failed to read file: {}", e))?;
    Ok(PythonFileContent { path, content })
}

#[derive(Debug, Serialize)]
pub struct PythonFileContent {
    pub path: String,
    pub content: String,
}

/// Write content to a Python file
#[tauri::command]
pub async fn write_python_file(path: String, content: String) -> Result<bool, String> {
    std::fs::write(&path, content).map_err(|e| format!("Failed to write file: {}", e))?;
    Ok(true)
}

/// Trinity node type definitions
#[derive(Debug, Serialize)]
pub struct TrinityNodeTypes {
    pub component: NodeTypeDefinition,
    pub system: NodeTypeDefinition,
    pub resource: NodeTypeDefinition,
    pub event: NodeTypeDefinition,
}

#[derive(Debug, Serialize)]
pub struct NodeTypeDefinition {
    pub name: String,
    pub description: String,
    pub category: String,
    pub decorator: String,
}

/// Get Trinity-specific node type definitions
#[tauri::command]
pub async fn get_trinity_node_types() -> Result<TrinityNodeTypes, String> {
    Ok(TrinityNodeTypes {
        component: NodeTypeDefinition {
            name: "Component".to_string(),
            description: "Data container for entities".to_string(),
            category: "ECS".to_string(),
            decorator: "@component".to_string(),
        },
        system: NodeTypeDefinition {
            name: "System".to_string(),
            description: "Logic that processes components".to_string(),
            category: "ECS".to_string(),
            decorator: "@system".to_string(),
        },
        resource: NodeTypeDefinition {
            name: "Resource".to_string(),
            description: "Singleton global state".to_string(),
            category: "ECS".to_string(),
            decorator: "@resource".to_string(),
        },
        event: NodeTypeDefinition {
            name: "Event".to_string(),
            description: "Message for system communication".to_string(),
            category: "ECS".to_string(),
            decorator: "@event".to_string(),
        },
    })
}
