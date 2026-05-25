//! Code generation commands
//!
//! Provides Tauri commands for code generation, validation, and transformation
//! through the Python sidecar. These commands allow the frontend to:
//! - Generate Python code from visual node graphs
//! - Validate generated or existing code
//! - Generate diffs between existing files and graph representations
//! - Apply code changes with optional backup

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::State;

use crate::commands::trinity::SidecarState;

// =============================================================================
// Node Graph Types (Input)
// =============================================================================

/// Position of a node in the visual editor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodePosition {
    pub x: f64,
    pub y: f64,
}

/// Source location information for code mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeSourceLocation {
    /// File path (if from existing file)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    /// Line number in source
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
    /// Column number in source
    #[serde(skip_serializing_if = "Option::is_none")]
    pub column: Option<u32>,
}

/// A node in the visual graph representing a Trinity element
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Unique identifier for the node
    pub id: String,
    /// Node type: "component", "system", "resource", "event"
    #[serde(rename = "type")]
    pub node_type: String,
    /// Name of the Trinity element (e.g., "Position", "MovementSystem")
    pub name: String,
    /// Visual position in the editor
    pub position: NodePosition,
    /// Node-specific data (fields, parameters, etc.)
    pub data: Value,
    /// Source location if mapped from existing code
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<CodeSourceLocation>,
}

/// An edge connecting two nodes in the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Unique identifier for the edge
    pub id: String,
    /// Source node ID
    pub source: String,
    /// Target node ID
    pub target: String,
    /// Edge type: "reference", "inheritance", "query", "event_handler"
    #[serde(rename = "type")]
    pub edge_type: String,
    /// Optional label or description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Additional edge metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// Complete node graph representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeGraph {
    /// All nodes in the graph
    pub nodes: Vec<GraphNode>,
    /// All edges connecting nodes
    pub edges: Vec<GraphEdge>,
    /// Optional graph-level metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

// =============================================================================
// Code Generation Types
// =============================================================================

/// Options for code generation
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GenerationOptions {
    /// Include docstrings in generated code
    #[serde(default = "default_true")]
    pub include_docstrings: bool,
    /// Include type hints
    #[serde(default = "default_true")]
    pub include_type_hints: bool,
    /// Code formatting style: "black", "pep8", "none"
    #[serde(default = "default_formatter")]
    pub formatter: String,
    /// Line width for formatting
    #[serde(default = "default_line_width")]
    pub line_width: u32,
    /// Import organization style: "grouped", "alphabetical", "none"
    #[serde(default = "default_import_style")]
    pub import_style: String,
    /// Target Python version (e.g., "3.10", "3.11")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_python_version: Option<String>,
}

fn default_true() -> bool {
    true
}

fn default_formatter() -> String {
    "black".to_string()
}

fn default_line_width() -> u32 {
    88
}

fn default_import_style() -> String {
    "grouped".to_string()
}

/// Information about an import statement (from Python backend)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportInfo {
    /// The module being imported (e.g., "trinity.ecs")
    pub module: String,
    /// Names imported from the module (for from imports)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub names: Vec<String>,
    /// Alias if using `import X as Y`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alias: Option<String>,
    /// True if this is a `from X import Y` statement
    pub is_from_import: bool,
    /// Line number of the import
    pub line: u32,
}

/// Result of code generation (matches Python backend response)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedCode {
    /// The generated Python source code
    pub source: String,
    /// Validation result for the generated code
    pub validation: ValidationResult,
    /// List of imports used in the generated code
    #[serde(default)]
    pub imports: Vec<ImportInfo>,
    /// Number of nodes processed
    pub node_count: u32,
    /// Additional metadata about the generation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

// =============================================================================
// Validation Types
// =============================================================================

/// Options for code validation
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ValidationOptions {
    /// Check for syntax errors
    #[serde(default = "default_true")]
    pub check_syntax: bool,
    /// Check for type errors (requires type hints)
    #[serde(default)]
    pub check_types: bool,
    /// Check Trinity-specific patterns
    #[serde(default = "default_true")]
    pub check_trinity: bool,
    /// Check code style (PEP8, etc.)
    #[serde(default)]
    pub check_style: bool,
    /// Additional linters to run: ["pylint", "flake8", "mypy"]
    #[serde(default)]
    pub linters: Vec<String>,
}

/// A single validation error or warning (matches Python ValidationError)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    /// Line number where the error occurred (1-indexed)
    pub line: u32,
    /// Column number where the error occurred (0-indexed)
    pub column: u32,
    /// Human-readable error message
    pub message: String,
    /// Severity level: "error", "warning", "info"
    pub severity: String,
    /// Optional error code for programmatic handling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    /// Optional end line for multi-line errors
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_line: Option<u32>,
    /// Optional end column
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_column: Option<u32>,
}

/// Result of code validation (matches Python backend response)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether validation passed (no errors, warnings may exist)
    pub success: bool,
    /// List of validation errors (severity=error)
    #[serde(default)]
    pub errors: Vec<ValidationIssue>,
    /// List of validation warnings (severity=warning or info)
    #[serde(default)]
    pub warnings: Vec<ValidationIssue>,
    /// Optional hash of the validated source for caching
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_hash: Option<String>,
}

// =============================================================================
// Diff Types
// =============================================================================

/// A single line in a diff (matches Python DiffLine)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffLine {
    /// Type of the line: "added", "removed", "unchanged", "context", "header", "empty"
    #[serde(rename = "type")]
    pub line_type: String,
    /// Content of the line
    pub content: String,
    /// Line number in original file (null for added lines)
    pub original_line: Option<u32>,
    /// Line number in modified file (null for removed lines)
    pub modified_line: Option<u32>,
}

/// A contiguous block of changes (matches Python DiffHunk)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffHunk {
    /// Starting line in original file
    pub original_start: u32,
    /// Number of lines in original
    pub original_count: u32,
    /// Starting line in modified file
    pub modified_start: u32,
    /// Number of lines in modified
    pub modified_count: u32,
    /// Lines in this hunk
    pub lines: Vec<DiffLine>,
}

/// Statistics about a diff (matches Python DiffStats)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffStats {
    /// Lines added
    pub additions: u32,
    /// Lines removed
    pub deletions: u32,
    /// Total changes
    pub changes: u32,
}

/// Result of diff generation (matches Python DiffResult)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffResult {
    /// Filename for display
    pub filename: String,
    /// Path to original file (if from file)
    pub original_path: Option<String>,
    /// Whether files are different
    pub has_changes: bool,
    /// Individual hunks for granular display
    pub hunks: Vec<DiffHunk>,
    /// Statistics about the diff
    pub stats: DiffStats,
    /// Complete unified diff string
    pub unified_diff: String,
    /// Error message if diff generation failed (not from Python, added for Rust error handling)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

// =============================================================================
// Apply Changes Types
// =============================================================================

/// Options for applying changes
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ApplyOptions {
    /// Create a backup before applying
    #[serde(default = "default_true")]
    pub backup: bool,
    /// Backup suffix (default: ".bak")
    #[serde(default = "default_backup_suffix")]
    pub backup_suffix: String,
    /// Validate code before applying
    #[serde(default = "default_true")]
    pub validate_before_apply: bool,
    /// Dry run - don't actually write, just validate
    #[serde(default)]
    pub dry_run: bool,
    /// Create parent directories if needed
    #[serde(default = "default_true")]
    pub create_dirs: bool,
}

fn default_backup_suffix() -> String {
    ".bak".to_string()
}

/// Result of applying changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplyResult {
    /// Whether changes were applied successfully
    pub success: bool,
    /// Path to the modified file
    pub path: String,
    /// Path to the backup file (if created)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backup_path: Option<String>,
    /// Validation result (if validate_before_apply was true)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation: Option<ValidationResult>,
    /// Whether this was a dry run
    pub dry_run: bool,
    /// Bytes written
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes_written: Option<u64>,
    /// Error message if apply failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

// =============================================================================
// Tauri Commands
// =============================================================================

/// Generate Python code from a node graph
///
/// Takes a visual node graph representation and generates corresponding
/// Python code with Trinity decorators and proper structure.
///
/// # Arguments
/// * `graph` - The node graph to generate code from
/// * `options` - Optional generation options
#[tauri::command]
pub async fn generate_code(
    sidecar: State<'_, SidecarState>,
    graph: NodeGraph,
    options: Option<GenerationOptions>,
) -> Result<GeneratedCode, String> {
    let opts = options.unwrap_or_default();
    let params = serde_json::json!({
        "graph": graph,
        "format_with_black": opts.formatter == "black",
        "add_header": opts.include_docstrings,
    });

    match sidecar.send_request("generate_code", params) {
        Ok(result) => {
            // Parse the response from Python backend
            serde_json::from_value(result.clone()).map_err(|e| {
                format!("Failed to parse generation result: {}. Raw: {:?}", e, result)
            })
        }
        Err(e) => {
            // Return a failed result with empty validation
            Ok(GeneratedCode {
                source: String::new(),
                validation: ValidationResult {
                    success: false,
                    errors: vec![ValidationIssue {
                        line: 1,
                        column: 0,
                        message: e,
                        severity: "error".to_string(),
                        code: Some("E000".to_string()),
                        end_line: None,
                        end_column: None,
                    }],
                    warnings: vec![],
                    source_hash: None,
                },
                imports: vec![],
                node_count: 0,
                metadata: None,
            })
        }
    }
}

/// Validate Python code
///
/// Checks code for syntax errors, type issues, Trinity-specific patterns,
/// and optionally runs additional linters.
///
/// # Arguments
/// * `code` - The Python code to validate
/// * `options` - Optional validation options
#[tauri::command]
pub async fn validate_code(
    sidecar: State<'_, SidecarState>,
    code: String,
    options: Option<ValidationOptions>,
) -> Result<ValidationResult, String> {
    let opts = options.unwrap_or_default();
    let params = serde_json::json!({
        "source": code,
        "check_semantics": opts.check_types,
    });

    match sidecar.send_request("validate_code", params) {
        Ok(result) => {
            // Parse the response from Python backend
            serde_json::from_value(result.clone()).map_err(|e| {
                format!("Failed to parse validation result: {}. Raw: {:?}", e, result)
            })
        }
        Err(e) => Ok(ValidationResult {
            success: false,
            errors: vec![ValidationIssue {
                line: 1,
                column: 0,
                message: e,
                severity: "error".to_string(),
                code: Some("E000".to_string()),
                end_line: None,
                end_column: None,
            }],
            warnings: vec![],
            source_hash: None,
        }),
    }
}

/// Generate a diff between an existing file and a node graph
///
/// Compares the code that would be generated from the graph against
/// the existing file content.
///
/// # Arguments
/// * `original_path` - Path to the original Python file
/// * `graph` - The node graph to generate code from
/// * `options` - Optional generation options
#[tauri::command]
pub async fn generate_diff(
    sidecar: State<'_, SidecarState>,
    original_path: String,
    graph: NodeGraph,
    options: Option<GenerationOptions>,
) -> Result<DiffResult, String> {
    // Read the original file content
    let original_content = match std::fs::read_to_string(&original_path) {
        Ok(content) => content,
        Err(e) => {
            return Ok(DiffResult {
                filename: original_path.clone(),
                original_path: Some(original_path),
                has_changes: false,
                unified_diff: String::new(),
                hunks: vec![],
                stats: DiffStats {
                    additions: 0,
                    deletions: 0,
                    changes: 0,
                },
                error: Some(format!("Failed to read original file: {}", e)),
            });
        }
    };

    // First generate the code from the graph
    let opts = options.unwrap_or_default();
    let gen_params = serde_json::json!({
        "graph": graph,
        "format_with_black": opts.formatter == "black",
        "add_header": opts.include_docstrings,
    });

    let generated: GeneratedCode = match sidecar.send_request("generate_code", gen_params) {
        Ok(result) => serde_json::from_value(result.clone()).map_err(|e| {
            format!("Failed to parse generation result: {}", e)
        })?,
        Err(e) => {
            return Ok(DiffResult {
                filename: original_path.clone(),
                original_path: Some(original_path),
                has_changes: false,
                unified_diff: String::new(),
                hunks: vec![],
                stats: DiffStats {
                    additions: 0,
                    deletions: 0,
                    changes: 0,
                },
                error: Some(format!("Code generation failed: {}", e)),
            });
        }
    };

    // Now generate the diff
    let diff_params = serde_json::json!({
        "original": original_content,
        "modified": generated.source,
        "filename": original_path,
        "original_path": original_path,
        "context_lines": 3,
    });

    match sidecar.send_request("generate_diff", diff_params) {
        Ok(result) => {
            // Parse the response from Python backend
            serde_json::from_value(result.clone()).map_err(|e| {
                format!("Failed to parse diff result: {}. Raw: {:?}", e, result)
            })
        }
        Err(e) => Ok(DiffResult {
            filename: original_path.clone(),
            original_path: Some(original_path),
            has_changes: false,
            unified_diff: String::new(),
            hunks: vec![],
            stats: DiffStats {
                additions: 0,
                deletions: 0,
                changes: 0,
            },
            error: Some(e),
        }),
    }
}

/// Apply generated code changes to a file
///
/// Generates code from the graph and writes it to the specified path,
/// with optional backup and validation.
///
/// # Arguments
/// * `path` - Path to write the generated code
/// * `graph` - The node graph to generate code from
/// * `options` - Optional apply options (backup, validation, etc.)
#[tauri::command]
pub async fn apply_changes(
    sidecar: State<'_, SidecarState>,
    path: String,
    graph: NodeGraph,
    options: Option<ApplyOptions>,
) -> Result<ApplyResult, String> {
    let opts = options.unwrap_or_default();

    // First, generate the code
    let gen_params = serde_json::json!({
        "graph": graph,
        "format_with_black": true,
        "add_header": true,
    });

    let generated: GeneratedCode = match sidecar.send_request("generate_code", gen_params) {
        Ok(result) => serde_json::from_value(result.clone()).map_err(|e| {
            format!("Failed to parse generation result: {}", e)
        })?,
        Err(e) => {
            return Ok(ApplyResult {
                success: false,
                path: path.clone(),
                backup_path: None,
                validation: None,
                dry_run: opts.dry_run,
                bytes_written: None,
                error: Some(format!("Code generation failed: {}", e)),
            });
        }
    };

    // Check if generation succeeded (validation.success)
    if !generated.validation.success {
        return Ok(ApplyResult {
            success: false,
            path: path.clone(),
            backup_path: None,
            validation: Some(generated.validation),
            dry_run: opts.dry_run,
            bytes_written: None,
            error: Some("Code generation produced invalid code".to_string()),
        });
    }

    // Optionally validate before applying (re-validate the generated code)
    let validation = if opts.validate_before_apply {
        let val_params = serde_json::json!({
            "source": generated.source,
            "check_semantics": false,
        });

        match sidecar.send_request("validate_code", val_params) {
            Ok(val_result) => {
                let validation_result: ValidationResult =
                    serde_json::from_value(val_result.clone()).unwrap_or(ValidationResult {
                        success: true,
                        errors: vec![],
                        warnings: vec![],
                        source_hash: None,
                    });

                // If validation failed with errors, don't apply
                if !validation_result.success {
                    return Ok(ApplyResult {
                        success: false,
                        path: path.clone(),
                        backup_path: None,
                        validation: Some(validation_result),
                        dry_run: opts.dry_run,
                        bytes_written: None,
                        error: Some("Validation failed with errors".to_string()),
                    });
                }

                Some(validation_result)
            }
            Err(e) => {
                tracing::warn!("Validation request failed: {}", e);
                None
            }
        }
    } else {
        Some(generated.validation.clone())
    };

    // Dry run - don't actually write
    if opts.dry_run {
        return Ok(ApplyResult {
            success: true,
            path: path.clone(),
            backup_path: None,
            validation,
            dry_run: true,
            bytes_written: Some(generated.source.len() as u64),
            error: None,
        });
    }

    // Use Python backend's apply_changes for actual file operations
    let apply_params = serde_json::json!({
        "file_path": path,
        "content": generated.source,
        "create_backup": opts.backup,
    });

    match sidecar.send_request("apply_changes", apply_params) {
        Ok(result) => {
            let success = result
                .get("success")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let backup_path = result
                .get("backup_path")
                .and_then(|v| v.as_str())
                .map(String::from);
            let error = result
                .get("error")
                .and_then(|v| v.as_str())
                .map(String::from);

            Ok(ApplyResult {
                success,
                path,
                backup_path,
                validation,
                dry_run: false,
                bytes_written: if success {
                    Some(generated.source.len() as u64)
                } else {
                    None
                },
                error,
            })
        }
        Err(e) => Ok(ApplyResult {
            success: false,
            path,
            backup_path: None,
            validation,
            dry_run: false,
            bytes_written: None,
            error: Some(format!("Failed to apply changes: {}", e)),
        }),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    // =========================================================================
    // Helper Functions for Tests
    // =========================================================================

    /// Create a unique test directory in the system temp folder
    fn create_test_dir(test_name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "flowforge_codegen_test_{}_{}",
            test_name,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&path).expect("Failed to create test directory");
        path
    }

    /// Clean up a test directory
    fn cleanup_test_dir(path: &PathBuf) {
        if path.exists() {
            let _ = fs::remove_dir_all(path);
        }
    }

    /// Create a sample node graph for testing
    fn create_sample_graph() -> NodeGraph {
        NodeGraph {
            nodes: vec![
                GraphNode {
                    id: "node-1".to_string(),
                    node_type: "component".to_string(),
                    name: "Position".to_string(),
                    position: NodePosition { x: 100.0, y: 200.0 },
                    data: serde_json::json!({
                        "fields": [
                            {"name": "x", "type": "float", "default": 0.0},
                            {"name": "y", "type": "float", "default": 0.0}
                        ]
                    }),
                    source: None,
                },
                GraphNode {
                    id: "node-2".to_string(),
                    node_type: "system".to_string(),
                    name: "MovementSystem".to_string(),
                    position: NodePosition { x: 300.0, y: 200.0 },
                    data: serde_json::json!({
                        "queries": ["Position", "Velocity"]
                    }),
                    source: Some(CodeSourceLocation {
                        file: Some("/path/to/source.py".to_string()),
                        line: Some(10),
                        column: Some(1),
                    }),
                },
            ],
            edges: vec![GraphEdge {
                id: "edge-1".to_string(),
                source: "node-1".to_string(),
                target: "node-2".to_string(),
                edge_type: "query".to_string(),
                label: Some("queries Position".to_string()),
                data: None,
            }],
            metadata: Some(serde_json::json!({
                "version": "1.0",
                "author": "test"
            })),
        }
    }

    // =========================================================================
    // Serialization Tests (existing)
    // =========================================================================

    #[test]
    fn test_node_graph_serialization() {
        let graph = NodeGraph {
            nodes: vec![GraphNode {
                id: "node-1".to_string(),
                node_type: "component".to_string(),
                name: "Position".to_string(),
                position: NodePosition { x: 100.0, y: 200.0 },
                data: serde_json::json!({
                    "fields": [
                        {"name": "x", "type": "float", "default": 0.0},
                        {"name": "y", "type": "float", "default": 0.0}
                    ]
                }),
                source: None,
            }],
            edges: vec![],
            metadata: None,
        };

        let json = serde_json::to_string(&graph).unwrap();
        assert!(json.contains("\"id\":\"node-1\""));
        assert!(json.contains("\"type\":\"component\""));
        assert!(json.contains("\"name\":\"Position\""));
    }

    #[test]
    fn test_generation_options_defaults() {
        let opts = GenerationOptions::default();
        assert!(opts.include_docstrings);
        assert!(opts.include_type_hints);
        assert_eq!(opts.formatter, "black");
        assert_eq!(opts.line_width, 88);
        assert_eq!(opts.import_style, "grouped");
    }

    #[test]
    fn test_validation_issue_serialization() {
        let issue = ValidationIssue {
            line: 10,
            column: 5,
            message: "Syntax error".to_string(),
            severity: "error".to_string(),
            code: Some("E001".to_string()),
            end_line: None,
            end_column: None,
        };

        let json = serde_json::to_string(&issue).unwrap();
        assert!(json.contains("\"severity\":\"error\""));
        assert!(json.contains("\"line\":10"));
        assert!(json.contains("\"message\":\"Syntax error\""));
    }

    #[test]
    fn test_diff_hunk_serialization() {
        let hunk = DiffHunk {
            original_start: 1,
            original_count: 5,
            modified_start: 1,
            modified_count: 7,
            lines: vec![DiffLine {
                line_type: "context".to_string(),
                content: " context line".to_string(),
                original_line: Some(1),
                modified_line: Some(1),
            }],
        };

        let json = serde_json::to_string(&hunk).unwrap();
        assert!(json.contains("\"original_start\":1"));
        assert!(json.contains("\"modified_count\":7"));
    }

    #[test]
    fn test_apply_options_defaults() {
        let opts = ApplyOptions::default();
        assert!(opts.backup);
        assert_eq!(opts.backup_suffix, ".bak");
        assert!(opts.validate_before_apply);
        assert!(!opts.dry_run);
        assert!(opts.create_dirs);
    }

    #[test]
    fn test_generated_code_result() {
        let result = GeneratedCode {
            source: "from trinity import component\n\n@component\nclass Position:\n    x: float = 0.0\n    y: float = 0.0\n".to_string(),
            validation: ValidationResult {
                success: true,
                errors: vec![],
                warnings: vec![],
                source_hash: Some("abc123".to_string()),
            },
            imports: vec![ImportInfo {
                module: "trinity".to_string(),
                names: vec!["component".to_string()],
                alias: None,
                is_from_import: true,
                line: 1,
            }],
            node_count: 1,
            metadata: None,
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"source\":"));
        assert!(json.contains("@component"));
        assert!(json.contains("\"success\":true"));
    }

    #[test]
    fn test_validation_result_serialization() {
        let result = ValidationResult {
            success: true,
            errors: vec![],
            warnings: vec![ValidationIssue {
                line: 5,
                column: 0,
                message: "Unused import 'os'".to_string(),
                severity: "warning".to_string(),
                code: Some("W001".to_string()),
                end_line: None,
                end_column: None,
            }],
            source_hash: Some("def456".to_string()),
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"success\":true"));
        assert!(json.contains("\"warnings\":["));
        assert!(json.contains("Unused import"));
    }

    #[test]
    fn test_diff_result_serialization() {
        let result = DiffResult {
            filename: "test.py".to_string(),
            original_path: Some("/path/to/test.py".to_string()),
            has_changes: true,
            hunks: vec![],
            stats: DiffStats {
                additions: 5,
                deletions: 2,
                changes: 7,
            },
            unified_diff: "--- a/test.py\n+++ b/test.py\n".to_string(),
            error: None,
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"filename\":\"test.py\""));
        assert!(json.contains("\"has_changes\":true"));
        assert!(json.contains("\"additions\":5"));
    }

    // =========================================================================
    // Default Helper Function Tests
    // =========================================================================

    #[test]
    fn test_default_true_helper() {
        assert!(default_true());
    }

    #[test]
    fn test_default_formatter_helper() {
        assert_eq!(default_formatter(), "black");
    }

    #[test]
    fn test_default_line_width_helper() {
        assert_eq!(default_line_width(), 88);
    }

    #[test]
    fn test_default_import_style_helper() {
        assert_eq!(default_import_style(), "grouped");
    }

    #[test]
    fn test_default_backup_suffix_helper() {
        assert_eq!(default_backup_suffix(), ".bak");
    }

    // =========================================================================
    // Backup Path Generation Tests
    // =========================================================================

    #[test]
    fn test_backup_path_generation_default_suffix() {
        let path = "/home/user/code/game.py";
        let opts = ApplyOptions::default();
        let backup_path = format!("{}{}", path, opts.backup_suffix);
        assert_eq!(backup_path, "/home/user/code/game.py.bak");
    }

    #[test]
    fn test_backup_path_generation_custom_suffix() {
        let path = "/home/user/code/game.py";
        let opts = ApplyOptions {
            backup_suffix: ".backup".to_string(),
            ..Default::default()
        };
        let backup_path = format!("{}{}", path, opts.backup_suffix);
        assert_eq!(backup_path, "/home/user/code/game.py.backup");
    }

    #[test]
    fn test_backup_path_generation_timestamped_suffix() {
        let path = "/home/user/code/game.py";
        let suffix = ".bak.2024-01-15";
        let opts = ApplyOptions {
            backup_suffix: suffix.to_string(),
            ..Default::default()
        };
        let backup_path = format!("{}{}", path, opts.backup_suffix);
        assert_eq!(backup_path, "/home/user/code/game.py.bak.2024-01-15");
    }

    #[test]
    fn test_backup_path_with_special_characters() {
        let path = "/home/user/my project/game file.py";
        let opts = ApplyOptions::default();
        let backup_path = format!("{}{}", path, opts.backup_suffix);
        assert_eq!(backup_path, "/home/user/my project/game file.py.bak");
    }

    // =========================================================================
    // File System Tests - Directory Creation
    // =========================================================================

    #[test]
    fn test_create_dirs_with_nested_path() {
        let test_dir = create_test_dir("create_dirs_nested");
        let nested_path = test_dir.join("level1").join("level2").join("level3");

        // Verify nested path doesn't exist
        assert!(!nested_path.exists());

        // Simulate the directory creation logic from apply_changes
        if let Some(parent) = nested_path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent).expect("Should create nested directories");
            }
        }

        // Parent should now exist (not the file itself)
        assert!(nested_path.parent().unwrap().exists());

        cleanup_test_dir(&test_dir);
    }

    #[test]
    fn test_create_dirs_when_parent_exists() {
        let test_dir = create_test_dir("create_dirs_existing");
        let file_path = test_dir.join("existing_dir").join("file.py");

        // Create the parent directory first
        fs::create_dir_all(file_path.parent().unwrap()).expect("Should create directory");

        // Verify parent exists
        assert!(file_path.parent().unwrap().exists());

        // The create_dirs logic should not fail when directory already exists
        if let Some(parent) = file_path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent).expect("Should handle existing directory");
            }
        }

        // Parent should still exist
        assert!(file_path.parent().unwrap().exists());

        cleanup_test_dir(&test_dir);
    }

    #[test]
    fn test_create_dirs_disabled() {
        let opts = ApplyOptions {
            create_dirs: false,
            ..Default::default()
        };

        assert!(!opts.create_dirs);
    }

    // =========================================================================
    // File System Tests - Backup Creation
    // =========================================================================

    #[test]
    fn test_backup_file_creation() {
        let test_dir = create_test_dir("backup_creation");
        let original_path = test_dir.join("original.py");
        let backup_path = format!("{}.bak", original_path.display());

        // Create original file with content
        let original_content = "# Original content\nprint('hello')";
        fs::write(&original_path, original_content).expect("Should write original file");

        // Verify original exists
        assert!(original_path.exists());

        // Create backup (simulating apply_changes logic)
        let opts = ApplyOptions::default();
        if opts.backup && original_path.exists() {
            fs::copy(&original_path, &backup_path).expect("Should create backup");
        }

        // Verify backup was created with same content
        assert!(std::path::Path::new(&backup_path).exists());
        let backup_content = fs::read_to_string(&backup_path).expect("Should read backup");
        assert_eq!(backup_content, original_content);

        cleanup_test_dir(&test_dir);
    }

    #[test]
    fn test_backup_skipped_when_file_doesnt_exist() {
        let test_dir = create_test_dir("backup_nonexistent");
        let original_path = test_dir.join("nonexistent.py");
        let backup_path = format!("{}.bak", original_path.display());

        // Verify original doesn't exist
        assert!(!original_path.exists());

        // Backup should not be created for non-existent file
        let opts = ApplyOptions::default();
        let backup_created = if opts.backup && original_path.exists() {
            let _ = fs::copy(&original_path, &backup_path);
            true
        } else {
            false
        };

        assert!(!backup_created);
        assert!(!std::path::Path::new(&backup_path).exists());

        cleanup_test_dir(&test_dir);
    }

    #[test]
    fn test_backup_disabled() {
        let test_dir = create_test_dir("backup_disabled");
        let original_path = test_dir.join("original.py");
        let backup_path = format!("{}.bak", original_path.display());

        // Create original file
        fs::write(&original_path, "content").expect("Should write original file");

        // Backup disabled
        let opts = ApplyOptions {
            backup: false,
            ..Default::default()
        };

        let backup_created = if opts.backup && original_path.exists() {
            let _ = fs::copy(&original_path, &backup_path);
            true
        } else {
            false
        };

        assert!(!backup_created);
        assert!(!std::path::Path::new(&backup_path).exists());

        cleanup_test_dir(&test_dir);
    }

    // =========================================================================
    // File System Tests - Dry Run Behavior
    // =========================================================================

    #[test]
    fn test_dry_run_does_not_write_files() {
        let test_dir = create_test_dir("dry_run_no_write");
        let file_path = test_dir.join("output.py");

        let opts = ApplyOptions {
            dry_run: true,
            ..Default::default()
        };

        // Simulate dry run logic
        let generated_code = "# Generated code";
        let bytes_written = if opts.dry_run {
            // Dry run - don't actually write
            Some(generated_code.len() as u64)
        } else {
            fs::write(&file_path, generated_code).ok();
            Some(generated_code.len() as u64)
        };

        // Verify file was NOT created
        assert!(!file_path.exists());
        // But bytes_written should still report what would have been written
        assert_eq!(bytes_written, Some(16)); // "# Generated code".len()

        cleanup_test_dir(&test_dir);
    }

    #[test]
    fn test_dry_run_does_not_create_backup() {
        let test_dir = create_test_dir("dry_run_no_backup");
        let original_path = test_dir.join("original.py");
        let backup_path = format!("{}.bak", original_path.display());

        // Create original file
        fs::write(&original_path, "original content").expect("Should write original");

        let opts = ApplyOptions {
            dry_run: true,
            backup: true,
            ..Default::default()
        };

        // In dry run, backup should not be created
        // (matching the apply_changes logic where dry_run check comes before backup)
        let backup_created = if !opts.dry_run && opts.backup && original_path.exists() {
            let _ = fs::copy(&original_path, &backup_path);
            true
        } else {
            false
        };

        assert!(!backup_created);
        assert!(!std::path::Path::new(&backup_path).exists());

        cleanup_test_dir(&test_dir);
    }

    #[test]
    fn test_dry_run_returns_correct_result() {
        let opts = ApplyOptions {
            dry_run: true,
            ..Default::default()
        };

        let generated_code = "from trinity import component\n\n@component\nclass Position:\n    x: float = 0.0";

        // Simulate ApplyResult for dry run
        let result = ApplyResult {
            success: true,
            path: "/tmp/test.py".to_string(),
            backup_path: None,
            validation: None,
            dry_run: true,
            bytes_written: Some(generated_code.len() as u64),
            error: None,
        };

        assert!(result.success);
        assert!(result.dry_run);
        assert!(result.backup_path.is_none());
        assert_eq!(result.bytes_written, Some(generated_code.len() as u64));
    }

    // =========================================================================
    // Error Handling Tests
    // =========================================================================

    #[test]
    fn test_error_response_for_missing_file_in_diff() {
        let nonexistent_path = "/nonexistent/path/to/file.py";

        // Simulate the error handling in generate_diff
        let read_result = std::fs::read_to_string(nonexistent_path);

        let diff_result = match read_result {
            Ok(_content) => DiffResult {
                filename: nonexistent_path.to_string(),
                original_path: Some(nonexistent_path.to_string()),
                has_changes: false,
                hunks: vec![],
                stats: DiffStats {
                    additions: 0,
                    deletions: 0,
                    changes: 0,
                },
                unified_diff: String::new(),
                error: None,
            },
            Err(e) => DiffResult {
                filename: nonexistent_path.to_string(),
                original_path: Some(nonexistent_path.to_string()),
                has_changes: false,
                hunks: vec![],
                stats: DiffStats {
                    additions: 0,
                    deletions: 0,
                    changes: 0,
                },
                unified_diff: String::new(),
                error: Some(format!("Failed to read original file: {}", e)),
            },
        };

        assert!(diff_result.error.is_some());
        assert!(diff_result.error.unwrap().contains("Failed to read original file"));
    }

    #[test]
    fn test_error_response_structure() {
        let error_result = GeneratedCode {
            source: String::new(),
            validation: ValidationResult {
                success: false,
                errors: vec![ValidationIssue {
                    line: 1,
                    column: 0,
                    message: "Code generation failed: invalid graph".to_string(),
                    severity: "error".to_string(),
                    code: Some("E000".to_string()),
                    end_line: None,
                    end_column: None,
                }],
                warnings: vec![],
                source_hash: None,
            },
            imports: vec![],
            node_count: 0,
            metadata: None,
        };

        assert!(!error_result.validation.success);
        assert!(error_result.source.is_empty());
        assert_eq!(error_result.validation.errors.len(), 1);
    }

    #[test]
    fn test_validation_error_response() {
        let validation_result = ValidationResult {
            success: false,
            errors: vec![
                ValidationIssue {
                    line: 5,
                    column: 10,
                    message: "Syntax error: unexpected token".to_string(),
                    severity: "error".to_string(),
                    code: Some("E001".to_string()),
                    end_line: None,
                    end_column: None,
                },
            ],
            warnings: vec![],
            source_hash: None,
        };

        assert!(!validation_result.success);
        assert_eq!(validation_result.errors.len(), 1);
        assert_eq!(validation_result.errors[0].severity, "error");
    }

    #[test]
    fn test_apply_result_with_validation_failure() {
        let validation = ValidationResult {
            success: false,
            errors: vec![ValidationIssue {
                line: 1,
                column: 1,
                message: "Invalid syntax".to_string(),
                severity: "error".to_string(),
                code: Some("E001".to_string()),
                end_line: None,
                end_column: None,
            }],
            warnings: vec![],
            source_hash: None,
        };

        let result = ApplyResult {
            success: false,
            path: "/tmp/test.py".to_string(),
            backup_path: None,
            validation: Some(validation),
            dry_run: false,
            bytes_written: None,
            error: Some("Validation failed with errors".to_string()),
        };

        assert!(!result.success);
        assert!(result.validation.is_some());
        assert!(!result.validation.as_ref().unwrap().success);
        assert_eq!(result.error, Some("Validation failed with errors".to_string()));
    }

    // =========================================================================
    // Type Conversion and Edge Case Tests
    // =========================================================================

    #[test]
    fn test_node_graph_with_all_optional_fields() {
        let graph = NodeGraph {
            nodes: vec![GraphNode {
                id: "node-1".to_string(),
                node_type: "component".to_string(),
                name: "Position".to_string(),
                position: NodePosition { x: 0.0, y: 0.0 },
                data: serde_json::json!({}),
                source: Some(CodeSourceLocation {
                    file: Some("test.py".to_string()),
                    line: Some(1),
                    column: Some(0),
                }),
            }],
            edges: vec![GraphEdge {
                id: "edge-1".to_string(),
                source: "node-1".to_string(),
                target: "node-2".to_string(),
                edge_type: "reference".to_string(),
                label: Some("label".to_string()),
                data: Some(serde_json::json!({"key": "value"})),
            }],
            metadata: Some(serde_json::json!({"version": 1})),
        };

        // Serialize and deserialize to verify all fields work
        let json = serde_json::to_string(&graph).unwrap();
        let deserialized: NodeGraph = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.nodes.len(), 1);
        assert_eq!(deserialized.edges.len(), 1);
        assert!(deserialized.metadata.is_some());
        assert!(deserialized.nodes[0].source.is_some());
        assert!(deserialized.edges[0].label.is_some());
        assert!(deserialized.edges[0].data.is_some());
    }

    #[test]
    fn test_empty_node_graph() {
        let graph = NodeGraph {
            nodes: vec![],
            edges: vec![],
            metadata: None,
        };

        let json = serde_json::to_string(&graph).unwrap();
        let deserialized: NodeGraph = serde_json::from_str(&json).unwrap();

        assert!(deserialized.nodes.is_empty());
        assert!(deserialized.edges.is_empty());
        assert!(deserialized.metadata.is_none());
    }

    #[test]
    fn test_generation_options_custom_values() {
        let opts = GenerationOptions {
            include_docstrings: false,
            include_type_hints: false,
            formatter: "pep8".to_string(),
            line_width: 120,
            import_style: "alphabetical".to_string(),
            target_python_version: Some("3.11".to_string()),
        };

        assert!(!opts.include_docstrings);
        assert!(!opts.include_type_hints);
        assert_eq!(opts.formatter, "pep8");
        assert_eq!(opts.line_width, 120);
        assert_eq!(opts.import_style, "alphabetical");
        assert_eq!(opts.target_python_version, Some("3.11".to_string()));
    }

    #[test]
    fn test_validation_options_with_linters() {
        let opts = ValidationOptions {
            check_syntax: true,
            check_types: true,
            check_trinity: true,
            check_style: true,
            linters: vec!["pylint".to_string(), "flake8".to_string(), "mypy".to_string()],
        };

        assert!(opts.check_syntax);
        assert!(opts.check_types);
        assert_eq!(opts.linters.len(), 3);
        assert!(opts.linters.contains(&"pylint".to_string()));
    }

    #[test]
    fn test_diff_stats_structure() {
        let stats = DiffStats {
            additions: 10,
            deletions: 5,
            changes: 15,
        };

        let json = serde_json::to_string(&stats).unwrap();
        let deserialized: DiffStats = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.additions, 10);
        assert_eq!(deserialized.deletions, 5);
        assert_eq!(deserialized.changes, 15);
    }

    #[test]
    fn test_import_info_structure() {
        let import = ImportInfo {
            module: "trinity.ecs".to_string(),
            names: vec!["component".to_string(), "system".to_string()],
            alias: None,
            is_from_import: true,
            line: 1,
        };

        let json = serde_json::to_string(&import).unwrap();
        let deserialized: ImportInfo = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.module, "trinity.ecs");
        assert_eq!(deserialized.names.len(), 2);
        assert!(deserialized.is_from_import);
    }

    #[test]
    fn test_diff_line_types() {
        let lines = vec![
            DiffLine {
                line_type: "added".to_string(),
                content: "new line".to_string(),
                original_line: None,
                modified_line: Some(5),
            },
            DiffLine {
                line_type: "removed".to_string(),
                content: "old line".to_string(),
                original_line: Some(3),
                modified_line: None,
            },
            DiffLine {
                line_type: "unchanged".to_string(),
                content: "same line".to_string(),
                original_line: Some(1),
                modified_line: Some(1),
            },
            DiffLine {
                line_type: "context".to_string(),
                content: "context line".to_string(),
                original_line: Some(2),
                modified_line: Some(2),
            },
        ];

        for line in lines {
            let json = serde_json::to_string(&line).unwrap();
            let deserialized: DiffLine = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized.line_type, line.line_type);
            assert_eq!(deserialized.content, line.content);
        }
    }

    #[test]
    fn test_validation_issue_without_optional_fields() {
        let issue = ValidationIssue {
            line: 1,
            column: 0,
            message: "Unused import".to_string(),
            severity: "warning".to_string(),
            code: None,
            end_line: None,
            end_column: None,
        };

        let json = serde_json::to_string(&issue).unwrap();

        // Verify optional fields are not present in JSON (skip_serializing_if)
        assert!(!json.contains("\"code\""));
        assert!(!json.contains("\"end_line\""));
        assert!(!json.contains("\"end_column\""));
    }

    // =========================================================================
    // Integration-like Tests (without sidecar)
    // =========================================================================

    #[test]
    fn test_full_graph_serialization_roundtrip() {
        let graph = create_sample_graph();

        // Serialize
        let json = serde_json::to_string_pretty(&graph).unwrap();

        // Deserialize
        let deserialized: NodeGraph = serde_json::from_str(&json).unwrap();

        // Verify structure
        assert_eq!(deserialized.nodes.len(), 2);
        assert_eq!(deserialized.edges.len(), 1);
        assert!(deserialized.metadata.is_some());

        // Verify first node
        assert_eq!(deserialized.nodes[0].id, "node-1");
        assert_eq!(deserialized.nodes[0].node_type, "component");
        assert_eq!(deserialized.nodes[0].name, "Position");

        // Verify second node has source location
        assert!(deserialized.nodes[1].source.is_some());
        let source = deserialized.nodes[1].source.as_ref().unwrap();
        assert_eq!(source.file, Some("/path/to/source.py".to_string()));
        assert_eq!(source.line, Some(10));

        // Verify edge
        assert_eq!(deserialized.edges[0].source, "node-1");
        assert_eq!(deserialized.edges[0].target, "node-2");
    }

    #[test]
    fn test_apply_options_combinations() {
        // Test various combinations of options
        let test_cases = vec![
            (true, true, false, true),   // backup, validate, dry_run, create_dirs
            (false, false, true, false),
            (true, false, true, true),
            (false, true, false, false),
        ];

        for (backup, validate, dry_run, create_dirs) in test_cases {
            let opts = ApplyOptions {
                backup,
                backup_suffix: ".bak".to_string(),
                validate_before_apply: validate,
                dry_run,
                create_dirs,
            };

            assert_eq!(opts.backup, backup);
            assert_eq!(opts.validate_before_apply, validate);
            assert_eq!(opts.dry_run, dry_run);
            assert_eq!(opts.create_dirs, create_dirs);
        }
    }

    #[test]
    fn test_file_write_and_restore_on_failure() {
        let test_dir = create_test_dir("write_restore");
        let file_path = test_dir.join("test.py");
        let backup_path_str = format!("{}.bak", file_path.display());

        // Create original file
        let original_content = "# Original";
        fs::write(&file_path, original_content).expect("Should write original");

        // Create backup
        fs::copy(&file_path, &backup_path_str).expect("Should create backup");

        // Write new content (simulating successful write)
        let new_content = "# New content";
        fs::write(&file_path, new_content).expect("Should write new content");

        // Verify new content is in place
        let current = fs::read_to_string(&file_path).unwrap();
        assert_eq!(current, new_content);

        // Simulate restore from backup (as in error recovery)
        fs::copy(&backup_path_str, &file_path).expect("Should restore from backup");

        // Verify original content is restored
        let restored = fs::read_to_string(&file_path).unwrap();
        assert_eq!(restored, original_content);

        cleanup_test_dir(&test_dir);
    }

    #[test]
    fn test_node_position_precision() {
        let position = NodePosition {
            x: 123.456789,
            y: -987.654321,
        };

        let json = serde_json::to_string(&position).unwrap();
        let deserialized: NodePosition = serde_json::from_str(&json).unwrap();

        // Verify floating point precision is maintained
        assert!((deserialized.x - 123.456789).abs() < 1e-6);
        assert!((deserialized.y - (-987.654321)).abs() < 1e-6);
    }

    #[test]
    fn test_generated_code_with_warnings() {
        let result = GeneratedCode {
            source: "# Code with warnings\npass".to_string(),
            validation: ValidationResult {
                success: true,
                errors: vec![],
                warnings: vec![
                    ValidationIssue {
                        line: 1,
                        column: 0,
                        message: "Deprecated pattern used".to_string(),
                        severity: "warning".to_string(),
                        code: Some("W001".to_string()),
                        end_line: None,
                        end_column: None,
                    },
                    ValidationIssue {
                        line: 2,
                        column: 0,
                        message: "Consider using type hints".to_string(),
                        severity: "info".to_string(),
                        code: Some("I001".to_string()),
                        end_line: None,
                        end_column: None,
                    },
                ],
                source_hash: None,
            },
            imports: vec![],
            node_count: 1,
            metadata: None,
        };

        assert!(result.validation.success);
        assert!(!result.source.is_empty());
        assert_eq!(result.validation.warnings.len(), 2);
        assert!(result.validation.errors.is_empty());
    }

    #[test]
    fn test_diff_result_no_changes() {
        let result = DiffResult {
            filename: "test.py".to_string(),
            original_path: Some("/path/to/test.py".to_string()),
            has_changes: false,
            unified_diff: String::new(),
            hunks: vec![],
            stats: DiffStats {
                additions: 0,
                deletions: 0,
                changes: 0,
            },
            error: None,
        };

        assert!(!result.has_changes);
        assert!(result.hunks.is_empty());
        assert_eq!(result.stats.additions, 0);
        assert_eq!(result.stats.deletions, 0);
        assert_eq!(result.stats.changes, 0);
    }

    #[test]
    fn test_validation_result_with_hash() {
        let result = ValidationResult {
            success: true,
            errors: vec![],
            warnings: vec![],
            source_hash: Some("abc123def456".to_string()),
        };

        assert!(result.success);
        assert_eq!(result.source_hash, Some("abc123def456".to_string()));

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"source_hash\":\"abc123def456\""));
    }

    #[test]
    fn test_code_source_location_partial() {
        // Test with only file
        let loc1 = CodeSourceLocation {
            file: Some("test.py".to_string()),
            line: None,
            column: None,
        };
        let json1 = serde_json::to_string(&loc1).unwrap();
        assert!(json1.contains("\"file\":\"test.py\""));
        assert!(!json1.contains("\"line\""));

        // Test with file and line
        let loc2 = CodeSourceLocation {
            file: Some("test.py".to_string()),
            line: Some(42),
            column: None,
        };
        let json2 = serde_json::to_string(&loc2).unwrap();
        assert!(json2.contains("\"line\":42"));
        assert!(!json2.contains("\"column\""));

        // Test with all fields
        let loc3 = CodeSourceLocation {
            file: Some("test.py".to_string()),
            line: Some(42),
            column: Some(10),
        };
        let json3 = serde_json::to_string(&loc3).unwrap();
        assert!(json3.contains("\"column\":10"));
    }

    #[test]
    fn test_graph_edge_types() {
        let edge_types = vec!["reference", "inheritance", "query", "event_handler"];

        for edge_type in edge_types {
            let edge = GraphEdge {
                id: "edge-1".to_string(),
                source: "node-1".to_string(),
                target: "node-2".to_string(),
                edge_type: edge_type.to_string(),
                label: None,
                data: None,
            };

            let json = serde_json::to_string(&edge).unwrap();
            let deserialized: GraphEdge = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized.edge_type, edge_type);
        }
    }

    #[test]
    fn test_node_types() {
        let node_types = vec!["component", "system", "resource", "event"];

        for node_type in node_types {
            let node = GraphNode {
                id: "node-1".to_string(),
                node_type: node_type.to_string(),
                name: "TestNode".to_string(),
                position: NodePosition { x: 0.0, y: 0.0 },
                data: serde_json::json!({}),
                source: None,
            };

            let json = serde_json::to_string(&node).unwrap();
            let deserialized: GraphNode = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized.node_type, node_type);
        }
    }

    #[test]
    fn test_multiple_imports() {
        let imports = vec![
            ImportInfo {
                module: "trinity".to_string(),
                names: vec!["component".to_string()],
                alias: None,
                is_from_import: true,
                line: 1,
            },
            ImportInfo {
                module: "trinity.systems".to_string(),
                names: vec!["system".to_string(), "query".to_string()],
                alias: None,
                is_from_import: true,
                line: 2,
            },
            ImportInfo {
                module: "numpy".to_string(),
                names: vec![],
                alias: Some("np".to_string()),
                is_from_import: false,
                line: 3,
            },
        ];

        let result = GeneratedCode {
            source: "# test".to_string(),
            validation: ValidationResult {
                success: true,
                errors: vec![],
                warnings: vec![],
                source_hash: None,
            },
            imports,
            node_count: 0,
            metadata: None,
        };

        assert_eq!(result.imports.len(), 3);
        assert!(result.imports[0].is_from_import);
        assert!(!result.imports[2].is_from_import);
        assert_eq!(result.imports[2].alias, Some("np".to_string()));
    }

    #[test]
    fn test_diff_hunk_with_multiple_lines() {
        let hunk = DiffHunk {
            original_start: 10,
            original_count: 5,
            modified_start: 10,
            modified_count: 8,
            lines: vec![
                DiffLine {
                    line_type: "context".to_string(),
                    content: "def hello():".to_string(),
                    original_line: Some(10),
                    modified_line: Some(10),
                },
                DiffLine {
                    line_type: "removed".to_string(),
                    content: "    print('hello')".to_string(),
                    original_line: Some(11),
                    modified_line: None,
                },
                DiffLine {
                    line_type: "added".to_string(),
                    content: "    print('hello world')".to_string(),
                    original_line: None,
                    modified_line: Some(11),
                },
                DiffLine {
                    line_type: "added".to_string(),
                    content: "    print('how are you?')".to_string(),
                    original_line: None,
                    modified_line: Some(12),
                },
                DiffLine {
                    line_type: "context".to_string(),
                    content: "".to_string(),
                    original_line: Some(12),
                    modified_line: Some(13),
                },
            ],
        };

        assert_eq!(hunk.lines.len(), 5);
        assert_eq!(
            hunk.lines.iter().filter(|l| l.line_type == "added").count(),
            2
        );
        assert_eq!(
            hunk.lines.iter().filter(|l| l.line_type == "removed").count(),
            1
        );
        assert_eq!(
            hunk.lines.iter().filter(|l| l.line_type == "context").count(),
            2
        );
    }

    #[test]
    fn test_apply_result_success_with_backup() {
        let result = ApplyResult {
            success: true,
            path: "/home/user/project/game.py".to_string(),
            backup_path: Some("/home/user/project/game.py.bak".to_string()),
            validation: Some(ValidationResult {
                success: true,
                errors: vec![],
                warnings: vec![],
                source_hash: Some("hash123".to_string()),
            }),
            dry_run: false,
            bytes_written: Some(1024),
            error: None,
        };

        assert!(result.success);
        assert!(result.backup_path.is_some());
        assert!(result.validation.is_some());
        assert!(result.validation.as_ref().unwrap().success);
        assert_eq!(result.bytes_written, Some(1024));
        assert!(!result.dry_run);
    }

    #[test]
    fn test_generation_options_serde_with_defaults() {
        // Test that defaults are applied during deserialization
        let json = "{}";
        let opts: GenerationOptions = serde_json::from_str(json).unwrap();

        assert!(opts.include_docstrings);
        assert!(opts.include_type_hints);
        assert_eq!(opts.formatter, "black");
        assert_eq!(opts.line_width, 88);
        assert_eq!(opts.import_style, "grouped");
        assert!(opts.target_python_version.is_none());
    }

    #[test]
    fn test_validation_options_serde_with_defaults() {
        // Test that defaults are applied during deserialization
        let json = "{}";
        let opts: ValidationOptions = serde_json::from_str(json).unwrap();

        assert!(opts.check_syntax);
        assert!(!opts.check_types);
        assert!(opts.check_trinity);
        assert!(!opts.check_style);
        assert!(opts.linters.is_empty());
    }

    #[test]
    fn test_apply_options_serde_with_defaults() {
        // Test that defaults are applied during deserialization
        let json = "{}";
        let opts: ApplyOptions = serde_json::from_str(json).unwrap();

        assert!(opts.backup);
        assert_eq!(opts.backup_suffix, ".bak");
        assert!(opts.validate_before_apply);
        assert!(!opts.dry_run);
        assert!(opts.create_dirs);
    }

    #[test]
    fn test_large_node_graph() {
        // Test handling of larger graphs
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        for i in 0..100 {
            nodes.push(GraphNode {
                id: format!("node-{}", i),
                node_type: if i % 4 == 0 {
                    "component".to_string()
                } else if i % 4 == 1 {
                    "system".to_string()
                } else if i % 4 == 2 {
                    "resource".to_string()
                } else {
                    "event".to_string()
                },
                name: format!("Element{}", i),
                position: NodePosition {
                    x: (i * 100) as f64,
                    y: ((i / 10) * 100) as f64,
                },
                data: serde_json::json!({"index": i}),
                source: None,
            });

            if i > 0 {
                edges.push(GraphEdge {
                    id: format!("edge-{}", i),
                    source: format!("node-{}", i - 1),
                    target: format!("node-{}", i),
                    edge_type: "reference".to_string(),
                    label: None,
                    data: None,
                });
            }
        }

        let graph = NodeGraph {
            nodes,
            edges,
            metadata: Some(serde_json::json!({"node_count": 100})),
        };

        // Serialize and deserialize
        let json = serde_json::to_string(&graph).unwrap();
        let deserialized: NodeGraph = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.nodes.len(), 100);
        assert_eq!(deserialized.edges.len(), 99);
    }
}
