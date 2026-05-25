//! File dialog and I/O commands

use crate::state::AppState;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;
use tauri::State;
use tauri_plugin_dialog::FilePath;

/// File entry for directory listing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    pub name: String,
    pub path: String,
    pub is_dir: bool,
    pub size: u64,
    pub modified: Option<u64>, // Unix timestamp
}

/// File filter for dialogs
#[derive(Debug, Deserialize)]
pub struct FileFilter {
    pub name: String,
    pub extensions: Vec<String>,
}

/// Open file dialog request
#[derive(Debug, Deserialize)]
pub struct OpenFileRequest {
    pub filters: Option<Vec<FileFilter>>,
    pub title: Option<String>,
    pub default_path: Option<String>,
}

/// Save file dialog request
#[derive(Debug, Deserialize)]
pub struct SaveFileRequest {
    pub filters: Option<Vec<FileFilter>>,
    pub title: Option<String>,
    pub default_path: Option<String>,
}

/// Workflow file content
#[derive(Debug, Serialize)]
pub struct WorkflowFileContent {
    pub path: String,
    pub content: Value,
}

/// Open file dialog
#[tauri::command]
pub async fn open_file_dialog(
    request: OpenFileRequest,
    state: State<'_, AppState>,
) -> Result<Option<String>, String> {
    use tauri_plugin_dialog::DialogExt;

    let dialog = state.app_handle.dialog();
    let mut builder = dialog.file();

    if let Some(title) = request.title {
        builder = builder.set_title(&title);
    }

    if let Some(filters) = request.filters {
        for filter in filters {
            let ext_refs: Vec<&str> = filter.extensions.iter().map(|s| s.as_str()).collect();
            builder = builder.add_filter(&filter.name, &ext_refs);
        }
    }

    // Show the dialog
    let file_path = builder.blocking_pick_file();

    match file_path {
        Some(FilePath::Path(path)) => Ok(Some(path.to_string_lossy().to_string())),
        Some(other) => Ok(Some(other.to_string())),
        None => Ok(None),
    }
}

/// Save file dialog
#[tauri::command]
pub async fn save_file_dialog(
    request: SaveFileRequest,
    state: State<'_, AppState>,
) -> Result<Option<String>, String> {
    use tauri_plugin_dialog::DialogExt;

    let dialog = state.app_handle.dialog();
    let mut builder = dialog.file();

    if let Some(title) = request.title {
        builder = builder.set_title(&title);
    }

    if let Some(filters) = request.filters {
        for filter in filters {
            let ext_refs: Vec<&str> = filter.extensions.iter().map(|s| s.as_str()).collect();
            builder = builder.add_filter(&filter.name, &ext_refs);
        }
    }

    // Show the dialog
    let file_path = builder.blocking_save_file();

    match file_path {
        Some(FilePath::Path(path)) => Ok(Some(path.to_string_lossy().to_string())),
        Some(other) => Ok(Some(other.to_string())),
        None => Ok(None),
    }
}

/// Read a workflow file
#[tauri::command]
pub async fn read_workflow_file(path: String) -> Result<WorkflowFileContent, String> {
    let content =
        std::fs::read_to_string(&path).map_err(|e| format!("Failed to read file: {}", e))?;

    let json: Value =
        serde_json::from_str(&content).map_err(|e| format!("Failed to parse JSON: {}", e))?;

    Ok(WorkflowFileContent {
        path,
        content: json,
    })
}

/// Write a workflow file
#[tauri::command]
pub async fn write_workflow_file(path: String, content: Value) -> Result<bool, String> {
    let json = serde_json::to_string_pretty(&content)
        .map_err(|e| format!("Failed to serialize JSON: {}", e))?;

    std::fs::write(&path, json).map_err(|e| format!("Failed to write file: {}", e))?;

    Ok(true)
}

/// Result of writing a file with backup
#[derive(Debug, Serialize)]
pub struct WriteFileResult {
    pub success: bool,
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backup_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Write a text file with backup support
#[tauri::command]
pub async fn write_text_file_with_backup(path: String, content: String) -> Result<WriteFileResult, String> {
    let file_path = Path::new(&path);

    // Create backup if file exists
    let backup_path = if file_path.exists() {
        let backup = format!("{}.bak", path);
        match std::fs::copy(&path, &backup) {
            Ok(_) => Some(backup),
            Err(e) => {
                tracing::warn!("Failed to create backup: {}", e);
                None
            }
        }
    } else {
        None
    };

    // Write the new content
    match std::fs::write(&path, &content) {
        Ok(_) => Ok(WriteFileResult {
            success: true,
            path,
            backup_path,
            error: None,
        }),
        Err(e) => {
            // Try to restore from backup if write failed
            if let Some(ref backup) = backup_path {
                if let Err(restore_err) = std::fs::copy(backup, &path) {
                    tracing::error!("Failed to restore from backup: {}", restore_err);
                }
            }

            Ok(WriteFileResult {
                success: false,
                path,
                backup_path: None,
                error: Some(format!("Failed to write file: {}", e)),
            })
        }
    }
}

/// Check if a file exists
#[tauri::command]
pub async fn file_exists(path: String) -> Result<bool, String> {
    Ok(Path::new(&path).exists())
}

/// Get file info (size, modified time, etc.)
#[derive(Debug, Serialize)]
pub struct FileInfo {
    pub path: String,
    pub exists: bool,
    pub size: Option<u64>,
    pub modified: Option<u64>,
    pub is_readonly: bool,
}

#[tauri::command]
pub async fn get_file_info(path: String) -> Result<FileInfo, String> {
    let file_path = Path::new(&path);

    if !file_path.exists() {
        return Ok(FileInfo {
            path,
            exists: false,
            size: None,
            modified: None,
            is_readonly: false,
        });
    }

    let metadata = std::fs::metadata(&path)
        .map_err(|e| format!("Failed to get file metadata: {}", e))?;

    let modified = metadata
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs());

    Ok(FileInfo {
        path,
        exists: true,
        size: Some(metadata.len()),
        modified,
        is_readonly: metadata.permissions().readonly(),
    })
}

/// List contents of a directory
#[tauri::command]
pub async fn list_directory(path: String) -> Result<Vec<FileEntry>, String> {
    let dir_path = Path::new(&path);

    if !dir_path.exists() {
        return Err(format!("Directory does not exist: {}", path));
    }

    if !dir_path.is_dir() {
        return Err(format!("Path is not a directory: {}", path));
    }

    let entries = std::fs::read_dir(&path)
        .map_err(|e| format!("Failed to read directory: {}", e))?;

    let mut file_entries: Vec<FileEntry> = Vec::new();

    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!("Failed to read directory entry: {}", e);
                continue;
            }
        };

        let entry_path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();

        let metadata = match entry.metadata() {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!("Failed to get metadata for {:?}: {}", entry_path, e);
                continue;
            }
        };

        let is_dir = metadata.is_dir();
        let size = if is_dir { 0 } else { metadata.len() };

        let modified = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs());

        file_entries.push(FileEntry {
            name,
            path: entry_path.to_string_lossy().to_string(),
            is_dir,
            size,
            modified,
        });
    }

    // Sort: directories first, then alphabetically by name
    file_entries.sort_by(|a, b| {
        match (a.is_dir, b.is_dir) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
        }
    });

    Ok(file_entries)
}

/// Get the workspace/project root directory
#[tauri::command]
pub async fn get_workspace_root() -> Result<Option<String>, String> {
    match std::env::current_dir() {
        Ok(cwd) => Ok(Some(cwd.to_string_lossy().to_string())),
        Err(e) => {
            tracing::warn!("Failed to get current working directory: {}", e);
            Ok(None)
        }
    }
}
