//! Editor integration commands
//!
//! Commands for opening source files in external editors.

use serde::{Deserialize, Serialize};
use std::process::Command;
use tauri_plugin_shell::ShellExt;

/// Response from opening a file in editor
#[derive(Debug, Serialize)]
pub struct OpenEditorResponse {
    pub success: bool,
    pub message: String,
}

/// Request to open a file in editor
#[derive(Debug, Deserialize)]
pub struct OpenEditorRequest {
    pub file: String,
    pub line: Option<u32>,
    pub editor_command: Option<String>,
}

/// Open a file in an external editor
///
/// # Arguments
/// * `file` - The path to the file to open
/// * `line` - Optional line number to jump to
/// * `editor_command` - Optional custom editor command template
///
/// # Editor Command Template
/// The editor_command can contain placeholders:
/// * `{file}` - Will be replaced with the file path
/// * `{line}` - Will be replaced with the line number (or 1 if not provided)
///
/// # Examples
/// * VS Code: `code --goto {file}:{line}`
/// * Sublime: `subl {file}:{line}`
/// * Vim: `vim +{line} {file}`
/// * Neovim: `nvim +{line} {file}`
#[tauri::command]
pub async fn open_in_editor(
    app: tauri::AppHandle,
    request: OpenEditorRequest,
) -> Result<OpenEditorResponse, String> {
    let file = &request.file;
    let line = request.line.unwrap_or(1);

    // Check if file exists
    if !std::path::Path::new(file).exists() {
        return Err(format!("File not found: {}", file));
    }

    // If a custom editor command is provided, use it
    if let Some(ref cmd_template) = request.editor_command {
        return execute_custom_editor_command(cmd_template, file, line).await;
    }

    // Try to detect and use common editors
    if let Ok(response) = try_vscode(file, line).await {
        return Ok(response);
    }

    // Fall back to system default
    open_with_system_default(&app, file).await
}

/// Execute a custom editor command with template substitution
async fn execute_custom_editor_command(
    cmd_template: &str,
    file: &str,
    line: u32,
) -> Result<OpenEditorResponse, String> {
    // Substitute placeholders
    let cmd = cmd_template
        .replace("{file}", file)
        .replace("{line}", &line.to_string());

    // Parse the command into program and args
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    if parts.is_empty() {
        return Err("Empty editor command".to_string());
    }

    let program = parts[0];
    let args: Vec<&str> = parts[1..].to_vec();

    // Execute the command
    match Command::new(program).args(&args).spawn() {
        Ok(_) => Ok(OpenEditorResponse {
            success: true,
            message: format!("Opened {} in editor", file),
        }),
        Err(e) => Err(format!("Failed to open editor: {}", e)),
    }
}

/// Try to open file in VS Code
async fn try_vscode(file: &str, line: u32) -> Result<OpenEditorResponse, String> {
    // Try 'code' command (VS Code)
    let result = Command::new("code")
        .args(["--goto", &format!("{}:{}", file, line)])
        .spawn();

    match result {
        Ok(_) => Ok(OpenEditorResponse {
            success: true,
            message: format!("Opened {} in VS Code at line {}", file, line),
        }),
        Err(_) => Err("VS Code not found".to_string()),
    }
}

/// Open file with system default application
async fn open_with_system_default(
    app: &tauri::AppHandle,
    file: &str,
) -> Result<OpenEditorResponse, String> {
    // Use Tauri's shell plugin to open with default app
    let shell = app.shell();

    #[cfg(target_os = "windows")]
    let open_cmd = "start";
    #[cfg(target_os = "macos")]
    let open_cmd = "open";
    #[cfg(target_os = "linux")]
    let open_cmd = "xdg-open";

    match shell.command(open_cmd).arg(file).spawn() {
        Ok(_) => Ok(OpenEditorResponse {
            success: true,
            message: format!("Opened {} with system default", file),
        }),
        Err(e) => Err(format!("Failed to open file: {}", e)),
    }
}

/// Get a list of detected editors on the system
#[tauri::command]
pub async fn detect_editors() -> Result<Vec<EditorInfo>, String> {
    let mut editors = Vec::new();

    // Check for VS Code
    if which_exists("code") {
        editors.push(EditorInfo {
            name: "Visual Studio Code".to_string(),
            command: "code --goto {file}:{line}".to_string(),
            detected: true,
        });
    }

    // Check for Cursor
    if which_exists("cursor") {
        editors.push(EditorInfo {
            name: "Cursor".to_string(),
            command: "cursor --goto {file}:{line}".to_string(),
            detected: true,
        });
    }

    // Check for Sublime Text
    if which_exists("subl") {
        editors.push(EditorInfo {
            name: "Sublime Text".to_string(),
            command: "subl {file}:{line}".to_string(),
            detected: true,
        });
    }

    // Check for Neovim
    if which_exists("nvim") {
        editors.push(EditorInfo {
            name: "Neovim".to_string(),
            command: "nvim +{line} {file}".to_string(),
            detected: true,
        });
    }

    // Check for Vim
    if which_exists("vim") {
        editors.push(EditorInfo {
            name: "Vim".to_string(),
            command: "vim +{line} {file}".to_string(),
            detected: true,
        });
    }

    // Check for Emacs
    if which_exists("emacs") {
        editors.push(EditorInfo {
            name: "Emacs".to_string(),
            command: "emacs +{line} {file}".to_string(),
            detected: true,
        });
    }

    // Check for Kate (KDE)
    if which_exists("kate") {
        editors.push(EditorInfo {
            name: "Kate".to_string(),
            command: "kate --line {line} {file}".to_string(),
            detected: true,
        });
    }

    // Check for Gedit (GNOME)
    if which_exists("gedit") {
        editors.push(EditorInfo {
            name: "Gedit".to_string(),
            command: "gedit +{line} {file}".to_string(),
            detected: true,
        });
    }

    // Always add system default as fallback
    editors.push(EditorInfo {
        name: "System Default".to_string(),
        command: "".to_string(), // Empty means use system default
        detected: true,
    });

    Ok(editors)
}

/// Information about a detected editor
#[derive(Debug, Serialize)]
pub struct EditorInfo {
    pub name: String,
    pub command: String,
    pub detected: bool,
}

/// Check if a command exists in PATH
fn which_exists(cmd: &str) -> bool {
    #[cfg(target_os = "windows")]
    let check_cmd = "where";
    #[cfg(not(target_os = "windows"))]
    let check_cmd = "which";

    Command::new(check_cmd)
        .arg(cmd)
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}
