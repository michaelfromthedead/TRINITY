//! Python sidecar process management
//!
//! Handles spawning and communicating with the Python execution engine.
//! Uses stdio for IPC with line-delimited JSON messages.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;
use tracing::{debug, error, info, warn};

// =============================================================================
// Constants
// =============================================================================

/// JSON-RPC protocol version
const JSONRPC_VERSION: &str = "2.0";

/// Timeout in milliseconds to wait for graceful shutdown before killing
const SHUTDOWN_TIMEOUT_MS: u64 = 100;

/// Default Python command to use
const DEFAULT_PYTHON_COMMAND: &str = "python3";

/// Python module to run as sidecar
const PYTHON_MODULE: &str = "flowforge_backend";

/// Errors that can occur during sidecar operations
#[derive(Error, Debug)]
pub enum SidecarError {
    #[error("Failed to spawn Python process: {0}")]
    SpawnError(#[from] std::io::Error),

    #[error("Failed to serialize request: {0}")]
    SerializeError(#[from] serde_json::Error),

    #[error("Sidecar process not running")]
    NotRunning,

    #[error("Failed to communicate with sidecar: {0}")]
    CommunicationError(String),

    #[error("Sidecar returned error: {code} - {message}")]
    ResponseError { code: i32, message: String },

    #[error("Request timed out")]
    Timeout,
}

pub type Result<T> = std::result::Result<T, SidecarError>;

/// JSON-RPC style request to the Python backend
#[derive(Debug, Serialize)]
struct JsonRpcRequest {
    jsonrpc: &'static str,
    id: u64,
    method: String,
    params: Value,
}

/// JSON-RPC style response from the Python backend
#[derive(Debug, Deserialize)]
struct JsonRpcResponse {
    #[allow(dead_code)]
    jsonrpc: String,
    id: u64,
    #[serde(default)]
    result: Option<Value>,
    #[serde(default)]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Deserialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

/// Manages the Python sidecar process lifecycle and communication
pub struct PythonSidecar {
    process: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    request_id: AtomicU64,
    python_command: String,
}

impl PythonSidecar {
    /// Find the flowforge project root by looking for the flowforge_backend directory
    fn find_project_root() -> Option<std::path::PathBuf> {
        // Try relative to the executable
        if let Ok(exe) = std::env::current_exe() {
            let mut dir = exe.parent().map(|p| p.to_path_buf());
            for _ in 0..10 {
                if let Some(ref d) = dir {
                    if d.join("flowforge_backend").is_dir() {
                        return Some(d.clone());
                    }
                    dir = d.parent().map(|p| p.to_path_buf());
                } else {
                    break;
                }
            }
        }

        // Try relative to cwd
        if let Ok(cwd) = std::env::current_dir() {
            let mut dir = Some(cwd);
            for _ in 0..10 {
                if let Some(ref d) = dir {
                    if d.join("flowforge_backend").is_dir() {
                        return Some(d.clone());
                    }
                    dir = d.parent().map(|p| p.to_path_buf());
                } else {
                    break;
                }
            }
        }

        None
    }

    /// Spawns a new Python sidecar process
    ///
    /// The sidecar runs `python -m flowforge_backend` and communicates
    /// via stdin/stdout using line-delimited JSON-RPC messages.
    pub fn spawn() -> Result<Self> {
        Self::spawn_with_command(DEFAULT_PYTHON_COMMAND)
    }

    /// Spawns a new Python sidecar with a custom Python command
    ///
    /// Useful for specifying python3, a virtual environment python, or full path
    pub fn spawn_with_command(python_command: &str) -> Result<Self> {
        // Resolve the project root: walk up from the executable/cwd to find flowforge_backend
        let project_root = Self::find_project_root()
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());

        // Try venv python first, fall back to provided command
        let venv_python = project_root
            .parent() // flowforge -> AI_GAME_ENGINE
            .map(|p| p.join(".venv/bin/python3"));
        let actual_command = match &venv_python {
            Some(vp) if vp.exists() => {
                info!("Using venv Python: {:?}", vp);
                vp.to_string_lossy().to_string()
            }
            _ => {
                info!("Using system Python: {}", python_command);
                python_command.to_string()
            }
        };

        info!(
            "Spawning Python sidecar with command: {} in {:?}",
            actual_command, project_root
        );

        // Build PYTHONPATH: include project root (for flowforge_backend) and its parent (for trinity)
        let mut python_path = project_root.to_string_lossy().to_string();
        if let Some(parent) = project_root.parent() {
            python_path = format!("{}:{}", parent.to_string_lossy(), python_path);
        }
        info!("PYTHONPATH: {}", python_path);

        let mut process = Command::new(&actual_command)
            .args(["-m", PYTHON_MODULE])
            .current_dir(&project_root)
            .env("PYTHONPATH", &python_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()) // Let stderr pass through for debugging
            .spawn()
            .map_err(|e| {
                error!("Failed to spawn Python sidecar: {}", e);
                SidecarError::SpawnError(e)
            })?;

        let stdin = process.stdin.take().ok_or_else(|| {
            SidecarError::CommunicationError("Failed to capture stdin".to_string())
        })?;

        let stdout = process.stdout.take().ok_or_else(|| {
            SidecarError::CommunicationError("Failed to capture stdout".to_string())
        })?;

        info!(
            "Python sidecar spawned successfully with PID: {:?}",
            process.id()
        );

        Ok(Self {
            process,
            stdin,
            stdout: BufReader::new(stdout),
            request_id: AtomicU64::new(1),
            python_command: python_command.to_string(),
        })
    }

    /// Sends a JSON-RPC request to the Python sidecar and waits for a response
    ///
    /// # Arguments
    /// * `method` - The method name to invoke on the Python side
    /// * `params` - The parameters to pass (as a JSON Value)
    ///
    /// # Returns
    /// The result value from the Python backend, or an error
    pub fn send_request(&mut self, method: &str, params: Value) -> Result<Value> {
        // Check if process is still running
        if !self.is_running() {
            return Err(SidecarError::NotRunning);
        }

        let id = self.request_id.fetch_add(1, Ordering::SeqCst);

        let request = JsonRpcRequest {
            jsonrpc: JSONRPC_VERSION,
            id,
            method: method.to_string(),
            params,
        };

        debug!("Sending request to sidecar: {:?}", request);

        // Serialize and send the request
        let request_json = serde_json::to_string(&request)?;
        writeln!(self.stdin, "{}", request_json).map_err(|e| {
            SidecarError::CommunicationError(format!("Failed to write request: {}", e))
        })?;
        self.stdin.flush().map_err(|e| {
            SidecarError::CommunicationError(format!("Failed to flush stdin: {}", e))
        })?;

        // Read the response
        let mut response_line = String::new();
        self.stdout.read_line(&mut response_line).map_err(|e| {
            SidecarError::CommunicationError(format!("Failed to read response: {}", e))
        })?;

        debug!("Received response from sidecar: {}", response_line.trim());

        // Parse the response
        let response: JsonRpcResponse = serde_json::from_str(&response_line)?;

        // Verify the response ID matches
        if response.id != id {
            warn!("Response ID mismatch: expected {}, got {}", id, response.id);
        }

        // Check for errors
        if let Some(error) = response.error {
            return Err(SidecarError::ResponseError {
                code: error.code,
                message: error.message,
            });
        }

        Ok(response.result.unwrap_or(Value::Null))
    }

    /// Checks if the Python process is still running
    pub fn is_running(&mut self) -> bool {
        match self.process.try_wait() {
            Ok(Some(status)) => {
                debug!("Python sidecar exited with status: {:?}", status);
                false
            }
            Ok(None) => true,
            Err(e) => {
                error!("Error checking process status: {}", e);
                false
            }
        }
    }

    /// Attempts to restart the sidecar if it has crashed
    ///
    /// Returns Ok(true) if restart was successful, Ok(false) if already running,
    /// or an error if restart failed.
    pub fn restart_if_crashed(&mut self) -> Result<bool> {
        if self.is_running() {
            return Ok(false);
        }

        warn!("Python sidecar crashed, attempting restart...");

        // Spawn a fresh process using the raw spawn logic
        let project_root = Self::find_project_root()
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
        let mut python_path = project_root.to_string_lossy().to_string();
        if let Some(parent) = project_root.parent() {
            python_path = format!("{}:{}", parent.to_string_lossy(), python_path);
        }
        let python_command = self.python_command.clone();
        let mut process = std::process::Command::new(&python_command)
            .args(["-m", PYTHON_MODULE])
            .current_dir(&project_root)
            .env("PYTHONPATH", &python_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(SidecarError::SpawnError)?;

        let stdin = process.stdin.take().ok_or_else(|| {
            SidecarError::CommunicationError("Failed to capture stdin".to_string())
        })?;
        let stdout = process.stdout.take().ok_or_else(|| {
            SidecarError::CommunicationError("Failed to capture stdout".to_string())
        })?;

        // Replace fields directly (self is &mut, so we can assign)
        self.process = process;
        self.stdin = stdin;
        self.stdout = BufReader::new(stdout);
        self.request_id = AtomicU64::new(1);

        info!("Python sidecar restarted successfully");
        Ok(true)
    }

    /// Gracefully shuts down the Python sidecar
    ///
    /// Sends a shutdown request to the Python process and waits for it to exit.
    /// If the process doesn't exit gracefully, it will be killed.
    pub fn shutdown(&mut self) {
        info!("Shutting down Python sidecar...");

        // Try to send a graceful shutdown request
        if let Err(e) = self.send_request("shutdown", Value::Null) {
            debug!("Shutdown request failed (may be expected): {}", e);
        }

        // Give the process a moment to exit gracefully
        std::thread::sleep(std::time::Duration::from_millis(SHUTDOWN_TIMEOUT_MS));

        // Check if it exited
        match self.process.try_wait() {
            Ok(Some(status)) => {
                info!("Python sidecar exited gracefully with status: {:?}", status);
            }
            Ok(None) => {
                // Process still running, kill it
                warn!("Python sidecar didn't exit gracefully, killing...");
                if let Err(e) = self.process.kill() {
                    error!("Failed to kill Python sidecar: {}", e);
                }
                // Wait for it to actually exit
                let _ = self.process.wait();
                info!("Python sidecar killed");
            }
            Err(e) => {
                error!("Error checking sidecar exit status: {}", e);
            }
        }
    }

    /// Gets the process ID of the sidecar
    pub fn pid(&self) -> u32 {
        self.process.id()
    }
}

impl Drop for PythonSidecar {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_rpc_request_serialization() {
        let request = JsonRpcRequest {
            jsonrpc: JSONRPC_VERSION,
            id: 1,
            method: "test_method".to_string(),
            params: serde_json::json!({"key": "value"}),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(json.contains("\"id\":1"));
        assert!(json.contains("\"method\":\"test_method\""));
        assert!(json.contains("\"params\":{\"key\":\"value\"}"));
    }

    #[test]
    fn test_json_rpc_response_deserialization() {
        let json = r#"{"jsonrpc":"2.0","id":1,"result":{"success":true}}"#;
        let response: JsonRpcResponse = serde_json::from_str(json).unwrap();

        assert_eq!(response.id, 1);
        assert!(response.result.is_some());
        assert!(response.error.is_none());
    }

    #[test]
    fn test_json_rpc_error_response_deserialization() {
        let json =
            r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32600,"message":"Invalid Request"}}"#;
        let response: JsonRpcResponse = serde_json::from_str(json).unwrap();

        assert_eq!(response.id, 1);
        assert!(response.result.is_none());
        assert!(response.error.is_some());

        let error = response.error.unwrap();
        assert_eq!(error.code, -32600);
        assert_eq!(error.message, "Invalid Request");
    }
}
