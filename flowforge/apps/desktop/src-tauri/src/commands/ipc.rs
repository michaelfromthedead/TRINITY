//! Generic IPC command for Python sidecar communication
//!
//! Provides a generic JSON-RPC style interface for calling Python backend methods.
//! This allows the frontend to call any registered Python handler without needing
//! a dedicated Rust command for each method.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::State;

use crate::commands::trinity::SidecarState;

// =============================================================================
// Request/Response Types
// =============================================================================

/// Generic IPC request following JSON-RPC style
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcRequest {
    /// Request ID for correlation
    pub id: String,
    /// Method name to call on the Python backend
    pub method: String,
    /// Parameters to pass to the method
    #[serde(default)]
    pub params: Value,
}

/// Generic IPC response following JSON-RPC style
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcResponse {
    /// Request ID for correlation
    pub id: String,
    /// Result of the method call (if successful)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    /// Error information (if failed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<IpcError>,
}

/// Error information for failed IPC calls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcError {
    /// Error code
    pub code: i32,
    /// Error message
    pub message: String,
    /// Additional error data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl IpcResponse {
    /// Create a successful response
    pub fn success(id: String, result: Value) -> Self {
        Self {
            id,
            result: Some(result),
            error: None,
        }
    }

    /// Create an error response
    pub fn error(id: String, code: i32, message: String, data: Option<Value>) -> Self {
        Self {
            id,
            result: None,
            error: Some(IpcError {
                code,
                message,
                data,
            }),
        }
    }
}

// =============================================================================
// Tauri Command
// =============================================================================

/// Generic IPC call to Python backend
///
/// Routes any method call to the Python sidecar's handler registry.
/// This provides a flexible interface for frontend-backend communication
/// without needing dedicated Rust commands for each operation.
///
/// # Arguments
/// * `request` - The IPC request containing method name and parameters
///
/// # Returns
/// IPC response with either the result or error information
///
/// # Example
/// ```typescript
/// const response = await invoke('ipc_call', {
///   request: {
///     id: 'req-1',
///     method: 'generate_code',
///     params: { graph: {...}, format_with_black: true }
///   }
/// });
/// ```
#[tauri::command]
pub async fn ipc_call(
    sidecar: State<'_, SidecarState>,
    request: IpcRequest,
) -> Result<IpcResponse, String> {
    let id = request.id.clone();
    let method = request.method.clone();

    match sidecar.send_request(&method, request.params) {
        Ok(result) => Ok(IpcResponse::success(id, result)),
        Err(e) => {
            // Parse error to determine appropriate error code
            let (code, message) = if e.contains("method not found") || e.contains("Method not found")
            {
                (-32601, format!("Method not found: {}", method))
            } else if e.contains("Invalid params") || e.contains("invalid params") {
                (-32602, e)
            } else if e.contains("Parse error") || e.contains("parse error") {
                (-32700, e)
            } else {
                (-32603, e) // Internal error
            };

            Ok(IpcResponse::error(id, code, message, None))
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ipc_request_serialization() {
        let request = IpcRequest {
            id: "test-1".to_string(),
            method: "generate_code".to_string(),
            params: serde_json::json!({"graph": {}}),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"id\":\"test-1\""));
        assert!(json.contains("\"method\":\"generate_code\""));
    }

    #[test]
    fn test_ipc_response_success() {
        let response = IpcResponse::success(
            "test-1".to_string(),
            serde_json::json!({"source": "print('hello')"}),
        );

        assert!(response.result.is_some());
        assert!(response.error.is_none());
    }

    #[test]
    fn test_ipc_response_error() {
        let response = IpcResponse::error(
            "test-1".to_string(),
            -32601,
            "Method not found".to_string(),
            None,
        );

        assert!(response.result.is_none());
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, -32601);
    }
}
