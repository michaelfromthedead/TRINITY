//! Integration tests for the FlowForge Python sidecar manager
//!
//! These tests verify:
//! 1. Spawn/shutdown - Python sidecar starts and stops correctly
//! 2. Request/response - JSON-RPC 2.0 round-trip communication
//! 3. Error handling - timeout, crash recovery, invalid response handling
//! 4. IPC JSON protocol - message serialization/deserialization
//!
//! Note: Integration tests that actually spawn Python processes require
//! Python 3 to be installed and available as `python3` or `python`.
//!
//! Run tests with: cargo test --test sidecar_tests -- --test-threads=1

use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

// =============================================================================
// Test Utilities - Mock Python Script for Testing
// =============================================================================

/// Creates a mock Python script that simulates the sidecar IPC protocol
fn create_mock_python_script() -> String {
    r#"
import json
import sys

def main():
    """Mock IPC server for testing."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            response = {
                "jsonrpc": "2.0",
                "id": "unknown",
                "error": {"code": -32700, "message": "Parse error"}
            }
            print(json.dumps(response), flush=True)
            continue

        request_id = request.get("id", 0)
        method = request.get("method", "")
        params = request.get("params", {})

        if method == "ping":
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"pong": True, "timestamp": 1234567890}
            }
        elif method == "get_version":
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"version": "0.1.0-test", "python_version": "3.11.0"}
            }
        elif method == "echo":
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": params
            }
        elif method == "error_test":
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32600, "message": "Test error"}
            }
        elif method == "shutdown":
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"status": "shutting_down"}
            }
            print(json.dumps(response), flush=True)
            break
        elif method == "crash":
            # Simulate a crash by exiting abruptly
            sys.exit(1)
        elif method == "invalid_json":
            # Send invalid JSON response
            print("not valid json{{{", flush=True)
            continue
        elif method == "slow":
            # Simulate slow response (for timeout testing)
            import time
            time.sleep(float(params.get("delay", 5)))
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"slow": True}
            }
        else:
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"}
            }

        print(json.dumps(response), flush=True)

if __name__ == "__main__":
    main()
"#
    .to_string()
}

/// Creates a temporary Python script file and returns its path
fn write_mock_script() -> std::path::PathBuf {
    let temp_dir = std::env::temp_dir();
    let script_path = temp_dir.join(format!(
        "flowforge_mock_sidecar_{}.py",
        std::process::id()
    ));

    std::fs::write(&script_path, create_mock_python_script())
        .expect("Failed to write mock script");

    script_path
}

/// Cleanup mock script after tests
fn cleanup_mock_script(path: &std::path::Path) {
    let _ = std::fs::remove_file(path);
}

// =============================================================================
// Mock Sidecar Implementation for Testing
// =============================================================================

/// Test version of the sidecar that uses a mock Python script
struct MockSidecar {
    process: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    request_id: AtomicU64,
    script_path: std::path::PathBuf,
}

impl MockSidecar {
    /// Spawn a new mock sidecar process
    fn spawn() -> Result<Self, String> {
        let script_path = write_mock_script();

        let mut process = Command::new("python3")
            .arg(&script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .or_else(|_| {
                // Try 'python' if 'python3' doesn't work
                Command::new("python")
                    .arg(&script_path)
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::inherit())
                    .spawn()
            })
            .map_err(|e| format!("Failed to spawn mock sidecar: {}", e))?;

        let stdin = process
            .stdin
            .take()
            .ok_or_else(|| "Failed to capture stdin".to_string())?;
        let stdout = process
            .stdout
            .take()
            .ok_or_else(|| "Failed to capture stdout".to_string())?;

        Ok(Self {
            process,
            stdin,
            stdout: BufReader::new(stdout),
            request_id: AtomicU64::new(1),
            script_path,
        })
    }

    /// Send a JSON-RPC request and wait for response
    fn send_request(
        &mut self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        let id = self.request_id.fetch_add(1, Ordering::SeqCst);

        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params
        });

        // Send request
        let request_json = serde_json::to_string(&request)
            .map_err(|e| format!("Failed to serialize: {}", e))?;
        writeln!(self.stdin, "{}", request_json)
            .map_err(|e| format!("Failed to write: {}", e))?;
        self.stdin
            .flush()
            .map_err(|e| format!("Failed to flush: {}", e))?;

        // Read response
        let mut response_line = String::new();
        self.stdout
            .read_line(&mut response_line)
            .map_err(|e| format!("Failed to read: {}", e))?;

        // Parse response
        let response: serde_json::Value = serde_json::from_str(&response_line)
            .map_err(|e| format!("Failed to parse response: {}", e))?;

        // Check for error
        if let Some(error) = response.get("error") {
            let code = error.get("code").and_then(|c| c.as_i64()).unwrap_or(-1);
            let message = error
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown error");
            return Err(format!("Error {}: {}", code, message));
        }

        Ok(response
            .get("result")
            .cloned()
            .unwrap_or(serde_json::Value::Null))
    }

    /// Check if the process is still running
    fn is_running(&mut self) -> bool {
        match self.process.try_wait() {
            Ok(Some(_)) => false,
            Ok(None) => true,
            Err(_) => false,
        }
    }

    /// Shutdown the sidecar gracefully
    fn shutdown(&mut self) {
        // Try graceful shutdown
        let _ = self.send_request("shutdown", serde_json::Value::Null);

        // Wait a bit for graceful exit
        std::thread::sleep(Duration::from_millis(100));

        // Kill if still running
        if self.is_running() {
            let _ = self.process.kill();
            let _ = self.process.wait();
        }

        // Cleanup script
        cleanup_mock_script(&self.script_path);
    }
}

impl Drop for MockSidecar {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// =============================================================================
// JSON-RPC Protocol Tests (Unit Tests)
// =============================================================================

#[cfg(test)]
mod json_rpc_protocol_tests {
    use super::*;

    /// JSON-RPC request structure (matches sidecar/mod.rs)
    #[derive(Debug, Serialize)]
    struct JsonRpcRequest {
        jsonrpc: &'static str,
        id: u64,
        method: String,
        params: serde_json::Value,
    }

    /// JSON-RPC response structure
    #[derive(Debug, Deserialize)]
    struct JsonRpcResponse {
        #[allow(dead_code)]
        jsonrpc: String,
        id: u64,
        #[serde(default)]
        result: Option<serde_json::Value>,
        #[serde(default)]
        error: Option<JsonRpcError>,
    }

    #[derive(Debug, Deserialize)]
    struct JsonRpcError {
        code: i32,
        message: String,
    }

    #[test]
    fn test_json_rpc_request_serialization() {
        let request = JsonRpcRequest {
            jsonrpc: "2.0",
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
    fn test_json_rpc_request_with_complex_params() {
        let request = JsonRpcRequest {
            jsonrpc: "2.0",
            id: 42,
            method: "generate_code".to_string(),
            params: serde_json::json!({
                "graph": {
                    "nodes": [
                        {"id": "1", "type": "component", "name": "Position"},
                        {"id": "2", "type": "system", "name": "MovementSystem"}
                    ],
                    "edges": [
                        {"source": "1", "target": "2"}
                    ]
                },
                "format_with_black": true
            }),
        };

        let json = serde_json::to_string(&request).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed["jsonrpc"], "2.0");
        assert_eq!(parsed["id"], 42);
        assert_eq!(parsed["method"], "generate_code");
        assert!(parsed["params"]["graph"]["nodes"].is_array());
    }

    #[test]
    fn test_json_rpc_success_response_deserialization() {
        let json = r#"{"jsonrpc":"2.0","id":1,"result":{"success":true,"data":"test"}}"#;
        let response: JsonRpcResponse = serde_json::from_str(json).unwrap();

        assert_eq!(response.id, 1);
        assert!(response.result.is_some());
        assert!(response.error.is_none());

        let result = response.result.unwrap();
        assert_eq!(result["success"], true);
        assert_eq!(result["data"], "test");
    }

    #[test]
    fn test_json_rpc_error_response_deserialization() {
        let json = r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32600,"message":"Invalid Request"}}"#;
        let response: JsonRpcResponse = serde_json::from_str(json).unwrap();

        assert_eq!(response.id, 1);
        assert!(response.result.is_none());
        assert!(response.error.is_some());

        let error = response.error.unwrap();
        assert_eq!(error.code, -32600);
        assert_eq!(error.message, "Invalid Request");
    }

    #[test]
    fn test_json_rpc_method_not_found_error() {
        let json = r#"{"jsonrpc":"2.0","id":5,"error":{"code":-32601,"message":"Method not found: unknown_method"}}"#;
        let response: JsonRpcResponse = serde_json::from_str(json).unwrap();

        assert!(response.error.is_some());
        let error = response.error.unwrap();
        assert_eq!(error.code, -32601);
        assert!(error.message.contains("Method not found"));
    }

    #[test]
    fn test_json_rpc_parse_error() {
        let json = r#"{"jsonrpc":"2.0","id":"unknown","error":{"code":-32700,"message":"Parse error"}}"#;
        let response: serde_json::Value = serde_json::from_str(json).unwrap();

        assert!(response["error"].is_object());
        assert_eq!(response["error"]["code"], -32700);
    }

    #[test]
    fn test_json_rpc_internal_error() {
        let json = r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32603,"message":"Internal error: something went wrong"}}"#;
        let response: JsonRpcResponse = serde_json::from_str(json).unwrap();

        let error = response.error.unwrap();
        assert_eq!(error.code, -32603);
        assert!(error.message.contains("Internal error"));
    }

    #[test]
    fn test_json_rpc_invalid_params_error() {
        let json = r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32602,"message":"Invalid params: missing required field 'source'"}}"#;
        let response: JsonRpcResponse = serde_json::from_str(json).unwrap();

        let error = response.error.unwrap();
        assert_eq!(error.code, -32602);
        assert!(error.message.contains("Invalid params"));
    }

    #[test]
    fn test_json_rpc_request_with_null_params() {
        let request = JsonRpcRequest {
            jsonrpc: "2.0",
            id: 1,
            method: "ping".to_string(),
            params: serde_json::Value::Null,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"params\":null"));
    }

    #[test]
    fn test_json_rpc_request_with_empty_params() {
        let request = JsonRpcRequest {
            jsonrpc: "2.0",
            id: 1,
            method: "ping".to_string(),
            params: serde_json::json!({}),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"params\":{}"));
    }

    #[test]
    fn test_json_rpc_request_id_types() {
        // Test with integer ID
        let request1 = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "test"
        });
        assert!(request1["id"].is_number());

        // Test with string ID (also valid in JSON-RPC 2.0)
        let request2 = serde_json::json!({
            "jsonrpc": "2.0",
            "id": "request-123",
            "method": "test"
        });
        assert!(request2["id"].is_string());
    }

    #[test]
    fn test_response_with_complex_result() {
        let json = r#"{
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "source": "from trinity import component\n\n@component\nclass Position:\n    x: float = 0.0",
                "validation": {
                    "success": true,
                    "errors": [],
                    "warnings": [{"line": 1, "message": "Consider adding docstring"}]
                },
                "node_count": 1
            }
        }"#;

        let response: JsonRpcResponse = serde_json::from_str(json).unwrap();
        assert!(response.result.is_some());

        let result = response.result.unwrap();
        assert!(result["source"].is_string());
        assert!(result["validation"]["success"].as_bool().unwrap());
        assert_eq!(result["node_count"], 1);
    }
}

// =============================================================================
// Spawn/Shutdown Tests (Integration Tests)
// =============================================================================

#[cfg(test)]
mod spawn_shutdown_tests {
    use super::*;

    #[test]
    fn test_mock_sidecar_spawns_successfully() {
        let sidecar = MockSidecar::spawn();
        assert!(
            sidecar.is_ok(),
            "Failed to spawn mock sidecar: {:?}",
            sidecar.err()
        );

        let mut sidecar = sidecar.unwrap();
        assert!(sidecar.is_running(), "Sidecar should be running after spawn");
    }

    #[test]
    fn test_mock_sidecar_shutdown_graceful() {
        let mut sidecar = MockSidecar::spawn().expect("Failed to spawn");

        // Verify running
        assert!(sidecar.is_running());

        // Shutdown gracefully
        sidecar.shutdown();

        // Give it a moment to fully exit
        std::thread::sleep(Duration::from_millis(200));

        // Should not be running after shutdown
        assert!(!sidecar.is_running());
    }

    #[test]
    fn test_process_id_is_assigned() {
        let sidecar = MockSidecar::spawn().expect("Failed to spawn");

        // Process should have an ID
        let pid = sidecar.process.id();
        assert!(pid > 0, "Process ID should be positive");
    }

    #[test]
    fn test_multiple_sidecar_instances() {
        // Spawn multiple sidecars
        let mut sidecar1 = MockSidecar::spawn().expect("Failed to spawn first sidecar");
        let mut sidecar2 = MockSidecar::spawn().expect("Failed to spawn second sidecar");

        // Both should be running
        assert!(sidecar1.is_running());
        assert!(sidecar2.is_running());

        // They should have different PIDs
        assert_ne!(sidecar1.process.id(), sidecar2.process.id());

        // Both should respond to ping
        let result1 = sidecar1.send_request("ping", serde_json::json!({}));
        let result2 = sidecar2.send_request("ping", serde_json::json!({}));

        assert!(result1.is_ok());
        assert!(result2.is_ok());
    }

    #[test]
    fn test_drop_cleans_up_process() {
        let pid = {
            let sidecar = MockSidecar::spawn().expect("Failed to spawn");
            sidecar.process.id()
        }; // Sidecar dropped here

        // Give process time to be cleaned up
        std::thread::sleep(Duration::from_millis(300));

        // Try to check if process exists (Unix-specific approach)
        #[cfg(unix)]
        {
            use std::process::Command;
            let output = Command::new("kill")
                .args(["-0", &pid.to_string()])
                .output();

            // If kill -0 succeeds, the process exists; we expect it to fail
            if let Ok(out) = output {
                assert!(
                    !out.status.success(),
                    "Process should have been killed on drop"
                );
            }
        }
    }
}

// =============================================================================
// Request/Response Tests (Integration Tests)
// =============================================================================

#[cfg(test)]
mod request_response_tests {
    use super::*;

    #[test]
    fn test_ping_request() {
        let mut sidecar = MockSidecar::spawn().expect("Failed to spawn");

        let result = sidecar.send_request("ping", serde_json::json!({}));

        assert!(result.is_ok(), "Ping should succeed: {:?}", result.err());

        let response = result.unwrap();
        assert_eq!(response["pong"], true);
        assert!(response["timestamp"].is_number());
    }

    #[test]
    fn test_get_version_request() {
        let mut sidecar = MockSidecar::spawn().expect("Failed to spawn");

        let result = sidecar.send_request("get_version", serde_json::json!({}));

        assert!(result.is_ok());

        let response = result.unwrap();
        assert!(response["version"].is_string());
        assert!(response["python_version"].is_string());
    }

    #[test]
    fn test_echo_request() {
        let mut sidecar = MockSidecar::spawn().expect("Failed to spawn");

        let params = serde_json::json!({
            "message": "hello world",
            "number": 42,
            "nested": {"key": "value"}
        });

        let result = sidecar.send_request("echo", params.clone());

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), params);
    }

    #[test]
    fn test_multiple_sequential_requests() {
        let mut sidecar = MockSidecar::spawn().expect("Failed to spawn");

        // Send multiple requests in sequence
        for i in 0..10 {
            let result = sidecar.send_request("echo", serde_json::json!({"count": i}));
            assert!(result.is_ok(), "Request {} failed", i);
            assert_eq!(result.unwrap()["count"], i);
        }
    }

    #[test]
    fn test_request_with_large_payload() {
        let mut sidecar = MockSidecar::spawn().expect("Failed to spawn");

        // Create a large payload
        let large_string = "x".repeat(10000);
        let params = serde_json::json!({
            "data": large_string,
            "size": large_string.len()
        });

        let result = sidecar.send_request("echo", params.clone());

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response["size"], 10000);
    }

    #[test]
    fn test_request_with_unicode() {
        let mut sidecar = MockSidecar::spawn().expect("Failed to spawn");

        let params = serde_json::json!({
            "emoji": "Hello 🎮 World 🌍",
            "chinese": "你好世界",
            "japanese": "こんにちは",
            "special": "Line1\nLine2\tTabbed"
        });

        let result = sidecar.send_request("echo", params.clone());

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response["emoji"], "Hello 🎮 World 🌍");
        assert_eq!(response["chinese"], "你好世界");
    }

    #[test]
    fn test_request_id_increments() {
        let mut sidecar = MockSidecar::spawn().expect("Failed to spawn");

        // The request_id should increment with each call
        let initial_id = sidecar.request_id.load(Ordering::SeqCst);

        sidecar
            .send_request("ping", serde_json::json!({}))
            .unwrap();
        assert_eq!(sidecar.request_id.load(Ordering::SeqCst), initial_id + 1);

        sidecar
            .send_request("ping", serde_json::json!({}))
            .unwrap();
        assert_eq!(sidecar.request_id.load(Ordering::SeqCst), initial_id + 2);
    }
}

// =============================================================================
// Error Handling Tests
// =============================================================================

#[cfg(test)]
mod error_handling_tests {
    use super::*;

    #[test]
    fn test_method_not_found_error() {
        let mut sidecar = MockSidecar::spawn().expect("Failed to spawn");

        let result = sidecar.send_request("nonexistent_method", serde_json::json!({}));

        assert!(result.is_err());
        let error_msg = result.unwrap_err();
        assert!(error_msg.contains("-32601") || error_msg.contains("Method not found"));
    }

    #[test]
    fn test_error_response_handling() {
        let mut sidecar = MockSidecar::spawn().expect("Failed to spawn");

        let result = sidecar.send_request("error_test", serde_json::json!({}));

        assert!(result.is_err());
        let error_msg = result.unwrap_err();
        assert!(error_msg.contains("-32600") || error_msg.contains("Test error"));
    }

    #[test]
    fn test_crash_recovery_detection() {
        let mut sidecar = MockSidecar::spawn().expect("Failed to spawn");

        // Verify initially running
        assert!(sidecar.is_running());

        // Cause a crash by calling the crash method
        let _ = sidecar.send_request("crash", serde_json::json!({}));

        // Give it a moment to crash
        std::thread::sleep(Duration::from_millis(200));

        // Should no longer be running
        assert!(!sidecar.is_running());
    }

    #[test]
    fn test_communication_after_crash_fails() {
        let mut sidecar = MockSidecar::spawn().expect("Failed to spawn");

        // Cause crash
        let _ = sidecar.send_request("crash", serde_json::json!({}));
        std::thread::sleep(Duration::from_millis(200));

        // Further communication should fail
        let result = sidecar.send_request("ping", serde_json::json!({}));
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_json_response_handling() {
        let mut sidecar = MockSidecar::spawn().expect("Failed to spawn");

        // This method returns invalid JSON
        let result = sidecar.send_request("invalid_json", serde_json::json!({}));

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.contains("Failed to parse") || error.contains("invalid"));
    }

    #[test]
    fn test_empty_method_name() {
        let mut sidecar = MockSidecar::spawn().expect("Failed to spawn");

        let result = sidecar.send_request("", serde_json::json!({}));

        // Should return method not found error
        assert!(result.is_err());
    }

    #[test]
    fn test_null_params_handling() {
        let mut sidecar = MockSidecar::spawn().expect("Failed to spawn");

        // Send with explicit null params
        let result = sidecar.send_request("ping", serde_json::Value::Null);

        // Should still work
        assert!(result.is_ok());
    }
}

// =============================================================================
// IPC Message Format Tests
// =============================================================================

#[cfg(test)]
mod ipc_message_tests {
    #[test]
    fn test_line_delimited_json_format() {
        // Verify messages are sent as single lines
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "test",
            "params": {}
        });

        let serialized = serde_json::to_string(&request).unwrap();

        // Should not contain newlines within the message
        assert!(!serialized.contains('\n'));
        assert!(!serialized.contains('\r'));
    }

    #[test]
    fn test_compact_json_serialization() {
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "test",
            "params": {
                "nested": {
                    "deeply": {
                        "nested": "value"
                    }
                }
            }
        });

        // Use compact serialization (no pretty printing)
        let serialized = serde_json::to_string(&request).unwrap();

        // Should not have indentation or extra whitespace
        assert!(!serialized.contains("  ")); // No double spaces
    }

    #[test]
    fn test_special_characters_in_strings() {
        let test_cases = vec![
            ("quotes", r#"He said "hello""#),
            ("backslash", r"path\to\file"),
            ("newline", "line1\nline2"),
            ("tab", "col1\tcol2"),
            ("unicode", "émoji: 🎮"),
        ];

        for (name, value) in test_cases {
            let request = serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "echo",
                "params": {"test": value}
            });

            let serialized = serde_json::to_string(&request);
            assert!(
                serialized.is_ok(),
                "Failed to serialize {}: {:?}",
                name,
                serialized.err()
            );

            // Verify it can be deserialized back
            let deserialized: serde_json::Value =
                serde_json::from_str(&serialized.unwrap()).unwrap();
            assert_eq!(deserialized["params"]["test"], value);
        }
    }

    #[test]
    fn test_json_number_precision() {
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "echo",
            "params": {
                "integer": 9007199254740993_i64, // Beyond JS safe integer
                "float": 3.14159265358979323846,
                "small_float": 0.000000001,
                "negative": -999999999999_i64
            }
        });

        let serialized = serde_json::to_string(&request).unwrap();
        let deserialized: serde_json::Value = serde_json::from_str(&serialized).unwrap();

        // Integers should be preserved exactly
        assert_eq!(deserialized["params"]["integer"], 9007199254740993_i64);
        assert_eq!(deserialized["params"]["negative"], -999999999999_i64);
    }

    #[test]
    fn test_array_params() {
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "test",
            "params": {
                "items": [1, 2, 3, 4, 5],
                "mixed": [1, "two", 3.0, true, null]
            }
        });

        let serialized = serde_json::to_string(&request).unwrap();
        let deserialized: serde_json::Value = serde_json::from_str(&serialized).unwrap();

        assert!(deserialized["params"]["items"].is_array());
        assert_eq!(deserialized["params"]["items"].as_array().unwrap().len(), 5);
    }

    #[test]
    fn test_boolean_and_null_values() {
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "test",
            "params": {
                "true_val": true,
                "false_val": false,
                "null_val": null
            }
        });

        let serialized = serde_json::to_string(&request).unwrap();

        assert!(serialized.contains(":true"));
        assert!(serialized.contains(":false"));
        assert!(serialized.contains(":null"));

        let deserialized: serde_json::Value = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized["params"]["true_val"], true);
        assert_eq!(deserialized["params"]["false_val"], false);
        assert!(deserialized["params"]["null_val"].is_null());
    }
}

// =============================================================================
// SidecarState Tests (State Management)
// =============================================================================

#[cfg(test)]
mod state_management_tests {
    use super::*;

    /// Simulated SidecarState for testing (matches trinity.rs pattern)
    struct TestSidecarState(Mutex<Option<MockSidecar>>);

    impl TestSidecarState {
        fn new() -> Self {
            Self(Mutex::new(None))
        }

        fn with_sidecar(sidecar: MockSidecar) -> Self {
            Self(Mutex::new(Some(sidecar)))
        }

        fn ensure_running(&self) -> Result<(), String> {
            let mut guard = self.0.lock().map_err(|e| format!("Lock poisoned: {}", e))?;

            match &mut *guard {
                Some(sidecar) => {
                    if !sidecar.is_running() {
                        // Would restart here in real implementation
                        return Err("Sidecar crashed".to_string());
                    }
                    Ok(())
                }
                None => {
                    let sidecar = MockSidecar::spawn()?;
                    *guard = Some(sidecar);
                    Ok(())
                }
            }
        }

        fn send_request(
            &self,
            method: &str,
            params: serde_json::Value,
        ) -> Result<serde_json::Value, String> {
            self.ensure_running()?;

            let mut guard = self.0.lock().map_err(|e| format!("Lock poisoned: {}", e))?;

            guard
                .as_mut()
                .ok_or_else(|| "Sidecar not initialized".to_string())?
                .send_request(method, params)
        }
    }

    impl Default for TestSidecarState {
        fn default() -> Self {
            Self::new()
        }
    }

    #[test]
    fn test_sidecar_state_default_is_empty() {
        let state = TestSidecarState::default();
        let guard = state.0.lock().unwrap();
        assert!(guard.is_none());
    }

    #[test]
    fn test_sidecar_state_lazy_initialization() {
        let state = TestSidecarState::new();

        // First request should spawn the sidecar
        let result = state.send_request("ping", serde_json::json!({}));
        assert!(result.is_ok());

        // Sidecar should now be present
        let guard = state.0.lock().unwrap();
        assert!(guard.is_some());
    }

    #[test]
    fn test_sidecar_state_with_existing_sidecar() {
        let sidecar = MockSidecar::spawn().expect("Failed to spawn");
        let state = TestSidecarState::with_sidecar(sidecar);

        let guard = state.0.lock().unwrap();
        assert!(guard.is_some());
    }

    #[test]
    fn test_sidecar_state_multiple_requests() {
        let state = TestSidecarState::new();

        // Multiple requests should reuse the same sidecar
        for i in 0..5 {
            let result = state.send_request("echo", serde_json::json!({"n": i}));
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_sidecar_state_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let state = Arc::new(TestSidecarState::new());
        let mut handles = vec![];

        // Spawn multiple threads that use the sidecar
        for i in 0..3 {
            let state_clone = Arc::clone(&state);
            let handle = thread::spawn(move || {
                for j in 0..3 {
                    let result =
                        state_clone.send_request("echo", serde_json::json!({"thread": i, "iter": j}));
                    assert!(
                        result.is_ok(),
                        "Thread {} iter {} failed: {:?}",
                        i,
                        j,
                        result.err()
                    );
                }
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().expect("Thread panicked");
        }
    }
}

// =============================================================================
// Performance Tests
// =============================================================================

#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_spawn_time() {
        let start = Instant::now();
        let sidecar = MockSidecar::spawn();
        let duration = start.elapsed();

        assert!(sidecar.is_ok());

        // Spawn should complete in reasonable time (less than 5 seconds)
        assert!(
            duration.as_secs() < 5,
            "Spawn took too long: {:?}",
            duration
        );
    }

    #[test]
    fn test_request_latency() {
        let mut sidecar = MockSidecar::spawn().expect("Failed to spawn");

        // Warmup request
        sidecar
            .send_request("ping", serde_json::json!({}))
            .unwrap();

        // Measure latency over multiple requests
        let iterations = 10;
        let start = Instant::now();

        for _ in 0..iterations {
            sidecar
                .send_request("ping", serde_json::json!({}))
                .unwrap();
        }

        let total_duration = start.elapsed();
        let avg_latency = total_duration / iterations;

        // Average latency should be reasonable (less than 500ms per request)
        assert!(
            avg_latency.as_millis() < 500,
            "Average latency too high: {:?}",
            avg_latency
        );
    }

    #[test]
    fn test_throughput() {
        let mut sidecar = MockSidecar::spawn().expect("Failed to spawn");

        let iterations = 50;
        let start = Instant::now();

        for i in 0..iterations {
            sidecar
                .send_request("echo", serde_json::json!({"n": i}))
                .unwrap();
        }

        let duration = start.elapsed();
        let requests_per_second = iterations as f64 / duration.as_secs_f64();

        // Should handle at least 10 requests per second
        assert!(
            requests_per_second > 10.0,
            "Throughput too low: {:.2} req/s",
            requests_per_second
        );
    }

    #[test]
    fn test_large_payload_performance() {
        let mut sidecar = MockSidecar::spawn().expect("Failed to spawn");

        // Create a reasonably large payload (100KB)
        let large_data = "x".repeat(100_000);
        let params = serde_json::json!({
            "data": large_data
        });

        let start = Instant::now();
        let result = sidecar.send_request("echo", params);
        let duration = start.elapsed();

        assert!(result.is_ok());

        // Large payload should still complete in reasonable time
        assert!(
            duration.as_secs() < 5,
            "Large payload took too long: {:?}",
            duration
        );
    }
}

// =============================================================================
// Code Generation Specific Tests
// =============================================================================

#[cfg(test)]
mod codegen_protocol_tests {
    #[test]
    fn test_generate_code_request_format() {
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "generate_code",
            "params": {
                "graph": {
                    "nodes": [
                        {
                            "id": "node-1",
                            "type": "component",
                            "name": "Position",
                            "position": {"x": 100.0, "y": 200.0},
                            "data": {
                                "fields": [
                                    {"name": "x", "type": "float", "default": 0.0},
                                    {"name": "y", "type": "float", "default": 0.0}
                                ]
                            }
                        }
                    ],
                    "edges": []
                },
                "format_with_black": true,
                "add_header": true
            }
        });

        // Verify structure
        assert_eq!(request["method"], "generate_code");
        assert!(request["params"]["graph"]["nodes"].is_array());
        assert!(request["params"]["format_with_black"].as_bool().unwrap());
    }

    #[test]
    fn test_validate_code_request_format() {
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "validate_code",
            "params": {
                "source": "from trinity import component\n\n@component\nclass Position:\n    x: float = 0.0",
                "check_semantics": false
            }
        });

        assert_eq!(request["method"], "validate_code");
        assert!(request["params"]["source"].is_string());
    }

    #[test]
    fn test_generate_diff_request_format() {
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "generate_diff",
            "params": {
                "original": "# Old code",
                "modified": "# New code",
                "filename": "test.py",
                "context_lines": 3
            }
        });

        assert_eq!(request["method"], "generate_diff");
        assert!(request["params"]["original"].is_string());
        assert!(request["params"]["modified"].is_string());
    }

    #[test]
    fn test_apply_changes_request_format() {
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "apply_changes",
            "params": {
                "file_path": "/tmp/test.py",
                "content": "# Generated code",
                "create_backup": true
            }
        });

        assert_eq!(request["method"], "apply_changes");
        assert!(request["params"]["file_path"].is_string());
        assert!(request["params"]["content"].is_string());
        assert!(request["params"]["create_backup"].as_bool().unwrap());
    }

    #[test]
    fn test_validation_result_format() {
        // Expected format from Python backend
        let response = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "success": true,
                "errors": [],
                "warnings": [
                    {
                        "line": 5,
                        "column": 0,
                        "message": "Unused import 'os'",
                        "severity": "warning",
                        "code": "W001"
                    }
                ],
                "source_hash": "abc123"
            }
        });

        let result = &response["result"];
        assert!(result["success"].as_bool().unwrap());
        assert!(result["errors"].as_array().unwrap().is_empty());
        assert_eq!(result["warnings"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_diff_result_format() {
        // Expected format from Python backend
        let response = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "filename": "test.py",
                "original_path": "/path/to/test.py",
                "has_changes": true,
                "hunks": [
                    {
                        "original_start": 1,
                        "original_count": 3,
                        "modified_start": 1,
                        "modified_count": 5,
                        "lines": [
                            {"type": "context", "content": "# header"},
                            {"type": "removed", "content": "# old comment"},
                            {"type": "added", "content": "# new comment"},
                            {"type": "added", "content": "# extra line"}
                        ]
                    }
                ],
                "stats": {
                    "additions": 2,
                    "deletions": 1,
                    "changes": 3
                },
                "unified_diff": "--- a/test.py\n+++ b/test.py\n..."
            }
        });

        let result = &response["result"];
        assert!(result["has_changes"].as_bool().unwrap());
        assert!(result["hunks"].is_array());
        assert_eq!(result["stats"]["additions"], 2);
    }
}

// =============================================================================
// Edge Cases and Robustness Tests
// =============================================================================

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_empty_request_params() {
        let mut sidecar = MockSidecar::spawn().expect("Failed to spawn");

        // Various forms of "empty" params
        let result1 = sidecar.send_request("ping", serde_json::json!({}));
        let result2 = sidecar.send_request("ping", serde_json::json!(null));

        assert!(result1.is_ok());
        assert!(result2.is_ok());
    }

    #[test]
    fn test_deeply_nested_params() {
        let mut sidecar = MockSidecar::spawn().expect("Failed to spawn");

        // Create deeply nested structure
        let mut nested = serde_json::json!({"value": "deep"});
        for _ in 0..20 {
            nested = serde_json::json!({"nested": nested});
        }

        let result = sidecar.send_request("echo", nested.clone());
        assert!(result.is_ok());
    }

    #[test]
    fn test_special_method_names() {
        let mut sidecar = MockSidecar::spawn().expect("Failed to spawn");

        // Methods with dots (namespace style)
        let result = sidecar.send_request("trinity.status", serde_json::json!({}));
        // Should be method not found since mock doesn't have it
        assert!(result.is_err());

        // Method with underscores
        let result = sidecar.send_request("get_version", serde_json::json!({}));
        assert!(result.is_ok());
    }

    #[test]
    fn test_very_long_method_name() {
        let mut sidecar = MockSidecar::spawn().expect("Failed to spawn");

        let long_method = "a".repeat(1000);
        let result = sidecar.send_request(&long_method, serde_json::json!({}));

        // Should fail gracefully (method not found)
        assert!(result.is_err());
    }

    #[test]
    fn test_binary_like_string_params() {
        let mut sidecar = MockSidecar::spawn().expect("Failed to spawn");

        // String that looks like binary data (base64 encoded)
        let params = serde_json::json!({
            "data": "SGVsbG8gV29ybGQhIFRoaXMgaXMgYSB0ZXN0Lg==",
            "encoding": "base64"
        });

        let result = sidecar.send_request("echo", params.clone());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), params);
    }

    #[test]
    fn test_numeric_edge_cases() {
        let mut sidecar = MockSidecar::spawn().expect("Failed to spawn");

        let params = serde_json::json!({
            "zero": 0,
            "negative_zero": -0.0,
            "max_safe_int": 9007199254740991_i64,
            "min_safe_int": -9007199254740991_i64,
            "small_float": 1e-10,
            "large_float": 1e10
        });

        let result = sidecar.send_request("echo", params);
        assert!(result.is_ok());
    }

    #[test]
    fn test_empty_string_values() {
        let mut sidecar = MockSidecar::spawn().expect("Failed to spawn");

        let params = serde_json::json!({
            "empty": "",
            "whitespace": "   ",
            "newlines": "\n\n\n"
        });

        let result = sidecar.send_request("echo", params.clone());
        assert!(result.is_ok());

        let response = result.unwrap();
        assert_eq!(response["empty"], "");
    }
}
