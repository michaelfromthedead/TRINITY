//! Workflow execution commands

use crate::state::AppState;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::State;

/// Workflow execution request
#[derive(Debug, Deserialize)]
pub struct ExecuteWorkflowRequest {
    pub workflow: Value,
    pub config: Option<ExecutionConfig>,
}

/// Execution configuration
#[derive(Debug, Deserialize)]
pub struct ExecutionConfig {
    pub mode: Option<String>,
    pub priority: Option<String>,
    pub timeout: Option<u64>,
}

/// Execution result
#[derive(Debug, Serialize)]
pub struct ExecutionResponse {
    pub execution_id: String,
}

/// Queue status response
#[derive(Debug, Serialize)]
pub struct QueueStatus {
    pub running: Option<RunningExecution>,
    pub pending: Vec<PendingExecution>,
    pub recent: Vec<RecentExecution>,
}

#[derive(Debug, Serialize)]
pub struct RunningExecution {
    pub execution_id: String,
    pub started_at: String,
    pub progress: f64,
}

#[derive(Debug, Serialize)]
pub struct PendingExecution {
    pub execution_id: String,
    pub queued_at: String,
    pub priority: String,
}

#[derive(Debug, Serialize)]
pub struct RecentExecution {
    pub execution_id: String,
    pub success: bool,
    pub completed_at: String,
    pub duration: u64,
}

/// Execute a workflow
#[tauri::command]
pub async fn execute_workflow(
    request: ExecuteWorkflowRequest,
    state: State<'_, AppState>,
) -> Result<ExecutionResponse, String> {
    tracing::info!("Executing workflow");

    // Generate execution ID
    let execution_id = uuid::Uuid::new_v4().to_string();

    // TODO: Forward to Bun sidecar for execution
    // For now, just return the execution ID

    Ok(ExecutionResponse { execution_id })
}

/// Get queue status
#[tauri::command]
pub async fn get_queue_status(state: State<'_, AppState>) -> Result<QueueStatus, String> {
    // TODO: Get actual queue status from execution manager

    Ok(QueueStatus {
        running: None,
        pending: vec![],
        recent: vec![],
    })
}

/// Cancel an execution
#[tauri::command]
pub async fn cancel_execution(
    execution_id: String,
    state: State<'_, AppState>,
) -> Result<bool, String> {
    tracing::info!("Cancelling execution: {}", execution_id);

    // TODO: Cancel execution in Bun sidecar

    Ok(true)
}
