//! System information commands

use crate::state::AppState;
use serde::Serialize;
use tauri::State;

/// Application information
#[derive(Debug, Serialize)]
pub struct AppInfo {
    pub name: String,
    pub version: String,
    pub tauri_version: String,
    pub platform: String,
    pub arch: String,
}

/// Ping response
#[derive(Debug, Serialize)]
pub struct PingResponse {
    pub pong: bool,
    pub timestamp: u64,
}

/// Get application info
#[tauri::command]
pub async fn get_app_info() -> Result<AppInfo, String> {
    Ok(AppInfo {
        name: "FlowForge".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        tauri_version: tauri::VERSION.to_string(),
        platform: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
    })
}

/// Ping the backend (health check)
#[tauri::command]
pub async fn ping() -> Result<PingResponse, String> {
    Ok(PingResponse {
        pong: true,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0),
    })
}
