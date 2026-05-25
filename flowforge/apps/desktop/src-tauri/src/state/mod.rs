//! Application state management

use anyhow::Result;
use tauri::AppHandle;

/// Global application state
pub struct AppState {
    /// Tauri app handle for accessing APIs
    pub app_handle: AppHandle,
}

impl AppState {
    /// Create new application state
    pub fn new(app_handle: AppHandle) -> Result<Self> {
        Ok(Self { app_handle })
    }
}
