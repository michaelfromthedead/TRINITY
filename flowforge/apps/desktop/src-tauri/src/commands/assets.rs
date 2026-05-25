//! Asset management commands

use crate::state::AppState;
use serde::{Deserialize, Serialize};
use tauri::{Manager, State};

/// Asset import request
#[derive(Debug, Deserialize)]
pub struct ImportAssetRequest {
    pub source_path: String,
    pub asset_type: Option<String>,
}

/// Asset import result
#[derive(Debug, Serialize)]
pub struct ImportAssetResult {
    pub id: String,
    pub local_path: String,
    pub asset_type: String,
}

/// Import an asset into the project
#[tauri::command]
pub async fn import_asset(
    request: ImportAssetRequest,
    state: State<'_, AppState>,
) -> Result<ImportAssetResult, String> {
    let source = std::path::Path::new(&request.source_path);

    if !source.exists() {
        return Err("Source file does not exist".to_string());
    }

    // Generate asset ID
    let asset_id = uuid::Uuid::new_v4().to_string();

    // Determine asset type from extension if not provided
    let asset_type = request.asset_type.unwrap_or_else(|| {
        source
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_lowercase())
            .unwrap_or_else(|| "unknown".to_string())
    });

    // Get app data directory for assets
    let assets_dir = state
        .app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to get app data dir: {}", e))?
        .join("assets");

    // Create assets directory if it doesn't exist
    std::fs::create_dir_all(&assets_dir)
        .map_err(|e| format!("Failed to create assets directory: {}", e))?;

    // Copy file to assets directory
    let file_name = source
        .file_name()
        .ok_or("Invalid source path")?
        .to_string_lossy();
    let dest_path = assets_dir.join(format!("{}_{}", asset_id, file_name));

    std::fs::copy(source, &dest_path).map_err(|e| format!("Failed to copy asset: {}", e))?;

    Ok(ImportAssetResult {
        id: asset_id,
        local_path: dest_path.to_string_lossy().to_string(),
        asset_type,
    })
}

/// Get the URL for accessing an asset
#[tauri::command]
pub async fn get_asset_url(local_path: String) -> Result<String, String> {
    // Convert local path to Tauri asset URL
    let path = std::path::Path::new(&local_path);

    if !path.exists() {
        return Err("Asset file does not exist".to_string());
    }

    // Return the asset protocol URL
    Ok(format!(
        "asset://localhost/{}",
        local_path.replace('\\', "/")
    ))
}
