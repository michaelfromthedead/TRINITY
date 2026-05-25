//! FlowForge Desktop Application
//!
//! A domain-agnostic visual programming environment built with Tauri.

#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod bridge_protocol;
mod commands;
mod plugins;
mod sidecar;
mod state;

use crate::bridge_protocol::TOTAL_ENDPOINTS;

use commands::trinity::SidecarState;
use tauri::Manager;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

fn main() {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "flowforge=debug,tauri=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting FlowForge");

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            // Initialize application state
            let app_state = state::AppState::new(app.handle().clone())?;
            app.manage(app_state);

            // Initialize Python sidecar state (lazy initialization)
            // The sidecar will be spawned on first Trinity command
            let sidecar_state = SidecarState::new();
            app.manage(sidecar_state);

            tracing::info!("Application state initialized");

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::workflow::execute_workflow,
            commands::workflow::get_queue_status,
            commands::workflow::cancel_execution,
            commands::nodes::get_object_info,
            commands::nodes::get_node_definition,
            commands::nodes::search_nodes,
            commands::files::open_file_dialog,
            commands::files::save_file_dialog,
            commands::files::read_workflow_file,
            commands::files::write_workflow_file,
            commands::files::write_text_file_with_backup,
            commands::files::file_exists,
            commands::files::get_file_info,
            commands::files::list_directory,
            commands::files::get_workspace_root,
            commands::assets::import_asset,
            commands::assets::get_asset_url,
            commands::system::get_app_info,
            commands::system::ping,
            commands::python::parse_python_file,
            commands::python::read_python_file,
            commands::python::write_python_file,
            commands::python::get_trinity_node_types,
            commands::editor::open_in_editor,
            commands::editor::detect_editors,
            // Trinity introspection commands
            commands::trinity::trinity_connect,
            commands::trinity::trinity_status,
            commands::trinity::trinity_registry_list,
            commands::trinity::trinity_instances_query,
            commands::trinity::trinity_events_recent,
            // Trinity inspection commands (Phase 3.3.5)
            commands::trinity::trinity_inspect,
            commands::trinity::trinity_inspector_get,
            // Code generation commands (Phase 3.3.6)
            commands::codegen::generate_code,
            commands::codegen::validate_code,
            commands::codegen::generate_diff,
            commands::codegen::apply_changes,
            // Generic IPC for Python backend
            commands::ipc::ipc_call,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
