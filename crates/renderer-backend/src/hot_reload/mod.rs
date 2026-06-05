//! Hot Reload Infrastructure for TRINITY Engine
//!
//! This module provides comprehensive hot-reload support including:
//! - Cross-platform file watching (`file_watcher`) - T-AS-6.1
//! - Shader recompilation and validation (`watcher`)
//! - Dependency graph management for material invalidation
//! - Atomic pipeline swapping
//!
//! # Architecture
//!
//! ```text
//! +----------------+     +------------------+     +----------------+
//! | FileWatcher    | --> | HotReloadWatcher | --> | PipelineSwapper|
//! | (OS events)    |     | (debounce,       |     | (atomic swap)  |
//! +----------------+     |  invalidation)   |     +----------------+
//!                        +------------------+
//!                               |
//!                               v
//!                        +------------------+
//!                        | ShaderRecompiler |
//!                        | (validation,     |
//!                        |  caching)        |
//!                        +------------------+
//! ```
//!
//! # Usage
//!
//! ## Low-level File Watching (T-AS-6.1)
//!
//! ```ignore
//! use renderer_backend::hot_reload::file_watcher::{FileWatcher, WatcherConfig};
//!
//! let mut watcher = FileWatcher::new(WatcherConfig::default())?;
//! watcher.watch(&PathBuf::from("shaders"))?;
//!
//! // In game loop:
//! for change in watcher.poll_events() {
//!     println!("Changed: {:?} ({:?})", change.path, change.kind);
//! }
//! ```
//!
//! ## High-level Hot Reload with Dependency Tracking
//!
//! ```ignore
//! use renderer_backend::hot_reload::{HotReloadWatcher, HotReloadConfig};
//!
//! let mut hot_reload = HotReloadWatcher::new(
//!     &[PathBuf::from("shaders/")],
//!     dep_graph,
//!     pipeline_table,
//!     HotReloadConfig::default(),
//! )?;
//!
//! // In game loop:
//! if let Ok(result) = hot_reload.process_changes() {
//!     if result.had_changes {
//!         println!("Reloaded {} materials", result.materials_reloaded);
//!     }
//! }
//! ```

// Cross-platform file watcher (T-AS-6.1)
pub mod file_watcher;

// Content-level change detection (T-AS-6.2)
pub mod content_change;

// Shader hot-reload propagation (T-AS-6.3)
pub mod shader_reload;

// Texture and mesh hot-reload (T-AS-6.4)
pub mod asset_reload;

// Material instance hot-reload and dependency viewer (T-AS-6.5)
pub mod material_reload;

// High-level hot reload watcher with dependency tracking
mod watcher;

// Re-export file_watcher types
pub use file_watcher::{
    FileChange, FileChangeKind, FileWatcher, FileWatcherError, MockFileWatcher, WatcherConfig,
    WatcherStats,
};

// Re-export content_change types (T-AS-6.2)
pub use content_change::{
    ChangeDetectorConfig, ChangeDetectorStats, ContentChange, ContentChangeDetector,
    ContentChangeKind,
};

// Re-export shader_reload types (T-AS-6.3)
pub use shader_reload::{
    ErrorSpan, FenceTracker, PipelineId, PsoHandle, PsoSwap, ReloadStatus, ShaderError,
    ShaderErrorKind, ShaderReloadEvent, ShaderReloadManager,
};

// Re-export asset_reload types (T-AS-6.4)
pub use asset_reload::{
    AssetReloadConfig, AssetReloadError, AssetReloadEvent, AssetReloadKind, AssetReloadManager,
    AssetReloadResult, AssetReloadStats, AssetReloadStatus, AssetSwap, CompressionQuality,
    GpuImage, MeshBuffers, MeshHandle, MeshReloadStage, MeshReloadState, MipmapFilter,
    StagingAllocation, StagingBufferConfig, StagingBufferPool, TextureFormat, TextureHandle,
    TextureReloadStage, TextureReloadState,
};

// Re-export material_reload types (T-AS-6.5)
pub use material_reload::{
    AssetType, DependencyNode, MaterialId, MaterialInstance, MaterialParameter,
    MaterialReloadEvent, MaterialReloadKind, MaterialReloadManager, MaterialValue,
};

// Re-export watcher types (original hot_reload.rs)
pub use watcher::{
    FileEvent, FileEventKind, FileWatcher as FileWatcherTrait, FrameGraphRebuildSignal,
    HotReloadConfig, HotReloadError, HotReloadStats, HotReloadWatcher, HotReloadWatcherBuilder,
    MockFileWatcher as LegacyMockFileWatcher, PipelineSwapper, ReloadEvent, ReloadResult,
    ShaderRecompiler,
};
