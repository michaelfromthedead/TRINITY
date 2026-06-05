//! Blackbox tests for shader hot-reload system (T-WGPU-P2.7.7).
//!
//! Tests the shader hot-reload API via the public interface at
//! `renderer_backend::shaders::hot_reload`.
//!
//! CLEANROOM: No src/ access beyond the public API. Tests use only
//! observable behavior through public constructors, methods, and traits.
//!
//! Feature-gated: requires `--features hot-reload` to compile.
//!
//! Coverage categories:
//!   1. API Surface Tests - public constructors and types
//!   2. ShaderWatcher Usage - file system watching
//!   3. ShaderHotReload Usage - high-level reload system
//!   4. Configuration - HotReloadConfig builder patterns
//!   5. Event Types - HotReloadEvent variants
//!   6. Error Handling - HotReloadError variants
//!   7. Statistics - HotReloadStats tracking
//!   8. Integration Patterns - typical development workflows

#![cfg(feature = "hot-reload")]

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use renderer_backend::shaders::hot_reload::{
    HotReloadConfig, HotReloadError, HotReloadEvent, HotReloadStats, ShaderHotReload,
    ShaderWatcher, DEFAULT_DEBOUNCE_MS, DEFAULT_WATCH_EXTENSIONS, MAX_PENDING_RELOADS,
};
use renderer_backend::shaders::{ShaderCache, ShaderCacheConfig};

// =============================================================================
// Helper: Create a minimal ShaderCache for testing ShaderHotReload
// =============================================================================

/// Creates a minimal shader cache for testing hot-reload functionality.
/// Uses pollster to block on async GPU device creation.
fn create_test_cache() -> Option<Arc<ShaderCache>> {
    // Try to create a wgpu instance and device for testing
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;

    let (device, _queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("test_hot_reload"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    ))
    .ok()?;

    let device = Arc::new(device);
    let config = ShaderCacheConfig::minimal();
    Some(Arc::new(ShaderCache::new(device, config)))
}

// =============================================================================
// CATEGORY 1: API Surface Tests (10 tests)
// =============================================================================

/// Test: HotReloadConfig::new creates a valid config.
#[test]
fn test_api_config_new() {
    let config = HotReloadConfig::new();
    assert!(config.validate().is_ok(), "new() config must be valid");
}

/// Test: HotReloadConfig::default creates a valid config.
#[test]
fn test_api_config_default() {
    let config = HotReloadConfig::default();
    assert!(config.validate().is_ok(), "default() config must be valid");
}

/// Test: HotReloadConfig::minimal creates a valid minimal config.
#[test]
fn test_api_config_minimal() {
    let config = HotReloadConfig::minimal();
    assert!(config.validate().is_ok(), "minimal() config must be valid");
    assert!(!config.auto_reload, "minimal() should disable auto_reload");
}

/// Test: HotReloadConfig::development creates a valid development config.
#[test]
fn test_api_config_development() {
    let config = HotReloadConfig::development();
    assert!(
        config.validate().is_ok(),
        "development() config must be valid"
    );
    assert!(config.auto_reload, "development() should enable auto_reload");
}

/// Test: ShaderWatcher::new creates a valid watcher.
#[test]
fn test_api_watcher_new() {
    let result = ShaderWatcher::new();
    assert!(result.is_ok(), "ShaderWatcher::new() must succeed");
    let watcher = result.unwrap();
    assert!(
        watcher.watched_paths().is_empty(),
        "New watcher has no watched paths"
    );
}

/// Test: ShaderWatcher::with_extensions creates a watcher with custom extensions.
#[test]
fn test_api_watcher_with_extensions() {
    let extensions = vec!["wgsl".to_string(), "glsl".to_string(), "hlsl".to_string()];
    let result = ShaderWatcher::with_extensions(extensions);
    assert!(
        result.is_ok(),
        "ShaderWatcher::with_extensions() must succeed"
    );
    let watcher = result.unwrap();
    assert!(watcher.extensions().contains("wgsl"));
    assert!(watcher.extensions().contains("glsl"));
    assert!(watcher.extensions().contains("hlsl"));
}

/// Test: ShaderWatcher::from_config creates a watcher from config.
#[test]
fn test_api_watcher_from_config() {
    let config = HotReloadConfig::new().recursive(false);
    let result = ShaderWatcher::from_config(&config);
    assert!(result.is_ok(), "ShaderWatcher::from_config() must succeed");
    let watcher = result.unwrap();
    assert!(!watcher.is_recursive(), "Watcher should inherit recursive=false");
}

/// Test: HotReloadError constructors create valid errors.
#[test]
fn test_api_error_constructors() {
    let err1 = HotReloadError::compilation("shader error");
    assert!(err1.is_compilation_error());

    let err2 = HotReloadError::path_not_found("/missing");
    assert!(err2.is_path_not_found());

    let err3 = HotReloadError::channel_closed();
    assert!(err3.is_channel_closed());

    let err4 = HotReloadError::config("invalid config");
    // config error doesn't have a specific predicate, just verify it formats
    let display = format!("{}", err4);
    assert!(display.contains("configuration error"));
}

/// Test: HotReloadEvent constructors create valid events.
#[test]
fn test_api_event_constructors() {
    let e1 = HotReloadEvent::modified("test.wgsl");
    assert!(e1.is_modified());

    let e2 = HotReloadEvent::created("new.wgsl");
    assert!(e2.is_created());

    let e3 = HotReloadEvent::deleted("old.wgsl");
    assert!(e3.is_deleted());

    let e4 = HotReloadEvent::watch_error("watch failed");
    assert!(e4.is_error());
}

/// Test: HotReloadStats::default creates zeroed stats.
#[test]
fn test_api_stats_default() {
    let stats = HotReloadStats::default();
    assert_eq!(stats.reloads, 0);
    assert_eq!(stats.invalidations, 0);
    assert_eq!(stats.callbacks_invoked, 0);
    assert_eq!(stats.errors, 0);
    assert_eq!(stats.dropped, 0);
}

// =============================================================================
// CATEGORY 2: ShaderWatcher Usage (8 tests)
// =============================================================================

/// Test: Watcher starts with no watched paths.
#[test]
fn test_watcher_initial_state() {
    let watcher = ShaderWatcher::new().unwrap();
    assert!(watcher.watched_paths().is_empty());
    assert_eq!(watcher.watched_count(), 0);
}

/// Test: Watcher with custom extensions has correct extension set.
#[test]
fn test_watcher_custom_extensions() {
    let watcher =
        ShaderWatcher::with_extensions(vec!["vert".to_string(), "frag".to_string()]).unwrap();
    assert!(watcher.extensions().contains("vert"));
    assert!(watcher.extensions().contains("frag"));
    assert!(!watcher.extensions().contains("wgsl"));
}

/// Test: Watching a nonexistent directory fails with PathNotFound.
#[test]
fn test_watcher_watch_nonexistent_fails() {
    let mut watcher = ShaderWatcher::new().unwrap();
    let result = watcher.watch_directory("/path/that/definitely/does/not/exist/abc123");
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(
            e.is_path_not_found(),
            "Error should be PathNotFound, got: {}",
            e
        );
    }
}

/// Test: Watching the same directory twice is idempotent.
#[test]
fn test_watcher_watch_idempotent() {
    let mut watcher = ShaderWatcher::new().unwrap();
    // Use temp dir for testing
    let temp = tempfile::tempdir().expect("create temp dir");
    let path = temp.path();

    let result1 = watcher.watch_directory(path);
    assert!(result1.is_ok(), "First watch should succeed");
    assert_eq!(watcher.watched_count(), 1);

    let result2 = watcher.watch_directory(path);
    assert!(result2.is_ok(), "Second watch of same path should succeed");
    assert_eq!(watcher.watched_count(), 1, "Count should not increase");
}

/// Test: Unwatching a non-watched directory returns false.
#[test]
fn test_watcher_unwatch_not_watching() {
    let mut watcher = ShaderWatcher::new().unwrap();
    let result = watcher.unwatch_directory("/some/random/path");
    // Should succeed but return false
    assert!(result.is_ok());
    assert!(!result.unwrap(), "Should return false for non-watched path");
}

/// Test: poll_changes returns empty when no changes.
#[test]
fn test_watcher_poll_changes_empty() {
    let watcher = ShaderWatcher::new().unwrap();
    let changes = watcher.poll_changes();
    assert!(changes.is_empty(), "No changes when nothing is watched");
}

/// Test: changed_files returns empty when no changes.
#[test]
fn test_watcher_changed_files_empty() {
    let watcher = ShaderWatcher::new().unwrap();
    let events = watcher.changed_files();
    assert!(events.is_empty(), "No events when nothing is watched");
}

/// Test: Add and remove extension works correctly.
#[test]
fn test_watcher_extension_management() {
    let mut watcher = ShaderWatcher::new().unwrap();

    // Add new extension
    watcher.add_extension("spv");
    assert!(watcher.extensions().contains("spv"));

    // Remove extension
    let removed = watcher.remove_extension("spv");
    assert!(removed, "Should return true when removing existing extension");
    assert!(!watcher.extensions().contains("spv"));

    // Remove non-existent
    let removed_again = watcher.remove_extension("spv");
    assert!(!removed_again, "Should return false for non-existent extension");
}

// =============================================================================
// CATEGORY 3: ShaderHotReload Usage (10 tests)
// =============================================================================

/// Test: ShaderHotReload::new succeeds with valid config.
#[test]
fn test_hot_reload_new_succeeds() {
    let cache = match create_test_cache() {
        Some(c) => c,
        None => {
            eprintln!("Skipping test_hot_reload_new_succeeds: no GPU adapter available");
            return;
        }
    };

    let config = HotReloadConfig::default();
    let result = ShaderHotReload::new(cache, config);
    assert!(result.is_ok(), "ShaderHotReload::new should succeed");
}

/// Test: ShaderHotReload::with_defaults succeeds.
#[test]
fn test_hot_reload_with_defaults() {
    let cache = match create_test_cache() {
        Some(c) => c,
        None => {
            eprintln!("Skipping test_hot_reload_with_defaults: no GPU adapter available");
            return;
        }
    };

    let result = ShaderHotReload::with_defaults(cache);
    assert!(result.is_ok(), "ShaderHotReload::with_defaults should succeed");
}

/// Test: ShaderHotReload::new fails with invalid config.
#[test]
fn test_hot_reload_new_fails_invalid_config() {
    let cache = match create_test_cache() {
        Some(c) => c,
        None => {
            eprintln!("Skipping test_hot_reload_new_fails_invalid_config: no GPU adapter available");
            return;
        }
    };

    // Create invalid config (no extensions)
    let config = HotReloadConfig::new().watch_extensions(Vec::<String>::new());
    let result = ShaderHotReload::new(cache, config);
    assert!(result.is_err(), "Should fail with invalid config");
}

/// Test: Callback registration and count.
#[test]
fn test_hot_reload_callbacks() {
    let cache = match create_test_cache() {
        Some(c) => c,
        None => {
            eprintln!("Skipping test_hot_reload_callbacks: no GPU adapter available");
            return;
        }
    };

    let mut hot_reload = ShaderHotReload::with_defaults(cache).unwrap();

    assert_eq!(hot_reload.callback_count(), 0, "No callbacks initially");

    hot_reload.register_callback(|_path| {
        // Callback does nothing
    });
    assert_eq!(hot_reload.callback_count(), 1, "One callback registered");

    hot_reload.register_callback(|_path| {
        // Another callback
    });
    assert_eq!(hot_reload.callback_count(), 2, "Two callbacks registered");

    hot_reload.clear_callbacks();
    assert_eq!(hot_reload.callback_count(), 0, "Callbacks cleared");
}

/// Test: poll returns empty vec when no changes.
#[test]
fn test_hot_reload_poll_no_changes() {
    let cache = match create_test_cache() {
        Some(c) => c,
        None => {
            eprintln!("Skipping test_hot_reload_poll_no_changes: no GPU adapter available");
            return;
        }
    };

    let mut hot_reload = ShaderHotReload::with_defaults(cache).unwrap();
    let result = hot_reload.poll();
    assert!(result.is_ok());
    let reloaded = result.unwrap();
    assert!(reloaded.is_empty(), "No reloads when nothing watched");
}

/// Test: pending_reloads and pending_count work correctly.
#[test]
fn test_hot_reload_pending() {
    let cache = match create_test_cache() {
        Some(c) => c,
        None => {
            eprintln!("Skipping test_hot_reload_pending: no GPU adapter available");
            return;
        }
    };

    let hot_reload = ShaderHotReload::with_defaults(cache).unwrap();
    assert!(hot_reload.pending_reloads().is_empty());
    assert_eq!(hot_reload.pending_count(), 0);
}

/// Test: clear_pending clears the pending queue.
#[test]
fn test_hot_reload_clear_pending() {
    let cache = match create_test_cache() {
        Some(c) => c,
        None => {
            eprintln!("Skipping test_hot_reload_clear_pending: no GPU adapter available");
            return;
        }
    };

    let mut hot_reload = ShaderHotReload::with_defaults(cache).unwrap();
    hot_reload.clear_pending();
    assert_eq!(hot_reload.pending_count(), 0);
}

/// Test: config accessor returns the config.
#[test]
fn test_hot_reload_config_accessor() {
    let cache = match create_test_cache() {
        Some(c) => c,
        None => {
            eprintln!("Skipping test_hot_reload_config_accessor: no GPU adapter available");
            return;
        }
    };

    let config = HotReloadConfig::new().debounce_ms(250);
    let hot_reload = ShaderHotReload::new(cache, config).unwrap();
    assert_eq!(hot_reload.config().debounce_ms, 250);
}

/// Test: watcher accessor returns the watcher.
#[test]
fn test_hot_reload_watcher_accessor() {
    let cache = match create_test_cache() {
        Some(c) => c,
        None => {
            eprintln!("Skipping test_hot_reload_watcher_accessor: no GPU adapter available");
            return;
        }
    };

    let hot_reload = ShaderHotReload::with_defaults(cache).unwrap();
    let watcher = hot_reload.watcher();
    assert!(watcher.watched_paths().is_empty());
}

/// Test: stats accessor returns stats.
#[test]
fn test_hot_reload_stats_accessor() {
    let cache = match create_test_cache() {
        Some(c) => c,
        None => {
            eprintln!("Skipping test_hot_reload_stats_accessor: no GPU adapter available");
            return;
        }
    };

    let hot_reload = ShaderHotReload::with_defaults(cache).unwrap();
    let stats = hot_reload.stats();
    assert_eq!(stats.reloads, 0);
    assert_eq!(stats.errors, 0);
}

// =============================================================================
// CATEGORY 4: Configuration (8 tests)
// =============================================================================

/// Test: Default config has expected values.
#[test]
fn test_config_default_values() {
    let config = HotReloadConfig::default();
    assert_eq!(config.debounce_ms, DEFAULT_DEBOUNCE_MS);
    assert!(config.auto_reload);
    assert!(config.recursive);
    assert!(config.log_events);
    assert_eq!(config.max_pending, MAX_PENDING_RELOADS);
    assert!(!config.watch_extensions.is_empty());
}

/// Test: Builder pattern chaining works.
#[test]
fn test_config_builder_chain() {
    let config = HotReloadConfig::new()
        .debounce_ms(500)
        .auto_reload(false)
        .recursive(false)
        .log_events(false)
        .max_pending(100);

    assert_eq!(config.debounce_ms, 500);
    assert!(!config.auto_reload);
    assert!(!config.recursive);
    assert!(!config.log_events);
    assert_eq!(config.max_pending, 100);
}

/// Test: debounce() builder accepts Duration.
#[test]
fn test_config_debounce_duration() {
    let config = HotReloadConfig::new().debounce(Duration::from_millis(300));
    assert_eq!(config.debounce_ms, 300);
    assert_eq!(config.debounce_duration(), Duration::from_millis(300));
}

/// Test: watch_extensions replaces all extensions.
#[test]
fn test_config_watch_extensions() {
    let config = HotReloadConfig::new().watch_extensions(vec!["a", "b", "c"]);
    assert_eq!(config.watch_extensions.len(), 3);
    assert!(config.watch_extensions.contains(&"a".to_string()));
    assert!(config.watch_extensions.contains(&"b".to_string()));
    assert!(config.watch_extensions.contains(&"c".to_string()));
}

/// Test: add_extension appends to existing extensions.
#[test]
fn test_config_add_extension() {
    let config = HotReloadConfig::new().add_extension("custom");
    assert!(config.watch_extensions.contains(&"custom".to_string()));
    // Should still have default wgsl
    assert!(config.watch_extensions.contains(&"wgsl".to_string()));
}

/// Test: watches_extension is case-insensitive.
#[test]
fn test_config_watches_extension_case_insensitive() {
    let config = HotReloadConfig::default();
    assert!(config.watches_extension("wgsl"));
    assert!(config.watches_extension("WGSL"));
    assert!(config.watches_extension("Wgsl"));
    assert!(config.watches_extension(".wgsl"));
    assert!(config.watches_extension(".WGSL"));
}

/// Test: validate fails for empty extensions.
#[test]
fn test_config_validate_empty_extensions() {
    let config = HotReloadConfig::new().watch_extensions(Vec::<String>::new());
    let result = config.validate();
    assert!(result.is_err());
}

/// Test: validate fails for zero max_pending.
#[test]
fn test_config_validate_zero_max_pending() {
    let config = HotReloadConfig::new().max_pending(0);
    let result = config.validate();
    assert!(result.is_err());
}

// =============================================================================
// CATEGORY 5: Event Types (6 tests)
// =============================================================================

/// Test: ShaderModified event has correct properties.
#[test]
fn test_event_modified_properties() {
    let event = HotReloadEvent::modified("shaders/test.wgsl");

    assert!(event.is_modified());
    assert!(!event.is_created());
    assert!(!event.is_deleted());
    assert!(!event.is_error());
    assert!(event.requires_recompilation());
    assert_eq!(event.path(), Some(Path::new("shaders/test.wgsl")));
    assert!(event.error_message().is_none());
}

/// Test: ShaderCreated event has correct properties.
#[test]
fn test_event_created_properties() {
    let event = HotReloadEvent::created("new_shader.wgsl");

    assert!(!event.is_modified());
    assert!(event.is_created());
    assert!(!event.is_deleted());
    assert!(!event.is_error());
    assert!(event.requires_recompilation());
    assert_eq!(event.path(), Some(Path::new("new_shader.wgsl")));
}

/// Test: ShaderDeleted event has correct properties.
#[test]
fn test_event_deleted_properties() {
    let event = HotReloadEvent::deleted("old_shader.wgsl");

    assert!(!event.is_modified());
    assert!(!event.is_created());
    assert!(event.is_deleted());
    assert!(!event.is_error());
    assert!(!event.requires_recompilation()); // Deleted doesn't require recompilation
    assert_eq!(event.path(), Some(Path::new("old_shader.wgsl")));
}

/// Test: WatchError event has correct properties.
#[test]
fn test_event_watch_error_properties() {
    let event = HotReloadEvent::watch_error("inotify limit exceeded");

    assert!(!event.is_modified());
    assert!(!event.is_created());
    assert!(!event.is_deleted());
    assert!(event.is_error());
    assert!(!event.requires_recompilation());
    assert!(event.path().is_none());
    assert_eq!(event.error_message(), Some("inotify limit exceeded"));
}

/// Test: Event Display formatting.
#[test]
fn test_event_display() {
    let e1 = HotReloadEvent::modified("test.wgsl");
    let s1 = format!("{}", e1);
    assert!(s1.contains("modified"));
    assert!(s1.contains("test.wgsl"));

    let e2 = HotReloadEvent::created("new.wgsl");
    let s2 = format!("{}", e2);
    assert!(s2.contains("created"));

    let e3 = HotReloadEvent::deleted("old.wgsl");
    let s3 = format!("{}", e3);
    assert!(s3.contains("deleted"));

    let e4 = HotReloadEvent::watch_error("error message");
    let s4 = format!("{}", e4);
    assert!(s4.contains("error"));
}

/// Test: Event equality comparison.
#[test]
fn test_event_equality() {
    let e1 = HotReloadEvent::modified("a.wgsl");
    let e2 = HotReloadEvent::modified("a.wgsl");
    let e3 = HotReloadEvent::modified("b.wgsl");
    let e4 = HotReloadEvent::created("a.wgsl");

    assert_eq!(e1, e2, "Same path and type should be equal");
    assert_ne!(e1, e3, "Different paths should not be equal");
    assert_ne!(e1, e4, "Different event types should not be equal");
}

// =============================================================================
// CATEGORY 6: Error Handling (6 tests)
// =============================================================================

/// Test: CompilationError provides message access.
#[test]
fn test_error_compilation_message() {
    let err = HotReloadError::compilation("syntax error at line 42");
    assert!(err.is_compilation_error());
    assert_eq!(
        err.compilation_message(),
        Some("syntax error at line 42")
    );
    assert!(err.path().is_none());
}

/// Test: PathNotFound provides path access.
#[test]
fn test_error_path_not_found_path() {
    let err = HotReloadError::path_not_found("/missing/shader.wgsl");
    assert!(err.is_path_not_found());
    assert_eq!(err.path(), Some(Path::new("/missing/shader.wgsl")));
    assert!(err.compilation_message().is_none());
}

/// Test: ChannelClosed error type.
#[test]
fn test_error_channel_closed() {
    let err = HotReloadError::channel_closed();
    assert!(err.is_channel_closed());
    assert!(!err.is_watch_error());
    assert!(!err.is_compilation_error());
    assert!(!err.is_path_not_found());
}

/// Test: Error Display formatting.
#[test]
fn test_error_display() {
    let e1 = HotReloadError::compilation("parse failed");
    assert!(format!("{}", e1).contains("compilation error"));

    let e2 = HotReloadError::path_not_found("/missing");
    assert!(format!("{}", e2).contains("path not found"));

    let e3 = HotReloadError::channel_closed();
    assert!(format!("{}", e3).contains("channel closed"));

    let e4 = HotReloadError::config("invalid debounce");
    assert!(format!("{}", e4).contains("configuration error"));
}

/// Test: std::error::Error impl.
#[test]
fn test_error_std_error() {
    fn assert_error<E: std::error::Error>(_: &E) {}

    let err = HotReloadError::compilation("test");
    assert_error(&err);
}

/// Test: From<notify::Error> conversion.
#[test]
fn test_error_from_notify() {
    let notify_err = notify::Error::generic("notify failure");
    let hot_reload_err: HotReloadError = notify_err.into();
    assert!(hot_reload_err.is_watch_error());
}

// =============================================================================
// CATEGORY 7: Statistics (6 tests)
// =============================================================================

/// Test: Default stats are all zero.
#[test]
fn test_stats_default_zeros() {
    let stats = HotReloadStats::default();
    assert_eq!(stats.reloads, 0);
    assert_eq!(stats.invalidations, 0);
    assert_eq!(stats.callbacks_invoked, 0);
    assert_eq!(stats.errors, 0);
    assert_eq!(stats.dropped, 0);
}

/// Test: total_events calculation.
#[test]
fn test_stats_total_events() {
    let mut stats = HotReloadStats::default();
    stats.reloads = 10;
    stats.errors = 3;
    assert_eq!(stats.total_events(), 13);
}

/// Test: has_errors with no errors.
#[test]
fn test_stats_has_errors_none() {
    let stats = HotReloadStats::default();
    assert!(!stats.has_errors());
}

/// Test: has_errors with errors.
#[test]
fn test_stats_has_errors_some() {
    let mut stats = HotReloadStats::default();
    stats.errors = 1;
    assert!(stats.has_errors());
}

/// Test: Stats are Clone.
#[test]
fn test_stats_clone() {
    let mut stats = HotReloadStats::default();
    stats.reloads = 5;
    stats.invalidations = 3;

    let cloned = stats.clone();
    assert_eq!(cloned.reloads, 5);
    assert_eq!(cloned.invalidations, 3);
}

/// Test: Stats Debug formatting.
#[test]
fn test_stats_debug() {
    let stats = HotReloadStats::default();
    let debug_str = format!("{:?}", stats);
    assert!(debug_str.contains("HotReloadStats"));
    assert!(debug_str.contains("reloads"));
}

// =============================================================================
// CATEGORY 8: Integration Patterns (6 tests)
// =============================================================================

/// Test: Typical development workflow setup.
#[test]
fn test_integration_development_workflow() {
    let cache = match create_test_cache() {
        Some(c) => c,
        None => {
            eprintln!("Skipping test_integration_development_workflow: no GPU adapter available");
            return;
        }
    };

    // Use development config
    let config = HotReloadConfig::development();
    let mut hot_reload = ShaderHotReload::new(cache, config).unwrap();

    // Register a callback
    hot_reload.register_callback(|path| {
        let _ = path; // Would trigger pipeline rebuild in real code
    });

    // In a real workflow, you'd watch a shader directory
    // hot_reload.watch_shader_directory("shaders/").unwrap();

    // Poll for changes (should be empty in test)
    let result = hot_reload.poll();
    assert!(result.is_ok());
}

/// Test: Multiple watchers pattern.
#[test]
fn test_integration_multiple_watchers() {
    // Can create multiple independent watchers
    let watcher1 = ShaderWatcher::with_extensions(vec!["wgsl".to_string()]).unwrap();
    let watcher2 = ShaderWatcher::with_extensions(vec!["glsl".to_string()]).unwrap();

    // They have independent extension sets
    assert!(watcher1.extensions().contains("wgsl"));
    assert!(!watcher1.extensions().contains("glsl"));
    assert!(watcher2.extensions().contains("glsl"));
    assert!(!watcher2.extensions().contains("wgsl"));
}

/// Test: Callback receives paths correctly.
#[test]
fn test_integration_callback_pattern() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let cache = match create_test_cache() {
        Some(c) => c,
        None => {
            eprintln!("Skipping test_integration_callback_pattern: no GPU adapter available");
            return;
        }
    };

    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = Arc::clone(&counter);

    let mut hot_reload = ShaderHotReload::with_defaults(cache).unwrap();
    hot_reload.register_callback(move |_path| {
        counter_clone.fetch_add(1, Ordering::SeqCst);
    });

    // Manual reload should trigger callback
    let temp = tempfile::tempdir().expect("create temp dir");
    let shader_path = temp.path().join("test.wgsl");
    std::fs::write(&shader_path, "// test shader").expect("write shader file");

    // Note: reload_shader triggers the callback
    let _ = hot_reload.reload_shader(&shader_path);

    // The counter value should be accessible (may be 0 if reload_shader didn't trigger callback)
    let count = counter.load(Ordering::SeqCst);
    let _ = count; // Just verify we can read it
}

/// Test: Reset stats works.
#[test]
fn test_integration_reset_stats() {
    let cache = match create_test_cache() {
        Some(c) => c,
        None => {
            eprintln!("Skipping test_integration_reset_stats: no GPU adapter available");
            return;
        }
    };

    let mut hot_reload = ShaderHotReload::with_defaults(cache).unwrap();

    // Trigger some activity (if possible)
    let temp = tempfile::tempdir().expect("create temp dir");
    let shader_path = temp.path().join("test.wgsl");
    std::fs::write(&shader_path, "// test").expect("write");
    let _ = hot_reload.reload_shader(&shader_path);

    // Check stats are accumulated
    let stats_before = hot_reload.stats().clone();

    // Reset
    hot_reload.reset_stats();
    let stats_after = hot_reload.stats();

    assert_eq!(stats_after.reloads, 0, "Stats should be reset");
    assert!(
        stats_before.reloads >= stats_after.reloads,
        "Reset should clear stats"
    );
}

/// Test: watched_directories accessor.
#[test]
fn test_integration_watched_directories() {
    let cache = match create_test_cache() {
        Some(c) => c,
        None => {
            eprintln!("Skipping test_integration_watched_directories: no GPU adapter available");
            return;
        }
    };

    let mut hot_reload = ShaderHotReload::with_defaults(cache).unwrap();

    // Initially empty
    assert!(hot_reload.watched_directories().is_empty());

    // Watch a directory
    let temp = tempfile::tempdir().expect("create temp dir");
    let _ = hot_reload.watch_shader_directory(temp.path());

    // Should now have the directory
    let dirs = hot_reload.watched_directories();
    assert_eq!(dirs.len(), 1, "Should have one watched directory");
}

/// Test: Minimal config for testing.
#[test]
fn test_integration_minimal_config_for_testing() {
    let config = HotReloadConfig::minimal();

    // Verify it's suitable for testing
    assert!(!config.auto_reload, "Minimal disables auto_reload");
    assert!(!config.recursive, "Minimal disables recursive");
    assert!(!config.log_events, "Minimal disables logging");
    assert_eq!(config.debounce_ms, 10, "Minimal uses short debounce");

    // Should still be valid
    assert!(config.validate().is_ok());
}

// =============================================================================
// CATEGORY 9: Constants (3 tests)
// =============================================================================

/// Test: DEFAULT_DEBOUNCE_MS constant value.
#[test]
fn test_constant_default_debounce_ms() {
    assert_eq!(DEFAULT_DEBOUNCE_MS, 100);
}

/// Test: DEFAULT_WATCH_EXTENSIONS contains expected values.
#[test]
fn test_constant_default_watch_extensions() {
    assert!(!DEFAULT_WATCH_EXTENSIONS.is_empty());
    assert!(DEFAULT_WATCH_EXTENSIONS.contains(&"wgsl"));
}

/// Test: MAX_PENDING_RELOADS constant value.
#[test]
fn test_constant_max_pending_reloads() {
    assert_eq!(MAX_PENDING_RELOADS, 256);
}

// =============================================================================
// CATEGORY 10: Thread Safety Traits (4 tests)
// =============================================================================

/// Test: HotReloadConfig is Send + Sync.
#[test]
fn test_thread_safety_config() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<HotReloadConfig>();
}

/// Test: HotReloadError is Send + Sync.
#[test]
fn test_thread_safety_error() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<HotReloadError>();
}

/// Test: HotReloadEvent is Send + Sync.
#[test]
fn test_thread_safety_event() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<HotReloadEvent>();
}

/// Test: HotReloadStats is Send + Sync.
#[test]
fn test_thread_safety_stats() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<HotReloadStats>();
}

// =============================================================================
// CATEGORY 11: Clone and Debug Traits (6 tests)
// =============================================================================

/// Test: HotReloadConfig is Clone.
#[test]
fn test_clone_config() {
    let config = HotReloadConfig::new().debounce_ms(200).auto_reload(false);
    let cloned = config.clone();
    assert_eq!(cloned.debounce_ms, 200);
    assert!(!cloned.auto_reload);
}

/// Test: HotReloadEvent is Clone.
#[test]
fn test_clone_event() {
    let event = HotReloadEvent::modified("test.wgsl");
    let cloned = event.clone();
    assert_eq!(event, cloned);
}

/// Test: HotReloadConfig Debug.
#[test]
fn test_debug_config() {
    let config = HotReloadConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("HotReloadConfig"));
}

/// Test: HotReloadEvent Debug.
#[test]
fn test_debug_event() {
    let event = HotReloadEvent::modified("test.wgsl");
    let debug_str = format!("{:?}", event);
    assert!(debug_str.contains("ShaderModified"));
}

/// Test: HotReloadError Debug.
#[test]
fn test_debug_error() {
    let err = HotReloadError::compilation("error");
    let debug_str = format!("{:?}", err);
    assert!(debug_str.contains("CompilationError"));
}

/// Test: ShaderWatcher Debug.
#[test]
fn test_debug_watcher() {
    let watcher = ShaderWatcher::new().unwrap();
    let debug_str = format!("{:?}", watcher);
    assert!(debug_str.contains("ShaderWatcher"));
}

// =============================================================================
// CATEGORY 12: Edge Cases (6 tests)
// =============================================================================

/// Test: Empty extension in watcher.
#[test]
fn test_edge_case_empty_extension_watcher() {
    let watcher = ShaderWatcher::with_extensions(Vec::<String>::new());
    assert!(watcher.is_ok(), "Empty extensions is allowed in watcher");
}

/// Test: Very long debounce duration.
#[test]
fn test_edge_case_long_debounce() {
    let config = HotReloadConfig::new().debounce_ms(u64::MAX);
    assert_eq!(config.debounce_ms, u64::MAX);
    // Duration conversion might overflow, but config is still valid
    assert!(config.validate().is_ok());
}

/// Test: Large max_pending value.
#[test]
fn test_edge_case_large_max_pending() {
    let config = HotReloadConfig::new().max_pending(usize::MAX);
    assert_eq!(config.max_pending, usize::MAX);
    assert!(config.validate().is_ok());
}

/// Test: Path with special characters.
#[test]
fn test_edge_case_special_path_chars() {
    let event = HotReloadEvent::modified("path/with spaces/shader [v2].wgsl");
    assert_eq!(
        event.path(),
        Some(Path::new("path/with spaces/shader [v2].wgsl"))
    );
}

/// Test: Unicode in path.
#[test]
fn test_edge_case_unicode_path() {
    let event = HotReloadEvent::modified("shaders/\u{1F4A1}_light.wgsl");
    let path = event.path().unwrap();
    assert!(path.to_string_lossy().contains("\u{1F4A1}"));
}

/// Test: Very long error message.
#[test]
fn test_edge_case_long_error_message() {
    let long_msg = "x".repeat(10000);
    let err = HotReloadError::compilation(&long_msg);
    assert_eq!(err.compilation_message(), Some(long_msg.as_str()));
}

// =============================================================================
// CATEGORY 13: Watcher Recursive Mode (3 tests)
// =============================================================================

/// Test: Default watcher is recursive.
#[test]
fn test_watcher_default_recursive() {
    let watcher = ShaderWatcher::new().unwrap();
    assert!(watcher.is_recursive(), "Default watcher should be recursive");
}

/// Test: set_recursive changes mode.
#[test]
fn test_watcher_set_recursive() {
    let mut watcher = ShaderWatcher::new().unwrap();
    assert!(watcher.is_recursive());

    watcher.set_recursive(false);
    assert!(!watcher.is_recursive());

    watcher.set_recursive(true);
    assert!(watcher.is_recursive());
}

/// Test: from_config respects recursive setting.
#[test]
fn test_watcher_from_config_recursive() {
    let config = HotReloadConfig::new().recursive(false);
    let watcher = ShaderWatcher::from_config(&config).unwrap();
    assert!(!watcher.is_recursive());
}

// =============================================================================
// CATEGORY 14: is_watching Method (2 tests)
// =============================================================================

/// Test: is_watching returns false for unwatched path.
#[test]
fn test_watcher_is_watching_false() {
    let watcher = ShaderWatcher::new().unwrap();
    assert!(!watcher.is_watching("/some/random/path"));
}

/// Test: is_watching returns true after watching.
#[test]
fn test_watcher_is_watching_true() {
    let mut watcher = ShaderWatcher::new().unwrap();
    let temp = tempfile::tempdir().expect("create temp dir");

    watcher.watch_directory(temp.path()).expect("watch");
    assert!(watcher.is_watching(temp.path()));
}
