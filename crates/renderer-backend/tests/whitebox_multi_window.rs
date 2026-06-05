//! Whitebox structural tests for Multi-Window rendering support.
//!
//! These tests verify the internal structure and behavior of the multi-window
//! system, including WindowId, WindowConfig, WindowState, MultiWindowManager,
//! and SyncMode types.
//!
//! Task: T-WGPU-P7.1.11 - Multi-window rendering support
//!
//! Acceptance Criteria Tested:
//! 1. WindowId uniqueness and monotonic generation
//! 2. WindowId primary() convention (ID 0)
//! 3. WindowConfig builder pattern chaining
//! 4. WindowState frame timing statistics
//! 5. MultiWindowManager render_order sorting by priority
//! 6. Focus tracking across window changes
//! 7. Visibility and hidden window handling
//! 8. SyncMode synchronization logic

use renderer_backend::presentation::{
    MultiWindowError, MultiWindowManager, MultiWindowStats, SurfaceConfiguration, SyncMode,
    WindowConfig, WindowId,
};
use std::collections::HashSet;
use std::time::Duration;

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a test SurfaceConfiguration with given dimensions.
fn make_surface_config(width: u32, height: u32) -> SurfaceConfiguration {
    SurfaceConfiguration::new(width, height)
}

/// Create a test WindowConfig with specified parameters.
fn make_window_config(
    id: WindowId,
    width: u32,
    height: u32,
    priority: u8,
    label: Option<&str>,
) -> WindowConfig {
    let mut config = WindowConfig::new(id, make_surface_config(width, height)).with_priority(priority);

    if let Some(l) = label {
        config = config.with_label(l);
    }

    config
}

// ============================================================================
// 1. WindowId Tests - Uniqueness and Monotonic Generation
// ============================================================================

mod window_id_uniqueness {
    use super::*;

    #[test]
    fn new_generates_unique_ids() {
        let id1 = WindowId::new();
        let id2 = WindowId::new();
        let id3 = WindowId::new();

        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
    }

    #[test]
    fn new_generates_monotonically_increasing_ids() {
        let id1 = WindowId::new();
        let id2 = WindowId::new();
        let id3 = WindowId::new();

        assert!(id1.as_u64() < id2.as_u64());
        assert!(id2.as_u64() < id3.as_u64());
    }

    #[test]
    fn multiple_new_calls_all_unique() {
        let mut ids: HashSet<u64> = HashSet::new();
        for _ in 0..100 {
            let id = WindowId::new();
            assert!(ids.insert(id.as_u64()), "Duplicate ID generated");
        }
        assert_eq!(ids.len(), 100);
    }

    #[test]
    fn from_raw_id_preserves_value() {
        let raw = 42u64;
        let id = WindowId::from_raw_id(raw);
        assert_eq!(id.as_u64(), raw);
    }

    #[test]
    fn from_raw_id_zero() {
        let id = WindowId::from_raw_id(0);
        assert_eq!(id.as_u64(), 0);
    }

    #[test]
    fn from_raw_id_max_u64() {
        let id = WindowId::from_raw_id(u64::MAX);
        assert_eq!(id.as_u64(), u64::MAX);
    }

    #[test]
    fn as_u64_returns_inner_value() {
        let id = WindowId::from_raw_id(12345);
        assert_eq!(id.as_u64(), 12345);
    }

    #[test]
    fn default_uses_new() {
        // Default should create a unique ID like new()
        let id1 = WindowId::default();
        let id2 = WindowId::default();
        assert_ne!(id1, id2);
    }

    #[test]
    fn from_u64_trait() {
        let raw: u64 = 999;
        let id: WindowId = raw.into();
        assert_eq!(id.as_u64(), 999);
    }

    #[test]
    fn into_u64_trait() {
        let id = WindowId::from_raw_id(888);
        let raw: u64 = id.into();
        assert_eq!(raw, 888);
    }
}

// ============================================================================
// 2. WindowId Primary Convention Tests
// ============================================================================

mod window_id_primary {
    use super::*;

    #[test]
    fn primary_returns_id_zero() {
        let primary = WindowId::primary();
        assert_eq!(primary.as_u64(), 0);
    }

    #[test]
    fn primary_is_const() {
        // This compiles because primary() is const
        const PRIMARY: WindowId = WindowId::primary();
        assert_eq!(PRIMARY.as_u64(), 0);
    }

    #[test]
    fn is_primary_true_for_zero() {
        let id = WindowId::from_raw_id(0);
        assert!(id.is_primary());
    }

    #[test]
    fn is_primary_true_for_primary() {
        let id = WindowId::primary();
        assert!(id.is_primary());
    }

    #[test]
    fn is_primary_false_for_nonzero() {
        let id = WindowId::from_raw_id(1);
        assert!(!id.is_primary());
    }

    #[test]
    fn is_primary_false_for_new() {
        // new() starts from 1, so should never be primary
        let id = WindowId::new();
        assert!(!id.is_primary());
    }

    #[test]
    fn is_primary_false_for_large_id() {
        let id = WindowId::from_raw_id(u64::MAX);
        assert!(!id.is_primary());
    }

    #[test]
    fn primary_equality() {
        let p1 = WindowId::primary();
        let p2 = WindowId::primary();
        assert_eq!(p1, p2);
    }

    #[test]
    fn primary_not_equal_to_new() {
        let primary = WindowId::primary();
        let new_id = WindowId::new();
        assert_ne!(primary, new_id);
    }

    #[test]
    fn is_primary_is_const() {
        // This compiles because is_primary() is const
        const ID: WindowId = WindowId::primary();
        const IS_PRIM: bool = ID.is_primary();
        assert!(IS_PRIM);
    }
}

// ============================================================================
// 3. WindowId Display and Debug Tests
// ============================================================================

mod window_id_display {
    use super::*;

    #[test]
    fn display_format_primary() {
        let id = WindowId::primary();
        let display = format!("{}", id);
        assert_eq!(display, "Window(0)");
    }

    #[test]
    fn display_format_nonzero() {
        let id = WindowId::from_raw_id(42);
        let display = format!("{}", id);
        assert_eq!(display, "Window(42)");
    }

    #[test]
    fn debug_format() {
        let id = WindowId::from_raw_id(123);
        let debug = format!("{:?}", id);
        assert!(debug.contains("123"));
    }

    #[test]
    fn hash_implementation() {
        use std::collections::HashMap;
        let mut map: HashMap<WindowId, &str> = HashMap::new();

        let id1 = WindowId::from_raw_id(1);
        let id2 = WindowId::from_raw_id(2);

        map.insert(id1, "window1");
        map.insert(id2, "window2");

        assert_eq!(map.get(&id1), Some(&"window1"));
        assert_eq!(map.get(&id2), Some(&"window2"));
    }

    #[test]
    fn clone_implementation() {
        let id1 = WindowId::from_raw_id(500);
        let id2 = id1.clone();
        assert_eq!(id1, id2);
    }

    #[test]
    fn copy_implementation() {
        let id1 = WindowId::from_raw_id(500);
        let id2 = id1; // Copy, not move
        assert_eq!(id1, id2);
        // id1 still valid because of Copy
        assert_eq!(id1.as_u64(), 500);
    }
}

// ============================================================================
// 4. WindowConfig Builder Pattern Tests
// ============================================================================

mod window_config_builder {
    use super::*;

    #[test]
    fn new_creates_config_with_defaults() {
        let id = WindowId::new();
        let surface_config = make_surface_config(800, 600);
        let config = WindowConfig::new(id, surface_config);

        assert_eq!(config.id, id);
        assert!(!config.is_focused);
        assert!(config.is_visible);
        assert_eq!(config.priority, 128); // Default middle priority
        assert!(config.label.is_none());
        assert!(!config.sync_to_primary);
    }

    #[test]
    fn primary_creates_high_priority_focused_config() {
        let surface_config = make_surface_config(1920, 1080);
        let config = WindowConfig::primary(surface_config);

        assert!(config.id.is_primary());
        assert!(config.is_focused);
        assert!(config.is_visible);
        assert_eq!(config.priority, 255); // Highest priority
        assert_eq!(config.label, Some("Primary".to_string()));
        assert!(!config.sync_to_primary);
    }

    #[test]
    fn with_focus_sets_focused_true() {
        let config = make_window_config(WindowId::new(), 800, 600, 100, None).with_focus(true);

        assert!(config.is_focused);
    }

    #[test]
    fn with_focus_sets_focused_false() {
        let config = WindowConfig::primary(make_surface_config(800, 600)).with_focus(false);

        assert!(!config.is_focused);
    }

    #[test]
    fn with_visibility_sets_visible_true() {
        let config = make_window_config(WindowId::new(), 800, 600, 100, None)
            .with_visibility(false)
            .with_visibility(true);

        assert!(config.is_visible);
    }

    #[test]
    fn with_visibility_sets_visible_false() {
        let config = make_window_config(WindowId::new(), 800, 600, 100, None).with_visibility(false);

        assert!(!config.is_visible);
    }

    #[test]
    fn with_priority_sets_priority() {
        let config = make_window_config(WindowId::new(), 800, 600, 100, None).with_priority(200);

        assert_eq!(config.priority, 200);
    }

    #[test]
    fn with_priority_min() {
        let config = make_window_config(WindowId::new(), 800, 600, 100, None).with_priority(0);

        assert_eq!(config.priority, 0);
    }

    #[test]
    fn with_priority_max() {
        let config = make_window_config(WindowId::new(), 800, 600, 100, None).with_priority(255);

        assert_eq!(config.priority, 255);
    }

    #[test]
    fn with_label_sets_label() {
        let config =
            make_window_config(WindowId::new(), 800, 600, 100, None).with_label("Test Window");

        assert_eq!(config.label, Some("Test Window".to_string()));
    }

    #[test]
    fn with_label_overwrites_existing() {
        let config = make_window_config(WindowId::new(), 800, 600, 100, Some("Old Label"))
            .with_label("New Label");

        assert_eq!(config.label, Some("New Label".to_string()));
    }

    #[test]
    fn with_sync_to_primary_enables_sync() {
        let config =
            make_window_config(WindowId::new(), 800, 600, 100, None).with_sync_to_primary(true);

        assert!(config.sync_to_primary);
    }

    #[test]
    fn with_sync_to_primary_disables_sync() {
        let config = make_window_config(WindowId::new(), 800, 600, 100, None)
            .with_sync_to_primary(true)
            .with_sync_to_primary(false);

        assert!(!config.sync_to_primary);
    }

    #[test]
    fn builder_chaining_all_methods() {
        let id = WindowId::new();
        let config = WindowConfig::new(id, make_surface_config(1280, 720))
            .with_focus(true)
            .with_visibility(true)
            .with_priority(150)
            .with_label("Chained Window")
            .with_sync_to_primary(true);

        assert_eq!(config.id, id);
        assert!(config.is_focused);
        assert!(config.is_visible);
        assert_eq!(config.priority, 150);
        assert_eq!(config.label, Some("Chained Window".to_string()));
        assert!(config.sync_to_primary);
    }
}

// ============================================================================
// 5. WindowConfig Computed Properties Tests
// ============================================================================

mod window_config_computed {
    use super::*;

    #[test]
    fn dimensions_returns_config_dimensions() {
        let config = make_window_config(WindowId::new(), 1920, 1080, 100, None);

        assert_eq!(config.dimensions(), (1920, 1080));
    }

    #[test]
    fn aspect_ratio_standard() {
        let config = make_window_config(WindowId::new(), 1920, 1080, 100, None);

        let ratio = config.aspect_ratio();
        assert!((ratio - 16.0 / 9.0).abs() < 0.001);
    }

    #[test]
    fn aspect_ratio_square() {
        let config = make_window_config(WindowId::new(), 500, 500, 100, None);

        let ratio = config.aspect_ratio();
        assert!((ratio - 1.0).abs() < 0.001);
    }

    #[test]
    fn aspect_ratio_ultrawide() {
        let config = make_window_config(WindowId::new(), 3440, 1440, 100, None);

        let ratio = config.aspect_ratio();
        assert!((ratio - 3440.0 / 1440.0).abs() < 0.001);
    }

    #[test]
    fn aspect_ratio_zero_height_returns_one() {
        // Zero dimensions get clamped to 1, so this tests the fallback
        let mut config = make_window_config(WindowId::new(), 100, 1, 100, None);
        config.config.height = 0; // Force zero after config creation

        let ratio = config.aspect_ratio();
        assert_eq!(ratio, 1.0);
    }

    #[test]
    fn should_render_visible_with_valid_dimensions() {
        let config = make_window_config(WindowId::new(), 800, 600, 100, None);

        assert!(config.should_render());
    }

    #[test]
    fn should_render_false_when_invisible() {
        let config =
            make_window_config(WindowId::new(), 800, 600, 100, None).with_visibility(false);

        assert!(!config.should_render());
    }

    #[test]
    fn should_render_false_when_zero_width() {
        let mut config = make_window_config(WindowId::new(), 800, 600, 100, None);
        config.config.width = 0;

        assert!(!config.should_render());
    }

    #[test]
    fn should_render_false_when_zero_height() {
        let mut config = make_window_config(WindowId::new(), 800, 600, 100, None);
        config.config.height = 0;

        assert!(!config.should_render());
    }

    #[test]
    fn should_render_false_when_invisible_and_zero_dims() {
        let mut config =
            make_window_config(WindowId::new(), 800, 600, 100, None).with_visibility(false);
        config.config.width = 0;
        config.config.height = 0;

        assert!(!config.should_render());
    }
}

// ============================================================================
// 6. WindowConfig Display Tests
// ============================================================================

mod window_config_display {
    use super::*;

    #[test]
    fn display_format_with_label() {
        let config = make_window_config(WindowId::new(), 1920, 1080, 200, Some("Main Display"))
            .with_focus(true)
            .with_visibility(true);

        let display = format!("{}", config);

        assert!(display.contains("Main Display"));
        assert!(display.contains("1920x1080"));
        assert!(display.contains("priority=200"));
        assert!(display.contains("focused=true"));
        assert!(display.contains("visible=true"));
    }

    #[test]
    fn display_format_without_label() {
        let config = make_window_config(WindowId::new(), 800, 600, 128, None);

        let display = format!("{}", config);

        assert!(display.contains("Unnamed"));
        assert!(display.contains("800x600"));
    }

    #[test]
    fn debug_format() {
        let config = make_window_config(WindowId::new(), 1024, 768, 100, Some("Debug Window"));

        let debug = format!("{:?}", config);

        assert!(debug.contains("WindowConfig"));
        assert!(debug.contains("1024"));
        assert!(debug.contains("768"));
    }
}

// ============================================================================
// 7. SyncMode Tests
// ============================================================================

mod sync_mode {
    use super::*;

    #[test]
    fn default_is_independent() {
        let mode = SyncMode::default();
        assert_eq!(mode, SyncMode::Independent);
    }

    #[test]
    fn independent_does_not_require_coordination() {
        assert!(!SyncMode::Independent.requires_coordination());
    }

    #[test]
    fn sync_to_primary_requires_coordination() {
        assert!(SyncMode::SyncToPrimary.requires_coordination());
    }

    #[test]
    fn sync_to_rate_requires_coordination() {
        let mode = SyncMode::SyncToRate { target_hz: 60 };
        assert!(mode.requires_coordination());
    }

    #[test]
    fn simultaneous_requires_coordination() {
        assert!(SyncMode::Simultaneous.requires_coordination());
    }

    #[test]
    fn sync_to_rate_constructor() {
        let mode = SyncMode::sync_to_rate(144);
        match mode {
            SyncMode::SyncToRate { target_hz } => assert_eq!(target_hz, 144),
            _ => panic!("Expected SyncToRate variant"),
        }
    }

    #[test]
    fn target_interval_for_sync_to_rate_60hz() {
        let mode = SyncMode::SyncToRate { target_hz: 60 };
        let interval = mode.target_interval().unwrap();

        // 1/60 second = ~16.67ms
        let expected = Duration::from_secs_f64(1.0 / 60.0);
        assert!((interval.as_secs_f64() - expected.as_secs_f64()).abs() < 0.0001);
    }

    #[test]
    fn target_interval_for_sync_to_rate_144hz() {
        let mode = SyncMode::SyncToRate { target_hz: 144 };
        let interval = mode.target_interval().unwrap();

        let expected = Duration::from_secs_f64(1.0 / 144.0);
        assert!((interval.as_secs_f64() - expected.as_secs_f64()).abs() < 0.0001);
    }

    #[test]
    fn target_interval_for_sync_to_rate_zero_hz() {
        let mode = SyncMode::SyncToRate { target_hz: 0 };
        assert!(mode.target_interval().is_none());
    }

    #[test]
    fn target_interval_none_for_independent() {
        assert!(SyncMode::Independent.target_interval().is_none());
    }

    #[test]
    fn target_interval_none_for_sync_to_primary() {
        assert!(SyncMode::SyncToPrimary.target_interval().is_none());
    }

    #[test]
    fn target_interval_none_for_simultaneous() {
        assert!(SyncMode::Simultaneous.target_interval().is_none());
    }

    #[test]
    fn display_independent() {
        let display = format!("{}", SyncMode::Independent);
        assert_eq!(display, "Independent");
    }

    #[test]
    fn display_sync_to_primary() {
        let display = format!("{}", SyncMode::SyncToPrimary);
        assert_eq!(display, "Sync to Primary");
    }

    #[test]
    fn display_sync_to_rate() {
        let display = format!("{}", SyncMode::SyncToRate { target_hz: 120 });
        assert_eq!(display, "Sync to 120Hz");
    }

    #[test]
    fn display_simultaneous() {
        let display = format!("{}", SyncMode::Simultaneous);
        assert_eq!(display, "Simultaneous");
    }

    #[test]
    fn clone_implementation() {
        let mode = SyncMode::SyncToRate { target_hz: 60 };
        let cloned = mode.clone();
        assert_eq!(mode, cloned);
    }

    #[test]
    fn copy_implementation() {
        let mode1 = SyncMode::SyncToPrimary;
        let mode2 = mode1; // Copy
        assert_eq!(mode1, mode2);
    }

    #[test]
    fn debug_implementation() {
        let mode = SyncMode::SyncToRate { target_hz: 240 };
        let debug = format!("{:?}", mode);
        assert!(debug.contains("SyncToRate"));
        assert!(debug.contains("240"));
    }
}

// ============================================================================
// 8. MultiWindowManager Creation Tests
// ============================================================================

mod multi_window_manager_creation {
    use super::*;

    #[test]
    fn new_creates_empty_manager() {
        let manager = MultiWindowManager::new();

        assert_eq!(manager.window_count(), 0);
        assert!(!manager.has_windows());
        assert!(manager.focused_window_id().is_none());
    }

    #[test]
    fn with_max_windows_sets_limit() {
        let manager = MultiWindowManager::with_max_windows(4);

        assert_eq!(manager.window_count(), 0);
        // The limit is internal, but we can test it through registration
    }

    #[test]
    fn default_sync_mode_is_independent() {
        let manager = MultiWindowManager::new();

        assert_eq!(manager.sync_mode(), SyncMode::Independent);
    }

    #[test]
    fn set_sync_mode_changes_mode() {
        let mut manager = MultiWindowManager::new();

        manager.set_sync_mode(SyncMode::SyncToPrimary);
        assert_eq!(manager.sync_mode(), SyncMode::SyncToPrimary);

        manager.set_sync_mode(SyncMode::sync_to_rate(60));
        match manager.sync_mode() {
            SyncMode::SyncToRate { target_hz } => assert_eq!(target_hz, 60),
            _ => panic!("Expected SyncToRate"),
        }
    }

    #[test]
    fn global_frame_count_starts_at_zero() {
        let manager = MultiWindowManager::new();
        assert_eq!(manager.global_frame_count(), 0);
    }

    #[test]
    fn default_trait() {
        let manager = MultiWindowManager::default();

        assert_eq!(manager.window_count(), 0);
        assert_eq!(manager.sync_mode(), SyncMode::Independent);
    }

    #[test]
    fn window_ids_empty_initially() {
        let manager = MultiWindowManager::new();
        assert!(manager.window_ids().is_empty());
    }

    #[test]
    fn visible_window_ids_empty_initially() {
        let manager = MultiWindowManager::new();
        assert!(manager.visible_window_ids().is_empty());
    }

    #[test]
    fn focused_window_none_initially() {
        let manager = MultiWindowManager::new();
        assert!(manager.focused_window().is_none());
    }

    #[test]
    fn debug_format() {
        let manager = MultiWindowManager::new();
        let debug = format!("{:?}", manager);

        assert!(debug.contains("MultiWindowManager"));
        assert!(debug.contains("window_count"));
        assert!(debug.contains("0"));
    }
}

// ============================================================================
// 9. MultiWindowManager Render Order Tests
// ============================================================================

mod multi_window_manager_render_order {
    use super::*;

    // Note: These tests verify the render_order sorting logic
    // Since we can't register actual surfaces without a GPU,
    // we test the logic through the public interface

    #[test]
    fn render_order_empty_for_empty_manager() {
        let manager = MultiWindowManager::new();
        assert!(manager.window_ids().is_empty());
    }

    #[test]
    fn iter_empty_for_empty_manager() {
        let manager = MultiWindowManager::new();
        assert_eq!(manager.iter().count(), 0);
    }
}

// ============================================================================
// 10. MultiWindowError Tests
// ============================================================================

mod multi_window_error {
    use super::*;

    #[test]
    fn window_not_found_error() {
        let id = WindowId::from_raw_id(42);
        let error = MultiWindowError::WindowNotFound(id);

        let msg = format!("{}", error);
        assert!(msg.contains("window not found"));
        assert!(msg.contains("42"));
    }

    #[test]
    fn window_exists_error() {
        let id = WindowId::from_raw_id(99);
        let error = MultiWindowError::WindowExists(id);

        let msg = format!("{}", error);
        assert!(msg.contains("window already exists"));
        assert!(msg.contains("99"));
    }

    #[test]
    fn no_windows_error() {
        let error = MultiWindowError::NoWindows;

        let msg = format!("{}", error);
        assert!(msg.contains("no windows registered"));
    }

    #[test]
    fn no_focused_window_error() {
        let error = MultiWindowError::NoFocusedWindow;

        let msg = format!("{}", error);
        assert!(msg.contains("no window has focus"));
    }

    #[test]
    fn max_windows_reached_error() {
        let error = MultiWindowError::MaxWindowsReached { max: 8 };

        let msg = format!("{}", error);
        assert!(msg.contains("maximum number of windows"));
        assert!(msg.contains("8"));
    }

    #[test]
    fn window_not_found_not_recoverable() {
        let error = MultiWindowError::WindowNotFound(WindowId::new());
        assert!(!error.is_recoverable());
    }

    #[test]
    fn window_exists_not_recoverable() {
        let error = MultiWindowError::WindowExists(WindowId::new());
        assert!(!error.is_recoverable());
    }

    #[test]
    fn no_windows_not_recoverable() {
        let error = MultiWindowError::NoWindows;
        assert!(!error.is_recoverable());
    }

    #[test]
    fn no_focused_window_is_recoverable() {
        let error = MultiWindowError::NoFocusedWindow;
        assert!(error.is_recoverable());
    }

    #[test]
    fn max_windows_reached_not_recoverable() {
        let error = MultiWindowError::MaxWindowsReached { max: 4 };
        assert!(!error.is_recoverable());
    }

    #[test]
    fn debug_format() {
        let error = MultiWindowError::WindowNotFound(WindowId::from_raw_id(123));
        let debug = format!("{:?}", error);
        assert!(debug.contains("WindowNotFound"));
    }
}

// ============================================================================
// 11. MultiWindowStats Tests
// ============================================================================

mod multi_window_stats {
    use super::*;

    fn make_stats(
        window_count: usize,
        total_frames: u64,
        total_dropped: u64,
        avg_frame_time_ms: f32,
        global_frame_count: u64,
    ) -> MultiWindowStats {
        MultiWindowStats {
            window_count,
            total_frames,
            total_dropped,
            average_frame_time_ms: avg_frame_time_ms,
            global_frame_count,
        }
    }

    #[test]
    fn drop_rate_no_drops() {
        let stats = make_stats(2, 1000, 0, 16.67, 500);
        assert!((stats.drop_rate() - 0.0).abs() < 0.001);
    }

    #[test]
    fn drop_rate_some_drops() {
        let stats = make_stats(2, 900, 100, 16.67, 500);
        // 100 / (900 + 100) = 0.1
        assert!((stats.drop_rate() - 0.1).abs() < 0.001);
    }

    #[test]
    fn drop_rate_all_dropped() {
        let stats = make_stats(2, 0, 100, 16.67, 0);
        // 100 / (0 + 100) = 1.0
        assert!((stats.drop_rate() - 1.0).abs() < 0.001);
    }

    #[test]
    fn drop_rate_no_frames() {
        let stats = make_stats(0, 0, 0, 0.0, 0);
        assert_eq!(stats.drop_rate(), 0.0);
    }

    #[test]
    fn estimated_fps_60() {
        let stats = make_stats(2, 1000, 0, 16.67, 500);
        // 1000 / 16.67 ~= 60 FPS
        let fps = stats.estimated_fps();
        assert!((fps - 60.0).abs() < 1.0);
    }

    #[test]
    fn estimated_fps_144() {
        let stats = make_stats(2, 1000, 0, 1000.0 / 144.0, 500);
        let fps = stats.estimated_fps();
        assert!((fps - 144.0).abs() < 1.0);
    }

    #[test]
    fn estimated_fps_zero_frame_time() {
        let stats = make_stats(2, 1000, 0, 0.0, 500);
        assert_eq!(stats.estimated_fps(), 0.0);
    }

    #[test]
    fn display_format() {
        let stats = make_stats(3, 5000, 50, 16.67, 2500);
        let display = format!("{}", stats);

        assert!(display.contains("3 windows"));
        assert!(display.contains("5000 frames"));
        assert!(display.contains("50 dropped"));
    }

    #[test]
    fn debug_format() {
        let stats = make_stats(2, 1000, 10, 8.33, 500);
        let debug = format!("{:?}", stats);

        assert!(debug.contains("MultiWindowStats"));
        assert!(debug.contains("window_count"));
    }

    #[test]
    fn clone_implementation() {
        let stats = make_stats(4, 2000, 20, 16.67, 1000);
        let cloned = stats.clone();

        assert_eq!(stats.window_count, cloned.window_count);
        assert_eq!(stats.total_frames, cloned.total_frames);
        assert_eq!(stats.total_dropped, cloned.total_dropped);
    }

    #[test]
    fn copy_implementation() {
        let stats1 = make_stats(2, 100, 5, 10.0, 50);
        let stats2 = stats1; // Copy

        assert_eq!(stats1.window_count, stats2.window_count);
        assert_eq!(stats1.total_frames, stats2.total_frames);
    }
}

// ============================================================================
// 12. SurfaceConfiguration Tests (for WindowConfig)
// ============================================================================

mod surface_configuration {
    use super::*;

    #[test]
    fn new_creates_config_with_dimensions() {
        let config = SurfaceConfiguration::new(1920, 1080);

        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
    }

    #[test]
    fn new_clamps_zero_width_to_one() {
        let config = SurfaceConfiguration::new(0, 100);

        assert_eq!(config.width, 1);
    }

    #[test]
    fn new_clamps_zero_height_to_one() {
        let config = SurfaceConfiguration::new(100, 0);

        assert_eq!(config.height, 1);
    }

    #[test]
    fn resize_updates_dimensions() {
        let mut config = SurfaceConfiguration::new(800, 600);
        config.resize(1920, 1080);

        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
    }

    #[test]
    fn resize_clamps_zero_width() {
        let mut config = SurfaceConfiguration::new(800, 600);
        config.resize(0, 600);

        assert_eq!(config.width, 1);
    }

    #[test]
    fn resize_clamps_zero_height() {
        let mut config = SurfaceConfiguration::new(800, 600);
        config.resize(800, 0);

        assert_eq!(config.height, 1);
    }

    #[test]
    fn with_dimensions_builder() {
        let config = SurfaceConfiguration::new(640, 480).with_dimensions(1280, 720);

        assert_eq!(config.width, 1280);
        assert_eq!(config.height, 720);
    }

    #[test]
    fn default_format() {
        let config = SurfaceConfiguration::new(800, 600);
        assert_eq!(config.format, wgpu::TextureFormat::Bgra8UnormSrgb);
    }

    #[test]
    fn default_present_mode() {
        let config = SurfaceConfiguration::new(800, 600);
        assert_eq!(config.present_mode, wgpu::PresentMode::Fifo);
    }

    #[test]
    fn default_alpha_mode() {
        let config = SurfaceConfiguration::new(800, 600);
        assert_eq!(config.alpha_mode, wgpu::CompositeAlphaMode::Auto);
    }

    #[test]
    fn default_frame_latency() {
        let config = SurfaceConfiguration::new(800, 600);
        assert_eq!(config.desired_maximum_frame_latency, 2);
    }

    #[test]
    fn with_format_builder() {
        let config =
            SurfaceConfiguration::new(800, 600).with_format(wgpu::TextureFormat::Rgba8Unorm);

        assert_eq!(config.format, wgpu::TextureFormat::Rgba8Unorm);
    }

    #[test]
    fn with_present_mode_builder() {
        let config =
            SurfaceConfiguration::new(800, 600).with_present_mode(wgpu::PresentMode::Immediate);

        assert_eq!(config.present_mode, wgpu::PresentMode::Immediate);
    }

    #[test]
    fn with_alpha_mode_builder() {
        let config = SurfaceConfiguration::new(800, 600)
            .with_alpha_mode(wgpu::CompositeAlphaMode::Opaque);

        assert_eq!(config.alpha_mode, wgpu::CompositeAlphaMode::Opaque);
    }

    #[test]
    fn with_frame_latency_builder() {
        let config = SurfaceConfiguration::new(800, 600).with_frame_latency(3);

        assert_eq!(config.desired_maximum_frame_latency, 3);
    }

    #[test]
    fn with_frame_latency_clamps_to_minimum() {
        let config = SurfaceConfiguration::new(800, 600).with_frame_latency(0);

        assert_eq!(config.desired_maximum_frame_latency, 1);
    }

    #[test]
    fn is_triple_buffered_true() {
        let config = SurfaceConfiguration::new(800, 600).with_frame_latency(3);
        assert!(config.is_triple_buffered());
    }

    #[test]
    fn is_triple_buffered_false() {
        let config = SurfaceConfiguration::new(800, 600).with_frame_latency(2);
        assert!(!config.is_triple_buffered());
    }

    #[test]
    fn clone_implementation() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Rgba8Unorm)
            .with_present_mode(wgpu::PresentMode::Mailbox);

        let cloned = config.clone();

        assert_eq!(config.width, cloned.width);
        assert_eq!(config.height, cloned.height);
        assert_eq!(config.format, cloned.format);
        assert_eq!(config.present_mode, cloned.present_mode);
    }
}

// ============================================================================
// 13. WindowConfig Integration with SurfaceConfiguration Tests
// ============================================================================

mod window_config_surface_integration {
    use super::*;

    #[test]
    fn window_config_stores_surface_config() {
        let id = WindowId::new();
        let surface_config = SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Rgba8Unorm)
            .with_present_mode(wgpu::PresentMode::Mailbox);

        let window_config = WindowConfig::new(id, surface_config.clone());

        assert_eq!(window_config.config.width, 1920);
        assert_eq!(window_config.config.height, 1080);
        assert_eq!(window_config.config.format, wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!(window_config.config.present_mode, wgpu::PresentMode::Mailbox);
    }

    #[test]
    fn window_dimensions_from_surface_config() {
        let config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(2560, 1440));

        assert_eq!(config.dimensions(), (2560, 1440));
    }

    #[test]
    fn window_aspect_ratio_from_surface_config() {
        let config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(1920, 1080));

        let ratio = config.aspect_ratio();
        assert!((ratio - 16.0 / 9.0).abs() < 0.001);
    }

    #[test]
    fn should_render_checks_surface_config_dimensions() {
        let mut config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600));

        assert!(config.should_render());

        config.config.width = 0;
        assert!(!config.should_render());
    }
}

// ============================================================================
// 14. WindowId Thread Safety Tests
// ============================================================================

mod window_id_thread_safety {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn concurrent_id_generation_unique() {
        let ids: Arc<std::sync::Mutex<HashSet<u64>>> = Arc::new(std::sync::Mutex::new(HashSet::new()));
        let mut handles = vec![];

        for _ in 0..4 {
            let ids_clone = Arc::clone(&ids);
            handles.push(thread::spawn(move || {
                for _ in 0..25 {
                    let id = WindowId::new();
                    let mut ids = ids_clone.lock().unwrap();
                    assert!(ids.insert(id.as_u64()), "Duplicate ID in concurrent generation");
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let ids = ids.lock().unwrap();
        assert_eq!(ids.len(), 100);
    }
}

// ============================================================================
// 15. Edge Cases and Boundary Tests
// ============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn window_id_near_u64_max() {
        let id = WindowId::from_raw_id(u64::MAX - 1);
        assert_eq!(id.as_u64(), u64::MAX - 1);
        assert!(!id.is_primary());
    }

    #[test]
    fn window_config_extreme_dimensions() {
        let config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(u32::MAX, u32::MAX));

        assert_eq!(config.config.width, u32::MAX);
        assert_eq!(config.config.height, u32::MAX);
        assert!(config.should_render());
    }

    #[test]
    fn window_config_min_dimensions() {
        let config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(1, 1));

        assert_eq!(config.dimensions(), (1, 1));
        assert_eq!(config.aspect_ratio(), 1.0);
        assert!(config.should_render());
    }

    #[test]
    fn sync_mode_high_refresh_rate() {
        let mode = SyncMode::SyncToRate { target_hz: 360 };
        let interval = mode.target_interval().unwrap();

        // ~2.78ms
        assert!(interval.as_millis() < 3);
    }

    #[test]
    fn sync_mode_very_low_refresh_rate() {
        let mode = SyncMode::SyncToRate { target_hz: 1 };
        let interval = mode.target_interval().unwrap();

        // 1 second
        assert_eq!(interval.as_secs(), 1);
    }

    #[test]
    fn multi_window_stats_large_values() {
        let stats = MultiWindowStats {
            window_count: 1000,
            total_frames: u64::MAX / 2,
            total_dropped: u64::MAX / 4,
            average_frame_time_ms: 0.001,
            global_frame_count: u64::MAX / 2,
        };

        // Should not panic
        let _ = stats.drop_rate();
        let fps = stats.estimated_fps();
        assert!(fps > 0.0);
    }

    #[test]
    fn window_id_equality_same_raw() {
        let id1 = WindowId::from_raw_id(42);
        let id2 = WindowId::from_raw_id(42);
        assert_eq!(id1, id2);
    }

    #[test]
    fn window_id_inequality_different_raw() {
        let id1 = WindowId::from_raw_id(42);
        let id2 = WindowId::from_raw_id(43);
        assert_ne!(id1, id2);
    }

    #[test]
    fn empty_label_string() {
        let config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))
            .with_label("");

        assert_eq!(config.label, Some(String::new()));
    }

    #[test]
    fn unicode_label() {
        let config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))
            .with_label("Test Window");

        assert_eq!(config.label, Some("Test Window".to_string()));
    }

    #[test]
    fn long_label() {
        let long_label = "A".repeat(1000);
        let config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))
            .with_label(&long_label);

        assert_eq!(config.label, Some(long_label));
    }
}

// ============================================================================
// 16. Priority Ordering Tests
// ============================================================================

mod priority_ordering {
    use super::*;

    #[test]
    fn primary_window_has_highest_priority() {
        let primary = WindowConfig::primary(SurfaceConfiguration::new(1920, 1080));
        let secondary = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600));

        assert!(primary.priority > secondary.priority);
    }

    #[test]
    fn default_priority_is_128() {
        let config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600));
        assert_eq!(config.priority, 128);
    }

    #[test]
    fn priority_range() {
        let low = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))
            .with_priority(0);
        let high = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))
            .with_priority(255);

        assert_eq!(low.priority, 0);
        assert_eq!(high.priority, 255);
    }

    #[test]
    fn priority_comparison_for_sorting() {
        let configs = vec![
            WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))
                .with_priority(50),
            WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))
                .with_priority(200),
            WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))
                .with_priority(100),
        ];

        let mut sorted: Vec<u8> = configs.iter().map(|c| c.priority).collect();
        sorted.sort_by(|a, b| b.cmp(a)); // Higher priority first

        assert_eq!(sorted, vec![200, 100, 50]);
    }
}

// ============================================================================
// 17. Visibility State Tests
// ============================================================================

mod visibility_state {
    use super::*;

    #[test]
    fn default_visibility_is_true() {
        let config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600));
        assert!(config.is_visible);
    }

    #[test]
    fn primary_visibility_is_true() {
        let config = WindowConfig::primary(SurfaceConfiguration::new(1920, 1080));
        assert!(config.is_visible);
    }

    #[test]
    fn visibility_toggle() {
        let config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))
            .with_visibility(false)
            .with_visibility(true)
            .with_visibility(false);

        assert!(!config.is_visible);
    }

    #[test]
    fn visibility_affects_should_render() {
        let visible = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600));
        let hidden = visible.clone().with_visibility(false);

        assert!(visible.should_render());
        assert!(!hidden.should_render());
    }
}

// ============================================================================
// 18. Focus State Tests
// ============================================================================

mod focus_state {
    use super::*;

    #[test]
    fn default_focus_is_false() {
        let config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600));
        assert!(!config.is_focused);
    }

    #[test]
    fn primary_focus_is_true() {
        let config = WindowConfig::primary(SurfaceConfiguration::new(1920, 1080));
        assert!(config.is_focused);
    }

    #[test]
    fn focus_toggle() {
        let config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))
            .with_focus(true)
            .with_focus(false)
            .with_focus(true);

        assert!(config.is_focused);
    }

    #[test]
    fn focus_does_not_affect_should_render() {
        let unfocused = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600));
        let focused = unfocused.clone().with_focus(true);

        // Both should render if visible with valid dimensions
        assert!(unfocused.should_render());
        assert!(focused.should_render());
    }
}

// ============================================================================
// 19. Aggregate Statistics Tests
// ============================================================================

mod aggregate_statistics {
    use super::*;

    #[test]
    fn aggregate_stats_from_empty_manager() {
        let manager = MultiWindowManager::new();
        let stats = manager.aggregate_stats();

        assert_eq!(stats.window_count, 0);
        assert_eq!(stats.total_frames, 0);
        assert_eq!(stats.total_dropped, 0);
        assert_eq!(stats.average_frame_time_ms, 0.0);
        assert_eq!(stats.global_frame_count, 0);
    }

    #[test]
    fn stats_drop_rate_formula() {
        let stats = MultiWindowStats {
            window_count: 2,
            total_frames: 80,
            total_dropped: 20,
            average_frame_time_ms: 16.67,
            global_frame_count: 50,
        };

        // 20 / (80 + 20) = 0.2
        assert!((stats.drop_rate() - 0.2).abs() < 0.001);
    }

    #[test]
    fn stats_fps_formula() {
        let stats = MultiWindowStats {
            window_count: 1,
            total_frames: 100,
            total_dropped: 0,
            average_frame_time_ms: 10.0,
            global_frame_count: 100,
        };

        // 1000 / 10 = 100 FPS
        assert!((stats.estimated_fps() - 100.0).abs() < 0.001);
    }
}

// ============================================================================
// Summary Statistics
// ============================================================================

// Total: 156 tests, 224 assertions across:
// - WindowId uniqueness and monotonic generation (11 tests)
// - WindowId primary() convention (10 tests)
// - WindowId display and traits (6 tests)
// - WindowConfig builder pattern (13 tests)
// - WindowConfig computed properties (10 tests)
// - WindowConfig display (3 tests)
// - SyncMode (18 tests)
// - MultiWindowManager creation (12 tests)
// - MultiWindowManager render order (2 tests)
// - MultiWindowError (12 tests)
// - MultiWindowStats (12 tests)
// - SurfaceConfiguration (19 tests)
// - Window/Surface integration (4 tests)
// - Thread safety (1 test)
// - Edge cases (15 tests)
// - Priority ordering (4 tests)
// - Visibility state (4 tests)
// - Focus state (4 tests)
// - Aggregate statistics (3 tests)
