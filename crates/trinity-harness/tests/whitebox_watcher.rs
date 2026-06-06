//! Whitebox tests for file watcher.

use std::path::PathBuf;
use std::time::Instant;
use trinity_harness::daemon::{
    ChangeKind, Debouncer, DaemonEvent, FileChange, FileWatcher, WatcherConfig,
};

// ==================== WatcherConfig ====================

#[test]
fn test_config_default() {
    let config = WatcherConfig::default();
    assert_eq!(config.root, PathBuf::from("."));
    assert!(config.extensions.contains(&"rs".to_string()));
    assert!(config.extensions.contains(&"py".to_string()));
    assert!(config.ignore_dirs.contains(&"target".to_string()));
}

#[test]
fn test_config_new() {
    let config = WatcherConfig::new("/project");
    assert_eq!(config.root, PathBuf::from("/project"));
}

#[test]
fn test_config_builder() {
    let config = WatcherConfig::new(".")
        .extensions(vec!["rs".to_string(), "ts".to_string()])
        .debounce(200)
        .ignore(vec!["build".to_string()]);

    assert_eq!(config.extensions.len(), 2);
    assert_eq!(config.debounce_ms, 200);
    assert!(config.ignore_dirs.contains(&"build".to_string()));
}

// ==================== FileChange ====================

#[test]
fn test_file_change() {
    let change = FileChange {
        path: PathBuf::from("src/lib.rs"),
        kind: ChangeKind::Modified,
        timestamp: Instant::now(),
    };

    assert_eq!(change.path, PathBuf::from("src/lib.rs"));
    assert_eq!(change.kind, ChangeKind::Modified);
}

#[test]
fn test_change_kind() {
    assert_eq!(ChangeKind::Created, ChangeKind::Created);
    assert_ne!(ChangeKind::Created, ChangeKind::Modified);
    assert_ne!(ChangeKind::Modified, ChangeKind::Deleted);
}

// ==================== FileChange to DaemonEvent ====================

#[test]
fn test_change_to_event_created() {
    let change = FileChange {
        path: PathBuf::from("src/new.rs"),
        kind: ChangeKind::Created,
        timestamp: Instant::now(),
    };

    let event: DaemonEvent = change.into();

    match event {
        DaemonEvent::FileCreated { path } => {
            assert!(path.contains("new.rs"));
        }
        _ => panic!("Expected FileCreated"),
    }
}

#[test]
fn test_change_to_event_modified() {
    let change = FileChange {
        path: PathBuf::from("src/lib.rs"),
        kind: ChangeKind::Modified,
        timestamp: Instant::now(),
    };

    let event: DaemonEvent = change.into();

    match event {
        DaemonEvent::FileModified { path } => {
            assert!(path.contains("lib.rs"));
        }
        _ => panic!("Expected FileModified"),
    }
}

#[test]
fn test_change_to_event_deleted() {
    let change = FileChange {
        path: PathBuf::from("src/old.rs"),
        kind: ChangeKind::Deleted,
        timestamp: Instant::now(),
    };

    let event: DaemonEvent = change.into();

    match event {
        DaemonEvent::FileDeleted { path } => {
            assert!(path.contains("old.rs"));
        }
        _ => panic!("Expected FileDeleted"),
    }
}

// ==================== FileWatcher ====================

#[test]
fn test_watcher_new() {
    let config = WatcherConfig::default();
    let (watcher, _receiver) = FileWatcher::new(config);

    assert!(!watcher.is_running());
    assert_eq!(watcher.tracked_files(), 0);
}

#[test]
fn test_watcher_start_stop() {
    let config = WatcherConfig::new(".").debounce(10);
    let (mut watcher, _receiver) = FileWatcher::new(config);

    watcher.start();
    assert!(watcher.is_running());

    watcher.stop();
    assert!(!watcher.is_running());
}

// ==================== Debouncer ====================

#[test]
fn test_debouncer_new() {
    let debouncer = Debouncer::new(100);
    assert_eq!(debouncer.debounce_ms(), 100);
}

#[test]
fn test_debouncer_should_process() {
    let mut debouncer = Debouncer::new(100);

    let change = FileChange {
        path: PathBuf::from("src/lib.rs"),
        kind: ChangeKind::Modified,
        timestamp: Instant::now(),
    };

    // First event should be processed
    assert!(debouncer.should_process(&change));

    // Immediate second event should be debounced
    assert!(!debouncer.should_process(&change));
}

#[test]
fn test_debouncer_different_files() {
    let mut debouncer = Debouncer::new(100);

    let change1 = FileChange {
        path: PathBuf::from("src/a.rs"),
        kind: ChangeKind::Modified,
        timestamp: Instant::now(),
    };

    let change2 = FileChange {
        path: PathBuf::from("src/b.rs"),
        kind: ChangeKind::Modified,
        timestamp: Instant::now(),
    };

    // Different files should both be processed
    assert!(debouncer.should_process(&change1));
    assert!(debouncer.should_process(&change2));
}

#[test]
fn test_debouncer_cleanup() {
    let mut debouncer = Debouncer::new(100);

    let change = FileChange {
        path: PathBuf::from("src/lib.rs"),
        kind: ChangeKind::Modified,
        timestamp: Instant::now(),
    };

    debouncer.should_process(&change);
    debouncer.cleanup();
    // Should not panic
}
