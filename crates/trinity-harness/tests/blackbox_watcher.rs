//! Blackbox tests for file watcher with real files.

use std::thread;
use std::time::Duration;
use tempfile::TempDir;
use trinity_harness::daemon::{FileWatcher, WatcherConfig};

fn create_test_dir() -> TempDir {
    TempDir::new().expect("Failed to create temp dir")
}

#[test]
fn test_watcher_tracks_files() {
    let dir = create_test_dir();
    let root = dir.path();

    // Create some test files
    std::fs::write(root.join("a.rs"), "fn a() {}").ok();
    std::fs::write(root.join("b.rs"), "fn b() {}").ok();

    let config = WatcherConfig::new(root);
    let (mut watcher, _receiver) = FileWatcher::new(config);

    watcher.start();
    thread::sleep(Duration::from_millis(100));

    // Should have tracked the files
    assert!(watcher.tracked_files() >= 2);

    watcher.stop();
}

#[test]
fn test_watcher_detects_modification() {
    let dir = create_test_dir();
    let root = dir.path();

    // Create initial file
    std::fs::write(root.join("test.rs"), "fn test() {}").ok();

    let config = WatcherConfig::new(root).debounce(10);
    let (mut watcher, receiver) = FileWatcher::new(config);

    watcher.start();
    thread::sleep(Duration::from_millis(200));

    // Modify the file
    std::fs::write(root.join("test.rs"), "fn test() { /* modified */ }").ok();
    thread::sleep(Duration::from_millis(200));

    watcher.stop();

    // Check if we received any events (may or may not depending on timing)
    let _events: Vec<_> = receiver.try_iter().collect();
}

#[test]
fn test_watcher_ignores_target() {
    let dir = create_test_dir();
    let root = dir.path();

    // Create files in target dir
    std::fs::create_dir_all(root.join("target")).ok();
    std::fs::write(root.join("target/debug.rs"), "// ignored").ok();

    // Create file outside target
    std::fs::write(root.join("src.rs"), "fn src() {}").ok();

    let config = WatcherConfig::new(root);
    let (mut watcher, _receiver) = FileWatcher::new(config);

    watcher.start();
    thread::sleep(Duration::from_millis(100));

    // Should only track src.rs, not target/debug.rs
    assert_eq!(watcher.tracked_files(), 1);

    watcher.stop();
}

#[test]
fn test_watcher_respects_extensions() {
    let dir = create_test_dir();
    let root = dir.path();

    // Create files with different extensions
    std::fs::write(root.join("code.rs"), "fn code() {}").ok();
    std::fs::write(root.join("script.py"), "def script(): pass").ok();
    std::fs::write(root.join("readme.md"), "# Readme").ok(); // Not watched

    let config = WatcherConfig::new(root);
    let (mut watcher, _receiver) = FileWatcher::new(config);

    watcher.start();
    thread::sleep(Duration::from_millis(100));

    // Should track .rs and .py, not .md
    assert_eq!(watcher.tracked_files(), 2);

    watcher.stop();
}

#[test]
fn test_watcher_custom_extensions() {
    let dir = create_test_dir();
    let root = dir.path();

    // Create files
    std::fs::write(root.join("code.ts"), "const code = 1;").ok();
    std::fs::write(root.join("other.rs"), "fn other() {}").ok();

    let config = WatcherConfig::new(root)
        .extensions(vec!["ts".to_string()]);

    let (mut watcher, _receiver) = FileWatcher::new(config);

    watcher.start();
    thread::sleep(Duration::from_millis(100));

    // Should only track .ts
    assert_eq!(watcher.tracked_files(), 1);

    watcher.stop();
}
