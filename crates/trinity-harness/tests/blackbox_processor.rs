//! Blackbox tests for event processor with graph integration.

use std::path::Path;
use tempfile::TempDir;
use trinity_harness::daemon::{DaemonEvent, EventProcessor, ProcessorConfig};
use trinity_harness::graph::GraphBuilder;
use trinity_harness::parsers::ParserRegistry;
use trinity_harness::runners::StateTracker;

fn create_test_dir() -> TempDir {
    TempDir::new().expect("Failed to create temp dir")
}

fn write_file(dir: &Path, name: &str, content: &str) {
    let path = dir.join(name);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(path, content).expect("Failed to write file");
}

#[test]
fn test_processor_with_graph() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/lib.rs", r#"
fn compute() -> i32 { 42 }
fn helper() -> i32 { compute() }
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (graph, _) = builder.full_scan(root).expect("Scan failed");

    let config = ProcessorConfig::default();
    let mut processor = EventProcessor::new(config);
    processor.build_from_graph(&graph);

    assert!(processor.registered_files() >= 1);
}

#[test]
fn test_processor_file_change_marks_dirty() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/lib.rs", r#"
fn compute() -> i32 { 42 }
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (graph, _) = builder.full_scan(root).expect("Scan failed");

    let config = ProcessorConfig::default();
    let mut processor = EventProcessor::new(config);
    let mut tracker = StateTracker::new();
    processor.build_from_graph(&graph);

    // Process a file change
    let event = DaemonEvent::FileModified {
        path: root.join("src/lib.rs").to_string_lossy().to_string(),
    };

    let result = processor.process_event(&event, &mut tracker);

    // Should have processed the event
    assert!(result.is_some());
}

#[test]
fn test_processor_dependency_propagation() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/lib.rs", r#"
fn core_fn() -> i32 { 42 }
"#);

    write_file(root, "src/main.rs", r#"
mod lib;
fn main() { let x = lib::core_fn(); }
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (graph, _) = builder.full_scan(root).expect("Scan failed");

    let mut config = ProcessorConfig::default();
    config.propagate_staleness = true;

    let mut processor = EventProcessor::new(config);
    processor.build_from_graph(&graph);

    // Should have registered dependencies
    assert!(processor.registered_files() >= 2);
}

#[test]
fn test_processor_batch_events() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/a.rs", "fn a() {}");
    write_file(root, "src/b.rs", "fn b() {}");
    write_file(root, "src/c.rs", "fn c() {}");

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (graph, _) = builder.full_scan(root).expect("Scan failed");

    let config = ProcessorConfig::default();
    let mut processor = EventProcessor::new(config);
    let mut tracker = StateTracker::new();
    processor.build_from_graph(&graph);

    // Process multiple events
    let events = vec![
        DaemonEvent::FileModified { path: root.join("src/a.rs").to_string_lossy().to_string() },
        DaemonEvent::FileModified { path: root.join("src/b.rs").to_string_lossy().to_string() },
    ];

    let result = trinity_harness::daemon::process_batch(&mut processor, &events, &mut tracker);

    assert_eq!(result.events_processed, 2);
}

#[test]
fn test_processor_ignored_events() {
    let config = ProcessorConfig::default();
    let mut processor = EventProcessor::new(config);
    let mut tracker = StateTracker::new();

    // These events don't affect file state
    let events = vec![
        DaemonEvent::Started,
        DaemonEvent::Stopped,
        DaemonEvent::Error { message: "test".to_string() },
    ];

    for event in &events {
        let result = processor.process_event(event, &mut tracker);
        assert!(result.is_none());
    }
}
