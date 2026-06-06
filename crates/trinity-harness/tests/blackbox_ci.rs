//! Blackbox tests for CI workflow generation with file output.

use std::path::Path;
use tempfile::TempDir;
use trinity_harness::ci::{generate_yaml, validate_workflow, WorkflowConfig};

fn create_test_dir() -> TempDir {
    TempDir::new().expect("Failed to create temp dir")
}

#[test]
fn test_generate_and_write_yaml() {
    let dir = create_test_dir();
    let workflow_path = dir.path().join(".github/workflows/harness.yml");

    std::fs::create_dir_all(workflow_path.parent().unwrap()).ok();

    let config = WorkflowConfig::default();
    let yaml = generate_yaml(&config);

    std::fs::write(&workflow_path, &yaml).expect("Failed to write");

    let content = std::fs::read_to_string(&workflow_path).expect("Failed to read");

    assert!(content.contains("name: Trinity Harness"));
    assert!(content.contains("jobs:"));
}

#[test]
fn test_validate_and_generate() {
    let config = WorkflowConfig::new("CI")
        .branches(vec!["main".to_string()])
        .rust_version("stable");

    let validation = validate_workflow(&config);
    assert!(validation.passed());

    let yaml = generate_yaml(&config);
    assert!(yaml.contains("name: CI"));
    assert!(yaml.contains("main"));
}

#[test]
fn test_full_workflow_config() {
    let config = WorkflowConfig::new("Full CI")
        .branches(vec![
            "master".to_string(),
            "develop".to_string(),
            "release/*".to_string(),
        ])
        .manual_trigger(true)
        .cache(true)
        .rust_version("1.75.0")
        .python_version("3.13");

    let validation = validate_workflow(&config);
    assert!(validation.passed());

    let yaml = generate_yaml(&config);

    assert!(yaml.contains("name: Full CI"));
    assert!(yaml.contains("master"));
    assert!(yaml.contains("develop"));
    assert!(yaml.contains("workflow_dispatch"));
}

#[test]
fn test_yaml_is_valid_structure() {
    let config = WorkflowConfig::default();
    let yaml = generate_yaml(&config);

    // Check YAML structure basics
    assert!(yaml.lines().count() > 10);
    assert!(yaml.contains("runs-on:"));
    assert!(yaml.contains("steps:"));
}

#[test]
fn test_workflow_file_path() {
    let expected = Path::new(".github/workflows/harness.yml");
    assert_eq!(expected.file_name().unwrap(), "harness.yml");
    assert!(expected.to_str().unwrap().contains(".github"));
}
