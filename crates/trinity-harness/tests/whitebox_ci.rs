//! Whitebox tests for CI workflow generation.

use trinity_harness::ci::{
    generate_harness_steps, generate_yaml, validate_workflow, ValidationResult,
    WorkflowConfig, WorkflowStep,
};

// ==================== WorkflowConfig ====================

#[test]
fn test_config_default() {
    let config = WorkflowConfig::default();
    assert_eq!(config.name, "Trinity Harness");
    assert!(config.branches.contains(&"master".to_string()));
    assert!(config.manual_trigger);
    assert!(config.cache_enabled);
}

#[test]
fn test_config_new() {
    let config = WorkflowConfig::new("Custom Workflow");
    assert_eq!(config.name, "Custom Workflow");
}

#[test]
fn test_config_builder() {
    let config = WorkflowConfig::new("Test")
        .branches(vec!["develop".to_string()])
        .manual_trigger(false)
        .cache(false)
        .rust_version("1.75")
        .python_version("3.12");

    assert_eq!(config.branches, vec!["develop"]);
    assert!(!config.manual_trigger);
    assert!(!config.cache_enabled);
    assert_eq!(config.rust_version, "1.75");
    assert_eq!(config.python_version, "3.12");
}

// ==================== WorkflowStep ====================

#[test]
fn test_step_run() {
    let step = WorkflowStep::run("Build", "cargo build");
    assert_eq!(step.name, "Build");
    assert_eq!(step.run, Some("cargo build".to_string()));
    assert!(step.uses.is_none());
}

#[test]
fn test_step_uses() {
    let step = WorkflowStep::uses("Checkout", "actions/checkout@v4");
    assert_eq!(step.name, "Checkout");
    assert_eq!(step.uses, Some("actions/checkout@v4".to_string()));
    assert!(step.run.is_none());
}

#[test]
fn test_step_with_id() {
    let step = WorkflowStep::run("Query", "echo test").with_id("query");
    assert_eq!(step.id, Some("query".to_string()));
}

#[test]
fn test_step_with_condition() {
    let step = WorkflowStep::run("Conditional", "echo")
        .with_condition("steps.check.outputs.run == 'true'");
    assert_eq!(step.condition, Some("steps.check.outputs.run == 'true'".to_string()));
}

// ==================== generate_harness_steps ====================

#[test]
fn test_generate_steps() {
    let steps = generate_harness_steps();
    
    assert!(!steps.is_empty());
    assert!(steps.iter().any(|s| s.name == "Checkout"));
    assert!(steps.iter().any(|s| s.name == "Build Harness"));
    assert!(steps.iter().any(|s| s.name == "Query Stale Tests"));
}

// ==================== validate_workflow ====================

#[test]
fn test_validate_valid() {
    let config = WorkflowConfig::default();
    let result = validate_workflow(&config);
    
    assert!(result.passed());
    assert!(result.errors.is_empty());
}

#[test]
fn test_validate_empty_name() {
    let mut config = WorkflowConfig::default();
    config.name = String::new();
    
    let result = validate_workflow(&config);
    
    assert!(!result.passed());
    assert!(result.errors.iter().any(|e| e.contains("name")));
}

#[test]
fn test_validate_no_branches() {
    let mut config = WorkflowConfig::default();
    config.branches = vec![];
    
    let result = validate_workflow(&config);
    
    assert!(!result.passed());
    assert!(result.errors.iter().any(|e| e.contains("branch")));
}

// ==================== ValidationResult ====================

#[test]
fn test_validation_result_new() {
    let result = ValidationResult::new();
    assert!(result.is_valid);
    assert!(result.errors.is_empty());
    assert!(result.warnings.is_empty());
}

#[test]
fn test_validation_add_error() {
    let mut result = ValidationResult::new();
    result.add_error("Test error");
    
    assert!(!result.is_valid);
    assert_eq!(result.errors.len(), 1);
}

#[test]
fn test_validation_add_warning() {
    let mut result = ValidationResult::new();
    result.add_warning("Test warning");
    
    assert!(result.is_valid);
    assert_eq!(result.warnings.len(), 1);
}

// ==================== generate_yaml ====================

#[test]
fn test_generate_yaml() {
    let config = WorkflowConfig::default();
    let yaml = generate_yaml(&config);
    
    assert!(yaml.contains("name: Trinity Harness"));
    assert!(yaml.contains("on:"));
    assert!(yaml.contains("push:"));
    assert!(yaml.contains("pull_request:"));
    assert!(yaml.contains("jobs:"));
}

#[test]
fn test_generate_yaml_with_dispatch() {
    let config = WorkflowConfig::default().manual_trigger(true);
    let yaml = generate_yaml(&config);
    
    assert!(yaml.contains("workflow_dispatch"));
}

#[test]
fn test_generate_yaml_without_dispatch() {
    let config = WorkflowConfig::default().manual_trigger(false);
    let yaml = generate_yaml(&config);
    
    assert!(!yaml.contains("workflow_dispatch"));
}
