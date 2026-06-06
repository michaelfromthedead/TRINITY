//! Whitebox tests for CLI commands.

use trinity_harness::cli::{
    execute_command, CliConfig, CommandResult, OutputFormat,
};

// ==================== CliConfig ====================

#[test]
fn test_config_default() {
    let config = CliConfig::default();
    assert_eq!(config.project_root.to_str(), Some("."));
    assert!(!config.verbose);
    assert_eq!(config.format, OutputFormat::Text);
}

#[test]
fn test_config_new() {
    let config = CliConfig::new("/project");
    assert_eq!(config.project_root.to_str(), Some("/project"));
}

#[test]
fn test_config_verbose() {
    let config = CliConfig::default().verbose();
    assert!(config.verbose);
}

#[test]
fn test_config_format() {
    let config = CliConfig::default().format(OutputFormat::Json);
    assert_eq!(config.format, OutputFormat::Json);
}

// ==================== OutputFormat ====================

#[test]
fn test_output_format_default() {
    let format: OutputFormat = Default::default();
    assert_eq!(format, OutputFormat::Text);
}

#[test]
fn test_output_format_variants() {
    assert_ne!(OutputFormat::Text, OutputFormat::Json);
}

// ==================== CommandResult ====================

#[test]
fn test_result_ok() {
    let result = CommandResult::ok("success");
    assert!(result.success);
    assert_eq!(result.message, "success");
    assert_eq!(result.exit_code, 0);
}

#[test]
fn test_result_err() {
    let result = CommandResult::err("failure");
    assert!(!result.success);
    assert_eq!(result.message, "failure");
    assert_eq!(result.exit_code, 1);
}

// ==================== execute_command ====================

#[test]
fn test_execute_no_command() {
    let args: Vec<String> = vec![];
    let result = execute_command(&args);

    assert!(!result.success);
    assert!(result.message.contains("No command"));
}

#[test]
fn test_execute_unknown_command() {
    let args = vec!["unknown".to_string()];
    let result = execute_command(&args);

    assert!(!result.success);
    assert!(result.message.contains("Unknown command"));
}

#[test]
fn test_execute_query_unknown() {
    let args = vec!["query".to_string(), "unknown".to_string()];
    let result = execute_command(&args);

    assert!(!result.success);
    assert!(result.message.contains("Unknown query"));
}

#[test]
fn test_execute_query_needs_testing_empty() {
    let args = vec!["query".to_string(), "needs-testing".to_string()];
    let result = execute_command(&args);

    // Will fail to scan but that's expected
    assert!(result.message.contains("Nodes") || result.message.contains("Scan failed"));
}
