//! Blackbox tests for documentation generation with file output.

use std::path::Path;
use tempfile::TempDir;
use trinity_harness::docs::{generate_all, validate_docs, DocSection};

fn create_test_dir() -> TempDir {
    TempDir::new().expect("Failed to create temp dir")
}

#[test]
fn test_generate_and_write_docs() {
    let dir = create_test_dir();
    let docs = generate_all();

    // Write each doc file
    let daemon_path = dir.path().join("DAEMON.md");
    let ci_path = dir.path().join("CI.md");
    let cli_path = dir.path().join("CLI.md");

    std::fs::write(&daemon_path, &docs.daemon).expect("Failed to write daemon docs");
    std::fs::write(&ci_path, &docs.ci).expect("Failed to write CI docs");
    std::fs::write(&cli_path, &docs.cli).expect("Failed to write CLI docs");

    assert!(daemon_path.exists());
    assert!(ci_path.exists());
    assert!(cli_path.exists());
}

#[test]
fn test_generate_combined_docs() {
    let dir = create_test_dir();
    let docs = generate_all();
    let combined = docs.combined();

    let path = dir.path().join("TRINITY_HARNESS.md");
    std::fs::write(&path, &combined).expect("Failed to write");

    let content = std::fs::read_to_string(&path).expect("Failed to read");

    assert!(content.contains("Daemon"));
    assert!(content.contains("CI"));
    assert!(content.contains("CLI"));
}

#[test]
fn test_docs_validation_workflow() {
    let docs = generate_all();

    // Validate
    let result = validate_docs(&docs);
    assert!(result.passed());

    // Check metrics
    assert!(docs.word_count() > 500);
    assert!(docs.section_count() >= 6);
}

#[test]
fn test_section_hierarchy() {
    let root = DocSection::new("Trinity Harness", "Main documentation")
        .with_subsection(
            DocSection::new("Getting Started", "Quick start guide")
                .with_subsection(DocSection::new("Installation", "Install steps"))
                .with_subsection(DocSection::new("Configuration", "Config options")),
        )
        .with_subsection(
            DocSection::new("Commands", "CLI commands")
                .with_subsection(DocSection::new("daemon", "Start daemon"))
                .with_subsection(DocSection::new("query", "Query state")),
        );

    let md = root.to_markdown(1);

    assert!(md.contains("# Trinity Harness"));
    assert!(md.contains("## Getting Started"));
    assert!(md.contains("### Installation"));
    assert!(md.contains("## Commands"));
}

#[test]
fn test_docs_contain_code_examples() {
    let docs = generate_all();

    // Daemon docs should have code examples
    assert!(docs.daemon.contains("```"));

    // CI docs should have YAML examples
    assert!(docs.ci.contains("```yaml") || docs.ci.contains("```"));

    // CLI docs should have command examples
    assert!(docs.cli.contains("```bash") || docs.cli.contains("```"));
}
