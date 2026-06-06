//! Whitebox tests for documentation generation.

use trinity_harness::docs::{
    ci_docs, cli_docs, daemon_docs, generate_all, validate_docs, DocSection, DocValidation,
    Documentation,
};

// ==================== Documentation functions ====================

#[test]
fn test_daemon_docs() {
    let docs = daemon_docs();
    
    assert!(!docs.is_empty());
    assert!(docs.contains("Daemon"));
    assert!(docs.contains("## "));
}

#[test]
fn test_ci_docs() {
    let docs = ci_docs();
    
    assert!(!docs.is_empty());
    assert!(docs.contains("CI"));
    assert!(docs.contains("GitHub"));
}

#[test]
fn test_cli_docs() {
    let docs = cli_docs();
    
    assert!(!docs.is_empty());
    assert!(docs.contains("CLI"));
    assert!(docs.contains("daemon"));
    assert!(docs.contains("query"));
}

#[test]
fn test_generate_all() {
    let docs = generate_all();
    
    assert!(!docs.daemon.is_empty());
    assert!(!docs.ci.is_empty());
    assert!(!docs.cli.is_empty());
}

// ==================== Documentation ====================

#[test]
fn test_documentation_combined() {
    let docs = generate_all();
    let combined = docs.combined();
    
    assert!(combined.contains("---"));
    assert!(combined.contains("Daemon"));
    assert!(combined.contains("CI"));
    assert!(combined.contains("CLI"));
}

#[test]
fn test_documentation_word_count() {
    let docs = generate_all();
    let count = docs.word_count();
    
    assert!(count > 100);
}

#[test]
fn test_documentation_section_count() {
    let docs = generate_all();
    let count = docs.section_count();
    
    assert!(count >= 3);
}

// ==================== DocSection ====================

#[test]
fn test_section_new() {
    let section = DocSection::new("Title", "Content here");
    
    assert_eq!(section.title, "Title");
    assert_eq!(section.content, "Content here");
    assert!(section.subsections.is_empty());
}

#[test]
fn test_section_with_subsection() {
    let section = DocSection::new("Parent", "Parent content")
        .with_subsection(DocSection::new("Child", "Child content"));
    
    assert_eq!(section.subsections.len(), 1);
    assert_eq!(section.subsections[0].title, "Child");
}

#[test]
fn test_section_to_markdown() {
    let section = DocSection::new("Title", "Content");
    let md = section.to_markdown(1);
    
    assert!(md.contains("# Title"));
    assert!(md.contains("Content"));
}

#[test]
fn test_section_nested_markdown() {
    let section = DocSection::new("Parent", "Parent content")
        .with_subsection(DocSection::new("Child", "Child content"));
    let md = section.to_markdown(1);
    
    assert!(md.contains("# Parent"));
    assert!(md.contains("## Child"));
}

// ==================== validate_docs ====================

#[test]
fn test_validate_docs_valid() {
    let docs = generate_all();
    let result = validate_docs(&docs);
    
    assert!(result.passed());
}

#[test]
fn test_validate_docs_empty_daemon() {
    let docs = Documentation {
        daemon: String::new(),
        ci: ci_docs(),
        cli: cli_docs(),
    };
    let result = validate_docs(&docs);
    
    assert!(!result.passed());
    assert!(result.errors.iter().any(|e| e.contains("Daemon")));
}

#[test]
fn test_validate_docs_empty_ci() {
    let docs = Documentation {
        daemon: daemon_docs(),
        ci: String::new(),
        cli: cli_docs(),
    };
    let result = validate_docs(&docs);
    
    assert!(!result.passed());
}

#[test]
fn test_validate_docs_empty_cli() {
    let docs = Documentation {
        daemon: daemon_docs(),
        ci: ci_docs(),
        cli: String::new(),
    };
    let result = validate_docs(&docs);
    
    assert!(!result.passed());
}

// ==================== DocValidation ====================

#[test]
fn test_validation_new() {
    let result = DocValidation::new();
    
    assert!(result.is_valid);
    assert!(result.errors.is_empty());
    assert!(result.warnings.is_empty());
}

#[test]
fn test_validation_add_error() {
    let mut result = DocValidation::new();
    result.add_error("Test error");
    
    assert!(!result.is_valid);
    assert_eq!(result.errors.len(), 1);
}

#[test]
fn test_validation_add_warning() {
    let mut result = DocValidation::new();
    result.add_warning("Test warning");
    
    assert!(result.is_valid);
    assert_eq!(result.warnings.len(), 1);
}

#[test]
fn test_validation_passed() {
    let mut result = DocValidation::new();
    assert!(result.passed());
    
    result.add_error("error");
    assert!(!result.passed());
}
