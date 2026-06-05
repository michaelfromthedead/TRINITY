//! Multi-language parser registry.

mod python;
mod rust;
mod wgsl;

pub use python::PythonParser;
pub use rust::RustParser;
pub use wgsl::WgslParser;

use std::path::Path;

/// Supported languages for parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Rust,
    Python,
    Wgsl,
}

/// A parsed code unit representing a function, struct, class, etc.
#[derive(Debug, Clone)]
pub struct CodeUnit {
    pub unit_type: UnitType,
    pub name: String,
    pub start_line: usize,
    pub end_line: usize,
    pub language: Language,
}

/// Types of code units that can be extracted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitType {
    Function,
    Struct,
    Class,
    Method,
    Module,
}

/// Registry holding all language parsers.
pub struct ParserRegistry {
    rust: RustParser,
    python: PythonParser,
    wgsl: WgslParser,
}

impl ParserRegistry {
    /// Create a new parser registry with all parsers initialized.
    pub fn new() -> Self {
        Self {
            rust: RustParser::new(),
            python: PythonParser::new(),
            wgsl: WgslParser::new(),
        }
    }

    /// Parse a file and return all code units found.
    pub fn parse_file(&self, _path: &Path, source: &str, lang: Language) -> Vec<CodeUnit> {
        match lang {
            Language::Rust => self.rust.parse(source),
            Language::Python => self.python.parse(source),
            Language::Wgsl => self.wgsl.parse(source),
        }
    }

    /// Detect language from file extension.
    pub fn detect_language(path: &Path) -> Option<Language> {
        path.extension().and_then(|ext| ext.to_str()).and_then(|ext| match ext {
            "rs" => Some(Language::Rust),
            "py" => Some(Language::Python),
            "wgsl" => Some(Language::Wgsl),
            _ => None,
        })
    }
}

impl Default for ParserRegistry {
    fn default() -> Self {
        Self::new()
    }
}
