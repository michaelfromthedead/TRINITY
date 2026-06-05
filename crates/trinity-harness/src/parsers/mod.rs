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

/// Content hashes for change detection and dependency tracking.
#[derive(Debug, Clone, Default)]
pub struct ContentHashes {
    /// Hash of the entire source text of the unit.
    pub full_hash: [u8; 32],
    /// Hash of just the signature (for functions: name + params + return type).
    pub signature_hash: [u8; 32],
    /// Hash of the body only (for functions/methods).
    pub body_hash: [u8; 32],
    /// Hash of field layout (for structs: field names, types, order).
    pub layout_hash: [u8; 32],
}

/// A parsed code unit representing a function, struct, class, etc.
#[derive(Debug, Clone)]
pub struct CodeUnit {
    pub unit_type: UnitType,
    pub name: String,
    pub start_line: usize,
    pub end_line: usize,
    pub language: Language,
    /// Content hashes for change detection.
    pub hashes: ContentHashes,
}

/// Types of code units that can be extracted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitType {
    Function,
    Struct,
    Enum,
    Class,
    Method,
    Module,
    Impl,
    Trait,
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
