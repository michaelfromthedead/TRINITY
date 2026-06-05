//! Rust source code parser using syn and tree-sitter.

use super::{CodeUnit, Language, UnitType};

/// Parser for Rust source code.
pub struct RustParser {
    _private: (),
}

impl RustParser {
    /// Create a new Rust parser.
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Parse Rust source and extract code units.
    pub fn parse(&self, source: &str) -> Vec<CodeUnit> {
        let mut units = Vec::new();

        if let Ok(file) = syn::parse_file(source) {
            for item in file.items {
                if let Some(unit) = self.extract_item(&item) {
                    units.push(unit);
                }
            }
        }

        units
    }

    fn extract_item(&self, item: &syn::Item) -> Option<CodeUnit> {
        match item {
            syn::Item::Fn(f) => Some(CodeUnit {
                unit_type: UnitType::Function,
                name: f.sig.ident.to_string(),
                start_line: 0,
                end_line: 0,
                language: Language::Rust,
            }),
            syn::Item::Struct(s) => Some(CodeUnit {
                unit_type: UnitType::Struct,
                name: s.ident.to_string(),
                start_line: 0,
                end_line: 0,
                language: Language::Rust,
            }),
            syn::Item::Impl(i) => {
                if let Some((_, path, _)) = &i.trait_ {
                    let trait_name = path.segments.last()?.ident.to_string();
                    Some(CodeUnit {
                        unit_type: UnitType::Method,
                        name: trait_name,
                        start_line: 0,
                        end_line: 0,
                        language: Language::Rust,
                    })
                } else {
                    None
                }
            }
            syn::Item::Mod(m) => Some(CodeUnit {
                unit_type: UnitType::Module,
                name: m.ident.to_string(),
                start_line: 0,
                end_line: 0,
                language: Language::Rust,
            }),
            _ => None,
        }
    }
}

impl Default for RustParser {
    fn default() -> Self {
        Self::new()
    }
}
