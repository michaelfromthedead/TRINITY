//! Python source code parser using rustpython-parser and tree-sitter.

use super::{CodeUnit, ContentHashes, Language, UnitType};
use rustpython_parser::{parse, Mode};

/// Parser for Python source code.
pub struct PythonParser {
    _private: (),
}

impl PythonParser {
    /// Create a new Python parser.
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Parse Python source and extract code units.
    pub fn parse(&self, source: &str) -> Vec<CodeUnit> {
        let mut units = Vec::new();

        if let Ok(ast) = parse(source, Mode::Module, "<input>") {
            self.extract_from_module(&ast, &mut units);
        }

        units
    }

    fn extract_from_module(
        &self,
        module: &rustpython_parser::ast::Mod,
        units: &mut Vec<CodeUnit>,
    ) {
        use rustpython_parser::ast::{Mod, Stmt};

        if let Mod::Module(m) = module {
            for stmt in &m.body {
                match stmt {
                    Stmt::FunctionDef(f) => {
                        units.push(CodeUnit {
                            unit_type: UnitType::Function,
                            name: f.name.to_string(),
                            start_line: u32::from(f.range.start()) as usize,
                            end_line: u32::from(f.range.end()) as usize,
                            language: Language::Python,
                            hashes: ContentHashes::default(),
                        });
                    }
                    Stmt::AsyncFunctionDef(f) => {
                        units.push(CodeUnit {
                            unit_type: UnitType::Function,
                            name: f.name.to_string(),
                            start_line: u32::from(f.range.start()) as usize,
                            end_line: u32::from(f.range.end()) as usize,
                            language: Language::Python,
                            hashes: ContentHashes::default(),
                        });
                    }
                    Stmt::ClassDef(c) => {
                        units.push(CodeUnit {
                            unit_type: UnitType::Class,
                            name: c.name.to_string(),
                            start_line: u32::from(c.range.start()) as usize,
                            end_line: u32::from(c.range.end()) as usize,
                            language: Language::Python,
                            hashes: ContentHashes::default(),
                        });
                    }
                    _ => {}
                }
            }
        }
    }
}

impl Default for PythonParser {
    fn default() -> Self {
        Self::new()
    }
}
