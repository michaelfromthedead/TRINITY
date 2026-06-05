//! Python source code parser using rustpython-parser and tree-sitter.

use super::{CodeUnit, ContentHashes, Language, UnitType};
use rustpython_parser::ast::{Arguments, Mod, Ranged, Stmt};
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
            let line_index = LineIndex::new(source);
            self.extract_from_module(source, &line_index, &ast, &mut units);
        }

        units
    }

    fn extract_from_module(
        &self,
        source: &str,
        line_index: &LineIndex,
        module: &Mod,
        units: &mut Vec<CodeUnit>,
    ) {
        if let Mod::Module(m) = module {
            for stmt in &m.body {
                self.extract_stmt(source, line_index, stmt, units);
            }
        }
    }

    fn extract_stmt(
        &self,
        source: &str,
        line_index: &LineIndex,
        stmt: &Stmt,
        units: &mut Vec<CodeUnit>,
    ) {
        match stmt {
            Stmt::FunctionDef(f) => {
                let start_offset = u32::from(f.range.start()) as usize;
                let end_offset = u32::from(f.range.end()) as usize;
                let (start_line, end_line) = line_index.offset_to_lines(start_offset, end_offset);
                let hashes = self.compute_function_hashes(source, start_offset, end_offset, f);

                units.push(CodeUnit {
                    unit_type: UnitType::Function,
                    name: f.name.to_string(),
                    start_line,
                    end_line,
                    language: Language::Python,
                    hashes,
                });
            }
            Stmt::AsyncFunctionDef(f) => {
                let start_offset = u32::from(f.range.start()) as usize;
                let end_offset = u32::from(f.range.end()) as usize;
                let (start_line, end_line) = line_index.offset_to_lines(start_offset, end_offset);
                let hashes =
                    self.compute_async_function_hashes(source, start_offset, end_offset, f);

                units.push(CodeUnit {
                    unit_type: UnitType::Function,
                    name: f.name.to_string(),
                    start_line,
                    end_line,
                    language: Language::Python,
                    hashes,
                });
            }
            Stmt::ClassDef(c) => {
                let start_offset = u32::from(c.range.start()) as usize;
                let end_offset = u32::from(c.range.end()) as usize;
                let (start_line, end_line) = line_index.offset_to_lines(start_offset, end_offset);
                let hashes = self.compute_class_hashes(source, start_offset, end_offset, c);

                units.push(CodeUnit {
                    unit_type: UnitType::Class,
                    name: c.name.to_string(),
                    start_line,
                    end_line,
                    language: Language::Python,
                    hashes,
                });
            }
            _ => {}
        }
    }

    /// Compute hashes for a function definition.
    fn compute_function_hashes(
        &self,
        source: &str,
        start_offset: usize,
        end_offset: usize,
        f: &rustpython_parser::ast::StmtFunctionDef,
    ) -> ContentHashes {
        let full_text = self.extract_text(source, start_offset, end_offset);

        // Signature: "def name(params) -> return_type"
        let sig_text = self.build_function_signature(&f.name, &f.args, &f.returns);

        // Body: everything after the signature line(s)
        let body_text = self.extract_function_body(source, f);

        ContentHashes {
            full_hash: blake3::hash(full_text.as_bytes()).into(),
            signature_hash: blake3::hash(sig_text.as_bytes()).into(),
            body_hash: blake3::hash(body_text.as_bytes()).into(),
            layout_hash: [0u8; 32], // Not applicable for functions
        }
    }

    /// Compute hashes for an async function definition.
    fn compute_async_function_hashes(
        &self,
        source: &str,
        start_offset: usize,
        end_offset: usize,
        f: &rustpython_parser::ast::StmtAsyncFunctionDef,
    ) -> ContentHashes {
        let full_text = self.extract_text(source, start_offset, end_offset);

        // Signature: "async def name(params) -> return_type"
        let sig_text = format!(
            "async {}",
            self.build_function_signature(&f.name, &f.args, &f.returns)
        );

        // Body: everything after the signature line(s)
        let body_text = self.extract_async_function_body(source, f);

        ContentHashes {
            full_hash: blake3::hash(full_text.as_bytes()).into(),
            signature_hash: blake3::hash(sig_text.as_bytes()).into(),
            body_hash: blake3::hash(body_text.as_bytes()).into(),
            layout_hash: [0u8; 32], // Not applicable for functions
        }
    }

    /// Compute hashes for a class definition.
    fn compute_class_hashes(
        &self,
        source: &str,
        start_offset: usize,
        end_offset: usize,
        c: &rustpython_parser::ast::StmtClassDef,
    ) -> ContentHashes {
        let full_text = self.extract_text(source, start_offset, end_offset);

        // Signature: "class Name(bases)"
        let sig_text = self.build_class_signature(&c.name, &c.bases);

        // Body: the class body (methods, attributes)
        let body_text = self.extract_class_body(source, c);

        // Layout: method names for structural comparison
        let layout_text = self.build_class_layout(c);

        ContentHashes {
            full_hash: blake3::hash(full_text.as_bytes()).into(),
            signature_hash: blake3::hash(sig_text.as_bytes()).into(),
            body_hash: blake3::hash(body_text.as_bytes()).into(),
            layout_hash: blake3::hash(layout_text.as_bytes()).into(),
        }
    }

    /// Extract text from source between byte offsets.
    fn extract_text(&self, source: &str, start: usize, end: usize) -> String {
        if start >= source.len() || end > source.len() || start >= end {
            return String::new();
        }
        source[start..end].to_string()
    }

    /// Build a function signature string from AST components.
    fn build_function_signature(
        &self,
        name: &rustpython_parser::ast::Identifier,
        args: &Box<Arguments>,
        returns: &Option<Box<rustpython_parser::ast::Expr>>,
    ) -> String {
        let mut sig = format!("def {}(", name);

        // Collect parameter names
        let mut param_parts = Vec::new();

        for arg in &args.posonlyargs {
            param_parts.push(arg.def.arg.to_string());
        }
        for arg in &args.args {
            param_parts.push(arg.def.arg.to_string());
        }
        if let Some(vararg) = &args.vararg {
            param_parts.push(format!("*{}", vararg.arg));
        }
        for arg in &args.kwonlyargs {
            param_parts.push(arg.def.arg.to_string());
        }
        if let Some(kwarg) = &args.kwarg {
            param_parts.push(format!("**{}", kwarg.arg));
        }

        sig.push_str(&param_parts.join(", "));
        sig.push(')');

        // Add return type if present
        if returns.is_some() {
            sig.push_str(" -> ...");
        }

        sig
    }

    /// Build a class signature string.
    fn build_class_signature(
        &self,
        name: &rustpython_parser::ast::Identifier,
        bases: &[rustpython_parser::ast::Expr],
    ) -> String {
        let mut sig = format!("class {}", name);

        if !bases.is_empty() {
            sig.push('(');
            let base_names: Vec<String> = bases.iter().map(|_| "...".to_string()).collect();
            sig.push_str(&base_names.join(", "));
            sig.push(')');
        }

        sig
    }

    /// Extract function body text (statements after the signature).
    fn extract_function_body(
        &self,
        source: &str,
        f: &rustpython_parser::ast::StmtFunctionDef,
    ) -> String {
        if f.body.is_empty() {
            return String::new();
        }

        let first_stmt = &f.body[0];
        let last_stmt = &f.body[f.body.len() - 1];

        let start = u32::from(first_stmt.range().start()) as usize;
        let end = u32::from(last_stmt.range().end()) as usize;

        self.extract_text(source, start, end)
    }

    /// Extract async function body text.
    fn extract_async_function_body(
        &self,
        source: &str,
        f: &rustpython_parser::ast::StmtAsyncFunctionDef,
    ) -> String {
        if f.body.is_empty() {
            return String::new();
        }

        let first_stmt = &f.body[0];
        let last_stmt = &f.body[f.body.len() - 1];

        let start = u32::from(first_stmt.range().start()) as usize;
        let end = u32::from(last_stmt.range().end()) as usize;

        self.extract_text(source, start, end)
    }

    /// Extract class body text.
    fn extract_class_body(&self, source: &str, c: &rustpython_parser::ast::StmtClassDef) -> String {
        if c.body.is_empty() {
            return String::new();
        }

        let first_stmt = &c.body[0];
        let last_stmt = &c.body[c.body.len() - 1];

        let start = u32::from(first_stmt.range().start()) as usize;
        let end = u32::from(last_stmt.range().end()) as usize;

        self.extract_text(source, start, end)
    }

    /// Build class layout string from method/attribute names.
    fn build_class_layout(&self, c: &rustpython_parser::ast::StmtClassDef) -> String {
        let mut parts = Vec::new();

        for stmt in &c.body {
            match stmt {
                Stmt::FunctionDef(f) => {
                    parts.push(format!("def:{}", f.name));
                }
                Stmt::AsyncFunctionDef(f) => {
                    parts.push(format!("async def:{}", f.name));
                }
                Stmt::Assign(_) => {
                    parts.push("attr".to_string());
                }
                Stmt::AnnAssign(a) => {
                    if let rustpython_parser::ast::Expr::Name(n) = a.target.as_ref() {
                        parts.push(format!("attr:{}", n.id));
                    }
                }
                _ => {}
            }
        }

        parts.join(",")
    }
}

impl Default for PythonParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Index for converting byte offsets to line numbers.
struct LineIndex {
    line_starts: Vec<usize>,
}

impl LineIndex {
    /// Create a new line index from source text.
    fn new(source: &str) -> Self {
        let mut line_starts = vec![0];
        for (i, c) in source.char_indices() {
            if c == '\n' {
                line_starts.push(i + 1);
            }
        }
        Self { line_starts }
    }

    /// Convert byte offsets to 1-indexed line numbers.
    fn offset_to_lines(&self, start_offset: usize, end_offset: usize) -> (usize, usize) {
        let start_line = self.offset_to_line(start_offset);
        let end_line = self.offset_to_line(end_offset);
        (start_line, end_line)
    }

    /// Convert a single byte offset to a 1-indexed line number.
    fn offset_to_line(&self, offset: usize) -> usize {
        match self.line_starts.binary_search(&offset) {
            Ok(line) => line + 1,
            Err(line) => line,
        }
    }
}
