//! Rust source code parser using syn and tree-sitter.

use super::{CodeUnit, ContentHashes, Language, UnitType};
use quote::ToTokens;
use syn::spanned::Spanned;

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
                self.extract_item(source, &item, &mut units);
            }
        }

        units
    }

    fn extract_item(&self, source: &str, item: &syn::Item, units: &mut Vec<CodeUnit>) {
        match item {
            syn::Item::Fn(f) => {
                let (start_line, end_line) = self.span_to_lines(f.span());
                let hashes = self.compute_function_hashes(source, f);
                units.push(CodeUnit {
                    unit_type: UnitType::Function,
                    name: f.sig.ident.to_string(),
                    start_line,
                    end_line,
                    language: Language::Rust,
                    hashes,
                });
            }
            syn::Item::Struct(s) => {
                let (start_line, end_line) = self.span_to_lines(s.span());
                let hashes = self.compute_struct_hashes(source, s);
                units.push(CodeUnit {
                    unit_type: UnitType::Struct,
                    name: s.ident.to_string(),
                    start_line,
                    end_line,
                    language: Language::Rust,
                    hashes,
                });
            }
            syn::Item::Enum(e) => {
                let (start_line, end_line) = self.span_to_lines(e.span());
                let hashes = self.compute_enum_hashes(source, e);
                units.push(CodeUnit {
                    unit_type: UnitType::Enum,
                    name: e.ident.to_string(),
                    start_line,
                    end_line,
                    language: Language::Rust,
                    hashes,
                });
            }
            syn::Item::Impl(i) => {
                let (start_line, end_line) = self.span_to_lines(i.span());
                let hashes = self.compute_impl_hashes(source, i);

                if let Some((_, path, _)) = &i.trait_ {
                    // Trait implementation
                    if let Some(seg) = path.segments.last() {
                        let trait_name = seg.ident.to_string();
                        let self_ty = self.type_to_string(&i.self_ty);
                        units.push(CodeUnit {
                            unit_type: UnitType::Impl,
                            name: format!("{} for {}", trait_name, self_ty),
                            start_line,
                            end_line,
                            language: Language::Rust,
                            hashes,
                        });
                    }
                } else {
                    // Inherent implementation
                    let self_ty = self.type_to_string(&i.self_ty);
                    units.push(CodeUnit {
                        unit_type: UnitType::Impl,
                        name: format!("impl {}", self_ty),
                        start_line,
                        end_line,
                        language: Language::Rust,
                        hashes,
                    });
                }
            }
            syn::Item::Mod(m) => {
                let (start_line, end_line) = self.span_to_lines(m.span());
                let hashes = self.compute_module_hashes(source, m);
                units.push(CodeUnit {
                    unit_type: UnitType::Module,
                    name: m.ident.to_string(),
                    start_line,
                    end_line,
                    language: Language::Rust,
                    hashes,
                });
            }
            syn::Item::Trait(t) => {
                let (start_line, end_line) = self.span_to_lines(t.span());
                let hashes = self.compute_trait_hashes(source, t);
                units.push(CodeUnit {
                    unit_type: UnitType::Trait,
                    name: t.ident.to_string(),
                    start_line,
                    end_line,
                    language: Language::Rust,
                    hashes,
                });
            }
            _ => {}
        }
    }

    /// Convert a proc_macro2::Span to (start_line, end_line).
    /// Lines are 1-indexed as per syn convention.
    fn span_to_lines(&self, span: proc_macro2::Span) -> (usize, usize) {
        let start = span.start();
        let end = span.end();
        (start.line, end.line)
    }

    /// Extract type name as string.
    fn type_to_string(&self, ty: &syn::Type) -> String {
        match ty {
            syn::Type::Path(p) => {
                p.path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect::<Vec<_>>()
                    .join("::")
            }
            _ => {
                eprintln!("[rust_parser] Unexpected type format (non-path type) — returning placeholder");
                "?".to_string()
            }
        }
    }

    /// Compute hashes for a function.
    fn compute_function_hashes(&self, source: &str, f: &syn::ItemFn) -> ContentHashes {
        let full_text = self.extract_span_text(source, f.span());
        let sig_text = self.extract_span_text(source, f.sig.span());
        let body_text = self.extract_span_text(source, f.block.span());

        ContentHashes {
            full_hash: blake3::hash(full_text.as_bytes()).into(),
            signature_hash: blake3::hash(sig_text.as_bytes()).into(),
            body_hash: blake3::hash(body_text.as_bytes()).into(),
            layout_hash: [0u8; 32], // Not applicable for functions
        }
    }

    /// Compute hashes for a struct.
    fn compute_struct_hashes(&self, source: &str, s: &syn::ItemStruct) -> ContentHashes {
        let full_text = self.extract_span_text(source, s.span());

        // Compute layout hash from field names and types
        let mut layout_parts = Vec::new();
        match &s.fields {
            syn::Fields::Named(fields) => {
                for field in &fields.named {
                    if let Some(ident) = &field.ident {
                        let ty_str = field.ty.to_token_stream().to_string();
                        layout_parts.push(format!("{}:{}", ident, ty_str));
                    }
                }
            }
            syn::Fields::Unnamed(fields) => {
                for (i, field) in fields.unnamed.iter().enumerate() {
                    let ty_str = field.ty.to_token_stream().to_string();
                    layout_parts.push(format!("{}:{}", i, ty_str));
                }
            }
            syn::Fields::Unit => {}
        }
        let layout_text = layout_parts.join(",");

        ContentHashes {
            full_hash: blake3::hash(full_text.as_bytes()).into(),
            signature_hash: blake3::hash(s.ident.to_string().as_bytes()).into(),
            body_hash: [0u8; 32], // Not applicable for structs
            layout_hash: blake3::hash(layout_text.as_bytes()).into(),
        }
    }

    /// Compute hashes for an enum.
    fn compute_enum_hashes(&self, source: &str, e: &syn::ItemEnum) -> ContentHashes {
        let full_text = self.extract_span_text(source, e.span());

        // Compute layout hash from variant names
        let layout_parts: Vec<String> = e
            .variants
            .iter()
            .map(|v| v.ident.to_string())
            .collect();
        let layout_text = layout_parts.join(",");

        ContentHashes {
            full_hash: blake3::hash(full_text.as_bytes()).into(),
            signature_hash: blake3::hash(e.ident.to_string().as_bytes()).into(),
            body_hash: [0u8; 32], // Not applicable for enums
            layout_hash: blake3::hash(layout_text.as_bytes()).into(),
        }
    }

    /// Compute hashes for an impl block.
    fn compute_impl_hashes(&self, source: &str, i: &syn::ItemImpl) -> ContentHashes {
        let full_text = self.extract_span_text(source, i.span());

        // Signature is the impl header (trait for type or impl type)
        let sig_text = if let Some((_, path, _)) = &i.trait_ {
            let path_str = path.to_token_stream().to_string();
            let ty_str = i.self_ty.to_token_stream().to_string();
            format!("{} for {}", path_str, ty_str)
        } else {
            i.self_ty.to_token_stream().to_string()
        };

        // Body hash from method names
        let body_parts: Vec<String> = i
            .items
            .iter()
            .filter_map(|item| {
                if let syn::ImplItem::Fn(m) = item {
                    Some(m.sig.ident.to_string())
                } else {
                    None
                }
            })
            .collect();
        let body_text = body_parts.join(",");

        ContentHashes {
            full_hash: blake3::hash(full_text.as_bytes()).into(),
            signature_hash: blake3::hash(sig_text.as_bytes()).into(),
            body_hash: blake3::hash(body_text.as_bytes()).into(),
            layout_hash: [0u8; 32], // Not applicable for impl blocks
        }
    }

    /// Compute hashes for a module.
    fn compute_module_hashes(&self, source: &str, m: &syn::ItemMod) -> ContentHashes {
        let full_text = self.extract_span_text(source, m.span());

        ContentHashes {
            full_hash: blake3::hash(full_text.as_bytes()).into(),
            signature_hash: blake3::hash(m.ident.to_string().as_bytes()).into(),
            body_hash: [0u8; 32], // Could hash content if inline
            layout_hash: [0u8; 32], // Not applicable for modules
        }
    }

    /// Compute hashes for a trait.
    fn compute_trait_hashes(&self, source: &str, t: &syn::ItemTrait) -> ContentHashes {
        let full_text = self.extract_span_text(source, t.span());

        // Method signatures form the body
        let body_parts: Vec<String> = t
            .items
            .iter()
            .filter_map(|item| {
                if let syn::TraitItem::Fn(m) = item {
                    Some(m.sig.ident.to_string())
                } else {
                    None
                }
            })
            .collect();
        let body_text = body_parts.join(",");

        ContentHashes {
            full_hash: blake3::hash(full_text.as_bytes()).into(),
            signature_hash: blake3::hash(t.ident.to_string().as_bytes()).into(),
            body_hash: blake3::hash(body_text.as_bytes()).into(),
            layout_hash: [0u8; 32], // Not applicable for traits
        }
    }

    /// Extract the source text for a span.
    fn extract_span_text(&self, source: &str, span: proc_macro2::Span) -> String {
        let start = span.start();
        let end = span.end();

        let lines: Vec<&str> = source.lines().collect();

        if start.line == 0 || end.line == 0 || start.line > lines.len() {
            return String::new();
        }

        let mut result = String::new();

        for (i, line) in lines.iter().enumerate() {
            let line_num = i + 1; // 1-indexed
            if line_num < start.line || line_num > end.line {
                continue;
            }

            if line_num == start.line && line_num == end.line {
                // Single line span
                let start_col = start.column.min(line.len());
                let end_col = end.column.min(line.len());
                result.push_str(&line[start_col..end_col]);
            } else if line_num == start.line {
                let start_col = start.column.min(line.len());
                result.push_str(&line[start_col..]);
                result.push('\n');
            } else if line_num == end.line {
                let end_col = end.column.min(line.len());
                result.push_str(&line[..end_col]);
            } else {
                result.push_str(line);
                result.push('\n');
            }
        }

        result
    }
}

impl Default for RustParser {
    fn default() -> Self {
        Self::new()
    }
}
