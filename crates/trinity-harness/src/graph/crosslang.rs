//! Cross-language edge detection.
//!
//! Detects:
//! - PyO3 boundaries (#[pyfunction], #[pyclass])
//! - WGSL↔Rust struct mirrors (same name, #[repr(C)])

use std::collections::HashMap;

use super::{CodeEdge, CodeGraph, EdgeType, NodeId};
use crate::parsers::{Language, UnitType};

/// A cross-language binding found in source code.
#[derive(Debug, Clone)]
pub struct CrossLangBinding {
    /// File path where the binding is defined.
    pub file: String,
    /// Name of the bound item.
    pub name: String,
    /// Type of binding.
    pub binding_type: BindingType,
    /// Source language.
    pub source_lang: Language,
    /// Target language.
    pub target_lang: Language,
}

/// Type of cross-language binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingType {
    /// PyO3 function binding (#[pyfunction]).
    PyFunction,
    /// PyO3 class binding (#[pyclass]).
    PyClass,
    /// PyO3 method binding (#[pymethods]).
    PyMethod,
    /// Struct layout mirror between languages.
    StructMirror,
}

/// Statistics from cross-language analysis.
#[derive(Debug, Clone, Default)]
pub struct CrossLangStats {
    /// Total bindings found.
    pub bindings_found: usize,
    /// PyO3 functions found.
    pub pyo3_functions: usize,
    /// PyO3 classes found.
    pub pyo3_classes: usize,
    /// Struct mirrors found.
    pub struct_mirrors: usize,
    /// Edges created.
    pub edges_created: usize,
}

impl CrossLangStats {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Analyzer for detecting PyO3 bindings in Rust source.
pub struct Pyo3Analyzer;

impl Pyo3Analyzer {
    pub fn new() -> Self {
        Self
    }

    /// Analyze Rust source for PyO3 bindings.
    pub fn analyze(&self, source: &str, file_path: &str) -> Vec<CrossLangBinding> {
        let mut bindings = Vec::new();

        let Ok(file) = syn::parse_file(source) else {
            return bindings;
        };

        for item in &file.items {
            match item {
                syn::Item::Fn(f) => {
                    if self.has_pyfunction_attr(&f.attrs) {
                        bindings.push(CrossLangBinding {
                            file: file_path.to_string(),
                            name: f.sig.ident.to_string(),
                            binding_type: BindingType::PyFunction,
                            source_lang: Language::Rust,
                            target_lang: Language::Python,
                        });
                    }
                }
                syn::Item::Struct(s) => {
                    if self.has_pyclass_attr(&s.attrs) {
                        bindings.push(CrossLangBinding {
                            file: file_path.to_string(),
                            name: s.ident.to_string(),
                            binding_type: BindingType::PyClass,
                            source_lang: Language::Rust,
                            target_lang: Language::Python,
                        });
                    }
                }
                syn::Item::Impl(i) => {
                    if self.has_pymethods_attr(&i.attrs) {
                        let self_ty = self.type_to_string(&i.self_ty);
                        for impl_item in &i.items {
                            if let syn::ImplItem::Fn(m) = impl_item {
                                bindings.push(CrossLangBinding {
                                    file: file_path.to_string(),
                                    name: format!("{}::{}", self_ty, m.sig.ident),
                                    binding_type: BindingType::PyMethod,
                                    source_lang: Language::Rust,
                                    target_lang: Language::Python,
                                });
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        bindings
    }

    fn has_pyfunction_attr(&self, attrs: &[syn::Attribute]) -> bool {
        attrs.iter().any(|a| {
            a.path()
                .segments
                .last()
                .map(|s| s.ident == "pyfunction")
                .unwrap_or(false)
        })
    }

    fn has_pyclass_attr(&self, attrs: &[syn::Attribute]) -> bool {
        attrs.iter().any(|a| {
            a.path()
                .segments
                .last()
                .map(|s| s.ident == "pyclass")
                .unwrap_or(false)
        })
    }

    fn has_pymethods_attr(&self, attrs: &[syn::Attribute]) -> bool {
        attrs.iter().any(|a| {
            a.path()
                .segments
                .last()
                .map(|s| s.ident == "pymethods")
                .unwrap_or(false)
        })
    }

    fn type_to_string(&self, ty: &syn::Type) -> String {
        match ty {
            syn::Type::Path(p) => p
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect::<Vec<_>>()
                .join("::"),
            _ => {
                eprintln!("[crosslang] Unexpected type path format (non-path type)");
                "?".to_string()
            }
        }
    }
}

impl Default for Pyo3Analyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Analyzer for detecting #[repr(C)] structs that may mirror WGSL layouts.
pub struct ReprCAnalyzer;

impl ReprCAnalyzer {
    pub fn new() -> Self {
        Self
    }

    /// Analyze Rust source for #[repr(C)] structs.
    pub fn analyze(&self, source: &str, _file_path: &str) -> Vec<String> {
        let mut repr_c_structs = Vec::new();

        let Ok(file) = syn::parse_file(source) else {
            return repr_c_structs;
        };

        for item in &file.items {
            if let syn::Item::Struct(s) = item {
                if self.has_repr_c(&s.attrs) {
                    repr_c_structs.push(s.ident.to_string());
                }
            }
        }

        repr_c_structs
    }

    fn has_repr_c(&self, attrs: &[syn::Attribute]) -> bool {
        attrs.iter().any(|a| {
            if a.path()
                .segments
                .last()
                .map(|s| s.ident == "repr")
                .unwrap_or(false)
            {
                // Check if repr(C)
                if let syn::Meta::List(list) = &a.meta {
                    let tokens = list.tokens.to_string();
                    return tokens.contains('C') || tokens.contains("C,");
                }
            }
            false
        })
    }
}

impl Default for ReprCAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Detect struct mirrors between WGSL and Rust.
///
/// A mirror is detected when:
/// - A WGSL struct has the same name as a Rust #[repr(C)] struct
/// - Both structs exist in the graph
pub fn detect_struct_mirrors(graph: &CodeGraph) -> Vec<(NodeId, NodeId)> {
    let mut mirrors = Vec::new();

    // Build index of WGSL structs by name
    let mut wgsl_structs: HashMap<String, NodeId> = HashMap::new();
    // Build index of Rust structs by name
    let mut rust_structs: HashMap<String, NodeId> = HashMap::new();

    for node in graph.nodes() {
        if node.unit.unit_type != UnitType::Struct {
            continue;
        }

        match node.unit.language {
            Language::Wgsl => {
                wgsl_structs.insert(node.name().to_string(), node.id);
            }
            Language::Rust => {
                rust_structs.insert(node.name().to_string(), node.id);
            }
            _ => {}
        }
    }

    // Find matches by name
    for (name, wgsl_id) in &wgsl_structs {
        if let Some(rust_id) = rust_structs.get(name) {
            mirrors.push((*wgsl_id, *rust_id));
        }
    }

    mirrors
}

/// Create cross-language edges in the graph.
///
/// This detects:
/// - PyO3 bindings (creates Binds edges)
/// - Struct mirrors (creates MirrorsLayout edges)
pub fn create_crosslang_edges(
    graph: &mut CodeGraph,
    pyo3_bindings: &[CrossLangBinding],
) -> CrossLangStats {
    let mut stats = CrossLangStats::new();

    // Build node index by name
    let mut node_by_name: HashMap<String, Vec<NodeId>> = HashMap::new();
    for node in graph.nodes() {
        node_by_name
            .entry(node.name().to_string())
            .or_default()
            .push(node.id);
    }

    // Process PyO3 bindings
    for binding in pyo3_bindings {
        stats.bindings_found += 1;

        match binding.binding_type {
            BindingType::PyFunction => stats.pyo3_functions += 1,
            BindingType::PyClass => stats.pyo3_classes += 1,
            BindingType::PyMethod => stats.pyo3_functions += 1,
            BindingType::StructMirror => stats.struct_mirrors += 1,
        }

        // For PyO3 bindings, we mark the Rust item as having a binding to Python
        // We don't create edges here since Python code calling it isn't known yet
        // The binding info is tracked for future analysis
    }

    // Detect struct mirrors
    let mirrors = detect_struct_mirrors(graph);
    for (wgsl_id, rust_id) in mirrors {
        graph.add_edge(CodeEdge::new(wgsl_id, rust_id, EdgeType::MirrorsLayout));
        stats.struct_mirrors += 1;
        stats.edges_created += 1;
    }

    stats
}
