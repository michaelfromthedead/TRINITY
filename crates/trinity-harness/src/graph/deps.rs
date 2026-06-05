//! Dependency detection and edge creation.

use std::collections::HashMap;

use super::{CodeEdge, CodeGraph, EdgeType, NodeId};

/// A raw dependency reference before resolution.
#[derive(Debug, Clone)]
pub struct RawDependency {
    /// File path of the source node.
    pub from_file: String,
    /// Name of the source code unit.
    pub from_name: String,
    /// Line number of the source (for disambiguation).
    pub from_line: usize,
    /// The name/path being referenced.
    pub to_ref: String,
    /// Type of dependency.
    pub dep_type: DepType,
}

/// Type of dependency relationship.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepType {
    /// Import/use statement.
    Imports,
    /// Function/method call.
    Calls,
    /// Type reference (in signatures, fields).
    Uses,
}

impl From<DepType> for EdgeType {
    fn from(dt: DepType) -> Self {
        match dt {
            DepType::Imports => EdgeType::Imports,
            DepType::Calls => EdgeType::Calls,
            DepType::Uses => EdgeType::Uses,
        }
    }
}

/// Statistics from dependency analysis.
#[derive(Debug, Clone, Default)]
pub struct DepStats {
    /// Raw dependencies found.
    pub deps_found: usize,
    /// Dependencies successfully resolved to edges.
    pub deps_resolved: usize,
    /// Dependencies that couldn't be resolved (external or missing).
    pub deps_unresolved: usize,
    /// Edges created per type.
    pub edges_by_type: HashMap<EdgeType, usize>,
}

impl DepStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_found(&mut self) {
        self.deps_found += 1;
    }

    pub fn record_resolved(&mut self, edge_type: EdgeType) {
        self.deps_resolved += 1;
        *self.edges_by_type.entry(edge_type).or_insert(0) += 1;
    }

    pub fn record_unresolved(&mut self) {
        self.deps_unresolved += 1;
    }
}

/// Analyzer for extracting dependencies from Rust source.
pub struct RustDepAnalyzer;

impl RustDepAnalyzer {
    pub fn new() -> Self {
        Self
    }

    /// Extract dependencies from Rust source code.
    pub fn analyze(&self, source: &str, file_path: &str) -> Vec<RawDependency> {
        let mut deps = Vec::new();

        let Ok(file) = syn::parse_file(source) else {
            return deps;
        };

        // Extract use statements (imports)
        for item in &file.items {
            match item {
                syn::Item::Use(u) => {
                    self.extract_use_deps(&u.tree, file_path, &mut deps);
                }
                syn::Item::Fn(f) => {
                    let fn_name = f.sig.ident.to_string();
                    let fn_line = f.sig.ident.span().start().line;
                    // Extract calls from function body
                    self.extract_calls_from_block(&f.block, file_path, &fn_name, fn_line, &mut deps);
                    // Extract type references from signature
                    self.extract_type_refs_from_sig(&f.sig, file_path, &fn_name, fn_line, &mut deps);
                }
                syn::Item::Impl(i) => {
                    let impl_name = self.type_to_string(&i.self_ty);
                    let impl_line = i.impl_token.span.start().line;
                    // Extract method calls
                    for item in &i.items {
                        if let syn::ImplItem::Fn(m) = item {
                            let method_name = format!("{}::{}", impl_name, m.sig.ident);
                            let method_line = m.sig.ident.span().start().line;
                            self.extract_calls_from_block(&m.block, file_path, &method_name, method_line, &mut deps);
                        }
                    }
                    // Extract trait reference
                    if let Some((_, path, _)) = &i.trait_ {
                        if let Some(seg) = path.segments.last() {
                            deps.push(RawDependency {
                                from_file: file_path.to_string(),
                                from_name: format!("impl {}", impl_name),
                                from_line: impl_line,
                                to_ref: seg.ident.to_string(),
                                dep_type: DepType::Uses,
                            });
                        }
                    }
                }
                syn::Item::Struct(s) => {
                    let struct_name = s.ident.to_string();
                    let struct_line = s.ident.span().start().line;
                    self.extract_field_type_refs(s, file_path, &struct_name, struct_line, &mut deps);
                }
                _ => {}
            }
        }

        deps
    }

    fn extract_use_deps(&self, tree: &syn::UseTree, file_path: &str, deps: &mut Vec<RawDependency>) {
        match tree {
            syn::UseTree::Path(p) => {
                self.extract_use_deps(&p.tree, file_path, deps);
            }
            syn::UseTree::Name(n) => {
                deps.push(RawDependency {
                    from_file: file_path.to_string(),
                    from_name: String::new(), // File-level import
                    from_line: 0,
                    to_ref: n.ident.to_string(),
                    dep_type: DepType::Imports,
                });
            }
            syn::UseTree::Rename(r) => {
                deps.push(RawDependency {
                    from_file: file_path.to_string(),
                    from_name: String::new(),
                    from_line: 0,
                    to_ref: r.ident.to_string(),
                    dep_type: DepType::Imports,
                });
            }
            syn::UseTree::Group(g) => {
                for item in &g.items {
                    self.extract_use_deps(item, file_path, deps);
                }
            }
            syn::UseTree::Glob(_) => {
                // Glob imports are hard to resolve statically
            }
        }
    }

    fn extract_calls_from_block(
        &self,
        block: &syn::Block,
        file_path: &str,
        from_name: &str,
        from_line: usize,
        deps: &mut Vec<RawDependency>,
    ) {
        use syn::visit::Visit;

        struct CallVisitor<'a> {
            deps: &'a mut Vec<RawDependency>,
            file_path: String,
            from_name: String,
            from_line: usize,
        }

        impl<'ast> syn::visit::Visit<'ast> for CallVisitor<'_> {
            fn visit_expr_call(&mut self, node: &'ast syn::ExprCall) {
                if let syn::Expr::Path(p) = node.func.as_ref() {
                    if let Some(seg) = p.path.segments.last() {
                        self.deps.push(RawDependency {
                            from_file: self.file_path.clone(),
                            from_name: self.from_name.clone(),
                            from_line: self.from_line,
                            to_ref: seg.ident.to_string(),
                            dep_type: DepType::Calls,
                        });
                    }
                }
                syn::visit::visit_expr_call(self, node);
            }

            fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
                self.deps.push(RawDependency {
                    from_file: self.file_path.clone(),
                    from_name: self.from_name.clone(),
                    from_line: self.from_line,
                    to_ref: node.method.to_string(),
                    dep_type: DepType::Calls,
                });
                syn::visit::visit_expr_method_call(self, node);
            }
        }

        let mut visitor = CallVisitor {
            deps,
            file_path: file_path.to_string(),
            from_name: from_name.to_string(),
            from_line,
        };
        visitor.visit_block(block);
    }

    fn extract_type_refs_from_sig(
        &self,
        sig: &syn::Signature,
        file_path: &str,
        from_name: &str,
        from_line: usize,
        deps: &mut Vec<RawDependency>,
    ) {
        // Extract types from parameters
        for arg in &sig.inputs {
            if let syn::FnArg::Typed(pat) = arg {
                self.extract_type_ref(&pat.ty, file_path, from_name, from_line, deps);
            }
        }
        // Extract return type
        if let syn::ReturnType::Type(_, ty) = &sig.output {
            self.extract_type_ref(ty, file_path, from_name, from_line, deps);
        }
    }

    fn extract_type_ref(
        &self,
        ty: &syn::Type,
        file_path: &str,
        from_name: &str,
        from_line: usize,
        deps: &mut Vec<RawDependency>,
    ) {
        match ty {
            syn::Type::Path(p) => {
                if let Some(seg) = p.path.segments.last() {
                    let name = seg.ident.to_string();
                    // Skip primitive types
                    if !is_primitive_type(&name) {
                        deps.push(RawDependency {
                            from_file: file_path.to_string(),
                            from_name: from_name.to_string(),
                            from_line,
                            to_ref: name,
                            dep_type: DepType::Uses,
                        });
                    }
                }
            }
            syn::Type::Reference(r) => {
                self.extract_type_ref(&r.elem, file_path, from_name, from_line, deps);
            }
            syn::Type::Slice(s) => {
                self.extract_type_ref(&s.elem, file_path, from_name, from_line, deps);
            }
            syn::Type::Array(a) => {
                self.extract_type_ref(&a.elem, file_path, from_name, from_line, deps);
            }
            syn::Type::Tuple(t) => {
                for elem in &t.elems {
                    self.extract_type_ref(elem, file_path, from_name, from_line, deps);
                }
            }
            _ => {}
        }
    }

    fn extract_field_type_refs(
        &self,
        s: &syn::ItemStruct,
        file_path: &str,
        struct_name: &str,
        struct_line: usize,
        deps: &mut Vec<RawDependency>,
    ) {
        match &s.fields {
            syn::Fields::Named(fields) => {
                for field in &fields.named {
                    self.extract_type_ref(&field.ty, file_path, struct_name, struct_line, deps);
                }
            }
            syn::Fields::Unnamed(fields) => {
                for field in &fields.unnamed {
                    self.extract_type_ref(&field.ty, file_path, struct_name, struct_line, deps);
                }
            }
            syn::Fields::Unit => {}
        }
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
            _ => "?".to_string(),
        }
    }
}

impl Default for RustDepAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Analyzer for extracting dependencies from Python source.
pub struct PythonDepAnalyzer;

impl PythonDepAnalyzer {
    pub fn new() -> Self {
        Self
    }

    /// Extract dependencies from Python source code.
    pub fn analyze(&self, source: &str, file_path: &str) -> Vec<RawDependency> {
        use rustpython_parser::ast::Mod;
        use rustpython_parser::{parse, Mode};

        let mut deps = Vec::new();

        let Ok(ast) = parse(source, Mode::Module, "<input>") else {
            return deps;
        };

        if let Mod::Module(m) = ast {
            for stmt in &m.body {
                self.extract_stmt_deps(source, file_path, stmt, "", 0, &mut deps);
            }
        }

        deps
    }

    fn extract_stmt_deps(
        &self,
        source: &str,
        file_path: &str,
        stmt: &rustpython_parser::ast::Stmt,
        context_name: &str,
        context_line: usize,
        deps: &mut Vec<RawDependency>,
    ) {
        use rustpython_parser::ast::Stmt;

        match stmt {
            Stmt::Import(i) => {
                for alias in &i.names {
                    deps.push(RawDependency {
                        from_file: file_path.to_string(),
                        from_name: context_name.to_string(),
                        from_line: context_line,
                        to_ref: alias.name.to_string(),
                        dep_type: DepType::Imports,
                    });
                }
            }
            Stmt::ImportFrom(i) => {
                if let Some(module) = &i.module {
                    deps.push(RawDependency {
                        from_file: file_path.to_string(),
                        from_name: context_name.to_string(),
                        from_line: context_line,
                        to_ref: module.to_string(),
                        dep_type: DepType::Imports,
                    });
                }
                for alias in &i.names {
                    deps.push(RawDependency {
                        from_file: file_path.to_string(),
                        from_name: context_name.to_string(),
                        from_line: context_line,
                        to_ref: alias.name.to_string(),
                        dep_type: DepType::Imports,
                    });
                }
            }
            Stmt::FunctionDef(f) => {
                let fn_name = f.name.to_string();
                let fn_line = u32::from(f.range.start()) as usize;
                // Extract calls from function body
                for body_stmt in &f.body {
                    self.extract_calls_from_stmt(file_path, &fn_name, fn_line, body_stmt, deps);
                }
            }
            Stmt::AsyncFunctionDef(f) => {
                let fn_name = f.name.to_string();
                let fn_line = u32::from(f.range.start()) as usize;
                for body_stmt in &f.body {
                    self.extract_calls_from_stmt(file_path, &fn_name, fn_line, body_stmt, deps);
                }
            }
            Stmt::ClassDef(c) => {
                let class_name = c.name.to_string();
                let class_line = u32::from(c.range.start()) as usize;
                // Extract base class references
                for base in &c.bases {
                    if let rustpython_parser::ast::Expr::Name(n) = base {
                        deps.push(RawDependency {
                            from_file: file_path.to_string(),
                            from_name: class_name.clone(),
                            from_line: class_line,
                            to_ref: n.id.to_string(),
                            dep_type: DepType::Uses,
                        });
                    }
                }
                // Recurse into class body
                for body_stmt in &c.body {
                    self.extract_stmt_deps(source, file_path, body_stmt, &class_name, class_line, deps);
                }
            }
            _ => {}
        }
    }

    fn extract_calls_from_stmt(
        &self,
        file_path: &str,
        from_name: &str,
        from_line: usize,
        stmt: &rustpython_parser::ast::Stmt,
        deps: &mut Vec<RawDependency>,
    ) {
        use rustpython_parser::ast::Stmt;

        match stmt {
            Stmt::Expr(e) => {
                self.extract_calls_from_expr(file_path, from_name, from_line, &e.value, deps);
            }
            Stmt::Assign(a) => {
                self.extract_calls_from_expr(file_path, from_name, from_line, &a.value, deps);
            }
            Stmt::Return(r) => {
                if let Some(value) = &r.value {
                    self.extract_calls_from_expr(file_path, from_name, from_line, value, deps);
                }
            }
            Stmt::If(i) => {
                self.extract_calls_from_expr(file_path, from_name, from_line, &i.test, deps);
                for body_stmt in &i.body {
                    self.extract_calls_from_stmt(file_path, from_name, from_line, body_stmt, deps);
                }
                for else_stmt in &i.orelse {
                    self.extract_calls_from_stmt(file_path, from_name, from_line, else_stmt, deps);
                }
            }
            Stmt::For(f) => {
                self.extract_calls_from_expr(file_path, from_name, from_line, &f.iter, deps);
                for body_stmt in &f.body {
                    self.extract_calls_from_stmt(file_path, from_name, from_line, body_stmt, deps);
                }
            }
            Stmt::While(w) => {
                self.extract_calls_from_expr(file_path, from_name, from_line, &w.test, deps);
                for body_stmt in &w.body {
                    self.extract_calls_from_stmt(file_path, from_name, from_line, body_stmt, deps);
                }
            }
            _ => {}
        }
    }

    fn extract_calls_from_expr(
        &self,
        file_path: &str,
        from_name: &str,
        from_line: usize,
        expr: &rustpython_parser::ast::Expr,
        deps: &mut Vec<RawDependency>,
    ) {
        use rustpython_parser::ast::Expr;

        match expr {
            Expr::Call(c) => {
                // Extract function name being called
                match c.func.as_ref() {
                    Expr::Name(n) => {
                        deps.push(RawDependency {
                            from_file: file_path.to_string(),
                            from_name: from_name.to_string(),
                            from_line,
                            to_ref: n.id.to_string(),
                            dep_type: DepType::Calls,
                        });
                    }
                    Expr::Attribute(a) => {
                        deps.push(RawDependency {
                            from_file: file_path.to_string(),
                            from_name: from_name.to_string(),
                            from_line,
                            to_ref: a.attr.to_string(),
                            dep_type: DepType::Calls,
                        });
                    }
                    _ => {}
                }
                // Recurse into arguments
                for arg in &c.args {
                    self.extract_calls_from_expr(file_path, from_name, from_line, arg, deps);
                }
            }
            Expr::BinOp(b) => {
                self.extract_calls_from_expr(file_path, from_name, from_line, &b.left, deps);
                self.extract_calls_from_expr(file_path, from_name, from_line, &b.right, deps);
            }
            Expr::Compare(c) => {
                self.extract_calls_from_expr(file_path, from_name, from_line, &c.left, deps);
                for comp in &c.comparators {
                    self.extract_calls_from_expr(file_path, from_name, from_line, comp, deps);
                }
            }
            Expr::List(l) => {
                for elem in &l.elts {
                    self.extract_calls_from_expr(file_path, from_name, from_line, elem, deps);
                }
            }
            Expr::Dict(d) => {
                for value in &d.values {
                    self.extract_calls_from_expr(file_path, from_name, from_line, value, deps);
                }
            }
            Expr::IfExp(i) => {
                self.extract_calls_from_expr(file_path, from_name, from_line, &i.test, deps);
                self.extract_calls_from_expr(file_path, from_name, from_line, &i.body, deps);
                self.extract_calls_from_expr(file_path, from_name, from_line, &i.orelse, deps);
            }
            _ => {}
        }
    }
}

impl Default for PythonDepAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if a type name is a Rust primitive.
fn is_primitive_type(name: &str) -> bool {
    matches!(
        name,
        "bool"
            | "char"
            | "str"
            | "i8"
            | "i16"
            | "i32"
            | "i64"
            | "i128"
            | "isize"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "u128"
            | "usize"
            | "f32"
            | "f64"
            | "String"
            | "Vec"
            | "Option"
            | "Result"
            | "Box"
            | "Rc"
            | "Arc"
            | "Cell"
            | "RefCell"
            | "Mutex"
            | "RwLock"
            | "HashMap"
            | "HashSet"
            | "BTreeMap"
            | "BTreeSet"
            | "Self"
    )
}

/// Resolve raw dependencies to edges in the graph.
pub fn resolve_deps_to_edges(
    graph: &mut CodeGraph,
    deps: &[RawDependency],
) -> DepStats {
    let mut stats = DepStats::new();

    // Build index: (file_path, name) -> NodeId
    let mut node_index: HashMap<(String, String), NodeId> = HashMap::new();
    // Also index by just name for cross-file resolution
    let mut name_index: HashMap<String, Vec<NodeId>> = HashMap::new();

    for node in graph.nodes() {
        let key = (node.file_path.clone(), node.name().to_string());
        node_index.insert(key, node.id);
        name_index
            .entry(node.name().to_string())
            .or_default()
            .push(node.id);
    }

    for dep in deps {
        stats.record_found();

        // Try to find source node
        let source_key = (dep.from_file.clone(), dep.from_name.clone());
        let source_id = if dep.from_name.is_empty() {
            // File-level dependency, skip edge creation (no source node)
            stats.record_unresolved();
            continue;
        } else {
            node_index.get(&source_key).copied()
        };

        // Try to find target node by name
        let target_ids = name_index.get(&dep.to_ref);

        match (source_id, target_ids) {
            (Some(src), Some(targets)) if !targets.is_empty() => {
                // Create edge to first matching target (could be smarter with qualification)
                let tgt = targets[0];
                let edge_type = EdgeType::from(dep.dep_type);
                graph.add_edge(CodeEdge::new(src, tgt, edge_type));
                stats.record_resolved(edge_type);
            }
            _ => {
                stats.record_unresolved();
            }
        }
    }

    stats
}
