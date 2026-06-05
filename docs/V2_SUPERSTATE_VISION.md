# V2 Superstate Vision — Code as Statechart

**Started:** 2026-06-04
**Status:** CONCEPTUAL / DESIGN PHASE
**Authors:** Michael + Claude (collaborative exploration)

---

## Executive Summary

What if code itself was modeled as a statechart? Where every unit (function, module, file) carries state (tested, stale, changed), and the entire codebase forms a hierarchical state machine that can be formally verified?

This document captures a design exploration that unifies:
- **superstate** — Harel statechart engine with verification
- **synth** — Constraint-driven test data generation
- **Testing harness** — QA and improvement campaigns

The central insight: **Code and tests are not two artifacts to keep in sync. They are two facets of the same stateful entity.**

---

## Part 1: The Problem

### Current Reality

```
┌─────────────┐         ┌─────────────┐
│    CODE     │ ~~~~~~~ │    TEST     │
│ (artifact)  │  "sync" │ (artifact)  │
└─────────────┘         └─────────────┘
      ↑                       ↑
      └── can drift ──────────┘
```

Code and tests are separate artifacts. They "drift" because:
- Tests can become stale (code changed, test didn't)
- Tests can lie (test passes but doesn't verify what it claims)
- Tests can miss (code has untested paths)
- Coverage metrics count lines, not behaviors

We run `cargo test`, it passes, we move on. Next day, someone changes a utility function that 47 other functions call. Are those 47 functions still "tested"? Technically yes (the tests still pass). But semantically? They're **stale** — tested against old assumptions.

### The Hidden State

Every function has implicit state we don't track:

| State | Meaning |
|-------|---------|
| `UNTOUCHED` | New code, never tested |
| `TESTED_GREEN` | Tests passed |
| `TESTED_RED` | Tests failed |
| `STALE_DIRECT` | This code changed since last test |
| `STALE_TRANSITIVE` | A dependency changed |
| `STALE_DEEP` | A dependency of a dependency changed |
| `QA_APPROVED` | Passed adversarial review |
| `UNKNOWN` | Tests exist but weren't run |

We just... don't track this. The information exists (in git, in test results, in CI logs) but it's not unified into a coherent state model.

---

## Part 2: Code as Statechart

### The Core Insight

Code isn't static. Code is **state** — and we've been pretending it isn't.

If we model the codebase as a Harel statechart:
- **Nodes** = code units (functions, modules, files, crates)
- **States** = test/QA status (GREEN, STALE, CHANGED, etc.)
- **Events** = changes (file modified, test run, QA approved)
- **Transitions** = state changes in response to events

Then the entire codebase becomes a formally verifiable state machine.

### The Two Graphs

There are actually **two structures** at play:

#### Structure 1: The Dependency Graph

```
┌─────────────────────────────────────────────────────────────────┐
│                    CODE DEPENDENCY GRAPH                        │
│                                                                 │
│    ┌─────────┐         ┌─────────┐         ┌─────────┐         │
│    │ module  │────────▶│ module  │────────▶│ module  │         │
│    │   A     │         │   B     │         │   C     │         │
│    └─────────┘         └─────────┘         └─────────┘         │
│         │                   │                                   │
│         │              ┌────┴────┐                              │
│         │              ▼         ▼                              │
│         │         ┌─────────┐ ┌─────────┐                       │
│         └────────▶│ module  │ │ module  │                       │
│                   │   D     │ │   E     │                       │
│                   └─────────┘ └─────────┘                       │
│                                                                 │
│    Nodes = Code Units (crate, module, function, etc.)          │
│    Edges = "depends on" / "calls" / "imports"                  │
└─────────────────────────────────────────────────────────────────┘
```

This is a **DAG** (directed acyclic graph). It's the *structure* of your code. Static. Changes only when code structure changes.

#### Structure 2: The Statechart (per Node)

```
┌─────────────────────────────────────────────────────────────────┐
│                    STATECHART (per code unit)                   │
│                                                                 │
│    ┌──────────────────────────────────────────────────────┐    │
│    │                      KNOWN                            │    │
│    │  ┌─────────┐     ┌─────────┐     ┌─────────┐         │    │
│    │  │ TESTED  │────▶│  STALE  │────▶│TESTED   │         │    │
│    │  │ GREEN   │     │         │     │ RED     │         │    │
│    │  └─────────┘     └─────────┘     └─────────┘         │    │
│    └──────────────────────────────────────────────────────┘    │
│                           │                                     │
│              ┌────────────┴────────────┐                       │
│              ▼                         ▼                       │
│    ┌─────────────────┐       ┌─────────────────┐               │
│    │   QA_APPROVED   │       │   QUARANTINED   │               │
│    └─────────────────┘       └─────────────────┘               │
│                                                                 │
│    This is the INTERNAL STATE of ONE node.                     │
│    Each node in the graph has its own statechart instance.     │
└─────────────────────────────────────────────────────────────────┘
```

This is a **state machine**. It's the *status* of one unit. Dynamic. Changes on events.

#### The Insight: Graph + Statechart = Propagation

The magic is combining them:

```
EVENT: "module_B.rs changed"
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│    ┌─────────┐         ┌─────────┐         ┌─────────┐         │
│    │    A    │────────▶│    B    │────────▶│    C    │         │
│    │ GREEN   │         │ CHANGED │         │ STALE   │         │
│    └─────────┘         └─────────┘         └─────────┘         │
│                             │                   ▲               │
│                        Event propagates         │               │
│                        along edges ─────────────┘               │
│                                                                 │
│    B changed → B is CHANGED                                    │
│    C depends on B → C is STALE_DIRECT                          │
│    (anything depending on C → STALE_TRANSITIVE)                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### The Two Graphs Clarified

| Aspect | Dependency Graph | Statechart |
|--------|-----------------|------------|
| **What** | Structure of code | State of one unit |
| **Nodes** | Code units | States (Green, Stale, etc.) |
| **Edges** | "depends on" | "transitions to" |
| **Instance** | One per codebase | One per code unit |
| **Changes** | When code structure changes | On every event |
| **Type** | DAG (usually) | FSM (finite state machine) |

The dependency graph is the *skeleton*, and each bone has a statechart heartbeat.

### The Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                         CRATE [STALE]                           │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    MODULE: gpu_driven [STALE]              │  │
│  │  ┌─────────────────┐  ┌─────────────────┐                 │  │
│  │  │ FILE: build.rs  │  │ FILE: object.rs │                 │  │
│  │  │     [GREEN]     │  │     [STALE]     │                 │  │
│  │  │  ┌───────────┐  │  │  ┌───────────┐  │                 │  │
│  │  │  │ fn cpu_*  │  │  │  │ fn new()  │  │                 │  │
│  │  │  │  [GREEN]  │  │  │  │ [GREEN]   │  │                 │  │
│  │  │  └───────────┘  │  │  └───────────┘  │                 │  │
│  │  │  ┌───────────┐  │  │  ┌───────────┐  │                 │  │
│  │  │  │ fn gpu_*  │  │  │  │ fn mesh() │  │                 │  │
│  │  │  │  [GREEN]  │  │  │  │ [CHANGED] │◄─┼── This changed  │  │
│  │  │  └───────────┘  │  │  └───────────┘  │                 │  │
│  │  └─────────────────┘  └─────────────────┘                 │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

State propagates **upward**:
- `fn mesh()` CHANGED → FILE `object.rs` STALE → MODULE `gpu_driven` STALE → CRATE STALE

But GREEN doesn't automatically propagate up:
- `fn mesh()` GREEN → FILE might still be STALE (other functions)
- FILE GREEN → MODULE GREEN only if ALL files GREEN

### Why All Granularities?

Traditional thinking: "Pick a unit level (file? function? module?) because tracking is expensive."

**Our insight:** Why choose? Track ALL levels simultaneously.

| Unit Level | Granularity | Tracking Overhead | Precision |
|------------|-------------|-------------------|-----------|
| Crate | Very coarse | Trivial | Very low |
| Module | Coarse | Low | Low |
| File | Medium | Low | Medium |
| Function | Fine | Medium | High |
| Block/Branch | Very fine | High | Very high |

The "overhead" argument was always about human cognitive load, not machine cost. Machines don't care if there are 10 states or 10,000.

**There is zero penalty for overlap.** If we test at file level AND function level, we're just testing more thoroughly. If overlap becomes a performance problem, we optimize later.

This is **hierarchical state** — exactly what Harel statecharts model:
- Crate = And-state (all modules exist in parallel)
- Module = And-state (all files exist in parallel)
- File = And-state (all functions exist in parallel)
- Function = Or-state or And-state (depending on control flow)

---

## Part 3: The Unified Substrate Architecture

This is the full architecture for a massive, unified testing harness that tracks code state across **Rust, Python, and WGSL** with proper AST parsing for each language.

### Layer 0: File System Foundation

```rust
/// The physical layer - what exists on disk
pub struct FileSystem {
    /// Root directories to watch
    roots: Vec<PathBuf>,
    
    /// File watcher
    watcher: notify::RecommendedWatcher,
    
    /// All known files
    files: HashMap<PathBuf, FileEntry>,
}

pub struct FileEntry {
    path: PathBuf,
    language: Language,
    hash: [u8; 32],
    last_modified: SystemTime,
    
    /// Raw source (lazy loaded)
    source: OnceCell<String>,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub enum Language {
    Rust,
    Python,
    Wgsl,
    Toml,    // Cargo.toml, pyproject.toml
    Json,    // test schemas
    Unknown,
}

impl Language {
    pub fn from_path(path: &Path) -> Self {
        match path.extension().and_then(|e| e.to_str()) {
            Some("rs") => Language::Rust,
            Some("py") => Language::Python,
            Some("wgsl") => Language::Wgsl,
            Some("toml") => Language::Toml,
            Some("json") => Language::Json,
            _ => Language::Unknown,
        }
    }
}
```

### Layer 1: AST Parsers (Per Language)

We use **both** tree-sitter (fast, incremental, multi-language) **and** full AST parsers (precise semantic analysis) for each language.

#### Rust Parser

```rust
use syn::{parse_file, Item, ItemFn, ItemStruct, ItemImpl, ItemMod};
use tree_sitter::Parser as TsParser;

pub struct RustParser {
    /// Full AST parser for deep analysis
    syn_enabled: bool,
    
    /// Fast incremental parser for change detection
    tree_sitter: TsParser,
    tree_sitter_lang: tree_sitter_rust::language(),
}

impl RustParser {
    pub fn parse(&self, source: &str) -> RustAst {
        // Tree-sitter for structure (fast, incremental)
        let tree = self.tree_sitter.parse(source, None).unwrap();
        
        // Syn for deep analysis (types, generics, macros)
        let syn_ast = if self.syn_enabled {
            Some(syn::parse_file(source).ok())
        } else {
            None
        };
        
        RustAst { tree, syn_ast }
    }
    
    pub fn extract_units(&self, ast: &RustAst) -> Vec<RustUnit> {
        let mut units = Vec::new();
        
        // Extract from syn AST
        if let Some(Some(file)) = &ast.syn_ast {
            for item in &file.items {
                match item {
                    Item::Fn(f) => units.push(RustUnit::Function {
                        name: f.sig.ident.to_string(),
                        visibility: extract_vis(&f.vis),
                        signature: extract_signature(&f.sig),
                        body_hash: hash_tokens(&f.block),
                    }),
                    Item::Struct(s) => units.push(RustUnit::Struct {
                        name: s.ident.to_string(),
                        fields: extract_fields(s),
                        layout_hash: compute_layout_hash(s),
                    }),
                    Item::Impl(i) => units.push(RustUnit::Impl {
                        self_ty: extract_type(&i.self_ty),
                        trait_: i.trait_.as_ref().map(|(_, t, _)| extract_path(t)),
                        methods: extract_impl_methods(i),
                    }),
                    Item::Mod(m) => units.push(RustUnit::Module {
                        name: m.ident.to_string(),
                        inline: m.content.is_some(),
                    }),
                    _ => {}
                }
            }
        }
        
        units
    }
}

#[derive(Debug, Clone)]
pub enum RustUnit {
    Crate { name: String, version: String },
    Module { name: String, inline: bool },
    Function { name: String, visibility: Visibility, signature: Signature, body_hash: [u8; 32] },
    Struct { name: String, fields: Vec<Field>, layout_hash: [u8; 32] },
    Enum { name: String, variants: Vec<Variant> },
    Impl { self_ty: String, trait_: Option<String>, methods: Vec<String> },
    Trait { name: String, methods: Vec<String> },
    Const { name: String, ty: String },
    Static { name: String, ty: String },
    TypeAlias { name: String, target: String },
    Macro { name: String },
}
```

#### Python Parser

```rust
use rustpython_parser::{parse_program, ast};
use tree_sitter::Parser as TsParser;

pub struct PythonParser {
    /// Full AST parser
    rustpython_enabled: bool,
    
    /// Fast incremental parser
    tree_sitter: TsParser,
    tree_sitter_lang: tree_sitter_python::language(),
}

impl PythonParser {
    pub fn parse(&self, source: &str) -> PythonAst {
        let tree = self.tree_sitter.parse(source, None).unwrap();
        
        let full_ast = if self.rustpython_enabled {
            rustpython_parser::parse_program(source, "<source>").ok()
        } else {
            None
        };
        
        PythonAst { tree, full_ast }
    }
    
    pub fn extract_units(&self, ast: &PythonAst) -> Vec<PythonUnit> {
        let mut units = Vec::new();
        
        if let Some(program) = &ast.full_ast {
            for stmt in &program.statements {
                match stmt {
                    ast::Stmt::FunctionDef(f) => units.push(PythonUnit::Function {
                        name: f.name.to_string(),
                        decorators: extract_decorators(&f.decorator_list),
                        signature: extract_py_signature(f),
                        body_hash: hash_stmts(&f.body),
                    }),
                    ast::Stmt::ClassDef(c) => units.push(PythonUnit::Class {
                        name: c.name.to_string(),
                        bases: extract_bases(&c.bases),
                        methods: extract_class_methods(c),
                    }),
                    ast::Stmt::Import(i) => units.push(PythonUnit::Import {
                        names: extract_import_names(i),
                    }),
                    ast::Stmt::ImportFrom(i) => units.push(PythonUnit::ImportFrom {
                        module: i.module.as_ref().map(|m| m.to_string()),
                        names: extract_import_names_from(i),
                    }),
                    _ => {}
                }
            }
        }
        
        units
    }
}

#[derive(Debug, Clone)]
pub enum PythonUnit {
    Module { name: String, docstring: Option<String> },
    Class { name: String, bases: Vec<String>, methods: Vec<String> },
    Function { name: String, decorators: Vec<String>, signature: PySignature, body_hash: [u8; 32] },
    Import { names: Vec<String> },
    ImportFrom { module: Option<String>, names: Vec<String> },
    Variable { name: String, type_hint: Option<String> },
}
```

#### WGSL Parser

```rust
use naga::{front::wgsl, Module as WgslModule};
use tree_sitter::Parser as TsParser;

pub struct WgslParser {
    /// Naga for full semantic analysis
    naga_enabled: bool,
    
    /// Tree-sitter for fast structural parsing
    tree_sitter: TsParser,
    tree_sitter_lang: tree_sitter_wgsl::language(),
}

impl WgslParser {
    pub fn parse(&self, source: &str) -> WgslAst {
        let tree = self.tree_sitter.parse(source, None).unwrap();
        
        let naga_module = if self.naga_enabled {
            wgsl::parse_str(source).ok()
        } else {
            None
        };
        
        WgslAst { tree, naga_module }
    }
    
    pub fn extract_units(&self, ast: &WgslAst) -> Vec<WgslUnit> {
        let mut units = Vec::new();
        
        if let Some(module) = &ast.naga_module {
            // Extract structs with layout info (CRITICAL for alignment bugs!)
            for (handle, ty) in module.types.iter() {
                if let naga::TypeInner::Struct { members, span } = &ty.inner {
                    units.push(WgslUnit::Struct {
                        name: ty.name.clone(),
                        members: members.iter().map(|m| WgslMember {
                            name: m.name.clone(),
                            ty: format_type(&module.types[m.ty]),
                            offset: m.offset,
                            size: compute_member_size(m, &module.types),
                        }).collect(),
                        total_size: *span,
                    });
                }
            }
            
            // Extract functions
            for (handle, func) in module.functions.iter() {
                units.push(WgslUnit::Function {
                    name: func.name.clone(),
                    stage: None,  // Not an entry point
                    workgroup_size: None,
                });
            }
            
            // Extract entry points
            for entry in &module.entry_points {
                units.push(WgslUnit::EntryPoint {
                    name: entry.name.clone(),
                    stage: entry.stage,
                    workgroup_size: entry.workgroup_size,
                });
            }
            
            // Extract bindings (uniform, storage, texture)
            for (handle, var) in module.global_variables.iter() {
                if let Some(binding) = &var.binding {
                    units.push(WgslUnit::Binding {
                        name: var.name.clone(),
                        group: binding.group,
                        binding: binding.binding,
                        address_space: var.space,
                    });
                }
            }
        }
        
        units
    }
}

#[derive(Debug, Clone)]
pub enum WgslUnit {
    Struct { 
        name: Option<String>, 
        members: Vec<WgslMember>, 
        total_size: u32 
    },
    Function { 
        name: Option<String>, 
        stage: Option<naga::ShaderStage>,
        workgroup_size: Option<[u32; 3]>,
    },
    EntryPoint { 
        name: String, 
        stage: naga::ShaderStage,
        workgroup_size: [u32; 3],
    },
    Binding { 
        name: Option<String>, 
        group: u32, 
        binding: u32,
        address_space: naga::AddressSpace,
    },
    Const { name: Option<String>, value: String },
}

#[derive(Debug, Clone)]
pub struct WgslMember {
    name: Option<String>,
    ty: String,
    offset: u32,  // CRITICAL: This catches alignment bugs!
    size: u32,
}
```

### Layer 2: Unified Code Graph

```rust
use petgraph::graph::{DiGraph, NodeIndex};
use slotmap::{SlotMap, new_key_type};

new_key_type! { pub struct NodeId; }
new_key_type! { pub struct EdgeId; }

/// The unified code graph spanning all languages
pub struct CodeGraph {
    /// All nodes in the graph
    nodes: SlotMap<NodeId, CodeNode>,
    
    /// All edges in the graph
    edges: SlotMap<EdgeId, CodeEdge>,
    
    /// Adjacency: node -> outgoing edges
    outgoing: HashMap<NodeId, Vec<EdgeId>>,
    
    /// Adjacency: node -> incoming edges
    incoming: HashMap<NodeId, Vec<EdgeId>>,
    
    /// Index: file path -> nodes in that file
    nodes_by_file: HashMap<PathBuf, Vec<NodeId>>,
    
    /// Index: qualified name -> node
    nodes_by_name: HashMap<QualifiedName, NodeId>,
    
    /// Index: language -> nodes
    nodes_by_language: HashMap<Language, Vec<NodeId>>,
    
    /// Hierarchical index: parent -> children
    children: HashMap<NodeId, Vec<NodeId>>,
    
    /// Hierarchical index: child -> parent
    parent: HashMap<NodeId, NodeId>,
}

/// A node in the code graph (unified across languages)
pub struct CodeNode {
    pub id: NodeId,
    pub file: PathBuf,
    pub span: Span,
    pub kind: CodeNodeKind,
    
    /// The statechart instance for this node
    pub state: CodeStateMachine,
    
    /// Optional contract
    pub contract: Option<Contract>,
    
    /// Cached hashes for change detection
    pub hashes: NodeHashes,
}

#[derive(Debug, Clone)]
pub struct Span {
    pub start_line: u32,
    pub start_col: u32,
    pub end_line: u32,
    pub end_col: u32,
}

#[derive(Debug, Clone)]
pub struct NodeHashes {
    /// Hash of the full source text
    pub full: [u8; 32],
    
    /// Hash of just the signature (for functions/structs)
    pub signature: Option<[u8; 32]>,
    
    /// Hash of just the body (for functions)
    pub body: Option<[u8; 32]>,
    
    /// Hash of memory layout (for structs - catches alignment bugs!)
    pub layout: Option<[u8; 32]>,
}

/// Unified node kinds across all languages
#[derive(Debug, Clone)]
pub enum CodeNodeKind {
    // === FILE LEVEL ===
    File { language: Language },
    
    // === RUST ===
    RustCrate { name: String, version: String },
    RustModule { name: String },
    RustFunction { name: String, sig: RustSignature },
    RustStruct { name: String, layout: StructLayout },
    RustEnum { name: String, variants: Vec<String> },
    RustImpl { self_ty: String, trait_: Option<String> },
    RustTrait { name: String },
    RustMacro { name: String },
    
    // === PYTHON ===
    PythonPackage { name: String },
    PythonModule { name: String },
    PythonClass { name: String, bases: Vec<String> },
    PythonFunction { name: String, sig: PythonSignature },
    PythonMethod { name: String, class: String },
    
    // === WGSL ===
    WgslShader { name: String },
    WgslStruct { name: String, layout: GpuLayout },
    WgslFunction { name: String },
    WgslEntryPoint { name: String, stage: ShaderStage },
    WgslBinding { group: u32, binding: u32 },
    
    // === CROSS-LANGUAGE ===
    /// A virtual node representing a cross-language boundary
    LanguageBoundary { from: Language, to: Language },
}

/// GPU struct layout (for catching alignment bugs)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GpuLayout {
    pub total_size: u32,
    pub alignment: u32,
    pub members: Vec<GpuMember>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GpuMember {
    pub name: String,
    pub offset: u32,
    pub size: u32,
    pub ty: String,
}
```

### Layer 3: Edges (Dependencies & Relationships)

```rust
/// An edge in the code graph
pub struct CodeEdge {
    pub id: EdgeId,
    pub from: NodeId,
    pub to: NodeId,
    pub kind: EdgeKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeKind {
    // === STRUCTURAL ===
    /// Parent contains child (file->function, class->method)
    Contains,
    
    // === DEPENDENCY ===
    /// A imports/uses B
    Imports,
    /// A calls B
    Calls,
    /// A inherits from B (class, trait impl)
    Inherits,
    /// A implements trait B
    Implements,
    /// A references type B
    References,
    
    // === TESTING ===
    /// Test A tests unit B
    Tests,
    /// A is tested by test B (inverse of Tests)
    TestedBy,
    
    // === CROSS-LANGUAGE ===
    /// Rust FFI calls Python (PyO3)
    PyO3Call,
    /// Python calls Rust (via PyO3)
    PyO3Callback,
    /// Rust uses WGSL shader
    UsesShader,
    /// WGSL struct mirrors Rust struct
    MirrorsLayout,
}

impl CodeGraph {
    /// Find all nodes that depend on the given node
    pub fn dependents(&self, node: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        self.incoming.get(&node)
            .into_iter()
            .flat_map(|edges| edges.iter())
            .filter_map(|&edge_id| {
                let edge = &self.edges[edge_id];
                match edge.kind {
                    EdgeKind::Imports | EdgeKind::Calls | EdgeKind::References |
                    EdgeKind::Inherits | EdgeKind::Implements | 
                    EdgeKind::UsesShader | EdgeKind::MirrorsLayout => Some(edge.from),
                    _ => None,
                }
            })
    }
    
    /// Find all nodes that this node depends on
    pub fn dependencies(&self, node: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        self.outgoing.get(&node)
            .into_iter()
            .flat_map(|edges| edges.iter())
            .filter_map(|&edge_id| {
                let edge = &self.edges[edge_id];
                match edge.kind {
                    EdgeKind::Imports | EdgeKind::Calls | EdgeKind::References |
                    EdgeKind::Inherits | EdgeKind::Implements |
                    EdgeKind::UsesShader | EdgeKind::MirrorsLayout => Some(edge.to),
                    _ => None,
                }
            })
    }
    
    /// Find the test nodes that test a given unit
    pub fn tests_for(&self, node: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        self.incoming.get(&node)
            .into_iter()
            .flat_map(|edges| edges.iter())
            .filter_map(|&edge_id| {
                let edge = &self.edges[edge_id];
                if edge.kind == EdgeKind::Tests {
                    Some(edge.from)
                } else {
                    None
                }
            })
    }
    
    /// Find WGSL structs that mirror a Rust struct
    pub fn gpu_mirrors(&self, rust_struct: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        self.incoming.get(&rust_struct)
            .into_iter()
            .flat_map(|edges| edges.iter())
            .filter_map(|&edge_id| {
                let edge = &self.edges[edge_id];
                if edge.kind == EdgeKind::MirrorsLayout {
                    Some(edge.from)
                } else {
                    None
                }
            })
    }
}
```

### Layer 4: Cross-Language Boundary Detection

```rust
pub struct CrossLanguageAnalyzer;

impl CrossLanguageAnalyzer {
    /// Detect Rust ↔ Python boundaries (PyO3)
    pub fn detect_pyo3_boundaries(graph: &mut CodeGraph) {
        // Find #[pyfunction], #[pyclass], #[pymethods]
        for (id, node) in &graph.nodes {
            if let CodeNodeKind::RustFunction { name, .. } = &node.kind {
                // Check if it has #[pyfunction] attribute
                // Create edge: RustFunction --PyO3Callback--> PythonModule
            }
        }
    }
    
    /// Detect Rust ↔ WGSL boundaries
    pub fn detect_shader_boundaries(graph: &mut CodeGraph) {
        // Find include_str!("shader.wgsl") or similar patterns
        // Match Rust structs with #[repr(C)] to WGSL structs
        
        for (rust_id, rust_node) in &graph.nodes {
            if let CodeNodeKind::RustStruct { name, layout } = &rust_node.kind {
                // Find WGSL struct with same name
                for (wgsl_id, wgsl_node) in &graph.nodes {
                    if let CodeNodeKind::WgslStruct { name: wgsl_name, layout: gpu_layout } = &wgsl_node.kind {
                        if name == wgsl_name {
                            // Create edge
                            graph.add_edge(CodeEdge {
                                from: *wgsl_id,
                                to: *rust_id,
                                kind: EdgeKind::MirrorsLayout,
                            });
                            
                            // CRITICAL: Validate layout match!
                            if !layouts_match(layout, gpu_layout) {
                                // This is the alignment bug we caught!
                                emit_warning!(
                                    "Layout mismatch: Rust {} vs WGSL {}",
                                    name, wgsl_name
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    
    /// Detect Python → WGSL boundaries (through Rust)
    pub fn detect_python_shader_boundaries(graph: &mut CodeGraph) {
        // Python calls Rust function that uses shader
        // Transitive edge: Python --PyO3Call--> Rust --UsesShader--> WGSL
    }
}
```

### Layer 5: Statechart Binding (superstate)

```rust
use superstate::{Statechart, Event, State};

/// The state machine for a code unit
#[derive(Statechart)]
pub struct CodeStateMachine {
    #[state]
    state: CodeState,
    
    #[history]
    last_known_state: Option<CodeState>,
}

#[derive(State, Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeState {
    // === INITIAL ===
    /// Never analyzed
    Unknown,
    /// Analyzed but never tested
    Untouched,
    
    // === TESTED ===
    /// Tests pass
    TestedGreen,
    /// Tests fail
    TestedRed,
    
    // === STALE ===
    /// Direct change to this unit
    StaleDirect,
    /// A dependency changed
    StaleTransitive,
    /// A dependency of a dependency changed
    StaleDeep,
    
    // === QA ===
    /// Passed adversarial review
    QaApproved,
    /// Under quarantine (failing, blocked)
    Quarantined,
}

#[derive(Event, Debug, Clone)]
pub enum CodeEvent {
    // === CHANGE EVENTS ===
    /// The source code changed
    SourceChanged,
    /// Only the signature changed (not body)
    SignatureChanged,
    /// Only the body changed (signature same)
    BodyChanged,
    /// Memory layout changed (for structs)
    LayoutChanged,
    
    // === DEPENDENCY EVENTS ===
    /// A direct dependency changed
    DependencyChanged,
    /// A transitive dependency changed
    TransitiveDependencyChanged,
    
    // === TEST EVENTS ===
    /// Tests were run and passed
    TestsPassed,
    /// Tests were run and failed
    TestsFailed,
    /// Tests were skipped
    TestsSkipped,
    
    // === QA EVENTS ===
    /// Adversarial review passed
    QaApproved,
    /// Flagged for quarantine
    Quarantine,
    /// Released from quarantine
    Release,
}

impl CodeStateMachine {
    pub fn transition(&mut self, event: CodeEvent) -> Option<CodeState> {
        use CodeState::*;
        use CodeEvent::*;
        
        let new_state = match (&self.state, event) {
            // Any state + change = stale
            (_, SourceChanged | SignatureChanged | BodyChanged | LayoutChanged) => {
                Some(StaleDirect)
            }
            
            // Green/Approved + dependency change = stale transitive
            (TestedGreen | QaApproved, DependencyChanged) => Some(StaleTransitive),
            (TestedGreen | QaApproved, TransitiveDependencyChanged) => Some(StaleDeep),
            
            // Already stale + more changes = stays stale
            (StaleDirect | StaleTransitive | StaleDeep, DependencyChanged) => None,
            
            // Run tests
            (_, TestsPassed) => Some(TestedGreen),
            (_, TestsFailed) => Some(TestedRed),
            
            // QA
            (TestedGreen, QaApproved) => Some(CodeState::QaApproved),
            (_, Quarantine) => Some(Quarantined),
            (Quarantined, Release) => {
                // Return to last known good state (history!)
                self.last_known_state.or(Some(Untouched))
            }
            
            _ => None,
        };
        
        if let Some(new) = new_state {
            if self.state == TestedGreen || self.state == CodeState::QaApproved {
                self.last_known_state = Some(self.state);
            }
            self.state = new;
        }
        
        new_state
    }
}
```

### Layer 6: Propagation Engine

```rust
pub struct PropagationEngine;

impl PropagationEngine {
    /// Propagate an event through the graph
    pub fn propagate(
        graph: &mut CodeGraph,
        origin: NodeId,
        event: CodeEvent,
    ) -> PropagationResult {
        let mut affected = Vec::new();
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        
        // Apply event to origin
        if let Some(new_state) = graph.nodes[origin].state.transition(event.clone()) {
            affected.push((origin, new_state));
        }
        
        // Determine propagation event
        let propagation_event = match &event {
            CodeEvent::SourceChanged | CodeEvent::SignatureChanged |
            CodeEvent::LayoutChanged => Some(CodeEvent::DependencyChanged),
            
            CodeEvent::BodyChanged => None, // Body-only changes don't propagate
            
            CodeEvent::DependencyChanged => Some(CodeEvent::TransitiveDependencyChanged),
            
            _ => None,
        };
        
        if let Some(prop_event) = propagation_event {
            // Seed queue with direct dependents
            for dependent in graph.dependents(origin) {
                queue.push_back((dependent, prop_event.clone(), 1));
            }
            
            // BFS propagation
            while let Some((node_id, event, depth)) = queue.pop_front() {
                if visited.contains(&node_id) {
                    continue;
                }
                visited.insert(node_id);
                
                // Apply event
                let node = &mut graph.nodes[node_id];
                if let Some(new_state) = node.state.transition(event.clone()) {
                    affected.push((node_id, new_state));
                    
                    // Continue propagation (transitive)
                    let next_event = if depth == 1 {
                        CodeEvent::TransitiveDependencyChanged
                    } else {
                        CodeEvent::TransitiveDependencyChanged
                    };
                    
                    for dependent in graph.dependents(node_id) {
                        queue.push_back((dependent, next_event.clone(), depth + 1));
                    }
                }
            }
        }
        
        // === SPECIAL: Layout changes propagate to GPU mirrors ===
        if matches!(event, CodeEvent::LayoutChanged) {
            for gpu_node in graph.gpu_mirrors(origin) {
                if let Some(new_state) = graph.nodes[gpu_node].state.transition(
                    CodeEvent::DependencyChanged
                ) {
                    affected.push((gpu_node, new_state));
                }
            }
        }
        
        PropagationResult { affected }
    }
}

pub struct PropagationResult {
    pub affected: Vec<(NodeId, CodeState)>,
}
```

### Layer 7: Query Interface

```rust
pub struct QueryEngine<'g> {
    graph: &'g CodeGraph,
}

impl<'g> QueryEngine<'g> {
    /// All nodes in a given state
    pub fn nodes_in_state(&self, state: CodeState) -> Vec<NodeId> {
        self.graph.nodes.iter()
            .filter(|(_, node)| node.state.state == state)
            .map(|(id, _)| id)
            .collect()
    }
    
    /// All stale nodes (any staleness)
    pub fn stale_nodes(&self) -> Vec<NodeId> {
        self.nodes_in_state(CodeState::StaleDirect)
            .into_iter()
            .chain(self.nodes_in_state(CodeState::StaleTransitive))
            .chain(self.nodes_in_state(CodeState::StaleDeep))
            .collect()
    }
    
    /// Nodes that need testing (stale or untested)
    pub fn needs_testing(&self) -> Vec<NodeId> {
        self.graph.nodes.iter()
            .filter(|(_, node)| matches!(
                node.state.state,
                CodeState::Untouched |
                CodeState::StaleDirect |
                CodeState::StaleTransitive |
                CodeState::StaleDeep
            ))
            .map(|(id, _)| id)
            .collect()
    }
    
    /// Nodes by language
    pub fn nodes_by_language(&self, lang: Language) -> impl Iterator<Item = NodeId> + '_ {
        self.graph.nodes_by_language.get(&lang)
            .into_iter()
            .flat_map(|ids| ids.iter().copied())
    }
    
    /// Layout mismatches between Rust and WGSL
    pub fn layout_mismatches(&self) -> Vec<(NodeId, NodeId, String)> {
        let mut mismatches = Vec::new();
        
        for (edge_id, edge) in &self.graph.edges {
            if edge.kind == EdgeKind::MirrorsLayout {
                let rust = &self.graph.nodes[edge.to];
                let wgsl = &self.graph.nodes[edge.from];
                
                if let (
                    CodeNodeKind::RustStruct { layout: rust_layout, .. },
                    CodeNodeKind::WgslStruct { layout: wgsl_layout, .. }
                ) = (&rust.kind, &wgsl.kind) {
                    if rust_layout.hash() != wgsl_layout.hash() {
                        mismatches.push((
                            edge.to,
                            edge.from,
                            diff_layouts(rust_layout, wgsl_layout),
                        ));
                    }
                }
            }
        }
        
        mismatches
    }
    
    /// Find test coverage for a node
    pub fn test_coverage(&self, node: NodeId) -> TestCoverage {
        let tests: Vec<_> = self.graph.tests_for(node).collect();
        let total = tests.len();
        let passing = tests.iter()
            .filter(|&&t| self.graph.nodes[t].state.state == CodeState::TestedGreen)
            .count();
        
        TestCoverage {
            tests,
            total,
            passing,
            failing: total - passing,
        }
    }
    
    /// Hierarchical query: all children of a module
    pub fn children(&self, node: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        self.graph.children.get(&node)
            .into_iter()
            .flat_map(|ids| ids.iter().copied())
    }
    
    /// Aggregate state for a parent (module, file, crate)
    pub fn aggregate_state(&self, parent: NodeId) -> AggregateState {
        let children: Vec<_> = self.children(parent).collect();
        
        let mut counts = HashMap::new();
        for child in &children {
            let state = self.graph.nodes[*child].state.state;
            *counts.entry(state).or_insert(0usize) += 1;
        }
        
        AggregateState {
            total: children.len(),
            by_state: counts,
            worst: self.worst_state(&children),
        }
    }
    
    fn worst_state(&self, nodes: &[NodeId]) -> CodeState {
        nodes.iter()
            .map(|&n| self.graph.nodes[n].state.state)
            .max_by_key(|s| s.severity())
            .unwrap_or(CodeState::Unknown)
    }
}

impl CodeState {
    fn severity(&self) -> u8 {
        match self {
            CodeState::TestedGreen => 0,
            CodeState::QaApproved => 0,
            CodeState::Untouched => 1,
            CodeState::Unknown => 2,
            CodeState::StaleDeep => 3,
            CodeState::StaleTransitive => 4,
            CodeState::StaleDirect => 5,
            CodeState::TestedRed => 6,
            CodeState::Quarantined => 7,
        }
    }
}
```

---

## Part 4: Full Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                              UNIFIED CODE SUBSTRATE                                  │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────┐ │
│  │                           FILE SYSTEM LAYER                                     │ │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐     │ │
│  │  │  *.rs    │   │  *.py    │   │  *.wgsl  │   │ *.toml   │   │  *.json  │     │ │
│  │  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘     │ │
│  │       │              │              │              │              │           │ │
│  │       ▼              ▼              ▼              ▼              ▼           │ │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐  │ │
│  │  │                         FILE WATCHER (notify)                           │  │ │
│  │  └────────────────────────────────┬────────────────────────────────────────┘  │ │
│  └───────────────────────────────────┼────────────────────────────────────────────┘ │
│                                      │                                              │
│  ┌───────────────────────────────────┼────────────────────────────────────────────┐ │
│  │                           AST PARSER LAYER                                      │ │
│  │                                   │                                             │ │
│  │       ┌───────────────────────────┼───────────────────────────┐                │ │
│  │       │                           │                           │                │ │
│  │       ▼                           ▼                           ▼                │ │
│  │  ┌─────────────┐            ┌─────────────┐            ┌─────────────┐         │ │
│  │  │ RUST PARSER │            │PYTHON PARSER│            │ WGSL PARSER │         │ │
│  │  │  syn +      │            │ rustpython  │            │  naga +     │         │ │
│  │  │ tree-sitter │            │ tree-sitter │            │ tree-sitter │         │ │
│  │  └──────┬──────┘            └──────┬──────┘            └──────┬──────┘         │ │
│  │         │                          │                          │                │ │
│  │         │    ┌────────────────┐    │    ┌────────────────┐    │                │ │
│  │         │    │ cargo metadata │    │    │pyproject.toml  │    │                │ │
│  │         │    └───────┬────────┘    │    └───────┬────────┘    │                │ │
│  │         │            │             │            │             │                │ │
│  │         ▼            ▼             ▼            ▼             ▼                │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────┐│ │
│  │  │                        UNIT EXTRACTION                                     ││ │
│  │  │   RustUnit[]          PythonUnit[]           WgslUnit[]                   ││ │
│  │  └───────────────────────────────────┬───────────────────────────────────────┘│ │
│  └──────────────────────────────────────┼────────────────────────────────────────┘ │
│                                         │                                          │
│  ┌──────────────────────────────────────┼────────────────────────────────────────┐ │
│  │                          GRAPH BUILDER LAYER                                   │ │
│  │                                      ▼                                         │ │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐  │ │
│  │  │                      UNIFIED CODE GRAPH                                  │  │ │
│  │  │                                                                          │  │ │
│  │  │   ┌─────────┐      Imports      ┌─────────┐     MirrorsLayout           │  │ │
│  │  │   │ Python  │─────────────────▶│  Rust   │◀──────────────────┐          │  │ │
│  │  │   │ Module  │     PyO3Call     │ Struct  │                   │          │  │ │
│  │  │   └─────────┘                  └────┬────┘              ┌────┴────┐     │  │ │
│  │  │        │                            │                   │  WGSL   │     │  │ │
│  │  │        │ Contains                   │ References        │ Struct  │     │  │ │
│  │  │        ▼                            ▼                   └─────────┘     │  │ │
│  │  │   ┌─────────┐                  ┌─────────┐                              │  │ │
│  │  │   │ Python  │                  │  Rust   │       UsesShader             │  │ │
│  │  │   │Function │                  │Function │──────────────────────────┐   │  │ │
│  │  │   └─────────┘                  └─────────┘                          │   │  │ │
│  │  │                                                                     ▼   │  │ │
│  │  │                                                              ┌─────────┐│  │ │
│  │  │                                                              │  WGSL   ││  │ │
│  │  │                                                              │EntryPt  ││  │ │
│  │  │                                                              └─────────┘│  │ │
│  │  └─────────────────────────────────────────────────────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────────────────────────┘ │
│                                         │                                          │
│  ┌──────────────────────────────────────┼────────────────────────────────────────┐ │
│  │                         STATECHART LAYER (superstate)                          │ │
│  │                                      │                                         │ │
│  │        Each node gets a CodeStateMachine instance                             │ │
│  │                                      │                                         │ │
│  │   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐        │ │
│  │   │ UNKNOWN │──▶│UNTOUCHED│──▶│ TESTED  │──▶│  STALE  │──▶│TESTED   │        │ │
│  │   │         │   │         │   │  GREEN  │   │ DIRECT  │   │  RED    │        │ │
│  │   └─────────┘   └─────────┘   └────┬────┘   └─────────┘   └─────────┘        │ │
│  │                                    │                                          │ │
│  │                                    ▼                                          │ │
│  │                              ┌───────────┐                                    │ │
│  │                              │QA_APPROVED│                                    │ │
│  │                              └───────────┘                                    │ │
│  └───────────────────────────────────────────────────────────────────────────────┘ │
│                                         │                                          │
│  ┌──────────────────────────────────────┼────────────────────────────────────────┐ │
│  │                        PROPAGATION ENGINE                                      │ │
│  │                                      │                                         │ │
│  │   Event at Node A ──▶ BFS traversal ──▶ Update dependents                     │ │
│  │                                                                                │ │
│  │   Layout change in Rust ──▶ Find WGSL mirrors ──▶ Mark STALE                  │ │
│  └───────────────────────────────────────────────────────────────────────────────┘ │
│                                         │                                          │
│  ┌──────────────────────────────────────┼────────────────────────────────────────┐ │
│  │                         QUERY INTERFACE                                        │ │
│  │                                      │                                         │ │
│  │   "Show me all STALE nodes"                                                   │ │
│  │   "Which Rust structs have WGSL mirrors?"                                     │ │
│  │   "What is the aggregate state of gpu_driven/?"                               │ │
│  │   "Find layout mismatches"                                                     │ │
│  │   "What needs testing?"                                                        │ │
│  └───────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 5: The Unified Code Unit

### The Duality Problem

Traditional model: Code and tests are separate artifacts that must be "kept in sync."

**But a coin doesn't need to sync its heads and tails.** They're aspects of the same object.

### The Solution: One Entity, Multiple Facets

```rust
/// A unified code entity with implementation, contract, and verification
struct CodeUnit {
    id: UnitId,
    name: String,
    
    // ══════════════════════════════════════════════════════════
    // IMPLEMENTATION FACET — what it does
    // ══════════════════════════════════════════════════════════
    
    /// Source code (or AST representation)
    source: Source,
    
    /// Dependencies this unit calls
    dependencies: Vec<UnitId>,
    
    /// Type signature
    signature: Signature,
    
    // ══════════════════════════════════════════════════════════
    // CONTRACT FACET — what it promises
    // ══════════════════════════════════════════════════════════
    
    /// Preconditions (what callers must provide)
    requires: Vec<Predicate>,
    
    /// Postconditions (what this promises to return)
    ensures: Vec<Predicate>,
    
    /// Invariants (what must always hold)
    invariants: Vec<Predicate>,
    
    /// Properties (algebraic laws it obeys)
    properties: Vec<Property>,
    
    // ══════════════════════════════════════════════════════════
    // STATE FACET — lifecycle tracking
    // ══════════════════════════════════════════════════════════
    
    /// Current lifecycle state
    state: CodeState,
    
    /// State machine for this unit's runtime behavior
    behavior: StateTree<InputEvent>,
    
    /// History of state transitions
    history: Vec<(Timestamp, CodeState, CodeEvent)>,
    
    // ══════════════════════════════════════════════════════════
    // VERIFICATION FACET — derived, not stored
    // ══════════════════════════════════════════════════════════
    // 
    // verification() — generated from contract
    // tests() — generated from properties
    // coverage() — computed from reachability
}
```

### Verification Is Derived, Not Written

```rust
impl CodeUnit {
    /// Verification is computed from the contract, not manually written
    fn verify(&self) -> VerificationResult {
        let mut results = vec![];
        
        // 1. Check preconditions are satisfiable
        for req in &self.requires {
            results.push(check_satisfiable(req));
        }
        
        // 2. Check postconditions follow from preconditions + implementation
        for ens in &self.ensures {
            results.push(prove_postcondition(&self.source, &self.requires, ens));
        }
        
        // 3. Check invariants hold across all reachable states
        let reachability = analyze_reachability(&self.behavior, None);
        for inv in &self.invariants {
            let inv_fn: Invariant = |cfg| inv.check(cfg);
            results.push(verify_invariant(inv_fn, &reachability));
        }
        
        // 4. Check properties via property-based testing
        // synth generates inputs satisfying `requires`
        for prop in &self.properties {
            results.push(prop.quickcheck(&self.source, 1000));
        }
        
        VerificationResult::aggregate(results)
    }
    
    /// "Tests" are just: does the implementation satisfy its contract?
    fn is_correct(&self) -> bool {
        self.verify().all_passed()
    }
}
```

### Example: The Old Way vs. The New Way

```rust
// ════════════════════════════════════════════════════════════════
// OLD WAY: Separate artifacts that drift
// ════════════════════════════════════════════════════════════════

// src/math.rs
fn add(a: i32, b: i32) -> i32 { a + b }

// tests/math_test.rs  (separate file, can forget to update)
#[test] 
fn test_add() { 
    assert_eq!(add(2, 2), 4); 
}
// Change add() to multiply, this test STILL PASSES (2*2 == 4)
// The test lies. The code drifted.

// ════════════════════════════════════════════════════════════════
// NEW WAY: Unified entity, verification is intrinsic
// ════════════════════════════════════════════════════════════════

let add = CodeUnit {
    name: "add".into(),
    source: Source::from(|a: i32, b: i32| a + b),
    
    // Contract
    requires: vec![],  // No preconditions
    ensures: vec![
        Predicate::Eq(Expr::Result, Expr::Add(Expr::Arg(0), Expr::Arg(1))),
    ],
    
    // Properties (algebraic laws)
    properties: vec![
        Property::Commutative,    // add(a,b) == add(b,a)
        Property::Associative,    // add(add(a,b),c) == add(a,add(b,c))
        Property::Identity(0),    // add(a,0) == a
    ],
    
    ..Default::default()
};

// Change implementation to multiply:
add.source = Source::from(|a: i32, b: i32| a * b);

add.verify();  // FAILS immediately:
               // - Identity(0) violated: a * 0 != a
               // - Postcondition violated: result != a + b
               // 
               // No separate test to "keep in sync"
               // The contract IS part of the unit
```

---

## Part 6: State Model Details

### Code States

```rust
/// The lifecycle state of a code unit
enum CodeState {
    // === Fresh States ===
    Untouched,              // New code, no tests ever
    Changed,                // Modified since last test run
    
    // === Test States ===
    TestedGreen,            // Tests passed
    TestedRed,              // Tests failed
    TestedSkipped,          // Tests exist but skipped (GPU, headless, etc.)
    
    // === Staleness States ===
    StaleDirect,            // This code changed, needs retest
    StaleTransitive,        // A dependency changed, needs retest
    StaleDeep,              // A dep of a dep changed (N hops away)
    
    // === QA States ===
    QaApproved,             // Passed adversarial review
    QaFlagged,              // Review found issues
    
    // === Meta States ===
    Deprecated,             // Marked for removal
    Quarantined,            // Known broken, isolated
}
```

### Events

```rust
/// Events that trigger state transitions
enum CodeEvent {
    // Source changes
    FileChanged(Path),
    FunctionChanged(FnId),
    SignatureChanged(FnId),     // Public API changed (more severe)
    
    // Test execution
    TestRan(TestId, Outcome),
    TestSkipped(TestId, Reason),
    
    // Dependency changes
    DependencyChanged(FnId),
    TransitiveDependencyChanged(FnId, Depth),
    
    // QA events
    QaPassed(FnId),
    QaFlagged(FnId, Finding),
    
    // Lifecycle events
    Deprecated(FnId),
    Quarantined(FnId),
    Restored(FnId),
    
    // Time-based
    StaleTimeout(Duration),     // Force retest after N days
}
```

### Transition Rules

```rust
// State machine transitions
impl CodeUnit {
    fn transition(&mut self, event: &CodeEvent) {
        self.state = match (&self.state, event) {
            // Any state + direct change → Changed
            (_, CodeEvent::FunctionChanged(id)) if *id == self.id => {
                CodeState::Changed
            }
            
            // Green + dependency change → Stale (transitive)
            (CodeState::TestedGreen, CodeEvent::DependencyChanged(_)) => {
                CodeState::StaleTransitive
            }
            
            // Changed/Stale + test pass → Green
            (CodeState::Changed | CodeState::StaleDirect | CodeState::StaleTransitive, 
             CodeEvent::TestRan(_, Outcome::Pass)) => {
                CodeState::TestedGreen
            }
            
            // Any + test fail → Red
            (_, CodeEvent::TestRan(_, Outcome::Fail)) => {
                CodeState::TestedRed
            }
            
            // Green + QA pass → Approved
            (CodeState::TestedGreen, CodeEvent::QaPassed(_)) => {
                CodeState::QaApproved
            }
            
            // Keep current state for unhandled transitions
            _ => self.state.clone(),
        };
        
        // Propagate to parent
        if let Some(parent) = &mut self.parent {
            parent.recompute_state();
        }
    }
}
```

### Propagation Rules

```rust
/// Upward propagation: children → parent
fn propagate_up(node: &mut CodeNode) {
    let child_states: Vec<_> = node.children.iter().map(|c| &c.state).collect();
    
    node.state = if child_states.iter().all(|s| **s == CodeState::TestedGreen) {
        CodeState::TestedGreen
    } else if child_states.iter().any(|s| **s == CodeState::TestedRed) {
        CodeState::TestedRed
    } else if child_states.iter().any(|s| matches!(s, CodeState::Changed | CodeState::StaleDirect)) {
        CodeState::StaleDirect
    } else if child_states.iter().any(|s| matches!(s, CodeState::StaleTransitive | CodeState::StaleDeep)) {
        CodeState::StaleTransitive
    } else {
        CodeState::Untouched
    };
    
    if let Some(parent) = &mut node.parent {
        propagate_up(parent);
    }
}

/// Downward propagation: parent → children (rare, specific events)
fn propagate_down(node: &mut CodeNode, event: &CodeEvent) {
    match event {
        CodeEvent::Quarantined(_) => {
            // Quarantine propagates down — all children are quarantined
            for child in &mut node.children {
                child.state = CodeState::Quarantined;
                propagate_down(child, event);
            }
        }
        _ => {
            // Most events don't propagate down
        }
    }
}
```

### Smart Staleness

Not all changes create equal staleness:

```rust
struct Change {
    unit: UnitId,
    kind: ChangeKind,
    semantic_hash: u64,  // Hash of PUBLIC interface
}

enum ChangeKind {
    Internal,           // Body changed, signature same
    SignatureChanged,   // Public API changed
    Removed,            // Unit deleted
    Added,              // New unit
}
```

**Rule:** If only `Internal` changed, dependents DON'T go stale. Their tests would still pass with the new implementation (assuming tests are behavioral, not white-box).

This dramatically reduces cascading staleness.

---

## Part 7: Integration with Existing Tools

### superstate

superstate provides the statechart engine:

```rust
// Model the code hierarchy as a statechart
let tree = StateTreeBuilder::<CodeEvent>::new("crate")
    .root("crate", StateType::And)  // Parallel modules
        .child("gpu_driven", StateType::And)  // Parallel files
            .child("build_indirect", StateType::And)  // Parallel functions
                .basic_child("cpu_build_indirect").end()
                .basic_child("select_lod").end()
            .end()
        .end()
    .done()
    .build();

// Track state with ConfigBitset — one bit per node
let mut config = tree.initial_config();

// Step through events
let result = step(&mut config, &CodeEvent::FileChanged(path), &tree, ...);

// Verify properties
let prop = CTLProperty::AG(
    Predicate::Implies(
        Predicate::InProduction(unit),
        Predicate::StateIs(unit, CodeState::QaApproved)
    )
);
verify_ctl(&prop, &reachability);
```

### synth

synth generates test inputs from contracts:

```rust
// Define a schema matching the function's input type
let input_schema = synth::Schema::Object {
    a: synth::Schema::Number { range: i32::MIN..i32::MAX },
    b: synth::Schema::Number { range: i32::MIN..i32::MAX },
};

// Add constraints from `requires`
let constraints = unit.requires.iter()
    .map(|r| r.to_synth_constraint())
    .collect();

// Generate inputs that satisfy preconditions
let inputs = synth::generate(input_schema, constraints, 1000);

// Check postconditions hold for all generated inputs
for input in inputs {
    let result = (unit.source)(input.a, input.b);
    for ens in &unit.ensures {
        assert!(ens.check(&input, &result));
    }
}
```

### The SDLC Workflow Connection

The `SDLC_WORKFLOW` from superstate's workflows/ already models:

```
DEV → TEST_UNIT → QA_UNIT → GREEN_LIGHT
```

With unified code units, this becomes:

```
CHANGED → DEV_REVIEWED → VERIFIED → QA_APPROVED → GREEN_LIGHT

Where:
- CHANGED: CodeState::Changed
- DEV_REVIEWED: Human reviewed, not yet tested
- VERIFIED: unit.verify() passed (contract holds)
- QA_APPROVED: CodeState::QaApproved (adversarial review)
- GREEN_LIGHT: Ready for production
```

The workflow isn't separate from the code — it's embedded in the code graph.

---

## Part 8: Verification as First-Class

### What superstate Provides

| Capability | Application |
|------------|-------------|
| **Reachability** | Which states are reachable from initial? |
| **Deadlock detection** | Are there stuck states? |
| **Invariant checking** | Does property hold in all states? |
| **CTL model checking** | AG, EF, AF, EG temporal properties |
| **History tracking** | State restoration after interruption |

### Applying to Code

```rust
// "No code reaches production without being GREEN"
let no_untested_production = CTLProperty::AG(
    Predicate::Implies(
        Predicate::InProduction,
        Predicate::Or(
            Predicate::StateIs(CodeState::TestedGreen),
            Predicate::StateIs(CodeState::QaApproved),
        )
    )
);

// "Every CHANGED state eventually reaches TESTED"
let liveness = CTLProperty::AF(
    Predicate::Not(Predicate::StateIs(CodeState::Changed))
);

// "STALE never persists across releases"
let no_stale_release = CTLProperty::AG(
    Predicate::Implies(
        Predicate::Event(CodeEvent::Release),
        Predicate::Not(Predicate::StateIs(CodeState::StaleTransitive))
    )
);
```

### Coverage Becomes State Coverage

Traditional: "80% of lines executed"
New: "100% of reachable states verified"

```rust
let reachability = analyze_reachability(&unit.behavior, None);
let stats = compute_stats(&reachability);

println!("Reachable configurations: {}", stats.reachable_count);
println!("Unreachable states: {}", stats.unreachable_state_count);
println!("Deadlocks: {}", stats.deadlock_count);

// Coverage = (verified_states / reachable_states) * 100
```

---

## Part 9: Dependencies

```toml
[dependencies]
# AST Parsing
syn = { version = "2", features = ["full", "parsing"] }
rustpython-parser = "0.3"
naga = { version = "0.19", features = ["wgsl-in"] }
tree-sitter = "0.22"
tree-sitter-rust = "0.21"
tree-sitter-python = "0.21"
tree-sitter-wgsl = "0.1"  # May need custom grammar

# Graph
petgraph = "0.6"
slotmap = "1.0"

# File System
notify = "6.0"
walkdir = "2.4"

# Hashing
sha2 = "0.10"
blake3 = "1.5"

# State Machine (our library)
superstate = { path = "../HARNESS/superstate" }

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Cargo integration
cargo_metadata = "0.18"
```

---

## Part 10: The Philosophical Shift

### Old World

| Aspect | Old Thinking |
|--------|--------------|
| Code and tests | Separate artifacts, manually synced |
| Testing | Activity done after coding |
| Coverage | Lines executed ÷ total lines |
| "Done" | Tests pass, human says so |
| Staleness | Not tracked |
| Dependencies | Implicit, discovered at runtime |

### New World

| Aspect | New Thinking |
|--------|--------------|
| Code and tests | Two facets of one entity |
| Testing | Intrinsic property of code |
| Coverage | States verified ÷ reachable states |
| "Done" | All invariants hold in all states |
| Staleness | Tracked, propagated, verified |
| Dependencies | Explicit, part of state graph |

### The Fundamental Insight

> "The test isn't a separate thing you forgot to update. The contract is part of the unit, and verification is automatic."

This is what Harel was getting at with statecharts — the system IS the specification. The specification isn't a separate document that describes the system. It's the system itself, viewed formally.

---

## Part 11: Implementation Roadmap

### Phase 1: Foundation

- [ ] Define `CodeUnit` struct with all facets
- [ ] Implement state enum and transition rules
- [ ] Create parser to extract code graph from Rust source
- [ ] Integrate with git for change detection

### Phase 2: Multi-Language AST

- [ ] Implement RustParser (syn + tree-sitter)
- [ ] Implement PythonParser (rustpython + tree-sitter)
- [ ] Implement WgslParser (naga + tree-sitter)
- [ ] Unify into CodeGraph with language-agnostic nodes

### Phase 3: State Tracking

- [ ] Implement upward state propagation
- [ ] Implement selective downward propagation
- [ ] Build ConfigBitset-based state storage
- [ ] Create event stream from git commits

### Phase 4: Cross-Language Boundaries

- [ ] Detect PyO3 Rust↔Python boundaries
- [ ] Detect Rust↔WGSL shader boundaries
- [ ] Implement MirrorsLayout edge detection
- [ ] Add layout mismatch warnings (catches alignment bugs!)

### Phase 5: superstate Integration

- [ ] Model code hierarchy as StateTree
- [ ] Implement step() for code state transitions
- [ ] Add reachability analysis for code graph
- [ ] Implement invariant verification

### Phase 6: synth Integration

- [ ] Generate input schemas from function signatures
- [ ] Convert `requires` predicates to synth constraints
- [ ] Generate test inputs automatically
- [ ] Check `ensures` predicates against outputs

### Phase 7: Query & Visualization

- [ ] Implement QueryEngine with all queries
- [ ] Build codebase state visualizer
- [ ] Color-code files/functions by state
- [ ] Show dependency graph with state propagation
- [ ] Integration with IDE (VS Code extension?)

### Phase 8: Workflow Integration

- [ ] Connect to SDLC_WORKFLOW
- [ ] Automatic state transitions on workflow events
- [ ] GREEN_LIGHT requires all states verified
- [ ] Release blocking on STALE code

---

## Appendix A: Relationship to Other Documents

| Document | Relationship |
|----------|--------------|
| `V1_ADVERSARIAL_REVIEW.md` | QA campaigns become state transitions |
| `V1_IMPROVEMENT_CAMPAIGNS.md` | Improvement passes operate on state graph |
| `V1_TESTING_TOOLS.md` | Tools feed into this unified system |
| superstate `/workflows/SDLC/` | Workflow becomes embedded in code state |
| synth | Generates inputs from contracts |

---

## Appendix B: Key Equations

```
Code State = f(Implementation × Contract × History)

Verification = ∀ state ∈ Reachable : Invariants(state) = true

Coverage = |Verified States| / |Reachable States|

Staleness Propagation:
  changed(A) ∧ depends(B, A) → stale(B)
  changed(A) ∧ depends(B, A) ∧ depends(C, B) → stale_deep(C)

GREEN_LIGHT = ∀ unit ∈ Release : state(unit) ∈ {TestedGreen, QaApproved}
```

---

## Appendix C: Open Questions

1. **Granularity of change detection** — AST diff? Semantic diff? Line diff?
2. **Contract language** — Predicate DSL? Rust macros? Separate file?
3. **Integration with existing tests** — How to migrate? Coexistence?
4. **Performance at scale** — 100K functions × 100 events/day?
5. **IDE integration** — Real-time state updates? Inline visualization?
6. **Distributed systems** — Code state across microservices?

---

**The code IS the state machine. The test IS the verification of contract against implementation. There is nothing to sync because there is only one thing.**
