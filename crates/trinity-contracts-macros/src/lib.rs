//! Procedural macros for trinity-contracts.
//!
//! Provides `#[contract]` attribute macro with support for:
//! - Outer attributes: `#[requires(...)]`, `#[ensures(...)]`
//! - Inner attributes: `#![requires(...)]`, `#![ensures(...)]`
//! - Function signature parsing

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{parse_macro_input, Attribute, Expr, FnArg, ItemFn, Pat, ReturnType, Type};

/// Parsed contract information.
#[derive(Default)]
struct ContractInfo {
    /// Preconditions (requires).
    requires: Vec<Expr>,
    /// Postconditions (ensures).
    ensures: Vec<Expr>,
    /// Invariants.
    invariants: Vec<Expr>,
    /// Parameter names.
    param_names: Vec<String>,
    /// Parameter types.
    param_types: Vec<Type>,
    /// Return type (if any).
    return_type: Option<Type>,
    /// Function name.
    func_name: String,
}

/// Contract attribute macro.
///
/// Adds runtime checks for preconditions and postconditions.
///
/// # Outer Attributes
///
/// ```ignore
/// #[contract]
/// #[requires(x > 0)]
/// #[ensures(result > x)]
/// fn double(x: i32) -> i32 {
///     x * 2
/// }
/// ```
///
/// # Inner Attributes
///
/// ```ignore
/// #[contract]
/// fn divide(a: i32, b: i32) -> i32 {
///     #![requires(b != 0)]
///     #![ensures(*result == a / b)]
///     a / b
/// }
/// ```
#[proc_macro_attribute]
pub fn contract(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);

    match expand_contract(input) {
        Ok(expanded) => TokenStream::from(expanded),
        Err(e) => TokenStream::from(e.to_compile_error()),
    }
}

fn expand_contract(mut func: ItemFn) -> syn::Result<TokenStream2> {
    // Extract contract info from function
    let mut info = extract_contract_info(&func);

    // Extract outer attributes
    let (outer_requires, outer_ensures, other_attrs) = extract_outer_attrs(&func.attrs);
    info.requires.extend(outer_requires);
    info.ensures.extend(outer_ensures);

    // Extract inner attributes from function body
    let (inner_requires, inner_ensures, inner_invariants) = extract_inner_attrs(&func.block);
    info.requires.extend(inner_requires);
    info.ensures.extend(inner_ensures);
    info.invariants.extend(inner_invariants);

    // Update function attributes (remove contract-related ones)
    func.attrs = other_attrs;

    // Remove inner attributes from the block
    let cleaned_block = remove_inner_attrs(&func.block);

    // Generate the expanded function
    let expanded = generate_contracted_function(&func, &info, &cleaned_block);

    Ok(expanded)
}

fn extract_contract_info(func: &ItemFn) -> ContractInfo {
    let mut info = ContractInfo::default();

    // Extract function name
    info.func_name = func.sig.ident.to_string();

    // Extract parameter names and types
    for arg in &func.sig.inputs {
        if let FnArg::Typed(pat_type) = arg {
            if let Pat::Ident(pat_ident) = &*pat_type.pat {
                info.param_names.push(pat_ident.ident.to_string());
                info.param_types.push((*pat_type.ty).clone());
            }
        }
    }

    // Extract return type
    if let ReturnType::Type(_, ty) = &func.sig.output {
        info.return_type = Some((**ty).clone());
    }

    info
}

fn extract_outer_attrs(attrs: &[Attribute]) -> (Vec<Expr>, Vec<Expr>, Vec<Attribute>) {
    let mut requires = Vec::new();
    let mut ensures = Vec::new();
    let mut other = Vec::new();

    for attr in attrs {
        if attr.path().is_ident("requires") {
            if let Ok(expr) = attr.parse_args::<Expr>() {
                requires.push(expr);
            }
        } else if attr.path().is_ident("ensures") {
            if let Ok(expr) = attr.parse_args::<Expr>() {
                ensures.push(expr);
            }
        } else if attr.path().is_ident("invariant") {
            // Skip for now, handled separately
        } else {
            other.push(attr.clone());
        }
    }

    (requires, ensures, other)
}

fn extract_inner_attrs(block: &syn::Block) -> (Vec<Expr>, Vec<Expr>, Vec<Expr>) {
    let mut requires = Vec::new();
    let mut ensures = Vec::new();
    let mut invariants = Vec::new();

    // Inner attributes in the block start with #!
    // They appear as stmt attributes in syn
    for stmt in &block.stmts {
        if let syn::Stmt::Item(syn::Item::Verbatim(tokens)) = stmt {
            // Parse inner attribute pattern: #![attr(...)]
            let s = tokens.to_string();
            if s.starts_with("#!") {
                if let Some(expr_str) = extract_inner_attr_expr(&s, "requires") {
                    if let Ok(expr) = syn::parse_str::<Expr>(&expr_str) {
                        requires.push(expr);
                    }
                } else if let Some(expr_str) = extract_inner_attr_expr(&s, "ensures") {
                    if let Ok(expr) = syn::parse_str::<Expr>(&expr_str) {
                        ensures.push(expr);
                    }
                } else if let Some(expr_str) = extract_inner_attr_expr(&s, "invariant") {
                    if let Ok(expr) = syn::parse_str::<Expr>(&expr_str) {
                        invariants.push(expr);
                    }
                }
            }
        }
    }

    (requires, ensures, invariants)
}

fn extract_inner_attr_expr(s: &str, attr_name: &str) -> Option<String> {
    let pattern = format!("#![{}(", attr_name);
    if s.starts_with(&pattern) {
        // Find matching closing paren
        let start = pattern.len();
        let end = s.len() - 2; // Skip )]
        if end > start {
            return Some(s[start..end].to_string());
        }
    }
    None
}

fn remove_inner_attrs(block: &syn::Block) -> syn::Block {
    let mut cleaned_stmts = Vec::new();

    for stmt in &block.stmts {
        // Skip inner attribute statements
        if let syn::Stmt::Item(syn::Item::Verbatim(tokens)) = stmt {
            let s = tokens.to_string();
            if s.starts_with("#![requires") || s.starts_with("#![ensures") || s.starts_with("#![invariant") {
                continue;
            }
        }
        cleaned_stmts.push(stmt.clone());
    }

    syn::Block {
        brace_token: block.brace_token,
        stmts: cleaned_stmts,
    }
}

fn generate_contracted_function(
    func: &ItemFn,
    info: &ContractInfo,
    cleaned_block: &syn::Block,
) -> TokenStream2 {
    let vis = &func.vis;
    let sig = &func.sig;
    let attrs = &func.attrs;

    // Generate precondition checks
    let precondition_checks: Vec<_> = info.requires.iter().map(|req| {
        quote! {
            debug_assert!(#req, "Precondition violated: {}", stringify!(#req));
        }
    }).collect();

    // Check if function has a return value
    let has_return = info.return_type.is_some();

    let body = if has_return && !info.ensures.is_empty() {
        // Generate postcondition checks with result binding
        let postcondition_checks: Vec<_> = info.ensures.iter().map(|ens| {
            quote! {
                debug_assert!({
                    let result = &__contract_result;
                    #ens
                }, "Postcondition violated: {}", stringify!(#ens));
            }
        }).collect();

        quote! {
            #(#precondition_checks)*
            let __contract_result = (|| #cleaned_block)();
            #(#postcondition_checks)*
            __contract_result
        }
    } else {
        quote! {
            #(#precondition_checks)*
            #cleaned_block
        }
    };

    quote! {
        #(#attrs)*
        #vis #sig {
            #body
        }
    }
}

/// Layout attribute for struct alignment checks.
#[proc_macro_attribute]
pub fn layout(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Full implementation in T-CONT-6.6
    item
}

/// Property attribute for algebraic properties.
#[proc_macro_attribute]
pub fn property(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Full implementation in T-CONT-6.7
    item
}

/// Invariant attribute for class/struct invariants.
#[proc_macro_attribute]
pub fn invariant(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Placeholder - invariants on types
    item
}
