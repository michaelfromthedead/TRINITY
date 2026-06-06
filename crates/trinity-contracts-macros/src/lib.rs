//! Procedural macros for trinity-contracts.
//!
//! Provides `#[contract]` attribute macro.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens};
use syn::{parse_macro_input, ItemFn, Attribute, Meta, Expr};

/// Contract attribute macro.
///
/// Adds runtime checks for preconditions and postconditions.
///
/// # Example
///
/// ```ignore
/// #[contract]
/// #[requires(x > 0)]
/// #[ensures(result > x)]
/// fn double(x: i32) -> i32 {
///     x * 2
/// }
/// ```
#[proc_macro_attribute]
pub fn contract(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    
    let expanded = expand_contract(input);
    
    TokenStream::from(expanded)
}

fn expand_contract(mut func: ItemFn) -> TokenStream2 {
    let func_name = &func.sig.ident;
    let vis = &func.vis;
    let sig = &func.sig;
    let block = &func.block;
    
    // Extract contract attributes
    let (requires, ensures, other_attrs) = extract_contract_attrs(&func.attrs);
    
    // Update function attributes
    func.attrs = other_attrs;
    
    // Generate precondition checks
    let precondition_checks = requires.iter().map(|req| {
        quote! {
            debug_assert!(#req, "Precondition violated: {}", stringify!(#req));
        }
    });
    
    // Generate postcondition checks (for functions with return values)
    let has_return = !matches!(func.sig.output, syn::ReturnType::Default);
    
    let body = if has_return && !ensures.is_empty() {
        let postcondition_checks = ensures.iter().map(|ens| {
            quote! {
                debug_assert!({
                    let result = &__contract_result;
                    #ens
                }, "Postcondition violated: {}", stringify!(#ens));
            }
        });
        
        quote! {
            #(#precondition_checks)*
            let __contract_result = (|| #block)();
            #(#postcondition_checks)*
            __contract_result
        }
    } else {
        quote! {
            #(#precondition_checks)*
            #block
        }
    };
    
    let attrs = &func.attrs;
    
    quote! {
        #(#attrs)*
        #vis #sig {
            #body
        }
    }
}

fn extract_contract_attrs(attrs: &[Attribute]) -> (Vec<Expr>, Vec<Expr>, Vec<Attribute>) {
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
        } else {
            other.push(attr.clone());
        }
    }
    
    (requires, ensures, other)
}

/// Layout attribute for struct alignment checks.
#[proc_macro_attribute]
pub fn layout(attr: TokenStream, item: TokenStream) -> TokenStream {
    // For now, pass through unchanged
    // Full implementation in T-CONT-6.6
    item
}

/// Property attribute for algebraic properties.
#[proc_macro_attribute]
pub fn property(attr: TokenStream, item: TokenStream) -> TokenStream {
    // For now, pass through unchanged
    // Full implementation in T-CONT-6.7
    item
}
