//! WGSL shader parser using naga.

use super::{CodeUnit, ContentHashes, Language, UnitType};

/// Parser for WGSL shader source code.
pub struct WgslParser {
    _private: (),
}

impl WgslParser {
    /// Create a new WGSL parser.
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Parse WGSL source and extract code units.
    pub fn parse(&self, source: &str) -> Vec<CodeUnit> {
        let mut units = Vec::new();

        let result = naga::front::wgsl::parse_str(source);
        if let Ok(module) = result {
            for (handle, ty) in module.types.iter() {
                if let Some(name) = &ty.name {
                    if matches!(ty.inner, naga::TypeInner::Struct { .. }) {
                        units.push(CodeUnit {
                            unit_type: UnitType::Struct,
                            name: name.clone(),
                            start_line: 0,
                            end_line: 0,
                            language: Language::Wgsl,
                            hashes: ContentHashes::default(),
                        });
                    }
                }
                let _ = handle;
            }

            for (_, func) in module.functions.iter() {
                if let Some(name) = &func.name {
                    units.push(CodeUnit {
                        unit_type: UnitType::Function,
                        name: name.clone(),
                        start_line: 0,
                        end_line: 0,
                        language: Language::Wgsl,
                        hashes: ContentHashes::default(),
                    });
                }
            }

            for ep in module.entry_points.iter() {
                units.push(CodeUnit {
                    unit_type: UnitType::Function,
                    name: ep.name.clone(),
                    start_line: 0,
                    end_line: 0,
                    language: Language::Wgsl,
                    hashes: ContentHashes::default(),
                });
            }
        }

        units
    }
}

impl Default for WgslParser {
    fn default() -> Self {
        Self::new()
    }
}
