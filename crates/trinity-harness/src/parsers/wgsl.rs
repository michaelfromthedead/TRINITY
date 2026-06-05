//! WGSL shader parser using naga.

use super::{CodeUnit, ContentHashes, Language, UnitType};
use naga::{Module, ShaderStage, TypeInner};

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
            self.extract_structs(source, &module, &mut units);
            self.extract_functions(source, &module, &mut units);
            self.extract_entry_points(source, &module, &mut units);
        }

        units
    }

    /// Extract struct definitions with full member layout info.
    fn extract_structs(&self, source: &str, module: &Module, units: &mut Vec<CodeUnit>) {
        for (_handle, ty) in module.types.iter() {
            if let Some(name) = &ty.name {
                if let TypeInner::Struct { members, span: _ } = &ty.inner {
                    let (start_line, end_line) = self.find_struct_lines(source, name);
                    let hashes = self.compute_struct_hashes(source, name, members);
                    units.push(CodeUnit {
                        unit_type: UnitType::Struct,
                        name: name.clone(),
                        start_line,
                        end_line,
                        language: Language::Wgsl,
                        hashes,
                    });
                }
            }
        }
    }

    /// Extract regular (non-entry-point) functions.
    fn extract_functions(&self, source: &str, module: &Module, units: &mut Vec<CodeUnit>) {
        for (_handle, func) in module.functions.iter() {
            if let Some(name) = &func.name {
                let (start_line, end_line) = self.find_function_lines(source, name);
                let hashes = self.compute_function_hashes(source, name, func);
                units.push(CodeUnit {
                    unit_type: UnitType::Function,
                    name: name.clone(),
                    start_line,
                    end_line,
                    language: Language::Wgsl,
                    hashes,
                });
            }
        }
    }

    /// Extract entry point functions (@vertex, @fragment, @compute).
    fn extract_entry_points(&self, source: &str, module: &Module, units: &mut Vec<CodeUnit>) {
        for ep in module.entry_points.iter() {
            let (start_line, end_line) = self.find_entry_point_lines(source, &ep.name, ep.stage);
            let hashes = self.compute_entry_point_hashes(source, &ep.name, &ep.function, ep.stage);
            units.push(CodeUnit {
                unit_type: UnitType::Function,
                name: ep.name.clone(),
                start_line,
                end_line,
                language: Language::Wgsl,
                hashes,
            });
        }
    }

    /// Compute hashes for a struct, including layout_hash with member offsets.
    fn compute_struct_hashes(
        &self,
        source: &str,
        name: &str,
        members: &[naga::StructMember],
    ) -> ContentHashes {
        let full_text = self.extract_struct_text(source, name);

        // Layout hash includes member names, types (as indices), and OFFSETS.
        // This is critical for detecting Rust/WGSL alignment mismatches.
        let mut layout_parts = Vec::new();
        for member in members {
            let member_name = member.name.as_deref().unwrap_or("_");
            // Include offset and type handle in layout representation
            layout_parts.push(format!(
                "{}@{}:ty{}",
                member_name,
                member.offset,
                member.ty.index()
            ));
        }
        let layout_text = layout_parts.join(",");

        // Signature hash is just the struct name
        let sig_text = name;

        ContentHashes {
            full_hash: blake3::hash(full_text.as_bytes()).into(),
            signature_hash: blake3::hash(sig_text.as_bytes()).into(),
            body_hash: [0u8; 32], // Not applicable for structs
            layout_hash: blake3::hash(layout_text.as_bytes()).into(),
        }
    }

    /// Compute hashes for a regular function.
    fn compute_function_hashes(
        &self,
        source: &str,
        name: &str,
        _func: &naga::Function,
    ) -> ContentHashes {
        let full_text = self.extract_function_text(source, name);

        // For signature, we use name + argument count (naga doesn't give us source-level signatures easily)
        let sig_text = name;

        ContentHashes {
            full_hash: blake3::hash(full_text.as_bytes()).into(),
            signature_hash: blake3::hash(sig_text.as_bytes()).into(),
            body_hash: blake3::hash(full_text.as_bytes()).into(), // Use full as body approximation
            layout_hash: [0u8; 32], // Not applicable for functions
        }
    }

    /// Compute hashes for an entry point function.
    fn compute_entry_point_hashes(
        &self,
        source: &str,
        name: &str,
        _func: &naga::Function,
        stage: ShaderStage,
    ) -> ContentHashes {
        let full_text = self.extract_entry_point_text(source, name, stage);

        // Include stage in signature for entry points
        let stage_str = match stage {
            ShaderStage::Vertex => "vertex",
            ShaderStage::Fragment => "fragment",
            ShaderStage::Compute => "compute",
        };
        let sig_text = format!("@{} fn {}", stage_str, name);

        ContentHashes {
            full_hash: blake3::hash(full_text.as_bytes()).into(),
            signature_hash: blake3::hash(sig_text.as_bytes()).into(),
            body_hash: blake3::hash(full_text.as_bytes()).into(),
            layout_hash: [0u8; 32], // Not applicable for functions
        }
    }

    /// Find line range for a struct definition in source.
    fn find_struct_lines(&self, source: &str, name: &str) -> (usize, usize) {
        let pattern = format!("struct {}", name);
        self.find_block_lines(source, &pattern)
    }

    /// Find line range for a regular function in source.
    fn find_function_lines(&self, source: &str, name: &str) -> (usize, usize) {
        // Look for "fn name" that is NOT preceded by @vertex/@fragment/@compute
        for (i, line) in source.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.contains(&format!("fn {}", name)) && !trimmed.starts_with('@') {
                // Check previous line for stage attribute
                if i > 0 {
                    let prev_line = source.lines().nth(i - 1).unwrap_or("").trim();
                    if prev_line.starts_with("@vertex")
                        || prev_line.starts_with("@fragment")
                        || prev_line.starts_with("@compute")
                    {
                        continue; // This is an entry point, skip
                    }
                }
                let start_line = i + 1;
                let end_line = self.find_closing_brace_line(source, start_line);
                return (start_line, end_line);
            }
        }
        (0, 0)
    }

    /// Find line range for an entry point function.
    fn find_entry_point_lines(
        &self,
        source: &str,
        name: &str,
        stage: ShaderStage,
    ) -> (usize, usize) {
        let stage_attr = match stage {
            ShaderStage::Vertex => "@vertex",
            ShaderStage::Fragment => "@fragment",
            ShaderStage::Compute => "@compute",
        };

        let lines: Vec<&str> = source.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if trimmed.contains(stage_attr) {
                // Look for the function on this line or the next
                if trimmed.contains(&format!("fn {}", name)) {
                    let start_line = i + 1;
                    let end_line = self.find_closing_brace_line(source, start_line);
                    return (start_line, end_line);
                } else if i + 1 < lines.len() && lines[i + 1].contains(&format!("fn {}", name)) {
                    let start_line = i + 1; // Attribute line
                    let end_line = self.find_closing_brace_line(source, i + 2);
                    return (start_line, end_line);
                }
            }
        }
        (0, 0)
    }

    /// Find start and end lines for a block (struct or function).
    fn find_block_lines(&self, source: &str, pattern: &str) -> (usize, usize) {
        for (i, line) in source.lines().enumerate() {
            if line.contains(pattern) {
                let start_line = i + 1;
                let end_line = self.find_closing_brace_line(source, start_line);
                return (start_line, end_line);
            }
        }
        (0, 0)
    }

    /// Find the line number of the closing brace for a block starting at given line.
    fn find_closing_brace_line(&self, source: &str, start_line: usize) -> usize {
        let lines: Vec<&str> = source.lines().collect();
        let mut brace_count = 0;
        let mut found_open = false;

        for (i, line) in lines.iter().enumerate().skip(start_line - 1) {
            for ch in line.chars() {
                if ch == '{' {
                    brace_count += 1;
                    found_open = true;
                } else if ch == '}' {
                    brace_count -= 1;
                    if found_open && brace_count == 0 {
                        return i + 1;
                    }
                }
            }
        }

        // Fallback: return start line if no closing brace found
        start_line
    }

    /// Extract source text for a struct.
    fn extract_struct_text(&self, source: &str, name: &str) -> String {
        let (start, end) = self.find_struct_lines(source, name);
        self.extract_lines(source, start, end)
    }

    /// Extract source text for a regular function.
    fn extract_function_text(&self, source: &str, name: &str) -> String {
        let (start, end) = self.find_function_lines(source, name);
        self.extract_lines(source, start, end)
    }

    /// Extract source text for an entry point.
    fn extract_entry_point_text(&self, source: &str, name: &str, stage: ShaderStage) -> String {
        let (start, end) = self.find_entry_point_lines(source, name, stage);
        self.extract_lines(source, start, end)
    }

    /// Extract lines from source (1-indexed, inclusive).
    fn extract_lines(&self, source: &str, start: usize, end: usize) -> String {
        if start == 0 || end == 0 || start > end {
            return String::new();
        }

        source
            .lines()
            .skip(start - 1)
            .take(end - start + 1)
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl Default for WgslParser {
    fn default() -> Self {
        Self::new()
    }
}
