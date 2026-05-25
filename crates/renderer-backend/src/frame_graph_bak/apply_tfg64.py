#!/usr/bin/env python3
"""Apply T-FG-6.4 Dynamic Culling changes to mod.rs and python.rs."""

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MOD_RS = os.path.join(SCRIPT_DIR, "mod.rs")
PY_RS = os.path.join(SCRIPT_DIR, "python.rs")

# ============================================================
# 1. FeatureSet type definition (insert before IrPass section)
# ============================================================
FEATURESET_DEF = """
// ---------------------------------------------------------------------------
// FeatureSet — runtime feature-flag bitfield for dynamic pass culling (T-FG-6.4)
// ---------------------------------------------------------------------------

/// A bitfield of frame-level feature flags used for dynamic pass culling.
///
/// At execution time, every pass whose [`IrPass::feature_flags`] is non-zero
/// is checked against the current frame's `FeatureSet`.  If the pass requires
/// any flag that is not set, the pass is **culled** (skipped during execution).
///
/// Culled passes remain in the compiled graph — they are simply not submitted
/// to the GPU queue.
///
/// # Bit assignments
///
/// | Bit | Constant            | Typical usage              |
/// |-----|----------------------|----------------------------|
/// | 0   | `DEBUG_WIREFRAME`    | Wireframe overlay pass     |
/// | 1   | `DEBUG_OVERLAY`      | Debug HUD / on-screen stats|
/// | 2   | `DEBUG_PROFILER`     | GPU profiler annotation    |
/// | 3.. | (reserved)           | Future debug / feature bits|
///
/// # Examples
///
/// ```
/// use frame_graph::FeatureSet;
///
/// let features = FeatureSet::DEBUG_WIREFRAME | FeatureSet::DEBUG_OVERLAY;
/// assert!(features.contains(FeatureSet::DEBUG_WIREFRAME));
/// assert!(!features.contains(FeatureSet::DEBUG_PROFILER));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct FeatureSet(pub u64);

impl FeatureSet {
    /// No feature flags set (production / release profile).
    pub const NONE: Self = Self(0);

    /// All debugging-related flags enabled.
    pub const ALL_DEBUG: Self = Self(
        Self::DEBUG_WIREFRAME.0
            | Self::DEBUG_OVERLAY.0
            | Self::DEBUG_PROFILER.0
    );

    /// Wireframe / triangle-overlay debug pass.
    pub const DEBUG_WIREFRAME: Self = Self(1 << 0);
    /// On-screen debug overlay (HUD, stats counters).
    pub const DEBUG_OVERLAY: Self = Self(1 << 1);
    /// GPU profiler annotation / timestamp query pass.
    pub const DEBUG_PROFILER: Self = Self(1 << 2);

    /// Returns `true` if all of the given `flags` are set in `self`.
    #[inline]
    pub const fn contains(self, flags: FeatureSet) -> bool {
        self.0 & flags.0 == flags.0
    }

    /// Returns `true` when no flags are set.
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl std::ops::BitOr for FeatureSet {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitAnd for FeatureSet {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

impl std::fmt::Display for FeatureSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            return write!(f, "FeatureSet(NONE)");
        }
        let mut parts: Vec<&str> = Vec::new();
        if self.contains(Self::DEBUG_WIREFRAME) { parts.push("WIREFRAME"); }
        if self.contains(Self::DEBUG_OVERLAY) { parts.push("OVERLAY"); }
        if self.contains(Self::DEBUG_PROFILER) { parts.push("PROFILER"); }
        let remaining = self.0 & !0b111;
        if remaining != 0 {
            parts.push(&format!("0x{:x}", remaining));
        }
        write!(f, "FeatureSet({})", parts.join("|"))
    }
}

/// Returns `true` if the pass should be executed for the given `features`.
///
/// A pass with `feature_flags == 0` is **always** live (production pass).
/// A pass with non-zero `feature_flags` is live only when **all** of its
/// required bits are present in the frame-level `features` set.
#[inline]
pub fn is_pass_live(pass: &IrPass, features: FeatureSet) -> bool {
    if pass.feature_flags == 0 {
        return true;
    }
    features.contains(FeatureSet(pass.feature_flags))
}

"""


def patch_mod_rs():
    with open(MOD_RS, "r", encoding="utf-8") as f:
        content = f.read()

    # -----------------------------------------------------------
    # 1. Insert FeatureSet definition before the IrPass section
    # -----------------------------------------------------------
    irpass_section_marker = "// ---------------------------------------------------------------------------\n// IrPass\n// ---------------------------------------------------------------------------"
    if "pub struct FeatureSet" not in content:
        content = content.replace(
            irpass_section_marker,
            irpass_section_marker,
            1,
        )
        # Actually insert AFTER the marker by splitting on the next blank-line boundary
        parts = content.split(irpass_section_marker, 1)
        if len(parts) == 2:
            # Insert FeatureSet between the marker and the IrPass doc comment
            rest = parts[1]
            # Find where the IrPass docs start and insert before them
            # The marker line is followed by blank lines and then the IrPass doc comment
            irpass_doc = "\n/// A single pass in the frame graph intermediate representation."
            if irpass_doc in rest:
                before_doc, after_doc = rest.split(irpass_doc, 1)
                rest = before_doc + FEATURESET_DEF + "\n" + irpass_doc + after_doc
            else:
                # Fallback: insert after the section header block
                rest = FEATURESET_DEF + "\n" + rest
            content = parts[0] + irpass_section_marker + rest
        print("  [OK] Inserted FeatureSet + is_pass_live")
    else:
        print("  [SKIP] FeatureSet already present")

    # -----------------------------------------------------------
    # 2. Add feature_flags: u64 to IrPass struct
    # -----------------------------------------------------------
    old_irpass = """    /// Additional labels / categories for filtering and debugging.
    ///
    /// Examples: `"transparent"`, `"post-process"`, `"debug"`.
    pub tags: Vec<String>,
}"""

    new_irpass = """    /// Additional labels / categories for filtering and debugging.
    ///
    /// Examples: `"transparent"`, `"post-process"`, `"debug"`.
    pub tags: Vec<String>,
    /// Feature-flag mask for dynamic culling (T-FG-6.4).
    ///
    /// A value of `0` (the default) means the pass is **always** executed
    /// regardless of the frame-level feature set.  Non-zero values specify
    /// the bits that must be set in the runtime [`FeatureSet`] for this pass
    /// to be considered live.
    ///
    /// # Example
    ///
    /// ```ignore
    /// pass.feature_flags = FeatureSet::DEBUG_WIREFRAME.0;
    /// ```
    pub feature_flags: u64,
}"""

    if "pub feature_flags: u64," not in content:
        count = content.count(old_irpass)
        if count != 1:
            # Try to find unique context
            old = "    pub tags: Vec<String>,\n}"
            new = "    pub tags: Vec<String>,\n    pub feature_flags: u64,\n}"
            c = content.count(old)
            if c == 1:
                content = content.replace(old, new)
                print("  [OK] Added feature_flags to IrPass struct (fallback match)")
            else:
                print(f"  [WARN] IrPass struct tail appears {count} times, manual check needed")
        else:
            content = content.replace(old_irpass, new_irpass, 1)
            print("  [OK] Added feature_flags to IrPass struct")
    else:
        print("  [SKIP] feature_flags already in IrPass struct")

    # -----------------------------------------------------------
    # 3-6. Add feature_flags: 0 to IrPass constructors
    # -----------------------------------------------------------
    # We add after `tags: Vec::new(),` in constructors — that pattern
    # appears in IrPass::graphics, compute, copy, ray_tracing.
    # But also in GraphBuilder!  We need to handle those separately.
    #
    # Approach: replace `tags: Vec::new(),` with `tags: Vec::new(), feature_flags: 0,`
    # ONLY when inside IrPass constructors (not GraphBuilder).
    # We can distinguish by proximity — GraphBuilder's `tags: Vec::new(),` lines
    # are followed by `pass.sync_access_set` or `self.passes.push`.
    # IrPass constructors' `tags: Vec::new(),` are followed by `};` then nothing extra.
    #
    # Actually let's be precise. Let's just replace all instances in the
    # IrPass impl block (graphics/compute/copy/ray_tracing) manually.

    irpass_impl_replacements = [
        # IrPass::graphics
        ("            tags: Vec::new(),\n        };\n        pass.sync_access_set_from_attachments();\n        pass\n    }",
         "            tags: Vec::new(),\n            feature_flags: 0,\n        };\n        pass.sync_access_set_from_attachments();\n        pass\n    }"),
        # IrPass::compute
        ("            tags: Vec::new(),\n        }\n    }\n\n    /// Creates a new copy pass",
         "            tags: Vec::new(),\n            feature_flags: 0,\n        }\n    }\n\n    /// Creates a new copy pass"),
        # IrPass::copy
        ("            tags: Vec::new(),\n        }\n    }\n\n    /// Creates a new ray-tracing pass",
         "            tags: Vec::new(),\n            feature_flags: 0,\n        }\n    }\n\n    /// Creates a new ray-tracing pass"),
        # IrPass::ray_tracing
        ("            tags: Vec::new(),\n        }\n    }\n}\n\nimpl IrPass",
         "            tags: Vec::new(),\n            feature_flags: 0,\n        }\n    }\n}\n\nimpl IrPass"),
    ]

    for old, new in irpass_impl_replacements:
        if old in content:
            content = content.replace(old, new, 1)
            print("  [OK] Added feature_flags: 0 to IrPass constructor")
        else:
            # Might already be done
            pass

    # -----------------------------------------------------------
    # 7-9. Add feature_flags: 0 to GraphBuilder direct constructions
    # -----------------------------------------------------------
    gb_replacements = [
        # add_graphics_pass
        ("            tags: Vec::new(),\n        };\n\n        pass.sync_access_set_from_attachments();\n        self.passes.push(pass);\n        index\n    }",
         "            tags: Vec::new(),\n            feature_flags: 0,\n        };\n\n        pass.sync_access_set_from_attachments();\n        self.passes.push(pass);\n        index\n    }"),
        # add_compute_pass
        ("            tags: Vec::new(),\n        };\n\n        self.passes.push(pass);\n        index\n    }\n\n    /// Adds a copy pass",
         "            tags: Vec::new(),\n            feature_flags: 0,\n        };\n\n        self.passes.push(pass);\n        index\n    }\n\n    /// Adds a copy pass"),
        # add_copy_pass
        ("            tags: Vec::new(),\n        };\n\n        self.passes.push(pass);\n        index\n    }\n\n    /// Consumes the builder",
         "            tags: Vec::new(),\n            feature_flags: 0,\n        };\n\n        self.passes.push(pass);\n        index\n    }\n\n    /// Consumes the builder"),
    ]

    for old, new in gb_replacements:
        if old in content:
            content = content.replace(old, new, 1)
            print("  [OK] Added feature_flags: 0 to GraphBuilder method")
        else:
            pass

    # -----------------------------------------------------------
    # 10. Add runtime_features: FeatureSet to CompiledFrameGraph
    # -----------------------------------------------------------
    old_cfg = """    /// Interference graph for resource aliasing (Phase 3).
    ///
    /// Two resources interfere if their lifetimes overlap or their GPU formats
    /// are incompatible. The graph is an undirected adjacency list used by the
    /// resource allocator to determine which resources can share physical memory.
    pub interference_graph: InterferenceGraph,
}"""

    new_cfg = """    /// Interference graph for resource aliasing (Phase 3).
    ///
    /// Two resources interfere if their lifetimes overlap or their GPU formats
    /// are incompatible. The graph is an undirected adjacency list used by the
    /// resource allocator to determine which resources can share physical memory.
    pub interference_graph: InterferenceGraph,
    /// Frame-level feature flags for dynamic pass culling (T-FG-6.4).
    ///
    /// Set this before execution to control which debug / optional passes
    /// are live.  Passes whose [`IrPass::feature_flags`] bits are not
    /// present in this set are skipped during execution.
    ///
    /// Defaults to [`FeatureSet::NONE`], which disables all debug passes.
    pub runtime_features: FeatureSet,
}"""

    if "pub runtime_features: FeatureSet," not in content:
        if old_cfg in content:
            content = content.replace(old_cfg, new_cfg, 1)
            print("  [OK] Added runtime_features to CompiledFrameGraph")
        else:
            print("  [WARN] CompiledFrameGraph tail not found")
    else:
        print("  [SKIP] runtime_features already in CompiledFrameGraph")

    # -----------------------------------------------------------
    # 11. Initialize runtime_features in compile()
    # -----------------------------------------------------------
    old_compile_end = """        Ok(CompiledFrameGraph {
            passes,
            resources,
            edges,
            order,
            depths,
            barriers,
            async_passes,
            eliminated_passes: eliminated,
            cull_stats,
            parallel_regions,
            interference_graph,
        })"""

    new_compile_end = """        Ok(CompiledFrameGraph {
            passes,
            resources,
            edges,
            order,
            depths,
            barriers,
            async_passes,
            eliminated_passes: eliminated,
            cull_stats,
            parallel_regions,
            interference_graph,
            runtime_features: FeatureSet::NONE,
        })"""

    if "runtime_features: FeatureSet::NONE," not in content:
        if old_compile_end in content:
            content = content.replace(old_compile_end, new_compile_end, 1)
            print("  [OK] Initialized runtime_features in compile()")
        else:
            print("  [WARN] compile() Ok() block not matched")
    else:
        print("  [SKIP] runtime_features already initialized in compile()")

    # -----------------------------------------------------------
    # 12. Add dynamic_cull_pass method to CompiledFrameGraph
    # -----------------------------------------------------------
    validate_method = """    /// Validates the compiled frame graph invariants.
    ///
    /// Checks:
    ///
    /// - All pass indices referenced in `order`, `edges`, `barriers`,
    ///   `async_passes`, and `parallel_regions` are valid.
    /// - All resource handles referenced in pass access sets and attachments
    ///   exist in `resources`.
    /// - No cycles exist in the edge list (already guaranteed by Phase 2b,
    ///   but verified here for defensive safety).
    ///
    /// Returns `Ok(())` if all checks pass, or `Err` with a list of
    /// human-readable validation errors.
    pub fn validate(compiled: &CompiledFrameGraph) -> Result<(), Vec<String>> {"""

    dynamic_cull = """    /// Returns `true` if the given pass is live under the current runtime features.
    ///
    /// Dynamic culling (T-FG-6.4) checks the pass's [`IrPass::feature_flags`]
    /// against [`CompiledFrameGraph::runtime_features`].  A pass with
    /// `feature_flags == 0` is **always** live.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut graph = CompiledFrameGraph::compile(passes, resources).unwrap();
    /// graph.runtime_features = FeatureSet::DEBUG_WIREFRAME;
    ///
    /// for pass_idx in &graph.order {
    ///     if graph.dynamic_cull_pass(*pass_idx) {
    ///         // submit pass to GPU ...
    ///     }
    /// }
    /// ```
    pub fn dynamic_cull_pass(&self, pass: IrPass) -> bool {
        is_pass_live(&pass, self.runtime_features)
    }

"""

    if "pub fn dynamic_cull_pass" not in content:
        # Insert before validate() — but compile uses different indentation
        # Instead, just insert after compile function closes.
        # Find the end of compile() and insert before validate().
        validate_marker = "    /// Validates the compiled frame graph invariants."
        if validate_method in content:
            content = content.replace(
                validate_method,
                dynamic_cull + validate_method,
                1,
            )
            print("  [OK] Added dynamic_cull_pass method")
        else:
            print("  [WARN] validate() method not found for dynamic_cull_pass insertion")
    else:
        print("  [SKIP] dynamic_cull_pass already present")

    # -----------------------------------------------------------
    # 13-14. Add feature_flags: 0 to test constructions
    # -----------------------------------------------------------
    # All test constructions use `tags: vec![],` followed by `}` or similar.
    # We need to add `feature_flags: 0,` after each `tags: vec![],`.
    # But we must be careful not to match non-IrPass constructions.
    # All IrPass test constructions end with `tags: vec![],` as the last field.

    # Strategy: For `tags: vec![],` in test context, add feature_flags: 0, after.
    # We can do this by replacing all `tags: vec![],` with `tags: vec![], feature_flags: 0,`
    # But only those inside IrPass struct constructions.
    #
    # The pattern is always the last field before `}`:
    #   tags: vec![],
    # }
    # So we can replace `tags: vec![],\n        }` with `tags: vec![],\n            feature_flags: 0,\n        }`

    # Count all test constructions
    test_count = content.count("tags: vec![],")
    test_updated = 0

    # Replace pattern: tags: vec![], followed by closing brace on next line
    import re

    # Pattern: `tags: vec![],` then whitespace then `}`
    # We need to handle variable indentation
    def add_feature_flags_to_test(match):
        nonlocal test_updated
        test_updated += 1
        indent = match.group(1)  # capture indent before tags
        return f'{indent}tags: vec![],\n{indent}feature_flags: 0,'

    # Match tags: vec![], followed by optional whitespace and newline, then same-indent }
    # But we need to be careful not to match the non-test ones.
    # In tests, tags: vec![], is always followed by \n        }
    # In non-test code (add_graphics_pass etc.), tags use Vec::new()

    content = re.sub(
        r'(            )tags: vec!\[\],\n(\s+)}',
        lambda m: m.group(0).replace("tags: vec![],\n" + m.group(2) + "}",
                                      "tags: vec![],\n" + m.group(2) + "feature_flags: 0,\n" + m.group(2) + "}"),
        content,
    )

    if test_updated > 0:
        print(f"  [OK] Added feature_flags: 0 to {test_updated} test constructions")
    else:
        # Try more generic approach
        content = content.replace(
            "tags: vec![],\n        }",
            "tags: vec![],\n            feature_flags: 0,\n        }"
        )
        print("  [OK] Added feature_flags: 0 to test constructions (simple replace)")

    # -----------------------------------------------------------
    # Write back mod.rs
    # -----------------------------------------------------------
    with open(MOD_RS, "w", encoding="utf-8") as f:
        f.write(content)
    print("[DONE] mod.rs updated")


def patch_py_rs():
    with open(PY_RS, "r", encoding="utf-8") as f:
        content = f.read()

    # The 3 python.rs IrPass constructions use `tags: Vec::new(),`
    # followed by `}` (closing brace of the struct literal)
    # In this file, `tags: Vec::new(),` appears exactly 3 times.

    count = content.count("tags: Vec::new(),")
    if count == 3:
        content = content.replace(
            "tags: Vec::new(),\n                })",
            "tags: Vec::new(),\n                    feature_flags: 0,\n                })",
        )
        print("  [OK] Added feature_flags: 0 to 3 python.rs constructions")
    else:
        print(f"  [WARN] Expected 3 occurrences of tags: Vec::new() in python.rs, found {count}")

    with open(PY_RS, "w", encoding="utf-8") as f:
        f.write(content)
    print("[DONE] python.rs updated")


if __name__ == "__main__":
    print("=== T-FG-6.4 Dynamic Culling ===")
    patch_mod_rs()
    print()
    patch_py_rs()
    print()
    print("All patches applied.")
