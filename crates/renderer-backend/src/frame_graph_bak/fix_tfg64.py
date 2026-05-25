#!/usr/bin/env python3
"""Fix remaining T-FG-6.4 issues: compile() Ok block, dynamic_cull_pass, test indent."""

import re

MOD_RS = "/home/user/dev/USER/PROJECTS_VOID/TRINITY/crates/renderer-backend/src/frame_graph/mod.rs"

with open(MOD_RS, "r", encoding="utf-8") as f:
    content = f.read()

# -----------------------------------------------------------
# 1. Fix test constructions: wrong indentation for feature_flags: 0,
#    Some have 8 spaces instead of 12 (matching the tags field indent).
# -----------------------------------------------------------
# The problem: some test constructions have:
#             tags: vec![],
#         feature_flags: 0,
#         }];
# But should be:
#             tags: vec![],
#             feature_flags: 0,
#         }];

# Count fixable patterns
wrong_indent = content.count("        feature_flags: 0,")
if wrong_indent > 0:
    # Replace all wrong-indent occurrences with proper 12-space indent
    content = content.replace(
        "        feature_flags: 0,\n        }",
        "            feature_flags: 0,\n        }"
    )
    # Also check if there's `];` variant
    content = content.replace(
        "        feature_flags: 0,\n        }];",
        "            feature_flags: 0,\n        }];"
    )
    content = content.replace(
        "        feature_flags: 0,\n        }",
        "            feature_flags: 0,\n        }"
    )
    # Safety: prevent double indentation
    content = content.replace(
        "            feature_flags: 0,feature_flags: 0,",
        "            feature_flags: 0,"
    )
    print(f"  [OK] Fixed {wrong_indent} wrong-indented feature_flags: 0 (to 12 spaces)")
else:
    # Check if they're already correct
    correct = content.count("            feature_flags: 0,")
    print(f"  [INFO] {correct} already correct indentation, {wrong_indent} wrong")

# -----------------------------------------------------------
# 2. Add runtime_features: FeatureSet::NONE to compile() Ok() block
# -----------------------------------------------------------
old_compile_ok = """        Ok(CompiledFrameGraph {
            passes,
            resources,
            edges,
            order: order.to_vec(),
            depths,
            barriers,
            async_passes,
            eliminated_passes: eliminated,
            cull_stats,
            parallel_regions,
        })"""

new_compile_ok = """        Ok(CompiledFrameGraph {
            passes,
            resources,
            edges,
            order: order.to_vec(),
            depths,
            barriers,
            async_passes,
            eliminated_passes: eliminated,
            cull_stats,
            parallel_regions,
            runtime_features: FeatureSet::NONE,
        })"""

if "runtime_features: FeatureSet::NONE" not in content:
    if old_compile_ok in content:
        content = content.replace(old_compile_ok, new_compile_ok, 1)
        print("  [OK] Added runtime_features to compile() Ok()")
    else:
        # Try with interference_graph (maybe the version changed)
        alt_old = """        Ok(CompiledFrameGraph {
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
        alt_new = """        Ok(CompiledFrameGraph {
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
        if alt_old in content:
            content = content.replace(alt_old, alt_new, 1)
            print("  [OK] Added runtime_features to compile() Ok() (alt pattern)")
        else:
            print("  [WARN] compile() Ok() block not matched with any pattern")
else:
    print("  [SKIP] runtime_features already in compile() Ok()")

# -----------------------------------------------------------
# 3. Add dynamic_cull_pass method before validate()
# -----------------------------------------------------------
dynamic_cull_method = """
    /// Returns `true` if the pass at the given index is live under
    /// the current runtime features (T-FG-6.4).
    ///
    /// A pass with `feature_flags == 0` is **always** live.
    /// Debug/optional passes whose required feature bits are not set
    /// in [`runtime_features`](Self::runtime_features) are culled.
    ///
    /// # Example
    ///
    /// ```ignore
    /// graph.runtime_features = FeatureSet::DEBUG_WIREFRAME;
    /// let live = graph.dynamic_cull_pass(PassIndex(3));
    /// ```
    #[inline]
    pub fn dynamic_cull_pass(&self, pass: &IrPass) -> bool {
        is_pass_live(pass, self.runtime_features)
    }

"""

validate_start = """    /// Validates the compiled frame graph and returns `Ok(())` or a vector"""

if "pub fn dynamic_cull_pass" not in content:
    if validate_start in content:
        content = content.replace(validate_start, dynamic_cull_method + validate_start, 1)
        print("  [OK] Added dynamic_cull_pass method before validate()")
    else:
        print("  [WARN] validate() not found for dynamic_cull_pass insertion")
else:
    print("  [SKIP] dynamic_cull_pass already present")

# -----------------------------------------------------------
# Write back
# -----------------------------------------------------------
with open(MOD_RS, "w", encoding="utf-8") as f:
    f.write(content)
print("[DONE] Fixes applied")
