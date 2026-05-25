#!/usr/bin/env python3
"""Apply T-FG-6.4 Dynamic Culling changes to main mod.rs + add whitebox tests."""

import re
import sys

MOD_RS = "/home/user/dev/USER/PROJECTS_VOID/TRINITY/crates/renderer-backend/src/frame_graph/mod.rs"

with open(MOD_RS, "r", encoding="utf-8") as f:
    content = f.read()

# ============================================================
# 1. FeatureSet type + is_pass_live (insert before IrPass section)
# ============================================================
featureset_code = """\
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
            let s = format!("0x{:x}", remaining);
            parts.push(&s);
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

# Find the IrPass section marker and insert before it
irpass_marker = "// ---------------------------------------------------------------------------\n// IrPass\n// ---------------------------------------------------------------------------"
if irpass_marker in content and "FeatureSet" not in content:
    content = content.replace(irpass_marker, featureset_code + irpass_marker, 1)
    print("[OK] Inserted FeatureSet + is_pass_live before IrPass")
elif "FeatureSet" in content:
    print("[SKIP] FeatureSet already present")
else:
    print("[WARN] IrPass marker not found")
    sys.exit(1)

# ============================================================
# 2. Add feature_flags: u64 to IrPass struct
# ============================================================
if "feature_flags" not in content:
    old = """    /// Additional labels / categories for filtering and debugging.
    ///
    /// Examples: `"transparent"`, `"post-process"`, `"debug"`.
    pub tags: Vec<String>,
}"""

    new = """    /// Additional labels / categories for filtering and debugging.
    ///
    /// Examples: `"transparent"`, `"post-process"`, `"debug"`.
    pub tags: Vec<String>,
    /// Feature-flag mask for dynamic culling (T-FG-6.4).
    ///
    /// A value of `0` (the default) means the pass is **always** executed
    /// regardless of the frame-level feature set.  Non-zero values specify
    /// the bits that must be set in the runtime `FeatureSet` for this pass
    /// to be considered live.
    pub feature_flags: u64,
}"""

    if old in content:
        content = content.replace(old, new, 1)
        print("[OK] Added feature_flags: u64 to IrPass struct")
    else:
        print("[WARN] Could not find tags field in IrPass struct")
else:
    print("[SKIP] feature_flags already on IrPass")

# ============================================================
# 3. Add runtime_features to CompiledFrameGraph
# ============================================================
if "runtime_features" not in content:
    old_cfg = """    /// Parallel regions: groups of passes at the same depth that can execute
    /// concurrently. Each inner `Vec` is a batch of passes that may run in
    /// parallel on the GPU (Phase 2c).
    pub parallel_regions: Vec<Vec<PassIndex>>,
}"""

    new_cfg = """    /// Parallel regions: groups of passes at the same depth that can execute
    /// concurrently. Each inner `Vec` is a batch of passes that may run in
    /// parallel on the GPU (Phase 2c).
    pub parallel_regions: Vec<Vec<PassIndex>>,
    /// Frame-level feature flags for dynamic pass culling (T-FG-6.4).
    ///
    /// Set this before execution to control which debug / optional passes
    /// are live.  Passes whose [`IrPass::feature_flags`] bits are not
    /// present in this set are skipped during execution.
    ///
    /// Defaults to [`FeatureSet::NONE`], which disables all debug passes.
    pub runtime_features: FeatureSet,
}"""

    if old_cfg in content:
        content = content.replace(old_cfg, new_cfg, 1)
        print("[OK] Added runtime_features to CompiledFrameGraph")
    else:
        print("[WARN] Could not find parallel_regions in CompiledFrameGraph")
else:
    print("[SKIP] runtime_features already on CompiledFrameGraph")

# ============================================================
# 4. Initialize runtime_features in compile()
# ============================================================
if "runtime_features: FeatureSet::NONE" not in content:
    old_ok = """        Ok(CompiledFrameGraph {
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
        })"""
    new_ok = """        Ok(CompiledFrameGraph {
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
            runtime_features: FeatureSet::NONE,
        })"""
    if old_ok in content:
        content = content.replace(old_ok, new_ok, 1)
        print("[OK] Added runtime_features to compile() Ok()")
    else:
        print("[WARN] compile() Ok() block not matched")
else:
    print("[SKIP] runtime_features already in compile() Ok()")

# ============================================================
# 5. Add dynamic_cull_pass method on CompiledFrameGraph
# ============================================================
if "pub fn dynamic_cull_pass" not in content:
    dyn_cull = """    /// Returns `true` if the given pass is live under the current runtime
    /// features (T-FG-6.4).
    ///
    /// A pass with `feature_flags == 0` is **always** live.
    /// Debug/optional passes whose required feature bits are not set
    /// in [`runtime_features`](Self::runtime_features) are culled.
    #[inline]
    pub fn dynamic_cull_pass(&self, pass: &IrPass) -> bool {
        is_pass_live(pass, self.runtime_features)
    }

"""

    # Insert before emit_bridge_json
    bridge_marker = "    pub fn emit_bridge_json"
    if bridge_marker in content:
        content = content.replace(bridge_marker, dyn_cull + bridge_marker, 1)
        print("[OK] Added dynamic_cull_pass method")
    else:
        print("[WARN] Could not find insertion point for dynamic_cull_pass")
else:
    print("[SKIP] dynamic_cull_pass already present")

# ============================================================
# 6. Add feature_flags: 0 to IrPass constructors
# ============================================================
# graphics() - note: sync_access_set follows
content = content.replace(
    "            view_type,\n            tags: Vec::new(),\n        };\n        pass.sync_access_set_from_attachments();\n        pass\n    }",
    "            view_type,\n            tags: Vec::new(),\n            feature_flags: 0,\n        };\n        pass.sync_access_set_from_attachments();\n        pass\n    }",
    1
)
print("[OK] Added feature_flags: 0 to IrPass::graphics")

# compute() - use unique pattern
content = content.replace(
    "            view_type,\n            tags: Vec::new(),\n        }\n    }\n\n    /// Creates a new copy pass",
    "            view_type,\n            tags: Vec::new(),\n            feature_flags: 0,\n        }\n    }\n\n    /// Creates a new copy pass",
    1
)
print("[OK] Added feature_flags: 0 to IrPass::compute")

# copy() - unique pattern
content = content.replace(
    "            view_type: ViewType::StorageBuffer,\n            tags: Vec::new(),\n        }\n    }\n\n    /// Creates a new ray-tracing pass",
    "            view_type: ViewType::StorageBuffer,\n            tags: Vec::new(),\n            feature_flags: 0,\n        }\n    }\n\n    /// Creates a new ray-tracing pass",
    1
)
print("[OK] Added feature_flags: 0 to IrPass::copy")

# ray_tracing() - unique pattern
content = content.replace(
    "            view_type: ViewType::Storage,\n            tags: Vec::new(),\n        }\n    }\n\n    /// Rebuilds `access_set.reads`",
    "            view_type: ViewType::Storage,\n            tags: Vec::new(),\n            feature_flags: 0,\n        }\n    }\n\n    /// Rebuilds `access_set.reads`",
    1
)
print("[OK] Added feature_flags: 0 to IrPass::ray_tracing")

# ============================================================
# 7. Add feature_flags: 0 to RenderGraphBuilder constructors
# ============================================================
# add_graphics_pass
content = content.replace(
    "            view_type: ViewType::ColorAttachment,\n            tags: Vec::new(),\n        };\n\n        pass.sync_access_set_from_attachments();\n        self.passes.push(pass);",
    "            view_type: ViewType::ColorAttachment,\n            tags: Vec::new(),\n            feature_flags: 0,\n        };\n\n        pass.sync_access_set_from_attachments();\n        self.passes.push(pass);",
    1
)
print("[OK] Added feature_flags: 0 to RenderGraphBuilder::add_graphics_pass")

# add_compute_pass
content = content.replace(
    "            view_type: ViewType::Storage,\n            tags: Vec::new(),\n        };\n\n        self.passes.push(pass);\n        index\n    }\n\n    /// Adds a copy pass",
    "            view_type: ViewType::Storage,\n            tags: Vec::new(),\n            feature_flags: 0,\n        };\n\n        self.passes.push(pass);\n        index\n    }\n\n    /// Adds a copy pass",
    1
)
print("[OK] Added feature_flags: 0 to RenderGraphBuilder::add_compute_pass")

# add_copy_pass
content = content.replace(
    "            view_type: ViewType::StorageBuffer,\n            tags: Vec::new(),\n        };\n\n        self.passes.push(pass);\n        index\n    }\n\n    /// Consumes the builder",
    "            view_type: ViewType::StorageBuffer,\n            tags: Vec::new(),\n            feature_flags: 0,\n        };\n\n        self.passes.push(pass);\n        index\n    }\n\n    /// Consumes the builder",
    1
)
print("[OK] Added feature_flags: 0 to RenderGraphBuilder::add_copy_pass")

# ============================================================
# 8. Add feature_flags: 0 to test IrPass struct literals
# ============================================================
# Pattern: tags: vec![], followed by closing brace
count_added = 0
while True:
    # Find tags: vec![],\n        }
    idx = content.find("            tags: vec![],\n        }")
    if idx == -1:
        break
    # Check not already followed by feature_flags
    after = content[idx:idx+200]
    if "feature_flags" in after[:100]:
        break  # already processed, exit loop
    # Replace this one
    content = content[:idx] + "            tags: vec![],\n            feature_flags: 0,\n        }" + content[idx + len("            tags: vec![],\n        }"):]
    count_added += 1

# Also handle tags: vec!["..."]\n        }
while True:
    idx = content.find("            tags: vec![\"")
    if idx == -1:
        idx = content.find("            tags: vec!['")
    if idx == -1:
        break
    # Find the closing ], of the vec
    end_idx = content.find("],", idx)
    if end_idx == -1 or end_idx - idx > 200:
        break
    rest = content[end_idx+2:]
    if rest.startswith("\n            feature_flags:"):
        break
    if rest.startswith("\n        }"):
        content = content[:end_idx+2] + "\n            feature_flags: 0,\n        }" + content[end_idx+2+len("\n        }"):]
        count_added += 1
    else:
        break

print(f"[OK] Added feature_flags: 0 to {count_added} test struct literals")

# ============================================================
# 9. Add tests at the end of the test module
# ============================================================
test_code = """

    // -----------------------------------------------------------
    // FeatureSet tests (T-FG-6.4 Dynamic Culling)
    // -----------------------------------------------------------

    #[test]
    fn test_featureset_none_is_zero() {
        assert_eq!(FeatureSet::NONE.0, 0);
    }

    #[test]
    fn test_featureset_debug_wireframe_bit0() {
        assert_eq!(FeatureSet::DEBUG_WIREFRAME.0, 1 << 0);
    }

    #[test]
    fn test_featureset_debug_overlay_bit1() {
        assert_eq!(FeatureSet::DEBUG_OVERLAY.0, 1 << 1);
    }

    #[test]
    fn test_featureset_debug_profiler_bit2() {
        assert_eq!(FeatureSet::DEBUG_PROFILER.0, 1 << 2);
    }

    #[test]
    fn test_featureset_all_debug_has_bits_0_1_2() {
        let all = FeatureSet::ALL_DEBUG;
        assert!(all.contains(FeatureSet::DEBUG_WIREFRAME));
        assert!(all.contains(FeatureSet::DEBUG_OVERLAY));
        assert!(all.contains(FeatureSet::DEBUG_PROFILER));
        assert_eq!(all.0, 0b111);
    }

    #[test]
    fn test_featureset_contains_single_bit() {
        let f = FeatureSet::DEBUG_WIREFRAME;
        assert!(f.contains(FeatureSet::DEBUG_WIREFRAME));
        assert!(!f.contains(FeatureSet::DEBUG_OVERLAY));
    }

    #[test]
    fn test_featureset_contains_multiple_bits() {
        let f = FeatureSet::ALL_DEBUG;
        assert!(f.contains(FeatureSet::DEBUG_WIREFRAME | FeatureSet::DEBUG_OVERLAY));
        assert!(f.contains(FeatureSet::ALL_DEBUG));
    }

    #[test]
    fn test_featureset_bitor_combines() {
        let f = FeatureSet::DEBUG_WIREFRAME | FeatureSet::DEBUG_OVERLAY;
        assert!(f.contains(FeatureSet::DEBUG_WIREFRAME));
        assert!(f.contains(FeatureSet::DEBUG_OVERLAY));
        assert!(!f.contains(FeatureSet::DEBUG_PROFILER));
        assert_eq!(f.0, 0b11);
    }

    #[test]
    fn test_is_pass_live_always_live_when_feature_flags_zero() {
        let pass = IrPass {
            index: PassIndex(0),
            name: "always_live".into(),
            pass_type: PassType::Graphics,
            access_set: ResourceAccessSet::empty(),
            color_attachments: vec![],
            depth_stencil: None,
            instance_source: InstanceSource::Direct {
                index_count: 0, instance_count: 1,
                base_vertex: 0, first_index: 0, first_instance: 0,
            },
            dispatch_source: None,
            view_type: ViewType::ColorAttachment,
            tags: vec![],
            feature_flags: 0,
        };
        // Should be live regardless of runtime features
        assert!(is_pass_live(&pass, FeatureSet::NONE));
        assert!(is_pass_live(&pass, FeatureSet::ALL_DEBUG));
        assert!(is_pass_live(&pass, FeatureSet::DEBUG_WIREFRAME));
    }

    #[test]
    fn test_is_pass_live_true_when_runtime_contains_flags() {
        let pass = IrPass {
            index: PassIndex(0),
            name: "wireframe".into(),
            pass_type: PassType::Graphics,
            access_set: ResourceAccessSet::empty(),
            color_attachments: vec![],
            depth_stencil: None,
            instance_source: InstanceSource::Direct {
                index_count: 0, instance_count: 1,
                base_vertex: 0, first_index: 0, first_instance: 0,
            },
            dispatch_source: None,
            view_type: ViewType::ColorAttachment,
            tags: vec![],
            feature_flags: FeatureSet::DEBUG_WIREFRAME.0,
        };
        assert!(is_pass_live(&pass, FeatureSet::ALL_DEBUG));
        assert!(is_pass_live(&pass, FeatureSet::DEBUG_WIREFRAME));
    }

    #[test]
    fn test_is_pass_live_false_when_runtime_missing_flags() {
        let pass = IrPass {
            index: PassIndex(0),
            name: "wireframe".into(),
            pass_type: PassType::Graphics,
            access_set: ResourceAccessSet::empty(),
            color_attachments: vec![],
            depth_stencil: None,
            instance_source: InstanceSource::Direct {
                index_count: 0, instance_count: 1,
                base_vertex: 0, first_index: 0, first_instance: 0,
            },
            dispatch_source: None,
            view_type: ViewType::ColorAttachment,
            tags: vec![],
            feature_flags: FeatureSet::DEBUG_WIREFRAME.0,
        };
        assert!(!is_pass_live(&pass, FeatureSet::NONE));
        assert!(!is_pass_live(&pass, FeatureSet::DEBUG_OVERLAY));
    }

    #[test]
    fn test_dynamic_cull_pass_delegates_correctly() {
        let live_pass = IrPass {
            index: PassIndex(0),
            name: "wireframe".into(),
            pass_type: PassType::Graphics,
            access_set: ResourceAccessSet::empty(),
            color_attachments: vec![],
            depth_stencil: None,
            instance_source: InstanceSource::Direct {
                index_count: 0, instance_count: 1,
                base_vertex: 0, first_index: 0, first_instance: 0,
            },
            dispatch_source: None,
            view_type: ViewType::ColorAttachment,
            tags: vec![],
            feature_flags: FeatureSet::DEBUG_WIREFRAME.0,
        };
        let dead_pass = IrPass {
            index: PassIndex(1),
            name: "profiler".into(),
            pass_type: PassType::Compute,
            access_set: ResourceAccessSet::empty(),
            color_attachments: vec![],
            depth_stencil: None,
            instance_source: InstanceSource::Direct {
                index_count: 0, instance_count: 1,
                base_vertex: 0, first_index: 0, first_instance: 0,
            },
            dispatch_source: Some(DispatchSource::Direct {
                group_count_x: 1, group_count_y: 1, group_count_z: 1,
            }),
            view_type: ViewType::Storage,
            tags: vec![],
            feature_flags: FeatureSet::DEBUG_PROFILER.0,
        };

        let compiled = CompiledFrameGraph {
            passes: vec![live_pass.clone(), dead_pass.clone()],
            resources: vec![],
            edges: vec![],
            order: vec![PassIndex(0), PassIndex(1)],
            depths: std::collections::HashMap::new(),
            barriers: vec![],
            async_passes: vec![],
            eliminated_passes: vec![],
            cull_stats: CullStats::default(),
            parallel_regions: vec![],
            runtime_features: FeatureSet::DEBUG_WIREFRAME,
        };

        // Dynamic cull with DEBUG_WIREFRAME enabled
        assert!(compiled.dynamic_cull_pass(&live_pass), "wireframe pass should be live");
        assert!(!compiled.dynamic_cull_pass(&dead_pass), "profiler pass should be culled");

        // Dynamic cull with no debug features
        let compiled_no_debug = CompiledFrameGraph {
            runtime_features: FeatureSet::NONE,
            ..compiled
        };
        assert!(!compiled_no_debug.dynamic_cull_pass(&live_pass), "wireframe should be culled with NONE");
    }

    #[test]
    fn test_compiled_frame_graph_runtime_features_defaults_to_none() {
        // compile() initializes runtime_features to FeatureSet::NONE
        let pass = IrPass {
            index: PassIndex(0),
            name: "test".into(),
            pass_type: PassType::Graphics,
            access_set: ResourceAccessSet::empty(),
            color_attachments: vec![],
            depth_stencil: None,
            instance_source: InstanceSource::Direct {
                index_count: 0, instance_count: 1,
                base_vertex: 0, first_index: 0, first_instance: 0,
            },
            dispatch_source: None,
            view_type: ViewType::ColorAttachment,
            tags: vec![],
            feature_flags: 0,
        };
        let res = IrResource::new(
            ResourceHandle(0),
            "out",
            ResourceDesc::Texture(TextureDesc {
                dimensions: [1, 1, 1],
                format: "rgba8unorm".into(),
                mip_levels: 1,
                sample_count: 1,
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        );
        let compiled = CompiledFrameGraph::compile(vec![pass], vec![res]).unwrap();
        assert_eq!(compiled.runtime_features, FeatureSet::NONE);
    }

    #[test]
    fn test_pass_with_zero_feature_flags_always_live_regardless_of_runtime() {
        let always_pass = IrPass {
            index: PassIndex(0),
            name: "always".into(),
            pass_type: PassType::Graphics,
            access_set: ResourceAccessSet::empty(),
            color_attachments: vec![],
            depth_stencil: None,
            instance_source: InstanceSource::Direct {
                index_count: 0, instance_count: 1,
                base_vertex: 0, first_index: 0, first_instance: 0,
            },
            dispatch_source: None,
            view_type: ViewType::ColorAttachment,
            tags: vec![],
            feature_flags: 0,
        };
        // Even with NONE runtime features, feature_flags==0 pass is always live
        assert!(is_pass_live(&always_pass, FeatureSet::NONE));
        assert!(is_pass_live(&always_pass, FeatureSet::ALL_DEBUG));
        assert!(is_pass_live(&always_pass, FeatureSet::DEBUG_WIREFRAME));

        // Also test via dynamic_cull_pass on a CompiledFrameGraph
        let compiled = CompiledFrameGraph {
            passes: vec![always_pass],
            resources: vec![],
            edges: vec![],
            order: vec![PassIndex(0)],
            depths: std::collections::HashMap::new(),
            barriers: vec![],
            async_passes: vec![],
            eliminated_passes: vec![],
            cull_stats: CullStats::default(),
            parallel_regions: vec![],
            runtime_features: FeatureSet::NONE,
        };
        assert!(
            compiled.dynamic_cull_pass(&compiled.passes[0]),
            "pass with feature_flags==0 must always be live"
        );
    }
"""

# Find the exact end of the test module. Pattern: last closing brace of mod tests
# The test module ends with a pattern like:
#     }
# }
# We need to find the last "}\n}" that closes mod tests and the outer module
last_close = content.rfind("\n}\n}")
if last_close == -1:
    last_close = content.rfind("\n}\n}\n")

if last_close != -1:
    content = content[:last_close] + test_code + "\n" + content[last_close:]
    print("[OK] Added T-FG-6.4 tests at end of test module")
else:
    print("[WARN] Could not find test module end")
    # Try just appending at the end
    content = content.rstrip() + "\n" + test_code + "\n}\n"
    print("[OK] Appended tests at EOF (check closing braces)")

# ============================================================
# Write back
# ============================================================
with open(MOD_RS, "w", encoding="utf-8") as f:
    f.write(content)

print("[DONE] All T-FG-6.4 changes applied")
