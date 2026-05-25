import os

path = '/home/user/dev/USER/PROJECTS_VOID/TRINITY/crates/renderer-backend/src/frame_graph/mod.rs'
with open(path, 'r') as f:
    content = f.read()

# =====================================================================
# 1. Add History(u32) variant to ResourceLifetime enum + update Display
# =====================================================================
old_enum = """/// Flag indicating whether a resource is transient (frame-local) or imported
/// (persistent, provided by the application).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResourceLifetime {
    /// Resource is allocated per-frame and may be aliased with other
    /// transient resources.
    Transient,
    /// Resource is imported from outside the frame graph. The compiler
    /// tracks its state but does not allocate or alias it.
    Imported,
}

impl fmt::Display for ResourceLifetime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transient => write!(f, "Transient"),
            Self::Imported => write!(f, "Imported"),
        }
    }
}"""

new_enum = """/// Flag indicating whether a resource is transient (frame-local) or imported
/// (persistent, provided by the application).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResourceLifetime {
    /// Resource is allocated per-frame and may be aliased with other
    /// transient resources.
    Transient,
    /// Resource is imported from outside the frame graph. The compiler
    /// tracks its state but does not allocate or alias it.
    Imported,
    /// Resource uses a ring buffer with `n` history slots. Frame N uses
    /// slot `N % n`, allowing the GPU to read `n-1` frames back without
    /// hazard.
    History(u32),
}

impl fmt::Display for ResourceLifetime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transient => write!(f, "Transient"),
            Self::Imported => write!(f, "Imported"),
            Self::History(n) => write!(f, "History({})", n),
        }
    }
}"""

assert old_enum in content, "Could not find ResourceLifetime enum"
content = content.replace(old_enum, new_enum, 1)
assert 'History(u32)' in content

# =====================================================================
# 2. Add HistorySlotManager before test module
# =====================================================================
old_tests_marker = """/// Returns `true` if `pass_idx` appears in `async_passes`.
///
/// Convenience helper for checking whether a pass was identified as
/// async-eligible during Phase 5 scheduling.
pub fn is_async_pass(pass_idx: PassIndex, async_passes: &[(PassIndex, String)]) -> bool {
    async_passes.iter().any(|(idx, _)| *idx == pass_idx)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------"""

new_tests_marker = """/// Returns `true` if `pass_idx` appears in `async_passes`.
///
/// Convenience helper for checking whether a pass was identified as
/// async-eligible during Phase 5 scheduling.
pub fn is_async_pass(pass_idx: PassIndex, async_passes: &[(PassIndex, String)]) -> bool {
    async_passes.iter().any(|(idx, _)| *idx == pass_idx)
}

// ---------------------------------------------------------------------------
// HistorySlotManager
// ---------------------------------------------------------------------------

/// Manages the mapping from resource handles to ring-buffer slot indices for
/// history resources ([`ResourceLifetime::History`]).
///
/// For each history resource the manager records the number of available slots
/// (`n`).  At frame `f` the caller resolves the active slot with
/// [`slot_for`](Self::slot_for), which returns `f % n`.  Non-history resources
/// always resolve to slot `0`.
pub struct HistorySlotManager {
    /// Maps a resource handle to its history slot count.
    slots: std::collections::HashMap<ResourceHandle, u32>,
}

impl HistorySlotManager {
    /// Creates an empty manager.
    pub fn new() -> Self {
        Self {
            slots: std::collections::HashMap::new(),
        }
    }

    /// Builds a manager from an iterator of resources, recording every
    /// resource whose lifetime is [`ResourceLifetime::History(n)`] and
    /// ignoring all others.
    pub fn from_resources<'a>(resources: impl IntoIterator<Item = &'a IrResource>) -> Self {
        let mut slots = std::collections::HashMap::new();
        for r in resources {
            if let ResourceLifetime::History(n) = r.lifetime {
                slots.insert(r.handle, n);
            }
        }
        Self { slots }
    }

    /// Returns the ring-buffer slot index for `handle` at `frame_number`.
    ///
    /// For history resources this is `frame_number % n` (wrapping arithmetic).
    /// For non-history resources (or unknown handles) this always returns `0`.
    pub fn slot_for(&self, handle: ResourceHandle, frame_number: u64) -> usize {
        match self.slots.get(&handle) {
            Some(&n) if n > 0 => (frame_number % n as u64) as usize,
            _ => 0,
        }
    }

    /// Returns the history length for `handle` if it is a history resource,
    /// or [`None`] otherwise.
    pub fn history_length(&self, handle: ResourceHandle) -> Option<u32> {
        self.slots.get(&handle).copied()
    }

    /// Returns `true` if `handle` was registered as a history resource.
    pub fn is_history(&self, handle: ResourceHandle) -> bool {
        self.slots.contains_key(&handle)
    }
}

impl Default for HistorySlotManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------"""

assert old_tests_marker in content, "Could not find test section marker"
content = content.replace(old_tests_marker, new_tests_marker, 1)
assert 'HistorySlotManager' in content

# =====================================================================
# 3. Update the existing display test + add 12 new tests
# =====================================================================
old_display_test = """    #[test]
    fn test_resource_lifetime_display() {
        assert_eq!(format!("{}", ResourceLifetime::Transient), "Transient");
        assert_eq!(format!("{}", ResourceLifetime::Imported), "Imported");
    }

    // -- EdgeType / IrEdge"""

new_display_and_tests = """    #[test]
    fn test_resource_lifetime_display() {
        assert_eq!(format!("{}", ResourceLifetime::Transient), "Transient");
        assert_eq!(format!("{}", ResourceLifetime::Imported), "Imported");
        assert_eq!(format!("{}", ResourceLifetime::History(2)), "History(2)");
    }

    #[test]
    fn test_history_new_is_empty() {
        let mgr = HistorySlotManager::new();
        assert_eq!(mgr.history_length(ResourceHandle(0)), None);
        assert!(!mgr.is_history(ResourceHandle(0)));
        assert_eq!(mgr.slot_for(ResourceHandle(0), 0), 0);
    }

    #[test]
    fn test_history_from_resources_one_history_n3() {
        let res = IrResource::new(
            ResourceHandle(1),
            "history_tex",
            ResourceDesc::Texture(TextureDesc {
                width: 256,
                height: 256,
                mip_levels: 1,
                array_layers: 1,
                format: "rgba8unorm".into(),
            }),
            ResourceLifetime::History(3),
            ResourceState::Uninitialized,
        );
        let mgr = HistorySlotManager::from_resources(&[res]);
        assert_eq!(mgr.history_length(ResourceHandle(1)), Some(3));
        assert!(mgr.is_history(ResourceHandle(1)));
    }

    #[test]
    fn test_history_from_resources_ignores_non_history() {
        let res_t = IrResource::new(
            ResourceHandle(1),
            "transient",
            ResourceDesc::Buffer(BufferDesc { size: 256, usage: "storage".into(), is_indirect_arg: false }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        );
        let res_i = IrResource::new(
            ResourceHandle(2),
            "imported",
            ResourceDesc::Buffer(BufferDesc { size: 256, usage: "storage".into(), is_indirect_arg: false }),
            ResourceLifetime::Imported,
            ResourceState::Uninitialized,
        );
        let mgr = HistorySlotManager::from_resources(&[res_t, res_i]);
        assert!(!mgr.is_history(ResourceHandle(1)));
        assert!(!mgr.is_history(ResourceHandle(2)));
        assert_eq!(mgr.history_length(ResourceHandle(1)), None);
        assert_eq!(mgr.history_length(ResourceHandle(2)), None);
    }

    #[test]
    fn test_history_slot_for_frame_0_returns_0_for_n3() {
        let res = IrResource::new(
            ResourceHandle(1),
            "h",
            ResourceDesc::Buffer(BufferDesc { size: 64, usage: "storage".into(), is_indirect_arg: false }),
            ResourceLifetime::History(3),
            ResourceState::Uninitialized,
        );
        let mgr = HistorySlotManager::from_resources(&[res]);
        assert_eq!(mgr.slot_for(ResourceHandle(1), 0), 0);
    }

    #[test]
    fn test_history_slot_for_frame_1_returns_1_for_n3() {
        let res = IrResource::new(
            ResourceHandle(1),
            "h",
            ResourceDesc::Buffer(BufferDesc { size: 64, usage: "storage".into(), is_indirect_arg: false }),
            ResourceLifetime::History(3),
            ResourceState::Uninitialized,
        );
        let mgr = HistorySlotManager::from_resources(&[res]);
        assert_eq!(mgr.slot_for(ResourceHandle(1), 1), 1);
    }

    #[test]
    fn test_history_slot_for_frame_3_wraps_to_0_for_n3() {
        let res = IrResource::new(
            ResourceHandle(1),
            "h",
            ResourceDesc::Buffer(BufferDesc { size: 64, usage: "storage".into(), is_indirect_arg: false }),
            ResourceLifetime::History(3),
            ResourceState::Uninitialized,
        );
        let mgr = HistorySlotManager::from_resources(&[res]);
        assert_eq!(mgr.slot_for(ResourceHandle(1), 3), 0);
    }

    #[test]
    fn test_history_slot_for_frame_5_returns_2_for_n3() {
        let res = IrResource::new(
            ResourceHandle(1),
            "h",
            ResourceDesc::Buffer(BufferDesc { size: 64, usage: "storage".into(), is_indirect_arg: false }),
            ResourceLifetime::History(3),
            ResourceState::Uninitialized,
        );
        let mgr = HistorySlotManager::from_resources(&[res]);
        assert_eq!(mgr.slot_for(ResourceHandle(1), 5), 2);
    }

    #[test]
    fn test_history_non_history_slot_for_always_0() {
        let res = IrResource::new(
            ResourceHandle(1),
            "t",
            ResourceDesc::Buffer(BufferDesc { size: 64, usage: "storage".into(), is_indirect_arg: false }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        );
        let mgr = HistorySlotManager::from_resources(&[res]);
        assert_eq!(mgr.slot_for(ResourceHandle(1), 0), 0);
        assert_eq!(mgr.slot_for(ResourceHandle(1), 42), 0);
    }

    #[test]
    fn test_history_length_returns_some_for_history_none_for_other() {
        let res_h = IrResource::new(
            ResourceHandle(1),
            "h",
            ResourceDesc::Buffer(BufferDesc { size: 64, usage: "storage".into(), is_indirect_arg: false }),
            ResourceLifetime::History(5),
            ResourceState::Uninitialized,
        );
        let res_t = IrResource::new(
            ResourceHandle(2),
            "t",
            ResourceDesc::Buffer(BufferDesc { size: 64, usage: "storage".into(), is_indirect_arg: false }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        );
        let mgr = HistorySlotManager::from_resources(&[res_h, res_t]);
        assert_eq!(mgr.history_length(ResourceHandle(1)), Some(5));
        assert_eq!(mgr.history_length(ResourceHandle(2)), None);
    }

    #[test]
    fn test_history_is_history_true_for_history_false_for_other() {
        let res_h = IrResource::new(
            ResourceHandle(1),
            "h",
            ResourceDesc::Buffer(BufferDesc { size: 64, usage: "storage".into(), is_indirect_arg: false }),
            ResourceLifetime::History(4),
            ResourceState::Uninitialized,
        );
        let res_i = IrResource::new(
            ResourceHandle(2),
            "i",
            ResourceDesc::Buffer(BufferDesc { size: 64, usage: "storage".into(), is_indirect_arg: false }),
            ResourceLifetime::Imported,
            ResourceState::Uninitialized,
        );
        let mgr = HistorySlotManager::from_resources(&[res_h, res_i]);
        assert!(mgr.is_history(ResourceHandle(1)));
        assert!(!mgr.is_history(ResourceHandle(2)));
    }

    #[test]
    fn test_history_multiple_resources_different_lengths() {
        let res_a = IrResource::new(
            ResourceHandle(10),
            "a",
            ResourceDesc::Buffer(BufferDesc { size: 64, usage: "storage".into(), is_indirect_arg: false }),
            ResourceLifetime::History(2),
            ResourceState::Uninitialized,
        );
        let res_b = IrResource::new(
            ResourceHandle(20),
            "b",
            ResourceDesc::Buffer(BufferDesc { size: 64, usage: "storage".into(), is_indirect_arg: false }),
            ResourceLifetime::History(4),
            ResourceState::Uninitialized,
        );
        let mgr = HistorySlotManager::from_resources(&[res_a, res_b]);
        assert_eq!(mgr.history_length(ResourceHandle(10)), Some(2));
        assert_eq!(mgr.history_length(ResourceHandle(20)), Some(4));
        // Frame 3: a uses 3%2=1, b uses 3%4=3
        assert_eq!(mgr.slot_for(ResourceHandle(10), 3), 1);
        assert_eq!(mgr.slot_for(ResourceHandle(20), 3), 3);
    }

    // -- EdgeType / IrEdge"""

assert old_display_test in content, "Could not find display test"
content = content.replace(old_display_test, new_display_and_tests, 1)
assert 'test_history_new_is_empty' in content
assert 'test_history_multiple_resources_different_lengths' in content

# =====================================================================
# Write
# =====================================================================
with open(path, 'w') as f:
    f.write(content)

print("All changes applied successfully")
