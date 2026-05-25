"""Atomically patch mod.rs with ResourceLifetime::History and HistorySlotManager + 12 tests."""
import os, tempfile

path = '/home/user/dev/USER/PROJECTS_VOID/TRINITY/crates/renderer-backend/src/frame_graph/mod.rs'
with open(path, 'r') as f:
    lines = [line.rstrip('\n') for line in f.readlines()]

# =====================================================================
# 1.  Insert History(u32) variant after the Imported line in ResourceLifetime
#     Lines around 883: "    Imported," then "}"
# =====================================================================
for idx in range(len(lines)):
    if lines[idx].strip() == 'Imported,' and '///' not in lines[idx]:
        indent = lines[idx][:len(lines[idx]) - len(lines[idx].lstrip())]
        lines[idx+1:idx+1] = [
            f'{indent}/// Resource uses a ring buffer with `n` history slots. Frame N uses',
            f'{indent}/// slot `N % n`, allowing the GPU to read `n-1` frames back without',
            f'{indent}/// hazard.',
            f'{indent}History(u32),',
        ]
        print(f"Inserted History(u32) after line {idx}")
        break

# =====================================================================
# 2.  Insert History case after "Imported =>" in Display impl
# =====================================================================
for idx in range(len(lines)):
    if 'Self::Imported => write!(f, "Imported"),' in lines[idx]:
        indent = lines[idx][:len(lines[idx]) - len(lines[idx].lstrip())]
        lines.insert(idx+1, f'{indent}Self::History(n) => write!(f, "History({{}})", n),')
        print(f"Inserted History case after line {idx}")
        break

# =====================================================================
# 3.  Insert HistorySlotManager struct before the test section
#     Find "// Tests" comment and insert before it
# =====================================================================
hsm_code = [
    '',
    '// ---------------------------------------------------------------------------',
    '// HistorySlotManager',
    '// ---------------------------------------------------------------------------',
    '',
    '/// Manages the mapping from resource handles to ring-buffer slot indices for',
    '/// history resources ([`ResourceLifetime::History`]).',
    '///',
    '/// For each history resource the manager records the number of available slots',
    '/// (`n`).  At frame `f` the caller resolves the active slot with',
    '/// [`slot_for`](Self::slot_for), which returns `f % n`.  Non-history resources',
    '/// always resolve to slot `0`.',
    'pub struct HistorySlotManager {',
    '    /// Maps a resource handle to its history slot count.',
    '    slots: std::collections::HashMap<ResourceHandle, u32>,',
    '}',
    '',
    'impl HistorySlotManager {',
    '    /// Creates an empty manager.',
    '    pub fn new() -> Self {',
    '        Self {',
    '            slots: std::collections::HashMap::new(),',
    '        }',
    '    }',
    '',
    '    /// Builds a manager from an iterator of resources, recording every',
    '    /// resource whose lifetime is [`ResourceLifetime::History(n)`] and',
    '    /// ignoring all others.',
    "    pub fn from_resources<'a>(resources: impl IntoIterator<Item = &'a IrResource>) -> Self {",
    '        let mut slots = std::collections::HashMap::new();',
    '        for r in resources {',
    '            if let ResourceLifetime::History(n) = r.lifetime {',
    '                slots.insert(r.handle, n);',
    '            }',
    '        }',
    '        Self { slots }',
    '    }',
    '',
    '    /// Returns the ring-buffer slot index for `handle` at `frame_number`.',
    '    ///',
    '    /// For history resources this is `frame_number % n` (wrapping arithmetic).',
    '    /// For non-history resources (or unknown handles) this always returns `0`.',
    '    pub fn slot_for(&self, handle: ResourceHandle, frame_number: u64) -> usize {',
    '        match self.slots.get(&handle) {',
    '            Some(&n) if n > 0 => (frame_number % n as u64) as usize,',
    '            _ => 0,',
    '        }',
    '    }',
    '',
    '    /// Returns the history length for `handle` if it is a history resource,',
    '    /// or [`None`] otherwise.',
    '    pub fn history_length(&self, handle: ResourceHandle) -> Option<u32> {',
    '        self.slots.get(&handle).copied()',
    '    }',
    '',
    '    /// Returns `true` if `handle` was registered as a history resource.',
    '    pub fn is_history(&self, handle: ResourceHandle) -> bool {',
    '        self.slots.contains_key(&handle)',
    '    }',
    '}',
    '',
    'impl Default for HistorySlotManager {',
    '    fn default() -> Self {',
    '        Self::new()',
    '    }',
    '}',
]

tests_code = [
    '',
    '    // -- HistorySlotManager --------------------------------------------------',
    '',
    '    #[test]',
    '    fn test_history_new_is_empty() {',
    '        let mgr = HistorySlotManager::new();',
    '        assert_eq!(mgr.history_length(ResourceHandle(0)), None);',
    '        assert!(!mgr.is_history(ResourceHandle(0)));',
    '        assert_eq!(mgr.slot_for(ResourceHandle(0), 0), 0);',
    '    }',
    '',
    '    #[test]',
    '    fn test_history_from_resources_one_history_n3() {',
    '        let res = IrResource::new(',
    '            ResourceHandle(1),',
    '            "history_tex",',
    '            ResourceDesc::Texture2D(TextureDesc {',
    '                width: 256,',
    '                height: 256,',
    '                mip_levels: 1,',
    '                array_layers: 1,',
    '                format: "rgba8unorm".into(),',
    '            }),',
    '            ResourceLifetime::History(3),',
    '            ResourceState::Uninitialized,',
    '        );',
    '        let mgr = HistorySlotManager::from_resources(&[res]);',
    '        assert_eq!(mgr.history_length(ResourceHandle(1)), Some(3));',
    '        assert!(mgr.is_history(ResourceHandle(1)));',
    '    }',
    '',
    '    #[test]',
    '    fn test_history_from_resources_ignores_non_history() {',
    '        let res_t = IrResource::new(',
    '            ResourceHandle(1),',
    '            "transient",',
    '            ResourceDesc::Buffer(BufferDesc { size: 256, usage: "storage".into(), is_indirect_arg: false }),',
    '            ResourceLifetime::Transient,',
    '            ResourceState::Uninitialized,',
    '        );',
    '        let res_i = IrResource::new(',
    '            ResourceHandle(2),',
    '            "imported",',
    '            ResourceDesc::Buffer(BufferDesc { size: 256, usage: "storage".into(), is_indirect_arg: false }),',
    '            ResourceLifetime::Imported,',
    '            ResourceState::Uninitialized,',
    '        );',
    '        let mgr = HistorySlotManager::from_resources(&[res_t, res_i]);',
    '        assert!(!mgr.is_history(ResourceHandle(1)));',
    '        assert!(!mgr.is_history(ResourceHandle(2)));',
    '        assert_eq!(mgr.history_length(ResourceHandle(1)), None);',
    '        assert_eq!(mgr.history_length(ResourceHandle(2)), None);',
    '    }',
    '',
    '    #[test]',
    '    fn test_history_slot_for_frame_0_returns_0_for_n3() {',
    '        let res = IrResource::new(',
    '            ResourceHandle(1),',
    '            "h",',
    '            ResourceDesc::Buffer(BufferDesc { size: 64, usage: "storage".into(), is_indirect_arg: false }),',
    '            ResourceLifetime::History(3),',
    '            ResourceState::Uninitialized,',
    '        );',
    '        let mgr = HistorySlotManager::from_resources(&[res]);',
    '        assert_eq!(mgr.slot_for(ResourceHandle(1), 0), 0);',
    '    }',
    '',
    '    #[test]',
    '    fn test_history_slot_for_frame_1_returns_1_for_n3() {',
    '        let res = IrResource::new(',
    '            ResourceHandle(1),',
    '            "h",',
    '            ResourceDesc::Buffer(BufferDesc { size: 64, usage: "storage".into(), is_indirect_arg: false }),',
    '            ResourceLifetime::History(3),',
    '            ResourceState::Uninitialized,',
    '        );',
    '        let mgr = HistorySlotManager::from_resources(&[res]);',
    '        assert_eq!(mgr.slot_for(ResourceHandle(1), 1), 1);',
    '    }',
    '',
    '    #[test]',
    '    fn test_history_slot_for_frame_3_wraps_to_0_for_n3() {',
    '        let res = IrResource::new(',
    '            ResourceHandle(1),',
    '            "h",',
    '            ResourceDesc::Buffer(BufferDesc { size: 64, usage: "storage".into(), is_indirect_arg: false }),',
    '            ResourceLifetime::History(3),',
    '            ResourceState::Uninitialized,',
    '        );',
    '        let mgr = HistorySlotManager::from_resources(&[res]);',
    '        assert_eq!(mgr.slot_for(ResourceHandle(1), 3), 0);',
    '    }',
    '',
    '    #[test]',
    '    fn test_history_slot_for_frame_5_returns_2_for_n3() {',
    '        let res = IrResource::new(',
    '            ResourceHandle(1),',
    '            "h",',
    '            ResourceDesc::Buffer(BufferDesc { size: 64, usage: "storage".into(), is_indirect_arg: false }),',
    '            ResourceLifetime::History(3),',
    '            ResourceState::Uninitialized,',
    '        );',
    '        let mgr = HistorySlotManager::from_resources(&[res]);',
    '        assert_eq!(mgr.slot_for(ResourceHandle(1), 5), 2);',
    '    }',
    '',
    '    #[test]',
    '    fn test_history_non_history_slot_for_always_0() {',
    '        let res = IrResource::new(',
    '            ResourceHandle(1),',
    '            "t",',
    '            ResourceDesc::Buffer(BufferDesc { size: 64, usage: "storage".into(), is_indirect_arg: false }),',
    '            ResourceLifetime::Transient,',
    '            ResourceState::Uninitialized,',
    '        );',
    '        let mgr = HistorySlotManager::from_resources(&[res]);',
    '        assert_eq!(mgr.slot_for(ResourceHandle(1), 0), 0);',
    '        assert_eq!(mgr.slot_for(ResourceHandle(1), 42), 0);',
    '    }',
    '',
    '    #[test]',
    '    fn test_history_length_returns_some_for_history_none_for_other() {',
    '        let res_h = IrResource::new(',
    '            ResourceHandle(1),',
    '            "h",',
    '            ResourceDesc::Buffer(BufferDesc { size: 64, usage: "storage".into(), is_indirect_arg: false }),',
    '            ResourceLifetime::History(5),',
    '            ResourceState::Uninitialized,',
    '        );',
    '        let res_t = IrResource::new(',
    '            ResourceHandle(2),',
    '            "t",',
    '            ResourceDesc::Buffer(BufferDesc { size: 64, usage: "storage".into(), is_indirect_arg: false }),',
    '            ResourceLifetime::Transient,',
    '            ResourceState::Uninitialized,',
    '        );',
    '        let mgr = HistorySlotManager::from_resources(&[res_h, res_t]);',
    '        assert_eq!(mgr.history_length(ResourceHandle(1)), Some(5));',
    '        assert_eq!(mgr.history_length(ResourceHandle(2)), None);',
    '    }',
    '',
    '    #[test]',
    '    fn test_history_is_history_true_for_history_false_for_other() {',
    '        let res_h = IrResource::new(',
    '            ResourceHandle(1),',
    '            "h",',
    '            ResourceDesc::Buffer(BufferDesc { size: 64, usage: "storage".into(), is_indirect_arg: false }),',
    '            ResourceLifetime::History(4),',
    '            ResourceState::Uninitialized,',
    '        );',
    '        let res_i = IrResource::new(',
    '            ResourceHandle(2),',
    '            "i",',
    '            ResourceDesc::Buffer(BufferDesc { size: 64, usage: "storage".into(), is_indirect_arg: false }),',
    '            ResourceLifetime::Imported,',
    '            ResourceState::Uninitialized,',
    '        );',
    '        let mgr = HistorySlotManager::from_resources(&[res_h, res_i]);',
    '        assert!(mgr.is_history(ResourceHandle(1)));',
    '        assert!(!mgr.is_history(ResourceHandle(2)));',
    '    }',
    '',
    '    #[test]',
    '    fn test_history_multiple_resources_different_lengths() {',
    '        let res_a = IrResource::new(',
    '            ResourceHandle(10),',
    '            "a",',
    '            ResourceDesc::Buffer(BufferDesc { size: 64, usage: "storage".into(), is_indirect_arg: false }),',
    '            ResourceLifetime::History(2),',
    '            ResourceState::Uninitialized,',
    '        );',
    '        let res_b = IrResource::new(',
    '            ResourceHandle(20),',
    '            "b",',
    '            ResourceDesc::Buffer(BufferDesc { size: 64, usage: "storage".into(), is_indirect_arg: false }),',
    '            ResourceLifetime::History(4),',
    '            ResourceState::Uninitialized,',
    '        );',
    '        let mgr = HistorySlotManager::from_resources(&[res_a, res_b]);',
    '        assert_eq!(mgr.history_length(ResourceHandle(10)), Some(2));',
    '        assert_eq!(mgr.history_length(ResourceHandle(20)), Some(4));',
    '        // Frame 3: a uses 3%2=1, b uses 3%4=3',
    '        assert_eq!(mgr.slot_for(ResourceHandle(10), 3), 1);',
    '        assert_eq!(mgr.slot_for(ResourceHandle(20), 3), 3);',
    '    }',
]

# =====================================================================
# 3.  Insert HistorySlotManager before the // Tests comment
# =====================================================================
tests_idx = None
for idx in range(len(lines)):
    if lines[idx].strip() == '// Tests' and idx > 3000:
        tests_idx = idx
        break
assert tests_idx is not None, "Could not find // Tests section"
# Insert HSM code before the tests section divider (before the ---- line)
# Go backwards from tests_idx to find the ---- line
for idx in range(tests_idx, tests_idx - 5, -1):
    if '// ---' in lines[idx]:
        insert_at = idx
        break
else:
    insert_at = tests_idx
# Insert the HSM code at insert_at (before the ---- divider)
lines[insert_at:insert_at] = hsm_code
print(f"Inserted HistorySlotManager before line {insert_at}")

# =====================================================================
# 4.  Update display test + add 12 tests
# =====================================================================
# Find the display test and add History(2) assertion
display_assert_idx = None
for idx in range(len(lines)):
    if lines[idx].strip() == 'assert_eq!(format!("{}", ResourceLifetime::Imported), "Imported");':
        display_assert_idx = idx
        break
assert display_assert_idx is not None, "Could not find display test assertion"
# Add the History(2) assertion after it
indent = lines[display_assert_idx][:len(lines[display_assert_idx]) - len(lines[display_assert_idx].lstrip())]
lines.insert(display_assert_idx + 1, f'{indent}assert_eq!(format!("{{}}", ResourceLifetime::History(2)), "History(2)");')
print(f"Added History(2) assertion after line {display_assert_idx}")

# Now insert tests after the closing brace of test_resource_lifetime_display
# Find the closing brace after our assertion
for idx in range(display_assert_idx, len(lines)):
    # Look for the closing } of this test (not the next test's attributes)
    if lines[idx].strip() == '}':
        # Check if next non-empty line has #[test] or a section comment
        next_lines = [l.strip() for l in lines[idx+1:idx+8]]
        if any(l.startswith('// --') for l in next_lines) or any(l.startswith('#[test]') for l in next_lines):
            # Insert tests after this closing brace
            lines[idx+1:idx+1] = tests_code
            print(f"Inserted 12 tests after line {idx}")
            break

# =====================================================================
# 5.  Write atomically via temp file
# =====================================================================
tmp = path + '.tmp'
with open(tmp, 'w') as f:
    for line in lines:
        f.write(line + '\n')
os.rename(tmp, path)
print("Written atomically via os.rename")

# Verify
with open(path, 'r') as f:
    final = f.read()
assert 'History(u32)' in final, "MISSING: History(u32)"
assert 'HistorySlotManager' in final, "MISSING: HistorySlotManager"
assert 'History(2)' in final, "MISSING: History(2)"
assert 'test_history_new_is_empty' in final, "MISSING: test_history_new_is_empty"
assert 'test_history_multiple_resources_different_lengths' in final, "MISSING: last test"
print("ALL VERIFIED")
