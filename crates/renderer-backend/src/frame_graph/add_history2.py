import os

path = '/home/user/dev/USER/PROJECTS_VOID/TRINITY/crates/renderer-backend/src/frame_graph/mod.rs'
with open(path, 'r') as f:
    lines = f.readlines()

# =====================================================================
# 1. Find and modify ResourceLifetime enum + Display impl
# =====================================================================
enum_start = None
enum_end = None
display_start = None
display_end = None

for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped == 'pub enum ResourceLifetime {':
        enum_start = i
    if enum_start is not None and stripped == '}' and enum_end is None:
        # Check if this is closing the enum (Imported) or the Display impl
        # Look backwards for Imported
        for j in range(i-1, max(0, i-5), -1):
            if 'Imported' in lines[j] and '///' not in lines[j]:
                enum_end = i
                break
        if enum_end is not None:
            continue
    if stripped == 'impl fmt::Display for ResourceLifetime {':
        display_start = i
    if display_start is not None and stripped == '}' and display_end is None:
        display_end = i
        break

print(f"Enum at lines {enum_start}-{enum_end}")
print(f"Display at lines {display_start}-{display_end}")

# Verify we found them
assert enum_start is not None, "Could not find enum start"
assert enum_end is not None, "Could not find enum end"
assert display_start is not None, "Could not find display start"
assert display_end is not None, "Could not find display end"

# Insert History(u32) before enum closing
# Need to find where Imported is within the enum
for i in range(enum_end, enum_start, -1):
    s = lines[i].strip()
    if s == 'Imported,' or s == 'Imported':
        # Insert after this line
        indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
        lines.insert(i+1, f'{indent}/// Resource uses a ring buffer with `n` history slots. Frame N uses\n')
        lines.insert(i+2, f'{indent}/// slot `N % n`, allowing the GPU to read `n-1` frames back without\n')
        lines.insert(i+3, f'{indent}/// hazard.\n')
        lines.insert(i+4, f'{indent}History(u32),\n')
        enum_end += 4
        display_start += 4
        display_end += 4
        break

# Add History case to Display impl
for i in range(display_start, display_end + 1):
    s = lines[i].strip()
    if s == 'Self::Imported => write!(f, "Imported"),':
        indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
        lines.insert(i+1, f'{indent}Self::History(n) => write!(f, "History({{}})", n),\n')
        display_end += 1
        break

# =====================================================================
# 2. Find and add HistorySlotManager before test module
# =====================================================================
tests_start = None
for i, line in enumerate(lines):
    if line.strip() == '#[cfg(test)]':
        tests_start = i
        break

assert tests_start is not None, "Could not find tests section"

# Find is_async_pass function and insert HistorySlotManager before // Tests
# Actually, let's just find and insert before the "// Tests" or "// ---------------------------------------------------------------------------\n// Tests" section
for i in range(tests_start - 10, tests_start):
    s = lines[i].strip()
    if s == '// Tests':
        hsm_start = i
        break
else:
    # Try to find is_async_pass
    for i in range(tests_start - 30, tests_start):
        if 'pub fn is_async_pass' in lines[i]:
            hsm_start = i + 3  # After the closing }
            break
    else:
        hsm_start = tests_start - 2

# Find the blank line before // Tests to insert HistorySlotManager
insert_before = tests_start
for i in range(tests_start - 1, max(0, tests_start - 10), -1):
    if lines[i].strip().startswith('// ---') and lines[i].strip().endswith('---'):
        insert_before = i
        break

hsm_code = '''// ---------------------------------------------------------------------------
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

'''
# Insert HistorySlotManager before the test section divider
for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped == '// ---------------------------------------------------------------------------':
        # Check if next non-empty line is "// Tests"
        for j in range(i+1, min(i+5, len(lines))):
            if lines[j].strip() == '// Tests':
                # Insert after this divider (before the one with Tests)
                insert_lines = hsm_code.split('\n')
                # Actually, insert before the divider
                for k, hl in enumerate(insert_lines):
                    lines.insert(i + k, hl + '\n' if hl else '\n')
                print(f"HistorySlotManager inserted before // Tests at line {i}")
                break
        break
else:
    # Fallback: insert just before #[cfg(test)]
    insert_lines = hsm_code.split('\n')
    for k, hl in enumerate(insert_lines):
        lines.insert(tests_start + k, hl + '\n' if hl else '\n')
    print(f"HistorySlotManager inserted before #[cfg(test)] at line {tests_start}")

# =====================================================================
# 3. Find the display test and update it + add 12 new tests
# =====================================================================
display_test_line = None
for i, line in enumerate(lines):
    if 'test_resource_lifetime_display' in line:
        display_test_line = i
        break

assert display_test_line is not None, "Could not find display test"

# Find the closing } of the display test
for i in range(display_test_line, display_test_line + 10):
    if lines[i].strip() == '}':
        # Check if History(2) is already in the test
        has_history = any('History(2)' in lines[j] for j in range(display_test_line, i+1))
        if not has_history:
            indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
            # Insert the History(2) assertion before the closing }
            lines.insert(i, f'{indent}assert_eq!(format!("{{}}", ResourceLifetime::History(2)), "History(2)");\n')
            # Now i is the line number of the assertion, closing brace is at i+1
            closing_brace = i + 1
        else:
            closing_brace = i
        break

# Add 12 new tests after the closing brace
tests_code = '''
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
            ResourceDesc::Texture2D(TextureDesc {
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
'''

# Insert tests after the closing brace of test_resource_lifetime_display
for k, tl in enumerate(tests_code.split('\n')):
    lines.insert(closing_brace + 1 + k, tl + '\n' if tl else '\n')

# =====================================================================
# Write
# =====================================================================
with open(path, 'w') as f:
    f.writelines(lines)

# Verify
with open(path, 'r') as f:
    final = f.read()

assert 'History(u32)' in final, "History(u32) not found"
assert 'HistorySlotManager' in final, "HistorySlotManager not found"
assert 'History(2)' in final, "History(2) in display test not found"
assert 'test_history_new_is_empty' in final, "test_history_new_is_empty not found"
assert 'test_history_multiple_resources_different_lengths' in final, "Last test not found"

print("ALL CHANGES VERIFIED SUCCESSFULLY")
