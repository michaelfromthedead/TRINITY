//! Deferred command buffer for batched structural ECS mutations.
//!
//! Mirrors the Python [`CommandBuffer`] at
//! `engine/core/ecs/command_buffer.py` with a Rust-native API that
//! operates directly on [`ComponentStore`].
//!
//! # Workflow
//!
//! 1. Push commands (`spawn`, `despawn`, `add_component`, `remove_component`).
//! 2. Call [`flush`](CommandBuffer::flush) to apply all mutations in order.
//! 3. Optionally call [`replay`](CommandBuffer::replay) on a fresh store for
//!    determinism verification.
//! 4. Call [`clear`](CommandBuffer::clear) to reset for the next frame.

use crate::component_store::ComponentStore;

// ── Command ──────────────────────────────────────────────────────────────

/// A deferred structural ECS command.
///
/// Each variant records the minimum data needed to apply the mutation to a
/// [`ComponentStore`] during [`CommandBuffer::flush`].
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    /// Spawn an entity with component data.
    ///
    /// Tuple is `(component_type_id, raw_bytes)` for each component.
    Spawn(u64, Vec<(u32, Vec<u8>)>),

    /// Despawn an entity.
    Despawn(u64),

    /// Add a component to an entity (archetype-migrating if the component
    /// type is new to the entity; overwriting in place if the entity already
    /// has this component type).
    AddComponent(u64, u32, Vec<u8>),

    /// Remove a component from an entity (archetype-migrating).
    ///
    /// No-op if the entity does not have this component type.  If removing
    /// the last component, the entity is despawned entirely.
    RemoveComponent(u64, u32),
}

// ── CommandBuffer ────────────────────────────────────────────────────────

/// Records commands and flushes them to a [`ComponentStore`].
///
/// Commands are applied **in insertion order** during [`flush`](Self::flush).
/// The last-flushed batch is retained so [`replay`](Self::replay) can
/// re-apply it for determinism verification.
///
/// # Frame lifecycle
///
/// ```ignore
/// let mut cb = CommandBuffer::new();
/// cb.push(Command::Spawn(42, vec![(1, pos_bytes)]));
/// cb.flush(&mut store);          // apply, snapshot for replay
/// cb.clear();                    // ready for next frame
/// ```
#[derive(Debug, Clone)]
pub struct CommandBuffer {
    commands: Vec<Command>,
    last_flushed: Vec<Command>,
}

impl CommandBuffer {
    /// Create a new, empty command buffer.
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
            last_flushed: Vec::new(),
        }
    }

    // ── Mutators ──────────────────────────────────────────────────────

    /// Append a command to the buffer.
    pub fn push(&mut self, cmd: Command) {
        self.commands.push(cmd);
    }

    /// Push a [`Command::Spawn`].
    pub fn spawn_command(&mut self, entity_id: u64, components: Vec<(u32, Vec<u8>)>) {
        self.push(Command::Spawn(entity_id, components));
    }

    /// Push a [`Command::Despawn`].
    pub fn despawn_command(&mut self, entity_id: u64) {
        self.push(Command::Despawn(entity_id));
    }

    /// Push a [`Command::AddComponent`].
    pub fn add_component_command(&mut self, entity_id: u64, component_id: u32, data: Vec<u8>) {
        self.push(Command::AddComponent(entity_id, component_id, data));
    }

    /// Push a [`Command::RemoveComponent`].
    pub fn remove_component_command(&mut self, entity_id: u64, component_id: u32) {
        self.push(Command::RemoveComponent(entity_id, component_id));
    }

    // ── Execution ─────────────────────────────────────────────────────

    /// Apply all buffered commands to `store` **in order**, then clear the
    /// working buffer.
    ///
    /// The flushed batch is captured in an internal snapshot so that
    /// [`replay`](Self::replay) can re-apply it later.
    pub fn flush(&mut self, store: &mut ComponentStore) {
        // Swap the current command vector into the snapshot, leaving
        // `self.commands` empty for the next frame.  This is infallible
        // (no allocation), so we stay panic-safe.
        let batch = std::mem::take(&mut self.commands);
        self.last_flushed = batch;

        for cmd in &self.last_flushed {
            apply_command(cmd, store);
        }
    }

    /// Re-apply the **last flushed** batch of commands to `store`.
    ///
    /// Intended for determinism verification: flush the buffer to a store,
    /// then replay the same commands onto a fresh store (or one restored to
    /// an equivalent initial state) and compare the resulting states.
    ///
    /// Has no effect if [`flush`](Self::flush) has never been called.
    pub fn replay(&self, store: &mut ComponentStore) {
        for cmd in &self.last_flushed {
            apply_command(cmd, store);
        }
    }

    // ── Buffer management ─────────────────────────────────────────────

    /// Remove all pending (not-yet-flushed) commands.
    ///
    /// The replay snapshot is **not** affected.
    pub fn clear(&mut self) {
        self.commands.clear();
    }

    /// Number of pending commands.
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Returns `true` when there are no pending commands.
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Borrow the last-flushed snapshot (for external inspection or
    /// serialisation).
    pub fn last_flushed(&self) -> &[Command] {
        &self.last_flushed
    }

    /// Borrow the pending commands (for inspection before flush).
    pub fn pending(&self) -> &[Command] {
        &self.commands
    }
}

impl Default for CommandBuffer {
    fn default() -> Self {
        Self::new()
    }
}

// ── Command dispatch ─────────────────────────────────────────────────────

/// Dispatch a single `Command` to `store`.
fn apply_command(cmd: &Command, store: &mut ComponentStore) {
    match cmd {
        Command::Spawn(entity_id, components) => {
            let ids: Vec<u32> = components.iter().map(|(id, _)| *id).collect();
            store.spawn(*entity_id, &ids, components);
        }
        Command::Despawn(entity_id) => {
            store.despawn(*entity_id);
        }
        Command::AddComponent(entity_id, component_id, data) => {
            apply_add_component(store, *entity_id, *component_id, data);
        }
        Command::RemoveComponent(entity_id, component_id) => {
            apply_remove_component(store, *entity_id, *component_id);
        }
    }
}

// ── Archetype migration helpers ──────────────────────────────────────────

/// Read **all** component data for `entity_id` from `store`.
///
/// Returns `None` when the entity does not exist.  Otherwise returns a
/// `Vec` of `(component_id, raw_bytes)` in the archetype's column order.
fn read_entity_data(store: &ComponentStore, entity_id: u64) -> Option<Vec<(u32, Vec<u8>)>> {
    let (arch_id, row) = *store.entity_index.get(&entity_id)?;
    let archetype = store.archetypes.get(&arch_id)?;

    let mut components = Vec::with_capacity(archetype.component_ids.len());
    for (col_idx, &cid) in archetype.component_ids.iter().enumerate() {
        // Determine stride from the type registry.
        let stride = store
            .registry
            .get(cid)
            .map(|info| info.size)
            .unwrap_or(0);

        let col = &archetype.columns[col_idx];
        let start = row * stride;
        let end = start + stride;

        if stride > 0 && end <= col.len() {
            components.push((cid, col[start..end].to_vec()));
        } else if stride > 0 {
            // Column not yet long enough for this row -- zero-fill.
            components.push((cid, vec![0u8; stride]));
        }
        // stride == 0 is a degenerate case (no registered type) -- skip.
    }
    Some(components)
}

/// Apply `AddComponent` -- archetype-migrate the entity or overwrite in
/// place if the component is already present.
fn apply_add_component(store: &mut ComponentStore, entity_id: u64, component_id: u32, data: &[u8]) {
    let current = match read_entity_data(store, entity_id) {
        Some(c) => c,
        None => return, // entity does not exist -- no-op
    };

    // If the entity already carries this component type, overwrite in place
    // (no archetype migration needed).
    if current.iter().any(|(id, _)| *id == component_id) {
        store.write_field(entity_id, component_id, 0, data);
        return;
    }

    // --- Archetype migration: despawn -> spawn with enlarged set ---

    let mut all_data = current;
    all_data.push((component_id, data.to_vec()));

    let ids: Vec<u32> = all_data.iter().map(|(id, _)| *id).collect();
    store.despawn(entity_id);
    store.spawn(entity_id, &ids, &all_data);
}

/// Apply `RemoveComponent` -- archetype-migrate the entity, or despawn it
/// if it would have no components remaining.
fn apply_remove_component(store: &mut ComponentStore, entity_id: u64, component_id: u32) {
    let current = match read_entity_data(store, entity_id) {
        Some(c) => c,
        None => return, // entity does not exist -- no-op
    };

    // Check whether the entity actually has this component.
    let had_it = current.iter().any(|(id, _)| *id == component_id);
    if !had_it {
        return; // no-op
    }

    // Filter out the target component.
    let filtered: Vec<(u32, Vec<u8>)> = current
        .into_iter()
        .filter(|(id, _)| *id != component_id)
        .collect();

    if filtered.is_empty() {
        // No components left -- despawn entirely.
        store.despawn(entity_id);
        return;
    }

    // --- Archetype migration: despawn -> spawn with reduced set ---
    let ids: Vec<u32> = filtered.iter().map(|(id, _)| *id).collect();
    store.despawn(entity_id);
    store.spawn(entity_id, &ids, &filtered);
}
